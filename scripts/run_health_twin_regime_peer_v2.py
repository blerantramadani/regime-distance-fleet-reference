#!/usr/bin/env python3
"""Evaluate dynamic operating-regime matched peers against local baselines.

This experiment holds the v1 normal-behaviour model fixed and changes only the
peer-reference layer.  It selects peer-count and distance-bandwidth parameters
on a chronological validation period, then evaluates once on the final test
period.  The selection objective is label-free residual stability, not fault
detection accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_baseline import (
    ALPHAS,
    AMBIENT_SOURCES,
    MONITORING_START,
    MIN_TRAINING_SCALE,
    SOURCE_COLUMNS,
    TARGETS,
    TEST_END,
    TRAIN_END,
    VALID_END,
    fit_ridge,
    mae,
    make_feature_matrix,
    predict,
    rmse,
    robust_training_keep,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PEER_COUNTS = [3, 5, 8, 12]
DISTANCE_BANDWIDTHS = [0.5, 1.0, 2.0]
MIN_PEERS_TO_SCORE = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def display(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def markdown_table(frame, columns=None):
    if columns is not None:
        frame = frame.loc[:, columns]
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def regime_matrix(data: pd.DataFrame, ambient_column: str) -> tuple[np.ndarray, list[str]]:
    power = data["active_power_mean"].to_numpy(dtype=float) / 2300.0
    wind = data["wind_speed_mean"].to_numpy(dtype=float)
    gen_rpm = data["generator_speed_mean"].to_numpy(dtype=float) / 1450.0
    main_rpm = data["main_shaft_speed_mean"].to_numpy(dtype=float)
    pitch = data[["pitch_a_mean", "pitch_b_mean", "pitch_c_mean"]].to_numpy(dtype=float)
    pitch_mean = np.mean(pitch, axis=1)
    pitch_spread = np.max(pitch, axis=1) - np.min(pitch, axis=1)
    ambient = data[ambient_column].to_numpy(dtype=float)
    return (
        np.column_stack([power, wind, gen_rpm, main_rpm, pitch_mean, pitch_spread, ambient]),
        [
            "power_ratio",
            "primary_wind_speed",
            "generator_rpm_ratio",
            "main_shaft_rpm",
            "pitch_mean",
            "pitch_spread",
            ambient_column,
        ],
    )


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2.0, side="left")])


def stable_summary(scores: pd.DataFrame) -> dict:
    value = scores["specific_residual"].to_numpy(dtype=float)
    value = value[np.isfinite(value)]
    if not len(value):
        return {
            "score_rows": 0,
            "specific_median_abs": np.nan,
            "specific_std": np.nan,
            "specific_p95_abs": np.nan,
            "mean_candidate_peers": np.nan,
            "mean_retained_peers": np.nan,
            "mean_selected_distance": np.nan,
            "pct_with_at_least_5_candidate_peers": np.nan,
        }
    return {
        "score_rows": int(len(value)),
        "specific_median_abs": float(np.median(np.abs(value))),
        "specific_std": float(np.std(value)),
        "specific_p95_abs": float(np.quantile(np.abs(value), 0.95)),
        "mean_candidate_peers": float(scores["candidate_peer_count"].mean()),
        "mean_retained_peers": float(scores["retained_peer_count"].mean()),
        "mean_selected_distance": float(scores["mean_selected_distance"].mean()),
        "pct_with_at_least_5_candidate_peers": float((scores["candidate_peer_count"] >= 5).mean() * 100),
    }


def peer_reference_scores(
    frame: pd.DataFrame,
    target: str,
    method: str,
    residual_scales: dict[str, float],
    peer_count: int | None = None,
    bandwidth: float | None = None,
    stabilize_kernel: bool = True,
) -> pd.DataFrame:
    """Return leave-one-out scores without inspecting peer health labels.

    `static_median` uses all peers and no residual trimming. `reliability_all`
    uses all peers with training-scale weights plus robust contemporaneous
    trimming. Nearest-peer ablations isolate unweighted subset selection,
    reliability weighting, and distance weighting. `regime_matched` combines
    reliability and Gaussian distance weights after nearest-peer selection.
    No variant isolates robust trimming as a separate factor.
    """
    rows = []
    for timestamp, group in frame.groupby("timestamp", sort=False):
        group = group.reset_index(drop=True)
        residual = group["residual"].to_numpy(dtype=float)
        descriptor = group[[f"regime_{index}" for index in range(7)]].to_numpy(dtype=float)
        turbine = group["turbine_id"].astype(str).to_numpy()
        for index in range(len(group)):
            candidates = np.array([other for other in range(len(group)) if other != index], dtype=int)
            if len(candidates) < MIN_PEERS_TO_SCORE:
                continue
            distances = np.sqrt(np.sum((descriptor[candidates] - descriptor[index]) ** 2, axis=1))
            nearest_methods = {
                "nearest_unweighted",
                "nearest_reliability",
                "nearest_distance",
                "regime_matched",
            }
            if method in nearest_methods:
                rank = np.argsort(distances, kind="stable")
                selected_count = min(int(peer_count), len(candidates))
                selected = candidates[rank[:selected_count]]
                selected_distances = distances[rank[:selected_count]]
            else:
                selected = candidates
                selected_distances = distances
            values = residual[selected]
            reference = np.nan
            retained_count = len(selected)
            if method in {"static_median", "nearest_unweighted"}:
                reference = float(np.median(values))
            else:
                centre = float(np.median(values))
                robust_scale = max(
                    1.4826 * float(np.median(np.abs(values - centre))),
                    float(np.median([residual_scales[turbine[peer]] for peer in selected])),
                    MIN_TRAINING_SCALE,
                )
                keep = np.abs(values - centre) <= 4.5 * robust_scale
                values = values[keep]
                selected = selected[keep]
                selected_distances = selected_distances[keep]
                retained_count = len(selected)
                if method in {"reliability_all", "nearest_reliability", "regime_matched"}:
                    weights = np.array(
                        [1.0 / max(residual_scales[turbine[peer]], MIN_TRAINING_SCALE) ** 2 for peer in selected],
                        dtype=float,
                    )
                elif method == "nearest_distance":
                    weights = np.ones(len(selected), dtype=float)
                else:
                    raise ValueError(f"Unknown peer method: {method}")
                if method in {"nearest_distance", "regime_matched"}:
                    log_kernel = -0.5 * (selected_distances / float(bandwidth)) ** 2
                    if stabilize_kernel:
                        log_kernel -= np.max(log_kernel)
                    weights *= np.exp(log_kernel)
                reference = weighted_median(values, weights)
            rows.append(
                {
                    "target": target,
                    "timestamp": timestamp,
                    "turbine_id": turbine[index],
                    "method": method,
                    "residual": float(residual[index]),
                    "peer_reference": reference,
                    "specific_residual": float(residual[index] - reference),
                    "candidate_peer_count": int(
                        len(candidates)
                        if method not in nearest_methods
                        else min(int(peer_count), len(candidates))
                    ),
                    "retained_peer_count": int(retained_count),
                    "mean_selected_distance": float(np.mean(selected_distances)),
                }
            )
    return pd.DataFrame(rows)


def fit_target_model(features, target_values, turbine, usable, train_period, valid_period, test_period):
    train_mask = usable & train_period
    valid_mask = usable & valid_period
    test_mask = usable & test_period
    x_train = features[train_mask]
    y_train = target_values[train_mask]
    x_valid = features[valid_mask]
    y_valid = target_values[valid_mask]
    initial = fit_ridge(x_train, y_train, alpha=10.0)
    initial_residual = y_train - predict(initial, x_train)
    retained, _ = robust_training_keep(initial_residual, turbine[train_mask])
    candidates = []
    models = {}
    for alpha in ALPHAS:
        model = fit_ridge(x_train[retained], y_train[retained], alpha=alpha)
        validation_prediction = predict(model, x_valid)
        candidates.append((alpha, rmse(y_valid, validation_prediction), mae(y_valid, validation_prediction)))
        models[alpha] = model
    alpha, validation_rmse, validation_mae = min(candidates, key=lambda value: value[1])
    model = models[alpha]
    train_residual = y_train - predict(model, x_train)
    _, residual_scales = robust_training_keep(train_residual, turbine[train_mask])
    return {
        "model": model,
        "alpha": alpha,
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "train_rows": int(len(x_train)),
        "train_rows_retained": int(retained.sum()),
        "valid_mask": valid_mask,
        "test_mask": test_mask,
        "residual_scales": residual_scales,
    }


def residual_frame(data, residual, mask, descriptor):
    frame = pd.DataFrame(
        {
            "timestamp": data.loc[mask, "timestamp"].to_numpy(),
            "turbine_id": data.loc[mask, "turbine_id"].astype(str).to_numpy(),
            "residual": residual,
        }
    )
    for index in range(descriptor.shape[1]):
        frame[f"regime_{index}"] = descriptor[mask, index]
    return frame.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiment_results" / "health_twin_regime_peer_v3",
    )
    parser.add_argument("--ambient-source", choices=sorted(AMBIENT_SOURCES), default="independent_meteorology")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Explicit provenance identifier. Defaults to a source-specific identifier.",
    )
    parser.add_argument(
        "--monitoring-start",
        default=None,
        help="Override source-specific first permitted timestamp. Independent meteorology cannot start before its quality gate.",
    )
    parser.add_argument("--train-end", default=str(TRAIN_END))
    parser.add_argument("--valid-end", default=str(VALID_END))
    parser.add_argument("--test-end", default=str(TEST_END))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ambient_specification = AMBIENT_SOURCES[args.ambient_source]
    ambient_column = ambient_specification["column"]
    monitoring_start = (
        pd.Timestamp(args.monitoring_start)
        if args.monitoring_start is not None
        else ambient_specification["default_start"]
    )
    if args.ambient_source == "independent_meteorology" and monitoring_start < MONITORING_START:
        raise ValueError("Independent meteorology cannot be used before the approved 2015-05-14 12:00 quality gate.")
    experiment_id = args.experiment_id or (
        "health_twin_regime_peer_v3"
        if args.ambient_source == "independent_meteorology"
        else "health_twin_local_ambient_regime_peer_v1"
    )
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    test_end = pd.Timestamp(args.test_end)
    if not monitoring_start < train_end < valid_end < test_end:
        raise ValueError("Require monitoring_start < train_end < valid_end < test_end.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "test_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite to replace it")

    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    features, feature_names = make_feature_matrix(data, turbine_ids, ambient_column)
    descriptor, descriptor_names = regime_matrix(data, ambient_column)
    base_eligible = (
        data["timestamp"].ge(monitoring_start)
        & data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(descriptor).all(axis=1)
    ).to_numpy()
    train_period = data["timestamp"].ge(monitoring_start).to_numpy() & data["timestamp"].lt(train_end).to_numpy()
    valid_period = data["timestamp"].ge(train_end).to_numpy() & data["timestamp"].lt(valid_end).to_numpy()
    test_period = data["timestamp"].ge(valid_end).to_numpy() & data["timestamp"].lt(test_end).to_numpy()
    regime_mean = np.mean(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.std(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
    descriptor = (descriptor - regime_mean) / regime_scale
    turbine = data["turbine_id"].astype(str).to_numpy()

    validation_grid = []
    test_metrics = []
    all_scores = []
    model_provenance = {}
    for target in TARGETS:
        target_values = data[target].to_numpy(dtype=float)
        usable = base_eligible & np.isfinite(target_values)
        fitted = fit_target_model(features, target_values, turbine, usable, train_period, valid_period, test_period)
        valid_mask = fitted["valid_mask"]
        test_mask = fitted["test_mask"]
        validation_residual = target_values[valid_mask] - predict(fitted["model"], features[valid_mask])
        test_prediction = predict(fitted["model"], features[test_mask])
        test_residual = target_values[test_mask] - test_prediction
        valid_frame = residual_frame(data, validation_residual, valid_mask, descriptor)
        test_frame = residual_frame(data, test_residual, test_mask, descriptor)

        for peer_count in PEER_COUNTS:
            for bandwidth in DISTANCE_BANDWIDTHS:
                candidate_scores = peer_reference_scores(
                    valid_frame,
                    target,
                    "regime_matched",
                    fitted["residual_scales"],
                    peer_count=peer_count,
                    bandwidth=bandwidth,
                )
                validation_grid.append(
                    {
                        "target": target,
                        "peer_count": peer_count,
                        "distance_bandwidth": bandwidth,
                        **stable_summary(candidate_scores),
                    }
                )
        candidate_grid = pd.DataFrame([row for row in validation_grid if row["target"] == target])
        selected = candidate_grid.sort_values(
            ["specific_median_abs", "specific_std", "peer_count", "distance_bandwidth"],
            kind="stable",
        ).iloc[0]
        selected_count = int(selected["peer_count"])
        selected_bandwidth = float(selected["distance_bandwidth"])
        baseline_static = peer_reference_scores(test_frame, target, "static_median", fitted["residual_scales"])
        baseline_reliability = peer_reference_scores(test_frame, target, "reliability_all", fitted["residual_scales"])
        regime_scores = peer_reference_scores(
            test_frame,
            target,
            "regime_matched",
            fitted["residual_scales"],
            peer_count=selected_count,
            bandwidth=selected_bandwidth,
        )
        all_scores.extend([baseline_static, baseline_reliability, regime_scores])
        for method, scores in [
            ("static_median", baseline_static),
            ("reliability_all", baseline_reliability),
            ("regime_matched", regime_scores),
        ]:
            summary = stable_summary(scores)
            test_metrics.append(
                {
                    "target": target,
                    "method": method,
                    "selected_peer_count": selected_count if method == "regime_matched" else np.nan,
                    "selected_distance_bandwidth": selected_bandwidth if method == "regime_matched" else np.nan,
                    "normal_model_validation_rmse": fitted["validation_rmse"],
                    "normal_model_test_rmse": rmse(target_values[test_mask], test_prediction),
                    "normal_model_test_mae": mae(target_values[test_mask], test_prediction),
                    **summary,
                }
            )
        model_provenance[target] = {
            "normal_model_alpha": fitted["alpha"],
            "normal_model_train_rows": fitted["train_rows"],
            "normal_model_train_rows_retained": fitted["train_rows_retained"],
            "selected_regime_peer_count": selected_count,
            "selected_regime_distance_bandwidth": selected_bandwidth,
            "validation_selection_objective": "minimum median absolute leave-one-out specific residual; tie-break by standard deviation, then fewer peers and narrower bandwidth",
            "residual_scales_by_turbine": fitted["residual_scales"],
        }
        print(f"[{target}] selected regime peers={selected_count}, bandwidth={selected_bandwidth}; test regime score std={stable_summary(regime_scores)['specific_std']:.3f}", flush=True)

    validation_grid = pd.DataFrame(validation_grid)
    test_metrics = pd.DataFrame(test_metrics)
    scores = pd.concat(all_scores, ignore_index=True)
    validation_grid.to_csv(args.output_dir / "validation_grid.csv", index=False)
    test_metrics.to_csv(metrics_path, index=False)
    scores.to_csv(args.output_dir / "test_peer_scores.csv.gz", index=False, compression="gzip", float_format="%.8g")
    wide = test_metrics.pivot(index="target", columns="method", values="specific_std")
    wide["regime_vs_static_std_change_pct"] = (wide["regime_matched"] / wide["static_median"] - 1.0) * 100.0
    wide["regime_vs_reliability_std_change_pct"] = (wide["regime_matched"] / wide["reliability_all"] - 1.0) * 100.0
    comparison = wide.reset_index()
    comparison.to_csv(args.output_dir / "test_comparison.csv", index=False)
    manifest = {
        "experiment": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential research result; no raw SCADA observations are redistributed",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "targets": TARGETS,
        "ambient_source": args.ambient_source,
        "ambient_column": ambient_column,
        "ambient_source_interpretation": ambient_specification["interpretation"],
        "normal_model_features": feature_names,
        "regime_distance_features": descriptor_names,
        "regime_standardization": "mean and standard deviation fitted on eligible training records only",
        "candidate_peer_counts": PEER_COUNTS,
        "candidate_distance_bandwidths": DISTANCE_BANDWIDTHS,
        "minimum_other_peers_to_score": MIN_PEERS_TO_SCORE,
        "splits": {
            "monitoring_start_inclusive": str(monitoring_start),
            "ambient_source_quality_contract": ambient_specification["interpretation"],
            "train_end_exclusive": str(train_end),
            "validation_end_exclusive": str(valid_end),
            "test_end_exclusive": str(test_end),
        },
        "selection_rule": "target-specific validation minimisation of median absolute leave-one-out specific residual; no test-period selection",
        "method_definitions": {
            "static_median": "all available leave-one-out peers; unweighted residual median",
            "reliability_all": "all available leave-one-out peers; training residual-scale weights plus contemporaneous robust trim",
            "regime_matched": "closest operating-regime peers selected by train-standardised covariate distance; reliability and Gaussian distance weights plus contemporaneous robust trim",
        },
        "model_provenance": model_provenance,
        "limits": [
            "Parameter selection uses a label-free residual-stability proxy, not failure-detection accuracy.",
            "Same-interval operating values are permitted only after interval completion for condition monitoring.",
            "No fault/event/maintenance labels are present, so lower specific-residual spread is feasibility evidence only.",
            "Direction and wake geometry remain excluded from the main peer rule because validated along-axis directional exposure is sparse.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(native(manifest), indent=2), encoding="utf-8")
    report = [
        f"# Dynamic regime-conditioned peer experiment: {experiment_id} — confidential local artifact",
        "",
        "## Question",
        "",
        "Can an operating-regime-matched peer reference produce a more stable turbine-specific thermal residual than a static fleet median or reliability-only fleet reference, when all choices are made before the final chronological test period?",
        "",
        "## Protocol",
        "",
        f"The normal-behaviour model is fit from {monitoring_start:%Y-%m-%d %H:%M} onward. Its ambient covariate is `{ambient_column}` ({ambient_specification['interpretation']}). Peer distance uses only power, primary wind speed, generator and main-shaft RPM, pitch mean/spread, and the selected ambient covariate. These variables are standardised on training data only. The target turbine is always excluded; neither target temperature nor a fault label enters peer selection.",
        "",
        f"Validation selects the number of nearest peers and Gaussian distance bandwidth by minimum median absolute leave-one-out specific residual. This is a label-free stability proxy only. The final test period is {valid_end:%B}–{(test_end - pd.Timedelta(days=1)):%B}.",
        "",
        "## Test comparison",
        "",
        markdown_table(comparison, ["target", "static_median", "reliability_all", "regime_matched", "regime_vs_static_std_change_pct", "regime_vs_reliability_std_change_pct"]),
        "",
        "A negative percentage indicates a lower test specific-residual standard deviation for regime-matched peers. This is not a detection-rate result and must not be interpreted as fault-diagnosis performance.",
        "",
        "## Selected dynamic configurations",
        "",
        markdown_table(test_metrics.loc[test_metrics["method"].eq("regime_matched")], ["target", "selected_peer_count", "selected_distance_bandwidth", "score_rows", "specific_median_abs", "specific_std", "pct_with_at_least_5_candidate_peers", "mean_selected_distance"]),
        "",
        "## Interpretation boundary",
        "",
        "This experiment tests the central matched-peer mechanism under a source-appropriate chronological protocol. A favourable stability result motivates rolling-season and leave-one-turbine-out tests; an unfavourable result means the method must be simplified or the novelty claim narrowed. Neither outcome licenses a claim of fault detection without independent events or maintenance evidence.",
    ]
    (args.output_dir / "RESULTS_v3.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
