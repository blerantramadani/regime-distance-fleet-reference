#!/usr/bin/env python3
"""Run a local label-free normal-behaviour / matched-peer baseline.

The baseline is a research feasibility benchmark, not a fault classifier.  It
fits each selected thermal signal from contemporaneous, completed-interval
operating conditions, then produces a leave-one-out peer residual reference.
All fitting, robust trimming, scaling and hyperparameter choice use only the
chronological training/validation periods.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "gear_oil_temp_mean",
    "gen_bearing_g_temp_mean",
    "gen_bearing_r_temp_mean",
    "main_bearing_temp_mean",
    "hydraulic_oil_temp_mean",
]
SOURCE_COLUMNS = [
    "timestamp",
    "turbine_id",
    "has_temperature_record",
    "active_power_mean",
    "wind_speed_mean",
    "generator_speed_mean",
    "main_shaft_speed_mean",
    "pitch_a_mean",
    "pitch_b_mean",
    "pitch_c_mean",
    "power_reference_end",
    "met_temperature_mean",
    "local_ambient_temp_mean",
    *TARGETS,
]
TRAIN_END = pd.Timestamp("2015-09-01 00:00:00")
VALID_END = pd.Timestamp("2015-11-01 00:00:00")
TEST_END = pd.Timestamp("2016-01-01 00:00:00")
MONITORING_START = pd.Timestamp("2015-05-14 12:00:00")
AMBIENT_SOURCES = {
    "independent_meteorology": {
        "column": "met_temperature_mean",
        "default_start": MONITORING_START,
        "interpretation": "independent meteorological temperature; quality-gated from 2015-05-14 12:00",
    },
    "turbine_local_proxy": {
        "column": "local_ambient_temp_mean",
        "default_start": pd.Timestamp("2015-01-01 00:00:00"),
        "interpretation": "turbine-local ambient proxy; candidate sensitivity covariate only, pending OEM signal semantics",
    },
}
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_TRAINING_SCALE = 0.25


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


def make_feature_matrix(
    data: pd.DataFrame, turbine_ids: list[str], ambient_column: str
) -> tuple[np.ndarray, list[str]]:
    power = data["active_power_mean"].to_numpy(dtype=float) / 2300.0
    wind = data["wind_speed_mean"].to_numpy(dtype=float)
    gen_rpm = data["generator_speed_mean"].to_numpy(dtype=float) / 1450.0
    main_rpm = data["main_shaft_speed_mean"].to_numpy(dtype=float)
    pitch = data[["pitch_a_mean", "pitch_b_mean", "pitch_c_mean"]].to_numpy(dtype=float)
    pitch_mean = np.mean(pitch, axis=1)
    pitch_spread = np.max(pitch, axis=1) - np.min(pitch, axis=1)
    ambient = data[ambient_column].to_numpy(dtype=float)
    timestamps = pd.DatetimeIndex(data["timestamp"])
    day_position = (
        timestamps.dayofyear.to_numpy(dtype=float) - 1.0
        + (timestamps.hour.to_numpy(dtype=float) * 60.0 + timestamps.minute.to_numpy(dtype=float)) / 1440.0
    )
    annual_angle = 2.0 * np.pi * day_position / 365.0
    columns = [
        power,
        power**2,
        wind,
        wind**2,
        gen_rpm,
        main_rpm,
        pitch_mean,
        pitch_spread,
        ambient,
        ambient**2,
        power * ambient,
        wind * ambient,
        np.sin(annual_angle),
        np.cos(annual_angle),
    ]
    names = [
        "power_ratio",
        "power_ratio_squared",
        "primary_wind_speed",
        "primary_wind_speed_squared",
        "generator_rpm_ratio",
        "main_shaft_rpm",
        "pitch_mean",
        "pitch_spread",
        f"{ambient_column}",
        f"{ambient_column}_squared",
        "power_x_ambient_temperature",
        "wind_x_ambient_temperature",
        "annual_sine",
        "annual_cosine",
    ]
    # One turbine is intentionally omitted: the explicit intercept is its
    # baseline; remaining dummies provide turbine-specific normal offsets.
    turbine = data["turbine_id"].astype(str).to_numpy()
    for turbine_id in turbine_ids[:-1]:
        columns.append((turbine == turbine_id).astype(float))
        names.append(f"turbine_is_{turbine_id}")
    return np.column_stack(columns), names


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    xtx = np.einsum("ij,ik->jk", design, design, optimize=True)
    xty = np.einsum("ij,i->j", design, y, optimize=True)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(xtx + penalty, xty)
    return {"mean": mean, "scale": scale, "beta": beta, "alpha": alpha}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    standardized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return np.einsum("ij,j->i", design, model["beta"], optimize=True)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def robust_training_keep(residual: np.ndarray, turbine: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    keep = np.zeros(len(residual), dtype=bool)
    scales: dict[str, float] = {}
    for turbine_id in np.unique(turbine):
        index = turbine == turbine_id
        values = residual[index]
        centre = np.median(values)
        scale = max(1.4826 * np.median(np.abs(values - centre)), MIN_TRAINING_SCALE)
        scales[str(turbine_id)] = float(scale)
        keep[index] = np.abs(values - centre) <= 4.5 * scale
    return keep, scales


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = ordered_weights.sum() / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), threshold, side="left")])


def peer_scores(test: pd.DataFrame, residual_scales: dict[str, float], target: str) -> tuple[pd.DataFrame, dict]:
    turbine_ids = sorted(residual_scales)
    residual_wide = test.pivot(index="timestamp", columns="turbine_id", values="residual").reindex(columns=turbine_ids)
    values = residual_wide.to_numpy(dtype=float)
    scales = np.array([residual_scales[turbine_id] for turbine_id in turbine_ids], dtype=float)
    rows = []
    for time_index, timestamp in enumerate(residual_wide.index):
        current = values[time_index]
        available = np.flatnonzero(np.isfinite(current))
        if not len(available):
            continue
        for current_index in available:
            peer_indices = available[available != current_index]
            peer_values = current[peer_indices]
            record = {
                "target": target,
                "timestamp": timestamp,
                "turbine_id": turbine_ids[current_index],
                "residual": float(current[current_index]),
                "other_peer_count": int(len(peer_indices)),
                "unweighted_peer_reference": np.nan,
                "reliability_peer_reference": np.nan,
                "unweighted_specific_residual": np.nan,
                "reliability_specific_residual": np.nan,
            }
            if len(peer_indices):
                unweighted = float(np.median(peer_values))
                peer_centre = unweighted
                peer_scale = max(
                    1.4826 * np.median(np.abs(peer_values - peer_centre)),
                    float(np.median(scales[peer_indices])),
                    MIN_TRAINING_SCALE,
                )
                keep = np.abs(peer_values - peer_centre) <= 4.5 * peer_scale
                robust_values = peer_values[keep]
                robust_weights = 1.0 / np.maximum(scales[peer_indices][keep], MIN_TRAINING_SCALE) ** 2
                reliability = weighted_median(robust_values, robust_weights)
                record.update(
                    {
                        "unweighted_peer_reference": unweighted,
                        "reliability_peer_reference": reliability,
                        "unweighted_specific_residual": float(current[current_index] - unweighted),
                        "reliability_specific_residual": float(current[current_index] - reliability),
                    }
                )
            rows.append(record)
    scores = pd.DataFrame(rows)
    valid_unweighted = scores["unweighted_specific_residual"].notna()
    valid_reliability = scores["reliability_specific_residual"].notna()
    summary = {
        "test_score_rows": int(len(scores)),
        "pct_with_at_least_3_peers": float((scores["other_peer_count"] >= 3).mean() * 100),
        "pct_with_at_least_5_peers": float((scores["other_peer_count"] >= 5).mean() * 100),
        "raw_residual_std": float(scores["residual"].std(ddof=0)),
        "unweighted_specific_std": float(scores.loc[valid_unweighted, "unweighted_specific_residual"].std(ddof=0)),
        "reliability_specific_std": float(scores.loc[valid_reliability, "reliability_specific_residual"].std(ddof=0)),
        "unweighted_peer_reference_std": float(scores.loc[valid_unweighted, "unweighted_peer_reference"].std(ddof=0)),
        "reliability_peer_reference_std": float(scores.loc[valid_reliability, "reliability_peer_reference"].std(ddof=0)),
    }
    return scores, summary


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
        default=PROJECT_ROOT / "data" / "experiment_results" / "health_twin_baseline_v2",
    )
    parser.add_argument("--ambient-source", choices=sorted(AMBIENT_SOURCES), default="independent_meteorology")
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
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    test_end = pd.Timestamp(args.test_end)
    if not monitoring_start < train_end < valid_end < test_end:
        raise ValueError("Require monitoring_start < train_end < valid_end < test_end.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.output_dir / "metrics.csv"
    if metrics_file.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_file} exists; use --overwrite to replace it")

    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    features, feature_names = make_feature_matrix(data, turbine_ids, ambient_column)
    base_eligible = (
        data["timestamp"].ge(monitoring_start)
        &
        data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
    ).to_numpy()
    train_period = data["timestamp"].ge(monitoring_start).to_numpy() & data["timestamp"].lt(train_end).to_numpy()
    valid_period = data["timestamp"].ge(train_end).to_numpy() & data["timestamp"].lt(valid_end).to_numpy()
    test_period = data["timestamp"].ge(valid_end).to_numpy() & data["timestamp"].lt(test_end).to_numpy()
    turbine_array = data["turbine_id"].astype(str).to_numpy()
    metric_rows = []
    all_scores = []
    model_summary = {}

    for target in TARGETS:
        target_values = data[target].to_numpy(dtype=float)
        usable = base_eligible & np.isfinite(target_values)
        train_mask = usable & train_period
        valid_mask = usable & valid_period
        test_mask = usable & test_period
        x_train = features[train_mask]
        y_train = target_values[train_mask]
        x_valid = features[valid_mask]
        y_valid = target_values[valid_mask]
        x_test = features[test_mask]
        y_test = target_values[test_mask]
        if min(len(x_train), len(x_valid), len(x_test)) == 0:
            raise ValueError(f"No eligible train/validation/test data for {target}.")

        initial = fit_ridge(x_train, y_train, alpha=10.0)
        initial_residual = y_train - predict(initial, x_train)
        retained, initial_scales = robust_training_keep(initial_residual, turbine_array[train_mask])
        x_train_robust = x_train[retained]
        y_train_robust = y_train[retained]
        alpha_scores = []
        models = {}
        for alpha in ALPHAS:
            candidate = fit_ridge(x_train_robust, y_train_robust, alpha=alpha)
            validation_prediction = predict(candidate, x_valid)
            alpha_scores.append((alpha, rmse(y_valid, validation_prediction), mae(y_valid, validation_prediction)))
            models[alpha] = candidate
        alpha, validation_rmse, validation_mae = min(alpha_scores, key=lambda item: item[1])
        model = models[alpha]
        train_prediction = predict(model, x_train)
        test_prediction = predict(model, x_test)
        train_residual = y_train - train_prediction
        _, residual_scales = robust_training_keep(train_residual, turbine_array[train_mask])
        test_frame = pd.DataFrame(
            {
                "timestamp": data.loc[test_mask, "timestamp"].to_numpy(),
                "turbine_id": turbine_array[test_mask],
                "residual": y_test - test_prediction,
            }
        )
        scores, peer_summary = peer_scores(test_frame, residual_scales, target)
        all_scores.append(scores)
        metric_rows.append(
            {
                "target": target,
                "train_rows_before_robust_trim": int(len(x_train)),
                "train_rows_retained": int(len(x_train_robust)),
                "train_trimmed_pct": float((1.0 - len(x_train_robust) / len(x_train)) * 100),
                "validation_rows": int(len(x_valid)),
                "test_rows": int(len(x_test)),
                "selected_ridge_alpha": alpha,
                "validation_rmse": validation_rmse,
                "validation_mae": validation_mae,
                "test_rmse": rmse(y_test, test_prediction),
                "test_mae": mae(y_test, test_prediction),
                **peer_summary,
            }
        )
        model_summary[target] = {
            "selected_ridge_alpha": alpha,
            "initial_training_residual_scales_by_turbine": initial_scales,
            "final_training_residual_scales_by_turbine": residual_scales,
            "alpha_validation": [
                {"alpha": candidate_alpha, "rmse": candidate_rmse, "mae": candidate_mae}
                for candidate_alpha, candidate_rmse, candidate_mae in alpha_scores
            ],
        }
        print(f"[{target}] test RMSE={metric_rows[-1]['test_rmse']:.3f}; peer rows={peer_summary['test_score_rows']:,}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    scores = pd.concat(all_scores, ignore_index=True)
    metrics.to_csv(metrics_file, index=False)
    scores.to_csv(args.output_dir / "test_peer_scores.csv.gz", index=False, compression="gzip", float_format="%.8g")
    manifest = {
        "experiment": "health_twin_baseline_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential research result; no raw SCADA observations are redistributed",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "targets": TARGETS,
        "ambient_source": args.ambient_source,
        "ambient_column": ambient_column,
        "ambient_source_interpretation": ambient_specification["interpretation"],
        "feature_names": feature_names,
        "splits": {
            "independent_meteorology_quality_gate_start_inclusive": str(monitoring_start),
            "train": {"start_inclusive": str(monitoring_start), "end_exclusive": str(train_end)},
            "validation": {"start_inclusive": str(train_end), "end_exclusive": str(valid_end)},
            "test": {"start_inclusive": str(valid_end), "end_exclusive": str(test_end)},
        },
        "eligibility": f"timestamp is on/after the first permitted timestamp for {args.ambient_source}; temperature row exists; power 0-2350 kW; primary wind 4-20 m/s; generator RPM>=100; power reference>=2200; all model features finite",
        "robust_training_rule": "initial ridge residual is trimmed separately per turbine at 4.5 times robust MAD scale; this is an unlabeled normal-behaviour proxy, not verified healthy-data curation",
        "peer_reference": "leave-one-out contemporaneous residual median; reliability version trims peer residuals around their robust median and weights remaining peers inverse to training residual-scale squared",
        "limitations": [
            "Same-interval covariates are permitted only for post-interval condition monitoring, not advance forecasting.",
            "No fault, warning, maintenance or availability labels are used; RMSE and residual stability are not fault-detection performance metrics.",
            "Direction and wake geometry are intentionally absent from this first baseline because the direction audit found sparse validated along-axis exposure.",
            "Component labels retain original abbreviations and require supplier confirmation before equipment-specific fault claims.",
        ],
        "model_summary": model_summary,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(native(manifest), indent=2), encoding="utf-8")
    report = [
        "# Health-twin normal-behaviour baseline v2 — confidential local artifact",
        "",
        "## What this establishes",
        "",
        f"This is a reproducible **label-free** benchmark for the five selected thermal SCADA channels. A ridge normal-behaviour model uses ambient and operating covariates from the completed 10-minute interval. The ambient source is `{ambient_column}` ({ambient_specification['interpretation']}). The model is trained chronologically; a leave-one-out contemporaneous peer reference is then formed only on the later test period.",
        "",
        "It does **not** establish fault-detection accuracy, diagnosis, remaining useful life, or causal component failure. No event labels are present in this backup.",
        "",
        "## Chronological protocol",
        "",
        f"- First permitted timestamp for the selected ambient source: {monitoring_start:%Y-%m-%d %H:%M}",
        f"- Train: {monitoring_start:%Y-%m-%d} through {(train_end - pd.Timedelta(days=1)):%Y-%m-%d}",
        f"- Validation: {train_end:%Y-%m-%d} through {(valid_end - pd.Timedelta(days=1)):%Y-%m-%d}",
        f"- Test: {valid_end:%Y-%m-%d} through {(test_end - pd.Timedelta(days=1)):%Y-%m-%d}",
        "- Robust trim: training-only, separately by turbine, using 4.5 robust-MAD residual scale",
        "- Peer reference: target excluded; unweighted median and reliability-weighted robust median are both retained",
        "",
        "## Results",
        "",
        markdown_table(metrics, ["target", "train_rows_retained", "validation_rmse", "test_rmse", "test_mae", "pct_with_at_least_5_peers", "raw_residual_std", "unweighted_specific_std", "reliability_specific_std"]),
        "",
        "The two specific-residual standard deviations are descriptive. Their change relative to raw residuals is not a proof of better anomaly detection; it quantifies how much common fleet variation the peer reference separates from turbine-specific variation.",
        "",
        "## Next controlled comparison",
        "",
        "The next version should compare (1) per-turbine normal-behaviour residual only, (2) static fleet residual median, and (3) reliability-weighted matched peers across rolling seasons and leave-one-turbine-out runs. Persistent and multi-signal scores should be evaluated with calibration/alert-rate stability and synthetic step/ramp sensitivity until independent maintenance evidence becomes available.",
    ]
    (args.output_dir / "BASELINE_RESULTS_v2.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
