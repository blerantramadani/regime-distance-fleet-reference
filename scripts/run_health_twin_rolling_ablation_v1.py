#!/usr/bin/env python3
"""Run pre-specified rolling-origin matched-peer ablations locally.

Every fold independently fits preprocessing and the normal-behaviour model,
selects the full matched-peer configuration on validation data, and evaluates
the next chronological month. Test data never selects a model, peer count,
bandwidth, target, or threshold. This is a repeated label-free backtest, not
fault-detection validation or untouched external confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_regime_peer_block_bootstrap import daily_sufficient_statistics, std_from_stats
from run_health_twin_baseline import (
    AMBIENT_SOURCES,
    MIN_TRAINING_SCALE,
    SOURCE_COLUMNS,
    TARGETS,
    fit_ridge,
    mae,
    make_feature_matrix,
    predict,
    rmse,
    robust_training_keep,
)
from run_health_twin_regime_peer_v2 import (
    peer_reference_scores,
    regime_matrix,
    residual_frame,
    stable_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
PEER_GRID = [3, 5, 8, 10, 12, 15]
BANDWIDTH_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
METHODS = [
    "static_median",
    "reliability_all",
    "nearest_unweighted",
    "nearest_reliability",
    "nearest_distance",
    "regime_matched",
]

LOCAL_FOLDS = [
    {"fold": "L1", "start": "2015-01-01", "train_end": "2015-04-01", "valid_end": "2015-05-01", "test_end": "2015-06-01"},
    {"fold": "L2", "start": "2015-01-01", "train_end": "2015-06-01", "valid_end": "2015-07-01", "test_end": "2015-08-01"},
    {"fold": "L3", "start": "2015-01-01", "train_end": "2015-08-01", "valid_end": "2015-09-01", "test_end": "2015-10-01"},
    {"fold": "L4", "start": "2015-01-01", "train_end": "2015-10-01", "valid_end": "2015-11-01", "test_end": "2015-12-01"},
    {"fold": "L5", "start": "2015-01-01", "train_end": "2015-11-01", "valid_end": "2015-12-01", "test_end": "2016-01-01"},
]
INDEPENDENT_FOLDS = [
    {"fold": "M1", "start": "2015-05-14 12:00", "train_end": "2015-08-01", "valid_end": "2015-09-01", "test_end": "2015-10-01"},
    {"fold": "M2", "start": "2015-05-14 12:00", "train_end": "2015-09-01", "valid_end": "2015-10-01", "test_end": "2015-11-01"},
    {"fold": "M3", "start": "2015-05-14 12:00", "train_end": "2015-10-01", "valid_end": "2015-11-01", "test_end": "2015-12-01"},
    {"fold": "M4", "start": "2015-05-14 12:00", "train_end": "2015-11-01", "valid_end": "2015-12-01", "test_end": "2016-01-01"},
]


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


def cell(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def markdown_table(frame, columns=None):
    if columns is not None:
        frame = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def fit_window(features, target, turbine, usable, train_period, valid_period, test_period):
    train_mask = usable & train_period
    valid_mask = usable & valid_period
    test_mask = usable & test_period
    x_train, y_train = features[train_mask], target[train_mask]
    x_valid, y_valid = features[valid_mask], target[valid_mask]
    if min(len(x_train), len(x_valid), int(test_mask.sum())) == 0:
        raise ValueError("Fold has an empty train, validation, or test partition.")
    initial = fit_ridge(x_train, y_train, alpha=10.0)
    retained, _ = robust_training_keep(y_train - predict(initial, x_train), turbine[train_mask])
    candidates, models = [], {}
    for alpha in ALPHA_GRID:
        model = fit_ridge(x_train[retained], y_train[retained], alpha=alpha)
        validation_prediction = predict(model, x_valid)
        candidates.append((alpha, rmse(y_valid, validation_prediction), mae(y_valid, validation_prediction)))
        models[alpha] = model
    alpha, validation_rmse, validation_mae = min(candidates, key=lambda item: item[1])
    model = models[alpha]
    _, residual_scales = robust_training_keep(
        y_train - predict(model, x_train), turbine[train_mask]
    )
    return {
        "model": model,
        "alpha": float(alpha),
        "alpha_boundary": bool(alpha in {min(ALPHA_GRID), max(ALPHA_GRID)}),
        "validation_rmse": validation_rmse,
        "validation_mae": validation_mae,
        "train_rows": int(len(x_train)),
        "train_rows_retained": int(retained.sum()),
        "valid_mask": valid_mask,
        "test_mask": test_mask,
        "residual_scales": residual_scales,
    }


def moving_block_ci(scores, comparator, block_days, iterations, seed):
    subset = scores.loc[scores["method"].isin([comparator, "regime_matched"])].copy()
    subset["day"] = subset["timestamp"].dt.floor("D")
    days, cube = daily_sufficient_statistics(subset, [comparator, "regime_matched"])
    if len(days) < block_days:
        return {"ci95_low_pct": np.nan, "ci95_high_pct": np.nan, "bootstrap_probability_improvement": np.nan}
    starts = np.arange(len(days) - block_days + 1)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected = []
        while len(selected) < len(days):
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block_days))
        standard_deviation = std_from_stats(cube[np.asarray(selected[: len(days)])].sum(axis=0))
        values[iteration] = (standard_deviation[1] / standard_deviation[0] - 1.0) * 100.0
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "ci95_low_pct": float(low),
        "ci95_high_pct": float(high),
        "bootstrap_probability_improvement": float(np.mean(values < 0.0)),
    }


def score_method(test_frame, target_name, method, residual_scales, peer_count, bandwidth):
    kwargs = {}
    if method.startswith("nearest_") or method == "regime_matched":
        kwargs["peer_count"] = peer_count
    if method in {"nearest_distance", "regime_matched"}:
        kwargs["bandwidth"] = bandwidth
    return peer_reference_scores(test_frame, target_name, method, residual_scales, **kwargs)


def validation_configuration_grid(frame, target_name, residual_scales):
    """Evaluate every K/bandwidth pair while computing distances only once.

    This is algebraically equivalent to repeated `peer_reference_scores` calls
    for the full regime-matched method, but avoids repeating group construction,
    pairwise distance calculation, and residual sorting for each configuration.
    """
    configuration_values = {(k, bandwidth): [] for k in PEER_GRID for bandwidth in BANDWIDTH_GRID}
    retained_counts = {(k, bandwidth): [] for k in PEER_GRID for bandwidth in BANDWIDTH_GRID}
    candidate_counts = {(k, bandwidth): [] for k in PEER_GRID for bandwidth in BANDWIDTH_GRID}
    selected_distances = {(k, bandwidth): [] for k in PEER_GRID for bandwidth in BANDWIDTH_GRID}
    for _, group in frame.groupby("timestamp", sort=False):
        group = group.reset_index(drop=True)
        residual = group["residual"].to_numpy(dtype=float)
        descriptor = group[[f"regime_{index}" for index in range(7)]].to_numpy(dtype=float)
        turbine = group["turbine_id"].astype(str).to_numpy()
        for index in range(len(group)):
            candidates = np.array([other for other in range(len(group)) if other != index], dtype=int)
            if len(candidates) < 3:
                continue
            distances = np.sqrt(np.sum((descriptor[candidates] - descriptor[index]) ** 2, axis=1))
            rank = np.argsort(distances, kind="stable")
            ordered_candidates = candidates[rank]
            ordered_distances = distances[rank]
            for peer_count in PEER_GRID:
                selected_count = min(peer_count, len(ordered_candidates))
                selected = ordered_candidates[:selected_count]
                distance = ordered_distances[:selected_count]
                values = residual[selected]
                centre = float(np.median(values))
                robust_scale = max(
                    1.4826 * float(np.median(np.abs(values - centre))),
                    float(np.median([residual_scales[turbine[peer]] for peer in selected])),
                    MIN_TRAINING_SCALE,
                )
                keep = np.abs(values - centre) <= 4.5 * robust_scale
                values = values[keep]
                selected = selected[keep]
                distance = distance[keep]
                value_order = np.argsort(values, kind="stable")
                values = values[value_order]
                distance = distance[value_order]
                reliability = np.array(
                    [1.0 / max(residual_scales[turbine[peer]], MIN_TRAINING_SCALE) ** 2 for peer in selected],
                    dtype=float,
                )[value_order]
                for bandwidth in BANDWIDTH_GRID:
                    log_kernel = -0.5 * (distance / bandwidth) ** 2
                    log_kernel -= np.max(log_kernel)
                    weights = reliability * np.exp(log_kernel)
                    reference = float(
                        values[
                            np.searchsorted(
                                np.cumsum(weights), weights.sum() / 2.0, side="left"
                            )
                        ]
                    )
                    key = (peer_count, bandwidth)
                    configuration_values[key].append(float(residual[index] - reference))
                    retained_counts[key].append(int(len(values)))
                    candidate_counts[key].append(int(selected_count))
                    selected_distances[key].append(float(np.mean(distance)))
    rows = []
    for peer_count in PEER_GRID:
        for bandwidth in BANDWIDTH_GRID:
            key = (peer_count, bandwidth)
            values = np.asarray(configuration_values[key], dtype=float)
            candidates = np.asarray(candidate_counts[key], dtype=float)
            rows.append(
                {
                    "target": target_name,
                    "peer_count": peer_count,
                    "distance_bandwidth": bandwidth,
                    "score_rows": int(len(values)),
                    "specific_median_abs": float(np.median(np.abs(values))),
                    "specific_std": float(np.std(values)),
                    "specific_p95_abs": float(np.quantile(np.abs(values), 0.95)),
                    "mean_candidate_peers": float(np.mean(candidates)),
                    "mean_retained_peers": float(np.mean(retained_counts[key])),
                    "mean_selected_distance": float(np.mean(selected_distances[key])),
                    "pct_with_at_least_5_candidate_peers": float(np.mean(candidates >= 5) * 100.0),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument("--protocol", choices=["local", "independent"], required=True)
    parser.add_argument("--fold", default="all", help="One fold ID or 'all'.")
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    unknown_targets = sorted(set(args.targets) - set(TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    fold_definitions = LOCAL_FOLDS if args.protocol == "local" else INDEPENDENT_FOLDS
    if args.fold != "all":
        fold_definitions = [fold for fold in fold_definitions if fold["fold"] == args.fold]
        if not fold_definitions:
            raise ValueError(f"Unknown fold {args.fold} for {args.protocol} protocol.")
    ambient_source = "turbine_local_proxy" if args.protocol == "local" else "independent_meteorology"
    ambient_specification = AMBIENT_SOURCES[ambient_source]
    ambient_column = ambient_specification["column"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "fold_target_method_results.csv"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use --overwrite to replace it")

    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    features, feature_names = make_feature_matrix(data, turbine_ids, ambient_column)
    raw_regime, regime_names = regime_matrix(data, ambient_column)
    base_eligible = (
        data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(raw_regime).all(axis=1)
    ).to_numpy()
    time = data["timestamp"]
    turbine = data["turbine_id"].astype(str).to_numpy()
    grid_rows, method_rows, comparison_rows = [], [], []
    model_provenance = {}

    for fold_index, raw_fold in enumerate(fold_definitions):
        fold = {key: (pd.Timestamp(value) if key != "fold" else value) for key, value in raw_fold.items()}
        train_period = time.ge(fold["start"]).to_numpy() & time.lt(fold["train_end"]).to_numpy()
        valid_period = time.ge(fold["train_end"]).to_numpy() & time.lt(fold["valid_end"]).to_numpy()
        test_period = time.ge(fold["valid_end"]).to_numpy() & time.lt(fold["test_end"]).to_numpy()
        eligible_train = base_eligible & train_period
        regime_mean = np.mean(raw_regime[eligible_train], axis=0)
        regime_scale = np.std(raw_regime[eligible_train], axis=0)
        regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
        regime = (raw_regime - regime_mean) / regime_scale
        train_ambient = data.loc[eligible_train, ambient_column]
        ambient_min, ambient_max = float(train_ambient.min()), float(train_ambient.max())
        fold_provenance = {}
        for target_index, target_name in enumerate(args.targets):
            target = data[target_name].to_numpy(dtype=float)
            usable = base_eligible & np.isfinite(target)
            fitted = fit_window(features, target, turbine, usable, train_period, valid_period, test_period)
            validation_residual = target[fitted["valid_mask"]] - predict(
                fitted["model"], features[fitted["valid_mask"]]
            )
            test_prediction = predict(fitted["model"], features[fitted["test_mask"]])
            test_residual = target[fitted["test_mask"]] - test_prediction
            validation_frame = residual_frame(data, validation_residual, fitted["valid_mask"], regime)
            test_frame = residual_frame(data, test_residual, fitted["test_mask"], regime)

            candidate_rows = validation_configuration_grid(
                validation_frame, target_name, fitted["residual_scales"]
            )
            for row in candidate_rows:
                row["fold"] = fold["fold"]
                grid_rows.append(row)
            selected = pd.DataFrame(candidate_rows).sort_values(
                ["specific_median_abs", "specific_std", "peer_count", "distance_bandwidth"],
                kind="stable",
            ).iloc[0]
            selected_count = int(selected["peer_count"])
            selected_bandwidth = float(selected["distance_bandwidth"])
            method_scores = []
            method_summaries = {}
            for method in METHODS:
                scores = score_method(
                    test_frame,
                    target_name,
                    method,
                    fitted["residual_scales"],
                    selected_count,
                    selected_bandwidth,
                )
                method_scores.append(scores)
                summary = stable_summary(scores)
                method_summaries[method] = summary
                method_rows.append(
                    {
                        "fold": fold["fold"],
                        "test_start": fold["valid_end"],
                        "test_end_exclusive": fold["test_end"],
                        "target": target_name,
                        "method": method,
                        "normal_model_alpha": fitted["alpha"],
                        "normal_model_alpha_boundary": fitted["alpha_boundary"],
                        "normal_model_train_rows": fitted["train_rows"],
                        "normal_model_train_rows_retained": fitted["train_rows_retained"],
                        "normal_model_validation_rmse": fitted["validation_rmse"],
                        "normal_model_test_rmse": rmse(target[fitted["test_mask"]], test_prediction),
                        "normal_model_test_mae": mae(target[fitted["test_mask"]], test_prediction),
                        "selected_peer_count": selected_count,
                        "selected_distance_bandwidth": selected_bandwidth,
                        "peer_count_boundary": selected_count in {min(PEER_GRID), max(PEER_GRID)},
                        "bandwidth_boundary": selected_bandwidth in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                        **summary,
                    }
                )
            combined_scores = pd.concat(method_scores, ignore_index=True)
            full_std = method_summaries["regime_matched"]["specific_std"]
            for comparator in METHODS[:-1]:
                comparator_std = method_summaries[comparator]["specific_std"]
                ci = moving_block_ci(
                    combined_scores,
                    comparator,
                    args.block_days,
                    args.bootstrap_iterations,
                    seed=20260831 + 100 * fold_index + target_index,
                )
                comparison_rows.append(
                    {
                        "fold": fold["fold"],
                        "target": target_name,
                        "comparator": comparator,
                        "full_vs_comparator_std_change_pct": (full_std / comparator_std - 1.0) * 100.0,
                        **ci,
                    }
                )
            test_ambient = data.loc[fitted["test_mask"], ambient_column]
            fold_provenance[target_name] = {
                "normal_model_alpha": fitted["alpha"],
                "normal_model_alpha_boundary": fitted["alpha_boundary"],
                "selected_peer_count": selected_count,
                "selected_distance_bandwidth": selected_bandwidth,
                "peer_count_boundary": selected_count in {min(PEER_GRID), max(PEER_GRID)},
                "bandwidth_boundary": selected_bandwidth in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                "training_ambient_min": ambient_min,
                "training_ambient_max": ambient_max,
                "test_ambient_outside_training_minmax_pct": float((~test_ambient.between(ambient_min, ambient_max)).mean() * 100.0),
            }
            print(
                f"[{fold['fold']}][{target_name}] alpha={fitted['alpha']:g}; peers={selected_count}; bw={selected_bandwidth:g}; full-vs-static={(full_std / method_summaries['static_median']['specific_std'] - 1.0) * 100.0:.2f}%",
                flush=True,
            )
        model_provenance[fold["fold"]] = fold_provenance

    grid = pd.DataFrame(grid_rows)
    methods = pd.DataFrame(method_rows)
    comparisons = pd.DataFrame(comparison_rows)
    grid.to_csv(args.output_dir / "validation_grid.csv", index=False)
    methods.to_csv(result_path, index=False)
    comparisons.to_csv(args.output_dir / "full_method_comparisons.csv", index=False)

    static = comparisons.loc[comparisons["comparator"].eq("static_median")]
    rolling_rows = []
    for target, group in static.groupby("target", sort=True):
        rolling_rows.append(
            {
                "target": target,
                "folds": int(len(group)),
                "folds_lower_spread": int((group["full_vs_comparator_std_change_pct"] < 0).sum()),
                "folds_ci_excludes_zero_below": int((group["ci95_high_pct"] < 0).sum()),
                "folds_ci_excludes_zero_above": int((group["ci95_low_pct"] > 0).sum()),
                "median_std_change_pct": float(group["full_vs_comparator_std_change_pct"].median()),
                "min_std_change_pct": float(group["full_vs_comparator_std_change_pct"].min()),
                "max_std_change_pct": float(group["full_vs_comparator_std_change_pct"].max()),
            }
        )
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(args.output_dir / "rolling_target_summary.csv", index=False)
    selected_rows = methods.loc[methods["method"].eq("regime_matched")]
    report = [
        f"# Rolling-origin matched-peer ablation v1: {args.protocol}",
        "",
        "This is a pre-specified repeated chronological backtest. Every fold performs all fitting and selection before its test month. Because 2015 informed method development, these folds are not described as untouched external confirmation.",
        "",
        "## Rolling comparison against static median",
        "",
        markdown_table(rolling),
        "",
        "## Selected configurations and full-method test results",
        "",
        markdown_table(
            selected_rows,
            [
                "fold",
                "target",
                "normal_model_alpha",
                "normal_model_alpha_boundary",
                "selected_peer_count",
                "peer_count_boundary",
                "selected_distance_bandwidth",
                "bandwidth_boundary",
                "specific_std",
                "specific_median_abs",
            ],
        ),
        "",
        "## Ablation comparisons",
        "",
        markdown_table(
            comparisons,
            [
                "fold",
                "target",
                "comparator",
                "full_vs_comparator_std_change_pct",
                "ci95_low_pct",
                "ci95_high_pct",
            ],
        ),
        "",
        "Negative change favours the full distance-plus-reliability method. Seven-day block-bootstrap intervals account for short-range temporal clustering within each test month. No result is fault-detection accuracy.",
    ]
    (args.output_dir / "ROLLING_ABLATION_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "experiment": f"health_twin_rolling_ablation_{args.protocol}_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential aggregate research result",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "protocol": args.protocol,
        "ambient_source": ambient_source,
        "ambient_column": ambient_column,
        "ambient_source_interpretation": ambient_specification["interpretation"],
        "targets": args.targets,
        "folds": fold_definitions,
        "normal_model_features": feature_names,
        "regime_distance_features": regime_names,
        "alpha_grid": ALPHA_GRID,
        "peer_count_grid": PEER_GRID,
        "bandwidth_grid": BANDWIDTH_GRID,
        "distance_weight_numerics": "Gaussian log-kernel is shifted by its maximum before exponentiation; weighted-median ratios are unchanged and underflow is avoided",
        "methods": METHODS,
        "selection_rule": "per fold and target: validation RMSE selects ridge alpha; validation median absolute full-method specific residual selects peer count and bandwidth; test selects nothing",
        "block_bootstrap": {"block_days": args.block_days, "iterations": args.bootstrap_iterations},
        "model_provenance": model_provenance,
        "confirmation_boundary": "repeated 2015 development backtest; not untouched external confirmation",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(native(manifest), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
