#!/usr/bin/env python3
"""Run a frozen all-peer trim–distance mechanism audit locally.

This post-hoc audit holds the rolling normal-behaviour protocol fixed and uses
the same all-peer leave-one-out candidate set for four factorial variants. It
isolates contemporaneous trimming, regime-distance weighting, and incremental
training-scale reliability weighting. It measures residual-state stability,
not fault-detection performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_regime_peer_block_bootstrap import daily_sufficient_statistics, std_from_stats
from run_health_twin_baseline import (
    AMBIENT_SOURCES,
    MIN_TRAINING_SCALE,
    SOURCE_COLUMNS,
    TARGETS,
    make_feature_matrix,
    mae,
    predict,
    rmse,
)
from run_health_twin_regime_peer_v2 import (
    peer_reference_scores,
    regime_matrix,
    residual_frame,
    weighted_median,
)
from run_health_twin_rolling_ablation_v1 import (
    BANDWIDTH_GRID,
    INDEPENDENT_FOLDS,
    LOCAL_FOLDS,
    fit_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "static_no_trim",
    "static_trim",
    "distance_trim",
    "distance_trim_reliability",
]
CONTRASTS = [
    ("trim", "static_trim", "static_no_trim"),
    ("distance", "distance_trim", "static_trim"),
    ("reliability", "distance_trim_reliability", "distance_trim"),
    ("end_to_end", "distance_trim_reliability", "static_no_trim"),
]
MIN_PEERS = 3
TRIM_MULTIPLIER = 4.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def markdown_table(frame: pd.DataFrame, columns=None) -> str:
    if columns is not None:
        frame = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def effective_peer_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.dot(weights, weights))
    if denominator <= 0.0:
        return np.nan
    return float(weights.sum() ** 2 / denominator)


def common_trim(
    residual: np.ndarray,
    candidate_indices: np.ndarray,
    turbine_ids: np.ndarray,
    residual_scales: dict[str, float],
) -> np.ndarray:
    values = residual[candidate_indices]
    centre = float(np.median(values))
    scale = max(
        1.4826 * float(np.median(np.abs(values - centre))),
        float(np.median([residual_scales[turbine_ids[index]] for index in candidate_indices])),
        MIN_TRAINING_SCALE,
    )
    return np.abs(values - centre) <= TRIM_MULTIPLIER * scale


def validation_bandwidth_grid(
    frame: pd.DataFrame,
    target: str,
    residual_scales: dict[str, float],
) -> list[dict]:
    state_values = {bandwidth: [] for bandwidth in BANDWIDTH_GRID}
    effective_counts = {bandwidth: [] for bandwidth in BANDWIDTH_GRID}
    retained_counts = []
    trim_fractions = []
    for _, group in frame.groupby("timestamp", sort=False):
        group = group.reset_index(drop=True)
        residual = group["residual"].to_numpy(dtype=float)
        descriptor = group[[f"regime_{index}" for index in range(7)]].to_numpy(dtype=float)
        turbine = group["turbine_id"].astype(str).to_numpy()
        for index in range(len(group)):
            candidates = np.asarray([other for other in range(len(group)) if other != index], dtype=int)
            if len(candidates) < MIN_PEERS:
                continue
            distances = np.sqrt(np.sum((descriptor[candidates] - descriptor[index]) ** 2, axis=1))
            keep = common_trim(residual, candidates, turbine, residual_scales)
            if int(keep.sum()) < MIN_PEERS:
                continue
            selected = candidates[keep]
            selected_distances = distances[keep]
            values = residual[selected]
            retained_counts.append(int(len(selected)))
            trim_fractions.append(float(1.0 - len(selected) / len(candidates)))
            for bandwidth in BANDWIDTH_GRID:
                log_kernel = -0.5 * (selected_distances / float(bandwidth)) ** 2
                log_kernel -= np.max(log_kernel)
                weights = np.exp(log_kernel)
                reference = weighted_median(values, weights)
                state_values[bandwidth].append(float(residual[index] - reference))
                effective_counts[bandwidth].append(effective_peer_count(weights))
    rows = []
    for bandwidth in BANDWIDTH_GRID:
        values = np.asarray(state_values[bandwidth], dtype=float)
        rows.append(
            {
                "target": target,
                "distance_bandwidth": float(bandwidth),
                "score_rows": int(len(values)),
                "specific_median_abs": float(np.median(np.abs(values))),
                "specific_std": float(np.std(values)),
                "specific_p95_abs": float(np.quantile(np.abs(values), 0.95)),
                "median_effective_peer_count": float(np.median(effective_counts[bandwidth])),
                "median_retained_peer_count": float(np.median(retained_counts)),
                "mean_trim_fraction": float(np.mean(trim_fractions)),
            }
        )
    return rows


def factorial_scores(
    frame: pd.DataFrame,
    target: str,
    residual_scales: dict[str, float],
    bandwidth: float,
) -> pd.DataFrame:
    rows = []
    for timestamp, group in frame.groupby("timestamp", sort=False):
        group = group.reset_index(drop=True)
        residual = group["residual"].to_numpy(dtype=float)
        descriptor = group[[f"regime_{index}" for index in range(7)]].to_numpy(dtype=float)
        turbine = group["turbine_id"].astype(str).to_numpy()
        for index in range(len(group)):
            candidates = np.asarray([other for other in range(len(group)) if other != index], dtype=int)
            if len(candidates) < MIN_PEERS:
                continue
            candidate_distances = np.sqrt(
                np.sum((descriptor[candidates] - descriptor[index]) ** 2, axis=1)
            )
            keep = common_trim(residual, candidates, turbine, residual_scales)
            if int(keep.sum()) < MIN_PEERS:
                continue
            selected = candidates[keep]
            selected_distances = candidate_distances[keep]
            candidate_values = residual[candidates]
            selected_values = residual[selected]
            log_kernel = -0.5 * (selected_distances / float(bandwidth)) ** 2
            log_kernel -= np.max(log_kernel)
            distance_weights = np.exp(log_kernel)
            reliability = np.asarray(
                [
                    1.0 / max(residual_scales[turbine[peer]], MIN_TRAINING_SCALE) ** 2
                    for peer in selected
                ],
                dtype=float,
            )
            full_weights = distance_weights * reliability
            references = {
                "static_no_trim": float(np.median(candidate_values)),
                "static_trim": float(np.median(selected_values)),
                "distance_trim": weighted_median(selected_values, distance_weights),
                "distance_trim_reliability": weighted_median(selected_values, full_weights),
            }
            effective_counts = {
                "static_no_trim": float(len(candidates)),
                "static_trim": float(len(selected)),
                "distance_trim": effective_peer_count(distance_weights),
                "distance_trim_reliability": effective_peer_count(full_weights),
            }
            retained_counts = {
                "static_no_trim": int(len(candidates)),
                "static_trim": int(len(selected)),
                "distance_trim": int(len(selected)),
                "distance_trim_reliability": int(len(selected)),
            }
            trim_fraction = float(1.0 - len(selected) / len(candidates))
            for method in METHODS:
                reference = references[method]
                rows.append(
                    {
                        "target": target,
                        "timestamp": timestamp,
                        "turbine_id": turbine[index],
                        "method": method,
                        "residual": float(residual[index]),
                        "peer_reference": reference,
                        "specific_residual": float(residual[index] - reference),
                        "candidate_peer_count": int(len(candidates)),
                        "retained_peer_count": retained_counts[method],
                        "effective_peer_count": effective_counts[method],
                        "trim_fraction": 0.0 if method == "static_no_trim" else trim_fraction,
                        "mean_selected_distance": float(
                            np.mean(candidate_distances)
                            if method == "static_no_trim"
                            else np.mean(selected_distances)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_method(scores: pd.DataFrame) -> dict:
    values = scores["specific_residual"].to_numpy(dtype=float)
    return {
        "score_rows": int(len(scores)),
        "specific_median_abs": float(np.median(np.abs(values))),
        "specific_std": float(np.std(values)),
        "specific_p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "mean_candidate_peers": float(scores["candidate_peer_count"].mean()),
        "mean_retained_peers": float(scores["retained_peer_count"].mean()),
        "median_effective_peers": float(scores["effective_peer_count"].median()),
        "p05_effective_peers": float(scores["effective_peer_count"].quantile(0.05)),
        "mean_trim_fraction": float(scores["trim_fraction"].mean()),
    }


def moving_block_contrast(
    scores: pd.DataFrame,
    numerator: str,
    denominator: str,
    block_days: int,
    iterations: int,
    seed: int,
) -> dict:
    subset = scores.loc[scores["method"].isin([denominator, numerator])].copy()
    subset["day"] = subset["timestamp"].dt.floor("D")
    days, cube = daily_sufficient_statistics(subset, [denominator, numerator])
    if len(days) < block_days:
        raise ValueError(f"Only {len(days)} test days for a {block_days}-day block.")
    observed_std = std_from_stats(cube.sum(axis=0))
    observed = float((observed_std[1] / observed_std[0] - 1.0) * 100.0)
    starts = np.arange(len(days) - block_days + 1)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected = []
        while len(selected) < len(days):
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block_days))
        standard_deviation = std_from_stats(
            cube[np.asarray(selected[: len(days)], dtype=int)].sum(axis=0)
        )
        draws[iteration] = (standard_deviation[1] / standard_deviation[0] - 1.0) * 100.0
    low, median, high = np.quantile(draws, [0.025, 0.5, 0.975])
    return {
        "test_days": int(len(days)),
        "block_days": int(block_days),
        "bootstrap_iterations": int(iterations),
        "observed_std_change_pct": observed,
        "bootstrap_median_std_change_pct": float(median),
        "ci95_low_pct": float(low),
        "ci95_high_pct": float(high),
        "bootstrap_probability_improvement": float(np.mean(draws < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument("--protocol", choices=["local", "independent"], required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--fold", default="all", help="One fold ID or 'all'.")
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--block-days", type=int, nargs="+", default=[3, 7, 14])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unknown_targets = sorted(set(args.targets) - set(TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    if args.bootstrap_iterations < 100 or min(args.block_days) < 1:
        raise ValueError("bootstrap-iterations must be >=100 and block-days must be >=1")
    fold_definitions = LOCAL_FOLDS if args.protocol == "local" else INDEPENDENT_FOLDS
    if args.fold != "all":
        fold_definitions = [fold for fold in fold_definitions if fold["fold"] == args.fold]
        if not fold_definitions:
            raise ValueError(f"Unknown fold {args.fold} for {args.protocol}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "MECHANISM_AUDIT.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; use --overwrite to replace it")
    reference_path = args.reference_dir / "fold_target_method_results.csv"
    reference = pd.read_csv(reference_path)
    ambient_source = "turbine_local_proxy" if args.protocol == "local" else "independent_meteorology"
    ambient_specification = AMBIENT_SOURCES[ambient_source]
    ambient_column = ambient_specification["column"]

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

    grid_rows = []
    method_rows = []
    contrast_rows = []
    model_provenance = {}
    for fold_index, raw_fold in enumerate(fold_definitions):
        fold = {
            key: (pd.Timestamp(value) if key != "fold" else value)
            for key, value in raw_fold.items()
        }
        train_period = time.ge(fold["start"]).to_numpy() & time.lt(fold["train_end"]).to_numpy()
        valid_period = time.ge(fold["train_end"]).to_numpy() & time.lt(fold["valid_end"]).to_numpy()
        test_period = time.ge(fold["valid_end"]).to_numpy() & time.lt(fold["test_end"]).to_numpy()
        eligible_train = base_eligible & train_period
        regime_mean = np.mean(raw_regime[eligible_train], axis=0)
        regime_scale = np.std(raw_regime[eligible_train], axis=0)
        regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
        regime = (raw_regime - regime_mean) / regime_scale
        fold_provenance = {}

        for target_index, target_name in enumerate(args.targets):
            target = data[target_name].to_numpy(dtype=float)
            usable = base_eligible & np.isfinite(target)
            fitted = fit_window(
                features,
                target,
                turbine,
                usable,
                train_period,
                valid_period,
                test_period,
            )
            locked_rows = reference.loc[
                reference["fold"].eq(fold["fold"])
                & reference["target"].eq(target_name)
                & reference["method"].eq("static_median")
            ]
            if len(locked_rows) != 1:
                raise ValueError(f"Expected one locked static row for {fold['fold']} {target_name}")
            locked = locked_rows.iloc[0]
            if not np.isclose(fitted["alpha"], locked["normal_model_alpha"], rtol=0.0, atol=1e-12):
                raise AssertionError(
                    f"Normal-model alpha mismatch for {fold['fold']} {target_name}: "
                    f"{fitted['alpha']} vs {locked['normal_model_alpha']}"
                )

            validation_residual = target[fitted["valid_mask"]] - predict(
                fitted["model"], features[fitted["valid_mask"]]
            )
            test_prediction = predict(fitted["model"], features[fitted["test_mask"]])
            test_residual = target[fitted["test_mask"]] - test_prediction
            validation_frame = residual_frame(data, validation_residual, fitted["valid_mask"], regime)
            test_frame = residual_frame(data, test_residual, fitted["test_mask"], regime)

            candidate_grid = validation_bandwidth_grid(
                validation_frame, target_name, fitted["residual_scales"]
            )
            for row in candidate_grid:
                grid_rows.append({"protocol": args.protocol, "fold": fold["fold"], **row})
            selected = pd.DataFrame(candidate_grid).sort_values(
                ["specific_median_abs", "specific_std", "distance_bandwidth"],
                kind="stable",
            ).iloc[0]
            selected_bandwidth = float(selected["distance_bandwidth"])
            scores = factorial_scores(
                test_frame,
                target_name,
                fitted["residual_scales"],
                selected_bandwidth,
            )
            locked_static_scores = peer_reference_scores(
                test_frame,
                target_name,
                "static_median",
                fitted["residual_scales"],
            )
            locked_static_recomputed = float(
                np.std(locked_static_scores["specific_residual"].to_numpy(dtype=float))
            )
            common_support_rows = int(scores["method"].eq("static_no_trim").sum())
            locked_support_rows = int(len(locked_static_scores))
            common_support_retained_pct = float(
                100.0 * common_support_rows / locked_support_rows
            )
            method_summaries = {}
            for method in METHODS:
                summary = summarize_method(scores.loc[scores["method"].eq(method)])
                method_summaries[method] = summary
                method_rows.append(
                    {
                        "protocol": args.protocol,
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
                        "selected_distance_bandwidth": selected_bandwidth,
                        "bandwidth_boundary": selected_bandwidth
                        in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                        "locked_static_score_rows": locked_support_rows,
                        "common_support_retained_pct": common_support_retained_pct,
                        **summary,
                    }
                )

            if not np.isclose(
                locked_static_recomputed,
                locked["specific_std"],
                rtol=1e-10,
                atol=1e-10,
            ):
                raise AssertionError(
                    f"Locked static score mismatch for {fold['fold']} {target_name}: "
                    f"{locked_static_recomputed} vs {locked['specific_std']}"
                )

            for contrast_index, (contrast, numerator, denominator) in enumerate(CONTRASTS):
                for block_days in sorted(set(args.block_days)):
                    result = moving_block_contrast(
                        scores,
                        numerator,
                        denominator,
                        block_days,
                        args.bootstrap_iterations,
                        seed=(
                            20260901
                            + 10000 * fold_index
                            + 1000 * target_index
                            + 100 * contrast_index
                            + block_days
                        ),
                    )
                    contrast_rows.append(
                        {
                            "protocol": args.protocol,
                            "fold": fold["fold"],
                            "target": target_name,
                            "contrast": contrast,
                            "numerator_method": numerator,
                            "denominator_method": denominator,
                            **result,
                        }
                    )
            fold_provenance[target_name] = {
                "normal_model_alpha": fitted["alpha"],
                "selected_distance_bandwidth": selected_bandwidth,
                "bandwidth_boundary": selected_bandwidth
                in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                "locked_static_std_reproduced": True,
                "locked_static_score_rows": locked_support_rows,
                "common_support_rows": common_support_rows,
                "common_support_retained_pct": common_support_retained_pct,
            }
            print(
                f"[{fold['fold']}][{target_name}] alpha={fitted['alpha']:g}; "
                f"all-peer bw={selected_bandwidth:g}; "
                f"distance-vs-trim={(method_summaries['distance_trim']['specific_std'] / method_summaries['static_trim']['specific_std'] - 1.0) * 100.0:.2f}%",
                flush=True,
            )
        model_provenance[fold["fold"]] = fold_provenance

    grid = pd.DataFrame(grid_rows)
    methods = pd.DataFrame(method_rows)
    contrasts = pd.DataFrame(contrast_rows)
    primary_contrasts = contrasts.loc[contrasts["block_days"].eq(7)].copy()
    contrast_summary = primary_contrasts.groupby(["target", "contrast"], as_index=False).agg(
        folds=("fold", "size"),
        folds_lower_spread=("observed_std_change_pct", lambda values: int((values < 0.0).sum())),
        folds_ci_below_zero=("ci95_high_pct", lambda values: int((values < 0.0).sum())),
        folds_ci_above_zero=("ci95_low_pct", lambda values: int((values > 0.0).sum())),
        median_std_change_pct=("observed_std_change_pct", "median"),
        min_std_change_pct=("observed_std_change_pct", "min"),
        max_std_change_pct=("observed_std_change_pct", "max"),
    )
    method_wide = methods.pivot_table(
        index=["protocol", "fold", "target"], columns="method", values="specific_std"
    ).reset_index()
    method_summary_rows = []
    for method in METHODS:
        change = (method_wide[method] / method_wide["static_no_trim"] - 1.0) * 100.0
        temporary = method_wide[["target"]].copy()
        temporary["method"] = method
        temporary["change_vs_static_no_trim_pct"] = change
        method_summary_rows.append(temporary)
    method_summary = (
        pd.concat(method_summary_rows, ignore_index=True)
        .groupby(["target", "method"], as_index=False)
        .agg(
            folds=("change_vs_static_no_trim_pct", "size"),
            folds_lower_than_static=(
                "change_vs_static_no_trim_pct",
                lambda values: int((values < 0.0).sum()),
            ),
            median_vs_static_no_trim_pct=("change_vs_static_no_trim_pct", "median"),
            min_vs_static_no_trim_pct=("change_vs_static_no_trim_pct", "min"),
            max_vs_static_no_trim_pct=("change_vs_static_no_trim_pct", "max"),
        )
    )

    grid.to_csv(args.output_dir / "validation_bandwidth_grid.csv", index=False)
    methods.to_csv(args.output_dir / "fold_target_method_results.csv", index=False)
    contrasts.to_csv(args.output_dir / "mechanism_contrasts.csv", index=False)
    contrast_summary.to_csv(args.output_dir / "protocol_target_contrast_summary.csv", index=False)
    method_summary.to_csv(args.output_dir / "protocol_target_method_summary.csv", index=False)

    evidence_label = (
        "primary post-gate independent-meteorology mechanism audit"
        if args.protocol == "independent"
        else "turbine-local ambient-proxy sensitivity audit"
    )
    report = [
        f"# All-peer trim–distance mechanism audit: {args.protocol}",
        "",
        f"This is a frozen post-hoc {evidence_label}. It uses the same all-peer leave-one-out candidate set and identical trimming support across variants. It is not fault-event validation or external confirmation.",
        "",
        "## Seven-day block-bootstrap contrast summary",
        "",
        markdown_table(contrast_summary),
        "",
        "The `distance` contrast isolates Gaussian regime-distance weighting relative to an equally trimmed, unweighted all-peer reference. The `trim` and `reliability` contrasts isolate the other two factors. The end-to-end contrast combines all factors.",
        "",
        "## Methods relative to the untrimmed static reference",
        "",
        markdown_table(method_summary),
        "",
        "## Selected all-peer bandwidths and diagnostics",
        "",
        markdown_table(
            methods.loc[methods["method"].eq("distance_trim")],
            [
                "fold",
                "target",
                "selected_distance_bandwidth",
                "bandwidth_boundary",
                "score_rows",
                "locked_static_score_rows",
                "common_support_retained_pct",
                "specific_std",
                "mean_retained_peers",
                "median_effective_peers",
                "p05_effective_peers",
                "mean_trim_fraction",
            ],
        ),
        "",
        "Every recomputed untrimmed static score matches the locked rolling artifact. Three-, seven-, and fourteen-day block results are retained in `mechanism_contrasts.csv`. Negative changes indicate lower residual-state spread only.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    output_names = [
        "MECHANISM_AUDIT.md",
        "validation_bandwidth_grid.csv",
        "fold_target_method_results.csv",
        "mechanism_contrasts.csv",
        "protocol_target_contrast_summary.csv",
        "protocol_target_method_summary.csv",
    ]
    provenance = {
        "artifact": f"health_twin_trim_distance_ablation_{args.protocol}_v1",
        "confidentiality": "local-only confidential aggregate research result",
        "status": "frozen post-hoc mechanism audit; not confirmatory",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "reference_dir": str(args.reference_dir),
        "reference_results_sha256": sha256_file(reference_path),
        "protocol": args.protocol,
        "ambient_source": ambient_source,
        "ambient_column": ambient_column,
        "targets": args.targets,
        "folds": fold_definitions,
        "normal_model_features": feature_names,
        "regime_distance_features": regime_names,
        "methods": METHODS,
        "contrasts": [
            {"contrast": contrast, "numerator": numerator, "denominator": denominator}
            for contrast, numerator, denominator in CONTRASTS
        ],
        "candidate_policy": "all eligible leave-one-out peers for every method",
        "common_support_rule": "if fewer than three peers remain after the common trim, omit the turbine-time row from every method",
        "trim_rule": {
            "centre": "peer residual median",
            "scale": "max(1.4826*MAD, median peer training residual scale, 0.25)",
            "multiplier": TRIM_MULTIPLIER,
        },
        "bandwidth_grid": BANDWIDTH_GRID,
        "bandwidth_selection": "validation distance_trim: minimum median absolute state, then standard deviation, then narrower bandwidth",
        "block_bootstrap": {
            "block_days": sorted(set(args.block_days)),
            "primary_block_days": 7,
            "iterations": args.bootstrap_iterations,
        },
        "model_provenance": model_provenance,
        "output_files_sha256": {
            name: sha256_file(args.output_dir / name) for name in output_names
        },
        "claim_boundary": "out-of-sample residual-state stability only; no fault accuracy, diagnosis, prognosis, or maintenance claim",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(native(provenance), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
