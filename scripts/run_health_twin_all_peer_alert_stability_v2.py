#!/usr/bin/env python3
"""Run the frozen post-hoc all-peer alert-alignment audit locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_alert_stability_v1 import (
    CALIBRATION_END,
    MONITORING_START,
    TEST_END,
    TRAIN_END,
    TUNING_END,
    TARGET_ROLES,
    apply_common_thresholds,
    apply_specific_thresholds,
    calibrate_common,
    calibrate_specific,
    common_states,
    markdown_table,
    monthly_summary,
    turbine_summary,
)
from run_health_twin_baseline import (
    AMBIENT_SOURCES,
    SOURCE_COLUMNS,
    TARGETS,
    make_feature_matrix,
    predict,
)
from run_health_twin_regime_peer_v2 import regime_matrix, residual_frame
from run_health_twin_rolling_ablation_v1 import ALPHA_GRID, BANDWIDTH_GRID, fit_window
from run_health_twin_trim_distance_ablation import (
    factorial_scores,
    validation_bandwidth_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD = "distance_trim"
THRESHOLD_QUANTILE = 0.995
PERSISTENCE_LENGTH = 3


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


def target_summary_table(turbine: pd.DataFrame, common_test: pd.DataFrame) -> pd.DataFrame:
    common_summary = common_test.groupby("target", as_index=False).agg(
        common_test_rows=("point_flag", "size"),
        common_point_rate_pct=("point_flag", lambda values: float(values.mean() * 100.0)),
        common_persistent_rate_pct=(
            "persistent_flag", lambda values: float(values.mean() * 100.0)
        ),
        common_episodes=("episode_start", "sum"),
    )
    summary = turbine.groupby(["target", "role"], as_index=False).agg(
        turbines=("turbine_id", "nunique"),
        median_calibration_point_rate_pct=("calibration_point_rate_pct", "median"),
        median_test_point_rate_pct=("test_point_rate_pct", "median"),
        q25_test_point_rate_pct=("test_point_rate_pct", lambda values: float(values.quantile(0.25))),
        q75_test_point_rate_pct=("test_point_rate_pct", lambda values: float(values.quantile(0.75))),
        max_test_point_rate_pct=("test_point_rate_pct", "max"),
        median_test_persistent_rate_pct=("test_persistent_rate_pct", "median"),
        total_test_episodes=("test_episodes", "sum"),
        median_episodes_per_30_days=("episodes_per_30_days", "median"),
        zero_episode_turbine_fraction=("test_episodes", lambda values: float((values == 0).mean())),
    ).merge(common_summary, on="target", how="left", validate="one_to_one")
    summary["median_point_rate_warning_gt5"] = summary[
        "median_test_point_rate_pct"
    ].gt(5.0)
    summary["common_point_rate_warning_gt5"] = summary[
        "common_point_rate_pct"
    ].gt(5.0)
    summary["turbines_point_rate_gt10"] = summary["target"].map(
        turbine.groupby("target")["test_point_rate_pct"].apply(
            lambda values: int((values > 10.0).sum())
        )
    )
    return summary


def compare_with_nearest(current: pd.DataFrame, comparison_dir: Path) -> pd.DataFrame:
    old_path = comparison_dir / "alert_target_summary.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    metrics = [
        "median_test_point_rate_pct",
        "max_test_point_rate_pct",
        "median_test_persistent_rate_pct",
        "common_point_rate_pct",
        "common_persistent_rate_pct",
    ]
    left = current[["target", "role", *metrics]].rename(
        columns={metric: f"all_peer_{metric}" for metric in metrics}
    )
    right = old[["target", *metrics]].rename(
        columns={metric: f"nearest_k_{metric}" for metric in metrics}
    )
    comparison = left.merge(right, on="target", how="inner", validate="one_to_one")
    for metric in metrics:
        comparison[f"all_peer_minus_nearest_k_{metric}"] = (
            comparison[f"all_peer_{metric}"] - comparison[f"nearest_k_{metric}"]
        )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/research_layers/health_twin_base_v1.csv.gz",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=PROJECT_ROOT / "data/experiment_results/health_twin_alert_stability_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "ALL_PEER_ALERT_ALIGNMENT_RESULTS.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; use --overwrite to replace it")

    ambient_column = AMBIENT_SOURCES["independent_meteorology"]["column"]
    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    turbine_id = data["turbine_id"].astype(str).to_numpy()
    features, feature_names = make_feature_matrix(data, turbine_ids, ambient_column)
    descriptor, descriptor_names = regime_matrix(data, ambient_column)
    base_eligible = (
        data["timestamp"].ge(MONITORING_START)
        & data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(descriptor).all(axis=1)
    ).to_numpy()
    train_period = data["timestamp"].ge(MONITORING_START).to_numpy() & data[
        "timestamp"
    ].lt(TRAIN_END).to_numpy()
    tuning_period = data["timestamp"].ge(TRAIN_END).to_numpy() & data[
        "timestamp"
    ].lt(TUNING_END).to_numpy()
    calibration_period = data["timestamp"].ge(TUNING_END).to_numpy() & data[
        "timestamp"
    ].lt(CALIBRATION_END).to_numpy()
    test_period = data["timestamp"].ge(CALIBRATION_END).to_numpy() & data[
        "timestamp"
    ].lt(TEST_END).to_numpy()

    regime_mean = np.mean(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.std(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
    descriptor = (descriptor - regime_mean) / regime_scale
    training_ambient = data.loc[base_eligible & train_period, ambient_column]
    ambient_min = float(training_ambient.min())
    ambient_max = float(training_ambient.max())

    grid_rows = []
    selected_rows = []
    model_provenance = {}
    calibration_scores = []
    test_scores = []
    test_residual_frames = []
    for target in TARGETS:
        target_values = data[target].to_numpy(dtype=float)
        usable = base_eligible & np.isfinite(target_values)
        fitted = fit_window(
            features,
            target_values,
            turbine_id,
            usable,
            train_period,
            tuning_period,
            calibration_period,
        )
        tuning_mask = fitted["valid_mask"]
        calibration_mask = usable & calibration_period
        target_test_mask = usable & test_period
        tuning_residual = target_values[tuning_mask] - predict(
            fitted["model"], features[tuning_mask]
        )
        calibration_residual = target_values[calibration_mask] - predict(
            fitted["model"], features[calibration_mask]
        )
        test_residual = target_values[target_test_mask] - predict(
            fitted["model"], features[target_test_mask]
        )
        tuning_frame = residual_frame(data, tuning_residual, tuning_mask, descriptor)
        calibration_frame = residual_frame(
            data, calibration_residual, calibration_mask, descriptor
        )
        test_frame = residual_frame(data, test_residual, target_test_mask, descriptor)
        test_residual_frames.append(test_frame.assign(target=target))

        target_grid = validation_bandwidth_grid(
            tuning_frame, target, fitted["residual_scales"]
        )
        grid_rows.extend(target_grid)
        selected = pd.DataFrame(target_grid).sort_values(
            ["specific_median_abs", "specific_std", "distance_bandwidth"],
            kind="stable",
        ).iloc[0]
        bandwidth = float(selected["distance_bandwidth"])

        calibration_all = factorial_scores(
            calibration_frame, target, fitted["residual_scales"], bandwidth
        )
        test_all = factorial_scores(
            test_frame, target, fitted["residual_scales"], bandwidth
        )
        calibration_score = calibration_all.loc[
            calibration_all["method"].eq(METHOD)
        ].reset_index(drop=True)
        test_score = test_all.loc[test_all["method"].eq(METHOD)].reset_index(drop=True)
        calibration_scores.append(calibration_score)
        test_scores.append(test_score)
        test_ambient = data.loc[target_test_mask, ambient_column]
        selected_rows.append(
            {
                "target": target,
                "role": TARGET_ROLES.get(target, "other"),
                "normal_model_alpha": fitted["alpha"],
                "alpha_boundary": fitted["alpha_boundary"],
                "selected_distance_bandwidth": bandwidth,
                "bandwidth_boundary": bandwidth
                in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                "tuning_score_rows": int(selected["score_rows"]),
                "tuning_median_effective_peers": float(
                    selected["median_effective_peer_count"]
                ),
                "tuning_mean_trim_fraction": float(selected["mean_trim_fraction"]),
                "calibration_score_rows": int(len(calibration_score)),
                "test_score_rows": int(len(test_score)),
                "test_median_effective_peers": float(
                    test_score["effective_peer_count"].median()
                ),
                "test_p05_effective_peers": float(
                    test_score["effective_peer_count"].quantile(0.05)
                ),
                "test_mean_trim_fraction": float(test_score["trim_fraction"].mean()),
                "test_ambient_outside_training_minmax_pct": float(
                    (~test_ambient.between(ambient_min, ambient_max)).mean() * 100.0
                ),
            }
        )
        model_provenance[target] = {
            "normal_model_alpha": fitted["alpha"],
            "residual_scales_by_turbine": fitted["residual_scales"],
            "selected_distance_bandwidth": bandwidth,
        }
        print(
            f"[{target}] alpha={fitted['alpha']:g}; all-peer bw={bandwidth:g}; "
            f"test rows={len(test_score)}",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    selected = pd.DataFrame(selected_rows)
    calibration_scores = pd.concat(calibration_scores, ignore_index=True)
    test_scores = pd.concat(test_scores, ignore_index=True)
    test_residual_frames = pd.concat(test_residual_frames, ignore_index=True)
    thresholds = calibrate_specific(calibration_scores)
    if not bool(thresholds["estimable"].all()):
        raise ValueError("At least one turbine-target threshold has fewer than 200 rows.")
    calibration_alerts = apply_specific_thresholds(calibration_scores, thresholds)
    test_alerts = apply_specific_thresholds(test_scores, thresholds)
    common_calibration = common_states(calibration_scores)
    common_test = common_states(test_scores)
    common_thresholds = calibrate_common(common_calibration)
    common_calibration_alerts = apply_common_thresholds(
        common_calibration, common_thresholds
    )
    common_test_alerts = apply_common_thresholds(common_test, common_thresholds)

    turbine = turbine_summary(calibration_alerts, test_alerts)
    month = monthly_summary(test_alerts)
    target_summary = target_summary_table(turbine, common_test_alerts)
    selected = selected.merge(
        thresholds.groupby("target", as_index=False).agg(
            min_calibration_rows=("calibration_rows", "min"),
            median_threshold=("absolute_deviation_threshold", "median"),
        ),
        on="target",
        how="left",
        validate="one_to_one",
    )
    comparison = compare_with_nearest(target_summary, args.comparison_dir)

    grid.to_csv(args.output_dir / "tuning_grid.csv", index=False)
    selected.to_csv(args.output_dir / "selected_configuration.csv", index=False)
    thresholds.to_csv(args.output_dir / "specific_thresholds.csv", index=False)
    common_thresholds.to_csv(args.output_dir / "common_thresholds.csv", index=False)
    turbine.to_csv(args.output_dir / "alert_turbine_summary.csv", index=False)
    target_summary.to_csv(args.output_dir / "alert_target_summary.csv", index=False)
    month.to_csv(args.output_dir / "alert_month_summary.csv", index=False)
    comparison.to_csv(args.output_dir / "alert_target_comparison.csv", index=False)
    gzip_options = {"method": "gzip", "compresslevel": 1, "mtime": 0}
    calibration_alerts.to_csv(
        args.output_dir / "calibration_alert_states.csv.gz",
        index=False,
        compression=gzip_options,
    )
    test_alerts.to_csv(
        args.output_dir / "test_alert_states.csv.gz", index=False, compression=gzip_options
    )
    test_residual_frames.to_csv(
        args.output_dir / "test_residual_frames.csv.gz",
        index=False,
        compression=gzip_options,
    )
    common_calibration_alerts.to_csv(
        args.output_dir / "calibration_common_alert_states.csv.gz",
        index=False,
        compression=gzip_options,
    )
    common_test_alerts.to_csv(
        args.output_dir / "test_common_alert_states.csv.gz",
        index=False,
        compression=gzip_options,
    )

    report = [
        "# All-peer alert alignment — confidential local artifact",
        "",
        "## Scope",
        "",
        "This frozen post-hoc alignment replaces the earlier nearest-K state estimator with the clean all-peer Gaussian distance estimator while preserving the original train--tune--calibrate--test chronology, 99.5th-percentile rule, and three-interval persistence. It is an unlabeled development comparison, not fault-event validation.",
        "",
        "## Selected all-peer configurations and support",
        "",
        markdown_table(
            selected,
            [
                "target",
                "role",
                "normal_model_alpha",
                "selected_distance_bandwidth",
                "min_calibration_rows",
                "test_median_effective_peers",
                "test_p05_effective_peers",
                "test_mean_trim_fraction",
                "test_ambient_outside_training_minmax_pct",
            ],
        ),
        "",
        "## All-peer test stability summary",
        "",
        markdown_table(
            target_summary,
            [
                "target",
                "role",
                "median_calibration_point_rate_pct",
                "median_test_point_rate_pct",
                "q25_test_point_rate_pct",
                "q75_test_point_rate_pct",
                "max_test_point_rate_pct",
                "median_test_persistent_rate_pct",
                "total_test_episodes",
                "common_point_rate_pct",
                "common_persistent_rate_pct",
                "turbines_point_rate_gt10",
            ],
        ),
        "",
        "## Difference from preserved nearest-K alert artifact",
        "",
        markdown_table(
            comparison,
            [
                "target",
                "role",
                "all_peer_median_test_point_rate_pct",
                "nearest_k_median_test_point_rate_pct",
                "all_peer_minus_nearest_k_median_test_point_rate_pct",
                "all_peer_max_test_point_rate_pct",
                "nearest_k_max_test_point_rate_pct",
                "all_peer_common_point_rate_pct",
                "nearest_k_common_point_rate_pct",
            ],
        )
        if not comparison.empty
        else "No comparison artifact was available.",
        "",
        "## Interpretation boundary",
        "",
        "These rates characterize a frozen label-free state and its sensitivity to estimator alignment and temporal shift. They are not false-alarm, sensitivity, specificity, diagnosis, or operational-maintenance metrics. All site-year periods remain development data.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    output_names = [
        "tuning_grid.csv",
        "selected_configuration.csv",
        "specific_thresholds.csv",
        "common_thresholds.csv",
        "alert_turbine_summary.csv",
        "alert_target_summary.csv",
        "alert_month_summary.csv",
        "alert_target_comparison.csv",
        "calibration_alert_states.csv.gz",
        "test_alert_states.csv.gz",
        "test_residual_frames.csv.gz",
        "calibration_common_alert_states.csv.gz",
        "test_common_alert_states.csv.gz",
        "ALL_PEER_ALERT_ALIGNMENT_RESULTS.md",
    ]
    provenance = {
        "artifact": "health_twin_all_peer_alert_stability_v2",
        "confidentiality": "local-only confidential alert alignment research",
        "pre_specification": "docs/HEALTH_TWIN_ALL_PEER_ALERT_ALIGNMENT_PLAN.md",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "comparison_dir": str(args.comparison_dir),
        "comparison_target_summary_sha256": sha256_file(
            args.comparison_dir / "alert_target_summary.csv"
        ),
        "ambient_source": "independent_meteorology",
        "ambient_column": ambient_column,
        "normal_model_features": feature_names,
        "regime_distance_features": descriptor_names,
        "splits": {
            "monitoring_start_inclusive": MONITORING_START,
            "train_end_exclusive": TRAIN_END,
            "tuning_end_exclusive": TUNING_END,
            "calibration_end_exclusive": CALIBRATION_END,
            "test_end_exclusive": TEST_END,
        },
        "method": METHOD,
        "candidate_peers": "all eligible leave-one-out peers",
        "targets": TARGETS,
        "model_provenance": model_provenance,
        "alpha_grid": ALPHA_GRID,
        "bandwidth_grid": BANDWIDTH_GRID,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "persistence_length": PERSISTENCE_LENGTH,
        "selection_or_fitting_on_test": "none",
        "outputs": {name: sha256_file(args.output_dir / name) for name in output_names},
        "limits": [
            "Post-hoc estimator-alignment audit, not external confirmation.",
            "No fault, alarm, maintenance, or availability labels are present.",
            "Test rates characterize development stability, not accuracy.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(native(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
