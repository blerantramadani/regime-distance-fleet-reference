#!/usr/bin/env python3
"""Run the pre-specified validation-only alert stability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_baseline import (
    AMBIENT_SOURCES,
    SOURCE_COLUMNS,
    TARGETS,
    make_feature_matrix,
    predict,
)
from run_health_twin_regime_peer_v2 import (
    peer_reference_scores,
    regime_matrix,
    residual_frame,
    weighted_median,
)
from run_health_twin_rolling_ablation_v1 import (
    ALPHA_GRID,
    BANDWIDTH_GRID,
    PEER_GRID,
    fit_window,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITORING_START = pd.Timestamp("2015-05-14 12:00:00")
TRAIN_END = pd.Timestamp("2015-09-01 00:00:00")
TUNING_END = pd.Timestamp("2015-10-01 00:00:00")
CALIBRATION_END = pd.Timestamp("2015-11-01 00:00:00")
TEST_END = pd.Timestamp("2016-01-01 00:00:00")
THRESHOLD_QUANTILE = 0.995
MIN_CALIBRATION_ROWS = 200
MAX_CONTIGUOUS_GAP = pd.Timedelta(minutes=20)
PERSISTENCE_LENGTH = 3
METHOD = "nearest_distance"
TARGET_ROLES = {
    "gear_oil_temp_mean": "predeclared primary",
    "gen_bearing_g_temp_mean": "predeclared primary",
    "gen_bearing_r_temp_mean": "predeclared primary",
    "main_bearing_temp_mean": "predeclared exploratory",
    "hydraulic_oil_temp_mean": "predeclared negative/control",
}


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


def markdown_table(frame, columns):
    selected = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def distance_validation_grid(frame, target, residual_scales):
    values_by_configuration = {
        (peer_count, bandwidth): []
        for peer_count in PEER_GRID
        for bandwidth in BANDWIDTH_GRID
    }
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
                selected_distances = ordered_distances[:selected_count]
                peer_values = residual[selected]
                centre = float(np.median(peer_values))
                robust_scale = max(
                    1.4826 * float(np.median(np.abs(peer_values - centre))),
                    float(np.median([residual_scales[turbine[peer]] for peer in selected])),
                    0.25,
                )
                keep = np.abs(peer_values - centre) <= 4.5 * robust_scale
                peer_values = peer_values[keep]
                selected_distances = selected_distances[keep]
                for bandwidth in BANDWIDTH_GRID:
                    log_kernel = -0.5 * (selected_distances / bandwidth) ** 2
                    log_kernel -= np.max(log_kernel)
                    reference = weighted_median(peer_values, np.exp(log_kernel))
                    values_by_configuration[(peer_count, bandwidth)].append(
                        float(residual[index] - reference)
                    )
    rows = []
    for peer_count in PEER_GRID:
        for bandwidth in BANDWIDTH_GRID:
            values = np.asarray(values_by_configuration[(peer_count, bandwidth)], dtype=float)
            rows.append(
                {
                    "target": target,
                    "peer_count": peer_count,
                    "distance_bandwidth": bandwidth,
                    "tuning_score_rows": int(len(values)),
                    "specific_median_abs": float(np.median(np.abs(values))),
                    "specific_std": float(np.std(values)),
                    "specific_p95_abs": float(np.quantile(np.abs(values), 0.95)),
                }
            )
    return rows


def add_persistence(frame, group_columns):
    output = []
    order_columns = [*group_columns, "timestamp"]
    for _, group in frame.sort_values(order_columns, kind="stable").groupby(
        group_columns, sort=False, dropna=False
    ):
        group = group.copy()
        run = 0
        previous_time = None
        previous_persistent = False
        persistent_flags = []
        episode_starts = []
        for timestamp, point_flag in zip(group["timestamp"], group["point_flag"]):
            contiguous = previous_time is not None and timestamp - previous_time <= MAX_CONTIGUOUS_GAP
            if bool(point_flag):
                run = run + 1 if contiguous else 1
            else:
                run = 0
            persistent = run >= PERSISTENCE_LENGTH
            episode_start = bool(persistent and not previous_persistent)
            persistent_flags.append(persistent)
            episode_starts.append(episode_start)
            previous_persistent = persistent
            previous_time = timestamp
        group["persistent_flag"] = persistent_flags
        group["episode_start"] = episode_starts
        output.append(group)
    return pd.concat(output, ignore_index=True) if output else frame.copy()


def calibrate_specific(calibration):
    rows = []
    for (target, turbine), group in calibration.groupby(["target", "turbine_id"], sort=True):
        state = group["specific_residual"].to_numpy(dtype=float)
        centre = float(np.median(state))
        deviation = np.abs(state - centre)
        estimable = len(state) >= MIN_CALIBRATION_ROWS
        rows.append(
            {
                "target": target,
                "turbine_id": turbine,
                "calibration_rows": int(len(state)),
                "state_centre": centre,
                "absolute_deviation_threshold": (
                    float(np.quantile(deviation, THRESHOLD_QUANTILE)) if estimable else np.nan
                ),
                "estimable": estimable,
            }
        )
    return pd.DataFrame(rows)


def apply_specific_thresholds(states, thresholds):
    scored = states.merge(
        thresholds,
        on=["target", "turbine_id"],
        how="left",
        validate="many_to_one",
    )
    scored["centred_absolute_state"] = (
        scored["specific_residual"] - scored["state_centre"]
    ).abs()
    scored["point_flag"] = (
        scored["estimable"]
        & scored["centred_absolute_state"].gt(scored["absolute_deviation_threshold"])
    )
    return add_persistence(scored, ["target", "turbine_id"])


def common_states(scores):
    return (
        scores.groupby(["target", "timestamp"], as_index=False)
        .agg(available_turbines=("turbine_id", "nunique"), common_state=("residual", "median"))
    )


def calibrate_common(calibration):
    rows = []
    for target, group in calibration.groupby("target", sort=True):
        state = group["common_state"].to_numpy(dtype=float)
        centre = float(np.median(state))
        deviation = np.abs(state - centre)
        estimable = len(state) >= MIN_CALIBRATION_ROWS
        rows.append(
            {
                "target": target,
                "calibration_rows": int(len(state)),
                "common_centre": centre,
                "common_absolute_deviation_threshold": (
                    float(np.quantile(deviation, THRESHOLD_QUANTILE)) if estimable else np.nan
                ),
                "estimable": estimable,
            }
        )
    return pd.DataFrame(rows)


def apply_common_thresholds(states, thresholds):
    scored = states.merge(thresholds, on="target", how="left", validate="many_to_one")
    scored["centred_absolute_state"] = (scored["common_state"] - scored["common_centre"]).abs()
    scored["point_flag"] = (
        scored["estimable"]
        & scored["centred_absolute_state"].gt(
            scored["common_absolute_deviation_threshold"]
        )
    )
    return add_persistence(scored, ["target"])


def turbine_summary(calibration, test):
    calibration_rate = calibration.groupby(["target", "turbine_id"])["point_flag"].mean()
    rows = []
    for (target, turbine), group in test.groupby(["target", "turbine_id"], sort=True):
        rows.append(
            {
                "target": target,
                "role": TARGET_ROLES.get(target, "other"),
                "turbine_id": turbine,
                "calibration_point_rate_pct": float(calibration_rate.loc[(target, turbine)] * 100.0),
                "test_rows": int(len(group)),
                "test_point_rate_pct": float(group["point_flag"].mean() * 100.0),
                "test_persistent_rate_pct": float(group["persistent_flag"].mean() * 100.0),
                "test_episodes": int(group["episode_start"].sum()),
                "episodes_per_30_days": float(group["episode_start"].sum() / 61.0 * 30.0),
            }
        )
    return pd.DataFrame(rows)


def monthly_summary(test):
    frame = test.copy()
    frame["month"] = frame["timestamp"].dt.to_period("M").astype(str)
    return frame.groupby(["target", "month"], as_index=False).agg(
        scored_rows=("point_flag", "size"),
        turbines=("turbine_id", "nunique"),
        point_rate_pct=("point_flag", lambda x: float(x.mean() * 100.0)),
        persistent_rate_pct=("persistent_flag", lambda x: float(x.mean() * 100.0)),
        episodes=("episode_start", "sum"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "ALERT_STABILITY_RESULTS.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; use --overwrite to replace it")

    ambient_specification = AMBIENT_SOURCES["independent_meteorology"]
    ambient_column = ambient_specification["column"]
    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    turbine = data["turbine_id"].astype(str).to_numpy()
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
    tuning_period = data["timestamp"].ge(TRAIN_END).to_numpy() & data["timestamp"].lt(
        TUNING_END
    ).to_numpy()
    calibration_period = data["timestamp"].ge(TUNING_END).to_numpy() & data[
        "timestamp"
    ].lt(CALIBRATION_END).to_numpy()
    test_period = data["timestamp"].ge(CALIBRATION_END).to_numpy() & data["timestamp"].lt(
        TEST_END
    ).to_numpy()
    regime_mean = np.mean(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.std(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
    descriptor = (descriptor - regime_mean) / regime_scale
    training_ambient = data.loc[base_eligible & train_period, ambient_column]
    ambient_min, ambient_max = float(training_ambient.min()), float(training_ambient.max())

    grid_rows = []
    selected_rows = []
    calibration_scores = []
    test_scores = []
    for target in TARGETS:
        target_values = data[target].to_numpy(dtype=float)
        usable = base_eligible & np.isfinite(target_values)
        fitted = fit_window(
            features,
            target_values,
            turbine,
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
        target_grid = distance_validation_grid(
            tuning_frame, target, fitted["residual_scales"]
        )
        grid_rows.extend(target_grid)
        selected = pd.DataFrame(target_grid).sort_values(
            ["specific_median_abs", "specific_std", "peer_count", "distance_bandwidth"],
            kind="stable",
        ).iloc[0]
        peer_count = int(selected["peer_count"])
        bandwidth = float(selected["distance_bandwidth"])
        calibration_score = peer_reference_scores(
            calibration_frame,
            target,
            METHOD,
            fitted["residual_scales"],
            peer_count=peer_count,
            bandwidth=bandwidth,
        )
        test_score = peer_reference_scores(
            test_frame,
            target,
            METHOD,
            fitted["residual_scales"],
            peer_count=peer_count,
            bandwidth=bandwidth,
        )
        calibration_scores.append(calibration_score)
        test_scores.append(test_score)
        test_ambient = data.loc[target_test_mask, ambient_column]
        selected_rows.append(
            {
                "target": target,
                "role": TARGET_ROLES.get(target, "other"),
                "normal_model_alpha": fitted["alpha"],
                "alpha_boundary": fitted["alpha_boundary"],
                "selected_peer_count": peer_count,
                "peer_count_boundary": peer_count in {min(PEER_GRID), max(PEER_GRID)},
                "selected_distance_bandwidth": bandwidth,
                "bandwidth_boundary": bandwidth
                in {min(BANDWIDTH_GRID), max(BANDWIDTH_GRID)},
                "test_ambient_outside_training_minmax_pct": float(
                    (~test_ambient.between(ambient_min, ambient_max)).mean() * 100.0
                ),
            }
        )
        print(
            f"[{target}] alpha={fitted['alpha']:g}; peers={peer_count}; bw={bandwidth:g}",
            flush=True,
        )

    grid = pd.DataFrame(grid_rows)
    selected = pd.DataFrame(selected_rows)
    calibration_scores = pd.concat(calibration_scores, ignore_index=True)
    test_scores = pd.concat(test_scores, ignore_index=True)
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
    common_summary = common_test_alerts.groupby("target", as_index=False).agg(
        common_test_rows=("point_flag", "size"),
        common_point_rate_pct=("point_flag", lambda x: float(x.mean() * 100.0)),
        common_persistent_rate_pct=("persistent_flag", lambda x: float(x.mean() * 100.0)),
        common_episodes=("episode_start", "sum"),
    )
    target_summary = turbine.groupby(["target", "role"], as_index=False).agg(
        turbines=("turbine_id", "nunique"),
        median_calibration_point_rate_pct=("calibration_point_rate_pct", "median"),
        median_test_point_rate_pct=("test_point_rate_pct", "median"),
        q25_test_point_rate_pct=("test_point_rate_pct", lambda x: float(x.quantile(0.25))),
        q75_test_point_rate_pct=("test_point_rate_pct", lambda x: float(x.quantile(0.75))),
        max_test_point_rate_pct=("test_point_rate_pct", "max"),
        median_test_persistent_rate_pct=("test_persistent_rate_pct", "median"),
        total_test_episodes=("test_episodes", "sum"),
        median_episodes_per_30_days=("episodes_per_30_days", "median"),
        zero_episode_turbine_fraction=("test_episodes", lambda x: float((x == 0).mean())),
    ).merge(common_summary, on="target", how="left", validate="one_to_one")
    target_summary["median_point_rate_warning_gt5"] = target_summary[
        "median_test_point_rate_pct"
    ].gt(5.0)
    target_summary["common_point_rate_warning_gt5"] = target_summary[
        "common_point_rate_pct"
    ].gt(5.0)
    target_summary["turbines_point_rate_gt10"] = target_summary["target"].map(
        turbine.groupby("target")["test_point_rate_pct"].apply(lambda x: int((x > 10.0).sum()))
    )
    selected = selected.merge(
        thresholds.groupby("target", as_index=False).agg(
            min_calibration_rows=("calibration_rows", "min"),
            median_threshold=("absolute_deviation_threshold", "median"),
        ),
        on="target",
        how="left",
        validate="one_to_one",
    )

    grid.to_csv(args.output_dir / "tuning_grid.csv", index=False)
    selected.to_csv(args.output_dir / "selected_configuration.csv", index=False)
    thresholds.to_csv(args.output_dir / "specific_thresholds.csv", index=False)
    common_thresholds.to_csv(args.output_dir / "common_thresholds.csv", index=False)
    turbine.to_csv(args.output_dir / "alert_turbine_summary.csv", index=False)
    target_summary.to_csv(args.output_dir / "alert_target_summary.csv", index=False)
    month.to_csv(args.output_dir / "alert_month_summary.csv", index=False)
    calibration_alerts.to_csv(
        args.output_dir / "calibration_alert_states.csv.gz", index=False, compression="gzip"
    )
    test_alerts.to_csv(
        args.output_dir / "test_alert_states.csv.gz", index=False, compression="gzip"
    )
    common_calibration_alerts.to_csv(
        args.output_dir / "calibration_common_alert_states.csv.gz",
        index=False,
        compression="gzip",
    )
    common_test_alerts.to_csv(
        args.output_dir / "test_common_alert_states.csv.gz",
        index=False,
        compression="gzip",
    )

    report = [
        "# Validation-only alert stability — confidential local artifact",
        "",
        "## Scope",
        "",
        "The normal model is trained through August, model and distance-peer parameters are selected on September, 99.5th-percentile thresholds are calibrated on October, and November--December selects nothing. Three consecutive exceedances with no gap above 20 minutes define persistence. These are unlabeled stability rates, not fault-detection metrics.",
        "",
        "## Selected configurations and support",
        "",
        markdown_table(
            selected,
            [
                "target",
                "role",
                "normal_model_alpha",
                "selected_peer_count",
                "selected_distance_bandwidth",
                "min_calibration_rows",
                "test_ambient_outside_training_minmax_pct",
            ],
        ),
        "",
        "## Test stability summary",
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
                "median_episodes_per_30_days",
                "common_point_rate_pct",
                "common_persistent_rate_pct",
                "turbines_point_rate_gt10",
            ],
        ),
        "",
        "A higher test rate than the nominal October calibration tail indicates temporal distribution shift, normal-model mismatch, or an unlabeled deviation; without event records it cannot be assigned to a component fault. Fleet-common flags are reported separately for the same reason.",
        "",
        "## Interpretation boundary",
        "",
        "This artifact evaluates threshold estimability, later-period rate stability, persistence, cross-turbine dispersion, and common-mode activity. It does not validate an alarm for operational use. All 2015 periods remain development data.",
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
        "calibration_alert_states.csv.gz",
        "test_alert_states.csv.gz",
        "calibration_common_alert_states.csv.gz",
        "test_common_alert_states.csv.gz",
        "ALERT_STABILITY_RESULTS.md",
    ]
    provenance = {
        "artifact": "health_twin_alert_stability_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential alert stability research",
        "pre_specification": "docs/HEALTH_TWIN_ALERT_STABILITY_PLAN.md",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
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
        "kernel_numerics": "stabilized Gaussian weights",
        "alpha_grid": ALPHA_GRID,
        "peer_grid": PEER_GRID,
        "bandwidth_grid": BANDWIDTH_GRID,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "minimum_calibration_rows": MIN_CALIBRATION_ROWS,
        "persistence_length": PERSISTENCE_LENGTH,
        "maximum_contiguous_gap_minutes": 20,
        "outputs": {name: sha256_file(args.output_dir / name) for name in output_names},
        "limits": [
            "No fault, alarm, maintenance, or availability labels are present.",
            "Test alert rates are developmental stability measures, not accuracy.",
            "Same-site 2015 data have informed prior method development.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(native(provenance), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
