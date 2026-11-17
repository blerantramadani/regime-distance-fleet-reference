#!/usr/bin/env python3
"""Apply frozen alert rules to pre-specified sustained synthetic shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANCHORS = [
    pd.Timestamp("2015-11-08 00:00:00"),
    pd.Timestamp("2015-11-22 00:00:00"),
    pd.Timestamp("2015-12-06 00:00:00"),
    pd.Timestamp("2015-12-20 00:00:00"),
]
DURATIONS_HOURS = [1, 6, 24]
AMPLITUDES = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
SHAPES = ["step", "drift"]
MAX_GAP = pd.Timedelta(minutes=20)
PERSISTENCE = 3
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


def robust_scale(values):
    values = np.asarray(values, dtype=float)
    centre = np.median(values)
    return max(float(1.4826 * np.median(np.abs(values - centre))), 1e-8)


def first_persistent_time(timestamps, point_flags):
    run = 0
    previous_time = None
    for timestamp, point_flag in zip(timestamps, point_flags):
        contiguous = previous_time is not None and timestamp - previous_time <= MAX_GAP
        if bool(point_flag):
            run = run + 1 if contiguous else 1
        else:
            run = 0
        if run >= PERSISTENCE:
            return timestamp
        previous_time = timestamp
    return None


def event_rows_for_group(
    group,
    channel,
    target,
    entity,
    state_column,
    centre,
    threshold,
    scale,
):
    rows = []
    group = group.sort_values("timestamp", kind="stable")
    for anchor_index, anchor in enumerate(ANCHORS, start=1):
        candidates = group.loc[group["timestamp"].between(anchor, anchor + pd.Timedelta(hours=1))]
        start = candidates["timestamp"].min() if not candidates.empty else pd.NaT
        for duration_hours in DURATIONS_HOURS:
            for shape in SHAPES:
                for amplitude in AMPLITUDES:
                    base_record = {
                        "channel": channel,
                        "target": target,
                        "role": TARGET_ROLES.get(target, "other"),
                        "entity": entity,
                        "anchor_index": anchor_index,
                        "anchor": anchor,
                        "event_start": start,
                        "shape": shape,
                        "duration_hours": duration_hours,
                        "amplitude_sigma": amplitude,
                        "absolute_amplitude_sigma": abs(amplitude),
                    }
                    if pd.isna(start):
                        rows.append(
                            {
                                **base_record,
                                "status": "no_scored_start",
                                "scored_intervals": 0,
                                "background_persistent": False,
                                "detected": False,
                                "time_to_alert_minutes": np.nan,
                            }
                        )
                        continue
                    end = start + pd.Timedelta(hours=duration_hours)
                    window = group.loc[
                        group["timestamp"].ge(start) & group["timestamp"].lt(end)
                    ].copy()
                    if len(window) < PERSISTENCE:
                        rows.append(
                            {
                                **base_record,
                                "status": "insufficient_scored_intervals",
                                "scored_intervals": int(len(window)),
                                "background_persistent": False,
                                "detected": False,
                                "time_to_alert_minutes": np.nan,
                            }
                        )
                        continue
                    baseline_flags = (window[state_column] - centre).abs().gt(threshold)
                    baseline_time = first_persistent_time(
                        window["timestamp"].tolist(), baseline_flags.tolist()
                    )
                    if baseline_time is not None:
                        rows.append(
                            {
                                **base_record,
                                "status": "background_active",
                                "scored_intervals": int(len(window)),
                                "background_persistent": True,
                                "detected": False,
                                "time_to_alert_minutes": np.nan,
                            }
                        )
                        continue
                    if shape == "step":
                        fraction = np.ones(len(window), dtype=float)
                    else:
                        denominator = max(duration_hours * 60.0 - 10.0, 1.0)
                        elapsed = (
                            window["timestamp"] - start
                        ).dt.total_seconds().to_numpy(dtype=float) / 60.0
                        fraction = np.clip(elapsed / denominator, 0.0, 1.0)
                    injected_state = window[state_column].to_numpy(dtype=float) + (
                        amplitude * scale * fraction
                    )
                    injected_flags = np.abs(injected_state - centre) > threshold
                    detection_time = first_persistent_time(
                        window["timestamp"].tolist(), injected_flags.tolist()
                    )
                    detected = detection_time is not None
                    rows.append(
                        {
                            **base_record,
                            "status": "evaluable",
                            "scored_intervals": int(len(window)),
                            "background_persistent": False,
                            "detected": detected,
                            "time_to_alert_minutes": (
                                float((detection_time - start).total_seconds() / 60.0)
                                if detected
                                else np.nan
                            ),
                        }
                    )
    return rows


def build_specific_events(calibration, test):
    scale_map = calibration.groupby(["target", "turbine_id"])["specific_residual"].apply(
        robust_scale
    )
    rows = []
    for (target, turbine), group in test.groupby(["target", "turbine_id"], sort=True):
        centre = float(group["state_centre"].iloc[0])
        threshold = float(group["absolute_deviation_threshold"].iloc[0])
        rows.extend(
            event_rows_for_group(
                group,
                "turbine_specific",
                target,
                turbine,
                "specific_residual",
                centre,
                threshold,
                float(scale_map.loc[(target, turbine)]),
            )
        )
    return rows


def build_common_events(calibration, test):
    scale_map = calibration.groupby("target")["common_state"].apply(robust_scale)
    rows = []
    for target, group in test.groupby("target", sort=True):
        centre = float(group["common_centre"].iloc[0])
        threshold = float(group["common_absolute_deviation_threshold"].iloc[0])
        rows.extend(
            event_rows_for_group(
                group,
                "fleet_common",
                target,
                "fleet",
                "common_state",
                centre,
                threshold,
                float(scale_map.loc[target]),
            )
        )
    return rows


def summarize(events, group_columns):
    rows = []
    for key, group in events.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        evaluable = group.loc[group["status"].eq("evaluable")]
        detected = evaluable.loc[evaluable["detected"]]
        row = dict(zip(group_columns, key))
        row.update(
            {
                "candidate_events": int(len(group)),
                "evaluable_events": int(len(evaluable)),
                "background_active_events": int(group["status"].eq("background_active").sum()),
                "insufficient_or_missing_events": int(
                    group["status"].isin(
                        ["no_scored_start", "insufficient_scored_intervals"]
                    ).sum()
                ),
                "detected_events": int(len(detected)),
                "detection_fraction": (
                    float(len(detected) / len(evaluable)) if len(evaluable) else np.nan
                ),
                "median_time_to_alert_minutes": (
                    float(detected["time_to_alert_minutes"].median())
                    if len(detected)
                    else np.nan
                ),
                "p90_time_to_alert_minutes": (
                    float(detected["time_to_alert_minutes"].quantile(0.90))
                    if len(detected)
                    else np.nan
                ),
                "median_scored_intervals": float(group["scored_intervals"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-name",
        default="health_twin_synthetic_alert_sensitivity_v1",
    )
    parser.add_argument(
        "--pre-specification",
        default="docs/HEALTH_TWIN_SYNTHETIC_ALERT_SENSITIVITY_PLAN.md",
    )
    parser.add_argument(
        "--deterministic-provenance",
        action="store_true",
        help="Omit the run timestamp so a fresh rerun can be compared byte-for-byte.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "SYNTHETIC_ALERT_SENSITIVITY.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; use --overwrite to replace it")

    paths = {
        "calibration_specific": args.input_dir / "calibration_alert_states.csv.gz",
        "test_specific": args.input_dir / "test_alert_states.csv.gz",
        "calibration_common": args.input_dir / "calibration_common_alert_states.csv.gz",
        "test_common": args.input_dir / "test_common_alert_states.csv.gz",
        "source_provenance": args.input_dir / "provenance.json",
    }
    calibration_specific = pd.read_csv(paths["calibration_specific"], parse_dates=["timestamp"])
    test_specific = pd.read_csv(paths["test_specific"], parse_dates=["timestamp"])
    calibration_common = pd.read_csv(paths["calibration_common"], parse_dates=["timestamp"])
    test_common = pd.read_csv(paths["test_common"], parse_dates=["timestamp"])

    specific_scales = (
        calibration_specific.groupby(["target", "turbine_id"])["specific_residual"]
        .apply(robust_scale)
        .rename("calibration_robust_scale")
        .reset_index()
    )
    specific_thresholds = calibration_specific[
        ["target", "turbine_id", "absolute_deviation_threshold"]
    ].drop_duplicates()
    specific_ratios = specific_scales.merge(
        specific_thresholds,
        on=["target", "turbine_id"],
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "turbine_id": "entity",
            "absolute_deviation_threshold": "threshold",
        }
    )
    specific_ratios["channel"] = "turbine_specific"
    common_scales = (
        calibration_common.groupby("target")["common_state"]
        .apply(robust_scale)
        .rename("calibration_robust_scale")
        .reset_index()
    )
    common_thresholds = calibration_common[
        ["target", "common_absolute_deviation_threshold"]
    ].drop_duplicates()
    common_ratios = common_scales.merge(
        common_thresholds, on="target", how="left", validate="one_to_one"
    ).rename(columns={"common_absolute_deviation_threshold": "threshold"})
    common_ratios["entity"] = "fleet"
    common_ratios["channel"] = "fleet_common"
    scale_ratios = pd.concat([specific_ratios, common_ratios], ignore_index=True)
    scale_ratios["role"] = scale_ratios["target"].map(TARGET_ROLES).fillna("other")
    scale_ratios["threshold_to_robust_scale"] = (
        scale_ratios["threshold"] / scale_ratios["calibration_robust_scale"]
    )
    scale_ratio_summary = scale_ratios.groupby(
        ["channel", "target", "role"], as_index=False
    ).agg(
        entities=("entity", "nunique"),
        median_threshold_sigma=("threshold_to_robust_scale", "median"),
        q25_threshold_sigma=("threshold_to_robust_scale", lambda x: float(x.quantile(0.25))),
        q75_threshold_sigma=("threshold_to_robust_scale", lambda x: float(x.quantile(0.75))),
        min_threshold_sigma=("threshold_to_robust_scale", "min"),
        max_threshold_sigma=("threshold_to_robust_scale", "max"),
    )

    event_rows = build_specific_events(calibration_specific, test_specific)
    event_rows.extend(build_common_events(calibration_common, test_common))
    events = pd.DataFrame(event_rows)
    detailed_columns = [
        "channel",
        "target",
        "role",
        "entity",
        "anchor_index",
        "anchor",
        "event_start",
        "shape",
        "duration_hours",
        "amplitude_sigma",
        "absolute_amplitude_sigma",
        "status",
        "scored_intervals",
        "background_persistent",
        "detected",
        "time_to_alert_minutes",
    ]
    events = events.loc[:, detailed_columns]
    expected_events = (
        len(TARGET_ROLES)
        * (16 + 1)
        * len(ANCHORS)
        * len(DURATIONS_HOURS)
        * len(SHAPES)
        * len(AMPLITUDES)
    )
    if len(events) != expected_events:
        raise ValueError(f"Expected {expected_events} synthetic events, found {len(events)}.")
    event_key = [
        "channel",
        "target",
        "entity",
        "anchor_index",
        "shape",
        "duration_hours",
        "amplitude_sigma",
    ]
    if events.duplicated(event_key).any():
        raise ValueError("Synthetic event grid contains duplicate configuration keys.")
    evaluable = events["status"].eq("evaluable")
    if events.loc[evaluable & events["detected"], "time_to_alert_minutes"].isna().any():
        raise ValueError("A detected evaluable event is missing time-to-alert.")
    if events.loc[evaluable & ~events["detected"], "time_to_alert_minutes"].notna().any():
        raise ValueError("A non-detected evaluable event has a time-to-alert value.")
    configuration_columns = [
        "channel",
        "target",
        "role",
        "shape",
        "duration_hours",
        "amplitude_sigma",
        "absolute_amplitude_sigma",
    ]
    configuration = summarize(events, configuration_columns)
    aggregate = summarize(
        events,
        ["channel", "shape", "duration_hours", "absolute_amplitude_sigma"],
    )
    target_channel = summarize(events, ["channel", "target", "role"])

    events.to_csv(args.output_dir / "synthetic_alert_events.csv", index=False)
    configuration.to_csv(args.output_dir / "synthetic_alert_configuration_summary.csv", index=False)
    aggregate.to_csv(args.output_dir / "synthetic_alert_aggregate_summary.csv", index=False)
    target_channel.to_csv(args.output_dir / "synthetic_alert_target_summary.csv", index=False)
    scale_ratios.to_csv(args.output_dir / "threshold_scale_ratio.csv", index=False)
    scale_ratio_summary.to_csv(
        args.output_dir / "threshold_scale_target_summary.csv", index=False
    )

    report = [
        "# Sustained synthetic alert sensitivity — confidential local artifact",
        "",
        "## Scope",
        "",
        "Frozen October thresholds and the frozen three-interval persistence rule were applied to pre-specified local specific-state and fleet-common step/drift perturbations at four fixed November--December anchors. No model, peer parameter, threshold, or rule was reselected. These are synthetic sensitivity results, not fault-detection accuracy.",
        "",
        "## Aggregate sensitivity",
        "",
        markdown_table(
            aggregate,
            [
                "channel",
                "shape",
                "duration_hours",
                "absolute_amplitude_sigma",
                "candidate_events",
                "evaluable_events",
                "background_active_events",
                "detected_events",
                "detection_fraction",
                "median_time_to_alert_minutes",
                "p90_time_to_alert_minutes",
            ],
        ),
        "",
        "## Target/channel coverage",
        "",
        markdown_table(
            target_channel,
            [
                "channel",
                "target",
                "role",
                "candidate_events",
                "evaluable_events",
                "background_active_events",
                "detected_events",
                "detection_fraction",
                "median_time_to_alert_minutes",
            ],
        ),
        "",
        "Background-active windows are excluded from the paired synthetic denominator but remain explicitly counted. Detection means three consecutive threshold exceedances within the perturbation window, with gaps no greater than 20 minutes.",
        "",
        "## Threshold size relative to calibration robust scale",
        "",
        markdown_table(
            scale_ratio_summary,
            [
                "channel",
                "target",
                "role",
                "entities",
                "median_threshold_sigma",
                "q25_threshold_sigma",
                "q75_threshold_sigma",
                "min_threshold_sigma",
                "max_threshold_sigma",
            ],
        ),
        "",
        "A 99.5th-percentile threshold can be several robust calibration scales from the centre when the state distribution is heavy-tailed. This table explains why a nominal one-to-three-sigma perturbation need not cross the frozen threshold and is not a reason to retune after viewing test outcomes.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment maps state-shift magnitude and duration to the response of a frozen label-free alert rule. Calibration-sigma shifts are not mapped to a physical failure severity, so the results cannot be interpreted as clinical-style sensitivity, fault recall, or maintenance lead time.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    output_names = [
        "synthetic_alert_events.csv",
        "synthetic_alert_configuration_summary.csv",
        "synthetic_alert_aggregate_summary.csv",
        "synthetic_alert_target_summary.csv",
        "threshold_scale_ratio.csv",
        "threshold_scale_target_summary.csv",
        "SYNTHETIC_ALERT_SENSITIVITY.md",
    ]
    provenance = {
        "artifact": args.artifact_name,
        "confidentiality": "local-only confidential synthetic alert research",
        "pre_specification": args.pre_specification,
        "inputs": {str(path): sha256_file(path) for path in paths.values()},
        "outputs": {name: sha256_file(args.output_dir / name) for name in output_names},
        "anchors": [str(anchor) for anchor in ANCHORS],
        "shapes": SHAPES,
        "durations_hours": DURATIONS_HOURS,
        "amplitudes_calibration_sigma": AMPLITUDES,
        "persistence_length": PERSISTENCE,
        "maximum_gap_minutes": 20,
        "selection_or_fitting": "none",
        "limits": [
            "Synthetic state shifts are not physical component-failure simulations.",
            "Background-active windows are excluded by a frozen paired rule and reported.",
            "No real event labels are available.",
        ],
    }
    if not args.deterministic_provenance:
        provenance["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
