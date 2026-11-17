#!/usr/bin/env python3
"""Run a pre-specified synthetic separation check on locked twin states.

This script changes locked test residuals only. It does not refit the normal
model, reselect peer parameters, set alert thresholds, or use event labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_baseline import MIN_TRAINING_SCALE, SOURCE_COLUMNS, make_feature_matrix
from run_health_twin_regime_peer_v2 import peer_reference_scores, regime_matrix


METHOD = "regime_matched"
AMPLITUDES = [1.0, 2.0, 3.0]
POLARITIES = [-1.0, 1.0]
MIN_TURBINES = 8
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


def quantile(values, probability):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, probability)) if len(array) else np.nan


def markdown_table(frame, columns):
    selected = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("—" if not np.isfinite(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reconstruct_descriptors(data, provenance):
    ambient_column = provenance["ambient_column"]
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    features, _ = make_feature_matrix(data, turbine_ids, ambient_column)
    descriptor, _ = regime_matrix(data, ambient_column)
    monitoring_start = pd.Timestamp(provenance["splits"]["monitoring_start_inclusive"])
    train_end = pd.Timestamp(provenance["splits"]["train_end_exclusive"])
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
    train_period = data["timestamp"].ge(monitoring_start).to_numpy() & data[
        "timestamp"
    ].lt(train_end).to_numpy()
    regime_mean = np.mean(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.std(descriptor[base_eligible & train_period], axis=0)
    regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
    descriptor = (descriptor - regime_mean) / regime_scale
    lookup = data[["timestamp", "turbine_id"]].copy()
    lookup["turbine_id"] = lookup["turbine_id"].astype(str)
    for index in range(descriptor.shape[1]):
        lookup[f"regime_{index}"] = descriptor[:, index]
    if lookup.duplicated(["timestamp", "turbine_id"]).any():
        raise ValueError("Descriptor lookup has duplicate timestamp-turbine keys.")
    return lookup


def baseline_frame(locked, lookup, target, residual_scales, peer_count, bandwidth):
    selected = locked.loc[locked["target"].eq(target)].copy()
    frame = selected[["timestamp", "turbine_id", "residual"]].merge(
        lookup, on=["timestamp", "turbine_id"], how="left", validate="one_to_one"
    )
    regime_columns = [f"regime_{index}" for index in range(7)]
    if frame[regime_columns].isna().any().any():
        raise ValueError(f"Missing reconstructed regime descriptor for {target}.")
    frame = frame.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    recomputed = peer_reference_scores(
        frame,
        target,
        METHOD,
        residual_scales,
        peer_count=peer_count,
        bandwidth=bandwidth,
        stabilize_kernel=False,
    )
    check = selected.merge(
        recomputed,
        on=["target", "timestamp", "turbine_id", "method"],
        suffixes=("_locked", "_recomputed"),
        validate="one_to_one",
    )
    reference_error = float(
        np.max(np.abs(check["peer_reference_locked"] - check["peer_reference_recomputed"]))
    )
    specific_error = float(
        np.max(np.abs(check["specific_residual_locked"] - check["specific_residual_recomputed"]))
    )
    if max(reference_error, specific_error) > 1e-5:
        raise ValueError(
            f"Locked baseline reproduction failed for {target}: "
            f"reference={reference_error}, specific={specific_error}."
        )
    return frame, recomputed, reference_error, specific_error


def eligible_simulation(frame, baseline):
    counts = frame.groupby("timestamp")["turbine_id"].transform("nunique")
    simulation = frame.loc[counts.ge(MIN_TURBINES)].copy()
    timestamps = set(simulation["timestamp"].unique())
    baseline = baseline.loc[baseline["timestamp"].isin(timestamps)].copy()
    if simulation.empty:
        raise ValueError("No timestamps satisfy the fixed minimum-turbine support rule.")
    return simulation, baseline


def rotating_events(frame, residual_scales):
    rows = []
    for position, (timestamp, group) in enumerate(frame.groupby("timestamp", sort=True)):
        turbines = sorted(group["turbine_id"].astype(str).unique().tolist())
        turbine = turbines[position % len(turbines)]
        rows.append(
            {
                "timestamp": timestamp,
                "injected_turbine": turbine,
                "training_scale": max(float(residual_scales[turbine]), MIN_TRAINING_SCALE),
            }
        )
    return pd.DataFrame(rows)


def local_scenario(frame, baseline, target, residual_scales, peer_count, bandwidth, multiplier):
    events = rotating_events(frame, residual_scales)
    modified = frame.merge(events, on="timestamp", how="left", validate="many_to_one")
    modified["delta"] = multiplier * modified["training_scale"]
    injected_mask = modified["turbine_id"].eq(modified["injected_turbine"])
    modified.loc[injected_mask, "residual"] += modified.loc[injected_mask, "delta"]
    injected = peer_reference_scores(
        modified.drop(columns=["injected_turbine", "training_scale", "delta"]),
        target,
        METHOD,
        residual_scales,
        peer_count=peer_count,
        bandwidth=bandwidth,
        stabilize_kernel=False,
    )
    paired = baseline.merge(
        injected,
        on=["target", "timestamp", "turbine_id", "method"],
        suffixes=("_base", "_injected"),
        validate="one_to_one",
    ).merge(events, on="timestamp", how="left", validate="many_to_one")
    paired["delta"] = multiplier * paired["training_scale"]
    paired["specific_change"] = (
        paired["specific_residual_injected"] - paired["specific_residual_base"]
    )
    target_rows = paired.loc[paired["turbine_id"].eq(paired["injected_turbine"])].copy()
    target_gain = target_rows["specific_change"] / target_rows["delta"]

    base_common = frame.groupby("timestamp")["residual"].median()
    injected_common = modified.groupby("timestamp")["residual"].median()
    common_change = injected_common - base_common
    event_delta = events.set_index("timestamp")["training_scale"] * multiplier
    common_leakage = common_change / event_delta

    other_rows = paired.loc[~paired["turbine_id"].eq(paired["injected_turbine"])].copy()
    other_spillover = other_rows["specific_change"].abs() / other_rows["delta"].abs()
    return {
        "scenario": "local_single_turbine",
        "target": target,
        "role": TARGET_ROLES.get(target, "other"),
        "sigma_multiplier": multiplier,
        "timestamps": int(events["timestamp"].nunique()),
        "injected_rows": int(len(target_rows)),
        "median_state_gain": float(np.median(target_gain)),
        "p05_state_gain": quantile(target_gain, 0.05),
        "p95_state_gain": quantile(target_gain, 0.95),
        "max_abs_gain_error": float(np.max(np.abs(target_gain - 1.0))),
        "median_abs_common_leakage": float(np.median(np.abs(common_leakage))),
        "p95_abs_common_leakage": quantile(np.abs(common_leakage), 0.95),
        "median_abs_specific_leakage": np.nan,
        "p95_abs_specific_leakage": np.nan,
        "median_abs_nontarget_spillover": float(np.median(other_spillover)),
        "p95_abs_nontarget_spillover": quantile(other_spillover, 0.95),
    }


def common_scenario(frame, baseline, target, residual_scales, peer_count, bandwidth, multiplier):
    scale = float(np.median(list(residual_scales.values())))
    delta = multiplier * max(scale, MIN_TRAINING_SCALE)
    modified = frame.copy()
    modified["residual"] += delta
    injected = peer_reference_scores(
        modified,
        target,
        METHOD,
        residual_scales,
        peer_count=peer_count,
        bandwidth=bandwidth,
        stabilize_kernel=False,
    )
    paired = baseline.merge(
        injected,
        on=["target", "timestamp", "turbine_id", "method"],
        suffixes=("_base", "_injected"),
        validate="one_to_one",
    )
    specific_leakage = (
        paired["specific_residual_injected"] - paired["specific_residual_base"]
    ) / abs(delta)
    base_common = frame.groupby("timestamp")["residual"].median()
    injected_common = modified.groupby("timestamp")["residual"].median()
    common_gain = (injected_common - base_common) / delta
    return {
        "scenario": "fleet_common",
        "target": target,
        "role": TARGET_ROLES.get(target, "other"),
        "sigma_multiplier": multiplier,
        "timestamps": int(frame["timestamp"].nunique()),
        "injected_rows": int(len(frame)),
        "median_state_gain": float(np.median(common_gain)),
        "p05_state_gain": quantile(common_gain, 0.05),
        "p95_state_gain": quantile(common_gain, 0.95),
        "max_abs_gain_error": float(np.max(np.abs(common_gain - 1.0))),
        "median_abs_common_leakage": np.nan,
        "p95_abs_common_leakage": np.nan,
        "median_abs_specific_leakage": float(np.median(np.abs(specific_leakage))),
        "p95_abs_specific_leakage": quantile(np.abs(specific_leakage), 0.95),
        "median_abs_nontarget_spillover": np.nan,
        "p95_abs_nontarget_spillover": np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "synthetic_scenario_results.csv"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use --overwrite to replace it")

    provenance = json.loads(args.source_provenance.read_text())
    locked = pd.read_csv(args.scores, parse_dates=["timestamp"])
    locked = locked.loc[locked["method"].eq(METHOD)].copy()
    locked["turbine_id"] = locked["turbine_id"].astype(str)
    if locked.duplicated(["target", "timestamp", "turbine_id"]).any():
        raise ValueError("Locked score input contains duplicate target-time-turbine rows.")
    data = pd.read_csv(args.source_data, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    lookup = reconstruct_descriptors(data, provenance)

    rows = []
    reproduction_rows = []
    for target in provenance["targets"]:
        model = provenance["model_provenance"][target]
        residual_scales = {str(key): float(value) for key, value in model["residual_scales_by_turbine"].items()}
        peer_count = int(model["selected_regime_peer_count"])
        bandwidth = float(model["selected_regime_distance_bandwidth"])
        frame, baseline, reference_error, specific_error = baseline_frame(
            locked, lookup, target, residual_scales, peer_count, bandwidth
        )
        simulation, simulation_baseline = eligible_simulation(frame, baseline)
        reproduction_rows.append(
            {
                "target": target,
                "locked_rows": int(len(baseline)),
                "simulation_timestamps": int(simulation["timestamp"].nunique()),
                "max_peer_reference_error": reference_error,
                "max_specific_state_error": specific_error,
            }
        )
        for amplitude in AMPLITUDES:
            for polarity in POLARITIES:
                multiplier = amplitude * polarity
                rows.append(
                    local_scenario(
                        simulation,
                        simulation_baseline,
                        target,
                        residual_scales,
                        peer_count,
                        bandwidth,
                        multiplier,
                    )
                )
                rows.append(
                    common_scenario(
                        simulation,
                        simulation_baseline,
                        target,
                        residual_scales,
                        peer_count,
                        bandwidth,
                        multiplier,
                    )
                )
        print(f"[{target}] synthetic scenarios complete", flush=True)

    results = pd.DataFrame(rows)
    reproduction = pd.DataFrame(reproduction_rows)
    results["check_pass"] = np.where(
        results["scenario"].eq("local_single_turbine"),
        results["max_abs_gain_error"].le(1e-6)
        & results["median_abs_common_leakage"].le(0.10),
        results["max_abs_gain_error"].le(1e-6)
        & results["p95_abs_specific_leakage"].le(1e-6),
    )
    results.to_csv(result_path, index=False)
    reproduction.to_csv(args.output_dir / "baseline_reproduction.csv", index=False)

    target_summary = results.groupby(["target", "role", "scenario"], as_index=False).agg(
        configurations=("sigma_multiplier", "size"),
        checks_passed=("check_pass", "sum"),
        max_abs_gain_error=("max_abs_gain_error", "max"),
        max_median_abs_common_leakage=("median_abs_common_leakage", "max"),
        max_p95_abs_specific_leakage=("p95_abs_specific_leakage", "max"),
        max_p95_abs_nontarget_spillover=("p95_abs_nontarget_spillover", "max"),
    )
    target_summary.to_csv(args.output_dir / "synthetic_target_summary.csv", index=False)
    report = [
        "# Synthetic twin-state separation — confidential local artifact",
        "",
        "## Scope",
        "",
        "Locked independent-meteorology test residuals were perturbed under the pre-specified local-single-turbine and fleet-common scenarios. The normal model and peer parameters were not refitted or reselected. This is an internal method sensitivity check, not real-fault validation.",
        "",
        "## Locked-baseline reproduction",
        "",
        markdown_table(
            reproduction,
            [
                "target",
                "locked_rows",
                "simulation_timestamps",
                "max_peer_reference_error",
                "max_specific_state_error",
            ],
        ),
        "",
        "## Scenario summary",
        "",
        markdown_table(
            target_summary,
            [
                "target",
                "role",
                "scenario",
                "configurations",
                "checks_passed",
                "max_abs_gain_error",
                "max_median_abs_common_leakage",
                "max_p95_abs_specific_leakage",
                "max_p95_abs_nontarget_spillover",
            ],
        ),
        "",
        "For local injections, gain is measured in the injected turbine-specific state, common leakage is the fleet-median response divided by the injected shift, and non-target spillover is the change in other turbines' specific states divided by the injected shift. For fleet-common injections, gain is measured in the common state and leakage in turbine-specific states.",
        "",
        "## Interpretation boundary",
        "",
        "Passing these checks confirms the intended algebraic separation on locked residuals and quantifies peer-reference spillover. It does not measure sensitivity, specificity, false-alarm rate, failure lead time, or maintenance value.",
    ]
    report_path = args.output_dir / "SYNTHETIC_STATE_SEPARATION.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    output_names = [
        "synthetic_scenario_results.csv",
        "baseline_reproduction.csv",
        "synthetic_target_summary.csv",
        "SYNTHETIC_STATE_SEPARATION.md",
    ]
    manifest = {
        "artifact": "health_twin_synthetic_state_separation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential synthetic method check",
        "pre_specification": "docs/HEALTH_TWIN_SYNTHETIC_STATE_SEPARATION_PLAN.md",
        "inputs": {
            str(args.scores): sha256_file(args.scores),
            str(args.source_data): sha256_file(args.source_data),
            str(args.source_provenance): sha256_file(args.source_provenance),
        },
        "outputs": {name: sha256_file(args.output_dir / name) for name in output_names},
        "method": METHOD,
        "kernel_numerics": "legacy unstabilized Gaussian arithmetic required to reproduce frozen v3 scores; new experiments use stabilized arithmetic",
        "amplitudes_training_sigma": AMPLITUDES,
        "polarities": POLARITIES,
        "minimum_scored_turbines": MIN_TURBINES,
        "selection_or_fitting": "none",
        "limits": [
            "Synthetic residual perturbations are not real fault events.",
            "The check uses one locked site-year development test.",
            "Alert thresholds and persistence are outside this artifact.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not bool(results["check_pass"].all()):
        raise RuntimeError("One or more pre-specified synthetic state checks failed; inspect outputs.")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
