#!/usr/bin/env python3
"""Run the frozen all-peer synthetic common/specific state-separation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_baseline import MIN_TRAINING_SCALE
from run_health_twin_synthetic_state_separation import (
    AMPLITUDES,
    MIN_TURBINES,
    POLARITIES,
    TARGET_ROLES,
    markdown_table,
    quantile,
    rotating_events,
)
from run_health_twin_regime_peer_v2 import weighted_median
from run_health_twin_trim_distance_ablation import common_trim


METHOD = "distance_trim"
COMMON_MIN_TURBINES = 4


def eligible_common_state(frame):
    """Median over operating-eligible finite residuals, before scoreability.

    The enclosing synthetic scenario retains its original >=8 specifically
    scoreable-turbine rule; this separate common-state minimum is not a new
    scenario eligibility rule. residual_frames already encode operating eligibility.
    """
    finite = frame.loc[np.isfinite(frame["residual"])].copy()
    if finite.duplicated(["timestamp", "turbine_id"]).any():
        raise ValueError("Duplicate operating-eligible timestamp-turbine residual.")
    common = finite.groupby("timestamp").agg(
        available_turbines=("turbine_id", "nunique"),
        common_state=("residual", "median"),
    )
    return common.loc[common["available_turbines"].ge(COMMON_MIN_TURBINES), "common_state"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_peer_scores(frame, target, residual_scales, bandwidth):
    rows = []
    for timestamp, group in frame.groupby("timestamp", sort=False):
        group = group.reset_index(drop=True)
        residual = group["residual"].to_numpy(dtype=float)
        descriptor = group[
            [f"regime_{index}" for index in range(7)]
        ].to_numpy(dtype=float)
        turbine = group["turbine_id"].astype(str).to_numpy()
        for index in range(len(group)):
            candidates = np.asarray(
                [other for other in range(len(group)) if other != index], dtype=int
            )
            if len(candidates) < 3:
                continue
            distances = np.sqrt(
                np.sum((descriptor[candidates] - descriptor[index]) ** 2, axis=1)
            )
            keep = common_trim(residual, candidates, turbine, residual_scales)
            if int(keep.sum()) < 3:
                continue
            selected = candidates[keep]
            selected_distances = distances[keep]
            values = residual[selected]
            log_kernel = -0.5 * (selected_distances / float(bandwidth)) ** 2
            log_kernel -= np.max(log_kernel)
            reference = weighted_median(values, np.exp(log_kernel))
            rows.append(
                {
                    "target": target,
                    "timestamp": timestamp,
                    "turbine_id": turbine[index],
                    "method": METHOD,
                    "residual": float(residual[index]),
                    "peer_reference": float(reference),
                    "specific_residual": float(residual[index] - reference),
                }
            )
    return pd.DataFrame(rows)


def baseline_frame(locked, residual_frames, target, residual_scales, bandwidth):
    selected = locked.loc[locked["target"].eq(target)].copy()
    frame = residual_frames.loc[residual_frames["target"].eq(target)].copy()
    frame = frame.drop(columns="target")
    regime_columns = [f"regime_{index}" for index in range(7)]
    if frame[regime_columns].isna().any().any():
        raise ValueError(f"Missing locked regime descriptor for {target}.")
    if frame.duplicated(["timestamp", "turbine_id"]).any():
        raise ValueError(f"Duplicate residual-frame key for {target}.")
    frame = frame.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    recomputed = all_peer_scores(frame, target, residual_scales, bandwidth)
    check = selected.merge(
        recomputed,
        on=["target", "timestamp", "turbine_id", "method"],
        suffixes=("_locked", "_recomputed"),
        validate="one_to_one",
    )
    if len(check) != len(selected) or len(check) != len(recomputed):
        raise ValueError(f"Locked/recomputed support differs for {target}.")
    residual_error = float(
        np.max(np.abs(check["residual_locked"] - check["residual_recomputed"]))
    )
    reference_error = float(
        np.max(np.abs(check["peer_reference_locked"] - check["peer_reference_recomputed"]))
    )
    specific_error = float(
        np.max(
            np.abs(
                check["specific_residual_locked"]
                - check["specific_residual_recomputed"]
            )
        )
    )
    if max(residual_error, reference_error, specific_error) > 1e-8:
        raise ValueError(
            f"Locked all-peer reproduction failed for {target}: "
            f"residual={residual_error}, reference={reference_error}, "
            f"specific={specific_error}."
        )
    return frame, recomputed, residual_error, reference_error, specific_error


def eligible_simulation(frame, baseline):
    scored_counts = baseline.groupby("timestamp")["turbine_id"].nunique()
    timestamps = set(scored_counts.loc[scored_counts.ge(MIN_TURBINES)].index)
    simulation = frame.loc[frame["timestamp"].isin(timestamps)].copy()
    baseline = baseline.loc[baseline["timestamp"].isin(timestamps)].copy()
    if simulation.empty:
        raise ValueError("No timestamps satisfy the fixed minimum-turbine support rule.")
    return simulation, baseline


def local_scenario(frame, baseline, target, residual_scales, bandwidth, multiplier):
    events = rotating_events(baseline, residual_scales)
    modified = frame.merge(events, on="timestamp", how="left", validate="many_to_one")
    modified["delta"] = multiplier * modified["training_scale"]
    injected_mask = modified["turbine_id"].eq(modified["injected_turbine"])
    modified.loc[injected_mask, "residual"] += modified.loc[injected_mask, "delta"]
    injected = all_peer_scores(
        modified.drop(columns=["injected_turbine", "training_scale", "delta"]),
        target,
        residual_scales,
        bandwidth,
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
    target_rows = paired.loc[
        paired["turbine_id"].eq(paired["injected_turbine"])
    ].copy()
    if target_rows["timestamp"].nunique() != events["timestamp"].nunique():
        raise ValueError(f"Injected target support changed for {target}.")
    target_gain = target_rows["specific_change"] / target_rows["delta"]

    base_common = eligible_common_state(frame)
    injected_common = eligible_common_state(modified)
    common_change = injected_common - base_common
    event_delta = events.set_index("timestamp")["training_scale"] * multiplier
    common_leakage = common_change / event_delta

    other_rows = paired.loc[
        ~paired["turbine_id"].eq(paired["injected_turbine"])
    ].copy()
    other_spillover = other_rows["specific_change"].abs() / other_rows["delta"].abs()
    return {
        "scenario": "local_single_turbine",
        "target": target,
        "role": TARGET_ROLES.get(target, "other"),
        "sigma_multiplier": multiplier,
        "timestamps": int(events["timestamp"].nunique()),
        "injected_rows": int(len(target_rows)),
        "paired_rows": int(len(paired)),
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


def common_scenario(frame, baseline, target, residual_scales, bandwidth, multiplier):
    scale = float(np.median(list(residual_scales.values())))
    delta = multiplier * max(scale, MIN_TRAINING_SCALE)
    modified = frame.copy()
    modified["residual"] += delta
    injected = all_peer_scores(modified, target, residual_scales, bandwidth)
    paired = baseline.merge(
        injected,
        on=["target", "timestamp", "turbine_id", "method"],
        suffixes=("_base", "_injected"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(injected):
        raise ValueError(f"Fleet-common support changed for {target}.")
    specific_leakage = (
        paired["specific_residual_injected"] - paired["specific_residual_base"]
    ) / abs(delta)
    base_common = eligible_common_state(frame)
    injected_common = eligible_common_state(modified)
    common_gain = (injected_common - base_common) / delta
    return {
        "scenario": "fleet_common",
        "target": target,
        "role": TARGET_ROLES.get(target, "other"),
        "sigma_multiplier": multiplier,
        "timestamps": int(frame["timestamp"].nunique()),
        "injected_rows": int(len(frame)),
        "paired_rows": int(len(paired)),
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
    parser.add_argument("--residual-frames", type=Path, required=True)
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
    residual_frames = pd.read_csv(args.residual_frames, parse_dates=["timestamp"])
    residual_frames["turbine_id"] = residual_frames["turbine_id"].astype(str)

    rows = []
    reproduction_rows = []
    for target in provenance["targets"]:
        model = provenance["model_provenance"][target]
        residual_scales = {
            str(key): float(value)
            for key, value in model["residual_scales_by_turbine"].items()
        }
        bandwidth = float(model["selected_distance_bandwidth"])
        frame, baseline, residual_error, reference_error, specific_error = baseline_frame(
            locked, residual_frames, target, residual_scales, bandwidth
        )
        simulation, simulation_baseline = eligible_simulation(frame, baseline)
        reproduction_rows.append(
            {
                "target": target,
                "locked_rows": int(len(baseline)),
                "simulation_timestamps": int(simulation["timestamp"].nunique()),
                "max_residual_error": residual_error,
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
                        bandwidth,
                        multiplier,
                    )
                )
        print(f"[{target}] all-peer synthetic scenarios complete", flush=True)

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
        "# All-peer synthetic state separation — confidential local artifact",
        "",
        "## Scope",
        "",
        "Frozen all-peer test residuals were perturbed under the pre-specified local-single-turbine and fleet-common scenarios. The normal model and Gaussian bandwidth were not refitted or reselected. This is a post-hoc implementation/alignment stress check, not real-fault validation.",
        "",
        "## Locked-baseline reproduction",
        "",
        markdown_table(
            reproduction,
            [
                "target",
                "locked_rows",
                "simulation_timestamps",
                "max_residual_error",
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
        "Local gain is measured in the injected turbine-specific state; common leakage is the fleet-median response divided by the injected shift; non-target spillover is the change in other turbines' specific states divided by the injected shift. Fleet-common gain is measured in the common state and leakage in turbine-specific states.",
        "",
        "## Interpretation boundary",
        "",
        "Passing confirms the intended algebra on a frozen development state and quantifies peer-reference spillover. It does not measure sensitivity, specificity, false-alarm rate, failure lead time, or maintenance value.",
    ]
    report_path = args.output_dir / "ALL_PEER_SYNTHETIC_STATE_SEPARATION.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    output_names = [
        "synthetic_scenario_results.csv",
        "baseline_reproduction.csv",
        "synthetic_target_summary.csv",
        "ALL_PEER_SYNTHETIC_STATE_SEPARATION.md",
    ]
    manifest = {
        "artifact": "health_twin_all_peer_synthetic_state_separation_v0_7_eligible_min4",
        "confidentiality": "local-only confidential synthetic method check",
        "pre_specification": "docs/HEALTH_TWIN_ALL_PEER_STATE_SEPARATION_PLAN.md",
        "inputs": {
            str(args.scores): sha256_file(args.scores),
            str(args.residual_frames): sha256_file(args.residual_frames),
            str(args.source_provenance): sha256_file(args.source_provenance),
        },
        "outputs": {name: sha256_file(args.output_dir / name) for name in output_names},
        "method": METHOD,
        "candidate_peers": "all eligible leave-one-out peers",
        "kernel_numerics": "stabilized Gaussian weights",
        "amplitudes_training_sigma": AMPLITUDES,
        "polarities": POLARITIES,
        "minimum_scored_turbines": MIN_TURBINES,
        "common_minimum_operating_eligible_finite_turbines": COMMON_MIN_TURBINES,
        "common_definition": "median of operating-eligible finite residuals before specific-scoreability filtering",
        "scenario_eligibility": "unchanged: at least 8 specifically scoreable turbines",
        "selection_or_fitting": "none",
        "limits": [
            "Post-hoc implementation/alignment stress check.",
            "Synthetic residual perturbations are not real fault events.",
            "The check uses one locked site-year development test.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not bool(results["check_pass"].all()):
        raise RuntimeError("One or more frozen all-peer state checks failed.")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
