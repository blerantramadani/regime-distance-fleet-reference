#!/usr/bin/env python3
"""Establish a local quality gate for independent meteorology in health models.

The 2015 Bogdanci health layer repeats meteorological records across all
turbines.  This audit turns the previously established rotor-wind dynamic start
into an explicit data-contract rule for every health experiment that uses an
independent meteorological covariate.  It reports aggregates only; no raw
SCADA observations are emitted outside the local workspace.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START = pd.Timestamp("2015-05-14 12:00:00")
MET_SIGNALS = [
    "met_temperature_mean",
    "met_wind_speed_mean",
    "met_wind_direction_mean",
    "met_humidity_mean",
    "met_pressure_mean",
]


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


def markdown_table(frame):
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def signal_summary(series: pd.Series) -> dict:
    value = pd.to_numeric(series, errors="coerce").dropna()
    if value.empty:
        return {
            "timestamps": int(len(series)),
            "missing_pct": 100.0,
            "unique_values": 0,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "std": np.nan,
            "median_abs_step": np.nan,
            "p95_abs_step": np.nan,
        }
    steps = value.diff().abs().dropna()
    return {
        "timestamps": int(len(series)),
        "missing_pct": float(series.isna().mean() * 100.0),
        "unique_values": int(value.nunique()),
        "min": float(value.min()),
        "max": float(value.max()),
        "median": float(value.median()),
        "std": float(value.std(ddof=0)),
        "median_abs_step": float(steps.median()) if not steps.empty else np.nan,
        "p95_abs_step": float(steps.quantile(0.95)) if not steps.empty else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument(
        "--dynamic-start",
        default=str(DEFAULT_START),
        help="Approved first timestamp for independent meteorological health covariates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiment_results" / "health_meteorology_gate_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dynamic_start = pd.Timestamp(args.dynamic_start)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_file = args.output_dir / "meteorology_health_gate.json"
    if gate_file.exists() and not args.overwrite:
        raise FileExistsError(f"{gate_file} exists; use --overwrite to replace it")

    usecols = ["timestamp", "turbine_id", *MET_SIGNALS]
    data = pd.read_csv(args.input, usecols=usecols, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable")
    duplicate_counts = data.groupby("timestamp", sort=False)[MET_SIGNALS].nunique(dropna=False)
    identical = bool(duplicate_counts.le(1).all().all())
    met = data.drop_duplicates("timestamp", keep="first").sort_values("timestamp", kind="stable")
    if met.empty:
        raise ValueError("No meteorological timestamps available.")
    if met["timestamp"].min() >= dynamic_start:
        raise ValueError("The declared dynamic start is not after an early segment.")

    rows = []
    for signal in MET_SIGNALS:
        before = signal_summary(met.loc[met["timestamp"].lt(dynamic_start), signal])
        after = signal_summary(met.loc[met["timestamp"].ge(dynamic_start), signal])
        rows.append(
            {
                "signal": signal,
                "pre_gate_timestamps": before["timestamps"],
                "pre_gate_missing_pct": before["missing_pct"],
                "pre_gate_unique_values": before["unique_values"],
                "pre_gate_min": before["min"],
                "pre_gate_max": before["max"],
                "pre_gate_median": before["median"],
                "pre_gate_std": before["std"],
                "post_gate_timestamps": after["timestamps"],
                "post_gate_missing_pct": after["missing_pct"],
                "post_gate_unique_values": after["unique_values"],
                "post_gate_min": after["min"],
                "post_gate_max": after["max"],
                "post_gate_median": after["median"],
                "post_gate_std": after["std"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "meteorology_pre_post_gate_summary.csv", index=False)
    month = met.assign(month=met["timestamp"].dt.to_period("M").astype(str))
    monthly = month.groupby("month", sort=True)[MET_SIGNALS].agg(["count", "median", "std", "min", "max"])
    monthly.columns = ["_".join(column) for column in monthly.columns]
    monthly.reset_index().to_csv(args.output_dir / "meteorology_monthly_health_summary.csv", index=False)

    near_gate = met.loc[
        met["timestamp"].between(dynamic_start - pd.Timedelta(minutes=20), dynamic_start + pd.Timedelta(minutes=20)),
        ["timestamp", *MET_SIGNALS],
    ].copy()
    near_gate.to_csv(args.output_dir / "meteorology_gate_transition_aggregate.csv", index=False)
    record = {
        "artifact": "health_meteorology_gate_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential research audit; no raw SCADA observations are redistributed",
        "input": str(args.input),
        "approved_independent_meteorology_start": str(dynamic_start),
        "health_model_rule": "Any normal-behaviour or matched-peer health model that includes an independent meteorological covariate must exclude earlier timestamps from fitting, tuning, and evaluation.",
        "meteorology_identical_across_turbines_per_timestamp": identical,
        "meteorological_timestamp_count": int(len(met)),
        "excluded_pre_gate_timestamp_count": int((met["timestamp"] < dynamic_start).sum()),
        "approved_post_gate_timestamp_count": int((met["timestamp"] >= dynamic_start).sum()),
        "basis": [
            "The independent rotor-wind signal was previously audited as effectively frozen until the approved start.",
            "Independent temperature exhibits the same material transition into a plausible seasonal range at that boundary.",
            "The gate is a data-quality exclusion, not an inference about turbine condition.",
        ],
        "scope": {
            "approved_for_post_gate_health_models": MET_SIGNALS,
            "not_a_full_meteorological_calibration": True,
            "no_direction_or_wake_claim": True,
        },
    }
    gate_file.write_text(json.dumps(native(record), indent=2) + "\n", encoding="utf-8")
    report = [
        "# Independent-meteorology quality gate for health models v1 — confidential local artifact",
        "",
        "## Decision",
        "",
        f"Independent meteorological covariates used in the thermal health model are approved only from **{dynamic_start:%Y-%m-%d %H:%M}** onward. Every experiment using these covariates must exclude all earlier records from fitting, validation selection, and testing.",
        "",
        "This decision extends the existing rotor-wind/direction quality audit to the health-model covariate contract. It is not a turbine-health conclusion and does not alter raw source data.",
        "",
        "## Evidence",
        "",
        f"Meteorological values are identical across the 16 turbine rows at each timestamp: **{identical}**. The annual layer has {len(met):,} meteorological timestamps: {(met['timestamp'] < dynamic_start).sum():,} pre-gate and {(met['timestamp'] >= dynamic_start).sum():,} post-gate.",
        "",
        markdown_table(summary),
        "",
        "The existing direction audit independently identifies the same boundary from the rotor-wind signal. The temperature channel transitions from a narrow non-seasonal early range to a materially varying seasonal range at the same point; therefore it must not be used to train an all-year independent-meteorology normal-behaviour model.",
        "",
        "## Consequence for recorded experiments",
        "",
        "The earlier `health_twin_baseline_v1`, `health_twin_regime_peer_v2`, and rolling-v3 outputs included pre-gate independent meteorology in at least one fitting window. They are retained only as quality-control/audit artifacts and are superseded for scientific interpretation. Clean post-gate reruns must use a separate output directory and retain this gate in their provenance.",
        "",
        "## Permitted next experiment",
        "",
        "Use the post-gate period only: train from 2015-05-14 to 2015-08-31, tune in September–October, and make one final November–December temporal evaluation. Report covariate support and seasonal limitations explicitly.",
    ]
    (args.output_dir / "HEALTH_METEOROLOGY_GATE_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
