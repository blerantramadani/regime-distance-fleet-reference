#!/usr/bin/env python3
"""Audit the turbine-local ambient-temperature signal as a sensitivity covariate.

`local_ambient_temp_mean` is retained under its delivered SCADA name.  This audit
does not assert sensor placement or measurement semantics; it tests only
availability, fleet consistency, annual dynamics, and post-gate agreement with
the independently measured meteorological temperature.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_GATE = pd.Timestamp("2015-05-14 12:00:00")
LOCAL = "local_ambient_temp_mean"
INDEPENDENT = "met_temperature_mean"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "research_layers" / "health_twin_base_v1.csv.gz",
    )
    parser.add_argument(
        "--quality-gate",
        default=str(QUALITY_GATE),
        help="Approved start for comparison to independent meteorology.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiment_results" / "local_ambient_proxy_audit_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    quality_gate = pd.Timestamp(args.quality_gate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "per_turbine_summary.csv"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")

    data = pd.read_csv(
        args.input,
        usecols=["timestamp", "turbine_id", LOCAL, INDEPENDENT],
        parse_dates=["timestamp"],
    ).sort_values(["timestamp", "turbine_id"], kind="stable")
    rows = []
    for turbine_id, group in data.groupby("turbine_id", sort=True):
        local = group[LOCAL]
        paired = group.loc[group["timestamp"].ge(quality_gate), [LOCAL, INDEPENDENT]].dropna()
        difference = paired[LOCAL] - paired[INDEPENDENT]
        rows.append(
            {
                "turbine_id": turbine_id,
                "annual_rows": int(len(group)),
                "local_missing_pct": float(local.isna().mean() * 100.0),
                "local_min": float(local.min()),
                "local_max": float(local.max()),
                "local_std": float(local.std(ddof=0)),
                "post_gate_pair_rows": int(len(paired)),
                "post_gate_correlation_with_independent": float(paired.corr().iloc[0, 1]),
                "post_gate_median_local_minus_independent": float(difference.median()),
                "post_gate_mad_local_minus_independent": float((difference - difference.median()).abs().median()),
            }
        )
    per_turbine = pd.DataFrame(rows)
    per_turbine.to_csv(output, index=False)
    fleet = data.pivot(index="timestamp", columns="turbine_id", values=LOCAL)
    fleet_spread = pd.DataFrame(
        {
            "timestamp": fleet.index,
            "available_turbines": fleet.notna().sum(axis=1),
            "fleet_median": fleet.median(axis=1),
            "fleet_iqr": fleet.quantile(0.75, axis=1) - fleet.quantile(0.25, axis=1),
            "fleet_std": fleet.std(axis=1, ddof=0),
        }
    )
    spread_summary = {
        "timestamp_count": int(len(fleet_spread)),
        "median_available_turbines": float(fleet_spread["available_turbines"].median()),
        "pct_timestamps_all_16_available": float((fleet_spread["available_turbines"] == 16).mean() * 100.0),
        "median_fleet_iqr_c": float(fleet_spread["fleet_iqr"].median()),
        "p95_fleet_iqr_c": float(fleet_spread["fleet_iqr"].quantile(0.95)),
        "median_fleet_std_c": float(fleet_spread["fleet_std"].median()),
        "p95_fleet_std_c": float(fleet_spread["fleet_std"].quantile(0.95)),
    }
    pd.DataFrame([spread_summary]).to_csv(args.output_dir / "fleet_spread_summary.csv", index=False)
    data["month"] = data["timestamp"].dt.to_period("M").astype(str)
    monthly = data.groupby("month", sort=True)[LOCAL].agg(["count", "median", "std", "min", "max"]).reset_index()
    monthly.to_csv(args.output_dir / "local_ambient_monthly_summary.csv", index=False)
    aggregate = {
        "artifact": "local_ambient_proxy_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential research audit; no raw SCADA observations are redistributed",
        "input": str(args.input),
        "signal": LOCAL,
        "quality_gate_for_external_comparison": str(quality_gate),
        "per_turbine_median_post_gate_correlation": float(per_turbine["post_gate_correlation_with_independent"].median()),
        "per_turbine_min_post_gate_correlation": float(per_turbine["post_gate_correlation_with_independent"].min()),
        "per_turbine_median_local_minus_independent": float(per_turbine["post_gate_median_local_minus_independent"].median()),
        "fleet_spread": spread_summary,
        "decision": "Candidate only for a separately labelled full-year local-ambient sensitivity. OEM/SCADA documentation is required before calling this signal a validated external ambient measurement.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(native(aggregate), indent=2) + "\n", encoding="utf-8")
    report = [
        "# Turbine-local ambient proxy audit v1 — confidential local artifact",
        "",
        "## Decision",
        "",
        "`local_ambient_temp_mean` is an eligible **candidate covariate for a separately labelled full-year sensitivity**. It is not being renamed or treated as a verified external ambient instrument: the delivered backup has no OEM signal dictionary confirming its placement or measurement semantics.",
        "",
        "## Annual availability and post-gate comparison",
        "",
        markdown_table(per_turbine),
        "",
        "## Fleet-consistency screen",
        "",
        f"Across timestamps, the median number of available turbine-local readings is **{spread_summary['median_available_turbines']:.0f}**; all 16 are available at **{spread_summary['pct_timestamps_all_16_available']:.2f}%** of timestamps. The median across-turbine IQR is **{spread_summary['median_fleet_iqr_c']:.3f}°C** (95th percentile **{spread_summary['p95_fleet_iqr_c']:.3f}°C**).",
        "",
        f"After the independent-meteorology gate, the median turbine-wise correlation with `met_temperature_mean` is **{aggregate['per_turbine_median_post_gate_correlation']:.3f}** (minimum **{aggregate['per_turbine_min_post_gate_correlation']:.3f}**). This supports use as a proxy sensitivity covariate, not an assertion of sensor identity.",
        "",
        "## Boundary",
        "",
        "The full-year local-ambient model will be reported separately from the primary post-gate independent-meteorology protocol. Any apparent advantage could reflect true site conditions, local sensor bias, or target-adjacent instrumentation; without a data dictionary it cannot be used to make an equipment-specific causal claim.",
    ]
    (args.output_dir / "LOCAL_AMBIENT_PROXY_AUDIT_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
