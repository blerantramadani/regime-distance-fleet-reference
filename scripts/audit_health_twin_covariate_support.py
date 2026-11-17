#!/usr/bin/env python3
"""Audit chronological covariate support for the local health-twin benchmark.

This is a scope diagnostic, not an outlier filter.  Later records remain in the
test set even when their operating covariates fall outside the training range;
the resulting percentages make seasonal extrapolation visible in the report.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONITORING_START = pd.Timestamp("2015-05-14 12:00:00")
TRAIN_END = pd.Timestamp("2015-09-01 00:00:00")
VALID_END = pd.Timestamp("2015-11-01 00:00:00")
TEST_END = pd.Timestamp("2016-01-01 00:00:00")
CORE_COVARIATES = [
    "active_power_mean",
    "wind_speed_mean",
    "generator_speed_mean",
    "main_shaft_speed_mean",
    "pitch_a_mean",
    "pitch_b_mean",
    "pitch_c_mean",
    "met_temperature_mean",
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


def in_range_pct(value: pd.Series, lower: float, upper: float) -> float:
    valid = pd.to_numeric(value, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    return float((~valid.between(lower, upper)).mean() * 100.0)


def split_summary(train: pd.DataFrame, later: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    for column in CORE_COVARIATES:
        x_train = pd.to_numeric(train[column], errors="coerce").dropna()
        if x_train.empty:
            raise ValueError(f"No training values for {column}.")
        q005, q995 = x_train.quantile([0.005, 0.995]).to_numpy(dtype=float)
        rows.append(
            {
                "period": period,
                "covariate": column,
                "train_min": float(x_train.min()),
                "train_q005": float(q005),
                "train_q995": float(q995),
                "train_max": float(x_train.max()),
                "later_min": float(pd.to_numeric(later[column], errors="coerce").min()),
                "later_max": float(pd.to_numeric(later[column], errors="coerce").max()),
                "later_outside_train_minmax_pct": in_range_pct(later[column], float(x_train.min()), float(x_train.max())),
                "later_outside_train_q005_q995_pct": in_range_pct(later[column], float(q005), float(q995)),
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
    parser.add_argument("--monitoring-start", default=str(MONITORING_START))
    parser.add_argument("--train-end", default=str(TRAIN_END))
    parser.add_argument("--valid-end", default=str(VALID_END))
    parser.add_argument("--test-end", default=str(TEST_END))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiment_results" / "health_twin_covariate_support_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    monitoring_start = pd.Timestamp(args.monitoring_start)
    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)
    test_end = pd.Timestamp(args.test_end)
    if not monitoring_start < train_end < valid_end < test_end:
        raise ValueError("Require monitoring_start < train_end < valid_end < test_end.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "covariate_support.csv"
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{output_csv} exists; use --overwrite to replace it")

    usecols = [
        "timestamp",
        "has_temperature_record",
        "power_reference_end",
        *CORE_COVARIATES,
    ]
    data = pd.read_csv(args.input, usecols=usecols, parse_dates=["timestamp"])
    operational = (
        data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & data[CORE_COVARIATES].notna().all(axis=1)
    )
    data = data.loc[operational].copy()
    train = data.loc[data["timestamp"].ge(monitoring_start) & data["timestamp"].lt(train_end)]
    valid = data.loc[data["timestamp"].ge(train_end) & data["timestamp"].lt(valid_end)]
    test = data.loc[data["timestamp"].ge(valid_end) & data["timestamp"].lt(test_end)]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("At least one chronological period has no eligible records.")
    support = pd.DataFrame(split_summary(train, valid, "validation") + split_summary(train, test, "test"))
    support.to_csv(output_csv, index=False)
    test_core = support.loc[support["period"].eq("test")]
    temperature = test_core.loc[test_core["covariate"].eq("met_temperature_mean")].iloc[0]
    report = [
        "# Post-gate health-twin covariate-support audit v1 — confidential local artifact",
        "",
        "## Purpose",
        "",
        "The clean post-gate baseline is a natural temporal test, not an artificial in-range-only test. This audit therefore quantifies when validation or test covariates exceed their training support. It does not remove those records or revise the resulting metrics.",
        "",
        f"Eligible rows: train **{len(train):,}**, validation **{len(valid):,}**, test **{len(test):,}**. The training period is {monitoring_start:%Y-%m-%d} through {(train_end - pd.Timedelta(days=1)):%Y-%m-%d}.",
        "",
        "## Marginal support",
        "",
        markdown_table(support),
        "",
        "## Interpretation",
        "",
        f"Independent ambient temperature is outside the training min–max range for **{temperature['later_outside_train_minmax_pct']:.2f}%** of otherwise eligible November–December test records, and outside the central 99% training interval for **{temperature['later_outside_train_q005_q995_pct']:.2f}%**. The clean matched-peer result is therefore valid as a post-gate temporal stability result, but it must not be presented as an unrestricted seasonal normal-model generalisation result.",
        "",
        "A next full-year sensitivity may use the turbine-local `local_ambient_temp_mean` signal only after documenting its intended measurement semantics and ensuring it is not a target-derived proxy. It must be reported separately from the independent-meteorology protocol.",
    ]
    (args.output_dir / "COVARIATE_SUPPORT_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    provenance = {
        "artifact": "health_twin_covariate_support_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only confidential research audit; no raw SCADA observations are redistributed",
        "input": str(args.input),
        "splits": {
            "monitoring_start": str(monitoring_start),
            "train_end_exclusive": str(train_end),
            "validation_end_exclusive": str(valid_end),
            "test_end_exclusive": str(test_end),
        },
        "support_definition": "Per-covariate percentage outside the training min/max and outside the training 0.5th–99.5th percentile interval; no records are removed.",
        "interpretation_rule": "Report material seasonal support gaps alongside any performance or residual-stability statistic.",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(native(provenance), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
