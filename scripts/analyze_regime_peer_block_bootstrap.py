#!/usr/bin/env python3
"""Moving-block bootstrap for locked matched-peer test comparisons.

The calculation resamples contiguous calendar-day blocks rather than treating
10-minute turbine observations as independent. It reads already locked test
scores and never re-fits a model or re-selects peer hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


COMPARATORS = ["static_median", "reliability_all"]
PROPOSED = "regime_matched"


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
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


def daily_sufficient_statistics(group: pd.DataFrame, methods: list[str]):
    aggregate = (
        group.groupby(["day", "method"], as_index=False)["specific_residual"]
        .agg(n="size", value_sum="sum", value_sumsq=lambda x: float(np.dot(x, x)))
    )
    days = np.array(sorted(aggregate["day"].unique()))
    cube = np.zeros((len(days), len(methods), 3), dtype=float)
    for method_index, method in enumerate(methods):
        aligned = aggregate.loc[aggregate["method"].eq(method)].set_index("day").reindex(days)
        if aligned[["n", "value_sum", "value_sumsq"]].isna().any().any():
            raise ValueError(f"Method {method} is missing at least one target-day.")
        cube[:, method_index, :] = aligned[["n", "value_sum", "value_sumsq"]].to_numpy(dtype=float)
    return days, cube


def std_from_stats(stats: np.ndarray) -> np.ndarray:
    variance = stats[:, 2] / stats[:, 0] - (stats[:, 1] / stats[:, 0]) ** 2
    return np.sqrt(np.maximum(variance, 0.0))


def bootstrap_target(group, block_days, iterations, rng):
    methods = [*COMPARATORS, PROPOSED]
    days, cube = daily_sufficient_statistics(group, methods)
    if len(days) < block_days:
        raise ValueError(f"Only {len(days)} test days; block length is {block_days}.")
    starts = np.arange(len(days) - block_days + 1)
    draws = {comparator: np.empty(iterations, dtype=float) for comparator in COMPARATORS}
    for iteration in range(iterations):
        selected = []
        while len(selected) < len(days):
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block_days))
        stats = cube[np.asarray(selected[: len(days)], dtype=int)].sum(axis=0)
        standard_deviation = std_from_stats(stats)
        for comparator_index, comparator in enumerate(COMPARATORS):
            draws[comparator][iteration] = (
                standard_deviation[-1] / standard_deviation[comparator_index] - 1.0
            ) * 100.0
    observed_std = std_from_stats(cube.sum(axis=0))
    rows = []
    for comparator_index, comparator in enumerate(COMPARATORS):
        values = draws[comparator]
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        observed = (observed_std[-1] / observed_std[comparator_index] - 1.0) * 100.0
        rows.append(
            {
                "comparator": comparator,
                "test_days": int(len(days)),
                "block_days": int(block_days),
                "bootstrap_iterations": int(iterations),
                "observed_std_change_pct": float(observed),
                "bootstrap_median_std_change_pct": float(median),
                "ci95_low_pct": float(low),
                "ci95_high_pct": float(high),
                "bootstrap_probability_improvement": float(np.mean(values < 0.0)),
                "directionally_robust_improvement": bool(high < 0.0),
                "directionally_robust_worsening": bool(low > 0.0),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_file = args.output_dir / "block_bootstrap_comparison.csv"
    if result_file.exists() and not args.overwrite:
        raise FileExistsError(f"{result_file} exists; use --overwrite to replace it")
    if args.block_days < 1 or args.iterations < 100:
        raise ValueError("block-days must be >=1 and iterations must be >=100.")

    scores = pd.read_csv(
        args.input,
        usecols=["target", "timestamp", "method", "specific_residual"],
        parse_dates=["timestamp"],
    )
    scores = scores.loc[scores["method"].isin([*COMPARATORS, PROPOSED])].copy()
    scores["day"] = scores["timestamp"].dt.floor("D")
    rng = np.random.default_rng(args.seed)
    rows = []
    for target, group in scores.groupby("target", sort=True):
        for row in bootstrap_target(group, args.block_days, args.iterations, rng):
            rows.append({"target": target, **row})
    results = pd.DataFrame(rows)
    results.to_csv(result_file, index=False)

    static = results.loc[results["comparator"].eq("static_median")].copy()
    static["decision"] = np.select(
        [static["directionally_robust_improvement"], static["directionally_robust_worsening"]],
        ["robust lower spread", "robust higher spread"],
        default="uncertain direction",
    )
    report = [
        f"# Seven-day block-bootstrap audit: {args.label}",
        "",
        "This local analysis resamples contiguous seven-day blocks from the already locked test period. It does not re-fit the normal-behaviour model, reselect peer parameters, or treat individual 10-minute turbine rows as independent replicates.",
        "",
        "## Regime-matched versus static median",
        "",
        markdown_table(
            static,
            [
                "target",
                "observed_std_change_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "bootstrap_probability_improvement",
                "decision",
            ],
        ),
        "",
        "Negative change means lower turbine-specific residual spread for the regime-matched method. These intervals are a dependence-aware sensitivity analysis, not a substitute for rolling-origin or external validation. They are unadjusted across the five predeclared targets.",
        "",
        "## All locked controls",
        "",
        markdown_table(
            results,
            [
                "target",
                "comparator",
                "observed_std_change_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "directionally_robust_improvement",
                "directionally_robust_worsening",
            ],
        ),
    ]
    (args.output_dir / "BLOCK_BOOTSTRAP_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    provenance = {
        "artifact": "locked_test_moving_block_bootstrap",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confidentiality": "local-only aggregate analysis",
        "label": args.label,
        "input": str(args.input),
        "methods": [*COMPARATORS, PROPOSED],
        "block_unit": "calendar day",
        "block_days": args.block_days,
        "iterations": args.iterations,
        "seed": args.seed,
        "selection": "none; all model and peer choices were already locked before this analysis",
        "limits": [
            "Post-hoc uncertainty analysis of one locked test period.",
            "Confidence intervals are not adjusted across the five targets.",
            "No fault or maintenance labels are available.",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(native(provenance), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
