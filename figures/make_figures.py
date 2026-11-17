#!/usr/bin/env python3
"""v0.13 (6 Nov 2026): comparison-channel label, unweighted/robust-scale wording, no in-figure notes or titles, larger fonts.
Create synchronized v0.7 scientific figures from frozen local aggregates.

Adapted from create_health_twin_energies_figures.py, without modifying the source.
Primary comparisons retain the original midpoint static median. A separate,
explicitly post-hoc figure shows harmonized lower-median sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


TARGET_ORDER = [
    "gear_oil_temp_mean",
    "gen_bearing_g_temp_mean",
    "gen_bearing_r_temp_mean",
    "main_bearing_temp_mean",
    "hydraulic_oil_temp_mean",
]
TARGET_LABELS = {
    "gear_oil_temp_mean": "Gear oil",
    "gen_bearing_g_temp_mean": "Generator bearing G",
    "gen_bearing_r_temp_mean": "Generator bearing R",
    "main_bearing_temp_mean": "Main bearing (exploratory)",
    "hydraulic_oil_temp_mean": "Hydraulic oil (comparison)",
}
PANEL_TITLES = {
    "gear_oil_temp_mean": "Gear oil",
    "gen_bearing_g_temp_mean": "Generator\nbearing G",
    "gen_bearing_r_temp_mean": "Generator\nbearing R",
    "main_bearing_temp_mean": "Main bearing\n(exploratory)",
    "hydraulic_oil_temp_mean": "Hydraulic oil\n(comparison)",
}
CONTRAST_ORDER = ["trim", "distance", "reliability", "end_to_end"]
CONTRAST_LABELS = {
    "trim": "Trim only",
    "distance": "Distance stage | same trim",
    "reliability": "Reliability | distance",
    "end_to_end": "End-to-end",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#6B7280",
    "dark": "#111827",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 13.5,
            "axes.labelsize": 13,
            "axes.edgecolor": "#374151",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": COLORS["dark"],
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for extension in ["png", "pdf"]:
        path = output_dir / f"{stem}.{extension}"
        metadata = None
        if extension == "pdf":
            metadata = {
                "Creator": "create_thermal_v07_figures.py",
                "CreationDate": None,
                "ModDate": None,
            }
        fig.savefig(
            path,
            dpi=320 if extension == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def add_box(ax, x, y, width, height, title, subtitle, facecolor) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#374151",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        weight="bold",
        fontsize=10,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=7.4,
        color="#4B5563",
    )


def add_arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color="#4B5563",
        )
    )


def figure_workflow(output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, 0.02, 0.36, 0.14, 0.30, "Fleet SCADA", "Completed intervals\nConfidential fleet", "#E8F3FA")
    add_box(ax, 0.20, 0.36, 0.14, 0.30, "Quality contract", "Meteorology gate · chronology", "#FFF3D6")
    add_box(ax, 0.38, 0.36, 0.14, 0.30, "Normal model", "Eligible finite residuals", "#E8F6F0")
    add_box(ax, 0.55, 0.25, 0.16, 0.23, "Fleet reference", "All peers · distance · leave-one-out", "#F3EAF5")
    add_box(ax, 0.76, 0.60, 0.18, 0.23, "Fleet-common state", "Eligible median · ≥4 turbines", "#FDECE6")
    add_box(ax, 0.76, 0.25, 0.18, 0.23, "Turbine-specific state", "Residual minus peer reference", "#E7F0FB")
    add_box(ax, 0.76, 0.035, 0.18, 0.16, "Frozen alert layers", "October calibration · Nov-Dec test", "#F3F4F6")
    add_arrow(ax, (0.16, 0.51), (0.20, 0.51))
    add_arrow(ax, (0.34, 0.51), (0.38, 0.51))
    add_arrow(ax, (0.52, 0.48), (0.55, 0.38))
    add_arrow(ax, (0.52, 0.62), (0.76, 0.72))
    add_arrow(ax, (0.71, 0.36), (0.76, 0.36))
    add_arrow(ax, (0.85, 0.25), (0.85, 0.201))
    ax.plot([0.946, 0.985, 0.985], [0.715, 0.715, 0.115], color="#4B5563", linewidth=1.3)
    add_arrow(ax, (0.985, 0.115), (0.946, 0.115))
    ax.text(0.635, 0.745, "Independent of specific-state scoreability", fontsize=7.5,
            color="#4B5563", ha="center", rotation=9)
    ax.text(
        0.5,
        0.95,
        "Quality-gated regime-distance fleet residual-state workflow",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.875,
        "No event label enters fitting or selection; the score is retrospective after each completed interval.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4B5563",
    )
    return save_figure(fig, output_dir, "fig01_method_workflow")


def figure_distance_forest(
    contrasts: pd.DataFrame,
    folds: list[str],
    title: str,
    stem: str,
    output_dir: Path,
) -> list[Path]:
    data = contrasts.loc[
        contrasts["contrast"].eq("distance")
        & contrasts["block_days"].eq(7)
        & contrasts["fold"].isin(folds)
    ].copy()
    x_min = float(np.floor(data["ci95_low_pct"].min() - 0.5))
    x_max = float(np.ceil(max(1.0, data["ci95_high_pct"].max() + 0.5)))
    fig, axes = plt.subplots(1, 5, figsize=(13, 5.0), sharex=True, sharey=True)
    for axis, target in zip(axes, TARGET_ORDER):
        subset = data.loc[data["target"].eq(target)].set_index("fold").reindex(folds).reset_index()
        if subset["observed_std_change_pct"].isna().any():
            raise ValueError(f"Missing fold for {target} in {stem}")
        y = np.arange(len(subset))
        effect = subset["observed_std_change_pct"].to_numpy(dtype=float)
        low = subset["ci95_low_pct"].to_numpy(dtype=float)
        high = subset["ci95_high_pct"].to_numpy(dtype=float)
        for index in range(len(subset)):
            color = COLORS["blue"] if high[index] < 0 else COLORS["orange"]
            axis.errorbar(
                effect[index],
                y[index],
                xerr=[[effect[index] - low[index]], [high[index] - effect[index]]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=4.8,
                linewidth=1.2,
            )
        axis.axvline(0, color="#111827", linestyle="--", linewidth=0.9)
        axis.set_title(PANEL_TITLES[target])
        axis.set_xlabel("Spread change (%)")
        axis.set_xlim(x_min, x_max)
        axis.set_xticks([-12, -8, -4, 0])
        axis.grid(axis="x", alpha=0.18)
        axis.invert_yaxis()
        axis.set_yticks(y)
        axis.set_yticklabels(subset["fold"])
    axes[0].set_ylabel("Chronological test fold")
    handles = [
        plt.Line2D([0], [0], marker="o", color=COLORS["blue"], linestyle="None", label="CI below zero"),
        plt.Line2D([0], [0], marker="o", color=COLORS["orange"], linestyle="None", label="CI includes or is above zero"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return save_figure(fig, output_dir, stem)


def figure_sequential_ablation(contrasts: pd.DataFrame, output_dir: Path) -> list[Path]:
    data = contrasts.loc[contrasts["block_days"].eq(7)].copy()
    summary = (
        data.groupby(["target", "contrast"], as_index=False)
        .agg(
            median_change=("observed_std_change_pct", "median"),
            folds=("fold", "size"),
            ci_below=("ci95_high_pct", lambda values: int((values < 0).sum())),
            ci_above=("ci95_low_pct", lambda values: int((values > 0).sum())),
        )
    )
    pivot = summary.pivot(index="target", columns="contrast", values="median_change").reindex(
        index=TARGET_ORDER, columns=CONTRAST_ORDER
    )
    values = pivot.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 0.5)
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        aspect="auto",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    ax.set_xticks(np.arange(len(CONTRAST_ORDER)))
    ax.set_xticklabels([CONTRAST_LABELS[item] for item in CONTRAST_ORDER], rotation=22, ha="right")
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticklabels([TARGET_LABELS[item] for item in TARGET_ORDER])
    ax.set_xlabel("Sequential ablation contrast")
    ax.set_ylabel("Thermal target role")
    for row_index, target in enumerate(TARGET_ORDER):
        for column_index, contrast in enumerate(CONTRAST_ORDER):
            record = summary.loc[
                summary["target"].eq(target) & summary["contrast"].eq(contrast)
            ].iloc[0]
            change = float(record["median_change"])
            support = f"↓ {int(record['ci_below'])}/{int(record['folds'])}   ↑ {int(record['ci_above'])}/{int(record['folds'])}"
            color = "white" if abs(change) > bound * 0.55 else COLORS["dark"]
            ax.text(
                column_index,
                row_index,
                f"{change:+.3f}%\n{support}",
                ha="center",
                va="center",
                fontsize=11,
                weight="bold",
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
    colorbar.set_label("Median spread change across four M folds (%)")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig03_sequential_ablation")


def figure_alert_rates(alerts: pd.DataFrame, output_dir: Path) -> list[Path]:
    alerts = alerts.set_index("target").reindex(TARGET_ORDER).reset_index()
    x = np.arange(len(alerts))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    specific = alerts["median_test_point_rate_pct"].to_numpy(dtype=float)
    common = alerts["common_point_rate_pct"].to_numpy(dtype=float)
    bars_specific = ax.bar(x - width / 2, specific, width, color=COLORS["blue"], label="Median turbine-specific")
    bars_common = ax.bar(x + width / 2, common, width, color=COLORS["orange"], label="Fleet-common")
    ax.axhline(5, color=COLORS["red"], linestyle="--", linewidth=1.0, label="Pre-specified 5% warning")
    ax.bar_label(bars_specific, fmt="%.2f%%", fontsize=10.5, padding=2)
    ax.bar_label(bars_common, fmt="%.2f%%", fontsize=10.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Point-exceedance rate, Nov–Dec (%)")
    ax.set_ylim(0, max(np.nanmax(common), np.nanmax(specific)) * 1.28)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig04_alert_rates")


def annotated_heatmap(ax, values, row_labels, column_labels, title, colorbar_label, vmax) -> None:
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value / max(vmax, 1e-12) > 0.58 else COLORS["dark"]
            ax.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=11.5, color=color, weight="bold")
    colorbar = plt.colorbar(image, ax=ax, shrink=0.80, pad=0.025)
    colorbar.set_label(colorbar_label)


def figure_synthetic_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> list[Path]:
    subset = sensitivity.loc[sensitivity["channel"].eq("turbine_specific")].copy()
    subset["detection_pct"] = subset["detection_fraction"] * 100.0
    row_order = [("step", 1), ("step", 6), ("step", 24), ("drift", 1), ("drift", 6), ("drift", 24)]
    values = np.full((len(row_order), 3), np.nan)
    for row_index, (shape, duration) in enumerate(row_order):
        current = subset.loc[
            subset["shape"].eq(shape) & subset["duration_hours"].eq(duration)
        ].set_index("absolute_amplitude_sigma")
        for column_index, amplitude in enumerate([1.0, 2.0, 3.0]):
            values[row_index, column_index] = current.loc[amplitude, "detection_pct"]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    annotated_heatmap(
        ax,
        values,
        [f"{shape.title()} · {duration} h" for shape, duration in row_order],
        ["1", "2", "3"],
        "",
        "Detected evaluable windows (%)",
        max(30, float(np.nanmax(values))),
    )
    ax.set_xlabel("Final shift (October robust scales)")
    ax.set_ylabel("Shift shape and duration")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig05_synthetic_sensitivity")


def figure_threshold_scale(thresholds: pd.DataFrame, output_dir: Path) -> list[Path]:
    specific = thresholds.loc[thresholds["channel"].eq("turbine_specific")].set_index("target").reindex(TARGET_ORDER)
    common = thresholds.loc[thresholds["channel"].eq("fleet_common")].set_index("target").reindex(TARGET_ORDER)
    x = np.arange(len(TARGET_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bars_specific = ax.bar(x - width / 2, specific["median_threshold_sigma"], width, color=COLORS["blue"], label="Turbine-specific median")
    bars_common = ax.bar(x + width / 2, common["median_threshold_sigma"], width, color=COLORS["orange"], label="Fleet-common")
    ax.bar_label(bars_specific, fmt="%.1f", fontsize=10.5, padding=2)
    ax.bar_label(bars_common, fmt="%.1f", fontsize=10.5, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGET_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Threshold (October robust scales)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig06_threshold_scale")


def figure_block_sensitivity(contrasts: pd.DataFrame, output_dir: Path) -> list[Path]:
    primary_targets = TARGET_ORDER[:3]
    data = contrasts.loc[
        contrasts["contrast"].eq("distance") & contrasts["target"].isin(primary_targets)
    ].copy()
    summary = (
        data.groupby(["target", "block_days"], as_index=False)
        .agg(ci_below=("ci95_high_pct", lambda values: int((values < 0).sum())))
    )
    pivot = summary.pivot(index="target", columns="block_days", values="ci_below").reindex(
        index=primary_targets, columns=[3, 7, 14]
    )
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["3 days", "7 days (primary)", "14 days"])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels([TARGET_LABELS[target] for target in primary_targets])
    ax.set_xlabel("Moving-block length")
    ax.set_ylabel("Primary channel")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{int(values[row, column])}/4 CI<0", ha="center", va="center", color="white" if values[row, column] >= 3 else COLORS["dark"], weight="bold")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.80, pad=0.025, ticks=[0, 1, 2, 3, 4])
    colorbar.set_label("M folds with interval below zero")
    fig.tight_layout()
    return save_figure(fig, output_dir, "figS02_block_sensitivity")


def figure_median_sensitivity(intervals: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Paired fold estimates; no pooling of M and L protocols or significance tests."""
    data = intervals.loc[intervals["block_days"].eq(7)].copy()
    fig, axes = plt.subplots(2, 5, figsize=(13, 7.8), sharex=True, sharey="row")
    protocols = [("independent", ["M1", "M2", "M3", "M4"]),
                 ("local_proxy", ["L1", "L2", "L3", "L4", "L5"])]
    variants = [("calendar_original_median", -0.12, COLORS["blue"], "s"),
                ("calendar_harmonized_median", 0.12, COLORS["orange"], "o")]
    # Use observed protocol labels rather than assuming an undocumented alias.
    local_labels = set(data["protocol"]) - {"independent"}
    assert len(local_labels) == 1, local_labels
    protocols[1] = (local_labels.pop(), protocols[1][1])
    xmin = np.floor(data.loc[data.variant.isin([v[0] for v in variants]), "ci_low_pct"].min()-0.5)
    xmax = np.ceil(data.loc[data.variant.isin([v[0] for v in variants]), "ci_high_pct"].max()+0.5)
    for row, (protocol, folds) in enumerate(protocols):
        for col, target in enumerate(TARGET_ORDER):
            ax = axes[row, col]
            for variant, offset, color, marker in variants:
                d = data.loc[data.protocol.eq(protocol) & data.target.eq(target) & data.variant.eq(variant)]
                assert not d.fold.duplicated().any()
                d = d.set_index("fold").reindex(folds)
                assert d.effect_pct.notna().all()
                e, lo, hi = (d[c].to_numpy() for c in ["effect_pct", "ci_low_pct", "ci_high_pct"])
                ax.errorbar(e, np.arange(len(folds))+offset, xerr=[e-lo, hi-e],
                            fmt=marker, color=color, capsize=2, markersize=4, linewidth=1.0)
            ax.axvline(0, color="#374151", linestyle="--", linewidth=.8)
            ax.grid(axis="x", alpha=.18)
            ax.set_yticks(np.arange(len(folds)), labels=folds)
            ax.set_ylim(len(folds)-.5, -.5)
            ax.set_xlim(xmin, xmax)
            ax.set_xticks([-15, -10, -5, 0])
            if row == 0:
                ax.set_title(PANEL_TITLES[target])
            else:
                ax.set_xlabel("Spread change (%)")
    axes[0, 0].set_ylabel("Primary independent-meteorology folds")
    axes[1, 0].set_ylabel("Local-proxy sensitivity folds")
    handles = [plt.Line2D([0], [0], marker="s", color=COLORS["blue"], label="Primary: midpoint-median comparator"),
               plt.Line2D([0], [0], marker="o", color=COLORS["orange"], label="Post hoc: lower-quantile comparator")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=[0,0,1,.93], h_pad=2)
    return save_figure(fig, output_dir, "figS03_median_convention_sensitivity")


def corrected_primary(primary: pd.DataFrame, m2: pd.DataFrame) -> pd.DataFrame:
    """Replace all 60 M2 CI pairs, preserving every original point estimate."""
    out = primary.copy()
    keys = ["target", "contrast", "block_days"]
    repair = m2.set_index(keys)
    assert len(repair) == 60 and not repair.index.duplicated().any()
    for i in out.index[out.fold.eq("M2")]:
        key = tuple(out.loc[i, keys])
        r = repair.loc[key]
        np.testing.assert_allclose(out.loc[i, "observed_std_change_pct"], r.effect_pct, atol=1e-10)
        out.loc[i, "ci95_low_pct"] = r.calendar_ci_low_pct
        out.loc[i, "ci95_high_pct"] = r.calendar_ci_high_pct
    pd.testing.assert_frame_equal(out.loc[~out.fold.eq("M2")], primary.loc[~primary.fold.eq("M2")])
    np.testing.assert_array_equal(out.observed_std_change_pct, primary.observed_std_change_pct)
    return out


def threshold_summary(ratios: pd.DataFrame) -> pd.DataFrame:
    return ratios.groupby(["channel", "target", "role"], as_index=False).agg(
        entities=("entity", "size"), median_threshold_sigma=("threshold_to_robust_scale", "median"),
        q25_threshold_sigma=("threshold_to_robust_scale", lambda s: s.quantile(.25)),
        q75_threshold_sigma=("threshold_to_robust_scale", lambda s: s.quantile(.75)),
        min_threshold_sigma=("threshold_to_robust_scale", "min"),
        max_threshold_sigma=("threshold_to_robust_scale", "max"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--correction-root", type=Path, default=Path("repairs_2026-09-18"))
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root, repair, synth = args.project_root, args.correction_root, args.synthetic_root
    input_paths = {
        "primary_original": root / "data/experiment_results/health_twin_trim_distance_ablation_independent_v1/mechanism_contrasts.csv",
        "sensitivity_original": root / "data/experiment_results/health_twin_trim_distance_ablation_local_v1/mechanism_contrasts.csv",
        "alerts_original": root / "data/experiment_results/health_twin_all_peer_alert_stability_v2/alert_target_summary.csv",
        "synthetic_original": root / "data/experiment_results/health_twin_all_peer_synthetic_alert_sensitivity_v2/synthetic_alert_aggregate_summary.csv",
        "thresholds_original": root / "data/experiment_results/health_twin_all_peer_synthetic_alert_sensitivity_v2/threshold_scale_target_summary.csv",
        "mechanism_M2_calendar": repair / "mechanism_M2/intervals.csv",
        "rolling_calendar_and_medians": repair / "rolling/intervals.csv",
        "common_definition": repair / "common/common_summary.csv",
        "synthetic_aligned": synth / "synthetic_alert_aggregate_summary.csv",
        "threshold_ratios_aligned": synth / "threshold_scale_ratio.csv",
    }
    for label, p in input_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")
    hashes_before = {k: sha256_file(p) for k,p in input_paths.items()}
    primary = corrected_primary(pd.read_csv(input_paths["primary_original"]), pd.read_csv(input_paths["mechanism_M2_calendar"]))
    sensitivity = pd.read_csv(input_paths["sensitivity_original"])
    rolling = pd.read_csv(input_paths["rolling_calendar_and_medians"])
    # Cross-check the plotted primary distance interval data against independent rerun.
    for d in primary.loc[primary.contrast.eq("distance")].itertuples():
        row = rolling.loc[rolling.fold.eq(d.fold) & rolling.target.eq(d.target) & rolling.block_days.eq(d.block_days) & rolling.variant.eq("calendar_original_median")]
        assert len(row) == 1
        np.testing.assert_allclose([d.observed_std_change_pct, d.ci95_low_pct, d.ci95_high_pct], row[["effect_pct", "ci_low_pct", "ci_high_pct"]].iloc[0].to_numpy(), atol=1e-9)
    alerts = pd.read_csv(input_paths["alerts_original"])
    common = pd.read_csv(input_paths["common_definition"])
    common = common.loc[common.variant.eq("eligible_min4") & common.period.eq("test")].set_index("target")
    assert len(common) == 5 and common.min_turbines.min() >= 4
    alert_plot = alerts[["target", "median_test_point_rate_pct"]].copy()
    alert_plot["common_point_rate_pct"] = alert_plot.target.map(common.point_rate_pct)
    assert alert_plot.common_point_rate_pct.notna().all()
    synthetic = pd.read_csv(input_paths["synthetic_aligned"])
    oldsynthetic = pd.read_csv(input_paths["synthetic_original"])
    pd.testing.assert_frame_equal(synthetic.loc[synthetic.channel.eq("turbine_specific")].reset_index(drop=True), oldsynthetic.loc[oldsynthetic.channel.eq("turbine_specific")].reset_index(drop=True))
    thresholds = threshold_summary(pd.read_csv(input_paths["threshold_ratios_aligned"]))
    oldthresholds = pd.read_csv(input_paths["thresholds_original"])
    tcols = ["target", "median_threshold_sigma"]
    pd.testing.assert_frame_equal(thresholds.loc[thresholds.channel.eq("turbine_specific"), tcols].reset_index(drop=True), oldthresholds.loc[oldthresholds.channel.eq("turbine_specific"), tcols].reset_index(drop=True))
    set_style()
    outputs = []
    outputs += figure_distance_forest(primary, ["M1", "M2", "M3", "M4"], "Primary distance-stage contrast: original static comparator", "fig02_primary_distance_forest", args.output_dir)
    outputs += figure_sequential_ablation(primary, args.output_dir)
    outputs += figure_alert_rates(alert_plot, args.output_dir)
    outputs += figure_synthetic_sensitivity(synthetic, args.output_dir)
    outputs += figure_threshold_scale(thresholds, args.output_dir)
    outputs += figure_distance_forest(sensitivity, ["L1", "L2", "L3", "L4", "L5"], "Local-proxy distance-stage sensitivity: original static comparator", "figS01_local_distance_forest", args.output_dir)
    outputs += figure_block_sensitivity(primary, args.output_dir)
    outputs += figure_median_sensitivity(rolling, args.output_dir)
    # Save compact plot data, excluding obsolete M2 bootstrap diagnostics.
    cols = ["protocol", "fold", "target", "contrast", "block_days", "observed_std_change_pct", "ci95_low_pct", "ci95_high_pct"]
    plot_data = {"plot_primary_contrasts.csv": primary[cols],
                 "plot_local_contrasts.csv": sensitivity[cols],
                 "plot_alert_rates.csv": alert_plot,
                 "plot_threshold_scale.csv": thresholds,
                 "plot_synthetic_sensitivity.csv": synthetic.loc[synthetic.channel.eq("turbine_specific")],
                 "plot_median_convention_sensitivity.csv": rolling.loc[rolling.block_days.eq(7) & rolling.variant.isin(["calendar_original_median", "calendar_harmonized_median"])]}
    for name, frame in plot_data.items():
        p = args.output_dir / name
        frame.to_csv(p, index=False)
        outputs.append(p)
    captions = [
        ("Figure 1", "fig01_method_workflow", "Quality-gated workflow. The common state is the median over all operating-eligible finite residuals, requiring at least four turbines and independent of turbine-specific scoreability. Common and specific states have separate frozen alert calibration."),
        ("Figure 2", "fig02_primary_distance_forest", "Primary M-fold distance-stage spread changes with the original midpoint static comparator. The distance reference uses a lower weighted median, so this stage comparison includes a median-convention difference. Bars are fold-specific 95% seven-day calendar-block bootstrap intervals; all M2 intervals use the corrected complete calendar. Negative values favour the distance stage."),
        ("Figure 3", "fig03_sequential_ablation", "Common-support sequential ablation across four primary M folds: trim, distance stage, incremental reliability, and end-to-end. Cells show median point changes and the number of fold intervals wholly below/above zero. All four M2 contrast intervals are calendar-corrected. The original midpoint static comparator remains primary; this is a developmental audit."),
        ("Figure 4", "fig04_alert_rates", "Later-period frozen-threshold point-exceedance activity. Turbine-specific rates remain unchanged. Common states use operating-eligible finite residuals and a four-turbine minimum; hydraulic common activity is 6.4231396%, with the other four common rates unchanged."),
        ("Figure 5", "fig05_synthetic_sensitivity", "Frozen-rule synthetic turbine-specific stress sensitivity by final magnitude, duration, and shape. Values are unchanged after aligned synthetic rerun; perturbations are not physical fault severities. Percentages aggregate target roles and polarities."),
        ("Figure 6", "fig06_threshold_scale", "Frozen 99.5th-percentile thresholds expressed in calibration robust-scale units. Common ratios are recalculated for the eligible-residual, four-turbine-minimum state. Turbine-specific ratios remain unchanged."),
        ("Figure S1", "figS01_local_distance_forest", "Five separate local-ambient-proxy sensitivity folds with original midpoint static comparator and lower weighted distance median. Numerical data are unchanged; bars show fold-specific 95% seven-day calendar-block bootstrap intervals. M and L folds are not pooled."),
        ("Figure S2", "figS02_block_sensitivity", "Counts of four primary M-fold distance-stage intervals below zero under 3-, 7-, and 14-day calendar blocks; M2 uses the corrected calendar for every length. Original point estimates and midpoint static comparator are retained."),
        ("Figure S3", "figS03_median_convention_sensitivity", "Explicitly post-hoc lower-median convention sensitivity alongside the original midpoint-static comparator. Both retain the identical lower weighted distance reference, residuals, trimming, score support, and bandwidth. Fold-specific 95% seven-day calendar-block intervals are shown. Primary M and local-proxy L protocols remain separate; the sensitivity does not replace or reselect the headline comparison."),
    ]
    checks = {
        "all_60_M2_interval_pairs_replaced": True,
        "non_M2_primary_records_unchanged": True,
        "primary_point_estimates_unchanged": True,
        "distance_intervals_match_independent_calendar_rerun": True,
        "specific_synthetic_aggregate_unchanged": True,
        "specific_threshold_medians_unchanged": True,
        "common_state_requires_at_least_four_eligible_finite_residuals": True,
        "source_inputs_unchanged_during_generation": hashes_before == {k: sha256_file(p) for k,p in input_paths.items()},
        "hydraulic_common_rate_pct": float(common.loc["hydraulic_oil_temp_mean", "point_rate_pct"]),
    }
    assert checks["source_inputs_unchanged_during_generation"]
    provenance = {
        "artifact": "Bogdanci_thermal_manuscript_v0.7_figures", "confidential": True,
        "generator": Path(__file__).name, "generator_sha256": sha256_file(Path(__file__)),
        "runtime": {"python": sys.version, "matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "design": "Original midpoint comparator primary; complete-calendar M2 bootstrap; common eligible-min4; lower-median sensitivity post hoc",
        "inputs": {k: {"path": str(p), "sha256": hashes_before[k]} for k,p in input_paths.items()},
        "outputs": {p.name: sha256_file(p) for p in outputs},
        "checks": checks, "captions": [{"label": a, "stem": b, "caption": c} for a,b,c in captions],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")
    lines = ["# Thermal manuscript v0.7 figure package", "", "Nine figures, each supplied as a 320-dpi PNG and vector PDF, regenerated from local aggregate artifacts. Original source files remain unchanged. No raw SCADA rows, real turbine identifiers, coordinates, or turbine maps are included in the figures.", "", "The primary comparator remains the original midpoint static median. Calendar repair changes M2 intervals only. The lower-median convention comparison is explicitly post-hoc sensitivity. The common state uses all operating-eligible finite residuals, requiring at least four turbines independently of specific-state scoreability.", "", "## Captions", ""]
    for label, stem, caption in captions:
        lines += [f"- **{label}** (`{stem}`). {caption}"]
    lines += ["", "## Reproduction", "", "Run `/usr/bin/python3 create_thermal_v07_figures.py` with `MPLCONFIGDIR` pointing to a writable temporary directory. Inputs are accepted through `--project-root`, `--correction-root`, and `--synthetic-root`; figures and compact plotted data are written to `--output-dir`. Existing package outputs in that directory are overwritten; no input artifact is modified.", "", "## Verification", "", "Generation asserts that all 60 M2 contrast/target/block-length interval pairs are repaired, other primary rows and all primary point effects remain unchanged, and plotted distance intervals agree with the independent calendar rerun. Specific synthetic aggregates and threshold medians must match the original inputs. See `visual_QA.md` for final rendering inspection.", "", "## Files and hashes", ""]
    for p in outputs:
        lines += [f"- `{p.name}` - SHA-256 `{sha256_file(p)}`"]
    lines += ["", "## Aggregate inputs", ""]
    for k,p in input_paths.items():
        lines += [f"- `{k}`: `{p}` - SHA-256 `{hashes_before[k]}`"]
    (args.output_dir / "FIGURE_MANIFEST.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
