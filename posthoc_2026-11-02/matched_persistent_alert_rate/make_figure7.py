"""Figure 7: detection against the persistent-alert rate for the three primary channels, 2015 (F2 part A)."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT
curves = pd.read_csv(ROOT / "results_2015" / "f2_curves.csv")

PANELS = [("gear_oil_temp_mean", "Gear oil"), ("gen_bearing_g_temp_mean", "Generator bearing G"), ("gen_bearing_r_temp_mean", "Generator bearing R")]
STYLE = {  # colour, line style, marker, label
    "distance": ("#2a78d6", "-", "o", "Distance reference"),
    "static_trim": ("#eb6834", "--", "s", "Unweighted leave-one-out"),
    "fleet_median_residual": ("#1baf7a", ":", "^", "Fleet median of residuals (C1)"),
}
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e2"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
for ax, (target, title) in zip(axes, PANELS):
    g = curves[curves.target == target]
    for rate in [0.1, 0.2, 0.5, 1.0]:
        ax.axvline(rate, color=GRID, lw=1, zorder=0)
    for ref, (colour, ls, marker, label) in STYLE.items():
        h = g[g.reference == ref].sort_values("test_alarm_rate_pct")
        ax.plot(h.test_alarm_rate_pct, h.detection_pct, ls=ls, color=colour, lw=1.6, marker=marker, ms=3.5,
                markeredgecolor="white", markeredgewidth=0.5, label=label, zorder=3)
        cal = h[(h.q - 0.995).abs() < 1e-9]
        ax.plot(cal.test_alarm_rate_pct, cal.detection_pct, marker=marker, ms=8, mfc="none", mec=colour, mew=1.4,
                ls="none", zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xlim(0.02, 6)
    ax.set_ylim(0, 80)
    ax.set_title(title, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Detected shifts (% of windows)")
fig.supxlabel("Persistent-alert rate without injection, November–December (% of scored rows, log scale)", fontsize=9, color=INK, y=0.04)
handles, labels = axes[0].get_legend_handles_labels()
ring = plt.Line2D([], [], marker="o", ms=8, mfc="none", mec=MUTED, mew=1.4, ls="none")
fig.legend(handles + [ring], labels + ["99.5th-percentile threshold"], loc="upper center", ncol=2, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 0.86))
for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"Figure7_detection_vs_alert_rate.{ext}", dpi=300, facecolor="white")
print("written", OUT)
