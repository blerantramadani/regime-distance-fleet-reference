"""Checks behind three statements in the F2 text (descriptive, run after the F2 results).

1. Two-way bootstrap (turbines x injection dates) of the F detection differences at the 99.5th-percentile threshold.
2. Matching on the October persistent-alert rate, read off the F2 2015 curves.
3. Whether excess detection changes any matched-rate reading of F2 2015.
Writes to ./results_2015_extra.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
F_RESULTS = ROOT.parents[1] / "posthoc_2026-10-30" / "detection" / "results"
OUT = ROOT / "results_2015_extra"
PRIMARY = {"gear_oil_temp_mean": "gear oil", "gen_bearing_g_temp_mean": "generator bearing G", "gen_bearing_r_temp_mean": "generator bearing R"}
REFS = ["distance", "static_trim", "fleet_median_residual"]


def two_way_bootstrap(seed=20261104, draws=2000):
    ev = pd.read_csv(F_RESULTS / "detection_events.csv.gz",
                     usecols=["target", "turbine_id", "anchor", "q995_evaluable", "q995_detected_distance", "q995_detected_static_trim"])
    rng = np.random.default_rng(seed)
    rows = []
    for target, name in PRIMARY.items():
        x = ev[(ev.target == target) & ev.q995_evaluable].copy()
        x["d"] = x.q995_detected_distance.astype(float) - x.q995_detected_static_trim.astype(float)
        turbines, anchors = sorted(x.turbine_id.unique()), sorted(x.anchor.unique())
        g = x.groupby(["turbine_id", "anchor"]).d.agg(["sum", "count"])
        s = g["sum"].unstack(fill_value=0).reindex(index=turbines, columns=anchors, fill_value=0).to_numpy()
        c = g["count"].unstack(fill_value=0).reindex(index=turbines, columns=anchors, fill_value=0).to_numpy()
        out = {"turbine": [], "two_way": []}
        for _ in range(draws):
            ti, ai = rng.integers(0, len(turbines), len(turbines)), rng.integers(0, len(anchors), len(anchors))
            for kind, (tt, aa) in {"turbine": (ti, np.arange(len(anchors))), "two_way": (ti, ai)}.items():
                out[kind].append(100 * s[np.ix_(tt, aa)].sum() / c[np.ix_(tt, aa)].sum())
        row = {"target": name, "difference_pp": 100 * x.d.mean()}
        for kind, vals in out.items():
            row[f"{kind}_ci_low"], row[f"{kind}_ci_high"] = np.percentile(vals, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def october_matching(rates=(0.05, 0.1, 0.2)):
    curves = pd.read_csv(ROOT / "results_2015" / "f2_curves.csv")
    rows = []
    for target, name in PRIMARY.items():
        for rate in rates:
            for ref in REFS:
                h = curves[(curves.target == target) & (curves.reference == ref) & (curves.october_alarm_rate_pct > 0)]
                h = h.sort_values("october_alarm_rate_pct")
                x = np.log10(h.october_alarm_rate_pct.to_numpy())
                ok = x.min() <= np.log10(rate) <= x.max()
                rows.append({"target": name, "october_alert_rate_pct": rate, "reference": ref,
                             "test_alert_rate_pct": np.interp(np.log10(rate), x, h.test_alarm_rate_pct) if ok else np.nan,
                             "detection_pct": np.interp(np.log10(rate), x, h.detection_pct) if ok else np.nan})
    return pd.DataFrame(rows)


def excess_agreement():
    c = pd.read_csv(ROOT / "results_2015" / "f2_matched_contrasts.csv")
    p = c.pivot_table(index=["target", "comparator", "alarm_rate_pct"], columns="measure", values="reading", aggfunc="first")
    return pd.DataFrame({"comparisons": [len(p)], "readings_that_differ": [int((p.detection != p.excess).sum())]})


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    for name, frame in [("two_way_bootstrap_q995.csv", two_way_bootstrap()), ("october_rate_matching.csv", october_matching()),
                        ("excess_vs_detection_readings.csv", excess_agreement())]:
        frame.to_csv(OUT / name, index=False)
        print(name)
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
