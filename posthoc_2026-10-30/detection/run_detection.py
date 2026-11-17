"""Detection of synthetic shifts at matched alert rates (see PROTOCOL.md).

Post hoc. Reuses the frozen alert chronology of health_twin_all_peer_alert_stability_v2
and the specific states of ../fleet_median_baselines. Writes only to ./results.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "fleet_median_baselines" / "run_chesterman.py"
spec = importlib.util.spec_from_file_location("run_chesterman", BASE)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

from run_health_twin_alert_stability_v1 import (  # noqa: E402
    CALIBRATION_END, MONITORING_START, TEST_END, TRAIN_END, TUNING_END)
from run_health_twin_baseline import AMBIENT_SOURCES, SOURCE_COLUMNS, TARGETS, make_feature_matrix, predict  # noqa: E402
from run_health_twin_regime_peer_v2 import regime_matrix, residual_frame  # noqa: E402
from run_health_twin_rolling_ablation_v1 import fit_window  # noqa: E402
from run_health_twin_trim_distance_ablation import validation_bandwidth_grid  # noqa: E402

SAVED = rc.PROJECT / "data/experiment_results/health_twin_all_peer_alert_stability_v2"
REFS = ["distance", "static_trim", "fleet_median_residual", "static_no_trim", "fleet_median_raw"]
PRIMARY_REFS = REFS[:3]
SELF_INCLUSIVE = {"fleet_median_residual": "residual", "fleet_median_raw": "raw"}
ANCHORS = pd.date_range("2015-11-02", "2015-12-30", freq="2D")
PAPER_ANCHORS = set(pd.to_datetime(["2015-11-08", "2015-11-22", "2015-12-06", "2015-12-20"]))
DURATIONS = [1, 6, 24]
SHAPES = ["step", "drift"]
AMPLITUDES = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
MAX_GAP = pd.Timedelta(minutes=20).value
PERSISTENCE = 3
MIN_CALIBRATION_ROWS = 200
TOL = 1e-6  # see the amendment in PROTOCOL.md
VARIANTS = [("q995", 0.995, "october"), ("q990", 0.99, "october"), ("q975", 0.975, "october"), ("oracle995", 0.995, "test")]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(frame, times, turbines):
    m = np.full((len(times), len(turbines)), np.nan)
    m[times.get_indexer(frame.timestamp), pd.Index(turbines).get_indexer(frame.turbine_id)] = frame.residual.to_numpy()
    return m


def persistent(flags, contiguous):
    """True if three consecutive point flags occur on contiguous scored rows (20-min rule)."""
    if len(flags) < PERSISTENCE:
        return False
    run = flags[2:] & flags[1:-1] & flags[:-2] & contiguous[2:] & contiguous[1:-1]
    return bool(run.any())


def series_persistent(flags, contiguous):
    out = np.zeros(len(flags), dtype=bool)
    if len(flags) >= PERSISTENCE:
        out[2:] = flags[2:] & flags[1:-1] & flags[:-2] & contiguous[2:] & contiguous[1:-1]
    return out


def thresholds(cal_spec, cal_valid, test_spec, test_valid, turbines, quantile, source):
    rows = {}
    for ref in REFS:
        centre = np.full(len(turbines), np.nan)
        limit = np.full(len(turbines), np.nan)
        for i in range(len(turbines)):
            cal = cal_spec[ref][cal_valid[:, i], i]
            if len(cal) < MIN_CALIBRATION_ROWS:
                raise ValueError(f"Turbine {turbines[i]} has fewer than {MIN_CALIBRATION_ROWS} calibration rows")
            centre[i] = np.median(cal)
            basis = cal if source == "october" else test_spec[ref][test_valid[:, i], i]
            limit[i] = np.quantile(np.abs(basis - centre[i]), quantile)
        rows[ref] = (centre, limit)
    return rows


def main():
    out = ROOT / "results"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    path = rc.PROJECT / "data/research_layers/health_twin_base_v1.csv.gz"
    manifest = json.loads(path.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(path)
    assert digest == manifest["output_sha256"]

    ambient = AMBIENT_SOURCES["independent_meteorology"]["column"]
    data = pd.read_csv(path, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    turbine_id = data["turbine_id"].astype(str).to_numpy()
    features, _ = make_feature_matrix(data, turbines, ambient)
    descriptor, _ = regime_matrix(data, ambient)
    eligible = (
        data["timestamp"].ge(MONITORING_START)
        & data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(descriptor).all(axis=1)
    ).to_numpy()
    time = data["timestamp"]
    train = (time.ge(MONITORING_START) & time.lt(TRAIN_END)).to_numpy()
    tuning = (time.ge(TRAIN_END) & time.lt(TUNING_END)).to_numpy()
    calibration = (time.ge(TUNING_END) & time.lt(CALIBRATION_END)).to_numpy()
    test = (time.ge(CALIBRATION_END) & time.lt(TEST_END)).to_numpy()
    mean = np.mean(descriptor[eligible & train], axis=0)
    scale = np.std(descriptor[eligible & train], axis=0)
    descriptor = (descriptor - mean) / np.where(scale > 1e-8, scale, 1.0)

    saved_config = pd.read_csv(SAVED / "selected_configuration.csv").set_index("target")
    saved_thresholds = pd.read_csv(SAVED / "specific_thresholds.csv")
    flag_cols = ["target", "timestamp", "turbine_id", "specific_residual", "point_flag"]
    saved_states = pd.read_csv(SAVED / "calibration_alert_states.csv.gz", usecols=flag_cols, parse_dates=["timestamp"])
    saved_test = pd.read_csv(SAVED / "test_alert_states.csv.gz", usecols=flag_cols, parse_dates=["timestamp"])

    events, background, checks = [], [], {}
    for ki, target in enumerate(TARGETS):
        y = data[target].to_numpy(dtype=float)
        usable = eligible & np.isfinite(y)
        fitted = fit_window(features, y, turbine_id, usable, train, tuning, calibration)
        assert fitted["alpha"] == float(saved_config.loc[target, "normal_model_alpha"])
        scales = fitted["residual_scales"]
        frames = {}
        for name, period in [("tuning", tuning), ("calibration", calibration), ("test", test)]:
            mask = fitted["valid_mask"] if name == "tuning" else usable & period
            frames[name] = residual_frame(data, y[mask] - predict(fitted["model"], features[mask]), mask, descriptor)
            frames[name + "_raw"] = residual_frame(data, y[mask], mask, descriptor)
            assert (frames[name][["timestamp", "turbine_id"]].to_numpy() == frames[name + "_raw"][["timestamp", "turbine_id"]].to_numpy()).all()
        grid = pd.DataFrame(validation_bandwidth_grid(frames["tuning"], target, scales))
        bandwidth = float(grid.sort_values(["specific_median_abs", "specific_std", "distance_bandwidth"], kind="stable").iloc[0]["distance_bandwidth"])
        assert bandwidth == float(saved_config.loc[target, "selected_distance_bandwidth"])

        cal_times, cal_valid, cal_spec = rc.states(frames["calibration"], frames["calibration_raw"], scales, bandwidth, turbines)
        test_times, test_valid, test_spec = rc.states(frames["test"], frames["test_raw"], scales, bandwidth, turbines)

        # Reproduction of the saved distance states, thresholds and point flags.
        primary = thresholds(cal_spec, cal_valid, test_spec, test_valid, turbines, 0.995, "october")
        centre, limit = primary["distance"]
        flag_mismatch, max_diff = {}, {}
        for label, times_, valid_, spec_, saved_ in [("calibration", cal_times, cal_valid, cal_spec, saved_states),
                                                    ("test", test_times, test_valid, test_spec, saved_test)]:
            ti, tj = np.where(valid_)
            mine = pd.DataFrame({"timestamp": times_.to_numpy()[ti], "turbine_id": np.array(turbines)[tj],
                                 "specific_residual": spec_["distance"][ti, tj],
                                 "flag": np.abs(spec_["distance"][ti, tj] - centre[tj]) > limit[tj]})
            merged = mine.merge(saved_[saved_.target == target], on=["timestamp", "turbine_id"], how="outer",
                                indicator=True, suffixes=("", "_saved"))
            assert (merged["_merge"] == "both").all(), f"{target}: {label} rows differ"
            np.testing.assert_allclose(merged.specific_residual, merged.specific_residual_saved, atol=TOL, rtol=0)
            max_diff[label] = float(np.abs(merged.specific_residual - merged.specific_residual_saved).max())
            flag_mismatch[label] = int((merged.flag != merged.point_flag.astype(bool)).sum())
        st = saved_thresholds[saved_thresholds.target == target].set_index("turbine_id").loc[turbines]
        np.testing.assert_allclose(centre, st.state_centre, atol=TOL, rtol=0)
        np.testing.assert_allclose(limit, st.absolute_deviation_threshold, atol=TOL, rtol=0)
        checks[target] = {"alpha": fitted["alpha"], "bandwidth": bandwidth, "calibration_rows": int(cal_valid.sum()),
                          "test_rows": int(test_valid.sum()), "max_state_difference": max_diff,
                          "max_threshold_difference": float(np.abs(limit - st.absolute_deviation_threshold.to_numpy()).max()),
                          "point_flag_mismatches": flag_mismatch}
        print(f"{target}: reproduction {checks[target]}", flush=True)

        mats = {"residual": matrix(frames["test"], test_times, turbines), "raw": matrix(frames["test_raw"], test_times, turbines)}
        stamps = test_times.to_numpy().astype("datetime64[ns]").astype("int64")
        limits = {name: thresholds(cal_spec, cal_valid, test_spec, test_valid, turbines, q, source) for name, q, source in VARIANTS}

        for i, tur in enumerate(turbines):
            rows = np.where(test_valid[:, i])[0]
            ts = stamps[rows]
            contiguous = np.r_[False, np.diff(ts) <= MAX_GAP]
            for name, _, _ in VARIANTS:
                for refname in REFS:
                    centre, limit = limits[name][refname]
                    flags = np.abs(test_spec[refname][rows, i] - centre[i]) > limit[i]
                    background.append({"target": target, "turbine_id": tur, "variant": name, "reference": refname,
                                       "rows": len(rows), "point": int(flags.sum()),
                                       "persistent": int(series_persistent(flags, contiguous).sum())})
            for anchor in ANCHORS:
                in_hour = np.where((ts >= anchor.value) & (ts <= (anchor + pd.Timedelta(hours=1)).value))[0]
                for hours in DURATIONS:
                    base = {"target": target, "turbine_id": tur, "anchor": anchor, "paper_anchor": anchor in PAPER_ANCHORS,
                            "duration_hours": hours}
                    if len(in_hour) == 0:
                        for shape in SHAPES:
                            for amp in AMPLITUDES:
                                events.append({**base, "shape": shape, "amplitude": amp, "status": "no_scored_start", "scored_intervals": 0})
                        continue
                    start = ts[in_hour[0]]
                    window = np.where((ts >= start) & (ts < start + pd.Timedelta(hours=hours).value))[0]
                    w_rows, w_ts, w_cont = rows[window], ts[window], contiguous[window].copy()
                    if len(w_cont):
                        w_cont[0] = False
                    if len(window) < PERSISTENCE:
                        for shape in SHAPES:
                            for amp in AMPLITUDES:
                                events.append({**base, "shape": shape, "amplitude": amp, "status": "insufficient_scored_intervals",
                                               "scored_intervals": len(window)})
                        continue
                    states = {refname: test_spec[refname][w_rows, i] for refname in REFS}
                    bg = {name: {refname: persistent(np.abs(states[refname] - limits[name][refname][0][i]) > limits[name][refname][1][i], w_cont)
                                 for refname in REFS} for name, _, _ in VARIANTS}
                    elapsed = (w_ts - start) / 6e10
                    for shape in SHAPES:
                        fraction = np.ones(len(window)) if shape == "step" else np.clip(elapsed / max(hours * 60.0 - 10.0, 1.0), 0.0, 1.0)
                        for amp in AMPLITUDES:
                            delta = amp * scales[tur] * fraction
                            injected = {}
                            for refname in REFS:
                                if refname in SELF_INCLUSIVE:
                                    block = mats[SELF_INCLUSIVE[refname]][w_rows].copy()
                                    block[:, i] += delta
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                                        injected[refname] = block[:, i] - np.nanmedian(block, axis=1)
                                else:
                                    injected[refname] = states[refname] + delta
                            row = {**base, "shape": shape, "amplitude": amp, "status": "scored", "scored_intervals": len(window)}
                            for name, _, _ in VARIANTS:
                                for refname in REFS:
                                    centre, limit = limits[name][refname]
                                    row[f"{name}_background_{refname}"] = bg[name][refname]
                                    row[f"{name}_detected_{refname}"] = persistent(np.abs(injected[refname] - centre[i]) > limit[i], w_cont)
                                row[f"{name}_evaluable"] = not any(bg[name][refname] for refname in PRIMARY_REFS)
                            events.append(row)
        print(f"{target}: alpha={fitted['alpha']:g}, bw={bandwidth:g}, test rows={int(test_valid.sum())}, events so far={len(events)}", flush=True)

    out.mkdir()
    ev = pd.DataFrame(events)
    ev.to_csv(out / "detection_events.csv.gz", index=False)
    bgf = pd.DataFrame(background)
    bg_summary = bgf.groupby(["variant", "target", "reference"], as_index=False)[["rows", "point", "persistent"]].sum()
    bg_summary["point_rate_pct"] = 100 * bg_summary.point / bg_summary.rows
    bg_summary["persistent_rate_pct"] = 100 * bg_summary.persistent / bg_summary.rows
    bg_summary.to_csv(out / "background_rates.csv", index=False)

    flag_cols = [c for c in ev.columns if c.endswith(tuple(REFS)) or c.endswith("_evaluable")]
    ev[flag_cols] = ev[flag_cols].fillna(False).astype(bool)
    summary, decision = [], []
    for vi, (variant, _, _) in enumerate(VARIANTS):
        scored = ev[ev[f"{variant}_evaluable"]]
        for subset_name, subset in [("all_anchors", scored), ("paper_anchors", scored[scored.paper_anchor])]:
            for target, g in subset.groupby("target"):
                for refname in REFS:
                    col = f"{variant}_detected_{refname}"
                    summary.append({"anchors": subset_name, "variant": variant, "target": target, "reference": refname, "by": "pooled",
                                    "level": "all", "evaluable": len(g), "detection_pct": 100 * g[col].mean()})
                    for by, key in [("abs_amplitude", g.amplitude.abs()), ("duration_hours", g.duration_hours), ("shape", g["shape"])]:
                        for level, h in g.groupby(key):
                            summary.append({"anchors": subset_name, "variant": variant, "target": target, "reference": refname, "by": by,
                                            "level": level, "evaluable": len(h), "detection_pct": 100 * h[col].mean()})
        for ki, target in enumerate(TARGETS):
            g = scored[scored.target == target]
            per = g.groupby("turbine_id").agg(n=("status", "size"), **{r: (f"{variant}_detected_{r}", "sum") for r in REFS})
            rng = np.random.default_rng(20261102 + 100 * ki + 10000 * vi)
            draws = []
            for _ in range(2000):
                pick = per.iloc[rng.integers(0, len(per), len(per))]
                rate = {r: pick[r].sum() / pick.n.sum() for r in PRIMARY_REFS}
                draws.append([100 * (rate["distance"] - rate["static_trim"]), 100 * (rate["distance"] - rate["fleet_median_residual"])])
            draws = np.array(draws)
            point = {r: per[r].sum() / per.n.sum() for r in REFS}
            d1 = 100 * (point["distance"] - point["static_trim"])
            d2 = 100 * (point["distance"] - point["fleet_median_residual"])
            lo1, hi1 = np.quantile(draws[:, 0], [0.025, 0.975])
            lo2, hi2 = np.quantile(draws[:, 1], [0.025, 0.975])
            decision.append({"variant": variant, "target": target, "evaluable": int(per.n.sum()),
                             **{f"detection_{r}_pct": 100 * point[r] for r in REFS},
                             "distance_minus_static_trim_pp": d1, "ci_low_vs_static_trim": lo1, "ci_high_vs_static_trim": hi1,
                             "distance_minus_C1_pp": d2, "ci_low_vs_C1": lo2, "ci_high_vs_C1": hi2,
                             "better_than_static_trim": bool(d1 > 0 and lo1 > 0), "better_than_C1": bool(d2 > 0 and lo2 > 0)})
    pd.DataFrame(summary).to_csv(out / "detection_summary.csv", index=False)
    dec = pd.DataFrame(decision)
    dec.to_csv(out / "detection_decision.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "base_runner_sha256": sha(BASE), "reproduction": checks,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    cols = ["variant", "target", "evaluable", "detection_distance_pct", "detection_static_trim_pct", "detection_fleet_median_residual_pct",
            "distance_minus_static_trim_pp", "ci_low_vs_static_trim", "ci_high_vs_static_trim",
            "distance_minus_C1_pp", "ci_low_vs_C1", "ci_high_vs_C1"]
    print(dec[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
