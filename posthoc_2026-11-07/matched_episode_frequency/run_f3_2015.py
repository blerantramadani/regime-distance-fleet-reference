"""F3 part A: detection at matched alarm-episode frequency, 2015 alert chronology (see PROTOCOL.md).

Post hoc. A copy of ../../posthoc_2026-11-02/matched_persistent_alert_rate/run_f2_2015.py in which only the matching axis changes:
detection is interpolated against persistent-alarm episodes per 100 scored turbine-days instead of the
flagged-row fraction. The F2 helper functions are loaded from the F2 runner. Writes only to ./results_2015.
"""
from pathlib import Path
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
F_DIR = ROOT.parents[1] / "posthoc_2026-10-30" / "detection"
F2_RUNNER = ROOT.parents[1] / "posthoc_2026-11-02" / "matched_persistent_alert_rate" / "run_f2_2015.py"
F2_CURVES_PATH = ROOT.parents[1] / "posthoc_2026-11-02" / "matched_persistent_alert_rate" / "results_2015" / "f2_curves.csv"
_spec = importlib.util.spec_from_file_location("run_f2_2015", F2_RUNNER)
f2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f2)
rd, Q, I995, DRAWS, REFS, PRIMARY = f2.rd, f2.Q, f2.I995, f2.DRAWS, f2.REFS, f2.PRIMARY
sha, grid_limits, persistent_rows, episodes, any_persistent, interpolate = (
    f2.sha, f2.grid_limits, f2.persistent_rows, f2.episodes, f2.any_persistent, f2.interpolate)

LEVELS = [5.0, 10.0, 20.0, 40.0]  # persistent-alarm episodes per 100 scored turbine-days (PROTOCOL.md, Section 3)
COMPARATORS = ["static_trim", "fleet_median_residual"]
F2_CURVES = pd.read_csv(F2_CURVES_PATH)
companions = []


def main():
    out = ROOT / "results_2015"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    path = rd.rc.PROJECT / "data/research_layers/health_twin_base_v1.csv.gz"
    manifest = json.loads(path.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(path)
    assert digest == manifest["output_sha256"]

    ambient = rd.AMBIENT_SOURCES["independent_meteorology"]["column"]
    data = pd.read_csv(path, usecols=rd.SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    turbine_id = data["turbine_id"].astype(str).to_numpy()
    features, _ = rd.make_feature_matrix(data, turbines, ambient)
    descriptor, _ = rd.regime_matrix(data, ambient)
    eligible = (
        data["timestamp"].ge(rd.MONITORING_START)
        & data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(descriptor).all(axis=1)
    ).to_numpy()
    time = data["timestamp"]
    train = (time.ge(rd.MONITORING_START) & time.lt(rd.TRAIN_END)).to_numpy()
    tuning = (time.ge(rd.TRAIN_END) & time.lt(rd.TUNING_END)).to_numpy()
    calibration = (time.ge(rd.TUNING_END) & time.lt(rd.CALIBRATION_END)).to_numpy()
    test = (time.ge(rd.CALIBRATION_END) & time.lt(rd.TEST_END)).to_numpy()
    mean = np.mean(descriptor[eligible & train], axis=0)
    scale = np.std(descriptor[eligible & train], axis=0)
    descriptor = (descriptor - mean) / np.where(scale > 1e-8, scale, 1.0)
    saved_config = pd.read_csv(rd.SAVED / "selected_configuration.csv").set_index("target")
    f_decision = pd.read_csv(F_DIR / "results" / "detection_decision.csv")

    anchors = list(rd.ANCHORS)
    nT, nA, nR, nQ = len(turbines), len(anchors), len(REFS), len(Q)
    contrasts, checks = [], {}
    for ki, target in enumerate(rd.TARGETS):
        y = data[target].to_numpy(dtype=float)
        usable = eligible & np.isfinite(y)
        fitted = rd.fit_window(features, y, turbine_id, usable, train, tuning, calibration)
        assert fitted["alpha"] == float(saved_config.loc[target, "normal_model_alpha"])
        scales = fitted["residual_scales"]
        frames = {}
        for name, period in [("tuning", tuning), ("calibration", calibration), ("test", test)]:
            mask = fitted["valid_mask"] if name == "tuning" else usable & period
            frames[name] = rd.residual_frame(data, y[mask] - rd.predict(fitted["model"], features[mask]), mask, descriptor)
            frames[name + "_raw"] = rd.residual_frame(data, y[mask], mask, descriptor)
        grid = pd.DataFrame(rd.validation_bandwidth_grid(frames["tuning"], target, scales))
        bandwidth = float(grid.sort_values(["specific_median_abs", "specific_std", "distance_bandwidth"], kind="stable").iloc[0]["distance_bandwidth"])
        assert bandwidth == float(saved_config.loc[target, "selected_distance_bandwidth"])
        cal_times, cal_valid, cal_spec = rd.rc.states(frames["calibration"], frames["calibration_raw"], scales, bandwidth, turbines)
        test_times, test_valid, test_spec = rd.rc.states(frames["test"], frames["test_raw"], scales, bandwidth, turbines)
        limits = grid_limits(cal_spec, cal_valid, turbines)
        mats = {"residual": rd.matrix(frames["test"], test_times, turbines), "raw": rd.matrix(frames["test_raw"], test_times, turbines)}
        stamps = test_times.to_numpy().astype("datetime64[ns]").astype("int64")
        cal_stamps = cal_times.to_numpy().astype("datetime64[ns]").astype("int64")

        bg_rows = np.zeros((nT, nR, nQ)); bg_eps = np.zeros((nT, nR, nQ)); rows_t = np.zeros(nT)
        cal_pers = np.zeros((nT, nR, nQ)); cal_rows = np.zeros(nT)
        det = np.zeros((nT, nA, nR, nQ)); bgw = np.zeros((nT, nA, nR, nQ)); nwin = np.zeros((nT, nA))
        f_eval = np.zeros(nT); f_det = np.zeros((nT, nR))

        for i, tur in enumerate(turbines):
            crow = np.where(cal_valid[:, i])[0]
            ccont = np.r_[False, np.diff(cal_stamps[crow]) <= rd.MAX_GAP]
            cal_rows[i] = len(crow)
            rows = np.where(test_valid[:, i])[0]
            ts = stamps[rows]
            contiguous = np.r_[False, np.diff(ts) <= rd.MAX_GAP]
            rows_t[i] = len(rows)
            for r, ref in enumerate(REFS):
                centre, lim = limits[ref]
                cflags = np.abs(cal_spec[ref][crow, i] - centre[i])[:, None] > lim[i][None, :]
                cal_pers[i, r] = persistent_rows(cflags, ccont).sum(axis=0)
                flags = np.abs(test_spec[ref][rows, i] - centre[i])[:, None] > lim[i][None, :]
                pers = persistent_rows(flags, contiguous)
                bg_rows[i, r] = pers.sum(axis=0)
                bg_eps[i, r] = episodes(pers, contiguous)
            for a, anchor in enumerate(anchors):
                in_hour = np.where((ts >= anchor.value) & (ts <= (anchor + pd.Timedelta(hours=1)).value))[0]
                if len(in_hour) == 0:
                    continue
                start = ts[in_hour[0]]
                for hours in rd.DURATIONS:
                    window = np.where((ts >= start) & (ts < start + pd.Timedelta(hours=hours).value))[0]
                    if len(window) < rd.PERSISTENCE:
                        continue
                    w_rows, w_ts, w_cont = rows[window], ts[window], contiguous[window].copy()
                    w_cont[0] = False
                    states = {ref: test_spec[ref][w_rows, i] for ref in REFS}
                    bg = {}
                    for r, ref in enumerate(REFS):
                        centre, lim = limits[ref]
                        bg[ref] = any_persistent(np.abs(states[ref] - centre[i])[:, None] > lim[i][None, :], w_cont)
                    f_evaluable = not any(bg[ref][I995] for ref in PRIMARY)
                    elapsed = (w_ts - start) / 6e10
                    for shape in rd.SHAPES:
                        fraction = np.ones(len(window)) if shape == "step" else np.clip(elapsed / max(hours * 60.0 - 10.0, 1.0), 0.0, 1.0)
                        for amp in rd.AMPLITUDES:
                            delta = amp * scales[tur] * fraction
                            nwin[i, a] += 1
                            if f_evaluable:
                                f_eval[i] += 1
                            for r, ref in enumerate(REFS):
                                if ref in rd.SELF_INCLUSIVE:
                                    block = mats[rd.SELF_INCLUSIVE[ref]][w_rows].copy()
                                    block[:, i] += delta
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                                        injected = block[:, i] - np.nanmedian(block, axis=1)
                                else:
                                    injected = states[ref] + delta
                                centre, lim = limits[ref]
                                hit = any_persistent(np.abs(injected - centre[i])[:, None] > lim[i][None, :], w_cont)
                                det[i, a, r] += hit
                                bgw[i, a, r] += bg[ref]
                                if f_evaluable:
                                    f_det[i, r] += hit[I995]

        # Reproduction of F at q = 0.995 under F's evaluability rule.
        fd = f_decision[(f_decision.variant == "q995") & (f_decision.target == target)].iloc[0]
        repro = {"evaluable_F2": int(f_eval.sum()), "evaluable_F": int(fd.evaluable)}
        for r, ref in enumerate(REFS):
            mine = 100 * f_det[:, r].sum() / f_eval.sum()
            repro[ref] = float(abs(mine - fd[f"detection_{ref}_pct"]))
        assert repro["evaluable_F2"] == repro["evaluable_F"], repro
        assert max(repro[ref] for ref in REFS) <= 1e-9, repro
        checks[target] = {"alpha": fitted["alpha"], "bandwidth": bandwidth, "reproduction_q995": repro}
        print(f"{target}: reproduction {repro}", flush=True)

        def curves_from(ti, ai):
            x = 100 * bg_rows[ti].sum(axis=0) / rows_t[ti].sum()
            ep = 100 * bg_eps[ti].sum(axis=0) / (rows_t[ti].sum() / 144.0)
            n = nwin[np.ix_(ti, ai)].sum()
            d = 100 * det[np.ix_(ti, ai)].sum(axis=(0, 1)) / n
            b = 100 * bgw[np.ix_(ti, ai)].sum(axis=(0, 1)) / n
            return x, ep, d, d - b

        allT, allA = np.arange(nT), np.arange(nA)
        x0, ep0, d0, e0 = curves_from(allT, allA)

        # Reproduction check 2 (PROTOCOL.md, Section 4): full-sample curves equal F2's.
        f2c, dev = F2_CURVES[F2_CURVES.target == target], 0.0
        for r, ref in enumerate(REFS):
            g = f2c[f2c.reference == ref].sort_values("q")
            assert np.allclose(g.q.to_numpy(), Q, rtol=0, atol=1e-12)
            for col, val in [("test_alarm_rate_pct", x0[r]), ("episodes_per_100_turbine_days", ep0[r]),
                             ("detection_pct", d0[r]), ("excess_detection_pct", e0[r])]:
                dev = max(dev, float(np.max(np.abs(g[col].to_numpy() - val))))
        assert dev <= 1e-9, (target, dev)
        checks[target]["f2_curve_max_abs_dev"] = dev

        # Descriptive companions (PROTOCOL.md, Section 3).
        with np.errstate(divide="ignore", invalid="ignore"):
            ep_len = bg_rows.sum(axis=0) / bg_eps.sum(axis=0)
        for r, ref in enumerate(REFS):
            for e in LEVELS:
                companions.append({"target": target, "reference": ref, "episodes_per_100_turbine_days": e,
                                   "flagged_row_pct": interpolate(ep0[r], x0[r], e),
                                   "mean_episode_rows": interpolate(ep0[r], ep_len[r], e),
                                   "detection_pct": interpolate(ep0[r], d0[r], e),
                                   "excess_detection_pct": interpolate(ep0[r], e0[r], e)})

        rng = np.random.default_rng(20261107 + 100 * ki)
        boot = {(m, c, e): [] for m in ["detection", "excess"] for c in COMPARATORS for e in LEVELS}
        for _ in range(DRAWS):
            ti, ai = rng.integers(0, nT, nT), rng.integers(0, nA, nA)
            _, ep, d, ex = curves_from(ti, ai)
            for m, val in [("detection", d), ("excess", ex)]:
                for c in COMPARATORS:
                    for e in LEVELS:
                        boot[(m, c, e)].append(interpolate(ep[0], val[0], e) - interpolate(ep[REFS.index(c)], val[REFS.index(c)], e))
        for m, val in [("detection", d0), ("excess", e0)]:
            for c in COMPARATORS:
                for e in LEVELS:
                    point = interpolate(ep0[0], val[0], e) - interpolate(ep0[REFS.index(c)], val[REFS.index(c)], e)
                    draws = np.array(boot[(m, c, e)])
                    ok = np.isfinite(draws)
                    lo, hi = (np.quantile(draws[ok], [0.025, 0.975]) if ok.sum() >= 0.95 * DRAWS else (np.nan, np.nan))
                    reading = ("not estimable" if not np.isfinite(point) or not np.isfinite(lo)
                               else "better" if point > 0 and lo > 0 else "worse" if point < 0 and hi < 0 else "no clear difference")
                    contrasts.append({"target": target, "measure": m, "comparator": c, "episodes_per_100_turbine_days": e,
                                      "distance_minus_comparator_pp": point, "ci_low": lo, "ci_high": hi,
                                      "finite_draws": int(ok.sum()), "reading": reading})
        print(f"{target}: done", flush=True)

    out.mkdir()
    con = pd.DataFrame(contrasts)
    con.to_csv(out / "f3_matched_contrasts.csv", index=False)
    pd.DataFrame(companions).to_csv(out / "f3_companions.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "f2_runner_sha256": sha(F2_RUNNER), "f2_curves_sha256": sha(F2_CURVES_PATH), "checks": checks,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(con[con.measure == "detection"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
