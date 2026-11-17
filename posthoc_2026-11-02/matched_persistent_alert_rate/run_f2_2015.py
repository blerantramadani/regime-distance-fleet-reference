"""F2 part A: detection versus persistent-alarm rate on the 2015 alert chronology (see PROTOCOL.md).

Post hoc. Reuses the data, models, states, scenarios and persistence rule of ../../posthoc_2026-10-30/detection/run_detection.py
and adds a threshold grid. Writes only to ./results_2015.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
F_DIR = ROOT.parents[1] / "posthoc_2026-10-30" / "detection"
spec = importlib.util.spec_from_file_location("run_detection", F_DIR / "run_detection.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)

Q = np.array([0.950, 0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.9925, 0.995, 0.996, 0.997, 0.998, 0.999])
I995 = int(np.where(np.isclose(Q, 0.995))[0][0])
ALARM_RATES = [0.1, 0.2, 0.5, 1.0]  # percent of scored test rows
DRAWS = 2000
REFS = rd.REFS
PRIMARY = rd.PRIMARY_REFS  # distance, static_trim, fleet_median_residual


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid_limits(cal_spec, cal_valid, turbines):
    """October centre and the October quantile thresholds on the grid Q, per reference and turbine."""
    out = {}
    for ref in REFS:
        centre = np.full(len(turbines), np.nan)
        limits = np.full((len(turbines), len(Q)), np.nan)
        for i in range(len(turbines)):
            cal = cal_spec[ref][cal_valid[:, i], i]
            if len(cal) < rd.MIN_CALIBRATION_ROWS:
                raise ValueError(f"Turbine {turbines[i]} has too few calibration rows")
            centre[i] = np.median(cal)
            limits[i] = np.quantile(np.abs(cal - centre[i]), Q)
        out[ref] = (centre, limits)
    return out


def persistent_rows(flags, contiguous):
    """Rows carrying a persistent flag, for each threshold column of a (rows x thresholds) flag matrix."""
    out = np.zeros(flags.shape, dtype=bool)
    if len(flags) >= rd.PERSISTENCE:
        link = (contiguous[2:] & contiguous[1:-1])[:, None]
        out[2:] = flags[2:] & flags[1:-1] & flags[:-2] & link
    return out


def episodes(pers, contiguous):
    previous = np.vstack([np.zeros((1, pers.shape[1]), dtype=bool), pers[:-1]]) & contiguous[:, None]
    return (pers & ~previous).sum(axis=0)


def any_persistent(flags, contiguous):
    if len(flags) < rd.PERSISTENCE:
        return np.zeros(flags.shape[1], dtype=bool)
    link = (contiguous[2:] & contiguous[1:-1])[:, None]
    return (flags[2:] & flags[1:-1] & flags[:-2] & link).any(axis=0)


def interpolate(x, d, a):
    """Detection at alarm rate a by linear interpolation in log10(rate); NaN outside the observed range."""
    keep = x > 0
    if keep.sum() < 2:
        return np.nan
    lx, dd = np.log10(x[keep]), d[keep]
    order = np.lexsort((dd, lx))
    lx, dd = lx[order], dd[order]
    la = np.log10(a)
    if la < lx[0] or la > lx[-1]:
        return np.nan
    return float(np.interp(la, lx, dd))


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
    curves, contrasts, checks = [], [], {}
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
            rows_sum = rows_t[ti].sum()
            x = 100 * bg_rows[ti].sum(axis=0) / rows_sum
            n = nwin[np.ix_(ti, ai)].sum()
            d = 100 * det[np.ix_(ti, ai)].sum(axis=(0, 1)) / n
            b = 100 * bgw[np.ix_(ti, ai)].sum(axis=(0, 1)) / n
            return x, d, d - b

        allT, allA = np.arange(nT), np.arange(nA)
        x0, d0, e0 = curves_from(allT, allA)
        eps0 = 100 * bg_eps.sum(axis=0) / (rows_t.sum() / 144.0)
        cal0 = 100 * cal_pers.sum(axis=0) / cal_rows.sum()
        for r, ref in enumerate(REFS):
            for k, q in enumerate(Q):
                curves.append({"target": target, "reference": ref, "q": q, "test_alarm_rate_pct": x0[r, k],
                               "episodes_per_100_turbine_days": eps0[r, k], "october_alarm_rate_pct": cal0[r, k],
                               "detection_pct": d0[r, k], "window_background_pct": d0[r, k] - e0[r, k],
                               "excess_detection_pct": e0[r, k], "scored_windows": int(nwin.sum())})

        rng = np.random.default_rng(20261103 + 100 * ki)
        boot = {(m, c, a): [] for m in ["detection", "excess"] for c in ["static_trim", "fleet_median_residual"] for a in ALARM_RATES}
        for _ in range(DRAWS):
            ti, ai = rng.integers(0, nT, nT), rng.integers(0, nA, nA)
            x, d, e = curves_from(ti, ai)
            for m, val in [("detection", d), ("excess", e)]:
                for c in ["static_trim", "fleet_median_residual"]:
                    for a in ALARM_RATES:
                        boot[(m, c, a)].append(interpolate(x[0], val[0], a) - interpolate(x[REFS.index(c)], val[REFS.index(c)], a))
        for m, val in [("detection", d0), ("excess", e0)]:
            for c in ["static_trim", "fleet_median_residual"]:
                for a in ALARM_RATES:
                    point = interpolate(x0[0], val[0], a) - interpolate(x0[REFS.index(c)], val[REFS.index(c)], a)
                    draws = np.array(boot[(m, c, a)])
                    ok = np.isfinite(draws)
                    lo, hi = (np.quantile(draws[ok], [0.025, 0.975]) if ok.sum() >= 0.95 * DRAWS else (np.nan, np.nan))
                    reading = ("not estimable" if not np.isfinite(point) or not np.isfinite(lo)
                               else "better" if point > 0 and lo > 0 else "worse" if point < 0 and hi < 0 else "no clear difference")
                    contrasts.append({"target": target, "measure": m, "comparator": c, "alarm_rate_pct": a,
                                      "distance_minus_comparator_pp": point, "ci_low": lo, "ci_high": hi,
                                      "finite_draws": int(ok.sum()), "reading": reading})
        print(f"{target}: done", flush=True)

    out.mkdir()
    pd.DataFrame(curves).to_csv(out / "f2_curves.csv", index=False)
    con = pd.DataFrame(contrasts)
    con.to_csv(out / "f2_matched_contrasts.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "f_runner_sha256": sha(F_DIR / "run_detection.py"), "checks": checks,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    show = con[(con.measure == "detection")]
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
