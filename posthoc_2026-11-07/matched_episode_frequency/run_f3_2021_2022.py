"""F3 part B: detection at matched alarm-episode frequency, 2021-2022 chronology (see PROTOCOL.md, Section 5).

Post hoc and descriptive. A copy of ../../posthoc_2026-11-02/matched_persistent_alert_rate/run_f2_2021_2022.py in which only the matching
axis changes (episodes per 100 scored turbine-days instead of the flagged-row fraction), for the fixed and
EWMA rules. The F2 helper functions are loaded from the F2 runner. Writes only to ./results_2021_2022.

    python3 run_f3_2021_2022.py --data <2021-2022 health_twin_base_v1.csv.gz>
"""
import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
F2_RUNNER = ROOT.parents[1] / "posthoc_2026-11-02" / "matched_persistent_alert_rate" / "run_f2_2021_2022.py"
F2_CURVES_PATH = ROOT.parents[1] / "posthoc_2026-11-02" / "matched_persistent_alert_rate" / "results_2021_2022" / "f2_curves.csv"
_spec = importlib.util.spec_from_file_location("run_f2_2021_2022", F2_RUNNER)
f2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f2)
hb, Q, I995, DRAWS, REFS, PRIMARY, RULES, SEED_OFFSET = (
    f2.hb, f2.Q, f2.I995, f2.DRAWS, f2.REFS, f2.PRIMARY, f2.RULES, f2.SEED_OFFSET)
sha, persistent_rows, episodes, any_persistent, interpolate = (
    f2.sha, f2.persistent_rows, f2.episodes, f2.any_persistent, f2.interpolate)

LEVELS = [5.0, 10.0, 20.0, 40.0]  # persistent-alarm episodes per 100 scored turbine-days (PROTOCOL.md, Section 3)
COMPARATORS = ["static_trim", "fleet_median_residual"]
F2_CURVES = pd.read_csv(F2_CURVES_PATH)
companions = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / "results_2021_2022"
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(args.data.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(args.data)
    assert digest == manifest["output_sha256"], "processed table does not match its manifest"
    saved = pd.read_csv(ROOT.parents[1] / "confirmation_2021-2022" / "results" / "h3_h4" / "h3_h4_detection.csv")

    data = pd.read_csv(args.data, usecols=hb.SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(
        ["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data.turbine_id.unique())
    turbine = data.turbine_id.to_numpy()
    time = data.timestamp
    x, rawdesc, eligible = hb.rc.prepare(data, hb.AMBIENT, turbines)
    train = (time.ge(hb.START) & time.lt(hb.TRAIN_END)).to_numpy()
    tuning = (time.ge(hb.TRAIN_END) & time.lt(hb.TUNING_END)).to_numpy()
    calibration = (time.ge(hb.TUNING_END) & time.lt(hb.CALIBRATION_END)).to_numpy()
    test = (time.ge(hb.CALIBRATION_END) & time.lt(hb.TEST_END)).to_numpy()
    scale = np.std(rawdesc[eligible & train], axis=0)
    descriptor = (rawdesc - np.mean(rawdesc[eligible & train], axis=0)) / np.where(scale > 1e-8, scale, 1.0)

    anchors = list(hb.ANCHORS)
    nT, nA, nR, nQ = len(turbines), len(anchors), len(REFS), len(Q)
    contrasts, checks = [], {}
    for ki, target in enumerate(hb.TARGETS):
        y = data[target].to_numpy(dtype=float)
        usable = eligible & np.isfinite(y)
        fitted = hb.fit_window(x, y, turbine, usable, train, tuning, calibration)
        scales = fitted["residual_scales"]
        frames = {}
        for name, mask in [("tuning", fitted["valid_mask"]), ("calibration", usable & calibration), ("test", usable & test)]:
            frames[name] = hb.residual_frame(data, y[mask] - hb.predict(fitted["model"], x[mask]), mask, descriptor)
            frames[name + "_raw"] = hb.residual_frame(data, y[mask], mask, descriptor)
        grid = pd.DataFrame(hb.validation_bandwidth_grid(frames["tuning"], target, scales))
        bandwidth = float(grid.sort_values(["specific_median_abs", "specific_std", "distance_bandwidth"], kind="stable").iloc[0]["distance_bandwidth"])
        cal_t, cal_v, cal_s = hb.rc.states(frames["calibration"], frames["calibration_raw"], scales, bandwidth, turbines)
        test_t, test_v, test_s = hb.rc.states(frames["test"], frames["test_raw"], scales, bandwidth, turbines)
        cal_stamps = cal_t.to_numpy().astype("datetime64[ns]").astype("int64")
        stamps = test_t.to_numpy().astype("datetime64[ns]").astype("int64")
        mats = {"residual": hb.matrix(frames["test"], test_t, turbines), "raw": hb.matrix(frames["test_raw"], test_t, turbines)}

        limits, cal_pers, cal_rows = {}, {rule: np.zeros((nT, nR, nQ)) for rule in RULES}, np.zeros(nT)
        for i in range(nT):
            crow = np.where(cal_v[:, i])[0]
            ccont = hb.contiguity(cal_stamps[crow])
            cal_rows[i] = len(crow)
            for r, ref in enumerate(REFS):
                s = cal_s[ref][crow, i]
                for rule in RULES:
                    stat = s if rule == "fixed" else hb.ewma(s, ccont)
                    centre = float(np.median(stat))
                    lim = np.quantile(np.abs(stat - centre), Q)
                    limits[(rule, ref, i)] = (centre, lim)
                    cal_pers[rule][i, r] = persistent_rows(np.abs(stat - centre)[:, None] > lim[None, :], ccont).sum(axis=0)

        acc = {rule: {"bg_rows": np.zeros((nT, nR, nQ)), "bg_eps": np.zeros((nT, nR, nQ)),
                      "det": np.zeros((nT, nA, nR, nQ)), "bgw": np.zeros((nT, nA, nR, nQ)),
                      "f_eval": np.zeros(nT), "f_det": np.zeros((nT, nR))} for rule in RULES}
        rows_t, nwin = np.zeros(nT), np.zeros((nT, nA))
        for i, tur in enumerate(turbines):
            rows = np.where(test_v[:, i])[0]
            ts = stamps[rows]
            cont = hb.contiguity(ts)
            rows_t[i] = len(rows)
            base_s = {ref: test_s[ref][rows, i] for ref in REFS}
            base_z = {ref: hb.ewma(base_s[ref], cont) for ref in REFS}
            for rule in RULES:
                for r, ref in enumerate(REFS):
                    centre, lim = limits[(rule, ref, i)]
                    stat = base_s[ref] if rule == "fixed" else base_z[ref]
                    pers = persistent_rows(np.abs(stat - centre)[:, None] > lim[None, :], cont)
                    acc[rule]["bg_rows"][i, r] = pers.sum(axis=0)
                    acc[rule]["bg_eps"][i, r] = episodes(pers, cont)
            for a, anchor in enumerate(anchors):
                in_hour = np.where((ts >= anchor.value) & (ts <= (anchor + pd.Timedelta(hours=1)).value))[0]
                if len(in_hour) == 0:
                    continue
                start = ts[in_hour[0]]
                for hours in hb.DURATIONS:
                    window = np.where((ts >= start) & (ts < start + pd.Timedelta(hours=hours).value))[0]
                    if len(window) < hb.PERSISTENCE:
                        continue
                    w_rows, w_ts = rows[window], ts[window]
                    w_cont = cont[window].copy()
                    first_cont = bool(w_cont[0])
                    w_cont[0] = False
                    z_prev = {ref: (base_z[ref][window[0] - 1] if first_cont else None) for ref in REFS}
                    ecal = np.r_[first_cont, cont[window][1:]]
                    bg = {}
                    for rule in RULES:
                        for ref in REFS:
                            centre, lim = limits[(rule, ref, i)]
                            stat = base_s[ref][window] if rule == "fixed" else base_z[ref][window]
                            bg[(rule, ref)] = any_persistent(np.abs(stat - centre)[:, None] > lim[None, :], w_cont)
                    f_evaluable = {rule: not any(bg[(rule, ref)][I995] for ref in PRIMARY) for rule in RULES}
                    elapsed = (w_ts - start) / 6e10
                    for shape in hb.SHAPES:
                        fraction = np.ones(len(window)) if shape == "step" else np.clip(elapsed / max(hours * 60.0 - 10.0, 1.0), 0.0, 1.0)
                        for amp in hb.AMPLITUDES:
                            delta = amp * scales[tur] * fraction
                            nwin[i, a] += 1
                            injected = {}
                            for ref in REFS:
                                if ref in hb.SELF_INCLUSIVE:
                                    block = mats[hb.SELF_INCLUSIVE[ref]][w_rows].copy()
                                    block[:, i] += delta
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
                                        injected[ref] = block[:, i] - np.nanmedian(block, axis=1)
                                else:
                                    injected[ref] = base_s[ref][window] + delta
                            for rule in RULES:
                                if f_evaluable[rule]:
                                    acc[rule]["f_eval"][i] += 1
                                for r, ref in enumerate(REFS):
                                    centre, lim = limits[(rule, ref, i)]
                                    stat = injected[ref] if rule == "fixed" else hb.ewma(injected[ref], ecal, z_prev[ref])
                                    hit = any_persistent(np.abs(stat - centre)[:, None] > lim[None, :], w_cont)
                                    acc[rule]["det"][i, a, r] += hit
                                    acc[rule]["bgw"][i, a, r] += bg[(rule, ref)]
                                    if f_evaluable[rule]:
                                        acc[rule]["f_det"][i, r] += hit[I995]

        checks[target] = {"alpha": fitted["alpha"], "bandwidth": bandwidth}
        for rule in RULES:
            A = acc[rule]
            sv = saved[(saved["rule"] == rule) & (saved.variant == "q995") & (saved.target == target)].iloc[0]
            repro = {"evaluable_F2": int(A["f_eval"].sum()), "evaluable_saved": int(sv.evaluable)}
            for r, ref in enumerate(REFS):
                repro[ref] = float(abs(100 * A["f_det"][:, r].sum() / A["f_eval"].sum() - sv[f"detection_{ref}_pct"]))
            assert repro["evaluable_F2"] == repro["evaluable_saved"], (rule, repro)
            assert max(repro[ref] for ref in REFS) <= 1e-9, (rule, repro)
            checks[target][f"reproduction_{rule}_q995"] = repro

            def curves_from(ti, ai, A=A):
                x = 100 * A["bg_rows"][ti].sum(axis=0) / rows_t[ti].sum()
                ep = 100 * A["bg_eps"][ti].sum(axis=0) / (rows_t[ti].sum() / 144.0)
                n = nwin[np.ix_(ti, ai)].sum()
                d = 100 * A["det"][np.ix_(ti, ai)].sum(axis=(0, 1)) / n
                b = 100 * A["bgw"][np.ix_(ti, ai)].sum(axis=(0, 1)) / n
                return x, ep, d, d - b

            allT, allA = np.arange(nT), np.arange(nA)
            x0, ep0, d0, e0 = curves_from(allT, allA)

            # Reproduction check 2 (PROTOCOL.md, Section 4): full-sample curves equal F2's.
            f2c, dev = F2_CURVES[(F2_CURVES.target == target) & (F2_CURVES.rule == rule)], 0.0
            for r, ref in enumerate(REFS):
                g = f2c[f2c.reference == ref].sort_values("q")
                assert np.allclose(g.q.to_numpy(), Q, rtol=0, atol=1e-12)
                for col, val in [("test_alarm_rate_pct", x0[r]), ("episodes_per_100_turbine_days", ep0[r]),
                                 ("detection_pct", d0[r]), ("excess_detection_pct", e0[r])]:
                    dev = max(dev, float(np.max(np.abs(g[col].to_numpy() - val))))
            assert dev <= 1e-9, (target, dev)
            checks[target][f"f2_curve_max_abs_dev_{rule}"] = dev

            # Descriptive companions (PROTOCOL.md, Section 3).
            with np.errstate(divide="ignore", invalid="ignore"):
                ep_len = A["bg_rows"].sum(axis=0) / A["bg_eps"].sum(axis=0)
            for r, ref in enumerate(REFS):
                for e in LEVELS:
                    companions.append({"rule": rule, "target": target, "reference": ref, "episodes_per_100_turbine_days": e,
                                       "flagged_row_pct": interpolate(ep0[r], x0[r], e),
                                       "mean_episode_rows": interpolate(ep0[r], ep_len[r], e),
                                       "detection_pct": interpolate(ep0[r], d0[r], e),
                                       "excess_detection_pct": interpolate(ep0[r], e0[r], e)})

            rng = np.random.default_rng(20261107 + 100 * ki + SEED_OFFSET[rule])
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
                        contrasts.append({"rule": rule, "target": target, "measure": m, "comparator": c, "episodes_per_100_turbine_days": e,
                                          "distance_minus_comparator_pp": point, "ci_low": lo, "ci_high": hi,
                                          "finite_draws": int(ok.sum()), "reading": reading})
        print(f"{target}: {checks[target]}", flush=True)

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
