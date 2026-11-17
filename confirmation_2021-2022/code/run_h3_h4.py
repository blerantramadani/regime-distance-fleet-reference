"""H3 (detection, 2015 rule) and H4 (detection, EWMA rule) of the 2021-2022 confirmation protocol.

Chronology: training April-September 2021, tuning October, calibration November,
test December 2021 to April 2022. Model, bandwidth and states come from the
unchanged 2015 functions; the event logic follows ../../posthoc_2026-10-30/detection/run_detection.py.

    python3 run_h3_h4.py --data <layer.csv.gz> --out <dir>
"""
import argparse
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = load("run_chesterman", ANALYSIS / "posthoc_2026-10-30" / "fleet_median_baselines" / "run_chesterman.py")
from run_health_twin_baseline import SOURCE_COLUMNS, TARGETS, predict  # noqa: E402
from run_health_twin_regime_peer_v2 import residual_frame  # noqa: E402
from run_health_twin_rolling_ablation_v1 import fit_window  # noqa: E402
from run_health_twin_trim_distance_ablation import validation_bandwidth_grid  # noqa: E402

AMBIENT = "local_ambient_temp_mean"
START, TRAIN_END, TUNING_END, CALIBRATION_END, TEST_END = (pd.Timestamp(t) for t in
    ["2021-04-01", "2021-10-01", "2021-11-01", "2021-12-01", "2022-05-01"])
REFS = ["distance", "static_trim", "fleet_median_residual", "static_no_trim", "fleet_median_raw"]
PRIMARY_REFS = REFS[:3]
SELF_INCLUSIVE = {"fleet_median_residual": "residual", "fleet_median_raw": "raw"}
ANCHORS = pd.date_range("2021-12-02", "2022-04-28", freq="2D")
DURATIONS = [1, 6, 24]
SHAPES = ["step", "drift"]
AMPLITUDES = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
MAX_GAP = pd.Timedelta(minutes=20).value
PERSISTENCE = 3
LAMBDA = 0.1
VARIANTS = [("q995", 0.995), ("q990", 0.99), ("q975", 0.975)]
RULES = ["fixed", "ewma"]
PRIMARY = TARGETS[:3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguity(stamps):
    return np.r_[False, np.diff(stamps) <= MAX_GAP]


def ewma(s, contiguous, z_prev=None):
    """EWMA with lambda = 0.1, restarted at the current value after a gap longer than 20 min."""
    z = np.empty(len(s))
    for k in range(len(s)):
        if contiguous[k] and (k > 0 or z_prev is not None):
            z[k] = LAMBDA * s[k] + (1 - LAMBDA) * (z[k - 1] if k > 0 else z_prev)
        else:
            z[k] = s[k]
    return z


def persistent(flags, contiguous):
    if len(flags) < PERSISTENCE:
        return False
    return bool((flags[2:] & flags[1:-1] & flags[:-2] & contiguous[2:] & contiguous[1:-1]).any())


def matrix(frame, times, turbines):
    m = np.full((len(times), len(turbines)), np.nan)
    m[times.get_indexer(frame.timestamp), pd.Index(turbines).get_indexer(frame.turbine_id)] = frame.residual.to_numpy()
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.data.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(args.data)
    assert digest == manifest["output_sha256"], "processed table does not match its manifest"
    data = pd.read_csv(args.data, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(
        ["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data.turbine_id.unique())
    turbine = data.turbine_id.to_numpy()
    time = data.timestamp
    x, rawdesc, eligible = rc.prepare(data, AMBIENT, turbines)
    train = (time.ge(START) & time.lt(TRAIN_END)).to_numpy()
    tuning = (time.ge(TRAIN_END) & time.lt(TUNING_END)).to_numpy()
    calibration = (time.ge(TUNING_END) & time.lt(CALIBRATION_END)).to_numpy()
    test = (time.ge(CALIBRATION_END) & time.lt(TEST_END)).to_numpy()
    scale = np.std(rawdesc[eligible & train], axis=0)
    descriptor = (rawdesc - np.mean(rawdesc[eligible & train], axis=0)) / np.where(scale > 1e-8, scale, 1.0)

    events, background, config = [], [], {}
    for ki, target in enumerate(TARGETS):
        y = data[target].to_numpy(dtype=float)
        usable = eligible & np.isfinite(y)
        fitted = fit_window(x, y, turbine, usable, train, tuning, calibration)
        scales = fitted["residual_scales"]
        frames = {}
        for name, mask in [("tuning", fitted["valid_mask"]), ("calibration", usable & calibration), ("test", usable & test)]:
            frames[name] = residual_frame(data, y[mask] - predict(fitted["model"], x[mask]), mask, descriptor)
            frames[name + "_raw"] = residual_frame(data, y[mask], mask, descriptor)
        grid = pd.DataFrame(validation_bandwidth_grid(frames["tuning"], target, scales))
        bandwidth = float(grid.sort_values(["specific_median_abs", "specific_std", "distance_bandwidth"], kind="stable").iloc[0]["distance_bandwidth"])
        config[target] = {"alpha": fitted["alpha"], "bandwidth": bandwidth}
        cal_t, cal_v, cal_s = rc.states(frames["calibration"], frames["calibration_raw"], scales, bandwidth, turbines)
        test_t, test_v, test_s = rc.states(frames["test"], frames["test_raw"], scales, bandwidth, turbines)
        cal_stamps = cal_t.to_numpy().astype("datetime64[ns]").astype("int64")
        stamps = test_t.to_numpy().astype("datetime64[ns]").astype("int64")
        mats = {"residual": matrix(frames["test"], test_t, turbines), "raw": matrix(frames["test_raw"], test_t, turbines)}

        # Centres and thresholds per rule, variant, reference and turbine, from the calibration month.
        limits = {}
        for i in range(len(turbines)):
            crow = np.where(cal_v[:, i])[0]
            ccont = contiguity(cal_stamps[crow])
            for refname in REFS:
                s = cal_s[refname][crow, i]
                for rule in RULES:
                    stat = s if rule == "fixed" else ewma(s, ccont)
                    centre = float(np.median(stat))
                    for vname, q in VARIANTS:
                        limits[(rule, vname, refname, i)] = (centre, float(np.quantile(np.abs(stat - centre), q)))

        for i, tur in enumerate(turbines):
            rows = np.where(test_v[:, i])[0]
            ts = stamps[rows]
            cont = contiguity(ts)
            base_s = {refname: test_s[refname][rows, i] for refname in REFS}
            base_z = {refname: ewma(base_s[refname], cont) for refname in REFS}
            for rule in RULES:
                for vname, _ in VARIANTS:
                    for refname in REFS:
                        centre, limit = limits[(rule, vname, refname, i)]
                        stat = base_s[refname] if rule == "fixed" else base_z[refname]
                        flags = np.abs(stat - centre) > limit
                        pers = np.zeros(len(flags), dtype=bool)
                        if len(flags) >= PERSISTENCE:
                            pers[2:] = flags[2:] & flags[1:-1] & flags[:-2] & cont[2:] & cont[1:-1]
                        background.append({"target": target, "turbine_id": tur, "rule": rule, "variant": vname, "reference": refname,
                                           "rows": len(rows), "point": int(flags.sum()), "persistent": int(pers.sum())})
            for anchor in ANCHORS:
                in_hour = np.where((ts >= anchor.value) & (ts <= (anchor + pd.Timedelta(hours=1)).value))[0]
                for hours in DURATIONS:
                    base = {"target": target, "turbine_id": tur, "anchor": anchor, "duration_hours": hours}
                    if len(in_hour) == 0:
                        for shape in SHAPES:
                            for amp in AMPLITUDES:
                                events.append({**base, "shape": shape, "amplitude": amp, "status": "no_scored_start", "scored_intervals": 0})
                        continue
                    a = in_hour[0]
                    start = ts[a]
                    window = np.where((ts >= start) & (ts < start + pd.Timedelta(hours=hours).value))[0]
                    if len(window) < PERSISTENCE:
                        for shape in SHAPES:
                            for amp in AMPLITUDES:
                                events.append({**base, "shape": shape, "amplitude": amp, "status": "insufficient_scored_intervals",
                                               "scored_intervals": len(window)})
                        continue
                    w_rows, w_ts = rows[window], ts[window]
                    w_cont = cont[window].copy()
                    first_cont = bool(w_cont[0])
                    w_cont[0] = False  # persistence restarts at the window start
                    z_prev = {refname: (base_z[refname][window[0] - 1] if first_cont else None) for refname in REFS}
                    ecal = np.r_[first_cont, cont[window][1:]]  # EWMA continuity keeps the history before the window
                    bg = {}
                    for rule in RULES:
                        for vname, _ in VARIANTS:
                            for refname in REFS:
                                centre, limit = limits[(rule, vname, refname, i)]
                                stat = base_s[refname][window] if rule == "fixed" else base_z[refname][window]
                                bg[(rule, vname, refname)] = persistent(np.abs(stat - centre) > limit, w_cont)
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
                                    injected[refname] = base_s[refname][window] + delta
                            row = {**base, "shape": shape, "amplitude": amp, "status": "scored", "scored_intervals": len(window)}
                            for rule in RULES:
                                for vname, _ in VARIANTS:
                                    for refname in REFS:
                                        centre, limit = limits[(rule, vname, refname, i)]
                                        stat = injected[refname] if rule == "fixed" else ewma(injected[refname], ecal, z_prev[refname])
                                        row[f"{rule}_{vname}_background_{refname}"] = bg[(rule, vname, refname)]
                                        row[f"{rule}_{vname}_detected_{refname}"] = persistent(np.abs(stat - centre) > limit, w_cont)
                                    row[f"{rule}_{vname}_evaluable"] = not any(bg[(rule, vname, r)] for r in PRIMARY_REFS)
                            events.append(row)
        print(f"{target}: alpha={fitted['alpha']:g}, bw={bandwidth:g}, test rows={int(test_v.sum())}, events={len(events)}", flush=True)

    args.out.mkdir(parents=True)
    ev = pd.DataFrame(events)
    flag_cols = [c for c in ev.columns if c.startswith(("fixed_", "ewma_"))]
    ev[flag_cols] = ev[flag_cols].astype("boolean").fillna(False).astype(bool)
    ev.to_csv(args.out / "detection_events.csv.gz", index=False)
    bgf = pd.DataFrame(background).groupby(["rule", "variant", "target", "reference"], as_index=False)[["rows", "point", "persistent"]].sum()
    bgf["point_rate_pct"] = 100 * bgf.point / bgf.rows
    bgf.to_csv(args.out / "background_rates.csv", index=False)
    decision = []
    for rule in RULES:
        for vi, (vname, _) in enumerate(VARIANTS):
            scored = ev[ev[f"{rule}_{vname}_evaluable"]]
            for ki, target in enumerate(TARGETS):
                g = scored[scored.target == target]
                per = g.groupby("turbine_id").agg(n=("status", "size"), **{r: (f"{rule}_{vname}_detected_{r}", "sum") for r in REFS})
                rng = np.random.default_rng(20270301 + 100 * ki + 10000 * vi + (500000 if rule == "ewma" else 0))
                draws = []
                for _ in range(2000):
                    pick = per.iloc[rng.integers(0, len(per), len(per))]
                    rate = {r: pick[r].sum() / pick.n.sum() for r in PRIMARY_REFS}
                    draws.append([100 * (rate["distance"] - rate["static_trim"]), 100 * (rate["distance"] - rate["fleet_median_residual"])])
                draws = np.array(draws)
                point = {r: per[r].sum() / per.n.sum() for r in REFS}
                d1 = 100 * (point["distance"] - point["static_trim"])
                lo1, hi1 = np.quantile(draws[:, 0], [0.025, 0.975])
                lo2, hi2 = np.quantile(draws[:, 1], [0.025, 0.975])
                decision.append({"rule": rule, "variant": vname, "target": target, "evaluable": int(per.n.sum()),
                                 **{f"detection_{r}_pct": 100 * point[r] for r in REFS},
                                 "distance_minus_static_pp": d1, "ci_low": lo1, "ci_high": hi1,
                                 "distance_minus_C1_pp": 100 * (point["distance"] - point["fleet_median_residual"]),
                                 "C1_ci_low": lo2, "C1_ci_high": hi2})
    dec = pd.DataFrame(decision)
    dec.to_csv(args.out / "h3_h4_detection.csv", index=False)
    prim = dec[(dec.variant == "q995") & dec.target.isin(PRIMARY)]
    fixed, ew = prim[prim.rule == "fixed"], prim[prim.rule == "ewma"]
    h4_channels = int(((ew.distance_minus_static_pp > 0) & (ew.ci_low > 0)).sum())
    summary = {"H3_replicated": bool((fixed.ci_low <= 0).all()), "H4_channels_holding": h4_channels, "H4_supported": h4_channels >= 2,
               "configuration": config, "input_sha256": digest, "runner_sha256": sha(__file__),
               "protocol_sha256": sha(Path(__file__).resolve().parents[1] / "PROTOCOL.md")}
    (args.out / "verification.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(prim[["rule", "target", "evaluable", "detection_distance_pct", "detection_static_trim_pct", "distance_minus_static_pp", "ci_low", "ci_high"]].round(2).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
