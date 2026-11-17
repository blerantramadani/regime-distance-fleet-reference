"""Gain-adjusted comparison of the distance reference with the self-inclusive fleet median (see PROTOCOL.md).

Post hoc. Reuses the frozen setup and the specific states of ../fleet_median_baselines.
Writes only to ./results.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "fleet_median_baselines" / "run_chesterman.py"
spec = importlib.util.spec_from_file_location("run_chesterman", BASE)
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DELTAS = [1.0, 2.0, 3.0]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(frame, values, turbines):
    times = pd.Index(frame.timestamp.unique()).sort_values()
    m = np.full((len(times), len(turbines)), np.nan)
    m[times.get_indexer(frame.timestamp), pd.Index(turbines).get_indexer(frame.turbine_id)] = values
    return m


def self_inclusive_gain(m, valid, scales, turbines, delta):
    """Mean gain of x_i - median_all(x) when x_i is shifted by +-delta scales, over scoring rows."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        base = np.nanmedian(m, axis=1)
        gains, weights = [], []
        for i, tur in enumerate(turbines):
            rows = valid[:, i]
            if not rows.any():
                continue
            for sign in (1.0, -1.0):
                shift = sign * delta * scales[tur]
                shifted = m.copy()
                shifted[:, i] += shift
                moved = np.nanmedian(shifted, axis=1) - base
                g = 1.0 - moved[rows] / shift
                gains.append(g.mean())
                weights.append(rows.sum())
    return float(np.average(gains, weights=weights))


def boot_ratio(cube, seed, num, den):
    rng = np.random.default_rng(seed)
    n = len(cube)
    starts = np.arange(n - 7 + 1)
    draws = []
    for _ in range(2000):
        selected = []
        while len(selected) < n:
            start = int(rng.choice(starts))
            selected.extend(range(start, start + 7))
        s = rc.sd(cube[np.array(selected[:n])].sum(axis=0))
        draws.append(s[num] / s[den])
    return np.array(draws)


def loo_gain_check(data, frame, raw_frame, scales, bandwidth, turbines):
    """Shift one turbine's residual and confirm the leave-one-out states move by exactly the shift."""
    times, valid, before = rc.states(frame, raw_frame, scales, bandwidth, turbines)
    tur = turbines[0]
    shifted = frame.copy()
    shift = 2.0 * scales[tur]
    shifted.loc[shifted.turbine_id == tur, "residual"] += shift
    _, valid2, after = rc.states(shifted, raw_frame, scales, bandwidth, turbines)
    col = 0
    rows = valid[:, col] & valid2[:, col]
    for name in ["static_no_trim", "static_trim", "distance"]:
        np.testing.assert_allclose((after[name][rows, col] - before[name][rows, col]) / shift, 1.0, atol=1e-9)
    return "passed"


def main():
    out = ROOT / "results"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    path = rc.PROJECT / "data/research_layers/health_twin_base_v1.csv.gz"
    manifest = json.loads(path.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(path)
    assert digest == manifest["output_sha256"]
    data = pd.read_csv(path, usecols=rc.SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(
        ["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data.turbine_id.unique())
    turbine = data.turbine_id.to_numpy()
    time = data.timestamp
    i_dist, i_c1, i_c2 = rc.METHODS.index("distance"), rc.METHODS.index("fleet_median_residual"), rc.METHODS.index("fleet_median_raw")

    rows, check = [], None
    for protocol, folds, ambient, offset in [("independent", rc.INDEPENDENT_FOLDS, "met_temperature_mean", 0),
                                             ("local", rc.LOCAL_FOLDS, "local_ambient_temp_mean", 5_000_000)]:
        locked_all = pd.read_csv(rc.PROJECT / f"data/experiment_results/health_twin_trim_distance_ablation_{protocol}_v1/fold_target_method_results.csv")
        x, rawdesc, eligible = rc.prepare(data, ambient, turbines)
        for fi, fold in enumerate(folds):
            train = (time.ge(fold["start"]) & time.lt(fold["train_end"])).to_numpy()
            test = (time.ge(fold["valid_end"]) & time.lt(fold["test_end"])).to_numpy()
            scale = np.std(rawdesc[eligible & train], axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            descriptor = (rawdesc - np.mean(rawdesc[eligible & train], axis=0)) / scale
            for ki, target in enumerate(rc.TARGETS):
                locked = locked_all[(locked_all.fold == fold["fold"]) & (locked_all.target == target)].set_index("method")
                alpha = float(locked.loc["distance_trim", "normal_model_alpha"])
                bandwidth = float(locked.loc["distance_trim", "selected_distance_bandwidth"])
                y = data[target].to_numpy()
                usable = eligible & np.isfinite(y)
                model, scales = rc.fit_frozen(x, y, turbine, usable & train, alpha)
                mask = usable & test
                frame = rc.residual_frame(data, y[mask] - predict_safe(rc, model, x[mask]), mask, descriptor)
                raw_frame = rc.residual_frame(data, y[mask], mask, descriptor)
                times, valid, specific = rc.states(frame, raw_frame, scales, bandwidth, turbines)
                spreads = {m: float(np.std(specific[m][valid])) for m in rc.METHODS}
                np.testing.assert_allclose(spreads["distance"], locked.loc["distance_trim", "specific_std"], atol=1e-9, rtol=1e-9)
                if check is None:
                    check = loo_gain_check(data, frame, raw_frame, scales, bandwidth, turbines)
                r = matrix(frame, frame.residual.to_numpy(), turbines)
                raw = matrix(raw_frame, raw_frame.residual.to_numpy(), turbines)
                cube = rc.daily_cube(times, valid, specific, fold["valid_end"], fold["test_end"])
                ratio_draws = boot_ratio(cube, 20261101 + offset + 10000 * fi + 100 * ki, i_dist, i_c1)
                for delta in DELTAS:
                    g1 = self_inclusive_gain(r, valid, scales, turbines, delta)
                    g2 = self_inclusive_gain(raw, valid, scales, turbines, delta)
                    adjusted = 100 * (spreads["distance"] / spreads["fleet_median_residual"] * g1 - 1)
                    low, high = np.quantile(100 * (ratio_draws * g1 - 1), [0.025, 0.975])
                    rows.append({"protocol": protocol, "fold": fold["fold"], "target": target, "delta_scales": delta,
                                 "gain_C1": g1, "gain_C2": g2,
                                 "sd_distance": spreads["distance"], "sd_C1": spreads["fleet_median_residual"], "sd_C2": spreads["fleet_median_raw"],
                                 "raw_distance_vs_C1_pct": 100 * (spreads["distance"] / spreads["fleet_median_residual"] - 1),
                                 "adjusted_distance_vs_C1_pct": adjusted, "adjusted_ci_low": float(low), "adjusted_ci_high": float(high),
                                 "adjusted_distance_vs_C2_pct": 100 * (spreads["distance"] / spreads["fleet_median_raw"] * g2 - 1)})
                last = [row for row in rows if row["fold"] == fold["fold"] and row["target"] == target and row["protocol"] == protocol]
                print(f"{protocol} {fold['fold']} {target}: " + " | ".join(
                    f"d={row['delta_scales']:.0f}: gain C1 {row['gain_C1']:.3f}, adj {row['adjusted_distance_vs_C1_pct']:+.2f}%" for row in last), flush=True)

    out.mkdir()
    table = pd.DataFrame(rows)
    table.to_csv(out / "gain_adjusted_effects.csv", index=False)
    decision = []
    for target in rc.PRIMARY:
        m = table[(table.protocol == "independent") & (table.target == target)]
        d1 = m[m.delta_scales == 1.0]
        d2 = m[m.delta_scales == 2.0]
        n1 = int((d1.adjusted_distance_vs_C1_pct < 0).sum())
        n2 = int((d2.adjusted_distance_vs_C1_pct < 0).sum())
        decision.append({"target": target, "folds_better_d1": f"{n1}/{len(d1)}", "folds_better_d2": f"{n2}/{len(d2)}",
                         "median_gain_C1_d1": float(d1.gain_C1.median()), "median_gain_C1_d2": float(d2.gain_C1.median()),
                         "median_adjusted_d1_pct": float(d1.adjusted_distance_vs_C1_pct.median()),
                         "median_adjusted_d2_pct": float(d2.adjusted_distance_vs_C1_pct.median()),
                         "supported": bool(n1 >= 3 and n2 >= 3)})
    pd.DataFrame(decision).to_csv(out / "gain_decision.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "base_runner_sha256": sha(BASE), "loo_gain_check": check,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(decision).to_string(index=False))


def predict_safe(module, model, x):
    return module.predict(model, x)


if __name__ == "__main__":
    main()
