"""Plain fleet-median baselines (Chesterman-style) on the primary scoring rows (see PROTOCOL.md).

C1: residual minus the median residual of all eligible turbines, target included, no trim.
C2: raw signal minus the median raw signal of all eligible turbines, target included, no model.
Reads the processed table in ../../data and writes only to ./results.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from run_health_twin_baseline import (SOURCE_COLUMNS, TARGETS, make_feature_matrix,  # noqa: E402
                                      fit_ridge, robust_training_keep, predict)
from run_health_twin_regime_peer_v2 import regime_matrix, residual_frame  # noqa: E402
from run_health_twin_rolling_ablation_v1 import INDEPENDENT_FOLDS, LOCAL_FOLDS  # noqa: E402

PRIMARY = TARGETS[:3]
METHODS = ["static_no_trim", "static_trim", "distance", "fleet_median_residual", "fleet_median_raw"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_frozen(x, y, turbine, train, alpha):
    initial = fit_ridge(x[train], y[train], alpha=10.0)
    keep, _ = robust_training_keep(y[train] - predict(initial, x[train]), turbine[train])
    model = fit_ridge(x[train][keep], y[train][keep], alpha=alpha)
    _, scales = robust_training_keep(y[train] - predict(model, x[train]), turbine[train])
    return model, scales


def prepare(data, ambient, turbines):
    x, _ = make_feature_matrix(data, turbines, ambient)
    desc, _ = regime_matrix(data, ambient)
    eligible = (data.has_temperature_record & data.active_power_mean.between(0, 2350)
                & data.wind_speed_mean.between(4, 20) & data.generator_speed_mean.ge(100)
                & data.power_reference_end.ge(2200) & np.isfinite(x).all(axis=1)
                & np.isfinite(desc).all(axis=1)).to_numpy()
    return x, desc, eligible


def states(frame, raw_frame, scales, bandwidth, turbines):
    """Specific states of all five methods on the primary scoring rows."""
    times = pd.Index(frame.timestamp.unique()).sort_values()
    a = times.get_indexer(frame.timestamp)
    b = pd.Index(turbines).get_indexer(frame.turbine_id)
    r = np.full((len(times), len(turbines)), np.nan)
    raw = np.full_like(r, np.nan)
    desc = np.full((*r.shape, 7), np.nan)
    r[a, b] = frame.residual.to_numpy()
    raw[a, b] = raw_frame.residual.to_numpy()
    desc[a, b, :] = frame[[f"regime_{i}" for i in range(7)]].to_numpy()
    candidates = np.isfinite(r[:, None, :]) & ~np.eye(len(turbines), dtype=bool)[None, :, :]
    v = np.where(candidates, r[:, None, :], np.nan)
    peer_scale = np.array([scales[t] for t in turbines])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        center = np.nanmedian(v, axis=2)
        scale = np.maximum(np.maximum(1.4826 * np.nanmedian(abs(v - center[:, :, None]), axis=2),
                                      np.nanmedian(np.where(candidates, peer_scale[None, None, :], np.nan), axis=2)), 0.25)
        keep = candidates & (abs(v - center[:, :, None]) <= 4.5 * scale[:, :, None])
        midpoint = np.nanmedian(np.where(keep, v, np.nan), axis=2)
        fleet_residual = np.nanmedian(r, axis=1)
        fleet_raw = np.nanmedian(raw, axis=1)
    count = keep.sum(axis=2)
    sorted_values = np.sort(np.where(keep, v, np.inf), axis=2, kind="stable")
    distances = np.sqrt(np.sum((desc[:, :, None, :] - desc[:, None, :, :]) ** 2, axis=3))
    kernel = np.where(keep, -0.5 * (distances / bandwidth) ** 2, -np.inf)
    maximum = np.max(kernel, axis=2, keepdims=True)
    weights = np.exp(kernel - np.where(np.isfinite(maximum), maximum, 0))
    order = np.argsort(np.where(keep, v, np.inf), axis=2, kind="stable")
    cumulative = np.cumsum(np.take_along_axis(weights, order, axis=2), axis=2)
    index = np.argmax(cumulative >= weights.sum(axis=2, keepdims=True) / 2, axis=2)
    distance = np.take_along_axis(sorted_values, index[:, :, None], axis=2)[:, :, 0]
    valid = np.isfinite(r) & (candidates.sum(axis=2) >= 3) & (count >= 3)
    out = {
        "static_no_trim": r - center,
        "static_trim": r - midpoint,
        "distance": r - distance,
        "fleet_median_residual": r - fleet_residual[:, None],
        "fleet_median_raw": raw - fleet_raw[:, None],
    }
    return times, valid, out


def daily_cube(times, valid, specific, start, end):
    days = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    ti, _ = np.where(valid)
    day = days.get_indexer(pd.DatetimeIndex(times.to_numpy()[ti]).floor("D"))
    assert (day >= 0).all()
    cube = np.zeros((len(days), len(METHODS), 3))
    for j, name in enumerate(METHODS):
        val = specific[name][valid]
        cube[:, j, 0] = np.bincount(day, minlength=len(days))
        cube[:, j, 1] = np.bincount(day, weights=val, minlength=len(days))
        cube[:, j, 2] = np.bincount(day, weights=val * val, minlength=len(days))
    return cube


def sd(stats):
    if (stats[..., 0] <= 0).any():
        raise ValueError("Bootstrap draw has zero support")
    return np.sqrt(np.maximum(stats[..., 2] / stats[..., 0] - (stats[..., 1] / stats[..., 0]) ** 2, 0))


def boot(cube, block, seed, num, dens):
    rng = np.random.default_rng(seed)
    n = len(cube)
    starts = np.arange(n - block + 1)
    draws = []
    for _ in range(2000):
        selected = []
        while len(selected) < n:
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block))
        s = sd(cube[np.array(selected[:n])].sum(axis=0))
        draws.append([100 * (s[num] / s[d] - 1) for d in dens])
    return np.quantile(np.array(draws), [0.025, 0.975], axis=0)


def main():
    out = ROOT / "results"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    path = PROJECT / "data/research_layers/health_twin_base_v1.csv.gz"
    manifest = json.loads(path.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(path)
    assert digest == manifest["output_sha256"], "health layer does not match its manifest"
    data = pd.read_csv(path, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(
        ["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data.turbine_id.unique())
    turbine = data.turbine_id.to_numpy()
    time = data.timestamp

    rows = []
    for protocol, folds, ambient, offset in [("independent", INDEPENDENT_FOLDS, "met_temperature_mean", 0),
                                             ("local", LOCAL_FOLDS, "local_ambient_temp_mean", 5_000_000)]:
        locked_all = pd.read_csv(PROJECT / f"data/experiment_results/health_twin_trim_distance_ablation_{protocol}_v1/fold_target_method_results.csv")
        x, rawdesc, eligible = prepare(data, ambient, turbines)
        for fi, fold in enumerate(folds):
            train = (time.ge(fold["start"]) & time.lt(fold["train_end"])).to_numpy()
            test = (time.ge(fold["valid_end"]) & time.lt(fold["test_end"])).to_numpy()
            scale = np.std(rawdesc[eligible & train], axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            descriptor = (rawdesc - np.mean(rawdesc[eligible & train], axis=0)) / scale
            for ki, target in enumerate(TARGETS):
                locked = locked_all[(locked_all.fold == fold["fold"]) & (locked_all.target == target)].set_index("method")
                alpha = float(locked.loc["distance_trim", "normal_model_alpha"])
                bandwidth = float(locked.loc["distance_trim", "selected_distance_bandwidth"])
                y = data[target].to_numpy()
                usable = eligible & np.isfinite(y)
                model, scales = fit_frozen(x, y, turbine, usable & train, alpha)
                mask = usable & test
                frame = residual_frame(data, y[mask] - predict(model, x[mask]), mask, descriptor)
                raw_frame = residual_frame(data, y[mask], mask, descriptor)
                assert (raw_frame.timestamp.to_numpy() == frame.timestamp.to_numpy()).all()
                assert (raw_frame.turbine_id.to_numpy() == frame.turbine_id.to_numpy()).all()
                times, valid, specific = states(frame, raw_frame, scales, bandwidth, turbines)
                spreads = {m: float(np.std(specific[m][valid])) for m in METHODS}
                for name, saved in [("static_no_trim", "static_no_trim"), ("static_trim", "static_trim"), ("distance", "distance_trim")]:
                    np.testing.assert_allclose(spreads[name], locked.loc[saved, "specific_std"], atol=1e-9, rtol=1e-9)
                assert int(valid.sum()) == int(locked.loc["distance_trim", "score_rows"])
                cube = daily_cube(times, valid, specific, fold["valid_end"], fold["test_end"])
                ci = boot(cube, 7, 20261031 + offset + 10000 * fi + 100 * ki,
                          METHODS.index("distance"), [METHODS.index("fleet_median_residual"), METHODS.index("fleet_median_raw")])
                row = {"protocol": protocol, "fold": fold["fold"], "target": target, "rows": int(valid.sum()),
                       **{f"sd_{m}": spreads[m] for m in METHODS},
                       "distance_vs_fleet_residual_pct": 100 * (spreads["distance"] / spreads["fleet_median_residual"] - 1),
                       "distance_vs_fleet_residual_ci_low": float(ci[0, 0]), "distance_vs_fleet_residual_ci_high": float(ci[1, 0]),
                       "distance_vs_fleet_raw_pct": 100 * (spreads["distance"] / spreads["fleet_median_raw"] - 1),
                       "distance_vs_fleet_raw_ci_low": float(ci[0, 1]), "distance_vs_fleet_raw_ci_high": float(ci[1, 1]),
                       "fleet_residual_vs_static_trim_pct": 100 * (spreads["fleet_median_residual"] / spreads["static_trim"] - 1)}
                rows.append(row)
                print(f"{protocol} {fold['fold']} {target}: dist vs C1 {row['distance_vs_fleet_residual_pct']:+.2f}% "
                      f"[{ci[0,0]:+.2f}, {ci[1,0]:+.2f}] | dist vs C2 {row['distance_vs_fleet_raw_pct']:+.2f}% "
                      f"[{ci[0,1]:+.2f}, {ci[1,1]:+.2f}]", flush=True)

    out.mkdir()
    table = pd.DataFrame(rows)
    table.to_csv(out / "chesterman_fold_effects.csv", index=False)
    decision = []
    for target in PRIMARY:
        m = table[(table.protocol == "independent") & (table.target == target)]
        lower = int((m.distance_vs_fleet_residual_pct < 0).sum())
        below = int((m.distance_vs_fleet_residual_ci_high < 0).sum())
        decision.append({"target": target, "folds_lower_than_C1": f"{lower}/{len(m)}", "folds_ci_below_zero": f"{below}/{len(m)}",
                         "median_vs_C1_pct": float(m.distance_vs_fleet_residual_pct.median()),
                         "median_vs_C2_pct": float(m.distance_vs_fleet_raw_pct.median()),
                         "improves_on_C1": bool(lower == len(m) and below >= 3)})
    pd.DataFrame(decision).to_csv(out / "chesterman_decision.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(decision).to_string(index=False))


if __name__ == "__main__":
    main()
