"""Permuted-weight placebo control for the regime-distance fleet reference (see PROTOCOL.md).

Reuses the frozen 2026-09-18 setup: same health layer, folds, alpha, bandwidth, trim and
three-peer minimum. Only the assignment of the Gaussian weights to the retained peers is
randomised. Reads the processed table in ../../data and writes only to ./results.
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

PERMUTATIONS = 100
PRIMARY = TARGETS[:3]


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


def peer_arrays(frame, scales, bandwidth, turbines):
    """Leave-one-out trim and Gaussian weights, identical to the 2026-09-18 vectorised scorer."""
    times = pd.Index(frame.timestamp.unique()).sort_values()
    a = times.get_indexer(frame.timestamp)
    b = pd.Index(turbines).get_indexer(frame.turbine_id)
    r = np.full((len(times), len(turbines)), np.nan)
    desc = np.full((*r.shape, 7), np.nan)
    r[a, b] = frame.residual.to_numpy()
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
    count = keep.sum(axis=2)
    sorted_values = np.sort(np.where(keep, v, np.inf), axis=2, kind="stable")
    lower = np.take_along_axis(sorted_values, np.maximum((count - 1) // 2, 0)[:, :, None], axis=2)[:, :, 0]
    distances = np.sqrt(np.sum((desc[:, :, None, :] - desc[:, None, :, :]) ** 2, axis=3))
    kernel = np.where(keep, -0.5 * (distances / bandwidth) ** 2, -np.inf)
    maximum = np.max(kernel, axis=2, keepdims=True)
    weights = np.exp(kernel - np.where(np.isfinite(maximum), maximum, 0))
    order = np.argsort(np.where(keep, v, np.inf), axis=2, kind="stable")
    valid = np.isfinite(r) & (candidates.sum(axis=2) >= 3) & (count >= 3)
    return dict(r=r, keep=keep, sorted_values=sorted_values, order=order, weights=weights,
                midpoint=midpoint, lower=lower, valid=valid)


def weighted_lower_median(arrays, weights):
    sorted_weights = np.take_along_axis(weights, arrays["order"], axis=2)
    cumulative = np.cumsum(sorted_weights, axis=2)
    index = np.argmax(cumulative >= weights.sum(axis=2, keepdims=True) / 2, axis=2)
    return np.take_along_axis(arrays["sorted_values"], index[:, :, None], axis=2)[:, :, 0]


def permute_weights(weights, keep, rng):
    """Shuffle each row's weights among its retained peers; non-retained peers keep weight zero."""
    t, n, m = weights.shape
    position = np.where(keep, np.arange(m)[None, None, :], m)
    kept_first = np.argsort(position, axis=2, kind="stable")
    values = np.take_along_axis(weights, kept_first, axis=2)
    count = keep.sum(axis=2)
    keys = np.where(np.arange(m)[None, None, :] < count[:, :, None], rng.random((t, n, m)), np.inf)
    shuffle = np.argsort(keys, axis=2, kind="stable")
    shuffled = np.take_along_axis(values, shuffle, axis=2)
    out = np.zeros_like(weights)
    np.put_along_axis(out, kept_first, shuffled, axis=2)
    return out


def spread(arrays, reference):
    valid = arrays["valid"]
    return float(np.std(arrays["r"][valid] - reference[valid]))


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
    saved = pd.read_csv(PROJECT / "repairs_2026-09-18/rolling/fold_effects.csv")

    rows, draws = [], []
    for protocol, folds, ambient, offset in [("independent", INDEPENDENT_FOLDS, "met_temperature_mean", 0),
                                             ("local", LOCAL_FOLDS, "local_ambient_temp_mean", 5_000_000)]:
        locked_all = pd.read_csv(PROJECT / f"data/experiment_results/health_twin_trim_distance_ablation_{protocol}_v1/fold_target_method_results.csv")
        x, raw, eligible = prepare(data, ambient, turbines)
        for fi, fold in enumerate(folds):
            train = (time.ge(fold["start"]) & time.lt(fold["train_end"])).to_numpy()
            test = (time.ge(fold["valid_end"]) & time.lt(fold["test_end"])).to_numpy()
            scale = np.std(raw[eligible & train], axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            descriptor = (raw - np.mean(raw[eligible & train], axis=0)) / scale
            for ki, target in enumerate(TARGETS):
                locked = locked_all[(locked_all.fold == fold["fold"]) & (locked_all.target == target)].set_index("method")
                alpha = float(locked.loc["distance_trim", "normal_model_alpha"])
                bandwidth = float(locked.loc["distance_trim", "selected_distance_bandwidth"])
                y = data[target].to_numpy()
                usable = eligible & np.isfinite(y)
                model, scales = fit_frozen(x, y, turbine, usable & train, alpha)
                mask = usable & test
                frame = residual_frame(data, y[mask] - predict(model, x[mask]), mask, descriptor)
                arrays = peer_arrays(frame, scales, bandwidth, turbines)

                sd_mid = spread(arrays, arrays["midpoint"])
                sd_low = spread(arrays, arrays["lower"])
                sd_dist = spread(arrays, weighted_lower_median(arrays, arrays["weights"]))
                # Reproduction of the saved September values.
                np.testing.assert_allclose(sd_mid, locked.loc["static_trim", "specific_std"], atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(sd_dist, locked.loc["distance_trim", "specific_std"], atol=1e-9, rtol=1e-9)
                ref = saved[(saved.protocol == protocol) & (saved.fold == fold["fold"]) & (saved.target == target)].iloc[0]
                np.testing.assert_allclose(100 * (sd_dist / sd_low - 1), ref.harmonized_effect_pct, atol=1e-9)
                assert int(arrays["valid"].sum()) == int(locked.loc["distance_trim", "score_rows"])

                placebo = []
                for p in range(PERMUTATIONS):
                    rng = np.random.default_rng(20261030 + offset + 10000 * fi + 100 * ki + p)
                    sd_p = spread(arrays, weighted_lower_median(arrays, permute_weights(arrays["weights"], arrays["keep"], rng)))
                    placebo.append(sd_p)
                    draws.append({"protocol": protocol, "fold": fold["fold"], "target": target, "permutation": p, "placebo_sd": sd_p})
                placebo = np.array(placebo)
                le = int((placebo <= sd_dist).sum())
                rows.append({
                    "protocol": protocol, "fold": fold["fold"], "target": target,
                    "rows": int(arrays["valid"].sum()), "alpha": alpha, "bandwidth": bandwidth,
                    "sd_static_midpoint": sd_mid, "sd_static_lower": sd_low, "sd_distance": sd_dist,
                    "sd_placebo_median": float(np.median(placebo)),
                    "distance_vs_midpoint_pct": 100 * (sd_dist / sd_mid - 1),
                    "distance_vs_lower_pct": 100 * (sd_dist / sd_low - 1),
                    "placebo_vs_lower_median_pct": float(np.median(100 * (placebo / sd_low - 1))),
                    "placebo_vs_lower_q025_pct": float(np.quantile(100 * (placebo / sd_low - 1), 0.025)),
                    "placebo_vs_lower_q975_pct": float(np.quantile(100 * (placebo / sd_low - 1), 0.975)),
                    "regime_effect_pct": 100 * (sd_dist / float(np.median(placebo)) - 1),
                    "permutations_le_distance": le,
                    "permutation_p": (1 + le) / (PERMUTATIONS + 1),
                })
                r = rows[-1]
                print(f"{protocol} {fold['fold']} {target}: distance {r['distance_vs_lower_pct']:+.3f}% | "
                      f"placebo {r['placebo_vs_lower_median_pct']:+.3f}% | regime {r['regime_effect_pct']:+.3f}% | p={r['permutation_p']:.3f}",
                      flush=True)

    out.mkdir()
    table = pd.DataFrame(rows)
    table.to_csv(out / "placebo_fold_effects.csv", index=False)
    pd.DataFrame(draws).to_csv(out / "placebo_draws.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    decision = []
    for target in PRIMARY:
        m = table[(table.protocol == "independent") & (table.target == target)]
        below = int((m.sd_distance < m.sd_placebo_median).sum())
        significant = int((m.permutation_p <= 0.05).sum())
        decision.append({"target": target, "folds_distance_below_placebo": f"{below}/{len(m)}",
                         "folds_p_le_0.05": f"{significant}/{len(m)}",
                         "median_regime_effect_pct": float(m.regime_effect_pct.median()),
                         "supported": bool(below == len(m) and significant >= 3)})
    pd.DataFrame(decision).to_csv(out / "placebo_decision.csv", index=False)
    provenance = {"input_sha256": digest, "runner_sha256": sha(__file__), "protocol_sha256": sha(ROOT / "PROTOCOL.md"),
                  "permutations": PERMUTATIONS, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "outputs": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "verification.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(decision).to_string(index=False))


if __name__ == "__main__":
    main()
