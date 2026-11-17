"""H1 (spread) and H2 (placebo) of the 2021-2022 confirmation protocol.

Uses the unchanged 2015 functions: fit_window (ridge alpha on the validation month),
validation_bandwidth_grid (bandwidth rule), the specific states of the fleet-median
baseline runner and the placebo functions. The fold table is the only new element.

    python3 run_h1_h2.py --data <layer.csv.gz> --folds N --out <dir>
    python3 run_h1_h2.py --data <2015 layer> --folds L2015 --check <2015 trim-distance local results> --out <dir> --no-placebo
"""
import argparse
import hashlib
import importlib.util
import json
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
pl = load("run_placebo", ANALYSIS / "posthoc_2026-10-30" / "placebo" / "run_placebo.py")
from run_health_twin_baseline import SOURCE_COLUMNS, TARGETS, predict  # noqa: E402
from run_health_twin_regime_peer_v2 import residual_frame  # noqa: E402
from run_health_twin_rolling_ablation_v1 import LOCAL_FOLDS, fit_window  # noqa: E402
from run_health_twin_trim_distance_ablation import validation_bandwidth_grid  # noqa: E402

AMBIENT = "local_ambient_temp_mean"
N_FOLDS = [{"fold": f"N{i + 1}", "start": "2021-04-01", "train_end": train_end, "valid_end": valid_end, "test_end": test_end}
           for i, (train_end, valid_end, test_end) in enumerate([
               ("2021-08-01", "2021-09-01", "2021-10-01"), ("2021-09-01", "2021-10-01", "2021-11-01"),
               ("2021-10-01", "2021-11-01", "2021-12-01"), ("2021-11-01", "2021-12-01", "2022-01-01"),
               ("2021-12-01", "2022-01-01", "2022-02-01"), ("2022-01-01", "2022-02-01", "2022-03-01"),
               ("2022-02-01", "2022-03-01", "2022-04-01"), ("2022-03-01", "2022-04-01", "2022-05-01")])]
PERMUTATIONS = 100
PRIMARY = TARGETS[:3]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", choices=["N", "L2015"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--no-placebo", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.data.with_name("health_twin_base_v1_manifest.json").read_text())
    digest = sha(args.data)
    assert digest == manifest["output_sha256"], "processed table does not match its manifest"
    folds = N_FOLDS if args.folds == "N" else LOCAL_FOLDS
    locked_all = pd.read_csv(args.check / "fold_target_method_results.csv") if args.check else None

    data = pd.read_csv(args.data, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(
        ["timestamp", "turbine_id"], kind="stable").reset_index(drop=True)
    turbines = sorted(data.turbine_id.unique())
    turbine = data.turbine_id.to_numpy()
    time = data.timestamp
    x, rawdesc, eligible = rc.prepare(data, AMBIENT, turbines)
    i_dist, i_static, i_c1 = (rc.METHODS.index(m) for m in ["distance", "static_trim", "fleet_median_residual"])

    rows, checks = [], []
    for fi, fold in enumerate(folds):
        train = (time.ge(fold["start"]) & time.lt(fold["train_end"])).to_numpy()
        valid = (time.ge(fold["train_end"]) & time.lt(fold["valid_end"])).to_numpy()
        test = (time.ge(fold["valid_end"]) & time.lt(fold["test_end"])).to_numpy()
        scale = np.std(rawdesc[eligible & train], axis=0)
        descriptor = (rawdesc - np.mean(rawdesc[eligible & train], axis=0)) / np.where(scale > 1e-8, scale, 1.0)
        for ki, target in enumerate(TARGETS):
            y = data[target].to_numpy(dtype=float)
            usable = eligible & np.isfinite(y)
            fitted = fit_window(x, y, turbine, usable, train, valid, test)
            scales = fitted["residual_scales"]
            vmask = fitted["valid_mask"]
            vframe = residual_frame(data, y[vmask] - predict(fitted["model"], x[vmask]), vmask, descriptor)
            grid = pd.DataFrame(validation_bandwidth_grid(vframe, target, scales))
            bandwidth = float(grid.sort_values(["specific_median_abs", "specific_std", "distance_bandwidth"], kind="stable").iloc[0]["distance_bandwidth"])
            mask = usable & test
            frame = residual_frame(data, y[mask] - predict(fitted["model"], x[mask]), mask, descriptor)
            raw_frame = residual_frame(data, y[mask], mask, descriptor)
            times, ok, specific = rc.states(frame, raw_frame, scales, bandwidth, turbines)
            sd = {m: float(np.std(specific[m][ok])) for m in rc.METHODS}
            if locked_all is not None:
                locked = locked_all[(locked_all.fold == fold["fold"]) & (locked_all.target == target)].set_index("method")
                checks.append({"fold": fold["fold"], "target": target,
                               "alpha_match": fitted["alpha"] == float(locked.loc["distance_trim", "normal_model_alpha"]),
                               "bandwidth_match": bandwidth == float(locked.loc["distance_trim", "selected_distance_bandwidth"]),
                               "sd_distance_diff": abs(sd["distance"] - float(locked.loc["distance_trim", "specific_std"])),
                               "sd_static_diff": abs(sd["static_trim"] - float(locked.loc["static_trim", "specific_std"])),
                               "rows_match": int(ok.sum()) == int(locked.loc["distance_trim", "score_rows"])})
                print(checks[-1], flush=True)
                continue
            cube = rc.daily_cube(times, ok, specific, fold["valid_end"], fold["test_end"])
            ci = rc.boot(cube, 7, 20270101 + 10000 * fi + 100 * ki, i_dist, [i_static, i_c1])
            row = {"fold": fold["fold"], "target": target, "rows": int(ok.sum()), "alpha": fitted["alpha"], "bandwidth": bandwidth,
                   **{f"sd_{m}": sd[m] for m in rc.METHODS},
                   "distance_vs_static_pct": 100 * (sd["distance"] / sd["static_trim"] - 1),
                   "ci_low": float(ci[0, 0]), "ci_high": float(ci[1, 0]),
                   "distance_vs_C1_pct": 100 * (sd["distance"] / sd["fleet_median_residual"] - 1),
                   "C1_ci_low": float(ci[0, 1]), "C1_ci_high": float(ci[1, 1]),
                   "distance_vs_C2_pct": 100 * (sd["distance"] / sd["fleet_median_raw"] - 1)}
            if not args.no_placebo:
                arrays = pl.peer_arrays(frame, scales, bandwidth, turbines)
                sd_dist = pl.spread(arrays, pl.weighted_lower_median(arrays, arrays["weights"]))
                np.testing.assert_allclose(sd_dist, sd["distance"], atol=1e-9, rtol=1e-9)
                placebo = np.array([pl.spread(arrays, pl.weighted_lower_median(
                    arrays, pl.permute_weights(arrays["weights"], arrays["keep"], np.random.default_rng(20270201 + 10000 * fi + 100 * ki + p))))
                    for p in range(PERMUTATIONS)])
                le = int((placebo <= sd_dist).sum())
                row.update({"sd_placebo_median": float(np.median(placebo)), "regime_effect_pct": 100 * (sd_dist / float(np.median(placebo)) - 1),
                            "permutations_le_distance": le, "permutation_p": (1 + le) / (PERMUTATIONS + 1)})
            rows.append(row)
            print(f"{fold['fold']} {target}: alpha={fitted['alpha']:g} bw={bandwidth:g} distance vs static {row['distance_vs_static_pct']:+.3f}% "
                  f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]" + (f" placebo p={row['permutation_p']:.3f}" if "permutation_p" in row else ""), flush=True)

    args.out.mkdir(parents=True)
    if checks:
        pd.DataFrame(checks).to_csv(args.out / "reproduction_2015_L.csv", index=False)
        return
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "h1_h2_fold_effects.csv", index=False)
    decision = []
    for target in PRIMARY:
        m = table[table.target == target]
        lower, below = int((m.distance_vs_static_pct < 0).sum()), int((m.ci_high < 0).sum())
        rec = {"target": target, "folds_lower": f"{lower}/{len(m)}", "folds_ci_below_zero": f"{below}/{len(m)}",
               "median_change_pct": float(m.distance_vs_static_pct.median()), "H1_holds": bool(lower >= 7 and below >= 6)}
        if "permutation_p" in m:
            zero = int((m.permutations_le_distance == 0).sum())
            rec.update({"folds_below_all_placebo": f"{zero}/{len(m)}", "H2_holds": bool(zero >= 6)})
        decision.append(rec)
    dec = pd.DataFrame(decision)
    dec.to_csv(args.out / "h1_h2_decision.csv", index=False)
    summary = {"H1_confirmed": bool(dec.H1_holds.sum() >= 2), "input_sha256": digest, "runner_sha256": sha(__file__),
               "protocol_sha256": sha(Path(__file__).resolve().parents[1] / "PROTOCOL.md")}
    (args.out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(dec.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
