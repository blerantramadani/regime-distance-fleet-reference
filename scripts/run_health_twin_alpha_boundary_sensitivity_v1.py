#!/usr/bin/env python3
"""Run a separate local sensitivity audit for ridge-grid boundary choices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_health_twin_baseline import (
    AMBIENT_SOURCES,
    SOURCE_COLUMNS,
    TARGETS,
    fit_ridge,
    mae,
    make_feature_matrix,
    predict,
    rmse,
    robust_training_keep,
)
from run_health_twin_regime_peer_v2 import regime_matrix, residual_frame
from run_health_twin_rolling_ablation_v1 import (
    ALPHA_GRID,
    INDEPENDENT_FOLDS,
    LOCAL_FOLDS,
)
from run_health_twin_trim_distance_ablation import (
    factorial_scores,
    moving_block_contrast,
    validation_bandwidth_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPANDED_ALPHA_GRID = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
]
METHOD = "distance_trim"
COMPARATOR = "static_trim"
BLOCK_DAYS = 7
BOOTSTRAP_ITERATIONS = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def cell(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def fit_expanded_window(
    features: np.ndarray,
    target: np.ndarray,
    turbine: np.ndarray,
    usable: np.ndarray,
    train_period: np.ndarray,
    valid_period: np.ndarray,
    test_period: np.ndarray,
) -> tuple[dict, list[dict]]:
    train_mask = usable & train_period
    valid_mask = usable & valid_period
    test_mask = usable & test_period
    x_train, y_train = features[train_mask], target[train_mask]
    x_valid, y_valid = features[valid_mask], target[valid_mask]
    if min(len(x_train), len(x_valid), int(test_mask.sum())) == 0:
        raise ValueError("Fold has an empty train, validation, or test partition.")

    initial = fit_ridge(x_train, y_train, alpha=10.0)
    retained, _ = robust_training_keep(
        y_train - predict(initial, x_train), turbine[train_mask]
    )
    candidates: list[dict] = []
    models = {}
    for alpha in EXPANDED_ALPHA_GRID:
        model = fit_ridge(x_train[retained], y_train[retained], alpha=alpha)
        validation_prediction = predict(model, x_valid)
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_rmse": rmse(y_valid, validation_prediction),
                "validation_mae": mae(y_valid, validation_prediction),
            }
        )
        models[alpha] = model
    selected = min(candidates, key=lambda row: row["validation_rmse"])
    alpha = float(selected["alpha"])
    model = models[alpha]
    _, residual_scales = robust_training_keep(
        y_train - predict(model, x_train), turbine[train_mask]
    )
    return (
        {
            "model": model,
            "alpha": alpha,
            "alpha_boundary": alpha
            in {min(EXPANDED_ALPHA_GRID), max(EXPANDED_ALPHA_GRID)},
            "validation_rmse": float(selected["validation_rmse"]),
            "validation_mae": float(selected["validation_mae"]),
            "train_rows": int(len(x_train)),
            "train_rows_retained": int(retained.sum()),
            "valid_mask": valid_mask,
            "test_mask": test_mask,
            "residual_scales": residual_scales,
        },
        candidates,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/research_layers/health_twin_base_v1.csv.gz",
    )
    parser.add_argument("--protocol", choices=["independent", "local"], required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "RIDGE_BOUNDARY_SENSITIVITY.md"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"{report_path} exists; use --overwrite to replace it")

    folds = INDEPENDENT_FOLDS if args.protocol == "independent" else LOCAL_FOLDS
    ambient_source = (
        "independent_meteorology"
        if args.protocol == "independent"
        else "turbine_local_proxy"
    )
    ambient_column = AMBIENT_SOURCES[ambient_source]["column"]
    reference_methods_path = args.reference_dir / "fold_target_method_results.csv"
    reference_contrasts_path = args.reference_dir / "mechanism_contrasts.csv"
    reference_methods = pd.read_csv(reference_methods_path)
    reference_methods = reference_methods.loc[
        reference_methods["method"].eq(METHOD)
    ].set_index(["fold", "target"])
    reference_contrasts = pd.read_csv(reference_contrasts_path)
    reference_contrasts = reference_contrasts.loc[
        reference_contrasts["contrast"].eq("distance")
        & reference_contrasts["block_days"].eq(BLOCK_DAYS)
    ].set_index(["fold", "target"])

    data = pd.read_csv(args.input, usecols=SOURCE_COLUMNS, parse_dates=["timestamp"])
    data = data.sort_values(["timestamp", "turbine_id"], kind="stable").reset_index(
        drop=True
    )
    turbine_ids = sorted(data["turbine_id"].dropna().astype(str).unique().tolist())
    turbine = data["turbine_id"].astype(str).to_numpy()
    features, feature_names = make_feature_matrix(data, turbine_ids, ambient_column)
    raw_regime, regime_names = regime_matrix(data, ambient_column)
    base_eligible = (
        data["has_temperature_record"]
        & data["active_power_mean"].between(0.0, 2350.0)
        & data["wind_speed_mean"].between(4.0, 20.0)
        & data["generator_speed_mean"].ge(100.0)
        & data["power_reference_end"].ge(2200.0)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(raw_regime).all(axis=1)
    ).to_numpy()
    time = data["timestamp"]

    validation_rows: list[dict] = []
    result_rows: list[dict] = []
    for fold_index, raw_fold in enumerate(folds):
        fold = {
            key: (pd.Timestamp(value) if key != "fold" else value)
            for key, value in raw_fold.items()
        }
        train_period = time.ge(fold["start"]).to_numpy() & time.lt(
            fold["train_end"]
        ).to_numpy()
        valid_period = time.ge(fold["train_end"]).to_numpy() & time.lt(
            fold["valid_end"]
        ).to_numpy()
        test_period = time.ge(fold["valid_end"]).to_numpy() & time.lt(
            fold["test_end"]
        ).to_numpy()
        eligible_train = base_eligible & train_period
        regime_mean = np.mean(raw_regime[eligible_train], axis=0)
        regime_scale = np.std(raw_regime[eligible_train], axis=0)
        regime_scale = np.where(regime_scale > 1e-8, regime_scale, 1.0)
        regime = (raw_regime - regime_mean) / regime_scale

        for target_index, target_name in enumerate(TARGETS):
            target = data[target_name].to_numpy(dtype=float)
            usable = base_eligible & np.isfinite(target)
            fitted, candidates = fit_expanded_window(
                features,
                target,
                turbine,
                usable,
                train_period,
                valid_period,
                test_period,
            )
            for candidate in candidates:
                validation_rows.append(
                    {
                        "protocol": args.protocol,
                        "fold": fold["fold"],
                        "target": target_name,
                        **candidate,
                    }
                )

            validation_residual = target[fitted["valid_mask"]] - predict(
                fitted["model"], features[fitted["valid_mask"]]
            )
            test_prediction = predict(
                fitted["model"], features[fitted["test_mask"]]
            )
            test_residual = target[fitted["test_mask"]] - test_prediction
            validation_frame = residual_frame(
                data, validation_residual, fitted["valid_mask"], regime
            )
            test_frame = residual_frame(
                data, test_residual, fitted["test_mask"], regime
            )
            bandwidth_grid = validation_bandwidth_grid(
                validation_frame, target_name, fitted["residual_scales"]
            )
            selected_bandwidth = float(
                pd.DataFrame(bandwidth_grid)
                .sort_values(
                    ["specific_median_abs", "specific_std", "distance_bandwidth"],
                    kind="stable",
                )
                .iloc[0]["distance_bandwidth"]
            )
            scores = factorial_scores(
                test_frame,
                target_name,
                fitted["residual_scales"],
                selected_bandwidth,
            )
            contrast = moving_block_contrast(
                scores,
                METHOD,
                COMPARATOR,
                BLOCK_DAYS,
                BOOTSTRAP_ITERATIONS,
                seed=(
                    20260901
                    + 10000 * fold_index
                    + 1000 * target_index
                    + 100
                    + BLOCK_DAYS
                ),
            )
            locked_method = reference_methods.loc[(fold["fold"], target_name)]
            locked_contrast = reference_contrasts.loc[(fold["fold"], target_name)]
            test_actual = target[fitted["test_mask"]]
            result_rows.append(
                {
                    "protocol": args.protocol,
                    "fold": fold["fold"],
                    "target": target_name,
                    "locked_alpha": float(locked_method["normal_model_alpha"]),
                    "locked_alpha_boundary": bool(
                        locked_method["normal_model_alpha_boundary"]
                    ),
                    "expanded_alpha": fitted["alpha"],
                    "expanded_alpha_boundary": fitted["alpha_boundary"],
                    "alpha_changed": not np.isclose(
                        fitted["alpha"],
                        float(locked_method["normal_model_alpha"]),
                        rtol=0.0,
                        atol=0.0,
                    ),
                    "locked_validation_rmse": float(
                        locked_method["normal_model_validation_rmse"]
                    ),
                    "expanded_validation_rmse": fitted["validation_rmse"],
                    "validation_rmse_change_pct": float(
                        (
                            fitted["validation_rmse"]
                            / float(locked_method["normal_model_validation_rmse"])
                            - 1.0
                        )
                        * 100.0
                    ),
                    "locked_test_rmse": float(locked_method["normal_model_test_rmse"]),
                    "expanded_test_rmse": rmse(test_actual, test_prediction),
                    "test_rmse_change_pct": float(
                        (
                            rmse(test_actual, test_prediction)
                            / float(locked_method["normal_model_test_rmse"])
                            - 1.0
                        )
                        * 100.0
                    ),
                    "locked_bandwidth": float(
                        locked_method["selected_distance_bandwidth"]
                    ),
                    "expanded_bandwidth": selected_bandwidth,
                    "bandwidth_changed": not np.isclose(
                        selected_bandwidth,
                        float(locked_method["selected_distance_bandwidth"]),
                        rtol=0.0,
                        atol=0.0,
                    ),
                    "locked_distance_std_change_pct": float(
                        locked_contrast["observed_std_change_pct"]
                    ),
                    "expanded_distance_std_change_pct": float(
                        contrast["observed_std_change_pct"]
                    ),
                    "expanded_minus_locked_distance_pct_points": float(
                        contrast["observed_std_change_pct"]
                        - float(locked_contrast["observed_std_change_pct"])
                    ),
                    "locked_ci95_low_pct": float(locked_contrast["ci95_low_pct"]),
                    "locked_ci95_high_pct": float(locked_contrast["ci95_high_pct"]),
                    "expanded_ci95_low_pct": float(contrast["ci95_low_pct"]),
                    "expanded_ci95_high_pct": float(contrast["ci95_high_pct"]),
                    "score_rows": int(
                        scores.loc[scores["method"].eq(METHOD)].shape[0]
                    ),
                }
            )
            print(
                f"[{fold['fold']}][{target_name}] "
                f"alpha {locked_method['normal_model_alpha']:g}->{fitted['alpha']:g}; "
                f"distance {locked_contrast['observed_std_change_pct']:.3f}%"
                f"->{contrast['observed_std_change_pct']:.3f}%",
                flush=True,
            )

    validation = pd.DataFrame(validation_rows)
    results = pd.DataFrame(result_rows)
    summary_rows = []
    for target_name, group in results.groupby("target", sort=False):
        summary_rows.append(
            {
                "protocol": args.protocol,
                "target": target_name,
                "folds": int(len(group)),
                "locked_alpha_boundary_folds": int(
                    group["locked_alpha_boundary"].sum()
                ),
                "alpha_changed_folds": int(group["alpha_changed"].sum()),
                "expanded_alpha_boundary_folds": int(
                    group["expanded_alpha_boundary"].sum()
                ),
                "median_validation_rmse_change_pct": float(
                    group["validation_rmse_change_pct"].median()
                ),
                "min_test_rmse_change_pct": float(group["test_rmse_change_pct"].min()),
                "median_test_rmse_change_pct": float(
                    group["test_rmse_change_pct"].median()
                ),
                "max_test_rmse_change_pct": float(group["test_rmse_change_pct"].max()),
                "locked_distance_folds_lower": int(
                    group["locked_distance_std_change_pct"].lt(0.0).sum()
                ),
                "expanded_distance_folds_lower": int(
                    group["expanded_distance_std_change_pct"].lt(0.0).sum()
                ),
                "locked_distance_ci_below_zero": int(
                    group["locked_ci95_high_pct"].lt(0.0).sum()
                ),
                "expanded_distance_ci_below_zero": int(
                    group["expanded_ci95_high_pct"].lt(0.0).sum()
                ),
                "locked_median_distance_std_change_pct": float(
                    group["locked_distance_std_change_pct"].median()
                ),
                "expanded_median_distance_std_change_pct": float(
                    group["expanded_distance_std_change_pct"].median()
                ),
                "max_abs_distance_change_pct_points": float(
                    group["expanded_minus_locked_distance_pct_points"].abs().max()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)

    validation_path = args.output_dir / "alpha_validation_grid.csv"
    results_path = args.output_dir / "fold_target_sensitivity.csv"
    summary_path = args.output_dir / "target_summary.csv"
    validation.to_csv(validation_path, index=False)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    report = [
        f"# Ridge-boundary sensitivity — {args.protocol}",
        "",
        "**Local-only post-hoc reviewer diagnostic; not external confirmation.**",
        "",
        "Only the ridge candidate grid was expanded. The gate, folds, targets, "
        "features, robust training trim, all-peer bandwidth rule, common-support "
        "comparator, and seven-day 2,000-iteration block bootstrap are unchanged.",
        "",
        markdown_table(
            summary,
            [
                "target",
                "folds",
                "locked_alpha_boundary_folds",
                "alpha_changed_folds",
                "expanded_alpha_boundary_folds",
                "median_validation_rmse_change_pct",
                "median_test_rmse_change_pct",
                "locked_distance_folds_lower",
                "expanded_distance_folds_lower",
                "locked_distance_ci_below_zero",
                "expanded_distance_ci_below_zero",
                "locked_median_distance_std_change_pct",
                "expanded_median_distance_std_change_pct",
                "max_abs_distance_change_pct_points",
            ],
        ),
        "",
        "Negative distance changes indicate lower specific-state spread for "
        "`distance_trim` relative to `static_trim`. Test data select nothing.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    output_files = [
        "RIDGE_BOUNDARY_SENSITIVITY.md",
        "alpha_validation_grid.csv",
        "fold_target_sensitivity.csv",
        "target_summary.csv",
    ]
    provenance = {
        "artifact": f"health_twin_alpha_boundary_sensitivity_{args.protocol}_v1",
        "confidentiality": "local-only aggregate reviewer diagnostic",
        "status": "frozen post-hoc sensitivity; does not replace primary artifacts",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "protocol": args.protocol,
        "reference_dir": str(args.reference_dir),
        "reference_methods_sha256": sha256_file(reference_methods_path),
        "reference_contrasts_sha256": sha256_file(reference_contrasts_path),
        "inputs": {
            "reference_methods": {
                "path": str(reference_methods_path),
                "sha256": sha256_file(reference_methods_path),
            },
            "reference_contrasts": {
                "path": str(reference_contrasts_path),
                "sha256": sha256_file(reference_contrasts_path),
            },
        },
        "original_alpha_grid": ALPHA_GRID,
        "expanded_alpha_grid": EXPANDED_ALPHA_GRID,
        "selection": "validation RMSE selects alpha; validation median absolute distance_trim state selects bandwidth; test selects nothing",
        "normal_model_features": feature_names,
        "regime_distance_features": regime_names,
        "block_bootstrap": {
            "block_days": BLOCK_DAYS,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed_rule": "same deterministic distance-contrast seed as frozen mechanism audit",
        },
        "outputs": {
            name: sha256_file(args.output_dir / name) for name in output_files
        },
        "claim_boundary": "ridge-grid robustness only; no external confirmation or fault-performance claim",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(native(provenance), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
