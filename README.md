# Regime-distance fleet reference for label-free thermal condition assessment

Analysis code for the manuscript “Regime-Distance Fleet References for Label-Free Thermal Monitoring of Wind Turbines: Detection Depends on How Alert Rates Are Matched” by Blerant Ramadani, Vangel Fustić and Goce Dzambazovski.

The code compares each turbine's thermal normal-behaviour residual with a leave-one-turbine-out reference built from all eligible peers and weighted by Gaussian operating-regime distance. It evaluates the reference in chronological folds, separates fleet-common and turbine-specific states, and tests alert rates and detection with synthetic shifts, at fixed calibration quantiles, at matched persistent-alert rates and at matched alarm-episode frequencies. A preregistered test repeats the main comparisons on 2021–2022 data from the same farm. No fault labels are used.

## Data

The study uses 10-min SCADA data from 2015 and from April 2021 to April 2022 for the 16 turbines of the Bogdanci wind farm (North Macedonia), provided by AD Elektrani na Severna Makedonija (ESM). The data are confidential and are not part of this repository. They can be shared only with the prior written permission of ESM.

The 2015 scripts start from a processed table, `data/research_layers/health_twin_base_v1.csv.gz`, with its manifest `health_twin_base_v1_manifest.json`. The 2021–2022 scripts take a table with the same columns and file names through `--data`. The required columns are listed in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md). Signal names in the code are neutral (for example `gear_oil_temp_mean`); the operator's original signal tags are not published. The step that builds the processed tables from the operator's database export is specific to the site and is not included.

Folder names in this repository differ from those used during the analysis. The protocols are kept exactly as they were written and hashed, so they refer to the original folder names:

| Name in the protocols | Folder here |
|---|---|
| `Bogdanci_dataset/scripts/` | `scripts/` |
| `Korrigjimet_2026-09-18/` | `repairs_2026-09-18/` |
| `Placebo_2026-10-30/` | `posthoc_2026-10-30/placebo/` |
| `Chesterman_baza_2026-10-30/` | `posthoc_2026-10-30/fleet_median_baselines/` |
| `Chesterman_gain_2026-10-30/` | `posthoc_2026-10-30/gain_adjusted/` |
| `Zbulimi_F_2026-10-30/` | `posthoc_2026-10-30/detection/` |
| `Zbulimi_F2_2026-11-02/` | `posthoc_2026-11-02/matched_persistent_alert_rate/` |
| `Zbulimi_F3_2026-11-07/` | `posthoc_2026-11-07/matched_episode_frequency/` |
| `Konfirmimi_2021-2022/` | `confirmation_2021-2022/` |

The scripts here differ from the analysis copies only in these paths and in the neutral signal names. Their SHA-256 hashes therefore differ from the runner hashes recorded in the protocols and in the original `verification.json` files. The 2021–2022 confirmation protocol is registered on OSF (https://osf.io/vxqhk).

## Requirements

The analyses were run with Python 3.9.6, NumPy 2.0.2 and pandas 2.3.3; the figures use Matplotlib 3.9.4:

```bash
pip install -r requirements.txt
```

## Structure

| Folder | Content |
|---|---|
| `scripts/` | Normal-behaviour model, regime-distance reference, rolling-origin folds, sequential ablation, expanded ridge grid, alert experiments, synthetic tests and the data-quality audits |
| `repairs_2026-09-18/` | Calendar repair of the M2 bootstrap, matched-median sensitivity and common-state support revision, run under the protocol in that folder |
| `posthoc_2026-10-30/` | Permuted-weight placebo, fleet-median baselines, gain-adjusted comparison and detection at a fixed calibration quantile, each with the protocol written before it was run; `detection/diagnostics/` holds the threshold-tail and threshold-scale checks of Section 3.6 |
| `posthoc_2026-11-02/` | Detection at matched persistent-alert rates (2015 and 2021–2022), the extra checks of Section 3.6 and the Figure 7 script |
| `posthoc_2026-11-07/` | Detection at matched alarm-episode frequencies (2015 and 2021–2022) |
| `confirmation_2021-2022/` | Preregistered protocol, deviations log and the runners for hypotheses H1–H4 |
| `figures/` | Script for Figures 2–6 and A1–A3 |

The analysis scripts write to their own output folders and refuse to overwrite existing results; the scripts in `scripts/` accept `--overwrite` for an intentional rerun. Use `--help` on any script in `scripts/` to see its options.

## Run order

Run from the repository root.

1. Data-quality audits:

   ```bash
   python3 scripts/audit_health_meteorological_gate.py
   python3 scripts/audit_local_ambient_proxy.py
   python3 scripts/audit_health_twin_covariate_support.py
   ```

2. Rolling-origin folds, then the sequential ablation on common support (repeat with `--protocol local` and the matching folder names):

   ```bash
   python3 scripts/run_health_twin_rolling_ablation_v1.py --protocol independent --output-dir data/experiment_results/health_twin_rolling_ablation_independent_v1
   python3 scripts/run_health_twin_trim_distance_ablation.py --protocol independent --reference-dir data/experiment_results/health_twin_rolling_ablation_independent_v1 --output-dir data/experiment_results/health_twin_trim_distance_ablation_independent_v1
   python3 scripts/run_health_twin_alpha_boundary_sensitivity_v1.py --protocol independent --reference-dir data/experiment_results/health_twin_trim_distance_ablation_independent_v1 --output-dir data/experiment_results/health_twin_alpha_boundary_sensitivity_independent_v1
   ```

3. Nested alert experiment:

   ```bash
   python3 scripts/run_health_twin_alert_stability_v1.py --output-dir data/experiment_results/health_twin_alert_stability_v1
   python3 scripts/run_health_twin_all_peer_alert_stability_v2.py --output-dir data/experiment_results/health_twin_all_peer_alert_stability_v2
   ```

4. Repairs and support sensitivities of 18 September 2026:

   ```bash
   python3 repairs_2026-09-18/run_thermal_checks.py --part rolling
   python3 repairs_2026-09-18/run_thermal_checks.py --part common
   python3 repairs_2026-09-18/correct_all_m2_contrasts.py
   python3 repairs_2026-09-18/update_expanded_alpha.py
   ```

5. Synthetic tests on the states of step 3. The first two commands give the state-separation results with the original and the adopted (at least four eligible turbines) common-state definition; the last two give the sustained-alert sensitivity with the original states and with the adopted definition:

   ```bash
   V2=data/experiment_results/health_twin_all_peer_alert_stability_v2
   python3 scripts/run_health_twin_all_peer_state_separation_v2.py --scores $V2/test_alert_states.csv.gz --residual-frames $V2/test_residual_frames.csv.gz --source-provenance $V2/provenance.json --output-dir data/experiment_results/synthetic_state_original
   python3 scripts/run_health_twin_all_peer_state_separation_eligible_min4.py --scores $V2/test_alert_states.csv.gz --residual-frames $V2/test_residual_frames.csv.gz --source-provenance $V2/provenance.json --output-dir data/experiment_results/synthetic_state_adopted
   python3 scripts/run_health_twin_synthetic_alert_sensitivity.py --input-dir $V2 --output-dir data/experiment_results/health_twin_all_peer_synthetic_alert_sensitivity_v2 --artifact-name health_twin_all_peer_synthetic_alert_sensitivity_v2 --deterministic-provenance
   mkdir -p data/experiment_results/synthetic_alert_adopted_inputs
   cp $V2/calibration_alert_states.csv.gz $V2/test_alert_states.csv.gz $V2/provenance.json data/experiment_results/synthetic_alert_adopted_inputs/
   cp repairs_2026-09-18/common/common_eligible_min4_calibration.csv.gz data/experiment_results/synthetic_alert_adopted_inputs/calibration_common_alert_states.csv.gz
   cp repairs_2026-09-18/common/common_eligible_min4_test.csv.gz data/experiment_results/synthetic_alert_adopted_inputs/test_common_alert_states.csv.gz
   python3 scripts/run_health_twin_synthetic_alert_sensitivity.py --input-dir data/experiment_results/synthetic_alert_adopted_inputs --output-dir data/experiment_results/synthetic_alert_adopted --artifact-name health_twin_synthetic_alert_sensitivity_v0_7_eligible_min4 --deterministic-provenance
   ```

6. Post hoc analyses of 30 October 2026:

   ```bash
   python3 posthoc_2026-10-30/placebo/run_placebo.py
   python3 posthoc_2026-10-30/fleet_median_baselines/run_chesterman.py
   python3 posthoc_2026-10-30/gain_adjusted/run_gain.py
   python3 posthoc_2026-10-30/detection/run_detection.py
   ```

   They read the outputs of steps 2–4 and write to a `results/` folder next to each script. The two diagnostics write their tables next to themselves:

   ```bash
   python3 posthoc_2026-10-30/detection/diagnostics/diag_tails.py
   python3 posthoc_2026-10-30/detection/diagnostics/diag_thr_scale.py
   ```

7. Preregistered confirmation on 2021–2022 data. `<layer>` is the 2021–2022 table; the second command is the 2015 reproduction check of the runner:

   ```bash
   python3 confirmation_2021-2022/code/run_h1_h2.py --data <layer> --folds N --out confirmation_2021-2022/results/h1_h2
   python3 confirmation_2021-2022/code/run_h1_h2.py --data data/research_layers/health_twin_base_v1.csv.gz --folds L2015 --check data/experiment_results/health_twin_trim_distance_ablation_local_v1 --out confirmation_2021-2022/results/h1_h2_check_2015 --no-placebo
   python3 confirmation_2021-2022/code/run_h3_h4.py --data <layer> --out confirmation_2021-2022/results/h3_h4
   ```

8. Detection at matched persistent-alert rates (after steps 6 and 7), then the extra checks and Figure 7:

   ```bash
   python3 posthoc_2026-11-02/matched_persistent_alert_rate/run_f2_2015.py
   python3 posthoc_2026-11-02/matched_persistent_alert_rate/run_f2_2021_2022.py --data <layer>
   python3 posthoc_2026-11-02/matched_persistent_alert_rate/extra_checks.py
   python3 posthoc_2026-11-02/matched_persistent_alert_rate/make_figure7.py
   ```

9. Detection at matched alarm-episode frequencies (after step 8):

   ```bash
   python3 posthoc_2026-11-07/matched_episode_frequency/run_f3_2015.py
   python3 posthoc_2026-11-07/matched_episode_frequency/run_f3_2021_2022.py --data <layer>
   ```

   Steps 8 and 9 write to `results_2015/` and `results_2021_2022/` next to the scripts and check that they reproduce the earlier analyses before any contrast is computed.

10. Figures 2–6 and A1–A3. The script reads the outputs of steps 2–5; `--synthetic-root` is the folder `data/experiment_results/synthetic_alert_adopted` of step 5:

    ```bash
    python3 figures/make_figures.py --synthetic-root data/experiment_results/synthetic_alert_adopted --output-dir figures/output
    ```

## Reproduction check

On 9 November 2026 the full run order above was executed from a clean copy of this repository on the confidential data, with the signal columns renamed to the neutral names. Of the 122 result tables, 115 were identical to the analysis outputs used in the manuscript. The other seven differ only in floating-point rounding: relative differences below 3 × 10⁻⁶ in two grids of daily sums, renamed columns in one audit table, and the bootstrap share of negative draws for trim-stage contrasts whose true effect is zero, because trimming removed no peer in that fold. No number reported in the manuscript changed.

## Licence

MIT, see [LICENSE](LICENSE).
