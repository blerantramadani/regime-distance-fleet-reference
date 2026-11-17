# Confirmation of the regime-distance fleet reference on Bogdanci 2021–2022 — protocol

**Status:** fixed on 30 October 2026 by B. Ramadani, before any outcome analysis of the 2021–2022 data, and registered on OSF under embargo. Co-authors: V. Fustić and G. Dzambazovski.

**What was looked at before writing it:** only the structural inventory in `../Inventari_2021-2022_2026-10-30/INVENTORY.md`, which covers tables, columns, station identifiers, coverage, signal availability and frozen signals. No residuals, spreads, alert rates or detection rates of the 2021–2022 data have been computed.

## 1. Purpose

To test, on data that played no part in developing the method, the four main findings of the 2015 analysis:

1. distance weighting lowers the spread of the turbine-specific residual against an equally trimmed unweighted leave-one-out reference;
2. this gain depends on matching the weights to the operating regime (permuted-weight placebo);
3. the lower spread does not give better detection under a fixed high-quantile alert rule;
4. a rule designed for small sustained shifts (EWMA) can use the lower spread. This is new: the 2015 paper only suggested it.

## 2. Data

Bogdanci SCADA backup from ESM: 70 monthly archives, April 2021 to 19 May 2022, checked against `SHA256SUMS.txt`. The 16 turbine stations and the meteorological station are the same as in 2015, and so are the signal names. Turbine labels T01–T16 follow the same station order as in 2015.

Known data limits, fixed now:

- The independent meteorological temperature is stuck at −56.05 from 2021-09-06, permanently from 2021-09-28 23:00. It cannot serve any test month.
- 31 May 2021 is missing for all turbines. T16 has no data in July and August 2021 and only partial data in June and September 2021.
- May 2022 is incomplete and is not used.

## 3. Consequence for the design

The confirmation uses the turbine-local ambient-temperature proxy (`turbine_local_proxy` in the code) as the ambient covariate, as in the L folds of the 2015 paper, where all three primary channels improved in 5/5 folds. The independent meteorological signals are not used at all. This choice follows from the sensor failure alone and is made before any outcome is seen.

## 4. Frozen method

Everything is identical to the 2015 code listed with hashes in Section 9. Nothing is tuned on the test months.

- Eligibility: active power 0–2350 kW, primary wind speed 4–20 m/s, generator speed at least 100 rpm, power reference at least 2200 kW, all features and descriptors finite.
- Normal-behaviour model: pooled ridge with the same 14 operating and seasonal terms (ambient = local proxy) and 15 turbine indicators; initial fit at α = 10 and training trim at 4.5 robust scales; α chosen from {10⁻², …, 10⁵} on the validation month by RMSE.
- Reference: leave-one-turbine-out over all eligible peers; seven standardised descriptors; Gaussian kernel with bandwidth chosen from {0.25, 0.5, 1, 2, 4} on the validation month by the 2015 rule; contemporaneous trim (4.5 × max(1.4826·MAD, median peer training scale, 0.25)); at least three peers; lower weighted median. The comparator is the equally trimmed unweighted midpoint median on the same rows.
- Common state: median of eligible finite residuals, at least four turbines.

## 5. Folds

Eight folds N1–N8, with training expanding from 2021-04-01, one validation month and the next month as the test month:

| Fold | Training | Validation | Test |
|---|---|---|---|
| N1 | Apr–Jul 2021 | Aug 2021 | Sep 2021 |
| N2 | Apr–Aug 2021 | Sep 2021 | Oct 2021 |
| N3 | Apr–Sep 2021 | Oct 2021 | Nov 2021 |
| N4 | Apr–Oct 2021 | Nov 2021 | Dec 2021 |
| N5 | Apr–Nov 2021 | Dec 2021 | Jan 2022 |
| N6 | Apr–Dec 2021 | Jan 2022 | Feb 2022 |
| N7 | Apr 2021–Jan 2022 | Feb 2022 | Mar 2022 |
| N8 | Apr 2021–Feb 2022 | Mar 2022 | Apr 2022 |

The code changes allowed for 2021–2022 are listed in Section 9.

## 6. Hypotheses and decision rules

Target roles as in 2015: gear oil, generator bearing G and generator bearing R are primary; main bearing is exploratory; hydraulic oil is the negative control.

**H1, spread (primary).** The percentage change in the standard deviation of the specific state, distance against unweighted, per fold. Intervals come from a seven-day calendar-block bootstrap with 2000 draws. H1 holds for a primary channel if the spread is lower in at least 7 of 8 folds **and** the 95% interval lies wholly below zero in at least 6 of 8. The 2015 result is **confirmed** if H1 holds for at least two of the three primary channels.

**H2, placebo.** One hundred permutations of the same weights among the retained peers, per fold. H2 holds for a primary channel if the distance spread is below every placebo spread (p ≈ 0.01) in at least 6 of 8 folds.

**H3, detection with the 2015 rule.** Chronology: training April–September 2021, tuning October, calibration November, test December 2021 to April 2022. Calibration as in 2015: centre and 99.5th-percentile threshold per turbine and target from the calibration month, and a persistent flag after three consecutive exceedances. Shifts: step and drift; 1, 6 and 24 h; ±1, ±2 and ±3 training residual scales; anchors every second day from 2021-12-02 to 2022-04-28. Evaluable windows as in 2015. Comparison of the distance and unweighted references, with paired differences and a cluster bootstrap over turbines (2000 draws). The 2015 finding (no detection gain) is **replicated** if, for every primary channel, the interval of distance − unweighted does not lie wholly above zero.

**H4, detection with an EWMA rule (new).** The same chronology and shifts, but the alert statistic is the exponentially weighted moving average z_t = 0.1·s_t + 0.9·z_{t−1} of the specific state, restarted after gaps longer than 20 min. Its centre and 99.5th-percentile threshold come from the calibration month, and the persistence rule is unchanged. z is computed over the whole test series, and after a shift is added to s_t inside a window, z is recomputed. H4 holds for a primary channel if distance − unweighted is positive with its interval wholly above zero. It is **supported** if it holds for at least two of the three primary channels.

**Descriptive, with no decision rule:** fleet-median baselines C1 and C2 with the gain adjustment; exploratory and control channels; alert rates; the common state; 99th and 97.5th-percentile thresholds for H3 and H4.

## 7. Reporting

All results are reported whatever their direction, including failures of H1–H4. The intervals are not adjusted for multiple comparisons. Any deviation from this protocol is written down with its date and reason; deviations made after results are seen are labelled as such.

## 8. Seeds

Spread bootstrap: 20270101 + 10000·fold_index + 100·target_index. Placebo permutations: 20270201 + 10000·fold_index + 100·target_index + permutation. Detection bootstrap (H3 and H4): 20270301 + 100·target_index + 10000·variant_index. Indices start at zero in the order of the fold table, the target list (gear oil, generator bearing G, generator bearing R, main bearing, hydraulic oil) and the threshold variants (99.5th, 99th, 97.5th percentile).

## 9. Code and data at fixing

The method is the code of the 2015 analysis, listed with SHA-256 hashes (paths relative to `04_Analiza_dhe_kodi/`):

| File | SHA-256 |
|---|---|
| `Bogdanci_dataset/scripts/audit_scada.py` | `f692549234d914f8efac19d17354fcdc0d08c36424f12921eb2a62b2098d1aa2` |
| `Bogdanci_dataset/scripts/materialize_health_twin_base.py` | `7952679c787eeb4b9eb73dca5b12d43906ccfa682ea8e4f1ec3761716409886e` |
| `Bogdanci_dataset/scripts/run_health_twin_baseline.py` | `bbc57d641439270ef546bdc96109f8981fc13547b65d56df8c4b141d17cd2daf` |
| `Bogdanci_dataset/scripts/run_health_twin_regime_peer_v2.py` | `d57b84f3acefcde9cf68d31928be63158117fc53bc3195c4192f8920c749def3` |
| `Bogdanci_dataset/scripts/analyze_regime_peer_block_bootstrap.py` | `e63b8e76458a70b608485028c2ea00a73d94466c0180b757e8ec78e9ac207d08` |
| `Bogdanci_dataset/scripts/run_health_twin_rolling_ablation_v1.py` | `8a427f751e88668ec99e6c2fe15a954e9b8646d0601029893e045d680bba64b0` |
| `Bogdanci_dataset/scripts/run_health_twin_trim_distance_ablation.py` | `694bcace8e59f619e0363e7235a9f78095f63114e3cb864549a07e4fdb009c28` |
| `Bogdanci_dataset/scripts/run_health_twin_alert_stability_v1.py` | `9748ee96d15f8ffb70ebde6381335b06cd44b41356625d9c92b7e42877a33dcf` |
| `Bogdanci_dataset/scripts/run_health_twin_all_peer_alert_stability_v2.py` | `4b8add8f2d34d292885815af17e67cd936eb2488d0d11e255625a34377fd2ea4` |
| `Placebo_2026-10-30/run_placebo.py` | `b2f7223fecd7b3a65e4f2626af8749e32f771d0bb884ad74cfe30e277c2c5759` |
| `Chesterman_baza_2026-10-30/run_chesterman.py` | `770cfbc343d17fb5fa39e05a62c4f58b64d56fc2bb721b5d08051ee82cd9722f` |
| `Chesterman_gain_2026-10-30/run_gain.py` | `73d71286313024edadc8277b0d8bd75626e69ad7381635bb3b3a3241ae3c5808` |
| `Zbulimi_F_2026-10-30/run_detection.py` | `16ee4765834dff37fdf57b3b3defcbc09cc33355ac806deccca4258f2c5d0c78` |
| `Inventari_2021-2022_2026-10-30/inventory.py` | `fddcc1ae9305568686f3a898ec34e0f47b2fcf1ff85198b9da79bfa8a3eb7a58` |

The 2021–2022 archives are identified by `SHA256SUMS.txt` (SHA-256 of that file: `ed88b2dacc20540f3cbe4657201ac7daff07009c0f46644c0008e6ba78dcc609`).

The only code changes allowed are: (a) the date window and archive pattern of the extraction (2015 → 1 April 2021 to 30 April 2022); (b) the fold table N1–N8 and the H3/H4 chronology; (c) the EWMA statistic for H4; (d) the seeds above. Every new or changed script is hashed before it is run, and its differences from the files above are recorded.

## 10. Order of work

1. Registration of this protocol on OSF under embargo; the registration date and identifier are recorded in this folder.
2. Extraction of the 2021–2022 processed table.
3. H1–H4 and the descriptive analyses, in that order.
