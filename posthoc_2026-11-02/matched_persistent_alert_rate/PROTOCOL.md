# F2: detection at matched persistent-alarm rates — protocol

**Status.** Written on 2 November 2026, before any F2 output was computed. This analysis is post hoc. It was prompted by an external review comment that thresholds matched on the October point-exceedance rate (0.5 %) do not give matched rates of persistent alarms, because a persistent alarm needs three consecutive exceedances. Before writing this protocol, we looked at the existing outputs of analysis F (`../Zbulimi_F_2026-10-30/results/background_rates.csv`, `detection_decision.csv`) and of the 2021–2022 runner (`../Konfirmimi_2021-2022/results/h3_h4/`) at the three thresholds already computed there (99.5, 99 and 97.5 %). That look suggested that the distance reference has fewer persistent background alarms and that, for gear oil, it may detect more at equal alarm rates. F2 tests this properly.

## 1. Question

At equal rates of persistent alarms in the test period, does the distance reference detect injected shifts better or worse than the equally trimmed unweighted leave-one-out reference (static_trim) and the fleet median of residuals (C1)?

## 2. What stays exactly as in F

Data, eligibility, chronology (training to 31 Aug 2015, tuning September, calibration October, test November–December), normal models and selected alphas, bandwidths, the five reference states, the October centre (median of the specific state per turbine and reference), the persistence rule (three consecutive point exceedances on rows no more than 20 min apart), and the synthetic scenarios: anchors every second day from 2 November to 30 December 2015, durations 1, 6 and 24 h, step and linear drift, final amplitudes ±1, ±2, ±3 training residual scales of the target turbine, added to that turbine's residual. For C1 the fleet median is recomputed with the shifted value. A window is scored if its first scored interval lies within one hour of the anchor and it has at least three scored intervals.

## 3. What is new

**Threshold grid.** For each turbine and reference, thresholds are the October quantiles q of |state − centre| for q ∈ Q = {0.950, 0.960, 0.970, 0.975, 0.980, 0.985, 0.990, 0.9925, 0.995, 0.996, 0.997, 0.998, 0.999}. Calibration remains October-only.

**Alarm rate (x-axis, primary).** For each reference and q, the persistent-alarm rate is the share of scored test rows in November–December, pooled over turbines, that carry a persistent flag without any injection. Secondary, reported descriptively: alarm episodes per 100 turbine-days, where an episode is a maximal run of persistent-flag rows on contiguous rows and turbine-days are scored test rows divided by 144. The October persistent-alarm rate is also reported descriptively.

**Scenario set.** All scored windows count, for every reference and threshold. Unlike F, windows with a background alarm are not excluded, so the scenario set is identical for all references and thresholds.

**Detection probability (primary).** D(ref, q) is the share of scored windows with a persistent flag inside the window after injection. **Excess detection (sensitivity).** E(ref, q) = D(ref, q) − B(ref, q), where B is the share of the same windows with a persistent flag without injection.

**Matched comparison.** For each reference, D is interpolated linearly against log10(alarm rate) between adjacent grid points, at the alarm rates a ∈ {0.1, 0.2, 0.5, 1.0} %. Points with a zero alarm rate are not used, and there is no extrapolation: if a lies outside a reference's range, that contrast is reported as not estimable. Contrasts: distance − static_trim and distance − C1, in percentage points, for the three primary channels. Main bearing and hydraulic oil are reported descriptively.

**Inference.** A two-way bootstrap with 2000 draws resamples turbines and anchors independently with replacement. In each draw, the alarm-rate curves are recomputed from the resampled turbines and the detection curves from the resampled turbine × anchor cells, and the interpolated contrasts are recomputed. 95 % percentile intervals. Seeds: 20261103 + 100 × target index (targets in the order gear oil, generator bearing G, generator bearing R, main bearing, hydraulic oil).

**Reading.** At a given a, the distance reference is called better if the contrast is positive with its interval wholly above zero, worse if negative with its interval wholly below zero, and not clearly different otherwise. Results are reported for every a, channel and comparator, whatever their direction. No adjustment for multiple comparisons.

## 4. Reproduction check

At q = 0.995, applying F's evaluability rule (no background persistent flag under the distance, static_trim or C1 reference in the window), the pooled detection rates must equal those in `../Zbulimi_F_2026-10-30/results/detection_decision.csv` to within 1e-9, with the same numbers of evaluable scenarios. If not, F2 stops and the difference is investigated before any F2 result is read.

## 5. Interpretation limits, fixed now

Matching on the test-period alarm rate is an evaluation of the detection/alarm trade-off of each reference. It is not a deployable calibration rule, because it uses test data to place the operating points. The shifts are synthetic residual shifts, not failures.

## 6. Part B: 2021–2022

After Part A, the same specification is applied without change to the 2021–2022 data, using the chronology, models and scenarios of `../Konfirmimi_2021-2022/code/run_h3_h4.py`, for both the fixed-threshold rule and the EWMA rule (λ = 0.1). Seeds: as above plus 50000 (fixed rule) and 60000 (EWMA rule). Part B is post hoc and descriptive; it does not alter the preregistered H3/H4 decisions. Its reproduction check uses `../Konfirmimi_2021-2022/results/h3_h4/h3_h4_detection.csv` at q = 0.995.

## 7. Outputs

Each part writes to its own new results folder: curves per target, reference and q (alarm rates, episodes, October alarm rate, D, B, E), the matched contrasts with intervals, and `verification.json` with input, runner and protocol hashes and the reproduction check. Nothing existing is overwritten.
