# Detection at matched alert rates (step F) — protocol

Written on 2026-10-30, before any computation. Post hoc analysis on the same 2015 site-year, specified after the placebo, C1 and gain results were known. It is not independent confirmation.

## Question

Does the lower spread of the distance reference lead to better detection of synthetic turbine-specific shifts when every reference is calibrated to the same nominal alert rate?

## Frozen setup

1. Same chronology as the nested alert experiment: training 2015-05-14 12:00 to 2015-08-31, tuning in September, calibration in October, test in November–December. The eligibility rule, the normal model (alpha selected on tuning) and the bandwidth (selected on tuning) are those of `health_twin_all_peer_alert_stability_v2`. Nothing is reselected.
2. References, all scored on the same rows (at least three candidate and three retained peers):
   - `distance_trim`, the proposed reference;
   - `static_trim`, the equally trimmed unweighted leave-one-out midpoint median and the primary comparator of the paper;
   - C1, the fleet median of residuals with the target included and no trimming.
   - `static_no_trim` and C2 (raw-signal fleet median) are reported descriptively.
3. Reproduction check: the `distance_trim` calibration states and specific thresholds must match the saved outputs of `health_twin_all_peer_alert_stability_v2` to 1e-9.

## Alert rule

As in the paper. For each turbine-target pair, the centre is the October median of the specific state and the threshold is the empirical 99.5th percentile of the absolute deviation from that centre. A point flag is raised when the deviation exceeds the threshold, and a persistent flag after three consecutive point exceedances with no gap longer than 20 min. Each reference has its own centres and thresholds, so all references have the same nominal calibration point rate of 0.5%.

## Injection

1. Anchors: every second day at 00:00 from 2015-11-02 to 2015-12-30 (30 anchors), so windows of up to 24 h never overlap. The four anchors of the paper (8 and 22 November, 6 and 20 December) are part of this grid and are also reported separately. The window starts at the first scored row within one hour after the anchor, as in the paper.
2. Shapes step and linear drift; durations 1, 6 and 24 h; final signed amplitudes ±1, ±2 and ±3.
3. The amplitude is expressed in units of the target turbine's training residual scale from the normal model. This scale is the same for every reference, so every reference receives the same physical shift. (The paper used reference-specific October robust scales, which would give a noisier reference a larger shift.)
4. The shift is added to the target turbine's residual (for C2, its raw temperature) on the rows inside the window, and the specific states are recomputed. For leave-one-out references, the target's own reference does not change. For C1 and C2, the fleet median is recomputed with the shifted value. One turbine is shifted per scenario.
5. A window is evaluable if it has at least three scored intervals and no background persistent flag under any of the three primary references. The evaluable set is therefore the same for all references and the comparisons are paired.
6. A scenario counts as detected if at least one persistent flag occurs inside the window.

## Endpoints

- **Primary:** pooled detection rate over all evaluable scenarios (shapes, durations, amplitudes, polarities, anchors and turbines) for each primary channel, and the paired differences distance − static_trim and distance − C1, in percentage points.
- **Intervals:** 95% cluster bootstrap over turbines (16 clusters, 2000 draws, seed 20261102 + 100·target_index).
- **Secondary:** detection by amplitude and by duration; background point and persistent rates in November–December without injection for each reference, to show whether the matched calibration rate holds in the test period.
- **Sensitivities:** thresholds at the 99th and 97.5th percentiles; and thresholds set per turbine-target so that each reference has a 0.5% background point rate in November–December. The last one uses test data and cannot be deployed; it only separates calibration drift from discrimination.

## Pre-specified reading

For each primary channel, the distance reference detects better than a comparator if the pooled paired difference is positive and its 95% interval lies wholly above zero. Main bearing and hydraulic oil are reported descriptively. All results are reported whatever their direction.

## Amendment (2026-10-30, before any detection result was computed)

The first run stopped at the reproduction check, before any detection result. Refitting the normal model gives residuals that differ from the saved alert run by up to 2.8e-7 for gear oil and 1.5e-8 for hydraulic oil, the two targets with ridge alpha = 0.01, where the ridge system is ill-conditioned. For the other three targets the difference is below 3e-11. The check on the distance states and thresholds is therefore applied at an absolute tolerance of 1e-6 instead of 1e-9. In addition, the run reports how many calibration and test point flags of the distance reference differ from the saved alert run. Nothing else changes.
