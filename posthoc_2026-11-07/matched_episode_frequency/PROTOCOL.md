# F3: detection at matched alarm-episode frequency (protocol)

**Status.** Written on 7 November 2026, before any F3 runner was written or run. This analysis is post hoc. F2 (`../Zbulimi_F2_2026-11-02/`) matched references on the fraction of scored test rows carrying a persistent flag. That fraction measures flagged time, not how often an operator is alerted, and the manuscript says so in four places. F3 matches on the number of alarm episodes instead.

What we saw before writing this protocol: F2 already saved the episode rate for every reference and threshold (`episodes_per_100_turbine_days` in `f2_curves.csv`). On 7 November we interpolated those saved full-sample curves, without any bootstrap, to see what an episode-matched comparison might show. For gear oil in 2015, at equal flagged time the distance reference had about 30–40 % more episodes than the unweighted reference. At equal episode rates, its detection point estimates were about 7–8 percentage points higher for gear oil and 6–7 points lower for generator bearing G at 5 and 10 episodes per 100 turbine-days. Generator bearing R was within about 1 point. These were only point estimates. The matching levels in Section 3 were chosen after this look, and that fact is reported with the results.

## 1. Question

At equal frequencies of persistent-alarm episodes in the test period, does the distance reference detect injected shifts better or worse than the equally trimmed unweighted leave-one-out reference (static_trim) and the fleet median of residuals (C1)?

## 2. What stays exactly as in F2

Everything that produces the curves: data, eligibility, chronology, normal models and alphas, bandwidths, reference states, October centres, the threshold grid Q = {0.950, …, 0.999}, the persistence rule, the episode definition (a maximal run of persistent-flag rows on contiguous rows, at most 20 min apart), the scenario set (all scored windows, background flags included), the detection probability D and the excess detection E = D − B. Turbine-days are scored test rows divided by 144.

F3 only changes the matching variable. No state, threshold or scenario is recomputed differently from F2.

## 3. What is new

**Matching variable.** For each reference and q, the episode rate is the number of persistent-alarm episodes in November–December without injection, summed over turbines, per 100 scored turbine-days. For the 16-turbine farm, 10 episodes per 100 turbine-days is about 1.6 episodes per day across the farm.

**Matching levels.** e ∈ {5, 10, 20, 40} episodes per 100 turbine-days, a doubling sequence that covers roughly the episode rates of the unweighted reference at the F2 flagged-row fractions of 0.1–1 %. As stated above, the levels were fixed after the point-estimate look.

**Matched comparison.** For each reference, D (primary) and E (sensitivity) are interpolated linearly against log10(episode rate) between adjacent grid points, at each e. The interpolation function of F2 is used unchanged: zero rates are dropped, and a level outside a reference's observed range gives "not estimable", with no extrapolation. Contrasts are distance − static_trim and distance − C1, in percentage points. The three primary channels are the decision set. Main bearing and hydraulic oil are reported descriptively.

**Descriptive companions.** At each matched e, the runner also reports the interpolated flagged-row fraction and the mean episode length (flagged rows divided by episodes, in 10-min rows) for each reference. These show whether a reference reaches a given episode rate with shorter or longer alarms.

**Inference.** Same two-way bootstrap as F2: 2000 draws, turbines and anchors resampled independently with replacement. In each draw the episode rate comes from the resampled turbines' full-period totals, D and E come from the resampled turbine × anchor cells, and the contrasts are recomputed after interpolation. 95 % percentile intervals. An interval is reported only if at least 95 % of draws give a finite contrast; otherwise the contrast is "not estimable", even if a full-sample point estimate exists. Seeds: 20261107 + 100 × target index (gear oil, generator bearing G, generator bearing R, main bearing, hydraulic oil).

**Reading.** Better if the contrast is positive and its interval lies wholly above zero. Worse if negative and wholly below zero. Otherwise no clear difference. Every level, channel and comparator is reported, whatever the direction. No multiple-comparison adjustment.

## 4. Reproduction checks

1. The runner must reproduce F's q = 0.995 detection rates and evaluable counts exactly, as F2 does. It inherits that assertion.
2. For every target, reference and q, the full-sample flagged-row fraction, episode rate, D and E must equal the values in the corresponding F2 `f2_curves.csv` to within 1e-9.

If either check fails, F3 stops and the difference is investigated before any F3 contrast is read.

## 5. Part B: 2021–2022

The same specification is applied unchanged to the 2021–2022 data through the F2 Part B runner, for both the fixed-threshold rule and the EWMA rule (λ = 0.1). Seeds: as in Section 3, plus 50000 for the fixed rule and 60000 for EWMA. Part B is post hoc and descriptive. It does not change the preregistered H3/H4 decisions. Reproduction check 2 uses `../Zbulimi_F2_2026-11-02/results_2021_2022/f2_curves.csv`.

## 6. Interpretation limits and use in the manuscript, fixed now

Like F2, episode matching uses the test period to place the operating points. It evaluates the trade-off and is not a deployable calibration rule. The shifts are synthetic residual shifts, not failures. Episode frequency and flagged time are both partial descriptions of alarm burden, and neither is a false-positive rate without event labels.

The results go into the manuscript whatever their direction: as columns or a companion block to Table 9 for 2015, as one or two sentences in Sections 3.6 and 4.5 for 2021–2022, and in the abstract and conclusions in place of the sentence saying that equal flagged time need not mean equal episode counts. If episode matching changes the reading for any primary channel compared with F2, the text says so.

## 7. Outputs

The runners are new files in this folder. They load the F2 runners' functions and change only the matching axis. Each part writes to its own new results folder (`results_2015/`, `results_2021_2022/`): matched contrasts with intervals and finite-draw counts, the descriptive companions, and `verification.json` with input, runner, protocol and F2-runner hashes and both reproduction checks. The SHA-256 of this protocol is recorded in `PROTOCOL.sha256` before any runner is written. Nothing existing is overwritten.
