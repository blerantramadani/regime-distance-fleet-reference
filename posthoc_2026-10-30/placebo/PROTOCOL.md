# Permuted-weight placebo control — protocol

Written on 2026-10-30, before any placebo computation. Post hoc control on the same 2015 site-year; not independent confirmation.

## Question

Does the spread reduction of the distance-weighted reference come from matching the regime weights to the right peers, or would any weighting with the same weight distribution give a similar reduction?

## Design

1. Reuse the frozen setup of the 2026-09-18 correction run: the same health layer (SHA-256 checked against its manifest), folds (M1–M4 primary, L1–L5 sensitivity), targets, eligibility rule, training-only standardisation, frozen ridge alpha and selected bandwidth per fold and target, contemporaneous trim and three-peer minimum. Nothing is reselected.
2. For every target row (turbine i, timestamp t), take the Gaussian weights of the retained peers and permute them at random among those same retained peers. The multiset of weights, the effective peer count, the retained peers and the lower weighted-median rule are unchanged; only the assignment of weights to peers is randomised.
3. Draw P = 100 permutations per fold and target, with fixed seeds 20261030 + 10000·fold_index + 100·target_index + p (p = 0…99), separately for the M and L protocols (L seeds offset by 5 000 000).
4. Before computing the placebo, reproduce the saved spreads: the static midpoint, static lower and distance standard deviations must equal the 2026-09-18 values to 1e-9.

## Statistics per fold and target

- Distance effect against the static midpoint and static lower references (reproduced values).
- Placebo effect against the static lower reference: median and 2.5–97.5 % range over the 100 permutations.
- Regime-information effect: percentage change of the distance spread relative to the median placebo spread.
- Permutation p-value: (1 + number of permutations with placebo spread ≤ distance spread) / (P + 1).

## Pre-specified reading

For each primary channel (gear-oil, generator-bearing-G, generator-bearing-R), regime information is supported if the distance spread is below the median placebo spread in all four M folds and the permutation p-value is ≤ 0.05 in at least three of the four. The exploratory main-bearing channel, the hydraulic control and the L folds are reported in the same way but do not enter the decision. All results are reported whatever their direction.
