# Plain fleet-median baselines (Chesterman-style) — protocol

Written on 2026-10-30, before any computation. Post hoc comparison on the same 2015 site-year; not independent confirmation.

## Question

How does the distance-weighted reference compare with the plain fleet median used in the closest prior work (Chesterman et al. 2021, 2023), which takes the median over all turbines at each timestamp, includes the target turbine and applies no trimming?

## Design

1. Same frozen setup as the primary analysis and the placebo control: health layer (SHA-256 checked), folds M1–M4 and L1–L5, targets, eligibility rule, frozen ridge alpha and bandwidth, training-only standardisation. Nothing is reselected.
2. Two baselines on exactly the same scoring rows as the primary comparison (rows with at least three candidate and three retained peers):
   - **C1, fleet median of residuals:** specific state = residual of turbine i minus the median of the pooled-model residuals of all eligible turbines at that timestamp, turbine i included, no trimming.
   - **C2, fleet median of raw signals:** specific state = raw target signal of turbine i minus the median raw signal of all eligible turbines at that timestamp, turbine i included, no normal-behaviour model. This is the implicit normal-behaviour model of Chesterman et al.
3. For reference, the existing comparators are recomputed on the same rows: the untrimmed leave-one-out median (static_no_trim), the trimmed leave-one-out midpoint median (static_trim) and the distance reference. Their spreads must reproduce the saved values to 1e-9.

## Statistics

- Spread (population standard deviation) of each specific state, per fold and target.
- Percentage change of the distance spread relative to C1 and to C2.
- 95 % intervals from a seven-day calendar-block bootstrap on the full test-month calendar, 2000 paired draws, seeds 20261031 + 10000·fold_index + 100·target_index (+5 000 000 for the L folds), with the same sufficient-statistics procedure as the primary analysis.

## Pre-specified reading

For each primary channel, the distance reference improves on the plain fleet median if its spread is lower than C1 in all four M folds and the seven-day interval lies wholly below zero in at least three of the four. C2 is reported descriptively, because its spread is not adjusted for operating conditions. Including the target turbine in the median shrinks the specific state slightly; this works in favour of C1 and C2, and it is reported as such. All results are reported whatever their direction.
