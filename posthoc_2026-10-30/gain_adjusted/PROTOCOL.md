# Gain-adjusted comparison with the self-inclusive fleet median — protocol

Written on 2026-10-30, after the C1 result of the Chesterman-baseline comparison was known and before this analysis was run. It is a post hoc analysis that explains that result; it does not replace it. The C1 spread comparison is reported unchanged.

## Question

The self-inclusive fleet median (C1) had a slightly smaller spread than the distance reference. Because C1 includes the target turbine in its own reference, part of every deviation of that turbine moves the reference with it. How large is this absorption, and how do the methods compare once spread is expressed per unit of retained signal?

## Design

1. Same frozen setup, folds, targets, scoring rows and specific states as the Chesterman-baseline run (C1, C2, static_no_trim, static_trim, distance).
2. For each target turbine and scoring row, add a shift of δ = ±1, ±2 and ±3 of that turbine's training residual scale to its own residual (for C2, to its raw signal) at that timestamp only, and recompute its specific state. The gain is the change in the specific state divided by δ. Leave-one-out references (static_no_trim, static_trim, distance) do not contain the target turbine, so their gain is exactly 1; this is checked numerically.
3. For each method and |δ|, take the mean gain over rows and signs, and define the adjusted spread as spread / mean gain (noise per unit of retained signal).
4. Compare the distance reference with C1 through the ratio of adjusted spreads, with a 95 % interval from the same seven-day calendar-block bootstrap of the spread ratio (2000 draws, seeds 20261101 + 10000·fold_index + 100·target_index, +5 000 000 for L), holding the full-sample gain fixed.

## Pre-specified reading

The leave-one-out design is supported against C1 for a primary channel if the distance reference has a smaller adjusted spread than C1 at |δ| = 1 and 2 training scales in at least three of the four M folds. Results for |δ| = 3, for C2, for the L folds and for the exploratory and control channels are reported descriptively. All results are reported whatever their direction.
