# Deviations and clarifications

Recorded in date order. None of the entries below was made after any outcome of the 2021-2022 data had been seen.

## 2026-10-30, before extraction

1. **Operating-signal layer.** The processed table is built in two steps. `materialize_power_forecasting_base.py` builds the operating and meteorological layer, and `materialize_health_twin_base.py` adds the temperature and pressure signals. The first script was missing from the hash table in Section 9 of PROTOCOL.md. It is used unchanged, with SHA-256 `3bb48debe58ce890303702653934535f585947f7e1b0e1f7dfa2acf13aef358a`.
2. **How the date window is changed.** No original script is edited. A new wrapper, `code/extract_2021_2022.py`, imports the original modules and changes only four things at run time: the archive folders, the archive name pattern (`2021-*`/`2022-*` instead of `2015-*`), the nominal month used to order archives (year and month, so that boundary timestamps keep the earlier archive as in 2015), and the window (2021-04-01 00:00 inclusive to 2022-05-01 00:00 exclusive).
3. Wrapper hash before its first run: `code/extract_2021_2022.py` SHA-256 `798d509aee210fcd1c750e12ac0c2569d99c45a94e92667af37f22cc75520641`.

## 2026-10-30, before any 2021-2022 outcome

4. **Runner check on 2015 data.** `code/run_h1_h2.py` was run on the 2015 local-proxy folds L1–L5. For all 25 fold–target pairs it reproduced the saved alpha, bandwidth and scoring rows exactly, and the saved spreads to within 2.4e-9. The method in the new runner is therefore the 2015 method.
5. **Runner hashes before their first run on 2021–2022 data:** `code/run_h1_h2.py` `455a329f7097cf1edbaa639b4310202b6f2d18f2a3aad6eb7bee7a7456b8f71a`; `code/run_h3_h4.py` `970bcae758a54a48d33cea01af15326835335078b262edf130aa055b99fda099`.
6. **EWMA start.** The protocol does not say how the EWMA starts. It starts at the current value after a gap longer than 20 min and at the first scored row. Inside an injection window it continues from the value just before the window when that row is contiguous.
