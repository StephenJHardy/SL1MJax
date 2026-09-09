# EVLA-C validation refresh — live status

This is a working ledger for
`docs/evla-c-full-jones-validation-refresh-plan.md`. It is not a publication
claim. A usable production full-Jones beam is not required.

## Locked experiment choices

- Product:
  `/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1`
- New EVLA-C planes:
  `/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_ninepub_20260908`
- Isolated Bacchus tree: `/tmp/sl1mjax-evla-c-refresh`
- Publication channels (MS `CHAN_FREQ` SPW 4):
  0/4500, 8/4516, 16/4532, 24/4548, 32/4564, 40/4580, 48/4596, 56/4612, 63/4626 MHz
- Residual Jones: field-9 stabilized plane, **native channel 32 only**.
  Applied to the other eight channels with
  `explicit_channel32_field9_plane_applied_to_all_publication_channels`.
  This is not silent extrapolation.
- Primary arms: `evla_c_source_diagonal`, `evla_c_source_full_jones`
- Query: `source_lm_feed`. No width/squint inheritance. SPW 5 sealed.

## Completed

- Phases 0–5 on Bacchus, with local ledger copies.
- Phase 6 v2 publication bundle, executed notebook, rendered figures, claims.
- Phase 7 focused + broader relevant tests, lint, whitespace, visual audit.

## Channel development evidence (not unseen validation)

Train main-lobe diagonal residual power and correlation:

| Channel | MHz | RR power | LL power | RR corr | LL corr |
| --- | --- | --- | --- | --- | --- |
| 0 | 4500 | 0.01606 | 0.01518 | 0.9923 | 0.9927 |
| 8 | 4516 | 0.00955 | 0.01034 | 0.9956 | 0.9952 |
| 16 | 4532 | 0.00709 | 0.00799 | 0.9967 | 0.9963 |
| 24 | 4548 | 0.00580 | 0.01344 | 0.9973 | 0.9935 |
| 32 | 4564 | 0.00616 | 0.00560 | 0.9972 | 0.9974 |
| 40 | 4580 | 0.01364 | 0.00630 | 0.9935 | 0.9972 |
| 48 | 4596 | 0.01124 | 0.00773 | 0.9950 | 0.9968 |
| 56 | 4612 | 0.01195 | 0.00990 | 0.9946 | 0.9958 |
| 63 | 4626 | 0.01816 | 0.01374 | 0.9915 | 0.9937 |

Only channels 16 and 32 are at or below the 1% accepted main-lobe cut on
both hands. Band edges are worse. Channel 24 is RR-good / LL-poor;
channel 40 is the reverse.

Full Jones on all nine channels: copolar non-regression passes; RL
improves slightly; LR worsens; pooled 95% intervals lie above zero on
both spatial and mover partitions; 0/5 holdout movers improve.

## Phase 4 classification

Formal outcome: **`inconclusive_sensitivity`**.

- Algebra: `diag + (full−diag)` recovers full.
- Zero injection is a null.
- Unit-model injection (A=1, φ=0) lowers pooled Δ from +6.8e-4 to +3.5e-4
  but the 95% interval remains above zero.
- A=3, φ=0 is recovered (Δ = −3.0e-4, CI below zero). Other phases are not.
- `production_accepted` is false. Do not retune the gates.

## Phase 5 C147-* ring (historical partitions)

Eight fields, 123,552 rows, actual source-to-pointing geometry, no HOLORASTER
shortcut. Channel-32 inner-holdout diagonal residual power RR/LL =
0.00908 / 0.00551. Channels 0 and 63 have empty scored hands.

## Phase 6 publication

- Bundle: `src/sl1mjax/data/vla_c_band_beam_validation_v2`
- Checksum: `fa01cc15f715e75ace7088ed1bc16a7fac75cfafd2b5b68fda071986f024d5d9`
- Notebook: `notebooks/vla_c_band_beam_validation.ipynb` (16/16 code cells)
- Summary: `docs/vla_c_band_beam_validation.md`
- v1 is not overwritten (checksum `2536a9f53e632b35f97ea94b16872b6f6c44afd34d9823226c4819f621ecab98`).
- Staged source snapshot: `88e62ba2fa7d46b0884d599ba512d503c98254f33c7d6fd0edf3a687be4fc8ad`.

## Blocked / limited

- Residual Jones has no per-channel axis. Full-Jones frequency interpretation
  is limited by that channel-32 plane.
- p32 vs p64 tails remain numerically limited (p95 ≈ 0.03) inside 3′.
- SPW-4 holdouts are development evidence, not unseen validation.
- SPW 5 stays sealed.
