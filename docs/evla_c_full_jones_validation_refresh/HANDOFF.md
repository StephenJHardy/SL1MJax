# EVLA-C validation refresh — handoff table

Terminal status against `docs/evla-c-full-jones-validation-refresh-plan.md`.
This is not a production acceptance document.

## Status table

| Item | State | Evidence |
| --- | --- | --- |
| Calibration application | pass (existing goldens; not a HOLORASTER/beam test) | `tests/test_casa_golden.py`; CASA/JAX fixtures unchanged |
| Coordinate / Jones controls | pass | independent NumPy RIME; omitted conjugation/parallactic/fringe/sign tests fail; same-index identity max \|Δ\| = 4.8e-7 |
| Numerical beam convergence | limited | p64 vs p32 at 4500/4564/4626 MHz; inner 3′ median \|ΔE\| ≈ 7e-6; p95 ≈ 0.033 |
| Diagonal fit | scored | only ch16 and ch32 pass the 1% main-lobe cut on both hands |
| Cross-hand sensitivity / evidence | inconclusive_sensitivity | unit injection not recovered; 3× template is; 0/5 movers; all nine channels same status |
| Frequency coverage | complete | 9/9 publication channel reports |
| Offset ring | complete (historical) | ch32 inner-holdout RR/LL 0.009/0.006; ch0/63 empty hands |
| Publication | complete | v2 bundle, executed notebook, 24 figures, claims C01–C09 |
| Production acceptance | refused | `production_accepted` stays false |

## Artifact paths and hashes

- Refresh product: `/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1`
- Nine-frequency planes: `/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_ninepub_20260908`
- Isolated package: `/tmp/sl1mjax-evla-c-refresh` (`PYTHONPATH`)
- Isolated scripts: `/tmp/sl1mjax-evla-c-refresh-scripts`
- Publication v1 (frozen): `src/sl1mjax/data/vla_c_band_beam_validation_v1` checksum `2536a9f53e632b35f97ea94b16872b6f6c44afd34d9823226c4819f621ecab98`
- Publication v2: `src/sl1mjax/data/vla_c_band_beam_validation_v2` checksum `fa01cc15f715e75ace7088ed1bc16a7fac75cfafd2b5b68fda071986f024d5d9`
- Executed notebook: `notebooks/vla_c_band_beam_validation.ipynb`
- Rendered summary: `docs/vla_c_band_beam_validation.md`
- Scientific MS: `/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms`
- Residual Jones SHA: `ff6aee560ab1521da1472ce833393ad01d3f64bbc1200f6e71b4b5088aef12bc`
- Git HEAD: `ef9b795d60b1de65408dc082ff179a0e9541ebb3` (dirty tree; not committed)
- Staged source snapshot (21 files): `88e62ba2fa7d46b0884d599ba512d503c98254f33c7d6fd0edf3a687be4fc8ad`
  (`docs/evla_c_full_jones_validation_refresh/experiment_manifest.json`)

## Three questions

1. Implementation: independent algebra matches production helpers at 1e-12; omitted conjugation, parallactic rotation, fringe sign, and coordinate sign fail their tests. CASA/JAX goldens still pass. Channel-32 identity vs coordinate-feed v2 is 4.8e-7.
2. Diagonal: EVLA-C at `source_lm_feed` is the leading SPW-4 development model in the main lobe at 4.564 GHz (RR/LL residual power 0.62% / 0.56%). Only channels 16 and 32 pass the 1% cut on both hands. Mid beam is qualified; outer raster is diagnostic.
3. Full Jones: these observations cannot distinguish the unit prediction from the diagonal (`inconclusive_sensitivity`) on every publication channel and on both spatial and mover partitions.

## Figure coverage

| ID | Treatment |
| --- | --- |
| F01–F04 | Reused after provenance. Observation / occupancy / antennas / CASA apply. Not beam evidence. |
| F05, F08, F10, F12, F15 | Replaced with EVLA-C / source predictions. |
| F06, F07, F09, F11 | Never published as recovered-voltage archives. Covered by F10, F15, F20–F22, F24. |
| F13 | Publication 20%-of-peak all-data estimator. Independent vs common mask is F29. |
| F14 | Not a new recovered-voltage archive. Paired holdouts are F18 / F26 / Phase 4. |
| F16 | Nine channels; main-lobe diagonal + full-Jones RL/LR residual; 1% cut; n in sidecar. |
| F17 | Phase 5 C147 ring, not the generic historical ring. |
| F18 | EVLA-C full vs diagonal; outcome `inconclusive_sensitivity`. |
| F19 | Sealed / not_run. |
| F20–F25 | Recomputed from EVLA-C complex predictions. |
| F26–F29 | Labelled generic/commanded controls plus matched EVLA-C comparisons. |

## Phase 4 by frequency and spatial support

All nine channels: spatial and mover pooled 95% CIs lie above zero; 0/5 holdout movers improve; copolar non-regression passes. Status on each support is `inconclusive_sensitivity`. See `phase4_frequency_status.json`. Injection recovery was scored on channel 32 and did not recover the unit model.

## Commands

```bash
# Bacchus scoring (already complete; do not restart if healthy)
PYTHONPATH=/tmp/sl1mjax-evla-c-refresh JAX_PLATFORMS=cpu \
  /home/stephen/checkouts/SL1MJax/.venv/bin/python -u \
  /tmp/sl1mjax-evla-c-refresh-scripts/run_thol0001_evla_c_validation_refresh.py \
  --stage frequency --repo-root /tmp/sl1mjax-evla-c-refresh \
  --output-dir /media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1 \
  --beam-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_ninepub_20260908

# Publication (does not overwrite v1)
REFRESH=/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_full_jones_validation_refresh_v1 \
  ./scripts/build_vla_c_band_beam_validation_v2.sh

# Local rebuild from staged refresh copies
uv run python scripts/build_vla_c_band_beam_validation_bundle.py \
  --refresh-dir docs/evla_c_full_jones_validation_refresh/stage/refresh \
  --existing-bundle src/sl1mjax/data/vla_c_band_beam_validation_v1 \
  --coordinate-feed-dir docs/evla_c_full_jones_validation_refresh/stage/coordinate_feed_v2 \
  --output-dir src/sl1mjax/data/vla_c_band_beam_validation_v2
uv run python scripts/write_vla_c_band_beam_validation_notebook.py
uv run python scripts/execute_vla_c_band_beam_validation_notebook.py --timeout 900
uv run python scripts/render_vla_c_band_beam_validation.py \
  --bundle src/sl1mjax/data/vla_c_band_beam_validation_v2 \
  --asset-dir docs/assets/vla_c_band_beam_validation \
  --output docs/vla_c_band_beam_validation.md
```

## Tests

Focused publication/refresh plus broader relevant holography, RIME, CASA,
CASSBEAM, and Jones tests: **251 passed**. Added regressions for C147
`l_rad`/`partition` plot keys and claim input hashes. Ruff and
`git diff --check` clean on the publication path.

Publication channel-32 main-lobe RR residual power matches the science
report exactly (`0.006160125312689845`).

## Defect / supersession list

- Generic commanded HOLORASTER comparison: historical.
- Historical generic full-Jones non-detection: not an EVLA-C test.
- Width 1.04 / empirical squint orientation: not inherited.
- SPW 5: sealed.
- F17 initially failed on C147 `l_rad` and float partition scores; plot now
  accepts both and ignores non-id partition keys.
- Phase 4 bootstrap and C147 MS load defects were fixed before scoring
  completed; see the refresh scripts, not the frozen v1 products.

## What the evidence supports

- EVLA-C at `source_lm_feed` is the leading SPW-4 diagonal development model
  in the main lobe at 4.564 GHz, and independently on the historical C147
  ring at that frequency.
- These observations do not recover the unit full-Jones prediction. The
  formal outcome is `inconclusive_sensitivity`, not a production beam.

## Next bounded step

A residual-Jones frequency axis. Do not retune the current gates.
