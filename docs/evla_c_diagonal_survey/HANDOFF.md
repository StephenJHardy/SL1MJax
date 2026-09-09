# EVLA-C diagonal survey — running handoff

Work against `docs/evla-c-band-diagonal-survey-and-imaging-handoff-plan.md`.
This is not a production acceptance. Full Jones is not a gate.

## Done

- Phase 0 manifest and calibration policy (no residual Jones).
- Both holography executions identified. 18 January is **upper-C**
  (5.988–8.010 GHz) and readable despite the inner `ms.tgz` CRC error.
- 32 Pass-A EVLA-C g1024/p32 planes, 4052–7948 MHz, plus Pass-B planes
  for SPWs 4 and 5, in
  `/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1`.
- Scientific-MS Pass A and Pass B on SPWs 4 and 5 (identity residual Jones).
  Table: `docs/evla_c_diagonal_survey/suitability_table.json`.
  Plot: `docs/evla_c_diagonal_survey/pass_b_mainlobe.png`.
- Opt-in catalog helper `open_survey_catalog` / `require_survey_frequency`
  (exact frequency only). Not wired into the production factory.
- Named imaging selector `stokes_i_beam("evla_c_diagonal_survey_v1")`
  with catalog root + digest. JAX operator and seven-pointing
  reconstruction/product writer are covered by unit tests.
- Frequency × radius suitability plot from all 44 scored reports.
- Phase 5 spatial/dB/signed-cut maps and `phase5_diagnostics.json` from
  44 channel exports. First-null positions are sampling-limited
  intervals. 3C391 native frequencies are 4536–4662 MHz at 2 MHz.
- Compact v3 pack at `src/sl1mjax/data/vla_c_band_beam_validation_v3`
  (does not overwrite v2; sha
  `8cd815eb08e99a986ee0413e80d681641ebbceea87cd35f0d4859b48f3ed5e61`).
  Notes: `docs/evla_c_diagonal_survey/PUBLICATION.md`. Executed
  notebooks: `notebooks/vla_c_band_beam_validation.ipynb` and
  `notebooks/vla_c_band_beam_validation_v3.ipynb`. Survey page:
  `docs/evla_c_diagonal_survey/SURVEY.md`.

## Isolated Bacchus commands

```bash
./scripts/stage_evla_c_diagonal_survey_bacchus.sh
PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu \
  /home/stephen/checkouts/SL1MJax/.venv/bin/python -u \
  /tmp/sl1mjax-evla-c-survey-scripts/run_thol0001_evla_c_diagonal_survey.py \
  --stage score --spw 5 --channel 32 \
  --scripts-dir /tmp/sl1mjax-evla-c-survey-scripts \
  --beam-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 \
  --measurement-set /media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms
```

Do not rsync onto `/home/stephen/checkouts/SL1MJax/src/sl1mjax`.

## Current evidence

1. Pass A is complete. Pass B is limited to lower-C SPWs 4–6. The other
   116 declared extras are `pass_a_only`, not scored. Residual Jones is
   not applied. `scientifically-qualified` means the empirical 1%
   main-lobe cut only; band-wide numerical qualification is incomplete.
2. 3C391 imaging uses the fixed 4536/4598/4662 MHz nodes, not a
   continuous 4436–5820 MHz acceptance. SPW-4 4548/4580 MHz already miss
   in one hand. Catalog digest
   `828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843`.
   Native seven-pointing smoke (32 rows × 3 frequencies, loss 0.0380) is
   an engineering check, not mosaic validation.
3. Uncertain or rejected: 4052–4308 MHz, 5308 MHz mild miss, 5948 MHz,
   all upper-C Pass A, Pass-B edges, mid/outer lobes, interpolation, and
   first-null alignment. The published 10′ radial dip is a coarse
   minimum, not a first null. Large SPW-4 ch24 LL / ch40 RR excursions
   remain unexplained; do not retune the beam from them.
4. Upper-C used 3C138 and 3C147 scans 3,53. Every upper-C scientific
   claim is archive-integrity unverified (inner `ms.tgz` CRC error).
   Do not overwrite the archive. A long optimized mosaic is subsequent.

## Frozen paths

Do not overwrite v1/v2 bundles, hashed CASSBEAM archives, the scientific
SPW 4+5 MS, or the production full-Jones factory.
