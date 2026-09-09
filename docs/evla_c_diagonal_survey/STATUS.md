# EVLA-C diagonal survey — live status

Ledger for `docs/evla-c-band-diagonal-survey-and-imaging-handoff-plan.md`.
This is not a production acceptance. Full Jones is not a gate.

## Locked choices

- Product: `evla_c_diagonal_survey_v1`
- Publication bundle: `vla_c_band_beam_validation_v3` (does not overwrite v2)
- Query: `source_lm_feed`. Unit EVLA-C. Frequency-dependent taper.
- Residual Jones: **not applied**
- SPW 5: declared diagonal frequency-transfer test
- Pass A: `floor(n*32/64)` (channel 32 when n=64)
- Pass B: channels `floor(n*k/64)` for k in 8, 24, 32, 40, 56
- Taper formula recorded as documented on 3.9–8.1 GHz only

## Phase 1 — identity (complete)

The 18 January 104 GiB destination is readable despite the inner `ms.tgz`
CRC error. Do not overwrite it. Casacore opened `SPECTRAL_WINDOW` and MAIN.

| Execution | Band | Science coverage | MAIN | Calibrated columns |
| --- | --- | --- | --- | --- |
| sb31628704 / eb31629959 | lower-C | 3.988–6.010 GHz, SPW 0–15 | 57,258,630 | DATA, FLAG only |
| sb31635131 / eb31644651 | **upper-C** | 5.988–8.010 GHz, SPW 0–15 | 51,448,878 | DATA, FLAG only |
| scientific SPW 4+5 subset | lower-C | same SPW table; rows only in 4 and 5 | 7,127,406 | DATA, CORRECTED_DATA, MODEL_DATA, FLAG |

Both holography executions have 16 science 64-channel SPWs plus auxiliary
SPW 16–17 at 4.832–5.086 GHz. Those auxiliaries must not classify the
18 January MS as mixed-band. Upper-C EVPA/check field is `J0521+1638`,
not the lower-C `J1331+3030`. Raster spacing is not copied from SPW 4.

Scientific MS row counts: SPW 4 and 5 each 3,563,703; all other SPWs 0.
New SPWs need new work copies and new solves. Do not apply channel-32
residual Jones.

Pass-A centres are 4052, 4180, …, 5948 MHz (lower-C) and 6052, …, 7948 MHz
(upper-C). See `frequency_table.json`.

## Phase 2 — SPW 6 calibrated

Policy is in `calibration_policy.json`. SPW 4/5 scoring used the existing
scientific MS. The writable SPW 6 work copy is at
`evla_c_diagonal_survey_v1/work_ms/lower_c_spw06.work.ms` (3,563,703 rows;
DATA, MODEL_DATA, CORRECTED_DATA). CASA 6.7 wrote tables under
`evla_c_diagonal_survey_v1/cal/spw06`, not `products/scientific`. The
scientific SPW 4+5 MS was not written. Remaining lower-C SPWs still need
their own work copies.

## Phase 3 — planes (Pass A complete)

32 EVLA-C g1024/p32 planes from 4052–7948 MHz are in
`/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1`
(`cassbeam_evla_c_diagonal_survey_v1`). Example 4692 MHz: 263169 rows,
feed_x = −0.94300, feedtaper = 12.2595 dB. Frozen archives were not written.

## Phase 4 — Pass A on the scientific MS (complete)

Both centres used **identity residual Jones** (not the refresh channel-32
plane). 538,489 development samples each.

| SPW | Frequency | Main-lobe RR / LL | Mid RR / LL | Status |
| --- | --- | --- | --- | --- |
| 4 | 4564 MHz | 0.00598 / 0.00538 | 0.0603 / 0.0570 | scientifically-qualified |
| 5 | 4692 MHz | 0.00589 / 0.00596 | 0.0520 / 0.0557 | scientifically-qualified |

SPW 5 is a declared transfer test. It meets the same 1% main-lobe cut as
SPW 4 under the survey method. This is not a production acceptance.

Pass B on SPWs 4 and 5 is complete (identity residual Jones):

| SPW | ch | MHz | RR | LL | Both ≤ 1% |
| --- | --- | --- | --- | --- | --- |
| 4 | 8 | 4516 | 0.00959 | 0.00961 | yes |
| 4 | 24 | 4548 | 0.00585 | 0.01397 | no (LL) |
| 4 | 32 | 4564 | 0.00598 | 0.00538 | yes |
| 4 | 40 | 4580 | 0.01641 | 0.00618 | no (RR) |
| 4 | 56 | 4612 | 0.01141 | 0.01032 | no |
| 5 | 8 | 4644 | 0.01751 | 0.00880 | no (RR) |
| 5 | 24 | 4676 | 0.00596 | 0.00651 | yes |
| 5 | 32 | 4692 | 0.00589 | 0.00596 | yes |
| 5 | 40 | 4708 | 0.00646 | 0.00615 | yes |
| 5 | 56 | 4740 | 0.01048 | 0.00875 | no (RR) |

The SPW-4 channel-24 LL and channel-40 RR excursions survive without
residual Jones. SPW 5 is cleaner at its three central Pass-B channels.
See `suitability_table.json`, `slot_status.json`, `pass_b_mainlobe.png`, and
`frequency_radius_suitability.png`. Pass A is complete; Pass B is
limited to lower-C SPWs 4–6. 44 of 160 declared science slots are
scored (16 empirical 1% main-lobe acceptances, 28 complete-with-
limitation). The other 116 are `pass_a_only`. That label is not
numerical qualification. Upper-C scores are archive-integrity
unverified.

| SPW | ch | MHz | RR / LL | Both ≤ 1% |
| --- | --- | --- | --- | --- |
| 6 | 8 | 4772 | 0.01047 / 0.00910 | no (RR) |
| 6 | 24 | 4804 | 0.00625 / 0.00607 | yes |
| 6 | 32 | 4820 | 0.00554 / 0.00584 | yes |
| 6 | 40 | 4836 | 0.00611 / 0.00614 | yes |
| 6 | 56 | 4868 | 0.01022 / 0.00974 | no (RR) |
| 0 | 32 | 4052 | 0.0696 / 0.1025 | no |
| 1 | 32 | 4180 | 0.2515 / 0.2344 | no |
| 2 | 32 | 4308 | 0.9466 / 0.9403 | no |
| 3 | 32 | 4436 | 0.00535 / 0.00567 | yes |
| 7 | 32 | 4948 | 0.00649 / 0.00583 | yes |
| 8 | 32 | 5052 | 0.00596 / 0.00646 | yes |
| 9 | 32 | 5180 | 0.00620 / 0.00636 | yes |
| 10 | 32 | 5308 | 0.01455 / 0.01473 | no |
| 11 | 32 | 5436 | 0.00678 / 0.00683 | yes |
| 12 | 32 | 5564 | 0.00684 / 0.00684 | yes |
| 13 | 32 | 5692 | 0.00649 / 0.00645 | yes |
| 14 | 32 | 5820 | 0.00699 / 0.00686 | yes |
| 15 | 32 | 5948 | 0.3001 / 0.2295 | no |

Lower-C Pass A is complete for science SPWs 0–15. Qualified centres
span 4436–5180 MHz and 5436–5820 MHz. SPW 10 at 5308 MHz is a mild
1.5% miss. SPW 15 at 5948 MHz fails like the low-C edge (30/23%).
SPW 0–2 remain completed limitations. The low-frequency cliff is still
4308 vs 4436 MHz.

Upper-C Pass A is complete (3C138 check, 3C147 flux 3,53). No centre
met the 1% cut. Hard failures: 6180–6436 MHz. Remaining centres are
1.1–2.1% misses. Remaining Pass-B channels outside SPWs 4–6 are
`pass_a_only`.

| Exec | SPW | MHz | RR / LL | Class |
| --- | --- | --- | --- | --- |
| upper-C | 0 | 6052 | 0.01395 / 0.01399 | mild miss |
| upper-C | 1 | 6180 | 0.95390 / 0.98017 | hard fail |
| upper-C | 2 | 6308 | 0.92769 / 0.95100 | hard fail |
| upper-C | 3 | 6436 | 0.94638 / 0.97193 | hard fail |
| upper-C | 4 | 6564 | 0.01171 / 0.01113 | mild miss |
| upper-C | 5 | 6692 | 0.01224 / 0.01154 | mild miss |
| upper-C | 6 | 6820 | 0.01204 / 0.01168 | mild miss |
| upper-C | 7 | 6948 | 0.01678 / 0.01615 | mild miss |
| upper-C | 8 | 7052 | 0.01415 / 0.01379 | mild miss |
| upper-C | 9 | 7180 | 0.01279 / 0.01283 | mild miss |
| upper-C | 10 | 7308 | 0.01463 / 0.01341 | mild miss |
| upper-C | 11 | 7436 | 0.01637 / 0.01400 | mild miss |
| upper-C | 12 | 7564 | 0.02054 / 0.01449 | mild miss |
| upper-C | 13 | 7692 | 0.01390 / 0.01392 | mild miss |
| upper-C | 14 | 7820 | 0.01413 / 0.01414 | mild miss |
| upper-C | 15 | 7948 | 0.01463 / 0.01472 | mild miss |

## Phase 5 — maps from scored exports (44 slots)

`docs/evla_c_diagonal_survey/phase5/` has per-channel measured/EVLA-C/residual
maps, dB/I displays, and `phase5_diagnostics.json` (radial residual power,
20%-of-peak squint, sampling-limited first-null intervals). The
frequency×radius map now uses 10′ radial bins, not only main/mid/outer
strata. Those 10′ dips are coarse radial minima, not first-null
locations. Model-cut first minima are interpolation diagnostics;
measured first nulls remain unresolved at HOLORASTER sampling.
A shared 8′/20′ radius remask of the 44 exports is in
`fixed_geometry_table.json`. It does not replace the historical scores.
Nine slots flip: 4308 MHz becomes an 8′-core pass, and several
higher-frequency historical 1% passes fail once the mask is no longer
allowed to shrink with measured amplitude. Matched-row and
scan/antenna influence still need a provenance re-export.
Bright-source offsets from the 3C391 catalog are recorded when a supported
cell exists.

## Phase 7 — v3 pack started (v2 preserved)

`src/sl1mjax/data/vla_c_band_beam_validation_v3` is a laptop-sized survey
bundle: suitability, inventory, phase-5 diagnostics, supersession ledger,
imaging handoff, and representative figures including the rejected SPW-2
4308 MHz maps. It does not overwrite v2 (checksum `fa01cc15…` unchanged).
v2 sources are snapshotted in `docs/vla_c_band_beam_validation_v2_snapshot/`.
The live notebook and `docs/vla_c_band_beam_validation.md` now render this
v3 pack. Parallel copies remain at
`notebooks/vla_c_band_beam_validation_v3.ipynb` and
`docs/evla_c_diagonal_survey/SURVEY.md`. Bundle sha after both Pass-A
surveys and Pass-B narrowing:
`8cd815eb08e99a986ee0413e80d681641ebbceea87cd35f0d4859b48f3ed5e61`.
Imaging nodes 4536/4598/4662 exist. Catalog digest after those nodes and
the SPW-6 Pass-B extras:
`828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843`.
Native smoke train/reload loss is 0.0380.

Focused plus relevant RIME/holography/beam/JAX/reconstruction/
publication tests: `126 passed` (`tests/test_evla_c_*.py`,
`test_voltage_reconstruction.py`, `test_beam_validation_bundle.py`,
`test_holography_holoraster_coordinates.py`,
`test_explicit_jax_adjoint.py`, `test_holography_forward_closure.py`).
Compiled-CASA gates were not run on this laptop (no CASA kernel here);
those remain an external skip, not a missing diagonal score.

## Phase 6 — opt-in adapter (native smoke complete)

`stokes_i_beam("evla_c_diagonal_survey_v1", survey_catalog_root=...,
survey_catalog_digest=...)` is the named selector. It does not load the
packaged 33×33 `diagonal_copolar` tables and does not promote full Jones.
Exact native frequency only; nearest-plane and Airy fallback are refused.
The JAX predict/adjoint path materialises survey rasters, and a
seven-pointing reconstruction plus product writer is covered by
`tests/test_evla_c_survey_beam.py`. Native 3C391 SPW0 is 64 channels at 2 MHz from 4536–4662 MHz. Exact
imaging nodes are 4536, 4598, and 4662 MHz (4598 is the lower neighbour
of the 4599 MHz arithmetic midpoint, which is not a native channel).
Existing Pass-B planes at 4548, 4564, 4580, 4612, and 4644 MHz are also
exact native matches. Interpolation remains refused.

```bash
PYTHONPATH=/tmp/sl1mjax-evla-c-survey \
  /home/stephen/checkouts/SL1MJax/.venv/bin/python -u \
  /tmp/sl1mjax-evla-c-survey-scripts/run_3c391_survey_smoke.py \
  --native-root <3c391_native_root> \
  --survey-catalog-root /media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 \
  --output /media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1/3c391_survey_smoke
```

## Two paths (9 September 2026)

Survey closeout is **not** a prerequisite for returning to imaging.

| Path | Order | Blocks imaging? |
| --- | --- | --- |
| Imaging | 4.536–4.662 GHz convergence → 64 native planes → smoke (adjoint/grads/memory/unsupported) → joint seven-pointing shared-sky mosaic | The relevant-frequency convergence and native-plane verification are the imaging gates |
| Survey | Re-export 44 slots with provenance; matched-row/scan diagnostics; read-only upper-C CRC; later low/mid/high convergence | No. Upper-C integrity blocks only trustworthy upper-C qualification |

The mosaic is an **experimental diagonal-beam imaging run**. Full Jones stays off. Fold 4 stays sealed. Topology starts fixed. Bright out-of-field components stay. Equal iteration budgets are not equally converged beam comparisons. Adding the remaining native planes will change the catalog digest; imaging must use the new digest.
