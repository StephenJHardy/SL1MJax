# Catalogue-background test for holography

## Question and scope

Does unmodelled emission around 3C147 contribute materially to the holography
residuals? The reference dishes still receive those sources through their own
off-axis beams. Pointing one dish away from 3C147 does not move the correlator
phase centre or remove the rest of the sky.

The diagnostic compares two fixed predictions on identical rows:

1. The existing calibrated 3C147 prediction.
2. That prediction plus independently catalogued background sources.

It does not fit fluxes, recalibrate data, modify flags, select a production beam,
or test full-Jones leakage. It must not use sealed survey channels or SPWs.
No Bacchus job is needed to run its synthetic tests. Do not interrupt, stage into,
or read large parts of the active survey work MS while that survey is running.

## Implementation and current readiness

- `src/sl1mjax/holography_background.py`: chunked prediction and paired scoring.
- `scripts/diagnose_holography_background.py`: immutable snapshot/CSV input,
  checksums, dry run, report, prediction export, and plots.
- `tests/test_holography_background.py`: analytic and runner regression tests.
- `scripts/prepare_holography_background.py`: fixed NVSS query and a read-only
  SPW-4 snapshot exporter with a calibrator closure against the survey export.

The predictor, scorer and first real-data snapshot exporter are implemented.
The exporter currently targets the already calibrated scientific SPW-4/5 MS,
channel 32 of SPW 4. It is not a general new-MS calibration pipeline. Existing
HOLORASTER plotting exports are insufficient: they do not provide all the UVW
and per-source, per-antenna feed-frame geometry required here.

The September 2026 real-data run queries NVSS above 10 mJy at 1.4 GHz within
one degree of 3C147, excludes the central two arcminutes, and retains 60 sources.
It tests fixed spectral indices -1.0, -0.7 and 0.0, with no fitted source fluxes.
The 32 selected development times retain 3,088 moving–reference rows. These are
a bounded diagnostic sample, not all dwell integrations or an unopened holdout.
Casacore supplies per-antenna apparent AZELGEO source directions. A minimal SIN
pointing-frame rotation preserves the survey's central-source query exactly.
This is a recorded extension of the survey's tangent-frame contract, not an
independent validation of the mount's off-axis frame convention.

## Prediction

For each unpolarized unresolved source of fixed flux density S, compute

```
D = diag(Ep_R * conj(Eq_R), Ep_L * conj(Eq_L))
V_source = S * Rp @ D @ Rq^H * exp(+2πi ν/c [u*l + v*m + w*(n-1)])
```

`Ep` and `Eq` are the voltage responses at that source's direction relative to
each dish's actual pointing. `Rp` and `Rq` are the same fixed residual
direction-independent Jones factors used in the calibrator prediction, expressed
in the calibrated sky basis. The diagnostic retains RR and LL from this matrix.
It sums complex visibilities over sources, not magnitudes. The positive fringe
sign follows the repository's CASA/original-pq UVW convention.

The sky is not evaluated at the moving dish's commanded raster offset. Each
background source has a different source-in-beam direction on **both** dishes.
The reference response is not identity for a background source. The full
`w*(n-1)` term remains even when the calibrator itself is at phase centre.

The first runner requires `casa_parang_true` calibrated data. Diagonal beam terms
commute with the circular parallactic matrix for this unpolarized hypothesis.
Residual Jones must already be in that same sky basis. Do not feed a new survey
`parang=False` diagonal product into this contract, or add full polarization apply
twice. An explicitly recorded identity residual is allowed only when appropriate
for both the data and the baseline calibrator prediction.

## Prepare a small, read-only experiment after the survey

1. Use an already opened SPW-4 development channel, initially channel 32. Select
   several complete settled dwell/scan groups across the main beam, null region,
   and outer raster. Keep original antenna order, native integrations, and native
   channels. Do not select rows because their residuals look large.
2. Freeze an external catalogue around 3C147 before inspecting this test's
   residuals. Select sky coverage using both dishes' beam footprints, not only the
   central imaging region. Record survey, epoch, selection radius, completeness,
   morphology, spectral assumptions, and source URLs. Existing catalogue-query
   machinery can help, but the 3C391 source list is not a 3C147 catalogue.
3. Remove 3C147 and its catalogue components to avoid double counting MODEL_DATA.
   The runner enforces a declared exclusion radius around the phase centre. This
   first contract is for HOLORASTER phase-centred on 3C147, not C147-* fields.
4. Export original MS UVW in metres and the matched calibrated RR/LL samples,
   weights, flags, row IDs, and calibrator-only prediction. Record calibration
   table hashes and application state. Do not recalibrate on these raster rows.
5. For every source and row, compute its direction cosines in each antenna's
   actual feed frame. Use the established casacore apparent-direction and pointing
   transforms. Do not add catalogue celestial `(l,m)` directly to native AZELGEO
   offsets. Validate the transform at the calibrator direction against the locked
   `source_lm_feed` path, on both original-pq baseline orders.
6. Supply the fixed residual Jones factors from the calibrator prediction. Reject
   unsupported calibration rows before freezing the snapshot. Freeze the score
   mask and dwell/scan clusters. Record that these are development data.
7. First dry-run the snapshot locally. Then score it against an exact-frequency
   EVLA-C plane from a checksummed survey catalogue. Do not silently scale a
   different frequency plane or extrapolate outside its raster.

Catalogue integrated fluxes are extrapolated using their recorded spectral
indices. Missing indices require an explicit `--default-spectral-index`.
Nonzero catalogue major axes require `--allow-unresolved-approximation`.
Neither option establishes that a source is physically unresolved or its spectrum
is correct. Evaluate plausible spectral and size uncertainties before interpreting
a null result. Time/bandwidth smearing is not integrated in this first runner;
check its significance using native exposure/channel width and the selected
baselines before interpreting fast-fringing distant sources.

## Snapshot contract

The NPZ contains no object arrays. It holds one frequency and one phase centre.
Rows and source order must match the accompanying manifest and catalogue CSV.

| Array | Shape / meaning |
| --- | --- |
| `uvw_m` | `(row,3)`, original MS pq UVW in metres |
| `frequency_hz` | scalar, native channel frequency |
| `beam_p_lmn`, `beam_q_lmn` | `(row,source,3)`, unit source directions in the feed frames |
| `residual_p`, `residual_q` | `(row,2,2)`, fixed sky-basis residual Jones |
| `measured`, `calibrator` | complex `(row,2)`, RR/LL in Jy |
| `weight`, `flag` | `(row,2)`, weights and boolean flags |
| `score_mask` | boolean `(row,)`, frozen eligible partition |
| `cluster_id` | `(row,)`, dwell/scan/visit identifiers, not unique row IDs |
| `row_id` | unique `(row,)`, original row provenance |

The CSV uses `sl1mjax.catalog.RadioCatalogSource` / `write_radio_catalog`.
The JSON manifest requires:

```json
{
  "schema_version": 1,
  "snapshot_sha256": "<sha256 of NPZ>",
  "catalogue_sha256": "<sha256 of CSV>",
  "phase_centre_rad": [0.0, 0.0],
  "source_names": ["<in CSV and geometry order>"],
  "geometry_provenance": "<MS, POINTING, frame conversion, revision>",
  "calibration_provenance": "<product and table hashes>",
  "residual_jones_provenance": "<fixed factors and basis; explain identity if used>",
  "calibration_state": "casa_parang_true",
  "uvw_convention": "casa_positive_fringe_original_pq",
  "beam_coordinate_frame": "feed_source_direction_cosines",
  "split_provenance": "<opened development selection and frozen mask>",
  "cluster_unit": "dwell",
  "calibrator_exclusion_arcsec": 60.0
}
```

Replace the illustrative phase centre and exclusion radius with measured metadata
and a catalogue-appropriate choice. Checksums establish file identity, not the
scientific correctness of supplied coordinates or calibration.

## Commands

```bash
uv run scripts/diagnose_holography_background.py \
  --snapshot snapshot.npz --manifest snapshot.json \
  --catalogue background.csv --dry-run

uv run scripts/diagnose_holography_background.py \
  --snapshot snapshot.npz --manifest snapshot.json \
  --catalogue background.csv \
  --beam-root /path/to/survey/catalogue --beam-digest <catalogue-sha256> \
  --output outputs/holography_background_run1
```

The scoring output directory must not already exist. The dry run neither loads
beam planes nor writes outputs. No command reads an MS or connects to Bacchus.

## Scoring, products, and interpretation

`report.json` records each source's fixed flux and apparent contribution, matched
support counts, calibrator-only and augmented residual power, and their paired
difference. Each power is weighted squared complex residual divided by weighted
observed power on the **same** supported rows. This denominator is not intrinsic
3C147 flux and must not be quoted as the publication's `|ΔV|/I` metric.

Uncertainties resample reduced dwell/scan cluster sums with multiplicity. Rows
within a cluster are not independent bootstrap draws. These intervals are
diagnostic; they do not capture catalogue or calibration systematics.

Unknown beam support is not zero. If any included source is unsupported for a
hand, that row/hand is excluded from both scores and counted explicitly. Physical
beam nulls remain supported zeros. A favourable supported-only result is not a
claim about excluded rows or uncatalogued sky.

`predictions.npz` preserves row identities, summed background, and support.
`background_comparison.png` shows the source layout and real/imaginary background
prediction against the measured calibrator residual. Use separately declared
score masks for radial regions if needed; do not derive those masks from residual
amplitude or pool them into a post-hoc acceptance gate.

- A coherent, transferable reduction supports background contamination as one
  contributor. Confirm on other opened dwell groups and baseline strata.
- A prediction much smaller than the residual constrains this **catalogued,
  fixed-spectrum, unresolved-source hypothesis**, not all possible background sky.
- A worse prediction requires checking flux, source size, frame, support, and
  smearing assumptions. It does not justify refitting the calibration or declaring
  CASSBEAM wrong.

## Test gates

Run the local analytic suite before any real snapshot:

```bash
uv run pytest tests/test_holography_background.py -q
```

Tests check both beam factors, conjugation, nonzero w-term, baseline reversal,
complex source addition, chunk invariance, fixed residual-Jones matrix algebra,
known null versus unsupported hand, flags, paired scores, cluster accounting,
spectral assumptions, and dry-run checksum enforcement. Real use additionally
requires a geometry closure against the locked calibrator query and a zero-added-
background baseline closure. No synthetic test alone certifies the MS exporter.
