# Real-data catalogue-background diagnostic — 9 September 2026

## Result

The catalogued background predicts a millijansky-scale contribution, not the
roughly half-Jansky residual floor in the scored sample. There is a small paired
LL improvement. It does not explain the dominant holography mismatch.

This is a bounded SPW-4 development diagnostic, not a complete sky inventory or
a null result for the entire raster. It neither selects a beam nor changes the
survey's calibration or flags.

## What ran

- Existing scientific SPW-4/5 MS, HOLORASTER, native SPW 4 channel 32:
  **4.564 GHz**, one-second exposures. CORRECTED_DATA was read once, not reapplied.
- 32 evenly spaced times among the survey's opened development rows. All their
  available selected moving–reference baselines were retained: **3,088 rows**.
- NVSS query: within 1 degree of 3C147, at least 10 mJy at 1.4 GHz, excluding
  the central 120 arcsec. **60 background sources**, 2.9711 Jy summed intrinsic
  flux at the catalogue frequency. The catalogue was frozen before scoring.
  Source: [NVSS through CDS/VizieR](https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/65).
- Fixed spectral indices **−1.0, −0.7, and 0.0**. No fitted source amplitudes,
  spectral indices, gain terms, coordinates, or flags.
- Both antenna beams, complex conjugation, original-pq UVW, and the full
  geometric phase including w. Identity residual Jones matches the survey.
- Per-antenna/time casacore J2000→AZELGEO transforms, followed by a spherical
  minimal-SIN pointing-frame rotation. The central-source prediction agrees
  with the frozen survey export to **1.3942×10⁻⁹ Jy maximum absolute difference**.
  This locks continuity with that baseline; it is not an independent proof of
  the mount's off-axis geometry.

The beam catalog digest was
`828c9ece95d3bc6c95bed8091840a6bceea3214e59b42895530b23224973d843`.
Snapshot SHA-256:
`4bbbcf42248d4fd3bd030c53c3f34c43a7d222b2f80663fbcd9615e47e4e4ef9`.
Catalogue SHA-256:
`730945936c8df5226272debb71a9f3dc169db73f5eadd9dacd21eb77864a9fb1`.
The complete isolated source snapshot has its own `source_revision.json`.

## Support is an important limitation

All-source beam support exists for **1,240 rows in 13 scan clusters** in each
hand. Those rows span mover offsets about **4.88–19.55 arcmin**. The full selected
sample extends to 72.49 arcmin.

The other **1,848 rows** were excluded from both arms because one or more
background sources fall outside the generated beam raster for a dish. The
sources themselves may be another degree away from the calibrator. Covering the
holography raster does not guarantee coverage of every source relative to every
moving pointing. These contributions were not treated as zero.

Consequently, this result does **not** clear background contamination in the
far-out raster or sources outside the catalogue selection.

## Scores

All values below use the same supported rows and original weights. RMS here is
an absolute complex-visibility RMS in Jy, not the publication residual-power
fraction and not a beam-voltage error.

| Fixed spectral index | Background RR RMS | Background LL RMS | Residual RR RMS | Residual LL RMS |
| --- | ---: | ---: | ---: | ---: |
| −1.0 | 0.532 mJy | 0.683 mJy | 480.86 mJy | 476.55 mJy |
| −0.7 | 0.758 mJy | 0.973 mJy | 480.86 mJy | 476.55 mJy |
| 0.0 | 1.733 mJy | 2.225 mJy | 480.86 mJy | 476.55 mJy |

For −0.7, the largest summed background sample is 2.47 mJy in RR and 2.88 mJy
in LL. Even the flat-spectrum scenario reaches only 5.64/6.58 mJy in those
supported rows. Source uncertainty remains, but these predictions are far below
the observed residuals under all three declared hypotheses.

The paired loss difference is `(calibrator + catalogue) − calibrator`, divided
by the same observed visibility power:

| Hand, α=−0.7 | Calibrator-only power | Plus catalogue | Paired difference | Scan-bootstrap 95% interval |
| --- | ---: | ---: | ---: | --- |
| RR | 0.032162868 | 0.032160803 | −2.065×10⁻⁶ | [−9.936×10⁻⁶, +3.226×10⁻⁷] |
| LL | 0.034283273 | 0.034279458 | −3.815×10⁻⁶ | [−1.170×10⁻⁵, −4.004×10⁻⁷] |

RR's interval crosses zero. LL's small improvement amounts to about **0.011% of
the original residual power**. The three spectral assumptions are not three
independent detections. Catalogue, geometry and beam uncertainties are not
included in these conditional bootstrap intervals.

## Interpretation and remaining limits

This argues against these catalogued sources being the main cause of the large
copolar residuals on the supported rows. It is consistent with a small real
background contribution. It does not establish the cause of the remaining
residuals, nor does it validate the suspect low/upper-band channels.

Sources were treated as unresolved and unpolarized using integrated catalogue
fluxes. Source size, variability, spectral curvature, diffuse emission, fainter
sources and survey-epoch differences remain unmodelled. Native exposure and
channel averaging were not integrated; the computed central-frequency signal
should not be interpreted as an exact prediction for strongly smeared sources.
As a scale check, a rectangular 2 MHz channel with a constant beam gives
`|sinc(Δν × geometric_delay)|` of at least 0.917 over the selected row/source
pairs, with median 0.9995. This approximation does not supply the actual channel
response, but bandwidth smearing at that scale cannot account for the hundreds-
fold gap between predicted background RMS and the measured residual RMS.
The outer beam and the extension of the pointing-frame convention also remain
uncertain. No scientific full-Jones claim follows from this test.

## Products and reproduction

Local products: `outputs/holography_background_20260909/`:

- `catalogue/`: pinned CSV, raw VizieR response and selection record.
- `snapshot/`: immutable geometry/data NPZ and manifest.
- `alpha_m10/`, `alpha_m07/`, `alpha_0/`: JSON scores, summed predictions and plots.
- `summary.json`: RMS, support, paired scores and radius summaries.
- `source_revision.json`: hashes of the staged code used on Bacchus.

Bacchus products:
`/media/stephen/astro/vla/extracted/commissioning/validation/scientific/background_source_test_20260909/`.
The catalogue and initial scripts were staged separately in
`/tmp/sl1mjax-background-20260909/`; the catalogue is also preserved locally.

Run `scripts/diagnose_holography_background.py` against the frozen snapshot and
CSV to regenerate a scenario in a **new** output directory. Run
`scripts/summarize_holography_background.py` to regenerate the compact summary.
See `docs/holography-background-source-test.md` for the complete input contract.
Survey products, source MS, calibration tables, and production factories were
not modified. The background jobs have finished.
