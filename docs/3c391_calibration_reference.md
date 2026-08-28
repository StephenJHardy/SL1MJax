# 3C391 CASA calibration reference

## Purpose

The public VLA TDEM0001 3C391 tutorial data provide the first real calibration
benchmark for SL1MJax. CASA remains an independent oracle; its calibration
tables and selected visibilities are exported into a portable NPZ/JSON pair so
the eventual JAX solver does not require CASA at runtime.

## Source and working layout

The immutable source archive is:

```text
data/3c391_ctm_mosaic_10s_spw0.ms.tgz
```

The current local preparation uses:

```text
data/3c391_ctm_mosaic_10s_spw0.ms.tgz   immutable archive
data/3c391/pristine/                    immutable extracted MS
data/3c391_work_v2/                     disposable CASA-calibrated MS
data/3c391/reference-v2/                K/B/G tables, summaries, and plots
data/3c391/reference-pol/               Kcross / Df / Xf tables (polarisation)
data/3c391/golden/                      portable compact NPZ/JSON reference
```

BagOfWinds remains an optional bulk copy. Scripts default to the local
`data/` paths above. Override with `SL1MJAX_3C391_MS`,
`SL1MJAX_3C391_REFERENCE`, `SL1MJAX_3C391_POL_REFERENCE`, and
`SL1MJAX_3C391_GOLDEN`.

## Dataset inventory

- 845,379 rows at 10-second integration;
- one 64-channel spectral window spanning 4.536–4.662 GHz;
- 2 MHz channels;
- RR, RL, LR, and LL correlations;
- 26 active antenna IDs;
- flux/bandpass calibrator `J1331+3030` (3C286);
- time-dependent gain calibrator `J1822-0938`;
- polarization calibrator `J0319+4130`;
- seven 3C391 mosaic fields;
- `DATA`, `MODEL_DATA`, and `CORRECTED_DATA` columns.

The reduced MS has no STATE rows or useful scan intents, despite carrying a
zero-valued row-level `STATE_ID`. Field names therefore provide the calibrator
roles for this historical dataset. Its initial effective flag fraction is
zero.

## Reproducible CASA processing

Run the calibration reference with CASA 6.7.6:

```bash
/Applications/CASA.app/Contents/MacOS/casa --nologger --nogui \
  -c scripts/create_3c391_calibration_reference.py
```

The script follows the CASA 6.7.2 continuum tutorial through calibration
application:

1. save the pristine flag version;
2. flag scan 1, antennas ea13/ea15, and the first 10 seconds of scans;
3. generate antenna-position corrections;
4. install the Perley–Butler 2017 3C286 model;
5. perform the diagnostic phase solve and flag ea05;
6. save the exact calibration-input flag version;
7. solve phase (`G0`), delay (`K0`), bandpass (`B0`), and complex gains (`G1`);
8. transfer the absolute flux scale to J1822-0938;
9. apply the tables to calibrators and target fields;
10. save the post-application flag version and diagnostic plots.

Export portable references with:

```bash
uv run scripts/export_3c391_calibration_golden.py
```

The committed copy is
`tests/fixtures/3c391_calibration_golden.{npz,json}`. Its provenance uses
portable dataset/directory names rather than local absolute paths.

## Reference results

- calibration-input flagged fraction: 28.2236%;
- post-application flagged fraction: 34.2856%;
- J1822-0938 flux at 4.599 GHz: `2.2960 ± 0.0069 Jy`;
- delay solutions span approximately -3.8 to +4.6 ns;
- all active exported antenna-position, phase, delay, bandpass, gain, and
  flux-scaled gain values are finite;
- the compact fixture is 5.6 MB and contains 1,228 3C286 rows at four times,
  1,570 J1822-0938 rows at six times, all 64 channels, RR/LL data, model and
  corrected visibilities, both relevant flag states, and the complete compact
  CASA calibration solutions.

CASA's corrected 3C286 visibility amplitude follows the installed resolved
source model. Its complex normalized residual is about 10% on the selected raw
samples, so that residual is a diagnostic rather than a solver acceptance
threshold. Initial JAX calibration tests should compare gauge-aligned antenna
solutions and their predicted corrected visibilities, not require exact
per-sample agreement with noisy observations.

## SL1MJax validation results

The portable test runs with no CASA runtime:

- imported CASA antenna-position/K/B/G application differs from exported CASA
  `CORRECTED_DATA` by `6.5e-4` normalized complex RMS;
- the staged JAX 3C286 solve gives train/holdout normalized RMS
  `0.0291/0.0310`, below the `0.12` acceptance threshold;
- the transferred J1822-0938 solve gives train/holdout RMS `0.0496/0.0493`;
- gain-ratio flux transfer yields `2.2867 Jy`, 0.41% below CASA's
  `2.2960 Jy`;
- all accepted gain, delay and bandpass solution values are finite, and the
  connected holdout prevents unseen antennas inside represented intervals.

The JAX central-channel stage also solves amplitude for numerical
conditioning; the final full gain stage re-estimates it after delay and
bandpass. The current bandpass uses independent channel values rather than a
smooth basis, and the selected fixture is a compact oracle rather than a full
pipeline replacement.

Target transfer and imaging are now validated on the central `3C391 C1`
pointing. The JAX-calibrated and CASA-calibrated bounded reconstructions have
`0.9960` image correlation and `0.0747` normalized RMS difference. See
[3c391_target_imaging.md](3c391_target_imaging.md) for the exact workflow,
products, limitations and measured visibility/image comparisons.

## Next implementation boundary

Add scalable gridded/NUFFT imaging, primary-beam support and joint mosaic
imaging before target self-calibration. Calibration work should next add
immutable reason-coded flags, robust/smooth solution models and uncertainty
estimates, then validate on a second VLA band/configuration before cross-hand
polarization calibration.
