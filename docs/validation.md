# First-release validation gates

The release is evaluated on an Apple Silicon CPU baseline unless a test records
another backend.

## Synthetic gate

For the deterministic 8 × 8, three-source, 128-row, two-channel fixture:

- all four linear or circular correlations are retained;
- parallel-hand analytic visibilities agree to `rtol=1e-10`, `atol=1e-12`;
- cross-hands are zero before noise;
- autodiff gradients agree with central finite differences to `rtol=1e-4`;
- training objective decreases by at least 95%;
- recovered peak flux is within 10% and image relative L2 error is below 15%;
- identical seeds produce identical images and objective histories;
- the CPU test completes within 30 seconds with a visibility chunk smaller than
  the complete sample count.

These thresholds are intentionally explicit and may only change alongside a
documented fixture or scientific-model change.

## Imaging validation and regularization

Imaging supports two deterministic, correlation-aware holdout strategies:

- `uv_cell` withholds complete occupied Fourier-plane cells and therefore tests
  interpolation across missing spatial-frequency regions;
- `random_row` withholds complete rows while retaining the training UV
  distribution as a less stringent generalization control.

Both keep every channel and correlation from a selected row together. Reports
include absolute weighted complex MSE and residual power normalized by observed
signal power for both train and holdout samples. The optimizer evaluates the
holdout at a configurable interval and returns the checkpoint with the lowest
holdout loss; holdout samples never contribute gradients.

Positive intensity is enforced by the softplus parameterization. The sparsity
penalty is the L1 norm of integrated pixel flux, `weight * sum(image)`, so its
meaning does not weaken merely because the image contains more pixels.
Smoothness remains an independently configurable mean squared adjacent-pixel
difference. These calculations preserve the configured FP32 or FP64 inference
precision.

## CASA golden gate

Compact CASA 6.7.6.14 fixtures provide an independent convention check without
requiring CASA during routine tests. Schema version 1 declares the sky
components and the currently absent noise and calibration effects.

- point sources cover centre, east, north, and diagonal offsets;
- a resolved, offset, circular 20-arcsec-FWHM Gaussian checks integrated flux,
  FWHM-to-standard-deviation conversion, Fourier normalization, and phase;
- every JAX prediction must have normalized complex RMS below `2e-4`;
- the fixture adapter rejects undeclared sky shapes, noise, or calibration
  effects until their corresponding model and validation logic are added.

The generation and export scripts are retained so future spectral, temporal,
noise, and calibration cases can extend the schema rather than create unrelated
test harnesses.

## MeasurementSet gate

Unit tests use a fake casacore table boundary to cover multi-field,
multi-data-description, multi-spectral-window, weight fallback, flags,
correlation metadata, and provenance without installing CASA.

A real-data release candidate additionally requires `SL1MJAX_TEST_MS` to point
to a small calibrated VLA MeasurementSet. The integration test extracts
`CORRECTED_DATA`, reloads the result in the core environment, images a selected
block, and requires finite correlation-resolved train and holdout metrics. Real
data are not committed to this repository.

The gate has also been exercised locally against:

- a CASA 6.7.6 `simobserve` VLA D-configuration fixture with three known point
  sources, two channels, and circular `RR`/`LL` products;
- the historical NGC2403 VLA UVFITS data imported with CASA, selecting
  unflagged channel 2 and every 100th row for a bounded direct-model run.

The NGC2403 archive has no `CORRECTED_DATA` column after UVFITS import, so that
gate explicitly selects `DATA`. This is recorded in canonical provenance and
does not imply that uncalibrated `DATA` is the preferred science input.

## Product gate

The CLI must produce:

- a WCS-labelled FITS Stokes-I image;
- JSON configuration, provenance, optimizer, prior, and split diagnostics;
- correlation-resolved predicted visibility and residual arrays;
- a resumable Optax checkpoint.
