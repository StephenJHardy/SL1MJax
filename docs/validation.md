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
