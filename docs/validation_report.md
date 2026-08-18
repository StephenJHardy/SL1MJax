# Staged validation report

Date: 2026-08-16

## Decision

**GO for a bounded NGC2403 ingestion and imaging smoke test.**

**NO-GO for scientific NGC2403 reconstruction or full AS649 reprocessing.**

The geometric Stokes-I RIME is now independently tied to CASA conventions. The
remaining limitations are predominantly imaging-model and real-data provenance
questions rather than unresolved phase, UVW, flux, polarization-product, or
FITS-axis conventions.

## Evidence

### Internal analytic and recovery gate

- Integrated point/pixel parameters are explicitly `Jy/pixel`; the default
  forward model no longer applies an implicit `1/n` projection factor.
- CASA MeasurementSet UVW values use
  `exp(+2πi [u l + v m + w(n - 1)])`.
- Exact spherical RA/Dec to `(l,m,n)` transforms round-trip across RA wrap and
  high declination.
- The deterministic 8×8 recovery gate reduces its objective by at least 95%,
  has relative image L2 error below 15%, and repeats bit-for-bit.

### CASA convention gate

- Four CASA 6.7.6.14 point-source fixtures cover centre, east, north, and
  diagonal offsets. A fifth resolved, offset circular-Gaussian fixture checks
  integrated flux, FWHM conversion, Fourier normalization, and phase.
- Compact committed golden subsets preserve UVW, frequencies, correlations,
  flags, weights, antennas, FIELD phase centre, and source truth.
- All five forward predictions have normalized complex RMS below `2e-4`.
- Deliberately reversing the phase sign fails the off-axis fixtures by more
  than 5%, making the sign test discriminating.
- The regular grid and FITS SIN WCS now agree: positive eastward `l` is toward
  lower image-column indices with `CDELT1 < 0`; positive `m` is northward.

### Geometry, sampling, and controlled realism

- Channel phase increments scale with frequency to floating-point tolerance.
- An isolated wide-field test verifies the full `w(n-1)` phase.
- CASA VLA A, C, and D configuration fixtures show the expected ordered UV
  extent and inverse resolution scaling.
- A deliberately off-grid source is recovered within one pixel, within 25% in
  a compact aperture, and within 15% normalized visibility RMS.
- Complex thermal-noise statistics, inverse-variance weights, flags,
  zero-weight samples, fixed gains, multiple correlation products, and
  multi-SPW extraction are gated.
- A Gaussian primary beam is available as an explicit, separate model; it is
  not silently mixed into the geometric RIME.

### Reconstruction diagnostics

- A direct naturally weighted adjoint DFT produces dirty-image and PSF
  diagnostics independent of gradient inference.
- CASA-derived east-source visibilities recover the source within one pixel,
  within 2% dirty-image peak flux, with a unit central PSF.
- Structured UV-cell holdout reconstruction reports low train and holdout
  weighted complex MSE and preserves the configured split seed.

## Test result

The complete default suite passes:

```text
84 passed, 1 skipped in 13.46s
```

The skipped test is the intentionally opt-in local VLA gate. Running it against
the imported NGC2403 UVFITS MeasurementSet, using `DATA`, channel 2, and row
stride 100, passes:

```text
1 passed in 3.26s
```

Ruff passes for the complete repository, and mypy reports no issues across all
40 source, test, and script files.

## NGC2403 interpretation and next boundary

The source `n2403.uvfits` is best treated as a likely calibrated target-only
export. Its calibration history is inferred from the old workflow rather than
proven by embedded provenance, and it is not the verified continuum-subtracted
product. The historical `n2403contsub2.bin` was produced later and used a
non-standard RR+LL merge, so it should not become the canonical input.

The next authorized step is therefore a bounded smoke workflow:

1. Import the UVFITS to a MeasurementSet once with CASA.
2. Extract selected active channels into the CASA-independent canonical Zarr
   representation.
3. Characterize flags, weights, channel frequencies, UV coverage, dirty image,
   PSF, residuals, and holdout behavior without claiming scientific fidelity.
4. Design and validate continuum subtraction in canonical visibility space
   against a CASA reference before line-cube reconstruction.

Full AS649 reprocessing remains out of scope until calibration and flagging can
be reproduced from the raw archive products with auditable provenance.

## Known limitations

- The current solver is a positive regular-grid Stokes-I model. Polarization
  products are first-class metadata and are predicted in their native ordered
  correlations, but a full Stokes sky and direction-dependent Jones inference
  are not yet implemented.
- Off-grid model error is materially larger than on-grid error. Gaussian and
  compound pixel bases are now available, but sub-pixel position inference
  remains necessary before quantitative real-source flux claims.
- The primary beam model is idealized and has not yet been matched to CASA's
  VLA beam implementation.
- The compact CASA golden fixtures test direct visibility physics, not exact
  parity with CASA gridding kernels or deconvolution.
