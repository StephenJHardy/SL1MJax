# Scientific conventions

This document defines SL1MJax independently of the historical SL1M and SL1MML
implementations. Those repositories may be used for diagnostic comparisons,
but they do not define this architecture or optimizer.

## Visibility coordinates

- Canonical UVW coordinates are metres in the standard baseline orientation
  from `antenna1` to `antenna2`.
- Each channel is converted to wavelengths with `frequency_hz / c`.
- Sky coordinates `(l, m)` are direction cosines relative to the phase centre,
  with `n = sqrt(1 - l² - m²)`.
- For UVW values stored by CASA MeasurementSets, the geometric phase is
  `exp(+2πi [u l + v m + w(n - 1)])`. This sign is fixed by CASA-generated
  east/north source fixtures rather than inherited from legacy code.
- Initial pixel/component parameters are integrated flux in `Jy/pixel`.
  Therefore they do not receive an automatic `1/n` factor. That Jacobian is
  available only for an explicitly defined brightness-density integral in
  direction-cosine coordinates.
- Reversing UVW conjugates the coherency visibility for a real sky.

## Pixel visibility models

All pixel models carry integrated flux, and every normalized pixel response is
one at zero baseline.

- `delta` uses the exact spherical geometric phase above.
- `gaussian-paraxial` and `gaussian-wide-field` use a circular Gaussian whose
  public `sigma_pixels` is the ordinary standard deviation in grid-pixel
  spacings. Hardy's Gaussian scale is therefore `sqrt(2π) sigma`, an explicit
  conversion made inside the visibility kernel.
- The paraxial Gaussian integrates the quadratic expansion
  `n - 1 ≈ -(l² + m²)/2` analytically. The wide-field mode multiplies it by the
  first omitted phase correction
  `exp(+2πi w [n - 1 + (l² + m²)/2])`. It approaches the exact delta phase as
  the Gaussian width approaches zero.
- `compound-paraxial` and `compound-wide-field` use the frozen four-Gaussian
  positive radial kernel found by `scripts/gaussian_kernel_search.py`. Its
  radial amplitudes are converted to signed integrated Gaussian weights
  `2π a_k sigma_k²`, which sum to one. One fitted grid parameter still denotes
  one pixel's integrated flux.

The approximation is part of the imaging configuration and must match the
model used to synthesize or interpret data. The delta model remains the
backward-compatible default.

## Polarization

Polarization is never inferred from array position. Every block records an
ordered list of correlation labels and its receptor basis.

For the first-release unpolarized Stokes-I sky, the ideal brightness coherency
uses the common radio convention in which each parallel hand measures `I`:

- linear basis: `XX = I`, `YY = I`, `XY = YX = 0`;
- circular basis: `RR = I`, `LL = I`, `RL = LR = 0`;
- an explicitly Stokes-converted `I` correlation measures `I`.

This convention permits later Q/U/V and Jones terms without a storage
migration. Scalar antenna gains act as
`g[antenna1] * conj(g[antenna2])` on every correlation. The initial imager uses
identity or externally supplied fixed gains.

## Weights, flags, and objective

A visibility contributes only when it is unflagged, finite, and has a finite
positive weight. The data term is the weighted mean squared complex residual,
normalized by active weight across all selected correlations.

Image intensities are represented by unconstrained parameters transformed with
softplus. Sparsity is an explicit prior on physical intensity, not a proximal
solver. Prior terms are normalized by parameter count so their scale does not
change with image size. Optimization uses gradients and Optax.

## Precision and batching

Scientific kernels enable JAX 64-bit mode. Direct visibility prediction is
chunked over samples and must not materialize the full
`visibility × sky-component` matrix.

## Required independent checks

Tests cover zero-baseline flux, linear and circular correlation mapping, known
fringe phase, conjugate symmetry, Gaussian analytic identities, spherical
quadrature, zero-width limits, compound normalization, masking and weighting,
finite-difference gradients, matched recovery for every pixel mode,
canonical-data round trips, and MeasurementSet extraction through a fake table
adapter.
