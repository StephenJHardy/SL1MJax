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
- The geometric phase is
  `exp(-2πi [u l + v m + w(n - 1)])`.
- The initial direct model includes the `1/n` projection factor.
- Reversing UVW conjugates the coherency visibility for a real sky.

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
fringe phase, conjugate symmetry, masking and weighting, finite-difference
gradients, gradient-based recovery, canonical-data round trips, and
MeasurementSet extraction through a fake table adapter.
