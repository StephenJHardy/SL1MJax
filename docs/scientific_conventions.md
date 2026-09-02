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

When circular products are unpacked to Stokes, the CASA convention is
`RR=I+V`, `LL=I-V`, `RL=Q+iU`, `LR=Q-iU`. Jones axes are `(R, L)`. Circular
P Jones is `diag(e^{-iχ}, e^{+iχ})` with the alt-az parallactic angle from
WGS84 geodetic latitude.

## Primary beam

Direction-dependent beam conventions are frozen in
[`vla-beam-reference-inventory.md`](vla-beam-reference-inventory.md) and
`sl1mjax.beam_conventions`.

- Sky-frame `(l, m)` are the same direction cosines as above: `l` east, `m`
  north.
- After `casa_parang_true` calibration the voltage beam is normalized to
  `E(0)=I`. On-axis G, D, X, and P are not applied again inside the beam.
- Unknown calibration-state identifiers are rejected.
- The unused analytic squint stores a receptor half-offset of 0.06 FWHM.
  Published VLA totals are ~0.05–0.06 FWHM. That unused path stays off.
  The Phase 5 diagonal beam uses the Memo 195 total
  \(2.4/\nu_{\rm GHz}\) arcmin and rotates with \(\chi\). It can create
  \(I\rightarrow V\) and cannot create \(I\rightarrow Q/U\).
- Antenna-frame feed position angle and R/L sign are internal conventions
  only. They are not a physically verified C-band oracle.
- The Perley 2016 C-band polynomial fails closed outside 4.052–7.948 GHz
  and outside its 5% power radius. `casa_nearest` is a CASA-parity policy,
  not a spectral interpolation model.
- The voltage-beam evaluator returns Jones on axes
  `(antenna, direction, channel, R/L, R/L)`. Scalar backends are
  diagonal. Perley voltage is the real nonnegative \(\sqrt{P}\). Airy
  voltage is the signed blocked-aperture pattern; that extra \(\pi\)
  phase is an analytic assumption, not part of the scalar power model.
  A composite uses Perley in its spatial support and Airy only for
  in-band directions outside that radius. The handover policy is
  explicit. The streamed operator applies \(E_p C E_q^H\) at each exact
  unique time. Imaging still uses the static Airy power path.
  Phase 6 generates a nominal CASSBEAM C-band Jones first. That
  evaluator uses nearest generated nodes within 64 MHz and CASSBEAM
  `λ/(F N dx)` raster coordinates. The stored origin is the dephased
  FFT DC after CASSBEAM's even-N `reflectMatrix2`, not the documented
  centre row. `casa_parang_true` applies \(P^H E P\); `uncalibrated`
  applies \(E P\). Unfrozen full Jones refuses evaluation, including
  `voltage_beam_for_mode("full_jones")`. A full-Jones interpolator must
  not run until a later visibility-domain or holography argument
  accepts the Jones convention. Stage 1 compares CASA's default
  EVLA ray-traced `awp2` `.pb` with the committed CASSBEAM evaluator.
  The CASA `.pb` products are frozen as a scalar reference; that
  freeze is not CASSBEAM acceptance. I-core agreement inside the 10%
  contour passes. Equivalence inside the 5% contour fails at the
  CASSBEAM first-null skirt; do not loosen the 5% tolerance. CASA
  I/RR/LL `.pb` planes are identical, so they are not a per-receptor
  Jones oracle. Stage 2 compares complex visibilities and requires the
  frozen Stage-1 oracle, not CASSBEAM–CASA equality. Loading CASSBEAM
  tables into CASA is not that oracle. Holography is sought
  only if residuals remain significant. A full-Jones artifact replaces analytic
  squint when it already contains squint, and must not re-apply
  on-axis D/X/P after `casa_parang_true`. Beyond full-Jones support
  the default is a tapered return to the scalar composite;
  `off_diagonal_valid` is cleared there. Do not hard-splice complex Jones
  onto Airy or Perley.

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

Software convention locks, not a beam freeze: finite-pixel integration
preserves the circular \(e^{\pm2i\chi}\) off-diagonal phase
(`test_finite_pixel_preserves_off_diagonal_two_chi_phase`), and an
unpolarised synthetic sky with leakage does not produce fitted Q/U when the
matching full-Jones beam is used
(`test_unpolarised_i_plus_leakage_does_not_invent_qu`). These passed
2026-09-02. They do not accept CASSBEAM off-diagonal sign or freeze full
Jones.
