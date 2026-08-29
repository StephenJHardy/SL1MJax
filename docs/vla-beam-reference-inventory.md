# VLA C-band beam reference inventory

Phase 1 of [`vla-beam-model-proposal.md`](vla-beam-model-proposal.md). The
scalar C-band inventory and internal coordinate conventions are frozen. The
full-polarisation artifact and the physical antenna-frame orientation are
identified routes, not frozen references. Phase 6 generates a nominal
CASSBEAM C-band Jones first and seeks Perley holography only if accepted-mode
residuals remain significant. Phases 2–5 add the voltage evaluator, the
streamed operator, and Memo 195 diagonal squint; they do not add a cache or
change the Airy predict path.

The machine-readable inventory lives in
[`src/sl1mjax/beam_conventions.py`](../src/sl1mjax/beam_conventions.py),
[`src/sl1mjax/data/vla_cband_perley2016.json`](../src/sl1mjax/data/vla_cband_perley2016.json),
and
[`src/sl1mjax/data/vla_cband_full_jones_pin.json`](../src/sl1mjax/data/vla_cband_full_jones_pin.json).
The acquisition log is
[`vla_cband_full_jones_acquisition.md`](vla_cband_full_jones_acquisition.md).
Tests in `tests/test_vla_beam_conventions.py` and
`tests/test_full_jones.py` lock the conventions.

## Artifacts

| ID | Kind | Band | Quantity | Frozen reference | Usable for 3C391 C-band |
| --- | --- | --- | --- | --- | --- |
| `sl1mjax_airy` | analytic | any | Stokes-I power | Yes | Yes: current production path |
| `perley2016_cband_stokes_i` | empirical | C | Stokes-I power polynomial | Yes | Yes: Phase 3 scalar backend |
| `casa_setpbairy` | analytic | any | Stokes-I power | Yes | Geometry oracle only |
| `nrao_gaussian_42_over_nu` | analytic | L–Q | FWHM | Yes | Width check only |
| `cassbeam_go` | electromagnetic | any | 2×2 voltage Jones | No | Identified export route |
| `perley2016_cband_holography_grids` | empirical | C | RR/RL/LR/LL holography | No | Identified reconstruction route |
| `jagannathan2017_asolver` | electromagnetic | L/S/C | 2×2 voltage Jones | No | No packaged C-band table |
| `jagannathan2021_atoz_plumber` | empirical | S | Zernike AIP | No | No: S-band, not C-band |
| `iheanetu2019_lband` | empirical | L | holographic Jones | No | No: must not be used as C-band |
| `evla195_diagonal_squint` | analytic | Cassegrain | Diagonal R/L voltage | No | Yes: Phase 5 magnitude; PA unverified |
| `sl1mjax_analytic_squint` | analytic | any | RR/LL power | No | No: convention not evidence-grade |

### Scalar empirical source

The C-band Stokes-I coefficients are Perley (2016), EVLA Memo 195, Table 5
([PDF](https://library.nrao.edu/public/memos/evla/EVLAM_195.pdf)). They are
empirical holography of 3C147 in 2015–2016, not an electromagnetic model.

- Polynomial: \(P(r)=1+a_2 r^2+a_4 r^4+a_6 r^6\)
- Radius: \(r=\theta_{\rm arcmin}\,\nu_{\rm GHz}\)
- Thirty-two spectral windows from 4052 MHz to 7948 MHz
- Fitted to the 5% power level on the oversampled central grid
- Array average of 14 well-behaved pointing antennas versus ea28
- Typical fit rms 0.5%
- C-band coefficients are frequency-dependent beyond \(1/\nu\) scaling. A
  single band-average polynomial is not sufficient.
- CASA `PBMath1DEVLA` has used these coefficients as the default EVLA beam
  since CASA 5.0. CASA is the engineering oracle; runtime must not invoke
  CASA.

Support stops at about 5% power. Samples outside 4.052–7.948 GHz or outside
that radius are unsupported and fail closed. The catalog must not return a
negative polynomial value as a beam power.

The Memo 195 sparse grid reaches the fourth null but was not used for the
published polynomial. Keep the extended Airy outer support as a separate
ablation until a validated far-sidelobe model exists.

### Frequency policy

`casa_nearest` matches CASA `PBMath1DEVLA.nearestVPArray`. It is a
CASA-parity oracle. It is not the SL1MJax spectral-discovery model.

Adjacent C-band nodes at 4.564 GHz and 4.692 GHz switch at 4.628 GHz. The
3C391 SPW0 span is about 4.536–4.662 GHz, so that switch lies inside the
observed band. The relative jump at the switch is about 1.1% at 4.6 arcmin
and about 4.4% at 8 arcmin. That is large enough to look like source
spectral structure.

An `interpolated` policy is declared and refused until a scientifically
validated interpolation exists. Phase 3 must not adopt nearest-window
selection as the default predict model without saying it is CASA parity.

### Full-polarisation source

There is no packaged public C-band full-Jones table analogous to plumber's
VLA S-band Zernike coefficients.

Identified C-band routes, not frozen references:

1. Export a 2×2 voltage Jones from CASSBEAM / CASA geometric optics (Brisken
   2003; Jagannathan et al. 2017). Bacchus has Ubuntu `cassbeam` 1.1-4build2.
   The nominal C-band input, numerical settings, and output hashes are in
   `src/sl1mjax/data/cassbeam_cband/`. Still unpinned: transmit-to-receive
   conversion, direction-axis orientation, and CASA `awp2` acceptance.
   The packaged 1.5 GHz example is not that input.
2. Reconstruct a measured Jones from the archived Perley holography grids
   (`CHOLO` FITAB/FITTP and `C-BEAM-ffffpp` UVHOL files) only if
   accepted-mode residuals remain significant. Still unpinned: an acquired
   artifact path, immutable checksum, correlation convention, and the
   antenna-average recipe.

Phase 6 acquisition order, not a freeze:

1. Nominal CASSBEAM C-band Jones on Bacchus, labelled electromagnetic.
2. CASA `awp2` acceptance of the diagonal/co-polar mode.
3. 3C391 residual inspection.
4. Processed holography only if those residuals stay significant.

Do not substitute Iheanetu (2019) L-band holography or plumber S-band
coefficients. `full_jones_reference_is_frozen()` is false.

Memo 195 §6.2 records that the holographic Q/U beams were often
phase-distorted and that RL and LR were not conjugates. A reconstructed
full-Jones artifact must declare that limitation.

## Coverage

| Artifact | Frequency | Direction | Antenna | Epoch |
| --- | --- | --- | --- | --- |
| Airy | any, scales as \(1/\nu\) | circular, cutoff \(0.8^\circ\) at 1 GHz | array-average ideal dish | geometric |
| Perley I polynomial | 4.052–7.948 GHz; `casa_nearest` or refused interpolation | radial, to 5% power | 14-antenna average | 2015–2016 |
| Perley holography grids | same SPWs, 28 MHz averages | central 17×17 plus sparse 23×23 to 4th null | selected holography antennas | 2015–2016 |
| CASSBEAM | evaluated at the requested \(\nu\) | full far-field including struts | parameterized VLA Cassegrain | physical model |
| Analytic squint | scales as \(1/\nu\) | opposite RR/LL displacements | same as Airy | unused |

3C391 C-band (~4.6 GHz) sits between the 4564 MHz and 4692 MHz Perley
windows and crosses their `casa_nearest` midpoint. Antenna-to-antenna
C-band power scatter at half power is about 2.5% in the memo. The first
production target remains an array-average beam.

## Frozen conventions

### Sky frame

Phase-centred direction cosines from `radec_to_lmn`:

- \(l\) increases with right ascension (east)
- \(m\) increases with declination (north)
- \(n=\sqrt{1-l^2-m^2}\)
- Beam samples are angular offsets from the pointing centre, not from the
  phase centre when those differ

### Antenna frame

VLA antennas are alt-az. Feed-fixed beam structure rotates on the sky with
the parallactic angle \(\chi\). The current Airy object stores one fixed
\(\chi\) and is not a per-row evaluator. Phase 4 must use
`parallactic_angle_rad` at each unique time.

### Receptors and Stokes

Circular Jones axes are \((R, L)\). CASA circular products:

\[
RR=I+V,\quad LL=I-V,\quad RL=Q+iU,\quad LR=Q-iU.
\]

This matches Memo 195 §6.1, \(I=(R+L)/2\) and \(V=(R-L)/2\), and
`circular_stokes_from_correlations`.

### Parallactic angle

\(\chi\) is the alt-az parallactic angle from WGS84 geodetic latitude,
antenna ECEF, GMST, and phase centre. At southern transit (hour angle zero,
source south of zenith) \(\chi=0\). Circular P Jones is
\(\mathrm{diag}(e^{-i\chi}, e^{+i\chi})\).

Internal locks, not a physical feed-frame oracle:

- At southern transit \(\chi=0\).
- For a source west of the meridian the CASA-compatible formula gives
  \(\chi>0\).
- The unused analytic squint at \(\chi=0\) places RR at \(+l\) and LL at
  \(-l\); at \(\chi=\pi/2\) that RR peak is at \(+m\).

Feed-ring position angle and the Memo 195 “RCP to the right when looking
from behind the antenna along the feed” statement are recorded but not
physically verified. There is no nonzero-\(\chi\) oracle that maps an
antenna-frame R/L or RL/LR pattern into sky-frame \(Q/U/V\). That remains a
Phase 5/6 blocker. `antenna_frame_polarization_is_physically_verified()`
is false.

### Calibration state and on-axis normalization

Every beam application must carry a calibration-state identifier.
`require_beam_calibration_state` accepts only:

| State | Meaning | Beam on axis |
| --- | --- | --- |
| `casa_parang_true` | 3C391 production path. `CORRECTED_DATA` already has CASA G/K/B. JAX applies Kcross, D, X, and P. | \(E^{\rm norm}(0)=I\) |
| `uncalibrated` | Raw visibilities with no direction-independent Jones | raw holographic \(E(0)\) may be kept |

Unknown identifiers are rejected. The beam must not guess a Jones order.

For `casa_parang_true` the direction-independent product is

\[
J = J_{GKB}\,J_{\mathrm{Kcross}}\,J_D\,J_X\,J_P
\]

and the measurement equation is \(V_{pq}=J_p E_p C E_q^H J_q^H\). On-axis
P, D, and G live in \(J\), not in \(E\). The beam still rotates with
\(\chi\) because the antenna-frame pattern is direction-dependent. Do not
apply on-axis P a second time inside the beam.

The current Airy power is already 1 on axis, so it satisfies this
normalization when squint is off.

## Squint audit

Two different lengths must never share a name:

| Name | Meaning |
| --- | --- |
| `receptor_half_offset` | Displacement of one receptor from the pointing centre |
| `total_rcp_lcp_separation` | Distance between the RR and LL power peaks |

CASA `BeamSquint` stores the RR offset from the pointing centre: a
receptor half-offset. LL is the negative of that vector.

Published totals, not half-offsets:

| Source | Total RCP–LCP separation |
| --- | --- |
| EVLA Memo 195 §6.1 | \(2.4/\nu_{\rm GHz}\) arcmin, all Cassegrain bands |
| Cotton & Uson 2008 | \(0.060\pm0.005\) of power FWHM |
| Napier & Gustincic 1977 | 0.053 FWHM, calculated |
| NVSS / Condon et al. 1998 | 0.055 FWHM at 1.4 GHz |

At C-band the Memo 195 total is about 0.057 of the catalog Gaussian FWHM,
so each receptor's half-offset is about 0.0285 FWHM.

The unused SL1MJax constant `VLA_SQUINT_FWHM_FRACTION=0.06` is applied as
a **receptor half-offset**. Opposite hands therefore have a total
separation of 0.12 FWHM, about 2.1 times the Memo 195 total. Cotton's
0.06 is a total, not a half-offset; using that number as a half-offset is
the factor-of-two error the proposal warned about.

`analytic_squint_is_evidence_grade()` is therefore false. Do not enable
squint for evidence-grade work until the C-band magnitude, feed-frame
position angle, and sign are compared with CASA or the holography grids.
When that happens, store half-offset and total separation as different
named quantities.

`DiagonalSquintVoltageBeam` produces \(I\rightarrow V\) and cannot
produce \(I\rightarrow Q/U\). Off-diagonal Jones is still required for
the rotating linear-polarisation leakage seen in the 64″ ancestor \(Q/U\)
experiment.

## Voltage evaluator

Phases 2 and 3 add `src/sl1mjax/voltage_beam.py`. The Jones axes are
`(antenna, direction, channel, receptor_out, receptor_in)` with receptors
`(R, L)`. Perley voltage is the real nonnegative square root of Stokes-I
power. Airy voltage is the signed blocked-aperture pattern. Array-average
backends ignore antenna id, parallactic angle, and elevation.

| Backend | Role |
| --- | --- |
| `AnalyticAiryVoltageBeam` | Exact power parity with `VLAPrimaryBeam` |
| `Perley2016CBandVoltageBeam` | Memo 195 Table 5, `casa_nearest` only |
| `CompositeScalarVoltageBeam` | In-band Perley support; explicit Airy handover |
| `DiagonalSquintVoltageBeam` | Memo 195 R/L offset; \(\chi\) rotation; \(E(0)=I\) |

The imaging predict path still uses the static Airy power operator.
`predict_voltage_beam` is the Phase 4 reference operator: one Jones slice
per exact unique time, then \(E_p C E_q^H\). It is not a cache.

The composite Airy fallback is spatial and in-band only. Frequencies
outside 4.052–7.948 GHz stay unsupported. `match_power` scales Airy so
power is continuous at the Perley 5% radius; `hard_splice` keeps the
~19% power jump and must be requested. Neither handover is a physical
far-sidelobe model.

Width tests check the transcribed Table 5 HWHM, not a live CASA
`PBMath1DEVLA` sample.

`analytic_squint_is_evidence_grade()` remains false for the unused Airy
`0.06` FWHM half-offset. Phase 5 uses `SquintMagnitudePolicy.EVLA195`
instead. Feed-frame PA is still physically unverified.

## Still open after Phases 1–6A

- No cache format
- Phase 6 routes are selected; no C-band full-Jones table is frozen
- BagOfWinds search on 2026-08-29 found no `C-BEAM-*`, `CW-BEAM-*`,
  `CHOLO-*`, or `UVHOL` products
- Compact holography must be requested from NRAO; do not start with the
  ~150 GB FITAB/FITTP databases
- Nominal CASSBEAM C-band Jones generated on Bacchus and checksummed;
  not CASA-accepted and not the imaging path. Nearest generated node
  within 64 MHz; raster scale is CASSBEAM `λ/(F N dx)`; origin is
  dephased DC after even-N `reflectMatrix2`; unfrozen full Jones
  refuses evaluation; CASA `awp2` oracle is not implemented
- Packaged `vla-1500MHz.in` is L-band and is not a C-band configuration
- No acquired Perley holography path or checksum
- No correlation-aware orientation-oracle *values* (the sample list is
  specified)
- Phase 6B interpolating evaluator not implemented
- Phase 6E/6F not started
- No frozen CASA `PBMath1DEVLA` beam samples (scalar CASA-parity only)
- Phase 5 fixed-sky transfer diagnostic not run
- No empirical or squinted backend in the 3C391 predict path
- Phase 4/5 operator not integrated into imaging
- No physically verified antenna-frame polarisation orientation
- No finer sky regions, RM, self-cal, or spatial \(V\)
