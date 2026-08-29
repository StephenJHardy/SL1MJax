# VLA direction-dependent beam model proposal

## Purpose

SL1MJax needs a forward model accurate enough that residual structure can be
interpreted as sky structure rather than an omitted instrumental response.
This becomes essential when searching for temporal, spectral, or polarised sky
effects. The VLA beam changes with direction, frequency, receptor, and
parallactic angle. An incomplete beam can therefore look like each of those
sky effects.

The current beam was sufficient to establish that bright sources outside the
main lobe cannot be omitted. It is not a measured full-polarisation VLA beam.
It is an ideal blocked-aperture Airy power pattern with extended support. Its
concentric sidelobes provide a plausible response to outer sources, but their
amplitudes, nulls, and polarisation response are approximate.

This proposal defines a direct VLA reference-beam evaluator and its integration
with the visibility operator. Correctness comes before beam caching,
parallactic-angle interpolation, or low-rank compression. The first execution
path evaluates one beam at each unique measurement timestep and reuses it for
all baselines at that time.

## Decision

Implement a versioned beam model whose fundamental output is a complex
$2\times2$ antenna voltage Jones matrix:

\[
E_a(l,m,\nu,t)=
\begin{pmatrix}
E_{RR} & E_{RL}\\
E_{LR} & E_{LL}
\end{pmatrix}.
\]

The public evaluator accepts direction, frequency, parallactic angle, antenna,
and pointing information. A backend may declare that it is array-averaged or
independent of some coordinates, but it must not silently discard them.

Use a direct streamed reference implementation:

1. group Measurement Set rows by their exact unique time;
2. calculate the parallactic angle and other beam coordinates at that time;
3. evaluate the voltage Jones over the requested sky positions and channels;
4. reuse that evaluation for every baseline at the time;
5. apply $E_p C^{\rm sky}E_q^{\rm H}$ inside the visibility operator;
6. release the timestep beam before evaluating the next one.

An array-average beam is the first production target. The interface retains an
antenna axis so antenna-dependent holography and pointing offsets can be added
without changing the measurement equation.

Do not design a persistent cache or compressed beam basis before the direct
implementation is profiled. Reusing one evaluation across the baselines at a
timestep is part of the reference execution model, not a speculative
optimization.

## Goals

The beam model should:

- represent the response as a complex voltage Jones rather than a scalar
  power correction;
- support all four correlations in their native receptor order;
- evaluate the beam at the exact channel frequencies;
- rotate antenna-frame structure with the row parallactic angle;
- distinguish an array-average reference beam from antenna-specific effects;
- normalize consistently with the direction-independent calibration already
  applied to the data;
- preserve the main-lobe and outer-source lessons from the accepted Stokes-I
  model;
- expose validity, applicability, model version, and reference-data
  provenance;
- provide a slow, clear execution path against which later optimizations can
  be tested;
- reduce false spatial, spectral, and temporal sky discoveries caused by beam
  mismatch.

The first implementation is not intended to:

- infer an arbitrary beam independently for every antenna and timestep;
- reproduce all mechanical changes of an individual VLA antenna;
- treat a published L-band full-Jones beam as a C-band reference;
- optimize cache layout before wall-time and memory measurements exist;
- make a detailed but unvalidated beam authoritative;
- replace calibration, pointing metadata, or held-out sky validation.

## Terminology

A *power beam* describes detected power attenuation. The present Stokes-I
model uses a real power beam $B(l,m,\nu)$.

A *voltage beam* describes the complex antenna response before correlation.
For a scalar antenna response, $B=|E|^2$.

An *E-Jones* is the direction-dependent $2\times2$ voltage response in the
receptor basis. Its diagonal elements are the co-polar receptor beams. Its
off-diagonal elements are direction-dependent leakage.

A *Mueller beam* is the baseline response obtained from the two antenna Jones
matrices. In a correlation-vector representation it is related to
$E_p\otimes E_q^*$. Do not store a full Mueller tensor when the two smaller
antenna Jones matrices are sufficient.

A *reference beam* is a deterministic, versioned model supplied by an
empirical, electromagnetic, or analytic artifact. It is not a beam inferred
from the target sky.

An *array-average beam* uses one reference response for every antenna. An
*antenna-specific beam* may differ by antenna and epoch.

*Beam support* is the direction and frequency domain in which the selected
artifact is validated. Numerical interpolation outside an artifact does not
create scientific support.

## Current SL1MJax beam

[`beam.py`](../src/sl1mjax/beam.py) currently implements:

- a circular Gaussian power beam;
- an analytic Airy power beam for a 25 m dish with a 2.5 m circular blockage;
- exact evaluation at each requested frequency;
- a fixed cutoff radius that scales as $1/\nu$;
- an optional idealized displacement of the RR and LL power beams;
- one fixed parallactic angle stored on the beam object.

The accepted 3C391 composite protocol uses the Airy model with its support
extended to $4^\circ$ at 1 GHz. At the 3C391 band this admits relevant bright
catalogue sources well beyond the main lobe. Held-out visibility tests showed
that this is better than truncating the beam near its early sidelobes.

The current model does not contain:

- the empirical VLA band-dependent radial primary beam;
- two-dimensional main-lobe or sidelobe asymmetry;
- row-dependent parallactic rotation;
- a complex voltage response;
- cross-hand beam attenuation;
- off-axis $I\rightarrow Q/U$ leakage;
- a full $2\times2$ E-Jones;
- frequency-dependent structure beyond geometric $1/\nu$ scaling;
- antenna-dependent pointing or beam variation;
- model uncertainty.

[`composite.py`](../src/sl1mjax/composite.py) prepares the present beam once as
pixel-by-channel arrays for each pointing. The explicit visibility operator
then applies those static arrays. A time-dependent Jones beam requires the
operator to accept a time index and a Jones plane, but it does not require a
new sky representation.

The first three-region coarse-polarisation experiment used the scalar Airy
beam with squint off and did not validate regional $Q/U$. A later experiment
derived 24 regions from the 64 arcsec Stokes-I ancestor, again without using
polarisation to choose the regions. In that experiment, regional $q,u$ beat a
single global $q+iu$ on every leave-one-pointing-out fold. Its relative
improvement over the global model was about 13--19% across the seven held-out
pointings.

The fitted polarisation pattern is not yet evidence for an astrophysical
image. The median regional fractional linear polarisation was about 0.97%, but
three cells exceeded 10% and the largest values were about 51%, 31%, and 12%.
Those cells contain relatively faint Stokes-I emission and lie far off axis.
They therefore have the pattern expected when a spatial sky model absorbs a
missing direction-dependent response. The suspect cells remain off axis in
most of the mosaic pointings, and the observation has little
parallactic-angle diversity. Leave-one-pointing-out transfer consequently
shows that the signal is repeatable, but does not cleanly distinguish a
sky-fixed pattern from a nearly sky-fixed rotating beam pattern.

This result strengthens the case for a full-Jones beam test. It does not
justify finer sky-polarisation regions, RM, frequency-dependent
polarisation, or spatial V before that test. Diagonal squint remains the
necessary test for apparent V. The off-diagonal beam is the necessary test for
whether the regional $Q/U$ gain is compensating for instrumental
$I\rightarrow Q/U$ leakage.

## Why the beam is part of temporal inference

For a static sky coherency $C(s,\nu)$, a simplified visibility is

\[
V_{pq}(\nu,t)=
\int E_p(s,\nu,t)
C(s,\nu)
E_q^{\rm H}(s,\nu,t)
e^{-2\pi i\,\boldsymbol u_{pq}(\nu,t)\cdot s}\,ds.
\]

The VLA is an altitude--azimuth telescope. An asymmetric antenna-frame beam
rotates relative to the sky as parallactic angle changes. A scalar or static
beam approximation leaves a residual with coherent time, direction, frequency,
and polarisation structure.

A flexible sky model can then assign that residual to:

- a variable source;
- a spatial split;
- a spectral index or curvature;
- circular polarisation from RR/LL imbalance;
- linear polarisation from cross-hand leakage;
- a low-RM polarised component.

The beam must therefore be established before the temporal and frequency
searches are treated as discovery evidence. Beam correction is not only a
post-imaging flux correction.

## Available VLA reference levels

There is no single public formula that represents every VLA band, antenna,
frequency, direction, epoch, and full Jones element. The implementation must
separate the evaluator contract from its reference backend.

### Empirical scalar beam

[EVLA Memo 195](https://library.nrao.edu/public/memos/evla/EVLAM_195.pdf)
reports measured beam sizes and polynomial fits across the VLA bands. Modern
CASA uses those Perley beam parameters for ordinary VLA primary-beam
correction. This is the first reference for the scalar C-band main beam.

The empirical radial beam is more faithful to the measured VLA main lobe than
the ideal Airy pattern. It is still an azimuthally averaged power response. It
does not supply the full cross-polarisation beam.

### Electromagnetic or ray-traced beam

VLA physical models such as cassbeam represent the feed position, primary and
secondary reflectors, blockage, support geometry, and diffraction. They can
produce complex, two-dimensional receptor beams. CASA A-projection uses
aperture-illumination machinery to account for direction-dependent,
frequency-dependent, and rotating beam effects when configured to do so.

CASA remains a useful independent oracle. Runtime SL1MJax beam evaluation must
not depend on invoking CASA inside an optimizer.

### Holographic full-Jones beam

Full-polarisation holography measures RR, RL, LR, and LL as complex functions
of direction and frequency. It exposes squint, beam squash, sidelobe
distortion, null movement, cross-polarisation, standing-wave structure, and
antenna differences.

[Iheanetu et al. 2019](https://academic.oup.com/mnras/article/485/3/4107/5374534)
describe compact representations of VLA L-band holography. That work
demonstrates the representation needed here, but an L-band artifact must not
be relabelled as a C-band beam. The first inventory task is to identify an
authoritative C-band full-Jones artifact or a reproducible way to export one
from an accepted physical model.

### Analytic fallback

The current Airy model remains useful for:

- tests with a known closed form;
- outer-field support when no validated empirical far-sidelobe model exists;
- controlled ablations;
- detecting whether a result depends on an empirical reference artifact.

It must be labelled `analytic_blocked_airy`, not `measured_vla`.

## Reference evaluator contract

The evaluator should be independent of cache policy and visibility layout. A
conceptual interface is:

```python
@dataclass(frozen=True)
class BeamCoordinates:
    l_rad: np.ndarray
    m_rad: np.ndarray
    frequency_hz: np.ndarray
    parallactic_angle_rad: np.ndarray
    antenna_id: np.ndarray | None = None
    pointing_offset_lm_rad: np.ndarray | None = None
    elevation_rad: np.ndarray | None = None


@dataclass(frozen=True)
class BeamEvaluation:
    jones: np.ndarray
    valid: np.ndarray
    provenance: dict[str, object]


class VoltageBeamModel(Protocol):
    def evaluate(self, coordinates: BeamCoordinates) -> BeamEvaluation: ...
```

The exact array-axis order must be fixed once and serialized. For one streamed
timestep, a suitable logical result is:

```text
(antenna_or_one, direction, channel, receptor_out, receptor_in)
```

The contract must define:

- whether $l,m$ are direction cosines or small-angle coordinates;
- whether directions are supplied in the sky or antenna frame;
- the receptor basis and order;
- the parallactic-angle convention;
- voltage normalization and phase convention;
- the reference frequency and interpolation rule;
- how a pointing offset is applied;
- whether the artifact is array-average or antenna-specific;
- the validity domain and extrapolation policy.

Backends must report the coordinates on which they actually depend. An
array-average model may ignore `antenna_id`, but its provenance must say so.

## Calibration normalization and Jones ordering

The reference beam cannot be multiplied into calibrated data without defining
which on-axis terms have already been removed.

Let $E_a^{\rm raw}(s,\nu,t)$ be a reference voltage beam. Direction-independent
calibration absorbs the on-axis receptor response. An indicative residual beam
is

\[
E_a^{\rm norm}(s,\nu,t)
=
\left[E_a^{\rm raw}(0,\nu,t)\right]^{-1}
E_a^{\rm raw}(s,\nu,t),
\]

so $E_a^{\rm norm}(0)=I$. The exact composition must match the adopted
calibration chain and be verified against synthetic and CASA reference data.

This normalization is important because a holographic Jones may contain
on-axis receptor gain, phase, and leakage already represented by G, D,
Kcross, or X. Applying them again would double-calibrate the data.

The contract must also state whether parallactic-angle correction was already
applied. The current 3C391 polarisation path uses CASA-compatible calibration
with `parang=True`. The direction-dependent beam still rotates relative to the
sky, but the on-axis P term must not be applied twice.

Record a calibration-state identifier with every beam application. Reject an
unknown state rather than guessing the Jones order.

## Squint convention audit

The dormant analytic squint uses `squint_fwhm_fraction=0.06` and shifts RR and
LL by the full amount in opposite directions. This creates a total RR--LL
separation of 0.12 FWHM.

Published VLA descriptions usually quote a total RCP--LCP separation of about
5--8% of the FWHM, with band- and frequency-dependent values in EVLA Memo 195.
The current parameter may therefore use the wrong half-separation convention.
Do not enable it for evidence-grade work until the following are explicit:

- whether the stored value is one receptor's offset or the total separation;
- the C-band magnitude as a function of frequency;
- the feed-frame position angle and sign;
- the parallactic-angle rotation convention;
- the relationship between the power-beam displacement and the voltage beam.

The audit should compare predicted off-axis RR/LL ratios with CASA or a VLA
reference at several directions, frequencies, and parallactic angles.

## Full-polarisation structure

### Diagonal receptor beam

A diagonal model is

\[
E(s,\nu,t)=
\begin{pmatrix}
E_R(s,\nu,t) & 0\\
0 & E_L(s,\nu,t)
\end{pmatrix}.
\]

It supplies:

- the empirical receptor beam shape;
- RR/LL squint;
- the correct cross-hand attenuation $E_RE_L^*$;
- parallactic rotation of the receptor patterns;
- frequency dependence.

For an ideal unpolarised sky it can create apparent Stokes V through unequal
RR and LL attenuation. It cannot create $I\rightarrow Q/U$ leakage.

### Off-diagonal beam

The full Jones includes $E_{RL}$ and $E_{LR}$. These terms produce the
direction-dependent Mueller response, including the familiar rotating
$I\rightarrow Q/U$ structure. The NRAO polarimetry guide warns that wide-field
linear-polarisation accuracy is limited by this angular response. Simulations
in [Jagannathan et al. 2017](https://arxiv.org/abs/1706.01501) show that it can
be comparable with astrophysical polarisation when parallactic-angle coverage
is limited.

This is directly relevant to the 3C391 coarse $Q/U$ result. Squint is a
necessary correction, particularly for V, but it is not the complete
linear-polarisation beam.

## Direct per-timestep execution

The `VisibilityBlock` already contains exact row times, frequencies, antenna
indices, phase centre, and correlations. Antenna positions and a tested
[`parallactic_angle_rad`](../src/sl1mjax/calibration_terms.py) function already
exist.

The reference execution path is:

```python
unique_time_s, row_time_index = np.unique(block.time_s, return_inverse=True)

for time_index, time_s in enumerate(unique_time_s):
    selected_rows = row_time_index == time_index
    coordinates = beam_coordinates_for_time(block, selected_rows, time_s)
    evaluation = beam.evaluate(coordinates)
    prediction[selected_rows] = predict_with_beam_jones(
        block,
        selected_rows,
        sky,
        evaluation,
    )
```

For an array-average beam, one Jones plane is shared by every baseline. For an
antenna-specific model, evaluate one Jones per participating antenna and form
the baseline response from `antenna1` and `antenna2`.

The first version may use a Python loop outside the JIT boundary. Clear
correctness, inspectable intermediate arrays, and deterministic memory use are
more important than one fused compilation at this stage.

The implementation should support three execution policies without changing
the physical evaluator:

- `stream`: evaluate, consume, and release one timestep;
- `retain_last`: keep the last slice while all rows at that time are used;
- `materialize`: retain all slices for a small test or benchmark.

`stream` is the default reference policy. `retain_last` describes normal
reuse within a timestep and does not need a general cache manager.

## Computational scale

For an array-average beam at one time, evaluation scales as

\[
O(N_{\rm pixel}N_{\rm channel}).
\]

The direct Fourier prediction at the same time scales as

\[
O(N_{\rm baseline}N_{\rm pixel}N_{\rm channel}).
\]

The VLA has up to 351 cross-correlation baselines for 27 antennas. A beam value
is therefore reused many times relative to the Fourier response. A tabulated
interpolation or moderate analytic evaluator should not dominate the current
direct DFT.

For the current 11,536-pixel, 8-channel model, one array-average $2\times2$
Jones slice uses about 2.8 MiB in complex64. At 64 channels it uses about
23 MiB. These sizes are practical for timestep streaming.

Do not materialize the full row-by-pixel-by-channel Mueller response. It would
duplicate a factored antenna response and could exceed memory rapidly.

## Visibility-operator integration

The existing direct operator streams visibility and pixel tiles and supplies
an explicit adjoint. Extend that operator rather than constructing a dense
measurement matrix.

Inside one visibility/pixel tile:

1. calculate the Fourier phase once;
2. form the intrinsic sky coherency for the pixels;
3. apply the two antenna E-Jones matrices;
4. sum the four predicted correlations through the shared Fourier response;
5. retain the same factorization in the adjoint.

A naive implementation that runs sixteen unrelated scalar DFTs would waste
the shared Fourier phase. A fused reference implementation may still be slower
than the scalar path, but it should express the coherency algebra directly and
remain testable.

The beam is initially fixed, so differentiation is required with respect to
the sky but not the beam artifact. Later low-dimensional pointing or beam-mode
parameters may become differentiable without changing the evaluator result
shape.

## Reference artifact

Serialize beam metadata separately from cached evaluations. A possible shape
is:

```python
@dataclass(frozen=True)
class BeamModelArtifact:
    schema_version: int
    model_id: str
    telescope: str
    backend: str
    receptor_basis: str
    band: str
    frequency_domain_hz: tuple[float, float]
    direction_domain: dict[str, object]
    antenna_scope: str
    epoch_scope: dict[str, object] | None
    frame_convention: dict[str, object]
    normalization: dict[str, object]
    parameter_dependencies: tuple[str, ...]
    source_artifacts: tuple[dict[str, object], ...]
    validation_summary: dict[str, object]
    provenance: dict[str, object]
```

The artifact must record:

- whether values are voltage, power, Jones, or Mueller;
- complex normalization and receptor ordering;
- direction-coordinate and parallactic-angle conventions;
- band, frequency samples, and interpolation rules;
- spatial support and extrapolation behavior;
- whether it is array-average or antenna-specific;
- the antennas, observation dates, or physical model used;
- raw artifact hashes and conversion software version;
- comparison with CASA or another independent reference;
- known limitations, including missing beam elements.

Unknown conventions must fail closed. A scalar or diagonal artifact must not
be silently promoted to a measured full-Jones beam.

## Beam validity and uncertainty

Every evaluation returns validity. Initial validity classes should distinguish:

| Support | Meaning |
|---|---|
| `measured` | Interpolation inside a measured beam artifact |
| `physical_model` | Supplied by a validated electromagnetic model |
| `analytic` | Supplied by an explicitly idealized model |
| `extrapolated` | Outside the validated frequency or direction domain |
| `unsupported` | No defensible response exists |

The first implementation may return a boolean array plus the artifact-level
class. Preserve an upgrade path to element-specific uncertainty or beam modes.

Beam uncertainty matters in weak sidelobes. A source flux and an uncertain
beam attenuation are partly degenerate. Do not quote precise intrinsic flux
for an outer source merely because the analytic beam produces a point
estimate.

Later inference may use low-dimensional nuisance parameters for:

- pointing offset;
- beam width;
- squint magnitude and direction;
- a small number of holographic residual modes.

Those parameters need external priors and held-out validation. The target must
not be allowed to invent an arbitrary beam that absorbs sky variability.

## Cache policy after profiling

The evaluator contract is also the natural boundary for a future cache. No
cache design is selected now.

If profiling shows a material cost, candidate policies include:

- cache exact unique-timestep slices;
- cache by rounded parallactic angle with a declared interpolation error;
- cache antenna-frame spatial planes and rotate coordinates;
- cache frequency basis planes and interpolate coefficients;
- use PCA, Zernike, or another low-rank representation;
- retain only active pixel and channel subsets;
- cache per-antenna perturbations around an array-average beam.

A cache key must include:

- beam model identifier and artifact hash;
- direction-grid or topology hash;
- channel frequencies;
- parallactic angle or time coordinate;
- antenna and pointing offsets where relevant;
- normalization and calibration-state identifiers;
- numeric precision and implementation version.

A cache hit must reproduce the direct evaluator within a predeclared error.
Cache interpolation is a model approximation and must be validated as such.

## Build plan

### Phase 1: reference inventory and conventions

1. Inventory the C-band scalar and full-polarisation reference artifacts that
   can be obtained from VLA/CASA sources.
2. Record whether each artifact is empirical, electromagnetic, or analytic.
3. Determine its frequency, direction, antenna, and epoch coverage.
4. Fix sky-frame, antenna-frame, R/L, Stokes, and parallactic-angle
   conventions.
5. Audit the existing squint half-separation convention.
6. Define how the beam is normalized after G/D/X/P calibration.

Deliverable: a short reference inventory and frozen convention tests. Do not
begin with a cache format.

### Phase 2: evaluator contract and analytic backend

1. Add `BeamCoordinates`, `BeamEvaluation`, and `VoltageBeamModel`.
2. Wrap the current Airy response as an explicit analytic backend.
3. Return a diagonal voltage Jones whose power reproduces the current scalar
   path.
4. Preserve the current static operator as a regression path.
5. Add validity and provenance.

Deliverable: the full-Jones interface with exact parity for the accepted
Stokes-I Airy model.

### Phase 3: empirical scalar C-band backend

1. Import the VLA C-band empirical radial coefficients with their provenance.
2. Evaluate them as a voltage beam with a documented phase convention.
3. Compare beam width and attenuation with CASA at selected frequencies and
   radii.
4. Preserve extended Airy support as a separate outer-field ablation until a
   validated empirical or physical far-sidelobe model replaces it.

Deliverable: a measured scalar main-beam model without losing the known outer
sky.

### Phase 4: streamed per-timestep operator

1. Group rows by exact unique time.
2. Calculate row/antenna parallactic angles using the existing geometry.
3. Evaluate one Jones slice at a time.
4. Apply the Jones coherency equation to all four correlations.
5. Implement the exact adjoint with respect to the sky.
6. Compare `stream` and `materialize` on small fixtures.

Deliverable: a correct, inspectable reference operator with bounded memory.

### Phase 5: diagonal C-band receptor beam

1. Add band-specific R/L squint magnitude, direction, and frequency scaling.
2. Rotate the receptor beams per timestep.
3. Use voltage products for RR, RL, LR, and LL.
4. Verify that diagonal squint produces $I\rightarrow V$ but not
   $I\rightarrow Q/U$ for an unpolarised sky.
5. Rerun the paired 3C391 squint-off/squint-on diagnostics.

Deliverable: the first physically coherent direction-dependent polarisation
beam.

### Phase 6: full-Jones C-band reference

1. Import or reproducibly generate the selected complex C-band Jones artifact.
2. Normalize it consistently with on-axis calibration.
3. Interpolate direction and frequency inside the validated domain.
4. Preserve its off-diagonal elements.
5. Compare correlations and Stokes leakage with the external oracle.
6. Rerun the 64 arcsec I-ancestor $Q/U$ validation without changing its sky
   regions, flags, calibration, or folds.
7. Compare three fixed explanations on the same held-out data:
   - scalar beam with regional $q,u$;
   - full-Jones beam with global $q,u$;
   - full-Jones beam with regional $q,u$.
8. Report whether the full-Jones beam reduces the extreme off-axis $p_L$, the
   held-out advantage assigned to regional sky freedom, and the residual
   dependence on beam radius and parallactic angle.

Deliverable: an array-average, time-rotating full-polarisation C-band beam.

### Phase 7: performance measurement

Profile separately:

- coordinate preparation;
- reference beam evaluation;
- data transfer to the accelerator;
- Jones/coherency application;
- Fourier response;
- forward and adjoint passes;
- peak memory by execution policy.

Only then choose whether exact-timestep caching, angle interpolation, or a
low-rank beam basis is needed.

### Phase 8: later antenna-specific extensions

Add only when validation shows a material limit:

- antenna-specific holographic residuals;
- POINTING-table offsets;
- time-dependent pointing or focus;
- elevation-dependent structure;
- beam uncertainty modes;
- epoch-dependent artifacts.

These extensions must retain an array-average control.

## Test plan

### Unit tests

Test:

- coordinate broadcasting and axis order;
- receptor-basis validation;
- centre normalization to identity;
- voltage-to-power consistency;
- frequency and direction validity boundaries;
- parallactic-angle sign and rotation;
- squint half-separation convention;
- pointing-offset composition;
- deterministic provenance and artifact hashing;
- refusal of unknown calibration state;
- equality of stream and materialized evaluation.

### Synthetic coherency tests

Use analytic skies to establish:

- an unpolarised centre source remains unpolarised;
- diagonal squint produces the expected antisymmetric apparent V off axis;
- diagonal squint does not create Q/U from unpolarised I;
- a polarized source receives the expected cross-hand attenuation;
- off-diagonal beam terms create the injected $I\rightarrow Q/U$ pattern;
- applying the correct beam recovers the injected static sky;
- omitting a rotating beam produces an apparent temporal residual;
- the correct beam removes that false temporal evidence;
- the physical Stokes constraint is preserved by the coherency calculation.

### Operator and gradient tests

Test:

- full-Jones prediction against a dense small-matrix implementation;
- the streamed forward result against a materialized response;
- the explicit adjoint inner-product identity;
- JAX gradients with respect to I, Q, U, and V against finite differences;
- missing correlation and flagged-product behavior;
- antenna-average and identical antenna-specific paths agree;
- chunk-size changes do not change the result.

### External reference tests

Build compact golden cases at:

- beam centre;
- several main-lobe radii;
- the half-power radius;
- both sides of the squint direction;
- the first null and selected sidelobes;
- several C-band frequencies;
- several parallactic angles;
- unpolarised, linearly polarised, and circularly polarised input coherencies.

Compare native RR, RL, LR, and LL predictions with CASA, cassbeam, holography,
or another selected oracle. Scalar power agreement alone is insufficient for
the full-Jones backend.

### Fixed 3C391 tests

Repeat the existing partitions without choosing new sky regions:

- the frozen Stokes-I composite sealed test;
- outer-catalogue source ablations;
- global $q+iu$ leave-one-pointing-out validation;
- the three-region coarse $Q/U$ validation;
- the 24-region, 64 arcsec I-ancestor $Q/U$ validation;
- baseline, time, and channel transfers in both directions;
- beam-radius and parallactic-angle cohorts;
- observed residual V versus predicted squint V;
- null temporal searches on the beam-corrected static sky.

The comparison must report analytic Airy, empirical scalar, diagonal squint,
and full-Jones stages separately. A final model should not conceal which beam
term produced an improvement. It must also report the fixed three-way
comparison between scalar-beam regional $q,u$, full-Jones global $q,u$, and
full-Jones regional $q,u$. This comparison tests whether beam structure can
explain the predictive gain currently assigned to spatial sky polarisation.

## Acceptance criteria

The evaluator contract is accepted when:

- the analytic backend reproduces the existing Stokes-I prediction;
- centre normalization and calibration ordering are explicit;
- the direct streamed and materialized paths agree numerically;
- synthetic full-Stokes cases pass the dense reference and gradient tests;
- the external golden covers direction, frequency, parallactic angle, and all
  four correlations;
- unsupported reference coordinates fail closed.

The empirical scalar beam is accepted when it matches the declared VLA/CASA
reference within a predeclared tolerance and does not lose the held-out outer
source improvement without an explicit replacement model.

The diagonal squint model is accepted when its band-specific separation and
rotation match the oracle and its 3C391 V prediction transfers across
pointings better than the no-squint model.

The full-Jones model is accepted when it:

- improves or preserves held-out full-correlation visibility loss;
- reduces beam-radius and parallactic-angle dependence;
- lowers false $Q/U/V$ on suitable controls;
- improves the transfer of a fixed sky polarisation model across pointings;
- materially reduces extreme off-axis fractional polarisation when that
  signal is produced by the scalar-beam mismatch;
- quantifies how much of the previous regional-$q,u$ advantage remains after
  the direction-dependent $I\rightarrow Q/U$ response is included;
- reduces null temporal or spectral candidates associated with beam rotation;
- preserves the established Stokes-I sky and outer-source evidence.

Passing an in-sample image comparison is not enough.

## Reporting requirements

Every beam run should report:

- artifact and implementation identifiers;
- receptor, coordinate, and calibration-state conventions;
- direction and frequency support;
- array-average or antenna-specific scope;
- number of unique times and beam evaluations;
- execution policy;
- time spent in beam evaluation, Jones application, DFT, and data movement;
- peak memory;
- validity and extrapolation counts;
- paired held-out scores against simpler beam stages;
- residuals by pointing, beam radius, parallactic angle, frequency, baseline,
  and correlation;
- whether the result is engineering, exploratory, or evidence-grade.

Store the beam artifact, sky checkpoint, calibration solution, flags, and
validation partitions by immutable identifier. A sky result without the exact
beam provenance is not reproducible.

## Risks and controls

### A detailed beam can still be the wrong beam

Use only artifacts whose band and applicability domain match the observation.
Keep simpler reference stages as ablations. Do not substitute L-band
holography for a C-band beam.

### Beam and calibration can be applied twice

Normalize the beam against its on-axis Jones and serialize the calibration
state. Test the complete order against an external oracle.

### A squint convention can be wrong by a factor of two

Store half-offset and total separation as different named quantities. Compare
both beam centres, not only one receptor, against the reference.

### Beam interpolation can create false spectra

Validate interpolation on withheld frequencies. Record extrapolation and do
not silently extend an artifact across a band edge.

### Beam rotation can create false temporal evidence

Use exact unique times in the reference path. Any later parallactic-angle
coarsening must reproduce the direct path and pass temporal null injections.

### Full materialization can exhaust memory

Stream one timestep and retain antenna Jones rather than baseline Mueller
tensors. Add a memory budget check before allocating reference slices.

### Beam freedom can absorb the sky

Keep the first beam deterministic and externally referenced. Later pointing
or residual-beam parameters need priors and held-out validation. Do not tune a
free beam image on the target.

### A scalar main beam can remove needed far-sidelobe support

Retain the extended Airy outer-field component as an explicit ablation until a
validated physical or empirical far-sidelobe model replaces it.

### CASA agreement can hide shared assumptions

CASA is an engineering oracle, not the only scientific evidence. Compare with
published beam measurements or holography where possible and retain real-data
cross-pointing validation.

## Relationship to other plans

This proposal supplies the direction-dependent E-Jones for
[`jones-polarization-calibration.md`](jones-polarization-calibration.md). The
on-axis G/K/B/Kcross/D/X/P chain and the beam must have one explicit order and
normalization.

It protects the temporal and spectral discovery described in
[`time_frequency_sky_model_handoff.md`](time_frequency_sky_model_handoff.md).
A rotating or chromatic beam should not be offered to the sky model as a
transient, spectral index, or RM component.

It complements
[`calibration-prior-proposal.md`](calibration-prior-proposal.md). The
calibration GP describes externally anchored direction-independent gain
variation. It should not be forced to absorb a direction-dependent beam error.

It also interacts with
[`calibration-flag-proposal.md`](calibration-flag-proposal.md). A beam artifact
outside its validated support is a model-support problem and must not silently
promote a visibility to evidence-grade use.

The outer-source evidence and extended Airy history remain documented in
[`3c391_corner_pixels.md`](3c391_corner_pixels.md). This proposal does not
withdraw that result. It changes the interpretation of the analytic
sidelobe amplitudes from precise VLA measurements to a validated useful
approximation.

## Immediate next step

Do not build a cache first.

Create the Phase 1 C-band reference inventory and freeze the beam conventions.
In particular, identify the source of the scalar empirical coefficients, the
best available full-Jones C-band artifact or physical oracle, and the meaning
of the published squint separation.

Then implement the evaluator contract and wrap the current Airy beam as a
normalized diagonal voltage Jones. This creates a regression-safe path from
the present model to the streamed per-timestep operator. Profile only after
the direct empirical and full-Jones candidates can be compared on the same
3C391 partitions.
