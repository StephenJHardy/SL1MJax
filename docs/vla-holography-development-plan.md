# VLA holography RIME development plan

## Purpose

SL1MJax needs an empirical C-band full-Jones beam whose coordinate,
polarisation, and normalization conventions are established by measurements.
The Perley observations behind EVLA Memo 195 provide that opportunity. They
sampled 3C147 with full correlations while some VLA antennas rastered across
the source and several reference antennas remained on axis.

These observations are not an ordinary sky mosaic. A mosaic gives every
antenna on a baseline the same pointing centre. In holography, antenna $p$
and antenna $q$ may have different pointing directions at the same time.
The forward model must therefore carry a pointing offset for each antenna and
time.

The current voltage-beam operator already evaluates a full-Jones
$E_p C E_q^{\rm H}$ measurement equation. It already distinguishes the two
antennas on a baseline, evaluates at unique times and native frequencies, and
packs circular correlations. Its current pointing offset is a single
$(l,m)$ pair shared by an entire evaluation. The main architectural work is
to add a validated per-antenna pointing model without changing the meaning of
the correlator phase centre.

This document defines the ingestion, RIME, calibration, beam-recovery,
testing, validation, and freeze sequence. The first target is the lower
C-band `THOL0001` execution from 14 January 2016. The upper-C execution and an
independent polarimetric science observation extend the validation after that
path works.

## Outcome

The target system should be able to:

- ingest VLA holography pointing metadata without flattening it into mosaic
  fields;
- distinguish the correlator phase centre, source direction, antenna pointing
  direction, and feed-frame beam coordinate;
- evaluate a scalar, diagonal, or full-Jones RIME with a different pointing
  offset for every antenna and time;
- reproduce the current ordinary-mosaic operator when all antenna offsets are
  equal;
- calibrate the holography data without allowing raster motion to be absorbed
  into direction-independent gains;
- compare Airy, Perley, CASSBEAM, and empirical holographic beams directly in
  visibility space;
- reconstruct complex RR/RL/LR/LL beam samples with uncertainty and support
  information;
- estimate array-average and antenna-specific beam products;
- determine the handedness, axis, conjugation, and parallactic-angle
  conventions from held-out measurements;
- package a versioned empirical C-band beam for the existing
  `VoltageBeamModel` interface;
- refuse evidence-grade full-Jones imaging until the empirical artifact and
  its conventions pass the declared gates.

This work does not initially require sky deconvolution. The raster target is a
compact calibrator, so the first holography operator has one source component
rather than a large adaptive image.

## Executive decision

Proceed in this order:

1. freeze an immutable inventory of each downloaded execution;
2. audit the Measurement Set pointing and observing-mode metadata;
3. add a per-time, per-antenna pointing representation;
4. extend the NumPy and JAX voltage operators to consume that representation;
5. prove the generalized RIME with synthetic data and ordinary-mosaic
   regressions;
6. create separate direction-independent and full-polarisation calibrated
   holography products;
7. validate the observed raster geometry and identify reference and moving
   antennas from the data;
8. compare the existing CASSBEAM beam directly with held-out holography
   visibilities under an explicit convention ladder;
9. recover diagonal empirical beams before recovering off-diagonal terms;
10. build an array-average full-Jones artifact with uncertainty and support;
11. freeze it only after cross-frequency, cross-antenna, and held-out spatial
    validation;
12. transfer the frozen beam to 3C391 and an independent C-band observation.

Software correctness gates fail closed. Missing metadata, invalid calibration
state, correlation-order ambiguity, and operator disagreement are errors.
Scientific gates may warn and continue during exploratory work, but the beam
factory must still refuse to label an artifact frozen until every required
scientific gate passes.

## Available observations

### Lower C band

The identified archive execution is:

- project: `THOL0001`;
- scheduling block: `sb31628704`;
- execution block: `eb31629959`;
- observation: 14 January 2016, 04:03:28--07:12:28 UTC;
- nominal coverage: 3.99--6.01 GHz;
- source: 3C147;
- archive size: about 122 GB;
- Memo 195 product family: `CHOLO-LO`.

Memo 195 describes two raster regimes in this execution:

- a dense 17 by 17 central grid with 1.72 arcmin spacing;
- a sparse 23 by 23 outer grid with 4.59 arcmin spacing.

The dense grid reaches about 13.7 arcmin from the centre. The sparse grid
reaches about 50.5 arcmin. Each settled raster point was observed for about
10 seconds with one-second correlator dumps.

### Upper C band

The upper-C observation was made on 18 January 2016 and covers approximately
5.99--8.01 GHz. Memo 195 describes a dense 17 by 17 grid with 1.28 arcmin
spacing and a sparse 21 by 21 grid with 3.40 arcmin spacing. Its archive
project and execution identifiers must be recorded when the request product
arrives.

### Independent polarimetric observation

The 24A-063 observation provides an independent calibration and science
transfer dataset. It is not a substitute for holography. Its role is to test
whether a beam and calibration convention established on the Perley raster
improves held-out full-polarisation visibilities on unrelated data.

### Processed reference products

Memo 195 names the smaller processed products:

- `C-BEAM-ffffpp01` for the dense central raster;
- `CW-BEAM-ffffpp01` for the sparse outer raster;
- `ffff` is the central frequency in MHz;
- `pp` is one of I, Q, U, V, RR, RL, LR, or LL.

These products are useful independent reconstruction oracles if they become
available. The raw Measurement Set path must not assume they will be
available.

## Measurement equation

### General form

For source direction $s$, the baseline visibility is

\[
V_{pq}(t,\nu)=
\int E_p(s-\Delta_p(t),\nu,t)
C(s,\nu)
E_q^{\rm H}(s-\Delta_q(t),\nu,t)
e^{-2\pi i\boldsymbol u_{pq}(t,\nu)\cdot s}\,d\Omega.
\]

Here:

- $C$ is the celestial sky coherency;
- $E_a$ is the direction-dependent antenna voltage Jones;
- $\Delta_a(t)$ is the pointing displacement of antenna $a$;
- $\boldsymbol u_{pq}$ is the baseline coordinate in wavelengths;
- $s$ remains relative to the correlator phase centre.

Changing an antenna pointing changes the argument of $E_a$. It does not
change the correlator phase centre or the Fourier kernel.

### Compact calibrator approximation

The beam changes negligibly across the angular extent of 3C147. The source
may nevertheless have baseline-dependent visibility structure. The useful
holography approximation is therefore

\[
V_{pq}(t,\nu) \simeq
E_p(-\Delta_p(t),\nu,t)
S_{pq}(t,\nu)
E_q^{\rm H}(-\Delta_q(t),\nu,t),
\]

where $S_{pq}$ is the coherency visibility of 3C147. A point-source model is
a special case, not an assumption to impose on all baselines.

For an unpolarised point source at the phase centre,

\[
S_{pq}=\frac{I_\nu}{2}
\begin{pmatrix}1&0\\0&1\end{pmatrix}.
\]

If $q$ is a fixed reference antenna and the calibrated on-axis response is
normalized to identity, then

\[
V_{pq} \simeq \frac{I_\nu}{2}E_p(-\Delta_p).
\]

This is why moving--reference baselines provide a direct beam measurement.
The approximation must still carry the source model, remaining reference
antenna response, and calibration uncertainty.

### Coordinate sign

If an antenna is commanded to point to positive $+l$ while the source stays
at the phase centre, the source appears at negative $-l$ in that antenna's
beam coordinate. This sign must be tested rather than inferred from a plot.

The implementation must name four different coordinates:

1. celestial direction relative to the correlator phase centre;
2. commanded or measured antenna pointing direction;
3. source direction relative to that antenna pointing;
4. antenna or feed-frame direction after the applicable rotation.

No field called simply `offset` should cross these boundaries without a frame
and sign declaration.

## Identifiability and gauge

Holography observes products $E_p S E_q^{\rm H}$, not an isolated absolute
Jones for either antenna. A transformation applied consistently to the source
coherency and every antenna can leave the visibilities unchanged. The result
therefore needs external anchors.

Use these anchors:

- the flux and structure model of 3C147 establishes the source amplitude and
  baseline dependence;
- on-axis G/K/B calibration establishes the direction-independent receptor
  amplitudes and phases;
- Kcross and a known-EVPA calibrator establish relative R--L delay and phase;
- on-axis D calibration removes the direction-independent leakage assigned to
  calibration rather than the beam;
- fixed reference antennas establish the on-axis beam normalization;
- the declared circular-feed convention establishes correlation packing.

The empirical beam in the `casa_parang_true` state should satisfy

\[
E_a(0,\nu)=I
\]

within the calibration uncertainty. The raw holography product may retain a
different on-axis Jones, but it must use a different calibration-state label.

Do not claim an absolute off-diagonal sign or sky-frame orientation from
3C147 alone. Require the known-EVPA calibrator and a non-zero parallactic-angle
test to close that convention.

## Target architecture

### Raw pointing table

Add a lossless representation of the relevant Measurement Set `POINTING`
columns. A proposed value object is `AntennaPointingTable`, containing:

- pointing sample time and interval;
- antenna identifier;
- `DIRECTION`, `TARGET`, and `POINTING_OFFSET` when present;
- coordinate reference and units for every direction column;
- tracking and on-source state when present;
- original row identifier;
- source Measurement Set and table hashes;
- any interpolation or missing-data flag.

Do not choose between `DIRECTION`, `TARGET`, and `POINTING_OFFSET` in the raw
reader. Preserve all available columns. The geometry audit determines which
column represents the holography raster in this execution.

### Resolved pointing grid

Resolve the raw table onto the visibility operator's sorted unique times as a
separate object. A proposed `ResolvedAntennaPointing` contains:

- `unique_time_s`, shape `(time,)`;
- `antenna_id`, shape `(antenna,)`;
- `offset_lm_rad`, shape `(time, antenna, 2)`;
- `valid`, shape `(time, antenna)`;
- `settled`, shape `(time, antenna)`;
- pointing role, such as reference, moving, transition, or unknown;
- coordinate and interpolation provenance.

This object should remain separate from `VisibilityBlock` initially. Normal
observations should not duplicate a large pointing array in every block. A
holography observation can pair a visibility block with one resolved pointing
object.

### Operator interface

Add an explicit pointing argument to the voltage operator. Configuration
objects should continue to describe execution policy, not carry measurement
data.

The accepted shapes should be:

- no pointing argument: zero offset for all antennas;
- one `(2,)` offset: the existing ordinary-mosaic behaviour;
- `(time, antenna, 2)`: the generalized holography behaviour.

Internally, promote the first two forms to the generalized form. At a unique
time, the evaluator should form coordinates on axes `(antenna, direction)`:

\[
l_{a,d}=l_d-\Delta l_{t,a},\qquad
m_{a,d}=m_d-\Delta m_{t,a}.
\]

The Jones output remains on the existing axes
`(antenna, direction, channel, receptor_out, receptor_in)`.

### JAX execution

The resolved pointing array is ordinary dynamic input. Its values must not be
captured as Python constants. The array shape should remain fixed within a
compiled capacity class.

The JAX operator should:

- gather one antenna-offset plane inside the existing unique-time scan;
- subtract it from each direction tile;
- evaluate each required antenna Jones once per time and direction tile;
- select the Jones for `antenna1` and `antenna2` as it does now;
- mask missing or transition pointing states;
- preserve fixed tile sizes and padded masks;
- avoid recompilation for changes in pointing values or raster position.

The holography point-source path will have one sky direction, so it should be
substantially cheaper than the imaging path. Optimization should follow
correctness measurements rather than precede them.

### Holography observation object

Introduce a thin object that binds:

- calibrated `VisibilityBlock` instances;
- the resolved per-antenna pointings;
- antenna roles;
- source model;
- calibration state;
- selected correlations and spectral windows;
- active and reason-coded masks;
- archive, calibration, and transformation provenance.

This object must refuse mismatched time systems, antenna maps, phase centres,
or calibration states.

## Calibration strategy

### Immutable products

Create named products using the same principle as the 3C391 full-polarisation
preparation:

1. a direction-independent product with antenna position, delay, bandpass,
   and gain calibration only;
2. a full-polarisation product with Kcross, D, X, and the declared
   parallactic-angle handling applied exactly once;
3. an untouched imported Measurement Set retained as the provenance root.

Each product needs a sidecar recording:

- input and output hashes;
- CASA version and commands;
- calibration-table hashes and order;
- source models used by `setjy` or equivalent;
- `calwt` and weight-propagation policy;
- active flag version;
- whether parallactic angle has been applied;
- whether any polarization term has already been applied.

### Protect the beam from self-calibration

Do not solve unrestricted antenna gains on the raster samples. A solver can
otherwise interpret the moving antenna's beam attenuation and phase as a
time-variable direction-independent gain.

Use designated calibration scans, on-axis samples, and the fixed reference
antennas to constrain temporal gains. Any interpolation across the raster must
be declared and validated by masking calibrator intervals. If a gain solution
uses raster data, it must explicitly model the beam in that solve.

### Source structure

Build or import a frequency-appropriate 3C147 coherency model. Test whether a
point model is adequate on the selected moving--reference baselines. If not,
use a compact component model or derive source visibilities from on-axis and
reference--reference data.

Source structure is common to all raster positions but changes with
$(u,v,\nu)$. Beam structure changes with pointing offset. The estimator must
not assign baseline-dependent source visibility to the beam.

### Weights and averaging

Keep native one-second samples and native channels through calibration and
pointing-state assignment. Discard or separately flag samples taken while an
antenna is moving between raster points.

Average only after:

- the pointing dwell has been identified;
- calibration has been applied;
- flags and weights have been propagated;
- phase variation within the bin has been checked;
- the requested comparison frequency has been selected.

The first Memo-comparison product may reproduce the central 28 MHz average of
each spectral window. Native-frequency products remain the evidence source
for spectral continuity and later interpolation.

## Beam estimation products

### Direct prediction before reconstruction

The first scientific use of the holography data should be to predict the
calibrated visibilities using the existing CASSBEAM tables. This tests the
RIME, pointing coordinates, and conventions without allowing an empirical
beam fit to absorb mistakes.

Compare at least:

- analytic Airy diagonal;
- Perley scalar co-polar response;
- CASSBEAM diagonal;
- CASSBEAM experimental full Jones.

Use exactly the same calibrated samples, source model, weights, flags, and
spatial holdouts for every beam.

### Diagonal recovery

Recover the complex co-polar receptor beams before fitting leakage. For each
moving antenna, frequency node, and raster position, combine all valid
moving--reference baselines using calibration- and noise-aware weights.

Produce separate R and L voltage samples. Derive but do not replace them with:

- Stokes-I power response;
- R/L peak displacement and squint;
- radial profiles and half-power widths;
- antenna-to-antenna scatter;
- dense and sparse raster continuity.

The diagonal result is the first empirical artifact that can be compared with
Memo 195 Table 5 and the current CASSBEAM diagonal.

### Full-Jones recovery

After the diagonal and convention gates pass, recover the complex
off-diagonal terms. Keep the native Jones representation rather than storing
only Q/U Mueller leakage.

The recovered artifact should have axes equivalent to:

`(antenna_or_average, frequency, l, m, receptor_out, receptor_in)`.

It should also store:

- sample count and effective weight;
- complex uncertainty or covariance summary;
- validity masks for co-polar and off-diagonal terms;
- dense or sparse raster provenance;
- calibration-state and normalization labels;
- source-model version;
- reference-antenna combination method;
- coordinate and receptor conventions;
- original and processed data hashes.

Do not infer valid leakage outside the sampled raster. A diagonal outer-field
fallback may remain valid while the off-diagonal support mask is false.

### Array-average and antenna-specific products

Retain antenna-specific estimates. Form an array-average reference only after
aligning normalization, pointing centre, complex phase, and receptor
conventions.

Use robust weighting rather than an unqualified arithmetic mean. Record which
antennas were included and why others were excluded. Publish the residual
antenna scatter as model uncertainty rather than hiding it in the mean.

The first imaging backend may use the array average. The data model should
allow antenna-specific evaluation later without another schema change.

## Convention investigation

Treat conventions as a finite model-selection ladder. At minimum test:

- $+l$ versus $-l$;
- $+m$ versus $-m$;
- whether the commanded pointing or source-relative direction carries the
  minus sign;
- raster-axis exchange;
- antenna $p/q$ order;
- visibility conjugation;
- RR/RL/LR/LL packing and R/L exchange;
- feed-frame rotation by $+\chi$ versus $-\chi$;
- placement of the circular parallactic Jones;
- calibrated versus uncalibrated beam normalization.

Do not choose the convention using the same samples used to estimate a beam
map. Fit or normalize on training raster positions and rank conventions on
held-out positions, reference antennas, times, and spectral windows.

A convention is accepted only if it wins consistently in complex visibility
loss and in the expected physical diagnostics. A radial Stokes-I profile is
not sufficient because several wrong conventions preserve radial power.

## Validation protocol

### Split axes

Different holdouts answer different questions. Use named, deterministic
splits rather than one random visibility mask.

1. **Spatial:** hold out complete raster cells, rows, columns, or compact
   spatial tiles. This tests interpolation and beam shape.
2. **Reference antenna:** hold out one or more fixed antennas. This tests
   whether the recovered moving-antenna beam depends on the reference.
3. **Moving antenna:** hold out selected rastering antennas. This tests the
   array-average product.
4. **Time:** hold out complete dwell intervals. This tests gain interpolation
   and temporal stability.
5. **Frequency:** hold out spectral windows or native channel groups. This
   tests beam-frequency interpolation.
6. **Parallactic angle:** hold out angle ranges where coverage permits. This
   tests feed-to-sky rotation.
7. **Correlation:** diagnose RR, LL, RL, and LR separately. Do not select a
   full-Jones model on an aggregate dominated by parallel hands.

Keep a final sealed combination of spatial cells, antennas, and frequency
nodes. Do not use it to choose interpolation order, smoothing, antenna
selection, or conventions.

### Metrics

Report:

- normalized weighted complex visibility loss;
- amplitude and phase residuals separately;
- loss by correlation, antenna, reference antenna, raster region, time,
  parallactic angle, and frequency;
- on-axis Jones normalization error;
- repeatability of the same raster cell across visits;
- disagreement among reference antennas;
- antenna-to-antenna beam scatter;
- co-polar FWHM, null location, and radial profile error;
- R/L peak and centroid separation;
- cross-polar amplitude and phase residuals;
- closure-like residuals on reference--reference and moving--moving
  baselines;
- uncertainty calibration on held-out samples.

An aggregate improvement is not enough for full Jones. RL and LR must improve
without materially degrading RR and LL, and the improvement must transfer
across spatial and antenna holdouts.

### Gate policy

Every report should classify each gate as `pass`, `warn`, `fail`, or
`not_run`.

- `fail` stops a correctness or provenance pipeline.
- `warn` permits exploratory products but marks them non-freezable.
- `not_run` must never be interpreted as a pass.
- the production full-Jones factory requires all mandatory gates to pass.

This allows exploratory work to continue when a scientific tolerance needs
revision while preserving a strict boundary around evidence-grade products.

## Phased implementation

## Phase 0: acquisition and immutable inventory

### Work

- Copy each delivered archive product into a read-only source location.
- Record project, scheduling block, execution block, dates, array
  configuration, source names, intents, bands, SPWs, correlations, antenna
  count, integration time, and table presence.
- Hash the archive product and all later calibration tables.
- Run the general Measurement Set inventory without averaging or modifying
  flags.
- Record whether `POINTING`, `SOURCE`, `FIELD`, `STATE`, `FEED`, `WEATHER`,
  `SYSCAL`, `SYSPOWER`, and `CALDEVICE` are present.
- Preserve the archive's configuration label and separately calculate the
  actual baseline distribution from antenna positions.

### Tests

- Inventory is deterministic across two reads.
- Four correlations are present and their numeric CASA codes are recorded.
- All referenced antenna IDs resolve to names and positions.
- Time and frequency axes are monotonic within each block.
- Hashes and calibration-state sidecars round trip.

### Gate

The product is immutable, fully inventoried, and traceable. Missing pointing
metadata is a blocker for the Measurement Set path. If it is absent, inspect
the SDM-BDF before attempting to infer the raster from visibilities.

## Phase 1: pointing metadata audit

### Work

- Add a read-only `POINTING` table inspector.
- Preserve every available direction column and its measure reference.
- Join pointing samples to antennas and visibility time ranges.
- Cluster stable dwell locations separately for each antenna.
- Identify reference antennas by near-zero motion and moving antennas by
  raster coverage;
- mark slew or unsettled intervals without deleting them;
- compare reconstructed dense and sparse grids with Memo 195.

### Tests

- A synthetic POINTING table round trips through the reader.
- Direction measure references and units cannot be omitted.
- Joining respects `INTERVAL` and rejects extrapolation beyond tolerance.
- Missing antenna-time samples remain invalid rather than becoming zero
  offsets.
- Exact spherical conversion agrees with known small-offset cases.
- Reversing the sign or exchanging axes produces the expected diagnostic
  failure.

### Gate

The data reveal the expected two antenna populations and the expected raster
dimensions and spacings. The selected direction column and sign are recorded
with evidence. Samples taken during motion have a separate reason code.

## Phase 2: generalized pointing representation

### Work

- Implement `AntennaPointingTable` and `ResolvedAntennaPointing` or equivalent
  immutable types.
- Resolve onto exact unique visibility times.
- Add validity, settled-state, and antenna-role masks.
- Define serialization and schema versions.
- Keep pointing data outside `BeamOperatorConfig`.

### Tests

- Serialization is lossless and versioned.
- Reordered antenna IDs and non-contiguous IDs resolve correctly.
- Duplicate, missing, and overlapping pointing intervals fail or receive
  explicit reason codes.
- A common pointing offset promotes exactly to the old shared-offset form.
- Zero offsets reproduce the no-pointing representation bit for bit where
  possible.

### Gate

One resolved object covers every active holography visibility row and antenna.
No active sample receives an assumed zero offset because metadata was absent.

## Phase 3: NumPy holography RIME

### Work

- Extend the reference voltage operator to accept `(time, antenna, 2)`
  pointing offsets.
- Evaluate source-relative coordinates separately for each antenna.
- Preserve phase-centre Fourier coordinates.
- Add a compact-source coherency visibility input.
- Return pointing-validity and beam-validity masks separately.
- Provide a dedicated holography wrapper that selects moving--reference
  baselines without changing the underlying RIME.

### Tests

- Manufactured scalar and full-Jones beams match hand-calculated
  $E_p S E_q^{\rm H}$ examples.
- Swapping antennas gives the expected Hermitian/conjugate result.
- A moving $p$ and fixed $q$ differ from fixed $p$ and moving $q$ in
  the expected way.
- Common offsets match the existing ordinary-mosaic operator.
- Source-at-phase-centre tests prove that pointing changes do not introduce a
  spurious Fourier phase.
- A resolved source coherency model is distinguished from a point source.
- Invalid pointing for either baseline antenna masks the visibility.

### Gate

The reference operator passes the algebraic, Hermitian, coordinate-sign, and
ordinary-mosaic regression tests at floating-point tolerance.

## Phase 4: JAX operator parity

### Work

- Add the generalized pointing input to the streamed JAX forward operator.
- Add it to the explicit tiled adjoint if beam or source parameters will be
  fitted with gradients.
- Keep pointing arrays dynamic and capacity shapes static.
- Preserve all correlation and support masks.

### Tests

- NumPy and JAX predictions agree for scalar, diagonal, and full-Jones beams.
- Tests cover unequal $p/q$ offsets, non-zero parallactic angle, all four
  correlations, flags, padding, and missing pointing samples.
- Explicit adjoint and VJP agree for real fitted parameters.
- Finite differences verify any beam-parameter gradient.
- Results are invariant across time and direction tile sizes.
- Compilation count does not increase when only pointing values change.

### Gate

Use the established numerical tolerances for the existing voltage operator:
approximately $10^{-10}$ for NumPy algebra where conditioning permits and
$10^{-7}$ relative for JAX execution. Record the actual error distribution;
do not loosen a failed test merely to meet these guide values.

## Phase 5: direction-independent calibration products

### Work

- Identify flux, bandpass, gain, leakage, and EVPA calibrator scans from the
  intents and source table.
- Set the appropriate calibrator models explicitly.
- Produce G/K/B-only and full-polarisation Measurement Sets.
- Keep the raster at native resolution through calibration.
- Exclude raster attenuation from unconstrained gain solving.
- Build a source coherency model for 3C147 and test its baseline dependence.
- Generate compact calibration goldens for JAX application.

### Tests

- Applying the CASA tables in JAX matches selected CASA corrected
  visibilities.
- The G/K/B-only and full-polarisation products differ where expected.
- A second polarisation application is refused.
- Calibrator apply-back residuals are reported on samples independent of the
  solve where possible.
- Gain interpolation is tested by masking complete calibration intervals.
- Reference--reference baselines remain stable through raster time.

### Gate

Calibration order, source models, flags, weights, and parallactic handling are
fully recorded. Cross-hand calibrator tests establish the on-axis receptor
convention. Remaining calibration uncertainty is propagated into later beam
comparisons.

## Phase 6: direct beam and convention comparison

### Work

- Predict the calibrated raster data with Airy, Perley, CASSBEAM diagonal,
  and CASSBEAM full Jones.
- Enumerate the convention ladder rather than editing signs ad hoc.
- Use training raster cells only for normalization.
- Score every candidate on identical held-out samples.
- Report dense central and sparse outer grids separately.

### Tests

- On-axis samples recover the expected calibration-state normalization.
- Known manufactured convention errors are detected by the ladder.
- Reference-antenna holdout gives compatible beam rankings.
- Results remain stable when individual antennas and spectral windows are
  removed.
- RR/LL and RL/LR results are reported separately.

### Gate

One coordinate and receptor convention wins consistently across the declared
holdouts. Its paired improvement has an uncertainty interval that excludes no
change in the decisive strata. Physical diagnostics, including squint
direction and cross-hand phase rotation, agree with the same convention.

If no convention wins, preserve the report and continue only with diagnostic
labels. Do not freeze full Jones.

## Phase 7: empirical diagonal beam

### Work

- Recover complex R and L co-polar beams for each suitable moving antenna.
- Combine reference antennas robustly.
- Estimate pointing-centre offsets without absorbing them into the raster
  coordinate definition.
- Compare power profiles with Memo 195 Table 5.
- Compare CASSBEAM diagonal predictions with the measured samples.
- Quantify antenna scatter and frequency dependence.

### Tests

- Same-cell estimates from different reference antennas agree within their
  uncertainty.
- Withheld raster cells are predicted by the selected spatial interpolator.
- Withheld frequencies are predicted by the selected frequency interpolator.
- Dense and sparse products agree in their overlap.
- R/L squint is repeatable across antennas and frequency.

### Gate

The diagonal artifact predicts held-out complex visibilities better than the
declared scalar baselines in the regions where receptor structure matters. It
has valid support, uncertainty, and normalization masks. Failure is a warning
for exploration but blocks empirical diagonal production status.

## Phase 8: empirical full-Jones beam

### Work

- Recover off-diagonal Jones samples after fixing the diagonal solution.
- Test joint refinement only after the staged solution is stable.
- Preserve independent RL and LR information; do not impose conjugacy unless
  the data support it.
- Estimate leakage uncertainty and support separately from co-polar support.
- Form antenna-specific and robust array-average products.

### Tests

- Cross-hand improvements transfer to held-out raster cells, antennas,
  frequencies, and parallactic angles.
- Injected off-diagonal patterns are recovered with correct sign and phase.
- An injected source-polarization term is not misidentified as beam leakage.
- Setting off-diagonal terms to zero reproduces the accepted diagonal result.
- RR/LL performance does not materially regress when RL/LR improves.
- Off-diagonal support ends at the sampled and validated boundary.

### Gate

The full-Jones artifact produces a repeatable held-out RL/LR improvement over
the diagonal artifact, retains acceptable parallel-hand performance, and
passes the EVPA/parallactic-angle convention gate. Otherwise publish the
diagonal product and retain full Jones as experimental.

## Phase 9: packaging and freeze

### Work

- Package the selected beam arrays and manifest in a deterministic,
  non-executable format such as NPZ plus JSON.
- Record hashes, units, axes, receptor order, coordinate frame,
  calibration state, normalization, interpolation, support, uncertainty,
  antenna membership, and source data lineage.
- Implement an empirical holography backend for `VoltageBeamModel`.
- Keep array-average and antenna-specific modes explicit.
- Add a freeze pin separate from the existing CASSBEAM pin.

### Tests

- Artifact loading is deterministic and hash checked.
- Axis order and complex dtype cannot be guessed.
- Exact nodes reproduce the stored samples.
- Interpolation respects validity and never extrapolates silently.
- Unknown calibration states and frequencies fail closed.
- Diagonal-only use clears off-diagonal support explicitly.
- The production factory refuses an unfrozen manifest.

### Gate

All mandatory software and scientific gates are present and passing. The
manifest identifies the exact lower/upper C-band source executions and all
processing code. Only then may the production `full_jones` factory return the
empirical artifact without an experimental override.

## Phase 10: transfer to imaging

### Work

- First predict the frozen 3C391 full-correlation holdout with no sky refit.
- Then refit Stokes I under the accepted diagonal empirical beam with a
  convergence-based stopping rule.
- Fit global Q/U with V fixed to zero as a calibration and convention check.
- Compare diagonal and full Jones on RL/LR while keeping the I ancestor fixed.
- Open spatial polarisation only after the global and calibrator gates pass.
- Repeat the protocol on 24A-063 or another independent C-band observation.

### Tests

- The empirical diagonal beam gives sensible Stokes-I transfer by pointing,
  hand, frequency, and parallactic angle.
- Full Jones improves held-out cross hands where holography predicts leakage.
- 3C286 or 3C138 retains its known Q/U and approximately zero V.
- An unpolarised calibrator does not acquire structured celestial Q/U/V.
- Results survive the independent-observation transfer without retuning beam
  conventions.

### Gate

The beam is scientifically useful only if it transfers beyond its own raster.
The final report must distinguish:

- correct holography reconstruction;
- better prediction of an unrelated calibrator;
- better prediction of a science target;
- evidence for additional sky polarisation freedom.

These are separate claims and should not share one pass flag.

## Required test fixtures

Build small committed fixtures rather than depending on the 100 GB source MS
in routine tests.

### Synthetic fixture

Include:

- three antennas: one fixed reference and two moving antennas;
- several unique times and raster offsets;
- a known non-diagonal Jones beam;
- RR, RL, LR, and LL;
- a compact source with optional resolved visibility structure;
- missing and transition pointing samples;
- known parallactic-angle variation.

This fixture proves algebra and recovery.

### Metadata fixture

Extract a minimal, redistributable representation of the real POINTING,
FIELD, STATE, ANTENNA, and time joins. It need not contain real
visibilities. It proves that future reader changes preserve the raster
geometry.

### Real visibility golden

After calibration, commit or checksum a small selection containing:

- on-axis and off-axis samples;
- more than one moving and reference antenna;
- dense and sparse raster positions;
- at least two frequencies;
- non-zero parallactic-angle separation;
- all four correlations.

Store the expected CASA-calibrated values and the selected pointing
coordinates. This fixture is the convention and no-double-apply gate.

## Failure modes and responses

### POINTING metadata is absent or incomplete

Inspect the original SDM-BDF and its pointing tables. Do not infer the full
raster solely from amplitudes. A visibility-derived pointing solution would
be scientifically circular because the assumed beam would determine the
coordinates used to validate that beam.

### Direction columns disagree

Preserve each candidate interpretation and compare its raster geometry with
the Memo and its held-out complex visibility prediction. Do not overwrite one
column with another during ingestion.

### Source structure dominates residuals

Restrict the initial solve to baselines where the point approximation passes,
or introduce a fixed 3C147 component model derived independently. Do not give
the beam arbitrary baseline dependence.

### Gain drift resembles the raster

Use reference--reference baselines and calibration scans to model temporal
gain. Hold out complete time intervals. Do not self-calibrate each raster
position independently.

### Reference antennas disagree

Report per-reference results. Investigate calibration, pointing, and on-axis
normalization before robust averaging. Excluding an antenna requires a reason
and an unchanged sealed evaluation.

### Cross hands remain inconsistent

Keep the diagonal artifact. Revisit Kcross, D, X, EVPA, source polarization,
and feed-frame rotation. Do not smooth RL/LR until a convention error
disappears.

### CASSBEAM and holography disagree

Localize the disagreement by antenna, frequency, direction, correlation, and
parallactic angle. The empirical data need not match a nominal physical dish,
but neither model should be declared wrong from an aggregate loss alone.

## Deliverables

The completed work should produce:

- an immutable archive inventory for each requested observation;
- a pointing audit report and plots of antenna raster tracks;
- a versioned per-antenna pointing representation;
- NumPy and JAX generalized pointing RIME paths;
- direction-independent and full-polarisation calibrated MS products;
- a 3C147 source-model artifact;
- a convention-ladder report;
- per-antenna diagonal and full-Jones beam estimates;
- an array-average beam with uncertainty and support;
- a Memo 195 and CASSBEAM comparison report;
- committed synthetic, metadata, and real-data golden fixtures;
- a frozen empirical-beam manifest, if all gates pass;
- 3C391 and independent-observation transfer reports.

## Immediate next work after delivery

When the first Measurement Set arrives:

1. run the immutable inventory;
2. inspect `POINTING` before running calibration;
3. reconstruct antenna tracks and settled raster cells;
4. identify reference and moving antennas from motion, then compare them with
   the Memo list;
5. verify the four correlations and calibrator scans;
6. make a tiny metadata fixture;
7. implement the generalized pointing types;
8. prove the NumPy RIME on synthetic data;
9. only then begin the expensive calibration and full visibility reduction.

This order gives an early answer to the main acquisition risk: whether the
prepared Measurement Set preserved enough per-antenna pointing information to
reconstruct the holography experiment correctly.

## Relationship to existing plans

This plan extends
[`vla-beam-model-proposal.md`](vla-beam-model-proposal.md) from a generated
array-average beam to an empirical per-antenna reference. It supplies the
holography route listed but not frozen in
[`vla-beam-reference-inventory.md`](vla-beam-reference-inventory.md).

The resulting beam remains inside the finite-component integral described by
[`beam-aware-pixel-proposal.md`](beam-aware-pixel-proposal.md). Holography does
not remove the need for beam-aware pixel integration in science imaging.

On-axis Kcross/D/X calibration follows
[`jones-polarization-calibration.md`](jones-polarization-calibration.md).
Holography measures the remaining direction-dependent Jones response; it must
not absorb the direction-independent calibration terms a second time.

The empirical beam becomes an imaging candidate only at Phase 10. Until then,
the current production imaging path and its full-Jones freeze refusal remain
unchanged.
