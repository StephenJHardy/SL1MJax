# Calibration prior learning proposal

## Purpose

SL1MJax needs a calibration prior that reflects how VLA gains vary in real
observations. The prior should let a target-time calibration model move away
from the transferred calibrator solution when the data support a move, while
making rapid, large, or poorly anchored changes improbable.

The calibration library can supply this prior. Repeated calibrator scans and
extended calibrator observations let us hide sections of known calibration
data, predict them from the remaining anchors, and measure which time models
give accurate and honest missing-data predictions.

This is not ordinary smoothing. The target product is a versioned distribution
over plausible calibration behaviour, including uncertainty and an
applicability domain. The runtime self-calibration solver will condition that
prior on the current observation's calibrator anchors. It will not freely tune
the prior on the target data it is meant to constrain.

The proposal initially covers time-dependent diagonal gain $G$. Frequency
dependence, differential R/L terms, leakage, and cross-hand phase follow only
after the diagonal benchmark is established.

## Decision

Learn calibration-prior hyperparameters with blocked prediction experiments on
a library of calibrator observations.

Use two validation levels:

1. a cheap solution-table benchmark for inventory, kernel screening, and
   development;
2. a definitive visibility-level benchmark in which withheld calibrator data
   is excluded from the gain solve, predicted from the retained anchors, and
   scored against its raw visibilities.

Fit a hierarchical distribution of hyperparameters across runs rather than one
universal point estimate. The first production prior applies to
antenna-relative, receptor-common log-amplitude and phase. Keep the
array-common amplitude, differential R/L amplitude, differential R/L phase,
D, and X externally anchored.

At target runtime, either fix the hyperparameters to the selected library
profile or marginalise over its learned hyperprior. Do not maximize them freely
on the target.

## Goals

The learned prior should:

- predict missing calibration intervals with calibrated uncertainty;
- distinguish ordinary interpolation from extrapolation across a long or
  invalid gap;
- express realistic amplitudes and timescales for antenna gains;
- avoid learning solver noise as instrumental variability;
- remain stable when one calibrator has intrinsic variability;
- give the self-calibration objective a measurable preference for small,
  slowly varying corrections;
- expose its training corpus, gauges, kernel, and applicability domain;
- compete against nearest and native linear transfer on held-out data.

The prior is not intended to:

- reconstruct measurements that were never recorded;
- repair a known missing receiver or arbitrary corrupted data;
- make a single unresolved calibrator identify its own flux variability and
  the array's common amplitude variation simultaneously;
- turn a smooth GP into a generic model for jumps or discrete hardware states;
- replace held-out validation of a target self-calibration update.

## Fundamental identifiability limit

For an unresolved calibrator with flux $S(t)$, a simplified parallel-hand
measurement is

\[
V_{pq}(t)=g_p(t)g_q^*(t)S(t).
\]

Multiplying every antenna gain by a common real amplitude $a(t)$ and replacing
$S(t)$ by $S(t)/|a(t)|^2$ leaves every visibility unchanged. A single point
calibrator therefore cannot distinguish its intrinsic or scintillating flux
from an array-common amplitude variation.

A GP prior does not remove this degeneracy. It only states which explanation
was more common in its training corpus. If the corpus silently forced every
calibrator to be constant, the learned common-amplitude prior would absorb
real calibrator variability and later encourage target self-calibration to do
the same.

The safe response is structural:

- learn antenna-relative gain behaviour separately from array-common
  amplitude;
- preserve a source-variability term when learning from calibrators;
- use multiple calibrators, stable field sources, switched power, or external
  flux information to constrain common amplitude;
- keep target-wide flux variability as an explicit sky alternative;
- state when the data cannot distinguish the alternatives.

The same principle applies to polarization. Differential R/L amplitude is
degenerate with Stokes V, and differential R/L phase rotates Q/U. Those modes
require their V=0 and EVPA anchors regardless of how much diagonal gain data is
available.

## Gain coordinates

### Receptor-common and receptor-differential modes

For antenna $a$, write residual log-amplitude relative to the transferred
calibrator solution as

\[
\delta A_{c,a}=\tfrac12(\delta A_{a,R}+\delta A_{a,L}),
\qquad
\delta A_{d,a}=\tfrac12(\delta A_{a,R}-\delta A_{a,L}),
\]

with the same decomposition for phase.

The first learned GP covers the common-receptor modes
$\delta A_{c,a}$ and $\delta\phi_{c,a}$. Differential modes remain tightly
anchored because they affect V and EVPA.

### Array-common and antenna-relative modes

Receptor-common amplitude needs another decomposition:

\[
\delta A_{c,a}(t)=C_A(t)+\widetilde{\delta A}_{c,a}(t),
\qquad
\sum_a w_a\widetilde{\delta A}_{c,a}(t)=0.
\]

$C_A(t)$ is the array-common amplitude and is degenerate with source-wide
flux. The zero-mean antenna-relative term
$\widetilde{\delta A}_{c,a}$ is identifiable from baseline differences and is
the safe first self-calibration mode.

An array-common phase cancels from interferometric baselines and is a gauge.
Remove it through the reference-antenna or zero-mean phase convention before
learning phase hyperparameters. Preserve the chosen gauge in the prior
artifact.

These two decompositions are independent. “Common R/L” protects polarization,
while “antenna-relative” protects source-wide total-intensity variability. Both
are required.

## Prior family

Write the runtime gain as

\[
g_{a,r}(t)=g^{\mathrm{cal}}_{a,r}(t)
\exp\!\left[\delta A_{a,r}(t)+i\,\delta\phi_{a,r}(t)\right].
\]

The library prior applies to the residual coordinates after deterministic
terms, gauges, and calibrator transfer have been removed.

### Initial kernel

Start with log-amplitude and locally unwrapped phase using a Matérn-3/2
kernel plus heteroscedastic measurement noise:

\[
k(\Delta t)
=
\sigma_g^2
\left(1+\frac{\sqrt{3}|\Delta t|}{\ell_g}\right)
\exp\!\left(-\frac{\sqrt{3}|\Delta t|}{\ell_g}\right)
+\sigma_n^2\,\mathbf 1_{\Delta t=0}.
\]

Here $\sigma_g$ is the typical residual gain amplitude, $\ell_g$ is its
correlation time, and $\sigma_n$ describes solution noise not already carried
by per-solution uncertainties. Matérn-(3/2) permits rougher behaviour than an
RBF kernel and is a safer initial physical assumption for atmospheric and
instrumental variation.

Candidate comparisons should include:

- nearest valid transfer;
- native piecewise-linear transfer;
- an Ornstein--Uhlenbeck or Matérn-1/2 process;
- Matérn-3/2;
- the existing circular RBF smoother as an engineering reference;
- an optional slow-plus-fast two-component kernel after one-component models
  are understood.

Do not add several kernel components merely because they improve training
likelihood. Every component must improve outer held-run prediction and retain
calibrated uncertainty.

### Phase representation

The current
[`circular_gp_gain_solution`](../src/sl1mjax/gain_time_models.py) models unit
phasors with independent RBF GPs and projects the posterior mean back to the
unit circle. This is useful engineering groundwork, but it is a post-hoc
smoother and does not provide the posterior covariance needed here.

For well-sampled runs, gauge-align and unwrap phase locally, then reject an
unwrap when adjacent supported points imply an ambiguous branch. For more
difficult runs, use a wrapped-normal or phasor likelihood with uncertainty
rather than treating the projected phasor mean as a Gaussian posterior.

Phase jumps need a separate model. A smooth GP must not respond to an ea05-like
two-state process by learning a very short length scale for all observations.

### Hierarchical hyperprior

For run $j$, an initial hierarchical model can use

\[
\log \sigma_j\sim\mathcal N(\mu_\sigma,\tau_\sigma^2),
\qquad
\log \ell_j\sim\mathcal N(\mu_\ell,\tau_\ell^2).
\]

The population parameters describe a band and observing regime. Individual
runs may differ, but sparse runs shrink towards the population rather than
choosing extreme hyperparameters.

The first version can estimate this hierarchy with empirical Bayes. A later
version may retain posterior samples. Store the spread as well as the central
value; one fixed length scale hides real run-to-run variation and produces
overconfident target predictions.

## Calibration-library requirements

Three calibrator visits support a basic interpolation check, but they do not
identify a useful correlation time. Masking one visit leaves only two anchors.

A strong time-prior run should preferably provide:

- at least 6--10 valid calibrator visits;
- a span several times longer than the candidate correlation time;
- several different time separations;
- native flags, weights, and solution uncertainties;
- raw calibrator visibilities and the exact source model;
- the fixed K/B and prior-calibration chain used during the G solve;
- antenna, receptor, band, frequency, elevation, and scan provenance;
- a trustworthy record of missing antennas and hard exclusions.

Extended calibrator scans and repeated visits answer different questions.
Extended scans constrain short-timescale gain and solution-noise behaviour.
Interleaved visits constrain prediction across target-like gaps. The library
needs both.

Repeated observations of one calibrator are useful, but a diverse stable
calibrator ensemble is safer. Multiple calibrators in the same scheduling
block are especially valuable because an instrument process can be shared
while each source has an independent variability term.

Profiles will probably need to differ by frequency band. They may later depend
on cadence, elevation range, weather regime, switched-power availability, or
instrument epoch. Do not fragment the corpus into many small profiles before
outer validation shows a real benefit.

## Library inventory

Extend the existing BagOfWinds calibration inventory with fields needed for
prior learning:

- observation and scheduling-block identity;
- calibrator field and known source-stability information;
- frequency band, spectral windows, and receptors;
- number of usable G scans and time span;
- scan cadence and gap distribution;
- solution interval and per-solution uncertainty or SNR;
- valid antenna/receptor counts by scan;
- raw visibility availability;
- K/B/G and prior-table provenance;
- weather, elevation, opacity, and switched-power availability;
- pipeline and calibrator-model versions.

Assign initial roles:

- `training_candidate`: meets the basic cadence and provenance requirements;
- `short_timescale_only`: extended scan but too few separated visits;
- `gap_prediction_only`: repeated scans but insufficient native within-scan
  resolution;
- `source_variability_risk`: known or suspected intrinsic variability;
- `engineering_only`: incomplete provenance or no raw visibility;
- `sealed_evaluation`: excluded from hyperparameter selection.

The current SRDP tables are still useful for diagonal G population statistics.
More raw calibrator observations will probably be required for definitive
visibility-level validation and for separating source variability from common
amplitude.

## Preparing gain solutions

Library gain tables cannot be concatenated without preparation. Before fitting
a population prior:

1. apply a consistent reference-antenna or zero-mean phase gauge;
2. separate receptor-common and receptor-differential coordinates;
3. remove the array-common log-amplitude before fitting the first prior;
4. retain the removed common amplitude as a separate source/instrument
   diagnostic;
5. express amplitude in log space;
6. preserve invalid knots rather than deleting them from the time coordinate;
7. carry per-solution uncertainty and effective sample count;
8. remove or model known deterministic elevation, opacity, gain-curve, or
   switched-power effects;
9. label discontinuities and known hardware state changes;
10. keep the original table and transformation provenance.

An invalid knot defines a gap. Filtering it out and interpolating freely between
the remaining first and last times would mislabel extrapolation as ordinary
support. This is the same coverage rule adopted in
[`calibration-flag-proposal.md`](calibration-flag-proposal.md).

## Blocked prediction benchmark

### Mask families

Random point holdout is too easy and does not represent the intended use. Use
structured masks:

- one complete calibrator scan;
- two or more adjacent scans;
- one antenna missing from one scan;
- all receptors for one antenna over a bounded interval;
- an initial or final block, testing extrapolation;
- gaps matched to the empirical target/calibrator cadence;
- a range of gap lengths in seconds and multiples of the median cadence;
- later, contiguous frequency blocks for B, D, and X priors.

Keep hard-invalid data outside both training and evaluation. A masked sample
must have trustworthy raw data and an independently usable reference solution
or visibility model.

### Solution-table screening level

For each mask:

1. remove the held solution coordinates from the candidate model;
2. condition on the remaining gain solutions and their uncertainty;
3. predict the held log-amplitude and phase;
4. score predictive density, point error, and interval coverage;
5. record performance against gap length and anchor geometry.

This level is fast and can screen kernels across the full table corpus. It is
not final evidence because the held gain estimates contain solver choices,
noise, flags, and gauges.

### Visibility-level acceptance level

For the definitive benchmark:

1. exclude the held calibrator rows from the G solve itself;
2. solve retained anchors with a fixed, recorded K/B and source model;
3. condition each candidate prior on those anchors;
4. predict gains and uncertainty at the held rows;
5. apply the complete Jones chain to the raw held visibilities;
6. score the held visibility likelihood and closure;
7. compare with nearest and native linear transfer.

If a time-independent K or B table used the held scan, state that dependency.
The strict benchmark should rebuild fixed terms without the held rows when
their reuse could leak material information into the G test.

### Nested selection

Use nested validation:

- inner blocked masks tune kernel family and hyperparameters within training
  runs;
- complete outer runs evaluate generalisation to a new observation;
- one or more sealed datasets remain unopened until the profile and acceptance
  thresholds are fixed.

Do not report inner cross-validation as corpus-level evidence. Runs from the
same scheduling block or near-duplicate pipeline products belong in the same
outer partition.

## Metrics

Point prediction error is necessary but insufficient. Score:

- held-visibility negative log predictive density;
- normalized complex residual power;
- closure phase and closure amplitude where applicable;
- log-amplitude and phase prediction error;
- empirical coverage of 68% and 95% predictive intervals;
- standardized residual distribution;
- calibration support as a function of gap duration;
- performance by antenna, receptor, band, elevation, and anchor distance;
- failure rate and numerical stability;
- comparison with unchanged linear and nearest transfer.

A slightly worse posterior mean with honest uncertainty is safer than a lower
MSE model whose intervals are too narrow. The prior will control whether target
data can move calibration parameters, so overconfidence and underconfidence
both have scientific consequences.

Also record a flexibility diagnostic: draw functions from the learned prior at
the target cadence and measure the amplitude of sky-like modes they could
absorb. This turns “the prior seems smooth” into a quantitative statement.

## Modelling calibrator source variability

### First safe scope

The first hyperprior fit uses only antenna-relative gain coordinates. A common
calibrator flux change cancels from these coordinates after the array-common
amplitude is removed. This provides useful instrumental information without
claiming to identify the source light curve.

### Extended source/instrument model

When the library contains multiple calibrators or external amplitude
information, introduce

\[
\log S_c(t)=\log S_{c,0}+s_c(t),
\]

where $s_c(t)$ is a source-specific process. Instrumental latent processes
may be shared across calibrators, while each $s_c$ is independent unless
physical information says otherwise.

Useful constraints include:

- multiple calibrators interleaved through one run;
- stable compact sources in the calibrator field;
- switched-power or noise-diode measurements;
- external monitoring of known variable calibrators;
- repeat epochs that separate source-specific from antenna-specific patterns.

For a single unresolved calibrator with no external information, common
amplitude remains unidentified. The code should report that fact rather than
assigning the variation to the instrument by convention and later calling it
evidence.

### Consequence for target variability

If a target's true variability lies in exactly the same measurement-equation
subspace as a permitted gain mode, data from that target alone cannot separate
them. Examples include source-wide flux variation versus array-common
amplitude.

Protection requires one or more of:

- freeze the degenerate gain mode;
- include a competing temporal sky model;
- use other sources or mosaic pointings;
- use external instrument measurements;
- compare on partitions that distinguish antenna-based and spatial responses.

The learned prior lowers the chance of absorption. It does not create
identifiability where none exists.

## Versioned prior artifact

Serialize the selected population prior separately from any one calibration
solution. A possible shape is:

```python
@dataclass(frozen=True)
class CalibrationPriorProfile:
    schema_version: int
    profile_id: str
    calibration_term: str
    coordinates: tuple[str, ...]
    band: str
    receptor_basis: str
    gauge: dict[str, object]
    kernel_family: str
    hyperprior: dict[str, object]
    noise_model: dict[str, object]
    jump_model: dict[str, object] | None
    applicability: dict[str, object]
    training_corpus: tuple[dict[str, object], ...]
    validation_summary: dict[str, object]
    provenance: dict[str, object]
```

The artifact must record:

- physical units and coordinate conventions;
- amplitude and phase transformations;
- array and receptor decompositions;
- kernel equation and implementation version;
- hyperparameter central values and population spread;
- measurement-noise treatment;
- gap-length limits for ordinary support;
- training, outer-validation, and sealed-run identities;
- corpus hashes and source-model versions;
- performance and uncertainty coverage by regime;
- known exclusions and failure cases.

Unknown schema fields or coordinate conventions must fail closed. A prior
trained at one band must not be silently applied at another.

## Runtime use

### Selecting a profile

Choose the most specific validated profile whose applicability domain contains
the observation. Initially this will probably be a band-level diagonal-G
profile. If no profile applies, fall back to conservative native transfer or a
broad explicitly labelled engineering prior.

Do not extrapolate between profiles by name alone. Report why a profile was
selected and how the observation differs from the training distribution in
cadence, duration, elevation, antenna count, and anchor distance.

### Conditioning on current calibrators

The runtime model is

\[
J_a(t,\nu)=\Delta G_a(t)J^{\mathrm{cal}}_a(t,\nu),
\]

with $\Delta G$ at the gain position in the Jones chain. Condition the
library prior on the current observation's calibrator anchors and their
uncertainty. Target-time latent points may then move inside that conditional
distribution.

Use one anchor formulation:

- analytically condition the GP on calibrator anchors and use the resulting
  conditional prior in the target objective; or
- retain an unconditional prior and include an explicit calibrator-anchor
  likelihood.

Do not condition and then add the same anchor likelihood again.

### Hyperparameters at target time

Preferred order:

1. use fixed profile hyperparameters;
2. marginalise over the stored population hyperprior;
3. allow current calibrator scans, but not target data, to shrink a run-specific
   hyperparameter posterior;
4. consider target-informed hyperparameter changes only as a separately
   validated model expansion.

Freely maximizing length scale and variance on the target defeats the purpose
of the prior. A bright target could select precisely the calibration freedom
needed to absorb its real variability.

### Calibration support and flags

The GP returns a mean, uncertainty, anchor distance, and support class. Feed
these into
[`calibration-flag-proposal.md`](calibration-flag-proposal.md):

- inside ordinary validated gaps: `supported`;
- beyond the normal gap limit but with a prediction: `extrapolated`;
- constrained mainly by the population profile: `prior_only`;
- outside the applicability domain or uncertainty limit: `unsupported`.

A posterior mean alone must not turn a CASA-flagged calibration gap into an
active visibility.

## 3C391 role

3C391 supplies a useful engineering benchmark but not fresh evidence for the
kernel choices already explored on it.

The established result is:

- fourteen native J1822−0938 gain epochs nearly reproduce CASA;
- native linear transfer scores 0.007478 on the selection fold;
- the best tested circular RBF GP scores 0.007637, 2.13% worse;
- every tested post-hoc smoother loses to native linear transfer.

This result must remain a regression. The learned-prior system should reproduce
the loss of post-hoc smoothing. It should not be tuned until the GP wins on the
same already-opened folds.

3C391 can then test a different question: mask one or more trustworthy
J1822−0938 scans, predict them from the other anchors, and measure predictive
uncertainty on held raw calibrator visibilities. It can also provide the first
controlled recovery experiment for target data that CASA excluded because a
bracketing gain solution was missing.

Use another observation or outer library partition for the scientific claim
that the learned profile generalises.

## Build plan

### Phase 1: prior-learning inventory

1. Extend the calibration corpus inventory with cadence, span, uncertainty,
   raw-data availability, source-risk, and profile metadata.
2. Identify runs with at least 6 valid gain visits and extended calibrator
   scans.
3. Group near-duplicate products into one outer-validation unit.
4. Declare engineering, training, outer-validation, and sealed roles.
5. Emit a versioned JSON inventory and summary report.

Do not fit a population kernel until the inventory shows how many genuinely
independent runs and calibrators are available. Acquire more calibrator data if
the outer partitions would otherwise contain only one or two runs.

### Phase 2: gain-coordinate preparation

1. Implement gauge alignment and log-amplitude extraction.
2. Separate receptor-common/differential and array-common/antenna-relative
   modes.
3. Preserve invalid knots and per-solution uncertainty.
4. Add source/common-amplitude diagnostics.
5. Round-trip the transformed coordinates with synthetic and real tables.

This phase must not alter the production calibration apply path.

### Phase 3: solution-table blocked benchmark

1. Implement deterministic contiguous mask families.
2. Compare nearest, linear, OU, Matérn-(3/2), and circular RBF candidates.
3. Score predictive density, error, and coverage by gap length.
4. Use nested leave-run-out partitions.
5. Select the smallest kernel family that generalises.

This phase screens candidates and exposes data problems. It does not yet
authorize target self-calibration.

### Phase 4: visibility-level benchmark

1. Re-solve G after removing each held calibrator cohort.
2. Predict held gains from retained anchors.
3. Apply the complete chain to raw held visibilities.
4. Score likelihood, residual power, and closure.
5. Fit the hierarchical population distribution from training runs only.
6. Open outer and sealed runs according to the declared protocol.

Visibility-level performance and uncertainty coverage determine the production
profile.

### Phase 5: versioned prior and interpolation support

1. Serialize `CalibrationPriorProfile` with corpus and validation provenance.
2. Add profile selection and applicability checks.
3. Return posterior mean, covariance representation, anchor distance, and
   support class.
4. Integrate support with the calibration quality policy.
5. Keep the current native transfer as the default control.

### Phase 6: anchored target self-calibration

1. Enable only antenna-relative, receptor-common target-time corrections.
2. Freeze array-common amplitude, differential R/L, Kcross, D, and X.
3. Hold the accepted spatial sky topology fixed during a calibration update.
4. Compare unchanged transfer, native/nearest alternatives, and the learned
   prior on independent data.
5. Reject updates that improve training without improving held baselines,
   times, and pointings.

The first self-cal experiment should use a static target or a sky model whose
time dependence has already been bounded. It should not use an unresolved
candidate scintillator as the sole constraint.

### Phase 7: later extensions

Only after diagonal time-G passes, consider:

- frequency-dependent bandpass priors with held channel blocks;
- common atmospheric latent processes across antennas;
- weather, elevation, opacity, or switched-power conditional means;
- differential R/L hyperpriors anchored by V=0 and EVPA calibrators;
- smooth leakage-frequency priors;
- explicit source-variability populations.

Each extension needs its own identifiability statement and held-data benchmark.

## Test plan

### Unit tests

Test:

- gain gauge transformations and inverse transformations;
- receptor-common/differential decomposition;
- array-common/antenna-relative decomposition and zero-mean constraint;
- log-amplitude and phase coordinate handling;
- preservation of invalid knots;
- blocked-mask determinism and non-overlap;
- profile serialization and unknown-schema failure;
- applicability-domain selection and conservative fallback;
- support-class changes with gap and anchor distance;
- no target data entering calibrator-only hyperparameter adaptation.

### Synthetic statistical tests

Generate known gain processes and require:

- recovery of population length-scale and variance distributions within
  declared tolerances;
- 68% and 95% interval coverage near their nominal rates;
- increasing uncertainty with gap duration and at run boundaries;
- no bias in antenna-relative hyperparameters when the point calibrator has a
  common scintillating flux term;
- explicit non-identifiability of source flux and array-common amplitude when
  no external constraint exists;
- correct separation when a second calibrator or external amplitude anchor is
  supplied;
- rejection of an RBF model on change-point data when a jump model predicts
  held blocks better;
- no promotion of predictions outside a profile's applicability domain.

### Visibility-level synthetic tests

Simulate a connected antenna array with a point calibrator and raw noise. Mask
one scan from the gain solve, then require:

- retained anchors recover the expected gauge;
- the posterior predicts held baseline visibilities;
- uncertainty propagation explains held residual dispersion;
- antenna-relative gain recovery cannot absorb an injected source-wide flux
  change;
- enabling array-common amplitude creates the expected degeneracy;
- differential R/L remains fixed unless its external anchor is present.

### Fixed 3C391 regression tests

Assert:

- fourteen J1822−0938 gain epochs and their native coordinates are retained;
- the existing native-linear and circular-GP sweep values remain reproducible;
- post-hoc RBF smoothing is not selected on the already-opened target fold;
- masked-scan experiments never use held rows in the G solve;
- predictive reports separate interpolation, extrapolation, and prior-only
  support;
- the first recovery cohort matches the provenance in the calibration-flag
  proposal;
- activating a recovery policy does not alter hard exclusions.

### Corpus acceptance tests

A production profile passes only if:

- it beats or matches native linear transfer in outer-run predictive
  likelihood over its declared gap range;
- its uncertainty intervals have acceptable empirical coverage;
- no one calibrator or scheduling block dominates the result;
- performance remains stable across antennas and receptors;
- source-variability-risk runs do not materially change the antenna-relative
  profile;
- sealed visibility-level evaluation passes;
- the profile can be reproduced from the recorded corpus manifest.

### Target self-calibration acceptance tests

The learned prior is useful only if the complete target procedure:

- improves held target data without degrading the original trusted cohort;
- keeps corrections inside the calibrator-informed distribution unless
  independent evidence supports a departure;
- does not change integrated target flux through an array-common amplitude
  mode;
- does not change V or EVPA through differential R/L freedom;
- remains stable across deterministic baseline, time, and pointing splits;
- rejects an over-flexible target-tuned hyperparameter alternative.

## Reporting requirements

Every prior-learning run should report:

- training, inner-validation, outer-validation, and sealed run identities;
- calibrator and source-risk labels;
- cadence, duration, band, and gap distributions;
- gauge and coordinate transformations;
- candidate kernel definitions and parameter counts;
- predictive likelihood, point error, and interval coverage;
- performance versus gap duration and anchor distance;
- results by antenna and receptor;
- common-amplitude diagnostics;
- failure and exclusion reasons;
- the selected profile and applicability domain.

Every target use should report:

- profile ID and corpus version;
- distance from the profile's training distribution;
- current calibrator anchors and uncertainties;
- whether hyperparameters were fixed, marginalised, or calibrator-adapted;
- posterior correction amplitude and timescale;
- number of supported, extrapolated, prior-only, and unsupported samples;
- held-data improvement against native transfer;
- competing temporal sky-model results where relevant.

## Risks and controls

### Solver noise can become a short GP timescale

Use per-solution uncertainties, a nugget term, extended scans, and the
visibility-level benchmark. Do not infer physical variability directly from
scatter in a heterogeneous table corpus.

### A variable calibrator can become an instrumental prior

Remove array-common amplitude from the first profile. Label source-risk runs,
fit source-specific latent terms when data permits, and test profile stability
with those runs removed.

### A smooth prior can erase real fast calibration variation

Native linear transfer is always a control. Select kernels on held prediction,
not aesthetic smoothness. Retain population spread and allow the current
calibrator data to update a run-specific prior.

### A flexible prior can absorb target variability

Freeze degenerate gain modes, do not tune hyperparameters freely on the target,
and compare calibration and sky alternatives on partitions that distinguish
their responses.

### Jumps can corrupt the population length scale

Detect and label discontinuities. Quarantine them from the smooth population
fit or model them with explicit change points. Do not make every run rough
because a small number of antennas changed state.

### Corpus selection can produce optimistic evidence

Partition by independent observation, not individual scan or table. Keep
near-duplicate pipeline products together. Predeclare sealed runs and report
failures, not only successful calibrators.

### A profile can be applied outside its domain

Store band, cadence, duration, anchor distance, and instrument-epoch limits.
Fail closed or fall back to conservative transfer when those limits are
exceeded.

## Relationship to other plans

This proposal supplies the empirical hyperprior for the calibrator-anchored
self-calibration target in
[`jones-polarization-calibration.md`](jones-polarization-calibration.md). That
document defines how the residual GP enters the Jones objective. This document
defines how its permissible variability is learned and validated.

It supplies uncertainty and support classes to
[`calibration-flag-proposal.md`](calibration-flag-proposal.md). A learned GP may
predict a value where CASA had no solution, but only the quality policy can
promote the associated visibility.

It also protects the time/frequency sky discovery described in
[`time_frequency_sky_model_handoff.md`](time_frequency_sky_model_handoff.md).
The calibration prior must not be allowed to absorb a sky mode merely because
that mode varies on a familiar instrumental timescale.

The existing cadence and gain-jump analysis in
[`analyze_calibration_weather.py`](../scripts/analyze_calibration_weather.py)
is an inventory input. Its step and cadence summaries are not yet a GP
hyperprior because they do not perform blocked predictive validation or
separate source, gauge, and solution-noise modes.

## Immediate next step

Build the Phase 1 inventory before implementing another GP. Count independent
runs with at least 6--10 usable G visits, extended calibrator scans, raw
visibility access, and adequate provenance. Declare outer and sealed groups.

Then implement the solution-table blocked benchmark with nearest and native
linear transfer as mandatory controls. Use the existing RBF GP only as a
reference candidate. Do not select a production kernel until the
visibility-level benchmark confirms both predictive performance and uncertainty
coverage.
