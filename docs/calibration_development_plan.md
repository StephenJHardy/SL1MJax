# Calibration and flagging development plan

## Purpose

This document proposes a staged route from SL1MJax's current fixed-gain
Stokes-I imager to a scientifically credible, differentiable calibration
system. The implementation must remain independent of CASA at runtime while
using CASA, analytic calculations, and preserved ideal visibilities as
independent validation oracles.

The historical SL1MML repository is useful evidence that joint sky and
instrument inference was intended. It is not the specification for this work.
Its working calibration path fitted only a scalar complex gain for one
polarization and one channel; its broader time/channel path was incomplete.

## Implemented diagonal-calibration tranche

The first parallel-hand calibration tranche was completed on 2026-08-19:

- canonical schema 1.1 preserves antenna, field/role, state/intent,
  observation and feed records, row-level IDs, `FLAG_ROW`, intervals and
  optional model visibilities while retaining schema 1.0 reads;
- `CalibrationSolution` represents time gains, delays, bandpasses, validity
  masks, coordinates, gauge, interpolation and provenance as a portable JAX
  PyTree;
- non-mutating application supports preserved or propagated weights and
  requires explicit extrapolation outside solution domains;
- deterministic RR/LL fixtures gate independent static/time gains, delays,
  bandpasses, flags, noise, gradients and checkpoint round trips;
- Optax stages solve central-channel gains, delays, normalized complex
  bandpasses and full time gains using connected baseline/time holdouts;
- diagnostics report train/holdout residuals, amplitude/phase residuals,
  closure, occupancy, disconnected domains and gauge-aligned solution
  differences;
- residual outliers are returned as proposals and never alter accepted flags;
- the committed 3C391 NPZ/JSON fixture validates CASA K/B/G/antenna-position
  application and runs the JAX solve without CASA.

On the committed sample, imported CASA application agrees with CASA
`CORRECTED_DATA` to `6.5e-4` normalized complex RMS. The JAX 3C286 solve has
train/holdout RMS `0.0291/0.0310`; J1822-0938 has `0.0496/0.0493`, and its
transferred flux is `2.2867 Jy` versus CASA's `2.2960 Jy`.

Applying the flux-scaled J1822-0938 gains to raw `3C391 C1` target data gives
`0.0651` visibility RMS relative to CASA calibration. Bounded reconstructions
from the two paths correlate at `0.9960` with `0.0747` normalized RMS
difference. The target comparison and its imaging limitations are documented
in [3c391_target_imaging.md](3c391_target_imaging.md).

Gain transfer currently defaults to nearest-neighbour time application. Linear
interpolation of amplitude and unwrapped phase is implemented but had not been
selected on target validation. The first controlled comparison is now
`scripts/sweep_3c391_calibration_interpolation.py`. It solves the external
calibrators once, applies nearest and linear gains to raw `DATA`, and scores
both against the seven-pointing consensus sky fitted only on outer-training
scans. The complete outer scans remain the calibration-validation set. The
score is normalized by frozen model power so an amplitude-biased calibration
cannot improve its own denominator. Distance from CASA `CORRECTED_DATA` is a
secondary diagnostic, not the selection objective.

The full seven-pointing sweep selects linear interpolation. On 158,280 common
samples from complete held-out scans, fixed-sky residual power falls from
0.01320 for nearest gains to 0.01035 for linear gains, a 21.6% reduction.
Normalized RMS falls from 0.1149 to 0.1017. All four frequency bins improve,
and six of seven pointings improve; C7 is 0.6% worse. Held-out distance from
CASA `CORRECTED_DATA` falls from 0.00417 to 0.00166 in normalized power. CASA
itself reaches 0.00908 against the same frozen sky, so linear interpolation
closes most but not all of the calibration gap. The 3C391 target workflow now
defaults explicitly to linear transfer, while the library-wide solution
default remains nearest until another dataset validates the choice.

A subsequent fixed-sky 2^3 term ablation localizes the remaining difference.
After putting CASA and SL1MJax in a common `G`/`B` gauge, replacing only the
SL1MJax time gains with CASA gains lowers normalized residual power by 21.2%
on validation and 18.8% on the sealed fold. CASA delay gives a further small
0.4%/1.0% improvement. CASA bandpass has no repeatable benefit. The complete
CASA solution scores 0.007462/0.007548, compared with 0.009602/0.009464 for
the selected SL1MJax solution. Propagated weights improve both paths by about
1%, and removing either flux transfer is strongly rejected at about 0.318
residual power.

The imported complete CASA solution reproduces stored CASA-corrected
visibilities to `1.18e-6` normalized sealed power, so application error is not
driving this result. The material difference is now the gain solve: the saved
SL1MJax solution has six epochs from the compact golden sample, while CASA has
fourteen epochs. The next tranche must solve `G` from all calibrator rows in
the locally cached MeasurementSet. Piecewise-linear native epochs, constrained
amplitude/unwrapped-phase bases, and a circular phase GP should then compete
under the same frozen-sky validation protocol. Residual classification remains
deferred until that comparison is complete.

The native full-row baseline is now complete. The gain solver separates the
per-row observation time from the gain-solution coordinate, so one `G` can be
shared across a scan without corrupting time-dependent antenna-position phase.
The fourteen gain knots use active-weight scan centroids and match CASA's
gain-table times to much better than one second. A 300-step solve over 39,733
calibrator rows took 15.7 seconds locally and reached train/holdout RMS
0.0726/0.0740. Its fold-3 frozen-sky residual power is 0.007478, 22.1% below
the six-knot result and only 0.25% above CASA's 0.007459. Fold 4 remains closed
until the constrained and GP candidates have been ranked.

That ranking is now complete. Seven irregular-time curvature penalties and
twelve circular RBF GPs were compared with native linear transfer on fold 3.
Native linear wins at 0.007478. The weakest curvature penalty is 0.54% worse,
and stronger penalties worsen monotonically. The best GP is 2.13% worse. The
frozen winner then scores 0.007607 on fold 4, only 0.79% above CASA's 0.007547.
The GP implementation remains useful for later anchored self-calibration, but
the external calibrator currently supports keeping all fourteen measured scan
gains without smoothing.

This tranche does not implement immutable reason-coded flag versions,
pre-calibration RFI discovery, calibrator catalogues, smooth bandpass bases,
uncertainty estimation, target self-calibration, cross-hands or leakage.

## Scope

The initial calibration release should:

- ingest the metadata needed to identify antennas, fields, calibrators, scans,
  intents, spectral windows, feeds, and observations;
- preserve all input flags and weights;
- represent reversible, reason-coded flags rather than one destructive Boolean
  mask;
- solve and apply diagonal antenna-based Jones terms for parallel-hand
  Stokes-I calibration;
- support static and time-variable complex gains, delays, and complex
  bandpasses;
- transfer an absolute flux scale from a standard calibrator through a
  secondary gain calibrator to a target;
- support deterministic automatic flagging and auditable semi-automated
  flagging;
- provide solution, residual, closure, flagging, and holdout diagnostics;
- export portable calibration products without requiring CASA in the core
  environment.

The first calibration release should not claim:

- full-Stokes polarization calibration;
- direction-dependent calibration;
- ionospheric or atmospheric tomography;
- blind calibration of an arbitrary unknown sky;
- exact compatibility with every CASA calibration-table representation;
- automated scientific judgement about ambiguous RFI or source structure.

Those capabilities should be added only after the diagonal Stokes-I system has
passed the gates below.

## Scientific principles

### Measurement equation

The long-term model should use the full matrix RIME:

```text
V_pq = J_p B J_q^H + N_pq
```

where `B` is the polarized sky coherency and `J` is an ordered product of
instrumental Jones terms. The first release may use diagonal Jones matrices and
parallel hands, but the storage and API must not assume that all terms are
scalar.

The staged instrumental model is:

```text
J_p(t, ν) = G_p(t) K_p(ν) B_p(t, ν)
```

with:

- `G`: time-dependent electronic/atmospheric complex gain;
- `K`: non-dispersive delay, represented as a phase slope with frequency;
- `B`: residual complex bandpass.

Later terms may include:

- `D`: polarization leakage;
- `P`: parallactic-angle/feed rotation;
- `E`: primary beam and other direction-dependent effects;
- dispersive ionospheric delay;
- pointing and antenna-position corrections.

Each term must be independently enabled, disabled, initialized, regularized,
serialized, and tested.

### Identifiability and gauge

Calibration has unavoidable degeneracies. They must be fixed explicitly rather
than left to optimizer initialization:

- select and record a reference antenna;
- fix its phase at the reference time/channel;
- anchor amplitude with a known calibrator spectrum or an explicit
  normalization constraint;
- define the reference frequency for delays and bandpasses;
- record disconnected antenna/time/channel solution domains;
- compare solutions only after applying the same gauge transformation.

If no valid amplitude or phase anchor exists, the solver must report the
unidentifiable degree of freedom instead of returning an apparently absolute
solution.

### Separation of concerns

Four operations must remain distinct:

1. observation and metadata extraction;
2. flagging and weight preparation;
3. calibration solving;
4. calibration application and imaging.

Applying a solution must not mutate the original canonical data. It should
return a new view or dataset with provenance linking the source data,
calibration product, flag version, and application settings.

## Canonical metadata prerequisites

Before implementing solvers, extend the canonical schema with dataset-level
metadata records.

### Antennas

Preserve from the MeasurementSet `ANTENNA` subtable:

- antenna ID and name;
- station or VLA pad name;
- ITRF position in metres;
- dish diameter;
- mount type;
- active antenna IDs in each block.

The observed MS positions are authoritative. CASA `vla.a/b/c/d.cfg` files are
useful for simulation but are not substitutes for observation metadata.

### Fields, sources, and scan intents

Preserve:

- field ID, field name, phase centre, delay direction and reference direction;
- source ID, source name, direction, spectral-window association and available
  rest frequencies;
- state ID and `OBS_MODE`;
- row-level state IDs;
- normalized intents such as target, flux, bandpass, phase, amplitude,
  polarization and pointing calibration;
- observation IDs, telescope name, observer and project ID.

Selection should support field names and intents, not only numeric IDs.

### Feeds and correlations

Preserve receptor type, feed ID, receptor angle, polarization basis and ordered
correlation products. This metadata is required before full-polarization Jones
terms can be trusted.

### Flags

Replace a single conceptual flag with a versioned flag set containing:

- the effective Boolean mask;
- one or more reason bits per visibility sample;
- the source of each reason: imported, structural, automatic, residual-based or
  manual;
- algorithm name, version and parameters;
- timestamp and parent flag version;
- summary counts by antenna, baseline, scan, field, SPW, channel and
  correlation.

The original MS flags must always remain recoverable.

## Calibration product model

Introduce a versioned `CalibrationSolution` PyTree and portable serialization.
It should include:

- Jones-term type and parameterization;
- complex values or real parameter components;
- antenna, time, frequency, SPW and correlation coordinates;
- validity and solution-quality masks;
- reference antenna and gauge convention;
- calibrator sky model and flux standard;
- optimizer configuration and convergence history;
- train and holdout metrics;
- input dataset and flag-version provenance;
- interpolation and extrapolation policy;
- software and schema versions.

Suggested public operations are:

```text
solve_calibration(dataset, sky_model, model, selection, config)
apply_calibration(dataset, solution, interpolation)
calibration_residuals(dataset, solution, sky_model)
write_calibration(solution, path)
read_calibration(path)
```

Application should fail clearly outside a solution's validity interval unless
extrapolation is explicitly requested.

## Solver architecture

### Parameterization

Use real-valued optimizer parameters with explicit transforms:

- log amplitude for positive gain amplitudes;
- unwrapped or unit-complex phase parameters with a documented convention;
- delay in seconds about a reference frequency;
- residual bandpass amplitudes and phases;
- spline, knot-grid, or low-order basis coefficients for smooth time/frequency
  behavior.

Independent free gain values for every integration and channel should be a
diagnostic upper bound, not the default model.

### Objective

Start with the existing weighted complex residual objective:

```text
sum(w |V_observed - V_model|²) / sum(w)
```

Add:

- robust alternatives such as Huber or Student-t residuals;
- time/frequency smoothness priors;
- weak unity priors for poorly constrained terms;
- solution-domain masks;
- explicit penalties or constraints implementing the gauge.

Flags, non-finite values and non-positive weights must contribute exactly zero.
Robust losses must not silently replace flagging: both the down-weighted
residuals and the final flags must be inspectable.

### Optimization strategy

Implement in increasing complexity:

1. solve gains against a fixed calibrator sky;
2. alternate gain and simple calibrator-source updates;
3. transfer fixed solutions to target data;
4. optionally alternate target imaging and gain self-calibration;
5. consider joint sky/instrument optimization only after the alternating path
   is independently validated.

Use JAX and Optax with deterministic seeds, x64 kernels, bounded chunking,
checkpointing, early stopping, and structured train/holdout splits. Never
select the best model using the same samples used for the final quality claim.

## Flagging system

Flagging should be staged from objective rules to increasingly interpretive
rules. Every operation must be reversible and reason-coded.

### Stage A: imported and structural flags

Apply deterministic checks:

- preserve MS `FLAG` and `FLAG_ROW`;
- reject non-finite visibilities, weights, UVW, frequencies and times;
- reject non-positive weights;
- optionally exclude autocorrelations;
- detect missing antennas, channels and undefined cells;
- flag known shadowing using antenna positions, dish diameters and source
  direction;
- support configurable scan-start quack intervals;
- support explicit channel-edge, antenna, baseline, scan, field, SPW,
  correlation and time-range selections.

These rules should be unit tested without statistical thresholds.

### Stage B: automatic pre-calibration flagging

Implement conservative deterministic algorithms:

- robust median/MAD amplitude clipping per SPW and correlation;
- time-difference and frequency-difference outlier detection;
- SumThreshold-style connected RFI detection in time-frequency planes;
- optional scale-invariant rank or morphological dilation of detected regions;
- antenna and baseline occupancy checks;
- spectral-kurtosis checks where the available integrations support them;
- zero or repeated-value detection;
- edge-channel policies based on known bandpass roll-off.

Thresholds must be derived within homogeneous groups. Data from different
fields, SPWs, correlations, scan intents or strongly varying source models must
not be pooled indiscriminately.

An optional AOFlagger backend may be considered later, but the default test
suite must not depend on an external binary.

### Stage C: calibration-aware flagging

After an initial conservative solve, inspect:

- normalized residual amplitude and phase;
- abrupt solution jumps;
- solution signal-to-noise and missing intervals;
- closure phase and closure amplitude outliers;
- antenna-based versus baseline-specific residual structure;
- correlation disagreement;
- time-frequency residual morphology.

Residual-based flagging must use cross-validation or held-out residuals where
practical. It must not iteratively erase a real source omitted from the sky
model. Set maximum new-flag fractions and stop for review when exceeded.

### Semi-automated review

Provide an auditable report and a declarative rule file rather than requiring a
custom interactive GUI in the first release. The report should show:

- flag occupancy by antenna, baseline, scan, field, SPW, channel and
  correlation;
- amplitude and phase versus time, frequency and UV distance;
- waterfall plots before and after each flag version;
- calibration solutions and residuals;
- closure diagnostics;
- automatic-rule candidates ranked by severity;
- expected impact of accepting each candidate.

Users should accept, reject or modify proposed rules in YAML or JSON and rerun
deterministically. A later UI may write the same rule format. Manual flagging
must never be represented only as clicks with no reproducible record.

### Flagging iteration policy

A default calibration run should follow:

1. import flags;
2. apply structural rules;
3. run conservative pre-calibration flagging;
4. freeze flag version and solve;
5. generate residual-based proposals;
6. review or accept proposals under configured limits;
7. create a child flag version and solve again;
8. stop when held-out residuals and flag occupancy stabilize.

Report both the metric improvement and the amount of removed data. Lower
training residuals alone are not evidence of better calibration.

## Standard calibrator models

Create an independently sourced calibrator package rather than copying
SL1MML's TensorFlow constants. It should:

- normalize common B1950, J2000 and 3C aliases;
- implement a named, cited flux-density standard such as Perley–Butler 2017;
- state the frequency and epoch validity of each model;
- distinguish scalar spectral models from resolved image/component models;
- include polarization models only where independently supported;
- expose model uncertainty;
- test published reference values at several frequencies.

Primary standards such as 3C286, 3C48 and 3C147 must not automatically be
treated as point sources. The selected model must account for observing band,
array configuration and baseline range.

## Synthetic calibration validation

CASA should create realistic VLA MeasurementSets and observing schedules.
Deterministic Jones corruptions should be injected either by SL1MJax or through
explicit CASA calibration tables. Preserve:

- ideal visibilities;
- corrupted visibilities;
- every injected Jones parameter;
- random seeds;
- sky/component truth;
- array, scan, SPW, correlation and noise metadata;
- CASA version and complete generation script.

CASA's random gain, thermal-noise, atmospheric and leakage simulation may be
used as secondary realism tests. Exact recovery gates should use deterministic
truth. CASA's `setbandpass` is documented as not implemented, so bandpass and
delay truth should come from explicit tables or direct deterministic
corruption.

### Calibration simulation ladder

1. One point calibrator, one channel, noiseless static scalar gains.
2. Multiple antennas with a deliberately chosen reference antenna and gauge.
3. Time-variable gains with interleaved phase-calibrator and target scans.
4. Multi-channel delay phase slopes with a known reference frequency.
5. Smooth complex bandpass plus deliberately bad edge channels.
6. Absolute flux calibrator, secondary gain calibrator and unknown target.
7. Gaussian noise with correct inverse-variance weights.
8. Injected flags, isolated outliers and connected time-frequency RFI.
9. Missing antennas, disconnected solution intervals and entirely flagged
   channels.
10. Parallel-hand receptor-dependent gains.
11. Full correlations, cross-hand delay, leakage and polarization angle.
12. Source-model error and resolved calibrators.

Each stage must first pass without noise, then with controlled noise, and then
with controlled flagging.

### Required synthetic assertions

After a common gauge transformation:

- recovered gain amplitude and phase match injected truth;
- recovered delay and bandpass coefficients match truth;
- applying solutions restores ideal visibilities;
- corrected visibility normalized complex RMS meets an explicit threshold;
- flux transfer recovers target flux without fitting the target;
- held-out residuals agree with the injected noise distribution;
- uncertainty or solution-quality estimates track empirical recovery error;
- flags achieve measured precision and recall against injected RFI masks;
- clean astronomical transients and spectral lines remain unflagged;
- repeated seeds reproduce solutions, flags and histories;
- chunked and unchunked calculations agree.

Thresholds should be fixed in fixture metadata and changed only alongside a
documented scientific-model revision.

### Residual handling and variable-sky protection

An existing flag is not a truth label. Flag evaluation must report four
counts: flagged residual-tail samples, flagged residual-bulk samples,
unflagged tail samples, and unflagged bulk samples. A flagged sample may be
considered for restoration only after recalibration without that flag. Its
old `CORRECTED_DATA` is not sufficient evidence because the flag may have
excluded it from the gain solve.

Residual handling has distinct operating modes:

- report-only mode never changes weights or flags;
- robust-weight mode continuously limits extreme influence but creates no
  hard flag;
- static-sky mode may propose hard residual flags when source constancy is a
  valid prior;
- transient-safe mode subtracts cross-validated temporal or spectral sky
  components before scoring corruption.

Transient-safe protection must use interferometric coherence, not just image
amplitude. Fit a candidate sky response on one baseline subset and validate
the coefficient on disjoint baselines. Protect it only if it predicts the
held-out complex visibilities. This applies equally to continuum variability
and narrow spectral-line emission. Antenna-local, baseline-local, and
non-closing residuals remain candidates for robust weighting or flags.

Instrumental metadata, non-finite values, zero weights, saturation, and
invalid calibration domains remain hard flags in every mode. Sky coherence
cannot override them. All model-residual decisions remain proposals until a
frozen validation comparison shows better calibration and imaging on samples
that did not select the proposal.

The 3C391 wide-field composite control now separates missing sky from the
remaining flagging problem. The improved fixed sky lowers sealed residual
power by 6.32% and lowers the old fixed-scale `z>6` population by 7.09%.
After estimating the narrower robust scale again, however, about 5% of active
samples remain in the heavy tail. The same sky changes residual power in the
currently flagged cohort by only -0.041%.

The immediate 3C391 task is therefore a fixed-sky calibration-complexity
study. Compare the current nearest solution assignment with progressively
smoother time models and interpolation, and select complexity using held-out
time and baseline cells. Do not train a residual classifier on the present
labels. The current tail is still a mixture of calibration error, remaining
sky error, and corruption, so such a classifier would learn the wrong target
and could suppress genuine variable emission.

This study is now complete. Linear transfer between all six native gain epochs
wins both the fold-3 selection score and the sealed fold-4 score. It lowers
sealed residual power by 19.5% relative to nearest transfer. Constant, linear,
and quadratic global time models all lose, so post-hoc smoothing removes real
calibrator information. The selected JAX result remains 25.4% above CASA in
sealed residual power. The gap occurs in every pointing and every frequency
bin. A term-by-term `G`, `K`, `B`, flux-scale, and weight-propagation
ablation against the portable CASA solution is now required before automatic
residual classification or target self-calibration.

## Independent CASA comparison

For selected synthetic fixtures:

1. solve equivalent CASA `G`, `K` and `B` tables;
2. apply CASA and SL1MJax solutions separately;
3. compare gauge-normalized solutions;
4. compare corrected visibilities, residuals, dirty images and PSFs;
5. compare flag occupancy and statistical weights;
6. explain expected differences in interpolation, normalization and robust
   loss rather than forcing byte equality.

Neither CASA nor SL1MJax should generate both the test input and the only
acceptance metric.

## Real-data ladder

### 3C391 tutorial

Use the documented CASA 6.7 workflow as the first reproducible calibration
reference:

- preserve the tutorial's initial flags and manual/quack rules;
- extract antenna, field, source and intent metadata;
- reproduce antenna-position, flux-model, phase, delay, bandpass, gain,
  flux-transfer, apply and statistical-weight stages in CASA;
- retain CASA tables, weblogs, commands and calibrated visibilities;
- initially compare SL1MJax application of imported or translated CASA
  solutions;
- only then solve individual terms in SL1MJax;
- use J1822-0938 as the compact gain-calibrator test;
- keep 3C286's resolved CASA model as the flux reference;
- defer cross-hand polarization calibration because the tutorial is
  intentionally Stokes I.

### Additional observations

Select later public VLA datasets with:

- successful pipeline calibration and available QA products;
- compact gain calibrators;
- single-pointing continuum targets with simple reference images;
- manageable size after field/SPW selection;
- at least one dataset each from D/C and A/B configurations;
- multiple bands and both linear scientific simplicity and realistic RFI.

Real data validates robustness and provenance, not exact parameter recovery.

## Implementation phases

### Phase 0: metadata and provenance

- Extend the canonical schema for antennas, fields, sources, states, feeds,
  observations and reason-coded flag versions.
- Extend MeasurementSet extraction and round-trip tests.
- Add field-name and intent selectors.
- Gate schema migration and backwards compatibility.

### Phase 1: flagging foundation

- Implement structural flag rules and declarative manual rules.
- Add immutable flag versions, summaries and provenance.
- Add synthetic injected-mask precision/recall tests.
- Produce machine-readable and visual reports.

### Phase 2: static diagonal gains

- Implement `CalibrationSolution` and scalar/diagonal `G`.
- Enforce reference-antenna and amplitude gauges.
- Solve fixed point-calibrator fixtures.
- Apply solutions without mutating source data.
- Compare against analytic truth and CASA.

### Phase 3: time-variable gains and flux transfer

- Add time knots and interpolation.
- Add standard calibrator spectra and resolved-model selection.
- Solve flux and phase calibrators and transfer to targets.
- Add solution SNR, validity intervals and holdout diagnostics.

### Phase 4: delays and bandpasses

- Add explicit `K` delay and smooth complex `B` terms.
- Support staged and joint solves.
- Add bad-edge-channel and disconnected-domain behavior.
- Compare corrected visibilities with CASA `K` and `B` workflows.

### Phase 5: automatic and calibration-aware flagging

- Add robust time-frequency detectors and SumThreshold-style flagging.
- Add closure and held-out residual proposals.
- Add review rules, safety limits and iteration stopping criteria.
- Test preservation of transients, spectral lines and source-model residuals.

### Phase 6: target self-calibration

- Alternate imaging and gain solving with frozen validation data.
- Require objective and held-out improvement.
- Guard against flux-scale drift and model absorption.
- Compare with CASA self-calibration on synthetic and 3C391 data.

### Phase 7: full polarization

- Upgrade to 2×2 coherencies and Jones matrices.
- Add parallactic-angle/feed rotation, cross-hand delay, leakage and
  polarization-angle calibration.
- Validate all ordered correlations and polarized calibrator models.

### Phase 8: direction-dependent effects

- Add primary beam, pointing and selected atmospheric/ionospheric terms only
  after direction-independent calibration is stable.
- Begin with deterministic synthetic fields and known beam models.

## Release gates

A calibration release is acceptable only when:

- all metadata and flag provenance round-trip losslessly;
- analytic Jones and gradient tests pass in x64;
- static and time-variable synthetic gains recover after gauge normalization;
- corrected synthetic visibilities match preserved ideal visibilities;
- delay, bandpass and flux-transfer stages pass independent gates;
- automatic flags meet fixture-specific precision/recall thresholds and do not
  remove protected astronomical signals;
- solution application is stable across chunk sizes;
- train and structured holdout metrics are reported separately;
- CASA comparison differences are bounded and explained;
- a bounded 3C391 calibrator workflow completes without CASA in the core
  solve/apply environment;
- all products contain enough provenance to recreate extraction, flagging,
  solving and application.

## Operational safeguards

- Never overwrite `DATA`, `CORRECTED_DATA`, weights or flags in the source MS.
- Never silently infer calibrator roles from source names when scan intents are
  available.
- Never use an automatically selected reference antenna without recording why
  it was selected.
- Never interpolate across a fully invalid solution interval without an
  explicit policy.
- Never present training residual improvement as independent validation.
- Never accept an automatic flagging pass that exceeds configured occupancy
  limits without review.
- Never treat a primary flux calibrator as a point source solely because it is
  called a calibrator.
- Never copy legacy calibrator coefficients without a current citation,
  validity range and numerical reference tests.

## Immediate next steps

1. Repeat the primary-calibrator and flux-transfer solve on all rows to test
   whether flux-scale uncertainty explains the remaining sub-percent CASA gap.
2. Add exact gain-weight propagation before averaging once coordinate handling
   supports a common unweighted time grid.
3. Define immutable, reason-coded flag versions and declarative manual rules.
4. Add conservative structural and pre-calibration time/frequency RFI rules.
5. Add smooth bandpass/time bases, robust losses and solution uncertainty.
6. Build independently sourced Perley–Butler calibrator models.
7. Exercise the same gates on another VLA band/configuration before target
   self-calibration.
8. Defer cross-hands, leakage and polarization angle until the diagonal system
   is stable on multiple observations.
