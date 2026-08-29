# Calibration flag and visibility recovery proposal

## Purpose

SL1MJax needs to distinguish bad measurements from measurements that CASA
excluded because its chosen calibration model could not supply a solution.
Those cases have different scientific consequences:

- data that was never recorded cannot be recovered;
- data from a receiver known to be absent or broken should remain excluded;
- data with an unstable instrumental state may require a different calibration
  model;
- otherwise sound target data may be recoverable when a more flexible,
  calibrator-anchored model fills a gap in calibration coverage;
- one correlation product may be usable even when another is not.

The current Measurement Set `FLAG` column cannot express these distinctions.
CASA also propagates calibration-table flags into that column by default, so a
single boolean eventually mixes acquisition faults, reduction policy, missing
calibration, and later statistical decisions.

This proposal defines a provenance-preserving quality model and a conservative
recovery process. It does not propose clearing existing flags and using the
current `CORRECTED_DATA` unchanged. Any candidate visibility must be corrected
again from the original `DATA` with the proposed calibration model and must
pass independent validation.

## Decision

Expose a small set of status levels to users, but do not store one status enum
as the source of truth. Store independent facts and decisions, then derive the
status needed by a particular operation.

The source facts are:

1. whether a visibility record exists;
2. acquisition and raw-data quality information;
3. explicit reduction exclusions and their reasons;
4. validity and uncertainty for each calibration term;
5. residual-quality evidence;
6. the analysis policy that decides which supported samples to use.

This separation lets a decision be revised without erasing why it was made. It
also lets a sample be valid for Stokes-I imaging but invalid for full-Stokes
imaging.

## Terminology

A *visibility sample* in this document is one row, channel, and correlation
cell. A row normally contains several channels and correlation products.

An absent row is not a flagged visibility. It is missing data. No calibration
model can create the measurement that would have occupied that row.

A *hard exclusion* is data that must not enter the present analysis because an
acquisition fact establishes that it is unusable. The reason remains visible
and immutable even if a later policy groups it with other excluded data.

A *recovery candidate* is recorded data that is presently excluded but for
which no hard acquisition fault is known. Recovery requires a new calibration
application and independent validation.

*Calibration support* says whether the selected calibration model can evaluate
all terms required for a given visibility product. It is not a judgment about
whether the raw visibility is good.

*Quarantine* means that data remains available for diagnostics and controlled
experiments but cannot affect the production sky or calibration solution.

## User-facing status

The interface can present the following derived states:

| Status | Meaning | Default action |
|---|---|---|
| `missing` | No visibility record exists | Cannot recover |
| `hard_invalid` | A known acquisition or hardware fault applies | Exclude |
| `instrument_suspect` | Data exists, but a known corruption or unstable state applies | Quarantine |
| `calibration_unsupported` | Raw data may be usable, but a required Jones term is unavailable | Recovery candidate |
| `calibration_extrapolated` | A model supplies a prediction outside ordinary measured or interpolated support | Use only with uncertainty and validation |
| `residual_suspect` | Calibration exists, but independent residual tests find an inconsistency | Quarantine or robustly downweight |
| `active` | The current operation has adequate data and calibration support | Use normally |

These names describe policy outcomes. They do not replace the underlying
reason records, term validity, or uncertainty.

There is no universal ordering between all states. For example, a sample can
be both calibration-extrapolated and residual-suspect. The final policy must
resolve such combinations conservatively.

## Why a single multi-level flag is insufficient

Calibration support depends on the requested product and measurement equation.
For a diagonal circular-feed chain, RR needs the R terms from both antennas and
LL needs the L terms from both antennas. RL and LR need the appropriate terms
from both receptors. A full \(2\times2\) correction may additionally require a
complete four-product coherency at that row and channel.

The ea04 case in 3C391 illustrates the problem. ea04 has usable diagonal,
cross-hand-delay, and R--L phase information from earlier scans. It has no
leakage solution because it was absent from the only 3C84 scan. Its parallel
hands can therefore remain usable under the CASA-compatible application, while
cross-hands involving ea04 are unsupported. One row-level status cannot state
that accurately.

The same issue occurs across time and frequency. A gain may be supported at one
time but extrapolated at another. A D or X table may cover only part of the
channel range. Validity therefore belongs to the calibration term at its native
antenna, time, frequency, and receptor coordinates. It should be projected to
visibility products only when a block is evaluated.

## Proposed source model

### 1. Data availability

Availability is determined from row coverage in the Measurement Set. Record
antenna participation by scan and time interval so missing observations are
not confused with flags.

This layer answers questions such as:

- Did both antennas participate in this scan?
- Was the relevant correlation product recorded?
- Does the channel exist in this spectral window?

Missing records have no visibility array element in the canonical block. A
coverage report, rather than a synthetic flag, represents them.

### 2. Acquisition and reduction reasons

Preserve reasons instead of reducing them immediately to a boolean. Initial
reason families should include:

- online or archive flag;
- absent receiver or declared hardware failure;
- observer-reported corruption;
- setup scan;
- scan-start quack;
- shadowing;
- non-finite data or non-positive base weight;
- diagnostic manual antenna, baseline, time, channel, or correlation exclusion.

Each reason record needs:

- a stable reason code;
- its source, such as observer log, archive flag, tutorial, or SL1MJax audit;
- the selector it affects;
- creation time and software/procedure version;
- a disposition: hard exclusion, quarantine, or reviewable policy;
- optional free-text evidence and an external or repository reference.

Most rules are sparse. “ea13 has no C-band receiver” should be stored once as
an antenna/frequency selector, not repeated as a string for every sample. An
irregular imported mask can still be stored as a compact bit array when no
sparse selector describes it.

Reason history is append-only. A later decision may supersede a policy, but it
must not delete the earlier reason.

### 3. Calibration-term coverage

Each `CalibrationSolution` term already carries a validity array. Extend that
concept with a support classification and uncertainty:

| Support | Meaning |
|---|---|
| `supported` | Evaluation is inside the declared solution domain using the normal transfer rule |
| `extrapolated` | A calibrated prediction exists outside normal interpolation support |
| `prior_only` | The value is supplied mainly by an external prior rather than this observation |
| `unsupported` | No defensible value exists |

For each term, retain useful diagnostics where available:

- posterior variance or a compact covariance-mode representation;
- distance in time or frequency from the nearest calibrator anchor;
- the anchors used for interpolation or extrapolation;
- solver rank, effective sample count, and signal-to-noise diagnostics;
- whether the value came from an imported CASA table, a direct estimator, an
  iterative solve, or a GP posterior.

The first implementation need not provide a complete covariance model. It
must, however, distinguish “valid by ordinary transfer” from “a value was
manufactured by unrestricted extrapolation.”

Interpolation support must preserve the original solution cadence and its
invalid knots. Removing invalid knots and interpolating across every remaining
value inside the first-to-last time range can silently bridge a long outage. A
numerical value may still be produced there, but it should be classified as
`extrapolated` unless the gap is within a declared ordinary interpolation
limit. Store the bracketing anchor identities, the gap length, and the maximum
supported gap with the transfer policy.

Term coverage should stay compact at antenna coordinates. The apply path can
derive correlation-level support using the receptor dependencies already used
by `_product_validity`. This avoids storing a large reason tensor for every
possible calibration chain.

### 4. Residual evidence and continuous weights

Residual evidence is model-dependent, so it must not be written back as an
acquisition fact. Keep:

- robust residual scores;
- the sky model and calibration version used to compute them;
- discovery, selection, and evaluation partition identities;
- protected temporal or spectral sky responses;
- proposed hard exclusions;
- continuous weight multipliers.

The existing non-mutating machinery in
[`flagging.py`](../src/sl1mjax/flagging.py) is the correct base. A high residual
may propose quarantine or a lower weight, but it does not prove that the
visibility was corrupt. Short baselines in 3C391 have already shown that a
missing diffuse sky component can produce a repeatable residual tail.

### 5. Analysis policy

The analysis policy derives an effective mask and weight for a named operation.
For example:

```text
effective_active =
    record_exists
    AND raw_data_finite
    AND no_applicable_hard_exclusion
    AND required_calibration_terms_supported_by_policy
    AND not_quarantined_for_this_analysis
    AND effective_weight > 0
```

Policies should be explicit and versioned. Initial policies could be:

- `casa_reproduction`: reproduce the selected CASA flag version and operator;
- `conservative_science`: accept supported terms only, with no extrapolation;
- `recovery_trial`: admit a declared candidate cohort using bounded
  extrapolation and uncertainty;
- `diagnostic_all_recorded`: expose recorded finite data without allowing it
  to enter an optimizer.

Changing a policy must not mutate the source visibility flags or calibration
solution.

## Compact implementation shape

Do not immediately expand `VisibilityBlock.flag` into a large general-purpose
object. Keep it as the conservative effective flag for existing call sites,
then introduce a sidecar that can derive that flag for new workflows.

A possible public shape is:

```python
@dataclass(frozen=True)
class QualityRule:
    reason: QualityReason
    source: str
    disposition: QualityDisposition
    selector: VisibilitySelector
    evidence: dict[str, object]


@dataclass(frozen=True)
class CalibrationCoverage:
    term: CalibrationTerm
    support: np.ndarray
    valid: np.ndarray
    variance: np.ndarray | None
    provenance: dict[str, object]


@dataclass(frozen=True)
class VisibilityQuality:
    acquisition_flag: np.ndarray
    irregular_reason_bits: np.ndarray | None
    rules: tuple[QualityRule, ...]
    residual_weight_multiplier: np.ndarray | None
    provenance: dict[str, object]
```

The exact class boundaries can change during implementation. The invariants
matter more:

- reasons and calibration validity remain separate;
- sparse rules remain sparse;
- term validity is projected to products lazily;
- the effective flag is reproducible from a policy;
- provenance survives serialization.

Use stable integer reason bits only for machine interchange and compact masks.
Use named enums in Python and include the code-to-name mapping in every
serialized schema. Unknown reason bits must fail closed rather than silently
becoming active.

## Importing CASA provenance

CASA uses one current `FLAG` column, but saved flag versions let us recover part
of the history. Import at least three snapshots when available:

1. the earliest retained or acquisition flag state;
2. the exact input flags used by the calibration solve;
3. the flags after calibration application.

For 3C391 these are:

- `sl1mjax_pristine`;
- `sl1mjax_calibration_input`;
- `sl1mjax_post_apply` or `sl1mjax_post_polcal`.

The differences have different meanings:

```text
precalibration_exclusion = calibration_input AND NOT pristine
apply_added              = post_apply AND NOT calibration_input
```

Do not automatically label every `apply_added` cell as bad raw data. Attribute
it to the calibration term that made it unusable. This can be done by:

- evaluating imported term-validity arrays through the JAX apply path;
- applying or trialling CASA terms one at a time on a disposable working copy;
- comparing named intermediate flag versions when they exist.

If exact attribution is not possible, use an explicit
`calibration_unavailable_unknown_term` reason. Do not guess.

The production importer should also record that CASA's default
`applymode="calflag"` writes calibration unavailability into the data flags.
`applymode="calonly"` is not a recovery mechanism: it can leave uncalibrated
data appearing active.

## 3C391 initial classification

The existing 3C391 data provides the first fixed acceptance fixture.

### Acquisition and manual exclusions

The local `sl1mjax_pristine` version has zero effective flags. The reproducible
tutorial procedure then:

- flags setup scan 1;
- flags the first 10 seconds of each scan;
- flags ea13 because the observer log says it had no C-band receiver;
- flags ea15 because the observer log reports corrupted data;
- flags ea05 after the diagnostic gain phases switch between two states.

ea13 is a hard exclusion. ea15 and ea05 remain quarantined rather than being
described as ordinary calibration gaps. Recovering ea05 would require a
change-point or discrete-state phase model; a smooth GP may average the two
states and make the correction worse.

### Diagonal calibration gaps

The calibration-input flag fraction is 28.2236%. The K/B/G `applycal` step
raises it to 34.2856%. The difference is 13,119,232 row-channel-correlation
cells, or 6.062 percentage points of the complete table.

Those added flags:

- affect the target fields rather than the calibrators;
- affect all 64 channels and all four recorded correlations together;
- occur on baselines whose endpoint antenna lacks a usable linearly
  interpolated gain solution.

The main target intervals are:

| Target scans | Affected antenna endpoints |
|---|---|
| 39--45 | ea03 |
| 71--77 | ea02, ea08, ea16 |
| 79--82 | ea02, ea08, ea16, ea26 |
| 83--84 | ea08, ea16, ea26 |
| 85 | ea26 |
| 87--93 | ea19, ea26 |
| 95--99 | ea04, ea19, ea24, ea26, ea27 |
| 100--101 | ea04, ea19, ea24, ea27 |

These are the first recovery candidates for a calibrator-anchored time model.
The target measurements exist, but one of the bracketing J1822−0938 solutions
does not. They must remain excluded under the conservative policy until a
recovery experiment passes.

### Polarisation-specific gap

ea04 has no rows in scans 102 or 103. Scan 103 is the only 3C84 leakage scan.
CASA consequently writes fully flagged ea04 entries in both `3c391.G84` and
`3c391.Df0`.

The polarisation application adds 6,200,960 flagged cells, or 2.865 percentage
points. They are RL/LR cells on baselines containing ea04. Kcross and Xf for
ea04 are supported by earlier 3C286 data.

The initial classification is therefore:

- ea04 RR/LL: potentially active under the CASA-compatible parallel-preserving
  operator when all diagonal terms are supported;
- ea04 RL/LR: calibration-unsupported because D is absent;
- exact full-coherency operations involving ea04: unsupported unless a new
  independently anchored leakage solution is supplied.

A time GP cannot infer an antenna leakage term that was never measured in its
only leakage-calibrator scan. Recovery needs another polarimetric anchor, such
as a constrained solve against the known 3C286 model or compatible external
calibrator data. Target polarization alone must not determine that value.

## Recovery workflow

Recovery is a model-comparison experiment, not an unflagging operation.

### 1. Declare the candidate cohort

Select candidates by provenance before looking at their science residual
improvement. Examples are one missing gain interpolation interval or all ea04
cross-hands. Do not combine unrelated reason classes in the first experiment.

Record the selector and freeze it in the experiment manifest.

### 2. Start from original data

Restore or read the flag state that existed before `applycal`. Read `DATA`, not
the possibly incomplete `CORRECTED_DATA`. Preserve hard exclusions. The
candidate cohort remains quarantined while the new correction is computed.

### 3. Fit a bounded calibration alternative

Use the smallest additional freedom appropriate to the failure:

- nearest valid gain transfer as a simple control;
- calibrator-anchored linear interpolation with explicit missing endpoints;
- the proposed anchored GP, including posterior variance;
- a change-point model for a demonstrated discontinuity;
- a new externally anchored polarimetric solve for a missing D/X term.

Do not use an unconstrained target gain or an unrestricted Jones matrix as the
first recovery model. It could absorb missing sky structure.

### 4. Correct candidates and retain uncertainty

Apply the complete chain from raw `DATA`. Mark each candidate as supported,
extrapolated, prior-only, or still unsupported. Propagate a useful calibration
uncertainty representation into the objective or effective weight.

A GP prediction with large posterior variance is not equivalent to a measured
calibration value. Prefer marginalisation over justified calibration modes. A
temporary inverse-variance weight is acceptable for the first engineering
experiment if its approximation is stated and tested.

### 5. Validate independently

Use partitions that answer different failure modes:

- held calibrator scans or integrations for interpolation quality;
- connected held baselines or antennas for antenna-solution transfer;
- held target times for generalisation beyond the recovered interval;
- held channels for any new frequency freedom;
- held mosaic pointings for sky/calibration separation;
- calibrator closure and polarization-floor tests.

The candidate samples used to fit a target calibration correction cannot also
be the sole evidence for accepting their recovery. Compare at least:

1. the unchanged conservative calibration;
2. a simple transfer such as nearest or linear interpolation;
3. the proposed flexible model.

### 6. Promote, retain in quarantine, or reject

Promotion changes only the named analysis policy. It does not delete the CASA
flag or its provenance.

A promoted cohort must identify:

- the calibration model and version that supports it;
- its support class and uncertainty;
- the validation partitions and metrics;
- the scope in time, frequency, antenna, receptor, and correlation;
- whether it is eligible for sky fitting, calibration fitting, or evaluation
  only.

## Build plan

### Phase 1: read-only provenance audit

Implement the data model and reporting without changing any objective or
effective flag.

1. Add named quality reasons, dispositions, selectors, and schema versioning.
2. Load CASA flag versions and compute differences without mutating the MS.
3. Inventory row availability by antenna and scan.
4. Attribute apply-added flags to calibration terms where the existing JAX
   validity path can prove the cause.
5. Emit JSON summaries and compact masks or sparse rules.

The report should reproduce the 3C391 counts in this document before later
phases begin.

### Phase 2: operation-dependent support

1. Extend calibration evaluation to return term-level support as well as the
   current final boolean validity.
2. Project antenna/receptor support to requested correlation products.
3. Define exact-coherency requirements separately from product-local
   requirements.
4. Add a policy object that derives an effective mask and weight.
5. Keep the default policy bit-for-bit compatible with the current conservative
   path.

This phase should make ea04's parallel/cross-hand distinction directly
inspectable without recovering any data.

### Phase 3: first diagonal recovery experiment

Choose one bounded time interval caused only by a missing bracketing gain. Do
not start with ea05, ea15, or polarisation leakage.

1. Establish nearest-transfer and existing-linear controls.
2. Add the calibrator-anchored GP prediction and its uncertainty.
3. Reapply K/B/G to raw candidate data.
4. Fit no new differential R/L, D, X, or sky-frequency freedom.
5. Compare all three models on predeclared held calibrator and target cohorts.
6. Seal the result before expanding to other intervals.

This experiment tests the recovery architecture with the lowest polarization
and hardware risk.

### Phase 4: integrate with sky/calibration major cycles

Allow a promoted cohort into the sky objective only after Phase 3 passes. The
major-cycle controller must freeze the quality policy during one model
comparison. Adding a recovered cohort is itself a model change and needs a
fresh validation score.

Reset optimizer state when the active cohort changes. Report improvements both
with and without the recovered samples so extra data volume cannot conceal a
worse fit on the original trusted cohort.

### Phase 5: polarisation and special-state experiments

Treat these as separate projects:

- solve ea04 leakage from an independent polarimetric anchor before admitting
  its cross-hands;
- test an ea05 discrete-state or change-point calibration model;
- inspect ea15 at native resolution to determine whether any bounded intervals
  lack corruption.

None of these should be enabled merely because the diagonal GP experiment
succeeds.

## Test plan

### Unit tests

Test:

- reason-code and schema round trips;
- sparse selector materialisation for antenna, scan, time, channel, and
  correlation scopes;
- precedence of hard exclusions over every recovery policy;
- unknown reasons failing closed;
- policy derivation without mutation of source arrays;
- product validity for RR, RL, LR, and LL from receptor-level term support;
- exact-coherency versus product-local requirements;
- an invalid solution knot creating an explicit gap rather than disappearing
  from the interpolation coordinate set;
- interpolation across a gap being reclassified when it exceeds the declared
  ordinary support limit;
- extrapolated and prior-only values remaining distinguishable from ordinary
  support;
- zero effective weight always producing an inactive objective sample.

### Synthetic calibration tests

Construct small connected arrays in which:

1. one gain solution is missing at a bracketing time;
2. the raw target visibility remains valid;
3. conservative linear transfer marks the affected products unsupported;
4. a bounded interpolator or GP recovers the known gain;
5. posterior uncertainty grows with anchor distance;
6. recovery improves held samples but an over-flexible alternative is rejected.

Add separate tests for:

- a hard-invalid antenna that can never be promoted;
- one invalid receptor affecting only its dependent correlation products;
- a missing D term preserving eligible parallel hands under the CASA-compatible
  operator but invalidating cross-hands;
- exact \(2\times2\) application requiring a complete valid coherency;
- a two-state phase process where a smooth GP performs worse than a change-point
  model.

### Fixed 3C391 regression tests

Using the saved flag versions and calibration tables, assert:

- `sl1mjax_pristine` has zero effective flags;
- ea05, ea13, and ea15 are fully excluded in
  `sl1mjax_calibration_input`;
- the calibration-input flag fraction is 28.2236%;
- diagonal apply adds 13,119,232 cells and reaches 34.2856%;
- those added cells occur in the expected target scan/antenna groups;
- ea04 has no rows in scan 103;
- G84 and Df are invalid for ea04 while Kcross and Xf are valid;
- polarisation apply adds 6,200,960 RL/LR cells involving ea04 and reaches
  37.1509%;
- the conservative derived mask reproduces the present fixture selections.

Count both cells and rows in reports. Always label which unit is being used.

### End-to-end acceptance tests

A recovery model passes only if:

- the original trusted cohort does not regress beyond a declared tolerance;
- held calibrator residuals or closure improve;
- held target baselines, times, and pointings improve consistently;
- gains remain within the calibrator-informed uncertainty envelope unless the
  evidence supports a departure;
- the result is stable across more than one deterministic partition;
- polarization recovery does not move the 3C286 (V=0) or EVPA anchors beyond
  their stated uncertainty;
- rerunning from the same raw data and manifest gives the same policy and
  counts.

Training loss on the newly admitted cohort is not an acceptance metric.

## Reporting requirements

Every calibration or imaging run should eventually report a quality waterfall:

```text
recorded
  - acquisition/hard exclusions
  - reduction-policy exclusions
  - calibration unsupported
  - uncertainty-policy exclusions
  - analysis quarantine
  = active samples
```

Break the waterfall down by field, scan, antenna endpoint, channel,
correlation, and reason. For full polarization, report parallel and cross-hands
separately.

Also report candidate recovery outcomes:

- number of cells and rows considered;
- number supported, extrapolated, retained in quarantine, and promoted;
- effective sensitivity added after weighting;
- validation change on the original trusted cohort;
- validation change on the candidate cohort;
- calibration posterior uncertainty and distance from anchors.

## Risks and controls

### Sky structure can masquerade as bad data

The 3C391 short-baseline residual tail is the main warning. Residual size alone
must not hard-flag data. Use sky-protection tests, baseline holdouts, and
component alternatives.

### Target self-calibration can manufacture support

An unconstrained target gain can fit the candidate visibilities and then claim
that they validate the gain. Keep calibrator anchors in the objective and use
independent partitions. Differential R/L and leakage freedom remain frozen in
the first recovery experiment.

### A smooth GP can hide state changes

GP smoothness is appropriate for slowly varying residual gains. It is not a
generic repair tool. Compare it with an unsmoothed or change-point alternative,
especially for ea05-like phase switching.

### Provenance can be lost during export

NPZ fixtures and Zarr blocks currently expose effective flags more readily
than their reasons. Version the quality schema and require round-trip tests.
Exports that cannot represent the full model must state which policy produced
their boolean flag.

### Quality tensors can become too large

Keep antenna-term coverage at native solution coordinates, keep broad reasons
as sparse selectors, and materialise product masks per streamed block. Store a
dense per-sample reason mask only for genuinely irregular decisions.

## Relationship to other plans

This proposal supplies the quality and provenance layer required by the
calibrator-anchored self-calibration target in
[`jones-polarization-calibration.md`](jones-polarization-calibration.md). The
anchored GP can turn some `calibration_unsupported` samples into
`calibration_extrapolated` samples, but this document defines the evidence
needed before they become `active`.

It also provides the calibration gate required by
[`time_frequency_sky_model_handoff.md`](time_frequency_sky_model_handoff.md).
Time- or frequency-dependent sky discovery must know whether a residual comes
from trusted data, a calibration extrapolation, or a quarantined cohort.

The existing residual modes remain useful. They become one evidence source in
the wider quality model rather than replacements for calibration validity or
acquisition provenance.

## Immediate next step

Build Phase 1 as a read-only audit with no change to imaging or calibration
results. The first deliverable is a versioned JSON report and tests that
reproduce the 3C391 provenance, flag-version differences, missing antenna
coverage, and term attribution listed above.

Only after that report is stable should the effective-mask policy enter the
calibration apply path. The first actual recovery experiment should be one
ordinary time-gain gap, not a known hardware fault and not ea04 leakage.
