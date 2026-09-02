# Beam-aware imaging development plan

## Purpose

The detailed VLA beam work has exposed a limitation in the current Stokes-I
image representation. SL1MJax can now evaluate time-dependent scalar,
diagonal, and experimental full-Jones voltage beams. The current voltage-beam
diagnostics nevertheless treat every fitted image component as a point at its
centre. The original quadtree operator retains the finite square Fourier
kernel, but pulls the scalar beam outside the pixel integral.

Neither approximation is adequate as the final basis for comparing an Airy
beam with a model containing measured main-lobe structure, squint, nulls,
sidelobes, and leakage. A detailed beam can lose because it is being combined
with the wrong numerical representation of the sky. Conversely, a flexible
sky can absorb an incorrect beam.

The next development stage therefore returns to Stokes-I imaging. It will
build a beam-aware finite-pixel operator, replace the overlapping coarse image
with a controlled wide-field hierarchy, reconstruct the sky under a sealed
validation protocol, and then compare beam families fairly. Full-Jones
polarisation imaging resumes only after this foundation is accepted.

This document turns
[`beam-aware-pixel-proposal.md`](beam-aware-pixel-proposal.md) into a phased
development and validation plan.

## Outcome

The target is a JAX imaging path with these properties:

- one shared celestial Stokes-I sky is used by all mosaic pointings;
- every finite sky component retains its declared shape and integrated flux;
- the antenna beam is integrated across finite components to a measured
  numerical tolerance;
- integration refinement adds no fitted sky freedom;
- sky refinement adds fitted freedom only after held-out evidence;
- known bright outer sources remain explicit catalogue components;
- a sparse, non-overlapping outer guard detects an inadequate field of view;
- Airy, Perley-plus-Airy, and CASSBEAM beam families receive comparable image
  fitting and topology-selection opportunities;
- model choice uses nested validation and a sealed outer test;
- calibration and polarisation work resume from the accepted Stokes-I model.

The plan does not assume that CASSBEAM must win a particular 3C391 metric. It
does require that the detailed beam be tested with a numerically appropriate
sky representation and a fair fitting protocol.

## Executive decision

Pause further scientific interpretation of the current fixed-sky beam ranking
and regional polarisation results. Preserve those results as regression and
diagnostic baselines.

Proceed in this order:

1. freeze the completed beam evaluator, convention tests, and existing
   diagnostic artifacts;
2. preserve finite component shapes in the streamed voltage operator;
3. implement fixed-depth subcell integration as the numerical reference;
4. implement a static-shape JAX integration plan and convergence audit;
5. construct a prefix-free wide-field sky with a refinable centre and a coarse
   outer guard;
6. rebuild Stokes I with sky splitting selected on held-out data;
7. compare beam families using equivalent topology-selection protocols;
8. repeat calibration and flag audits against the accepted sky;
9. resume diagonal and full-Jones polarisation imaging;
10. transfer the complete protocol to an independent C-band observation.

Do not introduce frequency-dependent or time-variable sky coefficients during
the initial reconstruction. The beam may still depend on exact time and
frequency. This isolates spatial sky structure from instrumental structure.

## Current evidence and limitations

### Existing Stokes-I sky

The current central sky uses a 104 by 104 root grid with 16 arcsec cells over a
27.73 arcmin square. Its frozen topology contains 10,591 leaves at 16 arcsec,
885 at 8 arcsec, and 60 at 4 arcsec before zero-flux leaves are removed from
the transfer diagnostic.

The composite model adds:

- a separate 64 by 64 grid of 60 arcsec positive pixels over 64 arcmin;
- three selected outer catalogue atoms in the documented composite run;
- extended Airy support for bright sources outside the central field.

The catalogue components explain most of the reproducible wide-field gain.
After they are present, the coarse field provides a smaller but consistent
sealed-test improvement of about 0.52%. This is evidence for a broad residual
component. It is not evidence that two overlapping positive images are the
best final basis.

### Existing voltage-beam comparison

The JAX voltage operator has passed its point-source prediction gate. In the
fixed Airy-fitted sky comparison, static Airy had the lowest mean held-out
loss. CASSBEAM diagonal was slightly better only on C1 and worse on C2--C7.
Full Jones and diagonal were nearly identical because the native fixture has
RR and LL but no cross-hand loss. Perley-plus-Airy was worst in the aggregate.

This comparison remains useful. It shows that the current detailed beam does
not transfer automatically to an Airy-fitted point-centre sky. It does not
show that the Airy beam is the best physical C-band model.

The experiment discarded the 4, 8, 16, and 60 arcsec component widths. It also
gave Airy the advantage of using fluxes and topology selected under an Airy
operator. These limitations must be removed before beam selection.

### Existing polarisation evidence

Regional Q/U freedom has improved some held-out tests, but the largest fitted
fractional polarisation values occur in faint off-axis cells. This is the
pattern expected when a spatial sky model absorbs an omitted direction-
dependent response. The present data also provide limited parallactic-angle
leverage.

The polarisation result therefore motivates the beam-aware imaging work. It
does not justify increasing Q/U/V freedom before the Stokes-I and beam basis is
settled.

## Development principles

### Separate scientific and numerical hierarchies

The sky hierarchy contains fitted coefficients. Splitting one sky leaf into
four free children adds spatial degrees of freedom. Such a split requires
training evidence, a complexity cost, a global refit, and held-out acceptance.

The integration hierarchy contains tied numerical samples. Dividing one
uniform square into subcells changes only how its response is calculated. All
subcell weights remain deterministic functions of the parent flux. This
refinement requires a numerical convergence test, not validation-set
selection.

The optimizer gradient must have one entry per fitted sky coefficient, not one
entry per integration node.

### Keep model selection out of the sealed test

Beam family, sky topology, regularization, integration tolerance, stopping
rule, and optimizer settings must be fixed before the outer sealed fold is
opened. Inner folds may select those choices.

The integration planner must not inspect measured validation residuals. It may
use geometry, beam values, numerical prediction differences, declared data
weights, and a training-derived sky amplitude estimate.

### Compare complete pipelines fairly

A detailed beam must not be judged only with an Airy-selected image. Airy must
also not be judged only with a CASSBEAM-selected image.

Each candidate beam should receive the same topology-selection budget, split
and merge protocol, optimizer convergence standard, and validation folds. A
common-topology refit should be retained as an attribution diagnostic, but the
primary scientific comparison is between complete, equally constrained
pipelines.

### Preserve known outer sky

Known bright outer sources remain explicit catalogue atoms. Their positions
and provenance remain fixed. Their non-negative fluxes may be fitted.

An extended source should use a declared Gaussian or multi-component shape
when evidence supports it. It must not be represented as a large square only
because it lies outside the central field.

### Make field-of-view failure observable

The outer guard is a sentinel for missing sky and an inadequate refinable
field. It is not a permanently overlapping second science image.

Persistent structured evidence in an unsplittable outer root should produce a
recorded escalation: expand the refinable field and start a new model-selection
run. It should not silently alter the sealed experiment.

### Preserve a slow reference

Adaptive moments, caching, interpolation, and batching must be checked against
a clear subcell reference. The reference does not have to be the production
optimizer path. It must be practical on representative subsets and callable
from tests and scientific audits.

## Target architecture

### Component table

Every sky component should carry:

- stable component identifier;
- component family: central tree, outer guard, or catalogue;
- basis type: uniform square, delta, Gaussian, or another declared shape;
- celestial centre;
- width or shape parameters;
- integrated Stokes coefficients;
- quadtree level, parent, and active-leaf status where applicable;
- whether scientific splitting is permitted;
- catalogue and topology provenance.

The component table is independent of pointing, time, frequency, and beam.

### Integration plan

The integration planner expands finite components into deterministic numerical
nodes. A packed plan should contain at least:

- `parent_index` for each node;
- celestial node coordinates;
- fixed parent-flux weight;
- subcell width and basis type;
- quadrature order or moment order;
- applicable measurement-block index;
- diagonal and off-diagonal validity information;
- numerical error estimate;
- reason for refinement or fallback.

The plan is created outside JIT and frozen for one optimization run.

### JAX execution

The jitted operator gathers parent coefficients at the integration nodes,
multiplies by fixed weights, evaluates the required Jones matrices, applies
the finite subcell kernel, and reduces node gradients back to their parents.

To control compilation:

- use a small set of quadrature orders such as 1, 4, 16, and 64 nodes;
- group or pad node batches into fixed capacity buckets;
- mask unused entries;
- pass time, frequency, PA, pointing, antenna, and coordinates as array values;
- stream fixed-size node tiles through the existing unique-time scan;
- compile separately only for structural changes such as beam family,
  correlation layout, dtype, or capacity class;
- rebuild and recompile only outside an inner optimization run.

### Proposed wide-field hierarchy

The preferred candidate is a 64 by 64 grid of 64 arcsec outer roots. Its total
width is about 68.27 arcmin. The central 26 by 26 roots can be refined twice to
16 arcsec, matching the existing 104 by 104 central geometry because

\[
26\times64\text{ arcsec}=104\times16\text{ arcsec}=1664\text{ arcsec}.
\]

The central roots are refinable under the existing split and merge protocol.
Outer roots are normally unsplittable during a reconstruction. The active leaf
set remains prefix-free, so a 64 arcsec parent is not fitted alongside its 16,
8, or 4 arcsec descendants.

The exact outer extent and central refinable boundary remain validation
choices. The 64 arcsec geometry is the first candidate because it nests the
existing central scales exactly. It is not accepted solely for convenience.

### Sparse outer guard

Known catalogue atoms are always evaluated. Outer guard roots start as virtual
candidates or zero-flux inactive leaves where the implementation permits it.

At defined topology rounds, one batched residual adjoint scores all guard
roots. Only supported roots are admitted to the fitted model. An admitted
outer root remains coarse within that run.

The guard reports a field-of-view escalation when any of these occurs:

- stable non-zero flux appears close to the refinable boundary;
- a virtual child contrast has strong and repeatable support;
- held-out residual structure persists inside the root;
- activating the root materially changes central fluxes;
- the root contributes significantly across several pointings;
- several adjacent roots indicate coherent extended structure.

An escalation starts a new run with a larger refinable region. It does not open
the sealed fold or modify an already sealed model.

## Data and validation protocol

### Fold roles

Use at least three logical levels:

1. **fit folds** optimize sky coefficients;
2. **inner validation folds** select topology, beam family, integration policy,
   regularization, and stopping rules;
3. **sealed outer fold** evaluates the final predeclared pipeline once.

Where data volume permits, rotate the inner fit and validation roles so a sky
split or beam preference must transfer across more than one partition. Keep
complete time bins or UV cells together according to the question being
tested.

Pointing-held-out, channel-held-out, baseline-held-out, and time-held-out tests
answer different questions. The primary split must be declared for each phase.
Secondary stratified scores should be reported without turning them into many
uncontrolled selection opportunities.

### Initial data resolution

The coarse 60-second and 32-MHz fixture may be used for inexpensive spatial
topology discovery. The beam should be evaluated consistently with the
averaged measurement rather than merely at an arbitrary bin centre if the
within-bin difference is significant.

The accepted spatial model must transfer to the finer time and frequency data
before temporal or spectral sky freedom is introduced. This prevents averaging
error from becoming part of the static sky.

### Calibration state

Use one fixed, documented direction-independent calibration during the first
beam and topology comparison. Do not let self-calibration vary between beam
candidates.

After the Stokes-I sky and beam are selected, repeat the calibration
complexity, interpolation, and flag-recovery audits. This determines which old
calibration residuals were caused by missing sky or beam structure.

## Phase 0: checkpoint the beam and imaging baselines

### Work

Create a versioned development manifest containing:

- the current Airy, Perley-plus-Airy, CASSBEAM diagonal, and experimental
  CASSBEAM full-Jones artifacts;
- feed-frame and `casa_parang_true` convention choices;
- CASSBEAM frequency support and interpolation rule;
- beam raster origin, squint diagnostics, and support masks;
- current JAX point-operator tests;
- current fixed-sky beam report;
- current composite sky checkpoint and component counts;
- current folds, flags, weights, calibration state, and channel averaging;
- the unimplemented or unfrozen CASA/full-Jones gates.

Do not freeze the experimental full-Jones scientific model merely to begin
Stokes-I development. Preserve its refusal in production factories and use an
explicit experimental flag in diagnostics.

### Tests

- Re-run the focused voltage-beam convention tests.
- Re-run the JAX-versus-NumPy point-source tests.
- Reproduce the current fixed-sky Airy and detailed-beam totals.
- Confirm that RR/LL-only data do not produce an RL/LR score.
- Confirm that the production full-Jones factory still refuses the unfrozen
  artifact.

### Gate 0

Proceed when all baseline artifacts are reproducible from a machine-readable
manifest and no scientific status has been silently upgraded.

## Phase 1: introduce a basis-preserving sky contract

### Work

Replace the voltage diagnostic's flattened `(l, m, flux)` representation with
the component table described above. Preserve widths for every central and
coarse square. Preserve catalogue atoms as deltas.

Add explicit conversion from the existing quadtree and composite checkpoints.
The conversion must report dropped zero-flux leaves separately from missing
components. It must preserve total intrinsic flux to numerical precision.

Keep the old point-centre path as a named diagnostic mode.

### Tests

- Round-trip component identifiers, centres, widths, types, and fluxes.
- Verify total and per-family flux before and after conversion.
- Verify the observed counts of 4, 8, 16, and 60 arcsec components.
- Verify catalogue atoms retain delta basis and exact positions.
- Reject unknown basis types and invalid widths.
- Reject a component table in which a quadtree parent and its children are all
  active unless the family explicitly declares a non-quadtree basis.

### Gate 1

Proceed when the voltage path can consume the sealed sky without discarding
finite component shapes and the old point-centre mode remains reproducible.

## Phase 2: build the finite-pixel reference operator

### Work

Implement a clear NumPy reference for uniform squares with:

- the analytic parent square kernel;
- 2 by 2 tied subcells;
- 4 by 4 tied subcells;
- an extensible maximum depth;
- scalar, diagonal, and full-Jones beam evaluation;
- per-node diagonal and off-diagonal validity;
- exact celestial-to-pointing coordinate transformation.

Each subcell uses its own finite square kernel. It is not a delta sample. Child
weights sum to the parent flux, and child responses sum to the parent response.

### Unit tests

- A constant scalar beam makes every subdivision depth reproduce the analytic
  parent square response.
- A constant non-trivial Jones produces the same identity for RR, RL, LR, and
  LL.
- Subcell centres tile the parent without gaps, overlap, reflection, or axis
  exchange.
- A vanishing square width approaches the delta kernel.
- Parent flux is invariant with depth.
- Non-zero `w` geometry follows the existing square-kernel convention.
- A node outside beam support contributes zero without invalidating valid
  siblings.
- Off-diagonal-invalid nodes do not become known zero leakage.

### Manufactured-beam tests

Use constant, linear, and quadratic complex beams whose square integrals are
known from analytic moments or a much finer oracle. Cover:

- amplitude gradients;
- phase gradients;
- diagonal R/L displacement;
- complex off-diagonal leakage;
- opposing pointing offsets;
- non-zero parallactic rotation.

### Gate 2

Proceed when constant-beam tiling agrees with the analytic square response at
the declared floating-point tolerance and manufactured beams show convergence
with subdivision depth.

## Phase 3: extend the streamed JAX operator

### Work

Implement packed integration nodes with a parent-index gather and reduction.
Integrate them with the unique-time Jones evaluation and direction tiling in
`predict_voltage_beam_jax`.

Provide fixed-capacity node buckets and row buckets. Generate masks outside
JIT. Freeze the plan during each value-and-gradient or proximal-SGD run.

Support these initial execution modes:

- delta catalogue atoms;
- centre-factorized finite squares;
- fixed 2 by 2 subcells;
- fixed 4 by 4 subcells;
- mixed orders in one packed plan.

Do not add moment acceleration in this phase.

### Tests

- JAX matches the NumPy reference in 64-bit mode for every execution mode.
- Mixed component widths match separate per-width calls.
- Mixed delta and square components match the sum of their separate calls.
- Parent gradients match finite differences.
- Integration-node count does not change gradient dimension.
- The complex dot-product adjoint identity passes.
- Single-hand RR and LL blocks still use the complete receptor Jones.
- Changes in time, channel values, PA, pointing IDs, and antenna IDs do not
  trigger shape-dependent Python branches.
- Padding and masks do not change values or gradients.
- Memory preflight refuses an over-budget materialization before allocation.
- Streamed and materialized reference paths agree where both fit.

### Performance tests

Measure compile time, peak memory, and execution time against:

- number of fitted parents;
- number of integration nodes;
- quadrature order;
- row and time counts;
- channel count;
- scalar, diagonal, and full-Jones beams.

Keep a benchmark representative of one 3C391 pointing and a smaller CI
benchmark. Record recompilation count as well as elapsed time.

### Gate 3

Proceed when the JAX operator matches the reference, gradients pass, memory is
bounded, and one frozen integration plan runs without recompilation inside an
optimization loop.

No absolute speed target is imposed before measurement. The full 3C391 path
must nevertheless remain practical enough to run the validation matrix.

## Phase 4: build the numerical integration planner

### Work

Build the plan outside JIT. Compare successive integration depths on a
stratified subset of real observation geometry. The planner should use both a
relative and absolute visibility-error threshold:

\[
\lVert V^{(d+1)}-V^{(d)}\rVert
<\epsilon_{\rm abs}+
\epsilon_{\rm rel}\lVert V^{(d+1)}\rVert.
\]

The initial implementation may assign one conservative order to each
component and pointing. The production representation may vary order by a
measurement regime containing pointing, PA range, channel range, and baseline
range.

Do not require a pixel to use its worst order globally unless fixed-shape
bucketing makes that trade-off worthwhile. A leaf can use one sample on axis
and more samples near a null in another pointing.

Use apparent contribution in the absolute error calculation. A faint,
strongly attenuated leaf should not consume unlimited work solely because the
fractional beam gradient is large.

Force conservative subdivision near:

- beam nulls and phase reversals;
- CASSBEAM raster boundaries;
- outer-field model handovers;
- frequency-node handovers;
- diagonal or off-diagonal validity boundaries;
- strong squint or leakage gradients.

### Tests

- Compare depths 0, 1, 2, and 3 for representative 4, 8, 16, and 60 arcsec
  components.
- Cover main-lobe centre, half power, pointing ring, nulls, sidelobes, and
  support boundaries.
- Cover low, middle, and high channels.
- Cover short, middle, and long baselines.
- Cover representative PA bins and both sides of the pointing centre.
- Verify the plan is unchanged when measured validation visibilities change.
- Verify increasing the requested tolerance cannot reduce the assigned order
  in the wrong direction.
- Verify a post-fit audit detects deliberately under-resolved plans.

### Gate 4

Proceed when the reference depth is converged over the declared beam support
and the planner's maximum error is small relative to both:

- the held-out loss differences used to compare beams;
- the residual change used to accept sky splits.

Freeze the selected tolerance before scientific beam comparison.

## Phase 5: introduce the prefix-free wide-field sky

### Work

Construct the 64 arcsec candidate root grid. Map the central 16, 8, and 4
arcsec geometry into its nested central region. Do not initially reuse fluxes
that have seen the new sealed fold.

Create three disjoint component roles:

1. refinable central hierarchy;
2. coarse outer guard roots;
3. explicit catalogue atoms.

Remove the overlapping central-plus-60-arcsec representation from this new
experimental path. Keep it available as an ablation baseline.

Implement batched virtual scoring for inactive guard roots. Activation adds a
coarse fitted coefficient but does not permit splitting during the same sealed
run.

### Geometry tests

- The active quadtree is prefix-free.
- The central footprint matches the intended 104 by 104, 16 arcsec geometry.
- Root, child, and celestial coordinate conventions agree exactly.
- No gap or overlap exists between central and outer tree leaves.
- Catalogue atoms do not alter tree prefix rules.
- Rendering conserves integrated flux across all levels.
- Pointing transforms keep one shared celestial position across all fields.

### Sentinel tests

Use synthetic datasets containing:

- no outer source;
- a known catalogue point source;
- one diffuse coarse outer component;
- a source just outside the refinable boundary;
- structured emission that cannot be represented by one outer root;
- a beam sidelobe error with no outer sky source.

The guard should remain inactive for the empty case. It should activate the
correct region for a supported coarse component. Structured outer emission
should raise a field-expansion signal rather than being mistaken for an
accepted final coarse model. A beam-only mismatch should be distinguishable
through pointing or PA behaviour where the synthetic geometry provides that
leverage.

### Gate 5

Proceed when the new sky is prefix-free, catalogue sources remain recoverable,
the empty guard has controlled false activation, and the synthetic
field-expansion cases are detected.

## Phase 6: reconstruct Stokes I with fixed beam candidates

### Work

Start with spatially constant, unpolarised Stokes I. Keep calibration, flags,
weights, time averaging, frequency averaging, catalogue selection, and fold
definitions fixed.

For each initial beam candidate:

- calculate and freeze its numerical integration plan;
- fit the same starting root geometry;
- screen sky splits in batches with the residual adjoint;
- exactly rescore a marked shortlist with the beam-aware operator;
- propose a ranked bulk split batch;
- warm-refit all active fluxes;
- accept the batch only if the declared inner validation improves;
- try smaller prefixes when the full batch fails;
- apply the existing merge hysteresis and complexity penalty;
- audit the integration plan after accepted topology rounds.

If a sky split changes leaf widths, generate the required integration nodes
before the next fit. This is a numerical planning step and does not inspect
validation residuals.

The first candidates should be:

1. static Airy;
2. Perley-plus-Airy composite;
3. CASSBEAM diagonal co-polar response and squint.

Experimental full Jones may be run as a diagnostic. It is not a production
Stokes-I candidate until its scientific freeze gate is satisfied. With RR/LL
only, it provides little leverage over the diagonal form.

### Batched split tests

- The batched residual screen matches individual candidate calculations on a
  small problem.
- Exact shortlist scores match explicit virtual-child predictions.
- Bulk acceptance matches a serial reference for a small candidate set.
- Rejected batches backtrack to ranked prefixes deterministically.
- Split and merge operations preserve prefix-free topology.
- Candidate ordering is stable within declared numerical tolerance across
  node-bucket sizes.
- No integration node is treated as a free child in split counts or penalties.

### Scientific diagnostics

Report by beam and topology round:

- train and inner-validation loss;
- leaf counts by width;
- active outer roots and catalogue fluxes;
- total intrinsic flux by component family;
- integration nodes and order distribution;
- split scores and accepted prefixes;
- merge decisions;
- loss by pointing, channel, time/PA bin, baseline bin, and hand;
- flux near the refinable boundary;
- field-expansion warnings;
- optimizer convergence and KKT diagnostics;
- compile count, peak memory, and elapsed time.

### Gate 6

Each beam pipeline must reach its stopping rule without using the outer sealed
fold. It must pass numerical integration audit and optimizer convergence
checks. No unresolved outer-guard escalation may be ignored.

If the guard requests a larger field, change the geometry and restart the
inner model-selection protocol. Do not patch the existing sealed run.

## Phase 7: compare beam families fairly

### Primary comparison

Treat each beam plus its validated sky topology as one complete candidate
pipeline. Give every pipeline the same:

- initial geometry;
- catalogue prior;
- split and merge budget;
- topology complexity penalty;
- optimizer family and convergence rule;
- inner-fold schedule;
- integration accuracy requirement;
- compute budget, or a declared convergence-equivalent budget.

Select the pipeline using only inner validation. Then open the outer sealed
fold once for the selected pipeline and its predeclared controls.

### Attribution comparison

Form a common prefix-free topology from the union of inner-validated beam
topologies. Reapply split/merge validity so this union does not receive free
unpenalized complexity. Refit each beam on that common topology.

This comparison answers whether a loss difference comes mainly from beam
response or from a topology that one beam caused the discovery process to
select. It is diagnostic and does not replace the complete-pipeline result.

### Required ablations

For each beam, compare:

- point-centre components;
- finite square with centre-factorized beam;
- converged beam-aware integration;
- central plus catalogue only;
- central plus active outer guard plus catalogue;
- the old overlapping 60 arcsec composite as a historical control;
- fixed original topology and newly selected topology.

Report paired changes by pointing. The opposite C4/C5 behaviour and the C1
on-axis result should remain visible rather than being averaged away.

### Gate 7

A production Stokes-I beam candidate must:

- pass all numerical and convention gates;
- improve or remain competitive under the complete-pipeline inner validation;
- avoid unexplained large losses in individual pointings or hands;
- transfer its selected topology to the sealed fold;
- produce stable catalogue and central fluxes across inner rotations;
- have no unresolved field-of-view warning;
- state clearly which physical beam features are supported and which remain
  experimental.

A detailed beam that fails this gate is not discarded. Its residual pattern is
returned to beam convention, interpolation, artifact, or antenna-variation
diagnosis. Airy winning the gate would be an engineering baseline, not a claim
that an ideal Airy pattern is the physical VLA beam.

## Phase 8: optimize integration without changing the result

### Work

After the reference pipeline is accepted, implement quadratic beam moments or
another accelerated integral in smooth beam regions. Retain subcell
integration near non-smooth features.

The moment terms remain deterministic functions of the parent sky coefficient
and beam. They do not become image gradient or curvature parameters.

Profile reuse across:

- common leaf widths;
- pointings;
- time or PA blocks;
- channel blocks;
- antenna planes;
- repeated integration nodes.

Consider a persistent cache only after its complete coordinate key and measured
memory cost are known.

### Tests

- Constant, linear, and quadratic manufactured beams match their analytic
  moment order.
- The accelerated operator matches converged subcells over its declared
  smooth support.
- Nulls, raster edges, and handovers force reference fallback.
- Values, gradients, and validity masks match the reference.
- The accepted image and held-out ranking remain unchanged within the declared
  numerical tolerance.
- Compile count does not grow with individual leaf decisions.

### Gate 8

Adopt acceleration only when it reduces measured cost without changing the
accepted scientific decisions. Keep a command-line mode that forces the
reference integrator.

## Phase 9: repeat calibration and flag audits

### Work

Freeze the accepted spatial sky and beam. Repeat the calibration experiments
that previously followed the composite Airy image:

- gain time complexity;
- interpolation versus native solution epochs;
- GP-prior calibration when ready;
- held-out baseline and time transfer;
- potentially recoverable visibility cohorts;
- residual outlier and hardware-flag classification.

Do not add sky variability in this phase. The purpose is to measure how much
of the former calibration residual was caused by the spatial sky and beam
model.

### Tests

- Reproduce the old calibration result with the historical sky.
- Apply identical folds and scoring to the accepted sky.
- Confirm direction-independent calibration cannot improve only by absorbing
  pointing-dependent beam residuals.
- Report recovered and worsened visibility cohorts separately.
- Preserve dead-data flags and missing samples.
- Require calibration choices to transfer across held-out times and baselines.

### Gate 9

Accept a revised calibration only when it improves the declared held-out
cohorts without degrading calibrator anchors or producing implausible antenna
solutions. Update the calibration prior and flag proposals with the result.

## Phase 10: resume polarisation imaging

### Work

Begin with calibrated RR, RL, LR, and LL data and the accepted Stokes-I sky.
Use the beam-aware operator for every correlation.

Follow this model ladder:

1. diagonal beam with I only;
2. full-Jones beam with I only, measuring instrumental cross-hand prediction;
3. global constant Q/U with V fixed to zero;
4. global constant V as a separate activation test;
5. one global RM if channel-held-out evidence supports it;
6. regional Q/U or V only after the global and beam terms are accepted.

Keep the production freeze rule for full Jones until the required convention
and independent-reference gates pass. Experimental runs must remain clearly
labelled.

### Tests

- Full-Jones NumPy and JAX predictions agree for all four correlations.
- The $e^{\pm2i\chi}$ off-diagonal phase convention is preserved through
  finite-pixel integration.
- Diagonal and off-diagonal validity propagate separately.
- An unpolarised synthetic sky with leakage does not produce fitted Q/U when
  the correct full-Jones beam is used.
- Injected Q, U, V, and RM are recovered without I-to-polarisation bias.
- Model activation transfers across held-out pointings, channels, baselines,
  times, and hands.
- The calibrator polarisation floor is reported before target activation.

**CASA `awp2` Stage 1 (2026-09-02):** Bacchus generated 18 default EVLA
ray-traced `.pb` planes (CASA 6.7.6.14; I/RR/LL at 4564 and 4692 MHz;
HA transit, +2 h, −2 h). Those products are frozen as a checksummed
scalar `.pb` reference in `src/sl1mjax/data/casa_awp2_oracle/`. That
freeze is not CASSBEAM acceptance. Split gates:

- `casa_awp2_scalar_core_compatible`: **pass** (I-only; centre, FWHM,
  RMS, and pointwise residual inside the 10% contour).
- `casa_awp2_scalar_5percent_equivalent`: **fail**. The I residual
  maximum is at 9.20–9.34 arcmin, near the CASSBEAM first null, where
  CASA has 5.9–6.3% power and CASSBEAM has 0.4–0.6%. Inside the 10%
  contour the maximum residual is 3.83–3.89%. Do not loosen the 5%
  tolerance.
- `casa_awp2_rrll_oracle_valid`: **false**. CASA I, RR, and LL planes
  are pixel-identical for every frequency and hour angle. The `.pb` is
  an image-domain PB/normalization product, not a per-receptor Jones.
- `casa_full_jones_convention_accepted`: **not_run**. Needs
  visibility-domain A-term tests or Perley holography.

`casa_awp2_accepted` and `diagonal_copolar_is_casa_accepted()` remain
false. Do not remove CASSBEAM squint to match these scalar `.pb`
files. Report: `outputs/casa_awp2_power_oracle/comparison.json`.

### Gate 10

Do not accept spatial polarisation unless it beats the beam-aware global model
on relevant held-out cross-hands and exceeds the calibrator systematic floor.
RR/LL-only evidence cannot freeze leakage or full Jones.

## Phase 11: transfer to fine resolution and an independent C-band dataset

### Work

Apply the accepted static sky, beam, integration, calibration, and
polarisation protocol to:

1. the finer time and frequency version of 3C391;
2. at least one independent C-band observation that did not determine the
   model choices.

Only after the fine-resolution static model transfers should SL1MJax activate
spectral or temporal sky terms.

Compare with an appropriate CASA reduction and, where available, CASA `awp2`
or another independent beam response. The comparison should include more than
one aggregate residual number.

### Tests

- Coarse-discovered spatial topology predicts fine-resolution holdouts.
- Channel and time residuals do not show patterns caused by averaging or beam
  interpolation.
- Source fluxes remain consistent across pointings and epochs within the
  declared calibration and beam uncertainty.
- The outer guard remains quiet or raises a reproducible field warning.
- Results are stratified by field position, baseline, channel, time, and hand.
- SL1MJax and CASA outputs are compared in image and visibility space.

### Gate 11

The beam-aware imaging path becomes the C-band default only when it transfers
to the independent observation and meets the predeclared comparison criteria.
This gate supports a claim about general C-band reduction rather than a single
tutorial dataset.

## Test matrix

### Continuous integration tests

Keep small deterministic tests for:

- component-table validation;
- parent/subcell flux conservation;
- constant-beam square identity;
- manufactured scalar and Jones beams;
- mixed delta and square components;
- NumPy/JAX agreement;
- gradients and adjoints;
- padding and masks;
- per-node validity;
- prefix-free geometry;
- virtual guard scoring;
- split batching and prefix backtracking;
- full-Jones correlation packing.

These tests should run without CASA or a real Measurement Set.

### Golden tests

Maintain compact frozen artifacts for:

- mixed-width scalar prediction;
- diagonal squint at several pointings and PA values;
- full-Jones off-diagonal phase and validity;
- one small mosaic with a shared finite sky;
- one short optimization with an accepted sky split;
- one outer-guard activation case.

Golden files must record conventions, units, shape order, normalization, and
artifact hashes.

### Optional real-data tests

Provide explicit environment-gated tests for:

- the 3C391 coarse fixture;
- native four-correlation calibrated data;
- CASA beam or imaging products;
- the independent C-band transfer dataset.

An unavailable external dataset should skip clearly. A malformed or
incompatible dataset should fail rather than skip silently.

## Scientific reporting

Every phase that uses real data should write a JSON report and enough arrays
to reproduce the summary plots. At minimum record:

- code revision and command line;
- input and artifact hashes;
- component geometry and flux by family;
- fitted parameter count;
- beam model and convention;
- integration mode, tolerance, and node counts;
- calibration, flags, weights, and fold identities;
- topology proposals, acceptances, and rejections;
- train, inner-validation, and sealed loss roles;
- loss by pointing, channel, PA/time, baseline, and hand;
- optimizer convergence;
- compile count, elapsed time, and peak memory;
- outer-guard activations and field-expansion warnings;
- which gates passed, failed, or remain experimental.

Recommended plots include:

- image topology and leaf scale;
- active guard roots and catalogue sources;
- integration depth over the sky for each pointing;
- numerical error versus beam radius and leaf width;
- residual and loss deltas by pointing;
- flux movement between beam pipelines;
- split stability across inner folds;
- RR/LL and RL/LR residual summaries;
- compute cost versus fitted leaves and integration nodes.

## Resource controls

The plan contains several potentially expensive axes: fitted leaves,
integration nodes, pointings, times, channels, antennas, correlations, beam
families, topology rounds, and validation folds.

Control them explicitly:

- screen thousands of sky candidates with one batched adjoint;
- run full global refits only for bulk proposals and a few ranked prefixes;
- choose integration order numerically without fitting alternative skies;
- freeze one integration plan during each optimization;
- use coarse data for initial spatial discovery;
- run detailed real-beam convergence on stratified subsets;
- use fixed-capacity JAX buckets and streamed time groups;
- eliminate unsupported beam/topology candidates before sealed runs;
- reserve full four-correlation and fine-resolution work for accepted spatial
  candidates;
- preserve intermediate checkpoints so a failed gate does not require earlier
  phases to be recomputed.

Compute budgets should be stated per phase. A candidate must not win merely
because it received more optimizer steps. Prefer a common convergence standard
over a common iteration count.

## Stop and escalation conditions

Stop the current phase and diagnose before proceeding when:

- subcell integration does not converge over required beam support;
- JAX and the reference disagree beyond tolerance;
- changing integration order changes fitted parameter count;
- a beam convention test fails;
- the outer guard repeatedly requests a larger field;
- catalogue flux changes implausibly between nearby folds;
- topology is unstable across inner rotations;
- optimizer convergence differs materially between beam candidates;
- detailed-beam losses are concentrated in opposite pointings or receptor
  hands without an understood convention or artifact explanation;
- a calibration change improves fitted data but fails held-out transfer;
- polarisation improvement exists only in RR/LL while leakage is being claimed;
- the sealed fold has been inspected before the model-selection protocol is
  frozen.

Each stop should produce a small diagnostic experiment. It should not be
resolved by adding unrestricted sky, beam, or calibration parameters.

## Deliverables by milestone

### Milestone A: trustworthy finite-pixel operator

- basis-preserving component table;
- NumPy subcell reference;
- streamed JAX mixed-basis operator;
- convergence planner;
- unit, gradient, adjoint, memory, and performance reports.

This milestone completes Phases 0--4.

### Milestone B: trustworthy wide-field Stokes-I reconstruction

- prefix-free central and guard geometry;
- catalogue integration;
- virtual guard scoring and expansion warnings;
- beam-aware split and merge loop;
- inner-validated Airy, Perley, and CASSBEAM diagonal reconstructions.

This milestone completes Phases 5--6.

### Milestone C: accepted C-band beam and sky pipeline

- complete-pipeline beam comparison;
- common-topology attribution report;
- sealed 3C391 result;
- accelerated integration if justified;
- repeated calibration and flag audit.

This milestone completes Phases 7--9.

### Milestone D: polarisation and independent transfer

- calibrated four-correlation beam-aware imaging;
- global then regional polarisation ladder;
- fine time/frequency transfer;
- independent C-band observation report;
- CASA comparison.

This milestone completes Phases 10--11.

## Immediate implementation slice

The first code change should be deliberately smaller than the full image
reconstruction:

1. define the component table with delta and uniform-square basis types;
2. import the sealed central, coarse, and catalogue components without losing
   widths;
3. implement constant-beam parent-square, 2 by 2, and 4 by 4 reference
   predictions;
4. prove the exact constant-beam tiling identity;
5. add parent-index gathering and gradient reduction to the JAX operator;
6. compare NumPy and JAX on a small mixed 16 and 60 arcsec sky;
7. benchmark one pointing with Airy and CASSBEAM diagonal;
8. write the Phase 1--3 report before changing the sky topology.

The next slice can then construct the prefix-free outer guard and reconnect the
existing batched split machinery. Moment acceleration, self-calibration,
frequency-dependent sky, time variability, and new polarisation freedom remain
outside these first two slices.
