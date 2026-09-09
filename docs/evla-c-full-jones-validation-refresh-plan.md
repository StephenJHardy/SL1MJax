# EVLA-C validation and publication refresh: coding-agent handoff

## Objective

Establish what the corrected EVLA-C CASSBEAM model predicts in the THOL0001
SPW-4 observations. Compare its diagonal and full-Jones predictions directly
with measured complex visibilities. Regenerate the complete existing validation
figure set, executed notebook, rendered summary, and claim registry from those
results.

The work must answer three questions separately:

1. Is the coordinate, Jones, calibration, and prediction implementation correct
   on independent numerical checks?
2. How well does the updated diagonal model describe the observations, by
   region, antenna, and frequency?
3. Does the updated full-Jones prediction improve cross-hands, or can these
   observations constrain that prediction at all?

Success is a reproducible answer, including an inconclusive or negative answer.
It is not promotion of a production beam. Do not tune the analysis to obtain a
positive result.

## Starting position and authority

This plan follows the coordinate/feed comparison and the publication refresh
discussion of 2026-09-08. Inspect the actual tree and artifacts before acting.
There are substantial uncommitted changes. Preserve them and record the exact
staged source snapshot, not just HEAD. Do not commit or push without authority.

The existing CASA/JAX calibration goldens support calibration application on
their tested cases. They do not validate HOLORASTER coordinates or the beam.
The previous full-Jones non-detection used generic VLA parameters at commanded
offsets. It is historical evidence about that model, not a test of EVLA-C at
corrected source-in-beam coordinates.

The latest EVLA-C comparison is diagonal only and at channel 32. Its feed change
improves both ranking holdouts relative to generic CASSBEAM at the same corrected
coordinates. Do not confuse that comparison with EVLA-C versus the old
generic/commanded baseline. Do not carry generic-beam width or squint fits into
the new model as accepted coefficients.

SPW 4 and the previously opened C147 offset-ring partitions are development
data. Preserving local partitions remains useful, but does not make them new
independent validation after repeated inspection. SPW 5 remains sealed.

## Scope and restrictions

- Run on Bacchus in a new named staging tree and new named output directories.
  Inspect active jobs and available memory first. Do not interrupt other work.
- Read the existing scientific MS and calibration products. Do not overwrite
  them, re-solve DI calibration, or fit HOLORASTER into DI calibration.
- Use the audited EVLA-C parameters and named `source_lm_feed` coordinates.
  Keep generic/commanded products as historical controls.
- Do not search the 128-member convention ladder. Resolve definitions using
  source, metadata, and independent tests, not the best visibility loss.
- Do not add a width correction, squint fit, temporal GP, scan offsets, peeling,
  empirical leakage map, or array-average beam fit in this experiment.
- Do not open SPW 5, fit across it, or change the production full-Jones factory.
- Immutable numerical reference artifacts are allowed. A checksummed artifact
  does not mean its beam is scientifically accepted.

## Implementation map

Start by reading these files and their relevant tests. Reuse the existing
pipeline, but do not assume that shared helpers provide independent validation.

| Area | Files |
| --- | --- |
| Definitions | `docs/thol0001_holoraster_coordinate_and_feed_audit.md`, `src/sl1mjax/holography_holoraster_coordinates.py`, `src/sl1mjax/cassbeam_evla_c.py` |
| Beam generation and loading | `scripts/generate_cassbeam_highres_model.py`, `src/sl1mjax/cassbeam_highres.py`, `src/sl1mjax/holography_highres_cassbeam.py` |
| Direct forward model | `src/sl1mjax/holography_beam_prior.py`, `scripts/run_thol0001_coordinate_feed_comparison.py`, `scripts/run_thol0001_holoraster_cassbeam_comparison.py` |
| Offset-ring comparison | `scripts/run_thol0001_c147_offset_ring.py`, `src/sl1mjax/holography_c147_offset_ring.py` |
| Bundle and claims | `scripts/build_vla_c_band_beam_validation_bundle.py`, `src/sl1mjax/beam_validation_outputs.py`, `src/sl1mjax/beam_validation_statistics.py`, `src/sl1mjax/beam_validation_claims.py` |
| Publication | `src/sl1mjax/beam_validation_plots.py`, `scripts/write_vla_c_band_beam_validation_notebook.py`, `scripts/execute_vla_c_band_beam_validation_notebook.py`, `scripts/render_vla_c_band_beam_validation.py` |

Known locations to verify:

- Scientific MS:
  `/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms`
- Scientific reports:
  `/media/stephen/astro/vla/extracted/commissioning/validation/scientific/`
- Existing EVLA-C channel-32 plane:
  `/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_20260908`
- Publication:
  `notebooks/vla_c_band_beam_validation.ipynb`,
  `docs/vla_c_band_beam_validation.md`, and
  `docs/assets/vla_c_band_beam_validation/`.
- Current compact bundle: `src/sl1mjax/data/vla_c_band_beam_validation_v1/`.
  Use a new versioned bundle directory for this refresh.

## Phase 0 — inventory and freeze the experiment specification

Write `experiment_manifest.json` and an initial `status.json` before expensive
work. Inventory MS state, calibration manifests, residual-Jones products,
source models, beam planes, comparison arrays, existing figures, and tests.
Record hashes and links to the exact source definitions used.

The manifest must fix:

- SPW ID and native channel IDs, frequencies from `CHAN_FREQ`, correlation
  ordering, units, calibration state, and source-coherency policy.
- Coordinate and frame definitions, beam normalization, artifact hashes,
  masks, partitions, baseline orientation, and frequency support policy.
- Estimands, loss denominators, uncertainty clusters, random seeds,
  numerical tolerances, scientific decision rules, and injection design.
- Region and phase/SNR masks. Keep masks identical between model arms.
  Record separate diagnostic masks if needed; never silently reselect rows.
- An explicit list of read/write paths and resumable task IDs.

The complete publication frequency set is SPW-4 channels
**0, 8, 16, 24, 32, 40, 48, 56, 63**. This is nine native-channel comparisons,
not an all-64-channel joint likelihood. Derive their frequencies from the MS.
Unsupported or fully flagged channels must appear with a reason and counts;
do not revive them or replace them with nearby channels.

Use the existing channel-32 development partitions for paired continuity.
Also report the full descriptive raster sample where it has been used in the
publication. Do not compare the 538,489-row development subset directly with
the old 747,058-row full-raster metric without a matched-row rescore.

Gate: all dependencies and figure replacements are accounted for. Unknown
provenance is a blocker for the affected claim, not permission to guess.

## Phase 1 — bounded correctness audit and independent controls

Trace selected real rows through the complete forward model. Cover movers on
both baseline sides, several references, origin and off-axis directions, both
rasters, multiple times, and supported band-edge channels.

Audit these boundaries explicitly:

1. `DATA` versus scientifically corrected data; table order; no double apply;
   per-row source model; residual-Jones gauge and frequency coverage.
2. Commanded antenna displacement versus source direction in the moved beam.
   Verify the AZELGEO-to-beam map using metadata and source definitions. Check
   whether offsets are native angular offsets or projected direction cosines.
   Document approximation error at the outer raster, not only at boresight.
3. EVLA-C feed parameters, the single CASA feed-x sign conversion, taper as a
   function of frequency, raster indexing, axis order, and complex hand packing.
4. Receive/transmit definitions, on-axis matrix normalization, and the correct
   side of every matrix multiplication. Any boresight factor moved out of the
   beam must remain consistently represented in residual calibration.
5. Parallactic angle from the validated apparent-coordinate calculation;
   spatial rotation versus polarization-basis rotation; no double rotation.
6. Baseline reversal, complex conjugation, independent hand flags and weights,
   and finite-value/support masks. Sparse or empty `WEIGHT_SPECTRUM` must use
   the established per-row fallback, not corrupt or discard valid weights.
7. The offset-field geometric fringe for C147-*; HOLORASTER is not a substitute
   for testing the offset-field phase-centre geometry.

Implement an explicit, small NumPy calculation independent of the production
assembly helpers. Test asymmetric complex Jones matrices with known entries,
I, I+Q, I+U, and I+V sources, nonzero parallactic angles, and reversed baselines.
Include tests that deliberately omit a sign, conjugation, rotation, or fringe
and must fail. A self-consistent forward/inverse round trip is not enough.

Re-run applicable CASA/JAX fixtures. Retain their documented tolerances.
Set a tight float64 algebra tolerance for the independent assembly test before
running it; keep ray-tracer convergence tolerances separate. Reproduce the
existing EVLA-C diagonal channel-32 prediction on matched rows. A difference
must be explained and versioned before using new scientific scores.

Gate: independent algebra, metadata/frame checks, calibration fixtures, and
identity checks pass. If a real defect is found, add a failing regression test,
fix it, and record exactly which earlier figures and claims it supersedes.
Do not use a convention search or a relaxed tolerance to bypass the defect.

## Phase 2 — complete EVLA-C frequency artifacts

Inventory existing artifacts before generating anything. Generate missing
full complex 2x2 EVLA-C planes at each of the nine exact native frequencies.
Use frequency-dependent taper and all other audited physical parameters.

The generator currently has defaults that include additional frequencies.
Pass an explicit SPW-4 list. Inspect filename/frequency rounding and catalog
selection; a request for a native plane must not silently return channel 32.

Use the existing g1024/p32 artifact as the starting resolution, not proof of
convergence. Verify spatial support over the measured coordinates. Compare
with finer angular sampling and aperture integration at channel 32 and both
band extremes. Expand the convergence check if frequency or outer-field
behavior warrants it. Record interpolation error at off-grid measured cells.
Ensure numerical prediction differences are small relative to the claimed
scientific changes and injected cross-hand signals. Mark unresolved regions
as numerically limited rather than asserting convergence everywhere.

Store raw complex planes, normalized planes or reproducible normalization,
input files, geometry and executable hashes, frequencies, support masks,
normalization matrices, convergence results, and generation logs.

Gate: every publication frequency has a valid plane or an explicit blocked
status. No nearest-plane substitution or untested frequency scaling.

## Phase 3 — channel-32 direct comparison, then all nine channels

Use one shared data extraction per channel. Preserve row identifiers and
per-hand calibration/flag/weight state. Stream batches and save predictions
once so subsequent scoring and plotting never rerun the full MS calculation.

Primary arms:

- `evla_c_source_diagonal`: diagonal projection of the normalized EVLA-C beam.
- `evla_c_source_full_jones`: the same normalized beam with off-diagonals.

Use the same calibration, source, coordinates, and support in both arms. A
diagonal beam does not imply predicted RL/LR is identically zero: source
polarization and residual calibration remain in both predictions.

The first comparison uses the unit physical model. No fitted complex leakage
scale, training SNR mask that zeros the prediction, or empirical beam recovery.
No accepted width/squint correction is inherited from the generic artifact.

Retain generic/commanded and generic/source diagonal predictions as labelled
controls for the coordinate/feed impact figures. Reuse them only when hashes,
row IDs, source/calibration state, and support match. Otherwise recompute or
show them separately as historical data.

Export all four complex correlations, predictions from both primary arms,
full-minus-diagonal predictions, flags, weights, source I/coherency provenance,
actual frequencies, both named coordinates, baseline orientation, mover and
reference IDs, times, scan/pass identifiers, cells, regions, and split IDs.
Do not reuse the recent RR/LL-only publication export as a full-Jones input.

Score and plot channel 32 first. Then run the identical analysis over all nine
channels, keeping native visibilities separate. A scientific null is not a
reason to stop the other channels. A software/contract failure blocks the
affected calculation until fixed.

Metrics must distinguish:

- Complex residual power, with the exact weight and denominator formula.
- Its corresponding relative RMS; this is not automatically RMS divided by
  unattenuated source I if the power denominator is predicted/observed power.
- Absolute Jy residuals and median/RMS `|delta V| / I_model`.
- Complex correlation and slope, phase residuals with signal/support masks,
  coherent means, and noise/coherence diagnostics.
- RR-minus-LL contrast, per-hand scores, and paired
  full-minus-diagonal losses on the same samples.

Do not normalize errors by a noisy or near-zero attenuated visibility. Do not
interpret a biased outer-beam magnitude ratio as a multiplicative correction.
Report row counts per metric and hand, not global bin counts.

Preserve paired covariance in uncertainties. Use spatial cells and movers for
their respective comparisons; account for repeated channels and shared
antenna/reference structure when aggregating. Do not treat nine observations
of one row or millions of correlated visibilities as independent evidence.
Report individual movers and references alongside intervals.

## Phase 4 — cross-hand sensitivity and scientific classification

Predeclare the pooled cross-hand test across the nine sampled channels, plus
per-channel and per-hand diagnostics. Do not select favorable channels after
looking. Propagate the documented 3C147 Q/U uncertainty identically in both
arms. If a nuisance value is estimated, use training data only. Verify the
frequency support of residual-Jones terms; do not silently extrapolate a
channel-32 residual solution through the band.

Use the established paired-loss convention: candidate minus diagonal, negative
is better. For the primary development-support decision, require the paired
95% interval for pooled RL/LR loss to lie below zero on both the spatial and
mover ranking partitions. Require at least four of five held-out movers to
improve, and report RL and LR separately so opposite behavior is not hidden.
Use the existing main-lobe copolar non-regression limit
`max(0.002, 0.10 * L_main_diagonal)` for each hand, only after confirming that
the loss definition matches the earlier protocol. Report whole-raster
copolar changes as well. These are operational development criteria, not a
claim of independent confirmatory significance after repeated SPW-4 use.
If support or the audited loss definition makes them inapplicable, resolve
and document that issue before the scientific run, not after seeing scores.

Inject a known direction-dependent off-diagonal perturbation into copied
visibility arrays in the same calibration/gauge state. Define its amplitude
unambiguously relative to the EVLA-C unit model or source I, and preserve the
on-axis convention. Use zero, unit-model amplitude, and several bracketing
amplitudes, phases, and clustered-noise realizations with fixed seeds.

Separate training and scoring samples. Fit any recovery coefficient only on
training copies and evaluate it on held-out copies. Also run a zero-injection
control and an independent manufactured non-template leakage case. Measuring
`alpha -> alpha + a` in a fit to the same injected template is only an algebra
test; it is not the sensitivity gate. Compare complex recovered increments,
not differences between absolute magnitudes that can cancel coherently.

The report must give one of these bounded outcomes:

- `supported_on_spw4_development`: consistent paired cross-hand improvement,
  no material copolar regression under predeclared limits, stable nuisance
  sensitivity, and adequate numerical/injection checks.
- `disfavoured_on_tested_support`: the experiment detects the unit prediction
  in controls, but the real-data likelihood disagrees with it with stated
  uncertainty. Mere absence of significant improvement is not enough.
- `inconclusive_sensitivity`: the experiment cannot distinguish the unit
  prediction from the diagonal at the required sensitivity.
- `blocked_implementation_or_contract`: a numerical or provenance failure
  prevents scientific interpretation.

Separate statuses are required by frequency and spatial support. No outcome
automatically accepts full Jones for production or across C band. An upper
limit, if reported, needs its template, nuisance assumptions, coverage method,
and support domain; do not recycle the old generic-model upper limit.

## Phase 5 — updated C147-* comparison

Re-predict the eight offset fields with EVLA-C diagonal and full Jones at the
same nine SPW-4 frequencies. Use actual source-to-pointing geometry, per-row
source model, and the required geometric fringe. Do not apply the HOLORASTER
negative-commanded-offset shortcut to these fields.

Keep the original field partitions and channel exclusion rules for comparable
development diagnostics. Describe previously opened partitions as historical
holdouts, not newly sealed independent evidence. Fit no new beam correction.
Update F17 and related claims; do not leave the generic-model ring result as
current EVLA-C evidence. Any old all-channel result stays explicitly historical
unless it is actually recomputed at all those frequencies.

## Phase 6 — regenerate the complete publication

Create one compact, checksummed bundle from the new science exports. Retain
enough aggregates, representative samples, and complex map/cut data to rerun
every figure and reported statistic without CASA, an MS, or Bacchus. Compute
statistics from full exported populations, not the scatter-plot subsample.

The existing figure set must be audited using this replacement matrix:

| Figures | Required treatment |
| --- | --- |
| F01-F04: observation, occupancy, antennas, calibration oracle | Reuse only after provenance checks; explicitly identify what these validate. |
| F05: on-axis amplitude | Recompute with the updated prediction and matched source scale. |
| F08, F10, F12, F15: scatter, maps, radius, strata | Replace with EVLA-C/source predictions. Keep raw points and cell means distinguishable. Use actual scan/pass metadata where available. |
| F13: squint | Recompute measured/model series on named matched support. Show independent-mask estimate and common-mask sensitivity; separate voltage-grid and visibility-domain estimands. |
| F16: frequency | Replace all nine channel entries; include diagonal and cross-hand diagnostics and per-channel counts/support. |
| F17: offset ring | Replace with the updated experiment from Phase 5. |
| F18: cross-hands | Replace generic non-detection with EVLA-C diagonal/full comparison, uncertainty, and sensitivity result. |
| F19: SPW 5 | Keep explicitly sealed/not run. |
| F20-F25: dB, signed cuts, phase, radial and antenna coherence, outer examples | Recompute predicted complex arrays and statistics, not just captions. Use measured occupied coordinates and signal masks. |
| F26-F29: coordinate/feed comparisons | Retain clearly labelled old-model controls and matched-row paired comparisons. Do not mix full-raster and development denominators. |

Add sensitivity and direct full-minus-diagonal panels if existing figures do
not cover them. The original proposal's unimplemented figures need an explicit
coverage list; this task must replace every existing affected figure, but need
not build an unrelated new recovered-voltage archive to fill old placeholders.

Distinguish measured calibrated `V/I_model` from recovered antenna voltage E,
voltage magnitude from power, and complex phase from magnitude-only plots.
Do not relabel old commanded-coordinate maps as corrected maps without
transforming their coordinates and checking their calibration/gauge.

Each sidecar must name model, coordinate convention, calibration/source state,
frequency, split, region/support mask, estimator, data counts, and source
artifact hashes. Validate cross-file compatibility in the bundle builder.

Regenerate:

- `notebooks/vla_c_band_beam_validation.ipynb` with clean run-all outputs;
- `docs/vla_c_band_beam_validation.md` and all figure sidecars/assets;
- the machine-readable claim registry and figure manifest;
- the stale current-status section in
  `docs/vla-c-band-beam-validation-notebook-proposal.md`;
- a short supersession ledger explaining which old claims remain valid,
  which are historical comparisons, and which were invalidated by bugs.

The opening summary must state the actual tested frequency range, diagonal
support by region, full-Jones outcome, known errors/limitations, and the next
bounded scientific step. It must not describe the whole C band as validated.
Historical generic results belong in an appendix, not the current evidence
table. Historical measured quantities may remain if their derivation is still
valid; they must not inherit new model labels.

## Phase 7 — verification and handoff

Run focused coordinate, EVLA-C generation/loading, RIME, calibration, scoring,
injection, bundle, claims, and notebook tests. Add regressions for every defect
found. Run the broader relevant suite and report exact failures/skips. Run
lint and `git diff --check` on changed code.

Check specifically that:

- A wrong coordinate sign, feed artifact, Jones packing, or baseline reversal
  fails an independent test.
- Cross-hands survive extraction and are not silently removed by RR/LL masks.
- Zero signal, near-null predictions, sparse weights, fully flagged channels,
  missing model planes, and zero support produce explicit valid statuses.
- Score denominators, bootstrap multiplicities, partition isolation, and
  complex coefficient intervals are correct on manufactured examples.
- Publication metrics agree with the science reports on matched populations.
- Old-model artifacts cannot enter a current EVLA-C claim by filename alone.
- Checksums are verified before bundle contents are used.
- The notebook executes from the compact bundle without remote or MS access.
- All existing affected figures are replaced or visibly marked blocked;
  no figure is silently omitted while the refresh is called complete.
- Rendered figures are visually inspected for axes, units, clipping, labels,
  phase wrapping, null masks, and misleading historical/current overlays.

## Execution, failure handling, and completion criteria

Use a resumable driver with checkpoints per channel, model, and field group.
Write `status.json` and partial reports atomically even on failure. Cache keys
must include code/input hashes, frequency, coordinate convention, model,
normalization, and calibration state. Never resume solely because a filename
exists. Measure one batch before scheduling the whole run; report elapsed time,
memory, throughput, and estimated remaining work. Do not promise an overnight
completion before this measurement.

Numerical/contract failures stop the affected scientific scoring. Scientific
warnings or non-detections do not prevent plotting valid results. Preserve
completed results if another channel fails. If something cannot be completed,
publish an explicit partial-status account rather than substitute an old plot.

The final handoff must include:

1. A concise status table: calibration application, coordinate/Jones controls,
   numerical beam convergence, diagonal fit, cross-hand sensitivity/evidence,
   frequency coverage, offset ring, publication, and production acceptance.
2. Exact artifact paths, source snapshot hashes, bundle checksum, test results,
   and reproducible commands for prediction, rescoring, and publication.
3. A defect/supersession list with regression tests and affected outputs.
4. Links to the executed notebook and regenerated summary.
5. What the evidence supports, what it does not support, and one next step.

The experiment is complete only when all nine channel tasks, the updated ring,
sensitivity checks, and publication coverage have terminal, explained states.
The publication refresh is complete only when every existing affected panel
and claim is current or explicitly blocked. A scientifically usable full-Jones
beam is a separate decision and is not required to finish this handoff.
