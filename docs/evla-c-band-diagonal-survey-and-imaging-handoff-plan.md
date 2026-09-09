# EVLA-C band-wide diagonal survey and imaging handoff

## Objective and completion target

Extend the EVLA-C beam comparison across the C-band frequencies actually
covered by the available Perley holography observations. Deliver a versioned
diagonal voltage-beam catalog, measured-versus-predicted plots, and an explicit
opt-in adapter for the 3C391 seven-pointing Stokes-I mosaic.

The purpose is to return to imaging with a known beam specification and known
limitations. This is not a project to obtain a perfect beam, fit every
discrepancy, or detect full-Jones leakage before imaging can proceed.

The deliverables are:

1. Identified lower/upper-C datasets and a frequency/calibration inventory.
2. Independent implementation tests and documented numerical convergence.
3. Direct diagonal visibility comparisons across all usable observed C-band
   SPWs, with explicit unavailable or unsupported cases.
4. Magnitude, complex-response, null-alignment, and sidelobe comparisons.
5. A new catalog and a tested imaging adapter with explicit support policy.
6. An executed notebook, rendered summary, checksummed compact bundle, and
   machine-readable claim/status records.
7. A small seven-pointing imaging-path smoke test and a runnable full-mosaic
   command. A long optimized mosaic reconstruction is a subsequent task.

Full Jones remains diagnostic. Its scientific acceptance is not a gate for
any diagonal deliverable.

## Scope decisions and precedence

This document records the user's decisions following the SPW-4 EVLA-C refresh.
It supersedes earlier blanket instructions to keep SPW 5 closed **for this
declared diagonal transfer experiment**, after its method is recorded.
It does not authorize changing the production full-Jones factory.

- A second observation archive is on Bacchus but is not extracted. Identify
  it in place. If it is upper-C, extract and process it. Do not stop merely
  because it is not yet ingested.
- Complete lower-C even if the second archive is not upper-C or extraction
  encounters a genuine blocker. Record the actual missing coverage.
- First pass: one supported centre channel per SPW. Second pass: a small
  fixed within-SPW frequency set. Do not begin with every native channel.
- SPW 5 is a declared diagonal frequency-transfer test, not a full-Jones
  reopen. Keep its initial results separately from any later development.
- Keep the unit EVLA-C physical model, frequency-dependent taper, and audited
  coordinate convention. Do not inherit width 1.04, squint corrections, or a
  searched convention from prior development.
- Preserve existing comparison definitions for historical continuity. Add
  common-support and source-normalized diagnostics where needed for honest
  frequency comparisons.
- Deliver a new versioned catalog and publication bundle. Preserve
  `vla_c_band_beam_validation_v2` and its source products.
- First imaging target: one shared Stokes-I sky for the seven 3C391 pointings,
  with the existing central field, outer guard, and explicit bright
  out-of-field source catalog. Do not quietly change the FOV or sky model.

Ordinary read-only inventory, safe extraction into a new directory, and
calibration on new work copies are within this workflow. Do not overwrite
original archives/MSs, stop unrelated jobs, delete products, or commit/push
without separate authority. Preserve unrelated uncommitted changes.

## What is known, and what is not

The identified lower-C execution is THOL0001 / sb31628704 / eb31629959,
observed on 14 January 2016. The plan records nominal coverage of about
3.99–6.01 GHz. The current scientific commissioning MS contains only SPWs 4
and 5; it is not the source for all lower-C channels.

The earlier plan describes upper-C on 18 January 2016 at about 5.99–8.01 GHz.
Treat those as expected metadata, not proof of the second archive's identity.
In particular, do not mistake a THOL0001 X-band execution for upper-C.

The latest SPW-4 publication is a useful preserved checkpoint. Its residual
Jones is a native channel-32 solution reused across its frequency series.
That is not a band-ready residual calibration. Its injection-based full-Jones
sensitivity interpretation and p32/p64 convergence tails need qualification;
do not inherit those as cleared gates in this survey.

Large channel-24 LL and channel-40 RR excursions strongly influence the
published frequency curves. Diagnose them in a parallel workstream. They
must not be absorbed into a beam fit. Their complete explanation is not a
prerequisite for inventory, model generation, or other supported SPWs.

## Code and data starting points

Inspect current versions before use. Historical defaults and hard-coded SPW,
field, antenna, frequency, and output-path choices need particular attention.

- `docs/vla-holography-development-plan.md`
- `docs/thol0001_holoraster_coordinate_and_feed_audit.md`
- `docs/evla-c-full-jones-validation-refresh-plan.md`
- `docs/evla_c_full_jones_validation_refresh/HANDOFF.md`
- `src/sl1mjax/holography_ms.py`, `holography.py`,
  `holography_holoraster_coordinates.py`, and `holography_full_jones.py`
- `src/sl1mjax/cassbeam_evla_c.py`, `cassbeam_highres.py`,
  `holography_highres_cassbeam.py`, and `holography_beam_prior.py`
- `scripts/generate_cassbeam_highres_model.py`
- `scripts/run_thol0001_evla_c_validation_refresh.py` and
  `src/sl1mjax/evla_c_*` modules
- `src/sl1mjax/cassbeam_beam.py`, `voltage_operator_jax.py`,
  `voltage_reconstruction.py`, and `scripts/run_3c391_phase6_bacchus.py`
- Existing `beam_validation_*` modules and notebook/bundle/render scripts.

Known Bacchus paths to verify:

- Archives: `/media/stephen/astro/vla/`.
- Original lower-C MS:
  `/media/stephen/astro/vla/extracted/THOL0001.sb31628704.eb31629959.57401.169024456016.ms`.
- Scientific commissioning products:
  `/media/stephen/astro/vla/extracted/commissioning/`.
- Beam artifacts: `/media/stephen/astro/vla/beam_models/`.

The existing `diagonal_copolar` factory loads the older packaged CASSBEAM
artifact. Do not assume that selecting this mode uses high-resolution EVLA-C.

## Phase 0 — lock the protocol and create a durable status ledger

Before opening new science samples, write a versioned `experiment_manifest.json`
containing the method, sampling, partitions, metrics, numerical tolerances,
scientific classifications, calibration policy, seeds, and read/write paths.
Record the complete staged source snapshot and dependencies, not only git HEAD.

Predeclare these frequency choices for 64-channel SPWs:

- Pass A: native channel 32.
- If 32 is unsupported, use the nearest supported channel to the centre,
  choosing the lower index on a tie. Decide support from flags/calibration
  metadata before inspecting model residuals. Record the substitution.
- Pass B: native channels 8, 24, 40, and 56, plus the Pass-A channel.
  Retain unavailable slots explicitly; do not replace them with better scores.
- For another channel count, use the corresponding fixed channel-index
  fractions, with a documented rounding rule chosen before scoring.

Keep native frequencies and visibilities separate. Read frequencies from each
MS's SPECTRAL_WINDOW table; SPW numbers do not identify frequencies globally.
This sampling does not validate every channel or smooth interpolation between
sampled frequencies. Any later denser diagnostic selection must be labelled
development and recorded as an extension.

Partition by physical groups, not DDID-dependent row order. Preserve the
existing SPW-4 masks for continuity. For new observations establish connected
calibrator holdouts, whole spatial-cell and mover partitions, and separate
reference/visit diagnostics. Do not union holdout axes until recovery becomes
unsupported. Do not claim exact off-axis repeatability when lattices differ.

Preserve whole new frequency windows or the upper-C execution as transfer
tests until the fixed lower-C method is ready. On first opening, record the
result without tuning. Once used to change the method, mark them development.

Gate: a reviewer can determine the exact intended run without guessing.

## Phase 1 — archive identity, extraction, and full-band inventory

List archive members and metadata without loading all visibilities. Identify
project, scheduling/execution blocks, observing date, band, and source. Validate
archive integrity where possible. Check free space for extraction, work copies,
calibration tables, model planes, and exports before launching writes.

Extract into a new explicit directory. Refuse absolute member paths, `..`
traversal, or links escaping the extraction root. Refuse overwriting existing
files. Keep the original archive unchanged and record its checksum.

Inventory each MS independently:

- DDID → SPW → POLARIZATION mapping; channel centres, widths, ordering, and
  correlations. Verify RR/RL/LR/LL packing from metadata.
- FIELD/SOURCE identities and directions, calibration visits, scan intents,
  antenna names/positions, reference/mover roles, and parallactic coverage.
- POINTING frames, units, interval joins, target/direction/offset relations,
  overlapping intervals, missing joins, and transition handling.
- Actual raster cells, pass-specific occupancy, extent and spacing. Never
  snap measured coordinates to the nominal Memo lattice.
- DATA, CORRECTED_DATA, MODEL_DATA, FLAG, FLAG_ROW, WEIGHT, and sparse or empty
  WEIGHT_SPECTRUM. Column presence is not proof of valid contents.

Required tests: shuffled DDIDs, nonmonotonic global MAIN times, per-antenna
pointing intervals, exact interval boundaries, ambiguous joins, reordered
antenna IDs, nonzero flags, and both baseline orientations. Reconstruct roles
and coordinates on a compact real metadata slice from each execution.

Gate: a table of actual frequency coverage and supported observations. A
missing upper-C product blocks upper-C only, not the entire survey.

## Phase 2 — calibrate each frequency range on new work copies

Use the validated CASA/JAX application contract and suitable calibration
solutions for each observation/SPW. Re-solving necessary calibrator terms on
new work copies is authorized. Reusing incompatible tables is not.

Assign the same per-frequency source flux/model gauge to all on-axis visits
of a calibrator. Keep its image-model visibilities distinct from the integrated
flux polynomial. Do not replace one with the other because the source name
matches. Verify source structure on appropriate baselines after calibration.

Do not solve gains or residual calibration on moving HOLORASTER baselines.
On-axis scans and properly selected reference–reference samples may supply
constraints under the recorded protocol. Residual-Jones frequency freedom
must earn support on calibrator holdouts, not beam residuals.

For every used term, record antenna/SPW/time/channel support, gainfield and
interpolation policy, reference frequency, flags, calibration frame, and gauge.
Missing solutions do not become identity. A justified global X term must be
explicit. Missing polarimetric support blocks cross-hand claims; retain
copolar diagnostics only where their validity is justified.

Required tests and evidence:

1. Flux-gauge consistency across fields and time. Reproduce a manufactured
   mixed 1 Jy/8 Jy model failure and require the gate to catch it.
2. Native-channel apply before any averaging; refusal of double application.
3. CASA/JAX injected-basis and real-visibility checks on representative new
   SPWs, edge/central channels, both baseline orientations, and time brackets.
4. Delay reference from the calibration table; CASA-faithful interpolation,
   including flagged brackets and float32 time handling; apparent-coordinate
   parallactic and antenna-position phase conventions.
5. Independent R/L masks, FLAG_ROW handling, and row-level weight fallback for
   absent/empty/sparse WEIGHT_SPECTRUM. No fabricated precision from nominal
   WEIGHT values and no reduced-chi-square claims without a noise model.
6. Held-out on-axis amplitudes, phases, R/L ratios, and channel dependence.
   Separate interpolation checks from end-to-end holdouts when a scan entered B.
7. A connected graph in calibration holdouts. Predicting an unseen antenna as
   identity tests a population assumption, not interpolation of its solution.

A channel-32 residual Jones must not be applied across the new survey. If no
validated residual-frequency model exists, use the supported base calibration
and explicitly report its residuals/limitations. Do not claim frequency-varying
beam error can be separated from uncharacterized calibration error.

### Excursion diagnosis, in parallel

Trace the saved channel-24 LL and channel-40 RR outlier row identities to
MS time/scan/baseline. Compare DATA, each cumulative calibration stage, and
CORRECTED_DATA across neighboring channels. Inspect all connected baselines,
not only the largest residuals. Determine where the excursions first appear.

The worst-1% trim is an influence diagnostic only. Do not adopt residual-based
flagging after seeing holdout scores. If a physical/metadata flagging rule is
justified, derive it on development data, save separate reason masks, and show
original and newly masked scores. Treat its subsequent tests as development.

Gate: supported calibration for each scored frequency, or an explicit local
block/qualification. No unsupported extrapolation hidden behind a warning.

## Phase 3 — EVLA-C generation, conventions, and numerical convergence

Generate only missing full-complex EVLA-C reference planes at the declared
frequencies. The delivered imaging catalog exposes diagonal R/L responses;
retaining raw full-Jones output does not accept off-diagonals.

Pin physical parameters to audited source definitions. Verify that the
frequency-dependent parameterization applies throughout the requested band,
including near receiver/model boundaries. Do not extrapolate the taper formula
or one feed setting beyond its documented range by accident.

Record exact native frequency, executable/source/geometry hashes, feed-x
conversion, taper, aperture grid, angular sampling, origin/index conventions,
Jones packing, units, normalization matrix, and extent. Never normalize to the
co-polar peak instead of boresight. Peaks above one are allowed.

Maintain numerical and scientific support separately. Test finer aperture
integration and finer angular sampling separately; p32/p64 agreement alone
does not test all numerical approximations. Use representative low/mid/high
frequencies, transitions, measured off-grid points, nulls, and outer lobes.
Compare complex E and predicted visibilities. Report regional percentiles,
maxima, and loss changes, not only the median near the centre.

Set error budgets before comparison: numerical errors must be materially
smaller than the regional accuracy claim or model difference being quoted.
Where that cannot be established, label the region numerically limited. Do
not relax a tolerance because a run failed or use a relative metric at a zero.

Independent tests must include:

- Hand-coded complex RIME assembly, separate from production helper calls.
- Asymmetric manufactured beam with distinguishable axes, hands, phases, and
  lobe signs; I, I+Q, I+U, and I+V inputs; several nonzero parallactic angles.
- HOLORASTER source-in-beam coordinates and normal offset-field geometry.
  Confirm whether native offsets are angles or direction cosines. Bound
  small-angle/projection error at the full measured extent.
- Baseline reversal, receive/transmit definition, column unpacking, and the
  noncommuting normalization/calibration multiplication order.
- Deliberately omitted conjugation, coordinate sign, spatial/basis rotation,
  frequency selection, and geometric fringe must fail discriminating tests.
- Grid-node lookup, off-grid interpolation, singleton dimensions, descending
  frequency axes, unsupported frequency/angle, and boresight identity.

Gate: per-frequency numerical artifact qualification. Hash equality or
forward/inverse self-consistency alone is not a convention oracle.

## Phase 4 — staged measured-versus-predicted survey

Run one SPW-centre comparison end to end before processing all centres. Measure
runtime, memory, and output size. Then process Pass A across the usable band,
followed by Pass B. Write checkpoints and partial reports after each channel.

Use the fixed unit diagonal model. Compare direct calibrated complex
visibilities; do not require empirical beam recovery. Keep source coherency
and residual calibration identical between any model arms. Generate optional
full-Jones overlays only with valid calibration/numerical support. They cannot
block the diagonal survey or acquire a detection label from a plot.

Store row identities, source model, all available measured hands, predictions,
weights/flags, both named coordinates, time/scan/pass, antenna IDs, frequency,
support, and partitions. Preserve quantities needed to rescore without reading
the MS or regenerating beam predictions. Stream bounded batches; no monolithic
row × direction × channel × antenna allocation.

For each hand report the exact estimands:

- Residual power: `sum(w * |Vobs - Vpred|^2) / sum(w * |Vobs|^2)`.
- Its square root, labelled relative RMS against observed visibility power.
- Absolute Jy residuals and source-normalized `|delta V| / I_model`.
- Complex slope/correlation, coherent residuals, supported phase statistics,
  sample counts, weight support, and noise-floor diagnostics.
- R-minus-L contrast on common hand support.

Keep legacy measured-response main/mid/outer bins for continuity, but disclose
their data dependence. Add a fixed geometry/reference-defined mask shared by
hands/model arms, and a common-row frequency comparison within each execution.
Do not let contaminated amplitude move a sample into the main lobe without
showing that selection effect. Regions at different frequencies should be
shown in both angular and frequency-scaled coordinates; they are different
questions and must have different labels.

Use cell/mover/scan/reference-aware uncertainties as appropriate. Do not count
adjacent channels or repeated baselines as independent samples. Do not silently
pool antenna and spatial holdout axes. An opened transfer dataset must remain
labelled as such after any method changes.

Retain existing numerical support classes in the source classifier, including
the 1% main-lobe residual-power cut for both hands and the existing mid-beam
rule. Serialize their exact definitions in the manifest. Outer remains
diagnostic unless new evidence supports a stronger statement. A threshold
failure is a measured limitation, not a process failure or automatic flag.

Gate: every planned channel has an explained status: scored, no usable data,
calibration unsupported, numerical limit, or execution failure. Missing data
are never scored as zero loss. Rendering cannot promote a scientific status.

## Phase 5 — nulls, sidelobes, and band-wide plots

Produce the plots the imaging question requires:

1. Per-frequency measured/predicted R/L response and complex residual maps on
   identical coordinates, with linear and dB displays.
2. Magnitude and signed Re/Im cuts through the centre along both axes and
   diagonals; additional azimuthal sectors for first-null/lobe comparison.
3. Null/minimum positions versus frequency and azimuth, with uncertainty and
   measured sampling resolution. Report an interval or unresolved minimum
   where sampling/noise prevents a precise zero.
4. Main-beam width and R-minus-L squint vectors. Show 20%-of-peak masks,
   independent/common-mask sensitivity, per-mover spread, and reference spread.
5. Radial magnitude, complex coherence, and phase residuals by frequency and
   mover. Mask unsupported phase using a predeclared signal rule.
6. A frequency × radius support/accuracy map, including calibration and
   numerical limitations, not just model residual thresholds.
7. Lower/upper execution overlap near 6 GHz, if supported. Distinguish actual
   matched frequencies/cells from interpolation and compare them separately.
8. Bright-source consequences: sampled offsets relevant to the 3C391 catalog,
   with predicted response and uncertainty/unsupported regions identified.

Label calibrated `V/I_model` as such. It is not automatically a recovered
antenna voltage. Do not mix per-antenna phases without a documented gauge, or
compare a power beam directly with a voltage beam. A common scalar lobe sign
cancels between identical antennas; antenna-dependent phases do not. Null
location and magnitude errors still matter for a bright source even when its
flux is adjustable. Low-SNR magnitude ratios are not multiplicative fixes.

No spatial interpolator should manufacture a precisely located zero between
widely spaced raster samples. Check interpolation with manufactured nulls at
the actual sampling and noise level. Show raw occupied cells alongside any
smoothed display. Phase uncertainty near a null must not be suppressed.

Gate: a reader can identify where the model matches, differs, or is untested,
without inferring precision from plot smoothness.

## Phase 6 — versioned catalog and opt-in imaging adapter

Deliver a new named catalog, for example `evla_c_diagonal_survey_v1`. Distinguish:

- Raw generated planes and numerical support.
- Empirically compared frequency/angle support and accuracy classes.
- Frequency/spatial interpolation policy and unvalidated model-only regions.
- Calibration/boresight gauge and array-average assumptions.

Model availability is not scientific validation. Test interpolation against
independently generated intermediate frequencies before enabling it. Where
interpolation is not demonstrated, generate the required imaging nodes or
refuse the query. Do not use a nearest plane without a recorded error bound.

Add an explicit opt-in beam selector that names the catalog and digest. Keep
the old selector reproducible. The adapter must not silently load the older
generic packaged tables or promote full Jones. Cache keys must include model,
normalization, frame, frequency, pointing, antenna assumptions, and resolution.

Exercise the adapter through the actual JAX prediction, explicit adjoint,
integration planner, reconstruction driver, checkpoints, and product writer.
A Python beam evaluator without support in those paths is not a completed
imaging adapter.

Required imaging tests:

- Independent direct DFT versus production prediction on asymmetric skies,
  two pointings, R/L squint, varied times/frequencies, and reversed baselines.
- JAX versus NumPy, explicit adjoint versus autodiff, inner-product identity,
  finite differences, and invariance to time/row/direction tile sizes.
- Extended square pixels integrated with the beam inside the integral;
  manufactured pixels crossing steep gradients/nulls; mixed integration depth,
  arbitrary parent order, padding, and flux-conserving refinement.
- One common source seen in multiple pointings with one shared flux. Verify
  that FOV/source coordinate transforms and pointing selection are correct.
- Frequency mismatch, off-grid unsupported samples, and one unsupported sky
  component. Unsupported support must not erase unrelated valid predictions.
- Memory preflight, chunked execution, restart equivalence, and cache
  invalidation after changing the artifact or coordinates.

Do not silently fall back to Airy beyond the catalog. For the imaging trial,
numerically supported but empirically uncertain outer response may be used
with explicit warnings. Truly unsupported important directions require an
extended generated grid or a named, documented fallback policy. Never pretend
they are known zeros.

The 3C391 smoke test must span all seven pointings using a small deterministic
training-only subset, not seven independent images. Keep its existing FOV,
guard, and bright-source components. Fit a few flux steps, save/reload a
checkpoint, predict again, and render products. Check finite losses/gradients,
consistency, memory, and timing. Do not demand KKT 1e-5 for this smoke test.

Gate: a reproducible command runs the actual opt-in seven-pointing path. A
long scientific mosaic fit is handed off with its known support and warnings.

## Phase 7 — reproducible publication and handoff

Create a new compact checksummed bundle. Preserve v2 unchanged. Build the
notebook and summary from the new bundle; preserve prior notebook/report
snapshots before replacing the current entry points. Keep scientific scores
separate from plotting subsamples and retain sufficient reduced statistics
and representative complex arrays to rerun every published number and figure.

Required publication products:

- Updated `notebooks/vla_c_band_beam_validation.ipynb` with clean run-all outputs.
- Updated `docs/vla_c_band_beam_validation.md` and complete figure sidecars.
- Band inventory, source/calibration provenance, null/sidelobe panels, and
  the frequency-by-radius suitability table.
- A supersession ledger: old generic results, current SPW-4 results, newly
  opened transfer results, and any corrected diagnostic claims.
- Exact catalog identifier and opt-in 3C391 command, plus clear limitations.

Each claim/figure must name its execution, frequency, model hash, coordinate
frame, source/calibration state, population, mask, estimator, and support.
No stale generic-plane or channel-32-only result may enter a band-wide claim.
Unsupported areas remain visible. Full Jones is diagnostic/not accepted.

Run from the compact bundle without CASA, the MS, network, or Bacchus paths.
Inspect every rendered figure for units, legends, hand labels, clipped axes,
phase wrapping, misleading smoothing, and historical/current ambiguity.

## Mandatory regression and failure-injection checklist

Tests must demonstrate that known wrong alternatives fail, not merely that
the current implementation agrees with itself. Add each discovered defect as
a failing regression before its fix. Review production-path coverage, not
only small utilities. At minimum cover:

| Prior failure mode | Required protection |
| --- | --- |
| Wrong commanded/source sign | Metadata-derived and asymmetric synthetic coordinate oracles. |
| Generic VLA feed labelled EVLA-C | Input/model hashes and known physical parameter assertions. |
| Raster origin or packing error | Asymmetric raw table fixture with exact origin and each complex hand. |
| Calibration gauge drift | Per-field/time/model apply-back test with deliberately inconsistent gauges. |
| CASA interpolation/reference-frequency differences | Table-level synthetic cases plus compiled-CASA fixtures where available. |
| Residual channel-32 extrapolation | Frequency-support validation and an explicitly failing unsupported query. |
| Missing/empty weights or hand flags | Sparse per-row fixtures, all-flagged hands, and unequal R/L support. |
| Wrong normalization near beam zeros | Exact metric tests in Jy, source-normalized units, and observed-power units. |
| Data-dependent main-lobe membership | Fixed-support comparison and outlier-contamination fixture. |
| Array-coordinate or time association mistakes | Shuffled rows, DDIDs, antennas, and exact match/rejoin tests. |
| Starved/overlapping holdouts | Partition accounting, disjointness, graph and spatial-support checks. |
| False convergence from a small median | Tail/regional visibility-error tests and null-crossing examples. |
| Stale cache or artifact | Hash-changing inputs must invalidate resume/cache and claims. |
| Report lost after another stage fails | Forced interruption and exception tests preserving finished results. |
| Statistical label unsupported by experiment | Known synthetic null, mismatch, and low-sensitivity cases with uncertainty. |
| Published sample count wrong | Per-hand/per-region denominators and counts match exported rows. |

If optional cross-hand sensitivity work is retained, do not reuse the flawed
test that adds a template to real data and calls it recovered only when fixed
full Jones beats fixed diagonal. Measure the complex injected increment with
separate train/score samples and clustered controls. Do not generalize a
channel-32 injection result to every frequency. This is not a requirement to
complete the diagonal survey.

## Operational safety, milestones, and final acceptance

Use a new staging tree with recorded source hashes. Check active Bacchus jobs
before scheduling. Use memory preflight and measured one-channel timing before
estimating the whole survey. No monolithic overnight `all` job without tested
checkpoint/resume behavior and partial-status output.

Maintain machine-readable states per execution/SPW/channel/region. Suggested
states: planned, running, complete, unsupported, scientifically-qualified,
numerically-limited, and blocked. Keep computational completion separate from
scientific support and production readiness.

Software/contract failures stop the affected calculation. Scientific threshold
failures produce warnings and completed diagnostics. Continue independent
supported work. Do not loosen thresholds or drop difficult samples to turn a
red status green. Never stop the entire survey because one channel or the
full-Jones overlay is inconclusive.

Milestones to report:

1. Archive identity and full frequency inventory, including upper-C yes/no.
2. First new SPW calibrated, numerical controls passed, and plots produced.
3. Pass-A centre-channel survey across every available SPW.
4. Pass-B within-SPW coverage and null/sidelobe comparisons.
5. Catalog and seven-pointing imaging smoke test.
6. Executed publication and final handoff.

Run focused suites after each stage, then the broader relevant CASA, RIME,
holography, beam, JAX operator, reconstruction, and publication suites before
handoff. Record exact commands, pass/fail/skip counts, and reasons for skipped
external gates. Run lint/type checks on changed modules and `git diff --check`.
Do not claim the entire test suite passed if only selected tests were run.

Completion means all declared frequencies have an explained terminal status,
the new catalog and adapter work on the supported imaging range, and the
publication faithfully reports the evidence. Missing upper-C or a scientifically
poor region narrows the delivered support; it must not be hidden or block a
useful lower-C handoff. State partial scope explicitly.

The final summary must answer: what can be used for the 3C391 mosaic now,
which frequencies/angles remain uncertain, what changed since SPW 4, whether
the large excursions were explained, and the exact next imaging command.
Full-Jones validation and a fitted beam correction remain separate projects.
