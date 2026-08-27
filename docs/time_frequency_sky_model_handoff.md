# Time/frequency sky-model splitting: status and handoff

## Purpose of this document

This document is a handoff for the next development session. It records why
time- and frequency-dependent sky components were introduced, what has been
implemented, what the 3C391 experiments established, and what should happen
next.

The short conclusion is:

- The native-resolution split machinery works and passes synthetic recovery
  tests.
- A spatially blind real-data search found an apparent one-minute sky
  decrement.
- A complete-scan model comparison showed that this signal is mainly an
  antenna-based calibration residual.
- Time-dependent sky splitting is therefore not yet scientifically reliable.
- Frequency is the better first axis for the next sky-model extension, after a
  small time-local calibration gate.
- The first frequency model should be a validated continuum slope model, not a
  free map per channel and not a Gaussian process.

## Scientific objective

The long-term objective is a sparse, physics-informed sky model that can adapt
in space, frequency, and time. The visibility likelihood remains the source of
evidence for every additional degree of freedom. Validation must decide whether
the data support a spatial refinement, a spectral term, a temporal term, or a
calibration term.

The present spatial model is a fixed union quadtree. Each leaf carries a
non-negative integrated flux. The forward model evaluates the exact leaf width,
wide-field phase, frequency-scaled UV coordinates, and frequency-dependent VLA
primary beam.

The present sky flux is intrinsically achromatic. A leaf or point component has
one flux value shared by every channel. Frequency affects the predicted
visibility through UV scaling and the primary beam, but not through an intrinsic
source spectrum. This is now the clearest missing physical term.

“Splitting in time or frequency” currently means adding a signed coefficient
with restricted time or frequency support to a fixed spatial atom. It does not
mean fitting an independent image in every time/channel cell. That distinction
is important because unrestricted independent images would be poorly
identified and difficult to regularise.

## Data and frozen model used so far

The native C1 fixture comes from the 3C391 VLA mosaic Measurement Set. It has:

- native 10-second integrations;
- 64 channels at 2 MHz spacing;
- frequencies from 4.536 to 4.662 GHz;
- RR and LL parallel hands;
- exact UVW coordinates in metres, converted per channel in the forward model;
- the extended Airy primary beam evaluated separately at every channel;
- the selected fourteen-epoch external-calibrator gain solution with native
  linear time interpolation;
- a frozen 11,536-leaf mosaic topology and frozen composite sky prediction.

The band spans 126 MHz, or 2.74% of the geometric-centre frequency of
4.5986 GHz. A typical spectral index of -0.7 changes intrinsic flux by about
1.92% from one band edge to the other. That is small per faint leaf, but it is
not small for the approximately 10 Jy integrated remnant.

The full Measurement Set is already available on Bacchus at:

`/home/stephen/checkouts/SL1MJax-frozen-20260824/data/3c391_ctm_mosaic_10s_spw0.ms`

## Development path and evidence

### 1. Native resolution was retained for model selection

The original imaging fixture averaged the target to 60 seconds and four
32 MHz channels. The native ablation compared this with 10-second, 2 MHz data.

At 60 seconds and 32 MHz only 1.15% of the native time-frequency samples remain.
The centroid approximation to the averaged forward model changes aggregate MSE
by only 0.014%. The main cost of averaging is therefore lost evidence, not a
large coordinate-smearing error in this dataset.

The mean retained-sample value of `w |V - V_model|^2` is 28.8 times higher at
60 seconds/32 MHz than at 10 seconds/2 MHz. Residuals are coherent inside the
large cells, while transient and line support is hidden by averaging.

The adopted workflow is now:

1. use coarse data for cheap spatial topology discovery and initial fitting;
2. use streamed native data for time, frequency, calibration, and flag model
   selection;
3. keep native predictions at their exact UVW, frequency, and beam response.

Implementation and results:

- `scripts/ablate_3c391_native_averaging.py`
- `outputs/3c391_native_averaging_ablation/`

### 2. Known-support time and frequency injections were recovered

The first recovery experiment injected a point-source response at the C1 phase
centre into the real calibrated residual. Candidate coefficients were fitted on
complete discovery baselines and evaluated on different baselines. Five
deterministic baseline partitions were used. Paired zero-injection controls
removed the coherent response already present in the real residual.

The first repeatable known-support selections were:

| support | first repeatable injected flux |
|---|---:|
| 10-second transient | 8.11 mJy |
| 30-second transient | 9.37 mJy |
| 60-second transient | 13.25 mJy |
| 1 channel / 2 MHz | 2.27 mJy |
| 2 channels / 4 MHz | 1.61 mJy |
| 4 channels / 8 MHz | 1.14 mJy |
| 8 channels / 16 MHz | 1.61 mJy |

Known spectral support is therefore easier to recover than known temporal
support in this fixture. It uses all integrations and pays no search trials
penalty. This result demonstrates sensitivity, not a complete discovery
protocol.

Implementation and results:

- `src/sl1mjax/sky_recovery.py`
- `scripts/run_3c391_native_injection_recovery.py`
- `outputs/3c391_native_injection_recovery/`

### 3. Blind support search exposed the trials penalty

The blind fixed-position search tests 350 refinements at the C1 phase centre:

- 104 time intervals with widths of 1, 3, and 6 integrations;
- 245 frequency intervals with widths of 1, 2, 4, and 8 channels;
- one signed linear coefficient in log frequency.

Discovery baselines rank all candidates. Selection baselines choose from a
shortlist. The chosen candidate is refitted on discovery plus selection. Only
that candidate is scored on the stage-held-out evaluation baselines.

The first repeatable blind recoveries were:

| injected component | blind recovery threshold |
|---|---:|
| 10-second interval | 16.22 mJy |
| 30-second interval | 18.73 mJy |
| 60-second interval | 13.25 mJy |
| 1 channel / 2 MHz | 18.18 mJy |
| 2 channels / 4 MHz | 12.85 mJy |
| 4 channels / 8 MHz | 9.09 mJy |
| 8 channels / 16 MHz | 6.43 mJy |
| log-frequency slope | 10 mJy edge-to-edge |

The spectral interval search loses much of its known-support advantage because
245 overlapping boxes compete. The broad eight-channel feature is still easier
to discover than the tested temporal events. The smooth log-frequency slope is
also recovered at a modest 10 mJy edge-to-edge signal.

The real zero-injection residual is a useful null. Discovery and selection find
apparently useful candidates, but the selected refinement worsens evaluation
loss in every partition. The acceptance fraction is zero. This established the
need for discovery, selection, and evaluation cohorts after mining many
candidates.

The present interval bank is a useful line/transient detector, but it should not
be the first continuum spectrum model. A physical low-order continuum family
has far fewer trials and should be tested first.

Implementation and results:

- `scripts/search_3c391_native_variability.py`
- `outputs/3c391_native_variability_search/summary.json`

### 4. Spatial search found an apparent time event

The search was extended from one fixed point to 768 bright hierarchical leaves.
The prefilter used frozen apparent flux only, so the holdout residual did not
choose the initial spatial candidates. Exact leaf responses were streamed in
leaf and row tiles.

All five baseline partitions selected the same final six-integration interval.
They selected adjacent level-1 leaves about 1.94 arcmin from the C1 phase centre.
The fitted interval decrement was -20.9 to -26.7 mJy. It reduced stage-held-out
loss by 0.31% to 0.64%.

This was repeatable, but it was not automatically astrophysical. A coherent
antenna gain error can project into adjacent image atoms and look like a local
sky decrement.

Implementation and results:

- `scripts/search_3c391_native_spatial_variability.py`
- `outputs/3c391_native_spatial_variability_search/`
- commit `570dd30`

### 5. Complete scan 29 identified the calibration confound

The selected interval is the final minute of C1 scan 29. A new diagnostic read
the complete five-minute native scan and compared four explanations over the
same final-minute support:

1. a local sky-leaf change;
2. a common multiplicative amplitude change;
3. a common primary-beam pointing shift;
4. per-antenna linearised amplitude and phase changes.

Every candidate also received the same static whole-sky scale and static local
leaf correction. The models were selected on whole-baseline holdouts. The
per-antenna model used ridge regularisation selected by validation.

All five partitions selected the per-antenna complex-gain event. Its mean
selection gain beyond the static nuisance model was 3.75%, with a range of
2.62% to 4.47%. The alternatives were much weaker:

| event family | mean selection gain beyond static | range |
|---|---:|---:|
| local sky leaf | 0.066% | 0.052--0.086% |
| common amplitude | 0.098% | 0.052--0.148% |
| common pointing | 0.306% | -0.183--1.108% |
| per-antenna complex gains | 3.747% | 2.618--4.472% |

After refitting, the gain event improved held-out residual power by 0.82% to
4.60% beyond the static nuisance model. The same antennas carried stable phase
terms across partitions. Antennas 2, 7, 9, and 20 were especially consistent.

The current interpretation is that the apparent spatial/time event is mainly an
antenna-based calibration residual projected into the sky dictionary. It is not
reliable evidence for a time split.

The test is held out within this diagnostic stage, but it is not pristine
discovery evidence. The leaf and minute came from the earlier spatial search,
which used other baseline partitions of the same fixture. The result is a
strong diagnosis, not an unbiased significance estimate.

Implementation and results:

- `src/sl1mjax/residual_models.py`
- `scripts/diagnose_3c391_scan_residual.py`
- `outputs/3c391_scan29_residual_diagnostic/`
- commits `d789bf0` and `4185ef5`

## What was learned about calibration

The selected external calibration is not generally poor. Moving from six gain
epochs to all fourteen calibrator scans removed almost the complete CASA/JAX
gap. Native linear interpolation beat every tested smoothing spline and
circular-GP model. It scored only 0.79% worse than CASA on the sealed fold.

The calibration-term ablation identified time-dependent `G` as the important
term. Replacing JAX `G` with CASA `G` improved sealed residual power by 18.7%.
Replacing delay `K` improved it by 1.6%. Replacing bandpass `B` improved it by
only 0.17%.

This distinction matters for the axis choice:

- Time-dependent target structure competes directly with a demonstrated
  time-dependent antenna-gain residual.
- Frequency-dependent sky structure can compete with bandpass error, but the
  current bandpass comparison found no material JAX/CASA discrepancy.
- Beam chromaticity and per-channel UV scaling are already included in the
  forward model.

Frequency is not calibration-free. A later spectral search must still compare
sky spectra with per-antenna bandpass perturbations. The evidence says this is a
smaller immediate confound than target-time gain variation.

## What was learned about frequency models

### Demonstrated

- Native frequency coordinates and UV scaling are correct in the forward path.
- The Airy beam is evaluated at each native channel.
- Signed spectral intervals can be fitted and selected with no evaluation
  leakage.
- Known spectral support is recoverable around 1--2 mJy at the phase centre.
- Blind compact intervals are recoverable around 6--18 mJy, depending on
  width.
- A smooth log-frequency slope is recoverable at 10 mJy edge-to-edge in the
  fixed-position test.
- The four coarse residual bins have 10--13% more power in the two outer bins
  than in the inner bins. No single channel bin dominates.

### Not yet demonstrated

- No real spectral sky component has been accepted.
- The spatially blind real-data search was dominated by the time-calibration
  event before frequency candidates could be interpreted.
- There is no joint full-sky continuum spectral fit.
- There is no per-leaf power law or Taylor coefficient in the composite model.
- There is no spectral regulariser coupling a slope to its non-negative base
  flux.
- There is no explicit competition between a sky spectral term and an antenna
  bandpass perturbation.
- Compact frequency boxes have only been tested at one fixed position in the
  injection study. The spatial search can score them, but no accepted real line
  has been established.

## Should frequency precede time?

Yes, for the next sky-model extension.

Frequency has three practical advantages:

1. A static spectrum uses every integration, so it has more evidence than a
   short temporal event.
2. The dominant calibration discrepancy is in time-dependent gains, not the
   bandpass.
3. Smooth continuum spectra have a strong physical low-dimensional model. A
   flat spectrum can first gain a slope, then curvature only if validation
   supports it.

Time should not be abandoned. It should be paused as a sky degree of freedom
until a constrained time-local gain model is available as a competing
explanation. Otherwise the sky will absorb calibration changes and create
false transient pixels.

The immediate order should be:

1. turn the scan-29 gain diagnostic into a validated time-local calibration
   correction;
2. freeze that calibration and rerun the native null search;
3. implement and test continuum frequency models on the fixed spatial tree;
4. return to sky-time splitting only when gain and sky candidates compete in
   the same selection protocol.

The calibration step should be small and bounded. It is not a request to build
full target self-calibration before spectral work.

## Recommended first frequency model

### Use a Taylor/power-law hierarchy

For continuum emission, start with

\[
I_j(\nu) = I_{0,j} + I_{1,j}\,x,\qquad
x = \log(\nu/\nu_0).
\]

`I0` is the existing non-negative leaf flux. `I1` is a signed spectral
coefficient. For a narrow band, `I1 / I0` approximates the spectral index.
This form remains linear in the fitted coefficients. It is therefore compatible
with the current direct operator, FISTA/proximal machinery, and exact
sufficient-statistics screens.

Do not give every leaf an unconstrained slope immediately. Use a hierarchy:

1. achromatic frozen sky;
2. one global spectral scale;
3. one slope per existing component group, such as central tree, coarse outer
   tree, and catalogue atoms;
4. slopes for a validation-selected shortlist of bright/residual-supported
   leaves;
5. curvature only after a slope model passes held-out tests.

Require non-negative flux across the observed band:

\[
I_{0,j} + I_{1,j} x_c \ge 0
\]

for every channel `c`. A group or weighted penalty should discourage large
slopes on leaves with negligible `I0`.

### Treat lines as a different family

A power law is inappropriate for a narrow H I line. Keep compact spectral
families as a separate branch:

- one-channel and dyadic multi-channel intervals;
- Hanning or compact spline atoms to reduce sensitivity to arbitrary interval
  edges;
- later, a one-dimensional Haar tree in frequency if line searches become
  large.

A useful spectral analogue of a spatial split is:

- parent: one coefficient over a frequency interval;
- children: two half-interval coefficients;
- detail: their signed Haar contrast;
- acceptance: discovery score, selection choice, then one held-out evaluation.

The current sliding-box bank is adequate for recovery tests. A dyadic/Hanning
dictionary will reduce the 245-candidate trials burden for production search.

### Do not start with a GP

A GP is not the best first frequency model for this narrow, regularly sampled
64-channel band. A slope or low-order Taylor term has clearer physical meaning,
fewer hyperparameters, cheaper validation, and lower risk of absorbing
bandpass residuals.

A frequency GP may become useful for a much wider band, smooth spectral
curvature, or uncertainty estimates. It should then compete against the Taylor
baseline. A time GP remains potentially useful for self-calibration knots, but
the external-calibrator experiment showed that smoothing known gain epochs can
remove real instrumental structure.

## Required calibration gate before spectral fitting

The scan-29 diagnostic currently fits one small-signal gain perturbation shared
by RR and LL over a preselected minute. It is not yet a calibration solution.

The next calibration implementation should:

1. allow separate RR and LL antenna amplitude/phase terms;
2. generate candidate time knots or change intervals using discovery baselines;
3. select duration and ridge strength on selection baselines;
4. refit only the selected model on discovery plus selection;
5. evaluate once on the stage-held-out baselines;
6. compare against a local sky event over the same support;
7. pass temporal-source injection/recovery tests before application;
8. apply an accepted correction to a new frozen native fixture;
9. rerun the null spatial/time/frequency search.

This gate protects real variable sources. A gain model should not be accepted
merely because it lowers training loss.

## Proposed frequency implementation tranche

### Phase A: global and component-group continuum slopes

1. Add a signed first-order spectral response to the composite predictor.
2. Keep the spatial topology and `I0` checkpoint fixed for the first screening
   experiment.
3. Compare achromatic, global-slope, and component-group-slope families.
4. Fit on discovery baselines, select model and regularisation on selection
   baselines, and evaluate only the chosen family.
5. Report aggregate and per-channel residual power.
6. Include a per-antenna low-order bandpass perturbation as a competing family
   or nuisance model.

### Phase B: adaptive per-leaf slopes

1. Form a unit spectral-slope response `x * A_j` for each candidate leaf.
2. Screen leaves in streamed spatial batches using matched residual and Gram
   statistics.
3. Shortlist only leaves with adequate frozen apparent flux or residual
   evidence.
4. Jointly refit selected `I0` and `I1` coefficients on the fixed topology.
5. Enforce positivity at both band edges.
6. Require repeatable held-out improvement across baseline partitions.

### Phase C: compact spectral features

1. Retain the existing exact interval-injection tests as regressions.
2. Add Hanning/dyadic atoms and compare their trials-adjusted recovery threshold
   with sliding boxes.
3. Test on the H I dataset separately from the 3C391 continuum dataset.
4. Allow signed line coefficients because continuum subtraction can expose
   absorption as well as emission.

## Validation and identifiability requirements

Whole-baseline holdout is necessary but not sufficient. Both a true sky
spectrum and an antenna bandpass error can predict unseen baselines.

The spectral study should therefore use several controls:

- retain external calibrator scans as the anchor for bandpass terms;
- repeat baseline partitions;
- check agreement across time ranges and mosaic pointings;
- compare sky-spectrum and antenna-bandpass families directly;
- run zero-injection controls on the real residual;
- inject off-centre spectra so beam chromaticity is exercised;
- inject gain and bandpass errors and verify that they are not selected as sky;
- do not use evaluation data to choose spatial location, support, model order,
  regularisation, or stopping time.

For model mining, require at least:

- positive selection improvement;
- positive held-out improvement after refit;
- the same model family in at least four of five partitions;
- correct recovery of synthetic slopes/lines at a declared flux threshold;
- no repeatable acceptance in the real zero-injection null;
- flatter per-channel residual power without degradation in any pointing.

## Current code map

Core time/frequency recovery:

- `src/sl1mjax/sky_recovery.py`
  - `SkyVariationCandidate`
  - `native_variation_candidates`
  - `temporal_support_mask`
  - `spectral_support_mask`
  - `blind_search_sky_variation`
  - `blind_search_quadtree_sky_variation`

Linear residual family fitting:

- `src/sl1mjax/residual_models.py`
  - tiled real-linear sufficient statistics;
  - ridge-selected residual models;
  - local sky, amplitude, pointing, and antenna-gain scan responses.

Composite sky and forward model:

- `src/sl1mjax/composite.py`
- `src/sl1mjax/direct_operator.py`
- `src/sl1mjax/quadtree.py`
- `src/sl1mjax/beam.py`

Experiment drivers:

- `scripts/ablate_3c391_native_averaging.py`
- `scripts/run_3c391_native_injection_recovery.py`
- `scripts/search_3c391_native_variability.py`
- `scripts/search_3c391_native_spatial_variability.py`
- `scripts/diagnose_3c391_scan_residual.py`

Relevant tests:

- `tests/test_sky_recovery.py`
- `tests/test_residual_models.py`
- `tests/test_3c391_native_injection_recovery_script.py`
- `tests/test_3c391_native_variability_search_script.py`
- `tests/test_3c391_native_spatial_variability_script.py`
- `tests/test_3c391_scan_residual_script.py`

Detailed investigation record:

- `docs/3c391_corner_pixels.md`, especially the sections from
  “Native-resolution averaging ablation” through
  “Complete scan-29 residual-model comparison”.
- `docs/hierarchical_pixels_proposal.md`, especially “Interaction with
  self-calibration” and “Temporal extension”.

## Artifact map

- Native averaging:
  `outputs/3c391_native_averaging_ablation/`
- Matched-support recovery:
  `outputs/3c391_native_injection_recovery/summary.json`
- Blind fixed-position search:
  `outputs/3c391_native_variability_search/summary.json`
- Spatially blind search:
  `outputs/3c391_native_spatial_variability_search/`
- Complete scan-29 diagnosis:
  `outputs/3c391_scan29_residual_diagnostic/`
- Selected fourteen-epoch calibration:
  `outputs/3c391_full_scan_gain_baseline/full_scan_calibration.npz`
- Frozen composite sky protocol:
  `outputs/3c391_composite_catalogue_stage3/protocol.json`
- Frozen sky checkpoint:
  `outputs/3c391_recovery_policy_fit_zero/sealed_active_only.npz`

## Operational notes for Bacchus

- SSH target outside the local network:
  `stephen@hardynet.dyndns.org`
- GPU: RTX 3080 Ti with 12 GB VRAM.
- The old checkout at `/home/stephen/checkouts/SL1MJax` may contain unrelated
  or dirty work. Do not overwrite it blindly.
- Recent isolated source staging used:
  `/home/stephen/checkouts/SL1MJax-spatial-20260827`
- The old checkout's `.venv` contains CUDA-enabled JAX and casacore.
- A typical isolated invocation used the old checkout for data paths and:

  ```bash
  PYTHONPATH=/home/stephen/checkouts/SL1MJax-spatial-20260827/src \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python /path/to/staged/script.py
  ```

The full native 3C391 Measurement Set is already on Bacchus. BagOfWinds is not
required for the next 3C391 experiment.

## Repository state at handoff

Relevant commits, newest first:

- `4185ef5` — document scan-29 gain residual diagnosis;
- `d789bf0` — add scan residual model diagnostic;
- `570dd30` — add spatial native variability search;
- `876aabd` — add blind native variability search;
- `4453c77` — add native variable-sky recovery tests;
- `ef6c6cb` — add native-resolution averaging ablation.

The full local test suite passed with 382 tests and one optional real-MS test
skipped after the scan-29 work.

## Recommended first task in the next session

Implement the bounded time-local calibration gate, apply it to the native C1
fixture, and rerun the zero-injection spatial/time/frequency search. Then begin
frequency work with a global and component-group log-frequency slope ablation.

The key question for that first spectral experiment is:

> Does a low-dimensional intrinsic continuum slope reduce held-out native
> residuals consistently across baselines, time ranges, channels, and mosaic
> pointings after a bandpass/gain nuisance model is included?

Do not start with independent channel maps, a two-dimensional time-frequency
surface, or a GP. Those models add freedom before the simpler physical baseline
has been tested.
