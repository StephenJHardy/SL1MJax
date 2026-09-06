# VLA C-band beam validation notebook proposal

## Purpose

SL1MJax needs a lasting scientific account of the VLA C-band beam work. The
account should explain the measurement problem, show how the implementation
was validated, and present the evidence for or against the CASSBEAM beam using
the THOL0001 holography observations.

An executable IPython notebook is a good format for this account. It can join
technical explanation, equations, measured results, and visual evidence in one
place. It can also let a reader rerun every figure and reported statistic.

The notebook must not become another reduction pipeline. Extracting a 116 GiB
Measurement Set, solving calibration, applying calibration tables, and
recovering hundreds of thousands of holography samples are production tasks.
Those tasks belong in tested command-line programs. The notebook should start
from compact, immutable science outputs produced by those programs.

This proposal defines the notebook structure, publication artifacts, plotting
method, scientific claims, reproducibility rules, tests, and acceptance gates.

## Decision

Create a pre-executed notebook and a rendered documentation page from a compact
versioned validation bundle.

Use three layers:

1. production scripts read the Measurement Set and write science outputs;
2. shared library functions calculate publication statistics and figures from
   those outputs;
3. the notebook calls the shared functions and explains the results.

Commit or otherwise publish all final plots in pre-rendered form. Also embed
them in the executed notebook. A reader should be able to understand the result
without installing Jupyter or running code.

The notebook must rerun without CASA, Bacchus, the original Measurement Set,
or access to `/media/stephen/astro/vla`. It should need only the repository,
the compact validation bundle, and the documented Python environment.

Do not freeze a scientific conclusion merely because the notebook renders.
The notebook displays gate state from the science outputs. It does not promote
an unfrozen beam or turn a warning into a pass.

## Intended outcome

The final publication should let a technically informed reader answer these
questions:

- What beam effect is being measured?
- Why are scalar Airy and Perley beams insufficient for full-polarisation work?
- How was the THOL0001 holography experiment reconstructed?
- How were moving and reference antennas distinguished?
- Which calibration and polarisation conventions were used?
- How closely does the JAX calibration application reproduce CASA?
- How was the empirical voltage beam recovered?
- Does CASSBEAM predict held-out holography data?
- Does it reproduce measured co-polar shape, phase, and R/L squint?
- How much variation exists between antennas and reference antennas?
- Does the result transfer from 4.564 to 4.692 GHz?
- What evidence would be required to accept the off-diagonal Jones terms?
- Which claims are accepted, provisional, failed, or not yet run?

The notebook should be useful as:

- a scientific validation report;
- an implementation-convention reference;
- a reproducible worked example;
- a regression baseline for later beam changes;
- evidence supporting a future paper or software release;
- a concise introduction for a new contributor.

## Scope

### Initial publication

The first accepted notebook covers the lower-C THOL0001 execution and the
diagonal voltage beam:

- 3C147 holography at 4.564 and 4.692 GHz;
- measured AZELGEO raster coordinates;
- antenna-specific R and L co-polar voltage responses;
- reference-antenna and repeated-sample uncertainty;
- Airy, Perley, CASSBEAM, and empirical diagonal comparisons;
- R/L squint magnitude and direction;
- held-out visibility prediction;
- calibration and coordinate convention validation.

### Later extension

The same publication may later add:

- off-diagonal full-Jones holography;
- RL/LR validation and the leakage floor;
- upper-C frequency and execution transfer;
- the unused C147 offset-ring prediction;
- transfer to 3C391 and 24A-063;
- antenna-specific beam models;
- a frozen empirical or CASSBEAM-derived production beam.

These sections must remain visibly marked as `not_run` or `blocked` until their
scientific gates pass. Empty sections should not be filled with simulated
results that resemble measurements.

## Non-goals

The notebook is not intended to:

- download or extract archive products;
- run CASA calibration;
- read the full Measurement Set;
- solve K, B, G, Kcross, D, or X;
- recover the beam from raw visibilities;
- select calibration conventions interactively;
- tune plotting thresholds until a preferred result appears;
- hide failed gates or inconvenient antennas;
- serve as the production beam artifact;
- replace machine-readable validation reports;
- make the full-Jones beam scientific before its cross-hand gates pass.

## Current prerequisite state

The notebook should only be finalized after the following inputs exist:

- a closed lower-C pointing-ingestion gate;
- a consistent 3C147 flux and structure model across fields 0, 9, and 10;
- versioned diagonal and full-polarisation calibration manifests;
- a passing CASA/JAX injected-basis operator oracle;
- a passing real-visibility calibration golden;
- calibrated HOLORASTER visibilities produced from `DATA` rather than the
  untouched raster `CORRECTED_DATA` column;
- diagonal per-reference recovery at 4.564 and 4.692 GHz;
- a completed reference-treatment report;
- declared empirical uncertainty and validity masks;
- held-out visibility scores for every beam being compared.

The current `first_beam_recovery` result is an unfrozen diagnostic. It is a
development input, not yet an accepted publication result. The full-Jones beam
remains blocked by the Df/Df+QU, 3C286, D-smoothness, and cross-hand-floor gates.

## Publication artifact architecture

### Proposed paths

Use paths of this form:

```text
notebooks/
  vla_c_band_beam_validation.ipynb

docs/
  vla_c_band_beam_validation.md
  assets/
    vla_c_band_beam_validation/
      figure_manifest.json
      01_observation_timeline.svg
      02_raster_occupancy.png
      ...

src/sl1mjax/data/
  vla_c_band_beam_validation_v1/
    manifest.json
    claims.json
    observation_summary.json
    calibration_summary.json
    convention_gates.json
    measured_diagonal.zarr/
    comparison_scores.zarr/
    bootstrap_results.zarr/
    plot_tables/

src/sl1mjax/
  beam_validation_outputs.py
  beam_validation_statistics.py
  beam_validation_plots.py

scripts/
  build_vla_c_band_beam_validation_bundle.py
  render_vla_c_band_beam_validation.py
  execute_vla_c_band_beam_validation_notebook.py
```

The exact names may change during implementation. The separation of roles
should not.

### Working outputs and publication bundle

Production runs may write large and temporary products beneath `outputs/` on
Bacchus. Those products are not automatically publication inputs.

The bundle builder should select, validate, reduce, and copy only the data
needed to reproduce the publication. It should fail if an input is missing,
unfrozen where freezing is required, or inconsistent with the declared
manifest.

The compact publication bundle should be immutable after release. A changed
beam, calibration table, mask, statistic, or plotting rule creates a new bundle
version rather than silently changing version 1.

### Size target

Aim for a bundle small enough to clone and execute on a laptop. A target below
50 MB compressed is reasonable. Do not include native visibilities when a
stratified residual table or sufficient statistic can reproduce the published
claim.

If later full-Jones products exceed a sensible repository size, publish the
complete bundle in a durable data archive and commit:

- a compact demonstration subset;
- checksums;
- the archive DOI or stable URL;
- the exact bundle schema;
- a retrieval command.

The notebook should still open with its pre-rendered outputs when the external
bundle is absent.

## Science-output contract

### Manifest

The root `manifest.json` should contain:

- schema version;
- publication version;
- creation timestamp;
- generating Git commit and dirty-tree state;
- source Measurement Set identity and archive hash;
- execution and scheduling block identifiers;
- CASA and SL1MJax versions;
- input artifact paths and hashes;
- calibration product and table hashes;
- source-model identity and coefficients;
- correlation order and receptor order;
- coordinate frames and units;
- parallactic-angle model;
- D-application contract;
- reference and on-axis gauge definitions;
- selected fields, scans, SPWs, channels, antennas, and masks;
- holdout definitions and random seeds;
- numerical tolerances;
- plot-code version;
- every included file and its checksum.

The manifest must distinguish data provenance from scientific acceptance. An
artifact can be checksummed and reproducible while remaining unfrozen or
scientifically rejected.

### Claim registry

Store every displayed scientific conclusion in `claims.json`. Each claim
should contain:

- stable claim identifier;
- plain-language statement;
- status: `pass`, `warn`, `fail`, `blocked`, or `not_run`;
- quantitative metric and uncertainty;
- acceptance threshold;
- input artifact hashes;
- validation split;
- supporting figure and table identifiers;
- limitations and exclusions;
- code version that evaluated the claim.

The notebook should render this registry. It should not recreate claim status
from prose or notebook cell order.

### Measured diagonal product

The compact measured product should preserve enough information to reproduce
the diagonal figures and validation statistics. Its labelled dimensions should
include, where applicable:

- frequency;
- moving antenna;
- reference antenna;
- raster pass;
- spatial sample or commanded coordinate;
- hand R or L;
- repeat or time group.

Variables should include:

- measured AZELGEO $l,m$ in radians;
- complex voltage response;
- absolute and on-axis-normalized voltage response;
- per-hand validity;
- sample and baseline counts;
- reference scatter;
- repeated-sample scatter;
- empirical uncertainty;
- beam-model predictions at the same coordinates;
- recovery and holdout membership;
- reason-coded exclusions.

Do not replace the measured pass-2 coordinates with the nominal Memo lattice.
Grid indices may be included as diagnostic labels, but the measured coordinate
remains authoritative.

### Validation score product

Keep stratified scores rather than only global means. Required strata include:

- beam model;
- frequency;
- hand or correlation;
- moving antenna;
- reference antenna;
- raster pass;
- radius bin;
- amplitude-support bin;
- holdout type;
- spatial cell or repeat group where practical.

This lets the notebook show paired differences and bootstrap uncertainty
without reading the original visibilities.

### Precision

Store complex values at sufficient precision to reproduce the accepted
tolerances. Reducing float64 outputs to float32 is allowed only after a test
shows that all published statistics and gate decisions remain unchanged.

## Shared analysis and plotting code

Do not implement substantial statistics directly in notebook cells. Put them
in importable, tested functions.

The shared code should provide:

- bundle loading and schema validation;
- claim-registry loading;
- coordinate and unit conversion;
- support and validity masking;
- antenna/reference stratification;
- paired residual calculation;
- bootstrap intervals;
- radial and axial profile calculation;
- phase gauge removal and connected-region unwrapping;
- squint estimation;
- beam-centre and width estimation;
- plot creation and deterministic styling;
- figure-sidecar creation.

The notebook and render script must call the same functions. Do not maintain a
notebook implementation and a separate publication implementation.

Plot functions should accept explicit arrays and configuration. They should
not search the filesystem, inspect environment variables, or silently select
the newest run.

## Notebook structure

### 1. Executive summary

Open with the outcome rather than the processing history.

Include:

- the current scientific status of the diagonal and full-Jones beams;
- a concise table of accepted and blocked claims;
- the principal held-out comparison;
- the principal co-polar and squint result;
- a clear statement of the supported frequency and coordinate domain.

This section should make sense when viewed without executing any code.

### 2. Why a measured voltage beam is needed

Describe the inference problem. An incomplete beam can appear as spatial,
spectral, polarised, or temporal sky structure.

Introduce the full measurement equation:

$$
V_{pq}(t,\nu)=
\int E_p(s,\nu,t)
C(s,\nu,t)
E_q^{\rm H}(s,\nu,t)
e^{-2\pi i\boldsymbol u_{pq}\cdot s}\,d\Omega.
$$

Explain the difference between:

- a scalar power beam;
- a diagonal R/L voltage beam;
- a full $2\times2$ Jones beam;
- an array-average beam;
- an antenna-specific beam.

State that CASSBEAM is a physical beam model whose conventions and predictive
accuracy must be validated against measurements.

### 3. THOL0001 observation

Describe the lower-C execution and its role in EVLA Memo 195.

Show:

- the observation timeline;
- calibrator and raster scans;
- moving and reference antenna sets;
- native spectral coverage;
- the locations of 4.564 and 4.692 GHz;
- dense and sparse raster occupancy;
- measured off-lattice pass-2 coordinates;
- pointing transition masks.

Explain why `POINTING_OFFSET` is an AZELGEO antenna coordinate and why
`ON_SOURCE` is not a valid selection field in this observation.

### 4. Calibration and source model

Describe the calibration ladder and the separation between:

- the CASA compatibility backend;
- the scientific calibration product;
- the future native continuous Jones model.

Document:

- the consistent 3C147 model across fields;
- K, B, G, Kcross, Df or Df+QU, and Xf;
- `calwt` and `parang` settings;
- reference antenna and global-X treatment;
- unsupported antennas and channels;
- the absolute flux gauge;
- the relative on-axis beam normalization.

Include validation of the point-source or structure model on held-out
reference--reference data. State any remaining closure scatter.

### 5. Convention and software correctness

Summarize the injected-basis oracle and real-visibility golden.

Show cumulative residuals for:

- K+B+G;
- Kcross;
- first-order D;
- Xf;
- parallactic angle.

Explain the source-derived CASA contract:

- K-table reference frequency;
- float32 amplitude/phase CPARAM interpolation;
- bracket and flag behaviour;
- first-order `JonesGenLin` D inversion;
- casacore J2000-to-HADEC parallactic angle using geocentric latitude;
- per-antenna ITRF direction for antenna-position phase.

The figures in this section validate software semantics. They are not evidence
that CASSBEAM is physically correct.

### 6. Holography recovery

Present the compact-source equation:

$$
V_{mr}=E_m S E_r^{\rm H}.
$$

Explain how the moving-antenna voltage is recovered against each reference:

$$
\widehat E_m^{(r)}=V_{mr}(S E_r^{\rm H})^{-1}.
$$

Document:

- baseline orientation and conjugation;
- independent RR and LL validity;
- per-reference recovery;
- reference identity and on-axis gauges;
- absolute and relative normalization;
- empirical uncertainty;
- why references are not combined before their treatment report passes.

Show the counts for per-reference samples, antenna-time groups,
moving-antenna spatial cells, and shared spatial coordinates. Use these names
consistently.

### 7. Measured diagonal voltage beam

Show R and L separately.

Required views include:

- voltage-amplitude maps;
- voltage-phase maps after the declared antenna gauge removal;
- validity and sample-density maps;
- reference scatter maps;
- moving-antenna scatter maps;
- dense and sparse raster maps;
- comparable-coordinate differences between passes;
- on-axis absolute and normalized distributions.

Phase smoothness should be calculated within connected regions and reported at
several power thresholds. Near-null phase must not dominate a main-lobe
smoothness claim.

### 8. CASSBEAM and analytic comparisons

Evaluate every model at the exact measured coordinates and frequencies.

Compare:

- analytic Airy diagonal;
- Perley scalar co-polar beam;
- CASSBEAM diagonal;
- empirical diagonal holography.

Required comparisons include:

- measured versus predicted complex voltage;
- amplitude residual maps;
- phase residual maps;
- radial profiles;
- principal-axis cuts;
- residual versus radius;
- results by moving antenna;
- results by reference antenna;
- results by raster pass;
- results by frequency.

Do not present interpolated colour maps without also displaying measured
support. An interpolation used only for display must be labelled and must not
enter a validation score.

### 9. R/L squint

Measure squint from main-lobe power, not a full-raster moment.

Show:

- R and L main-lobe contours;
- fitted or centroid beam centres;
- the R-to-L displacement vector;
- the Memo $2.4/\nu_{\rm GHz}$ expectation;
- CASSBEAM sampled on the identical coordinates and masks;
- estimator bias from sampled CASSBEAM;
- empirical/CASSBEAM ratio with uncertainty;
- per-antenna and per-reference distributions.

Use the same estimator, threshold, flags, grouping, and coordinate samples for
the empirical and CASSBEAM results.

### 10. Held-out visibility validation

This section supplies the principal scientific evidence.

Use declared splits such as:

- leave-one-reference-antenna-out;
- held-out samples within repeated dwells;
- later visits to comparable coordinates;
- the shared origin and close cross-pass pairs;
- leave-one-moving-antenna-out after an array model exists;
- interleaved spatial cells after a spatial model exists;
- 4.692 GHz frequency transfer after a frequency model exists;
- C147 offset-ring prediction without fitting those fields.

For each split, show:

- train and holdout definitions;
- active sample counts;
- paired model residuals;
- bootstrap uncertainty over independent groups;
- results by RR and LL;
- amplitude and phase residuals;
- whether the claimed winner is consistent across antennas and passes.

Do not treat individual visibilities as independent bootstrap samples. Use
antenna, reference, scan, repeat, or spatial-cell groups appropriate to the
claim.

### 11. Frequency transfer

Recover 4.692 GHz independently before fitting a frequency model.

First compare the two independent measured planes after the declared angular
scaling. Then test any frequency model on held-out data.

Show:

- width scaling;
- squint scaling;
- complex residual after $\nu\theta$ coordinate scaling;
- antenna stability between frequencies;
- regions where interpolation or extrapolation is unsupported.

The second frequency is not a valid holdout if it helped select or fit the
frequency model being scored.

### 12. Full-Jones extension

Keep this section present but explicitly blocked until its gates pass.

When opened, add:

- the Df/Df+QU uncertainty floor;
- 3C286 Q/U, EVPA, and V apply-back;
- D smoothness and antenna support;
- empirical RL/LR noise floor;
- recovered off-diagonal Jones amplitude and phase;
- held-out RL/LR prediction;
- RR/LL non-regression;
- parallactic-angle and frequency transfer;
- supported off-diagonal spatial domain.

The full-Jones section must not reuse a diagonal success as evidence for
leakage correctness.

### 13. Conclusions and limitations

End with the claim registry and a compact statement of what is supported.

State limitations such as:

- two initial frequencies;
- finite raster resolution;
- sparse cross-pass coordinate overlap;
- antenna and epoch coverage;
- calibrator structure uncertainty;
- reference and gain-transfer uncertainty;
- measured spatial support;
- blocked or untested full-Jones terms;
- status of upper-C and independent-science transfer.

Include failed and inconclusive results. They define the applicability domain
and protect later analyses from overstating the validation.

## Proposed figure set

Every figure should have a stable identifier. The identifier should appear in
the filename, notebook caption, rendered document, claim registry, and figure
manifest.

| ID | Figure | Main purpose |
|---|---|---|
| F01 | Observation and calibration timeline | Show the experiment and calibration anchors |
| F02 | Dense and sparse raster occupancy | Prove measured pointing support |
| F03 | Moving/reference antenna map | Show the holography baseline geometry |
| F04 | CASA/JAX cumulative operator residual | Lock calibration application semantics |
| F05 | Absolute and normalized on-axis response | Separate flux scale from beam normalization |
| F06 | R/L empirical voltage amplitude maps | Display measured co-polar response |
| F07 | R/L empirical voltage phase maps | Display complex beam phase and gauge treatment |
| F08 | Reference scatter and validity | Show empirical uncertainty and support |
| F09 | Antenna-to-antenna variation | Show whether an array average is adequate |
| F10 | CASSBEAM minus empirical amplitude | Locate physical model disagreement |
| F11 | CASSBEAM minus empirical phase | Locate complex model disagreement |
| F12 | Radial and axial profiles | Compare beam width, nulls, and asymmetry |
| F13 | R/L squint vectors | Compare empirical, sampled CASSBEAM, and Memo |
| F14 | Held-out paired residual differences | Establish predictive performance |
| F15 | Residuals by antenna/reference/pass | Detect concentrated failures |
| F16 | Independent frequency comparison | Test 4.564 to 4.692 GHz behaviour |
| F17 | C147 offset-ring prediction | Test an unused pointing set |
| F18 | Full-Jones cross-hand validation | Reserved until the full-Jones gate opens |

Not every figure must appear in the executive summary. All accepted claim
figures should appear in the notebook and rendered documentation.

## Plotting methodology

### Coordinate convention

Use a consistent displayed coordinate convention throughout. State:

- the native measured frame;
- whether the horizontal axis increases with positive or negative $l$;
- whether coordinates describe commanded pointing or source relative to beam;
- the conversion from radians to arcminutes;
- any feed-frame rotation.

Never correct a sign only in the plotting layer. Displayed coordinates must be
derived from the same named transformation used by the forward model.

### Complex quantities

Plot voltage amplitude and phase separately. Do not encode complex values in a
single colour wheel for quantitative comparison.

Mask phase where power is below a declared threshold. Unwrap phase only within
connected supported regions. State the removed gauge in every phase caption.

### Colour scales

Use fixed colour limits for paired empirical/model panels. Residual panels
should use symmetric limits around zero where appropriate.

Do not choose colour limits independently for each antenna when the purpose is
to compare antennas. If an individual diagnostic needs a different limit,
state it in the caption.

### Measured support

Distinguish:

- directly measured cells;
- model evaluations at measured coordinates;
- interpolated display surfaces;
- unsupported regions.

Unsupported regions should be transparent or hatched. They should not appear
as numerical zero.

### Uncertainty

Show uncertainty on profiles, squint, and model comparisons. Use grouped
bootstrap intervals or direct repeat/reference scatter as appropriate.

Record the resampling unit and seed. Do not bootstrap millions of correlated
visibility samples as though they were independent.

### Captions

Each caption should say:

- what is plotted;
- which data are train and holdout;
- which masks are active;
- whether the values are absolute or normalized;
- what uncertainty means;
- the practical conclusion;
- which claim identifier the figure supports.

## Pre-rendered output policy

### Required formats

Produce a lossless or high-resolution PNG for image-like maps. Prefer SVG for
line plots, diagrams, and tables when the renderer remains stable.

The executed notebook should embed its visible plots. The documentation page
should reference the files beneath `docs/assets/` so it renders without
executing the notebook.

### Figure manifest

For every figure, record:

- figure identifier;
- filename and checksum;
- generating function;
- plot configuration;
- source bundle checksum;
- claim identifiers;
- creation timestamp;
- software versions;
- deterministic seed if used.

### Stale-output protection

The renderer should fail if an existing figure was generated from a different
bundle or plotting configuration unless an explicit overwrite option is used.

A verification command should regenerate figures into a temporary directory
and compare their data-level sidecars. Pixel hashes may be too sensitive to
renderer versions, so numerical plot tables and metadata are authoritative.

## Notebook execution policy

### Clean execution

The committed notebook must be produced by a clean restart-and-run-all. Manual
cell execution order is not acceptable.

The execution wrapper should:

- create a temporary output notebook;
- execute with a fixed working directory;
- set deterministic random seeds;
- disable network access by convention;
- enforce a timeout per cell;
- capture the environment and bundle hashes;
- refuse unexpected warnings where practical;
- copy the result into place only after all checks pass.

### Cell design

Each section should have:

1. short explanatory Markdown;
2. a small data-loading or plotting call;
3. a visible result;
4. a caption or interpretation.

Avoid large hidden state. A reader should not need to inspect twenty prior
cells to understand which mask or beam is active.

### Dependencies

Add notebook tooling as an explicit documentation or validation dependency
group. Likely requirements include:

- Jupyter or an IPython kernel;
- `nbformat`;
- `nbclient` or `nbconvert`;
- Matplotlib;
- NumPy, xarray, and Zarr from the main project.

Pin or lock the environment used for the released notebook.

## Statistical methodology

### Primary endpoint

The primary endpoint should be held-out complex-visibility prediction. Beam
map resemblance is supporting evidence.

For beam model $b$, calculate a held-out score such as

$$
L_b=
\frac{\sum_i w_i\left|V_{b,i}-V_i\right|^2}
     {\sum_i w_i},
$$

but do not call this reduced $\chi^2$ unless the weights are calibrated inverse
variances and the degrees of freedom are included.

Report paired differences against the selected reference model:

$$
\Delta L_b=L_b-L_{\rm reference}.
$$

Negative $\Delta L_b$ means the candidate predicts the holdout better.

### Weighting

Nominal Measurement Set weights are provenance until their noise meaning is
validated. Use empirical uncertainty from references, repeats, channels, or
calibrator residuals where possible.

Show weighted and robust unweighted summaries when a small number of bright or
high-weight samples could dominate the result.

### Multiple strata

A global improvement is not sufficient if it hides severe failures in one
hand, pass, antenna, or beam region.

Require non-regression or an explained limitation in:

- RR and LL separately;
- dense and sparse passes;
- main lobe and outer support;
- moving antennas;
- reference antennas;
- both frequencies.

### Model selection and sealed data

Any split used to choose a coordinate convention, threshold, interpolator,
normalization, or beam parameter is validation data, not a final test.

Keep at least one transfer axis sealed until the analysis rules are fixed. The
upper-C execution and independent science observations are natural sealed
tests.

## Scientific claim gates

### Gate N0: bundle integrity

Require:

- every file matches its manifest checksum;
- schema validation passes;
- no absolute Bacchus paths are required;
- every declared claim resolves to data and figures;
- an unfrozen artifact cannot be labelled frozen.

### Gate N1: notebook reproducibility

Require:

- clean restart-and-run-all succeeds;
- execution requires no network, CASA, or Measurement Set;
- all numerical tables match the bundle;
- every pre-rendered figure has a valid sidecar;
- no cell depends on execution order outside the notebook order.

### Gate N2: software correctness narrative

Require:

- the CASA source-derived apply contract is recorded;
- injected-basis and real-visibility golden results are shown;
- correlation, receptor, baseline, D, X, P, and antpos conventions are stated;
- software agreement is not presented as beam validation.

### Gate N3: diagonal beam evidence

Require:

- calibrated absolute and normalized recovery are distinguished;
- reference treatment passes;
- empirical support and uncertainty are visible;
- both Memo frequencies are recovered independently;
- held-out RR/LL predictions are reported;
- squint estimator bias is measured on sampled CASSBEAM;
- conclusions are stable across declared antenna and pass strata.

### Gate N4: CASSBEAM acceptance claim

Require:

- CASSBEAM is evaluated at identical coordinates, frequencies, flags, and
  groupings;
- held-out prediction is the primary evidence;
- amplitude, phase, width, null, and squint diagnostics are reported;
- disagreements are localized and retained in the applicability statement;
- acceptance thresholds were fixed before opening sealed transfer data.

### Gate N5: full-Jones publication

Require:

- all polarisation calibration gates pass;
- off-diagonal support is explicit;
- held-out RL/LR improve relative to the diagonal model;
- RR/LL do not materially regress;
- handedness and parallactic transfer are demonstrated;
- the claim registry changes from `blocked` only through a versioned run.

### Gate N6: rendered release

Require:

- notebook, Markdown page, assets, and bundle share one release identifier;
- all links resolve;
- plots are legible at documentation-page width;
- captions contain units and claim identifiers;
- repository tests and notebook-specific tests pass;
- a fresh temporary-directory rebuild reproduces the release.

## Testing strategy

### Unit tests

Test:

- bundle schema validation;
- checksum validation;
- coordinate conversion and orientation;
- support masks;
- phase gauge removal;
- connected-region phase unwrapping;
- radial and axial profiles;
- squint estimation on manufactured beams;
- sampled-estimator bias;
- paired score calculation;
- grouped bootstrap determinism;
- claim-to-figure linkage;
- stale-figure detection.

### Manufactured-data tests

Generate compact manufactured beams with known:

- width;
- R/L displacement;
- amplitude asymmetry;
- smooth phase gradient;
- phase wrap;
- null;
- unsupported outer region;
- antenna variation.

Sample them on the actual dense and sparse coordinate patterns. Verify that the
published estimators recover the known values or report their sampling bias.

### Golden tests

Use compact fixed inputs to protect:

- the principal figure data;
- claim metrics;
- table formatting;
- coordinate signs;
- colour-scale configuration;
- notebook section order and required headings.

Prefer numerical plot-table goldens over pixel-perfect image comparisons.
Retain a small number of image dimension and non-empty-content checks.

### Notebook smoke test

Run the complete notebook against the compact bundle in a temporary directory.
Check:

- successful execution;
- absence of unexpected errors;
- expected claim statuses;
- expected figure count;
- no access outside the repository and temporary directory;
- reasonable execution time and peak memory.

### Documentation checks

Check:

- Markdown links;
- asset presence;
- figure manifest consistency;
- alt text for every figure;
- no unsupported claim language when a gate is warn, blocked, or not run.

## Phased implementation plan

### Phase 0: freeze the narrative contract

Define:

- notebook title and audience;
- claim identifiers;
- section order;
- initial figure list;
- bundle schema;
- accepted terminology;
- diagonal and full-Jones scope boundaries.

Gate: reviewers can map every planned conclusion to a science output and a
validation gate.

### Phase 1: build the compact bundle

Implement the bundle builder and schema validation. Populate it from the
accepted lower-C diagonal products.

Gate: bundle checksums pass and no notebook input points at Bacchus.

### Phase 2: shared statistics

Implement and test the scientific summary functions. Validate them on
manufactured beams and the compact bundle.

Gate: every proposed table and claim can be reproduced without plotting.

### Phase 3: shared plots

Implement deterministic plotting functions and the figure manifest. Produce
the full figure set from the bundle.

Gate: all figures have sidecars, units, masks, support declarations, and claim
links.

### Phase 4: notebook

Write the explanatory narrative and call the shared analysis and plotting
functions. Keep code cells short.

Gate: restart-and-run-all succeeds on a laptop without CASA or the MS.

### Phase 5: rendered documentation

Generate the Markdown or HTML publication and pre-rendered assets. Review the
result at normal documentation width and on a narrow display.

Gate: the rendered document is complete and understandable without executing
the notebook.

### Phase 6: scientific review

Review each claim against its evidence and gate. Check that failed and blocked
claims remain visible.

Gate: diagonal conclusions are supported by held-out measurements and no
full-Jones conclusion exceeds its calibration evidence.

### Phase 7: release and archive

Assign a publication version. Record repository commit, bundle hash, notebook
hash, and figure manifest hash. Optionally archive the bundle and rendered
document with a DOI.

Gate: a clean checkout can verify and rerun the released publication.

## Review checklist

Before accepting the notebook, confirm:

- [ ] The notebook does not read the full Measurement Set.
- [ ] The notebook does not invoke CASA.
- [ ] All plots are visible without execution.
- [ ] All plots rerun from compact science outputs.
- [ ] The executed notebook came from a clean run-all.
- [ ] The rendered page uses the same figure functions and bundle.
- [ ] Source-data and calibration hashes are recorded.
- [ ] Coordinate frames and signs are explicit.
- [ ] Absolute and normalized beam amplitudes are not confused.
- [ ] RR and LL are shown separately.
- [ ] Phase gauges and masks are stated.
- [ ] Measured and interpolated support are visually distinct.
- [ ] CASSBEAM and empirical data use identical sampling.
- [ ] Squint estimator bias is included.
- [ ] Validation splits are declared before scores.
- [ ] Bootstrap units reflect correlated data structure.
- [ ] Held-out visibility prediction is the primary endpoint.
- [ ] Per-antenna and per-reference failures remain visible.
- [ ] Frequency transfer does not train on its own holdout.
- [ ] C147 offset fields remain unused until prediction.
- [ ] Upper-C remains sealed until its protocol is fixed.
- [ ] Full Jones remains blocked until its separate gates pass.
- [ ] Every prose conclusion maps to a claim identifier.
- [ ] Every claim maps to machine-readable evidence.
- [ ] No compatibility result is described as physical validation.
- [ ] No unfrozen diagnostic is described as a production beam.

## Final recommendation

Build the notebook after the diagonal validation protocol stabilizes, but
define the output schema and claim registry now. This lets current science
scripts emit publication-ready outputs instead of requiring later forensic
reconstruction.

Keep the heavy computation in tested production programs. Keep statistics and
plots in shared library functions. Use the notebook to explain and display a
fixed validation bundle.

This arrangement gives the desired result: a visually complete document that
is immediately readable, scientifically auditable, and rerunnable from compact
science outputs long after the original Bacchus working directories have been
removed.
