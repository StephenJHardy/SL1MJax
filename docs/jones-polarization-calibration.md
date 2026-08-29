# Jones and polarisation calibration plan

## Purpose

This is the working plan for moving SL1MJax from diagonal parallel-hand
calibration to on-axis full Stokes. It supersedes the “defer cross-hands”
boundary in [3c391_calibration_reference.md](3c391_calibration_reference.md)
and Phase 7 of [calibration_development_plan.md](calibration_development_plan.md)
for sequencing. The older documents remain the record of the completed
K/B/G tranche.

The scientific aim is a calibration model whose frequency-dependent freedom is
constrained by calibrators, independently of the target sky. Otherwise
bandpass, R/L gain, R–L phase, or leakage can absorb the spectral and
polarised structure we want to discover, or the sky can explain residual
instrumental structure.

CASA remains an independent oracle. Runtime solving and application must not
require CASA.

## Decisions

Accepted:

- Change the calibration representation before adding sky \(q,u,v\) or
  per-pixel spectra.
- Keep 3C391 TDEM0001 as the first engineering dataset. It already has all
  four correlations, 3C286, 3C84, and the existing K/B/G path.
- Reproduce CASA D/X (and cross-hand delay) by applying imported tables
  first, then solving in JAX one term at a time.
- Keep calibration frequency structure conservative and calibrator-anchored.
- Treat a 3C391 \(v\) or \(q+iu\) detection as credible only if it exceeds
  an *independent* calibrator floor (3C286 with Kcross/Xf solved and
  evaluated on different scans or connected-baseline cohorts, and leakage
  checks that are not the same 3C84 scan used to solve Df), on held-out
  frequency channels as well as baselines, times, and pointings. Same-data
  3C286 apply-back is an engineering residual, not that floor.
- First target sky after frozen calibration: constant global \(q,u,v\) with
  **no RM**. That is a detection and calibration-check model, not a spatial
  polarisation image. It forces \(Q_k=qI_k\), \(U_k=uI_k\), \(V_k=vI_k\)
  everywhere and can miss polarised structure that cancels globally.
- Imaging sequence: global detection \(\to\) spatial polarisation
  activation on the accepted Stokes-I topology \(\to\) self-cal \(\to\)
  frequency/RM discovery. Polarisation activation is a separate decision
  from Stokes-I spatial splits. Candidate blocks are unpolarised; joint
  \(q,u\); \(v\); and full \(q,u,v\), with \(q^2+u^2+v^2\le 1\).
- Stokes-I spectral discovery does not block initial polarisation imaging.
  Reuse the accepted Stokes-I morphology at a reference frequency; defer
  new spectral freedom.
- Establish that constant-frequency spatial baseline before introducing
  frequency-dependent polarisation calibration or sky terms. Per-channel
  residuals remain diagnostics during the first imaging milestones, not
  new fitted degrees of freedom. RM is the next frequency model after
  that baseline, not part of the first \(q,u,v\) fit.
- Initial diagnostic images may use `casa_parallel_preserving`. Those
  results stay exploratory while exact 2×2 \(D\) still produces false
  \(V\). Exact-Jones refinement is required before evidence-grade claims;
  it does not block looking at images.
- Anchored self-cal starts only after the frozen sky represents the
  important spatial polarised structure. A global \(q,u,v\) model alone is
  not enough for an extended remnant.
- Inventory BagOfWinds for polarimetric products, but do not block first
  3C391 images on finding pipeline D/X tables. The current C-band SRDP cal
  tarballs are diagonal.

Deferred until the on-axis 2×2 path passes:

- Direction-dependent beam Mueller terms and squint-as-leakage. C-band
  artifacts and conventions are inventoried in
  [`vla-beam-reference-inventory.md`](vla-beam-reference-inventory.md).
- Multi-component synchrotron plus free-free polarised mixtures as a
  production sky.
- RM-synthesis or spatially varying RM. The 4.536–4.662 GHz span is about
  2.7% in frequency; Faraday resolution is set by the \(\lambda^2\) span
  ([Brentjens & de Bruyn 2005](https://arxiv.org/abs/astro-ph/0507349)).
  Given that narrow band and the imaging-first decision, test spatial
  polarised structure before any RM.
- Requiring a second VLA band or configuration before first 3C391
  polarisation images. Wider L/S data remain later RM leverage, not the
  first imaging milestone.
- Keep [`analyze_calibration_weather.py`](../scripts/analyze_calibration_weather.py)
  as *instrument weather*: an allegory for how the intervening medium and
  hardware state change the calibration, and on what timescales those
  features move. It is not a site-meteorology analysis. Joining the MS
  `WEATHER` table remains a side branch, not a rename.

## Why polarisation and frequency are coupled

Once full Stokes is enabled, \(q\), \(u\), and \(v\) must not be assumed
constant in frequency by default.

Write complex linear polarisation \(P(\nu)=Q(\nu)+iU(\nu)\). For one
synchrotron component,

\[
I_s(\nu)=A_s\left(\frac{\nu}{\nu_0}\right)^{\alpha_s},
\qquad
P_s(\nu)=I_s(\nu)\,p_s\,e^{2i[\chi_0+\mathrm{RM}\lambda^2]}\,D(\lambda^2).
\]

Faraday rotation mixes \(Q\) into \(U\). They must not receive unrelated
spectral polynomials. Internal and external Faraday structure can also
depolarise ([Burn 1966](https://adsabs.harvard.edu/pdf/1966MNRAS.133...67B);
[Sokoloff et al. 1998](https://academic.oup.com/mnras/article/299/1/189/1015189)).

Unpolarised free-free mixed with synchrotron gives \(I=I_s+I_{\mathrm{ff}}\)
and \(P\approx P_s\), so the observed fraction
\(q+iu=P_s/(I_s+I_{\mathrm{ff}})\) changes with frequency even if \(p_s\) is
intrinsic and constant. Multiple synchrotron components can cancel
differently with frequency.

Stokes \(V\) is separate: ordinary Faraday rotation mixes \(Q\) and \(U\)
but not \(V\). Intrinsic circular polarisation and Faraday conversion can
still give \(V(\nu)\) its own spectrum. A single shared spectral index for
\(I,Q,U,V\) is only the simplest null model.

A component sky that respects this is

\[
\begin{aligned}
I(\nu)&=\sum_c I_c(\nu),\\
Q(\nu)+iU(\nu)&=\sum_c I_c(\nu)\,p_{L,c}(\nu)\,e^{2i\chi_c(\nu)},\\
V(\nu)&=\sum_c I_c(\nu)\,v_c(\nu),
\end{aligned}
\]

with \(Q^2+U^2+V^2\le I^2\) at every frequency. That is the long-term sky
language. It is not the first thing to fit on 3C391.

On this band, synchrotron versus free-free separation will be weak without
external data or strong priors. A large RM can still move \(Q\) and \(U\)
because they depend on \(\lambda^2\). Complex Faraday structure will not be
resolved.

## Calibration–sky degeneracies

| Sky measurement | Main calibration degeneracy | Required calibrator constraint |
|---|---|---|
| \(I(\nu)\), spectral index and curvature | Common bandpass amplitude and flux-scale spectrum | Resolved spectral model for 3C286 (already in the K/B/G path) |
| \(v=V/I\) | Relative R/L gain and bandpass amplitude | Calibrator with \(V=0\), including uncertainty on that assumption (3C286 in the NRAO recipe) |
| \(q,u\), EVPA and RM | R–L delay, phase, and feed rotation | Known \(Q/U\) spectrum and EVPA (3C286) |
| Depolarisation or spatially varying \(q/u/v\) | Frequency-dependent leakage and beam Mueller terms | Low-polarisation leakage calibrator, parallactic-angle coverage; off-axis later (3C84) |

A global visibility-scale split \(\mathrm{RR}\to(1+\delta)M\),
\(\mathrm{LL}\to(1-\delta)M\) is identical to a global sky \(v=\delta\).
Per-hand scales do not break that degeneracy. The external \(V=0\) anchor is
what does.

## Current code and data

Snapshot of the tree and BagOfWinds facts. Dated notes below record what
changed after this baseline.

### Representation

[`CalibrationSolution`](../src/sl1mjax/calibration.py) writes schema
**v3**. Schema v2 introduced receptors ≠ correlations. Schema v3 added
Kcross, Df, Xf, circular P, named D operators (`exact` vs
`casa_parallel_preserving`), and solution validity domains. Schema 1 files
promote by inferring receptors from the products. Schema 1–2 readers
reject v3 rather than applying diagonal G/K/B and ignoring D/X.

Gains, delays, and bandpass are stored per feed receptor (`R`,`L` or
`X`,`Y`), not per packed correlation product. `receptor_count` is
`len(receptors)`. `(RR, RL, LR, LL)` is two receptors, not four.
Application packs products into a 2×2 coherency, forms
\(C^{\mathrm{obs}}=J_p C^{\mathrm{sky}} J_q^{\mathrm{H}}\) (or the inverse
to correct), and unpacks. Diagonal G/K/B promote to diagonal 2×2 matrices.

``apply_calibration`` builds
\(J=J_{GKB}J_{\mathrm{Kcross}}J_D J_X J_P\) with \(P\) closest to the sky.
Kcross / Df / Xf *solves* exist as NumPy direct estimators
([`polarization_inference.py`](../src/sl1mjax/polarization_inference.py)):
CASA-compatible initializers, not a self-calibration engine. First-order
Df is labelled `casa_parallel_preserving`. Diagonal Optax G/K/B solvers
still require one parallel-hand product per receptor. Exact 2×2 iterative
refinement is later.

This packing is the feed coherency. It is not the sky Stokes-V packing
\(V=(RR-LL)/2\) used by circular contrast.

Circular-feed parallel-hand contrast \(I_{\mathrm{RR}}=I(1+v)\),
\(I_{\mathrm{LL}}=I(1-v)\) exists as a diagnostic and operator option. It is
not leakage or linear-polarisation calibration.

### 3C391 locally (`data/`)

Copied Measurement Set (2026-08-28): `data/3c391_ctm_mosaic_10s_spw0.ms`
and the working copy `data/3c391_work_v2/3c391_ctm_mosaic_10s_spw0.ms`.
Scripts default here instead of BagOfWinds.

The MS has `CORR_TYPE` 5–8 (RR, RL, LR, LL), 64 channels at 2 MHz from
4.536 GHz, 845,379 rows, fields `J1331+3030` (3C286), `J1822-0938`,
`3C391 C1`–`C7`, and `J0319+4130` (3C84). The `WEATHER` subtable is
present. `SYSPOWER` and `CALDEVICE` are not.

`data/3c391/reference-v2` is antpos, K, B, G, and fluxscale only. The
committed golden fixture keeps RR and LL after that chain. Polarisation
tables go to `data/3c391/reference-pol`. The NRAO continuum tutorial never
ran `polcal`; it points at 3C75 for polarimetry. 3C84 and 3C286 are in the
MS with the usual NRAO roles (leakage and polarisation angle / \(V=0\)).
3C84 is a single ~5-minute scan at the end of the track, so leakage has
almost no parallactic coverage.

### SRDP calibration corpus

`/Volumes/BagOfWinds/NRAO/srdp-cals/` holds 29 archive tarballs. Seven are
extracted as CASA tables. Those tables are pipeline G, K, B, opacity,
gain-curve, switched-power (`G EVLASWPOW`), and antenna position. There are
no D, X, or cross-hand delay products in the extracted set.
[`analyze_srdp_calibration_corpus.py`](../scripts/analyze_srdp_calibration_corpus.py)
only names `finaldelay`, `finalBPcal`, `averagephasegain`,
`finalampgaincal`, and `finalphasegaincal`.

This corpus remains useful for diagonal cadence, empirical G/K/B priors, and
switched-power statistics. It is not a polarimetric solution library.

[`outputs/calibration_weather.json`](../outputs/calibration_weather.json)
summarises those diagonal tables (median gain cadence about 9 minutes,
median amplitude step about 0.24%, phase-gain step p95 about 20.5°). It
does not join `WEATHER`.

## Work order

### 1. Versioned 2×2 calibration representation — done 2026-08-28

Replaced the correlation-as-receptor assumption.

Shipped shape:

- receptor basis, two receptors `(R, L)` or `(X, Y)`;
- a separate ordered correlation-product axis;
- application
  \(C^{\mathrm{obs}}_{pq}=J_p C^{\mathrm{sky}}_{pq} J_q^{\mathrm{H}}\)
  followed by packing into RR, RL, LR, LL (or XX, XY, YX, YY);
- diagonal \(G\), \(K\), \(B\) stored per receptor and promoted to
  diagonal 2×2 matrices;
- schema v2 write; schema 1 read with promotion. Polarimetric solutions
  later write schema v3.

Not stored yet, apply-ready later:

- direction-dependent beam.

Gates:

- [x] synthetic identity and diagonal Jones recover packed parallel hands
      and cross-hands;
- [x] a non-diagonal \(D\) mixes a Stokes-I sky into RL/LR at the predicted
      level;
- [x] the committed RR/LL 3C391 golden still applies after promotion.

Still later in this item: serialized gauge choices and calibrator anchors
beyond what schema v1 already stored.

### 2. CASA full-polarisation 3C391 reference — CASA oracle, JAX apply, and CASA-compatible JAX solves 2026-08-28

On the local `data/3c391_work_v2` MS, after the current K/B/G tables:

1. Cross-hand delay (`Kcross` / equivalent).
2. Leakage on 3C84 (`Df` or `Df+QU` according to parallactic coverage).
3. R–L phase / polarisation angle on 3C286 (`Xf`), with a polarised
   3C286 model including \(V=0\) and the casaguide \(Q/U\).
4. Apply to calibrators and target fields.
5. Export a second compact golden: all four correlations, unaveraged
   channels, 3C286 and 3C84 rows, useful parallactic coverage, and the new
   tables.

CASA steps 1–5 are done. The polarised 3C286 `setjy` is a **constant
IQUV point source** across the 126 MHz band. It replaces the
channel-dependent / resolved Perley–Butler intensity model. That follows
the historical casaguide and is adequate for this first D/X reference.
It must not become the production calibrator model for spectral
discovery.

JAX then:

1. ~~Import and apply CASA D, X, and cross-hand delay through named
   operators.~~ Done 2026-08-28. Exact 2×2 \(D^{-1} C D^{-H}\) creates
   \(\sim 1.3\%\) false Stokes \(V\) on 3C286
   (\(v \approx -1.28\times 10^{-2}\) versus CASA \(-2.13\times 10^{-4}\))
   and is **not** the CASA apply. CASA `CORRECTED` RR/LL follow the
   diagonal chain. The CASA-oracle operator (`casa_parallel_preserving`)
   takes parallel hands from G/K/B+Kcross+X+P and cross-hands from the
   full 2×2. The golden gates RR/LL, channel-wise \(I,Q,U,V\), and
   \(V/I\), and JAX flags unsolved Df/Xf channels from the table domain
   rather than receiving `post_apply_flag` as input.
2. ~~Solve the same terms in JAX, one at a time, with connected
   holdouts.~~ Done 2026-08-28 as direct linear estimators (global Kcross,
   first-order Df, shared Xf). CASA table agreement is a regression
   oracle, not frequency-holdout evidence for per-channel D/X.
3. Check closure, residual calibrator polarisation, and transfer to C1.
   Diagnostic \(I,Q,U,V\) images may start with the frozen
   `casa_parallel_preserving` chain; they stay exploratory until
   exact-Jones refinement.
4. Keep CASA and JAX solutions in the fixture.

Do not average in frequency before this calibration and its discovery
tests.

### 3. Calibrator polarisation floor

Process 3C286 and 3C84 with the same chain as the target. Distinguish
independent floors from in-sample diagnostics.

- Apparent \(v\) on 3C286 against the \(V=0\) model, with an explicit
  uncertainty on that assumption. 3C286 was not used to solve Df, but
  the same 3C286 visibilities did contribute to Kcross/Xf. Apply-back
  on those rows is not an independent circular floor. The independent
  check is a scan or connected-baseline solve/validation split.
- Residual \(q,u\) on 3C286 against the known EVPA model. The same
  split applies: apply-back is engineering; held-out 3C286 is the
  linear-polarisation / EVPA floor.
- Leakage-calibrator residual polarisation on 3C84 is an **in-sample
  diagnostic**. The same single ~5-minute scan solved Df. The golden
  CORRECTED residual linear fraction \(\approx 1.4\times 10^{-4}\) shows
  that CASA apply-back on that scan is internally consistent. It does
  **not** establish an independent leakage floor. An independent 3C84
  check needs another scan, epoch, or source.

A 3C391 sky term is accepted only if it exceeds an independent floor and
remains consistent across frequency, time, and pointings on held-out
data. This replaces the earlier claim that RR/LL scales identify sky
Stokes \(V\).

### 4. Conservative calibration frequency models

Calibrators must establish every frequency degree of freedom.

Initial hierarchy:

- non-dispersive antenna delay \(\propto\nu\);
- optional dispersive phase \(\propto 1/\nu\) later, especially at L band;
- smooth per-receptor bandpass amplitude and phase within the SPW;
- a separate cross-hand delay and R–L phase;
- smooth complex leakage spectra, with discontinuities only where hardware
  warrants them;
- time-variable \(G\) with scan knots or interpolation (3C391 already
  favours this over one solution per field);
- direction-dependent beam and squint only after the on-axis model passes.

Do not fit an unconstrained value for every antenna, channel, time, and
matrix element.

Channel frequencies, widths, flags, and averaging history stay explicit.

### 5. Polarimetric BagOfWinds inventory

When the disk is mounted, extend the SRDP inventory to search table names
and CASA `VisCal` for:

- `Df` and other leakage solutions;
- `Xf` / R–L phase;
- cross-hand delay;
- polarisation calibrator models;
- full-correlation raw calibrator scans;
- parallactic-angle coverage;
- exact frequency coverage and averaging;
- pipeline and calibrator-model versions.

The first polarimetric *subset* should require both a known linear-angle
calibrator (3C286 or 3C138) and a low-polarisation leakage calibrator (3C84
or OQ208). Expect the present 29 C-band tarballs to fail that cut. They
remain the I-only stability sample. Wider L/S data are for later RM
leverage, not the first Jones milestone. K/Ka weather behaviour is useful
later, not first.

Do not download a new polarimetric corpus until the inventory says the
current tars cannot supply D/X oracles. The 3C391 D/X golden already
exists; this inventory is for later multi-dataset comparison, not to
unblock first images.

### 6. Calibration uncertainty in sky evidence

Point estimates are not enough for spectral or polarisation discovery.
After the apply/solve gates pass, the calibration product needs a compact
nuisance representation for:

- common bandpass spectral errors;
- differential R/L bandpass errors;
- R–L phase and delay;
- leakage spectra;
- flux-model uncertainty;
- time interpolation;
- later, beam and pointing.

Sky evidence should profile or marginalise over calibrator-informed modes.
A new \(q/u/v\), RM, curvature, or component is accepted only if held-out
evidence survives those alternatives.

Calibration-model selection and sky-model validation use separate
partitions: calibrator scans, frequency blocks, antennas or baselines,
parallactic angle, time, and mosaic pointings.

### 7. Target sky ladder (after frozen calibration)

Stokes-I spectral discovery does not block initial polarisation imaging.
Reuse the accepted Stokes-I morphology at a reference frequency and defer
new spectral freedom.

“Constant polarisation” currently means two different things. A global
\(q,u,v\) fit is a useful null model. It is **not** a spatial
polarisation image. It forces

\[
Q_k=qI_k,\qquad U_k=uI_k,\qquad V_k=vI_k
\]

in every pixel. It measures net, visibility-weighted fractional
polarisation and can miss polarised regions that cancel globally.

Near-term ladder:

1. Apply the frozen CASA-compatible calibration and make diagnostic
   \(I,Q,U,V\) images (`casa_parallel_preserving`). Exploratory while
   exact 2×2 \(D\) still produces false \(V\).
2. Fit global constant \(q,u,v\) against the frozen complex Stokes-I
   model \(M_I\), not observed \(\mathrm{Re}(I)\). Report null versus
   polarised loss and repeat the fit on deterministic baseline, time,
   and channel partitions. No RM.
3. Freeze the accepted Stokes-I topology. Test polarisation activation on
   I-only coarse regions, not on \(Q/U\) peaks. Candidate blocks for this
   step: unpolarised; one global \(q+iu\); joint regional \(q_r+iu_r\).
   Keep \(v=0\) in the fitted sky and keep reporting dirty \(V\).
4. Score those models by leave-one-pointing-out across all seven
   pointings plus held-out baselines, times, and channels. Add
   primary-beam-radius and parallactic-angle consistency checks. Do not
   reuse C7 as a sealed set for choosing the regions. Polarisation-driven
   pixel splits and spatial \(v\) wait.
5. Only later allow polarisation-driven spatial splits or independent
   per-pixel polarisation.
6. Introduce RM and other frequency dependence after this
   constant-frequency spatial baseline. One global RM is the first
   \(Q/U\) frequency model, not a substitute for spatial structure.

Direction:

\[
\text{global detection}
\to
\text{spatial polarisation activation}
\to
\text{self-cal}
\to
\text{frequency/RM discovery}.
\]

Activate a more complex level only when held-out data support it,
including held-out **frequency channels** once a frequency model exists.
A frequency-dependent bandpass or leakage residual must not be allowed
to masquerade as a sky spectrum.

Anchored self-cal starts only after the frozen sky represents the
important spatial polarised structure. A global \(q,u,v\) model alone is
probably inadequate for an extended remnant.

## Future-state target: calibrator-anchored polarisation self-calibration

The target state is not a post-hoc GP smoother over gains already solved on
the target. It is a self-calibration model whose zero point is the external
calibrator solution and whose permitted departures are controlled by a
Gaussian-process prior.

Write the direction-independent Jones chain as

\[
J_a(t,\nu)=\Delta G_a(t)\,J^{\mathrm{cal}}_a(t,\nu),
\]

where \(J^{\mathrm{cal}}\) is the K/B/G/Kcross/D/X/P transfer established by
the calibrators. A residual time gain belongs at the **gain** position in
that chain (\(\Delta G\,J^{\mathrm{cal}}\)), not as a generic right factor
\(J^{\mathrm{cal}}\Delta J\). Once differential gains no longer commute
with leakage, left-versus-right multiplication is a different measurement
equation. For a diagonal receptor gain,

\[
g_{a,r}(t)=g^{\mathrm{cal}}_{a,r}(t)
\exp\!\left[\delta A_{a,r}(t)+i\,\delta\phi_{a,r}(t)\right].
\]

The residual log-amplitude \(\delta A\) and phase \(\delta\phi\) have zero
mean at the transferred calibrator solution. Calibrator scans anchor them
near zero with uncertainties set by the calibrator solve. Additional latent
points at target times may move when the target likelihood supports a move.
The GP covariance controls how far that change propagates in time and how
quickly the solution returns towards the calibrator transfer.

A convenient implementation uses whitened latent variables,

\[
\boldsymbol\delta=L\mathbf z,\qquad \mathbf z\sim\mathcal N(0,I).
\]

Choose **one** formulation for the calibrator anchors:

- condition the covariance on the anchors, so \(LL^{\mathsf T}\) already
  encodes that constraint and the objective is the target visibility
  likelihood plus \(\|\mathbf z\|^2\); or
- keep an unconditional GP prior and add an explicit calibrator-anchor
  likelihood.

Do not do both. Conditioning the covariance and also adding the
calibrator-anchor likelihood counts the anchors twice.

Adding a target knot or enabling another Jones mode is an increase in model
freedom and must pass the same validation rule as adding a sky component.

[`circular_gp_gain_solution`](../src/sl1mjax/gain_time_models.py) is useful
kernel and circular-phase groundwork, but it is not this target architecture.
It computes a posterior-mean interpolation from an already solved gain table.
The future solver must place the anchored GP inside the visibility objective
and retain calibrator uncertainty. The previous 3C391 sweep also remains a
warning against assuming that smoothing helps: native fourteen-epoch linear
transfer beat every tested post-hoc GP. The anchored self-cal GP must compete
against a no-self-cal transfer and an unsmoothed candidate on independent
data.

### Jones freedom ladder

Use common and differential receptor coordinates so their sky degeneracies
are explicit:

\[
\delta A_c=\tfrac12(\delta A_R+\delta A_L),\qquad
\delta A_d=\tfrac12(\delta A_R-\delta A_L),
\]

with the same decomposition for phase.

1. Enable only common R/L time-dependent gain corrections. This is the first
   anchored-GP self-calibration mode and is closest to Stokes-I self-cal.
2. Keep differential R/L amplitude tightly anchored because it is degenerate
   with sky Stokes \(V\).
3. Keep differential R--L phase tightly anchored because it rotates sky
   \(Q/U\) and changes EVPA.
4. Freeze Kcross, D, and X at their external-calibrator values during the
   first target self-calibration experiments.
5. Admit time variation in differential gains, D, or X only after independent
   calibrator data establishes the required scale and validation data improves.

Do not put an unconstrained GP on the four complex entries of a Jones matrix.
Keep physical coordinates: log amplitude, circular phase, delay, and bounded
complex leakage. Preserve the reference-antenna, flux, \(V=0\), and EVPA
anchors explicitly.

### Major-cycle solver

The combined sky/calibration problem is non-convex even when each conditional
block is simple. These cycles start only after the frozen sky represents
the important spatial polarised structure (delivery B), not after a global
\(q,u,v\) detection fit. Use observable major cycles:

1. initialize Kcross, first-order Df, and Xf with the direct calibrator
   estimators and record that they are linearised/CASA-compatible estimates;
2. hold Jones fixed and solve or refine the sky;
3. freeze the sky and optimize the anchored GP latent variables through an
   exact \(2\times2\) visibility objective, with residual time gains as
   \(\Delta G\,J^{\mathrm{cal}}\);
4. reset sky-solver momentum and curvature estimates after a Jones update;
5. score both blocks on held baselines, time intervals, channels, and
   pointings;
6. stop or reject the calibration update if it improves training data without
   improving the pre-declared validation evidence.

The exact Jones update needs a JAX-native forward model and derivatives. The
optimizer may be Gauss--Newton, Levenberg--Marquardt, a StEFCal-style block
update, or Optax. Use of Optax is not itself the design goal; preserving the
measurement equation, gauges, anchors, and validation partitions is.

### Delivery order

**A. Diagnostic images, then global detection.** Apply the frozen external
K/B/G/Kcross/D/X/P chain with `casa_parallel_preserving` and make
\(I,Q,U,V\) images. Fit a constant global \(q,u,v\) model at \(\nu_0\) as
a detection and calibration check. Do not treat that fit as a spatial
polarisation image. Do not fit RM or another polarisation spectrum at this
milestone. Inspect channel residuals, but do not yet give calibration or
sky an independent value in every channel. Record the independent 3C286
floor; treat the 3C84 residual as in-sample only. Results stay exploratory
until exact-Jones refinement removes the false \(V\).

**B. Spatial polarisation activation.** Freeze the accepted Stokes-I
topology. Test polarisation on components or coarse regions: unpolarised;
joint \(q,u\); \(v\); full \(q,u,v\), with \(q^2+u^2+v^2\le 1\). Activate
\(q,u\) jointly; test \(v\) separately. Polarisation-driven spatial splits
and independent per-pixel polarisation come only after this regional
baseline.

**C. Anchored diagonal self-calibration.** Only after the frozen sky
represents the important spatial polarised structure, add target-time GP
points for the common R/L gain mode as \(\Delta G\,J^{\mathrm{cal}}\).
Compare against the unchanged calibrator transfer. Keep Kcross, D, X,
differential R/L gains, and frequency-dependent sky polarisation fixed.
This stage establishes whether target self-cal is useful without allowing
it to manufacture or erase \(Q,U,V\). A global \(q,u,v\) sky is probably
inadequate here for an extended remnant.

**D. Frequency dependence.** Restore a production frequency-dependent
3C286 model and introduce frequency freedom one term at a time. Keep
Kcross as a physical delay, use smooth per-receptor bandpass terms, model
D as a smooth complex spectrum, and model only residual X phase after the
delay. Select smoothness and knot counts with held-out channels. Only then
compare constant \(q,u,v\) with global RM, polarised spectral components,
or other sky frequency models. Spatial polarised structure is already in
the baseline; RM is not a substitute for it.

**E. Advanced polarisation self-calibration.** Consider differential R/L or
time-variable D/X only with multiple polarimetric calibrator scans or epochs.
BagOfWinds supplies the candidate corpus. Target data alone must not set the
absolute circular-polarisation scale, EVPA, or leakage gauge.

The future system reaches the quality standard of the Stokes-I calibration
only when the complete path has apply-back synthetic tests, independent
calibrator transfer, explicit solution coverage and uncertainty, structured
holdouts, CASA term ablations, and a sealed multi-dataset comparison. Agreement
with the 3C391 CASA tables is an engineering oracle, not that final evidence.

## Side branch: instrument weather versus site meteorology

[`analyze_calibration_weather.py`](../scripts/analyze_calibration_weather.py)
keeps its name. "Weather" is the allegory for the intervening medium and
the instrument state that move the calibration, and for the timescales of
those moves. Cadence and jump statistics are the intended product, not a
mislabel.

A later analysis should *join* solution times to the MS `WEATHER` table, and
to `SYSPOWER` / `CALDEVICE` when those subtables exist, then test residual
gain, delay, and bandpass against elevation, opacity, temperature, humidity,
wind, and switched power. Emit JSON, plots, and a short Markdown report.
3C391 already has `WEATHER`; several SRDP tables already have `G EVLASWPOW`.
That join is optional and must not delay diagnostic imaging or
exact-Jones refinement.

## Immediate next implementation

1. ~~`CalibrationSolution` schema v2: receptors ≠ correlations, 2×2 apply,
   diagonal promotion, golden K/B/G still green.~~ Done 2026-08-28.
2. ~~CASA polcal on the local 3C391 MS, with a polarised 3C286 model,
   disjoint NPZ labels, restored input flags, and tests that both cases
   load and that corrected 3C286 recovers 11.2% / 66°.~~ Done 2026-08-28.
3. ~~JAX import/apply of CASA D/X/Kcross as a single 2×2, gated only on
   aggregate RMS.~~ That over-claimed: exact \(D\) produced \(\sim 1.3\%\)
   false \(V\). Replaced by two named operators (`exact` and
   `casa_parallel_preserving`). The CASA-oracle path is gated on RR/LL,
   \(I,Q,U,V\), and \(V/I\), with solution validity (not borrowed
   `post_apply_flag` as the apply mask).
4. ~~JAX *solves* for Kcross / Df / Xf as CASA-compatible direct
   estimators.~~ Done 2026-08-28. First-order Df is labelled
   `casa_parallel_preserving`; later terms are absent from staged
   solutions; CASA table agreement is a regression oracle. Per-channel
   D/X cannot hold out frequency.
5. ~~Diagnostic \(I,Q,U,V\) images with the frozen CASA-compatible apply.~~
   First slice 2026-08-29: dirty Stokes images and
   [`diagnose_3c391_polarization.py`](../scripts/diagnose_3c391_polarization.py)
   on the polarisation golden. Exploratory only.
6. ~~Independent 3C286 polarisation floor (\(v\) and residual \(q,u\)).~~
   First slice 2026-08-29 labelled same-data 3C286 `independent`; that
   was wrong. The residual is apply-back because those rows contributed
   to Kcross/Xf. 3C84 remains `in_sample`. The validation-grade path
   solves Kcross/Xf on one 3C286 cohort and evaluates on another
   (`held_out_calibrator`).
7. ~~Global constant \(q,u,v\) as a detection and calibration check.~~
   First slice 2026-08-29 used \(\mathrm{Re}(I)\) as the intensity and
   only C1. That is invalid for an extended source. The validation-grade
   fit is \(\hat p=\sum w M_I^*(Q+iU)/\sum w|M_I|^2\) against the frozen
   mosaic prediction, with the analogous real \(v\), all seven pointings,
   a held-out pointing, and aligned dirty \(Q/U/V\). Those maps recur.
   Coarse joint \(q,u\) on I-only regions is the next imaging step, with
   \(v=0\) in the sky. Self-cal and RM wait.
8. Exact-Jones iterative refinement on a JAX-native 2×2 objective, with
   smooth time/frequency parameterizations. Keep the direct estimators as
   initializers; do not replace them with Optax. Extra D/X freedom only
   when held baselines, times, *and* channels improve. Residual time gains
   enter as \(\Delta G\,J^{\mathrm{cal}}\). Items 5–7 may run in parallel
   with 8: look at images; do not treat them as evidence-grade until
   exact Jones is in place.

The MS `WEATHER` join can proceed in parallel. It must not delay diagnostic
imaging or exact-Jones refinement.

## Progress log

### 2026-08-28 — schema v2

Done:

- `Receptor` enum and feed coherency pack/unpack in
  [`polarization.py`](../src/sl1mjax/polarization.py).
- `CalibrationSolution` stores `receptors`; `CALIBRATION_SCHEMA_VERSION = 2`.
- Diagonal G/K/B apply via \(J_p C J_q^{\mathrm{H}}\). A 2-receptor solution
  can correct a 4-product block.
- Schema 1 files promote by inferring `(R, L)` or `(X, Y)` from products.
- K/B/G 3C391 golden still applies.
- Instrument-weather allegory recorded in the cadence script docstring and
  this plan (not a rename).

Not done:

- Site-meteorology join.

### 2026-08-28 — first polcal is not an oracle

CASA 6.7.6 produced G84 / Kcross / Df / Xf tables, but review found two
blocking defects:

1. 3C286 `MODEL_DATA` had **zero RL/LR**. Intensity `setjy` with
   `3C286_C.im` was never followed by the casaguide polarised
   `setjy` (11.2%, \(Q=P\cos 66^\circ\), \(U=P\sin 66^\circ\), \(V=0\)).
   Kcross and Xf were therefore unanchored. Df on 3C84 may still be
   informative.
2. The exporter named the 3C84 visibility case `leakage`, colliding with
   the Df table prefix. `leakage_antenna1` / `leakage_flag` were table
   arrays of length 26, so the 3C84 case could not load.

Also: the script did not restore `sl1mjax_calibration_input` before
solving, so a second run used post-apply flags. Tests checked metadata
and the last axis of `*_data`, not MODEL EVPA or loadable coordinates.

Fixes in the scripts (same day): polarised 3C286 `setjy` from the first
`setjy` Stokes I; restore/delete flag versions; rename the 3C84 case to
`leakage_calibrator` and the Df table prefix to `dterms`; tests that load
both cases and require nonzero 3C286 RL/LR. `apply_calibration` wording
no longer claims a full 2×2 production apply path. Regenerated the same
day; see the next section.

### 2026-08-28 — regenerated polarisation oracle

CASA 6.7.6 was rerun after the review fixes. Input flags
`sl1mjax_calibration_input` are restored before `setjy`; an existing
`sl1mjax_post_polcal` version is deleted before re-saving.

3C286 `MODEL_DATA` now has the casaguide polarised model from the first
Perley-Butler Stokes I (\(I\approx 7.669\) Jy, \(P=0.112 I\),
\(Q=P\cos 66^\circ\), \(U=P\sin 66^\circ\), \(V=0\)). That 66° is
\(\mathrm{atan2}(U,Q)\), not IAU EVPA \(\chi=\tfrac12\mathrm{atan2}(U,Q)\)
(\(\approx 33^\circ\)).

Exported fixture
`tests/fixtures/3c391_polarization_golden.{npz,json}`:

| Case | Shape | Notes |
|---|---|---|
| `flux_angle` | `(2853, 64, 4)` | 3C286; MODEL 11.2% at 66°; CORRECTED \(\approx 11.10\%\) at \(66.04^\circ\) |
| `leakage_calibrator` | `(2205, 64, 4)` | 3C84; loads; CORRECTED residual linear fraction \(\approx 1.4\times 10^{-4}\) (same scan used to solve Df) |
| `dterms` | 26 rows, 54 ch | Df; median \(\lvert D\rvert\approx 0.072\) |
| `kcross` | 26 rows | FPARAM median \(\approx 3.57\) |
| `angle` | 26 rows, 54 ch | Xf; median phase \(\approx +56^\circ\); unflagged |

The 3C84 residual is a same-scan floor, not independent leakage
validation.

NRAO sequences polcal before fluxscale; this path still uses already
fluxscaled G on 3C286 plus a dedicated G84 on 3C84.

The second `setjy` is a constant IQUV point source. It does not
invalidate this narrowband CASA oracle. A later production 3C286 model
must restore channel-dependent Stokes I (Perley–Butler or equivalent)
and a polarised \(Q/U\) spectrum, not freeze one IQUV across the band.

### 2026-08-28 — JAX import/apply of Kcross, Df, Xf

`import_casa_polarization_solution` reads the K/B/G golden plus the
polarisation golden. Apply order, matching CASA `applycal(parang=True)`
on this dataset:

\[
J = J_{GKB}\,J_{\mathrm{Kcross}}\,J_D\,J_X\,J_P
\]

Two D operators:

- `exact`: invert the full 2×2 product. Required for synthetic round-trips
  and future JAX solves. It does **not** reproduce CASA RR/LL or calibrator
  \(V/I\).
- `casa_parallel_preserving`: parallel hands from the diagonal chain
  (G/K/B+Kcross+X+P, no D); cross-hands from the full 2×2 including D.
  This is the CASA-oracle apply. Import sets this.

Conventions:

- Kcross FPARAM is negated like parallel-hand K (`\(\tau_{\mathrm{s}}=-\mathrm{FPARAM}\times 10^{-9}\)`) and uses \(\nu-\nu_{\mathrm{ref}}\). The delay lives on receptor R; L is zero.
- Df is CASA `[[1, D_R], [D_L, 1]]`. Flagged antennas store \(|D|\sim 1\) junk. Those rows are **invalid**, not \(D=0\).
- Xf CPARAM is unit-amplitude; Jones is `diag(CPARAM, 1)`. Unsolved channels (outside the 54-channel `5~58` table) are invalid; nearest-edge D/X is not applied.
- Circular P is `diag(e^{-iχ}, e^{+iχ})` with \(\chi\) from WGS84 **geodetic** latitude, antenna ECEF, time, and phase centre.
- CASA `INTERVAL<=0` (G84) is unbounded.
- `propagate_weights=True` is refused when a leakage term is present: diagonal \(J_{ii}J_{jj}^*\) is not a D-term covariance.
- Exact \(D\) requires a complete unflagged coherency; a bad hand flags the whole sample. The CASA hybrid keeps product-local RR/LL flags. `corrupt_model` refuses `casa_parallel_preserving`.
- Kcross, Df, Xf, and circular P require receptors `(R, L)`.
- Written solutions are schema v3. Schema 1–2 readers reject v3 rather than applying diagonal G/K/B and ignoring D/X.

The CASA-oracle golden now checks RR/LL, \(I,Q,U,V\), and \(V/I\), and
JAX flags unsolved Df/Xf channels from the table. Exact 2×2 remains a
separate operator and still produces the false \(V\) if used as CASA
apply. JAX solves for these terms are CASA-compatible direct estimators
(global Kcross, first-order Df labelled `casa_parallel_preserving`, shared
Xf). They are initializers and CASA regression oracles, not a
self-calibration engine. Per-channel D/X has no frequency holdout until a
smooth spectral parameterization exists. Exact 2×2 iterative refinement
is next; do not replace the direct rung with Optax.

### 2026-08-28 — local 3C391 MS and first polcal

The 3C391 Measurement Set is in `data/` (`data/3c391_work_v2/` and
`data/3c391_ctm_mosaic_10s_spw0.ms`). CASA scripts now default to those
paths.

CASA 6.7.6 ran
[`create_3c391_polarization_reference.py`](../scripts/create_3c391_polarization_reference.py)
on `data/3c391_work_v2`. Tables in `data/3c391/reference-pol`:

| Table | VisCal | Notes |
|---|---|---|
| `3c391.G84` | G Jones | 26 ants; ~15% flagged (dead antennas) |
| `3c391.Kcross` | Kcross Jones | R–L delay; FPARAM median ~2.75 (CASA delay units) |
| `3c391.Df0` | Df Jones | 64 channels; median \|D\| ~0.073 |
| `3c391.Xf0` | Xf Jones | 64 channels; median phase ~−25° |

`applycal(parang=True)` wrote `CORRECTED_DATA` and flag version
`sl1mjax_post_polcal`. The first CASA process then died dumping
`result.json` (numpy arrays). Tables were already good; summary JSON was
rewritten. Plotms notices appeared because CASA `-c` did not forward
`--skip-plots`; the script now skips plots unless
`SL1MJAX_3C391_SKIP_PLOTS=0`.

The 3C391 Measurement Set is in `data/` (`data/3c391_work_v2/` and
`data/3c391_ctm_mosaic_10s_spw0.ms`). CASA scripts now default to those
paths. Polarisation solves write to `data/3c391/reference-pol`.

### Surprises

- **Naming.** "Weather" in `analyze_calibration_weather.py` was never site
  meteorology. Cadence/jumps stay; a `WEATHER` table join is extra, not a
  correction of the script's purpose.
- **Stokes I is 1×1, not 2×2.** Time-half selfcal and other Stokes-I
  blocks still call `identity_solution` with `Correlation.I`. That is a
  scalar Jones per antenna (`Receptor.I`), not a feed. Q/U/V as
  correlation products still cannot be Jones-packed.
- **Apply and solve diverged.** `apply_calibration` now composes G/K/B with
  optional Kcross, Df, Xf, and circular P. The Optax G/K/B solvers still
  treat the last visibility axis as one parallel hand per receptor and
  reject four-product blocks. D/X *solves* are NumPy direct estimators
  (CASA-compatible rung), not JAX-differentiable Optax. Exact 2×2
  refinement is still ahead.
- **Flagged Df is junk, not identity.** CASA leaves \(|D|\approx 1\) in
  flagged leakage rows. Applying those values as Jones destroys RR/LL.
  Invalid Df/Xf/Kcross now flag the visibility. \(D=0\) is not a stand-in
  for a missing solution.
- **Full 2×2 \(D\) vs CASA RR/LL.** After G/K/B, CASA `CORRECTED` RR/LL
  already match at \(7\times 10^{-4}\). Inserting Df as a full 2×2 moves
  parallel hands by \(\sim 2\%\) and creates \(\sim 1.3\%\) false Stokes
  \(V\). CASA import uses `casa_parallel_preserving`; `exact` remains for
  physics and future solves.
- **Flagged hands mix under exact \(D\).** A full 2×2 uses every product.
  Exact apply flags the whole sample if any hand is missing or flagged.
  The CASA hybrid keeps product-local flags on RR/LL and still requires a
  complete coherency for RL/LR. `corrupt_model` refuses the hybrid: it is
  not an invertible Jones chain.
- **Casaguide 66° is not IAU EVPA.** The NRAO 3C391 recipe sets
  \(Q=P\cos 66^\circ\), \(U=P\sin 66^\circ\). IAU \(\chi\) is half that
  argument (\(\approx 33^\circ\)). Follow the casaguide formula; do not
  substitute 33° into `setjy`.
- **Constant IQUV is not the production 3C286 spectrum.** The polarised
  `setjy` overwrites `3C286_C.im` with one I, Q, U, V for every channel.
  Fine for this 126 MHz D/X oracle; not the calibrator model for
  spectral-index or RM discovery.
- **NPZ key collision.** Naming the 3C84 visibility case `leakage` while
  the Df table used the same prefix overwrote coordinates and flags. The
  case is now `leakage_calibrator`; the table prefix is `dterms`.
- **Flag restore.** Solving without restoring `sl1mjax_calibration_input`
  made the second run use post-apply flags. The creation script now
  restores that version and replaces `sl1mjax_post_polcal`.
- **Two packings.** Feed coherency `[[RR, RL], [LR, LL]]` is not the sky
  Stokes-V packing \(V=(RR-LL)/2\) used by circular contrast. Keep them
  separate.
- **3C84 is not in Perley-Butler 2017.** `setjy` does not recognize
  `J0319+4130`. Leakage uses a manual unpolarized model; G84 absorbs I.
- **3C84 has no G in reference-v2.** Polarisation leakage needs a G on
  J0319+4130 first; K/B/G was never solved on that field.
- **3C84 is one scan.** Leakage is a single-interval Df, not a
  parallactic-angle curve.
- **3C286 is not a full-track polcal.** Field 0 only in scans 1–3 and 37.
  Xf is one interval, not an EVPA-vs-parallactic-angle curve.
- **CASA 6.7 `polcal` has no `parang`.** `gaincal` and `applycal` still do.
  Polarisation Jones application on the sky uses `applycal(parang=True)`.
- **SRDP still has no D/X.** Confirmed on BagOfWinds 2026-08-28; 3C391
  polcal has to be run, not imported from the C-band SRDP tarballs.
- **Balanced RR/LL split ≡ global sky \(v\).** Unchanged; the 2×2 path does
  not remove that degeneracy. 3C286 \(V=0\) remains the identifiability
  anchor.
- **Priors folded at the antenna.** Opacity/gain-curve/requantizer now
  multiply the antenna Jones before the 2×2 product, which is equivalent
  to the old per-receptor `J_p J_q^*` prior for diagonal terms and is what
  cross-hands need.
- **Golden fixture schema.** The CASA visibility golden
  `tests/fixtures/3c391_calibration_golden.json` stays `schema_version` 1
  (RR/LL DATA/CORRECTED_DATA export). That is not the
  `CalibrationSolution` schema. `tests/test_casa_golden.py` is a different
  sky-operator golden; do not bump it for Jones work.
- **Direct Df is linearised, not exact Jones.** `solve_leakage` is
  first-order LS and writes `leakage_application='casa_parallel_preserving'`.
  Identity starts must not inherit `exact`. An exact 2×2 refinement is a
  later step.
- **Rank-deficient D is invalid.** Per-channel leakage validity follows
  reduced-design rank and parameter support, not “antenna appeared.”
  \(D_R[\mathrm{ref}]=0\) is valid only if the reference antenna was
  observed.
- **Per-channel D/X cannot hold out frequency.** The connected split is
  baseline-time. CASA table agreement is a regression oracle, not
  validation evidence for enabling frequency-dependent D/X. Smooth
  spectral parameterization is required before channel holdout.
- **Staged polcal RMS drops later terms.** A Kcross result does not keep
  leftover CASA D/X; train/holdout RMS are isolated to that stage.
- **Df is a point-calibrator solver.** The first-order design uses real
  Stokes I and refuses nonzero model \(Q,U,V\) or baseline-varying
  coherency. Ratio estimators use weights \(w|p|^2\).

### 2026-08-29 — imaging ladder: detection is not an image

The calibrator-anchored Jones destination did not change. The imaging
sequence did.

A global \(q,u,v\) fit is a null model
(\(Q_k=qI_k\), \(U_k=uI_k\), \(V_k=vI_k\)). It is not a spatial
polarisation image and can miss cancelling structure. The first sky model
is constant \(q,u,v\) with **no RM**. Spatial activation on the accepted
Stokes-I topology (joint \(q,u\); \(v\) separately) comes next, then
self-cal, then RM. Polarisation activation is a separate decision from
Stokes-I spatial splits.

The 3C84 CORRECTED residual is an in-sample diagnostic from the same scan
that solved Df, not an independent polarisation floor. Same-data 3C286
is apply-back, not an independent \(V=0\) / EVPA check. The independent
3C286 floor is a scan or connected-baseline solve/validation split.

Diagnostic images may start with `casa_parallel_preserving` and stay
exploratory until exact-Jones refinement. Residual time gains belong at
the gain position (\(\Delta G\,J^{\mathrm{cal}}\)). The GP prior must
either condition on calibrator anchors or include an anchor likelihood,
not both.

First implementation slice: [`polarization_diagnostics.py`](../src/sl1mjax/polarization_diagnostics.py)
forms dirty \(I,Q,U,V\), an apply-back 3C286 residual, an in-sample 3C84
diagnostic, and a global \(q,u,v\) fit. The exploratory runner is
[`diagnose_3c391_polarization.py`](../scripts/diagnose_3c391_polarization.py).
The validation-grade runner is
[`validate_3c391_global_polarization.py`](../scripts/validate_3c391_global_polarization.py):
complex \(M_I\) regression, 3C286 holdout, all seven pointings, and
common-sky dirty \(Q/U/V\). Coarse joint \(Q/U\) activation is the
current imaging step; \(v\) stays off in the sky.

### 2026-08-29 — validation-grade global test

The first C1 global numbers used \(\mathrm{Re}(I_{\mathrm{obs}})\) and
labelled 3C286 independent. Both are withdrawn.

- The estimator is now \(\hat p=\sum w M_I^*(Q+iU)/\sum w|M_I|^2\),
  with a real coefficient for \(v\), plus null-versus-polarised loss.
  Observed \(\mathrm{Re}(I)\) is refused.
- 3C286 apply-back stays labelled `apply_back`. Kcross/Xf are solved on
  one scan/time cohort and evaluated on the other
  (`held_out_calibrator`).
- All seven pointings, a held-out C7, deterministic partitions, and
  aligned dirty \(Q/U/V\) are in
  [`outputs/3c391_global_polarization_validation/report.json`](../outputs/3c391_global_polarization_validation/report.json).
  Dirty \(Q/U\) recur (C7 vs C1–C6 cosine \(0.86/0.88\)). A global
  \(p_L=0.40\%\) is only modestly above the \(0.25\%\) held-out 3C286
  floor; the morphology is the stronger evidence. C7 is no longer sealed
  for choosing regions.

### 2026-08-29 — coarse joint \(Q/U\), \(v=0\)

Advance to a spatial-activation *test*, not an astrophysical
polarisation image.

- Freeze calibration, Stokes I, frequency, and averaging.
- Declare a small I-only region set
  (`central_inner` / `central_outer` / `widefield` from I-weighted
  median radius). Do not cut on \(Q/U\) peaks.
- Compare unpolarised, one global \(q+iu\), and joint regional
  \(q_r+iu_r\). The fitted sky has \(v=0\); keep reporting dirty \(V\).
- Score by leave-one-pointing-out across all seven pointings, plus
  baseline/time/channel halves, beam-radius and parallactic cohorts.
- Runner:
  [`validate_3c391_coarse_polarization.py`](../scripts/validate_3c391_coarse_polarization.py).
- Do not start RM, frequency-dependent polarisation, self-cal, or
  polarisation-driven pixel splits.
- A follow-up I-only partition uses the frozen 11,536-leaf mosaic
  consensus grouped onto 64″ ancestor cells (not one \(q,u\) per leaf),
  omitting the wide-field dictionary because the scalar Airy beam is
  least trusted off-axis.
