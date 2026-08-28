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
  the apparent polarisation recovered on nominally unpolarised or known-EVPA
  calibrator scans, on held-out frequency channels as well as baselines,
  times, and pointings.
- First target sky after that floor: constant \(v\), complex \(q+iu\) with at
  most one RM, and no independent per-pixel spectral indices until validation
  demands them.
- Inventory BagOfWinds for polarimetric products, but do not block 3C391
  polcal on finding pipeline D/X tables. The current C-band SRDP cal tarballs
  are diagonal.

Deferred until the on-axis 2×2 path passes:

- Direction-dependent beam Mueller terms and squint-as-leakage.
- Multi-component synchrotron plus free-free polarised mixtures as a
  production sky.
- RM-synthesis or spatially varying RM. The 4.536–4.662 GHz span is about
  2.7% in frequency; Faraday resolution is set by the \(\lambda^2\) span
  ([Brentjens & de Bruyn 2005](https://arxiv.org/abs/astro-ph/0507349)).
- Requiring a second VLA band or configuration before the 3C391 D/X golden
  exists.
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

[`CalibrationSolution`](../src/sl1mjax/calibration.py) is schema **v2**.
Gains, delays, and bandpass are stored per feed receptor (`R`,`L` or
`X`,`Y`), not per packed correlation product. `receptor_count` is
`len(receptors)`. `(RR, RL, LR, LL)` is two receptors, not four.
Application packs products into a 2×2 coherency, forms
\(C^{\mathrm{obs}}=J_p C^{\mathrm{sky}} J_q^{\mathrm{H}}\) (or the inverse
to correct), and unpacks. Diagonal G/K/B promote to diagonal 2×2 matrices.
Schema 1 files are read and promoted by inferring receptors from the
products.

Leakage \(D\) is not stored on the solution yet. ``apply_calibration``
still builds *diagonal* 2×2 matrices from G/K/B.
``apply_jones_to_coherency`` accepts full 2×2 Jones in unit tests, including
a non-diagonal \(D\) mixing Stokes I into RL/LR. That is not yet the
production apply path. Diagonal solvers still require one parallel-hand
product per receptor; 2×2 RL/LR *solving* is later.

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
- schema v2 write; schema 1 read with promotion.

Not stored yet, apply-ready:

- R–L delay and phase;
- leakage \(D\);
- feed/parallactic rotation;
- direction-dependent beam.

Gates:

- [x] synthetic identity and diagonal Jones recover packed parallel hands
      and cross-hands;
- [x] a non-diagonal \(D\) mixes a Stokes-I sky into RL/LR at the predicted
      level;
- [x] the committed RR/LL 3C391 golden still applies after promotion.

Still later in this item: serialized gauge choices and calibrator anchors
beyond what schema v1 already stored.

### 2. CASA full-polarisation 3C391 reference — in progress 2026-08-28

On the local `data/3c391_work_v2` MS, after the current K/B/G tables:

1. Cross-hand delay (`Kcross` / equivalent).
2. Leakage on 3C84 (`Df` or `Df+QU` according to parallactic coverage).
3. R–L phase / polarisation angle on 3C286 (`Xf`), with the Perley–Butler
   model including \(V=0\) and the known EVPA.
4. Apply to calibrators and target fields.
5. Export a second compact golden: all four correlations, unaveraged
   channels, 3C286 and 3C84 rows, useful parallactic coverage, and the new
   tables.

JAX then:

1. Import and apply CASA D, X, and cross-hand delay through the matrix
   path.
2. Compare corrected RR, RL, LR, and LL to CASA `CORRECTED_DATA` with a
   bounded, explained RMS (the K/B/G analogue of \(6.5\times 10^{-4}\)).
3. Solve the same terms in JAX, one at a time, with connected holdouts.
4. Check closure, residual calibrator polarisation, and transfer to C1.
5. Keep CASA and JAX solutions in the fixture.

Do not average in frequency before this calibration and its discovery
tests.

### 3. Calibrator polarisation floor

Process 3C286 and 3C84 with the same chain as the target.

- Apparent \(v\) on 3C286 against the \(V=0\) model, with an explicit
  uncertainty on that assumption.
- Residual \(q,u\) on 3C286 against the known EVPA model.
- Leakage-calibrator residual polarisation on 3C84.

A 3C391 sky term is accepted only if it exceeds that floor and remains
consistent across frequency, time, and pointings on held-out data. This
replaces the earlier claim that RR/LL scales identify sky Stokes \(V\).

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

Do not download a new polarimetric corpus until the 3C391 D/X golden exists
and the inventory says the current tars cannot supply D/X oracles.

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

### 7. Target sky ladder (after 1–3)

1. Stokes-I spectral model (global, then component-group). This band may
   not separate free-free; say so from held-out channels.
2. Constant global \(q,u,v\) at a reference frequency.
3. One global RM for \(Q/U\).
4. Separate unpolarised and polarised emission components if (2)–(3) leave
   structured residuals.
5. Component-level spectral and polarisation parameters.
6. Spatially varying polarisation or spectral factors last.

Activate a more complex level only when held-out data support it, including
held-out **frequency channels**. A frequency-dependent bandpass or leakage
residual must not be allowed to masquerade as a sky spectrum.

The first full-polarisation *target* model stays deliberately simple:
constant \(v\), complex \(q+iu\) with at most one RM, shared \(I(\nu)\) only
if the I-spectrum test asks for it.

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
That join is optional and must not delay Jones schema work or the 3C391
D/X golden.

## Immediate next implementation

1. ~~`CalibrationSolution` schema v2: receptors ≠ correlations, 2×2 apply,
   diagonal promotion, golden K/B/G still green.~~ Done 2026-08-28.
2. ~~CASA polcal on the local 3C391 MS, with a polarised 3C286 model,
   disjoint NPZ labels, restored input flags, and tests that both cases
   load and that corrected 3C286 recovers 11.2% / 66°.~~ Done 2026-08-28.
   The CASA tables are a D/X *oracle for import*, not yet applied in JAX.
3. JAX import/apply of CASA D/X/Kcross reproducing `CORRECTED_DATA`; then
   JAX solves.
4. 3C286/3C84 polarisation floor.
5. Only then a target \(q+iu\) / \(v\) model on C1.

The MS `WEATHER` join can proceed in parallel. It must not delay the
3C391 D/X golden.

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

- JAX import/apply of CASA `Kcross` / `Df` / `Xf` (tables and golden exist;
  `apply_calibration` is still diagonal G/K/B only).
- Stored leakage, R–L delay/phase, or 2×2 *solvers*.
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
validation. JAX still does not import or apply these tables.

NRAO sequences polcal before fluxscale; this path still uses already
fluxscaled G on 3C286 plus a dedicated G84 on 3C84.

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
- **Apply and solve diverged.** `apply_calibration` still builds only
  diagonal 2×2 matrices from G/K/B. Full 2×2 Jones is available through
  `apply_jones_to_coherency`, not the production apply path. The Optax
  G/K/B solvers still treat the last visibility axis as one parallel hand
  per receptor and reject four-product blocks. That is intentional until
  D/X import and solves exist.
- **Casaguide 66° is not IAU EVPA.** The NRAO 3C391 recipe sets
  \(Q=P\cos 66^\circ\), \(U=P\sin 66^\circ\). IAU \(\chi\) is half that
  argument (\(\approx 33^\circ\)). Follow the casaguide formula; do not
  substitute 33° into `setjy`.
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
