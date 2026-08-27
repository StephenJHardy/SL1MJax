# 3C391 corner-pixel flux

## Problem

On 3C391 C1, positive regular-grid and hierarchical reconstructions put
bright flux on the image rim, especially two opposite corners. Those pixels
are almost certainly empty: CASA CLEAN leaves them at zero, and widening the
field moves the spike to the *new* border rather than revealing a compact
source. We do not want the imager to attribute sky flux there, and we do not
want Haar screening to split those leaves.

This is not an indexing bug. Level-0 quadtree centres match `RegularGrid`
exactly, including FITS `CDELT1 < 0` (array column increases westward).

The remnant itself is real and interior. On the 128², 4″ grid it peaks at
array `(iy, ix) = (73, 94)`, about 2.1′ west-north of the phase centre
(position angle ≈ −73°).

## Revised diagnosis

The primary cause is the **scale of the structured UV holdout**, not a source
on the rim and not an unavoidable mode of the complete visibility data.

`uv_cell_split` divides the full UV extent into `cells_per_axis` bins in each
direction, then removes complete occupied cells. The previous default of 8
creates holes 2,700--3,700 wavelengths wide on the 1,024-row diagnostic
subset. Their Fourier-dual scales are only 55--76 arcsec. The reconstructed
field is 512 arcsec wide, so training asks the image to extrapolate across
large coherent Fourier holes that support many positive image-domain
interpolants. Some of the cheapest transient interpolants put flux on the
bounding box.

The appropriate cell scale depends on the image field of view. To keep one UV
cell narrower than roughly $1/\mathrm{FOV}$, this 128², 4 arcsec image needs
about 75 cells across `u` and 54 across `v`. A square 64×64 split is close;
8×8 is an order of magnitude too coarse.

The location of the hot edge is controlled by the held cells. With the same
data, model, optimizer, and 8×8 cell size, changing the split seed from 17 to
0 moves the peak from array `(127, 127)` to `(127, 0)`. Fitting all active rows
or using random-row validation also removes the corner peak. The natural VLA
anisotropy may shape the available null modes, but it does not uniquely select
the NW--SE pair described in the earlier diagnosis.

A second problem makes the artifact much worse: the legacy softplus/Adam
solver is strongly path-dependent and is not close to the physical
nonnegative-L1 optimum when early stopping returns. At `10⁻³ Jy` per pixel,
the initial 128² image contains 16.384 Jy. The corresponding 128², 192², and
256² final totals closely track their initialization masses. Softplus also
multiplies the physical-flux gradient by approximately the flux near zero, so
large KKT violations can look small in raw-parameter space.

Haar is therefore screening a bad training reconstruction. It correctly sees
the largest residual contrasts in the fitted model, but those contrasts were
created by validation geometry and optimizer dynamics rather than supported
sky structure.

## Evidence

All regular-grid numbers below are `square-paraxial`, 300 steps, sparsity
`10⁻⁴`, UV-cell holdout, CASA-corrected fixture block, no primary beam
unless stated. Corner labels are **array** coordinates with `origin=lower`:
array NE is celestial NW.

### Discriminating UV-split experiments

These controls use the same 1,024 evenly sampled rows, 128² square-paraxial
model, `10⁻⁴` L1, `10⁻³ Jy/pixel` initialization, and 1,000-step ceiling.
Only validation geometry changes.

| validation | holdout loss | image peak | max corner | total flux |
|---|---:|---|---:|---:|
| UV cells 8×8, seed 17 | 0.3487 | corner `(127,127)` | 0.00936 Jy | 16.24 Jy |
| UV cells 16×16 | 0.3099 | corner `(0,127)` | 0.00887 Jy | 14.55 Jy |
| UV cells 32×32 | 0.0278 | remnant `(75,93)` | 0.00337 Jy | 13.92 Jy |
| UV cells 64×64 | 0.0141 | remnant `(75,93)` | 0.00051 Jy | 13.90 Jy |
| random rows | 0.00838 | remnant `(72,94)` | 0.00065 Jy | 13.71 Jy |
| fit all rows | — | remnant `(72,94)` | 0.00089 Jy | 13.46 Jy |

At 8×8, changing only the seed to 0 moves the peak to the opposite horizontal
boundary at `(127,0)`, with 0.0141 Jy in that corner. This is direct evidence
that the held-cell pattern selects the edge mode.

The full 20,542-row, 300-step confirmation gives the same result. Changing
only `cells_per_axis` from 8 to 64 moves the peak back to the remnant, lowers
the maximum corner from about 0.0090 to 0.00107 Jy, and lowers held-out loss
from 0.369 to 0.0167.

The number of bins is not just a tuning constant. On the 1,024-row subset,
the UV extents are about 14,941 and 10,878 wavelengths. An 8×8 split makes
cells about 3,735 by 2,720 wavelengths. A 64×64 split makes them about 467 by
340 wavelengths, close to the reciprocal 512-arcsec field scale of about 403
wavelengths.

### Initialization and stationarity controls

With the original 8×8 split, changing only initial intensity gives:

| initialization | initial total | holdout loss | image peak | max corner | final total |
|---|---:|---:|---|---:|---:|
| `10⁻³ Jy/pixel` | 16.384 Jy | 0.3487 | corner | 0.00936 Jy | 16.24 Jy |
| `10⁻⁵ Jy/pixel` | 0.164 Jy | 0.3518 | near-zero image | 0.000013 Jy | 0.166 Jy |
| `10⁻⁷ Jy/pixel` | 0.00164 Jy | 0.2123 | remnant | 0.0000022 Jy | 4.53 Jy |

The low-start image is not evidence that `10⁻⁷` is a correct fix. It is a
different under-fitted point on an initialization-dependent optimization path.
When all rows are fitted, the `10⁻⁷` start still has 9,161 pixels below
`10⁻⁷ Jy`; 3,195 of them have a negative physical gradient and should grow.

The corner-peaked 8×8 holdout fit is also not stationary. Its maximum
projected physical-flux gradient is `8.7×10⁻³`, while its L1 coefficient is
`10⁻⁴`. Every pixel remains formally positive, but the largest gradient after
the softplus chain rule is only `3.9×10⁻⁶`. Early stopping reports convergence
because held-out loss stopped improving, not because the stated training
objective reached its constrained optimum.

### Physical-flux solver implementation and controls

The inference layer now supports four fixed-topology solvers:

- `softplus_adam`, retained as the compatibility control;
- monotone restarted `fista` with backtracking in physical flux;
- `proximal_sgd`, using unbiased randomly reshuffled row batches; and
- `hybrid`, which runs proximal SGD and then restarts in FISTA.

All three physical solvers use the same positive-L1 proximal map,

$$
I^+ = \max\left(0, I - \eta \nabla f(I) - \eta\lambda\right),
$$

so they produce exact zeros and a zero pixel can grow whenever its data
gradient overcomes the L1 penalty. FISTA backtracking determines a stable
step from the objective. The stochastic gradient uses the fixed global
training-weight denominator and rescales each uniformly sampled row batch;
using a separately normalized MSE for every batch would be biased when row
weights differ.

The physical solvers select their returned iterate by the complete training
objective. Holdout values are recorded for model selection but do not silently
replace constrained convergence. A projected physical KKT residual is recorded
separately. Full training and holdout operators are compacted to rows that can
contribute, rather than predicting flagged or opposite-fold rows and masking
them afterwards.

The zero-initialization control removes the earlier path dependence. On the
1,024-row fit-all case, 500 FISTA steps from `10⁻³ Jy/pixel` and from exactly
zero give respectively:

| initialization | objective | total flux | KKT residual | peak |
|---|---:|---:|---:|---|
| `10⁻³ Jy/pixel` | 0.00275220 | 9.4809 Jy | `1.25×10⁻⁵` | remnant `(73,94)` |
| exactly zero | 0.00275219 | 9.4807 Jy | `1.42×10⁻⁵` | remnant `(73,94)` |

The remaining difference is the finite 500-step limit, not a different basin.
This is a convex fixed-topology problem and the two paths are approaching the
same sparse solution.

The full 20,542-row comparison uses the field-appropriate 64×64 UV holdout,
128² square-paraxial pixels, `λ=10⁻⁴`, and the same `10⁻³ Jy/pixel` start.
There are 11,027 active training rows and 2,557 holdout rows. Times are from
the RTX 3080 Ti after active-row compaction.

| solver | update budget | time | train objective | final holdout | KKT residual | total flux | pixels ≤`10⁻⁷` |
|---|---:|---:|---:|---:|---:|---:|---:|
| softplus Adam | 300 full | 61.8 s | 0.008239 | 0.016710 | not available | 15.007 Jy | 0 |
| FISTA | 300 full | 70.8 s | 0.003680 | 0.004361 | `1.59×10⁻⁵` | 9.432 Jy | 7,699 |
| proximal SGD | 5,000 × 1,024 rows (≈464 passes) | 39.4 s | 0.003686 | 0.004181 | `2.86×10⁻³` | 9.439 Jy | 2,318 |
| hybrid | 1,500 × 1,024 rows + 150 full (≈289 passes) | 50.0 s | 0.003681 | 0.004255 | `3.19×10⁻⁵` | 9.434 Jy | 7,224 |

All physical methods peak on the remnant at `(73,94)`. FISTA's largest corner
is 1.10 mJy and its edge-flux fraction is 0.0051, compared with four positive
Adam corners up to 1.07 mJy and an edge fraction of 0.0132. More importantly,
the physical methods no longer carry the 16.384 Jy initialization mass into a
roughly 15 Jy answer.

SGD is therefore useful rather than ruled out. It reaches a near-FISTA
objective in 56% of FISTA's wall time, but its noisy KKT residual makes it a
poor final certificate. The hybrid recovers most of the deterministic
stationarity in 71% of FISTA's wall time.

Validation prefers earlier iterates in this single fold. Minimum holdout losses
were 0.00408 for FISTA, 0.00393 for SGD, and 0.00391 for the hybrid. Those are
stopping-time selections, not the converged solution for the stated `λ`.
A stronger `λ=3×10⁻⁴` FISTA control reduced the final holdout to 0.00421 and
the edge fraction to 0.0037. This supports selecting `λ` and the stochastic
schedule through repeated validation folds, followed by a final all-data fit.

### Regular grid, same 4″ pixels, growing FOV

From `outputs/wider_fov_comparison.json`:

| size | FOV | sky peak | remnant \(I\) | array-NE / NW / SE / SW | total flux |
|---|---|---|---|---|---|
| 128 | 8.53′ | remnant 0.0098 | 0.0098 | 0.0090 / 0.0002 / 0.0002 / 0.0024 | 15.6 Jy |
| 192 | 12.8′ | **corner 0.0135** | 0.0082 | 0.0135 / 0.0002 / 0.0002 / 0.0119 | 28.8 Jy |
| 256 | 17.1′ | **corner 0.0105** | 0.0062 | 0.0105 / 0.0002 / 0.0002 / 0.0055 | 57.7 Jy |

The hot pair is always the NW–SE diagonal. Total flux grows with area. The
peak follows the border. A source sitting just outside the 8.5′ box would
have landed *inside* the 12.8′ box, not on its new corner.

VLA C-band FWHM at 4.55 GHz is ≈ 9.2′ (half-power radius ≈ 4.6′). Even the
128 corners are already outside that circle (radius 6.0′, Gaussian \(B \approx
0.30\)). The 256 corners are at 12.0′ (\(B \approx 0.008\)).

### Hierarchical splits prefer the same rim

| run | holdout | first-round splits | on boundary |
|---|---|---|---|
| 1024-row subset | 0.349 → 0.272 after 32 splits | 32 | 75% |
| same 1024-row regular 128 | — | — | reconstruction itself peaks at corner (127, 127) |
| all 20,542 rows | 0.335 → 0.136 → 0.108 | 32 then 22 | 94% of first-round parents |

The 1024-row subset is a broken reconstruction, not just a broken screen.
The full-data hierarchical render *does* peak at the remnant (root `(73, 94)`
on the 512² fine grid) and still spends its split budget on the rim.

### CASA does not put model flux there

CASA `tclean` residual on the 128 product is quiet (peak ±4 mJy/beam, RMS
1.1 mJy/beam versus dirty peak 0.23 Jy/beam). The CLEAN **model** is zero on
all edges. The CASA dirty map’s brightest corner is a *different* corner
(sky SW, ~32 mJy). We are not matching a dirty-map sidelobe; we are fitting
a positive model the CLEAN loop refused.

### L1 strength

Sparsity `10⁻⁴` (the holdout-selection default) leaves the 128 NE corner at
0.009 Jy, almost the remnant. Sparsity `10⁻³` (the full-vis regularizer
chosen earlier) drops those corners to ~0.0002 Jy. The mode is buyable
under weak L1 and not under strong L1.

This remains useful regularization evidence, but it is not the root-cause
test. A finer UV split suppresses the corner at the same `10⁻⁴` weight, while
changing initialization also changes the result sharply. The previous sweep
therefore mixed L1 strength with validation geometry and optimizer path.

### Primary beam in the forward model

Hypothesis: \(V = \mathrm{DFT}(I B)\) with L1 on true sky \(I\) makes rim
pixels expensive (\(1/B\)), so leftover power should move inward.

From `outputs/beam_fov_comparison.json`, Gaussian VLA beam, same `10⁻⁴` L1:

| run | sky peak | remnant \(I\) | max corner \(I\) | apparent (\(IB\)) peak |
|---|---|---|---|---|
| 128, no beam | remnant 0.0098 | 0.0098 | 0.0090 | remnant |
| 128, Gaussian | **NE corner 0.019** | 0.0115 | **0.019** | remnant 0.0099 |
| 256, no beam | NE corner 0.010 | 0.0062 | 0.010 | near remnant |
| 256, Gaussian | **south rim 0.017** | 0.0092 | **0.00013** | remnant 0.0080 |

On 128, \(B \approx 0.30\) is only a \(3\times\) surcharge. The data term
paid it: sky \(I\) at the corner doubled. Apparent \(IB\) still peaks on the
remnant because the taper hides the rim.

On 256 the far corners died (\(110\times\) L1). The mode did not go to the
remnant. It parked on the south boundary at \(r \approx 8.8'\), \(B \approx
0.08\) (\(I = 0.017\) Jy). Interior peak is the remnant; apparent peak is
the remnant. Sky flux is still ~60 Jy, with ~28 Jy beyond \(1.5\times\)
FWHM.

The beam forbids pixels the dish has already killed. It does not force the
large-scale residual into the main lobe. Apparent maps look cleaner because
the rim is downweighted.

#### Airy beam with the physical-flux solvers

The solver comparison was repeated with the analytic blocked-aperture VLA
Airy power beam in the forward model. These runs use the complete 20,542-row
CASA-corrected C1 block, 128² 4″ square-paraxial pixels, the 64×64 UV holdout,
`λ=10⁻⁴`, and no feed squint. The apparent image is reported at the median
channel frequency, 4.599 GHz. The intrinsic image remains the optimized sky
`I`; the apparent image is `BI` at that reference frequency.

| beam | solver | time | train data | train objective | holdout | intrinsic flux | apparent flux |
|---|---|---:|---:|---:|---:|---:|---:|
| none | softplus Adam | 62.3 s | 0.006738 | 0.008239 | 0.016710 | 15.007 Jy | 15.007 Jy |
| Airy | softplus Adam | 66.9 s | 0.004621 | 0.006156 | 0.012357 | 15.342 Jy | 12.486 Jy |
| none | FISTA | 69.9 s | 0.002737 | 0.003680 | 0.004361 | 9.432 Jy | 9.432 Jy |
| Airy | FISTA | 76.7 s | 0.002735 | 0.003825 | 0.004209 | 10.904 Jy | 9.352 Jy |
| none | proximal SGD | 50.4 s | 0.002740 | 0.003684 | 0.004193 | 9.440 Jy | 9.440 Jy |
| Airy | proximal SGD | 55.8 s | 0.002738 | 0.003829 | **0.004073** | 10.908 Jy | 9.356 Jy |
| none | hybrid | 51.8 s | 0.002738 | 0.003681 | 0.004255 | 9.434 Jy | 9.434 Jy |
| Airy | hybrid | 56.5 s | 0.002736 | 0.003826 | 0.004146 | 10.907 Jy | 9.355 Jy |

Every reconstruction peaks on the remnant at `(73,94)`. Among the physical
solvers, adding the beam lowers holdout loss by 2.6–3.5% at the same `λ` and
schedule. Their Airy-beam intrinsic images correlate with the corresponding
no-beam images at 0.995–0.996, but require about 15.6% more intrinsic flux to
produce almost the same apparent flux. This is the expected de-attenuation,
not new measured flux.

The Airy beam suppresses the apparent boundary but does not remove intrinsic
boundary freedom. For FISTA, the intrinsic edge-flux fraction rises from
0.0051 to 0.0091 and the largest intrinsic corner rises from 1.10 to 2.41 mJy.
At 4.599 GHz the beam reduces that apparent corner to 0.70 mJy and the apparent
edge fraction to 0.0044. The hybrid Airy solution has a smaller 1.49 mJy
intrinsic maximum corner and is 0.998 correlated with the Airy FISTA image.

These results support including the beam for a single-pointing fit. They do
not establish the final regularization because `λ` was held fixed. The beam
changes the spatial sensitivity and therefore changes the effective prior in
the visibility domain. `λ` and stopping time must be reselected with the beam
enabled. A joint mosaic must also use the pointing centre of each field; one
C1-centred beam is not a valid model for all seven pointings.

The complete products are under
`outputs/3c391_solver_beam_matrix/`. Each run contains intrinsic and apparent
FITS images, optimizer histories, residual visibilities, a summary, and its
log. `comparison.png` shows all eight solutions on common scales and
`comparison_summary.json` contains the cross-solver and cross-beam metrics.

#### Airy-beam λ and stopping-time selection

FISTA was then run for 300 steps on three matched 64×64 UV-cell folds for
`λ ∈ {3×10⁻⁵, 10⁻⁴, 2×10⁻⁴, 3×10⁻⁴, 5×10⁻⁴, 10⁻³}`. Holdout and KKT values
were recorded every 25 steps. Using matched folds matters because the absolute
holdout loss varies much more between UV partitions than between nearby λ
values.

| λ | best mean holdout | selected step | mean holdout at 300 | mean KKT at 300 |
|---:|---:|---:|---:|---:|
| `3×10⁻⁵` | 0.005080 | 50 | 0.005254 | `1.69×10⁻⁵` |
| `10⁻⁴` | 0.004971 | 50 | 0.005100 | `1.86×10⁻⁵` |
| `2×10⁻⁴` | 0.004919 | 50 | 0.005036 | `1.68×10⁻⁵` |
| **`3×10⁻⁴`** | **0.004910** | **50** | **0.005033** | `2.02×10⁻⁵` |
| `5×10⁻⁴` | 0.005002 | 50 | 0.005167 | `2.47×10⁻⁵` |
| `10⁻³` | 0.005666 | 50 | 0.005978 | `2.67×10⁻⁵` |

The paired difference between `2×10⁻⁴` and `3×10⁻⁴` at step 50 is only
`(0.9 ± 1.2)×10⁻⁵` (mean ± standard error), so they are not distinguishable
with three folds. `3×10⁻⁴` is retained because it is the sparser member of
that tied pair. `5×10⁻⁴` is worse on every paired fold by
`(9.2 ± 1.2)×10⁻⁵`.

All λ values reach their predictive minimum near step 50. That checkpoint is
not an optimizer solution: its mean KKT residual is about `3×10⁻⁴`. The
stationary 300-step answer raises mean holdout by only about 2.5% at the
selected λ. Predictive early stopping is therefore useful for a final fixed
topology, but it is unsafe for split validation because a warm-started child
would receive continuation work that the parent did not. Topology comparisons
use a KKT threshold; prediction can use the separately selected 50-step rule.

The sweep products and curves are under
`outputs/3c391_airy_fista_selection/`.

#### First stationary hierarchical split attempt

The first full refinement used `λ=3×10⁻⁴`, the Airy beam, seed 17, 128² 4″
root pixels, a 500-step ceiling, and KKT tolerance `3×10⁻⁵`. It capped a bulk
proposal at 256 parents and required at least 0.1% relative holdout
improvement. Merging was disabled to isolate the split decision.

The baseline stopped after 200 steps with KKT `2.57×10⁻⁵`, training data loss
0.002832, holdout loss 0.004007, and 10.388 Jy total flux. Screening selected
only `(level=0, iy=118, ix=127)` and `(0,119,127)`, both on the right boundary.
The two-parent refit changed the penalized training objective by only
`8.3×10⁻⁶` fractionally and worsened holdout by 0.190%. Backtracking to the
strongest single parent improved training by `4.3×10⁻⁵` but worsened holdout
by 0.203%. Both proposals were correctly rejected.

The screen estimates only `6.3×10⁻⁷` total available training improvement,
about 0.008% of the penalized baseline objective. This is evidence that 4″
roots are already too fine for a useful split test: their children are 2″
super-resolution elements. The boundary ranking is also consistent with a
single C1 field absorbing emission associated with neighbouring mosaic
pointings. This motivated a 64² grid of 8″ roots over the same 512″ field,
splitting supported regions to 4″. It preserves 4,096 initial fitted pixels
while making the first split scientifically meaningful.

The approved 64² 8″ experiment was then run on seed 17 with the same λ,
Airy beam, KKT threshold, and 512″ field. The baseline has 4,096 leaves and
converged after 175 steps with KKT `2.75×10⁻⁵`, training data loss 0.002842,
holdout 0.003975, and 10.389 Jy total flux. The approximate screen found 2,172
eligible parents and marked 12. Exact beam-aware rescoring retained only
`(level=0, iy=59, ix=63)`, again on the right boundary. Its global refit
improved the penalized training objective by 0.0091% but worsened holdout by
0.0515%, so it was rejected.

This run exposed a remaining unselected hyperparameter. The per-leaf topology
penalty was `10⁻⁷`, so a split had to overcome `3×10⁻⁷` before it could be
marked. The exact rescore retained only one of the 12 candidates after this
charge. Exact shortlist scores are now retained separately from the
approximate screen.

The zero-penalty diagnostic found 205 positive exact candidates and selected
61 to cover 70% of their predicted improvement. Of these, 31 touch the image
boundary and 30 are interior. The boundary candidates contribute 65.4% of the
selected score. The first ten ranked candidates are all on the boundary; the
first interior candidate is rank 11. The full 61-parent refit improved the
training objective by 0.111% but worsened holdout by 0.031%. Prefix
backtracking to 30, 15, and 7 parents also worsened holdout by 0.50%, 0.36%,
and 0.28%, respectively, so all proposals were rejected.

This did not establish that the interior 8″→4″ candidates were unsupported.
Backtracking only tested prefixes of the exact ranking, so every smaller batch
was even more boundary-dominated. The initial proposal to prohibit boundary
splits was therefore too direct: it would hide a scoring defect rather than
correct it.

The exact Haar score was still an unconstrained quadratic,

$$
S = \tfrac12 g^T G^{-1}g.
$$

For a locally constant beam response `B`, the residual gradient scales as
`g ∝ B` and the curvature as `G ∝ B²`. Their factors cancel, so
the score allows an arbitrarily large intrinsic child contrast to compensate
for a weak beam. This is a valid upper bound for an unconstrained linear fit,
but it is not the correct final score for a positive sky with finite fitted
parent flux. The 512″ field also does not attenuate its boundary by 100: over
the four C1 channels the Airy power response is about 0.29 at a corner and
0.56 at a side midpoint, versus about 0.86 at the remnant. The approximately
0.01 response belongs to the corners of the earlier 1,024″ field.

Scoring now solves the three-dimensional Haar quadratic subject to four
non-negative children whose flux sums to the parent flux. The unconstrained
score remains a cheap upper bound, but the feasible score falls with `B`
once the required contrast exceeds the available parent flux. Exact gradients
and per-parent 3×3 curvature matrices are accumulated in candidate and
visibility tiles. A regression test confirms scalar equivalence, tiling
invariance, and the expected low-beam limit. The approximate per-level screen
uses the parent nearest the pointing centre as its curvature representative;
using the first corner would give zero curvature on a field extending beyond
the configured Airy cutoff.

A 64² 16″-root run covers 17.1′ with the wide-field kernel while keeping
4,096 initial leaves. Four validated rounds accepted 15, 67, 98, and 108
splits, with no selected boundary parent. Holdout fell from 0.008780 to
0.004992, 0.003912, 0.003853, and 0.003815. A fifth-round proposal was rejected:
its 92-, 46-, 23-, and 11-parent prefixes all improved training but worsened
holdout by 1.03%, 1.02%, 0.87%, and 0.38%. Validation therefore selected 4,960
leaves: 3,823 at 16″, 1,077 at 8″, and 60 at 4″. This is 30.3% of a uniform
128² 8″ grid over the same field, and its holdout is 4.0% lower than the
narrow uniform 8″ model's 0.003975.

The selected model contains 0.451 Jy intrinsic, 0.121 Jy beam-attenuated flux
outside the old square. Regions with `B < 0.1` contain only 1.6 mJy intrinsic
flux, and all exactly zero-beam leaves remain zero. The complete five-round
run, including four rejected global refits in the stopping round, took 311 s
on the RTX 3080 Ti.

The identical procedure was repeated for UV-fold seeds 29 and 43:

| seed | accepted rounds | final leaves | baseline holdout | final holdout | relative reduction |
|---:|---:|---:|---:|---:|---:|
| 17 | 4 | 4,960 | 0.008780 | 0.003815 | 56.6% |
| 29 | 6 | 5,548 | 0.011329 | 0.006156 | 45.7% |
| 43 | 5 | 5,272 | 0.011069 | 0.004879 | 55.9% |

Every fold stopped when the next round failed held-out validation, and none
accepted a boundary split. The first 16″→8″ round is highly stable: 14 parents
occur in all three folds and 17 occur in at least two, from a union of only
19. Across all accepted rounds, 310 root parents occur in at least two folds
and 154 occur in all three. This majority set covers 85%, 67%, and 78% of the
root refinements selected by seeds 17, 29, and 43.

The 8″→4″ topology is less stable. Only 16 level-1 parents occur in at least
two folds and three occur in all three; pairwise Jaccard overlap is 0.05–0.15.
The real-data result therefore supports a majority-vote 16″→8″ topology, but
only a small consensus set of 4″ refinements. The next scientific product
should refit that consensus topology on all active visibilities rather than
choose one fold's stopping tree.

That all-data refit is now complete. Majority support selects 310 level-0 and
16 level-1 splits, producing 5,074 leaves: 3,786 at 16″, 1,224 at 8″, and 64
at 4″. Physical-flux FISTA converged in 200 steps to KKT `2.28×10⁻⁵`.
The full-data objective is 0.005828, including data loss 0.002645 and L1 cost
0.003183. Total intrinsic flux is 10.609 Jy. The fit assigns 0.401 Jy
intrinsic, 0.108 Jy beam-attenuated flux outside the old square, and no flux
where the mean beam is below 0.1. This is the first final-parameter fit on all
20,542 rows; the fold fits above are retained only for topology selection.

C1 is one pointing of a seven-pointing mosaic. Neighbouring-field leakage
belongs outside this beam. A beam-weighted L1 will pull that leakage onto
high-\(B\) pixels (pedestal / remnant bias) if we let it.

### Frozen outer-scan confirmation

The earlier UV folds selected topology and reported validation on overlapping
visibility sets, so they were useful development evidence but not a sealed
generalisation test. A nested protocol now removes complete scans before any
topology work. Three inner UV-cell folds select splits from the remaining
rows, strict-majority support fixes the topology, and a final FISTA fit sees
only the outer-training scans. The removed scans are evaluated once at the
end. The 64² 16″ base grid, Airy beam, wide-field square kernel,
\(\lambda=3\times10^{-4}\), split budgets, and stopping rules are identical
for both pointings.

| pointing | sealed scans | test samples | inner split counts | consensus splits | base test power | hierarchical test power | reduction |
|---|---|---:|---|---:|---:|---:|---:|
| C1 | 29, 79, 95 | 21,952 (20.2%) | 199, 179, 198 | 168 | 0.01829 | 0.00993 | 45.7% |
| C2 | 30, 80, 96 | 23,368 (22.0%) | 167, 181, 201 | 148 | 0.03416 | 0.02578 | 24.5% |

Both final fits satisfy the physical positive-L1 KKT threshold. C1 ends with
4,600 leaves and C2 with 4,540. The independent C2 reconstruction also agrees
with the C1 CASA multiscale image over their common sky area. The result is
weaker than C1 but still substantial, so refinement is not merely fitting one
pointing's UV sampling or one fortunate validation partition.

These tests compare hierarchical and unrefined SL1MJax models under a sealed
protocol. The existing CASA image uses all scans, so it remains a useful image
reference rather than a matched outer-test estimator. Products and complete
machine-readable metrics are under
`outputs/3c391_{,c2_}hierarchical_frozen_protocol/`.

### Seven-pointing joint mosaic proof

The first joint mosaic fit now uses all seven CASA-corrected pointing blocks.
One sky is expressed about C1, then each leaf centre is transformed into every
block's local direction-cosine frame. Fourier phase and the Airy beam are
therefore pointing-specific. Flux and its L1 penalty are shared once across
the mosaic. The loss is one weight-normalized sum over all 755,048 active
correlation samples, rather than an unweighted average of seven field losses.

The image uses a 96² 16″ root grid, so the field is 25.6 arcmin wide. The C1
consensus hierarchy is embedded in its central 64² region; added outer roots
remain coarse. This produces a 9,216-leaf baseline and a 10,194-leaf
hierarchy. Both physical-flux FISTA fits converged:

| model | steps | KKT | total flux | normalized residual power |
|---|---:|---:|---:|---:|
| 16″ base grid | 75 | `1.91e-5` | 11.301 Jy | 0.01383 |
| embedded hierarchy | 125 | `2.78e-5` | 10.763 Jy | 0.00857 |

The hierarchy reduces the combined residual power by 38.0%. Every pointing
improves separately: C1 48.5%, C2 26.1%, C3 38.6%, C4 43.8%, C5 41.0%, C6
33.6%, and C7 25.5%. The restored central image remains close to CASA
multiscale CLEAN. No fitted flux lies outside the old central 64² footprint,
on the 96² boundary, or where every pointing has beam power below 0.1. The
larger field therefore did not recreate the old corner-pixel failure.

This is a joint forward-model and flux-inference proof. Its hierarchy was
selected from C1 and then refitted to C1--C7, so it remains the control for the
joint topology run below. Products are under
`outputs/3c391_mosaic_joint_fista/`.

### Joint mosaic topology discovery

Joint residual/Haar screening and validated refits are now implemented. The
screening gradient is the exact sum of the seven pointing-specific residual
projections under one global weight normalization. The cheap first pass uses
one representative 3x3 Haar Gram matrix per level. Marked parents are then
rescored with an exact per-parent Gram accumulated over every pointing, using
each field's local direction cosines, frequency-dependent Airy power beam, and
visibility weights. The final four-child solve preserves parent flux and
enforces non-negative child flux. Thus no artificial single “mosaic beam” is
used in either ranking or validation.

The production evaluation removes complete scans independently from every
pointing before topology discovery. Three inner 64×64 UV-cell folds with seeds
17, 29, and 43 discover trees from the remaining scans. This uses the finer
validation scale that suppressed the boundary interpolation mode in the
single-pointing controls. A split enters the final topology only with
strict-majority support. The unrefined and consensus skies are then fitted on
the outer-training scans and evaluated once on the sealed scans. The run uses
a 104² root grid at 16″, up to two split levels,
physical-flux FISTA, the wide-field square kernel, the Airy beam, and
\(\lambda=3\times10^{-4}\).

The requested 20% inner holdout corresponds to 19.6%, 20.3%, and 20.6% of
active samples globally in the three folds. Individual pointing fractions
range from 17.9% to 22.0%, so no fold is dominated by a severely underfilled
validation partition.

The larger 27.73′ field was chosen from the union of all seven beam patterns.
The six outer pointing centres are 2.45′ from C1. On the earlier 96² field,
the largest power response from any pointing on the boundary was 0.00338 and
the modeled 0.1% contour extended just beyond the grid. The 104² field
contains the complete CASA-compatible Airy support and leaves about 0.7′ of
zero-response guard at the nearest boundary. This statement applies to the
configured Airy model, which is hard-truncated at its CASA maximum radius; it
does not model coupling through real far sidelobes.

The repeated-fold run was launched on Bacchus on 2026-08-24. Its products and
logs are under `outputs/3c391_mosaic_hierarchical_frozen_104/`.

The first execution completed inner fold 17 in 2,466 s. It accepted 252
splits, stopped when the next batch failed validation, and ended with 11,572
leaves and KKT residual `1.94e-5`. Fold 29 then reached its sixth round, where
an eight-parent proposal was valid but missed the 0.1% holdout threshold. The
four-parent backtracking refit exposed an optimizer robustness bug: FISTA's
majorization test used a fixed `1e-12` allowance even for float32 GPU
reductions and eventually rejected every vanishing trial step.

The majorization allowance now scales with the arithmetic dtype and objective
magnitude. A failed prefix refit is also recorded and isolated so that smaller
prefixes can still be tested. Focused CPU and CUDA tests pass, including an
injected refit failure, and the full local suite has 259 passes and one
optional real-MS skip. The failed log is preserved as
`run.failed_fista_20260824.log`; the corrected run resumed fold 17 and
restarted fold 29 on 2026-08-25.

The corrected protocol completed all folds and the sealed comparison. Folds
17, 29, and 43 accepted 252, 256, and 238 splits. Their pairwise split-set
Jaccard scores are 0.65--0.69. Strict-majority support selects 225 root splits
and 15 second-level splits, or 240 splits in total. The consensus topology has
11,536 leaves: 10,591 at 16″, 885 at 8″, and 60 at 4″.

| model | leaves | KKT | total flux | train residual power | sealed residual power |
|---|---:|---:|---:|---:|---:|
| 16″ base | 10,816 | `1.48e-5` | 11.367 Jy | 0.01352 | 0.01512 |
| mosaic consensus | 11,536 | `2.87e-5` | 10.829 Jy | 0.00844 | 0.00906 |

The hierarchy reduces normalized residual power by 37.6% on outer-training
scans and 40.1% on the sealed scans. Every pointing improves independently:
C1 44.5%, C2 50.7%, C3 32.9%, C4 41.1%, C5 47.3%, C6 31.3%, and C7 29.6%.
This is a substantially stronger topology-selection result than the original
C1-derived joint-fit control because no test scan or single pointing chose the
consensus tree.

No consensus flux lies on the 104² boundary, outside the central 64² region,
or where the maximum power response of every pointing is below 0.1. The
wide-field guard therefore remains inactive in the fitted model, and the old
corner-pixel failure does not recur. Restored FITS and CASA-style comparisons
are included with the products.

### All-data mosaic refit and residual image

The validated 11,536-leaf consensus topology was then held fixed and fitted
to all 755,048 active samples. This is parameter estimation only: no
all-data residual was allowed to add or remove a split. A 10,816-leaf 16″
root model was fitted under the same Airy, wide-field, and
`lambda=3e-4` configuration as a control.

| all-data model | steps | converged | KKT | total flux | residual power |
|---|---:|---:|---:|---:|---:|
| 16″ base | 100 | yes | `2.58e-5` | 11.300 Jy | 0.01383 |
| mosaic consensus | 500 | no | `4.60e-5` | 10.763 Jy | 0.00857 |

The hierarchy reduces all-data normalized residual power by 38.0%. Each
pointing improves independently: C1 48.4%, C2 26.0%, C3 38.6%, C4 43.8%, C5
41.0%, C6 33.7%, and C7 25.6%. The four channel residual powers are 0.00909,
0.00816, 0.00807, and 0.00896, so no one frequency bin causes the gain. The
outer channels retain about 10--13% more residual power than the two inner
channels, which is a useful future spectral-model diagnostic.

A pointing-aware adjoint now forms residuals on one 208² grid at 8″. It uses
each pointing's own Fourier frame and Airy beam, then also supplies the local
sum-of-beam-squared sensitivity correction. Inside the 10% sensitivity
contour, the hierarchy lowers image RMS from 1.581 to 1.034 mJy/beam (34.6%),
robust RMS from 1.198 to 0.679 mJy/beam (43.4%), and absolute peak from 16.15
to 4.77 mJy/beam (70.4%). The central remnant-shaped residual is strongly
reduced, but the remaining residual has coherent broad structure. It should
not yet be interpreted as thermal noise.

The refined solve reached its 500-step limit and missed the physical KKT
tolerance by a factor of 1.53. The reported iterate is therefore the best
bounded result, not a certified stationary point. This does not erase the
sealed topology result, but it means optimizer progress reporting,
checkpointing, and either a longer deterministic finish or an SGD-to-FISTA
final refit should precede a production-scale claim. No positive all-data
flux lies on the 104² boundary, outside the old central 64² footprint, or
where every pointing has beam power at or below 0.1.

### Residual-tail and existing-flag audit

A leakage-safe residual audit now uses the frozen mosaic protocol. Robust
location and scale are fitted separately for every pointing, channel, and
correlation on the outer-training scans. The same score is then evaluated on
the held-out scans. The score is

\[
z = \frac{\sqrt{w}\,|V_{\rm observed}-V_{\rm model}|-
                \operatorname{median}_{\rm train}}
               {1.4826\,\operatorname{MAD}_{\rm train}}.
\]

At `z > 6`, 4.69% of discovery samples contain 63.8% of discovery residual
power. The held-out figures are 4.80% and 62.4%. The top 1% contains 34.7%
and 32.7% of residual power. The tail is therefore real and repeatable, but it
is not evidence for RFI by itself.

The strongest baseline rankings transfer almost unchanged to held-out scans.
For example, `ea04-ea08` has 89.7% discovery and 87.3% held-out outliers.
`ea04-ea16` has 82.2% and 78.5%, while `ea04-ea21` has 78.4% and 76.1%.
However, these rankings are strongly tied to projected baseline length. The
Pearson correlation between log median UV distance and discovery outlier
fraction is -0.70:

| median baseline range | baselines | median outlier fraction |
|---|---:|---:|
| below 0.75 kλ | 9 | 55.5% |
| 0.75--1.5 kλ | 26 | 26.2% |
| 1.5--3 kλ | 48 | 0.42% |
| 3--6 kλ | 73 | 0% |
| 6--12 kλ | 91 | 0% |

The dominant residual tail is thus more consistent with missing diffuse or
out-of-field emission, a beam/FOV deficiency, or another short-spacing model
error than with independent corrupt samples. The proposed baseline-removal
refit was deliberately not run. Removing these samples would improve the
reported residual by deleting the measurements that constrain the missing
large-scale sky.

The current MS flags were also audited in the opposite direction. Their
provenance is the CASA tutorial: scan 1, antennas `ea13`, `ea15`, and `ea05`,
plus a 10-second quack. They are not UV-selection flags. The frozen sky and
robust scales never see these samples. Predictions were evaluated at their
own averaged visibility coordinates:

| cohort | samples | fraction above `z=6` | tail share of residual power |
|---|---:|---:|---:|
| currently unflagged | 755,048 | 4.76% | 63.6% |
| currently flagged | 538,464 | 25.8% | 99.5% |

Of the flagged cohort, 399,414 samples, or 74.2%, are below `z=6`. At the
unflagged 99th-percentile threshold, 84.2% of flagged samples remain in the
residual bulk. The flagged set therefore mixes a small, extremely damaging
tail with many less exceptional values. This does not authorize immediate
unflagging: the affected samples were absent from the calibration solve, so
their present `CORRECTED_DATA` may not represent the calibration quality they
could attain. It does justify a later iterative test that recalibrates a
candidate subset before deciding whether to restore it.

Residual handling now has four explicit non-mutating modes:

1. `report_only` records rankings and never changes flags or weights;
2. `robust_weights` applies continuous Huber-like weights without hard flags;
3. `static_sky` permits hard residual proposals for a source known to be
   static;
4. `transient_safe` first protects residuals explained by a temporal or
   spectral sky response, then acts only on the orthogonal remainder.

The sky-coherence gate fits a real template coefficient on one set of
baselines and requires the same coefficient to reduce residual power on
disjoint baselines. A genuine variable point source has this interferometric
phase coherence. Baseline-local interference does not. Instrumental metadata
flags still override sky protection. These modes produce proposals only; no
audit path mutates the input block.

### Time, averaging, calibration, and outer-beam discrimination

The 104×104 root field is 27.73′ across. The six outer pointing centres are
2.45′ from C1. At the fixture's mean frequency of 4.599 GHz, the configured
Airy beam is set to zero outside 10.44′ from each pointing. The image therefore
covers the union of the *truncated* beam supports, but the forward model omits
the physical Airy sidelobes beyond the first null.

Time and frequency averaging are unlikely to create the short-baseline tail.
The fixture averages 64 input channels into four 32 MHz bins and uses 60-second
time bins. At 0.75 kλ and a 14--18′ offset, the expected decorrelation from
each operation is only of order (10^{-3}). Both effects grow with baseline
length, whereas the measured outlier fraction falls from 55.5% below 0.75 kλ
to nearly zero above 3 kλ. This still needs a matched finer-fixture check, but
its predicted signature is opposite to the observed one.

A contiguous time split initially looked ambiguous. Baseline outlier fractions
in the two halves have Pearson correlation 0.871, so the defect follows UV
geometry. A positive 64×64 residual sky fitted on either half improved that
half, but worsened the opposite half. The two halves sample different UV tracks,
so this is not a fair rejection of a static wide-field component.

The first outer-field fit also exposed a regularization error. Penalizing
intrinsic flux as (lambda\sum_j I_j) strongly rejects sources seen through a
weak sidelobe. A source behind a 1% power response pays about 100 times more
penalty than an equally visible central source. Mosaic quadtree inference now
accepts per-leaf sparsity weights. The diagnostic uses the relative column norm

\[
d_j = \frac{\sqrt{\sum_{p,c,r} w_{pcr} B_{pcj}^2}}
             {\max_k \sqrt{\sum_{p,c,r} w_{pcr} B_{pck}^2}},
\qquad
R(I)=\lambda\sum_j d_j I_j .
\]

Here (B_{pcj}) is the power-beam response of pointing (p), channel (c),
and leaf (j). This is a beam-only approximation to normalizing the columns of
the full measurement operator. It puts the L1 threshold on detectable flux
rather than intrinsic flux and combines all mosaic pointings in quadrature.

The decisive control alternates complete 60-second bins between discovery and
evaluation. The two partitions then have almost identical UV coverage; their
baseline outlier-fraction correlation is 0.9985. Results below are held-out
relative changes in weighted complex MSE, with negative values denoting an
improvement:

| residual component | direction | all UV | below 0.75 kλ | 0.75--1.5 kλ |
|---|---|---:|---:|---:|
| truncated Airy support | even→odd | -4.59% | -6.63% | -5.57% |
| truncated Airy support | odd→even | -4.60% | -6.72% | -5.50% |
| extended Airy sidelobes | even→odd | -7.30% | -10.17% | -9.32% |
| extended Airy sidelobes | odd→even | -7.33% | -10.31% | -9.24% |

The training and held-out improvements agree to within about 0.3 percentage
points in every row. A stable missing component is therefore present. About
4.6 percentage points come from a coarse correction inside the current beam
support. Extending the beam and field adds another 2.7 percentage points
overall and about 3.6 points below 0.75 kλ.

An external catalogue check independently confirms relevant outer sky. The
[NVSS catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/65) contains a
653.3 mJy source at 18:49:32.79, -00:38:02.0, or approximately
((l,m)=(+2.2′,+17.7′)). The
[VLASS 3 GHz catalogue](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/255/30)
places a 573.1 mJy component at the same position. Both independently fitted
coarse residual skies select the corresponding ((+2.5′,+17.5′)) cell as their
brightest component. The extended analytic Airy model predicts 4--11 mJy of
apparent flux from this source in five of the seven pointings, while the current
truncated beam predicts exactly zero.

A fixed-position delta atom for this catalogue source fits 0.752--0.758 Jy in
the two interleaved partitions and improves the opposite partition by
1.81--1.87% overall. The fitted flux should not be used as a source measurement:
the ideal Airy sidelobe, source structure, catalogue epoch, and 1′ diagnostic
grid all contribute uncertainty. The atom barely changes the residual below
0.75 kλ, so it explains only part of the broad tail. Other catalogue sources,
diffuse central structure, and beam error remain relevant.

A deliberately low-complexity self-calibration control fitted one static
residual antenna gain per pointing and contiguous time half. Phase-only gains
improved held-out baselines within the fitted half in 12/14 cases, with median
6.0%, but worsened the opposite half in 13/14 cases, with median 11.8%.
Amplitude-plus-phase gains improved every within-half result, with median
32.2%, but worsened 12/14 opposite-half results, with median 26.5%. Typical
solutions were 0.056 rad in phase and 3.4% in log amplitude. These fits are now
known to be confounded: an incomplete static sky projects into different
antenna gains as the UV track rotates. They do not establish time-varying gain
error. Self-calibration must be repeated after the wide-field sky is included.

The next imaging change should therefore be a two-tier model: retain the
high-resolution central hierarchy, add sparse fixed-position outer catalogue
atoms plus an optional coarse diffuse field, use non-truncated beam support,
and apply sensitivity-normalized sparsity weights. Refit the frozen topology
and repeat the residual and self-calibration audits before increasing gain
complexity. A matched finer time/frequency fixture remains the final averaging
control and requires the original measurement set.

That composite fixed-topology solver is now implemented. It jointly optimizes
an arbitrary tuple of quadtree dictionaries and exact delta atoms under one
positive weighted-L1 visibility objective. The result retains the prediction
from each group, so the central hierarchy, coarse field, and catalogue atoms
can be ablated without changing the forward model. A fixed prediction can also
be supplied for staged diagnostics, although the scientific comparison uses a
joint refit.

The 3C391 comparison uses five interleaved 60 s folds: three for fitting, one
for lambda and stopping-time selection, and one sealed test fold. The old
topology is reused, but its flux is initialized to zero because the saved flux
was fitted using data that overlap this new test. The beam is an analytic Airy
pattern with support extended beyond the first sidelobes. The outer field is a
64 square grid of 60 arcsec pixels. The production entry point is
`scripts/fit_3c391_composite.py`.

The exact outer atoms now come from a reproducible catalogue snapshot rather
than a hand-entered source. `scripts/build_3c391_radio_catalog.py` queries NVSS
for sources above 50 mJy within 1 degree, rejects sources inside the central
13.87 arcmin half-width and components larger than 45 arcsec, then ranks them
by their maximum predicted apparent flux over every pointing and channel. It
cross-matches the retained positions against reliable, non-duplicate VLASS
main-sample components within 20 arcsec. The checked snapshot contains three
VLASS components at offsets of 17.7, 32.4, and 38.3 arcmin. Their maximum
predicted apparent fluxes are 8.37, 1.63, and 0.75 mJy under the extended Airy
beam. This is an (I B) selection: a bright intrinsic source is included only
when at least one pointing has enough response for it to affect the data.

Catalogue fluxes are extrapolated to the observation frequency with a default
spectral index of -0.7 and used only to initialize the fit. Every catalogue
atom retains a free non-negative flux. The CSV records position, flux, shape,
epoch, catalogue, and source URL; the adjacent JSON records the complete query
and selection protocol. The current atom is a delta function even when VLASS
reports a small deconvolved size. A Gaussian or multi-component atom is a
future refinement if the held-out residual supports that extra freedom.

The frozen four-way ablation completed on the same folds at
`lambda=3e-4`:

| model | validation MSE | sealed-test MSE | sealed-test change |
|---|---:|---:|---:|
| central hierarchy | 0.00332916 | 0.00336603 | reference |
| central + coarse outer field | 0.00328038 | 0.00331372 | -1.55% |
| central + three catalogue atoms | 0.00314863 | 0.00316959 | -5.84% |
| central + coarse + catalogue | 0.00313282 | 0.00315317 | -6.32% |

The catalogue therefore explains most of the reproducible missing component.
The coarse field adds a smaller but consistent 0.52% sealed-test improvement
after the catalogue sources are present. The full model also reduces the
sealed short-baseline MSE from 0.0438203 to 0.0413862, or 5.56%. Its fitted
catalogue fluxes are 0.565, 0.694, and 0.571 Jy; these are almost unchanged
from the catalogue-only fit. This stability argues against strong degeneracy
between the exact sources and coarse field. The full fit selected its last
500-step checkpoint and reached KKT `3.44e-5`, just above the `3e-5` target.
The broad conclusion is held-out and stable, but a continuation fit is needed
before interpreting the small incremental coarse-field gain precisely.

### Post-composite residual and flag audit

The continuation fit reached a numerical plateau rather than a materially
different sky solution. Another 300 FISTA steps changed the physical KKT
residual from `3.44e-5` to `3.42e-5` and changed sealed-test MSE by less than
0.01%. Further continuation with the same float32 operator is not useful.

The residual audit was then repeated with the complete central, coarse, and
catalogue model. Sealed normalized residual power fell by 6.32%. On the
central model's fixed robust scale, the fraction above `z=6` fell from 4.899%
to 4.552%, a 7.09% relative reduction. This is the portion of the old tail
that can reasonably be attributed to the missing wide-field sky.

The conclusion changes when the robust scale is estimated again after the
bulk residual narrows. The full model still has 4.990% of sealed samples above
`z=6`, compared with 4.899% for the central model. It produces 25 validated
baseline candidates rather than 24. The persistent relative tail is therefore
not dominated by the missing catalogue sources or coarse outer field.

The reverse audit reaches the same conclusion. The composite sky changes
normalized residual power in the currently flagged cohort by only -0.041%.
Using the central model's fixed scale, its outlier fraction changes from
27.122% to 27.134%. About 72.9% of currently flagged samples remain in the
robust residual bulk, but they cannot be restored safely until calibration is
repeated with those samples present.

The next experiment should hold the composite sky fixed and vary calibration
time complexity and interpolation. Gain models must be selected on held-out
time and baseline cells. A residual classifier should wait until this test is
complete, because training it now would label a mixture of corruption,
calibration error, and remaining sky-model error as one class.

### Fixed-sky calibration complexity result

That experiment is complete. The independent calibrator solve provides six
gain epochs over the observation. Four time models were compared: one constant
gain, linear and quadratic trends in log amplitude and unwrapped phase, and
all six native gain epochs. Nearest and linear transfer were tested wherever
they differ. The composite sky remained fixed. Interleaved time fold 3 selected
the calibration model and fold 4 was opened only as the sealed check.

| gain time model | transfer | validation power | sealed-test power |
|---|---|---:|---:|
| six native epochs | linear | 0.009602 | 0.009464 |
| quadratic trend | linear | 0.010740 | 0.010694 |
| linear trend | linear | 0.011185 | 0.011253 |
| six native epochs | nearest | 0.011600 | 0.011758 |
| linear trend | nearest | 0.012942 | 0.013067 |
| quadratic trend | nearest | 0.013482 | 0.013549 |
| constant | nearest | 0.033429 | 0.033862 |

Native linear interpolation improves normalized residual power by 17.2% on
the selection fold and 19.5% on the sealed fold relative to native nearest
assignment. The identical ranking on both folds supports changing the default
transfer policy to linear. The trend models lose, so the six calibrator epochs
contain useful non-linear time structure that should not be collapsed into a
global low-order curve.

CASA-corrected data still score 0.007547 on the sealed fold. The selected JAX
calibration is 25.4% higher. The excess is broad: it is 18.8--35.1% across all
seven pointings and 23.6--26.1% across all four frequency bins. This does not
look like one bad pointing, one beam edge, or one band edge.

The selected calibration also does not remove the robust residual tail. With
separately estimated scales, sealed `z>6` fractions are 4.99% for CASA and
4.93% for the selected JAX calibration. On CASA's fixed scale, however, the
JAX fraction is 6.23%, and validated baseline candidates rise from 42 to 55.
The next calibration task is therefore a term-by-term comparison against the
portable CASA oracle. Substitute or compare `G`, `K`, `B`, flux scale, and
weight propagation independently before adding more target-sky gain freedom.

### Fixed-sky calibration-term ablation

The term ablation is complete. It used the same frozen composite sky and
interleaved folds as the time-complexity study. Fold 3 remained the selection
fold and fold 4 remained sealed. The complete 2^3 `G`/`K`/`B` factorial used
post-application flags and fixed MeasurementSet weights. Separate controls
removed the flux transfer and propagated gain weights.

CASA and SL1MJax put a constant complex antenna factor in different places
between `G` and `B`. Naive term substitution was therefore not identifiable:
one hybrid had normalized residual power above 2,000 while its complementary
hybrid was near 0.95. Before the final ablation, both solutions were put in the
same gauge by making each valid antenna/receptor bandpass unity at the nearest
valid reference channel and moving that factor into `G`. This transformation
preserves the complete Jones term to numerical precision. A fully invalid
antenna/receptor remains invalid and receives a neutral factor only for the
otherwise arbitrary stored value.

| calibration | validation power | sealed-test power |
|---|---:|---:|
| CASA `G`, CASA `K`, CASA `B`, propagated weights | 0.007403 | 0.007459 |
| CASA `G`, CASA `K`, CASA `B` | 0.007462 | 0.007548 |
| CASA `G`, CASA `K`, JAX `B` | 0.007526 | 0.007596 |
| CASA `G`, JAX `K`, JAX `B` | 0.007564 | 0.007687 |
| JAX `G`, CASA `K`, CASA `B` | 0.009469 | 0.009297 |
| JAX `G`, JAX `K`, JAX `B`, propagated weights | 0.009520 | 0.009363 |
| JAX `G`, JAX `K`, JAX `B` | 0.009602 | 0.009464 |

Replacing only `G` reduces validation power by 21.2% and sealed power by
18.8%. Replacing only `K` reduces them by 0.4% and 1.0%. Replacing only `B`
changes them by -0.02% and +0.19%, so there is no repeatable bandpass benefit.
Marginalizing over the other factorial terms gives the same conclusion:
`G`, `K`, and `B` improve sealed power by 18.7%, 1.6%, and 0.17% respectively.
The factorial interactions are below 0.6%.

The complete imported CASA solution reproduces stored CASA-corrected
visibilities to `1.18e-6` normalized sealed power. The comparison path is
therefore faithful enough to localize the gap. Propagating gain weights after
averaging improves sealed power by about 1.1% for either solution, but it does
not explain the JAX/CASA difference. Removing the flux transfer raises power
to about 0.318 for both solutions, so absolute flux scaling is essential but
is also not the source of the remaining gap.

The practical result is that the next calibration task is `G`, not a more
flexible target sky, another bandpass model, or residual classification. The
saved JAX solution has six gain epochs, while the CASA gain table has fourteen.
The next implementation should solve the gain calibrator from the complete
local MeasurementSet rather than the compact golden subset. It should compare
native per-interval linear transfer with a constrained amplitude/phase time
model under the same frozen-sky validation protocol. A circular phase GP is a
good candidate in that comparison, but it should compete against the complete
fourteen-epoch piecewise-linear baseline rather than replace it by assumption.

Reproduction artifacts are
`scripts/ablate_3c391_calibration_terms.py` and
`outputs/3c391_calibration_term_ablation/`.

### Full-row gain-calibrator baseline

The first follow-up is also complete. All 39,733 J1822-0938 rows were read
from the locally cached MeasurementSet with the saved pre-calibration flags.
They contain 3,198,080 active parallel-hand channel samples in fourteen
calibrator scans. The solver now accepts a per-row gain-solution coordinate
that is distinct from observation time. This keeps the actual timestamps for
antenna-position phase while sharing one `G` parameter across each scan. Each
gain knot is placed at the scan's active-weight time centroid. These centroids
agree with CASA's fourteen gain-table times to much better than one second.

The 300-step full-row solve took 15.7 seconds locally. Calibrator train and
connected-baseline holdout RMS are 0.0726 and 0.0740. With native linear gain
transfer, frozen-sky fold-3 residual power is 0.007478. This is 22.1% below
the six-knot result of 0.009602 and only 0.25% above the CASA value of 0.007459.
Every frequency bin remains finite and the aggregate validation RMS is 0.0865.

This result confirms that missing calibrator sampling, rather than target
self-calibration or bandpass complexity, caused almost all of the measured
CASA/JAX gap. Fold 4 remains closed. The constrained amplitude/phase and GP
models must be fitted and ranked on fold 3 before the final sealed comparison.

Reproduction artifacts are `scripts/fit_3c391_full_scan_gains.py`,
`outputs/3c391_full_gain_calibrator_fixture.zarr/`, and
`outputs/3c391_full_scan_gain_baseline/`.

### Gain-time smoothing and circular-GP selection

The fourteen-epoch gain solution was compared with two constrained time-model
families. Seven models penalized irregular-grid second derivatives in log
amplitude and unwrapped phase, with strengths from 0.01 to 10. Twelve RBF GP
models used length scales of 900, 1,800, 3,600, and 7,200 seconds and noise
variances of 0.001, 0.01, and 0.1. The GP models amplitude in log space and
phase as a complex unit vector, then projects the posterior phase mean back to
the unit circle. This makes phase-wrap handling explicit.

All twenty candidates used identical fixed `K`, `B`, antenna-position, flux,
flag, weight, and averaging terms. Application computes the complete native
Jones correction once per raw chunk, then applies each candidate through the
exact native/candidate baseline-gain ratio. A synthetic regression agrees with
independent full Jones application to `2e-14`. Per-pointing candidate blocks
are checkpointed before scoring.

| gain-time model | fold-3 power | change from native |
|---|---:|---:|
| native linear | 0.007478 | -- |
| curvature penalty 0.01 | 0.007519 | +0.54% |
| curvature penalty 0.03 | 0.007540 | +0.82% |
| curvature penalty 0.1 | 0.007572 | +1.25% |
| best circular GP: 1,800 s, noise 0.1 | 0.007637 | +2.13% |
| curvature penalty 10 | 0.008088 | +8.15% |

Native linear is the explicit zero-smoothing limit. The curvature sequence
worsens monotonically away from that limit, and the GP grid covers scales on
both sides of the calibrator cadence. This is therefore a supported selection,
not an unresolved optimum on a hyperparameter boundary. Fold 4 was opened
once after the ranking was fixed. Native linear scores 0.007607 there, versus
0.007547 for CASA, a remaining difference of 0.79%. Native is better on C4 and
C5, while CASA is better on the other five pointings.

The result does not reject GP-constrained self-calibration in general. It says
that smoothing these well-measured external-calibrator epochs removes useful
time structure. A later self-calibration GP can still anchor known calibrator
points and regularize additional learned points, but it must again compete
against an unsmoothed baseline on independent data.

Reproduction artifacts are `src/sl1mjax/gain_time_models.py`,
`scripts/sweep_3c391_gain_time_models.py`, and
`outputs/3c391_gain_time_model_sweep/`.

### Matched residual-tail audit after calibration selection

The residual flag audit was repeated after selecting the 14-epoch native
linear calibration. CASA and SL1MJax were compared on the intersection of
their post-calibration active samples. Both used the same frozen composite
sky, the same 60 s five-fold split, and the same robust-score threshold of 6.
No flags were changed.

On sealed fold 4, CASA has normalized residual power 0.007441 and an outlier
fraction of 4.993%. SL1MJax has 0.007560 and 5.066%. The SL1MJax outlier
fraction is 5.051% when evaluated with CASA's fitted robust scales. More
importantly for flag discovery, both calibrations validate exactly the same
25 baseline groups. Neither calibration adds or removes a candidate group.

The residual-tail evidence is therefore stable to the remaining 0.8--1.6%
calibration-level differences. It is reasonable to resume flagging work
without first developing a more complex external-calibrator interpolator.
The next flagger should retain at least two modes: a conservative
instrumental mode that requires repeated baseline, antenna, channel, or scan
support, and a transient-safe audit mode that reports isolated sky-coherent
events without automatically masking them.

The matching code is `scripts/compare_3c391_calibration_flags.py`, with tests
for visibility alignment and common-mask construction. Results are in
`outputs/3c391_calibration_flag_audit/`.

The retrospective flagged-data comparison was completed when the full
Measurement Set became available on BagOfWinds. CASA and SL1MJax were first
intersected at native row, channel, and correlation resolution, then
frequency- and time-averaged. This matters because removing invalid gains
after averaging changes the effective time and UVW coordinates. The final
paired cohorts contain 267,640 samples and have exactly equal coordinates.

Of these originally flagged samples, the SL1MJax-calibrated data have
normalized residual power 0.01888 and a score-above-6 fraction of 7.677%.
Thus 92.323% sit in the robust residual bulk. CASA `CORRECTED_DATA` is not a
useful control for this cohort: its normalized residual power is 1.915.
CASA calibration did not produce meaningful corrected values for samples
that were already flagged, while applying the external SL1MJax solution to
raw `DATA` did.

Twenty-two baselines reproduce a high outlier fraction in both discovery and
flagged evaluation samples. They contain 24,008 samples, or 8.97% of the
usable flagged cohort, and capture 60.13% of its score-above-6 samples. The
remaining 243,632 samples have an outlier fraction of 3.362%, lower than the
5.066% in the normally active sealed cohort. This supports a conservative
recovery experiment: retain unavailable-gain samples as flagged, retain the
22 persistent bad-baseline groups as flagged, and admit the remaining
originally flagged samples only in a new validation-controlled imaging run.
It does not support unflagging the whole Measurement Set in place.

The paired extraction is in `outputs/3c391_matched_existing_flag_audit/`.
The detailed audit and reason breakdown are in
`outputs/3c391_calibration_flag_audit/`. No Measurement Set flags were
changed during this experiment.

This work also exposed and fixed a validation bug in the physical-flux
solvers. FISTA, proximal SGD, and hybrid fits previously recorded holdout loss
but returned the lowest training objective. They now return the checkpoint
with the lowest holdout loss when a holdout is supplied. No-holdout fits retain
the previous minimum-objective behavior.

### Validation-controlled recovery result

The recovery-policy fit was then repeated from zero flux to remove warm-start
leakage. Four policies used folds 0--2 for fitting and fold 3 for selection.
They were evaluated on the same originally active samples, so admitting more
flagged data could not improve a candidate merely by changing its evaluation
set.

| policy | fold-3 weighted complex MSE | change from active only |
|---|---:|---:|
| active only | 0.00324258 | -- |
| robust flagged weights | 0.00328037 | +1.17% |
| supported flagged tail | 0.00328412 | +1.28% |
| whole supported baselines | 0.00329502 | +1.62% |

The active-only policy therefore remains selected. Its sealed fold-4 weighted
complex MSE is 0.00328519 and normalized residual power is 0.00781493 on
146,304 samples. This does not mean every existing flag marks bad data. It
means that the proposed broad recovery rules do not improve prediction of
independent active samples with the present calibration and static sky.

The best frozen sky was also evaluated directly on 267,640 usable,
pre-existing flagged averages. Only 7.11% exceed robust score 6, compared with
4.66% of the active cohort. A pointwise residual-score classifier has ROC AUC
0.489 against all active samples, so the existing flags are not pointwise
corruption labels. About 70.9% of the usable flagged cohort lies in the first
15 seconds of a scan and has only a 3.66% residual-tail fraction. The later
flagged samples contain most of the residual power and are much more strongly
associated with the validated bad-baseline support.

The practical flagger should therefore use contextual causes such as antenna,
scan start, gain validity, baseline persistence, and closure coherence. A
transient-safe mode should report isolated time-frequency events without
automatically converting them into corruption flags. The recovery artifacts
are in `outputs/3c391_recovery_policy_fit_zero/`. The paired visibility audit
and amplitude/phase diagnostics are in
`outputs/3c391_flagged_visibility_distribution/`.

### Native-resolution averaging ablation

The best sealed active-only sky was next evaluated on the original target
resolution available in this Measurement Set: 10-second integrations and 64
channels of 2 MHz. The sealed fold-4 time bins contain 12,691,712 active
parallel-hand complex samples across all seven pointings. Each observation was
predicted at its native UVW, frequency, and primary-beam response before any
averaging.

For every averaging case, `exact` means that native predictions and
observations were averaged with identical flags and weights. `Centroid` means
that the sky was instead evaluated once at the averaged UVW and channel
frequency, as in the current coarse fixture.

| averaging | retained samples | exact weighted MSE | exact normalized power | mean \(w|r|^2\) per sample | ratio to native | centroid MSE change |
|---|---:|---:|---:|---:|---:|---:|
| 10 s / 2 MHz | 12,691,712 (100%) | 0.00988387 | 0.0231451 | 0.09369 | 1.00 | +0.000001% |
| 20 s / 4 MHz | 3,353,536 (26.42%) | 0.00513640 | 0.0121632 | 0.18427 | 1.97 | +0.00035% |
| 30 s / 8 MHz | 1,132,928 (8.93%) | 0.00395174 | 0.00938427 | 0.41965 | 4.48 | +0.00121% |
| 60 s / 32 MHz | 146,304 (1.15%) | 0.00328472 | 0.00781382 | 2.70112 | 28.83 | +0.01429% |

The fall in weighted MSE and normalized residual power does **not** show that
coarser data fit better. Incoherent noise averages down while the denominator
still contains the summed weights. The resolution-comparable diagnostic is
the mean of $w|V-\hat V|^2$ over retained complex output samples. It rises by
28.8 times at 60 s / 32 MHz, so much of the remaining residual is coherent
within the large averaging cells. At the same time only 1.15% of the native
time-frequency samples remain. Short transients and narrow spectral features
would therefore lose most of their independent validation evidence.

For this dataset, the centroid forward-model approximation itself is small.
Its signed 60 s / 32 MHz MSE change ranges from -0.035% to +0.103% across
pointings. The largest aggregate baseline-bin change is +0.160% at 5--7.5
kilo-lambda. The recomputed centroid result has normalized residual power
0.007814932719, agreeing with the previously sealed value 0.007814932845 to
$1.3\times10^{-10}$. This is a strong end-to-end check of extraction,
calibration, fold selection, averaging, beam evaluation, and prediction.

The result supports a two-resolution workflow. Coarse data can remain the
cheap topology-discovery and initial-solve representation. Model selection for
time, frequency, flag recovery, and self-calibration should always be scored
on streamed native-resolution holdouts. A bin-integrated forward model is not
urgent for this particular 3C391 mosaic, but the conclusion is dataset- and
field-of-view-dependent and should be rechecked for longer baselines or bright
far-sidelobe sources.

The implementation is `scripts/ablate_3c391_native_averaging.py`. It can first
write compact calibrated holdout fixtures with `--extract-only`, then resume
GPU prediction one pointing at a time. Results and the resolved diagnostic are
in `outputs/3c391_native_averaging_ablation/`.

### Native transient and spectral injection recovery

The first variable-sky recovery test now works directly on the native C1
holdout. It injects a one-Jy point-source RIME response into the real
calibrated residual. The response uses the original UVW coordinates, all 64
channel frequencies, and the same extended VLA Airy beam as the frozen sky
model. The initial atom is at the C1 mosaic phase centre, where beam power is
close to one.

Each candidate coefficient is fitted on complete discovery baselines and
scored on disjoint evaluation baselines. Five deterministic repeated
whole-baseline holdouts are used. A paired zero-injection control removes a
pre-existing coherent residual at the atom position from the reported
injected coefficient and loss gain. This is important because a real-data
injection inherits calibration error, missing sky, and correlated residuals
rather than adding ideal thermal noise to a perfect simulation.

The matched support is known in this test. Temporal supports contain one,
three, or six native 10-second integrations. Spectral supports contain one,
two, four, or eight adjacent native 2 MHz channels. A static point atom at the
same position is evaluated as a competing model. A repeatable recovery
requires the supported model to have the lowest mean raw validation loss,
positive paired evidence relative to the zero-injection control, and a paired
win on at least four of five baseline splits.

| injected support | first repeatable selection | injected flux | first 25% protection |
|---|---:|---:|---:|
| 10 s transient | nominal SNR 4 | 8.11 mJy | 64.9 mJy |
| 30 s transient | nominal SNR 8 | 9.37 mJy | 74.9 mJy |
| 60 s transient | nominal SNR 16 | 13.25 mJy | not reached by 53.0 mJy |
| 1 channel / 2 MHz | nominal SNR 1 | 2.27 mJy | 72.7 mJy |
| 2 channels / 4 MHz | nominal SNR 1 | 1.61 mJy | 102.8 mJy |
| 4 channels / 8 MHz | nominal SNR 1 | 1.14 mJy | 72.7 mJy |
| 8 channels / 16 MHz | nominal SNR 2 | 1.61 mJy | not reached by 51.4 mJy |

The nominal SNR is the injected flux divided by the inverse square root of the
discovery response information. It is not an empirical detection sigma. The
real residual is correlated, so the repeatable temporal threshold is several
times larger than this independent-noise scale.

The current 25% event-power improvement threshold is appropriate as a
conservative rule for protecting a large residual from automatic flagging,
but it is not a suitable model-detection threshold. It misses variable-sky
components that are selected consistently by native validation at much lower
flux. Detection and flag protection should therefore remain separate
decisions. A later blind search must also use an additional selection fold or
a multiple-testing correction because this experiment supplies the true time,
frequency support and sky position.

The reusable implementation is `src/sl1mjax/sky_recovery.py`. The 3C391 driver
is `scripts/run_3c391_native_injection_recovery.py`. Its full machine-readable
result and cached unit response are in
`outputs/3c391_native_injection_recovery/`.

### Blind fixed-atom time and frequency search

The matched-support experiment does not include the cost of finding an event.
A first blind search now tests 350 native-resolution refinements at the C1
phase centre. The bank contains 104 sliding time intervals, 245 sliding
frequency intervals, and one linear slope in log frequency. Time intervals
have widths of 1, 3, and 6 native 10-second integrations. Frequency intervals
have widths of 1, 2, 4, and 8 native 2 MHz channels. Intervals do not cross a
scan gap.

The search fits a nested model. Both hypotheses can correct the static flux at
the atom. The larger hypothesis also adds one signed time- or frequency-varying
coefficient. This prevents an ordinary static sky error from being labelled as
variability. The interval coefficient is a contrast between samples inside and
outside the selected interval. The log-frequency coefficient is a screening
approximation to a physical spectral model, not yet a fitted power law.

Whole antenna baselines are divided into three cohorts. Discovery baselines fit
and rank all candidates. Selection baselines choose among the best 16 without
refitting them. The chosen candidate is then refitted on discovery plus
selection baselines. Only that candidate is scored on the sealed evaluation
baselines. Five deterministic baseline partitions measure stability. A
repeatable recovery requires the exact injected interval to improve sealed
loss on at least four of five partitions.

The real residual with no injection is an important null control. Discovery and
selection always find an apparently useful time interval, but all five choices
increase sealed loss. The acceptance fraction is therefore zero. This shows
why a two-way discovery/evaluation split is not sufficient after mining many
candidates.

| injected component | matched-support threshold | blind threshold |
|---|---:|---:|
| 10 s interval | 8.11 mJy | 16.22 mJy |
| 30 s interval | 9.37 mJy | 18.73 mJy |
| 60 s interval | 13.25 mJy | 13.25 mJy |
| 1 channel / 2 MHz | 2.27 mJy | 18.18 mJy |
| 2 channels / 4 MHz | 1.61 mJy | 12.85 mJy |
| 4 channels / 8 MHz | 1.14 mJy | 9.09 mJy |
| 8 channels / 16 MHz | 1.61 mJy | 6.43 mJy |
| log-frequency slope | not tested | 10 mJy edge-to-edge |

The time search pays a modest trials penalty because only 104 intervals are
tested and the longer events carry more information. The compact spectral
search pays about an eightfold amplitude penalty relative to known support.
There are 245 overlapping spectral intervals, and the real residual contains
coherent alternatives that differ between baseline cohorts. A stronger search
penalty or a poorer residual model could both cause this behaviour. The sealed
null result means it is not evidence for a real variable component at the phase
centre.

This is a fixed-position proof of the time and frequency selection protocol.
It does not yet decide which spatial leaf should gain extra freedom. The next
step is to screen spatial leaves using their residual response, apply this
three-cohort search only to a short spatial list, then jointly refit accepted
components with the static hierarchical sky. Smooth power-law, Hanning, spline,
or Gaussian-process models can then compete as extra validated families rather
than replacing the compact interval search.

The reusable search is in `src/sl1mjax/sky_recovery.py`. The native 3C391
driver is `scripts/search_3c391_native_variability.py`. Full fold rankings and
injection results are in
`outputs/3c391_native_variability_search/summary.json`.

### Spatially blind native variation search

The fixed-position protocol now extends over the fitted hierarchical sky. The
frozen composite model contains 11,536 central quadtree leaves. The first
bounded experiment takes the 768 leaves with the largest frozen apparent flux,
defined as fitted integrated flux times RMS primary-beam power across the 64
channels. This gate uses only the already-frozen sky and beam, so it does not
inspect the native holdout residual. It currently targets variation in sources
with persistent mean emission; a later empty-sky pass is still needed for
purely transient sources.

Every retained leaf uses its exact square-pixel width, wide-field phase, and
extended Airy beam response. The implementation streams responses in leaf and
row tiles and keeps only matched-filter sufficient statistics by time and
channel. It therefore searches 768 spatial leaves times 350 temporal and
spectral models without storing a visibility-by-leaf matrix. Discovery first
retains 16 spatial leaves and 64 joint spatial/variation candidates. A separate
baseline cohort selects one pair. That leaf's static correction and variation
coefficient are refitted together on discovery plus selection baselines before
one sealed-baseline evaluation.

Synthetic tests recover an injected off-centre leaf and its exact temporal or
spectral interval. The streamed calculation also agrees with materialized unit
responses to numerical precision. Corrupting only evaluation baselines changes
the reported evaluation loss but cannot change the selected leaf or interval.

The real C1 residual does not behave like the earlier phase-centre null. Five
complete baseline partitions all select the same six-integration interval,
`time_0016_w006`. The best leaf moves within one compact group of adjacent
level-1 leaves, as expected for highly correlated pixels representing the same
resolved feature.

| split seed | selected leaf `(level, iy, ix)` | static correction | interval correction | sealed loss reduction |
|---:|---|---:|---:|---:|
| 391 | `(1, 109, 117)` | -3.71 mJy | -24.47 mJy | 0.522% |
| 392 | `(1, 106, 118)` | -3.59 mJy | -26.73 mJy | 0.313% |
| 393 | `(1, 109, 117)` | -2.76 mJy | -24.57 mJy | 0.639% |
| 394 | `(1, 105, 118)` | -3.83 mJy | -21.94 mJy | 0.431% |
| 395 | `(1, 111, 116)` | -2.44 mJy | -20.92 mJy | 0.357% |

The selected region is about 1.94 arcmin from the C1 phase centre, near RA
282.3210 degrees and declination -0.9157 degrees. The interval is 2010-04-24
10:20:06 to 10:21:06 UTC, using the Measurement Set time convention. It is the
complete six-integration segment from scan 29 present in this fixture.

This is a repeatable model discrepancy, but it is not yet evidence for
minute-scale astrophysical variation. The fixture contains only the sealed
interleaved time fold. The immediately neighbouring scan-29 integrations are
absent, so the test cannot determine whether the decrement is confined to one
minute, persists through the scan, or tracks a calibration change. The common
negative sign across adjacent leaves also makes a scan-dependent gain or
pointing residual plausible. The next discriminating experiment should extract
the complete contiguous scan 29 at native resolution, preserve the frozen sky,
and compare a sky-local variation against per-antenna and scan-wide calibration
models before adding this component to the sky.

The reusable streamed search is in `src/sl1mjax/sky_recovery.py`. The driver is
`scripts/search_3c391_native_spatial_variability.py`. Full rankings for all five
partitions are in `outputs/3c391_native_spatial_variability_search/`.

### Complete scan-29 residual-model comparison

The contiguous-scan diagnostic resolves the main ambiguity in the spatial
search. Scan 29 is one five-minute C1 scan with 30 native 10-second integrations
and 300 baselines in the Measurement Set. Two integrations contain no active
parallel-hand samples after the existing flags and calibration validity mask.
The fitted block therefore contains 7,084 rows, 28 integrations, 64 channels,
two parallel hands, and 906,752 active complex samples. The previously detected
event covers the final six active integrations, or 194,304 samples.

The frozen composite sky is not refitted. Every candidate receives the same two
static nuisance terms: one fractional scale for the complete frozen sky and one
signed correction to the previously selected quadtree leaf. Four event models
then compete over the same fixed final-minute support:

1. a signed change in the local sky leaf;
2. one common multiplicative amplitude change;
3. a common amplitude plus two finite-difference primary-beam pointing shifts;
4. a common amplitude plus linearised per-antenna amplitude and phase changes.

Complete baselines are divided 60:20:20 between coefficient fitting, family and
ridge selection, and stage-held-out evaluation. Only the selected family is
refitted on the first two cohorts before evaluation. Large response matrices are
not materialised. The implementation accumulates complex weighted normal
equations in row tiles. Pointing derivatives and each exact square-leaf response
are cached separately from the frozen full-sky prediction.

All five partitions select the per-antenna complex-gain event. Four select a
ridge fraction of 0.01; seed 394 selects 0.0001. The first improvement column
below is measured on family-selection baselines against the common static
nuisance model. The second is the corresponding stage-held-out comparison after
both models are independently refitted on discovery plus selection baselines.

| split seed | selected ridge | selection gain beyond static | held-out gain beyond static | held-out gain from frozen sky |
|---:|---:|---:|---:|---:|
| 391 | 0.01 | 4.47% | 0.82% | 7.57% |
| 392 | 0.01 | 3.72% | 2.74% | 6.04% |
| 393 | 0.01 | 2.62% | 3.72% | 13.38% |
| 394 | 0.0001 | 4.12% | 1.87% | 9.53% |
| 395 | 0.01 | 3.81% | 4.60% | 16.85% |

The event-gain improvement beyond the static nuisance model is positive in
every held-out partition. Its mean is 2.75%, with a sample standard deviation
of 1.49 percentage points. The alternative families are much weaker on model
selection baselines:

| event family | mean gain beyond static | range across partitions |
|---|---:|---:|
| local sky leaf | 0.066% | 0.052--0.086% |
| common amplitude | 0.098% | 0.052--0.148% |
| common pointing | 0.306% | -0.183--1.108% |
| per-antenna complex gains | 3.747% | 2.618--4.472% |

The phase coefficients are also stable across baseline partitions. Relative to
the calibration reference antenna, antenna 2 is -0.200 to -0.225 rad, antenna 9
is -0.169 to -0.184 rad, antenna 20 is -0.151 to -0.174 rad, and antenna 7 is
-0.144 to -0.157 rad. Stable amplitude terms include antenna 23 at roughly
-3.2% to -7.1%, antenna 13 at -0.7% to -6.8%, and antenna 6 at -2.7% to -7.6%.
These coherent antenna identities are not expected from a real compact change
in one sky leaf.

The current conclusion is that the spatial decrement is mainly a projection of
an antenna-based calibration residual into the sky dictionary. A single common
amplitude does not explain it, and a common pointing shift is not stable across
partitions. The gain correction is deliberately a small-signal tangent model,
is shared between RR and LL, and is constant over the detected minute. It is a
diagnostic rather than a replacement calibration solution.

This evaluation is held out within the model-comparison stage, but it is not a
pristine sealed experiment. The leaf and minute were selected by the earlier
spatial search, which inspected other baseline partitions of the same native
fixture. The consistency of the antenna terms and the new four preceding
minutes are strong evidence, but the quoted percentages should not be treated
as an unbiased discovery significance.

The next implementation should turn this diagnostic into a constrained,
time-local calibration model. Candidate gain knots or change points should be
chosen on discovery baselines, ridge and duration on selection baselines, and
accepted only when they improve held-out visibilities and pass transient
injection/recovery tests. RR and LL should then be allowed separate antenna
terms. Once the corrected frozen-sky residual is available, the existing-flag
audit and conservative recovery policy should be rerun before training a
flagging classifier.

The reusable sufficient-statistics fitter is in
`src/sl1mjax/residual_models.py`. The complete-scan driver is
`scripts/diagnose_3c391_scan_residual.py`. Per-partition JSON and plots are in
`outputs/3c391_scan29_residual_diagnostic/`.

### Paraxial \(w\)-term error

\(|w|\) on the fixture reaches \(1.3\times 10^4\) wavelengths. Missing
\(e^{2\pi i w(n-1)}\) phase is \(\propto w r^2\):

| location | radius | max \(\lvert\Delta\phi\rvert\) | RMS |
|---|---|---|---|
| 128 remnant | 2.1′ | 0.016 rad | 0.004 |
| 128 corner | 6.0′ | 0.125 rad | 0.034 |
| 256 south rim | 8.8′ | 0.27 rad | 0.073 |
| 256 corner | 12.0′ | 0.50 rad | 0.14 |

This is worst on the rim and can dump some unmodeled wide-field residual
there. It is too small on the 128 field (~7° at the corner) to explain
75–94% boundary splits. The same rim mode appears in regular-grid
`square-paraxial` reconstructions that never run Haar.

Hierarchical imaging originally defaulted to paraxial and treated shared
Haar curvature as exact. That is now wide-field, with shared curvature
treated as exact only for paraxial squares and no beam (`c16c109`).

### Haar child order

`children()` is index-raster (sky SE, SW, NE, NW). The proposal’s Haar
matrix assumes celestial NW, NE, SW, SE. The old zip was a 180° relabel:
\(h_x \to -h_x\), \(h_y \to -h_y\), \(h_d\) unchanged. The three columns
still span the same mean-zero 4-space, so \(S_p = \tfrac12 q^T G^{-1} q\)
is invariant. That mismatch did not rerank splits. Scoring now uses
`QuadtreeLeaf.haar_children()`.

## Ruled out

- **FFT wrap / aliasing.** The operator is a direct DFT.
- **Quadtree vs regular-grid indexing.** Centres agree.
- **A compact source just outside the 8.5′ box.** The spike moved with the
  border at 12.8′ and 17.1′.
- **“Just cover the primary beam.”** The 192/256 windows already overshoot
  the FWHM; corners got brighter.
- **Haar label order.** Score-invariant 180° relabel.
- **Paraxial default as the main cause on 128.** Contributes at the rim;
  does not create the mode.
- **The 1,024-row subsample as the primary cause.** The full-row 8×8 split
  shows the same edge behaviour, and the full-row 64×64 control suppresses it.
- **A unique full-data NW--SE eigenmode.** Changing only the coarse split seed
  moves the hot pixel to another corner. Fitting all rows removes the corner
  peak on the diagnostic subset.

## What will not fix it by itself

- A wider FOV.
- A primary beam in the predict, unless L1 is also strong enough to refuse
  the first cheap ring inside the beam.
- Tightening `min_eigenvalue_ratio` while curvature is shared across the
  level (every leaf has the same ratio).
- `min_parent_flux` as a rim gate: the NW corner already has ~0.009 Jy,
  the same as the remnant.
- A very small initial intensity. It suppresses the corner by leaving much of
  the image unable to grow through softplus, but the resulting fit has large
  physical KKT violations.

## Recommended fixes and next experiments

1. **Make UV validation cells field-aware.** Choose cell widths in wavelengths
   from the image field, with $\Delta u \lesssim 1/\mathrm{FOV}_l$ and
   $\Delta v \lesssim 1/\mathrm{FOV}_m$. For this case, about 75×54 cells are
   indicated; 64×64 already works much better than 8×8. The implementation
   should support separate axis counts or physical cell widths.
2. **Target a sample or weight fraction, not a cell-count fraction.** Occupied
   cells have unequal populations. On the subset, asking for 20% of cells
   holds only 11--17% of rows depending on cell size. Select groups until the
   requested active weight is reached, while preserving radial and angular UV
   coverage in training.
3. **Use the implemented physical-flux solvers.** FISTA, proximal SGD, and the
   hybrid now enforce zero flux exactly and report a physical KKT residual.
   Keep softplus Adam only as a regression control. For large datasets, use
   SGD for fast approach and a short deterministic finish when an exact
   full-gradient pass remains feasible.
4. **Separate validation from the final scientific fit.** This is now done for
   the seven-pointing proof of concept: validation chose the topology, then a
   fixed tree was fitted on every active visibility. The all-data fit also
   showed that a 500-step cap is not always enough for the physical KKT gate.
5. **Use cross-fold agreement for topology decisions.** A genuine remnant
   detail should score under several balanced UV folds. A split supported only
   by one fold's boundary interpolant should not be accepted. This is more
   expensive but directly addresses topology selection leakage.
6. **Keep a rim gate only as a safety policy.** Refusing boundary splits can
   protect a production run, but it hides the validation/solver failure and
   should not be treated as the root fix.
7. **Stream stochastic and deterministic blocks from host storage.** The
   current row batches reduce operator work and device intermediates, but the
   canonical block is still held as one host object and JAX full-gradient
   calls receive a compact whole fold. Multi-scan and full-mosaic imaging will
   need a block iterator plus accumulated exact gradients/KKT checks.

Refinement now also has a resolution-aware hard ceiling. A Gaussian beam
estimated from the weighted UV second moment is 20.4 by 16.9 arcsec for C1 and
20.7 by 16.8 arcsec for the complete mosaic. With the default maximum of five
pixels across the minor beam, 16 arcsec root pixels may split to 4 arcsec but
not 2 arcsec. The configured maximum depth remains a second ceiling. This
calculation uses only UV coordinates, flags, and weights from the training
partition; Haar conditioning and held-out loss still control actual splits.

Starting with a 16² or 32² hierarchy and adding a broad component may still
help conditioning. Per-leaf wide-field Gram matrices may improve Haar ranking.
Neither addresses the demonstrated cause as directly as fixing validation
scale and the fixed-topology optimizer.

## Artifacts

- `outputs/wider_fov_comparison.json`
- `outputs/3c391_{128,192,256}_square-paraxial/`
- `outputs/beam_fov_comparison.json`
- `outputs/3c391_{128,256}_square-paraxial_gaussian-pb/`
- CASA residual comparison under `outputs/3c391_casa_clean_visibility_score/`
- Sealed C1 and C2 evaluations under
  `outputs/3c391_{,c2_}hierarchical_frozen_protocol/`
- Reproducible sealed evaluation: `scripts/evaluate_3c391_frozen_protocol.py`
- Frozen and CASA-style rendering: `scripts/plot_3c391_frozen_protocol.py`
- Joint seven-pointing products: `outputs/3c391_mosaic_joint_fista/`
- Joint mosaic fit and rendering: `scripts/image_3c391_mosaic.py` and
  `scripts/plot_3c391_mosaic.py`
- Joint mosaic topology discovery: `scripts/refine_3c391_mosaic.py`
- Repeated-fold mosaic selection:
  `outputs/3c391_mosaic_hierarchical_frozen_104/`
- All-data fixed-topology fit and common-grid residual products:
  `outputs/3c391_mosaic_hierarchical_consensus_all/`
- Reproducible final refit and residual imaging:
  `scripts/refit_3c391_mosaic_consensus.py` and
  `scripts/diagnose_3c391_mosaic_residuals.py`
- Leakage-safe residual and UV audit:
  `scripts/audit_3c391_residual_flags.py` and
  `outputs/3c391_residual_flag_audit/`
- Existing-flag false-positive audit:
  `scripts/audit_3c391_existing_flags.py` and
  `outputs/3c391_existing_flag_audit/`
- Post-composite residual-tail comparison:
  `scripts/compare_3c391_composite_residual_flags.py` and
  `outputs/3c391_composite_residual_flag_audit_stage2/`
- Post-composite existing-flag comparison:
  `scripts/compare_3c391_composite_existing_flags.py` and
  `outputs/3c391_composite_existing_flag_audit/`
- Short-baseline, beam-support, and time-partition study:
  `scripts/study_3c391_short_baselines.py` and
  `outputs/3c391_short_baseline_study_{stage1,sensitivity,interleaved,interleaved_airy}/`
- Fixed-position catalogue-source controls:
  `outputs/3c391_catalog_source_{contiguous,interleaved}/`
- Low-complexity target self-calibration control:
  `scripts/study_3c391_time_half_selfcal.py` and
  `outputs/3c391_time_half_selfcal/`
- Complete scan-29 residual-model comparison:
  `scripts/diagnose_3c391_scan_residual.py` and
  `outputs/3c391_scan29_residual_diagnostic/`
- Fixed-composite-sky gain complexity sweep:
  `scripts/sweep_3c391_calibration_interpolation.py` and
  `outputs/3c391_calibration_composite_time_complexity/`
- Composite central/coarse/catalogue fit and sealed protocol:
  `src/sl1mjax/composite.py`, `scripts/fit_3c391_composite.py`, and
  `outputs/3c391_composite_frozen_stage1/`
- Hierarchical proposal: `docs/hierarchical_pixels_proposal.md`
- Reproducible optimizer/KKT control:
  `scripts/diagnose_3c391_corner_optimizer.py`
