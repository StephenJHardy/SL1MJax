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
injected refit failure, and the full local suite has 254 passes and one
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
4. **Separate validation from the final scientific fit.** Use validation to
   choose L1 strength, topology budget, and stopping round. Then warm-refit the
   accepted topology on all active visibilities. The current diagnostic output
   is still the train-fold model.
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
- Hierarchical proposal: `docs/hierarchical_pixels_proposal.md`
- Reproducible optimizer/KKT control:
  `scripts/diagnose_3c391_corner_optimizer.py`
