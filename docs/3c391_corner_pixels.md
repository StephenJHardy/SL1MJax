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

A second problem makes the artifact much worse: the current softplus/Adam
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

C1 is one pointing of a seven-pointing mosaic. Neighbouring-field leakage
belongs outside this beam. A beam-weighted L1 will pull that leakage onto
high-\(B\) pixels (pedestal / remnant bias) if we let it.

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
3. **Replace or supplement softplus Adam with a solver in physical flux.** The
   fixed-topology problem is a convex nonnegative L1 quadratic. Projected or
   proximal gradient, FISTA, or a bounded quasi-Newton method can enforce zero
   flux exactly and expose a meaningful KKT stopping test. At minimum, record
   the projected physical gradient and do not call holdout patience
   “convergence.”
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
- Hierarchical proposal: `docs/hierarchical_pixels_proposal.md`
- Reproducible optimizer/KKT control:
  `scripts/diagnose_3c391_corner_optimizer.py`
