# Hierarchical pixels for SL1MJax

## Executive recommendation

The first implementation should use a flux-conserving quadtree with a
data-space split score. A split should be proposed when the visibility residual
has a statistically meaningful projection onto the three new within-parent
degrees of freedom created by four children. Parent flux should be a safety
gate and an initializer, not the main ranking criterion.

The most useful score is a small child-replacement problem. For each candidate
parent, temporarily remove its visibility response and solve a four-variable
non-negative quadratic problem for its children. This gives the exact change in
the current linear data objective while all other leaves are fixed. A cheaper
Haar-contrast gradient can screen candidates before this calculation. After a
batch of splits, run the normal global optimizer and require improvement on
held-out UV cells. Use the reverse calculation for merging.

This is a better fit to SL1MJax than a generic image-gradient rule. The direct
operator already works with arbitrary component coordinates. Each coefficient
already denotes integrated flux. A parent can therefore be replaced by four
children without changing total flux. The current compound pixel kernel was
also designed with parent-to-four-child consistency as one of its objectives.

The recommended development order is:

1. Build the quadtree data model and flux-preserving split/merge operations.
2. Establish flux and image-derivative heuristics as baselines.
3. Implement the residual/Haar screening score and exact four-child lookahead.
4. Add held-out acceptance, merge hysteresis, and an explicit tree-complexity
   penalty.
5. Compare against a uniform fine grid and exhaustive split lookahead on small
   synthetic problems.
6. Only then consider a learned split policy, adaptive free-position
   components, or trans-dimensional sampling.

The deterministic split path is now implemented through held-out batch
acceptance. This includes a level-batched residual adjoint, per-level Haar
curvature, complexity-aware Dörfler marking, warm global refits, and prefix
backtracking. A 4,096-parent test evaluates 16,384 virtual children without a
per-candidate predictor loop. The end-to-end driver starts from thousands of
root pixels and writes a FITS rendering, topology table, score table, residuals,
checkpoint, and decision summary. Merge hysteresis and repeated-seed
validation remain to be implemented.

A six-case benchmark compares every policy on the same candidates. Haar
selects the oracle leaf in all four structured cases. The local solve has
higher mean rank correlation but misses one anisotropic case because it cannot
adjust the other fitted leaves. This supports Haar as a screen and local
lookahead as a useful but conditional estimate.

Under the assumptions listed below, a tested deterministic prototype is about
three to five weeks of work for one developer. A useful reversible-jump MCMC
prototype is more likely to take three to six months and would still need a
separate scaling programme.

## What exists now

SL1MJax fits both a positive regular Stokes-I grid and the positive leaf fluxes
of a fixed quadtree topology. The physical flux is obtained with a softplus
transform. The data term is a weighted mean squared complex residual, and the
sparsity term is the sum of positive integrated pixel fluxes. The exact direct
DFT streams both the forward operation and its adjoint without storing the full
visibility-by-pixel matrix. Quadtree topology changes remain outside the jitted
optimizer, so each fit has fixed array shapes.

Several details make an adaptive hierarchy practical:

- [`predict_stokes_i`](../src/sl1mjax/rime.py) and the explicit DFT accept
  arbitrary equal-length lists of fluxes, `l` coordinates, and `m`
  coordinates. The regular grid is imposed by the inference wrapper, not by
  the measurement equation.
- Pixel coefficients have units of integrated Jy per component. A
  flux-preserving split therefore has an unambiguous initialization.
- The compound basis in [`sky.py`](../src/sl1mjax/sky.py) is a normalized
  Gaussian mixture. The search in
  [`gaussian_kernel_search.py`](../scripts/gaussian_kernel_search.py) explicitly
  compares a coarse kernel with four half-scale children at quarter-cell
  offsets.
- Train/holdout UV-cell splits and residual dirty-image diagnostics already
  exist. They can test whether refinement explains unseen visibilities rather
  than training noise.
- `infer_quadtree` fits one fixed topology with either the autodiff or streamed
  explicit operator. It accepts physical-flux warm starts, evaluates the
  primary beam at leaf centres, returns the fitted visibility residual, and
  records an explicit per-leaf topology penalty for comparisons between trees.
- `baseline_split_scores` reports parent flux, surface brightness, and
  scale-normalized image gradient and Laplacian scores. The accompanying
  exhaustive oracle flux-conservingly splits each candidate and globally
  refits all leaves. It can reset Optax or enumerate active sets for an exact
  small-problem solution, and reports train-objective and holdout changes.
- `residual_haar_scores` projects the weighted training residual onto the
  three zero-sum child-detail responses. It normalizes the gradient with the
  local 3x3 Gauss--Newton matrix and reports the predicted data-objective
  reduction, eigenvalues, condition ratio, ridge, and eligibility gates.
  `compare_haar_to_oracle` then reports per-leaf ranks, top-choice agreement,
  and Spearman correlation over an identical candidate set.
- `batched_residual_haar_scores` obtains all child correlations with one
  streamed adjoint per active level. Its shared curvature is exact for the
  paraxial square basis without a primary beam. Wide-field or beam-weighted
  runs must opt into approximate curvature, and the marked shortlist is then
  rescored per parent before a topology change.
- `select_bulk_splits` applies Dörfler-style score-mass marking with a bound on
  splits per round. It can subtract the explicit three-leaf complexity cost
  before ranking.
- `refine_quadtree_batch` divides parent flux among children, globally refits
  every active flux, and requires both penalized training objective and
  held-out loss to improve. A rejected batch is halved to its strongest prefix
  for a bounded number of retries.
- `reconstruct_hierarchical` connects the initial solve, batched score,
  marking, exact shortlist rescore, refit, validation, and stopping rules.
- `local_four_child_lookahead` removes the actual fitted parent response and
  solves the resulting four-child non-negative quadratic while holding all
  other leaves fixed. It includes L1 and the three-leaf complexity increment.
  An optional equality constraint preserves parent flux and isolates spatial
  detail. The solver enumerates the 16 possible active child sets, so it needs
  no general-purpose optimization dependency and gives deterministic results.
- `solve_quadtree_flux_active_set` provides a small-problem validation fit that
  enumerates all active leaf sets. The exhaustive oracle can use this solver to
  remove Optax convergence error from score comparisons. Its exponential cost
  is guarded by a maximum leaf count and is not intended for production trees.
- The explicit adjoint can compute residual correlations for many virtual
  children in batches. This is the expensive part of a derivative-based split
  rule, but it is already the operation that the code is designed to stream.

There are also four constraints:

- The dense-grid smoothness term in `sky_prior` has no direct meaning for
  unequal-area leaves. Quadtree inference currently uses positive L1 flux and
  a leaf-count penalty only.
- A quadtree prediction is grouped by depth because one `pixel_size_rad`
  applies to every component in an operator call. Very deep trees therefore
  require several operator calls per objective evaluation.
- JAX recompiles when array shapes change. Topology changes should therefore
  occur between optimization epochs, not inside a jitted optimizer step.
- The current positive L1 term is total flux. It does **not** penalize a
  flux-conserving split. A separate penalty or validation rule must control
  tree complexity.

The compound kernel's refinement property is promising but not exact. A local
evaluation using the frozen four-Gaussian parameters and the search script's
default grid gives a relative parent/children refinement mismatch of about
`0.368`. That value is too large to assume that an equal split leaves the
prediction unchanged. The split scorer should include the actual parent and
child responses. Kernel consistency should also become a quantitative gate in
the hierarchy tests.

The tracked sibling `SL1M` checkout contains the original regular-grid CUDA
implementation and a later untracked JAX port. Its tracked history does not
contain the remembered hierarchical split code. The tracked `SL1MML` checkout
contains a dense pixel grid and trainable single-delta and single-Gaussian
components, including trainable positions and Gaussian width, but no tree.
The earlier experiment may have lived in another branch or local copy. The
2013 paper still establishes the important design point: SL1M pixels may have
arbitrary positions and individual Gaussian scales [Hardy 2013].

## The split question

Let the current leaf responses form a linear operator $A_T$, where $T$ is
the current tree. Let $x \geq 0$ contain integrated leaf fluxes and let

\[
r = A_T x-y
\]

be the complex visibility residual. With active weights $W$ and their sum
$Z$, the present data objective is

\[
D(x,T)=\frac{1}{Z}r^HWr.
\]

The derivative with respect to the *physical flux* of a candidate atom
$a_j$ is

\[
g_j=\frac{2}{Z}\operatorname{Re}(a_j^HWr).
\]

For a new non-negative atom under an L1 penalty, the relevant reduced gradient
is $g_j+\lambda$. A negative value means that adding positive flux can lower
the objective. This is the same matched-residual quantity that appears in
CLEAN stopping conditions and sparse active-set methods. It directly asks
whether the measured data require the missing basis function.

The raw Optax gradient should not be used for this decision. If raw parameter
$z$ maps to flux $x=\operatorname{softplus}(z)$, then

\[
\frac{\partial J}{\partial z}
=\operatorname{sigmoid}(z)\frac{\partial J}{\partial x}.
\]

The sigmoid factor makes the score depend on the arbitrary current raw value.
Split scores should be evaluated in physical-flux coordinates.

### Why parent strength is not enough

A strength rule is simple: split leaves whose flux or peak surface brightness
exceeds a threshold. It will often find bright compact sources. It will also
split a bright smooth plateau even when the data contain no information about
its substructure. It can miss a faint sharp edge or a blended pair whose
combined parent flux is unremarkable.

An image derivative is better aligned with morphology. A gradient finds edges,
and a Laplacian or Hessian finds curvature and unresolved peaks. However, these
quantities are derivatives of the current reconstruction. They do not account
for UV coverage, weights, PSF sidelobes, primary-beam attenuation, or noise.
They are useful prefilters but weak final decision rules.

The useful derivative is the objective derivative in the *new child contrast
directions*. At a converged solution, the derivative with respect to the
existing parent amplitude is already near its constrained optimum. That says
nothing about the three spatial degrees of freedom that do not yet exist.

### Four children create three resolution degrees of freedom

Order four child fluxes as north-west, north-east, south-west, and south-east.
Their equal-flux direction controls total parent flux. Three Haar-like contrast
directions control left-right, north-south, and diagonal structure:

\[
h_x=(1,-1,1,-1),\quad
h_y=(1,1,-1,-1),\quad
h_d=(1,-1,-1,1).
\]

Let $C_p$ contain the four child visibility responses. The virtual detail
responses are $C_p h_x$, $C_p h_y$, and $C_p h_d$. Correlating them with
the weighted residual measures the information that the parent cannot express.
This is the quadtree analogue of a wavelet hierarchical surplus.

A curvature-normalized score is more reliable than gradient magnitude alone.
If $q_p$ is the three-element detail gradient and $G_p$ is the local
three-by-three weighted Gram or Gauss-Newton matrix, use

\[
S_p=\frac{1}{2}q_p^T(G_p+\epsilon I)^{-1}q_p.
\]

This estimates the objective reduction available from the new detail space.
It also suppresses apparently strong directions that the sampled baselines
cannot distinguish. Very small eigenvalues of $G_p$ identify unsupported
super-resolution.

### Recommended score: exact local child replacement

The Haar score is an efficient screen. The final proposal score should handle
the real kernel mismatch and positivity. For a parent response $a_p$, parent
flux $x_p$, and four child responses $C_p$, solve

\[
\min_{c\geq 0}
\quad \frac{1}{Z}\left\|W^{1/2}
\left(r-a_px_p+C_pc\right)\right\|_2^2
+\lambda(\mathbf{1}^Tc-x_p)+3\beta.
\]

Here $c$ contains the four child fluxes. The term $3\beta$ is the cost of
the three additional parameters. Subtract the current objective to obtain the
conditional split improvement. The problem has only four variables. Because
the visibility model is linear in flux, its data term is an exact quadratic
when other leaves are fixed. It can be solved as a small non-negative least
squares or bounded quadratic problem.

Two versions are useful:

- Constrain $\mathbf{1}^Tc=x_p$ to measure demand for spatial detail alone.
  The positive L1 term then cancels exactly.
- Allow total child flux to vary to estimate the complete local replacement.
  Keep the L1 and complexity terms in this score.

Both versions are implemented by `local_four_child_lookahead`. The complete
replacement is the default. `conserve_parent_flux=True` selects the simplex
version. The result includes the four non-negative child fluxes, train and
optional holdout changes, objective terms, and a deterministic ranking. The
companion `compare_lookahead_to_oracle` function measures rank correlation and
top-choice agreement against exhaustive global refits.

The constrained score answers the scientific split question more cleanly. The
unconstrained score is a better predictor of the next optimized objective. A
candidate should normally rank well under both.

## Proposed adaptive algorithm

Use the standard adaptive loop of solve, estimate, mark, refine, and repeat.
Residual-based adaptive finite-element methods and adaptive inverse
parameterizations use this structure because it separates continuous fitting
from discrete model changes [Bangerth 2008; Kaltenbacher & Offtermatt 2011].

1. **Solve.** Optimize fluxes on the current leaves until the training
   objective and held-out loss have stabilized.
2. **Estimate.** Compute physical-flux child-contrast gradients for every
   splittable leaf. Reject candidates with negligible flux, insufficient local
   Fisher information, or maximum depth.
3. **Screen.** Rank candidates with the curvature-normalized Haar score.
4. **Look ahead.** Run the exact four-child replacement solve for the leading
   candidates.
5. **Mark.** Select the smallest set that accounts for a chosen fraction of
   total positive predicted improvement. This is the useful idea behind
   Dörfler bulk marking. Also cap growth, for example at 10--25% of the current
   leaf count per round.
6. **Refine.** Replace each selected parent with four children. Initialize each
   child with one quarter of parent flux or with the local lookahead solution.
7. **Re-optimize.** Reset the optimizer for the first implementation. Reusing
   Adam moments across a softplus topology change is possible, but it is not a
   scientifically important first optimization.
8. **Validate.** Keep the batch only when held-out UV-cell loss, residual
   structure, or a stated information criterion improves beyond tolerance.
9. **Coarsen.** Test complete sibling groups with the reverse local problem.
   Merge only after the merge remains acceptable for two rounds. This
   hysteresis prevents split/merge oscillation.
10. **Stop.** Stop when no batch passes validation, the residual is consistent
    with its noise model, the leaf budget is reached, or all remaining detail
    directions are unidentifiable.

Do not accept individual splits solely on a tiny change in global held-out
MSE. One local change can be lost in a large visibility average. Use the
training score for ranking, accept a bounded batch, and use holdout data for
the batch-level decision. Report both.

### Controlling over-refinement

Positive L1 regularization is total flux. If a parent of flux $F$ becomes
four positive children whose fluxes sum to $F$, the L1 value remains $F$.
The current regularizer is therefore neutral to topology. Without another
control, a tree can keep fitting finer noise structure.

Three controls are worth testing:

- **Leaf-count penalty.** Add $\beta |T|$, or equivalently charge $3\beta$
  per quadtree split. This is simple and makes the local score explicit.
- **Hierarchical detail penalty.** Penalize the three Haar details at every
  active internal node. A group L2 penalty chooses whether a node has any
  detail. An L1 detail penalty permits anisotropic structure. This links
  refinement and regularization directly.
- **Tree prior.** Use a depth-dependent split probability such as
  $p_{\rm split}(d)=\alpha(1+d)^{-\gamma}$. Its negative log probability is a
  deterministic complexity penalty now and a genuine prior in later Bayesian
  inference.

The first prototype should combine a leaf-count penalty with held-out UV-cell
selection. A hierarchical detail penalty is the strongest second experiment.

## Alternative approaches and feasibility

The time estimates assume one developer, the existing direct operator, a
synthetic test harness, and no production-scale NUFFT work in the same tranche.
They include tests and diagnostic output, not only a demonstration notebook.

| Approach | Main decision signal | Expected quality | Compute | Prototype effort | Assessment |
|---|---|---:|---:|---:|---|
| Flux threshold | Parent integrated flux or brightness | Low to moderate | Very low | 1--2 days | Useful baseline only |
| Image gradient or Hessian | Local edge/curvature of current image | Moderate | Low | 3--5 days | Good prefilter; not data-aware enough alone |
| Residual/Haar derivative | Weighted residual projected onto child details | High | One batched virtual-child adjoint per round | 1--2 weeks | Recommended first scientific criterion |
| Exact four-child lookahead | Conditional objective reduction with positivity | High | Four-atom Gram and residual products per candidate | Additional 1--2 weeks | Recommended acceptance score |
| Full split-and-refit oracle | Global re-optimization after every candidate split | Very high on small tests | Very high | 2--4 weeks for a test oracle | Use to validate cheaper scores, not at scale |
| Hierarchical Haar model | Sparse scaling and detail coefficients on a tree | High for mixed smooth/sharp skies | Moderate | 4--8 weeks | Strong second-generation representation |
| Adaptive Gaussian/Asp components | Fit amplitude, location, and continuous scale | High for compact and resolved blobs | Moderate to high; non-convex fit | 4--8 weeks | Valuable alternative to literal pixels |
| Sliding Frank-Wolfe / gridless atoms | Add a continuous atom, then move all active atoms | High for point-like sparse skies | Moderate per atom; non-convex sliding | 6--10 weeks | Excellent point-source branch; weaker for diffuse fields without a mixed model |
| ARD or evidence maximization | Marginal likelihood and posterior coefficient variance | Potentially high | Requires repeated linear solves or covariance approximations | 6--12 weeks | Useful route to approximate uncertainty and automatic pruning |
| Differentiable split gates | Continuous gates over a pre-built maximum tree | Uncertain | High memory; difficult optimization | 6--10 weeks | JAX-friendly shapes, but many nearly duplicate atoms and fragile gates |
| Learned split policy | Supervised score or RL policy from local state | Unknown until trained | Cheap inference; expensive data generation | 2--4 months after an oracle exists | Learn the heuristic later, not the physics now |
| RJMCMC split/merge | Posterior probability over topology and flux | Principled uncertainty | Very high and difficult to mix | 3--6 months for a bounded prototype | Credible long-term research path |

### Lessons from radio imaging

Adaptive Scale Pixel CLEAN is the closest direct precedent. It argues that
spatial correlation length separates signal from noise better than amplitude
alone. It searches residual images smoothed at several scales, adds an adaptive
Gaussian component, and refits an active set [Bhatnagar & Cornwell 2004]. The
active set was itself screened using first derivatives. Later work retained
adaptive scale fitting while reducing its expensive convolution work, with a
reported order-of-magnitude runtime improvement in its tests [Zhang et al.
2016].

This supports two choices here. First, strength alone is insufficient. Second,
previous components must be allowed to change after a structural proposal.
SL1MJax can improve on the image-plane heuristic by evaluating the exact
ungridded visibility objective and its derivatives.

Multi-scale CLEAN and sparse wavelet methods use a fixed dictionary of scales.
They avoid topology management but spend coefficients on a redundant set of
atoms. SARA and related radio methods show that averaging sparsity over several
wavelet bases can represent both point-like and extended emission well
[Carrillo et al. 2012]. A hierarchical Haar or wavelet representation is
therefore a serious alternative if literal leaf geometry becomes awkward.

### Lessons from adaptive inverse problems

Adaptive inverse methods commonly use residual or adjoint information because
large model values do not imply large discretization error. Work on adaptive
parameterization has explicit refinement and coarsening indicators and reports
fewer unknowns without loss of inversion accuracy [Maina et al. 2017]. A
quadtree basis has also been used to refine image representations independently
of the forward mesh [Schweiger et al. 2016].

The relevant concept is hierarchical surplus: measure what becomes expressible
at the next level but is absent at the current one. The three child contrasts
provide exactly that quantity for a quadtree. Dörfler marking then gives a
practical way to refine a controlled batch instead of setting an unstable
absolute threshold.

### Avoiding the grid rather than refining it

For a sparse point-source sky, the hierarchy may be solving the wrong problem.
Off-grid methods treat the image as a positive measure with continuous source
positions. Sliding Frank-Wolfe alternates between adding a new atom from a
residual correlation oracle and jointly moving active atoms [Denoyelle et al.
2020]. This directly addresses the current off-grid model error.

A mixed model is likely to be strongest in the long run:

- continuous delta or compact Gaussian components for unresolved sources;
- hierarchical compound or wavelet leaves for diffuse emission;
- a shared visibility residual and objective for both.

This is more complex than the first quadtree, but it avoids forcing a point
source through many fine leaf splits.

### Learned refinement

Learned AMR policies can outperform or match hand-built marking rules on some
simulation tasks and can generalize to larger meshes [Yang et al. 2023]. That
does not remove the need for a reliable split objective. In this project, the
exact local lookahead can generate labels for a small model. Candidate features
could include flux, depth, Haar gradient, Gram eigenvalues, local residual
statistics, primary-beam response, and neighboring levels.

A supervised ranker is lower risk than reinforcement learning. Keep the
physics score and holdout test as acceptance gates. Consider RL only if the
greedy sequence has measurable long-horizon regret, such as early splits that
block a much more compact later tree.

## Representation and JAX implementation

### Tree state

Represent a leaf with integer `(level, iy, ix)` coordinates relative to a
fixed root grid. Derive its center, width, area, parent, and four child IDs from
those integers. Keep leaves in canonical Morton order. This makes topology
deterministic and checkpointable.

Each fitted leaf stores integrated flux. Rendering should be separate from
inference. A dense diagnostic image can integrate each leaf kernel onto a
chosen regular output grid. The output must state whether it is integrated Jy
per render pixel or surface brightness. Directly placing unequal-area leaf
fluxes into a FITS array would make `Jy/pixel` ambiguous.

### Multi-width operator

The least invasive implementation groups leaves by depth. All leaves in one
group share a width, so the existing operator can run once per level and the
predictions can be summed. This also keeps `_operator_factory` caching useful.
It avoids immediately changing all scalar width assumptions to vector widths.

A later kernel can accept a per-component width vector. The Gaussian algebra
already broadcasts naturally over pixels, but validation, cache keys, tiling,
and custom-VJP tests must be updated carefully.

Coarse Gaussian leaves assume that the primary beam and projection terms vary
little across one component. The root width must respect that approximation.
Alternatively, integrate the beam over coarse leaves or force refinement where
beam variation exceeds a stated tolerance.

### Inference boundary

Add a component-level inference function rather than complicating
`infer_regular_grid`. It should accept coordinates, level groups, initial
physical fluxes, and a graph or tree prior. The outer hierarchy driver should
own topology changes. Each fixed-topology optimization remains jitted and
differentiable.

Resetting the Optax state after a topology change is the safe first choice.
Initialize a split in physical space, then use `raw_from_intensity`. Warm-start
all unchanged fluxes exactly. Compilation cost should be amortized over the
many optimizer steps in each refinement epoch. If it is not, pad each level to
a bounded capacity and use an active mask.

### Complexity

With $V$ visibility samples and $P$ active leaves, a fit remains
$O(VP)$ under the direct operator. Scoring $C$ candidate parents requires
responses for at most $4C$ virtual children. Batch this work through the
existing tiled adjoint. The hierarchy wins when the selected leaf count is far
below a uniform fine grid.

The planned scalable gridded or NUFFT operator remains important. A compatible
future design can group dyadic levels, render sparse coefficients to one grid
per level, convolve with that level's kernel, and sum predicted visibilities.
The initial direct implementation should keep this separation visible.

## Validation programme

### Synthetic scenes

Use known-truth scenes that separate the candidate criteria:

- a bright smooth Gaussian, which a flux rule will tend to over-split;
- a faint sharp edge or thin ring, which tests derivative sensitivity;
- one and two off-grid point sources, including a pair below one coarse cell;
- mixed point and diffuse emission over a wide dynamic range;
- uniform surface brightness, where only boundaries should need detail;
- a noise-only field, where no split should survive validation;
- primary-beam attenuated sources at several radii;
- identical skies under sparse, anisotropic, and dense UV coverage;
- several thermal-noise realizations and weight scales.

Compare at least four policies: flux, image Hessian, residual/Haar, and exact
lookahead. Include a uniform fine grid and a fixed multiscale dictionary.

### Score validation

On 8x8 or 16x16 problems, enumerate every legal single split. Fully re-optimize
after each split. This produces an expensive oracle. Measure rank correlation
between each cheap score and true held-out improvement. Also measure regret:
the objective gap between the split chosen by a score and the best split.

This test is more informative than showing one attractive adaptive image. It
directly answers whether the criterion chooses the right pixel.

### Current synthetic benchmark

[`benchmark_hierarchical_refinement.py`](../scripts/benchmark_hierarchical_refinement.py)
runs diagonal, horizontal, faint, anisotropic-coverage, smooth-control, and
noise-control cases. It writes `summary.json`, `candidates.csv`,
`policy_summary.csv`, and `aggregate.csv`. Each candidate row contains the raw
scores, ranks, oracle train and holdout changes, Haar conditioning, and local
child fluxes. Run it with:

```console
uv run python scripts/benchmark_hierarchical_refinement.py
```

The initial run uses one seed and four root candidates per case. The four
structured cases all have a positive exact-oracle split. Results against that
training oracle are:

| Policy | Correct top leaf | Mean Spearman rho | Mean regret | Correct holdout top leaf |
|---|---:|---:|---:|---:|
| Parent flux | 0/4 | 0.35 | 1.99e-3 | 0/4 |
| Image gradient | 0/4 | 0.30 | 1.99e-3 | 0/4 |
| Image Laplacian | 0/4 | 0.15 | 1.99e-3 | 0/4 |
| Residual/Haar | 4/4 | 0.70 | 0 | 4/4 |
| Local child solve | 3/4 | 0.90 | 3.08e-4 | 3/4 |

Regret is the normalized training-objective improvement lost relative to the
best exhaustive split. The local solve's only miss is the anisotropic case. It
ranks the correct leaf second because the frozen other leaves cannot absorb a
strong correlated residual. The global oracle can readjust them. Haar still
selects the correct leaf because its zero-sum detail directions are less
sensitive to this flux coupling.

The exact oracle rejects the smooth and noise controls on training data. The
noise control still has a tiny positive held-out change of about `9e-8`, which
is small enough to be sampling noise. A held-out acceptance rule therefore
needs a practical effect-size threshold or repeated-split uncertainty, not only
the sign of one holdout difference. These results are a development baseline,
not a statistical claim: the next sweep should add seeds, noise levels, primary
beam offsets, and larger candidate grids.

### Real-data readiness checkpoint

[`image_3c391_hierarchical.py`](../scripts/image_3c391_hierarchical.py) runs the
adaptive path on one block of the portable 3C391 fixture. The default field is
128 by 128 root pixels at 4 arcsec, so the initial model has 16,384 fitted
fluxes. The command uses a 1,000-step ceiling with held-out early stopping.
Shorter fixed limits produced false split acceptance because the child fit was
also an ordinary continuation of an under-solved parent fit.

A 1,024-row GPU convergence audit stopped the unchanged parent topology after
380 steps, with the best held-out point at step 280. Starting refinement from
that fit accepted 32 splits and reduced held-out loss from `0.3487` to `0.2722`.
This is not yet a scientific validation. Seventy-five per cent of the selected
parents touched the image boundary, while the established full-data regular
reconstruction peaks in the interior. The quadtree and regular-grid coordinate
arrays agree exactly, so an indexing mismatch has been ruled out. The current
leading explanation is the deliberately sparse 1,024-row subset. Further
refinement is paused pending either a same-subset regular-grid control or a run
on all 20,542 rows.

The implementation checkpoint passes Ruff, strict mypy, and 214 tests; one
real-VLA release test remains opt-in. The focused CUDA run also passes all 32
quadtree and refinement tests on an RTX 3080 Ti.

### Scientific and engineering metrics

Report:

- train and UV-cell holdout weighted complex MSE;
- residual dirty-image peak, robust RMS, and spatial correlation;
- image error after rendering every method to a common grid and resolution;
- recovered total flux, component flux, and astrometric error;
- leaf count by level and false split rate in noise-only regions;
- stability of topology across seeds and noise realizations;
- smallest child-detail Gram eigenvalue for accepted splits;
- forward, adjoint, compile, scoring, and optimization time;
- peak device memory and visibility-leaf products evaluated.

An initial success gate could require the adaptive result to match the uniform
fine-grid holdout loss within 2%, use at most 25% of its active leaves, and
reduce end-to-end runtime. These numbers are engineering targets, not claims
about a universal scientific optimum. They should be revised after the first
synthetic sweep.

## Route to uncertainty and temporal skies

### Near-term uncertainty

Full trans-dimensional MCMC is not required to obtain the first uncertainty
signals. Cheaper stages are:

1. bootstrap or jackknife the visibility blocks and record split frequency;
2. compute a Laplace or Gauss-Newton covariance on the selected leaves;
3. estimate posterior variances with matrix-free solves and randomized
   diagonals;
4. use ARD or a hierarchical Bayesian sparsity model to prune weak details and
   infer noise or regularization scales.

Sparse Bayesian learning can select a small set of relevant basis functions and
provide probabilistic predictions [Tipping 2001]. Hierarchical Laplace models
were already identified in the original SL1M paper as a possible route to
automatic sparsity and covariance estimation [Babacan et al. 2010]. In radio
imaging, both information-field methods and proximal MCMC have been used for
uncertainty quantification, but their computational cost is substantial
[Arras et al. 2018; Cai et al. 2018].

### Reversible-jump model

A later Bayesian state could contain

\[
\{T, x_T, \lambda, \beta, \sigma_{\rm noise},\ldots\}.
\]

Useful proposals are split/merge, local flux redistribution, component
birth/death, and position/scale moves for continuous components. A split can
map parent flux $F$ to four positive proportions drawn from a Dirichlet
distribution. The reverse merge sums them. The Metropolis-Hastings ratio must
include proposal probabilities and the transformation Jacobian. This is an
application of reversible-jump MCMC, which was designed for parameter spaces
whose dimension changes [Green 1995]. Trans-dimensional Voronoi inversions in
geophysics show the intended outcome: local model resolution adapts to the
data, and averaging over sampled discretizations gives both a posterior field
and resolution uncertainty [Hawkins et al. 2019].

The linear visibility model offers one major optimization. Cache the current
predicted visibilities. A split proposal then needs only the removed parent and
four added child responses, not a new full $A_Tx_T$. This reduces a topology
proposal from $O(VP)$ to roughly $O(5V)$. Within-topology updates can use
gradient methods, HMC, or conditional proposals.

The hard parts remain posterior multimodality, correlated nearby leaves,
calibration uncertainty, and chain mixing. Parallel tempering or sequential
Monte Carlo may be more effective than one RJMCMC chain. The deterministic
split engine should seed topologies and informed proposals.

### Temporal extension

Independent trees per time bin will flicker. A temporal model should use a
shared union tree with time-dependent flux or detail coefficients. Aggregate
split evidence across time with a group norm, then allow a leaf to refine when
the sequence supports its extra spatial degrees of freedom. Penalize changes
in topology more strongly than changes in flux.

A simple first dynamic prior is a Gaussian random walk on transformed flux or
Haar details. Motion-aware evolution can come later. StarWarps demonstrates
that a Gaussian Markov model can propagate information between sparse
interferometric observations while learning source dynamics [Bouman et al.
2017]. In a Bayesian tree model, posterior split probability can be shared over
time while coefficients evolve. This turns the hierarchy into a spatial
resolution model rather than a frame-by-frame detection mask.

## Concrete first tranche

The first tranche should deliver the following artifacts:

- `HierarchicalGrid` or `QuadtreeSky` with deterministic IDs and split/merge;
- per-level grouped prediction with delta, Gaussian, and compound bases;
- exact flux conservation and parent/child response-consistency tests;
- a component inference path with L1 and leaf-count penalties;
- physical-flux residual/Haar scores;
- exact four-child and four-sibling local lookahead solvers;
- batch marking, held-out acceptance, merge hysteresis, and stopping rules;
- synthetic exhaustive-oracle experiments and a machine-readable score table;
- dense FITS rendering with explicit units and topology diagnostics;
- a bounded 3C391 comparison only after the synthetic gates pass.

The key experiment is not whether a tree can make a plausible image. It is
whether residual/Haar or local lookahead selects nearly the same splits as
exhaustive re-optimization, and whether those splits recover fine-grid quality
with materially fewer active components.

## References

- Arras, P. et al. (2018), [Radio Imaging With Information Field
  Theory](https://arxiv.org/abs/1803.02174).
- Babacan, S. D., Molina, R., & Katsaggelos, A. K. (2010), [Bayesian
  Compressive Sensing Using Laplace
  Priors](https://doi.org/10.1109/TIP.2009.2032894).
- Bangerth, W. (2008), [A Framework for the Adaptive Finite Element Solution of
  Large-Scale Inverse Problems](https://doi.org/10.1137/070690560).
- Bhatnagar, S. & Cornwell, T. J. (2004), [Scale sensitive deconvolution of
  interferometric images I. Adaptive Scale Pixel
  decomposition](https://doi.org/10.1051/0004-6361:20040354).
- Bouman, K. L. et al. (2017), [Reconstructing Video from Interferometric
  Measurements of Time-Varying Sources](https://arxiv.org/abs/1711.01357).
- Cai, X. et al. (2018), [Uncertainty quantification for radio interferometric
  imaging -- I. Proximal MCMC
  methods](https://doi.org/10.1093/mnras/sty2004).
- Carrillo, R. E., McEwen, J. D., & Wiaux, Y. (2012), [Sparsity Averaging
  Reweighted Analysis: a novel algorithm for radio-interferometric
  imaging](https://doi.org/10.1111/j.1365-2966.2012.21605.x).
- Denoyelle, Q. et al. (2020), [The Sliding Frank-Wolfe Algorithm and its
  Application to Super-Resolution
  Microscopy](https://doi.org/10.1088/1361-6420/ab2a29).
- Green, P. J. (1995), [Reversible jump Markov chain Monte Carlo computation
  and Bayesian model determination](https://doi.org/10.1093/biomet/82.4.711).
- Hardy, S. J. (2013), [Direct deconvolution of radio synthesis images using L1
  minimisation](https://arxiv.org/abs/1310.2078).
- Hawkins, R. et al. (2019), [Trans-Dimensional Surface Reconstruction With
  Different Classes of
  Parameterization](https://doi.org/10.1029/2018GC008022).
- Kaltenbacher, B. & Offtermatt, J. (2011), [A refinement and coarsening
  indicator algorithm for finding sparse solutions of inverse
  problems](https://doi.org/10.3934/ipi.2011.5.391).
- Kreuzer, C. & Siebert, K. G. (2011), [Decay rates of adaptive finite elements
  with Dörfler marking](https://doi.org/10.1007/s00211-010-0324-5).
- Maina, F. Z. & Ackerer, P. (2017), [Groundwater flow parameter
  estimation using refinement and coarsening indicators for adaptive
  downscaling parameterization](https://doi.org/10.1016/j.advwatres.2016.12.013).
- Schweiger, M. et al. (2016), [Basis mapping methods for forward and inverse
  problems](https://doi.org/10.1002/nme.5271).
- Tipping, M. E. (2001), [Sparse Bayesian Learning and the Relevance Vector
  Machine](https://www.jmlr.org/papers/v1/tipping01a.html).
- Yang, J. et al. (2023), [Reinforcement Learning for Adaptive Mesh
  Refinement](https://proceedings.mlr.press/v206/yang23e.html).
- Zhang, L. et al. (2016), [Efficient implementation of the adaptive scale
  pixel decomposition algorithm](https://arxiv.org/abs/1606.07872).
