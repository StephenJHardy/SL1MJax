# Visibility-validation and proximal radio imaging landscape

## Purpose

SL1MJax treats imaging as predictive model construction. A proposed change to
the sky, beam, polarisation model, or calibration model is accepted only when
it improves prediction of visibilities that were not used to fit that change.

This note places that design in the existing radio-interferometric imaging
literature. It addresses two questions:

1. Do other imaging systems use held-out visibilities in the machine-learning
   sense of testing generalisation?
2. Is the use of stochastic proximal optimisation followed by FISTA an
   established method, or a new invention?

The short answer is that all of the main ingredients have precedents. The exact
combination used by SL1MJax appears unusual. In particular, this survey did not
find another radio imager that uses grouped held-out visibilities as the
acceptance test for each local change to an adaptive sky hierarchy, and then
extends the same rule to polarisation and calibration freedom.

This is a technical landscape note, not a systematic literature review. It
records the sources found in an online search completed in September 2026.

## The SL1MJax starting point is SL1M

Hardy (2013) should be treated as the direct algorithmic ancestor of the
current work, not as a peripheral reference. This is a statement about the
SL1MJax project lineage. It is not a claim that Hardy (2013) was the first use
of FISTA or compressed sensing in radio imaging.

[Direct deconvolution of radio synthesis images using L1 minimisation](https://doi.org/10.1051/0004-6361/201321833)
formulated radio imaging as positive, sparse reconstruction directly against
ungridded visibilities. It used FISTA, evaluated the forward operator and its
transpose from analytic expressions rather than storing a huge matrix, and
ran the calculation on GPUs. It also supported:

- the non-coplanar-baseline term;
- arbitrary pixel positions;
- delta and multiscale Gaussian components;
- direction-dependent antenna gains;
- explicit visibility prediction rather than image-plane CLEAN subtraction;
- parallelisation across visibility and image blocks.

The paper demonstrated a one-megapixel image with more than 400,000
visibilities. It also stated the finite-component approximation that has become
important again in the beam-aware pixel work: a direction-dependent gain may
be evaluated at a component centre only when it does not vary significantly
over that component.

The current SL1MJax work therefore continues several ideas already present in
2013:

- direct visibility-domain reconstruction;
- positive L1 regularisation;
- FISTA as the fixed-model solver;
- analytic or on-the-fly measurement operators;
- irregular and multiscale sky representations;
- direction-dependent measurement effects;
- GPU-oriented decomposition of a very large linear operator.

The important new direction is not the use of FISTA. It is using unseen
visibilities to decide whether the model itself should acquire more freedom.

### Earlier compressed-sensing and FISTA lineage

The relevant sequence before Hardy (2013) is:

1. **Wiaux et al. (2009)** introduced compressed-sensing reconstruction to
   radio-interferometric imaging. Their work formulated basis-pursuit problems
   using sparse pixel or dictionary representations. It used convex splitting
   methods rather than FISTA.
2. **Wenger et al. (2010)** introduced SparseRI. It supported pixel and wavelet
   regularisers and used an accelerated iterative shrinkage framework. Its
   implementation was based on SpaRSA, not the canonical Beck--Teboulle FISTA
   update. Later reviews sometimes group these closely related accelerated
   shrinkage methods together.
3. **Li, Cornwell, and de Hoog (2010, 2011)** applied canonical FISTA to radio
   synthesis deconvolution. A 2010 conference paper preceded the complete 2011
   A&A treatment. The 2011 method operated on gridded Fourier data and tested
   both a direct pixel representation and an isotropic undecimated wavelet
   transform dictionary.
4. **Hardy (2013)** explicitly adopted the same FISTA optimisation family, but
   replaced the gridded Fourier operator with an on-demand mapping from sky
   components to raw ungridded visibilities. It added arbitrary component
   positions, multiscale Gaussian components, the non-coplanar-baseline term,
   direction-dependent antenna gains, positivity, and GPU evaluation of the
   dense operator and adjoint.

The Hardy paper itself makes the priority clear. It says that its minimisation
problem has the same form as Li et al. (2011), uses the same L1 minimisation
algorithm, and adopts FISTA because Li et al. had already demonstrated it in a
radio-synthesis context.

The defensible historical description is therefore:

> Hardy (2013) was an early application of FISTA to radio synthesis imaging and
> appears to have been the first to combine it with direct ungridded visibility
> fitting, an explicit wide-field direction-dependent measurement operator,
> arbitrary multiscale components, and on-demand GPU evaluation.

The qualification “appears to have been” is intentional. Establishing an
absolute first would require a systematic search of papers, proceedings,
technical reports, and software predating 2013. The sources examined for this
note establish that Hardy (2013) was not the first radio use of FISTA, while
supporting the narrower claim about its complete measurement and computational
architecture.

## Terminology

### Optimisation

For a fixed sky topology and fixed measurement operator, the present Stokes-I
problem has the form

\[
\min_{x\geq 0}
\frac{1}{Z}\sum_{j\in\mathcal T}
w_j\left|[A_Tx]_j-y_j\right|^2
+\lambda_1\lVert x\rVert_1,
\]

where:

- $x$ contains the non-negative component fluxes;
- $T$ is the fixed sky topology;
- $A_T$ is the RIME-derived measurement operator for that topology;
- $y_j$ is an observed complex visibility;
- $w_j$ is its weight;
- $\mathcal T$ is the training set;
- $Z$ normalises the active weights;
- $\lambda_1$ controls the positive L1 penalty.

FISTA is itself an accelerated proximal-gradient algorithm. It is therefore
slightly misleading to describe the SL1MJax method as “proximal followed by
FISTA.” The precise description is:

> stochastic proximal-gradient fitting during scalable optimisation and
> topology exploration, followed by deterministic full-batch, restarted FISTA
> after the topology and measurement operator are frozen.

Both stages use the same positive-L1 proximal map. The stochastic stage gives
cheap progress on large visibility sets. The FISTA stage gives a reproducible
fixed-problem finish and a meaningful KKT diagnostic.

FISTA's convex convergence result applies only to the fixed linear sky block.
It does not apply across topology changes, beam changes, or joint sky and gain
updates. Momentum and the Lipschitz estimate must be reset whenever one of
those changes alters the objective.

### Validation

Three different uses of data must remain distinct:

- **Training data** fit fluxes and other active parameters.
- **Selection data** decide whether to split or merge a sky component, enable a
  polarisation term, add calibration freedom, or choose between beam models.
- **Sealed test data** estimate the performance of the complete frozen
  procedure after all such choices are finished.

Repeatedly testing proposals on the same selection fold adapts the procedure to
that fold. The selection fold is no longer an unbiased final test, even though
none of its samples entered the gradient. This is why the current fold 4 must
stay sealed while folds 0--3 are used for fitting and development.

## Closest existing systems

| System | Visibility-domain inference | Proximal or sparse solver | Held-out visibility use | Relation to SL1MJax |
| --- | --- | --- | --- | --- |
| **SL1M, Hardy 2013** | Direct ungridded prediction with wide-field and direction-dependent terms | Positive L1 with FISTA | Synthetic truth tests and real-data comparisons, but not validation-gated topology | Direct computational and mathematical ancestor |
| **SparseRI, Wenger 2010** | Gridded aperture-synthesis imaging with pixel and wavelet representations | SpaRSA accelerated iterative shrinkage | Simulation and real-data assessment, but no per-change visibility gate | Important pre-FISTA-adjacent sparse-imaging precursor |
| **SASIR** | Sparse radio reconstruction, developed for LOFAR | FISTA | No per-change visibility gate found | Clear independent precedent for FISTA radio imaging |
| **PURIFY / SARA** | Continuous-visibility inverse problem | Proximal splitting, SDMM, reweighted analysis sparsity | Noise-constrained fidelity and simulation evaluation rather than adaptive held-out gating | Strong precedent for scalable convex and proximal imaging |
| **PRIISM** | Accepts Measurement Sets or visibility arrays | MFISTA with L1 and TSV, using FFT or NUFFT paths | K-fold CV selects regularisation strengths | Closest combination of an accelerated proximal solver and visibility CV |
| **EHT sparse modelling** | Complex visibility products, including closure quantities | L1 and TV sparse reconstruction | CV selects regularisation and effective resolution | Strong precedent for data-selected sparse imaging and full-polarisation reconstruction |
| **MPoL** | Differentiable visibility forward modelling in PyTorch | General gradient optimisers and RML penalties | Explicit train/test and k-fold visibility CV, including random-cell and UV-dartboard splits | Closest match to the machine-learning validation viewpoint |
| **uSARA / AIRI / PURIFY successors** | Large-scale visibility inverse problems | Forward-backward splitting and, for AIRI, a learned denoiser prior | Simulation and real-data validation, but no local held-out topology gate found | Closest modern scalable proximal and learned-prior family |
| **Repetti et al. joint imaging/calibration** | Joint visibility model for sky and direction-dependent effects | Non-convex block-coordinate forward-backward optimisation | Simulation-based assessment rather than held-out activation | Important precedent for a conditional sky solver inside self-calibration |
| **Polca SARA** | Full-polarisation imaging with direction-dependent calibration | Sparse optimisation | No equivalent local held-out activation rule found | Important full-Jones and calibration neighbour |
| **BIRO** | Fits physical sky and instrument models directly to raw visibilities | Bayesian sampling rather than FISTA | Bayesian evidence selects competing models | A principled alternative to predictive validation for model selection |
| **resolve** | Bayesian field reconstruction from visibilities | Variational or information-field inference | Posterior uncertainty rather than CV-driven topology | Alternative treatment of complexity and uncertainty |
| **R2D2** | Learned residual-to-residual visibility imaging | A series of learned residual networks | Generalisation is learned across simulated observations; stopping uses residual compatibility with noise | Machine-learning reconstruction, but not per-dataset held-out model growth |

### MPoL: the closest validation philosophy

MPoL deliberately adopts PyTorch's feed-forward, training, and testing idiom
for radio interferometry. Its cross-validation tools provide train/test pairs
from gridded visibilities. They include random UV-cell splits and “dartboard”
splits partitioned by baseline length and azimuth.

The associated ALMA study tested two CV procedures and found a broad range of
regularisation strengths with comparably good predictive power. That result is
important for SL1MJax: a tiny numerical minimum should not automatically win
over a simpler model within the same predictive uncertainty band.

MPoL is close to the desired viewpoint, but its documented CV machinery is
mainly used to select regularisation and RML configurations. It does not appear
to use held-out improvement as the acceptance rule for each local addition to
an adaptive image hierarchy.

### PRIISM: the closest optimiser and validation combination

PRIISM solves a sparse imaging objective containing least-squares visibility
error, L1 sparsity, and total squared variation. Its `mfista_fft` and
`mfista_nufft` paths use monotone FISTA. K-fold CV chooses the L1 and TSV
regularisation parameters.

PRIISM is therefore the clearest prior example of accelerated proximal radio
imaging combined with visibility cross-validation. The difference is again the
level at which CV acts. PRIISM uses it to select global processing parameters
and an image. SL1MJax proposes to use it repeatedly to govern local model
structure and later calibration freedom.

### EHT sparse modelling

The EHT sparse-modelling work uses L1 and TV regularisation and chooses their
strengths from the data with cross-validation. The full-polarisation extension
reconstructs all four Stokes parameters and reports different regularisation
requirements for the weaker Q/U signal.

This is a direct precedent for two important ideas:

- regularisation and effective resolution should be justified by predictive
  data rather than chosen only by image appearance;
- polarisation freedom requires its own evidence and cannot inherit the
  Stokes-I setting without a test.

It still does not provide the same pixel-wise activation ladder proposed here.

### PURIFY, SARA, uSARA, and AIRI

PURIFY established a scalable proximal-splitting alternative to CLEAN for
realistic continuous visibilities. Its original implementation used SDMM and
supported several sparsity priors, including SARA. Later work developed
distributed solvers, uncertainty methods, uSARA, and AIRI's learned denoiser
prior.

This family establishes that modern radio imaging can be built around
well-defined inverse problems, proximal operators, and large-scale parallel
operators. Its principal control is a prior or a noise-derived fidelity bound.
The surveyed material does not show the current SL1MJax rule that each local
model expansion must improve grouped unseen visibilities.

### Joint calibration and imaging

Repetti et al. formulate joint imaging and direction-dependent calibration as a
non-convex problem. Their block-coordinate forward-backward method alternates
between sky and gain blocks and supplies convergence guarantees appropriate to
that construction.

This supports the proposed long-term architecture for SL1MJax:

1. hold calibration fixed and solve the convex sky block;
2. hold the sky fixed and update a constrained calibration block;
3. reset the sky optimiser state after a calibration change;
4. retain a separate test for whether the extra calibration freedom predicts
   withheld data.

The optimisation precedent is strong. The held-out activation rule remains a
separate SL1MJax design choice.

### Bayesian alternatives

BIRO fits scientific and instrumental parameters directly to visibilities and
uses Bayesian evidence to distinguish source models. This is conceptually
close to asking whether additional freedom is justified, but it integrates
over a prior rather than measuring performance on withheld visibilities.

Bayesian evidence has an automatic complexity penalty and makes uncertainty
explicit. It is also much more expensive for large adaptive images. Resolve
and related information-field methods similarly put complexity control and
uncertainty in a generative posterior rather than in a train/validation split.

These methods are not competitors to be dismissed. They are useful reference
points for checking whether a validation threshold is behaving like a sensible
complexity penalty.

## What appears distinctive in SL1MJax

The survey supports a careful, limited claim. It does not support a claim that
visibility-domain sparse imaging, FISTA, full-Jones modelling, or visibility
cross-validation is new.

The combination that appears distinctive is:

1. represent the sky with an adaptive hierarchy rather than a fixed image;
2. generate local split and merge proposals;
3. fit proposal parameters only on training visibilities;
4. accept model freedom only when it improves paired prediction on grouped
   unseen visibilities;
5. keep numerical integration refinement separate from scientific sky
   refinement;
6. apply the same evidence ladder to Q, U, V, spectral terms, and calibration
   corrections;
7. preserve an outer visibility fold until the complete modelling procedure is
   frozen;
8. evaluate the resulting model through a time-, frequency-, baseline-,
   pointing-, and polarisation-aware RIME.

A concise description would be **validation-governed adaptive RIME
inference**. The novelty, if established, lies in the model-governance system
and its application across coupled sky and instrument degrees of freedom.

## A defensible validation protocol

### Split correlated observations by groups

Individual visibilities are not independent examples in the ordinary
machine-learning sense. Adjacent channels share bandpass errors. Adjacent
times share gains and atmosphere. Baselines share antennas. Correlations share
the same electronics and sky. Pointings share much of the same mosaic sky.

Randomly withholding individual samples can therefore leak nearly identical
information into training and validation. The protocol should retain several
grouped tests:

- contiguous time blocks;
- channel blocks or spectral windows;
- antenna or baseline groups;
- whole pointings where the mosaic permits it;
- correlations, with special attention to RL/LR;
- complete observations or targets for final transfer testing.

No single grouping answers every scientific question. Time holdout tests
temporal transfer. Channel holdout tests spectral transfer. Pointing holdout
tests sky/beam separation. Baseline or antenna holdout tests calibration and
array transfer.

### Use paired changes, not raw loss ranks

Every proposal should be scored on exactly the same active validation samples,
weights, flags, and numerical integration plan as its parent. For validation
groups $g$, record

\[
\Delta_g=L_g(\text{proposal})-L_g(\text{parent}).
\]

A negative value favours the proposal. The decision should consider:

- the mean or robust centre of $\Delta_g$;
- a block-bootstrap or group-level uncertainty interval;
- consistency across scientifically relevant groups;
- a minimum practical improvement, not just a negative floating-point value;
- the number of proposals tested in the round.

This is the predictive analogue of a complexity penalty. A useful conservative
rule is to prefer the simplest model whose score is statistically
indistinguishable from the best score. This is related to the
“one-standard-error” rule used in statistical model selection.

### Protect the sealed fold

A concrete hierarchy is:

1. use folds 0--2 to fit parameters;
2. use fold 3 for split, merge, beam, polarisation, and calibration choices;
3. freeze the topology, operator, priors, tolerances, and stopping rules;
4. open fold 4 once to report the selected procedure;
5. use a different observation for the strongest transfer claim.

If fold 4 changes any design decision, it has become another selection fold and
a new untouched dataset is required for final evaluation.

### Separate numerical and scientific refinement

Adding quadrature nodes to a finite sky component does not add a fitted sky
degree of freedom. It improves the accuracy of $A_T$. Its primary gate should
therefore be numerical convergence against a deeper integration reference.
Held-out loss is still useful as a regression check, but it should not decide
how much scientific complexity the sky receives.

Splitting one fitted sky leaf into independently fitted children does add
scientific freedom. That change requires held-out evidence.

Keeping these two hierarchies separate avoids spending the selection budget on
operator accuracy and avoids treating a numerical correction as an
astrophysical discovery.

## Limits of the validation viewpoint

### Predictive improvement is not physical identification

A lower held-out visibility loss shows that a model predicts the selected
unseen measurements better. It does not prove that the new term represents the
correct physical cause.

A beam error can be absorbed by the sky. A sky error can be absorbed by
self-calibration. Leakage can resemble Q/U. Differential R/L gain can resemble
V. A frequency-dependent beam can resemble a spectral index. These
compensations may generalise within one observation.

Physical identification therefore also requires:

- external calibrator anchors;
- convention and sign tests;
- transfer to other pointings and observations;
- beam or holography references;
- priors that prevent calibration from copying plausible sky variability;
- explicit reporting of unresolved degeneracies.

### Repeated proposal testing can overfit validation

An adaptive tree may evaluate thousands of candidate splits. Even if every
candidate is fitted only on training data, choosing the best result on fold 3
will eventually exploit noise or a systematic peculiarity of fold 3.

Controls should include:

- batched proposal evaluation followed by a limited acceptance budget;
- minimum effect sizes;
- group-consistency requirements;
- split penalties or false-discovery control;
- merge hysteresis;
- occasional rotation of inner folds during method development;
- a final sealed fold and independent observations.

### Optimiser convergence and predictive selection answer different questions

The KKT residual asks whether the fixed training objective is near a stationary
solution. Held-out loss asks whether the fitted model predicts other data.

A model can have a poor KKT residual and a good validation score, or the
reverse. Development runs should record both and warn when either is weak.
Production comparisons need a convergence tolerance tight enough that solver
error is smaller than the claimed difference between models.

## Implications for SL1MJax development

### Keep the hybrid optimiser

The stochastic-proximal-to-FISTA sequence is well motivated:

- stochastic proximal steps make large, time-dependent RIME fits tractable;
- explicit tiled adjoints reduce the cost of those steps;
- FISTA gives a deterministic fixed-topology finish;
- the positive-L1 proximal map creates exact zeros;
- KKT diagnostics reveal whether beam or topology rankings are smaller than
  remaining optimisation error.

The production report should call this a hybrid proximal solver and state both
stages. It should not imply that FISTA is separate from proximal optimisation.

### Make validation a first-class artifact

Every accepted model change should record:

- parent and proposal identifiers;
- training and selection sample hashes;
- grouping axis and fold definition;
- paired loss changes by group;
- uncertainty and practical threshold;
- convergence and KKT state;
- active flags and weights;
- beam, calibration, and sky schema versions;
- the reason for acceptance or rejection.

This record matters more than an attractive image. It is the evidence trail for
why the model acquired each degree of freedom.

### Benchmark against the closest neighbours

The most informative external comparisons are:

1. **Hardy 2013 / fixed-grid SL1M:** establish continuity and quantify what the
   adaptive and validation layers add.
2. **PRIISM:** compare regularisation chosen by ordinary k-fold CV with the
   hierarchical acceptance procedure.
3. **MPoL:** compare grouped split strategies and predictive diagnostics.
4. **CASA:** compare operational image quality, residuals, calibration
   transfer, and compute cost on ordinary C-band observations.
5. **A Bayesian model comparison on a small problem:** check whether validation
   and evidence prefer the same simple source alternatives.

The strongest eventual claim is not “FISTA beats CLEAN.” The 2013 work and a
large compressed-sensing literature already established that this is a useful
class of radio imaging method. The stronger new question is:

> Can a radio imager build a richer sky and instrument model while requiring
> every added freedom to demonstrate predictive value on unseen,
> scientifically structured visibility groups?

## References

- Aghabiglou, A. et al. (2024), [R2D2 image reconstruction with model
  uncertainty quantification in radio astronomy](https://arxiv.org/abs/2403.18052).
- Akiyama, K. et al. (2017), [Imaging the Schwarzschild-radius-scale structure
  of M87 with the Event Horizon Telescope using sparse
  modeling](https://doi.org/10.3847/1538-4357/aa6305).
- Akiyama, K. et al. (2017), [Super-resolution full polarimetric imaging for
  radio interferometry with sparse modeling](https://arxiv.org/abs/1702.00424).
- Beck, A. & Teboulle, M. (2009), [A fast iterative shrinkage-thresholding
  algorithm for linear inverse problems](https://doi.org/10.1137/080716542).
- Carrillo, R. E., McEwen, J. D., & Wiaux, Y. (2014), [PURIFY: a new approach
  to radio-interferometric imaging](https://doi.org/10.1093/mnras/stt2011).
- Czekala, I. et al. (2025), [Million Points of Light: a PyTorch library for
  radio interferometric imaging and inference](https://arxiv.org/abs/2502.00100).
- Girard, J. N. et al. (2015), [Sparse representations and convex optimization
  as tools for LOFAR radio interferometric
  imaging](https://arxiv.org/abs/1504.03896).
- Hardy, S. J. (2013), [Direct deconvolution of radio synthesis images using L1
  minimisation](https://doi.org/10.1051/0004-6361/201321833).
- Li, F., Cornwell, T. J., & de Hoog, F. (2010), [The applications of
  compressive sensing to radio
  astronomy](https://doi.org/10.1007/978-3-642-14654-1_46).
- Li, F., Cornwell, T. J., & de Hoog, F. (2011), [The application of
  compressive sampling to radio astronomy I:
  Deconvolution](https://doi.org/10.1051/0004-6361/201015045).
- Lochner, M. et al. (2015), [Bayesian inference for radio
  observations](https://doi.org/10.1093/mnras/stv679).
- Nakazato, T. et al., [PRIISM: Python module for radio interferometry imaging
  with sparse modeling](https://github.com/tnakazato/priism).
- Repetti, A. et al. (2017), [Non-convex optimization for self-calibration of
  direction-dependent effects in radio interferometric
  imaging](https://doi.org/10.1093/mnras/stx1197).
- Thouvenin, P.-A. et al. (2020), [Polca SARA: full polarization,
  direction-dependent calibration, and sparse imaging for radio
  interferometry](https://doi.org/10.1093/mnras/stz3426).
- Wenger, S. et al. (2010), [SparseRI: a compressed sensing framework for
  aperture synthesis imaging in radio
  astronomy](https://doi.org/10.1086/657252).
- Wiaux, Y. et al. (2009), [Compressed sensing imaging techniques for radio
  interferometry](https://doi.org/10.1111/j.1365-2966.2009.14665.x).
- Zawadzki, B. et al. (2023), [Regularized maximum-likelihood image synthesis
  and validation for ALMA continuum observations of protoplanetary
  disks](https://doi.org/10.1088/1538-3873/acdf84).
