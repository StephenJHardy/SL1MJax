# Beam-aware pixel integration proposal

## Purpose

SL1MJax represents extended emission with finite square quadtree leaves. The
visibility from one leaf is an integral over its area. The integrand contains
both the sky brightness and the direction-dependent antenna response. The beam
therefore belongs inside the pixel integral.

The current operators use two different approximations. The original quadtree
operator integrates the Fourier phase over each square analytically, but
samples the scalar beam only at the leaf centre. The newer streamed voltage
beam experiment flattens every positive component to its centre and treats it
as a delta function. The latter drops both the finite square kernel and the
beam variation across the square.

These approximations are most questionable for the 60 arcsec coarse field,
off-axis leaves, beam nulls and sidelobes, mosaics, squint, and full-Jones
leakage. They can favour the simple Airy beam that was used to fit the sealed
sky. They can also make a correct detailed beam appear worse because the beam
is evaluated with a different sky basis from the one used during imaging.

This proposal defines a beam-aware finite-pixel operator. It keeps the fitted
sky model fixed while improving the numerical integral. It also defines the
validation path required before this operator replaces either the current
quadtree operator or the point-centre beam experiment.

## Decision

Use adaptive subcell quadrature as the correctness reference for finite square
pixels. Each square leaf is recursively divided into four equal subcells. The
parent flux is distributed over those subcells according to the declared
within-pixel sky basis. The subcells are numerical integration nodes, not new
sky parameters.

For a uniform square leaf, every child initially carries one quarter of the
parent flux. The beam Jones is evaluated at each child centre. The analytic
square Fourier kernel is retained for the child area. The child responses are
then summed to obtain the response of the original leaf.

Use a hybrid production implementation after the reference path is validated:

1. use the existing centre evaluation where its error is below tolerance;
2. use beam derivatives and analytic pixel moments in smooth beam regions;
3. subdivide where the moment approximation does not meet tolerance;
4. always subdivide near nulls, support boundaries, model handovers, and other
   non-smooth regions;
5. compare every accelerated path against the subcell reference.

Do not make beam Taylor coefficients independently fitted image parameters.
They are deterministic consequences of the selected beam and the one fitted
flux coefficient for the leaf. A later model may add genuine within-pixel sky
structure, but that is a separate topology decision with its own validation
penalty.

Point catalogue atoms remain point sources. A source with a known finite shape
must declare that shape and use a matching integral. It must not inherit the
square-leaf rule accidentally.

## Goals

The beam-aware operator should:

- evaluate the product of sky basis, beam Jones, and Fourier phase over each
  finite component;
- preserve the integrated-flux meaning of every quadtree coefficient;
- support scalar, diagonal, and full-Jones voltage beams through one contract;
- use the same celestial sky for every pointing in a mosaic;
- evaluate pointing coordinates and feed rotation at each row or unique time;
- retain channel-dependent finite-pixel fringe integration;
- support mixed leaf widths without flattening all leaves to delta functions;
- expose numerical error estimates and convergence diagnostics;
- remain differentiable with respect to sky flux and any later differentiable
  beam parameters;
- stream directions, times, channels, and antenna planes within explicit
  memory limits;
- separate numerical integration refinement from scientific sky refinement;
- give detailed beam models a fair held-out comparison against the Airy model.

The first implementation is not intended to:

- infer subpixel sky gradients from the same parent flux;
- select a new quadtree topology;
- establish that the current overlapping coarse field is the final sky basis;
- fit pointing offsets, antenna-specific beam errors, or leakage corrections;
- make the unfrozen CASSBEAM full-Jones reference scientifically accepted;
- replace the independent CASA or holography beam comparisons;
- hide an expensive uniform oversampling rule inside the default imager.

## Terminology

A *sky leaf* is one fitted component in the scientific image model. Its
coefficient is integrated flux in Jy.

A *basis function* describes how that flux is distributed within the leaf. A
uniform square is the current quadtree basis. A delta function is the basis for
a point catalogue atom.

An *integration node* is a deterministic sample used to evaluate the response
of a basis function. It does not have an independently fitted coefficient.

*Sky refinement* replaces one fitted parent coefficient with several fitted
child coefficients. It adds degrees of freedom and requires held-out evidence.

*Integration refinement* uses more nodes to calculate the response of the
same fitted coefficient. It adds compute but no sky freedom.

The *centre-factorized approximation* evaluates the beam at a leaf centre and
pulls that value outside the finite-pixel Fourier integral.

The *subcell reference* divides a leaf into smaller uniform squares, evaluates
the beam at their centres, and sums their analytic square responses.

A *moment approximation* expands the beam or apparent coherency around the
leaf centre and combines the expansion with analytic moments of the square
Fourier kernel.

The *apparent coherency* for baseline antennas $p$ and $q$ is

\[
A_{pq}(s,\nu,t)=E_p(s,\nu,t)C(s,\nu)E_q^{\rm H}(s,\nu,t),
\]

where $E$ is the antenna voltage Jones and $C$ is the sky coherency.

## Measurement equation

For a finite sky component $k$, the exact contribution to a visibility is

\[
V_{pq,k}(\nu,t)=
\int_{\Omega_k}
E_p(s,\nu,t)C_k(s,\nu)E_q^{\rm H}(s,\nu,t)
e^{-2\pi i\,\boldsymbol u_{pq}(\nu,t)\cdot s}\,d\Omega.
\]

Here $s$ is a celestial direction within the component support $\Omega_k$.
The two Jones matrices depend on antenna, direction, frequency, time,
pointing, and receptor convention. The sky coherency may also vary within the
component if its basis says so.

For a uniform unpolarised square with integrated flux $I_k$, the current
centre-factorized form is approximately

\[
V_{pq,k}\simeq
E_p(s_k)C_kE_q^{\rm H}(s_k)
K_0(\boldsymbol u_{pq};s_k,w_k),
\]

where $s_k$ is the centre, $w_k$ is the angular width, and $K_0$ is the
analytic square-pixel Fourier response. In the paraxial case it contains

\[
\operatorname{sinc}(u w_k)\operatorname{sinc}(v w_k)
e^{-2\pi i(ul_k+vm_k)}.
\]

This expression is exact only when the Jones response is constant across the
square. A detailed beam can vary in amplitude, phase, receptor balance, and
leakage across the same area.

The point-centre voltage experiment makes a stronger approximation:

\[
V_{pq,k}\simeq
E_p(s_k)C_kE_q^{\rm H}(s_k)
e^{-2\pi i\,\boldsymbol u_{pq}\cdot s_k}.
\]

It treats the finite leaf as a delta source and drops the square sinc factors.
The excellent Airy point-operator agreement in the current JAX gate therefore
tests the implementation of this point model. It does not test equivalence to
the fitted quadtree sky representation.

## Why the approximation matters

### Finite leaves

The sealed 3C391 sky contains 16, 8, and 4 arcsec central quadtree leaves. It
also contains a separate 64 by 64 field of 60 arcsec squares. The smallest
central leaves are close to point-like on many baselines. The 60 arcsec
components are not.

Finite-pixel phase integration matters even for a constant beam. Beam-aware
integration adds a second effect when the voltage response changes across the
same square. The two effects must be tested separately.

### Mosaics

One celestial leaf appears at a different local beam coordinate in every
pointing. Pulling the beam outside the integral makes the error depend on the
pointing offset. A single fitted flux can then have inconsistent apparent
responses across the mosaic.

The error can be small at the mosaic centre and larger on the pointing ring.
That pattern resembles the current fixed-sky comparison, where CASSBEAM was
slightly better on C1 but worse on C2--C7. This does not prove that pixel
integration caused the result. It does mean the integration mismatch must be
removed before the beam ranking is treated as a scientific conclusion.

### Parallactic rotation and polarisation

An asymmetric full-Jones beam rotates relative to the celestial pixel. The
spatial average of $E_p C E_q^{\rm H}$ therefore changes with time even for a
static sky. A centre sample can misestimate this change.

For diagonal beams, the error can appear as an RR/LL imbalance and hence false
Stokes V. For off-diagonal beams, it can appear as false Q/U or an incorrect
cross-hand phase. The effect is strongest where the beam changes rapidly.

### Frequency structure

The pixel kernel changes with baseline coordinates measured in wavelengths.
The beam also narrows and changes structure with frequency. Evaluating both at
native channel frequencies is required before channel-held-out improvements
can be interpreted as sky spectra or rotation measure.

### Sidelobes and nulls

The response near a beam null can change sign or phase within a pixel even
when the centre response is close to zero. A centre sample can therefore miss
a non-zero integrated response. A low-order Taylor series can also fail there.
Subdivision is the safer rule.

Bright sources in sidelobes can dominate an aggregate validation loss. That
is a real measurement effect, but the integration must be accurate enough to
tell a beam-model difference from a numerical sampling difference.

## Current SL1MJax representations

### Central quadtree

The current frozen central topology has a 104 by 104 root grid with a 16
arcsec root width. Its footprint is about 27.73 arcmin square. The accepted
topology contains 10,591 leaves at 16 arcsec, 885 at 8 arcsec, and 60 at 4
arcsec.

[`predict_quadtree_stokes_i`](../src/sl1mjax/quadtree.py) groups leaves by
depth and calls the square-pixel visibility operator at the width for that
depth. It samples the scalar primary beam at leaf centres. It therefore keeps
the finite square Fourier kernel but factorizes the beam from the pixel
integral.

### Coarse field

The composite fit adds a 64 by 64 grid of 60 arcsec squares. Its footprint is
64 arcmin square, so it covers the entire central quadtree and extends beyond
it. The coarse component was introduced to test broad short-baseline
residuals. It was not introduced as a general multiresolution image basis.

After explicit catalogue atoms were added, the coarse field provided about a
0.52% incremental sealed-test improvement. It remains a useful diagnostic,
but that result does not establish that an overlapping positive coarse image
is the best permanent representation.

The sealed positive sky used by the current beam experiment contains 4,748
positive central components, 1,119 positive coarse components, and two
positive catalogue atoms. About half of the fitted coarse flux lies inside the
central footprint. The central and coarse components therefore overlap both
spatially and in the modes they can represent.

This proposal does not require that overlap to be retained. It does require
the declared 60 arcsec square basis to be integrated correctly while it is
used.

### Catalogue atoms

The current outer catalogue guard uses fixed-position delta atoms with free
non-negative flux. Those atoms should continue to use the delta kernel unless
a source is explicitly given a finite shape.

### Streamed voltage beam experiment

[`flatten_positive_sky`](../scripts/diagnose_3c391_voltage_beam_transfer.py)
currently exports every positive quadtree and coarse component as a centre
coordinate and flux. The widths are discarded.

[`predict_voltage_beam_jax`](../src/sl1mjax/voltage_operator_jax.py) then uses a
delta kernel for those components. It evaluates Airy, Perley-plus-Airy,
CASSBEAM diagonal, or experimental CASSBEAM full Jones efficiently by unique
time and direction tiles. The voltage-beam evaluation is useful, but the sky
side of this comparison is not yet the fitted finite-pixel model.

## Separation of sky and numerical integration

The implementation must carry two independent structures.

The scientific sky table contains:

- component identifier;
- component type;
- celestial centre;
- integrated Stokes coefficients;
- square width or other shape parameters;
- quadtree level and parent identity where applicable;
- provenance and active-state information.

The integration plan contains:

- quadrature rule;
- node offsets within each component;
- fixed node weights;
- estimated integration error;
- selected refinement depth;
- beam support and validity at the nodes.

Changing the integration plan must not change the number of fitted sky
coefficients. A one-Jy uniform parent remains one fitted one-Jy component
whether it uses one, four, sixteen, or sixty-four numerical subcells.

Changing the sky topology is different. Replacing that parent with four free
children creates three additional spatial degrees of freedom after total flux
is accounted for. That decision belongs to the split and merge validation in
[`hierarchical_pixels_proposal.md`](hierarchical_pixels_proposal.md).

## Reference subcell integrator

### Uniform square rule

At subdivision depth $d$, divide a square of width $w$ into $4^d$ equal
subcells. Each subcell has width $w/2^d$. For a uniform parent, each subcell
has flux $I_k/4^d$.

For subcell $j$, evaluate

\[
V_{pq,kj}=
E_p(s_{kj})C_{kj}E_q^{\rm H}(s_{kj})
K_0(\boldsymbol u_{pq};s_{kj},w/2^d).
\]

Then sum

\[
V_{pq,k}^{(d)}=\sum_j V_{pq,kj}.
\]

If the beam is constant, the tiled subcells must reproduce the parent square
kernel. This identity is a core unit test. It checks flux normalization, child
positions, phase signs, width conventions, and the definition of sinc.

### Celestial and pointing coordinates

Subcell offsets are first defined in the common mosaic tangent plane. Each
node is converted to an absolute celestial direction. That direction is then
converted to local $(l,m)$ coordinates for each pointing.

The beam evaluator receives the local coordinate, row time, exact frequency,
antenna identity, and parallactic angle. The Fourier phase receives the same
celestial direction in the phase-centre frame used for the baseline.

This order avoids treating a pointing-local square as if it were a different
sky component. All pointings still measure one shared sky.

### Full-Jones rule

The Jones matrix is evaluated independently for both antennas at every node.
The node sky coherency is then transformed as

\[
C_{pq,kj}^{\rm app}=E_p(s_{kj})C_{kj}E_q^{\rm H}(s_{kj}).
\]

The requested RR, RL, LR, and LL products are packed only after the complete
two-receptor multiplication. An RR-only data block must still use the full
two-receptor Jones space.

Beam validity is evaluated per node and per channel. Unsupported nodes
contribute zero and carry an explicit validity mask. A leaf is not invalidated
merely because one outer node is unsupported. Off-diagonal validity remains
separate from diagonal validity, as required by the voltage-beam contract.

### Adaptive convergence

The simplest reference rule compares successive subdivision depths:

\[
\epsilon_d =
\frac{\lVert V^{(d)}-V^{(d-1)}\rVert}
     {\max(\lVert V^{(d)}\rVert,V_{\rm floor})}.
\]

The norm and floor must be defined in the run manifest. A component may stop
when its predicted contribution meets both a relative and an absolute error
threshold. The absolute threshold prevents excessive refinement around a
nearly zero response.

The first correctness implementation may use a fixed maximum depth. It must
still report the convergence sequence. The production path may decide depth
from beam variation, component width, baseline range, frequency, and the
previous depth error.

The error decision must be conservative near:

- beam nulls and phase reversals;
- the boundary of a tabulated beam raster;
- diagonal or off-diagonal validity boundaries;
- the Perley-to-Airy outer-field handover;
- nearest-frequency-node boundaries;
- strong leakage or receptor imbalance;
- the celestial horizon or invalid direction-cosine region.

## Moment acceleration

### Expansion

In a smooth region, expand the apparent coherency around the leaf centre in
image-coordinate offsets $\delta l$ and $\delta m$:

\[
A(\delta l,\delta m)\simeq
A_0+A_l\delta l+A_m\delta m
+\frac{1}{2}A_{ll}\delta l^2
+A_{lm}\delta l\delta m
+\frac{1}{2}A_{mm}\delta m^2.
\]

The derivatives may be formed from the two antenna Jones matrices using the
product rule, or by differentiating the complete apparent coherency. They are
deterministic functions of the beam, time, frequency, baseline antennas, and
the parent sky coherency.

The corresponding visibility is a sum of analytic Fourier moments. If $K_0$
is the uniform square kernel, then

\[
K_l=\frac{1}{2\pi i}\frac{\partial K_0}{\partial u},
\qquad
K_m=\frac{1}{2\pi i}\frac{\partial K_0}{\partial v},
\]

with second derivatives giving $K_{ll}$, $K_{lm}$, and $K_{mm}$. Sign and
normalization must be verified against direct quadrature rather than assumed
from this shorthand.

The second-order response is then

\[
V\simeq A_0K_0+A_lK_l+A_mK_m
+\frac{1}{2}A_{ll}K_{ll}
+A_{lm}K_{lm}
+\frac{1}{2}A_{mm}K_{mm}.
\]

### Why linear order is not enough

A symmetric square has zero unphased first spatial moment. A centred symmetric
beam can also have zero first derivative. The leading integration error can
therefore be quadratic even when a linear expansion is available.

Linear order remains useful off axis and on long baselines, where the complex
Fourier moments need not vanish. The likely practical minimum is nevertheless
a quadratic expansion with a subdivision fallback.

### Derivative source

Analytic derivatives are preferable for analytic Airy and Perley models. JAX
automatic differentiation may be used where the evaluator is smooth.
Finite-difference derivatives are acceptable as a test oracle but should not
become the production default without a step-size study.

Tabulated CASSBEAM interpolation is only piecewise smooth. Derivatives can be
discontinuous at cell boundaries. Nearest frequency-node selection is also
non-differentiable at its handover. These regions should use subcell
quadrature unless a smoother, validated interpolant is introduced.

### No extra sky terms

The moment matrices multiply the same parent Stokes coefficient. They must not
appear as free per-leaf gradient or curvature coefficients in the optimizer.

Free coefficients would describe a more complex sky basis. Such a model may
be scientifically useful, but it would need positivity or coherency
constraints, a complexity penalty, split and merge rules, and held-out
selection. It is outside this numerical-integration change.

## Relation to the coarse 60 arcsec field

The present 60 arcsec field covers the full 64 arcmin frame. It is not simply
a coarser level of the current central tree because it overlaps active finer
leaves. A prefix-free quadtree normally keeps either a parent or its children
active, not both.

A cleaner future geometry can use 64 arcsec outer roots. The central 26 by 26
roots can be refined twice to 16 arcsec, then retain the existing 8 and 4
arcsec refinements. This nests naturally because

\[
26\times64\text{ arcsec}=104\times16\text{ arcsec}=1664\text{ arcsec}.
\]

That change would create one prefix-free wide-field tree. It should be tested
as a sky-model proposal, not bundled into the first beam-aware operator.

If the scientific model truly needs a smooth large-scale field plus finer
detail at the same location, it should say so explicitly. One option is a
coarse positive base plus constrained zero-mean detail coefficients. Two
unrelated overlapping positive images are harder to identify and can exchange
flux when the beam changes.

The immediate beam experiment should therefore report results both with and
without the coarse field. This shows whether a beam ranking is driven by the
large 60 arcsec cells or by the central and catalogue components.

## Build plan

### Phase 0: freeze the comparison contract

Record the exact sky checkpoint, component tables, splits, flags, weights,
channel frequencies, correlation order, calibration state, pointing centres,
and beam artifacts used by the current fixed-sky comparison.

Reproduce the present point-centre Airy JAX gate and held-out beam report. This
is a regression baseline, not an acceptance test for finite pixels.

The manifest must include component counts and flux by type and width. It must
state explicitly that the current transfer operator discarded finite widths.

### Phase 1: preserve the sky basis in the voltage path

Replace the flattened `(l, m, flux)` contract with a component contract that
also carries basis type and width. Keep point catalogue atoms distinct from
uniform squares.

Support mixed widths without expanding them into fitted children. Grouping by
width is acceptable if it improves compilation or kernel reuse.

Add a constant-Jones mode. Under that mode, the new voltage operator must
match the existing square-pixel quadtree prediction for scalar Stokes I.

### Phase 2: implement fixed-depth subcell quadrature

Add a clear reference function that evaluates one, four, sixteen, or sixty-four
subcells per square. Retain the analytic square kernel at every subcell.

Implement the scalar beam first, then diagonal Jones, then full Jones. The
three paths must share the same coordinate and quadrature code.

Keep this function usable outside the optimizer. It is the oracle for the
accelerated JAX path and for convention debugging.

### Phase 3: implement the streamed JAX reference

Extend the current unique-time voltage operator to stream integration nodes.
Reuse a Jones evaluation wherever antenna, time, channel, and node coordinates
are identical.

Avoid materializing a full
`time × antenna × node × channel × 2 × 2` cube unless the memory preflight says
it fits. Preserve the current pair-streaming fallback for antenna planes.

Add Stokes-I gradients first. Then verify gradients for Q, U, and V or their
chosen constrained parameterization before polarisation fitting uses this
path.

### Phase 4: add adaptive error control

Calculate successive-depth predictions on a representative set of rows,
baselines, channels, hands, and pointings. Use those measurements to set
relative and absolute integration tolerances.

The adaptive rule may use a cheap beam-variation screen, but acceptance is
based on the visibility difference between depths. The screen must not be the
only correctness criterion.

Record the fraction of components using each depth. Record which beam features
caused forced subdivision.

### Phase 5: add moment acceleration

Implement zeroth-, first-, and second-order apparent-coherency moments. Compare
each order with the converged subcell result across the supported beam domain.

Use moment evaluation only where the comparison meets the declared tolerance.
Fall back to subcells elsewhere. Keep an option that forces the reference path
for scientific audits.

### Phase 6: integrate with fixed-topology imaging

Add an explicit beam-integration mode to the imager. Do not silently change
the existing Airy default.

First fit the existing frozen topology and component set. Only fluxes may
move. Use time-grouped proximal SGD for the time-dependent beam path unless a
validated full-batch operator is available. Preserve sealed validation folds
and L1 weights across beam comparisons.

The mode must report whether each component used point, centre-square, moment,
or subcell integration.

### Phase 7: reconsider the sky topology

After the operator is accepted, compare the overlapping central-plus-coarse
model with a prefix-free wide-field hierarchy. Keep catalogue atoms in both
experiments.

Use the existing split, merge, and held-out topology rules. Do not choose the
new hierarchy because it is aesthetically cleaner. Require data-space and
sealed-test evidence.

### Phase 8: enable polarisation imaging

Run the diagonal and full-Jones paths on calibrated RR, RL, LR, and LL data.
Keep Stokes I fixed for the first global Q/U comparison and set V to zero
unless its own activation test passes.

Only then allow regional Q/U or V. A spatial polarisation term must improve
held-out pointings, channels, times, baselines, and relevant hands after the
beam-aware integration is active.

## Validation path

### Gate 1: geometry and flux conservation

Test every supported square width and subdivision depth.

- Child centres must tile the parent exactly.
- Child areas and flux weights must sum to the parent.
- The celestial centre must be invariant under subdivision.
- A zero-width limit must approach the delta kernel.
- Point atoms must be unchanged by square integration settings.
- A one-Jy component must remain one Jy after any integration refinement.

Use deliberately asymmetric coordinates to catch axis swaps and reflection
errors. Include directions on both sides of each pointing centre.

### Gate 2: constant-beam analytic identity

With a constant scalar or Jones beam, the sum of subcell square kernels must
match the analytic parent square kernel to floating-point tolerance.

Test multiple baseline orientations, lengths, frequencies, pixel widths, and
non-zero $w$ terms. Test all four correlations with a non-trivial constant
Jones matrix.

This gate isolates the pixel geometry from the beam. Failure means the
quadrature tiling or Fourier convention is wrong.

### Gate 3: manufactured smooth beams

Construct scalar and Jones beams that are constant, linear, and quadratic in
$l$ and $m$. Their finite-square integrals can be obtained from analytic
moments or very high-order numerical quadrature.

The centre rule should show its expected error. The matching moment order
should recover the manufactured result. Subcell predictions should converge
monotonically or at their expected numerical rate.

Include complex off-diagonal terms. Include a diagonal R/L displacement. This
tests product-rule derivatives and correlation packing without relying on the
scientific CASSBEAM artifact.

### Gate 4: NumPy and JAX agreement

Compare the streamed JAX operator with the clear NumPy reference at identical
nodes. Cover scalar Airy, Perley-plus-Airy, CASSBEAM diagonal, and experimental
CASSBEAM full Jones.

Compare visibility values, validity masks, and off-diagonal validity masks.
Test one- and two-antenna beam planes, single-hand data blocks, and all four
correlations.

Use 64-bit calculations for the oracle comparison. Set separate tolerances for
32-bit production runs if they remain supported.

### Gate 5: derivatives and adjoints

Check Stokes-I value and gradient against finite differences on a small mixed-
width sky. Check the linear operator and adjoint with a complex dot-product
identity.

Repeat for full Stokes before Q/U/V optimization uses this path. Include a
case in which only off-diagonal Jones terms couple I into cross-hands.

Integration refinement must not change the gradient dimension. The gradient
has one entry per fitted sky parameter, not one entry per integration node.

### Gate 6: convergence on real beam artifacts

For each detailed beam, compare depths 0, 1, 2, and 3 on a stratified sample
of the real 3C391 geometry. This corresponds to 1, 4, 16, and 64 subcells per
square.

Stratify by:

- 4, 8, 16, and 60 arcsec component width;
- main-lobe centre, half-power region, pointing ring, null, sidelobe, and
  raster boundary;
- low, middle, and high channel;
- short, middle, and long baselines;
- RR, LL, RL, and LR where present;
- representative parallactic-angle bins;
- central, coarse, and catalogue component groups.

The accepted reference depth is the first depth whose change is below both the
declared relative tolerance and the absolute visibility floor. If depth 3 is
not converged, increase the oracle depth or mark that region unsupported for
the accelerated path.

The numerical tolerance should be small relative to the beam-model loss
differences being interpreted. It should also be below the residual change
that would trigger a sky split or polarisation activation. A fixed percentage
without this scale comparison is not sufficient.

### Gate 7: mosaic coordinate conventions

Use one finite celestial component observed by several synthetic pointings.
Verify that its sky position and flux remain fixed while only local beam
coordinates change.

Test zero and non-zero parallactic angle. Test a known asymmetric Jones beam
whose expected rotation can be calculated independently. Include pointing
pairs on opposite sides of the mosaic centre to expose sign errors.

Repeat the CASSBEAM absolute-origin, squint-direction, and
$e^{\pm2i\chi}$ off-diagonal phase tests at subcell nodes. Pixel integration
must not reintroduce the convention errors already removed from the beam
evaluator.

### Gate 8: fixed-sky 3C391 ablation

Repeat the four-beam held-out comparison with the sealed fluxes fixed:

1. static Airy;
2. Perley-plus-Airy composite;
3. CASSBEAM diagonal;
4. experimental CASSBEAM full Jones.

For every beam, report at least these sky-integration variants:

1. point centres, reproducing the current experiment;
2. finite square with centre-factorized beam;
3. 2 by 2 subcell integration;
4. 4 by 4 subcell integration;
5. the converged reference or adaptive result.

Run component ablations for central plus catalogue, coarse plus catalogue, and
the full composite sky. This identifies whether a ranking change comes from
the 60 arcsec field.

Report loss by pointing, channel, time or parallactic-angle bin, baseline bin,
and hand. Do not use the aggregate mean alone. Preserve the exact split,
weights, flags, and correlations from the current fixed-sky run.

This gate answers whether the previous beam ranking was sensitive to the pixel
approximation. It does not choose the final beam because the sky was originally
fitted under Airy.

### Gate 9: controlled flux refit

Refit the same frozen geometry independently under each beam and integration
mode. Hold the component positions, widths, topology, folds, flags, weights,
and regularization contract fixed.

Fit only Stokes-I flux at first. Select optimizer settings without looking at
the sealed fold. Confirm convergence using objective and gradient diagnostics,
not a common iteration count alone.

Compare paired held-out changes and off-axis flux movement. Large flux
exchange between the overlapping central and coarse fields must be reported.
It can indicate a sky-basis identifiability problem rather than a beam win.

The detailed beam should not be rejected merely because it loses with an
Airy-fitted sky. It should also not be accepted merely because a flexible
refit absorbs its errors. The transfer and refit results must be read together.

### Gate 10: topology and representation test

If the 60 arcsec field drives material integration error or flux exchange,
compare it with the prefix-free wide-field hierarchy. Use identical beam,
calibration, data splits, and catalogue guards.

Require the usual inner validation for topology decisions and a sealed outer
test for the final representation. Record leaf counts, scales, flux, and
compute cost.

### Gate 11: full-polarisation test

Use calibrated four-correlation data. First compare diagonal and full Jones
with Stokes I fixed and a global Q/U model. Report RR/LL and RL/LR losses
separately.

Test enough parallactic-angle range to distinguish a sky-fixed polarisation
signal from a rotating feed-frame leakage pattern. The present 3C391 fixture
has limited leverage, so this gate may require a different calibrator or
target observation.

Do not freeze full Jones from an RR/LL-only result. Off-diagonal beam support
must be exercised by cross-hands.

### Gate 12: independent C-band transfer

Run the accepted operator and beam choices on at least one C-band observation
that did not influence the sky basis, beam settings, integration tolerance, or
optimizer configuration.

Compare against the existing SL1MJax Airy path and an appropriate CASA result.
Where available, compare beam slices or apparent source responses with CASA
`awp2`, holography, or another independent reference.

This gate supports the broader claim that the system can reduce an arbitrary
C-band observation reliably. The 3C391 tutorial remains a development case,
not the only scientific validation.

## Acceptance criteria

The beam-aware operator is ready for optional imaging use when all of the
following are true:

- the component contract preserves basis type, width, integrated flux, and
  provenance;
- constant-beam subdivision reproduces the analytic parent square response to
  the declared floating-point tolerance;
- NumPy and JAX reference predictions agree for scalar, diagonal, and
  full-Jones cases;
- gradients and adjoints pass for mixed component widths;
- integration refinement adds no fitted sky parameters;
- the real-beam convergence study establishes a maximum reference error over
  the supported domain;
- the adaptive or moment path matches that reference within its declared
  tolerance;
- validity and off-diagonal support are propagated per node without dropping
  otherwise usable rows;
- memory preflight fails closed and the streamed path remains within its
  declared budget;
- the current point-source and default Airy paths remain unchanged unless an
  explicit new mode is selected;
- fixed-sky and controlled-refit reports both exist for 3C391;
- no beam is promoted solely from an aggregate loss on an Airy-fitted sky.

The operator becomes the default finite-pixel path only after an independent
C-band transfer succeeds. Full-Jones imaging additionally requires a
four-correlation validation with adequate parallactic-angle leverage.

## Reporting contract

Every beam-aware imaging or diagnostic run should write a machine-readable
manifest with:

- code revision and command line;
- sky checkpoint hash;
- component counts and intrinsic flux by type and width;
- fitted parameter count;
- beam mode, artifact version, convention, and support;
- channel frequencies, pointing centres, time range, and PA range;
- correlation order and calibration state;
- split, flag, and weight provenance;
- integration mode and tolerance;
- maximum permitted and actual subdivision depth;
- component and flux fraction at each depth;
- number of Jones evaluations and visibility kernel evaluations;
- numerical convergence summaries;
- invalid and off-diagonal-unsupported node fractions;
- peak memory, compile time, and execution time;
- aggregate and stratified held-out losses;
- paired deltas against the declared reference;
- optimizer convergence diagnostics for refit runs;
- reasons for every acceptance or refusal gate.

Plots should include:

- integration error versus component width and beam radius;
- convergence versus subcell depth;
- a sky map coloured by selected integration depth;
- loss deltas by pointing and baseline length;
- RR and LL deltas, plus RL and LR when present;
- flux movement by component and off-axis radius after refitting;
- timing and memory versus node count;
- locations where moment evaluation fell back to subdivision.

## Performance design

The reference path should favour clarity. It may be too slow for a full fit,
but it must be practical on representative row and component subsets.

The production path should exploit these reuse opportunities:

- reuse celestial subcell coordinates across pointings;
- reuse pointing-local coordinates across baselines at one time;
- evaluate each antenna Jones once per unique time, node tile, and channel;
- reuse analytic moment kernels for components with the same width;
- keep catalogue delta atoms out of square quadrature;
- use low quadrature order for fine leaves that pass the error test;
- group work by time for proximal SGD;
- stream antenna pairs when the full Jones cube exceeds the memory budget.

Precomputation is allowed only when its cache key contains every coordinate on
which the result depends. At minimum this includes beam artifact, convention,
pointing, time or PA, antenna plane, channel frequency, component geometry,
and integration rule.

Profile before introducing interpolation or low-rank compression. The current
JAX voltage operator reduced a pointing experiment from hours to minutes, so a
moderate increase in node count may be acceptable. The measured cost and
validation benefit should determine the optimization work.

## Risks and controls

### Numerical nodes become hidden sky freedom

This would make comparison between integration orders invalid.

**Control:** tie every node weight deterministically to the parent coefficient
and assert that gradient dimension is unchanged.

### The detailed beam is blamed for a sky-basis mismatch

An Airy-fitted or overcomplete sky can favour Airy even under a correct
operator.

**Control:** report both fixed-sky transfer and frozen-geometry refit. Add the
prefix-free representation ablation if coarse/central flux exchange is large.

### Quadrature is excessive everywhere

A uniform 8 by 8 rule for thousands of components can erase the JAX speed
gain.

**Control:** establish the converged reference first, then use error-controlled
moments and adaptive subdivision. Keep a forced-reference audit mode.

### Taylor expansion fails at non-smooth beam features

Nulls, raster edges, handovers, and tabulated interpolation cells can violate
the smoothness assumption.

**Control:** detect or declare these regions and force subdivision. Validate
moments against the reference rather than beam derivative size alone.

### Support masks bias the integral

Dropping an entire component or row because one node is unsupported loses
valid data. Treating unsupported off-diagonal zeros as measured leakage is
also wrong.

**Control:** propagate diagonal and off-diagonal validity per node. Sum valid
support and report partial-support components explicitly.

### Coordinate transforms change the sky between pointings

A pointing-local subdivision could create slightly different celestial
components for each field.

**Control:** define nodes once in the common celestial frame, then transform
them to each pointing. Add opposite-pointing and PA convention tests.

### Beam and calibration normalization are applied twice

The direction-independent beam centre or receptor normalization may already
be absorbed by calibration.

**Control:** preserve the voltage-beam normalization contract and calibration
state in the manifest. Test an on-axis point source before finite off-axis
components.

### Bright sidelobe sources dominate aggregate loss

This can hide main-lobe or polarisation behaviour.

**Control:** report loss by source region, pointing, beam radius, baseline,
channel, time, and hand. Keep the aggregate loss, but do not use it alone.

### Sky refinement and integration refinement are selected together

The optimizer could attribute a validation gain to the wrong change.

**Control:** freeze topology while accepting the operator. Reopen topology only
in a later, separately reported phase.

## Relationship to other plans

[`vla-beam-model-proposal.md`](vla-beam-model-proposal.md) defines the voltage
Jones evaluator, conventions, support, streaming by time, and scientific beam
validation. This proposal defines how that Jones is integrated with a finite
sky basis. Neither proposal is complete without the other.

[`hierarchical_pixels_proposal.md`](hierarchical_pixels_proposal.md) defines
when the data justify extra fitted spatial degrees of freedom. Its split and
merge rules remain the authority for sky refinement. Numerical subdivision in
this proposal does not invoke those rules because it adds no fitted freedom.

[`3c391_corner_pixels.md`](3c391_corner_pixels.md) records the evidence for the
central hierarchy, outer catalogue sources, extended Airy support, and the
coarse short-baseline field. It supplies the component ablations and sealed
protocol needed by the validation path.

[`time_frequency_sky_model_handoff.md`](time_frequency_sky_model_handoff.md)
describes the later time and frequency discovery programme. Beam-aware pixel
integration is a prerequisite where a direction-dependent numerical error
could mimic spectral or temporal structure.

[`calibration-prior-proposal.md`](calibration-prior-proposal.md) constrains
direction-independent calibration changes using learned physical priors. The
beam-aware operator reduces the pressure for those calibration terms to absorb
direction-dependent sky errors.

[`calibration-flag-proposal.md`](calibration-flag-proposal.md) distinguishes
dead data from potentially recoverable calibration failures. Beam integration
must preserve those flag classes and must not convert partial beam support into
a hardware-quality flag.

## Immediate next step

Implement the basis-preserving reference gate before another scientific beam
ranking.

The smallest useful slice is:

1. carry `basis_type` and `width_rad` from the sealed sky into the voltage
   operator;
2. keep catalogue atoms as deltas;
3. reproduce the existing scalar Airy quadtree prediction with the analytic
   parent square kernel;
4. add 2 by 2 and 4 by 4 tied subcell evaluation;
5. demonstrate constant-beam tiling identity and convergence on representative
   16 and 60 arcsec 3C391 components;
6. repeat the fixed-sky four-beam report with point, centre-square, and
   converged subcell variants.

Do not begin with the Taylor optimization. The subcell implementation is the
reference that tells us whether the Taylor method is correct and whether pixel
integration changes the beam conclusion at all.
