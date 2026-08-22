"""Quadtree split scores and exhaustive one-split evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import (
    InferenceConfig,
    QuadtreeInferenceResult,
    infer_quadtree,
)
from sl1mjax.objective import effective_weight, weighted_complex_mse
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    QuadtreeTopology,
    predict_quadtree_stokes_i,
    predict_quadtree_stokes_i_explicit,
)
from sl1mjax.sky import GaussianApproximation, raw_from_intensity


@dataclass(frozen=True)
class QuadtreeObjectiveMetrics:
    """Comparable objective terms evaluated at one fitted topology."""

    training_data: float
    sparsity: float
    topology: float
    objective: float
    holdout_data: float | None


@dataclass(frozen=True)
class SplitBaselineScore:
    """Cheap image-space scores for one currently active leaf."""

    leaf: QuadtreeLeaf
    flux: float
    surface_brightness: float
    gradient: float
    laplacian: float


@dataclass(frozen=True)
class ResidualHaarScore:
    """Residual projection and local curvature for three child contrasts."""

    leaf: QuadtreeLeaf
    parent_flux: float
    gradient: tuple[float, float, float]
    gram: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, float, float]
    eigenvalue_ratio: float
    ridge: float
    raw_predicted_improvement: float
    predicted_improvement: float
    eligible: bool
    curvature_mode: str = "exact"


@dataclass(frozen=True)
class BulkSplitSelection:
    """Dörfler-style score marking subject to a topology-growth budget."""

    selected: tuple[QuadtreeLeaf, ...]
    available_improvement: float
    selected_improvement: float
    covered_fraction: float
    split_budget: int
    added_leaf_count: int


@dataclass(frozen=True)
class RefinementAttempt:
    """One warm global refit of a proposed prefix of split candidates."""

    selected: tuple[QuadtreeLeaf, ...]
    fit: QuadtreeInferenceResult
    metrics: QuadtreeObjectiveMetrics
    training_relative_improvement: float
    holdout_relative_improvement: float
    accepted: bool


@dataclass(frozen=True)
class RefinementBatchResult:
    """Validation result for a split batch and any backtracked prefixes."""

    baseline: QuadtreeObjectiveMetrics
    attempts: tuple[RefinementAttempt, ...]

    @property
    def accepted_attempt(self) -> RefinementAttempt | None:
        """The accepted attempt, or ``None`` when all prefixes were rejected."""

        return next((attempt for attempt in self.attempts if attempt.accepted), None)


@dataclass(frozen=True)
class SplitRankingEntry:
    """One candidate's positions in the Haar and exhaustive rankings."""

    leaf: QuadtreeLeaf
    haar_rank: int
    oracle_rank: int
    haar_improvement: float
    oracle_improvement: float


@dataclass(frozen=True)
class HaarOracleComparison:
    """Direct rank comparison between residual/Haar scores and the oracle."""

    entries: tuple[SplitRankingEntry, ...]
    spearman_rho: float
    top1_match: bool


@dataclass(frozen=True)
class LocalSplitEvaluation:
    """Conditional four-child replacement with all other leaves fixed."""

    leaf: QuadtreeLeaf
    parent_flux: float
    children: tuple[QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf]
    child_flux: tuple[float, float, float, float]
    metrics: QuadtreeObjectiveMetrics
    objective_change: float
    predicted_improvement: float
    holdout_change: float | None

    @property
    def active_children(self) -> int:
        """Number of children with strictly positive fitted flux."""

        return sum(flux > 0 for flux in self.child_flux)


@dataclass(frozen=True)
class LocalSplitResult:
    """Baseline and conditional solutions for requested split candidates."""

    baseline: QuadtreeObjectiveMetrics
    evaluations: tuple[LocalSplitEvaluation, ...]
    flux_conserving: bool

    @property
    def ranked(self) -> tuple[LocalSplitEvaluation, ...]:
        """Candidates from greatest to least conditional improvement."""

        return tuple(
            sorted(
                self.evaluations,
                key=lambda evaluation: (-evaluation.predicted_improvement, evaluation.leaf),
            )
        )

    @property
    def best(self) -> LocalSplitEvaluation | None:
        """The best improving local replacement, if one exists."""

        ranked = self.ranked
        if not ranked or ranked[0].predicted_improvement <= 0:
            return None
        return ranked[0]


@dataclass(frozen=True)
class LookaheadRankingEntry:
    """One candidate's positions in local-lookahead and oracle rankings."""

    leaf: QuadtreeLeaf
    lookahead_rank: int
    oracle_rank: int
    lookahead_improvement: float
    oracle_improvement: float


@dataclass(frozen=True)
class LookaheadOracleComparison:
    """Direct rank comparison between local lookahead and the oracle."""

    entries: tuple[LookaheadRankingEntry, ...]
    spearman_rho: float
    top1_match: bool


@dataclass(frozen=True)
class SingleSplitEvaluation:
    """Globally refitted objective after replacing one parent by four children."""

    leaf: QuadtreeLeaf
    parent_flux: float
    metrics: QuadtreeObjectiveMetrics
    fit: QuadtreeInferenceResult
    objective_change: float
    predicted_improvement: float
    holdout_change: float | None


@dataclass(frozen=True)
class ExhaustiveSplitResult:
    """Baseline fit and deterministic evaluations of every requested split."""

    baseline: QuadtreeObjectiveMetrics
    evaluations: tuple[SingleSplitEvaluation, ...]

    @property
    def ranked(self) -> tuple[SingleSplitEvaluation, ...]:
        """Candidates from greatest to least training-objective improvement."""

        return tuple(
            sorted(
                self.evaluations,
                key=lambda evaluation: (-evaluation.predicted_improvement, evaluation.leaf),
            )
        )

    @property
    def best(self) -> SingleSplitEvaluation | None:
        """The best improving candidate, or ``None`` when every split is worse."""

        ranked = self.ranked
        if not ranked or ranked[0].predicted_improvement <= 0:
            return None
        return ranked[0]


def _aligned_flux(topology: QuadtreeTopology, flux: np.ndarray) -> np.ndarray:
    flux_array = np.asarray(flux, dtype=np.float64).reshape(-1)
    if flux_array.shape != (len(topology.leaves),):
        raise ValueError("flux must contain exactly one value per topology leaf")
    if not np.all(np.isfinite(flux_array)):
        raise ValueError("flux must be finite")
    if np.any(flux_array < 0):
        raise ValueError("flux must be non-negative")
    return flux_array


def quadtree_objective_metrics(
    block: VisibilityBlock,
    fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
) -> QuadtreeObjectiveMetrics:
    """Recompute comparable train, prior, and optional holdout terms."""

    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if holdout_mask is not None:
        if holdout_mask.shape != block.shape:
            raise ValueError("holdout_mask must match the visibility block")
        if np.any(train_mask & holdout_mask):
            raise ValueError("train_mask and holdout_mask must be disjoint")
        if not np.any(holdout_mask):
            raise ValueError("holdout_mask must contain active samples")
    training_data = float(
        weighted_complex_mse(
            fit.prediction,
            block.visibility,
            block.weight,
            ~train_mask,
        )
    )
    sparsity = float(config.sparsity_weight * np.sum(fit.flux))
    topology = float(fit.leaf_penalty * len(fit.topology.leaves))
    holdout_data = (
        None
        if holdout_mask is None
        else float(
            weighted_complex_mse(
                fit.prediction,
                block.visibility,
                block.weight,
                ~holdout_mask,
            )
        )
    )
    return QuadtreeObjectiveMetrics(
        training_data=training_data,
        sparsity=sparsity,
        topology=topology,
        objective=training_data + sparsity + topology,
        holdout_data=holdout_data,
    )


def render_quadtree_surface_brightness(
    topology: QuadtreeTopology,
    flux: np.ndarray,
    *,
    level: int | None = None,
) -> np.ndarray:
    """Render piecewise-constant leaf brightness on a chosen dyadic level.

    Values have units of integrated flux per square radian. Multiplying the
    returned sum by the render-pixel area recovers total represented flux.
    Missing regions of a sparse topology are rendered as zero brightness.
    """

    flux_array = _aligned_flux(topology, flux)
    highest_leaf_level = max((leaf.level for leaf in topology.leaves), default=0)
    render_level = highest_leaf_level + 1 if level is None else level
    if render_level < highest_leaf_level:
        raise ValueError("render level must be at least the deepest leaf level")
    if render_level < 0:
        raise ValueError("render level must be non-negative")
    size = topology.grid.leaves_per_axis(render_level)
    image = np.zeros((size, size), dtype=np.float64)
    for leaf, leaf_flux in zip(topology.leaves, flux_array, strict=True):
        scale = 2 ** (render_level - leaf.level)
        row = leaf.iy * scale
        column = leaf.ix * scale
        width = topology.grid.leaf_width_rad(leaf.level)
        image[row : row + scale, column : column + scale] = leaf_flux / width**2
    return image


def baseline_split_scores(
    topology: QuadtreeTopology,
    flux: np.ndarray,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    render_level: int | None = None,
) -> tuple[SplitBaselineScore, ...]:
    """Compute parent-flux and dense image-derivative baseline scores.

    The gradient and Laplacian operate on a surface-brightness rendering. Their
    RMS values over each leaf are multiplied by one and two powers of the leaf
    width respectively. This removes the derivative units and makes scores at
    different depths comparable to local surface brightness.
    """

    flux_array = _aligned_flux(topology, flux)
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    selected = topology.leaves if candidates is None else tuple(candidates)
    if len(set(selected)) != len(selected):
        raise ValueError("candidates must be unique")
    missing = [leaf for leaf in selected if leaf not in topology.leaves]
    if missing:
        raise ValueError(f"candidate leaves are not active: {missing}")
    selected = tuple(
        sorted(leaf for leaf in selected if max_depth is None or leaf.level < max_depth)
    )

    highest_leaf_level = max((leaf.level for leaf in topology.leaves), default=0)
    evaluation_level = highest_leaf_level + 1 if render_level is None else render_level
    image = render_quadtree_surface_brightness(
        topology,
        flux_array,
        level=evaluation_level,
    )
    spacing = topology.grid.leaf_width_rad(evaluation_level)
    if image.shape[0] < 2:
        gradient_squared = np.zeros_like(image)
        laplacian = np.zeros_like(image)
    else:
        derivative_m, derivative_l = np.gradient(image, spacing, edge_order=1)
        gradient_squared = derivative_l**2 + derivative_m**2
        laplacian = np.gradient(derivative_l, spacing, axis=1, edge_order=1) + np.gradient(
            derivative_m, spacing, axis=0, edge_order=1
        )

    flux_by_leaf = dict(zip(topology.leaves, flux_array, strict=True))
    scores = []
    for leaf in selected:
        scale = 2 ** (evaluation_level - leaf.level)
        row = leaf.iy * scale
        column = leaf.ix * scale
        region = np.s_[row : row + scale, column : column + scale]
        width = topology.grid.leaf_width_rad(leaf.level)
        scores.append(
            SplitBaselineScore(
                leaf=leaf,
                flux=float(flux_by_leaf[leaf]),
                surface_brightness=float(flux_by_leaf[leaf] / width**2),
                gradient=float(width * np.sqrt(np.mean(gradient_squared[region]))),
                laplacian=float(width**2 * np.sqrt(np.mean(laplacian[region] ** 2))),
            )
        )
    return tuple(scores)


# Rows are celestial NW, NE, SW, SE from QuadtreeLeaf.haar_children().
# Columns are hx (east-west), hy (north-south), and hd (diagonal).
_HAAR_CHILD_DETAILS = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


def _build_residual_haar_score(
    leaf: QuadtreeLeaf,
    parent_flux: float,
    gradient: np.ndarray,
    gram: np.ndarray,
    *,
    min_parent_flux: float,
    min_curvature: float,
    min_eigenvalue_ratio: float,
    ridge_relative: float,
    curvature_mode: str = "exact",
) -> ResidualHaarScore:
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    curvature_scale = max(float(np.trace(gram) / 3.0), np.finfo(np.float64).eps)
    ridge = ridge_relative * curvature_scale
    regularized_gram = gram + ridge * np.eye(3)
    raw_improvement = max(
        0.0,
        float(0.5 * gradient @ np.linalg.solve(regularized_gram, gradient)),
    )
    maximum_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    eigenvalue_ratio = (
        max(float(eigenvalues[0]), 0.0) / maximum_eigenvalue if maximum_eigenvalue > 0 else 0.0
    )
    eligible = bool(
        parent_flux >= min_parent_flux
        and maximum_eigenvalue >= min_curvature
        and eigenvalue_ratio >= min_eigenvalue_ratio
    )
    return ResidualHaarScore(
        leaf=leaf,
        parent_flux=parent_flux,
        gradient=(float(gradient[0]), float(gradient[1]), float(gradient[2])),
        gram=tuple(tuple(float(value) for value in row) for row in gram),
        eigenvalues=(
            float(eigenvalues[0]),
            float(eigenvalues[1]),
            float(eigenvalues[2]),
        ),
        eigenvalue_ratio=eigenvalue_ratio,
        ridge=ridge,
        raw_predicted_improvement=raw_improvement,
        predicted_improvement=raw_improvement if eligible else 0.0,
        eligible=eligible,
        curvature_mode=curvature_mode,
    )


def _validate_haar_parameters(
    min_parent_flux: float,
    min_curvature: float,
    min_eigenvalue_ratio: float,
    ridge_relative: float,
) -> None:
    thresholds = {
        "min_parent_flux": min_parent_flux,
        "min_curvature": min_curvature,
        "min_eigenvalue_ratio": min_eigenvalue_ratio,
        "ridge_relative": ridge_relative,
    }
    if any(not np.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("Haar thresholds and ridge must be finite and non-negative")
    if ridge_relative == 0:
        raise ValueError("ridge_relative must be positive")


def _candidate_leaves(
    topology: QuadtreeTopology,
    candidates: tuple[QuadtreeLeaf, ...] | None,
    max_depth: int | None,
) -> tuple[QuadtreeLeaf, ...]:
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    selected = topology.leaves if candidates is None else tuple(candidates)
    if len(set(selected)) != len(selected):
        raise ValueError("candidates must be unique")
    missing = [leaf for leaf in selected if leaf not in topology.leaves]
    if missing:
        raise ValueError(f"candidate leaves are not active: {missing}")
    return tuple(sorted(leaf for leaf in selected if max_depth is None or leaf.level < max_depth))


def _predict_quadtree_flux(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    flux: np.ndarray,
    config: InferenceConfig,
    *,
    fixed_gains: np.ndarray | None,
    primary_beam: VLAPrimaryBeam | None,
    approximation: GaussianApproximation,
) -> np.ndarray:
    l, m = topology.centers()
    beam_i, beam_rr, beam_ll = predict_beam_weights(
        primary_beam,
        l,
        m,
        block.frequency_hz,
    )
    if config.operator_mode == "explicit":
        prediction = predict_quadtree_stokes_i_explicit(
            flux,
            topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            approximation=approximation,
            fixed_gains=fixed_gains,
            config=config.direct_dft,
            beam_weights=beam_i,
            beam_weights_rr=beam_rr,
            beam_weights_ll=beam_ll,
        )
    else:
        prediction = predict_quadtree_stokes_i(
            flux,
            topology,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            approximation=approximation,
            fixed_gains=fixed_gains,
            chunk_size=config.chunk_size,
            beam_weights=beam_i,
            beam_weights_rr=beam_rr,
            beam_weights_ll=beam_ll,
        )
    return np.asarray(prediction)


def _child_detail_responses(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    leaf: QuadtreeLeaf,
    config: InferenceConfig,
    *,
    fixed_gains: np.ndarray | None,
    primary_beam: VLAPrimaryBeam | None,
    approximation: GaussianApproximation,
) -> np.ndarray:
    children = leaf.haar_children()
    child_topology = QuadtreeTopology(topology.grid, children)
    detail_by_child = dict(zip(children, _HAAR_CHILD_DETAILS, strict=True))
    details = np.asarray(
        [detail_by_child[child] for child in child_topology.leaves],
        dtype=np.float64,
    )
    responses = [
        _predict_quadtree_flux(
            block,
            child_topology,
            detail,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        ).reshape(-1)
        for detail in details.T
    ]
    return np.stack(responses, axis=1)


def residual_haar_scores(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    min_parent_flux: float = 0.0,
    min_curvature: float = 0.0,
    min_eigenvalue_ratio: float = 1e-8,
    ridge_relative: float = 1e-8,
) -> tuple[ResidualHaarScore, ...]:
    """Estimate improvement from each leaf's three child-detail directions.

    The score uses the same normalized weighted complex MSE as inference. It
    evaluates signed virtual children but does not change the active topology.
    The three details preserve total flux, so the L1 term has zero directional
    derivative. Positivity and parent/child kernel mismatch remain the job of
    the subsequent four-child lookahead.
    """

    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if current_fit.prediction.shape != block.shape or current_fit.residual.shape != block.shape:
        raise ValueError("current_fit predictions must match the visibility block")
    _validate_haar_parameters(
        min_parent_flux,
        min_curvature,
        min_eigenvalue_ratio,
        ridge_relative,
    )

    approximation = GaussianApproximation(approximation)
    selected = _candidate_leaves(current_fit.topology, candidates, max_depth)
    flux_array = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux_array, strict=True))
    active_weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~train_mask),
        dtype=np.float64,
    ).reshape(-1)
    weight_sum = float(np.sum(active_weight))
    if weight_sum <= 0:
        raise ValueError("train_mask must contain positive-weight finite samples")
    residual = np.asarray(current_fit.residual).reshape(-1)
    residual = np.where(active_weight > 0, residual, 0.0)
    if not np.all(np.isfinite(residual)):
        raise ValueError("active current_fit residual samples must be finite")

    scores = []
    for leaf in selected:
        responses = _child_detail_responses(
            block,
            current_fit.topology,
            leaf,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        )
        weighted_responses = active_weight[:, None] * responses
        gradient = (2.0 / weight_sum) * np.real(responses.conj().T @ (active_weight * residual))
        gram = (2.0 / weight_sum) * np.real(responses.conj().T @ weighted_responses)
        parent_flux = float(flux_by_leaf[leaf])
        scores.append(
            _build_residual_haar_score(
                leaf,
                parent_flux,
                gradient,
                gram,
                min_parent_flux=min_parent_flux,
                min_curvature=min_curvature,
                min_eigenvalue_ratio=min_eigenvalue_ratio,
                ridge_relative=ridge_relative,
            )
        )
    return tuple(scores)


def batched_residual_haar_scores(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    min_parent_flux: float = 0.0,
    min_curvature: float = 0.0,
    min_eigenvalue_ratio: float = 1e-8,
    ridge_relative: float = 1e-8,
    allow_approximate_curvature: bool = False,
) -> tuple[ResidualHaarScore, ...]:
    """Screen all candidates with one streamed residual adjoint per level.

    Child residual correlations are exact. The local Gram matrix is evaluated
    once for a representative parent at each level. This curvature is exactly
    translation-invariant only for the paraxial square basis without a primary
    beam. Wide-field geometry and direction-dependent beam weights break that
    invariance, so they require ``allow_approximate_curvature=True``.
    :func:`reconstruct_hierarchical` enables that automatically when the
    kernel is not translation-invariant and then rescores the shortlist with
    :func:`residual_haar_scores`.
    """

    if config.operator_mode != "explicit":
        raise ValueError("batched Haar scoring requires operator_mode='explicit'")
    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if current_fit.prediction.shape != block.shape or current_fit.residual.shape != block.shape:
        raise ValueError("current_fit predictions must match the visibility block")
    _validate_haar_parameters(
        min_parent_flux,
        min_curvature,
        min_eigenvalue_ratio,
        ridge_relative,
    )
    approximation = GaussianApproximation(approximation)
    curvature_is_exact = approximation is GaussianApproximation.PARAXIAL and primary_beam is None
    if not curvature_is_exact and not allow_approximate_curvature:
        raise ValueError(
            "shared per-level curvature is exact only for paraxial pixels without "
            "a primary beam; pass allow_approximate_curvature=True and rescore the "
            "shortlist exactly"
        )

    selected = _candidate_leaves(current_fit.topology, candidates, max_depth)
    if not selected:
        return ()
    flux_array = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux_array, strict=True))
    active_weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~train_mask),
        dtype=np.float64,
    ).reshape(-1)
    weight_sum = float(np.sum(active_weight))
    if weight_sum <= 0:
        raise ValueError("train_mask must contain positive-weight finite samples")
    residual = np.asarray(current_fit.residual).reshape(-1)
    residual = np.where(active_weight > 0, residual, 0.0)
    if not np.all(np.isfinite(residual)):
        raise ValueError("active current_fit residual samples must be finite")

    selected_by_level: dict[int, list[QuadtreeLeaf]] = {}
    for leaf in selected:
        selected_by_level.setdefault(leaf.level, []).append(leaf)
    score_by_leaf: dict[QuadtreeLeaf, ResidualHaarScore] = {}
    real_dtype = config.direct_dft.real_dtype
    complex_dtype = config.direct_dft.complex_dtype
    weight_jax = jnp.asarray(active_weight.reshape(block.shape), dtype=real_dtype)
    residual_jax = jnp.asarray(residual.reshape(block.shape), dtype=complex_dtype)

    for parent_level, level_leaves_list in sorted(selected_by_level.items()):
        level_leaves = tuple(level_leaves_list)
        all_children = tuple(child for leaf in level_leaves for child in leaf.children())
        virtual_topology = QuadtreeTopology(current_fit.topology.grid, all_children)
        child_index = {child: index for index, child in enumerate(virtual_topology.leaves)}
        l, m = virtual_topology.centers()
        beam_i, beam_rr, beam_ll = predict_beam_weights(
            primary_beam,
            l,
            m,
            block.frequency_hz,
        )
        beam_i_jax = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
        beam_rr_jax = None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
        beam_ll_jax = None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)

        def residual_projection(
            child_flux: jax.Array,
            *,
            topology: QuadtreeTopology = virtual_topology,
            level_beam_i: jax.Array | None = beam_i_jax,
            level_beam_rr: jax.Array | None = beam_rr_jax,
            level_beam_ll: jax.Array | None = beam_ll_jax,
        ) -> jax.Array:
            prediction = predict_quadtree_stokes_i_explicit(
                child_flux,
                topology,
                block.uvw_m,
                block.frequency_hz,
                block.antenna1,
                block.antenna2,
                block.correlations,
                approximation=approximation,
                fixed_gains=fixed_gains,
                config=config.direct_dft,
                beam_weights=level_beam_i,
                beam_weights_rr=level_beam_rr,
                beam_weights_ll=level_beam_ll,
            )
            inner_product = jnp.sum(weight_jax * jnp.real(jnp.conj(prediction) * residual_jax))
            return (2.0 / weight_sum) * inner_product

        child_gradient = np.asarray(
            jax.jit(jax.grad(residual_projection))(
                jnp.zeros(len(virtual_topology.leaves), dtype=real_dtype)
            ),
            dtype=np.float64,
        )
        representative_responses = _child_detail_responses(
            block,
            current_fit.topology,
            level_leaves[0],
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        )
        representative_gram = (2.0 / weight_sum) * np.real(
            representative_responses.conj().T @ (active_weight[:, None] * representative_responses)
        )
        curvature_mode = (
            f"per_level_exact:{parent_level}"
            if curvature_is_exact
            else f"per_level_approximate:{parent_level}"
        )
        for leaf in level_leaves:
            indices = np.asarray(
                [child_index[child] for child in leaf.haar_children()]
            )
            gradient = child_gradient[indices] @ _HAAR_CHILD_DETAILS
            score_by_leaf[leaf] = _build_residual_haar_score(
                leaf,
                float(flux_by_leaf[leaf]),
                gradient,
                representative_gram,
                min_parent_flux=min_parent_flux,
                min_curvature=min_curvature,
                min_eigenvalue_ratio=min_eigenvalue_ratio,
                ridge_relative=ridge_relative,
                curvature_mode=curvature_mode,
            )
    return tuple(score_by_leaf[leaf] for leaf in selected)


def select_bulk_splits(
    scores: tuple[ResidualHaarScore, ...],
    current_leaf_count: int,
    *,
    target_improvement_fraction: float = 0.7,
    max_split_fraction: float = 0.05,
    max_splits: int | None = None,
    min_improvement: float = 0.0,
    split_cost: float = 0.0,
) -> BulkSplitSelection:
    """Select a score-dominant prefix while bounding quadtree growth.

    Each quadtree split replaces one leaf with four and therefore adds three
    active leaves. The fractional budget applies to parent splits, not final
    leaf growth: a 5% split budget grows the topology by at most about 15%.
    """

    if current_leaf_count < 1:
        raise ValueError("current_leaf_count must be positive")
    if not 0 < target_improvement_fraction <= 1:
        raise ValueError("target_improvement_fraction must be in (0, 1]")
    if not 0 < max_split_fraction <= 1:
        raise ValueError("max_split_fraction must be in (0, 1]")
    if max_splits is not None and max_splits < 1:
        raise ValueError("max_splits must be positive")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and non-negative")
    if not np.isfinite(split_cost) or split_cost < 0:
        raise ValueError("split_cost must be finite and non-negative")
    if len({score.leaf for score in scores}) != len(scores):
        raise ValueError("scores must contain unique leaves")

    eligible_scores = tuple(
        (score, score.predicted_improvement - split_cost)
        for score in scores
        if score.eligible and score.predicted_improvement - split_cost > min_improvement
    )
    ranked = tuple(
        sorted(
            eligible_scores,
            key=lambda item: (-item[1], item[0].leaf),
        )
    )
    available = float(sum(net_improvement for _, net_improvement in ranked))
    fractional_budget = max(1, int(np.ceil(max_split_fraction * current_leaf_count)))
    split_budget = fractional_budget if max_splits is None else min(fractional_budget, max_splits)
    selected = []
    selected_improvement = 0.0
    target = target_improvement_fraction * available
    for score, net_improvement in ranked[:split_budget]:
        selected.append(score.leaf)
        selected_improvement += net_improvement
        if selected_improvement >= target:
            break
    covered_fraction = selected_improvement / available if available > 0 else 0.0
    return BulkSplitSelection(
        selected=tuple(selected),
        available_improvement=available,
        selected_improvement=selected_improvement,
        covered_fraction=covered_fraction,
        split_budget=split_budget,
        added_leaf_count=3 * len(selected),
    )


def _relative_improvement(baseline: float, proposal: float) -> float:
    scale = max(abs(baseline), np.finfo(np.float64).eps)
    return float((baseline - proposal) / scale)


def _split_batch_initial_sky(
    current_fit: QuadtreeInferenceResult,
    selected: tuple[QuadtreeLeaf, ...],
) -> QuadtreeSky:
    sky = QuadtreeSky(
        current_fit.topology.grid,
        current_fit.topology.leaves,
        _aligned_flux(current_fit.topology, current_fit.flux),
    )
    for leaf in selected:
        sky = sky.split(leaf)
    return sky


def refine_quadtree_batch(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    config: InferenceConfig,
    selected: tuple[QuadtreeLeaf, ...],
    *,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    minimum_training_relative_improvement: float = 0.0,
    minimum_holdout_relative_improvement: float = 0.0,
    max_refits: int = 4,
) -> RefinementBatchResult:
    """Warm-refit and validate a ranked split batch, halving it on rejection.

    ``selected`` must be ordered from strongest to weakest. Every attempt
    starts from the same accepted fit. Parent flux is divided equally among
    children, then all leaf fluxes are optimized together. A proposal must
    improve both the penalized training objective and held-out loss by the
    requested relative tolerances. Rejected batches are backtracked to their
    strongest half, up to ``max_refits`` global optimizations.
    """

    if not selected:
        raise ValueError("selected must contain at least one leaf")
    if len(set(selected)) != len(selected):
        raise ValueError("selected leaves must be unique")
    missing = [leaf for leaf in selected if leaf not in current_fit.topology.leaves]
    if missing:
        raise ValueError(f"selected leaves are not active: {missing}")
    if train_mask.shape != block.shape or holdout_mask.shape != block.shape:
        raise ValueError("train_mask and holdout_mask must match the visibility block")
    if np.any(train_mask & holdout_mask):
        raise ValueError("train_mask and holdout_mask must be disjoint")
    if not np.any(holdout_mask):
        raise ValueError("holdout_mask must contain active samples")
    tolerances = (
        minimum_training_relative_improvement,
        minimum_holdout_relative_improvement,
    )
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("relative improvement thresholds must be finite and non-negative")
    if max_refits < 1:
        raise ValueError("max_refits must be positive")

    approximation = GaussianApproximation(approximation)
    baseline = quadtree_objective_metrics(
        block,
        current_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    if baseline.holdout_data is None:
        raise RuntimeError("holdout metrics were not computed")

    attempts: list[RefinementAttempt] = []
    batch_size = len(selected)
    while len(attempts) < max_refits:
        attempt_leaves = selected[:batch_size]
        initial_sky = _split_batch_initial_sky(current_fit, attempt_leaves)
        fit = infer_quadtree(
            block,
            initial_sky.topology,
            train_mask,
            config,
            holdout_mask=holdout_mask,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            leaf_penalty=current_fit.leaf_penalty,
            initial_flux=initial_sky.flux,
        )
        metrics = quadtree_objective_metrics(
            block,
            fit,
            train_mask,
            config,
            holdout_mask=holdout_mask,
        )
        if metrics.holdout_data is None:
            raise RuntimeError("holdout metrics were not computed")
        training_improvement = _relative_improvement(
            baseline.objective,
            metrics.objective,
        )
        holdout_improvement = _relative_improvement(
            baseline.holdout_data,
            metrics.holdout_data,
        )
        accepted = bool(
            np.isfinite(training_improvement)
            and np.isfinite(holdout_improvement)
            and training_improvement > minimum_training_relative_improvement
            and holdout_improvement > minimum_holdout_relative_improvement
        )
        attempts.append(
            RefinementAttempt(
                selected=attempt_leaves,
                fit=fit,
                metrics=metrics,
                training_relative_improvement=training_improvement,
                holdout_relative_improvement=holdout_improvement,
                accepted=accepted,
            )
        )
        if accepted or batch_size == 1:
            break
        batch_size = max(1, batch_size // 2)

    return RefinementBatchResult(
        baseline=baseline,
        attempts=tuple(attempts),
    )


def compare_haar_to_oracle(
    scores: tuple[ResidualHaarScore, ...],
    oracle: ExhaustiveSplitResult,
) -> HaarOracleComparison:
    """Compare deterministic score ranks over an identical candidate set."""

    if not scores:
        raise ValueError("scores must contain at least one candidate")
    score_by_leaf = {score.leaf: score for score in scores}
    oracle_by_leaf = {evaluation.leaf: evaluation for evaluation in oracle.evaluations}
    if len(score_by_leaf) != len(scores):
        raise ValueError("scores must contain unique leaves")
    if len(oracle_by_leaf) != len(oracle.evaluations):
        raise ValueError("oracle must contain unique leaves")
    if set(score_by_leaf) != set(oracle_by_leaf):
        raise ValueError("Haar scores and oracle must contain the same leaves")

    haar_order = sorted(
        scores,
        key=lambda score: (-score.predicted_improvement, score.leaf),
    )
    oracle_order = oracle.ranked
    haar_rank = {score.leaf: rank for rank, score in enumerate(haar_order, start=1)}
    oracle_rank = {evaluation.leaf: rank for rank, evaluation in enumerate(oracle_order, start=1)}
    entries = tuple(
        SplitRankingEntry(
            leaf=leaf,
            haar_rank=haar_rank[leaf],
            oracle_rank=oracle_rank[leaf],
            haar_improvement=score_by_leaf[leaf].predicted_improvement,
            oracle_improvement=oracle_by_leaf[leaf].predicted_improvement,
        )
        for leaf in sorted(score_by_leaf, key=oracle_rank.__getitem__)
    )
    count = len(entries)
    squared_rank_difference = sum((entry.haar_rank - entry.oracle_rank) ** 2 for entry in entries)
    spearman_rho = (
        1.0 if count == 1 else 1.0 - 6.0 * squared_rank_difference / (count * (count**2 - 1))
    )
    return HaarOracleComparison(
        entries=entries,
        spearman_rho=float(spearman_rho),
        top1_match=haar_order[0].leaf == oracle_order[0].leaf,
    )


def _child_response_matrix(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    leaf: QuadtreeLeaf,
    config: InferenceConfig,
    *,
    fixed_gains: np.ndarray | None,
    primary_beam: VLAPrimaryBeam | None,
    approximation: GaussianApproximation,
) -> tuple[QuadtreeTopology, np.ndarray]:
    child_topology = QuadtreeTopology(topology.grid, leaf.children())
    responses = [
        _predict_quadtree_flux(
            block,
            child_topology,
            unit_flux,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        ).reshape(-1)
        for unit_flux in np.eye(4, dtype=np.float64)
    ]
    return child_topology, np.stack(responses, axis=1)


def _solve_nonnegative_quadratic(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    total: float | None,
) -> np.ndarray:
    """Solve a small convex QP by enumerating its active faces."""

    parameter_count = linear.size
    best_value = np.inf
    best: np.ndarray | None = None
    for bitmask in range(1 << parameter_count):
        active = np.asarray([bool(bitmask & (1 << index)) for index in range(parameter_count)])
        active_count = int(np.sum(active))
        if active_count == 0:
            if total is not None and total > 0:
                continue
            candidate = np.zeros(parameter_count, dtype=np.float64)
        elif total is None:
            face_hessian = hessian[np.ix_(active, active)]
            face_linear = linear[active]
            active_flux, _, _, _ = np.linalg.lstsq(
                face_hessian,
                -face_linear,
                rcond=None,
            )
            stationarity = face_hessian @ active_flux + face_linear
            scale = max(1.0, float(np.linalg.norm(face_linear, ord=np.inf)))
            if float(np.linalg.norm(stationarity, ord=np.inf)) > 1e-9 * scale:
                continue
            candidate = np.zeros(parameter_count, dtype=np.float64)
            candidate[active] = active_flux
        else:
            face_hessian = hessian[np.ix_(active, active)]
            system = np.block(
                [
                    [face_hessian, np.ones((active_count, 1))],
                    [np.ones((1, active_count)), np.zeros((1, 1))],
                ]
            )
            right_hand_side = np.concatenate((-linear[active], [total]))
            solution, _, _, _ = np.linalg.lstsq(system, right_hand_side, rcond=None)
            scale = max(1.0, float(np.linalg.norm(right_hand_side, ord=np.inf)))
            residual_norm = float(np.linalg.norm(system @ solution - right_hand_side, ord=np.inf))
            if residual_norm > 1e-9 * scale:
                continue
            candidate = np.zeros(parameter_count, dtype=np.float64)
            candidate[active] = solution[:-1]

        tolerance = 1e-10 * max(
            1.0,
            0.0 if total is None else total,
            float(np.linalg.norm(candidate, ord=np.inf)),
        )
        if np.any(candidate < -tolerance):
            continue
        candidate = np.maximum(candidate, 0.0)
        candidate[candidate < tolerance] = 0.0
        if total is not None:
            candidate_sum = float(np.sum(candidate))
            if total == 0:
                candidate[:] = 0.0
            elif candidate_sum > 0:
                candidate *= total / candidate_sum
            else:
                continue
        value = float(0.5 * candidate @ hessian @ candidate + linear @ candidate)
        if value < best_value:
            best_value = value
            best = candidate
    if best is None:
        raise RuntimeError("non-negative four-child quadratic has no feasible solution")
    return best


def _weighted_residual_power(residual: np.ndarray, weight: np.ndarray) -> float:
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0:
        raise ValueError("evaluation mask must contain positive-weight finite samples")
    finite_residual = np.where(weight > 0, residual, 0.0)
    if not np.all(np.isfinite(finite_residual)):
        raise ValueError("active residual samples must be finite")
    return float(np.sum(weight * np.abs(finite_residual) ** 2) / weight_sum)


def solve_quadtree_flux_active_set(
    block: VisibilityBlock,
    topology: QuadtreeTopology,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    leaf_penalty: float = 0.0,
    max_leaves: int = 12,
) -> QuadtreeInferenceResult:
    """Exactly fit a small fixed topology by enumerating active leaf sets.

    This solver is intended for validation oracles. Its cost grows as
    ``2**len(topology.leaves)``, so ``max_leaves`` prevents accidental use on
    production trees. Unlike the Optax path, it solves the linear non-negative
    flux objective to numerical precision and cannot confuse optimizer progress
    with improvement from a topology change.
    """

    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if holdout_mask is not None:
        if holdout_mask.shape != block.shape:
            raise ValueError("holdout_mask must match the visibility block")
        if np.any(train_mask & holdout_mask):
            raise ValueError("train_mask and holdout_mask must be disjoint")
    if not topology.leaves:
        raise ValueError("topology must contain at least one leaf")
    if len(topology.leaves) > max_leaves:
        raise ValueError(
            f"active-set fit supports at most {max_leaves} leaves; received {len(topology.leaves)}"
        )
    if config.smoothness_weight != 0:
        raise ValueError("smoothness_weight is not defined for quadtree inference")
    if not np.isfinite(config.sparsity_weight) or config.sparsity_weight < 0:
        raise ValueError("sparsity_weight must be finite and non-negative")
    if not np.isfinite(leaf_penalty) or leaf_penalty < 0:
        raise ValueError("leaf_penalty must be finite and non-negative")

    approximation = GaussianApproximation(approximation)
    responses = [
        _predict_quadtree_flux(
            block,
            topology,
            unit_flux,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        ).reshape(-1)
        for unit_flux in np.eye(len(topology.leaves), dtype=np.float64)
    ]
    response_matrix = np.stack(responses, axis=1)
    training_weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~train_mask),
        dtype=np.float64,
    ).reshape(-1)
    weight_sum = float(np.sum(training_weight))
    if weight_sum <= 0:
        raise ValueError("train_mask must contain positive-weight finite samples")
    observation = np.asarray(block.visibility).reshape(-1)
    zero_model_residual = np.where(training_weight > 0, -observation, 0.0)
    weighted_responses = training_weight[:, None] * response_matrix
    hessian = (2.0 / weight_sum) * np.real(response_matrix.conj().T @ weighted_responses)
    hessian = 0.5 * (hessian + hessian.T)
    linear = (2.0 / weight_sum) * np.real(
        response_matrix.conj().T @ (training_weight * zero_model_residual)
    ) + config.sparsity_weight
    flux = _solve_nonnegative_quadratic(hessian, linear, total=None)
    prediction = (response_matrix @ flux).reshape(block.shape)
    residual = prediction - block.visibility
    training_data = _weighted_residual_power(residual.reshape(-1), training_weight)
    topology_penalty = float(leaf_penalty * len(topology.leaves))
    prior = float(config.sparsity_weight * np.sum(flux) + topology_penalty)
    objective = training_data + prior
    holdout_data = None
    if holdout_mask is not None:
        holdout_weight = np.asarray(
            effective_weight(block.visibility, block.weight, ~holdout_mask),
            dtype=np.float64,
        ).reshape(-1)
        holdout_data = _weighted_residual_power(residual.reshape(-1), holdout_weight)
    return QuadtreeInferenceResult(
        topology=topology,
        flux=flux,
        prediction=prediction,
        residual=residual,
        raw_parameters=np.asarray(raw_from_intensity(flux)),
        optimizer_state=None,
        objective_history=(objective,),
        data_history=(training_data,),
        prior_history=(prior,),
        holdout_history=() if holdout_data is None else (holdout_data,),
        holdout_steps=() if holdout_data is None else (0,),
        leaf_penalty=float(leaf_penalty),
        topology_penalty=topology_penalty,
        best_step=0,
        steps=0,
        converged=True,
    )


def local_four_child_lookahead(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    conserve_parent_flux: bool = False,
) -> LocalSplitResult:
    """Solve each exact conditional parent-to-four-child replacement.

    Every other fitted leaf remains fixed. The candidate solve includes the
    actual parent and child visibility kernels, child non-negativity, L1 flux
    change, and the cost of three additional leaves. With
    ``conserve_parent_flux=True``, the four child fluxes must sum to the parent
    flux, so the L1 term cancels and only spatial detail is tested.
    """

    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if current_fit.prediction.shape != block.shape or current_fit.residual.shape != block.shape:
        raise ValueError("current_fit predictions must match the visibility block")
    if not np.isfinite(config.sparsity_weight) or config.sparsity_weight < 0:
        raise ValueError("sparsity_weight must be finite and non-negative")
    approximation = GaussianApproximation(approximation)
    selected = _candidate_leaves(current_fit.topology, candidates, max_depth)
    baseline = quadtree_objective_metrics(
        block,
        current_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    flux = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux, strict=True))
    total_flux = float(np.sum(flux))
    residual = np.asarray(current_fit.residual).reshape(-1)
    training_weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~train_mask),
        dtype=np.float64,
    ).reshape(-1)
    training_weight_sum = float(np.sum(training_weight))
    if training_weight_sum <= 0:
        raise ValueError("train_mask must contain positive-weight finite samples")
    holdout_weight = (
        None
        if holdout_mask is None
        else np.asarray(
            effective_weight(block.visibility, block.weight, ~holdout_mask),
            dtype=np.float64,
        ).reshape(-1)
    )

    evaluations = []
    for leaf in selected:
        parent_flux = float(flux_by_leaf[leaf])
        parent_topology = QuadtreeTopology(current_fit.topology.grid, (leaf,))
        parent_response = _predict_quadtree_flux(
            block,
            parent_topology,
            np.ones(1, dtype=np.float64),
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        ).reshape(-1)
        child_topology, child_responses = _child_response_matrix(
            block,
            current_fit.topology,
            leaf,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        )
        residual_without_parent = residual - parent_flux * parent_response
        training_residual_without_parent = np.where(
            training_weight > 0,
            residual_without_parent,
            0.0,
        )
        weighted_children = training_weight[:, None] * child_responses
        hessian = (2.0 / training_weight_sum) * np.real(
            child_responses.conj().T @ weighted_children
        )
        hessian = 0.5 * (hessian + hessian.T)
        linear = (2.0 / training_weight_sum) * np.real(
            child_responses.conj().T @ (training_weight * training_residual_without_parent)
        ) + config.sparsity_weight
        child_flux = _solve_nonnegative_quadratic(
            hessian,
            linear,
            total=parent_flux if conserve_parent_flux else None,
        )
        candidate_residual = residual_without_parent + child_responses @ child_flux
        training_data = _weighted_residual_power(candidate_residual, training_weight)
        candidate_total_flux = total_flux - parent_flux + float(np.sum(child_flux))
        sparsity = float(config.sparsity_weight * candidate_total_flux)
        topology = float(current_fit.leaf_penalty * (len(current_fit.topology.leaves) + 3))
        holdout_data = (
            None
            if holdout_weight is None
            else _weighted_residual_power(candidate_residual, holdout_weight)
        )
        metrics = QuadtreeObjectiveMetrics(
            training_data=training_data,
            sparsity=sparsity,
            topology=topology,
            objective=training_data + sparsity + topology,
            holdout_data=holdout_data,
        )
        objective_change = metrics.objective - baseline.objective
        holdout_change = (
            None
            if baseline.holdout_data is None or holdout_data is None
            else holdout_data - baseline.holdout_data
        )
        evaluations.append(
            LocalSplitEvaluation(
                leaf=leaf,
                parent_flux=parent_flux,
                children=leaf.children(),
                child_flux=(
                    float(child_flux[0]),
                    float(child_flux[1]),
                    float(child_flux[2]),
                    float(child_flux[3]),
                ),
                metrics=metrics,
                objective_change=objective_change,
                predicted_improvement=-objective_change,
                holdout_change=holdout_change,
            )
        )
    return LocalSplitResult(
        baseline=baseline,
        evaluations=tuple(evaluations),
        flux_conserving=conserve_parent_flux,
    )


def compare_lookahead_to_oracle(
    lookahead: LocalSplitResult,
    oracle: ExhaustiveSplitResult,
) -> LookaheadOracleComparison:
    """Compare deterministic local-lookahead and global-refit oracle ranks."""

    if not lookahead.evaluations:
        raise ValueError("lookahead must contain at least one candidate")
    local_by_leaf = {evaluation.leaf: evaluation for evaluation in lookahead.evaluations}
    oracle_by_leaf = {evaluation.leaf: evaluation for evaluation in oracle.evaluations}
    if len(local_by_leaf) != len(lookahead.evaluations):
        raise ValueError("lookahead must contain unique leaves")
    if len(oracle_by_leaf) != len(oracle.evaluations):
        raise ValueError("oracle must contain unique leaves")
    if set(local_by_leaf) != set(oracle_by_leaf):
        raise ValueError("lookahead and oracle must contain the same leaves")

    local_order = lookahead.ranked
    oracle_order = oracle.ranked
    local_rank = {evaluation.leaf: rank for rank, evaluation in enumerate(local_order, start=1)}
    oracle_rank = {evaluation.leaf: rank for rank, evaluation in enumerate(oracle_order, start=1)}
    entries = tuple(
        LookaheadRankingEntry(
            leaf=leaf,
            lookahead_rank=local_rank[leaf],
            oracle_rank=oracle_rank[leaf],
            lookahead_improvement=local_by_leaf[leaf].predicted_improvement,
            oracle_improvement=oracle_by_leaf[leaf].predicted_improvement,
        )
        for leaf in sorted(local_by_leaf, key=oracle_rank.__getitem__)
    )
    count = len(entries)
    squared_rank_difference = sum(
        (entry.lookahead_rank - entry.oracle_rank) ** 2 for entry in entries
    )
    spearman_rho = (
        1.0 if count == 1 else 1.0 - 6.0 * squared_rank_difference / (count * (count**2 - 1))
    )
    return LookaheadOracleComparison(
        entries=entries,
        spearman_rho=float(spearman_rho),
        top1_match=local_order[0].leaf == oracle_order[0].leaf,
    )


def exhaustive_single_split_oracle(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    solver: Literal["optax", "active_set"] = "optax",
    active_set_max_leaves: int = 12,
) -> ExhaustiveSplitResult:
    """Globally refit every requested legal split on the same data.

    This is intentionally expensive and intended as a small-problem oracle.
    The default resets Optax and globally reoptimizes every leaf. The
    ``active_set`` solver instead enumerates all non-negative supports and is
    exact to numerical precision, but scales exponentially with leaf count. A
    supplied holdout mask is evaluated after fitting and does not affect the
    training solve. Objective changes include L1 flux and the current fit's
    per-leaf complexity penalty.
    """

    if current_fit.topology.leaves == ():
        raise ValueError("current_fit topology must contain at least one leaf")
    if solver not in {"optax", "active_set"}:
        raise ValueError("solver must be optax or active_set")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    selected = current_fit.topology.leaves if candidates is None else tuple(candidates)
    if len(set(selected)) != len(selected):
        raise ValueError("candidates must be unique")
    missing = [leaf for leaf in selected if leaf not in current_fit.topology.leaves]
    if missing:
        raise ValueError(f"candidate leaves are not active: {missing}")
    selected = tuple(
        sorted(leaf for leaf in selected if max_depth is None or leaf.level < max_depth)
    )

    baseline_fit = (
        current_fit
        if solver == "optax"
        else solve_quadtree_flux_active_set(
            block,
            current_fit.topology,
            train_mask,
            config,
            holdout_mask=holdout_mask,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            leaf_penalty=current_fit.leaf_penalty,
            max_leaves=active_set_max_leaves,
        )
    )
    baseline = quadtree_objective_metrics(
        block,
        baseline_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    sky = QuadtreeSky(
        baseline_fit.topology.grid,
        baseline_fit.topology.leaves,
        baseline_fit.flux,
    )
    flux_by_leaf = dict(zip(sky.leaves, sky.flux, strict=True))
    evaluations = []
    for leaf in selected:
        split_sky = sky.split(leaf)
        split_fit = (
            infer_quadtree(
                block,
                split_sky.topology,
                train_mask,
                config,
                fixed_gains=fixed_gains,
                primary_beam=primary_beam,
                approximation=approximation,
                leaf_penalty=current_fit.leaf_penalty,
                initial_flux=split_sky.flux,
            )
            if solver == "optax"
            else solve_quadtree_flux_active_set(
                block,
                split_sky.topology,
                train_mask,
                config,
                holdout_mask=holdout_mask,
                fixed_gains=fixed_gains,
                primary_beam=primary_beam,
                approximation=approximation,
                leaf_penalty=current_fit.leaf_penalty,
                max_leaves=active_set_max_leaves,
            )
        )
        metrics = quadtree_objective_metrics(
            block,
            split_fit,
            train_mask,
            config,
            holdout_mask=holdout_mask,
        )
        objective_change = metrics.objective - baseline.objective
        holdout_change = (
            None
            if baseline.holdout_data is None or metrics.holdout_data is None
            else metrics.holdout_data - baseline.holdout_data
        )
        evaluations.append(
            SingleSplitEvaluation(
                leaf=leaf,
                parent_flux=float(flux_by_leaf[leaf]),
                metrics=metrics,
                fit=split_fit,
                objective_change=objective_change,
                predicted_improvement=-objective_change,
                holdout_change=holdout_change,
            )
        )
    return ExhaustiveSplitResult(baseline=baseline, evaluations=tuple(evaluations))


# --- Coarsening (merge) with hysteresis ---
#
# The reverse of the split path above. A merge candidate is a complete
# four-sibling group that could be replaced by its parent. Scoring solves
# the exact reverse of the split lookahead's four-variable problem: a single
# non-negative variable (the merged parent's flux) replacing the four
# children's actual fitted response, with every other leaf held fixed.
# Acceptance reuses `RefinementBatchResult`/`RefinementAttempt` from the
# split path unchanged; only the batch's initial topology construction
# differs (`sky.merge(parent)` instead of `sky.split(leaf)`).
#
# Two independent guards prevent split/merge oscillation, per the proposal's
# "merge only after two rounds" rule: a candidate must score favorably for
# `required_streak` consecutive rounds before it is eligible, and a group
# whose parent was just split carries an explicit cooldown that overrides
# any accumulated streak. `MergeHysteresisState` carries both across rounds;
# `advance_merge_hysteresis` is the only place that mutates it.


def mergeable_parents(
    topology: QuadtreeTopology,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
) -> tuple[QuadtreeLeaf, ...]:
    """Complete four-sibling groups that can be replaced by their parent.

    A parent leaf is never itself active while all four children are (the
    topology's leaf set is prefix-free), so a candidate is identified purely
    by all four children being present. ``candidates`` restricts the search
    to specific parent identities and need not already be validated as
    mergeable; non-mergeable entries are silently dropped, matching how
    ``mergeable_parents`` is used to re-check hysteresis-tracked candidates
    whose children may since have been split further or merged away.
    """

    leaf_set = set(topology.leaves)
    if candidates is None:
        parents_to_check = {leaf.parent() for leaf in topology.leaves if leaf.level > 0}
    else:
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates must be unique")
        parents_to_check = set(candidates)
    return tuple(
        sorted(
            parent
            for parent in parents_to_check
            if all(child in leaf_set for child in parent.children())
        )
    )


@dataclass(frozen=True)
class LocalMergeEvaluation:
    """Conditional four-sibling-to-parent replacement with all other leaves fixed."""

    leaf: QuadtreeLeaf
    children: tuple[QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf]
    child_flux: tuple[float, float, float, float]
    parent_flux: float
    metrics: QuadtreeObjectiveMetrics
    objective_change: float
    predicted_improvement: float
    holdout_change: float | None


@dataclass(frozen=True)
class LocalMergeResult:
    """Baseline and conditional solutions for requested merge candidates."""

    baseline: QuadtreeObjectiveMetrics
    evaluations: tuple[LocalMergeEvaluation, ...]

    @property
    def ranked(self) -> tuple[LocalMergeEvaluation, ...]:
        """Candidates from greatest to least conditional improvement."""

        return tuple(
            sorted(
                self.evaluations,
                key=lambda evaluation: (-evaluation.predicted_improvement, evaluation.leaf),
            )
        )

    @property
    def best(self) -> LocalMergeEvaluation | None:
        """The best improving local merge, if one exists."""

        ranked = self.ranked
        if not ranked or ranked[0].predicted_improvement <= 0:
            return None
        return ranked[0]


def local_four_sibling_merge_lookahead(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
) -> LocalMergeResult:
    """Solve each exact conditional four-sibling-to-parent replacement.

    The reverse of ``local_four_child_lookahead``: for every complete
    sibling group, remove the four children's actual fitted response from
    the residual and solve the resulting one-variable non-negative
    quadratic for the merged parent's flux, holding every other leaf fixed.
    This is the "exact visibility response" merge score from the proposal.
    The solved flux is the locally optimal value, not simply the sum of the
    children; that sum is used instead to seed the actual warm-refit in
    ``merge_quadtree_batch``, matching how the split path seeds its
    warm-refit with equal quarters rather than its own lookahead optimum.
    """

    if train_mask.shape != block.shape:
        raise ValueError("train_mask must match the visibility block")
    if current_fit.prediction.shape != block.shape or current_fit.residual.shape != block.shape:
        raise ValueError("current_fit predictions must match the visibility block")
    if not np.isfinite(config.sparsity_weight) or config.sparsity_weight < 0:
        raise ValueError("sparsity_weight must be finite and non-negative")
    approximation = GaussianApproximation(approximation)
    selected = mergeable_parents(current_fit.topology, candidates=candidates)
    baseline = quadtree_objective_metrics(
        block,
        current_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    flux = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux, strict=True))
    total_flux = float(np.sum(flux))
    residual = np.asarray(current_fit.residual).reshape(-1)
    training_weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~train_mask),
        dtype=np.float64,
    ).reshape(-1)
    training_weight_sum = float(np.sum(training_weight))
    if training_weight_sum <= 0:
        raise ValueError("train_mask must contain positive-weight finite samples")
    holdout_weight = (
        None
        if holdout_mask is None
        else np.asarray(
            effective_weight(block.visibility, block.weight, ~holdout_mask),
            dtype=np.float64,
        ).reshape(-1)
    )

    evaluations = []
    for parent in selected:
        children = parent.children()
        child_flux = np.asarray([flux_by_leaf[child] for child in children], dtype=np.float64)
        _, child_responses = _child_response_matrix(
            block,
            current_fit.topology,
            parent,
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        )
        parent_topology = QuadtreeTopology(current_fit.topology.grid, (parent,))
        parent_response = _predict_quadtree_flux(
            block,
            parent_topology,
            np.ones(1, dtype=np.float64),
            config,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
        ).reshape(-1)
        residual_without_children = residual - child_responses @ child_flux
        training_residual_without_children = np.where(
            training_weight > 0,
            residual_without_children,
            0.0,
        )
        hessian_scalar = (2.0 / training_weight_sum) * float(
            np.real(np.vdot(parent_response, training_weight * parent_response))
        )
        linear_scalar = (2.0 / training_weight_sum) * float(
            np.real(
                np.vdot(
                    parent_response,
                    training_weight * training_residual_without_children,
                )
            )
        ) + config.sparsity_weight
        new_parent_flux = _solve_nonnegative_quadratic(
            np.asarray([[hessian_scalar]], dtype=np.float64),
            np.asarray([linear_scalar], dtype=np.float64),
            total=None,
        )
        candidate_residual = residual_without_children + parent_response * new_parent_flux[0]
        training_data = _weighted_residual_power(candidate_residual, training_weight)
        candidate_total_flux = total_flux - float(np.sum(child_flux)) + float(new_parent_flux[0])
        sparsity = float(config.sparsity_weight * candidate_total_flux)
        topology = float(current_fit.leaf_penalty * (len(current_fit.topology.leaves) - 3))
        holdout_data = (
            None
            if holdout_weight is None
            else _weighted_residual_power(candidate_residual, holdout_weight)
        )
        metrics = QuadtreeObjectiveMetrics(
            training_data=training_data,
            sparsity=sparsity,
            topology=topology,
            objective=training_data + sparsity + topology,
            holdout_data=holdout_data,
        )
        objective_change = metrics.objective - baseline.objective
        holdout_change = (
            None
            if baseline.holdout_data is None or holdout_data is None
            else holdout_data - baseline.holdout_data
        )
        evaluations.append(
            LocalMergeEvaluation(
                leaf=parent,
                children=children,
                child_flux=(
                    float(child_flux[0]),
                    float(child_flux[1]),
                    float(child_flux[2]),
                    float(child_flux[3]),
                ),
                parent_flux=float(new_parent_flux[0]),
                metrics=metrics,
                objective_change=objective_change,
                predicted_improvement=-objective_change,
                holdout_change=holdout_change,
            )
        )
    return LocalMergeResult(baseline=baseline, evaluations=tuple(evaluations))


@dataclass(frozen=True)
class MergeHysteresisState:
    """Per-parent-group streak and post-split cooldown bookkeeping.

    Both mappings are keyed by the merged-parent identity (the same
    ``QuadtreeLeaf`` returned by ``mergeable_parents``), so state persists
    correctly across rounds regardless of how the topology changes
    elsewhere. Construct with ``MergeHysteresisState.empty()`` at the start
    of a run; ``advance_merge_hysteresis`` is the only function that should
    produce a new state from an old one.
    """

    eligible_streak: dict[QuadtreeLeaf, int]
    split_cooldown: dict[QuadtreeLeaf, int]

    @staticmethod
    def empty() -> MergeHysteresisState:
        return MergeHysteresisState(eligible_streak={}, split_cooldown={})


def advance_merge_hysteresis(
    state: MergeHysteresisState,
    evaluations: tuple[LocalMergeEvaluation, ...],
    *,
    just_split: tuple[QuadtreeLeaf, ...] = (),
    cooldown_rounds: int = 1,
) -> MergeHysteresisState:
    """Update streak and cooldown bookkeeping for this round's gating.

    A candidate's streak increments only while it scores favorably
    (``predicted_improvement > 0``) and carries no active cooldown; any
    other outcome (ineligible, not evaluated this round, or still cooling
    down) resets it to zero by omission. ``just_split`` names parents whose
    children were just created by an accepted split this round: their
    cooldown is (re)armed to ``cooldown_rounds``, which also forces their
    streak to zero, so an immediate reverse merge needs a fresh streak once
    the cooldown lapses. The returned state reflects this round and is used
    both to gate this round's merge selection and as next round's input.
    """

    if cooldown_rounds < 0:
        raise ValueError("cooldown_rounds must be non-negative")
    if len({evaluation.leaf for evaluation in evaluations}) != len(evaluations):
        raise ValueError("evaluations must contain unique leaves")

    improvement_by_leaf = {
        evaluation.leaf: evaluation.predicted_improvement for evaluation in evaluations
    }
    tracked = (
        set(state.eligible_streak)
        | set(state.split_cooldown)
        | set(improvement_by_leaf)
        | set(just_split)
    )
    next_streak: dict[QuadtreeLeaf, int] = {}
    next_cooldown: dict[QuadtreeLeaf, int] = {}
    for candidate in tracked:
        remaining_cooldown = max(0, state.split_cooldown.get(candidate, 0) - 1)
        if candidate in just_split:
            remaining_cooldown = max(remaining_cooldown, cooldown_rounds)
        if remaining_cooldown > 0:
            next_cooldown[candidate] = remaining_cooldown
            continue
        improvement = improvement_by_leaf.get(candidate)
        if improvement is not None and improvement > 0:
            next_streak[candidate] = state.eligible_streak.get(candidate, 0) + 1
    return MergeHysteresisState(eligible_streak=next_streak, split_cooldown=next_cooldown)


@dataclass(frozen=True)
class BulkMergeSelection:
    """Dörfler-style merge marking subject to a topology-shrink budget."""

    selected: tuple[QuadtreeLeaf, ...]
    available_improvement: float
    selected_improvement: float
    covered_fraction: float
    merge_budget: int
    removed_leaf_count: int


def select_bulk_merges(
    evaluations: tuple[LocalMergeEvaluation, ...],
    hysteresis: MergeHysteresisState,
    current_leaf_count: int,
    *,
    required_streak: int = 2,
    target_improvement_fraction: float = 0.7,
    max_merge_fraction: float = 0.05,
    max_merges: int | None = None,
    min_improvement: float = 0.0,
) -> BulkMergeSelection:
    """Select a score-dominant prefix of hysteresis-eligible merge candidates.

    Each accepted merge replaces four leaves with one and therefore removes
    three active leaves; the fractional budget applies to merge operations
    the same way ``select_bulk_splits``'s applies to parent splits. A
    candidate is only eligible once ``hysteresis`` shows it has scored
    favorably for ``required_streak`` consecutive rounds with no active
    post-split cooldown (see ``advance_merge_hysteresis``).
    """

    if current_leaf_count < 1:
        raise ValueError("current_leaf_count must be positive")
    if not 0 < target_improvement_fraction <= 1:
        raise ValueError("target_improvement_fraction must be in (0, 1]")
    if not 0 < max_merge_fraction <= 1:
        raise ValueError("max_merge_fraction must be in (0, 1]")
    if max_merges is not None and max_merges < 1:
        raise ValueError("max_merges must be positive")
    if not np.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and non-negative")
    if required_streak < 1:
        raise ValueError("required_streak must be positive")
    if len({evaluation.leaf for evaluation in evaluations}) != len(evaluations):
        raise ValueError("evaluations must contain unique leaves")

    eligible = tuple(
        (evaluation, evaluation.predicted_improvement)
        for evaluation in evaluations
        if evaluation.predicted_improvement > min_improvement
        and hysteresis.eligible_streak.get(evaluation.leaf, 0) >= required_streak
    )
    ranked = tuple(sorted(eligible, key=lambda item: (-item[1], item[0].leaf)))
    available = float(sum(improvement for _, improvement in ranked))
    fractional_budget = max(1, int(np.ceil(max_merge_fraction * current_leaf_count)))
    merge_budget = fractional_budget if max_merges is None else min(fractional_budget, max_merges)
    selected = []
    selected_improvement = 0.0
    target = target_improvement_fraction * available
    for evaluation, improvement in ranked[:merge_budget]:
        selected.append(evaluation.leaf)
        selected_improvement += improvement
        if selected_improvement >= target:
            break
    covered_fraction = selected_improvement / available if available > 0 else 0.0
    return BulkMergeSelection(
        selected=tuple(selected),
        available_improvement=available,
        selected_improvement=selected_improvement,
        covered_fraction=covered_fraction,
        merge_budget=merge_budget,
        removed_leaf_count=3 * len(selected),
    )


def _merge_batch_initial_sky(
    current_fit: QuadtreeInferenceResult,
    selected: tuple[QuadtreeLeaf, ...],
) -> QuadtreeSky:
    sky = QuadtreeSky(
        current_fit.topology.grid,
        current_fit.topology.leaves,
        _aligned_flux(current_fit.topology, current_fit.flux),
    )
    for parent in selected:
        sky = sky.merge(parent)
    return sky


def merge_quadtree_batch(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    config: InferenceConfig,
    selected: tuple[QuadtreeLeaf, ...],
    *,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    minimum_training_relative_improvement: float = 0.0,
    minimum_holdout_relative_improvement: float = 0.0,
    max_refits: int = 4,
) -> RefinementBatchResult:
    """Warm-refit and validate a ranked merge batch, halving it on rejection.

    The mirror of ``refine_quadtree_batch``: ``selected`` are merged-parent
    identities (see ``mergeable_parents``), ordered strongest to weakest.
    Every attempt starts from the same accepted fit. Each merged parent is
    seeded with the sum of its children's flux (``QuadtreeSky.merge``'s
    definition), then all leaf fluxes are optimized together. A proposal
    must improve both the penalized training objective and held-out loss by
    the requested relative tolerances. Rejected batches are backtracked to
    their strongest half, up to ``max_refits`` global optimizations.
    """

    if not selected:
        raise ValueError("selected must contain at least one leaf")
    if len(set(selected)) != len(selected):
        raise ValueError("selected leaves must be unique")
    missing = [
        parent
        for parent in selected
        if any(child not in current_fit.topology.leaves for child in parent.children())
    ]
    if missing:
        raise ValueError(f"selected parents are not mergeable: {missing}")
    if train_mask.shape != block.shape or holdout_mask.shape != block.shape:
        raise ValueError("train_mask and holdout_mask must match the visibility block")
    if np.any(train_mask & holdout_mask):
        raise ValueError("train_mask and holdout_mask must be disjoint")
    if not np.any(holdout_mask):
        raise ValueError("holdout_mask must contain active samples")
    tolerances = (
        minimum_training_relative_improvement,
        minimum_holdout_relative_improvement,
    )
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("relative improvement thresholds must be finite and non-negative")
    if max_refits < 1:
        raise ValueError("max_refits must be positive")

    approximation = GaussianApproximation(approximation)
    baseline = quadtree_objective_metrics(
        block,
        current_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    if baseline.holdout_data is None:
        raise RuntimeError("holdout metrics were not computed")

    attempts: list[RefinementAttempt] = []
    batch_size = len(selected)
    while len(attempts) < max_refits:
        attempt_leaves = selected[:batch_size]
        initial_sky = _merge_batch_initial_sky(current_fit, attempt_leaves)
        fit = infer_quadtree(
            block,
            initial_sky.topology,
            train_mask,
            config,
            holdout_mask=holdout_mask,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            leaf_penalty=current_fit.leaf_penalty,
            initial_flux=initial_sky.flux,
        )
        metrics = quadtree_objective_metrics(
            block,
            fit,
            train_mask,
            config,
            holdout_mask=holdout_mask,
        )
        if metrics.holdout_data is None:
            raise RuntimeError("holdout metrics were not computed")
        training_improvement = _relative_improvement(
            baseline.objective,
            metrics.objective,
        )
        holdout_improvement = _relative_improvement(
            baseline.holdout_data,
            metrics.holdout_data,
        )
        accepted = bool(
            np.isfinite(training_improvement)
            and np.isfinite(holdout_improvement)
            and training_improvement > minimum_training_relative_improvement
            and holdout_improvement > minimum_holdout_relative_improvement
        )
        attempts.append(
            RefinementAttempt(
                selected=attempt_leaves,
                fit=fit,
                metrics=metrics,
                training_relative_improvement=training_improvement,
                holdout_relative_improvement=holdout_improvement,
                accepted=accepted,
            )
        )
        if accepted or batch_size == 1:
            break
        batch_size = max(1, batch_size // 2)

    return RefinementBatchResult(
        baseline=baseline,
        attempts=tuple(attempts),
    )


@dataclass(frozen=True)
class SingleMergeEvaluation:
    """Globally refitted objective after replacing one sibling group by its parent."""

    leaf: QuadtreeLeaf
    children: tuple[QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf, QuadtreeLeaf]
    child_flux: tuple[float, float, float, float]
    metrics: QuadtreeObjectiveMetrics
    fit: QuadtreeInferenceResult
    objective_change: float
    predicted_improvement: float
    holdout_change: float | None


@dataclass(frozen=True)
class ExhaustiveMergeResult:
    """Baseline fit and deterministic evaluations of every requested merge."""

    baseline: QuadtreeObjectiveMetrics
    evaluations: tuple[SingleMergeEvaluation, ...]

    @property
    def ranked(self) -> tuple[SingleMergeEvaluation, ...]:
        """Candidates from greatest to least training-objective improvement."""

        return tuple(
            sorted(
                self.evaluations,
                key=lambda evaluation: (-evaluation.predicted_improvement, evaluation.leaf),
            )
        )

    @property
    def best(self) -> SingleMergeEvaluation | None:
        """The best improving candidate, or ``None`` when every merge is worse."""

        ranked = self.ranked
        if not ranked or ranked[0].predicted_improvement <= 0:
            return None
        return ranked[0]


def exhaustive_single_merge_oracle(
    block: VisibilityBlock,
    current_fit: QuadtreeInferenceResult,
    train_mask: np.ndarray,
    config: InferenceConfig,
    *,
    holdout_mask: np.ndarray | None = None,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    fixed_gains: np.ndarray | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    solver: Literal["optax", "active_set"] = "optax",
    active_set_max_leaves: int = 12,
) -> ExhaustiveMergeResult:
    """Globally refit every requested legal merge on the same data.

    The mirror of ``exhaustive_single_split_oracle``: intentionally
    expensive, intended as a small-problem validation oracle for
    ``local_four_sibling_merge_lookahead``. Each merged parent is seeded
    with the sum of its children's flux and then globally refit. A supplied
    holdout mask is evaluated after fitting and does not affect the training
    solve.
    """

    if current_fit.topology.leaves == ():
        raise ValueError("current_fit topology must contain at least one leaf")
    if solver not in {"optax", "active_set"}:
        raise ValueError("solver must be optax or active_set")
    selected = mergeable_parents(current_fit.topology, candidates=candidates)
    if not selected:
        raise ValueError("no complete sibling groups are available to merge")

    baseline_fit = (
        current_fit
        if solver == "optax"
        else solve_quadtree_flux_active_set(
            block,
            current_fit.topology,
            train_mask,
            config,
            holdout_mask=holdout_mask,
            fixed_gains=fixed_gains,
            primary_beam=primary_beam,
            approximation=approximation,
            leaf_penalty=current_fit.leaf_penalty,
            max_leaves=active_set_max_leaves,
        )
    )
    baseline = quadtree_objective_metrics(
        block,
        baseline_fit,
        train_mask,
        config,
        holdout_mask=holdout_mask,
    )
    sky = QuadtreeSky(
        baseline_fit.topology.grid,
        baseline_fit.topology.leaves,
        baseline_fit.flux,
    )
    flux_by_leaf = dict(zip(sky.leaves, sky.flux, strict=True))
    evaluations = []
    for parent in selected:
        children = parent.children()
        merge_sky = sky.merge(parent)
        merge_fit = (
            infer_quadtree(
                block,
                merge_sky.topology,
                train_mask,
                config,
                fixed_gains=fixed_gains,
                primary_beam=primary_beam,
                approximation=approximation,
                leaf_penalty=current_fit.leaf_penalty,
                initial_flux=merge_sky.flux,
            )
            if solver == "optax"
            else solve_quadtree_flux_active_set(
                block,
                merge_sky.topology,
                train_mask,
                config,
                holdout_mask=holdout_mask,
                fixed_gains=fixed_gains,
                primary_beam=primary_beam,
                approximation=approximation,
                leaf_penalty=current_fit.leaf_penalty,
                max_leaves=active_set_max_leaves,
            )
        )
        metrics = quadtree_objective_metrics(
            block,
            merge_fit,
            train_mask,
            config,
            holdout_mask=holdout_mask,
        )
        objective_change = metrics.objective - baseline.objective
        holdout_change = (
            None
            if baseline.holdout_data is None or metrics.holdout_data is None
            else metrics.holdout_data - baseline.holdout_data
        )
        evaluations.append(
            SingleMergeEvaluation(
                leaf=parent,
                children=children,
                child_flux=(
                    float(flux_by_leaf[children[0]]),
                    float(flux_by_leaf[children[1]]),
                    float(flux_by_leaf[children[2]]),
                    float(flux_by_leaf[children[3]]),
                ),
                metrics=metrics,
                fit=merge_fit,
                objective_change=objective_change,
                predicted_improvement=-objective_change,
                holdout_change=holdout_change,
            )
        )
    return ExhaustiveMergeResult(baseline=baseline, evaluations=tuple(evaluations))


@dataclass(frozen=True)
class MergeRankingEntry:
    """One candidate's positions in local-merge-lookahead and oracle rankings."""

    leaf: QuadtreeLeaf
    lookahead_rank: int
    oracle_rank: int
    lookahead_improvement: float
    oracle_improvement: float


@dataclass(frozen=True)
class MergeLookaheadOracleComparison:
    """Direct rank comparison between local merge lookahead and the oracle."""

    entries: tuple[MergeRankingEntry, ...]
    spearman_rho: float
    top1_match: bool


def compare_merge_lookahead_to_oracle(
    lookahead: LocalMergeResult,
    oracle: ExhaustiveMergeResult,
) -> MergeLookaheadOracleComparison:
    """Compare deterministic local-merge-lookahead and global-refit oracle ranks."""

    if not lookahead.evaluations:
        raise ValueError("lookahead must contain at least one candidate")
    local_by_leaf = {evaluation.leaf: evaluation for evaluation in lookahead.evaluations}
    oracle_by_leaf = {evaluation.leaf: evaluation for evaluation in oracle.evaluations}
    if len(local_by_leaf) != len(lookahead.evaluations):
        raise ValueError("lookahead must contain unique leaves")
    if len(oracle_by_leaf) != len(oracle.evaluations):
        raise ValueError("oracle must contain unique leaves")
    if set(local_by_leaf) != set(oracle_by_leaf):
        raise ValueError("lookahead and oracle must contain the same leaves")

    local_order = lookahead.ranked
    oracle_order = oracle.ranked
    local_rank = {evaluation.leaf: rank for rank, evaluation in enumerate(local_order, start=1)}
    oracle_rank = {evaluation.leaf: rank for rank, evaluation in enumerate(oracle_order, start=1)}
    entries = tuple(
        MergeRankingEntry(
            leaf=leaf,
            lookahead_rank=local_rank[leaf],
            oracle_rank=oracle_rank[leaf],
            lookahead_improvement=local_by_leaf[leaf].predicted_improvement,
            oracle_improvement=oracle_by_leaf[leaf].predicted_improvement,
        )
        for leaf in sorted(local_by_leaf, key=oracle_rank.__getitem__)
    )
    count = len(entries)
    squared_rank_difference = sum(
        (entry.lookahead_rank - entry.oracle_rank) ** 2 for entry in entries
    )
    spearman_rho = (
        1.0 if count == 1 else 1.0 - 6.0 * squared_rank_difference / (count * (count**2 - 1))
    )
    return MergeLookaheadOracleComparison(
        entries=entries,
        spearman_rho=float(spearman_rho),
        top1_match=local_order[0].leaf == oracle_order[0].leaf,
    )
