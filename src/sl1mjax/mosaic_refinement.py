"""Adaptive quadtree splitting for a shared pointing-aware mosaic sky."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import predict_stokes_i_explicit
from sl1mjax.hierarchical_imaging import AdaptiveRefinementConfig
from sl1mjax.inference import (
    InferenceConfig,
    MosaicQuadtreeInferenceResult,
    _require_physical_flux_solver,
    infer_mosaic_quadtree,
)
from sl1mjax.objective import effective_weight
from sl1mjax.polarization import Correlation
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    QuadtreeTopology,
    predict_quadtree_stokes_i_explicit,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import (
    _HAAR_CHILD_DETAILS,
    BulkSplitSelection,
    QuadtreeObjectiveMetrics,
    ResidualHaarScore,
    _aligned_flux,
    _build_residual_haar_score,
    _candidate_leaves,
    _relative_improvement,
    _validate_haar_parameters,
    select_bulk_splits,
)
from sl1mjax.resolution import (
    SynthesizedBeamEstimate,
    estimate_synthesized_beam,
    resolution_limited_max_depth,
)
from sl1mjax.sky import GaussianApproximation, SquarePixelBasis


@dataclass(frozen=True)
class MosaicRefinementAttempt:
    """One globally refitted prefix of proposed mosaic splits."""

    selected: tuple[QuadtreeLeaf, ...]
    fit: MosaicQuadtreeInferenceResult
    metrics: QuadtreeObjectiveMetrics
    training_relative_improvement: float
    holdout_relative_improvement: float
    accepted: bool


@dataclass(frozen=True)
class MosaicRefinementFailure:
    """One split-prefix refit that failed before producing finite metrics."""

    selected: tuple[QuadtreeLeaf, ...]
    error: str


@dataclass(frozen=True)
class MosaicRefinementBatchResult:
    """Validation result for a mosaic split batch and backtracked prefixes."""

    baseline: QuadtreeObjectiveMetrics
    attempts: tuple[MosaicRefinementAttempt, ...]
    failures: tuple[MosaicRefinementFailure, ...] = ()

    @property
    def accepted_attempt(self) -> MosaicRefinementAttempt | None:
        """Return the first accepted attempt, if one exists."""

        return next((attempt for attempt in self.attempts if attempt.accepted), None)


@dataclass(frozen=True)
class MosaicAdaptiveRefinementRound:
    """Screening, exact ranking, and validation record for one round."""

    index: int
    leaf_count_before: int
    screening_scores: tuple[ResidualHaarScore, ...]
    exact_screening_scores: tuple[ResidualHaarScore, ...]
    selection: BulkSplitSelection
    validation: MosaicRefinementBatchResult | None


@dataclass(frozen=True)
class MosaicHierarchicalImagingResult:
    """Accepted joint mosaic fit and its adaptive split history."""

    inference: MosaicQuadtreeInferenceResult
    rounds: tuple[MosaicAdaptiveRefinementRound, ...]
    train_masks: tuple[np.ndarray, ...]
    holdout_masks: tuple[np.ndarray, ...]
    stop_reason: str
    elapsed_s: float
    synthesized_beam: SynthesizedBeamEstimate | None = None
    resolution_max_depth: int | None = None
    effective_max_depth: int = 0


def _validate_mosaic_inputs(
    blocks: tuple[VisibilityBlock, ...],
    fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    holdout_masks: tuple[np.ndarray, ...] | None = None,
) -> None:
    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if len(train_masks) != len(blocks):
        raise ValueError("train_masks must contain one mask per block")
    if len(fit.predictions) != len(blocks) or len(fit.residuals) != len(blocks):
        raise ValueError("fit must contain one prediction and residual per block")
    if holdout_masks is not None and len(holdout_masks) != len(blocks):
        raise ValueError("holdout_masks must contain one mask per block")
    for index, (block, train_mask, prediction, residual) in enumerate(
        zip(blocks, train_masks, fit.predictions, fit.residuals, strict=True)
    ):
        if train_mask.shape != block.shape:
            raise ValueError(f"train_masks[{index}] must match its visibility block")
        if prediction.shape != block.shape or residual.shape != block.shape:
            raise ValueError(f"fit arrays for block {index} must match the block")
        if holdout_masks is not None:
            holdout_mask = holdout_masks[index]
            if holdout_mask.shape != block.shape:
                raise ValueError(f"holdout_masks[{index}] must match its visibility block")
            if np.any(train_mask & holdout_mask):
                raise ValueError("train and holdout masks must be disjoint")


def _local_centres(
    topology: QuadtreeTopology,
    mosaic_phase_centre_rad: tuple[float, float],
    block: VisibilityBlock,
) -> tuple[np.ndarray, np.ndarray]:
    l, m = topology.centers()
    ra, dec = lmn_to_radec(*mosaic_phase_centre_rad, l, m)
    local_l, local_m, _ = radec_to_lmn(*block.phase_centre_rad, ra, dec)
    return local_l, local_m


def mosaic_quadtree_objective_metrics(
    blocks: tuple[VisibilityBlock, ...],
    fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    config: InferenceConfig,
    *,
    holdout_masks: tuple[np.ndarray, ...] | None = None,
) -> QuadtreeObjectiveMetrics:
    """Evaluate the shared objective with one denominator across pointings."""

    _validate_mosaic_inputs(blocks, fit, train_masks, holdout_masks)

    def data_term(masks: tuple[np.ndarray, ...], residuals: tuple[np.ndarray, ...]) -> float:
        numerator = 0.0
        denominator = 0.0
        for block, mask, residual in zip(blocks, masks, residuals, strict=True):
            weight = np.asarray(
                effective_weight(block.visibility, block.weight, ~mask),
                dtype=np.float64,
            )
            numerator += float(np.sum(weight * np.abs(np.where(weight > 0, residual, 0.0)) ** 2))
            denominator += float(np.sum(weight))
        if denominator <= 0:
            raise ValueError("masks must contain positive-weight finite samples")
        return numerator / denominator

    training_data = data_term(train_masks, fit.residuals)
    sparsity = float(config.sparsity_weight * np.sum(fit.flux))
    topology = float(fit.leaf_penalty * len(fit.topology.leaves))
    holdout_data = None
    if holdout_masks is not None:
        holdout_data = data_term(holdout_masks, fit.residuals)
    return QuadtreeObjectiveMetrics(
        training_data=training_data,
        sparsity=sparsity,
        topology=topology,
        objective=training_data + sparsity + topology,
        holdout_data=holdout_data,
    )


def batched_mosaic_residual_haar_scores(
    blocks: tuple[VisibilityBlock, ...],
    current_fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    config: InferenceConfig,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    min_parent_flux: float = 0.0,
    min_curvature: float = 0.0,
    min_eigenvalue_ratio: float = 1e-8,
    ridge_relative: float = 1e-8,
    constrain_child_flux: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[ResidualHaarScore, ...]:
    """Screen every parent using the summed mosaic residual projection.

    The residual gradient is exact. One representative 3x3 Gram matrix is
    accumulated across all pointings at each parent level. The beam and local
    direction cosines make that curvature approximate away from the
    representative position, so marked parents must be rescored exactly.
    """

    if config.operator_mode != "explicit":
        raise ValueError("mosaic Haar scoring requires operator_mode='explicit'")
    _validate_mosaic_inputs(blocks, current_fit, train_masks)
    _validate_haar_parameters(
        min_parent_flux,
        min_curvature,
        min_eigenvalue_ratio,
        ridge_relative,
    )
    approximation = GaussianApproximation(approximation)
    selected = _candidate_leaves(current_fit.topology, candidates, max_depth)
    if not selected:
        return ()
    flux = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux, strict=True))
    active_weights = tuple(
        np.asarray(
            effective_weight(block.visibility, block.weight, ~mask),
            dtype=np.float64,
        )
        for block, mask in zip(blocks, train_masks, strict=True)
    )
    weight_sum = float(sum(np.sum(weight) for weight in active_weights))
    if weight_sum <= 0:
        raise ValueError("train_masks must contain positive-weight finite samples")
    residuals = tuple(
        np.where(weight > 0, residual, 0.0)
        for weight, residual in zip(active_weights, current_fit.residuals, strict=True)
    )
    if any(not np.all(np.isfinite(residual)) for residual in residuals):
        raise ValueError("active residual samples must be finite")

    selected_by_level: dict[int, list[QuadtreeLeaf]] = {}
    for leaf in selected:
        selected_by_level.setdefault(leaf.level, []).append(leaf)
    score_by_leaf: dict[QuadtreeLeaf, ResidualHaarScore] = {}
    real_dtype = config.direct_dft.real_dtype
    complex_dtype = config.direct_dft.complex_dtype

    for parent_level, level_leaf_list in sorted(selected_by_level.items()):
        level_leaves = tuple(level_leaf_list)
        all_children = tuple(child for leaf in level_leaves for child in leaf.children())
        virtual_topology = QuadtreeTopology(current_fit.topology.grid, all_children)
        child_index = {child: index for index, child in enumerate(virtual_topology.leaves)}
        child_gradient = np.zeros(len(virtual_topology.leaves), dtype=np.float64)

        for block_index, (block, weight, residual) in enumerate(
            zip(blocks, active_weights, residuals, strict=True), start=1
        ):
            local_l, local_m = _local_centres(
                virtual_topology,
                current_fit.mosaic_phase_centre_rad,
                block,
            )
            beam_i, beam_rr, beam_ll = predict_beam_weights(
                primary_beam, local_l, local_m, block.frequency_hz
            )
            weight_jax = jnp.asarray(weight, dtype=real_dtype)
            residual_jax = jnp.asarray(residual, dtype=complex_dtype)
            local_centres = (local_l, local_m)

            beam_i_jax = None if beam_i is None else jnp.asarray(beam_i, dtype=real_dtype)
            beam_rr_jax = None if beam_rr is None else jnp.asarray(beam_rr, dtype=real_dtype)
            beam_ll_jax = None if beam_ll is None else jnp.asarray(beam_ll, dtype=real_dtype)

            def residual_projection(
                child_flux: jax.Array,
                *,
                selected_topology: QuadtreeTopology = virtual_topology,
                selected_block: VisibilityBlock = block,
                selected_beam_i: jax.Array | None = beam_i_jax,
                selected_beam_rr: jax.Array | None = beam_rr_jax,
                selected_beam_ll: jax.Array | None = beam_ll_jax,
                selected_centres: tuple[np.ndarray, np.ndarray] = local_centres,
                selected_weight: jax.Array = weight_jax,
                selected_residual: jax.Array = residual_jax,
            ) -> jax.Array:
                prediction = predict_quadtree_stokes_i_explicit(
                    child_flux,
                    selected_topology,
                    selected_block.uvw_m,
                    selected_block.frequency_hz,
                    selected_block.antenna1,
                    selected_block.antenna2,
                    selected_block.correlations,
                    approximation=approximation,
                    config=config.direct_dft,
                    beam_weights=selected_beam_i,
                    beam_weights_rr=selected_beam_rr,
                    beam_weights_ll=selected_beam_ll,
                    centers_lm=selected_centres,
                )
                return jnp.sum(selected_weight * jnp.real(jnp.conj(prediction) * selected_residual))

            child_gradient += np.asarray(
                jax.jit(jax.grad(residual_projection))(
                    jnp.zeros(len(virtual_topology.leaves), dtype=real_dtype)
                ),
                dtype=np.float64,
            )
            if progress is not None:
                progress(
                    f"approximate Haar level {parent_level}: pointing {block_index}/{len(blocks)}"
                )

        # The global mosaic phase centre is a stable representative. Each
        # pointing contributes with its own beam and local phase centre.
        representative = min(
            level_leaves,
            key=lambda leaf: sum(
                coordinate**2 for coordinate in current_fit.topology.grid.leaf_center_rad(leaf)
            ),
        )
        gram = np.zeros((3, 3), dtype=np.float64)
        child_topology = QuadtreeTopology(current_fit.topology.grid, representative.haar_children())
        details_by_child = dict(
            zip(representative.haar_children(), _HAAR_CHILD_DETAILS, strict=True)
        )
        details = np.asarray(
            [details_by_child[child] for child in child_topology.leaves],
            dtype=np.float64,
        )
        for block, weight in zip(blocks, active_weights, strict=True):
            local_l, local_m = _local_centres(
                child_topology,
                current_fit.mosaic_phase_centre_rad,
                block,
            )
            beam_i, beam_rr, beam_ll = predict_beam_weights(
                primary_beam, local_l, local_m, block.frequency_hz
            )
            responses = []
            for detail in details.T:
                prediction = predict_quadtree_stokes_i_explicit(
                    detail,
                    child_topology,
                    block.uvw_m,
                    block.frequency_hz,
                    block.antenna1,
                    block.antenna2,
                    block.correlations,
                    approximation=approximation,
                    config=config.direct_dft,
                    beam_weights=beam_i,
                    beam_weights_rr=beam_rr,
                    beam_weights_ll=beam_ll,
                    centers_lm=(local_l, local_m),
                )
                responses.append(np.asarray(prediction).reshape(-1))
            response_matrix = np.stack(responses, axis=1)
            flat_weight = weight.reshape(-1)
            gram += np.real(response_matrix.conj().T @ (flat_weight[:, None] * response_matrix))

        child_gradient *= 2.0 / weight_sum
        gram *= 2.0 / weight_sum
        for leaf in level_leaves:
            indices = np.asarray([child_index[child] for child in leaf.haar_children()])
            gradient = child_gradient[indices] @ _HAAR_CHILD_DETAILS
            score_by_leaf[leaf] = _build_residual_haar_score(
                leaf,
                float(flux_by_leaf[leaf]),
                gradient,
                gram,
                min_parent_flux=min_parent_flux,
                min_curvature=min_curvature,
                min_eigenvalue_ratio=min_eigenvalue_ratio,
                ridge_relative=ridge_relative,
                curvature_mode=f"mosaic_per_level_approximate:{parent_level}",
                constrain_child_flux=constrain_child_flux,
            )
    return tuple(score_by_leaf[leaf] for leaf in selected)


def batched_exact_mosaic_residual_haar_scores(
    blocks: tuple[VisibilityBlock, ...],
    current_fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    config: InferenceConfig,
    *,
    candidates: tuple[QuadtreeLeaf, ...] | None = None,
    max_depth: int | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    min_parent_flux: float = 0.0,
    min_curvature: float = 0.0,
    min_eigenvalue_ratio: float = 1e-8,
    ridge_relative: float = 1e-8,
    candidate_batch_size: int = 32,
    row_batch_size: int = 1024,
    constrain_child_flux: bool = True,
    progress: Callable[[str], None] | None = None,
) -> tuple[ResidualHaarScore, ...]:
    """Compute exact per-parent mosaic Haar scores in bounded tiles."""

    if config.operator_mode != "explicit":
        raise ValueError("mosaic Haar scoring requires operator_mode='explicit'")
    if candidate_batch_size < 1 or row_batch_size < 1:
        raise ValueError("candidate and row batch sizes must be positive")
    _validate_mosaic_inputs(blocks, current_fit, train_masks)
    _validate_haar_parameters(
        min_parent_flux,
        min_curvature,
        min_eigenvalue_ratio,
        ridge_relative,
    )
    approximation = GaussianApproximation(approximation)
    selected = _candidate_leaves(current_fit.topology, candidates, max_depth)
    if not selected:
        return ()
    flux = _aligned_flux(current_fit.topology, current_fit.flux)
    flux_by_leaf = dict(zip(current_fit.topology.leaves, flux, strict=True))
    active_weights = tuple(
        np.asarray(
            effective_weight(block.visibility, block.weight, ~mask),
            dtype=np.float64,
        )
        for block, mask in zip(blocks, train_masks, strict=True)
    )
    residuals = tuple(
        np.where(weight > 0, residual, 0.0)
        for weight, residual in zip(active_weights, current_fit.residuals, strict=True)
    )
    weight_sum = float(sum(np.sum(weight) for weight in active_weights))
    if weight_sum <= 0:
        raise ValueError("train_masks must contain positive-weight finite samples")

    by_level: dict[int, list[QuadtreeLeaf]] = {}
    for leaf in selected:
        by_level.setdefault(leaf.level, []).append(leaf)
    batches = tuple(
        (level, tuple(leaves[start : start + candidate_batch_size]))
        for level, leaves in sorted(by_level.items())
        for start in range(0, len(leaves), candidate_batch_size)
    )
    score_by_leaf: dict[QuadtreeLeaf, ResidualHaarScore] = {}
    real_dtype = config.direct_dft.real_dtype

    for batch_index, (parent_level, candidate_batch) in enumerate(batches, start=1):
        global_child_l = np.asarray(
            [
                [
                    current_fit.topology.grid.leaf_center_rad(child)[0]
                    for child in leaf.haar_children()
                ]
                for leaf in candidate_batch
            ]
        )
        global_child_m = np.asarray(
            [
                [
                    current_fit.topology.grid.leaf_center_rad(child)[1]
                    for child in leaf.haar_children()
                ]
                for leaf in candidate_batch
            ]
        )
        child_ra, child_dec = lmn_to_radec(
            *current_fit.mosaic_phase_centre_rad,
            global_child_l,
            global_child_m,
        )
        gradient = np.zeros((len(candidate_batch), 3), dtype=np.float64)
        gram = np.zeros((len(candidate_batch), 3, 3), dtype=np.float64)

        for block, active_weight, residual in zip(blocks, active_weights, residuals, strict=True):
            child_l, child_m, _ = radec_to_lmn(*block.phase_centre_rad, child_ra, child_dec)
            beam_i, beam_rr, beam_ll = predict_beam_weights(
                primary_beam,
                child_l.ravel(),
                child_m.ravel(),
                block.frequency_hz,
            )
            beam_shape = (len(candidate_batch), 4, block.shape[1])
            beam_mode = 0 if primary_beam is None else (2 if primary_beam.apply_squint else 1)
            dummy_beam = np.ones(beam_shape, dtype=np.float64)
            beam_i_batch = dummy_beam if beam_i is None else beam_i.reshape(beam_shape)
            beam_rr_batch = dummy_beam if beam_rr is None else beam_rr.reshape(beam_shape)
            beam_ll_batch = dummy_beam if beam_ll is None else beam_ll.reshape(beam_shape)
            child_l_jax = jnp.asarray(child_l, dtype=real_dtype)
            child_m_jax = jnp.asarray(child_m, dtype=real_dtype)
            beam_i_jax = jnp.asarray(beam_i_batch, dtype=real_dtype)
            beam_rr_jax = jnp.asarray(beam_rr_batch, dtype=real_dtype)
            beam_ll_jax = jnp.asarray(beam_ll_batch, dtype=real_dtype)

            selected_frequency = block.frequency_hz
            selected_correlations = block.correlations
            child_width = current_fit.topology.grid.leaf_width_rad(parent_level + 1)
            selected_beam_mode = beam_mode
            selected_child_l = child_l_jax
            selected_child_m = child_m_jax
            selected_beam_i = beam_i_jax
            selected_beam_rr = beam_rr_jax
            selected_beam_ll = beam_ll_jax

            def predict_models(
                uvw_m: jax.Array,
                antenna1: jax.Array,
                antenna2: jax.Array,
                frequency_hz: np.ndarray = selected_frequency,
                correlations: tuple[Correlation, ...] = selected_correlations,
                pixel_width_rad: float = child_width,
                beam_selection: int = selected_beam_mode,
                batch_l: jax.Array = selected_child_l,
                batch_m: jax.Array = selected_child_m,
                batch_beam_i: jax.Array = selected_beam_i,
                batch_beam_rr: jax.Array = selected_beam_rr,
                batch_beam_ll: jax.Array = selected_beam_ll,
            ) -> jax.Array:
                def predict_candidate(
                    candidate_l: jax.Array,
                    candidate_m: jax.Array,
                    candidate_beam_i: jax.Array,
                    candidate_beam_rr: jax.Array,
                    candidate_beam_ll: jax.Array,
                ) -> jax.Array:
                    def predict_detail(model_flux: jax.Array) -> jax.Array:
                        return predict_stokes_i_explicit(
                            model_flux,
                            candidate_l,
                            candidate_m,
                            uvw_m,
                            frequency_hz,
                            antenna1,
                            antenna2,
                            correlations,
                            pixel_basis=SquarePixelBasis(1.0, approximation),
                            pixel_size_rad=pixel_width_rad,
                            config=config.direct_dft,
                            beam_weights=(candidate_beam_i if beam_selection == 1 else None),
                            beam_weights_rr=(candidate_beam_rr if beam_selection == 2 else None),
                            beam_weights_ll=(candidate_beam_ll if beam_selection == 2 else None),
                        )

                    return jax.vmap(predict_detail)(
                        jnp.asarray(_HAAR_CHILD_DETAILS.T, dtype=real_dtype)
                    )

                return jax.vmap(predict_candidate)(
                    batch_l,
                    batch_m,
                    batch_beam_i,
                    batch_beam_rr,
                    batch_beam_ll,
                )

            predict_models_jit = jax.jit(predict_models)
            for row_start in range(0, block.shape[0], row_batch_size):
                row_stop = min(row_start + row_batch_size, block.shape[0])
                response_models = np.asarray(
                    predict_models_jit(
                        block.uvw_m[row_start:row_stop],
                        block.antenna1[row_start:row_stop],
                        block.antenna2[row_start:row_stop],
                    )
                )
                responses = response_models.transpose(2, 3, 4, 0, 1).reshape(
                    -1, len(candidate_batch), 3
                )
                weights = active_weight[row_start:row_stop].reshape(-1)
                residual_batch = residual[row_start:row_stop].reshape(-1)
                gradient += np.real(
                    np.einsum(
                        "scd,s,s->cd",
                        np.conj(responses),
                        weights,
                        residual_batch,
                        optimize=True,
                    )
                )
                gram += np.real(
                    np.einsum(
                        "scd,s,sce->cde",
                        np.conj(responses),
                        weights,
                        responses,
                        optimize=True,
                    )
                )

        gradient *= 2.0 / weight_sum
        gram *= 2.0 / weight_sum
        for index, leaf in enumerate(candidate_batch):
            score_by_leaf[leaf] = _build_residual_haar_score(
                leaf,
                float(flux_by_leaf[leaf]),
                gradient[index],
                gram[index],
                min_parent_flux=min_parent_flux,
                min_curvature=min_curvature,
                min_eigenvalue_ratio=min_eigenvalue_ratio,
                ridge_relative=ridge_relative,
                curvature_mode="mosaic_batched_exact",
                constrain_child_flux=constrain_child_flux,
            )
        if progress is not None:
            completed = sum(len(batch) for _, batch in batches[:batch_index])
            progress(
                f"exact mosaic Haar batch {batch_index}/{len(batches)}: "
                f"{completed}/{len(selected)} parents"
            )
    return tuple(score_by_leaf[leaf] for leaf in selected)


def _split_initial_sky(
    current_fit: MosaicQuadtreeInferenceResult,
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


def refine_mosaic_quadtree_batch(
    blocks: tuple[VisibilityBlock, ...],
    current_fit: MosaicQuadtreeInferenceResult,
    train_masks: tuple[np.ndarray, ...],
    holdout_masks: tuple[np.ndarray, ...],
    config: InferenceConfig,
    selected: tuple[QuadtreeLeaf, ...],
    *,
    primary_beam: VLAPrimaryBeam | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    minimum_training_relative_improvement: float = 0.0,
    minimum_holdout_relative_improvement: float = 0.0,
    max_refits: int = 4,
    progress: Callable[[str], None] | None = None,
) -> MosaicRefinementBatchResult:
    """Warm-refit and validate a ranked mosaic split batch."""

    _validate_mosaic_inputs(blocks, current_fit, train_masks, holdout_masks)
    if not selected:
        raise ValueError("selected must contain at least one leaf")
    if len(set(selected)) != len(selected):
        raise ValueError("selected leaves must be unique")
    missing = [leaf for leaf in selected if leaf not in current_fit.topology.leaves]
    if missing:
        raise ValueError(f"selected leaves are not active: {missing}")
    if not any(np.any(mask) for mask in holdout_masks):
        raise ValueError("holdout_masks must contain active samples")
    tolerances = (
        minimum_training_relative_improvement,
        minimum_holdout_relative_improvement,
    )
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("relative improvement thresholds must be finite and non-negative")
    if max_refits < 1:
        raise ValueError("max_refits must be positive")

    baseline = mosaic_quadtree_objective_metrics(
        blocks,
        current_fit,
        train_masks,
        config,
        holdout_masks=holdout_masks,
    )
    assert baseline.holdout_data is not None
    attempts = []
    failures = []
    batch_size = len(selected)
    refit_count = 0
    while refit_count < max_refits:
        attempt_leaves = selected[:batch_size]
        initial_sky = _split_initial_sky(current_fit, attempt_leaves)
        if progress is not None:
            progress(
                f"mosaic split refit {refit_count + 1}/{max_refits}: "
                f"{len(attempt_leaves)} parents, {len(initial_sky.leaves)} leaves"
            )
        refit_count += 1
        try:
            fit = infer_mosaic_quadtree(
                blocks,
                initial_sky.topology,
                train_masks,
                current_fit.mosaic_phase_centre_rad,
                config,
                holdout_masks=holdout_masks,
                primary_beam=primary_beam,
                approximation=approximation,
                leaf_penalty=current_fit.leaf_penalty,
                initial_flux=initial_sky.flux,
            )
        except RuntimeError as error:
            failure = MosaicRefinementFailure(
                selected=attempt_leaves,
                error=str(error),
            )
            failures.append(failure)
            if progress is not None:
                progress(
                    f"mosaic split refit {refit_count}: numerical failure for "
                    f"{len(attempt_leaves)} parents: {error}"
                )
            if batch_size == 1:
                break
            batch_size = max(1, batch_size // 2)
            continue
        metrics = mosaic_quadtree_objective_metrics(
            blocks,
            fit,
            train_masks,
            config,
            holdout_masks=holdout_masks,
        )
        assert metrics.holdout_data is not None
        training_improvement = _relative_improvement(baseline.objective, metrics.objective)
        holdout_improvement = _relative_improvement(baseline.holdout_data, metrics.holdout_data)
        accepted = bool(
            np.isfinite(training_improvement)
            and np.isfinite(holdout_improvement)
            and training_improvement > minimum_training_relative_improvement
            and holdout_improvement > minimum_holdout_relative_improvement
        )
        attempts.append(
            MosaicRefinementAttempt(
                selected=attempt_leaves,
                fit=fit,
                metrics=metrics,
                training_relative_improvement=training_improvement,
                holdout_relative_improvement=holdout_improvement,
                accepted=accepted,
            )
        )
        if progress is not None:
            progress(
                f"mosaic split refit {refit_count}: steps={fit.steps}, "
                f"KKT={fit.kkt_residual:.3g}, train={training_improvement:.6g}, "
                f"holdout={holdout_improvement:.6g}, accepted={accepted}"
            )
        if accepted or batch_size == 1:
            break
        batch_size = max(1, batch_size // 2)
    return MosaicRefinementBatchResult(
        baseline=baseline,
        attempts=tuple(attempts),
        failures=tuple(failures),
    )


def reconstruct_mosaic_hierarchical(
    blocks: tuple[VisibilityBlock, ...],
    train_masks: tuple[np.ndarray, ...],
    holdout_masks: tuple[np.ndarray, ...],
    mosaic_phase_centre_rad: tuple[float, float],
    config: AdaptiveRefinementConfig,
    *,
    primary_beam: VLAPrimaryBeam | None = None,
    progress: Callable[[str], None] | None = None,
) -> MosaicHierarchicalImagingResult:
    """Fit and adapt one sky using the combined evidence from every pointing."""

    if config.inference.operator_mode != "explicit":
        raise ValueError("adaptive mosaic imaging requires operator_mode='explicit'")
    _require_physical_flux_solver(
        config.inference.solver, context="adaptive mosaic imaging"
    )
    synthesized_beam = None
    resolution_max_depth = None
    effective_max_depth = config.max_depth
    if config.maximum_pixels_per_beam is not None:
        synthesized_beam = estimate_synthesized_beam(blocks, train_masks)
        resolution_max_depth = resolution_limited_max_depth(
            config.root_pixel_size_rad,
            synthesized_beam.minor_fwhm_rad,
            maximum_pixels_per_beam=config.maximum_pixels_per_beam,
        )
        effective_max_depth = min(effective_max_depth, resolution_max_depth)
        if progress is not None:
            progress(
                "mosaic resolution depth cap: "
                f"beam={np.rad2deg(synthesized_beam.major_fwhm_rad) * 3600:.3g}x"
                f"{np.rad2deg(synthesized_beam.minor_fwhm_rad) * 3600:.3g} arcsec, "
                f"requested={config.max_depth}, effective={effective_max_depth}"
            )
    initial_sky = quadtree_sky_from_regular_grid(
        config.root_size,
        config.root_pixel_size_rad,
        np.zeros(config.root_size**2),
    )
    started = perf_counter()
    if progress is not None:
        progress(f"initial mosaic fit: {len(blocks)} pointings, {len(initial_sky.leaves)} leaves")
    current_fit = infer_mosaic_quadtree(
        blocks,
        initial_sky.topology,
        train_masks,
        mosaic_phase_centre_rad,
        config.inference,
        holdout_masks=holdout_masks,
        primary_beam=primary_beam,
        approximation=config.approximation,
        leaf_penalty=config.leaf_penalty,
        initial_flux=initial_sky.flux,
    )
    rounds = []
    stop_reason = "maximum_rounds"
    for round_index in range(config.max_rounds):
        leaf_count = len(current_fit.topology.leaves)
        if progress is not None:
            progress(
                f"mosaic round {round_index + 1}/{config.max_rounds}: "
                f"joint screen over {leaf_count} leaves"
            )
        scores = batched_mosaic_residual_haar_scores(
            blocks,
            current_fit,
            train_masks,
            config.inference,
            max_depth=effective_max_depth,
            primary_beam=primary_beam,
            approximation=config.approximation,
            min_parent_flux=config.min_parent_flux,
            min_curvature=config.min_curvature,
            min_eigenvalue_ratio=config.min_eigenvalue_ratio,
            ridge_relative=config.ridge_relative,
            progress=progress,
        )
        selection = select_bulk_splits(
            scores,
            leaf_count,
            target_improvement_fraction=config.target_improvement_fraction,
            max_split_fraction=config.max_split_fraction,
            max_splits=config.max_splits_per_round,
            split_cost=3.0 * config.leaf_penalty,
        )
        exact_scores: tuple[ResidualHaarScore, ...] = ()
        if selection.selected:
            if progress is not None:
                progress(
                    f"mosaic round {round_index + 1}: exact joint rescore of "
                    f"{len(selection.selected)} parents"
                )
            exact_scores = batched_exact_mosaic_residual_haar_scores(
                blocks,
                current_fit,
                train_masks,
                config.inference,
                candidates=selection.selected,
                primary_beam=primary_beam,
                approximation=config.approximation,
                min_parent_flux=config.min_parent_flux,
                min_curvature=config.min_curvature,
                min_eigenvalue_ratio=config.min_eigenvalue_ratio,
                ridge_relative=config.ridge_relative,
                candidate_batch_size=config.score_candidate_batch_size,
                row_batch_size=config.score_row_batch_size,
                progress=progress,
            )
            selection = select_bulk_splits(
                exact_scores,
                leaf_count,
                target_improvement_fraction=config.target_improvement_fraction,
                max_split_fraction=config.max_split_fraction,
                max_splits=config.max_splits_per_round,
                split_cost=3.0 * config.leaf_penalty,
            )
        validation = None
        accepted = None
        if selection.selected:
            validation = refine_mosaic_quadtree_batch(
                blocks,
                current_fit,
                train_masks,
                holdout_masks,
                config.inference,
                selection.selected,
                primary_beam=primary_beam,
                approximation=config.approximation,
                minimum_training_relative_improvement=config.minimum_training_relative_improvement,
                minimum_holdout_relative_improvement=config.minimum_holdout_relative_improvement,
                max_refits=config.max_refits_per_round,
                progress=progress,
            )
            accepted = validation.accepted_attempt
            if accepted is not None:
                current_fit = accepted.fit
        rounds.append(
            MosaicAdaptiveRefinementRound(
                index=round_index,
                leaf_count_before=leaf_count,
                screening_scores=scores,
                exact_screening_scores=exact_scores,
                selection=selection,
                validation=validation,
            )
        )
        if accepted is None:
            stop_reason = (
                "no_eligible_splits" if not selection.selected else "split_validation_rejected"
            )
            break
        if progress is not None:
            progress(
                f"mosaic round {round_index + 1} accepted: "
                f"{len(current_fit.topology.leaves)} leaves"
            )
    return MosaicHierarchicalImagingResult(
        inference=current_fit,
        rounds=tuple(rounds),
        train_masks=train_masks,
        holdout_masks=holdout_masks,
        stop_reason=stop_reason,
        elapsed_s=perf_counter() - started,
        synthesized_beam=synthesized_beam,
        resolution_max_depth=resolution_max_depth,
        effective_max_depth=effective_max_depth,
    )
