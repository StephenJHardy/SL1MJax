"""Phase 6 Stokes-I reconstruction under fixed voltage-beam candidates.

Integration nodes stay numerical. Sky splits add fitted parents only after
inner-validation acceptance. The overlapping 60-arcsec coarse field is not
part of this path. Full Jones is diagnostic and is refused as a production
candidate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam_aware_imaging import (
    ComponentFamily,
    SkyComponent,
    SkyComponentTable,
    sky_table_from_mosaic_components,
    sky_table_from_records,
    sky_table_to_records,
)
from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.composite import MosaicQuadtreeComponent
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    IntegrationPlan,
    adjoint_voltage_from_plan,
    integration_plan_from_table,
    predict_voltage_from_plan,
    predict_voltage_from_plan_value_and_grad,
)
from sl1mjax.inference import InferenceConfig, positive_l1_kkt_residual
from sl1mjax.integration_planner import (
    IntegrationAuditReport,
    IntegrationPlannerReport,
    IntegrationTolerance,
    audit_integration_plan,
    integration_plan_from_planner,
    plan_integration,
)
from sl1mjax.objective import effective_weight
from sl1mjax.quadtree import QuadtreeGrid, QuadtreeLeaf, QuadtreeSky, QuadtreeTopology
from sl1mjax.refinement import (
    _HAAR_CHILD_DETAILS,
    BulkSplitSelection,
    LocalMergeEvaluation,
    MergeHysteresisState,
    QuadtreeObjectiveMetrics,
    ResidualHaarScore,
    _build_residual_haar_score,
    advance_merge_hysteresis,
    mergeable_parents,
    select_bulk_merges,
    select_bulk_splits,
)
from sl1mjax.voltage_beam import VoltageBeamModel
from sl1mjax.voltage_flux_refit import _time_batches

PRODUCTION_STOKES_I_BEAMS = ("static_scalar", "streamed_scalar", "diagonal_copolar")
DIAGNOSTIC_STOKES_I_BEAMS = ("full_jones",)


@dataclass(frozen=True)
class VoltageReconstructionConfig:
    """Fixed-beam Stokes-I reconstruction settings."""

    root_size: int = 104
    root_pixel_size_rad: float = np.deg2rad(16.0 / 3600.0)
    inference: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            solver="proximal_sgd",
            steps=40,
            learning_rate=0.05,
            patience=20,
            validation_interval=5,
            batch_grouping="times",
        )
    )
    tolerance: IntegrationTolerance = field(default_factory=IntegrationTolerance)
    operator: BeamOperatorConfig = field(default_factory=BeamOperatorConfig)
    leaf_penalty: float = 0.0
    max_rounds: int = 2
    max_depth: int = 2
    max_splits_per_round: int = 8
    max_split_fraction: float = 0.25
    target_improvement_fraction: float = 0.7
    min_parent_flux: float = 0.0
    min_curvature: float = 0.0
    min_eigenvalue_ratio: float = 1e-8
    ridge_relative: float = 1e-8
    merge_required_streak: int = 2
    max_merges_per_round: int | None = None
    strict_audit: bool = False
    parent_source: str = "central"
    screen_parent_limit: int | None = None
    kkt_max_batches: int | None = None
    operator_mode: Literal["vjp", "explicit_jax"] = "vjp"
    predict_batch_size_rows: int | None = None


@dataclass(frozen=True)
class VoltageFitResult:
    """One frozen-topology flux fit and its losses."""

    table: SkyComponentTable
    plan: IntegrationPlan
    planner_report: IntegrationPlannerReport
    flux: NDArray[np.float64]
    train_loss: float
    holdout_loss: float
    sparsity: float
    topology_cost: float
    kkt_residual: float
    steps: int
    converged: bool
    predictions: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    stationarity_history: tuple[float, ...] = ()
    objective_history: tuple[float, ...] = ()
    holdout_history: tuple[float, ...] = ()
    curve_steps: tuple[int, ...] = ()
    gradient: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class VoltageTopologyRound:
    """One split/merge round and the fit that followed it."""

    index: int
    screen: tuple[ResidualHaarScore, ...]
    shortlist: tuple[ResidualHaarScore, ...]
    selected_splits: tuple[QuadtreeLeaf, ...]
    accepted_splits: tuple[QuadtreeLeaf, ...]
    rejected_prefixes: tuple[tuple[QuadtreeLeaf, ...], ...]
    merge_evaluations: tuple[LocalMergeEvaluation, ...]
    accepted_merges: tuple[QuadtreeLeaf, ...]
    fit: VoltageFitResult
    audit: IntegrationAuditReport


@dataclass(frozen=True)
class VoltageReconstructionResult:
    """Completed inner Stokes-I reconstruction for one beam candidate."""

    beam_mode: str
    table: SkyComponentTable
    fit: VoltageFitResult
    rounds: tuple[VoltageTopologyRound, ...]
    stop_reason: str
    diagnostics: Mapping[str, Any]
    elapsed_s: float
    audit: IntegrationAuditReport
    hysteresis: MergeHysteresisState = field(default_factory=MergeHysteresisState.empty)


def stokes_i_beam(
    mode: str,
    *,
    allow_unfrozen_full_jones: bool = False,
) -> Any:
    """Return a production Stokes-I candidate, or refuse full Jones."""

    selected = str(mode)
    if selected == "full_jones":
        if not allow_unfrozen_full_jones:
            raise ValueError(
                "full Jones is not a production Stokes-I candidate until "
                "its scientific freeze gate is satisfied"
            )
        from sl1mjax.cassbeam_beam import CassbeamCBandVoltageBeam, load_cassbeam_cband_artifact

        return CassbeamCBandVoltageBeam(
            load_cassbeam_cband_artifact(),
            off_diagonal=True,
            allow_unfrozen=True,
        )
    if selected not in PRODUCTION_STOKES_I_BEAMS:
        raise ValueError(f"unknown Stokes-I beam candidate {selected!r}")
    return voltage_beam_for_mode(selected)


def starting_central_table(
    *,
    root_size: int,
    root_pixel_size_rad: float,
    mosaic_phase_centre_rad: tuple[float, float],
    flux: NDArray[np.float64] | None = None,
    catalogue: Sequence[SkyComponent] = (),
) -> SkyComponentTable:
    """Build the Phase 6 starting geometry: central roots plus optional deltas."""

    from sl1mjax.quadtree import quadtree_sky_from_regular_grid

    values = (
        np.zeros(root_size * root_size, dtype=np.float64)
        if flux is None
        else np.asarray(flux, dtype=np.float64).reshape(-1)
    )
    if values.size != root_size * root_size:
        raise ValueError("flux must contain one value per root cell")
    sky = quadtree_sky_from_regular_grid(root_size, root_pixel_size_rad, values)
    central = sky_table_from_mosaic_components(
        (MosaicQuadtreeComponent("central", sky.topology, sky.flux),),
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
        source="phase6_starting_roots",
    )
    extras = []
    for component in catalogue:
        if component.family is ComponentFamily.COARSE_FIELD:
            raise ValueError("starting catalogue cannot include the overlapping coarse field")
        extras.append(component)
    if not extras:
        return central
    return sky_table_from_records(
        [*sky_table_to_records(central), *[_component_record(item) for item in extras]],
        mosaic_phase_centre_rad=mosaic_phase_centre_rad,
    )


def reconstruct_voltage_stokes_i(
    table: SkyComponentTable,
    blocks: VisibilityBlock | Sequence[VisibilityBlock],
    beam: VoltageBeamModel | str,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
    train_masks: Sequence[np.ndarray] | None = None,
    holdout_masks: Sequence[np.ndarray] | None = None,
    config: VoltageReconstructionConfig | None = None,
    beam_mode: str | None = None,
    pointing_ids: Sequence[str] | None = None,
    hysteresis: MergeHysteresisState | None = None,
    on_checkpoint: Callable[[VoltageFitResult, MergeHysteresisState], None] | None = None,
    skip_flux_optimize: bool = False,
) -> VoltageReconstructionResult:
    """Fit one beam candidate with frozen integration plans and validated splits.

    ``skip_flux_optimize`` evaluates the incoming table flux (predict + KKT)
    and continues topology from there. Split and merge refits still optimize.
    """

    selected = VoltageReconstructionConfig() if config is None else config
    packed = _as_blocks(blocks)
    state = require_beam_calibration_state(calibration_state)
    if isinstance(beam, str):
        model = stokes_i_beam(beam)
        mode_name = beam
    else:
        model = beam
        mode_name = beam_mode or str(getattr(beam, "model_id", type(beam).__name__))
    _refuse_coarse_field(table)
    trains = _default_masks(packed, train_masks, default=True)
    holdouts = _default_masks(packed, holdout_masks, default=False)
    _validate_reconstruction_masks(packed, trains, holdouts)
    positions = np.asarray(antenna_position_m, dtype=np.float64)
    started = perf_counter()
    current = _fit_table(
        table,
        packed,
        model,
        antenna_position_m=positions,
        calibration_state=state,
        train_masks=trains,
        holdout_masks=holdouts,
        config=selected,
        pointing_ids=pointing_ids,
        skip_flux_optimize=skip_flux_optimize,
    )
    rounds: list[VoltageTopologyRound] = []
    hysteresis_state = MergeHysteresisState.empty() if hysteresis is None else hysteresis
    if on_checkpoint is not None:
        on_checkpoint(current, hysteresis_state)
    stop_reason = "maximum_rounds"
    for index in range(selected.max_rounds):
        screen = screen_virtual_splits(
            current,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            train_masks=trains,
            config=selected,
        )
        shortlist, selection = _exact_shortlist_selection(
            current,
            screen,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            train_masks=trains,
            config=selected,
        )
        accepted, rejected, split_fit = accept_ranked_splits(
            current,
            selection.selected,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            train_masks=trains,
            holdout_masks=holdouts,
            config=selected,
            pointing_ids=pointing_ids,
        )
        working = split_fit if accepted else current
        merge_evals = evaluate_virtual_merges(
            working,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            train_masks=trains,
            holdout_masks=holdouts,
            config=selected,
        )
        hysteresis_state = advance_merge_hysteresis(
            hysteresis_state,
            merge_evals,
            just_split=accepted,
        )
        merge_selection = select_bulk_merges(
            merge_evals,
            hysteresis_state,
            _central_leaf_count(working.table),
            required_streak=selected.merge_required_streak,
            max_merges=selected.max_merges_per_round,
        )
        merged_fit = working
        accepted_merges: tuple[QuadtreeLeaf, ...] = ()
        if merge_selection.selected:
            merged_table = apply_central_merges(working.table, merge_selection.selected, selected)
            candidate = _fit_table(
                merged_table,
                packed,
                model,
                antenna_position_m=positions,
                calibration_state=state,
                train_masks=trains,
                holdout_masks=holdouts,
                config=selected,
                pointing_ids=pointing_ids,
            )
            if _holdout_improved(working, candidate):
                merged_fit = candidate
                accepted_merges = merge_selection.selected
        audit = _audit_fit(
            merged_fit,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            config=selected,
        )
        rounds.append(
            VoltageTopologyRound(
                index=index,
                screen=screen,
                shortlist=shortlist,
                selected_splits=selection.selected,
                accepted_splits=accepted,
                rejected_prefixes=rejected,
                merge_evaluations=merge_evals,
                accepted_merges=accepted_merges,
                fit=merged_fit,
                audit=audit,
            )
        )
        if on_checkpoint is not None:
            on_checkpoint(merged_fit, hysteresis_state)
        if merged_fit.table.components == current.table.components:
            stop_reason = "no_accepted_topology_change"
            current = merged_fit
            break
        current = merged_fit
    if rounds:
        final_audit = rounds[-1].audit
    else:
        final_audit = _audit_fit(
            current,
            packed,
            model,
            antenna_position_m=positions,
            calibration_state=state,
            config=selected,
        )
    elapsed = perf_counter() - started
    return VoltageReconstructionResult(
        beam_mode=mode_name,
        table=current.table,
        fit=current,
        rounds=tuple(rounds),
        stop_reason=stop_reason,
        diagnostics=_diagnostics(mode_name, current, rounds, elapsed, final_audit),
        elapsed_s=elapsed,
        audit=final_audit,
        hysteresis=hysteresis_state,
    )


def reconstruct_stokes_i_candidates(
    table: SkyComponentTable,
    blocks: VisibilityBlock | Sequence[VisibilityBlock],
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str = "casa_parang_true",
    train_masks: Sequence[np.ndarray] | None = None,
    holdout_masks: Sequence[np.ndarray] | None = None,
    config: VoltageReconstructionConfig | None = None,
    beam_modes: Sequence[str] = PRODUCTION_STOKES_I_BEAMS,
    pointing_ids: Sequence[str] | None = None,
) -> dict[str, VoltageReconstructionResult]:
    """Run the same starting sky through every production Stokes-I beam."""

    results = {}
    for mode in beam_modes:
        results[mode] = reconstruct_voltage_stokes_i(
            table,
            blocks,
            mode,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            train_masks=train_masks,
            holdout_masks=holdout_masks,
            config=config,
            pointing_ids=pointing_ids,
        )
    return results


def screen_virtual_splits(
    fit: VoltageFitResult,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    candidates: Sequence[QuadtreeLeaf] | None = None,
) -> tuple[ResidualHaarScore, ...]:
    """Score every eligible parent from one residual adjoint over virtual children."""

    sky = _central_sky(fit.table, config)
    available = tuple(
        leaf
        for leaf in sky.topology.leaves
        if leaf.level < config.max_depth
        and float(sky.flux[sky.leaves.index(leaf)]) >= config.min_parent_flux
    )
    if candidates is None:
        selected_candidates = available
        if (
            config.screen_parent_limit is not None
            and len(selected_candidates) > config.screen_parent_limit
        ):
            selected_candidates = _shortlist_parents_by_gradient(
                fit,
                selected_candidates,
                int(config.screen_parent_limit),
            )
            print(
                f"screen shortlist {len(selected_candidates)} of {len(available)}",
                flush=True,
            )
    else:
        allowed = set(available)
        selected_candidates = tuple(leaf for leaf in candidates if leaf in allowed)
    candidates = selected_candidates
    if not candidates:
        return ()
    print(f"screen {len(candidates)} parents", flush=True)
    child_sky, child_index = _virtual_child_sky(sky, candidates)
    child_table = _rebuild_central_table(fit.table, child_sky)
    child_plan = integration_plan_from_table(child_table)
    leaf_index = _central_leaf_parent_index(child_table, child_plan)
    missing = [child for child in child_index if child not in leaf_index]
    if missing:
        raise ValueError("virtual-child plan is missing a central child parent")
    child_gradient = np.zeros(child_sky.flux.size, dtype=np.float64)
    weight_sum = 0.0
    residuals = tuple(fit.residuals)
    n_batch = 0
    screen_config = replace(
        config,
        inference=replace(
            config.inference, batch_size_rows=_predict_row_capacity(config)
        ),
    )
    for batch in _iter_bounded_batches(blocks, train_masks, screen_config):
        n_batch += 1
        if n_batch == 1 or n_batch % 20 == 0:
            print(f"screen batch {n_batch}", flush=True)
        residual = residuals[batch.source_index]
        packed = np.zeros_like(batch.block.visibility)
        count = int(batch.source_rows.size)
        packed[:count] = residual[batch.source_rows]
        weight = _sample_weight(batch.block, batch.mask)
        weight_sum += float(np.sum(weight))
        adjoint = np.asarray(
            adjoint_voltage_from_plan(
                np.where(weight > 0, packed * weight, 0.0),
                batch.block,
                child_plan,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                config=config.operator,
                backend="jax",
            ),
            dtype=np.float64,
        )
        for child, local in child_index.items():
            child_gradient[local] += adjoint[leaf_index[child]]
    if weight_sum <= 0:
        raise ValueError("train masks contain no finite positive-weight samples")
    child_gradient *= 2.0 / weight_sum
    representative = min(
        candidates,
        key=lambda leaf: leaf.iy**2 + leaf.ix**2 + leaf.level,
    )
    gram_sky, _gram_index = _virtual_child_sky(sky, (representative,))
    gram_table = _rebuild_central_table(fit.table, gram_sky)
    gram = _representative_haar_gram(
        gram_table,
        integration_plan_from_table(gram_table),
        blocks,
        train_masks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        candidates=(representative,),
    )
    scores = []
    for leaf in candidates:
        children = leaf.haar_children()
        indices = np.asarray([child_index[child] for child in children])
        gradient = child_gradient[indices] @ _HAAR_CHILD_DETAILS
        scores.append(
            _build_residual_haar_score(
                leaf,
                float(sky.flux[sky.leaves.index(leaf)]),
                gradient,
                gram,
                min_parent_flux=config.min_parent_flux,
                min_curvature=config.min_curvature,
                min_eigenvalue_ratio=config.min_eigenvalue_ratio,
                ridge_relative=config.ridge_relative,
                curvature_mode="voltage_batched_adjoint",
                constrain_child_flux=True,
            )
        )
    return tuple(scores)


def exact_virtual_child_scores(
    fit: VoltageFitResult,
    leaves: Sequence[QuadtreeLeaf],
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    child_flux_by_leaf: Mapping[QuadtreeLeaf, Sequence[float]] | None = None,
) -> tuple[float, ...]:
    """Return train-loss reductions from explicit virtual four-child skies."""

    sky = _central_sky(fit.table, config)
    baseline = fit.train_loss
    scores = []
    for leaf in leaves:
        child_flux = None if child_flux_by_leaf is None else child_flux_by_leaf.get(leaf)
        parent_flux = float(sky.flux[sky.leaves.index(leaf)])
        if child_flux is None:
            child_values = np.full(4, parent_flux / 4.0, dtype=np.float64)
        else:
            child_values = np.asarray(child_flux, dtype=np.float64)
        print(f"exact child {len(scores) + 1}/{len(leaves)}", flush=True)
        parent_table = _rebuild_central_table(
            fit.table,
            QuadtreeSky(sky.grid, (leaf,), np.asarray([parent_flux], dtype=np.float64)),
        )
        child_table = _rebuild_central_table(
            fit.table,
            QuadtreeSky(sky.grid, leaf.children(), child_values),
        )
        train_loss = _swapped_masked_loss(
            fit,
            parent_table,
            child_table,
            blocks,
            train_masks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            log_label="exact predict",
        )
        scores.append(float(baseline - train_loss))
    return tuple(scores)


def _planned_flux(
    table: SkyComponentTable,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
) -> tuple[IntegrationPlan, np.ndarray]:
    report = _plan_table(
        table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        pointing_ids=None,
    )
    return integration_plan_from_planner(table, report), _flux_vector(table)


def _swapped_masked_loss(
    fit: VoltageFitResult,
    subtract_table: SkyComponentTable,
    add_table: SkyComponentTable,
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
    log_label: str,
) -> float:
    """Loss of ``baseline - subtract + add`` on bounded JAX batches."""

    subtract_plan, subtract_flux = _planned_flux(
        subtract_table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
    )
    add_plan, add_flux = _planned_flux(
        add_table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
    )
    return _swapped_plan_loss(
        fit,
        subtract_plan,
        subtract_flux,
        add_plan,
        add_flux,
        blocks,
        masks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        log_label=log_label,
    )


def _swapped_plan_loss(
    fit: VoltageFitResult,
    subtract_plan: IntegrationPlan,
    subtract_flux: np.ndarray,
    add_plan: IntegrationPlan,
    add_flux: np.ndarray,
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
    log_label: str,
) -> float:
    predict_config = replace(
        config,
        inference=replace(
            config.inference, batch_size_rows=_predict_row_capacity(config)
        ),
    )
    numerator = 0.0
    denominator = 0.0
    n_batch = 0
    for batch in _iter_bounded_batches(blocks, masks, predict_config):
        n_batch += 1
        if n_batch == 1 or n_batch % 20 == 0:
            print(f"{log_label} batch {n_batch}", flush=True)
        subtracted = predict_voltage_from_plan(
            batch.block,
            subtract_plan,
            subtract_flux,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            backend="jax",
        ).visibility
        added = predict_voltage_from_plan(
            batch.block,
            add_plan,
            add_flux,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            backend="jax",
        ).visibility
        baseline = np.zeros_like(batch.block.visibility)
        count = int(batch.source_rows.size)
        baseline[:count] = np.asarray(fit.predictions[batch.source_index])[batch.source_rows]
        combined = baseline - np.asarray(subtracted) + np.asarray(added)
        num, den = _weighted_power(combined, batch.block, batch.mask)
        numerator += num
        denominator += den
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _exact_shortlist_selection(
    fit: VoltageFitResult,
    screen: Sequence[ResidualHaarScore],
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> tuple[tuple[ResidualHaarScore, ...], BulkSplitSelection]:
    """Mark an approximate shortlist, then rank with exact virtual-child scores."""

    eligible = tuple(item for item in screen if item.eligible)
    leaf_count = _central_leaf_count(fit.table)
    approximate = select_bulk_splits(
        eligible,
        leaf_count,
        target_improvement_fraction=config.target_improvement_fraction,
        max_split_fraction=config.max_split_fraction,
        max_splits=config.max_splits_per_round,
        split_cost=3.0 * config.leaf_penalty,
    )
    if not approximate.selected:
        return (), approximate
    by_leaf = {item.leaf: item for item in eligible}
    child_flux = {
        item.leaf: item.constrained_child_flux
        for item in eligible
        if item.leaf in set(approximate.selected) and item.constrained_child_flux is not None
    }
    reductions = exact_virtual_child_scores(
        fit,
        approximate.selected,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        train_masks=train_masks,
        config=config,
        child_flux_by_leaf=child_flux or None,
    )
    shortlist = tuple(
        replace(
            by_leaf[leaf],
            predicted_improvement=max(0.0, float(reduction)),
            raw_predicted_improvement=float(reduction),
            curvature_mode="voltage_exact_virtual_child",
        )
        for leaf, reduction in zip(approximate.selected, reductions, strict=True)
    )
    return shortlist, select_bulk_splits(
        shortlist,
        leaf_count,
        target_improvement_fraction=config.target_improvement_fraction,
        max_split_fraction=config.max_split_fraction,
        max_splits=config.max_splits_per_round,
        split_cost=3.0 * config.leaf_penalty,
    )


def accept_ranked_splits(
    fit: VoltageFitResult,
    ranked: Sequence[QuadtreeLeaf],
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    pointing_ids: Sequence[str] | None,
) -> tuple[tuple[QuadtreeLeaf, ...], tuple[tuple[QuadtreeLeaf, ...], ...], VoltageFitResult]:
    """Refit ranked prefixes, backtracking until inner validation improves."""

    rejected: list[tuple[QuadtreeLeaf, ...]] = []
    prefix = tuple(ranked)
    while prefix:
        print(f"accept prefix {len(prefix)}", flush=True)
        proposed = apply_central_splits(fit.table, prefix, config)
        candidate = _fit_table(
            proposed,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            train_masks=train_masks,
            holdout_masks=holdout_masks,
            config=config,
            pointing_ids=pointing_ids,
            compute_kkt=False,
        )
        if _holdout_improved(fit, candidate):
            scored = _predict_table(
                candidate.table,
                candidate.planner_report,
                blocks,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                train_masks=train_masks,
                holdout_masks=holdout_masks,
                config=config,
                flux=candidate.flux,
                plan=candidate.plan,
                compute_kkt=True,
            )
            return (
                prefix,
                tuple(rejected),
                replace(
                    scored,
                    steps=candidate.steps,
                    converged=scored.kkt_residual <= config.inference.kkt_tolerance,
                    stationarity_history=candidate.stationarity_history,
                    objective_history=candidate.objective_history,
                    holdout_history=candidate.holdout_history,
                    curve_steps=candidate.curve_steps,
                ),
            )
        rejected.append(prefix)
        if len(prefix) == 1:
            break
        prefix = prefix[: max(1, len(prefix) // 2)]
    return (), tuple(rejected), fit


def apply_central_splits(
    table: SkyComponentTable,
    leaves: Sequence[QuadtreeLeaf],
    config: VoltageReconstructionConfig,
    *,
    child_flux_by_leaf: Mapping[QuadtreeLeaf, Sequence[float]] | None = None,
) -> SkyComponentTable:
    """Replace selected central parents with four children. Nodes are not children."""

    sky = _central_sky(table, config)
    for leaf in leaves:
        child_flux = None if child_flux_by_leaf is None else child_flux_by_leaf.get(leaf)
        values = None if child_flux is None else np.asarray(child_flux, dtype=np.float64)
        sky = sky.split(leaf, values)
    return _rebuild_central_table(table, sky)


def apply_central_merges(
    table: SkyComponentTable,
    parents: Sequence[QuadtreeLeaf],
    config: VoltageReconstructionConfig,
) -> SkyComponentTable:
    sky = _central_sky(table, config)
    for parent in parents:
        sky = sky.merge(parent)
    return _rebuild_central_table(table, sky)


def evaluate_virtual_merges(
    fit: VoltageFitResult,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> tuple[LocalMergeEvaluation, ...]:
    sky = _central_sky(fit.table, config)
    baseline = _metrics(fit, config)
    evaluations = []
    for parent in mergeable_parents(sky.topology):
        children = parent.children()
        child_flux = (
            float(sky.flux[sky.leaves.index(children[0])]),
            float(sky.flux[sky.leaves.index(children[1])]),
            float(sky.flux[sky.leaves.index(children[2])]),
            float(sky.flux[sky.leaves.index(children[3])]),
        )
        merged = apply_central_merges(fit.table, (parent,), config)
        print(f"merge eval {len(evaluations) + 1}", flush=True)
        child_table = _rebuild_central_table(
            fit.table,
            QuadtreeSky(sky.grid, children, np.asarray(child_flux, dtype=np.float64)),
        )
        parent_table = _rebuild_central_table(
            fit.table,
            QuadtreeSky(
                sky.grid,
                (parent,),
                np.asarray([sum(child_flux)], dtype=np.float64),
            ),
        )
        child_plan, child_flux_vec = _planned_flux(
            child_table,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
        )
        parent_plan, parent_flux_vec = _planned_flux(
            parent_table,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
        )
        train_loss = _swapped_plan_loss(
            fit,
            child_plan,
            child_flux_vec,
            parent_plan,
            parent_flux_vec,
            blocks,
            train_masks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            log_label="merge train",
        )
        holdout_loss = _swapped_plan_loss(
            fit,
            child_plan,
            child_flux_vec,
            parent_plan,
            parent_flux_vec,
            blocks,
            holdout_masks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            log_label="merge holdout",
        )
        sparsity = float(config.inference.sparsity_weight * np.sum(_flux_vector(merged)))
        topology_cost = float(config.leaf_penalty * _central_leaf_count(merged))
        metrics = QuadtreeObjectiveMetrics(
            training_data=float(train_loss),
            sparsity=sparsity,
            topology=topology_cost,
            objective=float(train_loss + sparsity + topology_cost),
            holdout_data=None if not np.isfinite(holdout_loss) else float(holdout_loss),
        )
        evaluations.append(
            LocalMergeEvaluation(
                leaf=parent,
                children=children,
                child_flux=child_flux,
                parent_flux=float(sum(child_flux)),
                metrics=metrics,
                objective_change=metrics.objective - baseline.objective,
                predicted_improvement=baseline.objective - metrics.objective,
                holdout_change=(
                    None
                    if baseline.holdout_data is None or metrics.holdout_data is None
                    else metrics.holdout_data - baseline.holdout_data
                ),
            )
        )
    return tuple(evaluations)


def _fit_table(
    table: SkyComponentTable,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    pointing_ids: Sequence[str] | None,
    compute_kkt: bool = True,
    skip_flux_optimize: bool = False,
) -> VoltageFitResult:
    report = _plan_table(
        table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        pointing_ids=pointing_ids,
    )
    plan = integration_plan_from_planner(table, report)
    parents = _active_parents(table)
    initial = np.array([max(0.0, float(item.stokes_i_jy)) for item in parents], dtype=np.float64)
    if skip_flux_optimize:
        solved = _FluxSolve(flux=initial, steps=0)
    else:
        solved = _optimize_flux(
            plan,
            initial,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            train_masks=train_masks,
            holdout_masks=holdout_masks,
            config=config,
        )
    fitted = _table_with_flux(table, solved.flux)
    predicted = _predict_table(
        fitted,
        report,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        train_masks=train_masks,
        holdout_masks=holdout_masks,
        config=config,
        flux=solved.flux,
        plan=plan,
        compute_kkt=compute_kkt,
    )
    return replace(
        predicted,
        steps=solved.steps,
        converged=predicted.kkt_residual <= config.inference.kkt_tolerance,
        stationarity_history=solved.stationarity_history,
        objective_history=solved.objective_history,
        holdout_history=solved.holdout_history,
        curve_steps=solved.curve_steps,
    )


def _predict_table(
    table: SkyComponentTable,
    report: IntegrationPlannerReport | None,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    flux: np.ndarray,
    plan: IntegrationPlan | None = None,
    compute_kkt: bool = True,
) -> VoltageFitResult:
    planner = report or _plan_table(
        table,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
    )
    selected_plan = plan or integration_plan_from_planner(table, planner)
    predictions = []
    residuals = []
    train_num = 0.0
    train_den = 0.0
    hold_num = 0.0
    hold_den = 0.0
    for block, train, holdout in zip(blocks, train_masks, holdout_masks, strict=True):
        predicted = _predict_block_batched(
            block,
            selected_plan,
            flux,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
        )
        predictions.append(predicted)
        residuals.append(predicted - block.visibility)
        t_num, t_den = _weighted_power(predicted, block, train)
        h_num, h_den = _weighted_power(predicted, block, holdout)
        train_num += t_num
        train_den += t_den
        hold_num += h_num
        hold_den += h_den
    train_loss = train_num / train_den if train_den > 0 else float("nan")
    holdout_loss = hold_num / hold_den if hold_den > 0 else float("nan")
    sparsity = float(config.inference.sparsity_weight * np.sum(flux))
    topology_cost = float(config.leaf_penalty * _central_leaf_count(table))
    if compute_kkt:
        gradient = _total_gradient(
            selected_plan,
            flux,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            train_masks=train_masks,
            config=config,
        )
        kkt_residual = float(
            positive_l1_kkt_residual(
                jnp.asarray(flux),
                jnp.asarray(gradient),
                float(config.inference.sparsity_weight),
            )
        )
    else:
        kkt_residual = float("nan")
        gradient = None
    return VoltageFitResult(
        table=table,
        plan=selected_plan,
        planner_report=planner,
        flux=np.asarray(flux, dtype=np.float64),
        train_loss=float(train_loss),
        holdout_loss=float(holdout_loss),
        sparsity=sparsity,
        topology_cost=topology_cost,
        kkt_residual=kkt_residual,
        steps=0,
        converged=False,
        predictions=tuple(predictions),
        residuals=tuple(residuals),
        gradient=None if gradient is None else np.asarray(gradient, dtype=np.float64),
    )


@dataclass(frozen=True)
class _FluxSolve:
    flux: NDArray[np.float64]
    steps: int
    stationarity_history: tuple[float, ...] = ()
    objective_history: tuple[float, ...] = ()
    holdout_history: tuple[float, ...] = ()
    curve_steps: tuple[int, ...] = ()


def _optimize_flux(
    plan: IntegrationPlan,
    initial: np.ndarray,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> _FluxSolve:
    inference = config.inference
    if inference.solver != "proximal_sgd":
        raise ValueError(
            "voltage reconstruction uses time-grouped proximal SGD; "
            f"got solver={inference.solver!r}"
        )
    if inference.batch_grouping not in {"times", "rows"}:
        raise ValueError("batch_grouping must be 'times' or 'rows'")
    flux = np.asarray(initial, dtype=np.float64).reshape(-1)
    best = flux.copy()
    best_holdout = np.inf
    stale = 0
    completed = 0
    rng = np.random.default_rng(inference.random_seed)
    has_holdout = any(
        np.any(np.asarray(mask, dtype=bool) & block.active)
        for block, mask in zip(blocks, holdout_masks, strict=True)
    )
    penalty = float(inference.sparsity_weight)
    stationarity_history: list[float] = []
    objective_history: list[float] = []
    holdout_history: list[float] = []
    curve_steps: list[int] = []
    for step in range(1, inference.steps + 1):
        batch = _sample_training_batch(
            blocks,
            train_masks,
            config,
            rng,
        )
        _value, gradient = predict_voltage_from_plan_value_and_grad(
            jnp.asarray(flux),
            batch.block,
            plan,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            train_mask=batch.mask,
            operator_mode=config.operator_mode,
        )
        progress = (step - 1) / max(inference.steps - 1, 1)
        decay = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
        step_size = inference.learning_rate * decay
        flux = np.maximum(flux - step_size * (np.asarray(gradient) + penalty), 0.0)
        completed = step
        print(f"sgd step {step}/{inference.steps}", flush=True)
        if step % inference.validation_interval != 0 and step != inference.steps:
            continue
        candidate = np.asarray(flux, dtype=np.float64)
        train = _holdout_loss(
            plan,
            candidate,
            blocks,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            holdout_masks=train_masks,
            config=config,
            max_batches=2,
            label="train",
        )
        objective_history.append(float(train))
        curve_steps.append(int(step))
        print(f"train step {step} loss={train:.6g}", flush=True)
        if has_holdout:
            holdout = _holdout_loss(
                plan,
                candidate,
                blocks,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                holdout_masks=holdout_masks,
                config=config,
                max_batches=2,
            )
            holdout_history.append(float(holdout))
            print(f"holdout step {step} loss={holdout:.6g}", flush=True)
            improved = np.isfinite(holdout) and holdout < best_holdout - inference.min_delta
            if improved:
                best_holdout = holdout
                best = candidate.copy()
                stale = 0
            else:
                stale += 1
        else:
            best = candidate.copy()
            improved = True
            stale = 0
        if stale >= inference.patience:
            break
    return _FluxSolve(
        flux=np.asarray(best, dtype=np.float64),
        steps=completed,
        stationarity_history=tuple(stationarity_history),
        objective_history=tuple(objective_history),
        holdout_history=tuple(holdout_history),
        curve_steps=tuple(curve_steps),
    )


def _correlation_valid(result, sample: np.ndarray) -> np.ndarray:
    beam_ok = np.asarray(result.valid, dtype=bool)
    if beam_ok.shape == sample.shape:
        return beam_ok
    if beam_ok.ndim + 1 == sample.ndim and beam_ok.shape == sample.shape[:-1]:
        return np.broadcast_to(beam_ok[..., None], sample.shape)
    return np.ones_like(sample, dtype=bool)


def _predict_plan_value_and_grad_numpy(
    flux: np.ndarray,
    block: VisibilityBlock,
    plan: IntegrationPlan,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    train_mask: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Weighted MSE and parent gradient from the streamed NumPy adjoint."""

    result = predict_voltage_from_plan(
        block,
        plan,
        flux,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        backend="numpy",
    )
    predicted = np.asarray(result.visibility)
    sample = np.asarray(block.active if train_mask is None else train_mask, dtype=bool)
    if sample.shape != block.visibility.shape:
        raise ValueError("train_mask must match block.visibility")
    selected = sample & _correlation_valid(result, sample)
    weight = np.asarray(
        effective_weight(block.visibility, block.weight, ~selected),
        dtype=np.float64,
    )
    weight_sum = float(np.sum(weight))
    if weight_sum <= 0:
        return float("inf"), np.zeros(plan.parent_count, dtype=np.float64)
    residual = predicted - block.visibility
    data_loss = float(np.sum(weight * np.abs(residual) ** 2) / weight_sum)
    cotangent = np.where(weight > 0, (2.0 / weight_sum) * weight * residual, 0.0)
    gradient = np.asarray(
        adjoint_voltage_from_plan(
            cotangent,
            block,
            plan,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
        ),
        dtype=np.float64,
    )
    return data_loss, gradient


def _loss_and_gradient(
    plan: IntegrationPlan,
    flux: np.ndarray,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> tuple[float, np.ndarray]:
    numerator = 0.0
    denominator = 0.0
    gradient = np.zeros(flux.size, dtype=np.float64)
    n_batch = 0
    batches = _iter_bounded_batches(blocks, train_masks, config)
    if config.kkt_max_batches is not None:
        if config.kkt_max_batches < 1:
            raise ValueError("kkt_max_batches must be positive")
        if len(batches) > config.kkt_max_batches:
            rng = np.random.default_rng(config.inference.random_seed)
            chosen = np.sort(
                rng.choice(len(batches), config.kkt_max_batches, replace=False)
            )
            print(
                f"kkt batches {config.kkt_max_batches} of {len(batches)}",
                flush=True,
            )
            batches = tuple(batches[int(index)] for index in chosen)
    for batch in batches:
        n_batch += 1
        if n_batch == 1 or n_batch % 5 == 0 or n_batch == len(batches):
            print(f"kkt batch {n_batch}/{len(batches)}", flush=True)
        value, parent_grad = predict_voltage_from_plan_value_and_grad(
            jnp.asarray(flux),
            batch.block,
            plan,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            train_mask=batch.mask,
            operator_mode=config.operator_mode,
        )
        weight_sum = float(np.sum(_sample_weight(batch.block, batch.mask)))
        if weight_sum <= 0:
            continue
        numerator += float(value) * weight_sum
        denominator += weight_sum
        gradient += weight_sum * np.asarray(parent_grad, dtype=np.float64)
    if denominator <= 0:
        raise ValueError("train masks contain no finite positive-weight samples")
    data_loss = numerator / denominator
    gradient = gradient / denominator
    gradient = gradient + config.inference.sparsity_weight
    return data_loss + float(config.inference.sparsity_weight * np.sum(flux)), gradient


def _total_gradient(
    plan: IntegrationPlan,
    flux: np.ndarray,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> np.ndarray:
    _value, gradient = _loss_and_gradient(
        plan,
        flux,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        train_masks=train_masks,
        config=config,
    )
    return gradient - config.inference.sparsity_weight


def _holdout_loss(
    plan: IntegrationPlan,
    flux: np.ndarray,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    holdout_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    max_batches: int | None = None,
    label: str = "holdout",
) -> float:
    numerator = 0.0
    denominator = 0.0
    batches = _iter_bounded_batches(blocks, holdout_masks, config)
    if max_batches is not None:
        batches = batches[: int(max_batches)]
    print(f"{label} batches {len(batches)}", flush=True)
    for index, batch in enumerate(batches, start=1):
        if index == 1 or index % 20 == 0:
            print(f"{label} batch {index}", flush=True)
        result = predict_voltage_from_plan(
            batch.block,
            plan,
            flux,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            backend="jax",
        )
        num, den = _weighted_power(result.visibility, batch.block, batch.mask)
        numerator += num
        denominator += den
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _plan_table(
    table: SkyComponentTable,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
    pointing_ids: Sequence[str] | None = None,
) -> IntegrationPlannerReport:
    n_active = sum(1 for item in table.components if item.active)
    print(f"plan integration parents={n_active}", flush=True)
    return plan_integration(
        table,
        tuple(blocks),
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        tolerance=config.tolerance,
        config=config.operator,
        pointing_ids=pointing_ids,
    )


def _representative_haar_gram(
    child_table: SkyComponentTable,
    child_plan: IntegrationPlan,
    blocks: Sequence[VisibilityBlock],
    train_masks: Sequence[np.ndarray],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
    candidates: Sequence[QuadtreeLeaf],
) -> np.ndarray:
    representative = min(
        candidates,
        key=lambda leaf: leaf.iy**2 + leaf.ix**2 + leaf.level,
    )
    children = representative.haar_children()
    leaf_index = _central_leaf_parent_index(child_table, child_plan)
    missing = [child for child in children if child not in leaf_index]
    if missing:
        raise ValueError("virtual-child plan is missing a representative Haar parent")
    gram = np.zeros((3, 3), dtype=np.float64)
    weight_sum = 0.0
    gram_config = replace(
        config,
        inference=replace(
            config.inference, batch_size_rows=_predict_row_capacity(config)
        ),
    )
    for batch in _iter_bounded_batches(blocks, train_masks, gram_config):
        weight = _sample_weight(batch.block, batch.mask)
        weight_sum += float(np.sum(weight))
        responses = []
        for detail in _HAAR_CHILD_DETAILS.T:
            fluxes = np.zeros(child_plan.parent_count, dtype=np.float64)
            for child, amplitude in zip(children, detail, strict=True):
                fluxes[leaf_index[child]] = amplitude
            predicted = predict_voltage_from_plan(
                batch.block,
                child_plan,
                fluxes,
                beam,
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                config=config.operator,
                backend="jax",
            ).visibility
            responses.append(np.asarray(predicted).reshape(-1))
        matrix = np.stack(responses, axis=1)
        flat = weight.reshape(-1)
        gram += np.real(matrix.conj().T @ (flat[:, None] * matrix))
    if weight_sum <= 0:
        raise ValueError("train masks contain no finite positive-weight samples")
    return (2.0 / weight_sum) * gram


def _shortlist_parents_by_gradient(
    fit: VoltageFitResult,
    leaves: Sequence[QuadtreeLeaf],
    limit: int,
) -> tuple[QuadtreeLeaf, ...]:
    if fit.gradient is None:
        raise ValueError("parent-gradient shortlist requires a stored residual gradient")
    parent_index = _central_leaf_parent_index(fit.table, fit.plan)
    gradient = np.asarray(fit.gradient, dtype=np.float64)
    ranked = sorted(
        leaves,
        key=lambda leaf: abs(float(gradient[parent_index[leaf]])) if leaf in parent_index else -1.0,
        reverse=True,
    )
    return tuple(ranked[:limit])


def _virtual_child_sky(
    sky: QuadtreeSky, candidates: Sequence[QuadtreeLeaf]
) -> tuple[QuadtreeSky, dict[QuadtreeLeaf, int]]:
    leaves: list[QuadtreeLeaf] = []
    flux: list[float] = []
    for parent in candidates:
        parent_flux = float(sky.flux[sky.leaves.index(parent)])
        for child in parent.haar_children():
            leaves.append(child)
            flux.append(parent_flux / 4.0)
    virtual = QuadtreeSky(sky.grid, tuple(leaves), np.asarray(flux, dtype=np.float64))
    index = {leaf: i for i, leaf in enumerate(virtual.leaves)}
    return virtual, index


def _central_sky(table: SkyComponentTable, config: VoltageReconstructionConfig) -> QuadtreeSky:
    leaves: list[QuadtreeLeaf] = []
    flux: list[float] = []
    for component in table.components:
        if component.family is not ComponentFamily.CENTRAL_TREE or not component.active:
            continue
        if component.level is None or component.iy is None or component.ix is None:
            raise ValueError("central squares require level, iy, and ix")
        leaves.append(QuadtreeLeaf(int(component.level), int(component.iy), int(component.ix)))
        flux.append(float(component.stokes_i_jy))
    if not leaves:
        raise ValueError("sky table has no active central leaves")
    topology = QuadtreeTopology(
        QuadtreeGrid(config.root_size, config.root_pixel_size_rad),
        tuple(leaves),
    )
    by_leaf = dict(zip(leaves, flux, strict=True))
    return QuadtreeSky(
        topology.grid,
        topology.leaves,
        np.asarray([by_leaf[leaf] for leaf in topology.leaves], dtype=np.float64),
    )


def _rebuild_central_table(table: SkyComponentTable, sky: QuadtreeSky) -> SkyComponentTable:
    central = sky_table_from_mosaic_components(
        (MosaicQuadtreeComponent(table_source_name(table), sky.topology, sky.flux),),
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
        source=table.source,
    )
    others = [item for item in table.components if item.family is not ComponentFamily.CENTRAL_TREE]
    if not others:
        return central
    return sky_table_from_records(
        [*sky_table_to_records(central), *[_component_record(item) for item in others]],
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
    )


def table_source_name(table: SkyComponentTable) -> str:
    for component in table.components:
        if component.family is ComponentFamily.CENTRAL_TREE:
            return str(component.provenance.get("mosaic_name") or "central")
    return "central"


def _table_with_flux(table: SkyComponentTable, flux: np.ndarray) -> SkyComponentTable:
    parents = _active_parents(table)
    if flux.size != len(parents):
        raise ValueError("flux must match the active parent count")
    records = sky_table_to_records(table)
    by_id = {parent.component_id: float(value) for parent, value in zip(parents, flux, strict=True)}
    for record in records:
        if record["component_id"] in by_id:
            record["stokes_i_jy"] = by_id[record["component_id"]]
    return sky_table_from_records(records, mosaic_phase_centre_rad=table.mosaic_phase_centre_rad)


def _flux_vector(table: SkyComponentTable) -> np.ndarray:
    return np.asarray([item.stokes_i_jy for item in _active_parents(table)], dtype=np.float64)


def _active_parents(table: SkyComponentTable) -> tuple[SkyComponent, ...]:
    return tuple(component for component in table.components if component.active)


def _central_leaf_count(table: SkyComponentTable) -> int:
    return sum(
        1
        for component in table.components
        if component.family is ComponentFamily.CENTRAL_TREE and component.active
    )


def _refuse_coarse_field(table: SkyComponentTable) -> None:
    if any(
        component.family is ComponentFamily.COARSE_FIELD and component.active
        for component in table.components
    ):
        raise ValueError("Phase 6 Stokes-I reconstruction refuses the overlapping coarse field")


def _holdout_improved(baseline: VoltageFitResult, candidate: VoltageFitResult) -> bool:
    if not np.isfinite(baseline.holdout_loss) or not np.isfinite(candidate.holdout_loss):
        return False
    baseline_objective = baseline.holdout_loss + baseline.sparsity + baseline.topology_cost
    candidate_objective = candidate.holdout_loss + candidate.sparsity + candidate.topology_cost
    return candidate_objective < baseline_objective - 1.0e-15


def _metrics(
    fit: VoltageFitResult, config: VoltageReconstructionConfig
) -> QuadtreeObjectiveMetrics:
    holdout = None if not np.isfinite(fit.holdout_loss) else float(fit.holdout_loss)
    return QuadtreeObjectiveMetrics(
        training_data=float(fit.train_loss),
        sparsity=float(fit.sparsity),
        topology=float(fit.topology_cost),
        objective=float(fit.train_loss + fit.sparsity + fit.topology_cost),
        holdout_data=holdout,
    )


def _diagnostics(
    beam_mode: str,
    fit: VoltageFitResult,
    rounds: Sequence[VoltageTopologyRound],
    elapsed_s: float,
    audit: IntegrationAuditReport,
) -> dict[str, Any]:
    widths: dict[str, int] = {}
    families: dict[str, float] = {}
    depths: dict[int, int] = {}
    for component in fit.table.components:
        if not component.active:
            continue
        families[component.family.value] = (
            families.get(component.family.value, 0.0) + component.stokes_i_jy
        )
        if component.width_rad > 0:
            label = f"{component.width_rad * 206264.80624709636:.6g}"
            widths[label] = widths.get(label, 0) + 1
    for parent, depth in fit.planner_report.depth_by_parent().items():
        depths[depth] = depths.get(depth, 0) + 1
        del parent
    return {
        "beam_mode": beam_mode,
        "train_loss": fit.train_loss,
        "holdout_loss": fit.holdout_loss,
        "leaf_counts_by_width_arcsec": widths,
        "flux_by_family": families,
        "integration_nodes": fit.plan.node_count,
        "depth_histogram": {str(key): value for key, value in sorted(depths.items())},
        "split_accepted": [tuple(map(str, item.accepted_splits)) for item in rounds],
        "merge_accepted": [tuple(map(str, item.accepted_merges)) for item in rounds],
        "kkt_residual": fit.kkt_residual,
        "optimizer_converged": fit.converged,
        "optimizer_steps": fit.steps,
        "n_predictor_call": fit.planner_report.provenance.get("n_predictor_call"),
        "elapsed_s": elapsed_s,
        "audit_under_resolved": bool(audit.under_resolved),
        "parent_count": fit.plan.parent_count,
        "node_count": fit.plan.node_count,
        "nodes_are_not_parameters": fit.plan.parent_count != fit.plan.node_count
        or all(depth == 0 for depth in fit.planner_report.depth_by_parent().values()),
    }


def _audit_fit(
    fit: VoltageFitResult,
    blocks: Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
) -> IntegrationAuditReport:
    audit = audit_integration_plan(
        fit.table,
        fit.planner_report,
        blocks,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config.operator,
    )
    n_under = len(audit.under_resolved)
    print(f"audit under_resolved={n_under}/{len(audit.findings)}", flush=True)
    if config.strict_audit and n_under:
        raise ValueError("integration audit reported under-resolved parents")
    return audit


def _sample_weight(block: VisibilityBlock, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool) & block.active
    return np.asarray(
        effective_weight(block.visibility, block.weight, ~selected),
        dtype=np.float64,
    )


def _central_leaf_parent_index(
    table: SkyComponentTable, plan: IntegrationPlan
) -> dict[QuadtreeLeaf, int]:
    by_id = {parent_id: index for index, parent_id in enumerate(plan.parent_id)}
    mapping: dict[QuadtreeLeaf, int] = {}
    for component in table.components:
        if component.family is not ComponentFamily.CENTRAL_TREE or not component.active:
            continue
        if component.level is None or component.iy is None or component.ix is None:
            raise ValueError("central squares require level, iy, and ix")
        if component.component_id not in by_id:
            continue
        mapping[QuadtreeLeaf(int(component.level), int(component.iy), int(component.ix))] = by_id[
            component.component_id
        ]
    return mapping


@dataclass(frozen=True)
class VisibilityBatch:
    """One physically sliced, row-bounded visibility batch."""

    block: VisibilityBlock
    mask: np.ndarray
    source_index: int
    source_rows: np.ndarray


def _batch_row_capacity(config: VoltageReconstructionConfig) -> int:
    return int(config.inference.batch_size_rows)


def _predict_row_capacity(config: VoltageReconstructionConfig) -> int:
    if config.predict_batch_size_rows is None:
        return _batch_row_capacity(config)
    return max(1, int(config.predict_batch_size_rows))


def _require_bounded_batch(
    batch: VisibilityBatch, config: VoltageReconstructionConfig
) -> VisibilityBatch:
    inference = config.inference
    n_rows = int(batch.block.uvw_m.shape[0])
    if n_rows > inference.batch_size_rows:
        raise ValueError(
            "visibility batch has "
            f"{n_rows} rows, exceeding batch_size_rows={inference.batch_size_rows}"
        )
    if inference.batch_grouping == "times" and np.unique(batch.block.time_s).size > 1:
        raise ValueError("time batch must be a single unique timestamp")
    if batch.mask.shape != batch.block.visibility.shape:
        raise ValueError("batch mask must match the sliced visibility block")
    return batch


def _slice_visibility_batch(
    block: VisibilityBlock,
    rows: np.ndarray,
    mask: np.ndarray,
    *,
    source_index: int,
    pad_to: int,
    config: VoltageReconstructionConfig,
) -> VisibilityBatch:
    selected = np.asarray(rows, dtype=np.int32).reshape(-1)
    if selected.size == 0:
        raise ValueError("visibility batch requires at least one row")
    sliced = block.select_rows(selected, pad_to=pad_to)
    sliced_mask = np.zeros(sliced.shape, dtype=bool)
    sliced_mask[: selected.size] = np.asarray(mask, dtype=bool)[selected]
    return _require_bounded_batch(
        VisibilityBatch(
            block=sliced,
            mask=sliced_mask,
            source_index=source_index,
            source_rows=selected,
        ),
        config,
    )


def _predict_block_batched(
    block: VisibilityBlock,
    plan: IntegrationPlan,
    flux: np.ndarray,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: VoltageReconstructionConfig,
) -> np.ndarray:
    """Predict one pointing with the same bounded JAX batches as SGD."""

    predicted = np.zeros_like(block.visibility)
    cover = np.asarray(block.active, dtype=bool)
    if not np.any(cover):
        return predicted
    n_batch = 0
    predict_config = replace(
        config,
        inference=replace(
            config.inference, batch_size_rows=_predict_row_capacity(config)
        ),
    )
    for batch in _iter_bounded_batches((block,), (cover,), predict_config):
        n_batch += 1
        if n_batch == 1 or n_batch % 20 == 0:
            print(f"predict batch {n_batch}", flush=True)
        result = predict_voltage_from_plan(
            batch.block,
            plan,
            flux,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config.operator,
            backend="jax",
        )
        count = int(batch.source_rows.size)
        predicted[batch.source_rows] = np.asarray(result.visibility)[:count]
    return predicted


def _iter_bounded_batches(
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
) -> tuple[VisibilityBatch, ...]:
    """Cover every selected sample with physically sliced, padded batches."""

    inference = config.inference
    capacity = _batch_row_capacity(config)
    packed_blocks = tuple(blocks)
    packed_masks = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    batches: list[VisibilityBatch] = []
    if inference.batch_grouping == "times":
        groups = _time_batches(packed_blocks, packed_masks)
        for block_index, rows in groups:
            order = np.asarray(rows, dtype=np.int32)
            for start in range(0, order.size, capacity):
                chunk = order[start : start + capacity]
                batches.append(
                    _slice_visibility_batch(
                        packed_blocks[block_index],
                        chunk,
                        packed_masks[block_index],
                        source_index=block_index,
                        pad_to=capacity,
                        config=config,
                    )
                )
    else:
        for block_index, (block, mask) in enumerate(zip(packed_blocks, packed_masks, strict=True)):
            rows = np.flatnonzero(np.any(mask & block.active, axis=(1, 2)))
            for start in range(0, rows.size, capacity):
                chunk = rows[start : start + capacity]
                batches.append(
                    _slice_visibility_batch(
                        block,
                        chunk,
                        mask,
                        source_index=block_index,
                        pad_to=capacity,
                        config=config,
                    )
                )
    if not batches:
        raise ValueError("train masks contain no eligible rows")
    return tuple(batches)


def _sample_training_batch(
    blocks: Sequence[VisibilityBlock],
    train_masks: Sequence[np.ndarray],
    config: VoltageReconstructionConfig,
    rng: np.random.Generator,
) -> VisibilityBatch:
    inference = config.inference
    packed_blocks = tuple(blocks)
    packed_masks = tuple(np.asarray(mask, dtype=bool) for mask in train_masks)
    capacity = _batch_row_capacity(config)
    if inference.batch_grouping == "times":
        groups = _time_batches(packed_blocks, packed_masks)
        weights = np.asarray([group.size for _index, group in groups], dtype=np.float64)
        block_index, rows = groups[int(rng.choice(len(groups), p=weights / weights.sum()))]
        if rows.size > capacity:
            rows = rng.choice(rows, capacity, replace=False)
        return _slice_visibility_batch(
            packed_blocks[block_index],
            rows,
            packed_masks[block_index],
            source_index=block_index,
            pad_to=capacity,
            config=config,
        )
    eligible: list[tuple[int, np.ndarray]] = []
    for block_index, (block, mask) in enumerate(zip(packed_blocks, packed_masks, strict=True)):
        rows = np.flatnonzero(np.any(mask & block.active, axis=(1, 2)))
        if rows.size:
            eligible.append((block_index, rows))
    if not eligible:
        raise ValueError("train masks contain no eligible rows")
    sizes = np.asarray([rows.size for _index, rows in eligible], dtype=np.float64)
    block_index, rows = eligible[int(rng.choice(len(eligible), p=sizes / sizes.sum()))]
    if rows.size > capacity:
        rows = rng.choice(rows, capacity, replace=False)
    return _slice_visibility_batch(
        packed_blocks[block_index],
        rows,
        packed_masks[block_index],
        source_index=block_index,
        pad_to=capacity,
        config=config,
    )


def merge_hysteresis_to_records(state: MergeHysteresisState) -> dict[str, dict[str, int]]:
    def key(leaf: QuadtreeLeaf) -> str:
        return f"{leaf.level}:{leaf.iy}:{leaf.ix}"

    return {
        "eligible_streak": {key(leaf): int(value) for leaf, value in state.eligible_streak.items()},
        "split_cooldown": {key(leaf): int(value) for leaf, value in state.split_cooldown.items()},
    }


def merge_hysteresis_from_records(payload: Mapping[str, Any]) -> MergeHysteresisState:
    def leaf(text: str) -> QuadtreeLeaf:
        level, iy, ix = str(text).split(":")
        return QuadtreeLeaf(int(level), int(iy), int(ix))

    return MergeHysteresisState(
        eligible_streak={
            leaf(name): int(value)
            for name, value in dict(payload.get("eligible_streak", {})).items()
        },
        split_cooldown={
            leaf(name): int(value)
            for name, value in dict(payload.get("split_cooldown", {})).items()
        },
    )


def _validate_reconstruction_masks(
    blocks: Sequence[VisibilityBlock],
    train_masks: Sequence[np.ndarray],
    holdout_masks: Sequence[np.ndarray],
) -> None:
    if len(train_masks) != len(blocks) or len(holdout_masks) != len(blocks):
        raise ValueError("masks must contain one array per block")
    for index, (block, train, holdout) in enumerate(
        zip(blocks, train_masks, holdout_masks, strict=True)
    ):
        if train.shape != block.shape:
            raise ValueError(f"train_masks[{index}] must match its visibility block")
        if holdout.shape != block.shape:
            raise ValueError(f"holdout_masks[{index}] must match its visibility block")
        if np.any(train & holdout):
            raise ValueError("train and holdout masks must be disjoint")


def _weighted_power(
    prediction: np.ndarray, block: VisibilityBlock, mask: np.ndarray
) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool) & block.active
    finite = np.isfinite(block.weight) & (block.weight > 0)
    weight = np.where(selected & finite, block.weight, 0.0)
    residual = np.asarray(prediction) - block.visibility
    usable = (weight > 0) & np.isfinite(residual.real) & np.isfinite(residual.imag)
    if not np.any(usable):
        return 0.0, 0.0
    power = float(np.sum(weight[usable] * np.abs(residual[usable]) ** 2))
    return power, float(np.sum(weight[usable]))


def _as_blocks(blocks: VisibilityBlock | Sequence[VisibilityBlock]) -> tuple[VisibilityBlock, ...]:
    if isinstance(blocks, VisibilityBlock):
        return (blocks,)
    packed = tuple(blocks)
    if not packed:
        raise ValueError("reconstruction requires at least one visibility block")
    return packed


def _default_masks(
    blocks: Sequence[VisibilityBlock],
    masks: Sequence[np.ndarray] | None,
    *,
    default: bool,
) -> tuple[np.ndarray, ...]:
    if masks is None:
        if default:
            return tuple(np.asarray(block.active, dtype=bool) for block in blocks)
        return tuple(np.zeros(block.visibility.shape, dtype=bool) for block in blocks)
    packed = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if len(packed) != len(blocks):
        raise ValueError("masks must contain one array per block")
    return packed


def _component_record(component: SkyComponent) -> dict[str, Any]:
    return {
        "component_id": component.component_id,
        "family": component.family.value,
        "basis_type": component.basis_type.value,
        "l_rad": component.l_rad,
        "m_rad": component.m_rad,
        "stokes_i_jy": component.stokes_i_jy,
        "width_rad": component.width_rad,
        "level": component.level,
        "iy": component.iy,
        "ix": component.ix,
        "parent_id": component.parent_id,
        "active": component.active,
        "splitting_permitted": component.splitting_permitted,
        "provenance": dict(component.provenance),
    }
