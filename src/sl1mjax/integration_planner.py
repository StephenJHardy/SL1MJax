"""Numerical integration planner for finite-pixel voltage prediction.

The planner runs outside JIT. It compares successive subcell depths on a
geometry-only probe and assigns one order to each fitted parent and
pointing. It may use declared data weights and a training-derived flux
floor. It must not inspect measured visibilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_aware_imaging import (
    SkyBasisType,
    SkyComponent,
    SkyComponentTable,
    sky_table_from_records,
    sky_table_to_records,
)
from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import BeamOperatorConfig, BeamOperatorResult
from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.cassbeam_beam import CassbeamCBandVoltageBeam
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    IntegrationParentPolicy,
    IntegrationPlan,
    integration_plan_from_table,
    predict_voltage_from_plan,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import (
    BeamEvaluation,
    CompositeScalarVoltageBeam,
    VoltageBeamModel,
    beam_coordinates,
)

_CROSS_HANDS = {Correlation.RL, Correlation.LR}


class RefinementReason(StrEnum):
    """Why a parent kept or increased its integration depth."""

    CONVERGED = "converged"
    VISIBILITY_ERROR = "visibility_error"
    BEAM_NULL = "beam_null"
    PHASE_REVERSAL = "phase_reversal"
    RASTER_BOUNDARY = "raster_boundary"
    OUTER_HANDOVER = "outer_handover"
    FREQUENCY_NODE = "frequency_node"
    VALIDITY_BOUNDARY = "validity_boundary"
    SQUINT_GRADIENT = "squint_gradient"
    LEAKAGE_GRADIENT = "leakage_gradient"
    MAX_DEPTH = "max_depth"
    DELTA = "delta"


@dataclass(frozen=True)
class IntegrationTolerance:
    """Absolute and relative visibility-error thresholds.

    A depth ``d`` is accepted when

    ``||V^{d+1} - V^{d}|| < absolute + relative * ||V^{d+1}||``

    on the geometry probe. Apparent (beam-attenuated) visibilities enter
    both norms, so a faint leaf does not refine solely because the
    fractional beam gradient is large.

    ``oracle_depth`` is the independent audit reference. It defaults to
    ``max_depth + 1`` so a parent that stopped at the planner cap is not
    compared against itself.
    """

    absolute: float = 1.0e-4
    relative: float = 1.0e-3
    min_depth: int = 0
    max_depth: int = 3
    oracle_depth: int | None = None
    forced_feature_depth: int = 2
    flux_floor_jy: float = 0.01
    null_contrast: float = 0.05
    gradient_contrast: float = 0.05
    sample_grid: int = 3

    def __post_init__(self) -> None:
        if self.absolute < 0.0 or self.relative < 0.0:
            raise ValueError("visibility tolerances must be non-negative")
        if self.min_depth < 0 or self.max_depth < self.min_depth:
            raise ValueError("max_depth must be at least min_depth")
        if self.oracle_depth is not None and self.oracle_depth <= self.max_depth:
            raise ValueError("oracle_depth must exceed max_depth")
        if self.forced_feature_depth < 0:
            raise ValueError("forced_feature_depth must be non-negative")
        if self.flux_floor_jy < 0.0:
            raise ValueError("flux_floor_jy must be non-negative")
        if self.null_contrast <= 0.0 or self.gradient_contrast <= 0.0:
            raise ValueError("contrast thresholds must be positive")
        if self.sample_grid < 2:
            raise ValueError("sample_grid must be at least 2")

    def threshold(self, apparent_norm: float) -> float:
        return float(self.absolute + self.relative * apparent_norm)

    @property
    def resolved_oracle_depth(self) -> int:
        if self.oracle_depth is None:
            return self.max_depth + 1
        return int(self.oracle_depth)


@dataclass(frozen=True)
class ComponentPointingAssignment:
    """One planned depth for one fitted parent and one pointing."""

    component_id: str
    pointing_id: str
    depth: int
    error_estimate: float
    apparent_norm: float
    reasons: tuple[str, ...]
    channel_regimes_hz: tuple[float, ...] = ()
    depth_by_regime: tuple[int, ...] = ()


@dataclass(frozen=True)
class IntegrationPlannerReport:
    """Pointing-aware integration depths. Built outside JIT."""

    assignments: tuple[ComponentPointingAssignment, ...]
    tolerance: IntegrationTolerance
    pointing_ids: tuple[str, ...]
    parent_id: tuple[str, ...]
    provenance: Mapping[str, Any]

    def depth_by_parent(self, pointing_id: str | None = None) -> dict[str, int]:
        """Return planned depths, or the conservative max across pointings."""

        selected = [
            item
            for item in self.assignments
            if pointing_id is None or item.pointing_id == pointing_id
        ]
        if pointing_id is not None and not selected:
            raise ValueError(f"planner report has no pointing {pointing_id!r}")
        depths: dict[str, int] = {}
        for item in selected:
            depths[item.component_id] = max(item.depth, depths.get(item.component_id, 0))
        return {parent: depths[parent] for parent in self.parent_id if parent in depths}


@dataclass(frozen=True)
class IntegrationAuditFinding:
    """One parent whose frozen depth was compared to a finer reference."""

    component_id: str
    pointing_id: str
    planned_depth: int
    audit_depth: int
    error: float
    threshold: float
    under_resolved: bool


@dataclass(frozen=True)
class IntegrationAuditReport:
    """Post-fit check that a frozen plan still meets the tolerance."""

    findings: tuple[IntegrationAuditFinding, ...]
    tolerance: IntegrationTolerance

    @property
    def under_resolved(self) -> tuple[IntegrationAuditFinding, ...]:
        return tuple(item for item in self.findings if item.under_resolved)


@dataclass(frozen=True)
class _ChannelRegime:
    frequency_hz: float
    probe: VisibilityBlock


@dataclass(frozen=True)
class _CachedPrediction:
    visibility: np.ndarray
    valid: np.ndarray
    off_diagonal_valid: np.ndarray
    parent_id: tuple[str, ...]


def plan_integration(
    table: SkyComponentTable,
    blocks: VisibilityBlock | Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str,
    tolerance: IntegrationTolerance | None = None,
    config: BeamOperatorConfig | None = None,
    pointing_ids: Sequence[str] | None = None,
    parent_policy: IntegrationParentPolicy | str = IntegrationParentPolicy.ALL_ACTIVE,
) -> IntegrationPlannerReport:
    """Assign integration depths from geometry, beam, and flux estimates.

    Measured ``block.visibility`` is discarded before any comparison. The
    returned depths are safe to freeze for one optimization run.
    """

    selected = IntegrationTolerance() if tolerance is None else tolerance
    state = require_beam_calibration_state(calibration_state)
    operator = config or BeamOperatorConfig()
    policy = IntegrationParentPolicy(parent_policy)
    positions = np.asarray(antenna_position_m, dtype=np.float64)
    probes = [
        select_planner_probe(block, antenna_position_m=positions)
        for block in _as_blocks(blocks)
    ]
    names = _pointing_names(probes, pointing_ids)
    parents = _selected_parents(table, policy)
    assignments: list[ComponentPointingAssignment] = []
    n_predictor_call = 0
    n_regime = 0
    for probe, pointing_id in zip(probes, names, strict=True):
        pointing_assignments, n_call, n_local_regime = _plan_pointing(
            table,
            parents,
            probe,
            pointing_id,
            beam,
            antenna_position_m=positions,
            calibration_state=state,
            config=operator,
            tolerance=selected,
        )
        assignments.extend(pointing_assignments)
        n_predictor_call += n_call
        n_regime = max(n_regime, n_local_regime)
    return IntegrationPlannerReport(
        assignments=tuple(assignments),
        tolerance=selected,
        pointing_ids=names,
        parent_id=tuple(component.component_id for component in parents),
        provenance=_planner_provenance(
            table,
            parents,
            probes,
            names,
            beam,
            state,
            policy,
            n_predictor_call=n_predictor_call,
            n_channel_regime=n_regime,
        ),
    )


def integration_plan_from_planner(
    table: SkyComponentTable,
    report: IntegrationPlannerReport,
    *,
    pointing_id: str | None = None,
    parent_policy: IntegrationParentPolicy | str = IntegrationParentPolicy.ALL_ACTIVE,
    pad: bool = False,
    capacity: int | None = None,
) -> IntegrationPlan:
    """Expand the table with the planner's depths for one or all pointings."""

    policy = IntegrationParentPolicy(parent_policy)
    _require_planner_report(table, report, parent_policy=policy)
    if pointing_id is not None and pointing_id not in report.pointing_ids:
        raise ValueError(f"planner report has no pointing {pointing_id!r}")
    return integration_plan_from_table(
        table,
        depth_by_parent=report.depth_by_parent(pointing_id),
        parent_policy=policy,
        pad=pad,
        capacity=capacity,
    )


def audit_integration_plan(
    table: SkyComponentTable,
    report: IntegrationPlannerReport,
    blocks: VisibilityBlock | Sequence[VisibilityBlock],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    audit_depth: int | None = None,
    parent_policy: IntegrationParentPolicy | str = IntegrationParentPolicy.ALL_ACTIVE,
) -> IntegrationAuditReport:
    """Compare a frozen plan to a finer independent oracle depth."""

    state = require_beam_calibration_state(calibration_state)
    operator = config or BeamOperatorConfig()
    policy = IntegrationParentPolicy(parent_policy)
    positions = np.asarray(antenna_position_m, dtype=np.float64)
    probes = [
        select_planner_probe(block, antenna_position_m=positions)
        for block in _as_blocks(blocks)
    ]
    names = _pointing_names(probes, report.pointing_ids)
    parents = _require_planner_report(
        table,
        report,
        parent_policy=policy,
        beam=beam,
        calibration_state=state,
        probes=probes,
    )
    oracle = report.tolerance.resolved_oracle_depth if audit_depth is None else int(audit_depth)
    if oracle < 0:
        raise ValueError("audit_depth must be non-negative")
    findings: list[IntegrationAuditFinding] = []
    by_key = {
        (item.component_id, item.pointing_id): item for item in report.assignments
    }
    for probe, pointing_id in zip(probes, names, strict=True):
        assigned_items: list[ComponentPointingAssignment] = []
        for component in parents:
            assigned = by_key.get((component.component_id, pointing_id))
            if assigned is None:
                raise ValueError(
                    "planner report is missing an assignment for "
                    f"{component.component_id!r} at {pointing_id!r}"
                )
            assigned_items.append(assigned)
        findings.extend(
            _audit_pointing(
                table,
                parents,
                assigned_items,
                probe,
                pointing_id,
                beam,
                antenna_position_m=positions,
                calibration_state=state,
                config=operator,
                tolerance=report.tolerance,
                oracle=oracle,
            )
        )
    return IntegrationAuditReport(findings=tuple(findings), tolerance=report.tolerance)


def successive_depth_errors(
    table: SkyComponentTable,
    component: SkyComponent,
    block: VisibilityBlock,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    max_depth: int = 3,
    flux_floor_jy: float = 0.01,
) -> tuple[float, ...]:
    """Return ``||V^{d+1}-V^{d}||`` for ``d = 0 … max_depth-1``."""

    probe = select_planner_probe(
        block, antenna_position_m=np.asarray(antenna_position_m, dtype=np.float64)
    )
    errors = []
    for depth in range(max_depth):
        error, _apparent = _depth_pair_error(
            table,
            component,
            probe,
            beam,
            depth,
            depth + 1,
            antenna_position_m=np.asarray(antenna_position_m, dtype=np.float64),
            calibration_state=require_beam_calibration_state(calibration_state),
            config=config or BeamOperatorConfig(),
            flux_floor_jy=flux_floor_jy,
        )
        errors.append(error)
    return tuple(errors)


def select_planner_probe(
    block: VisibilityBlock,
    *,
    antenna_position_m: ArrayLike | None = None,
    n_baseline: int = 3,
    n_channel: int = 3,
    n_time: int = 3,
    n_orientation: int = 2,
    n_w: int = 2,
) -> VisibilityBlock:
    """Return a stratified geometry subset with measured visibilities cleared."""

    rows = _stratified_rows(
        block,
        n_baseline=n_baseline,
        n_time=n_time,
        n_orientation=n_orientation,
        n_w=n_w,
        antenna_position_m=antenna_position_m,
    )
    channels = _stratified_channels(block.frequency_hz, n_channel)
    vis = block.visibility[np.ix_(rows, channels)]
    return VisibilityBlock(
        uvw_m=block.uvw_m[rows],
        frequency_hz=block.frequency_hz[channels],
        visibility=np.zeros_like(vis),
        weight=block.weight[np.ix_(rows, channels)],
        flag=block.flag[np.ix_(rows, channels)],
        time_s=block.time_s[rows],
        antenna1=block.antenna1[rows],
        antenna2=block.antenna2[rows],
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
        provenance={
            **dict(block.provenance),
            "planner_probe": True,
            "source_row_count": int(block.uvw_m.shape[0]),
            "source_channel_count": int(block.frequency_hz.size),
        },
    )


def stratified_integration_probe(
    *,
    phase_centre_rad: tuple[float, float],
    antenna_position_m: np.ndarray,
    correlations: tuple[Correlation, ...] = (
        Correlation.RR,
        Correlation.RL,
        Correlation.LR,
        Correlation.LL,
    ),
    receptor_basis: ReceptorBasis = ReceptorBasis.CIRCULAR,
) -> VisibilityBlock:
    """Build a synthetic probe covering baseline, channel, PA, UV, and w bins."""

    del antenna_position_m
    uvw = np.array(
        [
            [20.0, 5.0, 2.0],
            [5.0, 20.0, -12.0],
            [200.0, -40.0, 8.0],
            [-40.0, 200.0, -25.0],
            [2000.0, 300.0, 40.0],
            [300.0, -2000.0, -40.0],
        ],
        dtype=np.float64,
    )
    times = np.array([5.0e9, 5.0e9 + 10_000.0, 5.0e9 + 20_000.0], dtype=np.float64)
    antenna1 = np.array([0, 0, 1, 0, 0, 1], dtype=np.int32)
    antenna2 = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
    stacked_uvw = np.tile(uvw, (times.size, 1))
    stacked_time = np.repeat(times, uvw.shape[0])
    stacked_ant1 = np.tile(antenna1, times.size)
    stacked_ant2 = np.tile(antenna2, times.size)
    frequency = np.array([4.536e9, 4.599e9, 4.662e9])
    dummy = np.zeros(
        (stacked_uvw.shape[0], frequency.size, len(correlations)),
        dtype=np.complex128,
    )
    return VisibilityBlock(
        uvw_m=stacked_uvw,
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=stacked_time,
        antenna1=stacked_ant1,
        antenna2=stacked_ant2,
        correlations=correlations,
        receptor_basis=receptor_basis,
        phase_centre_rad=phase_centre_rad,
        provenance={"planner_probe": True, "synthetic": True},
    )


def meets_visibility_tolerance(
    coarse: BeamOperatorResult,
    fine: BeamOperatorResult,
    block: VisibilityBlock,
    tolerance: IntegrationTolerance,
) -> tuple[bool, float, float]:
    """Return ``(accepted, error, apparent_norm)`` for one depth pair."""

    error, apparent = _visibility_error(coarse, fine, block)
    return error < tolerance.threshold(apparent) + 1.0e-15, error, apparent


def _plan_pointing(
    table: SkyComponentTable,
    parents: Sequence[SkyComponent],
    probe: VisibilityBlock,
    pointing_id: str,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    tolerance: IntegrationTolerance,
) -> tuple[list[ComponentPointingAssignment], int, int]:
    regimes = _channel_regimes(beam, probe)
    assignments: list[ComponentPointingAssignment] = []
    squares: list[SkyComponent] = []
    for component in parents:
        if _is_delta(component):
            assignments.append(
                ComponentPointingAssignment(
                    component_id=component.component_id,
                    pointing_id=pointing_id,
                    depth=0,
                    error_estimate=0.0,
                    apparent_norm=0.0,
                    reasons=(RefinementReason.DELTA.value,),
                    channel_regimes_hz=tuple(regime.frequency_hz for regime in regimes),
                    depth_by_regime=tuple(0 for _ in regimes),
                )
            )
        else:
            squares.append(component)
    if not squares:
        return assignments, 0, len(regimes)

    forced = {
        component.component_id: _forced_reasons(
            table,
            component,
            probe,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            tolerance=tolerance,
        )
        for component in squares
    }
    n_predictor_call = 0
    regime_depths: list[dict[str, int]] = []
    regime_errors: list[dict[str, tuple[float, float]]] = []
    regime_extra: list[dict[str, tuple[str, ...]]] = []
    for regime in regimes:
        depths, errors, extras, n_call = _batched_spatial_search(
            table,
            squares,
            regime.probe,
            forced,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            tolerance=tolerance,
        )
        regime_depths.append(depths)
        regime_errors.append(errors)
        regime_extra.append(extras)
        n_predictor_call += n_call
    frequency_node = len(regimes) > 1
    for component in squares:
        per_regime = tuple(item[component.component_id] for item in regime_depths)
        depth = max(per_regime)
        error, apparent = _select_regime_error(
            component.component_id, depth, regime_depths, regime_errors
        )
        reasons = [reason.value for reason in forced[component.component_id]]
        if frequency_node:
            reasons.append(RefinementReason.FREQUENCY_NODE.value)
        for extra in regime_extra:
            reasons.extend(extra[component.component_id])
        unique_reasons = _unique_strings(reasons)
        assignments.append(
            ComponentPointingAssignment(
                component_id=component.component_id,
                pointing_id=pointing_id,
                depth=depth,
                error_estimate=error,
                apparent_norm=apparent,
                reasons=unique_reasons,
                channel_regimes_hz=tuple(regime.frequency_hz for regime in regimes),
                depth_by_regime=per_regime,
            )
        )
    return assignments, n_predictor_call, len(regimes)


def _batched_spatial_search(
    table: SkyComponentTable,
    components: Sequence[SkyComponent],
    probe: VisibilityBlock,
    forced: Mapping[str, tuple[RefinementReason, ...]],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    tolerance: IntegrationTolerance,
) -> tuple[
    dict[str, int],
    dict[str, tuple[float, float]],
    dict[str, tuple[str, ...]],
    int,
]:
    starts: dict[str, int] = {}
    depths: dict[str, int] = {}
    errors: dict[str, tuple[float, float]] = {}
    extras: dict[str, list[str]] = {}
    searchable: list[SkyComponent] = []
    for component in components:
        start = tolerance.min_depth
        if forced[component.component_id]:
            start = max(start, min(tolerance.forced_feature_depth, tolerance.max_depth))
        starts[component.component_id] = start
        extras[component.component_id] = []
        if start >= tolerance.max_depth:
            depths[component.component_id] = tolerance.max_depth
            errors[component.component_id] = (0.0, 0.0)
            extras[component.component_id].append(RefinementReason.MAX_DEPTH.value)
        else:
            searchable.append(component)
    if not searchable:
        return depths, errors, {key: tuple(value) for key, value in extras.items()}, 0

    cache: dict[int, _CachedPrediction] = {}
    n_call = 0

    def predict_depth(depth: int) -> _CachedPrediction:
        nonlocal n_call
        cached = cache.get(depth)
        if cached is None:
            cache[depth] = _predict_parents(
                table,
                searchable,
                probe,
                beam,
                depth,
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                config=config,
                flux_floor_jy=tolerance.flux_floor_jy,
            )
            n_call += 1
            cached = cache[depth]
        return cached

    current = {
        component.component_id: starts[component.component_id] for component in searchable
    }
    pending = list(searchable)
    while pending:
        depth = min(current[component.component_id] for component in pending)
        if depth >= tolerance.max_depth:
            for component in pending:
                extras[component.component_id].append(RefinementReason.MAX_DEPTH.value)
                depths[component.component_id] = tolerance.max_depth
                errors.setdefault(component.component_id, (0.0, 0.0))
            break
        coarse = predict_depth(depth)
        fine = predict_depth(depth + 1)
        index = {parent: i for i, parent in enumerate(coarse.parent_id)}
        still: list[SkyComponent] = []
        for component in pending:
            component_id = component.component_id
            if current[component_id] != depth:
                still.append(component)
                continue
            error, apparent = _cached_parent_error(
                coarse, fine, index[component_id], probe
            )
            errors[component_id] = (error, apparent)
            if error < tolerance.threshold(apparent) + 1.0e-15:
                extras[component_id].append(RefinementReason.CONVERGED.value)
                depths[component_id] = depth
                continue
            extras[component_id].append(RefinementReason.VISIBILITY_ERROR.value)
            current[component_id] = depth + 1
            if current[component_id] >= tolerance.max_depth:
                extras[component_id].append(RefinementReason.MAX_DEPTH.value)
                depths[component_id] = tolerance.max_depth
            else:
                still.append(component)
        pending = still
    return depths, errors, {key: tuple(value) for key, value in extras.items()}, n_call


def _audit_pointing(
    table: SkyComponentTable,
    parents: Sequence[SkyComponent],
    assigned_items: Sequence[ComponentPointingAssignment],
    probe: VisibilityBlock,
    pointing_id: str,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    tolerance: IntegrationTolerance,
    oracle: int,
) -> list[IntegrationAuditFinding]:
    findings: list[IntegrationAuditFinding] = []
    squares: list[tuple[SkyComponent, ComponentPointingAssignment]] = []
    for component, assigned in zip(parents, assigned_items, strict=True):
        if _is_delta(component):
            findings.append(
                IntegrationAuditFinding(
                    component_id=component.component_id,
                    pointing_id=pointing_id,
                    planned_depth=assigned.depth,
                    audit_depth=max(assigned.depth, oracle),
                    error=0.0,
                    threshold=tolerance.threshold(0.0),
                    under_resolved=False,
                )
            )
        else:
            squares.append((component, assigned))
    if not squares:
        return findings

    depths_needed = {item.depth for _component, item in squares}
    comparable = [item.depth for _component, item in squares if item.depth < oracle]
    if comparable:
        depths_needed.add(oracle)
    cache: dict[int, _CachedPrediction] = {}
    for depth in sorted(depths_needed):
        cache[depth] = _predict_parents(
            table,
            [component for component, _assigned in squares],
            probe,
            beam,
            depth,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            flux_floor_jy=tolerance.flux_floor_jy,
        )
    parent_order = next(iter(cache.values())).parent_id
    index = {parent: i for i, parent in enumerate(parent_order)}
    for component, assigned in squares:
        unresolved_cap = _unresolved_max_depth(assigned.reasons)
        if assigned.depth >= oracle:
            error, apparent = 0.0, 0.0
            under = unresolved_cap
        else:
            error, apparent = _cached_parent_error(
                cache[assigned.depth],
                cache[oracle],
                index[component.component_id],
                probe,
            )
            under = unresolved_cap or error > tolerance.threshold(apparent) + 1.0e-15
        findings.append(
            IntegrationAuditFinding(
                component_id=component.component_id,
                pointing_id=pointing_id,
                planned_depth=assigned.depth,
                audit_depth=max(assigned.depth, oracle),
                error=error,
                threshold=tolerance.threshold(apparent),
                under_resolved=under,
            )
        )
    return findings


def _predict_parents(
    table: SkyComponentTable,
    components: Sequence[SkyComponent],
    probe: VisibilityBlock,
    beam: VoltageBeamModel,
    depth: int,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    flux_floor_jy: float,
) -> _CachedPrediction:
    planning = _planning_table(table, components, flux_floor_jy)
    fluxes = np.array(
        [
            max(abs(float(component.stokes_i_jy)), float(flux_floor_jy))
            for component in components
        ],
        dtype=np.float64,
    )
    result = predict_voltage_from_plan(
        probe,
        integration_plan_from_table(
            planning,
            depth_by_parent={
                component.component_id: int(depth) for component in components
            },
        ),
        fluxes,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        split_parents=True,
    )
    if result.parent_visibility is None:
        raise RuntimeError("batched planner requires parent_visibility")
    leakage = (
        result.valid
        if result.off_diagonal_valid is None
        else np.asarray(result.off_diagonal_valid, dtype=bool)
    )
    return _CachedPrediction(
        visibility=np.asarray(result.parent_visibility),
        valid=np.asarray(result.valid, dtype=bool),
        off_diagonal_valid=np.asarray(leakage, dtype=bool),
        parent_id=tuple(component.component_id for component in components),
    )


def _depth_pair_error(
    table: SkyComponentTable,
    component: SkyComponent,
    probe: VisibilityBlock,
    beam: VoltageBeamModel,
    coarse_depth: int,
    fine_depth: int,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    flux_floor_jy: float,
) -> tuple[float, float]:
    if fine_depth < coarse_depth:
        raise ValueError("fine_depth must be at least coarse_depth")
    if fine_depth == coarse_depth:
        raise ValueError("fine_depth must exceed coarse_depth")
    coarse = _predict_parents(
        table,
        (component,),
        probe,
        beam,
        coarse_depth,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        flux_floor_jy=flux_floor_jy,
    )
    fine = _predict_parents(
        table,
        (component,),
        probe,
        beam,
        fine_depth,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        flux_floor_jy=flux_floor_jy,
    )
    return _cached_parent_error(coarse, fine, 0, probe)


def _visibility_error(
    coarse: BeamOperatorResult,
    fine: BeamOperatorResult,
    block: VisibilityBlock,
) -> tuple[float, float]:
    mask = _sample_mask(coarse, block) & _sample_mask(fine, block)
    return _weighted_visibility_error(
        np.asarray(coarse.visibility),
        np.asarray(fine.visibility),
        mask,
        np.asarray(block.weight, dtype=np.float64),
    )


def _cached_parent_error(
    coarse: _CachedPrediction,
    fine: _CachedPrediction,
    parent: int,
    block: VisibilityBlock,
) -> tuple[float, float]:
    mask = _array_sample_mask(
        coarse.valid, coarse.off_diagonal_valid, block
    ) & _array_sample_mask(fine.valid, fine.off_diagonal_valid, block)
    return _weighted_visibility_error(
        coarse.visibility[parent],
        fine.visibility[parent],
        mask,
        np.asarray(block.weight, dtype=np.float64),
    )


def _weighted_visibility_error(
    coarse_vis: np.ndarray,
    fine_vis: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    if mask.shape != weight.shape:
        raise ValueError("planner weight mask must match the probe")
    weighted = weight * mask.astype(np.float64)
    error = float(np.sqrt(np.sum(weighted * np.square(np.abs(fine_vis - coarse_vis)))))
    apparent = float(np.sqrt(np.sum(weighted * np.square(np.abs(fine_vis)))))
    return error, apparent


def _sample_mask(result: BeamOperatorResult, block: VisibilityBlock) -> NDArray[np.bool_]:
    leakage = (
        result.valid
        if result.off_diagonal_valid is None
        else np.asarray(result.off_diagonal_valid, dtype=bool)
    )
    return _array_sample_mask(
        np.asarray(result.valid, dtype=bool),
        np.asarray(leakage, dtype=bool),
        block,
    )


def _array_sample_mask(
    copolar: np.ndarray,
    leakage: np.ndarray,
    block: VisibilityBlock,
) -> NDArray[np.bool_]:
    planes = []
    for correlation in block.correlations:
        planes.append(leakage if correlation in _CROSS_HANDS else copolar)
    return np.asarray(
        np.stack(planes, axis=-1) & ~np.asarray(block.flag, dtype=bool),
        dtype=bool,
    )


def _forced_reasons(
    table: SkyComponentTable,
    component: SkyComponent,
    probe: VisibilityBlock,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    tolerance: IntegrationTolerance,
) -> tuple[RefinementReason, ...]:
    local_l, local_m = _parent_sample_directions(
        component,
        table.mosaic_phase_centre_rad,
        probe.phase_centre_rad,
        tolerance.sample_grid,
    )
    reasons: list[RefinementReason] = []
    valid_parts: list[np.ndarray] = []
    off_parts: list[np.ndarray] = []
    jones_parts: list[np.ndarray] = []
    for time_s in np.unique(probe.time_s):
        evaluation = _evaluate_parent_beam_at_time(
            beam,
            local_l,
            local_m,
            probe,
            float(time_s),
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            pointing_offset_lm_rad=config.pointing_offset_lm_rad,
        )
        valid_parts.append(np.asarray(evaluation.valid, dtype=bool))
        off_parts.append(
            np.asarray(evaluation.valid, dtype=bool)
            if evaluation.off_diagonal_valid is None
            else np.asarray(evaluation.off_diagonal_valid, dtype=bool)
        )
        jones_parts.append(np.asarray(evaluation.jones))
    valid = np.concatenate([part.reshape(-1) for part in valid_parts])
    off_valid = np.concatenate([part.reshape(-1) for part in off_parts])
    jones = np.concatenate([part.reshape(-1, 2, 2) for part in jones_parts])
    reasons.extend(_jones_feature_reasons(valid, off_valid, jones, tolerance))
    reasons.extend(
        _model_specific_reasons(
            beam,
            local_l,
            local_m,
            probe,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            pointing_offset_lm_rad=config.pointing_offset_lm_rad,
        )
    )
    return _unique_reasons(reasons)


def _jones_feature_reasons(
    valid: np.ndarray,
    off_valid: np.ndarray,
    jones: np.ndarray,
    tolerance: IntegrationTolerance,
) -> tuple[RefinementReason, ...]:
    reasons: list[RefinementReason] = []
    if np.any(valid) and np.any(~valid):
        reasons.append(RefinementReason.VALIDITY_BOUNDARY)
    if np.any(off_valid) and np.any(~off_valid):
        if RefinementReason.VALIDITY_BOUNDARY not in reasons:
            reasons.append(RefinementReason.VALIDITY_BOUNDARY)
    if np.any(valid != off_valid):
        reasons.append(RefinementReason.RASTER_BOUNDARY)
    usable = valid
    if np.any(usable):
        j00 = jones[..., 0, 0][usable]
        j11 = jones[..., 1, 1][usable]
        j01 = jones[..., 0, 1][usable]
        scale = float(np.mean(np.abs(j00))) + 1.0e-15
        if np.any(np.real(j00) > 0.0) and np.any(np.real(j00) < 0.0):
            reasons.append(RefinementReason.PHASE_REVERSAL)
        if np.max(np.abs(j00)) > 0.0 and (
            np.min(np.abs(j00)) / (np.max(np.abs(j00)) + 1.0e-15)
            < tolerance.null_contrast
        ):
            reasons.append(RefinementReason.BEAM_NULL)
        if float(np.std(np.abs(j00) - np.abs(j11))) / scale > tolerance.gradient_contrast:
            reasons.append(RefinementReason.SQUINT_GRADIENT)
        if float(np.std(np.abs(j01))) / scale > tolerance.gradient_contrast:
            reasons.append(RefinementReason.LEAKAGE_GRADIENT)
    return tuple(reasons)


def _model_specific_reasons(
    beam: VoltageBeamModel,
    local_l: np.ndarray,
    local_m: np.ndarray,
    probe: VisibilityBlock,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    pointing_offset_lm_rad: tuple[float, float] | None,
) -> tuple[RefinementReason, ...]:
    reasons: list[RefinementReason] = []
    if isinstance(beam, CompositeScalarVoltageBeam):
        for time_s in np.unique(probe.time_s):
            main = _evaluate_parent_beam_at_time(
                beam.main,
                local_l,
                local_m,
                probe,
                float(time_s),
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                pointing_offset_lm_rad=pointing_offset_lm_rad,
            )
            if np.any(main.valid) and np.any(~main.valid):
                reasons.append(RefinementReason.OUTER_HANDOVER)
                break
    if isinstance(beam, CassbeamCBandVoltageBeam) and beam.outer is not None:
        inner_ok = _cassbeam_inner_support(beam, local_l, local_m)
        if np.any(inner_ok) and np.any(~inner_ok):
            reasons.append(RefinementReason.RASTER_BOUNDARY)
    return tuple(reasons)


def _channel_regimes(
    beam: VoltageBeamModel, probe: VisibilityBlock
) -> tuple[_ChannelRegime, ...]:
    if not isinstance(beam, CassbeamCBandVoltageBeam):
        return (_ChannelRegime(float(np.mean(probe.frequency_hz)), probe),)
    tables = beam.artifact.tables
    groups: dict[float, list[int]] = {}
    order: list[float] = []
    for index, frequency in enumerate(probe.frequency_hz):
        nearest = min(
            tables, key=lambda table: abs(table.frequency_hz - float(frequency))
        )
        key = float(nearest.frequency_hz)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(int(index))
    if len(order) == 1:
        return (_ChannelRegime(order[0], probe),)
    return tuple(
        _ChannelRegime(key, _subset_channels(probe, np.asarray(groups[key], dtype=np.intp)))
        for key in order
    )


def _subset_channels(block: VisibilityBlock, channels: np.ndarray) -> VisibilityBlock:
    return VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz[channels],
        visibility=block.visibility[:, channels],
        weight=block.weight[:, channels],
        flag=block.flag[:, channels],
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
        provenance={**dict(block.provenance), "channel_regime": True},
    )


def _cassbeam_inner_support(
    beam: CassbeamCBandVoltageBeam,
    local_l: np.ndarray,
    local_m: np.ndarray,
) -> NDArray[np.bool_]:
    table = beam.artifact.tables[0]
    l_min = float(np.min(table.l_rad))
    l_max = float(np.max(table.l_rad))
    m_min = float(np.min(table.m_rad))
    m_max = float(np.max(table.m_rad))
    return (local_l >= l_min) & (local_l <= l_max) & (local_m >= m_min) & (
        local_m <= m_max
    )


def _evaluate_parent_beam_at_time(
    beam: Any,
    local_l: np.ndarray,
    local_m: np.ndarray,
    probe: VisibilityBlock,
    time_s: float,
    *,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    pointing_offset_lm_rad: tuple[float, float] | None,
) -> BeamEvaluation:
    antennas = np.unique(np.concatenate((probe.antenna1, probe.antenna2)))
    chi = parallactic_angle_rad(
        np.asarray([time_s], dtype=np.float64),
        probe.phase_centre_rad,
        antenna_position_m,
    )[0]
    required = int(np.max(antennas)) + 1
    if chi.size < required:
        raise ValueError("antenna_position_m must cover every antenna in the probe")
    evaluation = beam.evaluate(
        beam_coordinates(
            local_l,
            local_m,
            probe.frequency_hz,
            parallactic_angle_rad=chi[antennas],
            pointing_offset_lm_rad=pointing_offset_lm_rad,
            antenna_id=antennas,
        ),
        calibration_state=calibration_state,
    )
    if not isinstance(evaluation, BeamEvaluation):
        raise TypeError("beam.evaluate must return a BeamEvaluation")
    return evaluation


def _parent_sample_directions(
    component: SkyComponent,
    mosaic_phase_centre_rad: tuple[float, float],
    block_phase_centre_rad: tuple[float, float],
    sample_grid: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if component.width_rad <= 0.0:
        offsets = np.array([[0.0, 0.0]])
    else:
        axis = np.linspace(-0.5, 0.5, sample_grid) * component.width_rad
        d_l, d_m = np.meshgrid(axis, axis, indexing="xy")
        offsets = np.column_stack((d_l.reshape(-1), d_m.reshape(-1)))
    mosaic_l = component.l_rad + offsets[:, 0]
    mosaic_m = component.m_rad + offsets[:, 1]
    mosaic = np.asarray(mosaic_phase_centre_rad, dtype=np.float64)
    block = np.asarray(block_phase_centre_rad, dtype=np.float64)
    if np.allclose(mosaic, block, rtol=0.0, atol=1.0e-15):
        return (
            np.asarray(mosaic_l, dtype=np.float64),
            np.asarray(mosaic_m, dtype=np.float64),
        )
    sky_ra, sky_dec = lmn_to_radec(mosaic[0], mosaic[1], mosaic_l, mosaic_m)
    local_l, local_m, _n = radec_to_lmn(block[0], block[1], sky_ra, sky_dec)
    return (
        np.asarray(local_l, dtype=np.float64),
        np.asarray(local_m, dtype=np.float64),
    )


def _planning_table(
    table: SkyComponentTable,
    components: Sequence[SkyComponent],
    flux_floor_jy: float,
) -> SkyComponentTable:
    subset = SkyComponentTable(
        components=tuple(components),
        mosaic_phase_centre_rad=table.mosaic_phase_centre_rad,
        report=table.report,
        source=table.source,
    )
    records = sky_table_to_records(subset)
    for record, component in zip(records, components, strict=True):
        record["stokes_i_jy"] = max(abs(float(component.stokes_i_jy)), float(flux_floor_jy))
    return sky_table_from_records(
        records, mosaic_phase_centre_rad=table.mosaic_phase_centre_rad
    )


def _selected_parents(
    table: SkyComponentTable,
    policy: IntegrationParentPolicy,
) -> tuple[SkyComponent, ...]:
    keep_nonpositive = policy is IntegrationParentPolicy.ALL_ACTIVE
    selected = tuple(
        component
        for component in table.components
        if component.active and (keep_nonpositive or component.stokes_i_jy > 0.0)
    )
    if not selected:
        raise ValueError("sky table has no selected parents")
    return selected


def _require_planner_report(
    table: SkyComponentTable,
    report: IntegrationPlannerReport,
    *,
    parent_policy: IntegrationParentPolicy,
    beam: VoltageBeamModel | None = None,
    calibration_state: BeamCalibrationState | None = None,
    probes: Sequence[VisibilityBlock] | None = None,
) -> tuple[SkyComponent, ...]:
    parents = _selected_parents(table, parent_policy)
    parent_ids = tuple(component.component_id for component in parents)
    required = (
        "parent_id",
        "parent_policy",
        "mosaic_phase_centre_rad",
        "beam_model_id",
        "calibration_state",
        "pointing_ids",
        "pointing_phase_centres_rad",
    )
    missing = [key for key in required if key not in report.provenance]
    if missing:
        raise ValueError(
            "planner report is missing provenance "
            f"{missing!r}; re-run plan_integration"
        )
    if tuple(report.parent_id) != parent_ids or tuple(report.provenance["parent_id"]) != parent_ids:
        raise ValueError(
            "planner report does not match the selected sky parents; "
            "re-run plan_integration after the table or parent policy changes"
        )
    if report.provenance["parent_policy"] != parent_policy.value:
        raise ValueError("planner report parent_policy does not match the consumer")
    if not np.allclose(
        report.provenance["mosaic_phase_centre_rad"],
        table.mosaic_phase_centre_rad,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("planner report mosaic phase centre does not match the table")
    planned = report.depth_by_parent()
    missing_depths = [parent for parent in parent_ids if parent not in planned]
    if missing_depths:
        raise ValueError(
            f"planner report is missing depths for {missing_depths!r}"
        )
    if beam is not None and report.provenance["beam_model_id"] != _beam_model_id(beam):
        raise ValueError("planner report beam_model_id does not match the consumer")
    if (
        calibration_state is not None
        and report.provenance["calibration_state"] != calibration_state.value
    ):
        raise ValueError("planner report calibration_state does not match the consumer")
    if probes is not None:
        stored = tuple(
            tuple(float(value) for value in centre)
            for centre in report.provenance["pointing_phase_centres_rad"]
        )
        actual = tuple(
            (float(probe.phase_centre_rad[0]), float(probe.phase_centre_rad[1]))
            for probe in probes
        )
        if len(stored) != len(actual) or not all(
            np.allclose(left, right, rtol=0.0, atol=1.0e-15)
            for left, right in zip(stored, actual, strict=True)
        ):
            raise ValueError("planner report pointing centres do not match the blocks")
    return parents


def _planner_provenance(
    table: SkyComponentTable,
    parents: Sequence[SkyComponent],
    probes: Sequence[VisibilityBlock],
    pointing_ids: Sequence[str],
    beam: VoltageBeamModel,
    calibration_state: BeamCalibrationState,
    policy: IntegrationParentPolicy,
    *,
    n_predictor_call: int,
    n_channel_regime: int,
) -> dict[str, Any]:
    return {
        "planner": "integration_planner",
        "parent_id": tuple(component.component_id for component in parents),
        "parent_policy": policy.value,
        "mosaic_phase_centre_rad": (
            float(table.mosaic_phase_centre_rad[0]),
            float(table.mosaic_phase_centre_rad[1]),
        ),
        "beam_model_id": _beam_model_id(beam),
        "calibration_state": calibration_state.value,
        "pointing_ids": tuple(pointing_ids),
        "pointing_phase_centres_rad": tuple(
            (float(probe.phase_centre_rad[0]), float(probe.phase_centre_rad[1]))
            for probe in probes
        ),
        "n_pointing": len(pointing_ids),
        "n_parent": len(parents),
        "n_predictor_call": int(n_predictor_call),
        "n_channel_regime": int(n_channel_regime),
        "batched_parent_visibilities": True,
        "used_measured_visibility": False,
    }


def _beam_model_id(beam: VoltageBeamModel) -> str:
    model_id = getattr(beam, "model_id", None)
    if model_id is not None:
        return str(model_id)
    return type(beam).__name__


def _as_blocks(
    blocks: VisibilityBlock | Sequence[VisibilityBlock],
) -> tuple[VisibilityBlock, ...]:
    if isinstance(blocks, VisibilityBlock):
        return (blocks,)
    packed = tuple(blocks)
    if not packed:
        raise ValueError("planner requires at least one visibility block")
    return packed


def _pointing_names(
    blocks: Sequence[VisibilityBlock],
    pointing_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if pointing_ids is not None:
        resolved = tuple(str(item) for item in pointing_ids)
        if len(resolved) != len(blocks):
            raise ValueError("pointing_ids must match the number of blocks")
        if len(set(resolved)) != len(resolved):
            raise ValueError("pointing_ids must be unique")
        return resolved
    names: list[str] = []
    for index, block in enumerate(blocks):
        labelled = block.provenance.get("pointing_id")
        if labelled is None:
            names.append(f"pointing_{index}")
        else:
            names.append(str(labelled))
    if len(set(names)) != len(names):
        raise ValueError("block pointing_id provenance values must be unique")
    return tuple(names)


def _stratified_rows(
    block: VisibilityBlock,
    *,
    n_baseline: int,
    n_time: int,
    n_orientation: int,
    n_w: int,
    antenna_position_m: ArrayLike | None,
) -> NDArray[np.intp]:
    n_row = int(block.uvw_m.shape[0])
    budget = max(n_baseline, 1) * max(n_time, 1) * max(n_orientation, 1) * max(n_w, 1)
    if n_row <= budget:
        return np.arange(n_row, dtype=np.intp)
    times, inverse = np.unique(block.time_s, return_inverse=True)
    time_picks = _parallactic_time_picks(
        block, times, n_time, antenna_position_m
    )
    selected: list[int] = []
    for time_index in time_picks:
        members = np.flatnonzero(inverse == time_index)
        selected.extend(
            _pick_geometry_members(
                members,
                block.uvw_m[members],
                n_baseline=n_baseline,
                n_orientation=n_orientation,
                n_w=n_w,
            )
        )
    return np.asarray(sorted(set(selected)), dtype=np.intp)


def _parallactic_time_picks(
    block: VisibilityBlock,
    times: np.ndarray,
    n_time: int,
    antenna_position_m: ArrayLike | None,
) -> NDArray[np.intp]:
    if times.size <= n_time:
        return np.arange(times.size, dtype=np.intp)
    if antenna_position_m is None:
        return np.unique(
            np.linspace(0, times.size - 1, num=n_time, dtype=int)
        ).astype(np.intp)
    chi = parallactic_angle_rad(
        times,
        block.phase_centre_rad,
        np.asarray(antenna_position_m, dtype=np.float64),
    )
    pa = np.mean(np.asarray(chi, dtype=np.float64), axis=1)
    selected: list[int] = []
    for quantile in np.linspace(0.0, 1.0, num=n_time):
        target = float(np.quantile(pa, quantile))
        chosen = int(np.argmin(np.abs(pa - target)))
        if chosen not in selected:
            selected.append(chosen)
    return np.asarray(selected, dtype=np.intp)


def _pick_geometry_members(
    members: np.ndarray,
    uvw: np.ndarray,
    *,
    n_baseline: int,
    n_orientation: int,
    n_w: int,
) -> list[int]:
    if members.size == 0:
        return []
    length = np.hypot(uvw[:, 0], uvw[:, 1])
    angle = np.arctan2(uvw[:, 1], uvw[:, 0])
    w_m = uvw[:, 2]
    n_len = min(max(n_baseline, 1), members.size)
    n_ang = min(max(n_orientation, 1), members.size)
    n_w_bins = min(max(n_w, 1), members.size)
    scale = np.array(
        [
            float(np.ptp(length)) + 1.0e-15,
            float(np.ptp(angle)) + 1.0e-15,
            float(np.ptp(w_m)) + 1.0e-15,
        ]
    )
    features = np.column_stack((length, angle, w_m))
    selected: list[int] = []
    for length_q in np.linspace(0.0, 1.0, num=n_len):
        for angle_q in np.linspace(0.0, 1.0, num=n_ang):
            for w_q in np.linspace(0.0, 1.0, num=n_w_bins):
                target = np.array(
                    [
                        float(np.quantile(length, length_q)),
                        float(np.quantile(angle, angle_q)),
                        float(np.quantile(w_m, w_q)),
                    ]
                )
                dist = np.linalg.norm((features - target) / scale, axis=1)
                chosen = int(members[int(np.argmin(dist))])
                if chosen not in selected:
                    selected.append(chosen)
    return selected


def _stratified_channels(frequency_hz: np.ndarray, n_channel: int) -> NDArray[np.intp]:
    count = int(frequency_hz.size)
    if count <= n_channel:
        return np.arange(count, dtype=np.intp)
    return np.unique(np.linspace(0, count - 1, num=n_channel, dtype=int)).astype(np.intp)


def _is_delta(component: SkyComponent) -> bool:
    return component.basis_type is SkyBasisType.DELTA or component.width_rad == 0.0


def _unresolved_max_depth(reasons: Sequence[str]) -> bool:
    return (
        RefinementReason.MAX_DEPTH.value in reasons
        and RefinementReason.CONVERGED.value not in reasons
    )


def _select_regime_error(
    component_id: str,
    depth: int,
    regime_depths: Sequence[Mapping[str, int]],
    regime_errors: Sequence[Mapping[str, tuple[float, float]]],
) -> tuple[float, float]:
    chosen = (0.0, 0.0)
    for depths, errors in zip(regime_depths, regime_errors, strict=True):
        if depths[component_id] == depth:
            chosen = errors[component_id]
    return chosen


def _unique_reasons(reasons: Sequence[RefinementReason]) -> tuple[RefinementReason, ...]:
    unique: list[RefinementReason] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return tuple(unique)


def _unique_strings(reasons: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    return tuple(unique)
