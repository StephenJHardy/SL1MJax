from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import sky_table_from_records, sky_table_to_records
from sl1mjax.beam_conventions import (
    PerleyFrequencyPolicy,
    require_beam_calibration_state,
    select_perley2016_cband_window,
)
from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.cassbeam_beam import (
    CassbeamCBandVoltageBeam,
    load_cassbeam_cband_artifact,
    voltage_beam_for_mode,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    ManufacturedVoltageBeam,
    integration_plan_from_table,
    predict_voltage_from_plan,
)
from sl1mjax.integration_planner import (
    IntegrationTolerance,
    RefinementReason,
    audit_integration_plan,
    integration_plan_from_planner,
    plan_integration,
    select_planner_probe,
    stratified_integration_probe,
    successive_depth_errors,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam, BeamCoordinates, BeamEvaluation

_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)
_IDENTITY = np.eye(2, dtype=np.complex128)
_NONTRIVIAL = np.array(
    [[1.1 + 0.0j, 0.2 - 0.1j], [0.15 + 0.05j, 0.9 + 0.0j]],
    dtype=np.complex128,
)
_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))
_WIDTHS_ARCSEC = (4.0, 8.0, 16.0, 60.0)


def _square_table(
    l_rad: float,
    m_rad: float,
    width_rad: float,
    flux: float,
    *,
    component_id: str = "central_tree:central:0:0:0",
):
    return sky_table_from_records(
        [
            {
                "component_id": component_id,
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": l_rad,
                "m_rad": m_rad,
                "stokes_i_jy": flux,
                "width_rad": width_rad,
                "level": 0,
                "iy": 0,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            }
        ],
        mosaic_phase_centre_rad=_PHASE,
    )


def _assignment(report, component_id: str, pointing_id: str = "pointing_0"):
    for item in report.assignments:
        if item.component_id == component_id and item.pointing_id == pointing_id:
            return item
    raise AssertionError(f"missing assignment {component_id} {pointing_id}")


def test_phase4_probe_clears_measured_visibilities_and_stratifies() -> None:
    n_row = 12
    n_channel = 7
    vis = (np.arange(n_row * n_channel * 2, dtype=np.float64) + 1j).reshape(
        n_row, n_channel, 2
    )
    block = VisibilityBlock(
        uvw_m=np.column_stack(
            (
                np.linspace(10.0, 3000.0, n_row),
                np.linspace(-5.0, 400.0, n_row),
                np.zeros(n_row),
            )
        ),
        frequency_hz=np.linspace(4.536e9, 4.662e9, n_channel),
        visibility=vis,
        weight=np.ones_like(vis, dtype=np.float64),
        flag=np.zeros(vis.shape, dtype=bool),
        time_s=np.array([5.0e9] * 6 + [5.0e9 + 10_000.0] * 6),
        antenna1=np.zeros(n_row, dtype=np.int32),
        antenna2=np.ones(n_row, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    probe = select_planner_probe(block)
    assert np.all(probe.visibility == 0.0)
    assert probe.frequency_hz.size == 3
    assert probe.frequency_hz[0] == block.frequency_hz[0]
    assert probe.frequency_hz[-1] == block.frequency_hz[-1]
    baselines = np.hypot(probe.uvw_m[:, 0], probe.uvw_m[:, 1])
    assert baselines.min() < 200.0
    assert baselines.max() > 1000.0
    assert np.unique(probe.time_s).size == 2


def test_phase4_synthetic_probe_covers_required_regimes() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    baselines = np.hypot(probe.uvw_m[:, 0], probe.uvw_m[:, 1])
    assert baselines.min() < 50.0
    assert np.any((baselines > 100.0) & (baselines < 500.0))
    assert baselines.max() > 1000.0
    assert probe.frequency_hz.size == 3
    assert np.unique(probe.time_s).size == 3
    assert np.any(probe.uvw_m[:, 2] > 0.0) and np.any(probe.uvw_m[:, 2] < 0.0)
    assert np.any(np.abs(probe.uvw_m[:, 0]) > np.abs(probe.uvw_m[:, 1]))
    assert np.any(np.abs(probe.uvw_m[:, 1]) > np.abs(probe.uvw_m[:, 0]))
    assert np.any(probe.uvw_m[:, 0] > 0.0) and np.any(probe.uvw_m[:, 0] < 0.0)


def test_phase4_constant_beam_stays_at_analytic_square() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    for width_arcsec in _WIDTHS_ARCSEC:
        table = _square_table(0.0, 0.0, np.deg2rad(width_arcsec / 3600.0), 1.2)
        report = plan_integration(
            table,
            probe,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
        )
        assigned = _assignment(report, table.components[0].component_id)
        assert assigned.depth == 0
        assert RefinementReason.CONVERGED.value in assigned.reasons


def test_phase4_depths_for_representative_widths_and_regimes() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_NONTRIVIAL * 12.0,
        hess_ll=_NONTRIVIAL * 40.0,
        hess_mm=_NONTRIVIAL * 25.0,
    )
    placements = {
        "main_lobe": (0.0, 0.0),
        "half_power": (0.0015, 0.0),
        "pointing_ring": (0.0, -0.0015),
        "sidelobe": (0.006, 0.0),
        "other_side": (-0.0015, 0.0),
    }
    for width_arcsec in _WIDTHS_ARCSEC:
        for name, (l_rad, m_rad) in placements.items():
            table = _square_table(l_rad, m_rad, np.deg2rad(width_arcsec / 3600.0), 1.0)
            errors = successive_depth_errors(
                table,
                table.components[0],
                probe,
                beam,
                antenna_position_m=_ANTENNA_POSITION_M,
                calibration_state="casa_parang_true",
                max_depth=3,
            )
            assert len(errors) == 3
            assert errors[0] >= errors[1] - 1.0e-12
            report = plan_integration(
                table,
                probe,
                beam,
                antenna_position_m=_ANTENNA_POSITION_M,
                calibration_state="casa_parang_true",
                tolerance=IntegrationTolerance(absolute=1.0e-6, relative=1.0e-4),
            )
            assigned = _assignment(report, table.components[0].component_id)
            assert 0 <= assigned.depth <= 3, name


def test_phase4_support_boundary_and_null_force_subdivision() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    width = np.deg2rad(60.0 / 3600.0)
    table = _square_table(0.0, 0.0, width, 1.0)
    boundary = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        valid_radius_rad=width / 4.0,
    )
    report = plan_integration(
        table,
        probe,
        boundary,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assigned = _assignment(report, table.components[0].component_id)
    assert assigned.depth >= 2
    assert RefinementReason.VALIDITY_BOUNDARY.value in assigned.reasons
    null = ManufacturedVoltageBeam(
        intercept=np.zeros((2, 2), dtype=np.complex128),
        grad_l=_IDENTITY * 80.0,
    )
    report = plan_integration(
        table,
        probe,
        null,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assigned = _assignment(report, table.components[0].component_id)
    assert assigned.depth >= 2
    assert {
        RefinementReason.PHASE_REVERSAL.value,
        RefinementReason.BEAM_NULL.value,
    } & set(assigned.reasons)


def test_phase4_plan_ignores_measured_visibilities() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    table = _square_table(0.001, 0.0, np.deg2rad(16.0 / 3600.0), 1.0)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, grad_l=_NONTRIVIAL * 8.0)
    first = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    mutated = VisibilityBlock(
        uvw_m=probe.uvw_m,
        frequency_hz=probe.frequency_hz,
        visibility=np.full(probe.visibility.shape, 12.0 + 3.0j),
        weight=probe.weight,
        flag=probe.flag,
        time_s=probe.time_s,
        antenna1=probe.antenna1,
        antenna2=probe.antenna2,
        correlations=probe.correlations,
        receptor_basis=probe.receptor_basis,
        phase_centre_rad=probe.phase_centre_rad,
        provenance=dict(probe.provenance),
    )
    second = plan_integration(
        table,
        mutated,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert first.depth_by_parent() == second.depth_by_parent()
    assert first.assignments[0].error_estimate == second.assignments[0].error_estimate


def test_phase4_looser_tolerance_cannot_increase_depth() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    table = _square_table(0.0, 0.0, np.deg2rad(60.0 / 3600.0), 1.4)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_NONTRIVIAL * 15.0,
        hess_ll=_NONTRIVIAL * 50.0,
    )
    tight = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(absolute=1.0e-8, relative=1.0e-6),
    )
    loose = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(absolute=1.0, relative=1.0),
    )
    parent = table.components[0].component_id
    assert _assignment(loose, parent).depth <= _assignment(tight, parent).depth


def test_phase4_audit_detects_under_resolved_plans() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    table = _square_table(0.0, 0.0, np.deg2rad(60.0 / 3600.0), 1.4)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_NONTRIVIAL * 15.0,
        hess_ll=_NONTRIVIAL * 50.0,
    )
    report = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(absolute=1.0e-8, relative=1.0e-6),
    )
    shallow = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(
            absolute=1.0e-8,
            relative=1.0e-6,
            min_depth=0,
            max_depth=0,
        ),
    )
    assert _assignment(shallow, table.components[0].component_id).depth == 0
    audit = audit_integration_plan(
        table,
        shallow,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        audit_depth=report.assignments[0].depth,
    )
    assert audit.under_resolved
    constant = plan_integration(
        table,
        probe,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    accepted = audit_integration_plan(
        table,
        constant,
        probe,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert not accepted.under_resolved
    assert constant.assignments[0].depth == 0
    assert RefinementReason.CONVERGED.value in constant.assignments[0].reasons


def test_phase4_delta_stays_at_depth_zero() -> None:
    table = sky_table_from_records(
        [
            {
                "component_id": "catalogue:nvss:delta:0",
                "family": "catalogue",
                "basis_type": "delta",
                "l_rad": 0.01,
                "m_rad": 0.0,
                "stokes_i_jy": 2.0,
                "width_rad": 0.0,
                "active": True,
                "provenance": {"mosaic_name": "nvss"},
            }
        ],
        mosaic_phase_centre_rad=_PHASE,
    )
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    report = plan_integration(
        table,
        probe,
        ManufacturedVoltageBeam(intercept=_NONTRIVIAL, grad_l=_IDENTITY * 20.0),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assigned = _assignment(report, "catalogue:nvss:delta:0")
    assert assigned.depth == 0
    assert assigned.reasons == (RefinementReason.DELTA.value,)


def test_phase4_zero_flux_uses_the_amplitude_floor() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    empty = _square_table(0.0, 0.0, np.deg2rad(60.0 / 3600.0), 0.0)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        hess_ll=_NONTRIVIAL * 60.0,
        hess_mm=_NONTRIVIAL * 40.0,
    )
    without_floor = plan_integration(
        empty,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(flux_floor_jy=0.0),
    )
    with_floor = plan_integration(
        empty,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(flux_floor_jy=0.05),
    )
    assert without_floor.assignments[0].apparent_norm == 0.0
    assert without_floor.assignments[0].depth == 0
    assert with_floor.assignments[0].apparent_norm > 0.0


def test_phase4_apparent_contribution_keeps_faint_leaves_cheap() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    width = np.deg2rad(60.0 / 3600.0)
    bright = _square_table(0.0, 0.0, width, 2.0)
    faint = _square_table(0.008, 0.0, width, 2.0)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_IDENTITY * -90.0,
        hess_ll=_IDENTITY * 40.0,
    )
    tolerance = IntegrationTolerance(absolute=5.0e-3, relative=0.0, flux_floor_jy=0.0)
    bright_report = plan_integration(
        bright,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=tolerance,
    )
    faint_report = plan_integration(
        faint,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=tolerance,
    )
    assert bright_report.assignments[0].apparent_norm > faint_report.assignments[0].apparent_norm
    assert faint_report.assignments[0].depth <= bright_report.assignments[0].depth


def test_phase4_pointings_may_use_different_orders() -> None:
    width = np.deg2rad(60.0 / 3600.0)
    table = _square_table(0.0, 0.0, width, 1.0)
    on_axis = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    offset_centre = (_PHASE[0] + np.deg2rad(0.4), _PHASE[1])
    offset = VisibilityBlock(
        uvw_m=on_axis.uvw_m,
        frequency_hz=on_axis.frequency_hz,
        visibility=on_axis.visibility,
        weight=on_axis.weight,
        flag=on_axis.flag,
        time_s=on_axis.time_s,
        antenna1=on_axis.antenna1,
        antenna2=on_axis.antenna2,
        correlations=on_axis.correlations,
        receptor_basis=on_axis.receptor_basis,
        phase_centre_rad=offset_centre,
        provenance={"pointing_id": "offset"},
    )
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        hess_ll=_NONTRIVIAL * 400.0,
        hess_mm=_NONTRIVIAL * 250.0,
        valid_radius_rad=0.003,
    )
    report = plan_integration(
        table,
        (on_axis, offset),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        pointing_ids=("on_axis", "offset"),
        tolerance=IntegrationTolerance(absolute=1.0e-10, relative=0.0),
    )
    on_axis_assigned = _assignment(report, table.components[0].component_id, "on_axis")
    offset_assigned = _assignment(report, table.components[0].component_id, "offset")
    assert on_axis_assigned.depth > offset_assigned.depth
    conservative = integration_plan_from_planner(table, report)
    assert conservative.parent_count == 1
    assert report.depth_by_parent()[table.components[0].component_id] == on_axis_assigned.depth


def test_phase4_pointing_offset_changes_the_local_beam() -> None:
    width = np.deg2rad(16.0 / 3600.0)
    table = _square_table(0.0, 0.0, width, 1.0)
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_NONTRIVIAL * 20.0,
    )
    centre = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(pointing_offset_lm_rad=(0.0, 0.0)),
        tolerance=IntegrationTolerance(absolute=1.0e-6, relative=1.0e-4),
    )
    shifted = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(pointing_offset_lm_rad=(0.01, 0.0)),
        tolerance=IntegrationTolerance(absolute=1.0e-6, relative=1.0e-4),
    )
    assert centre.assignments[0].error_estimate != shifted.assignments[0].error_estimate


def test_phase4_physical_beams_plan_inside_support() -> None:
    table = _square_table(0.001, 0.0, np.deg2rad(8.0 / 3600.0), 0.8)
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    beams = {
        "airy": AnalyticAiryVoltageBeam(),
        "perley_plus_airy": voltage_beam_for_mode("streamed_scalar"),
        "cassbeam_diagonal": voltage_beam_for_mode("diagonal_copolar"),
    }
    for beam in beams.values():
        report = plan_integration(
            table,
            probe,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
        )
        assigned = report.assignments[0]
        assert 0 <= assigned.depth <= 3
        plan = integration_plan_from_planner(table, report)
        assert plan.parent_count == 1


def test_phase4_composite_handover_and_cassbeam_frequency_nodes() -> None:
    frequency = 4.536e9
    window = select_perley2016_cband_window(
        frequency, policy=PerleyFrequencyPolicy.CASA_NEAREST
    )
    radius = np.deg2rad(window.support_radius_arcmin(frequency) / 60.0)
    width = np.deg2rad(60.0 / 3600.0)
    table = _square_table(radius, 0.0, width, 1.0)
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    composite = voltage_beam_for_mode("streamed_scalar")
    report = plan_integration(
        table,
        probe,
        composite,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    reasons = set(report.assignments[0].reasons)
    assert reasons & {
        RefinementReason.OUTER_HANDOVER.value,
        RefinementReason.VALIDITY_BOUNDARY.value,
    }
    artifact = load_cassbeam_cband_artifact()
    nodes = [float(table.frequency_hz) for table in artifact.tables[:2]]
    if len(set(nodes)) < 2:
        pytest.skip("CASSBEAM artifact does not expose two frequency nodes")
    node_probe = VisibilityBlock(
        uvw_m=probe.uvw_m,
        frequency_hz=np.array(nodes[:2]),
        visibility=np.zeros((probe.uvw_m.shape[0], 2, len(probe.correlations))),
        weight=np.ones((probe.uvw_m.shape[0], 2, len(probe.correlations))),
        flag=np.zeros((probe.uvw_m.shape[0], 2, len(probe.correlations)), dtype=bool),
        time_s=probe.time_s,
        antenna1=probe.antenna1,
        antenna2=probe.antenna2,
        correlations=probe.correlations,
        receptor_basis=probe.receptor_basis,
        phase_centre_rad=probe.phase_centre_rad,
    )
    cassbeam = CassbeamCBandVoltageBeam(
        artifact,
        off_diagonal=True,
        allow_unfrozen=True,
        outer=voltage_beam_for_mode("streamed_scalar"),
    )
    on_axis = _square_table(0.0, 0.0, width, 1.0)
    node_report = plan_integration(
        on_axis,
        node_probe,
        cassbeam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert RefinementReason.FREQUENCY_NODE.value in node_report.assignments[0].reasons
    assert len(node_report.assignments[0].channel_regimes_hz) == 2
    single_depths = []
    for frequency in nodes[:2]:
        single_probe = VisibilityBlock(
            uvw_m=node_probe.uvw_m,
            frequency_hz=np.array([frequency]),
            visibility=node_probe.visibility[:, :1],
            weight=node_probe.weight[:, :1],
            flag=node_probe.flag[:, :1],
            time_s=node_probe.time_s,
            antenna1=node_probe.antenna1,
            antenna2=node_probe.antenna2,
            correlations=node_probe.correlations,
            receptor_basis=node_probe.receptor_basis,
            phase_centre_rad=node_probe.phase_centre_rad,
        )
        single_depths.append(
            plan_integration(
                on_axis,
                single_probe,
                cassbeam,
                antenna_position_m=_ANTENNA_POSITION_M,
                calibration_state="casa_parang_true",
            ).assignments[0].depth
        )
    assert node_report.assignments[0].depth == max(single_depths)
    assert node_report.assignments[0].depth_by_regime == tuple(single_depths)


@dataclass(frozen=True)
class _LaterPAValidityBeam:
    """Identity Jones that only clips support after a parallactic cutoff."""

    late_chi: float
    valid_radius_rad: float
    model_id: str = "later_pa_validity"

    @property
    def antenna_planes_from_parallactic(self) -> bool:
        return True

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state,
    ) -> BeamEvaluation:
        require_beam_calibration_state(calibration_state)
        from sl1mjax.voltage_beam import _pointing_relative_lm

        l_rad, m_rad = _pointing_relative_lm(coordinates)
        n_channel = int(np.asarray(coordinates.frequency_hz).size)
        chi = np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64).reshape(-1)
        n_ant = int(chi.size)
        jones = np.broadcast_to(
            np.eye(2, dtype=np.complex128),
            (n_ant, l_rad.size, n_channel, 2, 2),
        ).copy()
        valid = np.ones((n_ant, l_rad.size, n_channel), dtype=bool)
        if float(np.mean(np.abs(chi))) >= self.late_chi:
            inside = (l_rad * l_rad + m_rad * m_rad) <= self.valid_radius_rad**2
            valid &= inside[None, :, None]
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            provenance={"model_id": self.model_id},
            off_diagonal_valid=valid,
        )


def test_phase4_audit_flags_unresolved_max_depth_even_when_oracle_matches() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    table = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.0)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    report = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        tolerance=IntegrationTolerance(min_depth=3, max_depth=3),
    )
    assigned = report.assignments[0]
    assert assigned.depth == 3
    assert RefinementReason.MAX_DEPTH.value in assigned.reasons
    assert RefinementReason.CONVERGED.value not in assigned.reasons
    audit = audit_integration_plan(
        table,
        report,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert audit.findings[0].audit_depth == 4
    assert audit.under_resolved


def test_phase4_feature_screen_uses_every_parallactic_bin() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    times = np.unique(probe.time_s)
    chi = parallactic_angle_rad(times, _PHASE, _ANTENNA_POSITION_M)
    mean_abs = np.mean(np.abs(chi), axis=1)
    if float(np.ptp(mean_abs)) < 1.0e-3:
        pytest.skip("synthetic probe times do not separate parallactic angle")
    cutoff = 0.5 * (float(np.min(mean_abs)) + float(np.max(mean_abs)))
    width = np.deg2rad(60.0 / 3600.0)
    table = _square_table(0.0, 0.0, width, 1.0)
    beam = _LaterPAValidityBeam(late_chi=cutoff, valid_radius_rad=width / 4.0)
    report = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assigned = _assignment(report, table.components[0].component_id)
    assert assigned.depth >= 2
    assert RefinementReason.VALIDITY_BOUNDARY.value in assigned.reasons


def test_phase4_stale_planner_report_fails_closed() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    table = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.0)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    report = plan_integration(
        table,
        probe,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    extra = _square_table(
        0.002,
        0.0,
        np.deg2rad(8.0 / 3600.0),
        0.5,
        component_id="central_tree:central:0:0:1",
    )
    combined = sky_table_from_records(
        [*sky_table_to_records(table), *sky_table_to_records(extra)],
        mosaic_phase_centre_rad=_PHASE,
    )
    with pytest.raises(ValueError, match="selected sky parents"):
        integration_plan_from_planner(combined, report)
    with pytest.raises(ValueError, match="parent_policy"):
        integration_plan_from_planner(
            table, report, parent_policy="positive_flux"
        )
    shifted = sky_table_from_records(
        sky_table_to_records(table),
        mosaic_phase_centre_rad=(_PHASE[0] + 1.0e-3, _PHASE[1]),
    )
    with pytest.raises(ValueError, match="mosaic phase centre"):
        integration_plan_from_planner(shifted, report)


def test_phase4_parent_visibilities_match_isolated_predicts() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    first = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.2)
    second = _square_table(
        0.001,
        0.0,
        np.deg2rad(8.0 / 3600.0),
        0.8,
        component_id="central_tree:central:0:0:1",
    )
    combined = sky_table_from_records(
        [*sky_table_to_records(first), *sky_table_to_records(second)],
        mosaic_phase_centre_rad=_PHASE,
    )
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL, grad_l=_IDENTITY * 6.0)
    plan = integration_plan_from_table(
        combined,
        depth_by_parent={
            first.components[0].component_id: 1,
            second.components[0].component_id: 1,
        },
    )
    fluxes = np.array([1.2, 0.8])
    joint = predict_voltage_from_plan(
        probe,
        plan,
        fluxes,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        split_parents=True,
    )
    assert joint.parent_visibility is not None
    assert joint.parent_visibility.shape[0] == 2
    np.testing.assert_allclose(
        joint.parent_visibility.sum(axis=0),
        joint.visibility,
        rtol=0.0,
        atol=1.0e-12,
    )
    isolated = []
    for table, flux in ((first, 1.2), (second, 0.8)):
        isolated.append(
            predict_voltage_from_plan(
                probe,
                integration_plan_from_table(
                    table,
                    depth_by_parent={table.components[0].component_id: 1},
                ),
                np.array([flux]),
                beam,
                antenna_position_m=_ANTENNA_POSITION_M,
                calibration_state="casa_parang_true",
            ).visibility
        )
    np.testing.assert_allclose(
        joint.parent_visibility[0], isolated[0], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        joint.parent_visibility[1], isolated[1], rtol=0.0, atol=1.0e-12
    )


def test_phase4_batched_planner_does_not_scale_with_serial_parent_predicts() -> None:
    probe = stratified_integration_probe(
        phase_centre_rad=_PHASE, antenna_position_m=_ANTENNA_POSITION_M
    )
    width = np.deg2rad(8.0 / 3600.0)
    records = []
    for index in range(32):
        records.append(
            {
                "component_id": f"central_tree:central:0:0:{index}",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 1.0e-4 * index,
                "m_rad": 0.0,
                "stokes_i_jy": 1.0,
                "width_rad": width,
                "level": 0,
                "iy": 0,
                "ix": index,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            }
        )
    table = sky_table_from_records(records, mosaic_phase_centre_rad=_PHASE)
    report = plan_integration(
        table,
        probe,
        ManufacturedVoltageBeam(intercept=_IDENTITY),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert report.provenance["batched_parent_visibilities"] is True
    assert int(report.provenance["n_predictor_call"]) <= report.tolerance.max_depth + 1
    assert int(report.provenance["n_predictor_call"]) < len(records)
    assert report.provenance["n_parent"] == 32


def test_phase4_probe_stratifies_pa_orientation_and_w() -> None:
    n_time = 8
    n_uv = 12
    times = 5.0e9 + np.linspace(0.0, 30_000.0, n_time)
    lengths = np.geomspace(20.0, 2500.0, n_uv)
    angles = np.linspace(0.0, 2.0 * np.pi, n_uv, endpoint=False)
    w_m = np.linspace(-40.0, 40.0, n_uv)
    rows = []
    row_times = []
    for time_s in times:
        for length, angle, w in zip(lengths, angles, w_m, strict=True):
            rows.append((length * np.cos(angle), length * np.sin(angle), w))
            row_times.append(time_s)
    uvw = np.asarray(rows, dtype=np.float64)
    vis = np.ones((uvw.shape[0], 5, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=uvw,
        frequency_hz=np.linspace(4.536e9, 4.662e9, 5),
        visibility=vis,
        weight=np.ones_like(vis, dtype=np.float64),
        flag=np.zeros(vis.shape, dtype=bool),
        time_s=np.asarray(row_times),
        antenna1=np.zeros(uvw.shape[0], dtype=np.int32),
        antenna2=np.ones(uvw.shape[0], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    probe = select_planner_probe(block, antenna_position_m=_ANTENNA_POSITION_M)
    assert np.all(probe.visibility == 0.0)
    assert np.unique(probe.time_s).size == 3
    baselines = np.hypot(probe.uvw_m[:, 0], probe.uvw_m[:, 1])
    assert baselines.min() < 100.0
    assert baselines.max() > 1000.0
    assert np.any(probe.uvw_m[:, 2] > 0.0) and np.any(probe.uvw_m[:, 2] < 0.0)
    assert np.any(np.abs(probe.uvw_m[:, 0]) > np.abs(probe.uvw_m[:, 1]))
    assert np.any(np.abs(probe.uvw_m[:, 1]) > np.abs(probe.uvw_m[:, 0]))
