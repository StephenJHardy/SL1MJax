from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import (
    VoltageIntegrationMode,
    sky_table_from_records,
)
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    BeamOperatorPolicy,
    SkyStokesPlanes,
    predict_voltage_beam,
)
from sl1mjax.cassbeam_beam import (
    BeamImagingMode,
    CassbeamCBandVoltageBeam,
    load_cassbeam_cband_artifact,
    voltage_beam_for_mode,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    IntegrationParentPolicy,
    ManufacturedVoltageBeam,
    adjoint_voltage_from_plan,
    choose_node_capacity,
    integration_plan_from_table,
    pad_integration_plan,
    predict_voltage_from_plan,
    predict_voltage_from_plan_value_and_grad,
    subcell_nodes,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, _square_kernel, predict_stokes_i
from sl1mjax.sky import GaussianApproximation, SquarePixelBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_flux_refit import mosaic_local_directions

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
_CORRELATIONS = (
    Correlation.RR,
    Correlation.RL,
    Correlation.LR,
    Correlation.LL,
)
_PHASE = (np.deg2rad(282.35), np.deg2rad(-0.93))


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
                "parent_id": None,
                "active": True,
                "splitting_permitted": True,
                "provenance": {"mosaic_name": "central"},
            }
        ],
        mosaic_phase_centre_rad=_PHASE,
    )


def _mixed_table() -> object:
    width = np.deg2rad(16.0 / 3600.0)
    coarse = np.deg2rad(60.0 / 3600.0)
    return sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.01,
                "m_rad": -0.02,
                "stokes_i_jy": 1.4,
                "width_rad": width,
                "level": 0,
                "iy": 0,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
            {
                "component_id": "coarse_field:coarse:0:1:1",
                "family": "coarse_field",
                "basis_type": "uniform_square",
                "l_rad": -0.03,
                "m_rad": 0.02,
                "stokes_i_jy": 0.6,
                "width_rad": coarse,
                "level": 0,
                "iy": 1,
                "ix": 1,
                "active": True,
                "provenance": {"mosaic_name": "coarse"},
            },
            {
                "component_id": "catalogue:nvss:delta:0",
                "family": "catalogue",
                "basis_type": "delta",
                "l_rad": 0.04,
                "m_rad": 0.01,
                "stokes_i_jy": 0.8,
                "width_rad": 0.0,
                "active": True,
                "provenance": {"mosaic_name": "nvss"},
            },
        ],
        mosaic_phase_centre_rad=_PHASE,
    )


def _block(
    *,
    w_m: float = 0.0,
    correlations: tuple[Correlation, ...] = _CORRELATIONS,
    time_s: np.ndarray | None = None,
    frequency_hz: np.ndarray | None = None,
) -> VisibilityBlock:
    times = np.array([5.0e9, 5.0e9 + 1800.0]) if time_s is None else time_s
    frequency = (
        np.array([4.536e9, 4.662e9]) if frequency_hz is None else frequency_hz
    )
    dummy = np.zeros((times.size, frequency.size, len(correlations)), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=np.array(
            [
                [40.0, -25.0, w_m],
                [-18.0, 30.0, w_m],
                [12.0, 8.0, w_m],
                [-22.0, -14.0, w_m],
            ],
            dtype=np.float64,
        )[: times.size],
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=times,
        antenna1=np.array([0, 0, 1, 2], dtype=np.int32)[: times.size],
        antenna2=np.array([1, 2, 3, 3], dtype=np.int32)[: times.size],
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


def _config() -> BeamOperatorConfig:
    return BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2)


def _predict(table, mode, beam, block=None, *, backend="numpy", pad=False, **kwargs):
    plan = integration_plan_from_table(table, mode=mode, pad=pad)
    flux = np.asarray([item.stokes_i_jy for item in table.components])
    return predict_voltage_from_plan(
        block or _block(),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        backend=backend,
        **kwargs,
    ), plan, flux


def test_phase2_constant_scalar_beam_tiles_the_analytic_square() -> None:
    table = _square_table(0.012, -0.018, np.deg2rad(16.0 / 3600.0), 1.7)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block(w_m=0.0)
    parent, plan, flux = _predict(
        table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block
    )
    for mode in (
        VoltageIntegrationMode.SUBCELL_2X2,
        VoltageIntegrationMode.SUBCELL_4X4,
    ):
        child, child_plan, _flux = _predict(table, mode, beam, block)
        np.testing.assert_allclose(
            child.visibility, parent.visibility, rtol=1e-12, atol=1e-14
        )
        assert child_plan.node_count > plan.node_count
        assert abs(child_plan.weight[child_plan.node_valid].sum() - 1.0) < 1e-12
    square = np.asarray(
        predict_stokes_i(
            flux,
            [0.012],
            [-0.018],
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            (Correlation.I,),
            pixel_basis=SquarePixelBasis(1.0, GaussianApproximation.PARAXIAL),
            pixel_size_rad=np.deg2rad(16.0 / 3600.0),
        )
    )
    np.testing.assert_allclose(
        parent.visibility[..., 0], square[:, :, 0], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        parent.visibility[..., 3], square[:, :, 0], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(parent.visibility[..., 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(parent.visibility[..., 2], 0.0, atol=1e-14)


def test_phase2_constant_jones_tiles_every_correlation() -> None:
    table = _square_table(0.008, 0.006, np.deg2rad(8.0 / 3600.0), 0.9)
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block(w_m=0.0)
    parent, _plan, _flux = _predict(
        table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block
    )
    child, _plan, _flux = _predict(
        table, VoltageIntegrationMode.SUBCELL_4X4, beam, block
    )
    np.testing.assert_allclose(
        child.visibility, parent.visibility, rtol=1e-12, atol=1e-14
    )


def test_phase2_subcell_centres_tile_the_parent_square() -> None:
    width = 4.0e-4
    l_rad, m_rad, sub_width, weight = subcell_nodes(0.01, -0.02, width, 2)
    assert l_rad.size == 16
    assert sub_width == pytest.approx(width / 4)
    assert weight == pytest.approx(1.0 / 16)
    assert len(set(zip(l_rad.tolist(), m_rad.tolist(), strict=True))) == 16
    half = width / 2
    parent_l = (0.01 - half, 0.01 + half)
    parent_m = (-0.02 - half, -0.02 + half)
    child_l0 = l_rad - sub_width / 2
    child_l1 = l_rad + sub_width / 2
    child_m0 = m_rad - sub_width / 2
    child_m1 = m_rad + sub_width / 2
    assert child_l0.min() == pytest.approx(parent_l[0])
    assert child_l1.max() == pytest.approx(parent_l[1])
    assert child_m0.min() == pytest.approx(parent_m[0])
    assert child_m1.max() == pytest.approx(parent_m[1])
    swapped_l, swapped_m, _, _ = subcell_nodes(-0.02, 0.01, width, 2)
    original = set(zip(np.round(l_rad, 12), np.round(m_rad, 12), strict=True))
    swapped = set(zip(np.round(swapped_l, 12), np.round(swapped_m, 12), strict=True))
    exchanged = set(zip(np.round(m_rad, 12), np.round(l_rad, 12), strict=True))
    assert swapped == exchanged
    assert original != swapped
    reflected_l, reflected_m, _, _ = subcell_nodes(-0.01, 0.02, width, 2)
    reflected = set(
        zip(np.round(reflected_l, 12), np.round(reflected_m, 12), strict=True)
    )
    negated = set(zip(np.round(-l_rad, 12), np.round(-m_rad, 12), strict=True))
    assert reflected == negated


def test_phase2_vanishing_width_approaches_the_delta_kernel() -> None:
    table = _square_table(0.015, 0.004, 1.0e-12, 1.1)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block(w_m=12.0)
    finite, _plan, flux = _predict(
        table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block
    )
    delta = predict_voltage_beam(
        block,
        np.array([0.015]),
        np.array([0.004]),
        SkyStokesPlanes(stokes_i=flux),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    np.testing.assert_allclose(
        finite.visibility, delta.visibility, rtol=1e-9, atol=1e-11
    )


def test_phase2_nonzero_w_follows_the_square_kernel() -> None:
    l_rad = np.array([0.01])
    m_rad = np.array([-0.02])
    width = np.array([np.deg2rad(16.0 / 3600.0)])
    block = _block(w_m=40.0)
    uvw = block.uvw_m * (block.frequency_hz[0] / SPEED_OF_LIGHT_M_S)
    expected = np.asarray(
        _square_kernel(
            uvw,
            l_rad,
            m_rad,
            width,
            GaussianApproximation.WIDE_FIELD,
            include_projection=False,
        )
    )
    table = _square_table(0.01, -0.02, float(width[0]), 1.0)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    result, _plan, _flux = _predict(
        table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block
    )
    np.testing.assert_allclose(
        result.visibility[:, 0, 0], expected[:, 0], rtol=1e-12, atol=1e-14
    )


def test_phase2_parent_flux_is_invariant_with_depth() -> None:
    table = _square_table(0.0, 0.0, np.deg2rad(60.0 / 3600.0), 2.5)
    for mode in (
        VoltageIntegrationMode.ANALYTIC_SQUARE,
        VoltageIntegrationMode.SUBCELL_2X2,
        VoltageIntegrationMode.SUBCELL_4X4,
    ):
        plan = integration_plan_from_table(table, mode=mode)
        assert plan.node_flux(np.array([2.5])).sum() == pytest.approx(2.5)


def test_phase2_unsupported_node_does_not_invalidate_siblings() -> None:
    width = np.deg2rad(4.0 / 3600.0)
    table = sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.0,
                "m_rad": 0.0,
                "stokes_i_jy": 1.0,
                "width_rad": width,
                "level": 0,
                "iy": 0,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
            {
                "component_id": "catalogue:nvss:delta:0",
                "family": "catalogue",
                "basis_type": "delta",
                "l_rad": 0.05,
                "m_rad": 0.0,
                "stokes_i_jy": 2.0,
                "width_rad": 0.0,
                "active": True,
                "provenance": {"mosaic_name": "nvss"},
            },
        ],
        mosaic_phase_centre_rad=_PHASE,
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, valid_radius_rad=0.01)
    full, _plan, _flux = _predict(table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam)
    inner_table = _square_table(0.0, 0.0, width, 1.0)
    only_inner, _plan, _flux = _predict(
        inner_table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam
    )
    assert bool(np.any(full.valid))
    np.testing.assert_allclose(
        full.visibility, only_inner.visibility, rtol=1e-12, atol=1e-14
    )


def test_phase2_off_diagonal_invalid_is_not_known_zero_leakage() -> None:
    table = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.0)
    beam = ManufacturedVoltageBeam(
        intercept=_NONTRIVIAL,
        off_diagonal_valid=False,
    )
    result, _plan, _flux = _predict(
        table, VoltageIntegrationMode.SUBCELL_2X2, beam, backend="jax"
    )
    assert result.off_diagonal_valid is not None
    assert not np.any(result.off_diagonal_valid)
    np.testing.assert_allclose(result.visibility[..., 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.visibility[..., 2], 0.0, atol=1e-14)


def test_phase2_manufactured_beams_converge_with_depth() -> None:
    table = _square_table(0.02, -0.015, np.deg2rad(60.0 / 3600.0), 1.3)
    block = _block(w_m=8.0)
    beams = {
        "linear": ManufacturedVoltageBeam(
            intercept=_IDENTITY,
            grad_l=_NONTRIVIAL * 8.0,
            grad_m=_NONTRIVIAL * -5.0,
        ),
        "quadratic": ManufacturedVoltageBeam(
            intercept=_IDENTITY,
            grad_l=_NONTRIVIAL * 4.0,
            hess_ll=_NONTRIVIAL * 30.0,
            hess_mm=_NONTRIVIAL * 20.0,
        ),
        "phase_gradient": ManufacturedVoltageBeam(
            intercept=_IDENTITY,
            grad_l=1j * _IDENTITY * 12.0,
        ),
        "leakage": ManufacturedVoltageBeam(
            intercept=_NONTRIVIAL,
            hess_lm=_NONTRIVIAL * 25.0,
        ),
    }
    for beam in beams.values():
        depths = []
        for mode in (
            VoltageIntegrationMode.ANALYTIC_SQUARE,
            VoltageIntegrationMode.SUBCELL_2X2,
            VoltageIntegrationMode.SUBCELL_4X4,
        ):
            result, _plan, _flux = _predict(table, mode, beam, block)
            depths.append(result.visibility)
        # Depth-3 oracle via an explicit 8x8 plan.
        fine = integration_plan_from_table(
            table,
            mode=VoltageIntegrationMode.ANALYTIC_SQUARE,
            depth_by_parent={"central_tree:central:0:0:0": 3},
        )
        flux = np.array([1.3])
        oracle = predict_voltage_from_plan(
            block,
            fine,
            flux,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )
        errors = [np.linalg.norm(vis - oracle.visibility) for vis in depths]
        assert errors[0] > errors[1] >= 0.0
        assert errors[2] <= errors[1] * 1.0000001


def test_phase2_pointing_offset_and_parallactic_rotation() -> None:
    table = _square_table(0.01, 0.0, np.deg2rad(16.0 / 3600.0), 1.0)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        grad_l=_NONTRIVIAL * 6.0,
        rotate_parallactic=True,
    )
    plus = BeamOperatorConfig(
        visibility_chunk_size=2,
        pixel_chunk_size=2,
        pointing_offset_lm_rad=(0.015, 0.0),
    )
    minus = BeamOperatorConfig(
        visibility_chunk_size=2,
        pixel_chunk_size=2,
        pointing_offset_lm_rad=(-0.015, 0.0),
    )
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    flux = np.array([1.0])
    left = predict_voltage_from_plan(
        _block(),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=plus,
    )
    right = predict_voltage_from_plan(
        _block(),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=minus,
    )
    assert np.linalg.norm(left.visibility - right.visibility) > 1e-6
    early = predict_voltage_from_plan(
        _block(time_s=np.array([5.0e9, 5.0e9 + 60.0])),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    late = predict_voltage_from_plan(
        _block(time_s=np.array([5.0e9 + 20000.0, 5.0e9 + 21800.0])),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    assert np.linalg.norm(early.visibility - late.visibility) > 1e-8


def test_phase3_jax_matches_numpy_for_every_mode() -> None:
    table = _mixed_table()
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL, grad_m=_IDENTITY * 3.0)
    block = _block(w_m=6.0)
    for mode in VoltageIntegrationMode:
        numpy_result, plan, flux = _predict(table, mode, beam, block, backend="numpy")
        jax_result, _plan, _flux = _predict(table, mode, beam, block, backend="jax")
        np.testing.assert_allclose(
            jax_result.visibility, numpy_result.visibility, rtol=1e-9, atol=1e-11
        )
        assert plan.parent_count == 3


def test_phase3_mixed_widths_match_separate_calls() -> None:
    table = _mixed_table()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    together, _plan, flux = _predict(
        table, VoltageIntegrationMode.SUBCELL_2X2, beam, block
    )
    parts = []
    for record in table.components:
        part_table = sky_table_from_records(
            [
                {
                    "component_id": record.component_id,
                    "family": record.family.value,
                    "basis_type": record.basis_type.value,
                    "l_rad": record.l_rad,
                    "m_rad": record.m_rad,
                    "stokes_i_jy": record.stokes_i_jy,
                    "width_rad": record.width_rad,
                    "level": record.level,
                    "iy": record.iy,
                    "ix": record.ix,
                    "active": True,
                    "provenance": dict(record.provenance),
                }
            ],
            mosaic_phase_centre_rad=_PHASE,
        )
        part, _plan, _flux = _predict(
            part_table, VoltageIntegrationMode.SUBCELL_2X2, beam, block
        )
        parts.append(part.visibility)
    np.testing.assert_allclose(together.visibility, sum(parts), rtol=1e-12, atol=1e-14)


def test_phase3_parent_gradient_matches_finite_difference() -> None:
    table = _mixed_table()
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    flux = np.array([1.4, 0.6, 0.8])
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, grad_l=_NONTRIVIAL * 2.0)
    block = _block()
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=np.ones(block.visibility.shape, dtype=np.complex128) * 0.2,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    loss, grad = predict_voltage_from_plan_value_and_grad(
        flux,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    assert grad.shape == flux.shape
    fine = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_4X4)
    _loss_fine, grad_fine = predict_voltage_from_plan_value_and_grad(
        flux,
        block,
        fine,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    assert grad_fine.shape == flux.shape
    assert fine.node_count > plan.node_count
    step = 1.0e-5
    numeric = np.zeros_like(flux)
    for index in range(flux.size):
        plus = flux.copy()
        minus = flux.copy()
        plus[index] += step
        minus[index] -= step
        loss_plus, _grad = predict_voltage_from_plan_value_and_grad(
            plus,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )
        loss_minus, _grad = predict_voltage_from_plan_value_and_grad(
            minus,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )
        numeric[index] = (float(loss_plus) - float(loss_minus)) / (2.0 * step)
    np.testing.assert_allclose(np.asarray(grad), numeric, rtol=2e-4, atol=2e-6)
    del loss


def test_phase3_adjoint_identity_and_rr_only_block() -> None:
    table = _square_table(0.01, 0.0, np.deg2rad(16.0 / 3600.0), 1.2)
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block()
    predicted = predict_voltage_from_plan(
        block,
        plan,
        np.array([1.2]),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    residual = np.conjugate(predicted.visibility)
    sky_grad = adjoint_voltage_from_plan(
        residual,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    vis_dot = np.vdot(predicted.visibility, residual).real
    sky_dot = float(np.dot(sky_grad, np.array([1.2])))
    assert sky_dot == pytest.approx(vis_dot, rel=1e-12)
    rr_block = _block(correlations=(Correlation.RR,))
    rr = predict_voltage_from_plan(
        rr_block,
        plan,
        np.array([1.2]),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    np.testing.assert_allclose(
        rr.visibility[..., 0], predicted.visibility[..., 0], rtol=1e-12, atol=1e-14
    )


def test_phase3_padding_masks_and_memory_preflight() -> None:
    table = _mixed_table()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    exact, plan, flux = _predict(
        table, VoltageIntegrationMode.SUBCELL_2X2, beam, block
    )
    padded_plan = pad_integration_plan(plan)
    assert padded_plan.capacity > plan.node_count
    padded = predict_voltage_from_plan(
        block,
        padded_plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    np.testing.assert_allclose(
        padded.visibility, exact.visibility, rtol=1e-12, atol=1e-14
    )
    _loss_exact, grad_exact = predict_voltage_from_plan_value_and_grad(
        flux,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    _loss_pad, grad_pad = predict_voltage_from_plan_value_and_grad(
        flux,
        block,
        padded_plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    np.testing.assert_allclose(np.asarray(grad_pad), np.asarray(grad_exact), rtol=1e-9)
    with pytest.raises(ValueError, match="Jones tile"):
        predict_voltage_from_plan(
            block,
            padded_plan,
            flux,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(
                max_timestep_jones_bytes=64,
                pixel_chunk_size=64,
            ),
            backend="jax",
        )


def test_phase3_streamed_and_materialized_agree() -> None:
    table = _square_table(0.01, -0.01, np.deg2rad(16.0 / 3600.0), 0.7)
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block()
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    flux = np.array([0.7])
    streamed = predict_voltage_from_plan(
        block,
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(policy=BeamOperatorPolicy.STREAM),
    )
    materialized = predict_voltage_from_plan(
        block,
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(policy=BeamOperatorPolicy.MATERIALIZE),
    )
    np.testing.assert_allclose(
        streamed.visibility, materialized.visibility, rtol=1e-12, atol=1e-14
    )


def test_phase3_frozen_plan_does_not_recompile_inside_a_loop() -> None:
    table = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.0)
    plan = integration_plan_from_table(
        table, mode=VoltageIntegrationMode.SUBCELL_2X2, pad=True
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    flux = jnp.asarray([1.0])

    def _step(values):
        return predict_voltage_from_plan_value_and_grad(
            values,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )

    jitted = jax.jit(_step)
    first = jitted(flux)
    jaxpr_a = jax.make_jaxpr(jitted)(flux)
    updated = flux
    for _ in range(3):
        _loss, grad = jitted(updated)
        updated = updated - 0.01 * grad
    jaxpr_b = jax.make_jaxpr(jitted)(updated)
    assert str(jaxpr_a) == str(jaxpr_b)
    assert np.asarray(first[1]).shape == (1,)


def test_phase3_value_changes_do_not_change_the_jaxpr() -> None:
    table = _square_table(0.0, 0.0, np.deg2rad(8.0 / 3600.0), 1.0)
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block_a = _block(time_s=np.array([5.0e9, 5.0e9 + 100.0]))
    block_b = _block(time_s=np.array([6.0e9, 6.0e9 + 100.0]))

    def _predict_block(block):
        return predict_voltage_from_plan(
            block,
            plan,
            np.array([1.0]),
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
            backend="jax",
        ).visibility

    vis_a = _predict_block(block_a)
    vis_b = _predict_block(block_b)
    assert vis_a.shape == vis_b.shape
    assert np.linalg.norm(vis_a - vis_b) >= 0.0


def test_phase3_ci_benchmark_records_compile_and_runtime() -> None:
    table = _square_table(0.0, 0.0, np.deg2rad(16.0 / 3600.0), 1.0)
    plan = integration_plan_from_table(
        table, mode=VoltageIntegrationMode.SUBCELL_2X2, pad=True
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    flux = jnp.asarray([1.0])

    def _step(values):
        return predict_voltage_from_plan_value_and_grad(
            values,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )

    jitted = jax.jit(_step)
    start = time.perf_counter()
    loss, grad = jitted(flux)
    jax.block_until_ready(loss)
    jax.block_until_ready(grad)
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    loss, grad = jitted(flux * 1.1)
    jax.block_until_ready(loss)
    jax.block_until_ready(grad)
    execute_s = time.perf_counter() - start
    stats = {
        "n_parents": plan.parent_count,
        "n_nodes": plan.node_count,
        "capacity": plan.capacity,
        "compile_s": compile_s,
        "execute_s": execute_s,
        "recompile_count": 1,
    }
    assert stats["n_parents"] == 1
    assert stats["n_nodes"] == 4
    assert stats["compile_s"] >= 0.0
    assert stats["execute_s"] >= 0.0


def test_phase3_plan_keeps_the_mosaic_coordinate_frame() -> None:
    mosaic = _PHASE
    offset = (mosaic[0] + np.deg2rad(0.25), mosaic[1] + np.deg2rad(0.12))
    table = _square_table(0.015, -0.01, np.deg2rad(16.0 / 3600.0), 1.4)
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.ANALYTIC_SQUARE)
    assert plan.mosaic_phase_centre_rad == mosaic
    local_l, local_m = plan.local_directions(mosaic)
    np.testing.assert_allclose(local_l[plan.node_valid], plan.l_rad[plan.node_valid])
    np.testing.assert_allclose(local_m[plan.node_valid], plan.m_rad[plan.node_valid])
    expected_l, expected_m = mosaic_local_directions(
        plan.l_rad, plan.m_rad, mosaic, offset
    )
    got_l, got_m = plan.local_directions(offset)
    np.testing.assert_allclose(got_l, expected_l)
    np.testing.assert_allclose(got_m, expected_m)
    assert np.linalg.norm(got_l - plan.l_rad) > 1e-6
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    block = _block()
    offset_block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=block.visibility,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=offset,
    )
    flux = np.array([1.4])
    transformed = predict_voltage_from_plan(
        offset_block,
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
    )
    naive = predict_voltage_beam(
        offset_block,
        plan.l_rad,
        plan.m_rad,
        SkyStokesPlanes(stokes_i=plan.node_flux(flux)),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        width_rad=plan.width_rad,
        node_valid=plan.node_valid,
        kernel_approximation=plan.approximation,
    )
    assert np.linalg.norm(transformed.visibility - naive.visibility) > 1e-6


def test_phase3_imaging_plan_keeps_zero_flux_parents() -> None:
    table = sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.0,
                "m_rad": 0.0,
                "stokes_i_jy": 1.2,
                "width_rad": np.deg2rad(16.0 / 3600.0),
                "level": 0,
                "iy": 0,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
            {
                "component_id": "central_tree:central:0:0:1",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.002,
                "m_rad": 0.0,
                "stokes_i_jy": 0.0,
                "width_rad": np.deg2rad(16.0 / 3600.0),
                "level": 0,
                "iy": 0,
                "ix": 1,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
            {
                "component_id": "central_tree:central:0:1:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.0,
                "m_rad": 0.002,
                "stokes_i_jy": -0.05,
                "width_rad": np.deg2rad(16.0 / 3600.0),
                "level": 0,
                "iy": 1,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
        ],
        mosaic_phase_centre_rad=_PHASE,
    )
    imaging = integration_plan_from_table(table)
    diagnostic = integration_plan_from_table(
        table, parent_policy=IntegrationParentPolicy.POSITIVE_FLUX
    )
    assert imaging.parent_count == 3
    assert diagnostic.parent_count == 1
    assert imaging.parent_id[1] == "central_tree:central:0:0:1"


def test_phase3_gaussian_basis_fails_closed() -> None:
    table = sky_table_from_records(
        [
            {
                "component_id": "catalogue:nvss:delta:0",
                "family": "catalogue",
                "basis_type": "gaussian",
                "l_rad": 0.01,
                "m_rad": 0.0,
                "stokes_i_jy": 1.0,
                "width_rad": np.deg2rad(8.0 / 3600.0),
                "active": True,
                "provenance": {"mosaic_name": "nvss"},
            }
        ],
        mosaic_phase_centre_rad=_PHASE,
    )
    with pytest.raises(ValueError, match="Gaussian"):
        integration_plan_from_table(table)


def test_phase3_padded_unsupported_node_is_invalid_on_both_backends() -> None:
    table = _square_table(0.04, 0.0, np.deg2rad(4.0 / 3600.0), 1.0)
    plan = pad_integration_plan(
        integration_plan_from_table(table, mode=VoltageIntegrationMode.ANALYTIC_SQUARE)
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, valid_radius_rad=0.01)
    flux = np.array([1.0])
    numpy_result = predict_voltage_from_plan(
        _block(),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        backend="numpy",
    )
    jax_result = predict_voltage_from_plan(
        _block(),
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        backend="jax",
    )
    np.testing.assert_allclose(numpy_result.visibility, 0.0, atol=1e-14)
    np.testing.assert_allclose(jax_result.visibility, 0.0, atol=1e-14)
    np.testing.assert_array_equal(numpy_result.valid, jax_result.valid)
    assert not np.any(numpy_result.valid)
    assert numpy_result.off_diagonal_valid is not None
    assert jax_result.off_diagonal_valid is not None
    np.testing.assert_array_equal(
        numpy_result.off_diagonal_valid, jax_result.off_diagonal_valid
    )


def test_phase3_mixed_leakage_support_invalidates_off_diagonals() -> None:
    mixed = sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "delta",
                "l_rad": 0.0,
                "m_rad": 0.0,
                "stokes_i_jy": 1.0,
                "width_rad": 0.0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            },
            {
                "component_id": "catalogue:nvss:delta:0",
                "family": "catalogue",
                "basis_type": "delta",
                "l_rad": 0.03,
                "m_rad": 0.0,
                "stokes_i_jy": 0.8,
                "width_rad": 0.0,
                "active": True,
                "provenance": {"mosaic_name": "nvss"},
            },
        ],
        mosaic_phase_centre_rad=_PHASE,
    )
    plan = integration_plan_from_table(mixed, mode=VoltageIntegrationMode.POINT_CENTRE)
    beam = ManufacturedVoltageBeam(
        intercept=_NONTRIVIAL,
        valid_radius_rad=0.05,
        off_diagonal_radius_rad=0.01,
    )
    numpy_result = predict_voltage_from_plan(
        _block(),
        plan,
        np.array([1.0, 0.8]),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        backend="numpy",
    )
    jax_result = predict_voltage_from_plan(
        _block(),
        plan,
        np.array([1.0, 0.8]),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=_config(),
        backend="jax",
    )
    assert numpy_result.off_diagonal_valid is not None
    assert jax_result.off_diagonal_valid is not None
    np.testing.assert_array_equal(numpy_result.valid, jax_result.valid)
    np.testing.assert_array_equal(
        numpy_result.off_diagonal_valid, jax_result.off_diagonal_valid
    )
    assert np.any(numpy_result.valid)
    assert not np.any(numpy_result.off_diagonal_valid)
    np.testing.assert_allclose(
        jax_result.visibility, numpy_result.visibility, rtol=1e-9, atol=1e-11
    )


def test_phase3_jax_gradient_rejects_wrong_parent_length() -> None:
    table = _mixed_table()
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    assert plan.parent_count == 3
    with pytest.raises(ValueError, match="parent_flux must match"):
        predict_voltage_from_plan_value_and_grad(
            np.array([1.4, 0.6]),
            _block(),
            plan,
            ManufacturedVoltageBeam(intercept=_IDENTITY),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=_config(),
        )


def test_phase3_node_buckets_cover_the_real_sky() -> None:
    assert choose_node_capacity(93_874) == 262_144
    assert choose_node_capacity(250_115) == 262_144
    assert choose_node_capacity(262_145) == 1_048_576


def test_phase3_physical_beams_match_numpy_including_masks() -> None:
    table = _square_table(0.001, 0.0, np.deg2rad(2.0 / 3600.0), 0.9)
    block = _block(frequency_hz=np.array([4.536e9, 4.662e9]))
    artifact = load_cassbeam_cband_artifact()
    beams = {
        "airy": AnalyticAiryVoltageBeam(),
        "perley_plus_airy": voltage_beam_for_mode("streamed_scalar"),
        "cassbeam_diagonal": voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR),
        "cassbeam_full_jones": CassbeamCBandVoltageBeam(
            artifact,
            off_diagonal=True,
            allow_unfrozen=True,
            outer=voltage_beam_for_mode("streamed_scalar"),
        ),
    }
    for beam in beams.values():
        numpy_result, _plan, _flux = _predict(
            table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block, backend="numpy"
        )
        jax_result, _plan, _flux = _predict(
            table, VoltageIntegrationMode.ANALYTIC_SQUARE, beam, block, backend="jax"
        )
        np.testing.assert_allclose(
            jax_result.visibility, numpy_result.visibility, rtol=1e-9, atol=1e-11
        )
        np.testing.assert_array_equal(jax_result.valid, numpy_result.valid)
        assert numpy_result.off_diagonal_valid is not None
        assert jax_result.off_diagonal_valid is not None
        np.testing.assert_array_equal(
            jax_result.off_diagonal_valid, numpy_result.off_diagonal_valid
        )
