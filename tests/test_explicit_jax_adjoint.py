"""Gates for the streamed explicit JAX voltage adjoint."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.beam import VLABeamCatalog
from sl1mjax.beam_aware_imaging import VoltageIntegrationMode, sky_table_from_records
from sl1mjax.beam_operator import BeamOperatorConfig, BeamOperatorPolicy
from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    IntegrationPlan,
    ManufacturedVoltageBeam,
    adjoint_voltage_from_plan,
    integration_plan_from_table,
    pad_integration_plan,
    predict_voltage_from_plan,
    predict_voltage_from_plan_value_and_grad,
)
from sl1mjax.objective import effective_weight
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_operator_jax import (
    _EXPLICIT_KERNELS,
    EXPLICIT_KERNEL_CACHE_LIMIT,
    _adjoint_voltage_beam_jax_arrays,
    _beam_cache_key,
    _lookup_explicit_kernel,
    _reduce_parent_gradient,
    _store_explicit_kernel,
    clear_explicit_kernels,
    explicit_adjoint_workspace_bytes,
    explicit_cached_kernel_count,
    explicit_kernel_build_count,
    predict_voltage_from_plan_value_and_grad_explicit_jax,
)
from sl1mjax.voltage_reconstruction import (
    VoltageReconstructionConfig,
    starting_central_table,
)

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
_ROOT = np.deg2rad(16.0 / 3600.0)
_RTOL = 1.0e-8
_ATOL = 1.0e-10


def _square_table(flux: float = 1.2):
    return sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.01,
                "m_rad": 0.0,
                "stokes_i_jy": flux,
                "width_rad": _ROOT,
                "level": 0,
                "iy": 0,
                "ix": 0,
                "active": True,
                "provenance": {"mosaic_name": "central"},
            }
        ],
        mosaic_phase_centre_rad=_PHASE,
    )


def _mixed_table():
    return sky_table_from_records(
        [
            {
                "component_id": "central_tree:central:0:0:0",
                "family": "central_tree",
                "basis_type": "uniform_square",
                "l_rad": 0.01,
                "m_rad": -0.02,
                "stokes_i_jy": 1.4,
                "width_rad": _ROOT,
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
    correlations: tuple[Correlation, ...] = (Correlation.RR, Correlation.LL),
    visibility: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    flag: np.ndarray | None = None,
) -> VisibilityBlock:
    times = np.array([5.0e9, 5.0e9 + 1800.0])
    frequency = np.array([4.536e9, 4.662e9])
    dummy = np.full(
        (times.size, frequency.size, len(correlations)),
        0.2 + 0.05j,
        dtype=np.complex128,
    )
    vis = dummy if visibility is None else visibility
    return VisibilityBlock(
        uvw_m=np.array([[40.0, -25.0, 2.0], [-18.0, 30.0, -1.0]], dtype=np.float64),
        frequency_hz=frequency,
        visibility=vis,
        weight=np.ones_like(vis, dtype=np.float64) if weight is None else weight,
        flag=np.zeros(vis.shape, dtype=bool) if flag is None else flag,
        time_s=times,
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([2, 3], dtype=np.int32),
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


def _value_and_grad(flux, block, plan, beam, *, mode: str, config=None, mask=None):
    loss, gradient = predict_voltage_from_plan_value_and_grad(
        flux,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config or BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
        train_mask=mask,
        operator_mode=mode,
    )
    return jax.block_until_ready(loss), jax.block_until_ready(gradient)


def test_reconstruction_defaults_to_vjp_oracle() -> None:
    assert VoltageReconstructionConfig().operator_mode == "vjp"


@pytest.mark.parametrize(
    "beam",
    [
        ManufacturedVoltageBeam(intercept=_IDENTITY, grad_l=_NONTRIVIAL * 2.0),
        ManufacturedVoltageBeam(intercept=_NONTRIVIAL, rotate_parallactic=True),
        AnalyticAiryVoltageBeam(),
        voltage_beam_for_mode("streamed_scalar"),
        voltage_beam_for_mode("diagonal_copolar"),
    ],
    ids=("manufactured", "parallactic", "airy", "composite", "cassbeam"),
)
def test_explicit_matches_vjp_oracle(beam) -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
    )
    flux = np.array([1.4])
    block = _block()
    loss_vjp, grad_vjp = _value_and_grad(flux, block, plan, beam, mode="vjp")
    loss_exp, grad_exp = _value_and_grad(flux, block, plan, beam, mode="explicit_jax")
    np.testing.assert_allclose(float(loss_exp), float(loss_vjp), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grad_exp), np.asarray(grad_vjp), rtol=_RTOL, atol=_ATOL
    )


@pytest.mark.parametrize(
    "correlations",
    [
        (Correlation.RR,),
        (Correlation.LL,),
        (Correlation.RR, Correlation.LL),
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
    ],
    ids=("rr", "ll", "rr_ll", "full_circular"),
)
def test_explicit_matches_vjp_for_correlation_subsets(correlations) -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
    )
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block(correlations=correlations)
    loss_vjp, grad_vjp = _value_and_grad(np.array([0.9]), block, plan, beam, mode="vjp")
    loss_exp, grad_exp = _value_and_grad(
        np.array([0.9]), block, plan, beam, mode="explicit_jax"
    )
    np.testing.assert_allclose(float(loss_exp), float(loss_vjp), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grad_exp), np.asarray(grad_vjp), rtol=_RTOL, atol=_ATOL
    )


def test_explicit_adjoint_matches_numpy_with_shared_cotangent() -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, grad_m=_NONTRIVIAL)
    block = _block()
    flux = np.array([1.1])
    predicted = predict_voltage_from_plan(
        block,
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
        backend="jax",
    )
    sample = np.asarray(block.active)
    flag = ~sample | ~np.asarray(predicted.valid)[..., None]
    if predicted.off_diagonal_valid is not None and Correlation.RL in block.correlations:
        leak = np.asarray(predicted.off_diagonal_valid)
        for index, correlation in enumerate(block.correlations):
            if correlation in {Correlation.RL, Correlation.LR}:
                flag[..., index] |= ~leak
    observation = np.asarray(block.visibility)
    prediction = np.asarray(predicted.visibility)
    weight = np.asarray(block.weight)
    active = np.asarray(effective_weight(observation, weight, flag))
    weight_sum = float(np.sum(active))
    residual = np.where(active > 0, prediction - observation, 0.0)
    hilbert_residual = (2.0 / weight_sum) * active * residual
    numpy_grad = adjoint_voltage_from_plan(
        hilbert_residual,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
    )
    _loss_e, explicit_grad = _value_and_grad(
        flux, block, plan, beam, mode="explicit_jax"
    )
    np.testing.assert_allclose(
        np.asarray(explicit_grad), numpy_grad, rtol=_RTOL, atol=_ATOL
    )


def test_explicit_adjoint_satisfies_real_inner_product() -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
    )
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block()
    flux = np.array([1.2])
    predicted = predict_voltage_from_plan(
        block,
        plan,
        flux,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
        backend="jax",
    )
    residual = np.conjugate(predicted.visibility)
    local_l, local_m = plan.local_directions(block.phase_centre_rad)
    node_grad = _adjoint_voltage_beam_jax_arrays(
        residual,
        block,
        local_l,
        local_m,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
        width_rad=plan.width_rad,
        node_valid=plan.node_valid,
        kernel_approximation=plan.approximation,
    )
    parent_grad = np.asarray(
        _reduce_parent_gradient(
            node_grad,
            jnp.asarray(plan.parent_index),
            jnp.asarray(plan.weight),
            jnp.asarray(plan.node_valid),
            plan.parent_count,
        )
    )
    vis_dot = np.vdot(predicted.visibility, residual).real
    sky_dot = float(np.dot(parent_grad, flux))
    assert sky_dot == pytest.approx(vis_dot, rel=1e-12)
    numpy_grad = adjoint_voltage_from_plan(
        residual,
        block,
        plan,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
    )
    np.testing.assert_allclose(parent_grad, numpy_grad, rtol=_RTOL, atol=_ATOL)


def test_explicit_matches_finite_difference() -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
    )
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY, grad_l=_NONTRIVIAL * 2.0)
    block = _block()
    flux = np.array([1.4])
    loss, gradient = _value_and_grad(flux, block, plan, beam, mode="explicit_jax")
    step = 1.0e-5
    plus, _ = _value_and_grad(flux + step, block, plan, beam, mode="explicit_jax")
    minus, _ = _value_and_grad(flux - step, block, plan, beam, mode="explicit_jax")
    numeric = (float(plus) - float(minus)) / (2.0 * step)
    np.testing.assert_allclose(float(gradient[0]), numeric, rtol=2e-4, atol=2e-6)
    assert np.isfinite(float(loss))


def test_explicit_invariant_across_direction_tile_sizes() -> None:
    plan = integration_plan_from_table(
        _square_table(), mode=VoltageIntegrationMode.SUBCELL_4X4
    )
    beam = AnalyticAiryVoltageBeam()
    block = _block()
    flux = np.array([0.75])
    grads = []
    losses = []
    for chunk in (1, 3, plan.node_count):
        loss, grad = _value_and_grad(
            flux,
            block,
            plan,
            beam,
            mode="explicit_jax",
            config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=chunk),
        )
        losses.append(float(loss))
        grads.append(np.asarray(grad))
    for other in grads[1:]:
        np.testing.assert_allclose(grads[0], other, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(losses[0], losses[1], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(losses[0], losses[2], rtol=_RTOL, atol=_ATOL)


def test_explicit_respects_flags_weights_padding_and_beam_masks() -> None:
    plan = pad_integration_plan(
        integration_plan_from_table(
            _square_table(), mode=VoltageIntegrationMode.SUBCELL_2X2
        )
    )
    assert not np.all(plan.node_valid)
    beam = ManufacturedVoltageBeam(
        intercept=_IDENTITY,
        valid_radius_rad=0.03,
        off_diagonal_valid=False,
    )
    block = _block(
        correlations=(
            Correlation.RR,
            Correlation.RL,
            Correlation.LR,
            Correlation.LL,
        )
    )
    weight = np.asarray(block.weight, dtype=np.float64)
    weight[0, 0, 0] = 0.0
    flag = np.zeros(block.visibility.shape, dtype=bool)
    flag[1, 1, -1] = True
    masked = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=block.visibility,
        weight=weight,
        flag=flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    flux = np.array([1.0])
    loss_vjp, grad_vjp = _value_and_grad(flux, masked, plan, beam, mode="vjp")
    loss_exp, grad_exp = _value_and_grad(flux, masked, plan, beam, mode="explicit_jax")
    np.testing.assert_allclose(float(loss_exp), float(loss_vjp), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grad_exp), np.asarray(grad_vjp), rtol=_RTOL, atol=_ATOL
    )
    active = effective_weight(masked.visibility, masked.weight, masked.flag)
    assert float(np.sum(active)) < float(np.size(masked.visibility))


def test_explicit_mixed_depths_and_parent_order() -> None:
    table = _mixed_table()
    plan = integration_plan_from_table(table, mode=VoltageIntegrationMode.SUBCELL_2X2)
    beam = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    block = _block()
    flux = np.array([item.stokes_i_jy for item in table.components])
    assert plan.parent_count == flux.size
    assert {int(index) for index in plan.parent_index[plan.node_valid]} == {0, 1}
    loss_vjp, grad_vjp = _value_and_grad(flux, block, plan, beam, mode="vjp")
    loss_exp, grad_exp = _value_and_grad(flux, block, plan, beam, mode="explicit_jax")
    np.testing.assert_allclose(float(loss_exp), float(loss_vjp), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grad_exp), np.asarray(grad_vjp), rtol=_RTOL, atol=_ATOL
    )

    perm = np.array([1, 0], dtype=np.int32)
    remapped_ids = [""] * plan.parent_count
    remapped_flux = np.empty_like(flux)
    for old, new in enumerate(perm):
        remapped_ids[int(new)] = plan.parent_id[old]
        remapped_flux[int(new)] = flux[old]
    remapped = IntegrationPlan(
        parent_index=perm[plan.parent_index],
        l_rad=plan.l_rad,
        m_rad=plan.m_rad,
        width_rad=plan.width_rad,
        weight=plan.weight,
        node_valid=plan.node_valid,
        parent_id=tuple(remapped_ids),
        mode=plan.mode,
        mosaic_phase_centre_rad=plan.mosaic_phase_centre_rad,
        approximation=plan.approximation,
    )
    _loss_r, grad_r = _value_and_grad(
        remapped_flux, block, remapped, beam, mode="explicit_jax"
    )
    np.testing.assert_allclose(
        np.asarray(grad_r)[perm], np.asarray(grad_exp), rtol=_RTOL, atol=_ATOL
    )


def test_explicit_rejects_unknown_operator_mode() -> None:
    plan = integration_plan_from_table(_square_table())
    with pytest.raises(ValueError, match="operator_mode"):
        predict_voltage_from_plan_value_and_grad(
            np.array([1.0]),
            _block(),
            plan,
            ManufacturedVoltageBeam(intercept=_IDENTITY),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            operator_mode="autodiff",
        )


def test_explicit_104x104_one_batch_matches_vjp() -> None:
    table = starting_central_table(
        root_size=104,
        root_pixel_size_rad=_ROOT,
        mosaic_phase_centre_rad=_PHASE,
    )
    plan = integration_plan_from_table(table)
    assert plan.parent_count == 104 * 104
    block = _block()
    flux = np.full(plan.parent_count, 0.01)
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=64)
    beam = AnalyticAiryVoltageBeam()
    _value_and_grad(flux, block, plan, beam, mode="explicit_jax", config=config)
    _value_and_grad(flux, block, plan, beam, mode="vjp", config=config)
    start = time.perf_counter()
    loss_e, grad_e = _value_and_grad(
        flux, block, plan, beam, mode="explicit_jax", config=config
    )
    explicit_s = time.perf_counter() - start
    start = time.perf_counter()
    loss_v, grad_v = _value_and_grad(flux, block, plan, beam, mode="vjp", config=config)
    vjp_s = time.perf_counter() - start
    np.testing.assert_allclose(float(loss_e), float(loss_v), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(grad_e), np.asarray(grad_v), rtol=1e-7, atol=1e-9
    )
    assert np.isfinite(float(loss_e))
    print(f"explicit={explicit_s:.3f}s vjp={vjp_s:.3f}s", flush=True)


def test_explicit_kernel_is_reused_across_calls() -> None:
    plan = integration_plan_from_table(_square_table())
    block = _block()
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    kwargs = {
        "block": block,
        "plan": plan,
        "beam": beam,
        "antenna_position_m": _ANTENNA_POSITION_M,
        "calibration_state": "casa_parang_true",
        "config": BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
    }
    clear_explicit_kernels()
    compiled = []
    for flux in (np.array([1.2]), np.array([0.4]), np.array([2.0])):
        loss, gradient = predict_voltage_from_plan_value_and_grad_explicit_jax(
            flux, **kwargs
        )
        jax.block_until_ready(loss)
        jax.block_until_ready(gradient)
        compiled.append(next(iter(_EXPLICIT_KERNELS.values())))
    assert explicit_kernel_build_count() == 1
    assert len(_EXPLICIT_KERNELS) == 1
    assert compiled[0] is compiled[1] is compiled[2]
    other = ManufacturedVoltageBeam(intercept=_NONTRIVIAL)
    loss, gradient = predict_voltage_from_plan_value_and_grad_explicit_jax(
        np.array([1.2]),
        block=block,
        plan=plan,
        beam=other,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
    )
    jax.block_until_ready(loss)
    jax.block_until_ready(gradient)
    assert explicit_kernel_build_count() == 2


def test_explicit_adjoint_workspace_is_parent_sized() -> None:
    parent_count = 62_431
    pixel_chunk = 512
    n_channel = 64
    workspace = explicit_adjoint_workspace_bytes(
        parent_count=parent_count, pixel_chunk_size=pixel_chunk
    )
    old_complex_accumulator = parent_count * n_channel * 64
    assert workspace == (parent_count + pixel_chunk) * 8
    assert workspace < old_complex_accumulator / 100


def test_explicit_airy_diameter_does_not_reuse_kernel() -> None:
    plan = integration_plan_from_table(_square_table())
    block = _block()
    flux = np.array([1.2])
    wide = AnalyticAiryVoltageBeam(catalog=VLABeamCatalog(dish_diameter_m=25.0))
    narrow = AnalyticAiryVoltageBeam(catalog=VLABeamCatalog(dish_diameter_m=20.0))
    assert _beam_cache_key(wide) != _beam_cache_key(narrow)
    clear_explicit_kernels()
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2)
    _value_and_grad(flux, block, plan, wide, mode="explicit_jax", config=config)
    loss_n, grad_n = _value_and_grad(
        flux, block, plan, narrow, mode="explicit_jax", config=config
    )
    loss_v, grad_v = _value_and_grad(
        flux, block, plan, narrow, mode="vjp", config=config
    )
    assert explicit_kernel_build_count() == 2
    np.testing.assert_allclose(float(loss_n), float(loss_v), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grad_n), np.asarray(grad_v), rtol=_RTOL, atol=_ATOL
    )


def test_explicit_rejects_materialize_and_jones_cap() -> None:
    plan = integration_plan_from_table(_square_table())
    block = _block()
    flux = np.array([1.2])
    beam = ManufacturedVoltageBeam(intercept=_IDENTITY)
    with pytest.raises(ValueError, match="STREAM"):
        predict_voltage_from_plan_value_and_grad_explicit_jax(
            flux,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(policy=BeamOperatorPolicy.MATERIALIZE),
        )
    with pytest.raises(ValueError, match="Jones tile"):
        predict_voltage_from_plan_value_and_grad_explicit_jax(
            flux,
            block,
            plan,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(max_timestep_jones_bytes=1),
        )


def test_explicit_kernel_cache_evicts_oldest() -> None:
    clear_explicit_kernels()
    limit = EXPLICIT_KERNEL_CACHE_LIMIT
    for index in range(limit + 2):
        _store_explicit_kernel((index,), object())
    assert explicit_cached_kernel_count() == limit
    assert _lookup_explicit_kernel((0,)) is None
    assert _lookup_explicit_kernel((1,)) is None
    assert _lookup_explicit_kernel((2,)) is not None
    assert len(_EXPLICIT_KERNELS) == limit
