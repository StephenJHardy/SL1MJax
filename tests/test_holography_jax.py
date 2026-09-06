"""JAX holography pointing: NumPy parity, adjoint, and compile invariance."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.beam_aware_imaging import VoltageIntegrationMode, sky_table_from_records
from sl1mjax.beam_operator import BeamOperatorConfig, SkyStokesPlanes, predict_voltage_beam
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import (
    ManufacturedVoltageBeam,
    integration_plan_from_table,
    predict_voltage_from_plan_value_and_grad,
)
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_operator_jax import (
    clear_explicit_kernels,
    explicit_kernel_build_count,
    predict_voltage_beam_jax,
    predict_voltage_beam_jax_value_and_grad,
)

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))
_HOLO_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)
_HOLO_JONES = np.array(
    [[1.0 + 0.0j, 0.08 - 0.02j], [0.05 + 0.03j, 0.9 + 0.0j]],
    dtype=np.complex128,
)
_HOLO_GRAD = np.array(
    [[0.4 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.3 + 0.0j]],
    dtype=np.complex128,
)
_HOLO_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _holo_block(
    *,
    time_s: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    uvw_m: np.ndarray | None = None,
) -> VisibilityBlock:
    rows = time_s.size
    if uvw_m is None:
        uvw_m = np.array([[12.0, -4.0, 2.0]] * rows, dtype=np.float64)
    dummy = np.zeros((rows, 1, 4), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=np.array([4.564e9]),
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=_HOLO_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )


_RTOL = 1.0e-7
_ATOL = 1.0e-10
_DIAGONAL = np.diag([1.05, 0.95]).astype(np.complex128)
_DIAGONAL_GRAD = np.diag([0.35, -0.25]).astype(np.complex128)


def _unequal_offsets(n_time: int, n_ant: int) -> np.ndarray:
    offsets = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    offsets[0, 0] = (np.deg2rad(3.0 / 60.0), np.deg2rad(-1.0 / 60.0))
    if n_ant > 1:
        offsets[0, 1] = (0.0, 0.0)
    if n_ant > 2:
        offsets[0, 2] = (np.deg2rad(-2.0 / 60.0), np.deg2rad(0.5 / 60.0))
    if n_time > 1:
        offsets[1] = offsets[0]
        offsets[1, 0, 0] += np.deg2rad(0.4 / 60.0)
    return offsets


def _compare_numpy_jax(beam, *, rotate_times: bool = False, n_dir: int = 1) -> None:
    times = np.array([5.0e9, 5.0e9, 5.0e9 + 1800.0]) if rotate_times else np.array([0.0, 0.0, 0.0])
    block = _holo_block(
        time_s=times,
        antenna1=np.array([0, 1, 0], dtype=np.int32),
        antenna2=np.array([1, 2, 2], dtype=np.int32),
    )
    n_time = int(np.unique(times).size)
    offsets = _unequal_offsets(n_time, block.antenna_count)
    valid = np.ones((n_time, block.antenna_count), dtype=bool)
    if n_dir == 1:
        l_rad = np.array([0.0])
        m_rad = np.array([0.0])
        sky = SkyStokesPlanes(stokes_i=np.array([1.1]))
    else:
        l_rad = np.array([0.0, np.sin(np.deg2rad(0.03))])
        m_rad = np.array([0.0, -np.sin(np.deg2rad(0.01))])
        sky = SkyStokesPlanes(stokes_i=np.array([1.1, 0.4]))
    kwargs = dict(
        block=block,
        l_rad=l_rad,
        m_rad=m_rad,
        sky=sky,
        beam=beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
        pointing_valid=valid,
    )
    reference = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    )
    predicted = predict_voltage_beam_jax(
        **kwargs, config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    )
    np.testing.assert_allclose(predicted.visibility, reference.visibility, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(predicted.valid, reference.valid)
    assert predicted.off_diagonal_valid is not None
    assert reference.off_diagonal_valid is not None
    np.testing.assert_array_equal(predicted.off_diagonal_valid, reference.off_diagonal_valid)


@pytest.mark.parametrize(
    "beam",
    [
        AnalyticAiryVoltageBeam(),
        ManufacturedVoltageBeam(intercept=_DIAGONAL, grad_l=_DIAGONAL_GRAD),
        ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD),
        ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD, rotate_parallactic=True),
    ],
    ids=("airy_scalar", "diagonal", "full_jones", "parallactic"),
)
def test_jax_holography_matches_numpy(beam) -> None:
    rotate = bool(getattr(beam, "rotate_parallactic", False))
    _compare_numpy_jax(beam, rotate_times=rotate, n_dir=2 if rotate else 1)


def test_jax_common_offsets_match_mosaic_config() -> None:
    block = _holo_block(
        time_s=np.array([0.0, 0.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    delta = (np.deg2rad(2.0 / 60.0), np.deg2rad(-1.0 / 60.0))
    sky = SkyStokesPlanes(stokes_i=np.array([1.1]))
    l_rad = np.array([0.01])
    m_rad = np.array([-0.004])
    mosaic = predict_voltage_beam_jax(
        block,
        l_rad,
        m_rad,
        sky,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(pointing_offset_lm_rad=delta),
    )
    per_antenna = np.broadcast_to(np.asarray(delta), (1, 3, 2)).copy()
    holography = predict_voltage_beam_jax(
        block,
        l_rad,
        m_rad,
        sky,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=per_antenna,
    )
    np.testing.assert_allclose(mosaic.visibility, holography.visibility, rtol=_RTOL, atol=_ATOL)


def test_jax_invalid_pointing_masks_and_flags() -> None:
    block = _holo_block(
        time_s=np.array([0.0, 0.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    flag = np.zeros(block.visibility.shape, dtype=bool)
    flag[1] = True
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=block.visibility,
        weight=block.weight,
        flag=flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    offsets = np.zeros((1, 3, 2), dtype=np.float64)
    offsets[0, 0] = (np.deg2rad(3.0 / 60.0), 0.0)
    valid = np.ones((1, 3), dtype=bool)
    valid[0, 2] = False
    kwargs = dict(
        block=block,
        l_rad=np.array([0.0]),
        m_rad=np.array([0.0]),
        sky=SkyStokesPlanes(stokes_i=np.array([1.0])),
        beam=ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD),
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
        pointing_valid=valid,
        config=BeamOperatorConfig(visibility_chunk_size=4, pixel_chunk_size=1),
    )
    reference = predict_voltage_beam(**kwargs)
    predicted = predict_voltage_beam_jax(**kwargs)
    np.testing.assert_allclose(predicted.visibility, reference.visibility, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(predicted.valid, reference.valid)
    assert not bool(predicted.valid[1, 0])
    np.testing.assert_allclose(predicted.visibility[1], 0.0, atol=1e-15)
    assert bool(block.active[0, 0, 0])
    assert not bool(block.active[1, 0, 0])


def test_jax_results_invariant_across_tile_sizes() -> None:
    block = _holo_block(
        time_s=np.array([0.0, 0.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    offsets = _unequal_offsets(1, 3)
    l_rad = np.array([0.0, np.sin(np.deg2rad(0.02))])
    m_rad = np.array([0.0, 0.0])
    sky = SkyStokesPlanes(stokes_i=np.array([1.0, 0.5]))
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    results = []
    for vis_chunk, pix_chunk in ((1, 1), (2, 2), (4, 1)):
        results.append(
            predict_voltage_beam_jax(
                block,
                l_rad,
                m_rad,
                sky,
                beam,
                antenna_position_m=_HOLO_POSITIONS,
                calibration_state="casa_parang_true",
                config=BeamOperatorConfig(
                    visibility_chunk_size=vis_chunk, pixel_chunk_size=pix_chunk
                ),
                antenna_pointing_lm_rad=offsets,
            )
        )
    for other in results[1:]:
        np.testing.assert_allclose(results[0].visibility, other.visibility, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_array_equal(results[0].valid, other.valid)


def test_jax_value_and_grad_matches_finite_difference() -> None:
    block = _holo_block(
        time_s=np.array([0.0, 0.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
        uvw_m=np.array([[20.0, -8.0, 3.0], [-11.0, 15.0, -2.0]]),
    )
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=np.full(block.visibility.shape, 0.3 + 0.05j),
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    intensity = np.array([1.2, 0.4])
    l_rad = np.array([0.0, np.sin(np.deg2rad(0.02))])
    m_rad = np.zeros(2)
    offsets = _unequal_offsets(1, 3)
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    values = jnp.asarray(intensity, dtype=jnp.float64)
    loss, gradient = predict_voltage_beam_jax_value_and_grad(
        values,
        block,
        l_rad,
        m_rad,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
    )
    numeric = np.empty(intensity.size, dtype=np.float64)
    step = 1.0e-6
    for index in range(intensity.size):
        perturbed = np.array(intensity, dtype=np.float64)
        perturbed[index] += step
        plus, _ = predict_voltage_beam_jax_value_and_grad(
            jnp.asarray(perturbed),
            block,
            l_rad,
            m_rad,
            beam,
            antenna_position_m=_HOLO_POSITIONS,
            calibration_state="casa_parang_true",
            antenna_pointing_lm_rad=offsets,
        )
        numeric[index] = (float(plus) - float(loss)) / step
    np.testing.assert_allclose(np.asarray(gradient), numeric, rtol=2e-4, atol=2e-6)


def _square_plan():
    return integration_plan_from_table(
        sky_table_from_records(
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
                }
            ],
            mosaic_phase_centre_rad=_PHASE,
        ),
        mode=VoltageIntegrationMode.SUBCELL_2X2,
    )


def test_explicit_adjoint_matches_vjp_with_holography_pointing() -> None:
    block = _holo_block(
        time_s=np.array([5.0e9, 5.0e9 + 900.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
        uvw_m=np.array([[40.0, -25.0, 2.0], [-18.0, 30.0, -1.0]]),
    )
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=np.full(block.visibility.shape, 0.2 + 0.05j),
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    offsets = _unequal_offsets(2, 3)
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    plan = _square_plan()
    flux = np.array([1.3])
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2)
    kwargs = dict(
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        config=config,
        antenna_pointing_lm_rad=offsets,
    )
    loss_vjp, grad_vjp = predict_voltage_from_plan_value_and_grad(
        flux, block, plan, beam, operator_mode="vjp", **kwargs
    )
    loss_exp, grad_exp = predict_voltage_from_plan_value_and_grad(
        flux, block, plan, beam, operator_mode="explicit_jax", **kwargs
    )
    np.testing.assert_allclose(float(loss_exp), float(loss_vjp), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(grad_exp), np.asarray(grad_vjp), rtol=1e-8, atol=1e-10)


def test_explicit_kernel_does_not_recompile_when_pointing_values_change() -> None:
    block = _holo_block(
        time_s=np.array([5.0e9, 5.0e9 + 900.0]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=np.full(block.visibility.shape, 0.2 + 0.05j),
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    plan = _square_plan()
    first = _unequal_offsets(2, 3)
    second = first.copy()
    second[0, 0, 0] += np.deg2rad(1.0 / 60.0)
    clear_explicit_kernels()
    kwargs = dict(
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2),
    )
    predict_voltage_from_plan_value_and_grad(
        np.array([1.0]),
        block,
        plan,
        beam,
        operator_mode="explicit_jax",
        antenna_pointing_lm_rad=first,
        **kwargs,
    )
    jax.block_until_ready(
        predict_voltage_from_plan_value_and_grad(
            np.array([1.0]),
            block,
            plan,
            beam,
            operator_mode="explicit_jax",
            antenna_pointing_lm_rad=first,
            **kwargs,
        )[0]
    )
    builds = explicit_kernel_build_count()
    assert builds >= 1
    predict_voltage_from_plan_value_and_grad(
        np.array([1.1]),
        block,
        plan,
        beam,
        operator_mode="explicit_jax",
        antenna_pointing_lm_rad=second,
        **kwargs,
    )
    assert explicit_kernel_build_count() == builds


def test_jax_refuses_config_and_antenna_pointing_together() -> None:
    block = _holo_block(
        time_s=np.array([0.0]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="do not pass both"):
        predict_voltage_beam_jax(
            block,
            np.array([0.0]),
            np.array([0.0]),
            SkyStokesPlanes(stokes_i=np.array([1.0])),
            ManufacturedVoltageBeam(intercept=_HOLO_JONES),
            antenna_position_m=_HOLO_POSITIONS,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(pointing_offset_lm_rad=(0.001, 0.0)),
            antenna_pointing_lm_rad=np.zeros((1, 2, 2)),
        )


def test_observation_jax_matches_committed_rime_golden() -> None:
    from sl1mjax.holography import load_synthetic_holography_rime_case

    observation, beam, expected = load_synthetic_holography_rime_case()
    numpy_result = observation.predict(beam, backend="numpy", moving_reference_only=True)
    jax_result = observation.predict(beam, backend="jax", moving_reference_only=True)
    np.testing.assert_allclose(
        numpy_result.visibility, jax_result.visibility, rtol=1e-7, atol=1e-10
    )
    np.testing.assert_array_equal(
        numpy_result.moving_reference_rows, jax_result.moving_reference_rows
    )
    np.testing.assert_allclose(
        jax_result.visibility[numpy_result.moving_reference_rows],
        expected[numpy_result.moving_reference_rows],
        rtol=1e-7,
        atol=1e-10,
    )
    assert jax_result.provenance["backend"] == "jax"
    assert not bool(jax_result.moving_reference_rows[1])
    np.testing.assert_allclose(jax_result.visibility[1], 0.0, atol=1e-15)


def test_resolved_source_jax_matches_numpy_and_ignores_uvw() -> None:
    from sl1mjax.holography import (
        load_synthetic_holography_rime_case,
        predict_holography_visibilities,
    )
    from sl1mjax.polarization import circular_stokes_to_coherency

    observation, beam, _expected = load_synthetic_holography_rime_case()
    source = np.zeros((observation.block.visibility.shape[0], 1, 2, 2), dtype=np.complex128)
    source[:] = circular_stokes_to_coherency(1.1, 0.0, 0.0, 0.0)
    source[0] *= 1.4
    kwargs = dict(
        block=observation.block,
        pointing=observation.pointing,
        beam=beam,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        source_coherency_visibility=source,
    )
    numpy_result = predict_holography_visibilities(**kwargs, backend="numpy")
    jax_result = predict_holography_visibilities(**kwargs, backend="jax")
    np.testing.assert_allclose(
        jax_result.visibility, numpy_result.visibility, rtol=1e-7, atol=1e-10
    )
    zero_uvw = VisibilityBlock(
        uvw_m=np.zeros_like(observation.block.uvw_m),
        frequency_hz=observation.block.frequency_hz,
        visibility=observation.block.visibility,
        weight=observation.block.weight,
        flag=observation.block.flag,
        time_s=observation.block.time_s,
        antenna1=observation.block.antenna1,
        antenna2=observation.block.antenna2,
        correlations=observation.block.correlations,
        receptor_basis=observation.block.receptor_basis,
        phase_centre_rad=observation.block.phase_centre_rad,
    )
    zeroed = predict_holography_visibilities(**{**kwargs, "block": zero_uvw}, backend="jax")
    np.testing.assert_allclose(zeroed.visibility, jax_result.visibility, rtol=1e-7, atol=1e-10)


def test_three_c147_source_model_jax_matches_numpy() -> None:
    from sl1mjax.holography import (
        HolographyObservation,
        load_synthetic_holography_rime_case,
        three_c147_point_source_model,
    )

    observation, beam, _expected = load_synthetic_holography_rime_case()
    bound = HolographyObservation(
        block=observation.block,
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        source_model=three_c147_point_source_model(observation.block.frequency_hz),
    )
    numpy_result = bound.predict(beam, backend="numpy")
    jax_result = bound.predict(beam, backend="jax")
    np.testing.assert_allclose(
        jax_result.visibility, numpy_result.visibility, rtol=1e-7, atol=1e-10
    )


def test_on_axis_jones_jax_matches_numpy_identity() -> None:
    from sl1mjax.holography import (
        evaluate_holography_on_axis_jones,
        load_synthetic_holography_rime_case,
    )

    observation, _beam, _expected = load_synthetic_holography_rime_case()
    identity = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128))
    numpy_jones = evaluate_holography_on_axis_jones(observation, identity, backend="numpy")
    jax_jones = evaluate_holography_on_axis_jones(observation, identity, backend="jax")
    assert numpy_jones.status == "pass"
    assert jax_jones.status == "pass"
    np.testing.assert_allclose(jax_jones.jones, numpy_jones.jones, rtol=1e-7, atol=1e-10)
