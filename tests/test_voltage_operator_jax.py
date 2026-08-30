from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam
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
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_operator_jax import (
    off_diagonal_support_mask_jax,
    predict_voltage_beam_jax,
    predict_voltage_beam_jax_value_and_grad,
)

_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)
_CORRELATIONS = (
    Correlation.RR,
    Correlation.RL,
    Correlation.LR,
    Correlation.LL,
)


def _block(
    *,
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
) -> VisibilityBlock:
    rows = time_s.size
    dummy = np.zeros((rows, frequency_hz.size, 4), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=np.array(
            [
                [13.0, -27.0, 4.0],
                [-19.0, 7.0, 3.0],
                [41.0, 17.0, -8.0],
                [-5.0, -11.0, 9.0],
            ],
            dtype=np.float64,
        )[:rows],
        frequency_hz=frequency_hz,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 0, 1, 2], dtype=np.int32)[:rows],
        antenna2=np.array([1, 2, 3, 3], dtype=np.int32)[:rows],
        correlations=_CORRELATIONS,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(0.3, 0.1),
    )


def _cband_problem() -> tuple[VisibilityBlock, np.ndarray, np.ndarray, np.ndarray]:
    l = np.array([0.0, np.sin(np.deg2rad(0.04))])
    m = np.zeros(2)
    intensity = np.array([1.2, 0.4])
    frequency = np.array([4.536e9, 4.662e9])
    time_s = np.array([5.0e9, 5.0e9, 5.0e9 + 3600.0, 5.0e9 + 3600.0])
    return _block(frequency_hz=frequency, time_s=time_s), l, m, intensity


def _compare_to_numpy(beam, *, calibration_state: str = "casa_parang_true") -> None:
    block, l, m, intensity = _cband_problem()
    sky = SkyStokesPlanes(stokes_i=intensity)
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    kwargs = dict(
        block=block,
        l_rad=l,
        m_rad=m,
        sky=sky,
        beam=beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state=calibration_state,
        config=config,
    )
    reference = predict_voltage_beam(**kwargs)
    predicted = predict_voltage_beam_jax(**kwargs)
    np.testing.assert_allclose(
        predicted.visibility, reference.visibility, rtol=1e-9, atol=1e-11
    )
    np.testing.assert_array_equal(predicted.valid, reference.valid)
    assert predicted.off_diagonal_valid is not None
    assert reference.off_diagonal_valid is not None
    np.testing.assert_array_equal(
        predicted.off_diagonal_valid, reference.off_diagonal_valid
    )


def test_jax_airy_matches_numpy_and_explicit_stokes_i() -> None:
    block, l, m, intensity = _cband_problem()
    predicted = predict_voltage_beam_jax(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    reference = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    weights = VLAPrimaryBeam(kind="airy").power_weights(l, m, block.frequency_hz)
    expected = np.asarray(
        predict_stokes_i_explicit(
            intensity,
            l,
            m,
            block.uvw_m,
            block.frequency_hz,
            block.antenna1,
            block.antenna2,
            block.correlations,
            config=DirectDFTConfig(visibility_chunk_size=3, pixel_chunk_size=2),
            beam_weights=weights,
        )
    )
    np.testing.assert_allclose(
        predicted.visibility, reference.visibility, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(predicted.visibility, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(predicted.valid, reference.valid)


def test_jax_composite_matches_numpy() -> None:
    _compare_to_numpy(voltage_beam_for_mode("streamed_scalar"))


def test_jax_cassbeam_diagonal_matches_numpy() -> None:
    _compare_to_numpy(voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR))


def test_jax_cassbeam_full_jones_matches_numpy_uncalibrated() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact,
        off_diagonal=True,
        allow_unfrozen=True,
        outer=voltage_beam_for_mode("streamed_scalar"),
    )
    _compare_to_numpy(beam, calibration_state="uncalibrated")


def test_full_jones_factory_still_refuses_unfrozen() -> None:
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode(BeamImagingMode.FULL_JONES)


def test_diagonal_beams_mark_known_zero_leakage_valid() -> None:
    block, l, m, intensity = _cband_problem()
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    beams = (
        AnalyticAiryVoltageBeam(),
        voltage_beam_for_mode("streamed_scalar"),
        voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR),
    )
    for beam in beams:
        result = predict_voltage_beam_jax(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=intensity),
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=config,
        )
        assert result.off_diagonal_valid is not None
        np.testing.assert_array_equal(result.off_diagonal_valid, result.valid)
        assert np.any(result.off_diagonal_valid)


def test_airy_loss_includes_known_zero_cross_hands() -> None:
    block, l, m, intensity = _cband_problem()
    beam = AnalyticAiryVoltageBeam()
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    result = predict_voltage_beam_jax(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
    )
    observed = np.array(result.visibility, copy=True)
    rl = block.correlations.index(Correlation.RL)
    clean = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=observed,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    corrupted = np.array(observed, copy=True)
    corrupted[..., rl] = 10.0
    masked = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=corrupted,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    values = jnp.asarray(intensity, dtype=jnp.float64)
    kwargs = dict(
        l_rad=l,
        m_rad=m,
        beam=beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
    )
    loss_clean, _ = predict_voltage_beam_jax_value_and_grad(values, clean, **kwargs)
    loss_corrupt, _ = predict_voltage_beam_jax_value_and_grad(values, masked, **kwargs)
    assert float(loss_clean) == pytest.approx(0.0, abs=1e-12)
    assert float(loss_corrupt) > 1.0


def test_jax_off_diagonal_support_mask_matches_numpy() -> None:
    from sl1mjax.voltage_beam import beam_coordinates

    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact,
        off_diagonal=True,
        allow_unfrozen=True,
        outer=voltage_beam_for_mode("streamed_scalar"),
    )
    l = np.array([0.0, np.sin(np.deg2rad(0.04)), float(artifact.tables[0].l_rad[-1]) + 0.005])
    m = np.zeros(3)
    frequency = np.array([4.564e9, 4.692e9])
    chi = np.array([0.1, -0.2, 0.0, 0.3])
    jax_mask = off_diagonal_support_mask_jax(
        beam,
        l,
        m,
        frequency,
        chi,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(pixel_chunk_size=2),
    )
    numpy_eval = beam.evaluate(
        beam_coordinates(l, m, frequency, parallactic_angle_rad=chi, antenna_id=np.arange(4)),
        calibration_state="casa_parang_true",
    )
    numpy_mask = np.all(numpy_eval.off_diagonal_valid, axis=(0, 2))
    np.testing.assert_array_equal(jax_mask, numpy_mask)


def test_jax_stokes_i_value_and_grad_matches_finite_difference() -> None:
    block, l, m, intensity = _cband_problem()
    beam = AnalyticAiryVoltageBeam()
    values = jnp.asarray(intensity, dtype=jnp.float64)
    loss, gradient = predict_voltage_beam_jax_value_and_grad(
        values,
        block,
        l,
        m,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert np.isfinite(float(loss))
    numeric = np.empty(intensity.size, dtype=np.float64)
    step = 1.0e-6
    for index in range(intensity.size):
        perturbed = np.array(intensity, dtype=np.float64)
        perturbed[index] += step
        plus, _ = predict_voltage_beam_jax_value_and_grad(
            jnp.asarray(perturbed),
            block,
            l,
            m,
            beam,
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
        )
        numeric[index] = (float(plus) - float(loss)) / step
    np.testing.assert_allclose(np.asarray(gradient), numeric, rtol=2e-4, atol=2e-6)


def test_jax_accepts_per_channel_stokes_i() -> None:
    block, l, m, intensity = _cband_problem()
    spectral = np.stack((intensity, 0.5 * intensity), axis=1)
    sky = SkyStokesPlanes(stokes_i=spectral)
    config = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    predicted = predict_voltage_beam_jax(
        block,
        l,
        m,
        sky,
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
    )
    reference = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=config,
    )
    np.testing.assert_allclose(
        predicted.visibility, reference.visibility, rtol=1e-12, atol=1e-14
    )
    assert predicted.visibility.shape[1] == block.frequency_hz.size


def test_jax_rejects_mis_shaped_stokes_planes() -> None:
    block, l, m, intensity = _cband_problem()
    with pytest.raises(ValueError, match="stokes_i"):
        predict_voltage_beam_jax(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=np.ones((intensity.size, 3))),
            AnalyticAiryVoltageBeam(),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
        )
    with pytest.raises(ValueError, match="stokes_q"):
        predict_voltage_beam_jax(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=intensity, stokes_q=np.ones(3)),
            AnalyticAiryVoltageBeam(),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
        )


def test_jax_enforces_jones_memory_and_stream_policy() -> None:
    block, l, m, intensity = _cband_problem()
    kwargs = dict(
        block=block,
        l_rad=l,
        m_rad=m,
        sky=SkyStokesPlanes(stokes_i=intensity),
        beam=AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    with pytest.raises(ValueError, match="Jones tile"):
        predict_voltage_beam_jax(
            **kwargs,
            config=BeamOperatorConfig(max_timestep_jones_bytes=1),
        )
    with pytest.raises(ValueError, match="STREAM"):
        predict_voltage_beam_jax(
            **kwargs,
            config=BeamOperatorConfig(policy=BeamOperatorPolicy.MATERIALIZE),
        )


def test_jax_train_mask_excludes_holdout_from_the_objective() -> None:
    block, l, m, intensity = _cband_problem()
    beam = AnalyticAiryVoltageBeam()
    prediction = predict_voltage_beam_jax(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    observed = np.array(prediction.visibility, copy=True)
    observed[0] *= 3.0
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=observed,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    train_mask = np.ones(block.visibility.shape, dtype=bool)
    train_mask[0] = False
    values = jnp.asarray(intensity, dtype=jnp.float64)
    kwargs = dict(
        block=block,
        l_rad=l,
        m_rad=m,
        beam=beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    loss_all, _ = predict_voltage_beam_jax_value_and_grad(values, **kwargs)
    loss_train, _ = predict_voltage_beam_jax_value_and_grad(
        values, **kwargs, train_mask=train_mask
    )
    assert float(loss_all) > float(loss_train)
    assert float(loss_train) == pytest.approx(0.0, abs=1e-12)


def test_full_jones_loss_ignores_unsupported_leakage() -> None:
    artifact = load_cassbeam_cband_artifact()
    diagonal = voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR)
    beam = CassbeamCBandVoltageBeam(
        artifact,
        off_diagonal=True,
        allow_unfrozen=True,
        outer=diagonal.outer,
    )
    table = artifact.tables[0]
    # Past the CASSBEAM raster for every antenna frame, still inside the
    # extended C-band Airy outer used by diagonal_copolar.
    l = np.array([float(table.l_rad[-1]) + 0.005])
    m = np.zeros(1)
    intensity = np.array([0.5])
    frequency = np.array([4.564e9])
    block = _block(frequency_hz=frequency, time_s=np.array([5.0e9, 5.0e9 + 60.0]))
    result = predict_voltage_beam_jax(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity, stokes_q=np.array([0.1])),
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    assert result.off_diagonal_valid is not None
    outside = ~result.off_diagonal_valid
    if not np.any(outside):
        pytest.skip("fixture directions remain inside the CASSBEAM raster")
    observed = np.array(result.visibility, copy=True)
    rl = block.correlations.index(Correlation.RL)
    clean = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=observed,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    observed = np.array(observed, copy=True)
    observed[outside, rl] = 10.0
    masked = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=observed,
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    values = jnp.asarray(intensity, dtype=jnp.float64)
    kwargs = dict(
        l_rad=l,
        m_rad=m,
        beam=beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    loss_clean, _ = predict_voltage_beam_jax_value_and_grad(
        values, clean, **kwargs
    )
    loss_corrupt, _ = predict_voltage_beam_jax_value_and_grad(
        values, masked, **kwargs
    )
    assert np.isfinite(float(loss_clean))
    assert float(loss_corrupt) == pytest.approx(float(loss_clean))
