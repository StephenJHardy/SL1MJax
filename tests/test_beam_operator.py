from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    BeamOperatorPolicy,
    SkyStokesPlanes,
    adjoint_voltage_beam,
    predict_voltage_beam,
    timestep_jones_bytes,
    unique_visibility_times,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig, predict_stokes_i_explicit
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    circular_stokes_from_correlations,
)
from sl1mjax.voltage_beam import (
    AnalyticAiryVoltageBeam,
    BeamEvaluation,
    CompositeHandoverPolicy,
    CompositeScalarVoltageBeam,
    DiagonalSquintVoltageBeam,
    Perley2016CBandVoltageBeam,
    beam_coordinates,
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


@dataclass(frozen=True)
class _ConstantJonesBeam:
    jones: np.ndarray
    model_id: str = "test_constant_jones"

    def evaluate(self, coordinates, *, calibration_state: str) -> BeamEvaluation:
        del calibration_state
        n_direction = coordinates.l_rad.size
        n_channel = coordinates.frequency_hz.size
        matrix = np.broadcast_to(
            np.asarray(self.jones, dtype=np.complex128),
            (1, n_direction, n_channel, 2, 2),
        ).copy()
        return BeamEvaluation(
            jones=matrix,
            valid=np.ones((1, n_direction, n_channel), dtype=bool),
            provenance={"model_id": self.model_id},
        )


@dataclass(frozen=True)
class _TiledAiryBeam:
    inner: AnalyticAiryVoltageBeam
    model_id: str = "tiled_analytic_blocked_airy"

    def evaluate(self, coordinates, *, calibration_state: str) -> BeamEvaluation:
        inner = self.inner.evaluate(coordinates, calibration_state=calibration_state)
        if coordinates.antenna_id is None:
            return inner
        n_antenna = coordinates.antenna_id.size
        return BeamEvaluation(
            jones=np.broadcast_to(
                inner.jones, (n_antenna, *inner.jones.shape[1:])
            ).copy(),
            valid=np.broadcast_to(
                inner.valid, (n_antenna, *inner.valid.shape[1:])
            ).copy(),
            provenance=inner.provenance,
        )


def _block(
    *,
    l: np.ndarray,
    m: np.ndarray,
    intensity: np.ndarray,
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
    uvw_m: np.ndarray | None = None,
    antenna1: np.ndarray | None = None,
    antenna2: np.ndarray | None = None,
    phase_centre_rad: tuple[float, float] = (0.3, 0.1),
) -> VisibilityBlock:
    rows = time_s.size
    if uvw_m is None:
        uvw_m = np.array(
            [
                [13.0, -27.0, 4.0],
                [-19.0, 7.0, 3.0],
                [41.0, 17.0, -8.0],
                [-5.0, -11.0, 9.0],
            ],
            dtype=np.float64,
        )[:rows]
    if antenna1 is None:
        antenna1 = np.array([0, 0, 1, 2], dtype=np.int32)[:rows]
    if antenna2 is None:
        antenna2 = np.array([1, 2, 3, 3], dtype=np.int32)[:rows]
    dummy = np.zeros((rows, frequency_hz.size, 4), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=_CORRELATIONS,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=phase_centre_rad,
    )


def _cband_problem() -> tuple[VisibilityBlock, np.ndarray, np.ndarray, np.ndarray]:
    l = np.array([0.0, np.sin(np.deg2rad(0.04))])
    m = np.zeros(2)
    intensity = np.array([1.2, 0.4])
    frequency = np.array([4.536e9, 4.662e9])
    time_s = np.array([5.0e9, 5.0e9, 5.0e9 + 3600.0, 5.0e9 + 3600.0])
    block = _block(
        l=l, m=m, intensity=intensity, frequency_hz=frequency, time_s=time_s
    )
    return block, l, m, intensity


def test_unique_times_are_exact_and_sorted() -> None:
    unique, inverse = unique_visibility_times([3.0, 1.0, 1.0, 3.0])
    np.testing.assert_array_equal(unique, [1.0, 3.0])
    np.testing.assert_array_equal(inverse, [1, 0, 0, 1])


def test_stream_and_materialize_match_for_airy_i() -> None:
    block, l, m, intensity = _cband_problem()
    sky = SkyStokesPlanes(stokes_i=intensity)
    beam = AnalyticAiryVoltageBeam()
    kwargs = dict(
        block=block,
        l_rad=l,
        m_rad=m,
        sky=sky,
        beam=beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    streamed = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(policy=BeamOperatorPolicy.STREAM)
    )
    materialized = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(policy=BeamOperatorPolicy.MATERIALIZE)
    )
    retained = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(policy=BeamOperatorPolicy.RETAIN_LAST)
    )
    np.testing.assert_allclose(streamed.visibility, materialized.visibility)
    np.testing.assert_allclose(streamed.visibility, retained.visibility)
    np.testing.assert_array_equal(streamed.valid, materialized.valid)
    assert streamed.last_evaluation is None
    assert retained.last_evaluation is not None
    assert materialized.materialized is not None
    assert len(materialized.materialized) == 2
    assert streamed.provenance["creates_cache"] is False
    assert streamed.provenance["parallactic_angle_rad_shape"] == [2, 4]


def test_chunk_size_does_not_change_the_result() -> None:
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
    full = predict_voltage_beam(**kwargs)
    chunked = predict_voltage_beam(
        **kwargs,
        config=BeamOperatorConfig(visibility_chunk_size=1, pixel_chunk_size=1),
    )
    np.testing.assert_allclose(full.visibility, chunked.visibility)


def test_airy_operator_matches_explicit_stokes_i_power_weights() -> None:
    block, l, m, intensity = _cband_problem()
    predicted = predict_voltage_beam(
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
    np.testing.assert_allclose(predicted.visibility, expected, rtol=1e-12, atol=1e-14)
    assert np.all(predicted.valid)


def test_on_axis_unpolarised_source_stays_unpolarised() -> None:
    l = np.array([0.0])
    m = np.array([0.0])
    intensity = np.array([2.0])
    block = _block(
        l=l,
        m=m,
        intensity=intensity,
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9, 5.0e9]),
        uvw_m=np.array([[10.0, -4.0, 1.0], [-8.0, 12.0, 2.0]]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        predicted.visibility, block.correlations
    )
    np.testing.assert_allclose(np.real(stokes_q), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.real(stokes_u), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.real(stokes_v), 0.0, atol=1e-14)
    assert np.all(np.real(stokes_i) > 0.0)


def test_off_diagonal_jones_creates_i_to_qu() -> None:
    l = np.array([0.0])
    m = np.array([0.0])
    intensity = np.array([1.0])
    block = _block(
        l=l,
        m=m,
        intensity=intensity,
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9]),
        uvw_m=np.array([[0.0, 0.0, 0.0]]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    leakage = 0.1
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        _ConstantJonesBeam(np.array([[1.0, leakage], [leakage, 1.0]])),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    _, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        predicted.visibility, block.correlations
    )
    assert abs(float(np.real(stokes_q[0, 0]))) > 0.05
    np.testing.assert_allclose(np.real(stokes_u), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.real(stokes_v), 0.0, atol=1e-14)


def test_adjoint_matches_the_real_inner_product() -> None:
    block, l, m, intensity = _cband_problem()
    sky = SkyStokesPlanes(
        stokes_i=intensity,
        stokes_q=np.array([0.05, -0.02]),
        stokes_u=np.array([-0.03, 0.04]),
        stokes_v=np.array([0.01, -0.015]),
    )
    beam = AnalyticAiryVoltageBeam()
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    rng = np.random.default_rng(4)
    residual = rng.normal(size=predicted.visibility.shape) + 1j * rng.normal(
        size=predicted.visibility.shape
    )
    gradient = adjoint_voltage_beam(
        residual,
        block,
        l,
        m,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    left = float(np.sum(predicted.visibility * np.conjugate(residual)).real)
    right = float(
        np.sum(sky.stokes_i[:, None] * gradient[0])
        + np.sum(sky.stokes_q[:, None] * gradient[1])
        + np.sum(sky.stokes_u[:, None] * gradient[2])
        + np.sum(sky.stokes_v[:, None] * gradient[3])
    )
    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


def test_array_average_and_identical_antenna_specific_paths_agree() -> None:
    block, l, m, intensity = _cband_problem()
    sky = SkyStokesPlanes(stokes_i=intensity)
    average = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    tiled = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        _TiledAiryBeam(AnalyticAiryVoltageBeam()),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    np.testing.assert_allclose(average.visibility, tiled.visibility)


def test_perley_out_of_band_rows_are_invalid() -> None:
    l = np.array([0.0])
    m = np.array([0.0])
    block = _block(
        l=l,
        m=m,
        intensity=np.array([1.0]),
        frequency_hz=np.array([1.4e9]),
        time_s=np.array([5.0e9]),
        uvw_m=np.array([[10.0, 0.0, 0.0]]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=np.array([1.0])),
        Perley2016CBandVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert not bool(predicted.valid[0, 0])
    np.testing.assert_allclose(predicted.visibility, 0.0)


def test_unknown_calibration_state_is_rejected() -> None:
    block, l, m, intensity = _cband_problem()
    with pytest.raises(ValueError, match="refuse to guess Jones order"):
        predict_voltage_beam(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=intensity),
            AnalyticAiryVoltageBeam(),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="maybe_parang",
        )


def test_materialize_budget_fails_closed() -> None:
    block, l, m, intensity = _cband_problem()
    slice_bytes = timestep_jones_bytes(1, l.size, block.frequency_hz.size)
    with pytest.raises(ValueError, match="materialized Jones slices"):
        predict_voltage_beam(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=intensity),
            AnalyticAiryVoltageBeam(),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(
                policy=BeamOperatorPolicy.MATERIALIZE,
                max_timestep_jones_bytes=slice_bytes,
            ),
        )


def test_missing_correlation_is_zero_in_the_adjoint() -> None:
    block, l, m, intensity = _cband_problem()
    slim = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=block.visibility[..., :2],
        weight=block.weight[..., :2],
        flag=block.flag[..., :2],
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=block.phase_centre_rad,
    )
    predicted = predict_voltage_beam(
        slim,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert predicted.visibility.shape[-1] == 2
    residual = np.ones_like(predicted.visibility)
    gradient = adjoint_voltage_beam(
        residual,
        slim,
        l,
        m,
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    left = float(np.sum(predicted.visibility * np.conjugate(residual)).real)
    right = float(np.sum(intensity[:, None] * gradient[0]))
    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


def test_operator_does_not_change_the_evaluator_contract() -> None:
    coordinates = beam_coordinates([0.0], [0.0], [4.6e9], parallactic_angle_rad=0.2)
    evaluation = AnalyticAiryVoltageBeam().evaluate(
        coordinates, calibration_state="casa_parang_true"
    )
    assert evaluation.jones.shape == (1, 1, 1, 2, 2)
    assert evaluation.provenance["ignored_coordinates"] == [
        "antenna_id",
        "parallactic_angle_rad",
        "elevation_rad",
    ]
    assert Receptor.R.value == "R"


def test_diagonal_squint_operator_makes_v_not_qu() -> None:
    l = np.array([np.sin(np.deg2rad(0.04))])
    m = np.array([0.0])
    intensity = np.array([1.0])
    block = _block(
        l=l,
        m=m,
        intensity=intensity,
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9, 5.0e9]),
        uvw_m=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam()),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        predicted.visibility, block.correlations
    )
    np.testing.assert_allclose(np.real(stokes_q), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.real(stokes_u), 0.0, atol=1e-14)
    assert np.any(np.abs(np.real(stokes_v)) > 1e-6)
    assert np.all(np.real(stokes_i) > 0.0)
    unsquinted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    _, _, _, unsquinted_v = circular_stokes_from_correlations(
        unsquinted.visibility, block.correlations
    )
    np.testing.assert_allclose(np.real(unsquinted_v), 0.0, atol=1e-14)


def test_diagonal_squint_stream_matches_materialize() -> None:
    block, l, m, intensity = _cband_problem()
    kwargs = dict(
        block=block,
        l_rad=l,
        m_rad=m,
        sky=SkyStokesPlanes(stokes_i=intensity),
        beam=DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam()),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    streamed = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(policy=BeamOperatorPolicy.STREAM)
    )
    materialized = predict_voltage_beam(
        **kwargs, config=BeamOperatorConfig(policy=BeamOperatorPolicy.MATERIALIZE)
    )
    np.testing.assert_allclose(streamed.visibility, materialized.visibility)
    assert materialized.materialized is not None
    assert materialized.materialized[0].jones.shape[0] == block.antenna_count


def test_unsupported_direction_does_not_zero_a_valid_pixel() -> None:
    l = np.array([0.0, np.sin(np.deg2rad(1.0))])
    m = np.zeros(2)
    intensity = np.array([1.0, 0.0])
    block = _block(
        l=l,
        m=m,
        intensity=intensity,
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9]),
        uvw_m=np.array([[10.0, -4.0, 1.0]]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    sky = SkyStokesPlanes(stokes_i=intensity)
    both = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        Perley2016CBandVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    on_axis = predict_voltage_beam(
        block,
        l[:1],
        m[:1],
        SkyStokesPlanes(stokes_i=intensity[:1]),
        Perley2016CBandVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(both.visibility, on_axis.visibility)
    assert bool(both.valid[0, 0])


def test_squint_streams_when_the_full_antenna_slice_exceeds_budget() -> None:
    block, l, m, intensity = _cband_problem()
    one_plane = timestep_jones_bytes(1, l.size, block.frequency_hz.size)
    full = timestep_jones_bytes(block.antenna_count, l.size, block.frequency_hz.size)
    assert full > one_plane
    streamed = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam()),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(max_timestep_jones_bytes=one_plane + 64),
    )
    batched = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam()),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(streamed.visibility, batched.visibility)
    with pytest.raises(ValueError, match="materialized Jones slices"):
        predict_voltage_beam(
            block,
            l,
            m,
            SkyStokesPlanes(stokes_i=intensity),
            DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam()),
            antenna_position_m=_ANTENNA_POSITION_M,
            calibration_state="casa_parang_true",
            config=BeamOperatorConfig(
                policy=BeamOperatorPolicy.MATERIALIZE,
                max_timestep_jones_bytes=one_plane + 64,
            ),
        )


def test_adjoint_is_exact_for_per_channel_stokes() -> None:
    block, l, m, intensity = _cband_problem()
    sky = SkyStokesPlanes(
        stokes_i=np.stack([intensity, 0.7 * intensity], axis=1),
        stokes_q=np.stack([0.05 * intensity, -0.02 * intensity], axis=1),
        stokes_u=np.zeros((l.size, 2)),
        stokes_v=np.zeros((l.size, 2)),
    )
    beam = AnalyticAiryVoltageBeam()
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        sky,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    rng = np.random.default_rng(5)
    residual = rng.normal(size=predicted.visibility.shape) + 1j * rng.normal(
        size=predicted.visibility.shape
    )
    gradient = adjoint_voltage_beam(
        residual,
        block,
        l,
        m,
        beam,
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert gradient[0].shape == (l.size, block.frequency_hz.size)
    left = float(np.sum(predicted.visibility * np.conjugate(residual)).real)
    right = float(
        np.sum(sky.stokes_i * gradient[0])
        + np.sum(sky.stokes_q * gradient[1])
        + np.sum(sky.stokes_u * gradient[2])
        + np.sum(sky.stokes_v * gradient[3])
    )
    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)


def test_rr_only_block_uses_the_full_jones() -> None:
    block, l, m, intensity = _cband_problem()
    slim = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=block.visibility[..., :1],
        weight=block.weight[..., :1],
        flag=block.flag[..., :1],
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=block.phase_centre_rad,
    )
    predicted = predict_voltage_beam(
        slim,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    full = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert predicted.visibility.shape[-1] == 1
    np.testing.assert_allclose(predicted.visibility[..., 0], full.visibility[..., 0])


def test_squint_accepts_the_composite_shape() -> None:
    l = np.array([0.0, np.sin(np.deg2rad(0.25))])
    m = np.zeros(2)
    intensity = np.array([1.0, 0.4])
    block = _block(
        l=l,
        m=m,
        intensity=intensity,
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9]),
        uvw_m=np.array([[0.0, 0.0, 0.0]]),
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=AnalyticAiryVoltageBeam(
            catalog=VLABeamCatalog(airy_max_radius_rad_at_1ghz=np.deg2rad(4.0))
        ),
        handover=CompositeHandoverPolicy.MATCH_POWER,
    )
    predicted = predict_voltage_beam(
        block,
        l,
        m,
        SkyStokesPlanes(stokes_i=intensity),
        DiagonalSquintVoltageBeam(shape=composite),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
    )
    assert np.all(predicted.valid)
    assert predicted.visibility.shape == (1, 1, 4)
