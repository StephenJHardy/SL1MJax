from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam import VLABeamCatalog, VLAPrimaryBeam
from sl1mjax.beam_conventions import (
    PERLEY2016_MINIMUM_VALID_POWER,
    BeamCalibrationState,
    PerleyFrequencyPolicy,
    SquintMagnitudePolicy,
    evla195_receptor_half_offset_rad,
    nearest_perley2016_cband_window,
    nrao_gaussian_fwhm_arcmin,
    perley2016_stokes_i_power,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import (
    AIRY_VOLTAGE_PHASE_CONVENTION,
    PERLEY_VOLTAGE_PHASE_CONVENTION,
    AnalyticAiryVoltageBeam,
    CompositeHandoverPolicy,
    CompositeScalarVoltageBeam,
    DiagonalSquintVoltageBeam,
    Perley2016CBandVoltageBeam,
    beam_coordinates,
    stokes_i_power_from_jones,
)


def _reference_j1(argument: np.ndarray) -> np.ndarray:
    """Integral J1, independent of the power-series implementation in beam.py."""

    values = np.asarray(argument, dtype=np.float64)
    theta = np.linspace(0.0, np.pi, 4001)
    return np.trapezoid(
        np.cos(theta - values[..., None] * np.sin(theta)),
        theta,
        axis=-1,
    ) / np.pi


def _reference_blocked_airy_power(
    l: np.ndarray, m: np.ndarray, frequency_hz: np.ndarray
) -> np.ndarray:
    radius = np.arcsin(np.hypot(l, m))
    wavelength = SPEED_OF_LIGHT_M_S / frequency_hz
    argument = np.pi * 25.0 * np.sin(radius) / wavelength
    blockage = 2.5 / 25.0

    def jinc(values: np.ndarray) -> np.ndarray:
        return np.divide(
            2.0 * _reference_j1(values),
            values,
            out=np.ones_like(values),
            where=np.abs(values) >= 1e-12,
        )

    voltage = (jinc(argument) - blockage**2 * jinc(blockage * argument)) / (
        1.0 - blockage**2
    )
    return np.square(voltage)


def _extended_outer() -> AnalyticAiryVoltageBeam:
    return AnalyticAiryVoltageBeam(
        catalog=VLABeamCatalog(airy_max_radius_rad_at_1ghz=np.deg2rad(4.0))
    )


def test_airy_voltage_jones_matches_static_and_independent_power() -> None:
    l = np.array([0.0, np.sin(np.deg2rad(0.04)), np.sin(np.deg2rad(0.12))])
    m = np.zeros(3)
    frequency = np.array([4.536e9, 4.662e9])
    static = VLAPrimaryBeam(kind="airy")
    evaluation = AnalyticAiryVoltageBeam().evaluate(
        beam_coordinates(l, m, frequency, parallactic_angle_rad=0.3),
        calibration_state="casa_parang_true",
    )
    expected = static.power(l[:, None], m[:, None], frequency[None, :])
    independent = _reference_blocked_airy_power(
        l[:, None], m[:, None], frequency[None, :]
    )
    power = stokes_i_power_from_jones(evaluation.jones)
    np.testing.assert_allclose(power[0], expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(power[0], independent, rtol=2e-4, atol=1e-12)
    assert evaluation.jones.shape == (1, 3, 2, 2, 2)
    assert evaluation.valid.shape == (1, 3, 2)
    assert np.all(evaluation.valid)
    np.testing.assert_allclose(evaluation.jones.imag, 0.0, atol=1e-15)
    np.testing.assert_allclose(evaluation.jones[..., 0, 1], 0.0)
    np.testing.assert_allclose(evaluation.jones[..., 1, 0], 0.0)
    np.testing.assert_allclose(evaluation.jones[0, 0, 0], np.eye(2))
    assert evaluation.provenance["model_id"] == "analytic_blocked_airy"
    assert evaluation.provenance["voltage_phase_convention"] == AIRY_VOLTAGE_PHASE_CONVENTION
    assert evaluation.provenance["ignored_coordinates"] == [
        "antenna_id",
        "parallactic_angle_rad",
        "elevation_rad",
    ]
    other_pa = AnalyticAiryVoltageBeam().evaluate(
        beam_coordinates(l, m, frequency, parallactic_angle_rad=-1.2),
        calibration_state=BeamCalibrationState.CASA_PARANG_TRUE,
    )
    np.testing.assert_allclose(other_pa.jones, evaluation.jones)


def test_airy_sidelobe_voltage_is_signed() -> None:
    frequency = np.array([4.6e9])
    l = np.sin(np.deg2rad(np.array([0.25])))
    evaluation = _extended_outer().evaluate(
        beam_coordinates(l, [0.0], frequency),
        calibration_state="casa_parang_true",
    )
    voltage = evaluation.jones[0, 0, 0, 0, 0]
    assert float(np.real(voltage)) < 0.0
    assert evaluation.provenance["voltage_phase_convention"] == AIRY_VOLTAGE_PHASE_CONVENTION


def test_airy_marks_horizon_invalid_and_zeros_beyond_cutoff() -> None:
    frequency = np.array([1.0e9])
    catalog = VLABeamCatalog()
    outside = float(np.sin(catalog.airy_max_radius_rad(1.0e9) * 1.01))
    evaluation = AnalyticAiryVoltageBeam(catalog=catalog).evaluate(
        beam_coordinates([0.0, outside, 0.8], [0.0, 0.0, 0.8], frequency),
        calibration_state="casa_parang_true",
    )
    assert bool(evaluation.valid[0, 0, 0])
    assert bool(evaluation.valid[0, 1, 0])
    assert not bool(evaluation.valid[0, 2, 0])
    assert stokes_i_power_from_jones(evaluation.jones)[0, 1, 0] == 0.0


def test_voltage_beam_rejects_unknown_calibration_state() -> None:
    coordinates = beam_coordinates([0.0], [0.0], [4.6e9])
    with pytest.raises(ValueError, match="refuse to guess Jones order"):
        AnalyticAiryVoltageBeam().evaluate(
            coordinates, calibration_state="maybe_parang"
        )


def test_perley_voltage_is_sqrt_of_catalog_power_inside_support() -> None:
    window = nearest_perley2016_cband_window(4.6e9)
    offsets_arcmin = np.array([0.0, 2.0, 4.6])
    l = np.sin(np.deg2rad(offsets_arcmin / 60.0))
    evaluation = Perley2016CBandVoltageBeam().evaluate(
        beam_coordinates(l, np.zeros_like(l), [4.6e9]),
        calibration_state="casa_parang_true",
    )
    expected = perley2016_stokes_i_power(offsets_arcmin, 4.6e9, window)
    np.testing.assert_allclose(
        stokes_i_power_from_jones(evaluation.jones)[0, :, 0], expected, rtol=1e-12
    )
    assert np.all(evaluation.valid)
    assert evaluation.provenance["frequency_policy"] == "casa_nearest"
    assert evaluation.provenance["support_class"] == "measured"
    assert evaluation.provenance["voltage_phase_convention"] == PERLEY_VOLTAGE_PHASE_CONVENTION
    catalog = evaluation.provenance["catalog"]
    assert isinstance(catalog, dict)
    assert catalog["selected_window_hz"][0] == pytest.approx(4.564e9)


def test_perley_fails_closed_outside_frequency_and_radius() -> None:
    l_far = np.sin(np.deg2rad(1.0))
    far = Perley2016CBandVoltageBeam().evaluate(
        beam_coordinates([0.0, l_far], [0.0, 0.0], [4.6e9]),
        calibration_state="casa_parang_true",
    )
    assert bool(far.valid[0, 0, 0])
    assert not bool(far.valid[0, 1, 0])
    assert stokes_i_power_from_jones(far.jones)[0, 1, 0] == 0.0
    lband = Perley2016CBandVoltageBeam().evaluate(
        beam_coordinates([0.0], [0.0], [1.4e9]),
        calibration_state="casa_parang_true",
    )
    assert not bool(lband.valid[0, 0, 0])
    with pytest.raises(ValueError, match="interpolation is not implemented"):
        Perley2016CBandVoltageBeam(frequency_policy=PerleyFrequencyPolicy.INTERPOLATED)


def test_perley_is_consistent_with_transcribed_table_hwhm() -> None:
    frequencies = np.array([4.564e9, 4.948e9, 6.052e9])
    windows = [nearest_perley2016_cband_window(float(frequency)) for frequency in frequencies]
    offsets = np.array(
        [
            window.hwhm_arcmin_ghz / (window.frequency_hz / 1.0e9)
            for window in windows
        ]
    )
    l = np.sin(np.deg2rad(offsets / 60.0))
    evaluation = Perley2016CBandVoltageBeam().evaluate(
        beam_coordinates(l, np.zeros_like(l), frequencies),
        calibration_state="casa_parang_true",
    )
    power = stokes_i_power_from_jones(evaluation.jones)[0]
    for index, window in enumerate(windows):
        assert power[index, index] == pytest.approx(0.5, abs=0.02)
        nrao_hwhm = float(nrao_gaussian_fwhm_arcmin(window.frequency_hz)) / 2.0
        perley_hwhm = window.hwhm_arcmin_ghz / (window.frequency_hz / 1.0e9)
        assert perley_hwhm == pytest.approx(nrao_hwhm, rel=0.06)
    catalog = evaluation.provenance["catalog"]
    assert isinstance(catalog, dict)
    assert "Memo 195 Table 5" in str(catalog["coefficient_source"])


def test_perley_and_airy_attenuation_at_selected_cband_radii() -> None:
    frequency = np.array([4.6e9])
    offsets_arcmin = np.array([0.0, 2.0, 4.0, 6.0])
    l = np.sin(np.deg2rad(offsets_arcmin / 60.0))
    coordinates = beam_coordinates(l, np.zeros_like(l), frequency)
    perley = Perley2016CBandVoltageBeam().evaluate(
        coordinates, calibration_state="casa_parang_true"
    )
    airy = AnalyticAiryVoltageBeam().evaluate(
        coordinates, calibration_state="casa_parang_true"
    )
    perley_power = stokes_i_power_from_jones(perley.jones)[0, :, 0]
    airy_power = stokes_i_power_from_jones(airy.jones)[0, :, 0]
    assert perley_power[0] == pytest.approx(1.0)
    assert airy_power[0] == pytest.approx(1.0)
    assert np.all(np.diff(perley_power) < 0.0)
    assert np.all(np.diff(airy_power) < 0.0)
    relative = np.abs(perley_power[1:] / airy_power[1:] - 1.0)
    assert np.all(relative < 0.15)


def test_composite_does_not_reopen_out_of_band_frequencies() -> None:
    coordinates = beam_coordinates([0.0, 0.0], [0.0, 0.0], [1.4e9, 1.0e10])
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=_extended_outer(),
        handover=CompositeHandoverPolicy.MATCH_POWER,
    ).evaluate(coordinates, calibration_state="casa_parang_true")
    assert not np.any(composite.valid)
    np.testing.assert_allclose(composite.jones, 0.0)


def test_composite_match_power_is_continuous_at_the_perley_edge() -> None:
    window = nearest_perley2016_cband_window(4.6e9)
    edge = window.support_radius_arcmin(4.6e9)
    offsets = np.array([0.0, 4.0, edge * (1.0 - 1e-8), edge * (1.0 + 1e-8), 15.0])
    l = np.sin(np.deg2rad(offsets / 60.0))
    coordinates = beam_coordinates(l, np.zeros_like(l), [4.6e9])
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=_extended_outer(),
        handover=CompositeHandoverPolicy.MATCH_POWER,
    ).evaluate(coordinates, calibration_state="casa_parang_true")
    power = stokes_i_power_from_jones(composite.jones)[0, :, 0]
    assert np.all(composite.valid)
    assert power[2] == pytest.approx(PERLEY2016_MINIMUM_VALID_POWER, rel=1e-6)
    assert power[3] == pytest.approx(power[2], rel=1e-6)
    assert composite.provenance["catalog"]["handover"] == "match_power"


def test_composite_hard_splice_jump_is_declared() -> None:
    window = nearest_perley2016_cband_window(4.6e9)
    edge = window.support_radius_arcmin(4.6e9)
    offsets = np.array([edge * (1.0 - 1e-8), edge * (1.0 + 1e-8)])
    l = np.sin(np.deg2rad(offsets / 60.0))
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=_extended_outer(),
        handover=CompositeHandoverPolicy.HARD_SPLICE,
    ).evaluate(
        beam_coordinates(l, np.zeros_like(l), [4.6e9]),
        calibration_state="casa_parang_true",
    )
    power = stokes_i_power_from_jones(composite.jones)[0, :, 0]
    jump = abs(power[1] / power[0] - 1.0)
    assert jump == pytest.approx(0.194, abs=0.02)
    assert composite.provenance["catalog"]["handover"] == "hard_splice"


def test_composite_uses_perley_in_the_main_lobe_and_airy_outside() -> None:
    frequency = np.array([4.6e9])
    offsets_arcmin = np.array([0.0, 4.0, 15.0])
    l = np.sin(np.deg2rad(offsets_arcmin / 60.0))
    coordinates = beam_coordinates(l, np.zeros_like(l), frequency)
    outer = _extended_outer()
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=outer,
        handover=CompositeHandoverPolicy.MATCH_POWER,
    ).evaluate(coordinates, calibration_state="casa_parang_true")
    perley = Perley2016CBandVoltageBeam().evaluate(
        coordinates, calibration_state="casa_parang_true"
    )
    assert np.all(composite.valid)
    np.testing.assert_allclose(composite.jones[0, :2], perley.jones[0, :2])
    assert not bool(perley.valid[0, 2, 0])
    assert bool(composite.valid[0, 2, 0])
    assert composite.provenance["support_class"] == "measured_with_analytic_outer"


def test_stokes_i_power_includes_off_diagonal_jones() -> None:
    jones = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    jones[..., 0, 1] = 0.4
    jones[..., 1, 0] = 0.3
    assert float(stokes_i_power_from_jones(jones)[0, 0, 0]) == pytest.approx(
        0.5 * (0.16 + 0.09)
    )


def test_pointing_offset_recenters_the_voltage_beam() -> None:
    pointing = (0.01, -0.02)
    evaluation = AnalyticAiryVoltageBeam().evaluate(
        beam_coordinates(
            [0.01, 0.0],
            [-0.02, 0.0],
            [4.6e9],
            pointing_offset_lm_rad=pointing,
        ),
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(evaluation.jones[0, 0, 0], np.eye(2), atol=1e-12)
    assert stokes_i_power_from_jones(evaluation.jones)[0, 1, 0] < 1.0


def _squint_beam() -> DiagonalSquintVoltageBeam:
    return DiagonalSquintVoltageBeam(shape=AnalyticAiryVoltageBeam())


def test_diagonal_squint_is_identity_on_axis_and_uses_memo_195() -> None:
    frequency = np.array([4.6e9])
    evaluation = _squint_beam().evaluate(
        beam_coordinates([0.0], [0.0], frequency, parallactic_angle_rad=0.0),
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(evaluation.jones[0, 0, 0], np.eye(2), atol=1e-12)
    catalog = evaluation.provenance["catalog"]
    assert isinstance(catalog, dict)
    assert catalog["magnitude_policy"] == SquintMagnitudePolicy.EVLA195.value
    assert catalog["legacy_analytic_half_offset_enabled"] is False
    assert evaluation.provenance["creates_i_to_v"] is True
    assert evaluation.provenance["creates_i_to_qu"] is False
    assert evaluation.provenance["ignored_coordinates"] == ["elevation_rad"]
    assert antenna_frame_still_unverified(evaluation)


def antenna_frame_still_unverified(evaluation) -> bool:
    catalog = evaluation.provenance["catalog"]
    assert isinstance(catalog, dict)
    return catalog["feed_frame_polarization"] == "physically_unverified"


def test_diagonal_squint_refuses_the_legacy_half_offset() -> None:
    with pytest.raises(ValueError, match="not evidence-grade"):
        DiagonalSquintVoltageBeam(
            shape=AnalyticAiryVoltageBeam(),
            magnitude=SquintMagnitudePolicy.LEGACY_ANALYTIC_HALF_OFFSET,
        )


def test_diagonal_squint_rotates_r_to_plus_l_at_chi_zero() -> None:
    frequency = 4.6e9
    half = float(evla195_receptor_half_offset_rad(frequency))
    peak = np.sin(half)
    evaluation = _squint_beam().evaluate(
        beam_coordinates(
            [-peak, peak],
            [0.0, 0.0],
            [frequency],
            parallactic_angle_rad=0.0,
        ),
        calibration_state="casa_parang_true",
    )
    power_r = np.abs(evaluation.jones[0, :, 0, 0, 0]) ** 2
    power_l = np.abs(evaluation.jones[0, :, 0, 1, 1]) ** 2
    assert float(power_r[1]) > float(power_r[0])
    assert float(power_l[0]) > float(power_l[1])
    north = _squint_beam().evaluate(
        beam_coordinates(
            [0.0, 0.0],
            [-peak, peak],
            [frequency],
            parallactic_angle_rad=0.5 * np.pi,
        ),
        calibration_state="casa_parang_true",
    )
    power_r_n = np.abs(north.jones[0, :, 0, 0, 0]) ** 2
    assert float(power_r_n[1]) > float(power_r_n[0])


def test_diagonal_squint_makes_v_not_qu_from_unpolarised_i() -> None:
    frequency = np.array([4.6e9])
    half = float(evla195_receptor_half_offset_rad(4.6e9))
    l = np.array([0.0, np.sin(3.0 * half), -np.sin(3.0 * half)])
    evaluation = _squint_beam().evaluate(
        beam_coordinates(l, np.zeros_like(l), frequency, parallactic_angle_rad=0.0),
        calibration_state="casa_parang_true",
    )
    rr = np.abs(evaluation.jones[0, :, 0, 0, 0]) ** 2
    ll = np.abs(evaluation.jones[0, :, 0, 1, 1]) ** 2
    apparent_v = 0.5 * (rr - ll)
    assert apparent_v[0] == pytest.approx(0.0, abs=1e-12)
    assert apparent_v[1] > 0.0
    assert apparent_v[2] < 0.0
    np.testing.assert_allclose(evaluation.jones[..., 0, 1], 0.0)
    np.testing.assert_allclose(evaluation.jones[..., 1, 0], 0.0)


def test_diagonal_squint_offset_scales_as_one_over_frequency() -> None:
    low = float(evla195_receptor_half_offset_rad(4.6e9))
    high = float(evla195_receptor_half_offset_rad(6.9e9))
    assert high == pytest.approx(low * 4.6 / 6.9, rel=1e-12)


def test_diagonal_squint_accepts_composite_shape() -> None:
    composite = CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=_extended_outer(),
        handover=CompositeHandoverPolicy.MATCH_POWER,
    )
    evaluation = DiagonalSquintVoltageBeam(shape=composite).evaluate(
        beam_coordinates(
            [0.0, np.sin(np.deg2rad(0.25))],
            [0.0, 0.0],
            [4.6e9],
            parallactic_angle_rad=0.0,
        ),
        calibration_state="casa_parang_true",
    )
    assert np.all(evaluation.valid)
    np.testing.assert_allclose(evaluation.jones[0, 0, 0], np.eye(2), atol=1e-12)
    assert evaluation.provenance["support_class"] == "measured_with_analytic_outer"
