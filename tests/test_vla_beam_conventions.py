from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.beam_conventions import (
    ANALYTIC_SQUINT_RECEPTOR_HALF_OFFSET_FWHM,
    ANTENNA_FRAME_POLARIZATION_LOCK,
    CASSBEAM_UNPINNED_REQUIREMENTS,
    CIRCULAR_P_JONES,
    CIRCULAR_STOKES,
    COTTON2008_TOTAL_SQUINT_FWHM_FRACTION,
    EVLA195_TOTAL_SQUINT_ARCMIN_GHZ,
    HOLOGRAPHY_UNPINNED_REQUIREMENTS,
    JONES_RECEPTOR_ORDER,
    OBSERVATION_3C391_FREQUENCY_HZ,
    ON_AXIS_DI_JONES_ORDER,
    PARALLACTIC_ANGLE_SIGN_LOCK,
    PERLEY2016_CATALOG_VERSION,
    PERLEY2016_MINIMUM_VALID_POWER,
    BeamCalibrationState,
    ConventionLock,
    PerleyFrequencyPolicy,
    analytic_squint_is_evidence_grade,
    analytic_squint_quantity,
    antenna_frame_polarization_is_physically_verified,
    artifact_by_id,
    beam_requires_identity_on_axis,
    casa_nearest_switch_frequency_hz,
    cband_reference_artifacts,
    current_analytic_receptor_half_offset_rad,
    current_analytic_total_squint_rad,
    current_versus_evla195_total_squint_ratio,
    evla195_receptor_half_offset_rad,
    evla195_total_squint_rad,
    gaussian_fwhm_rad,
    load_perley2016_cband_windows,
    nearest_perley2016_cband_window,
    nrao_gaussian_fwhm_arcmin,
    observation_3c391_crosses_casa_nearest_switch,
    perley2016_frequency_is_supported,
    perley2016_stokes_i_power,
    perley2016_stokes_i_validity,
    require_beam_calibration_state,
    select_perley2016_cband_window,
    sky_east_is_positive_l,
    sky_north_is_positive_m,
)
from sl1mjax.calibration_terms import geodetic_latitude_rad, parallactic_angle_rad
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    circular_parallactic_jones,
    circular_stokes_from_correlations,
)


def test_cband_inventory_labels_kind_and_refuses_lband_as_cband() -> None:
    artifacts = {artifact.artifact_id: artifact for artifact in cband_reference_artifacts()}
    assert artifacts["sl1mjax_airy"].kind == "analytic"
    assert artifacts["perley2016_cband_stokes_i"].kind == "empirical"
    assert artifacts["cassbeam_go"].kind == "electromagnetic"
    assert artifacts["perley2016_cband_stokes_i"].usable_for_cband is True
    assert artifacts["perley2016_cband_stokes_i"].frozen_reference is True
    assert artifacts["perley2016_cband_holography_grids"].usable_for_cband is True
    assert artifacts["perley2016_cband_holography_grids"].frozen_reference is False
    assert artifacts["perley2016_cband_holography_grids"].unpinned_requirements == (
        HOLOGRAPHY_UNPINNED_REQUIREMENTS
    )
    assert artifacts["cassbeam_go"].frozen_reference is False
    assert artifacts["cassbeam_go"].unpinned_requirements == CASSBEAM_UNPINNED_REQUIREMENTS
    assert artifacts["iheanetu2019_lband"].usable_for_cband is False
    assert artifacts["jagannathan2021_atoz_plumber"].usable_for_cband is False
    assert artifacts["jagannathan2021_atoz_plumber"].band == "S"
    assert artifacts["sl1mjax_analytic_squint"].usable_for_cband is False
    with pytest.raises(KeyError):
        artifact_by_id("missing")


def test_perley2016_catalog_covers_cband_and_is_unity_on_axis() -> None:
    windows = load_perley2016_cband_windows()
    assert len(windows) == 32
    assert windows[0].frequency_hz == pytest.approx(4.052e9)
    assert windows[-1].frequency_hz == pytest.approx(7.948e9)
    frequencies = [window.frequency_hz for window in windows]
    assert frequencies == sorted(frequencies)
    for window in windows:
        on_axis = float(perley2016_stokes_i_power(0.0, window.frequency_hz, window))
        half = float(
            perley2016_stokes_i_power(
                window.hwhm_arcmin_ghz / (window.frequency_hz / 1.0e9),
                window.frequency_hz,
                window,
            )
        )
        edge = window.support_radius_arcmin(window.frequency_hz)
        edge_power = float(perley2016_stokes_i_power(edge, window.frequency_hz, window))
        assert on_axis == pytest.approx(1.0, abs=1e-15)
        assert half == pytest.approx(0.5, abs=0.02)
        assert edge_power == pytest.approx(PERLEY2016_MINIMUM_VALID_POWER, abs=1e-12)


def test_perley2016_fails_closed_outside_frequency_and_radius_support() -> None:
    with pytest.raises(ValueError, match="outside Perley 2016 C-band support"):
        nearest_perley2016_cband_window(1.0e9)
    with pytest.raises(ValueError, match="finite positive"):
        nearest_perley2016_cband_window(float("nan"))
    window = nearest_perley2016_cband_window(4.6e9)
    with pytest.raises(ValueError, match="unsupported"):
        perley2016_stokes_i_power(60.0, 4.6e9, window)
    assert bool(perley2016_stokes_i_validity(60.0, 4.6e9, window)) is False
    outside = perley2016_stokes_i_power(60.0, 4.6e9, window, require_valid=False)
    assert np.isnan(outside)
    assert bool(perley2016_frequency_is_supported(4.052e9))
    assert bool(perley2016_frequency_is_supported(7.948e9))
    assert not bool(perley2016_frequency_is_supported(1.4e9))
    assert not bool(perley2016_frequency_is_supported(1.0e10))


def test_casa_nearest_is_not_an_interpolation_policy() -> None:
    window = select_perley2016_cband_window(
        4.6e9, policy=PerleyFrequencyPolicy.CASA_NEAREST
    )
    assert window.frequency_hz == pytest.approx(4.564e9)
    assert nearest_perley2016_cband_window(4.6e9).frequency_hz == window.frequency_hz
    assert PERLEY2016_CATALOG_VERSION == "evla-memo-195-table-5"
    with pytest.raises(ValueError, match="interpolation is not implemented"):
        select_perley2016_cband_window(4.6e9, policy=PerleyFrequencyPolicy.INTERPOLATED)
    assert observation_3c391_crosses_casa_nearest_switch() is True
    windows = load_perley2016_cband_windows()
    lower = next(item for item in windows if abs(item.frequency_hz - 4.564e9) < 0.5e6)
    upper = next(item for item in windows if abs(item.frequency_hz - 4.692e9) < 0.5e6)
    switch = casa_nearest_switch_frequency_hz(lower, upper)
    start, stop = OBSERVATION_3C391_FREQUENCY_HZ
    assert start < switch < stop
    below = select_perley2016_cband_window(
        np.nextafter(switch, start), policy=PerleyFrequencyPolicy.CASA_NEAREST
    )
    above = select_perley2016_cband_window(
        np.nextafter(switch, stop), policy=PerleyFrequencyPolicy.CASA_NEAREST
    )
    assert below.frequency_hz == pytest.approx(4.564e9)
    assert above.frequency_hz == pytest.approx(4.692e9)
    at_4p6 = float(perley2016_stokes_i_power(4.6, switch, below))
    at_8 = float(perley2016_stokes_i_power(8.0, switch, below))
    jump_4p6 = abs(float(perley2016_stokes_i_power(4.6, switch, above)) / at_4p6 - 1.0)
    jump_8 = abs(float(perley2016_stokes_i_power(8.0, switch, above)) / at_8 - 1.0)
    assert jump_4p6 == pytest.approx(0.011, abs=0.003)
    assert jump_8 == pytest.approx(0.044, abs=0.005)


def test_sky_frame_l_is_east_and_m_is_north() -> None:
    assert sky_east_is_positive_l()
    assert sky_north_is_positive_m()
    l_east, m_east, _ = radec_to_lmn(0.0, 0.0, 1.0e-3, 0.0)
    l_north, m_north, _ = radec_to_lmn(0.0, 0.0, 0.0, 1.0e-3)
    assert float(l_east) > 0.0
    assert float(m_east) == pytest.approx(0.0, abs=1e-15)
    assert float(m_north) > 0.0
    assert float(l_north) == pytest.approx(0.0, abs=1e-15)


def test_circular_receptors_stokes_and_p_jones_are_casa() -> None:
    assert JONES_RECEPTOR_ORDER == (Receptor.R, Receptor.L)
    assert CIRCULAR_STOKES == "RR=I+V, LL=I-V, RL=Q+iU, LR=Q-iU"
    assert CIRCULAR_P_JONES == "diag(exp(-i chi), exp(+i chi))"
    correlations = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    visibility = np.asarray([1.1 + 0.0j, 0.2 + 0.3j, 0.2 - 0.3j, 0.9 + 0.0j])
    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        visibility, correlations
    )
    assert float(np.real(stokes_i)) == pytest.approx(1.0)
    assert float(np.real(stokes_q)) == pytest.approx(0.2)
    assert float(np.real(stokes_u)) == pytest.approx(0.3)
    assert float(np.real(stokes_v)) == pytest.approx(0.1)
    chi = 0.3
    jones = circular_parallactic_jones(chi)
    assert jones[0, 0] == pytest.approx(np.exp(-1j * chi))
    assert jones[1, 1] == pytest.approx(np.exp(+1j * chi))
    assert jones[0, 1] == pytest.approx(0.0)
    assert jones[1, 0] == pytest.approx(0.0)


def _vla_antenna_and_meridian() -> tuple[np.ndarray, float, np.ndarray, float, float]:
    position = np.array([[-1_601_185.0, -5_041_977.0, 3_554_875.0]])
    latitude = float(geodetic_latitude_rad(position)[0])
    times = np.array([0.0])
    julian_date = times / 86400.0 + 2_400_000.5
    gmst = np.deg2rad(
        np.mod(280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0), 360.0)
    )
    longitude = float(np.arctan2(position[0, 1], position[0, 0]))
    return position, latitude, times, float(gmst[0]), longitude


def test_parallactic_angle_is_zero_at_southern_transit() -> None:
    position, latitude, times, gmst, longitude = _vla_antenna_and_meridian()
    assert np.rad2deg(latitude) == pytest.approx(34.08, abs=0.05)
    chi = parallactic_angle_rad(
        times,
        (gmst + longitude, latitude - np.deg2rad(20.0)),
        position,
    )
    assert float(chi[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_parallactic_angle_is_positive_for_western_hour_angle() -> None:
    position, latitude, times, gmst, longitude = _vla_antenna_and_meridian()
    chi = parallactic_angle_rad(
        times,
        (gmst + longitude - np.deg2rad(15.0), latitude - np.deg2rad(20.0)),
        position,
    )
    assert float(chi[0, 0]) > 0.0


def test_antenna_frame_polarization_orientation_is_not_physically_verified() -> None:
    assert ANTENNA_FRAME_POLARIZATION_LOCK is ConventionLock.PHYSICALLY_UNVERIFIED
    assert PARALLACTIC_ANGLE_SIGN_LOCK is ConventionLock.INTERNAL
    assert antenna_frame_polarization_is_physically_verified() is False


def test_calibration_state_rejects_unknown_identifiers() -> None:
    state = require_beam_calibration_state("casa_parang_true")
    assert state is BeamCalibrationState.CASA_PARANG_TRUE
    assert beam_requires_identity_on_axis(state) is True
    assert beam_requires_identity_on_axis("casa_parang_true") is True
    assert beam_requires_identity_on_axis(BeamCalibrationState.UNCALIBRATED) is False
    assert beam_requires_identity_on_axis("uncalibrated") is False
    assert ON_AXIS_DI_JONES_ORDER == "GKB Kcross D X P"
    with pytest.raises(ValueError, match="refuse to guess Jones order"):
        require_beam_calibration_state("maybe_parang")
    with pytest.raises(ValueError, match="refuse to guess Jones order"):
        beam_requires_identity_on_axis("maybe_parang")


def test_casa_parang_true_beam_is_identity_on_axis() -> None:
    beam = VLAPrimaryBeam(kind="airy")
    assert float(beam.power(0.0, 0.0, 4.6e9)) == pytest.approx(1.0, rel=1e-12)
    assert beam.apply_squint is False


def test_squint_half_offset_is_named_and_about_twice_memo_195() -> None:
    frequency = 4.6e9
    assert analytic_squint_quantity() == "receptor_half_offset"
    assert analytic_squint_is_evidence_grade() is False
    assert ANALYTIC_SQUINT_RECEPTOR_HALF_OFFSET_FWHM == pytest.approx(0.06)
    half = float(current_analytic_receptor_half_offset_rad(frequency))
    total = float(current_analytic_total_squint_rad(frequency))
    fwhm = float(gaussian_fwhm_rad(frequency))
    assert half == pytest.approx(0.06 * fwhm, rel=1e-12)
    assert total == pytest.approx(2.0 * half, rel=1e-12)
    published_total = float(evla195_total_squint_rad(frequency))
    published_half = float(evla195_receptor_half_offset_rad(frequency))
    assert published_total == pytest.approx(
        np.deg2rad(EVLA195_TOTAL_SQUINT_ARCMIN_GHZ / 60.0) * (1.0e9 / frequency)
    )
    assert published_half == pytest.approx(0.5 * published_total)
    ratio = current_versus_evla195_total_squint_ratio(frequency)
    assert ratio == pytest.approx(2.1, abs=0.15)
    published_fraction = published_total / fwhm
    assert published_fraction == pytest.approx(0.057, abs=0.004)
    assert COTTON2008_TOTAL_SQUINT_FWHM_FRACTION == pytest.approx(published_fraction, abs=0.01)


def test_unused_squint_rotation_is_an_internal_convention() -> None:
    frequency = 1.0e9
    offset = float(current_analytic_receptor_half_offset_rad(frequency))
    l_peak = np.sin(offset)
    east = VLAPrimaryBeam(kind="gaussian", apply_squint=True, parallactic_angle_rad=0.0)
    north = VLAPrimaryBeam(
        kind="gaussian", apply_squint=True, parallactic_angle_rad=0.5 * np.pi
    )
    rr_east = east.power([-l_peak, l_peak], [0.0, 0.0], frequency, receptor="RR")
    ll_east = east.power([-l_peak, l_peak], [0.0, 0.0], frequency, receptor="LL")
    rr_north = north.power([0.0, 0.0], [-l_peak, l_peak], frequency, receptor="RR")
    assert float(rr_east[1]) > float(rr_east[0])
    assert float(ll_east[0]) > float(ll_east[1])
    assert float(rr_north[1]) > float(rr_north[0])
    assert antenna_frame_polarization_is_physically_verified() is False


def test_nrao_gaussian_width_matches_catalog_fwhm() -> None:
    frequency = 4.6e9
    catalog_arcmin = float(np.rad2deg(gaussian_fwhm_rad(frequency)) * 60.0)
    nrao_arcmin = float(nrao_gaussian_fwhm_arcmin(frequency))
    assert nrao_arcmin == pytest.approx(42.0 / 4.6, rel=1e-12)
    assert catalog_arcmin == pytest.approx(nrao_arcmin, rel=0.01)
