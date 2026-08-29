from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam_conventions import OBSERVATION_3C391_FREQUENCY_HZ
from sl1mjax.cassbeam_beam import (
    CASA_PARANG_PARALLACTIC_BASIS,
    CASSBEAM_CBAND_MODEL_ID,
    CASSBEAM_OUTER_AIRY_MAX_RADIUS_RAD_AT_1GHZ,
    MAX_NEAREST_NODE_SEPARATION_HZ,
    UNCALIBRATED_PARALLACTIC_BASIS,
    BeamImagingMode,
    CassbeamCBandVoltageBeam,
    cassbeam_common_mode_offset_arcmin,
    cassbeam_frequency_support_hz,
    cassbeam_fwhm_versus_nrao,
    cassbeam_receptor_mainlobe_separation_arcmin,
    diagonal_copolar_is_casa_accepted,
    load_cassbeam_cband_artifact,
    observation_3c391_is_inside_cassbeam_nearest_support,
    voltage_beam_for_mode,
)
from sl1mjax.full_jones import TermPresence, refuse_analytic_squint_composition
from sl1mjax.polarization import circular_parallactic_jones
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import (
    AnalyticAiryVoltageBeam,
    BeamCoordinates,
    BeamEvaluation,
    CompositeScalarVoltageBeam,
    beam_coordinates,
)


class _AlwaysValidOuter:
    model_id = "stub_outer"

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: str,
    ) -> BeamEvaluation:
        del calibration_state
        n_dir = coordinates.l_rad.size
        n_chan = coordinates.frequency_hz.size
        jones = np.zeros((1, n_dir, n_chan, 2, 2), dtype=np.complex128)
        jones[..., 0, 0] = 0.3
        jones[..., 1, 1] = 0.4
        return BeamEvaluation(
            jones=jones,
            valid=np.ones((1, n_dir, n_chan), dtype=bool),
            provenance={},
        )


def test_generated_artifact_checksums_and_nodes() -> None:
    artifact = load_cassbeam_cband_artifact()
    assert artifact.pin.artifact_id == CASSBEAM_CBAND_MODEL_ID
    assert artifact.pin.frozen is False
    assert len(artifact.tables) == 2
    assert artifact.tables[0].frequency_hz == pytest.approx(4.564e9)
    assert artifact.tables[1].frequency_hz == pytest.approx(4.692e9)
    assert artifact.tables[0].jones.shape[0] == 33
    assert diagonal_copolar_is_casa_accepted() is False
    assert cassbeam_fwhm_versus_nrao(4.564e9) == pytest.approx(1.0, abs=0.03)


def test_raster_origin_is_dephased_dc_after_even_n_reflection() -> None:
    table = load_cassbeam_cband_artifact().tables[0]
    half = int(float(table.params["gridsize"])) // 2
    aperture_n = 2 * half
    spacing = 12.5 / half
    wavelength = SPEED_OF_LIGHT_M_S / table.frequency_hz
    expected = wavelength / (float(table.params["pixelsperbeam"]) * aperture_n * spacing)
    assert table.pixel_scale_rad == pytest.approx(expected, rel=1e-12)
    invented = table.fwhm_l_rad / float(table.params["pixelsperbeam"])
    assert table.pixel_scale_rad != pytest.approx(invented, rel=1e-3)
    assert table.l_origin_index == 15
    assert table.m_origin_index == 16
    assert table.l_rad[15] == pytest.approx(0.0)
    assert table.m_rad[16] == pytest.approx(0.0)
    power = 0.5 * (
        np.abs(table.jones[..., 0, 0]) ** 2 + np.abs(table.jones[..., 1, 1]) ** 2
    )
    peak_m, peak_l = (int(index) for index in np.unravel_index(int(np.argmax(power)), power.shape))
    assert (peak_m, peak_l) == (table.m_origin_index, table.l_origin_index)
    labelled_centre = power[16, 16]
    assert power[peak_m, peak_l] > labelled_centre
    offset_l, offset_m = cassbeam_common_mode_offset_arcmin(table.frequency_hz)
    assert float(np.hypot(offset_l, offset_m)) < 0.25 * np.rad2deg(
        table.pixel_scale_rad
    ) * 60.0
    memo195 = 2.4 / (table.frequency_hz / 1.0e9)
    assert cassbeam_receptor_mainlobe_separation_arcmin(
        table.frequency_hz
    ) == pytest.approx(memo195, rel=0.10)


def test_nearest_node_support_covers_3c391_and_refuses_7ghz() -> None:
    low, high = cassbeam_frequency_support_hz()
    assert low == pytest.approx(4.564e9 - MAX_NEAREST_NODE_SEPARATION_HZ)
    assert high == pytest.approx(4.692e9 + MAX_NEAREST_NODE_SEPARATION_HZ)
    start, stop = OBSERVATION_3C391_FREQUENCY_HZ
    assert observation_3c391_is_inside_cassbeam_nearest_support()
    assert low <= start < stop <= high
    beam = voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR)
    allowed = beam.evaluate(
        beam_coordinates(0.0, 0.0, start),
        calibration_state="casa_parang_true",
    )
    assert allowed.valid[0, 0, 0]
    with pytest.raises(ValueError, match="more than 64 MHz"):
        beam.evaluate(
            beam_coordinates(0.0, 0.0, 7.0e9),
            calibration_state="casa_parang_true",
        )
    with pytest.raises(ValueError, match="more than 64 MHz"):
        beam.evaluate(
            beam_coordinates(0.0, 0.0, 1.4e9),
            calibration_state="casa_parang_true",
        )


def test_diagonal_copolar_is_identity_on_axis_and_zeros_offdiag() -> None:
    beam = voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR)
    evaluation = beam.evaluate(
        beam_coordinates(0.0, 0.0, 4.6e9),
        calibration_state="casa_parang_true",
    )
    assert evaluation.valid[0, 0, 0]
    np.testing.assert_allclose(evaluation.jones[0, 0, 0], np.eye(2), atol=1e-12)
    assert evaluation.provenance["off_diagonal"] is False


def test_full_jones_mode_is_experimental_and_keeps_offdiag() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact, off_diagonal=True, allow_unfrozen=True
    )
    on_axis = beam.evaluate(
        beam_coordinates(0.0, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(on_axis.jones[0, 0, 0], np.eye(2), atol=1e-12)
    offset = 0.5 * artifact.tables[0].fwhm_l_rad
    off_axis = beam.evaluate(
        beam_coordinates(offset, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert abs(off_axis.jones[0, 0, 0, 0, 1]) + abs(off_axis.jones[0, 0, 0, 1, 0]) > 0.0
    assert off_axis.provenance["experimental"] is True


def test_full_jones_factory_and_default_constructor_refuse_unfrozen() -> None:
    artifact = load_cassbeam_cband_artifact()
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode(BeamImagingMode.FULL_JONES)
    with pytest.raises(ValueError, match="not frozen"):
        CassbeamCBandVoltageBeam(artifact, off_diagonal=True)


def test_outer_field_reverts_to_scalar_beyond_the_raster() -> None:
    artifact = load_cassbeam_cband_artifact()
    stubbed = CassbeamCBandVoltageBeam(
        artifact, off_diagonal=False, outer=_AlwaysValidOuter()
    )
    far = stubbed.evaluate(
        beam_coordinates(0.2, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert far.valid[0, 0, 0]
    np.testing.assert_allclose(far.jones[0, 0, 0], np.diag([0.3, 0.4]))
    factory = voltage_beam_for_mode(BeamImagingMode.DIAGONAL_COPOLAR)
    assert isinstance(factory, CassbeamCBandVoltageBeam)
    assert isinstance(factory.outer, CompositeScalarVoltageBeam)
    assert factory.outer.outer.catalog.airy_max_radius_rad_at_1ghz == pytest.approx(
        CASSBEAM_OUTER_AIRY_MAX_RADIUS_RAD_AT_1GHZ
    )
    just_outside = float(np.max(np.abs(artifact.tables[0].l_rad))) + artifact.tables[
        0
    ].pixel_scale_rad
    composed = factory.evaluate(
        beam_coordinates(just_outside, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert composed.valid[0, 0, 0]
    assert composed.off_diagonal_valid is not None
    assert composed.off_diagonal_valid[0, 0, 0]
    assert composed.jones[0, 0, 0, 0, 1] == 0
    assert composed.jones[0, 0, 0, 1, 0] == 0
    beyond_airy_cutoff = factory.evaluate(
        beam_coordinates(0.2, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert beyond_airy_cutoff.valid[0, 0, 0]
    np.testing.assert_allclose(beyond_airy_cutoff.jones[0, 0, 0], 0.0, atol=1e-15)
    off_sky = factory.evaluate(
        beam_coordinates(1.1, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert not bool(off_sky.valid[0, 0, 0])


def test_parallactic_rotation_moves_the_lookup() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(artifact, off_diagonal=False)
    offset = 0.4 * artifact.tables[0].fwhm_l_rad
    along_l = beam.evaluate(
        beam_coordinates(offset, 0.0, 4.564e9, parallactic_angle_rad=0.0),
        calibration_state="casa_parang_true",
    )
    along_m = beam.evaluate(
        beam_coordinates(0.0, offset, 4.564e9, parallactic_angle_rad=0.5 * np.pi),
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(along_l.jones, along_m.jones, atol=1e-10)


def test_full_jones_applies_parallactic_basis_transformation() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact, off_diagonal=True, allow_unfrozen=True
    )
    offset = 0.4 * artifact.tables[0].fwhm_l_rad
    chi = 0.5 * np.pi
    feed = beam.evaluate(
        beam_coordinates(0.0, -offset, 4.564e9, parallactic_angle_rad=0.0),
        calibration_state="casa_parang_true",
    )
    sky = beam.evaluate(
        beam_coordinates(offset, 0.0, 4.564e9, parallactic_angle_rad=chi),
        calibration_state="casa_parang_true",
    )
    parallactic = circular_parallactic_jones(chi)
    expected = np.conjugate(parallactic.T) @ feed.jones[0, 0, 0] @ parallactic
    np.testing.assert_allclose(sky.jones[0, 0, 0], expected, atol=1e-10)
    np.testing.assert_allclose(
        sky.jones[0, 0, 0, 0, 1], -feed.jones[0, 0, 0, 0, 1], atol=1e-10
    )
    np.testing.assert_allclose(
        sky.jones[0, 0, 0, 1, 0], -feed.jones[0, 0, 0, 1, 0], atol=1e-10
    )
    assert not np.allclose(sky.jones, feed.jones, atol=1e-10)
    assert sky.provenance["parallactic_basis"] == CASA_PARANG_PARALLACTIC_BASIS


def test_uncalibrated_applies_feed_frame_parallactic_product() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact, off_diagonal=True, allow_unfrozen=True
    )
    offset = 0.4 * artifact.tables[0].fwhm_l_rad
    chi = 0.5 * np.pi
    feed = beam.evaluate(
        beam_coordinates(0.0, -offset, 4.564e9, parallactic_angle_rad=0.0),
        calibration_state="uncalibrated",
    )
    sky = beam.evaluate(
        beam_coordinates(offset, 0.0, 4.564e9, parallactic_angle_rad=chi),
        calibration_state="uncalibrated",
    )
    parallactic = circular_parallactic_jones(chi)
    expected = feed.jones[0, 0, 0] @ parallactic
    casa_form = np.conjugate(parallactic.T) @ feed.jones[0, 0, 0] @ parallactic
    np.testing.assert_allclose(sky.jones[0, 0, 0], expected, atol=1e-10)
    assert not np.allclose(sky.jones[0, 0, 0], casa_form, atol=1e-10)
    assert sky.provenance["parallactic_basis"] == UNCALIBRATED_PARALLACTIC_BASIS


def test_full_jones_marks_outer_field_off_diagonals_unsupported() -> None:
    artifact = load_cassbeam_cband_artifact()
    beam = CassbeamCBandVoltageBeam(
        artifact,
        off_diagonal=True,
        allow_unfrozen=True,
        outer=_AlwaysValidOuter(),
    )
    just_outside = float(np.max(np.abs(artifact.tables[0].l_rad))) + artifact.tables[
        0
    ].pixel_scale_rad
    evaluation = beam.evaluate(
        beam_coordinates(just_outside, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert evaluation.valid[0, 0, 0]
    assert evaluation.off_diagonal_valid is not None
    assert not bool(evaluation.off_diagonal_valid[0, 0, 0])
    assert evaluation.jones[0, 0, 0, 0, 1] == 0
    assert evaluation.jones[0, 0, 0, 1, 0] == 0
    inside = beam.evaluate(
        beam_coordinates(0.0, 0.0, 4.564e9),
        calibration_state="casa_parang_true",
    )
    assert inside.valid[0, 0, 0]
    assert inside.off_diagonal_valid is not None
    assert inside.off_diagonal_valid[0, 0, 0]


def test_mode_factory_keeps_scalar_backends() -> None:
    assert isinstance(voltage_beam_for_mode("static_scalar"), AnalyticAiryVoltageBeam)
    streamed = voltage_beam_for_mode("streamed_scalar")
    assert isinstance(streamed, CompositeScalarVoltageBeam)
    assert isinstance(streamed.outer, AnalyticAiryVoltageBeam)
    assert streamed.outer.catalog.airy_max_radius_rad_at_1ghz == pytest.approx(
        np.deg2rad(0.8)
    )
    assert load_cassbeam_cband_artifact().pin.contents.squint is TermPresence.PRESENT
    with pytest.raises(ValueError, match="replace analytic squint"):
        refuse_analytic_squint_composition(load_cassbeam_cband_artifact().pin.contents)
