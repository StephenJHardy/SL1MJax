from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.cassbeam_evla_c import (
    CASSBEAM_EVLA_C_FEED_M,
    GENERIC_VLA_FEED_M,
    evla_c_feedtaper_db,
    evla_c_input_text,
    evla_c_versus_generic_feed,
)
from sl1mjax.holography import (
    commanded_offset_from_direction_target,
    pointing_convention_diagnostics,
    pointing_table_as_dict,
    pointing_table_from_dict,
    source_relative_lm_rad,
    synthetic_holography_pointing_table,
)
from sl1mjax.holography_holoraster_coordinates import (
    ARCMIN_TO_RAD,
    LOCKED_AXIS_MAP,
    azelgeo_to_cassbeam_lm,
    holoraster_ms_geometry_oracle,
    holoraster_row_coordinates,
    lock_azelgeo_cassbeam_axis_map,
    manufactured_asymmetric_copolar,
    refuse_bare_offset_lm_rad,
    refuse_convention_ladder,
    source_lm_feed_from_commanded_azelgeo,
    squint_vector_kinds,
)

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))


def test_source_in_beam_is_negative_commanded_offset() -> None:
    commanded = np.array(
        [[3.0 * ARCMIN_TO_RAD, 0.0], [0.0, -2.0 * ARCMIN_TO_RAD]],
        dtype=np.float64,
    )
    source = source_lm_feed_from_commanded_azelgeo(commanded)
    np.testing.assert_allclose(source, -commanded)
    sky = np.zeros((1, 2), dtype=np.float64)
    expected = source_relative_lm_rad(sky, commanded)[:, 0, :]
    np.testing.assert_allclose(source, expected)


def test_locked_axis_map_does_not_swap() -> None:
    azel = np.array([[0.01, -0.02]], dtype=np.float64)
    mapped = azelgeo_to_cassbeam_lm(azel, LOCKED_AXIS_MAP)
    np.testing.assert_allclose(mapped, azel)
    named = holoraster_row_coordinates(azel)
    np.testing.assert_allclose(named["commanded_offset_azelgeo"], azel)
    np.testing.assert_allclose(named["source_lm_feed"], -azel)


def test_geometry_oracle_passes_synthetic_direction_minus_target() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    predicted = commanded_offset_from_direction_target(table, _PHASE)
    stored = table.column("POINTING_OFFSET").values_rad
    np.testing.assert_allclose(predicted, stored, atol=1.0e-12)
    report = holoraster_ms_geometry_oracle(table, _PHASE)
    assert report["passes"] is True
    assert report["source_is_negative_commanded"] is True


def test_pointing_offset_column_is_not_scored_against_itself() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    native = pointing_convention_diagnostics(table, _PHASE, selected_column="POINTING_OFFSET")
    assert native.status == "pass"
    assert any("DIRECTION-TARGET" in note for note in native.notes)
    flipped = pointing_table_from_dict(pointing_table_as_dict(table))
    stored = np.asarray(flipped.column("POINTING_OFFSET").values_rad)
    object.__setattr__(flipped.column("POINTING_OFFSET"), "values_rad", -stored)
    failed = pointing_convention_diagnostics(flipped, _PHASE, selected_column="POINTING_OFFSET")
    assert failed.status == "fail"


def test_manufactured_track_locks_no_swap_and_rejects_commanded_as_source() -> None:
    track = np.array(
        [
            [4.0 * ARCMIN_TO_RAD, 0.0],
            [0.0, 3.0 * ARCMIN_TO_RAD],
            [2.0 * ARCMIN_TO_RAD, -1.5 * ARCMIN_TO_RAD],
        ],
        dtype=np.float64,
    )
    report = lock_azelgeo_cassbeam_axis_map(track)
    assert report["locked"] is True
    assert report["axis_map"] == LOCKED_AXIS_MAP.name
    assert report["commanded_as_source_residual"] > 0.1
    physical = manufactured_asymmetric_copolar(source_lm_feed_from_commanded_azelgeo(track))
    wrong = manufactured_asymmetric_copolar(track)
    assert float(np.max(np.abs(physical - wrong))) > 0.1


def test_squint_vector_labels_distinguish_memo195() -> None:
    kinds = squint_vector_kinds(r_minus_l_lm=(0.292, 0.290))
    assert kinds["code_kind"] == "r_minus_l"
    assert kinds["memo195_kind"] == "r_to_l"
    np.testing.assert_allclose(kinds["r_to_l"], [-0.292, -0.290])


def test_evla_c_feed_is_not_the_generic_vla_template() -> None:
    comparison = evla_c_versus_generic_feed()
    assert comparison["radius_difference_m"] == pytest.approx(0.0, abs=2.0e-5)
    assert comparison["angle_separation_deg"] == pytest.approx(30.2, abs=0.1)
    assert evla_c_feedtaper_db(4.564e9) == pytest.approx(12.2115, abs=1.0e-4)
    assert evla_c_feedtaper_db(6.0e9) == pytest.approx(12.75)
    assert CASSBEAM_EVLA_C_FEED_M[0] == pytest.approx(-0.94300)
    assert GENERIC_VLA_FEED_M[0] == pytest.approx(-0.6896837)
    text = evla_c_input_text(frequency_ghz=4.564, output_name="evla-cband-4564")
    assert "feed_x = -0.94300" in text
    assert "feedtaper = 12.2115" in text
    assert "generic" in text.lower()
    data = Path(__file__).resolve().parents[1] / "src" / "sl1mjax" / "data" / "cassbeam_cband"
    assert (data / "evla-cband-4564.in").read_text() == evla_c_input_text(
        frequency_ghz=4.564, output_name="evla-cband-4564"
    )
    assert (data / "evla-cband-4692.in").read_text() == evla_c_input_text(
        frequency_ghz=4.692, output_name="evla-cband-4692"
    )


def test_kernels_have_no_row_loop() -> None:
    assert "for row" not in inspect.getsource(source_lm_feed_from_commanded_azelgeo)
    assert "for row" not in inspect.getsource(azelgeo_to_cassbeam_lm)
    assert "for row" not in inspect.getsource(manufactured_asymmetric_copolar)
    with pytest.raises(RuntimeError, match="ambiguous"):
        refuse_bare_offset_lm_rad()
    with pytest.raises(RuntimeError, match="128-member"):
        refuse_convention_ladder()
