from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.beam_operator import BeamOperatorConfig, SkyStokesPlanes, predict_voltage_beam
from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography import (
    CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32,
    PERLEY_BUTLER_2017_3C147_TABLE5_JY_AT_4P564GHZ,
    THOL0001_EXECUTION_BLOCK,
    THOL0001_PROJECT,
    AntennaPointingRole,
    DirectionMeasure,
    HolographyConventionHypothesis,
    HolographyHoldoutAxis,
    HolographyObservation,
    HolographyRowReason,
    HolographySourceComponent,
    HolographySourceModel,
    Memo195LowerCRaster,
    ResolvedAntennaPointing,
    audit_resolved_pointing,
    casa_setjy_fluxd_jy,
    circular_visibility_to_source_coherency,
    commanded_pointing_lm_rad,
    compare_holography_beam_predictions,
    evaluate_holography_on_axis_jones,
    holography_archive_is_present,
    holography_commanded_pointing,
    holography_comparison_beams,
    holography_feed_frame_direction,
    holography_holdout_split,
    holography_phase_centre_sky,
    holography_raster_tracks,
    holography_source_relative_direction,
    inventory_as_dict,
    inventory_from_dict,
    inventory_measurement_set,
    load_synthetic_holography_metadata_fixture,
    load_synthetic_holography_pointing_table,
    load_synthetic_holography_rime_case,
    load_three_c147_source_model,
    perley_butler_2017_3c147_stokes_i_jy,
    point_source_is_adequate,
    pointing_convention_diagnostics,
    pointing_planes_for_block,
    pointing_table_as_dict,
    pointing_table_from_dict,
    predict_holography_visibilities,
    promote_operator_pointing,
    report_holography_residuals,
    require_holography_archive,
    resolve_antenna_pointing,
    resolved_pointing_as_dict,
    resolved_pointing_from_dict,
    score_holography_convention_ladder,
    source_relative_lm_rad,
    synthetic_holography_pointing_table,
    synthetic_holography_rime_case,
    synthetic_memo195_pointing_table,
    three_c147_casa_setjy_point_source_model,
    three_c147_flux_scale_report,
    three_c147_point_source_model,
    write_holography_raster_track_plot,
)
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    apply_jones_to_coherency,
    circular_stokes_to_coherency,
    unpack_coherency,
)

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))


def _fake_ms(root: Path) -> Path:
    measurement_set = root / "eb31629959.ms"
    for name in (
        "POINTING",
        "SOURCE",
        "FIELD",
        "STATE",
        "ANTENNA",
        "FEED",
        "POLARIZATION",
        "MAIN",
        "SPECTRAL_WINDOW",
    ):
        (measurement_set / name).mkdir(parents=True)
    (measurement_set / "ANTENNA" / "table.dat").write_text("ea01\nea02\nea03\n")
    (measurement_set / "ANTENNA" / "positions.txt").write_text(
        "-1601162.0 -5042003.0 3553983.0\n"
        "-1601100.0 -5042100.0 3553900.0\n"
        "-1601200.0 -5042190.0 3554000.0\n"
    )
    (measurement_set / "POLARIZATION" / "corr_type.txt").write_text("5 6 7 8\n")
    (measurement_set / "MAIN" / "time.txt").write_text("0.0 10.0 20.0 30.0\n")
    (measurement_set / "SPECTRAL_WINDOW" / "chan_freq.txt").write_text("4.536e9 4.662e9\n")
    (measurement_set / "POINTING" / "table.dat").write_text("pointing\n")
    return measurement_set


def test_archive_is_absent_until_download_lands() -> None:
    assert holography_archive_is_present(Path("data/holography")) is False
    with pytest.raises(FileNotFoundError, match="refuses to invent"):
        require_holography_archive(Path("data/holography"))


def test_inventory_is_deterministic_and_records_pointing(tmp_path: Path) -> None:
    measurement_set = _fake_ms(tmp_path)
    first = inventory_measurement_set(measurement_set)
    second = inventory_measurement_set(measurement_set)
    assert first.archive_sha256 == second.archive_sha256
    assert first.project == THOL0001_PROJECT
    assert first.execution_block == THOL0001_EXECUTION_BLOCK
    assert first.pointing_is_present() is True
    assert first.correlations == ("RR", "RL", "LR", "LL")
    assert first.correlation_codes == (5, 6, 7, 8)
    assert first.antennas_resolve() is True
    assert first.antenna_names == ("ea01", "ea02", "ea03")
    assert first.time_is_monotonic is True
    assert first.frequency_is_monotonic is True
    restored = inventory_from_dict(json.loads(json.dumps(inventory_as_dict(first))))
    assert restored.archive_sha256 == first.archive_sha256
    assert restored.table_presence == first.table_presence
    assert restored.correlation_codes == first.correlation_codes
    assert restored.antenna_names == first.antenna_names


def test_inventory_records_missing_pointing_without_repair(tmp_path: Path) -> None:
    measurement_set = _fake_ms(tmp_path)
    import shutil

    shutil.rmtree(measurement_set / "POINTING")
    inventory = inventory_measurement_set(measurement_set)
    assert inventory.pointing_is_present() is False
    assert any("POINTING" in note for note in inventory.notes)


def test_direction_measure_refuses_omitted_reference() -> None:
    values = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="measure reference"):
        DirectionMeasure("DIRECTION", values, " ", "rad")
    with pytest.raises(ValueError, match="units"):
        DirectionMeasure("TARGET", values, "J2000", "")


def test_synthetic_pointing_table_round_trips() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    names = [column.name for column in table.columns]
    assert names == ["DIRECTION", "TARGET", "POINTING_OFFSET"]
    restored = pointing_table_from_dict(json.loads(json.dumps(pointing_table_as_dict(table))))
    np.testing.assert_array_equal(restored.time_s, table.time_s)
    np.testing.assert_array_equal(restored.antenna_id, table.antenna_id)
    np.testing.assert_allclose(
        restored.column("DIRECTION").values_rad, table.column("DIRECTION").values_rad
    )
    assert restored.table_sha256 == table.table_sha256


def test_azelgeo_offsets_are_not_converted_as_radec() -> None:
    offset = np.array([[np.deg2rad(12.0 / 60.0), np.deg2rad(-8.0 / 60.0)]], dtype=np.float64)
    column = DirectionMeasure("POINTING_OFFSET", offset, "AZELGEO", "rad")
    native = commanded_pointing_lm_rad(column, _PHASE)
    np.testing.assert_allclose(native, offset)
    sky = DirectionMeasure("DIRECTION", offset, "J2000", "rad")
    converted = commanded_pointing_lm_rad(sky, _PHASE)
    assert not np.allclose(converted, offset)


def test_all_false_on_source_is_ignored() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    object.__setattr__(table, "on_source", np.zeros(table.time_s.size, dtype=bool))
    resolved = resolve_antenna_pointing(
        table,
        np.array([0.0, 10.0], dtype=np.float64),
        np.array([0, 1], dtype=np.int32),
        selected_column="DIRECTION",
        phase_centre_rad=_PHASE,
    )
    assert bool(np.any(resolved.settled))
    assert any("ON_SOURCE" in note for note in resolved.notes)


def test_spherical_conversion_agrees_with_known_east_offset() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    commanded = commanded_pointing_lm_rad(table.column("DIRECTION"), _PHASE)
    moving = table.antenna_id == 1
    first_move = np.flatnonzero(moving)[1]
    l_rad, m_rad, _n = radec_to_lmn(
        _PHASE[0],
        _PHASE[1],
        table.column("DIRECTION").values_rad[first_move, 0],
        table.column("DIRECTION").values_rad[first_move, 1],
    )
    np.testing.assert_allclose(commanded[first_move], (l_rad, m_rad), atol=1e-12)
    assert commanded[first_move, 0] > 0.0


def test_source_at_phase_centre_appears_at_minus_commanded_l() -> None:
    delta = np.array([[np.deg2rad(3.0 / 60.0), 0.0]], dtype=np.float64)
    sky = np.zeros((1, 2), dtype=np.float64)
    beam = source_relative_lm_rad(sky, delta, offset_sign="commanded_pointing")
    np.testing.assert_allclose(beam[0, 0], -delta[0], atol=1e-15)
    opposite = source_relative_lm_rad(sky, delta, offset_sign="source_minus_pointing")
    np.testing.assert_allclose(opposite[0, 0], delta[0], atol=1e-15)
    assert not np.allclose(beam, opposite)


def test_resolve_keeps_missing_samples_invalid() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    antennas = np.array([2, 0, 1], dtype=np.int32)
    resolved = resolve_antenna_pointing(
        table,
        times,
        antennas,
        selected_column="DIRECTION",
        phase_centre_rad=_PHASE,
    )
    assert resolved.antenna_id.tolist() == [2, 0, 1]
    assert not bool(resolved.valid[2, 2])
    assert np.isnan(resolved.offset_lm_rad[2, 2]).all()
    assert resolved.role[0, 1] == AntennaPointingRole.REFERENCE.value
    assert resolved.role[3, 2] == AntennaPointingRole.MOVING.value
    assert resolved.role[1, 2] == AntennaPointingRole.TRANSITION.value
    restored = resolved_pointing_from_dict(
        json.loads(json.dumps(resolved_pointing_as_dict(resolved)))
    )
    np.testing.assert_allclose(restored.offset_lm_rad, resolved.offset_lm_rad, equal_nan=True)
    np.testing.assert_array_equal(restored.valid, resolved.valid)


def test_overlapping_pointing_intervals_fail() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    overlapping = pointing_table_from_dict(pointing_table_as_dict(table))
    object.__setattr__(overlapping, "interval_s", np.full_like(overlapping.interval_s, 15.0))
    with pytest.raises(ValueError, match="overlapping"):
        resolve_antenna_pointing(
            overlapping,
            np.array([5.0], dtype=np.float64),
            np.array([0], dtype=np.int32),
            selected_column="DIRECTION",
            phase_centre_rad=_PHASE,
        )


def test_join_rejects_extrapolation_beyond_tolerance() -> None:
    table = synthetic_holography_pointing_table(phase_centre_rad=_PHASE)
    resolved = resolve_antenna_pointing(
        table,
        np.array([100.0], dtype=np.float64),
        np.array([0], dtype=np.int32),
        selected_column="DIRECTION",
        phase_centre_rad=_PHASE,
        join_rule="nearest_within_tolerance",
        join_tolerance_s=1.0,
    )
    assert not bool(resolved.valid[0, 0])


def test_shared_offset_promotes_and_missing_holography_offset_is_refused() -> None:
    shared = promote_operator_pointing((0.01, -0.02), n_time=3, n_antenna=4)
    assert shared.shape == (3, 4, 2)
    np.testing.assert_allclose(shared[2, 3], (0.01, -0.02))
    zeros = promote_operator_pointing(None, n_time=2, n_antenna=2, allow_missing_as_zero=True)
    np.testing.assert_array_equal(zeros, np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="refuses an assumed zero"):
        promote_operator_pointing(None, n_time=2, n_antenna=2)


def test_memo195_lower_c_raster_matches_documented_extent() -> None:
    raster = Memo195LowerCRaster()
    assert raster.dense_radius_arcmin == pytest.approx(13.76)
    assert raster.sparse_radius_arcmin == pytest.approx(50.49)


def test_inventory_records_nonmonotonic_axes(tmp_path: Path) -> None:
    measurement_set = _fake_ms(tmp_path)
    (measurement_set / "MAIN" / "time.txt").write_text("30.0 10.0 20.0\n")
    inventory = inventory_measurement_set(measurement_set)
    assert inventory.time_is_monotonic is False
    assert "TIME axis is not strictly increasing" in inventory.notes


def test_committed_fixture_matches_generator() -> None:
    loaded = load_synthetic_holography_pointing_table()
    generated = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    assert loaded.table_sha256 == generated.table_sha256
    np.testing.assert_allclose(
        loaded.column("DIRECTION").values_rad,
        generated.column("DIRECTION").values_rad,
    )


def test_committed_fixture_recovers_memo195_spacings_and_roles() -> None:
    table = load_synthetic_holography_pointing_table()
    times = np.unique(table.time_s)
    resolved = resolve_antenna_pointing(
        table,
        times,
        np.array([0, 1, 2], dtype=np.int32),
        selected_column="DIRECTION",
        phase_centre_rad=_PHASE,
        settled_jump_arcmin=8.0,
    )
    audit = audit_resolved_pointing(table=table, resolved=resolved, phase_centre_rad=_PHASE)
    roles = {track.antenna_id: track.role for track in audit.tracks}
    assert roles[0] == AntennaPointingRole.REFERENCE.value
    assert roles[1] == AntennaPointingRole.MOVING.value
    assert roles[2] == AntennaPointingRole.MOVING.value
    assert audit.memo195_dense_status == "pass"
    assert audit.memo195_sparse_status == "pass"
    assert audit.raster.dense_spacing_arcmin == pytest.approx(1.72, abs=0.05)
    assert audit.raster.sparse_spacing_arcmin == pytest.approx(4.59, abs=0.05)
    assert audit.convention.status == "pass"
    assert not bool(resolved.valid[4, 1])


def test_sign_flip_and_axis_swap_fail_convention_diagnostic() -> None:
    table = synthetic_memo195_pointing_table(phase_centre_rad=_PHASE)
    convention = pointing_convention_diagnostics(table, _PHASE)
    assert convention.status == "pass"
    assert convention.commanded_residual_arcmin < 0.02
    assert convention.flipped_residual_arcmin > 1.0
    assert convention.swapped_residual_arcmin > 1.0
    flipped = pointing_table_from_dict(pointing_table_as_dict(table))
    stored = np.asarray(flipped.column("POINTING_OFFSET").values_rad)
    object.__setattr__(
        flipped.column("POINTING_OFFSET"),
        "values_rad",
        -stored,
    )
    assert pointing_convention_diagnostics(flipped, _PHASE).status == "fail"
    swapped = pointing_table_from_dict(pointing_table_as_dict(table))
    stored = np.asarray(swapped.column("POINTING_OFFSET").values_rad)
    object.__setattr__(
        swapped.column("POINTING_OFFSET"),
        "values_rad",
        np.column_stack((stored[:, 1], stored[:, 0])),
    )
    assert pointing_convention_diagnostics(swapped, _PHASE).status == "fail"


_HOLO_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)
_HOLO_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
_HOLO_JONES = np.array(
    [[1.0 + 0.0j, 0.08 - 0.02j], [0.05 + 0.03j, 0.9 + 0.0j]],
    dtype=np.complex128,
)
_HOLO_GRAD = np.array(
    [[0.4 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.3 + 0.0j]],
    dtype=np.complex128,
)


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


def _resolved_offsets(
    unique_time_s: np.ndarray,
    offset_lm_rad: np.ndarray,
    valid: np.ndarray | None = None,
    role: np.ndarray | None = None,
) -> ResolvedAntennaPointing:
    n_time, n_ant, _ = offset_lm_rad.shape
    if valid is None:
        valid = np.isfinite(offset_lm_rad).all(axis=-1)
    if role is None:
        role = np.full((n_time, n_ant), AntennaPointingRole.UNKNOWN.value, dtype="U16")
        role[:, 0] = AntennaPointingRole.REFERENCE.value
        role[:, 1:] = AntennaPointingRole.MOVING.value
    return ResolvedAntennaPointing(
        unique_time_s=unique_time_s,
        antenna_id=np.arange(n_ant, dtype=np.int32),
        offset_lm_rad=offset_lm_rad,
        valid=valid,
        settled=valid,
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )


def test_identity_beam_pointing_does_not_add_fourier_phase() -> None:
    time_s = np.array([0.0, 0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 0], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
        uvw_m=np.array([[30.0, -12.0, 8.0], [30.0, -12.0, 8.0]]),
    )
    beam = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128))
    sky = SkyStokesPlanes(stokes_i=np.array([1.0]))
    zero = np.zeros((1, 3, 2), dtype=np.float64)
    moved = np.zeros((1, 3, 2), dtype=np.float64)
    moved[0, 1] = (np.deg2rad(4.0 / 60.0), 0.0)
    kwargs = dict(
        block=block,
        l_rad=np.array([0.0]),
        m_rad=np.array([0.0]),
        sky=sky,
        beam=beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
    )
    unmoved = predict_voltage_beam(**kwargs, antenna_pointing_lm_rad=zero)
    pointed = predict_voltage_beam(**kwargs, antenna_pointing_lm_rad=moved)
    np.testing.assert_allclose(unmoved.visibility, pointed.visibility, atol=1e-12)


def test_common_offsets_match_ordinary_mosaic_operator() -> None:
    time_s = np.array([0.0, 0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    sky = SkyStokesPlanes(stokes_i=np.array([1.1]))
    delta = (np.deg2rad(2.0 / 60.0), np.deg2rad(-1.0 / 60.0))
    shared = predict_voltage_beam(
        block,
        np.array([0.01]),
        np.array([-0.004]),
        sky,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(pointing_offset_lm_rad=delta),
    )
    per_antenna = np.broadcast_to(np.asarray(delta), (1, 3, 2)).copy()
    holography = predict_voltage_beam(
        block,
        np.array([0.01]),
        np.array([-0.004]),
        sky,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=per_antenna,
    )
    np.testing.assert_allclose(shared.visibility, holography.visibility, atol=1e-12)


def test_hand_calculated_full_jones_and_antenna_swap() -> None:
    time_s = np.array([0.0, 0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 0], dtype=np.int32),
        uvw_m=np.zeros((2, 3), dtype=np.float64),
    )
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    delta = np.deg2rad(3.0 / 60.0)
    offsets = np.zeros((1, 2, 2), dtype=np.float64)
    offsets[0, 0] = (delta, 0.0)
    sky = SkyStokesPlanes(stokes_i=np.array([1.0]))
    predicted = predict_voltage_beam(
        block,
        np.array([0.0]),
        np.array([0.0]),
        sky,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
    )
    e_moving = _HOLO_JONES + _HOLO_GRAD * (-delta)
    e_fixed = _HOLO_JONES
    coherency = circular_stokes_to_coherency(1.0, 0.0, 0.0, 0.0)
    expected_pq = unpack_coherency(
        apply_jones_to_coherency(coherency, e_moving, e_fixed),
        _HOLO_CORR,
        (Receptor.R, Receptor.L),
    )
    expected_qp = unpack_coherency(
        apply_jones_to_coherency(coherency, e_fixed, e_moving),
        _HOLO_CORR,
        (Receptor.R, Receptor.L),
    )
    np.testing.assert_allclose(predicted.visibility[0, 0], expected_pq, atol=1e-12)
    np.testing.assert_allclose(predicted.visibility[1, 0], expected_qp, atol=1e-12)
    matrix_pq = apply_jones_to_coherency(coherency, e_moving, e_fixed)
    matrix_qp = apply_jones_to_coherency(coherency, e_fixed, e_moving)
    np.testing.assert_allclose(matrix_qp, np.conjugate(np.swapaxes(matrix_pq, -1, -2)))
    assert not np.allclose(predicted.visibility[0], predicted.visibility[1])


def test_hand_calculated_scalar_beam() -> None:
    intercept = np.diag([1.2, 0.8]).astype(np.complex128)
    grad = np.diag([0.5, -0.4]).astype(np.complex128)
    time_s = np.array([0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
        uvw_m=np.zeros((1, 3), dtype=np.float64),
    )
    delta = np.deg2rad(2.0 / 60.0)
    offsets = np.zeros((1, 2, 2), dtype=np.float64)
    offsets[0, 0] = (delta, 0.0)
    predicted = predict_voltage_beam(
        block,
        np.array([0.0]),
        np.array([0.0]),
        SkyStokesPlanes(stokes_i=np.array([1.0])),
        ManufacturedVoltageBeam(intercept=intercept, grad_l=grad),
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
    )
    e_moving = intercept + grad * (-delta)
    e_fixed = intercept
    expected = unpack_coherency(
        apply_jones_to_coherency(
            circular_stokes_to_coherency(1.0, 0.0, 0.0, 0.0), e_moving, e_fixed
        ),
        _HOLO_CORR,
        (Receptor.R, Receptor.L),
    )
    np.testing.assert_allclose(predicted.visibility[0, 0], expected, atol=1e-12)


def test_parallactic_rotation_keeps_unequal_antenna_pointing() -> None:
    time_s = np.array([5.0e9, 5.0e9])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 1], dtype=np.int32),
        antenna2=np.array([1, 0], dtype=np.int32),
        uvw_m=np.zeros((2, 3), dtype=np.float64),
    )
    offsets = np.zeros((1, 2, 2), dtype=np.float64)
    offsets[0, 0] = (np.deg2rad(3.0 / 60.0), 0.0)
    beam = ManufacturedVoltageBeam(
        intercept=_HOLO_JONES, grad_l=_HOLO_GRAD, rotate_parallactic=True
    )
    predicted = predict_voltage_beam(
        block,
        np.array([0.0]),
        np.array([0.0]),
        SkyStokesPlanes(stokes_i=np.array([1.0])),
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
    )
    unmoved = predict_voltage_beam(
        block,
        np.array([0.0]),
        np.array([0.0]),
        SkyStokesPlanes(stokes_i=np.array([1.0])),
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=np.zeros_like(offsets),
    )
    assert predicted.valid.all()
    assert not np.allclose(predicted.visibility, unmoved.visibility)
    np.testing.assert_allclose(
        predicted.visibility[1],
        np.conjugate(predicted.visibility[0][..., (0, 2, 1, 3)]),
        atol=1e-12,
    )


def test_invalid_pointing_masks_either_baseline_antenna() -> None:
    time_s = np.array([0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0], dtype=np.int32),
        antenna2=np.array([1], dtype=np.int32),
    )
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES)
    offsets = np.zeros((1, 2, 2), dtype=np.float64)
    valid = np.ones((1, 2), dtype=bool)
    valid[0, 1] = False
    result = predict_voltage_beam(
        block,
        np.array([0.0]),
        np.array([0.0]),
        SkyStokesPlanes(stokes_i=np.array([1.0])),
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        antenna_pointing_lm_rad=offsets,
        pointing_valid=valid,
    )
    assert not bool(result.valid[0, 0])
    np.testing.assert_allclose(result.visibility[0], 0.0, atol=1e-15)


def test_holography_wrapper_selects_moving_reference_and_resolved_source() -> None:
    time_s = np.array([0.0, 0.0, 0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 1, 1], dtype=np.int32),
        antenna2=np.array([1, 0, 2], dtype=np.int32),
    )
    offsets = np.zeros((1, 3, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(3.0 / 60.0), 0.0)
    pointing = _resolved_offsets(np.array([0.0]), offsets)
    beam = ManufacturedVoltageBeam(intercept=_HOLO_JONES, grad_l=_HOLO_GRAD)
    point = predict_holography_visibilities(
        block,
        pointing,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        moving_reference_only=True,
    )
    assert bool(point.moving_reference_rows[0])
    assert bool(point.moving_reference_rows[1])
    assert not bool(point.moving_reference_rows[2])
    np.testing.assert_allclose(point.visibility[2], 0.0, atol=1e-15)
    source = np.zeros((3, 1, 2, 2), dtype=np.complex128)
    source[:] = circular_stokes_to_coherency(1.0, 0.0, 0.0, 0.0)
    source[0] *= 2.0
    resolved = predict_holography_visibilities(
        block,
        pointing,
        beam,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        source_coherency_visibility=source,
        moving_reference_only=False,
    )
    assert not np.allclose(resolved.visibility[0], point.visibility[0])
    assert resolved.provenance["source_model"] == "resolved_coherency_visibility"
    assert point.provenance["source_model"] == "phase_centre_sky"
    assert point.row_reason is not None
    assert point.row_reason[2] == "not_moving_reference"
    planes, valid = pointing_planes_for_block(pointing, np.array([0.0]), 3)
    np.testing.assert_allclose(planes[0, 1], offsets[0, 1])
    assert bool(valid[0, 0])


def test_row_reasons_distinguish_missing_and_unsettled() -> None:
    time_s = np.array([0.0, 0.0])
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 0], dtype=np.int32),
        antenna2=np.array([1, 2], dtype=np.int32),
    )
    offsets = np.zeros((1, 3, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(3.0 / 60.0), 0.0)
    valid = np.ones((1, 3), dtype=bool)
    valid[0, 2] = False
    settled = np.ones((1, 3), dtype=bool)
    settled[0, 1] = False
    role = np.full((1, 3), AntennaPointingRole.MOVING.value, dtype="U16")
    role[:, 0] = AntennaPointingRole.REFERENCE.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=np.array([0.0]),
        antenna_id=np.arange(3, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=valid,
        settled=settled,
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    result = predict_holography_visibilities(
        block,
        pointing,
        ManufacturedVoltageBeam(intercept=_HOLO_JONES),
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        mask_unsettled=True,
        moving_reference_only=False,
    )
    assert result.row_reason is not None
    assert result.row_reason[0] == HolographyRowReason.UNSETTLED.value
    assert result.row_reason[1] == HolographyRowReason.MISSING_POINTING.value


def test_observation_refuses_mismatched_times_antennas_and_phase_centre() -> None:
    observation, beam = synthetic_holography_rime_case()
    result = observation.predict(beam)
    assert result.provenance["backend"] == "numpy"
    with pytest.raises(ValueError, match="unique times"):
        HolographyObservation(
            block=observation.block,
            pointing=ResolvedAntennaPointing(
                unique_time_s=np.array([99.0, 109.0]),
                antenna_id=observation.pointing.antenna_id,
                offset_lm_rad=observation.pointing.offset_lm_rad,
                valid=observation.pointing.valid,
                settled=observation.pointing.settled,
                role=observation.pointing.role,
                selected_column=observation.pointing.selected_column,
                offset_sign=observation.pointing.offset_sign,
                join_rule=observation.pointing.join_rule,
                join_tolerance_s=0.0,
                measure_ref="J2000",
                units="rad",
            ),
            antenna_position_m=observation.antenna_position_m,
            calibration_state="casa_parang_true",
            phase_centre_rad=observation.phase_centre_rad,
        )
    slim = ResolvedAntennaPointing(
        unique_time_s=observation.pointing.unique_time_s,
        antenna_id=np.array([0, 1], dtype=np.int32),
        offset_lm_rad=observation.pointing.offset_lm_rad[:, :2],
        valid=observation.pointing.valid[:, :2],
        settled=observation.pointing.settled[:, :2],
        role=observation.pointing.role[:, :2],
        selected_column=observation.pointing.selected_column,
        offset_sign=observation.pointing.offset_sign,
        join_rule=observation.pointing.join_rule,
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    with pytest.raises(ValueError, match="no rows for antennas"):
        HolographyObservation(
            block=observation.block,
            pointing=slim,
            antenna_position_m=observation.antenna_position_m,
            calibration_state="casa_parang_true",
            phase_centre_rad=observation.phase_centre_rad,
        )
    with pytest.raises(ValueError, match="phase centre"):
        HolographyObservation(
            block=observation.block,
            pointing=observation.pointing,
            antenna_position_m=observation.antenna_position_m,
            calibration_state="casa_parang_true",
            phase_centre_rad=(0.0, 0.0),
        )


def test_committed_rime_golden_matches_numpy_operator() -> None:
    observation, beam, expected = load_synthetic_holography_rime_case()
    predicted = observation.predict(beam, moving_reference_only=False)
    np.testing.assert_allclose(predicted.visibility, expected, atol=1e-12)
    generated, generated_beam = synthetic_holography_rime_case()
    generated_vis = generated.predict(generated_beam).visibility
    np.testing.assert_allclose(generated_vis, expected, atol=1e-12)
    assert predicted.row_reason is not None
    assert set(predicted.row_reason) <= {
        "ok",
        "missing_pointing",
        "unsettled",
        "not_moving_reference",
    }


def test_observation_exposes_selected_samples_and_row_reasons() -> None:
    observation, beam, _expected = load_synthetic_holography_rime_case()
    assert observation.selected_correlations == ("RR", "RL", "LR", "LL")
    np.testing.assert_allclose(observation.selected_frequency_hz, observation.block.frequency_hz)
    assert observation.selected_spw_id == observation.block.spectral_window_id
    predicted = observation.predict(beam, moving_reference_only=False)
    np.testing.assert_array_equal(observation.row_reason(), predicted.row_reason)
    assert observation.active_row_mask().all()
    copolar = HolographyObservation(
        block=observation.block,
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        stokes_i=observation.stokes_i,
        selected_correlations=("RR", "LL"),
    )
    mask = copolar.sample_mask()
    assert mask[..., 0].all()
    assert mask[..., 3].all()
    assert not mask[..., 1].any()
    assert not mask[..., 2].any()
    with pytest.raises(ValueError, match="selected_correlations"):
        HolographyObservation(
            block=observation.block,
            pointing=observation.pointing,
            antenna_position_m=observation.antenna_position_m,
            calibration_state=observation.calibration_state,
            phase_centre_rad=observation.phase_centre_rad,
            selected_correlations=("XX",),
        )


def test_on_axis_identity_recovers_calibration_state_normalization() -> None:
    observation, _beam = synthetic_holography_rime_case()
    zeros = np.zeros_like(observation.pointing.offset_lm_rad)
    pointing = ResolvedAntennaPointing(
        unique_time_s=observation.pointing.unique_time_s,
        antenna_id=observation.pointing.antenna_id,
        offset_lm_rad=zeros,
        valid=observation.pointing.valid,
        settled=observation.pointing.settled,
        role=observation.pointing.role,
        selected_column=observation.pointing.selected_column,
        offset_sign=observation.pointing.offset_sign,
        join_rule=observation.pointing.join_rule,
        join_tolerance_s=observation.pointing.join_tolerance_s,
        measure_ref=observation.pointing.measure_ref,
        units=observation.pointing.units,
    )
    on_axis = HolographyObservation(
        block=observation.block,
        pointing=pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state="casa_parang_true",
        phase_centre_rad=observation.phase_centre_rad,
        stokes_i=1.0,
    )
    predicted = on_axis.predict(ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128)))
    expected = unpack_coherency(
        circular_stokes_to_coherency(1.0, 0.0, 0.0, 0.0),
        _HOLO_CORR,
        (Receptor.R, Receptor.L),
    )
    np.testing.assert_allclose(
        predicted.visibility,
        np.broadcast_to(expected, predicted.visibility.shape),
        atol=1e-12,
    )


def test_convention_ladder_detects_manufactured_sign_and_axis_errors() -> None:
    time_s = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 1, 0, 1], dtype=np.int32),
        antenna2=np.array([1, 2, 1, 2], dtype=np.int32),
    )
    offsets = np.zeros((2, 3, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(20.0 / 60.0), np.deg2rad(4.0 / 60.0))
    offsets[0, 2] = (np.deg2rad(8.0 / 60.0), np.deg2rad(-12.0 / 60.0))
    offsets[1, 1] = (np.deg2rad(-6.0 / 60.0), np.deg2rad(18.0 / 60.0))
    offsets[1, 2] = (np.deg2rad(14.0 / 60.0), np.deg2rad(9.0 / 60.0))
    pointing = _resolved_offsets(np.array([0.0, 10.0]), offsets)
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_HOLO_POSITIONS,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.1,
    )
    beam = ManufacturedVoltageBeam(
        intercept=_HOLO_JONES,
        grad_l=np.array(
            [[80.0 + 0.0j, 12.0 - 4.0j], [9.0 + 6.0j, -55.0 + 0.0j]],
            dtype=np.complex128,
        ),
        grad_m=np.array(
            [[18.0 + 0.0j, -22.0 + 8.0j], [14.0 - 10.0j, 30.0 + 0.0j]],
            dtype=np.complex128,
        ),
    )
    target = observation.predict(beam).visibility
    ladder = score_holography_convention_ladder(observation, beam, target)
    assert ladder.status == "pass"
    assert ladder.winner == "commanded_pointing"
    truth = ladder.score("commanded_pointing")
    assert truth.holdout_mse < 1e-12
    assert truth.rr_ll_mse < 1e-12
    assert truth.rl_lr_mse < 1e-12
    for name in (
        "source_minus_pointing",
        "flip_l",
        "flip_m",
        "swap_lm",
        "conjugate_visibility",
        "exchange_rl",
        "exchange_receptors",
        "swap_baseline_order",
    ):
        assert ladder.score(name).holdout_mse > 1e-4
    with pytest.raises(KeyError):
        ladder.score("flip_chi")


def test_convention_ladder_detects_flipped_parallactic_sign() -> None:
    time_s = np.array([0.0, 10.0], dtype=np.float64)
    block = _holo_block(
        time_s=time_s,
        antenna1=np.array([0, 0], dtype=np.int32),
        antenna2=np.array([1, 1], dtype=np.int32),
    )
    offsets = np.zeros((2, 2, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(20.0 / 60.0), 0.0)
    offsets[1, 1] = (0.0, np.deg2rad(20.0 / 60.0))
    pointing = _resolved_offsets(np.array([0.0, 10.0]), offsets)
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_HOLO_POSITIONS[:2],
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.0,
    )
    beam = ManufacturedVoltageBeam(
        intercept=_HOLO_JONES,
        grad_l=_HOLO_GRAD,
        rotate_parallactic=True,
    )
    target = observation.predict(beam).visibility
    ladder = score_holography_convention_ladder(
        observation,
        beam,
        target,
        hypotheses=(
            HolographyConventionHypothesis(name="commanded_pointing"),
            HolographyConventionHypothesis(name="flip_chi", chi_sign=-1),
        ),
    )
    assert ladder.winner == "commanded_pointing"
    assert ladder.score("commanded_pointing").holdout_mse < 1e-12
    assert ladder.score("flip_chi").holdout_mse > 1e-4


def test_direct_beam_comparison_does_not_freeze() -> None:
    observation, manufactured, expected = load_synthetic_holography_rime_case()
    beams = holography_comparison_beams()
    assert "analytic_airy_diagonal" in beams
    assert "perley_scalar_copolar" in beams
    assert "cassbeam_diagonal" in beams
    assert "cassbeam_experimental_full_jones" in beams
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode("full_jones")
    beams = dict(beams)
    beams["manufactured_full_jones"] = manufactured
    comparison = compare_holography_beam_predictions(observation, beams, expected)
    assert comparison.frozen is False
    assert comparison.status == "not_run"
    assert comparison.score("cassbeam_experimental_full_jones").experimental is True
    assert comparison.score("manufactured_full_jones").holdout_mse < 1e-12
    assert comparison.score("analytic_airy_diagonal").holdout_mse > 1e-4
    assert np.isfinite(comparison.score("perley_scalar_copolar").rr_ll_mse)
    assert np.isfinite(comparison.score("cassbeam_diagonal").rl_lr_mse)


def test_raster_tracks_and_plot(tmp_path: Path) -> None:
    observation, _beam, _expected = load_synthetic_holography_rime_case()
    tracks = holography_raster_tracks(observation.pointing)
    roles = {track.antenna_id: track.role for track in tracks}
    assert roles[0] == AntennaPointingRole.REFERENCE.value
    assert roles[1] == AntennaPointingRole.MOVING.value
    moving = next(track for track in tracks if track.antenna_id == 1)
    assert moving.offset_lm_arcmin.shape[0] == 2
    path = write_holography_raster_track_plot(
        observation.pointing,
        tmp_path / "holography_tracks.png",
    )
    assert path.exists()
    assert path.stat().st_size > 0


def test_three_c147_point_source_is_versioned_and_uvw_independent() -> None:
    frequency = np.array([4.564e9])
    flux = perley_butler_2017_3c147_stokes_i_jy(frequency)
    assert float(flux[0]) == pytest.approx(PERLEY_BUTLER_2017_3C147_TABLE5_JY_AT_4P564GHZ, abs=1e-5)
    report = three_c147_flux_scale_report(frequency)
    assert report["table5_coefficients"] == [1.4516, -0.6961, -0.2007, 0.0640, -0.0464, 0.0289]
    assert report["casa_setjy_model"] == "3C147_C.im"
    assert report["casa_setjy_model_data_jy"][0] == pytest.approx(
        CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32
    )
    assert report["absolute_flux_gate_material"] is True
    assert report["relative_beam_shape_harmless"] is True
    casa = three_c147_casa_setjy_point_source_model(
        frequency, np.array([CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32])
    )
    assert casa.standard == "Perley-Butler 2017 3C147_C.im"
    assert float(casa.stokes_i_jy[0]) == pytest.approx(8.028518676757812)
    packed = np.zeros((2, 1, 4), dtype=np.complex128)
    packed[0, 0] = (8.0 + 0.1j, 0.2 + 0.3j, 0.2 - 0.3j, 7.5 - 0.1j)
    packed[1, 0] = (6.0, 0.0, 0.0, 6.0)
    source = circular_visibility_to_source_coherency(
        packed, (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    )
    assert source.shape == (2, 1, 2, 2)
    assert source[0, 0, 0, 0] == packed[0, 0, 0]
    assert source[0, 0, 0, 1] == packed[0, 0, 1]
    assert source[0, 0, 1, 0] == packed[0, 0, 2]
    assert source[0, 0, 1, 1] == packed[0, 0, 3]
    assert source[1, 0, 0, 0] == 6.0
    fluxd = casa_setjy_fluxd_jy(
        {
            "fields": {
                "0": {
                    "0": {
                        "4": {"fluxd": [8.141902923583984, 0.0, 0.0, 0.0]},
                        "5": {"fluxd": [7.931094169616699, 0.0, 0.0, 0.0]},
                    }
                }
            }
        },
        field_id=0,
        spectral_window_id=5,
    )
    assert fluxd == pytest.approx(7.931094169616699)
    assert 4.0 < float(flux[0]) < 15.0
    with pytest.raises(ValueError, match="outside Perley-Butler"):
        perley_butler_2017_3c147_stokes_i_jy(1.0e6)
    model = load_three_c147_source_model(frequency)
    assert model.name == "3C147"
    assert model.kind == "point"
    assert model.standard == "Perley-Butler 2017"
    observation, beam, _expected = load_synthetic_holography_rime_case()
    bound = HolographyObservation(
        block=observation.block,
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        source_model=three_c147_point_source_model(observation.block.frequency_hz),
    )
    coherency = bound.source_model.evaluate_coherency(bound.block)
    assert coherency.shape == (4, 1, 2, 2)
    np.testing.assert_allclose(coherency[0], coherency[1], atol=1e-15)
    first = bound.predict(beam).visibility
    other_uvw = np.array(bound.block.uvw_m, copy=True)
    other_uvw *= 3.0
    from dataclasses import replace as _replace

    shifted = HolographyObservation(
        block=_replace(bound.block, uvw_m=other_uvw),
        pointing=bound.pointing,
        antenna_position_m=bound.antenna_position_m,
        calibration_state=bound.calibration_state,
        phase_centre_rad=bound.phase_centre_rad,
        source_model=bound.source_model,
    )
    np.testing.assert_allclose(shifted.predict(beam).visibility, first, atol=1e-12)


def test_compact_component_is_baseline_dependent_and_point_is_inadequate() -> None:
    observation, beam, _expected = load_synthetic_holography_rime_case()
    point = three_c147_point_source_model(observation.block.frequency_hz)
    compact = HolographySourceModel(
        name="3C147",
        kind="compact_component",
        standard="manufactured_structure",
        frequency_hz=observation.block.frequency_hz,
        stokes_i_jy=point.stokes_i_jy,
        components=(
            HolographySourceComponent(stokes_i=0.75, l_rad=0.0, m_rad=0.0),
            HolographySourceComponent(stokes_i=0.25, l_rad=2.0e-4, m_rad=-1.0e-4),
        ),
        notes=("manufactured compact structure; do not assign it to the beam",),
    )
    point_s = point.evaluate_coherency(observation.block)
    compact_s = compact.evaluate_coherency(observation.block)
    assert not np.allclose(point_s, compact_s)
    assert not np.allclose(compact_s[0], compact_s[1])
    adequate, relative = point_source_is_adequate(observation, compact, beam)
    assert adequate is False
    assert relative > 1.0e-3


def test_named_holdouts_are_deterministic() -> None:
    observation, _beam, _expected = load_synthetic_holography_rime_case()
    time_a = holography_holdout_split(observation, HolographyHoldoutAxis.TIME)
    time_b = holography_holdout_split(observation, HolographyHoldoutAxis.TIME)
    np.testing.assert_array_equal(time_a.train_row_mask, time_b.train_row_mask)
    assert not np.any(time_a.train_row_mask & time_a.holdout_row_mask)
    spatial = holography_holdout_split(observation, HolographyHoldoutAxis.SPATIAL)
    assert spatial.train_row_mask.any() and spatial.holdout_row_mask.any()
    two_ref, _beam = _two_reference_observation()
    moving = holography_holdout_split(
        two_ref, HolographyHoldoutAxis.MOVING_ANTENNA, holdout_antenna_id=1
    )
    assert moving.holdout_row_mask.any()
    assert moving.train_row_mask.any()
    reference = holography_holdout_split(
        two_ref, HolographyHoldoutAxis.REFERENCE_ANTENNA, holdout_antenna_id=0
    )
    assert reference.train_row_mask.any() and reference.holdout_row_mask.any()
    correlation = holography_holdout_split(observation, HolographyHoldoutAxis.CORRELATION)
    assert correlation.train_correlation_mask is not None
    assert bool(correlation.train_correlation_mask[0])
    assert bool(correlation.holdout_correlation_mask[1])
    two_chan = _two_channel_observation(observation)
    frequency = holography_holdout_split(two_chan, HolographyHoldoutAxis.FREQUENCY)
    assert frequency.train_channel_mask is not None
    assert int(frequency.train_channel_mask.sum()) == 1
    sealed = holography_holdout_split(observation, HolographyHoldoutAxis.TIME, sealed=True)
    with pytest.raises(ValueError, match="sealed holdout"):
        score_holography_convention_ladder(
            observation,
            ManufacturedVoltageBeam(intercept=_HOLO_JONES),
            observation.block.visibility,
            split=sealed,
        )


def test_reference_antenna_holdout_keeps_convention_winner() -> None:
    observation, beam = _two_reference_observation()
    target = observation.predict(beam).visibility
    first = score_holography_convention_ladder(
        observation,
        beam,
        target,
        split=holography_holdout_split(
            observation, HolographyHoldoutAxis.REFERENCE_ANTENNA, holdout_antenna_id=0
        ),
        hypotheses=(
            HolographyConventionHypothesis(name="commanded_pointing"),
            HolographyConventionHypothesis(name="flip_l", flip_l=True),
        ),
    )
    second = score_holography_convention_ladder(
        observation,
        beam,
        target,
        split=holography_holdout_split(
            observation, HolographyHoldoutAxis.REFERENCE_ANTENNA, holdout_antenna_id=3
        ),
        hypotheses=(
            HolographyConventionHypothesis(name="commanded_pointing"),
            HolographyConventionHypothesis(name="flip_l", flip_l=True),
        ),
    )
    assert first.winner == "commanded_pointing"
    assert second.winner == first.winner
    assert first.score("flip_l").holdout_mse > 1e-4
    assert second.score("flip_l").holdout_mse > 1e-4


def test_metadata_fixture_has_no_visibilities() -> None:
    fixture = load_synthetic_holography_metadata_fixture()
    assert fixture.source_name == "3C147"
    assert fixture.pointing.column("DIRECTION")
    assert fixture.pointing.column("TARGET")
    assert fixture.pointing.column("POINTING_OFFSET")
    assert "ea01" in fixture.antenna_names
    assert fixture.unique_time_s.size >= 2
    assert not hasattr(fixture, "visibility")


def _two_channel_observation(observation: HolographyObservation) -> HolographyObservation:
    from dataclasses import replace as _replace

    frequency = np.array([4.564e9, 4.662e9])
    n_row = observation.block.time_s.shape[0]
    dummy = np.zeros((n_row, 2, 4), dtype=np.complex128)
    block = _replace(
        observation.block,
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
    )
    return HolographyObservation(
        block=block,
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        stokes_i=observation.stokes_i,
    )


def _two_reference_observation() -> tuple[HolographyObservation, ManufacturedVoltageBeam]:
    time_s = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    block = VisibilityBlock(
        uvw_m=np.array(
            [[30.0, -12.0, 8.0], [18.0, 22.0, -4.0], [30.0, -12.0, 8.0], [18.0, 22.0, -4.0]]
        ),
        frequency_hz=np.array([4.564e9]),
        visibility=np.zeros((4, 1, 4), dtype=np.complex128),
        weight=np.ones((4, 1, 4), dtype=np.float64),
        flag=np.zeros((4, 1, 4), dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 3, 0, 3], dtype=np.int32),
        antenna2=np.array([1, 1, 2, 2], dtype=np.int32),
        correlations=_HOLO_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    offsets = np.zeros((2, 4, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(20.0 / 60.0), np.deg2rad(4.0 / 60.0))
    offsets[1, 1] = (np.deg2rad(-6.0 / 60.0), np.deg2rad(18.0 / 60.0))
    offsets[0, 2] = (np.deg2rad(12.0 / 60.0), np.deg2rad(-8.0 / 60.0))
    offsets[1, 2] = (np.deg2rad(16.0 / 60.0), np.deg2rad(10.0 / 60.0))
    role = np.full((2, 4), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, 1] = AntennaPointingRole.MOVING.value
    role[:, 2] = AntennaPointingRole.MOVING.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=np.array([0.0, 10.0]),
        antenna_id=np.arange(4, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((2, 4), dtype=bool),
        settled=np.ones((2, 4), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    positions = np.vstack(
        (
            _HOLO_POSITIONS,
            np.array([[-1_601_250.0, -5_042_250.0, 3_554_050.0]]),
        )
    )
    beam = ManufacturedVoltageBeam(
        intercept=_HOLO_JONES,
        grad_l=np.array(
            [[80.0 + 0.0j, 12.0 - 4.0j], [9.0 + 6.0j, -55.0 + 0.0j]],
            dtype=np.complex128,
        ),
        grad_m=np.array(
            [[18.0 + 0.0j, -22.0 + 8.0j], [14.0 - 10.0j, 30.0 + 0.0j]],
            dtype=np.complex128,
        ),
    )
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.1,
    )
    return observation, beam


def test_named_frames_keep_commanded_plus_l_as_source_minus_l() -> None:
    plus_l = np.deg2rad(1.72 / 60.0)
    sky = holography_phase_centre_sky(0.0, 0.0)
    pointing = holography_commanded_pointing((plus_l, 0.0))
    assert pointing.frame == "commanded_pointing"
    assert pointing.offset_sign == "commanded_pointing"
    relative = holography_source_relative_direction(sky, pointing)
    assert relative.frame == "source_relative"
    np.testing.assert_allclose(relative.l_rad, -plus_l)
    np.testing.assert_allclose(relative.m_rad, 0.0)
    feed = holography_feed_frame_direction(relative, 0.5 * np.pi)
    assert feed.frame == "feed_frame"
    np.testing.assert_allclose(feed.l_rad, 0.0, atol=1e-15)
    np.testing.assert_allclose(feed.m_rad, plus_l, atol=1e-15)
    with pytest.raises(ValueError, match="commanded_pointing"):
        holography_source_relative_direction(
            sky,
            holography_phase_centre_sky(plus_l, 0.0),
        )


def test_residual_report_separates_amp_phase_onaxis_and_closure() -> None:
    observation, beam, expected = load_synthetic_holography_rime_case()
    report = report_holography_residuals(observation, beam, expected)
    assert report.frozen is False
    assert report.gates["visibility"] == "pass"
    assert report.overall.complex_mse < 1e-12
    assert report.holdout.amplitude_rmse < 1e-8
    assert report.holdout.phase_rmse_rad < 1e-8
    assert "RR" in report.by_correlation
    assert "RL" in report.by_correlation
    assert report.closure.n_moving_reference == 2
    assert report.closure.n_moving_moving == 2
    assert report.closure.moving_reference_mse < 1e-12
    assert np.isfinite(report.holdout_reduced_chi2)
    assert report.on_axis.status == "fail"
    identity = evaluate_holography_on_axis_jones(
        observation,
        ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128)),
    )
    assert identity.status == "pass"
    assert identity.identity_error < 1e-12
