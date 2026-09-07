from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.cassbeam_beam import voltage_beam_for_mode
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyObservation,
    ResolvedAntennaPointing,
)
from sl1mjax.holography_diagonal import recover_holography_diagonal
from sl1mjax.holography_full_jones import (
    ARRAY_AVERAGE_ANTENNA_ID,
    EmpiricalHolographyVoltageBeam,
    HolographyFullJonesArtifact,
    average_holography_full_jones,
    classify_one_axis_visibility_holdout,
    combine_one_axis_results,
    freeze_interpolation_support,
    holography_one_axis_holdout_masks,
    interpolate_holography_full_jones,
    interpolate_holography_full_jones_batch,
    recover_holography_full_jones,
    thin_training_rows,
)
from sl1mjax.polarization import Correlation, ReceptorBasis, circular_stokes_to_coherency
from sl1mjax.voltage_beam import beam_coordinates

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))
_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)
_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _diagonal_observation(
    *,
    n_time: int = 2,
    n_ref: int = 1,
    frequency_hz: np.ndarray | None = None,
) -> HolographyObservation:
    times = np.array([10.0 * step for step in range(n_time)], dtype=np.float64)
    n_ant = 1 + n_ref
    rows_ant1 = []
    rows_ant2 = []
    rows_time = []
    for time_s in times:
        for ref in range(n_ref):
            rows_time.append(time_s)
            rows_ant1.append(ref)
            rows_ant2.append(n_ref)
    n_row = len(rows_time)
    channels = (
        np.asarray([4.564e9], dtype=np.float64)
        if frequency_hz is None
        else np.asarray(frequency_hz, dtype=np.float64)
    )
    n_chan = int(channels.size)
    block = VisibilityBlock(
        uvw_m=np.array([[20.0 + row, -8.0, 3.0] for row in range(n_row)], dtype=np.float64),
        frequency_hz=channels,
        visibility=np.zeros((n_row, n_chan, 4), dtype=np.complex128),
        weight=np.ones((n_row, n_chan, 4), dtype=np.float64),
        flag=np.zeros((n_row, n_chan, 4), dtype=bool),
        time_s=np.asarray(rows_time, dtype=np.float64),
        antenna1=np.asarray(rows_ant1, dtype=np.int32),
        antenna2=np.asarray(rows_ant2, dtype=np.int32),
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    offsets = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    for time_index in range(n_time):
        offsets[time_index, n_ref] = (
            np.deg2rad((-12.0 + 12.0 * time_index) / 60.0),
            0.0,
        )
    role = np.full((n_time, n_ant), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, n_ref] = AntennaPointingRole.MOVING.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(n_ant, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, n_ant), dtype=bool),
        settled=np.ones((n_time, n_ant), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_POSITIONS[:n_ant],
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.3,
    )


def _grid_observation(
    *,
    n_visit: int = 2,
    n_ref: int = 2,
    n_move: int = 2,
    n_l: int = 2,
    n_m: int = 2,
    cell_rad: float = np.deg2rad(6.0 / 60.0),
) -> HolographyObservation:
    n_ant = n_ref + n_move
    times = []
    offsets = []
    for _visit in range(n_visit):
        for i_l in range(n_l):
            for i_m in range(n_m):
                times.append(10.0 * len(times))
                offset = np.array(
                    [
                        (i_l - 0.5 * (n_l - 1)) * cell_rad,
                        (i_m - 0.5 * (n_m - 1)) * cell_rad,
                    ],
                    dtype=np.float64,
                )
                offsets.append(offset)
    times = np.asarray(times, dtype=np.float64)
    n_time = int(times.size)
    rows_time = []
    rows_ant1 = []
    rows_ant2 = []
    for time_s in times:
        for mover in range(n_ref, n_ant):
            for ref in range(n_ref):
                rows_time.append(time_s)
                rows_ant1.append(ref)
                rows_ant2.append(mover)
    n_row = len(rows_time)
    scan_at_time = np.repeat(10 + 40 * np.arange(n_visit), n_l * n_m)
    time_to_scan = {
        float(time_s): int(scan) for time_s, scan in zip(times, scan_at_time, strict=True)
    }
    block = VisibilityBlock(
        uvw_m=np.array([[20.0 + row, -8.0, 3.0] for row in range(n_row)], dtype=np.float64),
        frequency_hz=np.asarray([4.564e9], dtype=np.float64),
        visibility=np.zeros((n_row, 1, 4), dtype=np.complex128),
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=np.asarray(rows_time, dtype=np.float64),
        antenna1=np.asarray(rows_ant1, dtype=np.int32),
        antenna2=np.asarray(rows_ant2, dtype=np.int32),
        scan_id=np.asarray([time_to_scan[float(time_s)] for time_s in rows_time], dtype=np.int32),
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    pointing_off = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    for time_index, offset in enumerate(offsets):
        pointing_off[time_index, n_ref:] = offset
    role = np.full((n_time, n_ant), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, n_ref:] = AntennaPointingRole.MOVING.value
    positions = np.array(
        [
            [_POSITIONS[0, 0] + 20.0 * antenna, _POSITIONS[0, 1], _POSITIONS[0, 2]]
            for antenna in range(n_ant)
        ],
        dtype=np.float64,
    )
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(n_ant, dtype=np.int32),
        offset_lm_rad=pointing_off,
        valid=np.ones((n_time, n_ant), dtype=bool),
        settled=np.ones((n_time, n_ant), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.3,
    )


_FULL = np.eye(2, dtype=np.complex128)
_FULL_GRAD = np.array(
    [[35.0 + 0.0j, 6.0 - 2.0j], [-4.0 + 3.0j, -22.0 + 0.0j]],
    dtype=np.complex128,
)


def _truth_jones(beam: ManufacturedVoltageBeam, sample) -> np.ndarray:
    evaluation = beam.evaluate(
        beam_coordinates(
            np.array([0.0]),
            np.array([0.0]),
            np.array([sample.frequency_hz]),
            parallactic_angle_rad=np.array([0.0]),
            pointing_offset_lm_rad=sample.offset_lm_rad,
        ),
        calibration_state="casa_parang_true",
    )
    return np.asarray(evaluation.jones).reshape(-1, 2, 2)[0]


def test_recovers_asymmetric_full_jones_and_stays_unfrozen() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    assert artifact.frozen is False
    assert artifact.off_diagonal_valid.all()
    for sample in artifact.samples:
        truth = _truth_jones(beam, sample)
        np.testing.assert_allclose(sample.jones, truth, atol=1e-10)
        if float(np.hypot(*sample.offset_lm_rad)) > 1.0e-15:
            assert sample.jones[0, 1] != np.conjugate(sample.jones[1, 0])
    with pytest.raises(ValueError, match="not frozen"):
        from sl1mjax.holography_full_jones import HolographyFullJonesArtifact

        HolographyFullJonesArtifact(
            samples=artifact.samples,
            jones=artifact.jones,
            valid=artifact.valid,
            off_diagonal_valid=artifact.off_diagonal_valid,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            source_model_version=artifact.source_model_version,
            reference_combination=artifact.reference_combination,
            receptor_convention=artifact.receptor_convention,
            offset_sign=artifact.offset_sign,
            frozen=True,
        )


def test_batch_interpolation_matches_scalar() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    queries = np.stack([sample.offset_lm_rad for sample in artifact.samples], axis=0)
    movers = np.array([sample.moving_antenna_id for sample in artifact.samples], dtype=np.int32)
    freqs = np.array([sample.frequency_hz for sample in artifact.samples], dtype=np.float64)
    batch, ok, _leak = interpolate_holography_full_jones_batch(
        artifact, queries, moving_antenna_id=movers, frequency_hz=freqs
    )
    assert bool(np.all(ok))
    for index, sample in enumerate(artifact.samples):
        plane, valid, _ = interpolate_holography_full_jones(
            artifact,
            sample.offset_lm_rad,
            moving_antenna_id=sample.moving_antenna_id,
            frequency_hz=sample.frequency_hz,
        )
        assert valid
        np.testing.assert_allclose(batch[index], plane, atol=1e-10)


def test_zeroing_off_diagonals_matches_diagonal_recovery() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    full = recover_holography_full_jones(observation, predicted.visibility)
    diagonal = recover_holography_diagonal(observation, predicted.visibility)
    projected = full.to_diagonal()
    np.testing.assert_allclose(projected.jones, diagonal.jones, atol=1e-10)
    assert not bool(np.any(projected.off_diagonal_valid))


def test_source_polarization_is_not_misidentified_when_s_is_known() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128))
    from dataclasses import replace

    source = np.broadcast_to(
        circular_stokes_to_coherency(1.3, 0.2, -0.1, 0.0),
        (observation.block.time_s.shape[0], 1, 2, 2),
    ).copy()
    polarised_obs = replace(
        observation,
        source_model=None,
        source_coherency_visibility=source,
        stokes_i=1.3,
    )
    predicted = polarised_obs.predict(beam)
    recovered = recover_holography_full_jones(polarised_obs, predicted.visibility)
    for sample in recovered.samples:
        np.testing.assert_allclose(sample.jones, np.eye(2), atol=1e-10)
    wrong = recover_holography_full_jones(observation, predicted.visibility)
    off = np.max(np.abs(wrong.jones[:, 0, 1]))
    assert off > 1e-3


def test_moving_as_antenna1_recovers_the_same_jones() -> None:
    from dataclasses import replace

    observation = _diagonal_observation(n_time=2, n_ref=1)
    swapped = replace(
        observation,
        block=replace(
            observation.block,
            antenna1=np.array(observation.block.antenna2, copy=True),
            antenna2=np.array(observation.block.antenna1, copy=True),
        ),
    )
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = swapped.predict(beam)
    artifact = recover_holography_full_jones(swapped, predicted.visibility)
    for sample in artifact.samples:
        np.testing.assert_allclose(sample.jones, _truth_jones(beam, sample), atol=1e-10)


def test_interpolation_does_not_invent_leakage_outside_the_raster() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    mid = observation.pointing.offset_lm_rad[1, 1]
    jones, ok, leak = interpolate_holography_full_jones(
        artifact,
        mid,
        moving_antenna_id=1,
        frequency_hz=float(observation.block.frequency_hz[0]),
    )
    assert ok and leak
    withheld = next(sample for sample in artifact.samples if np.isclose(sample.unique_time_s, 10.0))
    np.testing.assert_allclose(jones, withheld.jones, atol=1e-12)
    far, far_ok, far_leak = interpolate_holography_full_jones(
        artifact,
        (np.deg2rad(2.0), np.deg2rad(2.0)),
        moving_antenna_id=1,
        frequency_hz=float(observation.block.frequency_hz[0]),
    )
    assert far_ok is False
    assert far_leak is False
    np.testing.assert_allclose(far, 0.0)


def _two_moving_observation(*, n_time: int = 3) -> HolographyObservation:
    times = np.array([10.0 * step for step in range(n_time)], dtype=np.float64)
    rows_ant1 = []
    rows_ant2 = []
    rows_time = []
    for time_s in times:
        for moving in (1, 2):
            rows_time.append(time_s)
            rows_ant1.append(0)
            rows_ant2.append(moving)
    n_row = len(rows_time)
    block = VisibilityBlock(
        uvw_m=np.array([[20.0 + row, -8.0, 3.0] for row in range(n_row)], dtype=np.float64),
        frequency_hz=np.array([4.564e9]),
        visibility=np.zeros((n_row, 1, 4), dtype=np.complex128),
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=np.asarray(rows_time, dtype=np.float64),
        antenna1=np.asarray(rows_ant1, dtype=np.int32),
        antenna2=np.asarray(rows_ant2, dtype=np.int32),
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    offsets = np.zeros((n_time, 3, 2), dtype=np.float64)
    for time_index in range(n_time):
        offset = (np.deg2rad((-12.0 + 12.0 * time_index) / 60.0), 0.0)
        offsets[time_index, 1] = offset
        offsets[time_index, 2] = offset
    role = np.full((n_time, 3), AntennaPointingRole.MOVING.value, dtype="U16")
    role[:, 0] = AntennaPointingRole.REFERENCE.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(3, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, 3), dtype=bool),
        settled=np.ones((n_time, 3), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=_POSITIONS,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=1.3,
    )


def test_two_moving_antennas_average_to_the_same_jones() -> None:
    observation = _two_moving_observation()
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    first = [sample for sample in artifact.samples if sample.moving_antenna_id == 1]
    second = [sample for sample in artifact.samples if sample.moving_antenna_id == 2]
    assert first and second
    for left, right in zip(first, second, strict=True):
        np.testing.assert_allclose(left.jones, right.jones, atol=1e-10)
        np.testing.assert_allclose(left.jones, _truth_jones(beam, left), atol=1e-10)
    averaged = average_holography_full_jones(artifact)
    assert averaged.frozen is False
    assert averaged.reference_combination == "inverse_variance_array_average"
    assert all(sample.moving_antenna_id == ARRAY_AVERAGE_ANTENNA_ID for sample in averaged.samples)
    for sample, left in zip(averaged.samples, first, strict=True):
        np.testing.assert_allclose(sample.jones, left.jones, atol=1e-10)
    empirical = EmpiricalHolographyVoltageBeam(averaged, allow_unfrozen=True)
    replay = observation.predict(empirical)
    np.testing.assert_allclose(replay.visibility, predicted.visibility, atol=1e-9)


def test_withheld_frequency_full_jones_is_interpolated() -> None:
    frequencies = np.array([4.4e9, 4.564e9, 4.8e9])
    observation = _diagonal_observation(n_time=2, n_ref=1, frequency_hz=frequencies)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    full = recover_holography_full_jones(observation, predicted.visibility)
    kept = tuple(sample for sample in full.samples if not np.isclose(sample.frequency_hz, 4.564e9))
    withheld = HolographyFullJonesArtifact(
        samples=kept,
        jones=np.stack([sample.jones for sample in kept], axis=0),
        valid=np.ones(len(kept), dtype=bool),
        off_diagonal_valid=np.ones(len(kept), dtype=bool),
        calibration_state=full.calibration_state,
        source_name=full.source_name,
        source_model_version=full.source_model_version,
        reference_combination=full.reference_combination,
        receptor_convention=full.receptor_convention,
        offset_sign=full.offset_sign,
    )
    offset = observation.pointing.offset_lm_rad[0, 1]
    jones, ok, leak = interpolate_holography_full_jones(
        withheld,
        offset,
        moving_antenna_id=1,
        frequency_hz=4.564e9,
    )
    assert ok and leak
    truth = next(sample for sample in full.samples if np.isclose(sample.frequency_hz, 4.564e9))
    np.testing.assert_allclose(jones, truth.jones, atol=1e-10)
    _far, far_ok, far_leak = interpolate_holography_full_jones(
        withheld,
        offset,
        moving_antenna_id=1,
        frequency_hz=6.0e9,
    )
    assert far_ok is False
    assert far_leak is False


def test_flagged_rl_does_not_invent_leakage() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    visibility = np.array(predicted.visibility, copy=True)
    visibility[:, :, 1] = 12.0 + 4.0j
    from dataclasses import replace

    flagged = np.array(observation.block.flag, copy=True)
    flagged[:, :, 1] = True
    broken = replace(
        observation,
        block=replace(observation.block, flag=flagged, visibility=visibility),
    )
    artifact = recover_holography_full_jones(broken, visibility)
    assert not bool(np.any(artifact.off_diagonal_valid))
    clean = recover_holography_full_jones(observation, predicted.visibility)
    np.testing.assert_allclose(artifact.jones[:, 0, 0], clean.jones[:, 0, 0], atol=1e-10)
    np.testing.assert_allclose(artifact.jones[:, 1, 1], clean.jones[:, 1, 1], atol=1e-10)


def test_empirical_beam_repredicts_and_factory_stays_refused() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    with pytest.raises(ValueError, match="unfrozen"):
        EmpiricalHolographyVoltageBeam(artifact)
    empirical = EmpiricalHolographyVoltageBeam(artifact, allow_unfrozen=True)
    replay = observation.predict(empirical)
    np.testing.assert_allclose(replay.visibility, predicted.visibility, atol=1e-9)
    assert replay.off_diagonal_valid is not None
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode("full_jones")


def test_one_axis_holdouts_are_not_unioned() -> None:
    observation = _grid_observation(n_visit=2, n_ref=2, n_move=2, n_l=2, n_m=2)
    loro = holography_one_axis_holdout_masks(
        observation, axis="leave_one_reference_out", held_reference_id=0
    )
    lovo = holography_one_axis_holdout_masks(observation, axis="leave_one_visit_out")
    spatial = holography_one_axis_holdout_masks(observation, axis="spatial_checkerboard")
    lomo = holography_one_axis_holdout_masks(
        observation, axis="leave_one_mover_out", held_moving_id=2
    )
    assert bool(np.any(loro["train"] & lovo["holdout"]))
    assert bool(np.any(loro["train"] & spatial["holdout"]))
    assert not bool(np.any(loro["train"] & loro["holdout"]))
    assert bool(np.any(loro["held_reference"]))
    assert bool(np.any(lovo["later_visits"]))
    assert bool(np.any(spatial["spatial"]))
    assert bool(np.any(lomo["held_moving"]))
    assert int(np.sum(loro["reserved"])) < int(np.sum(loro["holdout"]))
    thinned = thin_training_rows(observation, loro["train"], max_per_cell=1)
    assert int(np.sum(thinned)) < int(np.sum(loro["train"]))
    assert bool(np.all(thinned <= loro["train"]))


def test_leave_one_visit_out_follows_scan_pass_not_main_order() -> None:
    observation = _grid_observation(n_visit=2, n_ref=2, n_move=2, n_l=2, n_m=2)
    forward = holography_one_axis_holdout_masks(observation, axis="leave_one_visit_out")
    reversed_block = replace(
        observation.block,
        antenna1=observation.block.antenna1[::-1],
        antenna2=observation.block.antenna2[::-1],
        time_s=observation.block.time_s[::-1],
        visibility=observation.block.visibility[::-1],
        weight=observation.block.weight[::-1],
        flag=observation.block.flag[::-1],
        uvw_m=observation.block.uvw_m[::-1],
        scan_id=observation.block.scan_id[::-1],
    )
    reversed_obs = replace(observation, block=reversed_block)
    backward = holography_one_axis_holdout_masks(reversed_obs, axis="leave_one_visit_out")
    assert int(np.sum(forward["holdout"])) == int(np.sum(backward["holdout"]))
    assert int(np.sum(forward["holdout"])) == int(np.sum(forward["later_visits"]))
    assert bool(np.any(forward["holdout"] & (observation.block.scan_id == 50)))
    assert not bool(np.any(forward["holdout"] & (observation.block.scan_id == 10)))


def test_frozen_support_separates_interpolation_from_extrapolation() -> None:
    spacing = 0.002
    support = freeze_interpolation_support(
        {0: np.array([[0.0, 0.0], [spacing, 0.0], [2.0 * spacing, 0.0]], dtype=np.float64)}
    )
    exact = support.classify_offset(0, [spacing, 0.0])
    interior = support.classify_offset(0, [0.5 * spacing, 0.0])
    outside = support.classify_offset(0, [4.0 * spacing, 0.0])
    missing = support.classify_offset(1, [0.0, 0.0])
    assert exact["category"] == "exact"
    assert interior["category"] == "interpolation"
    assert outside["category"] == "extrapolation"
    assert missing["category"] == "unsupported"


def test_one_axis_classifier_does_not_pool_or_blame_lomo_zeros() -> None:
    lomo = classify_one_axis_visibility_holdout(
        axis="leave_one_mover_out",
        representation="per_antenna",
        support_fraction=0.0,
        n_holdout=100,
        n_supported=0,
        n_finite=0,
        rl_improved=None,
        lr_improved=None,
        rr_ll_regression=None,
    )
    assert lomo["outcome"] == "expected_unsupported"
    assert lomo["status"] == "pass"
    starved = classify_one_axis_visibility_holdout(
        axis="leave_one_reference_out",
        representation="per_antenna",
        support_fraction=0.06,
        n_holdout=100,
        n_supported=6,
        n_finite=6,
        rl_improved=False,
        lr_improved=False,
        rr_ll_regression=False,
    )
    assert starved["outcome"] == "inconclusive_support_starved"
    passed = classify_one_axis_visibility_holdout(
        axis="leave_one_reference_out",
        representation="per_antenna",
        support_fraction=0.9,
        n_holdout=100,
        n_supported=90,
        n_finite=90,
        rl_improved=True,
        lr_improved=True,
        rr_ll_regression=False,
        supported_rl_improved=True,
        supported_lr_improved=True,
        supported_rr_ll_regression=False,
    )
    assert passed["outcome"] == "full_jones_improves_supported_holdout"
    combined = combine_one_axis_results(
        {"leave_one_reference_out": passed, "leave_one_mover_out": lomo}
    )
    assert combined["pooled_score"] is None
    assert combined["do_not_combine"] is True


def test_loro_full_versus_diagonal_does_not_pool_references() -> None:
    from sl1mjax.holography_full_jones import (
        classify_loro_full_versus_diagonal,
        combine_loro_full_versus_diagonal,
    )

    passed = classify_loro_full_versus_diagonal(
        rl_improved=True,
        lr_improved=True,
        rr_ll_regression=False,
        n_finite=200,
    )
    starved = classify_loro_full_versus_diagonal(
        rl_improved=False,
        lr_improved=False,
        rr_ll_regression=False,
        n_finite=20,
    )
    failed = classify_loro_full_versus_diagonal(
        rl_improved=False,
        lr_improved=True,
        rr_ll_regression=False,
        n_finite=200,
    )
    assert passed["status"] == "pass"
    assert starved["status"] == "inconclusive"
    assert failed["status"] == "fail"
    combined = combine_loro_full_versus_diagonal({"ea26": passed, "ea03": starved})
    assert combined["status"] == "pass"
    assert combined["pooled_score"] is None
    assert combined["full_jones_frozen"] is False
    mixed = combine_loro_full_versus_diagonal({"ea26": passed, "ea07": failed})
    assert mixed["status"] == "fail"
    assert mixed["most_important_next_artifact"] == "cassbeam_diagonal_low_order_correction"
    assert mixed["full_jones_frozen"] is False
    assert mixed["spw5_closed"] is True
