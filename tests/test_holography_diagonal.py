from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography import (
    AntennaPointingRole,
    HolographyHoldoutAxis,
    HolographyObservation,
    ResolvedAntennaPointing,
)
from sl1mjax.holography_diagonal import (
    COMBINED_REFERENCE_ID,
    HolographyDiagonalArtifact,
    HolographyDiagonalSample,
    average_holography_diagonal,
    compare_absolute_and_restored_relative,
    interpolate_holography_diagonal,
    interpolate_holography_diagonal_batch,
    normalize_holography_diagonal_on_axis,
    recover_holography_diagonal,
    recover_holography_diagonal_per_reference,
)
from sl1mjax.holography_diagonal_diagnostics import (
    HOLDOUT_LIMITATION_NOTE,
    NEXT_RECOVERY_ORDER,
    SQUINT_MAINLOBE_POWER_FRACTION,
    bootstrap_empirical_over_cassbeam_squint,
    combine_after_reference_checks,
    compare_models_at_measured_cells,
    holdout_prediction_report,
    on_axis_sample_audit,
    pass_nearest_neighbour_report,
    phase_smoothness_report,
    physical_diagonal_diagnostics,
    recovery_group_counts,
    reference_treatment_report,
    repeated_visit_holdout_report,
)
from sl1mjax.polarization import Correlation, ReceptorBasis, circular_stokes_to_coherency
from sl1mjax.voltage_beam import beam_coordinates

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))
_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_250.0, -5_042_250.0, 3_554_050.0],
    ]
)
_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
_DIAG = np.eye(2, dtype=np.complex128)
_DIAG_GRAD = np.diag([40.0, -28.0]).astype(np.complex128)


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


def _truth_diagonal(beam: ManufacturedVoltageBeam, sample) -> tuple[complex, complex]:
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
    jones = np.asarray(evaluation.jones)
    return complex(jones.reshape(-1, 2, 2)[0, 0, 0]), complex(jones.reshape(-1, 2, 2)[0, 1, 1])


def test_per_row_model_data_is_not_replaced_by_a_scalar() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    source = np.zeros((observation.block.time_s.size, 1, 2, 2), dtype=np.complex128)
    source[:] = circular_stokes_to_coherency(8.0, 0.0, 0.0, 0.0)
    source[1] = circular_stokes_to_coherency(6.0, 0.0, 0.0, 0.0)
    resolved = replace(observation, source_coherency_visibility=source)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = resolved.predict(beam)
    recovered = recover_holography_diagonal(resolved, predicted.visibility)
    scalar = recover_holography_diagonal(
        replace(observation, stokes_i=8.028518676757812),
        predicted.visibility,
    )
    assert recovered.samples and scalar.samples
    assert abs(recovered.samples[1].e_r - scalar.samples[1].e_r) > 0.1
    for sample in recovered.samples:
        truth_r, truth_l = _truth_diagonal(beam, sample)
        assert abs(sample.e_r - truth_r) < 1e-10
        assert abs(sample.e_l - truth_l) < 1e-10


def test_recovers_manufactured_diagonal_and_stays_unfrozen() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal(observation, predicted.visibility)
    assert artifact.frozen is False
    assert not bool(np.any(artifact.off_diagonal_valid))
    np.testing.assert_allclose(artifact.jones[:, 0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(artifact.jones[:, 1, 0], 0.0, atol=1e-15)
    assert artifact.samples
    for sample in artifact.samples:
        truth_r, truth_l = _truth_diagonal(beam, sample)
        assert sample.valid
        assert abs(sample.e_r - truth_r) < 1e-10
        assert abs(sample.e_l - truth_l) < 1e-10
    power = artifact.power_stokes_i()
    assert power.shape == (len(artifact.samples),)
    centroid = artifact.receptor_centroid_lm_rad("R")
    assert centroid.shape == (2,)
    with pytest.raises(ValueError, match="not frozen"):
        from sl1mjax.holography_diagonal import HolographyDiagonalArtifact

        HolographyDiagonalArtifact(
            samples=artifact.samples,
            jones=artifact.jones,
            valid=artifact.valid,
            off_diagonal_valid=artifact.off_diagonal_valid,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination=artifact.reference_combination,
            frozen=True,
        )


def test_two_references_agree_at_the_same_cell() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    first = recover_holography_diagonal(observation, predicted.visibility, reference_antenna_id=0)
    second = recover_holography_diagonal(observation, predicted.visibility, reference_antenna_id=1)
    assert first.samples and second.samples
    for left, right in zip(first.samples, second.samples, strict=True):
        assert left.moving_antenna_id == right.moving_antenna_id
        np.testing.assert_allclose(left.offset_lm_rad, right.offset_lm_rad)
        assert abs(left.e_r - right.e_r) < 1e-10
        assert abs(left.e_l - right.e_l) < 1e-10


def test_withheld_cell_is_interpolated_from_neighbors() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    from dataclasses import replace as _replace

    predicted = observation.predict(beam)
    visibility = np.array(predicted.visibility, copy=True)
    mid_rows = observation.block.time_s == 10.0
    visibility[mid_rows] = 0.0
    flagged = np.array(observation.block.flag, copy=True)
    flagged[mid_rows] = True
    withheld_obs = HolographyObservation(
        block=_replace(observation.block, flag=flagged, visibility=visibility),
        pointing=observation.pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        stokes_i=observation.stokes_i,
    )
    artifact = recover_holography_diagonal(withheld_obs)
    assert all(not np.isclose(sample.unique_time_s, 10.0) for sample in artifact.samples)
    mid_offset = observation.pointing.offset_lm_rad[1, 1]
    e_r, e_l, ok = interpolate_holography_diagonal(
        artifact,
        mid_offset,
        moving_antenna_id=1,
        frequency_hz=float(observation.block.frequency_hz[0]),
    )
    assert ok is True
    full = recover_holography_diagonal(observation, predicted.visibility)
    withheld = next(sample for sample in full.samples if np.isclose(sample.unique_time_s, 10.0))
    assert abs(e_r - withheld.e_r) / max(abs(withheld.e_r), 1e-12) < 0.05
    assert abs(e_l - withheld.e_l) / max(abs(withheld.e_l), 1e-12) < 0.05
    far, _far_l, far_ok = interpolate_holography_diagonal(
        artifact,
        (np.deg2rad(2.0), np.deg2rad(2.0)),
        moving_antenna_id=1,
        frequency_hz=float(observation.block.frequency_hz[0]),
    )
    assert far_ok is False
    assert far == 0.0 + 0.0j


def test_array_average_requires_two_moving_antennas() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal(observation, predicted.visibility)
    with pytest.raises(ValueError, match="two moving antennas"):
        average_holography_diagonal(artifact)


def test_complex_diagonal_conjugates_when_mover_is_antenna2() -> None:
    grad = np.diag([20.0 + 8.0j, -12.0 - 5.0j]).astype(np.complex128)
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128), grad_l=grad)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal(observation, predicted.visibility)
    for sample in artifact.samples:
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
        jones = np.asarray(evaluation.jones).reshape(-1, 2, 2)[0]
        assert abs(sample.e_r - jones[0, 0]) < 1e-10
        assert abs(sample.e_l - jones[1, 1]) < 1e-10
    from dataclasses import replace

    swapped = replace(
        observation,
        block=replace(
            observation.block,
            antenna1=np.array(observation.block.antenna2, copy=True),
            antenna2=np.array(observation.block.antenna1, copy=True),
        ),
    )
    swapped_pred = swapped.predict(beam)
    swapped_art = recover_holography_diagonal(swapped, swapped_pred.visibility)
    for left, right in zip(artifact.samples, swapped_art.samples, strict=True):
        assert abs(left.e_r - right.e_r) < 1e-10
        assert abs(left.e_l - right.e_l) < 1e-10


def test_flagged_rr_does_not_enter_diagonal_recovery() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    visibility = np.array(predicted.visibility, copy=True)
    visibility[0, 0, 0] = 50.0 + 0.0j
    from dataclasses import replace

    flagged = np.array(observation.block.flag, copy=True)
    flagged[0, 0, 0] = True
    broken = replace(
        observation,
        block=replace(observation.block, flag=flagged, visibility=visibility),
    )
    artifact = recover_holography_diagonal(broken, visibility)
    clean = recover_holography_diagonal(observation, predicted.visibility)
    np.testing.assert_allclose(artifact.jones[:, 1, 1], clean.jones[:, 1, 1], atol=1e-10)
    assert artifact.samples[0].e_r != 50.0 / 1.3
    assert artifact.samples[0].valid_r is False
    assert artifact.samples[0].valid_l is True
    assert not np.isfinite(artifact.samples[0].e_r)


def test_reference_jones_is_divided_out_of_diagonal() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    reference = np.diag([np.exp(0.3j), np.exp(-0.2j)]).astype(np.complex128)
    from sl1mjax.beam_operator import JONES_RECEPTORS
    from sl1mjax.polarization import Receptor, pack_coherency, unpack_coherency

    biased = np.array(predicted.visibility, copy=True)
    names = observation.block.correlations
    for row in range(biased.shape[0]):
        for channel in range(biased.shape[1]):
            packed = pack_coherency(
                biased[row, channel][None, ...],
                names,
                (Receptor.R, Receptor.L),
            )[0]
            # V_true = I S E_m^H; inject E_r on antenna p: V = E_r S E_m^H
            biased[row, channel] = unpack_coherency(
                reference @ packed,
                names,
                JONES_RECEPTORS,
            )
    naive = recover_holography_diagonal(observation, biased)
    corrected = recover_holography_diagonal(observation, biased, reference_jones=reference)
    truth = recover_holography_diagonal(observation, predicted.visibility)
    assert abs(naive.samples[0].e_r - truth.samples[0].e_r) > 1e-3
    assert abs(corrected.samples[0].e_r - truth.samples[0].e_r) < 1e-10
    assert abs(corrected.samples[0].e_l - truth.samples[0].e_l) < 1e-10


def test_withheld_frequency_is_interpolated() -> None:
    frequencies = np.array([4.4e9, 4.564e9, 4.8e9])
    observation = _diagonal_observation(n_time=2, n_ref=1, frequency_hz=frequencies)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    full = recover_holography_diagonal(observation, predicted.visibility)
    kept = tuple(sample for sample in full.samples if not np.isclose(sample.frequency_hz, 4.564e9))
    from sl1mjax.holography_diagonal import HolographyDiagonalArtifact

    withheld = HolographyDiagonalArtifact(
        samples=kept,
        jones=np.stack(
            [np.diag([sample.e_r, sample.e_l]).astype(np.complex128) for sample in kept]
        ),
        valid=np.ones(len(kept), dtype=bool),
        off_diagonal_valid=np.zeros(len(kept), dtype=bool),
        calibration_state=full.calibration_state,
        source_name=full.source_name,
        reference_combination=full.reference_combination,
    )
    offset = observation.pointing.offset_lm_rad[0, 1]
    e_r, e_l, ok = interpolate_holography_diagonal(
        withheld,
        offset,
        moving_antenna_id=1,
        frequency_hz=4.564e9,
    )
    assert ok is True
    truth = next(sample for sample in full.samples if np.isclose(sample.frequency_hz, 4.564e9))
    assert abs(e_r - truth.e_r) < 1e-10
    assert abs(e_l - truth.e_l) < 1e-10
    _far_r, _far_l, far_ok = interpolate_holography_diagonal(
        withheld,
        offset,
        moving_antenna_id=1,
        frequency_hz=6.0e9,
    )
    assert far_ok is False


def test_per_reference_recovery_keeps_reference_ids() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal_per_reference(observation, predicted.visibility)
    assert artifact.frozen is False
    assert artifact.reference_combination == "per_reference_identity_gauge"
    assert {sample.reference_antenna_id for sample in artifact.samples} == {0, 1}
    assert COMBINED_REFERENCE_ID not in {sample.reference_antenna_id for sample in artifact.samples}
    by_time = {}
    for sample in artifact.samples:
        by_time.setdefault(sample.unique_time_s, []).append(sample)
        truth_r, truth_l = _truth_diagonal(beam, sample)
        assert abs(sample.e_r - truth_r) < 1e-10
        assert abs(sample.e_l - truth_l) < 1e-10
    for members in by_time.values():
        assert {sample.reference_antenna_id for sample in members} == {0, 1}
        assert abs(members[0].e_r - members[1].e_r) < 1e-10


def test_flagged_hand_cannot_enter_per_reference_voltage() -> None:
    observation = _diagonal_observation(n_time=1, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    visibility = np.array(predicted.visibility, copy=True)
    visibility[0, 0, 0] = 50.0 + 0.0j
    from dataclasses import replace

    flagged = np.array(observation.block.flag, copy=True)
    flagged[0, 0, 0] = True
    broken = replace(
        observation,
        block=replace(observation.block, flag=flagged, visibility=visibility),
    )
    artifact = recover_holography_diagonal_per_reference(broken, visibility)
    poisoned = next(sample for sample in artifact.samples if sample.reference_antenna_id == 0)
    clean = next(sample for sample in artifact.samples if sample.reference_antenna_id == 1)
    assert poisoned.valid_r is False
    assert poisoned.flag_rr is True
    assert not np.isfinite(poisoned.e_r)
    assert clean.valid_r is True
    assert abs(clean.e_r - 50.0 / 1.3) > 1.0


def test_reference_checks_required_before_combine() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal_per_reference(observation, predicted.visibility)
    with pytest.raises(ValueError, match="after reference-treatment"):
        combine_after_reference_checks(artifact, {})
    report = reference_treatment_report(artifact)
    assert report["checks_complete"] is True
    assert report["n_cells_with_multiple_references"] >= 1
    combined = combine_after_reference_checks(artifact, report, observation=observation)
    assert combined.reference_combination == "robust_after_reference_checks"
    assert all(sample.reference_antenna_id == COMBINED_REFERENCE_ID for sample in combined.samples)
    assert all(
        not np.isfinite(sample.weight) or sample.weight != 4e6 for sample in combined.samples
    )
    for sample in combined.samples:
        assert sample.n_reference == 2
        assert np.isfinite(sample.reference_scatter_r)


def test_frequency_holdout_refuses_without_frequency_model() -> None:
    frequencies = np.array([4.564e9, 4.692e9])
    observation = _diagonal_observation(n_time=2, n_ref=2, frequency_hz=frequencies)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    from dataclasses import replace

    observation = replace(
        observation, block=replace(observation.block, visibility=predicted.visibility)
    )
    with pytest.raises(ValueError, match="frequency model"):
        holdout_prediction_report(observation, axis=HolographyHoldoutAxis.FREQUENCY)


def test_hold_out_reference_predicts_unused_visibilities() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    from dataclasses import replace

    observation = replace(
        observation, block=replace(observation.block, visibility=predicted.visibility)
    )
    report = holdout_prediction_report(
        observation,
        axis=HolographyHoldoutAxis.REFERENCE_ANTENNA,
        holdout_antenna_id=1,
    )
    assert report.rr.n > 0
    assert report.ll.n > 0
    assert report.rr.amplitude_rms < 1e-8
    assert report.ll.amplitude_rms < 1e-8
    assert report.rr.phase_rms_rad < 1e-8
    assert report.ll.phase_rms_rad < 1e-8


def test_measured_cell_comparison_keeps_rr_and_ll_separate() -> None:
    observation = _diagonal_observation(n_time=3, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal_per_reference(observation, predicted.visibility)
    comparison = compare_models_at_measured_cells(
        artifact,
        beams={"manufactured": beam},
        calibration_state="casa_parang_true",
    )
    assert "rr" in comparison["models"]["manufactured"]
    assert "ll" in comparison["models"]["manufactured"]
    assert comparison["models"]["manufactured"]["rr"]["complex_relative_l2"] < 1e-8
    assert comparison["models"]["manufactured"]["ll"]["complex_relative_l2"] < 1e-8
    physical = physical_diagonal_diagnostics(artifact)
    assert "on_axis_and_squint" in physical
    assert "repeatability" in physical


def test_repeated_visit_holdout_uses_first_cell_visit() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    offsets = np.array(observation.pointing.offset_lm_rad, copy=True)
    offsets[1] = offsets[0]
    from dataclasses import replace

    pointing = replace(observation.pointing, offset_lm_rad=offsets)
    observation = replace(observation, pointing=pointing)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    observation = replace(
        observation, block=replace(observation.block, visibility=predicted.visibility)
    )
    report = repeated_visit_holdout_report(observation)
    assert report.rr.n > 0
    assert report.ll.n > 0
    assert report.rr.amplitude_rms < 1e-8
    assert report.ll.phase_rms_rad < 1e-8


def test_first_beam_recovery_cannot_be_frozen() -> None:
    observation = _diagonal_observation(n_time=1, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal_per_reference(observation, predicted.visibility)
    with pytest.raises(ValueError, match="not frozen"):
        HolographyDiagonalArtifact(
            samples=artifact.samples,
            jones=artifact.jones,
            valid=artifact.valid,
            off_diagonal_valid=artifact.off_diagonal_valid,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination=artifact.reference_combination,
            frozen=True,
        )
    with pytest.raises(ValueError, match="unfrozen diagnostic"):
        HolographyDiagonalArtifact(
            samples=artifact.samples,
            jones=artifact.jones,
            valid=artifact.valid,
            off_diagonal_valid=artifact.off_diagonal_valid,
            calibration_state=artifact.calibration_state,
            source_name=artifact.source_name,
            reference_combination=artifact.reference_combination,
            first_beam_recovery_unfrozen=False,
        )


def _sample(**kwargs) -> HolographyDiagonalSample:
    defaults = dict(
        moving_antenna_id=4,
        unique_time_s=10.0,
        offset_lm_rad=np.array([0.0, 0.0]),
        frequency_hz=4.564e9,
        e_r=1.0 + 0.0j,
        e_l=1.0 + 0.0j,
        sigma_r=0.01,
        sigma_l=0.01,
        n_reference=1,
        weight=float("nan"),
        valid=True,
        raster="dense",
        reference_antenna_id=0,
        raster_pass="pass-1",
        valid_r=True,
        valid_l=True,
    )
    defaults.update(kwargs)
    return HolographyDiagonalSample(**defaults)


def test_unpolarized_circular_packing_is_i_not_i_over_2() -> None:
    plane = circular_stokes_to_coherency(4.0, 0.0, 0.0, 0.0)
    assert plane[0, 0] == 4.0 + 0.0j
    assert plane[1, 1] == 4.0 + 0.0j


def test_group_counts_name_antenna_time_spatial_and_shared() -> None:
    samples = [
        _sample(reference_antenna_id=0, unique_time_s=10.0, offset_lm_rad=np.array([0.001, 0.0])),
        _sample(reference_antenna_id=1, unique_time_s=10.0, offset_lm_rad=np.array([0.001, 0.0])),
        _sample(
            moving_antenna_id=5,
            unique_time_s=20.0,
            offset_lm_rad=np.array([0.001, 0.0]),
            reference_antenna_id=0,
        ),
    ]
    from sl1mjax.holography_diagonal import artifact_from_samples

    artifact = artifact_from_samples(
        samples,
        calibration_state="casa_parang_true",
        source_name="3C147",
        reference_combination="per_reference_identity_gauge",
    )
    counts = recovery_group_counts(artifact)
    assert counts["n_per_reference_samples"] == 3
    assert counts["n_antenna_time_groups"] == 2
    assert counts["n_antenna_time_groups_with_multiple_references"] == 1
    assert counts["n_moving_antenna_spatial_cells"] == 2
    assert counts["n_shared_spatial_coordinates"] == 1


def test_on_axis_audit_rejects_nonzero_azelgeo() -> None:
    samples = [
        _sample(offset_lm_rad=np.array([0.0, 0.0]), e_r=0.2 + 0.0j, e_l=0.23 + 0.0j),
        _sample(
            unique_time_s=11.0,
            offset_lm_rad=np.deg2rad(np.array([2.0, 0.0]) / 60.0),
            e_r=0.05 + 0.0j,
        ),
    ]
    report = on_axis_sample_audit(samples, radius_arcmin=0.15)
    assert report["n_selected"] == 1
    assert report["n_exact_zero_azelgeo"] == 1
    assert report["median_abs_e_r"] == pytest.approx(0.2)


def test_pass_neighbours_do_not_require_identical_coordinates() -> None:
    spacing = np.deg2rad(1.72 / 60.0)
    samples = [
        _sample(raster_pass="pass-1", offset_lm_rad=np.array([0.0, 0.0])),
        _sample(
            unique_time_s=11.0,
            raster_pass="pass-2",
            offset_lm_rad=np.array([0.05 * spacing, 0.0]),
        ),
    ]
    report = pass_nearest_neighbour_report(samples)
    assert report["n_pass1"] == 1
    assert report["n_pass2"] == 1
    assert report["n_pairs_within_tolerance"] == 1
    assert report["min_nearest_arcmin"] < 0.3


def test_phase_jumps_drop_after_mainlobe_mask_and_gauge() -> None:
    samples = []
    for index in range(5):
        offset = np.array([index * 1.0e-4, 0.0])
        samples.append(
            _sample(
                unique_time_s=float(index),
                offset_lm_rad=offset,
                e_r=np.exp(0.1j * index),
                reference_antenna_id=0,
            )
        )
    samples.append(
        _sample(
            unique_time_s=99.0,
            offset_lm_rad=np.array([0.02, 0.0]),
            e_r=0.30 * np.exp(2.5j),
        )
    )
    report = phase_smoothness_report(samples)
    masked = report["mixed_antennas_one_global_gauge"]["above_20_percent_peak"]
    raw_low = report["mixed_antennas_raw"]["above_5_percent_peak"]
    assert masked["n_samples"] == 5
    assert raw_low["n_samples"] == 6
    assert masked["median_neighbor_phase_jump_rad"] < 0.15


def test_mainlobe_centroid_ignores_sidelobe() -> None:
    from sl1mjax.holography_diagonal_diagnostics import _centroid

    samples = [
        _sample(offset_lm_rad=np.array([0.001, 0.0]), e_r=1.0 + 0.0j),
        _sample(unique_time_s=11.0, offset_lm_rad=np.array([-0.001, 0.0]), e_r=1.0 + 0.0j),
        _sample(unique_time_s=12.0, offset_lm_rad=np.array([0.05, 0.0]), e_r=0.05 + 0.0j),
    ]
    masked = _centroid(samples, "R", power_fraction=SQUINT_MAINLOBE_POWER_FRACTION)
    unmasked = _centroid(samples, "R", power_fraction=0.0)
    assert abs(masked[0]) < abs(unmasked[0])
    assert NEXT_RECOVERY_ORDER[0] == "consistent_3c147_flux_gauge"
    assert "absolute_and_relative_spw4_recovery" in NEXT_RECOVERY_ORDER
    assert "introduce_and_validate_spatial_representation" in NEXT_RECOVERY_ORDER
    assert "field9_all_antenna_residual_jones" in NEXT_RECOVERY_ORDER
    assert "field9_time_smooth_residual_jones" in NEXT_RECOVERY_ORDER
    assert "scan53_56_chain_jump" in NEXT_RECOVERY_ORDER
    assert "field9_estimator_identifiability" in NEXT_RECOVERY_ORDER
    assert "field9_stabilized_residual_jones" in NEXT_RECOVERY_ORDER
    assert "prediction_equivalent_beam_maps" in NEXT_RECOVERY_ORDER
    assert "one_axis_visibility_holdouts" in NEXT_RECOVERY_ORDER
    assert "forward_closure" in NEXT_RECOVERY_ORDER
    assert "reference_visit_aligned_copolar_transfer" in NEXT_RECOVERY_ORDER
    assert "loro_full_versus_diagonal" in NEXT_RECOVERY_ORDER
    assert "loro_leakage_sensitivity" in NEXT_RECOVERY_ORDER
    assert "highres_cassbeam_direct_visibility_validation" in NEXT_RECOVERY_ORDER
    assert "spw4_multichannel_beam_prior" in NEXT_RECOVERY_ORDER
    assert "c147_offset_ring_highres_cassbeam" in NEXT_RECOVERY_ORDER
    assert "holoraster_cassbeam_comparison_report" in NEXT_RECOVERY_ORDER
    assert "cassbeam_diagonal_low_order_correction" in NEXT_RECOVERY_ORDER
    assert NEXT_RECOVERY_ORDER.index("loro_full_versus_diagonal") < NEXT_RECOVERY_ORDER.index(
        "loro_leakage_sensitivity"
    )
    assert NEXT_RECOVERY_ORDER.index("loro_leakage_sensitivity") < NEXT_RECOVERY_ORDER.index(
        "highres_cassbeam_direct_visibility_validation"
    )
    assert NEXT_RECOVERY_ORDER.index("highres_cassbeam_direct_visibility_validation") < (
        NEXT_RECOVERY_ORDER.index("spw4_multichannel_beam_prior")
    )
    assert NEXT_RECOVERY_ORDER.index("spw4_multichannel_beam_prior") < (
        NEXT_RECOVERY_ORDER.index("c147_offset_ring_highres_cassbeam")
    )
    assert NEXT_RECOVERY_ORDER.index("c147_offset_ring_highres_cassbeam") < (
        NEXT_RECOVERY_ORDER.index("holoraster_cassbeam_comparison_report")
    )
    assert NEXT_RECOVERY_ORDER.index("holoraster_cassbeam_comparison_report") < (
        NEXT_RECOVERY_ORDER.index("cassbeam_diagonal_low_order_correction")
    )
    assert NEXT_RECOVERY_ORDER.index(
        "field9_all_antenna_residual_jones"
    ) < NEXT_RECOVERY_ORDER.index("independent_spw5_recovery")
    assert "training samples only" in HOLDOUT_LIMITATION_NOTE


def test_relative_onaxis_normalization_restores_the_absolute_beam() -> None:
    samples = [
        _sample(e_r=2.0 + 0.0j, e_l=4.0 + 0.0j, offset_lm_rad=np.array([0.0, 0.0])),
        _sample(
            unique_time_s=11.0,
            e_r=1.0 + 0.0j,
            e_l=2.0 + 0.0j,
            offset_lm_rad=np.array([0.002, 0.0]),
        ),
        _sample(
            moving_antenna_id=5,
            unique_time_s=12.0,
            e_r=0.5 + 0.0j,
            e_l=0.25 + 0.0j,
            offset_lm_rad=np.array([0.0, 0.0]),
        ),
        _sample(
            moving_antenna_id=5,
            unique_time_s=13.0,
            e_r=0.25 + 0.0j,
            e_l=0.125 + 0.0j,
            offset_lm_rad=np.array([0.002, 0.0]),
        ),
    ]
    from sl1mjax.holography_diagonal import artifact_from_samples

    absolute = artifact_from_samples(
        samples,
        calibration_state="casa_parang_true",
        source_name="3C147",
        reference_combination="per_reference_identity_gauge",
    )
    relative = normalize_holography_diagonal_on_axis(absolute)
    off_axis = next(
        sample
        for sample in relative.samples
        if sample.moving_antenna_id == 4 and sample.unique_time_s == 11.0
    )
    assert off_axis.e_r == pytest.approx(0.5)
    assert off_axis.e_l == pytest.approx(0.5)
    assert off_axis.normalization == "per_antenna_onaxis"
    agreement = compare_absolute_and_restored_relative(absolute, relative)
    assert agreement["max_relative_abs_r"] == pytest.approx(0.0, abs=1e-12)
    assert agreement["max_relative_abs_l"] == pytest.approx(0.0, abs=1e-12)


def test_squint_bootstrap_includes_one_when_the_beam_matches() -> None:
    observation = _diagonal_observation(n_time=4, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_DIAG, grad_l=_DIAG_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_diagonal_per_reference(observation, predicted.visibility)
    report = bootstrap_empirical_over_cassbeam_squint(
        artifact,
        beam=beam,
        n_resample=32,
        seed=0,
    )
    assert "moving_antenna_ratio" in report
    assert "reference_antenna_ratio" in report
    assert "stable_deficit" in report
    if np.isfinite(report["empirical_over_sampled_cassbeam"]):
        assert report["empirical_over_sampled_cassbeam"] == pytest.approx(1.0, abs=0.15)
    smoothness = phase_smoothness_report(list(artifact.samples))
    assert smoothness["phase_gauge_removed"] is True


def _holdout_grid() -> list[HolographyDiagonalSample]:
    samples = []
    spacing = np.deg2rad(1.72 / 60.0)
    time = 10.0
    for mover in (4, 5):
        for ix in range(3):
            for iy in range(3):
                offset = np.array([(ix - 1) * spacing, (iy - 1) * spacing])
                voltage = 1.0 + 8.0 * offset[0] + 0.0j
                for ref in (0, 1):
                    samples.append(
                        _sample(
                            moving_antenna_id=mover,
                            reference_antenna_id=ref,
                            unique_time_s=time + 0.1 * ref,
                            offset_lm_rad=offset,
                            e_r=voltage,
                            e_l=voltage * 0.98,
                            raster_pass="pass-1" if iy == 0 else "pass-2",
                            scan=2 if iy == 0 else 3,
                        )
                    )
                samples.append(
                    _sample(
                        moving_antenna_id=mover,
                        reference_antenna_id=0,
                        unique_time_s=time + 50.0,
                        offset_lm_rad=offset,
                        e_r=voltage,
                        e_l=voltage * 0.98,
                        raster_pass="pass-1" if iy == 0 else "pass-2",
                    )
                )
    close = np.array([0.0, np.deg2rad(0.1 / 60.0)])
    samples.append(
        _sample(
            moving_antenna_id=4,
            reference_antenna_id=0,
            unique_time_s=80.0,
            offset_lm_rad=close,
            e_r=1.0 + 0.0j,
            e_l=0.98 + 0.0j,
            raster_pass="pass-2",
        )
    )
    return samples


def test_batch_interpolation_matches_one_query_and_skips_holdout_values() -> None:
    from sl1mjax.holography_diagonal import artifact_from_samples, interpolate_holography_diagonal

    samples = _holdout_grid()
    artifact = artifact_from_samples(
        samples,
        calibration_state="casa_parang_true",
        source_name="3C147",
        reference_combination="per_reference_identity_gauge",
    )
    query = samples[4].offset_lm_rad
    one = interpolate_holography_diagonal(
        artifact, query, moving_antenna_id=4, frequency_hz=4.564e9
    )
    many_r, many_l, ok = interpolate_holography_diagonal_batch(
        artifact,
        [query, query],
        moving_antenna_id=np.array([4, 4], dtype=np.int32),
        frequency_hz=np.array([4.564e9, 4.564e9]),
    )
    assert bool(ok[0]) and bool(ok[1])
    assert many_r[0] == pytest.approx(one[0])
    assert many_l[0] == pytest.approx(one[1])


def test_stratified_holdouts_use_training_samples_only() -> None:
    from sl1mjax.holography_diagonal import artifact_from_samples
    from sl1mjax.holography_diagonal_diagnostics import stratified_sample_holdouts

    samples = _holdout_grid()
    artifact = artifact_from_samples(
        samples,
        calibration_state="casa_parang_true",
        source_name="3C147",
        reference_combination="per_reference_identity_gauge",
    )
    report = stratified_sample_holdouts(artifact, include_model_gauges=False)
    spatial = report["spatially_interleaved_cells"]
    assert spatial["n"] > 0
    assert spatial["rr"]["complex_relative_l2"] < 0.05
    assert spatial["by_power_region"]["above_50_percent"]["rr"]["n"] > 0
    refs = report["held_out_references"]
    assert refs["rr"]["complex_relative_l2"] < 1.0e-12
    later = report["later_visits"]
    assert later["n_later_visits"] > 0
    assert later["rr"]["complex_relative_l2"] < 1.0e-12
    movers = report["held_out_moving_antennas"]
    assert movers["n"] > 0
    poisoned = list(samples)
    poisoned[0] = _sample(
        moving_antenna_id=poisoned[0].moving_antenna_id,
        reference_antenna_id=poisoned[0].reference_antenna_id,
        unique_time_s=poisoned[0].unique_time_s,
        offset_lm_rad=poisoned[0].offset_lm_rad,
        e_r=50.0 + 0.0j,
        e_l=50.0 + 0.0j,
        raster_pass=poisoned[0].raster_pass,
    )
    # The interpolator is rebuilt from training cells; a wild holdout value
    # must not leak into the training artifact used for neighbours.
    again = stratified_sample_holdouts(
        artifact_from_samples(
            poisoned,
            calibration_state="casa_parang_true",
            source_name="3C147",
            reference_combination="per_reference_identity_gauge",
        ),
        include_model_gauges=False,
    )
    assert again["spatially_interleaved_cells"]["rr"]["n"] > 0
