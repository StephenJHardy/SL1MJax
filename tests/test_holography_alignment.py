from __future__ import annotations

from dataclasses import replace

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import AntennaPointingRole, HolographyObservation, ResolvedAntennaPointing
from sl1mjax.holography_alignment import (
    BEAM_REPEATABILITY,
    UNCALIBRATED_VISIT_TRANSFER,
    align_recovered_jones,
    alignment_fit_mask,
    apparent_voltage_response,
    classify_reference_visit_aligned_copolar_transfer,
    combine_alignment_transfer_gates,
    copy_visit_factors,
    estimate_mover_visit_A,
    estimate_reference_visit_B,
    holoraster_pair_masks,
    score_copolar_residuals,
    visit_holdout_masks,
    visit_id_per_time,
    voltage_response_region_masks,
)
from sl1mjax.holography_diagonal import (
    holography_row_exclusion_counts,
    source_model_stokes_i,
)
from sl1mjax.holography_forward_closure import predict_row_visibility, recover_row_jones
from sl1mjax.polarization import Correlation, ReceptorBasis

_PHASE = (np.deg2rad(84.0), np.deg2rad(50.0))
_POSITIONS = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)
_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _alignment_observation() -> HolographyObservation:
    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    rows_time = []
    rows_ant1 = []
    rows_ant2 = []
    for time_s in times:
        rows_time.extend([time_s, time_s, time_s])
        rows_ant1.extend([0, 0, 1])
        rows_ant2.extend([1, 2, 2])
    n_row = len(rows_time)
    block = VisibilityBlock(
        uvw_m=np.array([[20.0 + row, -8.0, 3.0] for row in range(n_row)], dtype=np.float64),
        frequency_hz=np.asarray([4.564e9], dtype=np.float64),
        visibility=np.zeros((n_row, 1, 4), dtype=np.complex128),
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=np.asarray(rows_time, dtype=np.float64),
        antenna1=np.asarray(rows_ant1, dtype=np.int32),
        antenna2=np.asarray(rows_ant2, dtype=np.int32),
        scan_id=np.asarray([10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50], dtype=np.int32),
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
    )
    offsets = np.zeros((4, 3, 2), dtype=np.float64)
    offsets[1, 2] = (0.002, 0.0)
    offsets[3, 2] = (0.002, 0.0)
    role = np.full((4, 3), AntennaPointingRole.REFERENCE.value, dtype="U16")
    role[:, 2] = AntennaPointingRole.MOVING.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=np.arange(3, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((4, 3), dtype=bool),
        settled=np.ones((4, 3), dtype=bool),
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


def test_alignment_inverts_constant_A_and_B() -> None:
    source = np.eye(2, dtype=np.complex128) * 1.3
    a_m = np.array([[1.15, 0.04 - 0.02j], [-0.03, 0.92]], dtype=np.complex128)
    b_r = np.array([[0.88, 0.03j], [-0.02 + 0.01j, 1.07]], dtype=np.complex128)
    beam = np.array([[0.62, 0.05 + 0.02j], [-0.04, 0.58]], dtype=np.complex128)
    vis = predict_row_visibility(
        a_m @ beam @ np.conjugate(b_r.T), source, np.eye(2), np.eye(2), moving_is_p=True
    )
    recovered = recover_row_jones(vis, source, np.eye(2), np.eye(2), moving_is_p=True)
    aligned = align_recovered_jones(recovered, a_m, b_r)
    np.testing.assert_allclose(aligned, beam, atol=1.0e-9)


def test_fit_uses_origin_and_ref_ref_only() -> None:
    observation = _alignment_observation()
    geometry = holoraster_pair_masks(observation)
    assert int(np.sum(geometry["reference_reference"])) == 4
    assert int(np.sum(geometry["origin"])) == 4
    allowed = alignment_fit_mask(observation)
    assert bool(np.all(allowed == geometry["alignment_rows"]))
    assert not bool(np.any(geometry["moving_reference"] & ~geometry["origin"] & allowed))


def test_two_visit_holdout_meanings_differ() -> None:
    observation = _alignment_observation()
    visits = visit_id_per_time(observation)
    uncal = visit_holdout_masks(
        observation,
        kind=UNCALIBRATED_VISIT_TRANSFER,
        visit_id=visits,
        held_visit=1,
        reserve_outer_fold=False,
    )
    repeat = visit_holdout_masks(
        observation,
        kind=BEAM_REPEATABILITY,
        visit_id=visits,
        held_visit=1,
        reserve_outer_fold=False,
    )
    geometry = holoraster_pair_masks(observation)
    assert np.array_equal(visits, np.array([0, 0, 1, 1], dtype=np.int32))
    assert bool(np.any(uncal["holdout"] & geometry["origin"]))
    assert not bool(np.any(repeat["holdout"] & geometry["origin"]))
    assert bool(
        np.any(repeat["train"] & geometry["origin"] & (visits[geometry["time_index"]] == 1))
    )
    assert bool(np.all(repeat["alignment_exclude"] <= geometry["moving_reference"]))
    assert not bool(np.any(repeat["alignment_exclude"] & geometry["origin"]))


def test_classifier_requires_few_percent_floor() -> None:
    failed = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.42,
        unaligned_ll=0.40,
        aligned_rr=0.41,
        aligned_ll=0.39,
        direction_dependent=False,
    )
    assert failed["status"] == "fail"
    assert failed["full_jones_blocked"] is True
    assert failed["comparison_fair"] is False
    passed = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.42,
        unaligned_ll=0.40,
        aligned_rr=0.03,
        aligned_ll=0.02,
        direction_dependent=False,
    )
    assert passed["status"] == "pass"
    assert passed["comparison_fair"] is True
    assert passed["full_jones_blocked"] is True
    starved = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.03,
        unaligned_ll=0.02,
        aligned_rr=0.03,
        aligned_ll=0.02,
        direction_dependent=False,
        n_finite=33,
    )
    assert starved["status"] == "fail"
    assert starved["outcome"] == "insufficient_exact_cells"
    uncal = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.03,
        unaligned_ll=0.03,
        aligned_rr=0.03,
        aligned_ll=0.03,
        direction_dependent=False,
        n_finite=33,
        availability="insufficient_evidence",
    )
    assert uncal["status"] == "inconclusive"
    assert uncal["outcome"] == "inconclusive"
    repeat = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=float("nan"),
        unaligned_ll=float("nan"),
        aligned_rr=float("nan"),
        aligned_ll=float("nan"),
        direction_dependent=None,
        n_finite=0,
        availability="not_testable",
    )
    assert repeat["status"] == "inconclusive"
    assert repeat["outcome"] == "not_testable_on_this_dataset"


def test_injected_factors_are_recovered_from_allowed_rows() -> None:
    observation = _alignment_observation()
    source = np.eye(2, dtype=np.complex128) * 1.3
    identity = np.eye(2, dtype=np.complex128)
    a0 = np.array([[1.12, 0.03j], [0.0, 0.94]], dtype=np.complex128)
    a1 = np.array([[0.97, -0.02], [0.01, 1.08]], dtype=np.complex128)
    b1 = np.array([[0.9, 0.04], [-0.03, 1.05]], dtype=np.complex128)
    beam0 = identity
    beam1 = np.array([[0.7, 0.05], [-0.04, 0.65]], dtype=np.complex128)
    packed = np.zeros((12, 2, 2), dtype=np.complex128)
    for a_m, off_axis, start in ((a0, beam0, 0), (a1, beam1, 6)):
        packed[start] = predict_row_visibility(identity, source, identity, b1, moving_is_p=True)
        packed[start + 1] = predict_row_visibility(
            identity, source, a_m, identity, moving_is_p=False
        )
        packed[start + 2] = predict_row_visibility(identity, source, a_m, b1, moving_is_p=False)
        packed[start + 3] = predict_row_visibility(identity, source, identity, b1, moving_is_p=True)
        packed[start + 4] = predict_row_visibility(
            off_axis, source, a_m, identity, moving_is_p=False
        )
        packed[start + 5] = predict_row_visibility(off_axis, source, a_m, b1, moving_is_p=False)
    vis = np.stack(
        [packed[:, 0, 0], packed[:, 0, 1], packed[:, 1, 0], packed[:, 1, 1]],
        axis=-1,
    )[:, None, :]
    observation = replace(observation, block=replace(observation.block, visibility=vis))
    residual = {0: identity, 1: identity, 2: identity}
    chi = np.zeros((4, 3), dtype=np.float64)
    visits = visit_id_per_time(observation)
    allowed = alignment_fit_mask(observation)
    b_fit = estimate_reference_visit_B(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        reference_antenna_id=0,
        visit_id=visits,
        row_mask=allowed,
    )
    a_fit = estimate_mover_visit_A(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        visit_id=visits,
        reference_factors=b_fit,
        row_mask=allowed,
    )
    np.testing.assert_allclose(b_fit[(0, 1)], b1, atol=2.0e-6)
    np.testing.assert_allclose(a_fit[(0, 2)], a0, atol=2.0e-6)
    np.testing.assert_allclose(a_fit[(1, 2)], a1, atol=2.0e-6)
    off_axis = (
        holoraster_pair_masks(observation)["moving_reference"]
        & ~holoraster_pair_masks(observation)["origin"]
    )
    assert not bool(np.any(allowed & off_axis))
    transferred = copy_visit_factors(a_fit, source_visit=0, dest_visit=1)
    np.testing.assert_allclose(transferred[(1, 2)], a1, atol=2.0e-6)
    missing = copy_visit_factors({(0, 2): a0}, source_visit=0, dest_visit=1)
    np.testing.assert_allclose(missing[(1, 2)], a0, atol=1.0e-12)


def test_combined_gate_passes_on_available_loro() -> None:
    loro = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.42,
        unaligned_ll=0.40,
        aligned_rr=0.03,
        aligned_ll=0.02,
        direction_dependent=False,
    )
    repeat = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=float("nan"),
        unaligned_ll=float("nan"),
        aligned_rr=float("nan"),
        aligned_ll=float("nan"),
        direction_dependent=None,
        n_finite=0,
        availability="not_testable",
    )
    uncal = classify_reference_visit_aligned_copolar_transfer(
        unaligned_rr=0.03,
        unaligned_ll=0.03,
        aligned_rr=0.03,
        aligned_ll=0.03,
        direction_dependent=False,
        n_finite=33,
        availability="insufficient_evidence",
    )
    combined = combine_alignment_transfer_gates(
        leave_one_reference_out=loro,
        beam_repeatability=repeat,
        uncalibrated_visit_transfer=uncal,
    )
    assert combined["status"] == "pass"
    assert combined["comparison_fair"] is True
    assert combined["full_jones_blocked"] is True
    assert combined["full_jones_block_reason"] == "missing_fair_rl_lr_test"
    assert combined["most_important_next_artifact"] == "spw4_multichannel_beam_prior"
    assert uncal["status"] == "inconclusive"
    assert repeat["outcome"] == "not_testable_on_this_dataset"


def test_source_model_stokes_i_ignores_measured_beam_attenuation() -> None:
    observation = _alignment_observation()
    measured = np.full(observation.block.visibility.shape, 0.13 + 0.0j, dtype=np.complex128)
    observation = replace(
        observation,
        block=replace(observation.block, visibility=measured),
        stokes_i=1.3,
    )
    intensity = source_model_stokes_i(observation)
    np.testing.assert_allclose(intensity, 1.3)
    assert float(np.max(np.abs(measured))) < 0.2


def test_flagged_and_zero_weight_rows_are_excluded_from_alignment() -> None:
    observation = _alignment_observation()
    flag = np.zeros(observation.block.flag.shape, dtype=bool)
    flag[1] = True
    weight = np.ones(observation.block.weight.shape, dtype=np.float64)
    weight[2] = 0.0
    observation = replace(
        observation,
        block=replace(observation.block, flag=flag, weight=weight),
    )
    geometry = holoraster_pair_masks(observation)
    assert not bool(geometry["four_hand"][1])
    assert not bool(geometry["four_hand"][2])
    assert not bool(geometry["alignment_rows"][1])
    assert not bool(geometry["alignment_rows"][2])
    assert not bool(alignment_fit_mask(observation)[1])
    assert not bool(alignment_fit_mask(observation)[2])
    counts = holography_row_exclusion_counts(observation)
    assert counts["n_any_flag"] >= 1
    assert counts["n_invalid_weight"] >= 1
    assert counts["exclusive"]["flagged"] >= 1


def test_polarized_source_recovers_A_when_B_is_inside_rime() -> None:
    observation = _alignment_observation()
    source = np.array([[1.3, 0.2 + 0.1j], [0.2 - 0.1j, 1.05]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    a0 = np.array([[1.12, 0.03j], [0.0, 0.94]], dtype=np.complex128)
    a1 = np.array([[0.97, -0.02], [0.01, 1.08]], dtype=np.complex128)
    b1 = np.array([[0.9, 0.04], [-0.03, 1.05]], dtype=np.complex128)
    packed = np.zeros((12, 2, 2), dtype=np.complex128)
    for a_m, start in ((a0, 0), (a1, 6)):
        packed[start] = predict_row_visibility(identity, source, identity, b1, moving_is_p=True)
        packed[start + 1] = predict_row_visibility(
            identity, source, a_m, identity, moving_is_p=False
        )
        packed[start + 2] = predict_row_visibility(identity, source, a_m, b1, moving_is_p=False)
        packed[start + 3] = predict_row_visibility(identity, source, identity, b1, moving_is_p=True)
        packed[start + 4] = predict_row_visibility(
            identity, source, a_m, identity, moving_is_p=False
        )
        packed[start + 5] = predict_row_visibility(identity, source, a_m, b1, moving_is_p=False)
    vis = np.stack(
        [packed[:, 0, 0], packed[:, 0, 1], packed[:, 1, 0], packed[:, 1, 1]],
        axis=-1,
    )[:, None, :]
    source_vis = np.broadcast_to(source, (12, 1, 2, 2)).copy()
    observation = replace(
        observation,
        block=replace(observation.block, visibility=vis),
        source_coherency_visibility=source_vis,
        stokes_i=1.3,
    )
    residual = {0: identity, 1: identity, 2: identity}
    chi = np.zeros((4, 3), dtype=np.float64)
    visits = visit_id_per_time(observation)
    allowed = alignment_fit_mask(observation)
    b_fit = estimate_reference_visit_B(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        reference_antenna_id=0,
        visit_id=visits,
        row_mask=allowed,
    )
    a_fit = estimate_mover_visit_A(
        observation,
        residual_jones=residual,
        parallactic_angle_rad=chi,
        visit_id=visits,
        reference_factors=b_fit,
        row_mask=allowed,
    )
    np.testing.assert_allclose(a_fit[(0, 2)], a0, atol=2.0e-6)
    np.testing.assert_allclose(a_fit[(1, 2)], a1, atol=2.0e-6)


def test_source_normalized_residuals_stay_flat_when_beam_falls() -> None:
    n = 6
    intensity = np.full(n, 2.0, dtype=np.float64)
    voltage = np.array([1.0, 0.7, 0.4, 0.3, 0.1, 0.05], dtype=np.float64)
    residual_jy = 0.08 + 0.0j
    measured = np.zeros((n, 2, 2), dtype=np.complex128)
    predicted = np.zeros((n, 2, 2), dtype=np.complex128)
    measured[:, 0, 0] = voltage * intensity + residual_jy
    measured[:, 1, 1] = voltage * intensity + residual_jy
    predicted[:, 0, 0] = voltage * intensity
    predicted[:, 1, 1] = voltage * intensity
    rr_ok = np.ones(n, dtype=bool)
    ll_ok = np.ones(n, dtype=bool)
    rr_ok[1] = False
    scored = score_copolar_residuals(
        measured,
        predicted,
        intensity,
        rr_ok,
        ll_ok,
        np.ones(n, dtype=bool),
        apparent_voltage_response(measured, intensity, rr_ok, ll_ok),
    )
    np.testing.assert_allclose(scored["all"]["median_abs_rr_jy"], 0.08, atol=1.0e-12)
    np.testing.assert_allclose(scored["all"]["median_abs_rr_over_i"], 0.04, atol=1.0e-12)
    np.testing.assert_allclose(scored["main_lobe"]["median_abs_rr_over_i"], 0.04, atol=1.0e-12)
    assert scored["main_lobe"]["n"] == 2
    assert scored["main_lobe"]["n_rr"] == 1
    assert scored["main_lobe"]["n_ll"] == 2
    assert scored["mid"]["n"] == 2
    assert scored["outer_diagnostic"]["n"] == 2
    regions = voltage_response_region_masks(voltage)
    assert bool(regions["main_lobe"][0])
    assert bool(regions["outer_diagnostic"][-1])
