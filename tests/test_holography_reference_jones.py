from __future__ import annotations

import numpy as np
import pytest
from test_holography_full_jones import _FULL, _FULL_GRAD, _diagonal_observation, _grid_observation

from sl1mjax.beam_operator import JONES_RECEPTORS
from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography_full_jones import (
    HolographyFullJonesArtifact,
    HolographyFullJonesSample,
    classify_holdout_interpolation_support,
    classify_prediction_equivalent_beam_maps,
    freeze_interpolation_support_from_rows,
    holography_one_axis_holdout_masks,
    holography_visibility_holdout_masks,
    mask_unsupported_off_diagonals,
    moving_reference_row_geometry,
    offdiag_beam_map_difference,
    paired_full_versus_diagonal_scores,
    pin_on_axis_jones_to_identity,
    predict_moving_reference_from_beam,
    recover_holography_full_jones,
    refactor_on_axis_gauge,
)
from sl1mjax.holography_reference_jones import (
    align_residual_jones_maps,
    antenna_graph_is_connected,
    blocked_time_folds,
    classify_chain_jump_stage,
    classify_common_mode_vs_antenna,
    classify_crosshand_floor,
    classify_estimator_identifiability,
    classify_field9_time_transfer,
    classify_interleaved_onaxis_transfer,
    classify_residual_jones_visibility,
    classify_stabilized_residual_jones,
    classify_three_c147_qu_model,
    coherent_residual_report,
    connected_baseline_holdout_mask,
    deserialize_reference_jones,
    estimate_all_antenna_residual_jones,
    estimate_residual_reference_jones,
    field9_scan_clusters,
    gauge_invariance_abs,
    held_out_scan_cluster_masks,
    hierarchical_residual_jones_model,
    identical_channel_support,
    injected_feed_leakage_rotation,
    jones_parameter_jump,
    matched_scan_row_masks,
    predict_baseline_coherency,
    predicted_operator_difference,
    reference_jones_for_antenna,
    residual_holdout_report,
    scan_state_jump_report,
    serialize_reference_jones,
    sky_frame_residual_jones,
)
from sl1mjax.polarization import Receptor, pack_coherency, unpack_coherency


def test_crosshand_floor_is_blocking_until_shown_to_be_noise() -> None:
    blocked = classify_crosshand_floor(median_abs_rl_over_i=0.027)
    assert blocked["status"] == "fail"
    assert blocked["blocking"] is True
    noise = classify_crosshand_floor(
        median_abs_rl_over_i=0.027,
        coherent_mean_abs=0.002,
        averages_as_noise=True,
    )
    assert noise["blocking"] is False
    assert noise["status"] == "warn"
    small = classify_crosshand_floor(median_abs_rl_over_i=0.005)
    assert small["status"] == "pass"
    grouped = classify_crosshand_floor(
        median_abs_rl_over_i=0.027,
        coherent_mean_abs=0.002,
        averages_as_noise=True,
        max_group_coherent_abs=0.029,
    )
    assert grouped["status"] == "fail"
    assert grouped["blocking"] is True
    assert grouped["blocks_identity_gauge"] is True
    assert grouped["blocks_residual_jones_estimator"] is False


def test_measured_three_c147_qu_is_not_a_passed_zero_model() -> None:
    exact_zero = classify_three_c147_qu_model(
        q_over_i=0.001,
        u_over_i=0.0008,
        frac_pol=0.0013,
        represented_as_exact_zero=True,
    )
    assert exact_zero["status"] == "fail"
    assert exact_zero["blocking"] is True
    nuisance = classify_three_c147_qu_model(
        q_over_i=0.001,
        u_over_i=0.0008,
        frac_pol=0.0013,
        represented_as_exact_zero=False,
    )
    assert nuisance["status"] == "warn"
    assert nuisance["blocking"] is False
    assert "uncertainty" in str(nuisance["relationship_to_model_data"])


def test_coherent_residuals_distinguish_noise_from_a_bias() -> None:
    rng = np.random.default_rng(1)
    noise = (rng.normal(size=8000) + 1j * rng.normal(size=8000)) * 0.02
    noise_report = coherent_residual_report(noise)
    assert noise_report["averages_as_noise"] is True
    bias = 0.027 + noise
    bias_report = coherent_residual_report(bias)
    assert bias_report["averages_as_noise"] is False
    assert bias_report["coherent_mean_abs"] > 0.02


def test_reference_jones_lookup_refuses_silent_identity() -> None:
    plane = np.eye(2, dtype=np.complex128) * 1.01
    assert np.allclose(reference_jones_for_antenna(None, 2), np.eye(2))
    assert np.allclose(reference_jones_for_antenna({2: plane}, 2), plane)
    with pytest.raises(ValueError, match="no residual reference Jones"):
        reference_jones_for_antenna({2: plane}, 7)


def test_residual_reference_jones_recovers_small_epsilon() -> None:
    source = 8.0 * np.eye(2, dtype=np.complex128)
    true = {
        0: np.eye(2, dtype=np.complex128),
        1: np.eye(2) + np.array([[0.012, 0.008 - 0.004j], [0.006, -0.01]], dtype=np.complex128),
        2: np.eye(2) + np.array([[-0.009, 0.005 + 0.007j], [-0.004, 0.011]], dtype=np.complex128),
        3: np.eye(2) + np.array([[0.007, -0.006], [0.009 + 0.003j, -0.008]], dtype=np.complex128),
    }
    pairs = [(p, q) for p in true for q in true if p < q]
    antenna1 = np.array([p for p, _ in pairs], dtype=np.int32)
    antenna2 = np.array([q for _, q in pairs], dtype=np.int32)
    vis = np.stack(
        [predict_baseline_coherency(true[int(p)], source, true[int(q)]) for p, q in pairs]
    )
    recovered = estimate_residual_reference_jones(
        antenna1,
        antenna2,
        vis,
        source=source,
        gauge_antenna_id=0,
        ridge=1.0e-8,
    )
    for antenna, plane in true.items():
        np.testing.assert_allclose(recovered["jones"][antenna], plane, atol=5.0e-3)
    hold = estimate_residual_reference_jones(
        antenna1,
        antenna2,
        vis,
        source=source,
        gauge_antenna_id=0,
        holdout_antenna_id=3,
        ridge=1.0e-8,
    )
    np.testing.assert_allclose(hold["jones"][3], np.eye(2), atol=1.0e-12)
    assert hold["holdout"]["n"] > 0
    packed = serialize_reference_jones(recovered["jones"])
    restored = deserialize_reference_jones(packed)
    np.testing.assert_allclose(restored[1], recovered["jones"][1])


def test_per_antenna_reference_jones_recovers_full_jones() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=2)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    refs = {
        0: np.array([[1.02, 0.03 - 0.01j], [-0.02, 0.98 + 0.01j]], dtype=np.complex128),
        1: np.array([[0.97, -0.025], [0.015 + 0.02j, 1.03]], dtype=np.complex128),
    }
    biased = np.array(predicted.visibility, copy=True)
    names = observation.block.correlations
    for row in range(biased.shape[0]):
        packed = pack_coherency(biased[row], names, (Receptor.R, Receptor.L))
        reference = int(observation.block.antenna1[row])
        packed = refs[reference] @ packed
        biased[row] = unpack_coherency(packed, names, JONES_RECEPTORS)
    naive = recover_holography_full_jones(observation, biased)
    corrected = recover_holography_full_jones(observation, biased, reference_jones=refs)
    truth = recover_holography_full_jones(observation, predicted.visibility)
    assert np.max(np.abs(naive.jones - truth.jones)) > 0.01
    np.testing.assert_allclose(corrected.jones, truth.jones, atol=1.0e-10)


def test_unsupported_off_diagonals_stay_invalid_not_zero() -> None:
    from dataclasses import replace

    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(observation, predicted.visibility)
    noisy = []
    for sample in artifact.samples:
        sigma = np.full((2, 2), 1.0)
        sigma[0, 0] = 0.01
        sigma[1, 1] = 0.01
        noisy.append(replace(sample, sigma=sigma, off_diagonal_valid=True))
    masked = mask_unsupported_off_diagonals(
        replace(artifact, samples=tuple(noisy)),
        min_snr=3.0,
    )
    assert not bool(np.any(masked.off_diagonal_valid))
    assert not np.isfinite(masked.jones[:, 0, 1]).any()
    assert not np.isfinite(masked.jones[:, 1, 0]).any()
    assert np.isfinite(masked.jones[:, 0, 0]).all()
    from sl1mjax.holography_full_jones import zero_unsupported_off_diagonals

    zeroed = zero_unsupported_off_diagonals(
        replace(artifact, samples=tuple(noisy)),
        min_snr=3.0,
    )
    assert not bool(np.any(zeroed.off_diagonal_valid))
    np.testing.assert_allclose(zeroed.jones[:, 0, 1], 0.0)
    np.testing.assert_allclose(zeroed.jones[:, 1, 0], 0.0)
    assert np.isfinite(zeroed.jones[:, 0, 0]).all()


def test_field9_residual_jones_recovers_all_antennas_and_qu() -> None:
    ids = np.array([0, 1, 2, 3], dtype=np.int32)
    true_eps = {
        0: np.zeros((2, 2), dtype=np.complex128),
        1: np.array([[0.01, 0.02 - 0.01j], [0.015, -0.008]], dtype=np.complex128),
        2: np.array([[-0.007, 0.018], [-0.012 + 0.006j, 0.009]], dtype=np.complex128),
        3: np.array([[0.006, -0.016 + 0.01j], [0.011, -0.005]], dtype=np.complex128),
    }
    q_frac, u_frac = 0.002, -0.0015
    times = 8
    rows = []
    for step in range(times):
        chi0 = np.deg2rad(8.0 * step)
        for p_id in ids:
            for q_id in ids:
                if p_id >= q_id:
                    continue
                chi_p = chi0 + 0.02 * p_id
                chi_q = chi0 + 0.02 * q_id
                source = 8.0 * np.array(
                    [[1.0, q_frac + 1j * u_frac], [q_frac - 1j * u_frac, 1.0]],
                    dtype=np.complex128,
                )
                r_p = sky_frame_residual_jones(np.eye(2) + true_eps[int(p_id)], chi_p)
                r_q = sky_frame_residual_jones(np.eye(2) + true_eps[int(q_id)], chi_q)
                rows.append(
                    (
                        int(p_id),
                        int(q_id),
                        8.0,
                        chi_p,
                        chi_q,
                        predict_baseline_coherency(r_p, source, r_q),
                    )
                )
    antenna1 = np.array([row[0] for row in rows], dtype=np.int32)
    antenna2 = np.array([row[1] for row in rows], dtype=np.int32)
    intensity = np.array([row[2] for row in rows], dtype=np.float64)
    chi1 = np.array([row[3] for row in rows], dtype=np.float64)
    chi2 = np.array([row[4] for row in rows], dtype=np.float64)
    vis = np.stack([row[5] for row in rows])
    later = np.zeros(antenna1.size, dtype=bool)
    later[antenna1.size // 2 :] = True
    assert antenna_graph_is_connected(antenna1, antenna2, row_mask=~later)
    fit = estimate_all_antenna_residual_jones(
        antenna1,
        antenna2,
        vis,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        gauge_antenna_id=0,
        row_mask=~later,
        ridge=1.0e-8,
        qu_ridge=1.0e-10,
        n_iter=3,
    )
    np.testing.assert_allclose(fit["q_over_i"], q_frac, atol=8.0e-4)
    np.testing.assert_allclose(fit["u_over_i"], u_frac, atol=8.0e-4)
    for ant, eps in true_eps.items():
        np.testing.assert_allclose(fit["jones"][ant], np.eye(2) + eps, atol=8.0e-3)
    hold = residual_holdout_report(
        antenna1,
        antenna2,
        vis,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=fit["jones"],
        q_over_i=fit["q_over_i"],
        u_over_i=fit["u_over_i"],
        row_mask=later,
        cluster_ids=antenna1,
    )
    assert hold["graph_connected"] is True
    assert hold["median_abs_rl_over_i"] < 0.005
    train_pairs = connected_baseline_holdout_mask(antenna1, antenna2, holdout_fraction=0.2)
    assert antenna_graph_is_connected(antenna1, antenna2, row_mask=train_pairs)
    assert (
        float(
            gauge_invariance_abs(
                antenna1,
                antenna2,
                stokes_i=intensity,
                chi1=chi1,
                chi2=chi2,
                residual_jones=fit["jones"],
                q_over_i=fit["q_over_i"],
                u_over_i=fit["u_over_i"],
            )
        )
        < 1.0e-12
    )
    identity_ids = ids[:-1]
    identity_mask = (antenna1 != 3) & (antenna2 != 3)
    identity_fit = estimate_all_antenna_residual_jones(
        antenna1,
        antenna2,
        vis,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        antenna_ids=identity_ids,
        gauge_antenna_id=0,
        row_mask=identity_mask,
        ridge=1.0e-8,
        n_iter=2,
    )
    identity_jones = dict(identity_fit["jones"])
    identity_jones[3] = np.eye(2, dtype=np.complex128)
    prior = residual_holdout_report(
        antenna1,
        antenna2,
        vis,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        residual_jones=identity_jones,
        q_over_i=identity_fit["q_over_i"],
        u_over_i=identity_fit["u_over_i"],
        row_mask=~identity_mask,
    )
    assert prior["median_abs_rl_over_i"] > hold["median_abs_rl_over_i"]


def test_moving_antenna_residual_is_divided_out_of_full_jones() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    residuals = {
        0: np.array([[1.02, 0.025], [-0.02, 0.98]], dtype=np.complex128),
        1: np.array([[0.97, -0.03 + 0.01j], [0.018, 1.04]], dtype=np.complex128),
    }
    biased = np.array(predicted.visibility, copy=True)
    names = observation.block.correlations
    for row in range(biased.shape[0]):
        packed = pack_coherency(biased[row], names, (Receptor.R, Receptor.L))
        packed = residuals[0] @ packed @ residuals[1].conj().T
        biased[row] = unpack_coherency(packed, names, JONES_RECEPTORS)
    chi = np.zeros((2, 2), dtype=np.float64)
    naive = recover_holography_full_jones(observation, biased)
    corrected = recover_holography_full_jones(
        observation,
        biased,
        residual_jones=residuals,
        parallactic_angle_rad=chi,
    )
    truth = recover_holography_full_jones(observation, predicted.visibility)
    assert np.max(np.abs(naive.jones - truth.jones)) > 0.01
    np.testing.assert_allclose(corrected.jones, truth.jones, atol=1.0e-9)


def test_prediction_equivalent_r_maps_need_on_axis_beam_gauge() -> None:
    observation = _diagonal_observation(n_time=2, n_ref=1)
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    first = {
        0: np.array([[1.02, 0.025], [-0.02, 0.98]], dtype=np.complex128),
        1: np.array([[0.97, -0.03 + 0.01j], [0.018, 1.04]], dtype=np.complex128),
    }
    left = np.array([[1.0, 0.04], [-0.03, 1.0]], dtype=np.complex128)
    second = {0: first[0], 1: left @ first[1]}
    chi = np.zeros((2, 2), dtype=np.float64)
    map_a = recover_holography_full_jones(
        observation, predicted.visibility, residual_jones=first, parallactic_angle_rad=chi
    )
    map_b = recover_holography_full_jones(
        observation, predicted.visibility, residual_jones=second, parallactic_angle_rad=chi
    )
    before = offdiag_beam_map_difference(map_a, map_b)
    after = offdiag_beam_map_difference(
        pin_on_axis_jones_to_identity(map_a),
        pin_on_axis_jones_to_identity(map_b),
    )
    assert before["median_abs_rl"] > 0.01
    assert after["median_abs_rl"] < 1.0e-9
    gate = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=before["median_abs_rl"],
        median_abs_rl_map_delta_after_pin=after["median_abs_rl"],
    )
    assert gate["maps_differ_before_pin"] is True
    assert gate["maps_agree_after_pin"] is True
    assert gate["beam_maps_uniquely_physical"] is True
    assert gate["status"] == "pass"
    pinned_a, r_a, _origin_a = refactor_on_axis_gauge(map_a, first)
    pinned_b, r_b, _origin_b = refactor_on_axis_gauge(map_b, second)
    after_refactor = offdiag_beam_map_difference(pinned_a, pinned_b)
    assert after_refactor["median_abs_rl"] < 1.0e-9
    chi = np.zeros((2, 2), dtype=np.float64)
    pred_unpinned = predict_moving_reference_from_beam(
        observation,
        residual_jones=first,
        artifact=map_a,
        row_mask=np.ones(observation.block.time_s.size, dtype=bool),
        parallactic_angle_rad=chi,
    )
    pred_pinned = predict_moving_reference_from_beam(
        observation,
        residual_jones=r_a,
        artifact=pinned_a,
        row_mask=np.ones(observation.block.time_s.size, dtype=bool),
        parallactic_angle_rad=chi,
    )
    np.testing.assert_allclose(pred_unpinned, pred_pinned, atol=1.0e-9)
    lack = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=0.04,
        median_abs_rl_map_delta_after_pin=0.001,
        full_jones_rl_residual=0.03,
        diagonal_rl_residual=0.031,
        full_jones_rr_residual=0.02,
    )
    assert lack["outcome"] == "maps_agree_offdiag_lack_predictive_evidence"
    starved = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=0.001,
        median_abs_rl_map_delta_after_pin=0.001,
        full_jones_rl_residual=0.55,
        diagonal_rl_residual=0.55,
        full_jones_rr_residual=0.74,
    )
    assert starved["outcome"] == "inconclusive"
    differ = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=0.04,
        median_abs_rl_map_delta_after_pin=0.03,
    )
    assert differ["outcome"] == "maps_still_differ_after_pin"
    support = classify_prediction_equivalent_beam_maps(
        visibility_equivalent=True,
        median_abs_rl_map_delta=0.04,
        median_abs_rl_map_delta_after_pin=0.001,
        n_failing_antennas=1,
        n_antennas=8,
    )
    assert support["outcome"] == "antenna_support_problem"


def test_holography_holdouts_are_fixed_before_recovery() -> None:
    observation = _diagonal_observation(n_time=4, n_ref=2)
    masks = holography_visibility_holdout_masks(observation, held_reference_id=0)
    assert bool(np.any(masks["train"]))
    assert not bool(np.any(masks["train"] & masks["holdout"]))
    assert bool(np.any(masks["held_reference"]))
    recovered = recover_holography_full_jones(observation, row_mask=masks["train"])
    full = recover_holography_full_jones(observation)
    assert len(recovered.samples) <= len(full.samples)
    assert all(sample.copolar_valid for sample in recovered.samples)


def test_leave_one_reference_out_keeps_interpolation_support() -> None:
    observation = _grid_observation(n_visit=2, n_ref=2, n_move=1, n_l=3, n_m=1)
    masks = holography_one_axis_holdout_masks(
        observation,
        axis="leave_one_reference_out",
        held_reference_id=0,
        reserve_outer_fold=False,
    )
    support = freeze_interpolation_support_from_rows(observation, masks["train"])
    geometry = moving_reference_row_geometry(observation)
    labeled = classify_holdout_interpolation_support(support, geometry, masks["holdout"])
    assert labeled["support_fraction"] > 0.8
    assert labeled["n_unsupported"] == 0
    beam = ManufacturedVoltageBeam(intercept=_FULL, grad_l=_FULL_GRAD)
    predicted = observation.predict(beam)
    artifact = recover_holography_full_jones(
        observation, predicted.visibility, row_mask=masks["train"]
    )
    identity = {
        antenna: np.eye(2, dtype=np.complex128)
        for antenna in np.unique(
            np.concatenate([observation.block.antenna1, observation.block.antenna2])
        )
    }
    pred_full = predict_moving_reference_from_beam(
        observation,
        residual_jones=identity,
        artifact=artifact,
        row_mask=masks["holdout"],
    )
    pred_diag = predict_moving_reference_from_beam(
        observation,
        residual_jones=identity,
        artifact=artifact,
        row_mask=masks["holdout"],
        diagonal_only=True,
    )
    packed = pack_coherency(
        predicted.visibility, observation.block.correlations, (Receptor.R, Receptor.L)
    )
    plane = packed[:, 0] if packed.ndim == 4 else packed
    pred_f = pred_full[:, 0] if pred_full.ndim == 4 else pred_full
    pred_d = pred_diag[:, 0] if pred_diag.ndim == 4 else pred_diag
    intensity = np.real(0.5 * (plane[:, 0, 0] + plane[:, 1, 1]))
    scores = paired_full_versus_diagonal_scores(
        plane,
        pred_f,
        pred_d,
        stokes_i=intensity,
        row_mask=masks["holdout"],
        dwell_ids=geometry["time_index"],
        mover_ids=geometry["moving_id"],
        reference_ids=geometry["reference_id"],
        cell_ids=geometry["cell_l"],
    )
    assert scores["identical_rows"] is True
    assert scores["n_finite"] == scores["n"]
    assert scores["rr_ll_not_regressed"] is True
    assert "dwell" in scores["clustered_improvement"]


def test_residual_jones_visibility_gates_require_clustered_improvement() -> None:
    failed = classify_residual_jones_visibility(
        rl_improved_clusters=1,
        n_mover_clusters=3,
        n_reference_clusters=3,
        rr_ll_regression=False,
        max_group_coherent_abs=0.005,
        qu_interval_spread=0.001,
        gauge_invariance_abs=1.0e-12,
    )
    assert failed["status"] == "fail"
    passed = classify_residual_jones_visibility(
        rl_improved_clusters=3,
        n_mover_clusters=3,
        n_reference_clusters=3,
        rr_ll_regression=False,
        max_group_coherent_abs=0.005,
        qu_interval_spread=0.001,
        gauge_invariance_abs=1.0e-12,
    )
    assert passed["status"] == "pass"


def test_injected_feed_leakage_locks_plus_two_chi() -> None:
    report = injected_feed_leakage_rotation(np.linspace(-0.4, 0.4, 9))
    assert report["status"] == "pass"
    assert report["expected_locked_rl_phase"] == "exp(+2i chi)"
    assert report["locked_slope_darg_over_d_two_chi"] == pytest.approx(1.0, abs=0.05)
    assert report["opposite_slope_darg_over_d_two_chi"] == pytest.approx(-1.0, abs=0.05)


def test_identical_channel_support_rejects_time_varying_flags() -> None:
    finite = np.ones((6, 4), dtype=bool)
    finite[0, 1] = False
    keep, report = identical_channel_support(finite)
    assert report["time_dependent_support"] is True
    assert bool(keep[1]) is False
    assert int(report["n_kept"]) == 3
    stable = np.ones((6, 4), dtype=bool)
    keep_ok, stable_report = identical_channel_support(stable)
    assert stable_report["time_dependent_support"] is False
    assert bool(np.all(keep_ok))


def test_offdiag_drift_recovers_injected_linear_leakage() -> None:
    ids = np.array([0, 1, 2], dtype=np.int32)
    times = np.linspace(0.0, 1000.0, 6)
    rows = []
    for time_s in times:
        tau = (time_s - 500.0) / 1000.0
        for p_id in ids:
            for q_id in ids:
                if p_id >= q_id:
                    continue
                eps_p = np.zeros((2, 2), dtype=np.complex128)
                eps_q = np.zeros((2, 2), dtype=np.complex128)
                if p_id == 1:
                    eps_p[0, 1] = 0.02 + 0.03 * tau
                if q_id == 1:
                    eps_q[0, 1] = 0.02 + 0.03 * tau
                if p_id == 2:
                    eps_p[1, 0] = -0.015
                if q_id == 2:
                    eps_q[1, 0] = -0.015
                chi_p = 0.05 * p_id
                chi_q = 0.05 * q_id
                source = 8.0 * np.eye(2, dtype=np.complex128)
                r_p = sky_frame_residual_jones(np.eye(2) + eps_p, chi_p)
                r_q = sky_frame_residual_jones(np.eye(2) + eps_q, chi_q)
                rows.append(
                    (
                        int(p_id),
                        int(q_id),
                        time_s,
                        8.0,
                        chi_p,
                        chi_q,
                        predict_baseline_coherency(r_p, source, r_q),
                    )
                )
    antenna1 = np.array([row[0] for row in rows], dtype=np.int32)
    antenna2 = np.array([row[1] for row in rows], dtype=np.int32)
    time_s = np.array([row[2] for row in rows], dtype=np.float64)
    intensity = np.array([row[3] for row in rows], dtype=np.float64)
    chi1 = np.array([row[4] for row in rows], dtype=np.float64)
    chi2 = np.array([row[5] for row in rows], dtype=np.float64)
    vis = np.stack([row[6] for row in rows])
    fit = estimate_all_antenna_residual_jones(
        antenna1,
        antenna2,
        vis,
        stokes_i=intensity,
        chi1=chi1,
        chi2=chi2,
        gauge_antenna_id=0,
        time_s=time_s,
        offdiag_drift=True,
        drift_ridge=1.0e-8,
        ridge=1.0e-8,
        qu_ridge=1.0e-10,
        n_iter=3,
        time_scale_s=1000.0,
    )
    assert fit["offdiag_drift"] is True
    np.testing.assert_allclose(fit["jones"][1][0, 1], 0.02, atol=8.0e-3)
    np.testing.assert_allclose(fit["drift_offdiag"][1][0], 0.03, atol=8.0e-3)


def test_blocked_time_folds_stay_inside_the_training_mask() -> None:
    times = np.arange(12, dtype=np.float64)
    train = np.zeros(12, dtype=bool)
    train[:8] = True
    folds = blocked_time_folds(times, train, n_folds=2)
    assert len(folds) == 2
    assert not bool(np.any(folds[0] & ~train))
    assert not bool(np.any(folds[1] & ~train))
    assert int(np.sum(folds[0] | folds[1])) == 8


def test_scan_state_jumps_indicate_piecewise_offsets() -> None:
    identity = np.eye(2, dtype=np.complex128)
    jumped = identity.copy()
    jumped[0, 1] = 0.04
    report = scan_state_jump_report(
        {
            "10": {4: identity, 11: identity},
            "20": {4: jumped, 11: identity},
            "30": {4: jumped, 11: jumped},
            "40": {4: identity, 11: jumped},
        },
        jump_threshold=0.015,
    )
    assert report["n_jump"] >= 3
    assert report["piecewise_state_term_indicated"] is True


def test_field9_time_transfer_gate_requires_later_noise_like_drop() -> None:
    failed = classify_field9_time_transfer(
        later_coherent_abs=0.0099,
        later_averages_as_noise=False,
        baseline_averages_as_noise=True,
        qu_fold_spread=0.001,
        rr_ll_regression=False,
        antenna_later_coherent={"ea04": 0.012},
        antenna_static_coherent={"ea04": 0.011},
        previous_coherent_abs=0.0099,
    )
    assert failed["status"] == "fail"
    passed = classify_field9_time_transfer(
        later_coherent_abs=0.004,
        later_averages_as_noise=True,
        baseline_averages_as_noise=True,
        qu_fold_spread=0.001,
        rr_ll_regression=False,
        antenna_later_coherent={"ea04": 0.005, "ea11": 0.004, "ea21": 0.006},
        antenna_static_coherent={"ea04": 0.012, "ea11": 0.011, "ea21": 0.010},
        scale_coherent={"1e3": 0.004, "1e4": 0.0045},
        previous_coherent_abs=0.0099,
    )
    assert passed["status"] == "pass"


def test_pin_on_axis_jones_forces_identity_at_zero() -> None:
    on_axis = np.array([[1.03, 0.02], [-0.015, 0.97]], dtype=np.complex128)
    off_axis = on_axis @ np.array([[1.0, 0.01], [0.0, 1.0]], dtype=np.complex128)
    sigma = np.ones((2, 2), dtype=np.float64)
    samples = (
        HolographyFullJonesSample(
            moving_antenna_id=4,
            unique_time_s=1.0,
            offset_lm_rad=np.zeros(2, dtype=np.float64),
            frequency_hz=4.8e9,
            jones=on_axis,
            sigma=sigma,
            n_reference=1,
            weight=1.0,
            copolar_valid=True,
            off_diagonal_valid=True,
            raster="on_axis",
        ),
        HolographyFullJonesSample(
            moving_antenna_id=4,
            unique_time_s=2.0,
            offset_lm_rad=np.array([2.0e-4, 0.0], dtype=np.float64),
            frequency_hz=4.8e9,
            jones=off_axis,
            sigma=sigma,
            n_reference=1,
            weight=1.0,
            copolar_valid=True,
            off_diagonal_valid=True,
            raster="off",
        ),
    )
    artifact = HolographyFullJonesArtifact(
        samples=samples,
        jones=np.stack([on_axis, off_axis], axis=0),
        valid=np.array([True, True]),
        off_diagonal_valid=np.array([True, True]),
        calibration_state="casa_parang_true",
        source_name="3C147",
        source_model_version="stokes_i_point",
        reference_combination="test",
        receptor_convention="circular_R_L",
        offset_sign="commanded",
        frozen=False,
    )
    pinned = pin_on_axis_jones_to_identity(artifact)
    np.testing.assert_allclose(pinned.jones[0], np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(
        pinned.jones[1],
        np.linalg.inv(on_axis) @ off_axis,
        atol=1.0e-12,
    )


def test_chain_jump_is_attributed_to_the_first_increment() -> None:
    report = classify_chain_jump_stage(
        {
            "DATA": 0.002,
            "K_B_G": 0.003,
            "K_B_G_Kcross": 0.004,
            "K_B_G_Kcross_Df": 0.028,
            "K_B_G_Kcross_Df_Xf": 0.029,
            "K_B_G_Kcross_Df_Xf_P": 0.030,
        }
    )
    assert report["present_in_data"] is False
    assert report["first_increment_above_threshold"] == "K_B_G_Kcross_Df"
    assert report["first_stage_above_threshold"] == "K_B_G_Kcross_Df"
    data = classify_chain_jump_stage({"DATA": 0.04, "K_B_G": 0.041})
    assert data["present_in_data"] is True
    assert data["investigate_acquisition_state"] is True
    absent = classify_chain_jump_stage(
        {
            "DATA": 0.002,
            "K_B_G": 0.003,
            "K_B_G_Kcross": 0.003,
            "K_B_G_Kcross_Df": 0.004,
            "K_B_G_Kcross_Df_Xf": 0.004,
            "K_B_G_Kcross_Df_Xf_P": 0.004,
        }
    )
    assert absent["visibility_jump_absent"] is True
    assert absent["status"] == "pass"


def test_common_mode_jump_is_distinguished_from_antenna_scatter() -> None:
    common = classify_common_mode_vs_antenna({"ea04": 0.04 + 0.0j, "ea11": 0.041, "ea21": 0.039})
    assert common["common_mode"] is True
    scatter = classify_common_mode_vs_antenna({"ea04": 0.04, "ea11": -0.03, "ea21": 0.002})
    assert scatter["antenna_specific"] is True


def test_hierarchical_model_is_specified_not_fit() -> None:
    model = hierarchical_residual_jones_model()
    assert model["status"] == "specified_not_fit"
    assert model["unconstrained_per_scan_jones"] is False
    gate = classify_interleaved_onaxis_transfer(chain_stage="DATA")
    assert gate["status"] == "not_run"
    assert gate["smooth_gp_permanently_blocks_full_jones"] is False
    refused = classify_interleaved_onaxis_transfer(
        chain_stage="DATA",
        used_holography_crosshands_for_onaxis=True,
    )
    assert refused["status"] == "fail"


def test_matched_scans_share_baselines_and_sample_count() -> None:
    scan = np.array([53, 53, 53, 56, 56, 56, 56], dtype=np.int32)
    antenna1 = np.array([0, 0, 1, 0, 0, 1, 1], dtype=np.int32)
    antenna2 = np.array([1, 2, 2, 1, 2, 2, 0], dtype=np.int32)
    usable = np.ones(scan.size, dtype=bool)
    mask_a, mask_b, report = matched_scan_row_masks(scan, antenna1, antenna2, usable, 53, 56)
    assert report["n_matched_baselines"] == 3
    assert int(np.sum(mask_a)) == int(np.sum(mask_b))
    assert int(np.sum(mask_a)) == 3


def test_gauge_aligned_operator_not_raw_epsilon() -> None:
    first = {
        0: np.eye(2, dtype=np.complex128),
        1: np.eye(2) + np.array([[0.0, 0.03], [0.02, 0.0]], dtype=np.complex128),
    }
    phase = 0.4
    second = {ant: np.exp(1j * phase) * np.array(plane, copy=True) for ant, plane in first.items()}
    second[0] = np.eye(2, dtype=np.complex128)
    aligned, _ = align_residual_jones_maps(first, second, gauge_antenna_id=0)
    jump = jones_parameter_jump(first, aligned)
    assert jump["max_abs_delta"] < 0.02
    antenna1 = np.array([0], dtype=np.int32)
    antenna2 = np.array([1], dtype=np.int32)
    intensity = np.array([8.0])
    chi = np.zeros(1)
    operators = predicted_operator_difference(
        antenna1,
        antenna2,
        stokes_i=intensity,
        chi1=chi,
        chi2=chi,
        first_jones=first,
        second_jones=aligned,
        q_over_i=0.0,
        u_over_i=0.0,
        row_mask=np.array([True]),
    )
    assert operators["median_abs_rl_over_i"] < 0.005
    gate = classify_estimator_identifiability(
        parameter_max_abs_delta=0.25,
        operator_median_abs_rl=0.001,
        self_residual=0.01,
        cross_residual=0.012,
        gram_condition=1.0e7,
        qu_offdiag_spread=0.03,
        init_offdiag_spread=0.01,
    )
    assert gate["parameter_jump_is_physical"] is False
    assert gate["cross_apply_equivalent"] is True
    observed = classify_estimator_identifiability(
        parameter_max_abs_delta=0.19,
        operator_median_abs_rl=0.014,
        self_residual=0.025,
        cross_residual=0.025,
        gram_condition=970.0,
        qu_offdiag_spread=0.009,
        init_offdiag_spread=1.0e-7,
    )
    assert observed["parameter_jump_is_physical"] is False
    assert observed["cross_apply_equivalent"] is True
    assert observed["status"] == "pass"
    assert observed["gate_kind"] == "visibility_operator_identifiability"
    assert observed["jones_factors_unique"] is False
    assert observed["epsilon_gauge_dependent"] is True


def test_field9_scan_clusters_hold_out_complete_visits() -> None:
    scan = np.array([1, 1, 2, 2, 10, 10, 11, 11], dtype=np.int32)
    time = np.array([0.0, 1.0, 10.0, 11.0, 1000.0, 1001.0, 1010.0, 1011.0])
    usable = np.ones(scan.size, dtype=bool)
    clusters = field9_scan_clusters(scan, time, usable, gap_s=400.0)
    assert clusters == [[1, 2], [10, 11]]
    after_raster = field9_scan_clusters(
        scan,
        time,
        usable,
        predecessor_field={1: None, 2: 9, 10: 10, 11: 9},
        gap_s=1.0e6,
    )
    assert after_raster == [[1, 2], [10, 11]]
    train, hold, report = held_out_scan_cluster_masks(scan, usable, clusters)
    assert report["held_out_scans"] == [10, 11]
    assert int(np.sum(hold)) == 4
    assert int(np.sum(train)) == 4
    assert not bool(np.any(train & hold))


def test_stabilized_gate_is_operator_identifiability() -> None:
    gate = classify_stabilized_residual_jones(
        scan_cluster_residual=0.026,
        baseline_residual=0.025,
        sample_residual=0.024,
        family_operator_max_rl=0.012,
        family_cross_equivalent=True,
        n_held_out_scans=2,
    )
    assert gate["status"] == "pass"
    assert gate["gate_kind"] == "visibility_operator_identifiability"
    assert gate["jones_factors_unique"] is False
    assert gate["epsilon_gauge_dependent"] is True
