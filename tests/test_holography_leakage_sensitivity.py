from __future__ import annotations

from dataclasses import replace

import numpy as np
from test_holography_full_jones import _diagonal_observation

from sl1mjax.cassbeam_beam import CassbeamCBandVoltageBeam, load_cassbeam_cband_artifact
from sl1mjax.finite_pixel import ManufacturedVoltageBeam
from sl1mjax.holography_full_jones import (
    recover_holography_full_jones,
    recover_holography_full_jones_rows,
    refactor_on_axis_gauge,
    zero_unsupported_off_diagonals,
)
from sl1mjax.holography_leakage_sensitivity import (
    INJECTED_LEAKAGE_AMPLITUDES,
    aggregate_feed_frame_leakage,
    cassbeam_predicted_leakage_amplitude,
    classify_leakage_sensitivity,
    combine_channel_leakage_after_delay,
    evaluate_cassbeam_jones_at_offsets,
    feed_frame_leakage_ratios,
    inject_feed_frame_leakage,
    injection_recovery_point,
    recovered_leakage_from_artifact,
    unmasked_leakage_diagnostics,
)
from sl1mjax.voltage_beam import beam_coordinates


def _identity_observation(*, n_time: int = 3, n_ref: int = 2, frequency_hz=None):
    observation = _diagonal_observation(n_time=n_time, n_ref=n_ref, frequency_hz=frequency_hz)
    beam = ManufacturedVoltageBeam(intercept=np.eye(2, dtype=np.complex128))
    predicted = observation.predict(beam)
    return observation, predicted.visibility


def test_injected_feed_frame_leakage_is_recovered() -> None:
    observation, visibility = _identity_observation()
    injected = inject_feed_frame_leakage(observation, epsilon=0.03, visibility=visibility)
    artifact = recover_holography_full_jones(observation, injected)
    recovered = recovered_leakage_from_artifact(artifact, off_axis_only=True)
    assert abs(recovered["coherent_abs_rl"] - 0.03) < 5e-3
    assert abs(recovered["coherent_abs_lr"] - 0.03) < 5e-3
    rows = recover_holography_full_jones_rows(observation, injected)
    radius = np.hypot(rows["offset_lm_rad"][:, 0], rows["offset_lm_rad"][:, 1])
    d_rl, d_lr = feed_frame_leakage_ratios(rows["jones"])
    np.testing.assert_allclose(np.abs(d_rl[radius > 1.0e-6]), 0.03, atol=1e-8)
    np.testing.assert_allclose(np.abs(d_lr[radius > 1.0e-6]), 0.03, atol=1e-8)
    np.testing.assert_allclose(np.abs(d_rl[radius <= 1.0e-6]), 0.0, atol=1e-8)


def test_off_axis_injection_survives_on_axis_gauge() -> None:
    observation, visibility = _identity_observation(n_time=3, n_ref=2)
    injected = inject_feed_frame_leakage(observation, epsilon=0.03, visibility=visibility)
    unpinned = recover_holography_full_jones(observation, injected)
    residual = {0: np.eye(2), 1: np.eye(2), 2: np.eye(2)}
    pinned, _updated, _origin = refactor_on_axis_gauge(unpinned, residual)
    recovered = recovered_leakage_from_artifact(pinned, off_axis_only=True)
    assert abs(recovered["coherent_abs_rl"] - 0.03) < 1e-2


def test_injection_recovery_uses_increment_above_baseline() -> None:
    floor = {"coherent_abs_rl": 0.00876, "coherent_abs_lr": 0.00876}
    false_hit = injection_recovery_point(
        injected_amplitude=0.003,
        recovered=floor,
        baseline_abs=0.00876,
    )
    assert false_hit["recovered"] is False
    assert abs(false_hit["recovered_increment_abs_rl"]) < 1e-6
    genuine = injection_recovery_point(
        injected_amplitude=0.01,
        recovered={"coherent_abs_rl": 0.0188, "coherent_abs_lr": 0.0188},
        baseline_abs=0.0088,
    )
    assert genuine["recovered"] is True


def test_unmasked_coherent_mean_is_not_median_magnitude() -> None:
    observation, visibility = _identity_observation(n_time=4, n_ref=2)
    injected = inject_feed_frame_leakage(observation, epsilon=0.01 + 0.0j, visibility=visibility)
    rows = recover_holography_full_jones_rows(observation, injected)
    report = unmasked_leakage_diagnostics(rows)
    assert abs(report["median_abs_rl"] - 0.01) < 2e-3
    assert report["coherent_abs_rl"] > 0.005
    assert report["n_finite_rl"] == report["n"]
    assert "reference_scatter_rl" in report
    assert "repeat_visit_scatter_rl" in report
    assert report["n_meaning"].startswith("unmasked")


def test_aggregation_recovers_coherent_leak_that_per_time_snr_discards() -> None:
    observation, visibility = _identity_observation(n_time=12, n_ref=2)
    offsets = np.array(observation.pointing.offset_lm_rad, copy=True)
    offsets[:] = offsets[:1]
    observation = replace(
        observation,
        pointing=replace(observation.pointing, offset_lm_rad=offsets),
    )
    injected = inject_feed_frame_leakage(observation, epsilon=0.03, visibility=visibility)
    rng = np.random.default_rng(0)
    noisy = np.array(injected, copy=True)
    noise = 0.03 * (rng.normal(size=noisy.shape) + 1j * rng.normal(size=noisy.shape))
    noisy[..., 1] += noise[..., 1]
    noisy[..., 2] += noise[..., 2]
    per_time = recover_holography_full_jones(observation, noisy)
    masked = zero_unsupported_off_diagonals(per_time, min_snr=3.0)
    assert int(np.sum(masked.off_diagonal_valid)) < len(masked.samples)
    rows = recover_holography_full_jones_rows(observation, noisy)
    aggregated = aggregate_feed_frame_leakage(rows, min_snr=3.0, observation=observation)
    recovered = recovered_leakage_from_artifact(aggregated)
    assert recovered["n_off_diagonal_valid"] >= 1
    assert abs(recovered["coherent_abs_rl"] - 0.03) < 1e-2


def test_frequency_combine_after_delay_recovers_delayed_leak() -> None:
    frequencies = np.linspace(4.55e9, 4.58e9, 8)
    observation, visibility = _identity_observation(n_time=2, n_ref=2, frequency_hz=frequencies)
    injected = inject_feed_frame_leakage(observation, epsilon=0.01, visibility=visibility)
    delay = 2.0e-8
    delayed = np.array(injected, copy=True)
    phase = np.exp(1j * 2.0 * np.pi * delay * frequencies)
    delayed[..., 1] *= phase
    delayed[..., 2] *= phase
    rows = recover_holography_full_jones_rows(observation, delayed)
    combined = combine_channel_leakage_after_delay(rows)
    radius = np.hypot(combined["offset_lm_rad"][:, 0], combined["offset_lm_rad"][:, 1])
    off = radius > 1.0e-6
    assert abs(float(np.median(combined["coherent_abs_rl"][off])) - 0.01) < 3e-3
    assert int(np.min(combined["n_channel_combined"])) >= 8


def test_cassbeam_batch_matches_public_evaluate() -> None:
    offset = np.array([[0.002, -0.001]], dtype=np.float64)
    frequency = np.array([4.564e9], dtype=np.float64)
    chi = np.array([0.35], dtype=np.float64)
    jones, ok = evaluate_cassbeam_jones_at_offsets(offset, frequency, chi, off_diagonal=True)
    assert bool(ok[0])
    beam = CassbeamCBandVoltageBeam(
        load_cassbeam_cband_artifact(),
        off_diagonal=True,
        allow_unfrozen=True,
    )
    evaluation = beam.evaluate(
        beam_coordinates(
            np.array([0.0]),
            np.array([0.0]),
            frequency,
            parallactic_angle_rad=chi,
            pointing_offset_lm_rad=offset[0],
        ),
        calibration_state="casa_parang_true",
    )
    np.testing.assert_allclose(jones[0], evaluation.jones[0, 0, 0], atol=1e-10)
    amplitude = cassbeam_predicted_leakage_amplitude(offset, frequency, chi)
    assert amplitude["n"] == 1
    assert amplitude["median_abs_rl_over_rr"] > 0.0


def test_injection_curve_and_classifier_keep_full_jones_unfrozen() -> None:
    assert INJECTED_LEAKAGE_AMPLITUDES == (0.001, 0.003, 0.01, 0.03)
    recovered = {
        "coherent_abs_rl": 0.011,
        "coherent_abs_lr": 0.010,
        "median_abs_rl": 0.011,
        "n_off_diagonal_valid": 40,
        "off_diagonal_fraction": 0.8,
    }
    point = injection_recovery_point(
        injected_amplitude=0.01,
        recovered=recovered,
        heldout_scores={"rl_improved": True, "lr_improved": True, "rr_ll_regression": False},
        baseline_abs=0.0,
    )
    assert point["recovered"] is True
    curve = {
        "0.001": {"recovered": False},
        "0.003": {"recovered": False},
        "0.01": point,
        "0.03": {"recovered": True},
    }
    missing = classify_leakage_sensitivity(
        cassbeam_amplitude=0.01,
        injection_curve={key: {"recovered": False} for key in ("0.001", "0.003", "0.01", "0.03")},
        cassbeam_rl_improved=False,
        cassbeam_lr_improved=False,
        cassbeam_rr_ll_regression=False,
        real_coherent_abs=0.001,
    )
    assert missing["outcome"] == "injections_not_recovered_at_cassbeam_amplitude"
    against = classify_leakage_sensitivity(
        cassbeam_amplitude=0.01,
        injection_curve=curve,
        cassbeam_rl_improved=False,
        cassbeam_lr_improved=False,
        cassbeam_rr_ll_regression=False,
        real_coherent_abs=0.001,
    )
    assert against["outcome"] == "injections_recovered_real_leakage_absent"
    estimator = classify_leakage_sensitivity(
        cassbeam_amplitude=0.01,
        injection_curve=curve,
        cassbeam_rl_improved=True,
        cassbeam_lr_improved=True,
        cassbeam_rr_ll_regression=False,
        real_coherent_abs=0.008,
    )
    assert estimator["outcome"] == "cassbeam_direct_improves_crosshands"
    floor = classify_leakage_sensitivity(
        cassbeam_amplitude=0.01,
        injection_curve={key: {"recovered": True} for key in ("0.001", "0.003", "0.01", "0.03")},
        cassbeam_rl_improved=False,
        cassbeam_lr_improved=False,
        cassbeam_rr_ll_regression=False,
        real_coherent_abs=0.009,
    )
    assert floor["outcome"] == "cassbeam_direct_also_null"
    for report in (missing, against, estimator, floor):
        assert report["full_jones_frozen"] is False
        assert report["spw5_closed"] is True
        assert report["full_jones_blocked"] is True
        assert report["most_important_next_artifact"] == "cassbeam_diagonal_low_order_correction"
