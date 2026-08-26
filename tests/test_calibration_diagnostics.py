from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.calibration_diagnostics import (
    compare_solutions,
    diagnose_calibration,
    evaluate_fixed_sky_calibration,
    propose_residual_flags,
)
from sl1mjax.data.synthetic import simulate_calibration_case
from sl1mjax.split import calibration_split


def test_exact_truth_has_zero_residual_closure_and_connected_domains() -> None:
    case = simulate_calibration_case()
    split = calibration_split(case.block, seed=4)

    diagnostics = diagnose_calibration(case.block, case.truth, split)

    assert diagnostics.train.normalized_rms < 1e-13
    assert diagnostics.holdout.normalized_rms < 1e-13
    assert diagnostics.closure.phase_rms_rad < 1e-13
    assert diagnostics.closure.log_amplitude_rms < 1e-13
    assert diagnostics.closure.triangle_channel_count > 0
    assert diagnostics.domains.missing_domains == ()
    assert diagnostics.domains.disconnected_time_indices == ()


def test_solution_comparison_aligns_common_gauge() -> None:
    case = simulate_calibration_case()
    phase = np.exp(1j * np.linspace(0.1, 0.7, case.truth.gains.shape[0]))
    shifted = replace(case.truth, gains=case.truth.gains * phase[:, None, None])

    comparison = compare_solutions(shifted, case.truth)

    assert comparison.gain_log_amplitude_rms < 1e-14
    assert comparison.gain_phase_rms_rad < 1e-14
    assert comparison.delay_rms_s == 0
    assert comparison.bandpass_log_amplitude_rms < 1e-14
    assert comparison.bandpass_phase_rms_rad < 1e-14


def test_residual_flag_proposal_is_non_mutating_and_finds_outlier() -> None:
    case = simulate_calibration_case(noise_std=0.001, seed=9)
    visibility = case.block.visibility.copy()
    visibility[3, 4, 1] += 100.0
    block = replace(case.block, visibility=visibility)
    original_flag = block.flag.copy()

    proposal = propose_residual_flags(block, case.truth, sigma=5.0)

    assert proposal[3, 4, 1]
    np.testing.assert_array_equal(block.flag, original_flag)
    assert np.sum(proposal) < 10


def test_fixed_sky_calibration_uses_model_normalization_and_sealed_masks() -> None:
    case = simulate_calibration_case(noise_std=0.0, seed=12)
    assert case.block.model_visibility is not None
    model = case.block.model_visibility
    corrected = replace(
        case.block,
        visibility=model.copy(),
        flag=np.zeros(case.block.shape, dtype=bool),
    )
    biased = replace(corrected, visibility=0.5 * model)
    train = np.zeros(case.block.shape, dtype=bool)
    train[: case.block.shape[0] // 2] = True
    holdout = ~train

    exact = evaluate_fixed_sky_calibration(
        "exact",
        (corrected,),
        (model,),
        (train,),
        (holdout,),
    )
    amplitude_biased = evaluate_fixed_sky_calibration(
        "biased",
        (biased,),
        (model,),
        (train,),
        (holdout,),
    )

    assert exact.train["normalized_residual_power"] == 0.0
    assert exact.holdout["normalized_residual_power"] == 0.0
    assert amplitude_biased.train["normalized_residual_power"] == pytest.approx(0.25)
    assert amplitude_biased.holdout["normalized_residual_power"] == pytest.approx(0.25)
    assert len(amplitude_biased.per_channel) == case.block.frequency_hz.size
