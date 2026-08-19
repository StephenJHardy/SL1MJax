import numpy as np
import pytest

from sl1mjax.calibration import apply_calibration, baseline_jones
from sl1mjax.data.synthetic import (
    complete_baseline_schedule,
    simulate_calibration_case,
)


def test_complete_baseline_schedule_contains_every_antenna_per_time() -> None:
    times, first, second = complete_baseline_schedule(5, np.array([0.0, 10.0]))
    assert times.size == 20
    for time in np.unique(times):
        selected = times == time
        assert set(first[selected]) | set(second[selected]) == set(range(5))
        assert len(set(zip(first[selected], second[selected], strict=True))) == 10


@pytest.mark.parametrize("terms", [("G",), ("K",), ("B",), ("G", "K", "B")])
def test_noiseless_calibration_truth_recovers_model(
    terms: tuple[str, ...],
) -> None:
    case = simulate_calibration_case(terms=terms, seed=4)

    corrected = apply_calibration(case.block, case.truth)

    assert corrected.model_visibility is not None
    np.testing.assert_allclose(
        corrected.visibility,
        corrected.model_visibility,
        rtol=2e-13,
        atol=2e-13,
    )


def test_parallel_hands_have_independent_corruption() -> None:
    case = simulate_calibration_case(terms=("G",), seed=2)
    baseline, _ = baseline_jones(
        case.truth,
        case.block.time_s,
        case.block.frequency_hz,
        case.block.antenna1,
        case.block.antenna2,
    )

    assert not np.allclose(baseline[..., 0], baseline[..., 1])


def test_corrected_residual_is_consistent_with_injected_noise_and_flags() -> None:
    noise_std = 0.01
    case = simulate_calibration_case(
        noise_std=noise_std,
        flag_fraction=0.08,
        seed=8,
    )
    corrected = apply_calibration(case.block, case.truth)
    assert corrected.model_visibility is not None
    active = corrected.active
    residual = corrected.visibility - corrected.model_visibility

    normalized_rms = np.sqrt(np.mean(np.abs(residual[active]) ** 2)) / noise_std

    assert 0.7 < normalized_rms < 1.6
    np.testing.assert_array_equal(corrected.flag, case.block.flag)
