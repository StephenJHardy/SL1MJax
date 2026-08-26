from __future__ import annotations

from dataclasses import replace

import numpy as np

from sl1mjax.calibration import identity_solution
from sl1mjax.gain_time_models import circular_gp_gain_solution, smooth_gain_solution
from sl1mjax.polarization import Correlation


def _solution(time_s: np.ndarray):
    return identity_solution(
        antenna_count=3,
        correlations=(Correlation.RR, Correlation.LL),
        frequency_hz=np.array([1.0e9]),
        time_s=time_s,
        reference_antenna=0,
    )


def test_curvature_smoothing_preserves_linear_log_gain_and_reference_gauge() -> None:
    times = np.array([0.0, 1.0, 2.5, 4.0, 7.0])
    solution = _solution(times)
    gains = solution.gains.copy()
    gains[:, 1, 0] = np.exp(0.1 + 0.03 * times + 1j * (0.2 - 0.04 * times))
    gains[:, 1, 1] = np.exp(-0.2 + 0.02 * times + 1j * (-0.1 + 0.05 * times))

    smoothed = smooth_gain_solution(replace(solution, gains=gains), strength=10.0)

    np.testing.assert_allclose(smoothed.gains[:, 1, :], gains[:, 1, :], atol=2e-10)
    np.testing.assert_allclose(np.angle(smoothed.gains[:, 0, :]), 0.0, atol=1e-14)


def test_curvature_smoothing_reduces_gain_roughness_and_fills_missing_knots() -> None:
    times = np.arange(6, dtype=np.float64)
    solution = _solution(times)
    gains = solution.gains.copy()
    gains[:, 1, 0] = np.exp(0.1 * np.array([0.0, 1.0, -1.0, 1.0, -1.0, 0.0]))
    valid = solution.gain_valid.copy()
    valid[2, 1, 0] = False

    smoothed = smooth_gain_solution(replace(solution, gains=gains, gain_valid=valid), strength=3.0)

    original_log_amplitude = np.log(np.abs(gains[:, 1, 0]))
    fitted_log_amplitude = np.log(np.abs(smoothed.gains[:, 1, 0]))
    assert np.linalg.norm(np.diff(fitted_log_amplitude, n=2)) < np.linalg.norm(
        np.diff(original_log_amplitude, n=2)
    )
    assert np.all(smoothed.gain_valid[:, 1, 0])


def test_circular_gp_interpolates_across_phase_wrap_on_unit_circle() -> None:
    times = np.array([0.0, 1.0, 2.0])
    solution = _solution(times)
    gains = solution.gains.copy()
    phase = np.deg2rad(np.array([170.0, 180.0, -170.0]))
    gains[:, 1, :] = np.exp(1j * phase[:, None])
    evaluation = np.array([0.5, 1.0, 1.5])

    modeled = circular_gp_gain_solution(
        replace(solution, gains=gains),
        evaluation,
        length_scale_s=1.0,
        noise_variance=1e-6,
    )

    assert np.all(np.abs(np.angle(modeled.gains[:, 1, 0])) > np.deg2rad(150.0))
    np.testing.assert_allclose(np.angle(modeled.gains[:, 0, :]), 0.0, atol=1e-14)


def test_circular_gp_preserves_fully_missing_antenna_domain() -> None:
    times = np.array([0.0, 1.0, 2.0])
    solution = _solution(times)
    valid = solution.gain_valid.copy()
    valid[:, 2, :] = False

    modeled = circular_gp_gain_solution(
        replace(solution, gain_valid=valid),
        times,
        length_scale_s=1.0,
        noise_variance=1e-8,
    )

    assert not np.any(modeled.gain_valid[:, 2, :])
    np.testing.assert_allclose(modeled.gains[:, 1, :], 1.0, atol=1e-7)
