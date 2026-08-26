"""Deterministic time models for solved diagonal complex gains."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sl1mjax.calibration import CalibrationSolution


def _valid_gain(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid & np.isfinite(values) & (np.abs(values) > 0)


def _second_derivative_matrix(time_s: np.ndarray) -> np.ndarray:
    """Return an irregular-grid curvature operator with dimensionless time."""

    times = np.asarray(time_s, dtype=np.float64)
    if times.ndim != 1 or times.size < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be a strictly increasing vector with at least 3 values")
    cadence = float(np.median(np.diff(times)))
    coordinate = (times - times[0]) / cadence
    operator = np.zeros((times.size - 2, times.size), dtype=np.float64)
    for index in range(1, times.size - 1):
        left = coordinate[index] - coordinate[index - 1]
        right = coordinate[index + 1] - coordinate[index]
        scale = 2.0 / (left + right)
        operator[index - 1, index - 1] = scale / left
        operator[index - 1, index] = -scale * (1.0 / left + 1.0 / right)
        operator[index - 1, index + 1] = scale / right
    return operator


def _smooth_real_series(
    time_s: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    strength: float,
) -> np.ndarray:
    selected = np.flatnonzero(valid)
    if selected.size == 0:
        raise ValueError("real series has no valid observations")
    if selected.size == 1:
        return np.full(time_s.size, values[selected[0]], dtype=np.float64)
    if selected.size == 2:
        return np.interp(time_s, time_s[selected], values[selected])
    curvature = _second_derivative_matrix(time_s)
    observation = np.diag(valid.astype(np.float64))
    system = observation + strength * curvature.T @ curvature
    system += 1e-12 * np.eye(time_s.size)
    right = np.where(valid, values, 0.0)
    return np.linalg.solve(system, right)


def smooth_gain_solution(
    solution: CalibrationSolution,
    *,
    strength: float,
) -> CalibrationSolution:
    """Penalize curvature in gain log amplitude and unwrapped phase."""

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    times = solution.gain_time_s
    if times.size < 3:
        raise ValueError("gain smoothing requires at least three time knots")
    gains = np.ones_like(solution.gains)
    gain_valid = np.zeros_like(solution.gain_valid)
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            values = solution.gains[:, antenna, receptor]
            valid = _valid_gain(values, solution.gain_valid[:, antenna, receptor])
            if not np.any(valid):
                continue
            log_amplitude = np.zeros(times.size, dtype=np.float64)
            phase = np.zeros(times.size, dtype=np.float64)
            log_amplitude[valid] = np.log(np.abs(values[valid]))
            phase[valid] = np.unwrap(np.angle(values[valid]))
            fitted_amplitude = _smooth_real_series(times, log_amplitude, valid, strength)
            fitted_phase = _smooth_real_series(times, phase, valid, strength)
            gains[:, antenna, receptor] = np.exp(fitted_amplitude + 1j * fitted_phase)
            gain_valid[:, antenna, receptor] = True
    reference_phase = np.angle(gains[:, solution.reference_antenna, :])
    gains *= np.exp(-1j * reference_phase[:, None, :])
    return replace(
        solution,
        gains=gains,
        gain_valid=gain_valid,
        provenance={
            **solution.provenance,
            "gain_time_model": "second_derivative",
            "gain_time_smoothing_strength": strength,
            "gain_phase_representation": "unwrapped",
        },
    )


def _rbf_covariance(
    first_time_s: np.ndarray,
    second_time_s: np.ndarray,
    length_scale_s: float,
) -> np.ndarray:
    separation = first_time_s[:, None] - second_time_s[None, :]
    return np.exp(-0.5 * (separation / length_scale_s) ** 2)


def _gp_posterior_mean(
    observed_time_s: np.ndarray,
    observed: np.ndarray,
    evaluation_time_s: np.ndarray,
    *,
    length_scale_s: float,
    noise_variance: float,
    mean: complex | float,
) -> np.ndarray:
    covariance = _rbf_covariance(observed_time_s, observed_time_s, length_scale_s)
    covariance += (noise_variance + 1e-10) * np.eye(observed_time_s.size)
    cross = _rbf_covariance(evaluation_time_s, observed_time_s, length_scale_s)
    coefficients = np.linalg.solve(covariance, observed - mean)
    return mean + cross @ coefficients


def circular_gp_gain_solution(
    solution: CalibrationSolution,
    evaluation_time_s: np.ndarray,
    *,
    length_scale_s: float,
    noise_variance: float,
) -> CalibrationSolution:
    """Model log amplitude and unit-complex phase with independent RBF GPs.

    The real-valued amplitude GP acts on log amplitude. The phase GP acts on
    ``exp(1j * phase)`` and its posterior mean is projected back to the unit
    circle. This avoids choosing an unwrap branch across phase wraps.
    """

    if not np.isfinite(length_scale_s) or length_scale_s <= 0:
        raise ValueError("length_scale_s must be finite and positive")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and non-negative")
    evaluation = np.asarray(evaluation_time_s, dtype=np.float64)
    if (
        evaluation.ndim != 1
        or evaluation.size == 0
        or np.any(~np.isfinite(evaluation))
        or np.any(np.diff(evaluation) <= 0)
    ):
        raise ValueError("evaluation_time_s must be finite and strictly increasing")
    gains = np.ones(
        (evaluation.size, solution.antenna_count, solution.receptor_count),
        dtype=np.complex128,
    )
    gain_valid = np.zeros(gains.shape, dtype=bool)
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            values = solution.gains[:, antenna, receptor]
            valid = _valid_gain(values, solution.gain_valid[:, antenna, receptor])
            if not np.any(valid):
                continue
            observed_time = solution.gain_time_s[valid]
            observed = values[valid]
            if observed.size == 1:
                gains[:, antenna, receptor] = observed[0]
                gain_valid[:, antenna, receptor] = True
                continue
            log_amplitude = np.log(np.abs(observed))
            amplitude_mean = float(np.mean(log_amplitude))
            fitted_log_amplitude = _gp_posterior_mean(
                observed_time,
                log_amplitude,
                evaluation,
                length_scale_s=length_scale_s,
                noise_variance=noise_variance,
                mean=amplitude_mean,
            ).real
            unit_phase = observed / np.abs(observed)
            phase_mean = complex(np.mean(unit_phase))
            fitted_phase = _gp_posterior_mean(
                observed_time,
                unit_phase,
                evaluation,
                length_scale_s=length_scale_s,
                noise_variance=noise_variance,
                mean=phase_mean,
            )
            magnitude = np.abs(fitted_phase)
            unstable = magnitude < 1e-8
            if np.any(unstable):
                nearest = np.argmin(
                    np.abs(evaluation[unstable, None] - observed_time[None, :]),
                    axis=1,
                )
                fitted_phase[unstable] = unit_phase[nearest]
                magnitude[unstable] = 1.0
            fitted_phase /= magnitude
            gains[:, antenna, receptor] = np.exp(fitted_log_amplitude) * fitted_phase
            gain_valid[:, antenna, receptor] = True
    reference_phase = np.angle(gains[:, solution.reference_antenna, :])
    gains *= np.exp(-1j * reference_phase[:, None, :])
    return replace(
        solution,
        gains=gains,
        gain_time_s=evaluation,
        gain_valid=gain_valid,
        gain_interval_s=np.zeros(gains.shape, dtype=np.float64),
        provenance={
            **solution.provenance,
            "gain_time_model": "circular_rbf_gp",
            "gain_gp_length_scale_s": length_scale_s,
            "gain_gp_noise_variance": noise_variance,
            "gain_phase_representation": "unit_complex",
        },
    )
