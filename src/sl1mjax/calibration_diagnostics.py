"""Inspectable residual, closure, occupancy, and solution diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from sl1mjax.calibration import (
    CalibrationSolution,
    align_solution_gauge,
    apply_calibration,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.split import VisibilitySplit


@dataclass(frozen=True)
class ResidualDiagnostics:
    normalized_rms: float
    amplitude_median: float
    amplitude_mad: float
    phase_rms_rad: float
    sample_count: int


@dataclass(frozen=True)
class ClosureDiagnostics:
    phase_rms_rad: float
    log_amplitude_rms: float
    triangle_channel_count: int


@dataclass(frozen=True)
class DomainDiagnostics:
    occupancy: np.ndarray
    missing_domains: tuple[tuple[int, int], ...]
    disconnected_time_indices: tuple[int, ...]


@dataclass(frozen=True)
class SolutionComparison:
    gain_log_amplitude_rms: float
    gain_phase_rms_rad: float
    delay_rms_s: float
    bandpass_log_amplitude_rms: float
    bandpass_phase_rms_rad: float


@dataclass(frozen=True)
class CalibrationDiagnostics:
    train: ResidualDiagnostics
    holdout: ResidualDiagnostics
    closure: ClosureDiagnostics
    domains: DomainDiagnostics


def _residual_diagnostics(
    corrected: VisibilityBlock,
    model: np.ndarray,
    selected: np.ndarray,
) -> ResidualDiagnostics:
    selected &= corrected.active & np.isfinite(model)
    residual = corrected.visibility - model
    denominator = np.sum(corrected.weight[selected] * np.abs(model[selected]) ** 2)
    normalized_rms = np.sqrt(
        np.sum(corrected.weight[selected] * np.abs(residual[selected]) ** 2)
        / denominator
    )
    ratio = corrected.visibility[selected] / model[selected]
    amplitude_residual = np.abs(ratio) - 1.0
    phase = np.angle(ratio)
    median = np.median(amplitude_residual)
    return ResidualDiagnostics(
        normalized_rms=float(normalized_rms),
        amplitude_median=float(median),
        amplitude_mad=float(np.median(np.abs(amplitude_residual - median))),
        phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
        sample_count=int(np.sum(selected)),
    )


def _closure_diagnostics(
    corrected: VisibilityBlock, model: np.ndarray
) -> ClosureDiagnostics:
    ratio = np.divide(
        corrected.visibility,
        model,
        out=np.ones_like(corrected.visibility),
        where=corrected.active & np.isfinite(model) & (np.abs(model) > 0),
    )
    phases: list[np.ndarray] = []
    log_amplitudes: list[np.ndarray] = []
    for time in np.unique(corrected.time_s):
        rows = np.flatnonzero(corrected.time_s == time)
        pair_rows = {
            (int(corrected.antenna1[row]), int(corrected.antenna2[row])): row
            for row in rows
            if np.any(corrected.active[row])
        }
        antennas = sorted(
            set(corrected.antenna1[rows]) | set(corrected.antenna2[rows])
        )
        for first, second, third in combinations(antennas, 3):
            keys = ((first, second), (second, third), (first, third))
            if not all(key in pair_rows for key in keys):
                continue
            row12, row23, row13 = (pair_rows[key] for key in keys)
            valid = (
                corrected.active[row12]
                & corrected.active[row23]
                & corrected.active[row13]
            )
            closure = (
                ratio[row12] * ratio[row23] * np.conj(ratio[row13])
            )
            phases.append(np.angle(closure)[valid])
            log_amplitudes.append(
                np.log(np.maximum(np.abs(closure[valid]), 1e-300))
            )
    if not phases:
        return ClosureDiagnostics(np.nan, np.nan, 0)
    phase = np.concatenate(phases)
    log_amplitude = np.concatenate(log_amplitudes)
    return ClosureDiagnostics(
        phase_rms_rad=float(np.sqrt(np.mean(phase**2))),
        log_amplitude_rms=float(np.sqrt(np.mean(log_amplitude**2))),
        triangle_channel_count=int(phase.size),
    )


def _domain_diagnostics(
    block: VisibilityBlock, split: VisibilitySplit, antenna_count: int
) -> DomainDiagnostics:
    times, inverse = np.unique(block.time_s, return_inverse=True)
    occupancy = np.zeros((times.size, antenna_count), dtype=np.int64)
    train_rows = np.any(split.train, axis=(1, 2))
    for row in np.flatnonzero(train_rows):
        occupancy[inverse[row], block.antenna1[row]] += 1
        occupancy[inverse[row], block.antenna2[row]] += 1
    missing = tuple(
        (int(time), int(antenna))
        for time, antenna in np.argwhere(occupancy == 0)
    )
    disconnected: list[int] = []
    for time_index in range(times.size):
        active_antennas = set(
            map(int, np.flatnonzero(occupancy[time_index] > 0))
        )
        if not active_antennas:
            disconnected.append(time_index)
            continue
        adjacency: dict[int, set[int]] = {
            antenna: set() for antenna in active_antennas
        }
        for row in np.flatnonzero(train_rows & (inverse == time_index)):
            first = int(block.antenna1[row])
            second = int(block.antenna2[row])
            adjacency[first].add(second)
            adjacency[second].add(first)
        reached = {next(iter(active_antennas))}
        frontier = list(reached)
        while frontier:
            antenna = frontier.pop()
            for neighbour in adjacency[antenna] - reached:
                reached.add(neighbour)
                frontier.append(neighbour)
        if reached != active_antennas:
            disconnected.append(time_index)
    return DomainDiagnostics(occupancy, missing, tuple(disconnected))


def diagnose_calibration(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    split: VisibilitySplit,
) -> CalibrationDiagnostics:
    if block.model_visibility is None:
        raise ValueError("calibration diagnostics require model_visibility")
    corrected = apply_calibration(block, solution, extrapolate=True)
    return CalibrationDiagnostics(
        train=_residual_diagnostics(corrected, block.model_visibility, split.train),
        holdout=_residual_diagnostics(
            corrected, block.model_visibility, split.holdout
        ),
        closure=_closure_diagnostics(corrected, block.model_visibility),
        domains=_domain_diagnostics(block, split, solution.antenna_count),
    )


def compare_solutions(
    actual: CalibrationSolution, expected: CalibrationSolution
) -> SolutionComparison:
    if (
        actual.antenna_count != expected.antenna_count
        or actual.correlations != expected.correlations
        or actual.bandpass.shape != expected.bandpass.shape
    ):
        raise ValueError("solutions do not share antenna/receptor/frequency domains")
    actual = align_solution_gauge(actual)
    expected = align_solution_gauge(expected)
    gain_log_amplitude: list[float] = []
    gain_phase: list[float] = []
    for antenna in range(actual.antenna_count):
        for receptor in range(actual.receptor_count):
            actual_values = actual.gains[
                actual.gain_valid[:, antenna, receptor], antenna, receptor
            ]
            expected_values = expected.gains[
                expected.gain_valid[:, antenna, receptor], antenna, receptor
            ]
            if not actual_values.size or not expected_values.size:
                continue
            actual_median = np.median(actual_values)
            expected_median = np.median(expected_values)
            gain_log_amplitude.append(
                float(np.log(np.abs(actual_median) / np.abs(expected_median)))
            )
            gain_phase.append(
                float(np.angle(actual_median * np.conj(expected_median)))
            )
    valid_delay = actual.delay_valid & expected.delay_valid
    valid_bandpass = actual.bandpass_valid & expected.bandpass_valid
    bandpass_ratio = actual.bandpass[valid_bandpass] / expected.bandpass[
        valid_bandpass
    ]
    return SolutionComparison(
        gain_log_amplitude_rms=float(
            np.sqrt(np.mean(np.square(gain_log_amplitude)))
        ),
        gain_phase_rms_rad=float(np.sqrt(np.mean(np.square(gain_phase)))),
        delay_rms_s=float(
            np.sqrt(
                np.mean(
                    (actual.delays_s[valid_delay] - expected.delays_s[valid_delay])
                    ** 2
                )
            )
        ),
        bandpass_log_amplitude_rms=float(
            np.sqrt(np.mean(np.log(np.abs(bandpass_ratio)) ** 2))
        ),
        bandpass_phase_rms_rad=float(
            np.sqrt(np.mean(np.angle(bandpass_ratio) ** 2))
        ),
    )


def propose_residual_flags(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    sigma: float = 6.0,
) -> np.ndarray:
    """Return a proposal mask without changing the input block."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if block.model_visibility is None:
        raise ValueError("flag proposals require model_visibility")
    corrected = apply_calibration(block, solution, extrapolate=True)
    residual_amplitude = np.abs(
        corrected.visibility - block.model_visibility
    )
    proposal = np.zeros(block.shape, dtype=bool)
    for channel in range(block.frequency_hz.size):
        for receptor in range(len(block.correlations)):
            active = corrected.active[:, channel, receptor]
            values = residual_amplitude[active, channel, receptor]
            if not values.size:
                continue
            median = np.median(values)
            scale = 1.4826 * np.median(np.abs(values - median))
            threshold = median + sigma * max(scale, np.finfo(float).eps)
            proposal[:, channel, receptor] = (
                active
                & (residual_amplitude[:, channel, receptor] > threshold)
            )
    return proposal
