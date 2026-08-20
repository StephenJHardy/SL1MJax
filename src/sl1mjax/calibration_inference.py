"""Deterministic Optax solvers for diagonal parallel-hand calibration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from sl1mjax.calibration import (
    CalibrationSolution,
    apply_calibration,
    baseline_jones,
    identity_solution,
    read_calibration,
    write_calibration,
)
from sl1mjax.calibration_terms import CalibrationChain
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.split import VisibilitySplit, calibration_split


@dataclass(frozen=True)
class CalibrationSolveConfig:
    learning_rate: float = 0.03
    iterations: int = 600
    gradient_clip: float = 10.0
    max_chunk_rows: int = 2048
    holdout_fraction: float = 0.2
    seed: int = 0
    checkpoint_every: int = 0


@dataclass(frozen=True)
class CalibrationFitResult:
    solution: CalibrationSolution
    train_rms: float
    holdout_rms: float
    losses: tuple[float, ...]
    stage: str


def _require_model(block: VisibilityBlock) -> np.ndarray:
    if block.model_visibility is None:
        raise ValueError("calibration solving requires model_visibility")
    return block.model_visibility


def _weighted_error(
    observed: Array,
    predicted: Array,
    weight: Array,
    mask: Array,
    max_chunk_rows: int,
) -> Array:
    numerator = jnp.asarray(0.0, dtype=jnp.float64)
    denominator = jnp.asarray(0.0, dtype=jnp.float64)
    for start in range(0, observed.shape[0], max_chunk_rows):
        stop = min(start + max_chunk_rows, observed.shape[0])
        selected = mask[start:stop]
        selected_weight = jnp.where(selected, weight[start:stop], 0.0)
        residual = observed[start:stop] - predicted[start:stop]
        numerator += jnp.sum(selected_weight * jnp.abs(residual) ** 2)
        denominator += jnp.sum(
            selected_weight * jnp.abs(observed[start:stop]) ** 2
        )
    return numerator / jnp.maximum(denominator, jnp.finfo(jnp.float64).tiny)


def _optimize(
    initial: Any,
    objective: Callable[[Any], Array],
    config: CalibrationSolveConfig,
) -> tuple[Any, tuple[float, ...]]:
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adam(config.learning_rate),
    )
    state = optimizer.init(initial)
    parameters = initial
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    losses: list[float] = []
    best_parameters = parameters
    best_loss = np.inf
    for _ in range(config.iterations):
        loss, gradient = value_and_gradient(parameters)
        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError("calibration objective became non-finite")
        updates, state = optimizer.update(gradient, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_parameters = parameters
    return best_parameters, tuple(losses)


def _solution_time_indices(
    block: VisibilityBlock,
) -> tuple[np.ndarray, np.ndarray]:
    times, inverse = np.unique(block.time_s, return_inverse=True)
    return times, inverse.astype(np.int32)


def _fixed_baseline(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    omit: str,
    priors: CalibrationChain | None = None,
) -> Array:
    selected = solution
    if omit == "gain":
        selected = replace(
            selected,
            gains=np.ones_like(selected.gains),
            gain_valid=np.ones_like(selected.gain_valid),
        )
    elif omit == "delay":
        selected = replace(selected, delays_s=np.zeros_like(selected.delays_s))
    elif omit == "bandpass":
        selected = replace(selected, bandpass=np.ones_like(selected.bandpass))
    else:
        raise ValueError(f"unknown omitted term {omit!r}")
    baseline, _ = baseline_jones(
        selected,
        block.time_s,
        block.frequency_hz,
        block.antenna1,
        block.antenna2,
        extrapolate=True,
        phase_centre_rad=block.phase_centre_rad,
        priors=priors,
        spectral_window_id=block.spectral_window_id,
    )
    return jnp.where(jnp.isfinite(baseline), baseline, 1.0 + 0.0j)


def _rms_metrics(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    split: VisibilitySplit,
    priors: CalibrationChain | None = None,
) -> tuple[float, float]:
    corrected = apply_calibration(
        block, solution, extrapolate=True, priors=priors
    )
    model = _require_model(block)

    def metric(mask: np.ndarray) -> float:
        selected = mask & corrected.active
        numerator = np.sum(
            corrected.weight[selected]
            * np.abs(corrected.visibility[selected] - model[selected]) ** 2
        )
        denominator = np.sum(
            corrected.weight[selected] * np.abs(model[selected]) ** 2
        )
        return float(np.sqrt(numerator / denominator))

    return metric(split.train), metric(split.holdout)


def solve_time_gains(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit | None = None,
    config: CalibrationSolveConfig | None = None,
    phase_only: bool = False,
    channels: np.ndarray | None = None,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    config = CalibrationSolveConfig() if config is None else config
    split = (
        calibration_split(
            block,
            holdout_fraction=config.holdout_fraction,
            seed=config.seed,
        )
        if split is None
        else split
    )
    model = _require_model(block)
    times, time_index = _solution_time_indices(block)
    antenna_count = initial_solution.antenna_count
    receptor_count = initial_solution.receptor_count
    first = jnp.asarray(block.antenna1)
    second = jnp.asarray(block.antenna2)
    row_time = jnp.asarray(time_index)
    observed = jnp.asarray(np.where(block.active, block.visibility, 0.0))
    model_array = jnp.asarray(np.where(np.isfinite(model), model, 0.0))
    weight = jnp.asarray(np.where(block.active, block.weight, 0.0))
    train = jnp.asarray(split.train)
    if channels is not None:
        selected_channels = np.zeros(block.frequency_hz.size, dtype=bool)
        selected_channels[np.asarray(channels, dtype=np.int32)] = True
        train &= jnp.asarray(selected_channels[None, :, None])
    fixed = _fixed_baseline(
        block, initial_solution, omit="gain", priors=priors
    )
    initial_log_amplitude = np.zeros((times.size, antenna_count, receptor_count))
    initial_phase = np.zeros_like(initial_log_amplitude)
    if not phase_only:
        for receptor in range(receptor_count):
            selected = np.asarray(train[..., receptor])
            denominator = model[..., receptor] * np.asarray(fixed[..., receptor])
            valid = (
                selected
                & np.isfinite(denominator)
                & (np.abs(denominator) > 0)
            )
            if np.any(valid):
                baseline_amplitude = np.median(
                    np.abs(block.visibility[..., receptor][valid])
                    / np.abs(denominator[valid])
                )
                initial_log_amplitude[..., receptor] = 0.5 * np.log(
                    max(baseline_amplitude, 1e-12)
                )

    def objective(parameters: tuple[Array, Array]) -> Array:
        log_amplitude, phase = parameters
        if phase_only:
            log_amplitude = jnp.zeros_like(log_amplitude)
        reference = initial_solution.reference_antenna
        phase = phase - phase[:, reference : reference + 1, :]
        gains = jnp.exp(log_amplitude + 1j * phase)
        baseline = gains[row_time, first, :] * jnp.conj(
            gains[row_time, second, :]
        )
        predicted = model_array * fixed * baseline[:, None, :]
        return _weighted_error(
            observed,
            predicted,
            weight,
            train,
            config.max_chunk_rows,
        )

    parameters, losses = _optimize(
        (jnp.asarray(initial_log_amplitude), jnp.asarray(initial_phase)),
        objective,
        config,
    )
    log_amplitude, phase = (np.array(value, copy=True) for value in parameters)
    if phase_only:
        log_amplitude.fill(0.0)
    phase -= phase[
        :, initial_solution.reference_antenna : initial_solution.reference_antenna
        + 1,
        :,
    ]
    gains = np.exp(log_amplitude + 1j * phase)
    gain_valid = np.zeros(gains.shape, dtype=bool)
    row_active = np.any(split.train, axis=(1, 2))
    for row in np.flatnonzero(row_active):
        gain_valid[time_index[row], block.antenna1[row], :] = True
        gain_valid[time_index[row], block.antenna2[row], :] = True
    solution = replace(
        initial_solution,
        gains=gains,
        gain_time_s=times,
        gain_valid=gain_valid,
        gain_interval_s=np.zeros(gains.shape),
        provenance={
            **initial_solution.provenance,
            "last_stage": "phase_gain" if phase_only else "time_gain",
        },
    )
    train_rms, holdout_rms = _rms_metrics(
        block, solution, split, priors=priors
    )
    return CalibrationFitResult(
        solution,
        train_rms,
        holdout_rms,
        losses,
        "phase_gain" if phase_only else "time_gain",
    )


def solve_delays(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit,
    config: CalibrationSolveConfig | None = None,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    config = CalibrationSolveConfig() if config is None else config
    model = _require_model(block)
    fixed = _fixed_baseline(
        block, initial_solution, omit="delay", priors=priors
    )
    first = jnp.asarray(block.antenna1)
    second = jnp.asarray(block.antenna2)
    frequencies = jnp.asarray(
        block.frequency_hz - initial_solution.reference_frequency_hz
    )
    observed = jnp.asarray(np.where(block.active, block.visibility, 0.0))
    model_array = jnp.asarray(np.where(np.isfinite(model), model, 0.0))
    weight = jnp.asarray(np.where(block.active, block.weight, 0.0))
    train = jnp.asarray(split.train)

    def objective(delay_ns: Array) -> Array:
        delay_ns = delay_ns - delay_ns[
            initial_solution.reference_antenna : initial_solution.reference_antenna
            + 1,
            :,
        ]
        delay = delay_ns * 1e-9
        jones = jnp.exp(
            -2j * jnp.pi * frequencies[None, :, None] * delay[:, None, :]
        )
        baseline = jones[first] * jnp.conj(jones[second])
        predicted = model_array * fixed * baseline
        return _weighted_error(
            observed,
            predicted,
            weight,
            train,
            config.max_chunk_rows,
        )

    initial_delay_ns = initial_solution.delays_s * 1e9
    delay_ns, losses = _optimize(jnp.asarray(initial_delay_ns), objective, config)
    delay_ns = np.array(delay_ns, copy=True)
    delay_ns -= delay_ns[
        initial_solution.reference_antenna : initial_solution.reference_antenna
        + 1,
        :,
    ]
    solution = replace(
        initial_solution,
        delays_s=delay_ns * 1e-9,
        provenance={**initial_solution.provenance, "last_stage": "delay"},
    )
    train_rms, holdout_rms = _rms_metrics(
        block, solution, split, priors=priors
    )
    return CalibrationFitResult(solution, train_rms, holdout_rms, losses, "delay")


def solve_bandpass(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit,
    config: CalibrationSolveConfig | None = None,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    config = CalibrationSolveConfig() if config is None else config
    model = _require_model(block)
    fixed = _fixed_baseline(
        block, initial_solution, omit="bandpass", priors=priors
    )
    first = jnp.asarray(block.antenna1)
    second = jnp.asarray(block.antenna2)
    observed = jnp.asarray(np.where(block.active, block.visibility, 0.0))
    model_array = jnp.asarray(np.where(np.isfinite(model), model, 0.0))
    weight = jnp.asarray(np.where(block.active, block.weight, 0.0))
    train = jnp.asarray(split.train)
    reference_channel = int(
        np.argmin(
            np.abs(
                block.frequency_hz - initial_solution.reference_frequency_hz
            )
        )
    )
    initial_log_amplitude = np.log(
        np.maximum(np.abs(initial_solution.bandpass), 1e-12)
    )
    initial_phase = np.angle(initial_solution.bandpass)

    def objective(parameters: tuple[Array, Array]) -> Array:
        log_amplitude, phase = parameters
        log_amplitude = log_amplitude - log_amplitude[
            :, reference_channel : reference_channel + 1, :
        ]
        phase = phase - phase[
            initial_solution.reference_antenna : initial_solution.reference_antenna
            + 1,
            :,
            :,
        ]
        bandpass = jnp.exp(log_amplitude + 1j * phase)
        baseline = bandpass[first] * jnp.conj(bandpass[second])
        predicted = model_array * fixed * baseline
        return _weighted_error(
            observed,
            predicted,
            weight,
            train,
            config.max_chunk_rows,
        )

    parameters, losses = _optimize(
        (jnp.asarray(initial_log_amplitude), jnp.asarray(initial_phase)),
        objective,
        config,
    )
    log_amplitude, phase = (np.array(value, copy=True) for value in parameters)
    log_amplitude -= log_amplitude[
        :, reference_channel : reference_channel + 1, :
    ]
    phase -= phase[
        initial_solution.reference_antenna : initial_solution.reference_antenna
        + 1,
        :,
        :,
    ]
    solution = replace(
        initial_solution,
        bandpass=np.exp(log_amplitude + 1j * phase),
        provenance={**initial_solution.provenance, "last_stage": "bandpass"},
    )
    train_rms, holdout_rms = _rms_metrics(
        block, solution, split, priors=priors
    )
    return CalibrationFitResult(
        solution, train_rms, holdout_rms, losses, "bandpass"
    )


def solve_staged_calibration(
    block: VisibilityBlock,
    *,
    reference_antenna: int = 0,
    config: CalibrationSolveConfig | None = None,
    initial_solution: CalibrationSolution | None = None,
    priors: CalibrationChain | None = None,
) -> tuple[CalibrationFitResult, ...]:
    config = CalibrationSolveConfig() if config is None else config
    split = calibration_split(
        block,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )
    solution = (
        identity_solution(
            antenna_count=max(
                int(np.max(block.antenna1)), int(np.max(block.antenna2))
            )
            + 1,
            correlations=block.correlations,
            frequency_hz=block.frequency_hz,
            time_s=np.unique(block.time_s),
            reference_antenna=reference_antenna,
        )
        if initial_solution is None
        else initial_solution
    )
    if (
        solution.correlations != block.correlations
        or solution.bandpass_frequency_hz.shape != block.frequency_hz.shape
        or not np.allclose(
            solution.bandpass_frequency_hz, block.frequency_hz, rtol=1e-12
        )
    ):
        raise ValueError("initial solution does not match the visibility block")
    solution = replace(
        solution,
        reference_frequency_hz=float(
            block.frequency_hz[block.frequency_hz.size // 2]
        ),
        provenance={"solver": "sl1mjax", "staged": True},
    )
    central = np.asarray([block.frequency_hz.size // 2])
    central_gain = solve_time_gains(
        block,
        solution,
        split=split,
        config=config,
        phase_only=False,
        channels=central,
        priors=priors,
    )
    phase = replace(
        central_gain,
        stage="phase_gain",
        solution=replace(
            central_gain.solution,
            provenance={
                **central_gain.solution.provenance,
                "last_stage": "central_phase_gain",
                "conditioning_amplitude_solved": True,
            },
        ),
    )
    delay = solve_delays(
        block, phase.solution, split=split, config=config, priors=priors
    )
    bandpass = solve_bandpass(
        block, delay.solution, split=split, config=config, priors=priors
    )
    gain = solve_time_gains(
        block,
        bandpass.solution,
        split=split,
        config=config,
        priors=priors,
    )
    return phase, delay, bandpass, gain


def transfer_flux_scale(
    primary: CalibrationSolution,
    secondary: CalibrationSolution,
) -> float:
    ratios: list[float] = []
    for antenna in range(primary.antenna_count):
        for receptor in range(primary.receptor_count):
            primary_valid = primary.gain_valid[:, antenna, receptor]
            secondary_valid = secondary.gain_valid[:, antenna, receptor]
            if not np.any(primary_valid) or not np.any(secondary_valid):
                continue
            primary_amplitude = np.median(
                np.abs(primary.gains[primary_valid, antenna, receptor])
            )
            secondary_amplitude = np.median(
                np.abs(secondary.gains[secondary_valid, antenna, receptor])
            )
            ratios.append(float((secondary_amplitude / primary_amplitude) ** 2))
    if not ratios:
        raise ValueError("primary and secondary solutions have no common domains")
    return float(np.median(ratios))


def flux_scale_solution(
    secondary: CalibrationSolution, flux_jy: float
) -> CalibrationSolution:
    """Remove calibrator flux absorbed by unit-model secondary gain solutions."""

    if not np.isfinite(flux_jy) or flux_jy <= 0:
        raise ValueError("flux_jy must be finite and positive")
    return replace(
        secondary,
        gains=secondary.gains / np.sqrt(flux_jy),
        provenance={
            **secondary.provenance,
            "flux_scale": {
                "secondary_flux_jy": flux_jy,
                "gain_amplitude_divisor": float(np.sqrt(flux_jy)),
            },
        },
    )


def save_calibration_checkpoint(
    result: CalibrationFitResult, path: str | Path
) -> None:
    destination = Path(path)
    write_calibration(result.solution, destination)
    destination.with_suffix(".fit.json").write_text(
        json.dumps(
            {
                "stage": result.stage,
                "train_rms": result.train_rms,
                "holdout_rms": result.holdout_rms,
                "losses": list(result.losses),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_calibration_checkpoint(path: str | Path) -> CalibrationFitResult:
    source = Path(path)
    solution = read_calibration(source)
    fit = json.loads(source.with_suffix(".fit.json").read_text(encoding="utf-8"))
    return CalibrationFitResult(
        solution=solution,
        train_rms=float(fit["train_rms"]),
        holdout_rms=float(fit["holdout_rms"]),
        losses=tuple(float(value) for value in fit["losses"]),
        stage=str(fit["stage"]),
    )
