"""CASA-compatible direct Kcross / Df / Xf estimators.

These are initializers and CASA regression oracles, not a self-calibration
engine. First-order Df is labelled ``casa_parallel_preserving``; an exact
2×2 Jones refinement is a later, separate step. Per-channel Df/Xf cannot
hold out frequency until they use a smooth spectral parameterization.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from sl1mjax.calibration import (
    CalibrationSolution,
    apply_calibration,
    corrupt_model,
)
from sl1mjax.calibration_inference import (
    CalibrationFitResult,
    CalibrationSolveConfig,
    _require_model,
    _rms_metrics,
)
from sl1mjax.calibration_terms import CalibrationChain
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    circular_stokes_from_correlations,
)
from sl1mjax.split import VisibilitySplit, calibration_split

_CIRCULAR_COHERENCY = (
    Correlation.RR,
    Correlation.RL,
    Correlation.LR,
    Correlation.LL,
)
_POINT_MODEL_RTOL = 1e-5
_STOKES_LEAKAGE_TOLERANCE = 1e-6


def _require_circular_coherency(block: VisibilityBlock) -> None:
    if tuple(block.correlations) != _CIRCULAR_COHERENCY:
        raise ValueError("polarisation solvers require correlations (RR, RL, LR, LL)")


def _solve_split(
    block: VisibilityBlock,
    split: VisibilitySplit | None,
    config: CalibrationSolveConfig,
) -> VisibilitySplit:
    if split is not None:
        return split
    return calibration_split(
        block,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )


def _held_polarization_solution(
    solution: CalibrationSolution,
    *,
    cross_hand_delay: bool,
    leakage: bool,
    rl_phase: bool,
    parallactic: bool,
) -> CalibrationSolution:
    kwargs: dict[str, Any] = {"apply_parallactic_angle": parallactic}
    if not cross_hand_delay:
        kwargs["cross_hand_delay_s"] = None
        kwargs["cross_hand_delay_valid"] = None
    if not leakage:
        kwargs["leakage"] = None
        kwargs["leakage_frequency_hz"] = None
        kwargs["leakage_valid"] = None
        kwargs["leakage_application"] = "exact"
    if not rl_phase:
        kwargs["rl_phase"] = None
        kwargs["rl_phase_frequency_hz"] = None
        kwargs["rl_phase_valid"] = None
    if parallactic and solution.antenna_position_m is None:
        raise ValueError("parallactic-angle Jones needs antenna_position_m")
    return replace(solution, **kwargs)


def _cal_frequency_hz(
    solution: CalibrationSolution,
    block: VisibilityBlock,
    kind: str,
) -> np.ndarray:
    if kind == "leakage" and solution.leakage_frequency_hz is not None:
        return np.asarray(solution.leakage_frequency_hz, dtype=np.float64)
    if kind == "phase" and solution.rl_phase_frequency_hz is not None:
        return np.asarray(solution.rl_phase_frequency_hz, dtype=np.float64)
    return np.asarray(block.frequency_hz, dtype=np.float64)


def _channel_map(
    block_frequency_hz: np.ndarray, cal_frequency_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argmin(
        np.abs(block_frequency_hz[:, None] - cal_frequency_hz[None, :]), axis=0
    )
    in_band = (cal_frequency_hz >= block_frequency_hz.min()) & (
        cal_frequency_hz <= block_frequency_hz.max()
    )
    if block_frequency_hz.size > 1:
        width = float(np.median(np.diff(np.sort(block_frequency_hz))))
        in_band &= np.abs(block_frequency_hz[indices] - cal_frequency_hz) <= 0.51 * width
    return indices.astype(np.int32), in_band


def _weighted_ratio_spectrum(
    visibility: np.ndarray,
    predicted: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    *,
    slot: int,
    conjugate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least-squares mean of ``y/p`` using weights ``w|p|^2``."""

    predicted_slot = predicted[..., slot]
    visibility_slot = visibility[..., slot]
    usable = (
        mask[..., slot]
        & np.isfinite(predicted_slot)
        & (np.abs(predicted_slot) > 0)
    )
    amplitude_sq = np.abs(predicted_slot) ** 2
    selected_weight = np.where(usable, weight[..., slot] * amplitude_sq, 0.0)
    ratio = np.zeros(visibility.shape[:2], dtype=np.complex128)
    np.divide(
        visibility_slot,
        predicted_slot,
        out=ratio,
        where=selected_weight > 0,
    )
    if conjugate:
        ratio = np.conjugate(ratio)
    numerator = np.sum(selected_weight * ratio, axis=0)
    denominator = np.sum(selected_weight, axis=0)
    return numerator, denominator


def _fit_residual_delay_s(
    frequency_hz: np.ndarray,
    spectrum: np.ndarray,
    weight: np.ndarray,
    reference_frequency_hz: float,
) -> float:
    usable = (weight > 0) & np.isfinite(spectrum) & (np.abs(spectrum) > 0)
    if int(np.count_nonzero(usable)) < 2:
        raise ValueError("cross-hand delay solve needs at least two frequency channels")
    phase = np.unwrap(np.angle(spectrum))
    offset = frequency_hz - reference_frequency_hz
    design = np.stack((-2.0 * np.pi * offset, np.ones(offset.size)), axis=1)
    scale = np.sqrt(np.maximum(weight, 0.0))
    fitted, *_ = np.linalg.lstsq(
        design[usable] * scale[usable, None],
        phase[usable] * scale[usable],
        rcond=None,
    )
    return float(fitted[0])


def _require_unpolarised_point_model(
    block: VisibilityBlock, model: np.ndarray
) -> np.ndarray:
    """Return real Stokes I after refusing polarised or resolved models."""

    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        model, block.correlations
    )
    real_i = np.real(stokes_i)
    peak = float(np.max(np.abs(real_i)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("leakage solving requires positive Stokes I")
    if float(np.max(np.abs(np.imag(stokes_i)))) > _STOKES_LEAKAGE_TOLERANCE * peak:
        raise ValueError("leakage solving requires real Stokes I")
    for name, term in (("Q", stokes_q), ("U", stokes_u), ("V", stokes_v)):
        if float(np.max(np.abs(term))) > _STOKES_LEAKAGE_TOLERANCE * peak:
            raise ValueError(
                "leakage solving requires an unpolarised point-calibrator model "
                f"(nonzero Stokes {name})"
            )
    row_active = np.any(block.active, axis=(1, 2))
    for channel in range(model.shape[1]):
        sample = model[:, channel, :]
        selected = row_active & np.all(np.isfinite(sample), axis=-1)
        if not np.any(selected):
            continue
        reference = sample[int(np.flatnonzero(selected)[0])]
        if not np.allclose(
            sample[selected],
            reference,
            rtol=_POINT_MODEL_RTOL,
            atol=_STOKES_LEAKAGE_TOLERANCE * peak,
        ):
            raise ValueError(
                "leakage solving requires a phase-centred point-calibrator model"
            )
    return real_i


def _active_design_columns(design: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(design, axis=0)
    peak = float(np.max(norms, initial=0.0))
    if peak == 0.0:
        return np.zeros(design.shape[1], dtype=bool)
    tolerance = np.finfo(np.float64).eps * max(design.shape) * peak
    return norms > tolerance


def _full_rank_least_squares(
    design: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(solution, column_valid)`` only when the reduced design is full rank."""

    n_parameter = design.shape[1]
    active = _active_design_columns(design)
    n_active = int(np.count_nonzero(active))
    if n_active == 0:
        return None
    reduced = design[:, active]
    fitted_reduced, _, rank, _singular = np.linalg.lstsq(reduced, observed, rcond=None)
    if int(rank) < n_active:
        return None
    fitted = np.zeros(n_parameter, dtype=np.float64)
    fitted[active] = fitted_reduced
    valid = np.zeros(n_parameter, dtype=bool)
    valid[active] = True
    return fitted, valid


def _finalize_stage(
    block: VisibilityBlock,
    held: CalibrationSolution,
    solution: CalibrationSolution,
    split: VisibilitySplit,
    stage: str,
    extra_provenance: dict[str, Any],
    priors: CalibrationChain | None,
) -> CalibrationFitResult:
    train_mask = split.train & block.active
    holdout_mask = split.holdout & block.active
    if not np.any(train_mask):
        raise ValueError(f"{stage} solve has no training samples")
    baseline_train, baseline_holdout = _rms_metrics(block, held, split, priors=priors)
    train_rms, holdout_rms = _rms_metrics(block, solution, split, priors=priors)
    solution = replace(
        solution,
        provenance={
            **solution.provenance,
            "last_stage": stage,
            "solver_class": "casa_compatible_direct",
            "split_strategy": split.strategy,
            "train_samples": int(np.count_nonzero(train_mask)),
            "holdout_samples": int(np.count_nonzero(holdout_mask)),
            "baseline_train_rms": baseline_train,
            "baseline_holdout_rms": baseline_holdout,
            "frequency_holdout": False,
            **extra_provenance,
        },
    )
    return CalibrationFitResult(solution, train_rms, holdout_rms, (), stage)


def solve_cross_hand_delay(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit | None = None,
    config: CalibrationSolveConfig | None = None,
    apply_parallactic_angle: bool | None = None,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    """Solve a global R–L delay (CASA KCROSS) from RL/LR versus the model."""

    config = CalibrationSolveConfig() if config is None else config
    split = _solve_split(block, split, config)
    _require_circular_coherency(block)
    model = _require_model(block)
    if initial_solution.receptors != (Receptor.R, Receptor.L):
        raise ValueError("cross-hand delay requires receptors (R, L)")
    parallactic = (
        initial_solution.antenna_position_m is not None
        if apply_parallactic_angle is None
        else apply_parallactic_angle
    )
    held = _held_polarization_solution(
        initial_solution,
        cross_hand_delay=False,
        leakage=False,
        rl_phase=False,
        parallactic=parallactic,
    )
    predicted = np.asarray(
        corrupt_model(
            model,
            held,
            time_s=block.time_s,
            frequency_hz=block.frequency_hz,
            antenna1=block.antenna1,
            antenna2=block.antenna2,
            extrapolate=True,
            phase_centre_rad=block.phase_centre_rad,
            priors=priors,
            spectral_window_id=block.spectral_window_id,
        )
    )
    train = split.train & block.active
    rl_num, rl_den = _weighted_ratio_spectrum(
        block.visibility, predicted, block.weight, train, slot=1, conjugate=False
    )
    lr_num, lr_den = _weighted_ratio_spectrum(
        block.visibility, predicted, block.weight, train, slot=2, conjugate=True
    )
    denominator = rl_den + lr_den
    spectrum = np.zeros(block.frequency_hz.size, dtype=np.complex128)
    np.divide(
        rl_num + lr_num,
        denominator,
        out=spectrum,
        where=denominator > 0,
    )
    delay_s = _fit_residual_delay_s(
        block.frequency_hz,
        spectrum,
        denominator,
        initial_solution.reference_frequency_hz,
    )
    n_ant = initial_solution.antenna_count
    delay = np.zeros((n_ant, 2), dtype=np.float64)
    valid = np.ones((n_ant, 2), dtype=bool)
    delay[:, 0] = delay_s
    solution = replace(
        held,
        cross_hand_delay_s=delay,
        cross_hand_delay_valid=valid,
    )
    return _finalize_stage(
        block,
        held,
        solution,
        split,
        "kcross",
        {
            "kcross_model": "global_right_receptor",
            "frequency_parameterization": "global_delay",
        },
        priors,
    )


def _leakage_design(
    antenna: int,
    receptor: str,
    conjugated: bool,
    location: dict[tuple[int, str], int],
    n_parameter: int,
) -> np.ndarray:
    vector = np.zeros((2, n_parameter), dtype=np.float64)
    key = (antenna, receptor)
    if key not in location:
        return vector
    base = location[key]
    vector[0, base] = 1.0
    vector[1, base + 1] = -1.0 if conjugated else 1.0
    return vector


def solve_leakage(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit | None = None,
    config: CalibrationSolveConfig | None = None,
    apply_parallactic_angle: bool = False,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    """Solve first-order Df on an unpolarised point calibrator.

    The returned solution is a CASA-compatible linearised D estimate
    (``leakage_application='casa_parallel_preserving'``), not an exact
    2×2 Jones solution.
    """

    config = CalibrationSolveConfig() if config is None else config
    split = _solve_split(block, split, config)
    _require_circular_coherency(block)
    model = _require_model(block)
    if initial_solution.receptors != (Receptor.R, Receptor.L):
        raise ValueError("leakage solving requires receptors (R, L)")
    stokes_i = _require_unpolarised_point_model(block, model)
    held = _held_polarization_solution(
        initial_solution,
        cross_hand_delay=True,
        leakage=False,
        rl_phase=False,
        parallactic=apply_parallactic_angle,
    )
    corrected = apply_calibration(block, held, extrapolate=True, priors=priors)
    frequencies = _cal_frequency_hz(initial_solution, block, "leakage")
    channel_index, in_band = _channel_map(block.frequency_hz, frequencies)
    n_ant = initial_solution.antenna_count
    reference = initial_solution.reference_antenna
    leakage = np.zeros((n_ant, frequencies.size, 2), dtype=np.complex128)
    leakage_valid = np.zeros((n_ant, frequencies.size, 2), dtype=bool)
    train = split.train & corrected.active
    for cal_channel, (ms_channel, usable) in enumerate(
        zip(channel_index, in_band, strict=True)
    ):
        if not usable:
            continue
        parameter_keys: list[tuple[int, str]] = []
        for antenna in range(n_ant):
            if antenna != reference:
                parameter_keys.append((antenna, "R"))
            parameter_keys.append((antenna, "L"))
        location = {key: 2 * index for index, key in enumerate(parameter_keys)}
        n_parameter = 2 * len(parameter_keys)
        design_rows: list[np.ndarray] = []
        observed: list[float] = []
        appeared: set[int] = set()
        for slot, left, right in ((1, "R", "L"), (2, "L", "R")):
            selected = train[:, ms_channel, slot] & (stokes_i[:, ms_channel] > 0)
            for row in np.flatnonzero(selected):
                first = int(block.antenna1[row])
                second = int(block.antenna2[row])
                scale = float(stokes_i[row, ms_channel])
                vis = corrected.visibility[row, ms_channel, slot]
                weight = np.sqrt(max(float(block.weight[row, ms_channel, slot]), 0.0))
                if weight == 0.0 or scale == 0.0:
                    continue
                design = _leakage_design(
                    first, left, False, location, n_parameter
                ) + _leakage_design(second, right, True, location, n_parameter)
                design_rows.append(weight * scale * design[0])
                observed.append(weight * float(np.real(vis)))
                design_rows.append(weight * scale * design[1])
                observed.append(weight * float(np.imag(vis)))
                appeared.add(first)
                appeared.add(second)
        if not design_rows:
            continue
        fitted = _full_rank_least_squares(
            np.asarray(design_rows), np.asarray(observed)
        )
        if fitted is None:
            continue
        values, column_valid = fitted
        leakage[reference, cal_channel, 0] = 0.0
        leakage_valid[reference, cal_channel, 0] = reference in appeared
        for antenna, receptor in parameter_keys:
            base = location[(antenna, receptor)]
            receptor_slot = 0 if receptor == "R" else 1
            if not (column_valid[base] and column_valid[base + 1]):
                continue
            leakage[antenna, cal_channel, receptor_slot] = (
                values[base] + 1j * values[base + 1]
            )
            leakage_valid[antenna, cal_channel, receptor_slot] = True
    solution = replace(
        held,
        leakage=leakage,
        leakage_frequency_hz=frequencies,
        leakage_valid=leakage_valid,
        leakage_application="casa_parallel_preserving",
    )
    return _finalize_stage(
        block,
        held,
        solution,
        split,
        "leakage",
        {
            "leakage_model": "first_order_Df",
            "leakage_gauge": "D_R[ref]=0",
            "frequency_parameterization": "per_channel",
        },
        priors,
    )


def solve_rl_phase(
    block: VisibilityBlock,
    initial_solution: CalibrationSolution,
    *,
    split: VisibilitySplit | None = None,
    config: CalibrationSolveConfig | None = None,
    apply_parallactic_angle: bool | None = None,
    priors: CalibrationChain | None = None,
) -> CalibrationFitResult:
    """Solve a per-channel R–L phase shared by all antennas (CASA Xf)."""

    config = CalibrationSolveConfig() if config is None else config
    split = _solve_split(block, split, config)
    _require_circular_coherency(block)
    model = _require_model(block)
    if initial_solution.receptors != (Receptor.R, Receptor.L):
        raise ValueError("R–L phase solving requires receptors (R, L)")
    parallactic = (
        initial_solution.antenna_position_m is not None
        if apply_parallactic_angle is None
        else apply_parallactic_angle
    )
    held = _held_polarization_solution(
        initial_solution,
        cross_hand_delay=True,
        leakage=True,
        rl_phase=False,
        parallactic=parallactic,
    )
    corrected = apply_calibration(block, held, extrapolate=True, priors=priors)
    frequencies = _cal_frequency_hz(initial_solution, block, "phase")
    channel_index, in_band = _channel_map(block.frequency_hz, frequencies)
    train = split.train & corrected.active
    rl_num, rl_den = _weighted_ratio_spectrum(
        corrected.visibility, model, block.weight, train, slot=1, conjugate=False
    )
    lr_num, lr_den = _weighted_ratio_spectrum(
        corrected.visibility, model, block.weight, train, slot=2, conjugate=True
    )
    denominator = rl_den + lr_den
    spectrum = np.zeros(block.frequency_hz.size, dtype=np.complex128)
    np.divide(
        rl_num + lr_num,
        denominator,
        out=spectrum,
        where=denominator > 0,
    )
    n_ant = initial_solution.antenna_count
    phase = np.ones((n_ant, frequencies.size), dtype=np.complex128)
    valid = np.zeros((n_ant, frequencies.size), dtype=bool)
    for cal_channel, (ms_channel, usable) in enumerate(
        zip(channel_index, in_band, strict=True)
    ):
        if not usable or denominator[ms_channel] <= 0:
            continue
        value = spectrum[ms_channel]
        if not np.isfinite(value) or np.abs(value) == 0:
            continue
        factor = value / np.abs(value)
        phase[:, cal_channel] = factor
        valid[:, cal_channel] = True
    solution = replace(
        held,
        rl_phase=phase,
        rl_phase_frequency_hz=frequencies,
        rl_phase_valid=valid,
    )
    return _finalize_stage(
        block,
        held,
        solution,
        split,
        "rl_phase",
        {
            "xf_model": "shared_right_receptor_phase",
            "frequency_parameterization": "per_channel",
        },
        priors,
    )


def solve_polarization(
    flux_block: VisibilityBlock,
    leakage_block: VisibilityBlock,
    flux_solution: CalibrationSolution,
    leakage_solution: CalibrationSolution,
    *,
    flux_split: VisibilitySplit | None = None,
    leakage_split: VisibilitySplit | None = None,
    config: CalibrationSolveConfig | None = None,
    priors: CalibrationChain | None = None,
) -> tuple[CalibrationFitResult, CalibrationFitResult, CalibrationFitResult]:
    """Solve Kcross, Df, then Xf in the CASA 3C391 order."""

    config = CalibrationSolveConfig() if config is None else config
    kcross = solve_cross_hand_delay(
        flux_block, flux_solution, split=flux_split, config=config, priors=priors
    )
    leakage_initial = replace(
        leakage_solution,
        cross_hand_delay_s=kcross.solution.cross_hand_delay_s,
        cross_hand_delay_valid=kcross.solution.cross_hand_delay_valid,
        leakage=None,
        leakage_frequency_hz=None,
        leakage_valid=None,
        rl_phase=None,
        rl_phase_frequency_hz=None,
        rl_phase_valid=None,
    )
    leakage = solve_leakage(
        leakage_block,
        leakage_initial,
        split=leakage_split,
        config=config,
        priors=priors,
    )
    flux_with_leakage = replace(
        kcross.solution,
        leakage=leakage.solution.leakage,
        leakage_frequency_hz=leakage.solution.leakage_frequency_hz,
        leakage_valid=leakage.solution.leakage_valid,
        leakage_application=leakage.solution.leakage_application,
    )
    angle = solve_rl_phase(
        flux_block,
        flux_with_leakage,
        split=flux_split,
        config=config,
        priors=priors,
    )
    return kcross, leakage, angle
