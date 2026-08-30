"""Global fractional polarisation on a frozen Stokes-I voltage predict.

This is not RM, self-cal, per-pixel polarisation, or spatial V.
Regional q,u is a later step and is not started here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation
from sl1mjax.voltage_beam import VoltageBeamModel
from sl1mjax.voltage_flux_refit import mosaic_weighted_mse, predict_voltage_mosaic


@dataclass(frozen=True)
class GlobalQUFit:
    """One global ``q,u`` with ``v=0`` against a frozen I sky."""

    q: float
    u: float
    v: float
    train_loss: float
    holdout_loss: float | None
    steps: int
    intensity_frozen: bool = True
    regional_polarization: str = "not_started"


def require_circular_coherency(block: VisibilityBlock) -> None:
    needed = {Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL}
    if not needed.issubset(block.correlations):
        raise ValueError("polarisation experiment requires RR, RL, LR, and LL")


def _mosaic_loss(
    q: float,
    u: float,
    intensity: np.ndarray,
    blocks: tuple[VisibilityBlock, ...],
    local_directions: tuple[tuple[np.ndarray, np.ndarray], ...],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    masks: tuple[np.ndarray, ...],
    calibration_state: str,
    config: BeamOperatorConfig,
) -> float:
    predictions = predict_voltage_mosaic(
        intensity,
        blocks,
        local_directions,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        stokes_q=np.asarray(q, dtype=np.float64) * intensity,
        stokes_u=np.asarray(u, dtype=np.float64) * intensity,
    )
    return float(mosaic_weighted_mse(predictions, blocks, masks))


def fit_global_qu_voltage(
    blocks: tuple[VisibilityBlock, ...],
    intensity: np.ndarray,
    local_directions: tuple[tuple[np.ndarray, np.ndarray], ...],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    train_masks: tuple[np.ndarray, ...],
    holdout_masks: tuple[np.ndarray, ...] | None = None,
    calibration_state: str = "casa_parang_true",
    config: BeamOperatorConfig | None = None,
    steps: int = 8,
    learning_rate: float = 0.05,
    finite_difference: float = 1.0e-3,
    initial_q: float = 0.0,
    initial_u: float = 0.0,
) -> GlobalQUFit:
    """Fit one ``q,u`` (``v=0``) by voltage-operator mosaic MSE.

    Stokes I stays frozen. Predicted Q/U are ``q I`` and ``u I``.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    for block in blocks:
        require_circular_coherency(block)
    operator = config or BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=512)
    i_array = np.asarray(intensity, dtype=np.float64).reshape(-1)
    if i_array.size == 0 or np.any(~np.isfinite(i_array)) or np.any(i_array < 0):
        raise ValueError("intensity must be finite, non-negative, and nonempty")
    q = float(initial_q)
    u = float(initial_u)
    step_size = float(learning_rate)
    train_loss = _mosaic_loss(
        q,
        u,
        i_array,
        blocks,
        local_directions,
        beam,
        antenna_position_m=antenna_position_m,
        masks=train_masks,
        calibration_state=calibration_state,
        config=operator,
    )
    for _step in range(steps):
        plus_q = _mosaic_loss(
            q + finite_difference,
            u,
            i_array,
            blocks,
            local_directions,
            beam,
            antenna_position_m=antenna_position_m,
            masks=train_masks,
            calibration_state=calibration_state,
            config=operator,
        )
        plus_u = _mosaic_loss(
            q,
            u + finite_difference,
            i_array,
            blocks,
            local_directions,
            beam,
            antenna_position_m=antenna_position_m,
            masks=train_masks,
            calibration_state=calibration_state,
            config=operator,
        )
        q -= step_size * (plus_q - train_loss) / finite_difference
        u -= step_size * (plus_u - train_loss) / finite_difference
        train_loss = _mosaic_loss(
            q,
            u,
            i_array,
            blocks,
            local_directions,
            beam,
            antenna_position_m=antenna_position_m,
            masks=train_masks,
            calibration_state=calibration_state,
            config=operator,
        )
        step_size *= 0.92
    holdout = None
    if holdout_masks is not None:
        holdout = _mosaic_loss(
            q,
            u,
            i_array,
            blocks,
            local_directions,
            beam,
            antenna_position_m=antenna_position_m,
            masks=holdout_masks,
            calibration_state=calibration_state,
            config=operator,
        )
    return GlobalQUFit(
        q=float(q),
        u=float(u),
        v=0.0,
        train_loss=float(train_loss),
        holdout_loss=holdout,
        steps=steps,
    )


def compare_unpolarised_and_global_qu(
    blocks: tuple[VisibilityBlock, ...],
    intensity: np.ndarray,
    local_directions: tuple[tuple[np.ndarray, np.ndarray], ...],
    beam: VoltageBeamModel,
    *,
    antenna_position_m: np.ndarray,
    sample_masks: tuple[np.ndarray, ...],
    calibration_state: str = "casa_parang_true",
    config: BeamOperatorConfig | None = None,
    q: float,
    u: float,
) -> dict[str, Any]:
    """Score frozen I against I plus global q,u (v=0) on the same masks."""

    operator = config or BeamOperatorConfig(visibility_chunk_size=256, pixel_chunk_size=512)
    unpolarised = predict_voltage_mosaic(
        intensity,
        blocks,
        local_directions,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=operator,
    )
    polarised = predict_voltage_mosaic(
        intensity,
        blocks,
        local_directions,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=operator,
        stokes_q=np.asarray(q, dtype=np.float64) * np.asarray(intensity),
        stokes_u=np.asarray(u, dtype=np.float64) * np.asarray(intensity),
    )
    return {
        "unpolarised_mse": mosaic_weighted_mse(unpolarised, blocks, sample_masks),
        "global_qu_mse": mosaic_weighted_mse(polarised, blocks, sample_masks),
        "unpolarised_predictions": unpolarised,
        "global_qu_predictions": polarised,
    }
