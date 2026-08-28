"""Shared-support circular polarisation: I_RR = I(1+v), I_LL = I(1-v)."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.polarization import Correlation
from sl1mjax.residual_models import (
    RealLinearFit,
    RealLinearSufficientStatistics,
    fit_real_linear_statistics,
    real_linear_statistics,
    residual_power_for_coefficients,
)

CIRCULAR_CONTRAST_LIMIT = 1.0
_STOKES_V_PRODUCTS = {Correlation.RR, Correlation.LL, Correlation.V}


def requires_circular_parallel_hands(correlations: Sequence[Correlation]) -> None:
    """Reject circular contrast on products that cannot carry Stokes V."""

    if not any(correlation in _STOKES_V_PRODUCTS for correlation in correlations):
        raise ValueError("circular_contrast requires RR, LL, and/or Stokes V")


def uses_split_parallel_operator(
    circular_contrast: object,
    beam_weights_rr: object,
    beam_weights_ll: object,
) -> bool:
    """True when RR and LL must be formed separately.

    A missing contrast is v = 0.  The single-DFT Stokes-I path is valid only
    when there is no contrast *and* no hand-specific beam; otherwise a partial
    RR or LL beam would be silently dropped.
    """

    return (
        circular_contrast is not None
        or beam_weights_rr is not None
        or beam_weights_ll is not None
    )


def clip_circular_contrast(contrast: ArrayLike) -> Array:
    """Project circular contrast onto the physical box |v| <= 1."""

    return jnp.clip(
        jnp.asarray(contrast),
        -CIRCULAR_CONTRAST_LIMIT,
        CIRCULAR_CONTRAST_LIMIT,
    )


def parallel_hand_intensities(
    intensity: ArrayLike,
    circular_contrast: ArrayLike | None,
) -> tuple[Array, Array]:
    """Return intrinsic RR and LL flux for a shared Stokes I and contrast v.

    ``circular_contrast`` may be a scalar or one value per component.  A
    missing contrast is v = 0.  Values outside [-1, 1] are clipped.
    """

    selected = jnp.asarray(intensity).reshape(-1)
    if circular_contrast is None:
        return selected, selected
    contrast = jnp.asarray(circular_contrast, dtype=selected.dtype)
    if contrast.size == 1:
        contrast = jnp.broadcast_to(contrast.reshape(()), selected.shape)
    elif contrast.shape != selected.shape:
        raise ValueError(
            "circular_contrast must be scalar or contain one value per component"
        )
    contrast = clip_circular_contrast(contrast)
    return selected * (1.0 + contrast), selected * (1.0 - contrast)


def circular_contrast_response(
    model_visibility: np.ndarray,
    correlations: tuple[Correlation, ...],
) -> np.ndarray:
    """Packed response D such that V(v) = V0 + v D for a global contrast.

    For a shared v, RR scales as (1+v) and LL as (1-v), so D_RR = V0_RR and
    D_LL = -V0_LL.  Packed Stokes V is I v with I = (RR+LL)/2, so D_V is the
    unpolarised Stokes-I visibility.  Stokes I itself is invariant.  Other
    products are left at zero.

    A global v is identical to a balanced visibility-scale split
    RR → (1+δ) M_RR, LL → (1-δ) M_LL.  Separate per-hand scales do not
    break that degeneracy; they only flag unbalanced (one-sided) errors.
    """

    requires_circular_parallel_hands(correlations)
    model = np.asarray(model_visibility)
    if model.ndim < 1 or model.shape[-1] != len(correlations):
        raise ValueError("model_visibility last axis must match correlations")
    response = np.zeros_like(model)
    rr_index = next(
        (
            index
            for index, correlation in enumerate(correlations)
            if correlation is Correlation.RR
        ),
        None,
    )
    ll_index = next(
        (
            index
            for index, correlation in enumerate(correlations)
            if correlation is Correlation.LL
        ),
        None,
    )
    for index, correlation in enumerate(correlations):
        if correlation is Correlation.RR:
            response[..., index] = model[..., index]
        elif correlation is Correlation.LL:
            response[..., index] = -model[..., index]
        elif correlation is Correlation.V:
            if rr_index is not None and ll_index is not None:
                response[..., index] = 0.5 * (
                    model[..., rr_index] + model[..., ll_index]
                )
            elif rr_index is not None:
                response[..., index] = model[..., rr_index]
            elif ll_index is not None:
                response[..., index] = model[..., ll_index]
    return response


def apply_global_circular_contrast(
    model_visibility: np.ndarray,
    correlations: tuple[Correlation, ...],
    circular_contrast: float,
) -> np.ndarray:
    """Scale a v = 0 packed prediction by a global circular contrast."""

    contrast = float(np.clip(circular_contrast, -CIRCULAR_CONTRAST_LIMIT, CIRCULAR_CONTRAST_LIMIT))
    return np.asarray(model_visibility) + contrast * circular_contrast_response(
        model_visibility, correlations
    )


def fit_global_circular_contrast(
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    model_visibility: np.ndarray,
    correlations: tuple[Correlation, ...],
    *,
    ridge_fraction: float = 0.0,
) -> tuple[float, RealLinearFit, RealLinearSufficientStatistics]:
    """Fit one boxed circular contrast against a frozen Stokes-I prediction."""

    response = circular_contrast_response(model_visibility, correlations)
    statistics = real_linear_statistics(
        residual,
        weight,
        mask,
        response[..., None],
    )
    fit = fit_real_linear_statistics(statistics, ridge_fraction=ridge_fraction)
    contrast = float(
        np.clip(fit.coefficients[0], -CIRCULAR_CONTRAST_LIMIT, CIRCULAR_CONTRAST_LIMIT)
    )
    if contrast != float(fit.coefficients[0]):
        fit = RealLinearFit(
            coefficients=np.asarray([contrast], dtype=np.float64),
            ridge_fraction=fit.ridge_fraction,
            ridge_scale=fit.ridge_scale,
            rank=fit.rank,
            residual_power=residual_power_for_coefficients(
                statistics, np.asarray([contrast])
            ),
            weighted_complex_mse=float(
                residual_power_for_coefficients(statistics, np.asarray([contrast]))
                / statistics.weight_sum
            ),
        )
    return contrast, fit, statistics


def correlation_residual_power(
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    correlations: tuple[Correlation, ...],
) -> dict[str, float]:
    """Weighted residual power in each correlation product."""

    values = np.asarray(residual)
    sample_weight = np.asarray(weight, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if values.shape != sample_weight.shape or values.shape != selected.shape:
        raise ValueError("residual, weight, and mask must have the same shape")
    if values.shape[-1] != len(correlations):
        raise ValueError("residual last axis must match correlations")
    powers: dict[str, float] = {}
    usable = selected & np.isfinite(sample_weight) & (sample_weight > 0)
    for index, correlation in enumerate(correlations):
        sample = usable[..., index]
        if not np.any(sample):
            powers[correlation.value] = float("nan")
            continue
        piece = values[..., index][sample]
        finite = np.isfinite(piece.real) & np.isfinite(piece.imag)
        if not np.any(finite):
            powers[correlation.value] = float("nan")
            continue
        selected_weight = sample_weight[..., index][sample][finite]
        powers[correlation.value] = float(
            np.sum(selected_weight * np.abs(piece[finite]) ** 2)
        )
    return powers
