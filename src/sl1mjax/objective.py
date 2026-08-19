"""Correlation-aware data loss and differentiable sky priors."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def effective_weight(
    observation: ArrayLike, weight: ArrayLike, flag: ArrayLike
) -> Array:
    observation_array = jnp.asarray(observation)
    weight_array = jnp.asarray(weight, dtype=observation_array.real.dtype)
    active = (
        ~jnp.asarray(flag, dtype=bool)
        & jnp.isfinite(weight_array)
        & (weight_array > 0)
        & jnp.isfinite(observation_array.real)
        & jnp.isfinite(observation_array.imag)
    )
    return jnp.where(active, weight_array, 0.0)


def weighted_complex_mse(
    prediction: ArrayLike,
    observation: ArrayLike,
    weight: ArrayLike,
    flag: ArrayLike,
) -> Array:
    prediction_array = jnp.asarray(prediction)
    observation_array = jnp.asarray(observation)
    if prediction_array.shape != observation_array.shape:
        raise ValueError("prediction and observation shapes must match")
    active_weight = effective_weight(observation_array, weight, flag)
    residual = jnp.where(active_weight > 0, prediction_array - observation_array, 0.0)
    numerator = jnp.sum(active_weight * jnp.abs(residual) ** 2)
    denominator = jnp.sum(active_weight)
    return jnp.where(denominator > 0, numerator / denominator, jnp.inf)


def normalized_weighted_complex_mse(
    prediction: ArrayLike,
    observation: ArrayLike,
    weight: ArrayLike,
    flag: ArrayLike,
) -> Array:
    """Weighted residual power divided by weighted observed signal power."""

    prediction_array = jnp.asarray(prediction)
    observation_array = jnp.asarray(observation)
    if prediction_array.shape != observation_array.shape:
        raise ValueError("prediction and observation shapes must match")
    active_weight = effective_weight(observation_array, weight, flag)
    residual = jnp.where(
        active_weight > 0, prediction_array - observation_array, 0.0
    )
    numerator = jnp.sum(active_weight * jnp.abs(residual) ** 2)
    denominator = jnp.sum(active_weight * jnp.abs(observation_array) ** 2)
    return jnp.where(denominator > 0, numerator / denominator, jnp.inf)


def sky_prior(
    intensity: ArrayLike,
    *,
    size: int,
    sparsity_weight: float,
    smoothness_weight: float,
) -> Array:
    image = jnp.asarray(intensity).reshape(size, size)
    sparsity = jnp.sum(image)
    horizontal = jnp.mean((image[:, 1:] - image[:, :-1]) ** 2)
    vertical = jnp.mean((image[1:, :] - image[:-1, :]) ** 2)
    return sparsity_weight * sparsity + smoothness_weight * (horizontal + vertical) / 2
