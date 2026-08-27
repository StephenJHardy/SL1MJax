"""Held-out linear diagnostics for structured visibility residuals.

The routines in this module fit small, real-valued perturbations around a
frozen complex visibility model.  They operate on sufficient statistics, so a
large native-resolution scan can be accumulated in row tiles without keeping
one full visibility cube per fitted parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealLinearSufficientStatistics:
    """Normal-equation moments for a real coefficient, complex response model."""

    gram: np.ndarray
    matched: np.ndarray
    residual_power: float
    weight_sum: float
    sample_count: int

    @property
    def parameter_count(self) -> int:
        return int(self.matched.size)


@dataclass(frozen=True)
class RealLinearFit:
    """One regularised fit and its score on the statistics used to fit it."""

    coefficients: np.ndarray
    ridge_fraction: float
    ridge_scale: float
    rank: int
    residual_power: float
    weighted_complex_mse: float


def empty_real_linear_statistics(
    parameter_count: int,
) -> RealLinearSufficientStatistics:
    """Create an empty accumulator for ``parameter_count`` real coefficients."""

    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    return RealLinearSufficientStatistics(
        gram=np.zeros((parameter_count, parameter_count), dtype=np.float64),
        matched=np.zeros(parameter_count, dtype=np.float64),
        residual_power=0.0,
        weight_sum=0.0,
        sample_count=0,
    )


def real_linear_statistics(
    residual: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    responses: np.ndarray,
) -> RealLinearSufficientStatistics:
    """Return weighted real-linear moments for one visibility tile.

    ``responses`` has shape ``residual.shape + (parameter_count,)``.  A real
    coefficient vector ``c`` predicts the complex residual ``responses @ c``.
    """

    values = np.asarray(residual)
    sample_weight = np.asarray(weight, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    design = np.asarray(responses)
    if values.shape != sample_weight.shape or values.shape != selected.shape:
        raise ValueError("residual, weight, and mask must have the same shape")
    if design.ndim != values.ndim + 1 or design.shape[:-1] != values.shape:
        raise ValueError("responses must have shape residual.shape + (parameters,)")
    if design.shape[-1] < 1:
        raise ValueError("responses must contain at least one parameter")
    usable = (
        selected
        & np.isfinite(values.real)
        & np.isfinite(values.imag)
        & np.isfinite(sample_weight)
        & (sample_weight > 0)
        & np.all(np.isfinite(design.real) & np.isfinite(design.imag), axis=-1)
    )
    if not np.any(usable):
        return empty_real_linear_statistics(design.shape[-1])
    selected_design = design[usable].reshape(-1, design.shape[-1])
    selected_residual = values[usable].reshape(-1)
    selected_weight = sample_weight[usable].reshape(-1)
    weighted_design = selected_weight[:, None] * selected_design
    return RealLinearSufficientStatistics(
        gram=np.asarray(
            np.real(selected_design.conj().T @ weighted_design),
            dtype=np.float64,
        ),
        matched=np.asarray(
            np.real(selected_design.conj().T @ (selected_weight * selected_residual)),
            dtype=np.float64,
        ),
        residual_power=float(np.sum(selected_weight * np.abs(selected_residual) ** 2)),
        weight_sum=float(np.sum(selected_weight)),
        sample_count=int(np.count_nonzero(usable)),
    )


def add_real_linear_statistics(
    first: RealLinearSufficientStatistics,
    second: RealLinearSufficientStatistics,
) -> RealLinearSufficientStatistics:
    """Add moments accumulated from disjoint sample sets or row tiles."""

    if first.gram.shape != second.gram.shape:
        raise ValueError("statistics have different parameter counts")
    return RealLinearSufficientStatistics(
        gram=first.gram + second.gram,
        matched=first.matched + second.matched,
        residual_power=first.residual_power + second.residual_power,
        weight_sum=first.weight_sum + second.weight_sum,
        sample_count=first.sample_count + second.sample_count,
    )


def residual_power_for_coefficients(
    statistics: RealLinearSufficientStatistics,
    coefficients: np.ndarray,
) -> float:
    """Evaluate the unregularised weighted residual power from stored moments."""

    selected = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if selected.shape != statistics.matched.shape:
        raise ValueError("coefficients do not match the statistics")
    return float(
        statistics.residual_power
        - 2.0 * np.dot(selected, statistics.matched)
        + selected @ statistics.gram @ selected
    )


def fit_real_linear_statistics(
    statistics: RealLinearSufficientStatistics,
    *,
    ridge_fraction: float = 0.0,
    penalty: np.ndarray | None = None,
) -> RealLinearFit:
    """Fit real coefficients with scale-normalised diagonal ridge regularisation.

    ``ridge_fraction`` multiplies the mean positive diagonal curvature of the
    penalised parameters.  This keeps a ridge grid meaningful when response
    columns use different physical units.  Set a penalty entry to zero for an
    unregularised nuisance coefficient.
    """

    if not np.isfinite(ridge_fraction) or ridge_fraction < 0:
        raise ValueError("ridge_fraction must be finite and non-negative")
    if statistics.weight_sum <= 0 or statistics.sample_count == 0:
        raise ValueError("statistics contain no positive-weight samples")
    parameter_count = statistics.parameter_count
    if penalty is None:
        selected_penalty = np.ones(parameter_count, dtype=np.float64)
    else:
        selected_penalty = np.asarray(penalty, dtype=np.float64).reshape(-1)
        if selected_penalty.shape != (parameter_count,):
            raise ValueError("penalty must contain one value per parameter")
        if np.any(~np.isfinite(selected_penalty)) or np.any(selected_penalty < 0):
            raise ValueError("penalty must be finite and non-negative")
    penalised = selected_penalty > 0
    positive_diagonal = np.diag(statistics.gram)[penalised]
    positive_diagonal = positive_diagonal[positive_diagonal > 0]
    ridge_scale = float(np.mean(positive_diagonal)) if positive_diagonal.size else 1.0
    normal = statistics.gram + np.diag(ridge_fraction * ridge_scale * selected_penalty)
    coefficients, _, rank, _ = np.linalg.lstsq(
        normal,
        statistics.matched,
        rcond=None,
    )
    residual_power = residual_power_for_coefficients(statistics, coefficients)
    return RealLinearFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        ridge_fraction=float(ridge_fraction),
        ridge_scale=ridge_scale,
        rank=int(rank),
        residual_power=residual_power,
        weighted_complex_mse=float(residual_power / statistics.weight_sum),
    )


def score_real_linear_fit(
    statistics: RealLinearSufficientStatistics,
    coefficients: np.ndarray,
) -> tuple[float, float]:
    """Return residual power and weighted complex MSE on another cohort."""

    if statistics.weight_sum <= 0 or statistics.sample_count == 0:
        raise ValueError("statistics contain no positive-weight samples")
    power = residual_power_for_coefficients(statistics, coefficients)
    return power, float(power / statistics.weight_sum)


def scan_residual_response_matrix(
    family: str,
    model_visibility: np.ndarray,
    local_sky_response: np.ndarray,
    pointing_l_response_per_arcsec: np.ndarray,
    pointing_m_response_per_arcsec: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    event_rows: np.ndarray,
    *,
    antenna_ids: tuple[int, ...],
    reference_antenna: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Build one tiled response family for the scan residual diagnostic.

    Every family contains two common nuisance terms: a static fractional flux
    scale and a static correction to the previously discovered sky leaf.  The
    event families then add one structured explanation over the same fixed time
    support.  The returned penalty vector leaves the nuisance terms and simple
    event terms unregularised, but regularises antenna deviations.
    """

    model = np.asarray(model_visibility)
    local = np.asarray(local_sky_response)
    pointing_l = np.asarray(pointing_l_response_per_arcsec)
    pointing_m = np.asarray(pointing_m_response_per_arcsec)
    if not (model.shape == local.shape == pointing_l.shape == pointing_m.shape):
        raise ValueError("all visibility responses must have the same shape")
    if model.ndim != 3:
        raise ValueError(
            "visibility responses must have row, channel, correlation axes"
        )
    first = np.asarray(antenna1, dtype=np.int32).reshape(-1)
    second = np.asarray(antenna2, dtype=np.int32).reshape(-1)
    event = np.asarray(event_rows, dtype=bool).reshape(-1)
    if (
        first.shape != (model.shape[0],)
        or second.shape != first.shape
        or event.shape != first.shape
    ):
        raise ValueError("antenna and event arrays must match the response row axis")
    if reference_antenna not in antenna_ids:
        raise ValueError("reference_antenna must occur in antenna_ids")
    if len(set(antenna_ids)) != len(antenna_ids):
        raise ValueError("antenna_ids must be unique")
    event_cube = event[:, None, None]
    names = ["static_fractional_scale", "static_local_sky_jy"]
    columns = [model, local]
    penalty = [0.0, 0.0]
    if family == "static_nuisance":
        pass
    elif family == "local_sky_event":
        names.append("event_local_sky_jy")
        columns.append(np.where(event_cube, local, 0.0))
        penalty.append(0.0)
    elif family == "common_amplitude_event":
        names.append("event_fractional_scale")
        columns.append(np.where(event_cube, model, 0.0))
        penalty.append(0.0)
    elif family == "common_pointing_event":
        names.extend(
            (
                "event_fractional_scale",
                "event_pointing_l_arcsec",
                "event_pointing_m_arcsec",
            )
        )
        columns.extend(
            (
                np.where(event_cube, model, 0.0),
                np.where(event_cube, pointing_l, 0.0),
                np.where(event_cube, pointing_m, 0.0),
            )
        )
        penalty.extend((0.0, 0.0, 0.0))
    elif family == "antenna_gain_event":
        names.append("event_fractional_scale")
        columns.append(np.where(event_cube, model, 0.0))
        penalty.append(0.0)
        for antenna in antenna_ids:
            if antenna == reference_antenna:
                continue
            incidence = ((first == antenna) | (second == antenna))[:, None, None]
            signed = ((first == antenna).astype(np.float64) - (second == antenna))[
                :, None, None
            ]
            names.extend(
                (f"antenna_{antenna}_log_amplitude", f"antenna_{antenna}_phase_rad")
            )
            columns.extend(
                (
                    np.where(event_cube & incidence, model, 0.0),
                    np.where(event_cube, 1j * signed * model, 0.0),
                )
            )
            penalty.extend((1.0, 1.0))
    else:
        raise ValueError(f"unsupported scan residual family {family!r}")
    return (
        tuple(names),
        np.stack(columns, axis=-1),
        np.asarray(penalty, dtype=np.float64),
    )
