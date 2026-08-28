"""Jones calibration solutions, application, and CASA bridges.

Schema v1 stored one complex gain per correlation product and applied a
scalar ``J_p J_q*`` per hand.  Schema v2 stores one Jones term per *feed
receptor* and applies ``C_obs = J_p C_sky J_q^H``.
Diagonal G, K, and B promote to diagonal 2×2 matrices.  Schema 1 files are
read and promoted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.calibration_terms import (
    CalibrationChain,
    CalibrationCoordinates,
)
from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    apply_jones_to_coherency,
    correlation_receptor_pair,
    diagonal_jones_matrices,
    invert_jones,
    pack_coherency,
    receptors_for_correlations,
    unpack_coherency,
)

CALIBRATION_SCHEMA_VERSION = 2
SUPPORTED_CALIBRATION_SCHEMA_VERSIONS = frozenset({1, 2})
SPEED_OF_LIGHT_M_S = 299_792_458.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CalibrationSolution:
    """Portable Jones terms on explicit time/frequency coordinates.

    ``gains``, ``delays_s``, and ``bandpass`` are diagonal per *receptor*
    (R/L or X/Y), not per packed correlation product.  Application builds
    *diagonal* 2×2 Jones matrices from those terms and forms
    ``J_p C J_q^H``.  Off-diagonal leakage and R–L phase are not stored
    yet.  Full 2×2 matrices can be applied through
    ``apply_jones_to_coherency``; ``apply_calibration`` itself still
    constructs only the diagonal promotion of G/K/B.
    """

    gains: np.ndarray
    gain_time_s: np.ndarray
    gain_valid: np.ndarray
    gain_interval_s: np.ndarray
    delays_s: np.ndarray
    delay_valid: np.ndarray
    bandpass: np.ndarray
    bandpass_frequency_hz: np.ndarray
    bandpass_valid: np.ndarray
    correlations: tuple[Correlation, ...]
    reference_antenna: int
    reference_frequency_hz: float
    interpolation: str = "nearest"
    receptors: tuple[Receptor, ...] | None = None
    antenna_position_offset_m: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gains", np.asarray(self.gains, dtype=np.complex128))
        object.__setattr__(self, "gain_time_s", np.asarray(self.gain_time_s, dtype=np.float64))
        object.__setattr__(self, "gain_valid", np.asarray(self.gain_valid, dtype=bool))
        object.__setattr__(
            self,
            "gain_interval_s",
            np.asarray(self.gain_interval_s, dtype=np.float64),
        )
        object.__setattr__(self, "delays_s", np.asarray(self.delays_s, dtype=np.float64))
        object.__setattr__(self, "delay_valid", np.asarray(self.delay_valid, dtype=bool))
        object.__setattr__(self, "bandpass", np.asarray(self.bandpass, dtype=np.complex128))
        object.__setattr__(
            self,
            "bandpass_frequency_hz",
            np.asarray(self.bandpass_frequency_hz, dtype=np.float64),
        )
        object.__setattr__(self, "bandpass_valid", np.asarray(self.bandpass_valid, dtype=bool))
        object.__setattr__(
            self,
            "correlations",
            tuple(Correlation(value) for value in self.correlations),
        )
        inferred = receptors_for_correlations(self.correlations)
        if self.receptors is None:
            object.__setattr__(self, "receptors", inferred)
        else:
            object.__setattr__(
                self,
                "receptors",
                tuple(Receptor(value) for value in self.receptors),
            )
        if self.antenna_position_offset_m is not None:
            object.__setattr__(
                self,
                "antenna_position_offset_m",
                np.asarray(self.antenna_position_offset_m, dtype=np.float64),
            )
        self.validate()

    @property
    def antenna_count(self) -> int:
        return int(self.delays_s.shape[0])

    @property
    def receptor_count(self) -> int:
        assert self.receptors is not None
        return len(self.receptors)

    def validate(self) -> None:
        assert self.receptors is not None
        if not self.receptors:
            raise ValueError("at least one receptor is required")
        if len(set(self.receptors)) != len(self.receptors):
            raise ValueError("receptors must be unique")
        required = receptors_for_correlations(self.correlations)
        if any(receptor not in self.receptors for receptor in required):
            raise ValueError("receptors must include every feed used by correlations")
        receptors = self.receptor_count
        if self.gains.ndim != 3 or self.gains.shape[2] != receptors:
            raise ValueError("gains must have shape (time, antenna, receptor)")
        if self.gain_time_s.shape != (self.gains.shape[0],):
            raise ValueError("gain_time_s must match the gain time axis")
        if self.gain_valid.shape != self.gains.shape:
            raise ValueError("gain_valid must match gains")
        if self.gain_interval_s.shape != self.gains.shape:
            raise ValueError("gain_interval_s must match gains")
        if self.delays_s.shape != (self.gains.shape[1], receptors):
            raise ValueError("delays_s must have shape (antenna, receptor)")
        if self.delay_valid.shape != self.delays_s.shape:
            raise ValueError("delay_valid must match delays_s")
        expected_bandpass = (
            self.gains.shape[1],
            self.bandpass_frequency_hz.size,
            receptors,
        )
        if self.bandpass.shape != expected_bandpass:
            raise ValueError(f"bandpass must have shape {expected_bandpass}")
        if self.bandpass_valid.shape != self.bandpass.shape:
            raise ValueError("bandpass_valid must match bandpass")
        if not 0 <= self.reference_antenna < self.antenna_count:
            raise ValueError("reference_antenna is outside the antenna axis")
        if self.interpolation not in {"nearest", "linear"}:
            raise ValueError("interpolation must be nearest or linear")
        if self.antenna_position_offset_m is not None and self.antenna_position_offset_m.shape != (
            self.antenna_count,
            3,
        ):
            raise ValueError("antenna_position_offset_m must have shape (antenna, 3)")

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = (
            self.gains,
            self.gain_time_s,
            self.gain_valid,
            self.gain_interval_s,
            self.delays_s,
            self.delay_valid,
            self.bandpass,
            self.bandpass_frequency_hz,
            self.bandpass_valid,
            self.antenna_position_offset_m,
        )
        auxiliary = (
            self.correlations,
            self.receptors,
            self.reference_antenna,
            self.reference_frequency_hz,
            self.interpolation,
            json.dumps(self.provenance, sort_keys=True),
        )
        return children, auxiliary

    @classmethod
    def tree_unflatten(
        cls, auxiliary: tuple[Any, ...], children: tuple[Any, ...]
    ) -> CalibrationSolution:
        (
            correlations,
            receptors,
            reference_antenna,
            reference_frequency_hz,
            interpolation,
            raw,
        ) = auxiliary
        return cls(
            gains=children[0],
            gain_time_s=children[1],
            gain_valid=children[2],
            gain_interval_s=children[3],
            delays_s=children[4],
            delay_valid=children[5],
            bandpass=children[6],
            bandpass_frequency_hz=children[7],
            bandpass_valid=children[8],
            correlations=correlations,
            reference_antenna=reference_antenna,
            reference_frequency_hz=reference_frequency_hz,
            interpolation=interpolation,
            receptors=receptors,
            antenna_position_offset_m=children[9],
            provenance=json.loads(raw),
        )


@dataclass(frozen=True)
class CasaCalibrationGoldenCase:
    label: str
    field_id: int
    block: VisibilityBlock
    corrected_visibility: np.ndarray
    post_apply_flag: np.ndarray
    metadata: dict[str, Any]


def identity_solution(
    *,
    antenna_count: int,
    correlations: tuple[Correlation, ...],
    frequency_hz: ArrayLike,
    time_s: ArrayLike | tuple[float, ...] = (0.0,),
    reference_antenna: int = 0,
) -> CalibrationSolution:
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    products = tuple(Correlation(value) for value in correlations)
    receptors = receptors_for_correlations(products)
    n_receptor = len(receptors)
    return CalibrationSolution(
        gains=np.ones((times.size, antenna_count, n_receptor), dtype=np.complex128),
        gain_time_s=times,
        gain_valid=np.ones((times.size, antenna_count, n_receptor), dtype=bool),
        gain_interval_s=np.zeros((times.size, antenna_count, n_receptor), dtype=np.float64),
        delays_s=np.zeros((antenna_count, n_receptor)),
        delay_valid=np.ones((antenna_count, n_receptor), dtype=bool),
        bandpass=np.ones((antenna_count, frequencies.size, n_receptor), dtype=np.complex128),
        bandpass_frequency_hz=frequencies,
        bandpass_valid=np.ones((antenna_count, frequencies.size, n_receptor), dtype=bool),
        correlations=products,
        reference_antenna=reference_antenna,
        reference_frequency_hz=float(np.mean(frequencies)),
        receptors=receptors,
    )


def _interpolate_gains(solution: CalibrationSolution, time_s: np.ndarray) -> np.ndarray:
    output = np.empty(
        (time_s.size, solution.antenna_count, solution.receptor_count),
        dtype=np.complex128,
    )
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            valid = solution.gain_valid[:, antenna, receptor]
            if not np.any(valid):
                output[:, antenna, receptor] = np.nan + 1j * np.nan
                continue
            times = solution.gain_time_s[valid]
            values = solution.gains[valid, antenna, receptor]
            if solution.interpolation == "nearest" or times.size == 1:
                indices = np.argmin(np.abs(time_s[:, None] - times[None, :]), axis=1)
                output[:, antenna, receptor] = values[indices]
            else:
                amplitude = np.interp(time_s, times, np.abs(values))
                phase = np.interp(time_s, times, np.unwrap(np.angle(values)))
                output[:, antenna, receptor] = amplitude * np.exp(1j * phase)
    return output


def _frequency_indices(
    solution: CalibrationSolution,
    frequency_hz: np.ndarray,
    *,
    extrapolate: bool,
) -> np.ndarray:
    distance = np.abs(frequency_hz[:, None] - solution.bandpass_frequency_hz[None, :])
    indices = np.argmin(distance, axis=1)
    tolerance = np.maximum(np.abs(frequency_hz), 1.0) * 1e-10
    if not extrapolate and np.any(distance[np.arange(frequency_hz.size), indices] > tolerance):
        raise ValueError("requested frequency is outside the bandpass coordinate grid")
    return indices


def _sampled_antenna_jones(
    solution: CalibrationSolution,
    time_s: ArrayLike,
    frequency_hz: ArrayLike,
    *,
    extrapolate: bool,
    phase_centre_rad: tuple[float, float] | None,
    priors: CalibrationChain | None,
    spectral_window_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal antenna Jones and validity: row, channel, antenna, receptor."""

    times = np.asarray(time_s, dtype=np.float64)
    frequencies = np.asarray(frequency_hz, dtype=np.float64)
    gains = _interpolate_gains(solution, times)
    frequency_indices = _frequency_indices(solution, frequencies, extrapolate=extrapolate)
    bandpass = solution.bandpass[:, frequency_indices, :]
    bandpass_valid = solution.bandpass_valid[:, frequency_indices, :]
    offset_frequency = frequencies - solution.reference_frequency_hz
    delay = np.exp(-2j * np.pi * solution.delays_s[:, None, :] * offset_frequency[None, :, None])
    antenna_jones = (
        gains[:, None, :, :]
        * np.transpose(bandpass, (1, 0, 2))[None, :, :, :]
        * np.transpose(delay, (1, 0, 2))[None, :, :, :]
    )
    if solution.antenna_position_offset_m is not None and np.any(
        solution.antenna_position_offset_m
    ):
        if phase_centre_rad is None:
            raise ValueError("phase_centre_rad is required for antenna-position calibration")
        mjd = times / 86400.0
        julian_date = mjd + 2_400_000.5
        gmst_rad = np.deg2rad(
            np.mod(
                280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0),
                360.0,
            )
        )
        right_ascension, declination = phase_centre_rad
        hour_angle = right_ascension - gmst_rad
        direction_ecef = np.stack(
            (
                np.cos(declination) * np.cos(hour_angle),
                np.cos(declination) * np.sin(hour_angle),
                np.full(times.shape, np.sin(declination)),
            ),
            axis=1,
        )
        path_error_m = direction_ecef @ solution.antenna_position_offset_m.T
        position_jones = np.exp(
            2j * np.pi * frequencies[None, :, None] * path_error_m[:, None, :] / SPEED_OF_LIGHT_M_S
        )
        antenna_jones *= position_jones[:, :, :, None]
    nearest_gain = np.empty(
        (times.size, solution.antenna_count, solution.receptor_count), dtype=bool
    )
    for antenna in range(solution.antenna_count):
        for selected_receptor in range(solution.receptor_count):
            valid = solution.gain_valid[:, antenna, selected_receptor]
            if not np.any(valid):
                nearest_gain[:, antenna, selected_receptor] = False
                continue
            valid_times = solution.gain_time_s[valid]
            valid_intervals = solution.gain_interval_s[valid, antenna, selected_receptor]
            indices = np.argmin(np.abs(times[:, None] - valid_times[None, :]), axis=1)
            domain_valid = (
                np.ones(times.size, dtype=bool)
                if extrapolate
                else np.abs(times - valid_times[indices]) <= valid_intervals[indices] / 2
            )
            if solution.interpolation == "linear" and valid_times.size > 1:
                domain_valid |= (times >= valid_times.min()) & (times <= valid_times.max())
            nearest_gain[:, antenna, selected_receptor] = domain_valid & np.isfinite(
                solution.gains[valid, antenna, selected_receptor][indices]
            )
    antenna_valid = (
        nearest_gain[:, None, :, :]
        & solution.delay_valid[None, None, :, :]
        & np.transpose(bandpass_valid, (1, 0, 2))[None, :, :, :]
        & np.isfinite(antenna_jones)
        & (np.abs(antenna_jones) > 0)
    )
    if priors is not None and priors.terms:
        if priors.antenna_position_m is None:
            raise ValueError("prior calibration requires antenna positions")
        if phase_centre_rad is None:
            raise ValueError("prior calibration requires phase_centre_rad")
        prior_jones, prior_valid = priors.evaluate(
            CalibrationCoordinates(
                time_s=times,
                frequency_hz=frequencies,
                spectral_window_id=spectral_window_id,
                phase_centre_rad=phase_centre_rad,
                antenna_position_m=priors.antenna_position_m,
                receptor_count=solution.receptor_count,
            )
        )
        antenna_jones = antenna_jones * prior_jones
        antenna_valid = antenna_valid & prior_valid
    return antenna_jones, antenna_valid


def _gather_antenna(
    values: np.ndarray,
    antenna: np.ndarray,
) -> np.ndarray:
    row = np.arange(values.shape[0])[:, None, None]
    channel = np.arange(values.shape[1])[None, :, None]
    receptor = np.arange(values.shape[3])[None, None, :]
    return values[row, channel, antenna[:, None, None], receptor]


def _product_validity(
    antenna_valid: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> np.ndarray:
    first = _gather_antenna(antenna_valid, antenna1)
    second = _gather_antenna(antenna_valid, antenna2)
    index = {receptor: i for i, receptor in enumerate(receptors)}
    valid = np.empty((*first.shape[:2], len(correlations)), dtype=bool)
    for slot, correlation in enumerate(correlations):
        left, right = correlation_receptor_pair(correlation)
        valid[..., slot] = first[..., index[left]] & second[..., index[right]]
    return valid


def _product_scale(
    jones_p: np.ndarray,
    jones_q: np.ndarray,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> np.ndarray:
    """Diagonal Jones product per packed correlation: ``J_p,ii J_q,jj*``."""

    index = {receptor: i for i, receptor in enumerate(receptors)}
    scale = np.empty((*jones_p.shape[:-2], len(correlations)), dtype=np.complex128)
    for slot, correlation in enumerate(correlations):
        left, right = correlation_receptor_pair(correlation)
        i = index[left]
        j = index[right]
        scale[..., slot] = jones_p[..., i, i] * np.conj(jones_q[..., j, j])
    return scale


def _jones_matrices_for_block(
    solution: CalibrationSolution,
    time_s: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    extrapolate: bool,
    phase_centre_rad: tuple[float, float] | None,
    priors: CalibrationChain | None,
    spectral_window_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``J_p``, ``J_q``, and per-receptor antenna validity."""

    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    antenna_jones, antenna_valid = _sampled_antenna_jones(
        solution,
        time_s,
        frequency_hz,
        extrapolate=extrapolate,
        phase_centre_rad=phase_centre_rad,
        priors=priors,
        spectral_window_id=spectral_window_id,
    )
    matrices = diagonal_jones_matrices(antenna_jones)
    row = np.arange(matrices.shape[0])[:, None, None, None]
    channel = np.arange(matrices.shape[1])[None, :, None, None]
    receptor_i = np.arange(matrices.shape[3])[None, None, :, None]
    receptor_j = np.arange(matrices.shape[4])[None, None, None, :]
    jones_p = matrices[row, channel, first[:, None, None, None], receptor_i, receptor_j]
    jones_q = matrices[row, channel, second[:, None, None, None], receptor_i, receptor_j]
    return jones_p, jones_q, antenna_valid


def baseline_jones(
    solution: CalibrationSolution,
    time_s: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    *,
    extrapolate: bool = False,
    phase_centre_rad: tuple[float, float] | None = None,
    priors: CalibrationChain | None = None,
    spectral_window_id: int = 0,
) -> tuple[Array, Array]:
    """Evaluate `J_p conj(J_q)` and its validity for each receptor."""

    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    antenna_jones, antenna_valid = _sampled_antenna_jones(
        solution,
        time_s,
        frequency_hz,
        extrapolate=extrapolate,
        phase_centre_rad=phase_centre_rad,
        priors=priors,
        spectral_window_id=spectral_window_id,
    )
    j1 = _gather_antenna(antenna_jones, first)
    j2 = _gather_antenna(antenna_jones, second)
    v1 = _gather_antenna(antenna_valid, first)
    v2 = _gather_antenna(antenna_valid, second)
    return jnp.asarray(j1 * np.conj(j2)), jnp.asarray(v1 & v2)


def corrupt_model(
    model_visibility: ArrayLike,
    solution: CalibrationSolution,
    *,
    time_s: ArrayLike,
    frequency_hz: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    extrapolate: bool = False,
    phase_centre_rad: tuple[float, float] | None = None,
    priors: CalibrationChain | None = None,
    spectral_window_id: int = 0,
) -> Array:
    assert solution.receptors is not None
    model = np.asarray(model_visibility)
    if model.shape[-1] != len(solution.correlations):
        raise ValueError("model visibility last axis must match solution correlations")
    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    jones_p, jones_q, antenna_valid = _jones_matrices_for_block(
        solution,
        time_s,
        frequency_hz,
        first,
        second,
        extrapolate=extrapolate,
        phase_centre_rad=phase_centre_rad,
        priors=priors,
        spectral_window_id=spectral_window_id,
    )
    valid = _product_validity(
        antenna_valid,
        first,
        second,
        solution.correlations,
        solution.receptors,
    )
    packed = pack_coherency(model, solution.correlations, solution.receptors)
    corrupted = unpack_coherency(
        apply_jones_to_coherency(packed, jones_p, jones_q),
        solution.correlations,
        solution.receptors,
    )
    return jnp.asarray(np.where(valid, corrupted, 0.0))


def apply_calibration(
    block: VisibilityBlock,
    solution: CalibrationSolution,
    *,
    propagate_weights: bool = False,
    extrapolate: bool = False,
    priors: CalibrationChain | None = None,
) -> VisibilityBlock:
    assert solution.receptors is not None
    required = receptors_for_correlations(block.correlations)
    if any(receptor not in solution.receptors for receptor in required):
        raise ValueError("solution receptors must cover every feed in the block")
    jones_p, jones_q, antenna_valid = _jones_matrices_for_block(
        solution,
        block.time_s,
        block.frequency_hz,
        block.antenna1,
        block.antenna2,
        extrapolate=extrapolate,
        phase_centre_rad=block.phase_centre_rad,
        priors=priors,
        spectral_window_id=block.spectral_window_id,
    )
    valid_array = _product_validity(
        antenna_valid,
        np.asarray(block.antenna1, dtype=np.int32),
        np.asarray(block.antenna2, dtype=np.int32),
        block.correlations,
        solution.receptors,
    )
    jones_p_inv = invert_jones(jones_p)
    jones_q_inv = invert_jones(jones_q)
    finite_inverse = np.all(np.isfinite(jones_p_inv), axis=(-2, -1)) & np.all(
        np.isfinite(jones_q_inv), axis=(-2, -1)
    )
    valid_array = valid_array & finite_inverse[..., None]
    if not extrapolate and np.any(block.active & ~valid_array):
        raise ValueError("active visibility lies outside the calibration solution validity domain")
    packed = pack_coherency(block.visibility, block.correlations, solution.receptors)
    corrected = unpack_coherency(
        apply_jones_to_coherency(packed, jones_p_inv, jones_q_inv),
        block.correlations,
        solution.receptors,
    )
    corrected = np.where(valid_array, corrected, 0.0)
    flag = block.flag | ~valid_array
    if propagate_weights:
        scale = _product_scale(jones_p, jones_q, block.correlations, solution.receptors)
        weight = block.weight * np.abs(scale) ** 2
    else:
        weight = block.weight.copy()
    provenance = {
        **dict(block.provenance),
        "calibration": {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "reference_antenna": solution.reference_antenna,
            "reference_frequency_hz": solution.reference_frequency_hz,
            "interpolation": solution.interpolation,
            "propagate_weights": propagate_weights,
            "extrapolate": extrapolate,
            "solution_provenance": solution.provenance,
            "prior_provenance": None if priors is None else priors.provenance,
            "prior_terms": ([] if priors is None else [term.kind for term in priors.terms]),
            "receptors": [value.value for value in solution.receptors],
        },
    }
    return replace(
        block,
        visibility=corrected,
        weight=weight,
        flag=flag,
        provenance=provenance,
    )


def apply_calibration_dataset(
    dataset: VisibilityDataset,
    solution: CalibrationSolution,
    *,
    propagate_weights: bool = False,
    extrapolate: bool = False,
    priors: CalibrationChain | None = None,
) -> VisibilityDataset:
    return VisibilityDataset(
        tuple(
            apply_calibration(
                block,
                solution,
                propagate_weights=propagate_weights,
                extrapolate=extrapolate,
                priors=priors,
            )
            for block in dataset.blocks
        ),
        {
            **dict(dataset.provenance),
            "calibration_applied": True,
        },
        dataset.metadata,
    )


def align_solution_gauge(solution: CalibrationSolution) -> CalibrationSolution:
    gains = solution.gains.copy()
    reference = gains[:, solution.reference_antenna, :]
    phase = np.exp(-1j * np.angle(reference))
    gains *= phase[:, None, :]
    delays = solution.delays_s - solution.delays_s[solution.reference_antenna, :][None, :]
    bandpass = solution.bandpass.copy()
    reference_bandpass_phase = np.exp(-1j * np.angle(bandpass[solution.reference_antenna, :, :]))
    bandpass *= reference_bandpass_phase[None, :, :]
    return replace(solution, gains=gains, delays_s=delays, bandpass=bandpass)


def write_calibration(solution: CalibrationSolution, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            gains=solution.gains,
            gain_time_s=solution.gain_time_s,
            gain_valid=solution.gain_valid,
            gain_interval_s=solution.gain_interval_s,
            delays_s=solution.delays_s,
            delay_valid=solution.delay_valid,
            bandpass=solution.bandpass,
            bandpass_frequency_hz=solution.bandpass_frequency_hz,
            bandpass_valid=solution.bandpass_valid,
            antenna_position_offset_m=(
                np.empty((0, 3))
                if solution.antenna_position_offset_m is None
                else solution.antenna_position_offset_m
            ),
        )
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "correlations": [value.value for value in solution.correlations],
                "receptors": [value.value for value in solution.receptors or ()],
                "reference_antenna": solution.reference_antenna,
                "reference_frequency_hz": solution.reference_frequency_hz,
                "interpolation": solution.interpolation,
                "provenance": solution.provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def read_calibration(path: str | Path) -> CalibrationSolution:
    source = Path(path)
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_CALIBRATION_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported calibration schema {schema_version}")
    correlations = tuple(Correlation(value) for value in metadata["correlations"])
    receptors: tuple[Receptor, ...] | None
    if schema_version == 1:
        receptors = None
    else:
        receptors = tuple(Receptor(value) for value in metadata["receptors"])
    provenance = dict(metadata.get("provenance", {}))
    if schema_version != CALIBRATION_SCHEMA_VERSION:
        provenance = {
            **provenance,
            "promoted_from_schema": schema_version,
        }
    with np.load(source, allow_pickle=True) as arrays:
        antenna_position = arrays["antenna_position_offset_m"]
        if antenna_position.size == 0:
            antenna_position = None
        return CalibrationSolution(
            gains=arrays["gains"],
            gain_time_s=arrays["gain_time_s"],
            gain_valid=arrays["gain_valid"],
            gain_interval_s=arrays["gain_interval_s"],
            delays_s=arrays["delays_s"],
            delay_valid=arrays["delay_valid"],
            bandpass=arrays["bandpass"],
            bandpass_frequency_hz=arrays["bandpass_frequency_hz"],
            bandpass_valid=arrays["bandpass_valid"],
            correlations=correlations,
            reference_antenna=int(metadata["reference_antenna"]),
            reference_frequency_hz=float(metadata["reference_frequency_hz"]),
            interpolation=str(metadata["interpolation"]),
            receptors=receptors,
            antenna_position_offset_m=antenna_position,
            provenance=provenance,
        )


def _pivot_casa_table(
    arrays: Any,
    prefix: str,
    parameter: str,
    *,
    field_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    antenna = np.asarray(arrays[f"{prefix}_antenna1"], dtype=np.int32)
    times = np.asarray(arrays[f"{prefix}_time"], dtype=np.float64)
    flag = np.asarray(arrays[f"{prefix}_flag"], dtype=bool)
    values = np.asarray(arrays[f"{prefix}_{parameter}"])
    selected = np.ones(antenna.size, dtype=bool)
    field_key = f"{prefix}_field_id"
    if field_id is not None and field_key in arrays:
        selected &= np.asarray(arrays[field_key], dtype=np.int32) == field_id
    antenna = antenna[selected]
    times = times[selected]
    flag = flag[selected]
    values = values[selected]
    unique_times = np.unique(times)
    antenna_count = int(np.max(antenna)) + 1
    value_shape = values.shape[1:]
    output = np.ones((unique_times.size, antenna_count, *value_shape), values.dtype)
    valid = np.zeros(output.shape, dtype=bool)
    time_index = {value: index for index, value in enumerate(unique_times)}
    for row, (time, selected_antenna) in enumerate(zip(times, antenna, strict=True)):
        index = time_index[time]
        output[index, selected_antenna] = values[row]
        row_valid = ~flag[row]
        target = valid[index, selected_antenna]
        if target.shape == row_valid.shape:
            valid[index, selected_antenna] = row_valid
        else:
            valid[index, selected_antenna] = np.all(row_valid)
    return unique_times, output, valid


def import_casa_golden_solution(
    fixture: str | Path,
    *,
    field_id: int,
    interpolation: str = "nearest",
    gain_table: Literal["gain", "flux_gain"] = "flux_gain",
) -> CalibrationSolution:
    """Translate exported CASA K/B/G tables for one calibrator field."""

    source = Path(fixture)
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(source) as arrays:
        gain_times, gains, gain_valid = _pivot_casa_table(
            arrays, gain_table, "cparam", field_id=field_id
        )
        interval_times, gain_interval, _ = _pivot_casa_table(
            arrays, gain_table, "interval", field_id=field_id
        )
        if not np.array_equal(interval_times, gain_times):
            raise ValueError("CASA gain interval and parameter coordinates differ")
        gains = gains[:, :, 0, :]
        gain_valid = gain_valid[:, :, 0, :]
        gain_interval = np.broadcast_to(gain_interval[:, :, None], gains.shape).copy()
        _, delay, delay_valid = _pivot_casa_table(arrays, "delay", "fparam")
        # CASA K-table FPARAM uses the opposite phase slope to this module's
        # `exp(-2πi(ν-ν_ref)τ)` Jones convention.
        delay = -delay[0, :, 0, :] * 1e-9
        delay_valid = delay_valid[0, :, 0, :]
        _, bandpass, bandpass_valid = _pivot_casa_table(arrays, "bandpass", "cparam")
        bandpass = bandpass[0]
        bandpass_valid = bandpass_valid[0]
        frequencies = np.asarray(arrays["frequency_hz"], dtype=np.float64)
        antenna_names = np.asarray(arrays["antenna_name"]).astype(str)
        reference_name = str(metadata["reference_antenna"])
        reference_matches = np.flatnonzero(antenna_names == reference_name)
        if reference_matches.size != 1:
            raise ValueError(f"reference antenna {reference_name!r} not found")
        antpos = np.asarray(arrays["antpos_fparam"])[..., 0, :]
        antpos_antenna = np.asarray(arrays["antpos_antenna1"], dtype=np.int32)
        offsets = np.zeros((gains.shape[1], 3), dtype=np.float64)
        offsets[antpos_antenna] = antpos
    return CalibrationSolution(
        gains=gains,
        gain_time_s=gain_times,
        gain_valid=gain_valid,
        gain_interval_s=gain_interval,
        delays_s=delay,
        delay_valid=delay_valid,
        bandpass=bandpass,
        bandpass_frequency_hz=frequencies,
        bandpass_valid=bandpass_valid,
        correlations=tuple(Correlation(value) for value in metadata["correlations"]),
        reference_antenna=int(reference_matches[0]),
        reference_frequency_hz=float(metadata["flux_scale"]["freq"][0]),
        interpolation=interpolation,
        antenna_position_offset_m=offsets,
        provenance={
            "source": source.name,
            "casa_version": metadata["casa_version"],
            "field_id": field_id,
            "gain_table": gain_table,
            "antenna_position_application": "ecef_phase_applied",
            "calwt": False,
        },
    )


def load_casa_calibration_golden(
    fixture: str | Path,
    *,
    label: str,
) -> CasaCalibrationGoldenCase:
    source = Path(fixture)
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if label not in metadata["visibility_cases"]:
        raise ValueError(f"unknown CASA calibration golden case {label!r}")
    prefix = f"{label}_"
    with np.load(source) as arrays:
        field_id = int(metadata["visibility_cases"][label]["field_id"])
        phase_centre = tuple(
            float(value) for value in arrays["field_phase_direction_rad"][field_id]
        )
        block = VisibilityBlock(
            uvw_m=arrays[prefix + "uvw_m"],
            frequency_hz=arrays["frequency_hz"],
            visibility=arrays[prefix + "data"],
            model_visibility=arrays[prefix + "model_data"],
            weight=arrays[prefix + "weight"],
            flag=arrays[prefix + "flag"],
            time_s=arrays[prefix + "time_s"],
            antenna1=arrays[prefix + "antenna1"],
            antenna2=arrays[prefix + "antenna2"],
            field_id=arrays[prefix + "field_id"],
            scan_id=arrays[prefix + "scan_id"],
            correlations=tuple(Correlation(value) for value in metadata["correlations"]),
            receptor_basis=ReceptorBasis.CIRCULAR,
            phase_centre_rad=(phase_centre[0], phase_centre[1]),
            provenance={
                "source": source.name,
                "case": label,
                "effects": metadata["effects"],
            },
        )
        return CasaCalibrationGoldenCase(
            label=label,
            field_id=field_id,
            block=block,
            corrected_visibility=np.asarray(arrays[prefix + "corrected_data"], dtype=np.complex128),
            post_apply_flag=np.asarray(arrays[prefix + "post_apply_flag"], dtype=bool),
            metadata=metadata,
        )
