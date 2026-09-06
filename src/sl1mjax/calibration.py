"""Jones calibration solutions, application, and CASA bridges.

Schema v1 stored one complex gain per correlation product and applied a
scalar ``J_p J_q*`` per hand.  Schema v2 stores one Jones term per *feed
receptor* and applies ``C_obs = J_p C_sky J_q^H``.
Schema v3 adds Kcross, Df, Xf, antenna positions, parallactic-angle
application, and an explicit leakage operator.  Older readers that only
accept schema 1–2 reject v3 rather than silently applying diagonal G/K/B.
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
    parallactic_angle_rad,
)
from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    apply_jones_to_coherency,
    circular_parallactic_jones,
    correlation_receptor_pair,
    diagonal_jones_matrices,
    invert_jones,
    leakage_jones_matrices,
    multiply_jones,
    pack_coherency,
    receptor_phase_jones,
    receptors_for_correlations,
    unpack_coherency,
)

CALIBRATION_SCHEMA_VERSION = 3
SUPPORTED_CALIBRATION_SCHEMA_VERSIONS = frozenset({1, 2, 3})
SPEED_OF_LIGHT_M_S = 299_792_458.0
LeakageApplication = Literal["exact", "casa_parallel_preserving", "casa_first_order"]
LEAKAGE_APPLICATIONS = frozenset({"exact", "casa_parallel_preserving", "casa_first_order"})
ParallacticModel = Literal["gmst_geodetic", "casacore_hadec_geocentric"]
PARALLACTIC_MODELS = frozenset({"gmst_geodetic", "casacore_hadec_geocentric"})
INTERPOLATION_METHODS = frozenset(
    {"nearest", "linear", "linear_complex", "linear_amp_phase", "casa_linear"}
)
CASA_CPARAM_METHODS = frozenset({"casa_linear"})
PARALLEL_HAND_CORRELATIONS = frozenset(
    {
        Correlation.RR,
        Correlation.LL,
        Correlation.XX,
        Correlation.YY,
        Correlation.I,
    }
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CalibrationSolution:
    """Portable Jones terms on explicit time/frequency coordinates.

    ``gains``, ``delays_s``, and ``bandpass`` are diagonal per *feed
    receptor* (R/L or X/Y).  Application builds 2×2 Jones matrices and
    forms ``J_p C J_q^H``.  Diagonal G/K/B promote to diagonal matrices.
    Optional Kcross, Df, Xf, and circular parallactic-angle terms are
    multiplied on the right (closest to the sky last):

    ``J = J_{GKB} J_{Kcross} J_D J_X J_P``.

    ``leakage`` is ``(antenna, frequency, 2)`` when D is time-independent, or
    ``(time, antenna, frequency, 2)`` with ``leakage_time_s`` when CASA wrote
    more than one Df interval.     Time-dependent G and D CPARAM use the CASA 6.7.6 path only when
    ``interpolation='casa_linear'``: float32 amplitude and sequentially
    unwrapped phase. ``linear`` and ``linear_amp_phase`` are float64
    amplitude/unwrapped-phase. ``linear_complex`` is a diagnostic only.

    ``leakage_application`` selects the D operator:

    - ``exact``: invert the full 2×2 Jones product (physics / synthetic
      round-trips).
    - ``casa_parallel_preserving``: 3C391-style CASA hybrid — parallel
      hands from the diagonal chain (no D), cross-hands from the full 2x2.
    - ``casa_first_order``: THOL0001 ``applycal`` Df. All four correlations
      take the diagonal prefix. RL/LR then receive the first-order leaks
      ``-D_L,q* RR - D_R,p LL`` and ``-D_L,p RR - D_R,q* LL``. No D D
      products, and D does not rescale existing RL/LR.

    Full 2×2 Jones can also be applied through ``apply_jones_to_coherency``.
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
    antenna_position_m: np.ndarray | None = None
    cross_hand_delay_s: np.ndarray | None = None
    cross_hand_delay_valid: np.ndarray | None = None
    leakage: np.ndarray | None = None
    leakage_frequency_hz: np.ndarray | None = None
    leakage_valid: np.ndarray | None = None
    leakage_time_s: np.ndarray | None = None
    rl_phase: np.ndarray | None = None
    rl_phase_frequency_hz: np.ndarray | None = None
    rl_phase_valid: np.ndarray | None = None
    apply_parallactic_angle: bool = False
    leakage_application: LeakageApplication = "exact"
    parallactic_model: ParallacticModel = "gmst_geodetic"
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
        if self.antenna_position_m is not None:
            object.__setattr__(
                self,
                "antenna_position_m",
                np.asarray(self.antenna_position_m, dtype=np.float64),
            )
        if self.cross_hand_delay_s is not None:
            object.__setattr__(
                self,
                "cross_hand_delay_s",
                np.asarray(self.cross_hand_delay_s, dtype=np.float64),
            )
        if self.cross_hand_delay_valid is not None:
            object.__setattr__(
                self,
                "cross_hand_delay_valid",
                np.asarray(self.cross_hand_delay_valid, dtype=bool),
            )
        if self.leakage is not None:
            object.__setattr__(self, "leakage", np.asarray(self.leakage, dtype=np.complex128))
        if self.leakage_frequency_hz is not None:
            object.__setattr__(
                self,
                "leakage_frequency_hz",
                np.asarray(self.leakage_frequency_hz, dtype=np.float64),
            )
        if self.leakage_valid is not None:
            object.__setattr__(self, "leakage_valid", np.asarray(self.leakage_valid, dtype=bool))
        if self.leakage_time_s is not None:
            object.__setattr__(
                self, "leakage_time_s", np.asarray(self.leakage_time_s, dtype=np.float64)
            )
        if self.rl_phase is not None:
            object.__setattr__(self, "rl_phase", np.asarray(self.rl_phase, dtype=np.complex128))
        if self.rl_phase_frequency_hz is not None:
            object.__setattr__(
                self,
                "rl_phase_frequency_hz",
                np.asarray(self.rl_phase_frequency_hz, dtype=np.float64),
            )
        if self.rl_phase_valid is not None:
            object.__setattr__(self, "rl_phase_valid", np.asarray(self.rl_phase_valid, dtype=bool))
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
        if self.interpolation not in INTERPOLATION_METHODS:
            raise ValueError(
                "interpolation must be nearest, linear, casa_linear, "
                "linear_complex, or linear_amp_phase"
            )
        if self.parallactic_model not in PARALLACTIC_MODELS:
            raise ValueError("parallactic_model must be gmst_geodetic or casacore_hadec_geocentric")
        if self.leakage_application not in LEAKAGE_APPLICATIONS:
            raise ValueError(
                "leakage_application must be exact, casa_parallel_preserving, or casa_first_order"
            )
        if self.antenna_position_offset_m is not None and self.antenna_position_offset_m.shape != (
            self.antenna_count,
            3,
        ):
            raise ValueError("antenna_position_offset_m must have shape (antenna, 3)")
        if self.antenna_position_m is not None and self.antenna_position_m.shape != (
            self.antenna_count,
            3,
        ):
            raise ValueError("antenna_position_m must have shape (antenna, 3)")
        if self.apply_parallactic_angle and self.antenna_position_m is None:
            raise ValueError("apply_parallactic_angle requires antenna_position_m")
        n_ant = self.antenna_count
        n_rec = receptors
        if (self.cross_hand_delay_s is None) != (self.cross_hand_delay_valid is None):
            raise ValueError("cross_hand_delay_s and cross_hand_delay_valid must be set together")
        if self.cross_hand_delay_s is not None:
            if self.cross_hand_delay_s.shape != (n_ant, n_rec):
                raise ValueError("cross_hand_delay_s must have shape (antenna, receptor)")
            if self.cross_hand_delay_valid.shape != self.cross_hand_delay_s.shape:
                raise ValueError("cross_hand_delay_valid must match cross_hand_delay_s")
        if (self.leakage is None) != (self.leakage_frequency_hz is None) or (
            self.leakage is None
        ) != (self.leakage_valid is None):
            raise ValueError(
                "leakage, leakage_frequency_hz, and leakage_valid must be set together"
            )
        if self.leakage is not None:
            n_freq = self.leakage_frequency_hz.size
            if self.leakage.ndim == 3:
                if self.leakage_time_s is not None:
                    raise ValueError("leakage_time_s requires a leading time axis on leakage")
                expected_leakage = (n_ant, n_freq, 2)
            elif self.leakage.ndim == 4:
                if self.leakage_time_s is None:
                    raise ValueError("time-leading leakage requires leakage_time_s")
                expected_leakage = (self.leakage_time_s.size, n_ant, n_freq, 2)
            else:
                raise ValueError(
                    "leakage must have shape (antenna, frequency, 2) or "
                    "(time, antenna, frequency, 2)"
                )
            if self.leakage.shape != expected_leakage:
                raise ValueError(f"leakage must have shape {expected_leakage}")
            if self.leakage_valid.shape != self.leakage.shape:
                raise ValueError("leakage_valid must match leakage")
        if (self.rl_phase is None) != (self.rl_phase_frequency_hz is None) or (
            self.rl_phase is None
        ) != (self.rl_phase_valid is None):
            raise ValueError(
                "rl_phase, rl_phase_frequency_hz, and rl_phase_valid must be set together"
            )
        if self.rl_phase is not None:
            expected_phase = (n_ant, self.rl_phase_frequency_hz.size)
            if self.rl_phase.shape != expected_phase:
                raise ValueError(f"rl_phase must have shape {expected_phase}")
            if self.rl_phase_valid.shape != self.rl_phase.shape:
                raise ValueError("rl_phase_valid must match rl_phase")
        if (
            self.cross_hand_delay_s is not None
            or self.leakage is not None
            or self.rl_phase is not None
            or self.apply_parallactic_angle
        ) and self.receptors != (Receptor.R, Receptor.L):
            raise ValueError(
                "Kcross, Df, Xf, and circular parallactic-angle Jones require receptors (R, L)"
            )

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
            self.antenna_position_m,
            self.cross_hand_delay_s,
            self.cross_hand_delay_valid,
            self.leakage,
            self.leakage_frequency_hz,
            self.leakage_valid,
            self.leakage_time_s,
            self.rl_phase,
            self.rl_phase_frequency_hz,
            self.rl_phase_valid,
        )
        auxiliary = (
            self.correlations,
            self.receptors,
            self.reference_antenna,
            self.reference_frequency_hz,
            self.interpolation,
            self.apply_parallactic_angle,
            self.leakage_application,
            self.parallactic_model,
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
            apply_parallactic_angle,
            leakage_application,
            parallactic_model,
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
            antenna_position_m=children[10],
            cross_hand_delay_s=children[11],
            cross_hand_delay_valid=children[12],
            leakage=children[13],
            leakage_frequency_hz=children[14],
            leakage_valid=children[15],
            leakage_time_s=children[16],
            rl_phase=children[17],
            rl_phase_frequency_hz=children[18],
            rl_phase_valid=children[19],
            apply_parallactic_angle=apply_parallactic_angle,
            leakage_application=leakage_application,
            parallactic_model=parallactic_model,
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


def casa_cparam_time_ref(cal_times: np.ndarray) -> float:
    """CASA ``CTTimeInterp1`` reference: ``floor(first_time - 1)`` in double."""

    times = np.asarray(cal_times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("interpolation requires at least one calibration time")
    return float(np.floor(float(times[0]) - 1.0))


def casa_cparam_relative_times(times: np.ndarray, time_ref: float) -> np.ndarray:
    """Visibility or calibration times after CASA's double-to-Float32 conversion."""

    return np.asarray(np.asarray(times, dtype=np.float64) - float(time_ref), dtype=np.float32)


def unwrap_casa_phase_float32(phase: np.ndarray) -> np.ndarray:
    """Unwrap along time with CASA's sequential ±2π rule, in float32."""

    out = np.array(phase, dtype=np.float32, copy=True)
    two_pi = np.float32(2.0 * np.pi)
    limit = np.float32(np.pi)
    for index in range(1, out.size):
        delta = out[index] - out[index - 1]
        while delta > limit:
            out[index] -= two_pi
            delta = out[index] - out[index - 1]
        while delta < -limit:
            out[index] += two_pi
            delta = out[index] - out[index - 1]
    return out


def _casa_cparam_brackets(query: np.float32, cal_times: np.ndarray) -> tuple[int, int, bool]:
    times = np.asarray(cal_times, dtype=np.float32)
    if times.size == 1 or query <= times[0]:
        return 0, 0, True
    last = times.size - 1
    if query >= times[last]:
        return last, last, True
    left = int(np.searchsorted(times, query, side="right") - 1)
    if times[left] == query:
        return left, left, True
    return left, left + 1, False


def interpolate_casa_cparam(
    query_times: np.ndarray,
    cal_times: np.ndarray,
    values: np.ndarray,
    flags: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce CASA 6.7.6 ``CTTimeInterp1`` linear interpolation of CPARAM.

    Flagged rows still enter amplitude/phase tracking. The result is flagged
    when either selected bracket is flagged. ``INTERVAL`` is unused.
    """

    query = np.asarray(query_times, dtype=np.float64).reshape(-1)
    times = np.asarray(cal_times, dtype=np.float64).reshape(-1)
    series = np.asarray(values, dtype=np.complex128).reshape(-1)
    if times.size == 0:
        raise ValueError("interpolation requires at least one calibration time")
    if series.size != times.size:
        raise ValueError("CASA CPARAM interpolation requires one value per time")
    flagged = (
        np.zeros(times.size, dtype=bool)
        if flags is None
        else np.asarray(flags, dtype=bool).reshape(-1)
    )
    if flagged.size != times.size:
        raise ValueError("CASA CPARAM flags must match the calibration times")
    order = np.argsort(times, kind="stable")
    times = times[order]
    series = series[order]
    flagged = flagged[order]
    time_ref = casa_cparam_time_ref(times)
    time_f = casa_cparam_relative_times(times, time_ref)
    query_f = casa_cparam_relative_times(query, time_ref)
    amplitude = np.abs(series).astype(np.float32)
    phase = unwrap_casa_phase_float32(np.angle(series).astype(np.float32))
    output = np.empty(query.size, dtype=np.complex128)
    valid = np.empty(query.size, dtype=bool)
    for row, ftime in enumerate(query_f):
        left, right, nearest = _casa_cparam_brackets(np.float32(ftime), time_f)
        valid[row] = not bool(flagged[left] or flagged[right])
        if nearest or time_f[right] == time_f[left]:
            sampled_amp = amplitude[left]
            sampled_phase = phase[left]
        else:
            span = time_f[right] - time_f[left]
            frac = (np.float32(ftime) - time_f[left]) / span
            sampled_amp = amplitude[left] + frac * (amplitude[right] - amplitude[left])
            sampled_phase = phase[left] + frac * (phase[right] - phase[left])
        real = sampled_amp * np.cos(sampled_phase)
        imag = sampled_amp * np.sin(sampled_phase)
        output[row] = np.complex64(real + 1j * imag)
    return output, valid


def interpolate_complex_series(
    query_times: np.ndarray,
    cal_times: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
    flags: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate one complex series.

    ``casa_linear`` is the CASA 6.7.6 CPARAM path.
    ``linear`` and ``linear_amp_phase`` are float64 amplitude/unwrapped phase.
    ``linear_complex`` is a diagnostic Cartesian interpolator.
    """

    query = np.asarray(query_times, dtype=np.float64)
    times = np.asarray(cal_times, dtype=np.float64)
    series = np.asarray(values, dtype=np.complex128)
    if times.size == 0:
        raise ValueError("interpolation requires at least one calibration time")
    if method in CASA_CPARAM_METHODS:
        sampled, _valid = interpolate_casa_cparam(query, times, series, flags)
        return sampled
    if method == "nearest" or times.size == 1:
        indices = np.argmin(np.abs(query[:, None] - times[None, :]), axis=1)
        return series[indices]
    if method in {"linear", "linear_amp_phase"}:
        amplitude = np.interp(query, times, np.abs(series))
        phase = np.interp(query, times, np.unwrap(np.angle(series)))
        return amplitude * np.exp(1j * phase)
    if method == "linear_complex":
        return np.interp(query, times, series.real) + 1j * np.interp(query, times, series.imag)
    raise ValueError(f"unknown interpolation method {method!r}")


def _present_cparam(values: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(values).real) & np.isfinite(np.asarray(values).imag)


def _interpolate_gains(
    solution: CalibrationSolution, time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty(
        (time_s.size, solution.antenna_count, solution.receptor_count),
        dtype=np.complex128,
    )
    valid = np.zeros(output.shape, dtype=bool)
    method = solution.interpolation
    for antenna in range(solution.antenna_count):
        for receptor in range(solution.receptor_count):
            present = _present_cparam(solution.gains[:, antenna, receptor])
            if method not in CASA_CPARAM_METHODS:
                present = present & solution.gain_valid[:, antenna, receptor]
            if not np.any(present):
                output[:, antenna, receptor] = np.nan + 1j * np.nan
                continue
            cal_times = solution.gain_time_s[present]
            cal_values = solution.gains[present, antenna, receptor]
            if method in CASA_CPARAM_METHODS:
                sampled, ok = interpolate_casa_cparam(
                    time_s,
                    cal_times,
                    cal_values,
                    flags=~solution.gain_valid[present, antenna, receptor],
                )
                output[:, antenna, receptor] = sampled
                valid[:, antenna, receptor] = ok
                continue
            output[:, antenna, receptor] = interpolate_complex_series(
                time_s,
                cal_times,
                cal_values,
                method=method,
            )
            valid[:, antenna, receptor] = True
    return output, valid


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
    gains, casa_gain_valid = _interpolate_gains(solution, times)
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
        if (
            solution.parallactic_model == "casacore_hadec_geocentric"
            and solution.antenna_position_m is not None
        ):
            from sl1mjax.holography_calibration_measures import itrf_direction_casacore

            direction_ecef = itrf_direction_casacore(
                times,
                phase_centre_rad,
                solution.antenna_position_m,
                time_reference=str(solution.provenance.get("time_reference", "UTC")),
                direction_frame=str(solution.provenance.get("direction_frame", "J2000")),
            )
        else:
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
    if solution.interpolation in CASA_CPARAM_METHODS:
        nearest_gain = casa_gain_valid
    else:
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
                if solution.interpolation != "nearest" and valid_times.size > 1:
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


def _leakage_product_validity(
    leakage_valid: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    correlations: tuple[Correlation, ...],
    application: str,
) -> np.ndarray:
    """Per-correlation D validity.

    First-order D keeps RR/LL when a leakage entry is missing and invalidates
    only the cross-hand that depends on that entry. Exact Jones still needs
    every D term on both antennas.
    """

    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    rows = np.arange(leakage_valid.shape[0])
    d_p = leakage_valid[rows, :, first, :]
    d_q = leakage_valid[rows, :, second, :]
    antenna_ok = np.all(d_p, axis=-1) & np.all(d_q, axis=-1)
    valid = np.empty((*antenna_ok.shape, len(correlations)), dtype=bool)
    for slot, correlation in enumerate(correlations):
        if correlation in PARALLEL_HAND_CORRELATIONS and application != "exact":
            valid[..., slot] = True
        elif application == "casa_first_order" and correlation == Correlation.RL:
            valid[..., slot] = d_p[..., 0] & d_q[..., 1]
        elif application == "casa_first_order" and correlation == Correlation.LR:
            valid[..., slot] = d_p[..., 1] & d_q[..., 0]
        else:
            valid[..., slot] = antenna_ok
    return valid


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


def _full_feed_products(
    receptors: tuple[Receptor, ...],
) -> tuple[Correlation, ...] | None:
    if receptors == (Receptor.R, Receptor.L):
        return (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    if receptors == (Receptor.X, Receptor.Y):
        return (Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY)
    return None


def _missing_feed_products(
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> tuple[Correlation, ...]:
    needed = _full_feed_products(receptors)
    if needed is None:
        return ()
    present = set(correlations)
    return tuple(product for product in needed if product not in present)


def _complete_coherency_mask(
    visibility: np.ndarray,
    flag: np.ndarray,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> np.ndarray:
    """True where every two-feed product is present, finite, and unflagged."""

    needed = _full_feed_products(receptors)
    if needed is None:
        return np.all(np.isfinite(visibility) & ~flag, axis=-1)
    index = {correlation: slot for slot, correlation in enumerate(correlations)}
    complete = np.ones(visibility.shape[:2], dtype=bool)
    for product in needed:
        slot = index.get(product)
        if slot is None:
            return np.zeros(visibility.shape[:2], dtype=bool)
        complete &= np.isfinite(visibility[..., slot]) & ~flag[..., slot]
    return complete


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


def _nearest_frequency_indices(
    cal_frequency_hz: np.ndarray, frequency_hz: np.ndarray
) -> np.ndarray:
    return np.argmin(np.abs(frequency_hz[:, None] - cal_frequency_hz[None, :]), axis=1)


def _frequency_in_domain(
    cal_frequency_hz: np.ndarray,
    frequency_hz: np.ndarray,
    *,
    extrapolate: bool,
) -> np.ndarray:
    """True where visibilities may use a sampled D/X channel.

    Df/Xf never take a silent edge solution.  ``extrapolate`` here is only
    for tests that explicitly want nearest-neighbour; apply always passes
    False so unsolved channels are flagged.
    """

    if extrapolate or cal_frequency_hz.size == 0:
        return np.ones(frequency_hz.size, dtype=bool)
    return (frequency_hz >= cal_frequency_hz.min()) & (frequency_hz <= cal_frequency_hz.max())


def _sample_leakage(
    solution: CalibrationSolution,
    times: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage and validity with shape ``(row, channel, antenna, 2)``."""

    assert solution.leakage is not None
    assert solution.leakage_frequency_hz is not None
    assert solution.leakage_valid is not None
    indices = _nearest_frequency_indices(solution.leakage_frequency_hz, frequency_hz)
    in_domain = _frequency_in_domain(solution.leakage_frequency_hz, frequency_hz, extrapolate=False)
    n_row = times.size
    n_chan = frequency_hz.size
    n_ant = solution.antenna_count
    if solution.leakage.ndim == 3:
        sampled_valid = solution.leakage_valid[:, indices, :] & in_domain[None, :, None]
        sampled = np.where(sampled_valid, solution.leakage[:, indices, :], 0.0)
        values = np.broadcast_to(
            np.transpose(sampled, (1, 0, 2))[None, ...],
            (n_row, n_chan, n_ant, 2),
        ).copy()
        valid = np.broadcast_to(
            np.transpose(sampled_valid, (1, 0, 2))[None, ...],
            values.shape,
        ).copy()
        return values, valid
    assert solution.leakage_time_s is not None
    freq_values = solution.leakage[:, :, indices, :]
    freq_flag = ~solution.leakage_valid[:, :, indices, :]
    values = np.zeros((n_row, n_chan, n_ant, 2), dtype=np.complex128)
    valid = np.zeros(values.shape, dtype=bool)
    cal_times = solution.leakage_time_s
    method = solution.interpolation
    for antenna in range(n_ant):
        for receptor in range(2):
            for channel in range(n_chan):
                if not bool(in_domain[channel]):
                    continue
                series = freq_values[:, antenna, channel, receptor]
                present = _present_cparam(series)
                if method not in CASA_CPARAM_METHODS:
                    present = present & ~freq_flag[:, antenna, channel, receptor]
                if not np.any(present):
                    continue
                sample_times = cal_times[present]
                sample_values = series[present]
                if method in CASA_CPARAM_METHODS:
                    sampled, ok = interpolate_casa_cparam(
                        times,
                        sample_times,
                        sample_values,
                        flags=freq_flag[present, antenna, channel, receptor],
                    )
                    values[:, channel, antenna, receptor] = sampled
                    valid[:, channel, antenna, receptor] = ok
                    continue
                values[:, channel, antenna, receptor] = interpolate_complex_series(
                    times, sample_times, sample_values, method=method
                )
                if method == "nearest" or sample_times.size == 1:
                    valid[:, channel, antenna, receptor] = True
                    continue
                valid[:, channel, antenna, receptor] = (times >= sample_times.min()) & (
                    times <= sample_times.max()
                )
    return values, valid


def _parallactic_angles_for_solution(
    solution: CalibrationSolution,
    times: np.ndarray,
    phase_centre_rad: tuple[float, float],
) -> np.ndarray:
    if solution.antenna_position_m is None:
        raise ValueError("parallactic-angle Jones needs antenna_position_m")
    if solution.parallactic_model == "casacore_hadec_geocentric":
        from sl1mjax.holography_calibration_measures import parallactic_angle_casacore_rad

        return parallactic_angle_casacore_rad(
            times,
            phase_centre_rad,
            solution.antenna_position_m,
            time_reference=str(solution.provenance.get("time_reference", "UTC")),
            direction_frame=str(solution.provenance.get("direction_frame", "J2000")),
        )
    return parallactic_angle_rad(times, phase_centre_rad, solution.antenna_position_m)


def _compose_polarization_jones(
    solution: CalibrationSolution,
    *,
    times: np.ndarray,
    frequency_hz: np.ndarray,
    phase_centre_rad: tuple[float, float] | None,
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Right-multiply Kcross, Df, Xf, and P onto diagonal G/K/B Jones.

    Returns ``(with_d, without_d, pol_valid, leakage_valid)``.  Invalid
    Kcross/Xf still fill an identity term so the product stays finite, but
    ``pol_valid`` is False.  Invalid D is returned separately so first-order
    apply can keep usable parallel hands.
    """

    leakage_valid = np.ones((*matrices.shape[:3], 2), dtype=bool)
    if solution.receptor_count != 2:
        valid = np.ones(matrices.shape[:4], dtype=bool)
        return matrices, matrices, valid, leakage_valid
    composed = matrices
    pol_valid = np.ones(matrices.shape[:4], dtype=bool)
    if solution.cross_hand_delay_s is not None:
        assert solution.cross_hand_delay_valid is not None
        delay = np.where(
            solution.cross_hand_delay_valid,
            solution.cross_hand_delay_s,
            0.0,
        )
        offset = frequency_hz - solution.reference_frequency_hz
        factors = np.exp(-2j * np.pi * delay[:, None, :] * offset[None, :, None])
        kcross = diagonal_jones_matrices(np.transpose(factors, (1, 0, 2)))
        composed = multiply_jones(composed, kcross)
        pol_valid = pol_valid & solution.cross_hand_delay_valid[None, None, :, :]
    without_d = composed
    if solution.leakage is not None:
        sampled, sampled_valid = _sample_leakage(solution, times, frequency_hz)
        composed = multiply_jones(composed, leakage_jones_matrices(sampled))
        leakage_valid = sampled_valid
    if solution.rl_phase is not None:
        assert solution.rl_phase_frequency_hz is not None
        assert solution.rl_phase_valid is not None
        indices = _nearest_frequency_indices(solution.rl_phase_frequency_hz, frequency_hz)
        in_domain = _frequency_in_domain(
            solution.rl_phase_frequency_hz, frequency_hz, extrapolate=False
        )
        sampled_valid = solution.rl_phase_valid[:, indices] & in_domain[None, :]
        sampled = np.where(sampled_valid, solution.rl_phase[:, indices], 1.0)
        phase = receptor_phase_jones(np.transpose(sampled, (1, 0)))
        composed = multiply_jones(composed, phase)
        without_d = multiply_jones(without_d, phase)
        pol_valid = pol_valid & np.transpose(sampled_valid, (1, 0))[None, :, :, None]
    if solution.apply_parallactic_angle:
        if solution.antenna_position_m is None or phase_centre_rad is None:
            raise ValueError(
                "parallactic-angle Jones needs antenna_position_m and phase_centre_rad"
            )
        chi = _parallactic_angles_for_solution(solution, times, phase_centre_rad)
        parallactic = circular_parallactic_jones(chi)[:, None]
        composed = multiply_jones(composed, parallactic)
        without_d = multiply_jones(without_d, parallactic)
    return composed, without_d, pol_valid, leakage_valid


def _gather_jones(matrices: np.ndarray, antenna: np.ndarray) -> np.ndarray:
    row = np.arange(matrices.shape[0])[:, None, None, None]
    channel = np.arange(matrices.shape[1])[None, :, None, None]
    receptor_i = np.arange(matrices.shape[3])[None, None, :, None]
    receptor_j = np.arange(matrices.shape[4])[None, None, None, :]
    return matrices[row, channel, antenna[:, None, None, None], receptor_i, receptor_j]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full ``J_p``, ``J_q``, diagonal-chain ``J_p``, ``J_q``, validity, and D validity."""

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
    with_d, without_d, pol_valid, leakage_valid = _compose_polarization_jones(
        solution,
        times=np.asarray(time_s, dtype=np.float64),
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        phase_centre_rad=phase_centre_rad,
        matrices=matrices,
    )
    antenna_valid = antenna_valid & pol_valid
    return (
        _gather_jones(with_d, first),
        _gather_jones(with_d, second),
        _gather_jones(without_d, first),
        _gather_jones(without_d, second),
        antenna_valid,
        leakage_valid,
    )


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
    if solution.leakage_application in {"casa_parallel_preserving", "casa_first_order"}:
        raise ValueError(
            "corrupt_model requires an invertible Jones chain; "
            f"{solution.leakage_application} is an apply-only CASA oracle"
        )
    model = np.asarray(model_visibility)
    if model.shape[-1] != len(solution.correlations):
        raise ValueError("model visibility last axis must match solution correlations")
    if solution.leakage is not None and _missing_feed_products(
        solution.correlations, solution.receptors
    ):
        raise ValueError("exact leakage corruption requires a complete two-feed coherency")
    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    jones_p, jones_q, _, _, antenna_valid, leakage_valid = _jones_matrices_for_block(
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
    if solution.leakage is not None:
        valid = valid & _leakage_product_validity(
            leakage_valid,
            first,
            second,
            solution.correlations,
            solution.leakage_application,
        )
    packed = pack_coherency(model, solution.correlations, solution.receptors)
    corrupted = unpack_coherency(
        apply_jones_to_coherency(packed, jones_p, jones_q),
        solution.correlations,
        solution.receptors,
    )
    return jnp.asarray(np.where(valid, corrupted, 0.0))


def _casa_first_order_d_apply(
    visibility: np.ndarray,
    solution: CalibrationSolution,
    times: np.ndarray,
    frequency_hz: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    correlations: tuple[Correlation, ...],
) -> np.ndarray:
    """Add CASA first-order Df leaks after the diagonal prefix apply.

    Parallel hands and the existing RL/LR stay at the prefix values.
    Only RR and LL leak into the cross-hands.
    """

    index = {correlation: slot for slot, correlation in enumerate(correlations)}
    required = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    if any(correlation not in index for correlation in required):
        return visibility
    sampled, sampled_valid = _sample_leakage(solution, times, frequency_hz)
    first = np.asarray(antenna1, dtype=np.int32)
    second = np.asarray(antenna2, dtype=np.int32)
    rows = np.arange(visibility.shape[0])
    d_p = np.where(
        sampled_valid[rows, :, first, :],
        sampled[rows, :, first, :],
        0.0,
    )
    d_q = np.where(
        sampled_valid[rows, :, second, :],
        sampled[rows, :, second, :],
        0.0,
    )
    rr = visibility[..., index[Correlation.RR]]
    ll = visibility[..., index[Correlation.LL]]
    output = visibility.copy()
    output[..., index[Correlation.RL]] = (
        visibility[..., index[Correlation.RL]] - np.conjugate(d_q[..., 1]) * rr - d_p[..., 0] * ll
    )
    output[..., index[Correlation.LR]] = (
        visibility[..., index[Correlation.LR]] - d_p[..., 1] * rr - np.conjugate(d_q[..., 0]) * ll
    )
    return output


def _unpack_corrected(
    visibility: np.ndarray,
    jones_p: np.ndarray,
    jones_q: np.ndarray,
    correlations: tuple[Correlation, ...],
    receptors: tuple[Receptor, ...],
) -> tuple[np.ndarray, np.ndarray]:
    inverse_p = invert_jones(jones_p)
    inverse_q = invert_jones(jones_q)
    finite = np.all(np.isfinite(inverse_p), axis=(-2, -1)) & np.all(
        np.isfinite(inverse_q), axis=(-2, -1)
    )
    packed = pack_coherency(visibility, correlations, receptors)
    corrected = unpack_coherency(
        apply_jones_to_coherency(packed, inverse_p, inverse_q),
        correlations,
        receptors,
    )
    return corrected, finite


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
    if propagate_weights and solution.leakage is not None:
        raise ValueError(
            "propagate_weights is a diagonal approximation and is refused "
            "when leakage Jones is present"
        )
    use_casa = (
        solution.leakage_application in {"casa_parallel_preserving", "casa_first_order"}
        and solution.leakage is not None
    )
    use_first_order = (
        solution.leakage_application == "casa_first_order" and solution.leakage is not None
    )
    if (
        solution.leakage is not None
        and not use_casa
        and _missing_feed_products(block.correlations, solution.receptors)
    ):
        raise ValueError("exact leakage application requires a complete two-feed coherency")
    jones_p, jones_q, jones_p_diag, jones_q_diag, antenna_valid, leakage_valid = (
        _jones_matrices_for_block(
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
    )
    cal_valid = _product_validity(
        antenna_valid,
        np.asarray(block.antenna1, dtype=np.int32),
        np.asarray(block.antenna2, dtype=np.int32),
        block.correlations,
        solution.receptors,
    )
    corrected, finite_full = _unpack_corrected(
        block.visibility,
        jones_p,
        jones_q,
        block.correlations,
        solution.receptors,
    )
    complete = _complete_coherency_mask(
        block.visibility,
        block.flag,
        block.correlations,
        solution.receptors,
    )
    if use_casa:
        corrected_diag, finite_diag = _unpack_corrected(
            block.visibility,
            jones_p_diag,
            jones_q_diag,
            block.correlations,
            solution.receptors,
        )
        parallel = np.array(
            [correlation in PARALLEL_HAND_CORRELATIONS for correlation in block.correlations]
        )
        if use_first_order:
            times = np.asarray(block.time_s, dtype=np.float64)
            frequencies = np.asarray(block.frequency_hz, dtype=np.float64)
            first = np.asarray(block.antenna1, dtype=np.int32)
            second = np.asarray(block.antenna2, dtype=np.int32)
            has_suffix = solution.rl_phase is not None or solution.apply_parallactic_angle
            if has_suffix:
                prefix_solution = replace(
                    solution,
                    leakage=None,
                    leakage_frequency_hz=None,
                    leakage_valid=None,
                    leakage_time_s=None,
                    rl_phase=None,
                    rl_phase_frequency_hz=None,
                    rl_phase_valid=None,
                    apply_parallactic_angle=False,
                    leakage_application="exact",
                )
                jones_p_pre, jones_q_pre, _, _, _, _ = _jones_matrices_for_block(
                    prefix_solution,
                    block.time_s,
                    block.frequency_hz,
                    block.antenna1,
                    block.antenna2,
                    extrapolate=extrapolate,
                    phase_centre_rad=block.phase_centre_rad,
                    priors=priors,
                    spectral_window_id=block.spectral_window_id,
                )
                corrected_pre, finite_pre = _unpack_corrected(
                    block.visibility,
                    jones_p_pre,
                    jones_q_pre,
                    block.correlations,
                    solution.receptors,
                )
                leaked = _casa_first_order_d_apply(
                    corrected_pre, solution, times, frequencies, first, second, block.correlations
                )
                suffix_offsets = (
                    None
                    if solution.antenna_position_offset_m is None
                    else np.zeros_like(solution.antenna_position_offset_m)
                )
                suffix_solution = replace(
                    solution,
                    gains=np.ones_like(solution.gains),
                    delays_s=np.zeros_like(solution.delays_s),
                    bandpass=np.ones_like(solution.bandpass),
                    antenna_position_offset_m=suffix_offsets,
                    cross_hand_delay_s=None,
                    cross_hand_delay_valid=None,
                    leakage=None,
                    leakage_frequency_hz=None,
                    leakage_valid=None,
                    leakage_time_s=None,
                    leakage_application="exact",
                )
                jones_p_suf, jones_q_suf, _, _, _, _ = _jones_matrices_for_block(
                    suffix_solution,
                    block.time_s,
                    block.frequency_hz,
                    block.antenna1,
                    block.antenna2,
                    extrapolate=extrapolate,
                    phase_centre_rad=block.phase_centre_rad,
                    priors=None,
                    spectral_window_id=block.spectral_window_id,
                )
                corrected, finite_suf = _unpack_corrected(
                    leaked,
                    jones_p_suf,
                    jones_q_suf,
                    block.correlations,
                    solution.receptors,
                )
                cal_valid = cal_valid & finite_pre[..., None] & finite_suf[..., None]
            else:
                corrected = _casa_first_order_d_apply(
                    corrected_diag, solution, times, frequencies, first, second, block.correlations
                )
                cal_valid = cal_valid & finite_diag[..., None]
        else:
            corrected = np.where(parallel, corrected_diag, corrected)
            cal_valid = cal_valid & np.where(
                parallel, finite_diag[..., None], finite_full[..., None]
            )
        coherency_ok = np.where(parallel, True, complete[..., None])
    else:
        cal_valid = cal_valid & finite_full[..., None]
        coherency_ok = (
            complete[..., None] if solution.leakage is not None else np.ones_like(cal_valid)
        )
    leakage_ok = (
        None
        if solution.leakage is None
        else _leakage_product_validity(
            leakage_valid,
            np.asarray(block.antenna1, dtype=np.int32),
            np.asarray(block.antenna2, dtype=np.int32),
            block.correlations,
            solution.leakage_application,
        )
    )
    if solution.leakage is not None and solution.leakage_application == "exact":
        cal_valid = cal_valid & leakage_ok
    if not extrapolate and np.any(block.active & ~cal_valid):
        raise ValueError("active visibility lies outside the calibration solution validity domain")
    if leakage_ok is not None:
        cal_valid = cal_valid & leakage_ok
    valid_array = cal_valid & coherency_ok
    corrected = np.where(valid_array, corrected, 0.0)
    flag = block.flag | ~valid_array
    if propagate_weights:
        scale = _product_scale(jones_p_diag, jones_q_diag, block.correlations, solution.receptors)
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
            "apply_parallactic_angle": solution.apply_parallactic_angle,
            "leakage_application": solution.leakage_application,
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
    payload: dict[str, np.ndarray] = {
        "gains": solution.gains,
        "gain_time_s": solution.gain_time_s,
        "gain_valid": solution.gain_valid,
        "gain_interval_s": solution.gain_interval_s,
        "delays_s": solution.delays_s,
        "delay_valid": solution.delay_valid,
        "bandpass": solution.bandpass,
        "bandpass_frequency_hz": solution.bandpass_frequency_hz,
        "bandpass_valid": solution.bandpass_valid,
        "antenna_position_offset_m": (
            np.empty((0, 3))
            if solution.antenna_position_offset_m is None
            else solution.antenna_position_offset_m
        ),
    }
    optional = {
        "antenna_position_m": solution.antenna_position_m,
        "cross_hand_delay_s": solution.cross_hand_delay_s,
        "cross_hand_delay_valid": solution.cross_hand_delay_valid,
        "leakage": solution.leakage,
        "leakage_frequency_hz": solution.leakage_frequency_hz,
        "leakage_valid": solution.leakage_valid,
        "leakage_time_s": solution.leakage_time_s,
        "rl_phase": solution.rl_phase,
        "rl_phase_frequency_hz": solution.rl_phase_frequency_hz,
        "rl_phase_valid": solution.rl_phase_valid,
    }
    for name, value in optional.items():
        if value is not None:
            payload[name] = value
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    destination.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "correlations": [value.value for value in solution.correlations],
                "receptors": [value.value for value in solution.receptors or ()],
                "reference_antenna": solution.reference_antenna,
                "reference_frequency_hz": solution.reference_frequency_hz,
                "interpolation": solution.interpolation,
                "apply_parallactic_angle": solution.apply_parallactic_angle,
                "leakage_application": solution.leakage_application,
                "parallactic_model": solution.parallactic_model,
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

        def optional(name: str) -> np.ndarray | None:
            return np.array(arrays[name]) if name in arrays.files else None

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
            antenna_position_m=optional("antenna_position_m"),
            cross_hand_delay_s=optional("cross_hand_delay_s"),
            cross_hand_delay_valid=optional("cross_hand_delay_valid"),
            leakage=optional("leakage"),
            leakage_frequency_hz=optional("leakage_frequency_hz"),
            leakage_valid=optional("leakage_valid"),
            leakage_time_s=optional("leakage_time_s"),
            rl_phase=optional("rl_phase"),
            rl_phase_frequency_hz=optional("rl_phase_frequency_hz"),
            rl_phase_valid=optional("rl_phase_valid"),
            apply_parallactic_angle=bool(metadata.get("apply_parallactic_angle", False)),
            leakage_application=str(metadata.get("leakage_application", "exact")),
            parallactic_model=str(metadata.get("parallactic_model", "gmst_geodetic")),
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
    shape = (unique_times.size, antenna_count, *value_shape)
    if np.issubdtype(values.dtype, np.complexfloating):
        output = np.full(shape, np.nan + 1j * np.nan, dtype=values.dtype)
    else:
        output = np.ones(shape, dtype=values.dtype)
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
        gain_interval = np.where(gain_interval <= 0.0, np.inf, gain_interval)
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


def _embedded_cal_frequencies(ms_frequency_hz: np.ndarray, n_cal: int) -> np.ndarray:
    frequencies = np.asarray(ms_frequency_hz, dtype=np.float64)
    if n_cal == frequencies.size:
        return frequencies
    extra = frequencies.size - n_cal
    if n_cal < 1 or extra < 0:
        raise ValueError("calibration table channel count is incompatible with the MS")
    start = extra // 2
    return frequencies[start : start + n_cal]


def import_casa_polarization_solution(
    pol_fixture: str | Path,
    kbg_fixture: str | Path,
    *,
    label: str,
    interpolation: str = "nearest",
) -> CalibrationSolution:
    """Import CASA K/B/G plus Kcross/Df/Xf for one polarisation golden case.

    ``flux_angle`` uses fluxscaled G on 3C286. ``leakage_calibrator`` uses G84
    on 3C84.  Flagged Df/Xf/Kcross rows are invalid, not identity.  CASA
    ``INTERVAL<=0`` is treated as unbounded.  Apply uses
    ``leakage_application='casa_parallel_preserving'``.  Kcross FPARAM is
    negated like the parallel-hand K table.
    """

    pol_source = Path(pol_fixture)
    pol_metadata = json.loads(pol_source.with_suffix(".json").read_text(encoding="utf-8"))
    if label not in pol_metadata["visibility_cases"]:
        raise ValueError(f"unknown polarisation golden case {label!r}")
    field_id = int(pol_metadata["visibility_cases"][label]["field_id"])
    base = import_casa_golden_solution(
        kbg_fixture,
        field_id=0,
        interpolation=interpolation,
    )
    with np.load(pol_source) as arrays:
        frequencies = np.asarray(arrays["frequency_hz"], dtype=np.float64)
        antenna_position_m = np.asarray(arrays["antenna_position_m"], dtype=np.float64)
        if label == "leakage_calibrator":
            gain_times, gains, gain_valid = _pivot_casa_table(arrays, "leakage_gain", "cparam")
            interval_times, gain_interval, _ = _pivot_casa_table(arrays, "leakage_gain", "interval")
            if not np.array_equal(interval_times, gain_times):
                raise ValueError("G84 interval and parameter coordinates differ")
            gains = gains[:, :, 0, :]
            gain_valid = gain_valid[:, :, 0, :]
            gain_interval = np.broadcast_to(gain_interval[:, :, None], gains.shape).copy()
            gain_interval = np.where(gain_interval <= 0.0, np.inf, gain_interval)
            base = replace(
                base,
                gains=gains,
                gain_time_s=gain_times,
                gain_valid=gain_valid,
                gain_interval_s=gain_interval,
            )
        elif label != "flux_angle":
            raise ValueError(f"unsupported polarisation golden case {label!r}")
        _, kcross, kcross_valid = _pivot_casa_table(arrays, "kcross", "fparam")
        cross_hand_delay_s = -kcross[0, :, 0, :] * 1e-9
        cross_hand_delay_valid = kcross_valid[0, :, 0, :]
        _, leakage, leakage_valid = _pivot_casa_table(arrays, "dterms", "cparam")
        leakage = leakage[0]
        leakage_valid = leakage_valid[0]
        _, rl_phase, rl_phase_valid = _pivot_casa_table(arrays, "angle", "cparam")
        rl_phase = rl_phase[0, :, :, 0]
        rl_phase_valid = rl_phase_valid[0, :, :, 0]
        leakage_frequency_hz = _embedded_cal_frequencies(frequencies, leakage.shape[1])
        rl_phase_frequency_hz = _embedded_cal_frequencies(frequencies, rl_phase.shape[1])
    correlations = tuple(Correlation(value) for value in pol_metadata["correlations"])
    provenance = {
        **dict(base.provenance),
        "polarization_source": pol_source.name,
        "polarization_label": label,
        "field_id": field_id,
        "gain_table": "leakage_gain" if label == "leakage_calibrator" else "flux_gain",
        "kcross_convention": "negated_ns_like_K",
        "xf_convention": "diag(CPARAM, 1)",
        "df_flagged": "invalid",
        "jones_order": "GKB Kcross D X P",
        "leakage_application": "casa_parallel_preserving",
    }
    return replace(
        base,
        correlations=correlations,
        antenna_position_m=antenna_position_m,
        cross_hand_delay_s=cross_hand_delay_s,
        cross_hand_delay_valid=cross_hand_delay_valid,
        leakage=leakage,
        leakage_frequency_hz=leakage_frequency_hz,
        leakage_valid=leakage_valid,
        rl_phase=rl_phase,
        rl_phase_frequency_hz=rl_phase_frequency_hz,
        rl_phase_valid=rl_phase_valid,
        apply_parallactic_angle=True,
        leakage_application="casa_parallel_preserving",
        parallactic_model="gmst_geodetic",
        provenance={
            **provenance,
            "apply_contract": "casa_parallel_preserving_legacy_v1",
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
