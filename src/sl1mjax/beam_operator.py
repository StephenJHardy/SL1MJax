"""Streamed per-timestep voltage-Jones visibility operator.

Phase 4 evaluates one beam slice at each exact unique time, applies
``E_p C E_q^H`` to the packed circular coherency, and releases that
slice before the next time unless the caller asked to materialize or
retain it. This is the reference operator, not a cache and not the
3C391 imaging path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import (
    BeamCalibrationState,
    require_beam_calibration_state,
)
from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import (
    Correlation,
    Receptor,
    ReceptorBasis,
    apply_jones_to_coherency,
    circular_stokes_to_coherency,
    pack_coherency,
    unpack_coherency,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import (
    BeamEvaluation,
    VoltageBeamModel,
    beam_coordinates,
)

JONES_RECEPTORS = (Receptor.R, Receptor.L)
_COMPLEX128_JONES_BYTES = 2 * 2 * np.dtype(np.complex128).itemsize


class BeamOperatorPolicy(StrEnum):
    """How long a timestep Jones slice is retained.

    ``stream`` evaluates, consumes, and releases one time. ``retain_last``
    is the same prediction and keeps the last slice for inspection.
    ``materialize`` keeps every slice for a small fixture or benchmark.
    """

    STREAM = "stream"
    RETAIN_LAST = "retain_last"
    MATERIALIZE = "materialize"


@dataclass(frozen=True)
class BeamOperatorConfig:
    """Memory bounds and execution policy for the reference operator."""

    policy: BeamOperatorPolicy = BeamOperatorPolicy.STREAM
    max_timestep_jones_bytes: int = 64 * 1024**2
    visibility_chunk_size: int = 256
    pixel_chunk_size: int = 1024
    pointing_offset_lm_rad: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.max_timestep_jones_bytes < 1:
            raise ValueError("max_timestep_jones_bytes must be positive")
        if self.visibility_chunk_size < 1 or self.pixel_chunk_size < 1:
            raise ValueError("operator chunk sizes must be positive")


@dataclass(frozen=True)
class SkyStokesPlanes:
    """Real Stokes planes on the sky-direction axis, optionally per channel."""

    stokes_i: np.ndarray
    stokes_q: np.ndarray | None = None
    stokes_u: np.ndarray | None = None
    stokes_v: np.ndarray | None = None


@dataclass(frozen=True)
class BeamOperatorResult:
    """Predicted circular correlations and the slices the policy retained."""

    visibility: np.ndarray
    valid: np.ndarray
    provenance: dict[str, object]
    last_evaluation: BeamEvaluation | None = None
    materialized: tuple[BeamEvaluation, ...] | None = None


def unique_visibility_times(time_s: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique times and the row→time inverse index."""

    return np.unique(np.asarray(time_s, dtype=np.float64), return_inverse=True)


def timestep_jones_bytes(
    antenna_count: int, direction_count: int, channel_count: int
) -> int:
    """Bytes for one complex128 Jones slice, including the 2×2 receptors."""

    if min(antenna_count, direction_count, channel_count) < 1:
        raise ValueError("Jones slice dimensions must be positive")
    return antenna_count * direction_count * channel_count * _COMPLEX128_JONES_BYTES


def predict_voltage_beam(
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    sky: SkyStokesPlanes,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
) -> BeamOperatorResult:
    """Predict ``E_p C E_q^H`` visibilities, one exact unique time at a time."""

    selected = config or BeamOperatorConfig()
    state = require_beam_calibration_state(calibration_state)
    l, m, coherency = _prepare_sky(l_rad, m_rad, sky, block.frequency_hz.size)
    positions = _require_antenna_positions(antenna_position_m, block)
    _require_circular_block(block)
    unique_times, row_time_index = unique_visibility_times(block.time_s)
    plane_count = _antenna_plane_count(beam, block.antenna_count)
    batch_antennas = _can_batch_antenna_planes(
        selected,
        unique_times.size,
        plane_count=plane_count,
        direction_count=l.size,
        channel_count=block.frequency_hz.size,
    )
    prediction = np.zeros(block.visibility.shape, dtype=np.complex128)
    valid = np.zeros(block.visibility.shape[:2], dtype=bool)
    last: BeamEvaluation | None = None
    materialized: list[BeamEvaluation] = []
    antennas = np.arange(block.antenna_count, dtype=np.int32)
    for time_index, time_s in enumerate(unique_times):
        selected_rows = np.flatnonzero(row_time_index == time_index)
        if batch_antennas:
            evaluation, _parallactic = _evaluate_timestep(
                block,
                l,
                m,
                float(time_s),
                beam=beam,
                antenna_id=antennas,
                antenna_position_m=positions,
                calibration_state=state,
                pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
            )
            last = evaluation
            if selected.policy is BeamOperatorPolicy.MATERIALIZE:
                materialized.append(evaluation)
            _accumulate_timestep(
                prediction,
                valid,
                block,
                selected_rows,
                l,
                m,
                coherency,
                evaluation,
                selected,
            )
        else:
            last = _accumulate_timestep_streamed_antennas(
                prediction,
                valid,
                block,
                selected_rows,
                l,
                m,
                coherency,
                float(time_s),
                beam=beam,
                antenna_position_m=positions,
                calibration_state=state,
                config=selected,
            )
        if selected.policy is BeamOperatorPolicy.STREAM:
            last = None
    return BeamOperatorResult(
        visibility=prediction,
        valid=valid,
        provenance=_operator_provenance(
            block,
            beam,
            state,
            selected,
            unique_times,
            positions,
        ),
        last_evaluation=last,
        materialized=tuple(materialized) if materialized else None,
    )


def adjoint_voltage_beam(
    residual: ArrayLike,
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the real Stokes adjoint ``Re(Aᴴ residual)`` for a fixed beam.

    Each returned plane has shape ``(direction, channel)``. A sky that was
    broadcast across channels should contract the channel axis itself.
    """

    selected = config or BeamOperatorConfig()
    state = require_beam_calibration_state(calibration_state)
    l = np.asarray(l_rad, dtype=np.float64).reshape(-1)
    m = np.asarray(m_rad, dtype=np.float64).reshape(-1)
    if l.size != m.size or l.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    residual_array = np.asarray(residual, dtype=np.complex128)
    if residual_array.shape != block.visibility.shape:
        raise ValueError("residual must match block.visibility")
    positions = _require_antenna_positions(antenna_position_m, block)
    _require_circular_block(block)
    unique_times, row_time_index = unique_visibility_times(block.time_s)
    plane_count = _antenna_plane_count(beam, block.antenna_count)
    batch_antennas = _can_batch_antenna_planes(
        selected,
        unique_times.size,
        plane_count=plane_count,
        direction_count=l.size,
        channel_count=block.frequency_hz.size,
    )
    gradient = np.zeros((l.size, block.frequency_hz.size, 2, 2), dtype=np.complex128)
    antennas = np.arange(block.antenna_count, dtype=np.int32)
    for time_index, time_s in enumerate(unique_times):
        selected_rows = np.flatnonzero(row_time_index == time_index)
        if batch_antennas:
            evaluation, _parallactic = _evaluate_timestep(
                block,
                l,
                m,
                float(time_s),
                beam=beam,
                antenna_id=antennas,
                antenna_position_m=positions,
                calibration_state=state,
                pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
            )
            _accumulate_timestep_adjoint(
                gradient,
                residual_array,
                block,
                selected_rows,
                l,
                m,
                evaluation,
                selected,
            )
        else:
            _accumulate_timestep_adjoint_streamed_antennas(
                gradient,
                residual_array,
                block,
                selected_rows,
                l,
                m,
                float(time_s),
                beam=beam,
                antenna_position_m=positions,
                calibration_state=state,
                config=selected,
            )
    return _stokes_from_coherency_gradient(gradient)


def _evaluate_timestep(
    block: VisibilityBlock,
    l: np.ndarray,
    m: np.ndarray,
    time_s: float,
    *,
    beam: VoltageBeamModel,
    antenna_id: np.ndarray,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    pointing_offset_lm_rad: tuple[float, float] | None,
) -> tuple[BeamEvaluation, np.ndarray]:
    parallactic = parallactic_angle_rad(
        np.asarray([time_s]), block.phase_centre_rad, antenna_position_m
    )[0]
    required = int(np.max(antenna_id)) + 1
    if parallactic.size < required:
        raise ValueError("antenna_position_m must cover every antenna_id")
    coordinates = beam_coordinates(
        l,
        m,
        block.frequency_hz,
        parallactic_angle_rad=parallactic[np.asarray(antenna_id)],
        pointing_offset_lm_rad=pointing_offset_lm_rad,
        antenna_id=antenna_id,
    )
    return beam.evaluate(coordinates, calibration_state=calibration_state), parallactic


def _accumulate_timestep(
    prediction: np.ndarray,
    valid: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    sky_coherency: np.ndarray,
    evaluation: BeamEvaluation,
    config: BeamOperatorConfig,
) -> None:
    jones, valid_jones = _aligned_antenna_jones(evaluation, block.antenna_count)
    _accumulate_from_planes(
        prediction,
        valid,
        block,
        selected_rows,
        l,
        m,
        sky_coherency,
        jones,
        valid_jones,
        block.antenna1,
        block.antenna2,
        config,
    )


def _accumulate_timestep_streamed_antennas(
    prediction: np.ndarray,
    valid: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    sky_coherency: np.ndarray,
    time_s: float,
    *,
    beam: VoltageBeamModel,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
) -> BeamEvaluation:
    last: BeamEvaluation | None = None
    pairs = np.unique(
        np.stack(
            (block.antenna1[selected_rows], block.antenna2[selected_rows]), axis=1
        ),
        axis=0,
    )
    for antenna_p, antenna_q in pairs:
        evaluation_p, _ = _evaluate_timestep(
            block,
            l,
            m,
            time_s,
            beam=beam,
            antenna_id=np.asarray([antenna_p], dtype=np.int32),
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            pointing_offset_lm_rad=config.pointing_offset_lm_rad,
        )
        last = evaluation_p
        if antenna_q == antenna_p:
            evaluation_q = evaluation_p
        else:
            evaluation_q, _ = _evaluate_timestep(
                block,
                l,
                m,
                time_s,
                beam=beam,
                antenna_id=np.asarray([antenna_q], dtype=np.int32),
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                pointing_offset_lm_rad=config.pointing_offset_lm_rad,
            )
            last = evaluation_q
        pair_rows = selected_rows[
            (block.antenna1[selected_rows] == antenna_p)
            & (block.antenna2[selected_rows] == antenna_q)
        ]
        jones = np.concatenate((evaluation_p.jones, evaluation_q.jones), axis=0)
        valid_jones = np.concatenate((evaluation_p.valid, evaluation_q.valid), axis=0)
        antenna1 = np.zeros(block.antenna1.shape, dtype=np.int32)
        antenna2 = np.ones(block.antenna2.shape, dtype=np.int32)
        if antenna_q == antenna_p:
            antenna2 = antenna1
        _accumulate_from_planes(
            prediction,
            valid,
            block,
            pair_rows,
            l,
            m,
            sky_coherency,
            jones,
            valid_jones,
            antenna1,
            antenna2,
            config,
        )
    if last is None:
        raise ValueError("timestep has no rows")
    return last


def _accumulate_from_planes(
    prediction: np.ndarray,
    valid: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    sky_coherency: np.ndarray,
    jones: np.ndarray,
    valid_jones: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    config: BeamOperatorConfig,
) -> None:
    for row_start in range(0, selected_rows.size, config.visibility_chunk_size):
        row_stop = min(row_start + config.visibility_chunk_size, selected_rows.size)
        rows = selected_rows[row_start:row_stop]
        for pixel_start in range(0, l.size, config.pixel_chunk_size):
            pixel_stop = min(pixel_start + config.pixel_chunk_size, l.size)
            pixels = slice(pixel_start, pixel_stop)
            for channel in range(block.frequency_hz.size):
                kernel = _delta_kernel(
                    block.uvw_m[rows] * (block.frequency_hz[channel] / SPEED_OF_LIGHT_M_S),
                    l[pixels],
                    m[pixels],
                )
                if jones.shape[0] == 1:
                    pix_ok = valid_jones[0, pixels, channel]
                    apparent = apply_jones_to_coherency(
                        sky_coherency[pixels, channel],
                        jones[0, pixels, channel],
                        jones[0, pixels, channel],
                    )
                    apparent = np.where(pix_ok[:, None, None], apparent, 0.0)
                    packed = unpack_coherency(
                        apparent, block.correlations, JONES_RECEPTORS
                    )
                    prediction[rows, channel] += kernel @ packed
                    valid[rows, channel] |= np.any(pix_ok)
                    continue
                for local_row, row in enumerate(rows):
                    pix_ok = (
                        valid_jones[antenna1[row], pixels, channel]
                        & valid_jones[antenna2[row], pixels, channel]
                    )
                    apparent = apply_jones_to_coherency(
                        sky_coherency[pixels, channel],
                        jones[antenna1[row], pixels, channel],
                        jones[antenna2[row], pixels, channel],
                    )
                    apparent = np.where(pix_ok[:, None, None], apparent, 0.0)
                    packed = unpack_coherency(
                        apparent, block.correlations, JONES_RECEPTORS
                    )
                    prediction[row, channel] += kernel[local_row] @ packed
                    valid[row, channel] |= bool(np.any(pix_ok))


def _accumulate_timestep_adjoint(
    gradient: np.ndarray,
    residual: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    evaluation: BeamEvaluation,
    config: BeamOperatorConfig,
) -> None:
    jones, valid_jones = _aligned_antenna_jones(evaluation, block.antenna_count)
    _accumulate_adjoint_from_planes(
        gradient,
        residual,
        block,
        selected_rows,
        l,
        m,
        jones,
        valid_jones,
        block.antenna1,
        block.antenna2,
        config,
    )


def _accumulate_timestep_adjoint_streamed_antennas(
    gradient: np.ndarray,
    residual: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    time_s: float,
    *,
    beam: VoltageBeamModel,
    antenna_position_m: np.ndarray,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
) -> None:
    pairs = np.unique(
        np.stack(
            (block.antenna1[selected_rows], block.antenna2[selected_rows]), axis=1
        ),
        axis=0,
    )
    for antenna_p, antenna_q in pairs:
        evaluation_p, _ = _evaluate_timestep(
            block,
            l,
            m,
            time_s,
            beam=beam,
            antenna_id=np.asarray([antenna_p], dtype=np.int32),
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            pointing_offset_lm_rad=config.pointing_offset_lm_rad,
        )
        evaluation_q = evaluation_p
        if antenna_q != antenna_p:
            evaluation_q, _ = _evaluate_timestep(
                block,
                l,
                m,
                time_s,
                beam=beam,
                antenna_id=np.asarray([antenna_q], dtype=np.int32),
                antenna_position_m=antenna_position_m,
                calibration_state=calibration_state,
                pointing_offset_lm_rad=config.pointing_offset_lm_rad,
            )
        pair_rows = selected_rows[
            (block.antenna1[selected_rows] == antenna_p)
            & (block.antenna2[selected_rows] == antenna_q)
        ]
        jones = np.concatenate((evaluation_p.jones, evaluation_q.jones), axis=0)
        valid_jones = np.concatenate((evaluation_p.valid, evaluation_q.valid), axis=0)
        antenna1 = np.zeros(block.antenna1.shape, dtype=np.int32)
        antenna2 = np.ones(block.antenna2.shape, dtype=np.int32)
        if antenna_q == antenna_p:
            antenna2 = antenna1
        _accumulate_adjoint_from_planes(
            gradient,
            residual,
            block,
            pair_rows,
            l,
            m,
            jones,
            valid_jones,
            antenna1,
            antenna2,
            config,
        )


def _accumulate_adjoint_from_planes(
    gradient: np.ndarray,
    residual: np.ndarray,
    block: VisibilityBlock,
    selected_rows: np.ndarray,
    l: np.ndarray,
    m: np.ndarray,
    jones: np.ndarray,
    valid_jones: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    config: BeamOperatorConfig,
) -> None:
    packed_residual = pack_coherency(
        residual[selected_rows], block.correlations, JONES_RECEPTORS
    )
    for row_start in range(0, selected_rows.size, config.visibility_chunk_size):
        row_stop = min(row_start + config.visibility_chunk_size, selected_rows.size)
        local = slice(row_start, row_stop)
        rows = selected_rows[local]
        for pixel_start in range(0, l.size, config.pixel_chunk_size):
            pixel_stop = min(pixel_start + config.pixel_chunk_size, l.size)
            pixels = slice(pixel_start, pixel_stop)
            for channel in range(block.frequency_hz.size):
                kernel = _delta_kernel(
                    block.uvw_m[rows] * (block.frequency_hz[channel] / SPEED_OF_LIGHT_M_S),
                    l[pixels],
                    m[pixels],
                )
                if jones.shape[0] == 1:
                    pix_ok = valid_jones[0, pixels, channel]
                    pulled = np.einsum(
                        "rp,rij->pij",
                        np.conjugate(kernel),
                        packed_residual[local, channel],
                    )
                    pulled = np.where(pix_ok[:, None, None], pulled, 0.0)
                    left = np.conjugate(np.swapaxes(jones[0, pixels, channel], -1, -2))
                    right = jones[0, pixels, channel]
                    gradient[pixels, channel] += np.matmul(left, np.matmul(pulled, right))
                    continue
                for local_row, row in enumerate(rows):
                    pix_ok = (
                        valid_jones[antenna1[row], pixels, channel]
                        & valid_jones[antenna2[row], pixels, channel]
                    )
                    left = np.conjugate(
                        np.swapaxes(jones[antenna1[row], pixels, channel], -1, -2)
                    )
                    right = jones[antenna2[row], pixels, channel]
                    row_pull = (
                        np.conjugate(kernel[local_row])[:, None, None]
                        * packed_residual[row_start + local_row, channel]
                    )
                    row_pull = np.where(pix_ok[:, None, None], row_pull, 0.0)
                    gradient[pixels, channel] += np.matmul(left, np.matmul(row_pull, right))


def _aligned_antenna_jones(
    evaluation: BeamEvaluation, antenna_count: int
) -> tuple[np.ndarray, np.ndarray]:
    jones = np.asarray(evaluation.jones)
    valid = np.asarray(evaluation.valid, dtype=bool)
    if jones.shape[0] == 1:
        return jones, valid
    if jones.shape[0] != antenna_count:
        raise ValueError(
            "beam antenna axis must be 1 or match the block antenna count"
        )
    return jones, valid


def _antenna_plane_count(beam: VoltageBeamModel, antenna_count: int) -> int:
    if bool(getattr(beam, "antenna_planes_from_parallactic", False)):
        return antenna_count
    return 1


def _can_batch_antenna_planes(
    config: BeamOperatorConfig,
    time_count: int,
    *,
    plane_count: int,
    direction_count: int,
    channel_count: int,
) -> bool:
    one_plane = timestep_jones_bytes(1, direction_count, channel_count)
    if one_plane > config.max_timestep_jones_bytes:
        raise ValueError(
            "one timestep Jones slice requires "
            f"{one_plane} bytes, exceeding max_timestep_jones_bytes="
            f"{config.max_timestep_jones_bytes}"
        )
    full = one_plane * plane_count
    if config.policy is BeamOperatorPolicy.MATERIALIZE:
        total = full * time_count
        if total > config.max_timestep_jones_bytes:
            raise ValueError(
                "materialized Jones slices require "
                f"{total} bytes, exceeding max_timestep_jones_bytes="
                f"{config.max_timestep_jones_bytes}"
            )
        if full > config.max_timestep_jones_bytes:
            raise ValueError(
                "one timestep Jones slice requires "
                f"{full} bytes, exceeding max_timestep_jones_bytes="
                f"{config.max_timestep_jones_bytes}"
            )
        return True
    return full <= config.max_timestep_jones_bytes


def _prepare_sky(
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    sky: SkyStokesPlanes,
    channel_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l = np.asarray(l_rad, dtype=np.float64).reshape(-1)
    m = np.asarray(m_rad, dtype=np.float64).reshape(-1)
    if l.size != m.size or l.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    intensity = _plane(sky.stokes_i, l.size, channel_count, "stokes_i")
    stokes_q = _plane(sky.stokes_q, l.size, channel_count, "stokes_q", optional=True)
    stokes_u = _plane(sky.stokes_u, l.size, channel_count, "stokes_u", optional=True)
    stokes_v = _plane(sky.stokes_v, l.size, channel_count, "stokes_v", optional=True)
    return l, m, circular_stokes_to_coherency(intensity, stokes_q, stokes_u, stokes_v)


def _plane(
    values: ArrayLike | None,
    direction_count: int,
    channel_count: int,
    name: str,
    *,
    optional: bool = False,
) -> np.ndarray:
    if values is None:
        if not optional:
            raise ValueError(f"{name} is required")
        return np.zeros((direction_count, channel_count), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        if array.size != direction_count:
            raise ValueError(f"{name} must match the direction axis")
        array = np.broadcast_to(array[:, None], (direction_count, channel_count))
    elif array.shape != (direction_count, channel_count):
        raise ValueError(
            f"{name} must have shape ({direction_count},) or "
            f"({direction_count}, {channel_count})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(array, dtype=np.float64)


def _delta_kernel(uvw_wavelengths: np.ndarray, l: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = np.sqrt(np.maximum(1.0 - l * l - m * m, 0.0))
    phase = 2j * np.pi * (
        uvw_wavelengths[:, 0, None] * l[None, :]
        + uvw_wavelengths[:, 1, None] * m[None, :]
        + uvw_wavelengths[:, 2, None] * (n[None, :] - 1.0)
    )
    return np.asarray(np.exp(phase), dtype=np.complex128)


def _stokes_from_coherency_gradient(
    gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stokes_i = np.real(gradient[..., 0, 0] + gradient[..., 1, 1])
    stokes_q = np.real(gradient[..., 0, 1] + gradient[..., 1, 0])
    stokes_u = np.imag(gradient[..., 0, 1] - gradient[..., 1, 0])
    stokes_v = np.real(gradient[..., 0, 0] - gradient[..., 1, 1])
    return stokes_i, stokes_q, stokes_u, stokes_v


def _require_circular_block(block: VisibilityBlock) -> None:
    if block.receptor_basis is not ReceptorBasis.CIRCULAR:
        raise ValueError("voltage-beam operator requires circular correlations")
    unknown = set(block.correlations) - {
        Correlation.RR,
        Correlation.RL,
        Correlation.LR,
        Correlation.LL,
    }
    if unknown:
        raise ValueError(f"unsupported circular correlations {sorted(unknown)}")


def _require_antenna_positions(
    antenna_position_m: ArrayLike, block: VisibilityBlock
) -> NDArray[np.float64]:
    positions = np.asarray(antenna_position_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("antenna_position_m must have shape (antenna, 3)")
    if positions.shape[0] < block.antenna_count:
        raise ValueError("antenna_position_m must cover every antenna in the block")
    if not np.all(np.isfinite(positions)):
        raise ValueError("antenna_position_m must be finite")
    return positions


def _operator_provenance(
    block: VisibilityBlock,
    beam: VoltageBeamModel,
    calibration_state: BeamCalibrationState,
    config: BeamOperatorConfig,
    unique_times: np.ndarray,
    antenna_position_m: np.ndarray,
) -> dict[str, object]:
    return {
        "operator": "voltage_beam_stream",
        "model_id": beam.model_id,
        "policy": config.policy.value,
        "calibration_state": calibration_state.value,
        "jones_equation": "E_p C E_q^H",
        "pixel_basis": "delta",
        "unique_time_count": int(unique_times.size),
        "antenna_count": int(block.antenna_count),
        "receptors": [receptor.value for receptor in JONES_RECEPTORS],
        "parallactic_angle": "calibration_terms.parallactic_angle_rad",
        "parallactic_angle_rad_shape": list(
            parallactic_angle_rad(
                unique_times, block.phase_centre_rad, antenna_position_m
            ).shape
        ),
        "antenna_position_m_shape": list(antenna_position_m.shape),
        "pointing_offset_lm_rad": (
            None
            if config.pointing_offset_lm_rad is None
            else list(config.pointing_offset_lm_rad)
        ),
        "visibility_chunk_size": config.visibility_chunk_size,
        "pixel_chunk_size": config.pixel_chunk_size,
        "creates_cache": False,
    }
