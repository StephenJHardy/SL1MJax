"""Exploratory I, Q, U, V diagnostics on a frozen CASA-compatible apply.

These products are not evidence-grade while exact 2×2 Jones still produces
false Stokes V. A global ``q,u,v`` fit is a detection and calibration check,
not a spatial polarisation image.

The global fit uses the frozen complex Stokes-I model ``M_I`` as the
regressor. Observed ``Re(I)`` is not a valid intensity for an extended
source with complex visibilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.diagnostics import dirty_weighted_image_and_psf
from sl1mjax.direct_operator import DirectDFTConfig, direct_scalar_adjoint
from sl1mjax.polarization import (
    Correlation,
    circular_stokes_from_correlations,
    electric_vector_position_angle_rad,
    fractional_linear_polarisation,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.sky import RegularGrid
from sl1mjax.split import VisibilitySplit

Independence = Literal["apply_back", "held_out_calibrator", "in_sample"]
PARTITION_NAMES = (
    "baseline_even",
    "baseline_odd",
    "time_even",
    "time_odd",
    "channel_even",
    "channel_odd",
)


@dataclass(frozen=True)
class StokesVisibilityPlanes:
    stokes_i: NDArray[np.complex128]
    stokes_q: NDArray[np.complex128]
    stokes_u: NDArray[np.complex128]
    stokes_v: NDArray[np.complex128]
    weight_i: NDArray[np.float64]
    weight_linear: NDArray[np.float64]
    weight_v: NDArray[np.float64]


@dataclass(frozen=True)
class DirtyStokesImages:
    stokes_i: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]
    stokes_v: NDArray[np.float64]
    psf: NDArray[np.float64]
    peak_i: float
    peak_q: float
    peak_u: float
    peak_v: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PolarizationFloor:
    label: str
    independence: Independence
    stokes_i: float
    stokes_q: float
    stokes_u: float
    stokes_v: float
    q: float
    u: float
    v: float
    fractional_linear: float
    casaguide_angle_rad: float
    evpa_rad: float
    n_samples: int
    model_q: float | None
    model_u: float | None
    model_v: float | None
    residual_q: float | None
    residual_u: float | None
    residual_v: float | None
    residual_fractional_linear: float | None


@dataclass(frozen=True)
class GlobalFractionalPolarization:
    q: float
    u: float
    v: float
    fractional_linear: float
    casaguide_angle_rad: float
    evpa_rad: float
    n_samples: int
    null_linear_loss: float
    polarized_linear_loss: float
    null_v_loss: float
    polarized_v_loss: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MosaicDirtyStokesImages:
    stokes_i: NDArray[np.float64]
    stokes_q: NDArray[np.float64]
    stokes_u: NDArray[np.float64]
    stokes_v: NDArray[np.float64]
    sensitivity: NDArray[np.float64]
    sensitivity_fraction: NDArray[np.float64]
    per_pointing: dict[str, dict[str, NDArray[np.float64]]]
    recurrence: dict[str, Any]
    provenance: dict[str, Any]


def stokes_visibility_planes(block: VisibilityBlock) -> StokesVisibilityPlanes:
    """Unpack circular products into weighted Stokes visibility planes."""

    stokes_i, stokes_q, stokes_u, stokes_v = circular_stokes_from_correlations(
        block.visibility, block.correlations
    )
    active = block.active
    index = {correlation: slot for slot, correlation in enumerate(block.correlations)}
    rr = active[..., index[Correlation.RR]]
    rl = active[..., index[Correlation.RL]]
    lr = active[..., index[Correlation.LR]]
    ll = active[..., index[Correlation.LL]]
    parallel = rr & ll
    linear = rl & lr
    weight_rr = np.where(rr, block.weight[..., index[Correlation.RR]], 0.0)
    weight_ll = np.where(ll, block.weight[..., index[Correlation.LL]], 0.0)
    weight_rl = np.where(rl, block.weight[..., index[Correlation.RL]], 0.0)
    weight_lr = np.where(lr, block.weight[..., index[Correlation.LR]], 0.0)
    weight_i = np.where(parallel, 0.5 * (weight_rr + weight_ll), 0.0)
    weight_v = weight_i
    weight_linear = np.where(linear, 0.5 * (weight_rl + weight_lr), 0.0)
    return StokesVisibilityPlanes(
        stokes_i=np.where(parallel, stokes_i, 0.0),
        stokes_q=np.where(linear, stokes_q, 0.0),
        stokes_u=np.where(linear, stokes_u, 0.0),
        stokes_v=np.where(parallel, stokes_v, 0.0),
        weight_i=weight_i,
        weight_linear=weight_linear,
        weight_v=weight_v,
    )


def dirty_stokes_images(
    block: VisibilityBlock,
    grid: RegularGrid,
    *,
    chunk_size: int = 256,
) -> DirtyStokesImages:
    """Dirty-image Stokes I, Q, U, and V from a circular four-product block."""

    planes = stokes_visibility_planes(block)
    dirty_i, psf = dirty_weighted_image_and_psf(
        planes.weight_i * planes.stokes_i,
        planes.weight_i,
        block.uvw_m,
        block.frequency_hz,
        grid,
        chunk_size=chunk_size,
    )
    dirty_q, _ = dirty_weighted_image_and_psf(
        planes.weight_linear * planes.stokes_q,
        planes.weight_linear,
        block.uvw_m,
        block.frequency_hz,
        grid,
        chunk_size=chunk_size,
    )
    dirty_u, _ = dirty_weighted_image_and_psf(
        planes.weight_linear * planes.stokes_u,
        planes.weight_linear,
        block.uvw_m,
        block.frequency_hz,
        grid,
        chunk_size=chunk_size,
    )
    dirty_v, _ = dirty_weighted_image_and_psf(
        planes.weight_v * planes.stokes_v,
        planes.weight_v,
        block.uvw_m,
        block.frequency_hz,
        grid,
        chunk_size=chunk_size,
    )
    peak = int(np.argmax(np.abs(dirty_i)))
    return DirtyStokesImages(
        stokes_i=dirty_i,
        stokes_q=dirty_q,
        stokes_u=dirty_u,
        stokes_v=dirty_v,
        psf=psf,
        peak_i=float(dirty_i.ravel()[peak]),
        peak_q=float(dirty_q.ravel()[peak]),
        peak_u=float(dirty_u.ravel()[peak]),
        peak_v=float(dirty_v.ravel()[peak]),
        provenance={
            "kind": "exploratory_dirty_stokes",
            "evidence_grade": False,
            "leakage_application_note": (
                "interpret Stokes V as exploratory while exact Jones is open"
            ),
        },
    )


def _median_real(values: np.ndarray, weight: np.ndarray) -> float:
    selected = weight > 0
    if not np.any(selected):
        raise ValueError("no finite Stokes samples to summarise")
    return float(np.median(np.real(values[selected])))


def _fractional_summary(
    stokes_i: float, stokes_q: float, stokes_u: float, stokes_v: float
) -> tuple[float, float, float, float, float, float]:
    if stokes_i == 0.0:
        raise ValueError("Stokes I is zero; fractional polarisation is undefined")
    q = stokes_q / stokes_i
    u = stokes_u / stokes_i
    v = stokes_v / stokes_i
    return (
        q,
        u,
        v,
        float(fractional_linear_polarisation(stokes_q, stokes_u, stokes_i)),
        float(np.arctan2(stokes_u, stokes_q)),
        float(electric_vector_position_angle_rad(stokes_q, stokes_u)),
    )


def calibrator_polarization_floor(
    block: VisibilityBlock,
    *,
    independence: Independence,
    label: str = "",
    sample_mask: np.ndarray | None = None,
) -> PolarizationFloor:
    """Summarise apparent fractional polarisation on a calibrator scan."""

    selected = _restricted_block(block, sample_mask)
    planes = stokes_visibility_planes(selected)
    stokes_i = _median_real(planes.stokes_i, planes.weight_i)
    stokes_q = _median_real(planes.stokes_q, planes.weight_linear)
    stokes_u = _median_real(planes.stokes_u, planes.weight_linear)
    stokes_v = _median_real(planes.stokes_v, planes.weight_v)
    q, u, v, fraction, casaguide, evpa = _fractional_summary(
        stokes_i, stokes_q, stokes_u, stokes_v
    )
    model_q = model_u = model_v = None
    residual_q = residual_u = residual_v = residual_fraction = None
    if selected.model_visibility is not None:
        model_i, model_qs, model_us, model_vs = circular_stokes_from_correlations(
            selected.model_visibility, selected.correlations
        )
        model_weight = np.where(
            planes.weight_i > 0, np.ones_like(planes.weight_i), 0.0
        )
        model_i_med = _median_real(model_i, model_weight)
        model_q_jy = _median_real(model_qs, model_weight)
        model_u_jy = _median_real(model_us, model_weight)
        model_v_jy = _median_real(model_vs, model_weight)
        if model_i_med != 0.0:
            model_q = model_q_jy / model_i_med
            model_u = model_u_jy / model_i_med
            model_v = model_v_jy / model_i_med
            residual_q = q - model_q
            residual_u = u - model_u
            residual_v = v - model_v
            residual_fraction = float(np.hypot(residual_q, residual_u))
    return PolarizationFloor(
        label=label,
        independence=independence,
        stokes_i=stokes_i,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        q=q,
        u=u,
        v=v,
        fractional_linear=fraction,
        casaguide_angle_rad=casaguide,
        evpa_rad=evpa,
        n_samples=int(np.count_nonzero(planes.weight_linear > 0)),
        model_q=model_q,
        model_u=model_u,
        model_v=model_v,
        residual_q=residual_q,
        residual_u=residual_u,
        residual_v=residual_v,
        residual_fractional_linear=residual_fraction,
    )


def _restricted_block(
    block: VisibilityBlock, sample_mask: np.ndarray | None
) -> VisibilityBlock:
    if sample_mask is None:
        return block
    return replace(block, flag=block.flag | ~_broadcast_sample_mask(block, sample_mask))


def _broadcast_sample_mask(block: VisibilityBlock, sample_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(sample_mask, dtype=bool)
    if mask.shape == block.shape:
        return mask
    if mask.shape == block.visibility.shape[:2]:
        return np.broadcast_to(mask[..., None], block.shape)
    if mask.shape == (block.visibility.shape[0],):
        return np.broadcast_to(mask[:, None, None], block.shape)
    raise ValueError(
        "sample_mask must match the block, the row-channel plane, or the row axis"
    )


def _model_stokes_i(
    block: VisibilityBlock, model_stokes_i: np.ndarray | None
) -> NDArray[np.complex128]:
    if model_stokes_i is not None:
        array = np.asarray(model_stokes_i, dtype=np.complex128)
        if array.shape != block.visibility.shape[:2]:
            raise ValueError(
                f"model_stokes_i has shape {array.shape}; "
                f"expected {block.visibility.shape[:2]}"
            )
        return array
    if block.model_visibility is None:
        raise ValueError(
            "global q,u,v fit requires a frozen complex Stokes-I model M_I; "
            "observed Re(I) is not a valid regressor"
        )
    model_i, *_ = circular_stokes_from_correlations(
        block.model_visibility, block.correlations
    )
    return np.asarray(model_i, dtype=np.complex128)


def _row_channel_mask(
    block: VisibilityBlock, sample_mask: np.ndarray | None
) -> NDArray[np.bool_]:
    if sample_mask is None:
        return np.any(block.active, axis=2)
    return np.any(_broadcast_sample_mask(block, sample_mask) & block.active, axis=2)


@dataclass(frozen=True)
class _RegressionSums:
    linear_num: complex
    linear_den: float
    v_num: complex
    v_den: float
    null_linear_loss: float
    null_v_loss: float
    n_samples: int


def _block_regression_sums(
    block: VisibilityBlock,
    *,
    model_stokes_i: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
) -> tuple[_RegressionSums, NDArray[np.complex128], StokesVisibilityPlanes, np.ndarray, np.ndarray]:
    planes = stokes_visibility_planes(block)
    model_i = _model_stokes_i(block, model_stokes_i)
    selected = _row_channel_mask(block, sample_mask)
    linear = selected & (planes.weight_linear > 0)
    circular_v = selected & (planes.weight_v > 0)
    if not np.any(linear) or not np.any(circular_v):
        raise ValueError("global q,u,v fit needs finite M_I and polarised samples")
    p_vis = planes.stokes_q + 1j * planes.stokes_u
    linear_weight = planes.weight_linear
    v_weight = planes.weight_v
    sums = _RegressionSums(
        linear_num=complex(np.sum(linear_weight[linear] * np.conj(model_i[linear]) * p_vis[linear])),
        linear_den=float(np.sum(linear_weight[linear] * np.abs(model_i[linear]) ** 2)),
        v_num=complex(np.sum(v_weight[circular_v] * np.conj(model_i[circular_v]) * planes.stokes_v[circular_v])),
        v_den=float(np.sum(v_weight[circular_v] * np.abs(model_i[circular_v]) ** 2)),
        null_linear_loss=float(np.sum(linear_weight[linear] * np.abs(p_vis[linear]) ** 2)),
        null_v_loss=float(np.sum(v_weight[circular_v] * np.abs(planes.stokes_v[circular_v]) ** 2)),
        n_samples=int(np.count_nonzero(linear)),
    )
    return sums, model_i, planes, linear, circular_v


def _fit_from_sums(
    sums: _RegressionSums,
    *,
    polarized_linear_loss: float,
    polarized_v_loss: float,
) -> GlobalFractionalPolarization:
    if sums.linear_den <= 0.0 or sums.v_den <= 0.0:
        raise ValueError("global q,u,v fit has a zero M_I normaliser")
    p_hat = sums.linear_num / sums.linear_den
    q = float(np.real(p_hat))
    u = float(np.imag(p_hat))
    v = float(np.real(sums.v_num / sums.v_den))
    return GlobalFractionalPolarization(
        q=q,
        u=u,
        v=v,
        fractional_linear=float(np.hypot(q, u)),
        casaguide_angle_rad=float(np.arctan2(u, q)),
        evpa_rad=float(electric_vector_position_angle_rad(q, u)),
        n_samples=sums.n_samples,
        null_linear_loss=sums.null_linear_loss,
        polarized_linear_loss=polarized_linear_loss,
        null_v_loss=sums.null_v_loss,
        polarized_v_loss=polarized_v_loss,
        provenance={
            "kind": "global_fractional_polarization",
            "regressor": "complex_stokes_i_model",
            "spatial_image": False,
            "frequency_model": "constant",
            "rm": False,
        },
    )


def _polarized_losses(
    model_i: np.ndarray,
    planes: StokesVisibilityPlanes,
    linear: np.ndarray,
    circular_v: np.ndarray,
    q: float,
    u: float,
    v: float,
) -> tuple[float, float]:
    p_hat = q + 1j * u
    p_vis = planes.stokes_q + 1j * planes.stokes_u
    linear_loss = float(
        np.sum(
            planes.weight_linear[linear]
            * np.abs(p_vis[linear] - p_hat * model_i[linear]) ** 2
        )
    )
    v_loss = float(
        np.sum(
            planes.weight_v[circular_v]
            * np.abs(planes.stokes_v[circular_v] - v * model_i[circular_v]) ** 2
        )
    )
    return linear_loss, v_loss


def fit_global_fractional_polarization(
    block: VisibilityBlock,
    *,
    model_stokes_i: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
) -> GlobalFractionalPolarization:
    """Fit one visibility-weighted ``q,u,v`` against frozen complex ``M_I``.

    The linear estimator is

    ``p̂ = Σ w M_I* (Q+iU) / Σ w |M_I|²``,

    with the analogous real coefficient for ``v``. This is a net-polarisation
    detection statistic, not a spatial polarisation image. Observed ``Re(I)``
    is refused as the regressor.
    """

    sums, model_i, planes, linear, circular_v = _block_regression_sums(
        block, model_stokes_i=model_stokes_i, sample_mask=sample_mask
    )
    draft = _fit_from_sums(sums, polarized_linear_loss=0.0, polarized_v_loss=0.0)
    linear_loss, v_loss = _polarized_losses(
        model_i, planes, linear, circular_v, draft.q, draft.u, draft.v
    )
    return _fit_from_sums(
        sums, polarized_linear_loss=linear_loss, polarized_v_loss=v_loss
    )


def fit_global_fractional_polarization_blocks(
    blocks: tuple[VisibilityBlock, ...],
    *,
    sample_masks: tuple[np.ndarray | None, ...] | None = None,
) -> GlobalFractionalPolarization:
    """Joint complex ``q,u,v`` regression across several visibility blocks."""

    if not blocks:
        raise ValueError("at least one visibility block is required")
    masks = sample_masks if sample_masks is not None else (None,) * len(blocks)
    if len(masks) != len(blocks):
        raise ValueError("sample_masks must contain one array per block")
    prepared: list[
        tuple[_RegressionSums, NDArray[np.complex128], StokesVisibilityPlanes, np.ndarray, np.ndarray]
    ] = []
    total = _RegressionSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    for block, mask in zip(blocks, masks, strict=True):
        sums, model_i, planes, linear, circular_v = _block_regression_sums(
            block, sample_mask=mask
        )
        prepared.append((sums, model_i, planes, linear, circular_v))
        total = _RegressionSums(
            linear_num=total.linear_num + sums.linear_num,
            linear_den=total.linear_den + sums.linear_den,
            v_num=total.v_num + sums.v_num,
            v_den=total.v_den + sums.v_den,
            null_linear_loss=total.null_linear_loss + sums.null_linear_loss,
            null_v_loss=total.null_v_loss + sums.null_v_loss,
            n_samples=total.n_samples + sums.n_samples,
        )
    draft = _fit_from_sums(total, polarized_linear_loss=0.0, polarized_v_loss=0.0)
    linear_loss = 0.0
    v_loss = 0.0
    for _, model_i, planes, linear, circular_v in prepared:
        block_linear, block_v = _polarized_losses(
            model_i, planes, linear, circular_v, draft.q, draft.u, draft.v
        )
        linear_loss += block_linear
        v_loss += block_v
    return _fit_from_sums(
        total, polarized_linear_loss=linear_loss, polarized_v_loss=v_loss
    )


def evaluate_global_fractional_polarization(
    block: VisibilityBlock,
    q: float,
    u: float,
    v: float,
    *,
    model_stokes_i: np.ndarray | None = None,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a frozen ``q,u,v`` on one block without refitting."""

    sums, model_i, planes, linear, circular_v = _block_regression_sums(
        block, model_stokes_i=model_stokes_i, sample_mask=sample_mask
    )
    linear_loss, v_loss = _polarized_losses(
        model_i, planes, linear, circular_v, q, u, v
    )
    return {
        "null_linear_loss": sums.null_linear_loss,
        "polarized_linear_loss": linear_loss,
        "null_v_loss": sums.null_v_loss,
        "polarized_v_loss": v_loss,
        "n_samples": float(sums.n_samples),
    }


def deterministic_visibility_partitions(
    block: VisibilityBlock,
) -> dict[str, NDArray[np.bool_]]:
    """Even/odd baseline, time, and channel masks with no RNG."""

    first = np.minimum(block.antenna1, block.antenna2)
    second = np.maximum(block.antenna1, block.antenna2)
    pairs = tuple(sorted({(int(a), int(b)) for a, b in zip(first, second, strict=True)}))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    baseline_parity = np.asarray(
        [pair_index[(int(a), int(b))] % 2 for a, b in zip(first, second, strict=True)],
        dtype=np.int32,
    )
    times = np.unique(block.time_s)
    time_index = {float(time): index for index, time in enumerate(times)}
    time_parity = np.asarray(
        [time_index[float(time)] % 2 for time in block.time_s], dtype=np.int32
    )
    channel_parity = np.arange(block.frequency_hz.size, dtype=np.int32) % 2
    baseline_even = (baseline_parity == 0)[:, None, None] & block.active
    baseline_odd = (baseline_parity == 1)[:, None, None] & block.active
    time_even = (time_parity == 0)[:, None, None] & block.active
    time_odd = (time_parity == 1)[:, None, None] & block.active
    channel_even = (channel_parity == 0)[None, :, None] & block.active
    channel_odd = (channel_parity == 1)[None, :, None] & block.active
    return {
        "baseline_even": baseline_even,
        "baseline_odd": baseline_odd,
        "time_even": time_even,
        "time_odd": time_odd,
        "channel_even": channel_even,
        "channel_odd": channel_odd,
    }


def fit_partitioned_global_polarization(
    blocks: tuple[VisibilityBlock, ...],
) -> dict[str, GlobalFractionalPolarization]:
    """Repeat the complex global fit on deterministic baseline/time/channel halves."""

    if not blocks:
        raise ValueError("at least one visibility block is required")
    partitions = tuple(deterministic_visibility_partitions(block) for block in blocks)
    fitted: dict[str, GlobalFractionalPolarization] = {}
    for name in PARTITION_NAMES:
        masks = tuple(partition[name] for partition in partitions)
        if not any(np.any(mask) for mask in masks):
            raise ValueError(f"partition {name} has no active samples")
        fitted[name] = fit_global_fractional_polarization_blocks(
            blocks, sample_masks=masks
        )
    return fitted


def deterministic_calibrator_cohort_split(block: VisibilityBlock) -> VisibilitySplit:
    """Hold out the last scan, or the last unique-time cohort if scans are one."""

    row_active = np.any(block.active, axis=(1, 2))
    if block.scan_id is not None:
        scans = np.unique(block.scan_id[row_active])
        if scans.size >= 2:
            held_rows = row_active & (block.scan_id == scans[-1])
            train_rows = row_active & ~held_rows
            if np.any(train_rows) and np.any(held_rows):
                return VisibilitySplit(
                    np.broadcast_to(train_rows[:, None, None], block.shape) & block.active,
                    np.broadcast_to(held_rows[:, None, None], block.shape) & block.active,
                    "scan_holdout_last",
                )
    times = np.unique(block.time_s[row_active])
    if times.size < 2:
        raise ValueError("calibrator cohort split needs at least two scans or times")
    hold_count = max(1, times.size // 5)
    held_times = set(float(time) for time in times[-hold_count:])
    held_rows = row_active & np.isin(block.time_s, np.asarray(sorted(held_times)))
    train_rows = row_active & ~held_rows
    if not np.any(train_rows) or not np.any(held_rows):
        raise ValueError("calibrator cohort split produced an empty train or holdout")
    return VisibilitySplit(
        np.broadcast_to(train_rows[:, None, None], block.shape) & block.active,
        np.broadcast_to(held_rows[:, None, None], block.shape) & block.active,
        "time_holdout_last_cohort",
    )


def outer_image_robust_scale(
    image: np.ndarray, *, inner_fraction: float = 0.4
) -> float:
    """MAD-based scale of pixels outside a central radius."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("outer robust scale expects a square image")
    size = values.shape[0]
    centre = size / 2.0
    yy, xx = np.ogrid[:size, :size]
    outer = np.hypot(yy + 0.5 - centre, xx + 0.5 - centre) >= inner_fraction * (size / 2.0)
    selected = values[outer]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("outer image annulus has no finite pixels")
    return float(1.4826 * np.median(np.abs(selected - np.median(selected))))


def map_image_agreement(
    first: np.ndarray, second: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    left = np.asarray(first, dtype=np.float64)[valid]
    right = np.asarray(second, dtype=np.float64)[valid]
    if left.size < 8:
        raise ValueError("agreement test needs at least eight shared finite pixels")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
    return {
        "cosine": cosine,
        "pearson": pearson,
        "n_pixels": float(left.size),
        "peak_offset_pixels": float(
            np.hypot(
                *(
                    np.subtract(
                        np.unravel_index(int(np.argmax(np.abs(first))), first.shape),
                        np.unravel_index(int(np.argmax(np.abs(second))), second.shape),
                    )
                )
            )
        ),
    }


def _mosaic_sky_coordinates(
    grid: RegularGrid, mosaic_phase_centre_rad: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    global_l, global_m = grid.coordinates
    ra, dec = lmn_to_radec(
        mosaic_phase_centre_rad[0],
        mosaic_phase_centre_rad[1],
        global_l,
        global_m,
    )
    return ra, dec


def _pointing_dirty_plane(
    block: VisibilityBlock,
    visibility: np.ndarray,
    weight: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    primary_beam: VLAPrimaryBeam | None,
    config: DirectDFTConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    local_l, local_m, _ = radec_to_lmn(
        block.phase_centre_rad[0],
        block.phase_centre_rad[1],
        ra,
        dec,
    )
    beam_i, _, _ = predict_beam_weights(
        primary_beam,
        local_l,
        local_m,
        block.frequency_hz,
    )
    beam = (
        np.ones((local_l.size, block.frequency_hz.size), dtype=np.float64)
        if beam_i is None
        else np.asarray(beam_i, dtype=np.float64)
    )
    numerator = np.zeros(local_l.size, dtype=np.float64)
    sensitivity = np.zeros(local_l.size, dtype=np.float64)
    weighted_visibility = weight * visibility
    pointing_normalization = float(np.sum(weight))
    if pointing_normalization <= 0:
        raise ValueError("pointing has no positive-weight polarised samples")
    for channel, frequency in enumerate(block.frequency_hz):
        uvw_wavelengths = block.uvw_m * (frequency / SPEED_OF_LIGHT_M_S)
        adjoint = np.asarray(
            direct_scalar_adjoint(
                weighted_visibility[:, channel],
                local_l,
                local_m,
                uvw_wavelengths,
                config=config,
            ),
            dtype=np.float64,
        )
        numerator += beam[:, channel] * adjoint
        channel_weight = float(np.sum(weight[:, channel]))
        sensitivity += channel_weight * np.square(beam[:, channel])
    return numerator, sensitivity, pointing_normalization


def dirty_mosaic_stokes_images(
    blocks: tuple[VisibilityBlock, ...],
    grid: RegularGrid,
    mosaic_phase_centre_rad: tuple[float, float],
    *,
    labels: tuple[str, ...] | None = None,
    primary_beam: VLAPrimaryBeam | None = None,
    config: DirectDFTConfig | None = None,
    minimum_sensitivity_fraction: float = 0.1,
) -> MosaicDirtyStokesImages:
    """Align dirty I, Q, U, V from several pointings on one celestial grid."""

    if not blocks:
        raise ValueError("blocks must contain at least one visibility block")
    if labels is not None and len(labels) != len(blocks):
        raise ValueError("labels must contain one name per block")
    if not 0 <= minimum_sensitivity_fraction < 1:
        raise ValueError("minimum_sensitivity_fraction must be in [0, 1)")
    if primary_beam is not None and primary_beam.apply_squint:
        raise ValueError("joint Stokes imaging does not yet support beam squint")
    selected_config = config or DirectDFTConfig()
    selected_labels = labels or tuple(f"C{index}" for index in range(1, len(blocks) + 1))
    ra, dec = _mosaic_sky_coordinates(grid, mosaic_phase_centre_rad)
    shape = (grid.size, grid.size)
    planes = {
        "i": np.zeros(ra.size, dtype=np.float64),
        "q": np.zeros(ra.size, dtype=np.float64),
        "u": np.zeros(ra.size, dtype=np.float64),
        "v": np.zeros(ra.size, dtype=np.float64),
    }
    sensitivity = np.zeros(ra.size, dtype=np.float64)
    global_normalization = 0.0
    per_pointing: dict[str, dict[str, NDArray[np.float64]]] = {}

    for label, block in zip(selected_labels, blocks, strict=True):
        stokes = stokes_visibility_planes(block)
        pointing_images: dict[str, NDArray[np.float64]] = {}
        pointing_sensitivity = None
        pointing_norm = None
        for name, visibility, weight in (
            ("i", stokes.stokes_i, stokes.weight_i),
            ("q", stokes.stokes_q, stokes.weight_linear),
            ("u", stokes.stokes_u, stokes.weight_linear),
            ("v", stokes.stokes_v, stokes.weight_v),
        ):
            numerator, plane_sensitivity, normalization = _pointing_dirty_plane(
                block,
                visibility,
                weight,
                ra,
                dec,
                primary_beam=primary_beam,
                config=selected_config,
            )
            pointing_images[name] = (numerator / normalization).reshape(shape)
            if name == "i":
                pointing_sensitivity = plane_sensitivity
                pointing_norm = normalization
                sensitivity += plane_sensitivity
                global_normalization += normalization
            planes[name] += numerator
        assert pointing_sensitivity is not None and pointing_norm is not None
        per_pointing[label] = {
            "stokes_i": pointing_images["i"],
            "stokes_q": pointing_images["q"],
            "stokes_u": pointing_images["u"],
            "stokes_v": pointing_images["v"],
            "sensitivity": pointing_sensitivity.reshape(shape),
        }

    if global_normalization <= 0:
        raise ValueError("joint mosaic has no positive-weight samples")
    peak_sensitivity = float(np.max(sensitivity))
    if peak_sensitivity <= 0:
        raise ValueError("joint mosaic has zero beam sensitivity")
    fraction = sensitivity / peak_sensitivity
    valid = (sensitivity > 0) & (fraction >= minimum_sensitivity_fraction)
    joint = {
        name: (plane / global_normalization).reshape(shape) for name, plane in planes.items()
    }
    recurrence = _pointing_recurrence(per_pointing, valid.reshape(shape))
    return MosaicDirtyStokesImages(
        stokes_i=joint["i"],
        stokes_q=joint["q"],
        stokes_u=joint["u"],
        stokes_v=joint["v"],
        sensitivity=sensitivity.reshape(shape),
        sensitivity_fraction=fraction.reshape(shape),
        per_pointing=per_pointing,
        recurrence=recurrence,
        provenance={
            "kind": "exploratory_mosaic_dirty_stokes",
            "evidence_grade": False,
            "minimum_sensitivity_fraction": minimum_sensitivity_fraction,
            "labels": list(selected_labels),
        },
    )


def _pointing_recurrence(
    per_pointing: dict[str, dict[str, NDArray[np.float64]]],
    valid: np.ndarray,
) -> dict[str, Any]:
    labels = tuple(per_pointing)
    pairs: dict[str, Any] = {}
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1 :]:
            shared = (
                valid
                & np.isfinite(per_pointing[first]["stokes_q"])
                & np.isfinite(per_pointing[second]["stokes_q"])
            )
            pairs[f"{first}_{second}"] = {
                "q": map_image_agreement(
                    per_pointing[first]["stokes_q"],
                    per_pointing[second]["stokes_q"],
                    shared,
                ),
                "u": map_image_agreement(
                    per_pointing[first]["stokes_u"],
                    per_pointing[second]["stokes_u"],
                    shared,
                ),
                "v": map_image_agreement(
                    per_pointing[first]["stokes_v"],
                    per_pointing[second]["stokes_v"],
                    shared,
                ),
            }
    return {"pairs": pairs, "n_pointings": len(labels)}


def polarization_floor_as_dict(floor: PolarizationFloor) -> dict[str, Any]:
    payload = asdict(floor)
    payload["casaguide_angle_deg"] = float(np.rad2deg(floor.casaguide_angle_rad))
    payload["evpa_deg"] = float(np.rad2deg(floor.evpa_rad))
    return payload


def global_fractional_polarization_as_dict(
    fitted: GlobalFractionalPolarization,
) -> dict[str, Any]:
    payload = asdict(fitted)
    payload["casaguide_angle_deg"] = float(np.rad2deg(fitted.casaguide_angle_rad))
    payload["evpa_deg"] = float(np.rad2deg(fitted.evpa_rad))
    payload["linear_loss_ratio"] = (
        fitted.polarized_linear_loss / fitted.null_linear_loss
        if fitted.null_linear_loss > 0
        else None
    )
    payload["v_loss_ratio"] = (
        fitted.polarized_v_loss / fitted.null_v_loss if fitted.null_v_loss > 0 else None
    )
    return payload
