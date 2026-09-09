"""Sealed C147-* offset-ring validation of high-resolution CASSBEAM.

Fields 1–8 are prediction data only. Offsets come from FIELD.PHASE_DIR
versus the reconstructed 3C147 sky position. The 128-member convention
ladder is not searched. Full Jones stays experimental.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.coordinates import radec_to_lmn
from sl1mjax.holography_beam_prior import (
    ComplexMoments,
    accumulate_hand_moments,
    bootstrap_complex_from_moments,
    flatten_row_channel,
    stack_moments,
)
from sl1mjax.holography_calibration import C147_OFFSET_FIELD_IDS
from sl1mjax.holography_commissioning import ON_AXIS_3C147_NAMES
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION, QU_NUISANCE_PAIRS
from sl1mjax.rime import SPEED_OF_LIGHT_M_S

C147_OFFSET_RING = "c147_offset_ring_highres_cassbeam"
SMOKE_CHANNEL = 32
NATIVE_SPW4_CHANNELS = 64
CHANNEL_HOLD_WIDTH = 16
SOURCE_DIRECTION_AGREE_ARCSEC = 1.0
MIN_NULL_CLUSTERS = 8
RR_LL_REGRESSION_REL = 0.02
SCALE_PHASE_AGREE_RAD = np.deg2rad(45.0)
SCALE_ABS_RATIO_MAX = 2.0
ANTENNA_POWER_SHARE_MAX = 0.70
DIAGONAL_RR_LL_CLOSURE_REL = 0.15
PLACEHOLDER_UPPER_LIMIT = 8.0

OFFSET_RING_NOTE = (
    "C147-* fields are unused prediction data. They do not enter K/B/G/"
    "Kcross/D/X solving. Offsets are reconstructed from MS metadata, not "
    "from field names."
)
LOCKED_CONVENTION_NOTE = (
    "This run uses the already validated mount-frame native CASSBEAM "
    "packing and parallactic convention. The previous 128-member ladder "
    "is not searched."
)
FULL_JONES_EXPERIMENTAL_NOTE = (
    "Full Jones remains experimental. A well-supported null or a real "
    "clustered upper limit is a valid completion. The production factory "
    "is not modified."
)
NO_HOLORASTER_NOTE = (
    "HOLORASTER visibilities are not used. No unconstrained spatial "
    "leakage, scan offsets, or extra temporal Jones terms are fitted."
)


@dataclass(frozen=True)
class FieldSkyRecord:
    """One FIELD table entry. The name is audit metadata, not an offset."""

    field_id: int
    name: str
    phase_centre_rad: tuple[float, float]
    code: str = ""


@dataclass(frozen=True)
class SourceSkyRecord:
    """One SOURCE table direction. Names are matched only for 3C147."""

    name: str
    direction_rad: tuple[float, float]
    field_id: int | None = None


@dataclass(frozen=True)
class FieldOffsetGeometry:
    """3C147 direction relative to one C147-* phase centre."""

    field_id: int
    name: str
    phase_centre_rad: tuple[float, float]
    source_radec_rad: tuple[float, float]
    l_rad: float
    m_rad: float
    n_rad: float
    radius_rad: float
    position_angle_rad: float
    source_from: str


@dataclass(frozen=True)
class FieldPartition:
    """Field-level train / inner / sealed split. One field, one partition."""

    training: tuple[int, ...]
    inner_holdout: tuple[int, ...]
    sealed_holdout: tuple[int, ...]
    sealed_score: float
    inner_score: float
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OffsetRingUpperLimit:
    """Held-out 95% upper limit on a CASSBEAM-shaped off-diagonal term."""

    status: str
    alpha_abs_hat: float
    alpha_abs_ul95: float
    null_abs_95: float
    template_median_abs: float
    coherent_voltage_ul95: float
    n_clusters: int
    n_perm: int
    detected: bool
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status == "computed":
            required = (
                self.alpha_abs_hat,
                self.alpha_abs_ul95,
                self.null_abs_95,
                self.template_median_abs,
                self.coherent_voltage_ul95,
            )
            if any(not np.isfinite(value) or value < 0.0 for value in required):
                raise ValueError("computed upper limit must be finite and non-negative")
            if self.coherent_voltage_ul95 == PLACEHOLDER_UPPER_LIMIT and self.n_clusters < 1:
                raise ValueError("refusing a placeholder upper limit")


def select_3c147_sky_direction(
    fields: Sequence[FieldSkyRecord],
    sources: Sequence[SourceSkyRecord],
    *,
    agree_arcsec: float = SOURCE_DIRECTION_AGREE_ARCSEC,
) -> tuple[tuple[float, float], str]:
    """3C147 from on-axis field 0 and SOURCE. Offset-field names are ignored."""

    on_axis = [item for item in fields if int(item.field_id) == 0]
    if not on_axis:
        named = [item for item in fields if item.name in ON_AXIS_3C147_NAMES]
        if not named:
            raise ValueError("no on-axis 3C147 field; cannot reconstruct offsets")
        on_axis = named[:1]
    field_dir = on_axis[0].phase_centre_rad
    matches = [
        item
        for item in sources
        if item.name in ON_AXIS_3C147_NAMES or item.field_id in {0, on_axis[0].field_id}
    ]
    if not matches:
        return (float(field_dir[0]), float(field_dir[1])), "field_0_phase_dir"
    source_dir = matches[0].direction_rad
    sep = _angular_separation_arcsec(field_dir, source_dir)
    if sep > float(agree_arcsec):
        raise ValueError(
            f"FIELD 0 PHASE_DIR and SOURCE 3C147 disagree by {sep:.3f} arcsec"
        )
    return (float(source_dir[0]), float(source_dir[1])), "source_table_agrees_field_0"


def reconstruct_field_offsets(
    fields: Sequence[FieldSkyRecord],
    source_radec_rad: tuple[float, float],
    *,
    source_from: str,
    field_ids: Sequence[int] = C147_OFFSET_FIELD_IDS,
) -> tuple[FieldOffsetGeometry, ...]:
    """Exact 3C147 (l, m) relative to each field phase centre."""

    wanted = {int(item) for item in field_ids}
    by_id = {int(item.field_id): item for item in fields}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"FIELD table is missing offset field ids {missing}")
    source_ra, source_dec = float(source_radec_rad[0]), float(source_radec_rad[1])
    out: list[FieldOffsetGeometry] = []
    for field_id in field_ids:
        record = by_id[int(field_id)]
        ra0, dec0 = float(record.phase_centre_rad[0]), float(record.phase_centre_rad[1])
        l_rad, m_rad, n_rad = radec_to_lmn(ra0, dec0, source_ra, source_dec)
        l_val = float(np.asarray(l_rad).reshape(-1)[0])
        m_val = float(np.asarray(m_rad).reshape(-1)[0])
        n_val = float(np.asarray(n_rad).reshape(-1)[0])
        out.append(
            FieldOffsetGeometry(
                field_id=int(field_id),
                name=record.name,
                phase_centre_rad=(ra0, dec0),
                source_radec_rad=(source_ra, source_dec),
                l_rad=l_val,
                m_rad=m_val,
                n_rad=n_val,
                radius_rad=float(np.hypot(l_val, m_val)),
                position_angle_rad=float(np.arctan2(l_val, m_val)),
                source_from=source_from,
            )
        )
    return tuple(out)


def declare_field_partitions(
    geometries: Sequence[FieldOffsetGeometry],
) -> FieldPartition:
    """Split fields from reconstructed geometry before looking at visibilities."""

    items = tuple(geometries)
    if len(items) != 8:
        raise ValueError("offset-ring partitions require all eight C147-* fields")
    sealed = _best_opposite_pair(items)
    remaining = [item for item in items if item.field_id not in sealed]
    inner = _best_orthogonal_opposite_pair(remaining, items, sealed)
    used = set(sealed) | set(inner)
    training = tuple(sorted(item.field_id for item in items if item.field_id not in used))
    if len(training) != 4 or len(set(training + inner + sealed)) != 8:
        raise ValueError("field partitions must be a disjoint cover of fields 1–8")
    return FieldPartition(
        training=training,
        inner_holdout=inner,
        sealed_holdout=sealed,
        sealed_score=_pair_opposite_score(items, sealed),
        inner_score=_pair_opposite_score(items, inner),
        notes=(
            "Partitions use reconstructed (l, m) only",
            "Sealed pair is the most opposite similar-radius pair",
            "Inner pair is the remaining opposite pair most orthogonal to the sealed axis",
            "Every row of a field stays in exactly one partition",
        ),
    )


def field_partition_masks(
    field_id: ArrayLike,
    partition: FieldPartition,
) -> dict[str, NDArray[np.bool_]]:
    """Row masks. Every field is assigned to exactly one partition."""

    ids = np.asarray(field_id, dtype=np.int32).reshape(-1)
    train = np.isin(ids, np.asarray(partition.training, dtype=np.int32))
    inner = np.isin(ids, np.asarray(partition.inner_holdout, dtype=np.int32))
    sealed = np.isin(ids, np.asarray(partition.sealed_holdout, dtype=np.int32))
    assigned = train.astype(np.int8) + inner.astype(np.int8) + sealed.astype(np.int8)
    if bool(np.any(assigned > 1)):
        raise ValueError("contaminated split: a row belongs to more than one partition")
    wanted = set(C147_OFFSET_FIELD_IDS)
    present = set(int(value) for value in np.unique(ids))
    if present - wanted:
        extra = sorted(present - wanted)
        raise ValueError(f"non-offset fields present in prediction block: {extra}")
    return {
        "training": train,
        "inner_holdout": inner,
        "sealed_holdout": sealed,
        "decision": inner,
        "scored_holdout": inner | sealed,
    }


def source_relative_offsets(
    sky_lm_rad: ArrayLike,
    pointing_delta_lm_rad: ArrayLike,
) -> NDArray[np.float64]:
    """Per-sample beam-frame offsets: ``l_d - Δ`` in the commanded-pointing sign."""

    sky = np.asarray(sky_lm_rad, dtype=np.float64)
    delta = np.asarray(pointing_delta_lm_rad, dtype=np.float64)
    if sky.shape != delta.shape or sky.ndim != 2 or sky.shape[-1] != 2:
        raise ValueError("sky and pointing arrays must have shape (sample, 2)")
    # commanded_pointing is element-wise l_d - Δ, not an antenna×direction grid.
    return sky - delta


def geometric_fringe_phase(
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """CASA geometric phase ``exp(+2πi [u l + v m + w(n-1)])`` for every channel."""

    uvw = np.asarray(uvw_m, dtype=np.float64)
    freq = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    lm = np.asarray(lm_rad, dtype=np.float64)
    if uvw.ndim != 2 or uvw.shape[-1] != 3:
        raise ValueError("uvw_m must have shape (row, 3)")
    if lm.shape == (2,):
        lm = np.broadcast_to(lm, (uvw.shape[0], 2))
    if lm.shape != (uvw.shape[0], 2):
        raise ValueError("lm_rad must have shape (2,) or (row, 2)")
    l_rad = lm[:, 0]
    m_rad = lm[:, 1]
    n_rad = np.sqrt(np.maximum(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad))
    waves = uvw[:, :, None] * (freq[None, None, :] / SPEED_OF_LIGHT_M_S)
    phase = (
        waves[:, 0, :] * l_rad[:, None]
        + waves[:, 1, :] * m_rad[:, None]
        + waves[:, 2, :] * (n_rad[:, None] - 1.0)
    )
    return np.exp(2j * np.pi * phase)


def apply_geometric_fringe(
    visibilities: ArrayLike,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Multiply a dual-antenna Jones prediction by the source fringe."""

    vis = np.asarray(visibilities, dtype=np.complex128)
    phase = geometric_fringe_phase(uvw_m, frequency_hz, lm_rad)
    if vis.ndim == 3:
        return vis * phase[:, 0, None, None]
    if vis.ndim != 4:
        raise ValueError("visibilities must have shape (row, [channel,] 2, 2)")
    return vis * phase[..., None, None]


def predict_dual_antenna_numpy(
    residual_p: ArrayLike,
    beam_p: ArrayLike,
    source: ArrayLike,
    beam_q: ArrayLike,
    residual_q: ArrayLike,
) -> NDArray[np.complex128]:
    """Vectorized :math:`V=R_p E_p S E_q^H R_q^H` without a Python row loop."""

    e_p = _as_row_channel_jones(beam_p)
    e_q = _as_row_channel_jones(beam_q)
    n_row, n_chan = int(e_p.shape[0]), int(e_p.shape[1])
    if e_q.shape != e_p.shape:
        raise ValueError("antenna beams must share shape (row, channel, 2, 2)")
    sky = _broadcast_jones(source, n_row, n_chan)
    r_p = _broadcast_jones(residual_p, n_row, n_chan)
    r_q = _broadcast_jones(residual_q, n_row, n_chan)
    right = np.conjugate(np.swapaxes(e_q, -1, -2)) @ np.conjugate(np.swapaxes(r_q, -1, -2))
    return r_p @ e_p @ sky @ right


def predict_offset_field_visibilities(
    residual_p: ArrayLike,
    beam_p: ArrayLike,
    source: ArrayLike,
    beam_q: ArrayLike,
    residual_q: ArrayLike,
    *,
    uvw_m: ArrayLike,
    frequency_hz: ArrayLike,
    sky_lm_rad: ArrayLike,
) -> NDArray[np.complex128]:
    """Offset-field RIME including the CASA geometric fringe.

    HOLORASTER moving–reference rows have the source at the phase centre,
    so the fringe is identically one. C147-* fields do not.
    """

    return apply_geometric_fringe(
        predict_dual_antenna_numpy(residual_p, beam_p, source, beam_q, residual_q),
        uvw_m,
        frequency_hz,
        sky_lm_rad,
    )


def ring_azimuth_rad(offset_lm_rad: ArrayLike) -> NDArray[np.float64]:
    """Position angle ``atan2(l, m)`` used by the offset-ring partitions."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64)
    if offset.ndim == 1:
        offset = offset.reshape(1, 2)
    return np.arctan2(offset[:, 0], offset[:, 1])


def ns_ew_masks(offset_lm_rad: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Split a ring into the sealed N–S axis and the inner E–W axis."""

    pa = ring_azimuth_rad(offset_lm_rad)
    wrapped = np.abs((pa + np.pi) % np.pi - 0.5 * np.pi)
    east_west = wrapped <= 0.25 * np.pi
    return {"east_west": east_west, "north_south": ~east_west}


def template_geometry_transforms(template: ArrayLike) -> dict[str, NDArray[np.complex128]]:
    """Low-order cross-hand transforms. Not a 128-member convention search."""

    t = np.asarray(template, dtype=np.complex128)
    swapped = np.swapaxes(t, -2, -1).copy()
    return {
        "native": t,
        "conjugate": np.conjugate(t),
        "swap_rl_lr": swapped,
        "conjugate_swap_rl_lr": np.conjugate(swapped),
    }


def azimuthal_phaser(offset_lm_rad: ArrayLike, order: int) -> NDArray[np.complex128]:
    """Multiply a template by ``exp(i k φ)`` without changing its morphology."""

    return np.exp(1j * int(order) * ring_azimuth_rad(offset_lm_rad))


def diagnose_directional_disagreement(
    measured: ArrayLike,
    predicted_full: ArrayLike,
    predicted_diag: ArrayLike,
    weight: ArrayLike,
    offset_lm_rad: ArrayLike,
) -> dict[str, object]:
    """Exploratory E–W versus N–S diagnosis on opened SPW-4 ring fields.

    Reports geometry transforms and one-to-two-cycle azimuthal phasers.
    Does not search the 128-member convention ladder or select a model.
    """

    from sl1mjax.holography_beam_prior import flatten_row_channel

    meas = np.asarray(measured, dtype=np.complex128)
    full = np.asarray(predicted_full, dtype=np.complex128)
    diag = np.asarray(predicted_diag, dtype=np.complex128)
    wgt = np.asarray(weight, dtype=np.float64)
    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    if meas.ndim == 4:
        n_chan = int(meas.shape[1])
        meas, _, _ = flatten_row_channel(meas)
        full, _, _ = flatten_row_channel(full)
        diag, _, _ = flatten_row_channel(diag)
        wgt, _, _ = flatten_row_channel(wgt)
        offset = np.repeat(offset, n_chan, axis=0)
    increment = full - diag
    residual = meas - diag
    masks = ns_ew_masks(offset)

    def _corr(template: NDArray[np.complex128], mask: NDArray[np.bool_]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, row, col in (("rl", 0, 1), ("lr", 1, 0)):
            t = template[mask, row, col]
            r = residual[mask, row, col]
            w = wgt[mask, row, col]
            finite = np.isfinite(t) & np.isfinite(r) & np.isfinite(w) & (w > 0.0)
            tt = float(np.sum(w[finite] * np.abs(t[finite]) ** 2))
            rr = float(np.sum(w[finite] * np.abs(r[finite]) ** 2))
            tr = complex(np.sum(w[finite] * np.conjugate(t[finite]) * r[finite]))
            denom = np.sqrt(tt * rr) if tt > 0.0 and rr > 0.0 else np.nan
            out[f"{name}_correlation"] = float(np.abs(tr) / denom) if np.isfinite(denom) else float("nan")
            out[f"{name}_alpha"] = complex(tr / tt) if tt > 0.0 else complex(np.nan, np.nan)
            out[f"{name}_n"] = int(np.sum(finite))
        return out

    by_axis = {name: _corr(increment, mask) for name, mask in masks.items()}
    transforms = {}
    for name, template in template_geometry_transforms(increment).items():
        transforms[name] = {
            axis: _corr(template, mask) for axis, mask in masks.items()
        }
    phasers = {}
    pa = azimuthal_phaser(offset, 1)
    for order in (-2, -1, 1, 2):
        phase = np.exp(1j * order * np.angle(pa))
        phased = increment * phase[:, None, None]
        phasers[str(order)] = {
            axis: _corr(phased, mask) for axis, mask in masks.items()
        }
    return {
        "native_by_axis": by_axis,
        "geometry_transforms": transforms,
        "azimuthal_phasers": phasers,
        "convention_search_reopened": False,
        "model_selected": False,
        "note": (
            "Exploratory diagnosis of the E–W versus N–S transfer failure. "
            "SPW 5 stays sealed."
        ),
    }


def cluster_ids_scan_baseline(
    scan_id: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
) -> NDArray[np.int64]:
    """Independent scan × baseline groups for clustered bootstrap."""

    scan = np.asarray(scan_id, dtype=np.int64).reshape(-1)
    first = np.asarray(antenna1, dtype=np.int64).reshape(-1)
    second = np.asarray(antenna2, dtype=np.int64).reshape(-1)
    lo = np.minimum(first, second)
    hi = np.maximum(first, second)
    return scan * 1_000_000 + lo * 1_000 + hi


def moments_for_increment(
    template: ArrayLike,
    residual: ArrayLike,
    weight: ArrayLike,
    *,
    cluster_ids: ArrayLike,
    channel_ids: ArrayLike,
    field_ids: ArrayLike,
    antenna1: ArrayLike,
    antenna2: ArrayLike,
    hands: tuple[str, ...] = ("rr", "rl", "lr", "ll"),
) -> list[ComplexMoments]:
    """Batch moments. Field and baseline identity travel with each group."""

    t_all = np.asarray(template)
    if t_all.ndim == 4:
        n_chan = int(t_all.shape[1])
        clusters = np.repeat(np.asarray(cluster_ids, dtype=np.int64).reshape(-1), n_chan)
        fields = np.repeat(np.asarray(field_ids, dtype=np.int64).reshape(-1), n_chan)
        ant1 = np.repeat(np.asarray(antenna1, dtype=np.int64).reshape(-1), n_chan)
        ant2 = np.repeat(np.asarray(antenna2, dtype=np.int64).reshape(-1), n_chan)
        _flat_t, _rows, chan = flatten_row_channel(t_all)
        flat_r, _, _ = flatten_row_channel(residual)
        flat_w, _, _ = flatten_row_channel(weight)
        channels = chan if channel_ids is None else np.asarray(channel_ids).reshape(-1)
        if channels.size != _flat_t.shape[0]:
            channels = chan
    else:
        clusters = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
        fields = np.asarray(field_ids, dtype=np.int64).reshape(-1)
        ant1 = np.asarray(antenna1, dtype=np.int64).reshape(-1)
        ant2 = np.asarray(antenna2, dtype=np.int64).reshape(-1)
        channels = np.asarray(channel_ids, dtype=np.int64).reshape(-1)
        _flat_t = t_all
        flat_r = residual
        flat_w = weight
    return accumulate_hand_moments(
        _flat_t,
        flat_r,
        flat_w,
        cluster_ids=clusters,
        channel_ids=channels,
        dwell_ids=fields,
        mover_ids=ant1,
        reference_ids=ant2,
        cell_ids=fields,
        hands=hands,
    )


def select_training_qu(
    scores: Mapping[tuple[float, float], float],
) -> tuple[float, float]:
    """Pick the training Q/U pair with the lowest declared score.

    The score must be a training-only cross-hand or joint four-hand
    residual. RR/LL-only loss does not see Q/U in the circular basis.
    """

    if not scores:
        return 0.0, 0.0
    return min(scores.items(), key=lambda item: (float(item[1]), item[0][0], item[0][1]))[0]


def filter_moments_by_channel(
    moments: Sequence[ComplexMoments],
    channel_mask: ArrayLike,
    *,
    allow_empty: bool = False,
) -> tuple[ComplexMoments, ...]:
    """Keep moments whose native channel is True in ``channel_mask``."""

    keep = np.asarray(channel_mask, dtype=bool).reshape(-1)
    selected = tuple(
        item
        for item in moments
        if 0 <= int(item.channel) < keep.size and bool(keep[int(item.channel)])
    )
    if not selected and not allow_empty:
        raise ValueError("channel mask removed every moment")
    return selected


def hand_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
    *,
    hands: Sequence[tuple[str, int, int]] = (
        ("rr", 0, 0),
        ("ll", 1, 1),
        ("rl", 0, 1),
        ("lr", 1, 0),
    ),
) -> dict[str, object]:
    """Weighted relative residual power on the requested visibility hands."""

    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    wgt = np.asarray(weight, dtype=np.float64)
    if vis.ndim == 4:
        vis, _, _ = flatten_row_channel(vis)
        pred, _, _ = flatten_row_channel(pred)
        wgt, _, _ = flatten_row_channel(wgt)
    out: dict[str, object] = {}
    total = 0.0
    finite_total = True
    for name, row, col in hands:
        obs = vis[:, row, col]
        hat = pred[:, row, col]
        ww = wgt[:, row, col]
        finite = np.isfinite(obs) & np.isfinite(hat) & np.isfinite(ww) & (ww > 0.0)
        denom = float(np.sum(ww[finite] * np.abs(obs[finite]) ** 2))
        numer = float(np.sum(ww[finite] * np.abs(obs[finite] - hat[finite]) ** 2))
        rel = numer / denom if denom > 0.0 else float("nan")
        out[name] = {"relative_power": rel, "n": int(np.sum(finite))}
        if np.isfinite(rel):
            total += rel
        else:
            finite_total = False
    out["total"] = total if finite_total else float("nan")
    return out


def four_hand_residual_power(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> dict[str, object]:
    """Joint RR/LL/RL/LR residual used to select the Q/U nuisance."""

    return hand_residual_power(measured, predicted, weight)


def apply_channel_holdout(
    values: ArrayLike,
    channel_train: ArrayLike,
) -> NDArray:
    """Drop reserved holdout channels from a row–channel cube before any fit."""

    array = np.asarray(values)
    train = np.asarray(channel_train, dtype=bool).reshape(-1)
    if array.ndim < 2 or array.shape[1] != train.size:
        raise ValueError("channel holdout mask must match axis 1 of the cube")
    return array[:, train]


def qu_nuisance_pairs() -> tuple[tuple[float, float], ...]:
    return QU_NUISANCE_PAIRS


def locked_convention():
    return DEFAULT_CONVENTION


def refuse_convention_search(conventions: Sequence[object] | None) -> None:
    if conventions is not None and len(tuple(conventions)) > 1:
        raise ValueError("offset-ring validation does not search the convention ladder")


def clustered_null_upper_limit(
    moments: Sequence[ComplexMoments],
    *,
    template_median_abs: float,
    n_perm: int = 400,
    seed: int = 11,
) -> OffsetRingUpperLimit:
    """Radial 95% UL on |α| from held-out likelihood and a clustered null."""

    stacked = stack_moments(moments)
    labels = np.asarray([item.cluster_id for item in moments], dtype=np.int64)
    tt = np.asarray([item.tt for item in moments], dtype=np.float64)
    tr = np.asarray([item.tr for item in moments], dtype=np.complex128)
    n_clusters = int(np.unique(labels).size) if labels.size else 0
    if stacked.n <= 0 or stacked.tt <= 0.0 or n_clusters < MIN_NULL_CLUSTERS:
        raise ValueError(
            "held-out moments are insufficient for a real upper limit; "
            "refusing a placeholder"
        )
    if not np.isfinite(template_median_abs) or template_median_abs <= 0.0:
        raise ValueError("template_median_abs must be a positive finite visibility amplitude")
    alpha_hat = stacked.alpha
    abs_hat = float(np.abs(alpha_hat))
    rng = np.random.default_rng(int(seed))
    unique = np.unique(labels)
    order = np.searchsorted(unique, labels)
    null_abs = np.empty(int(n_perm), dtype=np.float64)
    denom = float(np.sum(tt))
    for i in range(int(n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=unique.size)
        flipped = tr * signs[order]
        draw = (np.sum(flipped) / denom) if denom > 0.0 else np.nan + 1j * np.nan
        null_abs[i] = float(np.abs(draw))
    finite_null = null_abs[np.isfinite(null_abs)]
    if finite_null.size < 20:
        raise ValueError("clustered null distribution is empty; refusing a placeholder")
    boot = bootstrap_complex_from_moments(moments, n_boot=int(n_perm), seed=int(seed) + 1)
    boot_ul = float(boot["abs_ul95"])
    null_95 = float(np.percentile(finite_null, 95))
    ul_alpha = float(max(abs_hat, boot_ul, null_95))
    detected = bool(abs_hat > null_95 and boot.get("consistent_with_zero") is False)
    return OffsetRingUpperLimit(
        status="computed",
        alpha_abs_hat=abs_hat,
        alpha_abs_ul95=ul_alpha,
        null_abs_95=null_95,
        template_median_abs=float(template_median_abs),
        coherent_voltage_ul95=float(ul_alpha * template_median_abs),
        n_clusters=n_clusters,
        n_perm=int(finite_null.size),
        detected=detected,
        notes=(
            "Upper limit uses held-out α = Σw t* r / Σw |t|²",
            "Null flips the sign of each scan×baseline cluster's matched filter",
            "α_ul95 is the clustered bootstrap 95th percentile of |α|",
            "coherent_voltage_ul95 = α_ul95 × median |t| on RL/LR",
        ),
    )


def refuse_placeholder_upper_limit(limit: OffsetRingUpperLimit | Mapping[str, object]) -> None:
    if isinstance(limit, OffsetRingUpperLimit):
        status = limit.status
        voltage = limit.coherent_voltage_ul95
        n_clusters = limit.n_clusters
    else:
        status = str(limit.get("status", ""))
        voltage = float(limit.get("coherent_voltage_ul95", float("nan")))
        n_clusters = int(limit.get("n_clusters", 0) or 0)
    if status != "computed":
        raise ValueError("upper limit was not computed from held-out moments")
    if not np.isfinite(voltage) or voltage <= 0.0:
        raise ValueError("upper limit must be a positive finite voltage amplitude")
    if n_clusters < MIN_NULL_CLUSTERS:
        raise ValueError("upper limit used too few clusters")
    if voltage == PLACEHOLDER_UPPER_LIMIT and n_clusters < 1:
        raise ValueError("refusing a placeholder upper limit")


def paired_loss_by_group(
    moments: Sequence[ComplexMoments],
    alpha_full: complex,
    *,
    group: str,
) -> dict[str, dict[str, float]]:
    """Full-minus-diagonal residual power by a moment metadata axis."""

    attr = {
        "field": "dwell",
        "channel": "channel",
        "baseline_antenna1": "mover",
        "baseline_antenna2": "reference",
        "cluster": "cluster_id",
        "hand_cell": "cell",
    }[group]
    grouped: dict[int, list[ComplexMoments]] = {}
    for item in moments:
        grouped.setdefault(int(getattr(item, attr)), []).append(item)
    out: dict[str, dict[str, float]] = {}
    for key, items in grouped.items():
        stacked = stack_moments(items)
        power_full = stacked.residual_power(complex(alpha_full))
        power_diag = stacked.residual_power(0.0 + 0.0j)
        out[str(key)] = {
            "n": float(stacked.n),
            "power_full": float(power_full),
            "power_diag": float(power_diag),
            "delta": float(power_full - power_diag),
        }
    return out


def scale_compatible(
    alphas: Sequence[complex],
    *,
    phase_tol_rad: float = SCALE_PHASE_AGREE_RAD,
    abs_ratio_max: float = SCALE_ABS_RATIO_MAX,
) -> dict[str, object]:
    """Whether fitted complex scales agree across fields or channel blocks."""

    values = [complex(item) for item in alphas if np.isfinite(complex(item).real)]
    if len(values) < 2:
        return {"compatible": False, "n": len(values), "reason": "too_few_estimates"}
    mags = np.asarray([abs(item) for item in values], dtype=np.float64)
    median = float(np.median(mags))
    if median <= 0.0:
        return {"compatible": bool(np.all(mags == 0.0)), "n": len(values), "median_abs": 0.0}
    ratio_ok = bool(np.all((mags / median <= abs_ratio_max) & (median / np.maximum(mags, 1e-15) <= abs_ratio_max)))
    phases = np.asarray([np.angle(item) for item in values], dtype=np.float64)
    phase_ok = True
    for i, left in enumerate(phases):
        for right in phases[i + 1 :]:
            if abs(_wrap_pi(left - right)) > float(phase_tol_rad):
                phase_ok = False
    return {
        "compatible": bool(ratio_ok and phase_ok),
        "n": len(values),
        "median_abs": median,
        "max_abs": float(np.max(mags)),
        "min_abs": float(np.min(mags)),
        "ratio_ok": ratio_ok,
        "phase_ok": phase_ok,
    }


def antenna_power_share(moments: Sequence[ComplexMoments]) -> dict[str, object]:
    """Detect a result that lives in a handful of antennas."""

    by_ant: dict[int, float] = {}
    total = 0.0
    for item in moments:
        by_ant[int(item.mover)] = by_ant.get(int(item.mover), 0.0) + float(item.tt)
        by_ant[int(item.reference)] = by_ant.get(int(item.reference), 0.0) + float(item.tt)
        total += float(item.tt)
    if total <= 0.0:
        return {"ok": False, "reason": "no_template_power", "top_share": float("nan")}
    ranked = sorted(by_ant.items(), key=lambda item: item[1], reverse=True)
    top_two = sum(weight for _ant, weight in ranked[:2]) / total
    return {
        "ok": bool(top_two <= ANTENNA_POWER_SHARE_MAX),
        "top_share": float(top_two),
        "n_antennas": len(by_ant),
        "ranked": [(int(ant), float(weight / total)) for ant, weight in ranked[:6]],
    }


def diagonal_rr_ll_closure(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> dict[str, object]:
    """Relative RR/LL residual power of the diagonal CASSBEAM prediction."""

    vis = np.asarray(measured, dtype=np.complex128)
    pred = np.asarray(predicted, dtype=np.complex128)
    wgt = np.asarray(weight, dtype=np.float64)
    if vis.ndim == 4:
        vis, _, _ = flatten_row_channel(vis)
        pred, _, _ = flatten_row_channel(pred)
        wgt, _, _ = flatten_row_channel(wgt)
    out: dict[str, object] = {}
    passed = True
    for name, row, col in (("rr", 0, 0), ("ll", 1, 1)):
        m = vis[:, row, col]
        p = pred[:, row, col]
        w = wgt[:, row, col]
        finite = np.isfinite(m) & np.isfinite(p) & np.isfinite(w) & (w > 0.0)
        denom = float(np.sum(w[finite] * np.abs(m[finite]) ** 2))
        num = float(np.sum(w[finite] * np.abs(m[finite] - p[finite]) ** 2))
        rel = num / denom if denom > 0.0 else float("nan")
        hand_ok = bool(np.isfinite(rel) and rel <= DIAGONAL_RR_LL_CLOSURE_REL)
        passed = passed and hand_ok
        out[name] = {"relative_power": rel, "n": int(np.sum(finite)), "ok": hand_ok}
    out["passed"] = passed
    return out


def channel32_smoke_gates(
    *,
    geometry_ok: bool,
    calibration_hashed: bool,
    predictions_finite: bool,
    diagonal_closure_ok: bool,
) -> dict[str, object]:
    passed = bool(geometry_ok and calibration_hashed and predictions_finite and diagonal_closure_ok)
    return {
        "passed": passed,
        "geometry_ok": bool(geometry_ok),
        "calibration_hashed": bool(calibration_hashed),
        "predictions_finite": bool(predictions_finite),
        "diagonal_rr_ll_closure_ok": bool(diagonal_closure_ok),
        "continue_to_spw4": passed,
        "channel": SMOKE_CHANNEL,
    }


def classify_c147_offset_ring(
    *,
    software_ok: bool,
    split_clean: bool,
    smoke_ok: bool,
    rr_ll_regression: bool,
    inner_rl_improves: bool,
    inner_lr_improves: bool,
    sealed_rl_improves: bool,
    sealed_lr_improves: bool,
    scale_ok: bool,
    antenna_ok: bool,
    unit_better_than_diag: bool,
    scaled_better_than_unit: bool,
    upper_limit: OffsetRingUpperLimit | Mapping[str, object] | None,
) -> dict[str, object]:
    """Scientific outcomes are recorded. Process failures fail closed."""

    if not software_ok:
        decision = "software_gate_failed"
        process_failure = True
        selected = None
    elif not split_clean:
        decision = "contaminated_split"
        process_failure = True
        selected = None
    elif not smoke_ok:
        decision = "smoke_failed"
        process_failure = True
        selected = None
    else:
        process_failure = False
        cross_inner = bool(inner_rl_improves and inner_lr_improves)
        cross_sealed = bool(sealed_rl_improves and sealed_lr_improves)
        if (
            not rr_ll_regression
            and cross_inner
            and cross_sealed
            and scale_ok
            and antenna_ok
            and unit_better_than_diag
            and not scaled_better_than_unit
        ):
            decision = "unit_full_jones_supported"
            selected = "full_jones_unit"
        elif (
            not rr_ll_regression
            and cross_inner
            and cross_sealed
            and scale_ok
            and antenna_ok
            and scaled_better_than_unit
        ):
            decision = "scaled_full_jones_supported"
            selected = "full_jones_smooth_scale"
        else:
            decision = "no_model_selected"
            selected = None
            if upper_limit is None:
                raise ValueError(
                    "no model selected requires a computed clustered upper limit"
                )
            refuse_placeholder_upper_limit(upper_limit)
    selected_ok = selected is not None
    return {
        "gate": C147_OFFSET_RING,
        "decision": decision,
        "selected_model": selected,
        "process_failure": process_failure,
        "scientific_decision": not process_failure,
        "status": "pass" if selected_ok else "fail",
        "blocking": not selected_ok,
        "full_jones_frozen": False,
        "production_factory_modified": False,
        "spw5_closed": not selected_ok,
        "spw5_opened": False,
        "most_important_next_artifact": "cassbeam_diagonal_low_order_correction",
        "notes": (
            OFFSET_RING_NOTE,
            LOCKED_CONVENTION_NOTE,
            FULL_JONES_EXPERIMENTAL_NOTE,
            NO_HOLORASTER_NOTE,
        ),
    }


def partition_to_dict(partition: FieldPartition) -> dict[str, object]:
    return {
        "training": list(partition.training),
        "inner_holdout": list(partition.inner_holdout),
        "sealed_holdout": list(partition.sealed_holdout),
        "sealed_score": partition.sealed_score,
        "inner_score": partition.inner_score,
        "notes": list(partition.notes),
    }


def geometry_to_dict(item: FieldOffsetGeometry) -> dict[str, object]:
    return {
        "field_id": item.field_id,
        "name": item.name,
        "phase_centre_rad": list(item.phase_centre_rad),
        "source_radec_rad": list(item.source_radec_rad),
        "l_rad": item.l_rad,
        "m_rad": item.m_rad,
        "n_rad": item.n_rad,
        "radius_rad": item.radius_rad,
        "position_angle_rad": item.position_angle_rad,
        "radius_arcmin": float(np.rad2deg(item.radius_rad) * 60.0),
        "position_angle_deg": float(np.rad2deg(item.position_angle_rad)),
        "source_from": item.source_from,
    }


def upper_limit_to_dict(limit: OffsetRingUpperLimit) -> dict[str, object]:
    return {
        "status": limit.status,
        "alpha_abs_hat": limit.alpha_abs_hat,
        "alpha_abs_ul95": limit.alpha_abs_ul95,
        "null_abs_95": limit.null_abs_95,
        "template_median_abs": limit.template_median_abs,
        "coherent_voltage_ul95": limit.coherent_voltage_ul95,
        "n_clusters": limit.n_clusters,
        "n_perm": limit.n_perm,
        "detected": limit.detected,
        "notes": list(limit.notes),
    }


def write_offset_ring_plots(
    output_dir,
    *,
    geometries: Sequence[FieldOffsetGeometry],
    measured: ArrayLike,
    predicted_diag: ArrayLike,
    predicted_full: ArrayLike,
    field_id: ArrayLike,
    frequencies_hz: ArrayLike,
    bootstrap: Mapping[str, object],
) -> list[str]:
    """Measured versus predicted complex visibilities and residual geometry."""

    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    meas = np.asarray(measured, dtype=np.complex128)
    diag = np.asarray(predicted_diag, dtype=np.complex128)
    full = np.asarray(predicted_full, dtype=np.complex128)
    if meas.ndim == 4:
        meas = meas[:, 0]
        diag = diag[:, 0]
        full = full[:, 0]
    ids = np.asarray(field_id, dtype=np.int32).reshape(-1)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0))
    hands = (("RR", 0, 0), ("RL", 0, 1), ("LR", 1, 0), ("LL", 1, 1))
    for ax, (name, row, col) in zip(axes.ravel(), hands, strict=True):
        ax.scatter(meas[:, row, col].real, meas[:, row, col].imag, s=4, alpha=0.25, label="measured")
        ax.scatter(diag[:, row, col].real, diag[:, row, col].imag, s=4, alpha=0.25, label="diag")
        ax.scatter(full[:, row, col].real, full[:, row, col].imag, s=4, alpha=0.25, label="full")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = root / "measured_vs_predicted.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    by_id = {item.field_id: item for item in geometries}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for ident in np.unique(ids):
        geom = by_id.get(int(ident))
        if geom is None:
            continue
        mask = ids == ident
        residual = np.abs(meas[mask, 0, 1] - diag[mask, 0, 1])
        ax.scatter(
            np.full(residual.size, np.rad2deg(geom.radius_rad) * 60.0),
            residual,
            s=6,
            alpha=0.3,
            label=f"field {int(ident)}",
        )
    ax.set_xlabel("offset radius (arcmin)")
    ax.set_ylabel("|RL residual| vs diagonal")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = root / "residual_vs_offset.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    freqs = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if meas.ndim == 3 and freqs.size:
        ax.plot(freqs / 1e6, np.full(freqs.size, np.nanmedian(np.abs(meas[:, 0, 1]))))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("median |RL|")
    fig.tight_layout()
    path = root / "residual_vs_frequency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    real_ci = bootstrap.get("real_ci95") or (float("nan"), float("nan"))
    imag_ci = bootstrap.get("imag_ci95") or (float("nan"), float("nan"))
    ax.errorbar(
        [float(bootstrap.get("real", float("nan")))],
        [float(bootstrap.get("imag", float("nan")))],
        xerr=[[abs(float(bootstrap.get("real", 0.0)) - float(real_ci[0]))],
              [abs(float(real_ci[1]) - float(bootstrap.get("real", 0.0)))]],
        yerr=[[abs(float(bootstrap.get("imag", 0.0)) - float(imag_ci[0]))],
              [abs(float(imag_ci[1]) - float(bootstrap.get("imag", 0.0)))]],
        fmt="o",
    )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_xlabel("Re α")
    ax.set_ylabel("Im α")
    ax.set_title("clustered bootstrap")
    fig.tight_layout()
    path = root / "bootstrap_alpha.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))
    return written


def _as_row_channel_jones(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 3:
        return array[:, None, :, :]
    if array.ndim != 4 or array.shape[-2:] != (2, 2):
        raise ValueError("Jones cubes must have shape (row, [channel,] 2, 2)")
    return array


def _broadcast_jones(values: ArrayLike, n_row: int, n_chan: int) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape == (2, 2):
        return np.broadcast_to(array, (n_row, n_chan, 2, 2))
    if array.ndim == 3:
        array = array[:, None, :, :]
    if array.shape != (n_row, n_chan, 2, 2) and array.shape != (n_row, 1, 2, 2):
        if array.shape == (1, n_chan, 2, 2):
            return np.broadcast_to(array, (n_row, n_chan, 2, 2))
        if array.shape[0] == n_row and array.shape[-2:] == (2, 2) and array.shape[1] == 1:
            return np.broadcast_to(array, (n_row, n_chan, 2, 2))
        raise ValueError("cannot broadcast Jones to (row, channel, 2, 2)")
    if array.shape[1] == 1 and n_chan != 1:
        return np.broadcast_to(array, (n_row, n_chan, 2, 2))
    return array


def _angular_separation_arcsec(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    ra1, dec1 = float(first[0]), float(first[1])
    ra2, dec2 = float(second[0]), float(second[1])
    cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0)


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _pair_opposite_score(
    geometries: Sequence[FieldOffsetGeometry],
    pair: tuple[int, int],
) -> float:
    by_id = {item.field_id: item for item in geometries}
    left, right = by_id[pair[0]], by_id[pair[1]]
    angle = abs(_wrap_pi(left.position_angle_rad - right.position_angle_rad))
    mean_r = 0.5 * (left.radius_rad + right.radius_rad)
    radius = abs(left.radius_rad - right.radius_rad) / max(mean_r, 1.0e-12)
    return float(abs(angle - np.pi) + 0.25 * radius)


def _best_opposite_pair(geometries: Sequence[FieldOffsetGeometry]) -> tuple[int, int]:
    ids = [item.field_id for item in geometries]
    best: tuple[int, int] | None = None
    best_score = float("inf")
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            pair = (min(left, right), max(left, right))
            score = _pair_opposite_score(geometries, pair)
            if score < best_score - 1.0e-12 or (
                abs(score - best_score) <= 1.0e-12 and (best is None or pair < best)
            ):
                best = pair
                best_score = score
    if best is None:
        raise ValueError("need at least two fields to declare an opposite pair")
    return best


def _best_orthogonal_opposite_pair(
    remaining: Sequence[FieldOffsetGeometry],
    all_geometries: Sequence[FieldOffsetGeometry],
    sealed: tuple[int, int],
) -> tuple[int, int]:
    by_id = {item.field_id: item for item in all_geometries}
    axis = by_id[sealed[0]].position_angle_rad
    ids = [item.field_id for item in remaining]
    best: tuple[int, int] | None = None
    best_key: tuple[float, float, tuple[int, int]] | None = None
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            pair = (min(left, right), max(left, right))
            opposite = _pair_opposite_score(all_geometries, pair)
            ortho = abs(abs(_wrap_pi(by_id[left].position_angle_rad - axis)) - 0.5 * np.pi)
            key = (ortho, opposite, pair)
            if best_key is None or key < best_key:
                best = pair
                best_key = key
    if best is None:
        raise ValueError("inner holdout requires a remaining opposite pair")
    return best
