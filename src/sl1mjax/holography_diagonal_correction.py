"""Frozen SPW-4 holdouts and the first CASSBEAM diagonal-correction ladder.

The publication treats CASSBEAM as the qualified diagonal prior. This
module freezes the development split used to select a nested, boresight-
preserving amplitude correction. It does not fit coefficients, does not
open SPW 5, and does not change the production beam factory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography import (
    THOL0001_LOWER_C_NATIVE_HZ,
    THOL0001_SPW4_CHANNEL_32_HZ,
    HolographyObservation,
)
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_calibration import C147_OFFSET_FIELD_IDS
from sl1mjax.holography_diagonal import THOL0001_REFERENCE_ANTENNA_NAMES

CASSBEAM_DIAGONAL_CORRECTION = "cassbeam_diagonal_low_order_correction"
SPW4_CORRECTION_PROTOCOL_VERSION = 2
SPW4_TRAINING_FREQUENCY_HZ = THOL0001_SPW4_CHANNEL_32_HZ
SPW5_FREQUENCY_HZ = THOL0001_LOWER_C_NATIVE_HZ[1]
HOLORASTER_FIELD_ID = 10
ON_AXIS_FIELD_IDS = (0, 9)
OFFSET_CELL_QUANT_PER_RAD = 180.0 * 60.0 / np.pi * 100.0
PAIRED_DELTA_BOOTSTRAP = 400
MIN_MOVERS_IMPROVING = 4
MATERIAL_MAINLOBE_ABS = 0.002
MATERIAL_MAINLOBE_REL = 0.10
COPOLAR_HANDS = (("rr", 0, 0), ("ll", 1, 1))

THOL0001_MOVING_ANTENNA_NAMES = (
    "ea04",
    "ea05",
    "ea06",
    "ea08",
    "ea09",
    "ea10",
    "ea11",
    "ea13",
    "ea15",
    "ea16",
    "ea17",
    "ea18",
    "ea19",
    "ea20",
    "ea21",
    "ea22",
    "ea23",
    "ea25",
    "ea27",
    "ea28",
)
SPW4_HOLDOUT_MOVING_ANTENNA_NAMES = ("ea08", "ea13", "ea18", "ea22", "ea28")
SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES = ("ea24", "ea26")
SPW4_TRAINING_MOVING_ANTENNA_NAMES = tuple(
    name for name in THOL0001_MOVING_ANTENNA_NAMES if name not in SPW4_HOLDOUT_MOVING_ANTENNA_NAMES
)
SPW4_TRAINING_REFERENCE_ANTENNA_NAMES = tuple(
    name
    for name in THOL0001_REFERENCE_ANTENNA_NAMES
    if name not in SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES
)

FIRST_LADDER_TERMS = (
    "identity",
    "rl_squint_scale",
    "beam_width",
    "pointing_offset",
    "first_sidelobe_radius_amplitude",
    "low_order_azimuthal",
)
RANKING_AXES = ("spatial", "moving")
DIAGNOSTIC_AXES = ("reference", "spatial_moving")

PROTOCOL_NOTE = (
    "SPW-4 correction training is HOLORASTER field 10 at native channel 32. "
    "Selection uses the paired loss difference ΔL=L_candidate-L_baseline. "
    "Both ranking holdouts must have a 95% interval lying entirely below "
    "zero. Spatial clusters are raster cells; mover clusters are the five "
    "held-out antennas. Individual visibilities are not bootstrap units. "
    "Reference holdouts are diagnostic. C147-* and SPW 5 stay unused."
)
BORESIGHT_NOTE = (
    "C_R(0)=C_L(0)=1. The correction cannot absorb the absolute flux gauge."
)
PHASE_NOTE = (
    "The first ladder is copolar amplitude and squint only. Phase corrections "
    "are refused until amplitude terms transfer."
)
SPW5_NOTE = (
    "SPW 5 remains sealed until the correction family and selection rule are "
    "fixed on SPW 4. It opens once, as the frequency-transfer test."
)
PAIRED_DELTA_NOTE = (
    "The acceptance interval is the clustered bootstrap of the paired "
    "difference, not an unpaired candidate interval compared with the "
    "baseline point estimate."
)
MOVER_GATE_NOTE = (
    "With five held-out movers, report every mover's paired ΔL. Accept only "
    "if at least four improve and none has a material main-lobe regression."
)


@dataclass(frozen=True)
class HoldoutScore:
    """Diagnostic residual-power table. Not used to accept a correction."""

    axis: str
    residual_power: float
    residual_power_lo: float
    residual_power_hi: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("holdout score needs samples")
        values = (self.residual_power, self.residual_power_lo, self.residual_power_hi)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("holdout score must be finite")
        if (
            self.residual_power_lo > self.residual_power
            or self.residual_power > self.residual_power_hi
        ):
            raise ValueError("holdout interval must contain the point estimate")


@dataclass(frozen=True)
class PairedLossDifference:
    """Clustered bootstrap of :math:`ΔL=L_{candidate}-L_{baseline}`."""

    axis: str
    delta: float
    delta_lo: float
    delta_hi: float
    n_clusters: int
    n_boot: int
    cluster_kind: str

    def __post_init__(self) -> None:
        if self.n_clusters < 1:
            raise ValueError("paired difference needs clusters")
        values = (self.delta, self.delta_lo, self.delta_hi)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("paired difference must be finite")
        if self.delta_lo > self.delta or self.delta > self.delta_hi:
            raise ValueError("paired interval must contain the point estimate")
        if self.cluster_kind not in {"spatial_cell", "moving_antenna", "reference_antenna"}:
            raise ValueError(f"unsupported bootstrap unit {self.cluster_kind!r}")

    def improves(self) -> bool:
        return float(self.delta_hi) < 0.0


@dataclass(frozen=True)
class MoverPairedDeltas:
    """Per-mover paired ΔL for the five frozen holdout antennas."""

    names: tuple[str, ...]
    delta: NDArray[np.float64]
    mainlobe_delta: NDArray[np.float64]
    mainlobe_baseline: NDArray[np.float64]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        delta = np.asarray(self.delta, dtype=np.float64).reshape(-1)
        main_delta = np.asarray(self.mainlobe_delta, dtype=np.float64).reshape(-1)
        main_base = np.asarray(self.mainlobe_baseline, dtype=np.float64).reshape(-1)
        if not (len(names) == delta.size == main_delta.size == main_base.size):
            raise ValueError("per-mover arrays must match the named movers")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "mainlobe_delta", main_delta)
        object.__setattr__(self, "mainlobe_baseline", main_base)

    @property
    def n_improving(self) -> int:
        return int(np.sum(self.delta < 0.0))

    @property
    def n_material_mainlobe_regression(self) -> int:
        floor = np.maximum(MATERIAL_MAINLOBE_ABS, MATERIAL_MAINLOBE_REL * self.mainlobe_baseline)
        return int(np.sum(self.mainlobe_delta > floor))

    def passes(self) -> bool:
        if self.names != SPW4_HOLDOUT_MOVING_ANTENNA_NAMES:
            raise ValueError("mover consistency gate requires the five frozen holdout movers")
        if not bool(np.all(np.isfinite(self.delta))) or not bool(
            np.all(np.isfinite(self.mainlobe_delta))
        ):
            return False
        return (
            self.n_improving >= MIN_MOVERS_IMPROVING and self.n_material_mainlobe_regression == 0
        )


@dataclass(frozen=True)
class Spw4CorrectionHoldouts:
    """Frozen train and ranking masks for the SPW-4 diagonal correction."""

    train: NDArray[np.bool_]
    spatial_holdout: NDArray[np.bool_]
    mover_holdout: NDArray[np.bool_]
    reference_holdout: NDArray[np.bool_]
    spatial_mover_holdout: NDArray[np.bool_]
    unused_c147: NDArray[np.bool_]
    protocol_version: int = SPW4_CORRECTION_PROTOCOL_VERSION
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            PROTOCOL_NOTE,
            PAIRED_DELTA_NOTE,
            MOVER_GATE_NOTE,
            BORESIGHT_NOTE,
            PHASE_NOTE,
            SPW5_NOTE,
        )
    )

    def __post_init__(self) -> None:
        arrays = (
            self.train,
            self.spatial_holdout,
            self.mover_holdout,
            self.reference_holdout,
            self.spatial_mover_holdout,
            self.unused_c147,
        )
        shapes = {np.asarray(item, dtype=bool).shape for item in arrays}
        if len(shapes) != 1:
            raise ValueError("correction holdout masks must share one row shape")
        train = np.asarray(self.train, dtype=bool).reshape(-1)
        spatial = np.asarray(self.spatial_holdout, dtype=bool).reshape(-1)
        mover = np.asarray(self.mover_holdout, dtype=bool).reshape(-1)
        reference = np.asarray(self.reference_holdout, dtype=bool).reshape(-1)
        both = np.asarray(self.spatial_mover_holdout, dtype=bool).reshape(-1)
        unused = np.asarray(self.unused_c147, dtype=bool).reshape(-1)
        ranking = train | spatial | mover
        if np.any(ranking & unused):
            raise ValueError("C147-* rows cannot enter training or ranking holdouts")
        if np.any(train & spatial) or np.any(train & mover) or np.any(train & reference):
            raise ValueError("training rows cannot overlap ranking or diagnostic holdouts")
        if np.any(spatial & mover):
            raise ValueError("spatial and mover ranking holdouts must be disjoint")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "spatial_holdout", spatial)
        object.__setattr__(self, "mover_holdout", mover)
        object.__setattr__(self, "reference_holdout", reference)
        object.__setattr__(self, "spatial_mover_holdout", both)
        object.__setattr__(self, "unused_c147", unused)


def frozen_antenna_partition() -> dict[str, tuple[str, ...]]:
    """Named THOL0001 antennas locked before any correction is fitted."""

    return {
        "training_movers": SPW4_TRAINING_MOVING_ANTENNA_NAMES,
        "holdout_movers": SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
        "training_references": SPW4_TRAINING_REFERENCE_ANTENNA_NAMES,
        "holdout_references": SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES,
        "all_movers": THOL0001_MOVING_ANTENNA_NAMES,
        "all_references": THOL0001_REFERENCE_ANTENNA_NAMES,
    }


def require_disjoint_antenna_cover() -> None:
    movers = set(SPW4_TRAINING_MOVING_ANTENNA_NAMES) | set(SPW4_HOLDOUT_MOVING_ANTENNA_NAMES)
    refs = set(SPW4_TRAINING_REFERENCE_ANTENNA_NAMES) | set(SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES)
    if movers != set(THOL0001_MOVING_ANTENNA_NAMES):
        raise ValueError("mover train/holdout must cover the published THOL0001 movers")
    if refs != set(THOL0001_REFERENCE_ANTENNA_NAMES):
        raise ValueError("reference train/holdout must cover the published THOL0001 refs")
    if set(SPW4_TRAINING_MOVING_ANTENNA_NAMES) & set(SPW4_HOLDOUT_MOVING_ANTENNA_NAMES):
        raise ValueError("mover training and holdout sets must be disjoint")
    if set(SPW4_TRAINING_REFERENCE_ANTENNA_NAMES) & set(SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES):
        raise ValueError("reference training and holdout sets must be disjoint")
    if movers & refs:
        raise ValueError("a THOL0001 antenna cannot be both mover and reference")


def refuse_spw5(
    *,
    frequency_hz: float | None = None,
    spectral_window_id: int | None = None,
    opened: bool = False,
) -> None:
    """SPW 5 stays sealed until the SPW-4 family and selection rule freeze."""

    if opened:
        raise RuntimeError("SPW 5 stays sealed until the SPW-4 correction family is fixed")
    if spectral_window_id == 5:
        raise RuntimeError("SPW 5 stays sealed until the SPW-4 correction family is fixed")
    if frequency_hz is not None and np.isclose(
        float(frequency_hz), SPW5_FREQUENCY_HZ, rtol=0.0, atol=0.5e6
    ):
        raise RuntimeError("SPW 5 stays sealed until the SPW-4 correction family is fixed")


def refuse_c147_training(field_id: ArrayLike) -> None:
    """Fields 1–8 remain unused prediction data."""

    fields = np.asarray(field_id, dtype=np.int32).reshape(-1)
    if np.any(np.isin(fields, np.asarray(C147_OFFSET_FIELD_IDS, dtype=np.int32))):
        raise RuntimeError("C147-* fields cannot train or rank the SPW-4 correction")


def refuse_phase_in_first_ladder(term: str) -> None:
    name = str(term)
    if name not in FIRST_LADDER_TERMS or "phase" in name:
        raise RuntimeError(
            "first ladder is amplitude-only; phase stays out until amplitude transfers"
        )


def require_boresight_unity(
    c_r0: complex | float,
    c_l0: complex | float,
    *,
    atol: float = 1.0e-12,
) -> None:
    """Refuse a correction that can absorb the absolute flux gauge."""

    if abs(complex(c_r0) - 1.0) > float(atol) or abs(complex(c_l0) - 1.0) > float(atol):
        raise ValueError("C_R(0)=C_L(0)=1; the correction cannot absorb the flux gauge")


def quantized_offset_keys(offset_lm_rad: ArrayLike) -> NDArray[np.int64]:
    """0.01-arcmin keys used by the frozen spatial checkerboard."""

    offset = np.asarray(offset_lm_rad, dtype=np.float64).reshape(-1, 2)
    keys = np.round(offset * OFFSET_CELL_QUANT_PER_RAD).astype(np.int64)
    finite = np.isfinite(offset).all(axis=1)
    keys[~finite] = np.iinfo(np.int32).min
    return keys


def spatial_holdout_from_offsets(offset_lm_rad: ArrayLike) -> NDArray[np.bool_]:
    """Hold out every other quantized ``(l, m)`` key in sorted order."""

    keys = quantized_offset_keys(offset_lm_rad)
    finite = keys[:, 0] != np.iinfo(np.int32).min
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("spatial holdout needs at least two quantized raster cells")
    low = keys[:, 1].astype(np.int64) & np.int64(0xFFFFFFFF)
    packed = (keys[:, 0].astype(np.int64) << 32) ^ low
    unique = np.unique(packed[finite])
    if unique.size < 2:
        raise ValueError("spatial holdout needs at least two quantized raster cells")
    return finite & np.isin(packed, unique[1::2])


def _ids_for_names(antenna_names: Sequence[str], wanted: Sequence[str]) -> NDArray[np.int32]:
    lookup = {str(name): index for index, name in enumerate(antenna_names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise ValueError(f"antenna name list is missing {missing}")
    return np.asarray([lookup[name] for name in wanted], dtype=np.int32)


def holoraster_correction_holdouts(
    observation: HolographyObservation,
    antenna_names: Sequence[str],
    *,
    holoraster_field_id: int = HOLORASTER_FIELD_ID,
) -> Spw4CorrectionHoldouts:
    """Same antenna/spatial split as the frozen SPW-4 protocol.

    This builder does not seal a spectral window. The SPW-4 correction
    product still goes through ``spw4_correction_holdouts``.
    """

    require_disjoint_antenna_cover()
    names = tuple(str(name) for name in antenna_names)
    if len(names) < int(observation.block.antenna_count):
        raise ValueError("antenna_names must cover every antenna id in the block")
    geometry = holoraster_pair_masks(observation)
    field_id = np.asarray(observation.block.field_id, dtype=np.int32).reshape(-1)
    unused_c147 = np.isin(field_id, np.asarray(C147_OFFSET_FIELD_IDS, dtype=np.int32))
    holoraster = (field_id == int(holoraster_field_id)) & geometry["moving_reference"]
    if not bool(np.any(holoraster)):
        raise ValueError(
            "SPW-4 correction training requires HOLORASTER field-10 moving-reference rows"
        )
    spatial_hold = spatial_holdout_from_offsets(geometry["offset_lm_rad"])
    spatial_train = holoraster & ~spatial_hold
    spatial_hold = holoraster & spatial_hold
    mover_ids = _ids_for_names(names, SPW4_HOLDOUT_MOVING_ANTENNA_NAMES)
    ref_ids = _ids_for_names(names, SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES)
    train_mover_ids = _ids_for_names(names, SPW4_TRAINING_MOVING_ANTENNA_NAMES)
    train_ref_ids = _ids_for_names(names, SPW4_TRAINING_REFERENCE_ANTENNA_NAMES)
    moving = np.asarray(geometry["moving_id"], dtype=np.int32)
    reference = np.asarray(geometry["reference_id"], dtype=np.int32)
    mover_train = np.isin(moving, train_mover_ids)
    mover_hold = np.isin(moving, mover_ids)
    ref_train = np.isin(reference, train_ref_ids)
    ref_hold = np.isin(reference, ref_ids)
    train = holoraster & spatial_train & mover_train & ref_train
    spatial = holoraster & spatial_hold & mover_train & ref_train
    movers = holoraster & spatial_train & mover_hold & ref_train
    refs = holoraster & spatial_train & mover_train & ref_hold
    both = holoraster & spatial_hold & mover_hold & ref_train
    if not bool(np.any(train)):
        raise ValueError("frozen SPW-4 split has no training rows")
    if not bool(np.any(spatial)) or not bool(np.any(movers)):
        raise ValueError("frozen SPW-4 split needs both spatial and mover holdout rows")
    return Spw4CorrectionHoldouts(
        train=train,
        spatial_holdout=spatial,
        mover_holdout=movers,
        reference_holdout=refs,
        spatial_mover_holdout=both,
        unused_c147=unused_c147,
    )


def spw4_correction_holdouts(
    observation: HolographyObservation,
    antenna_names: Sequence[str],
    *,
    holoraster_field_id: int = HOLORASTER_FIELD_ID,
) -> Spw4CorrectionHoldouts:
    """Build the frozen SPW-4 train / spatial / mover / reference masks."""

    frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64)
    if frequencies.size and bool(
        np.all(np.isclose(frequencies, SPW5_FREQUENCY_HZ, rtol=0.0, atol=0.5e6))
    ):
        refuse_spw5(frequency_hz=SPW5_FREQUENCY_HZ)
    refuse_spw5(spectral_window_id=int(observation.block.spectral_window_id))
    return holoraster_correction_holdouts(
        observation,
        antenna_names,
        holoraster_field_id=holoraster_field_id,
    )


def refuse_visibility_bootstrap(cluster_kind: str) -> None:
    """Individual visibilities are not independent evidence."""

    if str(cluster_kind) in {"visibility", "visibility_row", "row"}:
        raise RuntimeError("do not bootstrap individual visibilities as independent evidence")


def refuse_magnitude_loss(loss_name: str) -> None:
    if str(loss_name) in {"db_ratio", "magnitude_ratio", "abs_only", "magnitude"}:
        raise RuntimeError("fit complex visibilities, not magnitudes or dB ratios")


def packed_offset_keys(offset_lm_rad: ArrayLike) -> NDArray[np.int64]:
    keys = quantized_offset_keys(offset_lm_rad)
    low = keys[:, 1].astype(np.int64) & np.int64(0xFFFFFFFF)
    return (keys[:, 0].astype(np.int64) << 32) ^ low


def spatial_cluster_ids(offset_lm_rad: ArrayLike, mask: ArrayLike) -> NDArray[np.int64]:
    choose = np.asarray(mask, dtype=bool).reshape(-1)
    return packed_offset_keys(offset_lm_rad)[choose]


def mover_cluster_ids(moving_id: ArrayLike, mask: ArrayLike) -> NDArray[np.int64]:
    choose = np.asarray(mask, dtype=bool).reshape(-1)
    return np.asarray(moving_id, dtype=np.int64).reshape(-1)[choose]


def _require_single_channel_planes(values: ArrayLike, name: str) -> NDArray:
    array = np.asarray(values)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one channel")
        return array[:, 0]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (sample, 2, 2) or (sample, 1, 2, 2)")
    return array


def copolar_loss_parts(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-sample complex residual and observed power on RR and LL."""

    meas = _require_single_channel_planes(measured, "measured")
    pred = _require_single_channel_planes(predicted, "predicted")
    wgt = _require_single_channel_planes(weight, "weight")
    if meas.shape != pred.shape or meas.shape[:-2] != wgt.shape[:-2]:
        raise ValueError("measured, predicted, and weight must share the sample axis")
    numer = np.zeros(meas.shape[0], dtype=np.float64)
    denom = np.zeros(meas.shape[0], dtype=np.float64)
    for _name, row, col in COPOLAR_HANDS:
        obs = meas[:, row, col]
        hat = pred[:, row, col]
        ww = wgt[:, row, col]
        finite = np.isfinite(obs) & np.isfinite(hat) & np.isfinite(ww) & (ww > 0.0)
        resid = ww * np.abs(obs - hat) ** 2
        power = ww * np.abs(obs) ** 2
        numer = numer + np.where(finite, resid, 0.0)
        denom = denom + np.where(finite, power, 0.0)
    return numer, denom


def complex_visibility_loss(
    measured: ArrayLike,
    predicted: ArrayLike,
    weight: ArrayLike,
) -> float:
    numer, denom = copolar_loss_parts(measured, predicted, weight)
    total_den = float(np.sum(denom))
    if total_den <= 0.0:
        return float("nan")
    return float(np.sum(numer) / total_den)


def reduce_cluster_parts(
    numer: ArrayLike,
    denom: ArrayLike,
    labels: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Sum residual and observed power inside each cluster. No row bootstrap."""

    values_n = np.asarray(numer, dtype=np.float64).reshape(-1)
    values_d = np.asarray(denom, dtype=np.float64).reshape(-1)
    ids = np.asarray(labels).reshape(-1)
    if values_n.size != ids.size or values_d.size != ids.size:
        raise ValueError("loss parts and cluster labels must align")
    unique, inverse = np.unique(ids, return_inverse=True)
    return (
        unique.astype(np.int64, copy=False),
        np.bincount(inverse, weights=values_n),
        np.bincount(inverse, weights=values_d),
    )


def paired_delta_from_parts(
    candidate_numer: ArrayLike,
    candidate_denom: ArrayLike,
    baseline_numer: ArrayLike,
    baseline_denom: ArrayLike,
) -> float:
    cand_d = float(np.sum(candidate_denom))
    base_d = float(np.sum(baseline_denom))
    if cand_d <= 0.0 or base_d <= 0.0:
        return float("nan")
    return float(np.sum(candidate_numer) / cand_d - np.sum(baseline_numer) / base_d)


def bootstrap_paired_delta(
    candidate_numer: ArrayLike,
    candidate_denom: ArrayLike,
    baseline_numer: ArrayLike,
    baseline_denom: ArrayLike,
    *,
    axis: str,
    cluster_kind: str,
    n_boot: int = PAIRED_DELTA_BOOTSTRAP,
    seed: int = 0,
) -> PairedLossDifference:
    """Resample clusters, keeping every row that belongs to a drawn cluster."""

    refuse_visibility_bootstrap(cluster_kind)
    cand_n = np.asarray(candidate_numer, dtype=np.float64).reshape(-1)
    cand_d = np.asarray(candidate_denom, dtype=np.float64).reshape(-1)
    base_n = np.asarray(baseline_numer, dtype=np.float64).reshape(-1)
    base_d = np.asarray(baseline_denom, dtype=np.float64).reshape(-1)
    if not (cand_n.size == cand_d.size == base_n.size == base_d.size):
        raise ValueError("paired bootstrap parts must share one cluster axis")
    n_cluster = int(cand_n.size)
    point = paired_delta_from_parts(cand_n, cand_d, base_n, base_d)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=np.float64)
    for boot in range(int(n_boot)):
        index = rng.choice(n_cluster, size=n_cluster, replace=True)
        draws[boot] = paired_delta_from_parts(
            cand_n[index], cand_d[index], base_n[index], base_d[index]
        )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0 or not np.isfinite(point):
        raise ValueError("paired bootstrap produced no finite ΔL")
    return PairedLossDifference(
        axis=axis,
        delta=float(point),
        delta_lo=float(np.quantile(finite, 0.025)),
        delta_hi=float(np.quantile(finite, 0.975)),
        n_clusters=n_cluster,
        n_boot=int(finite.size),
        cluster_kind=cluster_kind,
    )


def score_paired_holdout(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
    labels: ArrayLike,
    *,
    axis: str,
    cluster_kind: str,
    n_boot: int = PAIRED_DELTA_BOOTSTRAP,
    seed: int = 0,
) -> PairedLossDifference:
    refuse_visibility_bootstrap(cluster_kind)
    cand_n, cand_d = copolar_loss_parts(measured, candidate, weight)
    base_n, base_d = copolar_loss_parts(measured, baseline, weight)
    _ids, c_n, c_d = reduce_cluster_parts(cand_n, cand_d, labels)
    _ids_b, b_n, b_d = reduce_cluster_parts(base_n, base_d, labels)
    if not np.array_equal(_ids, _ids_b):
        raise ValueError("candidate and baseline cluster labels drifted")
    return bootstrap_paired_delta(
        c_n,
        c_d,
        b_n,
        b_d,
        axis=axis,
        cluster_kind=cluster_kind,
        n_boot=n_boot,
        seed=seed,
    )


def mover_paired_deltas(
    measured: ArrayLike,
    candidate: ArrayLike,
    baseline: ArrayLike,
    weight: ArrayLike,
    moving_id: ArrayLike,
    antenna_names: Sequence[str],
    *,
    mainlobe_mask: ArrayLike | None = None,
) -> MoverPairedDeltas:
    """Point ΔL for each frozen holdout mover, plus main-lobe regression."""

    ids = np.asarray(moving_id, dtype=np.int32).reshape(-1)
    lookup = {str(name): index for index, name in enumerate(antenna_names)}
    cand_n, cand_d = copolar_loss_parts(measured, candidate, weight)
    base_n, base_d = copolar_loss_parts(measured, baseline, weight)
    if mainlobe_mask is None:
        main = np.ones(ids.size, dtype=bool)
    else:
        main = np.asarray(mainlobe_mask, dtype=bool).reshape(-1)
        if main.size != ids.size:
            raise ValueError("main-lobe mask must match the mover-holdout samples")
    deltas = []
    main_deltas = []
    main_base = []
    for name in SPW4_HOLDOUT_MOVING_ANTENNA_NAMES:
        antenna = lookup.get(name)
        if antenna is None:
            raise ValueError(f"antenna name list is missing holdout mover {name}")
        choose = ids == int(antenna)
        if not bool(np.any(choose)):
            raise ValueError(f"mover holdout has no rows for {name}")
        deltas.append(
            paired_delta_from_parts(
                cand_n[choose], cand_d[choose], base_n[choose], base_d[choose]
            )
        )
        lobe = choose & main
        if bool(np.any(lobe)):
            main_deltas.append(
                paired_delta_from_parts(cand_n[lobe], cand_d[lobe], base_n[lobe], base_d[lobe])
            )
            den = float(np.sum(base_d[lobe]))
            main_base.append(float(np.sum(base_n[lobe]) / den) if den > 0.0 else float("nan"))
        else:
            main_deltas.append(0.0)
            main_base.append(0.0)
    return MoverPairedDeltas(
        names=SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
        delta=np.asarray(deltas, dtype=np.float64),
        mainlobe_delta=np.asarray(main_deltas, dtype=np.float64),
        mainlobe_baseline=np.asarray(main_base, dtype=np.float64),
    )


def select_nested_correction(
    spatial: PairedLossDifference,
    moving: PairedLossDifference,
    mover_units: MoverPairedDeltas,
    *,
    reference: PairedLossDifference | None = None,
) -> str:
    """Accept a nested term only when both paired ranking intervals improve."""

    if spatial.axis != "spatial" or spatial.cluster_kind != "spatial_cell":
        raise ValueError("spatial gate must cluster by spatial cell")
    if moving.axis != "moving" or moving.cluster_kind != "moving_antenna":
        raise ValueError("mover gate must cluster by held-out mover")
    if reference is not None and reference.cluster_kind != "reference_antenna":
        raise ValueError("reference scores must cluster by reference antenna")
    if spatial.improves() and moving.improves() and mover_units.passes():
        return "accept_candidate"
    return "keep_baseline"


def frozen_protocol_payload() -> dict[str, object]:
    """Provenance record for the locked SPW-4 correction split."""

    require_disjoint_antenna_cover()
    return {
        "artifact": CASSBEAM_DIAGONAL_CORRECTION,
        "protocol_version": SPW4_CORRECTION_PROTOCOL_VERSION,
        "training_frequency_hz": SPW4_TRAINING_FREQUENCY_HZ,
        "spw5_frequency_hz": SPW5_FREQUENCY_HZ,
        "spw5_status": "sealed",
        "holoraster_field_id": HOLORASTER_FIELD_ID,
        "unused_field_ids": list(ON_AXIS_FIELD_IDS + C147_OFFSET_FIELD_IDS),
        "spatial_split": "quantized_offset_keys_sorted_odd",
        "offset_cell_quant_per_rad": OFFSET_CELL_QUANT_PER_RAD,
        "antennas": frozen_antenna_partition(),
        "first_ladder": list(FIRST_LADDER_TERMS),
        "ranking_axes": list(RANKING_AXES),
        "diagnostic_axes": list(DIAGNOSTIC_AXES),
        "selection_rule": "paired_delta_L_ci95_below_zero",
        "spatial_bootstrap_unit": "spatial_cell",
        "mover_bootstrap_unit": "moving_antenna",
        "visibility_bootstrap": False,
        "min_movers_improving": MIN_MOVERS_IMPROVING,
        "material_mainlobe_abs": MATERIAL_MAINLOBE_ABS,
        "material_mainlobe_rel": MATERIAL_MAINLOBE_REL,
        "boresight": "C_R(0)=C_L(0)=1",
        "phase_in_first_ladder": False,
        "full_jones": "experimental",
        "notes": [
            PROTOCOL_NOTE,
            PAIRED_DELTA_NOTE,
            MOVER_GATE_NOTE,
            BORESIGHT_NOTE,
            PHASE_NOTE,
            SPW5_NOTE,
        ],
    }


def protocol_as_mapping(payload: Mapping[str, object] | None = None) -> Mapping[str, object]:
    record = dict(payload or frozen_protocol_payload())
    if record.get("spw5_status") != "sealed":
        raise RuntimeError("SPW 5 stays sealed until the SPW-4 correction family is fixed")
    if record.get("phase_in_first_ladder") is not False:
        raise RuntimeError("first ladder cannot include phase")
    if record.get("boresight") != "C_R(0)=C_L(0)=1":
        raise RuntimeError("correction protocol lost the boresight-unity constraint")
    if record.get("selection_rule") != "paired_delta_L_ci95_below_zero":
        raise RuntimeError("correction protocol must use the paired ΔL interval")
    if record.get("visibility_bootstrap") is not False:
        raise RuntimeError("individual visibilities are not bootstrap units")
    return record
