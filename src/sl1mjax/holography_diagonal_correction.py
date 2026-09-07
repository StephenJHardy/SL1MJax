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
SPW4_CORRECTION_PROTOCOL_VERSION = 1
SPW4_TRAINING_FREQUENCY_HZ = THOL0001_SPW4_CHANNEL_32_HZ
SPW5_FREQUENCY_HZ = THOL0001_LOWER_C_NATIVE_HZ[1]
HOLORASTER_FIELD_ID = 10
ON_AXIS_FIELD_IDS = (0, 9)
OFFSET_CELL_QUANT_PER_RAD = 180.0 * 60.0 / np.pi * 100.0

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
    "Selection requires improvement on held-out movers and held-out spatial "
    "cells, each with a 95% interval. Reference holdouts are diagnostic. "
    "C147-* and SPW 5 stay unused until the family and selection rule freeze."
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


@dataclass(frozen=True)
class HoldoutScore:
    """One holdout residual-power estimate with a 95% interval."""

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
        default_factory=lambda: (PROTOCOL_NOTE, BORESIGHT_NOTE, PHASE_NOTE, SPW5_NOTE)
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


def spw4_correction_holdouts(
    observation: HolographyObservation,
    antenna_names: Sequence[str],
    *,
    holoraster_field_id: int = HOLORASTER_FIELD_ID,
) -> Spw4CorrectionHoldouts:
    """Build the frozen SPW-4 train / spatial / mover / reference masks."""

    require_disjoint_antenna_cover()
    frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64)
    if frequencies.size and bool(
        np.all(np.isclose(frequencies, SPW5_FREQUENCY_HZ, rtol=0.0, atol=0.5e6))
    ):
        refuse_spw5(frequency_hz=SPW5_FREQUENCY_HZ)
    refuse_spw5(spectral_window_id=int(observation.block.spectral_window_id))
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


def axis_improves(baseline: HoldoutScore, candidate: HoldoutScore) -> bool:
    """Candidate 95% interval lies entirely below the baseline point estimate."""

    if baseline.axis != candidate.axis:
        raise ValueError("holdout scores must share an axis")
    return float(candidate.residual_power_hi) < float(baseline.residual_power)


def select_nested_correction(
    baseline_spatial: HoldoutScore,
    candidate_spatial: HoldoutScore,
    baseline_mover: HoldoutScore,
    candidate_mover: HoldoutScore,
    *,
    reference: tuple[HoldoutScore, HoldoutScore] | None = None,
) -> str:
    """Accept a nested term only when both ranking holdouts improve."""

    if baseline_spatial.axis != "spatial" or candidate_spatial.axis != "spatial":
        raise ValueError("spatial scores must be labelled spatial")
    if baseline_mover.axis != "moving" or candidate_mover.axis != "moving":
        raise ValueError("mover scores must be labelled moving")
    if reference is not None:
        baseline_ref, candidate_ref = reference
        if baseline_ref.axis != "reference" or candidate_ref.axis != "reference":
            raise ValueError("reference scores must be labelled reference")
    if axis_improves(baseline_spatial, candidate_spatial) and axis_improves(
        baseline_mover, candidate_mover
    ):
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
        "boresight": "C_R(0)=C_L(0)=1",
        "phase_in_first_ladder": False,
        "full_jones": "experimental",
        "notes": [PROTOCOL_NOTE, BORESIGHT_NOTE, PHASE_NOTE, SPW5_NOTE],
    }


def protocol_as_mapping(payload: Mapping[str, object] | None = None) -> Mapping[str, object]:
    record = dict(payload or frozen_protocol_payload())
    if record.get("spw5_status") != "sealed":
        raise RuntimeError("SPW 5 stays sealed until the SPW-4 correction family is fixed")
    if record.get("phase_in_first_ladder") is not False:
        raise RuntimeError("first ladder cannot include phase")
    if record.get("boresight") != "C_R(0)=C_L(0)=1":
        raise RuntimeError("correction protocol lost the boresight-unity constraint")
    return record
