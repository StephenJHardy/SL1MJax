"""Memo 195 pointing reconstruction audit.

This path joins MAIN metadata to POINTING. It does not read visibilities,
apply calibration, or recover Jones matrices.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.holography import (
    THOL0001_HOLORASTER_FIELD,
    THOL0001_LOWER_C_NATIVE_HZ,
    AntennaPointingRole,
    AntennaPointingTable,
    HolographyGateStatus,
    HolographyPointingAudit,
    HolographyRowReason,
    JoinRule,
    Memo195LowerCRaster,
    OffsetSign,
    RasterGeometryEstimate,
    ResolvedAntennaPointing,
    audit_resolved_pointing,
    classify_antenna_roles,
    cluster_antenna_dwells,
    holography_execution_provenance,
    join_pointing_alignment,
    pointing_table_overlap_pairs,
    resolve_antenna_pointing,
)

NATIVE_CHANNEL_MAX_RESIDUAL_HZ = 1.1e6
MAP_ANTENNA_SURFACE_TOKEN = "MAP_ANTENNA_SURFACE"
_REASON_PRIORITY = (
    HolographyRowReason.MISSING_POINTING,
    HolographyRowReason.MIXED_STATE,
    HolographyRowReason.NOT_MAP_ANTENNA_SURFACE,
    HolographyRowReason.DWELL_JUMP,
    HolographyRowReason.TRANSITION_GUARD,
    HolographyRowReason.NOT_MOVING_REFERENCE,
    HolographyRowReason.UNSETTLED,
)


@dataclass(frozen=True)
class NativeChannelSelection:
    """One native channel nearest a Memo 195 holography frequency."""

    data_desc_id: int
    spectral_window_id: int
    polarization_id: int
    channel: int
    frequency_hz: float
    target_frequency_hz: float
    residual_hz: float


@dataclass(frozen=True)
class HolographyMainMetadata:
    """MAIN columns needed for pointing reconstruction. No visibility payload."""

    time_s: NDArray[np.float64]
    interval_s: NDArray[np.float64]
    antenna1: NDArray[np.int32]
    antenna2: NDArray[np.int32]
    field_id: NDArray[np.int32]
    state_id: NDArray[np.int32]
    scan_number: NDArray[np.int32]
    data_desc_id: NDArray[np.int32]
    field_names: tuple[str, ...]
    state_modes: tuple[str, ...]
    antenna_names: tuple[str, ...]
    phase_centre_rad: tuple[float, float]
    main_row_count: int = 0

    def __post_init__(self) -> None:
        n_row = int(np.asarray(self.time_s).reshape(-1).size)
        arrays = (
            self.interval_s,
            self.antenna1,
            self.antenna2,
            self.field_id,
            self.state_id,
            self.scan_number,
            self.data_desc_id,
        )
        if any(int(np.asarray(array).reshape(-1).size) != n_row for array in arrays):
            raise ValueError("MAIN metadata columns must have the same row count")
        object.__setattr__(self, "time_s", np.asarray(self.time_s, dtype=np.float64).reshape(-1))
        object.__setattr__(
            self, "interval_s", np.asarray(self.interval_s, dtype=np.float64).reshape(-1)
        )
        object.__setattr__(self, "antenna1", np.asarray(self.antenna1, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "antenna2", np.asarray(self.antenna2, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "field_id", np.asarray(self.field_id, dtype=np.int32).reshape(-1))
        object.__setattr__(self, "state_id", np.asarray(self.state_id, dtype=np.int32).reshape(-1))
        object.__setattr__(
            self, "scan_number", np.asarray(self.scan_number, dtype=np.int32).reshape(-1)
        )
        object.__setattr__(
            self, "data_desc_id", np.asarray(self.data_desc_id, dtype=np.int32).reshape(-1)
        )
        object.__setattr__(self, "field_names", tuple(self.field_names))
        object.__setattr__(self, "state_modes", tuple(self.state_modes))
        object.__setattr__(self, "antenna_names", tuple(self.antenna_names))


@dataclass(frozen=True)
class RasterCellOccupancy:
    """One recovered raster cell after clustering settled moving dwells."""

    l_arcmin: float
    m_arcmin: float
    sample_count: int
    moving_antenna_ids: tuple[int, ...]
    family: str


@dataclass(frozen=True)
class TimeAlignmentResiduals:
    """MAIN TIME minus joined POINTING.TIME on matched samples."""

    n_matched: int
    n_unmatched: int
    median_s: float
    p95_abs_s: float
    max_abs_s: float
    rms_s: float
    max_abs_fraction_of_half_interval: float = float("nan")
    n_outside_half_interval: int = 0
    max_pointing_interval_s: float = float("nan")


@dataclass(frozen=True)
class PointingReconstructionAudit:
    """First real-data gate: reconstruct the Memo 195 raster without visibilities."""

    provenance: dict[str, str]
    field_name: str
    selected_column: str
    native_channels: tuple[NativeChannelSelection, ...]
    moving_antenna_ids: tuple[int, ...]
    reference_antenna_ids: tuple[int, ...]
    moving_antenna_names: tuple[str, ...]
    reference_antenna_names: tuple[str, ...]
    raster_cells: tuple[RasterCellOccupancy, ...]
    raster: RasterGeometryEstimate
    memo195: Memo195LowerCRaster
    memo195_dense_status: HolographyGateStatus
    memo195_sparse_status: HolographyGateStatus
    sample_reasons: dict[str, int]
    row_reasons: dict[str, int]
    offset_range_arcmin: dict[str, float]
    cell_spacing_arcmin: dict[str, float | None]
    time_alignment: TimeAlignmentResiduals
    composition: dict[str, dict[str, int]]
    unique_times_by_ddid: dict[int, int]
    selected_main_rows: int
    unique_time_count: int
    notes: tuple[str, ...]
    pointing_audit: HolographyPointingAudit
    resolved: ResolvedAntennaPointing
    sample_reason_grid: NDArray[np.str_] | None = None
    scan_per_time: NDArray[np.int32] | None = None
    occupancy: object | None = None
    overlap_proof: object | None = None


def is_map_antenna_surface(obs_mode: str) -> bool:
    """True when STATE.OBS_MODE is a holography mapping intent."""

    return MAP_ANTENNA_SURFACE_TOKEN in str(obs_mode).upper()


def unique_times_within_groups(
    time_s: ArrayLike,
    group_id: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Sort unique times inside each group. Input MAIN order is not required."""

    times = np.asarray(time_s, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_id, dtype=np.int32).reshape(-1)
    if times.size != groups.size:
        raise ValueError("time_s and group_id must have the same length")
    unique_times: list[NDArray[np.float64]] = []
    unique_groups: list[NDArray[np.int32]] = []
    for gid in np.unique(groups):
        ordered = np.unique(times[groups == gid])
        unique_times.append(ordered)
        unique_groups.append(np.full(ordered.size, int(gid), dtype=np.int32))
    if not unique_times:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)
    return np.concatenate(unique_times), np.concatenate(unique_groups)


def select_native_channels(
    channel_frequency_hz: Mapping[int, ArrayLike],
    data_descriptions: Sequence[tuple[int, int, int]],
    targets_hz: tuple[float, ...] = THOL0001_LOWER_C_NATIVE_HZ,
    *,
    max_residual_hz: float = NATIVE_CHANNEL_MAX_RESIDUAL_HZ,
) -> tuple[NativeChannelSelection, ...]:
    """Pick the native channel nearest each target frequency."""

    selections: list[NativeChannelSelection] = []
    for target in targets_hz:
        best: NativeChannelSelection | None = None
        for data_desc_id, spectral_window_id, polarization_id in data_descriptions:
            if int(spectral_window_id) not in channel_frequency_hz:
                continue
            frequencies = np.asarray(
                channel_frequency_hz[int(spectral_window_id)], dtype=np.float64
            ).reshape(-1)
            if frequencies.size == 0:
                continue
            channel = int(np.argmin(np.abs(frequencies - float(target))))
            candidate = NativeChannelSelection(
                data_desc_id=int(data_desc_id),
                spectral_window_id=int(spectral_window_id),
                polarization_id=int(polarization_id),
                channel=channel,
                frequency_hz=float(frequencies[channel]),
                target_frequency_hz=float(target),
                residual_hz=float(frequencies[channel] - float(target)),
            )
            if best is None or abs(candidate.residual_hz) < abs(best.residual_hz):
                best = candidate
        if best is None:
            raise ValueError(f"no spectral window contains {target} Hz")
        if abs(best.residual_hz) > float(max_residual_hz):
            raise ValueError(f"nearest channel to {target} Hz is {best.residual_hz} Hz away")
        selections.append(best)
    return tuple(selections)


def reconstruct_holography_pointing(
    table: AntennaPointingTable,
    main: HolographyMainMetadata,
    *,
    field_name: str = THOL0001_HOLORASTER_FIELD,
    data_desc_ids: tuple[int, ...] | None = None,
    selected_column: str = "POINTING_OFFSET",
    offset_sign: OffsetSign = "commanded_pointing",
    join_rule: JoinRule = "interval",
    join_tolerance_s: float = 0.0,
    settled_jump_arcmin: float = 0.2,
    transition_guard_samples: int = 1,
    cluster_radius_arcmin: float = 0.15,
    provenance: Mapping[str, str] | None = None,
    native_channels: tuple[NativeChannelSelection, ...] = (),
    memo195: Memo195LowerCRaster | None = None,
) -> PointingReconstructionAudit:
    """Join selected MAIN times to POINTING and audit the recovered raster."""

    expected = memo195 or Memo195LowerCRaster()
    selected = _select_main_rows(main, field_name=field_name, data_desc_ids=data_desc_ids)
    if selected.size == 0:
        raise ValueError(
            f"no MAIN metadata rows remain for field {field_name!r} "
            f"and data_desc_ids={data_desc_ids}"
        )
    time_s = main.time_s[selected]
    group_times, group_ids = unique_times_within_groups(time_s, main.data_desc_id[selected])
    unique_time_s = np.unique(time_s)
    unique_times_by_ddid = {
        int(ddid): int(np.sum(group_ids == ddid)) for ddid in np.unique(group_ids)
    }
    notes = [
        "unique times are grouped and sorted within the selected DDID/SPW",
        "global MAIN TIME ordering is not required",
        "ON_SOURCE was not used for validity",
        f"{selected_column} is treated as a native raster offset when the measure is AZELGEO",
    ]
    antenna_id = np.unique(
        np.concatenate((main.antenna1[selected], main.antenna2[selected], table.antenna_id))
    ).astype(np.int32)
    resolved = resolve_antenna_pointing(
        table,
        unique_time_s,
        antenna_id,
        selected_column=selected_column,
        phase_centre_rad=main.phase_centre_rad,
        offset_sign=offset_sign,
        join_rule=join_rule,
        join_tolerance_s=join_tolerance_s,
        settled_jump_arcmin=settled_jump_arcmin,
        use_on_source=False,
    )
    time_state = _unique_time_state(
        unique_time_s, time_s, main.state_id[selected], main.state_modes
    )
    sample_reason = _sample_reason_grid(resolved.valid, resolved.settled)
    settled = np.array(resolved.settled, copy=True)
    settled, sample_reason = _apply_transition_guards(
        settled, resolved.valid, sample_reason, transition_guard_samples
    )
    settled, sample_reason = _apply_state_to_settled(settled, sample_reason, time_state)
    roles = classify_antenna_roles(resolved.offset_lm_rad, resolved.valid, settled)
    resolved = ResolvedAntennaPointing(
        unique_time_s=resolved.unique_time_s,
        antenna_id=resolved.antenna_id,
        offset_lm_rad=resolved.offset_lm_rad,
        valid=resolved.valid,
        settled=settled,
        role=roles,
        selected_column=resolved.selected_column,
        offset_sign=resolved.offset_sign,
        join_rule=resolved.join_rule,
        join_tolerance_s=resolved.join_tolerance_s,
        measure_ref=resolved.measure_ref,
        units=resolved.units,
        notes=tuple((*resolved.notes, *notes)),
    )
    pointing_audit = audit_resolved_pointing(
        resolved, table, main.phase_centre_rad, memo195=expected
    )
    residual_s, _pointing_time_s, pointing_interval_s, overlap_hits = join_pointing_alignment(
        table,
        unique_time_s,
        antenna_id,
        join_rule=join_rule,
        join_tolerance_s=join_tolerance_s,
    )
    time_index = np.searchsorted(unique_time_s, time_s)
    row_reasons = _count_row_reasons(
        main.antenna1[selected],
        main.antenna2[selected],
        time_index,
        resolved,
        sample_reason,
    )
    sample_counts = _count_labels(sample_reason.reshape(-1))
    moving_ids, reference_ids = _role_antennas(resolved)
    cells = _relabel_cells_by_nearest_spacing(
        _merge_raster_cells(
            cluster_antenna_dwells(resolved, cluster_radius_arcmin=cluster_radius_arcmin),
            expected,
            cluster_radius_arcmin=cluster_radius_arcmin,
        ),
        expected,
    )
    cell_raster = _combine_raster_estimates(
        pointing_audit.raster, _raster_from_cells(cells, expected)
    )
    if provenance is None:
        provenance = holography_execution_provenance(table.source_ms)
    from sl1mjax.holography_pointing_maps import (
        build_occupancy_report,
        scan_per_unique_time,
        summarize_alignment,
    )

    scan_times = scan_per_unique_time(unique_time_s, time_s, main.scan_number[selected])
    occupancy = build_occupancy_report(resolved, scan_times, memo195=expected)
    alignment, overlap_proof = summarize_alignment(
        residual_s,
        pointing_interval_s,
        resolved.valid,
        overlap_hits=overlap_hits,
        pointing_overlap_pairs=pointing_table_overlap_pairs(table),
        max_pointing_interval_s=float(np.max(table.interval_s)),
    )
    return PointingReconstructionAudit(
        provenance=dict(provenance),
        field_name=field_name,
        selected_column=selected_column,
        native_channels=native_channels,
        moving_antenna_ids=moving_ids,
        reference_antenna_ids=reference_ids,
        moving_antenna_names=_names_for(moving_ids, main.antenna_names),
        reference_antenna_names=_names_for(reference_ids, main.antenna_names),
        raster_cells=cells,
        raster=cell_raster,
        memo195=expected,
        memo195_dense_status="pass" if cell_raster.dense_spacing_arcmin is not None else "fail",
        memo195_sparse_status="pass" if cell_raster.sparse_spacing_arcmin is not None else "fail",
        sample_reasons=sample_counts,
        row_reasons=row_reasons,
        offset_range_arcmin=_offset_range_arcmin(resolved),
        cell_spacing_arcmin={
            "dense": cell_raster.dense_spacing_arcmin,
            "sparse": cell_raster.sparse_spacing_arcmin,
        },
        time_alignment=alignment,
        composition=_composition(main, selected),
        unique_times_by_ddid=unique_times_by_ddid,
        selected_main_rows=int(selected.size),
        unique_time_count=int(unique_time_s.size),
        notes=resolved.notes,
        pointing_audit=pointing_audit,
        resolved=resolved,
        sample_reason_grid=sample_reason,
        scan_per_time=scan_times,
        occupancy=occupancy,
        overlap_proof=overlap_proof,
    )


def pointing_reconstruction_audit_as_dict(audit: PointingReconstructionAudit) -> dict[str, object]:
    """JSON-safe audit payload. Arrays stay out of the report."""

    return {
        "provenance": dict(audit.provenance),
        "field_name": audit.field_name,
        "selected_column": audit.selected_column,
        "native_channels": [
            {
                "data_desc_id": channel.data_desc_id,
                "spectral_window_id": channel.spectral_window_id,
                "polarization_id": channel.polarization_id,
                "channel": channel.channel,
                "frequency_hz": channel.frequency_hz,
                "target_frequency_hz": channel.target_frequency_hz,
                "residual_hz": channel.residual_hz,
            }
            for channel in audit.native_channels
        ],
        "moving_antenna_ids": list(audit.moving_antenna_ids),
        "reference_antenna_ids": list(audit.reference_antenna_ids),
        "moving_antenna_names": list(audit.moving_antenna_names),
        "reference_antenna_names": list(audit.reference_antenna_names),
        "raster_cells": [
            {
                "l_arcmin": cell.l_arcmin,
                "m_arcmin": cell.m_arcmin,
                "sample_count": cell.sample_count,
                "moving_antenna_ids": list(cell.moving_antenna_ids),
                "family": cell.family,
            }
            for cell in audit.raster_cells
        ],
        "raster": {
            "spacings_arcmin": list(audit.raster.spacings_arcmin),
            "dense_spacing_arcmin": audit.raster.dense_spacing_arcmin,
            "sparse_spacing_arcmin": audit.raster.sparse_spacing_arcmin,
            "n_dwell": audit.raster.n_dwell,
        },
        "memo195": {
            "dense_n": audit.memo195.dense_n,
            "dense_spacing_arcmin": audit.memo195.dense_spacing_arcmin,
            "sparse_n": audit.memo195.sparse_n,
            "sparse_spacing_arcmin": audit.memo195.sparse_spacing_arcmin,
            "dense_status": audit.memo195_dense_status,
            "sparse_status": audit.memo195_sparse_status,
        },
        "sample_reasons": dict(audit.sample_reasons),
        "row_reasons": dict(audit.row_reasons),
        "offset_range_arcmin": dict(audit.offset_range_arcmin),
        "cell_spacing_arcmin": dict(audit.cell_spacing_arcmin),
        "time_alignment": {
            "n_matched": audit.time_alignment.n_matched,
            "n_unmatched": audit.time_alignment.n_unmatched,
            "median_s": audit.time_alignment.median_s,
            "p95_abs_s": audit.time_alignment.p95_abs_s,
            "max_abs_s": audit.time_alignment.max_abs_s,
            "rms_s": audit.time_alignment.rms_s,
            "max_abs_fraction_of_half_interval": (
                audit.time_alignment.max_abs_fraction_of_half_interval
            ),
            "n_outside_half_interval": audit.time_alignment.n_outside_half_interval,
            "max_pointing_interval_s": audit.time_alignment.max_pointing_interval_s,
        },
        "composition": {key: dict(value) for key, value in audit.composition.items()},
        "unique_times_by_ddid": {
            str(key): value for key, value in audit.unique_times_by_ddid.items()
        },
        "selected_main_rows": audit.selected_main_rows,
        "unique_time_count": audit.unique_time_count,
        "notes": list(audit.notes),
    }


def format_pointing_reconstruction_audit(audit: PointingReconstructionAudit) -> str:
    """Compact human-readable reconstruction report."""

    provenance = audit.provenance
    channels = ", ".join(
        f"DDID {channel.data_desc_id} SPW {channel.spectral_window_id} "
        f"ch {channel.channel} ({channel.frequency_hz / 1e9:.6f} GHz)"
        for channel in audit.native_channels
    )
    dense_cells = sum(1 for cell in audit.raster_cells if cell.family == "dense")
    sparse_cells = sum(1 for cell in audit.raster_cells if cell.family == "sparse")
    other_cells = sum(1 for cell in audit.raster_cells if cell.family not in {"dense", "sparse"})
    return "\n".join(
        (
            "THOL0001 pointing reconstruction audit",
            f"path project {provenance.get('project', '')} "
            f"{provenance.get('scheduling_block', '')} "
            f"{provenance.get('execution_block', '')}",
            f"OBSERVATION.PROJECT {provenance.get('observation_project', '') or '(absent)'}",
            f"field {audit.field_name}  column {audit.selected_column}",
            f"native channels {channels or '(none recorded)'}",
            f"moving antennas {', '.join(audit.moving_antenna_names) or '(none)'}",
            f"reference antennas {', '.join(audit.reference_antenna_names) or '(none)'}",
            f"raster cells dense={dense_cells} sparse={sparse_cells} other={other_cells} "
            f"occupancy_min={_occupancy_stat(audit.raster_cells, min)} "
            f"occupancy_max={_occupancy_stat(audit.raster_cells, max)}",
            f"Memo 195 dense {audit.memo195_dense_status} "
            f"({audit.cell_spacing_arcmin['dense']})  "
            f"sparse {audit.memo195_sparse_status} "
            f"({audit.cell_spacing_arcmin['sparse']})",
            f"sample reasons {audit.sample_reasons}",
            f"moving-reference row reasons {audit.row_reasons}",
            f"offset range arcmin {audit.offset_range_arcmin}",
            f"time alignment {audit.time_alignment}",
            f"overlap proof {audit.overlap_proof}",
            f"occupancy {getattr(audit.occupancy, 'explanation', None)}",
            f"composition {audit.composition}",
            f"unique times {audit.unique_time_count} by DDID {audit.unique_times_by_ddid}",
            f"selected MAIN rows {audit.selected_main_rows}",
        )
    )


def _select_main_rows(
    main: HolographyMainMetadata,
    *,
    field_name: str,
    data_desc_ids: tuple[int, ...] | None,
) -> NDArray[np.intp]:
    names = np.asarray(main.field_names, dtype=object)
    field_ids = np.flatnonzero(names == field_name)
    if field_ids.size == 0:
        raise ValueError(f"FIELD has no {field_name!r} entry")
    selected = np.isin(main.field_id, field_ids)
    if data_desc_ids is not None:
        selected &= np.isin(main.data_desc_id, np.asarray(data_desc_ids, dtype=np.int32))
    return np.flatnonzero(selected)


def _unique_time_state(
    unique_time_s: NDArray[np.float64],
    row_time_s: NDArray[np.float64],
    state_id: NDArray[np.int32],
    state_modes: tuple[str, ...],
) -> NDArray[np.str_]:
    labels = np.full(unique_time_s.size, "not_map", dtype="U16")
    time_index = np.searchsorted(unique_time_s, row_time_s)
    matched = (time_index < unique_time_s.size) & (
        unique_time_s[np.clip(time_index, 0, unique_time_s.size - 1)] == row_time_s
    )
    if not np.any(matched):
        return labels
    pairs = np.unique(
        np.stack((time_index[matched], state_id[matched]), axis=1),
        axis=0,
    )
    map_flag = np.array(
        [is_map_antenna_surface(_state_mode(int(state), state_modes)) for _, state in pairs],
        dtype=bool,
    )
    for index in np.unique(pairs[:, 0]):
        flags = map_flag[pairs[:, 0] == index]
        if flags.size and bool(np.all(flags)):
            labels[int(index)] = "map"
        elif bool(np.any(flags)) and not bool(np.all(flags)):
            labels[int(index)] = "mixed"
    return labels


def _state_mode(state_id: int, state_modes: tuple[str, ...]) -> str:
    if 0 <= state_id < len(state_modes):
        return state_modes[state_id]
    return ""


def _sample_reason_grid(valid: NDArray[np.bool_], settled: NDArray[np.bool_]) -> NDArray[np.str_]:
    reason = np.full(valid.shape, HolographyRowReason.MISSING_POINTING.value, dtype="U32")
    reason[valid] = HolographyRowReason.OK.value
    reason[valid & ~settled] = HolographyRowReason.DWELL_JUMP.value
    return reason


def _apply_transition_guards(
    settled: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    reason: NDArray[np.str_],
    guard_samples: int,
) -> tuple[NDArray[np.bool_], NDArray[np.str_]]:
    guarded = np.array(settled, copy=True)
    labeled = np.array(reason, copy=True)
    if int(guard_samples) <= 0:
        return guarded, labeled
    n_time = settled.shape[0]
    for antenna in range(settled.shape[1]):
        jumps = np.flatnonzero(valid[:, antenna] & ~settled[:, antenna])
        for time_index in jumps:
            for delta in range(-int(guard_samples), int(guard_samples) + 1):
                if delta == 0:
                    continue
                neighbor = int(time_index + delta)
                if neighbor < 0 or neighbor >= n_time:
                    continue
                if valid[neighbor, antenna] and guarded[neighbor, antenna]:
                    guarded[neighbor, antenna] = False
                    labeled[neighbor, antenna] = HolographyRowReason.TRANSITION_GUARD.value
    return guarded, labeled


def _apply_state_to_settled(
    settled: NDArray[np.bool_],
    reason: NDArray[np.str_],
    time_state: NDArray[np.str_],
) -> tuple[NDArray[np.bool_], NDArray[np.str_]]:
    guarded = np.array(settled, copy=True)
    labeled = np.array(reason, copy=True)
    for time_index, state in enumerate(time_state):
        if state == "map":
            continue
        code = (
            HolographyRowReason.MIXED_STATE.value
            if state == "mixed"
            else HolographyRowReason.NOT_MAP_ANTENNA_SURFACE.value
        )
        missing = labeled[time_index] == HolographyRowReason.MISSING_POINTING.value
        labeled[time_index, ~missing] = code
        guarded[time_index] = False
    return guarded, labeled


def _count_row_reasons(
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    time_index: NDArray[np.intp],
    resolved: ResolvedAntennaPointing,
    sample_reason: NDArray[np.str_],
) -> dict[str, int]:
    antenna_lookup = np.full(int(np.max(resolved.antenna_id)) + 1, -1, dtype=np.int32)
    antenna_lookup[resolved.antenna_id] = np.arange(resolved.antenna_id.size, dtype=np.int32)
    first_ok = (antenna1 >= 0) & (antenna1 < antenna_lookup.size)
    second_ok = (antenna2 >= 0) & (antenna2 < antenna_lookup.size)
    clipped_first = np.clip(antenna1, 0, antenna_lookup.size - 1)
    first_index = np.where(first_ok, antenna_lookup[clipped_first], -1)
    second_index = np.where(
        second_ok, antenna_lookup[np.clip(antenna2, 0, antenna_lookup.size - 1)], -1
    )
    missing = (first_index < 0) | (second_index < 0)
    safe_time = np.clip(time_index, 0, sample_reason.shape[0] - 1)
    safe_first = np.clip(first_index, 0, sample_reason.shape[1] - 1)
    safe_second = np.clip(second_index, 0, sample_reason.shape[1] - 1)
    first_reason = sample_reason[safe_time, safe_first]
    second_reason = sample_reason[safe_time, safe_second]
    chosen = np.full(antenna1.size, HolographyRowReason.MISSING_POINTING.value, dtype="U32")
    usable = ~missing
    if np.any(usable):
        chosen[usable] = _prior_reason_vector(first_reason[usable], second_reason[usable])
    ok = usable & (chosen == HolographyRowReason.OK.value)
    if np.any(ok):
        first_role = resolved.role[safe_time[ok], safe_first[ok]]
        second_role = resolved.role[safe_time[ok], safe_second[ok]]
        first_settled = resolved.settled[safe_time[ok], safe_first[ok]]
        second_settled = resolved.settled[safe_time[ok], safe_second[ok]]
        moving_ref = (
            (first_role == AntennaPointingRole.MOVING.value)
            & (second_role == AntennaPointingRole.REFERENCE.value)
        ) | (
            (first_role == AntennaPointingRole.REFERENCE.value)
            & (second_role == AntennaPointingRole.MOVING.value)
        )
        both_settled = first_settled & second_settled
        ok_indices = np.flatnonzero(ok)
        chosen[ok_indices[~moving_ref]] = HolographyRowReason.NOT_MOVING_REFERENCE.value
        chosen[ok_indices[moving_ref & ~both_settled]] = HolographyRowReason.UNSETTLED.value
        chosen[ok_indices[moving_ref & both_settled]] = HolographyRowReason.OK.value
    return _count_labels(chosen)


def _prior_reason(first: str, second: str) -> str:
    return str(_prior_reason_vector(np.array([first]), np.array([second]))[0])


def _prior_reason_vector(first: NDArray[np.str_], second: NDArray[np.str_]) -> NDArray[np.str_]:
    chosen = np.full(first.shape, HolographyRowReason.UNSETTLED.value, dtype="U32")
    for reason in reversed(_REASON_PRIORITY):
        chosen[(first == reason.value) | (second == reason.value)] = reason.value
    both_ok = (first == HolographyRowReason.OK.value) & (second == HolographyRowReason.OK.value)
    chosen[both_ok] = HolographyRowReason.OK.value
    return chosen


def _count_labels(labels: NDArray[np.str_]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(label) for label in labels)
    return dict(sorted(counts.items()))


def _role_antennas(resolved: ResolvedAntennaPointing) -> tuple[tuple[int, ...], tuple[int, ...]]:
    moving: list[int] = []
    reference: list[int] = []
    for index, antenna in enumerate(resolved.antenna_id):
        labels = resolved.role[resolved.valid[:, index], index]
        if np.any(labels == AntennaPointingRole.REFERENCE.value):
            reference.append(int(antenna))
        elif np.any(labels == AntennaPointingRole.MOVING.value) or np.any(
            labels == AntennaPointingRole.TRANSITION.value
        ):
            moving.append(int(antenna))
    return tuple(moving), tuple(reference)


def _merge_raster_cells(
    tracks: tuple,
    memo195: Memo195LowerCRaster,
    *,
    cluster_radius_arcmin: float,
) -> tuple[RasterCellOccupancy, ...]:
    cells: list[list[float]] = []
    counts: list[int] = []
    antennas: list[set[int]] = []
    families: list[list[str]] = []
    for track in tracks:
        if track.role != AntennaPointingRole.MOVING.value:
            continue
        family = _spacing_family(track.nearest_spacing_arcmin, memo195)
        for dwell, count in zip(track.dwell_lm_arcmin, track.sample_count, strict=True):
            assigned = False
            for index, cell in enumerate(cells):
                if float(np.hypot(dwell[0] - cell[0], dwell[1] - cell[1])) < cluster_radius_arcmin:
                    n_old = counts[index]
                    cells[index] = [
                        (cell[0] * n_old + float(dwell[0]) * int(count)) / (n_old + int(count)),
                        (cell[1] * n_old + float(dwell[1]) * int(count)) / (n_old + int(count)),
                    ]
                    counts[index] = n_old + int(count)
                    antennas[index].add(int(track.antenna_id))
                    families[index].append(family)
                    assigned = True
                    break
            if not assigned:
                cells.append([float(dwell[0]), float(dwell[1])])
                counts.append(int(count))
                antennas.append({int(track.antenna_id)})
                families.append([family])
    occupancy = []
    for cell, count, antenna_ids, family_votes in zip(
        cells, counts, antennas, families, strict=True
    ):
        occupancy.append(
            RasterCellOccupancy(
                l_arcmin=float(cell[0]),
                m_arcmin=float(cell[1]),
                sample_count=int(count),
                moving_antenna_ids=tuple(sorted(antenna_ids)),
                family=_majority(family_votes),
            )
        )
    occupancy.sort(key=lambda item: (item.family, item.l_arcmin, item.m_arcmin))
    return tuple(occupancy)


def _relabel_cells_by_nearest_spacing(
    cells: tuple[RasterCellOccupancy, ...],
    memo195: Memo195LowerCRaster,
    *,
    spacing_tolerance_arcmin: float = 0.2,
) -> tuple[RasterCellOccupancy, ...]:
    """Label merged cells from their own spacing, not per-antenna median spacing."""

    if len(cells) < 2:
        return cells
    coords = np.asarray([[cell.l_arcmin, cell.m_arcmin] for cell in cells], dtype=np.float64)
    relabeled: list[RasterCellOccupancy] = []
    for index, cell in enumerate(cells):
        delta = np.hypot(coords[:, 0] - coords[index, 0], coords[:, 1] - coords[index, 1])
        delta[index] = np.inf
        nearest = _spacing_family(
            float(np.min(delta)),
            memo195,
            spacing_tolerance_arcmin=spacing_tolerance_arcmin,
        )
        relabeled.append(replace(cell, family=nearest if nearest != "other" else cell.family))
    relabeled.sort(key=lambda item: (item.family, item.l_arcmin, item.m_arcmin))
    return tuple(relabeled)


def _raster_from_cells(
    cells: tuple[RasterCellOccupancy, ...],
    memo195: Memo195LowerCRaster,
    *,
    spacing_tolerance_arcmin: float = 0.2,
) -> RasterGeometryEstimate:
    if len(cells) < 2:
        return RasterGeometryEstimate((), None, None, sum(cell.sample_count for cell in cells))
    coords = np.asarray([[cell.l_arcmin, cell.m_arcmin] for cell in cells], dtype=np.float64)
    spacings: list[float] = []
    for index, cell in enumerate(coords):
        delta = np.hypot(coords[:, 0] - cell[0], coords[:, 1] - cell[1])
        delta[index] = np.inf
        spacings.append(float(np.min(delta)))
    dense = [
        spacing
        for spacing in spacings
        if abs(spacing - memo195.dense_spacing_arcmin) <= spacing_tolerance_arcmin
    ]
    sparse = [
        spacing
        for spacing in spacings
        if abs(spacing - memo195.sparse_spacing_arcmin) <= spacing_tolerance_arcmin
    ]
    return RasterGeometryEstimate(
        spacings_arcmin=tuple(spacings),
        dense_spacing_arcmin=float(np.mean(dense)) if dense else None,
        sparse_spacing_arcmin=float(np.mean(sparse)) if sparse else None,
        n_dwell=int(sum(cell.sample_count for cell in cells)),
    )


def _combine_raster_estimates(
    tracks: RasterGeometryEstimate,
    cells: RasterGeometryEstimate,
) -> RasterGeometryEstimate:
    """Prefer per-antenna track spacings; fill a missing population from cell NNs."""

    spacings = tuple(tracks.spacings_arcmin) + tuple(cells.spacings_arcmin)
    return RasterGeometryEstimate(
        spacings_arcmin=spacings,
        dense_spacing_arcmin=tracks.dense_spacing_arcmin or cells.dense_spacing_arcmin,
        sparse_spacing_arcmin=tracks.sparse_spacing_arcmin or cells.sparse_spacing_arcmin,
        n_dwell=max(tracks.n_dwell, cells.n_dwell),
    )


def _spacing_family(
    spacing_arcmin: float | None,
    memo195: Memo195LowerCRaster,
    *,
    spacing_tolerance_arcmin: float = 0.2,
) -> str:
    if spacing_arcmin is None:
        return "other"
    if abs(float(spacing_arcmin) - memo195.dense_spacing_arcmin) <= spacing_tolerance_arcmin:
        return "dense"
    if abs(float(spacing_arcmin) - memo195.sparse_spacing_arcmin) <= spacing_tolerance_arcmin:
        return "sparse"
    return "other"


def _majority(votes: list[str]) -> str:
    if not votes:
        return "other"
    counts = Counter(votes)
    return sorted(counts, key=lambda name: (-counts[name], name))[0]


def _offset_range_arcmin(resolved: ResolvedAntennaPointing) -> dict[str, float]:
    moving = resolved.valid & (
        (resolved.role == AntennaPointingRole.MOVING.value)
        | (resolved.role == AntennaPointingRole.TRANSITION.value)
    )
    if not np.any(moving):
        return {
            "l_min": float("nan"),
            "l_max": float("nan"),
            "m_min": float("nan"),
            "m_max": float("nan"),
            "radius_max": float("nan"),
        }
    offsets = np.rad2deg(resolved.offset_lm_rad[moving]) * 60.0
    radius = np.hypot(offsets[:, 0], offsets[:, 1])
    return {
        "l_min": float(np.min(offsets[:, 0])),
        "l_max": float(np.max(offsets[:, 0])),
        "m_min": float(np.min(offsets[:, 1])),
        "m_max": float(np.max(offsets[:, 1])),
        "radius_max": float(np.max(radius)),
    }


def _summarize_residuals(
    residual_s: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> TimeAlignmentResiduals:
    matched = residual_s[valid & np.isfinite(residual_s)]
    n_unmatched = int(np.sum(~valid))
    if matched.size == 0:
        return TimeAlignmentResiduals(
            0, n_unmatched, float("nan"), float("nan"), float("nan"), float("nan")
        )
    abs_residual = np.abs(matched)
    return TimeAlignmentResiduals(
        n_matched=int(matched.size),
        n_unmatched=n_unmatched,
        median_s=float(np.median(matched)),
        p95_abs_s=float(np.percentile(abs_residual, 95)),
        max_abs_s=float(np.max(abs_residual)),
        rms_s=float(np.sqrt(np.mean(matched * matched))),
    )


def _composition(
    main: HolographyMainMetadata, selected: NDArray[np.intp]
) -> dict[str, dict[str, int]]:
    fields = Counter(
        _name(int(field_id), main.field_names, "FIELD") for field_id in main.field_id[selected]
    )
    states = Counter(
        _state_mode(int(state_id), main.state_modes) or f"STATE_{state_id}"
        for state_id in main.state_id[selected]
    )
    scans = Counter(str(int(scan)) for scan in main.scan_number[selected])
    ddids = Counter(str(int(ddid)) for ddid in main.data_desc_id[selected])
    return {
        "field": dict(sorted(fields.items())),
        "state": dict(sorted(states.items())),
        "scan": dict(sorted(scans.items(), key=lambda item: int(item[0]))),
        "data_desc_id": dict(sorted(ddids.items(), key=lambda item: int(item[0]))),
    }


def _name(index: int, names: tuple[str, ...], prefix: str) -> str:
    if 0 <= index < len(names):
        return names[index]
    return f"{prefix}_{index}"


def _names_for(ids: tuple[int, ...], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_name(int(antenna), names, "ea") for antenna in ids)


def _occupancy_stat(cells: tuple[RasterCellOccupancy, ...], reducer) -> int | str:
    if not cells:
        return "n/a"
    return int(reducer(cell.sample_count for cell in cells))
