"""Derived commissioning Measurement Set and pre-calibration visibility audit.

The source archive MS is not modified. Calibration and Jones recovery stay
out of this module.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.holography import (
    THOL0001_HOLORASTER_FIELD,
    AntennaPointingRole,
    holography_execution_provenance,
)
from sl1mjax.holography_ms import (
    _read_antennas,
    _read_channel_frequencies,
    _read_data_descriptions,
    _read_field_names,
    _read_field_phase_centre,
    _read_observation,
    _read_state_modes,
    _tables,
    _taql_int_list,
    is_casacore_measurement_set,
    read_pointing_table_from_ms,
)
from sl1mjax.holography_pointing_audit import (
    reconstruct_holography_pointing,
    select_native_channels,
)

COMMISSIONING_SPWS = (4, 5)
ON_AXIS_3C147_NAMES = ("J0542+4951", "3C147", "3C 147")
D_SCAN_PREFIX = "C147-"
LEAKAGE_STATE_TOKEN = "CALIBRATE_POL"
EVPA_FIELD_NAMES = ("J1331+3030", "3C286", "3C 286")


@dataclass(frozen=True)
class CommissioningSelection:
    """Immutable subset specification. All 64 native channels are kept."""

    field_ids: tuple[int, ...]
    field_names: tuple[str, ...]
    data_desc_ids: tuple[int, ...]
    spectral_window_ids: tuple[int, ...]
    source_ms: str
    dest_ms: str
    provenance: dict[str, str]


def select_commissioning_fields(field_names: tuple[str, ...]) -> tuple[int, ...]:
    """On-axis 3C147, C147-* D-scan, HOLORASTER, and 3C286."""

    selected: list[int] = []
    for index, name in enumerate(field_names):
        if name == THOL0001_HOLORASTER_FIELD:
            selected.append(index)
        elif name.startswith(D_SCAN_PREFIX):
            selected.append(index)
        elif name in ON_AXIS_3C147_NAMES or name in EVPA_FIELD_NAMES:
            selected.append(index)
    if not selected:
        raise ValueError("commissioning fields were not found in FIELD.NAME")
    return tuple(selected)


def copy_commissioning_measurement_set(
    source: Path,
    destination: Path,
    *,
    spectral_window_ids: tuple[int, ...] = COMMISSIONING_SPWS,
    immutable: bool = True,
    kind: str = "thol0001_lower_c_commissioning_ms",
) -> CommissioningSelection:
    """Copy selected MAIN rows to a new MS. The source is opened read-only."""

    tables = _tables()
    measurement_set = Path(source)
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(f"commissioning MS already exists: {dest}")
    if not is_casacore_measurement_set(measurement_set):
        raise ValueError(f"not a casacore Measurement Set: {measurement_set}")
    observation = _read_observation(tables, measurement_set)
    provenance = holography_execution_provenance(
        measurement_set, observation_project=observation.get("project") or ""
    )
    field_names = _read_field_names(tables, measurement_set)
    field_ids = select_commissioning_fields(field_names)
    data_descriptions = _read_data_descriptions(tables, measurement_set)
    ddids = tuple(
        ddid for ddid, spw, _pol in data_descriptions if int(spw) in set(spectral_window_ids)
    )
    if not ddids:
        raise ValueError(f"no DATA_DESC_ID maps to SPWs {spectral_window_ids}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    query = f"FIELD_ID IN {_taql_int_list(field_ids)} && DATA_DESC_ID IN {_taql_int_list(ddids)}"
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        selected = main.query(query)
        try:
            selected.copy(str(dest), deep=True)
        finally:
            selected.close()
    if not is_casacore_measurement_set(dest):
        raise RuntimeError(f"commissioning copy did not produce an MS: {dest}")
    sidecar = dest.with_name(dest.name + ".provenance.json")
    selection = CommissioningSelection(
        field_ids=field_ids,
        field_names=tuple(field_names[index] for index in field_ids),
        data_desc_ids=ddids,
        spectral_window_ids=tuple(spectral_window_ids),
        source_ms=str(measurement_set),
        dest_ms=str(dest),
        provenance=dict(provenance),
    )
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": kind,
                "immutable": bool(immutable),
                "n_native_channels_kept": 64,
                "correlations": ("RR", "RL", "LR", "LL"),
                "channel_32_only": False,
                **asdict(selection),
            },
            indent=2,
        )
        + "\n"
    )
    if immutable:
        _make_tree_immutable(dest)
    return selection


def _make_tree_immutable(path: Path) -> None:
    root = Path(path)
    for directory, _dirs, files in os.walk(root):
        os.chmod(directory, 0o555)
        for name in files:
            os.chmod(os.path.join(directory, name), 0o444)


def audit_commissioning_visibilities(
    path: Path,
    *,
    chunk_rows: int = 32768,
    pointing_column: str = "POINTING_OFFSET",
) -> dict[str, object]:
    """Pre-calibration visibility audit. Does not apply or solve calibration."""

    tables = _tables()
    measurement_set = Path(path)
    field_names = _read_field_names(tables, measurement_set)
    state_modes = _read_state_modes(tables, measurement_set)
    antennas = _read_antennas(tables, measurement_set)
    antenna_names = antennas[1]
    observation = _read_observation(tables, measurement_set)
    provenance = holography_execution_provenance(
        measurement_set, observation_project=observation.get("project") or ""
    )
    frequencies = _read_channel_frequencies(tables, measurement_set)
    data_descriptions = _read_data_descriptions(tables, measurement_set)
    ddid_to_spw = {int(ddid): int(spw) for ddid, spw, _pol in data_descriptions}
    pointing = read_pointing_table_from_ms(measurement_set)
    holoraster_ids = tuple(
        index for index, name in enumerate(field_names) if name == THOL0001_HOLORASTER_FIELD
    )
    phase = (
        _read_field_phase_centre(tables, measurement_set, THOL0001_HOLORASTER_FIELD)
        if holoraster_ids
        else (0.0, 0.0)
    )
    native = select_native_channels(frequencies, data_descriptions)
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        row_count = int(main.nrows())
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
        antenna1 = np.asarray(main.getcol("ANTENNA1"), dtype=np.int32)
        antenna2 = np.asarray(main.getcol("ANTENNA2"), dtype=np.int32)
        field_id = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        state_id = np.asarray(main.getcol("STATE_ID"), dtype=np.int32)
        scan = np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
        n_chan, n_corr, columns = _visibility_shape(main)
        group_keys, group_id = _factorize_groups(scan, field_id, state_id, ddid)
        stats = _chunk_visibility_stats(
            main,
            row_count,
            int(chunk_rows),
            antenna1=antenna1,
            antenna2=antenna2,
            group_id=group_id,
            n_group=group_keys.shape[0],
            n_ant=int(max(np.max(antenna1), np.max(antenna2)) + 1),
            n_chan=n_chan,
            n_corr=n_corr,
            columns=columns,
        )
    roles = None
    if holoraster_ids:
        from sl1mjax.holography_ms import _read_selected_main_metadata

        main_meta = _read_selected_main_metadata(
            tables,
            measurement_set,
            field_ids=holoraster_ids,
            data_desc_ids=tuple(sorted(np.unique(ddid).tolist())),
            field_names=field_names,
            state_modes=state_modes,
            antenna_names=antenna_names,
            phase_centre_rad=phase,
        )
        reconstructed = reconstruct_holography_pointing(
            pointing,
            main_meta,
            field_name=THOL0001_HOLORASTER_FIELD,
            selected_column=pointing_column,
            provenance=provenance,
            native_channels=native,
        )
        roles = reconstructed.resolved
    baseline_kind = _baseline_kinds(antenna1, antenna2, time_s, roles)
    groups = _group_reports(
        group_keys,
        group_id,
        time_s,
        field_names,
        state_modes,
        ddid_to_spw,
        baseline_kind,
        stats,
        n_corr=n_corr,
    )
    return {
        "schema_version": 1,
        "provenance": dict(provenance),
        "path": str(measurement_set),
        "n_rows": row_count,
        "n_channels": n_chan,
        "n_correlations": n_corr,
        "columns": sorted(columns),
        "field_names": list(field_names),
        "state_modes": list(state_modes),
        "antenna_names": list(antenna_names),
        "native_channels": [
            {
                "data_desc_id": item.data_desc_id,
                "spectral_window_id": item.spectral_window_id,
                "channel": item.channel,
                "frequency_hz": item.frequency_hz,
            }
            for item in native
        ],
        "d_scan": _d_scan_report(field_names, state_modes, field_id, state_id, scan),
        "three_c286": _parallactic_report(
            tables, measurement_set, field_names, field_id, time_s, antennas[2]
        ),
        "groups": groups,
        "flag_fraction_by_antenna": {
            (antenna_names[ant] if ant < len(antenna_names) else str(ant)): value
            for ant, value in stats["flag_fraction_by_antenna"].items()
        },
        "flag_fraction_by_correlation": stats["flag_fraction_by_correlation"],
        "mean_weight_by_correlation": stats["mean_weight_by_correlation"],
        "cross_hands": stats["cross_hands"],
        "baseline_kind_totals": dict(sorted(Counter(map(str, baseline_kind)).items())),
        "moving_antennas_by_raster_scan": _moving_by_scan(roles, scan, time_s, antenna_names)
        if roles is not None
        else {},
        "notes": (
            "source MS was not modified",
            "all 64 native channels in SPWs 4 and 5 are present; channel 32 is not isolated",
            "WEIGHT_SPECTRUM is present but undefined; WEIGHT is the populated original weight",
            "calibration has not been applied",
        ),
    }


def _visibility_shape(main: Any) -> tuple[int, int, set[str]]:
    columns = set(main.colnames())
    flag = np.asarray(main.getcol("FLAG", 0, 1), dtype=bool)
    if flag.ndim == 2:
        return int(flag.shape[0]), int(flag.shape[1]), columns
    return int(flag.shape[1]), int(flag.shape[2]), columns


def _factorize_groups(
    scan: NDArray[np.int32],
    field_id: NDArray[np.int32],
    state_id: NDArray[np.int32],
    ddid: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    keys = np.stack((scan, field_id, state_id, ddid), axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    return unique_keys.astype(np.int32), inverse.astype(np.int32)


def _chunk_visibility_stats(
    main: Any,
    n_row: int,
    chunk_rows: int,
    *,
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    group_id: NDArray[np.int32],
    n_group: int,
    n_ant: int,
    n_chan: int,
    n_corr: int,
    columns: set[str],
) -> dict[str, object]:
    names = ("RR", "RL", "LR", "LL")
    flag_ant = np.zeros(n_ant, dtype=np.int64)
    vis_ant = np.zeros(n_ant, dtype=np.int64)
    flag_corr = np.zeros(n_corr, dtype=np.int64)
    vis_corr = np.zeros(n_corr, dtype=np.int64)
    weight_corr = np.zeros(n_corr, dtype=np.float64)
    weight_count = np.zeros(n_corr, dtype=np.int64)
    cross_amp = np.zeros(2, dtype=np.float64)
    cross_w = np.zeros(2, dtype=np.float64)
    cross_n = np.zeros(2, dtype=np.int64)
    group_flag = np.zeros((n_group, n_corr), dtype=np.int64)
    group_vis = np.zeros((n_group, n_corr), dtype=np.int64)
    group_weight = np.zeros((n_group, n_corr), dtype=np.float64)
    group_cross_amp = np.zeros((n_group, 2), dtype=np.float64)
    group_cross_w = np.zeros((n_group, 2), dtype=np.float64)
    group_cross_n = np.zeros((n_group, 2), dtype=np.int64)
    vis_per_row = n_chan * n_corr
    for start in range(0, n_row, chunk_rows):
        stop = min(start + chunk_rows, n_row)
        flag = np.asarray(main.getcol("FLAG", start, stop - start), dtype=bool)
        if flag.ndim == 2:
            flag = flag[:, None, :]
        chunk_groups = group_id[start:stop]
        flagged_rows = np.sum(flag, axis=(1, 2))
        vis_corr += int(flag.shape[0] * flag.shape[1])
        flag_corr += np.sum(flag, axis=(0, 1))
        for antenna in (antenna1[start:stop], antenna2[start:stop]):
            np.add.at(vis_ant, antenna, vis_per_row)
            np.add.at(flag_ant, antenna, flagged_rows)
        weight = None
        if "WEIGHT_SPECTRUM" in columns and main.iscelldefined("WEIGHT_SPECTRUM", start):
            try:
                weight = np.asarray(
                    main.getcol("WEIGHT_SPECTRUM", start, stop - start), dtype=np.float64
                )
            except RuntimeError:
                weight = None
            if weight is not None and weight.ndim == 2:
                weight = weight[:, None, :]
        if weight is None and "WEIGHT" in columns:
            weight = np.asarray(main.getcol("WEIGHT", start, stop - start), dtype=np.float64)
            weight = np.broadcast_to(weight[:, None, :], flag.shape)
        if weight is None:
            weight = np.ones(flag.shape, dtype=np.float64)
        usable = ~flag
        for corr in range(n_corr):
            n_flag = np.sum(flag[..., corr], axis=1)
            n_vis = np.full(stop - start, flag.shape[1], dtype=np.int64)
            np.add.at(group_flag[:, corr], chunk_groups, n_flag)
            np.add.at(group_vis[:, corr], chunk_groups, n_vis)
            w_sum = np.sum(weight[..., corr] * usable[..., corr], axis=1)
            np.add.at(group_weight[:, corr], chunk_groups, w_sum)
            weight_corr[corr] += float(np.sum(w_sum))
            weight_count[corr] += int(np.sum(usable[..., corr]))
        if "DATA" in columns:
            data = np.asarray(main.getcol("DATA", start, stop - start))
            if data.ndim == 2:
                data = data[:, None, :]
            for corr_index, corr in enumerate((1, 2)):
                if corr >= n_corr:
                    continue
                amp = np.abs(data[..., corr]) * usable[..., corr]
                w = weight[..., corr] * usable[..., corr]
                cross_amp[corr_index] += float(np.sum(amp * w))
                cross_w[corr_index] += float(np.sum(w))
                cross_n[corr_index] += int(np.sum(usable[..., corr]))
                np.add.at(group_cross_amp[:, corr_index], chunk_groups, np.sum(amp * w, axis=1))
                np.add.at(group_cross_w[:, corr_index], chunk_groups, np.sum(w, axis=1))
                np.add.at(
                    group_cross_n[:, corr_index],
                    chunk_groups,
                    np.sum(usable[..., corr], axis=1),
                )
    by_antenna = {
        ant: (float(flag_ant[ant] / vis_ant[ant]) if vis_ant[ant] else 0.0) for ant in range(n_ant)
    }
    by_corr = {
        names[corr]: (float(flag_corr[corr] / vis_corr[corr]) if vis_corr[corr] else 0.0)
        for corr in range(n_corr)
        if vis_corr[corr]
    }
    mean_weight = {
        names[corr]: (float(weight_corr[corr] / weight_count[corr]) if weight_count[corr] else 0.0)
        for corr in range(n_corr)
        if weight_count[corr]
    }
    cross = {
        "RL": {
            "weighted_mean_amp": float(cross_amp[0] / cross_w[0]) if cross_w[0] else 0.0,
            "mean_weight": float(cross_w[0] / cross_n[0]) if cross_n[0] else 0.0,
            "n": int(cross_n[0]),
        },
        "LR": {
            "weighted_mean_amp": float(cross_amp[1] / cross_w[1]) if cross_w[1] else 0.0,
            "mean_weight": float(cross_w[1] / cross_n[1]) if cross_n[1] else 0.0,
            "n": int(cross_n[1]),
        },
    }
    return {
        "flag_fraction_by_antenna": by_antenna,
        "flag_fraction_by_correlation": by_corr,
        "mean_weight_by_correlation": mean_weight,
        "cross_hands": cross,
        "group_flag": group_flag,
        "group_vis": group_vis,
        "group_weight": group_weight,
        "group_cross_amp": group_cross_amp,
        "group_cross_w": group_cross_w,
        "group_cross_n": group_cross_n,
    }


def _baseline_kinds(
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    time_s: NDArray[np.float64],
    roles: Any,
) -> NDArray[np.str_]:
    kind = np.full(antenna1.size, "on_axis", dtype="U24")
    if roles is None:
        return kind
    lookup = np.full(int(np.max(roles.antenna_id)) + 1, -1, dtype=np.int32)
    lookup[roles.antenna_id] = np.arange(roles.antenna_id.size, dtype=np.int32)
    time_index = np.searchsorted(roles.unique_time_s, time_s)
    in_range = time_index < roles.unique_time_s.size
    matched = in_range & (
        roles.unique_time_s[np.clip(time_index, 0, max(roles.unique_time_s.size - 1, 0))] == time_s
    )
    first_ok = matched & (antenna1 >= 0) & (antenna1 < lookup.size)
    second_ok = matched & (antenna2 >= 0) & (antenna2 < lookup.size)
    first = np.where(first_ok, lookup[np.clip(antenna1, 0, lookup.size - 1)], -1)
    second = np.where(second_ok, lookup[np.clip(antenna2, 0, lookup.size - 1)], -1)
    usable = (first >= 0) & (second >= 0)
    if not np.any(usable):
        return kind
    safe_time = np.clip(time_index, 0, roles.role.shape[0] - 1)
    first_role = roles.role[safe_time[usable], first[usable]]
    second_role = roles.role[safe_time[usable], second[usable]]
    rows = np.flatnonzero(usable)
    ref = AntennaPointingRole.REFERENCE.value
    mov = AntennaPointingRole.MOVING.value
    kind[rows[(first_role == ref) & (second_role == ref)]] = "reference_reference"
    kind[rows[(first_role == mov) & (second_role == mov)]] = "moving_moving"
    moving_ref = ((first_role == mov) & (second_role == ref)) | (
        (first_role == ref) & (second_role == mov)
    )
    kind[rows[moving_ref]] = "moving_reference"
    assigned = ((first_role == ref) | (first_role == mov)) & (
        (second_role == ref) | (second_role == mov)
    )
    kind[rows[~assigned]] = "other"
    return kind


def _group_reports(
    group_keys: NDArray[np.int32],
    group_id: NDArray[np.int32],
    time_s: NDArray[np.float64],
    field_names: tuple[str, ...],
    state_modes: tuple[str, ...],
    ddid_to_spw: dict[int, int],
    baseline_kind: NDArray[np.str_],
    stats: dict[str, object],
    *,
    n_corr: int,
) -> list[dict[str, object]]:
    names = ("RR", "RL", "LR", "LL")
    groups: list[dict[str, object]] = []
    kind_names, kind_codes = np.unique(baseline_kind, return_inverse=True)
    n_kind = kind_names.size
    kind_counts = np.zeros((group_keys.shape[0], n_kind), dtype=np.int64)
    np.add.at(kind_counts, (group_id, kind_codes), 1)
    n_rows = np.bincount(group_id, minlength=group_keys.shape[0])
    unique_times = np.zeros(group_keys.shape[0], dtype=np.int64)
    for index in range(group_keys.shape[0]):
        unique_times[index] = np.unique(time_s[group_id == index]).size
    group_flag = stats["group_flag"]
    group_vis = stats["group_vis"]
    group_weight = stats["group_weight"]
    group_cross_amp = stats["group_cross_amp"]
    group_cross_w = stats["group_cross_w"]
    group_cross_n = stats["group_cross_n"]
    for index, (scan_n, field_n, state_n, ddid_n) in enumerate(group_keys):
        kinds = {
            str(kind_names[kind]): int(kind_counts[index, kind])
            for kind in range(n_kind)
            if kind_counts[index, kind]
        }
        active = {
            names[corr]: int(group_vis[index, corr] - group_flag[index, corr])
            for corr in range(n_corr)
        }
        flag_fraction = {
            names[corr]: (
                float(group_flag[index, corr] / group_vis[index, corr])
                if group_vis[index, corr]
                else 0.0
            )
            for corr in range(n_corr)
        }
        mean_weight = {}
        for corr in range(n_corr):
            unflagged = group_vis[index, corr] - group_flag[index, corr]
            mean_weight[names[corr]] = (
                float(group_weight[index, corr] / unflagged) if unflagged else 0.0
            )
        groups.append(
            {
                "scan": int(scan_n),
                "field_id": int(field_n),
                "field": field_names[int(field_n)] if 0 <= field_n < len(field_names) else "",
                "state_id": int(state_n),
                "state": state_modes[int(state_n)] if 0 <= state_n < len(state_modes) else "",
                "data_desc_id": int(ddid_n),
                "spw": int(ddid_to_spw.get(int(ddid_n), -1)),
                "n_rows": int(n_rows[index]),
                "n_unique_times": int(unique_times[index]),
                "n_active_correlations": active,
                "baseline_kinds": dict(sorted(kinds.items())),
                "flag_fraction_by_correlation": flag_fraction,
                "mean_weight_by_correlation": mean_weight,
                "cross_hands": {
                    "RL": {
                        "weighted_mean_amp": (
                            float(group_cross_amp[index, 0] / group_cross_w[index, 0])
                            if group_cross_w[index, 0]
                            else 0.0
                        ),
                        "mean_weight": (
                            float(group_cross_w[index, 0] / group_cross_n[index, 0])
                            if group_cross_n[index, 0]
                            else 0.0
                        ),
                        "n": int(group_cross_n[index, 0]),
                    },
                    "LR": {
                        "weighted_mean_amp": (
                            float(group_cross_amp[index, 1] / group_cross_w[index, 1])
                            if group_cross_w[index, 1]
                            else 0.0
                        ),
                        "mean_weight": (
                            float(group_cross_w[index, 1] / group_cross_n[index, 1])
                            if group_cross_n[index, 1]
                            else 0.0
                        ),
                        "n": int(group_cross_n[index, 1]),
                    },
                },
            }
        )
    groups.sort(key=lambda item: (int(item["scan"]), int(item["data_desc_id"])))
    return groups


def _d_scan_report(
    field_names: tuple[str, ...],
    state_modes: tuple[str, ...],
    field_id: NDArray[np.int32],
    state_id: NDArray[np.int32],
    scan: NDArray[np.int32],
) -> dict[str, object]:
    d_fields = [name for name in field_names if name.startswith(D_SCAN_PREFIX)]
    expected = [f"C147-{tag}" for tag in ("N", "NW", "W", "SW", "S", "SE", "E", "NE")]
    rows = {
        name: int(np.sum(field_id == field_names.index(name)))
        for name in d_fields
        if name in field_names
    }
    states = (
        sorted(
            {
                state_modes[int(state)]
                for state in np.unique(
                    state_id[np.isin(field_id, [field_names.index(n) for n in d_fields])]
                )
                if 0 <= int(state) < len(state_modes)
            }
        )
        if d_fields
        else []
    )
    scans = (
        sorted(
            {
                int(value)
                for value in np.unique(
                    scan[np.isin(field_id, [field_names.index(n) for n in d_fields])]
                )
            }
        )
        if d_fields
        else []
    )
    return {
        "fields": d_fields,
        "expected_octant_fields": expected,
        "octants_complete": d_fields == expected,
        "n_rows_by_field": rows,
        "states": states,
        "scans": scans,
        "looks_like_intended_leakage_scan": bool(d_fields == expected and rows),
        "notes": (
            "C147-* is the eight-position offset ring around 3C147; "
            "STATE does not name CALIBRATE_POL, so leakage intent "
            "is inferred from the field pattern"
            if d_fields == expected
            else "C147-* octant set is incomplete"
        ),
    }


def _parallactic_report(
    tables: Any,
    measurement_set: Path,
    field_names: tuple[str, ...],
    field_id: NDArray[np.int32],
    time_s: NDArray[np.float64],
    antenna_position_m: tuple[tuple[float, float, float], ...],
) -> dict[str, object]:
    indices = [index for index, name in enumerate(field_names) if name in EVPA_FIELD_NAMES]
    if not indices:
        return {"present": False, "notes": "3C286 field is absent"}
    times = np.unique(time_s[np.isin(field_id, np.asarray(indices, dtype=np.int32))])
    if times.size == 0:
        return {"present": True, "n_times": 0, "notes": "3C286 field has no MAIN rows"}
    phase = _read_field_phase_centre(tables, measurement_set, field_names[indices[0]])
    from sl1mjax.calibration_terms import parallactic_angle_rad

    position = np.asarray(antenna_position_m, dtype=np.float64)[:1]
    chi = parallactic_angle_rad(times, phase, position)[:, 0]
    return {
        "present": True,
        "field_ids": indices,
        "field_names": [field_names[index] for index in indices],
        "n_times": int(times.size),
        "chi_rad_min": float(np.min(chi)),
        "chi_rad_max": float(np.max(chi)),
        "chi_rad_span": float(np.max(chi) - np.min(chi)),
        "chi_deg_span": float(np.rad2deg(np.max(chi) - np.min(chi))),
    }


def _moving_by_scan(
    roles: Any,
    scan: NDArray[np.int32],
    time_s: NDArray[np.float64],
    antenna_names: tuple[str, ...],
) -> dict[str, list[str]]:
    if roles is None:
        return {}
    time_index = np.searchsorted(roles.unique_time_s, time_s)
    in_range = time_index < roles.unique_time_s.size
    matched = in_range & (
        roles.unique_time_s[np.clip(time_index, 0, max(roles.unique_time_s.size - 1, 0))] == time_s
    )
    if not np.any(matched):
        return {}
    pairs = np.unique(np.stack((scan[matched], time_index[matched]), axis=1), axis=0)
    moving = roles.role == AntennaPointingRole.MOVING.value
    by_scan: dict[str, set[str]] = {}
    for scan_n, tidx in pairs:
        names = by_scan.setdefault(str(int(scan_n)), set())
        for ant_index in np.flatnonzero(moving[int(tidx)]):
            antenna = int(roles.antenna_id[int(ant_index)])
            if 0 <= antenna < len(antenna_names):
                names.add(antenna_names[antenna])
            else:
                names.add(str(antenna))
    return {
        key: sorted(value) for key, value in sorted(by_scan.items(), key=lambda item: int(item[0]))
    }


def write_visibility_audit(report: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return destination
