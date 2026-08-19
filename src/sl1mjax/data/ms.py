"""Optional MeasurementSet-to-canonical extractor."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
from sl1mjax.data.metadata import (
    AntennaRecord,
    CalibratorRole,
    FeedRecord,
    FieldRecord,
    ObservationMetadata,
    ObservationRecord,
    StateRecord,
)
from sl1mjax.polarization import Correlation, ReceptorBasis

_CORRELATION_CODES = {
    1: Correlation.I,
    2: Correlation.Q,
    3: Correlation.U,
    4: Correlation.V,
    5: Correlation.RR,
    6: Correlation.RL,
    7: Correlation.LR,
    8: Correlation.LL,
    9: Correlation.XX,
    10: Correlation.XY,
    11: Correlation.YX,
    12: Correlation.YY,
}


def _tables() -> Any:
    try:
        from casacore import tables
    except ImportError as exc:  # pragma: no cover - system-dependent optional extra
        raise RuntimeError(
            "MeasurementSet extraction requires `uv sync --extra ms` and system casacore"
        ) from exc
    return tables


def _basis(correlations: tuple[Correlation, ...]) -> ReceptorBasis:
    if all(value.value.startswith(("X", "Y")) for value in correlations):
        return ReceptorBasis.LINEAR
    if all(value.value.startswith(("R", "L")) for value in correlations):
        return ReceptorBasis.CIRCULAR
    stokes = {Correlation.I, Correlation.Q, Correlation.U, Correlation.V}
    if all(value in stokes for value in correlations):
        return ReceptorBasis.STOKES
    raise ValueError(f"mixed or unsupported correlation products: {correlations}")


def _read_column(table: Any, name: str, dtype: Any | None = None) -> Any:
    """Read a fixed-shape column or preserve variable-shaped cells by row."""

    try:
        return np.asarray(table.getcol(name), dtype=dtype)
    except (RuntimeError, ValueError):
        return tuple(
            np.asarray(table.getcell(name, row), dtype=dtype)
            for row in range(table.nrows())
        )


def _select_rows(values: Any, selected: np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.asarray(values[selected])
    return np.stack([values[row] for row in np.flatnonzero(selected)])


def _direction(value: Any) -> tuple[float, float]:
    direction = np.asarray(value, dtype=np.float64).reshape(-1, 2)[0]
    return float(direction[0]), float(direction[1])


def _intent_names(observation_mode: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.split("#", 1)[0].strip().lower()
                for item in observation_mode.split(",")
                if item.strip()
            }
        )
    )


_INTENT_ROLES = {
    "observe_target": CalibratorRole.TARGET,
    "calibrate_flux": CalibratorRole.FLUX,
    "calibrate_bandpass": CalibratorRole.BANDPASS,
    "calibrate_phase": CalibratorRole.PHASE,
    "calibrate_ampli": CalibratorRole.AMPLITUDE,
    "calibrate_polarization": CalibratorRole.POLARIZATION,
    "calibrate_pointing": CalibratorRole.POINTING,
}


def _extract_metadata(
    tables: Any,
    source: Path,
    field_id: np.ndarray,
    state_id: np.ndarray,
    *,
    role_overrides: Mapping[str, tuple[CalibratorRole, ...]] | None,
) -> ObservationMetadata:
    antennas: list[AntennaRecord] = []
    try:
        with tables.table(str(source / "ANTENNA"), readonly=True, ack=False) as table:
            columns = set(table.colnames())
            for row in range(table.nrows()):
                position = np.asarray(table.getcell("POSITION", row)).reshape(3)
                antennas.append(
                    AntennaRecord(
                        antenna_id=row,
                        name=str(table.getcell("NAME", row)),
                        station=(
                            str(table.getcell("STATION", row))
                            if "STATION" in columns
                            else ""
                        ),
                        position_m=(
                            float(position[0]),
                            float(position[1]),
                            float(position[2]),
                        ),
                        dish_diameter_m=(
                            float(table.getcell("DISH_DIAMETER", row))
                            if "DISH_DIAMETER" in columns
                            else np.nan
                        ),
                        mount=(
                            str(table.getcell("MOUNT", row))
                            if "MOUNT" in columns
                            else ""
                        ),
                    )
                )
    except (KeyError, RuntimeError):
        pass

    states: list[StateRecord] = []
    try:
        with tables.table(str(source / "STATE"), readonly=True, ack=False) as table:
            columns = set(table.colnames())
            for row in range(table.nrows()):
                mode = (
                    str(table.getcell("OBS_MODE", row))
                    if "OBS_MODE" in columns
                    else ""
                )
                states.append(StateRecord(row, mode, _intent_names(mode)))
    except (KeyError, RuntimeError):
        pass

    state_by_id = {state.state_id: state for state in states}
    fields: list[FieldRecord] = []
    with tables.table(str(source / "FIELD"), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        for row in range(table.nrows()):
            name = str(table.getcell("NAME", row)) if "NAME" in columns else str(row)
            roles = set(role_overrides.get(name, ())) if role_overrides else set()
            row_states = np.unique(state_id[field_id == row])
            for selected_state in row_states:
                state = state_by_id.get(int(selected_state))
                if state is None:
                    continue
                roles.update(
                    role
                    for prefix, role in _INTENT_ROLES.items()
                    if any(intent.startswith(prefix) for intent in state.intents)
                )
            phase = _direction(table.getcell("PHASE_DIR", row))
            fields.append(
                FieldRecord(
                    field_id=row,
                    name=name,
                    source_id=(
                        int(table.getcell("SOURCE_ID", row))
                        if "SOURCE_ID" in columns
                        else row
                    ),
                    phase_direction_rad=phase,
                    delay_direction_rad=(
                        _direction(table.getcell("DELAY_DIR", row))
                        if "DELAY_DIR" in columns
                        else phase
                    ),
                    reference_direction_rad=(
                        _direction(table.getcell("REFERENCE_DIR", row))
                        if "REFERENCE_DIR" in columns
                        else phase
                    ),
                    roles=tuple(sorted(roles, key=str)),
                )
            )

    observations: list[ObservationRecord] = []
    try:
        with tables.table(
            str(source / "OBSERVATION"), readonly=True, ack=False
        ) as table:
            columns = set(table.colnames())
            for row in range(table.nrows()):
                time_range = (
                    np.asarray(table.getcell("TIME_RANGE", row)).reshape(2)
                    if "TIME_RANGE" in columns
                    else np.asarray([np.nan, np.nan])
                )
                observations.append(
                    ObservationRecord(
                        observation_id=row,
                        telescope_name=(
                            str(table.getcell("TELESCOPE_NAME", row))
                            if "TELESCOPE_NAME" in columns
                            else ""
                        ),
                        observer=(
                            str(table.getcell("OBSERVER", row))
                            if "OBSERVER" in columns
                            else ""
                        ),
                        project=(
                            str(table.getcell("PROJECT", row))
                            if "PROJECT" in columns
                            else ""
                        ),
                        time_range_s=(float(time_range[0]), float(time_range[1])),
                    )
                )
    except (KeyError, RuntimeError):
        pass

    feeds: list[FeedRecord] = []
    try:
        with tables.table(str(source / "FEED"), readonly=True, ack=False) as table:
            columns = set(table.colnames())
            for row in range(table.nrows()):
                feeds.append(
                    FeedRecord(
                        antenna_id=int(table.getcell("ANTENNA_ID", row)),
                        feed_id=int(table.getcell("FEED_ID", row)),
                        spectral_window_id=int(table.getcell("SPECTRAL_WINDOW_ID", row)),
                        receptor_types=(
                            tuple(str(value) for value in table.getcell("POLARIZATION_TYPE", row))
                            if "POLARIZATION_TYPE" in columns
                            else ()
                        ),
                        receptor_angles_rad=(
                            tuple(
                                float(value)
                                for value in np.asarray(
                                    table.getcell("RECEPTOR_ANGLE", row)
                                ).ravel()
                            )
                            if "RECEPTOR_ANGLE" in columns
                            else ()
                        ),
                    )
                )
    except (KeyError, RuntimeError):
        pass
    return ObservationMetadata(
        antennas=tuple(antennas),
        fields=tuple(fields),
        states=tuple(states),
        observations=tuple(observations),
        feeds=tuple(feeds),
    )


def extract_measurement_set(
    path: str | Path,
    *,
    data_column: str = "CORRECTED_DATA",
    model_column: str | None = None,
    fields: tuple[int, ...] | None = None,
    field_names: tuple[str, ...] | None = None,
    roles: tuple[CalibratorRole, ...] | None = None,
    role_overrides: Mapping[str, tuple[CalibratorRole, ...]] | None = None,
    data_description_ids: tuple[int, ...] | None = None,
    channels: tuple[int, ...] | None = None,
    row_stride: int = 1,
) -> VisibilityDataset:
    """Extract compatible field/data-description blocks from a MeasurementSet."""

    if row_stride < 1:
        raise ValueError("row_stride must be positive")
    tables = _tables()
    source = Path(path)
    with tables.table(str(source), readonly=True, ack=False) as main:
        columns = set(main.colnames())
        if data_column not in columns:
            raise ValueError(f"MeasurementSet does not contain {data_column!r}")
        if model_column is not None and model_column not in columns:
            raise ValueError(f"MeasurementSet does not contain {model_column!r}")
        field_id = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
        rows = field_id.size

        def main_column(name: str, dtype: Any, default: int | float = 0) -> np.ndarray:
            if name in columns:
                return np.asarray(main.getcol(name), dtype=dtype)
            return np.full(rows, default, dtype=dtype)

        arrays: dict[str, Any] = {
            "visibility": _read_column(main, data_column),
            "model_visibility": (
                None if model_column is None else _read_column(main, model_column)
            ),
            "flag": _read_column(main, "FLAG", bool),
            "flag_row": main_column("FLAG_ROW", bool),
            "uvw_m": np.asarray(main.getcol("UVW"), dtype=np.float64),
            "time_s": np.asarray(main.getcol("TIME"), dtype=np.float64),
            "antenna1": np.asarray(main.getcol("ANTENNA1"), dtype=np.int32),
            "antenna2": np.asarray(main.getcol("ANTENNA2"), dtype=np.int32),
            "scan_id": np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32),
            "state_id": main_column("STATE_ID", np.int32),
            "observation_id": main_column("OBSERVATION_ID", np.int32),
            "feed1": main_column("FEED1", np.int32),
            "feed2": main_column("FEED2", np.int32),
            "interval_s": main_column("INTERVAL", np.float64, 1.0),
        }
        if "WEIGHT_SPECTRUM" in columns and main.iscelldefined("WEIGHT_SPECTRUM", 0):
            arrays["weight"] = _read_column(main, "WEIGHT_SPECTRUM", np.float64)
        else:
            arrays["row_weight"] = np.asarray(main.getcol("WEIGHT"), dtype=np.float64)

    with tables.table(str(source / "DATA_DESCRIPTION"), readonly=True, ack=False) as table:
        spectral_window_ids = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        polarization_ids = np.asarray(table.getcol("POLARIZATION_ID"), dtype=np.int32)
    with tables.table(str(source / "SPECTRAL_WINDOW"), readonly=True, ack=False) as table:
        channel_frequencies = [
            np.asarray(table.getcell("CHAN_FREQ", row), dtype=np.float64)
            for row in range(table.nrows())
        ]
    with tables.table(str(source / "POLARIZATION"), readonly=True, ack=False) as table:
        correlation_codes = [
            np.asarray(table.getcell("CORR_TYPE", row), dtype=np.int32)
            for row in range(table.nrows())
        ]
    with tables.table(str(source / "FIELD"), readonly=True, ack=False) as table:
        phase_centres = [
            np.asarray(table.getcell("PHASE_DIR", row), dtype=np.float64).reshape(-1, 2)[0]
            for row in range(table.nrows())
        ]

    metadata = _extract_metadata(
        tables,
        source,
        field_id,
        arrays["state_id"],
        role_overrides=role_overrides,
    )
    requested_fields = set(np.unique(field_id) if fields is None else fields)
    if field_names is not None:
        requested_fields &= {
            field.field_id for field in metadata.fields if field.name in field_names
        }
    if roles is not None:
        requested_roles = set(roles)
        requested_fields &= {
            field.field_id
            for field in metadata.fields
            if requested_roles.intersection(field.roles)
        }
    requested_ddids = set(np.unique(ddid) if data_description_ids is None else data_description_ids)
    blocks: list[VisibilityBlock] = []
    for selected_field in sorted(requested_fields):
        for selected_ddid in sorted(requested_ddids):
            selected = (field_id == selected_field) & (ddid == selected_ddid)
            if not np.any(selected):
                continue
            selected_rows = np.flatnonzero(selected)[::row_stride]
            selected = np.zeros_like(selected, dtype=bool)
            selected[selected_rows] = True
            spectral_window_id = int(spectral_window_ids[selected_ddid])
            polarization_id = int(polarization_ids[selected_ddid])
            all_frequencies = channel_frequencies[spectral_window_id]
            channel_indices = (
                np.arange(all_frequencies.size, dtype=np.int32)
                if channels is None
                else np.asarray(channels, dtype=np.int32)
            )
            if np.any(channel_indices < 0) or np.any(channel_indices >= all_frequencies.size):
                raise ValueError(
                    f"channel selection is outside spectral window {spectral_window_id}"
                )
            try:
                correlations = tuple(
                    _CORRELATION_CODES[int(code)] for code in correlation_codes[polarization_id]
                )
            except KeyError as exc:
                raise ValueError(f"unsupported CASA correlation code {exc.args[0]}") from exc
            visibility = _select_rows(arrays["visibility"], selected)[:, channel_indices, :]
            model_visibility = (
                None
                if arrays["model_visibility"] is None
                else _select_rows(arrays["model_visibility"], selected)[
                    :, channel_indices, :
                ]
            )
            if "weight" in arrays:
                weight = _select_rows(arrays["weight"], selected)[:, channel_indices, :]
            else:
                weight = np.broadcast_to(
                    arrays["row_weight"][selected, None, :], visibility.shape
                ).copy()
            selected_flag = _select_rows(arrays["flag"], selected)[
                :, channel_indices, :
            ]
            selected_flag |= arrays["flag_row"][selected, None, None]
            blocks.append(
                VisibilityBlock(
                    uvw_m=arrays["uvw_m"][selected],
                    frequency_hz=all_frequencies[channel_indices],
                    visibility=visibility,
                    weight=weight,
                    flag=selected_flag,
                    model_visibility=model_visibility,
                    time_s=arrays["time_s"][selected],
                    antenna1=arrays["antenna1"][selected],
                    antenna2=arrays["antenna2"][selected],
                    field_id=field_id[selected],
                    scan_id=arrays["scan_id"][selected],
                    state_id=arrays["state_id"][selected],
                    observation_id=arrays["observation_id"][selected],
                    feed1=arrays["feed1"][selected],
                    feed2=arrays["feed2"][selected],
                    interval_s=arrays["interval_s"][selected],
                    correlations=correlations,
                    receptor_basis=_basis(correlations),
                    phase_centre_rad=(
                        float(phase_centres[selected_field][0]),
                        float(phase_centres[selected_field][1]),
                    ),
                    data_description_id=selected_ddid,
                    spectral_window_id=spectral_window_id,
                    polarization_id=polarization_id,
                    provenance={
                        "source": str(source.resolve()),
                        "source_column": data_column,
                    "model_column": model_column,
                        "field_id": selected_field,
                        "data_description_id": selected_ddid,
                        "channel_indices": channel_indices,
                        "row_stride": row_stride,
                        "averaging": "none",
                        "flag_state": "preserved",
                    },
                )
            )
    if not blocks:
        raise ValueError("MeasurementSet selection produced no compatible blocks")
    return VisibilityDataset(
        tuple(blocks),
        {
            "extractor": "sl1mjax",
            "source": str(source.resolve()),
            "initial_compatibility": "VLA-oriented",
        },
        metadata,
    )
