"""Optional MeasurementSet-to-canonical extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
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


def extract_measurement_set(
    path: str | Path,
    *,
    data_column: str = "CORRECTED_DATA",
    fields: tuple[int, ...] | None = None,
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
        field_id = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        ddid = np.asarray(main.getcol("DATA_DESC_ID"), dtype=np.int32)
        arrays: dict[str, Any] = {
            "visibility": _read_column(main, data_column),
            "flag": _read_column(main, "FLAG", bool),
            "uvw_m": np.asarray(main.getcol("UVW"), dtype=np.float64),
            "time_s": np.asarray(main.getcol("TIME"), dtype=np.float64),
            "antenna1": np.asarray(main.getcol("ANTENNA1"), dtype=np.int32),
            "antenna2": np.asarray(main.getcol("ANTENNA2"), dtype=np.int32),
            "scan_id": np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32),
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

    requested_fields = set(np.unique(field_id) if fields is None else fields)
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
            if "weight" in arrays:
                weight = _select_rows(arrays["weight"], selected)[:, channel_indices, :]
            else:
                weight = np.broadcast_to(
                    arrays["row_weight"][selected, None, :], visibility.shape
                ).copy()
            blocks.append(
                VisibilityBlock(
                    uvw_m=arrays["uvw_m"][selected],
                    frequency_hz=all_frequencies[channel_indices],
                    visibility=visibility,
                    weight=weight,
                    flag=_select_rows(arrays["flag"], selected)[:, channel_indices, :],
                    time_s=arrays["time_s"][selected],
                    antenna1=arrays["antenna1"][selected],
                    antenna2=arrays["antenna2"][selected],
                    field_id=field_id[selected],
                    scan_id=arrays["scan_id"][selected],
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
    )
