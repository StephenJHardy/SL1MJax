"""Versioned canonical visibility blocks and collections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sl1mjax.data.metadata import ObservationMetadata
from sl1mjax.polarization import Correlation, ReceptorBasis, validate_correlations

SCHEMA_VERSION = "1.2"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1", SCHEMA_VERSION}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass(frozen=True)
class VisibilityBlock:
    uvw_m: np.ndarray
    frequency_hz: np.ndarray
    visibility: np.ndarray
    weight: np.ndarray
    flag: np.ndarray
    time_s: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    correlations: tuple[Correlation, ...]
    receptor_basis: ReceptorBasis
    model_visibility: np.ndarray | None = None
    field_id: np.ndarray | None = None
    scan_id: np.ndarray | None = None
    state_id: np.ndarray | None = None
    observation_id: np.ndarray | None = None
    feed1: np.ndarray | None = None
    feed2: np.ndarray | None = None
    interval_s: np.ndarray | None = None
    phase_centre_rad: tuple[float, float] = (0.0, 0.0)
    data_description_id: int = 0
    spectral_window_id: int = 0
    polarization_id: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uvw_m", np.asarray(self.uvw_m, dtype=np.float64))
        object.__setattr__(self, "frequency_hz", np.asarray(self.frequency_hz, dtype=np.float64))
        object.__setattr__(self, "visibility", np.asarray(self.visibility, dtype=np.complex128))
        object.__setattr__(self, "weight", np.asarray(self.weight, dtype=np.float64))
        object.__setattr__(self, "flag", np.asarray(self.flag, dtype=bool))
        object.__setattr__(self, "time_s", np.asarray(self.time_s, dtype=np.float64))
        object.__setattr__(self, "antenna1", np.asarray(self.antenna1, dtype=np.int32))
        object.__setattr__(self, "antenna2", np.asarray(self.antenna2, dtype=np.int32))
        object.__setattr__(
            self, "correlations", tuple(Correlation(value) for value in self.correlations)
        )
        object.__setattr__(self, "receptor_basis", ReceptorBasis(self.receptor_basis))
        if self.model_visibility is not None:
            object.__setattr__(
                self,
                "model_visibility",
                np.asarray(self.model_visibility, dtype=np.complex128),
            )
        rows = self.uvw_m.shape[0] if self.uvw_m.ndim else 0
        object.__setattr__(
            self,
            "field_id",
            np.zeros(rows, dtype=np.int32)
            if self.field_id is None
            else np.asarray(self.field_id, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "scan_id",
            np.zeros(rows, dtype=np.int32)
            if self.scan_id is None
            else np.asarray(self.scan_id, dtype=np.int32),
        )
        for name in ("state_id", "observation_id", "feed1", "feed2"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                np.zeros(rows, dtype=np.int32)
                if value is None
                else np.asarray(value, dtype=np.int32),
            )
        object.__setattr__(
            self,
            "interval_s",
            np.ones(rows, dtype=np.float64)
            if self.interval_s is None
            else np.asarray(self.interval_s, dtype=np.float64),
        )
        self.validate()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.visibility.shape

    @property
    def active(self) -> np.ndarray:
        return (
            ~self.flag
            & np.isfinite(self.weight)
            & (self.weight > 0)
            & np.isfinite(self.visibility.real)
            & np.isfinite(self.visibility.imag)
        )

    @property
    def antenna_count(self) -> int:
        return int(max(np.max(self.antenna1), np.max(self.antenna2)) + 1)

    def validate(self) -> None:
        validate_correlations(self.receptor_basis, self.correlations)
        if self.uvw_m.ndim != 2 or self.uvw_m.shape[1] != 3:
            raise ValueError("uvw_m must have shape (row, 3)")
        if self.frequency_hz.ndim != 1 or not np.all(
            np.isfinite(self.frequency_hz) & (self.frequency_hz > 0)
        ):
            raise ValueError("frequency_hz must contain finite positive values")
        expected = (self.uvw_m.shape[0], self.frequency_hz.size, len(self.correlations))
        for name, value in (
            ("visibility", self.visibility),
            ("weight", self.weight),
            ("flag", self.flag),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if self.model_visibility is not None and self.model_visibility.shape != expected:
            raise ValueError(
                f"model_visibility has shape {self.model_visibility.shape}; expected {expected}"
            )
        for name in ("time_s", "antenna1", "antenna2", "interval_s"):
            row_value = getattr(self, name)
            if row_value is None or row_value.shape != (expected[0],):
                raise ValueError(f"{name} must have shape ({expected[0]},)")
        for name in (
            "field_id",
            "scan_id",
            "state_id",
            "observation_id",
            "feed1",
            "feed2",
        ):
            value = getattr(self, name)
            if value is None or value.shape != (expected[0],):
                raise ValueError(f"{name} must have shape ({expected[0]},)")

    def select_rows(
        self,
        rows: np.ndarray,
        *,
        pad_to: int | None = None,
    ) -> VisibilityBlock:
        """Return a physically sliced block, optionally padded to a fixed row count.

        Padding appends flagged, zero-weight rows that share the first selected
        timestamp so a time-grouped batch keeps one unique time.
        """

        index = np.asarray(rows, dtype=np.int32).reshape(-1)
        n_source = int(self.uvw_m.shape[0])
        if index.size == 0:
            raise ValueError("select_rows requires at least one row")
        if np.any(index < 0) or np.any(index >= n_source):
            raise ValueError("row index is outside the visibility block")
        count = int(index.size)
        target = count if pad_to is None else int(pad_to)
        if target < count:
            raise ValueError("pad_to must be at least the number of selected rows")
        extra = target - count

        def take_row(array: np.ndarray, fill: np.ndarray) -> np.ndarray:
            selected = array[index]
            if extra == 0:
                return selected
            return np.concatenate((selected, np.broadcast_to(fill, (extra, *array.shape[1:]))))

        def row_array(array: np.ndarray | None) -> np.ndarray:
            if array is None:
                raise RuntimeError("row metadata must be populated before slicing")
            return array

        dummy_time = np.asarray(self.time_s[index[0]], dtype=self.time_s.dtype)
        visibility = take_row(
            self.visibility, np.zeros(self.visibility.shape[1:], dtype=self.visibility.dtype)
        )
        weight = take_row(self.weight, np.zeros(self.weight.shape[1:], dtype=self.weight.dtype))
        flag = take_row(self.flag, np.ones(self.flag.shape[1:], dtype=bool))
        model = None
        if self.model_visibility is not None:
            model = take_row(
                self.model_visibility,
                np.zeros(self.model_visibility.shape[1:], dtype=self.model_visibility.dtype),
            )
        return VisibilityBlock(
            uvw_m=take_row(self.uvw_m, np.zeros((3,), dtype=self.uvw_m.dtype)),
            frequency_hz=self.frequency_hz,
            visibility=visibility,
            weight=weight,
            flag=flag,
            time_s=take_row(self.time_s, dummy_time),
            antenna1=take_row(self.antenna1, np.asarray(self.antenna1[index[0]])),
            antenna2=take_row(self.antenna2, np.asarray(self.antenna2[index[0]])),
            correlations=self.correlations,
            receptor_basis=self.receptor_basis,
            model_visibility=model,
            field_id=take_row(
                row_array(self.field_id), np.asarray(row_array(self.field_id)[index[0]])
            ),
            scan_id=take_row(
                row_array(self.scan_id), np.asarray(row_array(self.scan_id)[index[0]])
            ),
            state_id=take_row(
                row_array(self.state_id), np.asarray(row_array(self.state_id)[index[0]])
            ),
            observation_id=take_row(
                row_array(self.observation_id),
                np.asarray(row_array(self.observation_id)[index[0]]),
            ),
            feed1=take_row(row_array(self.feed1), np.asarray(row_array(self.feed1)[index[0]])),
            feed2=take_row(row_array(self.feed2), np.asarray(row_array(self.feed2)[index[0]])),
            interval_s=take_row(
                row_array(self.interval_s), np.asarray(row_array(self.interval_s)[index[0]])
            ),
            phase_centre_rad=self.phase_centre_rad,
            data_description_id=self.data_description_id,
            spectral_window_id=self.spectral_window_id,
            polarization_id=self.polarization_id,
            provenance=dict(self.provenance),
        )

    def to_xarray(self) -> xr.Dataset:
        data_vars: dict[str, Any] = {
            "uvw_m": (("row", "uvw"), self.uvw_m),
            "visibility": (("row", "channel", "correlation"), self.visibility),
            "weight": (("row", "channel", "correlation"), self.weight),
            "flag": (("row", "channel", "correlation"), self.flag),
            "time_s": (("row",), self.time_s),
            "antenna1": (("row",), self.antenna1),
            "antenna2": (("row",), self.antenna2),
            "field_id": (("row",), self.field_id),
            "scan_id": (("row",), self.scan_id),
            "state_id": (("row",), self.state_id),
            "observation_id": (("row",), self.observation_id),
            "feed1": (("row",), self.feed1),
            "feed2": (("row",), self.feed2),
            "interval_s": (("row",), self.interval_s),
        }
        if self.model_visibility is not None:
            data_vars["model_visibility"] = (
                ("row", "channel", "correlation"),
                self.model_visibility,
            )
        return xr.Dataset(
            data_vars=data_vars,
            coords={
                "frequency_hz": (("channel",), self.frequency_hz),
                "correlation": [value.value for value in self.correlations],
                "uvw": ["u", "v", "w"],
            },
            attrs={
                "schema_version": SCHEMA_VERSION,
                "receptor_basis": self.receptor_basis.value,
                "phase_centre_ra_rad": float(self.phase_centre_rad[0]),
                "phase_centre_dec_rad": float(self.phase_centre_rad[1]),
                "data_description_id": self.data_description_id,
                "spectral_window_id": self.spectral_window_id,
                "polarization_id": self.polarization_id,
                "provenance_json": json.dumps(
                    dict(self.provenance), sort_keys=True, default=_json_default
                ),
            },
        )

    @classmethod
    def from_xarray(cls, dataset: xr.Dataset) -> VisibilityBlock:
        if dataset.attrs.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema {dataset.attrs.get('schema_version')!r}")
        rows = int(dataset.sizes["row"])

        def optional_row(name: str, dtype: Any) -> np.ndarray:
            if name in dataset:
                return np.asarray(dataset[name].values, dtype=dtype)
            return np.zeros(rows, dtype=dtype)

        return cls(
            uvw_m=dataset["uvw_m"].values,
            frequency_hz=dataset["frequency_hz"].values,
            visibility=dataset["visibility"].values,
            weight=dataset["weight"].values,
            flag=dataset["flag"].values,
            time_s=dataset["time_s"].values,
            antenna1=dataset["antenna1"].values,
            antenna2=dataset["antenna2"].values,
            field_id=dataset["field_id"].values,
            scan_id=dataset["scan_id"].values,
            state_id=optional_row("state_id", np.int32),
            observation_id=optional_row("observation_id", np.int32),
            feed1=optional_row("feed1", np.int32),
            feed2=optional_row("feed2", np.int32),
            interval_s=(
                np.asarray(dataset["interval_s"].values, dtype=np.float64)
                if "interval_s" in dataset
                else np.ones(rows, dtype=np.float64)
            ),
            model_visibility=(
                dataset["model_visibility"].values if "model_visibility" in dataset else None
            ),
            correlations=tuple(Correlation(str(value)) for value in dataset.correlation.values),
            receptor_basis=ReceptorBasis(dataset.attrs["receptor_basis"]),
            phase_centre_rad=(
                float(dataset.attrs["phase_centre_ra_rad"]),
                float(dataset.attrs["phase_centre_dec_rad"]),
            ),
            data_description_id=int(dataset.attrs["data_description_id"]),
            spectral_window_id=int(dataset.attrs["spectral_window_id"]),
            polarization_id=int(dataset.attrs["polarization_id"]),
            provenance=json.loads(dataset.attrs["provenance_json"]),
        )


@dataclass(frozen=True)
class VisibilityDataset:
    blocks: tuple[VisibilityBlock, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: ObservationMetadata | None = None

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a visibility dataset requires at least one block")


def write_dataset(dataset: VisibilityDataset, path: str | Path) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    block_names: list[str] = []
    for index, block in enumerate(dataset.blocks):
        name = f"block_{index:04d}.zarr"
        block.to_xarray().to_zarr(root / name, mode="w", consolidated=False)
        block_names.append(name)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "blocks": block_names,
        "provenance": dict(dataset.provenance),
        "metadata": (None if dataset.metadata is None else dataset.metadata.to_dict()),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_dataset(path: str | Path) -> VisibilityDataset:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported dataset schema {manifest.get('schema_version')!r}")
    blocks = []
    for name in manifest["blocks"]:
        with xr.open_zarr(root / name, consolidated=False) as stored:
            blocks.append(VisibilityBlock.from_xarray(stored.load()))
    return VisibilityDataset(
        tuple(blocks),
        manifest.get("provenance", {}),
        ObservationMetadata.from_dict(manifest.get("metadata")),
    )
