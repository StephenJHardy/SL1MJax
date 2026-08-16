"""Versioned canonical visibility blocks and collections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from sl1mjax.polarization import Correlation, ReceptorBasis, validate_correlations

SCHEMA_VERSION = "1.0"


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
    field_id: np.ndarray | None = None
    scan_id: np.ndarray | None = None
    phase_centre_rad: tuple[float, float] = (0.0, 0.0)
    data_description_id: int = 0
    spectral_window_id: int = 0
    polarization_id: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uvw_m", np.asarray(self.uvw_m, dtype=np.float64))
        object.__setattr__(
            self, "frequency_hz", np.asarray(self.frequency_hz, dtype=np.float64)
        )
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
        for name, value in (
            ("time_s", self.time_s),
            ("antenna1", self.antenna1),
            ("antenna2", self.antenna2),
        ):
            if value.shape != (expected[0],):
                raise ValueError(f"{name} must have shape ({expected[0]},)")
        for name in ("field_id", "scan_id"):
            value = getattr(self, name)
            if value is None or value.shape != (expected[0],):
                raise ValueError(f"{name} must have shape ({expected[0]},)")

    def to_xarray(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "uvw_m": (("row", "uvw"), self.uvw_m),
                "visibility": (("row", "channel", "correlation"), self.visibility),
                "weight": (("row", "channel", "correlation"), self.weight),
                "flag": (("row", "channel", "correlation"), self.flag),
                "time_s": (("row",), self.time_s),
                "antenna1": (("row",), self.antenna1),
                "antenna2": (("row",), self.antenna2),
                "field_id": (("row",), self.field_id),
                "scan_id": (("row",), self.scan_id),
            },
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
        if dataset.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {dataset.attrs.get('schema_version')!r}")
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
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_dataset(path: str | Path) -> VisibilityDataset:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema {manifest.get('schema_version')!r}")
    blocks = []
    for name in manifest["blocks"]:
        with xr.open_zarr(root / name, consolidated=False) as stored:
            blocks.append(VisibilityBlock.from_xarray(stored.load()))
    return VisibilityDataset(tuple(blocks), manifest.get("provenance", {}))
