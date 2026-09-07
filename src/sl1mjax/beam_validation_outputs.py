"""Compact VLA C-band beam-validation bundle.

The notebook and render script load this bundle. They do not read the
Measurement Set, invoke CASA, or follow Bacchus working-directory paths.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

SCHEMA_VERSION = 1
PUBLICATION_VERSION = "vla_c_band_beam_validation_v1"
BUNDLE_DIRNAME = PUBLICATION_VERSION
PLOT_TABLE_DIRNAME = "plot_tables"

REQUIRED_JSON = (
    "manifest.json",
    "claims.json",
    "observation_summary.json",
    "calibration_summary.json",
    "convention_gates.json",
    "holoraster_channel32.json",
    "holoraster_frequency.json",
    "squint_publication.json",
    "offset_ring.json",
)
REQUIRED_PLOT_TABLES = (
    "holoraster_scatter.npz",
    "holoraster_maps.npz",
    "raster_occupancy.npz",
    "residual_geometry.json",
    "frequency_series.json",
    "offset_ring_fields.json",
    "crosshand_quadrants.json",
)
BANNED_LOAD_PREFIXES = (
    "/media/stephen",
    "/tmp/sl1mjax-pointing-audit",
    "/tmp/sl1mjax-thol0001-scripts",
)
PROVENANCE_PATH_KEYS = frozenset(
    {
        "path",
        "measurement_set",
        "root",
        "tree",
        "output_dir",
        "working_directory",
    }
)


def default_bundle_root() -> Path:
    """Committed publication bundle inside the installed package data."""

    return Path(__file__).resolve().parent / "data" / BUNDLE_DIRNAME


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(payload: Mapping[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def iter_bundle_files(root: Path) -> tuple[Path, ...]:
    source = Path(root)
    files = [path for path in source.rglob("*") if path.is_file()]
    return tuple(sorted(files, key=lambda path: path.relative_to(source).as_posix()))


def file_checksums(root: Path) -> dict[str, str]:
    source = Path(root)
    return {
        path.relative_to(source).as_posix(): sha256_file(path) for path in iter_bundle_files(source)
    }


def contains_banned_load_path(value: object) -> str | None:
    """Return a banned prefix if ``value`` looks like a required filesystem path."""

    if isinstance(value, str):
        for prefix in BANNED_LOAD_PREFIXES:
            if value.startswith(prefix):
                return prefix
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in PROVENANCE_PATH_KEYS:
                continue
            found = contains_banned_load_path(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = contains_banned_load_path(item)
            if found is not None:
                return found
    return None


def sanitize_provenance_paths(value: object) -> object:
    """Keep basenames for identity; drop Bacchus load paths from copied products."""

    if isinstance(value, str):
        for prefix in BANNED_LOAD_PREFIXES:
            if value.startswith(prefix):
                return Path(value).name
        return value
    if isinstance(value, Mapping):
        return {str(key): sanitize_provenance_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_provenance_paths(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_provenance_paths(item) for item in value]
    return value


def validate_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    """Gate N0: schema, checksums, and no required Bacchus load paths."""

    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("bundle schema_version does not match the loader")
    if str(manifest.get("publication_version")) != PUBLICATION_VERSION:
        raise ValueError("bundle publication_version does not match the loader")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("manifest.files must list every included checksum")
    expected = {path.relative_to(root).as_posix() for path in iter_bundle_files(root)}
    expected.discard("manifest.json")
    listed = set(files)
    if listed != expected:
        missing = sorted(expected - listed)
        extra = sorted(listed - expected)
        raise ValueError(f"manifest files disagree: missing={missing} extra={extra}")
    for relative, digest in files.items():
        actual = sha256_file(root / str(relative))
        if actual != digest:
            raise ValueError(f"checksum mismatch for {relative}")
    banned = contains_banned_load_path(manifest)
    if banned is not None:
        raise ValueError(f"manifest requires a banned load path under {banned}")


@dataclass(frozen=True)
class ValidationBundle:
    """Loaded publication bundle. Arrays stay on disk until requested."""

    root: Path
    manifest: dict[str, Any]
    claims: dict[str, Any]
    observation: dict[str, Any]
    calibration: dict[str, Any]
    convention_gates: dict[str, Any]
    holoraster_channel32: dict[str, Any]
    holoraster_frequency: dict[str, Any]
    squint: dict[str, Any]
    offset_ring: dict[str, Any]
    residual_geometry: dict[str, Any]
    frequency_series: dict[str, Any]
    offset_ring_fields: dict[str, Any]
    crosshand_quadrants: dict[str, Any]

    def plot_table(self, name: str) -> dict[str, NDArray]:
        path = self.root / PLOT_TABLE_DIRNAME / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as handle:
            return {key: np.asarray(handle[key]) for key in handle.files}

    def json_table(self, name: str) -> dict[str, Any]:
        return load_json(self.root / PLOT_TABLE_DIRNAME / name)


def load_bundle(root: Path | None = None) -> ValidationBundle:
    """Load and validate the compact publication bundle."""

    source = Path(root) if root is not None else default_bundle_root()
    if not source.is_dir():
        raise FileNotFoundError(f"validation bundle is missing: {source}")
    for name in REQUIRED_JSON:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    tables = source / PLOT_TABLE_DIRNAME
    for name in REQUIRED_PLOT_TABLES:
        if not (tables / name).is_file():
            raise FileNotFoundError(tables / name)
    manifest = load_json(source / "manifest.json")
    validate_manifest(manifest, source)
    payload = {name: load_json(source / name) for name in REQUIRED_JSON if name != "manifest.json"}
    for name, document in payload.items():
        banned = contains_banned_load_path(document)
        if banned is not None:
            raise ValueError(f"{name} requires a banned load path under {banned}")
    residual = load_json(tables / "residual_geometry.json")
    frequency = load_json(tables / "frequency_series.json")
    fields = load_json(tables / "offset_ring_fields.json")
    quadrants = load_json(tables / "crosshand_quadrants.json")
    return ValidationBundle(
        root=source,
        manifest=manifest,
        claims=payload["claims.json"],
        observation=payload["observation_summary.json"],
        calibration=payload["calibration_summary.json"],
        convention_gates=payload["convention_gates.json"],
        holoraster_channel32=payload["holoraster_channel32.json"],
        holoraster_frequency=payload["holoraster_frequency.json"],
        squint=payload["squint_publication.json"],
        offset_ring=payload["offset_ring.json"],
        residual_geometry=residual,
        frequency_series=frequency,
        offset_ring_fields=fields,
        crosshand_quadrants=quadrants,
    )


def refuse_unfrozen_as_frozen(record: Mapping[str, Any]) -> None:
    if record.get("full_jones_frozen") in {True}:
        raise ValueError("full Jones is not frozen in this publication")
    if record.get("production_factory_modified") in {True}:
        raise ValueError("the production factory must remain unmodified")
    if record.get("model_selected") in {True}:
        raise ValueError("the comparison report must not select a model")


def write_plot_npz(path: Path, arrays: Mapping[str, ArrayLike]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{key: np.asarray(value) for key, value in arrays.items()})
    return destination


def write_validation_bundle(
    root: Path,
    *,
    observation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    convention_gates: Mapping[str, Any],
    holoraster_channel32: Mapping[str, Any],
    holoraster_frequency: Mapping[str, Any],
    squint: Mapping[str, Any],
    offset_ring: Mapping[str, Any],
    residual_geometry: Mapping[str, Any],
    frequency_series: Mapping[str, Any],
    offset_ring_fields: Mapping[str, Any],
    crosshand_quadrants: Mapping[str, Any],
    scatter: Mapping[str, ArrayLike],
    maps: Mapping[str, ArrayLike],
    occupancy: Mapping[str, ArrayLike],
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete checksummed publication bundle."""

    from sl1mjax.beam_validation_claims import refuse_full_raster_squint
    from sl1mjax.beam_validation_statistics import build_claims

    destination = Path(root)
    if destination.exists():
        for path in destination.rglob("*"):
            if path.is_file():
                path.unlink()
    tables = destination / PLOT_TABLE_DIRNAME
    tables.mkdir(parents=True, exist_ok=True)
    refuse_full_raster_squint(squint["measured"])
    refuse_full_raster_squint(squint["cassbeam"])
    refuse_unfrozen_as_frozen(holoraster_channel32)
    documents = {
        "observation_summary.json": sanitize_provenance_paths(dict(observation)),
        "calibration_summary.json": sanitize_provenance_paths(dict(calibration)),
        "convention_gates.json": sanitize_provenance_paths(dict(convention_gates)),
        "holoraster_channel32.json": sanitize_provenance_paths(dict(holoraster_channel32)),
        "holoraster_frequency.json": sanitize_provenance_paths(dict(holoraster_frequency)),
        "squint_publication.json": sanitize_provenance_paths(dict(squint)),
        "offset_ring.json": sanitize_provenance_paths(dict(offset_ring)),
    }
    for name, payload in documents.items():
        write_json(payload, destination / name)
    write_json(
        sanitize_provenance_paths(dict(residual_geometry)),
        tables / "residual_geometry.json",
    )
    write_json(
        sanitize_provenance_paths(dict(frequency_series)),
        tables / "frequency_series.json",
    )
    write_json(
        sanitize_provenance_paths(dict(offset_ring_fields)),
        tables / "offset_ring_fields.json",
    )
    write_json(
        sanitize_provenance_paths(dict(crosshand_quadrants)),
        tables / "crosshand_quadrants.json",
    )
    write_plot_npz(tables / "holoraster_scatter.npz", scatter)
    write_plot_npz(tables / "holoraster_maps.npz", maps)
    write_plot_npz(tables / "raster_occupancy.npz", occupancy)
    placeholder = {
        "schema_version": SCHEMA_VERSION,
        "publication_version": PUBLICATION_VERSION,
        "files": {},
        "bundle_sha256": "",
        **{key: value for key, value in dict(provenance or {}).items() if key != "files"},
    }
    write_json(placeholder, destination / "manifest.json")
    draft = ValidationBundle(
        root=destination,
        manifest=placeholder,
        claims={"claims": []},
        observation=load_json(destination / "observation_summary.json"),
        calibration=load_json(destination / "calibration_summary.json"),
        convention_gates=load_json(destination / "convention_gates.json"),
        holoraster_channel32=load_json(destination / "holoraster_channel32.json"),
        holoraster_frequency=load_json(destination / "holoraster_frequency.json"),
        squint=load_json(destination / "squint_publication.json"),
        offset_ring=load_json(destination / "offset_ring.json"),
        residual_geometry=load_json(tables / "residual_geometry.json"),
        frequency_series=load_json(tables / "frequency_series.json"),
        offset_ring_fields=load_json(tables / "offset_ring_fields.json"),
        crosshand_quadrants=load_json(tables / "crosshand_quadrants.json"),
    )
    write_json(build_claims(draft), destination / "claims.json")
    checksums = file_checksums(destination)
    checksums.pop("manifest.json", None)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "publication_version": PUBLICATION_VERSION,
        "files": checksums,
        **dict(provenance or {}),
    }
    manifest["bundle_sha256"] = sha256_bytes(
        json.dumps({"files": checksums}, sort_keys=True).encode()
    )
    write_json(manifest, destination / "manifest.json")
    load_bundle(destination)
    return destination
