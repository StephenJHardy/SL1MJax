"""Inventory and optionally extract NRAO SRDP calibration-table archives."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


def _safe_members(members: Iterable[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    safe: list[tarfile.TarInfo] = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        safe.append(member)
    return safe


def _table_roots(members: Iterable[tarfile.TarInfo]) -> list[str]:
    return sorted(
        {
            str(PurePosixPath(member.name).parent)
            for member in members
            if PurePosixPath(member.name).name == "table.dat"
        }
    )


def _matching_paths(
    members: Iterable[tarfile.TarInfo], terms: tuple[str, ...]
) -> list[str]:
    return sorted(
        {
            member.name
            for member in members
            if any(term in member.name.lower() for term in terms)
        }
    )


def inventory_archive(path: Path) -> dict[str, Any]:
    """Return a data-light inventory without extracting an archive."""

    with tarfile.open(path, "r:*") as archive:
        members = _safe_members(archive.getmembers())
    files = [member for member in members if member.isfile()]
    return {
        "archive": str(path.resolve()),
        "archive_bytes": path.stat().st_size,
        "member_count": len(members),
        "file_count": len(files),
        "uncompressed_file_bytes": sum(member.size for member in files),
        "calibration_table_roots": _table_roots(members),
        "flag_products": _matching_paths(members, ("flag",)),
        "qa_and_weblog_products": _matching_paths(
            members, ("qa", "weblog", "pipeline", ".html", ".log")
        ),
    }


def extract_archive(path: Path, destination: Path) -> Path:
    """Extract an archive with traversal and special-file protections."""

    target = destination / path.name.removesuffix(".tar").removesuffix(".tgz")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:*") as archive:
        _safe_members(archive.getmembers())
        archive.extractall(target, filter="data")
    return target


def _array_summary(value: Any) -> Any:
    try:
        array = value.tolist()
    except AttributeError:
        return value if isinstance(value, (bool, float, int, str)) else str(value)
    return array


def inspect_casa_table(path: Path) -> dict[str, Any]:
    """Read calibration-table structure when python-casacore is available."""

    try:
        from casacore import tables
    except ImportError:
        return {"path": str(path), "casacore_available": False}

    try:
        with tables.table(str(path), readonly=True, ack=False) as table:
            columns = list(table.colnames())
            info = {key: _array_summary(value) for key, value in table.info().items()}
            keywords = list(table.keywordnames())
            result: dict[str, Any] = {
                "path": str(path),
                "casacore_available": True,
                "row_count": int(table.nrows()),
                "columns": columns,
                "keywords": keywords,
                "table_info": info,
            }
            for name in ("VisCal", "MSName"):
                if name in keywords:
                    result[name.lower()] = _array_summary(table.getkeyword(name))
            for name in ("TIME", "INTERVAL", "ANTENNA1", "SPECTRAL_WINDOW_ID"):
                if name not in columns or table.nrows() == 0:
                    continue
                values = table.getcol(name)
                if name == "TIME":
                    result["time_range_s"] = [
                        float(values.min()),
                        float(values.max()),
                    ]
                elif name == "INTERVAL":
                    result["interval_range_s"] = [
                        float(values.min()),
                        float(values.max()),
                    ]
                else:
                    result[f"{name.lower()}s"] = sorted(
                        int(value) for value in set(values.ravel())
                    )
            if "FLAG" in columns and table.nrows() > 0:
                flags = table.getcol("FLAG")
                result["flagged_fraction"] = float(flags.mean())
            return result
    except RuntimeError as error:
        return {
            "path": str(path),
            "casacore_available": True,
            "error": str(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/srdp_calibration_inventory.json")
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        help="Safely extract each archive and inspect discovered CASA tables.",
    )
    arguments = parser.parse_args()

    inventories = []
    for archive in arguments.archives:
        inventory = inventory_archive(archive)
        if arguments.extract_to is not None:
            root = extract_archive(archive, arguments.extract_to)
            inventory["extracted_to"] = str(root.resolve())
            inventory["casa_tables"] = [
                inspect_casa_table(root / relative)
                for relative in inventory["calibration_table_roots"]
            ]
        inventories.append(inventory)

    payload = {"schema_version": 1, "archives": inventories}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
