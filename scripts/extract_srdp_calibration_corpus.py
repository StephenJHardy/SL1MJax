"""Safely extract calibration tables from nested NRAO archive products."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any


def _safe_members(
    members: Iterable[tarfile.TarInfo], *, regular_only: bool = True
) -> list[tarfile.TarInfo]:
    safe = []
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        if regular_only and not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
        safe.append(member)
    return safe


def _copy_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Any
) -> None:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"cannot read archive member: {member.name}")
    shutil.copyfileobj(source, destination)


def _dataset_id(archive: tarfile.TarFile) -> str:
    for member in archive.getmembers():
        for part in PurePosixPath(member.name).parts:
            if ".ms." in part:
                return part.split(".ms.", maxsplit=1)[0]
    raise ValueError("embedded product does not contain a MeasurementSet-derived name")


def _extract_embedded(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination_root: Path,
) -> tuple[str, str]:
    suffix = "".join(Path(member.name).suffixes)
    with tempfile.NamedTemporaryFile(
        suffix=suffix, dir=destination_root, prefix=".embedded-"
    ) as temporary:
        _copy_member(archive, member, temporary)
        temporary.flush()
        with tarfile.open(temporary.name, "r:*") as embedded:
            members = _safe_members(embedded.getmembers())
            dataset_id = _dataset_id(embedded)
            destination = destination_root / dataset_id
            destination.mkdir(parents=True, exist_ok=True)
            embedded.extractall(destination, members=members, filter="data")
    return dataset_id, str(destination.resolve())


def extract_product(path: Path, destination: Path) -> dict[str, Any]:
    """Extract the caltables and flag-version payloads from one outer archive."""

    result: dict[str, Any] = {
        "archive": str(path.resolve()),
        "calibration_tables": [],
        "flagversions": [],
    }
    with tarfile.open(path, "r:*") as outer:
        products = [
            member
            for member in _safe_members(outer.getmembers(), regular_only=False)
            if member.isfile() and member.name.endswith(".tar")
        ]
        for product in products:
            with tempfile.NamedTemporaryFile(
                suffix=".tar", dir=destination, prefix=".product-"
            ) as temporary:
                _copy_member(outer, product, temporary)
                temporary.flush()
                with tarfile.open(temporary.name, "r:*") as nested:
                    for member in _safe_members(
                        nested.getmembers(), regular_only=False
                    ):
                        lower = member.name.lower()
                        if not member.isfile():
                            continue
                        if "caltables" in lower and lower.endswith(
                            (".tgz", ".tar.gz")
                        ):
                            result["calibration_tables"].append(
                                _extract_embedded(
                                    nested, member, destination / "tables"
                                )
                            )
                        elif "flagversions" in lower and lower.endswith(
                            (".tgz", ".tar.gz")
                        ):
                            result["flagversions"].append(
                                _extract_embedded(
                                    nested, member, destination / "flagversions"
                                )
                            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.root / "corpus"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    (output / "flagversions").mkdir(exist_ok=True)
    results = [
        extract_product(path, output)
        for path in sorted(arguments.root.glob("NRAO_archive_*.tar"))
    ]
    payload = {
        "schema_version": 1,
        "archive_count": len(results),
        "calibration_product_count": sum(
            len(result["calibration_tables"]) for result in results
        ),
        "flagversion_product_count": sum(
            len(result["flagversions"]) for result in results
        ),
        "products": results,
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{payload['archive_count']} archives, "
        f"{payload['calibration_product_count']} calibration products, "
        f"{payload['flagversion_product_count']} flag-version products"
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
