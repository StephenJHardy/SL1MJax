from __future__ import annotations

import io
import runpy
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
BUILD_SCRIPT = runpy.run_path(str(ROOT / "scripts/build_nrao_calibration_corpus.py"))
INSPECT_SCRIPT = runpy.run_path(
    str(ROOT / "scripts/inspect_srdp_calibration_archive.py")
)
ANALYZE_SCRIPT = runpy.run_path(
    str(ROOT / "scripts/analyze_srdp_calibration_corpus.py")
)
select_balanced = cast(
    Callable[[list[dict[str, Any]], int], list[dict[str, Any]]],
    BUILD_SCRIPT["select_balanced"],
)
inventory_archive = cast(Callable[[Path], dict[str, Any]], INSPECT_SCRIPT["inventory_archive"])
extract_archive = cast(
    Callable[[Path, Path], Path], INSPECT_SCRIPT["extract_archive"]
)
table_role = cast(Callable[[str], str | None], ANALYZE_SCRIPT["_table_role"])
prior_table_role = cast(
    Callable[[str], str | None], ANALYZE_SCRIPT["_prior_table_role"]
)
invalid_pair_fraction = cast(
    Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ANALYZE_SCRIPT["_invalid_pair_fraction"],
)


def _record(source: str, configuration: str, day: int) -> dict[str, object]:
    return {
        "search_source": source,
        "configuration": configuration,
        "obs_start": f"2024-01-{day:02d}T00:00:00Z",
        "calibration_file": f"{source}-{configuration}-{day}.caltables.tar",
    }


def test_calibration_corpus_selection_balances_source_and_configuration() -> None:
    records = [
        _record(source, configuration, day)
        for source in ("3C286", "3C48")
        for configuration in ("C", "D")
        for day in range(1, 6)
    ]

    selected = select_balanced(records, 8)

    groups = {
        (record["search_source"], record["configuration"]) for record in selected
    }
    assert groups == {("3C286", "C"), ("3C286", "D"), ("3C48", "C"), ("3C48", "D")}
    assert len({record["calibration_file"] for record in selected}) == 8


def _add_bytes(archive: tarfile.TarFile, name: str, value: bytes = b"x") -> None:
    member = tarfile.TarInfo(name)
    member.size = len(value)
    archive.addfile(member, io.BytesIO(value))


def test_srdp_archive_inventory_and_safe_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "example.caltables.tar"
    with tarfile.open(archive_path, "w") as archive:
        _add_bytes(archive, "products/finalgaincal.g/table.dat")
        _add_bytes(archive, "products/finalflags.txt")
        _add_bytes(archive, "products/weblog/index.html")

    inventory = inventory_archive(archive_path)
    extracted = extract_archive(archive_path, tmp_path / "extracted")

    assert inventory["calibration_table_roots"] == ["products/finalgaincal.g"]
    assert inventory["flag_products"] == ["products/finalflags.txt"]
    assert (extracted / "products/finalgaincal.g/table.dat").is_file()


def test_srdp_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        _add_bytes(archive, "../escape")

    with pytest.raises(ValueError, match="unsafe archive member"):
        inventory_archive(archive_path)


def test_srdp_table_roles_and_invalid_pair_fraction() -> None:
    assert table_role("case.ms.hifv_finalcals.s13_2.finaldelay.tbl") == "delay"
    assert table_role("case.ms.hifv_finalcals.s13_4.finalBPcal.tbl") == "bandpass"
    assert table_role("case.ms.hifv_priorcals.s5_3.opac.tbl") is None
    assert prior_table_role("case.ms.hifv_priorcals.s5_3.opac.tbl") == "opacity"
    assert prior_table_role("case.ms.hifv_priorcals.s5_2.gc.tbl") == "gain_curve"
    assert prior_table_role("case.ms.hifv_priorcals.s5_4.rq.tbl") == "requantizer"

    antenna = np.asarray([0, 0, 1, 1], dtype=np.int32)
    spectral_window = np.asarray([0, 1, 0, 1], dtype=np.int32)
    valid_rows = np.asarray([True, False, True, True])
    assert invalid_pair_fraction(antenna, spectral_window, valid_rows) == 0.25
