#!/usr/bin/env bash
# Read-only upper-C archive integrity check. Does not extract or overwrite.
set -euo pipefail
ROOT="${ROOT:-/media/stephen/astro/vla}"
OUT="${OUT:-/media/stephen/astro/vla/extracted/commissioning/validation/scientific/evla_c_diagonal_survey_v1/upper_c_archive_integrity.json}"
MS="${MS:-/media/stephen/astro/vla/extracted/THOL0001.sb31635131.eb31644651.57405.16003953703.ms}"
PY="${PY:-/home/stephen/checkouts/SL1MJax/.venv/bin/python}"

"$PY" - "$ROOT" "$OUT" "$MS" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
ms = Path(sys.argv[3])
names = ("*31635131*", "*31644651*", "*57405.16003953703*")
found: list[Path] = []
listing = subprocess.run(
    [
        "find",
        str(root),
        "-maxdepth",
        "6",
        "(",
        "-name",
        names[0],
        "-o",
        "-name",
        names[1],
        "-o",
        "-name",
        names[2],
        ")",
        "-type",
        "f",
    ],
    capture_output=True,
    text=True,
    check=False,
)
if listing.stdout.strip():
    found.extend(Path(line) for line in listing.stdout.splitlines() if line.strip())
archives = sorted(
    {
        path.resolve()
        for path in found
        if path.suffix.lower() in {".tgz", ".tar", ".gz", ".zip"}
        or ".ms.tgz" in path.name
        or path.name.endswith(".tar.gz")
    }
)
members = []
for path in archives:
    with path.open("rb") as handle:
        prefix = hashlib.sha256(handle.read(8_388_608)).hexdigest()
    info = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256_prefix": prefix,
        "full_file_hash_skipped": path.stat().st_size > 8_388_608,
    }
    if path.suffix in {".gz", ".tgz"} or path.name.endswith(".tar.gz") or path.name.endswith(".ms.tgz"):
        gzip = subprocess.run(["gzip", "-t", str(path)], capture_output=True, text=True)
        info["gzip_test_returncode"] = gzip.returncode
        info["gzip_test_stderr"] = gzip.stderr[-500:]
        listed = subprocess.run(["tar", "-tzf", str(path)], capture_output=True, text=True)
        info["tar_list_returncode"] = listed.returncode
        info["tar_list_stderr"] = listed.stderr[-800:]
        names_out = [line for line in listed.stdout.splitlines() if line.strip()]
        info["n_members"] = len(names_out)
        info["ms_tgz_members"] = [name for name in names_out if "ms.tgz" in name or name.endswith(".ms/")][:20]
    members.append(info)

ms_info = {"exists": ms.exists(), "path": str(ms)}
if ms.exists():
    table = ms / "table.dat"
    ms_info["table_dat_bytes"] = table.stat().st_size if table.is_file() else None
    ms_info["n_entries"] = sum(1 for _ in ms.iterdir())

failed = any(
    item.get("gzip_test_returncode") not in (None, 0)
    or item.get("tar_list_returncode") not in (None, 0)
    for item in members
)
if not members:
    recommendation = "archive_tarball_not_found_extracted_ms_unverified"
elif failed:
    recommendation = "replacement_needed"
else:
    recommendation = "archives_readable_inspect_inner_ms_tgz"
payload = {
    "schema_version": 1,
    "read_only": True,
    "original_preserved": True,
    "search_root": str(root),
    "extracted_ms": ms_info,
    "archives": members,
    "recommendation": recommendation,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(output)
print("archives", len(members), "recommendation", recommendation)
PY
