"""Build a balanced index of public NRAO VLA calibration products.

The archive catalog is public, but product downloads are initiated through the
interactive archive portal. This script creates a reproducible shortlist and a
download manifest without requiring NRAO credentials.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

ARCHIVE_API = (
    "https://data.nrao.edu/archive-service/restapi_get_paged_exec_blocks"
)
PORTAL = "https://data.nrao.edu/portal/#"
CSV_FIELDS = (
    "selected",
    "search_source",
    "band",
    "configuration",
    "obs_start",
    "obs_stop",
    "project_code",
    "obs_id",
    "eb_id",
    "num_antennas",
    "num_scans",
    "visibility_bytes",
    "calibration_bytes",
    "calibration_file",
    "casa_version",
    "qa_class",
    "qa_status",
    "qa_notes",
    "science_product_locator",
    "calibration_product_locator",
    "product_url",
)


def _quoted(value: str) -> str:
    return json.dumps(value)


def _qa_metadata(notes: str | None) -> tuple[str, str | None]:
    text = notes or ""
    lower = text.lower()
    if "qa_standard: srdp" in lower:
        qa_class = "srdp"
    elif "qa_standard: non-srdp" in lower:
        qa_class = "staff_checked_non_srdp"
    else:
        qa_class = "legacy_restorable"
    status = None
    for line in text.splitlines():
        if line.lower().startswith("qa_status:"):
            status = line.partition(":")[2].strip() or None
            break
    return qa_class, status


def _fetch_json(parameters: Mapping[str, str | int], *, timeout_s: float) -> Any:
    url = f"{ARCHIVE_API}?{urlencode(parameters)}"
    with urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        return json.load(response)


def query_candidates(
    source: str,
    *,
    band: str,
    configurations: Sequence[str],
    page_size: int,
    timeout_s: float,
    single_band_only: bool,
) -> list[dict[str, Any]]:
    """Query public, restorable EVLA observations containing a calibrator."""

    records: list[dict[str, Any]] = []
    for configuration in configurations:
        start = 0
        while True:
            parameters: dict[str, str | int] = {
                "start": start,
                "rows": page_size,
                "sort": "obs_stop desc",
                "text_search_str": source,
                "show_cms_only": "true",
                "show_public_only": "true",
                "instrument_name": _quoted("EVLA"),
                "obs_band": _quoted(band),
                "vla_configuration": _quoted(configuration),
            }
            payload = _fetch_json(parameters, timeout_s=timeout_s)
            rows = payload["eb_list"]
            for row in rows:
                observed_bands = row.get("obs_band") or []
                if single_band_only and observed_bands != [band]:
                    continue
                for calibration in row.get("cals") or []:
                    qa_class, qa_status = _qa_metadata(calibration["qa_notes"])
                    records.append(
                        {
                            "selected": False,
                            "search_source": source,
                            "band": band,
                            "configuration": configuration,
                            "obs_start": row["obs_start"],
                            "obs_stop": row["obs_stop"],
                            "project_code": row["project_code"],
                            "obs_id": row["obs_id"],
                            "eb_id": row["eb_id"],
                            "num_antennas": row["num_antennas"],
                            "num_scans": row["num_scans"],
                            "visibility_bytes": row["access_estsize"],
                            "calibration_bytes": calibration["access_estsize"],
                            "calibration_file": calibration["file_name"],
                            "casa_version": calibration["casa_version"],
                            "qa_class": qa_class,
                            "qa_status": qa_status,
                            "qa_notes": calibration["qa_notes"],
                            "science_product_locator": row["sci_prod_locator"],
                            "calibration_product_locator": calibration[
                                "sci_prod_locator"
                            ],
                            "product_url": f"{PORTAL}/productViewer/{row['obs_id']}",
                        }
                    )
            start += len(rows)
            if not rows or start >= int(payload["n_results"]):
                break
    return records


def _evenly_spaced(records: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda record: record["obs_start"])
    if count >= len(ordered):
        return list(ordered)
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    }
    return [ordered[index] for index in sorted(indices)]


def select_balanced(
    records: Sequence[dict[str, Any]],
    sample_count: int,
    *,
    preferred_qa_class: str | None = "srdp",
) -> list[dict[str, Any]]:
    """Select observations across source, configuration, and observing date."""

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        unique.setdefault(record["calibration_file"], record)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in unique.values():
        groups[(record["search_source"], record["configuration"])].append(record)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    base, remainder = divmod(sample_count, len(keys))
    for index, key in enumerate(keys):
        count = base + (index < remainder)
        preferred = [
            record
            for record in groups[key]
            if record.get("qa_class") == preferred_qa_class
        ]
        group_selection = _evenly_spaced(preferred, min(count, len(preferred)))
        if len(group_selection) < count:
            selected_names = {
                record["calibration_file"] for record in group_selection
            }
            fallback = [
                record
                for record in groups[key]
                if record["calibration_file"] not in selected_names
            ]
            group_selection.extend(
                _evenly_spaced(fallback, count - len(group_selection))
            )
        selected.extend(group_selection)

    selected_names = {record["calibration_file"] for record in selected}
    if len(selected_names) < sample_count:
        remaining = [
            record
            for record in sorted(unique.values(), key=lambda item: item["obs_start"])
            if record["calibration_file"] not in selected_names
        ]
        selected.extend(_evenly_spaced(remaining, sample_count - len(selected_names)))
    return selected[:sample_count]


def _write_csv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _write_outputs(
    output: Path,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    arguments: argparse.Namespace,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selected_names = {record["calibration_file"] for record in selected}
    for record in records:
        record["selected"] = record["calibration_file"] in selected_names

    _write_csv(output / "candidates.csv", records)
    _write_csv(output / "selected.csv", selected)
    manifest = {
        "schema_version": 1,
        "query": {
            "sources": arguments.sources,
            "band": arguments.band,
            "configurations": arguments.configurations,
            "single_band_only": arguments.single_band_only,
        },
        "selection_policy": {
            "balanced_by": ["search_source", "configuration"],
            "date_sampling": "evenly_spaced",
            "preferred_qa_class": "srdp",
        },
        "candidate_count": len(records),
        "selected_count": len(selected),
        "selected_calibration_bytes": sum(
            int(record["calibration_bytes"]) for record in selected
        ),
        "selected_by_qa_class": {
            qa_class: sum(record["qa_class"] == qa_class for record in selected)
            for qa_class in sorted({record["qa_class"] for record in selected})
        },
        "selected": selected,
    }
    (output / "selected.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "download_files.txt").write_text(
        "".join(f"{record['calibration_file']}\n" for record in selected),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/nrao_calibration_corpus")
    )
    parser.add_argument("--sources", nargs="+", default=["3C286", "3C48"])
    parser.add_argument("--band", default="C")
    parser.add_argument("--configurations", nargs="+", default=["C", "D"])
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument(
        "--include-multiband",
        dest="single_band_only",
        action="store_false",
        help="Include observations that contain the requested band plus other bands.",
    )
    parser.set_defaults(single_band_only=True)
    arguments = parser.parse_args()
    if arguments.sample_count <= 0 or arguments.page_size <= 0:
        parser.error("--sample-count and --page-size must be positive")

    records = [
        record
        for source in arguments.sources
        for record in query_candidates(
            source,
            band=arguments.band,
            configurations=arguments.configurations,
            page_size=arguments.page_size,
            timeout_s=arguments.timeout_s,
            single_band_only=arguments.single_band_only,
        )
    ]
    selected = select_balanced(records, arguments.sample_count)
    if len(selected) < arguments.sample_count:
        parser.error(
            f"only {len(selected)} distinct products matched the requested cohort"
        )
    _write_outputs(arguments.output, records, selected, arguments=arguments)
    size_mib = sum(int(row["calibration_bytes"]) for row in selected) / 2**20
    print(
        f"{len(records)} candidates; selected {len(selected)} calibration archives "
        f"({size_mib:.1f} MiB)"
    )
    print(arguments.output / "selected.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
