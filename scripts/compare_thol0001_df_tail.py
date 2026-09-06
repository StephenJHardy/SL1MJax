"""Stratify the Df vs Df+QU tail. Do not treat p90 as a leakage floor yet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from casacore import tables

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_oracle import stratify_df_delta

REFERENCE = ("ea02", "ea03", "ea07", "ea12", "ea14", "ea24", "ea26")
MOVING = (
    "ea04",
    "ea05",
    "ea06",
    "ea08",
    "ea09",
    "ea10",
    "ea11",
    "ea13",
    "ea15",
    "ea16",
    "ea17",
    "ea18",
    "ea19",
    "ea20",
    "ea21",
    "ea22",
    "ea23",
    "ea25",
    "ea27",
    "ea28",
)


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    with tables.table(str(path), readonly=True, ack=False) as table:
        antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
        spw = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        flag = np.asarray(table.getcol("FLAG"), dtype=bool)
        values = np.asarray(table.getcol("CPARAM"))
        snr = np.asarray(table.getcol("SNR")) if "SNR" in table.colnames() else None
    return antenna, spw, flag, values, snr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--df", type=Path, required=True)
    parser.add_argument("--df-qu", type=Path, required=True)
    parser.add_argument("--measurement-set", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with tables.table(
        str(arguments.measurement_set / "ANTENNA"), readonly=True, ack=False
    ) as table:
        names = tuple(str(name) for name in table.getcol("NAME"))
    a1, s1, f1, v1, snr1 = _read(arguments.df)
    a2, s2, f2, v2, snr2 = _read(arguments.df_qu)
    if v1.shape != v2.shape or not np.array_equal(a1, a2):
        raise ValueError("Df and Df+QU tables are not aligned")
    report = stratify_df_delta(
        v1,
        v2,
        ~f1 & ~f2,
        antenna=a1,
        spectral_window_id=s1,
        antenna_names=names,
        reference_antennas=REFERENCE,
        moving_antennas=MOVING,
        snr=snr1,
    )
    report["df_keywords"] = _keywords(arguments.df)
    report["df_qu_keywords"] = _keywords(arguments.df_qu)
    report["df_qu_source_pol"] = _source_pol(arguments.df_qu)
    print(json_summary(report))
    if arguments.output is not None:
        write_json(report, arguments.output)
        print(arguments.output)
    return 0


def _keywords(path: Path) -> dict[str, object]:
    with tables.table(str(path), readonly=True, ack=False) as table:
        names = list(table.keywordnames())
        payload = {}
        for name in names:
            value = table.getkeyword(name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[name] = value
            elif isinstance(value, (list, tuple)) and len(value) <= 8:
                payload[name] = list(value)
        return payload


def _source_pol(path: Path) -> dict[str, object]:
    """Extract Q/U stored by CASA Df+QU, if the table recorded them."""

    with tables.table(str(path), readonly=True, ack=False) as table:
        keywords = {name: table.getkeyword(name) for name in table.keywordnames()}
    payload: dict[str, object] = {"found": False}
    for key, value in keywords.items():
        name = str(key).lower()
        if any(token in name for token in ("stokesq", "stokesu", "sourceq", "sourceu", "qu")):
            payload[str(key)] = value if isinstance(value, (str, int, float, list)) else str(value)
            payload["found"] = True
    if "PolnCalib" in keywords:
        payload["PolnCalib"] = str(keywords["PolnCalib"])
        payload["found"] = True
    return payload


def json_summary(report: dict) -> str:
    import json

    return json.dumps(
        {
            "median_abs_delta": report["median_abs_delta"],
            "p90_abs_delta": report["p90_abs_delta"],
            "n_tail": report["n_tail"],
            "edge_channel_tail_fraction": report["edge_channel_tail_fraction"],
            "gauge_aligned": report.get("gauge_aligned"),
            "concentrated_in_failed_subset": report["concentrated_in_failed_subset"],
            "adopt_as_floor": report["adopt_as_floor"],
            "df_qu_source_pol": report.get("df_qu_source_pol"),
            "worst_antennas": sorted(
                report["by_antenna"], key=lambda item: item["n_tail"], reverse=True
            )[:8],
        },
        indent=2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
