"""Check that fields 0 and 9 share one 3C147 model and that cal stays put.

The residual-calibration gate is the per-row complex ratio
CORRECTED_DATA / MODEL_DATA on reference-reference baselines. Amplitude
versus a scalar median MODEL can look like time drift when UV evolves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sl1mjax.holography import three_c147_flux_scale_report
from sl1mjax.holography_calibration import (
    CORRECTED_OVER_MODEL_RATIO_NOTE,
    corrected_over_model_ratio_is_stable,
    flux_gauge_is_consistent,
    write_json,
)

DEFAULT_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
REFERENCE_NAMES = ("ea02", "ea03", "ea07", "ea12", "ea14", "ea24", "ea26")
CHANNEL = 32
SPW = 4
FIELDS = (0, 9, 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", nargs="?", type=Path, default=DEFAULT_MS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", type=int, default=CHANNEL)
    parser.add_argument("--spw", type=int, default=SPW)
    arguments = parser.parse_args()
    report = report_flux_gauge(
        arguments.measurement_set,
        channel=arguments.channel,
        spectral_window_id=arguments.spw,
    )
    write_json(report, arguments.output)
    print(arguments.output)
    print(
        "fields_0_and_9_consistent",
        report["field_model_gauge"]["consistent"],
        "corrected_over_model",
        {
            field: {hand: payload["passed"] for hand, payload in series.items()}
            for field, series in report["corrected_over_model"].items()
        },
    )
    return 0 if report["passed"] else 1


def report_flux_gauge(
    measurement_set: Path,
    *,
    channel: int,
    spectral_window_id: int,
) -> dict:
    from sl1mjax.holography_ms import _tables

    tables = _tables()
    names = _antenna_names(tables, measurement_set)
    ref_ids = tuple(int(i) for i, name in enumerate(names) if name in REFERENCE_NAMES)
    ddid = _ddid_for_spw(tables, measurement_set, spectral_window_id)
    frequency = _channel_frequency(tables, measurement_set, spectral_window_id, channel)
    columns = _main_columns(tables, measurement_set)
    field_names = _field_names(tables, measurement_set)
    models = {}
    ratios = {}
    fields = {}
    for field_id in FIELDS:
        payload = _field_ref_ref(
            tables,
            measurement_set,
            field_id=field_id,
            data_desc_id=ddid,
            channel=channel,
            reference_ids=ref_ids,
            columns=columns,
        )
        payload["name"] = field_names[field_id] if field_id < len(field_names) else ""
        fields[str(field_id)] = payload
        model_rr = payload["MODEL_DATA"]["median_rr"]
        if field_id in (0, 9) and np.isfinite(model_rr):
            models[field_id] = float(model_rr)
        ratios[str(field_id)] = payload["corrected_over_model"]
    gauge = (
        flux_gauge_is_consistent(models)
        if len(models) == 2
        else {
            "consistent": False,
            "scientific_recovery_blocked": True,
            "reason": "MODEL_DATA missing on field 0 or 9",
        }
    )
    flux = three_c147_flux_scale_report(frequency)
    ratio_ok = all(
        hand.get("passed")
        for field in ratios.values()
        for hand in field.values()
        if isinstance(hand, dict) and "passed" in hand
    )
    passed = bool(gauge.get("consistent")) and bool(ratio_ok) and bool(ratios)
    return {
        "schema": "thol0001_flux_gauge_v2",
        "measurement_set": str(measurement_set),
        "frequency_hz": frequency,
        "spectral_window_id": spectral_window_id,
        "channel": channel,
        "flux_scale": flux,
        "field_model_gauge": gauge,
        "corrected_over_model": {
            field: {hand: _without_baseline_detail(payload) for hand, payload in series.items()}
            for field, series in ratios.items()
        },
        "fields": {
            key: {
                "name": value["name"],
                "n_reference_reference": value["n_reference_reference"],
                "MODEL_DATA": _without_series(value["MODEL_DATA"]),
                "CORRECTED_DATA": _without_series(value["CORRECTED_DATA"]),
                "DATA": _without_series(value["DATA"]),
            }
            for key, value in fields.items()
        },
        "passed": passed,
        "notes": [
            "Fields 0 and 9 must share one 3C147 MODEL_DATA scale",
            CORRECTED_OVER_MODEL_RATIO_NOTE,
            "Test amplitude and phase of CORRECTED/MODEL against time, field, baseline and hand",
            "A mixed G scale can become apparent beam structure because raster "
            "position is correlated with time",
            "Per-antenna on-axis normalization must not replace this gate",
        ],
    }


def _without_series(payload: dict) -> dict:
    return {
        key: value for key, value in payload.items() if key not in {"amp_rr", "amp_ll", "time_s"}
    }


def _without_baseline_detail(payload: dict) -> dict:
    summary = dict(payload)
    baselines = summary.get("by_baseline")
    if isinstance(baselines, dict):
        summary["n_baselines"] = len(baselines)
        summary["failed_baselines"] = [
            name for name, item in baselines.items() if not item.get("passed")
        ]
        summary["by_baseline"] = {
            name: {
                key: item[key]
                for key in (
                    "n",
                    "median_abs",
                    "relative_amp_drift",
                    "median_phase_rad",
                    "phase_drift_rad",
                    "passed",
                )
                if key in item
            }
            for name, item in baselines.items()
        }
    return summary


def _field_ref_ref(
    tables,
    measurement_set: Path,
    *,
    field_id: int,
    data_desc_id: int,
    channel: int,
    reference_ids: tuple[int, ...],
    columns: set[str],
) -> dict:
    query = (
        f"SELECT FROM '{measurement_set}' "
        f"WHERE FIELD_ID={int(field_id)} AND DATA_DESC_ID={int(data_desc_id)}"
    )
    with tables.taql(query) as selected:
        if selected.nrows() == 0:
            return {
                "n_rows": 0,
                "n_reference_reference": 0,
                "DATA": _empty_series(),
                "CORRECTED_DATA": _empty_series(),
                "MODEL_DATA": _empty_series(),
                "corrected_over_model": {
                    "RR": {"n": 0, "passed": False, "reason": "no rows"},
                    "LL": {"n": 0, "passed": False, "reason": "no rows"},
                },
            }
        antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32)
        antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32)
        time_s = np.asarray(selected.getcol("TIME"), dtype=np.float64)
        flag = np.asarray(selected.getcolslice("FLAG", [channel, 0], [channel, -1]), dtype=bool)
        if flag.ndim == 3:
            flag = flag[:, 0]
        data = _optional_slice(selected, "DATA", channel, columns)
        corrected = _optional_slice(selected, "CORRECTED_DATA", channel, columns)
        model = _optional_slice(selected, "MODEL_DATA", channel, columns)
    refs = set(reference_ids)
    ref_ref = np.array(
        [
            int(a) in refs and int(b) in refs and int(a) != int(b)
            for a, b in zip(antenna1, antenna2, strict=True)
        ]
    )
    return {
        "n_rows": int(antenna1.size),
        "n_reference_reference": int(np.sum(ref_ref)),
        "DATA": _series(data, flag, ref_ref, time_s),
        "CORRECTED_DATA": _series(corrected, flag, ref_ref, time_s),
        "MODEL_DATA": _series(model, flag, ref_ref, time_s),
        "corrected_over_model": {
            "RR": _hand_ratio(corrected, model, flag, ref_ref, time_s, antenna1, antenna2, 0),
            "LL": _hand_ratio(corrected, model, flag, ref_ref, time_s, antenna1, antenna2, 3),
        },
    }


def _empty_series() -> dict:
    return {
        "n": 0,
        "median_rr": float("nan"),
        "median_ll": float("nan"),
        "amp_rr": [],
        "amp_ll": [],
        "time_s": [],
    }


def _series(visibility, flag, rows, time_s) -> dict:
    if visibility is None:
        return _empty_series()
    vis = np.asarray(visibility)
    if vis.ndim == 3:
        vis = vis[:, 0]
    usable = rows & ~flag[:, 0] & np.isfinite(vis[:, 0])
    rr = np.abs(vis[usable, 0])
    ll_usable = rows & ~flag[:, 3] & np.isfinite(vis[:, 3])
    ll = np.abs(vis[ll_usable, 3])
    return {
        "n": int(np.sum(usable)),
        "median_rr": float(np.median(rr)) if rr.size else float("nan"),
        "median_ll": float(np.median(ll)) if ll.size else float("nan"),
        "p16_rr": float(np.percentile(rr, 16)) if rr.size else float("nan"),
        "p84_rr": float(np.percentile(rr, 84)) if rr.size else float("nan"),
        "amp_rr": rr.tolist(),
        "amp_ll": ll.tolist(),
        "time_s": np.asarray(time_s, dtype=np.float64)[usable].tolist(),
    }


def _hand_ratio(
    corrected,
    model,
    flag,
    rows,
    time_s,
    antenna1,
    antenna2,
    correlation: int,
) -> dict:
    if corrected is None or model is None:
        return {
            "n": 0,
            "passed": False,
            "reason": "CORRECTED_DATA or MODEL_DATA missing",
        }
    measured = _corr(corrected, correlation)
    predicted = _corr(model, correlation)
    flagged = _corr(flag, correlation).astype(bool)
    usable = rows & ~flagged & np.isfinite(measured) & np.isfinite(predicted)
    if int(np.sum(usable)) < 4:
        return {
            "n": int(np.sum(usable)),
            "passed": False,
            "reason": "fewer than four unflagged reference-reference samples",
        }
    return corrected_over_model_ratio_is_stable(
        measured[usable],
        predicted[usable],
        np.asarray(time_s, dtype=np.float64)[usable],
        antenna1=np.asarray(antenna1)[usable],
        antenna2=np.asarray(antenna2)[usable],
    )


def _corr(values, correlation: int) -> np.ndarray:
    vis = np.asarray(values)
    if vis.ndim == 3:
        vis = vis[:, 0]
    return vis[:, correlation]


def _optional_slice(selected, column: str, channel: int, columns: set[str]):
    if column not in columns:
        return None
    return np.asarray(selected.getcolslice(column, [channel, 0], [channel, -1]))


def _main_columns(tables, measurement_set: Path) -> set[str]:
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        return set(main.colnames())


def _antenna_names(tables, measurement_set: Path) -> list[str]:
    with tables.table(str(measurement_set / "ANTENNA"), readonly=True, ack=False) as antenna:
        return [str(name) for name in antenna.getcol("NAME")]


def _field_names(tables, measurement_set: Path) -> list[str]:
    with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field:
        return [str(name) for name in field.getcol("NAME")]


def _channel_frequency(
    tables, measurement_set: Path, spectral_window_id: int, channel: int
) -> float:
    with tables.table(str(measurement_set / "SPECTRAL_WINDOW"), readonly=True, ack=False) as window:
        return float(
            np.asarray(window.getcell("CHAN_FREQ", spectral_window_id)).reshape(-1)[channel]
        )


def _ddid_for_spw(tables, measurement_set: Path, spectral_window_id: int) -> int:
    with tables.table(str(measurement_set / "DATA_DESCRIPTION"), readonly=True, ack=False) as table:
        windows = np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
    matches = np.flatnonzero(windows == int(spectral_window_id))
    if matches.size != 1:
        raise ValueError(f"expected one DATA_DESC_ID for SPW {spectral_window_id}")
    return int(matches[0])


if __name__ == "__main__":
    raise SystemExit(main())
