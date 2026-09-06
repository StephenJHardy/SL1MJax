"""Measure 4.564 GHz amplitudes before attaching Perley-Butler and rerunning.

This does not recover a beam. It reports DATA, CORRECTED_DATA, and
MODEL_DATA so the 0.18 on-axis voltage can be attributed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.holography import perley_butler_2017_3c147_stokes_i_jy
from sl1mjax.holography_calibration import write_json
from sl1mjax.polarization import circular_stokes_to_coherency

DEFAULT_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.work.ms"
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
    report = diagnose_onaxis_amplitude(
        arguments.measurement_set,
        channel=arguments.channel,
        spectral_window_id=arguments.spw,
    )
    write_json(report, arguments.output)
    print(
        json.dumps({k: report[k] for k in report if k != "field_rows"}, indent=2, default=str)[
            :4000
        ]
    )
    print(arguments.output)
    return 0


def diagnose_onaxis_amplitude(
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
    field_names = _field_names(tables, measurement_set)
    packing = circular_stokes_to_coherency(1.0, 0.0, 0.0, 0.0)
    columns = _main_columns(tables, measurement_set)
    fields = {}
    for field_id in FIELDS:
        fields[str(field_id)] = _field_amplitudes(
            tables,
            measurement_set,
            field_id=field_id,
            data_desc_id=ddid,
            channel=channel,
            reference_ids=ref_ids,
            columns=columns,
        )
        fields[str(field_id)]["name"] = field_names[field_id] if field_id < len(field_names) else ""
    stokes_i = float(np.asarray(perley_butler_2017_3c147_stokes_i_jy(frequency)).reshape(-1)[0])
    return {
        "schema": "onaxis_amplitude_v1",
        "frozen": False,
        "measurement_set": str(measurement_set),
        "frequency_hz": frequency,
        "spectral_window_id": spectral_window_id,
        "channel": channel,
        "data_desc_id": ddid,
        "columns": sorted(columns),
        "reference_antenna_ids": list(ref_ids),
        "reference_antenna_names": [names[i] for i in ref_ids],
        "perley_butler_2017_3c147_stokes_i_jy": stokes_i,
        "circular_stokes_packing": {
            "RR": "I+V",
            "LL": "I-V",
            "unpolarized_RR": float(packing[0, 0].real),
            "unpolarized_LL": float(packing[1, 1].real),
            "not_I_over_2": True,
        },
        "fields": fields,
        "apply_direction": _apply_direction(fields),
        "notes": (
            "S=1 makes recovered on-axis |E| equal |CORRECTED| if E_r=1",
            "Replacing S=1 with Perley-Butler divides |E| further; it cannot raise 0.18 to 1",
            "R/L amplitude difference cannot be removed by a scalar spectrum",
            "Normalize each moving antenna by its measured on-axis response for beam shape",
            "Use Perley-Butler to validate the flux scale, not as the only beam normalisation",
        ),
    }


def _field_amplitudes(
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
            return {"n_rows": 0}
        antenna1 = np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32)
        antenna2 = np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32)
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
    moving_ref = np.array(
        [(int(a) in refs) != (int(b) in refs) for a, b in zip(antenna1, antenna2, strict=True)]
    )
    return {
        "n_rows": int(antenna1.size),
        "n_reference_reference": int(np.sum(ref_ref)),
        "n_moving_reference": int(np.sum(moving_ref)),
        "DATA": _hand_stats(data, flag, ref_ref, moving_ref),
        "CORRECTED_DATA": _hand_stats(corrected, flag, ref_ref, moving_ref),
        "MODEL_DATA": _hand_stats(model, flag, ref_ref, moving_ref),
    }


def _hand_stats(
    visibility: np.ndarray | None,
    flag: np.ndarray,
    ref_ref: np.ndarray,
    moving_ref: np.ndarray,
) -> dict | None:
    if visibility is None:
        return None
    vis = np.asarray(visibility)
    if vis.ndim == 3:
        vis = vis[:, 0]
    return {
        "reference_reference": _amp_pair(vis, flag, ref_ref),
        "moving_reference": _amp_pair(vis, flag, moving_ref),
        "all_unflagged": _amp_pair(vis, flag, np.ones(vis.shape[0], dtype=bool)),
    }


def _amp_pair(visibility: np.ndarray, flag: np.ndarray, rows: np.ndarray) -> dict:
    report = {}
    for name, corr in (("RR", 0), ("LL", 3)):
        usable = rows & ~flag[:, corr] & np.isfinite(visibility[:, corr])
        values = np.abs(visibility[usable, corr])
        report[name] = {
            "n": int(np.sum(usable)),
            "median_amp": float(np.median(values)) if values.size else float("nan"),
            "p16_amp": float(np.percentile(values, 16)) if values.size else float("nan"),
            "p84_amp": float(np.percentile(values, 84)) if values.size else float("nan"),
        }
    if report["RR"]["n"] and report["LL"]["n"]:
        report["rl_median_ratio"] = report["RR"]["median_amp"] / report["LL"]["median_amp"]
    else:
        report["rl_median_ratio"] = float("nan")
    return report


def _apply_direction(fields: dict) -> dict:
    field0 = fields.get("0", {})
    data = (field0.get("DATA") or {}).get("reference_reference") or {}
    corr = (field0.get("CORRECTED_DATA") or {}).get("reference_reference") or {}
    model = (field0.get("MODEL_DATA") or {}).get("reference_reference") or {}
    data_rr = (data.get("RR") or {}).get("median_amp", float("nan"))
    corr_rr = (corr.get("RR") or {}).get("median_amp", float("nan"))
    model_rr = (model.get("RR") or {}).get("median_amp", float("nan"))
    diagnosis = "undetermined"
    if np.isfinite(data_rr) and np.isfinite(corr_rr) and data_rr > 0.0:
        ratio = corr_rr / data_rr
        if np.isfinite(model_rr) and model_rr > 0.0:
            closer_to_model = abs(np.log(corr_rr / model_rr)) < abs(np.log(data_rr / model_rr))
            if closer_to_model and ratio > 1.0:
                diagnosis = "correction_toward_model"
            elif ratio < 0.5 and corr_rr < data_rr and corr_rr < model_rr:
                diagnosis = "possible_forward_apply_or_double_corrupt"
            elif abs(corr_rr / model_rr - 1.0) < 0.3:
                diagnosis = "correction_toward_model"
            else:
                diagnosis = "corrected_moved_toward_model_but_median_below_setjy"
        elif ratio < 0.5:
            diagnosis = "corrected_amplitude_much_smaller_than_data"
        else:
            diagnosis = "corrected_near_data"
    return {
        "field0_reference_reference_RR": {
            "DATA": data_rr,
            "CORRECTED_DATA": corr_rr,
            "MODEL_DATA": model_rr,
            "corrected_over_data": corr_rr / data_rr if data_rr else float("nan"),
            "corrected_over_model": corr_rr / model_rr if model_rr else float("nan"),
        },
        "diagnosis": diagnosis,
    }


def _optional_slice(selected, column: str, channel: int, columns: set[str]) -> np.ndarray | None:
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
