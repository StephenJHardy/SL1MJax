"""Locate the field-9 scan-53→56 jump in the CASA chain and acquisition state.

Does not fit per-scan Jones. Does not open SPW 5. Does not overwrite
scientific calibration tables or the work MS.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    _row_weights,
    _spectral_window_frequencies,
    _spectral_window_id,
    apply_imported_solution,
    calibration_solution_up_to,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_ms import _read_antennas, _tables
from sl1mjax.holography_reference_jones import (
    CALIBRATION_CHAIN_STAGES,
    CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
    DO_NOT_FIT_UNCONSTRAINED_PER_SCAN_JONES_NOTE,
    HIERARCHICAL_SCAN_STATE_MODEL_NOTE,
    INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
    INTERLEAVED_ONAXIS_TRANSFER_NOTE,
    SMOOTH_GP_REJECTED_NOTE,
    classify_chain_jump_stage,
    classify_common_mode_vs_antenna,
    classify_interleaved_onaxis_transfer,
    hierarchical_residual_jones_model,
)
from sl1mjax.polarization import Correlation, Receptor, ReceptorBasis, pack_coherency

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
PRODUCT_ROOT = Path("/media/stephen/astro/vla/extracted/commissioning/products/scientific")
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
FROM_SCAN = 53
TO_SCAN = 56
NATIVE_CHANNEL = 32
HOLORASTER_FIELD = 10
FIELD9 = 9


def _cal_tables() -> dict[str, Path]:
    return {
        "antpos": PRODUCT_ROOT / "diagonal" / "antpos.cal",
        "G1": PRODUCT_ROOT / "diagonal" / "G1.cal",
        "K0": PRODUCT_ROOT / "diagonal" / "K0.cal",
        "B0": PRODUCT_ROOT / "diagonal" / "B0.cal",
        "Kcross": PRODUCT_ROOT / "fullpol" / "Kcross.cal",
        "Df": PRODUCT_ROOT / "fullpol" / "Df.cal",
        "Xf": PRODUCT_ROOT / "fullpol" / "Xf.cal",
    }


def _inventory_table(path: Path, spectral_window_id: int) -> dict[str, object]:
    tables = _tables()
    with tables.table(str(path), readonly=True, ack=False) as table:
        columns = set(table.colnames())
        time = np.asarray(table.getcol("TIME"), dtype=np.float64)
        spw = (
            np.asarray(table.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
            if "SPECTRAL_WINDOW_ID" in columns
            else np.zeros(time.size, dtype=np.int32)
        )
        keep = spw == int(spectral_window_id)
        if not np.any(keep) and np.all(spw < 0):
            keep = np.ones(time.size, dtype=bool)
        scan = (
            np.asarray(table.getcol("SCAN_NUMBER"), dtype=np.int32)
            if "SCAN_NUMBER" in columns
            else None
        )
        field = (
            np.asarray(table.getcol("FIELD_ID"), dtype=np.int32) if "FIELD_ID" in columns else None
        )
        payload = {
            "path": str(path),
            "n_row": int(np.sum(keep)),
            "n_unique_time": int(np.unique(time[keep]).size),
            "times": sorted(float(value) for value in np.unique(time[keep])),
            "has_scan_number": scan is not None,
            "has_field_id": field is not None,
        }
        if scan is not None:
            payload["scans"] = sorted(int(value) for value in np.unique(scan[keep]))
            payload["has_scan_53"] = 53 in payload["scans"]
            payload["has_scan_56"] = 56 in payload["scans"]
        if field is not None:
            payload["fields"] = sorted(int(value) for value in np.unique(field[keep]))
        if "CPARAM" in columns:
            values = np.asarray(table.getcol("CPARAM"))
            payload["cparam_shape"] = list(values.shape)
        if "FPARAM" in columns:
            values = np.asarray(table.getcol("FPARAM"))
            payload["fparam_shape"] = list(values.shape)
    return payload


def _gain_scan_jump(
    path: Path, scan_times: dict[int, float], names: np.ndarray
) -> dict[str, object]:
    tables = _tables()
    with tables.table(str(path), readonly=True, ack=False) as table:
        time = np.asarray(table.getcol("TIME"), dtype=np.float64)
        antenna = np.asarray(table.getcol("ANTENNA1"), dtype=np.int32)
        values = np.asarray(table.getcol("CPARAM"))
        flag = np.asarray(table.getcol("FLAG"), dtype=bool)
    t53 = scan_times.get(FROM_SCAN)
    t56 = scan_times.get(TO_SCAN)
    if t53 is None or t56 is None:
        return {"status": "not_run", "reason": "missing scan times"}
    by_name = {}
    for ant in np.unique(antenna):
        rows = np.flatnonzero(antenna == int(ant))
        if rows.size == 0:
            continue
        i53 = rows[int(np.argmin(np.abs(time[rows] - t53)))]
        i56 = rows[int(np.argmin(np.abs(time[rows] - t56)))]
        if bool(np.any(flag[i53])) or bool(np.any(flag[i56])):
            continue
        g53 = np.asarray(values[i53], dtype=np.complex128).reshape(-1)
        g56 = np.asarray(values[i56], dtype=np.complex128).reshape(-1)
        ratio = g56 / np.where(np.abs(g53) > 1.0e-6, g53, np.nan)
        name = str(names[int(ant)]) if int(ant) < len(names) else str(int(ant))
        by_name[name] = {
            "abs_ratio_minus_one": float(np.nanmedian(np.abs(ratio - 1.0))),
            "phase_jump_deg": float(np.nanmedian(np.rad2deg(np.angle(ratio)))),
        }
    abs_jumps = [item["abs_ratio_minus_one"] for item in by_name.values()]
    return {
        "status": "pass",
        "median_abs_ratio_minus_one": float(np.median(abs_jumps)) if abs_jumps else float("nan"),
        "max_abs_ratio_minus_one": float(np.max(abs_jumps)) if abs_jumps else float("nan"),
        "by_antenna": by_name,
    }


def _load_scans(measurement_set: Path, scans: tuple[int, ...], data_desc_id: int = 4):
    tables = _tables()
    scan_list = ",".join(str(int(scan)) for scan in scans)
    query = (
        f"SELECT FROM '{measurement_set}' "
        f"WHERE FIELD_ID={FIELD9} AND DATA_DESC_ID={int(data_desc_id)} "
        f"AND SCAN_NUMBER IN [{scan_list}]"
    )
    with tables.taql(query) as selected:
        if selected.nrows() == 0:
            raise ValueError(f"no field-9 rows for scans {scans}")
        visibility = np.asarray(selected.getcol("DATA"))
        corrected = np.asarray(selected.getcol("CORRECTED_DATA"))
        flag = np.asarray(selected.getcol("FLAG"), dtype=bool)
        visibility = np.where(flag, np.nan, visibility)
        corrected = np.where(flag, np.nan, corrected)
        _ids, names, _positions = _read_antennas(tables, measurement_set)
        names = np.asarray(names)
        with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as field_table:
            direction = np.asarray(
                field_table.getcell("PHASE_DIR", FIELD9), dtype=np.float64
            ).reshape(-1, 2)
            phase = (float(direction[0, 0]), float(direction[0, 1]))
        block = VisibilityBlock(
            uvw_m=np.asarray(selected.getcol("UVW"), dtype=np.float64),
            frequency_hz=_spectral_window_frequencies(tables, measurement_set, int(data_desc_id)),
            visibility=visibility,
            weight=_row_weights(selected, visibility.shape),
            flag=flag,
            time_s=np.asarray(selected.getcol("TIME"), dtype=np.float64),
            antenna1=np.asarray(selected.getcol("ANTENNA1"), dtype=np.int32),
            antenna2=np.asarray(selected.getcol("ANTENNA2"), dtype=np.int32),
            correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
            receptor_basis=ReceptorBasis.CIRCULAR,
            field_id=np.asarray(selected.getcol("FIELD_ID"), dtype=np.int32),
            scan_id=np.asarray(selected.getcol("SCAN_NUMBER"), dtype=np.int32),
            phase_centre_rad=phase,
            data_description_id=int(data_desc_id),
            spectral_window_id=_spectral_window_id(tables, measurement_set, int(data_desc_id)),
            provenance={"source": str(measurement_set), "column": "DATA"},
        )
    return block, corrected, names


def _pack_channel(visibility: np.ndarray, channel: int) -> np.ndarray:
    return pack_coherency(
        visibility[:, int(channel)],
        (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        (Receptor.R, Receptor.L),
    )


def _scan_rl_scores(packed: np.ndarray, scan_id: np.ndarray, antenna1, antenna2, names) -> dict:
    scores = {}
    for scan in (FROM_SCAN, TO_SCAN):
        mask = scan_id == int(scan)
        plane = packed[mask]
        intensity = np.real(0.5 * (plane[:, 0, 0] + plane[:, 1, 1]))
        scale = np.maximum(np.abs(intensity), 1.0e-3)
        rl = plane[:, 0, 1] / scale
        finite = np.isfinite(rl) & np.isfinite(intensity) & (np.abs(intensity) > 1.0)
        rl = rl[finite]
        scores[str(scan)] = {
            "n": int(rl.size),
            "median_abs_rl_over_i": float(np.median(np.abs(rl))) if rl.size else float("nan"),
            "coherent_rl_over_i": float(np.abs(np.mean(rl))) if rl.size else float("nan"),
            "median_i": float(np.median(intensity[finite])) if np.any(finite) else float("nan"),
        }
        by_ant = {}
        for ant in np.unique(np.concatenate([antenna1[mask], antenna2[mask]])):
            rows = mask & ((antenna1 == ant) | (antenna2 == ant))
            sub = packed[rows]
            inten = np.real(0.5 * (sub[:, 0, 0] + sub[:, 1, 1]))
            usable = np.isfinite(sub[:, 0, 1]) & np.isfinite(inten) & (np.abs(inten) > 1.0)
            if int(np.sum(usable)) < 8:
                continue
            name = str(names[int(ant)]) if int(ant) < len(names) else str(int(ant))
            by_ant[name] = complex(
                np.mean(sub[usable, 0, 1] / np.maximum(np.abs(inten[usable]), 1.0e-3))
            )
        scores[str(scan)]["by_antenna"] = {
            name: [value.real, value.imag] for name, value in by_ant.items()
        }
        scores[str(scan)]["_by_antenna_complex"] = by_ant
    left = scores[str(FROM_SCAN)]
    right = scores[str(TO_SCAN)]
    jump = float(right["coherent_rl_over_i"]) - float(left["coherent_rl_over_i"])
    deltas = {}
    for name, value in (right.get("_by_antenna_complex") or {}).items():
        if name in (left.get("_by_antenna_complex") or {}):
            deltas[name] = value - left["_by_antenna_complex"][name]
    scores["jump_coherent_rl_over_i"] = abs(jump)
    scores["signed_jump_coherent_rl_over_i"] = jump
    scores["common_mode"] = classify_common_mode_vs_antenna(deltas)
    return scores


def _within_scan_settling(packed, time_s, scan_id) -> dict[str, object]:
    report = {}
    for scan in (FROM_SCAN, TO_SCAN):
        mask = scan_id == int(scan)
        if not np.any(mask):
            continue
        times = time_s[mask]
        order = np.argsort(times)
        plane = packed[mask][order]
        intensity = np.real(0.5 * (plane[:, 0, 0] + plane[:, 1, 1]))
        rl = plane[:, 0, 1] / np.maximum(np.abs(intensity), 1.0e-3)
        n = rl.size
        first = rl[: max(n // 5, 1)]
        settled = rl[max(n // 5, 1) :]
        report[str(scan)] = {
            "n": int(n),
            "duration_s": float(times.max() - times.min()) if n else float("nan"),
            "first_coherent_abs": float(np.abs(np.mean(first[np.isfinite(first)])))
            if first.size
            else float("nan"),
            "settled_coherent_abs": float(np.abs(np.mean(settled[np.isfinite(settled)])))
            if settled.size
            else float("nan"),
        }
    return report


def _pointing_state(
    measurement_set: Path, scan_times: dict[int, tuple[float, float]], names
) -> dict:
    tables = _tables()
    with tables.table(str(measurement_set / "POINTING"), readonly=True, ack=False) as pointing:
        columns = set(pointing.colnames())
        time = np.asarray(pointing.getcol("TIME"), dtype=np.float64)
        antenna = np.asarray(pointing.getcol("ANTENNA_ID"), dtype=np.int32)
        offset = (
            np.asarray(pointing.getcol("POINTING_OFFSET"), dtype=np.float64)
            if "POINTING_OFFSET" in columns
            else None
        )
    if offset is None:
        return {"status": "not_run", "reason": "no POINTING_OFFSET"}
    report = {}
    for scan, (start, stop) in scan_times.items():
        mid = 0.5 * (start + stop)
        by_ant = {}
        for ant in np.unique(antenna):
            rows = np.flatnonzero(antenna == int(ant))
            if rows.size == 0:
                continue
            index = rows[int(np.argmin(np.abs(time[rows] - mid)))]
            plane = np.asarray(offset[index]).reshape(-1)
            if plane.size < 2:
                continue
            name = str(names[int(ant)]) if int(ant) < len(names) else str(int(ant))
            by_ant[name] = {
                "l_arcmin": float(np.rad2deg(plane[0]) * 60.0),
                "m_arcmin": float(np.rad2deg(plane[1]) * 60.0),
                "radius_arcmin": float(np.hypot(np.rad2deg(plane[0]), np.rad2deg(plane[1])) * 60.0),
            }
        radii = [item["radius_arcmin"] for item in by_ant.values()]
        report[str(scan)] = {
            "median_radius_arcmin": float(np.median(radii)) if radii else float("nan"),
            "max_radius_arcmin": float(np.max(radii)) if radii else float("nan"),
            "by_antenna": by_ant,
        }
    return report


def _scan_neighbours(measurement_set: Path) -> dict[str, object]:
    tables = _tables()
    with tables.table(str(measurement_set), readonly=True, ack=False) as main:
        time = np.asarray(main.getcol("TIME"), dtype=np.float64)
        scan = np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
        field = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
    first: dict[int, tuple[float, int]] = {}
    last: dict[int, float] = {}
    for instant, scan_id, field_id in zip(time, scan, field, strict=True):
        current = first.get(int(scan_id))
        if current is None or float(instant) < current[0]:
            first[int(scan_id)] = (float(instant), int(field_id))
        last[int(scan_id)] = max(float(instant), last.get(int(scan_id), float(instant)))
    ordered = sorted(first, key=lambda scan_id: first[scan_id][0])
    info = {}
    for index, scan_id in enumerate(ordered):
        pred = (
            None
            if index == 0
            else {
                "scan": int(ordered[index - 1]),
                "field": int(first[ordered[index - 1]][1]),
                "gap_s": float(first[scan_id][0] - last[ordered[index - 1]]),
            }
        )
        info[str(int(scan_id))] = {
            "field": int(first[scan_id][1]),
            "t0": first[scan_id][0],
            "t1": last[scan_id],
            "predecessor": pred,
            "preceded_by_holoraster": bool(pred is not None and pred["field"] == HOLORASTER_FIELD),
        }
    between = [scan_id for scan_id in ordered if FROM_SCAN < int(scan_id) < TO_SCAN]
    return {
        "scan_53": info.get(str(FROM_SCAN)),
        "scan_56": info.get(str(TO_SCAN)),
        "between": [
            {"scan": int(scan_id), "field": info[str(int(scan_id))]["field"]} for scan_id in between
        ],
        "all": info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--spectral-window", type=int, default=4)
    arguments = parser.parse_args()
    if int(arguments.spectral_window) != 4:
        raise ValueError("SPW 5 stays closed until the SPW-4 separation succeeds")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    tables = _cal_tables()
    inventory = {name: _inventory_table(path, 4) for name, path in tables.items() if path.exists()}
    block, casa_corrected, names = _load_scans(arguments.measurement_set, (FROM_SCAN, TO_SCAN))
    native = min(int(NATIVE_CHANNEL), block.frequency_hz.size - 1)
    block = replace(
        block,
        frequency_hz=block.frequency_hz[native : native + 1],
        visibility=block.visibility[:, native : native + 1],
        weight=block.weight[:, native : native + 1],
        flag=block.flag[:, native : native + 1],
    )
    casa_corrected = casa_corrected[:, native : native + 1]
    scan_times = {
        int(scan): float(np.median(block.time_s[block.scan_id == int(scan)]))
        for scan in (FROM_SCAN, TO_SCAN)
        if np.any(block.scan_id == int(scan))
    }
    gain_jump = _gain_scan_jump(tables["G1"], scan_times, names)
    solution = import_thol0001_casa_tables(
        tables,
        measurement_set=arguments.measurement_set,
        spectral_window_id=int(block.spectral_window_id),
        product="fullpol",
    )
    stage_scores = {}
    stage_jumps = {}
    for stage in CALIBRATION_CHAIN_STAGES:
        prefix = calibration_solution_up_to(solution, stage)
        vis = (
            block.visibility
            if prefix is None
            else apply_imported_solution(block, prefix).visibility
        )
        packed = _pack_channel(vis, 0)
        scores = _scan_rl_scores(packed, block.scan_id, block.antenna1, block.antenna2, names)
        stage_jumps[stage] = float(scores["jump_coherent_rl_over_i"])
        stage_scores[stage] = {
            key: value
            for key, value in scores.items()
            if key != "_by_antenna_complex" and not str(key).startswith("_")
        }
        for scan_key in (str(FROM_SCAN), str(TO_SCAN)):
            stage_scores[stage][scan_key].pop("_by_antenna_complex", None)
    casa_packed = _pack_channel(casa_corrected, 0)
    casa_scores = _scan_rl_scores(casa_packed, block.scan_id, block.antenna1, block.antenna2, names)
    casa_scores[str(FROM_SCAN)].pop("_by_antenna_complex", None)
    casa_scores[str(TO_SCAN)].pop("_by_antenna_complex", None)
    chain = classify_chain_jump_stage(stage_jumps)
    neighbours = _scan_neighbours(arguments.measurement_set)
    span = {
        int(scan): (
            float(neighbours["all"][str(scan)]["t0"]),
            float(neighbours["all"][str(scan)]["t1"]),
        )
        for scan in (FROM_SCAN, TO_SCAN)
        if str(scan) in neighbours["all"]
    }
    pointing = _pointing_state(arguments.measurement_set, span, names)
    packed_data = _pack_channel(block.visibility, 0)
    settling = _within_scan_settling(packed_data, block.time_s, block.scan_id)
    acquisition = {
        "settling": settling,
        "pointing_offset": pointing,
        "neighbours": {
            "scan_53": neighbours.get("scan_53"),
            "scan_56": neighbours.get("scan_56"),
            "between": neighbours.get("between"),
        },
        "run_because_present_in_data": chain["investigate_acquisition_state"],
    }
    transfer = classify_interleaved_onaxis_transfer(
        chain_stage=chain["first_increment_above_threshold"]
        or chain["first_stage_above_threshold"],
        visit_repeatability_passed=None,
        used_holography_crosshands_for_onaxis=False,
        smooth_gp_rejected=True,
    )
    payload = {
        "schema": "thol0001_scan53_56_chain_jump_v1",
        "spectral_window_id": 4,
        "native_channel": native,
        "from_scan": FROM_SCAN,
        "to_scan": TO_SCAN,
        "table_inventory": inventory,
        "g1_scan_jump": gain_jump,
        "stages": stage_scores,
        "casa_corrected": {
            key: value for key, value in casa_scores.items() if key != "_by_antenna_complex"
        },
        "chain": chain,
        "acquisition": acquisition,
        "hierarchical_model": hierarchical_residual_jones_model(),
        "interleaved_onaxis_transfer": transfer,
        "notes": [
            SMOOTH_GP_REJECTED_NOTE,
            INTERLEAVED_ONAXIS_TRANSFER_NOTE,
            DO_NOT_FIT_UNCONSTRAINED_PER_SCAN_JONES_NOTE,
            HIERARCHICAL_SCAN_STATE_MODEL_NOTE,
            INTERLEAVED_BEAM_TRANSFER_GATE_NOTE,
            CASA_FIELD9_NOT_END_TO_END_HOLDOUT_NOTE,
        ],
    }
    path = arguments.output_dir / "scan53_56_chain_jump.json"
    write_json(payload, path)
    print(path)
    print(
        "first_stage",
        chain["first_stage_above_threshold"],
        "first_increment",
        chain["first_increment_above_threshold"],
    )
    print(
        "present_in_data",
        chain["present_in_data"],
        "G1 median |Δ|",
        gain_jump.get("median_abs_ratio_minus_one"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
