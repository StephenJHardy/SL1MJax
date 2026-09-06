"""Recover CASA χ and compare casacore measures across fields and antennas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.calibration_terms import parallactic_angle_rad
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_measures import (
    parallactic_angle_casacore_rad,
    read_field_direction_frame,
    read_ms_time_reference,
    read_receptor_angle_rad,
)
from sl1mjax.holography_calibration_oracle import (
    classify_chi_residual,
    effective_chi_from_p_operators,
    operator_from_basis_outputs,
    xf_operator_effect,
)
from sl1mjax.holography_ms import _read_field_names, _read_field_phase_centre, _tables


def _casa_operator(root: Path, stage: str, case: str, channel: int) -> np.ndarray:
    tables = _tables()
    outputs = np.zeros((4, 4), dtype=np.complex128)
    for index, name in enumerate(("RR", "RL", "LR", "LL")):
        path = root / "applied" / stage / f"{case}_{name}.ms"
        with tables.table(str(path), readonly=True, ack=False) as main:
            outputs[index] = np.asarray(main.getcol("CORRECTED_DATA"))[0, int(channel)]
    return operator_from_basis_outputs(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--measurement-set", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    tables = _tables()
    names = _read_field_names(tables, arguments.measurement_set)
    time_reference = read_ms_time_reference(arguments.measurement_set)
    receptor = read_receptor_angle_rad(arguments.measurement_set)
    root = arguments.product_root
    cal_tables = {
        "antpos": root / "diagonal" / "antpos.cal",
        "G1": root / "diagonal" / "G1.cal",
        "K0": root / "diagonal" / "K0.cal",
        "B0": root / "diagonal" / "B0.cal",
        "Kcross": root / "fullpol" / "Kcross.cal",
        "Df": root / "fullpol" / "Df.cal",
        "Xf": root / "fullpol" / "Xf.cal",
    }
    manifest = json.loads((arguments.oracle_root / "oracle_cases.json").read_text())
    channels = tuple(int(item) for item in manifest["channels"])
    reports = []
    residuals = []
    hour_angles = []
    antennas = []
    for case in manifest["cases"]:
        if case["basis"] != "RR":
            continue
        label = case["label"]
        block, _, _ = golden_visibility_block(Path(case["ms"]), data_column="DATA", data_desc_id=4)
        full = import_thol0001_casa_tables(
            cal_tables,
            measurement_set=Path(case["ms"]),
            spectral_window_id=block.spectral_window_id,
            product="fullpol",
        )
        measures_chi = parallactic_angle_casacore_rad(
            block.time_s,
            block.phase_centre_rad,
            full.antenna_position_m,
            time_reference=time_reference,
            receptor_angle_rad=receptor,
        )
        gmst_chi = parallactic_angle_rad(
            block.time_s, block.phase_centre_rad, full.antenna_position_m
        )
        p_ant = int(block.antenna1[0])
        q_ant = int(block.antenna2[0])
        predicted = float(measures_chi[0, p_ant] + measures_chi[0, q_ant])
        for channel in channels:
            casa_p = _casa_operator(arguments.oracle_root, "K_B_G_Kcross_Df_Xf_P", label, channel)
            casa_x = _casa_operator(arguments.oracle_root, "K_B_G_Kcross_Df_Xf", label, channel)
            casa_d = _casa_operator(arguments.oracle_root, "K_B_G_Kcross_Df", label, channel)
            recovered = effective_chi_from_p_operators(casa_p, casa_x)
            xf_effect = xf_operator_effect(casa_x, casa_d)
            residual_deg = float(
                np.rad2deg(np.angle(np.exp(1j * (recovered["two_chi_rad"] - predicted))))
            )
            item = {
                "label": label,
                "channel": channel,
                "antenna1": p_ant,
                "antenna2": q_ant,
                "time_s": float(block.time_s[0]),
                "casa": recovered,
                "measures_two_chi_deg": float(np.rad2deg(predicted)),
                "gmst_two_chi_deg": float(np.rad2deg(gmst_chi[0, p_ant] + gmst_chi[0, q_ant])),
                "residual_vs_measures_deg": residual_deg,
                "xf_effect": xf_effect,
                "receptor_angle_rad": [float(receptor[p_ant]), float(receptor[q_ant])],
                "time_reference": time_reference,
            }
            reports.append(item)
            residuals.append(residual_deg)
            hour_angles.append(float(measures_chi[0, p_ant]))
            antennas.append(p_ant)
            print(
                label,
                "ch",
                channel,
                "CASA",
                f"{recovered['two_chi_deg']:.6f}",
                "measures",
                f"{np.rad2deg(predicted):.6f}",
                "ddeg",
                f"{residual_deg:.3e}",
                "Xf_identity",
                xf_effect["identity"],
                "Xf_RL_deg",
                f"{xf_effect['rl_phase_deg']:.3f}",
            )
    field_scan = []
    with tables.table(str(arguments.measurement_set), readonly=True, ack=False) as main:
        field_ids = np.asarray(main.getcol("FIELD_ID"), dtype=np.int32)
        time_s = np.asarray(main.getcol("TIME"), dtype=np.float64)
        scan = np.asarray(main.getcol("SCAN_NUMBER"), dtype=np.int32)
    antenna_table = arguments.measurement_set / "ANTENNA"
    with tables.table(str(antenna_table), readonly=True, ack=False) as antenna:
        positions = np.asarray(antenna.getcol("POSITION"), dtype=np.float64)
        antenna_names = [str(name) for name in antenna.getcol("NAME")]
    selected_ants = [0, 1, 2, int(np.flatnonzero(np.array(antenna_names) == "ea26")[0])]
    for field_id in (0, 9, 10, 11):
        times = np.unique(time_s[field_ids == field_id])
        if times.size == 0:
            continue
        picks = np.unique(np.linspace(0, times.size - 1, num=min(5, times.size), dtype=int))
        phase = _read_field_phase_centre(tables, arguments.measurement_set, names[field_id])
        frame = read_field_direction_frame(arguments.measurement_set, field_id)
        for index in picks:
            measures_chi = parallactic_angle_casacore_rad(
                times[index : index + 1],
                phase,
                positions,
                time_reference=time_reference,
                direction_frame=frame,
                receptor_angle_rad=receptor,
            )
            gmst_chi = parallactic_angle_rad(times[index : index + 1], phase, positions)
            for antenna in selected_ants:
                field_scan.append(
                    {
                        "field_id": field_id,
                        "field_name": names[field_id],
                        "direction_frame": frame,
                        "time_s": float(times[index]),
                        "scan": int(scan[field_ids == field_id][0]),
                        "antenna": antenna,
                        "antenna_name": antenna_names[antenna],
                        "measures_chi_deg": float(np.rad2deg(measures_chi[0, antenna])),
                        "gmst_chi_deg": float(np.rad2deg(gmst_chi[0, antenna])),
                        "residual_gmst_minus_measures_deg": float(
                            np.rad2deg(
                                np.angle(
                                    np.exp(1j * (gmst_chi[0, antenna] - measures_chi[0, antenna]))
                                )
                            )
                        ),
                    }
                )
    gmst_residuals = np.array(
        [float(item["residual_gmst_minus_measures_deg"]) for item in field_scan]
    )
    payload = {
        "jones_recovery_blocked": True,
        "time_reference": time_reference,
        "receptor_angle_unique_rad": [float(value) for value in np.unique(receptor)],
        "oracle": reports,
        "field_scan": field_scan,
        "oracle_classification": classify_chi_residual(
            residual_deg=np.asarray(residuals, dtype=np.float64),
            hour_angle_rad=np.asarray(hour_angles, dtype=np.float64),
            antenna=np.asarray(antennas, dtype=np.int32),
        ),
        "gmst_versus_measures_classification": classify_chi_residual(
            residual_deg=gmst_residuals,
            field_id=np.asarray([item["field_id"] for item in field_scan], dtype=np.int32),
            antenna=np.asarray([item["antenna"] for item in field_scan], dtype=np.int32),
        ),
        "notes": (
            "CASA χ is recovered from +P versus +Xf operators",
            "Measures use UTC, J2000/ICRS PHASE_DIR, per-antenna ITRF, not POINTING_OFFSET",
            "FEED.RECEPTOR_ANGLE is included; it is identically zero on this MS",
        ),
    }
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    print("oracle_classification", payload["oracle_classification"])
    print("gmst_versus_measures", payload["gmst_versus_measures_classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
