"""Score G and Df interpolation methods against CASA 4x4 operators."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_oracle import (
    apply_basis_with_solution,
    compare_correction_operators,
    operator_from_basis_outputs,
    score_interpolation_methods,
    solution_for_stage,
)
from sl1mjax.holography_ms import _tables


def _casa_operator(root: Path, stage: str, case: str, channel: int) -> np.ndarray:
    tables = _tables()
    outputs = np.zeros((4, 4), dtype=np.complex128)
    for index, name in enumerate(("RR", "RL", "LR", "LL")):
        path = root / "applied" / stage / f"{case}_{name}.ms"
        with tables.table(str(path), readonly=True, ack=False) as main:
            outputs[index] = np.asarray(main.getcol("CORRECTED_DATA"))[0, int(channel)]
    return operator_from_basis_outputs(outputs)


def _with_interpolation(solution, method: str):
    return replace(solution, interpolation=method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", default="moving_held_t0")
    arguments = parser.parse_args()
    root = arguments.product_root
    sample = arguments.oracle_root / f"{arguments.case}_RR.ms"
    block, _, _ = golden_visibility_block(sample, data_column="DATA", data_desc_id=4)
    full = import_thol0001_casa_tables(
        {
            "antpos": root / "diagonal" / "antpos.cal",
            "G1": root / "diagonal" / "G1.cal",
            "K0": root / "diagonal" / "K0.cal",
            "B0": root / "diagonal" / "B0.cal",
            "Kcross": root / "fullpol" / "Kcross.cal",
            "Df": root / "fullpol" / "Df.cal",
            "Xf": root / "fullpol" / "Xf.cal",
        },
        measurement_set=sample,
        spectral_window_id=block.spectral_window_id,
        product="fullpol",
    )
    methods = ("nearest", "linear", "casa_linear", "linear_complex", "linear_amp_phase")
    reports = []
    for stage, label in (("K+B+G", "G"), ("K+B+G+Kcross+Df", "Df")):
        base = solution_for_stage(full, stage)
        for channel in (8, 56):
            casa = _casa_operator(
                arguments.oracle_root, stage.replace("+", "_"), arguments.case, channel
            )
            ranked = []
            for method in methods:
                solution = _with_interpolation(base, method)
                jax_op = operator_from_basis_outputs(
                    apply_basis_with_solution(block, solution, channel=channel)
                )
                comparison = compare_correction_operators(casa, jax_op)
                ranked.append(
                    {"method": method, **{k: comparison[k] for k in ("relative_l2", "max_abs")}}
                )
                print(
                    arguments.case, label, "ch", channel, method, f"{comparison['relative_l2']:.3e}"
                )
            ranked.sort(key=lambda item: float(item["relative_l2"]))
            reports.append(
                {
                    "case": arguments.case,
                    "stage": stage,
                    "channel": channel,
                    "query_time_s": float(block.time_s[0]),
                    "ranked": ranked,
                }
            )
    if full.leakage is not None and full.leakage_time_s is not None:
        query = block.time_s
        for channel in (8, 56):
            casa = _casa_operator(arguments.oracle_root, "K_B_G_Kcross_Df", arguments.case, channel)
            implied_q = -np.conjugate(casa[1, 0] / casa[0, 0])
            implied_p = -(casa[2, 0] / casa[0, 0])
            idx = int(np.argmin(np.abs(full.leakage_frequency_hz - block.frequency_hz[channel])))
            for antenna, implied, slot in (
                (int(block.antenna1[0]), implied_p, 1),
                (int(block.antenna2[0]), implied_q, 1),
            ):
                valid = full.leakage_valid[:, antenna, idx, slot]
                ranked = score_interpolation_methods(
                    query,
                    full.leakage_time_s[valid],
                    full.leakage[valid, antenna, idx, slot],
                    np.array([implied], dtype=np.complex128),
                )
                reports.append(
                    {
                        "term": "Df_cparam",
                        "channel": channel,
                        "antenna": antenna,
                        "interpolation": ranked,
                    }
                )
    payload = {"jones_recovery_blocked": True, "reports": reports}
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
