"""Compare CASA applycal 4×4 operators with JAX antenna-Jones operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_oracle import (
    BASIS_NAMES,
    apply_basis_with_solution,
    compare_correction_operators,
    operator_from_basis_outputs,
    solution_for_stage,
)
from sl1mjax.holography_ms import _tables

ORACLE_STAGES = (
    "K+B+G",
    "K+B+G+Kcross",
    "K+B+G+Kcross+Df",
    "K+B+G+Kcross+Df+Xf",
    "K+B+G+Kcross+Df+Xf+P",
)


def _casa_basis_visibility(path: Path, channel: int) -> np.ndarray:
    tables = _tables()
    if not path.is_dir() or not (path / "table.dat").exists():
        raise ValueError(f"{path} is not a complete MS")
    with tables.table(str(path), readonly=True, ack=False) as main:
        if "CORRECTED_DATA" not in main.colnames():
            raise ValueError(f"{path} has no CORRECTED_DATA")
        corrected = np.asarray(main.getcol("CORRECTED_DATA"))
    return np.asarray(corrected[0, int(channel)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-desc-id", type=int, default=4)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.oracle_root / "oracle_cases.json").read_text())
    channels = tuple(int(item) for item in manifest["channels"])
    cases = manifest["cases"]
    by_label: dict[str, dict[str, Path]] = {}
    meta: dict[str, dict[str, object]] = {}
    for case in cases:
        by_label.setdefault(case["label"], {})[case["basis"]] = Path(case["ms"])
        meta[case["label"]] = case
    root = arguments.product_root
    tables = {
        "antpos": root / "diagonal" / "antpos.cal",
        "G1": root / "diagonal" / "G1.cal",
        "K0": root / "diagonal" / "K0.cal",
        "B0": root / "diagonal" / "B0.cal",
        "Kcross": root / "fullpol" / "Kcross.cal",
        "Df": root / "fullpol" / "Df.cal",
        "Xf": root / "fullpol" / "Xf.cal",
    }
    reports = []
    passed = True
    for label, bases in sorted(by_label.items()):
        sample = bases[BASIS_NAMES[0]]
        block, _casa, _flag = golden_visibility_block(
            sample, data_column="DATA", data_desc_id=arguments.data_desc_id
        )
        full = import_thol0001_casa_tables(
            tables,
            measurement_set=sample,
            spectral_window_id=block.spectral_window_id,
            product="fullpol",
        )
        for stage in ORACLE_STAGES:
            solution = solution_for_stage(full, stage)
            for channel in channels:
                casa_outputs = np.zeros((4, 4), dtype=np.complex128)
                missing = False
                applied_root = arguments.oracle_root / "applied" / stage.replace("+", "_")
                for index, name in enumerate(BASIS_NAMES):
                    applied = applied_root / Path(bases[name]).name
                    try:
                        casa_outputs[index] = _casa_basis_visibility(applied, channel)
                    except ValueError, RuntimeError:
                        missing = True
                        break
                if missing:
                    reports.append(
                        {"label": label, "stage": stage, "channel": channel, "missing": True}
                    )
                    passed = False
                    continue
                casa_op = operator_from_basis_outputs(casa_outputs)
                jax_outputs = apply_basis_with_solution(block, solution, channel=channel)
                jax_op = operator_from_basis_outputs(jax_outputs)
                comparison = compare_correction_operators(casa_op, jax_op)
                item = {
                    "label": label,
                    "stage": stage,
                    "channel": channel,
                    "antenna1": meta[label].get("antenna1"),
                    "antenna2": meta[label].get("antenna2"),
                    "contains_reference": meta[label].get("contains_reference"),
                    "contains_global_x": meta[label].get("contains_global_x"),
                    **comparison,
                }
                reports.append(item)
                passed = passed and bool(comparison["passed"])
                print(
                    label,
                    stage,
                    "ch",
                    channel,
                    "rel_l2",
                    f"{comparison['relative_l2']:.3e}",
                    "PASS" if comparison["passed"] else "FAIL",
                )
    payload = {
        "passed": passed,
        "jones_recovery_blocked": True,
        "n": len(reports),
        "reports": reports,
    }
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
