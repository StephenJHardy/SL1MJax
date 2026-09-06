"""Compare CASA and JAX after each cumulative THOL0001 calibration stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    apply_imported_solution,
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_oracle import (
    BISECTION_STAGES,
    compare_stage_visibilities,
    solution_for_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bisection-root", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-desc-id", type=int, default=4)
    arguments = parser.parse_args()
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
    stages = {}
    passed = True
    for stage in BISECTION_STAGES:
        vis = arguments.bisection_root / f"stage_{stage.replace('+', '_')}.ms"
        if not vis.is_dir():
            stages[stage] = {"missing": True}
            passed = False
            continue
        block, casa, _flag = golden_visibility_block(
            vis, data_column="DATA", data_desc_id=arguments.data_desc_id
        )
        if casa is None:
            stages[stage] = {"missing_corrected_data": True}
            passed = False
            continue
        full = import_thol0001_casa_tables(
            tables,
            measurement_set=vis,
            spectral_window_id=block.spectral_window_id,
            product="fullpol",
        )
        solution = solution_for_stage(full, stage)
        jax_block = apply_imported_solution(block, solution)
        report = compare_stage_visibilities(
            casa,
            jax_block.visibility,
            ~block.flag & ~jax_block.flag,
            stage=stage,
            antenna1=block.antenna1,
            antenna2=block.antenna2,
            time_s=block.time_s,
            frequency_hz=block.frequency_hz,
        )
        stages[stage] = report
        passed = passed and bool(report["passed"])
        print(
            stage,
            report["classification"],
            {name: item.get("relative_l2") for name, item in report["per_correlation"].items()},
        )
    payload = {
        "passed": passed,
        "jones_recovery_blocked": True,
        "stages": stages,
    }
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
