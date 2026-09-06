"""Compare CASA CORRECTED_DATA to JAX apply from DATA on the golden MS."""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import compare_golden_measurement_set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument("--product", choices=("diagonal", "fullpol"), required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.product_root
    report = compare_golden_measurement_set(
        arguments.measurement_set,
        {
            "antpos": root / "diagonal" / "antpos.cal",
            "G1": root / "diagonal" / "G1.cal",
            "K0": root / "diagonal" / "K0.cal",
            "B0": root / "diagonal" / "B0.cal",
            "Kcross": root / "fullpol" / "Kcross.cal",
            "Df": root / "fullpol" / "Df.cal",
            "Xf": root / "fullpol" / "Xf.cal",
        },
        product=arguments.product,  # type: ignore[arg-type]
    )
    print(
        report["product"],
        "passed" if report["comparison"]["passed"] else "FAILED",
        report["comparison"],
    )
    if arguments.output is not None:
        write_json(report, arguments.output)
        print(arguments.output)
    return 0 if report["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
