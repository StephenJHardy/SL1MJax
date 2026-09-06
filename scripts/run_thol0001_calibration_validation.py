"""Run the next THOL0001 validation artifacts on Bacchus.

Copies a small HOLORASTER golden MS, writes term coverage and field-9
parallactic reports. Does not recover Jones. Uses the isolated package
tree; do not overwrite the Bacchus checkout ``__init__.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sl1mjax.holography_calibration import (
    field_parallactic_report,
    jones_recovery_gate,
    term_coverage_from_casa_tables,
    three_c286_surviving_graph,
    write_json,
)
from sl1mjax.holography_calibration_golden import (
    copy_golden_measurement_set,
    golden_casa_script_payload,
    select_golden_holoraster_rows,
    write_golden_casa_request,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-set",
        type=Path,
        default=Path(
            "/media/stephen/astro/vla/extracted/commissioning/"
            "THOL0001.sb31628704.eb31629959.lowerC.spw45.ms"
        ),
    )
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/validation"),
    )
    parser.add_argument("--skip-golden-copy", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    root = arguments.product_root
    write_json(jones_recovery_gate(), output / "jones_recovery_gate.json")
    graph = three_c286_surviving_graph(arguments.measurement_set)
    write_json(graph, output / "three_c286_graph.json")
    coverage = term_coverage_from_casa_tables(
        {
            "K": root / "diagonal" / "K0.cal",
            "B": root / "diagonal" / "B0.cal",
            "G": root / "diagonal" / "G1.cal",
            "Kcross": root / "fullpol" / "Kcross.cal",
            "Df": root / "fullpol" / "Df.cal",
            "Xf": root / "fullpol" / "Xf.cal",
        },
        measurement_set=arguments.measurement_set,
        three_c286_active=tuple(graph.get("active_antennas", ())),
        three_c286_absent=tuple(graph.get("absent_antennas", ())),
    )
    write_json(coverage, output / "term_coverage.json")
    write_json(
        field_parallactic_report(arguments.measurement_set, 9),
        output / "field9_parallactic.json",
    )
    write_json(
        field_parallactic_report(arguments.measurement_set, 11),
        output / "field11_parallactic.json",
    )
    if not arguments.skip_golden_copy:
        for product in ("diagonal", "fullpol"):
            dest = output / f"THOL0001.lowerC.golden.{product}.ms"
            if dest.exists():
                continue
            rows = select_golden_holoraster_rows(arguments.measurement_set)
            copy_golden_measurement_set(arguments.measurement_set, dest, rows)
            write_golden_casa_request(
                golden_casa_script_payload(
                    vis=dest,
                    product=product,  # type: ignore[arg-type]
                    tables={
                        "antpos": root / "diagonal" / "antpos.cal",
                        "G1": root / "diagonal" / "G1.cal",
                        "K0": root / "diagonal" / "K0.cal",
                        "B0": root / "diagonal" / "B0.cal",
                        "Kcross": root / "fullpol" / "Kcross.cal",
                        "Df": root / "fullpol" / "Df.cal",
                        "Xf": root / "fullpol" / "Xf.cal",
                    },
                ),
                dest.with_name(dest.name + ".request.json"),
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
