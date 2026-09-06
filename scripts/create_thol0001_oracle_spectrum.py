"""Re-inject one oracle baseline onto every channel for an a+b*nu Kcross fit."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_oracle import BASIS_NAMES, injected_basis_visibilities
from sl1mjax.holography_ms import _tables


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--label", default="ref_held_t0")
    parser.add_argument("--destination", type=Path)
    arguments = parser.parse_args()
    dest = arguments.destination or arguments.oracle_root / "spectrum"
    dest.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    cases = []
    for index, name in enumerate(BASIS_NAMES):
        source = arguments.oracle_root / f"{arguments.label}_{name}.ms"
        target = dest / f"{arguments.label}_{name}.ms"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        with tables.table(str(target), readonly=False, ack=False) as main:
            data = np.asarray(main.getcol("DATA"))
            injected = np.zeros_like(data)
            injected[:] = injected_basis_visibilities()[index]
            main.putcol("DATA", injected)
            if "FLAG" in main.colnames():
                main.putcol("FLAG", np.zeros(data.shape, dtype=bool))
        cases.append({"label": arguments.label, "basis": name, "ms": str(target)})
    write_json(
        {"label": arguments.label, "channels": "all", "cases": cases}, dest / "oracle_cases.json"
    )
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
