#!/usr/bin/env python3
"""Freeze the seven-point diagonal Stokes-I product as a polarisation-test ancestor.

Copies topology, fluxes, catalogue atoms, integration plan, folds, flags,
weights, calibration metadata, beam pin, conventions, and hashes. The copy is
the Jones-experiment sky, not the accepted C-band Stokes-I reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sl1mjax.fullpol_prep import freeze_polarisation_stokes_i_ancestor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/3c391_phase6_explicit/baseline/diagonal_copolar"
DEFAULT_DEST = ROOT / "outputs/3c391_fullpol_prep/frozen_diagonal_ancestor"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    payload = freeze_polarisation_stokes_i_ancestor(arguments.source, arguments.dest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
