#!/usr/bin/env python3
"""Inspect a finished full-pol diagnostic by hand, pointing, channel, and PA.

Does not freeze full Jones or interpret the result as evidence about the sky.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sl1mjax.fullpol_prep import (
    abort_fullpol_diagnostic_failures,
    inspect_fold3_diagnostic,
    stamp_diagnostic_interpretation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs/3c391_fullpol_prep/jones_compare/summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args(argv)


def _print_hand(name: str, metric: dict | None) -> None:
    if not metric:
        print(f"  {name}: missing")
        return
    print(
        f"  {name}: mse={metric.get('mse')} "
        f"power={metric.get('residual_power')} "
        f"samples={metric.get('samples')}"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    stamped = stamp_diagnostic_interpretation(summary)
    arguments.summary.write_text(
        json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = abort_fullpol_diagnostic_failures(summary=stamped)
    inspection = inspect_fold3_diagnostic(stamped)
    output = arguments.summary.with_name("inspection.json")
    output.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("kind", inspection["interpretation"]["kind"])
    print("question", inspection["interpretation"]["question"])
    print("do_not_freeze_full_jones", True)
    if failures:
        print("abort_failures")
        for item in failures:
            print(f"  {item}")
        return 2
    for beam_name, beam in inspection["beams"].items():
        print(beam_name, f"q={beam.get('q')} u={beam.get('u')}")
        for hand in ("total", "RR", "LL", "RL", "LR"):
            _print_hand(hand, beam.get(hand))
        for row in beam.get("by_pointing") or []:
            print(
                "  pointing",
                row.get("pointing_id"),
                "RL",
                (row.get("RL") or {}).get("mse"),
                "LR",
                (row.get("LR") or {}).get("mse"),
            )
        for row in beam.get("by_pa") or []:
            print(
                "  PA",
                row.get("pa_min_deg"),
                row.get("pa_max_deg"),
                "mse",
                row.get("mse"),
            )
        for row in beam.get("by_channel") or []:
            print("  channel", row.get("channel"), "mse", row.get("mse"))
    delta = inspection.get("delta_full_jones_minus_diagonal") or {}
    print("paired full_jones_minus_diagonal")
    for hand in ("total", "RR", "LL", "RL", "LR", "RR_LL", "RL_LR"):
        _print_hand(hand, delta.get(hand))
    print("leakage_evidence", inspection.get("leakage_evidence"))
    print("wrote", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
