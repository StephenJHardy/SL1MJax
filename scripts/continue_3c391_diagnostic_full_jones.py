#!/usr/bin/env python3
"""Continue diagnostic full-Jones imaging from a finished checkpoint.

Keeps the fitted topology and fluxes fixed as the starting image, then runs
more explicit-JAX SGD steps with regular train/holdout scores. Writes a new
product. Does not overwrite the 200-step ancestor and does not freeze Jones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sl1mjax.phase6_protocol import phase6_folds, sky_and_plan_from_product

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/3c391_fullpol_prep/jones_imaging/baseline/full_jones"
DEFAULT_OUTPUT = ROOT / "outputs/3c391_fullpol_prep/jones_imaging_continue"
DEFAULT_NATIVE = Path("outputs/3c391_native_averaging_ablation")
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    scripts_directory = str(Path(__file__).resolve().parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    from run_3c391_phase6_bacchus import (
        POINTINGS,
        _source_manifest,
        load_antenna_positions,
        load_pointing_blocks,
        protocol_config,
        run_one,
    )

    table, _flux, _plan = sky_and_plan_from_product(arguments.source)
    blocks = load_pointing_blocks(arguments.native_root, POINTINGS)
    folds = phase6_folds(blocks)
    antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
    config = protocol_config(
        steps=arguments.steps,
        max_rounds=0,
        max_splits_per_round=32,
        max_split_fraction=0.05,
        patience=arguments.patience,
        sparsity_weight=3e-4,
        strict_audit=False,
        operator_mode="explicit_jax",
        learning_rate=arguments.learning_rate,
        validation_interval=arguments.validation_interval,
    )
    source = _source_manifest()
    source.update(
        {
            "continued_from": str(arguments.source.resolve()),
            "diagnostic_full_jones": True,
            "do_not_freeze_full_jones": True,
            "fixed_topology": True,
        }
    )
    print(
        f"continue {arguments.source} -> {arguments.output}/continue/full_jones "
        f"steps={arguments.steps} val_every={arguments.validation_interval}",
        flush=True,
    )
    run_one(
        stage="continue",
        beam_mode="full_jones",
        table=table,
        blocks=blocks,
        folds=folds,
        pointing_ids=POINTINGS,
        antenna_position_m=antenna,
        config=config,
        output=arguments.output,
        source=source,
        allow_diagnostic_full_jones=True,
    )
    summary = arguments.output / "continue" / "full_jones" / "summary.json"
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        curve = payload.get("optimization_curve") or {}
        print(json.dumps(curve, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
