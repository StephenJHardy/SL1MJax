#!/usr/bin/env python3
"""One-batch VJP vs explicit-JAX gate on a fitted 3C391 product.

Run this on Bacchus after C1/C4 finish, from the staged tree, without
replacing the VJP commissioning jobs. A product is required so a passing
report cannot be mistaken for the depth-zero starting plan. The compare
uses the explicit production batch geometry. A passing report is required
before any reconstruction uses ``operator_mode=explicit_jax``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_3c391_phase6_bacchus import (  # noqa: E402
    DEFAULT_CATALOGUE,
    DEFAULT_NATIVE,
    DEFAULT_OUTPUT,
    DEFAULT_POL_GOLDEN,
    _config_for_beam,
    load_antenna_positions,
    load_pointing_blocks,
    protocol_config,
)

from sl1mjax.phase6_protocol import (  # noqa: E402
    compare_operator_modes,
    phase6_folds,
    sha256_file,
    sky_and_plan_from_product,
)
from sl1mjax.voltage_reconstruction import (  # noqa: E402
    PRODUCTION_STOKES_I_BEAMS,
    _sample_training_batch,
    stokes_i_beam,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "operator_compare.json")
    parser.add_argument("--pointing", default="C1")
    parser.add_argument("--beams", default=",".join(PRODUCTION_STOKES_I_BEAMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--product",
        type=Path,
        required=True,
        help="Required C1 product directory with fitted flux and planner depths.",
    )
    return parser


def _product_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for name in ("checkpoint.json", "integration_plan.json", "summary.json"):
        path = directory / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def _planner_depths(directory: Path) -> dict[str, int]:
    plan_path = directory / "integration_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["depths"].items()}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    beams = tuple(item.strip() for item in arguments.beams.split(",") if item.strip())
    if any(item not in PRODUCTION_STOKES_I_BEAMS for item in beams):
        parser.error(f"beams must be a subset of {PRODUCTION_STOKES_I_BEAMS}")
    product = arguments.product.resolve()
    if not product.is_dir():
        parser.error(f"product directory does not exist: {product}")

    blocks = load_pointing_blocks(arguments.native_root, (arguments.pointing,))
    antenna = load_antenna_positions(arguments.polarization_golden, blocks[0].antenna_count)
    folds = phase6_folds(blocks)
    _table, flux, plan = sky_and_plan_from_product(product)
    planner_depths = _planner_depths(product)
    source_hashes = {
        **_product_hashes(product),
        "catalogue": sha256_file(arguments.catalogue),
        "polarization_golden": sha256_file(arguments.polarization_golden),
    }
    base = protocol_config(
        steps=1,
        max_rounds=0,
        max_splits_per_round=8,
        max_split_fraction=0.05,
        patience=1,
        sparsity_weight=0.0,
        strict_audit=False,
        operator_mode="explicit_jax",
    )
    reports = []
    passed = True
    for beam_mode in beams:
        config = _config_for_beam(base, beam_mode)
        batch = _sample_training_batch(
            blocks, folds.train, config, np.random.default_rng(arguments.seed)
        )
        print(
            f"=== compare {beam_mode} {arguments.pointing} "
            f"product={product} "
            f"parents={plan.parent_count} nodes={plan.node_count} "
            f"rows={batch.block.uvw_m.shape[0]} "
            f"batch_size_rows={config.inference.batch_size_rows} "
            f"pixel_chunk={config.operator.pixel_chunk_size} ===",
            flush=True,
        )
        report = compare_operator_modes(
            flux,
            batch.block,
            plan,
            stokes_i_beam(beam_mode),
            antenna_position_m=antenna,
            calibration_state="casa_parang_true",
            config=config.operator,
            train_mask=batch.mask,
            rtol=arguments.rtol,
            atol=arguments.atol,
            batch_size_rows=config.inference.batch_size_rows,
            product=str(product),
            source_hashes=source_hashes,
            sampled_rows=batch.source_rows,
            planner_depths=planner_depths,
        )
        report = {
            "beam_mode": beam_mode,
            "pointing": arguments.pointing,
            "operator_mode": config.operator_mode,
            **report,
        }
        reports.append(report)
        passed = passed and bool(report["passed"])
        print(json.dumps(report), flush=True)
    payload = {
        "passed": passed,
        "product": str(product),
        "source_hashes": source_hashes,
        "reports": reports,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output} passed={passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
