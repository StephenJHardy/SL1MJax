#!/usr/bin/env python3
"""Cheap C4 diagonal integration-depth ablation on a frozen product.

Raises only the under-resolved parents to planner depth 4, then 5 if that
moves held-out loss by the diagonal-versus-streamed ranking threshold.
Does not change sky topology or start the seven-point baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_3c391_phase6_bacchus import (  # noqa: E402
    DEFAULT_CATALOGUE,
    DEFAULT_NATIVE,
    DEFAULT_POL_GOLDEN,
    _config_for_beam,
    load_antenna_positions,
    load_pointing_blocks,
    protocol_config,
)

from sl1mjax.finite_pixel import integration_plan_from_table  # noqa: E402
from sl1mjax.phase6_protocol import (  # noqa: E402
    RANKING_HOLD_OUT_THRESHOLD,
    choose_seven_point_integration_depth,
    flux_weighted_audit_error,
    load_under_resolved_findings,
    locate_under_resolved_parents,
    phase6_folds,
    raised_under_resolved_depths,
    sha256_file,
    sky_and_plan_from_product,
)
from sl1mjax.voltage_reconstruction import (  # noqa: E402
    _holdout_loss,
    stokes_i_beam,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL_GOLDEN)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pointing", default="C4")
    parser.add_argument("--beam", default="diagonal_copolar")
    parser.add_argument("--operator-mode", choices=("vjp", "explicit_jax"), default="explicit_jax")
    parser.add_argument("--threshold", type=float, default=RANKING_HOLD_OUT_THRESHOLD)
    return parser


def _planner_depths(product: Path) -> dict[str, int]:
    payload = json.loads((product / "integration_plan.json").read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["depths"].items()}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    product = arguments.product.resolve()
    table, flux, plan = sky_and_plan_from_product(product)
    findings = load_under_resolved_findings(product)
    depths = _planner_depths(product)
    under_ids = tuple(str(item["component_id"]) for item in findings)
    weighted = flux_weighted_audit_error(table, findings)
    location = locate_under_resolved_parents(table, findings)
    summary = json.loads((product / "summary.json").read_text(encoding="utf-8"))
    depth3_holdout = float(summary["holdout_loss"])

    blocks = load_pointing_blocks(arguments.native_root, (arguments.pointing,))
    antenna = load_antenna_positions(
        arguments.polarization_golden, blocks[0].antenna_count
    )
    folds = phase6_folds(blocks)
    config = _config_for_beam(
        protocol_config(
            steps=1,
            max_rounds=0,
            max_splits_per_round=8,
            max_split_fraction=0.05,
            patience=1,
            sparsity_weight=0.0,
            strict_audit=False,
            operator_mode=arguments.operator_mode,
        ),
        arguments.beam,
    )
    beam = stokes_i_beam(arguments.beam)
    deeper: dict[int, float] = {}
    plans: dict[int, dict[str, int]] = {}
    for extra in (4, 5):
        if extra == 5 and choose_seven_point_integration_depth(
            depth3_holdout, deeper, threshold=arguments.threshold
        ) == 3:
            break
        raised = raised_under_resolved_depths(depths, under_ids, extra)
        deeper_plan = integration_plan_from_table(table, depth_by_parent=raised)
        print(
            f"=== holdout depth={extra} under_resolved={len(under_ids)} "
            f"nodes={deeper_plan.node_count} parents={deeper_plan.parent_count} ===",
            flush=True,
        )
        holdout = _holdout_loss(
            deeper_plan,
            flux,
            blocks,
            beam,
            antenna_position_m=antenna,
            calibration_state="casa_parang_true",
            holdout_masks=folds.holdout,
            config=config,
        )
        deeper[extra] = float(holdout)
        plans[extra] = {
            "node_count": int(deeper_plan.node_count),
            "parent_count": int(deeper_plan.parent_count),
            "holdout_loss": float(holdout),
            "holdout_minus_depth3": float(holdout) - depth3_holdout,
        }
        print(json.dumps(plans[extra]), flush=True)

    selected = choose_seven_point_integration_depth(
        depth3_holdout, deeper, threshold=arguments.threshold
    )
    payload = {
        "product": str(product),
        "pointing": arguments.pointing,
        "beam_mode": arguments.beam,
        "operator_mode": arguments.operator_mode,
        "threshold": arguments.threshold,
        "depth3_holdout": depth3_holdout,
        "streamed_holdout_reference": None,
        "c4_diagonal_minus_streamed": None,
        "audit": weighted,
        "location": location,
        "deeper": plans,
        "seven_point_integration_max_depth": selected,
        "ranking_resolved": selected == 3,
        "source_hashes": {
            name: sha256_file(product / name)
            for name in (
                "checkpoint.json",
                "integration_plan.json",
                "summary.json",
                "audit_findings.json",
            )
            if (product / name).is_file()
        },
        "plan_nodes_depth3": int(plan.node_count),
        "n_under_resolved": len(under_ids),
    }
    streamed = product.parent / "streamed_scalar" / "summary.json"
    if streamed.is_file():
        streamed_holdout = float(json.loads(streamed.read_text(encoding="utf-8"))["holdout_loss"])
        payload["streamed_holdout_reference"] = streamed_holdout
        payload["c4_diagonal_minus_streamed"] = depth3_holdout - streamed_holdout
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(arguments.output),
                "seven_point_integration_max_depth": selected,
                "ranking_resolved": selected == 3,
                "deeper": plans,
                "flux_weighted_error": weighted["flux_weighted_error"],
                "location_counts": location["counts"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
