"""Real-product 3C286 gate before the expensive full-Jones target run.

Compares JAX-prepared G/K/B-only data to the CASA full-pol product on their
common mask. Recovers Q/I, U/I, the casaguide 66° argument, and V/I ~ 0.
Reports RR, RL, LR and LL separately. Apply-back is the imported CASA
tables; the connected holdout is the last scan or time cohort, not a re-solve.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration import import_casa_polarization_solution
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.fullpol_prep import (
    SCALAR_WEIGHT_LIMITATION,
    THREE_C286_FIELD_ID,
    apply_polarization_before_averaging,
    casa_comparison_mask,
    compare_dterm_crosshand_flags,
    correlation_sample_report,
    evaluate_three_c286_gate,
    load_calibration_state,
    three_c286_expectation,
)
from sl1mjax.polarization_diagnostics import (
    calibrator_polarization_floor,
    deterministic_calibrator_cohort_split,
    fit_global_fractional_polarization,
    global_fractional_polarization_as_dict,
    polarization_floor_as_dict,
)
from sl1mjax.voltage_flux_refit import score_visibility_prediction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POL = ROOT / "tests" / "fixtures" / "3c391_polarization_golden.npz"
DEFAULT_KBG = ROOT / "tests" / "fixtures" / "3c391_calibration_golden.npz"
DEFAULT_GKB = ROOT / "data/3c391/fullpol_prep/3c391_gkb_only_4corr.ms"
DEFAULT_CASA = ROOT / "data/3c391/fullpol_prep/3c391_casa_fullpol_4corr.ms"
DEFAULT_OUTPUT = ROOT / "outputs/3c391_fullpol_prep/calibrator_gate"


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _floor_payload(block, *, independence: str, label: str, sample_mask=None) -> dict[str, Any]:
    floor = polarization_floor_as_dict(
        calibrator_polarization_floor(
            block,
            independence=independence,
            label=label,
            sample_mask=sample_mask,
        )
    )
    try:
        fitted = global_fractional_polarization_as_dict(
            fit_global_fractional_polarization(block, sample_mask=sample_mask)
        )
    except ValueError as error:
        fitted = {"skipped": str(error)}
    return {
        "floor": floor,
        "global_quv": fitted,
        "gate": evaluate_three_c286_gate(floor),
        "correlations": correlation_sample_report(block, sample_mask),
    }


def _extract_field(path: Path, field_id: int):
    dataset = extract_measurement_set(
        path,
        data_column="CORRECTED_DATA",
        model_column="MODEL_DATA",
        fields=(field_id,),
    )
    if len(dataset.blocks) != 1:
        raise ValueError(f"{path} field {field_id} produced {len(dataset.blocks)} blocks")
    return dataset.blocks[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polarization-golden", type=Path, default=DEFAULT_POL)
    parser.add_argument("--calibration-golden", type=Path, default=DEFAULT_KBG)
    parser.add_argument("--gkb-measurement-set", type=Path, default=DEFAULT_GKB)
    parser.add_argument("--casa-measurement-set", type=Path, default=DEFAULT_CASA)
    parser.add_argument("--field", type=int, default=THREE_C286_FIELD_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if not arguments.gkb_measurement_set.is_dir():
        raise FileNotFoundError(arguments.gkb_measurement_set)
    if not arguments.casa_measurement_set.is_dir():
        raise FileNotFoundError(arguments.casa_measurement_set)
    gkb_state = load_calibration_state(arguments.gkb_measurement_set) or {}
    casa_state = load_calibration_state(arguments.casa_measurement_set) or {}
    if gkb_state.get("calibration_state") != "gkb_only":
        raise ValueError(
            "GKB product sidecar is not gkb_only; refusing a second polarisation apply"
        )
    if casa_state.get("do_not_apply_jax_polarisation") is not True:
        raise ValueError("CASA full-pol product is missing do_not_apply_jax_polarisation")

    print("extracting 3C286 from both products", flush=True)
    gkb = _extract_field(arguments.gkb_measurement_set, arguments.field)
    casa = _extract_field(arguments.casa_measurement_set, arguments.field)
    imported = import_casa_polarization_solution(
        arguments.polarization_golden,
        arguments.calibration_golden,
        label="flux_angle",
    )
    print("applying imported Kcross/D/X/P once at native resolution", flush=True)
    jax = apply_polarization_before_averaging(
        gkb,
        imported,
        frequency_bins=None,
        time_bin_seconds=None,
    )
    dterm = compare_dterm_crosshand_flags(gkb, jax, casa, imported)
    common = casa_comparison_mask(jax, casa)
    jax_common = replace(jax, flag=jax.flag | ~common)
    casa_common = replace(casa, flag=casa.flag | ~common)
    split = deterministic_calibrator_cohort_split(jax_common)
    residual = score_visibility_prediction(jax_common, casa_common.visibility, mask=common)

    report: dict[str, Any] = {
        "gate": "3c286_real_product",
        "expected": three_c286_expectation(),
        "weight_limitation": SCALAR_WEIGHT_LIMITATION,
        "leakage_application": "casa_parallel_preserving",
        "polarisation_applies": {
            "jax": 1,
            "casa": 1,
            "exactly_one_each": True,
        },
        "products": {
            "gkb": {
                "path": str(arguments.gkb_measurement_set.resolve()),
                "calibration_state": gkb_state,
            },
            "casa": {
                "path": str(arguments.casa_measurement_set.resolve()),
                "calibration_state": casa_state,
            },
        },
        "dterm_crosshand_agreement": dterm,
        "common_mask_samples": int(np.count_nonzero(common)),
        "apply_back": {
            "independence": "apply_back",
            "jax": _floor_payload(jax_common, independence="apply_back", label="jax_3c286"),
            "casa": _floor_payload(casa_common, independence="apply_back", label="casa_3c286"),
            "jax_versus_casa": residual,
        },
        "connected_holdout": {
            "independence": "held_out_calibrator",
            "strategy": split.strategy,
            "train_samples": int(np.count_nonzero(split.train)),
            "holdout_samples": int(np.count_nonzero(split.holdout)),
            "jax": _floor_payload(
                jax_common,
                independence="held_out_calibrator",
                label="jax_3c286_holdout",
                sample_mask=split.holdout,
            ),
            "casa": _floor_payload(
                casa_common,
                independence="held_out_calibrator",
                label="casa_3c286_holdout",
                sample_mask=split.holdout,
            ),
        },
        "masks": {
            "beam_comparison": "jax_prepared_active_only",
            "casa_comparison": "intersection_of_jax_and_casa_actives",
        },
    }
    failures: list[str] = []
    if not dterm["agreed"]:
        failures.append(
            f"JAX/CASA invalid-D RL/LR recall {dterm['recall']:.3f} "
            f"< {dterm['min_recall']}"
        )
    for family, side in (
        ("apply_back", "jax"),
        ("apply_back", "casa"),
        ("connected_holdout", "jax"),
        ("connected_holdout", "casa"),
    ):
        gate = report[family][side]["gate"]
        if not gate["passed"]:
            failures.extend(f"{family}/{side}: {item}" for item in gate["failures"])
    report["passed"] = not failures
    report["failures"] = failures
    arguments.output.mkdir(parents=True, exist_ok=True)
    output = arguments.output / "report.json"
    output.write_text(json.dumps(_to_json(report), indent=2, sort_keys=True) + "\n")
    print(output, flush=True)
    if failures:
        print("3C286 calibrator gate failed:", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
