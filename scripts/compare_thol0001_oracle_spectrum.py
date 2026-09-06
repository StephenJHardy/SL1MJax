"""Fit CASA-JAX residual phase as a + b*nu on the all-channel oracle row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_calibration_golden import (
    golden_visibility_block,
    import_thol0001_casa_tables,
)
from sl1mjax.holography_calibration_oracle import (
    BASIS_NAMES,
    apply_basis_with_solution,
    classify_kcross_phase_fit,
    delay_reference_offset_hz,
    fit_residual_phase_line,
    operator_from_basis_outputs,
    solution_for_stage,
)
from sl1mjax.holography_ms import _tables


def _casa_outputs(root: Path, stage: str, cases: list[dict], nchan: int) -> np.ndarray:
    tables = _tables()
    stacked = np.zeros((4, nchan, 4), dtype=np.complex128)
    by_basis = {case["basis"]: case for case in cases}
    for index, name in enumerate(BASIS_NAMES):
        path = root / "applied" / stage.replace("+", "_") / Path(by_basis[name]["ms"]).name
        with tables.table(str(path), readonly=True, ack=False) as main:
            stacked[index] = np.asarray(main.getcol("CORRECTED_DATA"))[0]
    return stacked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum-root", type=Path, required=True)
    parser.add_argument(
        "--product-root",
        type=Path,
        default=Path("/media/stephen/astro/vla/extracted/commissioning/products"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-desc-id", type=int, default=4)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.spectrum_root / "oracle_cases.json").read_text())
    sample = Path(manifest["cases"][0]["ms"])
    block, _, _ = golden_visibility_block(
        sample, data_column="DATA", data_desc_id=arguments.data_desc_id
    )
    root = arguments.product_root
    full = import_thol0001_casa_tables(
        {
            "antpos": root / "diagonal" / "antpos.cal",
            "G1": root / "diagonal" / "G1.cal",
            "K0": root / "diagonal" / "K0.cal",
            "B0": root / "diagonal" / "B0.cal",
            "Kcross": root / "fullpol" / "Kcross.cal",
            "Df": root / "fullpol" / "Df.cal",
            "Xf": root / "fullpol" / "Xf.cal",
        },
        measurement_set=sample,
        spectral_window_id=block.spectral_window_id,
        product="fullpol",
    )
    nchan = block.frequency_hz.size
    payload = {"jax_reference_frequency_hz": full.reference_frequency_hz, "stages": {}}
    for stage in ("K+B+G", "K+B+G+Kcross"):
        casa_stack = _casa_outputs(arguments.spectrum_root, stage, manifest["cases"], nchan)
        solution = solution_for_stage(full, stage)
        jax_phase = []
        casa_phase = []
        residual = []
        for channel in range(nchan):
            casa_op = operator_from_basis_outputs(casa_stack[:, channel, :])
            jax_out = apply_basis_with_solution(block, solution, channel=channel)
            jax_op = operator_from_basis_outputs(jax_out)
            casa_phase.append(float(np.angle(casa_op[1, 1])))
            jax_phase.append(float(np.angle(jax_op[1, 1])))
            residual.append(float(np.angle(jax_op[1, 1] * np.conjugate(casa_op[1, 1]))))
            if stage == "K+B+G":
                residual[-1] = float(np.angle(jax_op[0, 0] * np.conjugate(casa_op[0, 0])))
        fit = fit_residual_phase_line(block.frequency_hz, np.asarray(residual))
        delay = (
            None
            if full.cross_hand_delay_s is None
            else float(full.cross_hand_delay_s[block.antenna1[0], 0])
        )
        payload["stages"][stage] = {
            "fit": fit,
            "classification": classify_kcross_phase_fit(fit),
            "implied_ref_offset_hz": delay_reference_offset_hz(
                float(fit["intercept_rad"]), delay or 0.0
            ),
            "residual_phase_deg": np.rad2deg(np.asarray(residual)).tolist(),
        }
        print(stage, payload["stages"][stage]["classification"], fit)
    if arguments.output is not None:
        write_json(payload, arguments.output)
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
