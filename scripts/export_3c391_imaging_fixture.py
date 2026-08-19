"""Export portable averaged 3C391 blocks for GPU imaging without casacore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_3c391_target import _extract_target, _solve_calibration

from sl1mjax.calibration_inference import CalibrationSolveConfig
from sl1mjax.data.canonical import VisibilityDataset, write_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurement_set", type=Path)
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/fixtures/3c391_calibration_golden.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_imaging_fixture.zarr"),
    )
    parser.add_argument("--field-id", type=int, default=2)
    parser.add_argument("--frequency-bins", type=int, default=4)
    parser.add_argument("--time-bin-s", type=float, default=60.0)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--calibration-iterations", type=int, default=300)
    arguments = parser.parse_args()

    solve_config = CalibrationSolveConfig(
        iterations=arguments.calibration_iterations,
        learning_rate=0.03,
        seed=11,
    )
    solution, calibration_metrics = _solve_calibration(
        arguments.golden, solve_config
    )
    casa, jax_calibrated = _extract_target(
        arguments.measurement_set,
        solution,
        field_id=arguments.field_id,
        frequency_bins=arguments.frequency_bins,
        time_bin_s=arguments.time_bin_s,
        chunk_rows=arguments.chunk_rows,
    )
    provenance = {
        "fixture": "3C391 C1 averaged imaging comparison",
        "measurement_set": arguments.measurement_set.name,
        "case_order": ["casa_corrected", "jax_calibrated"],
        "field_id": arguments.field_id,
        "frequency_bins": arguments.frequency_bins,
        "time_bin_s": arguments.time_bin_s,
        "calibration": calibration_metrics,
    }
    write_dataset(
        VisibilityDataset((casa, jax_calibrated), provenance=provenance),
        arguments.output,
    )
    summary = {
        "output": str(arguments.output),
        "block_shapes": [list(casa.shape), list(jax_calibrated.shape)],
        "active_samples": [
            int(casa.active.sum()),
            int(jax_calibrated.active.sum()),
        ],
        "provenance": provenance,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
