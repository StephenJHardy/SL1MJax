"""Validate independent VLA priors against CASA on raw VLA data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sl1mjax.calibration_terms import (
    CalibrationChain,
    CalibrationCoordinates,
    compare_jones,
    import_casa_prior_table,
    prior_baseline_jones,
)
from sl1mjax.data import read_dataset
from sl1mjax.data.ms import extract_measurement_set
from sl1mjax.polarization import Correlation
from sl1mjax.vla_priors import (
    estimate_vla_zenith_opacity,
    generate_vla_gain_curve,
    generate_vla_requantizer,
    generate_vla_requantizer_from_ms,
)

TERM_GATES = {"gain_curve": 1e-3, "opacity": 5e-3, "requantizer": 1e-3}
COMBINED_GATE = 2e-3


def _coordinates(
    block: Any, antenna_position_m: np.ndarray, receptor_count: int
) -> CalibrationCoordinates:
    return CalibrationCoordinates(
        block.time_s,
        block.frequency_hz,
        block.spectral_window_id,
        block.phase_centre_rad,
        antenna_position_m,
        receptor_count,
    )


def validate(
    dataset_path: Path,
    gain_curve_table: Path,
    opacity_table: Path,
    requantizer_table: Path,
    *,
    rq_row_stride: int = 1,
    data_column: str = "DATA",
    fields: tuple[int, ...] | None = None,
    row_stride: int = 1,
) -> dict[str, Any]:
    """Run coordinate-aligned term and corrected-visibility comparisons."""

    measurement_set = (dataset_path / "table.dat").is_file()
    dataset = (
        extract_measurement_set(
            dataset_path,
            data_column=data_column,
            fields=fields,
            row_stride=row_stride,
            include_switched_power_metadata=False,
        )
        if measurement_set
        else read_dataset(dataset_path)
    )
    if dataset.metadata is None:
        raise ValueError("validation dataset has no portable observation metadata")
    metadata = dataset.metadata
    positions = np.asarray(
        [antenna.position_m for antenna in metadata.antennas], dtype=np.float64
    )
    observation_time = float(min(np.min(block.time_s) for block in dataset.blocks))
    generated_terms = (
        generate_vla_gain_curve(metadata, observation_time_s=observation_time),
        estimate_vla_zenith_opacity(metadata, observation_time_s=observation_time),
        (
            generate_vla_requantizer_from_ms(dataset_path)
            if measurement_set
            else generate_vla_requantizer(metadata)
        ),
    )
    reference_terms = (
        import_casa_prior_table(gain_curve_table),
        import_casa_prior_table(opacity_table),
        import_casa_prior_table(requantizer_table, row_stride=rq_row_stride),
    )
    generated_chain = CalibrationChain(
        generated_terms, positions, {"role": "independent"}
    )
    reference_chain = CalibrationChain(
        reference_terms, positions, {"role": "CASA oracle"}
    )
    block_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for block_index, block in enumerate(dataset.blocks):
        coordinates = _coordinates(
            block, positions, generated_terms[0].coefficients.shape[2]
        )
        term_reports: dict[str, Any] = {}
        for generated, reference in zip(
            generated_terms, reference_terms, strict=True
        ):
            generated_value, generated_valid = generated.evaluate(coordinates)
            reference_value, reference_valid = reference.evaluate(coordinates)
            comparison = compare_jones(
                generated_value,
                reference_value,
                generated_valid,
                reference_valid,
            )
            term_reports[generated.kind] = comparison.__dict__
            if comparison.relative_rms >= TERM_GATES[generated.kind]:
                failures.append(
                    f"block {block_index} {generated.kind} "
                    f"{comparison.relative_rms:.6g} >= {TERM_GATES[generated.kind]:.6g}"
                )
        generated_baseline, generated_valid = prior_baseline_jones(
            generated_chain,
            coordinates,
            block.antenna1,
            block.antenna2,
        )
        reference_baseline, reference_valid = prior_baseline_jones(
            reference_chain,
            coordinates,
            block.antenna1,
            block.antenna2,
        )
        parallel = [
            index
            for index, correlation in enumerate(block.correlations)
            if correlation
            in {
                Correlation.XX,
                Correlation.YY,
                Correlation.RR,
                Correlation.LL,
            }
        ]
        if len(parallel) != generated_baseline.shape[-1]:
            raise ValueError("validation requires both parallel-hand correlations")
        visibility = block.visibility[..., parallel]
        active = block.active[..., parallel] & generated_valid & reference_valid
        generated_corrected = np.divide(
            visibility,
            generated_baseline,
            out=np.zeros_like(visibility),
            where=active,
        )
        reference_corrected = np.divide(
            visibility,
            reference_baseline,
            out=np.zeros_like(visibility),
            where=active,
        )
        difference = generated_corrected[active] - reference_corrected[active]
        denominator = np.sum(np.abs(reference_corrected[active]) ** 2)
        normalized_rms = float(
            np.sqrt(np.sum(np.abs(difference) ** 2) / denominator)
        )
        flag_mismatches = int(np.count_nonzero(generated_valid != reference_valid))
        if normalized_rms >= COMBINED_GATE:
            failures.append(
                f"block {block_index} combined {normalized_rms:.6g} >= {COMBINED_GATE:.6g}"
            )
        if flag_mismatches:
            failures.append(
                f"block {block_index} has {flag_mismatches} prior flag mismatches"
            )
        block_reports.append(
            {
                "block_index": block_index,
                "spectral_window_id": block.spectral_window_id,
                "row_count": block.shape[0],
                "terms": term_reports,
                "combined_normalized_complex_rms": normalized_rms,
                "combined_flag_mismatch_count": flag_mismatches,
            }
        )
    return {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "gates": {**TERM_GATES, "combined": COMBINED_GATE},
        "passed": not failures,
        "failures": failures,
        "blocks": block_reports,
        "opacity_provenance": generated_terms[1].provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--gain-curve-table", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--requantizer-table", type=Path, required=True)
    parser.add_argument("--rq-row-stride", type=int, default=1)
    parser.add_argument("--data-column", default="DATA")
    parser.add_argument("--fields", help="comma-separated field IDs")
    parser.add_argument("--row-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = validate(
        arguments.dataset,
        arguments.gain_curve_table,
        arguments.opacity_table,
        arguments.requantizer_table,
        rq_row_stride=arguments.rq_row_stride,
        data_column=arguments.data_column,
        fields=(
            None
            if arguments.fields is None
            else tuple(int(value) for value in arguments.fields.split(","))
        ),
        row_stride=arguments.row_stride,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
