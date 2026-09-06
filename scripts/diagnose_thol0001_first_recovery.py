"""Named groupings, pass neighbours, masked phase, and CASSBEAM squint calibration.

Reads the stored per-reference samples. Does not snap rasters or freeze a beam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_diagonal import artifact_from_samples, samples_from_records
from sl1mjax.holography_diagonal_diagnostics import (
    HOLDOUT_LIMITATION_NOTE,
    NEXT_RECOVERY_ORDER,
    calibrate_squint_estimator,
    first_recovery_artifact_dict,
    on_axis_sample_audit,
    pass_nearest_neighbour_report,
    phase_smoothness_report,
    physical_diagonal_diagnostics,
    recovery_group_counts,
    reference_treatment_report,
    unique_measured_cells,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-cassbeam-squint", action="store_true")
    arguments = parser.parse_args()
    payload = json.loads(arguments.from_samples.read_text())
    samples = samples_from_records(payload["samples"])
    artifact = artifact_from_samples(
        samples,
        calibration_state=str(payload.get("calibration_state", "casa_parang_true")),
        source_name=str(payload.get("source_name", "3C147")),
        reference_combination=str(
            payload.get("reference_combination", "per_reference_identity_gauge")
        ),
        notes=tuple(payload.get("notes", ())),
    )
    valid = [sample for sample in artifact.samples if sample.valid]
    report = first_recovery_artifact_dict(
        artifact,
        reference_report=reference_treatment_report(artifact),
        physical=physical_diagonal_diagnostics(artifact),
        include_samples=False,
    )
    report["on_axis_sample_audit"] = on_axis_sample_audit(valid)
    report["pass_nearest_neighbour"] = pass_nearest_neighbour_report(valid)
    report["phase_smoothness"] = phase_smoothness_report(unique_measured_cells(valid))
    report["group_counts"] = recovery_group_counts(artifact)
    if not arguments.skip_cassbeam_squint:
        report["squint_estimator_calibration"] = calibrate_squint_estimator(artifact)
    report["next_recovery_order"] = list(NEXT_RECOVERY_ORDER)
    report["notes"] = list(report.get("notes", [])) + [HOLDOUT_LIMITATION_NOTE]
    report["samples_path"] = str(arguments.from_samples)
    write_json(report, arguments.output)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
