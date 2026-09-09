"""Compact v3 survey publication. Does not overwrite v1 or v2."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from sl1mjax.beam_validation_outputs import (
    PUBLICATION_VERSION,
    PUBLICATION_VERSION_V2,
    PUBLICATION_VERSION_V3,
    file_checksums,
    sanitize_provenance_paths,
    sha256_bytes,
    write_json,
)
from sl1mjax.evla_c_diagonal_survey import (
    CATALOG_ID,
    PUBLICATION_BUNDLE_VERSION,
    RESIDUAL_JONES_POLICY,
    refuse_frozen_write,
)
from sl1mjax.evla_c_survey_beam import IMAGING_NODE_MHZ, OPT_IN_SURVEY_BEAM


def survey_supersession_ledger() -> dict[str, object]:
    return {
        "publication_version": PUBLICATION_BUNDLE_VERSION,
        "production_accepted": False,
        "full_jones_is_gate": False,
        "entries": [
            {
                "item": "generic commanded HOLORASTER comparison",
                "status": "historical",
                "reason": "Used generic VLA feed at commanded AZELGEO offsets",
            },
            {
                "item": "vla_c_band_beam_validation_v2 SPW-4 refresh",
                "status": "preserved_checkpoint",
                "reason": "Channel-32 residual Jones reused across its series; not band-ready",
            },
            {
                "item": "SPW 5",
                "status": "declared_diagonal_frequency_transfer",
                "reason": "Opened under identity residual Jones; not a full-Jones reopen",
            },
            {
                "item": "channel-32 residual Jones apply",
                "status": "not_applied",
                "reason": "Survey policy is identity residual Jones",
            },
            {
                "item": "width 1.04 / inherited squint / convention ladder",
                "status": "not_inherited",
                "reason": "Unit EVLA-C physical model only",
            },
            {
                "item": "full Jones scientific acceptance",
                "status": "diagnostic_not_gate",
                "reason": "Refresh outcome inconclusive_sensitivity; not required for imaging",
            },
            {
                "item": "3C391 seven-pointing survey smoke",
                "status": "complete",
                "reason": "Finite train/reload loss on C1–C7 at 4536/4598/4662 MHz",
            },
            {
                "item": "Pass B within-SPW extras outside SPWs 4–6",
                "status": "pass_a_only",
                "reason": "Pass A complete; Pass B limited to lower-C SPWs 4–6",
            },
            {
                "item": "band-wide numerical convergence",
                "status": "incomplete",
                "reason": "No low/mid/high aperture-resolution and angular-sampling suite",
            },
            {
                "item": "upper-C archive integrity",
                "status": "unverified",
                "reason": "Inner ms.tgz CRC error; casacore readability is not a payload checksum",
            },
        ],
    }


def imaging_handoff(
    *,
    catalog_digest: str | None,
    scored_slots: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    accepted = [
        {
            "execution": slot.get("execution"),
            "frequency_hz": slot["frequency_hz"],
            "spectral_window_id": slot.get("spectral_window_id"),
            "channel": slot.get("channel"),
            "main_lobe_rr": slot.get("main_lobe_rr"),
            "main_lobe_ll": slot.get("main_lobe_ll"),
        }
        for slot in scored_slots
        if slot.get("main_lobe_both_hands_accepted")
    ]
    limited = [
        {
            "execution": slot.get("execution"),
            "frequency_hz": slot["frequency_hz"],
            "spectral_window_id": slot.get("spectral_window_id"),
            "channel": slot.get("channel"),
            "main_lobe_rr": slot.get("main_lobe_rr"),
            "main_lobe_ll": slot.get("main_lobe_ll"),
            "status": slot.get("status"),
        }
        for slot in scored_slots
        if not slot.get("main_lobe_both_hands_accepted")
    ]
    return {
        "catalog_id": CATALOG_ID,
        "beam_mode": OPT_IN_SURVEY_BEAM,
        "catalog_digest": catalog_digest,
        "interpolation_policy": "refused",
        "nearest_plane_substitution": False,
        "airy_fallback": False,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "imaging_node_mhz": list(IMAGING_NODE_MHZ),
        "native_3c391_hz": [4.536e9, 4.662e9],
        "native_3c391_channel_step_mhz": 2,
        "usable_main_lobe_slots": accepted,
        "limitation_slots": limited,
        "frequency_support": "sampled_centres_not_continuous",
        "pass_b_scope": "lower_c_spw_4_6_only",
        "numerical_qualification": "incomplete_no_band_wide_convergence",
        "upper_c_archive_integrity": "unverified",
        "imaging_channel_policy": "fixed_nodes_4536_4598_4662",
        "native_smoke": "engineering_check_not_mosaic_validation",
        "smoke_command": (
            "PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu "
            "python -u scripts/run_3c391_survey_smoke.py "
            "--native-root outputs/3c391_native_averaging_ablation "
            "--survey-catalog-root "
            "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 "
            + (
                f"--survey-catalog-digest {catalog_digest} "
                if catalog_digest
                else "--survey-catalog-digest <digest-after-manifest> "
            )
            + "--output outputs/3c391_survey_smoke"
        ),
        "mosaic_command": (
            "PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu "
            "python -u scripts/run_3c391_phase6_bacchus.py "
            "--stage baseline --beams evla_c_diagonal_survey_v1 "
            "--native-root outputs/3c391_native_averaging_ablation "
            "--survey-catalog-root "
            "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 "
            + (
                f"--survey-catalog-digest {catalog_digest} "
                if catalog_digest
                else "--survey-catalog-digest <digest-after-manifest> "
            )
            + "--output outputs/3c391_survey_mosaic"
        ),
        "command": (
            "PYTHONPATH=/tmp/sl1mjax-evla-c-survey JAX_PLATFORMS=cpu "
            "python -u scripts/run_3c391_phase6_bacchus.py "
            "--stage baseline --beams evla_c_diagonal_survey_v1 "
            "--native-root outputs/3c391_native_averaging_ablation "
            "--survey-catalog-root "
            "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1 "
            + (
                f"--survey-catalog-digest {catalog_digest} "
                if catalog_digest
                else "--survey-catalog-digest <digest-after-manifest> "
            )
            + "--output outputs/3c391_survey_mosaic"
        ),
    }


def write_survey_v3_bundle(
    *,
    ledger_dir: Path,
    output_dir: Path,
    catalog_digest: str | None = None,
    phase5_dir: Path | None = None,
) -> Path:
    """Write a laptop-sized survey bundle. Refuses v1/v2 destinations."""

    destination = Path(output_dir)
    refuse_frozen_write(destination)
    if PUBLICATION_VERSION in destination.parts or PUBLICATION_VERSION_V2 in destination.parts:
        raise RuntimeError("refusing to overwrite a frozen publication bundle")
    if destination.name != PUBLICATION_VERSION_V3 and PUBLICATION_VERSION_V3 not in destination.parts:
        raise ValueError(f"survey bundle must be named {PUBLICATION_VERSION_V3}")
    destination.mkdir(parents=True, exist_ok=True)
    ledger = Path(ledger_dir)
    suitability = json.loads((ledger / "suitability_table.json").read_text(encoding="utf-8"))
    documents = {
        "suitability_table.json": suitability,
        "frequency_table.json": json.loads((ledger / "frequency_table.json").read_text(encoding="utf-8")),
        "calibration_policy.json": json.loads(
            (ledger / "calibration_policy.json").read_text(encoding="utf-8")
        ),
        "execution_inventory.json": json.loads(
            (ledger / "execution_inventory.json").read_text(encoding="utf-8")
        ),
        "supersession_ledger.json": survey_supersession_ledger(),
        **(
            {
                "slot_status.json": json.loads(
                    (ledger / "slot_status.json").read_text(encoding="utf-8")
                )
            }
            if (ledger / "slot_status.json").is_file()
            else {}
        ),
        "imaging_handoff.json": imaging_handoff(
            catalog_digest=catalog_digest,
            scored_slots=suitability.get("slots") or (),
        ),
        **(
            {
                "smoke_summary.json": json.loads(
                    (ledger / "smoke_summary.json").read_text(encoding="utf-8")
                )
            }
            if (ledger / "smoke_summary.json").is_file()
            else {}
        ),
        **(
            {
                "catalog_record.json": json.loads(
                    (ledger / "catalog_record.json").read_text(encoding="utf-8")
                )
            }
            if (ledger / "catalog_record.json").is_file()
            else {}
        ),
        **(
            {
                "fixed_geometry_table.json": json.loads(
                    (ledger / "fixed_geometry_table.json").read_text(encoding="utf-8")
                )
            }
            if (ledger / "fixed_geometry_table.json").is_file()
            else {}
        ),
        **(
            {
                "status.json": json.loads(
                    (ledger / "status.json").read_text(encoding="utf-8")
                )
            }
            if (ledger / "status.json").is_file()
            else {}
        ),
    }
    phase5 = Path(phase5_dir) if phase5_dir is not None else ledger / "phase5"
    if (phase5 / "phase5_diagnostics.json").is_file():
        documents["phase5_diagnostics.json"] = json.loads(
            (phase5 / "phase5_diagnostics.json").read_text(encoding="utf-8")
        )
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    for name in (
        "frequency_radius_suitability.png",
        "pass_b_mainlobe.png",
        "nulls_vs_frequency.png",
        "spw02_channel32_spatial.png",
        "spw02_channel32_signed_cuts.png",
        "spw03_channel32_spatial.png",
        "spw03_channel32_signed_cuts.png",
        "spw04_channel32_spatial.png",
        "spw04_channel32_signed_cuts.png",
        "spw05_channel32_spatial.png",
        "spw05_channel32_signed_cuts.png",
        "spw06_channel32_spatial.png",
        "spw06_channel32_signed_cuts.png",
        "spw07_channel32_spatial.png",
        "spw07_channel32_signed_cuts.png",
        "spw15_channel32_spatial.png",
        "upper_c_spw00_channel32_spatial.png",
        "upper_c_spw01_channel32_spatial.png",
        "upper_c_spw15_channel32_spatial.png",
    ):
        for source in (phase5 / name, ledger / name):
            if source.is_file():
                shutil.copy2(source, figures / name)
                break
    for name, payload in documents.items():
        write_json(sanitize_provenance_paths(payload), destination / name)
    checksums = file_checksums(destination)
    checksums.pop("manifest.json", None)
    manifest = {
        "schema_version": 1,
        "publication_version": PUBLICATION_VERSION_V3,
        "catalog_id": CATALOG_ID,
        "production_accepted": False,
        "full_jones_is_gate": False,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "interpolation_policy": "refused",
        "preserves": [PUBLICATION_VERSION, PUBLICATION_VERSION_V2],
        "catalog_digest": catalog_digest,
        "files": checksums,
    }
    manifest["bundle_sha256"] = sha256_bytes(
        json.dumps({"files": checksums}, sort_keys=True).encode()
    )
    write_json(manifest, destination / "manifest.json")
    return destination
