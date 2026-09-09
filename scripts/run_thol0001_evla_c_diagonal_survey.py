"""Resumable EVLA-C band-wide diagonal survey driver.

Does not overwrite frozen products or apply residual Jones.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from sl1mjax.evla_c_diagonal_survey import (
    BEAM_ROOT,
    LOCAL_LEDGER,
    LOWER_C_MS,
    PRODUCT_DIR,
    SCIENTIFIC_SPW45_MS,
    STAGING_SCRIPTS,
    UPPER_C_CANDIDATE_MS,
    calibration_policy,
    default_source_paths,
    holoraster_calibration_policy,
    empty_status,
    experiment_manifest,
    planned_slots,
    refuse_frozen_write,
    source_snapshot,
    update_task,
    write_json_atomic,
)
from sl1mjax.holography_ms import read_measurement_set_light_inventory


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dirty(repo: Path) -> bool:
    try:
        text = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(text.strip())


def _inventory_execution(path: Path) -> dict[str, object]:
    refuse_frozen_write(path)
    identity = read_measurement_set_light_inventory(path)
    windows = list(identity.get("spectral_windows") or [])
    slots = planned_slots(windows)
    return {
        "identity": identity,
        "band_class": identity.get("band_class"),
        "spectral_windows": windows,
        "n_spw": len(windows),
        "n_pass_a": sum(1 for row in slots if row.get("pass") == "A"),
        "n_pass_b": sum(1 for row in slots if row.get("pass") == "B"),
        "planned_slots": slots,
        "frequency_hz_min": identity.get("frequency_hz_min"),
        "frequency_hz_max": identity.get("frequency_hz_max"),
        "science_frequency_hz_min": identity.get("science_frequency_hz_min"),
        "science_frequency_hz_max": identity.get("science_frequency_hz_max"),
    }


def _local_ledger(repo: Path) -> Path | None:
    root = Path(repo)
    plan = root / "docs" / "evla-c-band-diagonal-survey-and-imaging-handoff-plan.md"
    if plan.is_file():
        return root / LOCAL_LEDGER
    return None


def _write_both(output: Path, ledger: Path | None, name: str, payload: dict[str, object]) -> Path:
    written = write_json_atomic(output / name, payload)
    if ledger is not None:
        write_json_atomic(ledger / name, payload)
    return written


def _load_status(output: Path) -> dict[str, object]:
    path = output / "status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return empty_status()


def _safe_inventory(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"error": "path does not exist", "path": str(path)}
    try:
        return _inventory_execution(path)
    except Exception as exc:  # noqa: BLE001 — inventory must record a local failure
        return {"error": str(exc), "path": str(path)}


def cmd_phase0(repo: Path, output: Path) -> Path:
    snapshot = source_snapshot(default_source_paths(repo))
    manifest = experiment_manifest(
        source={
            "n_files": snapshot["n_files"],
            "tree_sha256": snapshot["tree_sha256"],
            "files": {Path(name).name: digest for name, digest in dict(snapshot["files"]).items()},
        },
        git_head=_git_head(repo),
        dirty_tree=_dirty(repo),
    )
    status = update_task(
        empty_status(),
        "phase0_manifest",
        state="complete",
        artifact=str(output / "experiment_manifest.json"),
    )
    ledger = _local_ledger(repo)
    _write_both(output, ledger, "experiment_manifest.json", manifest)
    _write_both(output, ledger, "status.json", status)
    return output / "experiment_manifest.json"


def cmd_policy(repo: Path, output: Path) -> Path:
    refuse_frozen_write(output)
    policy = calibration_policy()
    ledger = _local_ledger(repo)
    _write_both(output, ledger, "calibration_policy.json", policy)
    status = _load_status(output)
    status = update_task(
        status,
        "phase2_calibration_policy",
        state="complete",
        reason="policy recorded; new SPWs still need work copies",
        artifact=str(output / "calibration_policy.json"),
    )
    _write_both(output, ledger, "status.json", status)
    return output / "calibration_policy.json"


def cmd_inventory(repo: Path, output: Path, *, include_upper: bool) -> Path:
    refuse_frozen_write(PRODUCT_DIR)
    refuse_frozen_write(output)
    ledger = _local_ledger(repo)
    executions: dict[str, object] = {}
    status = _load_status(output)
    status = update_task(
        status,
        "phase0_manifest",
        state="complete",
        artifact=str(output / "experiment_manifest.json"),
    )
    lower = _safe_inventory(LOWER_C_MS)
    executions["lower_c"] = lower
    _write_both(output, ledger, "lower_c_inventory.json", lower)
    lower_ok = "error" not in lower
    status = update_task(
        status,
        "phase1_lower_c_inventory",
        state="complete" if lower_ok else "blocked",
        reason=None if lower_ok else str(lower.get("error")),
        artifact=str(output / "lower_c_inventory.json"),
    )
    if SCIENTIFIC_SPW45_MS.exists():
        scientific = _safe_inventory(SCIENTIFIC_SPW45_MS)
        executions["scientific_spw45"] = scientific
        _write_both(output, ledger, "scientific_spw45_inventory.json", scientific)
    if include_upper:
        upper = _safe_inventory(UPPER_C_CANDIDATE_MS)
        executions["upper_c_candidate"] = upper
        _write_both(output, ledger, "upper_c_candidate_inventory.json", upper)
        if "error" in upper:
            identity_state = "blocked"
            reason = str(upper.get("error"))
            inventory_state = "blocked"
        else:
            band = str(upper.get("band_class") or "unknown")
            identity_state = "complete"
            reason = band
            inventory_state = "complete" if band == "upper_c_candidate" else "blocked"
            if band != "upper_c_candidate":
                reason = f"identified as {band}; not used as upper-C"
        status = update_task(
            status,
            "phase1_upper_c_identity",
            state=identity_state,
            reason=reason,
            artifact=str(output / "upper_c_candidate_inventory.json"),
        )
        status = update_task(
            status,
            "phase1_upper_c_inventory",
            state=inventory_state,
            reason=reason,
            artifact=str(output / "upper_c_candidate_inventory.json"),
        )
    policy = calibration_policy()
    _write_both(output, ledger, "calibration_policy.json", policy)
    status = update_task(
        status,
        "phase2_calibration_policy",
        state="complete",
        reason="policy recorded; new SPWs still need work copies",
        artifact=str(output / "calibration_policy.json"),
    )
    _write_both(output, ledger, "execution_inventory.json", executions)
    _write_both(output, ledger, "status.json", status)
    return output / "execution_inventory.json"


def cmd_planes(repo: Path, output: Path, beam_root: Path) -> Path:
    from sl1mjax.cassbeam_highres import write_development_highres_manifest
    from sl1mjax.evla_c_survey_compare import SURVEY_MODEL_ID

    refuse_frozen_write(beam_root)
    refuse_frozen_write(output)
    manifest = write_development_highres_manifest(
        beam_root,
        model_id=SURVEY_MODEL_ID,
        name_prefix="evla-cband",
    )
    ledger = _local_ledger(repo)
    _write_both(output, ledger, "beam_manifest.json", manifest)
    status = _load_status(output)
    status = update_task(
        status,
        "phase3_planes",
        state="running",
        reason=f"{len(list(manifest.get('planes') or []))} generated planes",
        artifact=str(beam_root / "manifest.json"),
    )
    _write_both(output, ledger, "status.json", status)
    return beam_root / "manifest.json"


def cmd_score(
    repo: Path,
    output: Path,
    *,
    beam_root: Path,
    scripts_dir: Path,
    measurement_set: Path,
    spectral_window_id: int,
    channel: int,
    rewrite_catalog_manifest: bool = True,
    rewrite_status: bool = True,
) -> Path:
    from sl1mjax.cassbeam_highres import HighresCassbeamCatalog, write_development_highres_manifest
    from sl1mjax.evla_c_survey_compare import (
        SURVEY_MODEL_ID,
        load_comparison_module,
        load_holoraster_slot,
        predict_diagonal,
        score_extract,
        write_slot_products,
    )

    refuse_frozen_write(output)
    refuse_frozen_write(beam_root)
    print(
        f"score start SPW {spectral_window_id} ch{channel} beam={beam_root}",
        flush=True,
    )
    if rewrite_catalog_manifest:
        write_development_highres_manifest(
            beam_root,
            model_id=SURVEY_MODEL_ID,
            name_prefix="evla-cband",
        )
        print("score catalog manifest written", flush=True)
    catalog = HighresCassbeamCatalog(beam_root, expected_model_id=SURVEY_MODEL_ID)
    print("score catalog loaded", flush=True)
    cmp = load_comparison_module(scripts_dir)
    print("score comparison helpers loaded", flush=True)
    print("score extracting", flush=True)
    extract = load_holoraster_slot(
        cmp,
        measurement_set=measurement_set,
        spectral_window_id=int(spectral_window_id),
        channel=int(channel),
    )
    print(
        f"score extracted n={extract.get('n_development')} "
        f"freq={extract.get('frequency_hz')}",
        flush=True,
    )
    predicted = predict_diagonal(catalog=catalog, extract=extract)
    print("score predicted", flush=True)
    report = score_extract(extract, predicted)
    paths = write_slot_products(output, extract=extract, predicted=predicted, report=report)
    if rewrite_status:
        ledger = _local_ledger(repo)
        status = _load_status(output)
        task = "phase4_pass_a"
        status = update_task(
            status,
            task,
            state="running",
            reason=(
                f"scored SPW {spectral_window_id} ch{channel} "
                f"status={report.get('status')}"
            ),
            artifact=paths["report"],
        )
        _write_both(output, ledger, "status.json", status)
    return Path(paths["report"])


def cmd_workcopy(
    repo: Path,
    output: Path,
    *,
    spectral_window_id: int,
    source: Path,
    execution: str,
) -> Path:
    from sl1mjax.evla_c_diagonal_survey import UPPER_C_CANDIDATE_MS, work_ms_path
    from sl1mjax.holography_commissioning import copy_commissioning_measurement_set

    if execution not in {"lower_c", "upper_c"}:
        raise ValueError(f"unknown execution {execution!r}")
    if execution == "upper_c" and Path(source).resolve() != UPPER_C_CANDIDATE_MS.resolve():
        raise ValueError("upper-C work copies must come from the identified upper-C MS")
    if execution == "lower_c" and Path(source).resolve() != LOWER_C_MS.resolve():
        raise ValueError("lower-C work copies must come from the identified lower-C MS")
    refuse_frozen_write(output)
    dest = work_ms_path(execution, spectral_window_id)
    refuse_frozen_write(dest)
    print(f"workcopy {execution} SPW {spectral_window_id} {source} -> {dest}", flush=True)
    selection = copy_commissioning_measurement_set(
        source,
        dest,
        spectral_window_ids=(int(spectral_window_id),),
        immutable=False,
        kind="evla_c_diagonal_survey_work_ms",
    )
    payload = {
        "destination": str(dest),
        "source": str(source),
        "execution": execution,
        "spectral_window_ids": list(selection.spectral_window_ids),
        "field_ids": list(selection.field_ids),
        "field_names": list(selection.field_names),
        "immutable": False,
        "lower_c_scan_list_reusable": execution == "lower_c",
    }
    ledger = _local_ledger(repo)
    _write_both(
        output,
        ledger,
        f"work_ms_{execution}_spw{int(spectral_window_id):02d}.json",
        payload,
    )
    return dest


def cmd_scan_inventory(repo: Path, output: Path, *, source: Path, execution: str) -> Path:
    from sl1mjax.holography_ms import read_unique_scan_field_state

    refuse_frozen_write(output)
    rows = read_unique_scan_field_state(source)
    check_fields = sorted(
        {row["field_name"] for row in rows if str(row["field_name"]).startswith("J")}
    )
    payload = {
        "execution": execution,
        "source": str(source),
        "n_rows": len(rows),
        "check_fields": check_fields,
        "uses_3c286": any(name == "J1331+3030" for name in check_fields),
        "uses_3c138": any(name == "J0521+1638" for name in check_fields),
        "lower_c_scan_list_reusable": execution == "lower_c",
        "scans": rows,
    }
    ledger = _local_ledger(repo)
    dest = _write_both(output, ledger, f"{execution}_scan_inventory.json", payload)
    if execution == "upper_c":
        _write_both(
            output,
            ledger,
            "upper_c_calibration_policy.json",
            holoraster_calibration_policy(payload),
        )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument(
        "--stage",
        choices=(
            "phase0",
            "inventory",
            "policy",
            "planes",
            "score",
            "reexport",
            "workcopy",
            "scan-inventory",
        ),
        default="phase0",
    )
    parser.add_argument("--include-upper", action="store_true")
    parser.add_argument("--beam-root", type=Path, default=BEAM_ROOT)
    parser.add_argument("--scripts-dir", type=Path, default=STAGING_SCRIPTS)
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_SPW45_MS)
    parser.add_argument("--spw", type=int, default=5)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--execution", choices=("lower_c", "upper_c"), default="lower_c")
    parser.add_argument("--source-ms", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    arguments = parser.parse_args()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "phase0":
        print(cmd_phase0(arguments.repo_root, output))
        return
    if arguments.stage == "policy":
        print(cmd_policy(arguments.repo_root, output))
        return
    if arguments.stage == "planes":
        print(cmd_planes(arguments.repo_root, output, arguments.beam_root))
        return
    if arguments.stage == "workcopy":
        source = arguments.source_ms
        if source is None:
            source = UPPER_C_CANDIDATE_MS if arguments.execution == "upper_c" else LOWER_C_MS
        print(
            cmd_workcopy(
                arguments.repo_root,
                output,
                spectral_window_id=arguments.spw,
                source=source,
                execution=arguments.execution,
            )
        )
        return
    if arguments.stage in {"score", "reexport"}:
        export_dir = arguments.export_dir or output
        print(
            cmd_score(
                arguments.repo_root,
                export_dir,
                beam_root=arguments.beam_root,
                scripts_dir=arguments.scripts_dir,
                measurement_set=arguments.measurement_set,
                spectral_window_id=arguments.spw,
                channel=arguments.channel,
                rewrite_catalog_manifest=arguments.stage == "score",
                rewrite_status=arguments.stage == "score",
            )
        )
        return
    if arguments.stage == "scan-inventory":
        source = arguments.source_ms
        if source is None:
            source = UPPER_C_CANDIDATE_MS if arguments.execution == "upper_c" else LOWER_C_MS
        print(
            cmd_scan_inventory(
                arguments.repo_root,
                output,
                source=source,
                execution=arguments.execution,
            )
        )
        return
    print(cmd_inventory(arguments.repo_root, output, include_upper=arguments.include_upper))


if __name__ == "__main__":
    main()
