"""EVLA-C band-wide diagonal survey specification.

Locks sampling, calibration policy, and paths for
``docs/evla-c-band-diagonal-survey-and-imaging-handoff-plan.md``.
Does not promote a production beam or apply residual Jones.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPERIMENT_ID = "evla_c_diagonal_survey_v1"
CATALOG_ID = "evla_c_diagonal_survey_v1"
PUBLICATION_BUNDLE_VERSION = "vla_c_band_beam_validation_v3"
COORDINATE_QUERY = "source_lm_feed"
AXIS_MAP = "az_to_plus_l_el_to_plus_m_no_swap"
PRIMARY_ARM = "evla_c_source_diagonal"
DIAGNOSTIC_ARM = "evla_c_source_full_jones"
CORRELATION_ORDER = ("RR", "RL", "LR", "LL")
JONES_PACKING = "[[RR, RL], [LR, LL]]"
ALGEBRA_ATOL = 1.0e-12
MAIN_LOBE_ACCEPTED_MAX = 0.01
MID_BEAM_QUALIFIED_MAX = 0.12
PASS_A_NUMERATOR = 32
PASS_B_NUMERATORS = (8, 24, 32, 40, 56)
CHANNEL_INDEX_DENOMINATOR = 64
TAPER_VALID_HZ = (3.9e9, 8.1e9)
RESIDUAL_JONES_POLICY = "not_applied"
SPW5_ROLE = "declared_diagonal_frequency_transfer"

LOWER_C_MS = Path(
    "/media/stephen/astro/vla/extracted/"
    "THOL0001.sb31628704.eb31629959.57401.169024456016.ms"
)
UPPER_C_CANDIDATE_MS = Path(
    "/media/stephen/astro/vla/extracted/"
    "THOL0001.sb31635131.eb31644651.57405.16003953703.ms"
)
SCIENTIFIC_SPW45_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/"
    "THOL0001.lowerC.spw45.scientific.ms"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "evla_c_diagonal_survey_v1"
)
BEAM_ROOT = Path("/media/stephen/astro/vla/beam_models/cassbeam_evla_c_diagonal_survey_v1")
STAGING_PACKAGE = Path("/tmp/sl1mjax-evla-c-survey")
STAGING_SCRIPTS = Path("/tmp/sl1mjax-evla-c-survey-scripts")
LOCAL_LEDGER = Path("docs/evla_c_diagonal_survey")

FROZEN_WRITE_NAMES = frozenset(
    {
        "vla_c_band_beam_validation_v1",
        "vla_c_band_beam_validation_v2",
        "cassbeam_cband_full_jones_g1024_p32_20260906",
        "cassbeam_evla_c_g1024_p32_20260908",
        "cassbeam_evla_c_g1024_p32_ninepub_20260908",
        "holoraster_cassbeam_comparison",
        "holoraster_coordinate_feed_comparison_v1",
        "c147_offset_ring",
        "evla_c_full_jones_validation_refresh_v1",
    }
)

TASK_IDS = (
    "phase0_manifest",
    "phase1_lower_c_inventory",
    "phase1_upper_c_identity",
    "phase1_upper_c_inventory",
    "phase2_calibration_policy",
    "phase2_excursion_diagnosis",
    "phase3_planes",
    "phase3_convergence",
    "phase4_pass_a",
    "phase4_pass_b",
    "phase5_plots",
    "phase6_catalog",
    "phase6_imaging_adapter",
    "phase7_publication",
)


def pass_a_channel(n_chan: int) -> int:
    """Declared centre-channel index for a spectral window."""

    count = int(n_chan)
    if count <= 0:
        raise ValueError("n_chan must be positive")
    return min(count - 1, count * PASS_A_NUMERATOR // CHANNEL_INDEX_DENOMINATOR)


def pass_b_channels(n_chan: int) -> tuple[int, ...]:
    """Fixed within-SPW channel set, including the Pass-A centre."""

    count = int(n_chan)
    if count <= 0:
        raise ValueError("n_chan must be positive")
    chosen = {
        min(count - 1, count * numerator // CHANNEL_INDEX_DENOMINATOR)
        for numerator in PASS_B_NUMERATORS
    }
    return tuple(sorted(chosen))


def nearest_supported_channel(
    n_chan: int,
    supported: Sequence[int],
    *,
    preferred: int | None = None,
) -> int:
    """Replace an unsupported Pass-A channel. Lower index wins a distance tie."""

    count = int(n_chan)
    usable = sorted({int(item) for item in supported if 0 <= int(item) < count})
    if not usable:
        raise ValueError("no supported channel in this spectral window")
    target = pass_a_channel(count) if preferred is None else int(preferred)
    if target in usable:
        return target
    return min(usable, key=lambda index: (abs(index - target), index))


def taper_in_documented_range(frequency_hz: float) -> bool:
    lo, hi = TAPER_VALID_HZ
    return lo <= float(frequency_hz) <= hi


SCIENCE_SPW_ID_MAX = 15


def classify_execution_band(freq_min_hz: float, freq_max_hz: float) -> str:
    """Classify a contiguous science band from frequency extrema.

    Do not pass the extrema of a mixed table that still contains auxiliary
    windows from another band. Use ``classify_spectral_windows`` for that.
    """

    low = float(freq_min_hz)
    high = float(freq_max_hz)
    if high < low:
        raise ValueError("frequency max is below min")
    if low >= 5.9e9 and high <= 8.2e9:
        return "upper_c_candidate"
    if low >= 3.8e9 and high <= 6.2e9:
        return "lower_c"
    if low >= 8.0e9:
        return "not_c_band"
    return "ambiguous"


def classify_spectral_windows(windows: Sequence[Mapping[str, object]]) -> str:
    """Classify from window centres so auxiliary SPWs cannot flip the band."""

    centres: list[float] = []
    science_lo: list[float] = []
    science_hi: list[float] = []
    for window in windows:
        if window.get("freq_min_hz") is None or window.get("freq_max_hz") is None:
            continue
        low = float(window["freq_min_hz"])
        high = float(window["freq_max_hz"])
        centres.append(0.5 * (low + high))
        if int(window.get("spectral_window_id", -1)) <= SCIENCE_SPW_ID_MAX:
            science_lo.append(low)
            science_hi.append(high)
    n_x = sum(1 for centre in centres if centre >= 8.0e9)
    n_upper_core = sum(1 for centre in centres if 6.2e9 < centre <= 8.2e9)
    n_lower_core = sum(1 for centre in centres if 3.8e9 <= centre < 5.9e9)
    n_upper = sum(1 for centre in centres if 5.9e9 <= centre <= 8.2e9)
    n_lower = sum(1 for centre in centres if 3.8e9 <= centre <= 6.2e9)
    if n_x >= 8:
        return "not_c_band"
    if n_upper_core >= 8 or n_upper >= 12:
        return "upper_c_candidate"
    if n_lower_core >= 8 or n_lower >= 12:
        return "lower_c"
    if science_lo and science_hi:
        return classify_execution_band(min(science_lo), max(science_hi))
    return "ambiguous"


def window_role(window: Mapping[str, object], band_class: str) -> str:
    """Label science versus auxiliary windows after the execution is identified."""

    spw_id = int(window.get("spectral_window_id", -1))
    if spw_id < 0 or window.get("freq_min_hz") is None:
        return "unknown"
    centre = 0.5 * (float(window["freq_min_hz"]) + float(window["freq_max_hz"]))
    if band_class == "upper_c_candidate" and centre < 5.9e9:
        return "auxiliary_lower_c_frequency"
    if band_class == "lower_c" and centre > 6.2e9:
        return "auxiliary_upper_c_frequency"
    if spw_id > SCIENCE_SPW_ID_MAX:
        return "auxiliary"
    return "science"


def calibration_policy() -> dict[str, object]:
    """Locked calibration rules. This is not a solve script."""

    return {
        "reuse_incompatible_tables": False,
        "solve_on_holoraster_movers": False,
        "residual_jones": RESIDUAL_JONES_POLICY,
        "scientific_spw45": {
            "path": str(SCIENTIFIC_SPW45_MS),
            "role": "reuse_for_spw4_and_spw5_diagonal_only",
            "overwrite": False,
            "residual_jones": RESIDUAL_JONES_POLICY,
        },
        "new_spws": {
            "require_new_work_copies": True,
            "source_ms": str(LOWER_C_MS),
            "do_not_overwrite_scientific_ms": True,
            "do_not_overwrite_full_ms": True,
            "work_root": str(PRODUCT_DIR / "work_ms"),
        },
        "channel32_residual_jones": {
            "apply": False,
            "sha256": "ff6aee560ab1521da1472ce833393ad01d3f64bbc1200f6e71b4b5088aef12bc",
            "reason": "channel-32 residual is not a band-ready frequency model",
        },
        "full_jones_is_gate": False,
        "spw5_role": SPW5_ROLE,
    }


FIELD_3C147 = "J0542+4951"
FIELD_3C138 = "J0521+1638"
FIELD_3C286 = "J1331+3030"
LOWER_C_DEFAULT_FLUX_SCANS = "2,51"


def holoraster_calibration_policy(inventory: Mapping[str, object]) -> dict[str, object]:
    """Derive G/K/B scan lists from a scan/field/state inventory.

    Upper-C uses 3C147 plus a 3C138 check field. It must not inherit the
    lower-C 3C286 scan list.
    """

    scans = [dict(row) for row in list(inventory.get("scans") or ())]
    if not scans:
        raise ValueError("scan inventory is empty")
    execution = str(inventory.get("execution") or "")

    def _scans(*, field_name: str | None = None, field_id: int | None = None, mode_has: str) -> list[int]:
        found: set[int] = set()
        for row in scans:
            mode = str(row.get("state_mode") or "")
            if mode_has not in mode:
                continue
            if "CALIBRATE_POINTING" in mode:
                continue
            if field_name is not None and str(row.get("field_name")) != field_name:
                continue
            if field_id is not None and int(row["field_id"]) != int(field_id):
                continue
            found.add(int(row["scan_number"]))
        return sorted(found)

    flux_field_ids = sorted(
        {
            int(row["field_id"])
            for row in scans
            if str(row.get("field_name")) == FIELD_3C147
            and "CALIBRATE_FLUX" in str(row.get("state_mode") or "")
        }
    )
    phase_field_ids = sorted(
        {
            int(row["field_id"])
            for row in scans
            if str(row.get("field_name")) == FIELD_3C147
            and "CALIBRATE_PHASE" in str(row.get("state_mode") or "")
            and "CALIBRATE_FLUX" not in str(row.get("state_mode") or "")
        }
    )
    holoraster_ids = sorted(
        {int(row["field_id"]) for row in scans if str(row.get("field_name")) == "HOLORASTER"}
    )
    check_138 = sorted(
        {int(row["field_id"]) for row in scans if str(row.get("field_name")) == FIELD_3C138}
    )
    check_286 = sorted(
        {int(row["field_id"]) for row in scans if str(row.get("field_name")) == FIELD_3C286}
    )
    if len(flux_field_ids) != 1 or len(phase_field_ids) != 1 or len(holoraster_ids) != 1:
        raise ValueError("ambiguous 3C147/HOLORASTER field ids in scan inventory")
    flux_scans = _scans(field_id=flux_field_ids[0], mode_has="CALIBRATE_FLUX")
    if len(flux_scans) < 2:
        raise ValueError("need two 3C147 flux/bandpass scans")
    phase_scans = _scans(field_id=phase_field_ids[0], mode_has="CALIBRATE_PHASE")
    if not phase_scans:
        raise ValueError("no 3C147 phase scans")
    flux_csv = ",".join(str(scan) for scan in flux_scans)
    if execution == "upper_c" and flux_csv == LOWER_C_DEFAULT_FLUX_SCANS:
        raise ValueError("upper-C flux scans must not reuse the lower-C 2,51 list")
    uses_3c138 = bool(inventory.get("uses_3c138") or check_138)
    uses_3c286 = bool(inventory.get("uses_3c286") or check_286)
    if execution == "upper_c" and not uses_3c138:
        raise ValueError("upper-C inventory must include J0521+1638")
    if execution == "upper_c" and uses_3c286:
        raise ValueError("upper-C inventory unexpectedly includes 3C286")
    check_field = check_138[0] if check_138 else (check_286[0] if check_286 else None)
    on_axis = f"{flux_field_ids[0]},{phase_field_ids[0]}"
    prediction = f"{on_axis},{holoraster_ids[0]}"
    apply_fields = prediction if check_field is None else f"{prediction},{check_field}"
    return {
        "execution": execution,
        "flux_scans": flux_csv,
        "solve_flux_scan": str(flux_scans[0]),
        "held_out_flux_scan": str(flux_scans[-1]),
        "phase_scans": ",".join(str(scan) for scan in phase_scans),
        "flux_field": str(flux_field_ids[0]),
        "d_field": str(phase_field_ids[0]),
        "holoraster_field": str(holoraster_ids[0]),
        "check_field": None if check_field is None else str(check_field),
        "check_source": FIELD_3C138 if uses_3c138 else (FIELD_3C286 if uses_3c286 else None),
        "on_axis_fields": on_axis,
        "prediction_fields": prediction,
        "apply_fields": apply_fields,
        "uses_3c138": uses_3c138,
        "uses_3c286": uses_3c286,
        "lower_c_scan_list_reusable": execution == "lower_c",
        "residual_jones": RESIDUAL_JONES_POLICY,
        "solve_on_holoraster_movers": False,
    }


def planned_slots(windows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Declared Pass-A and Pass-B slots. Empty support stays empty."""

    slots: list[dict[str, object]] = []
    for window in windows:
        n_chan = int(window.get("n_chan") or 0)
        freqs = [float(value) for value in list(window.get("chan_freq_hz") or [])]
        if n_chan <= 0 or len(freqs) != n_chan:
            continue
        declared = window.get("pass_a_channel")
        pass_a = int(declared if declared is not None else pass_a_channel(n_chan))
        pass_b = tuple(
            int(index)
            for index in (window.get("pass_b_channels") or pass_b_channels(n_chan))
        )
        for pass_name, channels in (("A", (pass_a,)), ("B", pass_b)):
            for channel in channels:
                slots.append(
                    {
                        "spectral_window_id": int(window["spectral_window_id"]),
                        "pass": pass_name,
                        "channel": int(channel),
                        "frequency_hz": freqs[int(channel)],
                        "taper_in_documented_range": taper_in_documented_range(freqs[int(channel)]),
                        "is_pass_a": int(channel) == pass_a,
                    }
                )
    return slots


def build_slot_status(
    frequency_table: Mapping[str, object],
    scored_slots: Sequence[Mapping[str, object]],
    *,
    upper_c_inventory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Every science Pass-A/B slot gets an explained status. Missing is not zero."""

    upper_c_ready = bool(
        upper_c_inventory
        and upper_c_inventory.get("uses_3c138")
        and not upper_c_inventory.get("uses_3c286")
        and not upper_c_inventory.get("lower_c_scan_list_reusable")
    )
    scored_by_hz: dict[int, Mapping[str, object]] = {}
    for slot in scored_slots:
        if slot.get("frequency_hz") is None:
            continue
        scored_by_hz[int(round(float(slot["frequency_hz"])))] = slot
    executions = dict(frequency_table.get("executions") or {})
    pass_a_survey_complete = True
    for execution in ("lower_c", "upper_c"):
        payload = dict(executions.get(execution) or {})
        for window in payload.get("windows") or ():
            if window.get("role") != "science":
                continue
            if int(window.get("main_row_count") or 0) <= 0:
                continue
            if int(round(float(window["pass_a_frequency_hz"]))) not in scored_by_hz:
                pass_a_survey_complete = False
                break
        if not pass_a_survey_complete:
            break
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for execution in ("lower_c", "upper_c"):
        payload = dict(executions.get(execution) or {})
        for window in payload.get("windows") or ():
            if window.get("role") != "science":
                continue
            spw = int(window["spectral_window_id"])
            pass_a = int(window["pass_a_channel"])
            declared = [
                ("A", pass_a, float(window["pass_a_frequency_hz"])),
            ]
            for channel, frequency_hz in zip(
                window.get("pass_b_channels") or (),
                window.get("pass_b_frequency_hz") or (),
                strict=False,
            ):
                declared.append(("B", int(channel), float(frequency_hz)))
            for _pass_name, channel, frequency_hz in declared:
                key = (execution, spw, channel)
                if key in seen:
                    continue
                seen.add(key)
                hz_key = int(round(frequency_hz))
                scored = scored_by_hz.get(hz_key)
                row_count = int(window.get("main_row_count") or 0)
                pass_a_scored = (
                    int(round(float(window["pass_a_frequency_hz"]))) in scored_by_hz
                )
                is_pass_b = channel != pass_a
                if scored is not None:
                    state = str(scored.get("status") or "complete")
                    reason = "scored_identity_residual_jones"
                elif is_pass_b and pass_a_survey_complete:
                    state = "pass_a_only"
                    reason = "pass_b_not_opened_centre_survey_complete"
                elif execution == "upper_c":
                    state = "planned"
                    if pass_a_scored and is_pass_b:
                        reason = "pass_b_score_pending"
                    elif upper_c_ready:
                        reason = "upper_c_work_copy_or_score_pending"
                    else:
                        reason = "upper_c_3c138_scan_inventory_pending"
                elif row_count <= 0:
                    state = "no_usable_data"
                    reason = "archive_window_has_no_main_rows"
                else:
                    state = "planned"
                    reason = (
                        "pass_b_score_pending"
                        if pass_a_scored and is_pass_b
                        else "lower_c_work_copy_or_score_pending"
                    )
                rows.append(
                    {
                        "execution": execution,
                        "spectral_window_id": spw,
                        "pass": "A" if channel == pass_a else "B",
                        "channel": channel,
                        "frequency_hz": frequency_hz,
                        "taper_in_documented_range": bool(
                            window.get("taper_in_documented_range")
                        ),
                        "main_row_count": row_count,
                        "state": state,
                        "reason": reason,
                        "scored": scored is not None,
                        "main_lobe_both_hands_accepted": (
                            scored.get("main_lobe_both_hands_accepted")
                            if scored is not None
                            else None
                        ),
                    }
                )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["state"])] = counts.get(str(row["state"]), 0) + 1
    for row in rows:
        if row["execution"] == "upper_c":
            row["archive_integrity"] = "unverified"
        if row["scored"]:
            row["empirical_main_lobe_accepted"] = bool(row.get("main_lobe_both_hands_accepted"))
            row["numerically_qualified"] = False
    return {
        "schema_version": 1,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "pass_a_survey_complete": pass_a_survey_complete,
        "pass_b_scope": "lower_c_spw_4_6_only",
        "acceptance": "pass_a_complete_pass_b_limited",
        "numerical_qualification": "incomplete_no_band_wide_convergence",
        "upper_c_archive_integrity": "unverified",
        "n_slots": len(rows),
        "n_scored": sum(1 for row in rows if row["scored"]),
        "counts": counts,
        "slots": rows,
    }


def survey_holoraster_row_cache(measurement_set: Path, data_desc_id: int) -> Path | None:
    """Row-index sidecar for survey work copies only. Never archives or the scientific MS."""

    destination = Path(measurement_set)
    text = destination.resolve().as_posix()
    if f"{EXPERIMENT_ID}/work_ms" not in text:
        return None
    refuse_frozen_write(destination)
    if destination.resolve() == SCIENTIFIC_SPW45_MS.resolve():
        return None
    return destination.with_name(
        destination.name + f".holoraster_ddid{int(data_desc_id)}.rows.npy"
    )


def execution_from_measurement_set(measurement_set: Path) -> str:
    """Infer the holography execution from a survey work-copy name."""

    name = Path(measurement_set).name
    if name.startswith("upper_c_"):
        return "upper_c"
    return "lower_c"


def slot_product_stem(
    output_dir: Path,
    *,
    execution: str,
    spectral_window_id: int,
    channel: int,
) -> Path:
    """Lower-C keeps the historical `spwXX_channelYY` name. Upper-C is prefixed."""

    prefix = "" if execution == "lower_c" else f"{execution}_"
    return (
        Path(output_dir)
        / "channels"
        / f"{prefix}spw{int(spectral_window_id):02d}_channel{int(channel):02d}"
    )


def work_ms_path(execution: str, spectral_window_id: int) -> Path:
    """Writable work copy. Never the scientific SPW 4+5 MS or an archive MS."""

    name = f"{execution}_spw{int(spectral_window_id):02d}.work.ms"
    destination = PRODUCT_DIR / "work_ms" / name
    refuse_frozen_write(destination)
    if destination.resolve() == SCIENTIFIC_SPW45_MS.resolve():
        raise RuntimeError("refusing to replace the scientific SPW 4+5 Measurement Set")
    if destination.resolve() in {LOWER_C_MS.resolve(), UPPER_C_CANDIDATE_MS.resolve()}:
        raise RuntimeError("refusing to replace an archive Measurement Set")
    return destination


def refuse_residual_jones_apply() -> None:
    raise RuntimeError(
        "channel-32 residual Jones is not applied in the diagonal survey; "
        f"policy is {RESIDUAL_JONES_POLICY}"
    )


def refuse_frozen_write(path: Path) -> None:
    text = Path(path).resolve().as_posix()
    for name in FROZEN_WRITE_NAMES:
        if name in text:
            raise RuntimeError(f"refusing to overwrite frozen product {name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(paths: Sequence[Path]) -> dict[str, object]:
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in sorted(Path(item).resolve() for item in paths):
        if not path.is_file():
            continue
        hashed = sha256_file(path)
        files[path.as_posix()] = hashed
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashed.encode("utf-8"))
        digest.update(b"\0")
    return {
        "n_files": len(files),
        "tree_sha256": digest.hexdigest(),
        "files": files,
    }


def default_source_paths(repo_root: Path) -> tuple[Path, ...]:
    root = Path(repo_root)
    names = (
        "evla_c_diagonal_survey.py",
        "evla_c_independent_rime.py",
        "evla_c_validation_refresh.py",
        "cassbeam_evla_c.py",
        "cassbeam_highres.py",
        "holography_ms.py",
        "holography_holoraster_coordinates.py",
        "holography_beam_prior.py",
    )
    files = [root / "src" / "sl1mjax" / name for name in names]
    files.append(root / "scripts" / "run_thol0001_evla_c_diagonal_survey.py")
    files.append(root / "docs" / "evla-c-band-diagonal-survey-and-imaging-handoff-plan.md")
    return tuple(path for path in files if path.is_file())


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return destination


def empty_status() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": 1,
        "production_accepted": False,
        "full_jones_is_gate": False,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "spw5_role": SPW5_ROLE,
        "convention_ladder_reopened": False,
        "tasks": {
            task: {"state": "pending", "reason": None, "artifact": None} for task in TASK_IDS
        },
        "blocked": [],
        "outcome": None,
    }


def update_task(
    status: Mapping[str, object],
    task_id: str,
    *,
    state: str,
    reason: str | None = None,
    artifact: str | None = None,
) -> dict[str, object]:
    if task_id not in TASK_IDS:
        raise ValueError(f"unknown task {task_id!r}")
    allowed = {
        "pending",
        "running",
        "complete",
        "blocked",
        "failed",
        "unsupported",
        "scientifically-qualified",
        "numerically-limited",
    }
    if state not in allowed:
        raise ValueError(f"unknown task state {state!r}")
    out = dict(status)
    tasks = {str(key): dict(value) for key, value in dict(out.get("tasks") or {}).items()}
    tasks[task_id] = {"state": state, "reason": reason, "artifact": artifact}
    out["tasks"] = tasks
    out["blocked"] = [
        name for name, row in tasks.items() if row.get("state") in {"blocked", "failed"}
    ]
    return out


def experiment_manifest(
    *,
    source: Mapping[str, object] | None = None,
    git_head: str | None = None,
    dirty_tree: bool | None = None,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "catalog_id": CATALOG_ID,
        "publication_bundle": PUBLICATION_BUNDLE_VERSION,
        "schema_version": 1,
        "objective": (
            "Measure EVLA-C diagonal suitability across available THOL0001 "
            "C-band holography frequencies and hand off an opt-in 3C391 adapter. "
            "A usable limited range is a valid completion."
        ),
        "production_accepted": False,
        "full_jones_is_gate": False,
        "convention_ladder_reopened": False,
        "width_or_squint_fit": False,
        "nearest_plane_substitution": False,
        "residual_jones_policy": RESIDUAL_JONES_POLICY,
        "spw5_role": SPW5_ROLE,
        "coordinate_query": COORDINATE_QUERY,
        "axis_map": AXIS_MAP,
        "primary_arm": PRIMARY_ARM,
        "diagnostic_arm": DIAGNOSTIC_ARM,
        "correlation_order": list(CORRELATION_ORDER),
        "jones_packing": JONES_PACKING,
        "data_column": "CORRECTED_DATA",
        "source_coherency": "per-row MODEL_DATA / resolved 3C147 I, Q, U, V",
        "sampling": {
            "pass_a": "native channel floor(n*32/64); 32 when n=64",
            "pass_b": "native channels floor(n*k/64) for k in 8,24,32,40,56",
            "unsupported_pass_a": (
                "nearest supported channel to the declared centre; "
                "lower index on a tie"
            ),
            "frequency_source": "each MS SPECTRAL_WINDOW CHAN_FREQ",
            "spw_numbers_are_not_global_frequencies": True,
        },
        "classifiers": {
            "main_lobe_accepted_max_residual_power": MAIN_LOBE_ACCEPTED_MAX,
            "mid_beam_qualified_max_residual_power": MID_BEAM_QUALIFIED_MAX,
            "outer": "diagnostic unless new evidence supports a stronger class",
        },
        "taper": {
            "formula": "12.75 + 0.375 * (nu_GHz - 6.0)",
            "valid_hz": list(TAPER_VALID_HZ),
            "outside_range": "record numerically_unverified_taper; do not silently extrapolate",
        },
        "calibration": {
            "reuse_incompatible_tables": False,
            "solve_on_holoraster_movers": False,
            "residual_jones": RESIDUAL_JONES_POLICY,
            "scientific_spw45_ms": str(SCIENTIFIC_SPW45_MS),
            "new_spws_require_new_work_copies": True,
        },
        "paths": {
            "lower_c_ms": str(LOWER_C_MS),
            "upper_c_candidate_ms": str(UPPER_C_CANDIDATE_MS),
            "product_dir": str(PRODUCT_DIR),
            "beam_root": str(BEAM_ROOT),
            "staging_package": str(STAGING_PACKAGE),
            "staging_scripts": str(STAGING_SCRIPTS),
            "preserve": [
                "vla_c_band_beam_validation_v2",
                "evla_c_full_jones_validation_refresh_v1",
            ],
        },
        "task_ids": list(TASK_IDS),
        "git_head": git_head,
        "dirty_tree": dirty_tree,
        "source_snapshot": dict(source or {}),
    }
