"""EVLA-C diagonal and full-Jones validation refresh specification.

This module freezes paths, frequencies, residual-Jones frequency policy,
and resumable status I/O. It does not promote a production beam.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

EXPERIMENT_ID = "evla_c_full_jones_validation_refresh_v1"
PUBLICATION_BUNDLE_VERSION = "vla_c_band_beam_validation_v2"
SPECTRAL_WINDOW_ID = 4
HOLORASTER_FIELD_ID = 10
C147_OFFSET_FIELD_IDS = (1, 2, 3, 4, 5, 6, 7, 8)
PUBLICATION_CHANNELS = (0, 8, 16, 24, 32, 40, 48, 56, 63)
ALGEBRA_ATOL = 1.0e-12
ALGEBRA_RTOL = 0.0
BOOTSTRAP_N = 400
BOOTSTRAP_SEED = 0
INJECTION_SEED = 1
MAIN_LOBE_NONREGRESSION_ABS = 0.002
MAIN_LOBE_NONREGRESSION_FRAC = 0.10
CORRELATION_ORDER = ("RR", "RL", "LR", "LL")
JONES_PACKING = "[[RR, RL], [LR, LL]]"
COORDINATE_QUERY = "source_lm_feed"
AXIS_MAP = "az_to_plus_l_el_to_plus_m_no_swap"
PRIMARY_ARMS = ("evla_c_source_diagonal", "evla_c_source_full_jones")
CONTROL_ARMS = ("generic_commanded_diagonal", "generic_source_diagonal")
RESIDUAL_JONES_NATIVE_CHANNEL = 32
RESIDUAL_JONES_FREQUENCY_POLICY = (
    "explicit_channel32_field9_plane_applied_to_all_publication_channels"
)

SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/"
    "THOL0001.lowerC.spw45.scientific.ms"
)
SCIENTIFIC_ROOT = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific"
)
PRODUCT_DIR = SCIENTIFIC_ROOT / EXPERIMENT_ID
UNBLOCK_DIR = SCIENTIFIC_ROOT / "full_jones_unblock"
RESIDUAL_JONES_PATH = UNBLOCK_DIR / "field9_stabilized_residual_jones.json"
GENERIC_BEAM_ROOT = Path(
    "/media/stephen/astro/vla/beam_models/cassbeam_cband_full_jones_g1024_p32_20260906"
)
EXISTING_EVLA_C_ROOT = Path(
    "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_20260908"
)
NINECHAN_BEAM_ROOT = Path(
    "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_ninepub_20260908"
)
CONVERGENCE_BEAM_ROOT = Path(
    "/media/stephen/astro/vla/beam_models/cassbeam_evla_c_convergence_20260908"
)
EVLA_C_MODEL_ID = "cassbeam_evla_c_g1024_p32_spw4_ninepub"
STAGING_PACKAGE = Path("/tmp/sl1mjax-evla-c-refresh")
STAGING_SCRIPTS = Path("/tmp/sl1mjax-evla-c-refresh-scripts")
LOCAL_MANIFEST_DIR = Path("docs/evla_c_full_jones_validation_refresh")

FROZEN_WRITE_NAMES = frozenset(
    {
        "holoraster_cassbeam_comparison",
        "holoraster_cassbeam_correction",
        "holoraster_physical_squint_width_v1",
        "holoraster_spatial_convention_v1",
        "holoraster_coordinate_feed_comparison_v1",
        "c147_offset_ring",
        "vla_c_band_beam_validation_v1",
        "cassbeam_cband_full_jones_g1024_p32_20260906",
        "cassbeam_evla_c_g1024_p32_20260908",
    }
)

PUBLICATION_CHANNEL_HZ = {
    0: 4_500_000_000.0,
    8: 4_516_000_000.0,
    16: 4_532_000_000.0,
    24: 4_548_000_000.0,
    32: 4_564_000_000.0,
    40: 4_580_000_000.0,
    48: 4_596_000_000.0,
    56: 4_612_000_000.0,
    63: 4_626_000_000.0,
}
PUBLICATION_CHANNEL_MHZ = {
    channel: int(round(hz / 1.0e6)) for channel, hz in PUBLICATION_CHANNEL_HZ.items()
}

TASK_IDS = (
    "phase0_inventory",
    "phase1_independent_rime",
    "phase1_identity_channel32",
    "phase1_calibration_fixtures",
    "phase2_generate_nine_planes",
    "phase2_convergence",
    "phase3_extract_channel32",
    "phase3_score_channel32",
    *(f"phase3_extract_ch{channel}" for channel in PUBLICATION_CHANNELS if channel != 32),
    *(f"phase3_score_ch{channel}" for channel in PUBLICATION_CHANNELS if channel != 32),
    "phase4_injections",
    "phase4_classify",
    "phase5_c147_ring",
    "phase6_bundle",
    "phase6_notebook",
    "phase7_verify",
)

OUTCOMES = (
    "supported_on_spw4_development",
    "disfavoured_on_tested_support",
    "inconclusive_sensitivity",
    "blocked_implementation_or_contract",
)


def refuse_frozen_write(path: Path) -> None:
    resolved = Path(path).resolve()
    parts = set(resolved.parts)
    if parts & FROZEN_WRITE_NAMES:
        raise RuntimeError(f"refusing to write into a frozen product: {resolved}")
    if resolved == GENERIC_BEAM_ROOT.resolve():
        raise RuntimeError("refusing to write into the frozen generic CASSBEAM artifact")
    if resolved == EXISTING_EVLA_C_ROOT.resolve():
        raise RuntimeError("refusing to overwrite the existing EVLA-C channel-32 artifact")


def publication_frequency_hz(channel: int) -> float:
    if int(channel) not in PUBLICATION_CHANNEL_HZ:
        raise ValueError(f"channel {channel} is not in the publication set")
    return float(PUBLICATION_CHANNEL_HZ[int(channel)])


def publication_frequency_mhz(channel: int) -> int:
    return int(PUBLICATION_CHANNEL_MHZ[int(channel)])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        "evla_c_validation_refresh.py",
        "evla_c_independent_rime.py",
        "evla_c_metrics.py",
        "evla_c_sensitivity.py",
        "evla_c_holoraster_compare.py",
        "evla_c_c147_ring.py",
        "cassbeam_evla_c.py",
        "cassbeam_highres.py",
        "holography_beam_prior.py",
        "holography_holoraster_coordinates.py",
        "holography_c147_offset_ring.py",
        "holography_highres_cassbeam.py",
        "holography_full_jones.py",
        "polarization.py",
        "beam_validation_outputs.py",
        "beam_validation_statistics.py",
        "beam_validation_plots.py",
        "beam_validation_claims.py",
    )
    files = [root / "src" / "sl1mjax" / name for name in names]
    files.append(root / "scripts" / "run_thol0001_evla_c_validation_refresh.py")
    files.append(root / "scripts" / "generate_cassbeam_highres_model.py")
    files.append(root / "docs" / "evla-c-full-jones-validation-refresh-plan.md")
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


def load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def empty_status() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": 1,
        "production_accepted": False,
        "spw5_opened": False,
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
    if state not in {"pending", "running", "complete", "blocked", "failed"}:
        raise ValueError(f"unknown task state {state!r}")
    out = dict(status)
    tasks = {str(key): dict(value) for key, value in dict(out.get("tasks") or {}).items()}
    tasks[task_id] = {"state": state, "reason": reason, "artifact": artifact}
    out["tasks"] = tasks
    blocked = [
        name
        for name, row in tasks.items()
        if row.get("state") in {"blocked", "failed"}
    ]
    out["blocked"] = blocked
    return out


def experiment_manifest(
    *,
    source: Mapping[str, object] | None = None,
    residual_jones_sha256: str | None = None,
    existing_evla_plane_sha256: str | None = None,
    git_head: str | None = None,
    dirty_tree: bool | None = None,
) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": 1,
        "objective": (
            "Answer implementation correctness, EVLA-C diagonal description, "
            "and full-Jones cross-hand sensitivity separately. A negative or "
            "inconclusive result is a valid completion."
        ),
        "production_accepted": False,
        "spw5_opened": False,
        "convention_ladder_reopened": False,
        "width_or_squint_fit": False,
        "spectral_window_id": SPECTRAL_WINDOW_ID,
        "publication_channels": list(PUBLICATION_CHANNELS),
        "publication_channel_hz": {
            str(channel): hz for channel, hz in PUBLICATION_CHANNEL_HZ.items()
        },
        "publication_channel_mhz": {
            str(channel): mhz for channel, mhz in PUBLICATION_CHANNEL_MHZ.items()
        },
        "frequency_source": "MS SPECTRAL_WINDOW CHAN_FREQ row 4",
        "nearest_plane_substitution": False,
        "correlation_order": list(CORRELATION_ORDER),
        "jones_packing": JONES_PACKING,
        "units": {
            "visibility": "Jy",
            "angle": "radian",
            "frequency": "Hz",
        },
        "data_column": "CORRECTED_DATA",
        "source_coherency": "per-row MODEL_DATA / resolved 3C147 I, Q, U, V",
        "coordinate_query": COORDINATE_QUERY,
        "axis_map": AXIS_MAP,
        "beam_normalization": "inv(E(0)) @ E(s)",
        "primary_arms": list(PRIMARY_ARMS),
        "control_arms": list(CONTROL_ARMS),
        "holoraster_field_id": HOLORASTER_FIELD_ID,
        "c147_offset_field_ids": list(C147_OFFSET_FIELD_IDS),
        "c147_uses_holoraster_negative_offset": False,
        "residual_jones": {
            "path": RESIDUAL_JONES_PATH.as_posix(),
            "native_channel": RESIDUAL_JONES_NATIVE_CHANNEL,
            "frequency_policy": RESIDUAL_JONES_FREQUENCY_POLICY,
            "sha256": residual_jones_sha256,
            "silent_extrapolation": False,
        },
        "algebra_atol": ALGEBRA_ATOL,
        "algebra_rtol": ALGEBRA_RTOL,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "injection_seed": INJECTION_SEED,
        "paired_loss": "candidate_minus_diagonal_negative_better",
        "main_lobe_nonregression": {
            "abs": MAIN_LOBE_NONREGRESSION_ABS,
            "frac": MAIN_LOBE_NONREGRESSION_FRAC,
            "formula": "max(0.002, 0.10 * L_main_diagonal)",
        },
        "outcomes": list(OUTCOMES),
        "paths": {
            "measurement_set": SCIENTIFIC_MS.as_posix(),
            "product_dir": PRODUCT_DIR.as_posix(),
            "unblock": UNBLOCK_DIR.as_posix(),
            "generic_beam": GENERIC_BEAM_ROOT.as_posix(),
            "existing_evla_c": EXISTING_EVLA_C_ROOT.as_posix(),
            "ninechan_beam": NINECHAN_BEAM_ROOT.as_posix(),
            "convergence_beam": CONVERGENCE_BEAM_ROOT.as_posix(),
            "staging_package": STAGING_PACKAGE.as_posix(),
            "staging_scripts": STAGING_SCRIPTS.as_posix(),
            "publication_bundle": PUBLICATION_BUNDLE_VERSION,
        },
        "existing_evla_c_channel32_sha256": existing_evla_plane_sha256,
        "read_only_controls": [
            "holoraster_cassbeam_comparison",
            "holoraster_coordinate_feed_comparison_v1",
            "holoraster_physical_squint_width_v1",
            "vla_c_band_beam_validation_v1",
        ],
        "task_ids": list(TASK_IDS),
        "git_head": git_head,
        "dirty_tree": dirty_tree,
        "source_snapshot": dict(source or {}),
        "development_only": True,
    }
