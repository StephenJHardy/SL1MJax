"""Resumable EVLA-C diagonal / full-Jones validation refresh driver.

New Bacchus product and beam directories only. Frozen comparison products
and the generic CASSBEAM artifact are not overwritten. SPW 5 stays sealed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.cassbeam_evla_c import evla_c_feedtaper_db
from sl1mjax.cassbeam_highres import HighresCassbeamCatalog, write_development_highres_manifest
from sl1mjax.evla_c_c147_ring import load_field9_residual_jones, run_c147_publication_channels
from sl1mjax.evla_c_holoraster_compare import (
    export_channel_arrays,
    extract_holoraster_channel,
    identity_vs_prior_evla_diagonal,
    predict_primary_arms,
    score_extract,
)
from sl1mjax.evla_c_metrics import copolar_nonregression
from sl1mjax.evla_c_sensitivity import (
    INJECTION_AMPLITUDES,
    INJECTION_PHASES_DEG,
    classify_full_jones,
    inject_offdiagonal_visibility,
    manufactured_nontemplate_leakage,
    mover_improvement_count,
    pooled_crosshand_delta,
)
from sl1mjax.evla_c_validation_refresh import (
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    EVLA_C_MODEL_ID,
    EXISTING_EVLA_C_ROOT,
    EXPERIMENT_ID,
    GENERIC_BEAM_ROOT,
    INJECTION_SEED,
    NINECHAN_BEAM_ROOT,
    PRIMARY_ARMS,
    PRODUCT_DIR,
    PUBLICATION_CHANNELS,
    RESIDUAL_JONES_FREQUENCY_POLICY,
    RESIDUAL_JONES_PATH,
    SCIENTIFIC_MS,
    SCIENTIFIC_ROOT,
    UNBLOCK_DIR,
    default_source_paths,
    empty_status,
    experiment_manifest,
    publication_frequency_hz,
    publication_frequency_mhz,
    refuse_frozen_write,
    sha256_file,
    source_snapshot,
    update_task,
    write_json_atomic,
)
from sl1mjax.holography_diagonal_correction import (
    SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
    refuse_spw5,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_CAL = Path("/media/stephen/astro/vla/extracted/commissioning/products/scientific")


def _load_script(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison():
    return _load_script("run_thol0001_holoraster_cassbeam_comparison.py", "thol0001_holoraster_comparison")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    return value


def _load_status(output_dir: Path) -> dict:
    path = output_dir / "status.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return empty_status()


def _save_status(output_dir: Path, status: dict) -> None:
    refuse_frozen_write(output_dir)
    write_json_atomic(output_dir / "status.json", status)


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def phase0_inventory(output_dir: Path, repo: Path) -> dict:
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    residual_sha = sha256_file(RESIDUAL_JONES_PATH) if RESIDUAL_JONES_PATH.is_file() else None
    existing_plane = (
        EXISTING_EVLA_C_ROOT / "spw4" / "evla-cband-4564-g1024-p32.jones.dat"
    )
    existing_sha = sha256_file(existing_plane) if existing_plane.is_file() else None
    snapshot = source_snapshot(default_source_paths(repo))
    manifest = experiment_manifest(
        source=snapshot,
        residual_jones_sha256=residual_sha,
        existing_evla_plane_sha256=existing_sha,
        git_head=_git_head(repo),
        dirty_tree=True,
    )
    write_json_atomic(output_dir / "experiment_manifest.json", manifest)
    status = _load_status(output_dir)
    status = update_task(
        status, "phase0_inventory", state="complete", artifact="experiment_manifest.json"
    )
    _save_status(output_dir, status)
    return manifest


def _copy_existing_evla_plane(destination: Path) -> None:
    refuse_frozen_write(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reference").mkdir(exist_ok=True)
    (destination / "spw4").mkdir(exist_ok=True)
    for relative in (
        "reference/base.in",
        "reference/vla_geom",
        "spw4/evla-cband-4564-g1024-p32.jones.dat",
        "spw4/evla-cband-4564-g1024-p32.params",
        "spw4/evla-cband-4564-g1024-p32.log",
    ):
        source = EXISTING_EVLA_C_ROOT / relative
        target = destination / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            if sha256_file(source) != sha256_file(target):
                raise RuntimeError(f"refusing to replace different {target}")
            continue
        shutil.copy2(source, target)


def generate_missing_planes(*, output_dir: Path, beam_root: Path, binary: Path) -> dict:
    refuse_frozen_write(beam_root)
    _copy_existing_evla_plane(beam_root)
    missing = []
    for channel in PUBLICATION_CHANNELS:
        mhz = publication_frequency_mhz(channel)
        prefix = beam_root / "spw4" / f"evla-cband-{mhz}-g1024-p32"
        if not prefix.with_suffix(".jones.dat").is_file():
            missing.append(mhz)
    timings = []
    if missing:
        generator = _load_script("generate_cassbeam_highres_model.py", "generate_cassbeam_highres_model")
        started = time.perf_counter()
        generator.generate(
            binary=binary,
            base_input=beam_root / "reference" / "base.in",
            geometry=beam_root / "reference" / "vla_geom",
            output_dir=beam_root,
            frequencies_mhz=missing,
            reference_frequencies_mhz=set(),
            gridsize=1024,
            pixelsperbeam=32,
            name_prefix="evla-cband",
            feedtaper_mode="evla_c",
        )
        timings.append(
            {
                "n_planes": len(missing),
                "elapsed_s": time.perf_counter() - started,
                "frequencies_mhz": missing,
            }
        )
    prior = output_dir / "phase2_planes.json"
    if not missing and prior.is_file() and not timings:
        existing = json.loads(prior.read_text(encoding="utf-8"))
        if int(existing.get("n_planes") or 0) == len(PUBLICATION_CHANNELS):
            print("phase2 planes already present; skipping catalog reload", flush=True)
            return existing
    manifest = write_development_highres_manifest(
        beam_root, model_id=EVLA_C_MODEL_ID, name_prefix="evla-cband"
    )
    catalog = HighresCassbeamCatalog(beam_root, expected_model_id=EVLA_C_MODEL_ID)
    present = []
    for channel in PUBLICATION_CHANNELS:
        hz = publication_frequency_hz(channel)
        print(f"verify plane {channel} {hz/1e6:.3f} MHz", flush=True)
        plane = catalog.plane(hz)
        present.append(
            {
                "channel": int(channel),
                "frequency_hz": float(plane.frequency_hz),
                "frequency_mhz": int(plane.frequency_mhz),
                "feedtaper_db": evla_c_feedtaper_db(hz),
                "data_sha256": plane.data_sha256,
            }
        )
        if int(plane.frequency_mhz) != publication_frequency_mhz(channel):
            raise RuntimeError("catalog returned a different frequency plane")
    report = {
        "model_id": EVLA_C_MODEL_ID,
        "n_planes": len(present),
        "planes": present,
        "generation": timings,
        "catalog_planes": len(manifest["planes"]),
    }
    write_json_atomic(output_dir / "phase2_planes.json", report)
    return report


def _source_i(extract: dict) -> float:
    intensity = np.asarray(extract["intensity"], dtype=np.float64).reshape(-1)
    finite = intensity[np.isfinite(intensity)]
    if finite.size == 0:
        raise ValueError("no finite source I")
    return float(np.median(finite))


def _export_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    required = {"measured", "weight", "source_lm_feed", PRIMARY_ARMS[0], PRIMARY_ARMS[1]}
    try:
        with np.load(path, allow_pickle=False) as handle:
            return required.issubset(set(handle.files))
    except (OSError, ValueError):
        return False


def channel_product_complete(output_dir: Path, channel: int) -> bool:
    export = output_dir / "channels" / f"channel{channel:02d}_export.npz"
    report = output_dir / "channels" / f"channel{channel:02d}_report.json"
    return export.is_file() and report.is_file() and _export_is_complete(export)


def run_channel(
    *,
    channel: int,
    output_dir: Path,
    measurement_set: Path,
    catalog: HighresCassbeamCatalog,
    cmp,
    n_boot: int = BOOTSTRAP_N,
) -> dict:
    refuse_spw5(spectral_window_id=4, opened=False)
    expected = publication_frequency_hz(channel)
    report_path = output_dir / "channels" / f"channel{channel:02d}_report.json"
    if channel_product_complete(output_dir, channel):
        print(f"channel {channel} resume existing export/report", flush=True)
        return json.loads(report_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    print(f"channel {channel} extract start", flush=True)
    extract = extract_holoraster_channel(
        cmp,
        channel=channel,
        measurement_set=measurement_set,
        product_dir=UNBLOCK_DIR,
        names=None,
    )
    extract_s = time.perf_counter() - started
    print(f"channel {channel} extract done n={extract['n_development']} {extract_s:.1f}s", flush=True)
    if abs(float(extract["frequency_hz"]) - expected) > 5.0e3:
        raise RuntimeError(
            f"channel {channel} frequency {extract['frequency_hz']} != {expected}"
        )
    catalog.plane(float(extract["frequency_hz"]))
    pred_started = time.perf_counter()
    print(f"channel {channel} predict start", flush=True)
    predictions = predict_primary_arms(
        catalog=catalog,
        frequencies_hz=np.asarray([extract["frequency_hz"]], dtype=np.float64),
        offset_lm_rad=extract["source_lm"],
        chi_moving=extract["fields"]["chi_m"],
        chi_reference=extract["fields"]["chi_r"],
        moving_id=extract["fields"]["moving"],
        reference_id=extract["fields"]["reference"],
        moving_is_p=extract["fields"]["moving_is_p"],
        residual_jones=extract["residual"],
        source=extract["source"],
    )
    predict_s = time.perf_counter() - pred_started
    print(f"channel {channel} predict done {predict_s:.1f}s", flush=True)
    source_i = _source_i(extract)
    scores = score_extract(extract, predictions, source_i_jy=source_i)
    export_path = output_dir / "channels" / f"channel{channel:02d}_export.npz"
    export_channel_arrays(export_path, extract, predictions)
    names = extract["names"]
    name_to_id = {str(name): index for index, name in enumerate(names)}
    holdout_ids = [name_to_id[name] for name in SPW4_HOLDOUT_MOVING_ANTENNA_NAMES]
    spatial = extract["masks"]["spatial_holdout"]
    mover = extract["masks"]["mover_holdout"]
    main = np.asarray(extract["regions"]["main_lobe"], dtype=bool)
    sensitivity = {
        "spatial": pooled_crosshand_delta(
            extract["measured"][spatial],
            predictions[PRIMARY_ARMS[1]][spatial],
            predictions[PRIMARY_ARMS[0]][spatial],
            extract["weight"][spatial],
            extract["fields"]["rows"][spatial],
            n_boot=n_boot,
            seed=BOOTSTRAP_SEED,
        ),
        "mover": pooled_crosshand_delta(
            extract["measured"][mover],
            predictions[PRIMARY_ARMS[1]][mover],
            predictions[PRIMARY_ARMS[0]][mover],
            extract["weight"][mover],
            extract["fields"]["moving"][mover],
            n_boot=n_boot,
            seed=BOOTSTRAP_SEED + 3,
        ),
        "mover_units": mover_improvement_count(
            extract["measured"][mover],
            predictions[PRIMARY_ARMS[1]][mover],
            predictions[PRIMARY_ARMS[0]][mover],
            extract["weight"][mover],
            extract["fields"]["moving"][mover],
            holdout_ids,
        ),
        "copolar_main_lobe": copolar_nonregression(
            scores["arms"][PRIMARY_ARMS[1]]["train"]["main_lobe"],
            scores["arms"][PRIMARY_ARMS[0]]["train"]["main_lobe"],
        )
        if scores["arms"][PRIMARY_ARMS[1]]["train"]["main_lobe"].get("RR")
        else {"passes": False, "reason": "empty main-lobe train scores"},
    }
    report = {
        "channel": int(channel),
        "frequency_hz": float(extract["frequency_hz"]),
        "n_development": int(extract["n_development"]),
        "n_full_raster_usable": int(extract["n_full_raster_usable"]),
        "extract_s": extract_s,
        "predict_s": predict_s,
        "rows_per_s": float(extract["n_development"] / predict_s) if predict_s > 0 else None,
        "source_i_jy": source_i,
        "scores": scores,
        "sensitivity": sensitivity,
        "export": export_path.as_posix(),
        "residual_jones_native_channel": 32,
        "coordinate_query": "source_lm_feed",
        "main_lobe_n": int(np.sum(main)),
    }
    write_json_atomic(output_dir / "channels" / f"channel{channel:02d}_report.json", _jsonable(report))
    if channel == 32:
        prior = (
            SCIENTIFIC_ROOT
            / "holoraster_coordinate_feed_comparison_v2"
            / "channel32_model_comparison.npz"
        )
        identity = identity_vs_prior_evla_diagonal(export_path, prior)
        write_json_atomic(output_dir / "phase1_identity_channel32.json", _jsonable(identity))
        report["identity_vs_prior_evla_diagonal"] = identity
        write_json_atomic(output_dir / "channels" / f"channel{channel:02d}_report.json", _jsonable(report))
    return report


def run_injections(export_path: Path, output_dir: Path) -> dict:
    payload = np.load(export_path)
    measured = payload["measured"]
    weight = payload["weight"]
    diag = payload[PRIMARY_ARMS[0]]
    full = payload[PRIMARY_ARMS[1]]
    template = payload["full_minus_diagonal"]
    train = payload["train"]
    spatial = payload["spatial_holdout"]
    nontemplate = manufactured_nontemplate_leakage(measured, seed=INJECTION_SEED)
    recovered = diag + template
    algebra_ok = bool(np.allclose(recovered, full, atol=1.0e-12, rtol=0.0))
    grid = []
    unit_spatial = None
    zero_spatial = None
    for amplitude in INJECTION_AMPLITUDES:
        for phase_deg in INJECTION_PHASES_DEG:
            if amplitude == 0.0 and phase_deg != 0.0:
                continue
            injected = inject_offdiagonal_visibility(
                measured,
                template,
                amplitude=float(amplitude),
                phase_rad=float(np.deg2rad(phase_deg)),
            )
            scored = pooled_crosshand_delta(
                injected[spatial],
                full[spatial],
                diag[spatial],
                weight[spatial],
                payload["row_id"][spatial],
                n_boot=200,
                seed=INJECTION_SEED + int(10 * amplitude) + int(phase_deg),
            )
            row = {
                "amplitude": float(amplitude),
                "phase_deg": float(phase_deg),
                "spatial": scored,
            }
            grid.append(row)
            if amplitude == 0.0:
                zero_spatial = scored
            if amplitude == 1.0 and phase_deg == 0.0:
                unit_spatial = scored
    if unit_spatial is None or zero_spatial is None:
        raise RuntimeError("injection grid missing the zero or unit-phase-0 cases")
    report = {
        "algebra_recovers_full": algebra_ok,
        "zero_injection_improves": bool(zero_spatial["pooled_RL_LR"]["improves"]),
        "unit_injection_improves": bool(unit_spatial["pooled_RL_LR"]["improves"]),
        "nontemplate_differs": bool(
            not np.allclose(
                nontemplate,
                inject_offdiagonal_visibility(measured, template, amplitude=1.0),
            )
        ),
        "train_n": int(np.sum(train)),
        "spatial_n": int(np.sum(spatial)),
        "unit_spatial": unit_spatial,
        "zero_spatial": zero_spatial,
        "grid": grid,
        "amplitudes": list(INJECTION_AMPLITUDES),
        "phases_deg": list(INJECTION_PHASES_DEG),
        "residual_jones_frequency_policy": RESIDUAL_JONES_FREQUENCY_POLICY,
    }
    write_json_atomic(output_dir / "phase4_injections.json", _jsonable(report))
    return report


def classify_from_channel32(channel_report: dict, injection: dict) -> dict:
    sensitivity = channel_report["sensitivity"]
    return classify_full_jones(
        spatial=sensitivity["spatial"],
        mover=sensitivity["mover"],
        mover_units=sensitivity["mover_units"],
        copolar=sensitivity["copolar_main_lobe"],
        injection_detects_unit=bool(injection.get("unit_injection_improves")),
        injection_zero_is_null=not bool(injection.get("zero_injection_improves")),
        numerical_ok=bool(injection.get("algebra_recovers_full")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--beam-root", type=Path, default=NINECHAN_BEAM_ROOT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--cassbeam-binary", type=Path, default=Path("/usr/bin/cassbeam"))
    parser.add_argument(
        "--stage",
        choices=(
            "phase0",
            "phase2",
            "channel32",
            "frequency",
            "phase4",
            "phase5",
            "all",
        ),
        default="phase0",
    )
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_N)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    refuse_frozen_write(output_dir)
    refuse_frozen_write(arguments.beam_root)
    if arguments.beam_root.resolve() == GENERIC_BEAM_ROOT.resolve():
        raise RuntimeError("refusing to write into the frozen generic artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    refuse_spw5(spectral_window_id=4, opened=False)
    try:
        if arguments.stage in {"phase0", "all", "phase2", "channel32", "frequency", "phase4", "phase5"}:
            phase0_inventory(output_dir, arguments.repo_root)
        status = _load_status(output_dir)
        if arguments.stage in {"phase2", "all"}:
            status = update_task(status, "phase2_generate_nine_planes", state="running")
            _save_status(output_dir, status)
            generate_missing_planes(
                output_dir=output_dir,
                beam_root=arguments.beam_root,
                binary=arguments.cassbeam_binary,
            )
            status = update_task(
                status,
                "phase2_generate_nine_planes",
                state="complete",
                artifact="phase2_planes.json",
            )
            _save_status(output_dir, status)
        if arguments.stage in {"channel32", "frequency", "phase4", "all"}:
            catalog = HighresCassbeamCatalog(
                arguments.beam_root, expected_model_id=EVLA_C_MODEL_ID
            )
            cmp = _comparison()
            channels = (32,) if arguments.stage == "channel32" else PUBLICATION_CHANNELS
            if arguments.stage == "phase4":
                channels = (32,)
            first_batch = None
            for channel in channels:
                task = "phase3_score_channel32" if channel == 32 else f"phase3_score_ch{channel}"
                status = update_task(status, task, state="running")
                _save_status(output_dir, status)
                print(f"channel {channel} start", flush=True)
                report = run_channel(
                    channel=channel,
                    output_dir=output_dir,
                    measurement_set=arguments.measurement_set,
                    catalog=catalog,
                    cmp=cmp,
                    n_boot=arguments.n_boot,
                )
                if first_batch is None:
                    first_batch = {
                        "channel": channel,
                        "extract_s": report["extract_s"],
                        "predict_s": report["predict_s"],
                        "n": report["n_development"],
                        "estimated_remaining_s": report["predict_s"] * (len(channels) - 1),
                    }
                    write_json_atomic(output_dir / "first_batch_timing.json", first_batch)
                    print(json.dumps(first_batch), flush=True)
                status = update_task(
                    status,
                    task,
                    state="complete",
                    artifact=f"channels/channel{channel:02d}_report.json",
                )
                _save_status(output_dir, status)
        if arguments.stage in {"phase4", "all"}:
            export = output_dir / "channels" / "channel32_export.npz"
            injection = run_injections(export, output_dir)
            channel32 = json.loads(
                (output_dir / "channels" / "channel32_report.json").read_text(encoding="utf-8")
            )
            classification = classify_from_channel32(channel32, injection)
            write_json_atomic(output_dir / "phase4_classification.json", classification)
            status = update_task(
                status, "phase4_injections", state="complete", artifact="phase4_injections.json"
            )
            status = update_task(
                status, "phase4_classify", state="complete", artifact="phase4_classification.json"
            )
            _save_status(output_dir, status)
        if arguments.stage in {"phase5", "all"}:
            print("phase5: open EVLA-C catalog", flush=True)
            catalog = HighresCassbeamCatalog(
                arguments.beam_root, expected_model_id=EVLA_C_MODEL_ID
            )
            print("phase5: load residual Jones", flush=True)
            residual = load_field9_residual_jones()
            status = update_task(status, "phase5_c147_ring", state="running")
            _save_status(output_dir, status)
            print("phase5: start C147 nine-frequency ring", flush=True)
            ring = run_c147_publication_channels(
                measurement_set=arguments.measurement_set,
                cal_root=SCIENTIFIC_CAL,
                catalog=catalog,
                residual=residual,
                output_dir=output_dir / "c147_ring",
            )
            write_json_atomic(output_dir / "c147_ring" / "c147_report.json", _jsonable(ring))
            status = update_task(
                status, "phase5_c147_ring", state="complete", artifact="c147_ring/c147_report.json"
            )
            _save_status(output_dir, status)
        print(json.dumps({"experiment": EXPERIMENT_ID, "stage": arguments.stage, "ok": True}))
        return 0
    except Exception as exc:
        status = _load_status(output_dir)
        status = update_task(
            status,
            "phase7_verify",
            state="failed",
            reason=str(exc),
        )
        _save_status(output_dir, status)
        write_json_atomic(
            output_dir / "failure.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
