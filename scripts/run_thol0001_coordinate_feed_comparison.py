"""Compare commanded vs source-in-beam coordinates and generic vs EVLA-C feed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.cassbeam_highres import (
    DEFAULT_HIGHRES_ROOT,
    EVLA_C_HIGHRES_MODEL_ID,
    HighresCassbeamCatalog,
    write_development_highres_manifest,
)
from sl1mjax.holography import THOL0001_SPW4_CHANNEL_32_HZ
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_beam_prior import evaluate_holoraster_cassbeam
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_cassbeam_correction import (
    CorrectionSamples,
    catalog_diagonal_feed_lookup,
    region_masks_from_voltage,
)
from sl1mjax.holography_cassbeam_holoraster_report import locked_convention
from sl1mjax.holography_coordinate_feed_comparison import (
    COORDINATE_FEED_EXPERIMENT,
    EVLA_SOURCE,
    GENERIC_COMMANDED,
    GENERIC_SOURCE,
    interpret_coordinate_feed_scores,
    plane_hand_centroids_arcmin,
    query_coordinates,
    score_candidate_against_baseline,
)
from sl1mjax.holography_diagonal_correction import (
    HOLORASTER_FIELD_ID,
    refuse_c147_training,
    refuse_spw5,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates
from sl1mjax.holography_physical_squint import (
    PhysicalBeamState,
    physical_path_stages,
    require_physical_identity_matches_comparison,
)
from sl1mjax.holography_physical_squint_experiment import (
    WIDTH_GRID,
    fit_width_on_train,
    measure_direct_squint,
)

COMPARISON_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_coordinate_feed_comparison_v1"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
EVLA_ROOT = Path("/media/stephen/astro/vla/beam_models/cassbeam_evla_c_g1024_p32_20260908")
FROZEN_NAMES = (
    "holoraster_cassbeam_comparison",
    "holoraster_cassbeam_correction",
    "holoraster_physical_squint_width_v1",
    "holoraster_spatial_convention_v1",
    "c147_offset_ring",
    "vla_c_band_beam_validation_v1",
)


def _comparison():
    path = Path(__file__).with_name("run_thol0001_holoraster_cassbeam_comparison.py")
    spec = importlib.util.spec_from_file_location("thol0001_holoraster_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _squeeze_vis(vis):
    plane = np.asarray(vis)
    if plane.ndim == 4:
        if plane.shape[1] != 1:
            raise ValueError("coordinate-feed predictions must stay on one native channel")
        plane = plane[:, 0]
    return plane


def _diagonal_hands(values):
    """Pack RR and LL only for the compact publication export."""

    plane = _squeeze_vis(values)
    if plane.ndim != 3 or plane.shape[1:] != (2, 2):
        raise ValueError("visibility export must have shape (row, 2, 2)")
    return np.stack((plane[:, 0, 0], plane[:, 1, 1]), axis=1)


def _write_publication_export(
    path: Path,
    *,
    samples: CorrectionSamples,
    predictions: dict[str, np.ndarray],
    source_lm: np.ndarray,
    rr_ok: np.ndarray,
    ll_ok: np.ndarray,
) -> Path:
    """Write the three-model SPW-4 rows needed by the laptop notebook.

    This is a development-set comparison. It contains no MS path and no
    calibration tables. Complex values use 32-bit storage only after every
    scientific score has been computed at native precision.
    """

    required = (GENERIC_COMMANDED, GENERIC_SOURCE, EVLA_SOURCE)
    missing = [name for name in required if name not in predictions]
    if missing:
        raise RuntimeError(f"publication export requires all three models: {missing}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        measured_rr_ll=_diagonal_hands(samples.measured).astype(np.complex64),
        generic_commanded_rr_ll=_diagonal_hands(predictions[GENERIC_COMMANDED]).astype(
            np.complex64
        ),
        generic_source_lm_rr_ll=_diagonal_hands(predictions[GENERIC_SOURCE]).astype(
            np.complex64
        ),
        evla_c_source_lm_rr_ll=_diagonal_hands(predictions[EVLA_SOURCE]).astype(
            np.complex64
        ),
        weight_rr_ll=_diagonal_hands(samples.weight).real.astype(np.float32),
        commanded_offset_azelgeo=np.asarray(samples.offset_lm_rad, dtype=np.float64),
        source_lm_feed=np.asarray(source_lm, dtype=np.float64),
        moving_id=np.asarray(samples.moving_id, dtype=np.int16),
        reference_id=np.asarray(samples.reference_id, dtype=np.int16),
        train=np.asarray(samples.train, dtype=bool),
        spatial_holdout=np.asarray(samples.spatial_holdout, dtype=bool),
        mover_holdout=np.asarray(samples.mover_holdout, dtype=bool),
        reference_holdout=np.asarray(samples.reference_holdout, dtype=bool),
        main_lobe=np.asarray(samples.main_lobe, dtype=bool),
        mid=np.asarray(samples.mid, dtype=bool),
        outer_diagnostic=np.asarray(samples.outer, dtype=bool),
        rr_valid=np.asarray(rr_ok, dtype=bool),
        ll_valid=np.asarray(ll_ok, dtype=bool),
        frequency_hz=np.asarray(samples.frequency_hz, dtype=np.float64),
        source_i_jy=np.asarray(8.028518676757812, dtype=np.float64),
    )
    return destination


def _maybe_evla_catalog(root: Path) -> HighresCassbeamCatalog | None:
    manifest = root / "manifest.json"
    if not root.is_dir():
        return None
    if not manifest.is_file():
        try:
            write_development_highres_manifest(
                root, model_id=EVLA_C_HIGHRES_MODEL_ID, name_prefix="evla-cband"
            )
        except FileNotFoundError:
            return None
    return HighresCassbeamCatalog(root, expected_model_id=EVLA_C_HIGHRES_MODEL_ID)


def _readme(payload: dict) -> str:
    interp = payload.get("interpretation") or {}
    return (
        "\n".join(
            [
                "# SPW-4 coordinate and EVLA-C feed comparison",
                "",
                "This is **SPW-4 development**. SPW 5 stayed sealed.",
                "The 128-member convention ladder stayed closed.",
                "",
                f"Source-in-beam beats commanded: **{interp.get('source_lm_feed_beats_commanded')}**.",
                f"EVLA-C compared: **{interp.get('evla_c_compared')}**.",
                "",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--evla-root", type=Path, default=EVLA_ROOT)
    parser.add_argument("--identity-max-rows", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-width", action="store_true")
    arguments = parser.parse_args()
    print("coordinate_feed start", flush=True)
    refuse_spw5(spectral_window_id=4, opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES or output_dir == arguments.comparison_dir.resolve():
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    if "cassbeam_cband_full_jones_g1024_p32_20260906" in output_dir.as_posix():
        raise RuntimeError("refusing to write into the frozen generic CASSBEAM artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmp = _comparison()
    payload: dict[str, object] = {
        "artifact": COORDINATE_FEED_EXPERIMENT,
        "development_only": True,
        "spw5_closed": True,
        "command": list(sys.argv),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        payload["software_gates"] = run_software_gates(plane)
        payload["artifact_checksum"] = artifact_checksum_report(
            catalog, THOL0001_SPW4_CHANNEL_32_HZ
        )
        residual = cmp._residual_jones(arguments.product_dir)
        diag = cmp._diag()
        from sl1mjax.holography_ms import _read_antennas, _tables

        tables = _tables()
        _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = tuple(names)
        observation, chi, packed, _intensity, voltage, rr_ok, ll_ok = cmp._load_holoraster_channel(
            diag,
            tables,
            arguments.measurement_set,
            positions,
            channel=32,
        )
        frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64).reshape(-1)
        refuse_c147_training(observation.block.field_id)
        if np.any(np.asarray(observation.block.field_id) != HOLORASTER_FIELD_ID):
            raise ValueError("experiment samples must be HOLORASTER field 10")
        geometry = moving_reference_row_geometry(observation, require_all_channels=False)
        pair = holoraster_pair_masks(observation)
        usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
            pair["moving_reference"], dtype=bool
        )
        holdouts = spw4_correction_holdouts(observation, names)
        development = (
            holdouts.train
            | holdouts.spatial_holdout
            | holdouts.mover_holdout
            | holdouts.reference_holdout
        )
        keep = usable & development
        fields = cmp._row_fields(geometry, pair, chi, np.flatnonzero(keep))
        commanded = np.asarray(geometry["commanded_offset_azelgeo"], dtype=np.float64)[
            fields["rows"]
        ]
        source_lm = np.asarray(geometry["source_lm_feed"], dtype=np.float64)[fields["rows"]]
        np.testing.assert_allclose(commanded, fields["offset"])
        np.testing.assert_allclose(source_lm, query_coordinates(commanded, query="source_lm_feed"))
        source = np.asarray(observation.source_coherency_visibility, dtype=np.complex128)
        if source.shape == (2, 2):
            source_rows = source
        elif source.ndim == 3:
            source_rows = source[fields["rows"], None, :, :]
        else:
            source_rows = source[fields["rows"]]
        measured = np.asarray(packed, dtype=np.complex128)
        measured = measured[fields["rows"], 0] if measured.ndim == 4 else measured[fields["rows"]]
        weight = cmp._hand_weight(observation, keep)
        if weight.ndim == 4:
            weight = weight[:, 0]
        regions = region_masks_from_voltage(np.asarray(voltage, dtype=np.float64)[fields["rows"]])
        rr_flags = np.asarray(rr_ok, dtype=bool)[fields["rows"]]
        ll_flags = np.asarray(ll_ok, dtype=bool)[fields["rows"]]
        identity_rows = np.arange(commanded.shape[0])
        if int(arguments.identity_max_rows) > 0:
            mover_p = np.flatnonzero(fields["moving_is_p"])
            mover_q = np.flatnonzero(~fields["moving_is_p"])
            half = max(int(arguments.identity_max_rows) // 2, 1)
            identity_rows = np.unique(np.concatenate([mover_p[:half], mover_q[:half]]))
        print("identity gate rows", int(identity_rows.size), flush=True)
        source_id = (
            source_rows
            if np.asarray(source_rows).ndim == 2
            else np.asarray(source_rows)[identity_rows]
        )
        frozen = evaluate_holoraster_cassbeam(
            catalog=catalog,
            frequencies_hz=frequencies,
            convention=locked_convention(),
            offset_lm_rad=commanded[identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            residual_jones=residual,
            source=source_id,
            off_diagonal=False,
        )
        generic_lookup = catalog_diagonal_feed_lookup(catalog, float(frequencies[0]))
        identity_pred = physical_path_stages(
            commanded[identity_rows],
            generic_lookup,
            PhysicalBeamState(),
            residual_jones=residual,
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            source=source_id,
        )
        identity_report = require_physical_identity_matches_comparison(identity_pred, frozen)
        print(json.dumps(cmp._jsonable(identity_report)), flush=True)
        if identity_rows.size == commanded.shape[0]:
            commanded_vis = identity_pred["visibility"]
        else:
            commanded_vis = _squeeze_vis(
                evaluate_holoraster_cassbeam(
                    catalog=catalog,
                    frequencies_hz=frequencies,
                    convention=locked_convention(),
                    offset_lm_rad=commanded,
                    chi_moving=fields["chi_m"],
                    chi_reference=fields["chi_r"],
                    moving_id=fields["moving"],
                    reference_id=fields["reference"],
                    moving_is_p=fields["moving_is_p"],
                    residual_jones=residual,
                    source=source_rows,
                    off_diagonal=False,
                ).visibility
            )
        print("predict generic source_lm_feed", flush=True)
        source_vis = _squeeze_vis(
            evaluate_holoraster_cassbeam(
                catalog=catalog,
                frequencies_hz=frequencies,
                convention=locked_convention(),
                offset_lm_rad=source_lm,
                chi_moving=fields["chi_m"],
                chi_reference=fields["chi_r"],
                moving_id=fields["moving"],
                reference_id=fields["reference"],
                moving_is_p=fields["moving_is_p"],
                residual_jones=residual,
                source=source_rows,
                off_diagonal=False,
            ).visibility
        )
        samples = CorrectionSamples(
            offset_lm_rad=commanded,
            measured=measured,
            baseline=commanded_vis,
            weight=weight,
            source=source_rows,
            moving_is_p=fields["moving_is_p"],
            moving_id=fields["moving"],
            reference_id=fields["reference"],
            residual_jones=residual,
            parallactic_angle_rad=fields["chi_m"],
            reference_parallactic_angle_rad=fields["chi_r"],
            antenna_names=names,
            train=holdouts.train[keep],
            spatial_holdout=holdouts.spatial_holdout[keep],
            mover_holdout=holdouts.mover_holdout[keep],
            main_lobe=regions["main_lobe"],
            mid=regions["mid"],
            outer=regions["outer_diagnostic"],
            field_id=np.asarray(observation.block.field_id, dtype=np.int32)[keep],
            frequency_hz=float(frequencies[0]),
            spectral_window_id=4,
        )
        object.__setattr__(samples, "reference_holdout", holdouts.reference_holdout[keep])
        predictions = {
            GENERIC_COMMANDED: commanded_vis,
            GENERIC_SOURCE: source_vis,
        }
        evla_catalog = _maybe_evla_catalog(arguments.evla_root)
        evla_plane_record = None
        if evla_catalog is not None:
            print("predict EVLA-C source_lm_feed", flush=True)
            evla_plane = evla_catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
            evla_plane_record = plane_hand_centroids_arcmin(
                evla_plane, frequency_hz=float(frequencies[0]), series="evla_c_plane"
            )
            predictions[EVLA_SOURCE] = _squeeze_vis(
                evaluate_holoraster_cassbeam(
                    catalog=evla_catalog,
                    frequencies_hz=frequencies,
                    convention=locked_convention(),
                    offset_lm_rad=source_lm,
                    chi_moving=fields["chi_m"],
                    chi_reference=fields["chi_r"],
                    moving_id=fields["moving"],
                    reference_id=fields["reference"],
                    moving_is_p=fields["moving_is_p"],
                    residual_jones=residual,
                    source=source_rows,
                    off_diagonal=False,
                ).visibility
            )
        score_kw = {
            "rr_ok": rr_flags,
            "ll_ok": ll_flags,
            "n_boot": int(arguments.n_boot),
            "seed": int(arguments.seed),
        }
        paired = {
            "generic_source_lm_vs_generic_commanded": score_candidate_against_baseline(
                samples, predictions[GENERIC_SOURCE], predictions[GENERIC_COMMANDED], **score_kw
            ),
        }
        if EVLA_SOURCE in predictions:
            paired["evla_c_source_lm_vs_generic_commanded"] = score_candidate_against_baseline(
                samples, predictions[EVLA_SOURCE], predictions[GENERIC_COMMANDED], **score_kw
            )
            paired["evla_c_source_lm_vs_generic_source_lm"] = score_candidate_against_baseline(
                samples, predictions[EVLA_SOURCE], predictions[GENERIC_SOURCE], **score_kw
            )
        print("measure map-domain squint in both frames", flush=True)
        squint_kw = {
            "measured": samples.measured,
            "source": samples.source,
            "residual_jones": residual,
            "moving_id": samples.moving_id,
            "reference_id": samples.reference_id,
            "moving_is_p": samples.moving_is_p,
            "chi_moving": samples.parallactic_angle_rad,
            "chi_reference": samples.reference_parallactic_angle_rad,
            "weight": samples.weight,
            "rr_ok": rr_flags,
            "ll_ok": ll_flags,
            "train": samples.train,
            "antenna_names": names,
            "frequency_hz": float(frequencies[0]),
            "n_boot": int(arguments.n_boot),
            "seed": int(arguments.seed),
        }
        map_squint = {
            "commanded_labels": measure_direct_squint(offset_lm_rad=commanded, **squint_kw),
            "source_lm_labels": measure_direct_squint(offset_lm_rad=source_lm, **squint_kw),
        }
        width = None
        if not bool(arguments.skip_width):
            print("width grids at native delta", flush=True)
            width = {
                GENERIC_COMMANDED: fit_width_on_train(
                    samples,
                    generic_lookup,
                    delta_lm_rad=PhysicalBeamState().delta(),
                    grid=WIDTH_GRID,
                ),
            }
            # Width about native centres after the source-in-beam query.
            object.__setattr__(samples, "offset_lm_rad", source_lm)
            width[GENERIC_SOURCE] = fit_width_on_train(
                samples,
                generic_lookup,
                delta_lm_rad=PhysicalBeamState().delta(),
                grid=WIDTH_GRID,
            )
            object.__setattr__(samples, "offset_lm_rad", commanded)
        interpretation = interpret_coordinate_feed_scores(
            paired, evla_present=EVLA_SOURCE in predictions
        )
        publication_export = _write_publication_export(
            output_dir / "channel32_model_comparison.npz",
            samples=samples,
            predictions=predictions,
            source_lm=source_lm,
            rr_ok=rr_flags,
            ll_ok=ll_flags,
        )
        payload.update(
            {
                "identity_report": identity_report,
                "generic_plane_centroids": plane_hand_centroids_arcmin(
                    plane, frequency_hz=float(frequencies[0]), series="generic_vla_plane"
                ),
                "evla_plane_centroids": evla_plane_record,
                "map_squint": map_squint,
                "width_grids": width,
                "paired_scores": paired,
                "interpretation": interpretation,
                "publication_export": {
                    "filename": publication_export.name,
                    "n_rows": int(samples.train.size),
                    "models": list(predictions),
                    "coordinate": "source_lm_feed",
                    "scope": "SPW-4 frozen development rows",
                },
                "row_accounting": {
                    "n_used": int(samples.train.size),
                    "n_train": int(np.sum(samples.train)),
                    "n_spatial_holdout": int(np.sum(samples.spatial_holdout)),
                    "n_mover_holdout": int(np.sum(samples.mover_holdout)),
                    "label": "SPW-4 development; holdouts are not an unseen validation set",
                },
            }
        )
        write_json(cmp._jsonable(payload), output_dir / "report.json")
        write_json(cmp._jsonable(paired), output_dir / "paired_scores.json")
        write_json(cmp._jsonable(interpretation), output_dir / "interpretation.json")
        write_json(cmp._jsonable(map_squint), output_dir / "map_squint.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(json.dumps(cmp._jsonable(interpretation)), flush=True)
        return 0
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        write_json(cmp._jsonable(payload), output_dir / "report.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(payload["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
