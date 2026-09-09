"""SPW-4 discrete diagonal spatial-convention test. SPW 5 stays sealed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.holography import THOL0001_SPW4_CHANNEL_32_HZ
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_beam_prior import evaluate_holoraster_cassbeam
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    CorrectionSamples,
    catalog_diagonal_feed_lookup,
    region_masks_from_voltage,
)
from sl1mjax.holography_cassbeam_holoraster_report import locked_convention
from sl1mjax.holography_diagonal_correction import (
    HOLORASTER_FIELD_ID,
    complex_visibility_loss,
    packed_offset_keys,
    refuse_c147_training,
    refuse_spw5,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates
from sl1mjax.holography_physical_squint import (
    PhysicalBeamState,
    native_separation_rad,
    physical_path_stages,
    require_physical_identity_matches_comparison,
)
from sl1mjax.holography_physical_squint_experiment import (
    candidate_state,
    hand_loss,
    rr_ll_difference_loss,
    score_model,
)
from sl1mjax.holography_spatial_convention import (
    FROZEN_WIDTH,
    IDENTITY_TRANSFORM,
    MEASURED_INDEPENDENT_DELTA_ARCMIN,
    SPATIAL_CONVENTION_EXPERIMENT,
    SpatialTransform,
    select_spatial_transform,
    spatial_transform_catalog,
    transformed_path_stages,
    transformed_separation_rad,
    vector_alignment,
)

COMPARISON_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_spatial_convention_v1"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
FROZEN_NAMES = (
    "holoraster_cassbeam_comparison",
    "holoraster_cassbeam_correction",
    "holoraster_physical_squint_width_v1",
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


def _predict_transform(samples, lookup, transform: SpatialTransform, width: float, mask):
    choose = np.asarray(mask, dtype=bool)
    source = samples.source
    if np.asarray(source).ndim != 2:
        source = np.asarray(source)[choose]
    return transformed_path_stages(
        samples.offset_lm_rad[choose],
        lookup,
        transform,
        width=width,
        residual_jones=samples.residual_jones,
        moving_id=samples.moving_id[choose],
        reference_id=samples.reference_id[choose],
        moving_is_p=samples.moving_is_p[choose],
        chi_moving=samples.parallactic_angle_rad[choose],
        chi_reference=samples.reference_parallactic_angle_rad[choose],
        source=source,
    )["visibility"]


def _predict_state(samples, lookup, state: PhysicalBeamState, mask):
    choose = np.asarray(mask, dtype=bool)
    source = samples.source
    if np.asarray(source).ndim != 2:
        source = np.asarray(source)[choose]
    return physical_path_stages(
        samples.offset_lm_rad[choose],
        lookup,
        state,
        residual_jones=samples.residual_jones,
        moving_id=samples.moving_id[choose],
        reference_id=samples.reference_id[choose],
        moving_is_p=samples.moving_is_p[choose],
        chi_moving=samples.parallactic_angle_rad[choose],
        chi_reference=samples.reference_parallactic_angle_rad[choose],
        source=source,
    )["visibility"]


def _thin_mask(samples: CorrectionSamples, *, stride: int, max_train_cells: int) -> np.ndarray:
    n = samples.train.size
    keep = np.ones(n, dtype=bool)
    if int(stride) > 1:
        keep[:] = False
        keep[:: int(stride)] = True
        for mask in (samples.mover_holdout, samples.spatial_holdout, samples.train):
            for antenna in np.unique(samples.moving_id[mask]):
                rows = np.flatnonzero(mask & (samples.moving_id == int(antenna)))
                if rows.size and not bool(np.any(keep[rows])):
                    keep[rows[0]] = True
    if int(max_train_cells) > 0:
        keys = packed_offset_keys(samples.offset_lm_rad)
        train_keys = np.unique(keys[samples.train])[: int(max_train_cells)]
        keep = keep & (~samples.train | np.isin(keys, train_keys))
    return keep


def _subset_samples(samples: CorrectionSamples, keep: np.ndarray) -> CorrectionSamples:
    source = samples.source
    if source.ndim != 2:
        source = source[keep]
    out = CorrectionSamples(
        offset_lm_rad=samples.offset_lm_rad[keep],
        measured=samples.measured[keep],
        baseline=samples.baseline[keep],
        weight=samples.weight[keep],
        source=source,
        moving_is_p=samples.moving_is_p[keep],
        moving_id=samples.moving_id[keep],
        reference_id=samples.reference_id[keep],
        residual_jones=samples.residual_jones,
        parallactic_angle_rad=samples.parallactic_angle_rad[keep],
        reference_parallactic_angle_rad=samples.reference_parallactic_angle_rad[keep],
        antenna_names=samples.antenna_names,
        train=samples.train[keep],
        spatial_holdout=samples.spatial_holdout[keep],
        mover_holdout=samples.mover_holdout[keep],
        main_lobe=samples.main_lobe[keep],
        mid=samples.mid[keep],
        outer=samples.outer[keep],
        field_id=samples.field_id[keep],
        frequency_hz=samples.frequency_hz,
        spectral_window_id=samples.spectral_window_id,
    )
    object.__setattr__(
        out,
        "reference_holdout",
        np.asarray(samples.reference_holdout, dtype=bool)[keep],
    )
    return out


def _train_losses(samples, predicted, rr_ok, ll_ok) -> dict[str, float]:
    train = samples.train
    return {
        "combined": float(
            complex_visibility_loss(samples.measured[train], predicted, samples.weight[train])
        ),
        "rr": hand_loss(samples.measured[train], predicted, samples.weight[train], "rr"),
        "ll": hand_loss(samples.measured[train], predicted, samples.weight[train], "ll"),
        "rr_minus_ll": rr_ll_difference_loss(
            samples.measured[train],
            predicted,
            samples.weight[train],
            rr_ok[train],
            ll_ok[train],
        ),
    }


def _readme(payload: dict) -> str:
    interp = payload.get("interpretation") or {}
    return (
        "\n".join(
            [
                "# SPW-4 discrete spatial-convention test",
                "",
                "This is **SPW-4 development**. Width is frozen at 1.04.",
                "Native CASSBEAM squint remains physically present.",
                "Its current coordinate orientation is not treated as the production prior.",
                "SPW 5 remained sealed. The 128-member convention ladder stayed closed.",
                "",
                f"Selected: **{interp.get('selected_name')}**.",
                f"Locked: **{interp.get('locked')}**.",
                f"Empirical vs no-squint improves: **{interp.get('empirical_beats_nosquint')}**.",
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
    parser.add_argument("--identity-max-rows", type=int, default=0)
    parser.add_argument("--max-train-cells", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    print("spatial_convention start", flush=True)
    refuse_spw5(spectral_window_id=4, opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES or output_dir == arguments.comparison_dir.resolve():
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmp = _comparison()
    payload: dict[str, object] = {
        "artifact": SPATIAL_CONVENTION_EXPERIMENT,
        "development_only": True,
        "spw5_closed": True,
        "spw5_opened": False,
        "width_frozen": FROZEN_WIDTH,
        "native_orientation_not_production_prior": True,
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
        lookup = catalog_diagonal_feed_lookup(catalog, float(frequencies[0]))
        identity_rows = np.arange(fields["offset"].shape[0])
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
        comparison = evaluate_holoraster_cassbeam(
            catalog=catalog,
            frequencies_hz=frequencies,
            convention=locked_convention(),
            offset_lm_rad=fields["offset"][identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            residual_jones=residual,
            source=source_id,
            off_diagonal=False,
        )
        spatial_id = transformed_path_stages(
            fields["offset"][identity_rows],
            lookup,
            IDENTITY_TRANSFORM,
            width=1.0,
            residual_jones=residual,
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            source=source_id,
        )
        identity_report = require_physical_identity_matches_comparison(spatial_id, comparison)
        print(json.dumps(cmp._jsonable(identity_report)), flush=True)
        if identity_rows.size == fields["offset"].shape[0]:
            baseline = spatial_id["visibility"]
        else:
            baseline = transformed_path_stages(
                fields["offset"],
                lookup,
                IDENTITY_TRANSFORM,
                width=1.0,
                residual_jones=residual,
                moving_id=fields["moving"],
                reference_id=fields["reference"],
                moving_is_p=fields["moving_is_p"],
                chi_moving=fields["chi_m"],
                chi_reference=fields["chi_r"],
                source=source_rows,
            )["visibility"]
        samples = CorrectionSamples(
            offset_lm_rad=fields["offset"],
            measured=measured,
            baseline=baseline,
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
        thin = _thin_mask(
            samples,
            stride=int(arguments.stride),
            max_train_cells=int(arguments.max_train_cells),
        )
        if not bool(np.all(thin)):
            samples = _subset_samples(samples, thin)
            rr_flags = rr_flags[thin]
            ll_flags = ll_flags[thin]
            print("thinned rows", int(samples.train.size), flush=True)
        measured_delta = np.asarray(MEASURED_INDEPENDENT_DELTA_ARCMIN, dtype=np.float64)
        empirical_w1 = candidate_state(
            "empirical", empirical_delta_rad=measured_delta * ARCMIN_TO_RAD, width=1.0
        )
        empirical_w = candidate_state(
            "empirical_plus_width",
            empirical_delta_rad=measured_delta * ARCMIN_TO_RAD,
            width=FROZEN_WIDTH,
        )
        none_w = PhysicalBeamState(width=FROZEN_WIDTH, delta_lm_rad=(0.0, 0.0))
        none_w1 = PhysicalBeamState(width=1.0, delta_lm_rad=(0.0, 0.0))
        print("train-rank 16 spatial transforms at width", FROZEN_WIDTH, flush=True)
        records = []
        for transform in spatial_transform_catalog():
            print("train", transform.name, flush=True)
            pred = _predict_transform(samples, lookup, transform, FROZEN_WIDTH, samples.train)
            losses = _train_losses(samples, pred, rr_flags, ll_flags)
            delta = transformed_separation_rad(transform) / ARCMIN_TO_RAD
            records.append(
                {
                    "name": transform.name,
                    "is_identity": transform.is_identity(),
                    "l_sign": transform.l_sign,
                    "m_sign": transform.m_sign,
                    "swap_lm": transform.swap_lm,
                    "swap_rl": transform.swap_rl,
                    "delta_arcmin": [float(delta[0]), float(delta[1])],
                    "alignment": vector_alignment(delta, measured_delta),
                    **{f"train_{key}": value for key, value in losses.items()},
                }
            )
        selection = select_spatial_transform(records, measured_delta_arcmin=measured_delta)
        print("selection", json.dumps(cmp._jsonable(selection)), flush=True)
        print("full-row reference models", flush=True)
        predictions = {
            "native_w1": samples.baseline,
            "native_w104": _predict_transform(
                samples,
                lookup,
                IDENTITY_TRANSFORM,
                FROZEN_WIDTH,
                np.ones(samples.train.size, dtype=bool),
            ),
            "no_squint_w1": _predict_state(
                samples, lookup, none_w1, np.ones(samples.train.size, dtype=bool)
            ),
            "no_squint_w104": _predict_state(
                samples, lookup, none_w, np.ones(samples.train.size, dtype=bool)
            ),
            "empirical_w1": _predict_state(
                samples, lookup, empirical_w1, np.ones(samples.train.size, dtype=bool)
            ),
            "empirical_w104": _predict_state(
                samples, lookup, empirical_w, np.ones(samples.train.size, dtype=bool)
            ),
        }
        selected_transform = None
        if selection["selected"] is not None:
            selected_transform = SpatialTransform(
                selection["selected"]["l_sign"],
                selection["selected"]["m_sign"],
                selection["selected"]["swap_lm"],
                selection["selected"]["swap_rl"],
            )
            print("full selected", selected_transform.name, flush=True)
            predictions["selected_w104"] = _predict_transform(
                samples,
                lookup,
                selected_transform,
                FROZEN_WIDTH,
                np.ones(samples.train.size, dtype=bool),
            )
        score_kw = {
            "rr_ok": rr_flags,
            "ll_ok": ll_flags,
            "n_boot": int(arguments.n_boot),
            "seed": int(arguments.seed),
        }
        paired = {
            "empirical_w1_vs_nosquint_w1": score_model(
                samples, predictions["empirical_w1"], predictions["no_squint_w1"], **score_kw
            ),
            "empirical_w104_vs_nosquint_w104": score_model(
                samples, predictions["empirical_w104"], predictions["no_squint_w104"], **score_kw
            ),
            "native_w104_vs_nosquint_w104": score_model(
                samples, predictions["native_w104"], predictions["no_squint_w104"], **score_kw
            ),
            "empirical_w104_vs_native_w104": score_model(
                samples, predictions["empirical_w104"], predictions["native_w104"], **score_kw
            ),
        }
        if "selected_w104" in predictions:
            paired["selected_vs_native_w104"] = score_model(
                samples, predictions["selected_w104"], predictions["native_w104"], **score_kw
            )
            paired["selected_vs_nosquint_w104"] = score_model(
                samples, predictions["selected_w104"], predictions["no_squint_w104"], **score_kw
            )
            paired["selected_vs_empirical_w104"] = score_model(
                samples, predictions["selected_w104"], predictions["empirical_w104"], **score_kw
            )
        emp_vs_none = paired["empirical_w1_vs_nosquint_w1"]
        interpretation = {
            "selected_name": None if selected_transform is None else selected_transform.name,
            "locked": bool(selection["locked"]),
            "native_orientation_retained_as_production_prior": False,
            "keep_physical_squint": True,
            "width_frozen": FROZEN_WIDTH,
            "empirical_beats_nosquint": bool(
                emp_vs_none["spatial"]["improves"] and emp_vs_none["moving"]["improves"]
            ),
            "spw5_ready": False,
            "development_only": True,
        }
        if "selected_vs_native_w104" in paired and "selected_vs_nosquint_w104" in paired:
            interpretation["selected_beats_native"] = bool(
                paired["selected_vs_native_w104"]["spatial"]["improves"]
                and paired["selected_vs_native_w104"]["moving"]["improves"]
            )
            interpretation["selected_beats_nosquint"] = bool(
                paired["selected_vs_nosquint_w104"]["spatial"]["improves"]
                and paired["selected_vs_nosquint_w104"]["moving"]["improves"]
            )
            interpretation["locked"] = bool(
                selection["locked"]
                and interpretation["selected_beats_native"]
                and interpretation["selected_beats_nosquint"]
                and paired["selected_vs_native_w104"]["movers"]["unit_gate_passes"]
            )
        payload.update(
            {
                "identity_report": identity_report,
                "native_delta_arcmin": [
                    float(item) for item in native_separation_rad() / ARCMIN_TO_RAD
                ],
                "measured_delta_arcmin": [float(item) for item in measured_delta],
                "transform_records": records,
                "selection": selection,
                "paired_scores": paired,
                "interpretation": interpretation,
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
        write_json(cmp._jsonable(records), output_dir / "transform_records.json")
        write_json(cmp._jsonable(selection), output_dir / "selection.json")
        write_json(cmp._jsonable(paired), output_dir / "paired_scores.json")
        write_json(cmp._jsonable(interpretation), output_dir / "interpretation.json")
        write_json(
            cmp._jsonable(
                {
                    "spw5_opened": False,
                    "spw5_read": False,
                    "ready_for_later_transfer": False,
                    "locked_convention": interpretation["locked"],
                    "selected": interpretation["selected_name"],
                }
            ),
            output_dir / "spw5_recommendation.json",
        )
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
