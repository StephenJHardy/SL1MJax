"""Audit the validation-selected SPW-4 diagonal correction. No SPW 5, no refit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from sl1mjax.cassbeam_highres import DEFAULT_HIGHRES_ROOT, HighresCassbeamCatalog
from sl1mjax.holography import THOL0001_SPW4_CHANNEL_32_HZ
from sl1mjax.holography_alignment import holoraster_pair_masks
from sl1mjax.holography_calibration import write_json
from sl1mjax.holography_cassbeam_correction import (
    IDENTITY_CORRECTION,
    CorrectionSamples,
    catalog_diagonal_feed_lookup,
    predict_from_state,
    refuse_spw5,
    region_masks_from_voltage,
    require_identity_matches_baseline,
    require_positive_envelope,
)
from sl1mjax.holography_cassbeam_correction_audit import (
    VALIDATION_SELECTED_SPW4,
    accepted_validation_state,
    audit_payload,
    joint_squint_width_surface,
    leave_one_mover_sensitivity,
    model_offset_grid,
    refuse_reopening_rejected_ladder_terms,
    render_feed_frame_beams,
    split_region_hand_losses,
    squint_from_feed_jones,
    squint_from_visibilities,
    warped_squint_separation_arcmin,
)
from sl1mjax.holography_diagonal_correction import (
    HOLORASTER_FIELD_ID,
    frozen_protocol_payload,
    protocol_as_mapping,
    refuse_c147_training,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates

COMPARISON_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_correction"
)
UNBLOCK = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/full_jones_unblock"
)
SCIENTIFIC_MS = Path(
    "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.lowerC.spw45.scientific.ms"
)
CHANNEL = 32
FROZEN_NAMES = (
    "holoraster_cassbeam_comparison",
    "one_axis_visibility_holdouts.json",
    "c147_offset_ring",
)


def _comparison():
    path = Path(__file__).with_name("run_thol0001_holoraster_cassbeam_comparison.py")
    spec = importlib.util.spec_from_file_location("thol0001_holoraster_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plots(
    output_dir: Path,
    *,
    offset: np.ndarray,
    n_side: int,
    identity_jones: np.ndarray,
    corrected_jones: np.ndarray,
    surface: dict,
    movers: dict,
    squint: dict,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    half = 8.0
    extent = (-half, half, -half, half)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for row, jones, title in (
        (0, identity_jones, "identity CASSBEAM"),
        (1, corrected_jones, "validation-selected"),
    ):
        for col, hand, label in ((0, 0, "RR"), (1, 1, "LL")):
            plane = 20.0 * np.log10(np.maximum(np.abs(jones[:, hand, hand]), 1.0e-6)).reshape(
                n_side, n_side
            )
            image = axes[row, col].imshow(
                plane, origin="lower", extent=extent, vmin=-20.0, vmax=0.0, cmap="viridis"
            )
            axes[row, col].set_title(f"{title} {label} (dB)")
            axes[row, col].set_xlabel("l (arcmin)")
            axes[row, col].set_ylabel("m (arcmin)")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046)
        ratio = np.abs(jones[:, 0, 0]) - np.abs(jones[:, 1, 1])
        image = axes[row, 2].imshow(
            ratio.reshape(n_side, n_side), origin="lower", extent=extent, cmap="coolwarm"
        )
        axes[row, 2].set_title(f"{title} |RR|-|LL|")
        axes[row, 2].set_xlabel("l (arcmin)")
        axes[row, 2].set_ylabel("m (arcmin)")
        fig.colorbar(image, ax=axes[row, 2], fraction=0.046)
    fig.suptitle("Feed-frame copolar beams. Pointing stays rejected.")
    fig.tight_layout()
    path = output_dir / "corrected_rr_ll_beams.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(5.2, 5.0))
    for record, marker, name in (
        (squint["visibility_measured"], "o", "measured HOLORASTER"),
        (squint["visibility_identity"], "s", "identity visibilities"),
        (squint["model_identity"], "D", "identity model grid"),
        (squint["model_corrected"], "^", "corrected model grid"),
        (squint["visibility_corrected"], "v", "corrected visibilities"),
    ):
        axis.plot(
            [float(record["rr_l_arcmin"]), float(record["ll_l_arcmin"])],
            [float(record["rr_m_arcmin"]), float(record["ll_m_arcmin"])],
            lw=0.8,
        )
        axis.scatter(
            [float(record["rr_l_arcmin"])],
            [float(record["rr_m_arcmin"])],
            marker=marker,
            label=f"{name} {float(record['separation_arcmin']):.3f}′",
        )
        axis.scatter(
            [float(record["ll_l_arcmin"])],
            [float(record["ll_m_arcmin"])],
            marker=marker,
        )
    axis.axhline(0.0, color="k", lw=0.4)
    axis.axvline(0.0, color="k", lw=0.4)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("l (arcmin)")
    axis.set_ylabel("m (arcmin)")
    axis.set_title("Publication 20%-of-peak squint")
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = output_dir / "squint_20pct_identity_versus_corrected.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    loss = np.asarray(surface["loss"], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    image = axis.pcolormesh(
        surface["width_grid"],
        surface["squint_grid"],
        loss,
        shading="nearest",
        cmap="magma_r",
    )
    axis.scatter(
        [surface["accepted_width"]],
        [surface["accepted_squint"]],
        marker="o",
        facecolors="none",
        edgecolors="w",
        s=80,
        label="accepted (0.85, 1.04)",
    )
    axis.scatter(
        [surface["min_width"]],
        [surface["min_squint"]],
        marker="+",
        c="cyan",
        s=80,
        label="surface minimum",
    )
    axis.set_xlabel("width scale")
    axis.set_ylabel("squint scale")
    axis.set_title("Train-row complex visibility loss")
    axis.legend(fontsize=8)
    fig.colorbar(image, ax=axis, label="L")
    fig.tight_layout()
    path = output_dir / "joint_squint_width_train_loss.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))

    names = list(movers["movers"]["names"])
    delta = np.asarray(movers["movers"]["delta"], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.bar(names, delta)
    axis.axhline(0.0, color="k", lw=0.6)
    axis.set_ylabel("paired ΔL")
    axis.set_title("Holdout-mover paired ΔL")
    fig.tight_layout()
    path = output_dir / "holdout_mover_delta.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(str(path))
    return written


def _readme(payload: dict) -> str:
    interp = payload.get("interpretation") or {}
    return "\n".join(
        [
            "# Validation-selected SPW-4 diagonal correction",
            "",
            f"Artifact: `{VALIDATION_SELECTED_SPW4}`.",
            f"Status: **{interp.get('status')}**.",
            f"SPW 5 opened: **{payload.get('spw5_opened')}**.",
            "",
            "Pointing remains rejected. Sidelobe and azimuthal terms were not opened.",
            "This is not a general calibrated C-band beam unless the interpretation "
            "freezes the family.",
            "",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR / "validation_selected_audit")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    arguments = parser.parse_args()
    refuse_spw5(spectral_window_id=4, opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES or output_dir == arguments.comparison_dir.resolve():
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmp = _comparison()
    accepted = accepted_validation_state()
    refuse_reopening_rejected_ladder_terms(accepted)
    payload: dict[str, object] = {
        "artifact": VALIDATION_SELECTED_SPW4,
        "source_revision": cmp._source_revision(),
        "protocol": protocol_as_mapping(frozen_protocol_payload()),
        "spw5_closed": True,
        "spw5_opened": False,
        "model_selected": False,
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        software = run_software_gates(plane)
        payload["software_gates"] = software
        payload["artifact_checksum"] = artifact_checksum_report(
            catalog, THOL0001_SPW4_CHANNEL_32_HZ
        )
        if not bool(software.get("passed", False)):
            raise RuntimeError("high-resolution CASSBEAM software gates failed")
        residual = cmp._residual_jones(arguments.product_dir)
        diag = cmp._diag()
        from sl1mjax.holography_ms import _read_antennas, _tables

        tables = _tables()
        _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = tuple(names)
        positions = np.asarray(positions, dtype=np.float64)
        observation, chi, packed, intensity, voltage, rr_ok, ll_ok = cmp._load_holoraster_channel(
            diag,
            tables,
            arguments.measurement_set,
            positions,
            channel=CHANNEL,
        )
        del intensity, rr_ok, ll_ok
        frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64).reshape(-1)
        refuse_c147_training(observation.block.field_id)
        if np.any(np.asarray(observation.block.field_id) != HOLORASTER_FIELD_ID):
            raise ValueError("audit samples must be HOLORASTER field 10")
        geometry = moving_reference_row_geometry(observation, require_all_channels=False)
        pair = holoraster_pair_masks(observation)
        usable = np.asarray(geometry["usable"], dtype=bool) & np.asarray(
            pair["moving_reference"], dtype=bool
        )
        holdouts = spw4_correction_holdouts(observation, names)
        development = holdouts.train | holdouts.spatial_holdout | holdouts.mover_holdout
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
        if measured.ndim == 4:
            measured = measured[fields["rows"], 0]
        else:
            measured = measured[fields["rows"]]
        weight = cmp._hand_weight(observation, keep)
        if weight.ndim == 4:
            weight = weight[:, 0]
        voltage_rows = np.asarray(voltage, dtype=np.float64)[fields["rows"]]
        regions = region_masks_from_voltage(voltage_rows)
        lookup = catalog_diagonal_feed_lookup(catalog, float(frequencies[0]))
        print("predicting identity and accepted prefix", flush=True)
        samples = CorrectionSamples(
            offset_lm_rad=fields["offset"],
            measured=measured,
            baseline=np.zeros_like(measured),
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
        require_positive_envelope(samples.offset_lm_rad, IDENTITY_CORRECTION)
        require_positive_envelope(samples.offset_lm_rad, accepted)
        identity = predict_from_state(samples, lookup, IDENTITY_CORRECTION)
        samples = replace(samples, baseline=identity)
        require_identity_matches_baseline(identity, samples.baseline)
        corrected = predict_from_state(samples, lookup, accepted)
        print("rendering model beams and publication squint", flush=True)
        grid, n_side = model_offset_grid()
        identity_jones = render_feed_frame_beams(lookup, IDENTITY_CORRECTION, grid)
        corrected_jones = render_feed_frame_beams(lookup, accepted, grid)
        freq = float(frequencies[0])
        model_identity_sq = squint_from_feed_jones(
            grid, identity_jones, series="model_identity", frequency_hz=freq
        )
        model_corrected_sq = squint_from_feed_jones(
            grid, corrected_jones, series="model_corrected", frequency_hz=freq
        )
        vis_measured = squint_from_visibilities(
            samples.offset_lm_rad,
            samples.measured,
            samples.weight,
            series="measured",
            frequency_hz=freq,
        )
        vis_identity = squint_from_visibilities(
            samples.offset_lm_rad,
            identity,
            samples.weight,
            series="identity",
            frequency_hz=freq,
        )
        vis_corrected = squint_from_visibilities(
            samples.offset_lm_rad,
            corrected,
            samples.weight,
            series="corrected",
            frequency_hz=freq,
        )
        print("scoring split × region × hand losses", flush=True)
        losses_identity = split_region_hand_losses(samples, identity)
        losses_corrected = split_region_hand_losses(samples, corrected)
        print("scoring holdout movers and leave-one-mover", flush=True)
        movers = leave_one_mover_sensitivity(samples, corrected, identity)
        print("evaluating joint squint–width train surface", flush=True)
        train = samples.train
        n_train = int(np.sum(train))
        train_mask = np.ones(n_train, dtype=bool)
        spatial_flag = np.zeros(n_train, dtype=bool)
        mover_flag = np.zeros(n_train, dtype=bool)
        spatial_flag[0] = True
        mover_flag[1] = True
        train_mask[0] = False
        train_mask[1] = False
        train_samples = replace(
            samples,
            offset_lm_rad=samples.offset_lm_rad[train],
            measured=samples.measured[train],
            baseline=samples.baseline[train],
            weight=samples.weight[train],
            source=samples.source if samples.source.shape == (2, 2) else samples.source[train],
            moving_is_p=samples.moving_is_p[train],
            moving_id=samples.moving_id[train],
            reference_id=samples.reference_id[train],
            parallactic_angle_rad=samples.parallactic_angle_rad[train],
            reference_parallactic_angle_rad=samples.reference_parallactic_angle_rad[train],
            train=train_mask,
            spatial_holdout=spatial_flag,
            mover_holdout=mover_flag,
            main_lobe=samples.main_lobe[train],
            mid=samples.mid[train],
            outer=samples.outer[train],
            field_id=samples.field_id[train],
        )
        surface = joint_squint_width_surface(
            train_samples, lookup, row_mask=np.ones(n_train, dtype=bool)
        )
        payload.update(
            audit_payload(
                model_identity_squint=model_identity_sq,
                model_corrected_squint=model_corrected_sq,
                visibility_measured_squint=vis_measured,
                visibility_identity_squint=vis_identity,
                visibility_corrected_squint=vis_corrected,
                warped_separation_arcmin=warped_squint_separation_arcmin(accepted),
                losses_identity=losses_identity,
                losses_corrected=losses_corrected,
                surface=surface,
                movers=movers,
            )
        )
        if bool(payload["interpretation"]["open_spw5_transfer"]):
            raise RuntimeError("SPW 5 transfer must be a separate no-refit run after freeze")
        written = _write_plots(
            output_dir,
            offset=grid,
            n_side=n_side,
            identity_jones=identity_jones,
            corrected_jones=corrected_jones,
            surface=surface,
            movers=movers,
            squint=payload["squint"],
        )
        payload["plots"] = written
        write_json(cmp._jsonable(payload), output_dir / f"{VALIDATION_SELECTED_SPW4}.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(json.dumps(cmp._jsonable(payload["interpretation"])), flush=True)
        print(
            json.dumps(
                cmp._jsonable(
                    {
                        "model_corrected_arcmin": model_corrected_sq["separation_arcmin"],
                        "warped_arcmin": warped_squint_separation_arcmin(accepted),
                        "measured_arcmin": vis_measured["separation_arcmin"],
                        "min_squint": surface["min_squint"],
                        "min_width": surface["min_width"],
                        "identifiable": surface["identifiable_interior"],
                    }
                )
            ),
            flush=True,
        )
        return 0
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["traceback"] = traceback.format_exc()
        write_json(cmp._jsonable(payload), output_dir / f"{VALIDATION_SELECTED_SPW4}.json")
        (output_dir / "README.md").write_text(_readme(payload))
        print(payload["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
