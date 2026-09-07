"""SPW-4 physical squint and common-width experiment. SPW 5 stays sealed."""

from __future__ import annotations

import argparse
import hashlib
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
    CorrectionSamples,
    catalog_diagonal_feed_lookup,
    region_masks_from_voltage,
)
from sl1mjax.holography_cassbeam_holoraster_report import locked_convention
from sl1mjax.holography_diagonal_correction import (
    HOLORASTER_FIELD_ID,
    packed_offset_keys,
    refuse_c147_training,
    refuse_spw5,
    spw4_correction_holdouts,
)
from sl1mjax.holography_full_jones import moving_reference_row_geometry
from sl1mjax.holography_highres_cassbeam import artifact_checksum_report, run_software_gates
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    PHYSICAL_SQUINT_EXPERIMENT,
    apply_physical_feed_frame,
    native_separation_rad,
    parameterization_record,
    physical_path_stages,
    require_physical_identity_matches_comparison,
)
from sl1mjax.holography_physical_squint_experiment import (
    CANDIDATE_NAMES,
    WIDTH_GRID,
    _normalize_origin_power,
    candidate_state,
    fit_width_on_train,
    interpret_experiment,
    joint_width_squint_surface,
    measure_direct_squint,
    perpendicular_squint_profile,
    score_model,
    squint_from_common_map,
)
from sl1mjax.holography_pointing_maps import visit_id_per_time

COMPARISON_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_cassbeam_comparison"
)
PRODUCT_DIR = Path(
    "/media/stephen/astro/vla/extracted/commissioning/validation/scientific/"
    "holoraster_physical_squint_width_v1"
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
    "c147_offset_ring",
    "vla_c_band_beam_validation_v1",
)
ARCMIN = np.pi / (180.0 * 60.0)


def _comparison():
    path = Path(__file__).with_name("run_thol0001_holoraster_cassbeam_comparison.py")
    spec = importlib.util.spec_from_file_location("thol0001_holoraster_comparison", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readme(payload: dict) -> str:
    interp = payload.get("interpretation") or {}
    return (
        "\n".join(
            [
                "# SPW-4 physical squint and common-width experiment",
                "",
                "This is **SPW-4 development**. Spatial and mover holdouts already",
                "influenced model development. They are not an unseen validation set.",
                "",
                "Native CASSBEAM squint was never removed by default.",
                "SPW 5 remained sealed. Full Jones was untouched.",
                "No general C-band beam was frozen.",
                "",
                f"Outcome: **{interp.get('outcome')}**.",
                f"Width survived: **{interp.get('width_survived')}**.",
                f"Physical squint correction: **{interp.get('outcome')}**.",
                f"Empirical squint stored: **{interp.get('store_empirical_correction')}**.",
                f"Later SPW-5 transfer ready: **{interp.get('spw5_ready')}**.",
                "",
            ]
        )
        + "\n"
    )


def _code_hash() -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    import sl1mjax.holography_physical_squint as physical
    import sl1mjax.holography_physical_squint_experiment as experiment

    for module in (physical, experiment):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def _scatter_map(axis, offset, power, centroid, title):
    axis.scatter(
        offset[:, 0] / ARCMIN,
        offset[:, 1] / ARCMIN,
        c=10.0 * np.log10(np.maximum(power, 1.0e-6)),
        s=8,
        cmap="viridis",
    )
    if centroid is not None:
        axis.scatter([centroid[0] / ARCMIN], [centroid[1] / ARCMIN], c="red", marker="+", s=80)
    axis.set_title(title)
    axis.set_xlabel("l (arcmin)")
    axis.set_ylabel("m (arcmin)")
    axis.set_aspect("equal")


def _write_plots(output_dir: Path, bundle: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    maps = bundle["maps"]
    cass = bundle["cassbeam_maps"]
    emp = bundle["empirical_maps"]
    direct = bundle["direct"]
    scores = bundle["paired"]
    widths = bundle["widths"]
    joint = bundle.get("joint")
    perp = bundle.get("perp")
    radial = bundle["radial"]
    obs = bundle["obs_pred"]
    residual_maps = bundle["residual_maps"]

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6))
    _scatter_map(
        axes[0, 0],
        maps["offset"],
        maps["rr_power"],
        np.array(
            [direct["independent_masks"]["rr_l_arcmin"], direct["independent_masks"]["rr_m_arcmin"]]
        )
        * ARCMIN,
        "Map-domain measured |E_RR|^2 + centroid",
    )
    _scatter_map(
        axes[0, 1],
        maps["offset"],
        maps["ll_power"],
        np.array(
            [direct["independent_masks"]["ll_l_arcmin"], direct["independent_masks"]["ll_m_arcmin"]]
        )
        * ARCMIN,
        "Map-domain measured |E_LL|^2 + centroid",
    )
    _scatter_map(
        axes[1, 0], cass["offset"], cass["rr_power"], None, "Map-domain native CASSBEAM |E_RR|^2"
    )
    _scatter_map(
        axes[1, 1], cass["offset"], cass["ll_power"], None, "Map-domain native CASSBEAM |E_LL|^2"
    )
    fig.tight_layout()
    path = output_dir / "01_02_measured_and_native_maps.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    _scatter_map(
        axes[0], emp["offset"], emp["rr_power"], None, "Map-domain empirical-squint |E_RR|^2"
    )
    _scatter_map(
        axes[1], emp["offset"], emp["ll_power"], None, "Map-domain empirical-squint |E_LL|^2"
    )
    fig.tight_layout()
    path = output_dir / "03_empirical_maps.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(5.6, 5.2))
    for record in direct["by_mover"]:
        axis.arrow(
            0.0,
            0.0,
            float(record["dl_arcmin"]),
            float(record["dm_arcmin"]),
            length_includes_head=True,
            head_width=0.02,
            alpha=0.75,
        )
        axis.text(
            float(record["dl_arcmin"]), float(record["dm_arcmin"]), record["name"], fontsize=8
        )
    axis.axhline(0.0, color="k", lw=0.4)
    axis.axvline(0.0, color="k", lw=0.4)
    axis.set_title("Map-domain R−L centroid vectors by mover")
    axis.set_xlabel("Δl (arcmin)")
    axis.set_ylabel("Δm (arcmin)")
    fig.tight_layout()
    path = output_dir / "04_mover_centroid_vectors.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for axis, rows, title in (
        (axes[0], direct["by_mover"], "Map-domain squint by mover"),
        (axes[1], direct["by_reference"], "Map-domain squint by reference"),
    ):
        names = [item["name"] for item in rows]
        axis.bar(names, [item["separation_arcmin"] for item in rows])
        axis.set_title(title)
        axis.set_ylabel("separation (arcmin)")
        axis.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    path = output_dir / "05_squint_by_mover_reference.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    for axis, key, title in (
        (axes[0], "rr", "Visibility-domain RR"),
        (axes[1], "ll", "Visibility-domain LL"),
        (axes[2], "rr_ll", "Visibility-domain RR−LL"),
    ):
        axis.scatter(obs[key]["measured"].real, obs[key]["predicted"].real, s=4, alpha=0.35)
        axis.set_title(f"{title} Re(obs) vs Re(pred)")
        axis.set_xlabel("measured")
        axis.set_ylabel("native predicted")
    fig.tight_layout()
    path = output_dir / "06_obs_vs_pred.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(6.2, 3.8))
    for name, profile in radial.items():
        axis.plot(profile["radius_arcmin"], profile["loss"], label=name)
    axis.set_title("Visibility-domain radial residual power")
    axis.set_xlabel("radius (arcmin)")
    axis.set_ylabel("residual power")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "07_radial_residual_profiles.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(5.8, 3.6))
    for name, profile in widths.items():
        axis.plot(profile["grid"], profile["loss"], label=name)
        axis.axvline(profile["width"], ls="--", lw=0.8)
    axis.set_title("Visibility-domain width profiles on training rows")
    axis.set_xlabel("common width w")
    axis.set_ylabel("complex visibility loss")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "08_width_loss_profile.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    if perp is not None:
        fig, axis = plt.subplots(figsize=(5.8, 3.6))
        axis.plot(perp["grid_arcmin"], perp["loss"])
        axis.set_title("Visibility-domain perpendicular squint profile")
        axis.set_xlabel("δ_perp (arcmin)")
        axis.set_ylabel("train complex visibility loss")
        fig.tight_layout()
        path = output_dir / "09_perpendicular_squint_profile.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))

    if joint is not None:
        fig, axis = plt.subplots(figsize=(6.2, 4.6))
        image = axis.pcolormesh(
            joint["width_grid"],
            joint["magnitude_scale_grid"],
            np.asarray(joint["loss"]),
            shading="nearest",
            cmap="magma_r",
        )
        axis.scatter([joint["min_width"]], [joint["min_magnitude_scale"]], c="cyan", marker="+")
        axis.set_xlabel("common width w")
        axis.set_ylabel("physical squint magnitude / native")
        axis.set_title("Visibility-domain train loss: width vs |δ|")
        fig.colorbar(image, ax=axis, label="L")
        fig.tight_layout()
        path = output_dir / "10_joint_width_squint.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))

    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    movers = scores["empirical_vs_native"]["movers"]["movers"]
    axis.bar(movers["names"], movers["delta"])
    axis.axhline(0.0, color="k", lw=0.5)
    axis.set_title("Visibility-domain per-mover paired ΔL (empirical − native)")
    axis.set_ylabel("ΔL")
    fig.tight_layout()
    path = output_dir / "11_mover_paired_delta.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for axis, name, title in (
        (axes[0], "no_squint", "Visibility residual, no-squint"),
        (axes[1], "native", "Visibility residual, native CASSBEAM"),
        (axes[2], "empirical", "Visibility residual, empirical squint"),
    ):
        item = residual_maps[name]
        axis.scatter(
            item["offset"][:, 0] / ARCMIN,
            item["offset"][:, 1] / ARCMIN,
            c=item["loss"],
            s=8,
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        axis.set_aspect("equal")
    fig.tight_layout()
    path = output_dir / "12_residual_maps.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path))
    return written


def _radial_profile(offset, measured, predicted, weight, n_bin: int = 12) -> dict[str, list[float]]:
    radius = np.hypot(offset[:, 0], offset[:, 1]) / ARCMIN
    resid = np.abs(measured[:, 0, 0] - predicted[:, 0, 0]) ** 2
    resid = resid + np.abs(measured[:, 1, 1] - predicted[:, 1, 1]) ** 2
    ww = 0.5 * (weight[:, 0, 0] + weight[:, 1, 1])
    edges = np.linspace(0.0, max(float(np.max(radius)), 1.0), n_bin + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    loss = np.full(n_bin, np.nan)
    for index in range(n_bin):
        keep = (radius >= edges[index]) & (radius < edges[index + 1]) & (ww > 0.0)
        if bool(np.any(keep)):
            loss[index] = float(np.average(resid[keep], weights=ww[keep]))
    return {
        "radius_arcmin": [float(item) for item in centres],
        "loss": [float(item) for item in loss],
    }


def _cell_residual_map(offset, measured, predicted, weight) -> dict[str, np.ndarray]:
    from sl1mjax.holography_physical_squint_experiment import cell_hand_maps

    resid = 0.5 * (
        np.abs(measured[:, 0, 0] - predicted[:, 0, 0]) ** 2
        + np.abs(measured[:, 1, 1] - predicted[:, 1, 1]) ** 2
    )
    ww = 0.5 * (weight[:, 0, 0] + weight[:, 1, 1])
    centres, mean, _ = cell_hand_maps(offset, resid, ww, np.isfinite(resid) & (ww > 0.0))
    return {"offset": centres, "loss": mean}


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


def _subset_samples(
    samples: CorrectionSamples, keep: np.ndarray, reference_holdout: np.ndarray
) -> CorrectionSamples:
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
    object.__setattr__(out, "reference_holdout", np.asarray(reference_holdout, dtype=bool)[keep])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-set", type=Path, default=SCIENTIFIC_MS)
    parser.add_argument("--product-dir", type=Path, default=UNBLOCK)
    parser.add_argument("--comparison-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument("--output-dir", type=Path, default=PRODUCT_DIR)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_HIGHRES_ROOT)
    parser.add_argument("--max-train-cells", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--identity-max-rows", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-joint", action="store_true")
    arguments = parser.parse_args()
    print("physical_squint_width start", flush=True)
    refuse_spw5(spectral_window_id=4, opened=False)
    output_dir = arguments.output_dir.resolve()
    if output_dir.name in FROZEN_NAMES or output_dir == arguments.comparison_dir.resolve():
        raise RuntimeError(f"refusing to write into a frozen product path: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmp = _comparison()
    payload: dict[str, object] = {
        "artifact": PHYSICAL_SQUINT_EXPERIMENT,
        "development_only": True,
        "spw5_closed": True,
        "spw5_opened": False,
        "spw5_read": False,
        "full_jones_untouched": True,
        "native_squint_never_removed_by_default": True,
        "no_general_cband_beam_frozen": True,
        "command": list(sys.argv),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "code_sha256": _code_hash(),
    }
    try:
        catalog = HighresCassbeamCatalog(arguments.artifact_root)
        plane = catalog.plane(THOL0001_SPW4_CHANNEL_32_HZ)
        software = run_software_gates(plane)
        payload["software_gates"] = software
        payload["artifact_checksum"] = artifact_checksum_report(
            catalog, THOL0001_SPW4_CHANNEL_32_HZ
        )
        residual = cmp._residual_jones(arguments.product_dir)
        diag = cmp._diag()
        from sl1mjax.holography_ms import _read_antennas, _tables

        tables = _tables()
        _ids, names, positions = _read_antennas(tables, arguments.measurement_set)
        names = tuple(names)
        observation, chi, packed, intensity, voltage, rr_ok, ll_ok = cmp._load_holoraster_channel(
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
        voltage_rows = np.asarray(voltage, dtype=np.float64)[fields["rows"]]
        regions = region_masks_from_voltage(voltage_rows)
        rr_flags = np.asarray(rr_ok, dtype=bool)[fields["rows"]]
        ll_flags = np.asarray(ll_ok, dtype=bool)[fields["rows"]]
        lookup = catalog_diagonal_feed_lookup(catalog, float(frequencies[0]))
        convention = locked_convention()
        identity_rows = np.arange(fields["offset"].shape[0])
        if int(arguments.identity_max_rows) > 0:
            mover_p = np.flatnonzero(fields["moving_is_p"])
            mover_q = np.flatnonzero(~fields["moving_is_p"])
            half = max(int(arguments.identity_max_rows) // 2, 1)
            identity_rows = np.unique(np.concatenate([mover_p[:half], mover_q[:half]]))
        print("identity gate rows", int(identity_rows.size), flush=True)
        comparison = evaluate_holoraster_cassbeam(
            catalog=catalog,
            frequencies_hz=frequencies,
            convention=convention,
            offset_lm_rad=fields["offset"][identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            residual_jones=residual,
            source=source_rows
            if np.asarray(source_rows).ndim == 2
            else np.asarray(source_rows)[identity_rows],
            off_diagonal=False,
        )
        physical = physical_path_stages(
            fields["offset"][identity_rows],
            lookup,
            IDENTITY_PHYSICAL,
            residual_jones=residual,
            moving_id=fields["moving"][identity_rows],
            reference_id=fields["reference"][identity_rows],
            moving_is_p=fields["moving_is_p"][identity_rows],
            chi_moving=fields["chi_m"][identity_rows],
            chi_reference=fields["chi_r"][identity_rows],
            source=source_rows
            if np.asarray(source_rows).ndim == 2
            else np.asarray(source_rows)[identity_rows],
        )
        identity_report = require_physical_identity_matches_comparison(physical, comparison)
        print(json.dumps(cmp._jsonable(identity_report)), flush=True)
        if identity_rows.size == fields["offset"].shape[0]:
            baseline_vis = physical["visibility"]
        else:
            baseline_vis = physical_path_stages(
                fields["offset"],
                lookup,
                IDENTITY_PHYSICAL,
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
            baseline=baseline_vis,
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
        visits = visit_id_per_time(observation)
        time_index = np.asarray(geometry["time_index"], dtype=np.int32)[keep]
        visit = np.full(time_index.size, -1, dtype=np.int32)
        valid_time = time_index >= 0
        visit[valid_time] = np.asarray(visits, dtype=np.int32)[time_index[valid_time]]
        ref_hold = holdouts.reference_holdout[keep]
        thin = _thin_mask(
            samples,
            stride=int(arguments.stride),
            max_train_cells=int(arguments.max_train_cells),
        )
        if not bool(np.all(thin)):
            samples = _subset_samples(samples, thin, ref_hold)
            rr_flags = rr_flags[thin]
            ll_flags = ll_flags[thin]
            visit = visit[thin]
            print(
                "thinned rows", int(np.sum(thin)), "train", int(np.sum(samples.train)), flush=True
            )
        else:
            object.__setattr__(samples, "reference_holdout", ref_hold)
        print("direct training-map squint", flush=True)
        direct = measure_direct_squint(
            offset_lm_rad=samples.offset_lm_rad,
            measured=samples.measured,
            source=samples.source,
            residual_jones=residual,
            moving_id=samples.moving_id,
            reference_id=samples.reference_id,
            moving_is_p=samples.moving_is_p,
            chi_moving=samples.parallactic_angle_rad,
            chi_reference=samples.reference_parallactic_angle_rad,
            weight=samples.weight,
            rr_ok=rr_flags,
            ll_ok=ll_flags,
            train=samples.train,
            antenna_names=names,
            frequency_hz=float(frequencies[0]),
            visit_id=visit,
            n_boot=int(arguments.n_boot),
            seed=int(arguments.seed),
        )
        maps = direct.pop("maps")
        native_feed = apply_physical_feed_frame(maps["offset"], lookup, IDENTITY_PHYSICAL)
        cassbeam_maps = _normalize_origin_power(
            {
                "offset": maps["offset"],
                "rr_power": np.abs(native_feed[:, 0, 0]) ** 2,
                "ll_power": np.abs(native_feed[:, 1, 1]) ** 2,
                "weight": maps["weight"],
                "n_cells": maps["n_cells"],
            }
        )
        emp_state = candidate_state(
            "empirical",
            empirical_delta_rad=direct["delta_lm_rad"],
            width=1.0,
        )
        emp_feed = apply_physical_feed_frame(maps["offset"], lookup, emp_state)
        empirical_maps = _normalize_origin_power(
            {
                "offset": maps["offset"],
                "rr_power": np.abs(emp_feed[:, 0, 0]) ** 2,
                "ll_power": np.abs(emp_feed[:, 1, 1]) ** 2,
                "weight": maps["weight"],
                "n_cells": maps["n_cells"],
            }
        )
        cassbeam_sq = squint_from_common_map(
            cassbeam_maps, series="native_cassbeam_same_cells", frequency_hz=float(frequencies[0])
        )
        empirical_sq = squint_from_common_map(
            empirical_maps, series="empirical_same_cells", frequency_hz=float(frequencies[0])
        )
        measured_vec = np.asarray(direct["delta_lm_arcmin"], dtype=np.float64)
        model_vec = np.array(
            [
                float(empirical_sq["rr_l_arcmin"]) - float(empirical_sq["ll_l_arcmin"]),
                float(empirical_sq["rr_m_arcmin"]) - float(empirical_sq["ll_m_arcmin"]),
            ]
        )
        native_vec = np.array(
            [
                float(cassbeam_sq["rr_l_arcmin"]) - float(cassbeam_sq["ll_l_arcmin"]),
                float(cassbeam_sq["rr_m_arcmin"]) - float(cassbeam_sq["ll_m_arcmin"]),
            ]
        )
        native_cmd = native_separation_rad() / ARCMIN
        commanded = measured_vec - native_cmd
        model_change = model_vec - native_vec
        hand_centres = np.array(
            [
                cassbeam_sq["rr_l_arcmin"],
                cassbeam_sq["rr_m_arcmin"],
                cassbeam_sq["ll_l_arcmin"],
                cassbeam_sq["ll_m_arcmin"],
            ],
            dtype=np.float64,
        )
        support_ok = bool(np.all(np.isfinite(hand_centres)) and np.max(np.abs(hand_centres)) < 3.0)
        if float(np.hypot(*commanded)) <= 1.0e-6:
            algebra_ok = True
        else:
            algebra_ok = bool(np.dot(model_change, commanded) >= 0.0)
        centroid_direction_ok = bool((not support_ok) or algebra_ok)
        print(
            "centroid_gate",
            {
                "support_ok": support_ok,
                "algebra_ok": algebra_ok,
                "centroid_direction_ok": centroid_direction_ok,
            },
            flush=True,
        )
        print("fitting permitted width on train", flush=True)
        width_native = fit_width_on_train(
            samples, lookup, delta_lm_rad=IDENTITY_PHYSICAL.delta(), grid=WIDTH_GRID
        )
        width_empirical = fit_width_on_train(
            samples, lookup, delta_lm_rad=direct["delta_lm_rad"], grid=WIDTH_GRID
        )
        predictions: dict[str, np.ndarray] = {"native": samples.baseline}
        for name in CANDIDATE_NAMES:
            if name == "native":
                continue
            width = 1.0
            if name == "native_plus_width":
                width = float(width_native["width"])
            if name == "empirical_plus_width":
                width = float(width_empirical["width"])
            state = candidate_state(name, empirical_delta_rad=direct["delta_lm_rad"], width=width)
            print("predict", name, flush=True)
            predictions[name] = physical_path_stages(
                samples.offset_lm_rad,
                lookup,
                state,
                residual_jones=residual,
                moving_id=samples.moving_id,
                reference_id=samples.reference_id,
                moving_is_p=samples.moving_is_p,
                chi_moving=samples.parallactic_angle_rad,
                chi_reference=samples.reference_parallactic_angle_rad,
                source=samples.source,
            )["visibility"]
        score_kw = {
            "rr_ok": rr_flags,
            "ll_ok": ll_flags,
            "n_boot": int(arguments.n_boot),
            "seed": int(arguments.seed),
        }
        native_vs_none = score_model(
            samples, predictions["native"], predictions["no_squint"], **score_kw
        )
        empirical_vs_native = score_model(
            samples, predictions["empirical"], predictions["native"], **score_kw
        )
        width_native_vs_native = score_model(
            samples, predictions["native_plus_width"], predictions["native"], **score_kw
        )
        width_empirical_vs_empirical = score_model(
            samples, predictions["empirical_plus_width"], predictions["empirical"], **score_kw
        )
        versus_native = {
            name: score_model(samples, predictions[name], predictions["native"], **score_kw)
            for name in CANDIDATE_NAMES
        }
        interpretation = interpret_experiment(
            native_vs_none=native_vs_none,
            empirical_vs_native=empirical_vs_native,
            width_native_vs_native=width_native_vs_native,
            width_empirical_vs_empirical=width_empirical_vs_empirical,
            direct=direct,
            width_native=width_native,
            width_empirical=width_empirical,
            identity_ok=True,
            centroid_direction_ok=centroid_direction_ok,
        )
        joint = None
        perp = None
        if not bool(arguments.skip_joint):
            print("identifiability surfaces on train", flush=True)
            joint = joint_width_squint_surface(
                samples,
                lookup,
                width_grid=np.linspace(0.96, 1.10, 8),
                magnitude_scale_grid=np.linspace(0.0, 1.6, 9),
            )
            perp = perpendicular_squint_profile(
                samples,
                lookup,
                width=float(width_native["width"]),
                perp_arcmin_grid=np.linspace(-0.3, 0.3, 9),
            )
        row_accounting = {
            "n_usable_development": int(np.sum(keep)),
            "n_used": int(samples.train.size),
            "n_train": int(np.sum(samples.train)),
            "n_spatial_holdout": int(np.sum(samples.spatial_holdout)),
            "n_mover_holdout": int(np.sum(samples.mover_holdout)),
            "n_reference_holdout": int(np.sum(samples.reference_holdout)),
            "label": "SPW-4 development; holdouts are not an unseen validation set",
        }
        radial = {
            name: _radial_profile(
                samples.offset_lm_rad,
                samples.measured,
                predictions[name],
                samples.weight,
            )
            for name in (
                "no_squint",
                "native",
                "empirical",
                "native_plus_width",
                "empirical_plus_width",
            )
        }
        residual_maps = {
            name: _cell_residual_map(
                samples.offset_lm_rad,
                samples.measured,
                predictions[name],
                samples.weight,
            )
            for name in ("no_squint", "native", "empirical")
        }
        obs_pred = {
            "rr": {
                "measured": samples.measured[:, 0, 0],
                "predicted": predictions["native"][:, 0, 0],
            },
            "ll": {
                "measured": samples.measured[:, 1, 1],
                "predicted": predictions["native"][:, 1, 1],
            },
            "rr_ll": {
                "measured": samples.measured[:, 0, 0] - samples.measured[:, 1, 1],
                "predicted": predictions["native"][:, 0, 0] - predictions["native"][:, 1, 1],
            },
        }
        plots = _write_plots(
            output_dir,
            {
                "maps": maps,
                "cassbeam_maps": cassbeam_maps,
                "empirical_maps": empirical_maps,
                "direct": direct,
                "paired": {
                    "empirical_vs_native": empirical_vs_native,
                    "native_vs_none": native_vs_none,
                },
                "widths": {"native": width_native, "empirical": width_empirical},
                "joint": joint,
                "perp": perp,
                "radial": radial,
                "obs_pred": obs_pred,
                "residual_maps": residual_maps,
            },
        )
        payload.update(
            {
                "identity_report": identity_report,
                "interpretation": interpretation,
                "row_accounting": row_accounting,
                "source_revision": cmp._source_revision(),
                "plots": plots,
            }
        )
        write_json(cmp._jsonable(payload), output_dir / "report.json")
        write_json(cmp._jsonable(parameterization_record()), output_dir / "parameterization.json")
        write_json(cmp._jsonable(direct), output_dir / "direct_squint_measurement.json")
        write_json(
            cmp._jsonable(
                {
                    name: {
                        "width": (
                            width_native["width"]
                            if name == "native_plus_width"
                            else width_empirical["width"]
                            if name == "empirical_plus_width"
                            else 1.0
                        ),
                        "delta": (
                            [0.0, 0.0]
                            if name == "no_squint"
                            else list(direct["delta_lm_rad"])
                            if name.startswith("empirical")
                            else list(IDENTITY_PHYSICAL.delta())
                        ),
                    }
                    for name in CANDIDATE_NAMES
                }
            ),
            output_dir / "candidate_models.json",
        )
        write_json(
            cmp._jsonable(
                {
                    "versus_native": versus_native,
                    "native_vs_none": native_vs_none,
                    "empirical_vs_native": empirical_vs_native,
                    "width_native_vs_native": width_native_vs_native,
                    "width_empirical_vs_empirical": width_empirical_vs_empirical,
                }
            ),
            output_dir / "paired_scores.json",
        )
        write_json(
            cmp._jsonable(
                {
                    "width_native": width_native,
                    "width_empirical": width_empirical,
                    "centroid_direction_ok": centroid_direction_ok,
                    "centroid_support_ok": support_ok,
                    "centroid_algebra_ok": algebra_ok,
                    "native_same_cells": cassbeam_sq,
                    "empirical_same_cells": empirical_sq,
                    "joint_width_magnitude": joint,
                    "perpendicular_profile": perp,
                }
            ),
            output_dir / "identifiability.json",
        )
        write_json(
            cmp._jsonable(
                {
                    "spw5_opened": False,
                    "spw5_read": False,
                    "ready_for_later_transfer": bool(interpretation["spw5_ready"]),
                    "freeze_now": False,
                    "future_gate": (
                        "one-shot no-refit transfer of the frozen SPW-4 coefficients; "
                        "do not refit width or delta on SPW 5"
                    ),
                    "reason": interpretation["outcome"],
                }
            ),
            output_dir / "spw5_recommendation.json",
        )
        write_json(cmp._jsonable(row_accounting), output_dir / "row_accounting.json")
        np.savez_compressed(
            output_dir / "map_cells.npz",
            measured_offset=maps["offset"],
            measured_rr=maps["rr_power"],
            measured_ll=maps["ll_power"],
            cassbeam_rr=cassbeam_maps["rr_power"],
            cassbeam_ll=cassbeam_maps["ll_power"],
            empirical_rr=empirical_maps["rr_power"],
            empirical_ll=empirical_maps["ll_power"],
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
