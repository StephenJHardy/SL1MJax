"""Deterministic publication plots from the compact validation bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam_validation_outputs import ValidationBundle, sha256_file, write_json
from sl1mjax.beam_validation_statistics import (
    frequency_copolar_series,
    publication_squint_pair,
    residual_power_table,
)

FIGURE_IDS = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F08",
    "F10",
    "F12",
    "F13",
    "F16",
    "F17",
    "F18",
    "F19",
)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_figure_sidecar(
    image_path: Path,
    *,
    figure_id: str,
    function: str,
    claims: Sequence[str],
    table: Mapping[str, object],
    bundle_sha256: str | None,
    caption: str,
) -> Path:
    payload = {
        "figure_id": figure_id,
        "filename": image_path.name,
        "image_sha256": sha256_file(image_path),
        "generating_function": function,
        "claims": list(claims),
        "bundle_sha256": bundle_sha256,
        "created_utc": datetime.now(UTC).isoformat(),
        "caption": caption,
        "table": _jsonable(table),
        "table_sha256": hashlib.sha256(
            json.dumps(_jsonable(table), sort_keys=True).encode()
        ).hexdigest(),
    }
    return write_json(payload, image_path.with_suffix(image_path.suffix + ".sidecar.json"))


def _save(
    fig,
    output_dir: Path,
    filename: str,
    *,
    figure_id: str,
    function: str,
    claims: Sequence[str],
    table: Mapping[str, object],
    bundle_sha256: str | None,
    caption: str,
) -> list[Path]:
    plt = _pyplot()
    destination = Path(output_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    sidecar = write_figure_sidecar(
        destination,
        figure_id=figure_id,
        function=function,
        claims=claims,
        table=table,
        bundle_sha256=bundle_sha256,
        caption=caption,
    )
    return [destination, sidecar]


def plot_observation_timeline(
    observation: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8.5, 3.2))
    rows = list(observation.get("timeline") or ())
    for index, row in enumerate(rows):
        axis.barh(
            0,
            float(row["n_scans"]),
            left=float(row["scan_start"]),
            height=0.45,
            label=str(row["label"]),
        )
        axis.text(
            float(row["scan_start"]) + 0.4,
            0.35 + 0.18 * (index % 2),
            str(row["label"]),
            fontsize=8,
        )
    axis.set_yticks([])
    axis.set_xlabel("scan number")
    axis.set_title("THOL0001 lower-C observation anchors (SPW 4 only)")
    table = {"timeline": list(rows)}
    return _save(
        fig,
        output_dir,
        "01_observation_timeline.png",
        figure_id="F01",
        function="plot_observation_timeline",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Observation and calibration scan anchors. SPW 5 is sealed. Supports C01.",
    )


def plot_raster_occupancy(
    occupancy: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.2, 4.8))
    count = np.asarray(occupancy["count"], dtype=np.float64)
    l_ax = np.asarray(occupancy["l_arcmin"], dtype=np.float64)
    m_ax = np.asarray(occupancy["m_arcmin"], dtype=np.float64)
    display = np.ma.masked_where(count <= 0.0, count)
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    image = axis.imshow(display, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    axis.set_xlabel("l (arcmin)")
    axis.set_ylabel("m (arcmin)")
    axis.set_title("HOLORASTER measured occupancy")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="weighted samples")
    table = {
        "n_occupied": int(np.sum(count > 0.0)),
        "n_total": int(count.size),
        "max_count": float(np.max(count)),
    }
    return _save(
        fig,
        output_dir,
        "02_raster_occupancy.png",
        figure_id="F02",
        function="plot_raster_occupancy",
        claims=("C01", "C03"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Measured HOLORASTER occupancy. Unsupported cells are masked. Supports C01, C03.",
    )


def plot_antenna_roles(
    observation: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    names = list(observation.get("antenna_names") or ())
    references = set(observation.get("reference_antenna_names") or ())
    colors = ["C1" if name in references else "C0" for name in names]
    axis.bar(np.arange(len(names)), np.ones(len(names)), color=colors)
    axis.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=7)
    axis.set_yticks([])
    axis.set_title("Moving antennas (blue) and holography references (orange)")
    table = {
        "antenna_names": names,
        "reference_antenna_names": sorted(references),
        "n_moving": int(len(names) - len(references)),
    }
    return _save(
        fig,
        output_dir,
        "03_antenna_roles.png",
        figure_id="F03",
        function="plot_antenna_roles",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Holography reference set versus moving antennas. Supports C01.",
    )


def plot_convention_residuals(
    convention_gates: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.4, 3.6))
    stages = list((convention_gates.get("cumulative_residuals") or {}).items())
    labels = [name for name, _ in stages]
    medians = [float(row["median_rel_l2"]) for _, row in stages]
    axis.plot(np.arange(len(labels)), medians, "o-")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("median relative L2")
    axis.set_title("CASA/JAX cumulative operator residual (software semantics)")
    table = {"stages": labels, "median_rel_l2": medians}
    return _save(
        fig,
        output_dir,
        "04_casa_jax_operator_residual.png",
        figure_id="F04",
        function="plot_convention_residuals",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Software apply-contract residuals. Not physical beam evidence. Supports C01.",
    )


def plot_onaxis_amp(
    scatter: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for axis, hand in zip(axes, ("rr", "ll"), strict=True):
        obs = np.asarray(scatter[f"{hand}_obs_abs"], dtype=np.float64)
        pred = np.asarray(scatter[f"{hand}_pred_abs"], dtype=np.float64)
        onaxis = np.asarray(scatter[f"{hand}_onaxis"], dtype=bool)
        axis.hist(obs[onaxis], bins=24, alpha=0.55, label="measured")
        axis.hist(pred[onaxis], bins=24, alpha=0.55, label="CASSBEAM")
        axis.set_title(f"{hand.upper()} on-axis |V|")
        axis.set_xlabel("Jy")
        axis.legend(fontsize=8)
    table = {
        "n_onaxis_rr": int(np.sum(scatter["rr_onaxis"])),
        "n_onaxis_ll": int(np.sum(scatter["ll_onaxis"])),
        "median_obs_rr": float(
            np.median(scatter["rr_obs_abs"][np.asarray(scatter["rr_onaxis"])])
        ),
        "median_pred_rr": float(
            np.median(scatter["rr_pred_abs"][np.asarray(scatter["rr_onaxis"])])
        ),
    }
    return _save(
        fig,
        output_dir,
        "05_onaxis_amplitude.png",
        figure_id="F05",
        function="plot_onaxis_amp",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="On-axis absolute |V| for measured HOLORASTER and CASSBEAM. Supports C01.",
    )


def plot_cassbeam_scatter(
    scatter: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
    region: str = "main_lobe",
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    mask_key = f"{region}_mask"
    choose = np.asarray(scatter[mask_key], dtype=bool) if mask_key in scatter else np.ones(
        scatter["rr_obs_real"].shape, dtype=bool
    )
    panels = (
        (axes[0, 0], "rr", "real", "RR Re (Jy)"),
        (axes[0, 1], "rr", "imag", "RR Im (Jy)"),
        (axes[1, 0], "ll", "real", "LL Re (Jy)"),
        (axes[1, 1], "ll", "imag", "LL Im (Jy)"),
    )
    table: dict[str, object] = {"region": region, "n": int(np.sum(choose))}
    for axis, hand, part, label in panels:
        obs = np.asarray(scatter[f"{hand}_obs_{part}"], dtype=np.float64)[choose]
        pred = np.asarray(scatter[f"{hand}_pred_{part}"], dtype=np.float64)[choose]
        axis.scatter(pred, obs, s=4, alpha=0.25, linewidths=0)
        finite = np.isfinite(obs) & np.isfinite(pred)
        if bool(np.any(finite)):
            lo = float(min(np.min(pred[finite]), np.min(obs[finite])))
            hi = float(max(np.max(pred[finite]), np.max(obs[finite])))
            axis.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        axis.set_xlabel(f"CASSBEAM {label}")
        axis.set_ylabel(f"observed {label}")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{hand.upper()} {part} ({region.replace('_', ' ')})")
    return _save(
        fig,
        output_dir,
        f"08_cassbeam_scatter_{region}.png",
        figure_id="F08",
        function="plot_cassbeam_scatter",
        claims=("C01", "C02"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=f"Observed versus diagonal CASSBEAM {region.replace('_', ' ')}. Supports C01, C02.",
    )


def plot_spatial_residual_maps(
    maps: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    l_ax = np.asarray(maps["l_arcmin"], dtype=np.float64)
    m_ax = np.asarray(maps["m_arcmin"], dtype=np.float64)
    weight = np.asarray(maps["weight"], dtype=np.float64)
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    support = weight > 0.0
    titles = (
        ("rr_measured", "RR measured"),
        ("rr_cassbeam", "RR CASSBEAM"),
        ("rr_residual", "RR residual"),
        ("ll_measured", "LL measured"),
        ("ll_cassbeam", "LL CASSBEAM"),
        ("ll_residual", "LL residual"),
    )
    table = {"n_supported": int(np.sum(support))}
    for axis, (key, title) in zip(axes.ravel(), titles, strict=True):
        grid = np.abs(np.asarray(maps[key]))
        display = np.ma.masked_where(~support, grid)
        cmap = "magma" if "residual" in key else "viridis"
        image = axis.imshow(display, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axis.set_title(title)
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(
        fig,
        output_dir,
        "10_spatial_measured_cassbeam_residual.png",
        figure_id="F10",
        function="plot_spatial_residual_maps",
        claims=("C01", "C02", "C03"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Visibility-domain RR/LL maps. Unsupported cells are masked. Supports C01–C03.",
    )


def plot_residual_radius(
    geometry: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for axis, key, xlabel in (
        (axes[0], "median_vs_radius", "radius (arcmin)"),
        (axes[1], "median_vs_azimuth", "azimuth (deg)"),
    ):
        for hand, color in (("rr", "C0"), ("ll", "C1")):
            series = (geometry.get(hand) or {}).get(key) or {}
            axis.plot(
                series.get("x") or (),
                series.get("median") or (),
                "o-",
                color=color,
                label=hand.upper(),
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("median |residual| (Jy)")
        axis.legend(fontsize=8)
    axes[0].axvline(10.0, color="k", ls="--", lw=0.8, label="10′ ring")
    table = {hand: geometry.get(hand, {}).get("median_vs_radius") for hand in ("rr", "ll")}
    return _save(
        fig,
        output_dir,
        "12_residual_vs_radius.png",
        figure_id="F12",
        function="plot_residual_radius",
        claims=("C02", "C03"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Median copolar residual versus radius and azimuth. "
            "The 10′ ring is marked. Supports C02, C03."
        ),
    )


def plot_publication_squint(
    squint: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    pair = publication_squint_pair(squint)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    table: dict[str, object] = {}
    for axis, name in zip(axes, ("measured", "cassbeam"), strict=True):
        record = pair[name]
        axis.scatter(
            [float(record["rr_l_arcmin"])],
            [float(record["rr_m_arcmin"])],
            marker="o",
            label="RR",
        )
        axis.scatter(
            [float(record["ll_l_arcmin"])],
            [float(record["ll_m_arcmin"])],
            marker="s",
            label="LL",
        )
        axis.plot(
            [float(record["rr_l_arcmin"]), float(record["ll_l_arcmin"])],
            [float(record["rr_m_arcmin"]), float(record["ll_m_arcmin"])],
            "k-",
            lw=0.8,
        )
        axis.set_title(f"{name} 20%-of-peak  {float(record['separation_arcmin']):.3f}′")
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0.0, color="k", lw=0.4)
        axis.axvline(0.0, color="k", lw=0.4)
        axis.legend(fontsize=8)
        table[name] = {
            "separation_arcmin": float(record["separation_arcmin"]),
            "rr_l_arcmin": float(record["rr_l_arcmin"]),
            "rr_m_arcmin": float(record["rr_m_arcmin"]),
            "ll_l_arcmin": float(record["ll_l_arcmin"]),
            "ll_m_arcmin": float(record["ll_m_arcmin"]),
        }
    return _save(
        fig,
        output_dir,
        "13_squint_mainlobe_20pct.png",
        figure_id="F13",
        function="plot_publication_squint",
        claims=("C07",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Publication 20%-of-peak squint. "
            "Full-raster centroids are not shown. Supports C07."
        ),
    )


def plot_frequency_series(
    frequency: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    rows = frequency_copolar_series(frequency)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    freq = [row["frequency_hz"] / 1e9 for row in rows]
    axes[0].plot(freq, [row["rr_correlation"] for row in rows], "o-", label="RR")
    axes[0].plot(freq, [row["ll_correlation"] for row in rows], "s-", label="LL")
    axes[0].set_ylabel("|correlation|")
    axes[0].set_title("Copolar correlation")
    axes[1].plot(freq, [row["rr_residual_power"] for row in rows], "o-", label="RR")
    axes[1].plot(freq, [row["ll_residual_power"] for row in rows], "s-", label="LL")
    axes[1].set_ylabel("residual power")
    axes[1].set_title("Copolar residual power")
    for axis in axes:
        axis.set_xlabel("frequency (GHz)")
        axis.legend(fontsize=8)
        axis.axvline(4.564, color="k", ls="--", lw=0.7)
    table = {"channels": rows}
    return _save(
        fig,
        output_dir,
        "16_spw4_frequency.png",
        figure_id="F16",
        function="plot_frequency_series",
        claims=("C01", "C02"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="SPW-4 channel dependence. Frequency squint is omitted. Supports C01, C02.",
    )


def plot_offset_ring(
    fields: Mapping[str, Any],
    closure: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    records = list(fields.get("fields") or ())
    partitions = {
        name: set(ids)
        for name, ids in (fields.get("partition") or {}).items()
        if name != "notes"
    }
    colors = {
        "training": "C0",
        "inner_holdout": "C2",
        "sealed_holdout": "C3",
    }
    for record in records:
        field_id = int(record["field_id"])
        label = "training"
        for name, ids in partitions.items():
            if field_id in {int(item) for item in ids}:
                label = name
                break
        axes[0].scatter(
            [float(record["l_arcmin"])],
            [float(record["m_arcmin"])],
            color=colors.get(label, "k"),
            label=label if label not in axes[0].get_legend_handles_labels()[1] else None,
        )
        axes[0].annotate(
            str(record["name"]),
            (float(record["l_arcmin"]), float(record["m_arcmin"])),
            fontsize=7,
        )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("l (arcmin)")
    axes[0].set_ylabel("m (arcmin)")
    axes[0].set_title("C147-* unused prediction ring")
    axes[0].legend(fontsize=7)
    axes[1].bar(["RR", "LL"], [float(closure["rr"]), float(closure["ll"])], color=["C0", "C1"])
    axes[1].set_ylabel("diagonal residual power")
    axes[1].set_title("Channel-32 RR/LL closure")
    table = {"fields": records, "closure": dict(closure)}
    return _save(
        fig,
        output_dir,
        "17_c147_offset_ring.png",
        figure_id="F17",
        function="plot_offset_ring",
        claims=("C04", "C05"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Independent C147-* ring geometry and diagonal closure. Supports C04.",
    )


def plot_crosshand_null(
    scatter: Mapping[str, NDArray],
    quadrants: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    axes[0].scatter(
        scatter["rl_pred_real"],
        scatter["rl_obs_real"],
        s=4,
        alpha=0.2,
        label="experimental full Jones",
    )
    axes[0].scatter(
        scatter["rl_diag_real"],
        scatter["rl_obs_real"],
        s=4,
        alpha=0.2,
        label="diagonal null",
    )
    lo = float(np.nanmin(scatter["rl_obs_real"]))
    hi = float(np.nanmax(scatter["rl_obs_real"]))
    axes[0].plot([lo, hi], [lo, hi], "k-", lw=0.7)
    axes[0].set_title("RL real")
    axes[0].set_xlabel("predicted Re (Jy)")
    axes[0].set_ylabel("observed Re (Jy)")
    axes[0].legend(fontsize=7, loc="upper left")
    names = ("east", "west", "north", "south", "east_west", "north_south")
    labels = [name for name in names if name in quadrants]
    corrs = [
        float((quadrants[name].get("rl") or {}).get("correlation_abs") or np.nan)
        for name in labels
    ]
    axes[1].bar(np.arange(len(labels)), corrs)
    axes[1].set_xticks(
        np.arange(len(labels)),
        [name.replace("_", "\n") for name in labels],
        fontsize=7,
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("RL |correlation| by quadrant")
    table = {"quadrant_rl_correlation": dict(zip(labels, corrs, strict=True))}
    return _save(
        fig,
        output_dir,
        "18_experimental_crosshand.png",
        figure_id="F18",
        function="plot_crosshand_null",
        claims=("C05",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption="Experimental RL prediction versus the observed cloud. Supports C05.",
    )


def plot_spw5_sealed(output_dir: Path, *, bundle_sha256: str | None = None) -> list[Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(6.4, 2.6))
    axis.axis("off")
    axis.text(0.5, 0.55, "F19 reserved", ha="center", va="center", fontsize=16)
    axis.text(0.5, 0.28, "SPW 5 is sealed. Status: not_run.", ha="center", va="center", fontsize=10)
    return _save(
        fig,
        output_dir,
        "19_spw5_sealed.png",
        figure_id="F19",
        function="plot_spw5_sealed",
        claims=("C06",),
        table={"status": "not_run"},
        bundle_sha256=bundle_sha256,
        caption="SPW 5 remains sealed. Supports C06.",
    )


def write_all_figures(bundle: ValidationBundle, output_dir: Path) -> list[Path]:
    """Write every version-1 figure and sidecar from the bundle."""

    from sl1mjax.beam_validation_statistics import offset_ring_diagonal_closure

    bundle_sha = str(bundle.manifest.get("bundle_sha256") or "")
    scatter = bundle.plot_table("holoraster_scatter.npz")
    maps = bundle.plot_table("holoraster_maps.npz")
    occupancy = bundle.plot_table("raster_occupancy.npz")
    written: list[Path] = []
    written.extend(
        plot_observation_timeline(bundle.observation, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(plot_raster_occupancy(occupancy, output_dir, bundle_sha256=bundle_sha))
    written.extend(plot_antenna_roles(bundle.observation, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_convention_residuals(bundle.convention_gates, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(plot_onaxis_amp(scatter, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_cassbeam_scatter(scatter, output_dir, bundle_sha256=bundle_sha, region="main_lobe")
    )
    written.extend(plot_spatial_residual_maps(maps, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_residual_radius(bundle.residual_geometry, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(plot_publication_squint(bundle.squint, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_frequency_series(bundle.holoraster_frequency, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(
        plot_offset_ring(
            bundle.offset_ring_fields,
            offset_ring_diagonal_closure(bundle.offset_ring),
            output_dir,
            bundle_sha256=bundle_sha,
        )
    )
    written.extend(
        plot_crosshand_null(
            scatter,
            bundle.crosshand_quadrants,
            output_dir,
            bundle_sha256=bundle_sha,
        )
    )
    written.extend(plot_spw5_sealed(output_dir, bundle_sha256=bundle_sha))
    write_json(
        {
            "bundle_sha256": bundle_sha,
            "figures": [path.name for path in written if path.suffix == ".png"],
            "region_residual_power": residual_power_table(bundle.holoraster_channel32),
        },
        Path(output_dir) / "figure_manifest.json",
    )
    written.append(Path(output_dir) / "figure_manifest.json")
    return written
