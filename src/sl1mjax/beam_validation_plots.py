"""Deterministic publication plots from the compact validation bundle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam_validation_outputs import ValidationBundle, sha256_file, write_json
from sl1mjax.beam_validation_statistics import (
    amplitude_db,
    channel32_source_i_jy,
    frequency_copolar_series,
    map_axis_cut,
    phase_valid_mask,
    publication_squint_pair,
    residual_power_table,
    wrapped_phase_residual,
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
    "F15",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "F25",
    "F13",
    "F16",
    "F17",
    "F18",
    "F19",
    "F26",
    "F27",
    "F28",
    "F29",
)

_SIDECAR_PROVENANCE: ContextVar[dict[str, object] | None] = ContextVar(
    "sidecar_provenance", default=None
)


def sidecar_provenance(bundle: ValidationBundle) -> dict[str, object]:
    """Bundle-level sidecar fields required by the publication refresh."""

    channel32 = bundle.holoraster_channel32
    manifest = bundle.manifest
    return {
        "model": str(channel32.get("model") or "evla_c_source_lm"),
        "coordinate_convention": str(
            channel32.get("coordinate_query") or "source_lm_feed"
        ),
        "coordinate_query": str(channel32.get("coordinate_query") or "source_lm_feed"),
        "calibration_state": (
            "HOLORASTER field 10 CORRECTED_DATA; residual Jones is the "
            "field-9 native channel-32 plane"
        ),
        "source_state": "per-row MODEL_DATA / resolved 3C147 I, Q, U, V",
        "frequency_hz": channel32.get("frequency_hz"),
        "split": "SPW-4 development rows unless a figure names another mask",
        "region": "named per figure in table.region / table.mask",
        "estimator": (
            "complex residual power and the plotted statistic named in table"
        ),
        "n_rows": channel32.get("n_rows") or channel32.get("n_development"),
        "artifact_hashes": {
            "bundle_sha256": manifest.get("bundle_sha256"),
            "revision": manifest.get("revision"),
            "publication_version": manifest.get("publication_version"),
        },
        "development_only": True,
        "spw5_opened": False,
        "publication_version": manifest.get("publication_version"),
    }


COORDINATE_FEED_MODELS = (
    ("generic_commanded", "generic / commanded", "0.55"),
    ("generic_source_lm", "generic / source-in-beam", "C1"),
    ("evla_c_source_lm", "EVLA-C / source-in-beam", "C2"),
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
        **{
            key: value
            for key, value in (_SIDECAR_PROVENANCE.get() or {}).items()
            if key not in {"table", "claims", "figure_id"}
        },
    }
    if "region" in table:
        payload["region"] = table["region"]
    if "mask" in table:
        payload["support_mask"] = table["mask"]
    if "estimator" in table:
        payload["estimator"] = table["estimator"]
    for count_key in ("n_rows", "n", "n_samples"):
        if count_key in table:
            payload["n_rows"] = table[count_key]
            break
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
    if not fig.get_constrained_layout():
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
        scale = float(np.asarray(scatter.get("source_i_jy", 1.0)).reshape(-1)[0])
        onaxis = np.asarray(scatter[f"{hand}_onaxis"], dtype=bool)
        axis.hist(obs[onaxis] / scale, bins=24, alpha=0.55, label="measured")
        axis.hist(pred[onaxis] / scale, bins=24, alpha=0.55, label="CASSBEAM")
        axis.axvline(1.0, color="k", ls="--", lw=0.8, label="boresight")
        axis.set_title(f"{hand.upper()} on-axis |V|/I")
        axis.set_xlabel("V / I_model")
        axis.legend(fontsize=8)
    scale = float(np.asarray(scatter.get("source_i_jy", 1.0)).reshape(-1)[0])
    table = {
        "source_i_jy": scale,
        "n_onaxis_rr": int(np.sum(scatter["rr_onaxis"])),
        "n_onaxis_ll": int(np.sum(scatter["ll_onaxis"])),
        "median_obs_rr": float(
            np.median(scatter["rr_obs_abs"][np.asarray(scatter["rr_onaxis"])])
        ),
        "median_pred_rr": float(
            np.median(scatter["rr_pred_abs"][np.asarray(scatter["rr_onaxis"])])
        ),
        "median_obs_rr_over_i": float(
            np.median(scatter["rr_obs_abs"][np.asarray(scatter["rr_onaxis"])]) / scale
        ),
        "median_pred_rr_over_i": float(
            np.median(scatter["rr_pred_abs"][np.asarray(scatter["rr_onaxis"])]) / scale
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
        caption=(
            "On-axis |V|/I_model. Boresight is marked at one. Supports C01."
        ),
    )


def plot_cassbeam_scatter(
    scatter: Mapping[str, NDArray],
    output_dir: Path,
    *,
    cells: Mapping[str, NDArray] | None = None,
    bundle_sha256: str | None = None,
    region: str = "main_lobe",
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0))
    mask_key = f"{region}_mask"
    choose = (
        np.asarray(scatter[mask_key], dtype=bool)
        if mask_key in scatter
        else np.ones(scatter["rr_obs_real"].shape, dtype=bool)
    )
    scale = float(np.asarray(scatter.get("source_i_jy", 1.0)).reshape(-1)[0])
    panels = (
        (axes[0, 0], "rr", "abs", "|RR| / I"),
        (axes[0, 1], "ll", "abs", "|LL| / I"),
        (axes[1, 0], "rr", "real", "Re(RR) / I"),
        (axes[1, 1], "ll", "real", "Re(LL) / I"),
    )
    table: dict[str, object] = {
        "region": region,
        "n": int(np.sum(choose)),
        "source_i_jy": scale,
        "mask": "V/I_model >= 0.5",
    }
    for axis, hand, part, label in panels:
        obs = np.asarray(scatter[f"{hand}_obs_{part}"], dtype=np.float64)[choose] / scale
        pred = np.asarray(scatter[f"{hand}_pred_{part}"], dtype=np.float64)[choose] / scale
        axis.scatter(pred, obs, s=3, alpha=0.12, linewidths=0, color="0.6")
        if cells is not None and f"{hand}_pred_{part}" in cells:
            cell_pred = np.asarray(cells[f"{hand}_pred_{part}"], dtype=np.float64)
            cell_obs = np.asarray(cells[f"{hand}_obs_{part}"], dtype=np.float64)
            cell_err = np.asarray(cells[f"{hand}_err_{part}"], dtype=np.float64)
            if cell_pred.size:
                axis.errorbar(
                    cell_pred,
                    cell_obs,
                    yerr=cell_err,
                    fmt="o",
                    ms=4,
                    alpha=0.9,
                    color="C0",
                    ecolor="C0",
                    elinewidth=0.7,
                    label="cell mean ± std",
                )
        finite = np.isfinite(obs) & np.isfinite(pred)
        if bool(np.any(finite)):
            lo = float(min(np.min(pred[finite]), np.min(obs[finite])))
            hi = float(max(np.max(pred[finite]), np.max(obs[finite])))
            axis.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        if part == "abs":
            axis.scatter([1.0], [1.0], marker="+", s=80, color="k", zorder=5)
            axis.axhline(1.0, color="k", ls=":", lw=0.5)
            axis.axvline(1.0, color="k", ls=":", lw=0.5)
        axis.set_xlabel(f"CASSBEAM {label}")
        axis.set_ylabel(f"observed {label}")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{hand.upper()} {part} ({region.replace('_', ' ')})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=7, loc="upper left")
    return _save(
        fig,
        output_dir,
        f"08_cassbeam_scatter_{region}.png",
        figure_id="F08",
        function="plot_cassbeam_scatter",
        claims=("C01", "C02"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Main-lobe V/I_model using the scientific voltage mask. "
            "Grey points are raw visibilities; blue points are spatial-cell "
            "means with within-cell standard deviation. Boresight is (1, 1). "
            "Supports C01, C02."
        ),
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
        caption=(
            "Visibility-domain RR/LL |V| on a linear Jy scale. "
            "Sidelobes are compressed; see F20 for dB. Supports C01–C03."
        ),
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


def plot_residual_strata(
    strata: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0))
    scale = float(strata.get("source_i_jy") or 1.0)
    panels = (
        ("mover", "Moving antenna"),
        ("reference", "Reference antenna"),
        ("pass", "Raster family"),
    )
    table: dict[str, object] = {
        "source_i_jy": scale,
        "mask": strata.get("mask"),
        "raster_family": strata.get("raster_family"),
    }
    for axis, (key, title) in zip(axes, panels, strict=True):
        rr_rows = list(strata.get(key) or ())
        ll_rows = list(strata.get(f"{key}_ll") or ())
        names = [str(row.get("name") or row.get("id")) for row in rr_rows]
        if not names:
            names = [str(row.get("name") or row.get("id")) for row in ll_rows]
        by_ll = {str(row.get("name") or row.get("id")): row for row in ll_rows}
        rr = [float(row["median_abs"]) / scale for row in rr_rows]
        ll = [float((by_ll.get(name) or {}).get("median_abs", np.nan)) / scale for name in names]
        if not names:
            axis.set_title(title)
            continue
        index = np.arange(len(names))
        width = 0.4
        axis.bar(index - width / 2, rr, width=width, label="RR")
        axis.bar(index + width / 2, ll, width=width, label="LL")
        axis.set_xticks(index, names, rotation=90, fontsize=7)
        axis.set_ylabel("median |ΔV| / I")
        axis.set_title(title)
        axis.legend(fontsize=7)
        table[key] = rr_rows
        table[f"{key}_ll"] = ll_rows
    return _save(
        fig,
        output_dir,
        "15_residual_strata.png",
        figure_id="F15",
        function="plot_residual_strata",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Main-lobe median |ΔV|/I by mover, reference, and raster family. "
            "Raster family is nearest Memo 195 dense versus sparse occupancy, "
            "not a scan-id join. Supports C01."
        ),
    )


def plot_amplitude_db_maps(
    maps: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
    source_i_jy: float | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2))
    l_ax = np.asarray(maps["l_arcmin"], dtype=np.float64)
    m_ax = np.asarray(maps["m_arcmin"], dtype=np.float64)
    weight = np.asarray(maps["weight"], dtype=np.float64)
    support = weight > 0.0
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    peak = float(source_i_jy if source_i_jy is not None else channel32_source_i_jy())
    table = {"peak_jy": peak, "db_floor": -40.0}
    for row, hand in enumerate(("rr", "ll")):
        meas = np.asarray(maps[f"{hand}_measured"])
        pred = np.asarray(maps[f"{hand}_cassbeam"])
        panels = (
            (amplitude_db(meas, peak), "viridis", f"{hand.upper()} measured (dB)", (-40.0, 0.0)),
            (amplitude_db(pred, peak), "viridis", f"{hand.upper()} CASSBEAM (dB)", (-40.0, 0.0)),
            (
                20.0
                * np.log10(np.maximum(np.abs(meas) / np.maximum(np.abs(pred), 1.0e-3), 1.0e-2)),
                "coolwarm",
                f"{hand.upper()} 20 log10(|V|/|C|)",
                (-8.0, 8.0),
            ),
        )
        for axis, (grid, cmap, title, limits) in zip(axes[row], panels, strict=True):
            display = np.ma.masked_where(~support, grid)
            image = axis.imshow(
                display,
                origin="lower",
                extent=extent,
                cmap=cmap,
                aspect="equal",
                vmin=limits[0],
                vmax=limits[1],
            )
            axis.set_title(title)
            axis.set_xlabel("l (arcmin)")
            axis.set_ylabel("m (arcmin)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save(
        fig,
        output_dir,
        "20_amplitude_db.png",
        figure_id="F20",
        function="plot_amplitude_db_maps",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Normalized |V|/I in dB from 0 to −40. The ratio panel is "
            "20 log10(|measured|/|CASSBEAM|). Where CASSBEAM is 0.05–0.15 Jy "
            "and the residual floor is ~0.2 Jy, ratios are biased upward and "
            "are not a multiplicative correction. Supports C03."
        ),
    )


def _draw_signed_cut(
    axis,
    maps: Mapping[str, NDArray],
    kind: str,
    scale: float,
    *,
    min_abs_arcmin: float = 0.0,
) -> None:
    for hand, color in (("rr", "C0"), ("ll", "C1")):
        obs = map_axis_cut(maps[f"{hand}_measured"], maps["l_arcmin"], maps["m_arcmin"], kind=kind)
        pred = map_axis_cut(maps[f"{hand}_cassbeam"], maps["l_arcmin"], maps["m_arcmin"], kind=kind)
        x = np.asarray(obs["x_arcmin"], dtype=np.float64)
        keep = np.abs(x) >= float(min_abs_arcmin)
        x = x[keep]
        axis.plot(
            x,
            np.asarray(obs["value"]).real[keep] / scale,
            color=color,
            lw=1.1,
            label=f"{hand.upper()} Re meas",
        )
        axis.plot(
            x,
            np.asarray(pred["value"]).real[keep] / scale,
            color=color,
            ls="--",
            lw=1.0,
            label=f"{hand.upper()} Re CASS",
        )
        axis.plot(
            x,
            np.asarray(obs["value"]).imag[keep] / scale,
            color=color,
            lw=0.7,
            alpha=0.55,
        )
        axis.plot(
            x,
            np.asarray(pred["value"]).imag[keep] / scale,
            color=color,
            ls=":",
            lw=0.8,
            alpha=0.7,
        )
    axis.axhline(0.0, color="k", lw=0.5)
    axis.set_xlabel("offset (arcmin)")
    axis.set_ylabel("V / I")


def plot_signed_complex_cuts(
    maps: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
    source_i_jy: float | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.6))
    scale = float(source_i_jy if source_i_jy is not None else channel32_source_i_jy())
    cuts = (("m0", "m=0"), ("l0", "l=0"), ("diag", "diagonal"), ("antidiag", "anti-diagonal"))
    table: dict[str, object] = {
        "source_i_jy": scale,
        "outer_min_arcmin": 15.0,
        "symlog_linthresh": 0.05,
    }
    for axis, (kind, title) in zip(axes[:2].ravel(), cuts, strict=True):
        _draw_signed_cut(axis, maps, kind, scale)
        axis.set_yscale("symlog", linthresh=0.05)
        axis.set_title(f"{title} (symlog)")
    for axis, (kind, title) in zip(axes[2:].ravel(), cuts, strict=True):
        _draw_signed_cut(axis, maps, kind, scale, min_abs_arcmin=15.0)
        axis.set_ylim(-0.12, 0.12)
        axis.set_title(f"{title} (|offset| ≥ 15′)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    return _save(
        fig,
        output_dir,
        "21_signed_complex_cuts.png",
        figure_id="F21",
        function="plot_signed_complex_cuts",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Signed Re/Im voltage cuts. Top: symmetric-log so distant lobes "
            "remain visible. Bottom: linear outer-only inset (|offset| ≥ 15′). "
            "Solid/dashed are measured/CASSBEAM real; lighter/dotted are "
            "imaginary. Supports C03."
        ),
    )


def plot_masked_phase_maps(
    maps: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.2))
    l_ax = np.asarray(maps["l_arcmin"], dtype=np.float64)
    m_ax = np.asarray(maps["m_arcmin"], dtype=np.float64)
    weight = np.asarray(maps["weight"], dtype=np.float64)
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    table: dict[str, object] = {"amp_floor_jy": 0.05, "phase_status": "exploratory"}
    for row, hand in enumerate(("rr", "ll")):
        meas = np.asarray(maps[f"{hand}_measured"])
        pred = np.asarray(maps[f"{hand}_cassbeam"])
        valid = phase_valid_mask(meas, pred, weight)
        residual = wrapped_phase_residual(meas, pred)
        table[f"{hand}_n_phase"] = int(np.sum(valid))
        if bool(np.any(valid)):
            table[f"{hand}_median_phase_deg"] = float(np.degrees(np.median(residual[valid])))
        panels = (
            (np.angle(meas), f"{hand.upper()} measured phase"),
            (np.angle(pred), f"{hand.upper()} CASSBEAM phase"),
            (residual, f"{hand.upper()} arg(E_m E_c*)"),
        )
        for axis, (grid, title) in zip(axes[row], panels, strict=True):
            display = np.ma.masked_where(~valid, np.degrees(grid))
            image = axis.imshow(
                display,
                origin="lower",
                extent=extent,
                cmap="twilight",
                aspect="equal",
                vmin=-180.0,
                vmax=180.0,
            )
            axis.set_title(title)
            axis.set_xlabel("l (arcmin)")
            axis.set_ylabel("m (arcmin)")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="deg")
    return _save(
        fig,
        output_dir,
        "22_masked_phase.png",
        figure_id="F22",
        function="plot_masked_phase_maps",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Masked phase maps. Cells below 0.05 Jy or poorly occupied are grey. "
            "Phase is exploratory and is not a C03 metric."
        ),
    )


def plot_radial_coherence(
    coherence: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    table = {
        "phase_status": coherence.get("phase_status"),
        "amp_floor_jy": coherence.get("amp_floor_jy"),
        "extent": coherence.get("raster_extent_arcmin"),
    }
    for hand, color in (("rr", "C0"), ("ll", "C1")):
        rows = list(coherence.get(hand) or ())
        mid = [float(row["r_mid_arcmin"]) for row in rows]
        axes[0].plot(
            mid,
            [float(row["correlation_abs"]) for row in rows],
            "o-",
            color=color,
            label=hand.upper(),
        )
        axes[1].plot(
            mid,
            [float(np.degrees(np.arctan2(row["slope_imag"], row["slope_real"]))) for row in rows],
            "o-",
            color=color,
            label=hand.upper(),
        )
        axes[2].errorbar(
            mid,
            [float(row["circular_phase_deg"]) for row in rows],
            yerr=[float(row.get("circular_phase_std_deg") or 0.0) for row in rows],
            fmt="o-",
            color=color,
            label=hand.upper(),
        )
        table[hand] = rows
    axes[0].set_ylabel("complex correlation")
    axes[1].set_ylabel("arg(slope) (deg)")
    axes[2].set_ylabel("circular phase residual (deg)")
    for axis in axes:
        axis.set_xlabel("radius (arcmin)")
        axis.legend(fontsize=7)
        axis.axvline(40.0, color="k", ls=":", lw=0.6)
    return _save(
        fig,
        output_dir,
        "23_radial_coherence.png",
        figure_id="F23",
        function="plot_radial_coherence",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Visibility-domain coherence versus radius. Phase uses the 0.05 Jy "
            "amplitude floor. Exploratory. Supports C03."
        ),
    )


def plot_antenna_coherence(
    antenna: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    movers = list(antenna.get("movers") or ())
    table = {"n_movers": len(movers), "phase_status": antenna.get("phase_status")}
    for mover in movers:
        rr = list(mover.get("rr") or ())
        mid = [float(row["r_mid_arcmin"]) for row in rr]
        axes[0].plot(mid, [float(row["correlation_abs"]) for row in rr], color="0.7", lw=0.7)
        axes[1].plot(mid, [float(row["circular_phase_deg"]) for row in rr], color="0.7", lw=0.7)
    if movers:
        stack_c = np.array(
            [[float(row["correlation_abs"]) for row in mover.get("rr") or ()] for mover in movers],
            dtype=np.float64,
        )
        stack_p = np.array(
            [
                [float(row["circular_phase_deg"]) for row in mover.get("rr") or ()]
                for mover in movers
            ],
            dtype=np.float64,
        )
        mid = [float(row["r_mid_arcmin"]) for row in movers[0].get("rr") or ()]
        axes[0].plot(mid, np.nanmedian(stack_c, axis=0), "k-", lw=1.6, label="RR median")
        axes[1].plot(mid, np.nanmedian(stack_p, axis=0), "k-", lw=1.6, label="RR median")
    axes[0].set_ylabel("RR complex correlation")
    axes[1].set_ylabel("RR circular phase residual (deg)")
    for axis in axes:
        axis.set_xlabel("radius (arcmin)")
        axis.legend(fontsize=7)
    return _save(
        fig,
        output_dir,
        "24_antenna_coherence.png",
        figure_id="F24",
        function="plot_antenna_coherence",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Per-moving-antenna RR radial coherence. Grey lines are dishes; "
            "black is the median. Exploratory. Supports C03."
        ),
    )


def plot_bright_source_examples(
    examples: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    plt = _pyplot()
    rows = list(examples.get("examples") or ())
    fig, axes = plt.subplots(1, max(len(rows), 1), figsize=(2.4 * max(len(rows), 1) + 1.2, 3.6))
    if len(rows) <= 1:
        axes = [axes]
    table = {"note": examples.get("note"), "examples": rows}
    for axis, row in zip(axes, rows, strict=False):
        meas = complex(float(row["rr_meas_re"]), float(row["rr_meas_im"]))
        pred = complex(float(row["rr_pred_re"]), float(row["rr_pred_im"]))
        axis.plot([0.0, pred.real], [0.0, pred.imag], "C1-o", label="CASSBEAM")
        axis.plot([0.0, meas.real], [0.0, meas.imag], "C0-o", label="measured")
        axis.set_aspect("equal", adjustable="datalim")
        axis.axhline(0.0, color="k", lw=0.4)
        axis.axvline(0.0, color="k", lw=0.4)
        radius = float(row["radius_arcmin"])
        ell = float(row["l_arcmin"])
        emm = float(row["m_arcmin"])
        axis.set_title(f"{radius:.2f}′  (l,m)=({ell:.2f},{emm:.2f})")
        axis.set_xlabel("Re V (Jy)")
        axis.set_ylabel("Im V (Jy)")
    if rows:
        axes[0].legend(fontsize=7)
    fig.suptitle("Array-average RR voltage at example offsets", fontsize=10)
    return _save(
        fig,
        output_dir,
        "25_bright_source_examples.png",
        figure_id="F25",
        function="plot_bright_source_examples",
        claims=("C03",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Array-average RR voltage at the nearest occupied cell to each "
            "nominal radius. Titles show the exact radius and (l, m). A common "
            "scalar beam cancels voltage phase in Stokes I. Supports C03."
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
            "Publication 20%-of-peak all-data squint. "
            "Independent-mask versus common-mask sensitivity is F29. "
            "Full-raster centroids are not the publication estimator. Supports C07."
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
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    freq = [row["frequency_hz"] / 1e9 for row in rows]
    axes[0].plot(freq, [row["rr_correlation"] for row in rows], "o-", label="RR")
    axes[0].plot(freq, [row["ll_correlation"] for row in rows], "s-", label="LL")
    axes[0].set_ylabel("|correlation|")
    axes[0].set_title("Copolar correlation")
    axes[1].plot(freq, [row["rr_residual_power"] for row in rows], "o-", label="RR")
    axes[1].plot(freq, [row["ll_residual_power"] for row in rows], "s-", label="LL")
    axes[1].axhline(0.01, color="0.4", ls=":", lw=0.8, label="1% main-lobe cut")
    axes[1].set_ylabel("residual power")
    axes[1].set_title("Main-lobe copolar residual power")
    axes[2].plot(freq, [row["rl_residual_power"] for row in rows], "o-", label="RL")
    axes[2].plot(freq, [row["lr_residual_power"] for row in rows], "s-", label="LR")
    axes[2].set_ylabel("residual power")
    axes[2].set_title("Main-lobe full-Jones RL/LR residual")
    for axis in axes:
        axis.set_xlabel("frequency (GHz)")
        axis.legend(fontsize=7)
        axis.axvline(4.564, color="k", ls="--", lw=0.7)
    table = {
        "channels": rows,
        "region": "main_lobe",
        "estimator": "residual_power",
        "n_main_lobe": [int(row["n_main_lobe"]) for row in rows],
    }
    return _save(
        fig,
        output_dir,
        "16_spw4_frequency.png",
        figure_id="F16",
        function="plot_frequency_series",
        claims=("C01", "C02", "C05"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "SPW-4 main-lobe channel dependence. Left/centre: unit EVLA-C "
            "diagonal; the dotted line is the 1% accepted cut. Right: "
            "experimental full-Jones RL/LR residual power on the same "
            "support. Frequency squint is omitted. Supports C01, C02, C05."
        ),
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
        name: {int(item) for item in ids}
        for name, ids in (fields.get("partition") or {}).items()
        if name in {"training", "inner_holdout", "sealed_holdout"}
        and isinstance(ids, (list, tuple, set))
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
        if "l_arcmin" in record:
            l_arcmin = float(record["l_arcmin"])
            m_arcmin = float(record["m_arcmin"])
        else:
            l_arcmin = float(np.rad2deg(float(record["l_rad"])) * 60.0)
            m_arcmin = float(np.rad2deg(float(record["m_rad"])) * 60.0)
        axes[0].scatter(
            [l_arcmin],
            [m_arcmin],
            color=colors.get(label, "k"),
            label=label if label not in axes[0].get_legend_handles_labels()[1] else None,
        )
        axes[0].annotate(
            str(record["name"]),
            (l_arcmin, m_arcmin),
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
    classification: Mapping[str, Any] | None = None,
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
    outcome = str((classification or {}).get("outcome") or "not_classified")
    table = {
        "quadrant_rl_correlation": dict(zip(labels, corrs, strict=True)),
        "outcome": outcome,
        "classification": dict(classification or {}),
        "estimator": "paired residual_power full_minus_diagonal",
    }
    return _save(
        fig,
        output_dir,
        "18_experimental_crosshand.png",
        figure_id="F18",
        function="plot_crosshand_null",
        claims=("C05",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            f"EVLA-C full Jones versus diagonal on RL. Outcome: {outcome}. "
            "Supports C05."
        ),
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


def plot_coordinate_feed_impact(
    comparison: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    """Show absolute losses and paired model deltas on the frozen development split."""

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    metrics = comparison.get("metrics") or {}
    regions = ("main_lobe", "mid", "outer_diagnostic", "all")
    x = np.arange(len(regions), dtype=np.float64)
    width = 0.24
    absolute: dict[str, list[float]] = {}
    for model_index, (model, label, color) in enumerate(COORDINATE_FEED_MODELS):
        values = []
        for region in regions:
            hands = (((metrics.get(model) or {}).get("development") or {}).get(region) or {})
            values.append(
                0.5
                * (
                    float((hands.get("rr") or {}).get("residual_power", np.nan))
                    + float((hands.get("ll") or {}).get("residual_power", np.nan))
                )
            )
        absolute[model] = values
        axes[0].bar(
            x + (model_index - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, [name.replace("_", "\n") for name in regions])
    axes[0].set_ylabel("mean RR/LL residual power")
    axes[0].set_title("Absolute fit on development rows")
    axes[0].legend(fontsize=7)

    paired = comparison.get("paired_scores") or {}
    comparisons = (
        ("generic_source_lm_vs_generic_commanded", "coordinate only"),
        ("evla_c_source_lm_vs_generic_commanded", "combined change"),
        ("evla_c_source_lm_vs_generic_source_lm", "EVLA-C feed"),
    )
    labels = []
    table_pairs: dict[str, object] = {}
    for index, (key, label) in enumerate(comparisons):
        record = paired.get(key) or {}
        labels.append(label)
        table_pairs[key] = {}
        for offset, (axis_name, marker, color) in zip(
            (-0.08, 0.08),
            (("spatial", "o", "C0"), ("moving", "s", "C3")),
            strict=True,
        ):
            score = record.get(axis_name) or {}
            delta = float(score.get("delta", np.nan))
            lo = float(score.get("delta_lo", np.nan))
            hi = float(score.get("delta_hi", np.nan))
            axes[1].errorbar(
                index + offset,
                delta,
                yerr=np.array([[delta - lo], [hi - delta]]),
                fmt=marker,
                color=color,
                capsize=3,
                label=axis_name if index == 0 else None,
            )
            table_pairs[key][axis_name] = dict(score)
    axes[1].axhline(0.0, color="k", lw=0.7)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right")
    axes[1].set_ylabel("paired ΔL (candidate − baseline)")
    axes[1].set_title("Predeclared holdout effects (95% CI)")
    axes[1].legend(fontsize=8)
    table = {"absolute": absolute, "paired": table_pairs}
    return _save(
        fig,
        output_dir,
        "26_coordinate_feed_impact.png",
        figure_id="F26",
        function="plot_coordinate_feed_impact",
        claims=("C01", "C07"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Impact of correcting the query coordinate and replacing the generic "
            "VLA feed with CASA's EVLA-C parameters. Negative paired ΔL is better. "
            "The EVLA-C feed transfers across spatial and mover holdouts when both "
            "models use source-in-beam coordinates."
        ),
    )


def plot_coordinate_feed_scatter(
    scatter: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    """Compare measured and predicted copolar real parts for all three models."""

    plt = _pyplot()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    intensity = float(np.asarray(scatter["source_i_jy"]).reshape(-1)[0])
    main = np.asarray(scatter["main_lobe"], dtype=bool)
    table: dict[str, object] = {"source_i_jy": intensity, "region": "main_lobe"}
    for column, (model, label, _color) in enumerate(COORDINATE_FEED_MODELS):
        for row, hand in enumerate(("rr", "ll")):
            observed = np.asarray(scatter[f"measured_{hand}"])[main].real / intensity
            predicted = np.asarray(scatter[f"{model}_{hand}"])[main].real / intensity
            axis = axes[row, column]
            axis.scatter(predicted, observed, s=3, alpha=0.15, linewidths=0)
            finite = np.isfinite(observed) & np.isfinite(predicted)
            if bool(np.any(finite)):
                lo = float(min(np.min(observed[finite]), np.min(predicted[finite])))
                hi = float(max(np.max(observed[finite]), np.max(predicted[finite])))
                axis.plot([lo, hi], [lo, hi], "k-", lw=0.7)
            axis.set_title(f"{hand.upper()}: {label}")
            axis.set_xlabel("predicted Re(V)/I")
            axis.set_ylabel("measured Re(V)/I")
            table[f"{model}_{hand}_n"] = int(np.sum(finite))
    return _save(
        fig,
        output_dir,
        "27_coordinate_feed_scatter.png",
        figure_id="F27",
        function="plot_coordinate_feed_scatter",
        claims=("C01",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Main-lobe measured versus predicted copolar real visibility for the "
            "historical model, the coordinate correction alone, and EVLA-C at the "
            "correct source-in-beam coordinate."
        ),
    )


def plot_coordinate_feed_residual_maps(
    maps: Mapping[str, NDArray],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    """Compare spatial residual magnitude for the same three model predictions."""

    plt = _pyplot()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    l_axis = np.asarray(maps["l_arcmin"], dtype=np.float64)
    m_axis = np.asarray(maps["m_arcmin"], dtype=np.float64)
    extent = (float(l_axis[0]), float(l_axis[-1]), float(m_axis[0]), float(m_axis[-1]))
    residuals = []
    for hand in ("rr", "ll"):
        measured = np.asarray(maps[f"{hand}_measured"])
        support = np.asarray(maps[f"{hand}_weight"], dtype=np.float64) > 0.0
        for model, _label, _color in COORDINATE_FEED_MODELS:
            residual = np.abs(measured - np.asarray(maps[f"{model}_{hand}"]))
            residuals.append(residual[support])
    finite = np.concatenate([item[np.isfinite(item)] for item in residuals if item.size])
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    table: dict[str, object] = {"scale_max_jy": vmax, "coordinate": "source_lm_feed"}
    for row, hand in enumerate(("rr", "ll")):
        measured = np.asarray(maps[f"{hand}_measured"])
        support = np.asarray(maps[f"{hand}_weight"], dtype=np.float64) > 0.0
        for column, (model, label, _color) in enumerate(COORDINATE_FEED_MODELS):
            residual = np.abs(measured - np.asarray(maps[f"{model}_{hand}"]))
            image = axes[row, column].imshow(
                np.ma.masked_where(~support, residual),
                origin="lower",
                extent=extent,
                cmap="magma",
                aspect="equal",
                vmin=0.0,
                vmax=vmax,
            )
            axes[row, column].set_title(f"{hand.upper()}: {label}")
            axes[row, column].set_xlabel("source l in feed frame (arcmin)")
            axes[row, column].set_ylabel("source m in feed frame (arcmin)")
            table[f"{model}_{hand}_median_jy"] = float(np.nanmedian(residual[support]))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02, label="|ΔV| (Jy)")
    return _save(
        fig,
        output_dir,
        "28_coordinate_feed_residual_maps.png",
        figure_id="F28",
        function="plot_coordinate_feed_residual_maps",
        claims=("C01", "C02", "C03"),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "Copolar residual maps on the corrected source-in-beam axes. All panels "
            "use one colour scale, so the EVLA-C feed effect can be compared directly."
        ),
    )


def plot_coordinate_feed_squint(
    comparison: Mapping[str, Any],
    output_dir: Path,
    *,
    bundle_sha256: str | None = None,
) -> list[Path]:
    """Plot explicitly named R-minus-L squint vectors in one source/feed frame."""

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    measured = (
        ((comparison.get("map_squint") or {}).get("source_lm_labels") or {}).get(
            "independent_masks"
        )
        or {}
    )
    generic = comparison.get("generic_plane_centroids") or {}
    evla = comparison.get("evla_plane_centroids") or {}

    def vector(record: Mapping[str, Any]) -> tuple[float, float]:
        explicit = record.get("r_minus_l")
        if isinstance(explicit, Mapping) and explicit.get("r_minus_l") is not None:
            values = explicit["r_minus_l"]
            return float(values[0]), float(values[1])
        return (
            float(record.get("rr_l_arcmin", np.nan))
            - float(record.get("ll_l_arcmin", np.nan)),
            float(record.get("rr_m_arcmin", np.nan))
            - float(record.get("ll_m_arcmin", np.nan)),
        )

    records = (
        ("measured development map", measured, "C0"),
        ("generic VLA", generic, "0.55"),
        ("EVLA-C", evla, "C2"),
    )
    table: dict[str, object] = {"vector_kind": "r_minus_l", "frame": "source_lm_feed"}
    for label, record, color in records:
        dl, dm = vector(record)
        axis.arrow(0.0, 0.0, dl, dm, width=0.008, length_includes_head=True, color=color)
        axis.text(dl, dm, f" {label}", color=color, fontsize=9)
        table[label] = {
            "dl_arcmin": dl,
            "dm_arcmin": dm,
            "separation_arcmin": float(np.hypot(dl, dm)),
        }
    memo = float(measured.get("memo195_separation_arcmin", np.nan))
    axis.add_patch(plt.Circle((0.0, 0.0), memo, fill=False, ls=":", color="k"))
    axis.axhline(0.0, color="k", lw=0.4)
    axis.axvline(0.0, color="k", lw=0.4)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.85, 0.85)
    axis.set_ylim(-0.85, 0.85)
    axis.set_xlabel("R−L Δl (arcmin)")
    axis.set_ylabel("R−L Δm (arcmin)")
    axis.set_title("Squint vectors after coordinate/feed audit")
    table["memo195_separation_arcmin"] = memo
    return _save(
        fig,
        output_dir,
        "29_coordinate_feed_squint.png",
        figure_id="F29",
        function="plot_coordinate_feed_squint",
        claims=("C07",),
        table=table,
        bundle_sha256=bundle_sha256,
        caption=(
            "R-minus-L squint vectors in source-in-beam/feed-frame labels. The measured "
            "vector is the training-only independent-mask development estimate, not the "
            "all-data 0.515 arcmin publication magnitude. The EVLA-C feed recovers the "
            "expected separation and substantially reduces the direction discrepancy."
        ),
    )


def write_all_figures(bundle: ValidationBundle, output_dir: Path) -> list[Path]:
    """Write every publication figure and sidecar from the bundle."""

    token = _SIDECAR_PROVENANCE.set(sidecar_provenance(bundle))
    try:
        return _write_all_figures_body(bundle, output_dir)
    finally:
        _SIDECAR_PROVENANCE.reset(token)


def _write_all_figures_body(bundle: ValidationBundle, output_dir: Path) -> list[Path]:
    from sl1mjax.beam_validation_statistics import offset_ring_diagonal_closure

    bundle_sha = str(bundle.manifest.get("bundle_sha256") or "")
    scatter = bundle.plot_table("holoraster_scatter.npz")
    maps = bundle.plot_table("holoraster_maps.npz")
    cells = bundle.plot_table("holoraster_cells.npz")
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
        plot_cassbeam_scatter(
            scatter,
            output_dir,
            cells=cells,
            bundle_sha256=bundle_sha,
            region="main_lobe",
        )
    )
    written.extend(plot_spatial_residual_maps(maps, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_residual_radius(bundle.residual_geometry, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(
        plot_residual_strata(bundle.residual_strata, output_dir, bundle_sha256=bundle_sha)
    )
    source_i = float(
        bundle.radial_coherence.get("source_i_jy")
        or bundle.residual_strata.get("source_i_jy")
        or channel32_source_i_jy()
    )
    written.extend(
        plot_amplitude_db_maps(maps, output_dir, bundle_sha256=bundle_sha, source_i_jy=source_i)
    )
    written.extend(
        plot_signed_complex_cuts(maps, output_dir, bundle_sha256=bundle_sha, source_i_jy=source_i)
    )
    written.extend(plot_masked_phase_maps(maps, output_dir, bundle_sha256=bundle_sha))
    written.extend(
        plot_radial_coherence(bundle.radial_coherence, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(
        plot_antenna_coherence(bundle.antenna_coherence, output_dir, bundle_sha256=bundle_sha)
    )
    written.extend(
        plot_bright_source_examples(
            bundle.bright_source_examples, output_dir, bundle_sha256=bundle_sha
        )
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
            classification=bundle.holoraster_channel32.get("classification") or {},
        )
    )
    written.extend(plot_spw5_sealed(output_dir, bundle_sha256=bundle_sha))
    coordinate_scatter = bundle.plot_table("coordinate_feed_scatter.npz")
    coordinate_maps = bundle.plot_table("coordinate_feed_maps.npz")
    written.extend(
        plot_coordinate_feed_impact(
            bundle.coordinate_feed_comparison, output_dir, bundle_sha256=bundle_sha
        )
    )
    written.extend(
        plot_coordinate_feed_scatter(
            coordinate_scatter, output_dir, bundle_sha256=bundle_sha
        )
    )
    written.extend(
        plot_coordinate_feed_residual_maps(
            coordinate_maps, output_dir, bundle_sha256=bundle_sha
        )
    )
    written.extend(
        plot_coordinate_feed_squint(
            bundle.coordinate_feed_comparison, output_dir, bundle_sha256=bundle_sha
        )
    )
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
