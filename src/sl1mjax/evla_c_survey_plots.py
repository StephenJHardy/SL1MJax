"""Phase 5 maps from diagonal-survey channel exports.

Uses the scored visibility samples. Does not reopen the Measurement Set
and does not apply residual Jones.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_validation_statistics import (
    RADIAL_EDGES_ARCMIN,
    binned_complex_map,
    bright_source_examples,
    map_axis_cut,
    radial_coherence,
)
from sl1mjax.evla_c_diagonal_survey import MAIN_LOBE_ACCEPTED_MAX, refuse_frozen_write, write_json_atomic
from sl1mjax.holography_cassbeam_holoraster_report import squint_from_voltage_maps

HANDS = (("RR", 0, 0), ("LL", 1, 1))
CUT_KINDS = ("m0", "l0", "diag", "antidiag")
CUT_LABELS = {
    "m0": "m = 0",
    "l0": "l = 0",
    "diag": "l = m",
    "antidiag": "l = −m",
}


def load_survey_export(path: Path) -> dict[str, NDArray]:
    with np.load(path, allow_pickle=False) as handle:
        return {str(key): np.asarray(handle[key]) for key in handle.files}


def _hand_weight(weight: NDArray, row: int, col: int) -> NDArray[np.float64]:
    plane = np.asarray(weight)
    if plane.ndim == 3:
        return np.asarray(plane[:, row, col], dtype=np.float64)
    if plane.ndim == 2:
        return np.asarray(plane[:, row], dtype=np.float64)
    return np.asarray(plane, dtype=np.float64)


def maps_from_export(export: Mapping[str, NDArray]) -> dict[str, NDArray]:
    offset = np.asarray(export["source_lm_feed"], dtype=np.float64)
    measured = np.asarray(export["measured"])
    predicted = np.asarray(export["predicted"])
    weight = np.asarray(export["weight"])
    maps: dict[str, NDArray] = {}
    template = None
    for name, row, col in HANDS:
        meas = binned_complex_map(
            offset, measured[:, row, col], _hand_weight(weight, row, col)
        )
        pred = binned_complex_map(
            offset, predicted[:, row, col], _hand_weight(weight, row, col)
        )
        residual = binned_complex_map(
            offset,
            measured[:, row, col] - predicted[:, row, col],
            _hand_weight(weight, row, col),
        )
        key = name.lower()
        maps[f"{key}_measured"] = meas["mean"]
        maps[f"{key}_cassbeam"] = pred["mean"]
        maps[f"{key}_residual"] = residual["mean"]
        if template is None:
            maps["l_arcmin"] = meas["l_arcmin"]
            maps["m_arcmin"] = meas["m_arcmin"]
            maps["weight"] = meas["weight"]
            template = meas
    return maps


def radial_from_export(export: Mapping[str, NDArray]) -> dict[str, list[dict[str, float]]]:
    offset = np.asarray(export["source_lm_feed"], dtype=np.float64)
    measured = np.asarray(export["measured"])
    predicted = np.asarray(export["predicted"])
    weight = np.asarray(export["weight"])
    choose = np.ones(offset.shape[0], dtype=bool)
    return {
        name: radial_coherence(
            offset,
            measured[:, row, col],
            predicted[:, row, col],
            _hand_weight(weight, row, col),
            choose,
        )
        for name, row, col in HANDS
    }


def squint_from_export(export: Mapping[str, NDArray]) -> dict[str, object]:
    offset = np.asarray(export["source_lm_feed"], dtype=np.float64)
    measured = np.asarray(export["measured"])
    predicted = np.asarray(export["predicted"])
    weight = np.asarray(export["weight"])
    frequency = float(np.asarray(export["frequency_hz"]).reshape(-1)[0])
    return {
        "frequency_hz": frequency,
        "coordinate_query": "source_lm_feed",
        "measured": squint_from_voltage_maps(
            offset,
            np.abs(measured[:, 0, 0]) ** 2,
            np.abs(measured[:, 1, 1]) ** 2,
            _hand_weight(weight, 0, 0),
            _hand_weight(weight, 1, 1),
            frequency_hz=frequency,
            series="measured_source_lm",
        ),
        "cassbeam": squint_from_voltage_maps(
            offset,
            np.abs(predicted[:, 0, 0]) ** 2,
            np.abs(predicted[:, 1, 1]) ** 2,
            _hand_weight(weight, 0, 0),
            _hand_weight(weight, 1, 1),
            frequency_hz=frequency,
            series="evla_c_source_lm",
        ),
    }


def coarse_radial_minimum(
    rows: Sequence[Mapping[str, float]],
    *,
    amplitude_key: str = "median_abs_obs",
) -> dict[str, object]:
    """Minimum of a 10-arcmin radial median. Not a first-null location."""

    mids = np.asarray([row["r_mid_arcmin"] for row in rows], dtype=np.float64)
    amp = np.asarray([row.get(amplitude_key, float("nan")) for row in rows], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(mids) & np.isfinite(amp))
    if finite.size < 3:
        return {
            "quantity": "coarse_radial_minimum",
            "status": "unresolved",
            "reason": "too_few_radial_bins",
        }
    mids = mids[finite]
    amp = amp[finite]
    kept = [rows[int(index)] for index in finite]
    peak = int(np.argmax(amp[: min(3, amp.size)]))
    idx = None
    for i in range(peak + 1, amp.size - 1):
        if amp[i] <= amp[i - 1] and amp[i] <= amp[i + 1]:
            idx = i
            break
    if idx is None:
        rise = np.flatnonzero(np.diff(amp[peak:]) > 0.0)
        if rise.size:
            idx = peak + int(rise[0])
    if idx is None:
        return {
            "quantity": "coarse_radial_minimum",
            "status": "unresolved",
            "reason": "no_sidelobe_rise_in_radial_bins",
            "note": "10′ bins can skip the first minimum and land on a later one",
        }
    chosen = kept[idx]
    return {
        "quantity": "coarse_radial_minimum",
        "status": "coarse_minimum",
        "r_mid_arcmin": float(mids[idx]),
        "r_lo_arcmin": float(chosen.get("r_lo_arcmin", mids[idx])),
        "r_hi_arcmin": float(chosen.get("r_hi_arcmin", mids[idx])),
        "note": "Coarse radial statistic; not a first-null estimate",
    }


def first_null_from_radial(
    rows: Sequence[Mapping[str, float]],
    *,
    amplitude_key: str = "median_abs_obs",
) -> dict[str, object]:
    """Backward-compatible alias. The quantity is a coarse radial minimum."""

    return coarse_radial_minimum(rows, amplitude_key=amplitude_key)


def first_magnitude_minimum(
    offset_arcmin: ArrayLike,
    amplitude: ArrayLike,
) -> dict[str, object]:
    """First local magnitude minimum on the positive axis of a 1-D cut."""

    x = np.asarray(offset_arcmin, dtype=np.float64).reshape(-1)
    amp = np.abs(np.asarray(amplitude)).reshape(-1)
    if x.size != amp.size:
        raise ValueError("offset and amplitude length must match")
    keep = np.isfinite(x) & np.isfinite(amp) & (x >= -1.0e-9)
    x = x[keep]
    amp = amp[keep]
    if x.size < 3:
        return {"status": "unresolved", "reason": "too_few_samples"}
    order = np.argsort(x)
    x = x[order]
    amp = amp[order]
    peak = int(np.argmax(amp[: min(3, amp.size)]))
    for i in range(peak + 1, amp.size - 1):
        if amp[i] <= amp[i - 1] and amp[i] <= amp[i + 1] and amp[i] < amp[peak]:
            return {
                "status": "resolved",
                "r_arcmin": float(x[i]),
                "amplitude": float(amp[i]),
            }
    return {"status": "unresolved", "reason": "no_local_minimum"}


def model_first_minima_from_cuts(
    cuts: Mapping[str, Mapping[str, NDArray]],
) -> dict[str, dict[str, object]]:
    """Model-resolved first-minimum locations from predicted axis cuts."""

    out: dict[str, dict[str, object]] = {}
    for hand in ("rr", "ll"):
        per_cut: dict[str, object] = {}
        radii: list[float] = []
        for kind in CUT_KINDS:
            cut = cuts[f"{hand}_{kind}"]
            found = first_magnitude_minimum(cut["x_arcmin"], cut["predicted"])
            per_cut[kind] = found
            if found.get("status") == "resolved":
                radii.append(float(found["r_arcmin"]))
        out[hand.upper()] = {
            "cuts": per_cut,
            "median_r_arcmin": float(np.median(radii)) if radii else float("nan"),
            "n_resolved_cuts": len(radii),
        }
    return out


def measured_first_null_near_model(
    offset_lm: ArrayLike,
    measured: ArrayLike,
    weight: ArrayLike,
    *,
    model_r_arcmin: float,
    half_width_arcmin: float = 3.0,
) -> dict[str, object]:
    """Ask whether dense samples bracket a model first-minimum radius."""

    if not np.isfinite(model_r_arcmin):
        return {
            "status": "unresolved",
            "reason": "model_first_minimum_unavailable",
            "model_r_arcmin": float("nan"),
        }
    offset = np.asarray(offset_lm, dtype=np.float64)
    radius = np.hypot(offset[:, 0], offset[:, 1]) * (180.0 / np.pi) * 60.0
    amp = np.abs(np.asarray(measured)).reshape(-1)
    wt = np.asarray(weight, dtype=np.float64).reshape(-1)
    keep = np.isfinite(radius) & np.isfinite(amp) & (wt > 0.0)
    inner = keep & (radius < float(model_r_arcmin))
    outer = keep & (radius > float(model_r_arcmin))
    near = keep & (np.abs(radius - float(model_r_arcmin)) <= float(half_width_arcmin))
    if int(inner.sum()) < 3 or int(outer.sum()) < 3 or int(near.sum()) < 5:
        return {
            "status": "unresolved",
            "reason": "samples_cannot_bracket_model_minimum",
            "model_r_arcmin": float(model_r_arcmin),
            "n_inner": int(inner.sum()),
            "n_outer": int(outer.sum()),
            "n_near": int(near.sum()),
        }
    return {
        "status": "bracketed_unresolved_zero",
        "reason": "samples_exist_around_model_minimum_but_do_not_locate_a_zero",
        "model_r_arcmin": float(model_r_arcmin),
        "n_inner": int(inner.sum()),
        "n_outer": int(outer.sum()),
        "n_near": int(near.sum()),
        "near_median_abs": float(np.median(amp[near])),
    }


def rebin_radial_medians(
    radius_arcmin: ArrayLike,
    amplitude: ArrayLike,
    edges_arcmin: Sequence[float],
) -> list[dict[str, float]]:
    """Median amplitude in coarse radial bins. Used to test first-null relabelling."""

    radius = np.asarray(radius_arcmin, dtype=np.float64).reshape(-1)
    amp = np.abs(np.asarray(amplitude)).reshape(-1)
    edges = np.asarray(list(edges_arcmin), dtype=np.float64)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        keep = np.isfinite(radius) & np.isfinite(amp) & (radius >= lo) & (radius < hi)
        values = amp[keep]
        rows.append(
            {
                "r_lo_arcmin": float(lo),
                "r_hi_arcmin": float(hi),
                "r_mid_arcmin": float(0.5 * (lo + hi)),
                "median_abs_obs": float(np.median(values)) if values.size else float("nan"),
                "n": float(values.size),
            }
        )
    return rows


def bright_source_from_catalog(
    maps: Mapping[str, NDArray],
    *,
    catalog_path: Path,
    source_i_jy: float,
) -> list[dict[str, object]]:
    payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    atoms = list(payload.get("atoms") or ())
    radii = [0.0]
    radii.extend(float(atom.get("offset_arcmin") or 0.0) for atom in atoms)
    return bright_source_examples(maps, radii_arcmin=tuple(radii), source_i_jy=source_i_jy)


def signed_cuts_from_maps(maps: Mapping[str, NDArray]) -> dict[str, dict[str, NDArray]]:
    """Signed Re/Im axis and diagonal cuts of measured and EVLA-C maps."""

    cuts: dict[str, dict[str, NDArray]] = {}
    l_ax = maps["l_arcmin"]
    m_ax = maps["m_arcmin"]
    for hand in ("rr", "ll"):
        for kind in CUT_KINDS:
            measured = map_axis_cut(maps[f"{hand}_measured"], l_ax, m_ax, kind=kind)
            predicted = map_axis_cut(maps[f"{hand}_cassbeam"], l_ax, m_ax, kind=kind)
            cuts[f"{hand}_{kind}"] = {
                "x_arcmin": np.asarray(measured["x_arcmin"], dtype=np.float64),
                "measured": np.asarray(measured["value"]),
                "predicted": np.asarray(predicted["value"]),
            }
    return cuts


def jsonable_signed_cuts(cuts: Mapping[str, Mapping[str, NDArray]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, cut in cuts.items():
        measured = np.asarray(cut["measured"])
        predicted = np.asarray(cut["predicted"])
        payload[name] = {
            "x_arcmin": [float(value) for value in cut["x_arcmin"]],
            "measured_re": [float(value) for value in np.real(measured)],
            "measured_im": [float(value) for value in np.imag(measured)],
            "predicted_re": [float(value) for value in np.real(predicted)],
            "predicted_im": [float(value) for value in np.imag(predicted)],
        }
    return payload


def slot_diagnostics(
    export: Mapping[str, NDArray],
    *,
    report: Mapping[str, object] | None = None,
    catalog_path: Path | None = None,
) -> dict[str, object]:
    maps = maps_from_export(export)
    radial = radial_from_export(export)
    source_i = float((report or {}).get("source_i_jy") or 1.0)
    bright = []
    if catalog_path is not None and Path(catalog_path).is_file():
        bright = bright_source_from_catalog(
            maps, catalog_path=catalog_path, source_i_jy=source_i
        )
    cuts = signed_cuts_from_maps(maps)
    model_minima = model_first_minima_from_cuts(cuts)
    measured_nulls: dict[str, object] = {}
    for name, row, col in HANDS:
        measured_nulls[name] = measured_first_null_near_model(
            export["source_lm_feed"],
            np.asarray(export["measured"])[:, row, col],
            _hand_weight(np.asarray(export["weight"]), row, col),
            model_r_arcmin=float(model_minima[name]["median_r_arcmin"]),
        )
    return {
        "frequency_hz": float(np.asarray(export["frequency_hz"]).reshape(-1)[0]),
        "n": int(np.asarray(export["source_lm_feed"]).shape[0]),
        "maps": {key: np.asarray(value) for key, value in maps.items()},
        "radial": radial,
        "squint": squint_from_export(export),
        "coarse_radial_minima": {
            hand: coarse_radial_minimum(rows) for hand, rows in radial.items()
        },
        "nulls": {hand: coarse_radial_minimum(rows) for hand, rows in radial.items()},
        "model_first_minima": model_minima,
        "measured_first_null": measured_nulls,
        "cuts": cuts,
        "bright_sources": bright,
        "main_lobe_cut": MAIN_LOBE_ACCEPTED_MAX,
        "radial_edges_arcmin": list(RADIAL_EDGES_ARCMIN),
    }


def _imshow(axis, grid: NDArray, l_ax: NDArray, m_ax: NDArray, weight: NDArray, *, cmap: str, vmax=None):
    support = np.asarray(weight) > 0.0
    display = np.ma.masked_where(~support, np.asarray(grid))
    extent = (float(l_ax[0]), float(l_ax[-1]), float(m_ax[0]), float(m_ax[-1]))
    return axis.imshow(
        display,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
        vmax=vmax,
    )


def plot_slot_maps(diag: Mapping[str, object], output: Path, *, title: str) -> Path:
    import matplotlib.pyplot as plt

    maps = diag["maps"]
    l_ax = maps["l_arcmin"]
    m_ax = maps["m_arcmin"]
    weight = maps["weight"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    panels = (
        ("rr_measured", "RR measured", "viridis"),
        ("rr_cassbeam", "RR EVLA-C", "viridis"),
        ("rr_residual", "RR residual", "magma"),
        ("ll_measured", "LL measured", "viridis"),
        ("ll_cassbeam", "LL EVLA-C", "viridis"),
        ("ll_residual", "LL residual", "magma"),
    )
    for axis, (key, label, cmap) in zip(axes.ravel(), panels, strict=True):
        image = _imshow(axis, np.abs(maps[key]), l_ax, m_ax, weight, cmap=cmap)
        axis.set_title(label)
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def plot_slot_db(diag: Mapping[str, object], output: Path, *, title: str, source_i_jy: float) -> Path:
    import matplotlib.pyplot as plt

    maps = diag["maps"]
    l_ax = maps["l_arcmin"]
    m_ax = maps["m_arcmin"]
    weight = maps["weight"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.2))
    scale = max(float(source_i_jy), 1.0e-6)
    for axis, key, label in (
        (axes[0, 0], "rr_measured", "RR measured"),
        (axes[0, 1], "rr_cassbeam", "RR EVLA-C"),
        (axes[1, 0], "ll_measured", "LL measured"),
        (axes[1, 1], "ll_cassbeam", "LL EVLA-C"),
    ):
        db = 20.0 * np.log10(np.maximum(np.abs(maps[key]) / scale, 1.0e-6))
        image = _imshow(axis, db, l_ax, m_ax, weight, cmap="inferno", vmax=0.0)
        axis.set_title(f"{label} (dB / I)")
        axis.set_xlabel("l (arcmin)")
        axis.set_ylabel("m (arcmin)")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def plot_signed_cuts(diag: Mapping[str, object], output: Path, *, title: str) -> Path:
    import matplotlib.pyplot as plt

    cuts = diag["cuts"]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.8), sharex=True)
    for row, hand in enumerate(("rr", "ll")):
        for col, kind in enumerate(CUT_KINDS):
            axis = axes[row, col]
            cut = cuts[f"{hand}_{kind}"]
            x = np.asarray(cut["x_arcmin"], dtype=np.float64)
            measured = np.asarray(cut["measured"])
            predicted = np.asarray(cut["predicted"])
            axis.plot(x, np.real(measured), color="C0", lw=1.2, label="meas Re")
            axis.plot(x, np.imag(measured), color="C0", lw=1.0, ls=":", label="meas Im")
            axis.plot(x, np.real(predicted), color="C1", lw=1.2, label="EVLA-C Re")
            axis.plot(x, np.imag(predicted), color="C1", lw=1.0, ls=":", label="EVLA-C Im")
            axis.axhline(0.0, color="0.6", lw=0.6)
            axis.set_title(f"{hand.upper()} {CUT_LABELS[kind]}")
            if row == 1:
                axis.set_xlabel("offset (arcmin)")
            if col == 0:
                axis.set_ylabel("V / I_model")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def plot_nulls_vs_frequency(
    slots: Sequence[Mapping[str, object]], output: Path
) -> Path:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.6, 3.8))
    for hand, color in (("RR", "C0"), ("LL", "C3")):
        coarse_freq = []
        coarse_mid = []
        coarse_lo = []
        coarse_hi = []
        model_freq = []
        model_r = []
        for slot in slots:
            mhz = float(slot["frequency_hz"]) / 1.0e6
            coarse = (slot.get("coarse_radial_minima") or slot.get("nulls") or {}).get(hand) or {}
            if coarse.get("status") == "coarse_minimum":
                centre = float(coarse["r_mid_arcmin"])
                coarse_freq.append(mhz)
                coarse_mid.append(centre)
                coarse_lo.append(centre - float(coarse["r_lo_arcmin"]))
                coarse_hi.append(float(coarse["r_hi_arcmin"]) - centre)
            model = ((slot.get("model_first_minima") or {}).get(hand) or {})
            radius = model.get("median_r_arcmin")
            if radius is not None and np.isfinite(float(radius)):
                model_freq.append(mhz)
                model_r.append(float(radius))
        if coarse_freq:
            axis.errorbar(
                coarse_freq,
                coarse_mid,
                yerr=[coarse_lo, coarse_hi],
                fmt="o",
                color=color,
                alpha=0.55,
                label=f"{hand} coarse 10′ minimum",
                capsize=3,
            )
        if model_freq:
            axis.plot(
                model_freq,
                model_r,
                marker="x",
                linestyle="None",
                color=color,
                label=f"{hand} model first minimum",
            )
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("radius (arcmin)")
    axis.set_title("Model first minimum versus coarse 10′ radial minimum")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_frequency_radius_from_radial(
    slots: Sequence[Mapping[str, object]], output: Path
) -> Path:
    import matplotlib.pyplot as plt

    edges = list(RADIAL_EDGES_ARCMIN)
    freq = np.asarray([slot["frequency_hz"] for slot in slots], dtype=np.float64) / 1.0e6
    order = np.argsort(freq)
    slots = [slots[index] for index in order]
    freq = freq[order]
    grid = np.full((len(edges) - 1, len(slots)), np.nan)
    for i_slot, slot in enumerate(slots):
        rr = slot["radial"]["RR"]
        ll = slot["radial"]["LL"]
        for i_bin, (rr_row, ll_row) in enumerate(zip(rr, ll, strict=True)):
            values = [
                float(rr_row.get("residual_power", float("nan"))),
                float(ll_row.get("residual_power", float("nan"))),
            ]
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                grid[i_bin, i_slot] = max(finite)
    fig, axis = plt.subplots(figsize=(8.6, 3.8))
    image = axis.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="magma_r",
        vmin=0.0,
        vmax=max(0.05, float(np.nanmax(grid)) if np.any(np.isfinite(grid)) else 0.05),
    )
    axis.set_yticks(range(len(edges) - 1))
    axis.set_yticklabels([f"{lo:.0f}–{hi:.0f}′" for lo, hi in zip(edges[:-1], edges[1:])])
    ticks = np.linspace(0, max(len(slots) - 1, 0), num=min(len(slots), 8), dtype=int)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{freq[index]:.0f}" for index in ticks], rotation=30, ha="right")
    axis.set_xlabel("Frequency (MHz)")
    axis.set_title("Worse-hand residual power by frequency and radius")
    fig.colorbar(image, ax=axis, label="residual power")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_plot_products(
    channel_dir: Path,
    output_dir: Path,
    *,
    catalog_path: Path | None = None,
) -> dict[str, object]:
    refuse_frozen_write(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slots: list[dict[str, object]] = []
    written: list[str] = []
    for export_path in sorted(Path(channel_dir).glob("*channel*_export.npz")):
        report_path = export_path.with_name(export_path.name.replace("_export.npz", "_report.json"))
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        export = load_survey_export(export_path)
        diag = slot_diagnostics(export, report=report, catalog_path=catalog_path)
        stem = export_path.name.replace("_export.npz", "")
        freq_mhz = diag["frequency_hz"] / 1.0e6
        title = (
            f"{stem}  {freq_mhz:.0f} MHz  V/I_model  "
            f"identity residual Jones  n={diag['n']}"
        )
        written.append(
            str(
                plot_slot_maps(
                    diag, output_dir / f"{stem}_spatial.png", title=title
                )
            )
        )
        written.append(
            str(
                plot_slot_db(
                    diag,
                    output_dir / f"{stem}_amplitude_db.png",
                    title=title,
                    source_i_jy=float(report.get("source_i_jy") or 1.0),
                )
            )
        )
        written.append(
            str(
                plot_signed_cuts(
                    diag, output_dir / f"{stem}_signed_cuts.png", title=title
                )
            )
        )
        compact = {
            "file": export_path.name,
            "frequency_hz": diag["frequency_hz"],
            "n": diag["n"],
            "radial": diag["radial"],
            "squint": diag["squint"],
            "nulls": diag["nulls"],
            "coarse_radial_minima": diag["coarse_radial_minima"],
            "model_first_minima": diag["model_first_minima"],
            "measured_first_null": diag["measured_first_null"],
            "cuts": jsonable_signed_cuts(diag["cuts"]),
            "bright_sources": diag["bright_sources"],
            "status": report.get("status"),
            "spectral_window_id": report.get("spectral_window_id"),
            "channel": report.get("channel"),
        }
        slots.append(compact)
    radius_plot = output_dir / "frequency_radius_suitability.png"
    null_plot = output_dir / "nulls_vs_frequency.png"
    if slots:
        written.append(str(plot_frequency_radius_from_radial(slots, radius_plot)))
        written.append(str(plot_nulls_vs_frequency(slots, null_plot)))
    table = {
        "schema_version": 1,
        "n_slots": len(slots),
        "interpolation_validated": False,
        "residual_jones_policy": "not_applied",
        "slots": slots,
    }
    table_path = write_json_atomic(output_dir / "phase5_diagnostics.json", table)
    written.append(str(table_path))
    return {"n_slots": len(slots), "written": written}
