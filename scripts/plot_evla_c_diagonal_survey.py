"""Frequency and frequency-by-radius suitability plots from survey reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(table: Path) -> list[dict[str, object]]:
    payload = json.loads(table.read_text(encoding="utf-8"))
    return list(payload["slots"])


def plot_mainlobe(rows: list[dict[str, object]], output: Path) -> Path:
    freq = np.asarray([row["frequency_hz"] for row in rows], dtype=np.float64) / 1.0e6
    rr = np.asarray([row["main_lobe_rr"] for row in rows], dtype=np.float64)
    ll = np.asarray([row["main_lobe_ll"] for row in rows], dtype=np.float64)
    spw = np.asarray([row["spectral_window_id"] for row in rows], dtype=np.int32)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for index, window in enumerate(sorted(set(int(item) for item in spw))):
        keep = spw == window
        marker = markers[index % len(markers)]
        axis.plot(
            freq[keep],
            rr[keep],
            marker=marker,
            linestyle="None" if int(keep.sum()) == 1 else "-",
            color="C0",
            label=f"SPW {window} RR",
        )
        axis.plot(
            freq[keep],
            ll[keep],
            marker=marker,
            linestyle="None" if int(keep.sum()) == 1 else "-",
            color="C1",
            label=f"SPW {window} LL",
        )
    axis.axhline(0.01, color="0.3", linestyle="--", linewidth=1.0, label="1% main-lobe cut")
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("Main-lobe residual power")
    axis.set_title("EVLA-C diagonal survey — identity residual Jones")
    axis.legend(fontsize=7, ncol=3, loc="upper right")
    finite = np.concatenate([rr, ll])
    finite = finite[np.isfinite(finite)]
    axis.set_ylim(0.0, max(0.02, float(np.max(finite)) * 1.15) if finite.size else 0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_frequency_radius(rows: list[dict[str, object]], output: Path) -> Path:
    """Frequency × radius-stratum residual-power map from scored reports."""

    regions = (
        ("main_lobe", "main_lobe_rr", "main_lobe_ll"),
        ("mid", "mid_rr", "mid_ll"),
        ("outer", "outer_rr", "outer_ll"),
    )
    freq = np.asarray([row["frequency_hz"] for row in rows], dtype=np.float64) / 1.0e6
    order = np.argsort(freq)
    rows = [rows[index] for index in order]
    freq = freq[order]
    grid = np.full((len(regions), len(rows)), np.nan, dtype=np.float64)
    for i_row, row in enumerate(rows):
        for i_region, (_name, rr_key, ll_key) in enumerate(regions):
            rr = row.get(rr_key)
            ll = row.get(ll_key)
            values = [float(item) for item in (rr, ll) if item is not None]
            if values:
                grid[i_region, i_row] = max(values)
    fig, axis = plt.subplots(figsize=(8.4, 3.4))
    image = axis.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="magma_r",
        vmin=0.0,
        vmax=max(0.05, float(np.nanmax(grid)) if np.any(np.isfinite(grid)) else 0.05),
    )
    axis.set_yticks(range(len(regions)))
    axis.set_yticklabels([name for name, _rr, _ll in regions])
    tick = np.linspace(0, max(len(rows) - 1, 0), num=min(len(rows), 8), dtype=int)
    axis.set_xticks(tick)
    axis.set_xticklabels([f"{freq[index]:.0f}" for index in tick], rotation=30, ha="right")
    axis.set_xlabel("Frequency (MHz)")
    axis.set_title("Worse-hand residual power by frequency and radius stratum")
    fig.colorbar(image, ax=axis, label="residual power")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frequency-radius-output",
        type=Path,
        default=None,
        help="Optional frequency × radius suitability map.",
    )
    arguments = parser.parse_args()
    rows = _load_rows(arguments.table)
    print(plot_mainlobe(rows, arguments.output))
    radius_output = arguments.frequency_radius_output
    if radius_output is None:
        radius_output = arguments.output.with_name("frequency_radius_suitability.png")
    print(plot_frequency_radius(rows, radius_output))


if __name__ == "__main__":
    main()
