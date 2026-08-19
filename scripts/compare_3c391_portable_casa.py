"""Compare portable SL1MJax reconstructions with a CASA CLEAN reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _read(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdus:
        return (
            np.asarray(np.squeeze(hdus[0].data), dtype=np.float64),
            hdus[0].header.copy(),
        )


def _sample_bilinear(
    image: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    output = np.full(x.shape, np.nan, dtype=np.float64)
    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < image.shape[1] - 1)
        & (y < image.shape[0] - 1)
    )
    x0 = np.floor(x[valid]).astype(np.int64)
    y0 = np.floor(y[valid]).astype(np.int64)
    dx = x[valid] - x0
    dy = y[valid] - y0
    output[valid] = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x0 + 1] * dx * (1 - dy)
        + image[y0 + 1, x0] * (1 - dx) * dy
        + image[y0 + 1, x0 + 1] * dx * dy
    )
    return output


def _reproject(
    image: np.ndarray,
    source_header: fits.Header,
    destination_header: fits.Header,
    shape: tuple[int, int],
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    destination_wcs = WCS(destination_header).celestial
    source_wcs = WCS(source_header).celestial
    longitude, latitude = destination_wcs.pixel_to_world_values(x, y)
    source_x, source_y = source_wcs.world_to_pixel_values(
        longitude, latitude
    )
    return _sample_bilinear(image, source_x, source_y)


def _restoring_kernel(
    shape: tuple[int, int],
    *,
    pixel_size_deg: float,
    major_deg: float,
    minor_deg: float,
    position_angle_deg: float,
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    x -= (shape[1] - 1) / 2
    y -= (shape[0] - 1) / 2
    angle = np.deg2rad(position_angle_deg)
    major_axis = x * np.sin(angle) + y * np.cos(angle)
    minor_axis = x * np.cos(angle) - y * np.sin(angle)
    major_sigma = major_deg / (
        2 * np.sqrt(2 * np.log(2)) * pixel_size_deg
    )
    minor_sigma = minor_deg / (
        2 * np.sqrt(2 * np.log(2)) * pixel_size_deg
    )
    return np.asarray(
        np.exp(
            -0.5
            * (
                np.square(major_axis / major_sigma)
                + np.square(minor_axis / minor_sigma)
            )
        ),
        dtype=np.float64,
    )


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    shape = (
        image.shape[0] + kernel.shape[0] - 1,
        image.shape[1] + kernel.shape[1] - 1,
    )
    convolved = np.fft.irfft2(
        np.fft.rfft2(image, shape) * np.fft.rfft2(kernel, shape),
        shape,
    )
    start_y = (kernel.shape[0] - 1) // 2
    start_x = (kernel.shape[1] - 1) // 2
    return convolved[
        start_y : start_y + image.shape[0],
        start_x : start_x + image.shape[1],
    ]


def _metrics(
    actual: np.ndarray, expected: np.ndarray
) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(expected)
    actual_values = actual[valid]
    expected_values = expected[valid]
    return {
        "normalized_rms": float(
            np.sqrt(
                np.sum(np.square(actual_values - expected_values))
                / np.sum(np.square(expected_values))
            )
        ),
        "correlation": float(
            np.corrcoef(actual_values, expected_values)[0, 1]
        ),
        "actual_peak_jy_beam": float(np.max(actual_values)),
        "expected_peak_jy_beam": float(np.max(expected_values)),
        "pixel_count": int(actual_values.size),
    }


def _plot(
    expected: np.ndarray,
    actual: np.ndarray,
    path: Path,
    *,
    label: str,
    casa_label: str,
) -> None:
    difference = actual - expected
    valid_values = np.concatenate(
        (expected[np.isfinite(expected)], actual[np.isfinite(actual)])
    )
    limit = float(np.percentile(np.abs(valid_values), 99.5))
    difference_limit = float(
        np.percentile(np.abs(difference[np.isfinite(difference)]), 99.5)
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(14, 4.5), constrained_layout=True
    )
    for axis, image, title in zip(
        axes[:2],
        (expected, actual),
        (casa_label, label),
        strict=True,
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=-0.05 * limit,
            vmax=limit,
        )
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title(f"{label} − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle("3C391 C1 restored-image comparison")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("casa_clean", type=Path)
    parser.add_argument("jax_results", type=Path, nargs="+")
    parser.add_argument("--casa-label")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/3c391_portable_casa_comparison"),
    )
    arguments = parser.parse_args()
    casa_clean, casa_header = _read(arguments.casa_clean)
    casa_label = arguments.casa_label
    if casa_label is None:
        casa_label = (
            "CASA multiscale CLEAN"
            if "multiscale" in arguments.casa_clean.name
            else "CASA Högbom CLEAN"
        )
    arguments.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "casa_clean": arguments.casa_clean.name,
        "beam": {
            "major_deg": float(casa_header["BMAJ"]),
            "minor_deg": float(casa_header["BMIN"]),
            "position_angle_deg": float(casa_header["BPA"]),
        },
        "comparisons": {},
    }
    comparisons: dict[str, object] = {}
    for result_directory in arguments.jax_results:
        model_path = (
            result_directory / "casa_corrected_reconstruction.fits"
        )
        model, model_header = _read(model_path)
        kernel = _restoring_kernel(
            model.shape,
            pixel_size_deg=abs(float(model_header["CDELT1"])),
            major_deg=float(casa_header["BMAJ"]),
            minor_deg=float(casa_header["BMIN"]),
            position_angle_deg=float(casa_header["BPA"]),
        )
        restored = _convolve_same(model, kernel)
        reprojected = _reproject(
            casa_clean,
            casa_header,
            model_header,
            model.shape,
        )
        label = result_directory.name
        comparisons[label] = _metrics(restored, reprojected)
        _plot(
            reprojected,
            restored,
            arguments.output / f"{label}.png",
            label=label,
            casa_label=casa_label,
        )
        fits.PrimaryHDU(restored, header=model_header).writeto(
            arguments.output / f"{label}.restored.fits",
            overwrite=True,
        )
    results["comparisons"] = comparisons
    (arguments.output / "result.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
