"""Compare SL1MJax direct/gradient images with CASA dirty/CLEAN products."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

CASA = Path("outputs/3c391_casa_imaging")
JAX = Path("outputs/3c391_target")
OUTPUT = Path("outputs/3c391_clean_comparison")


def _read(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdus:
        return (
            np.asarray(np.squeeze(hdus[0].data), dtype=np.float64),
            hdus[0].header.copy(),
        )


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(expected)
    actual_values = actual[valid]
    expected_values = expected[valid]
    return {
        "normalized_rms": float(
            np.sqrt(
                np.sum((actual_values - expected_values) ** 2)
                / np.sum(expected_values**2)
            )
        ),
        "correlation": float(
            np.corrcoef(actual_values, expected_values)[0, 1]
        ),
        "actual_peak": float(np.max(actual_values)),
        "expected_peak": float(np.max(expected_values)),
        "pixel_count": int(actual_values.size),
    }


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    source_x, source_y = source_wcs.world_to_pixel_values(longitude, latitude)
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
    major_sigma = major_deg / (2 * np.sqrt(2 * np.log(2)) * pixel_size_deg)
    minor_sigma = minor_deg / (2 * np.sqrt(2 * np.log(2)) * pixel_size_deg)
    return np.asarray(
        np.exp(
            -0.5
            * (
                (major_axis / major_sigma) ** 2
                + (minor_axis / minor_sigma) ** 2
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


def _plot(
    expected: np.ndarray,
    actual: np.ndarray,
    path: Path,
    *,
    labels: tuple[str, str],
    title: str,
) -> None:
    difference = actual - expected
    valid_values = np.concatenate(
        (expected[np.isfinite(expected)], actual[np.isfinite(actual)])
    )
    limit = float(np.percentile(np.abs(valid_values), 99.5))
    difference_limit = float(
        np.percentile(np.abs(difference[np.isfinite(difference)]), 99.5)
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, label in zip(
        axes[:2], (expected, actual), labels, strict=True
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=-0.05 * limit,
            vmax=limit,
        )
        axis.set_title(label)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title(f"{labels[1]} − {labels[0]}")
    figure.colorbar(shown, ax=axes[2], fraction=0.046)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


OUTPUT.mkdir(parents=True, exist_ok=True)
casa_dirty, casa_dirty_header = _read(
    CASA / "3c391_c1_dirty.residual.fits"
)
jax_dirty, _ = _read(JAX / "casa_corrected_dirty.fits")
dirty_metrics = _metrics(jax_dirty, casa_dirty)
_plot(
    casa_dirty,
    jax_dirty,
    OUTPUT / "dirty_comparison.png",
    labels=("CASA wproject", "SL1MJax direct DFT"),
    title="3C391 C1 dirty-image operator comparison",
)

casa_clean, casa_clean_header = _read(
    CASA / "3c391_c1_hogbom.image.fits"
)
jax_model, jax_header = _read(JAX / "casa_corrected_reconstruction.fits")
kernel = _restoring_kernel(
    jax_model.shape,
    pixel_size_deg=abs(float(jax_header["CDELT1"])),
    major_deg=float(casa_clean_header["BMAJ"]),
    minor_deg=float(casa_clean_header["BMIN"]),
    position_angle_deg=float(casa_clean_header["BPA"]),
)
jax_restored = _convolve_same(jax_model, kernel)
casa_clean_reprojected = _reproject(
    casa_clean,
    casa_clean_header,
    jax_header,
    jax_model.shape,
)
clean_metrics = _metrics(jax_restored, casa_clean_reprojected)
_plot(
    casa_clean_reprojected,
    jax_restored,
    OUTPUT / "clean_comparison.png",
    labels=("CASA Högbom CLEAN", "SL1MJax restored"),
    title="3C391 C1 deconvolution comparison",
)
fits.PrimaryHDU(jax_restored, header=jax_header).writeto(
    OUTPUT / "jax_restored.fits", overwrite=True
)
fits.PrimaryHDU(casa_clean_reprojected, header=jax_header).writeto(
    OUTPUT / "casa_clean_reprojected.fits", overwrite=True
)

result = {
    "schema_version": 1,
    "dirty_operator_comparison": dirty_metrics,
    "deconvolution_comparison": clean_metrics,
    "casa_clean_beam": {
        "major_deg": float(casa_clean_header["BMAJ"]),
        "minor_deg": float(casa_clean_header["BMIN"]),
        "position_angle_deg": float(casa_clean_header["BPA"]),
    },
    "notes": {
        "dirty": "CASA uses full-resolution wproject; JAX uses averaged direct DFT",
        "deconvolution": (
            "CASA uses all target data; JAX uses 1000 averaged rows and a "
            "40x40 positive pixel model"
        ),
    },
}
(OUTPUT / "result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, sort_keys=True))
