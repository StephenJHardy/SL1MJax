"""Render comparable image products from a frozen 3C391 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from compare_3c391_portable_casa import (
    _convolve_same,
    _read,
    _reproject,
    _restoring_kernel,
)


def _finite_percentile(image: np.ndarray, percentile: float) -> float:
    values = np.abs(image[np.isfinite(image)])
    if values.size == 0:
        raise ValueError("image contains no finite values")
    return float(np.percentile(values, percentile))


def _valid_crop(*images: np.ndarray) -> tuple[slice, slice]:
    valid = np.logical_and.reduce(tuple(np.isfinite(image) for image in images))
    y, x = np.nonzero(valid)
    if y.size == 0:
        raise ValueError("images have no overlapping finite pixels")
    return slice(int(y.min()), int(y.max()) + 1), slice(int(x.min()), int(x.max()) + 1)


def _write_restored(
    protocol_directory: Path,
    casa_image_path: Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray]:
    model, model_header = _read(protocol_directory / "frozen_reconstruction.fits")
    casa_image, casa_header = _read(casa_image_path)
    kernel = _restoring_kernel(
        model.shape,
        pixel_size_deg=abs(float(model_header["CDELT1"])),
        major_deg=float(casa_header["BMAJ"]),
        minor_deg=float(casa_header["BMIN"]),
        position_angle_deg=float(casa_header["BPA"]),
    )
    restored = _convolve_same(model, kernel)
    restored_header = model_header.copy()
    restored_header["BUNIT"] = "Jy/beam"
    restored_header["BMAJ"] = float(casa_header["BMAJ"])
    restored_header["BMIN"] = float(casa_header["BMIN"])
    restored_header["BPA"] = float(casa_header["BPA"])
    fits.PrimaryHDU(restored, header=restored_header).writeto(
        protocol_directory / "frozen_reconstruction.restored.fits",
        overwrite=True,
    )
    casa_reprojected = _reproject(
        casa_image,
        casa_header,
        restored_header,
        restored.shape,
    )
    return restored, restored_header, casa_reprojected


def _plot_reconstruction(
    restored: np.ndarray,
    casa_reprojected: np.ndarray,
    output: Path,
) -> None:
    crop = _valid_crop(restored, casa_reprojected)
    sl1mjax = restored[crop]
    casa = casa_reprojected[crop]
    limit = _finite_percentile(np.concatenate((sl1mjax, casa), axis=0), 99.5)
    difference = sl1mjax - casa
    difference_limit = _finite_percentile(difference, 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title in zip(
        axes[:2],
        (casa, sl1mjax),
        ("CASA multiscale CLEAN", "SL1MJax frozen consensus"),
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
        figure.colorbar(shown, ax=axis, fraction=0.046, label="Jy/beam")
    shown = axes[2].imshow(
        difference,
        origin="lower",
        cmap="coolwarm",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("SL1MJax − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, label="Jy/beam")
    figure.suptitle("3C391 restored images on the CASA field of view")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_residuals(protocol_directory: Path, output: Path) -> None:
    train, _ = _read(protocol_directory / "frozen_train_residual_dirty.fits")
    test, _ = _read(protocol_directory / "frozen_test_residual_dirty.fits")
    full, _ = _read(protocol_directory / "frozen_full_residual_dirty.fits")
    limit = _finite_percentile(np.concatenate((train, test, full), axis=0), 99.5)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (train, test, full),
        ("Outer train scans", "Sealed outer-test scans", "All scans"),
        strict=True,
    ):
        shown = axis.imshow(
            image,
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        rms = float(np.sqrt(np.mean(np.square(image))))
        axis.set_title(f"{title}\nimage RMS {rms:.4f} Jy/beam")
        figure.colorbar(shown, ax=axis, fraction=0.046, label="Jy/beam")
    figure.suptitle("3C391 frozen-protocol residual dirty images (common scale)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_directory", type=Path)
    parser.add_argument(
        "--casa-image",
        type=Path,
        default=Path(
            "outputs/3c391_casa_imaging_128/3c391_c1_multiscale.image.fits"
        ),
    )
    arguments = parser.parse_args()

    summary_path = arguments.protocol_directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    restored, _, casa_reprojected = _write_restored(
        arguments.protocol_directory,
        arguments.casa_image,
    )
    _plot_reconstruction(
        restored,
        casa_reprojected,
        arguments.protocol_directory / "frozen_casa_multiscale_comparison.jpg",
    )
    _plot_residuals(
        arguments.protocol_directory,
        arguments.protocol_directory / "frozen_residuals.jpg",
    )
    print(
        json.dumps(
            {
                "test_scans": summary["outer_split"]["test_scans"],
                "test_improvement": summary[
                    "test_improvement_over_unrefined"
                ]["relative"],
                "restored_peak_jy_beam": float(np.max(restored)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
