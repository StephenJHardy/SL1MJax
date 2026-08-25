"""Render restored-image comparisons for the joint 3C391 mosaic fit."""

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


def _restore(
    model_path: Path,
    casa_header: fits.Header,
    output_path: Path,
    *,
    maximum_pixel_size_arcsec: float = 4.0,
) -> tuple[np.ndarray, fits.Header]:
    model, model_header = _read(model_path)
    pixel_size_deg = abs(float(model_header["CDELT1"]))
    target_pixel_size_deg = maximum_pixel_size_arcsec / 3600.0
    if pixel_size_deg > target_pixel_size_deg * (1 + 1e-9):
        factor = int(round(pixel_size_deg / target_pixel_size_deg))
        if not np.isclose(pixel_size_deg / factor, target_pixel_size_deg):
            raise ValueError("model pixel size is not an integer multiple of target size")
        model = np.repeat(np.repeat(model, factor, axis=0), factor, axis=1)
        model /= factor**2
        model_header["CRPIX1"] = factor * (float(model_header["CRPIX1"]) - 0.5) + 0.5
        model_header["CRPIX2"] = factor * (float(model_header["CRPIX2"]) - 0.5) + 0.5
        model_header["CDELT1"] = float(model_header["CDELT1"]) / factor
        model_header["CDELT2"] = float(model_header["CDELT2"]) / factor
        pixel_size_deg /= factor
    kernel = _restoring_kernel(
        model.shape,
        pixel_size_deg=pixel_size_deg,
        major_deg=float(casa_header["BMAJ"]),
        minor_deg=float(casa_header["BMIN"]),
        position_angle_deg=float(casa_header["BPA"]),
    )
    restored = _convolve_same(model, kernel)
    header = model_header.copy()
    header["BUNIT"] = "Jy/beam"
    header["BMAJ"] = float(casa_header["BMAJ"])
    header["BMIN"] = float(casa_header["BMIN"])
    header["BPA"] = float(casa_header["BPA"])
    fits.PrimaryHDU(restored, header=header).writeto(output_path, overwrite=True)
    return restored, header


def _crop_overlap(
    expected: np.ndarray,
    *actual: np.ndarray,
) -> tuple[slice, slice]:
    valid = np.isfinite(expected)
    for image in actual:
        valid &= np.isfinite(image)
    y, x = np.nonzero(valid)
    if y.size == 0:
        raise ValueError("images have no finite overlap")
    return slice(int(y.min()), int(y.max()) + 1), slice(int(x.min()), int(x.max()) + 1)


def _percentile(image: np.ndarray, value: float = 99.5) -> float:
    finite = np.abs(image[np.isfinite(image)])
    if finite.size == 0:
        raise ValueError("image contains no finite values")
    return float(np.percentile(finite, value))


def _plot_casa_comparison(
    casa: np.ndarray,
    hierarchical: np.ndarray,
    path: Path,
    *,
    pointing_count: int,
) -> None:
    crop = _crop_overlap(casa, hierarchical)
    casa_crop = casa[crop]
    hierarchy_crop = hierarchical[crop]
    difference = hierarchy_crop - casa_crop
    limit = _percentile(np.concatenate((casa_crop, hierarchy_crop), axis=0))
    difference_limit = _percentile(difference)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title in zip(
        axes[:2],
        (casa_crop, hierarchy_crop),
        ("CASA C1 multiscale CLEAN", f"Joint C1–C{pointing_count} hierarchy"),
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
    axes[2].set_title("Joint hierarchy − CASA")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, label="Jy/beam")
    figure.suptitle("3C391 joint-mosaic restored image on the CASA C1 field")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_model_comparison(
    baseline: np.ndarray,
    hierarchical: np.ndarray,
    path: Path,
    *,
    pointing_count: int,
    field_of_view_arcmin: float,
) -> None:
    difference = hierarchical - baseline
    limit = _percentile(np.concatenate((baseline, hierarchical), axis=0))
    difference_limit = _percentile(difference)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title in zip(
        axes[:2],
        (baseline, hierarchical),
        ("Joint 16-arcsec base grid", "Joint hierarchical grid"),
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
    axes[2].set_title("Hierarchy − base grid")
    figure.colorbar(shown, ax=axes[2], fraction=0.046, label="Jy/beam")
    figure.suptitle(
        f"3C391 C1–C{pointing_count} joint fit over the {field_of_view_arcmin:.2f}-arcmin field"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mosaic_directory", type=Path)
    parser.add_argument(
        "--casa-image",
        type=Path,
        default=Path("outputs/3c391_casa_imaging_128/3c391_c1_multiscale.image.fits"),
    )
    arguments = parser.parse_args()
    summary = json.loads((arguments.mosaic_directory / "summary.json").read_text(encoding="utf-8"))
    hierarchical_name = (
        "hierarchical_reconstruction.fits"
        if (arguments.mosaic_directory / "hierarchical_reconstruction.fits").exists()
        else "consensus_reconstruction.fits"
    )
    casa_image, casa_header = _read(arguments.casa_image)
    baseline, baseline_header = _restore(
        arguments.mosaic_directory / "baseline_reconstruction.fits",
        casa_header,
        arguments.mosaic_directory / "baseline_reconstruction.restored.fits",
    )
    hierarchical, hierarchical_header = _restore(
        arguments.mosaic_directory / hierarchical_name,
        casa_header,
        arguments.mosaic_directory / hierarchical_name.replace(".fits", ".restored.fits"),
    )
    baseline_reprojected = _reproject(
        baseline,
        baseline_header,
        hierarchical_header,
        hierarchical.shape,
    )
    casa_reprojected = _reproject(
        casa_image,
        casa_header,
        hierarchical_header,
        hierarchical.shape,
    )
    pointing_count = int(summary["pointing_count"])
    field_of_view_arcmin = float(
        summary.get(
            "field_of_view_arcmin",
            summary["root_size"] * summary["root_pixel_arcsec"] / 60.0,
        )
    )
    _plot_casa_comparison(
        casa_reprojected,
        hierarchical,
        arguments.mosaic_directory / "hierarchical_casa_c1_comparison.jpg",
        pointing_count=pointing_count,
    )
    _plot_model_comparison(
        baseline_reprojected,
        hierarchical,
        arguments.mosaic_directory / "baseline_hierarchical_comparison.jpg",
        pointing_count=pointing_count,
        field_of_view_arcmin=field_of_view_arcmin,
    )
    assert baseline_header["BUNIT"] == hierarchical_header["BUNIT"]
    print(
        json.dumps(
            {
                "baseline_peak_jy_beam": float(np.max(baseline)),
                "hierarchical_peak_jy_beam": float(np.max(hierarchical)),
                "pointing_count": pointing_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
