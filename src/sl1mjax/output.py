"""FITS, residual, checkpoint, and diagnostic products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from sl1mjax.imaging import ImagingResult
from sl1mjax.inference import save_checkpoint


def write_products(
    result: ImagingResult, image_path: str | Path
) -> tuple[Path, ...]:
    destination = Path(image_path)
    if destination.suffix.lower() not in {".fits", ".fit", ".fts"}:
        raise ValueError("image output must be FITS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["BUNIT"] = "Jy/pixel"
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX1"] = (result.grid.size + 1) / 2
    header["CRPIX2"] = (result.grid.size + 1) / 2
    header["CRVAL1"] = np.rad2deg(result.phase_centre_rad[0])
    header["CRVAL2"] = np.rad2deg(result.phase_centre_rad[1])
    header["CDELT1"] = -np.rad2deg(result.grid.pixel_size_rad)
    header["CDELT2"] = np.rad2deg(result.grid.pixel_size_rad)
    fits.PrimaryHDU(result.image, header=header).writeto(destination, overwrite=True)

    diagnostics = destination.with_suffix(".json")
    diagnostics.write_text(
        json.dumps(result.diagnostics(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    residuals = destination.with_suffix(".residuals.npz")
    residual_payload: dict[str, Any] = {
        "prediction": result.prediction,
        "residual": result.residual,
        "correlations": np.asarray(result.correlations),
    }
    additional_products: list[Path] = []
    if result.residual_evaluation is not None:
        evaluation = result.residual_evaluation
        residual_payload.update(
            {
                "full_residual_dirty": evaluation.full_dirty,
                "train_residual_dirty": evaluation.train_dirty,
                "holdout_residual_dirty": evaluation.holdout_dirty,
                "psf": evaluation.psf,
            }
        )
        for label, image, unit in (
            ("full-residual-dirty", evaluation.full_dirty, "Jy/beam"),
            ("train-residual-dirty", evaluation.train_dirty, "Jy/beam"),
            (
                "holdout-residual-dirty",
                evaluation.holdout_dirty,
                "Jy/beam",
            ),
            ("psf", evaluation.psf, "1"),
        ):
            product = destination.with_name(
                f"{destination.stem}.{label}.fits"
            )
            product_header = header.copy()
            product_header["BUNIT"] = unit
            fits.PrimaryHDU(image, header=product_header).writeto(
                product, overwrite=True
            )
            additional_products.append(product)
    with residuals.open("wb") as stream:
        np.savez(stream, **residual_payload)
    checkpoint = destination.with_suffix(".checkpoint.npz")
    save_checkpoint(checkpoint, result.inference)
    return (
        destination,
        diagnostics,
        residuals,
        checkpoint,
        *additional_products,
    )
