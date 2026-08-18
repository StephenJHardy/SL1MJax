"""Small extensible CASA golden suite for sky, noise, and calibration effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sl1mjax.polarization import Correlation
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import (
    DeltaPixelBasis,
    GaussianApproximation,
    GaussianPixelBasis,
    PixelBasis,
)

FIXTURE = Path(__file__).parent / "fixtures" / "casa_vla_golden.npz"
CASES = ("center", "east", "north", "diagonal", "gaussian")


@dataclass(frozen=True)
class CasaGoldenCase:
    """One compact CASA reference case and its declared physical effects."""

    name: str
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, name: str) -> CasaGoldenCase:
        suite = json.loads(FIXTURE.with_suffix(".json").read_text(encoding="utf-8"))
        if suite["schema_version"] != 1:
            raise ValueError(f"unsupported golden schema {suite['schema_version']}")
        prefix = f"{name}_"
        with np.load(FIXTURE) as fixture:
            arrays = {
                field: np.asarray(fixture[prefix + field])
                for field in (
                    "uvw_m",
                    "frequency_hz",
                    "visibility",
                    "weight",
                    "flag",
                    "antenna1",
                    "antenna2",
                )
            }
        return cls(name, suite["cases"][name], arrays)

    def sky_basis(self) -> tuple[PixelBasis, float | None]:
        sources = self.metadata["truth"]["sources"]
        shapes = [source.get("shape", {"kind": "point"}) for source in sources]
        kinds = {shape["kind"] for shape in shapes}
        if kinds == {"point"}:
            return DeltaPixelBasis(), None
        if kinds == {"circular_gaussian"}:
            fwhm = np.asarray([shape["fwhm_major_rad"] for shape in shapes])
            if not np.allclose(fwhm, fwhm[0], rtol=0, atol=0):
                raise ValueError("current golden adapter requires one shared Gaussian width")
            sigma_rad = float(fwhm[0] / (2 * np.sqrt(2 * np.log(2))))
            return (
                GaussianPixelBasis(1.0, GaussianApproximation.WIDE_FIELD),
                sigma_rad,
            )
        raise ValueError(f"unsupported golden sky shapes {sorted(kinds)}")

    def prediction(self) -> np.ndarray:
        effects = self.metadata["effects"]
        if effects != {"noise": "none", "calibration": "identity"}:
            raise ValueError(f"unsupported golden effects {effects}")
        sources = self.metadata["truth"]["sources"]
        pixel_basis, pixel_size_rad = self.sky_basis()
        return np.asarray(
            predict_stokes_i(
                np.asarray([source["flux_jy"] for source in sources]),
                np.asarray([source["l"] for source in sources]),
                np.asarray([source["m"] for source in sources]),
                self.arrays["uvw_m"],
                self.arrays["frequency_hz"],
                self.arrays["antenna1"],
                self.arrays["antenna2"],
                tuple(Correlation(value) for value in self.metadata["correlations"]),
                pixel_basis=pixel_basis,
                pixel_size_rad=pixel_size_rad,
            )
        )

    def normalized_complex_rms(self, prediction: np.ndarray) -> float:
        observed = self.arrays["visibility"]
        active_weight = np.where(
            self.arrays["flag"], 0.0, self.arrays["weight"]
        )
        return float(
            np.sqrt(
                np.sum(active_weight * np.abs(prediction - observed) ** 2)
                / np.sum(active_weight * np.abs(observed) ** 2)
            )
        )


@pytest.mark.parametrize("case_name", CASES)
def test_casa_sky_visibility_golden(case_name: str) -> None:
    case = CasaGoldenCase.load(case_name)
    predicted = case.prediction()
    assert case.normalized_complex_rms(predicted) < 2e-4

    if case_name != "center":
        assert case.normalized_complex_rms(np.conj(predicted)) > 0.05


def test_casa_gaussian_case_is_resolved_and_uses_fwhm_convention() -> None:
    case = CasaGoldenCase.load("gaussian")
    delta_prediction = np.asarray(
        predict_stokes_i(
            np.asarray([1.0]),
            np.asarray([case.metadata["truth"]["sources"][0]["l"]]),
            np.asarray([case.metadata["truth"]["sources"][0]["m"]]),
            case.arrays["uvw_m"],
            case.arrays["frequency_hz"],
            case.arrays["antenna1"],
            case.arrays["antenna2"],
            tuple(Correlation(value) for value in case.metadata["correlations"]),
        )
    )
    assert case.normalized_complex_rms(delta_prediction) > 0.1


def test_casa_fixture_direction_labels_have_expected_handedness() -> None:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    center = metadata["cases"]["center"]["truth"]["sources"][0]
    east = metadata["cases"]["east"]["truth"]["sources"][0]
    north = metadata["cases"]["north"]["truth"]["sources"][0]
    assert abs(center["l"]) < 1e-12
    assert abs(center["m"]) < 1e-12
    assert east["l"] > 0
    assert north["m"] > 0
