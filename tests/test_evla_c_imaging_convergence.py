from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.evla_c_imaging_convergence import (
    MAIN_LOBE_REL_L2_MAX,
    assert_sampling_change,
    classify_imaging_convergence,
    compare_on_production_nodes,
    native_3c391_mhz,
    raster_description,
    unsupported_direction_account,
)
from sl1mjax.evla_c_survey_beam import IMAGING_NODE_MHZ, NATIVE_3C391_MHZ


def test_native_3c391_mhz_is_the_64_channel_spw0_grid() -> None:
    assert NATIVE_3C391_MHZ == tuple(range(4536, 4664, 2))
    assert len(NATIVE_3C391_MHZ) == 64
    assert NATIVE_3C391_MHZ[0] == 4536
    assert NATIVE_3C391_MHZ[-1] == 4662
    assert IMAGING_NODE_MHZ == (4536, 4598, 4662)
    assert set(IMAGING_NODE_MHZ).issubset(NATIVE_3C391_MHZ)
    assert 4599 not in NATIVE_3C391_MHZ
    assert native_3c391_mhz() == NATIVE_3C391_MHZ


def test_assert_sampling_change_refuses_silent_extent_or_scale_swaps() -> None:
    production = {"pixel_scale_arcmin": 0.4, "half_extent_arcmin": 100.0}
    aperture = {"pixel_scale_arcmin": 0.4, "half_extent_arcmin": 200.0}
    angular = {"pixel_scale_arcmin": 0.2, "half_extent_arcmin": 50.0}
    same = assert_sampling_change(production, aperture, kind="aperture")
    assert same["same_pixel_scale"] is True
    assert same["same_half_extent"] is False
    changed = assert_sampling_change(production, angular, kind="angular")
    assert changed["same_pixel_scale"] is False
    with pytest.raises(ValueError, match="image sampling"):
        assert_sampling_change(production, angular, kind="aperture")
    with pytest.raises(ValueError, match="image sampling"):
        assert_sampling_change(production, aperture, kind="angular")


def test_compare_on_production_nodes_classifies_main_lobe() -> None:
    size = 21
    scale = np.deg2rad(0.5 / 60.0)
    axis = (np.arange(size) - 10) * scale
    ll, mm = np.meshgrid(axis, axis)
    radius = np.hypot(ll, mm)
    fall = 0.02 * (radius / scale) ** 2
    production = np.zeros((size, size, 2, 2), dtype=np.complex128)
    production[..., 0, 0] = 1.0 - fall
    production[..., 1, 1] = 1.0 - 0.8 * fall
    candidate = production * (1.0 + 5.0e-4)
    compared = compare_on_production_nodes(
        {"jones": production, "l_rad": axis, "m_rad": axis},
        {"jones": candidate, "l_rad": axis, "m_rad": axis},
    )
    classified = classify_imaging_convergence(compared)
    assert classified["main_lobe_nodes"] > 0
    assert classified["main_lobe_jones_rel_l2"] <= MAIN_LOBE_REL_L2_MAX
    assert classified["keep_production_g1024_p32"] is True
    assert classified["imaging_prerequisite_passed"] is True
    drifted = production.copy()
    drifted[..., 0, 0] *= 1.05
    failed = classify_imaging_convergence(
        compare_on_production_nodes(
            {"jones": production, "l_rad": axis, "m_rad": axis},
            {"jones": drifted, "l_rad": axis, "m_rad": axis},
        )
    )
    assert failed["numerically_qualified_0p002"] is False
    mild = production * (1.0 + 4.0e-3)
    kept = classify_imaging_convergence(
        compare_on_production_nodes(
            {"jones": production, "l_rad": axis, "m_rad": axis},
            {"jones": mild, "l_rad": axis, "m_rad": axis},
        )
    )
    assert kept["numerically_qualified_0p002"] is False
    assert kept["keep_production_g1024_p32"] is True
    assert kept["imaging_prerequisite_passed"] is True


def test_unsupported_direction_account_does_not_treat_missing_as_zero() -> None:
    valid = np.array([True, True, False, False])
    account = unsupported_direction_account(valid)
    assert account["n_supported"] == 2
    assert account["n_unsupported"] == 2
    assert account["missing_treated_as_zero"] is False


def test_raster_description_records_extent_and_sampling(tmp_path) -> None:
    geometry = tmp_path / "vla_geom"
    geometry.write_text("12.5\n", encoding="utf-8")
    coarse = raster_description(
        {"freq": "4.536", "gridsize": "1024", "pixelsperbeam": "32"},
        geometry,
        size=513,
    )
    fine_aperture = raster_description(
        {"freq": "4.536", "gridsize": "2048", "pixelsperbeam": "32"},
        geometry,
        size=1025,
    )
    fine_angular = raster_description(
        {"freq": "4.536", "gridsize": "1024", "pixelsperbeam": "64"},
        geometry,
        size=513,
    )
    assert coarse["pixel_scale_arcmin"] == pytest.approx(fine_aperture["pixel_scale_arcmin"])
    assert fine_aperture["half_extent_arcmin"] > coarse["half_extent_arcmin"]
    assert fine_angular["pixel_scale_arcmin"] < coarse["pixel_scale_arcmin"]
    assert fine_angular["half_extent_arcmin"] < coarse["half_extent_arcmin"]
