from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sl1mjax.composite import MosaicPointComponent, MosaicQuadtreeComponent
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.polarization_activation import (
    compare_linear_polarization_models,
    declare_coarse_stokes_i_regions,
    evaluate_linear_polarization,
    fit_global_linear_polarization,
    fit_regional_linear_polarization,
    leave_one_block_out_scores,
)
from sl1mjax.quadtree import quadtree_sky_from_regular_grid


def _block(
    region_stokes_i: dict[str, np.ndarray],
    q: dict[str, float],
    u: dict[str, float],
    *,
    rows: int = 40,
    channels: int = 4,
    seed: int = 4,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    uvw_m = rng.normal(0.0, 200.0, size=(rows, 3))
    uvw_m[:, 2] = 0.0
    model_i = np.zeros((rows, channels), dtype=np.complex128)
    q_plus_iu = np.zeros((rows, channels), dtype=np.complex128)
    for name, plane in region_stokes_i.items():
        model_i = model_i + plane
        q_plus_iu = q_plus_iu + (q[name] + 1j * u[name]) * plane
    visibility = np.zeros((rows, channels, 4), dtype=np.complex128)
    visibility[..., 0] = model_i
    visibility[..., 3] = model_i
    visibility[..., 1] = q_plus_iu
    visibility[..., 2] = np.conj(q_plus_iu)
    model_visibility = np.zeros_like(visibility)
    model_visibility[..., 0] = model_i
    model_visibility[..., 3] = model_i
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=np.linspace(4.55e9, 4.65e9, channels),
        visibility=visibility,
        model_visibility=model_visibility,
        weight=np.ones_like(visibility, dtype=np.float64),
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64) // 4,
        antenna1=np.arange(rows, dtype=np.int32) % 5,
        antenna2=(
            ((np.arange(rows, dtype=np.int32) % 5) + 1 + (np.arange(rows) // 5) % 3) % 6
        ).astype(np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )


def _phase_plane(rows: int, channels: int, offset: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uvw = rng.normal(0.0, 300.0, size=(rows, 3))
    frequency_hz = np.linspace(4.55e9, 4.65e9, channels)
    wavelength_m = 299792458.0 / frequency_hz
    return np.exp(2j * np.pi * uvw[:, 0, None] * offset / wavelength_m[None, :])


def test_ancestor_cells_group_frozen_i_leaves_without_widefield() -> None:
    flux = np.zeros(16, dtype=np.float64)
    flux[0] = 1.2
    flux[1] = 0.8
    flux[15] = 0.05
    central = quadtree_sky_from_regular_grid(4, np.deg2rad(16 / 3600), flux)
    coarse = quadtree_sky_from_regular_grid(2, np.deg2rad(64 / 3600), np.full(4, 0.01))
    regions = declare_coarse_stokes_i_regions(
        (
            MosaicQuadtreeComponent("central", central.topology, central.flux),
            MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux),
        ),
        scheme="central_ancestor_cells",
        ancestor_arcsec=32.0,
        minimum_region_i_jy=0.2,
        include_widefield=False,
    )
    names = [region.name for region in regions]
    assert "widefield" not in names
    assert all(region.provenance["qu_peaks"] is False for region in regions)
    assert all(region.provenance["selector"] == "frozen_hierarchical_i_ancestor_cells" for region in regions)
    bright = [region for region in regions if region.name != "central_remainder"]
    assert len(bright) >= 1
    assert all(region.stokes_i_jy >= 0.2 for region in bright)
    assert sum(region.stokes_i_jy for region in regions) == pytest.approx(2.05)


def test_regions_come_from_stokes_i_not_qu_peaks() -> None:
    flux = np.asarray([4.0, 0.3, 0.3, 0.3])
    central = quadtree_sky_from_regular_grid(2, 1.0e-3, flux)
    coarse = quadtree_sky_from_regular_grid(2, 4.0e-3, np.full(4, 0.05))
    catalogue = MosaicPointComponent(
        "catalogue",
        l_rad=np.asarray([2.0e-3]),
        m_rad=np.asarray([-1.5e-3]),
        flux=np.asarray([0.2]),
    )
    regions = declare_coarse_stokes_i_regions(
        (
            MosaicQuadtreeComponent("central", central.topology, central.flux),
            MosaicQuadtreeComponent("coarse", coarse.topology, coarse.flux),
            catalogue,
        )
    )
    names = [region.name for region in regions]
    assert names == ["central_inner", "central_outer", "widefield"]
    assert all(region.provenance["qu_peaks"] is False for region in regions)
    inner = next(region for region in regions if region.name == "central_inner")
    assert inner.stokes_i_jy == pytest.approx(4.0)
    assert inner.provenance["selector"] == "stokes_i_weighted_median_radius"


def test_regional_fit_recovers_distinct_qu_and_keeps_v_zero() -> None:
    rows, channels = 48, 4
    inner = 2.5 * _phase_plane(rows, channels, 1.5e-3, seed=1)
    outer = 1.5 * _phase_plane(rows, channels, -2.0e-3, seed=2)
    truth_q = {"inner": 0.08, "outer": -0.03}
    truth_u = {"inner": -0.04, "outer": 0.06}
    block = _block(
        {"inner": inner, "outer": outer},
        truth_q,
        truth_u,
        rows=rows,
        channels=channels,
    )
    regions = {"inner": (inner,), "outer": (outer,)}
    fitted = fit_regional_linear_polarization((block,), regions)
    assert fitted.v == 0.0
    assert fitted.q[0] == pytest.approx(0.08, abs=1e-12)
    assert fitted.u[0] == pytest.approx(-0.04, abs=1e-12)
    assert fitted.q[1] == pytest.approx(-0.03, abs=1e-12)
    assert fitted.u[1] == pytest.approx(0.06, abs=1e-12)
    assert fitted.polarized_linear_loss == pytest.approx(0.0, abs=1e-10)
    assert fitted.provenance["rm"] is False


def test_regional_model_beats_global_when_qu_cancels() -> None:
    rows, channels = 36, 4
    left = 2.0 * _phase_plane(rows, channels, 1.0e-3, seed=5)
    right = 2.0 * _phase_plane(rows, channels, -1.0e-3, seed=6)
    block = _block(
        {"left": left, "right": right},
        {"left": 0.05, "right": -0.05},
        {"left": 0.04, "right": -0.04},
        rows=rows,
        channels=channels,
    )
    regions = {"left": (left,), "right": (right,)}
    compared = compare_linear_polarization_models((block,), regions)
    assert compared["unpolarised"].v == compared["global"].v == compared["regional"].v == 0.0
    assert compared["regional"].polarized_linear_loss < compared["global"].polarized_linear_loss
    assert compared["global"].polarized_linear_loss <= compared["unpolarised"].polarized_linear_loss
    global_fit = fit_global_linear_polarization((block,), regions)
    assert abs(global_fit.q[0]) < 0.02


def test_leave_one_pointing_out_transfers_regional_qu() -> None:
    rows, channels = 32, 4
    first_inner = 2.0 * _phase_plane(rows, channels, 1.2e-3, seed=8)
    first_outer = 1.2 * _phase_plane(rows, channels, -1.8e-3, seed=9)
    second_inner = 2.0 * _phase_plane(rows, channels, 1.2e-3, seed=10)
    second_outer = 1.2 * _phase_plane(rows, channels, -1.8e-3, seed=11)
    q = {"inner": 0.06, "outer": -0.02}
    u = {"inner": 0.01, "outer": 0.05}
    first = _block({"inner": first_inner, "outer": first_outer}, q, u, rows=rows, channels=channels, seed=8)
    second = _block(
        {"inner": second_inner, "outer": second_outer}, q, u, rows=rows, channels=channels, seed=10
    )
    regions = {
        "inner": (first_inner, second_inner),
        "outer": (first_outer, second_outer),
    }
    scores = leave_one_block_out_scores((first, second), regions, labels=("C1", "C2"))
    held = scores["C2"]["holdout_scores"]
    assert held["regional"]["linear_loss_ratio"] < held["global"]["linear_loss_ratio"]
    assert held["regional"]["linear_loss_ratio"] < 1.0e-6
    assert scores["C2"]["fits"]["regional"]["v"] == 0.0


def test_evaluate_rejects_mismatched_region_coefficients() -> None:
    plane = np.ones((16, 2), dtype=np.complex128)
    block = _block({"a": plane}, {"a": 0.01}, {"a": 0.0}, rows=16, channels=2)
    with pytest.raises(ValueError, match="one value per region"):
        evaluate_linear_polarization((block,), {"a": (plane,)}, (0.01, 0.02), (0.0, 0.0))
