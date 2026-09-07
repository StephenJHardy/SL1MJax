from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.holography_alignment import VOLTAGE_MAINLOBE, VOLTAGE_MID
from sl1mjax.holography_cassbeam_holoraster_report import (
    CASSBEAM_DIAGONAL_CORRECTION,
    FULL_JONES_EXPERIMENTAL_LABEL,
    HOLORASTER_CASSBEAM_COMPARISON,
    binned_complex_map,
    classify_diagonal_region_support,
    classify_holoraster_comparison_report,
    locked_convention,
    mainlobe_power_mask,
    power_centroid_lm,
    quadrant_crosshand_summaries,
    refuse_convention_search,
    region_copolar_summaries,
    representative_baseline_ids,
    spatial_quadrant_masks,
    squint_from_voltage_maps,
    visibility_hand_summaries,
    weighted_hand_stats,
    write_holoraster_comparison_plots,
)
from sl1mjax.holography_highres_cassbeam import DEFAULT_CONVENTION


def _raster(n: int = 80) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    radius = np.linspace(0.0, 0.012, n)
    pa = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    offset = np.stack([radius * np.sin(pa), radius * np.cos(pa)], axis=1)
    voltage = np.exp(-0.5 * (radius / 0.003) ** 2)
    intensity = np.full(n, 8.0)
    diag = np.zeros((n, 2, 2), dtype=np.complex128)
    diag[:, 0, 0] = intensity * voltage
    diag[:, 1, 1] = intensity * voltage
    full = diag.copy()
    full[:, 0, 1] = 0.02 * intensity * voltage * np.exp(1j * pa)
    full[:, 1, 0] = 0.018 * intensity * voltage * np.exp(-1j * pa)
    measured = full + 0.01 * (
        rng.standard_normal(full.shape) + 1j * rng.standard_normal(full.shape)
    )
    weight = np.ones_like(measured, dtype=np.float64)
    return {
        "offset": offset,
        "voltage": voltage,
        "intensity": intensity,
        "diag": diag,
        "full": full,
        "measured": measured,
        "weight": weight,
        "mask": np.ones(n, dtype=bool),
        "rr_ok": np.ones(n, dtype=bool),
        "ll_ok": np.ones(n, dtype=bool),
        "mover": np.where(np.arange(n) < n // 2, 4, 7).astype(np.int32),
        "reference": np.full(n, 2, dtype=np.int32),
    }


def test_locked_convention_is_not_searched() -> None:
    assert locked_convention() == DEFAULT_CONVENTION
    refuse_convention_search([DEFAULT_CONVENTION])
    with pytest.raises(RuntimeError, match="does not search"):
        refuse_convention_search([DEFAULT_CONVENTION, DEFAULT_CONVENTION])


def test_weighted_stats_recover_one_to_one_and_scale() -> None:
    pred = np.array([1.0 + 0.2j, 2.0 - 0.1j, 0.5 + 0.0j], dtype=np.complex128)
    weight = np.ones(pred.size)
    same = weighted_hand_stats(pred, pred, weight)
    assert same["residual_power"] == pytest.approx(0.0)
    assert same["slope_real"] == pytest.approx(1.0)
    assert same["correlation_abs"] == pytest.approx(1.0)
    scaled = weighted_hand_stats(2.0 * pred, pred, weight)
    assert scaled["slope_real"] == pytest.approx(2.0)
    assert scaled["residual_power"] == pytest.approx(0.25)
    assert scaled["median_abs_ratio"] == pytest.approx(2.0)


def test_quadrants_separate_east_west_from_north_south() -> None:
    offset = np.array(
        [[0.002, 0.0], [-0.002, 0.0], [0.0, 0.002], [0.0, -0.002]],
        dtype=np.float64,
    )
    masks = spatial_quadrant_masks(offset)
    assert np.array_equal(masks["east_west"], [True, True, False, False])
    assert np.array_equal(masks["north_south"], [False, False, True, True])


def test_region_and_quadrant_summaries_are_vectorized() -> None:
    data = _raster()
    measured = data["measured"][:, None, :, :]
    diag = data["diag"][:, None, :, :]
    full = data["full"][:, None, :, :]
    weight = data["weight"][:, None, :, :]
    scored = region_copolar_summaries(
        measured,
        diag,
        weight,
        data["intensity"],
        data["rr_ok"],
        data["ll_ok"],
        data["mask"],
        voltage=data["voltage"],
    )
    assert "main_lobe" in scored
    assert scored["hand_residual_power"]["main_lobe"]["rr"]["n"] > 0
    assert scored["main_lobe"]["n"] == int(np.sum(data["voltage"] >= VOLTAGE_MAINLOBE))
    assert scored["mid"]["n"] == int(
        np.sum((data["voltage"] >= VOLTAGE_MID) & (data["voltage"] < VOLTAGE_MAINLOBE))
    )
    quads = quadrant_crosshand_summaries(
        measured,
        full,
        diag,
        weight,
        data["offset"],
        data["mask"],
    )
    assert quads["full_jones_label"] == FULL_JONES_EXPERIMENTAL_LABEL
    assert quads["east_west"]["rl"]["correlation_abs"] > 0.7
    source = inspect.getsource(visibility_hand_summaries)
    assert "for row" not in source
    assert "for sample" not in source


def test_squint_uses_independent_20pct_mainlobe_masks() -> None:
    offset = np.array(
        [[0.001, 0.0], [-0.001, 0.0], [0.012, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    rr = np.array([1.0, 0.04, 0.05, 0.02])
    ll = np.array([0.04, 1.0, 0.05, 0.02])
    rr_w = np.array([1.0, 1.0, 1.0, 0.0])
    ll_w = np.array([1.0, 1.0, 1.0, 1.0])
    assert int(np.sum(mainlobe_power_mask(rr, rr_w))) == 1
    biased = power_centroid_lm(offset, rr, rr_w)
    lobe = power_centroid_lm(offset, rr, rr_w, mask=mainlobe_power_mask(rr, rr_w))
    assert lobe[0] == pytest.approx(0.001)
    assert biased[0] > lobe[0]
    squint = squint_from_voltage_maps(
        offset,
        rr,
        ll,
        rr_w,
        ll_w,
        frequency_hz=4.564e9,
        series="measured",
    )
    assert squint["estimator"] == "mainlobe_20pct_peak"
    assert squint["publication_estimator"] is True
    assert squint["full_raster_is_publication_estimator"] is False
    assert squint["separation_arcmin"] == pytest.approx(2.0 * 0.001 * 180.0 * 60.0 / np.pi)
    assert squint["full_raster_separation_arcmin"] != pytest.approx(squint["separation_arcmin"])
    assert squint["n_rr"] == 1
    assert squint["n_ll"] == 1


def test_binned_map_and_representative_baselines() -> None:
    data = _raster()
    mapped = binned_complex_map(data["offset"], data["diag"][:, 0, 0], data["weight"][:, 0, 0])
    assert mapped["mean"].shape == (51, 51)
    assert np.isfinite(mapped["mean"]).any()
    pairs = representative_baseline_ids(data["mover"], data["reference"], data["mask"], count=2)
    assert pairs.shape[0] == 2
    assert set(map(tuple, pairs.tolist())) == {(4, 2), (7, 2)}


def test_classifier_never_selects_a_model() -> None:
    support = classify_diagonal_region_support(
        {
            "main_lobe": {"rr": {"residual_power": 0.006}, "ll": {"residual_power": 0.008}},
            "mid": {"rr": {"residual_power": 0.08}, "ll": {"residual_power": 0.09}},
            "outer_diagnostic": {"rr": {"residual_power": 0.32}, "ll": {"residual_power": 0.35}},
        }
    )
    assert support["main_lobe"] == "accepted"
    assert support["mid_beam"] == "qualified"
    assert support["outer_raster"] == "diagnostic"
    assert support["regions"]["main_lobe"]["matches_class"] is True
    gate = classify_holoraster_comparison_report(
        software_ok=True,
        predictions_finite=True,
        plots_written=True,
        diagonal_support=support,
    )
    assert gate["gate"] == HOLORASTER_CASSBEAM_COMPARISON
    assert gate["decision"] == "report_written"
    assert gate["model_selected"] is False
    assert gate["full_jones_label"] == FULL_JONES_EXPERIMENTAL_LABEL
    assert gate["full_jones_frozen"] is False
    assert gate["spw5_closed"] is True
    assert gate["spw5_full_jones_sealed"] is True
    assert gate["convention_search_reopened"] is False
    assert gate["most_important_next_artifact"] == CASSBEAM_DIAGONAL_CORRECTION
    failed = classify_holoraster_comparison_report(
        software_ok=False,
        predictions_finite=True,
        plots_written=True,
    )
    assert failed["process_failure"] is True
    assert failed["model_selected"] is False


def test_comparison_plots_write_the_requested_families(tmp_path: Path | None = None) -> None:
    data = _raster()
    measured = data["measured"][:, None, :, :]
    diag = data["diag"][:, None, :, :]
    full = data["full"][:, None, :, :]
    weight = data["weight"][:, None, :, :]
    root = Path(tmp_path or tempfile.mkdtemp())
    written = write_holoraster_comparison_plots(
        root,
        measured=measured,
        predicted_diag=diag,
        predicted_full=full,
        weight=weight,
        offset_lm_rad=data["offset"],
        moving_id=data["mover"],
        reference_id=data["reference"],
        row_mask=data["mask"],
        channel_id=np.array([32]),
        squint_records=[
            {
                "frequency_hz": 4.5e9,
                "rr_l_arcmin": 0.1,
                "rr_m_arcmin": 0.0,
                "ll_l_arcmin": -0.1,
                "ll_m_arcmin": 0.0,
                "separation_arcmin": 0.2,
            },
            {
                "frequency_hz": 4.6e9,
                "rr_l_arcmin": 0.09,
                "rr_m_arcmin": 0.0,
                "ll_l_arcmin": -0.09,
                "ll_m_arcmin": 0.0,
                "separation_arcmin": 0.18,
            },
        ],
    )
    names = {Path(path).name for path in written}
    assert "obs_vs_diag_rr_ll.png" in names
    assert "spatial_rr_measured_cassbeam_residual.png" in names
    assert "spatial_ll_measured_cassbeam_residual.png" in names
    assert "residual_vs_radius_azimuth_channel.png" in names
    assert "obs_vs_full_jones_rl_lr.png" in names
    assert "crosshand_quadrant_correlation.png" in names
    assert any(name.startswith("baseline_mover") for name in names)
