from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.beam_validation_claims import claim_registry, refuse_full_raster_squint
from sl1mjax.beam_validation_outputs import (
    PUBLICATION_VERSION,
    contains_banned_load_path,
    load_bundle,
    write_validation_bundle,
)
from sl1mjax.beam_validation_plots import plot_offset_ring, write_all_figures
from sl1mjax.beam_validation_statistics import (
    build_claims,
    classify_diagonal_region_support,
    complex_visibility_score,
    phase_valid_mask,
    publication_squint_pair,
    residual_power_table,
    scientific_voltage_masks,
    visibility_hand_weight,
)


def _hand_stats(
    *, residual_power: float, correlation: float = 0.97, n: int = 100
) -> dict[str, float]:
    return {
        "n": n,
        "slope_real": 1.03,
        "slope_imag": 0.004,
        "correlation_abs": correlation,
        "residual_power": residual_power,
        "median_abs_obs": 1.0,
        "median_abs_pred": 0.2,
        "median_abs_ratio": 1.6 if residual_power > 0.1 else 1.05,
    }


def manufactured_products() -> dict[str, object]:
    residual = {
        hand: {
            "median_vs_radius": {"x": [1.0, 10.0], "median": [0.01, 0.2]},
            "median_vs_azimuth": {"x": [-90.0, 90.0], "median": [0.05, 0.06]},
        }
        for hand in ("rr", "ll", "rl", "lr")
    }
    quadrants = {
        name: {
            "rl": _hand_stats(residual_power=0.99, correlation=0.08),
            "lr": _hand_stats(residual_power=0.99, correlation=0.07),
        }
        for name in ("east", "west", "north", "south", "east_west", "north_south")
    }
    channel32 = {
        "diagonal": {
            "rr": _hand_stats(residual_power=0.057),
            "ll": _hand_stats(residual_power=0.063),
            "rl": _hand_stats(residual_power=0.99, correlation=0.09),
            "lr": _hand_stats(residual_power=0.99, correlation=0.10),
        },
        "experimental_full_jones": {
            "rr": _hand_stats(residual_power=0.057),
            "ll": _hand_stats(residual_power=0.063),
            "rl": _hand_stats(residual_power=0.994, correlation=0.05),
            "lr": _hand_stats(residual_power=0.993, correlation=0.06),
        },
        "regions": {
            "main_lobe": {
                "median_abs_rr_over_i": 0.0483,
                "median_abs_ll_over_i": 0.0504,
                "median_abs_rr_jy": 0.388,
                "median_abs_ll_jy": 0.405,
            },
            "mid": {
                "median_abs_rr_over_i": 0.0668,
                "median_abs_ll_over_i": 0.0769,
                "median_abs_rr_jy": 0.536,
                "median_abs_ll_jy": 0.617,
            },
            "outer_diagnostic": {
                "median_abs_rr_over_i": 0.0354,
                "median_abs_ll_over_i": 0.0357,
                "median_abs_rr_jy": 0.284,
                "median_abs_ll_jy": 0.286,
            },
            "all": {
                "median_abs_rr_over_i": 0.0377,
                "median_abs_ll_over_i": 0.0381,
                "median_abs_rr_jy": 0.302,
                "median_abs_ll_jy": 0.306,
            },
            "hand_residual_power": {
                "main_lobe": {
                    "rr": _hand_stats(residual_power=0.0064),
                    "ll": _hand_stats(residual_power=0.0076),
                },
                "mid": {
                    "rr": _hand_stats(residual_power=0.077),
                    "ll": _hand_stats(residual_power=0.089),
                },
                "outer_diagnostic": {
                    "rr": _hand_stats(residual_power=0.32),
                    "ll": _hand_stats(residual_power=0.35),
                },
                "all": {
                    "rr": _hand_stats(residual_power=0.057),
                    "ll": _hand_stats(residual_power=0.063),
                },
            }
        },
        "quadrants": quadrants,
        "residual_geometry": residual,
        "n_rows": 100,
        "frequency_hz": 4.564e9,
        "model_selected": False,
        "full_jones_frozen": False,
        "production_factory_modified": False,
    }
    frequency = {
        "channels": [
            {
                "channel": 32,
                "frequency_hz": 4.564e9,
                "diagonal": channel32["diagonal"],
                "experimental_full_jones": channel32["experimental_full_jones"],
            }
        ]
    }
    squint = {
        "measured": {
            "series": "measured",
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
            "rr_l_arcmin": 0.14,
            "rr_m_arcmin": -0.11,
            "ll_l_arcmin": 0.0,
            "ll_m_arcmin": 0.38,
            "separation_arcmin": 0.515,
            "full_raster_separation_arcmin": 0.388,
            "memo195_separation_arcmin": 0.526,
            "frequency_hz": 4.564e9,
        },
        "cassbeam": {
            "series": "cassbeam",
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
            "rr_l_arcmin": 0.21,
            "rr_m_arcmin": 0.15,
            "ll_l_arcmin": -0.08,
            "ll_m_arcmin": -0.14,
            "separation_arcmin": 0.411,
            "full_raster_separation_arcmin": 0.411,
            "memo195_separation_arcmin": 0.526,
            "frequency_hz": 4.564e9,
        },
    }
    rng = np.random.default_rng(32)
    n = 64
    scatter = {
        "rr_obs_real": rng.normal(size=n).astype(np.float32),
        "rr_obs_imag": rng.normal(size=n).astype(np.float32),
        "rr_pred_real": rng.normal(size=n).astype(np.float32),
        "rr_pred_imag": rng.normal(size=n).astype(np.float32),
        "rr_obs_abs": rng.random(n).astype(np.float32) + 0.2,
        "rr_pred_abs": rng.random(n).astype(np.float32) + 0.2,
        "ll_obs_real": rng.normal(size=n).astype(np.float32),
        "ll_obs_imag": rng.normal(size=n).astype(np.float32),
        "ll_pred_real": rng.normal(size=n).astype(np.float32),
        "ll_pred_imag": rng.normal(size=n).astype(np.float32),
        "ll_obs_abs": rng.random(n).astype(np.float32) + 0.2,
        "ll_pred_abs": rng.random(n).astype(np.float32) + 0.2,
        "rl_obs_real": rng.normal(scale=1.5, size=n).astype(np.float32),
        "rl_pred_real": rng.normal(scale=0.2, size=n).astype(np.float32),
        "rl_diag_real": np.zeros(n, dtype=np.float32),
        "main_lobe_mask": np.ones(n, dtype=bool),
        "mid_mask": np.zeros(n, dtype=bool),
        "outer_mask": np.zeros(n, dtype=bool),
        "rr_onaxis": np.ones(n, dtype=bool),
        "ll_onaxis": np.ones(n, dtype=bool),
        "source_i_jy": np.asarray(8.028518676757812, dtype=np.float64),
    }
    cells = {
        "rr_pred_abs": np.array([0.9, 0.7], dtype=np.float32),
        "rr_obs_abs": np.array([0.92, 0.68], dtype=np.float32),
        "rr_err_abs": np.array([0.04, 0.05], dtype=np.float32),
        "rr_pred_real": np.array([0.88, 0.66], dtype=np.float32),
        "rr_obs_real": np.array([0.90, 0.64], dtype=np.float32),
        "rr_err_real": np.array([0.03, 0.04], dtype=np.float32),
        "ll_pred_abs": np.array([0.89, 0.71], dtype=np.float32),
        "ll_obs_abs": np.array([0.91, 0.69], dtype=np.float32),
        "ll_err_abs": np.array([0.04, 0.05], dtype=np.float32),
        "ll_pred_real": np.array([0.87, 0.65], dtype=np.float32),
        "ll_obs_real": np.array([0.89, 0.63], dtype=np.float32),
        "ll_err_real": np.array([0.03, 0.04], dtype=np.float32),
        "rr_n": np.array([8.0, 6.0], dtype=np.float32),
        "ll_n": np.array([8.0, 6.0], dtype=np.float32),
    }
    residual_strata = {
        "source_i_jy": 8.028518676757812,
        "mask": "V/I_model using CASA MODEL_DATA 3C147 at channel 32",
        "raster_family": (
            "Nearest Memo 195 dense versus sparse lattice. "
            "This is a pass-1/pass-2 occupancy proxy, not a scan-id join."
        ),
        "mover": [{"id": 0, "name": "ea01", "n": 12, "median_abs": 0.31}],
        "reference": [{"id": 1, "name": "ea02", "n": 12, "median_abs": 0.22}],
        "pass": [{"id": 1, "name": "dense / pass-1 occupancy", "n": 8, "median_abs": 0.28}],
        "mover_ll": [{"id": 0, "name": "ea01", "n": 12, "median_abs": 0.33}],
        "reference_ll": [{"id": 1, "name": "ea02", "n": 12, "median_abs": 0.21}],
        "pass_ll": [{"id": 1, "name": "dense / pass-1 occupancy", "n": 8, "median_abs": 0.29}],
    }
    grid = np.linspace(-20.0, 20.0, 8)
    maps = {
        "l_arcmin": grid,
        "m_arcmin": grid,
        "weight": np.ones((8, 8)),
        "rr_measured": np.ones((8, 8), dtype=np.complex128),
        "rr_cassbeam": np.ones((8, 8), dtype=np.complex128),
        "rr_residual": 0.05 * np.ones((8, 8), dtype=np.complex128),
        "ll_measured": np.ones((8, 8), dtype=np.complex128),
        "ll_cassbeam": np.ones((8, 8), dtype=np.complex128),
        "ll_residual": 0.06 * np.ones((8, 8), dtype=np.complex128),
    }
    occupancy = {"l_arcmin": grid, "m_arcmin": grid, "count": np.ones((8, 8))}
    coordinate_feed_scatter = {
        "source_i_jy": np.asarray(8.028518676757812),
        "measured_rr": np.ones(n, dtype=np.complex64),
        "measured_ll": np.ones(n, dtype=np.complex64),
        "generic_commanded_rr": np.full(n, 0.90, dtype=np.complex64),
        "generic_commanded_ll": np.full(n, 0.90, dtype=np.complex64),
        "generic_source_lm_rr": np.full(n, 0.92, dtype=np.complex64),
        "generic_source_lm_ll": np.full(n, 0.92, dtype=np.complex64),
        "evla_c_source_lm_rr": np.full(n, 0.98, dtype=np.complex64),
        "evla_c_source_lm_ll": np.full(n, 0.98, dtype=np.complex64),
        "main_lobe": np.ones(n, dtype=bool),
        "mid": np.zeros(n, dtype=bool),
        "outer_diagnostic": np.zeros(n, dtype=bool),
        "train": np.ones(n, dtype=bool),
        "spatial_holdout": np.zeros(n, dtype=bool),
        "mover_holdout": np.zeros(n, dtype=bool),
    }
    coordinate_feed_maps = {
        "l_arcmin": grid,
        "m_arcmin": grid,
        "rr_weight": np.ones((8, 8)),
        "ll_weight": np.ones((8, 8)),
        "rr_measured": np.ones((8, 8), dtype=np.complex64),
        "ll_measured": np.ones((8, 8), dtype=np.complex64),
    }
    for model, value in (
        ("generic_commanded", 0.90),
        ("generic_source_lm", 0.92),
        ("evla_c_source_lm", 0.98),
    ):
        coordinate_feed_maps[f"{model}_rr"] = np.full(
            (8, 8), value, dtype=np.complex64
        )
        coordinate_feed_maps[f"{model}_ll"] = np.full(
            (8, 8), value, dtype=np.complex64
        )
    return {
        "observation": {
            "project": "THOL0001",
            "antenna_names": ["ea01", "ea02", "ea03"],
            "reference_antenna_names": ["ea02"],
            "timeline": [{"label": "pass 1 raster", "scan_start": 18, "n_scans": 16}],
            "version1_frequency_hz": 4.564e9,
            "spw5_status": "sealed",
        },
        "calibration": {"reference_antenna": "ea02", "fullpol_parang": True, "calwt": False},
        "convention_gates": {
            "cumulative_residuals": {"K+B+G": {"median_rel_l2": 0.003}},
            "not_physical_beam_evidence": True,
        },
        "holoraster_channel32": channel32,
        "holoraster_frequency": frequency,
        "squint": squint,
        "offset_ring": {
            "closure": {
                "rr": {"relative_power": 0.0117, "n": 50, "ok": True},
                "ll": {"relative_power": 0.0090, "n": 50, "ok": True},
                "passed": True,
            },
            "radius_arcmin": 3.62,
            "q_u_is_frequency_holdout": False,
            "model_selected": False,
            "full_jones_frozen": False,
            "production_factory_modified": False,
        },
        "residual_geometry": residual,
        "residual_strata": residual_strata,
        "radial_coherence": {
            "source_i_jy": 8.028518676757812,
            "amp_floor_jy": 0.05,
            "phase_status": "exploratory",
            "raster_extent_arcmin": {"l_abs_max": 51.0, "m_abs_max": 51.0, "corner_max": 71.0},
            "rr": [
                {
                    "r_lo_arcmin": 0.0,
                    "r_hi_arcmin": 10.0,
                    "r_mid_arcmin": 5.0,
                    "n": 20,
                    "correlation_abs": 0.99,
                    "slope_real": 1.02,
                    "slope_imag": 0.0,
                    "residual_power": 0.01,
                    "circular_phase_deg": -1.0,
                    "circular_phase_std_deg": 2.0,
                    "median_phase_deg": -1.0,
                    "n_phase": 18,
                },
                {
                    "r_lo_arcmin": 40.0,
                    "r_hi_arcmin": 50.0,
                    "r_mid_arcmin": 45.0,
                    "n": 20,
                    "correlation_abs": 0.82,
                    "slope_real": 0.94,
                    "slope_imag": 0.04,
                    "residual_power": 0.32,
                    "circular_phase_deg": -10.0,
                    "circular_phase_std_deg": 8.0,
                    "median_phase_deg": -10.0,
                    "n_phase": 12,
                },
            ],
            "ll": [
                {
                    "r_lo_arcmin": 0.0,
                    "r_hi_arcmin": 10.0,
                    "r_mid_arcmin": 5.0,
                    "n": 20,
                    "correlation_abs": 0.98,
                    "slope_real": 1.01,
                    "slope_imag": 0.0,
                    "residual_power": 0.012,
                    "circular_phase_deg": -1.5,
                    "circular_phase_std_deg": 2.0,
                    "median_phase_deg": -1.5,
                    "n_phase": 18,
                },
                {
                    "r_lo_arcmin": 40.0,
                    "r_hi_arcmin": 50.0,
                    "r_mid_arcmin": 45.0,
                    "n": 20,
                    "correlation_abs": 0.81,
                    "slope_real": 0.91,
                    "slope_imag": 0.04,
                    "residual_power": 0.35,
                    "circular_phase_deg": -12.0,
                    "circular_phase_std_deg": 9.0,
                    "median_phase_deg": -12.0,
                    "n_phase": 12,
                },
            ],
            "map_phase_beyond_40_arcmin": {
                "rr": {"circular_phase_deg": -10.0, "n": 8},
                "ll": {"circular_phase_deg": -12.0, "n": 8},
            },
        },
        "antenna_coherence": {
            "phase_status": "exploratory",
            "movers": [
                {
                    "id": 0,
                    "name": "ea04",
                    "rr": [
                        {
                            "r_mid_arcmin": 5.0,
                            "correlation_abs": 0.99,
                            "circular_phase_deg": -1.0,
                            "slope_real": 1.0,
                            "slope_imag": 0.0,
                        },
                        {
                            "r_mid_arcmin": 45.0,
                            "correlation_abs": 0.80,
                            "circular_phase_deg": -11.0,
                            "slope_real": 0.9,
                            "slope_imag": 0.04,
                        },
                    ],
                    "ll": [],
                }
            ],
        },
        "bright_source_examples": {
            "note": "Array-average binned visibilities at example radii.",
            "source_i_jy": 8.028518676757812,
            "examples": [
                {
                    "label": "boresight",
                    "l_arcmin": 0.0,
                    "m_arcmin": 0.0,
                    "radius_arcmin": 0.0,
                    "rr_meas_re": 8.1,
                    "rr_meas_im": 0.1,
                    "rr_pred_re": 8.0,
                    "rr_pred_im": 0.0,
                    "ll_meas_re": 8.0,
                    "ll_meas_im": 0.0,
                    "ll_pred_re": 8.0,
                    "ll_pred_im": 0.0,
                    "rr_stokes_i_factor_meas": 1.02,
                    "rr_stokes_i_factor_pred": 1.0,
                },
                {
                    "label": "40′",
                    "l_arcmin": 40.0,
                    "m_arcmin": 0.0,
                    "radius_arcmin": 40.0,
                    "rr_meas_re": 0.4,
                    "rr_meas_im": -0.05,
                    "rr_pred_re": 0.25,
                    "rr_pred_im": 0.0,
                    "ll_meas_re": 0.38,
                    "ll_meas_im": -0.04,
                    "ll_pred_re": 0.24,
                    "ll_pred_im": 0.0,
                    "rr_stokes_i_factor_meas": 0.0025,
                    "rr_stokes_i_factor_pred": 0.001,
                },
            ],
        },
        "frequency_series": frequency,
        "offset_ring_fields": {
            "fields": [
                {
                    "field_id": 3,
                    "name": "C147-W",
                    "l_arcmin": 3.6,
                    "m_arcmin": 0.0,
                    "radius_arcmin": 3.6,
                },
                {
                    "field_id": 7,
                    "name": "C147-E",
                    "l_arcmin": -3.6,
                    "m_arcmin": 0.0,
                    "radius_arcmin": 3.6,
                },
            ],
            "partition": {"training": [2], "inner_holdout": [3, 7], "sealed_holdout": [1, 5]},
        },
        "crosshand_quadrants": quadrants,
        "scatter": scatter,
        "maps": maps,
        "cells": cells,
        "occupancy": occupancy,
        "coordinate_feed_comparison": {
            "scope": "SPW-4 frozen development rows",
            "development_only": True,
            "spw5_closed": True,
            "production_beam_frozen": False,
            "models": {
                "generic_commanded": "historical",
                "generic_source_lm": "corrected coordinate",
                "evla_c_source_lm": "corrected coordinate and feed",
            },
            "metrics": {
                model: {
                    "development": {
                        region: {
                            hand: {"residual_power": loss}
                            for hand in ("rr", "ll")
                        }
                        for region, loss in (
                            ("main_lobe", value),
                            ("mid", value * 2.0),
                            ("outer_diagnostic", value * 3.0),
                            ("all", value * 2.5),
                        )
                    }
                }
                for model, value in (
                    ("generic_commanded", 0.010),
                    ("generic_source_lm", 0.009),
                    ("evla_c_source_lm", 0.007),
                )
            },
            "paired_scores": {
                key: {
                    axis: {
                        "delta": delta,
                        "delta_lo": delta - 0.001,
                        "delta_hi": delta + 0.001,
                        "improves": delta + 0.001 < 0.0,
                    }
                    for axis in ("spatial", "moving")
                }
                for key, delta in (
                    ("generic_source_lm_vs_generic_commanded", -0.001),
                    ("evla_c_source_lm_vs_generic_commanded", -0.003),
                    ("evla_c_source_lm_vs_generic_source_lm", -0.002),
                )
            },
            "interpretation": {"evla_c_beats_generic_source": True},
            "map_squint": {
                "source_lm_labels": {
                    "independent_masks": {
                        "rr_l_arcmin": 0.1,
                        "rr_m_arcmin": 0.3,
                        "ll_l_arcmin": -0.1,
                        "ll_m_arcmin": -0.2,
                        "memo195_separation_arcmin": 0.526,
                    }
                }
            },
            "generic_plane_centroids": {
                "rr_l_arcmin": 0.2,
                "rr_m_arcmin": 0.1,
                "ll_l_arcmin": -0.2,
                "ll_m_arcmin": -0.1,
            },
            "evla_plane_centroids": {
                "rr_l_arcmin": 0.06,
                "rr_m_arcmin": 0.25,
                "ll_l_arcmin": -0.06,
                "ll_m_arcmin": -0.25,
            },
        },
        "coordinate_feed_scatter": coordinate_feed_scatter,
        "coordinate_feed_maps": coordinate_feed_maps,
    }


def write_manufactured_bundle(root: Path) -> Path:
    products = manufactured_products()
    return write_validation_bundle(root, **products)


def test_full_raster_squint_is_refused_for_publication() -> None:
    refuse_full_raster_squint(
        {
            "estimator": "mainlobe_20pct_peak",
            "publication_estimator": True,
            "full_raster_is_publication_estimator": False,
        }
    )
    with pytest.raises(ValueError, match="20%-of-peak"):
        refuse_full_raster_squint({"estimator": "full_raster_power_centroid"})


def test_bundle_round_trip_and_claims(tmp_path: Path) -> None:
    root = write_manufactured_bundle(tmp_path / "bundle")
    bundle = load_bundle(root)
    assert bundle.manifest["publication_version"] == PUBLICATION_VERSION
    assert bundle.coordinate_feed_comparison["interpretation"][
        "evla_c_beats_generic_source"
    ]
    residuals = residual_power_table(bundle.holoraster_channel32)
    assert residuals["main_lobe"]["rr"] < 0.01
    assert residuals["main_lobe"]["rr_rms"] == pytest.approx(0.0064**0.5)
    assert residuals["main_lobe"]["rr_median_abs_over_i"] == pytest.approx(0.0483)
    assert residuals["outer_diagnostic"]["rr_correlation"] == pytest.approx(0.97)
    claims = build_claims(bundle)
    by_id = {item["id"]: item for item in claims["claims"]}
    assert by_id["C01"]["status"] == "pass"
    assert by_id["C02"]["status"] == "warn"
    assert by_id["C03"]["status"] == "warn"
    assert by_id["C05"]["status"] == "blocked"
    assert by_id["C06"]["status"] == "not_run"
    assert by_id["C07"]["status"] == "pass"
    registry = claim_registry(bundle_root=root)
    assert registry["claims"][0]["status"] == "pass"
    pair = publication_squint_pair(bundle.squint)
    assert pair["measured"]["estimator"] == "mainlobe_20pct_peak"


def test_bundle_rejects_bacchus_load_paths(tmp_path: Path) -> None:
    products = manufactured_products()
    products["observation"] = {
        **products["observation"],
        "ms_path": "/media/stephen/astro/vla/extracted/commissioning/work/THOL0001.ms",
    }
    write_validation_bundle(tmp_path / "bundle", **products)
    loaded = json.loads((tmp_path / "bundle" / "observation_summary.json").read_text())
    assert loaded["ms_path"] == "THOL0001.ms"
    assert contains_banned_load_path(loaded) is None


def test_bundle_refuses_full_raster_publication_squint(tmp_path: Path) -> None:
    products = manufactured_products()
    products["squint"]["measured"]["estimator"] = "full_raster_power_centroid"
    with pytest.raises(ValueError, match="20%-of-peak"):
        write_validation_bundle(tmp_path / "bundle", **products)


def test_shared_plots_write_sidecars(tmp_path: Path) -> None:
    root = write_manufactured_bundle(tmp_path / "bundle")
    bundle = load_bundle(root)
    written = write_all_figures(bundle, tmp_path / "figures")
    pngs = [path for path in written if path.suffix == ".png"]
    assert len(pngs) >= 18
    names = {path.name for path in pngs}
    assert "05_onaxis_amplitude.png" in names
    assert "08_cassbeam_scatter_main_lobe.png" in names
    assert "15_residual_strata.png" in names
    assert "20_amplitude_db.png" in names
    assert "22_masked_phase.png" in names
    assert "23_radial_coherence.png" in names
    assert "24_antenna_coherence.png" in names
    assert "25_bright_source_examples.png" in names
    assert "26_coordinate_feed_impact.png" in names
    assert "27_coordinate_feed_scatter.png" in names
    assert "28_coordinate_feed_residual_maps.png" in names
    assert "29_coordinate_feed_squint.png" in names
    for image in pngs:
        sidecar = Path(str(image) + ".sidecar.json")
        assert sidecar.is_file()
        payload = json.loads(sidecar.read_text())
        assert payload["figure_id"].startswith("F")
        assert payload["table_sha256"]
        assert payload["claims"]
        assert payload["coordinate_query"]
        assert payload["coordinate_convention"]
        assert payload["model"]
        assert "frequency_hz" in payload
        assert "artifact_hashes" in payload
    scatter_sidecar = json.loads(
        (tmp_path / "figures" / "08_cassbeam_scatter_main_lobe.png.sidecar.json").read_text()
    )
    assert scatter_sidecar["table"]["mask"] == "V/I_model >= 0.5"
    cuts_sidecar = json.loads(
        (tmp_path / "figures" / "21_signed_complex_cuts.png.sidecar.json").read_text()
    )
    assert cuts_sidecar["table"]["outer_min_arcmin"] == 15.0
    examples_sidecar = json.loads(
        (tmp_path / "figures" / "25_bright_source_examples.png.sidecar.json").read_text()
    )
    assert "exact radius" in examples_sidecar["caption"]


def test_complex_visibility_score_and_phase_mask() -> None:
    pred = np.array([1.0, 0.8, 0.02], dtype=np.complex128)
    obs = pred * np.exp(1j * np.deg2rad(-10.0))
    obs[2] = 0.01
    score = complex_visibility_score(obs[:2], pred[:2], np.ones(2))
    assert score["correlation_abs"] == pytest.approx(1.0)
    assert score["circular_phase_deg"] == pytest.approx(-10.0, abs=0.2)
    mask = phase_valid_mask(obs, pred, np.ones(3), amp_floor_jy=0.05)
    assert mask.tolist() == [True, True, False]


def test_scientific_voltage_masks_use_model_intensity() -> None:
    measured = np.zeros((3, 1, 2, 2), dtype=np.complex128)
    measured[0, 0, 0, 0] = measured[0, 0, 1, 1] = 8.0
    measured[1, 0, 0, 0] = measured[1, 0, 1, 1] = 3.0
    measured[2, 0, 0, 0] = measured[2, 0, 1, 1] = 0.5
    weight = np.ones((3, 1, 2, 2), dtype=np.float64)
    masks = scientific_voltage_masks(measured, weight, intensity_jy=8.0)
    assert masks["main_lobe"].tolist() == [True, False, False]
    assert masks["mid"].tolist() == [False, True, False]
    assert masks["outer_diagnostic"].tolist() == [False, False, True]


def test_ll_map_weights_are_not_the_rr_weights() -> None:
    weight = np.zeros((4, 2, 2), dtype=np.float64)
    weight[:, 0, 0] = 1.0
    weight[:, 1, 1] = np.array([4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(visibility_hand_weight(weight, 0, 0), 1.0)
    np.testing.assert_allclose(visibility_hand_weight(weight, 1, 1), [4.0, 3.0, 2.0, 1.0])
    cube = weight[:, None, :, :]
    np.testing.assert_allclose(visibility_hand_weight(cube, 1, 1), [4.0, 3.0, 2.0, 1.0])


def test_main_lobe_gate_rejects_a_50_percent_residual() -> None:
    support = classify_diagonal_region_support(
        {
            "main_lobe": {"rr": {"residual_power": 0.50}, "ll": {"residual_power": 0.50}},
            "mid": {"rr": {"residual_power": 0.08}, "ll": {"residual_power": 0.09}},
            "outer_diagnostic": {"rr": {"residual_power": 0.32}, "ll": {"residual_power": 0.35}},
        }
    )
    assert support["main_lobe"] == "rejected"
    assert support["mid_beam"] == "qualified"


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    root = write_manufactured_bundle(tmp_path / "bundle")
    claims = json.loads((root / "claims.json").read_text())
    hashes = claims["claims"][0]["input_hashes"]
    assert hashes["holoraster_channel32"]
    assert hashes["offset_ring"]
    claims["tampered"] = True
    (root / "claims.json").write_text(json.dumps(claims) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(root)


def test_offset_ring_plot_accepts_c147_geometry(tmp_path: Path) -> None:
    fields = {
        "fields": [
            {
                "field_id": 3,
                "name": "C147-W",
                "l_rad": 0.001047,
                "m_rad": 0.0,
                "radius_arcmin": 3.6,
            }
        ],
        "partition": {
            "training": [2],
            "inner_holdout": [3],
            "sealed_holdout": [1],
            "inner_score": 0.009,
            "notes": ["historical"],
        },
    }
    written = plot_offset_ring(
        fields,
        {"rr": 0.009, "ll": 0.006},
        tmp_path,
    )
    names = {path.name for path in written}
    assert "17_c147_offset_ring.png" in names
