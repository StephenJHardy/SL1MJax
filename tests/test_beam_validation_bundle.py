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
from sl1mjax.beam_validation_plots import write_all_figures
from sl1mjax.beam_validation_statistics import (
    build_claims,
    publication_squint_pair,
    residual_power_table,
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
        "occupancy": occupancy,
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
    residuals = residual_power_table(bundle.holoraster_channel32)
    assert residuals["main_lobe"]["rr"] < 0.01
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
    assert len(pngs) >= 12
    for image in pngs:
        sidecar = Path(str(image) + ".sidecar.json")
        assert sidecar.is_file()
        payload = json.loads(sidecar.read_text())
        assert payload["figure_id"].startswith("F")
        assert payload["table_sha256"]
        assert payload["claims"]


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    root = write_manufactured_bundle(tmp_path / "bundle")
    claims = json.loads((root / "claims.json").read_text())
    claims["tampered"] = True
    (root / "claims.json").write_text(json.dumps(claims) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(root)
