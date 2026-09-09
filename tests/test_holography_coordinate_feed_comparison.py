from __future__ import annotations

import numpy as np

from sl1mjax.holography_coordinate_feed_comparison import (
    interpret_coordinate_feed_scores,
    query_coordinates,
)
from sl1mjax.holography_holoraster_coordinates import ARCMIN_TO_RAD


def test_source_query_is_negative_commanded() -> None:
    commanded = np.array([[3.0 * ARCMIN_TO_RAD, -1.0 * ARCMIN_TO_RAD]], dtype=np.float64)
    source = query_coordinates(commanded, query="source_lm_feed")
    np.testing.assert_allclose(source, -commanded)
    np.testing.assert_allclose(query_coordinates(commanded, query="commanded"), commanded)


def test_interpretation_requires_paired_holdout_improvement() -> None:
    paired = {
        "generic_source_lm_vs_generic_commanded": {
            "spatial": {"improves": True},
            "moving": {"improves": True},
        }
    }
    report = interpret_coordinate_feed_scores(paired, evla_present=False)
    assert report["source_lm_feed_beats_commanded"] is True
    assert report["evla_c_compared"] is False
    assert report["spw5_ready"] is False
    assert report["convention_ladder_reopened"] is False
    paired["generic_source_lm_vs_generic_commanded"]["moving"]["improves"] = False
    assert (
        interpret_coordinate_feed_scores(paired, evla_present=False)[
            "source_lm_feed_beats_commanded"
        ]
        is False
    )


def test_development_manifest_pairs_jones_dat_with_params(tmp_path) -> None:
    from sl1mjax.cassbeam_highres import write_development_highres_manifest

    root = tmp_path / "cassbeam_evla_c_g1024_p32_dev"
    spw4 = root / "spw4"
    reference = root / "reference"
    spw4.mkdir(parents=True)
    reference.mkdir()
    (spw4 / "evla-cband-4564-g1024-p32.jones.dat").write_text("0\n")
    (spw4 / "evla-cband-4564-g1024-p32.params").write_text("freq=4.564\n")
    (reference / "base.in").write_text("name = EVLA\n")
    (reference / "vla_geom").write_text("12.5\n")
    manifest = write_development_highres_manifest(
        root, model_id="cassbeam_evla_c_g1024_p32_spw4_dev", name_prefix="evla-cband"
    )
    assert manifest["planes"][0]["frequency_mhz"] == 4564
    assert manifest["planes"][0]["params"].endswith(".params")
    assert not manifest["planes"][0]["params"].endswith(".jones.params")


def test_runner_does_not_reopen_the_convention_ladder() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_thol0001_coordinate_feed_comparison.py"
    ).read_text()
    assert "convention_ladder(" not in source
    assert "holoraster_coordinate_feed_comparison_v1" in source
    assert "cassbeam_cband_full_jones_g1024_p32_20260906" in source
    assert "refuse_convention_ladder()" not in source
