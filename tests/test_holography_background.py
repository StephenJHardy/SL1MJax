"""Independent analytic expectations for the fixed-background visibility test."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.catalog import RadioCatalogSource, write_radio_catalog
from sl1mjax.holography_background import (
    C_M_S,
    BackgroundGeometry,
    compare_background,
    predict_background,
    source_fluxes,
)


def direction(l, m=0.0):
    return np.array([l, m, np.sqrt(1 - l * l - m * m)])


def geometry(n=8, sources=1):
    sky = np.broadcast_to(direction(0.02), (n, sources, 3)).copy()
    p = np.broadcast_to(direction(0.01), sky.shape).copy()
    q = np.broadcast_to(direction(0.03), sky.shape).copy()
    uvw = np.zeros((n, 3))
    uvw[:, 2] = np.linspace(0, 1000, n)
    return BackgroundGeometry(uvw, 4.5e9, sky, p, q)


def lookup(freq, l, m):
    # Different complex responses in each hand, varying with pointing.
    value = np.stack([1 + 5j * l, 0.7 - 3j * l], axis=-1)
    return value, np.ones(value.shape, dtype=bool)


def test_prediction_both_beams_full_w_and_baseline_reversal():
    g = geometry()
    got, support, _ = predict_background(g, np.array([2.0]), lookup)
    ep = np.array([1 + 0.05j, 0.7 - 0.03j])
    eq = np.array([1 + 0.15j, 0.7 - 0.09j])
    phase = np.exp(2j * np.pi * g.frequency_hz / C_M_S * g.uvw_m[:, 2] * (np.sqrt(1 - 0.02**2) - 1))
    expected = 2 * ep * eq.conj() * phase[:, None]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert support.all()
    assert not np.allclose(got, 2 * ep * phase[:, None])  # missing reference attenuation
    assert not np.allclose(got, 2 * ep * eq * phase[:, None])  # missing conjugation
    assert not np.allclose(got, 2 * ep * eq.conj())  # missing w
    reverse = BackgroundGeometry(-g.uvw_m, g.frequency_hz, g.source_lmn, g.beam_q_lmn, g.beam_p_lmn)
    back, _, _ = predict_background(reverse, np.array([2.0]), lookup)
    np.testing.assert_allclose(back, got.conj(), atol=1e-12)


def test_sum_and_chunk_invariance_and_reference_null():
    g = geometry(sources=2)
    a, _, _ = predict_background(g, np.array([1.0, 2.0]), lookup, row_chunk=1)
    b, _, _ = predict_background(g, np.array([1.0, 2.0]), lookup, row_chunk=7)
    np.testing.assert_array_equal(a, b)
    single, _, _ = predict_background(geometry(), np.array([3.0]), lookup)
    np.testing.assert_allclose(a, single)

    def null_reference(freq, l, m):
        values = np.where(l[:, None] > 0.02, 0.0, 1.0) * np.ones((len(l), 2))
        return values, np.ones(values.shape, dtype=bool)

    vis, valid, _ = predict_background(g, np.array([1.0, 2.0]), null_reference)
    np.testing.assert_array_equal(vis, 0)
    assert valid.all()  # physical null is known zero, not missing support


def test_fixed_residual_jones_matches_matrix_rime():
    g = geometry()
    base, _, _ = predict_background(g, np.array([1.0]), lookup)
    p = np.broadcast_to(np.array([[1.01, 0.02j], [0.03, 0.97]]), (8, 2, 2))
    q = np.broadcast_to(np.array([[0.99j, 0.01], [0.04j, 1.02]]), (8, 2, 2))
    got, valid, _ = predict_background(g, np.array([1.0]), lookup, residual_p=p, residual_q=q)
    expected = np.array([np.diag(p[i] @ np.diag(base[i]) @ q[i].conj().T) for i in range(8)])
    np.testing.assert_allclose(got, expected)
    assert valid.all()


def test_one_unsupported_hand_and_rear_hemisphere_are_not_zero():
    def unsupported(freq, l, m):
        v, ok = lookup(freq, l, m)
        ok[:, 0] = False
        return v, ok

    vis, valid, _ = predict_background(geometry(), np.array([1.0]), unsupported)
    assert np.isnan(vis[:, 0]).all() and np.isfinite(vis[:, 1]).all()
    assert not valid[:, 0].any() and valid[:, 1].all()
    g = geometry()
    g.beam_q_lmn[..., 2] *= -1
    vis, valid, _ = predict_background(g, np.array([1.0]), lookup)
    assert not valid.any() and np.isnan(vis).all()


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_bad_frequency_refused(bad):
    g = geometry()
    with pytest.raises(ValueError):
        predict_background(
            BackgroundGeometry(g.uvw_m, bad, g.source_lmn, g.beam_p_lmn, g.beam_q_lmn),
            np.array([1.0]),
            lookup,
        )


def test_nonunit_direction_refused():
    g = geometry()
    g.source_lmn[0, 0] = [2, 0, 0]
    with pytest.raises(ValueError, match="unit"):
        predict_background(g, np.array([1.0]), lookup)


def test_fixed_score_flags_support_and_cluster_counts():
    n = 12
    a = np.ones((n, 2), dtype=complex)
    b = np.full((n, 2), 0.2 + 0.1j)
    measured = a + b
    w = np.ones((n, 2))
    flag = np.zeros((n, 2), bool)
    support = ~flag
    flag[0, 0] = True
    measured[0, 0] = 1e9
    support[1, 0] = False
    mask = np.ones(n, bool)
    result = compare_background(
        measured, a, b, w, flag, support, mask, np.arange(n) // 3, n_boot=40
    )
    rr = result["hands"]["RR"]
    assert rr["scored_rows"] == 10 and rr["unsupported_rows"] == 1
    assert rr["n_clusters"] == 4
    assert rr["plus_catalogue_power"] == pytest.approx(0, abs=1e-28)
    assert rr["paired_ci95"][1] < 0
    worse = compare_background(a, a, b, w, flag, support, mask, np.arange(n) // 3, n_boot=40)
    assert worse["hands"]["LL"]["paired_ci95"][0] > 0


def test_null_support_zero_denominator_and_no_fit():
    zero = np.zeros((4, 2), complex)
    flag = np.zeros((4, 2), bool)
    result = compare_background(
        zero, zero, zero, np.ones((4, 2)), flag, ~flag, np.ones(4, bool), np.zeros(4), n_boot=5
    )
    assert result["hands"]["RR"]["status"] == "zero_observed_power"
    assert result["model_fitted"] is False
    result = compare_background(
        zero, zero, zero, np.ones((4, 2)), flag, flag, np.ones(4, bool), np.zeros(4), n_boot=5
    )
    assert result["hands"]["RR"]["status"] == "no_supported_samples"


def source():
    return RadioCatalogSource(
        "test", 1.0, 0.0, 1e9, 2.0, "synthetic", "synthetic://test", spectral_index=-1.0
    )


def test_spectrum_requires_explicit_unknown_prior():
    from dataclasses import replace

    np.testing.assert_allclose(source_fluxes([source()], 2e9), [1])
    s = replace(source(), spectral_index=None)
    with pytest.raises(ValueError, match="spectral index"):
        source_fluxes([s], 2e9)
    np.testing.assert_allclose(source_fluxes([s], 2e9, default_spectral_index=0), [2])


def test_snapshot_spherical_repoint_centre_and_norm():
    spec = importlib.util.spec_from_file_location(
        "background_prepare", Path(__file__).parents[1] / "scripts/prepare_holography_background.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    offset = np.array([0.013, -0.019])
    sources = np.array([direction(0), direction(0.02, -0.008)])
    got = module.repoint_sin(sources, offset)
    np.testing.assert_allclose(got[0, :2], -offset, atol=1e-14)
    np.testing.assert_allclose(np.sum(got**2, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(module.repoint_sin(sources, [0, 0]), sources)
    # Distances are preserved by the frame rotation, not tangent subtraction.
    np.testing.assert_allclose(got @ got.T, sources @ sources.T, atol=1e-14)
    assert np.max(np.abs(got[1, :2] - (sources[1, :2] - offset))) > 1e-7


def test_cli_dry_run_checksum_geometry_and_no_outputs(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "background_cli", Path(__file__).parents[1] / "scripts/diagnose_holography_background.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = tmp_path / "catalog.csv"
    write_radio_catalog(catalog, (source(),))
    g = geometry()
    snapshot = tmp_path / "snapshot.npz"
    np.savez(
        snapshot,
        uvw_m=g.uvw_m,
        frequency_hz=g.frequency_hz,
        beam_p_lmn=g.beam_p_lmn,
        beam_q_lmn=g.beam_q_lmn,
        residual_p=np.broadcast_to(np.eye(2), (8, 2, 2)),
        residual_q=np.broadcast_to(np.eye(2), (8, 2, 2)),
        measured=np.ones((8, 2), complex),
        calibrator=np.ones((8, 2), complex),
        weight=np.ones((8, 2)),
        flag=np.zeros((8, 2), bool),
        score_mask=np.ones(8, bool),
        cluster_id=np.arange(8) // 2,
        row_id=np.arange(8),
    )
    manifest = dict(
        schema_version=1,
        snapshot_sha256=module.sha256(snapshot),
        catalogue_sha256=module.sha256(catalog),
        phase_centre_rad=[0, 0],
        source_names=["test"],
        geometry_provenance="synthetic",
        calibration_provenance="synthetic",
        calibration_state="casa_parang_true",
        residual_jones_provenance="synthetic identity",
        uvw_convention="casa_positive_fringe_original_pq",
        beam_coordinate_frame="feed_source_direction_cosines",
        split_provenance="synthetic fixed",
        cluster_unit="dwell",
        calibrator_exclusion_arcsec=60,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "test",
            "--snapshot",
            str(snapshot),
            "--manifest",
            str(path),
            "--catalogue",
            str(catalog),
            "--output",
            str(output),
            "--dry-run",
        ],
    )
    module.main()
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert not output.exists()

    # Exercise the full writer with an independent analytic beam, no real MS,
    # external plane files, network access, or telescope runtime.
    from types import SimpleNamespace

    def plane_lookup(l, m, *, off_diagonal):
        assert off_diagonal is False
        values, valid = lookup(g.frequency_hz, l, m)
        matrix = np.zeros((len(l), 2, 2), complex)
        matrix[:, 0, 0], matrix[:, 1, 1] = values.T
        return matrix, valid.all(axis=1)

    fake = SimpleNamespace(
        catalog=SimpleNamespace(plane=lambda frequency: SimpleNamespace(lookup=plane_lookup))
    )
    monkeypatch.setattr(
        "sl1mjax.evla_c_survey_beam.voltage_beam_for_survey_catalog",
        lambda **kwargs: fake,
    )
    import sys

    monkeypatch.setattr(
        "sys.argv", sys.argv[:-1] + ["--beam-root", str(tmp_path), "--beam-digest", "synthetic"]
    )
    module.main()
    report = json.loads((output / "report.json").read_text())
    assert report["comparison"]["hands"]["RR"]["scored_rows"] == 8
    assert (output / "background_comparison.png").stat().st_size > 1000
    with np.load(output / "predictions.npz") as results:
        # Nonzero w evaluated for the CSV's 1-degree direction, not the
        # manufactured geometry() helper's different sky location.
        ep = np.array([1 + 0.05j, 0.7 - 0.03j])
        eq = np.array([1 + 0.15j, 0.7 - 0.09j])
        phase = np.exp(
            2j * np.pi * g.frequency_hz / C_M_S * g.uvw_m[:, 2] * (np.cos(np.deg2rad(1.0)) - 1)
        )
        expected = (2 / 4.5) * ep * eq.conj() * phase[:, None]
        np.testing.assert_allclose(results["background"], expected, atol=1e-11)
    with pytest.raises(ValueError, match="new directory"):
        module.main()
    manifest["snapshot_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        module.main()
