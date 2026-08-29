from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.casa_awp2_oracle import (
    CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE,
    POINT_SOURCE_ROLES,
    POWER_ORACLE_VOLTAGE_PATTERN,
    STOKES_MODELS,
    CasaAwp2PowerPlane,
    compare_frozen_power_oracle,
    compare_power_beams,
    compare_power_oracle_directory,
    geometric_visibility_phase,
    load_power_plane,
    polarization_oracle_is_frozen,
    power_oracle_is_frozen,
    predict_point_source_visibilities,
    sl1mjax_power_on_plane,
    stokes_model_values,
    write_sine_projected_power_fits,
)
from sl1mjax.cassbeam_beam import CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE as BEAM_TOLERANCE


def _synthetic_plane(tmp_path: Path) -> CasaAwp2PowerPlane:
    frequency = 4.564e9
    phase_centre = (np.deg2rad(180.0), np.deg2rad(45.0))
    cell = np.deg2rad(30.0 / 3600.0)
    size = 64
    dummy = np.ones((size, size), dtype=np.float64)
    path = tmp_path / "template.fits"
    write_sine_projected_power_fits(
        path,
        dummy,
        phase_centre_rad=phase_centre,
        cell_rad=cell,
        frequency_hz=frequency,
    )
    template = load_power_plane(
        path,
        frequency_hz=frequency,
        parallactic_angle_rad=0.0,
        stokes="I",
    )
    model = sl1mjax_power_on_plane(template)
    write_sine_projected_power_fits(
        path,
        model,
        phase_centre_rad=phase_centre,
        cell_rad=cell,
        frequency_hz=frequency,
    )
    return load_power_plane(
        path,
        frequency_hz=frequency,
        parallactic_angle_rad=0.0,
        stokes="I",
    )


def test_synthetic_power_comparison_is_identity(tmp_path: Path) -> None:
    plane = _synthetic_plane(tmp_path)
    model = sl1mjax_power_on_plane(plane)
    comparison = compare_power_beams(plane, model)
    assert comparison.accepted
    assert comparison.max_abs_residual < 1.0e-12
    assert abs(comparison.centre_offset_pixels) < 1.0e-6
    assert comparison.fwhm_casa_arcmin == pytest.approx(
        comparison.fwhm_sl1mjax_arcmin, rel=1.0e-6
    )


def test_synthetic_power_comparison_detects_a_centre_shift(tmp_path: Path) -> None:
    plane = _synthetic_plane(tmp_path)
    shifted = np.roll(plane.power, 2, axis=1)
    comparison = compare_power_beams(
        CasaAwp2PowerPlane(
            power=shifted,
            l_rad=plane.l_rad,
            m_rad=plane.m_rad,
            frequency_hz=plane.frequency_hz,
            parallactic_angle_rad=plane.parallactic_angle_rad,
            stokes=plane.stokes,
            path=plane.path,
        ),
        sl1mjax_power_on_plane(plane),
    )
    assert comparison.centre_offset_pixels > 1.0
    assert comparison.accepted is False


def test_point_source_visibility_contract() -> None:
    jones = np.eye(2, dtype=np.complex128)
    uvw = np.array([[100.0, -30.0, 5.0]], dtype=np.float64)
    frequency = 4.564e9
    phase = geometric_visibility_phase(uvw, frequency, 0.01, -0.005)
    vis = predict_point_source_visibilities(
        jones,
        jones,
        stokes_i=1.0,
        stokes_q=0.0,
        stokes_u=0.0,
        stokes_v=0.0,
        uvw_m=uvw,
        frequency_hz=frequency,
        l_rad=0.01,
        m_rad=-0.005,
    )
    np.testing.assert_allclose(vis[0], phase[0] * np.eye(2), atol=1e-12)
    stokes_i, stokes_q, stokes_u, stokes_v = stokes_model_values("I+Q")
    plus_q = predict_point_source_visibilities(
        jones,
        jones,
        stokes_i=stokes_i,
        stokes_q=stokes_q,
        stokes_u=stokes_u,
        stokes_v=stokes_v,
        uvw_m=uvw,
        frequency_hz=frequency,
        l_rad=0.0,
        m_rad=0.0,
    )
    assert plus_q[0, 0, 1] == pytest.approx(1.0)
    assert plus_q[0, 1, 0] == pytest.approx(1.0)
    assert POINT_SOURCE_ROLES[0] == "centre"
    assert STOKES_MODELS == ("I", "I+Q", "I+U", "I+V")


def test_frozen_oracles_are_absent() -> None:
    assert POWER_ORACLE_VOLTAGE_PATTERN == "casa_default_evla_raytraced"
    assert power_oracle_is_frozen() is False
    assert polarization_oracle_is_frozen() is False
    assert CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE == pytest.approx(BEAM_TOLERANCE)
    with pytest.raises(FileNotFoundError, match="not frozen"):
        compare_frozen_power_oracle()


def test_unfrozen_directory_requires_default_evla_pattern_and_pa(
    tmp_path: Path,
) -> None:
    plane = _synthetic_plane(tmp_path)
    digest = hashlib.sha256(plane.path.read_bytes()).hexdigest()
    payload = {
        "voltage_pattern": "cassbeam_via_vpmanager",
        "planes": [
            {
                "fits": plane.path.name,
                "stokes": "I",
                "frequency_hz": plane.frequency_hz,
                "parallactic_angle_rad": None,
                "sha256": digest,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="default EVLA"):
        compare_power_oracle_directory(tmp_path)
    payload["voltage_pattern"] = POWER_ORACLE_VOLTAGE_PATTERN
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parallactic_angle_rad"):
        compare_power_oracle_directory(tmp_path)


@pytest.mark.skipif(
    not power_oracle_is_frozen(),
    reason="CASA awp2 power oracle is not frozen; generate it on Bacchus",
)
def test_casa_awp2_main_lobe_contract() -> None:
    comparisons = compare_frozen_power_oracle()
    assert comparisons
    assert all(item.accepted for item in comparisons)
