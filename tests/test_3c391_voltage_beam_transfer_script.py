from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    SkyStokesPlanes,
    predict_voltage_beam,
)
from sl1mjax.cassbeam_beam import (
    CassbeamCBandVoltageBeam,
    load_cassbeam_cband_artifact,
    voltage_beam_for_mode,
)
from sl1mjax.composite import MosaicPointComponent
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_3c391_voltage_beam_transfer.py"
_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
    ]
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_3c391_voltage_beam_transfer",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flatten_positive_sky_drops_zero_flux() -> None:
    module = _module()
    components = (
        MosaicPointComponent(
            "catalogue",
            np.array([0.0, 0.01, -0.02]),
            np.array([0.0, 0.0, 0.01]),
            np.array([1.2, 0.0, 0.4]),
        ),
    )
    l_rad, m_rad, flux = module.flatten_positive_sky(components)
    np.testing.assert_allclose(l_rad, [0.0, -0.02])
    np.testing.assert_allclose(flux, [1.2, 0.4])
    assert m_rad.size == 2


def test_streamed_airy_matches_explicit_on_delta_sky() -> None:
    module = _module()
    l_rad = np.array([0.0, np.sin(np.deg2rad(0.03))])
    m_rad = np.zeros(2)
    flux = np.array([1.1, 0.3])
    frequency = np.array([4.536e9, 4.662e9])
    time_s = np.array([5.0e9, 5.0e9, 5.0e9 + 1800.0])
    dummy = np.zeros((3, 2, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.array([[20.0, -8.0, 1.0], [-11.0, 14.0, 2.0], [6.0, 9.0, -3.0]]),
        frequency_hz=frequency,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 0, 1], dtype=np.int32),
        antenna2=np.array([1, 2, 2], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(np.deg2rad(282.35), np.deg2rad(-0.93)),
    )
    explicit = module.explicit_airy_prediction(
        block,
        l_rad,
        m_rad,
        flux,
        airy_max_radius_rad_at_1ghz=np.deg2rad(4.0),
    )
    streamed = predict_voltage_beam(
        block,
        l_rad,
        m_rad,
        SkyStokesPlanes(stokes_i=flux),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1),
    )
    gate = module.operator_reproduces_explicit_airy(streamed.visibility, explicit)
    assert gate["accepted"]


def test_full_jones_diagnostic_beam_does_not_use_factory() -> None:
    module = _module()
    beams = module.construct_beams(np.deg2rad(4.0))
    assert isinstance(beams["full_jones_unfrozen"], CassbeamCBandVoltageBeam)
    assert beams["full_jones_unfrozen"].allow_unfrozen is True
    assert beams["full_jones_unfrozen"].off_diagonal is True
    factory = voltage_beam_for_mode("diagonal_copolar")
    assert factory.off_diagonal is False
    artifact = load_cassbeam_cband_artifact()
    assert artifact.pin.frozen is False
    with pytest.raises(ValueError, match="not frozen"):
        voltage_beam_for_mode("full_jones")


def test_missing_cross_hands_are_marked_absent() -> None:
    module = _module()
    dummy = np.ones((4, 2, 2), dtype=np.complex128)
    block = VisibilityBlock(
        uvw_m=np.ones((4, 3)),
        frequency_hz=np.array([4.564e9, 4.692e9]),
        visibility=dummy,
        weight=np.ones_like(dummy),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=np.linspace(1.0e9, 1.0e9 + 300.0, 4),
        antenna1=np.zeros(4, dtype=np.int32),
        antenna2=np.ones(4, dtype=np.int32),
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(0.3, -0.01),
    )
    scores = module.score_prediction(
        block,
        0.5 * dummy,
        antenna_position_m=_ANTENNA_POSITION_M,
        pointing_radius_arcmin=1.5,
        leakage_atom_fraction=0.8,
    )
    assert scores["correlations"]["RR"]["in_data"] is True
    assert scores["correlations"]["LL"]["in_data"] is True
    assert scores["correlations"]["RL"]["in_data"] is False
    assert scores["correlations"]["LR"]["held_out_loss"] is None
    assert scores["total"] > 0.0
