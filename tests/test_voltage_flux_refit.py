from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam_operator import BeamOperatorConfig
from sl1mjax.composite import MosaicPointComponent
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam
from sl1mjax.voltage_flux_refit import (
    flatten_sky_atoms,
    mosaic_local_directions,
    mosaic_weighted_mse,
    off_axis_atom_report,
    paired_score_delta,
    predict_voltage_mosaic,
    refit_stokes_i_fluxes,
    replace_component_fluxes,
    score_visibility_prediction,
    transfer_diagonal_is_consistent,
)
from sl1mjax.voltage_polarization import fit_global_qu_voltage, require_circular_coherency

_ANTENNA_POSITION_M = np.array(
    [
        [-1_601_162.0, -5_042_003.0, 3_553_983.0],
        [-1_601_100.0, -5_042_100.0, 3_553_900.0],
        [-1_601_200.0, -5_042_190.0, 3_554_000.0],
        [-1_601_050.0, -5_042_200.0, 3_553_850.0],
    ]
)
_CORRELATIONS = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _block(*, frequency_hz: np.ndarray, time_s: np.ndarray) -> VisibilityBlock:
    rows = time_s.size
    dummy = np.zeros((rows, frequency_hz.size, 4), dtype=np.complex128)
    return VisibilityBlock(
        uvw_m=np.array(
            [
                [13.0, -27.0, 4.0],
                [-19.0, 7.0, 3.0],
                [41.0, 17.0, -8.0],
                [-5.0, -11.0, 9.0],
            ],
            dtype=np.float64,
        )[:rows],
        frequency_hz=frequency_hz,
        visibility=dummy,
        weight=np.ones_like(dummy, dtype=np.float64),
        flag=np.zeros(dummy.shape, dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 0, 1, 2], dtype=np.int32)[:rows],
        antenna2=np.array([1, 2, 3, 3], dtype=np.int32)[:rows],
        correlations=_CORRELATIONS,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=(0.3, 0.1),
    )


def test_flatten_sky_keeps_zero_atoms() -> None:
    components = (
        MosaicPointComponent(
            "catalogue",
            np.array([0.0, 0.01]),
            np.array([0.0, 0.0]),
            np.array([1.2, 0.0]),
        ),
    )
    sky = flatten_sky_atoms(components)
    np.testing.assert_allclose(sky.flux, [1.2, 0.0])
    restored = replace_component_fluxes(components, np.array([0.8, 0.3]))
    np.testing.assert_allclose(restored[0].flux, [0.8, 0.3])
    np.testing.assert_allclose(restored[0].l_rad, components[0].l_rad)


def test_transfer_gate_requires_every_pointing() -> None:
    summary = {
        "pointings": {
            "C1": {"beams": {"static_scalar": {"total": 10.0}, "diagonal_copolar": {"total": 9.0}}},
            "C2": {"beams": {"static_scalar": {"total": 8.0}, "diagonal_copolar": {"total": 9.0}}},
        }
    }
    gate = transfer_diagonal_is_consistent(summary)
    assert gate["n_diagonal_beats_airy"] == 1
    assert gate["consistent"] is False


def test_paired_score_and_off_axis_report() -> None:
    block = _block(
        frequency_hz=np.array([4.6e9]),
        time_s=np.array([5.0e9, 5.0e9 + 60.0]),
    )
    prediction = np.zeros_like(block.visibility)
    block = VisibilityBlock(
        uvw_m=block.uvw_m,
        frequency_hz=block.frequency_hz,
        visibility=np.ones_like(block.visibility),
        weight=block.weight,
        flag=block.flag,
        time_s=block.time_s,
        antenna1=block.antenna1,
        antenna2=block.antenna2,
        correlations=block.correlations,
        receptor_basis=block.receptor_basis,
        phase_centre_rad=block.phase_centre_rad,
    )
    left = score_visibility_prediction(block, prediction)
    right = score_visibility_prediction(block, prediction * 0.0)
    delta = paired_score_delta(left, right)
    assert delta["total"] == pytest.approx(0.0)
    sky = flatten_sky_atoms(
        (
            MosaicPointComponent(
                "catalogue",
                np.array([0.0, np.sin(np.deg2rad(20.0 / 60.0))]),
                np.zeros(2),
                np.array([1.0, 2.0]),
            ),
        )
    )
    report = off_axis_atom_report(sky, sky.flux, np.array([1.0, 2.4]), radius_arcmin_cut=8.0)
    assert report["n_outside"] == 1
    assert report["flux_jy_delta"] == pytest.approx(0.4)


def test_refit_refuses_fista() -> None:
    block = _block(frequency_hz=np.array([4.6e9]), time_s=np.array([5.0e9, 5.0e9 + 30.0]))
    with pytest.raises(ValueError, match="FISTA"):
        refit_stokes_i_fluxes(
            (block,),
            (block.active,),
            ((np.array([0.0]), np.array([0.0])),),
            np.array([1.0]),
            AnalyticAiryVoltageBeam(),
            antenna_position_m=_ANTENNA_POSITION_M,
            config=InferenceConfig(solver="fista", steps=2),
        )


def test_refit_reduces_held_out_loss_on_airy() -> None:
    truth = np.array([1.1, 0.4])
    l_rad = np.array([0.0, np.sin(np.deg2rad(0.03))])
    m_rad = np.zeros(2)
    frequency = np.array([4.564e9, 4.692e9])
    time_s = np.array([5.0e9, 5.0e9, 5.0e9 + 1800.0, 5.0e9 + 1800.0])
    template = _block(frequency_hz=frequency, time_s=time_s)
    operator = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=2)
    predicted = predict_voltage_mosaic(
        truth,
        (template,),
        ((l_rad, m_rad),),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=operator,
    )[0]
    block = VisibilityBlock(
        uvw_m=template.uvw_m,
        frequency_hz=template.frequency_hz,
        visibility=predicted,
        weight=template.weight,
        flag=template.flag,
        time_s=template.time_s,
        antenna1=template.antenna1,
        antenna2=template.antenna2,
        correlations=template.correlations,
        receptor_basis=template.receptor_basis,
        phase_centre_rad=template.phase_centre_rad,
    )
    train = block.active.copy()
    train[2:] = False
    holdout = block.active.copy()
    holdout[:2] = False
    start = 0.4 * truth
    before = mosaic_weighted_mse((predicted * 0.4,), (block,), (holdout,))
    result = refit_stokes_i_fluxes(
        (block,),
        (train,),
        ((l_rad, m_rad),),
        start,
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        config=InferenceConfig(
            solver="proximal_sgd",
            batch_grouping="times",
            steps=4,
            learning_rate=0.2,
            sparsity_weight=0.0,
            validation_interval=4,
            operator_mode="autodiff",
        ),
        operator_config=operator,
        holdout_masks=(holdout,),
    )
    after = predict_voltage_mosaic(
        result.flux,
        (block,),
        ((l_rad, m_rad),),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=operator,
    )
    assert mosaic_weighted_mse(after, (block,), (holdout,)) < before
    assert np.all(result.flux >= 0)


def test_global_qu_keeps_v_zero_and_needs_four_hands() -> None:
    rr_ll = _block(frequency_hz=np.array([4.6e9]), time_s=np.array([5.0e9, 5.0e9 + 10.0]))
    rr_ll = VisibilityBlock(
        uvw_m=rr_ll.uvw_m,
        frequency_hz=rr_ll.frequency_hz,
        visibility=rr_ll.visibility[..., :2],
        weight=rr_ll.weight[..., :2],
        flag=rr_ll.flag[..., :2],
        time_s=rr_ll.time_s,
        antenna1=rr_ll.antenna1,
        antenna2=rr_ll.antenna2,
        correlations=(Correlation.RR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=rr_ll.phase_centre_rad,
    )
    with pytest.raises(ValueError, match="RR, RL, LR, and LL"):
        require_circular_coherency(rr_ll)
    l_rad = np.array([0.0])
    m_rad = np.array([0.0])
    frequency = np.array([4.564e9])
    time_s = np.array([5.0e9, 5.0e9 + 60.0])
    template = _block(frequency_hz=frequency, time_s=time_s)
    operator = BeamOperatorConfig(visibility_chunk_size=2, pixel_chunk_size=1)
    intensity = np.array([0.8])
    predicted = predict_voltage_mosaic(
        intensity,
        (template,),
        ((l_rad, m_rad),),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        calibration_state="casa_parang_true",
        config=operator,
        stokes_q=0.05 * intensity,
        stokes_u=-0.02 * intensity,
    )[0]
    block = VisibilityBlock(
        uvw_m=template.uvw_m,
        frequency_hz=template.frequency_hz,
        visibility=predicted,
        weight=template.weight,
        flag=template.flag,
        time_s=template.time_s,
        antenna1=template.antenna1,
        antenna2=template.antenna2,
        correlations=template.correlations,
        receptor_basis=template.receptor_basis,
        phase_centre_rad=template.phase_centre_rad,
    )
    fitted = fit_global_qu_voltage(
        (block,),
        intensity,
        ((l_rad, m_rad),),
        AnalyticAiryVoltageBeam(),
        antenna_position_m=_ANTENNA_POSITION_M,
        train_masks=(block.active,),
        config=operator,
        steps=2,
        learning_rate=0.02,
    )
    assert fitted.v == 0.0
    assert fitted.regional_polarization == "not_started"
    assert fitted.q != 0.0 or fitted.u != 0.0


def test_mosaic_local_directions_round_trip() -> None:
    l_rad = np.array([0.01, -0.02])
    m_rad = np.array([0.0, 0.015])
    centre = (0.3, 0.1)
    local_l, local_m = mosaic_local_directions(l_rad, m_rad, centre, centre)
    np.testing.assert_allclose(local_l, l_rad, atol=1e-12)
    np.testing.assert_allclose(local_m, m_rad, atol=1e-12)
