from __future__ import annotations

import inspect

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.holography import (
    THOL0001_SPW4_CHANNEL_32_HZ,
    AntennaPointingRole,
    HolographyObservation,
    ResolvedAntennaPointing,
)
from sl1mjax.holography_calibration import C147_OFFSET_FIELD_IDS, HELD_OUT_REFERENCE_ANTENNA
from sl1mjax.holography_diagonal import THOL0001_REFERENCE_ANTENNA_NAMES
from sl1mjax.holography_diagonal_correction import (
    FIRST_LADDER_TERMS,
    HOLORASTER_FIELD_ID,
    SPW4_HOLDOUT_MOVING_ANTENNA_NAMES,
    SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES,
    SPW4_TRAINING_FREQUENCY_HZ,
    SPW5_FREQUENCY_HZ,
    THOL0001_MOVING_ANTENNA_NAMES,
    HoldoutScore,
    axis_improves,
    frozen_protocol_payload,
    protocol_as_mapping,
    refuse_c147_training,
    refuse_phase_in_first_ladder,
    refuse_spw5,
    require_boresight_unity,
    require_disjoint_antenna_cover,
    select_nested_correction,
    spatial_holdout_from_offsets,
    spw4_correction_holdouts,
)
from sl1mjax.polarization import Correlation, ReceptorBasis

_PHASE = (1.4660765716752369, 0.8726646259971648)
_CORR = (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)


def _antenna_names() -> tuple[str, ...]:
    names = [""] * 29
    for index in range(1, 29):
        names[index] = f"ea{index:02d}"
    return tuple(names)


def _arcmin(l_arcmin: float, m_arcmin: float) -> tuple[float, float]:
    return (np.deg2rad(l_arcmin / 60.0), np.deg2rad(m_arcmin / 60.0))


def _toy_observation() -> HolographyObservation:
    """Four antennas, two raster cells, one unused C147 row."""

    antenna1 = np.array([4, 8, 4, 8, 4, 4], dtype=np.int32)
    antenna2 = np.array([2, 2, 2, 2, 24, 2], dtype=np.int32)
    time_s = np.array([0.0, 0.0, 10.0, 10.0, 20.0, 30.0], dtype=np.float64)
    field_id = np.array([10, 10, 10, 10, 10, 3], dtype=np.int32)
    n_row = time_s.size
    block = VisibilityBlock(
        uvw_m=np.broadcast_to(np.array([30.0, -12.0, 8.0]), (n_row, 3)).copy(),
        frequency_hz=np.array([THOL0001_SPW4_CHANNEL_32_HZ]),
        visibility=np.zeros((n_row, 1, 4), dtype=np.complex128),
        weight=np.ones((n_row, 1, 4), dtype=np.float64),
        flag=np.zeros((n_row, 1, 4), dtype=bool),
        time_s=time_s,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=_CORR,
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=_PHASE,
        field_id=field_id,
        spectral_window_id=4,
    )
    unique_time = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    n_time = unique_time.size
    n_ant = 25
    offsets = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    role = np.full((n_time, n_ant), AntennaPointingRole.UNKNOWN.value, dtype="U16")
    for time_index in range(n_time):
        role[time_index, 2] = AntennaPointingRole.REFERENCE.value
        role[time_index, 24] = AntennaPointingRole.REFERENCE.value
        role[time_index, 4] = AntennaPointingRole.MOVING.value
        role[time_index, 8] = AntennaPointingRole.MOVING.value
    offsets[0, 4] = _arcmin(10.0, 0.0)
    offsets[0, 8] = _arcmin(10.0, 0.0)
    offsets[1, 4] = _arcmin(20.0, 0.0)
    offsets[1, 8] = _arcmin(20.0, 0.0)
    offsets[2, 4] = _arcmin(10.0, 0.0)
    offsets[3, 4] = _arcmin(10.0, 0.0)
    pointing = ResolvedAntennaPointing(
        unique_time_s=unique_time,
        antenna_id=np.arange(n_ant, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=np.ones((n_time, n_ant), dtype=bool),
        settled=np.ones((n_time, n_ant), dtype=bool),
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="AZELGEO",
        units="rad",
    )
    positions = np.zeros((n_ant, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(-1.6e6, -1.5e6, n_ant)
    positions[:, 1] = -5.04e6
    positions[:, 2] = 3.55e6
    return HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=positions,
        calibration_state="casa_parang_true",
        phase_centre_rad=_PHASE,
        stokes_i=8.0,
    )


def _score(axis: str, point: float, lo: float, hi: float, n: int = 20) -> HoldoutScore:
    return HoldoutScore(
        axis=axis,
        residual_power=point,
        residual_power_lo=lo,
        residual_power_hi=hi,
        n=n,
    )


def test_frozen_partition_covers_published_antennas() -> None:
    require_disjoint_antenna_cover()
    assert SPW4_HOLDOUT_MOVING_ANTENNA_NAMES == THOL0001_MOVING_ANTENNA_NAMES[3::4]
    assert SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES == THOL0001_REFERENCE_ANTENNA_NAMES[5:]
    assert HELD_OUT_REFERENCE_ANTENNA in SPW4_HOLDOUT_REFERENCE_ANTENNA_NAMES
    payload = frozen_protocol_payload()
    assert payload["training_frequency_hz"] == SPW4_TRAINING_FREQUENCY_HZ
    assert payload["spw5_status"] == "sealed"
    assert payload["phase_in_first_ladder"] is False
    assert payload["boresight"] == "C_R(0)=C_L(0)=1"
    assert payload["first_ladder"] == list(FIRST_LADDER_TERMS)
    protocol_as_mapping(payload)


def test_first_ladder_refuses_phase_spw5_and_flux_gauge() -> None:
    refuse_phase_in_first_ladder("rl_squint_scale")
    with pytest.raises(RuntimeError, match="amplitude-only"):
        refuse_phase_in_first_ladder("outer_phase")
    with pytest.raises(RuntimeError, match="amplitude-only"):
        refuse_phase_in_first_ladder("phase_offset")
    refuse_spw5(frequency_hz=SPW4_TRAINING_FREQUENCY_HZ)
    with pytest.raises(RuntimeError, match="SPW 5"):
        refuse_spw5(opened=True)
    with pytest.raises(RuntimeError, match="SPW 5"):
        refuse_spw5(frequency_hz=SPW5_FREQUENCY_HZ)
    require_boresight_unity(1.0, 1.0)
    with pytest.raises(ValueError, match="flux gauge"):
        require_boresight_unity(1.02, 1.0)
    refuse_c147_training([HOLORASTER_FIELD_ID, 0, 9])
    with pytest.raises(RuntimeError, match="C147"):
        refuse_c147_training(C147_OFFSET_FIELD_IDS[:1])


def test_spatial_checkerboard_is_row_order_independent() -> None:
    first = np.array([_arcmin(10.0, 0.0), _arcmin(20.0, 0.0)])
    second = np.array([_arcmin(20.0, 0.0), _arcmin(10.0, 0.0)])
    hold_first = spatial_holdout_from_offsets(first)
    hold_second = spatial_holdout_from_offsets(second)
    np.testing.assert_array_equal(hold_first, [False, True])
    np.testing.assert_array_equal(hold_second, [True, False])


def test_frozen_masks_isolate_training_spatial_and_mover_holdouts() -> None:
    holdouts = spw4_correction_holdouts(_toy_observation(), _antenna_names())
    np.testing.assert_array_equal(holdouts.train, [True, False, False, False, False, False])
    np.testing.assert_array_equal(
        holdouts.spatial_holdout, [False, False, True, False, False, False]
    )
    np.testing.assert_array_equal(holdouts.mover_holdout, [False, True, False, False, False, False])
    np.testing.assert_array_equal(
        holdouts.reference_holdout, [False, False, False, False, True, False]
    )
    np.testing.assert_array_equal(
        holdouts.spatial_mover_holdout, [False, False, False, True, False, False]
    )
    np.testing.assert_array_equal(holdouts.unused_c147, [False, False, False, False, False, True])
    assert not np.any(holdouts.train & holdouts.unused_c147)


def test_selection_requires_both_ranking_holdouts_with_uncertainty() -> None:
    baseline_spatial = _score("spatial", 0.10, 0.08, 0.12)
    better_spatial = _score("spatial", 0.06, 0.05, 0.07)
    overlapping_spatial = _score("spatial", 0.09, 0.07, 0.11)
    baseline_mover = _score("moving", 0.11, 0.09, 0.13)
    better_mover = _score("moving", 0.07, 0.06, 0.08)
    worse_mover = _score("moving", 0.12, 0.10, 0.14)
    better_ref = _score("reference", 0.04, 0.03, 0.05)
    baseline_ref = _score("reference", 0.10, 0.08, 0.12)
    assert axis_improves(baseline_spatial, better_spatial)
    assert not axis_improves(baseline_spatial, overlapping_spatial)
    assert (
        select_nested_correction(baseline_spatial, better_spatial, baseline_mover, better_mover)
        == "accept_candidate"
    )
    assert (
        select_nested_correction(baseline_spatial, better_spatial, baseline_mover, worse_mover)
        == "keep_baseline"
    )
    assert (
        select_nested_correction(
            baseline_spatial,
            overlapping_spatial,
            baseline_mover,
            better_mover,
        )
        == "keep_baseline"
    )
    assert (
        select_nested_correction(
            baseline_spatial,
            overlapping_spatial,
            baseline_mover,
            worse_mover,
            reference=(baseline_ref, better_ref),
        )
        == "keep_baseline"
    )


def test_holdout_builder_has_no_row_loop() -> None:
    source = inspect.getsource(spw4_correction_holdouts)
    assert "for row" not in source
    assert "for _row" not in source


def test_protocol_payload_cannot_open_phase_or_spw5() -> None:
    payload = frozen_protocol_payload()
    payload["spw5_status"] = "open"
    with pytest.raises(RuntimeError, match="SPW 5"):
        protocol_as_mapping(payload)
    payload = frozen_protocol_payload()
    payload["phase_in_first_ladder"] = True
    with pytest.raises(RuntimeError, match="phase"):
        protocol_as_mapping(payload)
