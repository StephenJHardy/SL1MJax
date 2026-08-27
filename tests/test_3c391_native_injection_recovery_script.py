from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.direct_operator import DirectDFTConfig
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky_recovery import (
    fit_real_sky_component,
    inject_sky_component,
    spectral_support_mask,
    split_native_baselines,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_3c391_native_injection_recovery.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_3c391_native_injection_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block() -> tuple[VisibilityBlock, np.ndarray]:
    baselines = [(0, 1), (0, 2), (1, 2), (1, 3)]
    times = np.asarray([0.0, 10.0, 20.0, 100.0, 110.0, 120.0])
    antenna1 = np.tile([pair[0] for pair in baselines], times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], times.size)
    time_s = np.repeat(times, len(baselines))
    shape = (time_s.size, 4, 1)
    phase = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) * 0.03
    response = np.exp(1j * phase)
    block = VisibilityBlock(
        uvw_m=np.zeros((shape[0], 3)),
        frequency_hz=4.5e9 + np.arange(shape[1]) * 2e6,
        visibility=np.zeros(shape, dtype=np.complex128),
        model_visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=time_s,
        interval_s=np.full(shape[0], 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    return block, response


def test_central_interval_does_not_cross_native_time_gap() -> None:
    module = _module()
    block, _ = _block()

    start, duration, times = module._central_time_interval(block, 30.0)

    assert duration == 30.0
    assert times in {(0.0, 10.0, 20.0), (100.0, 110.0, 120.0)}
    assert start == times[0]


def test_event_payload_uses_event_null_denominator_for_static_fit() -> None:
    module = _module()
    block, response = _block()
    holdout = split_native_baselines(block, evaluation_fraction=0.5, seed=2)
    event = spectral_support_mask(block, first_channel=1, channel_count=1)
    injected = inject_sky_component(block, response, event, 2.0)
    static = fit_real_sky_component(
        injected,
        response,
        np.ones(block.shape, dtype=bool),
        holdout.discovery_mask,
        holdout.evaluation_mask,
    )

    payload = module._fit_payload(
        static,
        event_support=event,
        block=injected,
        evaluation_mask=holdout.evaluation_mask,
    )

    assert static.coefficient == pytest.approx(0.5)
    assert payload["event_evaluation_relative_improvement"] == pytest.approx(0.4375)


def test_paired_spectral_trial_prefers_supported_model() -> None:
    module = _module()
    block, response = _block()
    event = spectral_support_mask(block, first_channel=2, channel_count=1)

    result = module._run_support_trials(
        block,
        response,
        event,
        injection_snrs=(0.0, 8.0),
        evaluation_fraction=0.5,
        seed=8,
        split_count=3,
    )

    assert len(result["baseline_splits"]) == 3
    assert result["trials"][0]["selected_full_evaluation_model"] == "null"
    assert result["trials"][0]["paired_consensus_model"] == "null"
    recovered = result["trials"][1]
    assert recovered["recovery_fraction"] == pytest.approx(1.0)
    assert recovered["selected_full_evaluation_model"] == "supported"
    assert recovered["paired_consensus_model"] == "supported"
    assert recovered["paired_supported_selection_fraction"] == 1.0
    json.dumps(result, allow_nan=True)


def test_unit_response_cache_rejects_changed_protocol(tmp_path: Path) -> None:
    module = _module()
    block, _ = _block()
    cache = tmp_path / "unit.npy"
    direct = DirectDFTConfig(visibility_chunk_size=8, pixel_chunk_size=1, precision="float64")
    beam = VLAPrimaryBeam()

    first = module._unit_response(
        block,
        mosaic_phase_centre_rad=block.phase_centre_rad,
        l_rad=0.0,
        m_rad=0.0,
        primary_beam=beam,
        direct=direct,
        cache=cache,
    )
    resumed = module._unit_response(
        block,
        mosaic_phase_centre_rad=block.phase_centre_rad,
        l_rad=0.0,
        m_rad=0.0,
        primary_beam=beam,
        direct=direct,
        cache=cache,
    )

    np.testing.assert_array_equal(first, resumed)
    assert cache.with_suffix(".json").exists()
    with pytest.raises(ValueError, match="does not match"):
        module._unit_response(
            block,
            mosaic_phase_centre_rad=block.phase_centre_rad,
            l_rad=1e-3,
            m_rad=0.0,
            primary_beam=beam,
            direct=direct,
            cache=cache,
        )
