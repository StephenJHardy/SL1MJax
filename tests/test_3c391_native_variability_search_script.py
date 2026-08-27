from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky_recovery import inject_sky_component, native_variation_candidates

SCRIPT = Path(__file__).parents[1] / "scripts" / "search_3c391_native_variability.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "search_3c391_native_variability",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _block() -> tuple[VisibilityBlock, np.ndarray]:
    baselines = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    times = np.arange(6, dtype=np.float64) * 10.0
    antenna1 = np.tile([pair[0] for pair in baselines], times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], times.size)
    time_s = np.repeat(times, len(baselines))
    shape = (time_s.size, 4, 1)
    phase = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) * 0.037
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


def test_repeated_driver_search_recovers_exact_interval_and_serializes() -> None:
    module = _module()
    block, response = _block()
    candidates = native_variation_candidates(
        block,
        temporal_widths=(1, 2),
        spectral_widths=(1, 2),
    )
    candidate = module._central_candidate(
        candidates,
        block,
        kind="temporal_interval",
        width=2,
    )
    support = module._candidate_support(block, candidate)
    injected = inject_sky_component(block, response, support, 2.0)

    result = module._run_repeated_search(
        injected,
        response,
        candidates,
        true_candidate=candidate,
        split_count=3,
        seed=21,
        selection_fraction=1.0 / 6.0,
        evaluation_fraction=1.0 / 6.0,
        shortlist_size=10,
    )

    assert result["exact_candidate_selection_fraction"] == 1.0
    assert result["exact_candidate_acceptance_fraction"] == 1.0
    assert result["repeatably_recovered"]
    json.dumps(result, allow_nan=True)


def test_matched_support_threshold_reader_requires_repeatability() -> None:
    module = _module()
    case = {
        "trials": [
            {
                "injected_amplitude_jy": 0.01,
                "selected_full_evaluation_model": "supported",
                "paired_consensus_model": "supported",
                "paired_supported_selection_fraction": 0.6,
            },
            {
                "injected_amplitude_jy": 0.02,
                "selected_full_evaluation_model": "supported",
                "paired_consensus_model": "supported",
                "paired_supported_selection_fraction": 0.8,
            },
        ]
    }

    assert module._first_repeatable_amplitude(case) == pytest.approx(0.02)
