from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.sky_recovery import split_search_baselines

SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_3c391_scan_residual.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "diagnose_3c391_scan_residual",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    scripts_directory = str(SCRIPT.parent)
    sys.path.insert(0, scripts_directory)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts_directory)
    return module


def test_candidate_definition_reads_preselected_leaf_and_interval(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "selected_leaf": {"level": 1, "iy": 109, "ix": 117},
                    "selected_variation": {
                        "kind": "temporal_interval",
                        "coordinate_start": 1206.0,
                        "coordinate_stop": 1266.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    candidate = module._candidate_definition(path)

    assert (candidate.leaf.level, candidate.leaf.iy, candidate.leaf.ix) == (
        1,
        109,
        117,
    )
    assert candidate.event_start_s == 1206.0
    assert candidate.event_stop_s == 1266.0


def test_model_selection_recovers_local_event_on_unseen_baselines() -> None:
    module = _module()
    rng = np.random.default_rng(44)
    baselines = tuple(
        (first, second) for first in range(6) for second in range(first + 1, 6)
    )
    times = np.arange(5, dtype=np.float64) * 10.0
    antenna1 = np.tile([pair[0] for pair in baselines], times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], times.size)
    time_s = np.repeat(times, len(baselines))
    shape = (time_s.size, 3, 1)
    model = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))
    local = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))
    event_rows = time_s >= 30.0
    visibility = model - 0.35 * np.where(event_rows[:, None, None], local, 0.0)
    block = VisibilityBlock(
        uvw_m=np.zeros((shape[0], 3)),
        frequency_hz=4.5e9 + np.arange(shape[1]) * 2e6,
        visibility=visibility,
        model_visibility=model,
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=time_s,
        interval_s=np.full(shape[0], 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )
    split = split_search_baselines(
        block,
        selection_fraction=0.2,
        evaluation_fraction=0.2,
        seed=7,
    )
    parameter_names, penalties, statistics = module._accumulate_family_statistics(
        block,
        local,
        0.01 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape)),
        0.02 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape)),
        event_rows,
        split,
        antenna_ids=tuple(range(6)),
        reference_antenna=0,
        row_batch_size=13,
    )

    _, selected, _, coefficients, decision = module._fit_and_select(
        parameter_names,
        penalties,
        statistics,
        (0.0, 1e-4, 1e-2),
    )

    assert selected == "local_sky_event"
    event_index = parameter_names[selected].index("event_local_sky_jy")
    assert coefficients[event_index] == pytest.approx(-0.35)
    assert decision["sealed"]["relative_improvement_from_frozen"] > 0.99
    assert decision["sealed"]["relative_improvement_from_static_nuisance"] > 0.99


def test_time_series_event_labels_use_exact_large_ms_times() -> None:
    module = _module()
    rng = np.random.default_rng(8)
    baselines = ((0, 1), (0, 2), (1, 2))
    native_times = 4_778_821_000.0 + np.arange(4) * 10.0
    antenna1 = np.tile([pair[0] for pair in baselines], native_times.size)
    antenna2 = np.tile([pair[1] for pair in baselines], native_times.size)
    time_s = np.repeat(native_times, len(baselines))
    shape = (time_s.size, 2, 1)
    model = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))
    local = np.exp(1j * rng.uniform(-np.pi, np.pi, size=shape))
    event_rows = time_s >= native_times[2]
    block = VisibilityBlock(
        uvw_m=np.zeros((shape[0], 3)),
        frequency_hz=4.5e9 + np.arange(shape[1]) * 2e6,
        visibility=model - 0.2 * np.where(event_rows[:, None, None], local, 0.0),
        model_visibility=model,
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=time_s,
        interval_s=np.full(shape[0], 10.0),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=(Correlation.RR,),
        receptor_basis=ReceptorBasis.CIRCULAR,
    )

    result = module._sealed_time_series(
        block,
        "local_sky_event",
        np.asarray([0.0, 0.0, -0.2]),
        local,
        np.zeros(shape, dtype=np.complex128),
        np.zeros(shape, dtype=np.complex128),
        event_rows,
        block.active,
        antenna_ids=(0, 1, 2),
        reference_antenna=0,
        row_batch_size=5,
    )

    assert [row["in_event"] for row in result] == [False, False, True, True]
    assert all(row["weight_sum"] == row["sample_count"] for row in result)
