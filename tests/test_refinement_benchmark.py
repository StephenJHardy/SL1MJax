from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sl1mjax.quadtree import QuadtreeLeaf

_SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_hierarchical_refinement.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_hierarchical_refinement", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def test_ranking_metrics_reports_order_correlation_and_regret() -> None:
    leaves = tuple(QuadtreeLeaf(0, iy, ix) for iy in range(2) for ix in range(2))
    policy = dict(zip(leaves, [4.0, 3.0, 2.0, 1.0], strict=True))
    oracle = dict(zip(leaves, [3.0, 4.0, 1.0, 2.0], strict=True))

    comparison = benchmark.ranking_metrics(policy, oracle)

    assert comparison.selected == leaves[0]
    assert comparison.oracle_best == leaves[1]
    assert comparison.spearman_rho == pytest.approx(0.6)
    assert comparison.regret == pytest.approx(1.0)
    assert not comparison.top1_match


def test_benchmark_writes_candidate_and_policy_tables(tmp_path) -> None:
    config = benchmark.BenchmarkConfig(steps=40)
    case = benchmark.run_scenario(benchmark.SCENARIOS["diagonal_dense"], 13, config)

    assert len(case["candidates"]) == 4
    assert {row["policy"] for row in case["policies"]} == {
        "flux",
        "gradient",
        "laplacian",
        "haar",
        "lookahead",
    }
    assert all("oracle_train_rank" in row for row in case["candidates"])
    assert all("oracle_holdout_rank" in row for row in case["candidates"])

    payload = benchmark.write_results(tmp_path, [case], config)

    stored = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    candidates = list(
        csv.DictReader((tmp_path / "candidates.csv").open(encoding="utf-8"))
    )
    policies = list(
        csv.DictReader((tmp_path / "policy_summary.csv").open(encoding="utf-8"))
    )
    aggregate = list(
        csv.DictReader((tmp_path / "aggregate.csv").open(encoding="utf-8"))
    )
    assert stored["schema_version"] == "1.0"
    assert stored["aggregate"] == payload["aggregate"]
    assert len(candidates) == 4
    assert len(policies) == 5
    assert len(aggregate) == 5
