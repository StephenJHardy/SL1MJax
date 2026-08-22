#!/usr/bin/env python3
"""Compare hierarchical split policies with exhaustive single-split refits.

The default suite is intentionally small enough for routine development. It
uses four root leaves and varies sub-pixel morphology, UV coverage, and noise.
Every candidate score and rank is written to CSV and JSON, so larger suites can
reuse the schema without changing the analysis code.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from sl1mjax.beam import VLAPrimaryBeam, predict_beam_weights
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.inference import InferenceConfig, infer_quadtree
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.quadtree import (
    QuadtreeLeaf,
    QuadtreeSky,
    predict_quadtree_stokes_i,
    quadtree_sky_from_regular_grid,
)
from sl1mjax.refinement import (
    baseline_split_scores,
    exhaustive_single_split_oracle,
    local_four_child_lookahead,
    residual_haar_scores,
    solve_quadtree_flux_active_set,
)
from sl1mjax.sky import GaussianApproximation
from sl1mjax.split import uv_cell_split


@dataclass(frozen=True)
class Scenario:
    name: str
    root_flux: tuple[float, float, float, float]
    target: QuadtreeLeaf | None
    child_flux: tuple[float, float, float, float] | None
    coverage: str
    rows: int
    noise_std_jy: float

    def __post_init__(self) -> None:
        if (self.target is None) != (self.child_flux is None):
            raise ValueError("target and child_flux must either both be set or both be absent")
        if self.target is not None and self.child_flux is not None:
            parent_flux = self.root_flux[2 * self.target.iy + self.target.ix]
            if not np.isclose(sum(self.child_flux), parent_flux):
                raise ValueError("child flux must sum to the selected root flux")


@dataclass(frozen=True)
class BenchmarkConfig:
    steps: int = 400
    learning_rate: float = 0.1
    sparsity_weight: float = 1e-8
    leaf_penalty: float = 1e-6
    holdout_fraction: float = 0.2
    cells_per_axis: int = 6
    field_width_rad: float = 2e-4
    channels: int = 2
    solver: str = "active_set"


@dataclass(frozen=True)
class RankingMetrics:
    selected: QuadtreeLeaf
    oracle_best: QuadtreeLeaf
    ranks: dict[QuadtreeLeaf, int]
    oracle_ranks: dict[QuadtreeLeaf, int]
    spearman_rho: float
    top1_match: bool
    regret: float


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "diagonal_dense",
            (1.0, 0.3, 0.02, 0.01),
            QuadtreeLeaf(0, 0, 1),
            (0.24, 0.03, 0.02, 0.01),
            "dense",
            96,
            0.0,
        ),
        Scenario(
            "horizontal_sparse",
            (1.0, 0.3, 0.02, 0.01),
            QuadtreeLeaf(0, 0, 1),
            (0.135, 0.135, 0.015, 0.015),
            "sparse",
            48,
            1e-3,
        ),
        Scenario(
            "vertical_anisotropic",
            (1.0, 0.3, 0.02, 0.01),
            QuadtreeLeaf(0, 0, 1),
            (0.135, 0.015, 0.135, 0.015),
            "anisotropic",
            80,
            1e-3,
        ),
        Scenario(
            "faint_diagonal",
            (1.0, 0.25, 0.08, 0.02),
            QuadtreeLeaf(0, 1, 0),
            (0.065, 0.008, 0.005, 0.002),
            "dense",
            96,
            1e-3,
        ),
        Scenario(
            "smooth_control",
            (1.0, 0.3, 0.08, 0.02),
            None,
            None,
            "dense",
            96,
            0.0,
        ),
        Scenario(
            "noise_control",
            (1.0, 0.3, 0.08, 0.02),
            None,
            None,
            "dense",
            96,
            3e-3,
        ),
    )
}


def _leaf_id(leaf: QuadtreeLeaf) -> str:
    return f"{leaf.level}:{leaf.iy}:{leaf.ix}"


def _ranked(scores: dict[QuadtreeLeaf, float]) -> tuple[QuadtreeLeaf, ...]:
    return tuple(sorted(scores, key=lambda leaf: (-scores[leaf], leaf)))


def ranking_metrics(
    scores: dict[QuadtreeLeaf, float],
    oracle_scores: dict[QuadtreeLeaf, float],
) -> RankingMetrics:
    """Compare deterministic descending ranks and compute top-one regret."""

    if not scores or set(scores) != set(oracle_scores):
        raise ValueError("policy and oracle scores must cover the same non-empty leaves")
    policy_order = _ranked(scores)
    oracle_order = _ranked(oracle_scores)
    ranks = {leaf: rank for rank, leaf in enumerate(policy_order, start=1)}
    oracle_ranks = {leaf: rank for rank, leaf in enumerate(oracle_order, start=1)}
    count = len(scores)
    squared_difference = sum(
        (ranks[leaf] - oracle_ranks[leaf]) ** 2 for leaf in scores
    )
    rho = (
        1.0
        if count == 1
        else 1.0 - 6.0 * squared_difference / (count * (count**2 - 1))
    )
    selected = policy_order[0]
    oracle_best = oracle_order[0]
    return RankingMetrics(
        selected=selected,
        oracle_best=oracle_best,
        ranks=ranks,
        oracle_ranks=oracle_ranks,
        spearman_rho=float(rho),
        top1_match=selected == oracle_best,
        regret=max(0.0, oracle_scores[oracle_best] - oracle_scores[selected]),
    )


def _make_sky(scenario: Scenario, field_width_rad: float) -> tuple[QuadtreeSky, QuadtreeSky]:
    root = quadtree_sky_from_regular_grid(2, field_width_rad, scenario.root_flux)
    truth = (
        root
        if scenario.target is None or scenario.child_flux is None
        else root.split(scenario.target, child_flux=scenario.child_flux)
    )
    return root, truth


def _simulate_block(
    sky: QuadtreeSky,
    scenario: Scenario,
    seed: int,
    channels: int,
    primary_beam: VLAPrimaryBeam | None,
) -> VisibilityBlock:
    rng = np.random.default_rng(seed)
    uvw_m = rng.uniform(-6_000.0, 6_000.0, size=(scenario.rows, 3))
    if scenario.coverage == "anisotropic":
        uvw_m[:, 1] *= 0.06
    elif scenario.coverage == "sparse":
        uvw_m[:, :2] *= 0.8
    elif scenario.coverage != "dense":
        raise ValueError(f"unknown coverage {scenario.coverage!r}")
    uvw_m[:, 2] *= 0.2
    frequency_hz = 1.15e9 + np.arange(channels) * 8e6
    antenna1 = rng.integers(0, 5, scenario.rows, dtype=np.int32)
    antenna2 = (
        antenna1 + rng.integers(1, 5, scenario.rows, dtype=np.int32)
    ) % 5
    correlations = (Correlation.I,)
    l, m = sky.centers()
    beam_i, beam_rr, beam_ll = predict_beam_weights(
        primary_beam,
        l,
        m,
        frequency_hz,
    )
    model = np.asarray(
        predict_quadtree_stokes_i(
            sky.flux,
            sky.topology,
            uvw_m,
            frequency_hz,
            antenna1,
            antenna2,
            correlations,
            approximation=GaussianApproximation.PARAXIAL,
            beam_weights=beam_i,
            beam_weights_rr=beam_rr,
            beam_weights_ll=beam_ll,
        )
    )
    if scenario.noise_std_jy > 0:
        component_std = scenario.noise_std_jy / np.sqrt(2.0)
        noise = component_std * (
            rng.normal(size=model.shape) + 1j * rng.normal(size=model.shape)
        )
        visibility = model + noise
        weight = np.full(model.shape, 1.0 / scenario.noise_std_jy**2)
    else:
        visibility = model
        weight = np.ones(model.shape, dtype=np.float64)
    return VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequency_hz,
        visibility=visibility,
        weight=weight,
        flag=np.zeros(model.shape, dtype=bool),
        time_s=np.arange(scenario.rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=ReceptorBasis.STOKES,
        model_visibility=model,
        provenance={"scenario": scenario.name, "seed": seed},
    )


def _timed(call: Any) -> tuple[Any, float]:
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def run_scenario(
    scenario: Scenario,
    seed: int,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Run one complete score-versus-oracle comparison."""

    root_sky, truth = _make_sky(scenario, config.field_width_rad)
    primary_beam = None
    block = _simulate_block(truth, scenario, seed, config.channels, primary_beam)
    split = uv_cell_split(
        block,
        holdout_fraction=config.holdout_fraction,
        cells_per_axis=config.cells_per_axis,
        seed=seed + 10_000,
    )
    inference_config = InferenceConfig(
        steps=config.steps,
        learning_rate=config.learning_rate,
        sparsity_weight=config.sparsity_weight,
        initial_intensity=0.05,
        patience=config.steps + 1,
        chunk_size=41,
    )
    if config.solver == "active_set":
        root_fit, fit_seconds = _timed(
            lambda: solve_quadtree_flux_active_set(
                block,
                root_sky.topology,
                split.train,
                inference_config,
                holdout_mask=split.holdout,
                approximation=GaussianApproximation.PARAXIAL,
                leaf_penalty=config.leaf_penalty,
            )
        )
    elif config.solver == "optax":
        root_fit, fit_seconds = _timed(
            lambda: infer_quadtree(
                block,
                root_sky.topology,
                split.train,
                inference_config,
                approximation=GaussianApproximation.PARAXIAL,
                leaf_penalty=config.leaf_penalty,
            )
        )
    else:
        raise ValueError("solver must be active_set or optax")
    baseline, baseline_seconds = _timed(
        lambda: baseline_split_scores(root_fit.topology, root_fit.flux, max_depth=1)
    )
    haar, haar_seconds = _timed(
        lambda: residual_haar_scores(
            block,
            root_fit,
            split.train,
            inference_config,
            max_depth=1,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    lookahead, lookahead_seconds = _timed(
        lambda: local_four_child_lookahead(
            block,
            root_fit,
            split.train,
            inference_config,
            holdout_mask=split.holdout,
            max_depth=1,
            approximation=GaussianApproximation.PARAXIAL,
        )
    )
    oracle, oracle_seconds = _timed(
        lambda: exhaustive_single_split_oracle(
            block,
            root_fit,
            split.train,
            inference_config,
            holdout_mask=split.holdout,
            max_depth=1,
            approximation=GaussianApproximation.PARAXIAL,
            solver=config.solver,
        )
    )

    policy_scores = {
        "flux": {score.leaf: score.flux for score in baseline},
        "gradient": {score.leaf: score.gradient for score in baseline},
        "laplacian": {score.leaf: score.laplacian for score in baseline},
        "haar": {score.leaf: score.predicted_improvement for score in haar},
        "lookahead": {
            evaluation.leaf: evaluation.predicted_improvement
            for evaluation in lookahead.evaluations
        },
    }
    oracle_train = {
        evaluation.leaf: evaluation.predicted_improvement
        for evaluation in oracle.evaluations
    }
    oracle_holdout = {
        evaluation.leaf: -float(evaluation.holdout_change)
        for evaluation in oracle.evaluations
        if evaluation.holdout_change is not None
    }
    if set(oracle_train) != set(oracle_holdout):
        raise RuntimeError("oracle holdout metrics are incomplete")

    train_comparisons = {
        policy: ranking_metrics(scores, oracle_train)
        for policy, scores in policy_scores.items()
    }
    holdout_comparisons = {
        policy: ranking_metrics(scores, oracle_holdout)
        for policy, scores in policy_scores.items()
    }
    best_train = max(oracle_train.values())
    best_holdout = max(oracle_holdout.values())
    oracle_train_accepts = best_train > 0
    oracle_holdout_accepts = best_holdout > 0
    target_id = None if scenario.target is None else _leaf_id(scenario.target)
    policies = []
    for policy in policy_scores:
        train = train_comparisons[policy]
        holdout = holdout_comparisons[policy]
        policies.append(
            {
                "policy": policy,
                "selected_leaf": _leaf_id(train.selected),
                "selected_score": float(policy_scores[policy][train.selected]),
                "truth_target": target_id,
                "truth_hit": None if target_id is None else _leaf_id(train.selected) == target_id,
                "train_oracle_best_leaf": _leaf_id(train.oracle_best),
                "train_top1_match": train.top1_match,
                "train_spearman_rho": train.spearman_rho,
                "train_regret": train.regret,
                "train_oracle_accepts_split": oracle_train_accepts,
                "holdout_oracle_best_leaf": _leaf_id(holdout.oracle_best),
                "holdout_top1_match": holdout.top1_match,
                "holdout_spearman_rho": holdout.spearman_rho,
                "holdout_regret": holdout.regret,
                "holdout_oracle_accepts_split": oracle_holdout_accepts,
            }
        )

    baseline_by_leaf = {score.leaf: score for score in baseline}
    haar_by_leaf = {score.leaf: score for score in haar}
    lookahead_by_leaf = {
        evaluation.leaf: evaluation for evaluation in lookahead.evaluations
    }
    oracle_by_leaf = {evaluation.leaf: evaluation for evaluation in oracle.evaluations}
    candidates = []
    for leaf in sorted(oracle_train):
        baseline_score = baseline_by_leaf[leaf]
        haar_score = haar_by_leaf[leaf]
        local = lookahead_by_leaf[leaf]
        exhaustive = oracle_by_leaf[leaf]
        row: dict[str, Any] = {
            "scenario": scenario.name,
            "seed": seed,
            "leaf": _leaf_id(leaf),
            "truth_target": leaf == scenario.target,
            "parent_flux": float(local.parent_flux),
            "flux_score": float(baseline_score.flux),
            "gradient_score": float(baseline_score.gradient),
            "laplacian_score": float(baseline_score.laplacian),
            "haar_score": float(haar_score.predicted_improvement),
            "haar_raw_score": float(haar_score.raw_predicted_improvement),
            "haar_eligible": haar_score.eligible,
            "haar_min_eigenvalue": float(haar_score.eigenvalues[0]),
            "haar_eigenvalue_ratio": float(haar_score.eigenvalue_ratio),
            "lookahead_improvement": float(local.predicted_improvement),
            "lookahead_holdout_improvement": -float(local.holdout_change),
            "lookahead_active_children": local.active_children,
            "oracle_train_improvement": float(exhaustive.predicted_improvement),
            "oracle_holdout_improvement": -float(exhaustive.holdout_change),
        }
        for index, child_flux in enumerate(local.child_flux):
            row[f"lookahead_child_{index}_flux"] = float(child_flux)
        for policy, comparison in train_comparisons.items():
            row[f"{policy}_rank"] = comparison.ranks[leaf]
        row["oracle_train_rank"] = train_comparisons["lookahead"].oracle_ranks[leaf]
        row["oracle_holdout_rank"] = holdout_comparisons["lookahead"].oracle_ranks[leaf]
        candidates.append(row)

    return {
        "scenario": scenario.name,
        "seed": seed,
        "coverage": scenario.coverage,
        "rows": scenario.rows,
        "channels": config.channels,
        "noise_std_jy": scenario.noise_std_jy,
        "truth_target": target_id,
        "root_fit_best_step": root_fit.best_step,
        "root_fit_steps": root_fit.steps,
        "root_fit_objective_start": float(root_fit.objective_history[0]),
        "root_fit_objective_best": float(min(root_fit.objective_history)),
        "oracle_train_best_improvement": float(best_train),
        "oracle_train_accepts_split": oracle_train_accepts,
        "oracle_holdout_best_improvement": float(best_holdout),
        "oracle_holdout_accepts_split": oracle_holdout_accepts,
        "lookahead_accepts_split": lookahead.best is not None,
        "timing_s": {
            "root_fit": fit_seconds,
            "baseline": baseline_seconds,
            "haar": haar_seconds,
            "lookahead": lookahead_seconds,
            "oracle": oracle_seconds,
        },
        "policies": policies,
        "candidates": candidates,
    }


def aggregate_policy_metrics(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate policy accuracy, rank correlation, and regret over cases."""

    policy_names = sorted(
        {policy["policy"] for case in cases for policy in case["policies"]}
    )
    rows = []
    for policy_name in policy_names:
        selected = [
            policy
            for case in cases
            for policy in case["policies"]
            if policy["policy"] == policy_name
        ]
        truth_values = [
            policy["truth_hit"]
            for policy in selected
            if policy["truth_hit"] is not None
        ]
        accepted = [row for row in selected if row["train_oracle_accepts_split"]]
        rows.append(
            {
                "policy": policy_name,
                "case_count": len(selected),
                "structured_case_count": len(truth_values),
                "truth_hit_rate": (
                    None
                    if not truth_values
                    else float(np.mean(np.asarray(truth_values, dtype=np.float64)))
                ),
                "train_top1_rate": float(np.mean([row["train_top1_match"] for row in selected])),
                "train_mean_spearman_rho": float(
                    np.mean([row["train_spearman_rho"] for row in selected])
                ),
                "train_mean_regret": float(np.mean([row["train_regret"] for row in selected])),
                "accepted_case_count": len(accepted),
                "accepted_train_top1_rate": (
                    None
                    if not accepted
                    else float(np.mean([row["train_top1_match"] for row in accepted]))
                ),
                "accepted_train_mean_spearman_rho": (
                    None
                    if not accepted
                    else float(np.mean([row["train_spearman_rho"] for row in accepted]))
                ),
                "accepted_train_mean_regret": (
                    None
                    if not accepted
                    else float(np.mean([row["train_regret"] for row in accepted]))
                ),
                "accepted_holdout_top1_rate": (
                    None
                    if not accepted
                    else float(np.mean([row["holdout_top1_match"] for row in accepted]))
                ),
                "accepted_holdout_mean_spearman_rho": (
                    None
                    if not accepted
                    else float(
                        np.mean([row["holdout_spearman_rho"] for row in accepted])
                    )
                ),
                "accepted_holdout_mean_regret": (
                    None
                    if not accepted
                    else float(np.mean([row["holdout_regret"] for row in accepted]))
                ),
                "holdout_top1_rate": float(
                    np.mean([row["holdout_top1_match"] for row in selected])
                ),
                "holdout_mean_spearman_rho": float(
                    np.mean([row["holdout_spearman_rho"] for row in selected])
                ),
                "holdout_mean_regret": float(
                    np.mean([row["holdout_regret"] for row in selected])
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output: Path,
    cases: list[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    """Write the benchmark summary and normalized CSV tables."""

    output.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_policy_metrics(cases)
    candidate_rows = [row for case in cases for row in case["candidates"]]
    policy_rows = [
        {"scenario": case["scenario"], "seed": case["seed"], **policy}
        for case in cases
        for policy in case["policies"]
    ]
    payload = {
        "schema_version": "1.0",
        "config": asdict(config),
        "aggregate": aggregate,
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "candidates.csv", candidate_rows)
    _write_csv(output / "policy_summary.csv", policy_rows)
    _write_csv(output / "aggregate.csv", aggregate)
    return payload


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _comma_strings(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_refinement_benchmark"),
    )
    parser.add_argument(
        "--scenarios",
        type=_comma_strings,
        default=tuple(SCENARIOS),
        help="comma-separated scenario names",
    )
    parser.add_argument("--seeds", type=_comma_ints, default=(13,))
    parser.add_argument("--steps", type=int, default=BenchmarkConfig.steps)
    parser.add_argument("--learning-rate", type=float, default=BenchmarkConfig.learning_rate)
    parser.add_argument("--leaf-penalty", type=float, default=BenchmarkConfig.leaf_penalty)
    parser.add_argument(
        "--solver",
        choices=("active_set", "optax"),
        default=BenchmarkConfig.solver,
    )
    arguments = parser.parse_args()
    unknown = sorted(set(arguments.scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}")
    config = BenchmarkConfig(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        leaf_penalty=arguments.leaf_penalty,
        solver=arguments.solver,
    )
    cases = [
        run_scenario(SCENARIOS[name], seed, config)
        for name in arguments.scenarios
        for seed in arguments.seeds
    ]
    payload = write_results(arguments.output, cases, config)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(arguments.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
