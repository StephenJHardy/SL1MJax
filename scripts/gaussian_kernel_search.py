#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib>=3.11.1",
#     "scipy>=1.18.0",
# ]
# ///
"""Search for positive, monotone radial Gaussian-mixture sampling kernels.

The canonical kernel is

    K(r) = sum_j a_j exp(-r**2 / (2 sigma_j**2))

on a square lattice with unit spacing.  Amplitudes may be signed, but the
composite kernel is constrained to be non-negative and non-increasing on a
dense radial grid.  Unit integrated flux is imposed analytically:

    2*pi*sum_j a_j*sigma_j**2 = 1.

The global stage uses differential evolution with feasibility penalties; an
optional SLSQP stage then applies the sampled inequalities as hard constraints.
This is a numerical search, not a proof of positivity between grid points.

Examples
--------
Quick smoke test:
    uv run scripts/gaussian_kernel_search.py --quick --n-max 3

Default N=1,...,8 search and plots:
    uv run scripts/gaussian_kernel_search.py --plot --save-prefix kernel_search

More thorough search:
    uv run scripts/gaussian_kernel_search.py --de-maxiter 400 --de-popsize 15 \
        --pu-grid 61 --ref-grid 61 --radial-grid 1000 --plot
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

Array = NDArray[np.float64]


@dataclass
class Weights:
    """Weights for dimensionless terms in the scalar search objective."""

    pu: float = 1.0
    tail_integrated: float = 1.0
    tail_max_relative: float = 0.25
    refinement: float = 0.25
    conditioning: float = 1.0e-3


@dataclass
class Config:
    # Geometry and sampling grids.
    tail_radius: float = 1.0
    lattice_radius: int = 7
    pu_grid: int = 31
    ref_grid: int = 33
    ref_extent: float = 2.5
    radial_grid: int = 320
    radial_max: float | None = None

    # Parameter bounds and sampled shape constraints.
    sigma_min: float = 0.08
    sigma_max: float = 1.0
    min_sigma_ratio: float = 1.03
    amplitude_bound: float = 20.0
    max_flux_condition: float = 20.0
    constraint_tolerance: float = 2.0e-8

    # Optimizer controls.
    seed: int = 20260818
    de_maxiter: int = 80
    de_popsize: int = 6
    de_tol: float = 2.0e-6
    local_maxiter: int = 500
    penalty: float = 1.0e5
    use_local: bool = True


@dataclass
class Fit:
    n: int
    amplitudes: Array
    sigmas: Array
    metrics: dict[str, float]
    objective: float
    min_constraint: float
    global_message: str
    local_message: str


class KernelProblem:
    """Objective and constraints for one mixture order N."""

    def __init__(
        self,
        n: int,
        cfg: Config,
        weights: Weights,
        initial_amplitudes: Array | None = None,
        initial_sigmas: Array | None = None,
    ):
        if n < 1:
            raise ValueError("N must be at least 1")
        self.n = n
        self.cfg = cfg
        self.weights = weights
        if (initial_amplitudes is None) != (initial_sigmas is None):
            raise ValueError("initial amplitudes and sigmas must be provided together")
        self.initial_amplitudes = (
            None
            if initial_amplitudes is None
            else np.asarray(initial_amplitudes, dtype=np.float64)
        )
        self.initial_sigmas = (
            None if initial_sigmas is None else np.asarray(initial_sigmas, dtype=np.float64)
        )
        if self.initial_amplitudes is not None and self.initial_sigmas is not None:
            if self.initial_amplitudes.shape != (n,) or self.initial_sigmas.shape != (n,):
                raise ValueError(f"warm start must contain exactly {n} amplitudes and sigmas")
            if not np.all(np.isfinite(self.initial_amplitudes)):
                raise ValueError("warm-start amplitudes must be finite")
            if not np.all(np.isfinite(self.initial_sigmas)):
                raise ValueError("warm-start sigmas must be finite")
            if np.any(np.diff(self.initial_sigmas) < 0):
                raise ValueError("warm-start sigmas must be sorted")
            if np.any(
                (self.initial_sigmas < cfg.sigma_min)
                | (self.initial_sigmas > cfg.sigma_max)
            ):
                raise ValueError("warm-start sigmas must lie within configured bounds")

        if cfg.min_sigma_ratio ** (n - 1) > cfg.sigma_max / cfg.sigma_min:
            raise ValueError("sigma range is too small for N and min_sigma_ratio")

        self.log_sigma_min = math.log(cfg.sigma_min)
        self.log_sigma_max = math.log(cfg.sigma_max)
        self.min_log_gap = math.log(cfg.min_sigma_ratio)

        # A raw width vector is sorted in decode().  This avoids the roughly
        # 1/N! feasible-volume loss that explicit ordering causes during DE.
        self.bounds = (
            [(-cfg.amplitude_bound, cfg.amplitude_bound)] * (n - 1)
            + [(self.log_sigma_min, self.log_sigma_max)] * n
        )

        self.cell_axis = np.linspace(-0.5, 0.5, cfg.pu_grid)
        self.lattice_shifts = np.arange(
            -cfg.lattice_radius, cfg.lattice_radius + 1, dtype=float
        )

        axis = np.linspace(-cfg.ref_extent, cfg.ref_extent, cfg.ref_grid)
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        self.ref_r_coarse = np.hypot(xx, yy).ravel()
        # Children are at quarter-cell offsets.  The fine spacing is 1/2.
        offsets = ((-0.25, -0.25), (-0.25, 0.25),
                   (0.25, -0.25), (0.25, 0.25))
        self.ref_r_children = np.stack(
            [np.hypot(xx - dx, yy - dy).ravel() for dx, dy in offsets]
        )

        radial_max = cfg.radial_max
        if radial_max is None:
            radial_max = max(4.0 * cfg.tail_radius, 8.0 * cfg.sigma_max)
        if radial_max <= cfg.tail_radius:
            raise ValueError("radial_max must exceed tail_radius")
        self.radial_max = radial_max
        self.radial_r = np.linspace(0.0, radial_max, cfg.radial_grid)
        self.tail_r = np.linspace(
            cfg.tail_radius, radial_max, max(80, cfg.radial_grid // 2)
        )

    def decode(self, x: Array) -> tuple[Array, Array]:
        """Decode parameters, eliminating one amplitude via unit flux."""
        # Some SLSQP line searches transiently propose non-finite or just-outside
        # values even with bounds.  Clipping keeps those trial evaluations finite.
        raw_a = np.asarray(x[: self.n - 1], dtype=float)
        free_a = np.clip(
            np.nan_to_num(raw_a, nan=0.0, posinf=self.cfg.amplitude_bound,
                          neginf=-self.cfg.amplitude_bound),
            -self.cfg.amplitude_bound,
            self.cfg.amplitude_bound,
        )
        raw_logs = np.asarray(x[self.n - 1 :], dtype=float)
        raw_logs = np.nan_to_num(
            raw_logs, nan=0.5 * (self.log_sigma_min + self.log_sigma_max),
            posinf=self.log_sigma_max, neginf=self.log_sigma_min,
        )
        log_sigmas = np.sort(
            np.clip(raw_logs, self.log_sigma_min, self.log_sigma_max)
        )
        sigmas = np.exp(log_sigmas)
        target = 1.0 / (2.0 * np.pi)
        last_a = (target - np.dot(free_a, sigmas[:-1] ** 2)) / sigmas[-1] ** 2
        amplitudes = np.concatenate((free_a, np.array([last_a])))
        return amplitudes, sigmas

    @staticmethod
    def kernel(r: Array, amplitudes: Array, sigmas: Array) -> Array:
        r = np.asarray(r, dtype=float)
        exponent = -0.5 * (r[..., None] / sigmas) ** 2
        # Explicit elementwise reduction avoids dispatching thousands of tiny
        # vector products to BLAS during optimization.
        return np.sum(np.exp(exponent) * amplitudes, axis=-1)

    def pu_surface(
        self, amplitudes: Array, sigmas: Array, lattice_radius: int | None = None
    ) -> Array:
        """Truncated square-lattice sum over one periodic unit cell."""
        if lattice_radius is None:
            shifts = self.lattice_shifts
        else:
            shifts = np.arange(-lattice_radius, lattice_radius + 1, dtype=float)
        distances = self.cell_axis[:, None] - shifts[None, :]
        surface = np.zeros((self.cell_axis.size, self.cell_axis.size))
        for a, sigma in zip(amplitudes, sigmas, strict=True):
            one_d = np.exp(-0.5 * (distances / sigma) ** 2).sum(axis=1)
            surface += a * np.outer(one_d, one_d)
        return surface

    def metrics(self, amplitudes: Array, sigmas: Array) -> dict[str, float]:
        surface = self.pu_surface(amplitudes, sigmas)
        pu_max = float(np.max(np.abs(surface - 1.0)))
        pu_rms = float(np.sqrt(np.mean((surface - 1.0) ** 2)))

        # The denominator is one because unit integrated flux is exact.
        tail_terms = (
            2.0
            * np.pi
            * amplitudes
            * sigmas**2
            * np.exp(-0.5 * (self.cfg.tail_radius / sigmas) ** 2)
        )
        tail_integrated = float(np.sum(tail_terms))
        tail_values = self.kernel(self.tail_r, amplitudes, sigmas)
        tail_max = float(np.max(np.abs(tail_values)))
        peak = float(self.kernel(np.array([0.0]), amplitudes, sigmas)[0])
        tail_max_relative = tail_max / max(abs(peak), 1.0e-15)

        coarse = self.kernel(self.ref_r_coarse, amplitudes, sigmas)
        # A unit-flux child at half scale is 4*K(2r).  Averaging four
        # unit-flux children gives sum_q K(2|x-d_q|), also of unit flux.
        children = np.zeros_like(coarse)
        for child_r in self.ref_r_children:
            children += self.kernel(2.0 * child_r, amplitudes, sigmas)
        refinement = float(
            np.sqrt(np.mean((coarse - children) ** 2))
            / max(np.sqrt(np.mean(coarse**2)), 1.0e-15)
        )

        flux_condition = float(2.0 * np.pi * np.sum(np.abs(amplitudes) * sigmas**2))
        peak_condition = float(np.sum(np.abs(amplitudes)) / max(abs(peak), 1.0e-15))
        norm = float(2.0 * np.pi * np.dot(amplitudes, sigmas**2))
        radial_values = self.kernel(self.radial_r, amplitudes, sigmas)
        q = np.sum(
            np.exp(-0.5 * (self.radial_r[:, None] / sigmas) ** 2)
            * (amplitudes / sigmas**2),
            axis=1,
        )
        derivative = -self.radial_r * q

        return {
            "pu_max": pu_max,
            "pu_rms": pu_rms,
            "tail_integrated": tail_integrated,
            "tail_max": tail_max,
            "tail_max_relative": tail_max_relative,
            "refinement": refinement,
            "flux_condition": flux_condition,
            "peak_condition": peak_condition,
            "normalization": norm,
            "peak": peak,
            "min_kernel": float(np.min(radial_values)),
            "max_derivative": float(np.max(derivative)),
        }

    def core_objective(self, x: Array) -> float:
        amplitudes, sigmas = self.decode(x)
        m = self.metrics(amplitudes, sigmas)
        w = self.weights
        # abs() prevents an infeasible negative-tailed candidate from being
        # rewarded during the global penalty phase.
        return float(
            w.pu * m["pu_max"]
            + w.tail_integrated * abs(m["tail_integrated"])
            + w.tail_max_relative * m["tail_max_relative"]
            + w.refinement * m["refinement"]
            + w.conditioning * max(0.0, m["flux_condition"] - 1.0)
        )

    def constraint_values(self, x: Array) -> Array:
        """Return sampled g(x) values for constraints g(x) >= 0."""
        amplitudes, sigmas = self.decode(x)
        log_gaps = np.diff(np.log(sigmas)) - self.min_log_gap
        radial_values = self.kernel(self.radial_r, amplitudes, sigmas)

        # K'(r) = -r*q(r), so q >= 0 enforces non-increasing K for r > 0.
        q = np.sum(
            np.exp(-0.5 * (self.radial_r[:, None] / sigmas) ** 2)
            * (amplitudes / sigmas**2),
            axis=1,
        )
        flux_condition = 2.0 * np.pi * np.sum(np.abs(amplitudes) * sigmas**2)
        scalars = np.array(
            [
                amplitudes[-1],  # positive widest tail asymptotically
                self.cfg.amplitude_bound - np.max(np.abs(amplitudes)),
                self.cfg.max_flux_condition - flux_condition,
            ]
        )
        return np.concatenate((log_gaps, scalars, radial_values, q))

    def violation_score(self, x: Array) -> float:
        """Scale-free-ish violation measure used only by global DE."""
        amplitudes, sigmas = self.decode(x)
        gaps = np.minimum(np.diff(np.log(sigmas)) - self.min_log_gap, 0.0)
        k = self.kernel(self.radial_r, amplitudes, sigmas)
        q = np.sum(
            np.exp(-0.5 * (self.radial_r[:, None] / sigmas) ** 2)
            * (amplitudes / sigmas**2),
            axis=1,
        )
        k_scale = max(abs(float(k[0])), 1.0e-3)
        q_scale = max(abs(float(q[0])), 1.0e-3)
        flux_condition = 2.0 * np.pi * np.sum(np.abs(amplitudes) * sigmas**2)

        blocks = [
            gaps / max(self.min_log_gap, 1.0e-3),
            np.minimum(k / k_scale, 0.0),
            np.minimum(q / q_scale, 0.0),
            np.array([min(amplitudes[-1] / self.cfg.amplitude_bound, 0.0)]),
            np.array([
                min(1.0 - np.max(np.abs(amplitudes)) / self.cfg.amplitude_bound, 0.0)
            ]),
            np.array([
                min(1.0 - flux_condition / self.cfg.max_flux_condition, 0.0)
            ]),
        ]
        negative = np.concatenate(blocks)
        if negative.size == 0:
            return 0.0
        return float(np.mean(negative**2) + np.max(np.abs(negative)) ** 2)

    def penalized_objective(self, x: Array) -> float:
        return self.core_objective(x) + self.cfg.penalty * self.violation_score(x)

    def initial_point(self) -> Array:
        """A feasible single-active-Gaussian seed embedded in the N-term model."""
        if self.initial_amplitudes is not None and self.initial_sigmas is not None:
            return np.concatenate(
                (self.initial_amplitudes[:-1], np.log(self.initial_sigmas))
            )
        if self.n == 1:
            sigmas = np.array([min(max(0.48, self.cfg.sigma_min), self.cfg.sigma_max)])
        else:
            low = max(self.cfg.sigma_min * 1.05, 0.12)
            required_high = low * self.cfg.min_sigma_ratio ** (self.n - 1)
            high = min(self.cfg.sigma_max / 1.02, max(0.72, required_high * 1.01))
            if high <= required_high:
                low = self.cfg.sigma_min
                high = self.cfg.sigma_max
            sigmas = np.geomspace(low, high, self.n)
        free_a = np.zeros(self.n - 1)
        return np.concatenate((free_a, np.log(sigmas)))

    def fit(self) -> Fit:
        x0 = self.initial_point()
        de = differential_evolution(
            self.penalized_objective,
            self.bounds,
            seed=self.cfg.seed + self.n,
            maxiter=self.cfg.de_maxiter,
            popsize=self.cfg.de_popsize,
            tol=self.cfg.de_tol,
            polish=False,
            updating="immediate",
            workers=1,
            x0=x0,
        )

        local_message = "disabled"
        candidates = [x0, np.asarray(de.x, dtype=float)]
        if self.cfg.use_local:
            messages = []
            starts = [("global", np.asarray(de.x, dtype=float))]
            if self.initial_amplitudes is not None:
                starts.append(("warm", x0))
            for label, start in starts:
                local = minimize(
                    self.core_objective,
                    start,
                    method="SLSQP",
                    bounds=self.bounds,
                    constraints={"type": "ineq", "fun": self.constraint_values},
                    options={
                        "maxiter": self.cfg.local_maxiter,
                        "ftol": 1.0e-11,
                        "disp": False,
                    },
                )
                candidates.append(np.asarray(local.x, dtype=float))
                messages.append(f"{label}: {local.message}")
            local_message = "; ".join(messages)

        # Prefer feasible candidates; otherwise retain the least-bad penalized
        # candidate and make its violation visible in the printed summary.
        tol = self.cfg.constraint_tolerance

        def rank(x: Array) -> tuple[int, float]:
            min_g = float(np.min(self.constraint_values(x)))
            if min_g >= -tol:
                return (0, self.core_objective(x))
            return (1, self.penalized_objective(x))

        best = min(candidates, key=rank)
        amplitudes, sigmas = self.decode(best)
        metrics = self.metrics(amplitudes, sigmas)
        min_constraint = float(np.min(self.constraint_values(best)))
        return Fit(
            n=self.n,
            amplitudes=amplitudes,
            sigmas=sigmas,
            metrics=metrics,
            objective=self.core_objective(best),
            min_constraint=min_constraint,
            global_message=str(de.message),
            local_message=local_message,
        )


def print_results(fits: Sequence[Fit], cfg: Config, weights: Weights) -> None:
    print("\nObjective weights:", weights)
    print(
        "\n N   PU max     tail int   tail/peak   refine     cond     min(g)    objective"
    )
    print("-" * 88)
    for fit in fits:
        m = fit.metrics
        print(
            f"{fit.n:2d}  {m['pu_max']:9.2e}  {m['tail_integrated']:9.2e}  "
            f"{m['tail_max_relative']:9.2e}  {m['refinement']:9.2e}  "
            f"{m['flux_condition']:7.2f}  {fit.min_constraint:9.2e}  "
            f"{fit.objective:9.2e}"
        )

    for fit in fits:
        problem = KernelProblem(fit.n, cfg, weights)
        larger = problem.pu_surface(
            fit.amplitudes, fit.sigmas, cfg.lattice_radius + 2
        )
        normal = problem.pu_surface(fit.amplitudes, fit.sigmas)
        truncation_delta = float(np.max(np.abs(larger - normal)))
        print(f"\nN={fit.n}")
        print("  amplitudes =", np.array2string(fit.amplitudes, precision=10))
        print("  sigmas     =", np.array2string(fit.sigmas, precision=10))
        print(
            f"  normalization={fit.metrics['normalization']:.12g}, "
            f"peak={fit.metrics['peak']:.6g}, "
            f"min(K)={fit.metrics['min_kernel']:.3e}, "
            f"max(K')={fit.metrics['max_derivative']:.3e}"
        )
        print(
            f"  lattice truncation check (L versus L+2): {truncation_delta:.3e}"
        )
        if fit.min_constraint < -cfg.constraint_tolerance:
            print("  WARNING: sampled constraints were not fully satisfied")


def validate_fits(
    fits: Sequence[Fit],
    cfg: Config,
    weights: Weights,
    multiplier: int,
) -> None:
    """Re-evaluate final kernels on independent denser grids and in float32."""

    if multiplier <= 1:
        return
    dense_cfg = replace(
        cfg,
        pu_grid=(cfg.pu_grid - 1) * multiplier + 1,
        ref_grid=(cfg.ref_grid - 1) * multiplier + 1,
        radial_grid=cfg.radial_grid * multiplier,
    )
    print(
        "\nIndependent dense validation grids: "
        f"PU={dense_cfg.pu_grid}, refinement={dense_cfg.ref_grid}, "
        f"radial={dense_cfg.radial_grid}"
    )
    print(
        " N   PU max     tail int   tail/peak   refine     cond   "
        "min(K)    max(K')   f32 ΔK"
    )
    print("-" * 103)
    for fit in fits:
        problem = KernelProblem(fit.n, dense_cfg, weights)
        metrics = problem.metrics(fit.amplitudes, fit.sigmas)
        amplitudes_f32 = fit.amplitudes.astype(np.float32).astype(np.float64)
        sigmas_f32 = fit.sigmas.astype(np.float32).astype(np.float64)
        raw_norm = 2.0 * np.pi * np.dot(amplitudes_f32, sigmas_f32**2)
        amplitudes_f32 /= raw_norm
        rounded_metrics = problem.metrics(amplitudes_f32, sigmas_f32)
        radius = problem.radial_r
        f32_delta = float(
            np.max(
                np.abs(
                    problem.kernel(radius, amplitudes_f32, sigmas_f32)
                    - problem.kernel(radius, fit.amplitudes, fit.sigmas)
                )
            )
        )
        print(
            f"{fit.n:2d}  {metrics['pu_max']:9.2e}  {metrics['tail_integrated']:9.2e}  "
            f"{metrics['tail_max_relative']:9.2e}  {metrics['refinement']:9.2e}  "
            f"{metrics['flux_condition']:7.2f}  {metrics['min_kernel']:9.2e}  "
            f"{metrics['max_derivative']:9.2e}  {f32_delta:8.2e}"
        )
        if rounded_metrics["min_kernel"] < -dense_cfg.constraint_tolerance:
            print("    WARNING: normalized float32 coefficients violate positivity")
        if rounded_metrics["max_derivative"] > dense_cfg.constraint_tolerance:
            print("    WARNING: normalized float32 coefficients violate monotonicity")


def save_csv(fits: Sequence[Fit], path: Path) -> None:
    fields = [
        "N", "objective", "pu_max", "pu_rms", "tail_integrated", "tail_max",
        "tail_max_relative", "refinement", "flux_condition", "peak_condition",
        "normalization", "peak", "min_kernel", "max_derivative",
        "min_constraint", "amplitudes", "sigmas",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fit in fits:
            row: dict[str, int | float | str] = {
                "N": fit.n,
                "objective": fit.objective,
                "min_constraint": fit.min_constraint,
            }
            row.update(fit.metrics)
            row["amplitudes"] = " ".join(f"{v:.17g}" for v in fit.amplitudes)
            row["sigmas"] = " ".join(f"{v:.17g}" for v in fit.sigmas)
            writer.writerow(row)


def make_plots(
    fits: Sequence[Fit], cfg: Config, weights: Weights, save_prefix: str | None
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(fits)))

    rmax = max(2.0 * cfg.tail_radius, 2.0)
    r = np.linspace(0.0, rmax, 600)
    for fit, color in zip(fits, colors, strict=True):
        problem = KernelProblem(fit.n, cfg, weights)
        axes[0, 0].plot(
            r, problem.kernel(r, fit.amplitudes, fit.sigmas),
            color=color, label=f"N={fit.n}",
        )
    axes[0, 0].axvline(cfg.tail_radius, color="0.4", ls="--", lw=1)
    axes[0, 0].axhline(0.0, color="0.6", lw=0.8)
    axes[0, 0].set(title="Composite kernels", xlabel="r / lattice spacing", ylabel="K(r)")
    axes[0, 0].legend(ncol=2, fontsize=8)

    best = fits[-1]
    problem = KernelProblem(best.n, cfg, weights)
    ripple = problem.pu_surface(best.amplitudes, best.sigmas) - 1.0
    image = axes[0, 1].imshow(
        ripple,
        origin="lower",
        extent=(-0.5, 0.5, -0.5, 0.5),
        cmap="coolwarm",
        aspect="equal",
    )
    axes[0, 1].set(
        title=f"N={best.n} square-lattice PU ripple",
        xlabel="x", ylabel="y",
    )
    fig.colorbar(image, ax=axes[0, 1], label="sum K - 1")

    ns = np.array([fit.n for fit in fits])
    metric_series = {
        "PU max": [fit.metrics["pu_max"] for fit in fits],
        "tail fraction": [abs(fit.metrics["tail_integrated"]) for fit in fits],
        "tail / peak": [fit.metrics["tail_max_relative"] for fit in fits],
        "refinement": [fit.metrics["refinement"] for fit in fits],
    }
    for label, values in metric_series.items():
        axes[1, 0].semilogy(ns, np.maximum(values, 1.0e-16), "o-", label=label)
    axes[1, 0].set(
        title="Cost/accuracy trade-off", xlabel="number of Gaussians N",
        ylabel="metric (log scale)", xticks=ns,
    )
    axes[1, 0].grid(True, which="both", alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    condition = [fit.metrics["flux_condition"] for fit in fits]
    objective = [fit.objective for fit in fits]
    axes[1, 1].plot(ns, condition, "o-", color="tab:purple", label="flux condition")
    axes[1, 1].set(
        title="Cancellation and scalar objective", xlabel="N", ylabel="flux condition",
        xticks=ns,
    )
    twin = axes[1, 1].twinx()
    twin.semilogy(ns, np.maximum(objective, 1.0e-16), "s--", color="tab:orange",
                  label="objective")
    twin.set_ylabel("objective (log scale)")
    lines = axes[1, 1].get_lines() + twin.get_lines()
    axes[1, 1].legend(lines, [line.get_label() for line in lines], fontsize=8)

    if save_prefix:
        path = Path(f"{save_prefix}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print(f"Saved plot to {path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tail-radius", "--R", dest="tail_radius", type=float, default=1.0)
    parser.add_argument("--sigma-min", type=float, default=0.08)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--min-sigma-ratio", type=float, default=1.03)
    parser.add_argument("--amplitude-bound", type=float, default=20.0)
    parser.add_argument("--condition-max", type=float, default=20.0)
    parser.add_argument("--lattice-radius", type=int, default=7)
    parser.add_argument("--pu-grid", type=int, default=31)
    parser.add_argument("--ref-grid", type=int, default=33)
    parser.add_argument("--ref-extent", type=float, default=2.5)
    parser.add_argument("--radial-grid", type=int, default=320)
    parser.add_argument("--radial-max", type=float, default=None)
    parser.add_argument("--de-maxiter", type=int, default=80)
    parser.add_argument("--de-popsize", type=int, default=6)
    parser.add_argument("--local-maxiter", type=int, default=500)
    parser.add_argument("--penalty", type=float, default=1.0e5)
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="small grids and optimizer budgets for a smoke test")
    parser.add_argument("--w-pu", type=float, default=1.0)
    parser.add_argument("--w-tail-integrated", type=float, default=1.0)
    parser.add_argument("--w-tail-max", type=float, default=0.25)
    parser.add_argument("--w-refinement", type=float, default=0.25)
    parser.add_argument("--w-conditioning", type=float, default=1.0e-3)
    parser.add_argument("--initial-amplitudes", type=float, nargs="+", default=None)
    parser.add_argument("--initial-sigmas", type=float, nargs="+", default=None)
    parser.add_argument(
        "--validate-multiplier",
        type=int,
        default=1,
        help="re-evaluate final kernels on grids this many times denser",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-prefix", type=str, default=None,
                        help="save PREFIX.png and PREFIX.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_min < 1 or args.n_max < args.n_min:
        raise SystemExit("Require 1 <= n-min <= n-max")
    if (args.initial_amplitudes is None) != (args.initial_sigmas is None):
        raise SystemExit("Provide both --initial-amplitudes and --initial-sigmas")
    if args.initial_amplitudes is not None:
        if args.n_min != args.n_max:
            raise SystemExit("A warm start requires n-min and n-max to select one N")
        if len(args.initial_amplitudes) != args.n_min:
            raise SystemExit("Warm-start amplitude count must equal N")
        if len(args.initial_sigmas) != args.n_min:
            raise SystemExit("Warm-start sigma count must equal N")
    if args.validate_multiplier < 1:
        raise SystemExit("--validate-multiplier must be at least one")

    if args.quick:
        args.de_maxiter = min(args.de_maxiter, 15)
        args.de_popsize = min(args.de_popsize, 5)
        args.local_maxiter = min(args.local_maxiter, 150)
        args.pu_grid = min(args.pu_grid, 21)
        args.ref_grid = min(args.ref_grid, 23)
        args.radial_grid = min(args.radial_grid, 160)

    cfg = Config(
        tail_radius=args.tail_radius,
        lattice_radius=args.lattice_radius,
        pu_grid=args.pu_grid,
        ref_grid=args.ref_grid,
        ref_extent=args.ref_extent,
        radial_grid=args.radial_grid,
        radial_max=args.radial_max,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        min_sigma_ratio=args.min_sigma_ratio,
        amplitude_bound=args.amplitude_bound,
        max_flux_condition=args.condition_max,
        seed=args.seed,
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
        local_maxiter=args.local_maxiter,
        penalty=args.penalty,
        use_local=not args.no_local,
    )
    weights = Weights(
        pu=args.w_pu,
        tail_integrated=args.w_tail_integrated,
        tail_max_relative=args.w_tail_max,
        refinement=args.w_refinement,
        conditioning=args.w_conditioning,
    )

    print("Configuration:", cfg)
    fits: list[Fit] = []
    for n in range(args.n_min, args.n_max + 1):
        print(f"Searching N={n} ...", flush=True)
        fit = KernelProblem(
            n,
            cfg,
            weights,
            initial_amplitudes=args.initial_amplitudes,
            initial_sigmas=args.initial_sigmas,
        ).fit()
        fits.append(fit)
        print(
            f"  objective={fit.objective:.4e}, min constraint={fit.min_constraint:.3e}"
        )

    print_results(fits, cfg, weights)
    validate_fits(fits, cfg, weights, args.validate_multiplier)
    if args.save_prefix:
        csv_path = Path(f"{args.save_prefix}.csv")
        save_csv(fits, csv_path)
        print(f"Saved metrics and coefficients to {csv_path}")
    if args.plot or args.save_prefix:
        make_plots(fits, cfg, weights, args.save_prefix)


if __name__ == "__main__":
    main()
