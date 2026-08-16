"""Deterministic polarized synthetic observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import RegularGrid


@dataclass(frozen=True)
class PointSource:
    flux: float
    l: float
    m: float


def correlations_for_basis(basis: ReceptorBasis) -> tuple[Correlation, ...]:
    if basis is ReceptorBasis.LINEAR:
        return (Correlation.XX, Correlation.XY, Correlation.YX, Correlation.YY)
    if basis is ReceptorBasis.CIRCULAR:
        return (Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL)
    return (Correlation.I, Correlation.Q, Correlation.U, Correlation.V)


def default_sources(grid: RegularGrid) -> tuple[PointSource, ...]:
    l, m = grid.coordinates
    locations = (
        (grid.size // 2, grid.size // 2),
        (grid.size // 2 - 2, grid.size // 2 + 2),
        (grid.size // 2 + 2, grid.size // 2 - 2),
    )
    return tuple(
        PointSource(
            flux,
            float(l[row * grid.size + column]),
            float(m[row * grid.size + column]),
        )
        for flux, (row, column) in zip((1.0, 0.5, 0.25), locations, strict=True)
    )


def simulate_dataset(
    grid: RegularGrid,
    *,
    basis: ReceptorBasis = ReceptorBasis.LINEAR,
    sources: tuple[PointSource, ...] | None = None,
    rows: int = 256,
    channels: int = 1,
    antennas: int = 8,
    frequency_hz: float = 1.4e9,
    channel_width_hz: float = 1e6,
    max_baseline_m: float = 2_000.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> VisibilityDataset:
    if rows < 1 or channels < 1 or antennas < 2:
        raise ValueError("rows/channels must be positive and antennas must be at least two")
    rng = np.random.default_rng(seed)
    antenna1 = rng.integers(0, antennas, rows, dtype=np.int32)
    antenna2 = (antenna1 + rng.integers(1, antennas, rows, dtype=np.int32)) % antennas
    uvw_m = rng.uniform(-max_baseline_m, max_baseline_m, (rows, 3))
    uvw_m[:, 2] *= 0.25
    frequencies = frequency_hz + (
        np.arange(channels) - (channels - 1) / 2
    ) * channel_width_hz
    truth = sources or default_sources(grid)
    correlations = correlations_for_basis(basis)
    visibility = np.asarray(
        predict_stokes_i(
            np.asarray([source.flux for source in truth]),
            np.asarray([source.l for source in truth]),
            np.asarray([source.m for source in truth]),
            uvw_m,
            frequencies,
            antenna1,
            antenna2,
            correlations,
        )
    )
    if noise_std > 0:
        noise = rng.normal(size=visibility.shape) + 1j * rng.normal(size=visibility.shape)
        visibility = visibility + noise_std * noise / np.sqrt(2)
    weight = np.full(visibility.shape, 1 / noise_std**2 if noise_std else 1.0)
    provenance = {
        "generator": "sl1mjax",
        "seed": seed,
        "noise_std": noise_std,
        "grid_size": grid.size,
        "pixel_size_rad": grid.pixel_size_rad,
        "truth": [
            {"flux": source.flux, "l": source.l, "m": source.m} for source in truth
        ],
    }
    block = VisibilityBlock(
        uvw_m=uvw_m,
        frequency_hz=frequencies,
        visibility=visibility,
        weight=weight,
        flag=np.zeros(visibility.shape, dtype=bool),
        time_s=np.arange(rows, dtype=np.float64),
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=basis,
        provenance=provenance,
    )
    return VisibilityDataset((block,), provenance)
