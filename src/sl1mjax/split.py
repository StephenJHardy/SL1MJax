"""Structured correlation-aware train/holdout masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock


@dataclass(frozen=True)
class VisibilitySplit:
    train: np.ndarray
    holdout: np.ndarray
    strategy: str


def uv_cell_split(
    block: VisibilityBlock,
    *,
    holdout_fraction: float = 0.2,
    cells_per_axis: int = 8,
    seed: int = 0,
) -> VisibilitySplit:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    uv = block.uvw_m[:, :2]
    extent = np.maximum(np.max(np.abs(uv), axis=0), 1.0)
    cell_xy = np.clip(
        (((uv / extent) + 1) * cells_per_axis / 2).astype(np.int32),
        0,
        cells_per_axis - 1,
    )
    cell_id = cell_xy[:, 0] * cells_per_axis + cell_xy[:, 1]
    occupied = np.unique(cell_id)
    if occupied.size < 2:
        raise ValueError("at least two occupied uv cells are required")
    count = int(np.clip(round(holdout_fraction * occupied.size), 1, occupied.size - 1))
    held_cells = np.random.default_rng(seed).choice(occupied, count, replace=False)
    held_rows = np.isin(cell_id, held_cells)
    holdout = np.broadcast_to(held_rows[:, None, None], block.shape) & block.active
    train = ~holdout & block.active
    return VisibilitySplit(train, holdout, "uv_cell")
