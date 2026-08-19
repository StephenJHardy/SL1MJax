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


def random_row_split(
    block: VisibilityBlock,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> VisibilitySplit:
    """Hold out random complete rows, preserving channel/correlation grouping."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    active_rows = np.flatnonzero(np.any(block.active, axis=(1, 2)))
    if active_rows.size < 2:
        raise ValueError("at least two active rows are required")
    count = int(
        np.clip(
            round(holdout_fraction * active_rows.size),
            1,
            active_rows.size - 1,
        )
    )
    held_rows = np.zeros(block.shape[0], dtype=bool)
    selected = np.random.default_rng(seed).choice(
        active_rows, count, replace=False
    )
    held_rows[selected] = True
    holdout = held_rows[:, None, None] & block.active
    train = ~holdout & block.active
    return VisibilitySplit(train, holdout, "random_row")


def _antenna_graph_connected(
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    rows: np.ndarray,
    expected_antennas: set[int],
) -> bool:
    if not expected_antennas:
        return False
    adjacency: dict[int, set[int]] = {
        antenna: set() for antenna in expected_antennas
    }
    for row in np.flatnonzero(rows):
        first = int(antenna1[row])
        second = int(antenna2[row])
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached = {next(iter(expected_antennas))}
    frontier = list(reached)
    while frontier:
        antenna = frontier.pop()
        for neighbour in adjacency[antenna] - reached:
            reached.add(neighbour)
            frontier.append(neighbour)
    return reached == expected_antennas


def calibration_split(
    block: VisibilityBlock,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> VisibilitySplit:
    """Hold out baseline-time cells without disconnecting a solution interval."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    row_active = np.any(block.active, axis=(1, 2))
    unique_times = np.unique(block.time_s[row_active])
    train_rows = row_active.copy()
    for time in unique_times:
        time_rows = row_active & (block.time_s == time)
        selected = np.flatnonzero(time_rows)
        time_antennas = set(
            np.concatenate(
                (block.antenna1[selected], block.antenna2[selected])
            ).astype(int)
        )
        if not _antenna_graph_connected(
            block.antenna1, block.antenna2, time_rows, time_antennas
        ):
            raise ValueError(
                f"antenna graph is disconnected at solution time {time}"
            )
    groups: list[np.ndarray] = []
    for time in unique_times:
        rows = np.flatnonzero(row_active & (block.time_s == time))
        pairs = np.stack(
            (
                np.minimum(block.antenna1[rows], block.antenna2[rows]),
                np.maximum(block.antenna1[rows], block.antenna2[rows]),
            ),
            axis=1,
        )
        for pair in np.unique(pairs, axis=0):
            groups.append(
                rows[np.all(pairs == pair[None, :], axis=1)]
            )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    target = max(1, round(holdout_fraction * np.sum(row_active)))
    held_count = 0
    for group in groups:
        if held_count >= target:
            break
        candidate = train_rows.copy()
        candidate[group] = False
        time = block.time_s[group[0]]
        time_rows = candidate & (block.time_s == time)
        original_rows = np.flatnonzero(row_active & (block.time_s == time))
        time_antennas = set(
            np.concatenate(
                (
                    block.antenna1[original_rows],
                    block.antenna2[original_rows],
                )
            ).astype(int)
        )
        if _antenna_graph_connected(
            block.antenna1, block.antenna2, time_rows, time_antennas
        ):
            train_rows = candidate
            held_count += group.size
    if held_count == 0:
        raise ValueError("no connected calibration holdout could be constructed")
    holdout_rows = row_active & ~train_rows
    train = np.broadcast_to(train_rows[:, None, None], block.shape) & block.active
    holdout = np.broadcast_to(holdout_rows[:, None, None], block.shape) & block.active
    return VisibilitySplit(train, holdout, "connected_baseline_time")
