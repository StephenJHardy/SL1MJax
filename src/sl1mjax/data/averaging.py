"""Flag-aware visibility averaging for bounded real-data imaging."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sl1mjax.data.canonical import VisibilityBlock


def average_frequency_bins(
    block: VisibilityBlock, *, bin_count: int
) -> VisibilityBlock:
    """Average contiguous channels with inverse-variance weights."""

    if not 1 <= bin_count <= block.frequency_hz.size:
        raise ValueError("bin_count must be between one and the channel count")
    channel_groups = np.array_split(np.arange(block.frequency_hz.size), bin_count)
    shape = (block.shape[0], bin_count, block.shape[2])
    visibility = np.zeros(shape, dtype=np.complex128)
    weight = np.zeros(shape, dtype=np.float64)
    model = (
        None
        if block.model_visibility is None
        else np.zeros(shape, dtype=np.complex128)
    )
    frequencies = np.zeros(bin_count, dtype=np.float64)
    active_weight = np.where(block.active, block.weight, 0.0)
    for output_channel, channels in enumerate(channel_groups):
        selected_weight = active_weight[:, channels, :]
        summed_weight = np.sum(selected_weight, axis=1)
        weight[:, output_channel, :] = summed_weight
        visibility[:, output_channel, :] = np.divide(
            np.sum(
                selected_weight
                * np.where(
                    block.active[:, channels, :],
                    block.visibility[:, channels, :],
                    0.0,
                ),
                axis=1,
            ),
            summed_weight,
            out=np.zeros_like(summed_weight, dtype=np.complex128),
            where=summed_weight > 0,
        )
        if model is not None and block.model_visibility is not None:
            model[:, output_channel, :] = np.divide(
                np.sum(
                    selected_weight
                    * np.where(
                        block.active[:, channels, :],
                        block.model_visibility[:, channels, :],
                        0.0,
                    ),
                    axis=1,
                ),
                summed_weight,
                out=np.zeros_like(summed_weight, dtype=np.complex128),
                where=summed_weight > 0,
            )
        frequencies[output_channel] = float(
            np.mean(block.frequency_hz[channels])
        )
    return replace(
        block,
        frequency_hz=frequencies,
        visibility=visibility,
        model_visibility=model,
        weight=weight,
        flag=weight <= 0,
        provenance={
            **dict(block.provenance),
            "frequency_averaging": {
                "input_channels": block.frequency_hz.size,
                "output_channels": bin_count,
                "method": "inverse_variance",
            },
        },
    )


def average_time_bins(
    block: VisibilityBlock, *, bin_seconds: float
) -> VisibilityBlock:
    """Average rows within scan/baseline/fixed-width time bins."""

    if not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("bin_seconds must be finite and positive")
    field_id = block.field_id
    scan_id = block.scan_id
    state_id = block.state_id
    observation_id = block.observation_id
    feed1 = block.feed1
    feed2 = block.feed2
    assert field_id is not None
    assert scan_id is not None
    assert state_id is not None
    assert observation_id is not None
    assert feed1 is not None
    assert feed2 is not None
    time_bin = np.floor(block.time_s / bin_seconds).astype(np.int64)
    keys = np.stack(
        (
            field_id,
            scan_id,
            block.antenna1,
            block.antenna2,
            time_bin,
        ),
        axis=1,
    )
    _, first_row, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    group_count = first_row.size
    output_shape = (group_count, block.shape[1], block.shape[2])
    active_weight = np.where(block.active, block.weight, 0.0)
    weight = np.zeros(output_shape, dtype=np.float64)
    weighted_visibility = np.zeros(output_shape, dtype=np.complex128)
    np.add.at(weight, inverse, active_weight)
    np.add.at(
        weighted_visibility,
        inverse,
        active_weight * np.where(block.active, block.visibility, 0.0),
    )
    visibility = np.divide(
        weighted_visibility,
        weight,
        out=np.zeros_like(weighted_visibility),
        where=weight > 0,
    )
    model = None
    if block.model_visibility is not None:
        weighted_model = np.zeros(output_shape, dtype=np.complex128)
        np.add.at(
            weighted_model,
            inverse,
            active_weight
            * np.where(block.active, block.model_visibility, 0.0),
        )
        model = np.divide(
            weighted_model,
            weight,
            out=np.zeros_like(weighted_model),
            where=weight > 0,
        )
    row_weight = np.sum(active_weight, axis=(1, 2))
    group_row_weight = np.zeros(group_count, dtype=np.float64)
    weighted_time = np.zeros(group_count, dtype=np.float64)
    weighted_uvw = np.zeros((group_count, 3), dtype=np.float64)
    np.add.at(group_row_weight, inverse, row_weight)
    np.add.at(weighted_time, inverse, row_weight * block.time_s)
    np.add.at(weighted_uvw, inverse, row_weight[:, None] * block.uvw_m)
    time_s = np.divide(
        weighted_time,
        group_row_weight,
        out=block.time_s[first_row].copy(),
        where=group_row_weight > 0,
    )
    uvw_m = np.divide(
        weighted_uvw,
        group_row_weight[:, None],
        out=block.uvw_m[first_row].copy(),
        where=group_row_weight[:, None] > 0,
    )
    return replace(
        block,
        uvw_m=uvw_m,
        visibility=visibility,
        model_visibility=model,
        weight=weight,
        flag=weight <= 0,
        time_s=time_s,
        antenna1=block.antenna1[first_row],
        antenna2=block.antenna2[first_row],
        field_id=field_id[first_row],
        scan_id=scan_id[first_row],
        state_id=state_id[first_row],
        observation_id=observation_id[first_row],
        feed1=feed1[first_row],
        feed2=feed2[first_row],
        interval_s=np.full(group_count, bin_seconds),
        provenance={
            **dict(block.provenance),
            "time_averaging": {
                "input_rows": block.shape[0],
                "output_rows": group_count,
                "bin_seconds": bin_seconds,
                "method": "inverse_variance",
            },
        },
    )
