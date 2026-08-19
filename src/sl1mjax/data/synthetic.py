"""Deterministic polarized synthetic observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from sl1mjax.calibration import CalibrationSolution, corrupt_model, identity_solution
from sl1mjax.data.canonical import VisibilityBlock, VisibilityDataset
from sl1mjax.polarization import Correlation, ReceptorBasis
from sl1mjax.rime import predict_stokes_i
from sl1mjax.sky import DeltaPixelBasis, PixelBasis, RegularGrid


@dataclass(frozen=True)
class PointSource:
    flux: float
    l: float
    m: float


@dataclass(frozen=True)
class CalibrationSyntheticCase:
    block: VisibilityBlock
    truth: CalibrationSolution


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
    pixel_basis: PixelBasis | None = None,
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
    selected_pixel_basis = pixel_basis or DeltaPixelBasis()
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
            pixel_basis=selected_pixel_basis,
            pixel_size_rad=grid.pixel_size_rad,
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
        "pixel_basis": asdict(selected_pixel_basis),
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


def complete_baseline_schedule(
    antenna_count: int, time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every cross-correlation baseline at every requested time."""

    if antenna_count < 2:
        raise ValueError("at least two antennas are required")
    first, second = np.triu_indices(antenna_count, k=1)
    return (
        np.repeat(np.asarray(time_s, dtype=np.float64), first.size),
        np.tile(first.astype(np.int32), len(time_s)),
        np.tile(second.astype(np.int32), len(time_s)),
    )


def simulate_calibration_case(
    *,
    antenna_count: int = 6,
    time_count: int = 4,
    channel_count: int = 16,
    frequency_hz: float = 4.6e9,
    channel_width_hz: float = 2e6,
    flux_jy: float = 7.5,
    noise_std: float = 0.0,
    flag_fraction: float = 0.0,
    seed: int = 0,
    terms: tuple[str, ...] = ("G", "K", "B"),
) -> CalibrationSyntheticCase:
    """Create deterministic complete-baseline RR/LL calibration truth."""

    unknown_terms = set(terms) - {"G", "K", "B"}
    if unknown_terms:
        raise ValueError(f"unknown calibration terms: {sorted(unknown_terms)}")
    rng = np.random.default_rng(seed)
    solution_times = np.arange(time_count, dtype=np.float64) * 60.0
    row_times, antenna1, antenna2 = complete_baseline_schedule(
        antenna_count, solution_times
    )
    frequencies = frequency_hz + channel_width_hz * (
        np.arange(channel_count) - (channel_count - 1) / 2
    )
    correlations = (Correlation.RR, Correlation.LL)
    truth = identity_solution(
        antenna_count=antenna_count,
        correlations=correlations,
        frequency_hz=frequencies,
        time_s=solution_times,
        reference_antenna=0,
    )
    antenna = np.arange(antenna_count)[None, :, None]
    receptor = np.arange(2)[None, None, :]
    time = np.arange(time_count)[:, None, None]
    if "G" in terms:
        log_amplitude = 0.08 * np.sin(
            0.7 * antenna + 0.5 * receptor + 0.4 * time
        )
        phase = 0.35 * np.sin(0.9 * antenna - 0.3 * receptor + 0.6 * time)
        phase -= phase[:, :1, :]
        gains = np.exp(log_amplitude + 1j * phase)
    else:
        gains = truth.gains
    if "K" in terms:
        delays = (
            3.0e-9
            * np.sin(
                0.8 * np.arange(antenna_count)[:, None]
                + 0.6 * np.arange(2)[None, :]
            )
        )
        delays -= delays[:1, :]
    else:
        delays = truth.delays_s
    if "B" in terms:
        channel = np.linspace(-1.0, 1.0, channel_count)[None, :, None]
        antenna_axis = np.arange(antenna_count)[:, None, None]
        receptor_axis = np.arange(2)[None, None, :]
        log_bandpass = (
            0.05
            * np.sin(
                2.2 * channel + 0.4 * antenna_axis + 0.7 * receptor_axis
            )
        )
        bandpass_phase = (
            0.18
            * channel
            * np.sin(0.5 * antenna_axis + 0.8 * receptor_axis)
        )
        bandpass_phase -= bandpass_phase[:1, :, :]
        bandpass = np.exp(log_bandpass + 1j * bandpass_phase)
        reference_channel = channel_count // 2
        bandpass /= np.abs(bandpass[:, reference_channel : reference_channel + 1])
    else:
        bandpass = truth.bandpass
    truth = CalibrationSolution(
        gains=gains,
        gain_time_s=truth.gain_time_s,
        gain_valid=truth.gain_valid,
        gain_interval_s=truth.gain_interval_s,
        delays_s=delays,
        delay_valid=truth.delay_valid,
        bandpass=bandpass,
        bandpass_frequency_hz=truth.bandpass_frequency_hz,
        bandpass_valid=truth.bandpass_valid,
        correlations=truth.correlations,
        reference_antenna=truth.reference_antenna,
        reference_frequency_hz=float(frequencies[channel_count // 2]),
        provenance={
            "generator": "simulate_calibration_case",
            "seed": seed,
            "terms": list(terms),
        },
    )
    spectrum = flux_jy * (frequencies / truth.reference_frequency_hz) ** -0.7
    model = np.broadcast_to(
        spectrum[None, :, None], (row_times.size, channel_count, 2)
    ).astype(np.complex128)
    corrupted = np.array(
        corrupt_model(
            model,
            truth,
            time_s=row_times,
            frequency_hz=frequencies,
            antenna1=antenna1,
            antenna2=antenna2,
        ),
        copy=True,
    )
    if noise_std:
        corrupted += noise_std / np.sqrt(2) * (
            rng.normal(size=corrupted.shape) + 1j * rng.normal(size=corrupted.shape)
        )
    flag = rng.random(corrupted.shape) < flag_fraction
    block = VisibilityBlock(
        uvw_m=np.zeros((row_times.size, 3)),
        frequency_hz=frequencies,
        visibility=corrupted,
        model_visibility=model,
        weight=np.full(
            corrupted.shape, 1.0 if noise_std == 0 else 1.0 / noise_std**2
        ),
        flag=flag,
        time_s=row_times,
        antenna1=antenna1,
        antenna2=antenna2,
        correlations=correlations,
        receptor_basis=ReceptorBasis.CIRCULAR,
        provenance={"synthetic_calibration": True, "seed": seed},
    )
    return CalibrationSyntheticCase(block, truth)
