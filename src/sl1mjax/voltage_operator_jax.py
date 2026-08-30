"""JAX streamed voltage-Jones operator.

This is the device path that can sit on the optimiser. The NumPy
``predict_voltage_beam`` remains the inspectable reference. Imaging still
defaults to static Airy until a beam mode is enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from sl1mjax.beam import VLABeamCatalog, _airy_voltage
from sl1mjax.beam_conventions import (
    PERLEY2016_MINIMUM_VALID_POWER,
    BeamCalibrationState,
    PerleyFrequencyPolicy,
    beam_requires_identity_on_axis,
    perley2016_frequency_is_supported,
    require_beam_calibration_state,
    select_perley2016_cband_window,
)
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    BeamOperatorPolicy,
    BeamOperatorResult,
    SkyStokesPlanes,
    timestep_jones_bytes,
)
from sl1mjax.calibration_terms import WGS84_A_M, WGS84_E2
from sl1mjax.cassbeam_beam import (
    MAX_NEAREST_NODE_SEPARATION_HZ,
    CassbeamCBandVoltageBeam,
    load_cassbeam_cband_artifact,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.finite_pixel import IntegrationPlan, ManufacturedVoltageBeam
from sl1mjax.objective import weighted_complex_mse
from sl1mjax.polarization import Correlation
from sl1mjax.rime import SPEED_OF_LIGHT_M_S, _square_kernel
from sl1mjax.sky import GaussianApproximation
from sl1mjax.voltage_beam import (
    AnalyticAiryVoltageBeam,
    CompositeHandoverPolicy,
    CompositeScalarVoltageBeam,
    Perley2016CBandVoltageBeam,
    VoltageBeamModel,
)


def parallactic_angle_rad_jax(
    time_s: Array,
    phase_centre_rad: tuple[float, float],
    antenna_position_m: Array,
) -> Array:
    """Alt-az parallactic angle, same GMST as ``calibration_terms``."""

    times = jnp.asarray(time_s, dtype=jnp.float64)
    position = jnp.asarray(antenna_position_m, dtype=jnp.float64)
    equatorial = jnp.hypot(position[:, 0], position[:, 1])
    b_m = WGS84_A_M * jnp.sqrt(1.0 - WGS84_E2)
    e_prime2 = WGS84_E2 / (1.0 - WGS84_E2)
    theta = jnp.arctan2(position[:, 2] * WGS84_A_M, equatorial * b_m)
    latitude = jnp.arctan2(
        position[:, 2] + e_prime2 * b_m * jnp.sin(theta) ** 3,
        equatorial - WGS84_E2 * WGS84_A_M * jnp.cos(theta) ** 3,
    )
    longitude = jnp.arctan2(position[:, 1], position[:, 0])
    julian_date = times / 86400.0 + 2_400_000.5
    gmst = jnp.deg2rad(
        jnp.mod(280.46061837 + 360.98564736629 * (julian_date - 2_451_545.0), 360.0)
    )
    right_ascension, declination = phase_centre_rad
    hour_angle = gmst[:, None] + longitude[None, :] - right_ascension
    numerator = jnp.cos(latitude)[None, :] * jnp.sin(hour_angle)
    denominator = jnp.sin(latitude)[None, :] * jnp.cos(declination) - jnp.cos(latitude)[
        None, :
    ] * jnp.sin(declination) * jnp.cos(hour_angle)
    return jnp.arctan2(numerator, denominator)


def _j1(x: Array) -> Array:
    half = 0.5 * x
    half_squared = half * half
    term = jnp.ones_like(x)
    total = jnp.ones_like(x)
    for order in range(1, 28):
        term = term * (-half_squared) / (order * (order + 1.0))
        total = total + term
    return half * total


def _jinc(x: Array) -> Array:
    return jnp.where(jnp.abs(x) >= 1.0e-12, 2.0 * _j1(x) / x, 1.0)


def airy_voltage_jax(
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    catalog: VLABeamCatalog,
) -> tuple[Array, Array]:
    """Blocked-Airy voltage and visibility mask, matching ``_airy_voltage``."""

    l_grid, m_grid, frequency = jnp.broadcast_arrays(l_rad, m_rad, frequency_hz)
    radius = jnp.hypot(l_grid, m_grid)
    visible = (
        jnp.isfinite(l_grid)
        & jnp.isfinite(m_grid)
        & jnp.isfinite(frequency)
        & (frequency > 0.0)
        & (radius < 1.0)
    )
    l_safe = jnp.where(visible, l_grid, 0.0)
    m_safe = jnp.where(visible, m_grid, 0.0)
    angular = jnp.arcsin(jnp.hypot(l_safe, m_safe))
    wavelength = SPEED_OF_LIGHT_M_S / jnp.where(frequency > 0.0, frequency, 1.0)
    argument = jnp.pi * catalog.dish_diameter_m * jnp.sin(angular) / wavelength
    blockage_ratio = catalog.blockage_diameter_m / catalog.dish_diameter_m
    voltage = (
        _jinc(argument) - blockage_ratio**2 * _jinc(blockage_ratio * argument)
    ) / (1.0 - blockage_ratio**2)
    max_radius = catalog.airy_max_radius_rad_at_1ghz * (
        catalog.squint_reference_hz / jnp.where(frequency > 0.0, frequency, 1.0)
    )
    inside = visible & (angular <= max_radius)
    return jnp.where(inside, voltage, 0.0), inside


def _diagonal_jones(voltage: Array) -> Array:
    zeros = jnp.zeros_like(voltage)
    return jnp.stack(
        (
            jnp.stack((voltage, zeros), axis=-1),
            jnp.stack((zeros, voltage), axis=-1),
        ),
        axis=-2,
    )


def airy_jones_jax(
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    catalog: VLABeamCatalog,
) -> tuple[Array, Array]:
    """Return ``(1, direction, channel, 2, 2)`` Airy Jones."""

    voltage, valid = airy_voltage_jax(
        l_rad[:, None], m_rad[:, None], frequency_hz[None, :], catalog
    )
    return _diagonal_jones(voltage)[None, ...], valid[None, ...]


def _perley_channel_polynomials(frequency_hz: np.ndarray) -> dict[str, np.ndarray]:
    a2 = np.zeros(frequency_hz.size, dtype=np.float64)
    a4 = np.zeros(frequency_hz.size, dtype=np.float64)
    a6 = np.zeros(frequency_hz.size, dtype=np.float64)
    support = np.zeros(frequency_hz.size, dtype=np.float64)
    in_band = np.asarray(perley2016_frequency_is_supported(frequency_hz), dtype=bool)
    scales = np.full(frequency_hz.size, np.nan, dtype=np.float64)
    catalog = VLABeamCatalog()
    for channel, frequency in enumerate(frequency_hz):
        if not in_band[channel]:
            continue
        window = select_perley2016_cband_window(
            float(frequency), policy=PerleyFrequencyPolicy.CASA_NEAREST
        )
        a2[channel] = window.a2
        a4[channel] = window.a4
        a6[channel] = window.a6
        support[channel] = window.support_radius_arcmin(float(frequency))
        edge_l = np.sin(np.deg2rad(support[channel] / 60.0))
        voltage = _airy_voltage(
            np.asarray([edge_l]),
            np.asarray([0.0]),
            np.asarray([frequency]),
            catalog,
        )
        airy_power = float(np.square(voltage[0]))
        if np.hypot(edge_l, 0.0) < 1.0 and airy_power > 1.0e-12:
            scales[channel] = float(np.sqrt(PERLEY2016_MINIMUM_VALID_POWER / airy_power))
    return {
        "a2": a2,
        "a4": a4,
        "a6": a6,
        "support_arcmin": support,
        "in_band": in_band,
        "airy_match_scale": scales,
    }


def composite_jones_jax(
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    polynomials: dict[str, Array],
    airy_catalog: VLABeamCatalog,
) -> tuple[Array, Array]:
    """Perley main lobe plus Airy outer field, one antenna plane."""

    airy, airy_ok = airy_voltage_jax(
        l_rad[:, None], m_rad[:, None], frequency_hz[None, :], airy_catalog
    )
    radius = jnp.hypot(l_rad[:, None], m_rad[:, None])
    offset_arcmin = jnp.rad2deg(jnp.arcsin(jnp.minimum(radius, 1.0))) * 60.0
    freq = frequency_hz[None, :]
    radius_poly = offset_arcmin * (freq / 1.0e9)
    radius_sq = jnp.square(radius_poly)
    power = (
        1.0
        + polynomials["a2"][None, :] * radius_sq
        + polynomials["a4"][None, :] * jnp.square(radius_sq)
        + polynomials["a6"][None, :] * radius_sq**3
    )
    perley_ok = (
        polynomials["in_band"][None, :]
        & (radius < 1.0)
        & (offset_arcmin >= 0.0)
        & (offset_arcmin <= polynomials["support_arcmin"][None, :])
        & jnp.isfinite(power)
        & (power > 0.0)
    )
    perley = jnp.where(perley_ok, jnp.sqrt(jnp.maximum(power, 0.0)), 0.0)
    scale = polynomials["airy_match_scale"][None, :]
    usable_scale = jnp.isfinite(scale) & (scale > 0.0)
    outer_ok = (
        polynomials["in_band"][None, :] & ~perley_ok & airy_ok & usable_scale
    )
    voltage = jnp.where(perley_ok, perley, 0.0)
    voltage = jnp.where(outer_ok, airy * scale, voltage)
    valid = perley_ok | outer_ok
    return _diagonal_jones(voltage)[None, ...], valid[None, ...]


@lru_cache(maxsize=1)
def _cassbeam_tables() -> dict[str, Any]:
    artifact = load_cassbeam_cband_artifact()
    return {
        "jones": jnp.stack(
            [jnp.asarray(table.jones, dtype=jnp.complex128) for table in artifact.tables]
        ),
        "l_axis": jnp.stack(
            [jnp.asarray(table.l_rad, dtype=jnp.float64) for table in artifact.tables]
        ),
        "m_axis": jnp.stack(
            [jnp.asarray(table.m_rad, dtype=jnp.float64) for table in artifact.tables]
        ),
        "frequency_hz": jnp.asarray(
            [table.frequency_hz for table in artifact.tables], dtype=jnp.float64
        ),
        "l_origin": jnp.asarray(
            [table.l_origin_index for table in artifact.tables], dtype=jnp.int32
        ),
        "m_origin": jnp.asarray(
            [table.m_origin_index for table in artifact.tables], dtype=jnp.int32
        ),
    }


def _bilinear_plane(
    jones: Array, l_axis: Array, m_axis: Array, l_rad: Array, m_rad: Array
) -> tuple[Array, Array]:
    i = jnp.interp(l_rad, l_axis, jnp.arange(l_axis.size, dtype=l_axis.dtype))
    j = jnp.interp(m_rad, m_axis, jnp.arange(m_axis.size, dtype=m_axis.dtype))
    ok = (
        (l_rad >= l_axis[0])
        & (l_rad <= l_axis[-1])
        & (m_rad >= m_axis[0])
        & (m_rad <= m_axis[-1])
        & jnp.isfinite(i)
        & jnp.isfinite(j)
    )
    i0 = jnp.clip(jnp.floor(jnp.where(ok, i, 0.0)).astype(jnp.int32), 0, l_axis.size - 2)
    j0 = jnp.clip(jnp.floor(jnp.where(ok, j, 0.0)).astype(jnp.int32), 0, m_axis.size - 2)
    di = jnp.where(ok, i - i0, 0.0)
    dj = jnp.where(ok, j - j0, 0.0)
    g00 = jones[j0, i0]
    g10 = jones[j0, i0 + 1]
    g01 = jones[j0 + 1, i0]
    g11 = jones[j0 + 1, i0 + 1]
    plane = (1.0 - dj)[..., None, None] * (
        (1.0 - di)[..., None, None] * g00 + di[..., None, None] * g10
    ) + dj[..., None, None] * (
        (1.0 - di)[..., None, None] * g01 + di[..., None, None] * g11
    )
    return plane, ok


def cassbeam_jones_jax(
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    chi: Array,
    *,
    tables: dict[str, Array],
    off_diagonal: bool,
    calibration_state: BeamCalibrationState,
    outer_jones: Array | None,
    outer_valid: Array | None,
) -> tuple[Array, Array, Array]:
    """CASSBEAM Jones with optional tapered scalar outer field."""

    cosine = jnp.cos(chi)[:, None]
    sine = jnp.sin(chi)[:, None]
    l_ant = l_rad[None, :] * cosine + m_rad[None, :] * sine
    m_ant = -l_rad[None, :] * sine + m_rad[None, :] * cosine
    table_index = jnp.argmin(
        jnp.abs(tables["frequency_hz"][:, None] - frequency_hz[None, :]), axis=0
    )

    planes = []
    oks = []
    for index in range(int(tables["frequency_hz"].shape[0])):
        plane, ok = _bilinear_plane(
            tables["jones"][index],
            tables["l_axis"][index],
            tables["m_axis"][index],
            l_ant,
            m_ant,
        )
        center = tables["jones"][index][
            tables["m_origin"][index], tables["l_origin"][index]
        ]
        if not off_diagonal:
            diagonal = jnp.zeros_like(plane)
            diagonal = diagonal.at[..., 0, 0].set(plane[..., 0, 0])
            diagonal = diagonal.at[..., 1, 1].set(plane[..., 1, 1])
            plane = diagonal
        if beam_requires_identity_on_axis(calibration_state):
            if off_diagonal:
                plane = jnp.einsum("ij,...jk->...ik", jnp.linalg.inv(center), plane)
            else:
                scaled = jnp.zeros_like(plane)
                scaled = scaled.at[..., 0, 0].set(plane[..., 0, 0] / center[0, 0])
                scaled = scaled.at[..., 1, 1].set(plane[..., 1, 1] / center[1, 1])
                plane = scaled
        planes.append(plane)
        oks.append(ok)
    stacked = jnp.stack(planes, axis=0)
    stacked_ok = jnp.stack(oks, axis=0)
    plane = stacked[table_index]
    cassbeam_ok = stacked_ok[table_index]
    plane = jnp.transpose(plane, (1, 2, 0, 3, 4))
    cassbeam_ok = jnp.transpose(cassbeam_ok, (1, 2, 0))
    rotation = jnp.exp(-1j * chi)
    p_jones = jnp.zeros((chi.size, 2, 2), dtype=plane.dtype)
    p_jones = p_jones.at[:, 0, 0].set(rotation)
    p_jones = p_jones.at[:, 1, 1].set(jnp.conjugate(rotation))
    if calibration_state is BeamCalibrationState.CASA_PARANG_TRUE:
        p_h = jnp.conjugate(jnp.swapaxes(p_jones, -1, -2))
        plane = jnp.einsum("aij,adcjk,akl->adcil", p_h, plane, p_jones)
    elif calibration_state is BeamCalibrationState.UNCALIBRATED:
        plane = jnp.einsum("adcij,ajk->adcik", plane, p_jones)
    else:
        raise ValueError(f"unsupported beam calibration state {calibration_state!r}")
    valid = cassbeam_ok
    off_diagonal_valid = cassbeam_ok
    if outer_jones is not None and outer_valid is not None:
        scalar = jnp.broadcast_to(outer_jones, plane.shape)
        scalar_ok = jnp.broadcast_to(outer_valid, valid.shape)
        weight = valid.astype(plane.real.dtype)
        blended = weight[..., None, None] * plane + (1.0 - weight[..., None, None]) * scalar
        outside = ~valid
        blended = blended.at[..., 0, 1].set(jnp.where(outside, 0.0, blended[..., 0, 1]))
        blended = blended.at[..., 1, 0].set(jnp.where(outside, 0.0, blended[..., 1, 0]))
        usable = valid | scalar_ok
        plane = jnp.where(usable[..., None, None], blended, 0.0)
        valid = usable
        off_diagonal_valid = cassbeam_ok
    return plane, valid, off_diagonal_valid


def _circular_coherency(
    stokes_i: Array, stokes_q: Array, stokes_u: Array, stokes_v: Array
) -> Array:
    return jnp.stack(
        (
            jnp.stack((stokes_i + stokes_v, stokes_q + 1j * stokes_u), axis=-1),
            jnp.stack((stokes_q - 1j * stokes_u, stokes_i - stokes_v), axis=-1),
        ),
        axis=-2,
    )


def _mask_apparent_coherency(apparent: Array, ok: Array, off_ok: Array) -> Array:
    masked = jnp.where(ok[..., None, None], apparent, 0.0)
    rr = masked[..., 0, 0]
    rl = jnp.where(off_ok, masked[..., 0, 1], 0.0)
    lr = jnp.where(off_ok, masked[..., 1, 0], 0.0)
    ll = masked[..., 1, 1]
    return jnp.stack(
        (jnp.stack((rr, rl), axis=-1), jnp.stack((lr, ll), axis=-1)),
        axis=-2,
    )


def _unpack_correlations(coherency: Array, correlations: tuple[Correlation, ...]) -> Array:
    slots = {
        Correlation.RR: (0, 0),
        Correlation.RL: (0, 1),
        Correlation.LR: (1, 0),
        Correlation.LL: (1, 1),
    }
    return jnp.stack(
        [coherency[..., first, second] for first, second in (slots[item] for item in correlations)],
        axis=-1,
    )


def _delta_kernel(uvw_wavelengths: Array, l_rad: Array, m_rad: Array) -> Array:
    n_rad = jnp.sqrt(jnp.maximum(1.0 - l_rad * l_rad - m_rad * m_rad, 0.0))
    phase = 2j * jnp.pi * (
        uvw_wavelengths[:, 0, None] * l_rad[None, :]
        + uvw_wavelengths[:, 1, None] * m_rad[None, :]
        + uvw_wavelengths[:, 2, None] * (n_rad[None, :] - 1.0)
    )
    return jnp.exp(phase)


def _accumulate_rows(
    uvw_m: Array,
    frequency_hz: Array,
    antenna1: Array,
    antenna2: Array,
    l_rad: Array,
    m_rad: Array,
    coherency: Array,
    correlations: tuple[Correlation, ...],
    visibility_chunk_size: int,
    pixel_chunk_size: int,
    jones_for_tile: Callable[[Array, Array], tuple[Array, Array, Array]],
    width_rad: Array | None = None,
    node_valid: Array | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
) -> tuple[Array, Array, Array]:
    n_row = uvw_m.shape[0]
    n_channel = frequency_hz.size
    n_corr = len(correlations)
    n_dir = l_rad.size
    widths = (
        jnp.zeros((n_dir,), dtype=jnp.float64)
        if width_rad is None
        else jnp.asarray(width_rad, dtype=jnp.float64).reshape(-1)
    )
    valid_nodes = (
        jnp.ones((n_dir,), dtype=bool)
        if node_valid is None
        else jnp.asarray(node_valid, dtype=bool).reshape(-1)
    )
    use_square = width_rad is not None
    empty_vis = jnp.zeros((n_row, n_channel, n_corr), dtype=jnp.complex128)
    empty_ok = jnp.zeros((n_row, n_channel), dtype=jnp.int32)

    def visibility_body(
        row_start: int, carry: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        pred, copolar_ok, off_ok_count, mixed_count = carry
        rows = jnp.arange(visibility_chunk_size) + row_start
        in_row = rows < n_row
        safe_rows = jnp.where(in_row, rows, 0)
        uvw = uvw_m[safe_rows]
        a1 = antenna1[safe_rows]
        a2 = antenna2[safe_rows]

        def pixel_body(
            pixel_start: int, acc: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array]:
            vis, copolar, off_acc, mixed_acc = acc
            pixels = jnp.arange(pixel_chunk_size) + pixel_start
            in_pix = pixels < n_dir
            safe_pix = jnp.where(in_pix, pixels, 0)
            l_tile = l_rad[safe_pix]
            m_tile = m_rad[safe_pix]
            w_tile = widths[safe_pix]
            node_ok = valid_nodes[safe_pix]
            sky = coherency[safe_pix]
            jones, valid, off_valid = jones_for_tile(l_tile, m_tile)
            single_plane = jones.shape[0] == 1
            if single_plane:
                jp_tile = jones[0]
                jq_tile = jones[0]
                j_ok_p = valid[0]
                j_ok_q = valid[0]
                j_off_p = off_valid[0]
                j_off_q = off_valid[0]
                ok = j_ok_p & j_ok_q & in_pix[:, None] & node_ok[:, None]
                off_ok = j_off_p & j_off_q & in_pix[:, None] & node_ok[:, None]
            else:
                jp_tile = jones[a1]
                jq_tile = jones[a2]
                j_ok_p = valid[a1]
                j_ok_q = valid[a2]
                j_off_p = off_valid[a1]
                j_off_q = off_valid[a2]
                ok = j_ok_p & j_ok_q & in_pix[None, :, None] & node_ok[None, :, None]
                off_ok = (
                    j_off_p & j_off_q & in_pix[None, :, None] & node_ok[None, :, None]
                )

            def channel_body(channel: int, vis_acc: Array) -> Array:
                uvw_l = uvw * (frequency_hz[channel] / SPEED_OF_LIGHT_M_S)
                if use_square:
                    kernel = _square_kernel(
                        uvw_l,
                        l_tile,
                        m_tile,
                        w_tile,
                        approximation,
                        include_projection=False,
                    )
                else:
                    kernel = _delta_kernel(uvw_l, l_tile, m_tile)
                kernel = jnp.where(
                    in_row[:, None] & in_pix[None, :] & node_ok[None, :],
                    kernel,
                    0.0,
                )
                if single_plane:
                    apparent = jnp.einsum(
                        "dij,djk,dlk->dil",
                        jp_tile[:, channel],
                        sky[:, channel],
                        jnp.conjugate(jq_tile[:, channel]),
                    )
                    apparent = _mask_apparent_coherency(
                        apparent, ok[:, channel], off_ok[:, channel]
                    )
                    packed = _unpack_correlations(apparent, correlations)
                    return vis_acc.at[:, channel].add(
                        jnp.einsum("rd,dc->rc", kernel, packed)
                    )
                apparent = jnp.einsum(
                    "rdij,djk,rdlk->rdil",
                    jp_tile[:, :, channel],
                    sky[:, channel],
                    jnp.conjugate(jq_tile[:, :, channel]),
                )
                apparent = _mask_apparent_coherency(
                    apparent, ok[:, :, channel], off_ok[:, :, channel]
                )
                packed = _unpack_correlations(apparent, correlations)
                return vis_acc.at[:, channel].add(
                    jnp.einsum("rd,rdc->rc", kernel, packed)
                )

            vis = jax.lax.fori_loop(0, n_channel, channel_body, vis)
            if single_plane:
                tile_copolar = jnp.any(ok, axis=0)[None, :] & in_row[:, None]
                tile_off = jnp.any(off_ok, axis=0)[None, :] & in_row[:, None]
                tile_mixed = jnp.any(ok & ~off_ok, axis=0)[None, :] & in_row[:, None]
            else:
                tile_copolar = jnp.any(ok, axis=1) & in_row[:, None]
                tile_off = jnp.any(off_ok, axis=1) & in_row[:, None]
                tile_mixed = jnp.any(ok & ~off_ok, axis=1) & in_row[:, None]
            return (
                vis,
                copolar | tile_copolar,
                off_acc | tile_off,
                mixed_acc | tile_mixed,
            )

        tile_vis, tile_copolar, tile_off, tile_mixed = jax.lax.fori_loop(
            0,
            (n_dir + pixel_chunk_size - 1) // pixel_chunk_size,
            lambda index, acc: pixel_body(index * pixel_chunk_size, acc),
            (
                jnp.zeros((visibility_chunk_size, n_channel, n_corr), dtype=jnp.complex128),
                jnp.zeros((visibility_chunk_size, n_channel), dtype=bool),
                jnp.zeros((visibility_chunk_size, n_channel), dtype=bool),
                jnp.zeros((visibility_chunk_size, n_channel), dtype=bool),
            ),
        )
        tile_vis = jnp.where(in_row[:, None, None], tile_vis, 0.0)
        tile_copolar = tile_copolar & in_row[:, None]
        tile_off = tile_off & in_row[:, None]
        tile_mixed = tile_mixed & in_row[:, None]
        return (
            pred.at[safe_rows].add(tile_vis),
            copolar_ok.at[safe_rows].add(tile_copolar.astype(jnp.int32)),
            off_ok_count.at[safe_rows].add(tile_off.astype(jnp.int32)),
            mixed_count.at[safe_rows].add(tile_mixed.astype(jnp.int32)),
        )

    n_row_tiles = (n_row + visibility_chunk_size - 1) // visibility_chunk_size
    prediction, copolar_count, off_count, mixed_count = jax.lax.fori_loop(
        0,
        n_row_tiles,
        lambda index, carry: visibility_body(index * visibility_chunk_size, carry),
        (empty_vis, empty_ok, empty_ok, empty_ok),
    )
    return prediction, copolar_count > 0, (off_count > 0) & (mixed_count == 0)


def _sky_plane(
    values: ArrayLike | None,
    n_dir: int,
    n_channel: int,
    name: str,
    *,
    optional: bool = False,
) -> Array:
    if values is None:
        if not optional:
            raise ValueError(f"{name} is required")
        return jnp.zeros((n_dir, n_channel), dtype=jnp.float64)
    array = jnp.asarray(values, dtype=jnp.float64)
    if array.ndim == 1:
        if array.size != n_dir:
            raise ValueError(f"{name} must match the direction axis")
        return jnp.broadcast_to(array[:, None], (n_dir, n_channel))
    if array.shape != (n_dir, n_channel):
        raise ValueError(
            f"{name} must have shape ({n_dir},) or ({n_dir}, {n_channel})"
        )
    return array


def _prepare_coherency(
    sky: SkyStokesPlanes, n_dir: int, n_channel: int
) -> Array:
    intensity = _sky_plane(sky.stokes_i, n_dir, n_channel, "stokes_i")
    stokes_q = _sky_plane(sky.stokes_q, n_dir, n_channel, "stokes_q", optional=True)
    stokes_u = _sky_plane(sky.stokes_u, n_dir, n_channel, "stokes_u", optional=True)
    stokes_v = _sky_plane(sky.stokes_v, n_dir, n_channel, "stokes_v", optional=True)
    return _circular_coherency(intensity, stokes_q, stokes_u, stokes_v)


def _evaluate_beam_jones(
    beam: VoltageBeamModel,
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    chi: Array,
    calibration_state: BeamCalibrationState,
    polynomials: dict[str, Array] | None,
    cassbeam_tables: dict[str, Array] | None,
) -> tuple[Array, Array, Array]:
    if isinstance(beam, AnalyticAiryVoltageBeam):
        jones, valid = airy_jones_jax(l_rad, m_rad, frequency_hz, beam.catalog)
        return jones, valid, valid
    if isinstance(beam, CompositeScalarVoltageBeam):
        if beam.handover is not CompositeHandoverPolicy.MATCH_POWER:
            raise ValueError("JAX composite Jones only implements match_power handover")
        if polynomials is None:
            raise ValueError("composite Jones needs prepared Perley polynomials")
        jones, valid = composite_jones_jax(
            l_rad, m_rad, frequency_hz, polynomials, beam.outer.catalog
        )
        return jones, valid, valid
    if isinstance(beam, CassbeamCBandVoltageBeam):
        outer_jones = None
        outer_valid = None
        if beam.outer is not None:
            if isinstance(beam.outer, CompositeScalarVoltageBeam):
                if beam.outer.handover is not CompositeHandoverPolicy.MATCH_POWER:
                    raise ValueError(
                        "JAX CASSBEAM outer composite only implements match_power"
                    )
                if polynomials is None:
                    raise ValueError("CASSBEAM outer composite needs Perley polynomials")
                outer_jones, outer_valid = composite_jones_jax(
                    l_rad,
                    m_rad,
                    frequency_hz,
                    polynomials,
                    beam.outer.outer.catalog,
                )
            elif isinstance(beam.outer, AnalyticAiryVoltageBeam):
                outer_jones, outer_valid = airy_jones_jax(
                    l_rad, m_rad, frequency_hz, beam.outer.catalog
                )
            else:
                raise TypeError(f"unsupported CASSBEAM outer beam {type(beam.outer)!r}")
        if cassbeam_tables is None:
            raise ValueError("CASSBEAM Jones needs prepared tables")
        jones, valid, off_valid = cassbeam_jones_jax(
            l_rad,
            m_rad,
            frequency_hz,
            chi,
            tables=cassbeam_tables,
            off_diagonal=beam.off_diagonal,
            calibration_state=calibration_state,
            outer_jones=outer_jones,
            outer_valid=outer_valid,
        )
        if beam.off_diagonal:
            return jones, valid, off_valid
        return jones, valid, valid
    if isinstance(beam, ManufacturedVoltageBeam):
        return manufactured_jones_jax(
            beam, l_rad, m_rad, frequency_hz, chi, calibration_state
        )
    if isinstance(beam, Perley2016CBandVoltageBeam):
        if polynomials is None:
            raise ValueError("Perley Jones needs prepared polynomials")
        airy_catalog = VLABeamCatalog(airy_max_radius_rad_at_1ghz=np.deg2rad(0.0))
        jones, valid = composite_jones_jax(
            l_rad, m_rad, frequency_hz, polynomials, airy_catalog
        )
        return jones, valid, valid
    raise TypeError(f"no JAX evaluator for {type(beam)!r}")


def _time_groups(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(time_s, kind="stable")
    sorted_times = time_s[order]
    unique, starts, counts = np.unique(sorted_times, return_index=True, return_counts=True)
    max_count = int(np.max(counts))
    indices = np.zeros((unique.size, max_count), dtype=np.int32)
    valid = np.zeros((unique.size, max_count), dtype=bool)
    for index, (start, count) in enumerate(zip(starts, counts, strict=True)):
        indices[index, :count] = order[start : start + count]
        valid[index, :count] = True
    return unique, indices, valid


def _predict_voltage_beam_jax_arrays(
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    sky: SkyStokesPlanes,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    width_rad: ArrayLike | None = None,
    node_valid: ArrayLike | None = None,
    kernel_approximation: GaussianApproximation | str = GaussianApproximation.WIDE_FIELD,
) -> tuple[Array, Array, Array]:
    selected = config or BeamOperatorConfig()
    state = require_beam_calibration_state(calibration_state)
    approximation = GaussianApproximation(kernel_approximation)
    if selected.policy is not BeamOperatorPolicy.STREAM:
        raise ValueError("JAX voltage operator only implements STREAM policy")
    if not block.correlations:
        raise ValueError("block must contain circular correlations")
    unknown = set(block.correlations) - {
        Correlation.RR,
        Correlation.RL,
        Correlation.LR,
        Correlation.LL,
    }
    if unknown:
        raise ValueError(f"unsupported circular correlations {sorted(unknown)}")
    l_array = jnp.asarray(l_rad, dtype=jnp.float64).reshape(-1)
    m_array = jnp.asarray(m_rad, dtype=jnp.float64).reshape(-1)
    if l_array.size != m_array.size or l_array.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    frequency = jnp.asarray(block.frequency_hz, dtype=jnp.float64)
    positions = jnp.asarray(antenna_position_m, dtype=jnp.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("antenna_position_m must have shape (antenna, 3)")
    if positions.shape[0] < block.antenna_count:
        raise ValueError("antenna_position_m must cover every antenna in the block")
    depends_on_time = bool(getattr(beam, "antenna_planes_from_parallactic", False))
    plane_count = int(positions.shape[0]) if depends_on_time else 1
    tile_dirs = min(int(selected.pixel_chunk_size), int(l_array.size))
    jones_bytes = timestep_jones_bytes(plane_count, tile_dirs, int(frequency.size))
    if jones_bytes > selected.max_timestep_jones_bytes:
        raise ValueError(
            "Jones tile "
            f"({plane_count} antennas × {tile_dirs} directions × "
            f"{int(frequency.size)} channels) needs {jones_bytes} bytes; "
            f"max_timestep_jones_bytes is {selected.max_timestep_jones_bytes}. "
            "Reduce pixel_chunk_size."
        )
    coherency = _prepare_coherency(sky, int(l_array.size), int(frequency.size))
    polynomials = None
    cassbeam_tables = None
    if isinstance(beam, (CompositeScalarVoltageBeam, CassbeamCBandVoltageBeam)):
        polynomials = {
            key: jnp.asarray(value)
            for key, value in _perley_channel_polynomials(np.asarray(block.frequency_hz)).items()
        }
    if isinstance(beam, CassbeamCBandVoltageBeam):
        cassbeam_tables = _cassbeam_tables()
        table_hz = np.asarray(cassbeam_tables["frequency_hz"])
        freq_hz = np.asarray(block.frequency_hz, dtype=np.float64)
        separation = np.min(np.abs(table_hz[:, None] - freq_hz[None, :]), axis=0)
        if np.any(separation > MAX_NEAREST_NODE_SEPARATION_HZ):
            raise ValueError(
                "frequency is farther than "
                f"{MAX_NEAREST_NODE_SEPARATION_HZ / 1e6:.0f} MHz from a CASSBEAM node"
            )
    unique_times, row_index, row_valid = _time_groups(np.asarray(block.time_s, dtype=np.float64))
    unique = jnp.asarray(unique_times, dtype=jnp.float64)
    row_index_j = jnp.asarray(row_index, dtype=jnp.int32)
    row_valid_j = jnp.asarray(row_valid, dtype=bool)
    uvw = jnp.asarray(block.uvw_m, dtype=jnp.float64)
    antenna1 = jnp.asarray(block.antenna1, dtype=jnp.int32)
    antenna2 = jnp.asarray(block.antenna2, dtype=jnp.int32)
    correlations = block.correlations

    def time_body(
        carry: tuple[Array, Array, Array], time_index: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        prediction, copolar_valid, leakage_valid = carry
        time_s = unique[time_index]
        chi = parallactic_angle_rad_jax(
            time_s[None], block.phase_centre_rad, positions
        )[0]
        if not depends_on_time:
            chi = jnp.zeros((1,), dtype=jnp.float64)

        def jones_for_tile(l_tile: Array, m_tile: Array) -> tuple[Array, Array, Array]:
            if selected.pointing_offset_lm_rad is None:
                l_eval, m_eval = l_tile, m_tile
            else:
                l_eval = l_tile - jnp.asarray(
                    selected.pointing_offset_lm_rad[0], dtype=jnp.float64
                )
                m_eval = m_tile - jnp.asarray(
                    selected.pointing_offset_lm_rad[1], dtype=jnp.float64
                )
            return _evaluate_beam_jones(
                beam,
                l_eval,
                m_eval,
                frequency,
                chi,
                state,
                polynomials,
                cassbeam_tables,
            )

        rows = row_index_j[time_index]
        ok = row_valid_j[time_index]
        safe_rows = jnp.where(ok, rows, 0)
        contrib, tile_copolar, tile_leakage = _accumulate_rows(
            uvw[safe_rows],
            frequency,
            antenna1[safe_rows],
            antenna2[safe_rows],
            l_array,
            m_array,
            coherency,
            correlations,
            selected.visibility_chunk_size,
            selected.pixel_chunk_size,
            jones_for_tile,
            width_rad=(
                None
                if width_rad is None
                else jnp.asarray(width_rad, dtype=jnp.float64).reshape(-1)
            ),
            node_valid=(
                None
                if node_valid is None
                else jnp.asarray(node_valid, dtype=bool).reshape(-1)
            ),
            approximation=approximation,
        )
        contrib = jnp.where(ok[:, None, None], contrib, 0.0)
        tile_copolar = tile_copolar & ok[:, None]
        tile_leakage = tile_leakage & ok[:, None]
        return (
            (
                prediction.at[safe_rows].add(contrib),
                copolar_valid.at[safe_rows].add(tile_copolar.astype(jnp.int32)),
                leakage_valid.at[safe_rows].add(tile_leakage.astype(jnp.int32)),
            ),
            None,
        )

    empty_vis = jnp.zeros(
        (block.visibility.shape[0], frequency.size, len(correlations)),
        dtype=jnp.complex128,
    )
    empty_ok = jnp.zeros((block.visibility.shape[0], frequency.size), dtype=jnp.int32)
    (prediction, copolar_count, leakage_count), _ = jax.lax.scan(
        time_body,
        (empty_vis, empty_ok, empty_ok),
        jnp.arange(unique.shape[0], dtype=jnp.int32),
    )
    return prediction, copolar_count > 0, leakage_count > 0


def _correlation_validity(
    copolar_valid: Array,
    leakage_valid: Array,
    correlations: tuple[Correlation, ...],
) -> Array:
    planes = []
    for correlation in correlations:
        if correlation in {Correlation.RL, Correlation.LR}:
            planes.append(leakage_valid)
        else:
            planes.append(copolar_valid)
    return jnp.stack(planes, axis=-1)


def predict_voltage_beam_jax(
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    sky: SkyStokesPlanes,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    width_rad: ArrayLike | None = None,
    node_valid: ArrayLike | None = None,
    kernel_approximation: GaussianApproximation | str = GaussianApproximation.WIDE_FIELD,
) -> BeamOperatorResult:
    """Predict ``E_p C E_q^H`` visibilities with a JAX streamed operator."""

    prediction, copolar_valid, leakage_valid = _predict_voltage_beam_jax_arrays(
        block,
        l_rad,
        m_rad,
        sky,
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        config=config,
        width_rad=width_rad,
        node_valid=node_valid,
        kernel_approximation=kernel_approximation,
    )
    return BeamOperatorResult(
        visibility=np.asarray(prediction),
        valid=np.asarray(copolar_valid),
        provenance={
            "operator": "voltage_operator_jax",
            "beam_model_id": getattr(beam, "model_id", type(beam).__name__),
            "calibration_state": require_beam_calibration_state(calibration_state).value,
        },
        off_diagonal_valid=np.asarray(leakage_valid),
    )


def predict_voltage_beam_jax_value_and_grad(
    intensity: Array,
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    train_mask: ArrayLike | None = None,
    width_rad: ArrayLike | None = None,
    node_valid: ArrayLike | None = None,
    kernel_approximation: GaussianApproximation | str = GaussianApproximation.WIDE_FIELD,
) -> tuple[Array, Array]:
    """Return normalised weighted MSE and its Stokes-I gradient.

    ``train_mask`` defaults to ``block.active``. Cross-hand samples are
    excluded where ``off_diagonal_valid`` is false so an unsupported
    leakage zero is not treated as evidence.
    """

    def _loss(values: Array) -> Array:
        predicted, copolar_valid, leakage_valid = _predict_voltage_beam_jax_arrays(
            block,
            l_rad,
            m_rad,
            SkyStokesPlanes(stokes_i=cast(np.ndarray, values)),
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            width_rad=width_rad,
            node_valid=node_valid,
            kernel_approximation=kernel_approximation,
        )
        sample = jnp.asarray(block.active if train_mask is None else train_mask)
        if sample.shape != block.visibility.shape:
            raise ValueError("train_mask must match block.visibility")
        beam_ok = _correlation_validity(
            copolar_valid, leakage_valid, block.correlations
        )
        return weighted_complex_mse(
            predicted,
            jnp.asarray(block.visibility),
            jnp.asarray(block.weight),
            ~sample | ~beam_ok,
        )

    return jax.value_and_grad(_loss)(intensity)


def off_diagonal_support_mask_jax(
    beam: VoltageBeamModel,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    frequency_hz: ArrayLike,
    chi: ArrayLike,
    *,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
) -> np.ndarray:
    """Return ``(direction,)`` leakage support, tiled to the Jones budget."""

    selected = config or BeamOperatorConfig()
    if selected.policy is not BeamOperatorPolicy.STREAM:
        raise ValueError("JAX voltage operator only implements STREAM policy")
    state = require_beam_calibration_state(calibration_state)
    l_array = np.asarray(l_rad, dtype=np.float64).reshape(-1)
    m_array = np.asarray(m_rad, dtype=np.float64).reshape(-1)
    if l_array.size != m_array.size or l_array.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    chi_array = np.asarray(chi, dtype=np.float64).reshape(-1)
    depends_on_time = bool(getattr(beam, "antenna_planes_from_parallactic", False))
    plane_count = int(chi_array.size) if depends_on_time else 1
    tile_dirs = min(int(selected.pixel_chunk_size), int(l_array.size))
    jones_bytes = timestep_jones_bytes(plane_count, tile_dirs, int(frequency.size))
    if jones_bytes > selected.max_timestep_jones_bytes:
        raise ValueError(
            "Jones tile "
            f"({plane_count} antennas × {tile_dirs} directions × "
            f"{int(frequency.size)} channels) needs {jones_bytes} bytes; "
            f"max_timestep_jones_bytes is {selected.max_timestep_jones_bytes}. "
            "Reduce pixel_chunk_size."
        )
    polynomials = None
    cassbeam_tables = None
    if isinstance(beam, (CompositeScalarVoltageBeam, CassbeamCBandVoltageBeam)):
        polynomials = {
            key: jnp.asarray(value)
            for key, value in _perley_channel_polynomials(frequency).items()
        }
    if isinstance(beam, CassbeamCBandVoltageBeam):
        cassbeam_tables = _cassbeam_tables()
        table_hz = np.asarray(cassbeam_tables["frequency_hz"])
        separation = np.min(np.abs(table_hz[:, None] - frequency[None, :]), axis=0)
        if np.any(separation > MAX_NEAREST_NODE_SEPARATION_HZ):
            raise ValueError(
                "frequency is farther than "
                f"{MAX_NEAREST_NODE_SEPARATION_HZ / 1e6:.0f} MHz from a CASSBEAM node"
            )
    chi_j = jnp.asarray(chi_array, dtype=jnp.float64)
    frequency_j = jnp.asarray(frequency, dtype=jnp.float64)
    support = np.ones(l_array.size, dtype=bool)
    for start in range(0, l_array.size, tile_dirs):
        stop = min(start + tile_dirs, l_array.size)
        _jones, _valid, off_valid = _evaluate_beam_jones(
            beam,
            jnp.asarray(l_array[start:stop]),
            jnp.asarray(m_array[start:stop]),
            frequency_j,
            chi_j,
            state,
            polynomials,
            cassbeam_tables,
        )
        support[start:stop] = np.all(np.asarray(off_valid), axis=(0, 2))
    return support


def manufactured_jones_jax(
    beam: ManufacturedVoltageBeam,
    l_rad: Array,
    m_rad: Array,
    frequency_hz: Array,
    chi: Array,
    calibration_state: BeamCalibrationState,
) -> tuple[Array, Array, Array]:
    l_values = jnp.asarray(l_rad, dtype=jnp.float64).reshape(-1)
    m_values = jnp.asarray(m_rad, dtype=jnp.float64).reshape(-1)
    plane = _manufactured_polynomial(beam, l_values, m_values)
    n_channel = int(frequency_hz.size)
    if beam.rotate_parallactic:
        angles = jnp.asarray(chi, dtype=jnp.float64).reshape(-1)
        parallactic = jnp.zeros((angles.size, 2, 2), dtype=jnp.complex128)
        parallactic = parallactic.at[:, 0, 0].set(jnp.exp(-1j * angles))
        parallactic = parallactic.at[:, 1, 1].set(jnp.exp(1j * angles))
        if calibration_state is BeamCalibrationState.CASA_PARANG_TRUE:
            conjugate = jnp.conjugate(jnp.swapaxes(parallactic, -1, -2))
            jones = jnp.einsum("aij,djk,akl->adil", conjugate, plane, parallactic)
        else:
            jones = jnp.einsum("dij,ajk->adik", plane, parallactic)
        n_plane = angles.size
    else:
        jones = plane[None, ...]
        n_plane = 1
    jones = jnp.broadcast_to(
        jones[:, :, None, :, :],
        (n_plane, l_values.size, n_channel, 2, 2),
    )
    valid = jnp.ones(jones.shape[:3], dtype=bool)
    if beam.valid_radius_rad is not None:
        inside = (l_values * l_values + m_values * m_values) <= (
            beam.valid_radius_rad**2
        )
        valid = valid & inside[None, :, None]
    leakage = valid if beam.off_diagonal_valid else jnp.zeros_like(valid)
    if beam.off_diagonal_radius_rad is not None:
        leak_inside = (l_values * l_values + m_values * m_values) <= (
            beam.off_diagonal_radius_rad**2
        )
        leakage = leakage & leak_inside[None, :, None]
    return jones, valid, leakage


def predict_voltage_from_plan_value_and_grad_jax(
    parent_flux: ArrayLike,
    block: VisibilityBlock,
    plan: IntegrationPlan,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    train_mask: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Weighted MSE and Stokes-I gradient over fitted parents, not nodes."""

    parent = jnp.asarray(parent_flux, dtype=jnp.float64).reshape(-1)
    if int(parent.size) != plan.parent_count:
        raise ValueError("parent_flux must match the number of fitted parents")
    parent_index = jnp.asarray(plan.parent_index, dtype=jnp.int32)
    weight = jnp.asarray(plan.weight, dtype=jnp.float64)
    node_ok = jnp.asarray(plan.node_valid, dtype=bool)
    local_l, local_m = plan.local_directions(block.phase_centre_rad)

    def _loss(values: Array) -> Array:
        node_flux = values[parent_index] * weight
        node_flux = jnp.where(node_ok, node_flux, 0.0)
        predicted, copolar_valid, leakage_valid = _predict_voltage_beam_jax_arrays(
            block,
            local_l,
            local_m,
            SkyStokesPlanes(stokes_i=cast(np.ndarray, node_flux)),
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            width_rad=plan.width_rad,
            node_valid=plan.node_valid,
            kernel_approximation=plan.approximation,
        )
        sample = jnp.asarray(block.active if train_mask is None else train_mask)
        if sample.shape != block.visibility.shape:
            raise ValueError("train_mask must match block.visibility")
        beam_ok = _correlation_validity(
            copolar_valid, leakage_valid, block.correlations
        )
        return weighted_complex_mse(
            predicted,
            jnp.asarray(block.visibility),
            jnp.asarray(block.weight),
            ~sample | ~beam_ok,
        )

    return jax.value_and_grad(_loss)(parent)


def _manufactured_polynomial(
    beam: ManufacturedVoltageBeam,
    l_rad: Array,
    m_rad: Array,
) -> Array:
    intercept = _manufactured_coefficient(beam.intercept)
    plane = intercept + _manufactured_coefficient(beam.grad_l) * l_rad[:, None, None]
    plane = plane + _manufactured_coefficient(beam.grad_m) * m_rad[:, None, None]
    plane = plane + _manufactured_coefficient(beam.hess_ll) * (
        l_rad * l_rad
    )[:, None, None]
    plane = plane + _manufactured_coefficient(beam.hess_lm) * (
        l_rad * m_rad
    )[:, None, None]
    plane = plane + _manufactured_coefficient(beam.hess_mm) * (
        m_rad * m_rad
    )[:, None, None]
    return plane


def _manufactured_coefficient(value: Any) -> Array:
    if value is None:
        return jnp.zeros((2, 2), dtype=jnp.complex128)
    return jnp.asarray(value, dtype=jnp.complex128)
