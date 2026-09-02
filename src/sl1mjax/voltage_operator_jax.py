"""JAX streamed voltage-Jones operator.

This is the device path that can sit on the optimiser. The NumPy
``predict_voltage_beam`` remains the inspectable reference. Imaging still
defaults to static Airy until a beam mode is enabled.
"""

from __future__ import annotations

from collections.abc import Callable
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
from sl1mjax.objective import effective_weight, weighted_complex_mse
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


_CASSBEAM_TABLES_HOST: dict[str, Any] | None = None


def _is_tracer(value: object) -> bool:
    return isinstance(value, jax.core.Tracer)


def _build_cassbeam_tables() -> dict[str, Any]:
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
        "frequency_hz_host": np.asarray(
            [table.frequency_hz for table in artifact.tables], dtype=np.float64
        ),
    }


def _ensure_host_cassbeam_tables() -> dict[str, Any]:
    """Materialise CASSBEAM tables on the host. Safe to call before JIT."""

    global _CASSBEAM_TABLES_HOST
    if _CASSBEAM_TABLES_HOST is not None and _is_tracer(
        _CASSBEAM_TABLES_HOST["frequency_hz"]
    ):
        _CASSBEAM_TABLES_HOST = None
    if _CASSBEAM_TABLES_HOST is None:
        if _is_tracer(jnp.asarray(0.0)):
            raise RuntimeError("CASSBEAM tables must be materialised on the host before JIT")
        _CASSBEAM_TABLES_HOST = _build_cassbeam_tables()
    return _CASSBEAM_TABLES_HOST


def _cassbeam_tables() -> dict[str, Any]:
    if _CASSBEAM_TABLES_HOST is None or _is_tracer(_CASSBEAM_TABLES_HOST["frequency_hz"]):
        return _ensure_host_cassbeam_tables()
    return _CASSBEAM_TABLES_HOST


def _require_nearest_cassbeam_node(frequency_hz: np.ndarray) -> None:
    """Host-side nearest-node check. Must not see JAX tracers."""

    table_hz = _ensure_host_cassbeam_tables()["frequency_hz_host"]
    freq_hz = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    separation = np.min(np.abs(table_hz[:, None] - freq_hz[None, :]), axis=0)
    if np.any(separation > MAX_NEAREST_NODE_SEPARATION_HZ):
        raise ValueError(
            "frequency is farther than "
            f"{MAX_NEAREST_NODE_SEPARATION_HZ / 1e6:.0f} MHz from a CASSBEAM node"
        )


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

        pixel_body = jax.checkpoint(pixel_body)

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


def _pack_correlations(visibility: Array, correlations: tuple[Correlation, ...]) -> Array:
    packed = jnp.zeros((*visibility.shape[:-1], 2, 2), dtype=jnp.complex128)
    slots = {
        Correlation.RR: (0, 0),
        Correlation.RL: (0, 1),
        Correlation.LR: (1, 0),
        Correlation.LL: (1, 1),
    }
    for index, correlation in enumerate(correlations):
        first, second = slots[correlation]
        packed = packed.at[..., first, second].set(visibility[..., index])
    return packed


def _stokes_i_from_coherency_grad(gradient: Array) -> Array:
    return jnp.real(gradient[..., 0, 0] + gradient[..., 1, 1])


def explicit_adjoint_workspace_bytes(
    *,
    parent_count: int,
    pixel_chunk_size: int,
) -> int:
    """Peak extra bytes for the parent accumulator plus one Stokes-I tile."""

    return (int(parent_count) + int(pixel_chunk_size)) * 8


def _accumulate_adjoint_rows(
    residual: Array,
    uvw_m: Array,
    frequency_hz: Array,
    antenna1: Array,
    antenna2: Array,
    l_rad: Array,
    m_rad: Array,
    correlations: tuple[Correlation, ...],
    visibility_chunk_size: int,
    pixel_chunk_size: int,
    jones_for_tile: Callable[[Array, Array], tuple[Array, Array, Array]],
    width_rad: Array | None = None,
    node_valid: Array | None = None,
    approximation: GaussianApproximation = GaussianApproximation.WIDE_FIELD,
    parent_index: Array | None = None,
    node_weight: Array | None = None,
    parent_count: int | None = None,
) -> Array:
    """Stream ``Aᴴ residual`` onto parents, collapsing each direction tile."""

    n_row = uvw_m.shape[0]
    n_channel = frequency_hz.size
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
    indices = (
        jnp.arange(n_dir, dtype=jnp.int32)
        if parent_index is None
        else jnp.asarray(parent_index, dtype=jnp.int32).reshape(-1)
    )
    weights = (
        jnp.ones((n_dir,), dtype=jnp.float64)
        if node_weight is None
        else jnp.asarray(node_weight, dtype=jnp.float64).reshape(-1)
    )
    n_parent = n_dir if parent_count is None else int(parent_count)
    use_square = width_rad is not None
    packed_residual = _pack_correlations(residual, correlations)
    empty = jnp.zeros((n_parent,), dtype=jnp.float64)

    def visibility_body(row_start: int, gradient: Array) -> Array:
        rows = jnp.arange(visibility_chunk_size) + row_start
        in_row = rows < n_row
        safe_rows = jnp.where(in_row, rows, 0)
        uvw = uvw_m[safe_rows]
        a1 = antenna1[safe_rows]
        a2 = antenna2[safe_rows]
        packed = jnp.where(in_row[:, None, None, None], packed_residual[safe_rows], 0.0)

        def pixel_body(pixel_start: int, grad: Array) -> Array:
            pixels = jnp.arange(pixel_chunk_size) + pixel_start
            in_pix = pixels < n_dir
            safe_pix = jnp.where(in_pix, pixels, 0)
            l_tile = l_rad[safe_pix]
            m_tile = m_rad[safe_pix]
            w_tile = widths[safe_pix]
            weight_tile = weights[safe_pix]
            index_tile = indices[safe_pix]
            node_ok = valid_nodes[safe_pix] & in_pix
            jones, valid, off_valid = jones_for_tile(l_tile, m_tile)
            single_plane = jones.shape[0] == 1
            if single_plane:
                jp_tile = jones[0]
                jq_tile = jones[0]
                ok = valid[0] & node_ok[:, None]
                off_ok = off_valid[0] & node_ok[:, None]
            else:
                jp_tile = jones[a1]
                jq_tile = jones[a2]
                ok = (
                    valid[a1]
                    & valid[a2]
                    & in_pix[None, :, None]
                    & node_ok[None, :, None]
                )
                off_ok = (
                    off_valid[a1]
                    & off_valid[a2]
                    & in_pix[None, :, None]
                    & node_ok[None, :, None]
                )

            def channel_body(channel: int, tile_stokes: Array) -> Array:
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
                    in_row[:, None] & node_ok[None, :],
                    kernel,
                    0.0,
                )
                if single_plane:
                    pulled = jnp.einsum(
                        "rd,rij->dij",
                        jnp.conjugate(kernel),
                        packed[:, channel],
                    )
                    pulled = _mask_apparent_coherency(
                        pulled, ok[:, channel], off_ok[:, channel]
                    )
                    left = jnp.conjugate(jnp.swapaxes(jp_tile[:, channel], -1, -2))
                    right = jq_tile[:, channel]
                    d_coherency = jnp.einsum("dij,djk,dkl->dil", left, pulled, right)
                else:
                    row_pull = (
                        jnp.conjugate(kernel)[:, :, None, None]
                        * packed[:, None, channel]
                    )
                    row_pull = _mask_apparent_coherency(
                        row_pull, ok[:, :, channel], off_ok[:, :, channel]
                    )
                    left = jnp.conjugate(jnp.swapaxes(jp_tile[:, :, channel], -1, -2))
                    right = jq_tile[:, :, channel]
                    d_coherency = jnp.einsum(
                        "rdij,rdjk,rdkl->dil", left, row_pull, right
                    )
                stokes = _stokes_i_from_coherency_grad(d_coherency)
                return tile_stokes + jnp.where(in_pix, stokes, 0.0)

            tile_stokes = jax.lax.fori_loop(
                0,
                n_channel,
                channel_body,
                jnp.zeros((pixel_chunk_size,), dtype=jnp.float64),
            )
            contrib = jnp.where(node_ok, tile_stokes * weight_tile, 0.0)
            return grad.at[index_tile].add(contrib)

        pixel_body = jax.checkpoint(pixel_body)
        n_pixel_tiles = (n_dir + pixel_chunk_size - 1) // pixel_chunk_size
        return cast(
            Array,
            jax.lax.fori_loop(
                0,
                n_pixel_tiles,
                lambda index, acc: pixel_body(index * pixel_chunk_size, acc),
                gradient,
            ),
        )

    n_row_tiles = (n_row + visibility_chunk_size - 1) // visibility_chunk_size
    return cast(
        Array,
        jax.lax.fori_loop(
            0,
            n_row_tiles,
            lambda index, acc: visibility_body(index * visibility_chunk_size, acc),
            empty,
        ),
    )


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


_SUPPORTED_CIRCULAR = {
    Correlation.RR,
    Correlation.RL,
    Correlation.LR,
    Correlation.LL,
}


def _require_streamed_jax_operator(
    block: VisibilityBlock,
    beam: VoltageBeamModel,
    selected: BeamOperatorConfig,
    *,
    antenna_position_m: ArrayLike,
    n_direction: int,
) -> np.ndarray:
    """Fail closed on policy, correlations, antennas, and Jones-tile memory."""

    if selected.policy is not BeamOperatorPolicy.STREAM:
        raise ValueError("JAX voltage operator only implements STREAM policy")
    if not block.correlations:
        raise ValueError("block must contain circular correlations")
    unknown = set(block.correlations) - _SUPPORTED_CIRCULAR
    if unknown:
        raise ValueError(f"unsupported circular correlations {sorted(unknown)}")
    positions = np.asarray(antenna_position_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("antenna_position_m must have shape (antenna, 3)")
    if positions.shape[0] < block.antenna_count:
        raise ValueError("antenna_position_m must cover every antenna in the block")
    depends_on_time = bool(getattr(beam, "antenna_planes_from_parallactic", False))
    plane_count = int(positions.shape[0]) if depends_on_time else 1
    tile_dirs = min(int(selected.pixel_chunk_size), int(n_direction))
    n_channel = int(np.asarray(block.frequency_hz).size)
    jones_bytes = timestep_jones_bytes(plane_count, tile_dirs, n_channel)
    if jones_bytes > selected.max_timestep_jones_bytes:
        raise ValueError(
            "Jones tile "
            f"({plane_count} antennas × {tile_dirs} directions × "
            f"{n_channel} channels) needs {jones_bytes} bytes; "
            f"max_timestep_jones_bytes is {selected.max_timestep_jones_bytes}. "
            "Reduce pixel_chunk_size."
        )
    return positions


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
    l_array = jnp.asarray(l_rad, dtype=jnp.float64).reshape(-1)
    m_array = jnp.asarray(m_rad, dtype=jnp.float64).reshape(-1)
    if l_array.size != m_array.size or l_array.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    frequency = jnp.asarray(block.frequency_hz, dtype=jnp.float64)
    positions = _require_streamed_jax_operator(
        block,
        beam,
        selected,
        antenna_position_m=antenna_position_m,
        n_direction=int(l_array.size),
    )
    depends_on_time = bool(getattr(beam, "antenna_planes_from_parallactic", False))
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
        _require_nearest_cassbeam_node(block.frequency_hz)
    unique_times, row_index, row_valid = _time_groups(np.asarray(block.time_s, dtype=np.float64))
    width = None if width_rad is None else jnp.asarray(width_rad, dtype=jnp.float64).reshape(-1)
    nodes = None if node_valid is None else jnp.asarray(node_valid, dtype=bool).reshape(-1)
    return _predict_voltage_beam_device(
        uvw=jnp.asarray(block.uvw_m, dtype=jnp.float64),
        frequency=frequency,
        antenna1=jnp.asarray(block.antenna1, dtype=jnp.int32),
        antenna2=jnp.asarray(block.antenna2, dtype=jnp.int32),
        unique=jnp.asarray(unique_times, dtype=jnp.float64),
        row_index=jnp.asarray(row_index, dtype=jnp.int32),
        row_valid=jnp.asarray(row_valid, dtype=bool),
        l_array=l_array,
        m_array=m_array,
        coherency=coherency,
        n_row=int(block.visibility.shape[0]),
        n_corr=len(block.correlations),
        correlations=block.correlations,
        visibility_chunk_size=selected.visibility_chunk_size,
        pixel_chunk_size=selected.pixel_chunk_size,
        beam=beam,
        state=state,
        polynomials=polynomials,
        cassbeam_tables=cassbeam_tables,
        depends_on_time=depends_on_time,
        pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
        phase_centre=block.phase_centre_rad,
        positions=jnp.asarray(positions, dtype=jnp.float64),
        width=width,
        nodes=nodes,
        approximation=approximation,
    )


def _predict_voltage_beam_device(
    *,
    uvw: Array,
    frequency: Array,
    antenna1: Array,
    antenna2: Array,
    unique: Array,
    row_index: Array,
    row_valid: Array,
    l_array: Array,
    m_array: Array,
    coherency: Array,
    n_row: int,
    n_corr: int,
    correlations: tuple[Correlation, ...],
    visibility_chunk_size: int,
    pixel_chunk_size: int,
    beam: VoltageBeamModel,
    state: BeamCalibrationState,
    polynomials: dict[str, Array] | None,
    cassbeam_tables: dict[str, Array] | None,
    depends_on_time: bool,
    pointing_offset_lm_rad: tuple[float, float] | None,
    phase_centre: tuple[float, float],
    positions: Array,
    width: Array | None,
    nodes: Array | None,
    approximation: GaussianApproximation,
) -> tuple[Array, Array, Array]:
    def time_body(
        carry: tuple[Array, Array, Array], time_index: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        prediction, copolar_valid, leakage_valid = carry
        time_s = unique[time_index]
        chi = parallactic_angle_rad_jax(time_s[None], phase_centre, positions)[0]
        if not depends_on_time:
            chi = jnp.zeros((1,), dtype=jnp.float64)

        def jones_for_tile(l_tile: Array, m_tile: Array) -> tuple[Array, Array, Array]:
            if pointing_offset_lm_rad is None:
                l_eval, m_eval = l_tile, m_tile
            else:
                l_eval = l_tile - jnp.asarray(pointing_offset_lm_rad[0], dtype=jnp.float64)
                m_eval = m_tile - jnp.asarray(pointing_offset_lm_rad[1], dtype=jnp.float64)
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

        rows = row_index[time_index]
        ok = row_valid[time_index]
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
            visibility_chunk_size,
            pixel_chunk_size,
            jones_for_tile,
            width_rad=width,
            node_valid=nodes,
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

    empty_vis = jnp.zeros((n_row, frequency.size, n_corr), dtype=jnp.complex128)
    empty_ok = jnp.zeros((n_row, frequency.size), dtype=jnp.int32)
    (prediction, copolar_count, leakage_count), _ = jax.lax.scan(
        time_body,
        (empty_vis, empty_ok, empty_ok),
        jnp.arange(unique.shape[0], dtype=jnp.int32),
    )
    return prediction, copolar_count > 0, leakage_count > 0


def _adjoint_voltage_beam_jax_arrays(
    residual: Array,
    block: VisibilityBlock,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    config: BeamOperatorConfig | None = None,
    width_rad: ArrayLike | None = None,
    node_valid: ArrayLike | None = None,
    kernel_approximation: GaussianApproximation | str = GaussianApproximation.WIDE_FIELD,
    parent_index: ArrayLike | None = None,
    node_weight: ArrayLike | None = None,
    parent_count: int | None = None,
) -> Array:
    """Stokes-I gradient ``Aᴴ residual`` with the streamed JAX adjoint.

    When ``parent_index`` is provided the return is already reduced onto
    fitted parents. Otherwise it is one Stokes-I value per integration node.
    """

    selected = config or BeamOperatorConfig()
    state = require_beam_calibration_state(calibration_state)
    approximation = GaussianApproximation(kernel_approximation)
    l_array = jnp.asarray(l_rad, dtype=jnp.float64).reshape(-1)
    m_array = jnp.asarray(m_rad, dtype=jnp.float64).reshape(-1)
    if l_array.size != m_array.size or l_array.size == 0:
        raise ValueError("l_rad and m_rad must be nonempty and the same size")
    frequency = jnp.asarray(block.frequency_hz, dtype=jnp.float64)
    positions = _require_streamed_jax_operator(
        block,
        beam,
        selected,
        antenna_position_m=antenna_position_m,
        n_direction=int(l_array.size),
    )
    residual_array = jnp.asarray(residual, dtype=jnp.complex128)
    if residual_array.shape != block.visibility.shape:
        raise ValueError("residual must match block.visibility")
    depends_on_time = bool(getattr(beam, "antenna_planes_from_parallactic", False))
    polynomials = None
    cassbeam_tables = None
    if isinstance(beam, (CompositeScalarVoltageBeam, CassbeamCBandVoltageBeam)):
        polynomials = {
            key: jnp.asarray(value)
            for key, value in _perley_channel_polynomials(np.asarray(block.frequency_hz)).items()
        }
    if isinstance(beam, CassbeamCBandVoltageBeam):
        cassbeam_tables = _cassbeam_tables()
        _require_nearest_cassbeam_node(block.frequency_hz)
    unique_times, row_index, row_valid = _time_groups(np.asarray(block.time_s, dtype=np.float64))
    width = None if width_rad is None else jnp.asarray(width_rad, dtype=jnp.float64).reshape(-1)
    nodes = None if node_valid is None else jnp.asarray(node_valid, dtype=bool).reshape(-1)
    parents = (
        None
        if parent_index is None
        else jnp.asarray(parent_index, dtype=jnp.int32).reshape(-1)
    )
    weights = (
        None
        if node_weight is None
        else jnp.asarray(node_weight, dtype=jnp.float64).reshape(-1)
    )
    n_out = int(l_array.size) if parent_count is None else int(parent_count)
    return _adjoint_voltage_beam_device(
        residual=residual_array,
        uvw=jnp.asarray(block.uvw_m, dtype=jnp.float64),
        frequency=frequency,
        antenna1=jnp.asarray(block.antenna1, dtype=jnp.int32),
        antenna2=jnp.asarray(block.antenna2, dtype=jnp.int32),
        unique=jnp.asarray(unique_times, dtype=jnp.float64),
        row_index=jnp.asarray(row_index, dtype=jnp.int32),
        row_valid=jnp.asarray(row_valid, dtype=bool),
        l_array=l_array,
        m_array=m_array,
        correlations=block.correlations,
        visibility_chunk_size=selected.visibility_chunk_size,
        pixel_chunk_size=selected.pixel_chunk_size,
        beam=beam,
        state=state,
        polynomials=polynomials,
        cassbeam_tables=cassbeam_tables,
        depends_on_time=depends_on_time,
        pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
        phase_centre=block.phase_centre_rad,
        positions=jnp.asarray(positions, dtype=jnp.float64),
        width=width,
        nodes=nodes,
        approximation=approximation,
        parent_index=parents,
        node_weight=weights,
        parent_count=None if parent_count is None else n_out,
    )


def _adjoint_voltage_beam_device(
    *,
    residual: Array,
    uvw: Array,
    frequency: Array,
    antenna1: Array,
    antenna2: Array,
    unique: Array,
    row_index: Array,
    row_valid: Array,
    l_array: Array,
    m_array: Array,
    correlations: tuple[Correlation, ...],
    visibility_chunk_size: int,
    pixel_chunk_size: int,
    beam: VoltageBeamModel,
    state: BeamCalibrationState,
    polynomials: dict[str, Array] | None,
    cassbeam_tables: dict[str, Array] | None,
    depends_on_time: bool,
    pointing_offset_lm_rad: tuple[float, float] | None,
    phase_centre: tuple[float, float],
    positions: Array,
    width: Array | None,
    nodes: Array | None,
    approximation: GaussianApproximation,
    parent_index: Array | None,
    node_weight: Array | None,
    parent_count: int | None,
) -> Array:
    n_out = int(l_array.size) if parent_count is None else int(parent_count)

    def time_body(gradient: Array, time_index: Array) -> tuple[Array, None]:
        time_s = unique[time_index]
        chi = parallactic_angle_rad_jax(time_s[None], phase_centre, positions)[0]
        if not depends_on_time:
            chi = jnp.zeros((1,), dtype=jnp.float64)

        def jones_for_tile(l_tile: Array, m_tile: Array) -> tuple[Array, Array, Array]:
            if pointing_offset_lm_rad is None:
                l_eval, m_eval = l_tile, m_tile
            else:
                l_eval = l_tile - jnp.asarray(pointing_offset_lm_rad[0], dtype=jnp.float64)
                m_eval = m_tile - jnp.asarray(pointing_offset_lm_rad[1], dtype=jnp.float64)
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

        rows = row_index[time_index]
        ok = row_valid[time_index]
        safe_rows = jnp.where(ok, rows, 0)
        packed = jnp.where(ok[:, None, None], residual[safe_rows], 0.0)
        tile_grad = _accumulate_adjoint_rows(
            packed,
            uvw[safe_rows],
            frequency,
            antenna1[safe_rows],
            antenna2[safe_rows],
            l_array,
            m_array,
            correlations,
            visibility_chunk_size,
            pixel_chunk_size,
            jones_for_tile,
            width_rad=width,
            node_valid=nodes,
            approximation=approximation,
            parent_index=parent_index,
            node_weight=node_weight,
            parent_count=parent_count,
        )
        return gradient + tile_grad, None

    empty = jnp.zeros((n_out,), dtype=jnp.float64)
    gradient, _ = jax.lax.scan(
        time_body,
        empty,
        jnp.arange(unique.shape[0], dtype=jnp.int32),
    )
    return gradient


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
        _require_nearest_cassbeam_node(frequency)
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


def _padded_node_tile(
    array: np.ndarray, start: int, size: int, fill: Any
) -> np.ndarray:
    stop = min(start + size, int(array.shape[0]))
    tile = np.asarray(array[start:stop])
    if tile.shape[0] == size:
        return tile
    padded = np.full((size, *array.shape[1:]), fill, dtype=array.dtype)
    padded[: tile.shape[0]] = tile
    return padded


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
    """Weighted MSE and Stokes-I gradient over fitted parents, not nodes.

    Direction tiles are differentiated from Python so XLA cannot fuse the
    pixel loop into one scatter-add over every parent.
    """

    parent = jnp.asarray(parent_flux, dtype=jnp.float64).reshape(-1)
    if int(parent.size) != plan.parent_count:
        raise ValueError("parent_flux must match the number of fitted parents")
    if isinstance(beam, CassbeamCBandVoltageBeam):
        _ensure_host_cassbeam_tables()
        _require_nearest_cassbeam_node(np.asarray(block.frequency_hz))
    selected = config or BeamOperatorConfig()
    local_l, local_m = plan.local_directions(block.phase_centre_rad)
    sample = np.asarray(block.active if train_mask is None else train_mask, dtype=bool)
    if sample.shape != block.visibility.shape:
        raise ValueError("train_mask must match block.visibility")
    n_dir = int(local_l.size)
    tile = min(int(selected.pixel_chunk_size), n_dir)

    def _chunk_predict(
        values: Array,
        l_tile: Array,
        m_tile: Array,
        width_tile: Array,
        valid_tile: Array,
        parent_index_tile: Array,
        weight_tile: Array,
    ) -> tuple[Array, Array, Array]:
        node_flux = values[parent_index_tile] * weight_tile
        node_flux = jnp.where(valid_tile, node_flux, 0.0)
        return _predict_voltage_beam_jax_arrays(
            block,
            l_tile,
            m_tile,
            SkyStokesPlanes(stokes_i=cast(np.ndarray, node_flux)),
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=selected,
            width_rad=width_tile,
            node_valid=valid_tile,
            kernel_approximation=plan.approximation,
        )

    _chunk_predict = jax.jit(_chunk_predict)

    tiles: list[tuple[Array, Array, Array, Array, Array, Array]] = []
    predicted = jnp.zeros(block.visibility.shape, dtype=jnp.complex128)
    copolar_valid = jnp.zeros(block.visibility.shape[:2], dtype=bool)
    leakage_valid = jnp.zeros_like(copolar_valid)
    node_valid = np.asarray(plan.node_valid)
    for start in range(0, n_dir, tile):
        valid_tile = _padded_node_tile(node_valid, start, tile, False)
        if not np.any(valid_tile):
            continue
        packed = (
            jnp.asarray(_padded_node_tile(np.asarray(local_l), start, tile, 0.0)),
            jnp.asarray(_padded_node_tile(np.asarray(local_m), start, tile, 0.0)),
            jnp.asarray(_padded_node_tile(np.asarray(plan.width_rad), start, tile, 0.0)),
            jnp.asarray(valid_tile),
            jnp.asarray(_padded_node_tile(np.asarray(plan.parent_index), start, tile, 0)),
            jnp.asarray(_padded_node_tile(np.asarray(plan.weight), start, tile, 0.0)),
        )
        tiles.append(packed)
        pred_tile, copolar_tile, leakage_tile = _chunk_predict(parent, *packed)
        predicted = predicted + pred_tile
        copolar_valid = copolar_valid | copolar_tile
        leakage_valid = leakage_valid | leakage_tile
    n_tiles = len(tiles)
    if n_tiles > 1:
        print(f"vjp direction tiles={n_tiles} chunk={tile}", flush=True)

    beam_ok = _correlation_validity(copolar_valid, leakage_valid, block.correlations)
    observation = jnp.asarray(block.visibility)
    weight = jnp.asarray(block.weight)
    flag = ~jnp.asarray(sample) | ~beam_ok

    def _mse(prediction: Array) -> Array:
        return weighted_complex_mse(prediction, observation, weight, flag)

    def _fwd(
        values: Array,
        l_tile: Array,
        m_tile: Array,
        width_tile: Array,
        valid_tile: Array,
        parent_index_tile: Array,
        weight_tile: Array,
    ) -> Array:
        pred_tile, _copolar, _leakage = _chunk_predict(
            values,
            l_tile,
            m_tile,
            width_tile,
            valid_tile,
            parent_index_tile,
            weight_tile,
        )
        return pred_tile

    loss, mse_vjp = jax.vjp(_mse, predicted)
    cotangent = mse_vjp(jnp.ones((), dtype=loss.dtype))[0]
    gradient = jnp.zeros_like(parent)
    for index, packed in enumerate(tiles):
        if n_tiles > 1 and (index == 0 or (index + 1) % 64 == 0 or index + 1 == n_tiles):
            print(f"vjp tile {index + 1}/{n_tiles}", flush=True)
        _prediction, chunk_vjp = jax.vjp(_fwd, parent, *packed)
        gradient = gradient + chunk_vjp(cotangent)[0]
    return loss, gradient


def _reduce_parent_gradient(
    node_gradient: Array,
    parent_index: Array,
    weight: Array,
    node_valid: Array,
    parent_count: int,
) -> Array:
    values = jnp.asarray(node_gradient, dtype=jnp.float64)
    if values.ndim == 2:
        values = values.sum(axis=1)
    contrib = jnp.where(node_valid, values * weight, 0.0)
    return jnp.zeros((parent_count,), dtype=jnp.float64).at[parent_index].add(contrib)


_EXPLICIT_KERNELS: dict[tuple[Any, ...], Any] = {}
_EXPLICIT_KERNEL_BUILDS = 0
EXPLICIT_KERNEL_CACHE_LIMIT = 16


def explicit_kernel_build_count() -> int:
    return _EXPLICIT_KERNEL_BUILDS


def explicit_cached_kernel_count() -> int:
    return len(_EXPLICIT_KERNELS)


def clear_explicit_kernels() -> None:
    global _EXPLICIT_KERNEL_BUILDS
    _EXPLICIT_KERNELS.clear()
    _EXPLICIT_KERNEL_BUILDS = 0


def _lookup_explicit_kernel(key: tuple[Any, ...]) -> Any | None:
    cached = _EXPLICIT_KERNELS.pop(key, None)
    if cached is not None:
        _EXPLICIT_KERNELS[key] = cached
    return cached


def _store_explicit_kernel(key: tuple[Any, ...], compiled: Any) -> Any:
    while len(_EXPLICIT_KERNELS) >= EXPLICIT_KERNEL_CACHE_LIMIT:
        _EXPLICIT_KERNELS.pop(next(iter(_EXPLICIT_KERNELS)))
    _EXPLICIT_KERNELS[key] = compiled
    return compiled


def _array_cache_key(value: ArrayLike | None) -> tuple[Any, ...]:
    if value is None:
        return ("none",)
    array = np.asarray(value)
    return (array.shape, array.dtype.str, array.tobytes())


def _catalog_cache_key(catalog: VLABeamCatalog) -> tuple[Any, ...]:
    return (
        float(catalog.dish_diameter_m),
        float(catalog.blockage_diameter_m),
        float(catalog.gaussian_fwhm_factor),
        float(catalog.airy_max_radius_rad_at_1ghz),
        float(catalog.squint_fwhm_fraction),
        float(catalog.squint_reference_hz),
    )


def _beam_cache_key(beam: object) -> tuple[Any, ...]:
    identity = (type(beam).__name__, getattr(beam, "model_id", type(beam).__name__))
    if isinstance(beam, ManufacturedVoltageBeam):
        return (
            *identity,
            _array_cache_key(beam.intercept),
            _array_cache_key(beam.grad_l),
            _array_cache_key(beam.grad_m),
            _array_cache_key(beam.hess_ll),
            _array_cache_key(beam.hess_lm),
            _array_cache_key(beam.hess_mm),
            beam.valid_radius_rad,
            beam.off_diagonal_radius_rad,
            beam.off_diagonal_valid,
            beam.rotate_parallactic,
        )
    if isinstance(beam, AnalyticAiryVoltageBeam):
        return (*identity, _catalog_cache_key(beam.catalog))
    if isinstance(beam, Perley2016CBandVoltageBeam):
        return (*identity, beam.frequency_policy.value)
    if isinstance(beam, CompositeScalarVoltageBeam):
        return (
            *identity,
            beam.handover.value,
            _beam_cache_key(beam.main),
            _beam_cache_key(beam.outer),
        )
    if isinstance(beam, CassbeamCBandVoltageBeam):
        pin = beam.artifact.pin
        outer = None if beam.outer is None else _beam_cache_key(beam.outer)
        return (
            *identity,
            pin.artifact_id,
            pin.generator_or_path,
            pin.input_checksum,
            pin.output_checksum,
            pin.generator_version,
            beam.off_diagonal,
            beam.allow_unfrozen,
            outer,
        )
    return (*identity, id(beam))


def _explicit_kernel_key(
    beam: VoltageBeamModel,
    correlations: tuple[Correlation, ...],
    selected: BeamOperatorConfig,
    approximation: GaussianApproximation,
    *,
    n_row: int,
    n_channel: int,
    n_corr: int,
    n_dir: int,
    parent_count: int,
    n_unique: int,
    max_time_rows: int,
    n_ant: int,
    calibration_state: BeamCalibrationState,
    phase_centre: tuple[float, float],
    frequency_hz: ArrayLike,
) -> tuple[Any, ...]:
    frequency = tuple(np.asarray(frequency_hz, dtype=np.float64).reshape(-1).tolist())
    return (
        _beam_cache_key(beam),
        correlations,
        int(selected.visibility_chunk_size),
        int(selected.pixel_chunk_size),
        selected.pointing_offset_lm_rad,
        approximation.value,
        n_row,
        n_channel,
        n_corr,
        n_dir,
        parent_count,
        n_unique,
        max_time_rows,
        n_ant,
        calibration_state.value,
        phase_centre,
        frequency,
        bool(getattr(beam, "antenna_planes_from_parallactic", False)),
        bool(getattr(beam, "off_diagonal", False)),
    )


def _compiled_explicit_kernel(
    key: tuple[Any, ...],
    *,
    beam: VoltageBeamModel,
    correlations: tuple[Correlation, ...],
    selected: BeamOperatorConfig,
    approximation: GaussianApproximation,
    state: BeamCalibrationState,
    polynomials: dict[str, Array] | None,
    cassbeam_tables: dict[str, Array] | None,
    depends_on_time: bool,
    phase_centre: tuple[float, float],
    n_row: int,
    n_corr: int,
    parent_count: int,
) -> Any:
    global _EXPLICIT_KERNEL_BUILDS
    cached = _lookup_explicit_kernel(key)
    if cached is not None:
        return cached

    def kernel(
        values: Array,
        visibility: Array,
        vis_weight: Array,
        sample: Array,
        uvw: Array,
        frequency: Array,
        antenna1: Array,
        antenna2: Array,
        unique: Array,
        row_index: Array,
        row_valid: Array,
        l_rad: Array,
        m_rad: Array,
        width: Array,
        parent_index: Array,
        node_weight: Array,
        node_ok: Array,
        antenna_position_m: Array,
    ) -> tuple[Array, Array]:
        node_flux = jnp.where(node_ok, values[parent_index] * node_weight, 0.0)
        intensity = _sky_plane(node_flux, int(l_rad.size), int(frequency.size), "stokes_i")
        zeros = jnp.zeros_like(intensity)
        coherency = _circular_coherency(intensity, zeros, zeros, zeros)
        predicted, copolar_valid, leakage_valid = _predict_voltage_beam_device(
            uvw=uvw,
            frequency=frequency,
            antenna1=antenna1,
            antenna2=antenna2,
            unique=unique,
            row_index=row_index,
            row_valid=row_valid,
            l_array=l_rad,
            m_array=m_rad,
            coherency=coherency,
            n_row=n_row,
            n_corr=n_corr,
            correlations=correlations,
            visibility_chunk_size=selected.visibility_chunk_size,
            pixel_chunk_size=selected.pixel_chunk_size,
            beam=beam,
            state=state,
            polynomials=polynomials,
            cassbeam_tables=cassbeam_tables,
            depends_on_time=depends_on_time,
            pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
            phase_centre=phase_centre,
            positions=antenna_position_m,
            width=width,
            nodes=node_ok,
            approximation=approximation,
        )
        beam_ok = _correlation_validity(copolar_valid, leakage_valid, correlations)
        flag = (~sample) | (~beam_ok)
        loss = weighted_complex_mse(predicted, visibility, vis_weight, flag)
        active = effective_weight(visibility, vis_weight, flag)
        weight_sum = jnp.sum(active)
        residual = jnp.where(active > 0, predicted - visibility, 0.0)
        hilbert_residual = jnp.where(
            weight_sum > 0, (2.0 / weight_sum) * active * residual, 0.0
        )
        gradient = _adjoint_voltage_beam_device(
            residual=hilbert_residual,
            uvw=uvw,
            frequency=frequency,
            antenna1=antenna1,
            antenna2=antenna2,
            unique=unique,
            row_index=row_index,
            row_valid=row_valid,
            l_array=l_rad,
            m_array=m_rad,
            correlations=correlations,
            visibility_chunk_size=selected.visibility_chunk_size,
            pixel_chunk_size=selected.pixel_chunk_size,
            beam=beam,
            state=state,
            polynomials=polynomials,
            cassbeam_tables=cassbeam_tables,
            depends_on_time=depends_on_time,
            pointing_offset_lm_rad=selected.pointing_offset_lm_rad,
            phase_centre=phase_centre,
            positions=antenna_position_m,
            width=width,
            nodes=node_ok,
            approximation=approximation,
            parent_index=parent_index,
            node_weight=node_weight,
            parent_count=parent_count,
        )
        return loss, gradient

    compiled: Any = jax.jit(kernel)
    _store_explicit_kernel(key, compiled)
    _EXPLICIT_KERNEL_BUILDS += 1
    return compiled


def predict_voltage_from_plan_value_and_grad_explicit_jax(
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
    """Weighted MSE and parent gradient from one streamed forward plus adjoint.

    The compiled kernel is cached by beam, correlations, chunk sizes and
    batch/plan shapes. Parent flux and the visibility batch are arguments.
    """

    parent = jnp.asarray(parent_flux, dtype=jnp.float64).reshape(-1)
    if int(parent.size) != plan.parent_count:
        raise ValueError("parent_flux must match the number of fitted parents")
    if isinstance(beam, CassbeamCBandVoltageBeam):
        _ensure_host_cassbeam_tables()
        _require_nearest_cassbeam_node(np.asarray(block.frequency_hz))
    selected = config or BeamOperatorConfig()
    state = require_beam_calibration_state(calibration_state)
    approximation = GaussianApproximation(plan.approximation)
    local_l, local_m = plan.local_directions(block.phase_centre_rad)
    sample = np.asarray(block.active if train_mask is None else train_mask, dtype=bool)
    if sample.shape != block.visibility.shape:
        raise ValueError("train_mask must match block.visibility")
    positions = _require_streamed_jax_operator(
        block,
        beam,
        selected,
        antenna_position_m=antenna_position_m,
        n_direction=int(np.asarray(local_l).size),
    )
    unique_times, row_index, row_valid = _time_groups(np.asarray(block.time_s, dtype=np.float64))
    polynomials = None
    cassbeam_tables = None
    if isinstance(beam, (CompositeScalarVoltageBeam, CassbeamCBandVoltageBeam)):
        polynomials = {
            key: jnp.asarray(value)
            for key, value in _perley_channel_polynomials(np.asarray(block.frequency_hz)).items()
        }
    if isinstance(beam, CassbeamCBandVoltageBeam):
        cassbeam_tables = _cassbeam_tables()
    key = _explicit_kernel_key(
        beam,
        block.correlations,
        selected,
        approximation,
        n_row=int(block.visibility.shape[0]),
        n_channel=int(np.asarray(block.frequency_hz).size),
        n_corr=len(block.correlations),
        n_dir=int(plan.node_count),
        parent_count=int(plan.parent_count),
        n_unique=int(unique_times.size),
        max_time_rows=int(row_index.shape[1]),
        n_ant=int(positions.shape[0]),
        calibration_state=state,
        phase_centre=block.phase_centre_rad,
        frequency_hz=block.frequency_hz,
    )
    kernel = _compiled_explicit_kernel(
        key,
        beam=beam,
        correlations=block.correlations,
        selected=selected,
        approximation=approximation,
        state=state,
        polynomials=polynomials,
        cassbeam_tables=cassbeam_tables,
        depends_on_time=bool(getattr(beam, "antenna_planes_from_parallactic", False)),
        phase_centre=block.phase_centre_rad,
        n_row=int(block.visibility.shape[0]),
        n_corr=len(block.correlations),
        parent_count=int(plan.parent_count),
    )
    return cast(tuple[Array, Array], kernel(
        parent,
        jnp.asarray(block.visibility),
        jnp.asarray(block.weight),
        jnp.asarray(sample),
        jnp.asarray(block.uvw_m, dtype=jnp.float64),
        jnp.asarray(block.frequency_hz, dtype=jnp.float64),
        jnp.asarray(block.antenna1, dtype=jnp.int32),
        jnp.asarray(block.antenna2, dtype=jnp.int32),
        jnp.asarray(unique_times, dtype=jnp.float64),
        jnp.asarray(row_index, dtype=jnp.int32),
        jnp.asarray(row_valid, dtype=bool),
        jnp.asarray(local_l, dtype=jnp.float64),
        jnp.asarray(local_m, dtype=jnp.float64),
        jnp.asarray(plan.width_rad, dtype=jnp.float64),
        jnp.asarray(plan.parent_index, dtype=jnp.int32),
        jnp.asarray(plan.weight, dtype=jnp.float64),
        jnp.asarray(plan.node_valid, dtype=bool),
        jnp.asarray(positions, dtype=jnp.float64),
    ))


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
