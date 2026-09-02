"""Nominal CASSBEAM C-band voltage beam and Phase 6 imaging modes.

The tables were generated on Bacchus with Ubuntu cassbeam 1.1-4build2.
They are electromagnetic, not measured, and not CASA-accepted. Imaging
still uses the static Airy path. Unfrozen full Jones refuses evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sl1mjax.beam import VLABeamCatalog
from sl1mjax.beam_conventions import (
    CIRCULAR_P_JONES,
    CIRCULAR_STOKES,
    JONES_RECEPTOR_ORDER,
    OBSERVATION_3C391_FREQUENCY_HZ,
    ON_AXIS_DI_JONES_ORDER,
    BeamCalibrationState,
    beam_requires_identity_on_axis,
    nrao_gaussian_fwhm_arcmin,
    require_beam_calibration_state,
)
from sl1mjax.full_jones import (
    AntennaAveraging,
    FullJonesContents,
    FullJonesOuterFieldPolicy,
    FullJonesReferencePin,
    TermPresence,
    TransmitReceiveConvention,
    apply_full_jones_outer_field,
    require_frozen_full_jones_reference,
)
from sl1mjax.polarization import circular_parallactic_jones
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import (
    JONES_AXES,
    AnalyticAiryVoltageBeam,
    BeamCoordinates,
    BeamEvaluation,
    CompositeHandoverPolicy,
    CompositeScalarVoltageBeam,
    Perley2016CBandVoltageBeam,
)

_ARTIFACT_DIR = Path(__file__).with_name("data") / "cassbeam_cband"
_MANIFEST_NAME = "manifest.json"
CASSBEAM_CBAND_MODEL_ID = "cassbeam_nominal_cband"
DIAGONAL_COPOLAR_ACCEPTED = False
# Frozen CASA .pb products are a reference measurement, not this flag.
# Do not loosen the 5% contour to absorb the measured skirt/null offset.
CASA_AWP2_MAIN_LOBE_POWER_TOLERANCE = 0.05
# Half the 128 MHz generated-node spacing. Covers 3C391 SPW0 (4.536–4.662 GHz).
MAX_NEAREST_NODE_SEPARATION_HZ = 64.0e6
# Default CASA Airy (0.8 deg at 1 GHz) is smaller than this CASSBEAM raster
# at C-band. The composed outer field needs a wider analytic cutoff.
CASSBEAM_OUTER_AIRY_MAX_RADIUS_RAD_AT_1GHZ = np.deg2rad(4.0)
CASSBEAM_PARALLACTIC_BASIS_LOCK = "physically_unverified"
CASA_PARANG_PARALLACTIC_BASIS = "P^H E_feed(R_{-chi} s) P with CASA circular P"
UNCALIBRATED_PARALLACTIC_BASIS = "E_feed(R_{-chi} s) P with CASA circular P"
MAINLOBE_POWER_FRACTION = 0.20


class _OuterVoltageBeam(Protocol):
    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation: ...


class BeamImagingMode(StrEnum):
    """Explicit predict-path beam modes. Imaging still defaults to static Airy."""

    STATIC_SCALAR = "static_scalar"
    STREAMED_SCALAR = "streamed_scalar"
    DIAGONAL_COPOLAR = "diagonal_copolar"
    FULL_JONES = "full_jones"


@dataclass(frozen=True)
class CassbeamCBandTable:
    """One CASSBEAM frequency plane."""

    frequency_hz: float
    jones: NDArray[np.complex128]
    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    fwhm_l_rad: float
    fwhm_m_rad: float
    pixel_scale_rad: float
    l_origin_index: int
    m_origin_index: int
    params: dict[str, str]


@dataclass(frozen=True)
class CassbeamCBandArtifact:
    """Pinned nominal C-band CASSBEAM tables. Not scientifically frozen."""

    tables: tuple[CassbeamCBandTable, ...]
    manifest: dict[str, object]
    pin: FullJonesReferencePin


class CassbeamCBandVoltageBeam:
    """Nearest-node CASSBEAM Jones with bilinear (l, m) lookup.

    ``diagonal_copolar`` zeros off-diagonals. Full Jones keeps them and
    refuses evaluation unless the pin is frozen or ``allow_unfrozen`` is
    set for isolated tests. ``casa_parang_true`` applies ``P^H E P``;
    ``uncalibrated`` applies ``E P``. Signs remain physically unverified.
    """

    model_id: str = CASSBEAM_CBAND_MODEL_ID
    antenna_planes_from_parallactic: bool = True

    def __init__(
        self,
        artifact: CassbeamCBandArtifact,
        *,
        off_diagonal: bool,
        outer: _OuterVoltageBeam | None = None,
        allow_unfrozen: bool = False,
    ) -> None:
        self.artifact = artifact
        self.off_diagonal = bool(off_diagonal)
        self.outer = outer
        self.allow_unfrozen = bool(allow_unfrozen)
        if self.off_diagonal and not self.allow_unfrozen:
            require_frozen_full_jones_reference(artifact.pin)

    def evaluate(
        self,
        coordinates: BeamCoordinates,
        *,
        calibration_state: BeamCalibrationState | str,
    ) -> BeamEvaluation:
        state = require_beam_calibration_state(calibration_state)
        if self.off_diagonal and not self.allow_unfrozen:
            require_frozen_full_jones_reference(self.artifact.pin)
        l_off, m_off = _pointing_relative_lm(coordinates)
        antennas = (
            np.array([0], dtype=np.int32)
            if coordinates.antenna_id is None
            else np.asarray(coordinates.antenna_id)
        )
        chi = np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64)
        if chi.size == 1:
            chi = np.full(antennas.size, float(chi[0]), dtype=np.float64)
        if chi.size != antennas.size:
            raise ValueError("parallactic_angle_rad must be scalar or one value per antenna")
        jones = np.zeros(
            (antennas.size, l_off.size, coordinates.frequency_hz.size, 2, 2),
            dtype=np.complex128,
        )
        valid = np.zeros(jones.shape[:3], dtype=bool)
        selected_hz: list[float] = []
        for channel, frequency_hz in enumerate(coordinates.frequency_hz):
            table = _nearest_table(self.artifact.tables, float(frequency_hz))
            selected_hz.append(table.frequency_hz)
            for antenna_index, angle in enumerate(chi):
                l_ant, m_ant = _antenna_frame_lm(l_off, m_off, float(angle))
                plane, plane_ok = _bilinear_jones(table, l_ant, m_ant)
                if not self.off_diagonal:
                    plane = _diagonal_only(plane)
                if beam_requires_identity_on_axis(state):
                    plane = _normalize_on_axis(
                        plane, table, off_diagonal=self.off_diagonal
                    )
                plane = _apply_parallactic_jones(plane, float(angle), state)
                jones[antenna_index, :, channel] = plane
                valid[antenna_index, :, channel] = plane_ok
        cassbeam_valid = valid.copy()
        if self.outer is not None:
            scalar = self.outer.evaluate(coordinates, calibration_state=state)
            scalar_jones, scalar_valid = _broadcast_antenna_planes(
                scalar.jones, scalar.valid, antennas.size
            )
            composed = apply_full_jones_outer_field(
                jones,
                valid,
                scalar_jones,
                scalar_valid,
                policy=FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
            )
            jones = composed.jones
            valid = composed.valid
            off_diagonal_valid = (
                composed.off_diagonal_valid if self.off_diagonal else valid
            )
        elif self.off_diagonal:
            off_diagonal_valid = cassbeam_valid
        else:
            off_diagonal_valid = valid
        return BeamEvaluation(
            jones=jones,
            valid=valid,
            off_diagonal_valid=off_diagonal_valid,
            provenance={
                "model_id": self.model_id,
                "artifact_id": CASSBEAM_CBAND_MODEL_ID,
                "kind": "electromagnetic",
                "support_class": "analytic",
                "array_average": True,
                "jones_axes": list(JONES_AXES),
                "receptors": [receptor.value for receptor in JONES_RECEPTOR_ORDER],
                "receptor_basis": "circular",
                "direction_frame": "sky_direction_cosines",
                "voltage_phase_convention": "cassbeam_transmit_native",
                "on_axis_normalization": (
                    "E(0)=I at CASSBEAM dephased DC after even-N l reflection"
                ),
                "calibration_state": state.value,
                "on_axis_di_jones_order": ON_AXIS_DI_JONES_ORDER,
                "circular_stokes": CIRCULAR_STOKES,
                "circular_p_jones": CIRCULAR_P_JONES,
                "frequency_policy": "nearest_generated_node",
                "max_nearest_node_separation_hz": MAX_NEAREST_NODE_SEPARATION_HZ,
                "off_diagonal": self.off_diagonal,
                "experimental": self.off_diagonal or (not DIAGONAL_COPOLAR_ACCEPTED),
                "casa_awp2_accepted": DIAGONAL_COPOLAR_ACCEPTED,
                "feed_frame_polarization": CASSBEAM_PARALLACTIC_BASIS_LOCK,
                "parallactic_basis": (
                    CASA_PARANG_PARALLACTIC_BASIS
                    if state is BeamCalibrationState.CASA_PARANG_TRUE
                    else UNCALIBRATED_PARALLACTIC_BASIS
                ),
                "pixel_scale": "cassbeam_lambda_over_F_N_dx",
                "outer_field_policy": (
                    FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE.value
                    if self.outer is not None
                    else FullJonesOuterFieldPolicy.UNSUPPORTED.value
                ),
                "selected_window_hz": selected_hz,
                "ignored_coordinates": ["elevation_rad"],
            },
        )


def diagonal_copolar_is_casa_accepted() -> bool:
    """The nominal CASSBEAM diagonal mode is not yet CASA-accepted."""

    return DIAGONAL_COPOLAR_ACCEPTED


def cassbeam_cband_artifact_dir() -> Path:
    """Return the directory that holds the generated C-band tables."""

    return _ARTIFACT_DIR


def cassbeam_frequency_support_hz() -> tuple[float, float]:
    """Return nearest-node support, including the declared separation pad."""

    artifact = load_cassbeam_cband_artifact()
    low = artifact.tables[0].frequency_hz - MAX_NEAREST_NODE_SEPARATION_HZ
    high = artifact.tables[-1].frequency_hz + MAX_NEAREST_NODE_SEPARATION_HZ
    return low, high


@lru_cache(maxsize=1)
def load_cassbeam_cband_artifact() -> CassbeamCBandArtifact:
    """Load the generated tables and refuse to proceed if checksums drift."""

    manifest = json.loads((_ARTIFACT_DIR / _MANIFEST_NAME).read_text(encoding="utf-8"))
    files = manifest["files"]
    for name, digest in files.items():
        payload = (_ARTIFACT_DIR / name).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch for {name}")
    tables = tuple(
        _load_table(frequency_mhz) for frequency_mhz in manifest["frequencies_mhz"]
    )
    low = tables[0].frequency_hz - MAX_NEAREST_NODE_SEPARATION_HZ
    high = tables[-1].frequency_hz + MAX_NEAREST_NODE_SEPARATION_HZ
    pin = FullJonesReferencePin(
        artifact_id=CASSBEAM_CBAND_MODEL_ID,
        generator_or_path=str(manifest["work_directory"]),
        native_quantity="voltage_jones_2x2",
        native_basis="circular",
        receptor_order=JONES_RECEPTOR_ORDER,
        transmit_receive=TransmitReceiveConvention.TRANSMIT,
        direction_axis_orientation=None,
        frequency_support_hz=(low, high),
        direction_support=(
            "CASSBEAM sine-projected Jones raster using "
            "beampixelscale = λ/(F N dx). Origin is the dephased FFT DC "
            "after even-N reflectMatrix2 (l at N/2-1, m at N/2), not the "
            "documented centre row. Fail closed outside the raster. "
            "Frequencies use nearest generated node within "
            f"{MAX_NEAREST_NODE_SEPARATION_HZ / 1e6:.0f} MHz."
        ),
        antenna_averaging=AntennaAveraging.ARRAY_AVERAGE,
        contents=FullJonesContents(
            squint=TermPresence.PRESENT,
            off_diagonal_leakage=TermPresence.PRESENT,
            on_axis_g=TermPresence.ABSENT,
            on_axis_d=TermPresence.ABSENT,
            on_axis_x=TermPresence.ABSENT,
            on_axis_p=TermPresence.ABSENT,
        ),
        outer_field_policy=FullJonesOuterFieldPolicy.TAPERED_SCALAR_COMPOSITE,
        generator_version=str(manifest["package"]),
        input_checksum=str(files["vla-cband-4564.in"]),
        output_checksum=str(files["vla-cband-4564.jones.dat"]),
        frozen=False,
        unpinned_fields=(
            "transmit_to_receive conversion",
            "direction-axis orientation",
            "CASA awp2 acceptance",
        ),
        notes=str(manifest["notes"]),
    )
    return CassbeamCBandArtifact(tables=tables, manifest=manifest, pin=pin)


def voltage_beam_for_mode(
    mode: BeamImagingMode | str,
) -> AnalyticAiryVoltageBeam | CompositeScalarVoltageBeam | CassbeamCBandVoltageBeam:
    """Return the voltage-beam backend for an explicit imaging mode."""

    selected = BeamImagingMode(mode)
    if selected is BeamImagingMode.STATIC_SCALAR:
        return AnalyticAiryVoltageBeam()
    if selected is BeamImagingMode.STREAMED_SCALAR:
        return _scalar_composite()
    artifact = load_cassbeam_cband_artifact()
    if selected is BeamImagingMode.DIAGONAL_COPOLAR:
        return CassbeamCBandVoltageBeam(
            artifact,
            off_diagonal=False,
            outer=_scalar_composite(extend_airy_beyond_cassbeam=True),
        )
    if selected is BeamImagingMode.FULL_JONES:
        require_frozen_full_jones_reference(artifact.pin)
        return CassbeamCBandVoltageBeam(
            artifact,
            off_diagonal=True,
            outer=_scalar_composite(extend_airy_beyond_cassbeam=True),
        )
    raise ValueError(f"unknown beam imaging mode {mode!r}")


def _scalar_composite(
    *, extend_airy_beyond_cassbeam: bool = False
) -> CompositeScalarVoltageBeam:
    catalog = VLABeamCatalog()
    if extend_airy_beyond_cassbeam:
        catalog = VLABeamCatalog(
            airy_max_radius_rad_at_1ghz=CASSBEAM_OUTER_AIRY_MAX_RADIUS_RAD_AT_1GHZ
        )
    return CompositeScalarVoltageBeam(
        main=Perley2016CBandVoltageBeam(),
        outer=AnalyticAiryVoltageBeam(catalog=catalog),
        handover=CompositeHandoverPolicy.MATCH_POWER,
    )


def _load_table(frequency_mhz: int) -> CassbeamCBandTable:
    prefix = f"vla-cband-{frequency_mhz}"
    params = _parse_params(_ARTIFACT_DIR / f"{prefix}.params")
    raw = np.loadtxt(_ARTIFACT_DIR / f"{prefix}.jones.dat", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"{prefix}.jones.dat must have eight columns")
    size = int(np.round(np.sqrt(raw.shape[0])))
    if size * size != raw.shape[0] or size % 2 == 0:
        raise ValueError(f"{prefix}.jones.dat is not an odd square raster")
    jones = np.empty((size, size, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = (raw[:, 0] + 1j * raw[:, 1]).reshape(size, size)
    jones[..., 1, 0] = (raw[:, 2] + 1j * raw[:, 3]).reshape(size, size)
    jones[..., 0, 1] = (raw[:, 4] + 1j * raw[:, 5]).reshape(size, size)
    jones[..., 1, 1] = (raw[:, 6] + 1j * raw[:, 7]).reshape(size, size)
    scale = _cassbeam_pixel_scale_rad(params)
    recorded = params.get("beampixelscale")
    if recorded is not None:
        recorded_rad = np.deg2rad(float(recorded))
        if not np.isclose(recorded_rad, scale, rtol=1e-8, atol=0.0):
            raise ValueError("params beampixelscale disagrees with λ/(F N dx)")
    fwhm_l = np.deg2rad(float(params["fwhm_l"]))
    fwhm_m = np.deg2rad(float(params["fwhm_m"]))
    aperture_n, crop_start = _cassbeam_aperture_and_crop_start(params)
    l_origin, m_origin = _cassbeam_crop_origin_indices(aperture_n, crop_start)
    if size != aperture_n - 2 * crop_start + 1:
        raise ValueError("Jones raster size does not match the CASSBEAM crop")
    return CassbeamCBandTable(
        frequency_hz=float(params["freq"]) * 1.0e9,
        jones=jones,
        l_rad=np.asarray((np.arange(size) - l_origin) * scale, dtype=np.float64),
        m_rad=np.asarray((np.arange(size) - m_origin) * scale, dtype=np.float64),
        fwhm_l_rad=fwhm_l,
        fwhm_m_rad=fwhm_m,
        pixel_scale_rad=scale,
        l_origin_index=l_origin,
        m_origin_index=m_origin,
        params=params,
    )


def _parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("%", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _geom_radius_m() -> float:
    last = None
    for line in (_ARTIFACT_DIR / "vla_geom").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        last = float(stripped.split()[0])
    if last is None or last <= 0.0:
        raise ValueError("vla_geom must end with a positive primary radius")
    return last


def _cassbeam_aperture_and_crop_start(params: dict[str, str]) -> tuple[int, int]:
    """Return CASSBEAM ``N`` and the Jones crop start ``N/4``."""

    gridsize = int(float(params["gridsize"]))
    half = gridsize // 2
    if half % 2 == 1:
        half += 1
    aperture_n = 2 * half
    return aperture_n, aperture_n // 4


def _cassbeam_crop_origin_indices(aperture_n: int, crop_start: int) -> tuple[int, int]:
    """Return crop ``(l, m)`` indices of the dephased FFT DC.

    After ``fftshift``, DC is at ``N/2``. ``reflectMatrix2inplace`` then
    swaps ``M[j][i]`` with ``M[j][2N-1-i]``. On an even grid that moves
    the ``l`` DC from ``N/2`` to ``N/2-1``. ``m`` is not reflected.
    CASSBEAM still labels ``kx=(i-N/2)·factor``, so its documented
    centre row is one ``l`` pixel away from the stored peak.
    """

    dc_l = aperture_n // 2 - 1 if aperture_n % 2 == 0 else aperture_n // 2
    dc_m = aperture_n // 2
    return dc_l - crop_start, dc_m - crop_start


def _cassbeam_pixel_scale_rad(params: dict[str, str]) -> float:
    """Return CASSBEAM ``beampixelscale`` as λ / (F N dx).

    ``compute=jp`` does not write ``beampixelscale`` because that field is
    only stored when Stokes images are saved. The formula is the one used
    in ``calcIllumpolparams``.
    """

    aperture_n, _crop_start = _cassbeam_aperture_and_crop_start(params)
    half = aperture_n // 2
    spacing = _geom_radius_m() / half
    wavelength = SPEED_OF_LIGHT_M_S / (float(params["freq"]) * 1.0e9)
    pixels = float(params["pixelsperbeam"])
    if pixels <= 0.0 or spacing <= 0.0:
        raise ValueError("CASSBEAM grid parameters must be positive")
    return wavelength / (pixels * aperture_n * spacing)


def _nearest_table(
    tables: tuple[CassbeamCBandTable, ...], frequency_hz: float
) -> CassbeamCBandTable:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    selected = min(tables, key=lambda table: abs(table.frequency_hz - frequency_hz))
    if abs(selected.frequency_hz - frequency_hz) > MAX_NEAREST_NODE_SEPARATION_HZ:
        raise ValueError(
            "frequency is more than "
            f"{MAX_NEAREST_NODE_SEPARATION_HZ / 1e6:.0f} MHz from a "
            "generated CASSBEAM node"
        )
    return selected


def _pointing_relative_lm(coordinates: BeamCoordinates) -> tuple[np.ndarray, np.ndarray]:
    pointing = (
        np.zeros(2, dtype=np.float64)
        if coordinates.pointing_offset_lm_rad is None
        else coordinates.pointing_offset_lm_rad
    )
    return coordinates.l_rad - pointing[0], coordinates.m_rad - pointing[1]


def _antenna_frame_lm(
    l_rad: np.ndarray, m_rad: np.ndarray, chi: float
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(chi)
    sine = np.sin(chi)
    return l_rad * cosine + m_rad * sine, -l_rad * sine + m_rad * cosine


def _apply_parallactic_jones(
    jones: np.ndarray, chi: float, state: BeamCalibrationState
) -> np.ndarray:
    """Apply the calibration-state parallactic basis. Signs are unverified."""

    parallactic = circular_parallactic_jones(chi)
    if state is BeamCalibrationState.CASA_PARANG_TRUE:
        conjugate = np.conjugate(np.transpose(parallactic))
        return np.asarray(
            np.einsum("ij,...jk,kl->...il", conjugate, jones, parallactic),
            dtype=np.complex128,
        )
    if state is BeamCalibrationState.UNCALIBRATED:
        return np.asarray(
            np.einsum("...ij,jk->...ik", jones, parallactic),
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported beam calibration state {state!r}")


def _bilinear_jones(
    table: CassbeamCBandTable, l_rad: np.ndarray, m_rad: np.ndarray
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    l_axis = table.l_rad
    m_axis = table.m_rad
    i = np.interp(l_rad, l_axis, np.arange(l_axis.size), left=np.nan, right=np.nan)
    j = np.interp(m_rad, m_axis, np.arange(m_axis.size), left=np.nan, right=np.nan)
    ok = np.isfinite(i) & np.isfinite(j)
    i0 = np.floor(np.where(ok, i, 0.0)).astype(np.int64)
    j0 = np.floor(np.where(ok, j, 0.0)).astype(np.int64)
    i0 = np.clip(i0, 0, l_axis.size - 2)
    j0 = np.clip(j0, 0, m_axis.size - 2)
    di = np.where(ok, i - i0, 0.0)
    dj = np.where(ok, j - j0, 0.0)
    g00 = table.jones[j0, i0]
    g10 = table.jones[j0, i0 + 1]
    g01 = table.jones[j0 + 1, i0]
    g11 = table.jones[j0 + 1, i0 + 1]
    plane = (1.0 - dj)[:, None, None] * (
        (1.0 - di)[:, None, None] * g00 + di[:, None, None] * g10
    ) + dj[:, None, None] * (
        (1.0 - di)[:, None, None] * g01 + di[:, None, None] * g11
    )
    return np.asarray(plane, dtype=np.complex128), np.asarray(ok, dtype=bool)


def _diagonal_only(jones: np.ndarray) -> np.ndarray:
    out = np.zeros_like(jones)
    out[..., 0, 0] = jones[..., 0, 0]
    out[..., 1, 1] = jones[..., 1, 1]
    return out


def _normalize_on_axis(
    jones: np.ndarray,
    table: CassbeamCBandTable,
    *,
    off_diagonal: bool,
) -> np.ndarray:
    center = table.jones[table.m_origin_index, table.l_origin_index]
    if off_diagonal:
        return np.asarray(
            np.einsum("ij,...jk->...ik", np.linalg.inv(center), jones),
            dtype=np.complex128,
        )
    scale_r = center[0, 0]
    scale_l = center[1, 1]
    out = np.zeros_like(jones)
    out[..., 0, 0] = jones[..., 0, 0] / scale_r
    out[..., 1, 1] = jones[..., 1, 1] / scale_l
    return out


def _broadcast_antenna_planes(
    jones: np.ndarray, valid: np.ndarray, antenna_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if jones.shape[0] == antenna_count:
        return np.asarray(jones), np.asarray(valid)
    if jones.shape[0] != 1:
        raise ValueError("scalar outer beam must have one antenna plane")
    return (
        np.broadcast_to(jones, (antenna_count, *jones.shape[1:])).copy(),
        np.broadcast_to(valid, (antenna_count, *valid.shape[1:])).copy(),
    )


def cassbeam_fwhm_versus_nrao(frequency_hz: float) -> float:
    """Return CASSBEAM mean FWHM divided by the NRAO ``42/ν`` Gaussian."""

    table = _nearest_table(load_cassbeam_cband_artifact().tables, frequency_hz)
    cassbeam_arcmin = (
        0.5 * (np.rad2deg(table.fwhm_l_rad) + np.rad2deg(table.fwhm_m_rad)) * 60.0
    )
    nrao = float(np.asarray(nrao_gaussian_fwhm_arcmin(frequency_hz)))
    return float(cassbeam_arcmin / nrao)


def cassbeam_receptor_mainlobe_separation_arcmin(frequency_hz: float) -> float:
    """Return the RR/LL main-lobe power-centroid separation in arcmin.

    Each receptor uses pixels above ``MAINLOBE_POWER_FRACTION`` of its
    own peak. Full-raster centroids mix sidelobes and are not a squint
    oracle.
    """

    table = _nearest_table(load_cassbeam_cband_artifact().tables, frequency_hz)
    rr = np.abs(table.jones[..., 0, 0]) ** 2
    ll = np.abs(table.jones[..., 1, 1]) ** 2
    rr_l, rr_m = _mainlobe_centroid_rad(rr, table.l_rad, table.m_rad)
    ll_l, ll_m = _mainlobe_centroid_rad(ll, table.l_rad, table.m_rad)
    return float(np.rad2deg(np.hypot(rr_l - ll_l, rr_m - ll_m)) * 60.0)


def cassbeam_common_mode_offset_arcmin(frequency_hz: float) -> tuple[float, float]:
    """Return the mean co-polar main-lobe centroid in arcmin ``(l, m)``."""

    table = _nearest_table(load_cassbeam_cband_artifact().tables, frequency_hz)
    power = 0.5 * (
        np.abs(table.jones[..., 0, 0]) ** 2 + np.abs(table.jones[..., 1, 1]) ** 2
    )
    l_rad, m_rad = _mainlobe_centroid_rad(power, table.l_rad, table.m_rad)
    return (
        float(np.rad2deg(l_rad) * 60.0),
        float(np.rad2deg(m_rad) * 60.0),
    )


def _mainlobe_centroid_rad(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> tuple[float, float]:
    peak = float(np.max(power))
    if peak <= 0.0:
        raise ValueError("power map has no positive peak")
    masked = np.where(power >= MAINLOBE_POWER_FRACTION * peak, power, 0.0)
    return _power_centroid_rad(masked, l_rad, m_rad)


def _power_centroid_rad(
    power: np.ndarray, l_rad: np.ndarray, m_rad: np.ndarray
) -> tuple[float, float]:
    weights = np.asarray(power, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("power map has no positive weight")
    weights = weights / total
    return (
        float(np.sum(weights * l_rad[None, :])),
        float(np.sum(weights * m_rad[:, None])),
    )


def observation_3c391_is_inside_cassbeam_nearest_support() -> bool:
    """True if the 3C391 SPW0 span lies inside the nearest-node pad."""

    low, high = cassbeam_frequency_support_hz()
    start, stop = OBSERVATION_3C391_FREQUENCY_HZ
    return low <= start and stop <= high
