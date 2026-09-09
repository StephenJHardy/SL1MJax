"""VLA holography inventory and per-antenna pointing.

The archive Measurement Set is not required to construct or test these
types. Inventory of a real product fails closed until the files exist.
Do not flatten holography into mosaic fields. Do not store pointing
arrays on ``BeamOperatorConfig``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.beam_conventions import BeamCalibrationState, require_beam_calibration_state
from sl1mjax.beam_operator import (
    BeamOperatorConfig,
    BeamOperatorResult,
    SkyStokesPlanes,
    predict_voltage_beam,
    unique_visibility_times,
)
from sl1mjax.coordinates import lmn_to_radec, radec_to_lmn
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import (
    Correlation,
    ReceptorBasis,
    apply_jones_to_coherency,
    circular_stokes_to_coherency,
    pack_coherency,
    receptors_for_correlations,
    unpack_coherency,
)
from sl1mjax.rime import SPEED_OF_LIGHT_M_S
from sl1mjax.voltage_beam import VoltageBeamModel

HOLOGRAPHY_INVENTORY_SCHEMA_VERSION = 1
HOLOGRAPHY_POINTING_SCHEMA_VERSION = 1
HOLOGRAPHY_SOURCE_SCHEMA_VERSION = 1
HOLOGRAPHY_METADATA_SCHEMA_VERSION = 1
HOLOGRAPHY_ARCHIVE_ROOT = Path("data/holography")
SYNTHETIC_HOLOGRAPHY_FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "data" / "holography_synthetic"
)
CASA_CORRELATION_CODES = {"RR": 5, "RL": 6, "LR": 7, "LL": 8}
CASA_CORRELATION_FROM_CODE = {code: name for name, code in CASA_CORRELATION_CODES.items()}
THOL0001_PROJECT = "THOL0001"
THOL0001_SCHEDULING_BLOCK = "sb31628704"
THOL0001_EXECUTION_BLOCK = "eb31629959"
THOL0001_SOURCE = "3C147"
THOL0001_MEMO_PRODUCT = "CHOLO-LO"
THOL0001_HOLORASTER_FIELD = "HOLORASTER"
THOL0001_LOWER_C_NATIVE_HZ = (4.564e9, 4.692e9)
_EXECUTION_PATH_RE = re.compile(
    r"(?P<project>[A-Za-z0-9_-]+)\.sb(?P<sb>\d+)\.eb(?P<eb>\d+)",
    re.IGNORECASE,
)
REQUIRED_HOLOGRAPHY_TABLES = (
    "POINTING",
    "SOURCE",
    "FIELD",
    "STATE",
    "ANTENNA",
    "FEED",
)
OPTIONAL_HOLOGRAPHY_TABLES = ("WEATHER", "SYSCAL", "SYSPOWER", "CALDEVICE")
POINTING_DIRECTION_COLUMNS = ("DIRECTION", "TARGET", "POINTING_OFFSET")
MOUNT_OFFSET_REFS = frozenset({"AZELGEO", "AZEL", "AZELG", "AZELSW", "AZELNE", "SIN"})
JoinRule = Literal["interval", "nearest_within_tolerance"]
OffsetSign = Literal["commanded_pointing", "source_minus_pointing"]
HolographyGateStatus = Literal["pass", "warn", "fail", "not_run"]
HolographySourceKind = Literal["point", "compact_component", "resolved_coherency"]
HolographyCoordinateFrame = Literal[
    "phase_centre_sky",
    "commanded_pointing",
    "source_relative",
    "feed_frame",
]
PERLEY_BUTLER_2017_3C147_COEFFICIENTS = (
    1.4516,
    -0.6961,
    -0.2007,
    0.0640,
    -0.0464,
    0.0289,
)
PERLEY_BUTLER_2017_3C147_FREQUENCY_HZ = (5.0e7, 5.0e10)
PERLEY_BUTLER_2017_3C147_CITATION = "Perley & Butler 2017, ApJS 230, 7, Table 5"
CASA_SETJY_PERLEY_BUTLER_2017 = "Perley-Butler 2017"
CASA_SETJY_3C147_C_IM = "3C147_C.im"
THOL0001_SPW4_CHANNEL_32_HZ = 4.564e9
PERLEY_BUTLER_2017_3C147_TABLE5_JY_AT_4P564GHZ = 8.290006
CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32 = 8.028518676757812
CASA_SETJY_3C147_C_IM_FLUXD_JY_SPW4 = 8.141902923583984
THREE_C147_CASA_SETJY_VS_TABLE5_NOTE = (
    "The 3.2% gap is not one number. CASA 6.7.6 setjy Perley-Butler 2017 "
    "3C147_C.im returns epoch-dependent integrated fluxd 8.1419 Jy at the "
    "SPW 4 reference frequency. The resolved image produces a median sampled "
    "HOLORASTER MODEL_DATA of 8.031 Jy at 4.564 GHz (field-0 channel 32 is "
    "8.0285 Jy). Table 5 is 8.290 Jy at 4.564 GHz. Comparing a sampled "
    "visibility directly with Table 5 mixes temporal flux treatment and "
    "resolved structure. Scientific G uses the CASA setjy image; Table 5 "
    "remains the SL1MJax analytic scale."
)
MODEL_DATA_SOURCE_COHERENCY_NOTE = (
    "HOLORASTER recovery uses per-row field-10 MODEL_DATA as S_pq. "
    "Unpack RR/RL/LR/LL into the 2x2 coherency and pass it as "
    "source_coherency_visibility. CASA setjy fluxd is integrated-flux "
    "provenance only. Do not divide every baseline by one Stokes I."
)


class AntennaPointingRole(StrEnum):
    REFERENCE = "reference"
    MOVING = "moving"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class HolographyRowReason(StrEnum):
    OK = "ok"
    MISSING_POINTING = "missing_pointing"
    UNSETTLED = "unsettled"
    NOT_MOVING_REFERENCE = "not_moving_reference"
    NOT_MAP_ANTENNA_SURFACE = "not_map_antenna_surface"
    DWELL_JUMP = "dwell_jump"
    TRANSITION_GUARD = "transition_guard"
    MIXED_STATE = "mixed_state"


class HolographyHoldoutAxis(StrEnum):
    SPATIAL = "spatial"
    REFERENCE_ANTENNA = "reference_antenna"
    MOVING_ANTENNA = "moving_antenna"
    TIME = "time"
    FREQUENCY = "frequency"
    PARALLACTIC_ANGLE = "parallactic_angle"
    CORRELATION = "correlation"


@dataclass(frozen=True)
class HolographySourceComponent:
    """One compact source component relative to the correlator phase centre."""

    stokes_i: float
    l_rad: float = 0.0
    m_rad: float = 0.0
    stokes_q: float = 0.0
    stokes_u: float = 0.0
    stokes_v: float = 0.0


@dataclass(frozen=True)
class HolographySourceModel:
    """3C147 coherency model used as ``S_pq`` in the holography RIME.

    A point model is a special case. Compact components already include the
    Fourier factor in ``S_pq``. The beam operator must not apply it again.
    Do not assign baseline-dependent source visibility to the beam.
    """

    name: str
    kind: HolographySourceKind
    standard: str
    frequency_hz: NDArray[np.float64]
    stokes_i_jy: NDArray[np.float64]
    components: tuple[HolographySourceComponent, ...] = ()
    resolved_coherency: NDArray[np.complex128] | None = None
    schema_version: int = HOLOGRAPHY_SOURCE_SCHEMA_VERSION
    citation: str = PERLEY_BUTLER_2017_3C147_CITATION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_SOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported holography source-model schema")
        if self.kind not in {"point", "compact_component", "resolved_coherency"}:
            raise ValueError(f"unsupported holography source kind {self.kind!r}")
        frequencies = np.asarray(self.frequency_hz, dtype=np.float64).reshape(-1)
        intensity = np.asarray(self.stokes_i_jy, dtype=np.float64).reshape(-1)
        if frequencies.size == 0:
            raise ValueError("source model needs at least one frequency")
        if intensity.shape != frequencies.shape:
            raise ValueError("stokes_i_jy must match frequency_hz")
        if np.any(frequencies <= 0.0) or np.any(intensity <= 0.0):
            raise ValueError("source frequencies and Stokes I must be positive")
        object.__setattr__(self, "frequency_hz", frequencies)
        object.__setattr__(self, "stokes_i_jy", intensity)
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.kind == "compact_component" and not self.components:
            raise ValueError("compact_component source model needs components")
        if self.kind == "resolved_coherency":
            if self.resolved_coherency is None:
                raise ValueError("resolved_coherency source model needs the S_pq array")
            source = np.asarray(self.resolved_coherency, dtype=np.complex128)
            if source.ndim != 4 or source.shape[-2:] != (2, 2):
                raise ValueError("resolved_coherency must have shape (row, channel, 2, 2)")
            object.__setattr__(self, "resolved_coherency", source)

    def evaluate_coherency(self, block: VisibilityBlock) -> NDArray[np.complex128]:
        """Return ``S_pq`` on the block. Fourier is included for compact components."""

        intensity = _interpolate_source_stokes_i(
            self.frequency_hz, self.stokes_i_jy, block.frequency_hz
        )
        n_row = int(block.time_s.shape[0])
        n_chan = int(block.frequency_hz.size)
        if self.kind == "resolved_coherency":
            source = np.asarray(self.resolved_coherency, dtype=np.complex128)
            if source.shape != (n_row, n_chan, 2, 2):
                raise ValueError(f"resolved_coherency must have shape {(n_row, n_chan, 2, 2)}")
            return source
        if self.kind == "point":
            plane = circular_stokes_to_coherency(intensity, 0.0, 0.0, 0.0)
            return np.broadcast_to(plane[None, :, :, :], (n_row, n_chan, 2, 2)).copy()
        coherency = np.zeros((n_row, n_chan, 2, 2), dtype=np.complex128)
        scale = intensity / float(np.sum([component.stokes_i for component in self.components]))
        for component in self.components:
            plane = circular_stokes_to_coherency(
                component.stokes_i * scale,
                component.stokes_q * scale,
                component.stokes_u * scale,
                component.stokes_v * scale,
            )
            kernel = _compact_source_fourier(block, component.l_rad, component.m_rad)
            coherency += plane[None, :, :, :] * kernel[:, :, None, None]
        return coherency


def perley_butler_2017_3c147_stokes_i_jy(frequency_hz: ArrayLike) -> NDArray[np.float64]:
    """Stokes I of 3C147 from Perley & Butler 2017 Table 5."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    low, high = PERLEY_BUTLER_2017_3C147_FREQUENCY_HZ
    if np.any(frequency < low) or np.any(frequency > high):
        raise ValueError(
            f"frequency is outside Perley-Butler 2017 3C147 support [{low:.3e}, {high:.3e}] Hz"
        )
    log_nu = np.log10(frequency / 1.0e9)
    log_s = np.zeros(frequency.shape, dtype=np.float64)
    for power, coeff in enumerate(PERLEY_BUTLER_2017_3C147_COEFFICIENTS):
        log_s = log_s + float(coeff) * log_nu**power
    return np.asarray(10.0**log_s, dtype=np.float64)


def three_c147_flux_scale_report(frequency_hz: ArrayLike) -> dict[str, object]:
    """Record Table 5 and the CASA setjy 3C147_C.im result at one frequency."""

    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    table5 = perley_butler_2017_3c147_stokes_i_jy(frequency)
    casa_model_data = np.full(
        frequency.shape,
        float("nan"),
        dtype=np.float64,
    )
    native = np.isclose(frequency, THOL0001_SPW4_CHANNEL_32_HZ, rtol=0.0, atol=0.5e6)
    casa_model_data[native] = CASA_SETJY_3C147_C_IM_MODEL_DATA_JY_SPW4_CH32
    ratio = casa_model_data / table5
    return {
        "standard": CASA_SETJY_PERLEY_BUTLER_2017,
        "table5_citation": PERLEY_BUTLER_2017_3C147_CITATION,
        "table5_coefficients": list(PERLEY_BUTLER_2017_3C147_COEFFICIENTS),
        "casa_setjy_model": CASA_SETJY_3C147_C_IM,
        "casa_setjy_scalebychan": True,
        "frequency_hz": table5.size and frequency.tolist(),
        "table5_stokes_i_jy": table5.tolist(),
        "casa_setjy_model_data_jy": casa_model_data.tolist(),
        "casa_over_table5": ratio.tolist(),
        "relative_beam_shape_harmless": True,
        "absolute_flux_gate_material": True,
        "scientific_g_uses": "casa_setjy_3c147_c_im",
        "notes": (THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,),
    }


def circular_visibility_to_source_coherency(
    visibility: ArrayLike,
    correlations: tuple[Correlation, ...],
) -> NDArray[np.complex128]:
    """Unpack packed RR/RL/LR/LL visibilities into per-row ``S_pq``.

    CASA ``MODEL_DATA`` arrives as ``(row, channel, correlation)``. The
    holography RIME wants ``(row, channel, 2, 2)``. No Fourier factor is
    added: the Measurement Set model already includes source structure.
    """

    vis = np.asarray(visibility, dtype=np.complex128)
    if vis.ndim == 2:
        vis = vis[:, None, :]
    if vis.ndim != 3:
        raise ValueError("visibility must have shape (row, channel, correlation)")
    if vis.shape[-1] != len(correlations):
        raise ValueError("visibility correlation axis must match correlations")
    return pack_coherency(vis, correlations, receptors_for_correlations(correlations))


def casa_setjy_fluxd_jy(
    record: Mapping[str, Any],
    *,
    field_id: int,
    spectral_window_id: int,
) -> float:
    """Read CASA setjy ``fluxd[I]`` for one field and spectral window."""

    node: Any = record
    if isinstance(node, Mapping) and "fields" in node:
        fields = node["fields"]
        if not isinstance(fields, Mapping):
            raise ValueError("setjy record fields must be a mapping")
        node = fields.get(str(int(field_id)), fields)
    if isinstance(node, Mapping) and str(int(field_id)) in node:
        inner = node[str(int(field_id))]
        if isinstance(inner, Mapping) and (
            str(int(spectral_window_id)) in inner or "fluxd" in inner
        ):
            node = inner
    if not isinstance(node, Mapping):
        raise ValueError("setjy record does not contain a field mapping")
    window = node.get(str(int(spectral_window_id)))
    if not isinstance(window, Mapping) or "fluxd" not in window:
        raise ValueError(f"setjy record has no fluxd for field {field_id} SPW {spectral_window_id}")
    fluxd = np.asarray(window["fluxd"], dtype=np.float64).reshape(-1)
    if fluxd.size == 0:
        raise ValueError("setjy fluxd is empty")
    return float(fluxd[0])


def three_c147_casa_setjy_point_source_model(
    frequency_hz: ArrayLike,
    stokes_i_jy: ArrayLike,
) -> HolographySourceModel:
    """Point model on the CASA setjy 3C147_C.im scale used for scientific G."""

    frequencies = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    intensity = np.asarray(stokes_i_jy, dtype=np.float64).reshape(-1)
    if intensity.shape != frequencies.shape:
        raise ValueError("stokes_i_jy must match frequency_hz")
    return HolographySourceModel(
        name=THOL0001_SOURCE,
        kind="point",
        standard=f"{CASA_SETJY_PERLEY_BUTLER_2017} {CASA_SETJY_3C147_C_IM}",
        frequency_hz=frequencies,
        stokes_i_jy=intensity,
        citation=(
            "CASA 6.7.6 setjy Perley-Butler 2017 3C147_C.im; " + PERLEY_BUTLER_2017_3C147_CITATION
        ),
        notes=(
            "Scientific holography G is solved against this CASA setjy model",
            THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,
            "circular coherency uses the software I,Q,U,V packing, not I/2",
        ),
    )


def three_c147_point_source_model(frequency_hz: ArrayLike) -> HolographySourceModel:
    """Versioned unpolarised point-source model of 3C147."""

    frequencies = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    return HolographySourceModel(
        name=THOL0001_SOURCE,
        kind="point",
        standard="Perley-Butler 2017",
        frequency_hz=frequencies,
        stokes_i_jy=perley_butler_2017_3c147_stokes_i_jy(frequencies),
        notes=(
            "3C147 is known or suspected variable on multi-year timescales",
            "point model; test baseline dependence before assigning structure to the beam",
            "circular coherency uses the software I,Q,U,V packing, not I/2",
            THREE_C147_CASA_SETJY_VS_TABLE5_NOTE,
        ),
    )


def load_three_c147_source_model(
    frequency_hz: ArrayLike,
    root: Path | None = None,
) -> HolographySourceModel:
    """Load the committed 3C147 source-model artifact and evaluate at ``frequency_hz``."""

    path = (SYNTHETIC_HOLOGRAPHY_FIXTURE_ROOT if root is None else Path(root)) / (
        "three_c147_source_model.json"
    )
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != HOLOGRAPHY_SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported 3C147 source-model artifact schema")
    if str(payload.get("kind")) != "point":
        raise ValueError("committed 3C147 artifact is the point-source model")
    model = three_c147_point_source_model(frequency_hz)
    stored = tuple(float(value) for value in payload["coefficients"])
    if stored != PERLEY_BUTLER_2017_3C147_COEFFICIENTS:
        raise ValueError("committed 3C147 coefficients do not match the implementation")
    return replace(model, citation=str(payload.get("citation", model.citation)))


def _interpolate_source_stokes_i(
    model_frequency_hz: np.ndarray,
    stokes_i_jy: np.ndarray,
    frequency_hz: np.ndarray,
) -> NDArray[np.float64]:
    if model_frequency_hz.size == 1:
        return np.full(frequency_hz.shape, float(stokes_i_jy[0]), dtype=np.float64)
    order = np.argsort(model_frequency_hz)
    return np.interp(
        frequency_hz,
        model_frequency_hz[order],
        stokes_i_jy[order],
        left=float(stokes_i_jy[order[0]]),
        right=float(stokes_i_jy[order[-1]]),
    )


def _compact_source_fourier(
    block: VisibilityBlock,
    l_rad: float,
    m_rad: float,
) -> NDArray[np.complex128]:
    if l_rad == 0.0 and m_rad == 0.0:
        return np.ones((block.time_s.shape[0], block.frequency_hz.size), dtype=np.complex128)
    n_rad = float(np.sqrt(max(1.0 - l_rad * l_rad - m_rad * m_rad, 0.0)))
    uvw = block.uvw_m[:, None, :] * block.frequency_hz[None, :, None] / SPEED_OF_LIGHT_M_S
    phase = 2j * np.pi * (uvw[..., 0] * l_rad + uvw[..., 1] * m_rad + uvw[..., 2] * (n_rad - 1.0))
    return np.exp(phase)


@dataclass(frozen=True)
class Memo195LowerCRaster:
    """Expected THOL0001 raster geometry from EVLA Memo 195."""

    dense_n: int = 17
    dense_spacing_arcmin: float = 1.72
    sparse_n: int = 23
    sparse_spacing_arcmin: float = 4.59

    @property
    def dense_radius_arcmin(self) -> float:
        return 0.5 * (self.dense_n - 1) * self.dense_spacing_arcmin

    @property
    def sparse_radius_arcmin(self) -> float:
        return 0.5 * (self.sparse_n - 1) * self.sparse_spacing_arcmin


@dataclass(frozen=True)
class DirectionMeasure:
    """One POINTING direction column with its measure metadata."""

    name: str
    values_rad: NDArray[np.float64]
    measure_ref: str
    units: str

    def __post_init__(self) -> None:
        if self.name not in POINTING_DIRECTION_COLUMNS:
            raise ValueError(f"unsupported pointing column {self.name!r}")
        if not str(self.measure_ref).strip():
            raise ValueError(f"{self.name} measure reference cannot be omitted")
        if not str(self.units).strip():
            raise ValueError(f"{self.name} units cannot be omitted")
        values = np.asarray(self.values_rad, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"{self.name} must have shape (row, 2)")
        object.__setattr__(self, "values_rad", values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values_rad": self.values_rad.tolist(),
            "measure_ref": self.measure_ref,
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, payload: MappingLike) -> DirectionMeasure:
        return cls(
            name=str(payload["name"]),
            values_rad=np.asarray(payload["values_rad"], dtype=np.float64),
            measure_ref=str(payload["measure_ref"]),
            units=str(payload["units"]),
        )


MappingLike = dict[str, Any]


@dataclass(frozen=True)
class HolographyArchiveInventory:
    """Immutable record of one downloaded holography execution."""

    project: str
    scheduling_block: str
    execution_block: str
    source_name: str
    observation_utc: str
    path: str
    archive_sha256: str
    table_presence: dict[str, bool]
    correlations: tuple[str, ...]
    antenna_count: int
    schema_version: int = HOLOGRAPHY_INVENTORY_SCHEMA_VERSION
    memo_product: str = THOL0001_MEMO_PRODUCT
    notes: tuple[str, ...] = ()
    correlation_codes: tuple[int, ...] = ()
    antenna_ids: tuple[int, ...] = ()
    antenna_names: tuple[str, ...] = ()
    antenna_position_m: tuple[tuple[float, float, float], ...] = ()
    time_is_monotonic: bool = True
    frequency_is_monotonic: bool = True
    main_row_count: int = 0
    main_columns: tuple[str, ...] = ()
    has_data: bool = False
    has_corrected_data: bool = False
    field_names: tuple[str, ...] = ()
    state_modes: tuple[str, ...] = ()
    pointing_measure_refs: tuple[tuple[str, str], ...] = ()
    frequency_hz_min: float = 0.0
    frequency_hz_max: float = 0.0
    n_spw: int = 0
    n_scan: int = 0
    observation_project: str = ""

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_INVENTORY_SCHEMA_VERSION:
            raise ValueError("unsupported holography inventory schema")
        if self.antenna_count < 1:
            raise ValueError("holography inventory needs at least one antenna")
        if not self.archive_sha256:
            raise ValueError("holography inventory requires an archive hash")
        object.__setattr__(self, "table_presence", dict(self.table_presence))
        object.__setattr__(self, "correlations", tuple(self.correlations))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(
            self, "correlation_codes", tuple(int(code) for code in self.correlation_codes)
        )
        object.__setattr__(self, "antenna_ids", tuple(int(antenna) for antenna in self.antenna_ids))
        object.__setattr__(self, "antenna_names", tuple(str(name) for name in self.antenna_names))
        object.__setattr__(
            self,
            "antenna_position_m",
            tuple(
                (float(position[0]), float(position[1]), float(position[2]))
                for position in self.antenna_position_m
            ),
        )
        object.__setattr__(self, "main_columns", tuple(self.main_columns))
        object.__setattr__(self, "field_names", tuple(self.field_names))
        object.__setattr__(self, "state_modes", tuple(self.state_modes))
        object.__setattr__(
            self,
            "pointing_measure_refs",
            tuple((str(name), str(ref)) for name, ref in self.pointing_measure_refs),
        )

    def pointing_is_present(self) -> bool:
        return bool(self.table_presence.get("POINTING"))

    def antennas_resolve(self) -> bool:
        """True when every recorded antenna ID has a name and ITRF position."""

        n_ant = int(self.antenna_count)
        return (
            len(self.antenna_ids) == n_ant
            and len(self.antenna_names) == n_ant
            and len(self.antenna_position_m) == n_ant
            and len(set(self.antenna_ids)) == n_ant
            and all(name.strip() for name in self.antenna_names)
        )


@dataclass(frozen=True)
class AntennaPointingTable:
    """Lossless POINTING rows. No column is selected as the raster here."""

    time_s: NDArray[np.float64]
    interval_s: NDArray[np.float64]
    antenna_id: NDArray[np.int32]
    columns: tuple[DirectionMeasure, ...]
    tracking: NDArray[np.bool_] | None
    on_source: NDArray[np.bool_] | None
    row_id: NDArray[np.int32]
    source_ms: str
    table_sha256: str
    interpolation_flag: NDArray[np.bool_]
    schema_version: int = HOLOGRAPHY_POINTING_SCHEMA_VERSION
    time_is_midpoint: bool = True

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_POINTING_SCHEMA_VERSION:
            raise ValueError("unsupported holography pointing schema")
        time_s = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        interval_s = np.asarray(self.interval_s, dtype=np.float64).reshape(-1)
        antenna_id = np.asarray(self.antenna_id, dtype=np.int32).reshape(-1)
        row_id = np.asarray(self.row_id, dtype=np.int32).reshape(-1)
        interp = np.asarray(self.interpolation_flag, dtype=bool).reshape(-1)
        n_row = int(time_s.size)
        if n_row == 0:
            raise ValueError("AntennaPointingTable needs at least one row")
        for name, array in (
            ("interval_s", interval_s),
            ("antenna_id", antenna_id),
            ("row_id", row_id),
            ("interpolation_flag", interp),
        ):
            if int(array.size) != n_row:
                raise ValueError(f"{name} must have one value per pointing row")
        if not self.columns:
            raise ValueError("AntennaPointingTable must preserve at least one direction column")
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate POINTING direction columns")
        for column in self.columns:
            if column.values_rad.shape[0] != n_row:
                raise ValueError(f"{column.name} row count does not match TIME")
        for optional, label in ((self.tracking, "tracking"), (self.on_source, "on_source")):
            if optional is None:
                continue
            values = np.asarray(optional, dtype=bool).reshape(-1)
            if values.size != n_row:
                raise ValueError(f"{label} must have one value per pointing row")
            object.__setattr__(self, label, values)
        if not str(self.table_sha256).strip():
            raise ValueError("pointing table hash cannot be omitted")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "interval_s", interval_s)
        object.__setattr__(self, "antenna_id", antenna_id)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "interpolation_flag", interp)
        object.__setattr__(self, "columns", tuple(self.columns))

    def column(self, name: str) -> DirectionMeasure:
        for item in self.columns:
            if item.name == name:
                return item
        raise KeyError(f"POINTING column {name!r} was not preserved")


@dataclass(frozen=True)
class ResolvedAntennaPointing:
    """Pointing offsets on the visibility operator's unique-time grid."""

    unique_time_s: NDArray[np.float64]
    antenna_id: NDArray[np.int32]
    offset_lm_rad: NDArray[np.float64]
    valid: NDArray[np.bool_]
    settled: NDArray[np.bool_]
    role: NDArray[np.str_]
    selected_column: str
    offset_sign: OffsetSign
    join_rule: JoinRule
    join_tolerance_s: float
    measure_ref: str
    units: str
    schema_version: int = HOLOGRAPHY_POINTING_SCHEMA_VERSION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        times = np.asarray(self.unique_time_s, dtype=np.float64).reshape(-1)
        antennas = np.asarray(self.antenna_id, dtype=np.int32).reshape(-1)
        offsets = np.asarray(self.offset_lm_rad, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        settled = np.asarray(self.settled, dtype=bool)
        role = np.asarray(self.role, dtype="U16")
        n_time = int(times.size)
        n_ant = int(antennas.size)
        if n_time == 0 or n_ant == 0:
            raise ValueError("ResolvedAntennaPointing needs times and antennas")
        if offsets.shape != (n_time, n_ant, 2):
            raise ValueError("offset_lm_rad must have shape (time, antenna, 2)")
        if valid.shape != (n_time, n_ant) or settled.shape != (n_time, n_ant):
            raise ValueError("valid and settled must have shape (time, antenna)")
        if role.shape != (n_time, n_ant):
            raise ValueError("role must have shape (time, antenna)")
        if self.selected_column not in POINTING_DIRECTION_COLUMNS:
            raise ValueError("resolved pointing must name the selected column")
        if self.join_tolerance_s < 0.0:
            raise ValueError("join_tolerance_s cannot be negative")
        object.__setattr__(self, "unique_time_s", times)
        object.__setattr__(self, "antenna_id", antennas)
        object.__setattr__(self, "offset_lm_rad", offsets)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "settled", settled)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "notes", tuple(self.notes))

    def offset_for_time(self, time_s: float) -> NDArray[np.float64]:
        matches = np.nonzero(self.unique_time_s == float(time_s))[0]
        if matches.size != 1:
            raise KeyError(f"no unique pointing time {time_s}")
        return np.asarray(self.offset_lm_rad[int(matches[0])], dtype=np.float64)


def holography_archive_root(root: Path | None = None) -> Path:
    return HOLOGRAPHY_ARCHIVE_ROOT if root is None else Path(root)


def holography_archive_is_present(root: Path | None = None) -> bool:
    """True only when a downloaded holography product exists on disk."""

    path = holography_archive_root(root)
    if not path.exists():
        return False
    return any(path.rglob("*.ms")) or any(path.iterdir())


def require_holography_archive(root: Path | None = None) -> Path:
    path = holography_archive_root(root)
    if not holography_archive_is_present(path):
        raise FileNotFoundError(
            "holography archive is not present; inventory refuses to invent "
            f"THOL0001 products under {path}"
        )
    return path


def identities_from_holography_path(*paths: Path | str | None) -> dict[str, str]:
    """Parse project, SB, and EB from a VLA execution path. Not OBSERVATION.PROJECT."""

    for path in paths:
        if path is None:
            continue
        text = str(path)
        match = _EXECUTION_PATH_RE.search(Path(text).name) or _EXECUTION_PATH_RE.search(text)
        if match is None:
            continue
        return {
            "project": match.group("project"),
            "scheduling_block": f"sb{match.group('sb')}",
            "execution_block": f"eb{match.group('eb')}",
        }
    return {}


def holography_execution_provenance(
    *paths: Path | str | None,
    observation_project: str = "",
    fallback_project: str = THOL0001_PROJECT,
) -> dict[str, str]:
    """Keep path-derived SB/EB separate from the OBSERVATION.PROJECT UID."""

    identities = identities_from_holography_path(*paths)
    project = identities.get("project") or fallback_project
    return {
        "project": project,
        "scheduling_block": identities.get("scheduling_block", ""),
        "execution_block": identities.get("execution_block", ""),
        "observation_project": str(observation_project),
    }


def inventory_as_dict(inventory: HolographyArchiveInventory) -> dict[str, Any]:
    payload = asdict(inventory)
    payload["correlations"] = list(inventory.correlations)
    payload["notes"] = list(inventory.notes)
    payload["correlation_codes"] = list(inventory.correlation_codes)
    payload["antenna_ids"] = list(inventory.antenna_ids)
    payload["antenna_names"] = list(inventory.antenna_names)
    payload["antenna_position_m"] = [list(position) for position in inventory.antenna_position_m]
    payload["main_columns"] = list(inventory.main_columns)
    payload["field_names"] = list(inventory.field_names)
    payload["state_modes"] = list(inventory.state_modes)
    payload["pointing_measure_refs"] = [list(item) for item in inventory.pointing_measure_refs]
    return payload


def inventory_from_dict(payload: MappingLike) -> HolographyArchiveInventory:
    return HolographyArchiveInventory(
        project=str(payload["project"]),
        scheduling_block=str(payload["scheduling_block"]),
        execution_block=str(payload["execution_block"]),
        source_name=str(payload["source_name"]),
        observation_utc=str(payload["observation_utc"]),
        path=str(payload["path"]),
        archive_sha256=str(payload["archive_sha256"]),
        table_presence=dict(payload["table_presence"]),
        correlations=tuple(payload["correlations"]),
        antenna_count=int(payload["antenna_count"]),
        schema_version=int(payload.get("schema_version", HOLOGRAPHY_INVENTORY_SCHEMA_VERSION)),
        memo_product=str(payload.get("memo_product", THOL0001_MEMO_PRODUCT)),
        notes=tuple(payload.get("notes", ())),
        correlation_codes=tuple(payload.get("correlation_codes", ())),
        antenna_ids=tuple(payload.get("antenna_ids", ())),
        antenna_names=tuple(payload.get("antenna_names", ())),
        antenna_position_m=tuple(
            (float(position[0]), float(position[1]), float(position[2]))
            for position in payload.get("antenna_position_m", ())
        ),
        time_is_monotonic=bool(payload.get("time_is_monotonic", True)),
        frequency_is_monotonic=bool(payload.get("frequency_is_monotonic", True)),
        main_row_count=int(payload.get("main_row_count", 0)),
        main_columns=tuple(payload.get("main_columns", ())),
        has_data=bool(payload.get("has_data", False)),
        has_corrected_data=bool(payload.get("has_corrected_data", False)),
        field_names=tuple(payload.get("field_names", ())),
        state_modes=tuple(payload.get("state_modes", ())),
        pointing_measure_refs=tuple(
            (str(item[0]), str(item[1])) for item in payload.get("pointing_measure_refs", ())
        ),
        frequency_hz_min=float(payload.get("frequency_hz_min", 0.0)),
        frequency_hz_max=float(payload.get("frequency_hz_max", 0.0)),
        n_spw=int(payload.get("n_spw", 0)),
        n_scan=int(payload.get("n_scan", 0)),
        observation_project=str(payload.get("observation_project", "")),
    )


def inventory_measurement_set(
    path: Path,
    *,
    archive_path: Path | None = None,
) -> HolographyArchiveInventory:
    """Inventory a holography Measurement Set.

    A casacore MS is read with casacore. Synthetic fixtures keep the
    committed text-table path. Missing ``POINTING`` is recorded, not repaired.
    """

    measurement_set = Path(path)
    if not measurement_set.exists():
        raise FileNotFoundError(f"holography Measurement Set is absent: {measurement_set}")
    if (measurement_set / "table.dat").is_file() and (measurement_set / "ANTENNA").is_dir():
        from sl1mjax.holography_ms import inventory_casacore_measurement_set

        return inventory_casacore_measurement_set(measurement_set, archive_path=archive_path)
    digest = _hash_tree(measurement_set)
    presence = {
        name: (measurement_set / name).exists()
        for name in (*REQUIRED_HOLOGRAPHY_TABLES, *OPTIONAL_HOLOGRAPHY_TABLES)
    }
    notes: list[str] = []
    if not presence["POINTING"]:
        notes.append("POINTING table is absent; SDM-BDF inspection is required")
    correlations, correlation_codes = _read_correlation_inventory(measurement_set)
    antenna_ids, antenna_names, antenna_position_m = _read_antenna_inventory(measurement_set)
    time_mono, freq_mono = _read_axis_monotonic(measurement_set)
    if not time_mono:
        notes.append("TIME axis is not strictly increasing")
    if not freq_mono:
        notes.append("CHAN_FREQ axis is not strictly increasing")
    if antenna_ids and (
        len(antenna_ids) != len(antenna_names) or len(antenna_ids) != len(antenna_position_m)
    ):
        notes.append("antenna IDs do not resolve to names and positions")
    return HolographyArchiveInventory(
        project=THOL0001_PROJECT,
        scheduling_block=THOL0001_SCHEDULING_BLOCK,
        execution_block=THOL0001_EXECUTION_BLOCK,
        source_name=THOL0001_SOURCE,
        observation_utc="2016-01-14T04:03:28Z/2016-01-14T07:12:28Z",
        path=str(measurement_set),
        archive_sha256=digest,
        table_presence=presence,
        correlations=correlations,
        antenna_count=max(_count_antenna_rows(measurement_set), len(antenna_ids), 1),
        notes=tuple(notes),
        correlation_codes=correlation_codes,
        antenna_ids=antenna_ids,
        antenna_names=antenna_names,
        antenna_position_m=antenna_position_m,
        time_is_monotonic=time_mono,
        frequency_is_monotonic=freq_mono,
    )


def pointing_table_as_dict(table: AntennaPointingTable) -> dict[str, Any]:
    return {
        "schema_version": table.schema_version,
        "time_s": table.time_s.tolist(),
        "interval_s": table.interval_s.tolist(),
        "antenna_id": table.antenna_id.tolist(),
        "columns": [column.as_dict() for column in table.columns],
        "tracking": None if table.tracking is None else table.tracking.tolist(),
        "on_source": None if table.on_source is None else table.on_source.tolist(),
        "row_id": table.row_id.tolist(),
        "source_ms": table.source_ms,
        "table_sha256": table.table_sha256,
        "interpolation_flag": table.interpolation_flag.tolist(),
        "time_is_midpoint": table.time_is_midpoint,
    }


def pointing_table_from_dict(payload: MappingLike) -> AntennaPointingTable:
    return AntennaPointingTable(
        time_s=np.asarray(payload["time_s"], dtype=np.float64),
        interval_s=np.asarray(payload["interval_s"], dtype=np.float64),
        antenna_id=np.asarray(payload["antenna_id"], dtype=np.int32),
        columns=tuple(DirectionMeasure.from_dict(item) for item in payload["columns"]),
        tracking=None if payload.get("tracking") is None else np.asarray(payload["tracking"]),
        on_source=None if payload.get("on_source") is None else np.asarray(payload["on_source"]),
        row_id=np.asarray(payload["row_id"], dtype=np.int32),
        source_ms=str(payload["source_ms"]),
        table_sha256=str(payload["table_sha256"]),
        interpolation_flag=np.asarray(payload["interpolation_flag"], dtype=bool),
        schema_version=int(payload.get("schema_version", HOLOGRAPHY_POINTING_SCHEMA_VERSION)),
        time_is_midpoint=bool(payload.get("time_is_midpoint", True)),
    )


def resolved_pointing_as_dict(resolved: ResolvedAntennaPointing) -> dict[str, Any]:
    return {
        "schema_version": resolved.schema_version,
        "unique_time_s": resolved.unique_time_s.tolist(),
        "antenna_id": resolved.antenna_id.tolist(),
        "offset_lm_rad": resolved.offset_lm_rad.tolist(),
        "valid": resolved.valid.tolist(),
        "settled": resolved.settled.tolist(),
        "role": resolved.role.tolist(),
        "selected_column": resolved.selected_column,
        "offset_sign": resolved.offset_sign,
        "join_rule": resolved.join_rule,
        "join_tolerance_s": resolved.join_tolerance_s,
        "measure_ref": resolved.measure_ref,
        "units": resolved.units,
        "notes": list(resolved.notes),
    }


def resolved_pointing_from_dict(payload: MappingLike) -> ResolvedAntennaPointing:
    return ResolvedAntennaPointing(
        unique_time_s=np.asarray(payload["unique_time_s"], dtype=np.float64),
        antenna_id=np.asarray(payload["antenna_id"], dtype=np.int32),
        offset_lm_rad=np.asarray(payload["offset_lm_rad"], dtype=np.float64),
        valid=np.asarray(payload["valid"], dtype=bool),
        settled=np.asarray(payload["settled"], dtype=bool),
        role=np.asarray(payload["role"], dtype="U16"),
        selected_column=str(payload["selected_column"]),
        offset_sign=payload["offset_sign"],
        join_rule=payload["join_rule"],
        join_tolerance_s=float(payload["join_tolerance_s"]),
        measure_ref=str(payload["measure_ref"]),
        units=str(payload["units"]),
        schema_version=int(payload.get("schema_version", HOLOGRAPHY_POINTING_SCHEMA_VERSION)),
        notes=tuple(payload.get("notes", ())),
    )


def commanded_pointing_lm_rad(
    column: DirectionMeasure,
    phase_centre_rad: tuple[float, float],
) -> NDArray[np.float64]:
    """Return commanded pointing ``(l, m)`` in the column's native frame.

    ``AZELGEO`` / ``AZEL*`` / ``SIN`` values are already raster offsets and
    are not passed through an RA/Dec conversion. J2000/ICRS directions are
    converted relative to the correlator phase centre.
    """

    if column.units not in {"rad", "radian", "radians"}:
        raise ValueError(f"{column.name} units must be radians for spherical conversion")
    ref = str(column.measure_ref).strip().upper()
    if ref in MOUNT_OFFSET_REFS or ref.startswith("AZEL"):
        return np.asarray(column.values_rad, dtype=np.float64).copy()
    l_rad, m_rad, _n = radec_to_lmn(
        float(phase_centre_rad[0]),
        float(phase_centre_rad[1]),
        column.values_rad[:, 0],
        column.values_rad[:, 1],
    )
    return np.column_stack((l_rad, m_rad))


def commanded_offset_from_direction_target(
    table: AntennaPointingTable,
    phase_centre_rad: tuple[float, float],
) -> NDArray[np.float64]:
    """Return ``DIRECTION-TARGET`` in the same tangent frame as ``commanded_pointing_lm_rad``.

    For THOL0001 AZELGEO holography this is the stored ``POINTING_OFFSET``.
    It is the commanded antenna displacement, not the source coordinate
    inside the moved beam.
    """

    direction = commanded_pointing_lm_rad(table.column("DIRECTION"), phase_centre_rad)
    target = commanded_pointing_lm_rad(table.column("TARGET"), phase_centre_rad)
    return direction - target


def source_relative_lm_rad(
    sky_lm_rad: ArrayLike,
    pointing_delta_lm_rad: ArrayLike,
    *,
    offset_sign: OffsetSign = "commanded_pointing",
) -> NDArray[np.float64]:
    """Return beam-frame ``(l, m)`` for each antenna and sky direction.

    ``commanded_pointing`` implements ``l_{a,d}=l_d-Δ_{t,a}``. For a source
    at the phase centre and an antenna commanded to ``+l``, the source
    appears at ``-l``. ``source_minus_pointing`` is the opposite sign and
    exists so the convention ladder can fail it on purpose.
    """

    sky = np.asarray(sky_lm_rad, dtype=np.float64)
    delta = np.asarray(pointing_delta_lm_rad, dtype=np.float64)
    if sky.ndim != 2 or sky.shape[-1] != 2:
        raise ValueError("sky_lm_rad must have shape (direction, 2)")
    if delta.ndim != 2 or delta.shape[-1] != 2:
        raise ValueError("pointing_delta_lm_rad must have shape (antenna, 2)")
    if offset_sign == "commanded_pointing":
        return sky[None, :, :] - delta[:, None, :]
    if offset_sign == "source_minus_pointing":
        return sky[None, :, :] + delta[:, None, :]
    raise ValueError(f"unsupported offset_sign {offset_sign!r}")


@dataclass(frozen=True)
class HolographyDirectionCosines:
    """Direction cosines that name their frame and sign.

    The four holography frames are phase-centre sky, commanded pointing,
    source-relative beam coordinates, and feed-frame coordinates after ``χ``.
    A bare ``offset`` is refused.
    """

    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    frame: HolographyCoordinateFrame
    offset_sign: OffsetSign | None = None
    chi_sign: Literal[1, -1] | None = None

    def __post_init__(self) -> None:
        l_rad = np.asarray(self.l_rad, dtype=np.float64).reshape(-1)
        m_rad = np.asarray(self.m_rad, dtype=np.float64).reshape(-1)
        if l_rad.shape != m_rad.shape:
            raise ValueError("l_rad and m_rad must have the same shape")
        if self.frame not in {
            "phase_centre_sky",
            "commanded_pointing",
            "source_relative",
            "feed_frame",
        }:
            raise ValueError(f"unsupported holography frame {self.frame!r}")
        if self.frame in {"commanded_pointing", "source_relative", "feed_frame"}:
            if self.offset_sign is None:
                raise ValueError(f"{self.frame} must declare offset_sign")
        if self.frame == "feed_frame" and self.chi_sign not in {1, -1}:
            raise ValueError("feed_frame must declare chi_sign +1 or -1")
        object.__setattr__(self, "l_rad", l_rad)
        object.__setattr__(self, "m_rad", m_rad)

    def as_lm(self) -> NDArray[np.float64]:
        return np.column_stack((self.l_rad, self.m_rad))


def holography_phase_centre_sky(
    l_rad: ArrayLike,
    m_rad: ArrayLike,
) -> HolographyDirectionCosines:
    """Celestial direction relative to the correlator phase centre."""

    return HolographyDirectionCosines(
        l_rad=l_rad,
        m_rad=m_rad,
        frame="phase_centre_sky",
    )


def holography_commanded_pointing(
    delta_lm_rad: ArrayLike,
    *,
    offset_sign: OffsetSign = "commanded_pointing",
) -> HolographyDirectionCosines:
    """Commanded or measured antenna pointing relative to the phase centre."""

    delta = np.asarray(delta_lm_rad, dtype=np.float64)
    if delta.ndim == 1 and delta.shape == (2,):
        delta = delta.reshape(1, 2)
    if delta.ndim != 2 or delta.shape[-1] != 2:
        raise ValueError("commanded pointing must have shape (2,) or (antenna, 2)")
    return HolographyDirectionCosines(
        l_rad=delta[:, 0],
        m_rad=delta[:, 1],
        frame="commanded_pointing",
        offset_sign=offset_sign,
    )


def holography_source_relative_direction(
    sky: HolographyDirectionCosines,
    pointing: HolographyDirectionCosines,
) -> HolographyDirectionCosines:
    """Source direction in the antenna beam frame: ``l_d - Δ`` by default."""

    if sky.frame != "phase_centre_sky":
        raise ValueError("sky direction must be in the phase_centre_sky frame")
    if pointing.frame != "commanded_pointing":
        raise ValueError("pointing must be in the commanded_pointing frame")
    if pointing.offset_sign is None:
        raise ValueError("commanded pointing is missing offset_sign")
    relative = source_relative_lm_rad(
        sky.as_lm(),
        pointing.as_lm(),
        offset_sign=pointing.offset_sign,
    )
    return HolographyDirectionCosines(
        l_rad=relative[..., 0].reshape(-1),
        m_rad=relative[..., 1].reshape(-1),
        frame="source_relative",
        offset_sign=pointing.offset_sign,
    )


def holography_feed_frame_direction(
    source_relative: HolographyDirectionCosines,
    chi_rad: ArrayLike,
    *,
    chi_sign: Literal[1, -1] = 1,
) -> HolographyDirectionCosines:
    """Rotate source-relative ``(l, m)`` into the feed frame by ``χ``."""

    if source_relative.frame != "source_relative":
        raise ValueError("feed-frame rotation requires source_relative coordinates")
    chi = np.asarray(chi_rad, dtype=np.float64).reshape(-1) * float(chi_sign)
    l_rad = source_relative.l_rad
    m_rad = source_relative.m_rad
    if chi.size == 1:
        chi = np.full(l_rad.shape, float(chi[0]), dtype=np.float64)
    if chi.shape != l_rad.shape:
        raise ValueError("chi_rad must be scalar or one value per direction")
    cosine = np.cos(chi)
    sine = np.sin(chi)
    return HolographyDirectionCosines(
        l_rad=l_rad * cosine + m_rad * sine,
        m_rad=-l_rad * sine + m_rad * cosine,
        frame="feed_frame",
        offset_sign=source_relative.offset_sign,
        chi_sign=chi_sign,
    )


def promote_operator_pointing(
    pointing: ArrayLike | None,
    *,
    n_time: int,
    n_antenna: int,
    allow_missing_as_zero: bool = False,
) -> NDArray[np.float64]:
    """Promote no-offset or shared-offset forms to ``(time, antenna, 2)``.

    Holography must not use ``allow_missing_as_zero``. Ordinary mosaic
    regressions may, because a missing argument means a shared zero offset.
    """

    if n_time < 1 or n_antenna < 1:
        raise ValueError("pointing promotion needs positive time and antenna counts")
    if pointing is None:
        if not allow_missing_as_zero:
            raise ValueError(
                "holography refuses an assumed zero pointing offset when metadata is absent"
            )
        return np.zeros((n_time, n_antenna, 2), dtype=np.float64)
    array = np.asarray(pointing, dtype=np.float64)
    if array.shape == (2,):
        if not np.all(np.isfinite(array)):
            raise ValueError("shared pointing offset must be finite")
        return np.broadcast_to(array, (n_time, n_antenna, 2)).copy()
    if array.shape == (n_time, n_antenna, 2):
        return array
    raise ValueError(
        f"pointing must be omitted, shape (2,), or shape (time, antenna, 2); got {array.shape}"
    )


def resolve_antenna_pointing(
    table: AntennaPointingTable,
    unique_time_s: ArrayLike,
    antenna_id: ArrayLike,
    *,
    selected_column: str,
    phase_centre_rad: tuple[float, float],
    offset_sign: OffsetSign = "commanded_pointing",
    join_rule: JoinRule = "interval",
    join_tolerance_s: float = 0.0,
    settled_jump_arcmin: float = 0.2,
    use_on_source: bool = True,
) -> ResolvedAntennaPointing:
    """Join preserved POINTING rows onto exact unique visibility times.

    Missing antenna-time samples stay invalid. They are never filled with a
    zero offset.
    """

    times = np.asarray(unique_time_s, dtype=np.float64).reshape(-1)
    antennas = np.asarray(antenna_id, dtype=np.int32).reshape(-1)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("unique_time_s must be strictly increasing")
    column = table.column(selected_column)
    commanded = commanded_pointing_lm_rad(column, phase_centre_rad)
    n_time = int(times.size)
    n_ant = int(antennas.size)
    offsets = np.full((n_time, n_ant, 2), np.nan, dtype=np.float64)
    valid = np.zeros((n_time, n_ant), dtype=bool)
    interpolated = np.zeros((n_time, n_ant), dtype=bool)
    joined_rows = np.full((n_time, n_ant), -1, dtype=np.int32)
    for antenna_index, antenna in enumerate(antennas):
        rows = np.flatnonzero(table.antenna_id == antenna)
        if rows.size == 0:
            continue
        chosen = _join_pointing_indices(
            times,
            table.time_s[rows],
            table.interval_s[rows],
            join_rule=join_rule,
            join_tolerance_s=join_tolerance_s,
            time_is_midpoint=table.time_is_midpoint,
        )
        matched = chosen >= 0
        if not np.any(matched):
            continue
        source = rows[chosen[matched]]
        offsets[matched, antenna_index] = commanded[source]
        valid[matched, antenna_index] = True
        interpolated[matched, antenna_index] = table.interpolation_flag[source]
        joined_rows[matched, antenna_index] = source
    settled = _settled_mask(offsets, valid, settled_jump_arcmin)
    on_source_usable = table.on_source is not None and bool(np.any(table.on_source))
    if use_on_source and on_source_usable and table.on_source is not None:
        matched = joined_rows >= 0
        settled[matched] &= table.on_source[joined_rows[matched]]
    roles = classify_antenna_roles(offsets, valid, settled)
    notes = []
    if np.any(interpolated):
        notes.append("some joined samples carry an interpolation flag")
    if table.on_source is not None and not use_on_source:
        notes.append("ON_SOURCE was not used for validity")
    elif table.on_source is not None and not on_source_usable:
        notes.append("ON_SOURCE is false for every row and was ignored")
    if str(column.measure_ref).strip().upper() in MOUNT_OFFSET_REFS or str(
        column.measure_ref
    ).upper().startswith("AZEL"):
        notes.append(f"{selected_column} {column.measure_ref} values are native raster offsets")
    return ResolvedAntennaPointing(
        unique_time_s=times,
        antenna_id=antennas,
        offset_lm_rad=offsets,
        valid=valid,
        settled=settled,
        role=roles,
        selected_column=selected_column,
        offset_sign=offset_sign,
        join_rule=join_rule,
        join_tolerance_s=float(join_tolerance_s),
        measure_ref=column.measure_ref,
        units=column.units,
        notes=tuple(notes),
    )


def classify_antenna_roles(
    offset_lm_rad: ArrayLike,
    valid: ArrayLike,
    settled: ArrayLike,
    *,
    reference_radius_arcmin: float = 0.5,
    moving_radius_arcmin: float = 2.0,
) -> NDArray[np.str_]:
    """Label reference, moving, and transition samples from motion."""

    offsets = np.asarray(offset_lm_rad, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    dwell = np.asarray(settled, dtype=bool)
    radius = np.rad2deg(np.hypot(offsets[..., 0], offsets[..., 1])) * 60.0
    roles = np.full(ok.shape, AntennaPointingRole.UNKNOWN.value, dtype="U16")
    for antenna in range(ok.shape[1]):
        samples = ok[:, antenna]
        if not np.any(samples):
            continue
        peak = float(np.nanmax(radius[samples, antenna]))
        if peak < reference_radius_arcmin:
            roles[samples, antenna] = AntennaPointingRole.REFERENCE.value
        elif peak >= moving_radius_arcmin:
            roles[samples, antenna] = AntennaPointingRole.MOVING.value
        moving = roles[:, antenna] == AntennaPointingRole.MOVING.value
        roles[moving & ~dwell[:, antenna], antenna] = AntennaPointingRole.TRANSITION.value
    roles[~ok] = AntennaPointingRole.UNKNOWN.value
    return roles


@dataclass(frozen=True)
class AntennaDwellTrack:
    """Clustered settled pointing locations for one antenna."""

    antenna_id: int
    role: str
    dwell_lm_arcmin: NDArray[np.float64]
    sample_count: NDArray[np.int32]
    nearest_spacing_arcmin: float | None


@dataclass(frozen=True)
class RasterGeometryEstimate:
    """Recovered raster spacings from moving-antenna dwells."""

    spacings_arcmin: tuple[float, ...]
    dense_spacing_arcmin: float | None
    sparse_spacing_arcmin: float | None
    n_dwell: int


@dataclass(frozen=True)
class PointingConventionDiagnostics:
    """DIRECTION versus stored POINTING_OFFSET under named sign hypotheses."""

    selected_column: str
    commanded_residual_arcmin: float
    flipped_residual_arcmin: float
    swapped_residual_arcmin: float
    status: HolographyGateStatus
    notes: tuple[str, ...]


@dataclass(frozen=True)
class HolographyPointingAudit:
    """Phase-1 pointing audit that does not infer the raster from visibilities."""

    tracks: tuple[AntennaDwellTrack, ...]
    raster: RasterGeometryEstimate
    memo195: Memo195LowerCRaster
    memo195_dense_status: HolographyGateStatus
    memo195_sparse_status: HolographyGateStatus
    convention: PointingConventionDiagnostics
    selected_column: str
    offset_sign: OffsetSign


def cluster_antenna_dwells(
    resolved: ResolvedAntennaPointing,
    *,
    cluster_radius_arcmin: float = 0.15,
) -> tuple[AntennaDwellTrack, ...]:
    """Cluster settled dwells separately for each antenna."""

    tracks: list[AntennaDwellTrack] = []
    for antenna_index, antenna in enumerate(resolved.antenna_id):
        usable = resolved.valid[:, antenna_index] & resolved.settled[:, antenna_index]
        role = _track_role(resolved.role[:, antenna_index], resolved.valid[:, antenna_index])
        if not np.any(usable):
            tracks.append(
                AntennaDwellTrack(
                    antenna_id=int(antenna),
                    role=role,
                    dwell_lm_arcmin=np.zeros((0, 2), dtype=np.float64),
                    sample_count=np.zeros((0,), dtype=np.int32),
                    nearest_spacing_arcmin=None,
                )
            )
            continue
        samples = np.rad2deg(resolved.offset_lm_rad[usable, antenna_index]) * 60.0
        dwells: list[np.ndarray] = []
        counts: list[int] = []
        for sample in samples:
            assigned = False
            for index, dwell in enumerate(dwells):
                if float(np.hypot(*(sample - dwell))) < cluster_radius_arcmin:
                    n_old = counts[index]
                    dwells[index] = (dwell * n_old + sample) / (n_old + 1)
                    counts[index] = n_old + 1
                    assigned = True
                    break
            if not assigned:
                dwells.append(np.asarray(sample, dtype=np.float64).copy())
                counts.append(1)
        dwell_lm = np.asarray(dwells, dtype=np.float64)
        spacing = _nearest_spacing_arcmin(dwell_lm)
        tracks.append(
            AntennaDwellTrack(
                antenna_id=int(antenna),
                role=role,
                dwell_lm_arcmin=dwell_lm,
                sample_count=np.asarray(counts, dtype=np.int32),
                nearest_spacing_arcmin=spacing,
            )
        )
    return tuple(tracks)


def estimate_raster_geometry(
    tracks: tuple[AntennaDwellTrack, ...],
    *,
    memo195: Memo195LowerCRaster | None = None,
    spacing_tolerance_arcmin: float = 0.2,
) -> RasterGeometryEstimate:
    """Recover dense and sparse spacings from moving-antenna dwells."""

    expected = memo195 or Memo195LowerCRaster()
    spacings = [
        float(track.nearest_spacing_arcmin)
        for track in tracks
        if track.role == AntennaPointingRole.MOVING.value
        and track.nearest_spacing_arcmin is not None
    ]
    n_dwell = int(sum(int(track.dwell_lm_arcmin.shape[0]) for track in tracks))
    dense = _match_spacing(spacings, expected.dense_spacing_arcmin, spacing_tolerance_arcmin)
    sparse = _match_spacing(spacings, expected.sparse_spacing_arcmin, spacing_tolerance_arcmin)
    return RasterGeometryEstimate(
        spacings_arcmin=tuple(spacings),
        dense_spacing_arcmin=dense,
        sparse_spacing_arcmin=sparse,
        n_dwell=n_dwell,
    )


def pointing_convention_diagnostics(
    table: AntennaPointingTable,
    phase_centre_rad: tuple[float, float],
    *,
    selected_column: str = "DIRECTION",
    residual_limit_arcmin: float = 0.05,
) -> PointingConventionDiagnostics:
    """Score the physical pointing chain, not a column against itself.

    ``POINTING_OFFSET`` is the commanded displacement. When that column is
    selected, the diagnostic compares ``DIRECTION-TARGET`` with the stored
    offset. Comparing ``POINTING_OFFSET`` to itself cannot test the beam
    query sign.
    """

    names = [column.name for column in table.columns]
    if "POINTING_OFFSET" not in names:
        return PointingConventionDiagnostics(
            selected_column=selected_column,
            commanded_residual_arcmin=float("nan"),
            flipped_residual_arcmin=float("nan"),
            swapped_residual_arcmin=float("nan"),
            status="not_run",
            notes=("POINTING_OFFSET is absent; convention residual cannot be scored",),
        )
    notes: list[str] = []
    if selected_column == "POINTING_OFFSET":
        if "DIRECTION" not in names or "TARGET" not in names:
            return PointingConventionDiagnostics(
                selected_column=selected_column,
                commanded_residual_arcmin=float("nan"),
                flipped_residual_arcmin=float("nan"),
                swapped_residual_arcmin=float("nan"),
                status="fail",
                notes=(
                    "POINTING_OFFSET cannot be scored against itself; "
                    "DIRECTION and TARGET are required for DIRECTION-TARGET",
                ),
            )
        commanded = commanded_offset_from_direction_target(table, phase_centre_rad)
        notes.append("POINTING_OFFSET is scored as DIRECTION-TARGET, not against itself")
    else:
        commanded = commanded_pointing_lm_rad(table.column(selected_column), phase_centre_rad)
    stored = table.column("POINTING_OFFSET").values_rad
    nonzero = np.hypot(stored[:, 0], stored[:, 1]) > np.deg2rad(0.05 / 60.0)
    if not np.any(nonzero):
        return PointingConventionDiagnostics(
            selected_column=selected_column,
            commanded_residual_arcmin=0.0,
            flipped_residual_arcmin=0.0,
            swapped_residual_arcmin=0.0,
            status="fail",
            notes=("no nonzero POINTING_OFFSET samples to distinguish sign or axes",),
        )
    commanded_r = _offset_residual_arcmin(commanded, stored, nonzero)
    flipped_r = _offset_residual_arcmin(-commanded, stored, nonzero)
    swapped_r = _offset_residual_arcmin(
        np.column_stack((commanded[:, 1], commanded[:, 0])), stored, nonzero
    )
    unique_winner = commanded_r < 0.5 * min(flipped_r, swapped_r)
    status: HolographyGateStatus = (
        "pass" if unique_winner and commanded_r <= residual_limit_arcmin else "fail"
    )
    if not unique_winner:
        notes.append("sign flip or axis swap is not uniquely worse than commanded_pointing")
    if commanded_r > residual_limit_arcmin:
        notes.append("DIRECTION-TARGET to POINTING_OFFSET residual exceeds the limit")
    return PointingConventionDiagnostics(
        selected_column=selected_column,
        commanded_residual_arcmin=commanded_r,
        flipped_residual_arcmin=flipped_r,
        swapped_residual_arcmin=swapped_r,
        status=status,
        notes=tuple(notes),
    )


def audit_resolved_pointing(
    resolved: ResolvedAntennaPointing,
    table: AntennaPointingTable,
    phase_centre_rad: tuple[float, float],
    *,
    memo195: Memo195LowerCRaster | None = None,
    spacing_tolerance_arcmin: float = 0.2,
) -> HolographyPointingAudit:
    """Cluster tracks and score raster geometry plus the coordinate convention."""

    expected = memo195 or Memo195LowerCRaster()
    tracks = cluster_antenna_dwells(resolved)
    raster = estimate_raster_geometry(
        tracks, memo195=expected, spacing_tolerance_arcmin=spacing_tolerance_arcmin
    )
    convention = pointing_convention_diagnostics(
        table, phase_centre_rad, selected_column=resolved.selected_column
    )
    return HolographyPointingAudit(
        tracks=tracks,
        raster=raster,
        memo195=expected,
        memo195_dense_status="pass" if raster.dense_spacing_arcmin is not None else "fail",
        memo195_sparse_status="pass" if raster.sparse_spacing_arcmin is not None else "fail",
        convention=convention,
        selected_column=resolved.selected_column,
        offset_sign=resolved.offset_sign,
    )


def load_synthetic_holography_rime_case(
    root: Path | None = None,
) -> tuple[HolographyObservation, VoltageBeamModel, NDArray[np.complex128]]:
    """Load the committed three-antenna holography RIME golden."""

    from sl1mjax.finite_pixel import ManufacturedVoltageBeam

    path = (SYNTHETIC_HOLOGRAPHY_FIXTURE_ROOT if root is None else Path(root)) / "rime_case.json"
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported holography RIME fixture schema")
    block = VisibilityBlock(
        uvw_m=np.asarray(payload["uvw_m"], dtype=np.float64),
        frequency_hz=np.asarray(payload["frequency_hz"], dtype=np.float64),
        visibility=np.zeros(
            (
                len(payload["time_s"]),
                len(payload["frequency_hz"]),
                len(payload["correlations"]),
            ),
            dtype=np.complex128,
        ),
        weight=np.ones(
            (
                len(payload["time_s"]),
                len(payload["frequency_hz"]),
                len(payload["correlations"]),
            ),
            dtype=np.float64,
        ),
        flag=np.zeros(
            (
                len(payload["time_s"]),
                len(payload["frequency_hz"]),
                len(payload["correlations"]),
            ),
            dtype=bool,
        ),
        time_s=np.asarray(payload["time_s"], dtype=np.float64),
        antenna1=np.asarray(payload["antenna1"], dtype=np.int32),
        antenna2=np.asarray(payload["antenna2"], dtype=np.int32),
        correlations=tuple(Correlation(name) for name in payload["correlations"]),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=tuple(payload["phase_centre_rad"]),
    )
    pointing = resolved_pointing_from_dict(payload["pointing"])
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=np.asarray(payload["antenna_position_m"], dtype=np.float64),
        calibration_state=str(payload["calibration_state"]),
        phase_centre_rad=tuple(payload["phase_centre_rad"]),
        source_name=str(payload.get("source_name", THOL0001_SOURCE)),
        stokes_i=float(payload["stokes_i"]),
        provenance={"fixture": "holography_synthetic_rime_case"},
    )
    beam = ManufacturedVoltageBeam(
        intercept=_complex_matrix(payload["beam"]["intercept"]),
        grad_l=_complex_matrix(payload["beam"]["grad_l"]),
    )
    expected = np.asarray(payload["expected_visibility"], dtype=np.float64)
    expected = expected[..., 0] + 1j * expected[..., 1]
    return observation, beam, expected.astype(np.complex128)


def synthetic_holography_rime_case(
    *,
    phase_centre_rad: tuple[float, float] = (np.deg2rad(84.0), np.deg2rad(50.0)),
) -> tuple[HolographyObservation, VoltageBeamModel]:
    """Three-antenna holography observation used by the committed RIME golden."""

    from sl1mjax.finite_pixel import ManufacturedVoltageBeam

    time_s = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    block = VisibilityBlock(
        uvw_m=np.array(
            [
                [30.0, -12.0, 8.0],
                [18.0, 22.0, -4.0],
                [30.0, -12.0, 8.0],
                [18.0, 22.0, -4.0],
            ],
            dtype=np.float64,
        ),
        frequency_hz=np.array([4.564e9]),
        visibility=np.zeros((4, 1, 4), dtype=np.complex128),
        weight=np.ones((4, 1, 4), dtype=np.float64),
        flag=np.zeros((4, 1, 4), dtype=bool),
        time_s=time_s,
        antenna1=np.array([0, 1, 0, 1], dtype=np.int32),
        antenna2=np.array([1, 2, 1, 2], dtype=np.int32),
        correlations=(Correlation.RR, Correlation.RL, Correlation.LR, Correlation.LL),
        receptor_basis=ReceptorBasis.CIRCULAR,
        phase_centre_rad=phase_centre_rad,
    )
    offsets = np.zeros((2, 3, 2), dtype=np.float64)
    offsets[0, 1] = (np.deg2rad(1.72 / 60.0), 0.0)
    offsets[1, 1] = (0.0, np.deg2rad(1.72 / 60.0))
    offsets[0, 2] = (np.deg2rad(4.59 / 60.0), 0.0)
    offsets[1, 2] = (np.deg2rad(4.59 / 60.0), np.deg2rad(4.59 / 60.0))
    valid = np.ones((2, 3), dtype=bool)
    role = np.full((2, 3), AntennaPointingRole.MOVING.value, dtype="U16")
    role[:, 0] = AntennaPointingRole.REFERENCE.value
    pointing = ResolvedAntennaPointing(
        unique_time_s=np.array([0.0, 10.0], dtype=np.float64),
        antenna_id=np.arange(3, dtype=np.int32),
        offset_lm_rad=offsets,
        valid=valid,
        settled=valid,
        role=role,
        selected_column="DIRECTION",
        offset_sign="commanded_pointing",
        join_rule="interval",
        join_tolerance_s=0.0,
        measure_ref="J2000",
        units="rad",
    )
    beam = ManufacturedVoltageBeam(
        intercept=np.array(
            [[1.0 + 0.0j, 0.08 - 0.02j], [0.05 + 0.03j, 0.9 + 0.0j]],
            dtype=np.complex128,
        ),
        grad_l=np.array(
            [[0.4 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -0.3 + 0.0j]],
            dtype=np.complex128,
        ),
    )
    observation = HolographyObservation(
        block=block,
        pointing=pointing,
        antenna_position_m=np.array(
            [
                [-1_601_162.0, -5_042_003.0, 3_553_983.0],
                [-1_601_100.0, -5_042_100.0, 3_553_900.0],
                [-1_601_200.0, -5_042_190.0, 3_554_000.0],
            ]
        ),
        calibration_state="casa_parang_true",
        phase_centre_rad=phase_centre_rad,
        stokes_i=1.1,
        provenance={"fixture": "holography_synthetic_rime_case"},
    )
    return observation, beam


def _complex_matrix(values: ArrayLike) -> NDArray[np.complex128]:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != (2, 2):
        real = np.asarray(values, dtype=np.float64)
        if real.shape != (2, 2, 2):
            raise ValueError("Jones coefficient must be (2, 2) complex or (2, 2, 2) real/imag")
        array = real[..., 0] + 1j * real[..., 1]
    return array


def load_synthetic_holography_pointing_table(
    root: Path | None = None,
) -> AntennaPointingTable:
    """Load the committed three-antenna Memo 195-like pointing fixture."""

    path = (SYNTHETIC_HOLOGRAPHY_FIXTURE_ROOT if root is None else Path(root)) / (
        "pointing_table.json"
    )
    return pointing_table_from_dict(json.loads(path.read_text()))


def synthetic_memo195_pointing_table(
    *,
    phase_centre_rad: tuple[float, float] = (np.deg2rad(84.0), np.deg2rad(50.0)),
) -> AntennaPointingTable:
    """Asymmetric dense 3×3 plus sparse 2×2 raster for track and spacing tests."""

    raster = Memo195LowerCRaster()
    dense = [
        (float(ix) * raster.dense_spacing_arcmin, float(iy) * raster.dense_spacing_arcmin)
        for iy in (-1, 0, 1)
        for ix in (-1, 0, 1)
    ]
    sparse = [
        (0.0, 0.0),
        (raster.sparse_spacing_arcmin, 0.0),
        (0.0, raster.sparse_spacing_arcmin),
        (raster.sparse_spacing_arcmin, raster.sparse_spacing_arcmin),
    ]
    n_time = max(len(dense), len(sparse)) + 1
    times = np.arange(n_time, dtype=np.float64) * 10.0
    rows: list[dict[str, Any]] = []
    for time_index, time in enumerate(times):
        for antenna in (0, 1, 2):
            if antenna == 1 and time_index == 4:
                continue
            if antenna == 0:
                offset_arcmin = (0.0, 0.0)
                transition = False
            elif antenna == 1:
                if time_index >= len(dense):
                    offset_arcmin = dense[-1]
                    transition = False
                else:
                    offset_arcmin = dense[time_index]
                    transition = False
            elif time_index == len(sparse):
                offset_arcmin = (
                    0.5 * (sparse[-1][0] + sparse[0][0]),
                    0.5 * (sparse[-1][1] + sparse[0][1]),
                )
                transition = True
            elif time_index > len(sparse):
                offset_arcmin = sparse[-1]
                transition = False
            else:
                offset_arcmin = sparse[time_index]
                transition = False
            offset = np.deg2rad(np.asarray(offset_arcmin, dtype=np.float64) / 60.0)
            ra, dec = _pointing_radec(phase_centre_rad, offset)
            rows.append(
                {
                    "time_s": time,
                    "antenna_id": antenna,
                    "ra": ra,
                    "dec": dec,
                    "offset": offset,
                    "transition": transition,
                }
            )
    return _pointing_table_from_rows(
        rows, phase_centre_rad, source_ms="synthetic://thol0001-memo195"
    )


def _track_role(role: np.ndarray, valid: np.ndarray) -> str:
    labels = role[valid]
    if labels.size == 0:
        return AntennaPointingRole.UNKNOWN.value
    if np.any(labels == AntennaPointingRole.REFERENCE.value):
        return AntennaPointingRole.REFERENCE.value
    if np.any(labels == AntennaPointingRole.MOVING.value) or np.any(
        labels == AntennaPointingRole.TRANSITION.value
    ):
        return AntennaPointingRole.MOVING.value
    return AntennaPointingRole.UNKNOWN.value


def _nearest_spacing_arcmin(dwells: np.ndarray) -> float | None:
    if dwells.shape[0] < 2:
        return None
    distances = []
    for index, dwell in enumerate(dwells):
        others = np.delete(dwells, index, axis=0)
        distances.append(float(np.min(np.hypot(others[:, 0] - dwell[0], others[:, 1] - dwell[1]))))
    return float(np.median(distances))


def _match_spacing(spacings: list[float], expected: float, tolerance: float) -> float | None:
    matches = [spacing for spacing in spacings if abs(spacing - expected) <= tolerance]
    if not matches:
        return None
    return float(np.mean(matches))


def _offset_residual_arcmin(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    delta = left[mask] - right[mask]
    return float(np.max(np.rad2deg(np.hypot(delta[:, 0], delta[:, 1])) * 60.0))


def synthetic_holography_pointing_table(
    *,
    phase_centre_rad: tuple[float, float] = (np.deg2rad(84.0), np.deg2rad(50.0)),
) -> AntennaPointingTable:
    """Three-antenna fixture: one reference, two moving, one missing sample."""

    times = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    antennas = np.array([0, 1, 2], dtype=np.int32)
    raster = np.array(
        [
            [0.0, 0.0],
            [np.deg2rad(3.0 / 60.0), 0.0],
            [0.0, np.deg2rad(3.0 / 60.0)],
            [np.deg2rad(-3.0 / 60.0), 0.0],
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for time_index, time in enumerate(times):
        for antenna in antennas:
            if antenna == 1 and time_index == 2:
                continue
            if antenna == 0:
                offset = np.zeros(2, dtype=np.float64)
            elif antenna == 1:
                offset = raster[time_index]
            else:
                offset = raster[(time_index + 1) % raster.shape[0]]
            ra, dec = _pointing_radec(phase_centre_rad, offset)
            rows.append(
                {
                    "time_s": time,
                    "antenna_id": antenna,
                    "ra": ra,
                    "dec": dec,
                    "offset": offset,
                    "transition": antenna == 2 and time_index == 1,
                }
            )
    return _pointing_table_from_rows(rows, phase_centre_rad, source_ms="synthetic://thol0001")


def _pointing_table_from_rows(
    rows: list[dict[str, Any]],
    phase_centre_rad: tuple[float, float],
    *,
    source_ms: str,
    dwell: float = 10.0,
) -> AntennaPointingTable:
    n_row = len(rows)
    direction = np.array([[row["ra"], row["dec"]] for row in rows], dtype=np.float64)
    target = np.full((n_row, 2), phase_centre_rad, dtype=np.float64)
    pointing_offset = np.array([row["offset"] for row in rows], dtype=np.float64)
    payload = json.dumps(
        {
            "time": [float(row["time_s"]) for row in rows],
            "antenna": [int(row["antenna_id"]) for row in rows],
            "source_ms": source_ms,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return AntennaPointingTable(
        time_s=np.array([row["time_s"] for row in rows], dtype=np.float64),
        interval_s=np.full(n_row, dwell, dtype=np.float64),
        antenna_id=np.array([row["antenna_id"] for row in rows], dtype=np.int32),
        columns=(
            DirectionMeasure("DIRECTION", direction, "J2000", "rad"),
            DirectionMeasure("TARGET", target, "J2000", "rad"),
            DirectionMeasure("POINTING_OFFSET", pointing_offset, "SIN", "rad"),
        ),
        tracking=np.ones(n_row, dtype=bool),
        on_source=np.array([not row["transition"] for row in rows], dtype=bool),
        row_id=np.arange(n_row, dtype=np.int32),
        source_ms=source_ms,
        table_sha256=digest,
        interpolation_flag=np.zeros(n_row, dtype=bool),
    )


def _pointing_radec(
    phase_centre_rad: tuple[float, float], offset_lm_rad: np.ndarray
) -> tuple[float, float]:
    ra, dec = lmn_to_radec(
        float(phase_centre_rad[0]),
        float(phase_centre_rad[1]),
        float(offset_lm_rad[0]),
        float(offset_lm_rad[1]),
    )
    return float(np.asarray(ra)), float(np.asarray(dec))


def join_pointing_time_residuals(
    table: AntennaPointingTable,
    unique_time_s: ArrayLike,
    antenna_id: ArrayLike,
    *,
    join_rule: JoinRule = "interval",
    join_tolerance_s: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """MAIN-minus-POINTING residuals on the unique-time grid. Unmatched stay NaN."""

    residual_s, pointing_time_s, _interval_s, _overlap_hits = join_pointing_alignment(
        table,
        unique_time_s,
        antenna_id,
        join_rule=join_rule,
        join_tolerance_s=join_tolerance_s,
    )
    return residual_s, pointing_time_s


def join_pointing_alignment(
    table: AntennaPointingTable,
    unique_time_s: ArrayLike,
    antenna_id: ArrayLike,
    *,
    join_rule: JoinRule = "interval",
    join_tolerance_s: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """Residuals, POINTING times, POINTING intervals, and multi-interval hit count."""

    times = np.asarray(unique_time_s, dtype=np.float64).reshape(-1)
    antennas = np.asarray(antenna_id, dtype=np.int32).reshape(-1)
    residual_s = np.full((times.size, antennas.size), np.nan, dtype=np.float64)
    pointing_time_s = np.full((times.size, antennas.size), np.nan, dtype=np.float64)
    interval_s = np.full((times.size, antennas.size), np.nan, dtype=np.float64)
    overlap_hits = 0
    for antenna_index, antenna in enumerate(antennas):
        rows = np.flatnonzero(table.antenna_id == antenna)
        if rows.size == 0:
            continue
        chosen, n_overlap = _join_pointing_indices(
            times,
            table.time_s[rows],
            table.interval_s[rows],
            join_rule=join_rule,
            join_tolerance_s=join_tolerance_s,
            time_is_midpoint=table.time_is_midpoint,
            count_overlaps=True,
        )
        overlap_hits += int(n_overlap)
        matched = chosen >= 0
        if not np.any(matched):
            continue
        source = rows[chosen[matched]]
        pointing_time_s[matched, antenna_index] = table.time_s[source]
        interval_s[matched, antenna_index] = table.interval_s[source]
        residual_s[matched, antenna_index] = times[matched] - table.time_s[source]
    return residual_s, pointing_time_s, interval_s, overlap_hits


def pointing_table_overlap_pairs(table: AntennaPointingTable) -> int:
    """Count POINTING interval pairs that overlap. Query hits are a separate test."""

    total = 0
    for antenna in np.unique(table.antenna_id):
        rows = np.flatnonzero(table.antenna_id == antenna)
        if rows.size < 2:
            continue
        time_s = table.time_s[rows]
        interval_s = table.interval_s[rows]
        if table.time_is_midpoint:
            start = time_s - 0.5 * interval_s
            stop = time_s + 0.5 * interval_s
        else:
            start = time_s
            stop = time_s + interval_s
        order = np.argsort(start, kind="stable")
        ordered_start = start[order]
        ordered_stop = stop[order]
        total += int(np.sum(ordered_stop[:-1] > ordered_start[1:]))
    return total


def _join_pointing_indices(
    query_time_s: np.ndarray,
    sample_time_s: np.ndarray,
    interval_s: np.ndarray,
    *,
    join_rule: JoinRule,
    join_tolerance_s: float,
    time_is_midpoint: bool = True,
    count_overlaps: bool = False,
) -> NDArray[np.int32] | tuple[NDArray[np.int32], int]:
    """Return the sample index for each query time, or ``-1`` if unmatched."""

    queries = np.asarray(query_time_s, dtype=np.float64).reshape(-1)
    if join_rule == "interval":
        chosen, n_overlap = _join_interval_indices(
            queries,
            np.asarray(sample_time_s, dtype=np.float64).reshape(-1),
            np.asarray(interval_s, dtype=np.float64).reshape(-1),
            time_is_midpoint=time_is_midpoint,
        )
        if count_overlaps:
            return chosen, n_overlap
        if n_overlap:
            raise ValueError("overlapping POINTING intervals for one antenna-time")
        return chosen
    chosen = np.full(queries.size, -1, dtype=np.int32)
    for time_index, time in enumerate(queries):
        index = _join_pointing_row(
            float(time),
            sample_time_s,
            interval_s,
            join_rule=join_rule,
            join_tolerance_s=join_tolerance_s,
            time_is_midpoint=time_is_midpoint,
        )
        if index is not None:
            chosen[time_index] = int(index)
    if count_overlaps:
        return chosen, 0
    return chosen


def _join_interval_indices(
    query_time_s: np.ndarray,
    sample_time_s: np.ndarray,
    interval_s: np.ndarray,
    *,
    time_is_midpoint: bool,
) -> tuple[NDArray[np.int32], int]:
    if time_is_midpoint:
        start = sample_time_s - 0.5 * interval_s
        stop = sample_time_s + 0.5 * interval_s
    else:
        start = np.asarray(sample_time_s, dtype=np.float64)
        stop = start + np.asarray(interval_s, dtype=np.float64)
    if start.size == 0:
        return np.full(query_time_s.size, -1, dtype=np.int32), 0
    inside = (query_time_s[:, None] >= start[None, :]) & (query_time_s[:, None] < stop[None, :])
    matches = np.sum(inside, axis=1)
    chosen = np.full(query_time_s.size, -1, dtype=np.int32)
    unique = matches == 1
    if np.any(unique):
        chosen[unique] = np.argmax(inside[unique], axis=1).astype(np.int32)
    return chosen, int(np.sum(matches > 1))


def _join_pointing_row(
    time_s: float,
    sample_time_s: np.ndarray,
    interval_s: np.ndarray,
    *,
    join_rule: JoinRule,
    join_tolerance_s: float,
    time_is_midpoint: bool = True,
) -> int | None:
    if join_rule == "interval":
        half = 0.5 * np.asarray(interval_s, dtype=np.float64)
        if time_is_midpoint:
            inside = (sample_time_s - half <= time_s) & (time_s < sample_time_s + half)
        else:
            inside = (sample_time_s <= time_s) & (time_s < sample_time_s + interval_s)
        matches = np.flatnonzero(inside)
        if matches.size == 1:
            return int(matches[0])
        if matches.size > 1:
            raise ValueError("overlapping POINTING intervals for one antenna-time")
        return None
    if join_rule == "nearest_within_tolerance":
        delta = np.abs(sample_time_s - time_s)
        index = int(np.argmin(delta))
        if float(delta[index]) > float(join_tolerance_s):
            return None
        return index
    raise ValueError(f"unsupported join_rule {join_rule!r}")


def _settled_mask(offsets: np.ndarray, valid: np.ndarray, jump_arcmin: float) -> NDArray[np.bool_]:
    settled = np.array(valid, copy=True)
    jump = np.deg2rad(jump_arcmin / 60.0)
    for antenna in range(offsets.shape[1]):
        previous: np.ndarray | None = None
        for time_index in range(offsets.shape[0]):
            if not valid[time_index, antenna]:
                previous = None
                continue
            current = offsets[time_index, antenna]
            if previous is not None and float(np.hypot(*(current - previous))) > jump:
                settled[time_index, antenna] = False
            previous = current
    return settled


def _hash_tree(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            while True:
                block = handle.read(int(chunk_bytes))
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            while True:
                block = handle.read(int(chunk_bytes))
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _count_antenna_rows(measurement_set: Path) -> int:
    antenna = measurement_set / "ANTENNA"
    if not antenna.exists():
        return 0
    names = antenna / "table.dat"
    if names.exists():
        lines = [line for line in names.read_text().splitlines() if line.strip()]
        if lines:
            return len(lines)
    if antenna.is_file():
        return 1
    return max(1, len(list(antenna.glob("*"))))


def _read_correlation_inventory(
    measurement_set: Path,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    path = measurement_set / "POLARIZATION" / "corr_type.txt"
    if not path.exists():
        return ("RR", "RL", "LR", "LL"), (5, 6, 7, 8)
    codes = tuple(int(token) for token in path.read_text().split() if token.strip())
    names = tuple(CASA_CORRELATION_FROM_CODE.get(code, f"CODE_{code}") for code in codes)
    return names, codes


def _read_antenna_inventory(
    measurement_set: Path,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    names_path = measurement_set / "ANTENNA" / "table.dat"
    positions_path = measurement_set / "ANTENNA" / "positions.txt"
    names = (
        tuple(line.strip() for line in names_path.read_text().splitlines() if line.strip())
        if names_path.exists()
        else ()
    )
    positions: list[tuple[float, float, float]] = []
    if positions_path.exists():
        for line in positions_path.read_text().splitlines():
            if not line.strip():
                continue
            values = [float(token) for token in line.split()]
            if len(values) != 3:
                raise ValueError("ANTENNA positions.txt must have three ITRF values per row")
            positions.append((values[0], values[1], values[2]))
    ids = tuple(range(len(names))) if names else ()
    return ids, names, tuple(positions)


def _read_axis_monotonic(measurement_set: Path) -> tuple[bool, bool]:
    time_path = measurement_set / "MAIN" / "time.txt"
    freq_path = measurement_set / "SPECTRAL_WINDOW" / "chan_freq.txt"
    time_mono = _text_axis_is_monotonic(time_path) if time_path.exists() else True
    freq_mono = _text_axis_is_monotonic(freq_path) if freq_path.exists() else True
    return time_mono, freq_mono


def _text_axis_is_monotonic(path: Path) -> bool:
    values = np.asarray(
        [float(token) for token in path.read_text().split() if token.strip()],
        dtype=np.float64,
    )
    if values.size < 2:
        return True
    return bool(np.all(np.diff(values) > 0.0))


@dataclass(frozen=True)
class HolographyOperatorResult:
    """Holography visibilities plus separate pointing and beam masks."""

    visibility: NDArray[np.complex128]
    beam_valid: NDArray[np.bool_]
    pointing_valid: NDArray[np.bool_]
    off_diagonal_valid: NDArray[np.bool_]
    moving_reference_rows: NDArray[np.bool_]
    provenance: dict[str, Any]
    row_reason: NDArray[np.str_] | None = None


def pointing_planes_for_block(
    pointing: ResolvedAntennaPointing,
    unique_time_s: ArrayLike,
    antenna_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Align a resolved pointing object onto a visibility block's antennas."""

    times = np.asarray(unique_time_s, dtype=np.float64).reshape(-1)
    if times.shape != pointing.unique_time_s.shape or not np.array_equal(
        times, pointing.unique_time_s
    ):
        raise ValueError("resolved pointing unique times must match the visibility unique times")
    offsets = np.full((times.size, antenna_count, 2), np.nan, dtype=np.float64)
    valid = np.zeros((times.size, antenna_count), dtype=bool)
    for index, antenna in enumerate(pointing.antenna_id):
        antenna_i = int(antenna)
        if antenna_i < 0 or antenna_i >= antenna_count:
            raise ValueError(f"resolved antenna_id {antenna_i} is outside the block")
        offsets[:, antenna_i] = pointing.offset_lm_rad[:, index]
        valid[:, antenna_i] = pointing.valid[:, index]
    return offsets, valid


def moving_reference_row_mask(
    block: VisibilityBlock,
    pointing: ResolvedAntennaPointing,
) -> NDArray[np.bool_]:
    """True for baselines with one moving and one reference antenna."""

    unique_times, inverse = unique_visibility_times(block.time_s)
    _offsets, valid = pointing_planes_for_block(pointing, unique_times, block.antenna_count)
    roles = np.full((unique_times.size, block.antenna_count), "unknown", dtype="U16")
    for index, antenna in enumerate(pointing.antenna_id):
        roles[:, int(antenna)] = pointing.role[:, index]
    mask = np.zeros(block.time_s.shape[0], dtype=bool)
    for row, time_index in enumerate(inverse):
        if not (valid[time_index, block.antenna1[row]] and valid[time_index, block.antenna2[row]]):
            continue
        pair = {
            str(roles[time_index, block.antenna1[row]]),
            str(roles[time_index, block.antenna2[row]]),
        }
        mask[row] = pair == {
            AntennaPointingRole.MOVING.value,
            AntennaPointingRole.REFERENCE.value,
        }
    return mask


@dataclass(frozen=True)
class HolographyObservation:
    """Visibility block bound to resolved per-antenna pointing.

    Refuses mismatched unique times, antenna maps, phase centres, or
    calibration-state labels. Pointing stays outside ``BeamOperatorConfig``.
    """

    block: VisibilityBlock
    pointing: ResolvedAntennaPointing
    antenna_position_m: NDArray[np.float64]
    calibration_state: str
    phase_centre_rad: tuple[float, float]
    source_name: str = THOL0001_SOURCE
    stokes_i: float = 1.0
    source_coherency_visibility: NDArray[np.complex128] | None = None
    source_model: HolographySourceModel | None = None
    selected_correlations: tuple[str, ...] = ()
    selected_frequency_hz: NDArray[np.float64] | None = None
    selected_spw_id: int | None = None
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        state = require_beam_calibration_state(self.calibration_state)
        object.__setattr__(self, "calibration_state", state.value)
        positions = np.asarray(self.antenna_position_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("antenna_position_m must have shape (antenna, 3)")
        if positions.shape[0] < self.block.antenna_count:
            raise ValueError("antenna_position_m must cover every antenna in the block")
        object.__setattr__(self, "antenna_position_m", positions)
        centre = (float(self.phase_centre_rad[0]), float(self.phase_centre_rad[1]))
        if not np.allclose(centre, self.block.phase_centre_rad, atol=1e-15):
            raise ValueError("observation phase centre does not match the visibility block")
        object.__setattr__(self, "phase_centre_rad", centre)
        unique_times, _inverse = unique_visibility_times(self.block.time_s)
        if unique_times.shape != self.pointing.unique_time_s.shape or not np.array_equal(
            unique_times, self.pointing.unique_time_s
        ):
            raise ValueError("resolved pointing unique times do not match the visibility block")
        used = np.unique(np.concatenate((self.block.antenna1, self.block.antenna2)))
        known = set(int(antenna) for antenna in self.pointing.antenna_id)
        missing = [int(antenna) for antenna in used if int(antenna) not in known]
        if missing:
            raise ValueError(f"pointing has no rows for antennas {missing}")
        if self.source_coherency_visibility is not None:
            source = np.asarray(self.source_coherency_visibility, dtype=np.complex128)
            expected = (
                self.block.visibility.shape[0],
                self.block.frequency_hz.size,
                2,
                2,
            )
            if source.shape != expected:
                raise ValueError(f"source_coherency_visibility must have shape {expected}")
            object.__setattr__(self, "source_coherency_visibility", source)
        if self.source_model is not None and self.source_coherency_visibility is not None:
            raise ValueError("provide source_model or source_coherency_visibility, not both")
        known_correlations = _correlation_names(self.block.correlations)
        selected = (
            tuple(str(name) for name in self.selected_correlations)
            if self.selected_correlations
            else known_correlations
        )
        unknown = [name for name in selected if name not in known_correlations]
        if unknown:
            raise ValueError(f"selected_correlations {unknown} are not in the visibility block")
        object.__setattr__(self, "selected_correlations", selected)
        frequencies = np.asarray(self.block.frequency_hz, dtype=np.float64).reshape(-1)
        if self.selected_frequency_hz is None:
            selected_hz = frequencies.copy()
        else:
            selected_hz = np.asarray(self.selected_frequency_hz, dtype=np.float64).reshape(-1)
            if selected_hz.size == 0:
                raise ValueError("selected_frequency_hz cannot be empty")
            matched = np.any(
                np.isclose(selected_hz[:, None], frequencies[None, :], rtol=0.0, atol=1.0),
                axis=1,
            )
            if not bool(np.all(matched)):
                raise ValueError("selected_frequency_hz must be channels present in the block")
        object.__setattr__(self, "selected_frequency_hz", selected_hz)
        if self.selected_spw_id is None:
            object.__setattr__(self, "selected_spw_id", int(self.block.spectral_window_id))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    def pointing_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return time inverse, offsets, valid, settled, and moving--reference masks."""

        return _pointing_state_for_block(self.block, self.pointing)

    def row_reason(
        self,
        *,
        mask_unsettled: bool = True,
        moving_reference_only: bool = False,
    ) -> NDArray[np.str_]:
        """Reason code for each visibility row. Rows are not deleted."""

        _offsets, inverse, pointing_valid, settled, moving_ref = self.pointing_state()
        return holography_row_reasons(
            self.block,
            inverse,
            pointing_valid=pointing_valid,
            settled=settled,
            moving_reference=moving_ref,
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        )

    def active_row_mask(
        self,
        *,
        mask_unsettled: bool = True,
        moving_reference_only: bool = False,
    ) -> NDArray[np.bool_]:
        """True for rows that are usable under the observation reason codes."""

        return (
            self.row_reason(
                mask_unsettled=mask_unsettled,
                moving_reference_only=moving_reference_only,
            )
            == HolographyRowReason.OK.value
        )

    def sample_mask(
        self,
        *,
        row_mask: ArrayLike | None = None,
        mask_unsettled: bool = True,
        moving_reference_only: bool = False,
    ) -> NDArray[np.bool_]:
        """Row, channel, and correlation mask for selected observation samples."""

        rows = (
            np.asarray(row_mask, dtype=bool)
            if row_mask is not None
            else self.active_row_mask(
                mask_unsettled=mask_unsettled,
                moving_reference_only=moving_reference_only,
            )
        )
        return _observation_sample_mask(self, rows)

    def predict(
        self,
        beam: VoltageBeamModel,
        *,
        backend: str = "numpy",
        moving_reference_only: bool = False,
        mask_unsettled: bool = True,
        config: BeamOperatorConfig | None = None,
    ) -> HolographyOperatorResult:
        """Evaluate the holography RIME for this bound observation."""

        source = self.source_coherency_visibility
        if source is None and self.source_model is not None:
            source = self.source_model.evaluate_coherency(self.block)
        return predict_holography_visibilities(
            self.block,
            self.pointing,
            beam,
            antenna_position_m=self.antenna_position_m,
            calibration_state=self.calibration_state,
            stokes_i=self.stokes_i,
            source_coherency_visibility=source,
            moving_reference_only=moving_reference_only,
            mask_unsettled=mask_unsettled,
            config=config,
            backend=backend,
        )


def predict_holography_visibilities(
    block: VisibilityBlock,
    pointing: ResolvedAntennaPointing,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    sky: SkyStokesPlanes | None = None,
    l_rad: ArrayLike | None = None,
    m_rad: ArrayLike | None = None,
    source_coherency_visibility: ArrayLike | None = None,
    stokes_i: float = 1.0,
    moving_reference_only: bool = False,
    mask_unsettled: bool = True,
    config: BeamOperatorConfig | None = None,
    backend: str = "numpy",
) -> HolographyOperatorResult:
    """Holography wrapper around the voltage RIME.

    Pointing changes ``E_a`` only. The correlator phase centre and Fourier
    kernel are unchanged. A provided ``source_coherency_visibility`` is the
    already-resolved ``S_pq`` and receives no extra Fourier factor.
    """

    offsets, inverse, pointing_valid, settled, moving_ref = _pointing_state_for_block(
        block, pointing
    )
    pointing_ok = pointing_valid & settled if mask_unsettled else pointing_valid
    if backend not in {"numpy", "jax"}:
        raise ValueError("holography backend must be 'numpy' or 'jax'")
    if source_coherency_visibility is not None:
        if sky is not None or l_rad is not None or m_rad is not None:
            raise ValueError("resolved source coherency cannot be mixed with a sky grid")
        result = _predict_resolved_source_holography(
            block,
            offsets,
            pointing_ok,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            source_coherency_visibility=source_coherency_visibility,
            config=config,
            backend=backend,
        )
    else:
        sky_planes = sky or SkyStokesPlanes(stokes_i=np.asarray([stokes_i], dtype=np.float64))
        directions_l = (
            np.asarray(l_rad, dtype=np.float64).reshape(-1)
            if l_rad is not None
            else np.zeros(1, dtype=np.float64)
        )
        directions_m = (
            np.asarray(m_rad, dtype=np.float64).reshape(-1)
            if m_rad is not None
            else np.zeros(1, dtype=np.float64)
        )
        predict = predict_voltage_beam
        if backend == "jax":
            from sl1mjax.voltage_operator_jax import predict_voltage_beam_jax

            predict = predict_voltage_beam_jax
        predicted = predict(
            block,
            directions_l,
            directions_m,
            sky_planes,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            config=config,
            antenna_pointing_lm_rad=offsets,
            pointing_valid=pointing_ok,
        )
        result = predicted
    visibility = np.array(result.visibility, copy=True)
    beam_valid = np.array(result.valid, copy=True)
    leakage = (
        np.array(result.off_diagonal_valid, copy=True)
        if result.off_diagonal_valid is not None
        else np.array(beam_valid, copy=True)
    )
    pointing_rows = pointing_ok[inverse, block.antenna1] & pointing_ok[inverse, block.antenna2]
    if moving_reference_only:
        visibility[~moving_ref] = 0.0
        beam_valid[~moving_ref] = False
        leakage[~moving_ref] = False
    provenance = dict(result.provenance)
    provenance.update(
        {
            "operator": "holography_voltage",
            "backend": backend,
            "selected_column": pointing.selected_column,
            "offset_sign": pointing.offset_sign,
            "moving_reference_only": moving_reference_only,
            "mask_unsettled": mask_unsettled,
            "source_model": (
                "resolved_coherency_visibility"
                if source_coherency_visibility is not None
                else "phase_centre_sky"
            ),
        }
    )
    return HolographyOperatorResult(
        visibility=visibility,
        beam_valid=beam_valid,
        pointing_valid=np.broadcast_to(pointing_rows[:, None], beam_valid.shape).copy(),
        off_diagonal_valid=leakage,
        moving_reference_rows=moving_ref,
        provenance=provenance,
        row_reason=holography_row_reasons(
            block,
            inverse,
            pointing_valid=pointing_valid,
            settled=settled,
            moving_reference=moving_ref,
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        ),
    )


def holography_row_reasons(
    block: VisibilityBlock,
    time_inverse: np.ndarray,
    *,
    pointing_valid: np.ndarray,
    settled: np.ndarray,
    moving_reference: np.ndarray,
    mask_unsettled: bool,
    moving_reference_only: bool,
) -> NDArray[np.str_]:
    """Reason code for each visibility row. Rows are not deleted."""

    reasons = np.full(block.time_s.shape[0], HolographyRowReason.OK.value, dtype="U32")
    for row, time_index in enumerate(time_inverse):
        antenna_p = int(block.antenna1[row])
        antenna_q = int(block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            reasons[row] = HolographyRowReason.MISSING_POINTING.value
        elif mask_unsettled and not (
            settled[time_index, antenna_p] and settled[time_index, antenna_q]
        ):
            reasons[row] = HolographyRowReason.UNSETTLED.value
        elif moving_reference_only and not bool(moving_reference[row]):
            reasons[row] = HolographyRowReason.NOT_MOVING_REFERENCE.value
    return reasons


def _predict_resolved_source_holography(
    block: VisibilityBlock,
    offsets: np.ndarray,
    pointing_ok: np.ndarray,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    source_coherency_visibility: ArrayLike,
    config: BeamOperatorConfig | None,
    backend: str = "numpy",
) -> BeamOperatorResult:
    from sl1mjax.beam_operator import (
        JONES_RECEPTORS,
        _aligned_antenna_jones,
        _evaluate_timestep_antennas,
    )

    source = np.asarray(source_coherency_visibility, dtype=np.complex128)
    expected = (block.visibility.shape[0], block.frequency_hz.size, 2, 2)
    if source.shape != expected:
        raise ValueError(f"source_coherency_visibility must have shape {expected}")
    selected = config or BeamOperatorConfig()
    if backend == "jax":
        return _predict_resolved_source_holography_jax(
            block,
            offsets,
            pointing_ok,
            beam,
            antenna_position_m=antenna_position_m,
            calibration_state=calibration_state,
            source=source,
            config=selected,
        )
    unique_times, inverse = unique_visibility_times(block.time_s)
    prediction = np.zeros(block.visibility.shape, dtype=np.complex128)
    valid = np.zeros(block.visibility.shape[:2], dtype=bool)
    leakage = np.zeros(block.visibility.shape[:2], dtype=bool)
    antennas = np.arange(block.antenna_count, dtype=np.int32)
    last = None
    for time_index, time_s in enumerate(unique_times):
        rows = np.flatnonzero(inverse == time_index)
        keep = (
            pointing_ok[time_index, block.antenna1[rows]]
            & pointing_ok[time_index, block.antenna2[rows]]
        )
        rows = rows[keep]
        if rows.size == 0:
            continue
        evaluation, _ = _evaluate_timestep_antennas(
            block,
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            float(time_s),
            beam=beam,
            antenna_id=antennas,
            antenna_position_m=np.asarray(antenna_position_m, dtype=np.float64),
            calibration_state=calibration_state,
            pointing_offset_lm_rad=offsets[time_index],
        )
        last = evaluation
        jones, valid_jones, off_jones = _aligned_antenna_jones(evaluation, block.antenna_count)
        for row in rows:
            antenna_p = int(block.antenna1[row])
            antenna_q = int(block.antenna2[row])
            plane_p = 0 if jones.shape[0] == 1 else antenna_p
            plane_q = 0 if jones.shape[0] == 1 else antenna_q
            for channel in range(block.frequency_hz.size):
                if not (valid_jones[plane_p, 0, channel] and valid_jones[plane_q, 0, channel]):
                    continue
                apparent = apply_jones_to_coherency(
                    source[row, channel],
                    jones[plane_p, 0, channel],
                    jones[plane_q, 0, channel],
                )
                prediction[row, channel] = unpack_coherency(
                    apparent,
                    block.correlations,
                    JONES_RECEPTORS,
                )
                valid[row, channel] = True
                leakage[row, channel] = bool(
                    off_jones[plane_p, 0, channel] and off_jones[plane_q, 0, channel]
                )
    return BeamOperatorResult(
        visibility=prediction,
        valid=valid,
        provenance={
            "operator": "holography_resolved_source",
            "policy": selected.policy.value,
            "fourier_factor": "included_in_source_coherency",
        },
        last_evaluation=last,
        off_diagonal_valid=leakage,
    )


def _predict_resolved_source_holography_jax(
    block: VisibilityBlock,
    offsets: np.ndarray,
    pointing_ok: np.ndarray,
    beam: VoltageBeamModel,
    *,
    antenna_position_m: ArrayLike,
    calibration_state: BeamCalibrationState | str,
    source: np.ndarray,
    config: BeamOperatorConfig,
) -> BeamOperatorResult:
    import jax.numpy as jnp

    from sl1mjax.voltage_operator_jax import (
        _unpack_correlations,
        evaluate_antenna_jones_jax,
    )

    unique_times, inverse = unique_visibility_times(block.time_s)
    jones, valid, leakage = evaluate_antenna_jones_jax(
        block,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        beam,
        antenna_position_m=antenna_position_m,
        calibration_state=calibration_state,
        antenna_pointing_lm_rad=offsets,
        pointing_valid=pointing_ok,
        config=config,
    )
    time_index = jnp.asarray(inverse, dtype=jnp.int32)
    antenna_p = jnp.asarray(block.antenna1, dtype=jnp.int32)
    antenna_q = jnp.asarray(block.antenna2, dtype=jnp.int32)
    jones_j = jnp.asarray(jones)
    valid_j = jnp.asarray(valid)
    leakage_j = jnp.asarray(leakage)
    source_j = jnp.asarray(source)
    jp = jones_j[time_index, antenna_p, 0]
    jq = jones_j[time_index, antenna_q, 0]
    apparent = jnp.matmul(jp, jnp.matmul(source_j, jnp.conjugate(jnp.swapaxes(jq, -1, -2))))
    packed = _unpack_correlations(apparent, block.correlations)
    row_ok = (
        jnp.asarray(pointing_ok)[time_index, antenna_p]
        & jnp.asarray(pointing_ok)[time_index, antenna_q]
    )
    beam_ok = valid_j[time_index, antenna_p, 0] & valid_j[time_index, antenna_q, 0]
    leak_ok = leakage_j[time_index, antenna_p, 0] & leakage_j[time_index, antenna_q, 0]
    usable = row_ok[:, None] & beam_ok
    prediction = jnp.where(usable[..., None], packed, 0.0)
    return BeamOperatorResult(
        visibility=np.asarray(prediction),
        valid=np.asarray(usable),
        provenance={
            "operator": "holography_resolved_source",
            "backend": "jax",
            "policy": config.policy.value,
            "fourier_factor": "included_in_source_coherency",
        },
        off_diagonal_valid=np.asarray(row_ok[:, None] & leak_ok),
    )


def _pointing_state_for_block(
    block: VisibilityBlock,
    pointing: ResolvedAntennaPointing,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_times, inverse = unique_visibility_times(block.time_s)
    offsets, pointing_valid = pointing_planes_for_block(pointing, unique_times, block.antenna_count)
    settled = np.zeros_like(pointing_valid)
    for index, antenna in enumerate(pointing.antenna_id):
        settled[:, int(antenna)] = pointing.settled[:, index]
    moving_ref = moving_reference_row_mask(block, pointing)
    return offsets, inverse, pointing_valid, settled, moving_ref


def _correlation_names(correlations: tuple[Any, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for item in correlations:
        names.append(item.value if isinstance(item, Correlation) else str(item))
    return tuple(names)


def _observation_sample_mask(
    observation: HolographyObservation,
    row_mask: ArrayLike,
) -> NDArray[np.bool_]:
    rows = np.asarray(row_mask, dtype=bool).reshape(-1)
    if rows.shape[0] != observation.block.time_s.shape[0]:
        raise ValueError("row_mask must have one entry per visibility row")
    names = _correlation_names(observation.block.correlations)
    selected = set(observation.selected_correlations)
    corr_ok = np.array([name in selected for name in names], dtype=bool)
    frequencies = np.asarray(observation.block.frequency_hz, dtype=np.float64).reshape(-1)
    selected_hz = np.asarray(observation.selected_frequency_hz, dtype=np.float64).reshape(-1)
    freq_ok = np.any(
        np.isclose(frequencies[:, None], selected_hz[None, :], rtol=0.0, atol=1.0),
        axis=1,
    )
    flag = np.asarray(observation.block.flag, dtype=bool)
    return rows[:, None, None] & freq_ok[None, :, None] & corr_ok[None, None, :] & ~flag


def _observation_for_convention(
    observation: HolographyObservation,
    hypothesis: HolographyConventionHypothesis,
) -> HolographyObservation:
    bound = _observation_with_pointing(
        observation,
        apply_convention_to_pointing(observation.pointing, hypothesis),
    )
    if not hypothesis.swap_baseline_order:
        return bound
    return replace(
        bound,
        block=replace(
            bound.block,
            antenna1=np.asarray(bound.block.antenna2, dtype=np.int32),
            antenna2=np.asarray(bound.block.antenna1, dtype=np.int32),
        ),
    )


def _observation_with_pointing(
    observation: HolographyObservation,
    pointing: ResolvedAntennaPointing,
) -> HolographyObservation:
    return HolographyObservation(
        block=observation.block,
        pointing=pointing,
        antenna_position_m=observation.antenna_position_m,
        calibration_state=observation.calibration_state,
        phase_centre_rad=observation.phase_centre_rad,
        source_name=observation.source_name,
        stokes_i=observation.stokes_i,
        source_coherency_visibility=observation.source_coherency_visibility,
        source_model=observation.source_model,
        selected_correlations=observation.selected_correlations,
        selected_frequency_hz=observation.selected_frequency_hz,
        selected_spw_id=observation.selected_spw_id,
        provenance=observation.provenance,
    )


@dataclass(frozen=True)
class HolographyConventionHypothesis:
    """One named transform of pointing coordinates or packed visibilities."""

    name: str
    offset_sign: OffsetSign = "commanded_pointing"
    flip_l: bool = False
    flip_m: bool = False
    swap_lm: bool = False
    conjugate_visibility: bool = False
    exchange_rl: bool = False
    exchange_receptors: bool = False
    chi_sign: Literal[1, -1] = 1
    swap_baseline_order: bool = False
    apply_parallactic: bool = True

    def is_identity(self) -> bool:
        return (
            self.offset_sign == "commanded_pointing"
            and not self.flip_l
            and not self.flip_m
            and not self.swap_lm
            and not self.conjugate_visibility
            and not self.exchange_rl
            and not self.exchange_receptors
            and self.chi_sign == 1
            and not self.swap_baseline_order
            and self.apply_parallactic
        )


DEFAULT_HOLOGRAPHY_CONVENTIONS: tuple[HolographyConventionHypothesis, ...] = (
    HolographyConventionHypothesis(name="commanded_pointing"),
    HolographyConventionHypothesis(
        name="source_minus_pointing",
        offset_sign="source_minus_pointing",
    ),
    HolographyConventionHypothesis(name="flip_l", flip_l=True),
    HolographyConventionHypothesis(name="flip_m", flip_m=True),
    HolographyConventionHypothesis(name="swap_lm", swap_lm=True),
    HolographyConventionHypothesis(
        name="conjugate_visibility",
        conjugate_visibility=True,
    ),
    HolographyConventionHypothesis(name="exchange_rl", exchange_rl=True),
    HolographyConventionHypothesis(
        name="exchange_receptors",
        exchange_receptors=True,
    ),
    HolographyConventionHypothesis(name="flip_chi", chi_sign=-1),
    HolographyConventionHypothesis(
        name="mount_frame_no_parallactic",
        apply_parallactic=False,
    ),
    HolographyConventionHypothesis(
        name="swap_baseline_order",
        swap_baseline_order=True,
    ),
)


def apply_convention_to_offsets(
    offset_lm_rad: ArrayLike,
    hypothesis: HolographyConventionHypothesis,
) -> NDArray[np.float64]:
    """Transform commanded pointing offsets under one convention hypothesis."""

    offsets = np.array(offset_lm_rad, dtype=np.float64, copy=True)
    if hypothesis.offset_sign == "source_minus_pointing":
        offsets = -offsets
    elif hypothesis.offset_sign != "commanded_pointing":
        raise ValueError(f"unsupported offset_sign {hypothesis.offset_sign!r}")
    if hypothesis.flip_l:
        offsets[..., 0] *= -1.0
    if hypothesis.flip_m:
        offsets[..., 1] *= -1.0
    if hypothesis.swap_lm:
        offsets = np.stack((offsets[..., 1], offsets[..., 0]), axis=-1)
    return offsets


def apply_convention_to_visibility(
    visibility: ArrayLike,
    correlations: tuple[Any, ...],
    hypothesis: HolographyConventionHypothesis,
) -> NDArray[np.complex128]:
    """Transform packed visibilities under conjugation or receptor-order hypotheses."""

    names = _correlation_names(correlations)
    index = {name: i for i, name in enumerate(names)}
    packed = np.array(visibility, dtype=np.complex128, copy=True)
    if hypothesis.conjugate_visibility:
        packed = np.conjugate(packed)
    if hypothesis.exchange_rl and "RL" in index and "LR" in index:
        swapped = np.array(packed, copy=True)
        swapped[..., index["RL"]] = packed[..., index["LR"]]
        swapped[..., index["LR"]] = packed[..., index["RL"]]
        packed = swapped
    if hypothesis.exchange_receptors:
        exchanged = np.array(packed, copy=True)
        if "RR" in index and "LL" in index:
            exchanged[..., index["RR"]] = packed[..., index["LL"]]
            exchanged[..., index["LL"]] = packed[..., index["RR"]]
        if "RL" in index and "LR" in index:
            exchanged[..., index["RL"]] = packed[..., index["LR"]]
            exchanged[..., index["LR"]] = packed[..., index["RL"]]
        packed = exchanged
    return packed


def apply_convention_to_pointing(
    pointing: ResolvedAntennaPointing,
    hypothesis: HolographyConventionHypothesis,
) -> ResolvedAntennaPointing:
    """Return a copy of ``pointing`` with hypothesis-transformed offsets."""

    return replace(
        pointing,
        offset_lm_rad=apply_convention_to_offsets(pointing.offset_lm_rad, hypothesis),
        offset_sign=hypothesis.offset_sign,
        notes=pointing.notes + (f"convention:{hypothesis.name}",),
    )


@dataclass(frozen=True)
class HolographyConventionScore:
    name: str
    train_mse: float
    holdout_mse: float
    rr_ll_mse: float
    rl_lr_mse: float
    scale: complex
    n_train: int
    n_holdout: int


@dataclass(frozen=True)
class HolographyConventionLadder:
    """Held-out ranking of pointing and packing conventions.

    A winner is accepted only when its holdout loss is uniquely better.
    The ladder never freezes a beam.
    """

    scores: tuple[HolographyConventionScore, ...]
    winner: str | None
    status: HolographyGateStatus
    notes: tuple[str, ...] = ()

    def score(self, name: str) -> HolographyConventionScore:
        for item in self.scores:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class HolographyHoldoutSplit:
    """Named deterministic train/holdout masks. Sealed splits cannot rank models."""

    name: str
    axis: HolographyHoldoutAxis
    train_row_mask: NDArray[np.bool_]
    holdout_row_mask: NDArray[np.bool_]
    train_channel_mask: NDArray[np.bool_] | None = None
    holdout_channel_mask: NDArray[np.bool_] | None = None
    train_correlation_mask: NDArray[np.bool_] | None = None
    holdout_correlation_mask: NDArray[np.bool_] | None = None
    sealed: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        train_rows = np.asarray(self.train_row_mask, dtype=bool).reshape(-1)
        holdout_rows = np.asarray(self.holdout_row_mask, dtype=bool).reshape(-1)
        if train_rows.shape != holdout_rows.shape:
            raise ValueError("train and holdout row masks must have the same shape")
        object.__setattr__(self, "train_row_mask", train_rows)
        object.__setattr__(self, "holdout_row_mask", holdout_rows)
        object.__setattr__(self, "axis", HolographyHoldoutAxis(self.axis))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.axis not in {
            HolographyHoldoutAxis.FREQUENCY,
            HolographyHoldoutAxis.CORRELATION,
        } and np.any(train_rows & holdout_rows):
            raise ValueError("train and holdout row masks must be disjoint")
        for label, mask in (
            ("train_channel_mask", self.train_channel_mask),
            ("holdout_channel_mask", self.holdout_channel_mask),
            ("train_correlation_mask", self.train_correlation_mask),
            ("holdout_correlation_mask", self.holdout_correlation_mask),
        ):
            if mask is None:
                continue
            object.__setattr__(self, label, np.asarray(mask, dtype=bool).reshape(-1))


def default_time_holdout_masks(
    block: VisibilityBlock,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Split unique times in order: first half train, second half holdout."""

    split = holography_holdout_split_from_block(block, HolographyHoldoutAxis.TIME)
    return split.train_row_mask, split.holdout_row_mask


def holography_holdout_split(
    observation: HolographyObservation,
    axis: HolographyHoldoutAxis | str,
    *,
    holdout_antenna_id: int | None = None,
    sealed: bool = False,
) -> HolographyHoldoutSplit:
    """Return one named deterministic split. Does not draw a random mask."""

    selected = HolographyHoldoutAxis(axis)
    n_row = int(observation.block.time_s.shape[0])
    n_chan = int(observation.block.frequency_hz.size)
    names = _correlation_names(observation.block.correlations)
    if selected is HolographyHoldoutAxis.TIME:
        split = holography_holdout_split_from_block(observation.block, selected)
        return replace(split, sealed=sealed)
    if selected is HolographyHoldoutAxis.SPATIAL:
        cells = _moving_offset_cell_ids(observation)
        unique = np.array(sorted(set(int(cell) for cell in cells if cell >= 0)), dtype=np.int64)
        if unique.size < 2:
            raise ValueError("spatial holdout needs at least two moving raster cells")
        holdout_cells = set(int(cell) for cell in unique[1::2])
        holdout = np.array([int(cell) in holdout_cells for cell in cells], dtype=bool)
        train = (cells >= 0) & ~holdout
        return HolographyHoldoutSplit(
            name="spatial_raster_cells",
            axis=selected,
            train_row_mask=train,
            holdout_row_mask=holdout,
            sealed=sealed,
            notes=("checkerboard of quantized moving-antenna raster cells",),
        )
    if selected is HolographyHoldoutAxis.REFERENCE_ANTENNA:
        antenna = _require_holdout_antenna(observation, holdout_antenna_id, "reference")
        holdout = _rows_with_role(observation, antenna, AntennaPointingRole.REFERENCE)
        train = _rows_with_any_role(observation, AntennaPointingRole.REFERENCE) & ~holdout
        if not bool(np.any(train)) or not bool(np.any(holdout)):
            raise ValueError("reference-antenna holdout needs at least two reference antennas")
        return HolographyHoldoutSplit(
            name=f"reference_antenna_{antenna}",
            axis=selected,
            train_row_mask=train,
            holdout_row_mask=holdout,
            sealed=sealed,
            notes=(f"hold out reference antenna {antenna}",),
        )
    if selected is HolographyHoldoutAxis.MOVING_ANTENNA:
        antenna = _require_holdout_antenna(observation, holdout_antenna_id, "moving")
        holdout = _rows_with_role(observation, antenna, AntennaPointingRole.MOVING)
        train = _rows_with_any_role(observation, AntennaPointingRole.MOVING) & ~holdout
        if not bool(np.any(train)) or not bool(np.any(holdout)):
            raise ValueError("moving-antenna holdout needs at least two moving antennas")
        return HolographyHoldoutSplit(
            name=f"moving_antenna_{antenna}",
            axis=selected,
            train_row_mask=train,
            holdout_row_mask=holdout,
            sealed=sealed,
            notes=(f"hold out moving antenna {antenna}",),
        )
    if selected is HolographyHoldoutAxis.FREQUENCY:
        if n_chan < 2:
            raise ValueError("frequency holdout needs at least two channels")
        train_chan = np.arange(n_chan) % 2 == 0
        return HolographyHoldoutSplit(
            name="frequency_channels",
            axis=selected,
            train_row_mask=np.ones(n_row, dtype=bool),
            holdout_row_mask=np.ones(n_row, dtype=bool),
            train_channel_mask=train_chan,
            holdout_channel_mask=~train_chan,
            sealed=sealed,
            notes=("interleaved channels; even train, odd holdout",),
        )
    if selected is HolographyHoldoutAxis.CORRELATION:
        copolar = np.array([name in {"RR", "LL"} for name in names], dtype=bool)
        cross = np.array([name in {"RL", "LR"} for name in names], dtype=bool)
        if not bool(np.any(copolar)) or not bool(np.any(cross)):
            raise ValueError("correlation holdout needs parallel and cross hands")
        return HolographyHoldoutSplit(
            name="correlation_cross_hands",
            axis=selected,
            train_row_mask=np.ones(n_row, dtype=bool),
            holdout_row_mask=np.ones(n_row, dtype=bool),
            train_correlation_mask=copolar,
            holdout_correlation_mask=cross,
            sealed=sealed,
            notes=("train on RR/LL; hold out RL/LR",),
        )
    if selected is HolographyHoldoutAxis.PARALLACTIC_ANGLE:
        chi = _row_parallactic_rad(observation)
        unique = np.unique(np.round(chi, decimals=8))
        if unique.size < 2:
            raise ValueError("parallactic holdout needs at least two distinct χ values")
        cut = unique[unique.size // 2]
        holdout = chi >= cut
        return HolographyHoldoutSplit(
            name="parallactic_angle",
            axis=selected,
            train_row_mask=~holdout,
            holdout_row_mask=holdout,
            sealed=sealed,
            notes=("hold out the upper half of row parallactic angles",),
        )
    raise ValueError(f"unsupported holography holdout axis {axis!r}")


def holography_holdout_split_from_block(
    block: VisibilityBlock,
    axis: HolographyHoldoutAxis,
) -> HolographyHoldoutSplit:
    if axis is not HolographyHoldoutAxis.TIME:
        raise ValueError("block-only construction supports the time axis")
    _unique, inverse = unique_visibility_times(block.time_s)
    n_time = int(_unique.size)
    if n_time < 2:
        raise ValueError("time holdout needs at least two unique times")
    train = inverse % 2 == 0
    return HolographyHoldoutSplit(
        name="time_unique_interleaved",
        axis=axis,
        train_row_mask=train,
        holdout_row_mask=~train,
        notes=("interleaved unique times; even train, odd holdout",),
    )


def holdout_sample_masks(
    observation: HolographyObservation,
    split: HolographyHoldoutSplit,
    *,
    mask_unsettled: bool = True,
    moving_reference_only: bool = False,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Expand a named split onto selected observation samples."""

    if split.sealed:
        raise ValueError("sealed holdout cannot be used to rank conventions or beams")
    active = observation.active_row_mask(
        mask_unsettled=mask_unsettled,
        moving_reference_only=moving_reference_only,
    )
    train = observation.sample_mask(row_mask=split.train_row_mask & active)
    holdout = observation.sample_mask(row_mask=split.holdout_row_mask & active)
    if split.train_channel_mask is not None:
        train = train & split.train_channel_mask[None, :, None]
        holdout = holdout & split.holdout_channel_mask[None, :, None]
    if split.train_correlation_mask is not None:
        train = train & split.train_correlation_mask[None, None, :]
        holdout = holdout & split.holdout_correlation_mask[None, None, :]
    return train, holdout


def point_source_is_adequate(
    observation: HolographyObservation,
    structured: HolographySourceModel,
    beam: VoltageBeamModel,
    *,
    split: HolographyHoldoutSplit | None = None,
    relative_tolerance: float = 1.0e-3,
) -> tuple[bool, float]:
    """True when a point ``S_pq`` matches a structured model on the holdout.

    Used to decide whether a point model is adequate on moving--reference
    baselines. The beam is not fitted.
    """

    if observation.source_model is None:
        point = three_c147_point_source_model(observation.block.frequency_hz)
    elif observation.source_model.kind == "point":
        point = observation.source_model
    else:
        point = three_c147_point_source_model(observation.block.frequency_hz)
    selected = split or holography_holdout_split(observation, HolographyHoldoutAxis.TIME)
    _train, holdout = holdout_sample_masks(observation, selected)
    point_obs = replace(observation, source_model=point, source_coherency_visibility=None)
    struct_obs = replace(observation, source_model=structured, source_coherency_visibility=None)
    predicted = point_obs.predict(beam).visibility
    target = struct_obs.predict(beam).visibility
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    residual = _weighted_mse(predicted, target, weight, holdout)
    scale = _weighted_mse(target, np.zeros_like(target), weight, holdout)
    relative = residual / scale if scale > 0.0 else residual
    return bool(relative <= relative_tolerance), float(relative)


def _moving_offset_cell_ids(observation: HolographyObservation) -> NDArray[np.int64]:
    offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full(offsets.shape[:2], "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    cells = np.full(observation.block.time_s.shape[0], -1, dtype=np.int64)
    quant = 180.0 * 60.0 / np.pi * 100.0
    seen: dict[tuple[int, int], int] = {}
    next_id = 0
    for row, time_index in enumerate(inverse):
        for antenna in (
            int(observation.block.antenna1[row]),
            int(observation.block.antenna2[row]),
        ):
            if not pointing_valid[time_index, antenna]:
                continue
            if roles[time_index, antenna] != AntennaPointingRole.MOVING.value:
                continue
            key = (
                int(np.round(offsets[time_index, antenna, 0] * quant)),
                int(np.round(offsets[time_index, antenna, 1] * quant)),
            )
            if key not in seen:
                seen[key] = next_id
                next_id += 1
            cells[row] = seen[key]
    return cells


def _require_holdout_antenna(
    observation: HolographyObservation,
    antenna_id: int | None,
    role: str,
) -> int:
    if antenna_id is None:
        raise ValueError(f"{role}-antenna holdout requires holdout_antenna_id")
    if int(antenna_id) not in {int(item) for item in observation.pointing.antenna_id}:
        raise ValueError(f"holdout antenna {antenna_id} is not in the pointing table")
    return int(antenna_id)


def _rows_with_role(
    observation: HolographyObservation,
    antenna_id: int,
    role: AntennaPointingRole,
) -> NDArray[np.bool_]:
    _offsets, inverse, _valid, _settled, _moving = observation.pointing_state()
    index = int(np.flatnonzero(observation.pointing.antenna_id == antenna_id)[0])
    mask = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    for row, time_index in enumerate(inverse):
        if observation.pointing.role[time_index, index] != role.value:
            continue
        if (
            int(observation.block.antenna1[row]) == antenna_id
            or int(observation.block.antenna2[row]) == antenna_id
        ):
            mask[row] = True
    return mask


def _rows_with_any_role(
    observation: HolographyObservation,
    role: AntennaPointingRole,
) -> NDArray[np.bool_]:
    mask = np.zeros(observation.block.time_s.shape[0], dtype=bool)
    for antenna in observation.pointing.antenna_id:
        mask = mask | _rows_with_role(observation, int(antenna), role)
    return mask


def _row_parallactic_rad(observation: HolographyObservation) -> NDArray[np.float64]:
    from sl1mjax.calibration_terms import parallactic_angle_rad

    chi = parallactic_angle_rad(
        observation.block.time_s,
        observation.phase_centre_rad,
        observation.antenna_position_m[: observation.block.antenna_count],
    )
    return 0.5 * (
        chi[np.arange(observation.block.time_s.shape[0]), observation.block.antenna1]
        + chi[np.arange(observation.block.time_s.shape[0]), observation.block.antenna2]
    )


def score_holography_convention_ladder(
    observation: HolographyObservation,
    beam: VoltageBeamModel,
    target_visibility: ArrayLike,
    *,
    hypotheses: tuple[HolographyConventionHypothesis, ...] = DEFAULT_HOLOGRAPHY_CONVENTIONS,
    split: HolographyHoldoutSplit | None = None,
    train_mask: ArrayLike | None = None,
    holdout_mask: ArrayLike | None = None,
    backend: str = "numpy",
    moving_reference_only: bool = False,
    mask_unsettled: bool = True,
    config: BeamOperatorConfig | None = None,
) -> HolographyConventionLadder:
    """Predict under each hypothesis; normalize on train; rank on holdout.

    Known manufactured sign and axis errors must lose. The ladder does not
    estimate or freeze a beam map.
    """

    if not hypotheses:
        raise ValueError("convention ladder needs at least one hypothesis")
    target = np.asarray(target_visibility, dtype=np.complex128)
    if target.shape != observation.block.visibility.shape:
        raise ValueError("target_visibility must match the observation visibility shape")
    if split is not None:
        if train_mask is not None or holdout_mask is not None:
            raise ValueError("pass split or explicit row masks, not both")
        train_samples, holdout_samples = holdout_sample_masks(
            observation,
            split,
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        )
    else:
        if train_mask is None or holdout_mask is None:
            default_train, default_holdout = default_time_holdout_masks(observation.block)
            train_rows = default_train if train_mask is None else np.asarray(train_mask, dtype=bool)
            holdout_rows = (
                default_holdout if holdout_mask is None else np.asarray(holdout_mask, dtype=bool)
            )
        else:
            train_rows = np.asarray(train_mask, dtype=bool)
            holdout_rows = np.asarray(holdout_mask, dtype=bool)
        if (
            train_rows.shape != observation.block.time_s.shape
            or holdout_rows.shape != train_rows.shape
        ):
            raise ValueError("train and holdout masks must have one entry per visibility row")
        if np.any(train_rows & holdout_rows):
            raise ValueError("train and holdout masks must be disjoint")
        active = observation.active_row_mask(
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        )
        train_samples = observation.sample_mask(row_mask=train_rows & active)
        holdout_samples = observation.sample_mask(row_mask=holdout_rows & active)
    if not bool(np.any(train_samples)) or not bool(np.any(holdout_samples)):
        raise ValueError("convention ladder needs usable training and holdout samples")
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    names = _correlation_names(observation.block.correlations)
    copolar = _named_correlation_mask(names, ("RR", "LL"))
    cross = _named_correlation_mask(names, ("RL", "LR"))
    scores: list[HolographyConventionScore] = []
    for hypothesis in hypotheses:
        if (
            hypothesis.chi_sign != 1 or not hypothesis.apply_parallactic
        ) and not _beam_uses_parallactic(beam):
            continue
        predicted = _observation_for_convention(observation, hypothesis).predict(
            _beam_for_convention(beam, hypothesis),
            backend=backend,
            moving_reference_only=moving_reference_only,
            mask_unsettled=mask_unsettled,
            config=config,
        )
        model = apply_convention_to_visibility(
            predicted.visibility,
            observation.block.correlations,
            hypothesis,
        )
        scale = _complex_scale(model, target, weight, train_samples)
        scaled = scale * model
        scores.append(
            HolographyConventionScore(
                name=hypothesis.name,
                train_mse=_weighted_mse(scaled, target, weight, train_samples),
                holdout_mse=_weighted_mse(scaled, target, weight, holdout_samples),
                rr_ll_mse=_weighted_mse(
                    scaled, target, weight, holdout_samples & copolar[None, None, :]
                ),
                rl_lr_mse=_weighted_mse(
                    scaled, target, weight, holdout_samples & cross[None, None, :]
                ),
                scale=complex(scale),
                n_train=int(np.count_nonzero(train_samples)),
                n_holdout=int(np.count_nonzero(holdout_samples)),
            )
        )
    ranked = tuple(sorted(scores, key=lambda item: (item.holdout_mse, item.name)))
    winner, status, notes = _unique_holdout_winner(ranked)
    return HolographyConventionLadder(
        scores=tuple(scores),
        winner=winner,
        status=status,
        notes=notes,
    )


def _beam_uses_parallactic(beam: VoltageBeamModel) -> bool:
    return bool(
        getattr(beam, "antenna_planes_from_parallactic", False)
        or getattr(beam, "rotate_parallactic", False)
    )


def _beam_for_convention(
    beam: VoltageBeamModel,
    hypothesis: HolographyConventionHypothesis,
) -> VoltageBeamModel:
    if not hypothesis.apply_parallactic:
        return _FixedParallacticVoltageBeam(beam, 0.0)
    if hypothesis.chi_sign == 1:
        return beam
    return _ParallacticSignVoltageBeam(beam, hypothesis.chi_sign)


class _FixedParallacticVoltageBeam:
    """Evaluate an inner beam at a fixed ``χ``. Diagnostic wrapper only."""

    def __init__(self, inner: VoltageBeamModel, chi_rad: float) -> None:
        self._inner = inner
        self._chi_rad = float(chi_rad)
        self.model_id = getattr(inner, "model_id", "fixed_parallactic")
        self.antenna_planes_from_parallactic = False

    def evaluate(self, coordinates, *, calibration_state):
        fixed = replace(
            coordinates,
            parallactic_angle_rad=np.full(
                np.asarray(coordinates.parallactic_angle_rad).shape,
                self._chi_rad,
                dtype=np.float64,
            ),
        )
        return self._inner.evaluate(fixed, calibration_state=calibration_state)


class _ParallacticSignVoltageBeam:
    """Evaluate an inner beam at ``chi_sign * χ``. Diagnostic wrapper only."""

    def __init__(self, inner: VoltageBeamModel, chi_sign: int) -> None:
        if chi_sign not in {-1, 1}:
            raise ValueError("chi_sign must be +1 or -1")
        self._inner = inner
        self._chi_sign = int(chi_sign)
        self.model_id = getattr(inner, "model_id", "parallactic_sign")
        self.antenna_planes_from_parallactic = True

    def evaluate(self, coordinates, *, calibration_state):
        if self._chi_sign == 1:
            return self._inner.evaluate(coordinates, calibration_state=calibration_state)
        flipped = replace(
            coordinates,
            parallactic_angle_rad=-np.asarray(coordinates.parallactic_angle_rad, dtype=np.float64),
        )
        return self._inner.evaluate(flipped, calibration_state=calibration_state)


def _named_correlation_mask(
    names: tuple[str, ...],
    wanted: tuple[str, ...],
) -> NDArray[np.bool_]:
    wanted_set = set(wanted)
    return np.array([name in wanted_set for name in names], dtype=bool)


def _complex_scale(
    model: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> complex:
    usable = np.asarray(mask, dtype=bool) & np.isfinite(weight) & (weight > 0.0)
    if not bool(np.any(usable)):
        return 1.0 + 0.0j
    w = np.where(usable, weight, 0.0)
    predicted = np.where(usable, model, 0.0)
    observed = np.where(usable, target, 0.0)
    denominator = np.sum(w * np.abs(predicted) ** 2)
    if float(np.abs(denominator)) <= 0.0:
        return 1.0 + 0.0j
    return complex(np.sum(w * observed * np.conjugate(predicted)) / denominator)


def _weighted_mse(
    model: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
) -> float:
    usable = np.asarray(mask, dtype=bool) & np.isfinite(weight) & (weight > 0.0)
    if not bool(np.any(usable)):
        return float("nan")
    w = np.where(usable, weight, 0.0)
    residual = np.where(usable, model - target, 0.0)
    return float(np.sum(w * np.abs(residual) ** 2) / np.sum(w))


def _unique_holdout_winner(
    ranked: tuple[HolographyConventionScore, ...],
) -> tuple[str | None, HolographyGateStatus, tuple[str, ...]]:
    if not ranked:
        return None, "fail", ("no convention scores",)
    best = ranked[0]
    if len(ranked) == 1:
        return best.name, "pass", ("single hypothesis",)
    second = ranked[1]
    gap = float(second.holdout_mse - best.holdout_mse)
    unique = (
        np.isfinite(best.holdout_mse)
        and np.isfinite(second.holdout_mse)
        and second.holdout_mse > best.holdout_mse
    )
    if unique:
        return (
            best.name,
            "pass",
            (f"holdout gap {gap:.6g} versus {second.name}",),
        )
    return (
        None,
        "warn",
        (
            "no unique holdout winner; preserve the report and do not freeze",
            f"best={best.name} mse={best.holdout_mse:.6g}",
            f"next={second.name} mse={second.holdout_mse:.6g}",
        ),
    )


@dataclass(frozen=True)
class HolographyBeamPredictionScore:
    name: str
    train_mse: float
    holdout_mse: float
    rr_ll_mse: float
    rl_lr_mse: float
    dense_mse: float
    sparse_mse: float
    scale: complex
    experimental: bool = False


@dataclass(frozen=True)
class HolographyBeamComparison:
    """Same-sample visibility scores for existing beams. Does not freeze."""

    scores: tuple[HolographyBeamPredictionScore, ...]
    frozen: bool = False
    status: HolographyGateStatus = "not_run"
    notes: tuple[str, ...] = ()

    def score(self, name: str) -> HolographyBeamPredictionScore:
        for item in self.scores:
            if item.name == name:
                return item
        raise KeyError(name)


def holography_comparison_beams() -> dict[str, VoltageBeamModel]:
    """Airy, Perley, CASSBEAM diagonal, and experimental CASSBEAM full Jones.

    The production full-Jones factory stays refused. Experimental off-diagonal
    CASSBEAM is constructed with ``allow_unfrozen=True`` for diagnostics only.
    """

    from sl1mjax.cassbeam_beam import CassbeamCBandVoltageBeam, voltage_beam_for_mode
    from sl1mjax.voltage_beam import AnalyticAiryVoltageBeam, Perley2016CBandVoltageBeam

    diagonal = voltage_beam_for_mode("diagonal_copolar")
    return {
        "analytic_airy_diagonal": AnalyticAiryVoltageBeam(),
        "perley_scalar_copolar": Perley2016CBandVoltageBeam(),
        "cassbeam_diagonal": diagonal,
        "cassbeam_experimental_full_jones": CassbeamCBandVoltageBeam(
            diagonal.artifact,
            off_diagonal=True,
            allow_unfrozen=True,
            outer=diagonal.outer,
        ),
    }


def compare_holography_beam_predictions(
    observation: HolographyObservation,
    beams: Mapping[str, VoltageBeamModel],
    target_visibility: ArrayLike,
    *,
    split: HolographyHoldoutSplit | None = None,
    train_mask: ArrayLike | None = None,
    holdout_mask: ArrayLike | None = None,
    backend: str = "numpy",
    moving_reference_only: bool = False,
    mask_unsettled: bool = True,
    config: BeamOperatorConfig | None = None,
    experimental: frozenset[str] | None = None,
) -> HolographyBeamComparison:
    """Score existing beams on identical holography samples. Does not freeze."""

    if not beams:
        raise ValueError("beam comparison needs at least one beam")
    target = np.asarray(target_visibility, dtype=np.complex128)
    if target.shape != observation.block.visibility.shape:
        raise ValueError("target_visibility must match the observation visibility shape")
    if split is not None:
        if train_mask is not None or holdout_mask is not None:
            raise ValueError("pass split or explicit row masks, not both")
        train_samples, holdout_samples = holdout_sample_masks(
            observation,
            split,
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        )
    else:
        if train_mask is None or holdout_mask is None:
            default_train, default_holdout = default_time_holdout_masks(observation.block)
            train_rows = default_train if train_mask is None else np.asarray(train_mask, dtype=bool)
            holdout_rows = (
                default_holdout if holdout_mask is None else np.asarray(holdout_mask, dtype=bool)
            )
        else:
            train_rows = np.asarray(train_mask, dtype=bool)
            holdout_rows = np.asarray(holdout_mask, dtype=bool)
        active = observation.active_row_mask(
            mask_unsettled=mask_unsettled,
            moving_reference_only=moving_reference_only,
        )
        train_samples = observation.sample_mask(row_mask=train_rows & active)
        holdout_samples = observation.sample_mask(row_mask=holdout_rows & active)
    if not bool(np.any(train_samples)) or not bool(np.any(holdout_samples)):
        raise ValueError("beam comparison needs usable training and holdout samples")
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    names = _correlation_names(observation.block.correlations)
    copolar = _named_correlation_mask(names, ("RR", "LL"))
    cross = _named_correlation_mask(names, ("RL", "LR"))
    radius = moving_antenna_radius_arcmin(observation)
    memo = Memo195LowerCRaster()
    dense_rows = np.isfinite(radius) & (radius <= memo.dense_radius_arcmin)
    sparse_rows = np.isfinite(radius) & (radius > memo.dense_radius_arcmin)
    marked_experimental = experimental or frozenset({"cassbeam_experimental_full_jones"})
    scores: list[HolographyBeamPredictionScore] = []
    for name, beam in beams.items():
        predicted = observation.predict(
            beam,
            backend=backend,
            moving_reference_only=moving_reference_only,
            mask_unsettled=mask_unsettled,
            config=config,
        )
        scale = _complex_scale(predicted.visibility, target, weight, train_samples)
        scaled = scale * predicted.visibility
        holdout_dense = holdout_samples & dense_rows[:, None, None]
        holdout_sparse = holdout_samples & sparse_rows[:, None, None]
        scores.append(
            HolographyBeamPredictionScore(
                name=name,
                train_mse=_weighted_mse(scaled, target, weight, train_samples),
                holdout_mse=_weighted_mse(scaled, target, weight, holdout_samples),
                rr_ll_mse=_weighted_mse(
                    scaled, target, weight, holdout_samples & copolar[None, None, :]
                ),
                rl_lr_mse=_weighted_mse(
                    scaled, target, weight, holdout_samples & cross[None, None, :]
                ),
                dense_mse=_weighted_mse(scaled, target, weight, holdout_dense),
                sparse_mse=_weighted_mse(scaled, target, weight, holdout_sparse),
                scale=complex(scale),
                experimental=name in marked_experimental,
            )
        )
    return HolographyBeamComparison(
        scores=tuple(scores),
        frozen=False,
        status="not_run",
        notes=(
            "direct prediction comparison; not a freeze decision",
            "CASSBEAM full Jones remains experimental and unfrozen",
        ),
    )


def moving_antenna_radius_arcmin(observation: HolographyObservation) -> NDArray[np.float64]:
    """Radius of the moving antenna pointing for each visibility row."""

    offsets, inverse, pointing_valid, _settled, _moving_ref = observation.pointing_state()
    roles = np.full((offsets.shape[0], offsets.shape[1]), "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    radius = np.full(observation.block.time_s.shape[0], np.nan, dtype=np.float64)
    rad2arcmin = 180.0 * 60.0 / np.pi
    for row, time_index in enumerate(inverse):
        for antenna in (
            int(observation.block.antenna1[row]),
            int(observation.block.antenna2[row]),
        ):
            if not pointing_valid[time_index, antenna]:
                continue
            if roles[time_index, antenna] != AntennaPointingRole.MOVING.value:
                continue
            offset = offsets[time_index, antenna]
            radius[row] = float(np.hypot(offset[0], offset[1]) * rad2arcmin)
    return radius


@dataclass(frozen=True)
class HolographyRasterTrack:
    antenna_id: int
    role: str
    offset_lm_arcmin: NDArray[np.float64]
    unique_time_s: NDArray[np.float64]
    settled: NDArray[np.bool_]


def holography_raster_tracks(
    pointing: ResolvedAntennaPointing,
    *,
    settled_only: bool = True,
) -> tuple[HolographyRasterTrack, ...]:
    """Per-antenna raster samples in arcmin. Missing samples stay absent."""

    rad2arcmin = 180.0 * 60.0 / np.pi
    tracks: list[HolographyRasterTrack] = []
    for index, antenna in enumerate(pointing.antenna_id):
        usable = pointing.valid[:, index]
        if settled_only:
            usable = usable & pointing.settled[:, index]
        if not bool(np.any(usable)):
            continue
        roles = pointing.role[usable, index]
        role = str(roles[0]) if roles.size else AntennaPointingRole.UNKNOWN.value
        tracks.append(
            HolographyRasterTrack(
                antenna_id=int(antenna),
                role=role,
                offset_lm_arcmin=pointing.offset_lm_rad[usable, index] * rad2arcmin,
                unique_time_s=pointing.unique_time_s[usable],
                settled=pointing.settled[usable, index],
            )
        )
    return tuple(tracks)


def write_holography_raster_track_plot(
    pointing: ResolvedAntennaPointing,
    path: Path,
    *,
    settled_only: bool = False,
) -> Path:
    """Write a pointing-track scatter plot. Does not infer the raster from visibilities."""

    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tracks = holography_raster_tracks(pointing, settled_only=settled_only)
    figure, axes = plt.subplots(figsize=(6.0, 6.0))
    for track in tracks:
        axes.plot(
            track.offset_lm_arcmin[:, 0],
            track.offset_lm_arcmin[:, 1],
            marker="o",
            linestyle="-",
            label=f"ant {track.antenna_id} ({track.role})",
        )
    axes.set_xlabel("commanded l / arcmin")
    axes.set_ylabel("commanded m / arcmin")
    axes.set_aspect("equal", adjustable="box")
    axes.axhline(0.0, color="0.7", linewidth=0.6)
    axes.axvline(0.0, color="0.7", linewidth=0.6)
    axes.legend(loc="best", fontsize=8)
    axes.set_title("Holography antenna pointing tracks")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination


@dataclass(frozen=True)
class HolographyMetadataFixture:
    """Tiny POINTING/FIELD/STATE/ANTENNA fixture. No visibilities."""

    source_name: str
    phase_centre_rad: tuple[float, float]
    unique_time_s: NDArray[np.float64]
    antenna_names: tuple[str, ...]
    antenna_position_m: NDArray[np.float64]
    state_labels: tuple[str, ...]
    pointing: AntennaPointingTable
    schema_version: int = HOLOGRAPHY_METADATA_SCHEMA_VERSION
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if int(self.schema_version) != HOLOGRAPHY_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported holography metadata fixture schema")
        times = np.asarray(self.unique_time_s, dtype=np.float64).reshape(-1)
        positions = np.asarray(self.antenna_position_m, dtype=np.float64)
        if times.size == 0:
            raise ValueError("metadata fixture needs unique times")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("antenna_position_m must have shape (antenna, 3)")
        if positions.shape[0] != len(self.antenna_names):
            raise ValueError("antenna names and positions must match")
        object.__setattr__(self, "unique_time_s", times)
        object.__setattr__(self, "antenna_position_m", positions)
        object.__setattr__(self, "antenna_names", tuple(self.antenna_names))
        object.__setattr__(self, "state_labels", tuple(self.state_labels))
        object.__setattr__(self, "notes", tuple(self.notes))


@dataclass(frozen=True)
class HolographyResidualStratum:
    name: str
    n_sample: int
    complex_mse: float
    amplitude_rmse: float
    phase_rmse_rad: float
    rr_ll_mse: float
    rl_lr_mse: float


@dataclass(frozen=True)
class HolographyOnAxisJones:
    """``E_a(0)`` at zero pointing. Fourier is not applied."""

    jones: NDArray[np.complex128]
    valid: NDArray[np.bool_]
    identity_error: float
    status: HolographyGateStatus
    calibration_state: str


@dataclass(frozen=True)
class HolographyClosureResiduals:
    moving_reference_mse: float
    reference_reference_mse: float
    moving_moving_mse: float
    n_moving_reference: int
    n_reference_reference: int
    n_moving_moving: int


@dataclass(frozen=True)
class HolographyResidualReport:
    """Visibility residual, on-axis Jones, and closure-like baseline-class losses.

    Does not freeze a beam. Gates are ``pass``, ``warn``, ``fail``, or ``not_run``.
    """

    overall: HolographyResidualStratum
    holdout: HolographyResidualStratum
    by_correlation: dict[str, HolographyResidualStratum]
    on_axis: HolographyOnAxisJones
    closure: HolographyClosureResiduals
    holdout_reduced_chi2: float
    gates: dict[str, HolographyGateStatus]
    frozen: bool = False
    notes: tuple[str, ...] = ()


def evaluate_holography_on_axis_jones(
    observation: HolographyObservation,
    beam: VoltageBeamModel,
    *,
    backend: str = "numpy",
) -> HolographyOnAxisJones:
    """Evaluate ``E_a(0)`` with zero pointing. Used as a calibration-state gate."""

    unique_times, _inverse = unique_visibility_times(observation.block.time_s)
    n_time = int(unique_times.size)
    n_ant = int(observation.block.antenna_count)
    zeros = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    valid = np.ones((n_time, n_ant), dtype=bool)
    if backend == "jax":
        from sl1mjax.voltage_operator_jax import evaluate_antenna_jones_jax

        jones, jones_valid, _leak = evaluate_antenna_jones_jax(
            observation.block,
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            beam,
            antenna_position_m=observation.antenna_position_m,
            calibration_state=observation.calibration_state,
            antenna_pointing_lm_rad=zeros,
            pointing_valid=valid,
        )
        plane = np.asarray(jones[:, :, 0], dtype=np.complex128)
        plane_ok = np.asarray(jones_valid[:, :, 0], dtype=bool)
    else:
        from sl1mjax.beam_operator import _evaluate_timestep_antennas

        antennas = np.arange(n_ant, dtype=np.int32)
        planes = []
        ok = []
        for time_s in unique_times:
            evaluation, _ = _evaluate_timestep_antennas(
                observation.block,
                np.zeros(1, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
                float(time_s),
                beam=beam,
                antenna_id=antennas,
                antenna_position_m=observation.antenna_position_m,
                calibration_state=observation.calibration_state,
                pointing_offset_lm_rad=np.zeros(n_ant * 2, dtype=np.float64).reshape(n_ant, 2),
            )
            plane_t = np.asarray(evaluation.jones[:, 0], dtype=np.complex128)
            ok_t = np.asarray(evaluation.valid[:, 0], dtype=bool)
            if plane_t.shape[0] == 1 and n_ant > 1:
                plane_t = np.broadcast_to(plane_t, (n_ant, *plane_t.shape[1:])).copy()
                ok_t = np.broadcast_to(ok_t, (n_ant, *ok_t.shape[1:])).copy()
            planes.append(plane_t)
            ok.append(ok_t)
        plane = np.stack(planes, axis=0)
        plane_ok = np.stack(ok, axis=0)
    identity = np.eye(2, dtype=np.complex128)
    error = (
        float(np.max(np.abs(plane - identity)[plane_ok]))
        if bool(np.any(plane_ok))
        else float("nan")
    )
    state = observation.calibration_state
    if state == "casa_parang_true":
        status: HolographyGateStatus = "pass" if np.isfinite(error) and error < 1.0e-6 else "fail"
    else:
        status = "not_run"
    return HolographyOnAxisJones(
        jones=plane,
        valid=plane_ok,
        identity_error=error,
        status=status,
        calibration_state=state,
    )


def report_holography_residuals(
    observation: HolographyObservation,
    beam: VoltageBeamModel,
    target_visibility: ArrayLike,
    *,
    split: HolographyHoldoutSplit | None = None,
    backend: str = "numpy",
    moving_reference_only: bool = False,
    mask_unsettled: bool = True,
    config: BeamOperatorConfig | None = None,
) -> HolographyResidualReport:
    """Score predicted holography visibilities without fitting or freezing a beam."""

    target = np.asarray(target_visibility, dtype=np.complex128)
    if target.shape != observation.block.visibility.shape:
        raise ValueError("target_visibility must match the observation visibility shape")
    predicted = observation.predict(
        beam,
        backend=backend,
        moving_reference_only=moving_reference_only,
        mask_unsettled=mask_unsettled,
        config=config,
    )
    selected = split or holography_holdout_split(observation, HolographyHoldoutAxis.TIME)
    train_samples, holdout_samples = holdout_sample_masks(
        observation,
        selected,
        mask_unsettled=mask_unsettled,
        moving_reference_only=moving_reference_only,
    )
    weight = np.asarray(observation.block.weight, dtype=np.float64)
    scale = _complex_scale(predicted.visibility, target, weight, train_samples)
    model = scale * predicted.visibility
    names = _correlation_names(observation.block.correlations)
    overall_mask = observation.sample_mask(
        mask_unsettled=mask_unsettled,
        moving_reference_only=moving_reference_only,
    )
    overall = _residual_stratum("overall", model, target, weight, overall_mask, names)
    holdout = _residual_stratum("holdout", model, target, weight, holdout_samples, names)
    by_correlation = {
        name: _residual_stratum(
            name,
            model,
            target,
            weight,
            holdout_samples & (np.arange(len(names)) == index)[None, None, :],
            names,
        )
        for index, name in enumerate(names)
    }
    closure = _closure_residuals(observation, model, target, weight, overall_mask)
    on_axis = evaluate_holography_on_axis_jones(observation, beam, backend=backend)
    usable = np.asarray(holdout_samples, dtype=bool) & np.isfinite(weight) & (weight > 0.0)
    n_holdout = int(np.count_nonzero(usable))
    residual = np.where(usable, model - target, 0.0)
    weighted = np.where(usable, weight, 0.0)
    holdout_reduced = float(np.sum(weighted * np.abs(residual) ** 2) / max(n_holdout - 1, 1))
    visibility_status: HolographyGateStatus = "pass" if np.isfinite(holdout.complex_mse) else "fail"
    closure_status: HolographyGateStatus = (
        "pass"
        if (
            (closure.n_reference_reference == 0 or closure.reference_reference_mse < 1.0e-8)
            and (closure.n_moving_moving == 0 or closure.moving_moving_mse < 1.0e-8)
        )
        else "warn"
    )
    return HolographyResidualReport(
        overall=overall,
        holdout=holdout,
        by_correlation=by_correlation,
        on_axis=on_axis,
        closure=closure,
        holdout_reduced_chi2=float(holdout_reduced),
        gates={
            "visibility": visibility_status,
            "on_axis_jones": on_axis.status,
            "closure": closure_status,
        },
        frozen=False,
        notes=(
            "residual report does not freeze a beam",
            f"holdout split {selected.name}",
            "holdout_reduced_chi2 is sum(w|r|^2)/(n-1); it depends on weight normalization",
        ),
    )


def _residual_stratum(
    name: str,
    model: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    correlation_names: tuple[str, ...],
) -> HolographyResidualStratum:
    copolar = _named_correlation_mask(correlation_names, ("RR", "LL"))
    cross = _named_correlation_mask(correlation_names, ("RL", "LR"))
    usable = np.asarray(mask, dtype=bool) & np.isfinite(weight) & (weight > 0.0)
    n_sample = int(np.count_nonzero(usable))
    complex_mse = _weighted_mse(model, target, weight, usable)
    if n_sample == 0:
        amplitude = float("nan")
        phase = float("nan")
    else:
        w = np.where(usable, weight, 0.0)
        amplitude = float(np.sqrt(np.sum(w * (np.abs(model) - np.abs(target)) ** 2) / np.sum(w)))
        product = np.where(usable, model * np.conjugate(target), 0.0)
        phase = float(np.sqrt(np.sum(w * np.angle(product) ** 2) / np.sum(w)))
    return HolographyResidualStratum(
        name=name,
        n_sample=n_sample,
        complex_mse=complex_mse,
        amplitude_rmse=amplitude,
        phase_rmse_rad=phase,
        rr_ll_mse=_weighted_mse(model, target, weight, usable & copolar[None, None, :]),
        rl_lr_mse=_weighted_mse(model, target, weight, usable & cross[None, None, :]),
    )


def _closure_residuals(
    observation: HolographyObservation,
    model: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    sample_mask: np.ndarray,
) -> HolographyClosureResiduals:
    moving_ref = moving_reference_row_mask(observation.block, observation.pointing)
    pair = _baseline_role_pairs(observation)
    ref_ref = pair == "reference_reference"
    moving_moving = pair == "moving_moving"
    return HolographyClosureResiduals(
        moving_reference_mse=_weighted_mse(
            model, target, weight, sample_mask & moving_ref[:, None, None]
        ),
        reference_reference_mse=_weighted_mse(
            model, target, weight, sample_mask & ref_ref[:, None, None]
        ),
        moving_moving_mse=_weighted_mse(
            model, target, weight, sample_mask & moving_moving[:, None, None]
        ),
        n_moving_reference=int(np.count_nonzero(moving_ref)),
        n_reference_reference=int(np.count_nonzero(ref_ref)),
        n_moving_moving=int(np.count_nonzero(moving_moving)),
    )


def _baseline_role_pairs(observation: HolographyObservation) -> NDArray[np.str_]:
    _offsets, inverse, pointing_valid, _settled, _moving = observation.pointing_state()
    roles = np.full((_offsets.shape[0], _offsets.shape[1]), "unknown", dtype="U16")
    for index, antenna in enumerate(observation.pointing.antenna_id):
        roles[:, int(antenna)] = observation.pointing.role[:, index]
    pairs = np.full(observation.block.time_s.shape[0], "unknown", dtype="U32")
    for row, time_index in enumerate(inverse):
        antenna_p = int(observation.block.antenna1[row])
        antenna_q = int(observation.block.antenna2[row])
        if not (pointing_valid[time_index, antenna_p] and pointing_valid[time_index, antenna_q]):
            continue
        pair = {
            str(roles[time_index, antenna_p]),
            str(roles[time_index, antenna_q]),
        }
        if pair == {
            AntennaPointingRole.MOVING.value,
            AntennaPointingRole.REFERENCE.value,
        }:
            pairs[row] = "moving_reference"
        elif pair == {AntennaPointingRole.REFERENCE.value}:
            pairs[row] = "reference_reference"
        elif pair == {AntennaPointingRole.MOVING.value}:
            pairs[row] = "moving_moving"
    return pairs


def load_synthetic_holography_metadata_fixture(
    root: Path | None = None,
) -> HolographyMetadataFixture:
    """Load the committed metadata-only fixture. Visibilities are not present."""

    path = (SYNTHETIC_HOLOGRAPHY_FIXTURE_ROOT if root is None else Path(root)) / (
        "metadata_fixture.json"
    )
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != HOLOGRAPHY_METADATA_SCHEMA_VERSION:
        raise ValueError("unsupported holography metadata fixture schema")
    if "visibility" in payload or "expected_visibility" in payload:
        raise ValueError("metadata fixture must not contain visibilities")
    pointing = load_synthetic_holography_pointing_table(root)
    return HolographyMetadataFixture(
        source_name=str(payload.get("source_name", THOL0001_SOURCE)),
        phase_centre_rad=(
            float(payload["phase_centre_rad"][0]),
            float(payload["phase_centre_rad"][1]),
        ),
        unique_time_s=np.asarray(payload["unique_time_s"], dtype=np.float64),
        antenna_names=tuple(payload["antenna_names"]),
        antenna_position_m=np.asarray(payload["antenna_position_m"], dtype=np.float64),
        state_labels=tuple(payload["state_labels"]),
        pointing=pointing,
        notes=tuple(payload.get("notes", ())),
    )
