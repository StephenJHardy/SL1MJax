"""External high-resolution CASSBEAM artifact. Not the production factory.

Loads the numerically converged 513×513 g1024/p32 planes by exact native
frequency. It does not open the committed 33×33 tables, does not use
nearest-frequency substitution, and does not freeze full Jones.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sl1mjax.cassbeam_beam import (
    _cassbeam_aperture_and_crop_start,
    _cassbeam_crop_origin_indices,
    _parse_params,
)
from sl1mjax.polarization import invert_jones
from sl1mjax.rime import SPEED_OF_LIGHT_M_S

DEFAULT_HIGHRES_ROOT = Path(
    "/media/stephen/astro/vla/beam_models/cassbeam_cband_full_jones_g1024_p32_20260906"
)
HIGHRES_MODEL_ID = "cassbeam_vla_cband_full_jones_g1024_p32_spw4_v1"
EXPECTED_RASTER = (513, 513, 2, 2)
NATIVE_COLUMNS = ("Re_RR", "Im_RR", "Re_LR", "Im_LR", "Re_RL", "Im_RL", "Re_LL", "Im_LL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geom_radius_m(path: Path) -> float:
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            last = float(stripped.split()[0])
    if last is None or last <= 0.0:
        raise ValueError(f"{path} must end with a positive primary radius")
    return last


def _pixel_scale_rad(params: dict[str, str], geom_path: Path) -> float:
    aperture_n, _crop = _cassbeam_aperture_and_crop_start(params)
    half = aperture_n // 2
    spacing = _geom_radius_m(geom_path) / half
    wavelength = SPEED_OF_LIGHT_M_S / (float(params["freq"]) * 1.0e9)
    pixels = float(params["pixelsperbeam"])
    if pixels <= 0.0 or spacing <= 0.0:
        raise ValueError("CASSBEAM grid parameters must be positive")
    return wavelength / (pixels * aperture_n * spacing)


def bilinear_jones(
    jones: ArrayLike,
    l_axis: ArrayLike,
    m_axis: ArrayLike,
    l_rad: ArrayLike,
    m_rad: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
    """Bilinear lookup on a CASSBEAM (m, l) raster. Vectorized."""

    table = np.asarray(jones, dtype=np.complex128)
    l_ax = np.asarray(l_axis, dtype=np.float64)
    m_ax = np.asarray(m_axis, dtype=np.float64)
    l_q = np.asarray(l_rad, dtype=np.float64).reshape(-1)
    m_q = np.asarray(m_rad, dtype=np.float64).reshape(-1)
    i = np.interp(l_q, l_ax, np.arange(l_ax.size), left=np.nan, right=np.nan)
    j = np.interp(m_q, m_ax, np.arange(m_ax.size), left=np.nan, right=np.nan)
    ok = np.isfinite(i) & np.isfinite(j)
    i0 = np.clip(np.floor(np.where(ok, i, 0.0)).astype(np.int64), 0, l_ax.size - 2)
    j0 = np.clip(np.floor(np.where(ok, j, 0.0)).astype(np.int64), 0, m_ax.size - 2)
    di = np.where(ok, i - i0, 0.0)
    dj = np.where(ok, j - j0, 0.0)
    g00 = table[j0, i0]
    g10 = table[j0, i0 + 1]
    g01 = table[j0 + 1, i0]
    g11 = table[j0 + 1, i0 + 1]
    plane = (1.0 - dj)[:, None, None] * (
        (1.0 - di)[:, None, None] * g00 + di[:, None, None] * g10
    ) + dj[:, None, None] * ((1.0 - di)[:, None, None] * g01 + di[:, None, None] * g11)
    return np.asarray(plane, dtype=np.complex128), np.asarray(ok, dtype=bool)


def diagonal_projection(jones: ArrayLike) -> NDArray[np.complex128]:
    """Zero off-diagonals of an already-normalized Jones matrix."""

    plane = np.array(jones, dtype=np.complex128, copy=True)
    plane[..., 0, 1] = 0.0
    plane[..., 1, 0] = 0.0
    return plane


@dataclass(frozen=True)
class HighresCassbeamPlane:
    """One exact native-frequency 513×513 plane after E(0)^{-1} E(s)."""

    frequency_hz: float
    frequency_mhz: int
    jones_native: NDArray[np.complex128]
    jones_norm: NDArray[np.complex128]
    l_rad: NDArray[np.float64]
    m_rad: NDArray[np.float64]
    pixel_scale_rad: float
    l_origin_index: int
    m_origin_index: int
    data_sha256: str
    params_sha256: str

    def origin_native(self) -> NDArray[np.complex128]:
        return np.asarray(
            self.jones_native[self.m_origin_index, self.l_origin_index],
            dtype=np.complex128,
        )

    def lookup_native(
        self,
        l_rad: ArrayLike,
        m_rad: ArrayLike,
    ) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
        return bilinear_jones(self.jones_native, self.l_rad, self.m_rad, l_rad, m_rad)

    def lookup(
        self,
        l_rad: ArrayLike,
        m_rad: ArrayLike,
        *,
        off_diagonal: bool,
    ) -> tuple[NDArray[np.complex128], NDArray[np.bool_]]:
        plane, ok = bilinear_jones(self.jones_norm, self.l_rad, self.m_rad, l_rad, m_rad)
        if not off_diagonal:
            plane = diagonal_projection(plane)
        return plane, ok

    def node_jones(self, i_l: int, i_m: int, *, off_diagonal: bool) -> NDArray[np.complex128]:
        plane = np.array(self.jones_norm[int(i_m), int(i_l)], dtype=np.complex128, copy=True)
        if not off_diagonal:
            plane = diagonal_projection(plane)
        return plane


class HighresCassbeamCatalog:
    """Lazy external-artifact catalog. Exact frequency only."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        expected_model_id: str = HIGHRES_MODEL_ID,
        expected_raster: tuple[int, int, int, int] = EXPECTED_RASTER,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_HIGHRES_ROOT
        self.expected_model_id = str(expected_model_id)
        self.expected_raster = tuple(int(item) for item in expected_raster)
        if "cassbeam_cband" in self.root.parts and "g1024" not in str(self.root):
            raise ValueError("refusing the committed 33×33 CASSBEAM artifact")
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(self.manifest.get("model_id")) != self.expected_model_id:
            raise ValueError(f"unexpected high-res model_id {self.manifest.get('model_id')!r}")
        raster = self.manifest.get("raster") or {}
        if tuple(raster.get("shape") or ()) != self.expected_raster:
            raise ValueError(
                f"high-res raster must be {self.expected_raster[0]}×{self.expected_raster[1]}×2×2"
            )
        if str(raster.get("science_normalization")) != "inv(E(0)) @ E(s)":
            raise ValueError("high-res artifact must record inv(E(0)) @ E(s)")
        self.files_sha256 = {
            str(key): str(value) for key, value in (self.manifest.get("files_sha256") or {}).items()
        }
        if not self.files_sha256:
            raise ValueError("high-res manifest is missing files_sha256")
        self._planes_by_mhz = {
            int(plane["frequency_mhz"]): plane for plane in self.manifest["planes"]
        }
        self.geom_path = self.root / "reference" / "vla_geom"
        if not self.geom_path.is_file():
            raise FileNotFoundError(self.geom_path)
        self._verify_file(self.geom_path.relative_to(self.root).as_posix())
        self.frozen = False
        self.production_factory = False

    def frequency_mhz_list(self) -> tuple[int, ...]:
        return tuple(sorted(self._planes_by_mhz))

    def require_exact_mhz(self, frequency_hz: float) -> int:
        mhz = int(np.round(float(frequency_hz) / 1.0e6))
        if mhz not in self._planes_by_mhz:
            raise ValueError(
                f"no exact high-res CASSBEAM plane at {mhz} MHz; "
                "nearest-frequency substitution is refused"
            )
        recorded = float(self._planes_by_mhz[mhz]["frequency_mhz"]) * 1.0e6
        if abs(float(frequency_hz) - recorded) > 5.0e3:
            raise ValueError(f"frequency {frequency_hz} Hz is not the exact native plane {mhz} MHz")
        return mhz

    def plane(self, frequency_hz: float) -> HighresCassbeamPlane:
        mhz = self.require_exact_mhz(frequency_hz)
        return _load_highres_plane(
            str(self.root.resolve()),
            int(mhz),
            self.expected_model_id,
            self.expected_raster,
        )

    def _verify_file(self, relative: str) -> str:
        expected = self.files_sha256.get(relative)
        if expected is None:
            raise ValueError(f"no checksum for {relative}")
        actual = _sha256(self.root / relative)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {relative}")
        return actual


@lru_cache(maxsize=72)
def _load_highres_plane(
    root_s: str,
    frequency_mhz: int,
    expected_model_id: str,
    expected_raster: tuple[int, int, int, int],
) -> HighresCassbeamPlane:
    catalog = HighresCassbeamCatalog(
        Path(root_s),
        expected_model_id=expected_model_id,
        expected_raster=expected_raster,
    )
    record = catalog._planes_by_mhz[int(frequency_mhz)]
    if tuple(record.get("native_columns") or ()) != NATIVE_COLUMNS:
        raise ValueError("unexpected native column order")
    if tuple(record.get("shape") or ()) != catalog.expected_raster:
        raise ValueError(
            f"plane shape must be {catalog.expected_raster[0]}×{catalog.expected_raster[1]}×2×2"
        )
    data_rel = str(record["data"])
    params_rel = str(record["params"])
    data_sha = catalog._verify_file(data_rel)
    params_sha = catalog._verify_file(params_rel)
    params = _parse_params(catalog.root / params_rel)
    raw = np.loadtxt(catalog.root / data_rel, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 8:
        raise ValueError(f"{data_rel} must have eight columns")
    size = int(catalog.expected_raster[0])
    if raw.shape[0] != size * int(catalog.expected_raster[1]):
        raise ValueError(f"{data_rel} is not a {size}×{size} raster")
    native = np.empty((size, size, 2, 2), dtype=np.complex128)
    native[..., 0, 0] = (raw[:, 0] + 1j * raw[:, 1]).reshape(size, size)
    native[..., 1, 0] = (raw[:, 2] + 1j * raw[:, 3]).reshape(size, size)
    native[..., 0, 1] = (raw[:, 4] + 1j * raw[:, 5]).reshape(size, size)
    native[..., 1, 1] = (raw[:, 6] + 1j * raw[:, 7]).reshape(size, size)
    if not bool(np.all(np.isfinite(native))):
        raise ValueError(f"{data_rel} contains non-finite Jones samples")
    aperture_n, crop_start = _cassbeam_aperture_and_crop_start(params)
    l_origin, m_origin = _cassbeam_crop_origin_indices(aperture_n, crop_start)
    if size != aperture_n - 2 * crop_start + 1:
        raise ValueError("Jones raster size does not match the CASSBEAM crop")
    scale = _pixel_scale_rad(params, catalog.geom_path)
    center = native[m_origin, l_origin]
    inverse = invert_jones(center)
    if not bool(np.all(np.isfinite(inverse))):
        raise ValueError("E(0) is not invertible")
    norm = np.einsum("ij,mnjk->mnik", inverse, native)
    return HighresCassbeamPlane(
        frequency_hz=float(params["freq"]) * 1.0e9,
        frequency_mhz=int(frequency_mhz),
        jones_native=native,
        jones_norm=np.asarray(norm, dtype=np.complex128),
        l_rad=np.asarray((np.arange(size) - l_origin) * scale, dtype=np.float64),
        m_rad=np.asarray((np.arange(size) - m_origin) * scale, dtype=np.float64),
        pixel_scale_rad=float(scale),
        l_origin_index=int(l_origin),
        m_origin_index=int(m_origin),
        data_sha256=data_sha,
        params_sha256=params_sha,
    )


EVLA_C_HIGHRES_MODEL_ID = "cassbeam_evla_c_g1024_p32_spw4_dev"


def write_development_highres_manifest(
    root: Path,
    *,
    model_id: str,
    name_prefix: str,
    gridsize: int = 1024,
    pixelsperbeam: int = 32,
) -> dict[str, object]:
    """Checksum whatever exact planes exist. Not a production freeze."""

    root = Path(root)
    files: dict[str, str] = {}
    planes: list[dict[str, object]] = []
    pattern = f"{name_prefix}-*-g{gridsize}-p{pixelsperbeam}.jones.dat"
    for data_path in sorted(root.glob(f"**/{pattern}")):
        if not data_path.name.endswith(".jones.dat"):
            raise ValueError(f"unexpected Jones name {data_path.name}")
        params_path = data_path.with_name(data_path.name[: -len(".jones.dat")] + ".params")
        if not params_path.is_file():
            raise FileNotFoundError(params_path)
        frequency_mhz = int(data_path.name.split("-")[2])
        group = data_path.parent.name
        for path in (data_path, params_path):
            files[path.relative_to(root).as_posix()] = _sha256(path)
        planes.append(
            {
                "frequency_mhz": frequency_mhz,
                "group": group,
                "shape": [513, 513, 2, 2],
                "native_columns": list(NATIVE_COLUMNS),
                "data": data_path.relative_to(root).as_posix(),
                "params": params_path.relative_to(root).as_posix(),
            }
        )
    if not planes:
        raise FileNotFoundError(f"no {pattern} planes under {root}")
    for relative in ("reference/base.in", "reference/vla_geom"):
        files[relative] = _sha256(root / relative)
    manifest = {
        "schema_version": 1,
        "model_id": str(model_id),
        "artifact_kind": "electromagnetic_voltage_jones",
        "development_only": True,
        "production_accepted": False,
        "full_jones_frozen": False,
        "raster": {
            "shape": [513, 513, 2, 2],
            "science_normalization": "inv(E(0)) @ E(s)",
        },
        "planes": planes,
        "files_sha256": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
