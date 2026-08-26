from __future__ import annotations

import numpy as np
import pytest

from sl1mjax.beam import VLAPrimaryBeam
from sl1mjax.catalog import (
    RadioCatalogSource,
    read_radio_catalog,
    select_catalog_guard_atoms,
    write_radio_catalog,
)
from sl1mjax.data.canonical import VisibilityBlock
from sl1mjax.polarization import Correlation, ReceptorBasis


def _block(phase_centre_rad: tuple[float, float] = (4.2, -0.3)) -> VisibilityBlock:
    shape = (2, 2, 1)
    return VisibilityBlock(
        uvw_m=np.zeros((2, 3)),
        frequency_hz=np.asarray([1.0e9, 2.0e9]),
        visibility=np.zeros(shape, dtype=np.complex128),
        weight=np.ones(shape),
        flag=np.zeros(shape, dtype=bool),
        time_s=np.arange(2, dtype=np.float64),
        antenna1=np.zeros(2, dtype=np.int32),
        antenna2=np.ones(2, dtype=np.int32),
        correlations=(Correlation.I,),
        receptor_basis=ReceptorBasis.STOKES,
        phase_centre_rad=phase_centre_rad,
    )


def _source(
    name: str,
    ra_offset_rad: float,
    *,
    flux_jy: float = 1.0,
    major_axis_arcsec: float | None = 10.0,
) -> RadioCatalogSource:
    return RadioCatalogSource(
        name=name,
        ra_deg=np.rad2deg(4.2 + ra_offset_rad),
        dec_deg=np.rad2deg(-0.3),
        reference_frequency_hz=1.0e9,
        integrated_flux_jy=flux_jy,
        major_axis_arcsec=major_axis_arcsec,
        catalog="test",
        reference_url="https://example.invalid/catalog",
    )


def test_catalog_guard_selection_excludes_central_extended_and_invisible_sources() -> None:
    block = _block()
    sources = (
        _source("central", 1e-4),
        _source("bright_outer", 0.006, flux_jy=2.0),
        _source("extended_outer", 0.006, flux_jy=3.0, major_axis_arcsec=90.0),
        _source("faint_outer", 0.006, flux_jy=1e-4),
    )

    selected = select_catalog_guard_atoms(
        sources,
        (block,),
        block.phase_centre_rad,
        primary_beam=VLAPrimaryBeam(kind="gaussian"),
        central_half_width_rad=(0.001, 0.001),
        minimum_apparent_flux_jy=1e-3,
        default_spectral_index=-0.7,
        maximum_major_axis_arcsec=45.0,
    )

    assert [atom.source.name for atom in selected] == ["bright_outer"]
    assert selected[0].initial_flux_jy == pytest.approx(2.0 * 1.5**-0.7)
    assert 0 < selected[0].maximum_beam_power < 1
    assert 0 < selected[0].maximum_apparent_flux_jy < 2.0


def test_read_radio_catalog_preserves_provenance_and_optional_values(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    path.write_text(
        "name,ra_deg,dec_deg,reference_frequency_hz,integrated_flux_jy,"
        "peak_flux_jy,major_axis_arcsec,minor_axis_arcsec,position_angle_deg,"
        "spectral_index,catalog,epoch,reference_url\n"
        "source_a,282.1,-0.5,1.4e9,0.7,,18.0,12.0,45.0,,NVSS,J2000,"
        "https://example.invalid/nvss\n",
        encoding="utf-8",
    )

    sources = read_radio_catalog(path)

    assert len(sources) == 1
    assert sources[0].name == "source_a"
    assert sources[0].peak_flux_jy is None
    assert sources[0].major_axis_arcsec == 18.0
    assert sources[0].catalog == "NVSS"


def test_radio_catalog_csv_round_trip(tmp_path) -> None:
    path = tmp_path / "catalog.csv"
    expected = (
        RadioCatalogSource(
            name="source_a",
            ra_deg=282.1,
            dec_deg=-0.5,
            reference_frequency_hz=3.0e9,
            integrated_flux_jy=0.7,
            peak_flux_jy=0.4,
            major_axis_arcsec=12.0,
            minor_axis_arcsec=8.0,
            position_angle_deg=30.0,
            spectral_index=-0.65,
            catalog="VLASS",
            epoch="2018",
            reference_url="https://example.invalid/vlass",
        ),
    )

    write_radio_catalog(path, expected)

    assert read_radio_catalog(path) == expected


def test_radio_catalog_requires_unique_names() -> None:
    source = _source("duplicate", 0.01)

    with pytest.raises(ValueError, match="unique"):
        select_catalog_guard_atoms(
            (source, source),
            (_block(),),
            (4.2, -0.3),
            primary_beam=VLAPrimaryBeam(kind="gaussian"),
            central_half_width_rad=(0.001, 0.001),
            minimum_apparent_flux_jy=0.0,
        )
