from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_3c391_radio_catalog.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_3c391_radio_catalog", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_nvss_tsv_preserves_position_flux_shape_and_provenance() -> None:
    module = _module()
    text = """# VizieR response
NVSS\tRAJ2000\tDEJ2000\tS1.4\te_S1.4\tMajAxis\tMinAxis\tPA
 \t\"h:m:s\"\t\"d:m:s\"\tmJy\tmJy\tarcsec\tarcsec\tdeg
--------------\t-----------\t-----------\t--------\t-------\t-----\t-----\t-----
184932-003802 \t18 49 32.79\t-00 38 02.0\t   653.3\t   22.9\t 16.9\t 16.9\t 75.0
"""

    sources = module.parse_nvss_tsv(text)

    assert len(sources) == 1
    source = sources[0]
    assert source.name == "NVSS_184932-003802"
    assert source.ra_deg == pytest.approx(282.386625)
    assert source.dec_deg == pytest.approx(-0.6338888889)
    assert source.integrated_flux_jy == pytest.approx(0.6533)
    assert source.major_axis_arcsec == 16.9
    assert source.catalog == "NVSS VIII/65"


def test_query_url_pins_catalog_columns_radius_and_flux_threshold() -> None:
    module = _module()

    url = module._query_url(
        ra_deg=282.3,
        dec_deg=-0.9,
        radius_deg=1.0,
        minimum_flux_mjy=50.0,
    )

    assert "VIII%2F65%2Fnvss" in url
    assert "S1.4=%3E%3D50" in url
    assert "-c.r=1" in url
    assert "RAJ2000" in url


def test_parse_vlass_tsv_keeps_only_reliable_main_components() -> None:
    module = _module()
    header = (
        "CompName\tRAJ2000\tDEJ2000\tFtot\tFpeak\tSCode\tDCMaj\tDCMin\tDCPA\t"
        "NVSSdist\tDupFlag\tQualFlag\tMainSample"
    )
    accepted = (
        "VLASS1QLCIR J184932.56-003805.2\t282.38568307\t-0.63478545\t573.143\t"
        "64.744\tM\t13.1541\t11.5766\t169.9680\t4.62\t0\t0\t1"
    )
    rejected = (
        "VLASS1QLCIR J184933.40-003753.5\t282.38917988\t-0.63154858\t1.951\t"
        "1.028\tC\t3.2665\t2.2962\t58.1317\t12.53\t0\t2\t0"
    )
    text = "\n".join(("# VizieR response", header, accepted, rejected))

    sources = module.parse_vlass_tsv(text)

    assert len(sources) == 1
    assert sources[0].name == "VLASS_J184932.56-003805.2"
    assert sources[0].integrated_flux_jy == pytest.approx(0.573143)
    assert sources[0].peak_flux_jy == pytest.approx(0.064744)
    assert sources[0].reference_frequency_hz == 3.0e9
