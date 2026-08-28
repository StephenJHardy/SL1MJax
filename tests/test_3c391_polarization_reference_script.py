from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "create_3c391_polarization_reference.py"
REPO = Path(__file__).parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("create_3c391_polarization_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_paths_use_local_data_not_bagofwinds() -> None:
    module = _module()

    measurement_set = module.default_measurement_set()
    reference = module.default_kbg_reference()
    output = module.default_pol_reference()

    assert "BagOfWinds" not in str(measurement_set)
    assert "BagOfWinds" not in str(reference)
    assert "BagOfWinds" not in str(output)
    assert measurement_set.name == "3c391_ctm_mosaic_10s_spw0.ms"
    assert reference == (REPO / "data/3c391/reference-v2").resolve()
    assert output == (REPO / "data/3c391/reference-pol").resolve()


def test_gaintable_plan_adds_each_new_term_to_later_stages() -> None:
    module = _module()
    reference = Path("/tmp/kbg")
    output = Path("/tmp/pol")

    plan = module.gaintable_plan(reference, output)

    assert plan["leakage_gain"] == [
        "/tmp/kbg/3c391.antpos",
        "/tmp/kbg/3c391.K0",
        "/tmp/kbg/3c391.B0",
    ]
    assert plan["kcross"][-1].endswith("3c391.B0")
    assert "fluxscale1" in plan["kcross"][1]
    assert plan["dterms"][-1].endswith("3c391.Kcross")
    assert any(path.endswith("3c391.G84") for path in plan["dterms"])
    assert plan["angle"][-1].endswith("3c391.Df0")
    assert plan["apply_science"][-1].endswith("3c391.Xf0")
    assert len(plan["apply_science"]) == 7
    assert plan["apply_leakage"][1].endswith("3c391.G84")


def test_local_ms_has_four_circular_products_and_3c84() -> None:
    module = _module()
    measurement_set = module.default_measurement_set()
    if not measurement_set.is_dir():
        pytest.skip("local 3C391 Measurement Set is not present")

    import numpy as np
    from casacore import tables

    with tables.table(str(measurement_set / "POLARIZATION"), readonly=True, ack=False) as table:
        corr_type = np.asarray(table.getcol("CORR_TYPE"))
    with tables.table(str(measurement_set / "FIELD"), readonly=True, ack=False) as table:
        names = [str(name) for name in table.getcol("NAME")]

    assert corr_type.tolist() == [[5, 6, 7, 8]]
    assert names[0] == "J1331+3030"
    assert names[9] == "J0319+4130"
    assert (REPO / "data/3c391/reference-v2/3c391.K0").is_dir()


def test_three_c286_casaguide_qu_matches_nrao_recipe() -> None:
    module = _module()
    polarised, stokes_q, stokes_u = module.three_c286_casaguide_qu(7.74664)
    assert polarised == pytest.approx(0.112 * 7.74664)
    assert stokes_q == pytest.approx(polarised * math.cos(math.radians(66.0)))
    assert stokes_u == pytest.approx(polarised * math.sin(math.radians(66.0)))
    assert "constant IQUV" in module.THREE_C286_POLARISED_SPECTRAL_MODEL
    assert "Perley-Butler" in module.THREE_C286_POLARISED_SPECTRAL_MODEL


def test_stokes_i_from_setjy_reads_nested_fluxd() -> None:
    module = _module()
    result = {"0": {"0": {"fluxd": [7.57, 0.0, 0.0, 0.0]}}}
    assert module.stokes_i_from_setjy(result) == pytest.approx(7.57)


def test_flag_version_names_read_casa_list_payload() -> None:
    module = _module()
    listing = {
        0: {"name": "sl1mjax_calibration_input"},
        1: {"name": "sl1mjax_post_polcal"},
    }
    assert module.flag_version_names(listing) == {
        "sl1mjax_calibration_input",
        "sl1mjax_post_polcal",
    }
