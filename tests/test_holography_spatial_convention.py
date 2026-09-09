from __future__ import annotations

import inspect

import numpy as np
import pytest
from test_holography_highres_cassbeam import _FREQ_HZ, _write_test_catalog

from sl1mjax.holography_cassbeam_correction import (
    ARCMIN_TO_RAD,
    cassbeam_hand_peaks_rad,
    catalog_diagonal_feed_lookup,
    gaussian_diagonal_lookup,
)
from sl1mjax.holography_physical_squint import (
    IDENTITY_PHYSICAL,
    PhysicalBeamState,
    apply_physical_feed_frame,
    native_separation_rad,
    physical_query_coordinates,
)
from sl1mjax.holography_spatial_convention import (
    FROZEN_WIDTH,
    IDENTITY_TRANSFORM,
    SpatialTransform,
    apply_spatial_transform,
    apply_transformed_feed_frame,
    identity_queries_match_physical,
    refuse_convention_ladder,
    refuse_magnitude_fit,
    select_spatial_transform,
    spatial_transform_catalog,
    transformed_query_coordinates,
    transformed_separation_rad,
    vector_alignment,
)


def test_catalog_has_sixteen_discrete_transforms() -> None:
    catalog = spatial_transform_catalog()
    assert len(catalog) == 16
    assert len({item.name for item in catalog}) == 16
    assert any(item.is_identity() for item in catalog)
    with pytest.raises(RuntimeError, match="128-member"):
        refuse_convention_ladder()
    with pytest.raises(RuntimeError, match="magnitude"):
        refuse_magnitude_fit()


def test_identity_query_matches_physical_identity() -> None:
    offset = np.array([[0.0, 0.0], [1.2e-4, -8.0e-5], [3.0e-4, 2.0e-4]])
    identity_queries_match_physical(offset)
    q_r, q_l = transformed_query_coordinates(offset, IDENTITY_TRANSFORM, width=1.0)
    p_r, p_l = physical_query_coordinates(offset, IDENTITY_PHYSICAL)
    np.testing.assert_allclose(q_r, p_r)
    np.testing.assert_allclose(q_l, p_l)


def test_identity_width_matches_physical_width() -> None:
    offset = np.array([[2.0e-4, -1.0e-4], [0.0, 3.0e-4]])
    q_r, q_l = transformed_query_coordinates(offset, IDENTITY_TRANSFORM, width=FROZEN_WIDTH)
    p_r, p_l = physical_query_coordinates(offset, PhysicalBeamState(width=FROZEN_WIDTH))
    np.testing.assert_allclose(q_r, p_r)
    np.testing.assert_allclose(q_l, p_l)


def test_inverse_round_trip() -> None:
    offset = np.array([[1.0e-4, -2.0e-4], [3.0e-4, 4.0e-4]])
    for transform in spatial_transform_catalog():
        mapped = apply_spatial_transform(offset, transform)
        back = apply_spatial_transform(mapped, transform.inverse())
        np.testing.assert_allclose(back, offset)


def test_minus_m_and_rl_swap_rotate_the_native_vector() -> None:
    native = native_separation_rad() / ARCMIN_TO_RAD
    flipped = transformed_separation_rad(SpatialTransform(m_sign=-1)) / ARCMIN_TO_RAD
    np.testing.assert_allclose(flipped, np.array([native[0], -native[1]]))
    swapped = transformed_separation_rad(SpatialTransform(swap_rl=True)) / ARCMIN_TO_RAD
    np.testing.assert_allclose(swapped, -native)
    measured = np.array([0.252, -0.675])
    align_flip = vector_alignment(flipped, measured)
    align_native = vector_alignment(native, measured)
    assert align_flip["aligns"]
    assert not align_native["aligns"]
    assert align_flip["angle_deg"] < align_native["angle_deg"]


def test_identity_feed_matches_physical_on_catalog(tmp_path) -> None:
    catalog = _write_test_catalog(tmp_path)
    lookup = catalog_diagonal_feed_lookup(catalog, _FREQ_HZ)
    offset = np.array([[0.0, 0.0], [4.0e-4, -2.0e-4], [-3.0e-4, 1.0e-4]])
    physical = apply_physical_feed_frame(offset, lookup, IDENTITY_PHYSICAL)
    spatial = apply_transformed_feed_frame(offset, lookup, IDENTITY_TRANSFORM, width=1.0)
    np.testing.assert_allclose(spatial, physical, rtol=0.0, atol=1.0e-12)
    wide_p = apply_physical_feed_frame(offset, lookup, PhysicalBeamState(width=FROZEN_WIDTH))
    wide_s = apply_transformed_feed_frame(offset, lookup, IDENTITY_TRANSFORM, width=FROZEN_WIDTH)
    np.testing.assert_allclose(wide_s, wide_p, rtol=0.0, atol=1.0e-12)


def test_transform_moves_the_whole_gaussian_beam() -> None:
    peak_r, peak_l = cassbeam_hand_peaks_rad()
    lookup = gaussian_diagonal_lookup(peak_r, peak_l, 8.0 * ARCMIN_TO_RAD)
    transform = SpatialTransform(m_sign=-1)
    mapped_r, _mapped_l = (
        apply_spatial_transform(peak_r, transform)[0],
        apply_spatial_transform(peak_l, transform)[0],
    )
    native = apply_transformed_feed_frame(
        peak_r.reshape(1, 2), lookup, IDENTITY_TRANSFORM, width=1.0
    )
    moved = apply_transformed_feed_frame(mapped_r.reshape(1, 2), lookup, transform, width=1.0)
    np.testing.assert_allclose(moved[0, 0, 0], native[0, 0, 0], rtol=0.0, atol=1.0e-12)


def test_selection_requires_alignment_and_prefers_rr_minus_ll() -> None:
    native = native_separation_rad() / ARCMIN_TO_RAD
    flipped = np.array([native[0], -native[1]])
    records = [
        {
            "name": IDENTITY_TRANSFORM.name,
            "is_identity": True,
            "delta_arcmin": [float(native[0]), float(native[1])],
            "train_rr_minus_ll": 1.3,
            "train_combined": 0.05,
            "train_rr": 0.05,
            "train_ll": 0.05,
        },
        {
            "name": "l+1_m-1",
            "is_identity": False,
            "delta_arcmin": [float(flipped[0]), float(flipped[1])],
            "train_rr_minus_ll": 0.9,
            "train_combined": 0.048,
            "train_rr": 0.049,
            "train_ll": 0.047,
        },
        {
            "name": "anti",
            "is_identity": False,
            "delta_arcmin": [-0.252, 0.675],
            "train_rr_minus_ll": 0.8,
            "train_combined": 0.04,
            "train_rr": 0.04,
            "train_ll": 0.04,
        },
    ]
    selected = select_spatial_transform(records, measured_delta_arcmin=(0.252, -0.675))
    assert selected["locked"] is True
    assert selected["selected"]["name"] == "l+1_m-1"


def test_kernels_have_no_row_loop() -> None:
    assert "for row" not in inspect.getsource(apply_spatial_transform)
    assert "for row" not in inspect.getsource(transformed_query_coordinates)
    assert "for row" not in inspect.getsource(apply_transformed_feed_frame)


def test_runner_stays_on_the_discrete_sixteen() -> None:
    from pathlib import Path

    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("scripts", "run_thol0001_spatial_convention.py")
        .read_text()
    )
    assert "convention_ladder" not in source
    assert "refuse_convention_ladder()" not in source
    assert "empirical_plus_width" in source
    assert "holoraster_spatial_convention_v1" in source
    assert "holoraster_physical_squint_width_v1" in source
