"""The atlas base-table builder (workflow/operational_space_atlas/build_efit_base.py) on a packaged product."""

import importlib.util
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1] / "workflow/operational_space_atlas"


def _build_efit_base():
    spec = importlib.util.spec_from_file_location("build_efit_base", HERE / "build_efit_base.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def product():
    from vaft.omas.sample import sample_ods
    try:
        return sample_ods(41524)
    except FileNotFoundError:
        pytest.skip("the 41524 pipeline product is repository-only")


def test_b0_is_paired_with_the_slice_on_the_equilibrium_time_base(product):
    # `_row` picked b0 by slice index with a clamp, so a b0 stored on a coarser or
    # offset base silently gave another time's field with base_status "valid"
    # (cold review 0.8.0 delta-absorb-16 confinement F4; match by time, not index).
    beb = _build_efit_base()
    times = np.asarray(product["equilibrium.time"], dtype=float)
    b0 = np.array(np.atleast_1d(product["equilibrium.vacuum_toroidal_field.b0"]), dtype=float)
    assert b0.size == times.size > 4
    reconstructed = [i for i in range(times.size) if f"equilibrium.time_slice.{i}.boundary.outline.r" in product]
    k = reconstructed[len(reconstructed) // 3]
    ramp = b0 * np.linspace(1.0, 1.3, b0.size)   # a field that differs from slice to slice
    try:
        product["equilibrium.vacuum_toroidal_field.b0"] = ramp
        row = beb._row(product, float(times[k]))
        assert row["base_status"] == "valid" and row["b0_t"] == pytest.approx(abs(ramp[k]))
        reference = row
        # a constant field stored once is the one legal shortcut
        product["equilibrium.vacuum_toroidal_field.b0"] = ramp[k:k + 1]
        once = beb._row(product, float(times[k]))
        assert once["base_status"] == "valid" and once["b0_t"] == pytest.approx(reference["b0_t"])
        assert once["normalized_current"] == pytest.approx(reference["normalized_current"])
        # any other length cannot be paired with the slice: refused, not clamped
        for bad in (ramp[::4], ramp[1:]):
            product["equilibrium.vacuum_toroidal_field.b0"] = bad
            refused = beb._row(product, float(times[k]))
            assert refused["base_status"] == "failed", refused
            assert f"{bad.size} entries for {times.size}" in refused["base_reason"]
            assert "b0_t" not in refused and "normalized_current" not in refused
    finally:
        product["equilibrium.vacuum_toroidal_field.b0"] = b0


def test_the_volume_beta_columns_share_the_normalization_of_the_troyon_plane(product):
    # #1691 stage 2B: <beta_N> must be the ratio of the plotted axes, beta_B / I_N, or the companion
    # Troyon plane would read a different normalization than its own axes
    beb = _build_efit_base()
    times = np.asarray(product["equilibrium.time"], dtype=float)
    reconstructed = [i for i in range(times.size) if f"equilibrium.time_slice.{i}.boundary.outline.r" in product]
    row = beb._row(product, float(times[reconstructed[len(reconstructed) // 2]]))
    assert row["base_status"] == "valid"
    assert np.isfinite(row["volume_beta"]) and row["volume_beta"] > 0
    assert row["normalized_volume_beta"] == pytest.approx(row["volume_beta"] / row["normalized_current"])
    assert row["aspect_ratio"] == pytest.approx(row["major_radius_geo_m"] / row["minor_radius_m"])
    # at VEST's aspect ratio the total field energy is not B_0^2: the two betas are different numbers
    assert abs(row["volume_beta"] / row["toroidal_beta"] - 1.0) > 0.05
    assert {"volume_beta", "normalized_volume_beta", "aspect_ratio"} <= set(beb.UNITS)
    # y = x on beta_n_normalizations compares like with like only while both betas share one normalization:
    # the DD routine's beta_N and this <beta_N> take a and I_p by different paths, which agree to about 1 %
    assert row["normalized_beta"] / row["normalized_volume_beta"] == pytest.approx(
        row["toroidal_beta"] / row["volume_beta"], rel=0.02)
