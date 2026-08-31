"""Smoke-test the plot surface against the packaged sample ODS."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import vaft


@pytest.fixture(scope="module")
def sample_ods():
    return vaft.omas.load(vaft.data.sample(39915, "omas"))


def test_pf_coil_time_current_renders_from_an_ods(sample_ods):
    figure, axes = vaft.omas.plot_pf_coil_time_current(sample_ods)
    assert axes.lines
    assert axes.get_ylabel().startswith("Coil Current")
    plt.close(figure)


def test_legacy_entry_point_still_works_and_warns(sample_ods):
    vaft.plot.__dict__.pop("time_pf_active_current", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = vaft.plot.time_pf_active_current

    assert any(item.category is DeprecationWarning for item in caught)
    assert "pf_coil_time_current" in str(caught[0].message)
    legacy(sample_ods)
    plt.close("all")
