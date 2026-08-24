"""Smoke-test the plot surface against the packaged sample ODS."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from omas import ODS

import vaft
from vaft.data.resources import data_path


@pytest.fixture(scope="module")
def sample_ods():
    return ODS().load(str(data_path("omas/39915.json")), consistency_check=False)


def test_pf_active_time_current_renders_from_an_ods(sample_ods):
    figure, axes = vaft.omas.plot_pf_active_time_current(sample_ods)
    assert axes.lines
    assert axes.get_ylabel().startswith("Coil Current")
    plt.close(figure)


def test_legacy_entry_point_still_works_and_warns(sample_ods):
    vaft.plot.__dict__.pop("time_pf_active_current", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = vaft.plot.time_pf_active_current

    assert any(item.category is DeprecationWarning for item in caught)
    assert "pf_active_time_current" in str(caught[0].message)
    legacy(sample_ods)
    plt.close("all")
