"""Smoke-test the plot surface against the packaged sample ODS."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import vaft

from _sample_fixtures import sample_ods as _packaged_sample


@pytest.fixture(scope="module")
def sample_ods():
    return _packaged_sample(39915)


def test_pf_coil_time_current_renders_from_an_ods(sample_ods):
    figure, axes = vaft.omas.plot_pf_coil_time_current(sample_ods)
    assert axes.lines
    assert axes.get_ylabel().startswith("Coil Current")
    plt.close(figure)


def test_legacy_entry_point_is_gone_and_names_its_replacement():
    # The 0.5.0 legacy names warned until 0.9.0 deleted them.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match="vaft.omas.plot_pf_coil_time_current"):
            vaft.plot.time_pf_active_current
