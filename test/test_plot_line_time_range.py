"""``time_range=`` on a time-history line plot sets the window it names.

It was accepted by option validation and then ignored by every LineRecipe
plot, so ``plot_flux_loop_time_flux(ods, time_range=(-5e-3, 30e-3))`` drew the
whole record (#254's canonical analysis task asks for exactly that window).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.plot.backend.recipes import RECIPES, _build_line_series


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods(39915)


def test_the_time_range_becomes_the_axis_limits_in_display_units(ods):
    window = (0.30, 0.33)
    model = _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"], time_range=window)
    scale = model.series[0].x[0] / np.asarray(ods["magnetics.flux_loop.0.flux.time"], dtype=float)[0]
    np.testing.assert_allclose(model.x_limits, (window[0] * scale, window[1] * scale))


def test_an_explicit_x_limits_still_wins(ods):
    model = _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"],
                               time_range=(0.30, 0.33), x_limits=(1.0, 2.0))
    assert model.x_limits == (1.0, 2.0)


def test_the_public_plot_draws_the_requested_window(ods):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(ods, time_range=(0.30, 0.33))
    axis = np.ravel(axes)[0] if isinstance(axes, np.ndarray) else axes
    low, high = axis.get_xlim()
    assert low / high == pytest.approx(0.30 / 0.33)
    plt.close(figure)
