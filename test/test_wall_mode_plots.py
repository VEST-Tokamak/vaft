"""Plot contracts for the wall eigenmode views (vaft #473)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas as vomas
from vaft.plot.backend.recipes import RECIPES
from vaft.plot.models import GeometryLayers, Panels


@pytest.fixture(scope="module")
def packaged():
    from vaft.omas.process_wrapper import compute_wall_mode_basis_ods
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    return ods, compute_wall_mode_basis_ods(ods)


def test_the_mode_shape_colours_only_the_chosen_segment(packaged):
    ods, basis = packaged
    model = RECIPES["passive_structure_geometry_wall_mode"].builder(ods, basis=basis, segment="W1", mode=0)
    assert isinstance(model, GeometryLayers)
    coloured = [layer for layer in model.layers if isinstance(layer.style.get("color"), tuple)]
    grey = [layer for layer in model.layers if layer.style.get("color") == "0.75"]
    assert len(coloured) == 240 and len(grey) == 950 - 240
    assert "W1 mode 0" in model.title and "ms" in model.title
    assert sum(1 for layer in model.layers if layer.label) == 2


def test_a_mode_outside_the_segment_is_refused(packaged):
    ods, basis = packaged
    with pytest.raises(ValueError, match="does not exist"):
        RECIPES["passive_structure_geometry_wall_mode"].builder(ods, basis=basis, segment="W9_U", mode=99)


def test_the_spectrum_has_one_series_per_segment_plus_the_whole_wall(packaged):
    ods, basis = packaged
    model = RECIPES["passive_structure_spectrum_wall_time"].builder(ods, basis=basis, max_modes=5)
    assert isinstance(model, Panels) and len(model.models) == 1
    panel = model.models[0]
    labels = [series.label for series in panel.series]
    assert labels[:-1] == [seg.id for seg in basis.segments] and labels[-1] == "whole wall"
    assert panel.log_y and all(series.y.size == 5 for series in panel.series)
    assert all(np.all(np.diff(series.y) < 0) for series in panel.series)


def test_the_adapters_render_figures(packaged):
    ods, basis = packaged
    figure, axes = vomas.plot_passive_structure_geometry_wall_mode(ods, basis=basis, segment="W7_L", mode=1)
    assert "W7_L mode 1" in axes.get_title()
    plt.close(figure)
    figure, axes = vomas.plot_passive_structure_spectrum_wall_time(ods, basis=basis, max_modes=3)
    assert np.asarray(axes).ravel()[0].get_yscale() == "log"
    plt.close(figure)
