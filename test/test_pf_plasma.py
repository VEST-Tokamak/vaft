"""The plasma current as pf_plasma elements: writer, reader, and the plot."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.omas.pf_plasma import (
    GEOMETRY_TYPE_RECTANGLE,
    plasma_current_total,
    plasma_elements,
    set_plasma_elements,
)
from vaft.plot.backend.recipes import RECIPES
from vaft.plot.models import GeometryLayers


def _four_filaments():
    ods = ODS()
    time = np.array([0.30, 0.31, 0.32])
    r = np.array([0.40, 0.40, 0.40, 0.55])
    z = np.array([0.00, 0.15, -0.15, 0.00])
    currents = np.outer([0.5, 0.15, 0.15, 0.2], [0.0, 8.0e4, 6.0e4])
    set_plasma_elements(ods, r, z, width=0.01, height=0.01, currents=currents, time=time,
                        code_name="VFIT", code_parameters="<parameters><source>filament</source></parameters>",
                        comment="four filaments")
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.1, 0.8, 0.8, 0.1])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-1.0, -1.0, 1.0, 1.0])
    return ods, r, z, currents, time


def test_the_writer_lays_elements_out_as_imas_rectangles():
    ods, r, z, currents, time = _four_filaments()
    assert ods["pf_plasma.ids_properties.homogeneous_time"] == 1
    np.testing.assert_array_equal(ods["pf_plasma.time"], time)
    assert len(ods["pf_plasma.element"]) == 4
    assert ods["pf_plasma.element.3.geometry.geometry_type"] == GEOMETRY_TYPE_RECTANGLE
    assert ods["pf_plasma.element.3.geometry.rectangle.r"] == pytest.approx(0.55)
    assert ods["pf_plasma.element.3.area"] == pytest.approx(1.0e-4)
    np.testing.assert_allclose(ods["pf_plasma.element.0.current"], currents[0])
    assert ods["pf_plasma.code.name"] == "VFIT"
    # the ODS passes OMAS' own consistency check for the IDS layout
    ods.satisfy_imas_requirements()


def test_the_writer_replaces_rather_than_appends_and_validates_shapes():
    ods, r, z, currents, time = _four_filaments()
    set_plasma_elements(ods, r[:2], z[:2], width=0.02, height=0.02, currents=currents[:2], time=time)
    assert len(ods["pf_plasma.element"]) == 2
    with pytest.raises(ValueError, match="shape"):
        set_plasma_elements(ods, r, z, width=0.01, height=0.01, currents=currents[:, :2], time=time)
    with pytest.raises(ValueError, match="positive"):
        set_plasma_elements(ods, r, z, width=0.0, height=0.01, currents=currents, time=time)


def test_the_reader_returns_the_instant_of_largest_current_by_default():
    ods, r, z, currents, time = _four_filaments()
    axis, total = plasma_current_total(ods)
    np.testing.assert_allclose(total, currents.sum(axis=0))
    elements = plasma_elements(ods)
    assert elements["time"] == pytest.approx(0.31) and elements["index"] == 1
    np.testing.assert_allclose(elements["current"], currents[:, 1])
    np.testing.assert_allclose(elements["r"], r)
    assert elements["total"] == pytest.approx(8.0e4)
    at_end = plasma_elements(ods, time=0.3199)
    np.testing.assert_allclose(at_end["current"], currents[:, 2])


def test_the_reader_refuses_an_empty_ids():
    with pytest.raises(ValueError, match="no elements"):
        plasma_elements(ODS())


def test_the_plot_colours_elements_by_current_and_draws_the_limiter():
    ods, r, z, currents, time = _four_filaments()
    model = RECIPES["pf_plasma_geometry_poloidal"].builder(ods)
    assert isinstance(model, GeometryLayers)
    element_layers = [layer for layer in model.layers if layer.label != "limiter"]
    assert len(element_layers) == 4 and model.layers[-1].label == "limiter"
    colours = [layer.style["color"] for layer in element_layers]
    # the centre filament carries the most current: the most saturated colour
    assert colours[0] != colours[1] and all(isinstance(c, tuple) for c in colours)
    assert "Ip = 80.0 kA" in model.title and "0.3100 s" in model.title
    at_end = RECIPES["pf_plasma_geometry_poloidal"].builder(ods, time=0.32)
    assert "Ip = 60.0 kA" in at_end.title


def test_the_adapter_renders_a_figure():
    ods, *_ = _four_filaments()
    figure, axes = vomas.plot_pf_plasma_geometry_poloidal(ods)
    assert "plasma elements" in axes.get_title()
    plt.close(figure)
