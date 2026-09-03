"""Machine geometry drawn as the machine's parts, not one object per element.

PF coils are read from every element in either IMAS form (rectangle or
outline); the passive structure is one structure; the top view places every
diagnostic channel that stores a toroidal position.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import (
    _element_outlines,
    _topview_diagnostic_layers,
    build_model,
    entry_supports,
)
from vaft.plot.models import GeometryLayer, GeometryLayers
from vaft.plot.renderers.geometry import render_geometry_layers


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def test_pf_coils_stored_as_rectangles_are_drawn_from_every_element(sample):
    assert entry_supports(sample, "pf_coil_geometry_poloidal")
    model = build_model("pf_coil_geometry_poloidal", normalize_entries(sample))
    polygons = [layer for layer in model.layers if layer.kind == "polygon"]
    elements = sum(len(sample[f"pf_active.coil.{i}.element"]) for i in range(len(sample["pf_active.coil"])))
    assert len(polygons) == elements and elements > 300
    names = [f"PF{i + 1}" for i in range(10)]
    assert [layer.label for layer in polygons if layer.label] == names
    # Each coil is annotated at least once, and an up/down pair twice.
    annotations = [layer for layer in model.layers if layer.kind == "text"]
    assert sorted({layer.label for layer in annotations}) == sorted(names)
    pf3 = [layer for layer in annotations if layer.label == "PF3"]
    assert len(pf3) == 2 and np.sign(pf3[0].z[0]) != np.sign(pf3[1].z[0])


def test_the_passive_structure_is_one_legend_entry(sample):
    model = build_model("passive_structure_geometry_poloidal", normalize_entries(sample))
    assert len(model.layers) == len(sample["pf_passive.loop"])
    assert [layer.label for layer in model.layers if layer.label] == ["Passive structure"]


def test_the_machine_view_names_sets_not_loops(sample):
    figure, axes = vaft.omas.plot_machine_geometry_poloidal(sample, show=False)
    legend = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend == ["Passive structure", "PF coils", "Flux Loops", "B-field Probes"]
    notes = {text.get_text() for text in axes.texts}
    assert {f"PF{i + 1}" for i in range(10)} <= notes  # coil names, beside the sensor indices
    plt.close(figure)


def test_element_outlines_accept_either_imas_form():
    ods = omas.ODS(consistency_check=False)
    ods["pf_active.coil.0.element.0.geometry.geometry_type"] = 1
    ods["pf_active.coil.0.element.0.geometry.outline.r"] = np.array([1.0, 1.1, 1.1, 1.0])
    ods["pf_active.coil.0.element.0.geometry.outline.z"] = np.array([0.0, 0.0, 0.1, 0.1])
    ods["pf_active.coil.0.element.1.geometry.rectangle.r"] = 1.5
    ods["pf_active.coil.0.element.1.geometry.rectangle.z"] = 0.5
    ods["pf_active.coil.0.element.1.geometry.rectangle.width"] = 0.2
    ods["pf_active.coil.0.element.1.geometry.rectangle.height"] = 0.1
    ods["pf_active.coil.0.element.2.geometry.geometry_type"] = 4  # arcs: not drawn
    outlines = _element_outlines(ods, "pf_active.coil.0")
    assert len(outlines) == 2
    assert np.allclose(outlines[1][0], [1.4, 1.6, 1.6, 1.4]) and np.allclose(outlines[1][1], [0.45, 0.45, 0.55, 0.55])
    with pytest.raises(ValueError, match="no coil element geometry"):
        build_model("pf_coil_geometry_poloidal", normalize_entries(omas.ODS(consistency_check=False)))


def test_a_text_layer_is_an_annotation_not_a_legend_entry():
    with pytest.raises(ValueError, match="one coordinate"):
        GeometryLayer(r=[0.0, 1.0], z=[0.0, 1.0], kind="text", label="x")
    model = GeometryLayers(layers=(
        GeometryLayer(r=[0.0, 1.0], z=[0.0, 0.0], kind="polyline", label="line"),
        GeometryLayer(r=[0.5], z=[0.0], kind="text", label="note"),
    ))
    figure, axes = render_geometry_layers(model, show=False)
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["line"]
    assert [t.get_text() for t in axes.texts] == ["note"]
    plt.close(figure)


def test_the_top_view_places_diagnostics_that_store_a_toroidal_position(sample):
    layers = {layer.label: layer for layer in _topview_diagnostic_layers(sample) if layer.label}
    assert set(layers) == {"B-pol probes", "B-tor probes"}
    probe = sample["magnetics.b_field_pol_probe.0.position"]
    r, phi = float(probe["r"]), float(probe["phi"])
    points = layers["B-pol probes"]
    assert np.isclose(points.r[0], r * np.cos(phi)) and np.isclose(points.z[0], r * np.sin(phi))
    figure, axes = vaft.omas.plot_machine_geometry_topview(sample, show=False)
    legend = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend[-2:] == ["B-pol probes", "B-tor probes"]
    plt.close(figure)


def test_the_top_view_does_not_invent_a_toroidal_angle():
    ods = omas.ODS(consistency_check=False)
    ods["soft_x_rays.channel.0.line_of_sight.first_point.r"] = 1.0
    ods["soft_x_rays.channel.0.line_of_sight.first_point.phi"] = 0.5
    ods["soft_x_rays.channel.0.line_of_sight.second_point.r"] = 0.2
    ods["soft_x_rays.channel.0.line_of_sight.second_point.phi"] = 0.5
    ods["soft_x_rays.channel.1.line_of_sight.first_point.r"] = 1.0   # no phi: not placed
    ods["soft_x_rays.channel.1.line_of_sight.second_point.r"] = 0.2
    ods["magnetics.flux_loop.0.position.0.r"] = 0.8
    ods["magnetics.flux_loop.0.position.0.phi"] = 0.0
    ods["thomson_scattering.channel.0.position.r"] = 0.6  # no phi: not placed
    layers = _topview_diagnostic_layers(ods)
    assert [layer.label for layer in layers if layer.label] == ["Flux loops", "Soft X-ray LOS"]
    segments = [layer for layer in layers if layer.kind == "polyline" and layer.r.size == 2]
    assert len(segments) == 1
    assert np.allclose(segments[0].r, [np.cos(0.5), 0.2 * np.cos(0.5)])
    ring = [layer for layer in layers if layer.label == "Flux loops"][0]
    assert np.isclose(np.hypot(ring.r, ring.z).max(), 0.8)
    model = build_model("machine_geometry_topview", normalize_entries(ods))
    assert any(layer.label == "Soft X-ray LOS" for layer in model.layers)


def test_the_standalone_coil_view_names_coils_beside_them_and_drops_the_legend(sample):
    figure, axes = vaft.omas.plot_pf_coil_geometry_poloidal(sample, show=False)
    assert axes.get_legend() is None and sorted({t.get_text() for t in axes.texts}) == sorted(f"PF{i + 1}" for i in range(10))
    plt.close(figure)
    figure, axes = vaft.omas.plot_machine_geometry_poloidal(sample, show=False)
    assert axes.get_legend() is not None
    assert not any("(" in t.get_text() for t in axes.get_legend().get_texts())
    low, high = axes.get_ylim()
    assert low < -1.4 and high > 1.4  # padded beyond the outermost coil at |Z| = 1.25
    plt.close(figure)


def test_magnetic_sensors_are_annotated_with_their_index(sample):
    model = build_model("magnetics_geometry_poloidal", normalize_entries(sample))
    notes = [layer for layer in model.layers if layer.kind == "text"]
    probes = sample["magnetics.b_field_pol_probe"]
    loops = sample["magnetics.flux_loop"]
    with_position = sum(1 for i in range(len(probes)) if "position.r" in probes[i]) + \
        sum(1 for i in range(len(loops)) if "position.0.r" in loops[i])
    assert len(notes) == with_position
    first = next(layer for layer in notes if layer.label == "0")
    assert np.isclose(first.r[0], float(loops[0]["position.0.r"]))


def test_index_annotations_follow_the_points_through_a_position_gap():
    ods = omas.ODS(consistency_check=False)
    for index, (r, z) in enumerate(((1.0, 0.1), (1.1, None), (1.2, 0.3))):
        ods[f"magnetics.flux_loop.{index}.position.0.r"] = r
        if z is not None:
            ods[f"magnetics.flux_loop.{index}.position.0.z"] = z
    model = build_model("magnetics_geometry_poloidal", normalize_entries(ods))
    points = next(layer for layer in model.layers if layer.kind == "points")
    notes = {layer.label: (float(layer.r[0]), float(layer.z[0])) for layer in model.layers if layer.kind == "text"}
    assert points.r.size == 2 and notes == {"0": (1.0, 0.1), "2": (1.2, 0.3)}
