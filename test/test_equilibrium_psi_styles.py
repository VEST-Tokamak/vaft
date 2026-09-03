"""Poloidal-flux map styles: the axis, the nested surfaces and the LCFS are identifiable."""

from __future__ import annotations

import contextlib
import copy
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import PSI_STYLES, build_model


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _grid_index(field, r, z):
    return int(np.argmin(np.abs(field.z - z))), int(np.argmin(np.abs(field.r - r)))


def test_flux_surfaces_are_the_default_and_stay_inside_the_plasma(sample):
    assert PSI_STYLES[0] == "surfaces"
    model = build_model("equilibrium_field_psi", normalize_entries(sample), time_slice=4)
    axis = float(sample["equilibrium.time_slice.4.global_quantities.psi_axis"]) * 1e3
    boundary = float(sample["equilibrium.time_slice.4.global_quantities.psi_boundary"]) * 1e3
    assert model.filled is False and model.value_label == "Poloidal Flux [mWb]"
    levels = np.asarray(model.contour_levels)
    assert levels.size == 9 and np.allclose(levels, axis + (boundary - axis) * np.linspace(0.1, 0.9, 9))
    assert model.secondary_levels and min(model.secondary_levels) > boundary
    assert {layer.label for layer in model.overlays if layer.label} == {"Boundary", "Magnetic axis"}
    # The plasma levels are confined to the stored boundary: inside at the
    # axis, outside at the outboard PF coil that carries the same psi values.
    axis_r = float(sample["equilibrium.time_slice.4.global_quantities.magnetic_axis.r"])
    assert model.region[_grid_index(model, axis_r, 0.0)]
    assert not model.region[_grid_index(model, 0.95, 0.5)]


def test_the_normalized_style_reads_zero_at_the_axis_and_one_at_the_boundary(sample):
    model = build_model("equilibrium_field_psi", normalize_entries(sample), time_slice=4, style="normalized")
    assert model.filled and np.allclose(model.contour_levels, np.linspace(0, 1, 11))
    inside = np.asarray(model.values)[model.region]
    assert inside.min() == pytest.approx(0.0, abs=0.02) and inside.max() == pytest.approx(1.0, abs=0.05)


def test_the_filled_style_is_the_raw_field_in_the_display_unit(sample):
    model = build_model("equilibrium_field_psi", normalize_entries(sample), time_slice=4, style="filled")
    raw = np.asarray(sample["equilibrium.time_slice.4.profiles_2d.0.psi"]).T
    assert model.filled and model.contour_levels is None and model.region is None
    assert np.allclose(model.values, raw * 1e3)
    assert any(layer.label == "Magnetic axis" for layer in model.overlays)


def test_every_style_renders_with_the_same_colorbar_unit(sample):
    for style in PSI_STYLES:
        figure, axes = vaft.omas.plot_equilibrium_field_psi(sample, time_slice=4, style=style)
        labels = [a.get_ylabel() for a in figure.axes if a is not axes]
        assert labels and ("[mWb]" in labels[-1] or r"\psi_N" in labels[-1]), style
        plt.close(figure)
    with pytest.raises(ValueError, match="style must be one of"):
        vaft.omas.plot_equilibrium_field_psi(sample, style="contour")


def test_without_stored_psi_axis_the_profiles_supply_the_span(sample):
    ods = copy.deepcopy(sample)
    del ods["equilibrium.time_slice.4.global_quantities.psi_axis"]
    model = build_model("equilibrium_field_psi", normalize_entries(ods), time_slice=4)
    psi = np.asarray(sample["equilibrium.time_slice.4.profiles_1d.psi"]) * 1e3
    assert np.asarray(model.contour_levels)[0] == pytest.approx(psi[0] + 0.1 * (psi[-1] - psi[0]))
    del ods["equilibrium.time_slice.4.profiles_1d.psi"]
    # No span at all: the default degrades to the filled map, a named style raises.
    degraded = build_model("equilibrium_field_psi", normalize_entries(ods), time_slice=4)
    assert degraded.filled and degraded.region is None
    with pytest.raises(ValueError, match="only style='filled'"):
        build_model("equilibrium_field_psi", normalize_entries(ods), time_slice=4, style="surfaces")
    assert build_model("equilibrium_field_psi", normalize_entries(ods), time_slice=4, style="filled").filled


def test_the_overview_draws_the_flux_map_in_the_requested_style(sample):
    model = build_model("equilibrium_overview", normalize_entries(sample), time_slice=4, style="normalized")
    assert model.models[0].filled and model.models[0].value_label.startswith("Normalized")
    assert build_model("equilibrium_overview", normalize_entries(sample), time_slice=4).models[0].filled is False
