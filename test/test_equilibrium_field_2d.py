"""Issue #483: one 2-D equilibrium map, any quantity, with chosen overlays.

``field=`` picks what is drawn -- the stored flux, or a quantity derived on
a private copy from the profiles the slice indexes -- and ``overlay=`` picks
what is drawn over it.  The derivations are the honest ones: pressure is a
flux function and is mapped, the current density is evaluated locally
(issue #316), and the toroidal field is the real ``F(psi)/R``.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import (
    DEFAULT_POLOIDAL_OVERLAYS,
    EQUILIBRIUM_FIELDS,
    EQUILIBRIUM_FIELD_NAMES,
    MACHINE_OVERLAYS,
    POLOIDAL_OVERLAYS,
    build_model,
)
from vaft.plot.display import PSI_STYLES
from vaft.plot.models import Field2D

SLICE = 4


@pytest.fixture(scope="module")
def sample():
    return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def entries(sample):
    return normalize_entries(sample)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _map(entries, **options):
    return build_model("equilibrium_field_2d", entries, time_slice=SLICE, **options)


def _labels(model):
    return {layer.label for layer in model.overlays if layer.label and layer.kind != "text"}


# ---------------------------------------------------------------------------
# the fields
# ---------------------------------------------------------------------------

def test_the_vocabulary_and_its_table():
    assert EQUILIBRIUM_FIELD_NAMES == (
        "psi", "j_tor", "pressure", "b_field_r", "b_field_z", "b_field_tor"
    )
    assert EQUILIBRIUM_FIELDS["psi"].styles == PSI_STYLES
    assert EQUILIBRIUM_FIELDS["j_tor"].updater == "update_equilibrium_profiles_2d_j_tor"
    assert EQUILIBRIUM_FIELDS["pressure"].unit == "Pa"
    assert all(EQUILIBRIUM_FIELDS[name].unit == "T" for name in ("b_field_r", "b_field_z", "b_field_tor"))


@pytest.mark.parametrize("field", EQUILIBRIUM_FIELD_NAMES)
def test_every_field_draws_on_the_stored_grid(entries, sample, field):
    model = _map(entries, field=field)
    r = np.asarray(sample[f"equilibrium.time_slice.{SLICE}.profiles_2d.0.grid.dim1"])
    z = np.asarray(sample[f"equilibrium.time_slice.{SLICE}.profiles_2d.0.grid.dim2"])
    assert isinstance(model, Field2D)
    assert model.values.shape == (z.size, r.size)  # the (Z, R) transpose
    assert np.isfinite(model.values).any()
    assert model.display is not None
    assert model.value_label.endswith(f"[{model.display.unit}]")
    assert EQUILIBRIUM_FIELDS[field].label in model.value_label


def test_the_derived_fields_are_the_honest_ones(entries, sample):
    """Not the legacy splat: pressure is mapped, j_tor is local, B_phi is F/R."""
    private = copy.deepcopy(sample)
    vaft.omas.update_equilibrium_profiles_2d_j_tor(private, time_slice=SLICE)
    vaft.omas.update_equilibrium_profiles_2d_b_field(private, time_slice=SLICE)
    base = f"equilibrium.time_slice.{SLICE}.profiles_2d.0"
    for field, scale in (("j_tor", 1e-6), ("b_field_tor", 1e3)):
        drawn = _map(entries, field=field).values
        stored = np.asarray(private[f"{base}.{field}"]).T * scale
        np.testing.assert_allclose(drawn, stored, equal_nan=True)
    # The confined quantities stop at the boundary; the field does not.
    assert np.isnan(_map(entries, field="pressure").values).any()
    assert np.isfinite(_map(entries, field="b_field_tor").values).all()


def test_the_callers_ods_is_never_written(sample, entries):
    for field in EQUILIBRIUM_FIELD_NAMES:
        _map(entries, field=field)
    stored = sample[f"equilibrium.time_slice.{SLICE}.profiles_2d.0"]
    assert set(stored.keys()) <= {"grid", "grid_type", "psi"}


def test_pressure_is_mapped_on_the_slices_own_psi_grid(entries, sample):
    """The DD gives no 2-D pressure leaf, so it is mapped where it is drawn."""
    drawn = _map(entries, field="pressure").values
    ts = sample[f"equilibrium.time_slice.{SLICE}"]
    psi_2d = np.asarray(ts["profiles_2d.0.psi"]).T
    psi_1d = np.asarray(ts["profiles_1d.psi"])
    pressure = np.asarray(ts["profiles_1d.pressure"])
    axis = float(ts["global_quantities.psi_axis"])
    edge = float(ts["global_quantities.psi_boundary"])
    psi_norm = (psi_2d - axis) / (edge - axis)
    expected = np.where(psi_norm <= 1.0, np.interp(psi_norm, (psi_1d - axis) / (edge - axis), pressure), np.nan)
    np.testing.assert_allclose(drawn, expected, equal_nan=True)
    assert "pressure" not in sample[f"equilibrium.time_slice.{SLICE}.profiles_2d.0"]
    with pytest.raises(ValueError, match="needs profiles_1d.pressure"):
        bare = copy.deepcopy(sample)
        for index in range(len(bare["equilibrium.time_slice"])):
            del bare[f"equilibrium.time_slice.{index}.profiles_1d.pressure"]
        build_model("equilibrium_field_2d", normalize_entries(bare), time_slice=SLICE, field="pressure")


def test_a_field_a_slice_cannot_derive_says_so(sample):
    bare = copy.deepcopy(sample)
    for index in range(len(bare["equilibrium.time_slice"])):
        del bare[f"equilibrium.time_slice.{index}.profiles_1d.f"]
    with pytest.raises(ValueError, match="neither stored nor derivable"):
        build_model("equilibrium_field_2d", normalize_entries(bare), time_slice=SLICE, field="b_field_tor")


def test_units_and_styles_are_per_field(entries):
    assert _map(entries, field="j_tor", units="A/m^2").display.unit == "A/m^2"
    assert _map(entries, field="psi", units="Wb").display.unit == "Wb"
    for style in PSI_STYLES:
        assert _map(entries, field="psi", style=style) is not None
    # A style is the flux map's; another field is one filled map and ignores it,
    # so the interactive style control can travel with a field change.
    assert _map(entries, field="j_tor", style="surfaces").filled is True
    with pytest.raises(ValueError, match="field= one of psi, j_tor"):
        _map(entries, field="q")


def test_the_psi_alias_draws_exactly_the_same_map(entries):
    alias = build_model("equilibrium_field_psi", entries, time_slice=SLICE)
    canonical = _map(entries, field="psi")
    np.testing.assert_allclose(alias.values, canonical.values)
    assert alias.value_label == canonical.value_label
    assert _labels(alias) == _labels(canonical)
    with pytest.raises(ValueError, match="takes field= one of psi; got 'j_tor'"):
        build_model("equilibrium_field_psi", entries, time_slice=SLICE, field="j_tor")


# ---------------------------------------------------------------------------
# the overlays
# ---------------------------------------------------------------------------

def test_the_default_overlays_are_the_machine_plus_the_fields_own(entries):
    assert POLOIDAL_OVERLAYS == ("coils", "passive", "wall", "boundary", "axis")
    assert DEFAULT_POLOIDAL_OVERLAYS == ("coils", "wall")
    assert _labels(_map(entries)) == {"PF coils", "Boundary", "Magnetic axis"}


def test_overlays_are_chosen_and_validated(entries):
    assert _labels(_map(entries, overlay=("wall",), style="filled")) == set()
    assert _labels(_map(entries, overlay=("wall", "axis"), style="filled")) == {"Magnetic axis"}
    assert "Passive structure" in _labels(_map(entries, overlay=("passive",), style="filled"))
    with pytest.raises(ValueError, match="unknown overlay 'lcfs'"):
        _map(entries, overlay="lcfs")


def test_a_style_that_needs_the_region_keeps_the_boundary(entries):
    """The outline is what confines the levels, not decoration."""
    surfaces = _map(entries, overlay=("wall",), style="surfaces")
    assert "Boundary" in _labels(surfaces) and surfaces.region is not None
    filled = _map(entries, overlay=("wall",), style="filled")
    assert "Boundary" not in _labels(filled) and filled.region is None


def test_the_machine_view_takes_the_same_names(entries):
    assert MACHINE_OVERLAYS == ("coils", "passive", "wall", "diagnostics")
    everything = build_model("machine_geometry_poloidal", entries)
    wall_only = build_model("machine_geometry_poloidal", entries, overlay=("wall",))
    assert len(wall_only.layers) < len(everything.layers)
    assert len(build_model("machine_geometry_poloidal", entries, overlay=("coils", "wall")).layers) < len(everything.layers)


def test_the_slice_overview_stays_free_of_the_machine(entries):
    panels = build_model("equilibrium_overview", entries, time_slice=SLICE)
    field = next(m for m in panels.models if isinstance(m, Field2D))
    assert _labels(field) == {"Boundary", "Magnetic axis"}


# ---------------------------------------------------------------------------
# discovery, controls, rendering
# ---------------------------------------------------------------------------

def test_discovery_states_the_fields_and_narrows_them(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_field_2d")
    assert record.fields["default"] == "psi"
    assert record.fields["options"] == EQUILIBRIUM_FIELD_NAMES
    assert record.overlays == POLOIDAL_OVERLAYS
    assert "fields: psi (default) | j_tor" in str(
        vaft.omas.available_plots(sample, query="equilibrium", view="field")
    )
    bare = copy.deepcopy(sample)
    for index in range(len(bare["equilibrium.time_slice"])):
        del bare[f"equilibrium.time_slice.{index}.profiles_1d.f"]
        del bare[f"equilibrium.time_slice.{index}.profiles_1d.pressure"]
    narrowed = next(r for r in vaft.omas.available_plots(bare) if r.name == "equilibrium_field_2d")
    assert narrowed.fields["options"] == ("psi", "j_tor")
    assert "pressure" in narrowed.fields["declared"]


def test_the_control_layer_offers_field_and_overlay(sample):
    from vaft.plot.controls import controls_for

    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_field_2d")
    names = [c.name for c in controls_for(record)]
    assert names == ["time_slice", "units", "field", "overlay", "style"]
    field = next(c for c in controls_for(record) if c.name == "field")
    overlay = next(c for c in controls_for(record) if c.name == "overlay")
    assert field.options == EQUILIBRIUM_FIELD_NAMES and field.default == "psi"
    assert overlay.kind == "multi" and overlay.default == DEFAULT_POLOIDAL_OVERLAYS + ("boundary", "axis")
    result = vaft.omas.plot_equilibrium_field_2d(sample, interactive=True, interaction_backend="none")
    result.state.set("field", "j_tor")
    assert result.axes is not None


def test_the_legacy_six_panel_name_now_resolves_here():
    from vaft.plot._migration import DEPRECATED

    assert DEPRECATED["equilibrium_2d_profiles"] == "equilibrium_field_2d"


def test_both_renderers_draw_every_field(sample):
    for field in ("psi", "j_tor", "b_field_tor"):
        figure, axes = vaft.omas.plot_equilibrium_field_2d(sample, time_slice=SLICE, field=field)
        assert axes.collections or axes.images
        plt.close(figure)
    plotly = vaft.omas.plot_equilibrium_field_2d(sample, time_slice=SLICE, field="pressure", backend="plotly")
    assert len(plotly.data) >= 1


def test_omas_and_imas_agree_for_every_field(sample):
    from test_imas_omas_plot_equivalence import assert_models_equal
    from vaft.imas.access import IDSEntry

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for field in EQUILIBRIUM_FIELD_NAMES:
            expected = build_model("equilibrium_field_2d", [("39915", sample)], time_slice=SLICE, field=field)
            actual = build_model("equilibrium_field_2d", [("39915", entry)], time_slice=SLICE, field=field)
            assert_models_equal(actual, expected)


def test_the_vectorised_field_matches_the_point_interpolator(sample):
    from vaft.process.equilibrium import equilibrium_field_on_grid, make_equilibrium_field_interpolator

    ts = sample[f"equilibrium.time_slice.{SLICE}"]
    r = np.asarray(ts["profiles_2d.0.grid.dim1"])
    z = np.asarray(ts["profiles_2d.0.grid.dim2"])
    psi = np.asarray(ts["profiles_2d.0.psi"])
    psi_1d = np.asarray(ts["profiles_1d.psi"])
    f_1d = np.asarray(ts["profiles_1d.f"])
    grid = equilibrium_field_on_grid(r, z, psi, psi_1d, f_1d, cocos=1)
    point = make_equilibrium_field_interpolator(r, z, psi, psi_1d, f_1d, cocos=1)
    for i, j in ((10, 20), (64, 64), (100, 30)):
        np.testing.assert_allclose([component[i, j] for component in grid], point(r[i], z[j]), rtol=1e-12)


def test_the_unit_and_style_controls_belong_to_the_flux_field(sample):
    """Their values must not travel onto a field with another quantity."""
    from vaft.plot.controls import controls_for
    from vaft.plot.navigation import ControlState

    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_field_2d")
    state = ControlState(controls_for(record))
    assert state.as_options()["units"] == "mWb" and state.as_style() == {}
    assert state.as_options()["style"] == "surfaces"
    state.set("field", "j_tor")
    options = state.as_options()
    assert options["field"] == "j_tor" and "units" not in options and "style" not in options
    state.set("field", "psi")
    assert state.as_options()["units"] == "mWb"
