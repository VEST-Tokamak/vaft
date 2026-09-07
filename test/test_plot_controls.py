"""Issue #480: a validated option vocabulary and discovery-driven controls."""

from __future__ import annotations

import pytest

import vaft
from vaft.plot.backend.options import (
    EXTRACTION_OPTIONS,
    OPTION_SCHEMA,
    STYLE_OPTIONS,
    OptionSpec,
    choices_for,
    split_options,
    validate_options,
)
from vaft.plot.controls import CONTROL_KINDS, ControlSpec, controls_for


@pytest.fixture(scope="module")
def catalog():
    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    return {record.name: record for record in vaft.omas.available_plots(ods)}


# ---------------------------------------------------------------------------
# option schema
# ---------------------------------------------------------------------------

def test_the_schema_names_every_extraction_option_once():
    assert set(OPTION_SCHEMA) == EXTRACTION_OPTIONS
    assert "selection" in OPTION_SCHEMA and "units" in OPTION_SCHEMA
    with pytest.raises(ValueError, match="kind must be one of"):
        OptionSpec("x", "weird")


def test_the_style_set_is_read_off_the_renderers():
    assert STYLE_OPTIONS == frozenset({
        "cmap", "colorbar", "colorbar_ax", "figsize", "fps", "grid", "interval_ms",
        "legend", "save_path", "uncertainty", "validity",
    })
    assert not (STYLE_OPTIONS & EXTRACTION_OPTIONS)


def test_vocabularies_resolve_to_the_live_constants():
    from vaft.plot.backend.recipes import LAYOUTS, ORIENTATIONS, SYNTHETIC_MODES
    from vaft.plot.display import PSI_STYLES

    assert choices_for(OPTION_SCHEMA["layout"]) == LAYOUTS
    assert choices_for(OPTION_SCHEMA["orientation"]) == ORIENTATIONS
    assert choices_for(OPTION_SCHEMA["synthetic"]) == SYNTHETIC_MODES
    assert choices_for(OPTION_SCHEMA["style"]) == PSI_STYLES
    assert choices_for(OPTION_SCHEMA["title"]) is None


def test_unknown_and_out_of_vocabulary_options_are_refused_by_name():
    validate_options("plasma_current_time", {"selection": "active", "layout": "subplots", "legend": False, "_panel_member": True})
    with pytest.raises(ValueError, match="does not take an option named 'selecton'.*extraction options.*style options"):
        validate_options("plasma_current_time", {"selecton": "active"})
    with pytest.raises(ValueError, match="layout must be one of overlay, subplots, grouped"):
        validate_options("plasma_current_time", {"layout": "stacked"})
    validate_options("x", {"selection": [0, 1]})  # a list is the recipe's business


def test_split_routes_extraction_and_style_apart():
    extraction, style = split_options({"selection": "valid", "legend": False, "_panel_member": True, "_convention": "Wb"})
    assert extraction == {"selection": "valid", "_panel_member": True, "_convention": "Wb"}
    assert style == {"legend": False}


# ---------------------------------------------------------------------------
# control specs
# ---------------------------------------------------------------------------

def test_control_specs_validate_their_shape():
    assert CONTROL_KINDS == ("toggle", "choice", "multi", "range", "text")
    with pytest.raises(ValueError, match="kind must be one of"):
        ControlSpec("x", "dial", "X")
    with pytest.raises(ValueError, match="offers no choices"):
        ControlSpec("x", "choice", "X")
    with pytest.raises(ValueError, match="not one of its choices"):
        ControlSpec("x", "choice", "X", "c", ("a", "b"))
    with pytest.raises(ValueError, match="needs \\(min, max, step\\)"):
        ControlSpec("x", "range", "X", 0, (0, 5))
    with pytest.raises(ValueError, match="one label per option"):
        ControlSpec("x", "choice", "X", "a", ("a", "b"), ("only",))
    spec = ControlSpec("layout", "choice", "Layout", "overlay", ("overlay", "subplots"))
    assert spec.validate("subplots") == "subplots"
    with pytest.raises(ValueError, match="layout must be one of"):
        spec.validate("stacked")
    multi = ControlSpec("channels", "multi", "Channels", (), (0, 1, 2))
    assert multi.validate([2, 0]) == (2, 0)
    with pytest.raises(ValueError, match="not among"):
        multi.validate([7])
    position = ControlSpec("time_slice", "range", "Slice", 2, (0, 8, 1))
    assert position.validate(3.0) == 3
    with pytest.raises(ValueError, match="must lie in"):
        position.validate(9)


def test_a_channel_line_offers_selection_layout_unit_sign_and_validity(catalog):
    controls = controls_for(catalog["flux_loop_time_flux"])
    names = [c.name for c in controls]
    # No synthetic control: this input has no reconstruction overlay to show.
    assert names == ["selection", "channels", "layout", "yunit", "x", "orientation", "validity"]
    by_name = {c.name: c for c in controls}
    assert by_name["selection"].options == ("inboard_mid", "outboard_mid", "inboard", "outboard", "active", "valid", "all")
    assert by_name["selection"].default == "active"
    assert len(by_name["channels"].options) == 11 and by_name["channels"].labels[0].endswith("Flux Loop - #3")
    assert by_name["layout"].options == ("overlay", "subplots", "grouped") and by_name["layout"].default == "overlay"
    assert by_name["yunit"].options == ("Wb", "mWb") and by_name["yunit"].default == "mWb"
    assert by_name["orientation"].default == "canonical"
    assert by_name["validity"].options == ("show", "mask", "ignore") and by_name["validity"].group == "style"
    assert [c.name for c in controls_for(catalog["flux_loop_time_flux"], include_style=False)][-1] == "orientation"
    assert controls_for(catalog["flux_loop_time_flux"], include_backend=True)[-1].name == "backend"


def test_a_profile_offers_its_slice_and_its_sign(catalog):
    controls = controls_for(catalog["equilibrium_profile_q"])
    assert [c.name for c in controls] == ["time_slice", "coordinate", "orientation"]
    slices = controls[0]
    assert slices.kind == "choice" and slices.group == "slice"
    assert slices.options == tuple(catalog["equilibrium_profile_q"].slices["usable"])
    assert slices.labels[0].startswith("0: 316.0 ms")
    coordinate = controls[1]
    assert coordinate.options == ("rho_tor_norm", "psi_norm", "sqrt_phi_norm", "r_major", "r_minor")
    assert coordinate.default == "rho_tor_norm" and coordinate.labels[0] == "Normalized Toroidal Flux rho_N"
    assert controls[2].default == "intuitive"


def test_the_psi_map_offers_units_and_style_but_the_vacuum_map_no_style(catalog):
    names = [c.name for c in controls_for(catalog["equilibrium_field_psi"])]
    assert names[-2:] == ["units", "style"]
    units = next(c for c in controls_for(catalog["equilibrium_field_psi"]) if c.name == "units")
    assert units.options == ("Wb", "mWb", "Wb/rad", "mWb/rad") and units.default == "mWb"
    assert "style" not in [c.name for c in controls_for(catalog["equilibrium_field_psi_vacuum"])]


def test_a_spectrogram_offers_the_analysis_it_states_and_nothing_else(catalog):
    controls = controls_for(catalog["mirnov_spectrogram"])
    assert [c.name for c in controls] == ["method"]
    assert controls[0].options == ("stft", "hann_fft", "cwt") and controls[0].default == "stft"


def test_the_synthetic_control_needs_the_overlay_to_be_available(catalog):
    record = catalog["flux_loop_time_flux"]
    from dataclasses import replace

    unavailable = replace(record, synthetic={"overlay": "equilibrium", "available": False})
    assert "synthetic" not in [c.name for c in controls_for(unavailable)]
    available = replace(record, synthetic={"overlay": "equilibrium", "available": True})
    synthetic = next(c for c in controls_for(available) if c.name == "synthetic")
    assert synthetic.options == ("none", "equilibrium", "both") and synthetic.default == "none"
    # after selection, channels, layout, yunit and the abscissa (#481)
    assert [c.name for c in controls_for(available)].index("synthetic") == 5
