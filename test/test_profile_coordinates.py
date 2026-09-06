"""Issue #479: the radial coordinates of an equilibrium 1-D profile.

Five names, validated at every adapter; ``r_major`` mirrors the profile
through the magnetic axis and marks the axis and the limiter; ``r_minor``
is the midplane half-width; ``sqrt_phi_norm`` is the toroidal coordinate
sourced from a stored ``phi``.  What a slice cannot supply degrades with a
relabelled axis (issue #276), never with a wrong label.
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
from vaft.plot.backend.recipes import build_model
from vaft.plot.display import COORDINATE_LABELS, PROFILE_COORDINATES
from vaft.plot.models import Profile1D, ReferenceLine

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


def _q(entries, **options):
    return build_model("equilibrium_profile_q", entries, time_slice=SLICE, **options)


def test_the_vocabulary_and_its_labels():
    assert PROFILE_COORDINATES == ("rho_tor_norm", "psi_norm", "sqrt_phi_norm", "r_major", "r_minor")
    assert set(COORDINATE_LABELS) == set(PROFILE_COORDINATES) | {"index"}


def test_an_unknown_coordinate_is_refused_naming_the_options(sample, entries):
    with pytest.raises(ValueError, match="one of rho_tor_norm, psi_norm, sqrt_phi_norm, r_major, r_minor; got 'rho'"):
        _q(entries, coordinate="rho")
    with pytest.raises(ValueError, match="coordinate must be one of rho_tor_norm"):
        vaft.omas.plot_equilibrium_profile_q(sample, coordinate="rho")
    # A diagnostic profile offers its own coordinates only.
    from vaft.plot.backend.recipes import RECIPES, ProfileRecipe, _coordinate_options

    thomson = next(r for n, r in RECIPES.items() if isinstance(r, ProfileRecipe) and r.coordinate_paths)
    assert _coordinate_options(thomson) == tuple(thomson.coordinate_paths)


@pytest.mark.parametrize("coordinate", PROFILE_COORDINATES)
def test_every_coordinate_resolves_on_the_packaged_shot(entries, coordinate):
    model = _q(entries, coordinate=coordinate)
    x = np.asarray(model.series[0].x)
    stored = 129
    assert model.coordinate_label == COORDINATE_LABELS[coordinate]
    assert np.all(np.isfinite(x))
    assert x.size == (2 * stored if coordinate == "r_major" else stored)
    assert np.all(np.diff(x) >= -1e-9)  # monotone (the radial ones within a grid step)
    if coordinate in ("rho_tor_norm", "psi_norm", "sqrt_phi_norm"):
        assert x[0] == pytest.approx(0.0, abs=1e-9) and x[-1] == pytest.approx(1.0, abs=1e-9)
        assert model.x_limits == (0.0, 1.0) and model.reference_lines == ()


def test_r_major_mirrors_the_profile_through_the_axis_and_marks_axis_and_limiter(sample, entries):
    model = _q(entries, coordinate="r_major")
    stored = _q(entries, coordinate="psi_norm").series[0]
    trace = model.series[0]
    n = np.asarray(stored.y).size
    np.testing.assert_allclose(trace.y[:n], stored.y[::-1])
    np.testing.assert_allclose(trace.y[n:], stored.y)
    axis_r = float(sample[f"equilibrium.time_slice.{SLICE}.global_quantities.magnetic_axis.r"])
    grid = np.diff(sample[f"equilibrium.time_slice.{SLICE}.profiles_2d.0.grid.dim1"])[0]
    assert abs(trace.x[n - 1] - axis_r) <= grid and abs(trace.x[n] - axis_r) <= grid
    labels = {line.label for line in model.reference_lines}
    assert labels == {"Magnetic axis", "Limiter", ""}
    by_label = {line.label: line.x for line in model.reference_lines}
    assert by_label["Magnetic axis"] == pytest.approx(axis_r)
    assert by_label["Limiter"] == pytest.approx(0.104, abs=1e-3)
    assert max(line.x for line in model.reference_lines) == pytest.approx(0.76, abs=1e-3)
    assert model.x_limits == pytest.approx((0.104 - 0.02, 0.76 + 0.02), abs=1e-3)
    assert all(isinstance(line, ReferenceLine) for line in model.reference_lines)


def test_without_wall_geometry_only_the_axis_is_marked(sample):
    bare = copy.deepcopy(sample)
    del bare["wall"]
    model = build_model("equilibrium_profile_q", normalize_entries(bare), time_slice=SLICE, coordinate="r_major")
    assert [line.label for line in model.reference_lines] == ["Magnetic axis"]
    assert model.x_limits is None


def test_r_minor_is_the_midplane_half_width_with_the_lcfs_marked(sample, entries):
    model = _q(entries, coordinate="r_minor")
    major = _q(entries, coordinate="r_major").series[0]
    n = np.asarray(model.series[0].x).size
    inboard, outboard = major.x[:n][::-1], major.x[n:]
    np.testing.assert_allclose(model.series[0].x, (outboard - inboard) / 2.0)
    assert [line.label for line in model.reference_lines] == ["LCFS"]
    assert model.reference_lines[0].x == pytest.approx(model.series[0].x[-1])
    assert model.x_limits[1] == pytest.approx(1.05 * model.series[0].x[-1])


def test_x_limits_are_honoured_when_given(entries):
    assert _q(entries, coordinate="rho_tor_norm", x_limits=(0.2, 0.6)).x_limits == (0.2, 0.6)
    assert _q(entries, coordinate="r_major", x_limits=(0.3, 0.5)).x_limits == (0.3, 0.5)


def test_sqrt_phi_norm_prefers_a_stored_phi_and_equals_rho_tor_norm_otherwise(sample, entries):
    derived = _q(entries, coordinate="sqrt_phi_norm").series[0].x
    rho = _q(entries, coordinate="rho_tor_norm").series[0].x
    np.testing.assert_allclose(derived, rho, atol=1e-12)
    with_phi = copy.deepcopy(sample)
    phi = np.linspace(0.0, 2.0, 129) ** 1.5  # a stored phi that differs from the integrated one
    with_phi[f"equilibrium.time_slice.{SLICE}.profiles_1d.phi"] = phi
    model = build_model("equilibrium_profile_q", normalize_entries(with_phi), time_slice=SLICE, coordinate="sqrt_phi_norm")
    np.testing.assert_allclose(model.series[0].x, np.sqrt(phi / phi[-1]))
    assert model.coordinate_label == COORDINATE_LABELS["sqrt_phi_norm"]


def test_the_callers_ods_is_never_written(sample, entries):
    for coordinate in PROFILE_COORDINATES:
        _q(entries, coordinate=coordinate)
    profiles = sample[f"equilibrium.time_slice.{SLICE}.profiles_1d"]
    assert "r_inboard" not in profiles and "r_outboard" not in profiles and "phi" not in profiles


def test_a_slice_without_a_boundary_degrades_with_a_relabelled_axis(entries):
    model = build_model("equilibrium_profile_q", entries, time_slice=8, coordinate="r_major")
    assert model.coordinate_label in (COORDINATE_LABELS["psi_norm"], COORDINATE_LABELS["index"])
    assert model.reference_lines == ()


def test_the_proxy_rho_tor_norm_is_still_refused_through_the_choice_path(sample, entries):
    from vaft.data._derived import is_rho_pol_proxy

    stored = np.asarray(sample[f"equilibrium.time_slice.{SLICE}.profiles_1d.rho_tor_norm"])
    psi = np.asarray(sample[f"equilibrium.time_slice.{SLICE}.profiles_1d.psi"])
    psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
    assert is_rho_pol_proxy(stored, psi_norm)
    drawn = _q(entries, coordinate="rho_tor_norm").series[0].x
    assert not np.allclose(drawn, stored)


def test_entries_that_disagree_fall_back_together(sample):
    without_q = copy.deepcopy(sample)
    del without_q[f"equilibrium.time_slice.{SLICE}.profiles_1d.q"]
    without_q[f"equilibrium.time_slice.{SLICE}.profiles_1d.q"] = np.asarray(sample[f"equilibrium.time_slice.{SLICE}.profiles_1d.q"])
    without_boundary = copy.deepcopy(sample)
    del without_boundary[f"equilibrium.time_slice.{SLICE}.boundary.outline"]
    model = build_model(
        "equilibrium_profile_q", [("a", sample), ("b", without_boundary)], time_slice=SLICE, coordinate="r_major"
    )
    # One entry cannot supply r_major: both are drawn against the common fallback.
    assert model.coordinate_label != COORDINATE_LABELS["r_major"]
    assert len({np.asarray(trace.x).size for trace in model.series}) == 1


def test_the_overview_and_the_navigator_take_the_coordinate(sample, entries):
    panels = build_model("equilibrium_overview", entries, time_slice=SLICE, coordinate="r_major")
    profiles = [m for m in panels.models if isinstance(m, Profile1D)]
    assert profiles and all(p.coordinate_label == COORDINATE_LABELS["r_major"] for p in profiles)
    assert all(any(line.label == "Magnetic axis" for line in p.reference_lines) for p in profiles)
    result = vaft.omas.plot_equilibrium_interactive(sample, coordinate="r_minor", backend="none")
    labels = {axis.get_xlabel() for axis in result.axes}
    assert COORDINATE_LABELS["r_minor"] in labels


def test_both_renderers_draw_the_reference_lines(sample):
    figure, axes = vaft.omas.plot_equilibrium_profile_q(sample, time_slice=SLICE, coordinate="r_major")
    legend_labels = [t.get_text() for t in axes.get_legend().get_texts()]
    assert "Magnetic axis" in legend_labels and "Limiter" in legend_labels
    assert sum(1 for line in axes.get_lines() if line.get_xdata()[0] == line.get_xdata()[-1] and len(line.get_xdata()) == 2) >= 3
    plotly = vaft.omas.plot_equilibrium_profile_q(sample, time_slice=SLICE, coordinate="r_major", backend="plotly")
    assert sum(1 for shape in plotly.layout.shapes if shape.type == "line") == 3


def test_discovery_and_controls_offer_the_coordinates(sample):
    record = next(r for r in vaft.omas.available_plots(sample) if r.name == "equilibrium_profile_q")
    assert record.coordinates["default"] == "rho_tor_norm"
    assert record.coordinates["options"] == PROFILE_COORDINATES
    assert record.controls == ("time_slice", "coordinate", "orientation")
    assert "coordinates: rho_tor_norm (default) | psi_norm | sqrt_phi_norm | r_major | r_minor" in str(
        vaft.omas.available_plots(sample, query="equilibrium", view="profile")
    )
    bare = copy.deepcopy(sample)
    del bare[f"equilibrium.time_slice.{SLICE}.boundary.outline"]
    narrowed = next(r for r in vaft.omas.available_plots(bare) if r.name == "equilibrium_profile_q")
    assert "r_major" not in narrowed.coordinates["options"] and "r_major" in narrowed.coordinates["declared"]


def test_omas_and_imas_agree_for_every_coordinate(sample):
    from vaft.imas.access import IDSEntry
    from test_imas_omas_plot_equivalence import assert_models_equal

    with vaft.imas.load(vaft.data.sample(39915, representation="imas"), imas_version="3.41.0") as handle:
        entry = IDSEntry(handle)
        for coordinate in PROFILE_COORDINATES:
            expected = build_model("equilibrium_profile_q", [("39915", sample)], time_slice=SLICE, coordinate=coordinate)
            actual = build_model("equilibrium_profile_q", [("39915", entry)], time_slice=SLICE, coordinate=coordinate)
            assert_models_equal(actual, expected)
