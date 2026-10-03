"""Rational-surface resolution and its equilibrium/profile overlays (issue #506).

One resolver, :func:`vaft.process.equilibrium.rational_surfaces`, answers
``q_target -> radius``; the 2-D flux-surface contours and the 1-D profile
markers of :mod:`vaft.plot` are drawn from its records and never search ``q``
themselves.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vaft.process.equilibrium import (
    RATIONAL_SURFACE_ABSENT,
    RATIONAL_SURFACE_PRESENT,
    find_rational_surfaces,
    rational_surfaces,
)


PSI = np.linspace(0.0, 1.0, 101)


def _one(surfaces, q_target):
    (match,) = [s for s in surfaces if s.q_target == pytest.approx(q_target)]
    return match


# --- resolver ------------------------------------------------------------------


def test_a_monotonic_profile_has_one_root_where_q_crosses():
    q = 1.0 + 3.0 * PSI  # q = 2 at psi_N = 1/3
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    assert surface.status == RATIONAL_SURFACE_PRESENT and surface.present
    (root,) = surface.roots
    assert root.psi_norm == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert root.rho_pol_norm == pytest.approx(np.sqrt(1.0 / 3.0))
    assert root.rho_tor_norm is None and root.root_index == 0


def test_a_value_the_profile_never_reaches_is_absent_not_invented():
    q = 1.0 + 3.0 * PSI
    (surface,) = rational_surfaces(PSI, q, q_targets=[5.0])
    assert surface.status == RATIONAL_SURFACE_ABSENT and not surface.present
    assert surface.roots == ()


def test_an_exact_node_is_one_root():
    q = np.linspace(1.0, 3.0, 5)  # q = 2 exactly at node 2
    psi = np.linspace(0.0, 1.0, 5)
    (surface,) = rational_surfaces(psi, q, q_targets=[2.0])
    assert [r.psi_norm for r in surface.roots] == [0.5]


@pytest.mark.parametrize("target, where", [(1.0, 0.0), (4.0, 1.0)])
def test_roots_on_the_axis_and_on_the_edge_are_kept(target, where):
    q = 1.0 + 3.0 * PSI
    (surface,) = rational_surfaces(PSI, q, q_targets=[target])
    assert [r.psi_norm for r in surface.roots] == [where]


def test_reversed_shear_returns_both_roots_ordered_outward():
    q = 3.0 - 6.0 * PSI * (1.0 - PSI)  # min 1.5 at psi_N = 0.5
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    positions = [r.psi_norm for r in surface.roots]
    assert len(positions) == 2 and positions[0] < 0.5 < positions[1]
    assert [r.root_index for r in surface.roots] == [0, 1]
    for position in positions:
        assert 3.0 - 6.0 * position * (1.0 - position) == pytest.approx(2.0, abs=1e-3)


def test_several_targets_come_back_one_record_each_ascending():
    q = 1.0 + 3.0 * PSI
    surfaces = rational_surfaces(PSI, q, q_targets=[3.0, 1.5, 2.0])
    assert [s.q_target for s in surfaces] == [1.5, 2.0, 3.0]
    assert all(s.present for s in surfaces)


def test_harmonics_with_one_ratio_share_one_surface_and_keep_both_names():
    q = 1.0 + 3.0 * PSI
    surfaces = rational_surfaces(PSI, q, resonances=[(2, 1), (4, 2), (3, 2)])
    assert [s.q_target for s in surfaces] == [1.5, 2.0]
    q2 = _one(surfaces, 2.0)
    assert q2.harmonics == ((2, 1), (4, 2)) and not q2.requested_q
    assert len(q2.roots) == 1
    # a plain value meets the harmonics that reduce to it
    merged = rational_surfaces(PSI, q, q_targets=[2.0], resonances=[(2, 1)])
    assert len(merged) == 1 and merged[0].requested_q and merged[0].harmonics == ((2, 1),)


def test_a_nan_gap_is_never_bridged():
    q = 1.0 + 3.0 * PSI
    q[30:40] = np.nan  # q = 2 (psi_N = 1/3) falls inside the gap
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0])
    assert surface.status == RATIONAL_SURFACE_ABSENT
    # find_rational_surfaces drops the samples and interpolates across them
    assert find_rational_surfaces(PSI, q, 1, m_range=(2, 2))["psi_n_rational"].size == 1


def test_nothing_is_extrapolated_beyond_the_profile():
    psi = np.linspace(0.0, 0.8, 81)
    q = 1.0 + 3.0 * psi  # reaches 3.4 at the last sample; 3.5 lies beyond
    (surface,) = rational_surfaces(psi, q, q_targets=[3.5])
    assert surface.status == RATIONAL_SURFACE_ABSENT


def test_rho_tor_norm_is_carried_only_when_supplied():
    q = 1.0 + 3.0 * PSI
    rho_tor = PSI ** 0.7
    (surface,) = rational_surfaces(PSI, q, q_targets=[2.0], rho_tor_norm=rho_tor)
    (root,) = surface.roots
    assert root.rho_tor_norm == pytest.approx(np.interp(root.psi_norm, PSI, rho_tor))
    assert root.rho_pol_norm == pytest.approx(np.sqrt(root.psi_norm))


def test_the_sign_of_q_does_not_decide_resonance():
    q = 1.0 + 3.0 * PSI
    positive = rational_surfaces(PSI, q, q_targets=[2.0], resonances=[(3, 2)])
    negative = rational_surfaces(PSI, -q, q_targets=[2.0], resonances=[(3, -2)])
    assert [[r.psi_norm for r in s.roots] for s in positive] == [
        [r.psi_norm for r in s.roots] for s in negative
    ]


@pytest.mark.parametrize("kwargs", [
    {}, {"q_targets": [0.0]}, {"q_targets": [np.nan]}, {"resonances": [(2, 0)]},
    {"resonances": [(2.5, 1)]}, {"resonances": [2]},
])
def test_bad_requests_are_refused(kwargs):
    with pytest.raises(ValueError):
        rational_surfaces(PSI, 1.0 + PSI, **kwargs)


def test_a_non_increasing_grid_is_refused():
    with pytest.raises(ValueError, match="increase"):
        rational_surfaces(PSI[::-1], 1.0 + PSI, q_targets=[1.5])


# --- view integration ----------------------------------------------------------

import vaft.omas as vo  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

from vaft.plot.backend import recipes as R  # noqa: E402
from vaft.plot.backend.options import validate_options  # noqa: E402


@pytest.fixture(scope="module")
def ods():
    return vo.sample_ods()


@pytest.fixture
def fresh_ods():
    return vo.sample_ods()


def _rational_layers(model):
    return [layer for layer in model.overlays if str(layer.style.get("color", "")).startswith("palette:")]


def _psi_n_at(ods, time_slice, r, z):
    base = f"equilibrium.time_slice.{time_slice}"
    psi = np.asarray(ods[f"{base}.profiles_2d.0.psi"], dtype=float)
    grid_r = np.asarray(ods[f"{base}.profiles_2d.0.grid.dim1"], dtype=float)
    grid_z = np.asarray(ods[f"{base}.profiles_2d.0.grid.dim2"], dtype=float)
    axis, edge = R._psi_axis_boundary(ods, time_slice)
    spline = RectBivariateSpline(grid_r, grid_z, (psi - axis) / (edge - axis))
    return spline.ev(r, z)


def _reverse_shear(ods, time_slice=0):
    """Give slice ``time_slice`` a reversed-shear q: q = 2 at two radii."""
    base = f"equilibrium.time_slice.{time_slice}.profiles_1d"
    psi = np.asarray(ods[f"{base}.psi"], dtype=float)
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
    ods[f"{base}.q"] = 3.0 - 6.0 * psi_n * (1.0 - psi_n)
    return ods


def test_the_2d_contour_lies_on_the_resolved_flux_surface(ods):
    model = vo.extract_equilibrium_field_psi(ods, time_slice=0, rational_q=[5.0])
    (layer,) = _rational_layers(model)
    assert layer.role == R.EQUILIBRIUM_ROLE and layer.kind == "polyline"
    assert layer.label == "q = 5" and layer.style["linestyle"] == "--"
    record = model.metadata["rational_surfaces"]
    (surface,) = record["surfaces"]
    (root,) = surface["roots"]
    finite = np.isfinite(layer.r)
    on = _psi_n_at(ods, 0, layer.r[finite], layer.z[finite])
    assert np.allclose(on, root["psi_norm"], atol=5e-3)
    # the root itself is where |q| = 5 on the slice's own profile
    base = "equilibrium.time_slice.0"
    psi = np.asarray(ods[f"{base}.profiles_1d.psi"], dtype=float)
    axis, edge = R._psi_axis_boundary(ods, 0)
    q = np.abs(np.asarray(ods[f"{base}.profiles_1d.q"], dtype=float))
    assert np.interp(root["psi_norm"], (psi - axis) / (edge - axis), q) == pytest.approx(5.0, rel=1e-6)


def test_two_roots_draw_two_contours_with_one_legend_entry(fresh_ods):
    _reverse_shear(fresh_ods)
    model = vo.extract_equilibrium_field_psi(fresh_ods, time_slice=0, rational_q=[2.0])
    layers = _rational_layers(model)
    assert len(layers) == 2
    assert [layer.label for layer in layers] == ["q = 2", ""]
    roots = model.metadata["rational_surfaces"]["surfaces"][0]["roots"]
    assert len(roots) == 2 and roots[0]["psi_norm"] < roots[1]["psi_norm"]
    for layer, root in zip(layers, roots):
        finite = np.isfinite(layer.r)
        assert np.allclose(_psi_n_at(fresh_ods, 0, layer.r[finite], layer.z[finite]), root["psi_norm"], atol=5e-3)


def test_harmonics_with_one_ratio_draw_one_contour(ods):
    model = vo.extract_equilibrium_field_psi(ods, time_slice=0, resonances=[(3, 1), (6, 2)])
    (layer,) = _rational_layers(model)
    assert layer.label == "3/1, 6/2 (q = 3)"


def test_no_request_leaves_the_map_unchanged(ods):
    plain = vo.extract_equilibrium_field_psi(ods, time_slice=0)
    base = R._build_field_2d_base(ods, R.RECIPES["equilibrium_field_psi"], time_slice=0)
    assert plain.to_xarray().identical(base.to_xarray())
    assert plain.metadata == {} and not _rational_layers(plain)
    profile = vo.extract_equilibrium_profile_q(ods, time_slice=0)
    assert profile.metadata == {} and all(
        not str(line.style.get("color", "")).startswith("palette:") for line in profile.reference_lines
    )


def test_the_rendered_map_differs_only_by_the_overlay(ods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axes = vo.plot_equilibrium_field_psi(ods, time_slice=0)
    before = [line.get_label() for line in axes.lines]
    _, axes2 = vo.plot_equilibrium_field_psi(ods, time_slice=0, rational_q=[5.0])
    after = [line.get_label() for line in axes2.lines]
    assert after[: len(before)] == before and after[len(before):] == ["q = 5"]
    assert len(axes2.collections) == len(axes.collections)
    plt.close("all")


def test_an_absent_surface_warns_and_draws_nothing(ods):
    with pytest.warns(UserWarning, match=r"q = 50 does not occur"):
        model = vo.extract_equilibrium_field_psi(ods, time_slice=0, rational_q=[50.0])
    assert not _rational_layers(model)
    assert model.metadata["rational_surfaces"]["surfaces"][0]["status"] == "absent"


@pytest.mark.parametrize("coordinate", ["psi_norm", "rho_tor_norm"])
def test_a_profile_marker_sits_where_its_own_curve_crosses(ods, coordinate):
    model = vo.extract_equilibrium_profile_q(ods, time_slice=0, coordinate=coordinate, rational_q=[5.0])
    assert model.metadata["rational_surfaces"]["coordinate"] == coordinate
    (line,) = [l for l in model.reference_lines if l.label == "q = 5"]
    (trace,) = model.series
    assert np.interp(line.x, trace.x, np.abs(trace.y)) == pytest.approx(5.0, rel=1e-6)
    root = model.metadata["rational_surfaces"]["surfaces"][0]["roots"][0]
    assert line.x == pytest.approx(root[coordinate])


def test_the_markers_carry_over_to_another_profile_of_the_same_slice(ods):
    q_view = vo.extract_equilibrium_profile_q(ods, time_slice=0, rational_q=[5.0])
    pressure = vo.extract_equilibrium_profile_pressure(ods, time_slice=0, rational_q=[5.0])
    assert [l.x for l in pressure.reference_lines] == [l.x for l in q_view.reference_lines]


def test_two_roots_mark_twice_on_a_profile(fresh_ods):
    _reverse_shear(fresh_ods)
    model = vo.extract_equilibrium_profile_q(fresh_ods, time_slice=0, coordinate="psi_norm", rational_q=[2.0])
    lines = [l for l in model.reference_lines if str(l.style.get("color", "")).startswith("palette:")]
    assert [l.label for l in lines] == ["q = 2", ""]
    assert lines[0].x < 0.5 < lines[1].x


def test_an_unsupported_abscissa_warns_and_marks_nothing(ods):
    with pytest.warns(UserWarning, match="not marked on a r_major abscissa"):
        model = vo.extract_equilibrium_profile_q(ods, time_slice=0, coordinate="r_major", rational_q=[5.0])
    assert all(not str(l.style.get("color", "")).startswith("palette:") for l in model.reference_lines)


def test_the_overlay_uses_the_slice_time_resolves_to(ods):
    times = [float(ods[f"equilibrium.time_slice.{i}.time"]) for i in range(len(ods["equilibrium.time_slice"]))]
    target = 6
    by_time = vo.extract_equilibrium_field_psi(ods, time=times[target] + 1e-4, rational_q=[5.0])
    by_index = vo.extract_equilibrium_field_psi(ods, time_slice=target, rational_q=[5.0])
    record = by_time.metadata["rational_surfaces"]
    assert record["equilibrium_time_slice"] == target
    assert record["equilibrium_time"] == pytest.approx(times[target])
    assert by_time.to_xarray().identical(by_index.to_xarray())
    # and it differs from slice 0's, so the slice really is the one used
    other = vo.extract_equilibrium_field_psi(ods, time_slice=0, rational_q=[5.0])
    assert other.metadata["rational_surfaces"]["surfaces"][0]["roots"] != record["surfaces"][0]["roots"]
    profile = vo.extract_equilibrium_profile_q(ods, time=times[target], rational_q=[5.0])
    assert profile.metadata["rational_surfaces"]["equilibrium_time_slice"] == target


def test_a_core_profile_states_which_equilibrium_its_markers_came_from(fresh_ods):
    eq_time = float(fresh_ods["equilibrium.time_slice.6.time"])
    rho = np.linspace(0.0, 1.0, 21)
    fresh_ods["core_profiles.time"] = np.array([eq_time + 2e-4])
    fresh_ods["core_profiles.profiles_1d.0.time"] = eq_time + 2e-4
    fresh_ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = rho
    fresh_ods["core_profiles.profiles_1d.0.grid.rho_pol_norm"] = rho
    fresh_ods["core_profiles.profiles_1d.0.electrons.temperature"] = 100.0 * (1.0 - rho ** 2)
    with pytest.warns(UserWarning, match="not the profile's own time"):
        model = vo.extract_electron_temperature_profile(fresh_ods, time_slice=0, rational_q=[5.0])
    record = model.metadata["rational_surfaces"]
    assert record["equilibrium_time_slice"] == 6
    assert record["profile_time"] == pytest.approx(eq_time + 2e-4)
    (line,) = [l for l in model.reference_lines if l.label.startswith("q = 5")]
    assert "eq. t =" in line.label
    assert line.x == pytest.approx(record["surfaces"][0]["roots"][0]["rho_tor_norm"])


@pytest.mark.parametrize("name", ["wall_geometry_poloidal", "ion_temperature_profile", "flux_loop_time_flux"])
def test_a_plot_that_does_not_declare_the_options_refuses_them(name):
    for key, value in (("rational_q", [2.0]), ("resonances", [(2, 1)])):
        with pytest.raises(ValueError, match=f"does not take an option named '{key}'"):
            validate_options(name, {key: value})


def test_the_declaring_plots_accept_the_options_and_say_so():
    for name in R.RATIONAL_SURFACE_PLOTS:
        validate_options(name, {"rational_q": [2.0], "resonances": [(2, 1)]})


def test_discovery_lists_the_overlay_for_declaring_plots_only(ods):
    catalog = {record.name: record for record in vo.available_plots(ods)}
    assert catalog["equilibrium_field_psi"].annotations["rational_surfaces"]["options"] == (
        "rational_q", "resonances")
    assert catalog["equilibrium_profile_q"].annotations
    assert not catalog["wall_geometry_poloidal"].annotations
