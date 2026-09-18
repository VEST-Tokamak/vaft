"""Equilibrium views for the MHD-equilibrium session (#952).

Shape histories from the IDS boundary leaves, the kinetic pressure family in
the EFIT constraint views, the per-family constraint weights, the
pressure-weight scan, X-points over a psi map, and the analytic Miller and
Solov'ev plots.
"""

import copy
import logging
import warnings

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

import vaft.omas
from vaft.omas.entries import normalize_entries
from vaft.plot.backend.recipes import build_model
from vaft.plot.models import Field2D, GeometryLayers, LineSeries, Panels, TextPanel


@pytest.fixture(scope="module")
def sample():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ods = vaft.omas.sample_ods(39915)
    logging.disable(logging.NOTSET)
    return ods


@pytest.fixture(scope="module")
def derived(sample):
    ods = copy.deepcopy(sample)
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vaft.omas.update_equilibrium_derived_profiles(ods)
    logging.disable(logging.NOTSET)
    return ods


# ---------------------------------------------------------------------------
# shape histories
# ---------------------------------------------------------------------------


def _leaf(ods, name):
    count = len(ods["equilibrium.time_slice"])
    out = []
    for index in range(count):
        path = f"equilibrium.time_slice.{index}.boundary.{name}"
        out.append(float(ods[path]) if path in ods else np.nan)
    return np.array(out)


@pytest.mark.parametrize("quantity", [
    "elongation", "triangularity", "triangularity_upper", "triangularity_lower", "minor_radius",
])
def test_shape_histories_draw_the_boundary_leaves(derived, quantity):
    model = build_model(f"equilibrium_time_{quantity}", normalize_entries(derived))
    assert isinstance(model, LineSeries)
    y = model.series[0].y
    stored = _leaf(derived, quantity)
    finite = np.isfinite(stored)
    np.testing.assert_allclose(y[np.isfinite(y)], stored[finite])


def test_the_updater_fills_the_boundary_leaves_it_is_documented_to(derived):
    kappa = _leaf(derived, "elongation")
    upper, lower, mean = (_leaf(derived, name) for name in ("triangularity_upper", "triangularity_lower", "triangularity"))
    assert np.isfinite(kappa[:8]).all() and np.all((kappa[:8] > 1.2) & (kappa[:8] < 2.0))
    np.testing.assert_allclose(mean[:8], 0.5 * (upper[:8] + lower[:8]))


def test_a_slice_with_no_plasma_gets_no_geometric_axis(derived):
    # slice 8 of 39915 has psi_axis == psi_boundary and no current: the updater
    # used to write the grid centre (R = 1.7 m) as its geometric axis.  It now
    # writes NaN -- not an absent leaf -- so a colon read stays rectangular and
    # float() of the leaf is nan rather than a LookupError.
    assert np.isnan(float(derived["equilibrium.time_slice.8.boundary.geometric_axis.r"]))
    assert np.isnan(float(derived["equilibrium.time_slice.8.boundary.geometric_axis.z"]))
    column = np.asarray(derived["equilibrium.time_slice.:.boundary.geometric_axis.r"], float)
    assert column.shape == (len(derived["equilibrium.time_slice"]),)
    assert np.isfinite(column[:8]).all() and np.isnan(column[8])
    model = build_model("equilibrium_time_major_radius", normalize_entries(derived))
    assert np.nanmax(model.series[0].y) < 0.6


def test_a_no_plasma_slice_with_a_degenerate_outline_gets_nan_too():
    from vaft.omas.update import update_equilibrium_boundary

    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = [0.30, 0.31]
    for index in range(2):
        root = f"equilibrium.time_slice.{index}"
        ods[f"{root}.time"] = ods["equilibrium.time"][index]
        ods[f"{root}.global_quantities.psi_axis"] = 0.01
        ods[f"{root}.global_quantities.psi_boundary"] = 0.01
        ods[f"{root}.global_quantities.ip"] = 0.0
        ods[f"{root}.global_quantities.magnetic_axis.r"] = 0.45
        ods[f"{root}.global_quantities.magnetic_axis.z"] = 0.0
    # slice 0: fewer than 3 outline points; slice 1: no outline at all
    ods["equilibrium.time_slice.0.boundary.outline.r"] = [0.4, 0.5]
    ods["equilibrium.time_slice.0.boundary.outline.z"] = [0.0, 0.1]
    logging.disable(logging.WARNING)
    update_equilibrium_boundary(ods)
    logging.disable(logging.NOTSET)
    column = np.asarray(ods["equilibrium.time_slice.:.boundary.geometric_axis.r"], float)
    assert column.shape == (2,) and np.isnan(column).all()
    assert np.isnan(np.asarray(ods["equilibrium.time_slice.:.boundary.geometric_axis.z"], float)).all()


def test_the_shape_panel_has_one_member_per_descriptor(derived):
    model = build_model("equilibrium_time_shape", normalize_entries(derived))
    assert isinstance(model, Panels) and len(model.models) == 5


# ---------------------------------------------------------------------------
# the kinetic pressure family and the constraint views
# ---------------------------------------------------------------------------


def _with_pressure_constraint(sample, factor=1.0, slice_index=0):
    ods = copy.deepcopy(sample)
    root = f"equilibrium.time_slice.{slice_index}"
    psi = np.asarray(ods[f"{root}.profiles_1d.psi"], float)
    axis = float(ods[f"{root}.global_quantities.psi_axis"])
    edge = float(ods[f"{root}.global_quantities.psi_boundary"])
    psi_n = (psi - axis) / (edge - axis)
    blend = factor / (1.0 + factor)
    p_kin = 800.0 * (1 - psi_n) ** 1.5
    ods[f"{root}.profiles_1d.pressure"] = (1 - blend) * np.asarray(ods[f"{root}.profiles_1d.pressure"], float) + blend * p_kin
    ods[f"{root}.global_quantities.beta_pol"] = 0.3 * (1 + 0.3 * blend)
    ods[f"{root}.global_quantities.li_3"] = 0.9 * (1 - 0.1 * blend)
    points = np.array([0.05, 0.2, 0.4, 0.6, 0.8])
    measured = 800.0 * (1 - points) ** 1.5
    reconstructed = np.interp(points, psi_n, ods[f"{root}.profiles_1d.pressure"])
    sigma = 80.0
    for j, x in enumerate(points):
        base = f"{root}.constraints.pressure.{j}"
        ods[f"{base}.position.psi"] = axis + x * (edge - axis)
        ods[f"{base}.measured"] = measured[j]
        ods[f"{base}.measured_error_upper"] = sigma
        ods[f"{base}.reconstructed"] = reconstructed[j]
        ods[f"{base}.weight"] = factor / sigma
        ods[f"{base}.chi_squared"] = ((measured[j] - reconstructed[j]) * factor / sigma) ** 2
    return ods


def test_the_pressure_family_enters_the_fit_quality_but_not_the_magnetic_grade(sample):
    from vaft.omas.efit_quality import FAMILIES, fit_quality_metrics
    from vaft.validation.equilibrium import validate_magnetic_fit

    ods = _with_pressure_constraint(sample)
    metrics = fit_quality_metrics(ods, time_slice=0)
    pressure = metrics["families"]["pressure"]
    table_chi = sum(float(ods[f"equilibrium.time_slice.0.constraints.pressure.{j}.chi_squared"]) for j in range(5))
    assert pressure["chi_squared_sum"] == pytest.approx(table_chi)
    assert pressure["sigma_unit_factor"] == pytest.approx(1.0)
    assert "pressure" in metrics["chi_squared_share"]
    assert "pressure" not in {family for family, *_ in FAMILIES}
    assert "pressure" not in validate_magnetic_fit(ods, time_slice=0)
    # a magnetics-only reconstruction reports no pressure family at all
    assert "pressure" not in fit_quality_metrics(sample, time_slice=0)["families"]


def test_the_residual_and_constraint_views_show_the_pressure_family_only_when_present(sample):
    with_pressure = _with_pressure_constraint(sample)
    for name in ("equilibrium_overview_residuals", "equilibrium_overview_constraints"):
        titles = [m.title for m in build_model(name, normalize_entries(with_pressure)).models]
        assert any("Kinetic pressure" in title for title in titles), name
        titles = [m.title for m in build_model(name, normalize_entries(sample)).models]
        assert not any("Kinetic pressure" in title for title in titles), name


def test_constraint_weights_put_the_fitted_sigma_beside_the_stored_one(sample):
    from vaft.omas.efit_quality import constraint_table, normalized_residuals, sigma_unit_factor

    ods = _with_pressure_constraint(sample)
    model = build_model("equilibrium_overview_constraint_weights", normalize_entries(ods))
    assert isinstance(model, Panels) and model.ncols == 3
    rows = [model.models[i:i + 3] for i in range(0, len(model.models), 3)]
    pressure_row = next(row for row in rows if row[0].title.startswith("Kinetic pressure"))
    table = constraint_table(ods, time_slice=0, family="pressure")
    k, _ = sigma_unit_factor(table)
    sigma, weight, z = pressure_row
    fitted = next(series for series in sigma.series if "σ_eff" in series.label)
    np.testing.assert_allclose(fitted.y, k / table.weight)
    np.testing.assert_allclose(weight.series[0].y, table.weight)
    np.testing.assert_allclose(z.series[0].y, normalized_residuals(table, k))
    # the synthetic weight is 1/sigma, so the fitted and stored sigmas agree
    assert "median σ_eff/σ = 1" in sigma.title


def test_a_pressure_weight_scan_is_labelled_and_ordered_by_its_factors(sample):
    scan = {10.0: _with_pressure_constraint(sample, 10.0), 0.1: _with_pressure_constraint(sample, 0.1),
            1.0: _with_pressure_constraint(sample, 1.0)}
    entries = normalize_entries(scan)
    assert [label for label, _ in entries] == ["10.0", "0.1", "1.0"]
    model = build_model("equilibrium_overview_pressure_weight_scan", entries)
    pressure = model.models[0]
    curves = [series.label for series in pressure.series if "weight" in series.label]
    assert curves == ["weight ×0.1", "weight ×1", "weight ×10"]
    assert any("constraint" in series.label for series in pressure.series)
    beta = next(m for m in model.models if isinstance(m, LineSeries) and m.title.startswith("β_p"))
    assert np.all(np.diff(beta.series[0].y) > 0)
    np.testing.assert_allclose(beta.series[0].x, [-1.0, 0.0, 1.0])  # log10 of the factors
    table = next(m for m in model.models if isinstance(m, TextPanel))
    assert len(table.lines) == 4 and table.lines[1].startswith("×0.1")


# ---------------------------------------------------------------------------
# X-points over the psi map
# ---------------------------------------------------------------------------


def _solovev_ods(topology):
    from vaft.process.equilibrium import solovev_example

    eq = solovev_example(topology)
    ods = ODS(consistency_check=False)
    ts = "equilibrium.time_slice.0"
    ods["equilibrium.ids_properties.cocos"] = 11
    ods["equilibrium.time"] = np.array([0.3])
    ods[f"{ts}.time"] = 0.3
    ods[f"{ts}.profiles_2d.0.grid.dim1"] = eq.r
    ods[f"{ts}.profiles_2d.0.grid.dim2"] = eq.z
    ods[f"{ts}.profiles_2d.0.psi"] = eq.psi
    ods[f"{ts}.global_quantities.psi_axis"] = eq.psi_axis
    ods[f"{ts}.global_quantities.psi_boundary"] = eq.psi_boundary
    ods[f"{ts}.global_quantities.magnetic_axis.r"] = eq.magnetic_axis[0]
    ods[f"{ts}.global_quantities.magnetic_axis.z"] = eq.magnetic_axis[1]
    ods[f"{ts}.boundary.outline.r"] = eq.lcfs.r
    ods[f"{ts}.boundary.outline.z"] = eq.lcfs.z
    return ods, eq


@pytest.mark.parametrize("topology, count", [("double_null", 2), ("single_null", 1)])
def test_x_points_overlay_marks_the_boundary_saddles(topology, count):
    ods, eq = _solovev_ods(topology)
    model = build_model("equilibrium_field_2d", normalize_entries(ods), overlay=("boundary", "axis", "x_points"))
    layer = next(layer for layer in model.overlays if layer.label.startswith("X-point"))
    assert layer.r.size == count
    requested = np.array(eq.metadata["x_points_requested"], dtype=float).reshape(-1, 2)
    for r, z in zip(layer.r, layer.z):
        assert np.min(np.hypot(requested[:, 0] - r, requested[:, 1] - z)) < 0.01


def test_x_points_are_not_drawn_unless_asked(sample):
    model = build_model("equilibrium_field_2d", normalize_entries(sample))
    assert not any(layer.label.startswith("X-point") for layer in model.overlays)
    model = build_model("equilibrium_field_2d", normalize_entries(sample), overlay=("wall", "x_points"))
    # 39915 is limited: no boundary saddle, and none outside the limiter is drawn
    assert not any(layer.label.startswith("X-point") for layer in model.overlays)


# ---------------------------------------------------------------------------
# analytic plots
# ---------------------------------------------------------------------------


def test_miller_surfaces_are_labelled_by_what_varies():
    from vaft.plot import miller_surfaces_model
    from vaft.process.equilibrium import miller_surfaces

    model = miller_surfaces_model(miller_surfaces(0.25, r0=0.4, kappa=1.7, delta=[-0.3, 0.0, 0.3]))
    assert isinstance(model, GeometryLayers)
    assert [layer.label for layer in model.layers] == ["δ=-0.3", "δ=0", "δ=0.3"]
    assert "δ" in model.title
    top = [float(np.max(layer.z)) for layer in model.layers]
    np.testing.assert_allclose(top, 1.7 * 0.25, rtol=1e-3)


@pytest.mark.parametrize("topology, active", [("limited", 0), ("single_null", 1), ("double_null", 2)])
def test_the_solovev_plot_marks_its_topology(topology, active):
    from vaft.plot import solovev_equilibrium_model
    from vaft.process.equilibrium import solovev_example

    model = solovev_equilibrium_model(solovev_example(topology))
    assert isinstance(model, Field2D)
    assert topology.replace("_", " ") in model.title
    x_layers = [layer for layer in model.overlays if layer.label == "X-point"]
    assert sum(layer.r.size for layer in x_layers) == active
    assert 1.0 in list(model.contour_levels)
    # psi_N is 0 at the axis and 1 on the LCFS
    assert np.nanmin(model.values) == pytest.approx(0.0, abs=0.02)


def test_the_analytic_plots_render():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vaft.plot import plot_miller_surfaces, plot_solovev_equilibrium
    from vaft.process.equilibrium import miller_surfaces, solovev_example

    fig, ax = plot_solovev_equilibrium(solovev_example("single_null"))
    plt.close(fig)
    fig, ax = plot_miller_surfaces(miller_surfaces([0.1, 0.2], r0=0.4, kappa=1.5))
    plt.close(fig)


def test_the_pressure_scan_drops_points_without_psi_and_never_normalizes_by_them(sample):
    from vaft.plot.backend.recipes import _normalized_psi

    ods = _with_pressure_constraint(sample)
    root = "equilibrium.time_slice.0"
    # two points lose position.psi and carry only rho_tor_norm: rho**2 is not psi_N
    for j in (1, 3):
        del ods[f"{root}.constraints.pressure.{j}.position.psi"]
        ods[f"{root}.constraints.pressure.{j}.position.rho_tor_norm"] = 0.5
    model = build_model("equilibrium_overview_pressure_weight_scan", normalize_entries({1.0: ods}))
    points = [s for s in model.models[0].series if "constraint" in s.label][0]
    assert points.x.size == 3
    np.testing.assert_allclose(points.x, [0.05, 0.4, 0.8], atol=1e-9)
    # a degenerate slice: no psi_N from the points' own first and last value
    flat = copy.deepcopy(ods)
    flat[f"{root}.global_quantities.psi_boundary"] = flat[f"{root}.global_quantities.psi_axis"]
    psi_points = np.array([0.1, 0.2, 0.3])
    assert _normalized_psi(flat, root, psi_points) is None
    profile = np.asarray(flat[f"{root}.profiles_1d.psi"], float)
    normalized = _normalized_psi(flat, root, psi_points, profile_psi=profile)
    np.testing.assert_allclose(normalized, (psi_points - profile[0]) / (profile[-1] - profile[0]))
