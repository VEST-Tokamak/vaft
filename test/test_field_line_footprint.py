"""The relative footprint proxy and the surface areas that integrate it (#1099).

Tested against closed forms rather than against the legacy implementation,
because the legacy is what the migration exists to replace: it multiplied the
proxy by ``1 / area``, and the test that would have caught that is the one
asking whether the per-target integral converges under grid refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.field_line_topology import (
    CONNECTION_LENGTH_FORMS,
    FOOTPRINT_PROXY_MODELS,
    FootprintProxyModel,
    connection_length_weight,
    footprint_heat_load_proxy,
    footprint_incident_total,
    footprint_proxy_model,
    incidence_factor,
    target_incident_fractions,
    toroidal_surface_cell_areas,
    toroidal_surface_node_areas,
    upstream_flux_weight,
)


def cylinder(n_v, n_u, *, radius=2.0, height=1.5, span_deg=90.0):
    """A cylindrical target: ``n_v`` heights swept over ``n_u`` angles."""
    z_axis = np.linspace(0.0, height, n_v)
    phi_axis = np.linspace(0.0, span_deg, n_u)
    z = np.tile(z_axis[:, None], (1, n_u))
    phi = np.tile(phi_axis[None, :], (n_v, 1))
    return np.full((n_v, n_u), float(radius)), z, phi


def shoelace(polygon):
    """Exact area of a planar polygon, as an independent reference."""
    x, y = np.asarray(polygon)[:, 0], np.asarray(polygon)[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0


# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------


def test_a_planar_cell_matches_the_shoelace_area_exactly():
    """On a flat surface the two-triangle construction is the polygon area,
    so it can be checked against a closed form rather than a tolerance.  The
    mesh is deliberately irregular: a uniform one would pass with a formula
    that only happened to be right on rectangles."""
    r_axis = np.array([1.0, 1.4, 2.9, 3.1])
    phi_axis = np.array([0.0, 7.0, 51.0])
    r = np.tile(r_axis[:, None], (1, phi_axis.size))
    phi = np.tile(phi_axis[None, :], (r_axis.size, 1))
    z = np.zeros_like(r)

    areas = toroidal_surface_cell_areas(r, z, phi)

    radians = np.deg2rad(phi)
    x, y = r * np.cos(radians), r * np.sin(radians)
    for j in range(r_axis.size - 1):
        for i in range(phi_axis.size - 1):
            corners = [(x[j, i], y[j, i]), (x[j, i + 1], y[j, i + 1]),
                       (x[j + 1, i + 1], y[j + 1, i + 1]), (x[j + 1, i], y[j + 1, i])]
            assert areas[j, i] == pytest.approx(shoelace(corners), rel=1e-14)


def test_the_cell_areas_converge_on_the_analytic_cylinder():
    """A flat quadrilateral inscribed in a cylinder is slightly smaller than
    the curved patch it spans, by a factor that vanishes as the mesh refines.
    The total is therefore an approximation that converges, not an identity."""
    exact = 2.0 * np.deg2rad(90.0) * 1.5
    errors = []
    for count in (10, 20, 40, 80):
        total = toroidal_surface_cell_areas(*cylinder(3, count)).sum()
        errors.append(abs(total - exact) / exact)
    assert errors[-1] < 1e-4
    # Second order: each doubling cuts the error by about four.
    ratios = [before / after for before, after in zip(errors, errors[1:])]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), ratios


def test_the_node_areas_sum_to_the_surface_area():
    """Splitting each cell between its four corners keeps the total, which is
    what lets an integral over nodes be the same integral as over cells."""
    mesh = cylinder(7, 9)
    cells = toroidal_surface_cell_areas(*mesh)
    nodes = toroidal_surface_node_areas(*mesh)
    assert nodes.shape == mesh[0].shape
    assert nodes.sum() == pytest.approx(cells.sum(), rel=1e-14)


def test_a_node_carries_a_quarter_of_each_cell_it_touches():
    mesh = cylinder(4, 5)
    cells = toroidal_surface_cell_areas(*mesh)
    nodes = toroidal_surface_node_areas(*mesh)
    assert nodes[0, 0] == pytest.approx(cells[0, 0] / 4.0)
    assert nodes[0, 2] == pytest.approx((cells[0, 1] + cells[0, 2]) / 4.0)
    assert nodes[1, 2] == pytest.approx(
        (cells[0, 1] + cells[0, 2] + cells[1, 1] + cells[1, 2]) / 4.0
    )


def test_areas_refuse_arrays_that_do_not_describe_one_mesh():
    r, z, phi = cylinder(3, 4)
    with pytest.raises(ValueError, match="share a shape"):
        toroidal_surface_cell_areas(r, z[:, :2], phi)
    with pytest.raises(ValueError, match="at least two nodes"):
        toroidal_surface_cell_areas(r[:1], z[:1], phi[:1])


# ---------------------------------------------------------------------------
# The three weights
# ---------------------------------------------------------------------------


def test_the_upstream_weight_is_the_exponential_it_claims_to_be():
    psi = np.array([1.0, 1.02, 1.04])
    weight = upstream_flux_weight(psi, separatrix_flux=1.0, decay_width=0.02)
    np.testing.assert_allclose(weight, [1.0, np.exp(-1.0), np.exp(-2.0)])


def test_a_line_that_reached_inside_the_separatrix_is_clipped_or_not_as_asked():
    psi = np.array([0.9])
    clipped = upstream_flux_weight(psi, separatrix_flux=1.0, decay_width=0.02)
    free = upstream_flux_weight(psi, separatrix_flux=1.0, decay_width=0.02,
                                clip_inside=False)
    assert clipped[0] == pytest.approx(1.0)
    assert free[0] == pytest.approx(np.exp(5.0))


def test_a_non_positive_channel_width_is_refused():
    with pytest.raises(ValueError, match="decay_width must be positive"):
        upstream_flux_weight([1.0], separatrix_flux=1.0, decay_width=0.0)


def test_the_incidence_factor_is_the_sine_of_the_angle_from_the_surface():
    """FLARE's alphaS is measured from the target, not from its normal, so a
    grazing line has a small angle and a small factor."""
    np.testing.assert_allclose(
        incidence_factor([0.0, 30.0, 90.0, -30.0]), [0.0, 0.5, 1.0, 0.5], atol=1e-15
    )


@pytest.mark.parametrize("form", CONNECTION_LENGTH_FORMS)
def test_each_connection_length_form_is_its_closed_form(form):
    length = np.array([0.0, 50.0, 100.0])
    weight = connection_length_weight(length, scale=50.0, form=form)
    expected = {
        "inverse": [1.0, 0.5, 1.0 / 3.0],
        "exponential": [1.0, np.exp(-1.0), np.exp(-2.0)],
        "none": [1.0, 1.0, 1.0],
    }[form]
    np.testing.assert_allclose(weight, expected)


def test_a_line_the_tracer_could_not_follow_carries_no_weight():
    weight = connection_length_weight([np.nan, np.inf, -5.0], scale=50.0, form="inverse")
    np.testing.assert_allclose(weight, [0.0, 0.0, 1.0])


def test_an_unknown_connection_length_form_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="inverse"):
        connection_length_weight([1.0], scale=50.0, form="quadratic")


def test_a_flat_connection_weight_needs_no_scale():
    np.testing.assert_allclose(
        connection_length_weight([1.0, 2.0], scale=0.0, form="none"), [1.0, 1.0]
    )


# ---------------------------------------------------------------------------
# The model is a named preset, never a default
# ---------------------------------------------------------------------------


def test_the_registered_models_carry_the_coefficients_the_scans_used():
    model = footprint_proxy_model("gpec_research_inverse")
    assert (model.decay_width, model.connection_length_scale) == (0.02, 50.0)
    assert model.connection_length_form == "inverse"
    assert set(FOOTPRINT_PROXY_MODELS) == {
        "gpec_research_inverse", "gpec_research_exponential"
    }


def test_an_unknown_model_lists_the_registered_ones():
    with pytest.raises(KeyError, match="gpec_research_inverse"):
        footprint_proxy_model("whatever")


def test_a_model_validates_its_own_coefficients():
    with pytest.raises(ValueError, match="decay_width"):
        FootprintProxyModel("m", 0.0, 1.0, 50.0, "inverse", True)
    with pytest.raises(ValueError, match="connection_length_form"):
        FootprintProxyModel("m", 0.02, 1.0, 50.0, "linear", True)
    with pytest.raises(ValueError, match="connection_length_scale"):
        FootprintProxyModel("m", 0.02, 1.0, 0.0, "inverse", True)
    with pytest.raises(ValueError, match="must be named"):
        FootprintProxyModel("", 0.02, 1.0, 50.0, "inverse", True)


def test_the_model_describes_itself_for_a_provenance_record():
    text = footprint_proxy_model("gpec_research_exponential").describe()
    assert "lambda_psi=0.02" in text and "L_0=50.0 m" in text
    assert "form=exponential" in text


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------


MODEL = FOOTPRINT_PROXY_MODELS["gpec_research_inverse"]


def proxy_inputs(shape=(3, 4)):
    return {
        "min_psi_norm": np.full(shape, 1.01),
        "connection_length": np.full(shape, 50.0),
        "area": np.full(shape, 0.5),
    }


def test_the_proxy_is_the_product_of_its_three_factors():
    result = footprint_heat_load_proxy(
        **proxy_inputs(), incidence_angle_deg=np.full((3, 4), 30.0), model=MODEL
    )
    np.testing.assert_allclose(
        result.density, result.flux_weight * result.incidence * result.connection_weight
    )
    np.testing.assert_allclose(result.flux_weight, np.exp(-0.5))
    np.testing.assert_allclose(result.incidence, 0.5)
    np.testing.assert_allclose(result.connection_weight, 0.5)


def test_a_missing_incidence_column_must_be_opted_out_of_explicitly():
    """The committed reference footprint has no alphaS, and the legacy gave
    that case a factor of one with nothing in the output to say so."""
    result = footprint_heat_load_proxy(
        **proxy_inputs(), incidence_angle_deg=None, model=MODEL
    )
    assert result.incidence_applied is False
    np.testing.assert_allclose(result.incidence, 1.0)
    assert "omitted by the caller" in result.describe()

    applied = footprint_heat_load_proxy(
        **proxy_inputs(), incidence_angle_deg=np.full((3, 4), 90.0), model=MODEL
    )
    assert applied.incidence_applied is True
    assert "alphaS" in applied.describe()


def test_the_proxy_refuses_inputs_on_different_grids():
    inputs = proxy_inputs()
    inputs["area"] = np.full((3, 5), 0.5)
    with pytest.raises(ValueError, match="same grid"):
        footprint_heat_load_proxy(**inputs, incidence_angle_deg=None, model=MODEL)


def test_the_incidence_angles_must_be_on_the_grid_too():
    with pytest.raises(ValueError, match="same grid"):
        footprint_heat_load_proxy(
            **proxy_inputs(), incidence_angle_deg=np.zeros((2, 2)), model=MODEL
        )


def test_the_relative_field_peaks_at_one_and_survives_an_empty_map():
    result = footprint_heat_load_proxy(
        min_psi_norm=[[1.0, 1.02]], connection_length=[[0.0, 0.0]],
        area=[[1.0, 1.0]], incidence_angle_deg=None, model=MODEL,
    )
    assert result.relative.max() == pytest.approx(1.0)

    dark = footprint_heat_load_proxy(
        min_psi_norm=[[1.0, 1.0]], connection_length=[[np.nan, np.nan]],
        area=[[1.0, 1.0]], incidence_angle_deg=None, model=MODEL,
    )
    np.testing.assert_allclose(dark.relative, 0.0)


# ---------------------------------------------------------------------------
# The integral, which is where the legacy's extra 1/area factor shows
# ---------------------------------------------------------------------------


def exponential_target(n_v, n_u, *, decay=0.25, radius=2.0, height=1.5, span_deg=90.0):
    """A cylindrical target whose proxy is ``exp(-z / decay)`` exactly."""
    r, z, phi = cylinder(n_v, n_u, radius=radius, height=height, span_deg=span_deg)
    model = FootprintProxyModel(
        name="analytic", decay_width=decay, separatrix_flux=0.0,
        connection_length_scale=1.0, connection_length_form="none", clip_inside=False,
    )
    return footprint_heat_load_proxy(
        min_psi_norm=z,
        connection_length=np.zeros_like(z),
        area=toroidal_surface_node_areas(r, z, phi),
        incidence_angle_deg=None,
        model=model,
    )


def test_the_incident_total_converges_on_the_analytic_integral():
    """``integral of exp(-z/l) dA`` over the cylinder is
    ``R dphi l (1 - exp(-H/l))``.  The sum over node areas is a trapezoidal
    Riemann sum of it, so refining the grid drives it there."""
    decay, radius, height, span = 0.25, 2.0, 1.5, 90.0
    exact = (radius * np.deg2rad(span) * decay * (1.0 - np.exp(-height / decay)))

    errors = []
    for count in (9, 17, 33, 65):
        total = footprint_incident_total(
            exponential_target(count, count, decay=decay, radius=radius,
                               height=height, span_deg=span)
        )
        errors.append(abs(total - exact) / exact)
    assert errors[-1] < 1e-3
    ratios = [before / after for before, after in zip(errors, errors[1:])]
    assert all(3.0 < ratio < 5.0 for ratio in ratios), ratios


def test_the_legacy_hit_density_factor_does_not_converge():
    """With the legacy's extra ``1 / area``, the integral reduces to the bare
    sum of the weights -- so it grows with the number of nodes the caller
    chose rather than settling on the target's own value.  A fraction built
    from it is a fraction of the grid."""
    totals = []
    for count in (9, 17, 33):
        target = exponential_target(count, count)
        totals.append(float(np.nansum(target.density)))
    growth = [after / before for before, after in zip(totals, totals[1:])]
    # Roughly four, because both axes double; nothing like the constant a
    # converging integral gives.
    assert all(ratio > 3.0 for ratio in growth), growth


def test_a_line_the_tracer_lost_does_not_erase_its_target():
    target = exponential_target(5, 5)
    spoiled = target.density.copy()
    spoiled[0, 0] = np.nan
    total = footprint_incident_total(
        type(target)(
            density=spoiled, area=target.area, flux_weight=target.flux_weight,
            incidence=target.incidence, connection_weight=target.connection_weight,
            model=target.model, incidence_applied=target.incidence_applied,
        )
    )
    assert np.isfinite(total) and total > 0.0


# ---------------------------------------------------------------------------
# Shares between targets
# ---------------------------------------------------------------------------


def test_the_shares_add_to_one():
    shares = target_incident_fractions({"inner": 3.0, "outer": 1.0})
    assert shares == {"inner": 0.75, "outer": 0.25}
    assert sum(shares.values()) == pytest.approx(1.0)


def test_a_share_of_nothing_is_refused_rather_than_split_evenly():
    with pytest.raises(ValueError, match="would be an invention"):
        target_incident_fractions({"inner": 0.0, "outer": 0.0})


def test_a_negative_or_non_finite_total_is_refused():
    with pytest.raises(ValueError, match="'inner'"):
        target_incident_fractions({"inner": -1.0, "outer": 1.0})
    with pytest.raises(ValueError, match="'outer'"):
        target_incident_fractions({"inner": 1.0, "outer": np.nan})


def test_no_targets_at_all_is_refused():
    with pytest.raises(ValueError, match="no targets"):
        target_incident_fractions({})
