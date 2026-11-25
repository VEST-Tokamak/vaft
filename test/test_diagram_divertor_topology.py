"""Divertor topology and signed separatrix-balance diagrams (#2325 phase E).

The geometry is the analytic family psi = (R-1)^2 + Z^2 - Z^4/a^2 + s Z^3,
whose saddle fluxes and outboard-midplane intersections are known in closed
form: R = 1 + sqrt(psi_X) at Z = 0. The diagrams must draw what
derive_boundary_representation reports, and that must agree with the exact
values -- the tests check the drawing against the closed form, not against
the code that drew it.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _divertor_topology as dt
from vaft.diagram._scene import Label, Marker, Polyline
from vaft.formula.equilibrium import separatrix_radial_separation
from vaft.formula.sol import secondary_separatrix_exponential_factor, separatrix_imbalance_to_heat_flux_width

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = ("divertor_topology_classification", "secondary_separatrix_balance", "sol_magnetic_connectivity",
            "separatrix_balance_transport_scale", "separatrix_balance_calculation", "limited_plasma_topology",
            "lower_single_null_topology", "double_null_topology")


def _exact_saddle_fluxes(skew):
    """Lower and upper saddle fluxes of the family, by hand."""
    a = dt.FAMILY_A
    roots = sorted(np.roots([4.0 / a**2, -3.0 * skew, -2.0]).real)
    return [float(z**2 - z**4 / a**2 + skew * z**3) for z in roots]


def _exact_d_r_sep(skew):
    psi_l, psi_u = _exact_saddle_fluxes(skew)
    return np.sqrt(psi_l) - np.sqrt(psi_u)


@pytest.fixture(scope="module")
def diagrams():
    return {name: getattr(vaft.diagram, name)() for name in BUILDERS}


# -- the model the diagrams draw ---------------------------------------------------


@pytest.mark.parametrize("key, title, skew, tip", dt.CLASSIFICATION_CASES, ids=[c[0] for c in dt.CLASSIFICATION_CASES])
def test_each_classification_panel_is_what_the_process_reports(key, title, skew, tip):
    state = dt.balance_state(skew, limiter_tip=tip)
    expected = {"near_double_null": "lower_single_null"}.get(key, key)
    assert state["topology"] == expected
    assert state["d_r_sep"] == pytest.approx(_exact_d_r_sep(skew), abs=5e-4)
    if key == "limited":
        assert not state["available"] and state["quality"]["applicability"] == "geometric_only"
        assert not any(p.active for p in state["representation"].x_points)
    else:
        assert state["available"] and state["quality"]["applicability"] == "diverted_balance"
    sign = {"lower_single_null": -1, "near_double_null": -1, "upper_single_null": 1}.get(key)
    if sign is not None:
        assert np.sign(state["d_r_sep"]) == sign


def test_the_classification_draws_the_process_x_points_and_no_others(diagrams):
    d = diagrams["divertor_topology_classification"]
    for key, _, skew, tip in dt.CLASSIFICATION_CASES:
        rep = dt.balance_state(skew, limiter_tip=tip)["representation"]
        markers = [i for i in d.scene.items if isinstance(i, Marker) and i.kind == "x" and i.role.endswith(f":{key}")]
        assert len(markers) == len(rep.x_points)
        active = [m for m in markers if m.role.startswith("active_x_point")]
        assert len(active) == sum(p.active for p in rep.x_points)
        assert d.model[key]["topology"] == rep.topology.value
    # a balanced DN has two active X-points; a limited plasma none
    assert sum(1 for i in d.scene.items if i.role == "active_x_point:double_null") == 2
    assert not [i for i in d.scene.items if i.role == "active_x_point:limited"]
    assert [i for i in d.scene.items if i.role == "external_separatrix:limited"]


@pytest.mark.parametrize("skew", [0.08, 0.0, -0.08])
def test_drawn_separatrices_lie_on_their_saddle_flux_and_reach_the_x_point(skew):
    state = dt.balance_state(skew)
    exact = _exact_saddle_fluxes(skew)
    for branch, point in dt._x_points(state).items():
        assert point.psi == pytest.approx(exact[0 if branch == "lower" else 1], abs=1e-6)
        runs = dt._separatrix_runs(state, point)
        assert runs
        for run in runs:
            assert np.allclose(dt.family_flux(run[:, 0], run[:, 1], skew), point.psi, atol=2e-4)
        assert min(float(np.min(np.hypot(run[:, 0] - point.r, run[:, 1] - point.z))) for run in runs) < 0.01


@pytest.mark.parametrize("key, title, skew, tip", dt.CLASSIFICATION_CASES, ids=[c[0] for c in dt.CLASSIFICATION_CASES])
def test_the_equilibrium_hands_the_process_a_closed_lcfs_inside_the_wall(key, title, skew, tip):
    from matplotlib.path import Path as MPath

    state = dt.balance_state(skew, limiter_tip=tip)
    eq = state["equilibrium"]
    lcfs = np.stack([eq.lcfs.r, eq.lcfs.z], -1)
    assert np.allclose(lcfs[0], lcfs[-1])
    assert MPath(state["wall"]).contains_points(lcfs, radius=1e-6).all() or MPath(state["wall"]).contains_points(
        lcfs, radius=-1e-6).all()
    evidence = state["representation"].topology_evidence
    if tip is None:
        # a diverted boundary nowhere near the wall: the topology rests on the saddles, not on a wall contact
        assert lcfs[:, 1].min() > -0.47 and lcfs[:, 1].max() < 0.47
        assert evidence["wall_contact_m"] > 2 * evidence["wall_contact_tolerance_m"]
    else:
        assert evidence["wall_contact_m"] <= evidence["wall_contact_tolerance_m"]


def test_the_lcfs_of_a_diverted_panel_is_the_closed_primary_separatrix(diagrams):
    d = diagrams["divertor_topology_classification"]
    for key, _, skew, tip in dt.CLASSIFICATION_CASES:
        loops = [i for i in d.scene.items if i.role == f"lcfs:{key}"]
        assert len(loops) == 1 and loops[0].closed
    # the LSN LCFS, back in metres, sits on the lower saddle flux and spans the confined height range
    state = dt.balance_state(0.08)
    bounds = dt._core_bounds(state)
    (lower,) = [p for p in state["representation"].x_points if p.active]
    pieces = [piece for run in dt._separatrix_runs(state, lower) for piece, core in dt._split_at_x_points(run, bounds)
              if core]
    assert len(pieces) == 1
    loop = pieces[0]
    assert np.allclose(dt.family_flux(loop[:, 0], loop[:, 1], 0.08), _exact_saddle_fluxes(0.08)[0], atol=2e-4)
    assert np.hypot(*(loop[0] - loop[-1])) < 0.02  # it closes through the X-point
    assert loop[:, 1].min() == pytest.approx(lower.z, abs=0.01)


# -- secondary separatrix balance ---------------------------------------------------------


def test_balance_values_match_the_closed_form_and_the_sign_convention(diagrams):
    d = diagrams["secondary_separatrix_balance"]
    for key, skew in (("lower_biased", 0.08), ("balanced", 0.0), ("upper_biased", -0.08)):
        m = d.model[key]
        psi_l, psi_u = _exact_saddle_fluxes(skew)
        assert m["r_lower"] == pytest.approx(1.0 + np.sqrt(psi_l), abs=3e-4)
        assert m["r_upper"] == pytest.approx(1.0 + np.sqrt(psi_u), abs=3e-4)
        assert m["d_r_sep"] == pytest.approx(m["d_r_sep_formula"], abs=1e-12)
        assert m["d_r_sep_formula"] == pytest.approx(float(separatrix_radial_separation(m["r_lower"], m["r_upper"])))
    assert d.model["lower_biased"]["d_r_sep"] < 0 < d.model["upper_biased"]["d_r_sep"]
    assert abs(d.model["balanced"]["d_r_sep"]) <= d.model["balanced"]["resolution"]
    assert d.model["lower_biased"]["d_r_sep"] == pytest.approx(-d.model["upper_biased"]["d_r_sep"], abs=1e-6)
    # the flux imbalance has the sign of dR_sep in this convention (psi rises outward)
    assert d.model["lower_biased"]["flux_imbalance"] < 0 < d.model["upper_biased"]["flux_imbalance"]


def test_the_midplane_markers_sit_at_the_process_intersections(diagrams):
    d = diagrams["secondary_separatrix_balance"]
    m = d.model["lower_biased"]
    markers = [i for i in d.scene.items if isinstance(i, Marker)
               and i.role in ("r_lower:lower_biased", "r_upper:lower_biased")]
    # the R-Z view (4.6 cm/m) is the top row
    top = max(i.at[1] for i in markers)
    rz = [i for i in markers if abs(i.at[1] - top) < 1e-9]
    assert len(rz) == 2
    lower = next(i for i in rz if i.role.startswith("r_lower"))
    upper = next(i for i in rz if i.role.startswith("r_upper"))
    assert (lower.at[0] - upper.at[0]) / 4.6 == pytest.approx(_exact_d_r_sep(0.08), abs=3e-4)
    assert m["d_r_sep"] < 0


# -- SOL connectivity --------------------------------------------------------------------


def _classes(skew):
    return {t["class"] for t in dt.connectivity_traces(dt.balance_state(skew))}


def test_connectivity_changes_across_the_secondary_separatrix():
    common = {"closed", "common_outboard", "common_inboard", "private_lower", "private_upper"}
    assert _classes(0.15) == common | {"primary_lower"}
    assert _classes(0.0) == common
    assert _classes(-0.15) == common | {"primary_upper"}


def test_a_traced_line_stays_on_its_flux_surface_and_ends_on_the_wall():
    from matplotlib.path import Path as MPath

    state = dt.balance_state(0.15)
    wall = MPath(state["wall"])
    for trace in dt.connectivity_traces(state):
        path = trace["path"]
        assert np.allclose(dt.family_flux(path[:, 0], path[:, 1], 0.15), trace["level"], atol=2e-4)
        if trace["closed"]:
            continue
        for end in trace["ends"]:
            # on the wall: inside a slightly grown wall, not inside a slightly shrunk one
            assert wall.contains_point(end, radius=1e-6) or wall.contains_point(end, radius=-1e-6)
            centre = np.array([1.0, 0.0])
            assert not wall.contains_point(centre + (np.asarray(end) - centre) * 1.001)
        assert all(name.endswith("target") for name in trace["targets"])


def test_the_primary_sol_reaches_only_the_primary_divertor():
    state = dt.balance_state(0.15)
    (between,) = [t for t in dt.connectivity_traces(state) if t["region"] == "between"]
    assert sorted(between["targets"]) == ["lower inner target", "lower outer target"]
    assert between["crosses_axis_height"]
    # its flux lies between the two saddle fluxes
    psi_l, psi_u = _exact_saddle_fluxes(0.15)
    assert psi_l < between["level"] < psi_u


def test_the_connectivity_diagram_draws_every_traced_line(diagrams):
    d = diagrams["sol_magnetic_connectivity"]
    for key in ("lower_biased", "balanced", "upper_biased"):
        drawn = [i for i in d.scene.items if isinstance(i, Polyline) and i.role.startswith("field_line:")
                 and i.role.endswith(f":{key}")]
        open_lines = [c for c in d.model[key]["connections"] if c[1] != "closed"]
        assert len(drawn) == len(open_lines)


def _trace(targets, z):
    return {"closed": False, "targets": targets, "crosses_axis_height": bool(np.any(z > 0) and np.any(z < 0))}


def test_connection_class_needs_an_inner_and_an_outer_target_for_one_divertor():
    assert dt.connection_class(_trace(("lower inner target", "lower outer target"), np.array([-0.5, 0.3]))) \
        == "primary_lower"
    assert dt.connection_class(_trace(("lower inner target", "lower outer target"), np.array([-0.5, -0.4]))) \
        == "private_lower"
    assert dt.connection_class(_trace(("lower outer target", "lower outer target"), np.array([-0.5, 0.3]))) \
        == "other"
    assert dt.connection_class(_trace(("lower wall", "upper outer target"), np.array([-0.5, 0.3]))) == "other"
    assert "other" in dt._CLASS_TEXT


def _fill_roles(d, key=None):
    return {i.role.split(":")[1] for i in d.scene.items if isinstance(i, Polyline) and i.role.startswith("region:")
            and (key is None or i.role.endswith(f":{key}"))}


def test_the_connectivity_regions_are_shaded_and_named(diagrams):
    d = diagrams["sol_magnetic_connectivity"]
    assert _fill_roles(d, "lower_biased") == {"common_sol", "primary_sol_lower", "core", "private_lower",
                                               "private_upper"}
    assert _fill_roles(d, "balanced") == {"common_sol", "core", "private_lower", "private_upper"}
    assert _fill_roles(d, "upper_biased") == {"common_sol", "primary_sol_upper", "core", "private_lower",
                                               "private_upper"}
    pointers = {i.role for i in d.scene.items if isinstance(i, Label) and i.role.startswith("pointer:")}
    assert "pointer:primary_sol:lower_biased" in pointers and "pointer:primary_sol:balanced" not in pointers


def test_a_primary_sol_fill_holds_only_primary_sol_flux():
    """Every vertex of the shaded primary SOL lies between the two X-point fluxes, outside the core."""
    state = dt.balance_state(0.15)
    psi_l, psi_u = _exact_saddle_fluxes(0.15)
    fills = [i for i in dt.region_fills(state, lambda p: np.asarray(p, float)) if i.role == "region:primary_sol_lower"]
    assert fills
    points = np.array([p for f in fills for p in f.points])
    psi = dt.family_flux(points[:, 0], points[:, 1], 0.15)
    inside = dt._wall_path(state["wall"]).contains_points(points, radius=-1e-3)
    assert np.all(psi[inside] <= psi_u + 2e-3)


@pytest.mark.parametrize("name, topology, fills", [
    ("limited_plasma_topology", "limited", {"limiter_sol", "core"}),
    ("lower_single_null_topology", "lower_single_null",
     {"common_sol", "primary_sol_lower", "core", "private_lower", "private_upper"}),
    ("double_null_topology", "double_null", {"common_sol", "core", "private_lower", "private_upper"}),
])
def test_single_topology_diagrams(name, topology, fills, diagrams):
    d = diagrams[name]
    assert d.model["topology"] == topology
    assert _fill_roles(d) == fills
    assert len([i for i in d.scene.items if isinstance(i, Label) and i.role.startswith("pointer:")]) >= 7
    if topology == "limited":
        assert d.model["applicability"] == "geometric_only" and not d.model["strike_points"]


@pytest.mark.parametrize("name, skew, names", [
    ("lower_single_null_topology", 0.12, {"lower inner", "lower outer"}),
    ("double_null_topology", 0.0, {"lower inner", "lower outer", "upper inner", "upper outer"}),
])
def test_strike_points_are_where_the_primary_separatrix_meets_the_targets(name, skew, names, diagrams):
    strikes = diagrams[name].model["strike_points"]
    assert set(strikes) == names
    psi_l, psi_u = _exact_saddle_fluxes(skew)
    for key, (r, z) in strikes.items():
        assert dt.family_flux(r, z, skew) == pytest.approx(min(psi_l, psi_u), abs=2e-4)
        assert abs(z) >= 0.45 and np.sign(z) == (-1 if key.startswith("lower") else 1)
        x_r = 1.0
        assert (r < x_r) == key.endswith("inner")
    if name == "double_null_topology":
        # on the floor and roof: R = 1 +- sqrt(psi_X - Z^2 + Z^4/a^2), the same four times
        expected = np.sqrt(psi_l - 0.36 + 0.6**4 / dt.FAMILY_A**2)
        for key, (r, z) in strikes.items():
            assert abs(r - 1.0) == pytest.approx(expected, abs=2e-4)


def test_a_seed_outside_the_wall_is_refused():
    with pytest.raises(ValueError, match="outside the wall"):
        dt.trace_field_line(dt.balance_state(0.0), (1.49, 0.0))


# -- transport scale -------------------------------------------------------------------------


def test_the_transport_scale_uses_the_formulas_and_says_balance_is_not_power_balance(diagrams):
    d = diagrams["separatrix_balance_transport_scale"]
    m = d.model
    for v, chi, f in zip(m["d_r_sep"], m["chi"], m["factor"]):
        assert chi == pytest.approx(float(separatrix_imbalance_to_heat_flux_width(v, m["lambda_q"])))
        assert f == pytest.approx(float(secondary_separatrix_exponential_factor(v, m["lambda_q"])))
        assert f == pytest.approx(np.exp(-abs(v) / m["lambda_q"]))
    # the explanation is the builder's docstring, not text on the figure
    doc = vaft.diagram.separatrix_balance_transport_scale.__doc__
    assert "not thermal or power balance" in doc and "not a transport prediction" in doc
    assert "not the fraction of power" in doc and "divides between" in doc
    # the equations are the formulas' own catalog entries, not restatements
    from vaft.diagram._equations import formula_equation

    (box,) = [i for i in d.scene.items if isinstance(i, Label) and i.role == "equations"]
    assert formula_equation(separatrix_imbalance_to_heat_flux_width) in box.text
    assert formula_equation(secondary_separatrix_exponential_factor) in box.text



# -- the procedure ---------------------------------------------------------------------------


def test_the_procedure_reports_the_near_double_null_values(diagrams):
    d = diagrams["separatrix_balance_calculation"]
    m = d.model
    assert m["topology"] == "lower_single_null"
    assert m["roles"] == {"lower": "primary", "upper": "secondary"}
    assert m["d_r_sep"] == pytest.approx(_exact_d_r_sep(0.01), abs=5e-4)
    # the process's first-order estimate, against the exact separation (psi = (R-1)^2 on the midplane)
    assert m["linearized"] == pytest.approx(_exact_d_r_sep(0.01), rel=0.02)
    roles = {i.role for i in d.scene.items}
    assert {"flux_map", "critical_points", "selection", "intersections", "result", "available", "resolution_limited",
            "geometric_only", "unavailable"} <= roles


# -- rendering contract ---------------------------------------------------------------------


@pytest.mark.parametrize("name", BUILDERS)
def test_no_grey_note_text(name, diagrams):
    """Explanations live in black legends, labels and boxes, not in a grey footnote."""
    assert not [i for i in diagrams[name].scene.items if isinstance(i, Label) and i.style.split(",")[0] == "note"]
    assert not [i for i in diagrams[name].scene.items if isinstance(i, Label) and "text=black!60" in i.style]


def test_every_procedure_box_carries_a_picture(diagrams):
    d = diagrams["separatrix_balance_calculation"]
    roles = {i.role for i in d.scene.items}
    for step in ("flux_map", "critical_points", "selection", "intersections", "result", "available",
                 "resolution_limited", "geometric_only", "unavailable"):
        assert f"glyph:{step}" in roles
    # the critical-point glyph marks exactly the saddles the process found
    saddles = [i for i in d.scene.items if isinstance(i, Marker) and i.kind == "x" and i.role == "glyph:critical_points"]
    assert len(saddles) == len(dt.balance_state(0.01)["representation"].x_points)
    assert d.model["outcome_examples"]["resolution_limited"][0] == "double_null"
    assert d.model["outcome_examples"]["geometric_only"] == ("limited", "geometric_only")


@pytest.mark.parametrize("name", BUILDERS)
def test_labels_false_drops_the_annotations(name):
    bare = getattr(vaft.diagram, name)(labels=False)
    texts = [i for i in bare.scene.items if isinstance(i, Label) and i.text
             and not i.role.startswith(("axes", "ticks", "legend"))]
    assert not [i for i in texts if i.role.split(":")[0] in ("title", "caption", "note", "row", "d_r_sep")]
    with pytest.raises(ValueError):
        getattr(vaft.diagram, name)(labels="yes")


@pytest.mark.parametrize("name", ["divertor_topology_classification", "secondary_separatrix_balance",
                                  "sol_magnetic_connectivity", "separatrix_balance_calculation",
                                  "limited_plasma_topology", "lower_single_null_topology", "double_null_topology"])
def test_rendering_is_stable_under_last_bit_noise_in_psi(name, monkeypatch, diagrams):
    base = diagrams[name].tikz
    original = dt.analytic_balance_equilibrium
    rng = np.random.default_rng(7)

    def noisy(*args, **kwargs):
        eq = original(*args, **kwargs)
        psi = np.asarray(eq.psi)
        return dataclasses.replace(eq, psi=psi * (1.0 + 1e-15 * rng.standard_normal(psi.shape)))

    monkeypatch.setattr(dt, "analytic_balance_equilibrium", noisy)
    assert getattr(vaft.diagram, name)().tikz == base


def test_the_limited_caption_never_claims_a_diverted_balance(diagrams):
    d = diagrams["divertor_topology_classification"]
    (caption,) = [i for i in d.scene.items if i.role == "caption:limited"]
    assert "geometric only" in caption.text and "diverted" not in caption.text
    for key in ("lower_single_null", "near_double_null", "upper_single_null"):
        (caption,) = [i for i in d.scene.items if i.role == f"caption:{key}"]
        assert "diverted balance" in caption.text


def test_a_double_null_the_grid_cannot_resolve_says_so():
    state = dt.balance_state(0.001)
    assert state["topology"] == "double_null" and state["d_r_sep"] < 0
    assert dt._topology_text(state) == "DN within resolution"
    d = vaft.diagram.secondary_separatrix_balance(bias=0.001)
    (caption,) = [i for i in d.scene.items if i.role == "caption:lower_biased"]
    assert "within resolution" in caption.text and "balanced" not in caption.text


def test_a_large_bias_keeps_every_marker_on_its_axes():
    d = vaft.diagram.secondary_separatrix_balance(bias=0.2)
    for key in ("lower_biased", "upper_biased"):
        axes = [i for i in d.scene.items if i.role in (f"flux_axis:{key}", f"axes:{key}") and hasattr(i, "start")]
        x_min = min(min(a.start[0], a.end[0]) for a in axes)
        x_max = max(max(a.start[0], a.end[0]) for a in axes)
        markers = [i for i in d.scene.items if isinstance(i, Marker) and i.role.endswith(f":{key}")
                   and i.role.startswith(("psi_", "r_"))]
        assert markers and all(x_min - 1e-9 <= m.at[0] <= x_max + 1e-9 for m in markers)
        lines = [i for i in d.scene.items if isinstance(i, Polyline) and i.role.startswith("psi_")
                 and i.role.endswith(f":{key}")]
        y_top = max(max(a.start[1], a.end[1]) for a in axes if a.role.startswith("axes"))
        assert all(p[1] <= y_top for line in lines for p in line.points)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="skew"):
        dt.analytic_balance_equilibrium(0.5)
    with pytest.raises(ValueError, match="skew"):
        dt.analytic_balance_equilibrium(True)
    with pytest.raises(ValueError, match="limiter_tip"):
        dt.analytic_balance_equilibrium(0.08, limiter_tip=0.62)
    with pytest.raises(ValueError, match="limiter_tip"):
        dt.analytic_balance_equilibrium(0.08, limiter_tip=1.2)  # through the axis
    for bad in ("0.003", 0.003, (True,), (0.5,)):
        with pytest.raises(ValueError, match="d_r_sep"):
            vaft.diagram.separatrix_balance_transport_scale(d_r_sep=bad)
    with pytest.raises(ValueError, match="lambda_q"):
        vaft.diagram.separatrix_balance_transport_scale(lambda_q=True)
    with pytest.raises(ValueError, match="bias"):
        vaft.diagram.secondary_separatrix_balance(bias=-0.05)
    with pytest.raises(ValueError, match="lambda_q"):
        vaft.diagram.separatrix_balance_transport_scale(lambda_q=0.0)
    with pytest.raises(ValueError, match="d_r_sep"):
        vaft.diagram.separatrix_balance_transport_scale(d_r_sep=(0.0,))


def test_builders_are_registered_and_documented():
    from vaft.diagram import build

    text = (ROOT / "docs" / "_guide" / "Diagrams.md").read_text(encoding="utf-8")
    for name in BUILDERS:
        assert name in vaft.diagram.__all__
        assert build.CANONICAL[f"{name}.svg"] == (name, {})
        assert f"`{name}`" in text and f"/assets/diagrams/{name}.svg" in text
