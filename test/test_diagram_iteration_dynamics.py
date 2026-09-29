"""Iteration behaviour, branch bifurcation and branch selection (#1093): computed, not sketched."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _iteration_dynamics as it
from vaft.diagram._scene import Arrow, Label, Marker


def _period(orbit, tol=1e-6):
    tail = orbit[-8:]
    for p in (1, 2, 4):
        if np.all(np.abs(tail[p:] - tail[:-p]) < tol):
            return p
    return None


def test_the_four_behaviours_are_what_their_titles_say():
    # periods are checked on long orbits of the same maps; the panels show the first N_ITERATIONS
    orbits = {name: it.iterate(name, 400) for name in it.ITERATION_MAPS}
    assert _period(orbits["fixed_point"]) == 1
    assert orbits["fixed_point"][-1] == pytest.approx(1.0 - 1.0 / 2.8, abs=1e-6)
    assert _period(orbits["two_cycle"]) == 2
    assert _period(orbits["limit_cycle"]) == 4
    diverging = np.abs(it.iterate("divergence") - 0.5)
    assert np.all(np.diff(diverging) > 0) and diverging[-1] > 0.5
    # the drawn limit cycle is already on its period-4 orbit by the end of the panel
    np.testing.assert_allclose(it.iterate("limit_cycle")[-4:], orbits["limit_cycle"][-4:], atol=0.01)


def test_every_iteration_panel_shares_one_format():
    diagram = vaft.diagram.iteration_behavior()
    for name in it.ITERATION_MAPS:
        assert diagram.scene.role(f"{name}:orbit") and diagram.scene.role(f"{name}:fixed_point")
        assert len([m for m in diagram.scene.role(f"{name}:iterate") if isinstance(m, Marker)]) > 10
    # the escaping orbit is marked, the bounded ones are not
    assert diagram.scene.role("divergence:escape") and not diagram.scene.role("two_cycle:escape")
    assert sorted(l.text for l in diagram.scene.items if isinstance(l, Label) and l.text == "$k$") == ["$k$"] * 4


def test_branches_follow_the_normal_form_and_stability():
    chart = vaft.diagram.branch_bifurcation().model
    for name in ("stable_lower", "stable_upper", "unstable"):
        lam, x = chart.curves[name].T
        np.testing.assert_allclose(lam, x**3 - x, atol=1e-12)
        slope = 1.0 - 3.0 * x**2  # d(lambda + x - x^3)/dx
        assert np.all(slope < 0) if name.startswith("stable") else np.all(slope > 0)
    # folds: where the equilibrium curve turns, d lambda / dx = 0
    for point in ("fold_upper", "fold_lower"):
        lam, x = chart.points[point]
        assert 3 * x**2 - 1 == pytest.approx(0.0, abs=1e-12)
        assert lam == pytest.approx(x**3 - x)
    # each jump lands on the other stable branch at the fold's lambda
    for fold, to in (("fold_lower", "jump_up_to"), ("fold_upper", "jump_down_to")):
        assert chart.points[to][0] == chart.points[fold][0]
        lam, x = chart.points[to]
        assert x**3 - x == pytest.approx(lam) and abs(x) > it.FOLD_X


def test_hysteresis_directions_are_drawn_on_the_right_folds_and_branches():
    diagram = vaft.diagram.branch_bifurcation()
    scene, chart = diagram.scene, diagram.model
    up = [a for a in scene.role("jump_up") if isinstance(a, Arrow)][0]
    down = [a for a in scene.role("jump_down") if isinstance(a, Arrow)][0]
    assert up.end[1] > up.start[1] and down.end[1] < down.start[1]
    # the upward jump leaves the fold at the larger lambda
    assert up.start[0] > down.start[0]
    assert up.start == pytest.approx(tuple(chart.to_cm(np.array(chart.points["fold_lower"]))))
    increasing = [a for a in scene.role("increasing") if isinstance(a, Arrow)][0]
    decreasing = [a for a in scene.role("decreasing") if isinstance(a, Arrow)][0]
    assert increasing.end[0] > increasing.start[0] and decreasing.end[0] < decreasing.start[0]
    # increasing runs on the lower branch, decreasing on the upper, both inside the bistable range
    zero = chart.to_cm(np.array([0.0, 0.0]))[1]
    assert increasing.start[1] < zero < decreasing.start[1]
    lo, hi = (chart.to_cm(np.array([s * it.FOLD_LAMBDA, 0.0]))[0] for s in (-1, 1))
    for arrow in (increasing, decreasing):
        assert lo < arrow.start[0] < hi and lo < arrow.end[0] < hi


def test_stable_and_unstable_differ_by_line_style():
    scene = vaft.diagram.branch_bifurcation().scene
    style = {p.role: p.style for p in scene.items if hasattr(p, "points")}
    assert style["stable_lower"] == style["stable_upper"] == "boundary"
    assert style["unstable"] == "approx"
    basin = {p.role: p.style for p in vaft.diagram.basin_of_attraction().scene.items if hasattr(p, "points")}
    assert basin["basin_boundary"] == "approx" and basin["branch_A"] == basin["branch_B"] == "boundary"


def test_the_basin_boundary_decides_the_branch():
    chart = vaft.diagram.basin_of_attraction().model
    for x0 in it.INITIAL_CONDITIONS:
        t, x = chart.curves[f"trajectory {x0:+.2f}"].T
        # x' = x - x^3 conserves x^2 / (1 - x^2) e^{-2t}: an exact check independent of the closed form
        np.testing.assert_allclose(x**2 / (1.0 - x**2) * np.exp(-2.0 * t), x0**2 / (1.0 - x0**2), rtol=1e-9)
        assert x[-1] == pytest.approx(np.sign(x0), abs=0.02)
    assert chart.curves["basin_boundary"][0, 1] == 0.0


def test_numerical_cycling_is_drawn_apart_from_branch_structure():
    diagram = vaft.diagram.grid_induced_two_cycle()
    model = diagram.model
    a, b = np.asarray(model["cell_A"]), np.asarray(model["cell_B"])
    from_a, from_b = np.asarray(model["optimum_from_A"]), np.asarray(model["optimum_from_B"])
    assert np.linalg.norm(b - a) == pytest.approx(model["grid_spacing"])  # the hop is one grid cell
    # the snap map has no fixed point: solved from A the optimum is nearer B, and from B nearer A
    assert np.linalg.norm(from_a - b) < np.linalg.norm(from_a - a)
    assert np.linalg.norm(from_b - a) < np.linalg.norm(from_b - b)
    hops = [h for h in diagram.scene.role("hop") if isinstance(h, Arrow)]
    assert len(hops) == 2
    ends = sorted(tuple(np.round(h.end, 1)) for h in hops)
    assert ends[0][0] < ends[1][0]  # one hop lands at A, the other at B
    # the index alternates while the residual stays flat
    index = [m.at[1] for m in diagram.scene.role("index") if isinstance(m, Marker)]
    assert len(set(np.round(index, 6))) == 2 and all(index[i] != index[i + 1] for i in range(len(index) - 1))
    residual = [p for p in diagram.scene.role("residual") if hasattr(p, "points")][0]
    assert len({y for _, y in residual.points}) == 1
    # no bifurcation curve or basin here: the artifact is not a branch
    assert not diagram.scene.role("unstable") and not diagram.scene.role("basin_boundary")


def test_branch_selection_composes_the_two_panels_unchanged():
    composed = vaft.diagram.branch_selection().scene
    basin = vaft.diagram.basin_of_attraction().scene
    grid = vaft.diagram.grid_induced_two_cycle().scene
    assert composed.items[: len(basin.items)] == basin.items
    assert len(composed.items) == len(basin.items) + len(grid.items) + 1
    assert composed.role("caption")


@pytest.mark.parametrize("name, source", [("branch_bifurcation", "fold_normal_form_equilibria"),
                                          ("basin_of_attraction", "relaxation")])
def test_rendering_is_stable_under_last_bit_noise(monkeypatch, name, source):
    # x**3 and exp go through libm, whose last bit can differ between platforms; the TikZ must not
    base = getattr(vaft.diagram, name)().tikz
    original = getattr(it, source)
    rng = np.random.default_rng(3)
    for _ in range(3):
        monkeypatch.setattr(it, source, lambda *a: (lambda r: r * (1 + 1e-15 * rng.standard_normal(np.shape(r))))(
            original(*a)))
        assert getattr(vaft.diagram, name)().tikz == base


def test_labels_off_and_validation():
    for build in (vaft.diagram.iteration_behavior, vaft.diagram.branch_bifurcation,
                  vaft.diagram.basin_of_attraction, vaft.diagram.grid_induced_two_cycle,
                  vaft.diagram.branch_selection):
        labelled = [i for i in build(labels=False).scene.items if isinstance(i, Label)]
        assert not [l for l in labelled if l.role not in ("axes", "ticks")]
        with pytest.raises(ValueError):
            build(labels="yes")
