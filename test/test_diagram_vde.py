"""VDE diagrams (#1042): scraping lowers the edge q, the branches and timescales are the formulas'."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _vde as vd
from vaft.diagram._scene import Label
from vaft.formula.geometry import cylindrical_poloidal_field, cylindrical_safety_factor_from_r_B


def test_scraping_moves_the_edge_inward_to_lower_q():
    frames = vd.hot_vde_frames()["frames"]
    rho = [f["rho"] for f in frames]
    q = [f["q_edge"] for f in frames]
    a = [f["a"] for f in frames]
    assert rho == sorted(rho, reverse=True) and rho[0] > rho[-1]
    assert q == sorted(q, reverse=True) and q[0] > 1.5 * q[-1]
    # the cylindrical estimate at fixed I_p falls as a^2
    eq = vd.hot_vde_frames()["geometry"].equilibrium
    for f in frames:
        expected = cylindrical_safety_factor_from_r_B(f["a"], cylindrical_poloidal_field(f["a"], eq.ip), eq.bt0, eq.r0)
        assert f["q_cyl"] == pytest.approx(expected)
    assert frames[-1]["q_cyl"] / frames[0]["q_cyl"] == pytest.approx((a[-1] / a[0]) ** 2)


def test_the_kept_surface_fits_inside_the_wall():
    from matplotlib.path import Path

    data = vd.hot_vde_frames()
    g = data["geometry"]
    wall = Path(np.stack([g.equilibrium.limiter.r, g.equilibrium.limiter.z], -1))
    for f in data["frames"]:
        s = g.surface(f["rho"])
        assert wall.contains_points(np.stack([s.R, s.Z - f["shift"]], -1)).all()


def test_the_cold_vde_branches_are_a_pitchfork_at_the_critical_current():
    I = np.linspace(0, 2, 401)
    off = vd.cold_vde_branches(I, 1.0, 1.0)
    assert np.all(off[I >= 1.0] == 0.0)
    np.testing.assert_allclose(off[I < 1.0] ** 2, 1.0 - I[I < 1.0])


def test_the_timescales_come_from_the_formulas():
    t = vd.vde_timescales_table()
    assert t["tau_wall_m1"] == pytest.approx(t["tau_w"] / 2)
    assert t["tau_CQ_20eV"] / t["tau_CQ_5eV"] == pytest.approx(4.0**1.5)  # L/R scales as T_e^(3/2)
    assert t["tau_A"] < 1e-5 < t["tau_CQ_5eV"]


@pytest.mark.parametrize("name", ["hot_vde_sequence", "cold_vde_bifurcation", "plasma_wall_halo_current",
                                  "vde_timescales"])
def test_every_vde_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    assert sum(isinstance(i, Label) for i in fn(labels=False).scene.items) < sum(
        isinstance(i, Label) for i in fn().scene.items)
