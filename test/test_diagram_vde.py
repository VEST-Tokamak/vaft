"""VDE diagrams (#1042): scraping lowers the edge q, the branches and timescales are the formulas'."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _vde as vd
from vaft.diagram._scene import Label
from vaft.formula.geometry import cylindrical_poloidal_field, cylindrical_safety_factor_from_r_B


def test_scraping_moves_the_edge_inward_to_lower_q():
    data = vd.hot_vde_frames()
    frames, g = data["frames"], data["geometry"]
    rho = [f["rho"] for f in frames]
    q = [f["q_edge"] for f in frames]
    assert rho == sorted(rho, reverse=True) and rho[0] > rho[-1]
    assert q == sorted(q, reverse=True) and q[0] > 1.5 * q[-1]
    # the labelled q is the frozen profile's on the new edge surface
    for f in frames:
        assert f["q_edge"] == pytest.approx(float(g.q(f["rho"])))
    # the cylindrical estimate at fixed I_p is a ratio: it falls as a^2
    a = [f["a"] for f in frames]
    assert frames[-1]["q_cyl"] / frames[0]["q_cyl"] == pytest.approx((a[-1] / a[0]) ** 2)


def test_the_kept_surface_fits_inside_the_wall_and_the_next_one_would_not():
    data = vd.hot_vde_frames()
    g = data["geometry"]
    wall = np.stack([g.equilibrium.limiter.r, g.equilibrium.limiter.z], -1)
    for f in data["frames"]:
        s = g.surface(f["rho"])
        pts = np.stack([s.R, s.Z - f["shift"]], -1)
        assert vd._inside(wall, pts).all()
        if f["rho"] < 0.97:
            s2 = g.surface(round(f["rho"] + 0.02, 4))
            pts2 = np.stack([s2.R, s2.Z - f["shift"]], -1)
            assert not (vd._inside(wall, pts2).all() and vd._distance_to_polygon(wall, pts2).min() > 5e-4)


def test_the_halo_force_pushes_the_wall_away_from_the_plasma():
    h = vd.halo_circuit()
    # independent of the code's own choice: J along the wall path, B along +phi (into the page);
    # (R, phi, Z) right-handed: J x B = B_phi (J_R Z_hat - J_Z R_hat)
    J = h["wall_path"][-1] - h["wall_path"][0]
    F = vd.B_PHI_SIGN * np.array([-J[1], J[0]])
    assert np.dot(F, h["normal_out"]) > 0
    np.testing.assert_allclose(F / np.linalg.norm(F), h["force"], atol=1e-12)
    # the circuit touches the wall and closes through the plasma edge region
    wall = h["wall"]
    assert vd._distance_to_polygon(wall, h["wall_path"]).max() < 1e-9
    assert np.linalg.norm(h["circuit"][0] - h["circuit"][-1]) < 0.05


def test_the_cold_vde_branches_are_a_pitchfork_at_the_critical_current():
    I = np.linspace(0, 2, 401)
    off = vd.cold_vde_branches(I, 1.0, 1.0)
    assert np.all(off[I >= 1.0] == 0.0)
    np.testing.assert_allclose(off[I < 1.0] ** 2, 1.0 - I[I < 1.0])
    # the drawn trajectory ends at the wall, not beyond it
    d = vaft.diagram.cold_vde_bifurcation()
    traj = d.scene.role("trajectory")[0]
    assert min(p[1] for p in traj.points) >= -1e-9 + min(p[1] for p in d.scene.role("wall")[0].points)


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
