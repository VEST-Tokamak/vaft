"""Equilibrium-aware phenomena (#1209): prescribed physics drawn on an equilibrium's own surfaces."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _equilibrium_geometry as eg
from vaft.diagram import _sawtooth as st
from vaft.diagram._mhd_mode import SURFACES, envelope
from vaft.diagram._scene import Label
from vaft.formula.equilibrium import flux_perturbation_from_normal_displacement
from vaft.formula.stability import kadomtsev_mixing_radius


def test_flux_freezing_is_minus_xi_dot_grad_psi():
    assert flux_perturbation_from_normal_displacement(0.01, 2.0) == pytest.approx(-0.02)
    with pytest.raises(ValueError):
        flux_perturbation_from_normal_displacement(0.01, -1.0)


def test_the_mixing_radius_is_sqrt2_r1_for_a_parabolic_inverse_q():
    r = np.linspace(0.0, 1.0, 4001)
    q = 1.0 / (1.0 + 0.25 * (1.0 - (r / 0.4) ** 2))
    assert kadomtsev_mixing_radius(r, q) == pytest.approx(math.sqrt(2) * 0.4, rel=1e-5)
    with pytest.raises(ValueError, match="no q = 1"):
        kadomtsev_mixing_radius(r, 1.2 + r)
    with pytest.raises(ValueError):
        kadomtsev_mixing_radius(r[::-1], q)


def test_the_adapter_normals_are_unit_and_outward():
    g = eg.default_equilibrium()
    assert g.q0 < 1.0 < g.q_profile[-1]
    s = g.surface(0.5)
    np.testing.assert_allclose(np.hypot(s.normal_R, s.normal_Z), 1.0, atol=1e-9)
    # a step along the normal raises psi_N
    step = 1e-3
    assert np.all(g.sfl.psi_norm(s.R + step * s.normal_R, s.Z + step * s.normal_Z) > 0.25)
    # q crosses one where rho_at_q says
    assert float(g.q(g.rho_at_q(1.0))) == pytest.approx(1.0, abs=1e-6)
    # the (rho, theta*) map lands on the surface
    R, Z = g.to_rz(np.full(8, 0.5), np.linspace(0, 2 * math.pi, 8, endpoint=False))
    np.testing.assert_allclose(g.sfl.psi_norm(R, Z), 0.25, atol=2e-3)


def test_zero_amplitude_recovers_the_equilibrium():
    m = vaft.diagram.kink_mode(amplitude=0.0).model
    for s in m["surfaces"]:
        np.testing.assert_allclose(s["R"], s["R0"])
        np.testing.assert_allclose(s["Z"], s["Z0"])


@pytest.mark.parametrize("phase", [0.0, 0.7])
def test_the_displacement_follows_the_harmonic_phase_along_the_normal(phase):
    m = vaft.diagram.kink_mode(m=2, n=1, radial_profile="global", amplitude=0.05, phase=phase).model
    for s in m["surfaces"]:
        F = envelope("global", s["rho"], m=2, rho_s=m["rho_s"])
        expected = 0.05 * m["minor_radius"] * F * np.cos(2 * s["theta_star"] + phase)
        np.testing.assert_allclose(s["xi"], expected, atol=1e-12)
        moved = np.stack([s["R"] - s["R0"], s["Z"] - s["Z0"]], -1)
        normal = np.stack(s["normal"], -1)
        cross = moved[:, 0] * normal[:, 1] - moved[:, 1] * normal[:, 0]
        np.testing.assert_allclose(cross, 0.0, atol=1e-12)  # along the normal
        np.testing.assert_allclose(np.sum(moved * normal, -1), s["xi"], atol=1e-12)


def test_the_internal_envelope_stays_inside_the_resonant_surface():
    m = vaft.diagram.kink_mode().model
    rho_s = m["rho_s"]
    for s in m["surfaces"]:
        peak = np.abs(s["xi"]).max() / (m["amplitude"] * m["minor_radius"])
        if s["rho"] < rho_s - 0.1:
            assert peak > 0.95
        if s["rho"] > rho_s + 0.1:
            assert peak < 0.01
    assert m["axis_shift"][0] > 0.0  # the core moves towards theta* = 0 (outboard) at phase 0


def _nested(surfaces) -> bool:
    """Every point of each surface lies strictly inside the next one out."""
    from matplotlib.path import Path

    for inner, outer in zip(surfaces[:-1], surfaces[1:]):
        if not Path(np.stack(outer, -1)).contains_points(np.stack(inner, -1)).all():
            return False
    return True


@pytest.mark.parametrize("m, profile", [(1, "internal"), (2, "internal"), (1, "global"), (2, "global"), (2, "edge")])
@pytest.mark.parametrize("amplitude", [0.03, 0.06, 0.1, 0.15])
def test_no_surface_inversion_over_the_documented_amplitude_range(m, profile, amplitude):
    from vaft.diagram._mhd_mode import displacement

    g = eg.default_equilibrium()
    rho_s = g.rho_at_q(m)
    dense = np.round(np.linspace(0.1, 0.95, 35), 4)  # far denser than the drawn surfaces
    moved = []
    for rho in dense:
        d = displacement(g, rho, n=1, amplitude=amplitude, harmonics={m: 1.0}, kind=profile, rho_s=rho_s, phase=0.3)
        moved.append((d["R"], d["Z"]))
    assert _nested(moved)


def test_the_internal_envelope_needs_its_resonant_surface():
    with pytest.raises(ValueError, match="q = 5/1"):
        vaft.diagram.kink_mode(m=5, n=1)


@pytest.mark.parametrize("amplitude", [0.06, 0.15])
def test_the_precursor_keeps_nested_topology(amplitude):
    m = vaft.diagram.sawtooth(stage="precursor", amplitude=amplitude).model
    assert m["rho_1"] == pytest.approx(eg.default_equilibrium().rho_at_q(1.0))
    assert m["axis_shift"][0] > 0.0
    assert _nested([(s["R"], s["Z"]) for s in m["surfaces"]])
    assert m["rho_mix"] is None  # the precursor does not need it


def test_the_reconnection_stage_has_its_1_1_x_and_o_points():
    m = vaft.diagram.sawtooth(stage="reconnection", reconnection_fraction=0.5).model
    # the X-point is where core and outer separatrix touch, on theta* = 0; the O-point opposite
    assert m["shift"] + m["rho_c"] == pytest.approx(m["rho_o"])
    assert m["x_point"] == (pytest.approx(m["rho_o"]), 0.0)
    assert m["o_point"][0] < 0.0 and -m["rho_o"] < m["o_point"][0] < m["shift"] - m["rho_c"]
    assert m["rho_1"] <= m["rho_o"] <= m["rho_mix"]
    # island surfaces are closed crescents around the O-point, none encircles the hot core centre
    from matplotlib.path import Path

    assert m["island"]
    for line in m["island"]:
        assert not Path(line).contains_point((m["shift"], 0.0))
    assert any(Path(line).contains_point(m["o_point"]) for line in m["island"])
    # the reconnected fraction grows the island and shrinks the core
    m2 = vaft.diagram.sawtooth(stage="reconnection", reconnection_fraction=0.8).model
    assert m2["rho_c"] < m["rho_c"] and m2["rho_o"] > m["rho_o"]


def test_the_post_crash_profile_is_flat_inside_the_mixing_radius_at_conserved_int_T_rho():
    m = vaft.diagram.sawtooth(stage="post_crash").model
    rho, before, after = m["rho"], m["T_before"], m["T_after"]
    inside = rho <= m["rho_mix"]
    assert np.ptp(after[inside]) == 0.0
    np.testing.assert_allclose(after[~inside], before[~inside])
    assert np.trapezoid(after[inside] * rho[inside], rho[inside]) == pytest.approx(
        np.trapezoid(before[inside] * rho[inside], rho[inside]))
    assert m["rho_1"] < m["rho_mix"]


def test_a_sawtooth_needs_q_equal_one():
    from vaft.process.equilibrium import solovev_example

    high_q = solovev_example("limited", a_parameter=0.6)  # q > 1 everywhere
    with pytest.raises(ValueError, match="q = 1"):
        vaft.diagram.sawtooth(high_q)


def test_the_equilibrium_argument_is_checked():
    with pytest.raises(TypeError):
        vaft.diagram.kink_mode(equilibrium="39915")
    for bad in ({"radial_profile": "tearing"}, {"amplitude": 0.5}, {"m": 0}, {"harmonics": {}}):
        with pytest.raises(ValueError):
            vaft.diagram.kink_mode(**bad)
    for bad in ({"stage": "crash"}, {"reconnection_fraction": 1.0}):
        with pytest.raises(ValueError):
            vaft.diagram.sawtooth(**bad)


@pytest.mark.parametrize("fn, kwargs", [("kink_mode", {}), ("sawtooth", {"stage": "precursor"}),
                                        ("sawtooth", {"stage": "reconnection"}), ("sawtooth", {"stage": "post_crash"}),
                                        ("stochastic_layer", {}), ("separatrix_lobes", {})])
def test_every_phenomenon_diagram_is_deterministic_and_exported(fn, kwargs):
    f = getattr(vaft.diagram, fn)
    assert f(**kwargs).tikz == f(**kwargs).tikz
    assert fn in vaft.diagram.__all__
    d = f(**kwargs)
    assert d.model["classification"] in ("synthetic_parameterization", "reduced_model", "reduced_hamiltonian_model")
    assert d.scene.role("note") and not f(**kwargs, labels=False).scene.role("note")
    assert sum(isinstance(i, Label) for i in f(**kwargs, labels=False).scene.items) < sum(
        isinstance(i, Label) for i in d.scene.items)


def test_the_mixing_radius_is_taken_in_toroidal_flux():
    # exact 1/1 helical flux in poloidal flux: int_0^{psi_mix} (1 - q) dpsi_N = 0 (cylindrical r is rho_tor)
    g = eg.default_equilibrium()
    rho_mix = vaft.diagram.sawtooth(stage="post_crash").model["rho_mix"]
    psi = np.linspace(0.0, rho_mix**2, 4001)
    q = g.q(np.sqrt(psi))
    assert np.trapezoid(1.0 - q, psi) == pytest.approx(0.0, abs=2e-4)


def test_the_adapter_reads_q_in_the_records_own_unit():
    import dataclasses

    from vaft.process.equilibrium import solovev_example

    eq = solovev_example("limited", a_parameter=0.0)
    unknown = dataclasses.replace(eq.convention, cocos=None, psi_per_radian=None)
    with pytest.raises(ValueError, match="2 pi"):
        eg.EquilibriumGeometry(dataclasses.replace(eq, convention=unknown, q=None))
    full_weber = dataclasses.replace(eq.convention, cocos=None, psi_per_radian=False)
    g = eg.EquilibriumGeometry(dataclasses.replace(eq, convention=full_weber, q=None))
    assert g.q0 == pytest.approx(eg.default_equilibrium().q0, rel=1e-6)


def test_a_single_null_equilibrium_works_too():
    from vaft.process.equilibrium import solovev_example

    sn = solovev_example("single_null", a_parameter=0.0)
    for stage in ("precursor", "reconnection", "post_crash"):
        m = vaft.diagram.sawtooth(sn, stage=stage).model
        assert m["rho_1"] is not None
    assert vaft.diagram.kink_mode(sn).model["rho_s"] is not None


# --- part 2: reduced Hamiltonian topology --------------------------------------------------------------------------

from vaft.diagram import _magnetic_topology as mt  # noqa: E402


@pytest.mark.parametrize("regime", ["isolated", "touching", "overlapping"])
def test_the_overlap_parameter_is_the_requested_one(regime):
    from vaft.process.perturbation import chirikov

    m = vaft.diagram.stochastic_layer(regime=regime).model
    assert m["sigma"] == pytest.approx(mt.OVERLAPS[regime])
    assert m["sigma"] == pytest.approx(float(chirikov(m["x_resonance"], m["widths"], definition="pair")[0]))
    np.testing.assert_allclose(m["widths"], 4 * np.sqrt(m["eps"] / m["iota_prime"]))
    # the resonances sit where q = m/n
    g = eg.default_equilibrium()
    for (mm, nn), x in zip(m["resonances"], m["x_resonance"]):
        assert float(g.q(np.sqrt(x))) == pytest.approx(mm / nn, abs=1e-6)


def test_isolated_islands_confine_and_overlapping_ones_do_not():
    # a line seeded on an O-point stays within its pendulum half-width when isolated
    iso = vaft.diagram.stochastic_layer(regime="isolated").model
    x = iso["punctures"][..., 1]
    for k in range(2):
        line = x[:, mt._SEEDS + 4 * k]  # the O-point seed of resonance k
        assert np.ptp(line) < iso["widths"][k] * 0.6
    # overlapping: some line wanders across both resonant surfaces
    over = vaft.diagram.stochastic_layer(regime="overlapping").model
    xo = over["punctures"][..., 1]
    lo, hi = over["x_resonance"].min(), over["x_resonance"].max()
    assert any(line.min() < lo and line.max() > hi for line in xo.T)
    # isolated: no line crosses both
    assert not any(line.min() < iso["x_resonance"].min() and line.max() > iso["x_resonance"].max() for line in x.T)


def test_the_field_line_map_without_perturbation_keeps_psi():
    g = eg.default_equilibrium()
    p = mt.field_line_map(g, mt.RESONANCES, np.zeros(2), [0.5, 0.7], [0.0, 1.0], turns=5)
    np.testing.assert_allclose(p[..., 1], [[0.5, 0.7]] * 5, atol=1e-12)


def test_the_unperturbed_manifolds_lie_on_the_separatrix_and_have_one_strike_point():
    m = vaft.diagram.separatrix_lobes(perturbation=0.0).model
    assert m["x_point"] == pytest.approx(m["x_point_unperturbed"], abs=1e-8)
    lam_u, lam_s = sorted(np.abs(m["multipliers"]), reverse=True)
    assert lam_u > 1 > lam_s and lam_u * lam_s == pytest.approx(1.0, rel=1e-4)  # area-preserving map
    assert len(m["strike_points"]) == 1
    # and they lie on the separatrix flux
    from scipy.interpolate import RectBivariateSpline

    eq = mt.lobe_model(0.0)["equilibrium"]
    sp = RectBivariateSpline(eq.r, eq.z, np.asarray(eq.psi) / (2 * math.pi))
    dpsi = abs(eq.psi_boundary - eq.psi_axis) / (2 * math.pi)
    for branch in m["manifolds"]["unstable"] + m["manifolds"]["stable"]:
        pts = branch[np.isfinite(branch).all(1)]
        pts = pts[pts[:, 1] > m["target_z"]]
        assert np.abs(sp.ev(pts[:, 0], pts[:, 1]) - m["psi_x"]).max() < 1e-4 * dpsi


def test_the_x_point_moves_linearly_with_the_perturbation_and_its_sign_is_the_phase():
    x0 = np.array(mt.lobe_model(0.0)["x_point"])
    d1 = np.hypot(*(np.array(mt.lobe_model(0.01)["x_point"]) - x0))
    d2 = np.hypot(*(np.array(mt.lobe_model(0.02)["x_point"]) - x0))
    assert d2 / d1 == pytest.approx(2.0, rel=0.1)
    # cos(arg + pi) = -cos(arg): phase pi is the perturbation with the other sign
    a = np.array(mt.lobe_model(0.01, phase=math.pi)["x_point"])
    b = np.array(mt.lobe_model(-0.01)["x_point"])
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_the_perturbation_splits_the_manifolds_and_the_strike_point():
    m = vaft.diagram.separatrix_lobes().model
    lam_u, lam_s = sorted(np.abs(m["multipliers"]), reverse=True)
    assert lam_u * lam_s == pytest.approx(1.0, rel=1e-4)
    assert len(m["strike_points"]) >= 2 and len(m["strike_points_stable"]) >= 2  # one strike point becomes several
    for bad in ({"perturbation": True}, {"n": 0}, {"m": 2.5}):
        with pytest.raises(ValueError):
            vaft.diagram.separatrix_lobes(**bad)
    # unstable and stable manifolds cross away from the X-point: homoclinic points, hence lobes
    xp = np.array(m["x_point"])

    def pieces(branches):
        out = []
        for b in branches:
            a, c = b[:-1], b[1:]
            ok = np.isfinite(a).all(1) & np.isfinite(c).all(1) & (np.hypot(*(c - a).T) < 0.01)
            far = np.hypot(*(a - xp).T) > 0.02
            out.append(np.stack([a[ok & far], c[ok & far]], 1))
        return np.concatenate(out)

    U, S = pieces(m["manifolds"]["unstable"])[::3], pieces(m["manifolds"]["stable"])[::3]

    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    crossings = 0
    for seg in U[:: max(1, len(U) // 3000)]:
        p1, p2 = seg
        o1 = orient(p1, p2, S[:, 0]) != orient(p1, p2, S[:, 1])
        o2 = orient(S[:, 0], S[:, 1], p1) != orient(S[:, 0], S[:, 1], p2)
        crossings += int(np.sum(o1 & o2))
    assert crossings > 0


def test_separatrix_lobes_on_another_single_null_equilibrium():
    from vaft.process.equilibrium import solovev_example

    eq = solovev_example("single_null", a_parameter=0.0, major_radius=0.8, aspect_ratio=2.5)
    m = vaft.diagram.separatrix_lobes(eq).model
    # the fixed point is hyperbolic and area-preserving on this equilibrium too
    lam_u, lam_s = sorted(np.abs(m["multipliers"]), reverse=True)
    assert lam_u > 1.0 and lam_u * lam_s == pytest.approx(1.0, rel=1e-4)
    # it sits next to the equilibrium's own X-point, below the axis and near the bottom of its boundary
    assert m["x_point"][1] < float(eq.magnetic_axis[1])
    assert abs(m["x_point"][1] - float(np.min(eq.lcfs.z))) < 0.05 * float(np.ptp(eq.lcfs.z))
    assert len(m["strike_points"]) >= 2
    with pytest.raises(ValueError, match="single-null"):
        vaft.diagram.separatrix_lobes(solovev_example("limited", a_parameter=0.0))


def test_the_lobe_map_reads_psi_in_the_records_own_unit():
    """A per-radian record (COCOS 1, what every g-file loads as) traces the same field lines as its full-weber
    twin; before the map divided by 2 pi regardless, and the multipliers came out (15.9, 0.06) vs (1.55, 0.64)
    on the same null. Cold review 0.8.0 diagram-B F1."""
    import dataclasses

    from vaft.process.equilibrium import convert_cocos, solovev_example

    eq11 = solovev_example("single_null", a_parameter=0.0, major_radius=0.8, aspect_ratio=2.5)
    eq1 = convert_cocos(eq11, 1)
    assert eq11.convention.psi_per_radian is False and eq1.convention.psi_per_radian is True
    a, b = mt.lobe_model_for(eq11), mt.lobe_model_for(eq1)
    np.testing.assert_allclose(b["multipliers"], a["multipliers"], rtol=1e-6)
    np.testing.assert_allclose(b["x_point"], a["x_point"], atol=1e-9)
    for name in ("unstable", "stable"):
        np.testing.assert_allclose(mt.strike_points(b["manifolds"][name], b["target_z"]),
                                   mt.strike_points(a["manifolds"][name], a["target_z"]), atol=1e-6)
    # a record that declares neither its COCOS nor its flux unit is refused, as the q adapter refuses it
    unknown = dataclasses.replace(eq11, convention=dataclasses.replace(eq11.convention, cocos=None,
                                                                        candidates=(), psi_per_radian=None))
    with pytest.raises(ValueError, match="2 pi"):
        mt.lobe_model_for(unknown)


def test_the_kink_model_label_prints_the_coefficient_the_drawing_uses():
    """A negative or complex sideband was printed as its modulus while the displacement used the signed/complex
    value, so the figure's equation disagreed with its drawing by a phase. Cold review 0.8.0 diagram-B F3."""
    def model_text(**kw):
        (label,) = vaft.diagram.kink_mode(m=2, n=1, radial_profile="global", **kw).scene.role("model")
        return label.text

    assert "(e^{i2\\theta^*})" in model_text()
    assert "(e^{i\\theta^*} - 0.5\\,e^{i2\\theta^*})" in model_text(harmonics={1: 1.0, 2: -0.5})
    assert "(-e^{i\\theta^*} + 0.5\\,e^{i2\\theta^*})" in model_text(harmonics={1: -1.0, 2: 0.5})
    assert "(e^{i\\theta^*} + 0.3\\,e^{i(2\\theta^* +1.57)})" in model_text(harmonics={1: 1.0, 2: 0.3j})


def test_editing_a_lobe_diagrams_model_does_not_change_the_next_build():
    """The manifolds come from an lru_cached model; handing the cached lists out let a caller's edit turn the
    next build's six unstable-manifold polylines into one. Cold review 0.8.0 diagram-B F6."""
    before = len(vaft.diagram.separatrix_lobes().scene.role("unstable_manifold"))
    d = vaft.diagram.separatrix_lobes()
    d.model["manifolds"]["unstable"][0][:] = 0.0
    d.model["manifolds"]["stable"].clear()
    assert len(vaft.diagram.separatrix_lobes().scene.role("unstable_manifold")) == before
    assert len(vaft.diagram.separatrix_lobes().model["manifolds"]["stable"]) == 2
