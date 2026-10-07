"""Current diffusion and current drive (#1605): the evolution is the formula's, the claims are the physics'."""

import math

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram._current_diffusion import _CD_EARLY, _STEP, _Cylinder, _drive_states, _ramp
from vaft.formula.constants import MU0
from vaft.formula.ordering import resistive_diffusion_time
from vaft.formula.geometry import cylindrical_current_diffusion_rate


# ---------------------------------------------------------------------------
# the formulas
# ---------------------------------------------------------------------------


def test_the_diffusion_time_is_mu0_a2_over_eta():
    assert resistive_diffusion_time(0.3, 7.8e-7) == pytest.approx(MU0 * 0.09 / 7.8e-7)
    np.testing.assert_allclose(resistive_diffusion_time(np.array([1.0, 2.0]), 1.0), [MU0, 4 * MU0])
    for a, eta in ((0.0, 1.0), (1.0, -1.0)):
        with pytest.raises(ValueError):
            resistive_diffusion_time(a, eta)


def test_the_rate_of_a_quartic_current_with_uniform_eta_is_analytic():
    # I = r^4: (r/mu0) d/dr[(eta/r) 4 r^3] = 8 eta r^2 / mu0
    r = np.linspace(0.0, 0.5, 1001)
    eta = np.full_like(r, 2e-6)
    rate = cylindrical_current_diffusion_rate(r, r**4, eta)
    np.testing.assert_allclose(rate[1:-1], 8 * 2e-6 * r[1:-1] ** 2 / MU0, rtol=1e-6)
    assert rate[0] == 0.0 and np.isnan(rate[-1])


def test_uniform_e_is_a_steady_state_to_second_order():
    # j = E / eta with one E across the radius does not evolve; the residual falls as the spacing squared
    def residual(n):
        r = np.linspace(0.0, 1.0, n)
        eta = 1e-6 * (0.1 + 0.9 * (1 - r**2) ** 1.5) ** -1.5
        j = 1.0 / eta
        I = np.concatenate([[0.0], np.cumsum(np.pi * (j[1:] * r[1:] + j[:-1] * r[:-1]) * np.diff(r))])
        return np.max(np.abs(cylindrical_current_diffusion_rate(r, I, eta)[1:-1])) / I[-1]

    assert residual(401) < residual(201) / 3.0


def test_a_driven_current_equal_to_the_total_is_steady():
    # j_ni = j everywhere: E = eta (j - j_ni) = 0, nothing evolves whatever eta is
    r = np.linspace(0.0, 1.0, 401)
    j = (1 - r**2) ** 2
    I = np.pi * (1 - (1 - r**2) ** 3) / 3.0  # 2 pi int j r dr
    eta = 1e-6 * (1 + 30 * r**4)
    rate = cylindrical_current_diffusion_rate(r, I, eta, j_ni=j)
    assert np.max(np.abs(rate[1:-1])) < 1e-4 * np.max(np.abs(cylindrical_current_diffusion_rate(r, I, eta)[1:-1]))


@pytest.mark.parametrize("bad", [
    {"r": np.array([0.1, 0.2, 0.3])},
    {"r": np.array([0.0, 0.5])},
    {"eta": np.array([1.0, 0.0, 1.0])},
    {"I": np.zeros(2)},
    {"j_ni": np.zeros(4)},
])
def test_the_rate_refuses_bad_input(bad):
    kwargs = {"r": np.array([0.0, 0.5, 1.0]), "I": np.zeros(3), "eta": np.ones(3), **bad}
    with pytest.raises(ValueError):
        cylindrical_current_diffusion_rate(**kwargs)


# ---------------------------------------------------------------------------
# the model integrates the formula
# ---------------------------------------------------------------------------


def test_time_is_in_units_of_the_core_diffusion_time():
    cyl = _Cylinder()
    assert cyl.tau_R == pytest.approx(1.0)
    assert cyl.eta[-1] / cyl.eta[0] == pytest.approx((100.0 / 10.0) ** 1.5)  # Spitzer, T_e^{-3/2}


def test_a_backward_euler_step_satisfies_the_formula():
    cyl = _Cylinder()
    I0 = cyl.x**2
    dt = _STEP  # one implicit step
    source = cyl.source(0.5, 0.1, 0.2)
    I1 = cyl.evolve(I0, 0.0, [dt], j_ni=source)[dt]
    rate = cylindrical_current_diffusion_rate(cyl.x, I1, cyl.eta, source)
    np.testing.assert_allclose(((I1 - I0) / dt)[1:-1], rate[1:-1], rtol=1e-8, atol=1e-8)
    assert I1[0] == 0.0 and I1[-1] == pytest.approx(1.0)


def test_the_source_carries_its_fraction_of_the_current():
    cyl = _Cylinder()
    s = cyl.source(0.45, 0.2, 0.35)
    assert 2 * math.pi * np.trapezoid(s * cyl.x, cyl.x) == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# current diffusion
# ---------------------------------------------------------------------------


def test_the_fast_ramp_leaves_a_skin_and_a_reversed_q_that_relax():
    m = vaft.diagram.current_diffusion().model
    x, stages = m["x"], m["stages"]
    early, mid, relaxed = stages["early"], stages["penetration"], stages["relaxed"]
    for st in stages.values():
        assert st["I"][-1] == pytest.approx(1.0) and st["I"][0] == 0.0
    # early: hollow current, q_min off axis, negative shear inside it
    assert early["j"][0] < 0.5 * early["j"].max() and x[np.argmax(early["j"])] > 0.3
    i = int(np.argmin(early["q"]))
    assert x[i] > 0.4 and np.all(early["s"][5:i - 5] < 0)
    # penetration: q_min moves in and the reversal weakens
    k = int(np.argmin(mid["q"]))
    assert 0.0 < x[k] < x[i] and mid["q"][0] - mid["q"][k] < early["q"][0] - early["q"][i]
    # relaxed: peaked current, monotonic q
    assert np.argmax(relaxed["j"]) == 0 and np.all(np.diff(relaxed["q"]) > -1e-9)


def test_relaxed_current_is_one_e_over_eta_and_uniform_eta_relaxes_flat():
    m = vaft.diagram.current_diffusion().model
    x = m["x"]
    e = m["eta"] * m["stages"]["relaxed"]["j"]
    inner = slice(5, -5)
    assert np.std(e[inner]) / np.mean(e[inner]) < 1e-3
    flat = m["uniform_eta"]["relaxed"]
    np.testing.assert_allclose(flat["j"][inner], 1.0, rtol=2e-3)
    np.testing.assert_allclose(flat["q"][1:], 3.5, rtol=2e-3)
    assert m["stages"]["relaxed"]["j"][0] > 2.0  # the hot core alone makes it peaked
    assert x[0] == 0.0


def test_the_ramp_is_much_faster_than_diffusion():
    m = vaft.diagram.current_diffusion().model
    assert m["t_ramp"] <= 0.02 * m["tau_R"]


# ---------------------------------------------------------------------------
# current drive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deposition", ["off_axis", "on_axis"])
def test_the_driven_source_persists_and_the_ohmic_part_relaxes_to_one_e(deposition):
    data = _drive_states(deposition)
    x = data["x"]
    cyl = _Cylinder()
    inner = slice(5, -5)
    for name in ("ECCD", "NBCD"):
        row = data["rows"][name]
        for when in ("early", "relaxed"):
            assert row[when]["I"][-1] == pytest.approx(1.0)  # fixed I_p
        # relaxed: total = source + E/eta with one E
        ohmic = (row["relaxed"]["j"] - np.pi * row["source"]) * cyl.eta / cyl.eta[0]
        assert np.std(ohmic[inner]) / abs(np.mean(ohmic[inner])) < 5e-3
        # early: the total has not yet taken up the source; the ohmic current shields it
        centre = int(np.argmax(row["source"]))
        gain_early = row["early"]["j"][centre] - row["before"]["j"][centre]
        assert gain_early < np.pi * row["source"][centre]
    assert data["rows"]["ECCD"]["parameters"][1] < data["rows"]["NBCD"]["parameters"][1]  # ECCD narrower


def test_off_axis_drive_reverses_or_dents_the_shear_and_on_axis_drive_lowers_q0():
    off = _drive_states("off_axis")["rows"]
    on = _drive_states("on_axis")["rows"]
    x = _drive_states("off_axis")["x"]
    assert np.any(off["ECCD"]["relaxed"]["s"][1:-1] < -0.1)      # local negative shear at the deposition
    nb = off["NBCD"]["relaxed"]
    assert x[int(np.argmin(nb["q"]))] > 0.2                      # q_min off axis
    for name in ("ECCD", "NBCD"):
        assert on[name]["relaxed"]["q"][0] < on[name]["before"]["q"][0]


def test_the_diagram_draws_what_the_model_holds():
    d = vaft.diagram.current_drive_profiles("off_axis")
    assert d.model["deposition"] == "off_axis"
    texts = " ".join(item.text for item in d.scene.items if hasattr(item, "text"))
    for name in ("Ohmic", "ECCD", "NBCD"):
        assert name in texts
    assert "persists" in texts and "not universal" in texts


@pytest.mark.parametrize("call", [
    lambda: vaft.diagram.current_drive_profiles("edge"),
    lambda: vaft.diagram.current_drive_profiles(("off_axis",)),
    lambda: vaft.diagram.current_drive_profiles(labels="yes"),
    lambda: vaft.diagram.current_diffusion(labels=0),
])
def test_bad_arguments_are_refused(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("builder", ["current_diffusion", "current_drive_profiles"])
def test_labels_false_draws_no_text_beyond_the_axes(builder):
    d = getattr(vaft.diagram, builder)(labels=False)
    roles = {item.role for item in d.scene.items if hasattr(item, "text")}
    assert roles <= {"axes", "ticks"} | {f"actuator:{n}" for n in ("Ohmic", "ECCD", "NBCD")}


def test_the_early_drive_snapshot_is_shortly_after_switch_on():
    assert _CD_EARLY < 0.01
    _, states = _ramp()
    assert set(states) == {"early", "penetration", "relaxed"}


def test_off_scale_q_leaves_the_chart_instead_of_being_capped():
    # q = q_a rho^2 / I is smooth wherever j is: a capped plateau would draw a corner (a q' jump) that is not there
    from vaft.diagram._scene import Polyline

    d = vaft.diagram.current_diffusion()
    curves = [item for item in d.scene.items if isinstance(item, Polyline) and item.role in ("q", "q_uniform_eta")]
    assert curves
    for curve in curves:
        y = np.array([p[1] for p in curve.points])
        dy = np.diff(y)
        assert np.max(np.abs(np.diff(dy))) < 0.2 * np.max(np.abs(dy)) + 1e-9
        assert not np.any(np.isclose(dy[:5], 0.0) & (y[:5] > y.min() + 1.0))  # no flat run at the top
