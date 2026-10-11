"""The timescale hierarchy (#1627): every time from a formula kernel, the ordering ratios consistent."""

import pytest

import vaft.diagram
from vaft.diagram._orderings import ILLUSTRATIVE_STATE, timescales
from vaft.formula.ordering import evolution_time, resistive_diffusion_time


def test_times_are_ordered_on_the_axis_and_the_ratios_follow_from_them():
    d = vaft.diagram.timescale_hierarchy()
    times, ratios = d.model["times"], d.model["ratios"]
    assert ratios["lundquist"] == pytest.approx(times["resistive"] / times["alfven"], rel=1e-9)
    assert ratios["quasi_static"] == pytest.approx(times["evolution"] / times["alfven"])
    assert ratios["relaxation"] == pytest.approx(times["pulse"] / times["resistive"])
    assert times["evolution"] == pytest.approx(evolution_time(ILLUSTRATIVE_STATE["I_p"], ILLUSTRATIVE_STATE["dI_dt"]))
    # the inverse electron gyrofrequency is shortest, the resistive time among the longest
    assert min(times, key=times.get) == "electron_gyration"
    # markers sit in the same order as the values
    xs = {k: [it for it in d.scene.role(f"time:{k}") if hasattr(it, "kind")][0].at[0] for k in times}
    assert sorted(times, key=times.get) == sorted(xs, key=xs.get)


def test_only_the_pulse_is_an_input():
    assert vaft.diagram.timescale_hierarchy().model["inputs"] == {"pulse"}
    assert {k for k, (_, _, given) in timescales().items() if given} == {"pulse"}


def test_the_resistive_time_uses_the_ordering_length():
    times = timescales()
    s = dict(ILLUSTRATIVE_STATE)
    eta_ratio = times["resistive"][1] / resistive_diffusion_time(s["a"], 1.0)
    assert eta_ratio > 0  # eta is positive and the length is the minor radius
    s["a"] *= 2.0
    assert timescales(s)["resistive"][1] == pytest.approx(4.0 * times["resistive"][1])


def test_deterministic_exported_and_labels():
    fn = vaft.diagram.timescale_hierarchy
    assert fn().tikz == fn().tikz
    assert "timescale_hierarchy" in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")


def test_timescales_include_the_guides_bounce_transit_and_sound_times():
    from vaft.diagram._orderings import timescales

    from vaft.formula.constants import MI_P
    from vaft.formula.neoclassical import deeply_trapped_bounce_frequency, transit_frequency
    from vaft.formula.ordering import thermal_speed
    from vaft.formula.sol import ion_sound_speed

    s = ILLUSTRATIVE_STATE
    t = {k: v[1] for k, v in timescales().items()}
    v_ti = thermal_speed(s["T_i"], MI_P)
    assert t["sound"] == pytest.approx(s["a"] / ion_sound_speed(s["T_e"], s["T_i"], MI_P))
    assert t["transit"] == pytest.approx(1.0 / transit_frequency(v_ti, s["q"], s["R0"]))
    assert t["bounce"] == pytest.approx(1.0 / deeply_trapped_bounce_frequency(v_ti, s["q"], s["R0"], 0.5 * s["a"] / s["R0"]))
    # a deeply trapped ion bounces slower than a passing one transits, and both after the sound crossing
    assert t["sound"] < t["transit"] < t["bounce"]
    assert t["alfven"] < t["sound"]
    note = vaft.diagram.timescale_hierarchy().scene.role("note")[0].text
    assert "Drift precession" in note and "illustrative state" in note


def test_spatial_scales_are_kernel_values_on_the_same_state():
    from vaft.diagram._orderings import ILLUSTRATIVE_STATE, lengths
    from vaft.formula.constants import MI_P
    from vaft.formula.neoclassical import banana_width
    from vaft.formula.ordering import inertial_length, sound_gyroradius

    from vaft.formula.constants import ME, QE
    from vaft.formula.ordering import thermal_speed
    from vaft.formula.particle import larmor_radius

    s, v = ILLUSTRATIVE_STATE, {k: val[1] for k, val in lengths().items()}
    assert v["rho_i"] == pytest.approx(larmor_radius(QE, MI_P, thermal_speed(s["T_i"], MI_P), s["B"]))
    assert v["rho_e"] == pytest.approx(larmor_radius(QE, ME, thermal_speed(s["T_e"], ME), s["B"]))
    # a different state moves the lengths: they are computed, not stored
    assert lengths({**s, "B": 2 * s["B"]})["rho_i"][1] == pytest.approx(0.5 * v["rho_i"])
    assert v["d_i"] == pytest.approx(inertial_length(s["n"], MI_P))
    assert v["rho_s"] == pytest.approx(sound_gyroradius(s["T_e"], MI_P, s["B"]))
    assert v["banana"] == pytest.approx(banana_width(v["rho_i"], s["q"], 0.5 * s["a"] / s["R0"]))
    assert v["rho_e"] < v["rho_i"] < v["banana"] < v["minor"] < v["connection"]
    d = vaft.diagram.spatial_scale_orderings()
    assert d.model["assumed"] == {"gradient", "minor", "major", "connection"}
    assert "need a mode" in d.scene.role("ratios")[0].text
    assert "spatial_scale_orderings" in vaft.diagram.__all__
    assert d.tikz == vaft.diagram.spatial_scale_orderings().tikz
    with pytest.raises(ValueError):
        vaft.diagram.spatial_scale_orderings(labels="yes")


def test_close_neighbours_never_share_a_stacking_level():
    from vaft.diagram._orderings import _free_level

    placed = [(0.0, False, 0), (1.0, False, 1)]
    assert _free_level(placed, 1.5, False, 2.4) == 2          # both earlier neighbours are close
    assert _free_level(placed, 1.5, True, 2.4) == 0           # the other side is free
    assert _free_level([(0.0, False, 1)], 0.5, False, 2.4) == 0   # level 0 is free again
