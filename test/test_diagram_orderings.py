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
