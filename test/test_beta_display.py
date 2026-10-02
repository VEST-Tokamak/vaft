"""Beta family display conventions (issue #947, phase 3).

The stored value is canonical -- beta_t and beta_p as fractions, beta_N as
its Troyon number -- and the display only scales and labels it.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft.omas as vo
from vaft.plot.display import resolve_display

BETA = {"beta_tor": 0.04, "beta_pol": 0.35, "beta_normal": 2.8}


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def sample():
    ods = vo.sample_ods()
    for index in range(len(ods["equilibrium.time_slice"])):
        for leaf, value in BETA.items():
            ods[f"equilibrium.time_slice.{index}.global_quantities.{leaf}"] = value
    return ods


@pytest.mark.parametrize("quantity, unit, scale", [
    ("beta_t", "%", 100.0),
    ("beta_p", "", 1.0),
    ("beta_n", "%·m·T/MA", 1.0),
])
def test_each_beta_has_one_display_convention(quantity, unit, scale):
    display = resolve_display("", subject="equilibrium", quantity=quantity)
    assert (display.unit, display.scale) == (unit, scale)


def test_a_beta_refuses_another_unit():
    with pytest.raises(ValueError, match="displayed as"):
        resolve_display("", unit="%", subject="equilibrium", quantity="beta_p")


@pytest.mark.parametrize("stem, leaf, shown, label", [
    ("beta_t", "beta_tor", 4.0, "[%]"),
    ("beta_p", "beta_pol", 0.35, None),
    ("beta_n", "beta_normal", 2.8, "[%·m·T/MA]"),
])
def test_the_plot_scales_the_drawing_not_the_data(sample, stem, leaf, shown, label):
    before = copy.deepcopy(sample)
    figure, axes = getattr(vo, f"plot_equilibrium_time_{stem}")(sample)
    assert np.allclose(axes.get_lines()[0].get_ydata(), shown)
    if label is None:
        assert "[" not in axes.get_ylabel()
    else:
        assert axes.get_ylabel().endswith(label)
    stored = float(sample[f"equilibrium.time_slice.0.global_quantities.{leaf}"])
    assert stored == BETA[leaf] == float(before[f"equilibrium.time_slice.0.global_quantities.{leaf}"])


def test_discovery_reports_the_beta_conventions(sample):
    records = {r.name: r for r in vo.available_plots(sample)}
    assert records["equilibrium_time_beta_n"].display["unit"] == "%·m·T/MA"
    assert records["equilibrium_time_beta_p"].display["unit"] == ""
    assert records["equilibrium_time_beta_t"].display["unit"] == "%"


def test_the_slice_summary_states_beta_n_with_its_units(sample):

    figure, axes = vo.plot_equilibrium_overview(sample)
    texts = [t.get_text() for ax in figure.axes for t in ax.texts]
    joined = "\n".join(texts)
    assert "beta_N" in joined and "2.8 %·m·T/MA" in joined
    assert "beta_p" in joined and "0.35\n" in joined + "\n"
