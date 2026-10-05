"""vaft.plot.gyrokinetics draws on synthetic CGYRO-shaped data (#1354 stage 4)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.code.gacode.cgyro import collect_cgyro_outputs
from vaft.plot import gyrokinetics as gkplot

from test_cgyro_adapter import write_run


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_linear_spectrum_from_runs_and_overlay(tmp_path):
    runs = [
        collect_cgyro_outputs(write_run(tmp_path / f"ky{i}", gamma=0.1 * (i + 1),
                                        exit_message=("Linear converged" if i else
                                                      "Linear terminated at max time")))
        for i in range(3)
    ]
    spectrum = gkplot.cgyro_linear_spectrum(runs)
    assert spectrum["ky"].size == 3 and not spectrum["converged"][0]
    figure, axes = gkplot.plot_linear_spectrum(
        spectrum,
        references={"TGLF SAT2": {"ky": [0.2, 0.3, 0.4],
                                  "gamma": np.ones((3, 2)), "omega": -np.ones((3, 2))}},
    )
    assert len(axes) == 2
    labels = [line.get_label() for line in axes[0].get_lines()]
    assert "CGYRO" in labels and "TGLF SAT2" in labels


def test_eigenfunction_panels_per_field(tmp_path):
    run = collect_cgyro_outputs(write_run(tmp_path / "run"))
    figure, axes = gkplot.plot_eigenfunction(
        run.grid["thetab"], {"phi": run.ballooning["phi"], "a_parallel": 0.1j * run.ballooning["phi"]})
    assert len(axes) == 2


def test_convergence_and_flux_figures():
    gkplot.plot_convergence(
        {"n_theta": {"value": [24, 32, 48], "gamma": [0.24, 0.25, 0.25], "omega": [0.4, 0.4, 0.41]}},
        baseline={"gamma": 0.25, "omega": 0.4},
    )
    t = np.linspace(0, 100, 50)
    gkplot.plot_flux_trace(t, {"Q_i": np.sin(t) + 2}, window=(50, 100),
                           references={"TGLF SAT0": 1.66})
    gkplot.plot_flux_ky_spectrum([0.1, 0.2, 0.3], {"Q_i": [1, 2, 1]})


def test_a_supplied_axes_is_used():
    figure, ax = plt.subplots()
    out_figure, out_ax = gkplot.plot_flux_ky_spectrum([0.1], {"Q": [1]}, ax=ax)
    assert out_ax is ax and out_figure is figure
