"""Predator--prey plots (#1820): they draw process results and derive nothing themselves."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from vaft.formula import turbulence as pp
from vaft.plot import gyrokinetics as gkplot
from vaft.process import gyrokinetics as gk

PARAMS = dict(gamma_eff=0.3, coupling_suppression=0.5, coupling_drive=0.2, gamma_zonal=0.1)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def data():
    t = np.linspace(0.0, 120.0, 600)
    sol = solve_ivp(lambda _, y: pp.predator_prey_rhs(y[0], y[1], **PARAMS), (0, 120), [1.2, 0.3],
                    t_eval=t, method="LSODA", rtol=1e-10, atol=1e-12)
    n, e = sol.y
    norm = {key: "synthetic" for key in gk.NORMALIZATION_KEYS}
    fit = gk.fit_predator_prey_model(t, n, e, normalization=norm, window=(10.0, 110.0))
    metrics = gk.predator_prey_cycle_metrics(t, n, e, transport=2 * n)
    return t, n, e, fit, metrics


def test_the_trace_follows_the_figure_axes_contract_and_drops_the_missing_panel(data):
    t, n, e, fit, metrics = data
    figure, panels = gkplot.plot_predator_prey_trace(t, n, e)
    assert len(panels) == 2
    figure, panels = gkplot.plot_predator_prey_trace(t, n, e, transport=2 * n, fit=fit, metrics=metrics)
    assert len(panels) == 3
    fitted = [line for line in panels[0].get_lines() if "fit" in str(line.get_label())]
    data_line = [line for line in panels[0].get_lines() if line.get_label() == "N (turbulence)"][0]
    assert fitted and fitted[0].get_linestyle() != data_line.get_linestyle()
    assert np.allclose(fitted[0].get_xdata(), t[(t >= 10.0) & (t <= 110.0)])


def test_a_fit_from_another_time_axis_is_refused(data):
    t, n, e, fit, _ = data
    with pytest.raises(ValueError, match="time axis"):
        gkplot.plot_predator_prey_trace(t[::2], n[::2], e[::2], fit=fit)


def test_supplied_axes_are_used_and_no_figure_is_created(data):
    t, n, e, _, _ = data
    figure, axes = plt.subplots(2, 1)
    before = plt.get_fignums()
    out_figure, out_axes = gkplot.plot_predator_prey_trace(t, n, e, ax=axes)
    assert out_figure is figure and plt.get_fignums() == before


def test_the_phase_space_shows_the_fitted_structure_from_the_formula(data):
    t, n, e, fit, _ = data
    _, axes = gkplot.plot_predator_prey_phase_space(n, e, time=t, fit=fit)
    n_star, e_star = pp.predator_prey_fixed_point(
        fit.gamma_eff, fit.coupling_suppression, fit.coupling_drive, fit.gamma_zonal)
    star = [line for line in axes.get_lines() if line.get_label() == "fixed point"][0]
    assert star.get_xdata()[0] == pytest.approx(n_star) and star.get_ydata()[0] == pytest.approx(e_star)
    labels = {str(line.get_label()) for line in axes.get_lines()}
    assert {"LV fit", r"$\dot N = 0$", r"$\dot E = 0$"} <= labels
    assert axes.collections                                       # trajectory + vector field


def test_the_phase_space_without_a_fit_draws_only_the_data(data):
    _, n, e, _, _ = data
    _, axes = gkplot.plot_predator_prey_phase_space(n, e, log=False)
    assert not [line for line in axes.get_lines() if line.get_label() == "fixed point"]


def test_the_lag_plot_marks_the_estimates_it_is_given(data):
    _, _, _, fit, metrics = data
    model = pp.predator_prey_response_lag(fit.gamma_eff, fit.gamma_zonal)
    _, axes = gkplot.plot_predator_prey_lag(
        metrics.correlation_lags, metrics.cross_correlation,
        peak_lag=metrics.peak_response_lag_mean, model_lag=model,
        correlation_lag=metrics.correlation_lag)
    labels = [str(line.get_label()) for line in axes.get_lines()]
    assert any(label.startswith("correlation maximum") for label in labels)
    assert any(label.startswith("peak response lag") for label in labels)
    assert any(label.startswith("LV model lag") for label in labels)
    _, bare = gkplot.plot_predator_prey_lag(metrics.correlation_lags, metrics.cross_correlation)
    assert not any("lag" in str(line.get_label()) and "corr" not in str(line.get_label())
                   for line in bare.get_lines())


def test_plotting_does_no_analysis(data, monkeypatch):
    """The renderers must not detect peaks, correlate or fit: break those and plot."""
    t, n, e, fit, metrics = data

    def forbidden(*_, **__):
        raise AssertionError("a plot called an analysis routine")

    for name in ("predator_prey_cycle_metrics", "fit_predator_prey_model", "_peaks"):
        monkeypatch.setattr(gk, name, forbidden)
    import scipy.signal
    import scipy.optimize
    monkeypatch.setattr(scipy.signal, "find_peaks", forbidden)
    monkeypatch.setattr(scipy.optimize, "least_squares", forbidden)
    gkplot.plot_predator_prey_trace(t, n, e, transport=2 * n, fit=fit, metrics=metrics)
    gkplot.plot_predator_prey_phase_space(n, e, time=t, fit=fit)
    gkplot.plot_predator_prey_lag(metrics.correlation_lags, metrics.cross_correlation,
                                  correlation_lag=metrics.correlation_lag)
