"""Predator--prey plots: they draw process results and derive nothing themselves."""

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
    from matplotlib.quiver import Quiver

    assert any(isinstance(c, Quiver) for c in axes.collections)


@pytest.mark.parametrize("log", [True, False])
def test_the_vector_field_points_along_the_model_flow_on_screen(data, log):
    """Each arrow's screen direction is the model's (dN/dt, dE/dt) mapped through the
    axes' scale: same signs, and the screen slope matches the scaled ratio."""
    from matplotlib.quiver import Quiver

    _, n, e, fit, _ = data
    _, axes = gkplot.plot_predator_prey_phase_space(n, e, fit=fit, log=log)
    quiver = [c for c in axes.collections if isinstance(c, Quiver)][0]
    gn, ge = quiver.X, quiver.Y
    dn, de = pp.predator_prey_rhs(gn, ge, fit.gamma_eff, fit.coupling_suppression,
                                  fit.coupling_drive, fit.gamma_zonal)
    assert np.all(np.sign(quiver.U) == np.sign(dn))
    assert np.all(np.sign(quiver.V) == np.sign(de))
    # screen direction through the axes transform: a tiny step along the flow
    trans = axes.transData
    k = len(gn) // 2
    h = 1e-6 * min(gn[k] / max(abs(dn[k]), 1e-300), ge[k] / max(abs(de[k]), 1e-300))
    base = trans.transform((gn[k], ge[k]))
    step = trans.transform((gn[k] + h * dn[k], ge[k] + h * de[k]))
    expected = np.arctan2(*(step - base)[::-1])
    drawn = np.arctan2(quiver.V[k], quiver.U[k])
    assert drawn == pytest.approx(expected, abs=2e-2)


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
    import scipy.integrate
    import scipy.optimize
    import scipy.signal
    for module, name in ((scipy.signal, "find_peaks"), (scipy.signal, "correlate"),
                         (scipy.optimize, "least_squares"), (scipy.optimize, "curve_fit"),
                         (scipy.integrate, "solve_ivp"), (np, "correlate")):
        monkeypatch.setattr(module, name, forbidden)
    gkplot.plot_predator_prey_trace(t, n, e, transport=2 * n, fit=fit, metrics=metrics)
    gkplot.plot_predator_prey_phase_space(n, e, time=t, fit=fit)
    gkplot.plot_predator_prey_lag(metrics.correlation_lags, metrics.cross_correlation,
                                  correlation_lag=metrics.correlation_lag)


def test_bad_optional_inputs_are_refused_or_dropped(data):
    t, n, e, _, _ = data
    with pytest.raises(ValueError, match="transport"):
        gkplot.plot_predator_prey_trace(t, n, e, transport=np.ones(t.size + 1))
    _, axes = gkplot.plot_predator_prey_phase_space(n, e, arrows=0, log=False)
    assert not axes.texts and not [c for c in axes.get_children() if type(c).__name__ == "Annotation"]
