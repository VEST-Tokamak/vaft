import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.plot.analysis import analysis_diagnostics
from vaft.plot.onedim import plot_onedim_profile
from vaft.plot.topview import equilibrium_CX_topview, plot_topview


def test_analysis_diagnostics_handles_missing_optional_data():
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 42
    ods["magnetics.time"] = np.array([0.0, 0.1])
    ods["magnetics.ip.0.time"] = np.array([0.0, 0.1])
    ods["magnetics.ip.0.data"] = np.array([0.0, 1000.0])
    fig, axes = analysis_diagnostics(ods, show=False)
    assert axes.shape == (5, 2)
    assert len(axes[0, 0].lines) == 1
    assert axes[1, 0].texts[0].get_text() == "Data unavailable"
    plt.close(fig)


def test_plot_onedim_profile_uses_selected_equilibrium_slice():
    ods = ODS(consistency_check=False)
    ods["equilibrium.time_slice.0.time"] = 0.0
    ods["equilibrium.time_slice.1.profiles_1d.rho_tor_norm"] = np.array([0.0, 0.5, 1.0])
    ods["equilibrium.time_slice.1.profiles_1d.pressure"] = np.array([3.0, 2.0, 1.0])
    fig, ax = plot_onedim_profile(
        ods,
        "equilibrium.time_slice.0.profiles_1d.pressure",
        "Pressure",
        time_slice=1,
        show=False,
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 0.5, 1.0])
    plt.close(fig)


def test_plot_onedim_profile_raises_instead_of_returning_empty_figure():
    with pytest.raises(ValueError, match="No plottable"):
        plot_onedim_profile(
            ODS(),
            "equilibrium.time_slice.0.profiles_1d.pressure",
            "Pressure",
            show=False,
        )


def test_plot_onedim_profile_uses_core_profile_grid():
    ods = ODS(consistency_check=False)
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = np.array([0.0, 1.0])
    ods["core_profiles.profiles_1d.0.electrons.density"] = np.array([2.0, 1.0])
    fig, ax = plot_onedim_profile(
        ods,
        "core_profiles.profiles_1d.0.electrons.density",
        "Density",
        show=False,
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.0, 1.0])
    plt.close(fig)


class _TopviewODS(dict):
    def plot_equilibrium_CX_topview(self, *, time_index=None, time=None, ax=None, **kwargs):
        del time_index, time, kwargs
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return {"ax": ax}

    plot_lh_antennas_CX_topview = plot_equilibrium_CX_topview
    plot_ec_launchers_CX_topview = plot_equilibrium_CX_topview
    plot_pellets_trajectory_CX_topview = plot_equilibrium_CX_topview


def test_topview_wrapper_and_composite_delegate_to_omas_renderer():
    ods = _TopviewODS(equilibrium={})
    fig, ax = equilibrium_CX_topview(ods, show=False)
    assert len(ax.lines) == 1
    plt.close(fig)

    fig, ax = plot_topview(ods, show=False)
    assert len(ax.lines) == 1
    plt.close(fig)
