"""Interactive equilibrium exploration (issue #261 §14-17).

One selected slice, shared by the time marker, the 2-D flux, the profiles and
the global quantities; every selection snaps to a stored slice; the
scientific contract (``SliceNavigator``) is separate from the widget backend,
and static plotting never imports a widget toolkit.
"""

import contextlib
import io
import subprocess
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.plot.navigation import SliceNavigator


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shot():
    return _load("samples/39915/omas.json.gz")


def _markers(result):
    return [line for axis in result.history_axes for line in axis.lines if line.get_linestyle() == ":"]


# ---------------------------------------------------------------------------
# The scientific contract: SliceNavigator (sections 15, 16, 17)
# ---------------------------------------------------------------------------

def test_the_navigator_snaps_to_stored_slices_and_never_interpolates():
    nav = SliceNavigator([0.10, 0.20, 0.30, 0.40])
    assert nav.select(0.26) == (2, 0.30)
    assert nav.select(0.25) == (1, 0.20)  # a tie goes to the earlier slice
    assert nav.select(9.0) == (3, 0.40)   # clamped to the stored range
    assert nav.time in nav.times


def test_the_navigator_only_ever_holds_a_usable_slice():
    nav = SliceNavigator([0.1, 0.2, 0.3, 0.4, 0.5], usable=[0, 2, 4])
    assert nav.selected == 2 and nav.position == 1
    assert nav.select(0.21) == (2, 0.3)      # 0.2 is nearer but not usable
    with pytest.raises(ValueError, match="not usable"):
        nav.select_index(1)
    assert nav.step(+5) == (4, 0.5) and nav.step(-9) == (0, 0.1)
    with pytest.raises(ValueError, match="at least one usable"):
        SliceNavigator([np.nan, np.nan])


def test_observers_run_once_per_change_and_can_leave():
    nav = SliceNavigator([0.1, 0.2, 0.3])
    seen = []
    leave = nav.subscribe(lambda n: seen.append(n.selected))
    nav.select_index(0); nav.select_index(0); nav.select_index(2)
    assert seen == [0, 2]
    leave(); nav.select_index(1)
    assert seen == [0, 2]


# ---------------------------------------------------------------------------
# The figure (sections 14, 15)
# ---------------------------------------------------------------------------

def test_the_interactive_figure_shares_one_selected_slice(shot):
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="none")
    figure, axes, nav = result
    assert axes.shape == (7,) and len(result.history_axes) == 2 and result.widget is None
    assert nav.selected == 4  # the representative slice, as the static overview
    assert "t = 320.00 ms (slice 5 of 9, selected)" in figure._suptitle.get_text()
    assert all(float(m.get_xdata()[0]) == nav.time for m in _markers(result))
    nav.select(0.3245)
    assert nav.selected == 6
    assert "t = 325.00 ms (slice 7 of 9, selected)" in figure._suptitle.get_text()
    assert all(float(m.get_xdata()[0]) == 0.325 for m in _markers(result))
    assert [panel.get_title() for panel in axes.ravel()] == [
        "Poloidal flux", "Pressure", "Safety Factor q", "Toroidal Current Density (derived)",
        "dp/dpsi", "F dF/dpsi", "Global quantities",
    ]
    plt.close(figure)


def test_time_and_time_slice_pick_the_initial_slice(shot):
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="none", time=0.3195)
    assert result.navigator.selected == 3
    plt.close(result.figure)
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="none", time_slice=0)
    assert result.navigator.selected == 0
    plt.close(result.figure)


def test_the_matplotlib_slider_and_the_navigator_follow_each_other(shot):
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="matplotlib", time=0.318)
    from matplotlib.widgets import Slider
    assert isinstance(result.widget, Slider) and result.widget.val == 2
    result.widget.set_val(7)
    assert result.navigator.selected == 7
    assert "slice 8 of 9" in result.figure._suptitle.get_text()
    result.navigator.select_index(1)
    assert result.widget.val == 1
    plt.close(result.figure)


def test_one_shot_at_a_time_and_known_backends_only(shot):
    odc = omas.ODC()
    odc["a"] = shot
    odc["b"] = shot
    with pytest.raises(ValueError, match="one shot at a time"):
        vaft.omas.plot_equilibrium_interactive(odc, backend="none")
    with pytest.raises(ValueError, match="backend must be one of"):
        vaft.omas.plot_equilibrium_interactive(shot, backend="qt")


def test_static_plotting_never_imports_a_widget_toolkit():
    # The plotting layer imports neither toolkit, and the interactive module
    # imports neither until a backend is asked for.  (vaft.omas itself has
    # long imported ipywidgets through vaft.process.magnetics; that is not
    # the plotting layer's doing and is not asserted here.)
    # matplotlib.pyplot imports matplotlib.widgets itself, so only ipywidgets
    # is a meaningful witness.
    code = (
        "import sys, vaft.plot; a = 'ipywidgets' in sys.modules; "
        "import vaft.omas; before = 'ipywidgets' in sys.modules; "
        "import vaft.omas.interactive; print(a, ('ipywidgets' in sys.modules) and not before)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split()[-2:] == ["False", "False"], out.stdout


def test_discovery_names_the_entry_point_as_an_interaction_mode(shot):
    record = vaft.omas.available_plots(shot, query="equilibrium", view="overview").find("equilibrium_overview")
    assert record.interaction == ("static", "time-navigable")
    assert record.interaction_entry_points == {"time-navigable": "plot_equilibrium_interactive()"}
    text = str(vaft.omas.available_plots(query="equilibrium", view="overview"))
    assert "interaction: static | time-navigable" in text
    detailed = str(vaft.omas.available_plots(query="equilibrium", view="overview", detail=True))
    assert "time-navigable: vaft.omas.plot_equilibrium_interactive()" in detailed
    # It is an interaction capability, not a view: no such canonical plot exists.
    assert "equilibrium_interactive" not in vaft.plot.canonical_names()


# ---------------------------------------------------------------------------
# Independent review of the interactive figure
# ---------------------------------------------------------------------------

def test_redrawing_leaves_the_figure_as_it_found_it(shot):
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="none")
    count = len(result.figure.axes)
    result.figure.canvas.draw()  # positions settle only once the figure is laid out
    # The layout position: the drawn box also follows each slice's data extent
    # through the panel's equal aspect, which is content, not layout.
    width = result.axes[0].get_position(original=True).width
    for index in result.navigator.usable:
        result.navigator.select_index(index)
        result.figure.canvas.draw()
        assert result.axes[0].get_position(original=True).width == pytest.approx(width)
    assert len(result.figure.axes) == count
    plt.close(result.figure)
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="matplotlib")
    count = len(result.figure.axes)
    for position in range(len(result.navigator.usable)):
        result.widget.set_val(position)
    assert len(result.figure.axes) == count
    plt.close(result.figure)


def test_a_non_finite_time_is_refused():
    nav = SliceNavigator([0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        nav.select(float("nan"))


def test_the_current_history_is_the_measured_waveform_with_the_reconstruction(shot):
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="none")
    current = result.history_axes[0]
    assert "Plasma Current" in current.get_title()
    labels = [line.get_label() for line in current.lines if line.get_linestyle() != ":"]
    assert labels[0] == "39915"  # the magnetics waveform, not the equilibrium's own Ip
    plt.close(result.figure)
