"""``time_range=`` on a time-history line plot sets the window it names.

It was accepted by option validation and then ignored by every LineRecipe
plot, so ``plot_flux_loop_time_flux(ods, time_range=(-5e-3, 30e-3))`` drew the
whole record (#254's canonical analysis task asks for exactly that window).
"""

import matplotlib

matplotlib.use("Agg")

import dataclasses
import hashlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import vaft
from vaft.plot.backend import recipes
from vaft.plot.models import LineSeries, Panels
from vaft.plot.backend.options import OPTION_SCHEMA, validate_options
from vaft.plot.backend.recipes import RECIPES, _build_line_series, takes_time_range


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods(39915)


def test_the_time_range_becomes_the_axis_limits_in_display_units(ods):
    window = (0.30, 0.33)
    model = _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"], time_range=window)
    scale = model.series[0].x[0] / np.asarray(ods["magnetics.flux_loop.0.flux.time"], dtype=float)[0]
    np.testing.assert_allclose(model.x_limits, (window[0] * scale, window[1] * scale))


def test_an_explicit_x_limits_still_wins(ods):
    model = _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"],
                               time_range=(0.30, 0.33), x_limits=(1.0, 2.0))
    assert model.x_limits == (1.0, 2.0)


def test_the_public_plot_draws_the_requested_window_in_its_axis_unit(ods):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(ods, time_range=(0.30, 0.33), xunit="ms")
    axis = np.ravel(axes)[0] if isinstance(axes, np.ndarray) else axes
    assert "[ms]" in axis.get_xlabel()
    np.testing.assert_allclose(axis.get_xlim(), (300.0, 330.0))
    plt.close(figure)


@pytest.mark.parametrize("layout", ["subplots", "grouped"])
def test_a_split_layout_keeps_the_window_on_every_panel(ods, layout):
    figure, axes = vaft.omas.plot_flux_loop_time_flux(ods, time_range=(0.30, 0.33), xunit="ms", layout=layout)
    for axis in np.ravel(axes):
        if axis.has_data():
            np.testing.assert_allclose(axis.get_xlim(), (300.0, 330.0))
    plt.close(figure)


def test_a_reversed_window_is_refused(ods):
    with pytest.raises(ValueError, match="stop > start"):
        _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"], time_range=(0.33, 0.30))


# -- honoured or refused, never ignored (cold review 0.8.0 delta-absorb-2 F2) ---
# ``time_range=`` is a global option, and only the time abscissa honoured it:
# ``x="index"`` on a line plot and every profile, map and drawing drew the
# whole record silently.  The rule is the one ``time=`` follows.


def test_an_index_abscissa_refuses_a_time_window(ods):
    with pytest.raises(ValueError, match="time axis.*x='index'"):
        _build_line_series([("", ods)], RECIPES["flux_loop_time_flux"], time_range=(0.30, 0.33), x="index")
    with pytest.raises(ValueError, match="'flux_loop_time_flux' draws x='index'"):
        vaft.omas.plot_flux_loop_time_flux(ods, time_range=(0.30, 0.33), x="index")


def test_a_plot_without_a_time_history_refuses_a_time_window(ods):
    with pytest.raises(ValueError, match="'equilibrium_profile_q' takes no time_range="):
        vaft.omas.render_plot("equilibrium_profile_q", ods, time_range=(0.30, 0.33))
    with pytest.raises(ValueError, match="takes no time_range="):
        vaft.omas.extract_equilibrium_profile_q(ods, time_range=(0.30, 0.33))
    with pytest.raises(ValueError, match="takes no time_range="):
        validate_options("equilibrium_profile_q", {"time_range": (0.30, 0.33)})
    validate_options("equilibrium_profile_q", {"time_range": None})
    validate_options("flux_loop_time_flux", {"time_range": (0.30, 0.33)})
    assert "honoured" in OPTION_SCHEMA["time_range"].description


def _sample_plot_names(ods) -> list[str]:
    names = sorted(getattr(card, "name", card) for card in vaft.omas.available_plots(ods))
    return [name for name in names if hasattr(vaft.omas, f"extract_{name}")]


def _digest(model) -> str:
    sha = hashlib.sha1()

    def walk(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            for field in dataclasses.fields(item):
                walk(getattr(item, field.name))
        elif isinstance(item, np.ndarray):
            if item.dtype.kind in "fiub":
                sha.update(np.ascontiguousarray(np.nan_to_num(item.astype(float))).tobytes())
            else:
                sha.update(repr(item.tolist()).encode())
        elif isinstance(item, (list, tuple)):
            for member in item:
                walk(member)
        elif isinstance(item, dict):
            for _, member in sorted(item.items(), key=lambda pair: str(pair[0])):
                walk(member)
        elif callable(item):
            sha.update(getattr(item, "__name__", "callable").encode())
        else:
            sha.update(repr(item).encode())

    walk(model)
    return sha.hexdigest()


def _line_models(model) -> list:
    if isinstance(model, Panels):
        return [member for child in model.models for member in _line_models(child)]
    return [model] if isinstance(model, LineSeries) else []


def test_time_range_is_honoured_or_refused_by_every_sample_plot(ods):
    """No registered plot the sample supports accepts ``time_range=`` and ignores it."""
    window = (0.30, 0.33)
    names = _sample_plot_names(ods)
    assert len(names) > 40
    ignored, unwindowed, honoured, refused = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            extract = getattr(vaft.omas, f"extract_{name}")
            if not takes_time_range(name):
                with pytest.raises(ValueError, match="takes no time_range="):
                    extract(ods, time_range=window)
                refused.append(name)
                continue
            model = extract(ods, time_range=window)
            if _digest(model) == _digest(extract(ods)):
                ignored.append(name)
            # A line plot (alone or in a composite) narrows its x limits to the
            # window in the axis's display unit; the spectral views trim the
            # signal instead and their frequency axes have no limits to check.
            for line in _line_models(model):
                if not line.x_label.lower().startswith("time"):
                    continue
                if line.x_limits is None or not np.allclose(
                    np.asarray(line.x_limits) / np.asarray(window), line.x_limits[0] / window[0]
                ):
                    unwindowed.append(name)
            honoured.append(name)
    assert not ignored, f"time_range= accepted and ignored by {ignored}"
    assert not unwindowed, f"time_range= did not set the x limits of {unwindowed}"
    assert {"flux_loop_time_flux", "plasma_current_time", "magnetics_overview", "mirnov_spectrogram",
            "mirnov_spectrum", "startup_proxies_time"} <= set(honoured)
    assert {"equilibrium_profile_q", "equilibrium_field_psi", "equilibrium_overview_profiles",
            "wall_geometry_poloidal"} <= set(refused)


def _reads_time_range(func, seen: set) -> str | None:
    """The name of ``func`` or of a helper it calls that reads ``options["time_range"]``."""
    import inspect
    import re

    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    if re.search(r"\btime_range\b", source):
        return getattr(func, "__name__", repr(func))
    for helper_name in sorted(set(re.findall(r"\b(_[A-Za-z0-9_]+)\(", source))):
        helper = getattr(recipes, helper_name, None)
        if not callable(helper) or isinstance(helper, type) or helper in seen:
            continue
        seen.add(helper)
        found = _reads_time_range(helper, seen)
        if found:
            return found
    return None


def test_every_builder_that_reads_time_range_declares_it():
    """A computed view windows to ``time_range=`` exactly when its recipe says so."""
    for name, recipe in RECIPES.items():
        if not isinstance(recipe, recipes.CallableRecipe):
            continue
        reader = _reads_time_range(recipe.builder, set())
        if recipe.windowed:
            assert reader is not None, f"{name} declares windowed but no builder reads time_range"
        else:
            assert reader is None, f"{name} refuses time_range= but {reader} reads it"


def test_a_composite_windows_its_histories_and_leaves_the_rest_alone(ods):
    figure, axes = vaft.omas.plot_magnetics_overview(ods, time_range=(0.30, 0.33), xunit="ms")
    for axis in np.ravel(axes):
        if axis.has_data():
            np.testing.assert_allclose(axis.get_xlim(), (300.0, 330.0))
    plt.close(figure)
