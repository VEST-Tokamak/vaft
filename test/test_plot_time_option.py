"""``time=`` selects the slice by time, or is refused: never accepted and ignored.

Cold review plot G1: the option schema advertises ``time=`` for every plot, yet
the generic profile / 2-D field / geometry builders and several computed views
read only ``time_slice=`` and drew slice 0 for any ``time=``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import vaft.omas as vo
from vaft.plot.backend import recipes
from vaft.plot.backend.options import validate_options

SLICE = 6


@pytest.fixture(scope="module")
def ods():
    return vo.sample_ods()


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


def _sample_plot_names(ods) -> list[str]:
    names = sorted(getattr(card, "name", card) for card in vo.available_plots(ods))
    return [name for name in names if hasattr(vo, f"extract_{name}")]


def test_time_is_honoured_or_refused_by_every_sample_plot(ods):
    """No registered plot the sample supports accepts ``time=`` and ignores it."""
    instant = float(ods[f"equilibrium.time_slice.{SLICE}.time"])
    names = _sample_plot_names(ods)
    assert len(names) > 40
    ignored, mismatched, honoured = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in names:
            extract = getattr(vo, f"extract_{name}")
            axis = recipes.time_axis_of(name)
            if not axis:
                with pytest.raises(ValueError, match="takes no time="):
                    extract(ods, time=instant)
                continue
            if isinstance(recipes.RECIPES[name], recipes.PanelRecipe) and not any(
                recipes.time_axis_of(member) and recipes.entry_supports(ods, member)
                for member in recipes.RECIPES[name].members
            ):
                # the one member that draws an instant has no data in the sample
                with pytest.raises(ValueError, match="takes no time="):
                    extract(ods, time=instant)
                continue
            default = _digest(extract(ods))
            by_time = _digest(extract(ods, time=instant))
            if by_time == default:
                ignored.append(name)
            if axis == "equilibrium.time_slice" or (
                axis != recipes.OWN_TIME and isinstance(recipes.RECIPES[name], recipes.ProfileRecipe)
            ):
                if by_time != _digest(extract(ods, time_slice=SLICE)):
                    mismatched.append(name)
            honoured.append(name)
    assert not ignored, f"time= accepted and ignored by {ignored}"
    assert not mismatched, f"time= and the time_slice= of the same instant differ for {mismatched}"
    # The fifteen the review named are among them.
    assert {
        "equilibrium_profile_q", "equilibrium_profile_f", "equilibrium_profile_ffprime",
        "equilibrium_profile_pprime", "equilibrium_profile_pressure",
        "equilibrium_overview_profiles", "equilibrium_field_psi", "equilibrium_field_2d",
        "equilibrium_geometry_boundary", "equilibrium_geometry_topview",
        "machine_geometry_topview", "equilibrium_overview_constraints",
        "equilibrium_overview_residuals", "equilibrium_overview_fit_quality",
        "equilibrium_overview_verification",
        # and the ones that always honoured it keep doing so
        "vacuum_field_midplane", "vacuum_field", "equilibrium_overview",
    } <= set(honoured)


def test_every_builder_that_reads_time_declares_it():
    """A computed view whose builder takes ``time`` is not refused it."""
    import inspect

    for name, recipe in recipes.RECIPES.items():
        if not isinstance(recipe, recipes.CallableRecipe) or recipe.time_axis:
            continue
        source = inspect.getsource(recipe.builder)
        assert 'options.get("time")' not in source and 'options["time"]' not in source, name
        assert "time" not in inspect.signature(recipe.builder).parameters, name


def test_time_beside_a_conflicting_time_slice_is_refused(ods):
    instant = float(ods[f"equilibrium.time_slice.{SLICE}.time"])
    with pytest.raises(ValueError, match="pass one of them"):
        vo.extract_equilibrium_profile_q(ods, time=instant, time_slice=2)
    same = vo.extract_equilibrium_profile_q(ods, time=instant, time_slice=SLICE)
    assert _digest(same) == _digest(vo.extract_equilibrium_profile_q(ods, time_slice=SLICE))


def test_validation_refuses_time_for_a_plot_without_an_instant():
    with pytest.raises(ValueError, match="takes no time="):
        validate_options("plasma_current_time", {"time": 0.3})
    validate_options("plasma_current_time", {"time": None})
    validate_options("equilibrium_profile_q", {"time": 0.3})


def test_core_profile_time_is_matched_on_the_core_profiles_times(ods):
    """The slice is the nearest of the IDS the plot slices, not of the equilibrium."""
    from omas import ODS

    synthetic = ODS()
    synthetic.update(ods)  # keep the equilibrium beside it: 9 slices, other times
    stored = [0.331, 0.325, 0.316]  # deliberately not ascending
    rho = np.linspace(0.0, 1.0, 21)
    synthetic["core_profiles.ids_properties.homogeneous_time"] = 1
    synthetic["core_profiles.time"] = np.asarray(stored)
    for index, instant in enumerate(stored):
        base = f"core_profiles.profiles_1d.{index}"
        synthetic[f"{base}.time"] = instant
        synthetic[f"{base}.grid.rho_tor_norm"] = rho
        synthetic[f"{base}.grid.rho_pol_norm"] = rho
        synthetic[f"{base}.electrons.temperature"] = 1000.0 * instant * (1.0 - rho**2) + 1.0
    model = vo.extract_electron_temperature_profile(synthetic, time=0.317)
    peak = max(float(np.nanmax(series.y)) for series in model.series)
    assert peak == pytest.approx(317.0)
