"""The static equilibrium overview is one slice from one figure (issue #261 §11-13).

The slice is the representative one unless ``time=`` (snapped to a stored
slice, never interpolated) or ``time_slice=`` says otherwise; the title states
which slice was drawn and why.  The four time histories the name used to draw
live on as ``plot_equilibrium_overview_histories``.
"""

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import omas
import pytest

import vaft
import vaft.omas
from vaft.omas._plot_recipes import (
    build_model,
    normalize_entries,
    representative_slice,
    resolve_time_slice,
)
from vaft.plot.models import Field2D, Panels, Profile1D, TextPanel


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shots():
    return {
        39915: _load("samples/39915/omas.json.gz"),
        41524: _load("samples/41524/imas.nc"),
    }


def _with_volumes(ods, volumes):
    for index, volume in enumerate(volumes):
        ods[f"equilibrium.time_slice.{index}.global_quantities.volume"] = volume
    return ods


# ---------------------------------------------------------------------------
# Representative slice (section 12)
# ---------------------------------------------------------------------------

def test_no_sample_stores_a_volume_so_the_middle_usable_slice_is_representative(shots):
    assert representative_slice(shots[39915]) == (4, "middle usable slice (no volume stored)")
    assert representative_slice(shots[41524]) == (3, "middle usable slice (no volume stored)")


def test_the_largest_volume_wins_when_volumes_are_stored():
    ods = _with_volumes(_load("samples/39915/omas.json.gz"), [0.1, 0.4, 0.9, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1])
    # Ties go to the earlier slice; NaN volumes are ignored, not chosen.
    assert representative_slice(ods) == (2, "largest plasma volume")
    ods["equilibrium.time_slice.2.global_quantities.volume"] = np.nan
    assert representative_slice(ods) == (3, "largest plasma volume")


def test_a_slice_without_a_2d_psi_is_not_usable():
    ods = omas.ODS()
    ods["equilibrium.time_slice.0.time"] = 0.1
    with pytest.raises(ValueError, match="no usable equilibrium slice"):
        representative_slice(ods)


# ---------------------------------------------------------------------------
# time= resolves to a real slice (section 13)
# ---------------------------------------------------------------------------

def test_time_snaps_to_the_nearest_stored_slice_and_reports_it(shots):
    index, stored, reason = resolve_time_slice(shots[39915], time=0.3195)
    assert (index, stored) == (3, 0.319)
    assert reason == "nearest stored slice to t = 319.50 ms"
    # Nothing is interpolated: the returned time is a stored one.
    assert stored in [float(shots[39915][f"equilibrium.time_slice.{i}.time"]) for i in range(9)]


def test_time_slice_names_a_stored_slice_directly(shots):
    assert resolve_time_slice(shots[39915], time_slice=0)[:2] == (0, 0.316)
    with pytest.raises(ValueError, match="outside the 9 stored slices"):
        resolve_time_slice(shots[39915], time_slice=99)
    with pytest.raises(ValueError, match="either time= or time_slice="):
        resolve_time_slice(shots[39915], time=0.3, time_slice=1)


# ---------------------------------------------------------------------------
# The figure (section 11)
# ---------------------------------------------------------------------------

def test_the_overview_has_the_same_shape_on_every_shot(shots):
    for shot, ods in shots.items():
        figure, axes = vaft.omas.plot_equilibrium_overview(ods)
        assert axes.shape == (2, 2)
        assert [panel.get_title() for panel in axes.ravel()] == [
            "Poloidal flux", "Pressure", "Safety Factor q", "Global quantities"
        ]
        assert figure._suptitle.get_text().startswith(f"Equilibrium slice #{shot} — t = ")
        assert "middle usable slice (no volume stored)" in figure._suptitle.get_text()
        plt.close(figure)


def test_the_model_composes_canonical_members_for_one_slice(shots):
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=2)
    assert isinstance(model, Panels)
    field, pressure, q, text = model.models
    assert isinstance(field, Field2D) and isinstance(pressure, Profile1D) and isinstance(q, Profile1D)
    assert isinstance(text, TextPanel)
    # The 2-D panel is the same one plot_equilibrium_field_psi draws for that slice.
    same = build_model("equilibrium_field_psi", normalize_entries(shots[39915]), time_slice=2)
    assert np.array_equal(field.values, same.values)
    assert "(slice 3 of 9, requested slice)" in model.suptitle


def test_global_quantities_follow_the_display_policy_and_say_what_is_missing(shots):
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=4)
    lines = dict(line.split(None, 1) for line in model.models[3].lines)
    assert lines["Ip"].endswith(" kA") and lines["psi_axis"].endswith(" mWb")
    assert lines["B_tor"].startswith("at axis") or "B_tor at axis" in "\n".join(model.models[3].lines)
    assert lines["beta_p"] == "not stored" and lines["volume"] == "not stored"
    ods = _with_volumes(_load("samples/39915/omas.json.gz"), [0.0] * 4 + [0.25] + [0.0] * 4)
    model = build_model("equilibrium_overview", normalize_entries(ods))
    assert "largest plasma volume" in model.suptitle
    assert any(line.startswith("volume") and "0.25 m^3" in line for line in model.models[3].lines)


def test_explicit_time_shows_the_resolved_time_in_the_title(shots):
    figure, axes = vaft.omas.plot_equilibrium_overview(shots[39915], time=0.3245)
    title = figure._suptitle.get_text()
    assert "t = 325.00 ms" in title and "nearest stored slice to t = 324.50 ms" in title
    plt.close(figure)


def test_the_histories_composite_survives_under_its_own_name(shots):
    figure, axes = vaft.omas.plot_equilibrium_overview_histories(shots[39915])
    assert figure._suptitle.get_text().startswith("Equilibrium Time Histories")
    assert "Equilibrium Plasma Current" in [panel.get_title() for panel in axes.ravel()]
    plt.close(figure)
    assert "equilibrium_overview_histories" in vaft.plot.canonical_names()


def test_discovery_states_the_contents_and_the_interaction_mode(shots):
    record = vaft.omas.available_plots(shots[39915], query="equilibrium", view="overview").find(
        "equilibrium_overview"
    )
    assert record.interaction == ("static",)
    assert record.overview_members == ("poloidal flux", "pressure", "q", "global quantities")
    text = str(vaft.omas.available_plots(query="equilibrium", view="overview"))
    assert "interaction: static" in text and "overview: poloidal flux · pressure · q · global quantities" in text
