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
        # Three columns: the flux map, the plasma profiles, the fitted source
        # terms and the global quantities.  No packaged reconstruction stores
        # j_tor; it is derived for the slice and the panel says so.
        assert axes.shape == (7,)
        assert [panel.get_title() for panel in axes.ravel()] == [
            "Poloidal flux", "Pressure", "Safety Factor q", "Toroidal Current Density (derived)",
            "dp/dpsi", "F dF/dpsi", "Global quantities",
        ]
        assert figure._suptitle.get_text().startswith(f"Equilibrium slice #{shot} — t = ")
        assert "middle usable slice (no volume stored)" in figure._suptitle.get_text()
        plt.close(figure)


def test_the_model_composes_canonical_members_for_one_slice(shots):
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=2)
    assert isinstance(model, Panels)
    field, pressure, q, j_tor, pprime, ffprime, text = model.models
    assert isinstance(field, Field2D) and isinstance(pressure, Profile1D) and isinstance(q, Profile1D)
    assert isinstance(j_tor, Profile1D) and j_tor.title.endswith("(derived)") and j_tor.series
    assert isinstance(pprime, Profile1D) and isinstance(ffprime, Profile1D)
    assert isinstance(text, TextPanel)
    assert "j_tor" not in shots[39915]["equilibrium.time_slice.2.profiles_1d"]
    assert model.nrows == 3 and model.ncols == 3 and model.spans[0] == (0, 0, 3, 1)
    # The 2-D panel is the same one plot_equilibrium_field_psi draws for that slice.
    same = build_model("equilibrium_field_psi", normalize_entries(shots[39915]), time_slice=2)
    assert np.allclose(field.values, same.values) and field.value_label == same.value_label
    assert "(slice 3 of 9, requested slice)" in model.suptitle


def test_global_quantities_follow_the_display_policy_and_say_what_is_missing(shots):
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=4)
    lines = dict(line.split(None, 1) for line in model.models[-1].lines)
    assert lines["Ip"].endswith(" kA") and lines["psi_axis"].endswith(" mWb")
    assert lines["B_tor"].startswith("at axis") or "B_tor at axis" in "\n".join(model.models[-1].lines)
    # Quantities the g-file does not store are derived on the private copy
    # (issue #475) and shown like any stored value; nothing here reads
    # "not stored" because everything in the table is derivable from 39915.
    derived = {"beta_p": "", "beta_N": "", "li_3": "", "volume": " m^3", "area": " m^2"}
    for label, unit in derived.items():
        assert lines[label] != "not stored" and lines[label].endswith(unit), (label, lines[label])
        float(lines[label].split()[0])
    assert lines["W_mhd"].endswith((" J", " kJ")) and float(lines["W_mhd"].split()[0]) > 0
    assert abs(float(lines["volume"].split()[0]) - 0.5377) < 1e-3
    assert "not stored" not in "\n".join(model.models[-1].lines)
    # A stored value wins over a derived one.
    ods = _with_volumes(_load("samples/39915/omas.json.gz"), [0.0] * 4 + [0.25] + [0.0] * 4)
    model = build_model("equilibrium_overview", normalize_entries(ods))
    assert "largest plasma volume" in model.suptitle
    assert any(line.startswith("volume") and "0.25 m^3" in line for line in model.models[-1].lines)


def test_derivation_never_writes_the_callers_ods():
    ods = _load("samples/39915/omas.json.gz")
    before = len(ods.flat())
    model = build_model("equilibrium_overview", normalize_entries(ods))  # representative slice: 4
    assert "beta_pol" not in ods["equilibrium.time_slice.4.global_quantities"]
    assert len(ods.flat()) == before
    # The representative slice is still chosen from what the input stores.
    index, _, reason = resolve_time_slice(ods)
    assert index == 4 and reason == "middle usable slice (no volume stored)"
    assert "no volume stored" in model.suptitle


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
    assert record.interaction[0] == "static"
    assert record.overview_members == ("poloidal flux", "pressure", "q", "global quantities")
    text = str(vaft.omas.available_plots(query="equilibrium", view="overview"))
    assert "interaction: static" in text and "overview: poloidal flux · pressure · q · global quantities" in text


# ---------------------------------------------------------------------------
# Independent review of the slice summary
# ---------------------------------------------------------------------------

def test_a_slice_the_solver_disowned_is_not_usable():
    ods = _with_volumes(_load("samples/39915/omas.json.gz"), [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flags = np.zeros(9)
    flags[2] = -1  # IMAS: negative output_flag means "shall not be used"
    ods["equilibrium.code.output_flag"] = flags
    index, reason = representative_slice(ods)
    assert index != 2
    assert resolve_time_slice(ods, time=0.318)[0] != 2


def test_the_deprecated_history_function_points_at_the_histories():
    from vaft.plot._migration import DEPRECATED
    assert DEPRECATED["time_equilibrium_analysis"] == "equilibrium_overview_histories"


def test_the_flux_map_shares_the_text_panel_unit_and_marks_the_axis(shots):
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=4)
    field = model.models[0]
    assert field.value_label == "Poloidal Flux [mWb]"
    same = build_model("equilibrium_field_psi", normalize_entries(shots[39915]), time_slice=4)
    assert np.allclose(field.values, same.values)
    axis = [layer for layer in field.overlays if layer.label == "Magnetic axis"]
    assert len(axis) == 1 and axis[0].kind == "points"
    assert np.isclose(axis[0].r[0], 0.3645, atol=1e-3)


def test_a_time_outside_the_stored_slices_warns_and_snaps(shots):
    with pytest.warns(UserWarning, match="outside the stored equilibrium slices"):
        index, stored, _ = resolve_time_slice(shots[39915], time=1000.0)
    assert index == 8
    with pytest.raises(ValueError, match="not a slice index"):
        resolve_time_slice(shots[39915], time_slice=2.7)


def test_the_efit_qa_stage_keeps_its_history_artifact():
    from vaft.database.production_qa import STAGE_VALIDATION_PLOTS
    efit = {plot.plot: plot for plot in STAGE_VALIDATION_PLOTS["efit"]}
    assert "equilibrium_overview" not in efit
    assert efit["equilibrium_overview_histories"].filename == "equilibrium_overview.png"


def test_a_profile_that_cannot_be_derived_is_omitted_and_the_column_restacks(shots, monkeypatch):
    import vaft.omas

    def refuse(*args, **kwargs):
        raise RuntimeError("no derivation here")

    monkeypatch.setattr(vaft.omas, "update_equilibrium_derived_profiles", refuse)
    model = build_model("equilibrium_overview", normalize_entries(shots[39915]), time_slice=2)
    # j_tor is omitted and its column re-stacks: two half-height panels.
    assert len(model.models) == 6 and (model.nrows, model.ncols) == (6, 3)
    assert model.spans[:3] == ((0, 0, 6, 1), (0, 1, 3, 1), (3, 1, 3, 1))
    assert model.spans[3:] == ((0, 2, 2, 1), (2, 2, 2, 1), (4, 2, 2, 1))
    assert [m.title for m in model.models[1:3]] == ["Pressure", "Safety Factor q"]
    # Without a derivation, every derivable quantity reads "not stored" again.
    lines = dict(line.split(None, 1) for line in model.models[-1].lines)
    assert all(lines[label] == "not stored" for label in ("beta_p", "beta_N", "li_3", "volume", "area", "W_mhd"))
    assert lines["Ip"].endswith(" kA")
    figure, axes = vaft.omas.plot_equilibrium_overview(shots[39915], time_slice=2)
    assert axes.shape == (6,) and [a.get_title() for a in axes] == [
        "Poloidal flux", "Pressure", "Safety Factor q", "dp/dpsi", "F dF/dpsi", "Global quantities",
    ]
    # The figure keeps the height of a three-panel column, not six grid rows.
    assert figure.get_size_inches()[1] < 12
    plt.close(figure)


def test_the_overview_columns_restack_to_fill_their_height():
    from vaft.plot.backend.recipes import _overview_spans

    assert _overview_spans([3, 3]) == (3, 3, ((0, 0, 3, 1), (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1),
                                              (0, 2, 1, 1), (1, 2, 1, 1), (2, 2, 1, 1)))
    assert _overview_spans([2, 3])[:2] == (6, 3)
    assert _overview_spans([0, 3]) == (3, 2, ((0, 0, 3, 1), (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1)))
    assert _overview_spans([1, 1]) == (1, 3, ((0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1)))
