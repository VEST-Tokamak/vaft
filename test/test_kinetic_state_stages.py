"""``kinetic_overview_state`` as an inference chain: stages, assumptions, animation.

The chain is the maintainer's: equilibrium and Thomson in, ``T_i = T_e`` and
``n_C = n_O`` assumed, the Romero ``Z_eff`` first, then ``n_I,eff``, ``n_i``,
``Z_eff(rho)`` and the pressure closure.  The fixture is
``romero_first_case`` of ``workflow/kinetic_state/archive_to_ods.py``: the
40326 archive's equilibrium and Thomson with a **synthetic** Romero value,
written through the real composition writer, until Lane KP's stages
exist.  The plot reads only stored paths and records.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import vaft.omas  # noqa: E402
from vaft.omas.plotting import normalize_entries  # noqa: E402
from vaft.plot.backend.discovery import describe_one, sequence_values  # noqa: E402
from vaft.plot.backend.kinetic_state import (  # noqa: E402
    DEFAULT_STAGE_COLUMNS, QUANTITY_STAGES, STAGES, THOMSON_POINT_STYLE,
)
from vaft.plot.controls import controls_for  # noqa: E402

from _kinetic_state_fixture import (  # noqa: E402
    E, SYNTHETIC_ZEFF_ROMERO, TI_EQUALS_TE_RECORD, kinetic_state_case, romero_first_case,
)

NAME = "kinetic_overview_state"
CP = "core_profiles.profiles_1d.0"
DPI = 30


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def romero():
    return _quiet(romero_first_case)


@pytest.fixture(scope="module")
def archive():
    return _quiet(kinetic_state_case, 40326, "magnetics")


def _extract(ods, **options):
    return _quiet(vaft.omas.extract_kinetic_overview_state, ods, **options)


def _quantities(model):
    return [t["quantity"] for panel in model.models for t in panel.metadata.get("traces", ())]


def _traces(model):
    out = {}
    for panel in model.models:
        for series, record in zip(panel.series, panel.metadata.get("traces", ())):
            assert series.label == record["label"]
            out[record["quantity"]] = (series, record)
    return out


# --- the synthetic Romero-first fixture -------------------------------------------------


def test_the_fixture_states_the_chain_in_its_records(romero):
    ods = romero.ods
    assert ods[f"{CP}.t_i_average_fit.parameters"].startswith(TI_EQUALS_TE_RECORD)
    np.testing.assert_array_equal(ods[f"{CP}.t_i_average"], romero.t_e)
    np.testing.assert_array_equal(ods["core_profiles.global_quantities.z_eff_resistive"], [SYNTHETIC_ZEFF_ROMERO])
    for k in (1, 2):  # C and O, through the real writer
        record = ods[f"{CP}.ion.{k}.density_fit.parameters"]
        assert record.startswith("origin=inferred; method=romero_partition") and "assumption=n_C=n_O" in record
    np.testing.assert_allclose(romero.notes["zeff_mean"], SYNTHETIC_ZEFF_ROMERO, rtol=1e-10)
    assert "resistive_method=romero" in ods[f"{CP}.zeff_fit.parameters"]
    assert "SYNTHETIC" in ods["core_profiles.code.name"]
    # n_C = n_O: equal elemental densities.
    np.testing.assert_allclose(ods[f"{CP}.ion.1.density"], ods[f"{CP}.ion.2.density"], rtol=1e-12)


def test_the_fixture_roles_read_back_as_the_chain(romero):
    roles = {q: record["role"] for q, (_, record) in _traces(_extract(romero.ods)).items()}
    assert roles["T_i"] == "assumed" and roles["p_i"] == "assumed"
    assert roles["n_impurity_ion"] == roles["n_main_ion"] == roles["Z_eff"] == "inferred"
    assert roles["p_kin"] == "display sum"  # T_i assumed, not a pressure partition: a test, not an identity
    assert roles["p_e_thomson"] == "independent validation"


# --- stage cuts ------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", range(1, 9))
def test_a_stage_draws_exactly_the_quantities_up_to_it(romero, stage):
    full = set(_quantities(_extract(romero.ods)))
    cut = _extract(romero.ods, stage=stage)
    assert set(_quantities(cut)) == {q for q in full if QUANTITY_STAGES[q] <= stage}
    assert f"Stage {stage}/8" in cut.suptitle


def test_stage_one_is_the_data_and_stage_four_adds_the_reference(romero):
    assert set(_quantities(_extract(romero.ods, stage=1))) == {"p_eq", "p_e_thomson", "n_e_thomson", "T_e_thomson"}
    added = set(_quantities(_extract(romero.ods, stage=4))) - set(_quantities(_extract(romero.ods, stage=3)))
    assert added == {"Z_eff_reference"}


def test_every_stage_draws_its_traces_as_the_full_figure_does(romero):
    """No stage is faded: a cut only leaves later stages out."""
    full = _traces(_extract(romero.ods))
    for quantity, (series, record) in _traces(_extract(romero.ods, stage=6)).items():
        assert dict(series.style) == dict(full[quantity][0].style), quantity
        assert "alpha" not in series.style
        assert record["emphasis"] == ("newest" if QUANTITY_STAGES[quantity] == 6 else "earlier")


def test_thomson_points_are_small_with_capped_bars(romero, archive):
    for case in (romero, archive):
        traces = _traces(_extract(case.ods))
        for quantity in ("p_e_thomson", "n_e_thomson", "T_e_thomson"):
            series = traces[quantity][0]
            assert series.yerr is not None
            assert {k: series.style[k] for k in THOMSON_POINT_STYLE} == THOMSON_POINT_STYLE, quantity
    figure, axes = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, format="slide")
    (container,) = [c for c in np.asarray(axes)[1, 0].containers if type(c).__name__ == "ErrorbarContainer"]
    data_line, caps, bars = container.lines
    assert data_line.get_markersize() == pytest.approx(0.4 * 6.0 * 1.8)  # slide: 1.8 x Matplotlib's 6 pt
    assert caps and bars[0].get_linewidth()[0] == pytest.approx(1.3)


def test_every_stage_keeps_the_full_figures_limits(romero):
    limits = [[panel.y_limits for panel in _extract(romero.ods, stage=k).models] for k in range(1, 9)]
    assert all(row == limits[0] for row in limits)
    assert all(lim is not None for lim in limits[0])
    full = _extract(romero.ods)
    assert all(panel.y_limits is None for panel in full.models)  # the static figure still autoscales
    for panel, (low, high) in zip(full.models, limits[0]):
        for series in panel.series:
            finite = series.y[np.isfinite(series.y)]
            assert not finite.size or (low <= finite.min() and finite.max() <= high)


def test_each_stage_legend_names_only_what_the_stage_draws(romero):
    """No room is reserved for traces not drawn yet; the canvas stays the slide's."""
    for stage in (1, 4, 8):
        figure, axes = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, stage=stage, format="slide")
        assert figure.get_size_inches().tolist() == pytest.approx([11.0, 5.5], abs=0.6)
        model = _extract(romero.ods, stage=stage)
        for axis, panel in zip(np.asarray(axes).ravel(), model.models):
            legend = axis.get_legend()
            texts = [t.get_text() for t in legend.get_texts()] if legend else []
            assert texts == [s.label for s in panel.series if s.label] or sorted(texts) == sorted(
                s.label for s in panel.series), (stage, texts)
            assert all(t.get_alpha() in (None, 1.0) for t in (legend.get_texts() if legend else ()))


def test_a_stage_with_nothing_stored_says_so(archive):
    ods = copy.deepcopy(archive.ods)
    ods[f"{CP}.zeff_fit.parameters"] = "origin=derived; method=impurity_model_preset"
    assert "(nothing stored)" in _extract(ods, stage=4).suptitle


@pytest.mark.parametrize("value", [0, 9, 2.5, True, "3", -1])
def test_an_invalid_stage_is_refused(romero, value):
    with pytest.raises(ValueError, match="stage must be an inference stage"):
        _extract(romero.ods, stage=value)


# --- assumptions -------------------------------------------------------------------------------


def test_assumptions_are_read_from_the_records(romero):
    model = _extract(romero.ods)
    assert model.suptitle.endswith(
        "\nAssumed: $T_i = T_e$; $n_C = n_O$"
        "\nInferred: $Z_{\\mathrm{eff}}^{\\mathrm{res}} = 2.00 \\pm 0.20$ (Romero)")
    stated = model.models[0].metadata["assumptions"]
    assert [a["statement"] for a in stated] == ["T_i = T_e", "n_C = n_O"]
    assert stated[1]["record"].endswith("ion.1.density_fit.parameters")  # the stage that first relies on it
    (inferred,) = model.models[0].metadata["inferences"]
    assert (inferred["quantity"], inferred["statement"]) == ("Z_eff_reference", "Z_eff^res = 2.00 ± 0.20 (Romero)")
    assert inferred["record"].endswith("ion.1.density_fit.parameters")  # a record stating the normalization
    # A cut names only what its stages assumed or inferred.
    assert "n_C" not in _extract(romero.ods, stage=4).suptitle
    assert "Inferred" in _extract(romero.ods, stage=4).suptitle
    assert "Inferred" not in _extract(romero.ods, stage=3).suptitle
    assert "n_C = n_O" in _extract(romero.ods, stage=5).suptitle
    assert "Assumed" not in _extract(romero.ods, stage=2).suptitle


def test_a_preset_composition_lists_its_target_as_assumed(archive):
    model = _extract(archive.ods)
    assert model.suptitle.endswith(r"Assumed: $\langle Z_{\mathrm{eff}}\rangle_{n_e} = 2.00$ (target)")
    assert [a["statement"] for a in model.models[0].metadata["assumptions"]] == ["<Z_eff>_ne = 2.00 (target)"]
    assert model.models[0].metadata["inferences"] == [] and "Inferred" not in model.suptitle
    reference, record = _traces(model)["Z_eff_reference"]
    assert reference.label == r"target $\langle Z_{\mathrm{eff}}\rangle = 2.00$ (assumed)"
    assert record["role"] == "assumed"


def _with_normalization(ods, text):
    """``ods`` whose composition and zeff records state ``text`` as their normalization fields."""
    ods = copy.deepcopy(ods)
    for path in (f"{CP}.ion.0.density_fit.parameters", f"{CP}.ion.1.density_fit.parameters",
                 f"{CP}.ion.2.density_fit.parameters", f"{CP}.zeff_fit.parameters"):
        ods[path] = f"origin=derived; method=openadas_transient; {text}"
    return ods


@pytest.mark.parametrize("fields, kind, statement, caption", [
    ("normalization=ne_weighted_mean; target_zeff=2", "assumptions", "<Z_eff>_ne = 2.00 (target)",
     r"$\langle Z_{\mathrm{eff}}\rangle_{n_e} = 2.00$ (target)"),
    ("normalization=axis; target_zeff=1.5", "assumptions", "Z_eff(0) = 1.50 (target)",
     r"$Z_{\mathrm{eff}}(0) = 1.50$ (target)"),
    ("normalization=fixed; target_zeff=2", "assumptions", "Z_eff^stripped = 2.00 (target)",
     r"$Z_{\mathrm{eff}}^{\mathrm{stripped}} = 2.00$ (target)"),
    ("normalization=resistive_closure; resistive_target=1.7403228517823983; resistive_method=romero",
     "inferences", "Z_eff^res = 1.74 (Romero)", r"$Z_{\mathrm{eff}}^{\mathrm{res}} = 1.74$ (Romero)"),
    ("normalization=resistive_closure; resistive_target=1.74; resistive_sigma=0.2148; resistive_method=romero",
     "inferences", "Z_eff^res = 1.74 ± 0.21 (Romero)",
     r"$Z_{\mathrm{eff}}^{\mathrm{res}} = 1.74 \pm 0.21$ (Romero)"),
    ("normalization=resistive_closure; resistive_method=romero", "inferences",
     "Z_eff^res closure (Romero; value not recorded)",
     r"$Z_{\mathrm{eff}}^{\mathrm{res}}$ closure (Romero; value not recorded)"),
])
def test_each_normalization_names_the_zeff_it_closes_on(romero, fields, kind, statement, caption):
    model = _extract(_with_normalization(romero.ods, fields))
    meta = model.models[0].metadata
    assert statement in [a["statement"] for a in meta[kind]]
    other = "inferences" if kind == "assumptions" else "assumptions"
    assert not any("Z_eff" in a["statement"] for a in meta[other])
    assert caption in model.suptitle


def test_the_constraint_is_never_paired_with_a_value_no_record_states(romero):
    """The stored z_eff_resistive is the reference line, not the closure's value."""
    ods = _with_normalization(romero.ods, "normalization=resistive_closure; resistive_method=romero")
    ods["core_profiles.global_quantities.z_eff_resistive"] = [1.7403228517823983]
    assert [a["statement"] for a in _extract(ods).models[0].metadata["inferences"]] == [
        "Z_eff^res closure (Romero; value not recorded)"]
    ods = _with_normalization(romero.ods, "normalization=ne_weighted_mean")  # no target recorded
    meta = _extract(ods).models[0].metadata
    assert not any("Z_eff" in a["statement"] for a in meta["assumptions"] + meta["inferences"])


@pytest.mark.parametrize("sigma, legend", [
    (None, r"$Z_{\mathrm{eff}}^{\mathrm{res}} = 1.74$ (stored, Romero)"),
    ("0.2148", r"$Z_{\mathrm{eff}}^{\mathrm{res}} = 1.74 \pm 0.21$ (stored, Romero)"),
])
def test_the_reference_label_has_two_decimals_and_its_sigma(romero, sigma, legend):
    ods = copy.deepcopy(romero.ods)
    ods["core_profiles.global_quantities.z_eff_resistive"] = [1.7403228517823983]
    ods[f"{CP}.zeff_fit.parameters"] = "origin=inferred; method=romero_partition; resistive_method=romero" + (
        f"; resistive_sigma={sigma}" if sigma else "")
    assert _traces(_extract(ods))["Z_eff_reference"][0].label == legend


def test_an_assumption_is_never_hard_coded(romero):
    ods = copy.deepcopy(romero.ods)
    for k in (1, 2):
        ods[f"{CP}.ion.{k}.density_fit.parameters"] = "origin=inferred; method=romero_partition; assumption=n_C=2n_O"
    ods[f"{CP}.zeff_fit.parameters"] = "origin=inferred; method=romero_partition"
    assert "Assumed: $T_i = T_e$; $n_C = 2n_O$" in _extract(ods).suptitle


@pytest.mark.parametrize("record, role, statement", [
    ("ti_te_ratio=1; status=assumed", "assumed", "T_i = T_e"),
    ("ti_te_ratio=0.5; status=assumed", "assumed", "T_i = 0.5 T_e"),
    ("origin=assumed", "assumed", "T_i (assumed)"),
    ("origin=assumed; method=ti_equals_te", "assumed", "T_i (ti_equals_te)"),
    ("origin=inferred; method=equilibrium_pressure_partition", "inferred", None),
])
def test_the_note_lists_exactly_the_legends_assumed_entries(romero, record, role, statement):
    ods = copy.deepcopy(romero.ods)
    for path in (f"{CP}.t_i_average_fit.parameters", f"{CP}.ion.0.temperature_fit.parameters"):
        ods[path] = record
    model = _extract(ods)
    assert _traces(model)["T_i"][1]["role"] == role
    stated = [a["statement"] for a in model.models[0].metadata["assumptions"] if a["quantity"] == "T_i"]
    assert stated == ([] if statement is None else [statement])


def test_a_derived_record_is_drawn_derived_and_not_listed(archive):
    model = _extract(archive.ods)
    traces = _traces(model)
    assert {q: traces[q][1]["role"] for q in ("n_main_ion", "n_impurity_ion", "Z_eff")} == dict.fromkeys(
        ("n_main_ion", "n_impurity_ion", "Z_eff"), "derived")
    assert all("(derived)" in traces[q][0].label for q in ("n_main_ion", "n_impurity_ion", "Z_eff"))
    assert [a["statement"] for a in model.models[0].metadata["assumptions"]] == ["<Z_eff>_ne = 2.00 (target)"]


def test_a_ratio_prior_is_listed_as_assumed():
    case = _quiet(kinetic_state_case, 39915, "electron_kinetic")
    stated = [a["statement"] for a in _extract(case.ods).models[0].metadata["assumptions"]]
    assert stated[0].startswith("T_i = ") and stated[0].endswith(" T_e")


# --- impurity pressure ---------------------------------------------------------------------------


def test_the_impurity_pressure_is_drawn_as_part_of_p_i(romero):
    traces = _traces(_extract(romero.ods))
    p_imp, record = traces["p_imp"]
    np.testing.assert_allclose(p_imp.y, romero.notes["p_imp"], rtol=1e-12)
    assert record["role"] == "stored" and record["path"].count(".pressure") == 2
    assert "part of" in p_imp.label
    np.testing.assert_allclose(traces["p_kin"][0].y, traces["p_e"][0].y + traces["p_i"][0].y, rtol=1e-12)
    np.testing.assert_allclose(traces["p_i"][0].y, romero.p_i, rtol=1e-12)


def test_without_a_stored_impurity_pressure_it_is_omitted_and_named(archive):
    series, record = _traces(_extract(archive.ods))["p_imp"]
    assert record["role"] == "unavailable" and not np.isfinite(series.y).any()
    assert "not stored" in series.label


# --- layout="stages" --------------------------------------------------------------------------


def test_the_stages_layout_puts_one_column_per_stage(romero):
    model = _extract(romero.ods, layout="stages")
    columns = len(DEFAULT_STAGE_COLUMNS)
    assert (model.nrows, model.ncols) == (4, columns + 1)
    titles = [m.title for m in model.models[:columns]]
    assert titles[0].startswith("1: ") and titles[-1] == "8: closure"
    for row in range(4):
        cells = model.models[row * (columns + 1): (row + 1) * (columns + 1)]
        assert type(cells[-1]).__name__ == "TextPanel"  # the row's legend cell
        assert len({c.y_limits for c in cells[:-1]}) == 1  # one range per row
        for cell, stage in zip(cells[:-1], DEFAULT_STAGE_COLUMNS):
            assert cell.metadata["stage"] == stage
            assert {t["quantity"] for t in cell.metadata["traces"]} <= {
                q for q, s in QUANTITY_STAGES.items() if s <= stage}
    assert "Assumed: $T_i = T_e$; $n_C = n_O$" in model.suptitle


def test_the_stages_layout_renders_one_legend_per_row_on_a_slide(romero):
    figure, axes = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, layout="stages", format="slide",
                          stages=(1, 4, 8))
    grid = np.asarray(axes)
    assert grid.shape == (4, 4)
    assert all(axis.get_legend() is not None for axis in grid[:, -1])
    assert all(axis.get_legend() is None for axis in grid[:, :-1].ravel())
    assert [axis.get_title() for axis in grid[0, :-1]] == ["1: data", r"4: $Z_{\mathrm{eff}}$ ref.", "8: closure"]
    labels = [t.get_text() for t in grid[0, -1].get_legend().get_texts()]
    assert r"$p_{\mathrm{eq}}$" in labels and "TS span" not in labels
    figure.canvas.draw()
    assert grid[0, 0].get_window_extent().height > 0.12 * figure.bbox.height  # rows are not squeezed


@pytest.mark.parametrize("options, match", [
    ({"layout": "stages", "stage": 3}, "not stage="),
    ({"stages": (1, 2)}, "layout='stages'"),
    ({"layout": "stages", "stages": (4, 2)}, "increasing"),
    ({"layout": "stages", "stages": (1, 9)}, "inference stage"),
    ({"layout": "stages", "stages": 4}, "sequence of stages"),
])
def test_stage_columns_are_validated(romero, options, match):
    with pytest.raises(ValueError, match=match):
        _extract(romero.ods, **options)


# --- the animation contract: a categorical driver -----------------------------------------


def test_the_stage_is_the_plots_categorical_slice_control(romero):
    record = describe_one(NAME, normalize_entries(romero.ods))
    assert record.sequence["kind"] == "categorical" and record.sequence["selected"] is None
    (control,) = [c for c in controls_for(record) if c.group == "slice"]
    assert (control.name, control.kind, control.default) == ("stage", "choice", None)
    assert control.options == tuple(range(1, 9))
    assert control.labels[3] == "Stage 4/8: Z_eff reference" and len(control.labels) == len(STAGES)
    coordinate, unit, values = sequence_values(NAME, normalize_entries(romero.ods), "stage")
    assert (coordinate, unit, values.tolist()) == ("stage", None, list(range(1, 9)))


def test_animation_walks_the_eight_stages_at_the_selected_time(romero, tmp_path):
    animation = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, time=romero.time, animation=True,
                       fps=1, dpi=DPI)
    assert len(animation) == 8 and animation.driver.indices == tuple(range(1, 9))
    assert animation.driver.labels[0] == "Stage 1/8: Thomson and equilibrium"
    assert animation.metadata["driver"]["labels"][-1] == "Stage 8/8: pressure closure"
    assert animation.metadata["options"]["time"] == pytest.approx(romero.time)
    frames = _quiet(lambda: list(animation.frames()))
    assert len(frames) == 8 and len({f.shape for f in frames}) == 1
    limits, _ = animation._layout()
    full = [panel.y_limits for panel in _extract(romero.ods, stage=1).models]
    for number, lim in enumerate(full):
        assert limits[number][1] == pytest.approx(lim)
    path = _quiet(animation.save, tmp_path / "stages.gif")
    from PIL import Image

    with Image.open(path) as written:
        assert written.n_frames == 8
    sidecar = json.loads((tmp_path / "stages.gif.json").read_text())
    assert sidecar["driver"]["name"] == "stage" and sidecar["presentation"]["fps"] == 1


def test_animation_takes_a_subset_and_refuses_what_is_not_a_stage(romero):
    animation = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, stage=[1, 4, 8], animation=True)
    assert animation.driver.indices == (1, 4, 8) and "Stage 4/8" in animation._stamp(1)
    for value, match in (([1, 9], "not offered"), (4, "one state is a static plot"), ([2, 2], "repeats"),
                         ([3], "one state is a static plot")):
        with pytest.raises(ValueError, match=match):
            _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, stage=value, animation=True)
    with pytest.raises(ValueError, match="not times"):
        _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, time_range=(0.3, 0.4), animation=True)


def test_the_thomson_pressure_is_e_n_t_at_every_stage(romero):
    series, _ = _traces(_extract(romero.ods, stage=1))["p_e_thomson"]
    order = np.argsort(romero.ts_rho)
    np.testing.assert_allclose(series.y, (E * romero.ts_n_e * romero.ts_t_e)[order], rtol=1e-12)


def _legend_overlaps(figure, axes):
    """``(legend panel, other panel, artist)`` where a legend box meets another panel or its y labels."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panels = list(np.asarray(axes).ravel())
    found = []
    for i, axis in enumerate(panels):
        legend = axis.get_legend()
        if legend is None:
            continue
        box = legend.get_window_extent(renderer)
        for j, other in enumerate(panels):
            if other is axis:
                continue
            for artist in (other, other.yaxis.label, *other.get_yticklabels()):
                extent = artist.get_window_extent(renderer)
                if extent.width and box.overlaps(extent):
                    found.append((i, j, type(artist).__name__))
    return found


@pytest.mark.parametrize("options", [{"stage": 1}, {"stage": 5}, {"stage": 8}, {}, {"layout": "stack"}])
def test_no_legend_overlaps_a_neighbouring_panel_on_a_slide(romero, archive, options):
    """Long role labels with a four-line caption (the 39916 chain-B render) once ran into the density label."""
    ods = copy.deepcopy(romero.ods)
    ods[f"{CP}.zeff_fit.parameters"] += "; resistive_sigma=0.21"  # the longest caption the chain writes
    for case in (ods, archive.ods):
        figure, axes = _quiet(vaft.omas.plot_kinetic_overview_state, case, format="slide", **options)
        assert _legend_overlaps(figure, axes) == [], options
        plt.close(figure)


def test_nothing_is_listed_without_an_assumed_record():
    model = _extract(_quiet(kinetic_state_case, 40326, "magnetics", records=False).ods)
    meta = model.models[0].metadata
    assert "Assumed" not in model.suptitle and meta["assumptions"] == [] and meta["inferences"] == []


# --- cold-review findings -------------------------------------------------------------------


@pytest.mark.parametrize("uncertainty", ["none", "band", "auto", "errorbar"])
def test_thomson_bar_styles_never_reach_a_plain_line(romero, uncertainty):
    """elinewidth/capsize belong to errorbar(): plot() refuses them (cold review blocker)."""
    figure, _ = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, format="slide", uncertainty=uncertainty)
    plt.close(figure)


def test_thomson_without_stored_sigma_draws(romero):
    ods = copy.deepcopy(romero.ods)
    for c in range(len(ods["thomson_scattering.channel"])):
        for signal in ("n_e", "t_e"):
            del ods[f"thomson_scattering.channel.{c}.{signal}.data_error_upper"]
    assert _traces(_extract(ods))["n_e_thomson"][0].yerr is None
    for options in ({}, {"stage": 1}, {"layout": "stages", "format": "slide"}):
        figure, _ = _quiet(vaft.omas.plot_kinetic_overview_state, ods, **options)
        plt.close(figure)


def test_the_stages_layout_refuses_plotly(romero):
    pytest.importorskip("plotly")
    with pytest.raises(NotImplementedError, match="Matplotlib only: layout='stages'"):
        _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, layout="stages", backend="plotly")


def test_the_stages_layout_cannot_be_animated(romero):
    with pytest.raises(ValueError, match="layout='stages' cannot be animated over stage"):
        _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, layout="stages", animation=True)


@pytest.mark.parametrize("fmt", ["single_column", "double_column", "screen"])
def test_the_stages_layout_refuses_a_narrow_canvas(romero, fmt):
    with pytest.raises(ValueError, match="at least 8 in wide"):
        _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, layout="stages", format=fmt)
    with pytest.raises(ValueError, match="at least 8 in wide"):
        _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, layout="stages", figsize=(6, 5))


def test_the_full_figure_can_be_chosen_again_after_a_stage(romero):
    from vaft.plot.navigation import ControlState

    record = describe_one(NAME, normalize_entries(romero.ods))
    state = ControlState([c for c in controls_for(record) if c.group == "slice"])
    state.set("stage", 4)
    assert state.as_options() == {"stage": 4}
    state.set("stage", None)
    assert state.as_options() == {}


def test_stage_frames_carry_no_stamp_over_the_caption(romero):
    animation = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, stage=[1, 8], animation=True, dpi=DPI)
    assert animation.frame_label is False and animation.metadata["presentation"]["frame_label"] is False
    figure = _quiet(animation._figure, 0)
    assert not [t for t in figure.texts if t is not figure._suptitle]  # the caption alone names the stage
    plt.close(figure)


def test_the_impurity_pressure_is_part_of_p_i_only_when_p_i_is_drawn(romero):
    assert "part of" in _traces(_extract(romero.ods))["p_imp"][0].label
    ods = copy.deepcopy(romero.ods)
    del ods[f"{CP}.pressure_ion_total"]
    traces = _traces(_extract(ods))
    assert "p_i" not in traces and traces["p_imp"][0].label == r"$p_{\mathrm{imp}}$ (stored)"
