"""``kinetic_overview_state`` as an inference chain: stages, assumptions, animation (issue #1957).

The chain is the maintainer's: equilibrium and Thomson in, ``T_i = T_e`` and
``n_C = n_O`` assumed, the Romero ``Z_eff`` first, then ``n_I,eff``, ``n_i``,
``Z_eff(rho)`` and the pressure closure.  The fixture is
``romero_first_case`` of ``workflow/kinetic_state/archive_to_ods.py``: the
40326 archive's equilibrium and Thomson with a **synthetic** Romero value,
written through the real composition writer, until Lane KP's stages (#1842)
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
    # The span is stage-1 data: it is there from the first stage.
    assert _extract(romero.ods, stage=1).models[0].reference_lines


def test_every_stage_draws_its_traces_as_the_full_figure_does(romero):
    """No stage is faded (user review on #1957): a cut only leaves later stages out."""
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


def test_a_sequence_of_stages_keeps_one_layout(romero):
    positions = []
    for stage in (1, 4, 8):
        figure, axes = _quiet(vaft.omas.plot_kinetic_overview_state, romero.ods, stage=stage, format="slide")
        positions.append(np.array([a.get_position().bounds for a in np.asarray(axes).ravel()]))
        assert figure.get_size_inches().tolist() == pytest.approx([11.0, 5.5], abs=0.6)
    for other in positions[1:]:
        np.testing.assert_allclose(other, positions[0], atol=3e-3)


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
        r"Assumed: $T_i = T_e$; $n_C = n_O$; $\langle Z_{\mathrm{eff}}\rangle = 2$ (Romero)")
    stated = model.models[0].metadata["assumptions"]
    assert [a["statement"] for a in stated] == ["T_i = T_e", "n_C = n_O", "<Z_eff> = 2 (Romero)"]
    assert stated[2]["record"].endswith("ion.1.density_fit.parameters")  # normalization=resistive_closure
    assert stated[1]["record"].endswith("ion.1.density_fit.parameters")  # the stage that first relies on it
    # A cut names only what its stages assumed.
    assert "n_C" not in _extract(romero.ods, stage=4).suptitle
    assert "n_C = n_O" in _extract(romero.ods, stage=5).suptitle
    assert _extract(romero.ods, stage=2).suptitle.endswith("Assumed: none")  # one line count per stage


def test_a_preset_composition_lists_its_target_and_nothing_else(archive):
    model = _extract(archive.ods)
    assert model.suptitle.endswith(r"Assumed: $\langle Z_{\mathrm{eff}}\rangle = 2$ (target)")
    assert [a["statement"] for a in model.models[0].metadata["assumptions"]] == ["<Z_eff> = 2 (target)"]


def test_nothing_is_listed_without_an_assumed_record():
    model = _extract(_quiet(kinetic_state_case, 40326, "magnetics", records=False).ods)
    assert "Assumed" not in model.suptitle and model.models[0].metadata["assumptions"] == []


def test_the_romero_constraint_is_read_from_the_records(romero):
    ods = copy.deepcopy(romero.ods)
    ods["core_profiles.global_quantities.z_eff_resistive"] = [1.7]
    ods[f"{CP}.zeff_fit.parameters"] = ods[f"{CP}.zeff_fit.parameters"].replace("romero", "spitzer")
    assert "<Z_eff> = 1.7 (Spitzer)" in [a["statement"] for a in
                                         _extract(ods).models[0].metadata["assumptions"]]
    for k in (1, 2):
        ods[f"{CP}.ion.{k}.density_fit.parameters"] += "; resistive_target=1.9"
    assert "<Z_eff> = 1.9 (Spitzer)" in [a["statement"] for a in
                                         _extract(ods).models[0].metadata["assumptions"]]


def test_an_assumption_is_never_hard_coded(romero):
    ods = copy.deepcopy(romero.ods)
    for k in (1, 2):
        ods[f"{CP}.ion.{k}.density_fit.parameters"] = "origin=inferred; method=romero_partition; assumption=n_C=2n_O"
    ods[f"{CP}.zeff_fit.parameters"] = "origin=inferred; method=romero_partition"
    assert "Assumed: $T_i = T_e$; $n_C = 2n_O$" in _extract(ods).suptitle
    assert "(Resistive)" in _extract(ods).suptitle  # resistive_closure stated, its method no longer


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
    assert r"$p_{\mathrm{eq}}$" in labels and "TS span" in labels
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


# --- the #1049 contract: a categorical driver ----------------------------------------------


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
