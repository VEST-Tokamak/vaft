"""``kinetic_overview_state``: one matched kinetic state from stored paths (issue #1837).

The fixtures are the #1839 archive cases rebuilt by
``workflow/kinetic_state/archive_to_ods.py`` through the real writers
(``populate_radial_impurity_profiles``, ``infer_ti_pressure_partition``), so the
roles the plot shows come from records a writer produced, not from the test.
The numbers checked are the archive notebooks' own: 117/129 finite ``T_i`` and
``Z_eff(0) = 2.022`` for 40326/322 ms, 128/129 and ``Z_eff(0) = 1.328`` for
39915/316 ms, ``p_e + p_i = p_eq`` where ``T_i`` is finite, and the Thomson
pressure ``e n_e T_e`` with its independent 1 sigma.
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

import vaft  # noqa: E402
import vaft.omas  # noqa: E402
from vaft.plot.backend.kinetic_state import p_kin_role  # noqa: E402
from vaft.plot.backend.recipes import missing_required_path  # noqa: E402

from _kinetic_state_fixture import E, kinetic_state_case, load_archive  # noqa: E402

NAME = "kinetic_overview_state"
CP = "core_profiles.profiles_1d.0"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _case(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinetic_state_case(*args, **kwargs)


@pytest.fixture(scope="module")
def case_40326():
    return _case(40326, "magnetics")


@pytest.fixture(scope="module")
def case_39915_kinetic():
    return _case(39915, "electron_kinetic")


@pytest.fixture
def ods(case_40326):
    return copy.deepcopy(case_40326.ods)


def _extract(ods, **options):
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return vaft.omas.extract_kinetic_overview_state(ods, **options)


def _traces(model):
    """``{quantity: (series, trace record)}`` across the four panels."""
    out = {}
    for panel in model.models:
        records = panel.metadata["traces"]
        assert len(records) == len(panel.series)
        for series, record in zip(panel.series, records):
            assert series.label == record["label"]
            out[record["quantity"]] = (series, record)
    return out


# --- rendering ---------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["slide", "single_column", None])
@pytest.mark.parametrize("layout, shape", [("grid", (2, 2)), ("stack", (4, 1))])
def test_40326_renders_both_layouts_in_the_claimed_formats(case_40326, layout, shape, fmt):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure, axes = vaft.omas.plot_kinetic_overview_state(case_40326.ods, layout=layout, format=fmt)
    assert np.asarray(axes).shape == shape
    legends = [axis.get_legend() for axis in np.asarray(axes).ravel()]
    labels = [text.get_text() for legend in legends if legend for text in legend.get_texts()]
    assert any("closure identity" in label for label in labels)
    assert any("independent validation" in label for label in labels)
    assert "322.0 ms" in figure._suptitle.get_text()
    # Beside the panels on a wide canvas, inside them on a single column.
    figure.canvas.draw()
    outside = fmt != "single_column"
    for axis, legend in zip(np.asarray(axes).ravel(), legends):
        beside = legend.get_window_extent().x0 >= axis.get_window_extent().x1 - 1
        assert beside == outside


def test_the_layout_changes_the_grid_not_the_panels(case_40326):
    grid, stack = _extract(case_40326.ods, layout="grid"), _extract(case_40326.ods, layout="stack")
    assert (grid.nrows, grid.ncols, stack.nrows, stack.ncols) == (2, 2, 4, 1)
    assert [m.metadata["panel"] for m in grid.models] == ["pressure", "density", "temperature", "zeff"]
    for a, b in zip(grid.models, stack.models):
        assert [s.label for s in a.series] == [s.label for s in b.series]


def test_plotly_draws_the_state_and_its_measured_span(case_40326):
    pytest.importorskip("plotly")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = vaft.omas.plot_kinetic_overview_state(case_40326.ods, backend="plotly")
    assert len(figure.data) >= 10
    rects = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    span = _extract(case_40326.ods).models[0].reference_lines[0]
    assert any(np.isclose(r.x0, span.x) and np.isclose(r.x1, span.x_end) for r in rects)


def test_the_span_reaches_xarray(case_40326):
    pytest.importorskip("xarray")
    lines = _extract(case_40326.ods).models[0].to_xarray().attrs["reference_lines"]
    lines = json.loads(lines) if isinstance(lines, str) else lines
    assert "x_end" in lines[0]


# --- the archive notebook's numbers -------------------------------------------------


def test_40326_matches_the_archive_notebook(case_40326):
    s = load_archive(40326)
    traces = _traces(_extract(case_40326.ods))
    ti = traces["T_i"][0]
    assert int(np.isfinite(ti.y).sum()) == 117  # "Physical magnetic-EFIT Ti grid points: 117 / 129"
    np.testing.assert_allclose(ti.y, case_40326.t_i, rtol=0, atol=0, equal_nan=True)
    assert "117/129 finite" in ti.label

    zeff = traces["Z_eff"][0]
    defined = case_40326.valid_charge
    np.testing.assert_allclose(zeff.y[defined], case_40326.zeff[defined], rtol=1e-12)
    np.testing.assert_allclose(zeff.y[0], s["impurity_model"]["atlas_zeff_axis"], rtol=1e-12)
    np.testing.assert_allclose(np.interp(0.9, zeff.x[defined], zeff.y[defined]),
                               s["impurity_model"]["atlas_zeff_rho09"], rtol=1e-12)
    np.testing.assert_array_equal(traces["Z_eff_reference"][0].y, [2.0, 2.0])

    # Densities as the writer stored them: diluted main ion, charged C + O.
    np.testing.assert_allclose(traces["n_main_ion"][0].y[defined] * 1e19, case_40326.n_h[defined], rtol=1e-12)
    np.testing.assert_allclose(traces["n_impurity_ion"][0].y[defined] * 1e19, case_40326.n_imp[defined],
                               rtol=1e-12)
    assert traces["n_impurity_ion"][1]["path"].endswith("state[:].density")

    # The closure: p_e + p_i reproduces p_eq where T_i is finite, by construction.
    p_kin, p_eq = traces["p_kin"][0], traces["p_eq"][0]
    np.testing.assert_allclose(p_eq.x, p_kin.x, atol=1e-10)  # one grid for this case
    valid = np.isfinite(p_kin.y)
    assert valid.sum() == 117
    np.testing.assert_allclose(p_kin.y[valid], p_eq.y[valid], rtol=1e-12, atol=1e-12)

    ts = s["mapped_thomson"]["magnetics"]
    order = np.argsort(ts["rho_tor_norm"])
    p_ts = traces["p_e_thomson"][0]
    np.testing.assert_allclose(p_ts.x, np.asarray(ts["rho_tor_norm"])[order], atol=2e-4)
    np.testing.assert_allclose(p_ts.y, np.asarray(ts["p_e_Pa"])[order], rtol=1e-12)
    np.testing.assert_allclose(p_ts.yerr, case_40326.ts_sigma_p_e[order], rtol=1e-12)
    n_ts = traces["n_e_thomson"][0]
    np.testing.assert_allclose(n_ts.y * 1e19, np.asarray(ts["electron_density_m3"])[order], rtol=1e-12)
    np.testing.assert_allclose(n_ts.yerr * 1e19, np.asarray(ts["electron_density_error_m3"])[order], rtol=1e-12)


def test_39915_magnetic_case_matches_its_notebook():
    case = _case(39915, "magnetics")
    traces = _traces(_extract(case.ods))
    assert int(np.isfinite(traces["T_i"][0].y).sum()) == 128  # "Physical T_i points: 128/129"
    np.testing.assert_allclose(traces["Z_eff"][0].y[0], 1.3276842011353556, rtol=1e-12)
    finite = np.isfinite(traces["p_kin"][0].y)
    np.testing.assert_allclose(traces["p_kin"][0].y[finite], traces["p_eq"][0].y[finite], rtol=1e-12, atol=1e-12)


# --- roles from the writers' records --------------------------------------------------


def test_the_roles_are_the_records_the_writers_wrote(case_40326):
    model = _extract(case_40326.ods)
    traces = _traces(model)
    expected = {
        "n_e": "fit", "T_e": "fit",                     # coordinate=...; method=external
        "n_main_ion": "assumed", "n_impurity_ion": "assumed", "Z_eff": "assumed",   # origin=derived
        "T_i": "inferred", "p_i": "inferred",           # origin=inferred (inferred_ti_text)
        "p_kin": "closure identity",                    # method=equilibrium_pressure_partition
        "n_e_thomson": "fit input", "T_e_thomson": "fit input",
        "p_e_thomson": "independent validation",        # equilibrium_lineage=magnetics
        "p_eq": "stored", "p_e": "stored",
    }
    assert {q: traces[q][1]["role"] for q in expected} == expected
    meta = model.models[0].metadata
    assert (meta["equilibrium_lineage"], meta["equilibrium_occurrence"], meta["equilibrium_occurrence_verified"]) \
        == ("magnetics", 0, True)


def test_39915_kinetic_efit_shows_its_prior_and_infers_nothing(case_39915_kinetic):
    model = _extract(case_39915_kinetic.ods)
    traces = _traces(model)
    assert traces["T_i"][1]["role"] == "assumed"  # the EFIT's own T_i = T_e prior, never re-inferred
    np.testing.assert_allclose(traces["T_i"][0].y, case_39915_kinetic.t_e)
    assert "p_i" not in traces and "p_kin" not in traces
    assert traces["p_e_thomson"][1]["role"] == "fit input"  # TS constrained this EFIT
    np.testing.assert_allclose(traces["p_eq"][0].y.min(), -0.845745309, atol=1e-10)
    assert model.models[0].metadata["equilibrium_lineage"] == "electron_kinetic"


def test_non_finite_t_i_points_stay_empty(ods):
    ti = np.asarray(ods[f"{CP}.t_i_average"], float).copy()
    ti[:5] = np.inf
    ods[f"{CP}.t_i_average"] = ti
    assert not np.isfinite(_traces(_extract(ods))["T_i"][0].y[:5]).any()


@pytest.mark.parametrize("options", [{"records": False}, {"lineage_fields": False}])
def test_absent_records_are_labelled_stored_never_guessed(options):
    model = _extract(_case(40326, "magnetics", **options).ods)
    traces = _traces(model)
    if not options.get("records", True):
        for quantity in ("n_e", "n_main_ion", "n_impurity_ion", "T_e", "T_i", "p_i", "Z_eff"):
            assert traces[quantity][1]["role"] == "stored", quantity
            assert "(stored" in traces[quantity][0].label, quantity
        assert traces["p_kin"][1]["role"] == "display sum"
        assert traces["n_e_thomson"][1]["role"] == "measured"
        assert "Z_eff_reference" not in traces  # no target recorded, none invented
    assert traces["p_e_thomson"][1]["role"] == "measured"  # no lineage recorded
    assert model.models[0].metadata["equilibrium_lineage"] is None


def test_without_a_record_the_callers_occurrence_is_unverified():
    model = _extract(_case(40326, "magnetics", lineage_fields=False).ods, equilibrium_occurrence=3)
    meta = model.models[0].metadata
    assert (meta["equilibrium_occurrence"], meta["equilibrium_occurrence_verified"]) == (3, False)
    assert "#3 (unverified)" in model.suptitle


def test_the_p_kin_rule_lives_in_one_function():
    assert p_kin_role({"origin": "inferred", "method": "equilibrium_pressure_partition"}) == "closure identity"
    assert p_kin_role({"origin": "inferred", "method": "charge_exchange_fit"}) == "display sum"
    assert p_kin_role({"ti_te_ratio": "1", "status": "assumed"}) == "display sum"
    assert p_kin_role(None) == "display sum"


def test_no_sigma_is_drawn_where_the_contract_carries_none(case_40326, case_39915_kinetic):
    for case, quantities in ((case_40326, ("p_eq", "p_i", "T_i", "n_main_ion", "n_impurity_ion", "Z_eff",
                                           "Z_eff_reference")),
                             (case_39915_kinetic, ("p_eq", "T_i", "n_main_ion", "n_impurity_ion", "Z_eff"))):
        traces = _traces(_extract(case.ods))
        for quantity in quantities:
            assert traces[quantity][0].yerr is None, quantity
            assert traces[quantity][1]["sigma"] == "none", quantity
        for quantity in ("p_e_thomson", "n_e_thomson", "T_e_thomson"):
            assert traces[quantity][0].yerr is not None, quantity


# --- malformed records ------------------------------------------------------------------


@pytest.mark.parametrize("path, text, match", [
    (f"{CP}.zeff_fit.parameters", "free text with no field", "no key=value"),
    (f"{CP}.zeff_fit.parameters", "origin=guessed; method=x", "origin='guessed'"),
    (f"{CP}.zeff_fit.parameters", "origin=derived; method=x; target_zeff=abc", "target_zeff"),
    (f"{CP}.zeff_fit.parameters", "origin=derived; method=x; target_zeff=0.5", "target_zeff"),
    (f"{CP}.ion.0.temperature_fit.parameters",
     "origin=inferred; method=equilibrium_pressure_partition; equilibrium_occurrence=zero", "equilibrium_occurrence"),
    (f"{CP}.ion.0.temperature_fit.parameters",
     "origin=inferred; method=x; equilibrium_lineage=two words", "equilibrium_lineage"),
    (f"{CP}.t_i_average_fit.parameters", "origin=mystery; method=x", "no known grammar"),
    (f"{CP}.electrons.density_fit.parameters", "method=external", "neither origin"),
])
def test_a_malformed_record_is_refused_not_read_as_absent(ods, path, text, match):
    ods[path] = text
    with pytest.raises(ValueError, match=match):
        _extract(ods)


def test_a_record_that_is_not_text_is_refused(ods):
    ods[f"{CP}.zeff_fit.parameters"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="not record text"):
        _extract(ods)


@pytest.mark.parametrize("value", ["0", True, 1.0, -1])
def test_equilibrium_occurrence_must_be_an_integer(case_40326, value):
    with pytest.raises(ValueError, match="non-negative integer"):
        _extract(case_40326.ods, equilibrium_occurrence=value)


# --- refusals ------------------------------------------------------------------------


def test_a_sqrt_psi_proxy_coordinate_is_refused_never_relabelled(ods):
    psi = np.asarray(ods[f"{CP}.grid.psi"], float)
    axis = float(ods["equilibrium.time_slice.0.global_quantities.psi_axis"])
    edge = float(ods["equilibrium.time_slice.0.global_quantities.psi_boundary"])
    ods[f"{CP}.grid.rho_tor_norm"] = np.sqrt(np.clip((psi - axis) / (edge - axis), 0, None))
    with pytest.raises(ValueError, match="proxy"):
        _extract(ods)
    assert "proxy" in missing_required_path(ods, NAME)
    assert "Poloidal" in _extract(ods, coordinate="psi_norm").models[0].coordinate_label


def test_an_equilibrium_occurrence_mismatch_is_refused(case_40326):
    with pytest.raises(ValueError, match="occurrence 0"):
        _extract(case_40326.ods, equilibrium_occurrence=1)
    assert _extract(case_40326.ods, equilibrium_occurrence=0).models[0].metadata["equilibrium_occurrence"] == 0


def test_an_equilibrium_more_than_one_step_away_is_refused(ods, case_40326):
    ods["equilibrium.time"] = np.array([case_40326.time + 0.002])
    ods["equilibrium.time_slice.0.time"] = case_40326.time + 0.002
    with pytest.raises(ValueError, match="equilibrium step"):
        _extract(ods)


def test_a_time_far_from_every_core_slice_is_refused(case_40326):
    with pytest.raises(ValueError, match="core_profiles step"):
        _extract(case_40326.ods, time=case_40326.time + 0.01)
    assert _extract(case_40326.ods, time=case_40326.time + 1e-4).models[0].metadata["time"] == case_40326.time


def test_the_equilibrium_is_matched_by_time_not_index(ods, case_40326):
    t = case_40326.time
    ods["equilibrium.time_slice.1"] = copy.deepcopy(ods["equilibrium.time_slice.0"])
    ods["equilibrium.time_slice.0.time"] = t + 0.004
    ods["equilibrium.time_slice.0.profiles_1d.pressure"] = 2.0 * case_40326.p_eq
    ods["equilibrium.time"] = np.array([t + 0.004, t])
    np.testing.assert_allclose(_traces(_extract(ods))["p_eq"][0].y, case_40326.p_eq)
    assert _extract(ods).models[0].metadata["equilibrium_slice"] == 1


def test_resistive_zeff_pairs_with_its_slice_by_time(ods, case_40326):
    t = case_40326.time
    for root in ("equilibrium.time_slice", "core_profiles.profiles_1d"):
        ods[f"{root}.1"] = copy.deepcopy(ods[f"{root}.0"])
        ods[f"{root}.1.time"] = t + 0.004
    ods["equilibrium.time"] = np.array([t, t + 0.004])
    ods["core_profiles.time"] = np.array([t, t + 0.004])
    ods["core_profiles.global_quantities.z_eff_resistive"] = np.array([1.7, 3.0])
    first = _traces(_extract(ods, time=t))["Z_eff_reference"]
    second = _traces(_extract(ods, time=t + 0.004))["Z_eff_reference"]
    np.testing.assert_array_equal(first[0].y, [1.7, 1.7])
    np.testing.assert_array_equal(second[0].y, [3.0, 3.0])
    assert first[1]["path"] == "core_profiles.global_quantities.z_eff_resistive"
    # A resistive array off the time base is not paired at all: the recorded target remains.
    ods["core_profiles.global_quantities.z_eff_resistive"] = np.array([1.7])
    assert _traces(_extract(ods, time=t))["Z_eff_reference"][1]["path"].endswith("target_zeff")


def test_unknown_options_are_refused(case_40326):
    with pytest.raises(ValueError, match="layout"):
        _extract(case_40326.ods, layout="overlay")
    with pytest.raises(ValueError, match="time_slice"):
        _extract(case_40326.ods, time_slice=0)
    with pytest.raises(ValueError, match="equilibrium_occurrence"):
        vaft.omas.extract_kinetic_overview_profiles(case_40326.ods, equilibrium_occurrence=0)


# --- Thomson time and validity ------------------------------------------------------


def test_a_thomson_sample_more_than_one_step_away_is_left_out(ods, case_40326):
    ods["thomson_scattering.time"] = np.array([case_40326.time + 0.002])
    model = _extract(ods)
    assert not {"n_e_thomson", "T_e_thomson", "p_e_thomson"} & set(_traces(model))
    assert model.models[0].metadata["thomson_note"] and "no Thomson sample" in model.suptitle
    assert model.models[0].reference_lines == ()


def test_thomson_without_a_time_base_is_refused(ods):
    del ods["thomson_scattering.time"]
    with pytest.raises(ValueError, match="no time base"):
        _extract(ods)


def test_a_channel_whose_length_mismatches_the_time_base_is_skipped(ods):
    ods["thomson_scattering.channel.0.n_e.data"] = np.array([1e19, 2e19])
    model = _extract(ods)
    assert any(item[0] == 0 and "samples for a time base" in item[1]
               for item in model.models[0].metadata["thomson_skipped"])
    assert len(_traces(model)["n_e_thomson"][0].x) == 4


def test_per_channel_time_is_honoured_without_a_homogeneous_base(ods, case_40326):
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 0
    del ods["thomson_scattering.time"]
    for c in range(5):
        for signal in ("n_e", "t_e"):
            ods[f"thomson_scattering.channel.{c}.{signal}.time"] = np.array([case_40326.time])
    ods["thomson_scattering.channel.4.t_e.time"] = np.array([case_40326.time + 0.002])
    traces = _traces(_extract(ods))
    assert len(traces["n_e_thomson"][0].x) == 5 and len(traces["T_e_thomson"][0].x) == 4


def test_thomson_validity_flags_demote_points_and_shrink_the_span(ods):
    full = _extract(ods).models[0].reference_lines[0]
    # Channel 4 (R = 0.255 m) is the outermost mapped channel, rho_tor_norm 0.476.
    ods["thomson_scattering.channel.4.n_e.validity"] = -1
    ods["thomson_scattering.channel.4.t_e.validity_timed"] = np.array([-2])
    model = _extract(ods)
    traces = _traces(model)
    for quantity in ("n_e_thomson", "T_e_thomson", "p_e_thomson"):
        mask = traces[quantity][0].valid_mask
        assert mask is not None and int((~mask).sum()) == 1 and not mask[-1], quantity
    span = model.models[0].reference_lines[0]
    assert span.x_end < full.x_end


def test_without_thomson_ions_or_zeff_the_rest_is_drawn(ods):
    for path in ("thomson_scattering", f"{CP}.ion", f"{CP}.zeff", f"{CP}.zeff_fit", f"{CP}.pressure_ion_total"):
        del ods[path]
    traces = _traces(_extract(ods))
    assert {"p_eq", "p_e", "n_e", "T_e", "T_i"} <= set(traces)
    assert not {"n_main_ion", "n_impurity_ion", "Z_eff", "p_e_thomson", "p_i", "p_kin"} & set(traces)


def test_a_missing_equilibrium_pressure_is_stated(ods):
    del ods["equilibrium.time_slice.0.profiles_1d.pressure"]
    series, record = _traces(_extract(ods))["p_eq"]
    assert record["role"] == "unavailable" and "unavailable" in series.label
    assert not np.isfinite(series.y).any()


def test_stored_labels_reach_the_legend_as_plain_text(ods):
    ods[f"{CP}.ion.1.label"] = r"C_6 $\alpha$"
    label = _traces(_extract(ods))["n_impurity_ion"][0].label
    assert "C_6 alpha" in label and label.count("$") == 2  # only the plot's own math span


def test_a_stored_electron_pressure_is_preferred_and_its_absence_is_labelled_derived(ods, case_40326):
    assert _traces(_extract(ods))["p_e"][1]["path"].endswith("electrons.pressure")
    del ods[f"{CP}.electrons.pressure"]
    series, record = _traces(_extract(ods))["p_e"]
    assert record["role"] == "derived" and "derived" in series.label
    np.testing.assert_allclose(series.y, E * case_40326.n_e * case_40326.t_e, rtol=1e-14)


# --- the shared plotting changes stay private ---------------------------------------------


@pytest.mark.parametrize("key", ["legend_loc", "legend_ncols", "legend_fontsize"])
def test_legend_placement_is_not_a_caller_option(key):
    from vaft.plot.backend.options import validate_options

    with pytest.raises(ValueError, match="does not take an option"):
        validate_options("equilibrium_profile_q", {key: "outside"})


def test_profile_legend_defaults_are_unchanged():
    from matplotlib.font_manager import FontProperties

    from vaft.plot.models import Profile1D, Series
    from vaft.plot.renderers.profiles import render_profile_1d

    x = np.linspace(0, 1, 5)
    model = Profile1D(series=(Series(x=x, y=x, label="a"), Series(x=x, y=2 * x, label="b")))
    _, axis = render_profile_1d(model)
    legend = axis.get_legend()
    assert legend._loc == 0 and legend._ncols == 1  # "best", one column, in the axes
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(FontProperties(size="small").get_size_in_points())


# --- fixture honesty ---------------------------------------------------------------------


@pytest.mark.parametrize("shot, lineage", [(40326, "magnetics"), (40326, "electron_kinetic"), (39915, "magnetics")])
def test_the_synthetic_map_returns_the_archived_channel_coordinates(shot, lineage):
    """The converter's psi map is invented; its channel mapping is the archive's."""
    from vaft.process.profile import equilibrium_mapping_points

    case = _case(shot, lineage)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mapped = equilibrium_mapping_points(case.ods, case.ts_r, np.zeros(case.ts_r.size), time=case.time)
    np.testing.assert_allclose(mapped.psi_norm, case.ts_psi_norm, atol=1e-12)
    np.testing.assert_allclose(mapped.rho_tor_norm, case.ts_rho, atol=2e-4)


def test_the_composition_comes_from_the_writer(case_40326):
    """``origin=derived; method=openadas_transient`` is the writer's spelling, and it filled one point."""
    record = case_40326.ods[f"{CP}.zeff_fit.parameters"]
    assert record.startswith("origin=derived; method=openadas_transient; normalization=ne_weighted_mean")
    assert "filled_points=1" in record and record.endswith("target_zeff=2")
    zeff = np.asarray(case_40326.ods[f"{CP}.zeff"], float)
    assert np.isfinite(zeff).all() and not np.isfinite(case_40326.zeff).all()


# --- discovery ----------------------------------------------------------------------------


def test_the_plot_is_discoverable_and_declares_its_paths(case_40326):
    assert missing_required_path(case_40326.ods, NAME) is None
    assert NAME in vaft.plot.__all__
    paths = {path.canonical for path in vaft.omas.dd_kinetic_overview_state()}
    assert {"core_profiles/profiles_1d(:)/zeff_fit/parameters", "core_profiles/profiles_1d(:)/t_i_average",
            "thomson_scattering/channel(:)/n_e/validity_timed", "thomson_scattering/channel(:)/position/r",
            "core_profiles/profiles_1d(:)/ion(:)/state(:)/density"} <= paths
    assert "core_profiles/code/parameters" not in paths
    sample = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert missing_required_path(sample, NAME) is not None
