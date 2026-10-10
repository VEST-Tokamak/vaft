"""``kinetic_overview_state``: one matched kinetic state from stored paths (issue #1837).

The fixtures are the #1839 archive cases turned into ODS by
``test/_kinetic_state_fixture.py``, which does the notebooks' composition and
``T_i`` algebra and stores the result; the plot itself only reads.  The
numbers checked here are the archive notebooks' own: 117/129 valid ``T_i``
points and ``Z_eff(0) = 2.022`` for 40326/322 ms, 128/129 and
``Z_eff(0) = 1.328`` for 39915/316 ms, ``p_e + p_i = p_eq`` to rounding where
``T_i`` is valid, and the Thomson pressure ``e n_e T_e`` with its independent
1 sigma.
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
from vaft.plot.backend.recipes import missing_required_path  # noqa: E402

from _kinetic_state_fixture import E, kinetic_state_case, load_archive  # noqa: E402

NAME = "kinetic_overview_state"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def case_40326():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinetic_state_case(40326, "magnetics")


@pytest.fixture(scope="module")
def case_39915_kinetic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinetic_state_case(39915, "electron_kinetic")


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


@pytest.mark.parametrize("layout, shape", [("grid", (2, 2)), ("stack", (4, 1))])
def test_40326_renders_both_layouts(case_40326, layout, shape):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure, axes = vaft.omas.plot_kinetic_overview_state(case_40326.ods, layout=layout, format="slide")
    assert np.asarray(axes).shape == shape
    labels = [text.get_text() for axis in np.asarray(axes).ravel()
              for text in (axis.get_legend().get_texts() if axis.get_legend() else [])]
    assert any("closure identity" in label for label in labels)
    assert any("independent validation" in label for label in labels)
    assert "322.0 ms" in figure._suptitle.get_text()


def test_the_layout_changes_the_grid_not_the_panels(case_40326):
    grid, stack = _extract(case_40326.ods, layout="grid"), _extract(case_40326.ods, layout="stack")
    assert (grid.nrows, grid.ncols, stack.nrows, stack.ncols) == (2, 2, 4, 1)
    assert [m.metadata["panel"] for m in grid.models] == ["pressure", "density", "temperature", "zeff"]
    for a, b in zip(grid.models, stack.models):
        assert [s.label for s in a.series] == [s.label for s in b.series]


def test_plotly_draws_the_same_state(case_40326):
    pytest.importorskip("plotly")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = vaft.omas.plot_kinetic_overview_state(case_40326.ods, backend="plotly")
    assert len(figure.data) >= 10


# --- the archive notebook's numbers -------------------------------------------------


def test_40326_matches_the_archive_notebook(case_40326):
    s = load_archive(40326)
    traces = _traces(_extract(case_40326.ods))
    ti = traces["T_i"][0]
    assert int(np.isfinite(ti.y).sum()) == 117  # "Physical magnetic-EFIT Ti grid points: 117 / 129"
    np.testing.assert_allclose(ti.y, case_40326.t_i, rtol=0, atol=0, equal_nan=True)
    assert "117/129 valid" in ti.label

    zeff = traces["Z_eff"][0]
    np.testing.assert_allclose(zeff.y[0], s["impurity_model"]["atlas_zeff_axis"], rtol=1e-12)
    finite = np.isfinite(zeff.y)
    np.testing.assert_allclose(np.interp(0.9, zeff.x[finite], zeff.y[finite]),
                               s["impurity_model"]["atlas_zeff_rho09"], rtol=1e-12)
    reference = traces["Z_eff_reference"][0]
    np.testing.assert_array_equal(reference.y, [2.0, 2.0])
    assert "assumed" in reference.label

    # The closure: p_e + p_i reproduces p_eq where T_i is valid, by construction.
    p_kin, p_eq = traces["p_kin"][0], traces["p_eq"][0]
    np.testing.assert_allclose(p_eq.x, p_kin.x, atol=1e-10)  # one grid for this case
    valid = np.isfinite(p_kin.y)
    assert valid.sum() == 117
    np.testing.assert_allclose(p_kin.y[valid], p_eq.y[valid], rtol=1e-12, atol=1e-12)
    assert traces["p_kin"][1]["role"] == "closure identity"

    # Thomson: mapped through the selected equilibrium to the archived rho, and
    # e n_e T_e with the notebook's independent 1 sigma.
    ts = s["mapped_thomson"]["magnetics"]
    p_ts = traces["p_e_thomson"][0]
    order = np.argsort(ts["rho_tor_norm"])
    np.testing.assert_allclose(np.sort(p_ts.x), np.asarray(ts["rho_tor_norm"])[order], atol=2e-4)
    by_x = np.argsort(p_ts.x)
    np.testing.assert_allclose(p_ts.y[by_x], np.asarray(ts["p_e_Pa"])[order], rtol=1e-12)
    np.testing.assert_allclose(p_ts.yerr[by_x], case_40326.ts_sigma_p_e[order], rtol=1e-12)
    n_ts = traces["n_e_thomson"][0]
    np.testing.assert_allclose(n_ts.y[np.argsort(n_ts.x)] * 1e19,
                               np.asarray(ts["electron_density_m3"])[order], rtol=1e-12)
    np.testing.assert_allclose(n_ts.yerr[np.argsort(n_ts.x)] * 1e19,
                               np.asarray(ts["electron_density_error_m3"])[order], rtol=1e-12)


def test_39915_magnetic_case_matches_its_notebook():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = kinetic_state_case(39915, "magnetics")
    traces = _traces(_extract(case.ods))
    assert int(np.isfinite(traces["T_i"][0].y).sum()) == 128  # "Physical T_i points: 128/129"
    np.testing.assert_allclose(traces["Z_eff"][0].y[0], 1.3276842011353556, rtol=1e-12)
    finite = np.isfinite(traces["p_kin"][0].y)
    np.testing.assert_allclose(traces["p_kin"][0].y[finite], traces["p_eq"][0].y[finite], rtol=1e-12, atol=1e-12)


# --- the kinetic-EFIT rejection --------------------------------------------------


def test_39915_kinetic_efit_rejection_leaves_t_i_empty(case_39915_kinetic):
    ods = case_39915_kinetic.ods
    # The stored T_i is the EFIT's own T_i = T_e prior, finite everywhere ...
    assert np.isfinite(np.asarray(ods["core_profiles.profiles_1d.0.t_i_average"])).all()
    traces = _traces(_extract(ods))
    # ... and the record says none of it is a result: nothing is drawn.
    assert not np.isfinite(traces["T_i"][0].y).any()
    assert "0/129 valid" in traces["T_i"][0].label and "invalid" in traces["T_i"][0].label
    assert not np.isfinite(traces["p_i"][0].y).any()
    assert "p_kin" not in traces  # no sum of an invalid ion pressure
    assert traces["p_e_thomson"][1]["role"] == "fit input"  # TS constrained this EFIT
    np.testing.assert_allclose(traces["p_eq"][0].y.min(), -0.845745309, atol=1e-10)
    meta = _extract(ods).models[0].metadata
    assert meta["equilibrium_lineage"] == "electron_kinetic" and meta["t_i_valid_range"] is None


def test_no_sigma_is_drawn_where_none_is_stored(case_40326, case_39915_kinetic):
    for case in (case_40326, case_39915_kinetic):
        traces = _traces(_extract(case.ods))
        for quantity in ("p_eq", "p_i", "T_i", "n_main_ion", "n_impurity_ion", "Z_eff", "Z_eff_reference"):
            series, record = traces[quantity]
            assert series.yerr is None, quantity
            assert record["sigma"] == "none", quantity
        for quantity in ("p_e_thomson", "n_e_thomson", "T_e_thomson"):
            assert traces[quantity][0].yerr is not None, quantity


# --- refusals ------------------------------------------------------------------------


def _proxy(ods):
    ods = copy.deepcopy(ods)
    psi = np.asarray(ods["core_profiles.profiles_1d.0.grid.psi"], float)
    axis = float(ods["equilibrium.time_slice.0.global_quantities.psi_axis"])
    edge = float(ods["equilibrium.time_slice.0.global_quantities.psi_boundary"])
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = np.sqrt(np.clip((psi - axis) / (edge - axis), 0, None))
    return ods


def test_a_sqrt_psi_proxy_coordinate_is_refused_never_relabelled(case_40326):
    proxy = _proxy(case_40326.ods)
    with pytest.raises(ValueError, match="proxy"):
        _extract(proxy)
    assert "proxy" in missing_required_path(proxy, NAME)
    # psi_norm does not read the proxy, so it stays drawable.
    model = _extract(proxy, coordinate="psi_norm")
    assert "Poloidal" in model.models[0].coordinate_label


def test_an_equilibrium_occurrence_mismatch_is_refused(case_40326):
    with pytest.raises(ValueError, match="occurrence 0"):
        _extract(case_40326.ods, equilibrium_occurrence=1)
    model = _extract(case_40326.ods, equilibrium_occurrence=0)
    assert model.models[0].metadata["equilibrium_occurrence"] == 0


def test_an_equilibrium_more_than_one_step_away_is_refused(case_40326):
    ods = copy.deepcopy(case_40326.ods)
    ods["equilibrium.time"] = np.array([case_40326.time + 0.002])
    ods["equilibrium.time_slice.0.time"] = case_40326.time + 0.002
    with pytest.raises(ValueError, match="equilibrium step"):
        _extract(ods)


def test_the_equilibrium_is_matched_by_time_not_index(case_40326):
    ods = copy.deepcopy(case_40326.ods)
    t = case_40326.time
    ods["equilibrium.time_slice.1"] = copy.deepcopy(ods["equilibrium.time_slice.0"])
    ods["equilibrium.time_slice.0.time"] = t + 0.004
    ods["equilibrium.time_slice.0.profiles_1d.pressure"] = 2.0 * case_40326.p_eq
    ods["equilibrium.time"] = np.array([t + 0.004, t])
    traces = _traces(_extract(ods))
    np.testing.assert_allclose(traces["p_eq"][0].y, case_40326.p_eq)
    assert _extract(ods).models[0].metadata["equilibrium_slice"] == 1


def test_t_i_validity_must_be_on_the_core_grid(case_40326):
    ods = copy.deepcopy(case_40326.ods)
    record = json.loads(ods["core_profiles.code.parameters"])
    record["kinetic_state"]["t_i_valid"] = record["kinetic_state"]["t_i_valid"][:-1]
    ods["core_profiles.code.parameters"] = json.dumps(record)
    with pytest.raises(ValueError, match="t_i_valid"):
        _extract(ods)


def test_unknown_options_are_refused(case_40326):
    with pytest.raises(ValueError, match="layout"):
        _extract(case_40326.ods, layout="overlay")
    with pytest.raises(ValueError, match="time_slice"):
        _extract(case_40326.ods, time_slice=0)
    with pytest.raises(ValueError, match="equilibrium_occurrence"):
        vaft.omas.extract_kinetic_overview_profiles(case_40326.ods, equilibrium_occurrence=0)


# --- roles -----------------------------------------------------------------------------


@pytest.mark.parametrize("options", [{"roles": False}, {"kinetic_state": False}])
def test_absent_roles_are_labelled_stored_never_guessed(options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = kinetic_state_case(40326, "magnetics", **options)
    traces = _traces(_extract(case.ods))
    for quantity in ("p_eq", "p_e", "p_i", "n_e", "n_main_ion", "n_impurity_ion", "T_e", "T_i", "Z_eff"):
        assert traces[quantity][1]["role"] == "stored", quantity
        assert "(stored" in traces[quantity][0].label, quantity
    for quantity in ("p_e_thomson", "n_e_thomson", "T_e_thomson"):
        assert traces[quantity][1]["role"] == "measured"
    assert traces["p_kin"][1]["role"] == "display sum"  # not called a closure without a record
    if not options.get("kinetic_state", True):
        assert "Z_eff_reference" not in traces  # no target recorded, none invented


def test_a_stored_electron_pressure_is_preferred_and_its_absence_is_labelled_derived(case_40326):
    traces = _traces(_extract(case_40326.ods))
    assert traces["p_e"][1]["path"].endswith("electrons.pressure")
    ods = copy.deepcopy(case_40326.ods)
    del ods["core_profiles.profiles_1d.0.electrons.pressure"]
    series, record = _traces(_extract(ods))["p_e"]
    assert record["role"] == "derived" and "derived" in series.label
    np.testing.assert_allclose(series.y, E * case_40326.n_e * case_40326.t_e, rtol=1e-14)


def test_a_stored_resistive_zeff_is_the_reference(case_40326):
    ods = copy.deepcopy(case_40326.ods)
    ods["core_profiles.global_quantities.z_eff_resistive"] = np.array([1.7])
    series, record = _traces(_extract(ods))["Z_eff_reference"]
    np.testing.assert_array_equal(series.y, [1.7, 1.7])
    assert record["path"] == "core_profiles.global_quantities.z_eff_resistive"


def test_the_measured_span_is_shaded(case_40326):
    model = _extract(case_40326.ods)
    lines = [line for panel in model.models for line in panel.reference_lines]
    assert len(lines) == 4 and [bool(line.label) for line in lines] == [True, False, False, False]
    ts_x = _traces(model)["p_e_thomson"][0].x
    assert (lines[0].x, lines[0].x_end) == (ts_x.min(), ts_x.max())


# --- fixture honesty ---------------------------------------------------------------------


@pytest.mark.parametrize("shot, lineage", [(40326, "magnetics"), (40326, "electron_kinetic"), (39915, "magnetics")])
def test_the_synthetic_map_returns_the_archived_channel_coordinates(shot, lineage):
    """The fixture's psi map is invented; its channel mapping is the archive's."""
    from vaft.process.profile import equilibrium_mapping_points

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        case = kinetic_state_case(shot, lineage)
        mapped = equilibrium_mapping_points(case.ods, case.ts_r, np.zeros(case.ts_r.size), time=case.time)
    np.testing.assert_allclose(mapped.psi_norm, case.ts_psi_norm, atol=1e-12)
    np.testing.assert_allclose(mapped.rho_tor_norm, case.ts_rho, atol=2e-4)


# --- discovery ----------------------------------------------------------------------------


def test_the_plot_is_discoverable_and_declares_its_paths(case_40326):
    assert missing_required_path(case_40326.ods, NAME) is None
    assert NAME in vaft.plot.__all__
    paths = {path.canonical for path in vaft.omas.dd_kinetic_overview_state()}
    assert {"core_profiles/code/parameters", "core_profiles/profiles_1d(:)/t_i_average",
            "thomson_scattering/channel(:)/position/r"} <= paths
    sample = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert missing_required_path(sample, NAME) is not None
