"""Impurity composition and radial Z_eff plots (#1565 Sec. 8): synthetic ADF11 tables, no network."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from omas import ODS

import vaft.plot
from vaft.plot.models import Panels, Profile1D
from vaft.process.impurity import resolve_radial_composition

from _adf11_synthetic import synthetic_adf11_tables

RHO = np.linspace(0.0, 1.0, 11)
TIMES = (0.300, 0.310)


def _te(k):
    return (300.0 + 50.0 * k) * (1.0 - 0.95 * RHO**2)


def _ne(k):
    return (1.5e19 + 1e18 * k) * (1.0 - 0.7 * RHO**2)


def _ods(*, zeff=None, record=None, equilibrium=False) -> ODS:
    """Two core_profiles slices (300 and 310 ms); optionally a stored Z_eff and an equilibrium."""
    ods = ODS()
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.asarray(TIMES)
    for k, t in enumerate(TIMES):
        base = f"core_profiles.profiles_1d.{k}"
        ods[f"{base}.time"] = t
        ods[f"{base}.grid.rho_tor_norm"] = RHO
        ods[f"{base}.electrons.temperature"] = _te(k)
        ods[f"{base}.electrons.density_thermal"] = _ne(k)
        if zeff is not None:
            ods[f"{base}.zeff"] = np.full(RHO.size, zeff + k)
        if record is not None:
            ods[f"{base}.zeff_fit.parameters"] = record
    if equilibrium:
        ods["equilibrium.ids_properties.homogeneous_time"] = 1
        ods["equilibrium.time"] = np.asarray(TIMES)
        for k, t in enumerate(TIMES):
            eq = f"equilibrium.time_slice.{k}"
            ods[f"{eq}.time"] = t
            ods[f"{eq}.profiles_1d.rho_tor_norm"] = RHO
            ods[f"{eq}.profiles_1d.volume"] = 0.5 * RHO**3     # not the rho drho stand-in
    return ods


def _extract(name, ods, **options):
    options.setdefault("adf11_tables", synthetic_adf11_tables())
    if name == "core_profiles_profile_zeff":
        options.pop("adf11_tables")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.plot.extract(name, ods, **options)


# --- the stored Z_eff ----------------------------------------------------------------


@pytest.mark.parametrize(("record", "expected"), [
    ("origin=measured; method=visible_bremsstrahlung", "measured"),
    ("origin=assumed; method=impurity_model_preset", "assumed, not measured"),
    ("origin=derived; method=charge_states", "derived, not measured"),
    (None, "origin not recorded"),
])
def test_the_stored_zeff_is_labelled_by_the_origin_its_record_states(record, expected):
    model = _extract("core_profiles_profile_zeff", _ods(zeff=2.0, record=record))
    assert isinstance(model, Profile1D)
    (series,) = model.series
    assert series.label == f"Z_eff ({expected})"
    assert expected in model.title
    if record is None or "measured" not in record.split(";")[0]:
        assert "(measured)" not in series.label
    np.testing.assert_allclose(series.y, 2.0)


def test_the_stored_zeff_slice_is_chosen_by_time():
    ods = _ods(zeff=2.0, record="origin=assumed; method=x")
    later = _extract("core_profiles_profile_zeff", ods, time=0.309)
    np.testing.assert_allclose(later.series[0].y, 3.0)
    assert later.metadata["time"] == pytest.approx(0.310)
    by_index = _extract("core_profiles_profile_zeff", ods, time_slice=1)
    np.testing.assert_array_equal(by_index.series[0].y, later.series[0].y)
    assert by_index.title == later.title


def test_a_slice_without_zeff_is_not_offered_and_nothing_is_created():
    ods = _ods()
    assert R_missing(ods, "core_profiles_profile_zeff")
    with pytest.raises(ValueError):
        _extract("core_profiles_profile_zeff", ods)
    assert "core_profiles.profiles_1d.0.zeff" not in ods
    assert "core_profiles.profiles_1d.0.zeff_fit" not in ods


def R_missing(ods, name):
    from vaft.plot.backend.recipes import missing_required_path

    return missing_required_path(ods, name)


# --- the composition from T_e, n_e ---------------------------------------------------


def test_the_composition_is_the_process_layer_result_on_the_slice_named_by_time():
    model = _extract("impurity_profile_composition", _ods(equilibrium=True), time=0.311,
                     plasma_age_s=0.02)
    assert isinstance(model, Panels) and len(model.models) == 4
    edges = np.concatenate(([0.0], 0.5 * (RHO[1:] + RHO[:-1]), [1.0]))
    dv = np.clip(np.diff(np.interp(edges, RHO, 0.5 * RHO**3)), 0.0, None)   # the equilibrium's V(rho)
    expected = resolve_radial_composition(
        _te(1), _ne(1), RHO, {"C": 0.5, "O": 0.5}, normalization="ne_weighted_mean", target_zeff=2.0,
        plasma_age_s=0.02, volume_weights=dv, tables=synthetic_adf11_tables())
    zeff, moments, reduced, fractions = model.models
    np.testing.assert_allclose(zeff.series[0].y, expected.zeff)
    np.testing.assert_allclose(moments.series[0].y, expected.mean_charge[:, 0])
    np.testing.assert_allclose(reduced.series[0].y, expected.effective_charge)
    np.testing.assert_allclose(fractions.series[0].y, expected.dilution_fraction)
    assert "(model)" in model.suptitle and "VEST preset" in model.suptitle
    assert "310.00 ms" in model.suptitle
    labels = [s.label for s in zeff.series]
    assert labels[:2] == ["<Z_eff>_ne = 2", "fully stripped at Z_eff = 2"]
    assert any("C6+ fully stripped" == s.label for s in moments.series)


def test_points_too_young_for_coronal_charge_states_are_marked():
    young = _extract("impurity_profile_composition", _ods(), plasma_age_s=1e-7)
    marks = [s for s in young.models[0].series if s.label.startswith("not coronal at")]
    assert marks and marks[0].style["marker"] == "x"
    assert marks[0].x[0] == RHO[0] and marks[0].x[-1] == RHO[-1]   # the whole young span
    old = _extract("impurity_profile_composition", _ods(), plasma_age_s=10.0)
    assert not [s for s in old.models[0].series if s.label.startswith("not coronal")]


def test_without_a_plasma_age_the_coronal_check_is_not_drawn_and_transient_is_refused():
    model = _extract("impurity_profile_composition", _ods())
    assert "coronal check not drawn" in model.models[1].title
    assert not [s for s in model.models[0].series if s.label.startswith("not coronal")]
    with pytest.raises(ValueError, match="plasma_age_s"):
        _extract("impurity_profile_composition", _ods(), ionization="transient")


def test_transient_ionization_draws_the_coronal_reference_beside_the_mean_charge():
    model = _extract("impurity_profile_composition", _ods(), ionization="transient", plasma_age_s=1e-3)
    labels = [s.label for s in model.models[1].series]
    assert "<Z>_C coronal" in labels and "<Z>_O coronal" in labels
    assert model.models[0].title.startswith("transient charge states")


def test_a_stored_zeff_is_overlaid_under_its_own_origin():
    model = _extract("impurity_profile_composition",
                     _ods(zeff=1.8, record="origin=assumed; method=impurity_model_preset"))
    assert "stored (assumed, not measured)" in [s.label for s in model.models[0].series]


@pytest.mark.parametrize("bad", [("resistive_closure",), ("ne_weighted",)])
def test_a_normalization_the_view_cannot_draw_is_refused(bad):
    with pytest.raises(ValueError):
        _extract("impurity_profile_composition", _ods(), normalizations=bad)


def test_a_caller_composition_and_target_are_used_and_stated():
    model = _extract("impurity_profile_composition", _ods(), impurity_model={"C": 2.0, "O": 1.0},
                     target_zeff=1.5, normalizations=("axis",))
    assert model.models[0].series[0].y[0] == pytest.approx(1.5)
    assert "C:O = 2:1 (caller)" in model.suptitle
    assert "Z_eff(0) = 1.5" in model.models[0].title


def test_the_composition_does_not_write_into_the_ods():
    ods = _ods()
    before = sorted(ods.flat())
    _extract("impurity_profile_composition", ods)
    _extract("impurity_profile_charge_state_fraction", ods)
    assert sorted(ods.flat()) == before


def test_charge_state_fractions_sum_to_one_per_element():
    model = _extract("impurity_profile_charge_state_fraction", _ods())
    assert model.models[0].title.startswith("C, coronal; no plasma age")
    assert model.models[1].title == "O, coronal"
    for panel, z_n in zip(model.models, (6, 8)):
        assert len(panel.series) == z_n + 1
        np.testing.assert_allclose(np.sum([s.y for s in panel.series], axis=0), 1.0, atol=1e-9)


def test_the_plasma_age_defaults_to_the_onset_the_diagnostics_give():
    import contextlib
    import io

    import vaft
    import vaft.omas

    from _synthetic_inputs import make_core_profiles

    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ods = make_core_profiles(vaft.omas.load(vaft.data.sample(39915, representation="omas")))
        from vaft.omas.plasma_timing import plasma_timing

        onset = plasma_timing(ods).onset
    model = _extract("impurity_profile_composition", ods)
    t = float(ods["core_profiles.profiles_1d.0.time"])
    assert model.models[1].title.startswith(f"plasma age {(t - onset) * 1e3:.1f} ms")
