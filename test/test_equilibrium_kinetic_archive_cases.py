"""The archived VEST kinetic-state cases: frozen numbers and the evidence they carry.

These are the independent numerical checks of the original 39915/316 ms and
40326/322 ms archive notebooks, kept as regression tests when those notebooks
were merged into ``notebooks/equilibrium_kinetic_consistency_and_inference.ipynb``.
Expected values are literals from the archives and the original notebooks'
outputs, or are recomputed here from the raw snapshot arrays, so the converter
is never checked against itself.  Checks already made by
``test_kinetic_overview_state.py`` (finite ``T_i`` counts on the plot, roles,
``Z_eff`` on axis) are not repeated.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process.kinetic_closure import infer_ti_pressure_partition

from _kinetic_state_fixture import (
    E, TI_TE_RATIO_SIGMA, kinetic_efit_assessment, kinetic_efit_pressure_policies, kinetic_state_case,
    load_archive, state_table,
)


@pytest.fixture(scope="module")
def magnetic():
    return {shot: kinetic_state_case(shot, "magnetics") for shot in (39915, 40326)}


@pytest.fixture(scope="module")
def policies():
    return {shot: kinetic_efit_pressure_policies(shot) for shot in (39915, 40326)}


def _raw_composition(shot):
    """n_i/n_e and Z_eff straight from the snapshot's charge fractions (n_C = n_O)."""
    s = load_archive(shot)
    ne = np.asarray(s["core_profiles"]["electron_density_m3"], float)
    scale = float(s["impurity_model"].get("scale", s["openadas_charge_states"].get("scale")))
    fC = np.asarray(s["openadas_charge_states"]["C"], float)
    fO = np.asarray(s["openadas_charge_states"]["O"], float)
    qC, qO = np.arange(7), np.arange(9)
    n_each = 0.5 * scale * ne
    n_h = ne - n_each * (fC @ qC) - n_each * (fO @ qO)
    n_ion = n_h + n_each * (1 - fC[:, 0]) + n_each * (1 - fO[:, 0])
    zeff = (n_h + n_each * (fC @ qC**2) + n_each * (fO @ qO**2)) / ne
    valid = np.isfinite(fC).all(axis=1) & np.isfinite(fO).all(axis=1)
    return n_h, n_ion, zeff, valid, ne


@pytest.mark.parametrize("shot", [39915, 40326])
def test_composition_is_quasineutral_and_meets_its_weighted_mean_target(shot, magnetic):
    n_h, n_ion, zeff, valid, ne = _raw_composition(shot)
    case = magnetic[shot]
    assert valid.sum() == 128
    assert np.all(n_h[valid] >= 0)
    np.testing.assert_allclose(case.zeff[valid], zeff[valid], rtol=1e-12)
    np.testing.assert_allclose((case.n_h + case.n_imp)[valid], n_ion[valid], rtol=1e-12)
    np.testing.assert_allclose(case.notes["zeff_mean"], 2.0, rtol=1e-12)


def test_39915_zeff_off_axis_matches_the_atlas(magnetic):
    case = magnetic[39915]
    vc = case.valid_charge
    np.testing.assert_allclose(np.interp(0.9, case.rho[vc], case.zeff[vc]), 1.7190221792755789, rtol=1e-12)


@pytest.mark.parametrize("shot, n_valid", [(39915, 128), (40326, 117)])
def test_the_partition_closes_the_magnetic_pressure_exactly(shot, n_valid, magnetic):
    case = magnetic[shot]
    assert case.valid_ti.sum() == n_valid
    np.testing.assert_allclose(case.p_kin[case.valid_ti], case.p_eq[case.valid_ti], rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(case.t_i[~case.valid_ti]))


def test_39915_ti_range_is_unchanged(magnetic):
    np.testing.assert_allclose([np.nanmin(magnetic[39915].t_i), np.nanmax(magnetic[39915].t_i)],
                               [1.02, 293.61], atol=5e-3)


def test_40326_channel_pressure_residuals_are_positive_with_the_archived_values(magnetic):
    case = magnetic[40326]
    ts = load_archive(40326)["mapped_thomson"]["magnetics"]
    rt = np.asarray(ts["rho_tor_norm"], float)
    ne_ts, te_ts = np.asarray(ts["electron_density_m3"], float), np.asarray(ts["electron_temperature_eV"], float)
    vc = case.valid_charge
    fraction = np.interp(rt, case.rho[vc], ((case.n_h + case.n_imp) / case.n_e)[vc])
    ti = (np.asarray(ts["p_eq_Pa"], float) - E * ne_ts * te_ts) / (E * ne_ts * fraction)
    np.testing.assert_allclose(ti, [90.7, 43.1, 33.5, 41.0, 47.9], atol=0.05)


def test_the_39915_state_table_has_the_archived_columns():
    columns, values = state_table(39915)
    assert columns[:3] == ["rho_tor_norm", "p_eq_Pa", "p_e_Pa"] and columns[-1] == "n_O_q8_m3"
    assert len(columns) == len(values) == 17 + 7 + 9


# --- the electron-kinetic EFIT ----------------------------------------------------------


@pytest.mark.parametrize("shot", [39915, 40326])
def test_the_electron_kinetic_partition_is_refused_as_circular(shot, magnetic):
    s = load_archive(shot)
    case = magnetic[shot]
    refused = infer_ti_pressure_partition(
        float(s["electron_kinetic_equilibrium"]["pressure_Pa"][0]), case.n_e[0], case.t_e[0],
        composition={"ion_density_per_electron": float((case.n_h + case.n_imp)[0] / case.n_e[0]),
                     "composition_source": "assumed transient C/O"},
        sigma_p_eq=np.nan, sigma_n_e=np.nan, sigma_t_e=np.nan, equilibrium_lineage="electron_kinetic")
    assert refused["eligible"] is False and "circular" in refused["reason"]
    assert "circular" in kinetic_efit_pressure_policies(shot).refusal


def test_39915_converged_but_fails_admissibility():
    a = kinetic_efit_assessment(39915)
    assert (a.manifest_status, a.manifest_converged, a.efit_status) == ("success", True, "valid")
    assert a.kinetic_admissible == "fail" and a.kinetic_admissible_reasons == "pressure_min -0.845745309"
    assert (a.chi2, a.r_w, a.r_w_on_pair_span, a.c_p) == (87.72, 1.865508346488106, 1.8239024316193149,
                                                          -0.0035870475053022834)
    assert a.source_keys == ("electron_kinetic_assessment",)
    p = np.asarray(load_archive(39915)["electron_kinetic_equilibrium"]["pressure_Pa"], float)
    assert np.count_nonzero(p < 0) == 4
    np.testing.assert_allclose(p.min(), -0.845745309, atol=1e-10)


def test_40326_passes_admissibility_from_two_separate_records():
    a = kinetic_efit_assessment(40326)
    assert (a.manifest_status, a.manifest_converged, a.efit_status, a.kinetic_admissible) == \
        ("success", True, "valid", "pass")
    assert (a.chi2, a.r_w_on_pair_span, a.c_p) == (361.7, 1.9026945911314008, 0.00944033940242725)
    assert a.r_w != a.r_w_on_pair_span
    assert a.source_keys == ("state_assessment.electron_kinetic", "kinetic_manifest")
    p = np.asarray(load_archive(40326)["electron_kinetic_equilibrium"]["pressure_Pa"], float)
    np.testing.assert_allclose(p.min(), 1.08194241, atol=1e-10)


@pytest.mark.parametrize("shot", [39915, 40326])
def test_both_efits_used_the_same_declared_prior(shot):
    a = kinetic_efit_assessment(shot)
    assert (a.pressure_constraint_encoding, a.ti_lineage, a.ti_te_ratio_assumed, a.ts_status, a.ts_points) == \
        ("raw6", "ti_eq_te_assumed", 1.0, "matched", 5)


def test_39915_thomson_stays_on_its_magnetic_mapping(policies):
    pol = policies[39915]
    s = load_archive(39915)
    assert (pol.mapping_source, pol.mapping_lineage, pol.mapping_native_to_equilibrium) == \
        ("mapped_thomson", "magnetics", False)
    np.testing.assert_array_equal(pol.ts_rho, s["mapped_thomson"]["rho_tor_norm"])
    assert not np.allclose(pol.rho_kinetic, pol.rho_magnetic)  # two grids, never interchanged
    np.testing.assert_allclose(np.max(np.abs(pol.rho_kinetic - pol.rho_magnetic)), 0.01169741227252652, rtol=1e-12)


def test_40326_thomson_uses_the_kinetic_efit_mapping(policies):
    pol = policies[40326]
    s = load_archive(40326)["mapped_thomson"]
    assert (pol.mapping_source, pol.mapping_lineage, pol.mapping_native_to_equilibrium) == \
        ("mapped_thomson.electron_kinetic", "electron_kinetic", True)
    np.testing.assert_array_equal(pol.ts_rho, s["electron_kinetic"]["rho_tor_norm"])
    assert not np.array_equal(pol.ts_rho, s["magnetics"]["rho_tor_norm"])
    np.testing.assert_allclose(np.max(np.abs(pol.rho_kinetic - pol.rho_magnetic)), 0.0023924624677103545,
                               rtol=1e-12)


@pytest.mark.parametrize("shot", [39915, 40326])
def test_pressure_policies_follow_their_definitions(shot, policies):
    pol = policies[shot]
    _, n_ion, _, valid, ne = _raw_composition(shot)
    te = np.asarray(load_archive(shot)["core_profiles"]["electron_temperature_eV"], float)
    np.testing.assert_allclose(pol.p_prior, 2 * E * ne * te, rtol=1e-12)
    np.testing.assert_allclose(pol.p_counterfactual[valid], (E * ne * te + E * n_ion * te)[valid], rtol=1e-12)
    assert np.all(pol.p_counterfactual[valid] <= pol.p_prior[valid] + 1e-10)  # dilution only lowers it
    mapped = load_archive(shot)["mapped_thomson"]
    ts = mapped.get("electron_kinetic", mapped)  # 39915: the one (magnetic) mapping
    n, t = np.asarray(ts["electron_density_m3"], float), np.asarray(ts["electron_temperature_eV"], float)
    sn, st = np.asarray(ts["electron_density_error_m3"], float), np.asarray(ts["electron_temperature_error_eV"], float)
    sigma = E * np.sqrt((2 * t * sn) ** 2 + (2 * n * st) ** 2 + (n * t * TI_TE_RATIO_SIGMA) ** 2)
    np.testing.assert_allclose(pol.ts_p_input, 2 * E * n * t, rtol=1e-12)
    np.testing.assert_allclose(pol.ts_sigma_p_input, sigma, rtol=1e-12)


def _patched_archive(monkeypatch, shot, edit):
    import archive_to_ods

    snapshot = load_archive(shot)
    edit(snapshot)
    monkeypatch.setattr(archive_to_ods, "load_archive", lambda _shot: snapshot)


@pytest.mark.parametrize("helper", [kinetic_efit_pressure_policies, state_table, kinetic_state_case])
def test_helpers_refuse_profiles_off_the_equilibrium_grid(monkeypatch, helper):
    def shift(s):
        rho = np.asarray(s["magnetic_equilibrium"]["rho_tor_norm"], float)
        s["magnetic_equilibrium"]["rho_tor_norm"] = list(np.sqrt(rho))  # same ends, different interior
    _patched_archive(monkeypatch, 40326, shift)
    with pytest.raises(ValueError, match="one rho_tor_norm grid"):
        helper(40326)


@pytest.mark.parametrize("helper", [kinetic_efit_pressure_policies, kinetic_state_case])
def test_helpers_refuse_thomson_ne_and_te_at_different_rho(monkeypatch, helper):
    def shift(s):
        ts = s["mapped_thomson"]
        ts["density_rho_tor_norm"] = [r + 0.01 for r in ts["density_rho_tor_norm"]]
    _patched_archive(monkeypatch, 39915, shift)
    with pytest.raises(ValueError, match="same rho_tor_norm"):
        helper(39915)


def test_assessment_refuses_disagreeing_chi2_records(monkeypatch):
    def edit(s):
        s["state_assessment"]["electron_kinetic"]["kinetic_chi2"] = "400.0"
    _patched_archive(monkeypatch, 40326, edit)
    with pytest.raises(ValueError, match="disagrees with the manifest"):
        kinetic_efit_assessment(40326)
