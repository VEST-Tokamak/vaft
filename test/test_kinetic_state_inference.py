"""The staged kinetic-state inference (#1842): Romero first, T_i = T_e, n_C = n_O.

Built on the packaged 39915 equilibrium (nine slices, weber flux, given a real toroidal
flux so its ``rho_tor_norm`` is a coordinate and not the sqrt(psi_N) proxy) with synthetic
electron profiles and synthetic ADF11 tables, so Lane Z's Romero balance, Lane L's
resistive closure and the dilution-aware pressure all run for real, offline.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from omas import ODS

from vaft.data.open_adas import read_adf11
from vaft.machine_mapping.core_profiles import classify_ti_record, ti_record_fields
from vaft.ods_access import path_value
from vaft.omas.kinetic_state_inference import STAGE_NAMES, kinetic_state_stages

pytest.importorskip("skimage")

SHOT = 39915
TIME = 0.317
RHO = np.linspace(0.0, 1.0, 41)
LOG_NE = (10.0, 14.0)
LOG_TE = (0.0, 1.0, 2.0, 3.0)


def _table(path, blocks):
    lines = [f"{len(blocks)} {len(LOG_NE)} {len(LOG_TE)} / synthetic", "",
             " ".join(map(str, LOG_NE)), " ".join(map(str, LOG_TE))]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(f"{v:.3f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _element(tmp_path, symbol, z):
    scd, acd = [], []
    for j in range(z):
        threshold = 0.6 + 0.35 * j
        per_te = [-8.0 - 3.0 * max(threshold - t, 0.0) for t in LOG_TE]
        scd.append([v for v in per_te for _ in LOG_NE])
        acd.append([-11.0] * (len(LOG_NE) * len(LOG_TE)))
    read_adf11.cache_clear()
    return (_table(tmp_path / f"acd96_{symbol.lower()}.dat", acd),
            _table(tmp_path / f"scd96_{symbol.lower()}.dat", scd))


@pytest.fixture
def tables(tmp_path):
    return {"C": _element(tmp_path, "C", 6), "O": _element(tmp_path, "O", 8)}


@pytest.fixture(scope="module")
def sample():
    from vaft.omas.sample import sample_ods

    logging.disable(logging.WARNING)
    try:
        return sample_ods(SHOT)
    except FileNotFoundError:
        pytest.skip("the 39915 sample is repository-only")
    finally:
        logging.disable(logging.NOTSET)


def _ods(sample, *, phi=True, t_scale=1.0, with_thomson=True):
    """The equilibrium of 39915 plus a core_profiles slice at three of its times."""
    ods = ODS(consistency_check=False)
    ods["equilibrium"] = sample["equilibrium"]
    if phi:
        for i in range(len(ods["equilibrium.time_slice"])):
            base = f"equilibrium.time_slice.{i}.profiles_1d"
            ods[f"{base}.phi"] = 0.01 * np.asarray(ods[f"{base}.rho_tor_norm"], dtype=float) ** 2
    times = np.asarray(sample["equilibrium.time"], dtype=float)[[1, 2, 3]]
    ods["core_profiles.time"] = times
    for j, (i, t) in enumerate(zip((1, 2, 3), times)):
        base = f"core_profiles.profiles_1d.{j}"
        eq = f"equilibrium.time_slice.{i}.profiles_1d"
        psi = np.asarray(ods[f"{eq}.psi"], dtype=float)
        psi_n = (psi - psi[0]) / (psi[-1] - psi[0])
        rho_eq = np.asarray(ods[f"{eq}.rho_tor_norm"], dtype=float)
        order = np.argsort(rho_eq)
        ods[f"{base}.time"] = t
        ods[f"{base}.grid.rho_tor_norm"] = RHO
        # the profile grid's own flux label, consistent with this equilibrium slice
        ods[f"{base}.grid.rho_pol_norm"] = np.sqrt(np.clip(np.interp(RHO, rho_eq[order], psi_n[order]), 0.0, None))
        ods[f"{base}.electrons.temperature"] = t_scale * (10.0 + 150.0 * (1.0 - RHO**2) ** 1.5)
        ods[f"{base}.electrons.density"] = 0.4e19 + 1.4e19 * (1.0 - RHO**2)
    return ods


def _run(sample, tables, **kwargs):
    ods = _ods(sample, **{k: kwargs.pop(k) for k in ("phi", "t_scale", "with_thomson") if k in kwargs})
    return ods, kinetic_state_stages(ods, time=TIME, ionization="coronal", tables=tables, **kwargs)


def test_the_stage_names_and_order_are_the_contract():
    assert STAGE_NAMES == ("data", "thomson_fits", "romero_zeff", "impurity_amplitude", "ion_density",
                           "zeff_profile", "pressure_closure")


def test_every_stage_is_made_and_the_chain_is_complete(sample, tables):
    _, result = _run(sample, tables)
    assert result.names == STAGE_NAMES
    assert result.complete and all(stage.ok for stage in result.stages)
    assert result.product is not None
    assert result.time == TIME and result.rho.size == RHO.size


def test_the_constraint_is_closed_exactly_through_the_resistive_projection(sample, tables):
    """Stage 4 fixes the amplitude so that stage 6's Z_eff(rho) projects back onto the Romero scalar."""
    _, result = _run(sample, tables)
    z_res = result.stage("romero_zeff").fields["z_eff_resistive"]
    zeff = result.stage("zeff_profile").fields
    assert 1.0 < z_res < 8.0
    assert zeff["z_eff_resistive_equivalent"] == pytest.approx(z_res, abs=1e-6)
    assert abs(zeff["closure_error"]) < 1e-6
    assert result.stage("impurity_amplitude").provenance["resistive_target"] == pytest.approx(z_res)
    assert result.stage("impurity_amplitude").provenance["normalization"] == "resistive_closure"


def test_the_impurity_mix_is_carbon_and_oxygen_at_equal_densities(sample, tables):
    _, result = _run(sample, tables)
    amplitude = result.stage("impurity_amplitude")
    assert amplitude.fields["n_c_over_n_o"] == pytest.approx(1.0)
    n = amplitude.fields["n_by_element"]
    good = np.isfinite(n["C"]) & (n["O"] > 0)
    assert np.allclose(n["C"][good], n["O"][good])
    assert amplitude.assumptions == {"assume_nc_eq_no": 1, "n_c_over_n_o": 1.0}
    assert amplitude.fields["scale"] > 0.0 and np.all(amplitude.fields["n_I_eff"][np.isfinite(amplitude.fields["n_I_eff"])] >= 0.0)


def test_quasineutrality_and_zeff_follow_from_the_charge_states(sample, tables):
    _, result = _run(sample, tables)
    ne = result.stage("thomson_fits").fields["n_e"]
    ions = result.stage("ion_density").fields
    charge = ions["n_main"].copy()
    charge_square = ions["n_main"].copy()
    for states in ions["n_state"].values():
        for q, n in enumerate(states, start=1):
            charge += q * n
            charge_square += q * q * n
    assert np.allclose(charge, ne, rtol=1e-9), "n_e = n_main + sum_s Z_s n_s"
    zeff = result.stage("zeff_profile").fields["z_eff"]
    good = np.isfinite(zeff)
    assert np.allclose((charge_square / ne)[good], zeff[good], rtol=1e-6), "Z_eff = sum Z^2 n / n_e"
    assert np.allclose(ions["n_i_total"], ions["n_main"] + sum(n for states in ions["n_state"].values() for n in states))


def test_the_pressure_closure_is_a_check_with_t_i_equal_t_e(sample, tables):
    _, result = _run(sample, tables)
    closure = result.stage("pressure_closure")
    f = closure.fields
    assert np.array_equal(f["T_i"], result.stage("thomson_fits").fields["T_e"])
    assert np.allclose(f["p_thermal"], f["p_e"] + f["p_i"] + f["p_imp"], equal_nan=True)
    inside = result.stage("data").fields["inside"]
    assert np.allclose(f["residual"][inside], (f["p_eq"] - f["p_thermal"])[inside])
    assert closure.provenance["role"] == "consistency_check" and "never minimised" in closure.provenance["note"]
    assert closure.provenance["origin"] == "assumed" and closure.assumptions == {"assume_ti_eq_te": 1}
    # it is reported, not used: the Romero amplitude is the same whatever the pressure says
    assert result.stage("impurity_amplitude").provenance["role"] == "inferred_from_constraint"


def test_each_stage_says_what_it_adds_and_where_it_lands(sample, tables):
    _, result = _run(sample, tables)
    assert result.stage("data").adds[:3] == ("rho", "psi_norm", "inside")
    assert result.stage("romero_zeff").lands == {
        "z_eff_resistive": "core_profiles.global_quantities.z_eff_resistive[j]"}
    assert result.stage("zeff_profile").lands == {"z_eff": "core_profiles.profiles_1d[j].zeff"}
    assert set(result.stage("pressure_closure").lands) == {"p_e", "p_i_total", "p_thermal", "T_i"}
    assert result.stage("data").lands == {}, "the inputs are not written anywhere"
    for stage in result.stages:
        assert stage.label and set(stage.record()) >= {"name", "status", "adds", "assumptions", "provenance", "lands"}


def test_upto_gives_one_animation_frame_per_stage_cumulatively(sample, tables):
    _, result = _run(sample, tables)
    frames = [set(result.upto(name)) for name in STAGE_NAMES]
    assert all(frames[k] < frames[k + 1] for k in range(len(frames) - 1)), "each frame only adds"
    assert {"p_eq", "n_e", "T_e", "p_e"} <= frames[1]
    assert "z_eff_resistive" in frames[2] and "scale" not in frames[2]
    assert "scale" in frames[3] and "n_main" not in frames[3]
    assert "z_eff" in frames[5] and "residual" not in frames[5]
    assert "residual" in frames[6] and "p_eq" in frames[6]
    with pytest.raises(KeyError, match="no stage"):
        result.upto("nonsense")


def test_the_stored_product_is_what_the_last_stage_leaves_behind(sample, tables):
    _, result = _run(sample, tables)
    product = result.product
    base = "core_profiles.profiles_1d.0"
    f = result.stage("pressure_closure").fields
    assert np.allclose(np.asarray(product[f"{base}.pressure_thermal"], dtype=float), f["p_thermal"], equal_nan=True)
    assert np.allclose(np.asarray(product[f"{base}.electrons.pressure"], dtype=float), f["p_e"], equal_nan=True)
    assert np.array_equal(np.asarray(product[f"{base}.t_i_average"], dtype=float), f["T_i"])
    z = result.stage("zeff_profile").fields["z_eff"]
    stored = np.asarray(product[f"{base}.zeff"], dtype=float)
    good = np.isfinite(z)
    assert np.allclose(stored[good], z[good])
    assert np.asarray(product["core_profiles.global_quantities.z_eff_resistive"], dtype=float)[0] == pytest.approx(
        result.stage("romero_zeff").fields["z_eff_resistive"])


def test_the_assumptions_are_recorded_in_the_fit_parameters(sample, tables):
    _, result = _run(sample, tables)
    base = "core_profiles.profiles_1d.0"
    ion = result.product[f"{base}.ion.0.temperature_fit.parameters"]
    assert classify_ti_record(ion) == "assumed", "the ratio grammar: origin=assumed would read as unknown"
    fields = ti_record_fields(ion)
    assert fields["ti_te_ratio"] == "1" and fields["status"] == "assumed" and fields["assume_ti_eq_te"] == "1"
    assert fields["equilibrium_lineage"] == "magnetics" and fields["equilibrium_occurrence"] == "0"
    zeff = ti_record_fields(result.product[f"{base}.zeff_fit.parameters"])
    assert zeff["normalization"] == "resistive_closure" and zeff["assume_nc_eq_no"] == "1"
    assert zeff["n_c_over_n_o"] == "1.0" and float(zeff["resistive_target"]) == pytest.approx(
        result.stage("romero_zeff").fields["z_eff_resistive"])
    assert ti_record_fields(result.product[f"{base}.ion.0.density_fit.parameters"])["pressure_closure"] == "infer_kinetic_closure"


def test_the_input_ods_is_not_modified(sample, tables):
    ods = _ods(sample)
    before = path_value(ods, "core_profiles.profiles_1d.0.zeff", None)
    kinetic_state_stages(ods, time=TIME, ionization="coronal", tables=tables)
    assert before is None and path_value(ods, "core_profiles.profiles_1d.0.zeff", None) is None
    assert path_value(ods, "core_profiles.profiles_1d.0.ion", None) is None


def test_an_estimate_on_a_bound_stops_the_chain_and_names_the_stage(sample, tables):
    """No identifiable Romero value, no constraint: nothing downstream is filled in."""
    _, result = _run(sample, tables, t_scale=0.5)
    romero = result.stage("romero_zeff")
    assert not romero.ok and "bound_hit" in romero.reason and "no constraint" in romero.reason
    assert romero.fields["estimate_status"] == "bound_hit" and romero.fields["z_eff_resistive"] is None
    for name in STAGE_NAMES[3:]:
        stage = result.stage(name)
        assert not stage.ok and "romero_zeff" in stage.reason and stage.fields == {}
    assert result.stage("data").ok and result.stage("thomson_fits").ok
    assert result.product is None and not result.complete
    assert set(result.upto("thomson_fits")) >= {"p_eq", "n_e"}


def test_a_proxy_toroidal_coordinate_stops_the_chain_at_the_data(sample, tables):
    _, result = _run(sample, tables, phi=False)
    assert not result.stage("data").ok and "proxy" in result.stage("data").reason
    assert all(not stage.ok for stage in result.stages)
    assert all("data" in stage.reason for stage in result.stages[1:])


def test_a_time_with_no_equilibrium_is_named(sample, tables):
    ods = _ods(sample)
    result = kinetic_state_stages(ods, time=0.5, ionization="coronal", tables=tables)
    assert not result.stage("data").ok and "beyond" in result.stage("data").reason


def test_transient_ionization_without_a_plasma_age_stops_at_the_amplitude(sample, tables):
    ods = _ods(sample)
    result = kinetic_state_stages(ods, time=TIME, ionization="transient", tables=tables)
    assert result.stage("romero_zeff").ok
    amplitude = result.stage("impurity_amplitude")
    assert not amplitude.ok and "ionisation age" in amplitude.reason
    assert result.product is None


def test_an_unknown_ion_temperature_mode_is_refused(sample, tables):
    with pytest.raises(ValueError, match="ion_temperature"):
        _run(sample, tables, ion_temperature="measured")


def test_the_inferred_ion_temperature_closes_the_pressure_with_the_romero_composition(sample, tables):
    _, result = _run(sample, tables, ion_temperature="inferred")
    assert result.stage("impurity_amplitude").ok, "the Romero-closed composition is unchanged by the T_i mode"
    closure = result.stage("pressure_closure")
    if not closure.ok:  # the sample's pressure may leave no significant p_i: named, not guessed
        assert "p_i" in closure.reason or "T_i" in closure.reason or "eligible" in closure.reason
        return
    f = closure.fields
    assert closure.provenance["origin"] == "inferred" and closure.provenance["role"] == "closure"
    assert closure.assumptions == {"assume_ti_eq_te": 0}
    valid = np.isfinite(f["T_i"])
    assert valid.any()
    assert np.allclose(f["p_thermal"][valid], f["p_eq"][valid], rtol=1e-6), "the partition closes p_thermal onto p_eq"
    record = str(result.product["core_profiles.profiles_1d.%d.ion.0.temperature_fit.parameters" % 0])
    assert "origin=inferred" in record and "not measured" in record


def test_the_ratio_estimate_is_widened_by_the_radial_spread_not_just_the_formal_error():
    from vaft.omas.kinetic_state_inference import _ti_ratio_estimate

    te = np.full(100, 100.0)
    flat = _ti_ratio_estimate(np.full(100, 1.5), np.full(100, 5.0), te)
    assert flat["ratio"] == pytest.approx(1.5) and flat["sigma"] == pytest.approx(flat["sigma_formal"])
    varying = _ti_ratio_estimate(np.linspace(0.8, 2.2, 100), np.full(100, 5.0), te, n_independent=4)
    assert varying["sigma"] > 5 * varying["sigma_formal"], "correlated points: the spread sets the error"
    with pytest.raises(ValueError, match="fewer than three"):
        _ti_ratio_estimate(np.array([1.0, np.nan]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_the_ratio_mode_puts_one_t_i_over_t_e_on_the_whole_profile(sample, tables):
    _, result = _run(sample, tables, ion_temperature="inferred_ratio")
    closure = result.stage("pressure_closure")
    if not closure.ok:
        assert "T_i" in closure.reason or "p_i" in closure.reason or "ratio" in closure.reason
        return
    f = closure.fields
    r = f["ti_ratio_estimate"]["ratio"]
    assert np.allclose(f["T_i_over_T_e"], r) and f["T_i_pointwise"] is not None
    assert closure.provenance["role"] == "closure_scalar"
    assert closure.assumptions == {"assume_ti_eq_te": 0, "assume_ti_profile_shape_eq_te": 1}


def test_the_ratio_profile_picks_the_order_the_points_support():
    from vaft.omas.kinetic_state_inference import _ti_ratio_profile

    rho = np.linspace(0.1, 0.9, 60)
    te, sig = np.full(60, 100.0), np.full(60, 2.0)
    flat = _ti_ratio_profile(rho, np.full(60, 1.3), sig, te, np.linspace(0, 1, 11), n_independent=8)
    assert flat["order"] == 0 and flat["ratio"] == pytest.approx(1.3)
    slope = _ti_ratio_profile(rho, 1.0 + rho, sig, te, np.linspace(0, 1, 11), n_independent=8)
    assert slope["order"] >= 1 and slope["ratio"][-1] == pytest.approx(2.0, rel=0.05)
    out = np.linspace(0, 1, 11)
    assert slope["extrapolated"][0] and slope["extrapolated"][-1] and not slope["extrapolated"][5]


def test_the_ratio_profile_mode_is_continuous_and_marks_its_extrapolation(sample, tables):
    _, result = _run(sample, tables, ion_temperature="inferred_profile")
    closure = result.stage("pressure_closure")
    if not closure.ok:
        assert "T_i" in closure.reason or "p_i" in closure.reason or "ratio" in closure.reason
        return
    f = closure.fields
    assert np.all(np.isfinite(f["T_i"])), "continuous over the whole profile"
    assert closure.provenance["role"] == "closure_profile"
    assert closure.assumptions["ti_extrapolated_outside_rho_valid"] == 1


def test_a_window_ratio_is_the_inverse_variance_mean_widened_by_excess_scatter():
    from vaft.omas.kinetic_state_inference import combine_ti_ratios

    quiet = combine_ti_ratios([1.2, 1.3, 1.25], [0.2, 0.2, 0.2])
    assert quiet["ratio"] == pytest.approx(1.25) and quiet["sigma"] == pytest.approx(quiet["sigma_formal"])
    noisy = combine_ti_ratios([0.5, 2.0, 1.0, 1.8], [0.1, 0.1, 0.1, 0.1])
    assert noisy["sigma"] > 3 * noisy["sigma_formal"]
    with pytest.raises(ValueError):
        combine_ti_ratios([1.0], [0.1])


def test_a_sloped_fit_over_a_wide_axis_gap_warns_that_the_axis_is_extrapolated():
    from vaft.omas.kinetic_state_inference import _ti_ratio_profile

    rho = np.linspace(0.35, 0.95, 60)
    fit = _ti_ratio_profile(rho, 0.5 + 1.5 * rho, np.full(60, 1.0), np.full(60, 100.0),
                            np.linspace(0, 1, 11), n_independent=10)
    assert fit["order"] >= 1 and fit["warnings"] and "axis_extrapolated" in fit["warnings"][0]
    flat = _ti_ratio_profile(rho, np.full(60, 1.2), np.full(60, 1.0), np.full(60, 100.0),
                             np.linspace(0, 1, 11), n_independent=10)
    assert flat["order"] == 0 and flat["warnings"] == [], "a constant ratio has no axis to extrapolate"
