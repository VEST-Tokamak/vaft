"""The #1627 §2 summary layer: ordering quantities of measured states (Lane AP, #1629).

Each builder is checked against an independent hand computation from the
textbook definition, so the test catches a wrong kernel, a wrong unit or a
wrong scale -- not only a crash. Missing inputs must give NaN, never a guess.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vaft.process.ordering_state import (
    COVERAGE_TIERS,
    contract_population,
    global_ordering_quantities,
    ordering_margins,
    ordering_coverage,
    ordering_table,
    profile_ordering_quantities,
    time_history_ordering_quantities,
)
from vaft.validation.orderings import ORDERING_QUANTITIES

MU0 = 4e-7 * math.pi
QE, ME, MP, C = 1.602176634e-19, 9.10938356e-31, 1.67262192e-27, 299792458.0
STATE = dict(minor_radius=0.27, major_radius=0.38, b0=0.18, n_e=1.0e19, t_e=50.0, z_eff=2.0, ln_lambda=15.0)


def _spitzer(t_e, z_eff, ln_lambda):
    return 5.2e-5 * z_eff * ln_lambda / t_e ** 1.5


def test_global_quantities_match_their_definitions():
    g = global_ordering_quantities(**STATE, beta=0.03)
    v_a = STATE["b0"] / math.sqrt(MU0 * STATE["n_e"] * MP)
    eta = _spitzer(50.0, 2.0, 15.0)
    assert g["lundquist_number"] == pytest.approx(MU0 * 0.27 * v_a / eta, rel=1e-3)
    d_i = C / math.sqrt(STATE["n_e"] * QE ** 2 / (8.8541878128e-12 * MP))
    assert g["ion_skin_depth_over_a"] == pytest.approx(d_i / 0.27, rel=1e-6)
    assert g["inverse_aspect_ratio"] == pytest.approx(0.27 / 0.38)
    assert g["beta_over_inverse_aspect_ratio"] == pytest.approx(0.03 / (0.27 / 0.38))
    assert set(g) <= set(ORDERING_QUANTITIES)


@pytest.mark.parametrize("missing", ["n_e", "t_e", "z_eff", "ln_lambda"])
def test_a_missing_input_leaves_only_what_depends_on_it_unassessed(missing):
    g = global_ordering_quantities(**{**STATE, missing: None})
    assert math.isnan(g["lundquist_number"])
    assert math.isfinite(g["inverse_aspect_ratio"])
    assert math.isnan(g["beta"])  # never defaulted
    assert math.isfinite(g["ion_skin_depth_over_a"]) == (missing != "n_e")


def test_time_history_quantities_match_their_definitions_and_refuse_a_zero_rate():
    th = time_history_ordering_quantities(**{k: v for k, v in STATE.items() if k != "major_radius"},
                                          plasma_current=6.0e4, current_rate=-5.0e6, plasma_age=0.01)
    tau_a = 0.27 / (0.18 / math.sqrt(MU0 * 1e19 * MP))
    assert th["tau_evolution_over_tau_alfven"] == pytest.approx((6e4 / 5e6) / tau_a, rel=1e-3)
    tau_r = MU0 * 0.27 ** 2 / _spitzer(50.0, 2.0, 15.0)
    assert th["tau_age_over_tau_resistive"] == pytest.approx(0.01 / tau_r, rel=1e-3)
    flat = time_history_ordering_quantities(**{k: v for k, v in STATE.items() if k != "major_radius"},
                                            plasma_current=6.0e4, current_rate=0.0)
    assert math.isnan(flat["tau_evolution_over_tau_alfven"])


def _profiles(n=12, with_ti=True):
    r = np.linspace(0.02, 0.25, n)
    l_t = 0.08
    t_e = 120.0 * np.exp(-r / l_t)  # constant L_Te = 0.08 m
    return dict(minor_radius_coordinate=r, n_e=np.full(n, 1e19), t_e=t_e, magnetic_field=0.18,
                safety_factor=np.full(n, 2.0), major_radius=0.38, z_eff=2.0, ln_lambda=15.0,
                t_i=0.5 * t_e if with_ti else None), l_t


def test_profile_gradient_lengths_and_gyroradii_match_their_definitions():
    kwargs, l_t = _profiles()
    p = profile_ordering_quantities(**kwargs)
    rho_s = np.sqrt(MP * kwargs["t_e"] * QE) / (QE * 0.18)
    inner = slice(2, -2)  # one-sided differences at the ends are first order
    np.testing.assert_allclose(p["rho_s_over_LTe"][inner], (rho_s / l_t)[inner], rtol=2e-2)
    rho_i = np.sqrt(kwargs["t_i"] * QE / MP) * MP / (QE * 0.18)
    np.testing.assert_allclose(p["rho_i_over_LTi"][inner], (rho_i / l_t)[inner], rtol=2e-2)
    tau_e = 3.44e5 * kwargs["t_e"] ** 1.5 / (2.0 * 1e13 * 15.0)  # NRL: n in cm^-3
    lam_e = np.sqrt(kwargs["t_e"] * QE / ME) * tau_e
    np.testing.assert_allclose(p["electron_parallel_knudsen_number"], lam_e / (2.0 * 0.38), rtol=1e-3)
    assert set(p) <= set(ORDERING_QUANTITIES)


def test_without_an_ion_temperature_every_ion_quantity_is_unassessed():
    kwargs, _ = _profiles(with_ti=False)
    p = profile_ordering_quantities(**kwargs)
    for name in ("rho_i_over_LTi", "ion_parallel_knudsen_number", "ion_magnetization", "ion_collisionality"):
        assert np.all(np.isnan(p[name])), name
    assert np.all(np.isfinite(p["electron_magnetization"]))


def test_a_bad_sample_is_unassessed_and_the_rest_survives():
    kwargs, _ = _profiles()
    kwargs["n_e"] = kwargs["n_e"].copy()
    kwargs["n_e"][5] = np.nan
    p = profile_ordering_quantities(**kwargs)
    assert np.isnan(p["electron_magnetization"][5])
    assert np.isfinite(p["electron_magnetization"][4])


def test_profiles_must_share_one_increasing_grid():
    kwargs, _ = _profiles()
    with pytest.raises(ValueError, match="shape"):
        profile_ordering_quantities(**{**kwargs, "t_e": kwargs["t_e"][:-1]})
    with pytest.raises(ValueError, match="increase"):
        profile_ordering_quantities(**{**kwargs, "minor_radius_coordinate": kwargs["minor_radius_coordinate"][::-1]})


def test_the_table_builder_reads_columns_and_treats_absent_ones_as_missing():
    states = pd.DataFrame({"a_m": [0.27, 0.27], "r_geo_m": [0.38, 0.38], "b_t_T": [0.18, 0.18],
                           "density": [1e19, np.nan], "i_p_A": [6e4, 6e4], "dip_dt_A_s": [-5e6, -5e6]},
                          index=["s1", "s2"])
    table = ordering_table(states, columns={"n_e": "density"})
    assert list(table.index) == ["s1", "s2"]
    assert math.isfinite(table.loc["s1", "ion_skin_depth_over_a"])
    assert math.isnan(table.loc["s2", "ion_skin_depth_over_a"])
    assert table["lundquist_number"].isna().all()  # no T_e column at all
    assert table.attrs["columns"]["n_e"] == "density"
    assert "t_e" not in table.attrs["columns"]


def test_coverage_counts_every_registered_quantity_by_tier():
    table = pd.DataFrame({"inverse_aspect_ratio": [0.7, 0.7, np.nan], "lundquist_number": [1e5, np.nan, np.nan]})
    coverage = ordering_coverage(table)
    assert set(coverage.index) == set(ORDERING_QUANTITIES)
    assert coverage.loc["inverse_aspect_ratio", "fraction"] == pytest.approx(2 / 3)
    assert coverage.loc["k_perp_rho_i", "tier"] == "tier4_mode_and_layer"
    assert coverage.loc["sonic_mach_number", "tier"] == "tier3_flow"
    assert coverage.loc["k_perp_rho_i", "assessed"] == 0
    assert set(coverage["tier"]) <= set(COVERAGE_TIERS)


def test_coverage_counts_profile_quantities_on_the_points_table():
    states = pd.DataFrame({"lundquist_number": [1e5, np.nan]})
    points = pd.DataFrame({"rho_s_over_LTe": [0.01, 0.02, np.nan, 0.03]})
    coverage = ordering_coverage(states, points)
    assert (coverage.loc["lundquist_number", "of"], coverage.loc["lundquist_number", "fraction"]) == ("states", 0.5)
    assert (coverage.loc["rho_s_over_LTe", "of"], coverage.loc["rho_s_over_LTe", "fraction"]) == ("points", 0.75)


def test_margins_count_a_global_ordering_once_per_state_not_per_point():
    from vaft.validation.orderings import contract

    states = pd.DataFrame({"lundquist_number": [1e4, 1e6]})
    points = pd.DataFrame({"lundquist_number": [1e4] * 9 + [1e6], "rho_s_over_LTe": [0.01] * 10})
    margins = ordering_margins({"m": contract("ideal_single_fluid_mhd"), "f": contract("flr_small_fluid")},
                               states, points)
    s_row = margins.loc["lundquist_number (large)"]
    assert (s_row["per"], s_row["n"], s_row["median_margin"]) == ("state", 2, pytest.approx(5.0))
    assert margins.loc["rho_s_over_LTe (small)", "per"] == "point"
    assert margins.loc["rho_s_over_LTe (small)", "median_margin"] == pytest.approx(2.0)
    assert list(margins["median_margin"]) == sorted(margins["median_margin"])


def test_a_state_level_contract_is_evaluated_per_state_and_a_local_one_per_point():
    from vaft.validation.orderings import contract

    states = pd.DataFrame({"tau_age_over_tau_resistive": [0.5, 2.0]})
    points = pd.DataFrame({"tau_age_over_tau_resistive": [0.5] * 5 + [2.0], "electron_collisionality": [3.0] * 6})
    relaxed = contract_population(contract("resistively_relaxed_current"), states, points)
    assert relaxed.attrs["evaluated_on"] == "states" and len(relaxed) == 2
    banana = contract_population(contract("banana_regime_neoclassical"), states, points)
    assert banana.attrs["evaluated_on"] == "points" and len(banana) == 6


def test_the_population_summary_separates_decided_from_evaluated_orderings():
    from vaft.validation.applicability import evaluate_population, summarize_population
    from vaft.validation.orderings import contract

    c = contract("braginskii_two_fluid")
    rows = pd.DataFrame({"electron_parallel_knudsen_number": [0.1, 3.0, 0.2],
                         "electron_magnetization": [1e4, 1e4, 1e4]})
    summary = summarize_population(evaluate_population(c, rows), c)
    assert summary["n"] == 3
    assert summary["OUTSIDE"] == pytest.approx(1 / 3)
    assert summary["UNASSESSED"] == pytest.approx(2 / 3)
    assert summary["evaluated_hold"] == pytest.approx(2 / 3)
    assert summary["main_limitation"] == "electron_parallel_knudsen_number"
    assert "ion_magnetization" in summary["unassessed"]
    assert set(summary["assessable"]) == {"electron_parallel_knudsen_number", "electron_magnetization"}


def test_a_missing_safety_factor_removes_only_the_quantities_that_use_it():
    kwargs, _ = _profiles()
    p = profile_ordering_quantities(**{**kwargs, "safety_factor": np.full(12, np.nan)})
    for name in ("electron_parallel_knudsen_number", "electron_collisionality",
                 "ion_parallel_knudsen_number", "ion_collisionality"):
        assert np.all(np.isnan(p[name])), name
    for name in ("rho_s_over_LTe", "electron_magnetization", "debye_length_over_L", "rho_i_over_LTi",
                 "ion_magnetization"):
        assert np.any(np.isfinite(p[name])), name


def test_magnetization_collisionality_and_debye_match_their_definitions():
    kwargs, l_t = _profiles()
    p = profile_ordering_quantities(**kwargs)
    t_e, t_i, n, r = kwargs["t_e"], kwargs["t_i"], 1e19, kwargs["minor_radius_coordinate"]
    tau_e = 3.44e5 * t_e ** 1.5 / (2.0 * 1e13 * 15.0)
    np.testing.assert_allclose(p["electron_magnetization"], QE * 0.18 / ME * tau_e, rtol=1e-3)
    tau_i = 2.09e7 * t_i ** 1.5 / (1e13 * 15.0)  # NRL, mu = 1, Z = 1
    np.testing.assert_allclose(p["ion_magnetization"], QE * 0.18 / MP * tau_i, rtol=1e-3)
    lam_i = np.sqrt(t_i * QE / MP) * tau_i
    np.testing.assert_allclose(p["ion_parallel_knudsen_number"], lam_i / (2.0 * 0.38), rtol=1e-3)
    lam_d = np.sqrt(8.8541878128e-12 * t_e / (n * QE))
    inner = slice(2, -2)
    np.testing.assert_allclose(p["debye_length_over_L"][inner], (lam_d / l_t)[inner], rtol=2e-2)
    # Sauter Eq. (18b): nu*_e = 6.921e-18 q R n_e Z_eff lnLambda / (T_e^2 eps^1.5)
    eps = r / 0.38
    nu_e = 6.921e-18 * 2.0 * 0.38 * n * 2.0 * 15.0 / (t_e ** 2 * eps ** 1.5)
    np.testing.assert_allclose(p["electron_collisionality"], nu_e, rtol=1e-3)


def test_the_table_feeds_the_ordering_contracts_directly():
    from vaft.validation.applicability import evaluate_population
    from vaft.validation.orderings import contract

    states = pd.DataFrame({"a_m": [0.27], "r_geo_m": [0.38], "b_t_T": [0.18], "n_e_m3": [1e19],
                           "t_e_eV": [50.0], "z_eff": [2.0], "ln_lambda": [15.0]})
    population = evaluate_population(contract("ideal_single_fluid_mhd"), ordering_table(states))
    assert population.loc[0, "lundquist_number_status"] == "SUPPORTED"
    assert population.loc[0, "status"] == "UNASSESSED"  # profile and mode orderings are absent


def test_a_partially_known_ion_temperature_keeps_its_gradient_where_known():
    kwargs, l_t = _profiles()
    t_i = kwargs["t_i"].copy()
    t_i[:3] = np.nan  # e.g. inferred only inside the Thomson span
    p = profile_ordering_quantities(**{**kwargs, "t_i": t_i})
    assert np.all(np.isnan(p["rho_i_over_LTi"][:3]))
    assert np.all(np.isfinite(p["rho_i_over_LTi"][4:-1]))


def test_no_assessable_ordering_reads_as_no_evidence_not_as_failure():
    from vaft.validation.applicability import evaluate_population, summarize_population
    from vaft.validation.orderings import contract

    c = contract("local_neoclassical")
    summary = summarize_population(evaluate_population(c, pd.DataFrame({"unrelated": [1.0, 2.0]})), c)
    assert math.isnan(summary["evaluated_hold"])
    assert summary["UNASSESSED"] == 1.0 and summary["assessable"] == ()
