"""Impurity-mixture algebra (#1565 Stage A): the issue's reference values, conservation and round trips."""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.atomic import (
    mean_charge_from_charge_state_densities,
    mean_square_charge_from_charge_state_densities,
    z_eff_from_n_s_Z_s,
)
from vaft.formula.impurity import (
    dilution_fraction_from_species,
    dilution_fraction_from_zeff,
    expand_effective_impurity,
    impurity_mixture_moments,
    main_ion_density_from_species,
    main_ion_density_from_zeff,
    main_ion_fraction_from_zeff,
    reduce_impurity_mixture,
    solve_impurity_mixture_for_target_zeff,
)

CO_W = [0.5, 0.5]
CO_Z = [6.0, 8.0]
CO_A = [12.0, 16.0]


def _plasma_zeff(fractions, charges, main_fraction, main_charge=1.0):
    """Z_eff of the whole plasma (main ion included), per unit n_e."""
    n = np.append(np.atleast_1d(fractions), main_fraction)
    z = np.append(np.atleast_1d(charges), main_charge)
    return z_eff_from_n_s_Z_s(n, z, n_e=1.0)


# --- the #1565 reference case: C6+:O8+ = 1:1, Z_eff = 2 ---------------------


def test_reference_moments():
    m = impurity_mixture_moments(CO_W, CO_Z, CO_A)
    assert m.S1 == pytest.approx(7.0)
    assert m.S2 == pytest.approx(50.0)
    assert m.A_bar == pytest.approx(14.0)


def test_reference_densities_are_one_eighty_sixth():
    sol = solve_impurity_mixture_for_target_zeff(2.0, CO_W, CO_Z)
    np.testing.assert_allclose(sol.impurity_fractions, [1 / 86, 1 / 86], rtol=1e-14)
    assert sol.alpha == pytest.approx(1 / 43, rel=1e-14)
    assert sol.main_ion_fraction == pytest.approx(1 - 14 / 86, rel=1e-14)
    assert _plasma_zeff(sol.impurity_fractions, CO_Z, sol.main_ion_fraction) == pytest.approx(2.0, rel=1e-14)


def test_reference_reduction():
    sol = solve_impurity_mixture_for_target_zeff(2.0, CO_W, CO_Z)
    eff = reduce_impurity_mixture(sol.impurity_fractions, CO_Z, CO_A)
    assert eff.charge == pytest.approx(50 / 7, rel=1e-14)
    assert eff.density == pytest.approx(49 / 2150, rel=1e-14)
    assert eff.mass == pytest.approx(14.2857, abs=1e-4)
    assert eff.mass == pytest.approx(14.0 * 50 / 49, rel=1e-14)   # A_bar S2 / S1^2
    # the pseudo-impurity alone reproduces Z_eff = 2 with the same main ion
    assert _plasma_zeff(eff.density, eff.charge, sol.main_ion_fraction) == pytest.approx(2.0, rel=1e-14)


def test_reduction_conserves_charge_z2_and_mass():
    n = np.array([0.013, 0.004, 0.002])
    z = np.array([6.0, 8.0, 7.0])
    a = np.array([12.011, 15.999, 14.007])
    eff = reduce_impurity_mixture(n, z, a)
    assert eff.density * eff.charge == pytest.approx(np.sum(n * z), rel=1e-14)
    assert eff.density * eff.charge**2 == pytest.approx(np.sum(n * z**2), rel=1e-14)
    assert eff.density * eff.mass == pytest.approx(np.sum(n * a), rel=1e-14)


def test_round_trip_reduce_expand():
    w = np.array([0.3, 0.4, 0.3])
    z = np.array([6.0, 8.0, 7.0])
    a = np.array([12.0, 16.0, 14.0])
    sol = solve_impurity_mixture_for_target_zeff(2.0, w, z)
    eff = reduce_impurity_mixture(sol.impurity_fractions, z, a)
    back = expand_effective_impurity(eff.density, eff.charge, w, z, eff.mass, a)
    np.testing.assert_allclose(back, sol.impurity_fractions, rtol=1e-12)


def test_expand_refuses_a_charge_the_composition_does_not_reduce_to():
    with pytest.raises(ValueError, match="S2/S1"):
        expand_effective_impurity(0.02, 6.0, CO_W, CO_Z)
    with pytest.raises(ValueError, match="effective mass"):
        expand_effective_impurity(0.02, 50 / 7, CO_W, CO_Z, mass=12.0, masses=CO_A)
    with pytest.raises(ValueError, match="pass masses"):
        expand_effective_impurity(0.02, 50 / 7, CO_W, CO_Z, mass=14.0)


# --- arbitrary mixtures, main-ion charge, profiles ----------------------------


def test_non_equal_mixture_reaches_its_target():
    w = [0.3, 0.4, 0.3]
    z = [6.0, 8.0, 7.0]
    sol = solve_impurity_mixture_for_target_zeff(1.7, w, z)
    np.testing.assert_allclose(sol.impurity_fractions / sol.alpha, w, rtol=1e-14)
    assert _plasma_zeff(sol.impurity_fractions, z, sol.main_ion_fraction) == pytest.approx(1.7, rel=1e-13)
    # quasi-neutrality
    assert sol.main_ion_fraction + np.sum(sol.impurity_fractions * np.asarray(z)) == pytest.approx(1.0)


def test_helium_main_ion():
    sol = solve_impurity_mixture_for_target_zeff(3.0, CO_W, CO_Z, main_ion_charge=2.0)
    assert 2.0 * sol.main_ion_fraction + np.sum(sol.impurity_fractions * CO_Z) == pytest.approx(1.0)
    assert _plasma_zeff(sol.impurity_fractions, CO_Z, sol.main_ion_fraction, 2.0) == pytest.approx(3.0)


def test_profile_target_broadcasts_over_rho():
    target = np.linspace(1.0, 2.5, 7)
    sol = solve_impurity_mixture_for_target_zeff(target, CO_W, CO_Z)
    assert sol.impurity_fractions.shape == (7, 2)
    zeff = [
        _plasma_zeff(sol.impurity_fractions[k], CO_Z, sol.main_ion_fraction[k]) for k in range(7)
    ]
    np.testing.assert_allclose(zeff, target, rtol=1e-13)
    assert sol.impurity_fractions[0] == pytest.approx([0.0, 0.0])


def test_profile_charges_broadcast_over_rho():
    # partially ionised carbon at the edge: per-rho charges along the leading axis
    z = np.array([[6.0, 8.0], [5.0, 8.0], [4.0, 7.5]])
    m = impurity_mixture_moments(CO_W, z)
    np.testing.assert_allclose(m.S1, [7.0, 6.5, 5.75])
    eff = reduce_impurity_mixture(np.full((3, 2), 0.01), z)
    assert eff.charge.shape == (3,)


# --- invalid input -------------------------------------------------------------


@pytest.mark.parametrize("target", [0.9, 50 / 7 + 0.01])
def test_target_outside_the_reachable_interval_raises(target):
    with pytest.raises(ValueError, match="target_zeff"):
        solve_impurity_mixture_for_target_zeff(target, CO_W, CO_Z)


def test_target_at_the_ceiling_leaves_no_main_ion():
    sol = solve_impurity_mixture_for_target_zeff(50 / 7, CO_W, CO_Z)
    assert sol.main_ion_fraction == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("charges", [[0.0, 8.0], [-6.0, 8.0], [np.nan, 8.0]])
def test_invalid_charge_states_raise(charges):
    with pytest.raises(ValueError):
        solve_impurity_mixture_for_target_zeff(2.0, CO_W, charges)


def test_weights_must_be_normalised_and_non_negative():
    with pytest.raises(ValueError, match="sum to one"):
        impurity_mixture_moments([1.0, 1.0], CO_Z)
    with pytest.raises(ValueError, match="non-negative"):
        impurity_mixture_moments([1.5, -0.5], CO_Z)
    with pytest.raises(ValueError, match="species axis"):
        impurity_mixture_moments(1.0, 6.0)


def test_a_mixture_no_more_charged_than_the_main_ion_cannot_raise_zeff():
    with pytest.raises(ValueError, match="no more charged"):
        solve_impurity_mixture_for_target_zeff(1.5, [1.0], [1.0])


# --- dilution (#1565 comment) --------------------------------------------------


def test_dilution_from_zeff_matches_the_species_route():
    sol = solve_impurity_mixture_for_target_zeff(2.0, CO_W, CO_Z)
    eff = reduce_impurity_mixture(sol.impurity_fractions, CO_Z)
    f_main = main_ion_fraction_from_zeff(2.0, eff.charge)
    assert f_main == pytest.approx(36 / 43, rel=1e-14)
    assert f_main == pytest.approx(sol.main_ion_fraction, rel=1e-14)
    assert dilution_fraction_from_zeff(2.0, eff.charge) == pytest.approx(7 / 43, rel=1e-13)
    assert dilution_fraction_from_species(1.0, sol.impurity_fractions, CO_Z) == pytest.approx(7 / 43, rel=1e-13)


def test_same_zeff_different_dilution():
    """A Z_eff scan is not a dilution scan (#1565 comment)."""
    assert main_ion_fraction_from_zeff(2.0, 6.0) == pytest.approx(0.8)
    assert main_ion_fraction_from_zeff(2.0, 50 / 7) == pytest.approx(36 / 43)


def test_main_ion_density_routes_agree_on_profiles():
    ne = np.array([1e19, 2e19, 3e19])
    n_imp = np.outer(ne, [1 / 86, 1 / 86])
    from_species = main_ion_density_from_species(ne, n_imp, CO_Z)
    from_zeff = main_ion_density_from_zeff(ne, 2.0, 50 / 7)
    np.testing.assert_allclose(from_species, from_zeff, rtol=1e-13)


def test_overcharged_impurities_raise():
    with pytest.raises(ValueError, match="more charge"):
        main_ion_density_from_species(1.0, [0.1, 0.1], CO_Z)
    with pytest.raises(ValueError, match=r"\[Z_m, Z_I\]"):
        main_ion_fraction_from_zeff(7.0, 6.0)


# --- <Z^2> of a charge-state distribution --------------------------------------


def test_mean_square_charge_keeps_the_variance_mean_charge_drops():
    f = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5])   # C4+ and C6+, half each
    assert mean_charge_from_charge_state_densities(f) == pytest.approx(5.0)
    assert mean_square_charge_from_charge_state_densities(f) == pytest.approx(26.0)
    profile = np.vstack([f, np.eye(7)[6]])
    np.testing.assert_allclose(mean_square_charge_from_charge_state_densities(profile), [26.0, 36.0])
