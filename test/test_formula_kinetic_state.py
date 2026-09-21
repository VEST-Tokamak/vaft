"""Numeric coverage for the derived kinetic-state formulas.

The single-impurity inversion of Z_eff (``vaft.formula.atomic``),
the normalised gradient scale length ``a/L`` (``vaft.formula.utils``) and the
species thermal pressures (``vaft.formula.equilibrium``).  Every expected value
here is computed by hand from the definition, not by calling the formula.
"""

from __future__ import annotations

import numpy as np
import pytest

import vaft.formula
from vaft.formula.atomic import impurity_fraction_from_effective_charge, z_eff_from_n_s_Z_s
from vaft.formula.constants import QE
from vaft.formula.equilibrium import electron_pressure, ion_pressure
from vaft.formula.utils import normalized_gradient_scale_length


# Z_eff itself is z_eff_from_n_s_Z_s (#968), covered in its own tests.


# --- single-impurity inversion --------------------------------------------------


def test_carbon_at_z_eff_two_is_one_thirtieth():
    assert impurity_fraction_from_effective_charge(2.0, 6.0) == pytest.approx(1.0 / 30.0, rel=1e-14)


def test_no_impurity_at_unit_effective_charge():
    assert impurity_fraction_from_effective_charge(1.0, 6.0) == 0.0


@pytest.mark.parametrize("z_imp", [2.0, 6.0, 8.0, 74.0])
@pytest.mark.parametrize("z_eff", [1.0, 1.3, 2.0])
def test_inversion_round_trips_through_effective_charge(z_eff, z_imp):
    if z_eff > z_imp:
        pytest.skip("unreachable")
    n_e = 5.0e18
    frac = impurity_fraction_from_effective_charge(z_eff, z_imp)
    n_imp = frac * n_e
    n_h = n_e - z_imp * n_imp
    back = z_eff_from_n_s_Z_s([n_h, n_imp], [1.0, z_imp], n_e)
    assert back == pytest.approx(z_eff, rel=1e-12)


def test_inversion_is_elementwise_on_a_profile():
    z_eff = np.array([1.0, 1.5, 2.0])
    np.testing.assert_allclose(
        impurity_fraction_from_effective_charge(z_eff, 6.0), (z_eff - 1.0) / 30.0, rtol=1e-14
    )


@pytest.mark.parametrize("z_eff", [0.9, 6.5, np.nan])
def test_inversion_rejects_unreachable_effective_charge(z_eff):
    with pytest.raises(ValueError, match="Z_eff"):
        impurity_fraction_from_effective_charge(z_eff, 6.0)


def test_inversion_needs_a_multiply_charged_impurity():
    with pytest.raises(ValueError, match="Z > 1"):
        impurity_fraction_from_effective_charge(1.0, 1.0)


# --- a/L --------------------------------------------------------------------------


def test_exponential_profile_has_constant_normalised_gradient():
    a, L = 0.4, 0.1
    x = np.linspace(0.0, 0.4, 4001)
    y = 3.0 * np.exp(-x / L)
    a_over_l = normalized_gradient_scale_length(x, y, a)
    np.testing.assert_allclose(a_over_l[1:-1], a / L, rtol=1e-5)
    # the one-sided end points are first-order, so only loosely
    np.testing.assert_allclose(a_over_l[[0, -1]], a / L, rtol=5e-3)


def test_parabola_matches_its_analytic_normalised_gradient_in_the_interior():
    # y = 1 - x^2 on a uniform grid: the central difference is exact for a
    # quadratic, so the interior agrees to rounding.
    a = 0.5
    x = np.linspace(0.0, 0.9, 91)
    y = 1.0 - x**2
    expected = a * 2.0 * x / (1.0 - x**2)
    np.testing.assert_allclose(
        normalized_gradient_scale_length(x, y, a)[1:-1], expected[1:-1], rtol=1e-11
    )


def test_normalised_gradient_differentiates_the_last_axis_of_a_stack():
    x = np.linspace(0.0, 1.0, 201)
    stack = np.stack([np.exp(-x / 0.2), 2.0 * np.exp(-x / 0.5)])
    out = normalized_gradient_scale_length(x, stack, 1.0)
    assert out.shape == stack.shape
    np.testing.assert_allclose(out[0, 1:-1], 5.0, rtol=1e-3)
    np.testing.assert_allclose(out[1, 1:-1], 2.0, rtol=1e-3)


def test_normalised_gradient_is_nan_where_the_profile_vanishes():
    x = np.linspace(0.0, 1.0, 11)
    y = 1.0 - x
    with pytest.warns(RuntimeWarning, match="returning nan"):
        out = normalized_gradient_scale_length(x, y, 1.0)
    assert np.isnan(out[-1])
    assert np.all(np.isfinite(out[:-1]))


# --- thermal pressure -------------------------------------------------------------


def test_electron_pressure_of_1e19_at_100_eV():
    p = electron_pressure(1.0e19, 100.0)
    assert p == pytest.approx(1.0e19 * 100.0 * QE, rel=1e-15)
    assert p == pytest.approx(160.2177, rel=1e-6)


def test_ion_pressure_is_the_same_law_per_species():
    n_i = np.array([1.0e19, 5.0e18])
    T_i = np.array([50.0, 200.0])
    np.testing.assert_allclose(ion_pressure(n_i, T_i), n_i * T_i * QE, rtol=1e-15)


def test_pressure_rejects_negative_temperature():
    with pytest.raises(ValueError, match="T_e"):
        electron_pressure(1.0e19, -1.0)


# --- namespace ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "z_eff_from_n_s_Z_s",
        "impurity_fraction_from_effective_charge",
        "normalized_gradient_scale_length",
        "electron_pressure",
        "ion_pressure",
    ],
)
def test_the_new_formulas_resolve_from_the_package(name):
    assert callable(getattr(vaft.formula, name))
    assert name in vaft.formula.__all__
