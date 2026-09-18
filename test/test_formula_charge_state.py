"""Mean charge and effective charge (#783 3.10).

Both are definitions, so what is pinned is that they agree with the two
functions they sit between: the coronal distribution
``fractional_abundances`` produces, and the two-species inverse GACODE's input
writer already uses to *choose* densities for a target ``Z_eff``.
"""

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.inputs import impurity_fractions
from vaft.data.open_adas import read_adf11
from vaft.formula.atomic import (
    fractional_abundances,
    mean_charge_from_charge_state_densities,
    z_eff_from_n_s_Z_s,
)


def _adf11_text(blocks):
    lines = ["2 2 2 / synthetic", "", "10.0 14.0", "0.0 2.0"]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(str(value) for value in values))
    return "\n".join(lines) + "\n"


def _tables(tmp_path: Path, log_ionisation, log_recombination=-8.0):
    """Synthetic two-rate-block tables with S/alpha = 10**(log_S - log_alpha)."""
    acd = tmp_path / "acd96_c.dat"
    scd = tmp_path / "scd96_c.dat"
    acd.write_text(_adf11_text([[log_recombination] * 4] * 2), encoding="ascii")
    scd.write_text(_adf11_text([[log_ionisation] * 4] * 2), encoding="ascii")
    read_adf11.cache_clear()
    return acd, scd


# ---------------------------------------------------------------------------
# Mean charge
# ---------------------------------------------------------------------------

def test_mean_charge_of_a_coronal_distribution_is_its_analytic_value(tmp_path):
    # S/alpha = 10 for both steps makes the distribution proportional to
    # [1, 10, 100], whose mean charge is (0 + 10 + 200) / 111.  Passing
    # fractional_abundances' output straight in is the use the layout
    # convention exists for.
    acd, scd = _tables(tmp_path, log_ionisation=-7.0)
    fractions = fractional_abundances(1.0e19, 100.0, acd, scd)
    assert fractions == pytest.approx(np.array([1.0, 10.0, 100.0]) / 111.0, rel=1e-12)
    assert mean_charge_from_charge_state_densities(fractions) == pytest.approx(
        210.0 / 111.0, rel=1e-12, abs=0.0
    )


def test_mean_charge_of_an_even_coronal_distribution_is_the_middle_state(tmp_path):
    acd, scd = _tables(tmp_path, log_ionisation=-8.0)
    fractions = fractional_abundances(1.0e19, 100.0, acd, scd)
    assert mean_charge_from_charge_state_densities(fractions) == pytest.approx(
        1.0, rel=1e-12, abs=0.0
    )


def test_mean_charge_limits():
    assert mean_charge_from_charge_state_densities([1.0, 0.0, 0.0]) == 0.0
    assert mean_charge_from_charge_state_densities([0.0] * 6 + [1.0]) == 6.0


def test_mean_charge_does_not_need_normalised_input():
    fractions = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
    assert mean_charge_from_charge_state_densities(3.7e17 * fractions) == pytest.approx(
        mean_charge_from_charge_state_densities(fractions), rel=1e-14, abs=0.0
    )


def test_mean_charge_respects_the_axis():
    fractions = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
    stacked = np.stack([fractions, fractions[::-1]])
    expected = [
        mean_charge_from_charge_state_densities(fractions),
        mean_charge_from_charge_state_densities(fractions[::-1]),
    ]
    assert mean_charge_from_charge_state_densities(stacked) == pytest.approx(expected)
    assert mean_charge_from_charge_state_densities(stacked.T, axis=0) == pytest.approx(expected)


@pytest.mark.parametrize("n_z", [[0.0, 0.0, 0.0], [1.0, -1.0], [np.nan, 1.0], []])
def test_mean_charge_refuses_a_distribution_it_cannot_average(n_z):
    with pytest.raises(ValueError):
        mean_charge_from_charge_state_densities(n_z)


# ---------------------------------------------------------------------------
# Effective charge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z_impurity, target", [(6, 2.0), (6, 1.3), (8, 3.0)])
def test_z_eff_inverts_the_gacode_impurity_construction(z_impurity, target):
    # GACODE's input writer picks the densities that make Z_eff hit a target;
    # this must read the same Z_eff back, with or without an explicit n_e.
    n_main, n_impurity = impurity_fractions(target, z_impurity)
    densities = [n_main, n_impurity]
    charges = [1, z_impurity]
    assert z_eff_from_n_s_Z_s(densities, charges) == pytest.approx(target, rel=1e-12)
    assert z_eff_from_n_s_Z_s(densities, charges, n_e=1.0) == pytest.approx(target, rel=1e-12)


def test_a_pure_hydrogenic_plasma_has_unit_z_eff():
    assert z_eff_from_n_s_Z_s([1.0e19], [1]) == 1.0


def test_collapsing_an_impurity_onto_its_mean_charge_loses_its_variance():
    # Z_eff weights Z^2, so replacing a charge-resolved impurity by one species
    # at its mean charge underestimates Z_eff by exactly n_I Var(Z) / n_e; the
    # quasi-neutral n_e is unchanged because it weights Z only linearly.
    fractions = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.25, 0.1])
    charges = np.arange(fractions.size)
    n_impurity, n_hydrogen = 1.0e17, 1.0e19
    mean = mean_charge_from_charge_state_densities(fractions)
    variance = float(np.sum(fractions * charges**2)) - mean**2
    resolved = z_eff_from_n_s_Z_s(
        np.r_[n_hydrogen, n_impurity * fractions], np.r_[1.0, charges]
    )
    collapsed = z_eff_from_n_s_Z_s([n_hydrogen, n_impurity], [1.0, mean])
    n_e = n_hydrogen + n_impurity * mean
    assert resolved - collapsed == pytest.approx(n_impurity * variance / n_e, rel=1e-10)


def test_a_measured_n_e_is_taken_as_given():
    # Not reconciled with quasi-neutrality: an n_e twice the quasi-neutral one
    # halves Z_eff, which is how a density error shows up.
    densities, charges = [1.0e19, 1.0e17], [1, 6]
    quasi_neutral = z_eff_from_n_s_Z_s(densities, charges)
    n_e = 1.0e19 + 6.0e17
    assert z_eff_from_n_s_Z_s(densities, charges, n_e=2.0 * n_e) == pytest.approx(
        0.5 * quasi_neutral, rel=1e-14, abs=0.0
    )


def test_z_eff_broadcasts_over_leading_axes():
    densities = np.array([[1.0e19, 1.0e17], [1.0e19, 2.0e17]])
    out = z_eff_from_n_s_Z_s(densities, [1, 6])
    assert out.shape == (2,)
    assert out[1] > out[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_s": [1.0, -1.0], "Z_s": [1, 1]},
        {"n_s": [1.0], "Z_s": [-1]},
        {"n_s": [1.0], "Z_s": [1], "n_e": 0.0},
        {"n_s": [1.0], "Z_s": [0]},
    ],
    ids=["negative_density", "negative_charge", "zero_n_e", "no_charged_species"],
)
def test_z_eff_refuses_a_non_physical_input(kwargs):
    with pytest.raises(ValueError):
        z_eff_from_n_s_Z_s(**kwargs)
