"""Recovering the resonant response from a stored spectral field.

The point of these functions is that a run's ``Phi_res`` need not be stored
to be known: it follows from the perturbed field on the flux-coordinate grid
plus equilibrium geometry. The fixtures here are analytic, because a
synthetic field with a jump you chose is the only one that can tell a
correct extraction from one that merely looks plausible; the agreement with
GPEC's own numbers is reported in the pull request and re-measured against
the restricted reference rather than committed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.stability import (
    island_width_from_resonant_flux,
    resonant_flux_from_delta,
)
from vaft.process.perturbation import (
    SINGULAR_OFFSET,
    ResonantJump,
    resonant_delta,
    resonant_geometric_factor,
)


def kinked_field(psi, psi_res, *, left_slope, right_slope, value=0.0):
    """A field with a chosen derivative discontinuity at ``psi_res``."""
    d = np.asarray(psi, dtype=float) - psi_res
    return value + np.where(d < 0, left_slope * d, right_slope * d).astype(complex)


#: A real GPEC run clusters its radial grid around the singular surfaces --
#: on the DIII-D reference the spacing there is 2.7e-8 to 6.4e-5 against a
#: jump evaluated 1.6e-4 away. A uniform fixture has to be fine enough to
#: put at least four points in the fitting band, which at n = 3 and steep
#: shear is only 2e-4 wide.
GRID = np.linspace(0.2, 0.9, 40001)
PSI_RES = 0.55


def extract(field, **kwargs):
    defaults = dict(
        psi_rational=PSI_RES, area=1.0, dq_dpsi_norm=1.0, chi1=1.0 / (2 * np.pi),
        m_pol=2, n_tor=1,
    )
    defaults.update(kwargs)
    return resonant_delta(GRID, field, **defaults)


# --------------------------------------------------------------------------
# The jump
# --------------------------------------------------------------------------


def test_the_jump_in_the_derivative_is_what_comes_back():
    """chi1 = 1/2pi and area = 1 make delta the bare slope difference."""
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    got = extract(field)
    assert got.delta.real == pytest.approx(3.0, rel=1e-6)
    assert got.delta.imag == pytest.approx(0.0, abs=1e-9)


def test_a_field_with_no_kink_has_no_resonance():
    field = kinked_field(GRID, PSI_RES, left_slope=2.5, right_slope=2.5)
    assert abs(extract(field).delta) < 1e-9


def test_the_screened_value_at_the_surface_does_not_matter():
    """The resonant harmonic is screened at its own surface in an ideal
    solution -- measured at q = 2 of the DIII-D reference, 2.4e-9 against
    1.2e-4, 3.7e-4 and 5.0e-4 for m = 1, 3 and 4 -- so an extraction that
    sampled the field there would recover nothing. Two fields with the same
    kink and different offsets give the same delta."""
    a = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0, value=0.0)
    b = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0, value=17.0)
    assert extract(a).delta == pytest.approx(extract(b).delta, rel=1e-9)


def test_a_complex_kink_keeps_its_phase():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0 + 0.0j, right_slope=1.0 + 3.0j)
    got = extract(field)
    assert got.delta == pytest.approx(3.0j, rel=1e-6)


def test_area_and_chi1_scale_the_result_as_the_formula_says():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    base = extract(field).delta
    assert extract(field, area=2.0).delta == pytest.approx(2.0 * base, rel=1e-9)
    assert extract(field, chi1=1.0 / np.pi).delta == pytest.approx(0.5 * base, rel=1e-9)


def test_the_evaluation_distance_follows_the_shear_and_is_reported():
    """GPEC places the points at sing_spot / (n |q'|), so a steeper q profile
    is sampled closer in. A caller comparing against a run's Delta needs to
    know where the jump was taken."""
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    assert extract(field).offset == pytest.approx(SINGULAR_OFFSET)
    assert extract(field, dq_dpsi_norm=4.0).offset == pytest.approx(SINGULAR_OFFSET / 4)
    assert extract(field, n_tor=2).offset == pytest.approx(SINGULAR_OFFSET / 2)
    assert extract(field, dq_dpsi_norm=-1.0).offset == pytest.approx(SINGULAR_OFFSET)


def test_the_mode_numbers_are_carried_not_re_derived():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    got = extract(field, m_pol=7, n_tor=3)
    assert isinstance(got, ResonantJump)
    assert (got.m_pol, got.n_tor, got.psi_norm) == (7, 3, PSI_RES)


def test_zero_shear_is_refused_rather_than_dividing_by_it():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    with pytest.raises(ValueError, match="no distance to evaluate at"):
        extract(field, dq_dpsi_norm=0.0)


def test_a_surface_outside_the_grid_is_refused():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    with pytest.raises(ValueError, match="outside the grid"):
        extract(field, psi_rational=0.95)


def test_a_grid_too_sparse_on_one_side_is_refused_not_extrapolated():
    """A cubic through fewer than four points is not a fit."""
    sparse = np.linspace(0.2, 0.9, 60)
    field = kinked_field(sparse, PSI_RES, left_slope=1.0, right_slope=4.0)
    with pytest.raises(ValueError, match="a one-sided cubic fit needs at least four"):
        resonant_delta(sparse, field, psi_rational=PSI_RES, area=1.0,
                       dq_dpsi_norm=1.0, chi1=1.0, m_pol=2, n_tor=1)


def test_mismatched_or_non_finite_inputs_are_refused():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    with pytest.raises(ValueError, match="grid points against"):
        resonant_delta(GRID, field[:-1], psi_rational=PSI_RES, area=1.0,
                       dq_dpsi_norm=1.0, chi1=1.0, m_pol=2, n_tor=1)
    broken = field.copy()
    broken[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract(broken)


def test_a_non_positive_mode_number_is_refused():
    field = kinked_field(GRID, PSI_RES, left_slope=1.0, right_slope=4.0)
    with pytest.raises(ValueError, match="n_tor must be positive"):
        extract(field, n_tor=0)


# --------------------------------------------------------------------------
# The geometric factor
# --------------------------------------------------------------------------


def test_the_factor_inverts_the_flux_it_was_measured_from():
    delta = np.array([1.0 + 2.0j, -3.0 + 1.0j])
    factor = np.array([0.05, 0.02])
    n = 2
    phi = resonant_flux_from_delta(delta, factor, n)
    np.testing.assert_allclose(resonant_geometric_factor(delta, phi, n), factor)


def test_the_factor_is_real_and_the_sign_is_gpecs():
    """Measured, the phase of n * Phi_res / Delta is exactly 180 degrees, so
    the factor is real and positive and the flux opposes the jump."""
    delta = np.array([2.0 + 0j])
    phi = resonant_flux_from_delta(delta, np.array([0.05]), 1)
    assert phi[0].real < 0
    assert resonant_geometric_factor(delta, phi, 1)[0] > 0


def test_a_pair_whose_ratio_is_not_real_is_refused():
    """It means the delta and the flux did not come from one run, and the
    real part alone would be a plausible-looking wrong number."""
    with pytest.raises(ValueError, match="off the real axis"):
        resonant_geometric_factor(np.array([1.0 + 0j]), np.array([-0.05 - 0.05j]), 1)


def test_a_zero_delta_has_no_measurable_factor():
    with pytest.raises(ValueError, match="delta is zero"):
        resonant_geometric_factor(np.array([0.0 + 0j]), np.array([1.0 + 0j]), 1)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="values of delta against"):
        resonant_geometric_factor(np.ones(3, dtype=complex), np.ones(2, dtype=complex), 1)


def test_the_flux_falls_as_one_over_n():
    """Measured across n = 1, 2 and 3 of the DIII-D reference, n * Phi / Delta
    agrees at every shared surface to 0.6 per cent -- so the factor belongs
    to the equilibrium and the mode number enters only here."""
    delta, factor = np.array([1.0 + 0j]), np.array([0.05])
    one = resonant_flux_from_delta(delta, factor, 1)
    assert resonant_flux_from_delta(delta, factor, 3) == pytest.approx(one / 3)


# --------------------------------------------------------------------------
# The island width
# --------------------------------------------------------------------------


def test_the_island_width_matches_the_closed_form():
    phi, area, q, dq, m, chi1 = 5.0e-4, 50.0, 2.0, 3.0, 2, 1.65
    shear = m * dq / q**2
    expected = 2 * np.sqrt(abs(4 * phi * area / (2 * np.pi * shear * q * chi1)))
    got = island_width_from_resonant_flux(phi, area, q, dq, m, chi1)
    assert got == pytest.approx(expected)


def test_the_width_grows_as_the_square_root_of_the_flux():
    args = (50.0, 2.0, 3.0, 2, 1.65)
    one = island_width_from_resonant_flux(1.0e-4, *args)
    four = island_width_from_resonant_flux(4.0e-4, *args)
    assert four == pytest.approx(2.0 * one)


def test_a_complex_flux_contributes_its_magnitude():
    """The width has no phase; taking the magnitude before the root is what
    keeps a complex flux from returning nan."""
    args = (50.0, 2.0, 3.0, 2, 1.65)
    got = island_width_from_resonant_flux(3.0e-4 + 4.0e-4j, *args)
    assert np.isfinite(got)
    assert got == pytest.approx(island_width_from_resonant_flux(5.0e-4, *args))


def test_the_width_is_elementwise_over_surfaces():
    phi = np.array([1.0e-4, 4.0e-4])
    got = island_width_from_resonant_flux(phi, np.array([50.0, 50.0]), np.array([2.0, 2.0]),
                                          np.array([3.0, 3.0]), np.array([2, 2]), 1.65)
    assert got.shape == (2,)
    assert got[1] == pytest.approx(2.0 * got[0])


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------


def test_a_field_and_a_geometry_give_an_island_width():
    """End to end: a stored harmonic with a known kink, through delta and the
    flux, to a width -- the path that lets a run's resonant table be
    reconstructed from the field alone."""
    field = kinked_field(GRID, PSI_RES, left_slope=0.0, right_slope=2.0e-3)
    jump = resonant_delta(GRID, field, psi_rational=PSI_RES, area=50.0,
                          dq_dpsi_norm=3.0, chi1=1.65, m_pol=2, n_tor=1)
    phi = resonant_flux_from_delta(jump.delta, 0.05, 1)
    width = island_width_from_resonant_flux(phi, 50.0, 2.0, 3.0, 2, 1.65)
    assert np.isfinite(width) and width > 0
    # Doubling the drive doubles the flux and so raises the width by sqrt(2).
    stronger = resonant_delta(GRID, 2 * field, psi_rational=PSI_RES, area=50.0,
                              dq_dpsi_norm=3.0, chi1=1.65, m_pol=2, n_tor=1)
    wider = island_width_from_resonant_flux(
        resonant_flux_from_delta(stronger.delta, 0.05, 1), 50.0, 2.0, 3.0, 2, 1.65)
    assert wider == pytest.approx(np.sqrt(2.0) * width, rel=1e-6)
