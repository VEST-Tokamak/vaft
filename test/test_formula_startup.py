"""The breakdown chain: prefill, Townsend avalanche, Lloyd threshold (#783).

The contract half of the coverage -- summary, units, ordered parameters,
references, policy-list membership -- is already gated by
``test_formula_docstrings.py`` and ``test_formula_catalog.py``. What is left,
and what this file pins, is the physics: that the two kernels are the same
model seen twice, that the published torr coefficients survive a pascal API
exactly, and that the domain edge blanks rather than invents a number.
"""

import warnings

import numpy as np
import pytest

from vaft.formula.constants import K_BOLTZMANN, MU0, PA_PER_TORR
from vaft.formula.startup import (
    atomic_inventory_from_molecular_gas,
    breakdown_margin,
    ejiri_f3_from_alpha,
    ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max,
    flux_closure_margin_from_E_t_a_B_stray_eta,
    lloyd_breakdown_field,
    neutral_density_from_pressure,
    plasma_self_field_from_I_p_a,
    startup_geometry_from_limiter_radii,
    townsend_breakdown_field,
    townsend_ionization_coefficient,
    vertical_field_from_I_p_R0_a_beta_p_li,
)
from vaft.formula.startup import _HIRSHMAN_A as HIRSHMAN_A
from vaft.formula.startup import (
    d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li,
    plasma_external_inductance_hirshman_from_R_eps_kappa,
    plasma_inductance_hirshman_from_R_eps_kappa_li,
)

#: Lloyd's published hydrogen coefficients, per torr.
A_TORR = 510.0
B_TORR = 1.25e4
#: The same two in SI pressure, for the Townsend form.
A_PA = A_TORR / PA_PER_TORR
B_PA = B_TORR / PA_PER_TORR

#: Shot 39915: the prefill its barometry recorded, and the drive its coils made
#: at R = 0.4 m. Tutorial session 02 reads both from the packaged ODS.
VEST_PREFILL_PA = 2.658e-3
VEST_DRIVE_V_PER_M = 1.906


# ---------------------------------------------------------------------------
# Pascal in, torr coefficients pinned
# ---------------------------------------------------------------------------

def test_the_threshold_is_the_published_torr_expression_evaluated_in_pascal():
    """Exact, not approximate: the coefficients are the published ones, so the
    only arithmetic between the two forms is the definition of a torr."""
    pressure = np.array([2.658e-3, 1e-2, 5e-2])[:, None]
    length = np.array([400.0, 1000.0, 4000.0])[None, :]

    p_torr = pressure / PA_PER_TORR
    assert (A_TORR * p_torr * length > 1.0).all(), "grid must stay in the domain"
    expected = B_TORR * p_torr / np.log(A_TORR * p_torr * length)

    np.testing.assert_allclose(
        lloyd_breakdown_field(pressure, length), expected, rtol=1e-12
    )


def test_the_rounded_si_coefficients_disagree_where_it_matters():
    """Why the torr values are pinned rather than the rounded SI ones issue
    #783 quotes: four significant figures is not enough near the domain edge,
    which is exactly where a spherical tokamak's prefill sits."""
    rounded = lambda p, L: 93.76 * p / np.log(3.825 * p * L)  # noqa: E731

    far = lloyd_breakdown_field(1e-2, 400.0)
    assert far == pytest.approx(rounded(1e-2, 400.0), rel=1e-3)

    near_edge = lloyd_breakdown_field(VEST_PREFILL_PA, 100.0)
    assert not np.isclose(near_edge, rounded(VEST_PREFILL_PA, 100.0), rtol=1e-3)


def test_the_torr_is_a_definition_not_a_measurement():
    assert PA_PER_TORR == 101325.0 / 760.0
    assert PA_PER_TORR == pytest.approx(133.322368, rel=1e-8)


def test_the_plot_layer_shares_this_definition_rather_than_copying_it():
    """``vaft/plot/display.py`` and ``vaft/plot/time.py`` each carried their own
    rounded copy, and the two disagreed (133.322368 against a bare 133.322).
    Both now import this one, so the axis label and the physics cannot drift."""
    from vaft.plot import display, time

    assert display.PA_PER_TORR is PA_PER_TORR
    assert time.PA_PER_TORR is PA_PER_TORR


# ---------------------------------------------------------------------------
# The two kernels are one model
# ---------------------------------------------------------------------------

def test_the_threshold_is_the_field_at_which_one_avalanche_length_fits():
    """Lloyd's constants *are* the Townsend A and B for hydrogen: the threshold
    is where alpha L = 1. This ties the two functions together and pins the
    pascal-torr conversion a second, independent time."""
    for pressure in (VEST_PREFILL_PA, 1e-2, 5e-2):
        for length in (400.0, 800.0, 2000.0):
            field = lloyd_breakdown_field(pressure, length)
            alpha = townsend_ionization_coefficient(field, pressure, A_PA, B_PA)
            assert alpha * length == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# The domain edge
# ---------------------------------------------------------------------------

def test_no_threshold_exists_below_the_avalanche_closure():
    """Below A p L = 1 the avalanche cannot close at any field, so the honest
    answer is that no threshold exists -- not a large one."""
    edge = PA_PER_TORR / (A_TORR * VEST_PREFILL_PA)
    assert edge == pytest.approx(98.35, abs=0.02)

    with pytest.warns(RuntimeWarning, match="returning nan"):
        assert np.isnan(lloyd_breakdown_field(VEST_PREFILL_PA, edge * 0.9))


def test_the_guard_is_the_domain_edge_and_nothing_wider():
    """nan marks exactly the closed form's domain, elementwise: a point blanks
    when A p L <= 1 and survives otherwise."""
    length = np.linspace(50.0, 300.0, 501)
    argument = A_TORR * (VEST_PREFILL_PA / PA_PER_TORR) * length
    with pytest.warns(RuntimeWarning):
        field = lloyd_breakdown_field(VEST_PREFILL_PA, length)
    np.testing.assert_array_equal(np.isnan(field), argument <= 1.0)


def test_the_pole_is_never_returned_as_an_infinite_field():
    """ln(A p L) -> 0 at the edge, so the closed form has a pole there. The
    guard owns the pole itself; a hair outside it the threshold is merely very
    large, which is the fit diverging and not a number to quote."""
    with pytest.warns(RuntimeWarning):
        field = lloyd_breakdown_field(VEST_PREFILL_PA, np.linspace(90.0, 110.0, 4001))
    assert not np.isinf(field).any()


def test_a_sweep_across_the_edge_blanks_only_the_invalid_entries():
    lengths = np.array([50.0, 90.0, 120.0, 300.0, 1000.0])
    with pytest.warns(RuntimeWarning):
        field = lloyd_breakdown_field(VEST_PREFILL_PA, lengths)
    assert np.isnan(field[:2]).all()
    assert np.isfinite(field[2:]).all()
    assert (field[2:] > 0.0).all()


def test_just_above_the_edge_the_threshold_is_large_and_falling():
    """Proves the nan region is a domain edge and not a masked sign flip."""
    edge = PA_PER_TORR / (A_TORR * VEST_PREFILL_PA)
    lengths = edge * np.array([1.02, 1.2, 2.0, 5.0])
    field = lloyd_breakdown_field(VEST_PREFILL_PA, lengths)
    assert field[0] > 10.0
    assert np.all(np.diff(field) < 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_Pa": 0.0, "connection_length_m": 200.0},
        {"p_Pa": -1e-3, "connection_length_m": 200.0},
        {"p_Pa": np.inf, "connection_length_m": 200.0},
        {"p_Pa": np.nan, "connection_length_m": 200.0},
        {"p_Pa": 1e-3, "connection_length_m": 0.0},
        {"p_Pa": 1e-3, "connection_length_m": -5.0},
    ],
)
def test_an_unphysical_input_raises_rather_than_blanking(kwargs):
    """A negative pressure is a caller mistake, not a regime: the traceback is
    more useful than a nan."""
    with pytest.raises(ValueError):
        lloyd_breakdown_field(**kwargs)


def test_a_blank_connection_length_propagates_rather_than_raising():
    """A connection-length map is `nan` outside the wall, so feeding it to a
    threshold is the ordinary thing to do. Only the length is tolerant: a `nan`
    pressure is still refused, above, because nothing hands one over deliberately."""
    length = np.array([np.nan, 200.0, np.nan])
    field = lloyd_breakdown_field(VEST_PREFILL_PA, length)
    assert np.isnan(field[0]) and np.isnan(field[2])
    assert np.isfinite(field[1])
    assert np.isnan(breakdown_margin(VEST_DRIVE_V_PER_M, field)[0])


def test_a_healthy_call_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(lloyd_breakdown_field(VEST_PREFILL_PA, 400.0))


# ---------------------------------------------------------------------------
# Limiting trends
# ---------------------------------------------------------------------------

def test_a_longer_connection_length_lowers_the_threshold():
    """Toy model 1's whole message: a better null buys connection length, and
    connection length buys a lower threshold."""
    lengths = np.array([120.0, 200.0, 400.0, 1000.0, 5000.0])
    field = lloyd_breakdown_field(VEST_PREFILL_PA, lengths)
    assert np.all(np.diff(field) < 0.0)


def test_the_threshold_has_a_paschen_minimum_at_apl_equals_e():
    """dE/dp vanishes at ln(A p L) = 1, so the curve turns there -- the
    tokamak's Paschen minimum, with the electrode gap replaced by L."""
    length = 400.0
    p_min = np.e * PA_PER_TORR / (A_TORR * length)
    below = lloyd_breakdown_field(p_min * 0.5, length)
    at = lloyd_breakdown_field(p_min, length)
    above = lloyd_breakdown_field(p_min * 2.0, length)
    assert at < below and at < above


# ---------------------------------------------------------------------------
# The margin
# ---------------------------------------------------------------------------

def test_the_margin_is_linear_in_the_drive_and_unity_at_threshold():
    threshold = lloyd_breakdown_field(VEST_PREFILL_PA, 400.0)
    assert breakdown_margin(threshold, threshold) == pytest.approx(1.0)
    assert breakdown_margin(2.0 * threshold, threshold) == pytest.approx(
        2.0 * breakdown_margin(threshold, threshold)
    )


def test_a_flux_convention_cannot_flip_the_margin():
    """The two flux-to-voltage kernels disagree on sign (#354). An avalanche
    does not care, and neither does this."""
    threshold = lloyd_breakdown_field(VEST_PREFILL_PA, 400.0)
    assert breakdown_margin(-VEST_DRIVE_V_PER_M, threshold) == pytest.approx(
        breakdown_margin(VEST_DRIVE_V_PER_M, threshold)
    )


def test_a_blank_threshold_propagates_into_the_margin():
    assert np.isnan(breakdown_margin(VEST_DRIVE_V_PER_M, np.nan))


# ---------------------------------------------------------------------------
# The VEST anchor: what session 02 will print
# ---------------------------------------------------------------------------

def test_the_measured_drive_needs_a_hundred_metres_of_connection_length():
    """At this prefill the breakdown question is entirely the connection
    length -- the one quantity VAFT cannot compute. Session 02 says so, and
    these are the numbers it says it with."""
    edge = PA_PER_TORR / (A_TORR * VEST_PREFILL_PA)
    assert edge == pytest.approx(98.35, abs=0.02)

    below = breakdown_margin(
        VEST_DRIVE_V_PER_M, lloyd_breakdown_field(VEST_PREFILL_PA, 100.0)
    )
    above = breakdown_margin(
        VEST_DRIVE_V_PER_M, lloyd_breakdown_field(VEST_PREFILL_PA, 120.0)
    )
    assert below < 1.0 < above

    crossing = np.array(
        [
            length
            for length in np.linspace(edge * 1.001, 200.0, 20001)
            if breakdown_margin(
                VEST_DRIVE_V_PER_M, lloyd_breakdown_field(VEST_PREFILL_PA, length)
            )
            >= 1.0
        ]
    ).min()
    assert crossing == pytest.approx(112.09, abs=0.05)


# ---------------------------------------------------------------------------
# Prefill inventory
# ---------------------------------------------------------------------------

def test_the_molecular_density_is_the_ideal_gas_law():
    assert neutral_density_from_pressure(VEST_PREFILL_PA) == pytest.approx(
        VEST_PREFILL_PA / (K_BOLTZMANN * 300.0), rel=1e-12
    )
    with pytest.raises(ValueError):
        neutral_density_from_pressure(0.0)
    with pytest.raises(ValueError):
        neutral_density_from_pressure(1e-3, T_gas_K=-5.0)


def test_the_atom_inventory_is_an_argument_not_a_species_guess():
    molecular = neutral_density_from_pressure(VEST_PREFILL_PA)
    assert atomic_inventory_from_molecular_gas(molecular) == pytest.approx(
        2.0 * VEST_PREFILL_PA / (K_BOLTZMANN * 300.0), rel=1e-12
    )
    # A monatomic fill is the same number: the factor is the caller's.
    assert atomic_inventory_from_molecular_gas(
        molecular, atoms_per_molecule=1
    ) == pytest.approx(molecular)


# ---------------------------------------------------------------------------
# Scalar in, scalar out
# ---------------------------------------------------------------------------

def test_a_scalar_call_returns_a_float_and_an_array_call_broadcasts():
    assert isinstance(lloyd_breakdown_field(1e-2, 400.0), float)
    assert isinstance(neutral_density_from_pressure(1e-2), float)
    assert isinstance(breakdown_margin(1.0, 2.0), float)
    assert isinstance(
        townsend_ionization_coefficient(2.0, 1e-2, A_PA, B_PA), float
    )

    field = lloyd_breakdown_field(np.array([1e-2, 2e-2])[:, None],
                                  np.array([400.0, 800.0, 1600.0])[None, :])
    assert isinstance(field, np.ndarray) and field.shape == (2, 3)


# ==========================================================================
# After the avalanche: a current channel, and whether it can close (#783)
# ==========================================================================

def test_the_generic_inversion_reproduces_the_lloyd_wrapper_exactly():
    # lloyd_breakdown_field is now this function with the hydrogen
    # coefficients and a pascal-to-torr conversion in front.  Equality, not
    # approximate equality: the wrapper must not re-derive the arithmetic.
    for p_torr, length in [(1e-4, 200.0), (1e-3, 100.0), (5e-3, 40.0)]:
        assert lloyd_breakdown_field(p_torr * PA_PER_TORR, length) == (
            townsend_breakdown_field(p_torr, length, A_TORR, B_TORR)
        )


def test_the_generic_inversion_is_the_field_where_one_avalanche_length_fits():
    # E_BD is defined by alpha(E_BD) * L == 1; that identity is the whole
    # content of the inversion, so pin it rather than the algebra.
    p_pa, length = 1.3e-1, 100.0
    field = townsend_breakdown_field(p_pa, length, A_PA, B_PA)
    assert townsend_ionization_coefficient(field, p_pa, A_PA, B_PA) == (
        pytest.approx(1.0 / length, rel=1e-12, abs=0.0)
    )


def test_the_generic_inversion_carries_the_pressure_unit_in_its_coefficients():
    # The function does not know the unit: the same physical point expressed
    # in torr and in pascal must agree once A and B are converted with it.
    p_torr, length = 1e-3, 100.0
    assert townsend_breakdown_field(p_torr, length, A_TORR, B_TORR) == (
        pytest.approx(
            townsend_breakdown_field(p_torr * PA_PER_TORR, length, A_PA, B_PA),
            rel=1e-12,
            abs=0.0,
        )
    )


def test_the_generic_inversion_blanks_below_its_domain():
    with pytest.warns(RuntimeWarning, match="avalanche cannot close"):
        assert np.isnan(townsend_breakdown_field(1e-9, 1.0, A_PA, B_PA))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p": 0.0}, {"p": -1.0}, {"connection_length_m": 0.0},
        {"A": 0.0}, {"B": -1.0},
    ],
)
def test_the_generic_inversion_rejects_a_non_physical_input(kwargs):
    call = {"p": 1e-1, "connection_length_m": 100.0, "A": A_PA, "B": B_PA}
    call.update(kwargs)
    with pytest.raises(ValueError):
        townsend_breakdown_field(**call)


def test_limiter_aperture_geometry_is_the_midplane_chord():
    R0, a = startup_geometry_from_limiter_radii(0.15, 0.85)
    assert R0 == pytest.approx(0.5, rel=1e-13, abs=0.0)
    assert a == pytest.approx(0.35, rel=1e-13, abs=0.0)
    # The two are the sum and difference halves, so they reconstruct the input.
    assert R0 - a == pytest.approx(0.15, rel=1e-13, abs=0.0)
    assert R0 + a == pytest.approx(0.85, rel=1e-13, abs=0.0)


def test_limiter_aperture_geometry_rejects_an_inverted_aperture():
    with pytest.raises(ValueError, match="R_outboard_m must exceed"):
        startup_geometry_from_limiter_radii(0.85, 0.15)
    with pytest.raises(ValueError, match="R_outboard_m must exceed"):
        startup_geometry_from_limiter_radii(0.5, 0.5)


def test_self_field_is_amperes_law_around_the_channel():
    assert plasma_self_field_from_I_p_a(1.0e4, 0.35) == pytest.approx(
        MU0 * 1.0e4 / (2.0 * np.pi * 0.35), rel=1e-13, abs=0.0
    )
    # Linear in current, inverse in radius -- the two scalings that make the
    # flux-closure comparison behave the way the criterion assumes.
    base = plasma_self_field_from_I_p_a(1.0e4, 0.35)
    assert plasma_self_field_from_I_p_a(2.0e4, 0.35) == pytest.approx(2.0 * base)
    assert plasma_self_field_from_I_p_a(1.0e4, 0.70) == pytest.approx(0.5 * base)


def test_vertical_field_matches_the_shafranov_bracket():
    I_p, R0, a, beta_p, li = 1.0e5, 1.8, 0.4, 0.3, 0.9
    bracket = np.log(8.0 * R0 / a) + beta_p + 0.5 * li - 1.5
    assert vertical_field_from_I_p_R0_a_beta_p_li(
        I_p, R0, a, beta_p, li
    ) == pytest.approx(MU0 * I_p * bracket / (4.0 * np.pi * R0), rel=1e-13, abs=0.0)


def test_vertical_field_moves_the_documented_way_with_beta_p_and_li():
    args = (1.0e5, 1.8, 0.4)
    base = vertical_field_from_I_p_R0_a_beta_p_li(*args, 0.3, 0.9)
    # Both enter the bracket with a positive coefficient, li at half weight.
    assert vertical_field_from_I_p_R0_a_beta_p_li(*args, 1.3, 0.9) > base
    assert vertical_field_from_I_p_R0_a_beta_p_li(*args, 0.3, 1.9) > base
    delta_beta = vertical_field_from_I_p_R0_a_beta_p_li(*args, 1.3, 0.9) - base
    delta_li = vertical_field_from_I_p_R0_a_beta_p_li(*args, 0.3, 1.9) - base
    assert delta_li == pytest.approx(0.5 * delta_beta, rel=1e-12, abs=0.0)


def test_vertical_field_rejects_a_minor_radius_that_is_not_smaller():
    with pytest.raises(ValueError, match="a_m must be smaller"):
        vertical_field_from_I_p_R0_a_beta_p_li(1.0e5, 0.4, 0.4, 0.3, 0.9)


def test_flux_closure_margin_is_the_self_field_over_the_stray_field():
    # The margin is defined as B_p,self / B_stray after substituting the
    # reduced Ohmic current, so composing the two kernels must reproduce it.
    E_t, a, B_stray, eta = 5.0, 0.35, 1.0e-3, 1.0e-5
    I_p = np.pi * a**2 * E_t / eta
    assert flux_closure_margin_from_E_t_a_B_stray_eta(
        E_t, a, B_stray, eta
    ) == pytest.approx(
        plasma_self_field_from_I_p_a(I_p, a) / B_stray, rel=1e-12, abs=0.0
    )


def test_flux_closure_margin_crosses_one_at_the_documented_threshold():
    # M_FC > 1 is the same statement as E_t a / (B_stray eta) > 2/mu0.
    E_t, a, eta = 5.0, 0.35, 1.0e-5
    B_at_unity = 0.5 * MU0 * E_t * a / eta
    assert flux_closure_margin_from_E_t_a_B_stray_eta(
        E_t, a, B_at_unity, eta
    ) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert flux_closure_margin_from_E_t_a_B_stray_eta(
        E_t, a, 2.0 * B_at_unity, eta
    ) < 1.0


def test_flux_closure_margin_matches_the_legacy_gauss_constant():
    # The gauss form quotes 1.6e2 V/(G Ohm m); that is 2/mu0 = 1.59e6 with the
    # 1e-4 of the unit change absorbed.  Pin the number a reader will meet.
    assert 2.0 / MU0 * 1e-4 == pytest.approx(1.59e2, rel=1e-2)


# ==========================================================================
# Before the avalanche: the Ejiri mirror proxy (#676)
# ==========================================================================

def test_ejiri_alpha_takes_the_stricter_of_its_two_loss_channels():
    # Geometry chosen so the inboard branch dominates, then so the curvature
    # branch does; alpha must follow whichever is larger, not average them.
    # Both R_C values keep the curvature argument positive, so this exercises
    # the max() and not the clip: at R_C = 0.2 the curvature branch is 0.500
    # against an inboard 0.655, and at R_C = 0.6 it is 1.658.
    R_S, R_LIN = 0.5, 0.15
    inboard = np.sqrt(R_LIN / (R_S - R_LIN))
    for R_C, Z_max in [(0.2, 0.4), (0.6, 0.4)]:
        curvature = np.sqrt((2.0 * R_C * R_S - Z_max**2) / Z_max**2)
        assert curvature > 0.0
        assert ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(
            R_S, R_LIN, R_C, Z_max
        ) == pytest.approx(max(inboard, curvature), rel=1e-12, abs=0.0)
    # ... and the two cases really did land on opposite branches.
    assert np.sqrt((2 * 0.2 * R_S - 0.4**2) / 0.4**2) < inboard
    assert np.sqrt((2 * 0.6 * R_S - 0.4**2) / 0.4**2) > inboard


def test_ejiri_alpha_floors_the_curvature_branch_at_zero():
    # 2 R_C R_S < Z_max^2 would take the square root of a negative number; the
    # branch is clipped so the inboard channel still decides.
    R_S, R_LIN = 0.5, 0.15
    alpha = ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(R_S, R_LIN, 0.01, 2.0)
    assert np.isfinite(alpha)
    assert alpha == pytest.approx(
        np.sqrt(R_LIN / (R_S - R_LIN)), rel=1e-12, abs=0.0
    )


def test_ejiri_alpha_rejects_a_start_inside_the_inboard_limiter():
    with pytest.raises(ValueError, match="R_LIN_m must be smaller"):
        ejiri_mirror_alpha_from_R_S_R_LIN_R_C_Z_max(0.15, 0.5, 0.6, 0.4)


def test_ejiri_f3_matches_its_closed_form():
    for alpha in (0.0, 0.5, 1.0, 3.7):
        assert ejiri_f3_from_alpha(alpha) == pytest.approx(
            (2.0 + 3.0 * alpha**2) / (2.0 * (1.0 + alpha**2) ** 1.5),
            rel=1e-13,
            abs=0.0,
        )


def test_ejiri_f3_confines_everything_at_zero_slope_and_decays_monotonically():
    assert ejiri_f3_from_alpha(0.0) == pytest.approx(1.0, rel=1e-13, abs=0.0)
    values = ejiri_f3_from_alpha(np.linspace(0.0, 20.0, 400))
    assert np.all(np.diff(values) < 0.0)


def test_ejiri_f3_approaches_its_large_slope_asymptote():
    # F3 -> 3/(2 alpha), which is what makes a larger alpha the worse
    # configuration rather than the better one.
    for alpha in (50.0, 500.0):
        assert ejiri_f3_from_alpha(alpha) == pytest.approx(
            3.0 / (2.0 * alpha), rel=1e-3
        )


def test_ejiri_f3_rejects_a_negative_slope():
    with pytest.raises(ValueError, match="alpha must be finite and non-negative"):
        ejiri_f3_from_alpha(-0.1)


# ==========================================================================
# The Hirshman inductance behind the low-aspect-ratio vertical field (#783)
# ==========================================================================

def test_the_vertical_field_reduces_to_the_circular_form_at_unit_elongation():
    # The elongation factor l_kappa = sqrt((1+kappa^2)/2) is 1 at kappa = 1, so
    # the Mitarai form and the circular Shafranov form must coincide exactly.
    I_p, R0, a, beta_p, li = 8.0e6, 6.2, 1.9, 2.5, 0.5
    circular = MU0 * I_p / (4.0 * np.pi * R0) * (
        np.log(8.0 * R0 / a) + beta_p + 0.5 * li - 1.5
    )
    assert vertical_field_from_I_p_R0_a_beta_p_li(I_p, R0, a, beta_p, li) == (
        pytest.approx(circular, rel=1e-13, abs=0.0)
    )
    assert vertical_field_from_I_p_R0_a_beta_p_li(
        I_p, R0, a, beta_p, li, kappa=1.0
    ) == pytest.approx(circular, rel=1e-13, abs=0.0)


def test_the_vertical_field_matches_mitarai_equation_1_3():
    # Paper parameters, ITER-FEAT row of its table 1.
    I_p, R0, a, beta_p, li, kappa = 8.0e6, 6.2, 1.9, 2.5, 0.5, 1.7
    l_kappa = np.sqrt(0.5 * (1.0 + kappa**2))
    expected = MU0 * I_p / (4.0 * np.pi * R0) * (
        np.log(8.0 * R0 / (a * l_kappa)) + beta_p + 0.5 * li - 1.5
    )
    assert vertical_field_from_I_p_R0_a_beta_p_li(
        I_p, R0, a, beta_p, li, kappa
    ) == pytest.approx(expected, rel=1e-13, abs=0.0)
    # Elongation shortens the effective minor radius, so it lowers the field.
    assert expected < vertical_field_from_I_p_R0_a_beta_p_li(
        I_p, R0, a, beta_p, li
    )


def test_the_hirshman_inductance_reproduces_the_mitarai_table():
    # Mitarai table 1 reports L_p = 9.18 uH for R = 6.2, a = 1.9, kappa = 1.7,
    # li = 0.5 from his own circular expression; the Hirshman fit is a
    # different formula, so this pins that the two agree to a few percent
    # rather than exactly -- which is the whole reason to have both.
    R0, a, kappa, li = 6.2, 1.9, 1.7, 0.5
    eps = a / R0
    l_kappa = np.sqrt(0.5 * (1.0 + kappa**2))
    mitarai = MU0 * R0 * (np.log(8.0 * R0 / (a * l_kappa)) + 0.5 * li - 2.0)
    assert mitarai == pytest.approx(9.18e-6, rel=5e-3)
    hirshman = plasma_inductance_hirshman_from_R_eps_kappa_li(R0, eps, kappa, li)
    assert hirshman == pytest.approx(mitarai, rel=0.15)


def test_the_hirshman_inductance_splits_into_external_and_internal():
    R0, eps, kappa, li = 1.0, 0.3, 1.6, 0.8
    external = plasma_external_inductance_hirshman_from_R_eps_kappa(R0, eps, kappa)
    total = plasma_inductance_hirshman_from_R_eps_kappa_li(R0, eps, kappa, li)
    assert total - external == pytest.approx(MU0 * R0 * 0.5 * li, rel=1e-12, abs=0.0)
    # The external part does not know li at all.
    assert plasma_inductance_hirshman_from_R_eps_kappa_li(
        R0, eps, kappa, 0.0
    ) == pytest.approx(external, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("eps", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("convention", ["fixed", "inboard", "outboard"])
def test_the_inductance_derivative_matches_a_finite_difference(eps, convention):
    # The analytic derivative is the claim; a finite difference of the
    # inductance itself is the independent check, and it has to be taken with
    # the minor radius held the way the convention says.
    R0, kappa, li = 1.0, 1.6, 0.8
    a0 = eps * R0
    step = 1e-7

    def inductance(R):
        if convention == "fixed":
            a = a0
        elif convention == "inboard":
            a = R - (R0 - a0)
        else:
            a = (R0 + a0) - R
        return plasma_inductance_hirshman_from_R_eps_kappa_li(R, a / R, kappa, li)

    finite = (inductance(R0 + step) - inductance(R0 - step)) / (2.0 * step)
    assert d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
        R0, eps, kappa, li, minor_radius=convention
    ) == pytest.approx(finite, rel=1e-6)


def test_the_minor_radius_convention_changes_the_sign_not_just_the_size():
    # This is the trap the docstring warns about: at eps = 0.3 the inboard
    # convention is negative while the other two are positive, so a radial
    # force balance solved with the wrong one pushes the plasma the other way.
    values = {
        convention: d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
            1.0, 0.3, 1.0, 0.0, minor_radius=convention
        )
        for convention in ("fixed", "inboard", "outboard")
    }
    assert values["inboard"] < 0.0 < values["fixed"] < values["outboard"]


def test_the_fixed_convention_is_the_one_that_reproduces_mitarai():
    # Substituting an inductance into B_VE = -(mu0 Ip/4 pi R)[dLp/dR / mu0
    # + beta_p - 1/2] with the minor radius *held* is what gives Mitarai's
    # Eq. (1.3); that is why 'fixed' is the default.
    I_p, R0, a, beta_p, li, kappa = 8.0e6, 6.2, 1.9, 2.5, 0.5, 1.7
    l_kappa = np.sqrt(0.5 * (1.0 + kappa**2))
    step = 1e-7

    def circular_inductance(R):
        return MU0 * R * (np.log(8.0 * R / (a * l_kappa)) + 0.5 * li - 2.0)

    dLp_dR = (circular_inductance(R0 + step) - circular_inductance(R0 - step)) / (
        2.0 * step
    )
    reconstructed = MU0 * I_p / (4.0 * np.pi * R0) * (
        dLp_dR / MU0 + beta_p - 0.5
    )
    assert reconstructed == pytest.approx(
        vertical_field_from_I_p_R0_a_beta_p_li(I_p, R0, a, beta_p, li, kappa),
        rel=1e-6,
    )


def test_the_closed_form_derivative_matches_a_finite_difference():
    # da/deps is differentiated by hand, so the finite difference is the only
    # independent check there is.  It also settles the one coefficient that is
    # easy to get wrong: a1 + a3/2 = 6.435, where 5.935 is off by 16 % at
    # eps = 0.1 and by nothing at all that inspection would catch.
    from vaft.formula.startup import _hirshman_a, _hirshman_da

    step = 1e-7
    for eps in (0.1, 0.3, 0.5, 0.7):
        finite = (_hirshman_a(eps + step) - _hirshman_a(eps - step)) / (2.0 * step)
        assert _hirshman_da(eps) == pytest.approx(finite, rel=1e-7)

    a1, _, a3, _ = HIRSHMAN_A
    assert a1 + 0.5 * a3 == pytest.approx(6.435, rel=1e-12, abs=0.0)


def test_the_derivative_constant_is_a2_minus_a4():
    # The other easy slip: a2 + a4 = 3.26 rather than 0.84, and the
    # finite-difference test above is what rejects it.
    _, a2, _, a4 = HIRSHMAN_A
    assert a2 - a4 == pytest.approx(0.84, rel=1e-12, abs=0.0)
    assert a2 + a4 != pytest.approx(0.84, rel=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [{"epsilon": 0.0}, {"epsilon": 1.0}, {"epsilon": 1.5}, {"kappa": 0.0},
     {"R_m": -1.0}],
)
def test_the_hirshman_helpers_reject_a_non_physical_geometry(kwargs):
    call = {"R_m": 1.0, "epsilon": 0.3, "kappa": 1.6}
    call.update(kwargs)
    with pytest.raises(ValueError):
        plasma_external_inductance_hirshman_from_R_eps_kappa(**call)


def test_an_unknown_minor_radius_convention_is_rejected():
    with pytest.raises(ValueError, match="unknown minor_radius"):
        d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
            1.0, 0.3, 1.6, 0.8, minor_radius="limiter"
        )


# ==========================================================================
# Cold review of #851
# ==========================================================================

def test_the_generic_inversion_accepts_array_coefficients():
    # The point of a generic kernel is sweeping a gas catalogue, so array A and
    # B must survive.  _maybe_scalar has to be told about them or it calls
    # float() on an ndarray and NumPy 2 raises.
    A = np.array([A_TORR, 600.0])
    field = townsend_breakdown_field(1e-3, 100.0, A, B_TORR)
    assert isinstance(field, np.ndarray)
    assert field.shape == (2,)
    assert field[0] == pytest.approx(
        townsend_breakdown_field(1e-3, 100.0, A_TORR, B_TORR), rel=1e-13, abs=0.0
    )
    # ... and so must an array B, on its own.
    assert townsend_breakdown_field(
        1e-3, 100.0, A_TORR, np.array([B_TORR, 2 * B_TORR])
    ) == pytest.approx(
        np.array([1.0, 2.0])
        * townsend_breakdown_field(1e-3, 100.0, A_TORR, B_TORR),
        rel=1e-13,
        abs=0.0,
    )


def test_the_degenerate_warning_blames_the_caller_not_the_module():
    # The Lloyd wrapper delegates now; without the stacklevel passed through,
    # every blanked point in a pressure sweep would be attributed to a line
    # inside startup.py and the user could not tell which call produced it.
    for call in (
        lambda: lloyd_breakdown_field(1e-9, 1.0),
        lambda: townsend_breakdown_field(1e-30, 1.0, A_PA, B_PA),
    ):
        with pytest.warns(RuntimeWarning, match="avalanche cannot close") as record:
            call()
        assert record[0].filename == __file__


def test_the_vertical_field_keeps_the_sign_of_its_bracket():
    # The bracket goes negative for a strongly elongated, low-beta, low-aspect
    # case: l_kappa shrinks the logarithm below 3/2.  That is a real reversal
    # of the required field, so it is returned signed rather than clipped --
    # and the Returns section says so.
    value = vertical_field_from_I_p_R0_a_beta_p_li(
        1.0e4, 0.5, 0.35, 0.0, 0.0, kappa=4.0
    )
    assert value < 0.0
    l_kappa = np.sqrt(0.5 * (1.0 + 4.0**2))
    bracket = np.log(8.0 * 0.5 / (0.35 * l_kappa)) - 1.5
    assert bracket < 0.0
    assert value == pytest.approx(
        MU0 * 1.0e4 * bracket / (4.0 * np.pi * 0.5), rel=1e-13, abs=0.0
    )


def test_the_inductance_derivative_does_not_depend_on_the_major_radius():
    # Expressed in epsilon the derivative is mu0 times a dimensionless
    # function, so R_m is carried for symmetry and scalar detection only.
    # Pinned so that a shaping term added later has to change this on purpose.
    for convention in ("fixed", "inboard", "outboard"):
        assert d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
            1.0, 0.3, 1.6, 0.8, minor_radius=convention
        ) == d_plasma_inductance_dR_hirshman_from_R_eps_kappa_li(
            99.0, 0.3, 1.6, 0.8, minor_radius=convention
        )


# ------------------------------------------------------------------
# Electron-cyclotron resonance: where a 2.45 GHz source seeds electrons
# ------------------------------------------------------------------

def test_a_245_ghz_source_resonates_at_875_millitesla():
    from vaft.formula.constants import ME, QE
    from vaft.formula.startup import electron_cyclotron_resonance_field

    field = electron_cyclotron_resonance_field(2.45e9)
    assert isinstance(field, float)
    assert field == pytest.approx(2.0 * np.pi * ME * 2.45e9 / QE, rel=1e-15)
    assert field == pytest.approx(0.0875234741, rel=1e-9)


def test_a_higher_harmonic_resonates_at_a_proportionally_weaker_field():
    from vaft.formula.startup import electron_cyclotron_resonance_field

    fields = electron_cyclotron_resonance_field(np.array([2.45e9, 4.9e9]))
    assert electron_cyclotron_resonance_field(2.45e9, harmonic=2) == pytest.approx(fields[0] / 2)
    np.testing.assert_allclose(fields[1], 2 * fields[0])


def test_the_resonance_radius_is_where_a_one_over_r_field_meets_the_resonant_field():
    from vaft.formula.startup import (
        electron_cyclotron_resonance_field,
        electron_cyclotron_resonance_radius,
    )

    b_t_r = np.array([0.0, 0.035, -0.061])
    radius = electron_cyclotron_resonance_radius(b_t_r, 2.45e9)
    b_ecr = electron_cyclotron_resonance_field(2.45e9)
    # the vacuum field evaluated at the returned radius is the resonant field
    np.testing.assert_allclose(np.abs(b_t_r[1:]) / radius[1:], b_ecr)
    assert radius[0] == 0.0
    assert radius[2] == pytest.approx(0.061 / 0.0875234741, rel=1e-8)


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan])
def test_an_unphysical_frequency_raises(frequency):
    from vaft.formula.startup import electron_cyclotron_resonance_field

    with pytest.raises(ValueError, match="frequency_Hz"):
        electron_cyclotron_resonance_field(frequency)


@pytest.mark.parametrize("harmonic", [0, -1, 1.5, True, np.nan])
def test_a_harmonic_must_be_a_positive_integer(harmonic):
    from vaft.formula.startup import electron_cyclotron_resonance_field

    with pytest.raises(ValueError, match="harmonic"):
        electron_cyclotron_resonance_field(2.45e9, harmonic=harmonic)


def test_a_missing_toroidal_field_is_not_silently_a_zero_radius():
    from vaft.formula.startup import electron_cyclotron_resonance_radius

    with pytest.raises(ValueError, match="B_T_R_Tm"):
        electron_cyclotron_resonance_radius(np.array([0.05, np.nan]), 2.45e9)
