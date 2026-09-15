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

from vaft.formula.constants import K_BOLTZMANN, PA_PER_TORR
from vaft.formula.startup import (
    atomic_inventory_from_molecular_gas,
    breakdown_margin,
    lloyd_breakdown_field,
    neutral_density_from_pressure,
    townsend_ionization_coefficient,
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
