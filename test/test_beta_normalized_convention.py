"""beta_N is quoted in %·m·T/MA, the only convention Troyon's 2.8 lives in (#349).

`beta_N_from_beta_a_B0_Ip` evaluated `beta * a * B0 / I_p` literally, with beta
documented as a fraction and I_p in amperes. That is 1e-8 times the conventional
number -- x100 to reach percent, x1e6 to reach megaamperes -- so the result could
not be compared with the Troyon limit, nor with `kink_stability_criterion` or
`beta_stability_boundary`, which are written against it.

The change is breaking and deliberately so: nothing can have depended on the old
value being right, because it was not a beta_N in any convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.formula.stability import beta_N_from_beta_a_B0_Ip

#: ITER 15 MA baseline: beta_t ~ 2.5 %, a = 2.0 m, B_0 = 5.3 T, I_p = 15 MA.
ITER_BASELINE = dict(beta_percent=2.5, a=2.0, B0=5.3, I_p_MA=15.0)


def test_the_iter_baseline_lands_where_the_literature_puts_it():
    """~1.8 is the familiar figure for this plasma, and it is below 2.8."""
    beta_n = beta_N_from_beta_a_B0_Ip(**ITER_BASELINE)
    assert beta_n == pytest.approx(2.5 * 2.0 * 5.3 / 15.0)
    assert 1.5 < beta_n < 2.0
    assert beta_n < 2.8, "the baseline should sit below the Troyon limit"


def test_the_troyon_limit_is_reachable_in_these_units():
    """A beta_N of 2.8 must correspond to a physical beta, not 2.8e8 of one."""
    limit = 2.8
    beta_at_limit = limit * ITER_BASELINE["I_p_MA"] / (ITER_BASELINE["a"] * ITER_BASELINE["B0"])
    assert 3.0 < beta_at_limit < 4.5, "beta at the Troyon limit is a few percent"
    assert beta_N_from_beta_a_B0_Ip(
        beta_percent=beta_at_limit, a=2.0, B0=5.3, I_p_MA=15.0
    ) == pytest.approx(limit)


def test_the_old_si_call_was_1e8_below_the_convention():
    """The negative control: what the previous signature returned.

    Kept because the failure was silent -- an SI caller got a number that looked
    like a beta_N and was eight orders of magnitude from one.
    """
    conventional = beta_N_from_beta_a_B0_Ip(**ITER_BASELINE)
    si_style = 0.025 * 2.0 * 5.3 / 15e6  # fraction and amperes, as it used to be
    assert conventional / si_style == pytest.approx(1e8)


def test_it_scales_the_way_its_definition_says():
    base = beta_N_from_beta_a_B0_Ip(**ITER_BASELINE)
    doubled_beta = beta_N_from_beta_a_B0_Ip(**{**ITER_BASELINE, "beta_percent": 5.0})
    doubled_current = beta_N_from_beta_a_B0_Ip(**{**ITER_BASELINE, "I_p_MA": 30.0})
    assert doubled_beta == pytest.approx(2.0 * base)
    assert doubled_current == pytest.approx(0.5 * base)


def test_the_parameter_names_carry_their_units():
    """A keyword caller written for the old signature must fail, not mislead."""
    with pytest.raises(TypeError):
        beta_N_from_beta_a_B0_Ip(beta=0.025, a=2.0, B0=5.3, I_p=15e6)


def test_it_is_vectorised_over_a_scan():
    beta = np.array([1.0, 2.5, 4.0])
    result = beta_N_from_beta_a_B0_Ip(beta, 2.0, 5.3, 15.0)
    np.testing.assert_allclose(result, beta * 2.0 * 5.3 / 15.0)
