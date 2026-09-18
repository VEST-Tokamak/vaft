"""Dimensional internal inductance and its li_3 normalisation (#782).

``L_i = 2 W_int / I_p^2`` is Romero's definition and ``li_3 = 2 L_i / (mu0 R0)``
the IMAS one; what is pinned is that the three functions, the existing
``li_3_from_Bp2_volume_integral`` and the start-up circuit's internal term all
describe the same henries.
"""

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.formula.equilibrium import (
    internal_inductance_from_li_3_R0,
    internal_inductance_from_W_int_Ip,
    li_3_from_Bp2_volume_integral,
    li_3_from_internal_inductance_R0,
)
from vaft.formula.startup import (
    plasma_external_inductance_hirshman_from_R_eps_kappa,
    plasma_inductance_hirshman_from_R_eps_kappa_li,
)


@pytest.mark.parametrize("ip, r0", [(1.0e5, 0.4), (-2.0e5, 0.6)])
def test_a_uniform_current_cylinder_has_li_3_one_half_by_every_route(ip, r0):
    # B_p = mu0 I r / (2 pi a^2) inside, so int B_p^2 dV = mu0^2 I^2 R0 / 4 and
    # W_int = mu0 I^2 R0 / 8 whatever a is: L_i = mu0 R0 / 4, li_3 = 1/2.
    w_int = MU0 * ip**2 * r0 / 8.0
    l_i = internal_inductance_from_W_int_Ip(w_int, ip)
    assert l_i == pytest.approx(MU0 * r0 / 4.0, rel=1e-14, abs=0.0)
    assert li_3_from_internal_inductance_R0(l_i, r0) == pytest.approx(0.5, rel=1e-14, abs=0.0)
    assert li_3_from_Bp2_volume_integral(2.0 * MU0 * w_int, ip, r0) == pytest.approx(
        li_3_from_internal_inductance_R0(l_i, r0), rel=1e-14, abs=0.0
    )


def test_the_li_3_conversions_are_inverse():
    for li_3, r0 in [(0.3, 0.4), (1.1, 0.45), (2.0, 6.2)]:
        l_i = internal_inductance_from_li_3_R0(li_3, r0)
        assert li_3_from_internal_inductance_R0(l_i, r0) == pytest.approx(li_3, rel=1e-14, abs=0.0)


def test_the_circuit_internal_term_is_this_internal_inductance():
    # The Hirshman total adds mu0 R li / 2 to the external fit; with li = li_3
    # that difference must be exactly L_i from the li_3 conversion.
    r, eps, kappa, li_3 = 0.45, 0.7, 1.8, 0.9
    total = plasma_inductance_hirshman_from_R_eps_kappa_li(r, eps, kappa, li_3)
    external = plasma_external_inductance_hirshman_from_R_eps_kappa(r, eps, kappa)
    assert total - external == pytest.approx(
        internal_inductance_from_li_3_R0(li_3, r), rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("ip", [0.0, np.nan, np.inf])
def test_no_internal_inductance_without_a_current(ip):
    with pytest.raises(ValueError, match="Ip"):
        internal_inductance_from_W_int_Ip(1.0, ip)


@pytest.mark.parametrize("r0", [0.0, -0.4, np.nan])
def test_li_3_needs_a_positive_normalising_radius(r0):
    with pytest.raises(ValueError, match="R0"):
        li_3_from_internal_inductance_R0(1.0e-7, r0)
