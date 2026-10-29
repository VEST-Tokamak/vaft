"""q_min is the q of smallest magnitude, kept with its sign.

Under COCOS 3-6 and 13-16 q is negative (Sauter Table I), so a plain
``q.min()`` returns the largest |q|. The OMAS updater must agree with the EQDSK
and VFIT readers, which take ``argmin(|q|)``.
"""
from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.data.cocos import cocos_spec
from vaft.omas.update import update_equilibrium_global_quantities_q_min

# Reversed-shear profile: the minimum of |q| sits off axis at rho = 0.5.
Q_POSITIVE = np.array([1.6, 1.3, 1.1, 1.4, 2.5, 4.0])
RHO = np.linspace(0.0, 1.0, Q_POSITIVE.size)


def _ods(q: np.ndarray, *, rho: bool = True) -> ODS:
    ods = ODS()
    ods["equilibrium.time_slice.0.profiles_1d.q"] = q
    if rho:
        ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = RHO
    return ods


@pytest.mark.parametrize("cocos", [1, 3, 5, 11, 13, 15])
def test_q_min_is_the_smallest_magnitude_with_its_sign(cocos):
    # sign(q) = sigma_Ip sigma_B0 sigma_rhotp; take Ip and B0 positive.
    sign = cocos_spec(cocos).expected_sign("q", sigma_ip=1, sigma_b0=1)
    ods = _ods(sign * Q_POSITIVE)

    update_equilibrium_global_quantities_q_min(ods)

    gq = ods["equilibrium.time_slice.0.global_quantities"]
    assert gq["q_min.value"] == pytest.approx(sign * 1.1)
    assert gq["q_min.rho_tor_norm"] == pytest.approx(RHO[2])


def test_negative_q_is_not_reported_as_its_largest_magnitude():
    ods = _ods(-Q_POSITIVE)
    update_equilibrium_global_quantities_q_min(ods)
    value = ods["equilibrium.time_slice.0.global_quantities.q_min.value"]
    assert value != pytest.approx(-Q_POSITIVE.max())
    assert value == pytest.approx(-1.1)


def test_q_min_matches_the_eqdsk_reader_definition():
    q = -Q_POSITIVE
    ods = _ods(q)
    update_equilibrium_global_quantities_q_min(ods)
    expected = q[int(np.argmin(np.abs(q)))]
    assert ods["equilibrium.time_slice.0.global_quantities.q_min.value"] == expected


def test_nonfinite_samples_are_skipped_and_rho_is_optional():
    q = np.array([np.nan, -1.3, -1.1, -2.0])
    ods = _ods(q, rho=False)
    update_equilibrium_global_quantities_q_min(ods)
    gq = ods["equilibrium.time_slice.0.global_quantities"]
    assert gq["q_min.value"] == pytest.approx(-1.1)
    assert "q_min.rho_tor_norm" not in gq


def test_an_all_nan_profile_writes_nothing():
    ods = _ods(np.full(4, np.nan), rho=False)
    update_equilibrium_global_quantities_q_min(ods)
    assert "q_min" not in ods["equilibrium.time_slice.0.global_quantities"]


def test_a_stale_location_is_cleared_when_no_rho_is_available():
    ods = _ods(-Q_POSITIVE, rho=False)
    ods["equilibrium.time_slice.0.global_quantities.q_min.rho_tor_norm"] = 0.9
    update_equilibrium_global_quantities_q_min(ods)
    gq = ods["equilibrium.time_slice.0.global_quantities"]
    assert gq["q_min.value"] == pytest.approx(-1.1)
    assert "q_min.rho_tor_norm" not in gq


def test_the_summary_reads_the_signed_value():
    from vaft.database._summary import _extract_q_min

    ods = _ods(-Q_POSITIVE)
    update_equilibrium_global_quantities_q_min(ods)
    assert _extract_q_min(ods["equilibrium.time_slice.0"]) == pytest.approx(-1.1)
