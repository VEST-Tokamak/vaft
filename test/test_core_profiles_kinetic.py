"""Tests for the extended core_profiles(): real Ti/Vtor, kinetic pressure, eq grid."""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import core_profiles

E_CHARGE = 1.602176634e-19


def _ne(psin):
    return 4e18 * (1.0 - np.asarray(psin, float) ** 2)


def _te(psin):
    return 300.0 * (1.0 - np.asarray(psin, float) ** 2) ** 1.5


def _ti(psin):
    return 20.0 * (1.0 - np.asarray(psin, float) ** 2)


def _vtor(psin):
    return 1.5e4 * (1.0 - np.asarray(psin, float))


def _make_ods(with_eq=True, n_grid=33):
    ods = ODS()
    times = np.array([0.299, 0.300, 0.301])
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = times
    psi_n_ch = np.linspace(0.05, 0.85, 5)
    for i, p in enumerate(psi_n_ch):
        ods[f"thomson_scattering.channel.{i}.position.r"] = 0.25 + 0.05 * i
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = np.full(3, float(_te(p)))
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = np.full(3, float(_ne(p)))
    if with_eq:
        rho_tor = np.linspace(0.0, 1.0, n_grid)
        # monotone psi from axis (-0.05 Wb) to boundary (0.0 Wb): psi_N = rho_tor^2
        psi = -0.05 * (1.0 - rho_tor**2)
        ods["equilibrium.ids_properties.homogeneous_time"] = 1
        ods["equilibrium.time"] = np.array([0.300])
        ods["equilibrium.time_slice.0.time"] = 0.300
        ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = rho_tor
        ods["equilibrium.time_slice.0.profiles_1d.psi"] = psi
    return ods, psi_n_ch


def test_kinetic_core_profiles_on_equilibrium_grid():
    ods, psi_n_ch = _make_ods()
    core_profiles(
        ods, 300.0, psi_n_ch, _ne, _te,
        T_i_function=_ti, V_tor_function=_vtor,
    )
    base = "core_profiles.profiles_1d.0"
    rho_tor = np.linspace(0.0, 1.0, 33)
    psi = -0.05 * (1.0 - rho_tor**2)
    psin = rho_tor**2

    np.testing.assert_allclose(ods[f"{base}.grid.rho_tor_norm"], rho_tor)
    np.testing.assert_allclose(ods[f"{base}.grid.psi"], psi)
    np.testing.assert_allclose(ods[f"{base}.electrons.temperature"], _te(psin), rtol=1e-10)
    np.testing.assert_allclose(ods[f"{base}.ion.0.temperature"], _ti(psin), rtol=1e-10)
    np.testing.assert_allclose(ods[f"{base}.ion.0.velocity.toroidal"], _vtor(psin), rtol=1e-10)
    np.testing.assert_allclose(
        ods[f"{base}.pressure_thermal"],
        E_CHARGE * _ne(psin) * (_te(psin) + _ti(psin)),
        rtol=1e-10,
    )
    # H+ metadata and quasi-neutrality (ni = ne)
    assert str(ods[f"{base}.ion.0.label"]) == "H+"
    assert float(ods[f"{base}.ion.0.element.0.a"]) == pytest.approx(1.0, abs=0.01)
    assert float(ods[f"{base}.ion.0.element.0.z_n"]) == pytest.approx(1.0)
    np.testing.assert_allclose(ods[f"{base}.ion.0.density"], _ne(psin), rtol=1e-10)
    # IDS bookkeeping
    assert int(ods["core_profiles.ids_properties.homogeneous_time"]) == 1
    np.testing.assert_allclose(ods["core_profiles.time"], [0.300])


def test_duplicate_time_replaced():
    ods, psi_n_ch = _make_ods()
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te, T_i_function=_ti)
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te, T_i_function=_ti)
    assert len(ods["core_profiles.profiles_1d"]) == 1


def test_fallback_grid_is_not_labeled_rho_tor_norm():
    ods, psi_n_ch = _make_ods(with_eq=False)
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te)
    base = "core_profiles.profiles_1d.0"
    # psi_N-based fallback grid must be stored as rho_pol_norm = sqrt(psi_N)
    assert f"{base}.grid.rho_pol_norm" in ods
    np.testing.assert_allclose(
        ods[f"{base}.grid.rho_pol_norm"], np.sqrt(np.linspace(0, 1, 100))
    )
    assert f"{base}.grid.rho_tor_norm" not in ods


def test_legacy_call_without_ti_uses_te():
    ods, psi_n_ch = _make_ods()
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te)
    base = "core_profiles.profiles_1d.0"
    np.testing.assert_allclose(
        ods[f"{base}.ion.0.temperature"], ods[f"{base}.electrons.temperature"]
    )
