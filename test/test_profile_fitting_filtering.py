"""Channel filtering / time-tolerance tests for the TS profile fitter and mappers."""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import profile_fitting_thomson_scattering


def _make_ts_ods(n_ch=7, times=(0.299, 0.300, 0.301)):
    """Synthetic TS ODS: Te ~ 300*(1-psiN^2)^1.5, ne ~ 4e18*(1-psiN^2)."""
    rng = np.random.default_rng(2)
    times = np.asarray(times, float)
    psi_n = np.linspace(0.05, 0.85, n_ch)
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = times
    for i in range(n_ch):
        te = 300.0 * (1.0 - psi_n[i] ** 2) ** 1.5 + rng.normal(0, 3, times.size)
        ne = 4e18 * (1.0 - psi_n[i] ** 2) + rng.normal(0, 4e16, times.size)
        ods[f"thomson_scattering.channel.{i}.position.r"] = 0.25 + 0.05 * i
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = te
        ods[f"thomson_scattering.channel.{i}.t_e.data_error_upper"] = np.full(times.size, 8.0)
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = ne
        ods[f"thomson_scattering.channel.{i}.n_e.data_error_upper"] = np.full(times.size, 2e17)
    return ods, psi_n


def _fit(ods, rho, time_ms=300.0, **kwargs):
    return profile_fitting_thomson_scattering(
        ods, time_ms, rho, Te_order=2, Ne_order=2,
        fitting_function_te="polynomial", fitting_function_ne="polynomial",
        **kwargs,
    )


def test_zero_sigma_channel_does_not_hijack_fit():
    ods, psi_n = _make_ts_ods()
    _, te_ref, *_ = _fit(ods, psi_n)
    axis_ref = float(te_ref(np.array([0.0]))[0])

    # poison channel 3: dead 0.1 eV point with zero uncertainty at every time
    ods["thomson_scattering.channel.3.t_e.data"] = np.full(3, 0.1)
    ods["thomson_scattering.channel.3.t_e.data_error_upper"] = np.zeros(3)
    _, te_bad, *_ = _fit(ods, psi_n)
    axis_bad = float(te_bad(np.array([0.0]))[0])
    at_poisoned = float(te_bad(np.array([psi_n[3]]))[0])

    assert abs(axis_bad - axis_ref) / axis_ref < 0.3
    # with the dead channel dropped, the fit must not collapse onto 0.1 eV there
    assert at_poisoned > 100.0


def test_nan_rho_channel_is_dropped():
    ods, psi_n = _make_ts_ods()
    rho = psi_n.copy()
    rho[6] = np.nan  # e.g. outside-LCFS channel from the mapper
    n_e_f, te_f, *_ = _fit(ods, rho)
    assert np.isfinite(float(te_f(np.array([0.2]))[0]))
    assert np.isfinite(float(n_e_f(np.array([0.2]))[0]))


def test_nearest_time_within_tolerance():
    ods, psi_n = _make_ts_ods()
    _fit(ods, psi_n, time_ms=300.4)  # nearest 300 ms, within default 1.0 ms


def test_time_out_of_tolerance_raises():
    ods, psi_n = _make_ts_ods()
    with pytest.raises(ValueError):
        _fit(ods, psi_n, time_ms=305.0)


def test_returned_te_profile_clipped_nonnegative():
    ods, psi_n = _make_ts_ods()
    *_, t_e_rho = _fit(ods, psi_n)
    assert np.all(t_e_rho >= 0.0)
