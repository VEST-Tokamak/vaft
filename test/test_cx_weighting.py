"""Regression tests for the charge-exchange uncertainty-weighting fix.

Bug: OMAS (>=0.94.2) splits an assigned uarray immediately into ``<leaf>.data``
(nominal) + ``<leaf>.data_error_upper``. The old CX code took its sigma from
``unumpy.std_devs(ion['t_i.data'])`` on the re-read ``.data`` -> ALL ZEROS ->
CX fits were silently UNWEIGHTED. The fix adds ``_leaf_values_and_errors`` which
reads ``.data_error_upper`` explicitly (mirroring the already-correct Thomson
path), ``_sanitize_std`` to stop single-channel weight blow-up, and writes
``ion.0.temperature_fit.measured_error_upper`` in ``core_profiles``.

Run: ``pytest test_cx_weighting.py -v``
"""
import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS
from uncertainties import unumpy

from vaft.process import profile


# --- synthetic geometry / physics -------------------------------------------
N_CX = 11
RHO_CX = np.linspace(0.05, 0.95, N_CX)
OUTLIER_I = N_CX // 2                       # a mid-radius channel (rho ~ 0.5)
RHO_OUT = float(RHO_CX[OUTLIER_I])
TIME_MS = 300.0
TIME_S = TIME_MS / 1e3


def _ti_true(rho):
    """Smooth Ti(rho) that rolls off to 0 at the edge (matches the poly model)."""
    return 300.0 * (1.0 - np.clip(np.asarray(rho, float), 0.0, 1.0))


def _set_ti_uarray(ods, i, val, err):
    """Assign a uarray to the t_i leaf -> OMAS splits it into data + error."""
    ods[f"charge_exchange.channel.{i}.ion.0.t_i.data"] = unumpy.uarray([val], [err])


def _build_cx_ods():
    """CX IDS with real per-channel errors; one high-sigma mid-radius outlier."""
    ods = ODS()
    ods["charge_exchange.time"] = np.array([TIME_S])

    ti_vals = _ti_true(RHO_CX).astype(float)
    ti_errs = np.full(N_CX, 8.0)             # small, honest per-channel sigma

    # outlier: value 250 eV above the trend, but with a HUGE sigma (500 eV)
    ti_vals[OUTLIER_I] = _ti_true(RHO_OUT) + 250.0
    ti_errs[OUTLIER_I] = 500.0

    for i in range(N_CX):
        _set_ti_uarray(ods, i, ti_vals[i], ti_errs[i])
        # velocity_tor must exist for the loop; values are irrelevant here
        ods[f"charge_exchange.channel.{i}.ion.0.velocity_tor.data"] = \
            unumpy.uarray([1.0e3], [1.0e2])
    return ods, ti_vals, ti_errs


# --- 1. document the bug: std_devs on the re-read .data is all-zero ----------
def test_omas_split_makes_std_devs_zero():
    ods, _, ti_errs = _build_cx_ods()

    data_leaf = ods["charge_exchange.channel.0.ion.0.t_i.data"]
    # The whole point of the bug: the nominal .data carries NO uncertainty.
    assert np.allclose(unumpy.std_devs(data_leaf), 0.0), (
        "expected OMAS to strip uncertainty from the re-read .data "
        "(this is the silent-unweighting bug the fix addresses)"
    )
    # ...but the real error is retrievable from the sibling leaf.
    err_leaf = np.asarray(
        ods["charge_exchange.channel.0.ion.0.t_i.data_error_upper"], dtype=float
    )
    assert np.isclose(err_leaf.reshape(-1)[0], ti_errs[0])


def test_helper_recovers_real_sigma():
    ods, ti_vals, ti_errs = _build_cx_ods()
    node = ods["charge_exchange.channel.0.ion.0.t_i"]
    val, sig = profile._leaf_values_and_errors(node, 0)
    assert np.isclose(val, ti_vals[0])
    assert np.isclose(sig, ti_errs[0])          # NOT zero -> bug fixed


def test_sanitize_std_replaces_bad_sigmas_with_median():
    out = profile._sanitize_std([10.0, 0.0, np.nan, 20.0, 30.0])
    assert np.all(np.isfinite(out)) and np.all(out > 0)
    med = np.median([10.0, 20.0, 30.0])
    assert np.isclose(out[1], med) and np.isclose(out[2], med)
    # all-bad -> uniform (unweighted) fallback, never a blow-up
    uni = profile._sanitize_std([0.0, np.nan, -1.0])
    assert np.allclose(uni, 1.0)


# --- 2. the fit is genuinely error-weighted ---------------------------------
def test_cx_fit_downweights_high_sigma_outlier():
    ods, _, _ = _build_cx_ods()

    # weighted: real .data_error_upper is used as 1/sigma weights
    _, ti_fn_w, *_ = profile.profile_fitting_charge_exchange(
        ods, TIME_MS, RHO_CX,
        fitting_function_ti="polynomial", Ti_order=3, uncertainty_option=1,
    )
    # unweighted reference: identical data/model, weights off
    _, ti_fn_u, *_ = profile.profile_fitting_charge_exchange(
        ods, TIME_MS, RHO_CX,
        fitting_function_ti="polynomial", Ti_order=3, uncertainty_option=0,
    )

    truth = float(_ti_true(RHO_OUT))
    err_weighted = abs(float(ti_fn_w(RHO_OUT)) - truth)
    err_unweighted = abs(float(ti_fn_u(RHO_OUT)) - truth)

    # The high-sigma outlier drags the unweighted fit up; the weighted fit,
    # trusting the small-sigma channels, stays near the true trend.
    assert err_weighted < err_unweighted, (
        f"weighted fit should be closer to truth at the outlier radius: "
        f"weighted err={err_weighted:.1f} eV vs unweighted err={err_unweighted:.1f} eV"
    )
    # and it should be meaningfully down-weighted, not a marginal difference
    assert err_weighted < 0.5 * err_unweighted


# --- 3. core_profiles writes measured_error_upper ---------------------------
def _add_ts_and_equilibrium(ods):
    n_ts = 8
    rho_ts = np.linspace(0.1, 0.9, n_ts)
    ods["thomson_scattering.time"] = np.array([TIME_S])
    for i in range(n_ts):
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = np.array([1e19 * (1 - rho_ts[i])])
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = np.array([300.0 * (1 - rho_ts[i])])

    ods["equilibrium.time"] = np.array([TIME_S])
    grid = np.linspace(0.0, 1.0, 20)
    ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = grid
    ods["equilibrium.time_slice.0.profiles_1d.psi"] = grid  # monotonic psi_N proxy
    return rho_ts


def test_core_profiles_stores_measured_error_upper():
    ods, _, ti_errs = _build_cx_ods()
    rho_ts = _add_ts_and_equilibrium(ods)

    _, ti_fn, *_ = profile.profile_fitting_charge_exchange(
        ods, TIME_MS, RHO_CX,
        fitting_function_ti="polynomial", Ti_order=3, uncertainty_option=1,
    )

    n_e_fn = lambda r: 1e19 * (1.0 - np.clip(np.asarray(r, float), 0, 1))
    t_e_fn = lambda r: 300.0 * (1.0 - np.clip(np.asarray(r, float), 0, 1))

    profile.core_profiles(
        ods, TIME_MS, rho_ts, n_e_fn, t_e_fn,
        T_i_function=ti_fn, ti_mapped_rho_position=RHO_CX,
    )

    key = "core_profiles.profiles_1d.0.ion.0.temperature_fit.measured_error_upper"
    assert key in ods, "measured_error_upper was not written"
    stored = np.asarray(ods[key], dtype=float)
    # one entry per finite-rho CX channel, all positive & finite, real values
    assert stored.shape == RHO_CX.shape
    assert np.all(np.isfinite(stored)) and np.all(stored > 0)
    assert np.isclose(stored[0], ti_errs[0])
    assert np.isclose(stored[OUTLIER_I], ti_errs[OUTLIER_I])


# --- 4. backward compatibility: Ti=Te fallback still works -------------------
def test_core_profiles_ti_equals_te_fallback():
    ods, _, _ = _build_cx_ods()
    rho_ts = _add_ts_and_equilibrium(ods)

    n_e_fn = lambda r: 1e19 * (1.0 - np.clip(np.asarray(r, float), 0, 1))
    t_e_fn = lambda r: 300.0 * (1.0 - np.clip(np.asarray(r, float), 0, 1))

    # no T_i_function -> ion.0.temperature falls back to Te, no CX metadata
    profile.core_profiles(ods, TIME_MS, rho_ts, n_e_fn, t_e_fn)

    te = np.asarray(ods["core_profiles.profiles_1d.0.electrons.temperature"], float)
    ti = np.asarray(ods["core_profiles.profiles_1d.0.ion.0.temperature"], float)
    assert np.allclose(ti, te)
    assert "core_profiles.profiles_1d.0.ion.0.temperature_fit.measured_error_upper" not in ods
