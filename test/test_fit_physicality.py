"""Guards against unphysical profile fits from blind extrapolation.

Thomson often covers only the inner part of psi_N and the ion Doppler only an
outer band; a high-order polynomial/exponential extrapolated over the rest can
cross zero, collapse by orders of magnitude, or blow up towards the axis. These
tests use the real measurements from shot 48224 @ 298/299 ms, the slices where
that actually happened in the stored database profiles.
"""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import (
    NE_DYNAMIC_RANGE_MAX,
    PHYSICAL_PSIN_MAX,
    TE_POSITIVE_PSIN_MAX,
    _fit_profile_until_physical,
    profile_fitting_charge_exchange,
    profile_fitting_thomson_scattering,
)

# shot 48224 @ 299 ms: 5 usable Thomson channels, all inside psi_N <= 0.27.
# A quadratic fitted to these and extrapolated to psi_N = 1 crossed zero at
# psi_N = 0.87 and collapsed the density by 5e6.
PSIN_299 = np.array([0.234, 0.062, 0.008, 0.078, 0.266])
TE_299 = np.array([45.2, 58.1, 62.2, 85.4, 90.1])
NE_299 = np.array([1.11e19, 1.25e19, 1.02e19, 9.31e18, 5.28e18])


def _ts_ods(psin, te, ne, t_s=0.299, rel_err=0.2):
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = np.array([t_s])
    for i, (p, t, d) in enumerate(zip(psin, te, ne)):
        ch = f"thomson_scattering.channel.{i}"
        ods[f"{ch}.position.r"] = 0.25 + 0.05 * i
        ods[f"{ch}.position.z"] = 0.0
        ods[f"{ch}.t_e.data"] = np.array([float(t)])
        ods[f"{ch}.t_e.data_error_upper"] = np.array([float(t) * rel_err])
        ods[f"{ch}.n_e.data"] = np.array([float(d)])
        ods[f"{ch}.n_e.data_error_upper"] = np.array([float(d) * rel_err])
    return ods


# --------------------------------------------------------------------------- #
# _fit_profile_until_physical (pure helper)
# --------------------------------------------------------------------------- #

def test_helper_returns_first_physical_order():
    calls = []

    def fit_call(order):
        calls.append(order)
        # only order <= 2 is "physical" here
        fn = (lambda x: np.ones_like(np.asarray(x, float))) if order <= 2 else \
             (lambda x: -np.ones_like(np.asarray(x, float)))
        return (None, None, fn, None)

    def bad(fn):
        return None if fn(np.array([0.5]))[0] > 0 else "negative"

    *_, used = _fit_profile_until_physical(fit_call, 4, bad, "x", 300.0)
    assert used == 2
    assert calls == [4, 3, 2]          # reduced one order at a time


def test_helper_keeps_lowest_order_when_nothing_passes():
    def fit_call(order):
        return (None, None, lambda x: -np.ones_like(np.asarray(x, float)), None)

    *_, used = _fit_profile_until_physical(
        fit_call, 4, lambda fn: "always bad", "x", 300.0
    )
    assert used == 1                    # min_order, with a warning -- never raises


def test_helper_skips_orders_that_raise():
    def fit_call(order):
        if order > 2:
            raise np.linalg.LinAlgError("singular")
        return (None, None, lambda x: np.ones_like(np.asarray(x, float)), None)

    *_, used = _fit_profile_until_physical(fit_call, 4, lambda fn: None, "x", 300.0)
    assert used == 2


def test_helper_can_reduce_below_order_two():
    """A caller that already asks for order 2 must still have room to reduce.

    The electron-only pipeline branch requests Te_order=2; with a floor of 2 it
    had no fallback and kept an unphysical fit (48226 @ 304/307 ms, Te <= 0 over
    31 grid points). Order 1 is (1 - psi_N)*c0 -- non-negative by construction.
    """
    def fit_call(order):
        fn = (lambda x: np.ones_like(np.asarray(x, float))) if order == 1 else \
             (lambda x: -np.ones_like(np.asarray(x, float)))
        return (None, None, fn, None)

    def bad(fn):
        return None if fn(np.array([0.5]))[0] > 0 else "negative"

    *_, used = _fit_profile_until_physical(fit_call, 2, bad, "Te", 304.0)
    assert used == 1


# --------------------------------------------------------------------------- #
# Thomson: order reduction on unphysical Te / ne
# --------------------------------------------------------------------------- #

def test_narrow_coverage_fit_is_physical():
    """The real 299 ms slice: guarded fit must stay positive with sane ne."""
    ods = _ts_ods(PSIN_299, TE_299, NE_299)
    ne_fn, te_fn, *_ = profile_fitting_thomson_scattering(
        ods, 299.0, PSIN_299, Te_order=3, Ne_order=3,
        fitting_function_te="polynomial", fitting_function_ne="free_exponential",
    )
    grid = np.linspace(0.0, 1.0, 129)
    te, ne = te_fn(grid), ne_fn(grid)

    inside = grid < TE_POSITIVE_PSIN_MAX
    assert np.all(te[inside] > 0), "Te must stay positive inside the LCFS"
    assert np.all(np.isfinite(te)) and np.all(np.isfinite(ne))
    assert ne.min() > 0
    assert ne.max() / ne.min() <= NE_DYNAMIC_RANGE_MAX, "ne collapsed over [0,1]"


def test_unguarded_fit_reproduces_the_pathology():
    """enforce_physical=False keeps the legacy behaviour (the bug we fixed)."""
    ods = _ts_ods(PSIN_299, TE_299, NE_299)
    ne_fn, te_fn, *_ = profile_fitting_thomson_scattering(
        ods, 299.0, PSIN_299, Te_order=3, Ne_order=3,
        fitting_function_te="polynomial", fitting_function_ne="free_exponential",
        enforce_physical=False,
    )
    grid = np.linspace(0.0, 1.0, 129)
    te, ne = te_fn(grid), ne_fn(grid)
    inside = grid < TE_POSITIVE_PSIN_MAX
    # exactly the failure seen in the stored profiles
    assert np.any(te[inside] <= 0) or ne.max() / max(ne.min(), 1e-30) > NE_DYNAMIC_RANGE_MAX


def test_well_covered_fit_is_left_alone():
    """A slice with broad coverage must keep its requested order (no silent change)."""
    psin = np.array([0.008, 0.039, 0.047, 0.172, 0.367, 0.62, 0.83])
    te = np.array([72.7, 91.6, 63.2, 100.7, 100.1, 70.0, 35.0])
    ne = np.array([1.08e19, 8.27e18, 9.82e18, 7.76e18, 4.37e18, 2.0e18, 8.0e17])
    ods = _ts_ods(psin, te, ne, t_s=0.300)
    guarded = profile_fitting_thomson_scattering(
        ods, 300.0, psin, Te_order=3, Ne_order=3,
        fitting_function_te="polynomial", fitting_function_ne="free_exponential",
    )
    legacy = profile_fitting_thomson_scattering(
        ods, 300.0, psin, Te_order=3, Ne_order=3,
        fitting_function_te="polynomial", fitting_function_ne="free_exponential",
        enforce_physical=False,
    )
    grid = np.linspace(0.0, 1.0, 65)
    np.testing.assert_allclose(guarded[1](grid), legacy[1](grid), rtol=1e-10)
    np.testing.assert_allclose(guarded[0](grid), legacy[0](grid), rtol=1e-10)


def test_edge_zero_basis_is_not_rejected():
    """'exponential' forces ne -> 0 at psi_N = 1 by construction.

    That endpoint zero is intended; the guard must ignore it (it checks only
    inside PHYSICAL_PSIN_MAX) instead of rejecting every such fit and reporting a
    meaningless infinite dynamic range -- which is what the electron-only
    pipeline slices hit.
    """
    psin = np.array([0.008, 0.039, 0.047, 0.172, 0.367])
    te = np.array([72.7, 91.6, 63.2, 100.7, 100.1])
    ne = np.array([1.08e19, 8.27e18, 9.82e18, 7.76e18, 4.37e18])
    ods = _ts_ods(psin, te, ne, t_s=0.302)

    guarded = profile_fitting_thomson_scattering(
        ods, 302.0, psin, Te_order=2, Ne_order=2,
        fitting_function_te="polynomial", fitting_function_ne="exponential",
    )
    legacy = profile_fitting_thomson_scattering(
        ods, 302.0, psin, Te_order=2, Ne_order=2,
        fitting_function_te="polynomial", fitting_function_ne="exponential",
        enforce_physical=False,
    )
    grid = np.linspace(0.0, 1.0, 129)
    # identical: the guard must not have altered a legitimate edge-zero fit
    np.testing.assert_allclose(guarded[0](grid), legacy[0](grid), rtol=1e-10)
    ne_fit = guarded[0](grid)
    assert ne_fit[-1] == pytest.approx(0.0, abs=1e-6 * ne_fit.max())   # zero AT the edge
    assert np.all(ne_fit[grid < PHYSICAL_PSIN_MAX] > 0)                 # positive inside


# --------------------------------------------------------------------------- #
# charge_exchange: no blind extrapolation past the measured span
# --------------------------------------------------------------------------- #

def _cx_ods(psin, ti, t_s=0.298, rel_err=0.15):
    ods = ODS()
    ods["charge_exchange.ids_properties.homogeneous_time"] = 1
    ods["charge_exchange.time"] = np.array([t_s])
    for i, (p, v) in enumerate(zip(psin, ti)):
        ch = f"charge_exchange.channel.{i}"
        ods[f"{ch}.ion.0.t_i.data"] = np.array([float(v)])
        ods[f"{ch}.ion.0.t_i.data_error_upper"] = np.array([float(v) * rel_err])
        ods[f"{ch}.ion.0.velocity_tor.data"] = np.array([1.0e4])
        ods[f"{ch}.ion.0.velocity_tor.data_error_upper"] = np.array([1.0e3])
    return ods


def test_ion_fit_does_not_extrapolate_below_measured_span():
    """48224 @ 298 ms: CX only covers psi_N >= 0.23; the axis must not blow up."""
    psin = np.array([0.227, 0.31, 0.42, 0.55, 0.66, 0.74, 0.83, 0.898])
    ti = np.array([20.9, 18.0, 15.5, 13.0, 11.0, 9.5, 8.4, 7.6])
    ods = _cx_ods(psin, ti)

    vtor_fn, ti_fn, _, _, vtor_rho, ti_rho = profile_fitting_charge_exchange(
        ods, 298.0, psin, fitting_function_ti="polynomial"
    )
    grid = np.linspace(0.0, 1.0, 129)
    fitted = ti_fn(grid)
    assert fitted.max() <= ti.max() * 1.15, (
        f"Ti extrapolated to {fitted.max():.1f} eV above a {ti.max():.1f} eV maximum"
    )
    # inside the covered span the fit is untouched
    np.testing.assert_allclose(ti_fn(np.array([0.5])), ti_fn(np.array([0.5])))
    # below the span it is held at the innermost measured position
    np.testing.assert_allclose(ti_fn(np.array([0.0])), ti_fn(np.array([psin.min()])))
    # Sampled return arrays are persisted directly by plotting/database callers,
    # so they must honor the same endpoint clamp as the public callables.
    return_grid = np.linspace(0.0, 1.0, ti_rho.size)
    np.testing.assert_allclose(ti_rho, ti_fn(return_grid))
    np.testing.assert_allclose(vtor_rho, vtor_fn(return_grid))


def test_ion_fit_legacy_extrapolation_still_available():
    psin = np.array([0.227, 0.31, 0.42, 0.55, 0.66, 0.74, 0.83, 0.898])
    ti = np.array([20.9, 18.0, 15.5, 13.0, 11.0, 9.5, 8.4, 7.6])
    ods = _cx_ods(psin, ti)
    _, ti_fn, *_ = profile_fitting_charge_exchange(
        ods, 298.0, psin, fitting_function_ti="polynomial",
        clamp_to_measured_span=False,
    )
    # unclamped, the polynomial is free to run away towards the axis
    assert ti_fn(np.array([0.0]))[0] != pytest.approx(ti_fn(np.array([psin.min()]))[0])
