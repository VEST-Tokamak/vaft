"""Tests for the statistical Ti/Te ratio: estimator, core_profiles fallback,
and the Thomson-only kinetic pressure path in vaft.code.efit.

The coefficient (TI_TE_RATIO_VEST) is fitted offline on the shots that carry
BOTH electron (Thomson) and ion (IDS/charge_exchange) profiles
(ids_test/fit_ti_te_ratio.py); these tests cover the machinery, not the value.
"""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.process.profile import (
    TI_TE_RATIO_VEST,
    TI_TE_RATIO_VEST_SIGMA,
    core_profiles,
    fit_ti_te_ratio,
)
from vaft.code import efit as km

E_CHARGE = 1.602176634e-19


# --------------------------------------------------------------------------- #
# fit_ti_te_ratio (pure estimator)
# --------------------------------------------------------------------------- #

def test_fit_ti_te_ratio_recovers_synthetic_alpha():
    rng = np.random.default_rng(7)
    alpha_true = 0.25
    te = rng.uniform(40.0, 120.0, 200)
    sti = np.full_like(te, 1.5)
    ste = np.full_like(te, 4.0)
    ti = alpha_true * te + rng.normal(0.0, 1.5, te.size)
    res = fit_ti_te_ratio(te, ti, te_std=ste, ti_std=sti)
    assert res["n_points"] == 200
    assert res["alpha"] == pytest.approx(alpha_true, abs=3 * res["alpha_se"])
    assert res["alpha_se"] > 0
    assert res["alpha_scatter"] > 0
    assert res["chi2_red"] == pytest.approx(1.0, abs=0.4)


def test_fit_ti_te_ratio_exact_proportionality_and_filtering():
    te = np.array([50.0, 100.0, np.nan, -5.0, 80.0])
    ti = 0.3 * te
    res = fit_ti_te_ratio(te, ti)              # no sigmas -> plain LSQ
    assert res["alpha"] == pytest.approx(0.3, rel=1e-12)
    assert res["n_points"] == 3                # NaN and Te<=0 pairs dropped


def test_fit_ti_te_ratio_validates_input():
    with pytest.raises(ValueError):
        fit_ti_te_ratio([1.0, 2.0], [1.0])     # length mismatch
    with pytest.raises(ValueError):
        fit_ti_te_ratio([np.nan], [1.0])       # <2 valid pairs


def test_vest_constants_sane():
    # guard against accidental edits: ratio in the bootstrap CI, sigma positive
    assert 0.10 < TI_TE_RATIO_VEST < 0.25
    assert 0.0 < TI_TE_RATIO_VEST_SIGMA < TI_TE_RATIO_VEST


# --------------------------------------------------------------------------- #
# core_profiles: ratio fallback (Thomson-only slice)
# --------------------------------------------------------------------------- #

def _ne(psin):
    return 4e18 * (1.0 - np.asarray(psin, float) ** 2)


def _te(psin):
    return 300.0 * (1.0 - np.asarray(psin, float) ** 2) ** 1.5


def _make_ts_only_ods(n_grid=33):
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
    rho_tor = np.linspace(0.0, 1.0, n_grid)
    psi = -0.05 * (1.0 - rho_tor**2)           # psi_N = rho_tor^2
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    ods["equilibrium.time"] = np.array([0.300])
    ods["equilibrium.time_slice.0.time"] = 0.300
    ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = rho_tor
    ods["equilibrium.time_slice.0.profiles_1d.psi"] = psi
    return ods, psi_n_ch


def test_core_profiles_ratio_fallback_writes_ti_and_pressure():
    ods, psi_n_ch = _make_ts_only_ods()
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te, ti_te_ratio=0.2)
    base = "core_profiles.profiles_1d.0"
    psin = np.linspace(0.0, 1.0, 33) ** 2      # equilibrium grid: rho_tor^2
    np.testing.assert_allclose(
        ods[f"{base}.ion.0.temperature"], 0.2 * _te(psin), rtol=1e-10
    )
    np.testing.assert_allclose(
        ods[f"{base}.pressure_thermal"],
        E_CHARGE * _ne(psin) * (1.0 + 0.2) * _te(psin),
        rtol=1e-10,
    )
    assert f"{base}.ion.0.temperature_fit.measured" not in ods


def test_core_profiles_legacy_fallback_unchanged():
    ods, psi_n_ch = _make_ts_only_ods()
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te)   # no ratio -> legacy Ti=Te
    base = "core_profiles.profiles_1d.0"
    psin = np.linspace(0.0, 1.0, 33) ** 2
    np.testing.assert_allclose(
        ods[f"{base}.ion.0.temperature"], _te(psin), rtol=1e-10
    )
    assert f"{base}.pressure_thermal" not in ods    # legacy writes no pressure


def test_core_profiles_ratio_ignored_with_real_ion_fit():
    def _ti(psin):
        return 20.0 * (1.0 - np.asarray(psin, float) ** 2)

    ods, psi_n_ch = _make_ts_only_ods()
    core_profiles(ods, 300.0, psi_n_ch, _ne, _te, T_i_function=_ti, ti_te_ratio=0.2)
    base = "core_profiles.profiles_1d.0"
    psin = np.linspace(0.0, 1.0, 33) ** 2
    np.testing.assert_allclose(
        ods[f"{base}.ion.0.temperature"], _ti(psin), rtol=1e-10
    )
    np.testing.assert_allclose(
        ods[f"{base}.pressure_thermal"],
        E_CHARGE * _ne(psin) * (_te(psin) + _ti(psin)),
        rtol=1e-10,
    )


@pytest.mark.parametrize("ratio", [-0.1, np.nan, np.inf])
def test_core_profiles_rejects_invalid_ratio(ratio):
    ods, psi_n_ch = _make_ts_only_ods()
    with pytest.raises(ValueError, match="finite and non-negative"):
        core_profiles(
            ods,
            300.0,
            psi_n_ch,
            _ne,
            _te,
            ti_te_ratio=ratio,
        )


# --------------------------------------------------------------------------- #
# EFIT kinetic mode: Thomson-only pressure points (statistical fallback)
# --------------------------------------------------------------------------- #

class FakeNode(dict):
    """dict resolving dotted keys and numeric list indices like omas."""

    def __getitem__(self, key):
        if not isinstance(key, str):
            return super().__getitem__(key)
        cur = self
        for part in key.split("."):
            if isinstance(cur, (list, tuple)):
                cur = cur[int(part)]
            else:
                cur = dict.__getitem__(cur, part)
        return cur

    def __contains__(self, key):
        try:
            self[key]
            return True
        except Exception:
            return False


def _make_ts_only_raw_ods():
    """thomson_scattering only -- NO charge_exchange tree at all."""
    t = [0.299, 0.300, 0.301]

    def ts_channel(r, ne, sne, te, ste):
        ch = FakeNode()
        ch["position"] = FakeNode(r=r)
        ch["n_e"] = FakeNode(data=[0, ne, 0], data_error_upper=[0, sne, 0])
        ch["t_e"] = FakeNode(data=[0, te, 0], data_error_upper=[0, ste, 0])
        return ch

    ts = FakeNode()
    ts["time"] = t
    ts["channel"] = [
        ts_channel(0.40, 1.0e19, 1.0e18, 100.0, 5.0),
        ts_channel(0.55, 1.0e19, 1.0e18, 80.0, 4.0),
    ]
    ods = FakeNode()
    ods["thomson_scattering"] = ts
    return ods


def test_ts_only_fallback_default_uses_vest_ratio():
    ods = _make_ts_only_raw_ods()
    R, ne, Te, sne, sTe, Ti, sTi = km._raw_ne_te_ti(ods, 300.0, None)
    np.testing.assert_allclose(Ti, TI_TE_RATIO_VEST * Te, rtol=1e-12)
    np.testing.assert_allclose(
        sTi,
        np.sqrt((TI_TE_RATIO_VEST * sTe) ** 2 + (TI_TE_RATIO_VEST_SIGMA * Te) ** 2),
        rtol=1e-12,
    )

    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw6")
    p_exp = km.EQE * ne * (1.0 + TI_TE_RATIO_VEST) * Te
    sig_exp = km.EQE * np.sqrt(
        ((1.0 + TI_TE_RATIO_VEST) * Te * sne) ** 2
        + (ne * (1.0 + TI_TE_RATIO_VEST) * sTe) ** 2
        + (ne * Te * TI_TE_RATIO_VEST_SIGMA) ** 2
    )
    sig_exp = np.maximum(sig_exp, 0.05 * p_exp)
    assert len(pts) == len(R) + 1                       # raw6 edge anchor kept
    np.testing.assert_allclose(pts.pressr[:-1], p_exp, rtol=1e-12)
    np.testing.assert_allclose(pts.sigpre[:-1], sig_exp, rtol=1e-12)


def test_ts_only_strict_mode_raises():
    ods = _make_ts_only_raw_ods()
    with pytest.raises(Exception):
        km._raw_ne_te_ti(ods, 300.0, None, ti_te_ratio=None)
    with pytest.raises(Exception):
        km.kinetic_pressure_points(ods, 300.0, None, encoding="raw5",
                                   ti_te_ratio=None)


def test_ts_only_custom_ratio_and_sigma():
    ods = _make_ts_only_raw_ods()
    _, _, Te, _, sTe, Ti, sTi = km._raw_ne_te_ti(
        ods, 300.0, None, ti_te_ratio=0.3, ti_te_ratio_sigma=0.1
    )
    np.testing.assert_allclose(Ti, 0.3 * Te, rtol=1e-12)
    np.testing.assert_allclose(
        sTi, np.sqrt((0.3 * sTe) ** 2 + (0.1 * Te) ** 2), rtol=1e-12
    )


def test_ts_only_bad_ratio_string_rejected():
    with pytest.raises(ValueError):
        km._resolve_ti_te_ratio("bogus")


@pytest.mark.parametrize(
    "ratio,sigma",
    [
        (-0.1, 0.1),
        (np.nan, 0.1),
        (np.inf, 0.1),
        (0.2, -0.1),
        (0.2, np.nan),
    ],
)
def test_ts_only_invalid_ratio_or_sigma_rejected(ratio, sigma):
    with pytest.raises(ValueError, match="finite and non-negative"):
        km._resolve_ti_te_ratio(ratio, sigma)


def test_ts_only_invalid_uncertainties_use_finite_fallbacks():
    ods = _make_ts_only_raw_ods()
    ods["thomson_scattering"]["channel"][0]["n_e"]["data_error_upper"][1] = np.nan
    ods["thomson_scattering"]["channel"][0]["t_e"]["data_error_upper"][1] = 0.0

    R, ne, Te, sne, sTe, *_ = km._raw_ne_te_ti(ods, 300.0, None)
    assert R.size == 2
    assert sne[0] == pytest.approx(0.1 * ne[0])
    assert sTe[0] == pytest.approx(0.1 * Te[0])

    points = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw5")
    assert np.all(np.isfinite(points.sigpre))
    assert np.all(np.asarray(points.sigpre) > 0.0)


def test_fallback_not_used_when_ion_data_present():
    # rebuild the two-diagnostic fixture from test_efit_kinetic and check the
    # ratio parameter does not perturb the CX-based Ti
    t = [0.299, 0.300, 0.301]

    def cx_channel(r, ti, sti):
        ch = FakeNode()
        ch["position"] = FakeNode(r=FakeNode(time=t, data=[r, r, r]))
        ion = FakeNode()
        ion["t_i"] = FakeNode(data=[ti, ti, ti], data_error_upper=[sti, sti, sti])
        ch["ion"] = [ion]
        return ch

    ods = _make_ts_only_raw_ods()
    cx = FakeNode()
    cx["time"] = t
    cx["channel"] = [
        cx_channel(0.35, 30.0, 3.0),
        cx_channel(0.45, 30.0, 3.0),
        cx_channel(0.55, 30.0, 3.0),
        cx_channel(0.65, 30.0, 3.0),
    ]
    # A single dead/malformed channel must be skipped rather than causing the
    # outer statistical fallback to discard all four valid measurements.
    malformed = FakeNode()
    malformed["position"] = FakeNode()  # missing position.r.time/data
    malformed["ion"] = []
    cx["channel"].append(malformed)
    ods["charge_exchange"] = cx

    _, _, Te, _, _, Ti, _ = km._raw_ne_te_ti(ods, 300.0, None, ti_te_ratio=0.9)
    # constant 30 eV CX field -> fitted Ti ~ 30 eV, NOT 0.9*Te
    np.testing.assert_allclose(Ti, 30.0, rtol=1e-6)
