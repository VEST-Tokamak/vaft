"""The power-balance cache behind `compute_tau_E_exp` and
`compute_tau_E_engineering_parameters` follows the ODS content and lifetime
(cold review 0.8.0 process-ml-and-omas-wrappers F2).

It was keyed by ``id(ods)`` and the two slice counts alone, so an in-place
edit that kept the slice count -- the "re-fit, then recompute" loop of the
kinetic notebooks -- was served the stale balance (tau_E 51 % high in the
verifier's repro), and a freed id reused by a later ODS could be served
another shot's balance.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pytest

pytest.importorskip("omas")

from omas import ODS

from vaft.omas import formula_wrapper as fw


@pytest.fixture(autouse=True)
def _empty_cache():
    fw._POWER_BALANCE_CACHE.clear()
    yield
    fw._POWER_BALANCE_CACHE.clear()


def _two_slice_ods(temperature: float) -> ODS:
    ods = ODS(consistency_check=False)
    rho = np.linspace(0.0, 1.0, 5)
    for k, t in enumerate((0.30, 0.31)):
        ods[f"equilibrium.time_slice.{k}.time"] = t
        ods[f"core_profiles.profiles_1d.{k}.time"] = t
        ods[f"core_profiles.profiles_1d.{k}.grid.rho_tor_norm"] = rho
        ods[f"core_profiles.profiles_1d.{k}.electrons.temperature"] = np.full(rho.size, temperature)
    return ods


def _balance_from_temperature(ods):
    """Stand-in for compute_power_balance: P_loss tracks the ODS content."""
    t_e = float(np.asarray(ods["core_profiles.profiles_1d.0.electrons.temperature"])[0])
    return {"time": np.asarray([0.30, 0.31]), "P_loss": np.asarray([t_e, t_e])}


def test_an_in_place_edit_with_the_same_slice_count_is_recomputed(monkeypatch):
    calls = []

    def fake(ods, *args, **kwargs):
        calls.append(1)
        return _balance_from_temperature(ods)

    monkeypatch.setattr(fw, "compute_power_balance", fake)
    ods = _two_slice_ods(100.0)
    first = fw._get_cached_power_balance(ods)
    assert first["P_loss"][0] == 100.0 and len(calls) == 1
    # Untouched: the entry is served.
    assert fw._get_cached_power_balance(ods) is first and len(calls) == 1

    for k in range(2):
        leaf = f"core_profiles.profiles_1d.{k}.electrons.temperature"
        ods[leaf] = np.asarray(ods[leaf], float) * 2.0
    second = fw._get_cached_power_balance(ods)
    assert len(calls) == 2, "same slice count, new content: must recompute"
    assert second["P_loss"][0] == 200.0


def test_a_freed_ods_leaves_no_entry_for_a_reused_id(monkeypatch):
    monkeypatch.setattr(fw, "compute_power_balance", _balance_from_temperature)
    ods = _two_slice_ods(100.0)
    fw._get_cached_power_balance(ods)
    assert id(ods) in fw._POWER_BALANCE_CACHE
    del ods
    gc.collect()
    assert fw._POWER_BALANCE_CACHE == {}


def test_compute_tau_e_exp_follows_a_profile_edit_and_its_revert():
    """The verifier's repro through the public entry point, on real data."""
    from vaft.omas.sample import sample_ods

    from _synthetic_inputs import make_power_balance

    ods = make_power_balance(sample_ods(39915))
    n = len(ods["core_profiles.profiles_1d"])
    i = n // 2
    leaves = ("electrons.temperature", "ion.0.temperature", "pressure_thermal")

    def scale(factor):
        for k in range(n):
            cp = ods["core_profiles.profiles_1d"][k]
            for leaf in leaves:
                cp[leaf] = np.asarray(cp[leaf], float) * factor

    # The balance is cached on the doubled profiles; the revert keeps the
    # slice count, so the old cache served the hot P_loss to the reverted ODS
    # (W_th is not cached, which is why tau_E moved at all: 0.00575 s against
    # the 0.00381 s a cleared cache gives in the verifier's repro).
    logging.disable(logging.WARNING)
    try:
        scale(2.0)
        tau_hot = fw.compute_tau_E_exp(ods, i)
        scale(0.5)
        tau_back = fw.compute_tau_E_exp(ods, i)
        fw._POWER_BALANCE_CACHE.clear()
        tau_true = fw.compute_tau_E_exp(ods, i)
    finally:
        logging.disable(logging.NOTSET)

    assert tau_hot != pytest.approx(tau_true, rel=1e-3), "the edit must reach tau_E"
    assert tau_back == pytest.approx(tau_true, rel=1e-9), "the revert must not be served the hot balance"
