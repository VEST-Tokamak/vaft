"""`compute_power_balance` builds W_th with the same 3/2 that
`compute_tau_E_exp` uses (issue #1282; cold review 0.8.0
process-ml-and-omas-wrappers F1).

The balance used 2/3 for the dW/dt term while tau_E divided a 3/2 W_th by
the resulting P_loss, so dW/dt was 4/9 of the ideal-plasma value and P_loss
40-60 % low on a short-pulse machine where dW/dt is a large term.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("omas")

from omas import ODS

from vaft.omas.formula_wrapper import compute_power_balance


def test_dwdt_is_the_derivative_of_three_halves_p_v():
    # Three slices, unit volume, <p>_V rising by 100 Pa every 10 ms: with
    # W_th = 3/2 <p>_V V the stored energy is linear in time, so any
    # derivative stencil returns the hand value 150 J / 0.01 s exactly.
    times = [0.30, 0.31, 0.32]
    pressures = np.asarray([100.0, 200.0, 300.0])
    ods = ODS(consistency_check=False)
    rho = np.linspace(0.0, 1.0, 11)
    for k, t in enumerate(times):
        eq = f"equilibrium.time_slice.{k}"
        ods[f"{eq}.time"] = t
        ods[f"{eq}.global_quantities.ip"] = 10_000.0
        ods[f"{eq}.global_quantities.volume"] = 1.0
        ods[f"{eq}.global_quantities.magnetic_axis.b_field_tor"] = 0.1
        ods[f"{eq}.global_quantities.magnetic_axis.r"] = 0.4
        ods[f"{eq}.profiles_1d.rho_tor_norm"] = rho
        ods[f"{eq}.profiles_1d.volume"] = rho
        cp = f"core_profiles.profiles_1d.{k}"
        ods[f"{cp}.time"] = t
        ods[f"{cp}.grid.rho_tor_norm"] = rho
        ods[f"{cp}.electrons.density"] = np.full(rho.size, 1.0e19)
        ods[f"{cp}.electrons.temperature"] = np.full(rho.size, 100.0)

    n = len(times)
    voltage = (np.asarray(times), np.ones(n), np.full(n, 0.25), np.full(n, 0.75))
    with (
        patch("vaft.omas.formula_wrapper.compute_voltage_consumption", return_value=voltage),
        patch(
            "vaft.omas.process_wrapper.compute_ohmic_heating_power_from_core_profiles",
            return_value=100.0,
        ),
        patch("vaft.omas.update.update_equilibrium_global_quantities_volume"),
        patch(
            "vaft.omas.formula_wrapper.compute_volume_averaged_pressure",
            return_value=pressures,
        ),
        patch(
            "vaft.omas.formula_wrapper._compute_sync_radiation_power_series",
            return_value=np.zeros(n),
        ),
        patch(
            "vaft.omas.formula_wrapper._compute_bremsstrahlung_power_series",
            return_value=np.zeros(n),
        ),
    ):
        result = compute_power_balance(ods, include_line_radiation=False)

    np.testing.assert_allclose(result["time"], times)
    # W_th = 3/2 * 100 Pa * 1 m^3 = 150 J per step, 10 ms apart.
    np.testing.assert_allclose(result["dWdt"], np.full(n, 150.0 / 0.01), rtol=1e-12)
    # P_loss = P_heat - dW/dt - P_rad with P_heat = 100 W and no radiation.
    np.testing.assert_allclose(result["P_loss"], 100.0 - 15_000.0, rtol=1e-12)


def test_the_balance_and_tau_e_use_one_stored_energy_on_a_multi_slice_ods():
    from vaft.omas.process_wrapper import compute_volume_averaged_pressure
    from vaft.omas.sample import sample_ods
    from vaft.process.numerical import time_derivative

    from _synthetic_inputs import make_power_balance

    ods = make_power_balance(sample_ods(39915))
    logging.disable(logging.WARNING)
    try:
        balance = compute_power_balance(ods, include_line_radiation=False)
        # The W_th compute_tau_E_exp forms per slice: <p>_V * 3/2 * V.
        p_vol = np.asarray(
            compute_volume_averaged_pressure(ods, time_slice=None, option="core_profiles"),
            float,
        )
    finally:
        logging.disable(logging.NOTSET)
    n = len(ods["equilibrium.time_slice"])
    volume = np.asarray(
        [float(ods[f"equilibrium.time_slice.{i}.global_quantities.volume"]) for i in range(n)]
    )
    t = np.asarray([float(ods[f"equilibrium.time_slice.{i}.time"]) for i in range(n)])
    w_th = p_vol * 1.5 * volume
    assert np.isfinite(w_th).all() and (w_th > 0).all()
    np.testing.assert_allclose(balance["time"], t)
    np.testing.assert_allclose(balance["dWdt"], time_derivative(t, w_th), rtol=1e-9)
