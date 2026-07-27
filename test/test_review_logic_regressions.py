from unittest.mock import patch

import numpy as np
from omas import ODS

from vaft.omas.formula_wrapper import compute_power_balance


def test_compute_power_balance_runs_synchrotron_profile_path():
    ods = ODS(consistency_check=False)
    rho = np.linspace(0.0, 1.0, 11)
    ods["equilibrium.time_slice.0.time"] = 0.3
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 10_000.0
    ods["equilibrium.time_slice.0.global_quantities.volume"] = 1.0
    ods["equilibrium.time_slice.0.global_quantities.magnetic_axis.b_field_tor"] = 0.1
    ods["equilibrium.time_slice.0.global_quantities.magnetic_axis.r"] = 0.4
    ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = rho
    ods["equilibrium.time_slice.0.profiles_1d.volume"] = rho
    ods["core_profiles.profiles_1d.0.time"] = 0.3
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = rho
    ods["core_profiles.profiles_1d.0.electrons.density"] = np.full(rho.size, 1.0e19)
    ods["core_profiles.profiles_1d.0.electrons.temperature"] = np.full(rho.size, 100.0)

    voltage = (
        np.asarray([0.3]),
        np.asarray([1.0]),
        np.asarray([0.25]),
        np.asarray([0.75]),
    )
    with (
        patch("vaft.omas.formula_wrapper.compute_voltage_consumption", return_value=voltage),
        patch(
            "vaft.omas.process_wrapper.compute_ohmic_heating_power_from_core_profiles",
            return_value=100.0,
        ),
        patch("vaft.omas.update.update_equilibrium_global_quantities_volume"),
        patch(
            "vaft.omas.formula_wrapper.compute_volume_averaged_pressure",
            return_value=np.asarray([150.0]),
        ),
        patch(
            "vaft.omas.formula_wrapper._compute_bremsstrahlung_power_series",
            return_value=np.asarray([0.0]),
        ),
    ):
        result = compute_power_balance(ods, include_line_radiation=False)

    np.testing.assert_allclose(result["time"], [0.3])
    assert np.isfinite(result["P_sync"][0])
    assert result["P_sync"][0] >= 0.0
