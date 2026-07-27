import math

import pytest

from vaft.omas.formula_wrapper import compute_tau_E_scaling


ENGINEERING_PARAMETERS = {
    "I_p": 0.4e6,
    "B_t": 1.0,
    "P_loss": 0.5e6,
    "n_e": 2.0e19,
    "n_e_line_avg": 2.0e19,
    "n_e_vol_avg": 1.8e19,
    "M": 2.0,
    "R": 0.4,
    "epsilon": 0.6,
    "kappa": 1.5,
}


def test_compute_tau_e_scaling_requires_an_explicit_scaling_name():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'scaling'"):
        compute_tau_E_scaling(None, 0, eng_params=ENGINEERING_PARAMETERS)

    explicit_value = compute_tau_E_scaling(
        None,
        0,
        scaling="H98y2",
        eng_params=ENGINEERING_PARAMETERS,
    )

    assert explicit_value > 0.0
    assert math.isfinite(explicit_value)
