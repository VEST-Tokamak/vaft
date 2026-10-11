import math

import pytest

from vaft.formula import confinement_time_from_engineering_parameters
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


def _sample_with_elongation():
    """Sample 39915 with a power balance and the elongation the sample lacks."""
    pytest.importorskip("omas")
    from _synthetic_inputs import make_power_balance

    from vaft.omas.sample import sample_ods

    ods = make_power_balance(sample_ods(39915))
    index = len(ods["equilibrium.time_slice"]) // 2
    ods["equilibrium.time_slice"][index]["boundary.elongation"] = 1.6
    return ods, index


def test_itpa20_on_the_ods_path_reads_the_boundary_triangularity(caplog):
    from vaft.omas import formula_wrapper as fw

    ods, index = _sample_with_elongation()
    boundary = ods["equilibrium.time_slice"][index]["boundary"]
    assert "triangularity" not in boundary and "triangularity_upper" not in boundary
    # Without a triangularity: NaN, and the warning names what is missing.
    with caplog.at_level("WARNING", logger=fw.logger.name):
        assert math.isnan(fw.compute_tau_E_scaling(ods, index, scaling="ITPA20"))
    assert "triangularity" in caplog.text
    assert math.isfinite(fw.compute_tau_E_scaling(ods, index, scaling="H98y2"))

    # Upper and lower present: their mean is passed as delta.
    ods["equilibrium.time_slice"][index]["boundary.triangularity_upper"] = 0.4
    ods["equilibrium.time_slice"][index]["boundary.triangularity_lower"] = 0.2
    params = fw.compute_tau_E_engineering_parameters(ods, index)
    assert params["delta"] == pytest.approx(0.3)
    expected = confinement_time_from_engineering_parameters(
        I_p=params["I_p"], B_t=params["B_t"], P_loss=params["P_loss"], n_e=params["n_e_line_avg"],
        M=params["M"], R=params["R"], epsilon=params["epsilon"], kappa=params["kappa"],
        delta=0.3, scaling="ITPA20",
    )
    assert fw.compute_tau_E_scaling(ods, index, scaling="ITPA20") == pytest.approx(expected, rel=1e-12)


def test_boundary_triangularity_falls_back_to_the_single_value():
    pytest.importorskip("omas")
    from omas import ODS

    from vaft.omas.formula_wrapper import _boundary_triangularity

    ods = ODS()
    ods["equilibrium.time_slice.0.boundary.triangularity"] = 0.25
    assert _boundary_triangularity(ods["equilibrium.time_slice"][0]) == pytest.approx(0.25)
    ods["equilibrium.time_slice.0.boundary.triangularity_upper"] = 0.5
    # Only one of upper/lower: the single value still decides.
    assert _boundary_triangularity(ods["equilibrium.time_slice"][0]) == pytest.approx(0.25)
    ods["equilibrium.time_slice.0.boundary.triangularity_lower"] = 0.1
    assert _boundary_triangularity(ods["equilibrium.time_slice"][0]) == pytest.approx(0.3)
    empty = ODS()
    empty["equilibrium.time_slice.0.boundary.minor_radius"] = 0.3
    assert _boundary_triangularity(empty["equilibrium.time_slice"][0]) is None
