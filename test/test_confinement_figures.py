"""Lane D figure functions (workflow/confinement_scaling/figures.py) on synthetic inputs."""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

HERE = pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling"


def _figures():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("lane_d_figures", HERE / "figures.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def _closures():
    rows = [{"data": "primary:A", "model": "free", "mu_rho_imposed": np.nan, "a_i_p": 1.95, "a_b_t": 0.61,
             "a_p_net": -0.90, "se_i_p": 0.3, "se_b_t": 0.31, "se_p_net": 0.11,
             "mu_rho_completed": -13.0, "mu_rho_completed_se": 17.0, "one_plus_aP_over_se": 0.87}]
    for name, mu in (("bohm", -2.0), ("gyro_bohm", -3.0)):
        rows.append({"data": "primary:A", "model": name, "mu_rho_imposed": mu, "a_i_p": 1.6, "a_b_t": 0.1,
                     "a_p_net": -0.78, "se_i_p": 0.2, "se_b_t": 0.1, "se_p_net": 0.05})
    rows.append({"data": "primary:B_with_n_e", "model": "free", "mu_rho_imposed": np.nan, "a_i_p": 2.0,
                 "a_b_t": 0.2, "a_p_net": -1.0})
    nstx = pd.DataFrame([{"scaling": "NSTX2006L (Kaye 2006)", "a_i_p": 1.01, "a_b_t": 0.7, "a_p_net": -0.37,
                          "mu_rho_completed": -4.03},
                         {"scaling": "VEST primary:A", "a_i_p": 1.95, "a_b_t": 0.61, "a_p_net": -0.9,
                          "mu_rho_completed": -13.0}])
    return pd.DataFrame(rows), nstx


def test_exponent_table_keeps_one_data_set_and_flags_the_undetermined_index():
    figures = _figures()
    closures, nstx = _closures()
    table = figures.exponent_table(closures, nstx)
    assert list(table["kind"]) == ["vest_free", "vest_closure", "vest_closure", "reference", "reference"]
    assert table.loc[0, "mu_rho_undetermined"]
    assert table.loc[1, "mu_rho"] == -2.0  # an imposed closure reports its imposed value
    assert "VEST primary:A" not in set(table["label"])  # VEST rows of the NSTX file are not references


def test_exponent_figure_draws_without_error():
    figures = _figures()
    fig, axes = figures.exponent_figure(figures.exponent_table(*_closures()))
    assert len(axes) == 2 and "ASSUMED" in axes[1].get_xlabel()
    matplotlib.pyplot.close(fig)


def test_population_table_labels_vest_and_spherical_tokamaks():
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    figures = _figures()
    base = {c: np.nan for c in CONFINEMENT_COLUMNS}
    db5 = pd.DataFrame([{**base, "machine": m, "selected": sel, "tau_e_th_s": 0.05, "i_p_A": 1e6}
                        for m, sel in (("NSTX", True), ("JET", True), ("JET", False))])
    vest = pd.DataFrame([{**base, "machine": "VEST", "shot": 1, "time_s": 0.32, "tau_e_th_s": 1e-3,
                          "i_p_A": 8e4, "quality_status": "evaluated", "ip_rate_1_s": 1.0,
                          "ip_change_per_tau": 0.01, "dwdt_fraction": 0.1, "rule_finite": True},
                         {**base, "machine": "VEST", "shot": 1, "time_s": 0.33, "tau_e_th_s": 1e-3,
                          "i_p_A": 8e4, "quality_status": "evaluated", "ip_rate_1_s": 1.0,
                          "ip_change_per_tau": 0.5, "dwdt_fraction": 0.1, "rule_finite": True}])
    table = figures.population_table(db5, vest)
    assert list(table["population"]) == ["NSTX (DB5)", "conventional tokamaks (DB5)", figures.VEST_LABEL]
