"""Lane D extensions (workflow/confinement_scaling/extensions.py): the decompositions on synthetic data."""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling"


def _ext():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("lane_d_extensions", HERE / "extensions.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def _frame(seed=0):
    """Shots with a between-shot exponent 2 on I and a within-shot exponent 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for shot in range(12):
        level = np.exp(rng.normal(np.log(1e5), 0.3))
        for k in range(4):
            ip = level * np.exp(rng.normal(0, 0.1))
            tau = 1e-3 * (level / 1e5) ** 2.0 * (ip / level) ** 1.0
            rows.append({"shot": shot, "i_p_A": ip, "tau_e_th_s": tau})
    return pd.DataFrame(rows)


def test_within_and_between_separate_the_two_exponents():
    from vaft.process.confinement import fit_confinement_scaling

    ext = _ext()
    f = _frame()
    cols = {"tau": "tau_e_th_s", "i_p": "i_p_A"}
    y, x, g = ext.within_shot(f, cols)
    assert fit_confinement_scaling(y, x, g).exponents()["i_p"] == pytest.approx(1.0, abs=1e-9)
    # Shot means of log(ip) differ from log(level) by the slice noise, so the
    # between estimate is near 2 but not exact.
    y, x, g = ext.between_shot(f, cols)
    assert fit_confinement_scaling(y, x, g).exponents()["i_p"] == pytest.approx(2.0, abs=0.15)


def test_within_drops_single_slice_shots():
    ext = _ext()
    f = pd.concat([_frame(), pd.DataFrame([{"shot": 99, "i_p_A": 1e5, "tau_e_th_s": 1e-3}])])
    _, _, g = ext.within_shot(f, {"tau": "tau_e_th_s", "i_p": "i_p_A"})
    assert 99 not in set(g)


def test_dimensionless_columns_recover_the_average_temperature():
    ext = _ext()
    n, t_ev, a, r, ka = 1e19, 100.0, 0.25, 0.4, 1.5
    volume = 2 * np.pi**2 * a**2 * r * ka
    row = {"n_e_line_avg_m3": n, "w_th_J": 3 * n * t_ev * 1.602176634e-19 * volume, "a_m": a, "r_geo_m": r,
           "kappa_area": ka, "epsilon": a / r, "m_eff_amu": 1.0, "b_t_T": 0.2, "i_p_A": 1e5,
           "tau_e_th_s": 1e-3, "shot": 1, "time_s": 0.3}
    out = ext.dimensionless_columns(pd.DataFrame([row, {**row, "n_e_line_avg_m3": np.nan}]))
    assert len(out) == 1 and out["t_avg_eV"].iloc[0] == pytest.approx(t_ev)
    assert np.isfinite(out[["rho_star", "beta_t", "nu_star", "q_cyl", "omega_tau"]].to_numpy()).all()


def test_incomplete_rows_do_not_enter_any_shot_mean():
    """A NaN in one column must remove the whole row, not just that column's mean."""
    ext = _ext()
    f = pd.DataFrame({"shot": [1, 1, 2, 2, 2], "tau_e_th_s": [1.0, 2.0, 1.0, 2.0, 4.0],
                      "p_loss_W": [1.0, np.nan, 1.0, 2.0, 4.0]})
    cols = {"tau": "tau_e_th_s", "p": "p_loss_W"}
    y, x, g = ext.within_shot(f, cols)
    assert list(g) == [2, 2, 2]  # shot 1 has one complete row left: no within information
    np.testing.assert_allclose(np.log(y), np.log(x["p"]))
    y, x, g = ext.between_shot(f, cols)
    np.testing.assert_allclose(np.log(y), np.log(x["p"]))  # shot 1 mean from its one complete row
