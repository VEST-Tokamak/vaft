"""Lane D extensions (workflow/confinement_scaling/extensions.py): the decompositions on synthetic data."""

import importlib.util
import io
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


def test_campaign_diagnostics_reports_per_block_ratios():
    ext = _ext()
    f = pd.DataFrame({"shot": [40000, 40000, 42900, 42900], "time_s": [0.32] * 4,
                      "w_th_J": [200.0, 220.0, 600.0, 660.0], "w_e_ts_J": [100.0, 110.0, 150.0, np.nan],
                      "p_ohm_W": [1e5] * 4, "p_ohm_efit_flux_W": [1e5] * 4, "i_p_A": [8e4] * 4,
                      "n_e_line_avg_m3": [5e18] * 4, "b0r0_Tm": [0.06] * 4})
    state = pd.DataFrame({"shot": [40000, 42900, 42900], "time_efit_s": [0.32, 0.32, 0.33],
                          "efit_lineage": ["magnetics"] * 3, "r_sum": [2.0, 3.0, 99.0],
                          "probe_reduced_chi2": [1.0, 5.0, 99.0], "loop_reduced_chi2": [1.0] * 3,
                          "betap": [0.2] * 3})
    out = ext.campaign_diagnostics(f, state).set_index("quantity")
    # The 0.33 s state row is not one of the frame's slices, so it must not enter.
    assert out.loc["lane_k_r_sum_p_efit_over_p_e", "429xx-430xx"] == pytest.approx(3.0)
    assert out.loc["w_mhd_over_w_e_ts", "399xx-403xx"] == pytest.approx(2.0)
    assert out.loc["w_mhd_over_w_e_ts", "429xx-430xx"] == pytest.approx(4.0)  # NaN W_e row skipped
    assert list(ext.block_indicator(f)) == [1.0, 1.0, np.e, np.e]


def test_block_offset_by_thomson_consistency_splits_and_counts():
    """Synthetic: W ~ I_p n_e, the 429xx block offset sits only in the inconsistent rows."""
    ext = _ext()
    rng = np.random.default_rng(1521)
    rows = []
    for shot in list(range(39900, 39910)) + list(range(42900, 42910)):
        for k in range(4):
            ip, v, n = rng.uniform(5e4, 1.2e5), rng.uniform(1, 3), rng.uniform(4e18, 1.5e19)
            consistent = bool(rng.random() < 0.5)
            bias = 1.5 if (shot >= 42000 and not consistent) else 1.0
            w = 1e-3 * ip * (n / 1e19) ** 0.4 * bias * np.exp(rng.normal(0, 0.02))
            rows.append({"shot": shot, "i_p_A": ip, "v_loop_V": v, "n_e_line_avg_m3": n, "w_th_J": w,
                         "w_e_ts_J": w / 2.0, "thomson_consistent": consistent,
                         "efit_quality": "good" if consistent else "admissible"})
    offsets, census = ext.block_offset_by_thomson_consistency(pd.DataFrame(rows))
    get = lambda subset: offsets[(offsets.subset == subset) & (offsets.energy == "W_mhd")  # noqa: E731
                                 & offsets.with_density].iloc[0]
    assert abs(get("Thomson-consistent").block_offset_log) < 0.05
    assert get("Thomson-inconsistent").block_offset_log == pytest.approx(np.log(1.5), abs=0.05)
    assert set(census["block"]) == {"399xx-403xx", "429xx-430xx"}
    assert int(census["consistent"].sum() + census["inconsistent"].sum()) == len(rows)

    # A CSV round trip turns the verdict into strings, with NaN where Thomson is absent.
    frame = pd.DataFrame(rows)
    frame["thomson_consistent"] = frame["thomson_consistent"].astype(object)
    frame.loc[frame.index[:3], "thomson_consistent"] = np.nan
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    read_back = pd.read_csv(buffer)
    _, census_csv = ext.block_offset_by_thomson_consistency(read_back)
    assert int(census_csv["consistent"].sum() + census_csv["inconsistent"].sum()) == len(rows) - 3

    # Without the column every row is "no Thomson", and the split fits report errors instead of raising.
    offsets_none, census_none = ext.block_offset_by_thomson_consistency(frame.drop(columns="thomson_consistent"))
    assert int(census_none["consistent"].sum() + census_none["inconsistent"].sum()) == 0
    split = offsets_none[offsets_none.subset != "all Thomson rows"]
    assert split["error"].notna().all()

    # A single-block subset cannot carry a block offset: an error row, not an exception.
    single, _ = ext.block_offset_by_thomson_consistency(frame[frame.shot < 42000])
    assert single["error"].notna().all()


def _tglf_link():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("lane_d_tglf_link", HERE / "tglf_link.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def test_miller_volume_matches_a_circular_torus():
    link = _tglf_link()
    r = np.linspace(0.01, 0.3, 30)
    volume, dvdr = link.miller_volume(r, np.full_like(r, 0.4), np.ones_like(r), np.zeros_like(r))
    assert volume == pytest.approx(2 * np.pi**2 * 0.4 * r**2, rel=1e-4)
    assert dvdr[5:-5] == pytest.approx(4 * np.pi**2 * 0.4 * r[5:-5], rel=1e-3)
    # Elongation scales the volume linearly.
    volume_k, _ = link.miller_volume(r, np.full_like(r, 0.4), np.full_like(r, 1.8), np.zeros_like(r))
    assert volume_k == pytest.approx(1.8 * volume, rel=1e-6)


def test_power_link_converts_gyro_bohm_flux_to_watts_by_state_key():
    link = _tglf_link()
    key = {"shot": 39915, "time_efit_s": 0.317, "efit_lineage": "magnetics"}
    sens = pd.DataFrame([{**key, "r_over_a": ra, "tglf_config": cfg, "sat_rule": sat, "field_model": "es",
                          "qe_gb": 2.0, "qi_gb": 1.0} for ra in (0.4, 0.8) for cfg, sat in (("a", 0), ("b", 3))])
    transport = pd.DataFrame([{**key, "r_over_a": ra, "q_gb_W_m2": 1000.0, "qe_neo_W_m2": 10.0,
                               "qi_neo_W_m2": 0.0, "a_m": 0.3, "tglf_config": "b"} for ra in (0.4, 0.8)])
    # A second time on the same shot must not leak into the match (time, not index).
    confinement = pd.DataFrame([{"shot": 39915, "time_efit_s": 0.320, "p_ohm_W": 9e9, "p_net_W": 9e9,
                                 "efit_quality": "good", "thomson_consistent": True},
                                {"shot": 39915, "time_efit_s": 0.317, "p_ohm_W": 1.2e5, "p_net_W": 1e5,
                                 "efit_quality": "good", "thomson_consistent": True}])
    r = np.linspace(0.0, 0.3, 61)

    def geometry(shot, time_s, lineage):
        return r, np.full_like(r, 0.4), np.ones_like(r), np.zeros_like(r)

    out = link.power_link(sens, transport, confinement, geometry)
    assert len(out) == 4 and (out["p_net_W"] == 1e5).all()
    at = out[(out.r_over_a == 0.8)].iloc[0]
    vp = 4 * np.pi**2 * 0.4 * 0.24
    assert at["P_tglf_W"] == pytest.approx(3.0 * 1000.0 * vp, rel=2e-3)
    assert at["P_neo_W"] == pytest.approx(10.0 * vp, rel=2e-3)
    assert at["ratio_tglf_to_p_net"] == pytest.approx(at["P_tglf_W"] / 1e5)
    s = link.summary(out)
    assert len(s) == 1 and s["P_tglf_min_W"].iloc[0] == pytest.approx(s["P_tglf_max_W"].iloc[0])

    # An electron_kinetic state takes its magnetics twin's power; keys spelled a little
    # differently (float time, solved r/a) still join.
    kin = sens.assign(efit_lineage="electron_kinetic", time_efit_s=0.31700000000000006,
                      r_over_a=sens["r_over_a"] + 8e-5)
    both = link.power_link(pd.concat([sens, kin]), pd.concat([transport, transport.assign(efit_lineage="electron_kinetic")]),
                           confinement, geometry)
    assert len(both) == 8 and (both["p_net_W"] == 1e5).all()
    assert len(link.summary(both)) == 2

    # A state without a measured power is refused, not left NaN.
    with pytest.raises(ValueError, match="measured power"):
        link.power_link(sens, transport, confinement.iloc[:1], geometry)
    with pytest.raises(ValueError, match="delta"):
        link.miller_volume(r, r, r, None)

    # A profile whose edge disagrees with the atlas minor radius is refused.
    with pytest.raises(ValueError, match="a_m"):
        link.power_link(sens, transport.assign(a_m=0.25), confinement, geometry)
