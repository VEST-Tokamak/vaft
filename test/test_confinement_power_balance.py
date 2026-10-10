"""Power balance and slice qualification for confinement scaling (#548, Lane D)."""

import numpy as np
import pytest

from vaft.formula.constants import MU0
from vaft.process.confinement import (
    confinement_exclusion_table,
    confinement_power_balance,
    ohmic_confinement_series,
    confinement_slice_decision,
    confinement_slice_evidence,
    resistive_loop_voltage,
    smoothed_time_derivative,
)


# --- derivative ---------------------------------------------------------------


def test_local_polynomial_derivative_is_exact_for_a_quadratic_on_a_gappy_grid():
    t = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 12.0]) * 1e-3
    y = 3.0 + 40.0 * t - 2.0e4 * t**2
    got = smoothed_time_derivative(t, y, window_s=5e-3, polyorder=2)
    np.testing.assert_allclose(got[:-1], (40.0 - 4.0e4 * t)[:-1], rtol=0, atol=1e-8)
    # 12 ms has no neighbour within 2.5 ms: one sample cannot carry a quadratic.
    assert np.isnan(got[-1])


def test_local_polynomial_derivative_evaluates_between_samples():
    t = np.linspace(0.0, 0.01, 251)
    y = np.sin(300.0 * t)
    at = np.array([0.0031, 0.0057])
    got = smoothed_time_derivative(t, y, window_s=1e-3, polyorder=2, at=at)
    np.testing.assert_allclose(got, 300.0 * np.cos(300.0 * at), rtol=5e-3)


def test_smoothing_reduces_noise_on_the_derivative():
    rng = np.random.default_rng(548)
    t = np.linspace(0.0, 0.02, 501)
    y = 100.0 * t + rng.normal(0.0, 1e-3, t.size)
    raw = smoothed_time_derivative(t, y)
    smooth = smoothed_time_derivative(t, y, window_s=2e-3)
    assert np.std(smooth[20:-20] - 100.0) < 0.2 * np.std(raw[20:-20] - 100.0)


def test_a_window_with_too_few_samples_gives_nan_not_a_guess():
    t = np.array([0.0, 1.0, 10.0, 11.0])
    got = smoothed_time_derivative(t, t**2, window_s=2.5, polyorder=2)
    assert np.all(np.isnan(got))


@pytest.mark.parametrize(
    "kwargs", [dict(window_s=0.0), dict(at=np.array([0.5])), dict(window_s=2.0, polyorder=0)]
)
def test_derivative_rejects_bad_requests(kwargs):
    with pytest.raises(ValueError):
        smoothed_time_derivative(np.arange(5.0), np.arange(5.0), **kwargs)


def test_derivative_rejects_a_non_increasing_axis():
    with pytest.raises(ValueError, match="increasing"):
        smoothed_time_derivative(np.array([0.0, 2.0, 1.0]), np.zeros(3))


# --- resistive voltage --------------------------------------------------------


def test_resistive_voltage_removes_exactly_the_internal_inductive_voltage():
    """Synthetic ramp: W_int = L_i I^2 / 2 with li_3 and I_p both changing."""
    r0 = 0.4
    t = np.linspace(0.0, 0.01, 401)
    ip = 1e5 * (1.0 - 30.0 * t)
    li = 1.0 + 20.0 * t
    v_res_true = 2.0 + 0.0 * t
    w_int = 0.25 * MU0 * r0 * li * ip**2
    v_loop = v_res_true + np.gradient(w_int, t) / ip

    split = resistive_loop_voltage(
        v_loop, ip, np.gradient(ip, t), li, r0, dli3_dt=np.gradient(li, t)
    )
    np.testing.assert_allclose(split.v_res[1:-1], v_res_true[1:-1], atol=1e-6)
    np.testing.assert_allclose(split.r_p, split.v_res / ip)
    np.testing.assert_allclose(split.l_int, 0.5 * MU0 * r0 * li)
    # Holding li_3 fixed drops the 1/4 mu0 R0 I dli/dt term and nothing else.
    fixed = resistive_loop_voltage(v_loop, ip, np.gradient(ip, t), li, r0)
    np.testing.assert_allclose(
        fixed.v_res - split.v_res, 0.25 * MU0 * r0 * ip * np.gradient(li, t), rtol=1e-9
    )


def test_resistance_is_undefined_without_current():
    split = resistive_loop_voltage(np.ones(2), np.array([0.0, 1e5]), np.zeros(2), np.ones(2), 0.4)
    assert np.isnan(split.r_p[0]) and split.r_p[1] == pytest.approx(1e-5)


def test_resistive_voltage_rejects_a_non_positive_radius():
    with pytest.raises(ValueError, match="r0"):
        resistive_loop_voltage(np.ones(2), np.ones(2), np.ones(2), np.ones(2), 0.0)


# --- power balance ------------------------------------------------------------


def _ramp():
    t = np.linspace(0.30, 0.32, 21)  # 1 ms EFIT-like grid
    ip = 8e4 - 2e6 * (t - 0.30)
    v_res = 3.0 * np.ones_like(t)
    w = 120.0 + 2.0e3 * (t - 0.30)  # dW/dt = 2 kW
    return t, ip, v_res, w


def test_power_balance_closes_on_an_analytic_ramp():
    t, ip, v_res, w = _ramp()
    pb = confinement_power_balance(t, ip, v_res, w, dwdt_window_s=5e-3)
    np.testing.assert_allclose(pb.p_ohmic, ip * v_res)
    np.testing.assert_allclose(pb.dwdt, 2.0e3, rtol=1e-9)
    np.testing.assert_allclose(pb.p_net, ip * v_res - 2.0e3, rtol=1e-12)
    np.testing.assert_allclose(pb.tau_e_net, w / (ip * v_res - 2.0e3), rtol=1e-12)


def test_unmeasured_radiation_leaves_the_transport_loss_undefined():
    t, ip, v_res, w = _ramp()
    pb = confinement_power_balance(t, ip, v_res, w)
    assert np.all(np.isnan(pb.p_transport)) and np.all(np.isnan(pb.tau_e_transport))
    p_rad = np.full_like(t, 1.0e4)
    p_rad[3] = np.nan
    pb = confinement_power_balance(t, ip, v_res, w, p_rad=p_rad)
    np.testing.assert_allclose(np.delete(pb.p_transport, 3), np.delete(pb.p_net - 1.0e4, 3))
    assert np.isnan(pb.p_transport[3])


def test_a_non_positive_loss_power_has_no_confinement_time():
    t, ip, v_res, w = _ramp()
    v_res = v_res.copy()
    v_res[5] = -1.0
    pb = confinement_power_balance(t, ip, v_res, w)
    assert pb.p_net[5] < 0 and np.isnan(pb.tau_e_net[5])


def test_power_balance_rejects_mismatched_lengths():
    t, ip, v_res, w = _ramp()
    with pytest.raises(ValueError, match="w_th"):
        confinement_power_balance(t, ip, v_res, w[:-1])


# --- slice qualification ------------------------------------------------------


def _evidence():
    ip = np.array([8e4, 8e4, 2e4, 8e4, 8e4])
    dip = np.array([0.0, -4e6, 0.0, 0.0, 0.0])
    w = np.array([100.0, 100.0, 100.0, 100.0, np.nan])
    p = np.array([2e5, 2e5, 2e5, 2e5, 2e5])
    dw = np.array([1e4, 1e4, 1e4, 1.5e5, 1e4])
    tau = np.array([5e-4, 5e-4, 5e-4, 5e-4, np.nan])
    return confinement_slice_evidence(ip, dip, w, p, dw, tau)


def test_evidence_measures_rates_against_the_confinement_time():
    ev = _evidence()
    np.testing.assert_allclose(ev.ip_rate[1], 50.0)
    np.testing.assert_allclose(ev.ip_change_per_tau[1], 50.0 * 5e-4)
    np.testing.assert_allclose(ev.dwdt_fraction[3], 0.75)
    assert ev.finite.tolist() == [True, True, True, True, False]


def test_decision_applies_each_rule_and_their_conjunction():
    rules = confinement_slice_decision(
        _evidence(), ip_min=3e4, max_dwdt_fraction=0.5, max_ip_change_per_tau=0.01
    )
    assert rules["ip_min"].tolist() == [True, True, False, True, True]
    assert rules["dwdt_fraction"].tolist() == [True, True, True, False, True]
    assert rules["ip_change_per_tau"].tolist() == [True, False, True, True, False]
    assert rules["accepted"].tolist() == [True, False, False, False, False]


def test_exclusion_table_counts_alone_and_in_sequence():
    rules = confinement_slice_decision(
        _evidence(), ip_min=3e4, max_dwdt_fraction=0.5, max_ip_change_per_tau=0.01
    )
    table = {row["rule"]: row for row in confinement_exclusion_table(rules)}
    assert table["finite"]["failed"] == 1 and table["finite"]["remaining"] == 4
    assert table["ip_min"]["failed_only_this"] == 1
    # Slice 4 fails both "finite" and "ip_change_per_tau" (NaN tau): removed once.
    assert table["ip_change_per_tau"]["failed"] == 2
    assert table["ip_change_per_tau"]["removed_in_sequence"] == 1
    assert table["ip_change_per_tau"]["remaining"] == 1
    assert "accepted" not in table


def test_exclusion_table_rejects_rules_of_different_length():
    with pytest.raises(ValueError):
        confinement_exclusion_table({"a": np.ones(2, bool), "b": np.ones(3, bool)})


# --- the Tier A table builder (workflow/confinement_scaling) ------------------


def _builder():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling/build_table.py"
    spec = importlib.util.spec_from_file_location("lane_d_build_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_matches_slices_by_time_not_position():
    build = _builder()
    times = np.array([0.323, 0.321, 0.322])  # deliberately out of order
    assert build._match(times, 0.3220) == 2
    assert build._match(times, 0.32204) == 2  # inside the 5e-5 s tolerance
    assert build._match(times, 0.3221) is None  # a neighbour never stands in
    assert build._match(np.array([]), 0.322) is None  # an empty time base is no match


def test_builder_extension_columns_do_not_shadow_the_canonical_schema():
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    build = _builder()
    assert not set(build.EXTENSION_UNITS) & set(CONFINEMENT_COLUMNS)


def test_builder_keeps_p_ohm_and_r_p_for_a_single_labelled_slice():
    """One labelled slice has no dli_3/dt or dW/dt, but P_OH = I_p V_res and R_p need no rate.

    Before the fix the all-NaN dli_3/dt array was passed on, so V_ind, V_res,
    P_OH and R_p were NaN as well (packaged 40324 row, cold review 0.8.0
    delta-absorb-13 F1); only the rate-dependent columns may be NaN.
    """
    build = _builder()
    t, ip, dip, v_loop, li3, r0, w = (np.array([0.312]), np.array([60e3]), np.array([1.5e6]),
                                     np.array([2.5]), np.array([0.8]), 0.4, np.array([800.0]))
    split, pb, ev, orientation = build.power_balance_series(
        t, ip, dip, v_loop, li3, r0, w, dwdt_window_s=3e-3, rate_polyorder=1)
    expected_v_ind = 0.5 * MU0 * r0 * li3 * dip  # l_i held fixed: no dli_3/dt term
    assert split.v_ind == pytest.approx(expected_v_ind)
    assert np.isfinite(split.v_res).all() and np.isfinite(split.r_p).all()
    assert pb.p_ohmic == pytest.approx(ip * (v_loop - expected_v_ind))
    assert np.isnan(pb.dwdt).all() and np.isnan(pb.p_net).all() and np.isnan(pb.tau_e_net).all()
    assert orientation == 1.0
    assert ev.ip_abs.shape == (1,)

    # Two labelled slices further apart than the 3 ms window hold no rate either
    # (4 of the 16 two-slice rows of the packaged table): P_OH stays, dW/dt is NaN.
    args2 = (np.array([60e3, 62e3]), np.array([1.5e6, 1.4e6]), np.array([2.5, 2.4]),
             np.array([0.8, 0.9]), r0, np.array([800.0, 900.0]))
    split2, pb2, _, _ = build.power_balance_series(np.array([0.310, 0.312]), *args2,
                                                   dwdt_window_s=3e-3, rate_polyorder=1)
    np.testing.assert_allclose(split2.v_ind, 0.5 * MU0 * r0 * args2[3] * args2[1])
    assert np.isfinite(pb2.p_ohmic).all() and np.isnan(pb2.dwdt).all() and np.isnan(pb2.tau_e_net).all()

    # Two labelled slices inside the window take the rate path and keep the dli_3/dt term.
    split3, pb3, _, _ = build.power_balance_series(np.array([0.310, 0.311]), *args2,
                                                   dwdt_window_s=3e-3, rate_polyorder=1)
    assert np.isfinite(pb3.dwdt).all() and np.isfinite(pb3.p_net).all()
    np.testing.assert_allclose(split3.v_ind - 0.5 * MU0 * r0 * args2[3] * args2[1],
                               0.25 * MU0 * r0 * args2[0] * 100.0)  # dli_3/dt = 0.1 per 1 ms


def _lane_d_module(name):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lane_d_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_decision_columns_use_the_primary_selection():
    """accepted/selected in table.csv are the #1490 primary selection, the one every result quotes.

    The builder used to apply the sensitivity thresholds (0.5 / 0.05) while the
    README, fit.py and the notebooks call 0.20 / 1.0 the primary selection
    (cold review 0.8.0 delta-absorb-13 F2). The README's numbers and the
    packaged table's `selected` column must match the code.
    """
    import pathlib
    import re

    build = _builder()
    fit = _lane_d_module("fit")
    assert build.PRIMARY_THRESHOLDS == fit.SELECTIONS["primary"]
    assert build.PRIMARY_THRESHOLDS != fit.SELECTIONS["sensitivity"]

    readme = (pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling/README.md").read_text(
        encoding="utf-8")
    pattern = r"^\| (primary|sensitivity) \| ≤ ([\d.]+) \| ≤ ([\d.]+) \| ≥ (\d+) kA \|$"
    rows = {m[0]: m[1:] for m in re.findall(pattern, readme, re.M)}
    for name, (ip_change, dwdt, ip_ka) in rows.items():
        sel = fit.SELECTIONS[name]
        assert float(ip_change) == sel["max_ip_change_per_tau"]
        assert float(dwdt) == sel["max_dwdt_fraction"]
        assert float(ip_ka) * 1e3 == sel["ip_min"]
    assert set(rows) == {"primary", "sensitivity"}
    assert "provisional" not in readme.split("## Slice quality")[1].split("## Run")[0]

    import pandas as pd
    from vaft.data import data_path

    table = pd.read_csv(data_path("confinement/vest_tier_a_confinement.csv"))
    primary = fit.select(table, fit.SELECTIONS["primary"])
    np.testing.assert_array_equal(table["selected"].to_numpy(bool), primary)
    np.testing.assert_array_equal(table["accepted"].to_numpy(bool), primary)
    assert int(primary.sum()) == 59 and table.loc[primary, "shot"].nunique() == 19  # the quoted primary set


def test_lane_d_text_io_never_uses_the_locale_encoding():
    """Repository files are read and written as UTF-8 (cold review 0.8.0 delta-absorb-13 F5)."""
    import json
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = {p: p.read_text(encoding="utf-8") for p in (root / "workflow/confinement_scaling").glob("*.py")}
    for name in ("confinement_time_scaling", "multi_machine_confinement_database", "tokamak_power_balance"):
        nb = json.loads((root / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        sources[root / "notebooks" / f"{name}.ipynb"] = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    text_io = re.compile(r"(?<![\w.])open\(|\.read_text\(|\.write_text\(")
    offenders = []
    for path, text in sources.items():
        for number, line in enumerate(text.splitlines(), 1):
            if text_io.search(line) and "encoding=" not in line and '"rb"' not in line and "gzip" not in line:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


# --- #1905: the shared Ohmic confinement contract -------------------------------------------------------------

def test_inductive_voltage_splits_into_current_and_profile_terms():
    t = np.linspace(0.30, 0.31, 11)
    ip, dip = 1e5 + 2e6 * (t - 0.30), np.full(t.size, 2e6)
    li, dli = 0.6 + 5.0 * (t - 0.30), np.full(t.size, 5.0)
    r0 = 0.4
    split = resistive_loop_voltage(np.full(t.size, 2.0), ip, dip, li, r0, dli3_dt=dli)
    np.testing.assert_allclose(split.v_ind_current, 0.5 * MU0 * r0 * li * dip)
    np.testing.assert_allclose(split.v_ind_profile, 0.25 * MU0 * r0 * ip * dli)
    np.testing.assert_allclose(split.v_ind, split.v_ind_current + split.v_ind_profile)
    assert not split.li3_rate_fallback.any()


def test_missing_li3_rate_is_held_only_when_asked_and_always_flagged():
    args = (np.full(3, 2.0), np.full(3, 1e5), np.full(3, 1e6), np.full(3, 0.6), 0.4)
    rate = np.array([1.0, np.nan, 1.0])
    held = resistive_loop_voltage(*args, dli3_dt=rate, hold_li3_where_missing=True)
    assert held.li3_rate_fallback.tolist() == [False, True, False]
    assert np.isfinite(held.v_res).all() and held.v_ind_profile[1] == 0.0
    kept = resistive_loop_voltage(*args, dli3_dt=rate)  # default: a NaN rate is not hidden
    assert np.isnan(kept.v_res[1]) and not kept.li3_rate_fallback.any()
    none = resistive_loop_voltage(*args)  # no rate at all: the profile term is absent, exactly zero
    assert none.li3_rate_fallback.all() and (none.v_ind_profile == 0.0).all()


@pytest.mark.parametrize("shot, time_s, tau_ms", [(39917, 0.321, 2.33143), (42962, 0.333, 2.26342)])
def test_reference_series_reproduce_from_their_inputs(shot, time_s, tau_ms):
    """#1905 Sec. 8: the frozen Tier A inputs give back every derived column of the atlas build."""
    import pathlib

    import pandas as pd

    path = pathlib.Path(__file__).resolve().parents[1] / f"vaft/data/confinement/vest_{shot}_power_balance_series.csv"
    s = pd.read_csv(path)
    chain = ohmic_confinement_series(
        s.time_s.to_numpy(), s.ip_meas_A.to_numpy(), s.dip_dt_A_s.to_numpy(), s.v_loop_V.to_numpy(),
        s.li_3.to_numpy(), 0.4, s.w_mhd_J.to_numpy(), rate_window_s=3e-3, rate_polyorder=1)
    for column, value in (("v_ind_V", chain.v_ind), ("v_res_V", chain.v_res), ("dwdt_W", chain.dwdt),
                          ("p_ohm_W", chain.p_ohmic), ("p_net_W", chain.p_net), ("tau_e_net_s", chain.tau_e_net)):
        np.testing.assert_allclose(value, s[column].to_numpy(), rtol=1e-9, atol=1e-12, err_msg=column)
    k = int(np.argmin(abs(chain.time - time_s)))
    assert chain.tau_e_net[k] * 1e3 == pytest.approx(tau_ms, rel=1e-5)
    np.testing.assert_allclose(chain.v_ind_current + chain.v_ind_profile, chain.v_ind)
    assert chain.r0 == 0.4 and chain.orientation == 1.0


def test_no_energy_rate_is_never_replaced_by_zero():
    """An isolated slice keeps P_OH (l_i held, flagged) but has no dW/dt, P_net or tau_E, with a reason."""
    t = np.array([0.310, 0.311, 0.320])  # the last slice is 9 ms from the others: outside a 3 ms window
    n = t.size
    chain = ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.full(n, 1.5), np.array([0.6, 0.62, 0.7]),
                                     0.4, np.array([200.0, 210.0, 230.0]), rate_window_s=3e-3)
    assert np.isfinite(chain.dwdt[:2]).all() and np.isnan(chain.dwdt[2])
    assert np.isfinite(chain.p_ohmic).all() and chain.li3_rate_fallback.tolist() == [False, False, True]
    assert np.isnan(chain.p_net[2]) and np.isnan(chain.tau_e_net[2])
    assert chain.missing_reason[2] == "no energy rate in the window" and chain.missing_reason[0] == ""
    single = ohmic_confinement_series(t[:1], [8e4], [0.0], [1.5], [0.6], 0.4, [200.0], rate_window_s=3e-3)
    assert np.isfinite(single.p_ohmic).all() and np.isnan(single.dwdt).all() and np.isnan(single.tau_e_net).all()


def test_ohmic_chain_records_its_settings_and_named_approximations():
    t = np.linspace(0.30, 0.31, 6)
    n = t.size
    chain = ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.full(n, 1.5), np.full(n, 0.6), 0.4,
                                     np.full(n, 200.0), rate_window_s=3e-3,
                                     assumptions={"v_loop": "measured FL10", "w": "magnetics EFIT 1.5 int p dV"})
    assert "inferred, not measured" in chain.assumptions["v_res"]
    assert chain.assumptions["v_loop"] == "measured FL10" and chain.assumptions["w"].startswith("magnetics")
    assert np.isnan(chain.p_transport).all() and "unmeasured" in chain.assumptions["p_rad"]
    assert chain.settings["rate_window_s"] == 3e-3 and chain.settings["rate_polyorder"] == 1
    assert "constant-inductance" in chain.settings["li3_rate_fallback"]
    assert chain.settings["dwdt_fallback"].startswith("none")
    np.testing.assert_allclose(chain.power_balance().tau_e_net, chain.tau_e_net)


def test_ohmic_chain_flags_reversed_orientation_and_non_positive_power():
    t = np.linspace(0.30, 0.31, 6)
    n = t.size
    chain = ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.full(n, -1.5), np.full(n, 0.6), 0.4,
                                     np.full(n, 200.0), rate_window_s=5e-3)  # 2 ms spacing: 5 ms holds a rate
    assert chain.orientation == -1.0 and (chain.p_ohmic < 0).all()
    assert np.isnan(chain.tau_e_net).all() and set(chain.missing_reason) == {"P_net not positive"}
    with pytest.raises(ValueError):
        ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.ones(n), np.ones(n), 0.4, np.ones(n),
                                 rate_window_s=0.0)
    with pytest.raises(ValueError):
        ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.ones(n), np.ones(n), 0.0, np.ones(n),
                                 rate_window_s=3e-3)
    with pytest.raises(ValueError):  # validated even where no rate is taken
        ohmic_confinement_series(t[:1], [8e4], [0.0], [1.0], [1.0], 0.4, [1.0], rate_window_s=3e-3, rate_polyorder=0)
    with pytest.raises(ValueError):
        ohmic_confinement_series(t[::-1], np.full(n, 8e4), np.zeros(n), np.ones(n), np.ones(n), 0.4, np.ones(n),
                                 rate_window_s=3e-3)


def test_noisy_irregular_energy_rate_is_recovered_by_the_local_fit():
    rng = np.random.default_rng(3)
    t = np.sort(0.30 + rng.uniform(0, 0.02, 40))
    w = 200.0 + 4e3 * (t - 0.30) + rng.normal(0, 0.05, t.size)  # dW/dt = 4 kW
    n = t.size
    chain = ohmic_confinement_series(t, np.full(n, 8e4), np.zeros(n), np.full(n, 1.5), np.full(n, 0.6), 0.4, w,
                                     rate_window_s=6e-3)
    assert np.nanmedian(chain.dwdt) == pytest.approx(4e3, rel=0.05)
    np.testing.assert_allclose(chain.p_net, chain.p_ohmic - chain.dwdt)
