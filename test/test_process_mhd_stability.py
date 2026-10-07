"""DCON post-processing from the ODS payload (#940, sections 8 and 10).

Real DCON output (the #792 fixtures: shot 39915 at 319 ms, n=1, full edge and
peak-dW truncated) is mapped through ``mhd_linear`` and read back with
``extract_dcon_stability``, so everything here runs from the final ODS.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.machine_mapping.mhd_linear import extract_dcon_stability, mhd_linear
from vaft.process.mhd_stability import (
    criterion_intervals,
    dcon_edge_comparison,
    dcon_edge_scan,
    dcon_local_stability,
)

REFERENCE = Path(__file__).resolve().parent / "data" / "gpec" / "dcon_edge_792"


def _row(tmp_path: Path, treatment: str) -> dict:
    source = tmp_path / treatment
    shutil.copytree(REFERENCE / treatment, source)
    ods = ODS(consistency_check=False)
    mhd_linear(ods, str(source), {"module": "dcon", "modes": [1]})
    [row] = extract_dcon_stability(ods)
    return row


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dcon940")
    return {t: _row(tmp, t) for t in ("full_edge", "peak_dw_truncated")}


def test_intervals_and_crossings_on_a_known_profile():
    psi = np.linspace(0.0, 1.0, 11)
    values = np.array([-1, -1, 1, 1, -1, -1, -1, 2, 2, 2, -1], dtype=float)
    out = criterion_intervals(psi, values, unstable_side="positive")
    last = 0.9 + 0.1 * 2 / 3
    np.testing.assert_allclose(out["zero_crossings"], [0.15, 0.35, 0.6 + 0.1 / 3, last])
    np.testing.assert_allclose(out["unstable_intervals"], [[0.15, 0.35], [0.6 + 0.1 / 3, last]])
    assert out["unstable_fraction"] == pytest.approx(5 / 11)
    assert out["extremum"] == 2 and out["psi_n_at_extremum"] == pytest.approx(0.7)
    flipped = criterion_intervals(psi, -values, unstable_side="negative")
    np.testing.assert_allclose(flipped["unstable_intervals"], out["unstable_intervals"])


def test_unevaluated_samples_break_intervals_and_are_never_marginal():
    psi = np.linspace(0.0, 1.0, 5)
    values = np.array([1.0, 1.0, 0.0, 1.0, -1.0])
    mask = np.array([True, True, False, True, True])
    out = criterion_intervals(psi, values, unstable_side="positive", evaluated=mask)
    # The unevaluated zero at 0.5 is neither a crossing nor a gap filler.
    assert out["zero_crossings"] == [pytest.approx(0.875)]
    assert out["unstable_intervals"] == [[0.0, 0.25], [0.75, pytest.approx(0.875)]]
    assert out["n_evaluated"] == 4
    empty = criterion_intervals(psi, values, unstable_side="positive", evaluated=np.zeros(5, bool))
    assert empty["unstable_intervals"] == [] and empty["extremum"] is None


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="unstable_side"):
        criterion_intervals([0, 1], [1, 2], unstable_side="up")
    with pytest.raises(ValueError, match="length"):
        criterion_intervals([0, 1], [1, 2, 3], unstable_side="positive")
    with pytest.raises(ValueError, match="increase"):
        criterion_intervals([1, 0], [1, 2], unstable_side="positive")


def test_local_stability_on_real_dcon(rows):
    row = rows["full_edge"]
    out = dcon_local_stability(row)
    # This equilibrium is Mercier and ballooning stable everywhere, but D_R > 0 on 20 surfaces.
    assert out["mercier"]["unstable_intervals"] == [] and out["mercier"]["unstable_fraction"] == 0
    assert out["ballooning"]["unstable_intervals"] == []
    resistive = out["resistive_interchange"]
    assert resistive["unstable_fraction"] == pytest.approx(20 / 129)
    assert resistive["extremum"] == pytest.approx(row["max_D_R"])
    assert resistive["psi_n_at_extremum"] == pytest.approx(row["psi_n_at_max_D_R"])
    assert out["ballooning"]["extremum"] == pytest.approx(row["min_C_A"])
    for lo, hi in resistive["unstable_intervals"]:
        assert lo <= hi


def test_an_unevaluated_criterion_is_none(rows):
    row = dict(rows["full_edge"], ballooning_evaluated=False, mercier_evaluated=None)
    assert dcon_local_stability(row) == {"mercier": None, "resistive_interchange": None, "ballooning": None}


def test_edge_scan_finds_dcons_own_truncation(rows):
    out = dcon_edge_scan(rows["peak_dw_truncated"])
    assert out["truncated_at_peak"] is True
    assert out["psi_n_at_peak"] == pytest.approx(rows["peak_dw_truncated"]["psilim"])
    # q(0.95) = 8.807 on the nominal grid 8 + i/20: 17 pre-edge entries (sing.f:226-238).
    assert out["peak_search_start"] == 17
    assert out["negative_intervals"] and len(out["zero_crossings_psi_n"]) >= 2
    assert dcon_edge_scan(rows["full_edge"]) is None


def test_pre_edge_entries_are_not_searched(rows):
    row = rows["peak_dw_truncated"]
    dw = np.asarray(row["edge_scan"]["dW"]).copy()
    dw[0] = 10 * abs(dw.real).max()
    raised = dict(row, edge_scan=dict(row["edge_scan"], dW=dw))
    assert dcon_edge_scan(raised)["truncated_at_peak"] is True


def test_edge_comparison(rows):
    full, truncated = rows["full_edge"], rows["peak_dw_truncated"]
    out = dcon_edge_comparison(full, truncated)
    assert out["delta_W_edge"] == pytest.approx(full["W_t"].real - truncated["W_t"].real)
    assert out["sign_agreement"] is True and out["n_tor"] == 1
    with pytest.raises(ValueError, match="full_edge"):
        dcon_edge_comparison(truncated, full)
    with pytest.raises(ValueError, match="different n"):
        dcon_edge_comparison(full, dict(truncated, n_tor=2))


def _dcon_edge_scan_replay(q_of_psi, psiedge, qlim0, n_tor, steps, dw_of_psi):
    """Replay sing.f:226-238 and ode.f ode_record_edge on a synthetic q(psi)."""
    q_s = q_of_psi(psiedge)
    q_start = int(q_s)
    size = int(np.ceil((qlim0 - q_start) * n_tor * 20))
    q_edge = q_start + np.arange(size) / (20 * n_tor)
    pre_edge = int(np.count_nonzero(q_edge < q_s))  # 0-based start of the search
    psi_edge = np.zeros(size)
    dw = np.full(size, -1e300, dtype=complex)
    i = 0
    for p in steps:
        qq = q_of_psi(p)
        if qq > qlim0:  # DCON integrates to psilim, where q = qlim
            break
        if qq >= q_edge[i] and p >= psiedge:
            dw[i], q_edge[i], psi_edge[i] = dw_of_psi(p), qq, p
            i = min(i + 1, size - 1)
    row = {"n_tor": n_tor, "requested_psiedge": psiedge, "psilim": None, "qlim": None,
           "edge_scan": {"psi_n": psi_edge, "q": q_edge, "dW": dw}}
    filled = psi_edge > 0
    searched = np.flatnonzero(filled & (np.arange(size) >= pre_edge))
    peak = searched[int(np.argmax(dw[searched].real))]
    return row, pre_edge, float(psi_edge[peak])


def _linear_q(q0, slope, psi0=0.95):
    return lambda p: q0 + slope * (p - psi0)


@pytest.mark.parametrize("q_psiedge", [2.9995, 2.5999, 3.4123])
def test_peak_search_start_matches_dcon_near_integers_and_grid_points(q_psiedge):
    # Coarse ODE steps, so the first filled entry sits past an integer (2.9995)
    # or past a nominal grid point (2.5999 vs 2.60); dW falls outward, so the
    # first searched entry is the peak and any start error shows.
    q = _linear_q(q_psiedge, 30.0)
    steps = np.arange(0.95003, 0.994, 0.00003)
    row, pre_edge, dcon_peak = _dcon_edge_scan_replay(q, 0.95, 4.0, 1, steps, lambda p: complex(1.0 - p))
    out = dcon_edge_scan(dict(row, psilim=dcon_peak))
    assert out["q_edge_start"] == int(q_psiedge)
    assert out["peak_search_start"] == pre_edge and out["peak_search_start_ambiguous"] is False
    assert out["truncated_at_peak"] is True


def test_unfilled_entries_are_skipped_and_pin_q_edge_start():
    q = _linear_q(2.9995, 30.0)
    steps = np.arange(0.95003, 0.9612, 0.00003)  # stop before the scan is full
    row, pre_edge, dcon_peak = _dcon_edge_scan_replay(q, 0.95, 4.0, 1, steps, lambda p: complex(np.sin(40 * p)))
    unfilled = int(np.count_nonzero(row["edge_scan"]["psi_n"] == 0))
    assert unfilled > 0
    out = dcon_edge_scan(dict(row, psilim=dcon_peak))
    assert out["n_points"] == row["edge_scan"]["psi_n"].size - unfilled
    assert out["q_edge_start"] == 2 and out["truncated_at_peak"] is True


def test_an_edge_scan_without_n_raises(rows):
    with pytest.raises(ValueError, match="n_tor"):
        dcon_edge_scan(dict(rows["peak_dw_truncated"], n_tor=None))


def test_ballooning_reads_only_evaluated_surfaces(rows):
    row = rows["full_edge"]
    ca = np.asarray(row["C_A"], dtype=float).copy()
    mask = np.ones(ca.size, dtype=bool)
    mask[:5] = False
    ca[:5] = 0.0  # an old payload's unevaluated zeros
    out = dcon_local_stability(dict(row, C_A=ca, C_A_evaluated=mask))["ballooning"]
    assert out["n_evaluated"] == ca.size - 5 and out["zero_crossings"] == []


def test_an_ambiguous_start_whose_ends_disagree_gives_no_bool():
    # q(psiedge) = 2.60005 sits 5e-5 from the nominal point 2.60 with a tolerance of
    # 9e-5 (a tenth of the first ODE step): the start is ambiguous, and a larger dW on
    # the entry DCON never searched flips the verdict between the two admissible
    # starts. The bool was still returned; #1774's validator says indeterminate
    # (cold review 0.8.0 delta-absorb-18 stability-opspace F3).
    q = _linear_q(2.60005, 30.0)
    steps = np.arange(0.95003, 0.994, 0.00003)
    row, pre_edge, dcon_peak = _dcon_edge_scan_replay(q, 0.95, 4.0, 1, steps, lambda p: complex(1.0 - p))
    assert pre_edge == 13
    out = dcon_edge_scan(dict(row, psilim=dcon_peak))
    assert out["peak_search_start"] == pre_edge and out["peak_search_start_ambiguous"] is True
    assert out["peak_search_start_bracket"] == [12, 13]
    assert out["truncated_at_peak"] is None
    assert out["psi_n_at_peak"] == pytest.approx(dcon_peak)   # the point estimate still follows DCON
    assert out["psi_n_at_peak_bracket"][1] == pytest.approx(dcon_peak)
    assert out["psi_n_at_peak_bracket"][0] < dcon_peak
    # the same ambiguous start with the peak past both ends: the ends agree, so a bool
    row2, _, dcon_peak2 = _dcon_edge_scan_replay(q, 0.95, 4.0, 1, steps, lambda p: complex(-(p - 0.97) ** 2))
    out2 = dcon_edge_scan(dict(row2, psilim=dcon_peak2))
    assert out2["peak_search_start_ambiguous"] is True and out2["peak_search_start_bracket"] == [12, 13]
    assert out2["truncated_at_peak"] is True
    assert dcon_edge_scan(dict(row2, psilim=float(row2["edge_scan"]["psi_n"][20])))["truncated_at_peak"] is False
    # a pinned start keeps a one-entry bracket (the pre-existing clear case)
    out3 = dcon_edge_scan(dict(_dcon_edge_scan_replay(_linear_q(2.5999, 30.0), 0.95, 4.0, 1, steps,
                                                      lambda p: complex(1.0 - p))[0], psilim=None))
    assert out3["peak_search_start_ambiguous"] is False and len(out3["peak_search_start_bracket"]) == 1


def test_the_792_profile_at_psiedge_0_92617_is_indeterminate():
    """The verifier's case: q(0.92617) = 7.79964 on the #792 profile, 4e-4 below the nominal 7.80."""
    from scipy.interpolate import CubicSpline

    from vaft.code.gpec import read_dcon_output

    out = read_dcon_output(REFERENCE / "peak_dw_truncated", mode=1)
    spline = CubicSpline(np.asarray(out.psi_n, float), np.asarray(out.q, float))
    psiedge, step = 0.92617, 1.0 / 20
    q_edge_start = int(spline(psiedge))
    size = int(np.ceil((float(out.q[-1]) - q_edge_start) * 20))
    nominal = q_edge_start + np.arange(size) * step
    pre_edge = int(np.count_nonzero(nominal < float(spline(psiedge))))
    psi, q, nxt = np.zeros(size), nominal.copy(), 0
    for k in range(100000):   # one entry at the first ODE step (3.25e-4) whose q reaches the nominal value
        p = psiedge + (k + 1) * 3.25e-4
        if nxt >= size or p > float(out.psi_n[-1]):
            break
        if float(spline(p)) >= nominal[nxt]:
            psi[nxt], q[nxt] = p, float(spline(p))
            nxt += 1
    psi, q = psi[:nxt], q[:nxt]
    dw = -np.linspace(1.0, 2.0, nxt) + 0j
    dw[pre_edge] = 5.0                       # DCON's peak: the first entry it searches
    row = {"edge_scan": {"psi_n": psi, "q": q, "dW": dw}, "requested_psiedge": psiedge,
           "psilim": float(psi[pre_edge]), "qlim": float(q[pre_edge]), "n_tor": 1}
    # The linear extrapolation lands 4e-5 below the cubic value, 4e-4 below the nominal
    # 7.80, inside the 1.2e-3 tolerance: a start of 17 would miss DCON's peak entry.
    disagree = dcon_edge_scan(row)
    assert disagree["peak_search_start"] == pre_edge == 16 and disagree["peak_search_start_ambiguous"] is True
    assert disagree["peak_search_start_bracket"] == [16, 17]
    assert disagree["truncated_at_peak"] is None and disagree["psi_n_at_peak"] == pytest.approx(row["psilim"])
    assert disagree["psi_n_at_peak_bracket"] == [pytest.approx(row["psilim"]), pytest.approx(float(psi[17]))]
    # the clear case on the same profile: the peak past both ends of the bracket
    dw[pre_edge], dw[pre_edge + 4] = -1.0, 5.0
    clear = dcon_edge_scan(dict(row, psilim=float(psi[pre_edge + 4])))
    assert clear["peak_search_start_ambiguous"] is True and clear["truncated_at_peak"] is True
    assert dcon_edge_scan(dict(row, psilim=float(psi[pre_edge])))["truncated_at_peak"] is False


def test_a_c_a_profile_without_its_mask_is_refused_not_read_as_marginal():
    # DCON stores an unevaluated surface as exactly 0, its marginal value. Without the
    # mask every stored zero counted as a crossing (6 instead of 1, n_evaluated 10 instead
    # of 5) although the docstring promised otherwise
    # (cold review 0.8.0 delta-absorb-18 stability-opspace F5).
    psi_n = np.linspace(0.05, 0.95, 10)
    ca = np.array([0.3, 0.2, 0.0, 0.0, 0.1, -0.2, 0.0, 0.05, 0.0, 0.0])
    row = {"psi_n": psi_n, "D_I": np.full(10, -0.1), "D_R": np.full(10, -0.1), "C_A": ca,
           "mercier_evaluated": True, "ballooning_evaluated": True, "C_A_evaluated": None}
    with pytest.raises(ValueError, match="C_A_evaluated"):
        dcon_local_stability(row)
    with pytest.raises(ValueError, match="C_A_evaluated"):
        dcon_local_stability(dict(row, C_A_evaluated=(ca != 0)[:-1]))
    out = dcon_local_stability(dict(row, C_A_evaluated=ca != 0))["ballooning"]
    assert out["n_evaluated"] == 5 and len(out["zero_crossings"]) == 1
    assert out["unstable_intervals"] == [[pytest.approx(0.483, abs=1e-3), pytest.approx(0.55)]]
    # a run that never evaluated ballooning still gives None, mask or not
    assert dcon_local_stability(dict(row, ballooning_evaluated=False))["ballooning"] is None
