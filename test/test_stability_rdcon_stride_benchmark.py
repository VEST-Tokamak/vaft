"""The RDCON vs STRIDE Δ′ benchmark rules (#143)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "workflow" / "stability_atlas"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("benchmark_rdcon_stride", ROOT / "benchmark_rdcon_stride.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _out(m, psi, matrix):
    return SimpleNamespace(
        m=np.asarray(m), psi_n_rational=np.asarray(psi, dtype=float), Delta_prime=np.asarray(matrix, dtype=complex),
        mlow=-10, mhigh=30,
    )


M = [3, 4, 5]
PSI = [0.4, 0.6, 0.8]
BASE = np.array([[-10.0, 2.0, 0.5], [1.0, -20.0, 3.0], [0.2, 4.0, -30.0]])


def test_identical_converged_codes_pass(bench):
    out = _out(M, PSI, BASE)
    row, surfaces = bench.compare_pair((out, out), (out, out))
    assert row["status"] == "PASS" and row["n_both_converged"] == 3 and row["frobenius_plain"] == 0
    assert all(s["codes_agree"] for s in surfaces)


def test_a_transposed_stride_matrix_is_a_convention_mismatch(bench):
    r = _out(M, PSI, BASE)
    s = _out(M, PSI, BASE.T)
    row, _ = bench.compare_pair((r, r), (s, s))
    assert row["frobenius_transpose"] == pytest.approx(0) and row["status"] == "CONVENTION_MISMATCH"


def test_surfaces_are_matched_by_mode_number_not_position(bench):
    r = _out(M, PSI, BASE)
    # STRIDE lists the same surfaces in reverse order.
    order = [2, 1, 0]
    s = _out([M[i] for i in order], [PSI[i] for i in order], BASE[np.ix_(order, order)])
    row, surfaces = bench.compare_pair((r, r), (s, s))
    assert row["status"] == "PASS" and all(x["relative_difference"] == 0 for x in surfaces)
    assert row["frobenius_plain"] == pytest.approx(0)  # the submatrices are aligned by m, not by index


def test_codes_that_are_not_internally_converged_are_unresolved_not_disagreeing(bench):
    r256, r512 = _out(M, PSI, BASE), _out(M, PSI, -3 * BASE)  # RDCON flips sign with resolution
    s = _out(M, PSI, 5 * BASE)
    row, _ = bench.compare_pair((r256, r512), (s, s))
    assert row["n_both_converged"] == 0 and row["status"] == "NUMERICALLY_UNRESOLVED"


def test_converged_but_different_codes_disagree(bench):
    r = _out(M, PSI, BASE)
    s = _out(M, PSI, np.diag([-10.0, -20.0, +30.0]) + (BASE - np.diag(np.diag(BASE))))
    row, surfaces = bench.compare_pair((r, r), (s, s))
    assert row["status"] == "PHYSICS_DISAGREEMENT"
    assert [x["codes_agree"] for x in surfaces] == [True, True, False]


def test_a_missing_run_is_a_solver_failure(bench):
    out = _out(M, PSI, BASE)
    row, surfaces = bench.compare_pair((None, None), (out, out))
    assert row["status"] == "SOLVER_FAILURE" and row["failed"] == "rdcon_mpsi256;rdcon_mpsi512" and surfaces == []


def test_different_surface_sets_are_partially_comparable(bench):
    r = _out(M + [6], PSI + [0.9], np.pad(BASE, ((0, 1), (0, 1)), constant_values=1.0) - np.diag([0, 0, 0, 41.0]))
    s = _out(M, PSI, BASE)
    row, _ = bench.compare_pair((r, r), (s, s))
    assert row["n_common"] == 3 and row["status"] == "PARTIALLY_COMPARABLE"


def test_like_for_like_stride_variants_match_rdcon_truncation(bench):
    for variant in bench.STRIDE_VARIANTS:
        assert variant.patches["stride.in"] == {"delta_mhigh": 16}
    assert {v.patches["equil.in"]["mpsi"] for v in bench.STRIDE_VARIANTS} == {256, 512}


def test_pass_needs_every_common_surface_converged(bench):
    r256 = _out(M, PSI, BASE)
    r512 = _out(M, PSI, np.diag([-10.0, 200.0, -300.0]))  # only m=3 converged in RDCON
    s_ = _out(M, PSI, np.diag([-10.0, -2000.0, 300.0]))
    row, _ = bench.compare_pair((r256, r512), (s_, s_))
    assert row["n_both_converged"] == 1 and row["status"] == "PARTIALLY_COMPARABLE"


def test_a_missing_check_run_is_a_failure_not_non_convergence(bench):
    out = _out(M, PSI, BASE)
    row, _ = bench.compare_pair((out, None), (out, out))
    assert row["status"] == "SOLVER_FAILURE" and row["failed"] == "rdcon_mpsi512"


def test_same_m_at_a_different_surface_is_not_matched(bench):
    r = _out(M, PSI, BASE)
    s_ = _out(M, [0.4, 0.62, 0.8], BASE)  # m=4 sits elsewhere in STRIDE
    row, surfaces = bench.compare_pair((r, r), (s_, s_))
    assert row["n_common"] == 2 and [x["m"] for x in surfaces] == [3, 5]


def test_duplicate_m_on_two_surfaces_is_kept_apart(bench):
    m, psi = [3, 3, 4], [0.3, 0.7, 0.8]
    out = _out(m, psi, np.diag([-1.0, -2.0, -3.0]))
    row, surfaces = bench.compare_pair((out, out), (out, out))
    assert row["n_common"] == 3 and sorted(x["delta_prime_rdcon"] for x in surfaces) == [-3.0, -2.0, -1.0]


def test_unconverged_noise_cannot_fake_a_convention_mismatch(bench):
    # Large entries tied to the unconverged m=5 make the *full* matrices look
    # transposed (r[0,2] == s[2,0]); the converged m=3,4 block matches as is.
    r_matrix = BASE.copy(); r_matrix[0, 2] = 1e6
    s_matrix = BASE.copy(); s_matrix[2, 0] = 1e6
    r256 = _out(M, PSI, r_matrix)
    r512 = _out(M, PSI, np.diag([-10.0, -20.0, 500.0]))  # m=5 flips sign: unconverged in RDCON
    s_ = _out(M, PSI, s_matrix)
    row, _ = bench.compare_pair((r256, r512), (s_, s_))
    assert row["frobenius_transpose"] < row["frobenius_plain"] / 2  # the full matrix alone would mislead
    assert row["status"] != "CONVENTION_MISMATCH"


def test_no_surfaces_at_all_is_not_applicable(bench):
    empty = _out([], [], np.zeros((0, 0)))
    row, _ = bench.compare_pair((empty, empty), (empty, empty))
    assert row["status"] == "NOT_APPLICABLE"


def test_truncation_match_is_recorded(bench):
    out = _out(M, PSI, BASE)
    other = _out(M, PSI, BASE); other.mhigh = 22
    row, _ = bench.compare_pair((out, out), (other, other))
    assert row["truncation_match"] is False
