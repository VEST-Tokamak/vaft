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
    assert row["status"] == "SOLVER_FAILURE" and row["failed"] == "rdcon" and surfaces == []


def test_different_surface_sets_are_partially_comparable(bench):
    r = _out(M + [6], PSI + [0.9], np.pad(BASE, ((0, 1), (0, 1)), constant_values=1.0) - np.diag([0, 0, 0, 41.0]))
    s = _out(M, PSI, BASE)
    row, _ = bench.compare_pair((r, r), (s, s))
    assert row["n_common"] == 3 and row["status"] == "PARTIALLY_COMPARABLE"


def test_like_for_like_stride_variants_match_rdcon_truncation(bench):
    for variant in bench.STRIDE_VARIANTS:
        assert variant.patches["stride.in"] == {"delta_mhigh": 16}
    assert {v.patches["equil.in"]["mpsi"] for v in bench.STRIDE_VARIANTS} == {256, 512}
