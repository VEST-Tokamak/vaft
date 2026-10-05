"""Verification of DCON/RDCON/STRIDE results in the validation layer (#142).

Real GPEC output (the #792 DCON fixtures and the #939 RDCON fixture), plus
copies altered to break exactly one rule each.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vaft.code.gpec import read_dcon_output, read_pest3_matching_output
from vaft.validation.model import ValidationStatus
from vaft.validation.registry import CHECKS
from vaft.validation.stability import STABILITY_CHECKS, validate_stability

DATA = Path(__file__).resolve().parent / "data" / "gpec"
PASS, WARN, FAIL = "pass", "warn", "fail"
INDETERMINATE, NOT_AVAILABLE = "indeterminate", "not_available"


@pytest.fixture(scope="module")
def full():
    return read_dcon_output(DATA / "dcon_edge_792" / "full_edge", mode=1)


@pytest.fixture(scope="module")
def truncated():
    return read_dcon_output(DATA / "dcon_edge_792" / "peak_dw_truncated", mode=1)


@pytest.fixture(scope="module")
def rdcon():
    return read_pest3_matching_output(DATA / "rdcon_39915_319_n1", solver="rdcon", mode=1)


def test_real_dcon_runs_verify(full, truncated):
    for out in (full, truncated):
        checks = validate_stability(dcon=out)["verification"]
        assert checks["dcon_energies"]["status"] == PASS
        assert checks["dcon_imaginary_part"]["status"] == PASS
        assert checks["dcon_local_criteria"]["status"] == PASS
        assert checks["dcon_edge"]["status"] == PASS
        # The trimmed fixture carries no W_t matrix, and there is no second run.
        assert checks["dcon_hermiticity"]["status"] == NOT_AVAILABLE
        assert checks["dcon_resolution"]["status"] == NOT_AVAILABLE


def test_real_rdcon_run_verifies(rdcon):
    report = validate_stability(matching=rdcon)
    checks = report["verification"]
    assert checks["matching_matrices"]["status"] == PASS and checks["matching_matrices"]["msing"] == 9
    assert checks["matching_surfaces"]["status"] == PASS
    assert checks["matching_resolution"]["status"] == NOT_AVAILABLE
    assert report["provenance"]["matching_solver"] == "rdcon"


def test_report_shape_and_vocabulary(full, rdcon):
    report = validate_stability(dcon=full, matching=rdcon)
    assert report["schema_version"] == 1 and set(report["summary"]) == {"dcon", "matching"}
    statuses = {c["status"] for c in report["verification"].values()} | set(report["summary"].values()) | {report["status"]}
    assert statuses <= {s.value for s in ValidationStatus}
    # Partly unavailable evidence is never a pass.
    assert report["summary"]["dcon"] == INDETERMINATE


def test_every_check_is_described_and_stays_out_of_the_equilibrium_registry(full, rdcon):
    report = validate_stability(dcon=full, matching=rdcon)
    keys = {f"verification.{name}" for name in report["verification"]}
    assert keys == set(STABILITY_CHECKS)
    assert not keys & set(CHECKS)
    for spec in STABILITY_CHECKS.values():
        assert spec.measure == "rule" and spec.tolerance is None


def test_a_dominant_imaginary_part_warns(full):
    bad = dataclasses.replace(full, W_t_eigenvalue=np.asarray(full.W_t_eigenvalue) + 1j * 10 * abs(full.total1.real))
    assert validate_stability(dcon=bad)["verification"]["dcon_imaginary_part"]["status"] == WARN


def test_a_non_hermitian_matrix_warns_and_a_hermitian_one_passes(full):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    hermitian = dataclasses.replace(full, W_t=a + a.conj().T)
    skew = dataclasses.replace(full, W_t=a - a.conj().T)
    assert validate_stability(dcon=hermitian)["verification"]["dcon_hermiticity"]["status"] == PASS
    assert validate_stability(dcon=skew)["verification"]["dcon_hermiticity"]["status"] == WARN


def test_unevaluated_criteria_are_indeterminate_and_nonfinite_ones_fail(full):
    off = dataclasses.replace(full, evaluation=dataclasses.replace(full.evaluation, bal_flag=False))
    assert validate_stability(dcon=off)["verification"]["dcon_local_criteria"]["status"] == INDETERMINATE
    di = np.asarray(full.di, dtype=float).copy()
    di[3] = np.nan
    broken = dataclasses.replace(full, di=di)
    assert validate_stability(dcon=broken)["verification"]["dcon_local_criteria"]["status"] == FAIL


def test_a_truncated_run_off_its_peak_fails(truncated):
    moved = dataclasses.replace(truncated, psilim=float(truncated.psilim) - 0.01)
    check = validate_stability(dcon=moved)["verification"]["dcon_edge"]
    assert check["status"] == FAIL and check["psi_n_at_dW_peak"] == pytest.approx(truncated.psilim)


def test_resolution_sign_agreement(full):
    flipped = dataclasses.replace(full, W_t_eigenvalue=-np.asarray(full.W_t_eigenvalue))
    assert validate_stability(dcon=full, dcon_check=full)["verification"]["dcon_resolution"]["status"] == PASS
    assert validate_stability(dcon=full, dcon_check=flipped)["verification"]["dcon_resolution"]["status"] == INDETERMINATE


def test_a_misassigned_surface_fails(rdcon):
    q = np.asarray(rdcon.q_rational, dtype=float).copy()
    q[2] += 0.7  # n*q no longer rounds to its m
    bad = dataclasses.replace(rdcon, q_rational=q)
    check = validate_stability(matching=bad)["verification"]["matching_surfaces"]
    assert check["status"] == FAIL and check["misassigned_m"] == [int(rdcon.m[2])]


def test_a_non_finite_or_misshapen_delta_prime_fails(rdcon):
    dp = np.asarray(rdcon.Delta_prime).copy()
    dp[0, 0] = np.nan
    assert validate_stability(matching=dataclasses.replace(rdcon, Delta_prime=dp))["verification"]["matching_matrices"]["status"] == FAIL
    assert validate_stability(matching=dataclasses.replace(rdcon, Delta_prime=dp[:-1]))["verification"]["matching_matrices"]["status"] == FAIL


def test_matching_resolution_is_indeterminate_when_a_surface_moves(rdcon):
    dp = np.asarray(rdcon.Delta_prime).copy()
    dp[4, 4] = -3 * dp[4, 4]
    changed = dataclasses.replace(rdcon, Delta_prime=dp)
    same = validate_stability(matching=rdcon, matching_check=rdcon)["verification"]["matching_resolution"]
    moved = validate_stability(matching=rdcon, matching_check=changed)["verification"]["matching_resolution"]
    assert same["status"] == PASS and same["consistent"] == 9
    assert moved["status"] == INDETERMINATE and moved["inconsistent"] == 1


def test_no_runs_at_all_is_not_available():
    report = validate_stability()
    assert report["status"] == NOT_AVAILABLE
    assert all(c["status"] == NOT_AVAILABLE for c in report["verification"].values())
