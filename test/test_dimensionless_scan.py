"""The TGLF dimensionless-parameter scan: which inputs move with which dimensionless parameter, and the table.

No TGLF is needed. The solver records are written by hand, the way ``_run_one``
writes them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scan():
    path = ROOT / "workflow" / "transport_atlas" / "dimensionless_scan.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_dimensionless_scan", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = {"XNUE": 0.9, "BETAE": 0.005, "P_PRIME_LOC": -0.004, "Q_LOC": 3.0, "Q_PRIME_LOC": 6.0,
        "DEBYE": 0.005, "SAT_RULE": 3}


def test_q_moves_at_fixed_shear_beta_and_collisionality(scan):
    up = scan.perturbed(BASE, "q", +1.0, 0.1)
    assert up["Q_LOC"] == pytest.approx(3.3)
    # shear s = Q_PRIME (r/q)^2 is unchanged; beta* = -8 pi P_PRIME r / q is unchanged
    assert up["Q_PRIME_LOC"] / up["Q_LOC"] ** 2 == pytest.approx(BASE["Q_PRIME_LOC"] / BASE["Q_LOC"] ** 2)
    assert up["P_PRIME_LOC"] / up["Q_LOC"] == pytest.approx(BASE["P_PRIME_LOC"] / BASE["Q_LOC"])
    # nu* ~ XNUE q (Verdoolaege form): held fixed
    assert up["XNUE"] * up["Q_LOC"] == pytest.approx(BASE["XNUE"] * BASE["Q_LOC"])
    assert up["BETAE"] == BASE["BETAE"] and up["DEBYE"] == BASE["DEBYE"]


def test_beta_and_nu_star_move_only_their_inputs(scan):
    beta = scan.perturbed(BASE, "beta", -1.0, 0.1)
    assert beta["BETAE"] == pytest.approx(0.0045) and beta["P_PRIME_LOC"] == pytest.approx(-0.0036)
    assert {k for k in BASE if beta[k] != BASE[k]} == {"BETAE", "P_PRIME_LOC"}
    nu = scan.perturbed(BASE, "nu_star", +1.0, 0.1)
    assert {k for k in BASE if nu[k] != BASE[k]} == {"XNUE"}
    with pytest.raises(KeyError, match="XNUE"):
        scan.perturbed({"BETAE": 1.0}, "nu_star", 1.0)


def test_sensitivity_is_the_centred_log_slope(scan):
    assert scan.sensitivity(1.1 ** -1.5, 0.9 ** -1.5, 0.1) == pytest.approx(-1.5)
    assert scan.sensitivity(0.0, 1.0) is None and scan.sensitivity(-1.0, -2.0) is None
    assert scan.sensitivity(None, 1.0) is None and scan.sensitivity(math.inf, 1.0) is None


def _write_case(out, *, exponents, base=(2.0, 1.0, 0.5), fail=()):
    """One surface with Q = Q0 * f^exponent[scan] for every quantity."""
    leaf = out / "runs" / "tglf-sat3-em-bper" / "39915" / "magnetics" / "00317" / "r0.70"
    native = out / "native"
    native.mkdir(parents=True, exist_ok=True)
    (native / "outputs.json").write_text(json.dumps({
        "schema_version": 3, "directory": str(native),
        "gbflux": {"energy": {"__array__": list(base)}, "particle": {"__array__": [0, 0, 0]}}}))
    leaf.mkdir(parents=True)
    (leaf / "case.json").write_text(json.dumps({
        "config": "tglf-sat3-em-bper", "shot": 39915, "time_efit_s": 0.317, "efit_lineage": "magnetics",
        "efit_quality": "good", "state_identity": "s", "r_over_a": 0.7, "sat_rule": 3,
        "native_dir": str(native)}))

    def record(name, factor):
        d = leaf / name
        d.mkdir()
        energy = [b * factor for b in base]
        status = "failed" if name in fail else "solved"
        fluxes = {"qe": energy[0], "qi": sum(energy[1:]), "q_tot": sum(energy)} if status == "solved" else None
        (d / "record.json").write_text(json.dumps({"status": status, "fluxes_gb": fluxes}))

    record("base", 1.0)
    for name, exponent in exponents.items():
        record(f"{name}_plus", 1.1 ** exponent)
        record(f"{name}_minus", 0.9 ** exponent)
    (out / "scan.json").write_text(json.dumps({"delta": 0.1, "scans": {}, "configs": []}))


def test_the_table_recovers_known_exponents_and_keeps_gb_and_dimensional_apart(scan, tmp_path):
    _write_case(tmp_path, exponents={"nu_star": -0.8, "beta": 0.3, "q": 1.2})
    rows = scan.build_rows(tmp_path, 0.1)
    (qe,) = [r for r in rows if r["quantity"] == "qe"]
    assert qe["status"] == "ok" and qe["base_reproduction"] == pytest.approx(0.0)
    assert qe["S_gb_nu_star"] == pytest.approx(-0.8) and qe["S_gb_q"] == pytest.approx(1.2)
    assert qe["S_gb_rho_star"] == 0.0 and qe["S_dim_rho_star"] == 2.0
    assert qe["S_dim_beta"] == qe["S_gb_beta"] == pytest.approx(0.3)
    assert qe["S_gb_beta_plus_side"] == pytest.approx(0.3) and qe["S_gb_beta_minus_side"] == pytest.approx(0.3)
    summary = scan.build_summary(rows, {"nu_star": {"direct": 0.42, "direct_se": 0.35}})
    nu = [s for s in summary if s["quantity"] == "qe" and s["index"] == "mu_nu"][0]
    assert nu["tglf_mu_median"] == pytest.approx(0.8) and nu["fit548_direct"] == 0.42
    rho = [s for s in summary if s["quantity"] == "qe" and s["index"] == "mu_rho"][0]
    assert rho["tglf_mu_median"] == -3.0 and "construction" in rho["note"]


def test_a_failed_side_is_reported_not_filled(scan, tmp_path):
    _write_case(tmp_path, exponents={"nu_star": -0.8, "beta": 0.3, "q": 1.2}, fail=("q_minus",))
    (qe,) = [r for r in scan.build_rows(tmp_path, 0.1) if r["quantity"] == "qe"]
    assert qe["status"] == "partial" and qe["S_gb_q"] is None and "q: run failed" in qe["reason"]
    assert qe["S_gb_q_plus_side"] == pytest.approx(1.2)


def test_reference_indices_read_both_548_fits(scan, tmp_path):
    (tmp_path / "extensions").mkdir()
    (tmp_path / "closures").mkdir()
    (tmp_path / "extensions" / "coefficients.csv").write_text(
        'fit,term,coef,se_cluster\n"dimensionless:rho,beta,nu,q",nu_star,0.42,0.35\n'
        '"dimensionless:rho,beta,nu",nu_star,0.51,0.29\n')   # a reduced fit must not win
    (tmp_path / "closures" / "closures.csv").write_text(
        "data,model,mu_nu_completed,mu_nu_completed_se\nprimary:A,free,-2.2,1.5\nprimary:A,bohm,,\n")
    found = scan.reference_indices(tmp_path)
    assert found["nu_star"] == {"direct": 0.42, "direct_se": 0.35, "completed": -2.2, "completed_se": 1.5}
