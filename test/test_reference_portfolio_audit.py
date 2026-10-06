"""Coverage classification of the reference-portfolio audit (#1712)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_audit():
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "reference_validation"
        / "audit_portfolio.py"
    )
    spec = importlib.util.spec_from_file_location("audit_portfolio", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["audit_portfolio"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit()


def _product(tmp_path, stage, shot, mtime, *, status="success", lineage=(), payload=None):
    directory = tmp_path / "omas" / stage / Path(*lineage) / str(shot)
    (directory / "output").mkdir(parents=True)
    (directory / "metadata").mkdir()
    output = directory / "output" / f"{stage}.json"
    output.write_text(json.dumps(payload or {"x": "y" * 2048}))
    (directory / "metadata" / "manifest.json").write_text(
        json.dumps({"status": status, "stage": stage, "output": {"sha256": "abc"}})
    )
    os.utime(output, (mtime, mtime))
    return directory


def _classified(scan):
    """Classify every product in topological order, as ``report`` does."""
    from vaft.database.sources import STAGE_REPLICATION

    order = list(AUDIT.UPSTREAM_ORDER)
    products = sorted(
        scan["products"],
        key=lambda p: (order.index(p["stage"]) if p["stage"] in order else len(order), p["path"]),
    )
    out = {}
    for product in products:
        state, reason = AUDIT._classify(product, scan["products"], STAGE_REPLICATION)
        product["_state"] = state
        out["/".join([product["stage"], *product["lineage"]])] = (state, reason)
    return out


def test_scan_finds_family_and_flat_products(tmp_path):
    _product(tmp_path, "diagnostics", 1, 1_000)
    _product(tmp_path, "efit", 1, 2_000, lineage=("magnetic",))
    scan = AUDIT.scan([str(tmp_path)], ["1"])
    paths = sorted(p["path"] for p in scan["products"])
    assert paths == ["omas/diagnostics/1", "omas/efit/magnetic/1"]
    efit = next(p for p in scan["products"] if p["stage"] == "efit")
    assert efit["lineage"] == ["magnetic"]
    assert efit["manifest"]["sha256"] == "abc"


def test_product_older_than_its_input_is_stale_and_staleness_propagates(tmp_path):
    _product(tmp_path, "diagnostics", 1, 1_000)
    _product(tmp_path, "eddy", 1, 5_000)
    _product(tmp_path, "efit", 1, 2_000, lineage=("magnetic",))
    _product(tmp_path, "chease", 1, 6_000, lineage=("magnetic",))
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["eddy"][0] == "available"
    assert states["efit/magnetic"][0] == "regeneration-required"
    assert "eddy" in states["efit/magnetic"][1]
    # Newer than its EFIT, but built on a stale one.
    assert states["chease/magnetic"][0] == "regeneration-required"
    assert "itself stale" in states["chease/magnetic"][1]


def test_failed_manifest_is_validation_failed_but_small_success_is_not(tmp_path):
    _product(
        tmp_path,
        "efit",
        1,
        1_000,
        status="no_output",
        lineage=("magnetic",),
        payload={"equilibrium": {"ids_properties": {"comment": "EFIT output unavailable"}}},
    )
    _product(tmp_path, "shotlog", 1, 1_000, payload={"pulse_schedule": {}})
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["efit/magnetic"][0] == "validation-failed"
    assert "EFIT output unavailable" in states["efit/magnetic"][1]
    assert states["shotlog"][0] == "available"


def test_unavailable_is_a_normal_outcome_not_a_failure(tmp_path):
    # Pipeline 2 writes `unavailable` when the input does not exist for the
    # shot (no CES upload; the non-applicable one of electron/kinetic EFIT).
    _product(tmp_path, "ces", 1, 1_000, status="unavailable", payload={"charge_exchange": {}})
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["ces"][0] == "input-unavailable"


def test_stale_efit_makes_the_profile_chain_stale(tmp_path):
    _product(tmp_path, "eddy", 1, 1_000)
    _product(tmp_path, "efit", 1, 3_000, lineage=("magnetic",))
    _product(tmp_path, "thomson", 1, 1_000)
    _product(tmp_path, "core_profiles", 1, 2_000)
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["core_profiles"][0] == "regeneration-required"
    assert "efit" in states["core_profiles"][1]


def test_failed_upstream_makes_a_newer_product_stale(tmp_path):
    _product(tmp_path, "eddy", 1, 1_000)
    _product(tmp_path, "efit", 1, 2_000, status="failed", lineage=("magnetic",))
    _product(tmp_path, "chease", 1, 3_000, lineage=("magnetic",))
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["chease/magnetic"][0] == "regeneration-required"
    assert "failed validation" in states["chease/magnetic"][1]


def test_staleness_is_judged_per_lineage(tmp_path):
    _product(tmp_path, "eddy", 1, 1_000)
    _product(tmp_path, "efit", 1, 2_000, lineage=("magnetic",))
    _product(tmp_path, "mhd_linear", 1, 5_000, lineage=("magnetic", "chease", "rdcon"))
    _product(tmp_path, "mhd_linear", 1, 1_500, lineage=("magnetic", "chease", "stride"))
    _product(tmp_path, "chease", 1, 3_000, lineage=("magnetic",))
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["mhd_linear/magnetic/chease/rdcon"][0] == "available"
    assert states["mhd_linear/magnetic/chease/stride"][0] == "regeneration-required"


def test_matrix_keeps_lineages_apart_and_shows_the_worst_row():
    rows = [
        {"shot": 1, "stage": "mhd_linear", "lineage": "magnetic/chease/rdcon", "filedb": "/srv/vest.filedb",
         "output": {"mtime": "2026-10-01T00:00:00+00:00"}, "state": "available", "ids": "mhd_linear"},
        {"shot": 1, "stage": "mhd_linear", "lineage": "magnetic/chease/stride", "filedb": "/srv/vest.filedb",
         "output": {"mtime": "2026-09-01T00:00:00+00:00"}, "state": "regeneration-required", "ids": "mhd_linear"},
        {"shot": 1, "stage": "mhd_linear", "lineage": "magnetic/chease/stride", "filedb": "/srv/vest.filedb",
         "output": {"mtime": "2026-09-01T00:00:00+00:00"}, "state": "available", "ids": "ntms"},
    ]
    header, _, line = AUDIT._matrix(rows)
    assert "mhd_linear/chease/rdcon" in header and "mhd_linear/chease/stride" in header
    cells = [c.strip() for c in line.split("|")]
    assert "ok 10-01" in cells and "STALE 09-01" in cells


def test_partial_diagnostics_narrow_per_ids():
    product = {"manifest": {"channel_status": {"magnetics": "partial", "langmuir_probes": "unavailable"}}}
    assert AUDIT._ids_state(product, "magnetics", "available", "")[0] == "available"
    assert AUDIT._ids_state(product, "langmuir_probes", "available", "")[0] == "input-unavailable"
    assert AUDIT._ids_state(product, "tf", "available", "") == ("available", "")


def test_missing_commit_is_reported_as_a_provenance_gap(tmp_path):
    _product(tmp_path, "diagnostics", 1, 1_000)
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["diagnostics"] == (
        "available",
        "provenance gap: manifest records no VAFT commit",
    )
