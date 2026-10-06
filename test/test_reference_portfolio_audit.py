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
    from vaft.database.sources import STAGE_REPLICATION

    roots = {}
    for product in scan["products"]:
        roots.setdefault(product["root"], {})[product["stage"]] = product
    out = {}
    for stage in AUDIT.UPSTREAM_ORDER:
        for root, by_stage in roots.items():
            if stage in by_stage:
                state, reason = AUDIT._classify(by_stage[stage], by_stage, STAGE_REPLICATION)
                by_stage[stage]["_state"] = state
                out[stage] = (state, reason)
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
    assert states["efit"][0] == "regeneration-required"
    assert "eddy" in states["efit"][1]
    # Newer than its EFIT, but built on a stale one.
    assert states["chease"][0] == "regeneration-required"
    assert "itself stale" in states["chease"][1]


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
    from vaft.database.sources import STAGE_REPLICATION

    scan = AUDIT.scan([str(tmp_path)], ["1"])
    by_stage = {p["stage"]: p for p in scan["products"]}
    efit_state, efit_reason = AUDIT._classify(by_stage["efit"], by_stage, STAGE_REPLICATION)
    assert efit_state == "validation-failed"
    assert "EFIT output unavailable" in efit_reason
    shotlog_state, _ = AUDIT._classify(by_stage["shotlog"], by_stage, STAGE_REPLICATION)
    assert shotlog_state == "available"


def test_missing_commit_is_reported_as_a_provenance_gap(tmp_path):
    _product(tmp_path, "diagnostics", 1, 1_000)
    states = _classified(AUDIT.scan([str(tmp_path)], ["1"]))
    assert states["diagnostics"] == (
        "available",
        "provenance gap: manifest records no VAFT commit",
    )
