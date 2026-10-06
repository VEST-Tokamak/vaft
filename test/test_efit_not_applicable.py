"""A shot EFIT does not apply to is a result, not a failure (#205).

A vacuum shot reached the EFIT constraint step and failed there ("every one of
the 65 selected instants carries less than 15000.0 A", vestserver 48927,
2026-10-06), so the new-shot worker retried it until it gave up.  The
constraint step now records the verdict, and the k-file, EFIT and EFIT-ODS
steps pass it on the way they pass on ``efit.run=false``: the EFIT stage ends
``no_output`` with ``skipped: not applicable: ...``, which replication accepts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from omas import load_omas_json, save_omas_json

from _plasma_timing_fixtures import current, grid, pickup_only, synthetic_ods
from vaft.code.efit.applicability import (
    constraints_not_applicable_reason,
    kfile_manifest_not_applicable_reason,
    kfile_manifest_paths,
    kfile_manifest_text,
    not_applicable_constraints,
)
from vaft.database.replication import ProductNotEligibleError, _nothing_to_replicate

WORKFLOW = Path(__file__).parents[1] / "workflow/automatic_pipeline_1_routine_data_processing"


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_not_applicable", WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONSTRAINTS = _module("generate_constraints_ods")


def _vacuum():
    t = grid()
    ods = synthetic_ods(ip=pickup_only(t), t=t)
    ods["magnetics.ip.0.time"] = t  # the vacuum cut reads the node's own clock
    return ods


def _run_constraints(monkeypatch, tmp_path, ods):
    monkeypatch.setattr(CONSTRAINTS, "compose_stage_products", lambda **_: (ods, {}))
    output = tmp_path / "constraints" / "48927_constraints.json"
    monkeypatch.setattr(sys, "argv", [
        "generate_constraints_ods.py", "--shot", "48927", "--eddy-ods", "e", "--diagnostics-ods", "d",
        "--output", str(output),
    ])
    return CONSTRAINTS.main(), output


def test_a_vacuum_shot_records_efit_as_not_applicable(monkeypatch, tmp_path):
    code, output = _run_constraints(monkeypatch, tmp_path, _vacuum())

    assert code == 0
    reason = constraints_not_applicable_reason(load_omas_json(str(output), consistency_check=False))
    assert reason.startswith("Vacuum shot (no plasma-current pulse")
    assert "below CUTIP 15000 A" in reason


def test_a_plasma_shot_below_cutip_still_fails(monkeypatch, tmp_path):
    """The timing saw a discharge the current does not support: a fault, not a result."""
    t = grid()
    ods = synthetic_ods(ip=current(t, peak=60e3), t=t)
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)

    assert CONSTRAINTS._efit_not_applicable(ods, error) is None


def test_the_cut_still_raises_a_value_error():
    """Existing callers catching ValueError keep working."""
    assert issubclass(CONSTRAINTS.NoPlasmaCurrentError, ValueError)


def test_a_constraints_product_with_slices_is_reconstructed_whatever_its_comment():
    ods = not_applicable_constraints("Vacuum shot")
    assert constraints_not_applicable_reason(ods) == "Vacuum shot"
    ods["equilibrium.time"] = np.array([0.31])
    assert constraints_not_applicable_reason(ods) is None


def test_kfile_manifest_marker_is_not_a_path():
    text = kfile_manifest_text("Vacuum shot (no pulse)\n over two lines")
    assert kfile_manifest_paths(text) == []
    assert kfile_manifest_not_applicable_reason(text) == "Vacuum shot (no pulse) over two lines"
    assert kfile_manifest_not_applicable_reason("/a/k048927.00310\n") is None


def _script(name, *args):
    result = subprocess.run([sys.executable, str(WORKFLOW / f"{name}.py"), *map(str, args)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]


def test_the_verdict_runs_through_kfile_efit_and_replication(tmp_path):
    """constraints -> k-file -> EFIT (switched on, even without a binary) -> EFIT ODS -> replication."""
    shot = 48927
    root = tmp_path / "efit" / "magnetic" / str(shot)
    constraints = root / "constraints" / "constraints.json"
    constraints.parent.mkdir(parents=True)
    save_omas_json(not_applicable_constraints("Vacuum shot (no light); all 65 below CUTIP"), str(constraints))
    kfiles = root / "kfile" / "manifest.txt"
    _script("generate_kfile", "--shot", shot, "--constraints-ods", constraints, "--output", kfiles)
    assert not list(kfiles.parent.glob("k0*"))

    gfiles, status, artifacts = root / "gfile" / "manifest.txt", root / "status.txt", root / "artifacts.json"
    _script("run_efit_reconstruction", "--shot", shot, "--kfile-manifest", kfiles,
            "--gfile-manifest", gfiles, "--status", status, "--artifact-manifest", artifacts,
            "--run", "true", "--executable", tmp_path / "no-efit")
    assert status.read_text().startswith("skipped: not applicable: Vacuum shot (no light)")
    assert gfiles.read_text() == ""

    product, manifest_path = root / "output" / "efit.json", root / "metadata" / "manifest.json"
    _script("generate_efit_ods", "--shot", shot, "--gfile-manifest", gfiles, "--status", status,
            "--constraints-ods", constraints, "--kfile-manifest", kfiles,
            "--artifact-manifest", artifacts, "--output", product, "--metadata", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "no_output" and manifest["slice_statuses"] == []
    assert "not applicable" in _nothing_to_replicate(manifest, "efit")


def test_replication_still_refuses_a_skip_that_is_a_fault():
    manifest = {"status": "no_output", "efit_status": "skipped: EFIT executable unavailable: /x"}
    with pytest.raises(ProductNotEligibleError):
        _nothing_to_replicate(manifest, "efit")
