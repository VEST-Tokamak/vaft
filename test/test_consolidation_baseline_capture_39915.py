"""End-to-end baseline capture of shot 39915 into a scratch root (issue #2166).

Runs the four offline stages twice (~1 min) and checks the record is complete,
traceable and reproducible: every stage captured, the raw input's and config's
sha256 equal to those the packaged manifest froze (identities recorded
independently of this run), two captures identical leaf for leaf, and the
packaged sample tree byte-identical afterwards.  ``slow``: release confidence.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflow" / "architecture_audit" / "capture_baseline.py"


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_39915_chain_is_captured_leaf_by_leaf(tmp_path):
    spec = importlib.util.spec_from_file_location("capture_baseline", SCRIPT)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)

    sample = ROOT / "vaft" / "data" / "samples" / "39915"
    before = _tree_digest(sample)
    root = capture.check_output_root(tmp_path / "baseline", environ={})
    root.mkdir()
    record = capture.capture_shot(39915, root, environ={})

    manifest = yaml.safe_load((sample / "manifest.yaml").read_text(encoding="utf-8"))
    assert record["inputs"]["raw_dump"]["sha256"] == manifest["pipeline"]["raw_sha256"]
    assert record["inputs"]["pipeline_config"]["sha256"] == manifest["pipeline"]["config_sha256"]
    assert list(record["stages"]) == list(capture.STAGES)
    for stage, entry in record["stages"].items():
        assert entry["status"] == "captured", (stage, entry.get("log_tail"))
        assert entry["leaves"], stage
    magnetics = [p for p in record["stages"]["diagnostics"]["leaves"] if p.startswith("magnetics.b_field_pol_probe.")]
    assert any(p.endswith(".field.data") for p in magnetics)
    assert (root / "39915" / "baseline.json").is_file()

    # Reproducible: a second capture into a fresh root agrees leaf for leaf.
    again_root = capture.check_output_root(tmp_path / "again", environ={})
    again_root.mkdir()
    again = capture.capture_shot(39915, again_root, environ={})
    for stage in capture.STAGES:
        assert again["stages"][stage]["leaves"] == record["stages"][stage]["leaves"], stage
        assert again["stages"][stage]["product_sha256"] == record["stages"][stage]["product_sha256"], stage

    assert _tree_digest(sample) == before
