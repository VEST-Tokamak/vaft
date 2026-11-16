"""The consolidation regression portfolio is honest about what it holds.

``test/data/consolidation/case_manifest.yaml`` lists the cases the migration
must be qualified on.  These checks keep it a *frozen-evidence* record: every
packaged input it names still has the sha256 it was frozen with, availability
claims match the checkout, and no case is reported as passed merely by being
listed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test" / "data" / "consolidation" / "case_manifest.yaml"
EVIDENCE_TYPES = {"historical_regression", "analytic", "experimental_physical"}


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(tree, key):
    """First value of ``key`` anywhere in a nested mapping, or None."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def test_case_ids_are_unique_and_evidence_types_are_declared(manifest):
    ids = [case["id"] for case in manifest["cases"]]
    assert len(ids) == len(set(ids))
    for case in manifest["cases"]:
        assert case["evidence_types"], case["id"]
        assert set(case["evidence_types"]) <= EVIDENCE_TYPES, case["id"]
        assert case["purpose"] and case["selection"], case["id"]


def test_nothing_is_passed_by_being_listed(manifest):
    """A passed status needs the run that produced it, recorded with its revision."""
    for case in manifest["cases"]:
        assert case["status"] in manifest["status_vocabulary"], case["id"]
        if case["status"] == "passed":
            assert case.get("result", {}).get("revision"), f"{case['id']} passed without a recorded run"


def test_the_requested_portfolio_is_covered(manifest):
    """The portfolio's initial candidate shots, each present (availability may differ)."""
    shots = {shot for case in manifest["cases"] for shot in case["shots"]}
    assert {39915, 39437, 39438, 46402, 46403, 46404, 45895, 45896, 48371, 48372, 48224} <= shots
    for shot in (41524, 41672, 42962):
        assert shot in shots


def test_every_packaged_input_still_has_its_frozen_sha256(manifest):
    for case in manifest["cases"]:
        for item in case["required_inputs"]:
            if not item.get("available_offline"):
                assert item.get("source"), f"{case['id']}:{item['name']} must say where it lives"
                continue
            path = ROOT / item["path"]
            assert path.is_file(), f"{case['id']}:{item['name']} claims offline availability but {path} is absent"
            if "sha256" in item:
                assert _sha256(path) == item["sha256"], f"{case['id']}:{item['name']} changed on disk"


def test_vestserver_dependence_matches_the_inputs(manifest):
    for case in manifest["cases"]:
        offline = all(item.get("available_offline") for item in case["required_inputs"])
        assert case["requires_vestserver_raw"] is (not offline), case["id"]


def test_the_frozen_hashes_agree_with_the_sample_manifests(manifest):
    """Cross-check against an identity recorded independently, by each sample's own manifest."""
    for shot in (39915, 41524, 41672, 42962):
        sample = yaml.safe_load((ROOT / f"vaft/data/samples/{shot}/manifest.yaml").read_text(encoding="utf-8"))
        declared_raw = (sample.get("pipeline") or {}).get("raw_sha256") or sample["source"]["raw_sha256"]
        case = next(c for c in manifest["cases"] if c["id"] == f"ref-{shot}-chain")
        raw = next(item for item in case["required_inputs"] if item["name"] == "raw_dump")
        assert raw["sha256"] == declared_raw
        omas = next(item for item in case["required_inputs"] if item["name"] == "packaged_omas")
        assert omas["sha256"] == sample["representations"]["omas"]["sha256"]
        config = next(item for item in case["required_inputs"] if item["name"] == "pipeline_config")
        declared_config = (sample.get("pipeline") or {}).get("config_sha256") or _find(sample, "configuration_sha256")
        if shot in (39915, 42962):  # the two sample manifests that record their config's hash
            assert config["sha256"] == declared_config
        canonical = next(item for item in case["required_inputs"] if item["name"] == "canonical_product")
        assert canonical["sha256"] == sample["generation"]["canonical_sha256"]


def test_the_committed_baseline_summary_covers_the_offline_chains(manifest):
    import json

    for case in manifest["cases"]:
        baseline = case.get("baseline")
        if not baseline:
            continue
        summary = json.loads((ROOT / baseline["summary"]).read_text(encoding="utf-8"))
        assert summary["schema"] == "vaft.consolidation.baseline-summary/1"
        assert summary["code_sha"] == baseline["code_sha"]
        assert summary["captured_revision"]["dirty"] is False
        shot = summary["shots"][baseline["shot_key"]]
        assert {stage: entry["status"] for stage, entry in shot["stages"].items()} == {
            "static": "captured", "diagnostics": "captured", "eddy": "captured", "constraints": "captured",
        }
        # The raw input the baseline was built from is the one this case froze.
        raw = next(item for item in case["required_inputs"] if item["name"] == "raw_dump")
        assert shot["inputs"]["raw_dump"]["sha256"] == raw["sha256"]
        config = next(item for item in case["required_inputs"] if item["name"] == "pipeline_config")
        assert shot["inputs"]["pipeline_config"]["sha256"] == config["sha256"]
        # The captured commit carried exactly code_sha's vaft/ and pipeline-1
        # trees (when both commits are in this clone; shallow CI may lack them).
        import subprocess

        commit = summary["captured_revision"]["commit"]
        have = all(
            subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{ref}^{{commit}}"],
                           capture_output=True, check=False).returncode == 0
            for ref in (commit, summary["code_sha"])
        )
        if have:
            assert subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--quiet", summary["code_sha"], commit, "--",
                 *summary["code_trees_identical_to_code_sha"]], check=False,
            ).returncode == 0
        assert case["status"] == "not_tested", "a captured baseline is a reference, not a pass"
