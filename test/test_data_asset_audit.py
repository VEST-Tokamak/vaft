"""The data-asset audit (#2307) reads a tree through git plumbing only."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _load_audit():
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "data_asset_audit"
        / "audit_assets.py"
    )
    spec = importlib.util.spec_from_file_location("audit_assets", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["audit_assets"] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit()

LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:" + "a" * 64 + "\n"
    "size 134459248\n"
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "audit@example.invalid")
    _git(root, "config", "user.name", "audit")
    _git(root, "config", "commit.gpgsign", "false")

    _write(
        root,
        "pyproject.toml",
        b'[tool.setuptools.package-data]\nvaft = ["data/samples/*/manifest.yaml"]\n',
    )
    good = gzip.compress(b'{"equilibrium": {}}', mtime=0)
    bad = gzip.compress(b'{"magnetics": {}}', mtime=0)
    _write(root, "vaft/data/samples/11111/omas.json.gz", good)
    _write(root, "vaft/data/samples/22222/omas.json.gz", bad)
    for shot, sha in ((11111, hashlib.sha256(good).hexdigest()), (22222, "0" * 64)):
        manifest = {
            "schema_version": 1,
            "shot": shot,
            "reference_id": f"test-{shot}-v1",
            "imas_dd_version": "3.41.0",
            "representations": {
                "omas": {"path": "omas.json.gz", "sha256": sha, "size": 1, "package": "repository-only"}
            },
        }
        _write(root, f"vaft/data/samples/{shot}/manifest.yaml", yaml.safe_dump(manifest).encode())
    _write(root, "vaft/data/efit/rv129129.ddd", LFS_POINTER.encode())
    _write(root, "vaft/data/efit/mhdin.dat", b"&in3\n/\n")
    _write(root, "test/test_reader.py", b'PATH = "vaft/data/samples/11111/omas.json.gz"\n')
    _write(root, "big/blob.bin", b"x" * 4096)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")

    monkeypatch.setattr(AUDIT, "REPO_ROOT", root)
    monkeypatch.setattr(AUDIT, "UNASSIGNED_THRESHOLD", 1024)
    return root


def _group(inventory, group_id):
    return next(row for row in inventory["groups"] if row["id"] == group_id)


def test_specific_groups_win_over_catch_alls():
    assert AUDIT.group_of("vaft/data/samples/39915/omas.json.gz", AUDIT.GROUPS).id == (
        "samples.shot.representations"
    )
    assert AUDIT.group_of("vaft/data/samples/39915/source/raw.json.gz", AUDIT.GROUPS).id == (
        "samples.shot.sources"
    )
    assert AUDIT.group_of("vaft/data/samples/39915/mhd_stability/x.json.gz", AUDIT.GROUPS).id == (
        "samples.shot.stage_products"
    )
    assert AUDIT.group_of("vaft/data/efit/brzgfc.dat", AUDIT.GROUPS).id == (
        "efit.generated_tables_plain"
    )
    # History attribution must not let the vaft/data/* catch-all claim retired layouts.
    assert AUDIT.group_of("vaft/data/omas/39915.json", AUDIT.GROUPS).id == "data.geometry_config"
    assert AUDIT.group_of("vaft/data/omas/39915.json", AUDIT.HISTORY_ORDER).id == (
        "history.retired_sample_layouts"
    )


def test_every_group_uses_a_known_category_and_decision():
    ids = [group.id for group in AUDIT.GROUPS]
    assert len(ids) == len(set(ids))
    for group in AUDIT.GROUPS:
        assert group.category in AUDIT.CATEGORIES
        assert group.decision in {"decided", "proposed", "open"}


def test_lfs_pointer_and_wheel_substitution():
    assert AUDIT.lfs_pointer(LFS_POINTER.encode()) == {"oid": "sha256:" + "a" * 64, "size": 134459248}
    assert AUDIT.lfs_pointer(b"\x1f\x8b binary") is None
    globs = ["vaft/data/samples/39915/omas.json.gz"]
    assert not AUDIT.in_wheel("vaft/data/samples/39915/omas.json.gz", globs)
    assert AUDIT.in_wheel("vaft/data/wheel_samples/39915/omas.json.gz", globs)


def test_inventory_reads_the_tree_through_git(repo):
    inventory = AUDIT.build_inventory("HEAD")

    samples = {row["shot"]: row for row in inventory["samples"]}
    assert samples[11111]["representations"][0]["sha256_matches"] is True
    assert samples[22222]["representations"][0]["sha256_matches"] is False

    lfs = _group(inventory, "efit.green_tables_lfs")
    assert lfs["tree"]["lfs_pointers"] == 1
    assert lfs["tree"]["lfs_object_bytes"] == 134459248

    reps = _group(inventory, "samples.shot.representations")
    assert reps["consumers"]["files"] == ["test/test_reader.py"]
    assert _group(inventory, "samples.shot.manifests")["tree"]["wheel_files"] == 2

    assert inventory["unassigned_large_files"] == [{"path": "big/blob.bin", "bytes": 4096}]
    assert _group(inventory, "history.gh_pages")["tree"] == "not applicable"


def test_inventory_check_detects_drift(repo, tmp_path):
    output = tmp_path / "inventory.yaml"
    assert AUDIT.main(["inventory", "--output", str(output)]) == 0
    assert AUDIT.main(["inventory", "--check", "--output", str(output)]) == 0
    output.write_text(output.read_text() + "# edited\n")
    assert AUDIT.main(["inventory", "--check", "--output", str(output)]) == 1


def test_measure_attributes_history_without_network(repo):
    result = AUDIT.measure(["HEAD"], network=False)
    tree = result["trees"]["HEAD"]
    assert tree["lfs_object_bytes"] == 134459248
    assert "pypi" not in result
    # The fixture repository has no origin refs; the history scope is then empty.
    assert result["history"]["refs"] == 0
