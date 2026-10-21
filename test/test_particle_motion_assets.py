"""Particle-motion reference assets: integrity, skip-unchanged writes and --check (#2117)."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from workflow.particle_motion import run

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/particle-motion"


LOCK = ROOT / "docs/media.lock.json"


def _tracked():
    listed = subprocess.run(
        ["git", "ls-files", "docs/assets/particle-motion"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return {Path(path).name for path in listed}


def test_committed_files_match_their_manifest_digests():
    """Fast integrity check, no regeneration or network.

    Every artifact is either committed with the recorded digest, or kept outside
    git and pinned in docs/media.lock.json with that same digest (#2117), so the
    lock cannot silently fall behind a regeneration.
    """
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    locked = json.loads(LOCK.read_text(encoding="utf-8"))["files"]
    tracked = _tracked() - {"manifest.json"}
    for name, entry in artifacts.items():
        assert entry["kind"] == ("data" if name.endswith(".json") else "render")
        key = f"assets/particle-motion/{name}"
        if key in locked:
            assert name not in tracked, f"{name} is locked outside git but also committed"
            assert locked[key]["sha256"] == entry["sha256"], f"{name}: lock is stale; run media.py lock"
            assert locked[key]["bytes"] == entry["bytes"], name
        else:
            assert name in tracked, f"{name} is neither committed nor locked"
            data = (ASSETS / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
            assert len(data) == entry["bytes"], name
    assert tracked <= set(artifacts), "a committed file the manifest does not record"


@pytest.fixture
def renderer(monkeypatch):
    """Replace the expensive physics + TeX render with a controllable stub."""
    files = {"a.gif": b"GIF89a-1", "a.gif.json": b'{"x":1}\n', "a.svg": b"<svg/>"}

    def render(output, frames, dpi):
        for name, data in files.items():
            (output / name).write_bytes(data)
        artifacts = {
            p.name: {"sha256": run._sha256(p.read_bytes()), "bytes": p.stat().st_size,
                     "kind": run._kind(p.name)}
            for p in sorted(output.iterdir())
        }
        document = {"examples": [{"id": "a"}], "artifacts": artifacts}
        (output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
        return document

    monkeypatch.setattr(run, "_render", render)
    return files


def test_an_unchanged_regeneration_writes_nothing(tmp_path, renderer):
    run.generate(tmp_path)
    stamps = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    run.generate(tmp_path)
    assert {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()} == stamps

    renderer["a.gif.json"] = b'{"x":2}\n'
    run.generate(tmp_path)
    after = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    assert after["a.gif.json"] != stamps["a.gif.json"]
    assert after["a.gif"] == stamps["a.gif"] and after["a.svg"] == stamps["a.svg"]


def test_files_the_generator_no_longer_produces_are_removed(tmp_path, renderer):
    (tmp_path / "old.svg").write_bytes(b"<svg/>")
    run.generate(tmp_path)
    assert not (tmp_path / "old.svg").exists()


def test_a_fresh_tree_checks_clean(tmp_path, renderer):
    run.generate(tmp_path)
    assert run.check(tmp_path) == ([], [])


def test_stale_data_is_an_error(tmp_path, renderer):
    run.generate(tmp_path)
    renderer["a.gif.json"] = b'{"x":2}\n'
    problems, _ = run.check(tmp_path)
    assert any("a.gif.json: stale" in line for line in problems)


def test_render_drift_is_a_warning_unless_strict(tmp_path, renderer):
    run.generate(tmp_path)
    renderer["a.gif"] = b"GIF89a-2"
    problems, warnings = run.check(tmp_path)
    assert problems == []
    assert any("a.gif: render drift" in line for line in warnings)
    assert any(line.startswith("toolchain here:") for line in warnings)
    problems, _ = run.check(tmp_path, strict=True)
    assert any("a.gif: render drift" in line for line in problems)


def test_a_hand_edited_file_fails_integrity(tmp_path, renderer):
    run.generate(tmp_path)
    (tmp_path / "a.svg").write_bytes(b"<svg>edited</svg>")
    problems, _ = run.check(tmp_path)
    assert any("a.svg: does not match its manifest digest" in line for line in problems)


def test_missing_and_orphaned_files_are_named(tmp_path, renderer):
    run.generate(tmp_path)
    (tmp_path / "a.gif").unlink()
    (tmp_path / "stray.png").write_bytes(b"x")
    problems, _ = run.check(tmp_path)
    assert any("a.gif: missing" in line for line in problems)
    assert any("stray.png: orphaned" in line for line in problems)
