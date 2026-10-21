"""Particle-motion reference assets: integrity, skip-unchanged writes and --check (#2117)."""

import hashlib
import json
from pathlib import Path

import pytest

from workflow.particle_motion import run

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/particle-motion"


def test_committed_files_match_their_manifest_digests():
    """Fast integrity check, no regeneration: what is committed is what was generated."""
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    present = {p.name for p in ASSETS.iterdir() if p.is_file()} - {"manifest.json"}
    assert set(artifacts) == present
    for name, entry in artifacts.items():
        data = (ASSETS / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
        assert len(data) == entry["bytes"], name
        assert entry["kind"] == ("data" if name.endswith(".json") else "render")


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
