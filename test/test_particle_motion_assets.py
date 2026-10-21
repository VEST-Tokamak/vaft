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
    files = _Files({"a.gif": b"GIF89a-1", "a.gif.json": b'{"x":1}\n', "a.svg": b"<svg/>"})
    toolchain = {"value": {"stub": "renderer"}}  # differs from this machine's

    def render(output, frames, dpi):
        for name, data in files.items():
            (output / name).write_bytes(data)
        artifacts = {
            p.name: {"sha256": run._sha256(p.read_bytes()), "bytes": p.stat().st_size,
                     "kind": run._kind(p.name)}
            for p in sorted(output.iterdir())
        }
        document = {"generator": run.GENERATOR, "examples": [{"id": "a"}],
                    "artifacts": artifacts, "toolchain": toolchain["value"]}
        (output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
        return document

    monkeypatch.setattr(run, "_render", render)
    files.toolchain = toolchain
    return files


class _Files(dict):
    pass


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
    run.generate(tmp_path)
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


def test_with_the_recorded_toolchain_a_render_difference_is_stale(tmp_path, renderer):
    """A plotting change must not hide as toolchain drift on the machine that rendered."""
    renderer.toolchain["value"] = run._toolchain()
    run.generate(tmp_path)
    renderer["a.svg"] = b"<svg>replotted</svg>"
    problems, warnings = run.check(tmp_path)
    assert any("a.svg: stale" in line for line in problems)
    assert not any("render drift" in line for line in warnings)


def test_a_directory_the_generator_does_not_own_is_refused(tmp_path, renderer):
    (tmp_path / "committed.svg").write_bytes(b"<svg/>")
    with pytest.raises(SystemExit, match="refusing"):
        run.generate(tmp_path)
    assert (tmp_path / "committed.svg").exists(), "nothing was pruned"


def test_check_names_files_outside_the_artifacts_table(tmp_path, renderer):
    run.generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["artifacts"]["a.svg"]
    manifest["artifacts"]["gone.png"] = {"sha256": "0" * 64, "bytes": 1, "kind": "render"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    problems, _ = run.check(tmp_path)
    assert any("a.svg: present but not recorded" in line for line in problems)
    assert any("gone.png: recorded in the manifest but not present" in line for line in problems)


def test_dotfiles_are_neither_orphans_nor_artifacts(tmp_path, renderer):
    run.generate(tmp_path)
    (tmp_path / ".DS_Store").write_bytes(b"finder")
    run.generate(tmp_path)
    assert (tmp_path / ".DS_Store").exists()
    assert run.check(tmp_path)[0] == []
