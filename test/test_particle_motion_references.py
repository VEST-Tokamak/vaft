"""Provenance and freshness of docs/assets/particle-motion (cold review 0.9.0 E-plot-diagram F1).

The published rows are numbers the orbit code produced. The manifest must name
the commit and the digest of every scientific source they depend on, the
documentation catalog gate must refuse a manifest whose sources changed, and a
re-traced row must still match what is published.
"""

import json
import re
import shutil

import pytest

from vaft.diagram import _particle_references as references
from vaft.diagram._particle_references import (
    EXAMPLES,
    ROOT,
    SCHEMA_VERSION,
    check_references,
    scientific_source_paths,
)

MANIFEST = ROOT / "docs/assets/particle-motion/manifest.json"


def _copy_tree(tmp_path):
    """The committed manifest and the sources it names, under a scratch root."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        destination = tmp_path / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source["path"], destination)
    target = tmp_path / "docs/assets/particle-motion/manifest.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(MANIFEST, target)
    return target, manifest


def _write(target, manifest):
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_committed_manifest_records_commit_and_matching_source_digests():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["generated_commit"])
    assert [s["path"] for s in manifest["sources"]] == scientific_source_paths()
    # The code every published number depends on is named explicitly.
    assert {
        "vaft/formula/particle.py",
        "vaft/data/particle.py",
        "vaft/process/particle.py",
        "vaft/process/particle_fields.py",
        "vaft/plot/particle.py",
        "vaft/diagram/_particle_states.py",
        "vaft/diagram/_guiding_center.py",
        "vaft/diagram/_ripple.py",
        "workflow/particle_motion/run.py",
    } <= set(scientific_source_paths())
    rows = check_references()
    assert [(r["id"], r["case"], r["builder"]) for r in rows] == list(EXAMPLES)


def test_a_recorded_digest_that_no_longer_matches_names_the_source(tmp_path):
    target, manifest = _copy_tree(tmp_path)
    victim = next(s for s in manifest["sources"] if s["path"] == "vaft/process/particle.py")
    victim["sha256"] = "0" * 64
    _write(target, manifest)
    with pytest.raises(ValueError, match=r"Stale particle-motion scientific source: vaft/process/particle\.py"):
        check_references(tmp_path)


def test_an_edited_source_is_reported_as_stale(tmp_path):
    target, _ = _copy_tree(tmp_path)
    assert check_references(tmp_path)
    source = tmp_path / "vaft/formula/particle.py"
    source.write_bytes(source.read_bytes() + b"\n# a later change to the Boris step\n")
    with pytest.raises(ValueError, match=r"Stale particle-motion scientific source: vaft/formula/particle\.py"):
        check_references(tmp_path)


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.pop("generated_commit"), "generated_commit"),
    (lambda m: m.__setitem__("generated_commit", "HEAD"), "generated_commit"),
    (lambda m: m["sources"].pop(), "sources drift"),
    (lambda m: m.__setitem__("schema_version", 1), "manifest version"),
    (lambda m: m["examples"].pop(), "examples drift"),
])
def test_missing_or_tampered_provenance_is_refused(tmp_path, mutate, message):
    target, manifest = _copy_tree(tmp_path)
    mutate(manifest)
    _write(target, manifest)
    with pytest.raises(ValueError, match=message):
        check_references(tmp_path)


def test_re_traced_cheap_rows_match_the_published_numbers():
    # gyration and the parabolic mirror trace in about a second; the tokamak
    # cases take ~8 s each and are left to ``run.py --check``.
    assert len(check_references(recompute=("cartesian", "mirror", "invariants"))) == 6


def test_a_published_number_the_code_no_longer_produces_is_reported(tmp_path):
    target, manifest = _copy_tree(tmp_path)
    row = next(r for r in manifest["examples"] if r["id"] == "mirror")
    assert row["bounce_cycles"], "the mirror example publishes a bounce period"
    row["bounce_cycles"][0]["period_s"] *= 1.01
    _write(target, manifest)
    with pytest.raises(ValueError, match=r"Stale particle-motion reference: mirror bounce_cycles"):
        check_references(tmp_path, recompute=("mirror",))


def test_platform_roundoff_in_a_cancellation_residual_is_not_staleness(tmp_path):
    # closure_relative is a cancellation residual of an adaptive trace: another
    # platform's last-ulp mass or charge moves it by ~1e-10 while the physics
    # is unchanged, so it is compared to an absolute noise floor (cold review
    # of the provenance fix, F1). A real change in the published period still
    # fails (the test above).
    target, manifest = _copy_tree(tmp_path)
    row = next(r for r in manifest["examples"] if r["id"] == "mirror")
    cycle = row["bounce_cycles"][0]
    cycle["closure_relative"] += 5e-10
    cycle["arc_velocity_relative_error"] -= 5e-10
    _write(target, manifest)
    assert len(check_references(tmp_path, recompute=("mirror",))) == 6
    cycle["closure_relative"] += 1e-8
    _write(target, manifest)
    with pytest.raises(ValueError, match=r"Stale particle-motion reference: mirror bounce_cycles"):
        check_references(tmp_path, recompute=("mirror",))


def test_the_manifest_directory_can_be_checked_on_its_own(tmp_path):
    target, manifest = _copy_tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    victim = next(s for s in manifest["sources"] if s["path"] == "workflow/particle_motion/run.py")
    victim["sha256"] = "f" * 64
    _write(elsewhere / "manifest.json", manifest)
    assert check_references(tmp_path)  # the tree's own manifest is untouched
    with pytest.raises(ValueError, match=r"Stale particle-motion scientific source: workflow/particle_motion/run\.py"):
        check_references(tmp_path, directory=elsewhere)


def test_the_documentation_catalog_gate_runs_the_check(monkeypatch):
    import vaft.diagram.docs_catalog as catalog

    def refuse(root):
        raise ValueError("Stale particle-motion scientific source: vaft/process/particle.py")

    monkeypatch.setattr(references, "check_references", refuse)
    with pytest.raises(ValueError, match="Stale particle-motion scientific source"):
        catalog.documentation_snapshot()
