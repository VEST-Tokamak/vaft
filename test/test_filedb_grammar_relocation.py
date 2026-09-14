"""Moving a canonical tree onto the lineage-bearing grammar (#527 step 4).

A FileDB written before `family`/`refinement`/`product` became path segments
resolves to nothing afterwards: `FileDB.efit(shot, family="magnetic")` looks
under `efit/magnetic/{shot}` and the data is at `efit/{shot}`. Nothing warns --
the resolver returns a path, the path is simply empty -- so the relocation is
what makes an existing deployment readable again.

It is a *relocation*, not a rewrite. Directories are renamed; no file in the
tree is opened, read, hashed or written, which is what bounds the damage an
interrupted run can do to "some subtrees moved, the rest did not".
"""

import json

import pytest

from vaft.database.filedb import (
    FileDB,
    FileDBPathError,
    audit_filedb_grammar,
    relocate_filedb_grammar,
)


pytestmark = pytest.mark.core


SHOT = 39915


def _old_grammar_tree(root):
    """A canonical tree as it was written before the lineage segments existed."""
    payload = {
        f"efit/{SHOT}/output/g0{SHOT}.00325": "gfile",
        f"efit/{SHOT}/metadata/efit_status.txt": "completed",
        f"chease/{SHOT}/output/refined_gfiles_generated.txt": "1",
        f"gpec/dcon/{SHOT}/n=1/work/dcon.out": "dcon n=1",
        f"gpec/dcon/{SHOT}/n=2/work/dcon.out": "dcon n=2",
        f"gpec/rdcon/{SHOT}/n=1/metadata/status.txt": "completed",
        f"gpec/ideal-gpec/{SHOT}/n=1/work/gpec.out": "gpec",
        f"omas/efit/{SHOT}/output/efit.json": "{}",
        f"omas/chease/{SHOT}/output/chease.json": "{}",
        # Stages that belong to no family must be left exactly where they are.
        f"omas/diagnostics/{SHOT}/output/diagnostics.json": "{}",
        f"raw/{SHOT}/vest_{SHOT}_daq_raw.json.gz": "raw",
        "omas/static/vest-2019/output/static.json": "{}",
    }
    for relative, text in payload.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return payload


def test_the_plan_names_every_subtree_that_has_to_move(tmp_path):
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)

    report = audit_filedb_grammar(root)

    moved = {item.source: item.target for item in report.relocations}
    assert moved == {
        f"efit/{SHOT}": f"efit/magnetic/{SHOT}",
        f"chease/{SHOT}": f"chease/magnetic/{SHOT}",
        # The whole code subtree travels in one rename, so every (shot, mode)
        # cell under it moves at once rather than one rename per cell.
        "gpec/dcon": "gpec/magnetic/chease/dcon",
        "gpec/rdcon": "gpec/magnetic/chease/rdcon",
        "gpec/ideal-gpec": "gpec/magnetic/chease/ideal-gpec",
        f"omas/efit/{SHOT}": f"omas/efit/magnetic/{SHOT}",
        f"omas/chease/{SHOT}": f"omas/chease/magnetic/{SHOT}",
    }
    assert report.safe_to_apply


def test_a_dry_run_changes_nothing(tmp_path):
    root = tmp_path / "FileDB"
    before = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in [root / k for k in _old_grammar_tree(root)]
    }

    audit_filedb_grammar(root)
    relocate_filedb_grammar(root)  # apply defaults to False

    after = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_the_moved_tree_resolves_through_the_canonical_resolver(tmp_path):
    """The point of the move: what the resolver asks for is where the data is."""
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)

    relocate_filedb_grammar(root, apply=True)

    db = FileDB(root)
    assert (
        db.efit(SHOT, family="magnetic", artifact="output") / f"g0{SHOT}.00325"
    ).read_text(encoding="utf-8") == "gfile"
    assert (
        db.chease(SHOT, family="magnetic", artifact="output")
        / "refined_gfiles_generated.txt"
    ).exists()
    for mode, text in ((1, "dcon n=1"), (2, "dcon n=2")):
        cell = db.gpec(
            "dcon", SHOT, mode, family="magnetic", refinement="chease", artifact="work"
        )
        assert (cell / "dcon.out").read_text(encoding="utf-8") == text
    assert db.omas_product("efit", shot=SHOT, family="magnetic").exists()


def test_a_dcon_cell_keeps_the_legacy_product_rather_than_an_edge_branch(tmp_path):
    """The tree records no edge treatment, so neither branch can be asserted.

    Filing it as `dcon-peeling` would put provenance in the tree that the move
    cannot verify -- the run may have truncated at the dW peak, and nothing on
    disk says which.
    """
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)

    relocate_filedb_grammar(root, apply=True)

    assert (root / "gpec/magnetic/chease/dcon").is_dir()
    assert not (root / "gpec/magnetic/chease/dcon-peeling").exists()
    assert not (root / "gpec/magnetic/chease/dcon-kink").exists()


def test_stages_that_belong_to_no_family_are_left_alone(tmp_path):
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)

    relocate_filedb_grammar(root, apply=True)

    assert (root / f"omas/diagnostics/{SHOT}/output/diagnostics.json").exists()
    assert (root / f"raw/{SHOT}/vest_{SHOT}_daq_raw.json.gz").exists()
    assert (root / "omas/static/vest-2019/output/static.json").exists()
    assert not (root / "omas/diagnostics/magnetic").exists()


def test_running_it_twice_is_a_no_op(tmp_path):
    """A shot directory and a family directory can never be confused.

    No `EquilibriumFamily`, `Refinement` or `StabilityProduct` value is all
    digits, so `efit/39915` is only ever the old shape and `efit/magnetic` only
    ever the new one. Without that the second run would nest the tree again.
    """
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)
    relocate_filedb_grammar(root, apply=True)
    settled = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))

    second = relocate_filedb_grammar(root, apply=True)

    assert second.relocations == ()
    assert set(second.already_canonical) >= {"efit/magnetic", "gpec/magnetic"}
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == settled


def test_a_half_migrated_tree_resumes_rather_than_starting_over(tmp_path):
    """An interrupted run leaves some subtrees moved and some not.

    Directories are renamed one at a time, so that state is reachable -- a
    killed process, a full disk, an operator's Ctrl-C. Re-running has to finish
    the job, not refuse because part of it is done and not nest what already
    moved.
    """
    root = tmp_path / "FileDB"
    (root / "gpec/magnetic/chease/rdcon" / str(SHOT) / "n=1/work").mkdir(parents=True)
    (root / "gpec/dcon" / str(SHOT) / "n=1/work").mkdir(parents=True)
    (root / "efit/magnetic" / str(SHOT) / "output").mkdir(parents=True)
    (root / "efit/41524/output").mkdir(parents=True)

    report = audit_filedb_grammar(root)

    assert {item.source for item in report.relocations} == {"efit/41524", "gpec/dcon"}
    assert set(report.already_canonical) == {"efit/magnetic", "gpec/magnetic"}
    assert report.safe_to_apply

    relocate_filedb_grammar(root, apply=True)

    assert (root / "efit/magnetic/41524/output").is_dir()
    assert (root / f"gpec/magnetic/chease/dcon/{SHOT}/n=1/work").is_dir()
    # What had already moved is where it was, not nested a second time.
    assert (root / f"gpec/magnetic/chease/rdcon/{SHOT}/n=1/work").is_dir()
    assert not (root / "gpec/magnetic/chease/magnetic").exists()


def test_an_occupied_destination_stops_the_move_rather_than_merging(tmp_path):
    """Two subtrees claiming one destination were written by different runs.

    Which is authoritative is not a question a path rewriter can answer, so it
    refuses -- and refuses before moving anything, so the tree is not left half
    migrated.
    """
    root = tmp_path / "FileDB"
    _old_grammar_tree(root)
    occupied = root / f"efit/magnetic/{SHOT}/output/g0{SHOT}.00325"
    occupied.parent.mkdir(parents=True)
    occupied.write_text("a different run wrote this", encoding="utf-8")

    report = audit_filedb_grammar(root)
    assert report.collisions == (f"efit/{SHOT}",)
    assert not report.safe_to_apply

    with pytest.raises(FileDBPathError, match="Refusing to relocate"):
        relocate_filedb_grammar(root, apply=True)

    # Nothing moved, including the subtrees that had no collision.
    assert (root / f"chease/{SHOT}/output/refined_gfiles_generated.txt").exists()
    assert occupied.read_text(encoding="utf-8") == "a different run wrote this"


def test_the_cli_dry_runs_by_default_and_reports_collisions_in_its_exit_code(
    tmp_path, capsys
):
    from vaft.cli.filedb import main

    root = tmp_path / "FileDB"
    _old_grammar_tree(root)

    assert main(["relocate", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["summary"]["relocations"] == 7
    assert (root / f"efit/{SHOT}/output/g0{SHOT}.00325").exists(), "dry run moved data"

    (root / f"efit/magnetic/{SHOT}").mkdir(parents=True)
    # An unappliable plan is a result, not a crash -- but the exit code has to
    # say so, or an operator scripting the dry run reads collisions as success.
    assert main(["relocate", str(root)]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["collisions"] == 1


def test_an_unreadable_root_is_named_rather_than_silently_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        audit_filedb_grammar(tmp_path / "does-not-exist")
