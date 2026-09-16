"""The EFIT -> CHEASE -> stability chain is checkable, not merely recorded (#527 step 6).

Before this, every stage recorded the *path* of what it consumed. A path
identifies the file that was read only if that file never changed afterwards --
exactly the assumption `replication.is_reusable` exists so that replication does
not have to make, and one provenance was still making. "Which EFIT produced this
CHEASE" was answerable only by trusting a path.

Each stage now records digests, and the digests join, so the question becomes a
comparison. These tests hold the three properties that make the answer worth
having: a well-formed chain verifies, a chain across two runs is caught, and a
chain with nothing recorded reports that rather than passing.
"""

import pytest

from vaft.database.provenance import NOT_RECORDED, verify_chain


pytestmark = pytest.mark.core


EFIT_GFILE = "a" * 64
REFINED_A = "b" * 64
EFIT_GFILE_2 = "c" * 64
REFINED_B = "d" * 64


def _manifest(*pairs):
    return {
        "input": [
            {
                "path": f"g0{i}",
                "sha256": consumed,
                "output_sha256": produced,
                "status": "completed",
            }
            for i, (consumed, produced) in enumerate(pairs)
        ]
    }


def test_a_well_formed_chain_verifies_end_to_end():
    report = verify_chain(
        chease_manifest=_manifest((EFIT_GFILE, REFINED_A)),
        stability_input_sha256=REFINED_A,
        efit_gfile_sha256=EFIT_GFILE,
        shot=39915,
    )

    assert report.verified
    assert report.broken == ()
    assert report.unrecorded == ()
    assert [link.stage for link in report.links] == ["chease", "stability"]


def test_a_stability_run_on_another_shots_equilibrium_is_caught():
    """The failure this exists to make visible.

    Two products can sit in the right directories, carry the right shot, and
    still not be the same analysis -- a CHEASE rerun between the refinement and
    the solve is enough. Paths cannot see that; digests can.
    """
    report = verify_chain(
        chease_manifest=_manifest((EFIT_GFILE, REFINED_A)),
        stability_input_sha256=REFINED_B,
    )

    assert not report.verified
    assert len(report.broken) == 1
    assert "different runs" in report.broken[0].detail


def test_a_chain_with_nothing_recorded_does_not_pass():
    """An absent digest is not agreement.

    Products written before the chain existed carry "". Reporting them as
    verified would make the check useless exactly where it matters most -- on
    the archive that predates it.
    """
    report = verify_chain(
        chease_manifest=_manifest((NOT_RECORDED, NOT_RECORDED)),
        stability_input_sha256=REFINED_A,
    )

    assert not report.verified
    assert len(report.unrecorded) == 1
    assert report.broken == ()
    assert "predates the provenance chain" in report.unrecorded[0].detail


def test_any_refined_slice_of_the_shot_satisfies_the_stability_link():
    """A CHEASE stage refines every time slice; a stability cell runs on one.

    Requiring a particular entry would fail every well-formed chain.
    """
    manifest = _manifest((EFIT_GFILE, REFINED_A), (EFIT_GFILE_2, REFINED_B))

    for consumed in (REFINED_A, REFINED_B):
        assert verify_chain(
            chease_manifest=manifest, stability_input_sha256=consumed
        ).verified


def test_the_efit_link_is_optional_and_absent_means_unchecked_not_passed():
    """Omitting the EFIT digest checks one link, not zero, and says so."""
    manifest = _manifest((EFIT_GFILE, REFINED_A))

    without = verify_chain(chease_manifest=manifest, stability_input_sha256=REFINED_A)
    assert without.verified
    assert [link.stage for link in without.links] == ["stability"]

    wrong = verify_chain(
        chease_manifest=manifest,
        stability_input_sha256=REFINED_A,
        efit_gfile_sha256=EFIT_GFILE_2,
    )
    assert not wrong.verified
    assert wrong.broken[0].upstream_stage == "efit"


def test_an_empty_manifest_reports_nothing_verified_rather_than_everything():
    report = verify_chain(chease_manifest={}, stability_input_sha256=REFINED_A)

    assert not report.verified
    assert len(report.unrecorded) == 1


def test_the_report_serializes_for_a_stage_manifest():
    payload = verify_chain(
        chease_manifest=_manifest((EFIT_GFILE, REFINED_A)),
        stability_input_sha256=REFINED_A,
        efit_gfile_sha256=EFIT_GFILE,
        shot=39915,
    ).to_dict()

    assert payload["verified"] is True
    assert payload["shot"] == 39915
    assert payload["summary"] == {
        "links": 2,
        "verified": 2,
        "mismatch": 0,
        "not_recorded": 0,
    }
    import json

    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# The producers: each stage has to record what the verifier reads.
# ---------------------------------------------------------------------------


def test_chease_records_the_digest_of_what_it_read_and_what_it_wrote(tmp_path):
    """Both ends of the middle link, taken from real files."""
    import sys

    workflow = "workflow/automatic_pipeline_1_routine_data_processing"
    sys.path.insert(0, workflow)
    try:
        import run_chease_refinement as chease
    finally:
        sys.path.remove(workflow)

    gfile = tmp_path / "g039915.00325"
    gfile.write_text("EFITD reconstruction", encoding="utf-8")

    digest = chease._sha256(gfile)

    assert len(digest) == 64
    assert digest == chease._sha256(gfile), "hashing must be deterministic"
    # Provenance is about a run, never a reason to fail one.
    assert chease._sha256(tmp_path / "absent") == NOT_RECORDED
    assert chease._sha256(None) == NOT_RECORDED


def test_the_chease_stage_manifest_carries_an_input_block(tmp_path):
    """It carried none at all, which is why the chain could not be walked."""
    import sys

    workflow = "workflow/automatic_pipeline_1_routine_data_processing"
    sys.path.insert(0, workflow)
    try:
        import generate_chease_ods as stage
    finally:
        sys.path.remove(workflow)

    runs = {
        "records": [
            {
                "input": "/db/efit/magnetic/39915/output/g039915.00325",
                "input_sha256": EFIT_GFILE,
                "output_sha256": REFINED_A,
                "status": "completed",
            },
            {"input": "/db/.../g039915.00999", "status": "missing_input"},
        ]
    }

    block = stage._provenance_inputs(runs)

    assert block[0] == {
        "path": "/db/efit/magnetic/39915/output/g039915.00325",
        "sha256": EFIT_GFILE,
        "output_sha256": REFINED_A,
        "status": "completed",
    }
    # A record from before the digests existed says "not recorded" rather than
    # claiming a hash nobody took.
    assert block[1]["sha256"] == NOT_RECORDED
    assert block[1]["status"] == "missing_input"

    # And the block the stage writes is what the verifier reads.
    assert verify_chain(
        chease_manifest={"input": block},
        stability_input_sha256=REFINED_A,
        efit_gfile_sha256=EFIT_GFILE,
    ).verified


def test_the_stability_suite_records_the_equilibrium_it_consumed(tmp_path):
    from vaft.code.gpec import _equilibrium_sha256

    geqdsk = tmp_path / "g039915.00325"
    geqdsk.write_text("CHEASE refined", encoding="utf-8")

    assert len(_equilibrium_sha256(geqdsk)) == 64
    assert _equilibrium_sha256(tmp_path / "absent") == NOT_RECORDED
