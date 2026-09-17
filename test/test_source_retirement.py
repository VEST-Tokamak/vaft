"""Retiring `chease-mhd-stability` without losing what only it holds (#94, step 9).

The gate matters more than the migration here, because the migration is
deliberately partial: the stability half of the combined source **cannot** be
copied forward. A faithful copy would have to say which product each result came
from, and the combined product does not record it -- DCON's two edge treatments
write the same fields at the same position, and RDCON's and STRIDE's `ntms`
surfaces are appended to one AOS with the solver named only on the whole IDS.

So the stability results are regenerated rather than moved, and the deletion
gate checks for that. These tests pin the refusals, because a retirement that
deletes the last copy of something is not recoverable by re-running it.
"""

import pytest

from vaft.database import retirement
from vaft.database.retirement import (
    RetirementError,
    RetirementReport,
    ShotRetirement,
    delete_retired_source,
    plan_retirement,
)


pytestmark = pytest.mark.core


@pytest.fixture
def holdings(monkeypatch):
    """A fake HSDS: `{(shot, source): [ids, ...]}`, patched at the read seam."""
    store: dict[tuple[int, str], list[str]] = {}

    def fake_has_ids(shot, source, ids):
        held = store.get((int(shot), source), [])
        return any(name in held for name in ids)

    monkeypatch.setattr(retirement, "_has_ids", fake_has_ids)
    return store


def _report(*rows: ShotRetirement) -> RetirementReport:
    return RetirementReport(source="chease-mhd-stability", applied=False, shots=rows)


def test_a_shot_whose_refinement_moved_and_stability_was_rerun_is_deletable(holdings):
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium", "mhd_linear"]
    holdings[(39915, "main/chease")] = ["equilibrium"]
    holdings[(39915, "main/chease/rdcon")] = ["mhd_linear", "ntms"]

    report = plan_retirement([39915])

    (row,) = report.shots
    assert row.refinement == "already-present"
    assert row.stability == "superseded"
    assert row.deletable
    assert report.deletable


def test_stability_that_was_never_regenerated_blocks_deletion(holdings):
    """The case the gate exists for.

    Those results cannot be copied forward -- nothing records which product
    produced them -- so deleting the source would be the only copy going away.
    """
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium", "mhd_linear", "ntms"]
    holdings[(39915, "main/chease")] = ["equilibrium"]

    report = plan_retirement([39915])

    (row,) = report.shots
    assert row.stability == "not-regenerated"
    assert not row.deletable
    assert not report.deletable
    assert "cannot be attributed to a product" in row.detail

    with pytest.raises(RetirementError, match="still hold data only this source has"):
        delete_retired_source(report)


def test_a_refinement_that_was_never_copied_blocks_deletion(holdings):
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium"]

    report = plan_retirement([39915])

    (row,) = report.shots
    assert row.refinement == "not-copied"
    assert not report.deletable


def test_a_shot_the_old_source_never_held_does_not_block(holdings):
    """Absence is not a blocker; there is nothing to lose."""
    report = plan_retirement([41524])

    (row,) = report.shots
    assert row.refinement == "absent" and row.stability == "absent"
    assert row.deletable


def test_an_empty_report_is_not_deletable():
    """"No shots examined" and "every shot safe" are different findings.

    A gate that cannot tell them apart would green-light a source nobody looked
    at, which is the most expensive way to be wrong here.
    """
    report = _report()

    assert not report.deletable
    with pytest.raises(RetirementError, match="no shots were examined"):
        delete_retired_source(report)


def test_one_blocking_shot_stops_the_whole_deletion(holdings):
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium"]
    holdings[(39915, "main/chease")] = ["equilibrium"]
    holdings[(41524, "chease-mhd-stability")] = ["mhd_linear"]

    report = plan_retirement([39915, 41524])

    assert [row.shot for row in report.blocking] == [41524]
    assert not report.deletable


def test_a_clean_plan_still_will_not_delete_from_inside_vaft(holdings):
    """The last step is an administrator's, and says so.

    Removing a whole namespace is not something a library call should be able to
    do as a side effect of a maintenance script, however clean the plan looks.
    """
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium"]
    holdings[(39915, "main/chease")] = ["equilibrium"]
    report = plan_retirement([39915])
    assert report.deletable

    # The dry run says what would happen...
    assert "would delete" in delete_retired_source(report)["detail"]

    # ...and applying it hands the operator the command instead of running it.
    with pytest.raises(RetirementError, match="hsdel /chease-mhd-stability/"):
        delete_retired_source(report, apply=True)


def test_the_report_serializes_for_review(holdings):
    holdings[(39915, "chease-mhd-stability")] = ["equilibrium", "mhd_linear"]
    holdings[(39915, "main/chease")] = ["equilibrium"]

    payload = plan_retirement([39915]).to_dict()

    assert payload["dry_run"] is True
    assert payload["deletable"] is False
    assert payload["summary"] == {"shots": 1, "deletable": 0, "blocking": 1}
    import json

    json.dumps(payload, allow_nan=False)


def test_copying_a_refinement_verifies_the_destination_before_reporting_success(
    monkeypatch,
):
    """A copy that trusts the write is how a retirement deletes the only copy.

    The destination is read back and checked; if nothing arrives, the call
    raises and says the source has not been changed.
    """
    from omas import ODS

    written: list[tuple[int, str]] = []

    def fake_load(shot, *, source=None, paths=None, **kwargs):
        ods = ODS(consistency_check=False)
        if source == "chease-mhd-stability" or (shot, source) in written:
            ods["equilibrium.ids_properties.comment"] = "refinement"
        return ods

    monkeypatch.setattr("vaft.database.load", fake_load)
    monkeypatch.setattr(
        "vaft.database.save",
        lambda ods, shot, *, source=None, **kwargs: written.append((shot, source)),
    )

    report = retirement.copy_refinement(39915, apply=True)

    assert report["applied"] and report["verified"]
    assert written == [(39915, "main/chease")]


def test_a_copy_that_does_not_arrive_raises_rather_than_reporting_success(monkeypatch):
    from omas import ODS

    def fake_load(shot, *, source=None, paths=None, **kwargs):
        ods = ODS(consistency_check=False)
        if source == "chease-mhd-stability":
            ods["equilibrium.ids_properties.comment"] = "refinement"
        return ods  # the destination stays empty

    monkeypatch.setattr("vaft.database.load", fake_load)
    monkeypatch.setattr("vaft.database.save", lambda *a, **k: None)

    with pytest.raises(RetirementError, match="read nothing back"):
        retirement.copy_refinement(39915, apply=True)


def test_the_dry_run_copies_nothing(monkeypatch):
    from omas import ODS

    saves: list = []
    monkeypatch.setattr(
        "vaft.database.load",
        lambda shot, *, source=None, paths=None, **kwargs: (
            lambda o: (o.__setitem__("equilibrium.ids_properties.comment", "x"), o)[1]
        )(ODS(consistency_check=False)),
    )
    monkeypatch.setattr("vaft.database.save", lambda *a, **k: saves.append(a))

    report = retirement.copy_refinement(39915)

    assert saves == []
    assert report["applied"] is False
    assert "would copy" in report["detail"]
