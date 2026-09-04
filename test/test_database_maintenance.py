"""Stripping IMPA from shots published before the split (issue #305).

This rewrites published baseline data, so the safety property under test is not
"it removes the channels" but "it refuses to move a surviving one". k-files and
the EFIT constraint builder address probes positionally: a shift would silently
re-point them at the wrong sensor, which is worse than leaving the residue.
"""

from __future__ import annotations

import pytest
from omas import ODS

from vaft.database import maintenance
from vaft.database.maintenance import (
    ImpaStripError,
    inspect_impa_residue,
    strip_impa_from_source,
)


def _published(impa_at_tail: bool = True) -> ODS:
    """A `main`-shaped magnetics product with the array appended, or interleaved."""
    ods = ODS(consistency_check=False)
    names = ["MD_A", "MD_B", "impa:IMPA Bz 01", "impa:IMPA Bz 02"]
    if not impa_at_tail:
        names = ["MD_A", "impa:IMPA Bz 01", "MD_B", "impa:IMPA Bz 02"]
    for index, identifier in enumerate(names):
        ods[f"magnetics.b_field_pol_probe.{index}.identifier"] = identifier
        ods[f"magnetics.b_field_pol_probe.{index}.name"] = identifier
    ods["magnetics.b_field_tor_probe.0.identifier"] = "impa:IMPA 01"
    return ods


def test_a_tail_block_is_reported_as_removable():
    residue = inspect_impa_residue(_published())
    assert residue.carries_impa and residue.removable
    assert residue["nodes"]["magnetics.b_field_pol_probe"] == [2, 3]


def test_an_interleaved_block_is_refused_with_the_reason():
    residue = inspect_impa_residue(_published(impa_at_tail=False))
    assert residue.carries_impa and not residue.removable
    assert "would move a surviving index" in residue["refusals"][0]


def test_a_clean_product_reports_nothing_to_do():
    ods = ODS(consistency_check=False)
    ods["magnetics.b_field_pol_probe.0.identifier"] = "MD_A"
    residue = inspect_impa_residue(ods)
    assert not residue.carries_impa and not residue.removable


@pytest.fixture
def published(monkeypatch):
    state = {"ods": _published(), "saved": []}

    monkeypatch.setattr(
        maintenance, "_sources", __import__("vaft.database.sources", fromlist=["x"])
    )
    monkeypatch.setattr(
        "vaft.database.load", lambda shot, **kwargs: state["ods"]
    )
    monkeypatch.setattr(
        "vaft.database.save",
        lambda ods, shot, **kwargs: state["saved"].append((shot, kwargs.get("source"), ods)),
    )
    monkeypatch.setattr(
        "vaft.database.replication._fetch_remote_master", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "vaft.database.replication.merge_remote_master", lambda *a, **k: ()
    )
    return state


def test_a_dry_run_reports_without_writing(published):
    report = strip_impa_from_source(39915, apply=False)

    assert report["carries_impa"] and report["removed"] == 3
    assert report["applied"] is False
    assert published["saved"] == []
    assert len(published["ods"]["magnetics.b_field_pol_probe"]) == 4


def test_applying_truncates_and_leaves_every_surviving_index_in_place(published):
    report = strip_impa_from_source(39915, apply=True)

    assert report["applied"] and report["removed"] == 3
    probes = published["ods"]["magnetics.b_field_pol_probe"]
    assert len(probes) == 2
    assert [str(probes[index]["name"]) for index in range(2)] == ["MD_A", "MD_B"]
    assert "b_field_tor_probe" not in published["ods"]["magnetics"]
    assert [entry[1] for entry in published["saved"]] == ["main"]


def test_an_interleaved_block_is_refused_before_any_write(published):
    published["ods"] = _published(impa_at_tail=False)

    with pytest.raises(ImpaStripError, match="Refusing to strip"):
        strip_impa_from_source(39915, apply=True)
    assert published["saved"] == []


def test_the_read_only_legacy_source_is_never_repaired():
    from vaft.database.sources import ReadOnlySourceError

    with pytest.raises(ReadOnlySourceError):
        strip_impa_from_source(39915, source="public")
