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


def test_inspecting_a_shot_does_not_invent_the_probe_array_it_lacks():
    """Probing an absent node materializes it, and `--apply` saves this ODS.

    A pre-split `main` shot with only poloidal probes would otherwise gain an
    empty `b_field_tor_probe` from being repaired.
    """
    ods = _published()
    del ods["magnetics.b_field_tor_probe"]

    residue = inspect_impa_residue(ods)
    assert "b_field_tor_probe" not in ods["magnetics"]

    from vaft.database.maintenance import _strip

    _strip(ods, residue)
    assert sorted(ods["magnetics"].keys()) == ["b_field_pol_probe"]


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


# --------------------------------------------------------------------------- #
# the rewrite must never leave `main/{shot}` stage-only (cold review data F2)
# --------------------------------------------------------------------------- #


class _FakeRemote:
    """A shot folder on a fake HSDS: real ``master.h5`` bytes, listed files."""

    def __init__(self, tmp_path, monkeypatch, *, linked, files):
        import shutil

        import h5py

        from vaft.database import replication

        self.master = tmp_path / "remote-master.h5"
        self.files = list(files)
        self.events: list[str] = []
        with h5py.File(self.master, "w") as handle:
            for name in linked:
                handle[name] = h5py.ExternalLink(f"./{name}.h5", name)

        def fetch(source, shot, target):
            self.events.append("fetch")
            shutil.copy2(self.master, target)
            return target

        def hsload(local, uri):
            self.events.append("hsload master")
            shutil.copy2(local, self.master)

        monkeypatch.setattr(replication, "_fetch_remote_master", fetch)
        monkeypatch.setattr(
            replication,
            "_require_remote_entries",
            lambda source, shot: tuple(sorted(self.files + ["master.h5"])),
        )
        monkeypatch.setattr(
            replication, "_remote_entries", replication._require_remote_entries
        )
        monkeypatch.setattr("vaft.database.transport.run_hsload", hsload)
        monkeypatch.setattr(
            "vaft.database.transport.verify_uploaded_image", lambda *a, **k: None
        )

    def save(self, ods, shot, *, source=None, finalize_master=None, **kwargs):
        """What `vaft.database.save` does: a magnetics-only master, sent last."""
        import shutil
        import tempfile
        from pathlib import Path

        import h5py

        self.events.append("save")
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / "master.h5"
            with h5py.File(local, "w") as handle:
                handle["magnetics"] = h5py.ExternalLink("./magnetics.h5", "magnetics")
            if finalize_master is not None:
                finalize_master(local)
            shutil.copy2(local, self.master)

    def links(self):
        from vaft.database.staging import external_h5_links

        return external_h5_links(self.master)


_UNION = ["equilibrium", "magnetics", "pf_active"]
_UNION_FILES = [f"{name}.h5" for name in _UNION]


def test_the_master_is_merged_before_it_lands_so_a_failed_net_costs_nothing(
    tmp_path, monkeypatch
):
    remote = _FakeRemote(tmp_path, monkeypatch, linked=_UNION, files=_UNION_FILES)
    monkeypatch.setattr("vaft.database.load", lambda shot, **kwargs: _published())
    monkeypatch.setattr("vaft.database.save", remote.save)

    def net_fails(*args, **kwargs):
        raise RuntimeError("hsget failed")

    monkeypatch.setattr("vaft.database.replication.merge_remote_master", net_fails)

    with pytest.raises(maintenance.ImpaStripWriteError, match="hsget failed"):
        strip_impa_from_source(39915, apply=True)

    assert remote.links() == _UNION_FILES


def test_the_batch_report_says_when_a_failed_shot_was_already_being_written(
    tmp_path, monkeypatch
):
    remote = _FakeRemote(tmp_path, monkeypatch, linked=_UNION, files=_UNION_FILES)
    monkeypatch.setattr("vaft.database.load", lambda shot, **kwargs: _published())
    monkeypatch.setattr("vaft.database.save", remote.save)
    monkeypatch.setattr(
        "vaft.database.replication.merge_remote_master",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("hsget failed")),
    )

    (report,) = maintenance.strip_impa_from_shots([39915], apply=True)

    assert report["applied"] is False and report["write_started"] is True


def _stripped() -> ODS:
    ods = ODS(consistency_check=False)
    ods["magnetics.b_field_pol_probe.0.identifier"] = "MD_A"
    return ods


def test_a_rerun_repairs_a_master_an_interrupted_run_left_stage_only(
    tmp_path, monkeypatch
):
    remote = _FakeRemote(
        tmp_path, monkeypatch, linked=["magnetics"], files=_UNION_FILES
    )
    monkeypatch.setattr("vaft.database.load", lambda shot, **kwargs: _stripped())
    monkeypatch.setattr("vaft.database.save", remote.save)

    dry = strip_impa_from_source(39915, apply=False)
    assert dry["master_unlinked"] == ["equilibrium.h5", "pf_active.h5"]
    assert dry["master_repaired"] is False
    assert remote.links() == ["magnetics.h5"]

    report = strip_impa_from_source(39915, apply=True)

    assert report["master_repaired"] is True
    assert remote.links() == _UNION_FILES
    assert "save" not in remote.events, "nothing to strip, so no payload rewrite"

    import h5py

    with h5py.File(remote.master, "r") as handle:
        link = handle.get("equilibrium", getlink=True)
        assert (link.filename, link.path) == ("./equilibrium.h5", "equilibrium")


def test_a_whole_master_is_left_alone_on_a_rerun(tmp_path, monkeypatch):
    remote = _FakeRemote(tmp_path, monkeypatch, linked=_UNION, files=_UNION_FILES)
    monkeypatch.setattr("vaft.database.load", lambda shot, **kwargs: _stripped())

    report = strip_impa_from_source(39915, apply=True)

    assert report["master_unlinked"] == [] and report["master_repaired"] is False
    assert "hsload master" not in remote.events
