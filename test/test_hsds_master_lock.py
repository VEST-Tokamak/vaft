"""Concurrent writes of one shot keep every IDS linked from its master (#913).

The remote side is an in-memory HSDS: a folder per ``(source, shot)`` holding
file bytes. Everything between the replication call and the upload of each
file is the real code -- ``replicate_stage``, ``save_ods``'s locked publish,
the merging finalizers -- so the race and its fix are exercised, not modelled.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import time

import h5py
import pytest
from omas import ODS

from vaft.database import _master_lock, maintenance, ods as ods_module, replication
from vaft.database.filedb import FileDB
from vaft.database.staging import external_h5_links

SHOT = 39620


class FakeHSDS:
    def __init__(self):
        self.folders: dict[tuple[str, int], dict[str, bytes]] = {}
        self.payload_delay = 0.0
        #: When set, every payload upload waits here: two writers meet inside
        #: the window between their master read and their master replace.
        self.barrier: threading.Barrier | None = None
        self._guard = threading.Lock()

    @staticmethod
    def parse(uri: str) -> tuple[str, int, str]:
        head, name = uri[len("hdf5://"):].rsplit("/", 1)
        source, shot = head.rsplit("/", 1)
        return source, int(shot), name

    def entries(self, source, shot):
        with self._guard:
            return tuple(sorted(self.folders.get((source, int(shot)), {})))

    def get(self, uri, target):
        source, shot, name = self.parse(uri)
        with self._guard:
            data = self.folders[(source, shot)][name]
        Path(target).write_bytes(data)
        return Path(target)

    def put(self, local, uri):
        source, shot, name = self.parse(uri)
        if name != "master.h5":
            if self.barrier is not None:
                self.barrier.wait()
            time.sleep(self.payload_delay)  # the window between the master read and its replacement
        data = Path(local).read_bytes()
        with self._guard:
            self.folders.setdefault((source, shot), {})[name] = data
        return uri

    def master_links(self, source, shot, tmp_path):
        target = tmp_path / f"read-{time.monotonic_ns()}.h5"
        self.get(f"hdf5://{source}/{shot}/master.h5", target)
        return external_h5_links(target)


def _write_staged_shot(directory: Path, ids_names) -> None:
    """What the IMAS writer leaves for one write: one file per IDS and a master."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ids_names:
        with h5py.File(directory / f"{name}.h5", "w") as handle:
            handle.create_group(name)
    with h5py.File(directory / "master.h5", "w") as master:
        for name in ids_names:
            master[name] = h5py.ExternalLink(f"{name}.h5", f"/{name}")


@pytest.fixture
def hsds(monkeypatch, tmp_path):
    fake = FakeHSDS()
    monkeypatch.setenv(_master_lock.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setattr(replication, "_remote_entries", fake.entries)
    monkeypatch.setattr(replication, "_require_remote_entries", fake.entries)
    monkeypatch.setattr("vaft.database.transport.run_hsget", fake.get)
    monkeypatch.setattr("vaft.database.transport.run_hsload", fake.put)
    monkeypatch.setattr("vaft.database.transport.verify_uploaded_image", lambda *a, **k: None)
    monkeypatch.setattr(ods_module, "_upload_local_image", lambda local, uri: fake.put(local, uri))
    monkeypatch.setattr("vaft.database.utils.require_source_exists", lambda source: None)
    monkeypatch.setattr(ods_module, "ensure_shot_folder", lambda source, shot: None)
    monkeypatch.setattr(replication, "_round_trip", lambda ods, **kwargs: {"passed": True})

    staging_root = tmp_path / "staging"

    def fake_save(ods, shot, *, source=None, occurrence=None, finalize_master=None, **kwargs):
        """`vaft.database.save` up to the IMAS writer; the real locked publish after it."""
        names = sorted(key for key in ods.keys())
        directory = staging_root / f"{source}-{shot}-{'-'.join(names)}-{threading.get_ident()}"
        _write_staged_shot(directory, names)
        ods_module._publish_staged_shot(directory, source, int(shot), finalize_master)
        return f"hdf5://{source}/{shot}/"

    monkeypatch.setattr("vaft.database.save", fake_save)
    return fake


@pytest.fixture
def two_stages(tmp_path):
    """Completed diagnostics and eddy products of one shot, as the 2026-09-17 batch had."""
    from vaft.omas import save as save_local

    db = FileDB(tmp_path / "filedb")
    for stage, ids in (("diagnostics", "magnetics"), ("eddy", "pf_passive")):
        ods = ODS(consistency_check=False)
        ods[f"{ids}.ids_properties.comment"] = f"owned by {stage}"
        product = db.omas_product(stage, shot=SHOT)
        product.parent.mkdir(parents=True, exist_ok=True)
        save_local(ods, product)
        manifest = db.omas_manifest(stage, shot=SHOT)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"stage": stage, "status": "success"}), encoding="utf-8")
    return db


def _replicate_concurrently(db):
    errors = []

    def run(stage):
        try:
            replication.replicate_stage(stage, SHOT, filedb=db, attempts=1)
        except Exception as error:  # pragma: no cover - surfaced below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(stage,)) for stage in ("diagnostics", "eddy")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


# --------------------------------------------------------------------------- #
# the race and its fix
# --------------------------------------------------------------------------- #
def test_without_the_fix_concurrent_stages_lose_a_link(hsds, two_stages, monkeypatch, tmp_path):
    """The control: the pre-#913 path (no lock, no re-read) reproduces the loss."""
    monkeypatch.setattr(replication, "shot_master_lock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(_master_lock, "shot_master_lock", lambda *a, **k: nullcontext())
    monkeypatch.setattr(ods_module, "_merge_current_master", lambda source, shot, then=None: then or (lambda m: None))
    hsds.barrier = threading.Barrier(2, timeout=30)  # both payloads in flight at once: the 09-17 overlap
    _replicate_concurrently(two_stages)
    linked = hsds.master_links("main", SHOT, tmp_path)
    assert set(hsds.entries("main", SHOT)) >= {"magnetics.h5", "pf_passive.h5"}
    assert not {"magnetics.h5", "pf_passive.h5"} <= set(linked), linked


def test_concurrent_stages_of_one_shot_keep_both_links(hsds, two_stages, tmp_path):
    hsds.payload_delay = 0.3
    _replicate_concurrently(two_stages)
    linked = set(hsds.master_links("main", SHOT, tmp_path))
    assert {"magnetics.h5", "pf_passive.h5"} <= linked


def test_the_reread_alone_keeps_a_link_written_by_a_writer_outside_the_lock(hsds, monkeypatch, tmp_path):
    """A master changed meanwhile (another host, a hand-run hsload) is merged, not overwritten."""
    first = tmp_path / "first"
    _write_staged_shot(first, ["magnetics"])
    ods_module._publish_staged_shot(first, "main", SHOT)

    second = tmp_path / "second"
    _write_staged_shot(second, ["pf_passive"])

    def outsider(local_master):
        # Lands after this write's payload and before its master replaces the
        # stored one, without taking the lock.
        hsds.put(_write_and_return(tmp_path / "outsider", ["equilibrium"], "equilibrium.h5"),
                 f"hdf5://main/{SHOT}/equilibrium.h5")
        with h5py.File(tmp_path / "outsider" / "master.h5", "w") as master:
            for name in ("magnetics", "equilibrium"):
                master[name] = h5py.ExternalLink(f"{name}.h5", f"/{name}")
        hsds.put(tmp_path / "outsider" / "master.h5", f"hdf5://main/{SHOT}/master.h5")

    original = ods_module._merge_current_master

    def with_outsider(source, shot, then=None):
        merge = original(source, shot, then)

        def finalize(local_master):
            outsider(local_master)
            merge(local_master)

        return finalize

    monkeypatch.setattr(ods_module, "_merge_current_master", with_outsider)
    ods_module._publish_staged_shot(second, "main", SHOT)
    assert {"magnetics.h5", "pf_passive.h5", "equilibrium.h5"} <= set(hsds.master_links("main", SHOT, tmp_path))


def _write_and_return(directory: Path, names, filename) -> Path:
    _write_staged_shot(directory, names)
    return directory / filename


def test_a_plain_save_merges_instead_of_replacing_the_master(hsds, tmp_path):
    """The corrective updaters call `database.save` with no finalizer; that used to
    replace the shot's master with one naming only their own IDS."""
    stored = tmp_path / "stored"
    _write_staged_shot(stored, ["magnetics", "pf_passive"])
    ods_module._publish_staged_shot(stored, "main", SHOT)

    update = tmp_path / "update"
    _write_staged_shot(update, ["core_profiles"])
    ods_module._publish_staged_shot(update, "main", SHOT)
    assert set(hsds.master_links("main", SHOT, tmp_path)) == {"magnetics.h5", "pf_passive.h5", "core_profiles.h5"}


# --------------------------------------------------------------------------- #
# the lock itself
# --------------------------------------------------------------------------- #
def test_one_shot_is_serialized_and_different_shots_are_not(monkeypatch, tmp_path):
    monkeypatch.setenv(_master_lock.LOCK_DIR_ENV, str(tmp_path))
    holding, release = threading.Event(), threading.Event()

    def holder():
        with _master_lock.shot_master_lock("main", 1):
            holding.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert holding.wait(10)
        with _master_lock.shot_master_lock("main", 2, timeout=5):
            pass  # another shot: no wait
        with pytest.raises(_master_lock.MasterLockTimeout):
            with _master_lock.shot_master_lock("main", 1, timeout=0.3):
                pass  # the same shot: waits, and here gives up
    finally:
        release.set()
        thread.join()
    with _master_lock.shot_master_lock("main", 1, timeout=5):
        pass  # free again once released


def test_lock_files_are_shareable_and_a_read_only_one_still_locks(monkeypatch, tmp_path):
    """Another account's lock file must not lock this one out (flock needs no write access)."""
    import os
    import stat

    monkeypatch.setenv(_master_lock.LOCK_DIR_ENV, str(tmp_path / "locks"))
    with _master_lock.shot_master_lock("main", 5):
        pass
    created = _master_lock.lock_path("main", 5)
    assert stat.S_IMODE(created.stat().st_mode) == 0o666
    if os.name == "posix" and os.geteuid() != 0:
        created.chmod(0o444)  # as if created read-only by another account
        with _master_lock.shot_master_lock("main", 5, timeout=5):
            pass


def test_the_lock_is_reentrant_within_a_thread(monkeypatch, tmp_path):
    monkeypatch.setenv(_master_lock.LOCK_DIR_ENV, str(tmp_path))
    with _master_lock.shot_master_lock("main", 7):
        with _master_lock.shot_master_lock("main", 7):
            pass
        with _master_lock.shot_master_lock("main", 7):
            pass


def test_a_held_lock_times_out_with_a_named_error(monkeypatch, tmp_path):
    monkeypatch.setenv(_master_lock.LOCK_DIR_ENV, str(tmp_path))
    holding = threading.Event()
    release = threading.Event()

    def holder():
        with _master_lock.shot_master_lock("main", 9):
            holding.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    holding.wait(5)
    try:
        with pytest.raises(_master_lock.MasterLockTimeout, match="shot 9"):
            with _master_lock.shot_master_lock("main", 9, timeout=0.3):
                pass
    finally:
        release.set()
        thread.join()


def test_the_lock_directory_is_fixed_not_tmpdir(monkeypatch, tmp_path):
    """Two writers with different $TMPDIR must still meet at one lock."""
    monkeypatch.delenv(_master_lock.LOCK_DIR_ENV, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert _master_lock.lock_path("main/chease/dcon-peeling", 48916) == Path(
        "/tmp/vaft-hsds-locks/main__chease__dcon-peeling.48916.lock"
    )


# --------------------------------------------------------------------------- #
# audit and repair
# --------------------------------------------------------------------------- #
def test_the_audit_finds_and_repairs_the_2026_09_17_state(hsds, monkeypatch, tmp_path):
    """39620 after the batch: every file present, the master linking two of them."""
    monkeypatch.setattr("vaft.database.sources.resolve", lambda source, writable=False, **k: source or "main")
    folder = tmp_path / "39620"
    _write_staged_shot(folder, ["dataset_description", "magnetics", "pf_active", "pf_passive"])
    for name in ("dataset_description", "magnetics", "pf_active", "pf_passive"):
        hsds.put(folder / f"{name}.h5", f"hdf5://main/{SHOT}/{name}.h5")
    with h5py.File(folder / "master.h5", "w") as master:
        for name in ("dataset_description", "pf_passive"):
            master[name] = h5py.ExternalLink(f"{name}.h5", f"/{name}")
    hsds.put(folder / "master.h5", f"hdf5://main/{SHOT}/master.h5")

    report = maintenance.audit_master_link(SHOT)
    assert report["status"] == maintenance.MASTER_LINKS_MISSING
    assert report["missing"] == ["magnetics.h5", "pf_active.h5"]

    assert maintenance.audit_master_link(SHOT, repair=True)["status"] == maintenance.MASTER_REPAIRED
    assert maintenance.audit_master_link(SHOT)["status"] == maintenance.MASTER_COMPLETE
    assert set(hsds.master_links("main", SHOT, tmp_path)) == {
        "dataset_description.h5", "magnetics.h5", "pf_active.h5", "pf_passive.h5"
    }


def test_the_audit_reports_absent_masterless_and_unreadable_shots(hsds, monkeypatch, tmp_path):
    monkeypatch.setattr("vaft.database.sources.resolve", lambda source, writable=False, **k: source or "main")
    assert maintenance.audit_master_link(1)["status"] == maintenance.MASTER_ABSENT

    folder = tmp_path / "2"
    _write_staged_shot(folder, ["magnetics"])
    hsds.put(folder / "magnetics.h5", "hdf5://main/2/magnetics.h5")
    report = maintenance.audit_master_link(2)
    assert report["status"] == maintenance.MASTER_MISSING and report["missing"] == ["magnetics.h5"]

    def busy(source, shot):
        raise OSError(503, "service unavailable")

    monkeypatch.setattr(replication, "_remote_entries", busy)
    report = maintenance.audit_master_link(3)
    assert report["status"] == maintenance.MASTER_UNREADABLE and "503" in report["error"]


def test_a_folder_of_derived_images_only_is_absent_not_masterless(hsds, monkeypatch, tmp_path):
    monkeypatch.setattr("vaft.database.sources.resolve", lambda source, writable=False, **k: source or "main")
    image = tmp_path / "magnetics.h5image.h5"
    image.write_bytes(b"image")
    hsds.put(image, "hdf5://main/4/magnetics.h5image.h5")
    report = maintenance.audit_master_link(4)
    assert report["status"] == maintenance.MASTER_ABSENT and "derived" in report["note"]


@pytest.mark.parametrize(
    ("status", "code"),
    [("complete", 0), ("absent", 0), ("repaired", 0), ("links_missing", 1), ("no_master", 1), ("unreadable", 1)],
)
def test_the_audit_command_fails_while_data_is_hidden(monkeypatch, status, code, capsys):
    from vaft.cli import maintenance as cli

    monkeypatch.setattr(
        "vaft.database.maintenance.audit_master_links",
        lambda shots, source=None, repair=False: [{"shot": 1, "status": status, "missing": ["a.h5"]}],
    )
    assert cli.main(["audit-masters", "--shots", "1"]) == code


class _FakeFolders:
    """h5pyd.Folder's create semantics: x creates only on 404, otherwise opens."""

    def __init__(self, existing=(), domains=(), refuse=None):
        self.existing = set(existing)
        self.domains = set(domains)
        self.refuse = refuse
        self.created = []

    def __call__(self, path, mode="r"):
        node = SimpleNamespace(_obj_class="folder")
        if path in self.domains:
            node._obj_class = "domain"
            return node
        if path in self.existing:
            return node
        if mode != "x":
            raise OSError(404, "Not Found")
        if self.refuse is not None:
            status = self.refuse
            if status == 409:
                # Another host created it between our GET and PUT.
                self.existing.add(path)
            raise OSError(status, "refused")
        self.existing.add(path)
        self.created.append(path)
        return node


def test_a_new_shot_folder_is_created_on_first_write(monkeypatch):
    from vaft.database import utils

    folders = _FakeFolders(existing={"/main/"})
    monkeypatch.setattr(utils.h5pyd, "Folder", folders)
    utils.ensure_shot_folder("main", 50001)
    utils.ensure_shot_folder("main", 50001)
    assert folders.created == ["/main/50001/"]


def test_losing_the_create_race_to_another_host_is_success(monkeypatch):
    from vaft.database import utils

    folders = _FakeFolders(refuse=409)
    monkeypatch.setattr(utils.h5pyd, "Folder", folders)
    utils.ensure_shot_folder("main", 50001)


def test_a_forbidden_create_names_the_fix(monkeypatch):
    from vaft.database import utils

    monkeypatch.setattr(utils.h5pyd, "Folder", _FakeFolders(refuse=403))
    with pytest.raises(utils.ShotFolderError, match="hstouch -o <owner> /main/50001/"):
        utils.ensure_shot_folder("main", 50001)


def test_a_shot_path_that_is_a_domain_is_refused(monkeypatch):
    from vaft.database import utils

    monkeypatch.setattr(utils.h5pyd, "Folder", _FakeFolders(domains={"/main/50001/"}))
    with pytest.raises(utils.ShotFolderError, match="domain, not a folder"):
        utils.ensure_shot_folder("main", 50001)


def test_the_publish_creates_the_folder_under_the_lock_before_uploading(hsds, monkeypatch, tmp_path):
    order = []
    held = []

    def ensure(source, shot):
        held.append(_master_lock._held.get((f"{source}:{shot}", threading.get_ident()), 0))
        order.append(("folder", source, shot))

    monkeypatch.setattr(ods_module, "ensure_shot_folder", ensure)
    monkeypatch.setattr(
        ods_module, "_upload_local_shot", lambda **kwargs: order.append(("upload",)) or []
    )
    staged = tmp_path / "new"
    _write_staged_shot(staged, ["magnetics"])
    ods_module._publish_staged_shot(staged, "main", 50001)
    assert order == [("folder", "main", 50001), ("upload",)]
    assert held == [1]
