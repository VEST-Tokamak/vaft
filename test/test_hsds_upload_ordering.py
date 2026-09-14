"""The master is the commit point of a shot write, so it is written last.

A shot folder holds one `<ids>.h5` per IDS plus a `master.h5` whose external
links are what a reader resolves the shot's contents from -- not the folder
listing. Uploading is file-by-file and cannot be made atomic, so the order
decides what a crash leaves behind.

Two windows used to leave previously published IDS unreachable:

* the upload order was plain alphabetical, so `master.h5` went up ahead of part
  of the new payload for four of the seven replicating stages;
* the pre-write master's links were merged back in *after* the write, so the
  remote master described only the in-flight stage for the length of a
  fetch-merge-upload round trip.

These pin both shut. What they deliberately do not claim is transactionality:
an interrupted write may still leave its own payload half-uploaded.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import h5py
import pytest

from vaft.database import ods as ods_module
from vaft.database.ods import MASTER_FILENAME, upload_order
from vaft.database.sources import STAGE_REPLICATION
from vaft.database.staging import external_h5_links


def _master(path: Path, links) -> Path:
    with h5py.File(path, "w") as handle:
        for name in links:
            handle[name] = h5py.ExternalLink(f"{name}.h5", f"/{name}")
    return path


def _payload(directory: Path, names) -> None:
    for name in names:
        with h5py.File(directory / f"{name}.h5", "w") as handle:
            handle.create_dataset(name, data=[1, 2, 3])


class FakeRemote:
    """A shot folder that records upload order and can fail on demand."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploaded: list[str] = []
        self.fail_on: str | None = None

    def upload(self, local_path: Path, remote_uri: str) -> str:
        name = Path(local_path).name
        if self.fail_on is not None and name == self.fail_on:
            self.uploaded.append(f"!{name}")
            raise RuntimeError(f"simulated crash while uploading {name}")
        shutil.copy2(local_path, self.root / name)
        self.uploaded.append(name)
        return remote_uri

    def links(self) -> list[str] | None:
        master = self.root / MASTER_FILENAME
        return external_h5_links(master) if master.exists() else None


@pytest.fixture
def remote(tmp_path, monkeypatch):
    fake = FakeRemote(tmp_path / "remote")
    monkeypatch.setattr(ods_module, "_upload_local_image", fake.upload)
    return fake


# --------------------------------------------------------------------------- #
# the invariant itself
# --------------------------------------------------------------------------- #


def test_master_is_ordered_last_for_every_replicating_stage():
    """Asserted from the table, not from a list of stage names.

    `efit`, `impa` and `chease` used to pass only because their IDS names sort
    before the string `master`. Iterating `STAGE_REPLICATION` means a stage
    added later cannot quietly reintroduce the hazard.
    """
    checked = 0
    for stage, entry in STAGE_REPLICATION.items():
        if entry.source is None or not entry.ids:
            continue
        files = [f"{name}.h5" for name in entry.ids] + [
            "dataset_description.h5",
            MASTER_FILENAME,
        ]
        assert upload_order(files)[-1] == MASTER_FILENAME, stage
        checked += 1
    assert checked, "no replicating stage was checked; the table shape changed"


def test_payload_keeps_a_deterministic_order():
    order = upload_order(
        ["master.h5", "tf.h5", "barometry.h5", "dataset_description.h5"]
    )
    assert order == [
        "barometry.h5",
        "dataset_description.h5",
        "tf.h5",
        "master.h5",
    ]


def test_a_shot_with_no_master_is_left_alone():
    assert upload_order(["equilibrium.h5", "a.h5"]) == ["a.h5", "equilibrium.h5"]


# --------------------------------------------------------------------------- #
# what a crash leaves behind
# --------------------------------------------------------------------------- #


def test_a_crash_mid_payload_leaves_the_previous_master_intact(tmp_path, remote):
    """Requirement: failure after one or more payload files, before the master."""
    _master(remote.root / MASTER_FILENAME, ["magnetics", "pf_active"])
    for name in ("magnetics", "pf_active"):
        (remote.root / f"{name}.h5").touch()
    before = remote.links()

    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    # pf_active sorts after master.h5: under the old alphabetical order the
    # master would already be remote by the time this upload failed.
    _payload(shot_dir, ["pf_active", "tf"])
    _master(shot_dir / MASTER_FILENAME, ["pf_active", "tf"])
    remote.fail_on = "tf.h5"

    with pytest.raises(RuntimeError, match="simulated crash"):
        ods_module._upload_local_shot(shot_dir=shot_dir, directory="main", shot=1)

    assert MASTER_FILENAME not in remote.uploaded
    assert remote.links() == before, "the shot still reads as it did before"


def test_a_crash_just_before_the_master_still_leaves_the_shot_readable(tmp_path, remote):
    """Requirement: failure immediately before the final master update."""
    _master(remote.root / MASTER_FILENAME, ["magnetics"])
    (remote.root / "magnetics.h5").touch()

    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["pf_active"])
    _master(shot_dir / MASTER_FILENAME, ["pf_active"])
    remote.fail_on = MASTER_FILENAME

    with pytest.raises(RuntimeError, match="simulated crash"):
        ods_module._upload_local_shot(shot_dir=shot_dir, directory="main", shot=1)

    # The payload landed, but the commit point did not, so the shot still
    # resolves to exactly what it held before.
    assert (remote.root / "pf_active.h5").exists()
    assert remote.links() == ["magnetics.h5"]


def test_an_unrelated_ids_survives_an_interrupted_write(tmp_path, remote):
    """Requirement: a previously published unrelated IDS stays reachable."""
    _master(remote.root / MASTER_FILENAME, ["magnetics", "pf_active"])
    for name in ("magnetics", "pf_active"):
        (remote.root / f"{name}.h5").touch()

    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["spectrometer_uv", "tf"])
    _master(shot_dir / MASTER_FILENAME, ["spectrometer_uv", "tf"])
    remote.fail_on = "tf.h5"

    with pytest.raises(RuntimeError):
        ods_module._upload_local_shot(shot_dir=shot_dir, directory="main", shot=1)

    assert set(remote.links()) == {"magnetics.h5", "pf_active.h5"}


# --------------------------------------------------------------------------- #
# the master that lands is never stage-only
# --------------------------------------------------------------------------- #


def test_the_master_is_merged_before_it_is_uploaded(tmp_path, remote):
    """The commit point publishes the union, not this stage alone.

    Merging after the write would leave the remote master describing only the
    in-flight stage for a whole round trip. Here the hook runs on the local
    master first, so the remote never observes that state.
    """
    previous = _master(tmp_path / "previous.h5", ["magnetics", "pf_active"])
    for name in ("magnetics", "pf_active"):
        (remote.root / f"{name}.h5").touch()

    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["tf"])
    _master(shot_dir / MASTER_FILENAME, ["tf"])

    observed: list[list[str]] = []

    def finalize(local_master: Path) -> None:
        from vaft.database.staging import merge_master_links

        merge_master_links(
            local_master,
            previous,
            present_files=[p.name for p in remote.root.glob("*.h5")],
        )
        observed.append(external_h5_links(local_master))

    ods_module._upload_local_shot(
        shot_dir=shot_dir, directory="main", shot=1, finalize_master=finalize
    )

    assert observed, "the hook never ran"
    assert set(observed[0]) == {"tf.h5", "magnetics.h5", "pf_active.h5"}
    # Every remote state the master ever had names all three.
    assert set(remote.links()) == {"tf.h5", "magnetics.h5", "pf_active.h5"}
    assert remote.uploaded[-1] == MASTER_FILENAME


def test_the_hook_runs_after_the_payload_is_already_remote(tmp_path, remote):
    """It may consult the remote listing, so the payload must be there first."""
    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["tf"])
    _master(shot_dir / MASTER_FILENAME, ["tf"])

    seen: list[list[str]] = []
    ods_module._upload_local_shot(
        shot_dir=shot_dir,
        directory="main",
        shot=1,
        finalize_master=lambda _p: seen.append(sorted(remote.uploaded)),
    )

    assert seen == [["tf.h5"]]


# --------------------------------------------------------------------------- #
# retry and idempotency
# --------------------------------------------------------------------------- #


def test_a_retry_after_a_crash_reaches_the_clean_write_state(tmp_path, remote):
    """Requirement: a successful retry equals a clean write."""
    _master(remote.root / MASTER_FILENAME, ["magnetics"])
    (remote.root / "magnetics.h5").touch()

    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["pf_active", "tf"])

    def fresh_master() -> None:
        _master(shot_dir / MASTER_FILENAME, ["pf_active", "tf"])

    def finalize(local_master: Path) -> None:
        from vaft.database.staging import merge_master_links

        merge_master_links(
            local_master,
            _master(tmp_path / "prev.h5", ["magnetics"]),
            present_files=[p.name for p in remote.root.glob("*.h5")],
        )

    fresh_master()
    remote.fail_on = "tf.h5"
    with pytest.raises(RuntimeError):
        ods_module._upload_local_shot(
            shot_dir=shot_dir, directory="main", shot=1, finalize_master=finalize
        )

    # The state *between* the attempts is the half this test is named for: while
    # the retry is pending the shot must still resolve to what it held before.
    # Asserting only the end state passes whatever the upload order is, because
    # the retry converges either way.
    assert remote.links() == ["magnetics.h5"]

    remote.fail_on = None
    fresh_master()
    ods_module._upload_local_shot(
        shot_dir=shot_dir, directory="main", shot=1, finalize_master=finalize
    )

    assert set(remote.links()) == {"pf_active.h5", "tf.h5", "magnetics.h5"}
    assert (remote.root / "tf.h5").exists()


def test_publishing_twice_is_idempotent(tmp_path, remote):
    """Requirement: repeated publication stays idempotent."""
    shot_dir = tmp_path / "local"
    shot_dir.mkdir()
    _payload(shot_dir, ["pf_active", "tf"])
    _master(shot_dir / MASTER_FILENAME, ["pf_active", "tf"])

    ods_module._upload_local_shot(shot_dir=shot_dir, directory="main", shot=1)
    first_links = remote.links()
    first_names = sorted(p.name for p in remote.root.glob("*.h5"))

    ods_module._upload_local_shot(shot_dir=shot_dir, directory="main", shot=1)

    assert remote.links() == first_links
    assert sorted(p.name for p in remote.root.glob("*.h5")) == first_names


# --------------------------------------------------------------------------- #
# the hook is refused where it could not run
# --------------------------------------------------------------------------- #


def test_a_local_write_refuses_the_hook_instead_of_ignoring_it(tmp_path):
    """A local write uploads nothing, so the hook would never fire."""
    from omas import ODS

    from vaft.database.ods import save_ods

    with pytest.raises(TypeError, match="env='server' only"):
        save_ods(
            ODS(),
            1,
            env="local",
            path=str(tmp_path),
            finalize_master=lambda _p: None,
        )


def test_the_native_ids_path_refuses_the_hook(monkeypatch):
    """The IMAS writer manages its own master, so the hook cannot apply.

    Type detection is patched rather than satisfied with a real `IDSToplevel`:
    what is under test is the guard, not `_is_imas_ids`.
    """
    import vaft.database as database

    monkeypatch.setattr(database, "_is_imas_ids", lambda obj: True)

    with pytest.raises(TypeError, match="OMAS write path only"):
        database.save(object(), 1, finalize_master=lambda _p: None)
