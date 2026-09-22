"""EntryStore contract for the DD layer (#1132): MemoryStore and IMASStore.

A store knows ``(IDS, occurrence)`` and nothing semantic; these tests hold
both implementations to the same behaviour, and hold IMASStore to what a raw
``DBEntry`` reports for the same entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

imas = pytest.importorskip("imas")

from vaft.imas import EntryStore, IMASStore, MemoryStore, MutableEntryStore, StoreSet  # noqa: E402
from vaft.imas._dd._store import stored_names  # noqa: E402

SAMPLE_39915 = Path(__file__).resolve().parents[1] / "vaft" / "data" / "samples" / "39915" / "imas.nc"


def _equilibrium(factory, name: str = ""):
    ids = factory.equilibrium()
    ids.ids_properties.homogeneous_time = 2
    ids.ids_properties.name = name
    return ids


def _write_entry(path: Path, dd_version: str, layout) -> Path:
    """Write ``[(ids, occurrence, name), ...]`` as a native IMAS HDF5 entry."""
    with imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version=dd_version) as entry:
        for ids_name, occurrence, name in layout:
            ids = getattr(entry.factory, ids_name)()
            ids.ids_properties.homogeneous_time = 2
            ids.ids_properties.name = name
            entry.put(ids, occurrence)
    return path


SPARSE = [("equilibrium", 0, "a"), ("equilibrium", 3, ""), ("equilibrium", 4, "b"), ("magnetics", 0, "routine")]


@pytest.fixture
def sparse_entry(tmp_path) -> Path:
    return _write_entry(tmp_path / "entry", "3.41.0", SPARSE)


def test_both_stores_satisfy_the_protocols(sparse_entry):
    memory = MemoryStore("3.41.0")
    with IMASStore(sparse_entry) as native:
        assert isinstance(memory, MutableEntryStore)
        assert isinstance(native, EntryStore)
        assert not native.writable


def test_memory_store_is_keyed_by_occurrence_and_copies():
    store = MemoryStore("3.41.0")
    ids = _equilibrium(store.factory, "a")
    store.put(ids, 4)
    ids.ids_properties.name = "changed after put"
    assert store.occurrences("equilibrium") == (4,)
    assert store.has("equilibrium", 4) and not store.has("equilibrium", 0)
    assert store.available_ids() == ("equilibrium",)
    first = store.get("equilibrium", 4)
    assert str(first.ids_properties.name) == "a"
    assert first is not store.get("equilibrium", 4)
    store.delete("equilibrium", 4)
    assert store.occurrences("equilibrium") == ()


def test_memory_store_converts_at_the_boundary():
    store = MemoryStore("4.0.0")
    store.put(_equilibrium(imas.IDSFactory("3.41.0"), "a"), 0)
    assert store.get("equilibrium", 0)._dd_version == "4.0.0"


def test_memory_store_refuses_a_missing_occurrence():
    with pytest.raises(KeyError, match="stored occurrences"):
        MemoryStore("3.41.0").get("equilibrium", 1)


def test_imas_store_discovers_sparse_occurrences_like_dbentry(sparse_entry):
    with imas.DBEntry(f"imas:hdf5?path={sparse_entry}", "r", dd_version="3.41.0") as raw:
        expected = [int(v) for v in raw.list_all_occurrences("equilibrium")]
    with IMASStore(sparse_entry) as store:
        assert store.dd_version == "3.41.0"
        assert list(store.occurrences("equilibrium")) == expected == [0, 3, 4]
        assert store.available_ids() == ("equilibrium", "magnetics")
        assert store.occurrence_names("equilibrium") == {0: "a", 3: "", 4: "b"}
        lazy = store.get("equilibrium", 4, lazy=True)
        assert str(lazy.ids_properties.name) == "b"


def test_imas_store_reads_a_dd4_entry_at_its_own_version(tmp_path):
    # Opening DD-3 data at the default DD-4 version raises in imas-python; the
    # store must open an entry at the version it was written with.
    entry = _write_entry(tmp_path / "dd4", "4.0.0", [("equilibrium", 2, "k")])
    with IMASStore(entry) as store:
        assert store.dd_version == "4.0.0"
        assert store.occurrence_names("equilibrium") == {2: "k"}


def test_imas_store_reads_the_packaged_netcdf_sample():
    with IMASStore(SAMPLE_39915) as store:
        assert "equilibrium" in store.available_ids()
        assert store.occurrences("equilibrium") == (0,)
        assert store.occurrence_names("equilibrium") == {0: ""}


def test_a_second_store_can_open_an_entry_already_open(sparse_entry):
    with IMASStore(sparse_entry) as first, IMASStore(sparse_entry) as second:
        assert first.occurrences("equilibrium") == second.occurrences("equilibrium")


def test_writable_entry_puts_one_occurrence_and_keeps_siblings(sparse_entry):
    with IMASStore.writable_entry(sparse_entry, mode="a", dd_version="3.41.0") as store:
        store.put(_equilibrium(imas.IDSFactory("3.41.0"), "c"), 7)
    with IMASStore(sparse_entry) as store:
        assert store.occurrence_names("equilibrium") == {0: "a", 3: "", 4: "b", 7: "c"}


def test_read_only_store_refuses_writes(sparse_entry):
    with IMASStore(sparse_entry) as store, pytest.raises(PermissionError):
        store.put(_equilibrium(imas.IDSFactory("3.41.0")), 1)


def test_stored_names_are_returned_verbatim_not_interpreted(sparse_entry):
    with IMASStore(sparse_entry) as store:
        assert stored_names(store, "equilibrium")[3] == ""


def test_store_set_owns_and_closes_its_stores():
    a, b = MemoryStore("3.41.0"), MemoryStore("3.41.0")
    stores = StoreSet({"main": a}, kinetic=b)
    assert list(stores) == ["main", "kinetic"]
    stores.close()
    assert stores.closed and not a.writable and not b.writable
    with pytest.raises(RuntimeError):
        stores["main"]
    with pytest.raises(KeyError, match="no store named"):
        StoreSet(main=MemoryStore("3.41.0"))["missing"]
    with pytest.raises(TypeError):
        StoreSet(main=object())


def test_a_closed_store_does_not_silently_reopen(sparse_entry):
    store = IMASStore(sparse_entry)
    store.close()
    for call in (lambda: store.occurrences("equilibrium"), lambda: store.get("equilibrium", 0),
                 lambda: store.available_ids()):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_sparse_netcdf_occurrences_and_the_layout_fallback(tmp_path):
    path = tmp_path / "sparse.nc"
    with imas.DBEntry(str(path), "w", dd_version="3.41.0") as raw:
        raw.put(_equilibrium(raw.factory, "only"), 3)
    with IMASStore(path) as store:
        assert store.occurrences("equilibrium") == (3,)
        # The layout used when the Access Layer cannot list occurrences reads
        # netCDF's <ids>/<n> groups, not "occurrence 0 of every key".
        assert store.handle._layout["equilibrium"] == (3,)
