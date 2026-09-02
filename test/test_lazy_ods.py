from __future__ import annotations

import numpy as np
import pytest

from vaft.database import lazy_ods
from vaft.database.lazy_ods import HSDSODS, HSDSStore, LazyODSClosedError


class FakeDataset:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype
        self.reads = []

    def __getitem__(self, selection):
        self.reads.append(selection)
        return self.value[selection]


class FakeGroup(dict):
    def items(self):
        return super().items()


class FakeFile(FakeGroup):
    def __init__(self, root_name, root):
        super().__init__({root_name: root})
        self.closed = False

    def close(self):
        self.closed = True


class FakeH5pyd:
    def __init__(self, files):
        self.files = files
        self.opened = []
        self.folder_calls = []

    def File(self, uri, mode):
        assert mode == "r"
        self.opened.append(uri)
        return self.files[uri]

    def Folder(self, path, mode=None):
        self.folder_calls.append(path)
        shot = path.strip("/").split("/")[-1]
        return [name.rsplit("/", 1)[-1] for name in self.files if f"/{shot}/" in name]


@pytest.fixture
def fake_hsds():
    logical_psi = np.arange(24).reshape(2, 1, 3, 4)
    datasets = {
        "time": FakeDataset([0.1, 0.2]),
        "time_slice[]&AOS_SHAPE": FakeDataset([2]),
        "time_slice[]&global_quantities&ip": FakeDataset([10.0, 20.0]),
        "time_slice[]&boundary&outline&r": FakeDataset(
            [[1.0, 2.0, 3.0, -1.0, -1.0], [4.0, 5.0, -1.0, -1.0, -1.0]]
        ),
        "time_slice[]&boundary&outline&r_SHAPE": FakeDataset([[3], [2]]),
        "time_slice[]&profiles_2d[]&AOS_SHAPE": FakeDataset([[1], [1]]),
        # IMAS HDF5 stores the two trailing leaf dimensions in reverse order.
        "time_slice[]&profiles_2d[]&psi": FakeDataset(
            logical_psi.transpose(0, 1, 3, 2)
        ),
        "time_slice[]&boundary&x_point[]&AOS_SHAPE": FakeDataset([[2], [1]]),
        "time_slice[]&boundary&x_point[]&r": FakeDataset([[6.0, 7.0], [8.0, -1.0]]),
        "ids_properties&comment": FakeDataset(np.asarray(b"synthetic")),
    }
    equilibrium = FakeFile("equilibrium", FakeGroup(datasets))
    other_signal = FakeDataset([99.0])
    other = FakeFile("core_profiles", FakeGroup({"time": other_signal}))
    module = FakeH5pyd(
        {
            "hdf5://main/39915/equilibrium.h5": equilibrium,
            "hdf5://main/39915/core_profiles.h5": other,
        }
    )
    return module, datasets, equilibrium, other, other_signal


def make_ods(fake_hsds, ids=("equilibrium", "core_profiles")):
    module, *_ = fake_hsds
    store = HSDSStore(39915, ids=ids, h5pyd_module=module)
    return HSDSODS(store=store, consistency_check=False), store


def test_metadata_trie_opens_only_visited_ids_and_does_not_read_leaf(fake_hsds):
    module, datasets, *_ = fake_hsds
    ods, store = make_ods(fake_hsds)

    assert set(ods.keys()) == {"core_profiles", "equilibrium"}
    assert module.opened == []
    assert "equilibrium.time_slice.0.profiles_2d.0.psi" in ods
    assert store.opened_ids == ("equilibrium",)
    assert datasets["time_slice[]&profiles_2d[]&psi"].reads == []
    assert "hdf5://main/39915/core_profiles.h5" not in module.opened


def test_store_reports_client_side_lazy_read_metrics(fake_hsds):
    ods, store = make_ods(fake_hsds, ids="equilibrium")
    value = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
    _ = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]

    metrics = store.metrics
    assert metrics["ids_domain_open_count"] == 1
    assert metrics["metadata_dataset_count"] > 0
    # One selection for two reads is the guarantee that matters: the payload is
    # not re-read. Since issue #329 the second read is served by the ODS and
    # never reaches the store, so the store's own `leaf_cache_hits` no longer
    # fires through this path -- it is checked directly below instead.
    assert metrics["payload_selection_count"] == 1
    assert metrics["returned_logical_bytes"] == value.nbytes


def test_the_store_still_caches_for_a_caller_that_bypasses_the_ods(fake_hsds):
    """`leaf_cache_hits` is shadowed by the ODS-level cache added for #329, so it
    is exercised here at the level where it still applies: a caller holding the
    store directly."""
    _ods, store = make_ods(fake_hsds, ids="equilibrium")
    path = ["equilibrium", "time_slice", 0, "profiles_2d", 0, "psi"]

    first = store[path]
    assert store.metrics["leaf_cache_hits"] == 0
    second = store[path]
    assert store.metrics["leaf_cache_hits"] >= 1
    assert store.metrics["payload_selection_count"] == 1
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_leaf_selection_shape_slicing_and_value_conversion(fake_hsds):
    ods, _ = make_ods(fake_hsds, ids="equilibrium")
    datasets = fake_hsds[1]

    assert ods["equilibrium.time_slice.1.global_quantities.ip"] == 20.0
    np.testing.assert_array_equal(
        ods["equilibrium.time_slice.0.boundary.outline.r"],
        np.asarray([1.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        ods["equilibrium.time_slice.1.profiles_2d.0.psi"],
        np.arange(24).reshape(2, 1, 3, 4)[1, 0],
    )
    assert ods["equilibrium.time_slice.0.boundary.x_point.1.r"] == 7.0
    assert ods["equilibrium.ids_properties.comment"] == "synthetic"

    assert datasets["time_slice[]&global_quantities&ip"].reads == [(1,)]
    assert datasets["time_slice[]&boundary&outline&r"].reads == [
        (0, slice(0, 3))
    ]
    assert datasets["time_slice[]&profiles_2d[]&psi"].reads == [
        (1, 0, slice(None), slice(None))
    ]


def test_nested_aos_lengths_come_from_aos_shape(fake_hsds):
    ods, _ = make_ods(fake_hsds, ids="equilibrium")

    assert ods["equilibrium.time_slice"].keys() == [0, 1]
    assert ods["equilibrium.time_slice.0.boundary.x_point"].keys() == [0, 1]
    assert ods["equilibrium.time_slice.1.boundary.x_point"].keys() == [0]


def test_leaf_is_cached_and_uncached_access_fails_after_close(fake_hsds):
    ods, store = make_ods(fake_hsds, ids="equilibrium")
    signal = fake_hsds[1]["time_slice[]&global_quantities&ip"]

    assert ods["equilibrium.time_slice.0.global_quantities.ip"] == 10.0
    assert len(signal.reads) == 1
    ods.close()
    assert store.closed
    assert ods["equilibrium.time_slice.0.global_quantities.ip"] == 10.0
    assert len(signal.reads) == 1
    with pytest.raises(Exception) as error:
        ods["equilibrium.time_slice.1.global_quantities.ip"]
    assert "closed" in str(error.value).lower()


def test_context_manager_returns_ods_and_closes_all_handles(fake_hsds):
    _, _, equilibrium, other, _ = fake_hsds
    ods, store = make_ods(fake_hsds)
    with ods as opened:
        assert opened is ods
        opened["equilibrium.time"]
        opened["core_profiles.time"]
    assert store.closed
    assert equilibrium.closed
    assert other.closed


def test_discovered_ids_and_cached_leaf_remain_usable_after_close(fake_hsds):
    module, *_ = fake_hsds
    store = HSDSStore(39915, h5pyd_module=module)
    ods = HSDSODS(store=store, consistency_check=False)
    assert ods["equilibrium.time"] is not None
    ods.close()
    np.testing.assert_array_equal(ods["equilibrium.time"], [0.1, 0.2])
    assert module.folder_calls == ["/main/39915/"]


def test_public_open_ods_uses_direct_h5pyd_without_folder_when_ids_given(
    fake_hsds, monkeypatch
):
    module, *_ = fake_hsds
    monkeypatch.setattr(lazy_ods, "h5pyd", module)

    ods = lazy_ods.open_ods(39915, ids="equilibrium", consistency_check=False)
    assert isinstance(ods, HSDSODS)
    assert ods["equilibrium.time_slice.0.global_quantities.ip"] == 10.0
    assert module.folder_calls == []


def test_store_rejects_empty_ids(fake_hsds):
    module, *_ = fake_hsds
    with pytest.raises(ValueError, match="at least one"):
        HSDSStore(39915, ids=[], h5pyd_module=module)


def test_a_write_to_a_stored_leaf_is_kept_not_clobbered(fake_hsds):
    """Issue #329. ``__getitem__`` re-fetched on every read of a path the store
    holds, so an assignment appeared to succeed and the next read handed back the
    stored value again -- silently. That made every in-place updater a no-op on
    HSDS-backed data for any leaf the database already had, while writes to
    leaves it lacked did persist: the same column of the same sheet could mix
    VAFT-derived and database-stored values with no marker.

    The class fetches leaves that are *missing*; a present one is the caller's.
    """
    ods, store = make_ods(fake_hsds, ids="equilibrium")

    assert float(ods["equilibrium.time_slice.0.global_quantities.ip"]) == 10.0
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 1234.5
    assert float(ods["equilibrium.time_slice.0.global_quantities.ip"]) == 1234.5
    # Still there on a third read -- the fetch does not creep back.
    assert float(ods["equilibrium.time_slice.0.global_quantities.ip"]) == 1234.5
    # And the sibling slice is untouched, so the override is not global.
    assert float(ods["equilibrium.time_slice.1.global_quantities.ip"]) == 20.0


def test_a_write_before_any_read_is_also_kept(fake_hsds):
    """The clobber did not need a prior read: the fetch ran on the write's own
    read-back."""
    ods, _store = make_ods(fake_hsds, ids="equilibrium")
    ods["equilibrium.time_slice.0.global_quantities.ip"] = 7.5
    assert float(ods["equilibrium.time_slice.0.global_quantities.ip"]) == 7.5


def test_an_array_leaf_can_be_replaced(fake_hsds):
    """Profiles are what the equilibrium updaters actually rewrite."""
    ods, _store = make_ods(fake_hsds, ids="equilibrium")
    original = np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"])
    assert original.size

    replacement = np.zeros_like(original) + 3.0
    ods["equilibrium.time_slice.0.profiles_2d.0.psi"] = replacement
    np.testing.assert_array_equal(
        np.asarray(ods["equilibrium.time_slice.0.profiles_2d.0.psi"]), replacement
    )


def test_a_cached_leaf_is_not_fetched_twice(fake_hsds):
    """The same change makes the cache actually a cache: the second read must not
    go back to the store."""
    _module, datasets, *_ = fake_hsds
    ods, _store = make_ods(fake_hsds, ids="equilibrium")
    dataset = datasets["time_slice[]&profiles_2d[]&psi"]

    _ = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
    reads_after_first = len(dataset.reads)
    _ = ods["equilibrium.time_slice.0.profiles_2d.0.psi"]
    assert len(dataset.reads) == reads_after_first


def test_lazy_fetching_still_works_for_leaves_never_written(fake_hsds):
    """The fix must not turn lazy loading off."""
    module, datasets, *_ = fake_hsds
    ods, store = make_ods(fake_hsds, ids="equilibrium")

    assert datasets["time_slice[]&global_quantities&ip"].reads == []
    assert float(ods["equilibrium.time_slice.1.global_quantities.ip"]) == 20.0
    assert datasets["time_slice[]&global_quantities&ip"].reads
    np.testing.assert_array_equal(ods["equilibrium.time"], [0.1, 0.2])
