"""DD / DDView / DDCollection semantics (#1127, #1128, #1132).

The contract under test: an occurrence's name is its stored
``ids_properties.name``; unnamed occurrences stay numeric; a default exists
only as policy and is never substituted; a view selects per IDS, is immutable,
and shares the DD's live objects. Names here are test-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

imas = pytest.importorskip("imas")

from vaft.imas import (  # noqa: E402
    DD,
    AmbiguousInstanceError,
    CatalogMismatchError,
    CoherenceError,
    DDCollection,
    DDView,
    IMASStore,
    InstanceCatalog,
    InstanceKey,
    InstanceUnavailableError,
    MemoryStore,
    NoDefaultError,
    PhysicalLocator,
    StorageBindings,
    UnknownInstanceError,
)
from vaft.imas._dd import format_reference, parse_reference, read_references, write_references  # noqa: E402


def _ids(factory, ids_name: str, name: str = "", upstream=()):
    ids = getattr(factory, ids_name)()
    ids.ids_properties.homogeneous_time = 2
    ids.ids_properties.name = name
    if upstream:
        write_references(ids, upstream)
    return ids


def _memory(layout, dd_version: str = "3.41.0") -> MemoryStore:
    store = MemoryStore(dd_version)
    for ids_name, occurrence, name, *upstream in layout:
        store.put(_ids(store.factory, ids_name, name, *upstream), occurrence)
    return store


ENTRY = [
    ("equilibrium", 0, "instance-a"),
    ("equilibrium", 3, ""),
    ("equilibrium", 4, "instance-b"),
    ("mhd_linear", 1, "stability-b", ["equilibrium/instance-b"]),
    ("magnetics", 0, "routine"),
]


@pytest.fixture
def dd():
    with DD(_memory(ENTRY)) as value:
        yield value


# -- discovery ----------------------------------------------------------------


def test_names_come_from_the_payload_and_unnamed_stay_numeric(dd):
    infos = dd.instances("equilibrium")
    assert [(i.occurrence, i.name) for i in infos] == [(0, "instance-a"), (3, None), (4, "instance-b")]
    assert infos[0].name_source == "payload" and infos[1].name_source is None
    assert dd.available_ids() == ("equilibrium", "magnetics", "mhd_linear")
    assert dd.instances("core_profiles") == ()


def test_unknown_ids_names_are_rejected(dd):
    with pytest.raises(ValueError, match="not an IDS"):
        dd.instances("equilibrum")
    with pytest.raises(AttributeError):
        dd.equilibrum


# -- defaults -----------------------------------------------------------------


def test_no_policy_means_no_default_even_when_occurrence_0_exists(dd):
    with pytest.raises(NoDefaultError, match="dd.view\\(magnetics="):
        dd.magnetics
    assert dd.default("magnetics") is None


def test_an_explicit_default_policy_resolves(dd):
    with DD(_memory(ENTRY), defaults={"equilibrium": "instance-b", "magnetics": 0}) as policy:
        assert str(policy.equilibrium.ids_properties.name) == "instance-b"
        assert str(policy["magnetics"].ids_properties.name) == "routine"


def test_a_missing_default_is_reported_not_substituted():
    with DD(_memory(ENTRY), defaults={"equilibrium": "instance-z"}) as policy:
        with pytest.raises(InstanceUnavailableError, match="no other instance is substituted"):
            policy.equilibrium


def test_catalog_default_and_registered_but_absent():
    catalog = InstanceCatalog.from_mapping({
        "equilibrium": {
            "default": "instance-a",
            "instances": [
                {"occurrence": 0, "name": "instance-a", "description": "a"},
                {"occurrence": 4, "name": "instance-b", "description": "b"},
                {"occurrence": 6, "name": "instance-c", "description": "c"},
            ],
        }
    })
    with DD(_memory(ENTRY), catalog=catalog) as managed:
        assert str(managed.equilibrium.ids_properties.name) == "instance-a"
        assert not managed.has("equilibrium", "instance-c")
        with pytest.raises(InstanceUnavailableError, match="registered but not stored"):
            managed.get("equilibrium", "instance-c")
        with pytest.raises(UnknownInstanceError):
            managed.get("equilibrium", "instance-q")


def test_catalog_payload_disagreement_fails_loudly():
    catalog = InstanceCatalog.from_mapping({
        "equilibrium": {
            "default": None,
            "instances": [{"occurrence": 3, "name": "instance-b", "description": "b"}],
        }
    })
    with DD(_memory(ENTRY), catalog=catalog) as managed:
        with pytest.raises(CatalogMismatchError, match="allocates occurrence 3"):
            managed.get("equilibrium", "instance-a")
        assert not managed.validate().ok


# -- selection ----------------------------------------------------------------


def test_views_select_per_ids_and_reach_unnamed_occurrences(dd):
    view = dd.view(equilibrium=3, mhd_linear="stability-b", magnetics="routine")
    assert view.occurrences() == {"equilibrium": 3, "magnetics": 0, "mhd_linear": 1}
    assert view.equilibrium is dd.get("equilibrium", 3)


def test_a_bad_selection_fails_at_view_construction(dd):
    with pytest.raises(UnknownInstanceError, match="stored:"):
        dd.view(equilibrium="instance-q")
    with pytest.raises(UnknownInstanceError):
        dd.view(equilibrium=9)
    with pytest.raises(TypeError):
        dd.view(equilibrium=True)


def test_views_are_immutable_and_share_live_objects(dd):
    a = dd.view(equilibrium="instance-b")
    b = a.view(mhd_linear="stability-b")
    assert a.overrides.keys() == {"equilibrium"}
    assert a.equilibrium is b.equilibrium is dd.get("equilibrium", "instance-b")
    with pytest.raises(AttributeError, match="immutable"):
        a.x = 1
    with pytest.raises(TypeError):
        a.overrides["magnetics"] = None
    assert a == dd.view(equilibrium="instance-b") and hash(a) == hash(dd.view(equilibrium="instance-b"))


def test_a_view_selection_omits_ids_without_a_default(dd):
    assert set(dd.view(equilibrium=0).selection()) == {"equilibrium"}


def test_duplicate_names_are_ambiguous_not_first_wins():
    with DD(_memory([("equilibrium", 0, "twin"), ("equilibrium", 1, "twin")])) as dup:
        with pytest.raises(AmbiguousInstanceError):
            dup.view(equilibrium="twin")
        assert len(dup.instances("equilibrium")) == 2


# -- coherence ----------------------------------------------------------------


def test_mixed_views_are_allowed_but_reported(dd):
    view = dd.view(equilibrium="instance-a", mhd_linear="stability-b")
    report = view.validate()
    assert not report.coherent
    (mismatch,) = report.mismatches
    assert mismatch.requires == "equilibrium/instance-b" and mismatch.source == "provenance"
    assert dd.view(equilibrium="instance-b", mhd_linear="stability-b").validate().coherent


def test_validation_is_scoped(dd):
    view = dd.view(equilibrium="instance-a", mhd_linear="stability-b")
    assert view.validate(ids=["equilibrium"]).coherent
    assert not view.validate(ids=["mhd_linear"]).coherent


def test_an_absent_upstream_is_missing_and_an_unselected_one_is_not_fatal():
    layout = [("mhd_linear", 0, "orphan", ["equilibrium/instance-z"]), ("equilibrium", 0, "instance-a")]
    with DD(_memory(layout)) as entry:
        missing = entry.view(mhd_linear="orphan").validate()
        assert [d.status for d in missing.dependencies] == ["missing"] and not missing.coherent
    layout = [("mhd_linear", 0, "child", ["equilibrium/instance-a"]), ("equilibrium", 0, "instance-a")]
    with DD(_memory(layout)) as entry:
        report = entry.view(mhd_linear="child").validate()
        assert [d.status for d in report.dependencies] == ["unselected"] and report.coherent


def test_occurrence_references_mean_the_same_entry():
    layout = [("equilibrium", 3, ""), ("mhd_linear", 0, "child", ["equilibrium:3"])]
    with DD(_memory(layout)) as entry:
        assert entry.view(equilibrium=3, mhd_linear="child").validate().coherent


def test_an_incoherent_default_view_is_refused():
    layout = [("equilibrium", 0, "instance-a"), ("equilibrium", 4, "instance-b"),
              ("mhd_linear", 0, "child", ["equilibrium/instance-b"])]
    with DD(_memory(layout), defaults={"equilibrium": "instance-a", "mhd_linear": "child"}) as entry:
        with pytest.raises(CoherenceError, match="not coherent"):
            entry.default_view()
    with DD(_memory(layout), defaults={"equilibrium": "instance-b", "mhd_linear": "child"}) as entry:
        assert entry.default_view().mhd_linear is entry.mhd_linear


def test_catalog_inputs_add_to_stored_provenance():
    catalog = InstanceCatalog.from_mapping({
        "equilibrium": {"default": None, "instances": [
            {"occurrence": 0, "name": "instance-a", "description": "a"},
            {"occurrence": 4, "name": "instance-b", "description": "b"},
        ]},
        "magnetics": {"default": None, "instances": [
            {"occurrence": 0, "name": "routine", "description": "m",
             "input": {"ids": "equilibrium", "name": "instance-a"}},
        ]},
    })
    with DD(_memory(ENTRY), catalog=catalog) as managed:
        report = managed.view(equilibrium="instance-b", magnetics="routine").validate()
        assert [(d.source, d.status) for d in report.mismatches] == [("catalog", "mismatch")]


@pytest.mark.parametrize("dd_version", ["3.41.0", "4.0.0"])
def test_provenance_round_trips_in_both_dd_major_versions(dd_version):
    ids = _ids(imas.IDSFactory(dd_version), "mhd_linear", "x")
    write_references(ids, ["equilibrium/instance-b", "equilibrium:3"])
    write_references(ids, ["equilibrium/instance-c"])  # replaces VAFT refs only
    assert read_references(ids) == ("equilibrium/instance-c",)


def test_reference_grammar():
    assert format_reference("equilibrium", name="kinetic-efit") == "equilibrium/kinetic-efit"
    assert format_reference("equilibrium", occurrence=4) == "equilibrium:4"
    assert parse_reference("equilibrium:4").occurrence == 4
    assert parse_reference("imas:hdf5?path=/x#equilibrium") is None
    with pytest.raises(ValueError):
        format_reference("equilibrium", name="a", occurrence=1)


# -- migration bindings --------------------------------------------------------


def _legacy():
    return {"main": _memory([("equilibrium", 0, "")]), "main/chease": _memory([("equilibrium", 0, "")])}


def test_bindings_name_unnamed_legacy_occurrences():
    bindings = StorageBindings({
        InstanceKey("equilibrium", "instance-a"): PhysicalLocator("main", "equilibrium", 0),
        InstanceKey("equilibrium", "instance-a-refined"): PhysicalLocator("main/chease", "equilibrium", 0),
    })
    with DD(_legacy(), bindings=bindings, defaults={"equilibrium": "instance-a"}) as entry:
        infos = entry.instances("equilibrium")
        assert [(i.store, i.name, i.name_source) for i in infos] == [
            ("main", "instance-a", "binding"), ("main/chease", "instance-a-refined", "binding")
        ]
        refined = entry.view(equilibrium="instance-a-refined")
        assert refined.instance("equilibrium").store == "main/chease"
        with pytest.raises(AmbiguousInstanceError, match="unique name"):
            entry.view(equilibrium=0)


def test_the_payload_name_wins_and_disagreement_is_an_error():
    stores = {"main": _memory([("equilibrium", 0, "instance-a")])}
    bindings = StorageBindings({InstanceKey("equilibrium", "other"): PhysicalLocator("main", "equilibrium", 0)})
    with DD(stores, bindings=bindings) as entry, pytest.raises(CatalogMismatchError):
        entry.instances("equilibrium")


def test_bindings_are_one_to_one():
    locator = PhysicalLocator("main", "equilibrium", 0)
    with pytest.raises(ValueError, match="bound twice"):
        StorageBindings([(InstanceKey("equilibrium", "a"), locator), (InstanceKey("equilibrium", "b"), locator)])


# -- persistence and lifecycle ------------------------------------------------


def test_save_instance_writes_the_name_and_keeps_siblings(dd):
    view = dd.view(equilibrium=3)
    view.equilibrium.ids_properties.comment = "edited"
    info = dd.save_instance("equilibrium", "instance-n", occurrence=3)
    assert (info.occurrence, info.name) == (3, "instance-n")
    store = dd.stores["main"]
    assert store.occurrence_names("equilibrium") == {0: "instance-a", 3: "instance-n", 4: "instance-b"}
    assert str(store.get("equilibrium", 3).ids_properties.comment) == "edited"


def test_save_instance_never_overwrites_another_name(dd):
    with pytest.raises(CatalogMismatchError, match="refusing to overwrite"):
        dd.save_instance("equilibrium", "instance-n", data=_ids(imas.IDSFactory("3.41.0"), "equilibrium"), occurrence=4)
    payload = _ids(imas.IDSFactory("3.41.0"), "equilibrium", "instance-q")
    with pytest.raises(CatalogMismatchError, match="payload's ids_properties.name"):
        dd.save_instance("equilibrium", "instance-a", data=payload)


def test_mutation_is_not_persistence(dd):
    dd.get("equilibrium", "instance-a").ids_properties.comment = "in memory only"
    assert str(dd.stores["main"].get("equilibrium", 0).ids_properties.comment) == ""


def test_close_keeps_loaded_objects_and_refuses_new_reads(dd):
    loaded = dd.get("equilibrium", "instance-a")
    dd.close()
    assert str(loaded.ids_properties.name) == "instance-a"
    assert dd.get("equilibrium", "instance-a") is loaded
    with pytest.raises(RuntimeError, match="closed"):
        dd.get("equilibrium", "instance-b")


def test_stores_at_different_dd_versions_need_a_declared_version():
    def stores():
        return {"a": _memory([("equilibrium", 0, "x")], "3.41.0"), "b": _memory([("equilibrium", 0, "y")], "3.42.0")}

    refused = stores()
    with pytest.raises(ValueError, match="pass dd_version"):
        DD(refused)
    assert not refused["a"].writable  # the refused DD closed what it owned
    with DD(stores(), dd_version="3.42.0") as entry:
        assert entry.get("equilibrium", "x")._dd_version == "3.42.0"


def test_a_catalog_free_native_entry_opens_directly(tmp_path):
    path = tmp_path / "third_party"
    with imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version="4.0.0") as raw:
        for occurrence, name in [(2, "k"), (5, "")]:
            raw.put(_ids(raw.factory, "equilibrium", name), occurrence)
    with DD.open(path) as entry:
        assert entry.dd_version == "4.0.0"
        assert [(i.occurrence, i.name) for i in entry.instances("equilibrium")] == [(2, "k"), (5, None)]
        with pytest.raises(NoDefaultError):
            entry.equilibrium
        assert str(entry.view(equilibrium=5).equilibrium.ids_properties.name) == ""


def test_save_through_a_writable_native_store(tmp_path):
    path = tmp_path / "entry"
    store = IMASStore.writable_entry(path, mode="w", dd_version="3.41.0")
    with DD(store) as entry:
        entry.save_instance("equilibrium", "instance-a", data=_ids(imas.IDSFactory("3.41.0"), "equilibrium"), occurrence=2)
    with DD.open(path) as reopened:
        assert [(i.occurrence, i.name) for i in reopened.instances("equilibrium")] == [(2, "instance-a")]


# -- collection ----------------------------------------------------------------


def test_collection_is_keyed_lazy_and_owns_its_members():
    built = []

    def factory(key):
        def make():
            built.append(key)
            return DD(_memory([("equilibrium", 0, f"x-{key}")]))
        return make

    with DDCollection({("vest", 1): factory(1), ("run", "r7"): factory(2)}) as collection:
        assert built == [] and len(collection) == 2
        assert collection[("vest", 1)].instances("equilibrium")[0].name == "x-1"
        assert built == [1] and collection.loaded_keys() == (("vest", 1),)
        assert collection.available_ids() == ("equilibrium",)
        assert list(collection.select(has=["equilibrium"])) == [("vest", 1), ("run", "r7")]
        assert collection.map(lambda d: d.instances("equilibrium")[0].name) == {
            ("vest", 1): "x-1", ("run", "r7"): "x-2"
        }
        members = [collection[key] for key in collection]
    assert all(member.closed for member in members)


def test_collection_rejects_non_dd_members():
    with pytest.raises(TypeError):
        DDCollection({1: "not a DD"})
    with pytest.raises(TypeError, match="returned str"):
        DDCollection({1: lambda: "x"})[1]


def test_isinstance_ddview(dd):
    assert isinstance(dd.view(), DDView)


# -- regressions from the cold review of the first increment -----------------


def _catalog(block_by_ids):
    return InstanceCatalog.from_mapping(block_by_ids)


def test_catalog_and_bindings_together_on_a_legacy_layout():
    # Every legacy variant is occurrence 0 of its own entry; the binding is its
    # allocation until consolidation, so the catalog must not call it a mismatch.
    catalog = _catalog({"equilibrium": {"default": "instance-a", "instances": [
        {"occurrence": 0, "name": "instance-a", "description": "a"},
        {"occurrence": 1, "name": "instance-a-refined", "description": "refined",
         "input": {"ids": "equilibrium", "name": "instance-a"}},
    ]}})
    bindings = StorageBindings({
        InstanceKey("equilibrium", "instance-a"): PhysicalLocator("main", "equilibrium", 0),
        InstanceKey("equilibrium", "instance-a-refined"): PhysicalLocator("main/chease", "equilibrium", 0),
    })
    with DD(_legacy(), catalog=catalog, bindings=bindings) as entry:
        refined = entry.view(equilibrium="instance-a-refined")
        assert refined.instance("equilibrium").store == "main/chease"
        assert entry.validate().ok


def test_a_payload_still_named_by_an_alias_is_reachable_and_renamable():
    catalog = _catalog({"equilibrium": {"default": "new-name", "instances": [
        {"occurrence": 0, "name": "new-name", "description": "x", "aliases": ["old-name"]},
    ]}})
    with DD(_memory([("equilibrium", 0, "old-name")]), catalog=catalog) as entry:
        assert entry.equilibrium is entry.get("equilibrium", "old-name")
        assert entry.get("equilibrium", "new-name") is entry.equilibrium
        info = entry.save_instance("equilibrium", "new-name")
        assert (info.occurrence, info.name) == (0, "new-name")


def test_a_missing_default_makes_the_default_state_incoherent():
    with DD(_memory(ENTRY), defaults={"equilibrium": "instance-z", "magnetics": 0}) as entry:
        assert "equilibrium" not in entry.view().selection()
        report = entry.view().validate()
        assert not report.coherent and dict(report.unavailable_defaults) == {"equilibrium": "instance-z"}
        assert not entry.validate().ok
        with pytest.raises(CoherenceError):
            entry.default_view()


def test_a_saved_object_is_cached_at_the_dd_version():
    with DD(MemoryStore("3.41.0"), dd_version="3.42.0") as entry:
        entry.save_instance("equilibrium", "x", data=_ids(imas.IDSFactory("3.41.0"), "equilibrium"), occurrence=0)
        assert entry.get("equilibrium", "x")._dd_version == "3.42.0"


def test_save_refused_for_a_read_only_store_leaves_the_object_alone(tmp_path):
    path = tmp_path / "ro"
    with imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version="3.41.0") as raw:
        raw.put(_ids(raw.factory, "equilibrium", ""), 0)
    with DD.open(path) as entry:
        live = entry.get("equilibrium", 0)
        with pytest.raises(PermissionError):
            entry.save_instance("equilibrium", "named", occurrence=0)
        assert str(live.ids_properties.name) == ""


def test_a_refused_construction_closes_the_store(tmp_path):
    path = tmp_path / "e"
    with imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version="3.41.0") as raw:
        raw.put(_ids(raw.factory, "equilibrium", "a"), 0)
    store = IMASStore(path)
    with pytest.raises(ValueError, match="not an IDS"):
        DD(store, defaults={"equilibrum": "a"})
    with pytest.raises(RuntimeError, match="closed"):
        store.occurrences("equilibrium")


def test_open_presents_dd4_data_at_a_dd3_version(tmp_path):
    path = tmp_path / "dd4"
    with imas.DBEntry(f"imas:hdf5?path={path}", "w", dd_version="4.0.0") as raw:
        raw.put(_ids(raw.factory, "equilibrium", "k"), 1)
    with DD.open(path, dd_version="3.41.0") as entry:
        assert entry.view(equilibrium="k").equilibrium._dd_version == "3.41.0"


def test_a_selected_subcollection_does_not_close_its_parents_members():
    parent = DDCollection({1: DD(_memory([("equilibrium", 0, "a")]))})
    with parent.select(has=["equilibrium"]) as sub:
        assert list(sub) == [1]
    assert not parent[1].closed
    parent.close()
    assert parent[1].closed


def test_foreign_dd4_references_survive_a_rewrite_whole():
    ids = _ids(imas.IDSFactory("4.0.0"), "mhd_linear", "x")
    node = ids.ids_properties.provenance.node
    node.resize(1)
    node[0].path = ""
    node[0].reference.resize(1)
    node[0].reference[0].name = "imas:hdf5?path=/elsewhere#equilibrium"
    node[0].reference[0].timestamp = "2026-01-01T00:00:00Z"
    write_references(ids, ["equilibrium/a"])
    write_references(ids, ["equilibrium/b"])
    refs = node[0].reference
    assert [(str(r.name), str(r.timestamp)) for r in refs] == [
        ("imas:hdf5?path=/elsewhere#equilibrium", "2026-01-01T00:00:00Z"),
        ("equilibrium/b", ""),
    ]
