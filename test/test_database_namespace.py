from types import ModuleType
from unittest.mock import Mock, patch

import vaft.database as database


def _fake_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_database_load_defaults_to_the_main_source():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.load(39915, imas_version="3.41.0")

    assert result == "ods"
    load_ods.assert_called_once_with(
        39915,
        source="main",
        occurrence={},
        paths=None,
        imas_version="3.41.0",
        cache="auto",
        transport="auto",
    )


def test_database_load_preserves_global_occurrence_for_full_omas_load():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        database.load(39915, occurrence=2, imas_version="3.41.0")

    assert load_ods.call_args.kwargs["occurrence"] == {"*": 2}


def test_database_load_accepts_legacy_positional_source():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.load(39915, "public", imas_version="3.41.0")

    assert result == "ods"
    assert load_ods.call_args.kwargs["source"] == "public"


def test_database_load_still_reads_the_legacy_public_source():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        database.load(39915, source="public", imas_version="3.41.0")

    assert load_ods.call_args.kwargs["source"] == "public"


def test_database_load_accepts_the_deprecated_directory_alias():
    import pytest

    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        with pytest.warns(DeprecationWarning, match="deprecated alias"):
            database.load(39915, directory="public", imas_version="3.41.0")

    assert load_ods.call_args.kwargs["source"] == "public"


def test_database_load_rejects_two_names_for_one_source():
    import pytest

    with pytest.raises(TypeError, match="only one of source"):
        database.load(39915, "main", directory="public", imas_version="3.41.0")


def test_database_load_rejects_a_source_outside_the_catalog():
    import pytest

    with pytest.raises(ValueError, match="Unknown HSDS source"):
        database.load(39915, source="typoo", imas_version="3.41.0")


def test_database_load_rejects_named_storage_key():
    import pytest

    with pytest.raises(ValueError):
        database.load("39915_test", imas_version="3.41.0")


def test_database_rejects_uri_and_local_path_sources():
    import pytest

    with pytest.raises(ValueError, match="bare HSDS namespace"):
        database.load(39915, source="hdf5://public", imas_version="3.41.0")
    with pytest.raises(ValueError, match="bare HSDS namespace"):
        database.load(39915, source="/tmp/data", imas_version="3.41.0")


def test_database_load_uses_native_ids_for_imas_representation():
    load_ids = Mock(return_value="ids")
    fake_ids = _fake_module("vaft.database.ids", load=load_ids)
    with patch.dict("sys.modules", {"vaft.database.ids": fake_ids}):
        result = database.load(
            39915,
            representation="imas",
            paths="dataset_description",
            imas_version="3.41.0",
        )

    assert result == "ids"
    load_ids.assert_called_once_with(
        39915,
        "dataset_description",
        source="main",
        occurrence={},
        dd_version="3.41.0",
        cache="auto",
        transport="auto",
    )


def test_database_open_routes_to_direct_lazy_loader():
    open_ods = Mock(return_value="lazy")
    fake_lazy = _fake_module("vaft.database.lazy_ods", open_ods=open_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": fake_lazy}):
        result = database.open(39915, paths="equilibrium", imas_version="3.41.0")

    assert result == "lazy"
    open_ods.assert_called_once_with(
        39915,
        source="main",
        ids=["equilibrium"],
        imas_version="3.41.0",
    )


def test_database_open_allows_detailed_omas_paths_but_scopes_the_root_ids():
    open_ods = Mock(return_value="lazy")
    fake_lazy = _fake_module("vaft.database.lazy_ods", open_ods=open_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": fake_lazy}):
        database.open(
            39915,
            paths="equilibrium.time_slice.0.profiles_2d.0.psi",
            imas_version="3.41.0",
        )

    assert open_ods.call_args.kwargs["ids"] == ["equilibrium"]


def test_database_open_routes_native_imas_to_the_direct_adapter():
    open_imas = Mock(return_value="native-lazy")
    fake_lazy = _fake_module("vaft.database.lazy_imas", open_imas=open_imas)
    with (
        patch.dict("sys.modules", {"vaft.database.lazy_imas": fake_lazy}),
        patch.object(database, "_infer_remote_imas_version", return_value="3.41.0"),
    ):
        result = database.open(
            39915, representation="imas", paths="equilibrium", imas_version="3.41.0"
        )

    assert result == "native-lazy"
    open_imas.assert_called_once_with(
        39915, source="main", ids=["equilibrium"], imas_version="3.41.0"
    )


def test_database_save_forwards_per_ids_derived_policy():
    save_ods = Mock(return_value="hdf5://main/39915/")
    fake_ods = _fake_module("vaft.database.ods", save_ods=save_ods)
    data = object()
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.save(data, 39915, derived_cache="imas-images")

    assert result == "hdf5://main/39915/"
    save_ods.assert_called_once_with(
        data,
        39915,
        source="main",
        occurrence={},
        imas_version=None,
        derived_cache="imas-images",
    )


def test_database_save_refuses_the_read_only_legacy_source_before_any_io():
    import pytest

    from vaft.database.sources import ReadOnlySourceError

    save_ods = Mock()
    fake_ods = _fake_module("vaft.database.ods", save_ods=save_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        with pytest.raises(ReadOnlySourceError, match="read-only legacy reference"):
            database.save(object(), 39915, source="public")

    save_ods.assert_not_called()


def test_database_save_keeps_named_sources_isolated():
    save_ods = Mock(return_value="uri")
    fake_ods = _fake_module("vaft.database.ods", save_ods=save_ods)
    data = object()
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        database.save(data, 39915)
        database.save(data, 39915, source="chease-mhd-stability")

    assert [call.kwargs["source"] for call in save_ods.call_args_list] == [
        "main",
        "chease-mhd-stability",
    ]


def test_database_save_accepts_the_deprecated_target_alias():
    import pytest

    save_ods = Mock(return_value="uri")
    fake_ods = _fake_module("vaft.database.ods", save_ods=save_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        with pytest.warns(DeprecationWarning, match="deprecated alias"):
            database.save(object(), 39915, target="vfit-gse")

    assert save_ods.call_args.kwargs["source"] == "vfit-gse"


def test_native_save_rejects_omas_derived_cache_before_io():
    import pytest

    from vaft.database import ids

    with pytest.raises(ValueError, match="native IDS save supports"):
        ids.save(object(), 39915, derived_cache="both")


def test_save_ods_records_the_written_source_and_restores_the_callers_ods(monkeypatch):
    """The namespace written is the provenance, even when the ODS came from another.

    ``save_omas_imas`` only fills ``data_entry.user`` when it is absent, so a
    corrective updater that loads from ``public`` and republishes to ``main``
    would otherwise ship an ODS labelled ``public``.
    """
    import omas

    from vaft.database import ods as ods_module

    ods = omas.ODS()
    ods["dataset_description.data_entry.user"] = "public"
    seen = {}

    def fake_save_omas_imas(data, **kwargs):
        seen["in_ods"] = data["dataset_description.data_entry.user"]
        seen["kwarg"] = kwargs["user"]

    monkeypatch.setattr(ods_module, "save_omas_imas", fake_save_omas_imas)
    monkeypatch.setattr(ods_module, "is_connect", lambda: True)
    monkeypatch.setattr(ods_module, "require_source_exists", lambda source: None)
    monkeypatch.setattr(ods_module, "_upload_local_shot", lambda **kwargs: [])

    uri = ods_module.save_ods(ods, 39915, derived_cache="none")

    assert uri == "hdf5://main/39915/"
    assert seen == {"in_ods": "main", "kwarg": "main"}
    assert ods["dataset_description.data_entry.user"] == "public"


def test_save_ods_leaves_no_provenance_behind_when_the_ods_carried_none(monkeypatch):
    import omas

    from vaft.database import ods as ods_module

    ods = omas.ODS()
    monkeypatch.setattr(ods_module, "save_omas_imas", lambda data, **kwargs: None)
    monkeypatch.setattr(ods_module, "is_connect", lambda: True)
    monkeypatch.setattr(ods_module, "require_source_exists", lambda source: None)
    monkeypatch.setattr(ods_module, "_upload_local_shot", lambda **kwargs: [])

    ods_module.save_ods(ods, 39915, source="vfit-gse", derived_cache="none")

    assert "dataset_description.data_entry.user" not in ods


def test_save_ods_checks_the_source_exists_before_staging_anything(monkeypatch):
    from vaft.database import ods as ods_module
    from vaft.database.sources import MissingSourceError

    import omas

    staged = []

    def fail(source):
        raise MissingSourceError(source, "domain not found")

    monkeypatch.setattr(ods_module, "is_connect", lambda: True)
    monkeypatch.setattr(ods_module, "require_source_exists", fail)
    monkeypatch.setattr(
        ods_module, "save_omas_imas", lambda data, **kwargs: staged.append(kwargs)
    )

    import pytest

    with pytest.raises(MissingSourceError, match="hstouch"):
        ods_module.save_ods(omas.ODS(), 39915, source="vfit-element")

    assert staged == []


def test_save_ods_refuses_a_user_that_contradicts_the_destination(monkeypatch):
    """`user=` must not be able to mislabel which namespace a shot was written to.

    On the server path the URI opens the data entry, so `user` has no effect
    beyond `dataset_description.data_entry.user` -- the field that records the
    destination. An override there could only be false.
    """
    import omas
    import pytest

    from vaft.database import ods as ods_module

    touched = []
    monkeypatch.setattr(ods_module, "is_connect", lambda: touched.append("connect"))
    monkeypatch.setattr(
        ods_module, "require_source_exists", lambda source: touched.append("probe")
    )

    with pytest.raises(ValueError, match="contradicts the HSDS source"):
        ods_module.save_ods(omas.ODS(), 39915, source="main", user="public")

    # Rejected as an argument error, before any connection or remote probe.
    assert touched == []


def test_save_ods_accepts_a_user_that_agrees_with_the_destination(monkeypatch):
    import omas

    from vaft.database import ods as ods_module

    monkeypatch.setattr(ods_module, "save_omas_imas", lambda data, **kwargs: None)
    monkeypatch.setattr(ods_module, "is_connect", lambda: True)
    monkeypatch.setattr(ods_module, "require_source_exists", lambda source: None)
    monkeypatch.setattr(ods_module, "_upload_local_shot", lambda **kwargs: [])

    uri = ods_module.save_ods(
        omas.ODS(), 39915, source="vfit-gse", user="vfit-gse", derived_cache="none"
    )

    assert uri == "hdf5://vfit-gse/39915/"
