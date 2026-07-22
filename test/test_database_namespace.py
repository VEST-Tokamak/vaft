from types import ModuleType
from unittest.mock import Mock, patch

import vaft.database as database


def _fake_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_database_load_defaults_to_omas_loader_with_bare_source():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.load(39915, imas_version="3.41.0")

    assert result == "ods"
    load_ods.assert_called_once_with(
        39915,
        directory="public",
        occurrence={},
        paths=None,
        imas_version="3.41.0",
        cache="auto",
    )


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
        directory="public",
        occurrence={},
        dd_version="3.41.0",
        cache="auto",
    )


def test_database_open_routes_to_direct_lazy_loader():
    open_ods = Mock(return_value="lazy")
    fake_lazy = _fake_module("vaft.database.lazy_ods", open_ods=open_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": fake_lazy}):
        result = database.open(39915, paths="equilibrium", imas_version="3.41.0")

    assert result == "lazy"
    open_ods.assert_called_once_with(
        39915,
        directory="public",
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
