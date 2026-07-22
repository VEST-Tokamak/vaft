from types import ModuleType
from unittest.mock import Mock, patch

import vaft.database as database


def _fake_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_database_load_defaults_to_ods_loader():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.load(39915)

    assert result == "ods"
    load_ods.assert_called_once_with(39915, directory="public")


def test_database_load_preserves_legacy_directory_positional_argument():
    load_ods = Mock(return_value="ods")
    fake_ods = _fake_module("vaft.database.ods", load_ods=load_ods)
    with patch.dict("sys.modules", {"vaft.database.ods": fake_ods}):
        result = database.load(39915, "public_omas", paths=["magnetics"])

    assert result == "ods"
    load_ods.assert_called_once_with(39915, directory="public_omas", paths=["magnetics"])


def test_database_load_uses_ids_loader_only_with_explicit_ids_name():
    load_ids = Mock(return_value="ids")
    fake_ids = _fake_module("vaft.database.ids", load=load_ids)
    with patch.dict("sys.modules", {"vaft.database.ids": fake_ids}):
        result = database.load(39915, ids_name="dataset_description", dd_version="3.41.0")

    assert result == "ids"
    load_ids.assert_called_once_with(
        39915,
        "dataset_description",
        directory="public",
        dd_version="3.41.0",
    )


def test_database_load_ids_alias_routes_to_native_ids_loader():
    load_ids = Mock(return_value="ids")
    fake_ids = _fake_module("vaft.database.ids", load=load_ids)
    with patch.dict("sys.modules", {"vaft.database.ids": fake_ids}):
        result = database.load_ids(39915, "dataset_description", directory="public")

    assert result == "ids"
    load_ids.assert_called_once_with(39915, "dataset_description", directory="public")


def test_database_open_ods_routes_to_direct_lazy_loader():
    open_ods = Mock(return_value="lazy")
    fake_lazy = _fake_module("vaft.database.lazy_ods", open_ods=open_ods)
    with patch.dict("sys.modules", {"vaft.database.lazy_ods": fake_lazy}):
        result = database.open_ods(39915, ids="equilibrium")

    assert result == "lazy"
    open_ods.assert_called_once_with(39915, ids="equilibrium")
