import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import vaft.imas.omas_imas as omas_imas


class _DummyDBEntry:
    def __init__(self):
        self.calls = []
        self.factory = SimpleNamespace()
        self.factory.summary = lambda: "factory-summary"

    def get(self, key, occ):
        self.calls.append((key, occ))
        return f"{key}:{occ}"


class _DummyIdsHandle:
    def __init__(self):
        self.closed = False
        self.occurrence = {}
        self.put_calls = []
        self.dataset_description = SimpleNamespace(
            ids_properties=SimpleNamespace(homogeneous_time=1),
            time=[0.1],
        )

    def close(self):
        self.closed = True

    def put_ids(self, value, ds, occ):
        self.put_calls.append((ds, occ, value))


class _DummyIdsCollection:
    def __init__(self):
        self.calls = []
        self.equilibrium = SimpleNamespace(
            ids_properties=SimpleNamespace(homogeneous_time=1),
            time=[0.1],
        )
        self.summary = SimpleNamespace(
            ids_properties=SimpleNamespace(homogeneous_time=1),
            time=[0.1],
        )

    def get_ids(self, _obj, ds, occ):
        self.calls.append((ds, occ))
        if ds == "summary":
            raise _FakeSkip("empty IDS")

    def get_ids_slice(self, _obj, ds, _time, occ):
        self.calls.append((ds, occ))
        if ds == "summary":
            raise _FakeSkip("empty IDS")


class _FakeSkip(Exception):
    pass


class ImasBranchRegressionTests(unittest.TestCase):
    def test_ids_uses_occurrence_dict_per_ids_name(self):
        dbentry = _DummyDBEntry()
        ids = omas_imas.IDS(dbentry, {"summary": 7})

        value = ids.summary

        self.assertEqual(value, "summary:7")
        self.assertEqual(dbentry.calls, [("summary", 7)])

    def test_save_omas_imas_uri_uses_append_mode_and_dd_version(self):
        ods = omas_imas.ODS()
        ids = _DummyIdsHandle()

        with patch.object(omas_imas, "imas_open_uri", return_value=ids) as open_uri:
            returned_paths = omas_imas.save_omas_imas(
                ods,
                uri="imas:hdf5?path=/tmp/imas-save",
                new=False,
                imas_version="9.9.9",
                verbose=False,
            )

        self.assertEqual(returned_paths, [])
        self.assertTrue(ids.closed)
        self.assertEqual(
            ids.put_calls[0][:2],
            ("dataset_description", 0),
        )
        self.assertEqual(open_uri.call_args.kwargs["mode"], "a")
        self.assertEqual(open_uri.call_args.kwargs["dd_version"], "9.9.9")

    def test_load_omas_imas_uri_uses_read_mode_default_dd_version_and_skips_dataset_description(self):
        ids = _DummyIdsHandle()

        with (
            patch.object(omas_imas, "imas_open_uri", return_value=ids) as open_uri,
            patch.object(omas_imas, "infer_fetch_paths", return_value=([], [])),
        ):
            ods = omas_imas.load_omas_imas(
                uri="imas:hdf5?path=/tmp/imas-load",
                verbose=False,
            )

        self.assertTrue(ids.closed)
        self.assertEqual(open_uri.call_args.kwargs["mode"], "r")
        self.assertEqual(
            open_uri.call_args.kwargs["dd_version"],
            omas_imas.IMAS_DD_VERSION_CONVERSION,
        )
        self.assertNotIn("dataset_description", list(ods.keys()))

    def test_load_omas_imas_uri_respects_explicit_imas_version(self):
        ids = _DummyIdsHandle()

        with (
            patch.object(omas_imas, "imas_open_uri", return_value=ids) as open_uri,
            patch.object(omas_imas, "infer_fetch_paths", return_value=([], [])),
        ):
            omas_imas.load_omas_imas(
                uri="imas:hdf5?path=/tmp/imas-load-explicit",
                imas_version="5.1.0",
                verbose=False,
            )

        self.assertEqual(open_uri.call_args.kwargs["dd_version"], "5.1.0")

    def test_infer_fetch_paths_skips_empty_ids_for_configured_exceptions(self):
        ids = _DummyIdsCollection()

        with (
            patch.object(omas_imas, "_IDS_SKIP_EXCEPTIONS", (_FakeSkip,)),
            patch.object(omas_imas, "list_structures", return_value=["equilibrium", "summary"]),
            patch.object(omas_imas, "load_structure", return_value=(None, {})),
            patch.object(omas_imas, "filled_paths_in_ids", return_value=[["equilibrium", "time"]]),
        ):
            paths, joined = omas_imas.infer_fetch_paths(
                ids,
                occurrence={"equilibrium": 2, "summary": 5},
                paths=None,
                time=None,
                imas_version="3.41.0",
                verbose=False,
            )

        self.assertEqual(ids.calls, [("equilibrium", 2), ("summary", 5)])
        self.assertEqual(paths, [["equilibrium", "time"]])
        self.assertEqual(joined, ["equilibrium.time"])

    def test_user_defaults_resolve_at_call_time_not_import_time(self):
        # An import-time os.environ read baked the builder's username (and
        # $HOME) into these signatures and so into the rendered API reference.
        for obj in (omas_imas.load_omas_imas, omas_imas.browse_imas, omas_imas.dynamic_omas_imas):
            defaults = [
                p.default
                for p in inspect.signature(obj).parameters.values()
                if p.default is not inspect.Parameter.empty
            ]
            self.assertIs(inspect.signature(obj).parameters["user"].default, omas_imas._CURRENT_USER)
            for default in defaults:
                self.assertNotIn("imasdb", repr(default), obj.__name__)

        with patch.dict(omas_imas.os.environ, {"USER": "someone_else"}):
            handle = omas_imas.dynamic_omas_imas.__new__(omas_imas.dynamic_omas_imas)
            omas_imas.dynamic_omas_imas.__init__(handle, machine="vest", pulse=1)
            self.assertEqual(handle.kw["user"], "someone_else")
            with patch.object(omas_imas, "recursive_glob", return_value=[]):
                self.assertEqual(omas_imas.browse_imas(quiet=True), {"someone_else": {}})


if __name__ == "__main__":
    unittest.main()
