from __future__ import annotations

from pathlib import Path

import numpy as np

import vaft
from vaft.database import ods as database_ods


class _Dataset:
    def __init__(self, shape=None, dtype=np.uint8, value=None):
        self.value = np.zeros(shape, dtype=dtype) if value is None else np.asarray(value, dtype=dtype)

    def __getitem__(self, selection):
        return self.value[selection]

    def __setitem__(self, selection, value):
        self.value[selection] = value


class _Domain(dict):
    def __init__(self, modified="1", **kwargs):
        super().__init__(**kwargs)
        self.modified = modified
        self.attrs = {"HDF5_BACKEND_VERSION": "1.0"}

    def create_dataset(self, name, shape, dtype, **_kwargs):
        dataset = _Dataset(shape, dtype)
        self[name] = dataset
        return dataset

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _H5PYD:
    def __init__(self):
        self.shot = {"master.h5": _Domain(), "equilibrium.h5": _Domain()}
        self.derived = {}

    def Folder(self, path, mode=None):
        assert path == "/public/1/"
        return list(self.shot)

    def File(self, uri, mode):
        name = uri.rsplit("/", 1)[-1]
        if name in self.shot:
            return self.shot[name]
        if mode == "w":
            domain = _Domain()
            self.derived[name] = domain
            return domain
        return self.derived[name]


def test_derived_omas_cache_is_revision_and_checksum_validated(monkeypatch):
    fake = _H5PYD()
    monkeypatch.setattr(database_ods, "h5pyd", fake)
    monkeypatch.setattr(database_ods, "_require_h5pyd", lambda: None)
    source = vaft.omas.load(Path("vaft/data/efit/g039915.00317"))

    uri = database_ods._publish_derived_omas_cache(source, "public", 1, "3.41.0")
    assert uri == "hdf5://public/1.omas.h5"
    restored = database_ods._load_derived_omas_cache("public", 1, consistency_check=False)
    assert restored is not None
    assert restored["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == source[
        "equilibrium.time_slice.0.profiles_2d.0.psi"
    ].shape

    fake.shot["equilibrium.h5"].modified = "2"
    assert database_ods._load_derived_omas_cache("public", 1, consistency_check=False) is None
