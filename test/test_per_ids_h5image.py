from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from vaft.database import h5image
from vaft.database import ods as ods_database


class _LocalH5PYD:
    def __init__(self, root: Path):
        self.root = root

    def File(self, uri: str, mode: str):
        filename = uri.rsplit("/", 1)[-1]
        return h5py.File(self.root / filename, mode)


def _source_info():
    return {
        "remote_uri": "hdf5://public/123/equilibrium.h5",
        "last_modified": "revision-1",
        "backend_version": "1.0",
    }


def test_per_ids_h5image_is_byte_exact_and_manifest_validated(monkeypatch, tmp_path):
    source = tmp_path / "equilibrium.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["HDF5_BACKEND_VERSION"] = "1.0"
        handle.create_dataset("equilibrium/time", data=[0.1, 0.2, 0.3])
    original = source.read_bytes()

    monkeypatch.setattr(h5image, "h5pyd", _LocalH5PYD(tmp_path))
    monkeypatch.setattr(h5image, "_require_h5pyd", lambda: None)
    result = h5image.publish_image(
        source,
        "public",
        123,
        imas_version="3.41.0",
        source_info=_source_info(),
    )
    assert result["uri"].endswith("/equilibrium.h5image.h5")
    assert result["manifest"]["source_sha256"]

    restored = tmp_path / "restored" / "equilibrium.h5"
    stats = h5image.materialize_image(
        "public", 123, "equilibrium.h5", restored, source_info=_source_info()
    )
    assert restored.read_bytes() == original
    assert stats["transport"] == "h5image"

    stale = dict(_source_info(), last_modified="revision-2")
    with pytest.raises(h5image.H5ImageUnavailableError, match="stale"):
        h5image.materialize_image(
            "public", 123, "equilibrium.h5", tmp_path / "stale.h5", source_info=stale
        )


def test_derived_filename_validation():
    assert h5image.derived_filename("master.h5") == "master.h5image.h5"
    assert h5image.is_derived_filename("equilibrium.h5image.h5")
    with pytest.raises(ValueError):
        h5image.derived_filename("nested/equilibrium.h5")


def test_source_revision_excludes_derived_domains(monkeypatch):
    opened = []

    class Domain:
        modified = "revision-1"
        attrs = {"HDF5_BACKEND_VERSION": "1.0"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeH5PYD:
        @staticmethod
        def Folder(_path, mode=None):
            return ["master.h5", "equilibrium.h5", "equilibrium.h5image.h5"]

        @staticmethod
        def File(uri, _mode):
            opened.append(uri)
            return Domain()

    monkeypatch.setattr(ods_database, "h5pyd", FakeH5PYD())
    monkeypatch.setattr(ods_database, "_require_h5pyd", lambda: None)
    _revision, domains = ods_database._source_revision("public", 123)

    assert set(domains) == {"master.h5", "equilibrium.h5"}
    assert not any("h5image" in uri for uri in opened)
