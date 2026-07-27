"""Import smoke coverage for the NumPy 2 / h5py compatibility regression."""

from __future__ import annotations

from importlib.metadata import version


def test_hdf5_clients_and_database_import_with_supported_binary_stack():
    import h5py
    import h5pyd
    import numpy as np
    import vaft.database

    assert version("h5py") == h5py.__version__
    assert tuple(int(part) for part in h5py.__version__.split(".")[:2]) >= (3, 16)
    assert tuple(int(part) for part in h5pyd.__version__.split(".")[:2]) >= (0, 20)
    assert int(np.__version__.split(".", 1)[0]) in {1, 2}
    assert vaft.database is not None
