from __future__ import annotations

import os

import numpy as np
import pytest

from vaft.database.lazy_imas import HSDSIMASHandle, LazyIMASClosedError


class _Dataset:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype
        self.size = self.value.size

    def __getitem__(self, selection):
        return self.value[selection]


class _Group(dict):
    def items(self):
        return super().items()


class _File(_Group):
    def close(self):
        self.closed = True


class _H5PYD:
    def __init__(self, domains):
        self.domains = domains

    def File(self, uri, mode):
        name = uri.rsplit('/', 1)[-1].removesuffix('.h5')
        return self.domains[name]

    def Folder(self, _path, mode=None):
        return [f"{name}.h5" for name in self.domains]


def _fake_hsds():
    equilibrium = _Group(
        {
            "ids_properties&homogeneous_time": _Dataset(1),
            "time": _Dataset([0.1, 0.2]),
            "time_slice[]&AOS_SHAPE": _Dataset([2]),
            "time_slice[]&profiles_2d[]&AOS_SHAPE": _Dataset([[1], [1]]),
            "time_slice[]&profiles_2d[]&psi": _Dataset(
                np.arange(12, dtype=float).reshape(2, 1, 2, 3)
            ),
            "time_slice[]&profiles_2d[]&psi_SHAPE": _Dataset([[[2, 3]], [[2, 3]]]),
        }
    )
    magnetics = _Group(
        {
            "ids_properties&homogeneous_time": _Dataset(1),
            "time": _Dataset([0.1, 0.2, 0.3]),
            "ip[]&AOS_SHAPE": _Dataset([1]),
            "ip[]&data": _Dataset([[1.0, 2.0, 3.0]]),
            "ip[]&data_SHAPE": _Dataset([[3]]),
        }
    )
    return _H5PYD({"equilibrium": _File({"equilibrium": equilibrium}), "magnetics": _File({"magnetics": magnetics})})


def test_native_lazy_ids_select_1d_and_2d_leaves_without_staging():
    handle = HSDSIMASHandle(
        1, ids=["equilibrium", "magnetics"], imas_version="3.41.0", h5pyd_module=_fake_hsds()
    )
    equilibrium = handle.get("equilibrium")
    magnetics = handle.get("magnetics")

    assert equilibrium._lazy is True
    assert magnetics._lazy is True
    np.testing.assert_allclose(magnetics.time, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(magnetics.ip[0].data, [1.0, 2.0, 3.0])
    # The adapter restores IMAS/NumPy axis order after selecting AOS axes.
    assert equilibrium.time_slice[0].profiles_2d[0].psi.shape == (3, 2)
    assert handle.metrics["ids_domain_open_count"] == 2


def test_native_lazy_leaf_cache_and_close_contract():
    handle = HSDSIMASHandle(1, ids="equilibrium", imas_version="3.41.0", h5pyd_module=_fake_hsds())
    equilibrium = handle.get()
    first = np.asarray(equilibrium.time)
    selections = handle.metrics["payload_selection_count"]
    second = np.asarray(equilibrium.time)
    np.testing.assert_allclose(first, second)
    assert handle.metrics["payload_selection_count"] == selections

    handle.close()
    # Already materialized native leaves remain usable.
    np.testing.assert_allclose(equilibrium.time, first)
    with pytest.raises(LazyIMASClosedError):
        _ = equilibrium.time_slice


@pytest.mark.integration
def test_public_native_lazy_imas_parity_for_equilibrium_and_magnetics():
    if os.environ.get("VAFT_RUN_HSDS_INTEGRATION") != "1":
        pytest.skip("set VAFT_RUN_HSDS_INTEGRATION=1 for public HSDS native lazy test")
    import vaft

    with vaft.database.open(39915, source="public", representation="imas", paths="equilibrium") as handle:
        equilibrium = handle.get()
        assert equilibrium._lazy is True
        assert equilibrium.time_slice[0].profiles_2d[0].psi.shape == (513, 513)
        assert handle.metrics["ids_domain_open_count"] == 1
    with vaft.database.open(39915, source="public", representation="imas", paths="magnetics") as handle:
        magnetics = handle.get()
        assert magnetics.time.shape == (2000,)
        assert magnetics.ip[0].data.shape == (2000,)
