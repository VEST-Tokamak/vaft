from __future__ import annotations

from types import SimpleNamespace

import imas
import numpy as np

from vaft.imas import omas_imas


def _equilibrium_fixture():
    equilibrium = imas.IDSFactory("3.41.0").new("equilibrium")
    equilibrium.ids_properties.homogeneous_time = 1
    equilibrium.time = np.asarray([0.1])
    equilibrium.time_slice.resize(1)
    equilibrium.time_slice[0].time = 0.1
    equilibrium.time_slice[0].global_quantities.ip = 42.0
    equilibrium.time_slice[0].global_quantities.ip_error_upper = 0.5
    equilibrium.time_slice[0].profiles_2d.resize(1)
    equilibrium.time_slice[0].profiles_2d[0].psi = np.arange(12.0).reshape(3, 4)
    return equilibrium


def test_al5_fast_converter_gets_once_and_preserves_nested_array():
    native = _equilibrium_fixture()

    class Entry:
        def __init__(self):
            self.calls = []

        def get(self, name, occurrence):
            self.calls.append((name, occurrence))
            return native

    entry = Entry()
    wrapper = SimpleNamespace(DBentry=entry)
    ods = omas_imas._load_al5_ods(
        wrapper,
        occurrence={"equilibrium": 2},
        paths=["equilibrium"],
        time=None,
        imas_version="3.41.0",
        skip_uncertainties=False,
        consistency_check=False,
        verbose=False,
    )

    assert entry.calls == [("equilibrium", 2)]
    np.testing.assert_array_equal(
        ods["equilibrium.time_slice.0.profiles_2d.0.psi"],
        native.time_slice[0].profiles_2d[0].psi.value,
    )
    assert ods["equilibrium.ids_properties.occurrence"] == 2


def test_al5_fast_converter_honors_leaf_selection():
    native = _equilibrium_fixture()
    entry = SimpleNamespace(get=lambda _name, _occurrence: native)
    ods = omas_imas._load_al5_ods(
        SimpleNamespace(DBentry=entry),
        occurrence={},
        paths=["equilibrium.time_slice.0.profiles_2d.0.psi"],
        time=None,
        imas_version="3.41.0",
        skip_uncertainties=False,
        consistency_check=False,
        verbose=False,
    )

    assert "equilibrium.time" not in ods
    assert ods["equilibrium.time_slice.0.profiles_2d.0.psi"].shape == (3, 4)


def test_al5_fast_converter_includes_selected_uncertainty_companion():
    native = _equilibrium_fixture()
    entry = SimpleNamespace(get=lambda _name, _occurrence: native)
    ods = omas_imas._load_al5_ods(
        SimpleNamespace(DBentry=entry),
        occurrence={},
        paths=["equilibrium.time_slice.0.global_quantities.ip"],
        time=None,
        imas_version="3.41.0",
        skip_uncertainties=False,
        consistency_check=False,
        verbose=False,
    )

    assert ods["equilibrium.time_slice.0.global_quantities.ip"] == 42.0
    assert ods["equilibrium.time_slice.0.global_quantities.ip_error_upper"] == 0.5
