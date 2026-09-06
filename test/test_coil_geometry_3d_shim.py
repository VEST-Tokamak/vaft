"""The old ``coil_geometry_3d`` name keeps working, warns once, and re-exports the same objects."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


def test_shim_warns_and_reexports_the_same_objects():
    sys.modules.pop("vaft.machine_mapping.coil_geometry_3d", None)
    with pytest.warns(DeprecationWarning, match="coils_non_axisymmetric_geometry"):
        legacy = importlib.import_module("vaft.machine_mapping.coil_geometry_3d")
    canonical = importlib.import_module("vaft.machine_mapping.coils_non_axisymmetric_geometry")
    for name in canonical.__all__:
        assert getattr(legacy, name) is getattr(canonical, name), name


def test_canonical_module_imports_without_warnings():
    sys.modules.pop("vaft.machine_mapping.coils_non_axisymmetric_geometry", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        importlib.import_module("vaft.machine_mapping.coils_non_axisymmetric_geometry")
