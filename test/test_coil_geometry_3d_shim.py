"""The old ``coil_geometry_3d`` name keeps working, warns, and re-exports the same objects."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

CANONICAL = "vaft.machine_mapping.coils_non_axisymmetric_geometry"
LEGACY = "vaft.machine_mapping.coil_geometry_3d"


def test_shim_warns_and_reexports_the_same_objects():
    sys.modules.pop(LEGACY, None)
    with pytest.warns(DeprecationWarning, match="coils_non_axisymmetric_geometry"):
        legacy = importlib.import_module(LEGACY)
    canonical = importlib.import_module(CANONICAL)
    assert list(legacy.__all__) == list(canonical.__all__)
    assert legacy.__all__ is not canonical.__all__
    for name in canonical.__all__:
        assert getattr(legacy, name) is getattr(canonical, name), name
    # Support names callers historically imported from the old module.
    from vaft.machine_mapping.utils import VestConfigurationError

    assert legacy.VestConfigurationError is VestConfigurationError
    assert legacy.data_path is canonical.data_path


def test_canonical_module_imports_without_warnings():
    # A fresh interpreter is the only way to exercise the real first import.
    subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", f"import {CANONICAL}"],
        check=True,
        capture_output=True,
        text=True,
    )
