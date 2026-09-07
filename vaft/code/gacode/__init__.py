"""Adapters for the GACODE suite: shared runtime, profiles, and NEO.

GACODE is one source tree carrying several solvers, so this package owns the
boundary once rather than once per solver:

    equilibrium + core_profiles
              |
              v
        GACODEProfile            vaft.code.gacode
              |
              v
        input.gacode             the suite's shared profile spine
              |
      +-------+--------+
      v                v
    NEO              TGLF / CGYRO   (issue #553)

``input.gacode`` is an interoperability format, not VAFT's kinetic state: the
canonical state stays in IMAS/OMAS and is converted deterministically here.

Importing this package does not require GACODE.  Executable and platform
resolution happen when something is run, so ``from vaft.code import *`` and the
whole test suite work with the suite absent.

Typical use::

    from vaft.code.gacode import GACODEConfig, neo

    config = GACODEConfig(home="~/git/gacode", platform="GFORTRAN_OSX_BREW")
    result = neo.run_neo_case(profile, workdir="runs/48224", config=config)
    result.outputs_native.bootstrap_current_parallel
"""

from __future__ import annotations

from ._runtime import (
    available_platforms,
    find_gacode_executable,
    gacode_environment,
    gacode_home,
    gacode_platform,
    launcher_relative_path,
    require_gacode_executable,
    run_gacode,
)
from ._types import (
    GACODE_COMPATIBILITY_ENVS,
    GACODE_HOME_ENV,
    GACODE_PLATFORM_ENV,
    GACODE_ROOT_ENV,
    GACODEConfig,
    SUITE_CODES,
    SUPPORTED_CODES,
)

__all__ = [
    "GACODEConfig",
    "GACODE_COMPATIBILITY_ENVS",
    "GACODE_HOME_ENV",
    "GACODE_PLATFORM_ENV",
    "GACODE_ROOT_ENV",
    "SUITE_CODES",
    "SUPPORTED_CODES",
    "available_platforms",
    "find_gacode_executable",
    "gacode_environment",
    "gacode_home",
    "gacode_platform",
    "launcher_relative_path",
    "require_gacode_executable",
    "run_gacode",
]


def __getattr__(name: str):
    # `neo` is a subpackage, imported on first use so that `vaft.code.gacode`
    # itself stays as light as the rest of `vaft.code`.
    if name == "neo":
        from importlib import import_module

        module = import_module(".neo", __name__)
        globals()["neo"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, "neo"])
