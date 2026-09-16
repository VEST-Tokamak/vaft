"""TGLF: the GACODE suite's quasilinear turbulent-transport model.

    GACODEProfile ---> TGLFInput(rho) ---> input.tglf ---> TGLF ---> TglfOutputs
                                                                         |
                                                 solver-native source of truth

Unlike NEO, TGLF reads no ``input.gacode``: it is a *local* code, taking dimensionless
quantities at one flux surface. The projection that produces them lives in
:mod:`~vaft.code.gacode.tglf.inputs` and is held against GACODE's own ``locpargen``
rather than trusted, which is how the three convention traps in its docstring were
found.

The surrogate backend lives in :mod:`~vaft.code.gacode.tglf.surrogate` and is an
accelerated implementation of *this* contract rather than a second one: its feature
vector is built from the same :func:`tglf_parameters` keys the native run is written
from. It is imported lazily, so neither ``onnxruntime`` nor any model artifact is
needed to use the native backend.
"""

from __future__ import annotations

from ._types import (
    MAX_SPECIES,
    TGLF_DEFAULT_SCALARS,
    TGLF_DEFAULT_SPECIES,
    TGLFConfig,
    TGLFResult,
)
from .inputs import (
    LocalConversionError,
    TGLFInput,
    TGLFInputs,
    bound_deriv,
    prepare_tglf_case,
    prepare_tglf_input,
    tglf_parameters,
    write_input_tglf,
)
from .outputs import (
    GBFLUX_QUANTITIES,
    SCHEMA,
    SCHEMA_VERSION,
    TglfOutputs,
    collect_tglf_outputs,
)
from .runner import TGLFExecutionError, read_tglf_case, run_tglf, run_tglf_case

_SUBPACKAGES = ("surrogate",)


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_SUBPACKAGES])


__all__ = [
    "GBFLUX_QUANTITIES",
    "LocalConversionError",
    "MAX_SPECIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TGLFConfig",
    "TGLFExecutionError",
    "TGLFInput",
    "TGLFInputs",
    "TGLFResult",
    "TGLF_DEFAULT_SCALARS",
    "TGLF_DEFAULT_SPECIES",
    "TglfOutputs",
    "bound_deriv",
    "collect_tglf_outputs",
    "prepare_tglf_case",
    "prepare_tglf_input",
    "read_tglf_case",
    "run_tglf",
    "run_tglf_case",
    "tglf_parameters",
    "write_input_tglf",
]
