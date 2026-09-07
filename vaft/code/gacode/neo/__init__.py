"""NEO: the GACODE suite's drift-kinetic neoclassical solver.

    GACODEProfile ---> input.gacode + input.neo ---> NEO ---> NeoOutputs
                                                                  |
                                          solver-native source of truth

``NeoOutputs`` is the whole run, in NEO's own units and on NEO's grid.  It is
deliberately *not* an IDS: the audit of which quantities have a defensible
IMAS home is phase 5 of issue #550, and until it is done a mapping would be
name-matching rather than physics.

The VAFT-native analytic counterpart is :mod:`vaft.formula.neoclassical`, kept
a separate computational identity on purpose: Sauter and Redl answer a related
but different question from a drift-kinetic solve, and hiding both behind one
``model=`` switch would conceal that.
"""

from __future__ import annotations

from ._types import NEO_DEFAULTS, NEOConfig, NEOResult
from .inputs import NEOInputs, neo_parameters, prepare_neo_case, write_input_neo
from .outputs import (
    SCHEMA,
    SCHEMA_VERSION,
    THEORY_SCALARS,
    NeoGrid,
    NeoNormalisation,
    NeoOutputs,
    collect_neo_outputs,
)
from .runner import NEOExecutionError, read_neo_case, run_neo, run_neo_case

__all__ = [
    "NEOConfig",
    "NEOExecutionError",
    "NEOInputs",
    "NEOResult",
    "NEO_DEFAULTS",
    "NeoGrid",
    "NeoNormalisation",
    "NeoOutputs",
    "SCHEMA",
    "SCHEMA_VERSION",
    "THEORY_SCALARS",
    "collect_neo_outputs",
    "neo_parameters",
    "prepare_neo_case",
    "read_neo_case",
    "run_neo",
    "run_neo_case",
    "write_input_neo",
]
