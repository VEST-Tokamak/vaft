"""TES forward-equilibrium adapter.

A Python-first wrapper around the TES (Tokamak Equilibrium Solver) ``rtes``
binary that follows the common ``vaft.code.base`` protocol:

    ods ── prepare_tes_inputs ──▶ C-format input
        ── run_tes (rtes) ──────▶ g-file / a-file / .RESULT
        ── collect_tes_outputs ─▶ ods.equilibrium  (via vaft.data.eqdsk)

Typical use::

    from vaft.code import tes
    cfg = tes.TESConfig(executable="/path/to/rtes", shot=39915, time=0.325,
                        bt0=0.15, betap_type=1, eddy=True)
    inputs = tes.prepare_tes_inputs(ods, cfg)
    result = tes.run_tes(inputs, cfg)
    eq_ods = result.ods           # equilibrium populated from the TES g-file
"""

from .config import TESConfig, TESInputs, TESResult
from .inputs import (
    prepare_tes_inputs,
    write_tes_cinput,
    write_tes_namelist,
)
from .runner import run_tes
from .outputs import (
    collect_tes_outputs,
    parse_result_scalars,
    parse_result_coils,
)
from .scan import scan_tes

__all__ = [
    "TESConfig",
    "TESInputs",
    "TESResult",
    "prepare_tes_inputs",
    "write_tes_cinput",
    "write_tes_namelist",
    "run_tes",
    "collect_tes_outputs",
    "parse_result_scalars",
    "parse_result_coils",
    "scan_tes",
]
