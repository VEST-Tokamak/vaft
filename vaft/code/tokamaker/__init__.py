"""TokaMaker equilibrium adapter (static, quasi-static, and stability).

A Python-first wrapper around the TokaMaker free-boundary Grad-Shafranov
solver from the Open FUSION Toolkit (arXiv:2311.07719) following the common
``vaft.code.base`` protocol. Unlike the subprocess adapters TokaMaker runs
in-process through ``OpenFUSIONToolkit.TokaMaker`` (imported lazily, only
when meshing/solving):

    ods ── tokamaker_geometry_from_ods ─▶ geometry dict (+ vessel conductors)
        ── build_tokamaker_mesh ────────▶ vest_gs_mesh_<hash>.h5   (cached)
        ── prepare_tokamaker_inputs ────▶ TokaMakerInputs
        ── run_tokamaker (in-process) ──▶ g<shot>.<time> + tokamaker_result.json
        ── collect_tokamaker_outputs ───▶ ods.equilibrium  (via vaft.data.eqdsk)

Time-dependent branches (require ``include_vessel=True`` so the VEST vessel
is meshed as resistive conductor regions built from ``pf_passive``):

        ── prepare_tokamaker_evolution_inputs ─▶ TokaMakerEvolutionInputs
        ── run_tokamaker_evolution ────────────▶ per-slice g-files + eddy currents
                                                 ──▶ merged multi-slice equilibrium IDS
        ── run_tokamaker_wall_eigenmodes ──────▶ wall L/R time constants (eig_wall)
        ── run_tokamaker_vertical_stability ───▶ n=0 growth rate gamma (eig_td)

Typical use::

    from vaft.code import tokamaker
    cfg = tokamaker.TokaMakerConfig(shot=39915, time=0.325, workdir="tok_run")
    inputs = tokamaker.prepare_tokamaker_inputs(ods, cfg)
    result = tokamaker.run_tokamaker(inputs, cfg)   # builds the mesh on first use
    eq_ods = result.ods           # equilibrium populated from the g-file
"""

from .config import (
    TokaMakerConfig,
    TokaMakerEvolutionInputs,
    TokaMakerEvolutionResult,
    TokaMakerInputs,
    TokaMakerResult,
    TokaMakerStabilityResult,
    TokaMakerStepRecord,
)
from .geometry import geometry_signature, tokamaker_geometry_from_ods
from .vessel import vessel_segments_from_ods
from .inputs import (
    prepare_tokamaker_evolution_inputs,
    prepare_tokamaker_inputs,
    resolve_mesh_file,
)
from .mesh import build_tokamaker_mesh
from .runner import run_tokamaker
from .evolve import run_tokamaker_evolution
from .stability import run_tokamaker_vertical_stability, run_tokamaker_wall_eigenmodes
from .outputs import (
    collect_tokamaker_evolution_outputs,
    collect_tokamaker_outputs,
    collect_tokamaker_stability_outputs,
    parse_stats_sidecar,
)
from .scan import scan_tokamaker

__all__ = [
    "TokaMakerConfig",
    "TokaMakerInputs",
    "TokaMakerResult",
    "TokaMakerEvolutionInputs",
    "TokaMakerEvolutionResult",
    "TokaMakerStepRecord",
    "TokaMakerStabilityResult",
    "tokamaker_geometry_from_ods",
    "geometry_signature",
    "vessel_segments_from_ods",
    "prepare_tokamaker_inputs",
    "prepare_tokamaker_evolution_inputs",
    "resolve_mesh_file",
    "build_tokamaker_mesh",
    "run_tokamaker",
    "run_tokamaker_evolution",
    "run_tokamaker_wall_eigenmodes",
    "run_tokamaker_vertical_stability",
    "collect_tokamaker_outputs",
    "collect_tokamaker_evolution_outputs",
    "collect_tokamaker_stability_outputs",
    "parse_stats_sidecar",
    "scan_tokamaker",
]
