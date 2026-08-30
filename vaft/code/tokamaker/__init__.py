"""TokaMaker forward-equilibrium adapter.

A Python-first wrapper around the TokaMaker free-boundary Grad-Shafranov
solver from the Open FUSION Toolkit (arXiv:2311.07719) following the common
``vaft.code.base`` protocol. Unlike the subprocess adapters TokaMaker runs
in-process through ``OpenFUSIONToolkit.TokaMaker`` (imported lazily, only
when meshing/solving):

    ods ── tokamaker_geometry_from_ods ─▶ geometry dict
        ── build_tokamaker_mesh ────────▶ vest_gs_mesh_<hash>.h5   (cached)
        ── prepare_tokamaker_inputs ────▶ TokaMakerInputs
        ── run_tokamaker (in-process) ──▶ g<shot>.<time> + tokamaker_result.json
        ── collect_tokamaker_outputs ───▶ ods.equilibrium  (via vaft.data.eqdsk)

Typical use::

    from vaft.code import tokamaker
    cfg = tokamaker.TokaMakerConfig(shot=39915, time=0.325, workdir="tok_run")
    inputs = tokamaker.prepare_tokamaker_inputs(ods, cfg)
    result = tokamaker.run_tokamaker(inputs, cfg)   # builds the mesh on first use
    eq_ods = result.ods           # equilibrium populated from the g-file
"""

from .config import TokaMakerConfig, TokaMakerInputs, TokaMakerResult
from .geometry import geometry_signature, tokamaker_geometry_from_ods
from .inputs import prepare_tokamaker_inputs, resolve_mesh_file
from .mesh import build_tokamaker_mesh
from .runner import run_tokamaker
from .outputs import collect_tokamaker_outputs, parse_stats_sidecar
from .scan import scan_tokamaker

__all__ = [
    "TokaMakerConfig",
    "TokaMakerInputs",
    "TokaMakerResult",
    "tokamaker_geometry_from_ods",
    "geometry_signature",
    "prepare_tokamaker_inputs",
    "resolve_mesh_file",
    "build_tokamaker_mesh",
    "run_tokamaker",
    "collect_tokamaker_outputs",
    "parse_stats_sidecar",
    "scan_tokamaker",
]
