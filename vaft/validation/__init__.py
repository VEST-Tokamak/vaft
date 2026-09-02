"""VAFT's scientific assessment layer (issue #253).

``vaft.validation`` evaluates the credibility of data and computational results;
it does not generate them.  The calculations stay with their domain providers --
:mod:`vaft.omas.efit_quality` for EFIT residuals and chi-square,
:mod:`vaft.omas.vacuum_magnetics` for synthetic vacuum response,
:mod:`vaft.process.equilibrium` for equilibrium physics kernels,
:mod:`vaft.formula.statistics` for the statistics themselves -- and this package
composes and interprets what they produce.

Verification lives here as a *category* of validation rather than as a separate
top-level namespace: "was the calculation performed as intended?" and "is the
result credible for its intended use?" are two questions about one object.  See
:data:`vaft.validation.model.CATEGORIES`.

Layout::

    model.py          the status vocabulary and taxonomy
    imas.py           native IMAS/OMAS validity and status fields
    stage_evidence.py per-stage preconditions and metrics, composed from domain providers

The dependency direction runs one way: :mod:`vaft.database.production_qa`
consumes ``stage_evidence`` to decide which figures a stage owes and how to
persist them, and the validation core knows nothing about FileDB paths,
Snakemake outputs, figure persistence or artifact hashing.  Every submodule is
reached lazily, so ``import vaft.validation`` costs nothing but the vocabulary,
and no plotting backend or database layer is imported until asked for.
"""

from __future__ import annotations

from pathlib import Path as _Path

from .model import CATEGORIES, ValidationStatus

#: The package's own submodules, discovered rather than listed, so a new domain
#: module is reachable as ``vaft.validation.<name>`` the moment it exists and
#: cannot be shadowed by a stale hand-maintained tuple.
_SUBMODULES = frozenset(
    path.stem for path in _Path(__file__).parent.glob("*.py") if path.stem != "__init__"
)

#: Per-stage evidence -- preconditions and metrics -- which still lives in this
#: package (``stage_evidence``) and is re-exported here as it always was.
_EVIDENCE_EXPORTS = (
    "STAGE_METRICS",
    "STAGE_PRECONDITIONS",
)

#: The stage-QA *artifact* contract -- which figures a stage owes, their names,
#: the renderer, the manifest.  That is a database-layer question and moved to
#: :mod:`vaft.database.production_qa` (issue #338).  Reaching it through this
#: package still works, and warns, so the workflow and any external caller are
#: told where it went rather than broken.
_ARTIFACT_EXPORTS = (
    "STAGE_VALIDATION_PLOTS",
    "ValidationPlot",
    "mhd_linear_run_coverage_model",
    "raw_acquisition_qa_model",
    "render_stage_plots",
    "stage_plot_filenames",
    "stages",
    "validation_plots",
)

__all__ = [
    "CATEGORIES",
    "ValidationStatus",
    *_EVIDENCE_EXPORTS,
    *_ARTIFACT_EXPORTS,
]


def __getattr__(name: str):
    """Resolve submodules, stage evidence, and the moved artifact names on first use.

    Evidence names -- and any private helper ``stage_evidence`` defines, such as
    the ``_efit_metrics`` one test reaches for by name -- resolve silently:
    they still live in this package.  Artifact names resolve too, but through a
    :class:`DeprecationWarning`, because they now live in
    :mod:`vaft.database.production_qa`; that import is deliberately lazy so a
    plain ``import vaft.validation`` never pulls the database layer in.
    """
    if name in _SUBMODULES:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _ARTIFACT_EXPORTS:
        import warnings

        warnings.warn(
            f"vaft.validation.{name} moved to vaft.database.production_qa (issue "
            "#338): it is stage-QA artifact policy, not scientific validation. "
            f"Import it from there.",
            DeprecationWarning,
            stacklevel=2,
        )
        from vaft.database import production_qa

        value = getattr(production_qa, name)
        globals()[name] = value
        return value

    from . import stage_evidence

    try:
        value = getattr(stage_evidence, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
