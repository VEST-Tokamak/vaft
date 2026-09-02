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
    production_qa.py  FileDB production-stage QA policy and artifact orchestration

The dependency direction runs one way: production QA consumes validation, and
the validation core knows nothing about FileDB paths, Snakemake outputs, figure
persistence or artifact hashing.  ``production_qa`` is therefore reached
lazily, so ``import vaft.validation`` costs nothing but the vocabulary, and no
plotting backend is imported until a figure is actually asked for.
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

#: Names ``production_qa`` owns, re-exported here for the workflow and test
#: modules that have always imported them from ``vaft.validation``.
_PRODUCTION_QA_EXPORTS = (
    "STAGE_METRICS",
    "STAGE_PRECONDITIONS",
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
    *_PRODUCTION_QA_EXPORTS,
]


def __getattr__(name: str):
    """Resolve production-QA names, and submodules, on first use.

    Anything ``production_qa`` defines resolves here, not just its ``__all__``:
    the migration from the former flat ``vaft/validation.py`` must not break a
    caller that reached for one of its helpers by name.
    """
    if name in _SUBMODULES:
        from importlib import import_module

        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    from . import production_qa

    try:
        value = getattr(production_qa, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
