"""The read-only VAFT MCP tools as plain Python functions (#1423).

Each function answers one MCP tool call by delegating to an existing public
VAFT API -- ``vaft.help``, the formula and process catalogs, the validation
registry, the plot registry and its declared DD paths, the packaged samples,
the boundary registry and ``vaft.plot.extract`` -- and converting the answer
to bounded JSON.  No scientific computation happens here, and nothing here
writes a file, contacts a server, runs a solver or executes caller-supplied
code.  The functions are importable and testable without the ``mcp`` SDK;
:mod:`vaft.mcp.server` registers them as tools.

Results are dictionaries.  A list that was cut to ``limit`` says so with
``total`` and ``truncated``; arrays come back as shape, dtype, finite range and
a strided preview (see :mod:`vaft.mcp._jsonable`).

Errors are :class:`ToolInputError` (a ``ValueError``) with a message that says
what to call instead; the server turns them into MCP tool errors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._jsonable import Bounded

__all__ = ["TOOLS", "ToolInputError"]

#: Hard ceilings, whatever a caller asks for.
MAX_LIMIT = 500
MAX_POINTS = 5000
#: Longest list kept inside an extracted view model (e.g. a 2-D map's machine overlays).
MAX_EXTRACT_ITEMS = 50
DEFAULT_REFERENCE_SHOT = 39915
DEFAULT_ATLAS_GROUPS = ("efit_quality", "efit_lineage")
ATLAS_ENV = "VAFT_ATLAS_DIR"
ATLAS_SUFFIXES = (".csv", ".parquet")


class ToolInputError(ValueError):
    """A tool call VAFT cannot answer as asked; the message says why and what to try."""


def _message(error: BaseException) -> str:
    # KeyError wraps its message in quotes; the others print as is.
    return str(error.args[0]) if isinstance(error, KeyError) and error.args else str(error)


def _clamp(value: int, ceiling: int, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{name} must be an integer, got {value!r}") from None
    if value < 1:
        raise ToolInputError(f"{name} must be at least 1, got {value}")
    return min(value, ceiling)


def _page(rows: list, limit: int) -> dict[str, Any]:
    limit = _clamp(limit, MAX_LIMIT, "limit")
    return {"total": len(rows), "truncated": len(rows) > limit, "items": rows[:limit]}


def _without_source_code(row: dict) -> dict:
    """A catalog row without the function body: location stays, code does not."""
    row = dict(row)
    source = dict(row.get("source") or {})
    source.pop("code", None)
    row["source"] = source
    return row


# ---------------------------------------------------------------------------
# capability discovery (vaft.help)
# ---------------------------------------------------------------------------


def get_capabilities() -> dict[str, Any]:
    """What VAFT can do: the help topics with one-line summaries, plus the overview page.

    Start here. Each topic can be opened with get_capability(topic).
    """
    from vaft._help import help as vaft_help
    from vaft._help._registry import TOPICS

    return {
        "topics": [{"name": name, "summary": topic.summary, "has_items": bool(topic.item)}
                   for name, topic in TOPICS.items()],
        "overview": vaft_help().as_dict(),
    }


#: Per-item help answered by the matching MCP tool, so an item view never
#: carries more (or less) than the dedicated tool returns.
_ITEM_ROUTES = {
    "formula": lambda item: describe_formula(item),
    "process": lambda item: describe_process(item),
    "validation": lambda item: describe_validation_check(item),
    "plot": lambda item: list_plots(query=item),
    "data": lambda item: describe_sample(int(item)),
}


def get_capability(topic: str, item: str | None = None) -> dict[str, Any]:
    """One help topic (defaults, sections, entry points), or one item within it.

    topic is a name from get_capabilities. item, when given, is an entry of that
    topic: a formula or process name, a validation check key, a plot query, a
    sample shot number, a database source name or a CLI command.
    """
    from vaft._help import help as vaft_help
    from vaft._help import topics

    name = str(topic).strip().lower()
    if name not in topics():
        raise ToolInputError(f"unknown capability topic {topic!r}; choose from: {', '.join(topics())}")
    if item is None:
        return vaft_help(name).as_dict()
    if name in _ITEM_ROUTES:
        try:
            return {"topic": name, "item": str(item), "result": _ITEM_ROUTES[name](str(item))}
        except ValueError as error:
            raise ToolInputError(_message(error)) from None
    try:
        result = vaft_help(name, str(item))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return {"topic": name, "item": str(item), "result": Bounded(drop_keys={"code"})(result)}


# ---------------------------------------------------------------------------
# formulas and processes (their catalogs)
# ---------------------------------------------------------------------------


def _catalog_row(spec) -> dict[str, Any]:
    return {
        "id": spec.qualname,
        "name": spec.name,
        "category": spec.category,
        "signature": spec.signature,
        "summary": spec.as_dict()["summary"],
    }


def search_formulas(text: str, category: str | None = None, limit: int = 50) -> dict[str, Any]:
    """Search the formula catalog by words in names, summaries, parameters and references.

    Returns matching formulas (id, category, signature, one-line summary); open one
    with describe_formula(id). An empty text lists every formula.
    """
    from vaft.formula import catalog

    try:
        specs = catalog.search(str(text), category=category) if str(text).strip() else catalog.list_formulas(category)
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return {"query": text, "category": category, **_page([_catalog_row(s) for s in specs], limit)}


def describe_formula(name: str) -> dict[str, Any]:
    """Everything the formula catalog knows about one formula.

    Signature, parameters and returns with units, definition sections, references,
    empirical/convention flags and the source location (not the source code).
    name is a bare name ("q95") or "category.name".
    """
    from vaft.formula import catalog

    try:
        spec = catalog.describe(str(name))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return _without_source_code(spec.as_dict())


def search_processes(text: str, category: str | None = None, limit: int = 50) -> dict[str, Any]:
    """Search the processing-function catalog (signal and data processing).

    Returns matching processes (id, category, signature, one-line summary); open one
    with describe_process(id). An empty text lists every process.
    """
    from vaft.process import catalog

    try:
        specs = catalog.search(str(text), category=category) if str(text).strip() else catalog.list_processes(category)
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return {"query": text, "category": category, **_page([_catalog_row(s) for s in specs], limit)}


def describe_process(name: str) -> dict[str, Any]:
    """Everything the process catalog knows about one processing function.

    Signature, parameters, returns, provenance, machine scope and the source
    location (not the source code). name is a bare name or "category.name".
    """
    from vaft.process import catalog

    try:
        spec = catalog.describe(str(name))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return _without_source_code(spec.as_dict())


# ---------------------------------------------------------------------------
# validation registry
# ---------------------------------------------------------------------------


def _check_dict(spec) -> dict[str, Any]:
    from vaft.validation.registry import MEASURES

    row = Bounded()(spec)
    row["measure_description"] = MEASURES.get(spec.measure, "")
    return row


def list_validation_checks(category: str | None = None) -> dict[str, Any]:
    """The registered verification/validation checks, optionally of one category.

    Each check has its key, category, headline unit, provider, method, measure and
    (warn, fail) tolerance.
    """
    from vaft.validation.registry import CHECKS

    categories = sorted({spec.category for spec in CHECKS.values()})
    if category is not None and category not in categories:
        raise ToolInputError(f"no validation category {category!r}; choose from: {', '.join(categories)}")
    rows = [_check_dict(spec) for spec in CHECKS.values() if category is None or spec.category == category]
    return {"categories": categories, "total": len(rows), "items": rows}


def describe_validation_check(key: str) -> dict[str, Any]:
    """One validation check by key ("<category>.<check>"), as the registry declares it."""
    from vaft.validation.registry import describe

    try:
        return _check_dict(describe(str(key)))
    except KeyError as error:
        raise ToolInputError(_message(error)) from None


# ---------------------------------------------------------------------------
# plots: discovery, declared requirements and extraction
# ---------------------------------------------------------------------------

_PLOT_ROW_KEYS = ("name", "subject", "view", "quantity", "domain", "status", "description")


def _plot_catalog(**filters):
    from vaft.plot import available_plots

    status = filters.pop("status", "canonical")
    try:
        return available_plots(status=None if status in (None, "all") else status, **filters)
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None


def list_plots(
    query: str | None = None,
    domain: str | None = None,
    subject: str | None = None,
    view: str | None = None,
    status: str = "canonical",
    limit: int = 200,
) -> dict[str, Any]:
    """The canonical plots VAFT can draw, filtered by a semantic query or by facets.

    query resolves through the plot alias registry ("ip", "q profile", "boundary");
    domain/subject/view filter exactly; status "all" includes non-canonical plots.
    Each row names the plot; describe it with describe_plot(name).
    """
    rows = [
        {key: getattr(record, key) for key in _PLOT_ROW_KEYS}
        for record in _plot_catalog(query=query, domain=domain, subject=subject, view=view, status=status)
    ]
    return {"query": query, **_page(rows, limit)}


def describe_plot(name: str) -> dict[str, Any]:
    """Everything the plot registry knows about one canonical plot.

    Identity (subject/view/quantity, aliases), the IDS it reads, required and
    optional data paths, the view model it produces and the rendering backends.
    """
    for record in _plot_catalog(status="all"):
        if record.name == name:
            return Bounded()(record.as_dict())
    raise ToolInputError(f"no plot named {name!r}; list them with list_plots(query=...)")


def get_plot_requirements(name: str) -> dict[str, Any]:
    """The Data Dictionary paths canonical plot name reads, data paths first, without any data.

    Each path has its IDS, canonical DD path, role (data/coordinate/abscissa/...),
    coordinate and units -- the same declaration vaft.plot.dd(name) returns.
    """
    from vaft.plot import dd

    try:
        paths = dd(str(name))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return {"name": name, "paths": Bounded()(list(paths))}


def extract_plot_data(
    name: str,
    shot: int = DEFAULT_REFERENCE_SHOT,
    max_points: int = 200,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The numbers behind canonical plot name for a packaged reference shot, undrawn.

    Runs VAFT's own extraction for that plot and returns its view model: labels
    and units as stored, and every array as shape, dtype, finite min/max over all
    values and a strided preview of at most max_points values (array[::stride],
    not a resampling). options are the plot's extraction options (no rendering
    keywords). shot must be one of list_samples().
    """
    from vaft.plot import extract

    from ._source import load_reference_shot

    from vaft.plot.registry import get_spec

    points = _clamp(max_points, MAX_POINTS, "max_points")
    try:
        get_spec(str(name))
    except KeyError:
        raise ToolInputError(f"no plot named {name!r}; list them with list_plots(query=...)") from None
    if options is not None and not isinstance(options, dict):
        raise ToolInputError(f"options must be an object of extraction keywords, got {type(options).__name__}")
    source = load_reference_shot(shot)
    try:
        model = extract(str(name), source, **dict(options or {}))
    except AttributeError as error:
        if f"extract_{name}" not in str(error):
            raise
        raise ToolInputError(f"plot {name!r} has no extraction view (a composite or rendering-only plot)") from None
    except (KeyError, ValueError, TypeError) as error:
        raise ToolInputError(f"{name} on shot {shot}: {_message(error)}") from None
    bounded = Bounded(max_items=MAX_EXTRACT_ITEMS, max_points=points)
    data = bounded(model)
    return {
        "name": name,
        "shot": int(shot),
        "model": type(model).__name__,
        "max_points": points,
        "truncated": bounded.truncated,
        "data": data,
    }


# ---------------------------------------------------------------------------
# packaged samples
# ---------------------------------------------------------------------------


def list_samples() -> dict[str, Any]:
    """The packaged reference shots, with what each contains and whether its data is installed here."""
    from vaft.data import available_samples, sample, sample_manifest

    rows = []
    for shot in available_samples():
        manifest = sample_manifest(shot)
        try:
            installed = Path(sample(shot, representation="omas")).is_file()
        except (FileNotFoundError, KeyError, ValueError):
            installed = False
        pipeline = manifest.get("pipeline", {}) if isinstance(manifest, dict) else {}
        rows.append({
            "shot": int(shot),
            "reference_id": manifest.get("reference_id"),
            "machine": manifest.get("machine"),
            "scope": list(pipeline.get("scope", ())),
            "installed": installed,
        })
    return {"total": len(rows), "items": rows}


def describe_sample(shot: int) -> dict[str, Any]:
    """The full manifest of one packaged reference shot: provenance, pipeline scope and contents."""
    from vaft.data import available_samples, sample_manifest

    shot = int(shot)
    if shot not in tuple(int(s) for s in available_samples()):
        raise ToolInputError(f"no packaged reference shot {shot}; available: {list(available_samples())}")
    return Bounded()(sample_manifest(shot))


# ---------------------------------------------------------------------------
# operational boundaries
# ---------------------------------------------------------------------------


def list_boundaries(family: str | None = None) -> dict[str, Any]:
    """The registered operational boundaries (density, beta, current, L-H limits), optionally of one family."""
    from vaft.formula.boundaries import get_boundary
    from vaft.formula.boundaries import list_boundaries as keys

    rows = []
    for key in keys(family):
        entry = get_boundary(key)
        target = getattr(entry, "target", None)
        rows.append({
            "key": key,
            "kind": type(entry).__name__,
            "family": getattr(entry, "family", ""),
            "target": getattr(target, "name", ""),
            "unit": getattr(target, "unit", ""),
            "form": getattr(entry, "form", ""),
            "allowed_side": getattr(entry, "allowed_side", ""),
            "regime": getattr(entry, "regime", ""),
        })
    families = sorted({getattr(get_boundary(k), "family", "") for k in keys()})
    if family is not None and not rows:
        raise ToolInputError(f"no boundary family {family!r}; choose from: {', '.join(families)}")
    return {"families": families, "total": len(rows), "items": rows}


def describe_boundary(key: str) -> dict[str, Any]:
    """One registered boundary: target and inputs with units, form, sources, applicability, uncertainty."""
    from vaft.formula.boundaries import get_boundary

    try:
        entry = get_boundary(str(key))
    except KeyError as error:
        raise ToolInputError(_message(error)) from None
    return {"kind": type(entry).__name__, **Bounded()(entry)}


# ---------------------------------------------------------------------------
# operating-space atlas tables (local, opt-in directory)
# ---------------------------------------------------------------------------


def _atlas_root() -> Path:
    configured = os.environ.get(ATLAS_ENV, "").strip()
    if not configured:
        raise ToolInputError(
            f"{ATLAS_ENV} is not set: point it at the directory holding the atlas tables "
            f"(.csv/.parquet) before starting the server"
        )
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise ToolInputError(f"{ATLAS_ENV}={configured!r} is not a directory")
    return root.resolve(strict=True)


def _atlas_tables(root: Path) -> list[str]:
    tables = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in ATLAS_SUFFIXES and path.is_file():
            try:
                path.resolve(strict=True).relative_to(root)
            except ValueError:  # a symlink leading out of the directory
                continue
            tables.append(path.relative_to(root).as_posix())
    return tables


def _atlas_table(root: Path, path: str | None) -> Path:
    if path is None:
        tables = _atlas_tables(root)
        if len(tables) != 1:
            raise ToolInputError(
                f"{len(tables)} atlas tables under {ATLAS_ENV}; pass path= one of: {tables[:50]}"
            )
        path = tables[0]
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ToolInputError(f"no atlas table {path!r} under {ATLAS_ENV}") from None
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ToolInputError(f"atlas table {path!r} is outside {ATLAS_ENV}; only files under it are read") from None
    if not resolved.is_file() or resolved.suffix.lower() not in ATLAS_SUFFIXES:
        raise ToolInputError(f"atlas table {path!r} is not a {' or '.join(ATLAS_SUFFIXES)} file")
    return resolved


def get_atlas_summary(
    path: str | None = None,
    group_by: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Summarise an operating-space atlas table from the directory named by VAFT_ATLAS_DIR.

    Reads one .csv or .parquet file under that directory (path is relative to it;
    omit it when the directory holds exactly one table) and returns its columns,
    row count, row counts per value of each group_by column (default efit_quality
    and efit_lineage) and its first limit rows. Files outside the directory are refused.
    """
    import pandas as pd

    root = _atlas_root()
    table = _atlas_table(root, path)
    try:
        frame = pd.read_parquet(table) if table.suffix.lower() == ".parquet" else pd.read_csv(table)
    except ImportError as error:
        raise ToolInputError(f"reading {table.name} needs an optional dependency: {error}") from None
    groups = list(DEFAULT_ATLAS_GROUPS if group_by is None else group_by)
    counts: dict[str, Any] = {}
    missing = []
    bounded = Bounded(max_items=MAX_LIMIT)
    for column in groups:
        if column not in frame.columns:
            missing.append(column)
            continue
        tally = frame[column].value_counts(dropna=False)
        counts[column] = [
            {"value": bounded(None if pd.isna(value) else value), "rows": int(rows)}
            for value, rows in list(tally.items())[:MAX_LIMIT]
        ]
    head = frame.head(_clamp(limit, MAX_LIMIT, "limit"))
    rows = [
        {str(k): bounded(None if _isna(v) else (v.isoformat() if hasattr(v, "isoformat") else v))
         for k, v in record.items()}
        for record in head.to_dict(orient="records")
    ]
    return {
        "path": table.relative_to(root).as_posix(),
        "rows": int(len(frame)),
        "columns": [{"name": str(c), "dtype": str(t)} for c, t in frame.dtypes.items()],
        "group_by": groups,
        "missing_group_columns": missing,
        "counts": counts,
        "head": rows,
        "head_truncated": len(frame) > len(head),
    }


def _isna(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # array-like cells
        return False


#: Every tool the server exposes, in listing order.  Each is read-only.
TOOLS = (
    get_capabilities,
    get_capability,
    search_formulas,
    describe_formula,
    search_processes,
    describe_process,
    list_validation_checks,
    describe_validation_check,
    list_plots,
    describe_plot,
    get_plot_requirements,
    extract_plot_data,
    list_samples,
    describe_sample,
    list_boundaries,
    describe_boundary,
    get_atlas_summary,
)
