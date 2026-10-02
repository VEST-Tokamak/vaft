"""The read-only VAFT MCP tools as plain Python functions (#1423).

Each function answers one MCP tool call by delegating to an existing public
VAFT API -- ``vaft.help``, the formula and process catalogs, the validation
registry, the plot registry and its declared DD paths, the packaged samples,
the boundary registry and ``vaft.plot.extract`` -- and converting the answer
to bounded JSON.  No scientific computation happens here, and nothing here
writes a file, contacts a server, runs a solver or executes caller-supplied
code.  The functions are importable and testable without the ``mcp`` SDK;
:mod:`vaft.mcp.server` registers them as tools.

Results are dictionaries, and every one carries ``truncated``: ``count`` of
the places something was shortened (a list cut to ``limit``, a long string, an
array preview) and the first of those ``paths``.  Arrays come back as shape,
dtype, finite range and a strided preview (see :mod:`vaft.mcp._jsonable`).

Errors are :class:`ToolInputError` (a ``ValueError``) with a message that says
what to call instead; the server turns them into MCP tool errors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._jsonable import REPORTED_PATHS, Bounded, bounded_json

__all__ = ["TOOLS", "ToolInputError"]

#: Hard ceilings, whatever a caller asks for.
MAX_LIMIT = 500
MAX_POINTS = 5000
#: Longest list kept inside an extracted view model (e.g. a 2-D map's machine overlays).
MAX_EXTRACT_ITEMS = 50
#: Byte cap on one extraction result; beyond it arrays are reduced to statistics.
MAX_EXTRACT_BYTES = 50_000
#: Atlas bounds: file size, listed columns, count columns, cell text.
MAX_ATLAS_BYTES = 50 * 1024 * 1024
MAX_ATLAS_COLUMNS = 200
MAX_ATLAS_GROUPS = 3
MAX_ATLAS_CELL = 200
DEFAULT_REFERENCE_SHOT = 39915
ATLAS_ENV = "VAFT_ATLAS_DIR"
#: Longest lane README carried by describe_atlas_table.
MAX_README = 12_000


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


def _page(rows: list, limit: int) -> tuple[dict[str, Any], list[str]]:
    """The first ``limit`` rows, the total, and a truncation note when rows were cut."""
    limit = _clamp(limit, MAX_LIMIT, "limit")
    notes = [f"$.items ({len(rows)} items, kept {limit})"] if len(rows) > limit else []
    return {"total": len(rows), "items": rows[:limit]}, notes


def _finish(payload: Any, *, converter: Bounded | None = None, notes=(), converted: bool = False) -> dict[str, Any]:
    """``payload`` as bounded JSON with its ``truncated`` record.

    A dictionary result gets ``truncated`` beside its own keys; anything else
    (or a dictionary that already uses that key) is returned under ``result``.
    """
    converter = converter or Bounded()
    data = _redacted(payload if converted else converter(payload))
    cut = list(notes) + converter.truncated
    record = {"count": len(cut), "paths": cut[:REPORTED_PATHS]}
    if isinstance(data, dict) and "truncated" not in data:
        return {**data, "truncated": record}
    return {"result": data, "truncated": record}


#: Environment variables whose values never leave the server, whatever printed them.
SECRET_ENV = ("HS_PASSWORD", "HS_API_KEY")


def _redactions() -> list[tuple[str, str]]:
    """(text, replacement) pairs: secret values first, then the home directory as ``~``."""
    pairs = [(value, "<redacted>") for name in SECRET_ENV if len(value := os.environ.get(name, "")) >= 4]
    home = str(Path.home())
    if len(home) > 1:
        pairs.append((home, "~"))
    return pairs


def _redacted(data: Any, pairs: list[tuple[str, str]] | None = None) -> Any:
    """``data`` (already plain JSON) with secrets and the home directory scrubbed from every string.

    Help pages report where a configuration file lives and which variables are
    set; an agent needs the fact, not the account's absolute path or a value.
    """
    pairs = _redactions() if pairs is None else pairs
    if isinstance(data, str):
        for text, replacement in pairs:
            data = data.replace(text, replacement)
        return data
    if isinstance(data, dict):
        return {_redacted(key, pairs): _redacted(value, pairs) for key, value in data.items()}
    if isinstance(data, list):
        return [_redacted(value, pairs) for value in data]
    return data


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
    from vaft._help import topics

    overview = vaft_help().as_dict()
    summaries = {label: text for section in overview["sections"] if section["title"] == "Topics"
                 for label, text in section["rows"]}
    return _finish({
        "topics": [{"name": name, "summary": summaries.get(name, "")} for name in topics()],
        "overview": overview,
    })


def _shot_number(item: str) -> int:
    try:
        return int(str(item).strip().lstrip("#"))
    except ValueError:
        raise ToolInputError(f"a data item is a sample shot number such as 39915, got {item!r}") from None


#: Per-item help answered by the matching MCP tool, so an item view never
#: carries more (or less) than the dedicated tool returns.
_ITEM_ROUTES = {
    "formula": lambda item: describe_formula(item),
    "process": lambda item: describe_process(item),
    "validation": lambda item: describe_validation_check(item),
    "plot": lambda item: list_plots(query=item),
    "data": lambda item: describe_sample(_shot_number(item)),
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
        return _finish(vaft_help(name).as_dict())
    if name in _ITEM_ROUTES:
        try:
            routed = dict(_ITEM_ROUTES[name](str(item)))
        except ValueError as error:
            raise ToolInputError(_message(error)) from None
        record = routed.pop("truncated")
        return {"topic": name, "item": str(item), "result": routed, "truncated": record}
    try:
        result = vaft_help(name, str(item))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return _finish({"topic": name, "item": str(item), "result": result},
                   converter=Bounded(drop_keys={"code"}))


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
    page, notes = _page([_catalog_row(s) for s in specs], limit)
    return _finish({"query": text, "category": category, **page}, notes=notes)


def describe_formula(name: str) -> dict[str, Any]:
    """Everything the formula catalog knows about one formula.

    Signature, parameters and returns with units, definition sections, references,
    empirical/convention flags and the source location (not the source code).
    name is a bare name or "category.name" (see search_formulas).
    """
    from vaft.formula import catalog

    try:
        spec = catalog.describe(str(name))
    except (KeyError, ValueError) as error:
        raise ToolInputError(_message(error)) from None
    return _finish(_without_source_code(spec.as_dict()))


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
    page, notes = _page([_catalog_row(s) for s in specs], limit)
    return _finish({"query": text, "category": category, **page}, notes=notes)


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
    return _finish(_without_source_code(spec.as_dict()))


# ---------------------------------------------------------------------------
# validation registry
# ---------------------------------------------------------------------------


def _check_dict(spec) -> dict[str, Any]:
    import dataclasses

    from vaft.validation.registry import MEASURES

    row = dataclasses.asdict(spec)
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
    return _finish({"categories": categories, "total": len(rows), "items": rows})


def describe_validation_check(key: str) -> dict[str, Any]:
    """One validation check by key ("<category>.<check>"), as the registry declares it."""
    from vaft.validation.registry import describe

    try:
        return _finish(_check_dict(describe(str(key))))
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
    page, notes = _page(rows, limit)
    return _finish({"query": query, **page}, notes=notes)


def describe_plot(name: str) -> dict[str, Any]:
    """Everything the plot registry knows about one canonical plot.

    Identity (subject/view/quantity, aliases), the IDS it reads, required and
    optional data paths, the view model it produces and the rendering backends.
    """
    for record in _plot_catalog(status="all"):
        if record.name == name:
            return _finish(record.as_dict())
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
    return _finish({"name": name, "paths": list(paths)})


def extraction_option_names() -> tuple[str, ...]:
    """The extraction option keys ``extract_plot_data`` accepts, from VAFT's option schema."""
    from vaft.plot.backend.options import EXTRACTION_OPTIONS

    return tuple(sorted(EXTRACTION_OPTIONS))


def _checked_options(options: Any) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ToolInputError(f"options must be an object of extraction keywords, got {type(options).__name__}")
    allowed = set(extraction_option_names())
    refused = sorted(str(k) for k in options if not isinstance(k, str) or k.startswith("_") or k not in allowed)
    if refused:
        raise ToolInputError(
            f"options {refused} are not extraction options; allowed: {', '.join(sorted(allowed))}"
        )
    return dict(options)


def extract_plot_data(
    name: str,
    shot: int = DEFAULT_REFERENCE_SHOT,
    max_points: int = 200,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The numbers behind canonical plot name for a packaged reference shot, undrawn.

    Runs VAFT's own extraction for that plot and returns its view model: labels
    and units as stored, and every array as shape, dtype and finite min/max over
    all values, plus a strided preview (array[::stride], not a resampling).
    max_points is the budget for the whole result, shared across its arrays; a
    result over about 50 kB keeps array statistics only. options are the plot's
    extraction options (no rendering keywords). shot must be one of list_samples().
    """
    from vaft.plot import extract
    from vaft.plot.registry import get_spec

    from ._source import load_reference_shot

    points = _clamp(max_points, MAX_POINTS, "max_points")
    try:
        get_spec(str(name))
    except KeyError:
        raise ToolInputError(f"no plot named {name!r}; list them with list_plots(query=...)") from None
    keywords = _checked_options(options)
    source = load_reference_shot(shot)
    try:
        model = extract(str(name), source, **keywords)
    except AttributeError as error:
        if f"extract_{name}" not in str(error):
            raise
        raise ToolInputError(f"plot {name!r} has no extraction view (a composite or rendering-only plot)") from None
    except (LookupError, ValueError, TypeError) as error:
        raise ToolInputError(f"{name} on shot {shot}: {type(error).__name__}: {_message(error)}") from None
    data, converter = bounded_json(model, budget=points, max_bytes=MAX_EXTRACT_BYTES, max_items=MAX_EXTRACT_ITEMS)
    payload = {"name": name, "shot": int(shot), "model": type(model).__name__, "max_points": points, "data": data}
    return _finish(payload, converter=converter, converted=True)


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
    return _finish({"total": len(rows), "items": rows})


def describe_sample(shot: int) -> dict[str, Any]:
    """The full manifest of one packaged reference shot: provenance, pipeline scope and contents."""
    from vaft.data import available_samples, sample_manifest

    shot = int(shot)
    if shot not in tuple(int(s) for s in available_samples()):
        raise ToolInputError(f"no packaged reference shot {shot}; available: {list(available_samples())}")
    return _finish(sample_manifest(shot))


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
    return _finish({"families": families, "total": len(rows), "items": rows})


def describe_boundary(key: str) -> dict[str, Any]:
    """One registered boundary: target and inputs with units, form, sources, applicability, uncertainty."""
    from vaft.formula.boundaries import get_boundary

    try:
        entry = get_boundary(str(key))
    except KeyError as error:
        raise ToolInputError(_message(error)) from None
    converter = Bounded()
    return _finish({"kind": type(entry).__name__, **converter(entry)}, converter=converter, converted=True)


# ---------------------------------------------------------------------------
# datasets: packaged shots, local artifacts, database shots (_source)
# ---------------------------------------------------------------------------


def _dataset(shot, artifact, database_source):
    from ._paths import PathRefused
    from ._source import load_dataset

    try:
        return load_dataset(shot=shot, artifact=artifact, database_source=database_source)
    except PathRefused as error:
        raise ToolInputError(str(error)) from None


def _times(ods, ids: str):
    import numpy as np

    from vaft.ods_access import path_value

    value = path_value(ods, f"{ids}.time", None)
    if value is None:
        return None
    try:
        times = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        return None
    return times


def inspect_dataset(
    shot: int | None = None,
    artifact: str | None = None,
    database_source: str | None = None,
) -> dict[str, Any]:
    """What one dataset holds: each IDS with its number of times and time range, plus provenance.

    Name the dataset by shot (a packaged sample, see list_samples; other shots are
    read from the database only when the server enables it) or by artifact, a path
    relative to VAFT_ARTIFACT_DIR (g-file, OMAS JSON, IMAS netCDF or HDF5).
    database_source picks a database namespace (e.g. 'main') for a shot.
    """
    import numpy as np

    dataset = _dataset(shot, artifact, database_source)
    ods = dataset.data
    rows = []
    for ids in sorted(str(k) for k in ods.keys()):
        times = _times(ods, ids)
        finite = times[np.isfinite(times)] if times is not None else None
        rows.append({
            "ids": ids,
            "times": None if times is None else int(times.size),
            "time_first_s": float(finite.min()) if finite is not None and finite.size else None,
            "time_last_s": float(finite.max()) if finite is not None and finite.size else None,
        })
    return _finish({"dataset": dataset.provenance, "total": len(rows), "items": rows})


def _slice_index(times, time: float, tolerance: float | None) -> tuple[int, float, float]:
    """The slice nearest ``time``, refused beyond ``tolerance`` (default: half the median spacing)."""
    import numpy as np

    if times is None or not np.isfinite(times).any():
        raise ToolInputError("this dataset has no slice times to match against")
    if tolerance is None:
        spacing = np.diff(np.sort(times[np.isfinite(times)]))
        tolerance = 0.5 * float(np.median(spacing)) if spacing.size else 1e-3
    tolerance = float(tolerance)
    distance = np.abs(np.where(np.isfinite(times), times, np.inf) - float(time))
    index = int(np.argmin(distance))
    if distance[index] > tolerance:
        raise ToolInputError(
            f"no slice within {tolerance:.3g} s of t = {float(time):.6g} s (nearest {float(times[index]):.6g} s); "
            f"list_equilibrium_times gives the slice times, or pass a wider tolerance"
        )
    return index, float(times[index]), tolerance


def inspect_data_path(
    path: str,
    shot: int | None = None,
    artifact: str | None = None,
    database_source: str | None = None,
    time: float | None = None,
    tolerance: float | None = None,
    max_points: int = 200,
) -> dict[str, Any]:
    """The value at one Data Dictionary path of a dataset, bounded, without creating anything.

    path is dotted, e.g. 'equilibrium.time_slice.*.global_quantities.q_95'. A '*'
    stands for the time-slice index and needs time (seconds): the slice nearest that
    time is used, refused beyond tolerance (default half the slice spacing), and the
    matched time is reported. Arrays come back as shape, finite range and a strided
    preview of at most max_points values. An absent path answers present: false.
    """
    import numpy as np

    from vaft.ods_access import path_exists, path_value

    text = str(path).strip()
    if not text or len(text) > 300 or text.count("*") > 1:
        raise ToolInputError("path is one dotted Data Dictionary path with at most one '*'")
    points = _clamp(max_points, MAX_POINTS, "max_points")
    dataset = _dataset(shot, artifact, database_source)
    ods = dataset.data
    matched = None
    if "*" in text:
        if time is None:
            raise ToolInputError("a '*' in path is a time-slice index: pass time (s) to pick the slice")
        head = text.split("*", 1)[0].rstrip(".")
        ids = head.split(".", 1)[0]
        times = _times(ods, ids)
        index, actual, tol = _slice_index(times, float(time), tolerance)
        text = text.replace("*", str(index), 1)
        matched = {"requested_time_s": float(time), "time_s": actual, "tolerance_s": tol, "slice": index}
    elif time is not None:
        raise ToolInputError("time selects a slice only through a '*' in path")
    present = path_exists(ods, text)
    value = path_value(ods, text, None) if present else None
    if present and hasattr(value, "keys") and not isinstance(value, dict):
        value = {"children": sorted(str(k) for k in value.keys())}
    elif present and isinstance(value, (list, tuple)):
        value = np.asarray(value) if all(isinstance(v, (int, float)) for v in value) else value
    payload = {"dataset": dataset.provenance, "path": text, "present": present, "matched": matched, "value": value}
    data, converter = bounded_json(payload, budget=points, max_bytes=MAX_EXTRACT_BYTES, max_items=MAX_EXTRACT_ITEMS)
    return _finish(data, converter=converter, converted=True)


def list_equilibrium_times(
    shot: int | None = None,
    artifact: str | None = None,
    database_source: str | None = None,
) -> dict[str, Any]:
    """The equilibrium slice times of a dataset (seconds), in stored order, with their count and range."""
    import numpy as np

    dataset = _dataset(shot, artifact, database_source)
    times = _times(dataset.data, "equilibrium")
    if times is None:
        raise ToolInputError("this dataset holds no equilibrium times")
    finite = times[np.isfinite(times)]
    notes = [f"$.times_s ({times.size} times, kept {MAX_LIMIT})"] if times.size > MAX_LIMIT else []
    payload = {
        "dataset": dataset.provenance,
        "count": int(times.size),
        "time_first_s": float(finite.min()) if finite.size else None,
        "time_last_s": float(finite.max()) if finite.size else None,
        "times_s": [float(t) if np.isfinite(t) else None for t in times[:MAX_LIMIT]],
    }
    return _finish(payload, converter=Bounded(max_items=MAX_LIMIT), notes=notes)


def get_equilibrium_summary(
    shot: int | None = None,
    artifact: str | None = None,
    database_source: str | None = None,
    time: float | None = None,
    tolerance: float | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Global equilibrium quantities of a dataset: Ip, q95, q_axis, q_min, beta_N, beta_p,
    beta_t, li_3, W_mhd, R0, a, elongation, triangularity, volume, ... (units in the names).

    These are VAFT's equilibrium_global summary columns, the same table
    vaft.database.summary builds. With time (s) the one slice nearest that time is
    returned, refused beyond tolerance (default half the slice spacing); without
    it, the first limit slices in time order.
    """
    import numpy as np

    from vaft.database._summary import extract_equilibrium_global

    dataset = _dataset(shot, artifact, database_source)
    rows = extract_equilibrium_global(dataset.data, dataset.shot)
    if not rows:
        raise ToolInputError("this dataset holds no equilibrium time slices")
    times = np.array([np.nan if r.get("time_s") is None else float(r["time_s"]) for r in rows])
    matched = None
    notes: list[str] = []
    if time is not None:
        index, actual, tol = _slice_index(times, float(time), tolerance)
        selected = [rows[index]]
        matched = {"requested_time_s": float(time), "time_s": actual, "tolerance_s": tol}
    else:
        limit = _clamp(limit, MAX_LIMIT, "limit")
        selected = rows[:limit]
        if len(rows) > limit:
            notes.append(f"$.slices ({len(rows)} slices, kept {limit})")
    payload = {"dataset": dataset.provenance, "matched": matched, "count": len(rows), "slices": selected}
    return _finish(payload, converter=Bounded(max_items=MAX_LIMIT), notes=notes)


# ---------------------------------------------------------------------------
# campaign atlas tables (local, opt-in directory; registry in _atlas)
# ---------------------------------------------------------------------------


def _atlas_root() -> Path:
    from ._paths import PathRefused, configured_root

    try:
        return configured_root(ATLAS_ENV, "campaign atlas (the directory with v1/, transport/, stability/, ...)")
    except PathRefused as error:
        raise ToolInputError(str(error)) from None


def _atlas_entry(name: str):
    from . import _atlas

    table = _atlas.BY_NAME.get(str(name).strip())
    if table is None:
        raise ToolInputError(f"unknown atlas table {str(name)[:100]!r}; choose from: {', '.join(_atlas.BY_NAME)}")
    return table


def _atlas_file(root: Path, relative: str | None) -> Path | None:
    """A registry file under the atlas root, or None when absent (or escaping through a symlink)."""
    from ._paths import contained

    if relative is None:
        return None
    return contained(root, root.joinpath(*relative.split("/")))


def _atlas_csv(root: Path, table) -> Path:
    path = _atlas_file(root, table.path)
    if path is None:
        raise ToolInputError(f"atlas table {table.name!r} is not present under {ATLAS_ENV} (expected {table.path})")
    size = path.stat().st_size
    if size > MAX_ATLAS_BYTES:
        raise ToolInputError(f"atlas table {table.name!r} is {size} bytes; the limit is {MAX_ATLAS_BYTES}")
    return path


def _atlas_frame(root: Path, table):
    import pandas as pd

    from . import _atlas

    frame = pd.read_csv(_atlas_csv(root, table), low_memory=False)
    frame.columns = [str(c) for c in frame.columns]
    return _atlas.normalise_frame(frame)


def _atlas_schema(root: Path, table) -> dict | None:
    from . import _atlas

    return _atlas.read_json(_atlas_file(root, table.schema))


def _atlas_provenance(root: Path, table, frame=None) -> dict[str, Any]:
    from . import _atlas

    path = _atlas_csv(root, table)
    provenance = {"lane": table.lane, "issue": table.issue, "file": table.path, "sha256": _atlas.file_sha256(path)}
    provenance.update(_atlas.manifest_provenance(_atlas.read_json(_atlas_file(root, table.manifest))))
    if frame is not None:
        for column in ("atlas_version", "contract_version"):
            if column in frame.columns:
                provenance[column] = sorted({str(v) for v in frame[column].dropna().unique()})[:10]
    return provenance


def _cell(value: Any) -> Any:
    """A table cell as JSON: missing values (NaN, None, NaT) become null."""
    import pandas as pd

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # array-like cells
        return value
    return value.isoformat() if hasattr(value, "isoformat") else value


def _csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def list_atlas_tables() -> dict[str, Any]:
    """The campaign atlas tables this server can read from VAFT_ATLAS_DIR, by lane.

    Each entry gives the table name, owning lane and issue, what one row is (key),
    a summary, the lane's rules, and whether the file is present with its row count.
    Open one with describe_atlas_table(name) before querying it.
    """
    from . import _atlas

    root = _atlas_root()
    rows = []
    for table in _atlas.TABLES:
        path = _atlas_file(root, table.path)
        present = path is not None and path.stat().st_size <= MAX_ATLAS_BYTES
        rows.append({
            "name": table.name,
            "lane": table.lane,
            "issue": table.issue,
            "key": list(table.key),
            "must_filter": list(table.pinned),
            "summary": table.summary,
            "rules": list(table.rules),
            "present": present,
            "rows": _csv_rows(path) if present else None,
        })
    return _finish({"total": len(rows), "items": rows})


def describe_atlas_table(name: str) -> dict[str, Any]:
    """One atlas table: every column with its unit and definition from the lane's schema,
    the row key, the columns a query must pin, the lane's rules and caveats, the
    lane README when there is one, and provenance (file sha256, build commit).

    Read the rules before comparing rows: e.g. stability values must never be
    combined across n_tor, and transport fluxes depend on the TGLF configuration.
    """
    from . import _atlas

    root = _atlas_root()
    table = _atlas_entry(name)
    frame = _atlas_frame(root, table)
    schema = _atlas_schema(root, table) or {}
    described = _atlas.column_descriptions(table, schema)
    patterns = schema.get("column_patterns") or {}
    columns = [{"name": c, "dtype": str(frame[c].dtype), **_atlas.describe_column(c, described, patterns)}
               for c in frame.columns]
    notes = []
    if len(columns) > MAX_ATLAS_COLUMNS:
        notes.append(f"$.columns ({len(columns)} columns, kept {MAX_ATLAS_COLUMNS})")
    readme = _atlas_file(root, table.readme)
    payload = {
        "name": table.name,
        "lane": table.lane,
        "issue": table.issue,
        "summary": table.summary,
        "rows": int(len(frame)),
        "key": list(table.key),
        "must_filter": list(table.pinned),
        "default_columns": [c for c in table.default_columns if c in frame.columns],
        "rules": list(table.rules) + [str(r) for r in schema.get("rules", []) if isinstance(r, str)],
        "caveats": list(table.caveats),
        "description": schema.get("description") if isinstance(schema.get("description"), str) else None,
        "column_count": len(columns),
        "columns": columns[:MAX_ATLAS_COLUMNS],
        "readme": readme.read_text(encoding="utf-8", errors="replace")[:MAX_README] if readme else None,
        "provenance": _atlas_provenance(root, table, frame),
    }
    converter = Bounded(max_items=MAX_ATLAS_COLUMNS, max_string=MAX_README)
    return _finish(payload, converter=converter, notes=notes)


_OPERATORS = ("==", "!=", "<", "<=", ">", ">=", "in", "not_in", "isnull", "notnull")


def _condition(frame, clause: Any):
    from . import _atlas

    if not isinstance(clause, dict) or "column" not in clause:
        raise ToolInputError(f"each where clause is {{column, op, value}}, got {str(clause)[:200]!r}")
    column = _atlas.normalised_column(str(clause["column"]))
    op = str(clause.get("op", "=="))
    if column not in frame.columns:
        raise ToolInputError(f"no column {column!r} in this table; describe_atlas_table lists them")
    if op not in _OPERATORS:
        raise ToolInputError(f"unknown op {op!r}; choose from: {', '.join(_OPERATORS)}")
    series = frame[column]
    if op == "isnull":
        return series.isna()
    if op == "notnull":
        return series.notna()
    value = clause.get("value")
    if op in ("in", "not_in"):
        if not isinstance(value, list) or len(value) > MAX_LIMIT:
            raise ToolInputError(f"op {op!r} takes a list of at most {MAX_LIMIT} values")
        values = [_atlas.normalised_value(column, v) for v in value]
        mask = series.isin(values)
        return ~mask if op == "not_in" else mask
    if isinstance(value, (list, dict)):
        raise ToolInputError(f"op {op!r} takes one scalar value")
    value = _atlas.normalised_value(column, value)
    try:
        if op == "==":
            return series == value
        if op == "!=":
            return series != value
        return {"<": series.lt, "<=": series.le, ">": series.gt, ">=": series.ge}[op](value)
    except TypeError:
        raise ToolInputError(f"cannot compare column {column!r} ({series.dtype}) with {value!r}") from None


def query_atlas_table(
    name: str,
    where: list[dict] | None = None,
    columns: list[str] | None = None,
    order_by: list[str] | None = None,
    count_by: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Rows of one atlas table, filtered by value, with the lane's rules attached.

    where is a list of {column, op, value} clauses, all of which must hold; op is one
    of ==, !=, <, <=, >, >=, in, not_in (value a list), isnull, notnull. Tables with
    must_filter columns (stability: n_tor; stability_surfaces: n_tor and solver;
    transport_sensitivity: tglf_config) refuse a query that does not pin each of
    them with ==, because the lane forbids mixing their values. columns selects
    the returned columns (default: the table's default_columns, key first);
    order_by sorts, '-name' descending; count_by (up to 3 columns) adds row counts
    per value over all matching rows. Old spellings (efit_label, magnetics-only)
    are accepted and returned in State key contract v1 spelling.
    """
    from . import _atlas

    root = _atlas_root()
    table = _atlas_entry(name)
    frame = _atlas_frame(root, table)
    clauses = list(where or [])
    if len(clauses) > 20:
        raise ToolInputError("where takes at most 20 clauses")
    pinned = {_atlas.normalised_column(str(c.get("column"))) for c in clauses
              if isinstance(c, dict) and str(c.get("op", "==")) == "=="}
    missing = [column for column in table.pinned if column not in pinned]
    if missing:
        raise ToolInputError(
            f"atlas table {table.name!r} must be queried with an == filter on {', '.join(missing)}: "
            + " ".join(table.rules[:1])
        )
    mask = None
    for clause in clauses:
        condition = _condition(frame, clause)
        mask = condition if mask is None else (mask & condition)
    matched = frame if mask is None else frame[mask.fillna(False).astype(bool)]

    if order_by:
        keys = [str(k) for k in order_by][:5]
        names = [_atlas.normalised_column(k[1:] if k.startswith("-") else k) for k in keys]
        absent = [n for n in names if n not in frame.columns]
        if absent:
            raise ToolInputError(f"cannot order by unknown columns {absent}")
        matched = matched.sort_values(names, ascending=[not k.startswith("-") for k in keys], na_position="last")

    if columns:
        chosen = [_atlas.normalised_column(str(c)) for c in columns]
        absent = [c for c in chosen if c not in frame.columns]
        if absent:
            raise ToolInputError(f"unknown columns {absent[:20]}; describe_atlas_table lists them")
    else:
        chosen = [c for c in table.default_columns if c in frame.columns] or list(frame.columns)
    chosen = list(dict.fromkeys([k for k in table.key if k in frame.columns] + chosen))
    notes = []
    if len(chosen) > MAX_ATLAS_COLUMNS:
        notes.append(f"$.columns ({len(chosen)} columns, kept {MAX_ATLAS_COLUMNS})")
        chosen = chosen[:MAX_ATLAS_COLUMNS]

    counts: dict[str, Any] = {}
    for column in [_atlas.normalised_column(str(c)) for c in (count_by or [])][:MAX_ATLAS_GROUPS]:
        if column not in frame.columns:
            raise ToolInputError(f"cannot count by unknown column {column!r}")
        tally = matched[column].astype(object).where(matched[column].notna(), None).value_counts(dropna=False)
        counts[column] = [{"value": _cell(value), "rows": int(n)} for value, n in list(tally.items())[:MAX_LIMIT]]

    limit = _clamp(limit, MAX_LIMIT, "limit")
    if len(matched) > limit:
        notes.append(f"$.rows ({len(matched)} rows matched, kept {limit})")
    rows = [{column: _cell(value) for column, value in record.items()}
            for record in matched[chosen].head(limit).to_dict(orient="records")]
    payload = {
        "name": table.name,
        "key": list(table.key),
        "where": clauses,
        "columns": chosen,
        "total_matched": int(len(matched)),
        "rows": rows,
        "counts": counts,
        "rules": list(table.rules),
        "caveats": list(table.caveats),
        "provenance": _atlas_provenance(root, table, frame),
    }
    data, converter = bounded_json(payload, budget=MAX_POINTS, max_bytes=MAX_EXTRACT_BYTES,
                                   max_items=MAX_LIMIT, max_string=MAX_ATLAS_CELL)
    return _finish(data, converter=converter, notes=notes, converted=True)


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
    inspect_dataset,
    inspect_data_path,
    list_equilibrium_times,
    get_equilibrium_summary,
    list_atlas_tables,
    describe_atlas_table,
    query_atlas_table,
)
