"""The read-only VAFT MCP tools as plain Python functions (#1423).

Each function answers one MCP tool call by delegating to an existing public
VAFT API -- ``vaft.help``, the formula and process catalogs, the validation
registry, the plot registry and its declared DD paths, the packaged samples,
the boundary registry and ``vaft.plot.extract`` -- and converting the answer
to bounded JSON.  No scientific computation is added here (an extraction runs
VAFT's own, with size-like options capped), and nothing here
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

import math
import os
import re
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
#: Atlas bounds: file size, listed columns, group columns, cell text, directory scan.
MAX_ATLAS_BYTES = 50 * 1024 * 1024
MAX_ATLAS_COLUMNS = 200
MAX_ATLAS_GROUPS = 3
MAX_ATLAS_CELL = 200
MAX_ATLAS_SCAN = 1000
MAX_ATLAS_DEPTH = 3
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


def _redactions() -> list[tuple[re.Pattern, str]]:
    """(pattern, replacement) pairs: secret values first, then the home directory as ``~``.

    A secret is replaced wherever it appears, so a very short one would also
    rewrite unrelated text; values under four characters are left alone (they
    are not credentials worth the damage).  The home directory is replaced
    only as a whole path component, so ``/Users/yun`` never eats ``/Users/yunho``.
    """
    pairs = [(re.compile(re.escape(value)), "<redacted>")
             for name in SECRET_ENV if len(value := os.environ.get(name, "")) >= 4]
    home = str(Path.home()).rstrip("/\\")
    if len(home) > 1:
        pairs.append((re.compile(re.escape(home) + r"(?![^/\\\s)\]'\",;:])"), "~"))
    return pairs


def _redacted(data: Any, pairs: list | None = None) -> Any:
    """``data`` (already plain JSON) with secrets and the home directory scrubbed from every string.

    Help pages report where a configuration file lives and which variables are
    set; an agent needs the fact, not the account's absolute path or a value.
    """
    pairs = _redactions() if pairs is None else pairs
    if isinstance(data, str):
        for pattern, replacement in pairs:
            data = pattern.sub(replacement, data)
        return data
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            name = new = _redacted(key, pairs)
            suffix = 1
            while new in out:  # two keys that redact to one name both survive
                suffix += 1
                new = f"{name}#{suffix}"
            out[new] = _redacted(value, pairs)
        return out
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
    files = sorted(k for k in options if k.endswith("_path"))
    if files:
        raise ToolInputError(f"options {files} name local files, which this server does not open")
    for name, value in options.items():
        _check_option_size(name, value)
    return dict(options)


#: Ceilings on the options that size a computation, so one call cannot claim the
#: machine (a grid_shape of 20000 x 20000 is a 4e8-row Green's matrix).
OPTION_CEILINGS = {
    "resolution": 512, "max_turns": 1000, "n_frequencies": 512, "nperseg": 1 << 16, "noverlap": 1 << 16,
    "num_modes": 32, "max_modes": 32, "max_harmonics": 64, "window_frames": 4096, "background_frames": 4096,
    "normalisation_frames": 4096, "ncols": 16, "frame_index": 1 << 20, "max_length_m": 1000.0,
}
#: Product of a grid_shape, and length of any list-valued option.
MAX_GRID_CELLS = 256 * 256
MAX_OPTION_ITEMS = 256


def _check_option_size(name: str, value: Any) -> None:
    import numpy as np

    if isinstance(value, (list, tuple)):
        flat = np.asarray(value, dtype=object).ravel() if value else ()
        if len(flat) > MAX_OPTION_ITEMS:
            raise ToolInputError(f"option {name!r} has {len(flat)} values; the limit is {MAX_OPTION_ITEMS}")
        if name == "grid_shape":
            try:
                cells = int(np.prod([int(v) for v in value]))
            except (TypeError, ValueError):
                raise ToolInputError("option 'grid_shape' is a list of integers") from None
            if cells > MAX_GRID_CELLS or min(int(v) for v in value) < 1:
                raise ToolInputError(f"option 'grid_shape' covers {cells} cells; the limit is {MAX_GRID_CELLS}")
        return
    ceiling = OPTION_CEILINGS.get(name)
    if ceiling is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value) or abs(value) > ceiling:
            raise ToolInputError(f"option {name!r} = {value!r} exceeds its ceiling {ceiling}")


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
        raise ToolInputError(f"{ATLAS_ENV} does not name a directory")
    return root.resolve(strict=True)


def _refused(path: Any) -> ToolInputError:
    """The one answer for every refused path, so a refusal reveals nothing about the filesystem."""
    return ToolInputError(
        f"no readable atlas table {str(path)[:200]!r}: pass a relative .csv or .parquet path "
        f"inside {ATLAS_ENV} (omit path to use the only table there)"
    )


def _lexical_parts(path: str) -> tuple[str, ...]:
    """Validate ``path`` without touching the filesystem; its parts when acceptable.

    Absolute, drive-qualified and UNC paths, ``..`` components, NUL bytes and
    other suffixes are refused before any ``stat``, so a hostile path never
    reaches the disk (or, on Windows, an SMB share).
    """
    from pathlib import PurePosixPath, PureWindowsPath

    text = str(path)
    if not text or "\x00" in text or text.startswith(("/", "\\", "~")) or ":" in text:
        raise _refused(path)
    windows = PureWindowsPath(text)
    if windows.is_absolute() or windows.drive or windows.root or PurePosixPath(text).is_absolute():
        raise _refused(path)
    parts = tuple(part for part in text.replace("\\", "/").split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise _refused(path)
    if PurePosixPath(parts[-1]).suffix.lower() not in ATLAS_SUFFIXES:
        raise _refused(path)
    return parts


def _contained(root: Path, candidate: Path) -> Path | None:
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return None
    return resolved if resolved.is_file() else None


def _atlas_tables(root: Path) -> tuple[list[str], bool]:
    """Tables under ``root``, at most :data:`MAX_ATLAS_DEPTH` deep and :data:`MAX_ATLAS_SCAN` entries."""
    tables: list[str] = []
    scanned = 0
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        depth = len(Path(directory).relative_to(root).parts)
        if depth >= MAX_ATLAS_DEPTH:
            subdirectories[:] = []
        subdirectories.sort()
        for name in sorted(files):
            scanned += 1
            if scanned > MAX_ATLAS_SCAN:
                return tables, True
            candidate = Path(directory) / name
            if candidate.suffix.lower() in ATLAS_SUFFIXES and _contained(root, candidate):
                tables.append(candidate.relative_to(root).as_posix())
    return tables, False


def _atlas_table(root: Path, path: str | None) -> Path:
    if path is None:
        tables, incomplete = _atlas_tables(root)
        if len(tables) != 1 or incomplete:
            raise ToolInputError(
                f"{len(tables)}{'+' if incomplete else ''} atlas tables under {ATLAS_ENV}; "
                f"pass path= one of: {tables[:50]}"
            )
        path = tables[0]
    parts = _lexical_parts(path)
    resolved = _contained(root, root.joinpath(*parts))
    if resolved is None or resolved.suffix.lower() not in ATLAS_SUFFIXES:
        raise _refused(path)
    return resolved


def _isna(value: Any) -> bool:
    import pandas as pd

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # array-like cells
        return False


def _cell(value: Any) -> Any:
    if _isna(value):
        return None
    return value.isoformat() if hasattr(value, "isoformat") else value


def _read_atlas(table: Path, groups: list[str], limit: int):
    """Column names and dtypes, row count, the group columns in full, and the first ``limit`` rows."""
    import pandas as pd

    if table.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(table)
        schema = handle.schema_arrow
        columns = [(field.name, str(field.type)) for field in schema]
        rows = handle.metadata.num_rows
        present = [g for g in groups if g in schema.names]
        grouped = handle.read(columns=present).to_pandas() if present else pd.DataFrame()
        batch = next(handle.iter_batches(batch_size=limit), None)
        head = batch.to_pandas() if batch is not None else pd.DataFrame(columns=schema.names)
        return columns, rows, grouped, head.head(limit)
    header = pd.read_csv(table, nrows=0)
    names = [str(c) for c in header.columns]
    present = [g for g in groups if g in names]
    counted = pd.read_csv(table, usecols=present or names[:1]) if names else pd.DataFrame()
    head = pd.read_csv(table, nrows=limit) if names else pd.DataFrame()
    columns = [(str(c), str(t)) for c, t in head.dtypes.items()] or [(n, "object") for n in names]
    return columns, int(len(counted)), counted[present] if present else pd.DataFrame(), head


def get_atlas_summary(
    path: str | None = None,
    group_by: list[str] | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Summarise an operating-space atlas table from the directory named by VAFT_ATLAS_DIR.

    Reads one .csv or .parquet file (at most 50 MB) under that directory; path is
    relative to it, and may be omitted when the directory holds exactly one table.
    Returns its columns, row count, row counts per value of up to three group_by
    columns (default efit_quality and efit_lineage) and its first limit rows.
    Absolute paths, '..' and anything resolving outside the directory are refused.
    """
    root = _atlas_root()
    table = _atlas_table(root, path)
    size = table.stat().st_size
    if size > MAX_ATLAS_BYTES:
        raise ToolInputError(f"atlas table is {size} bytes; the limit is {MAX_ATLAS_BYTES}")
    groups = [str(g) for g in (DEFAULT_ATLAS_GROUPS if group_by is None else group_by)]
    if len(groups) > MAX_ATLAS_GROUPS:
        raise ToolInputError(f"group_by takes at most {MAX_ATLAS_GROUPS} columns, got {len(groups)}")
    limit = _clamp(limit, MAX_LIMIT, "limit")
    try:
        columns, rows, grouped, head = _read_atlas(table, groups, limit)
    except ImportError as error:
        raise ToolInputError(f"reading {table.suffix} tables needs an optional dependency: {error}") from None
    notes = []
    if len(columns) > MAX_ATLAS_COLUMNS:
        notes.append(f"$.columns and $.head ({len(columns)} columns, kept {MAX_ATLAS_COLUMNS})")
    counts: dict[str, Any] = {}
    for column in groups:
        if column not in grouped.columns:
            continue
        tally = grouped[column].map(lambda v: None if _isna(v) else (v if isinstance(v, (str, int, float, bool)) else str(v)))
        tally = tally.value_counts(dropna=False)
        if len(tally) > MAX_LIMIT:
            notes.append(f"$.counts.{column} ({len(tally)} values, kept {MAX_LIMIT})")
        counts[column] = [
            {"value": None if _isna(value) else value, "rows": int(n)}
            for value, n in list(tally.items())[:MAX_LIMIT]
        ]
    payload = {
        "path": table.relative_to(root).as_posix(),
        "rows": int(rows),
        "column_count": len(columns),
        "columns": [{"name": n, "dtype": t} for n, t in columns[:MAX_ATLAS_COLUMNS]],
        "group_by": groups,
        "missing_group_columns": [g for g in groups if g not in counts],
        "counts": counts,
        "head": [{str(k): _cell(v) for k, v in list(record.items())[:MAX_ATLAS_COLUMNS]}
                 for record in head.to_dict(orient="records")],
        "head_truncated": int(rows) > len(head),
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
    get_atlas_summary,
)
