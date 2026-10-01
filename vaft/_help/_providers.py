"""Topic providers for :func:`vaft.help`.

Each provider asks a subsystem's own discovery API for counts, representative
entries and defaults, and returns the dynamic fields of a
:class:`~vaft._help._model.HelpPage`.  Subsystems are imported inside the
function bodies, so importing this module costs nothing and ``vaft.help("x")``
imports only what topic ``x`` needs.

Rules every provider keeps:

- read only: a provider writes no environment variable, rcParams entry,
  backend or file, and installs, launches or contacts nothing.  A topic does
  import the subsystem it describes, and the first import of a subsystem can
  carry its libraries' own import-time effects (Matplotlib building its font
  cache, omas compiling its Cython helpers).  ``database``, ``code`` (probe
  included), ``validation``, ``data`` and ``cli`` avoid those imports entirely;
- no secret: HSDS configuration is reported through
  :func:`vaft.database.hscfg.configured_keys` (key *names*) and never through
  a function that returns values.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from collections import Counter

from ._model import Default, Section

_SCIENTIFIC = (
    Default("coordinates", "explicit / data-derived", "scientific", "never set by help or setup"),
    Default("COCOS", "explicit / metadata-derived", "scientific", "declared by the data or the caller"),
    Default("filtering", "explicit", "scientific", "passed per call"),
    Default("fitting", "explicit", "scientific", "passed per call"),
    Default("slice selection", "explicit", "scientific", "passed per call"),
)


def _first(items, n: int = 5) -> str:
    items = list(items)
    more = f", ... (+{len(items) - n})" if len(items) > n else ""
    return ", ".join(str(i) for i in items[:n]) + more


def _sentence(text: str) -> str:
    head, dot, _rest = str(text).partition(". ")
    return head + ("." if dot else "")


# -- overview ----------------------------------------------------------------
def overview(topic, *, probe: bool = False) -> dict:
    from ..version import __version__
    from ._registry import TOPICS

    rows = tuple((name, t.summary) for name, t in TOPICS.items() if name != "overview")
    return {
        "sections": (
            Section("Version", (("vaft", __version__),)),
            Section("Topics", rows, note="vaft.help('<topic>')  |  vaft help <topic>"),
            Section(
                "Kinds of default",
                (
                    ("runtime", "where and how things run (backend, environment)"),
                    ("data-access", "which source, eager/lazy loading, cache"),
                    ("presentation", "figure format and theme"),
                    ("scientific", "never a convenience default: explicit or data-derived"),
                ),
            ),
        ),
    }


# -- formula / process -------------------------------------------------------
def _catalog_sections(categories, noun: str, module: str) -> tuple:
    rows = tuple((c.name, f"{c.count:>4}  {c.title}".rstrip()) for c in categories if c.count)
    total = sum(c.count for c in categories)
    return (
        Section(
            f"{total} {noun} in {len(rows)} categories",
            rows,
            note=f"search: {module}.search('text'); list: {module}.list_{noun}(category)",
        ),
    )


def formula(topic, *, probe: bool = False) -> dict:
    from vaft.formula import catalog

    return {"sections": _catalog_sections(catalog.categories(), "formulas", "vaft.formula.catalog")}


def formula_item(name: str):
    from vaft.formula import catalog

    return catalog.show(name)


def process(topic, *, probe: bool = False) -> dict:
    from vaft.process import catalog

    return {"sections": _catalog_sections(catalog.categories(), "processes", "vaft.process.catalog")}


def process_item(name: str):
    from vaft.process import catalog

    return catalog.describe(name)


# -- validation --------------------------------------------------------------
def validation(topic, *, probe: bool = False) -> dict:
    from vaft.validation import registry

    by_category = Counter(spec.category for spec in registry.CHECKS.values())
    rows = tuple(
        (category, f"{count:>3}  {_first(s.key for s in registry.checks_in(category))}")
        for category, count in by_category.items()
    )
    return {"sections": (Section(f"{len(registry.CHECKS)} checks", rows),)}


def validation_item(key: str):
    from vaft.validation import registry

    return registry.describe(key)


# -- plot --------------------------------------------------------------------
def _frontend_kind() -> str:
    """Terminal / IPython / Jupyter / VS Code, without importing IPython."""
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None and hasattr(ipython, "get_ipython") else None
    name = type(shell).__name__ if shell is not None else ""
    if name == "ZMQInteractiveShell":
        return "vscode" if any(k.startswith("VSCODE_") for k in os.environ) else "jupyter"
    return "ipython" if name else "terminal"


def _plot_runtime() -> tuple[list[Default], list[str]]:
    """Report the Matplotlib backend without making Matplotlib choose one.

    ``matplotlib.get_backend()`` resolves the automatic backend -- a change of
    rcParams -- so the environment is read through
    :func:`vaft.plot.environment.detect_environment` only once a backend is
    already in place.
    """
    import matplotlib

    defaults: list[Default] = []
    resolved = matplotlib.rcParams._get_backend_or_none()
    ipympl = importlib.util.find_spec("ipympl") is not None
    if resolved is not None:
        from vaft.plot.environment import (
            default_interaction_backend,
            detect_environment,
        )

        env = detect_environment()
        defaults += [
            Default("environment", env.kind, "runtime"),
            Default("backend", env.backend, "runtime", "already selected in this process"),
            Default("live figures", "yes" if env.live_figures else "no", "runtime"),
            Default(
                "interaction (auto)",
                default_interaction_backend(env),
                "runtime",
                "live canvas -> matplotlib; kernel with ipywidgets -> ipywidgets; else none",
            ),
        ]
    else:
        defaults += [
            Default("environment", _frontend_kind(), "runtime"),
            Default("backend", "not chosen yet", "runtime", "Matplotlib picks one at the first figure"),
        ]
    defaults.append(Default("ipympl", "installed" if ipympl else "not installed", "runtime"))
    env_backend = os.environ.get("MPLBACKEND")
    if env_backend:
        defaults.append(Default("MPLBACKEND", env_backend, "runtime", "set in the environment; respected"))
    return defaults, []


def plot(topic, *, probe: bool = False) -> dict:
    from vaft.plot import presentation
    from vaft.plot.registry import VIEWS, available_plots

    runtime, warnings = _plot_runtime()
    plots = available_plots()
    by_view = Counter(getattr(p, "view", "") for p in plots)
    rows = tuple((view, str(by_view[view])) for view in VIEWS if by_view.get(view))
    defaults = runtime + [
        Default("format", str(presentation.DEFAULT_FORMAT), "presentation", "format=None on a plot call"),
        Default("theme", "none", "presentation", "theme=None keeps Matplotlib's own look"),
        Default("interaction", "auto", "presentation", "backend='auto' follows the environment"),
    ]
    return {
        "defaults": tuple(defaults),
        "sections": (
            Section(
                f"{len(plots)} canonical plots by view",
                rows,
                note="search: vaft.plot.available_plots(query='...'); tree: print(vaft.plot.available_plots())",
            ),
        ),
        "optional": (("ipympl", "live notebook figures (pan/zoom, sliders on the canvas)"),),
        "warnings": tuple(warnings),
    }


def plot_item(query: str):
    from vaft.plot.registry import available_plots

    return available_plots(query=query)


# -- database ----------------------------------------------------------------
#: Non-secret HSDS settings help may report as set / not set.
_PUBLIC_HS_KEYS = (("endpoint", "hs_endpoint", "HS_ENDPOINT"), ("username", "hs_username", "HS_USERNAME"))


def _load_defaults() -> dict[str, object]:
    """Keyword defaults of :func:`vaft.database.load`, read from its source.

    Resolving ``vaft.database.load`` imports omas, IMAS and Matplotlib; the
    signature is all help needs, so it is read with :mod:`ast` instead.
    """
    try:
        origin = importlib.util.find_spec("vaft.database").origin
        with open(origin, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=origin)
    except (OSError, SyntaxError, TypeError, ValueError):  # e.g. no source shipped
        return {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load":
            return {
                arg.arg: ast.literal_eval(default)
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if isinstance(default, ast.Constant)
            }
    return {}


def database(topic, *, probe: bool = False) -> dict:
    # Imported by full name, never ``from vaft.database import hscfg``: that
    # form first asks the package's ``__getattr__``, whose fallback imports the
    # h5pyd/omas I/O modules (``hscfg`` is not in ``vaft.database.__all__``).
    hscfg = importlib.import_module("vaft.database.hscfg")
    sources = importlib.import_module("vaft.database.sources")

    load = _load_defaults()
    defaults = (
        Default("source", sources.DEFAULT_SOURCE, "data-access", "source=None in load/open/save"),
        Default("loading", "load: eager, open: lazy", "data-access", "pick the function, not a flag"),
        Default("representation", str(load.get("representation", "see vaft.database.load")), "data-access"),
        Default("cache", str(load.get("cache", "see vaft.database.load")), "data-access"),
        Default("transport", str(load.get("transport", "see vaft.database.load")), "data-access"),
    ) + _SCIENTIFIC[:2]

    path = hscfg.active_path()
    try:
        configured = set(hscfg.configured_keys(path)) if path.is_file() else set()
    except OSError:
        configured = set()
    rows = []
    for label, key, variable in _PUBLIC_HS_KEYS:
        where = []
        if key in configured:
            where.append(str(path))
        if os.environ.get(variable):
            where.append(f"${variable}")
        rows.append((label, f"set ({', '.join(where)})" if where else "not set"))
    ready = all(r[1] != "not set" for r in rows)
    known = sources.known_sources()
    return {
        "defaults": defaults,
        "sections": (
            Section(
                "HSDS configuration",
                tuple(rows),
                note=("configured; nothing was contacted" if ready else "run `vaft hsds configure` to set it up"),
            ),
            Section(
                f"{len(known)} sources",
                tuple((s.name, _sentence(s.purpose)) for s in known[:8]),
                note="all: vaft.database.sources.known_sources()",
            ),
        ),
    }


def database_item(name: str):
    from vaft.database import sources

    return sources.describe(name)


# -- code --------------------------------------------------------------------
#: (display name, installation-root variable, executable beneath that root).
#: The layouts are the adapters' own ``*_HOME_EXECUTABLE`` constants (pinned by
#: test/test_help.py); they are restated here because importing an adapter
#: module pulls in omas and the database layer.
CODES = (
    ("EFIT", "EFITHOME", "bin/efit"),
    ("CHEASE", "CHEASEHOME", "bin/chease"),
    ("GPEC", "GPECHOME", "bin/dcon"),
    ("GACODE", "GACODEHOME", "neo/bin/neo"),
    ("NUBEAM", "NUBEAMHOME", "bin/nubeam_comp_exec"),
    ("GENRAY", "GENRAYHOME", "bin/xgenray"),
    ("FLARE", "FLAREHOME", "bin/flare"),
    ("TES", "TESHOME", "bin/rtes"),
)


def _probe(name: str, variable: str, relative: str) -> str:
    """Look for one executable under its ``$HOME``; run nothing."""
    from vaft.code._executables import executable_from_home

    try:
        found = executable_from_home(
            os.environ.get(variable), home_variable=variable, relative_path=relative, code_name=name
        )
    except (FileNotFoundError, PermissionError) as error:
        return f"broken install ({type(error).__name__})"
    return "available" if found else "unavailable"


def code(topic, *, probe: bool = False) -> dict:
    rows = []
    for name, variable, relative in CODES:
        home = f"${variable} {'set' if os.environ.get(variable) else 'unset'}"
        rows.append((name, f"{_probe(name, variable, relative)}  ({home})" if probe else home))
    note = (
        "looked for each executable under its $HOME layout; nothing was run "
        "(adapters may also accept an explicit executable or a legacy variable)"
        if probe
        else "vaft.help('code', probe=True) or `vaft help code --probe` looks for the executables"
    )
    return {
        "sections": (Section("External codes", tuple(rows), note=note),),
        "setup": ("install/README.md (installers never run implicitly)",),
    }


# -- data --------------------------------------------------------------------
def data(topic, *, probe: bool = False) -> dict:
    from vaft.data import resources

    rows = []
    for shot in resources.available_samples():
        manifest = resources.sample_manifest(shot)
        parts = []
        for name, record in manifest["representations"].items():
            present = resources.data_path(f"samples/{shot}/{record['path']}").is_file()
            parts.append(f"{name} ({record.get('package', '?')}{'' if present else ', not in this install'})")
        rows.append((str(shot), ", ".join(parts)))
    return {
        "sections": (
            Section(
                f"{len(rows)} sample shots",
                tuple(rows),
                note="path: vaft.data.resources.sample(shot, representation='omas')",
            ),
        ),
    }


def data_item(shot: str):
    from vaft.data import resources

    try:
        number = int(shot)
    except ValueError:
        raise ValueError(f"sample shot must be an integer, got {shot!r}") from None
    return resources.sample_manifest(number)


# -- omas / imas -------------------------------------------------------------
def _entries(describe, what: str) -> dict:
    entries = describe()
    return {
        "sections": (
            Section(
                f"{len(entries)} plot entries on {what}",
                tuple((e.name, getattr(e, "view", "")) for e in list(entries)[:8]),
                note="all: print(describe()); filter: describe(query='...')",
            ),
        )
    }


def omas(topic, *, probe: bool = False) -> dict:
    from vaft.omas.discovery import describe

    return _entries(describe, "ODS")


def omas_item(query: str):
    from vaft.omas.discovery import describe

    return describe(query=query)


def imas(topic, *, probe: bool = False) -> dict:
    from vaft.imas.discovery import describe

    return _entries(describe, "IDS")


def imas_item(query: str):
    from vaft.imas.discovery import describe

    return describe(query=query)


# -- cli ---------------------------------------------------------------------
def cli(topic, *, probe: bool = False) -> dict:
    from vaft.cli._main import _COMMANDS

    rows = tuple((f"vaft {name}", description) for name, (_module, description) in _COMMANDS.items())
    return {"sections": (Section("Commands", rows, note="syntax: vaft <command> --help"),)}


def cli_item(command: str) -> str:
    from vaft.cli._main import _COMMANDS

    if command not in _COMMANDS:
        raise KeyError(f"unknown command {command!r}; choose from: {', '.join(_COMMANDS)}")
    return f"vaft {command}: {_COMMANDS[command][1]}\nsyntax and arguments: vaft {command} --help"
