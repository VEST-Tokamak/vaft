"""Topic providers for :func:`vaft.help`.

Each provider asks a subsystem's own discovery API for counts, representative
entries and defaults, and returns the dynamic fields of a
:class:`~vaft._help._model.HelpPage`.  Subsystems are imported inside the
function bodies, so importing this module costs nothing and ``vaft.help("x")``
imports only what topic ``x`` needs.

Rules every provider keeps:

- read only: no environment variable, rcParams entry, backend or file is
  written; nothing is installed, launched or contacted over the network;
- no secret: HSDS configuration is reported through
  :func:`vaft.database.hscfg.configured_keys` (key *names*) and never through
  a function that returns values.
"""

from __future__ import annotations

import importlib.util
import inspect
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


def database(topic, *, probe: bool = False) -> dict:
    import vaft.database as db
    from vaft.database import hscfg, sources

    load = inspect.signature(db.load).parameters
    defaults = (
        Default("source", sources.DEFAULT_SOURCE, "data-access", "source=None in load/open/save"),
        Default("loading", "load: eager, open: lazy", "data-access", "pick the function, not a flag"),
        Default("representation", str(load["representation"].default), "data-access"),
        Default("cache", str(load["cache"].default), "data-access"),
        Default("transport", str(load["transport"].default), "data-access"),
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
def _gpec():
    from vaft.code.gpec import find_gpec_executable

    return find_gpec_executable("dcon")


def _efit():
    from vaft.code.efit.magnetic import find_efit_executable

    return find_efit_executable()


def _chease():
    from vaft.code.chease import find_chease_executable

    return find_chease_executable()


def _gacode():
    from vaft.code.gacode._runtime import find_gacode_executable

    return find_gacode_executable(code="neo")


def _nubeam():
    from vaft.code.nubeam.runner import find_nubeam_executable

    return find_nubeam_executable()


def _genray():
    from vaft.code.genray.runner import find_genray_executable

    return find_genray_executable()


def _flare():
    from vaft.code.flare import flare_executable

    return flare_executable()


#: (display name, installation-root variable, finder or None).
CODES = (
    ("EFIT", "EFITHOME", _efit),
    ("CHEASE", "CHEASEHOME", _chease),
    ("GPEC", "GPECHOME", _gpec),
    ("GACODE", "GACODEHOME", _gacode),
    ("NUBEAM", "NUBEAMHOME", _nubeam),
    ("GENRAY", "GENRAYHOME", _genray),
    ("FLARE", "FLAREHOME", _flare),
    ("TES", "TESHOME", None),
)


def _probe(finder, home_set: bool) -> str:
    """Ask an adapter's own finder; never launch anything.

    Adapters differ in how they say "not configured" -- some return ``None``,
    some raise -- so an error only means a broken install when the code's
    installation root is actually set.
    """
    try:
        found = finder()
    except Exception as error:  # noqa: BLE001 - FileNotFoundError, PermissionError, or an import
        return f"broken install ({type(error).__name__})" if home_set else "unavailable"
    return "available" if found else "unavailable"


def code(topic, *, probe: bool = False) -> dict:
    rows = []
    for name, variable, finder in CODES:
        home_set = bool(os.environ.get(variable))
        home = f"${variable} {'set' if home_set else 'unset'}"
        if probe and finder is not None:
            rows.append((name, f"{_probe(finder, home_set)}  ({home})"))
        else:
            rows.append((name, home))
    note = (
        "probed each adapter's executable finder (no process was launched)"
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

    return resources.sample_manifest(int(shot))


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
