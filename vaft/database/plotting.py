"""Plot adapters for database shots: ``plot_<canonical-stem>(shot, source=...)``.

Every ``plot_<stem>`` has two twins (umbrella #434): ``dd_<stem>()`` lists the
Data Dictionary paths the plot reads without touching the database, and
``extract_<stem>(shot, source=..., *, lazy=True, occurrence=None, label="shot",
**extraction_options)`` opens the same IDS the plot would and returns the view
model undrawn -- the one ``vaft.omas.extract_<stem>`` builds from a loaded
ODS, so ``.to_xarray()`` works on it.  A rendering keyword is refused by
``extract_*``.

A database adapter names *which shot in which source* to draw; it does not
interpret data.  Each adapter asks the registry which IDS the plot needs
(:func:`vaft.plot.backend.recipes.required_ids`), opens those the shot stores
over HSDS -- lazily by default, so only the leaves the plot touches travel;
an optional IDS the shot lacks (an overlay's wall) is skipped, a required
one is refused by name -- and hands the resulting ODS to the OMAS adapter (issue #63).  Nothing here
imports Matplotlib, and nothing here reaches into ``vaft.machine_mapping``.

Every adapter shares one signature::

    plot_<stem>(shot, source=None, *, lazy=True, occurrence=None,
                ax=None, show=False, label="shot", **options)

``shot`` is an int or a list of ints (one entry each, in that order);
``source`` is resolved by :func:`vaft.database.sources.resolve` and defaults
to ``main``.  ``lazy=True`` uses :func:`vaft.database.open`; ``lazy=False``
stages the declared IDS with :func:`vaft.database.load` and is the only way
to ask for an ``occurrence`` -- every source stores one occurrence per IDS,
so the lazy path reads occurrence 0 by construction.  ``label="shot"`` names
each entry by the shot the caller asked for, which needs no remote read.
``interactive=True`` loads the declared IDS eagerly whatever ``lazy`` says:
the controls redraw after the call returns, when a lazy store would already
be closed.

:func:`available_plots` answers for a shot without downloading it: the
shot's IDS domains are listed and a plot is available when the IDS it needs
are present.  Leaf-level facts (channels, flagged, synthetic) need a loaded
ODS -- pass one to :func:`vaft.omas.available_plots`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

__all__ = [
    "available_plots",
    "compose",
    "dd",
    "extract",
    "plot_diagnostics_time_interactive",
    "plot_equilibrium_interactive",
    "render",
    "render_to_file",
    "stored_ids",
]


def _resolve_source(source: str | None) -> str:
    from .sources import resolve

    return resolve(source)


def _declared_ids(name: str) -> list[str]:
    from vaft.plot.backend.recipes import required_ids

    roots = ["dataset_description", *required_ids(name)]
    return list(dict.fromkeys(roots))


def _shots(shot: Any) -> list[int]:
    if isinstance(shot, (list, tuple)):
        return [int(item) for item in shot]
    return [int(shot)]


def _labels(shots: Sequence[int], label: Any) -> Any:
    """The labels handed to the OMAS adapter.

    The shot is what the caller named and must not depend on a remote read,
    so ``"shot"``/``"pulse"`` are the shot numbers; ``"key"`` is the position;
    an explicit sequence and ``"run"`` are forwarded as they are.
    """
    if isinstance(label, (list, tuple)) or label == "run":
        return label
    if label == "key":
        return [str(i) for i in range(len(shots))]
    return [str(shot) for shot in shots]


def _stored_subset(
    shot: int, source: str, ids: Sequence[str], name: str
) -> list[str]:
    """``ids`` narrowed to what ``shot`` stores; refuses when plot ``name`` cannot be drawn.

    A plot declares every IDS it may read, the optional ones too -- the
    wall an equilibrium map overlays -- and asking a store for a domain the
    shot lacks fails: the eager path with ``FileNotFoundError``, the lazy
    one with a 404 on first read.  So the shot's domains are listed (no data
    read) and only the declared IDS it stores are opened.  An IDS the plot
    needs stays an error, judged by the rule :func:`available_plots` uses.
    """
    from vaft.plot.backend.discovery import missing_required_ids

    stored = set(stored_ids(shot, source))
    missing = missing_required_ids(stored, name)
    if missing is not None:
        raise FileNotFoundError(
            f"plot {name!r} needs IDS {missing}, which shot {shot} does not store in "
            f"{source!r}. Stored IDS: {', '.join(sorted(stored)) or '(none)'}."
        )
    return [root for root in ids if root in stored]


def _open_all(
    stack: contextlib.ExitStack, shots: Sequence[int], source: str, ids: Sequence[str],
    *, lazy: bool, occurrence: Any, name: str,
) -> list[Any]:
    from . import load, open

    objects = []
    for shot in shots:
        present = _stored_subset(shot, source, ids, name)
        if lazy:
            ods = open(shot, source=source, paths=present)
            if hasattr(ods, "close"):
                stack.callback(ods.close)
        else:
            ods = load(shot, source=source, paths=present, occurrence=occurrence)
        objects.append(ods)
    return objects


def _asks_for_another_occurrence(occurrence: Any) -> bool:
    """Whether ``occurrence`` names anything but occurrence 0."""
    if occurrence is None:
        return False
    if isinstance(occurrence, dict):
        return any(int(value) != 0 for value in occurrence.values())
    return int(occurrence) != 0


def render(
    name: str,
    shot: Any,
    source: str | None = None,
    *,
    lazy: bool = True,
    occurrence: Any = None,
    ax: Any = None,
    show: bool = False,
    label: Any = "shot",
    **options: Any,
) -> tuple[Any, Any]:
    """Open what plot ``name`` needs of ``shot`` in ``source`` and render it.

    ``backend="plotly"`` among the options returns a Plotly figure instead of
    ``(Figure, Axes)`` (see :mod:`vaft.plot.backends`).
    """
    if options.get("interactive") and lazy:
        # The controls rebuild the model on every widget event, long after
        # this call returns and its lazy store has closed: load once instead,
        # with the eager path's whole contract (an occurrence is honoured).
        lazy = False
    _refuse_lazy_occurrence(lazy, occurrence)
    resolved = _resolve_source(source)
    shots = _shots(shot)
    ids = _declared_ids(name)
    from vaft.omas.plotting import render as render_ods

    with contextlib.ExitStack() as stack:
        objects = _open_all(stack, shots, resolved, ids, lazy=lazy, occurrence=occurrence, name=name)
        source_object = objects[0] if len(objects) == 1 else objects
        # Models copy what they read into their own arrays and renderers never
        # keep the data object, so the lazy stores may close on the way out.
        return render_ods(name, source_object, ax=ax, show=show, label=_labels(shots, label), **options)


def compose(
    composition: Any,
    shot: Any,
    source: str | None = None,
    *,
    occurrence: Any = None,
    backend: str | None = None,
    show: bool = False,
    label: Any = "shot",
    **presentation: Any,
) -> Any:
    """Draw a :class:`vaft.plot.FigureComposition` for database shots (issue #1467).

    For each shot the IDS every cell declares are narrowed to what the shot
    stores (an overlay's optional wall is skipped, a missing required IDS is
    an error naming the plot) and loaded once; the figure is then composed
    as :func:`vaft.omas.compose` composes it.  Not :func:`vaft.database.compose`,
    which joins two *sources* of one shot (#305).
    """
    from vaft.omas.entries import normalize_entries
    from vaft.plot.composition import as_composition, render_composition

    from . import load

    composition = as_composition(composition)
    resolved = _resolve_source(source)
    shots = _shots(shot)
    objects = []
    for number in shots:
        wanted: list[str] = []
        for cell in composition.cells:
            for root in _stored_subset(number, resolved, _declared_ids(cell.plot), cell.plot):
                if root not in wanted:
                    wanted.append(root)
        objects.append(load(number, source=resolved, paths=wanted, occurrence=occurrence))
    source_object = objects[0] if len(objects) == 1 else objects
    entries = normalize_entries(source_object, label=_labels(shots, label))
    return render_composition(
        composition, entries, backend=backend, show=show,
        namespace="vaft.database", subject="vaft.database.load(shot)", **presentation,
    )


def _refuse_lazy_occurrence(lazy: bool, occurrence: Any) -> None:
    if lazy and _asks_for_another_occurrence(occurrence):
        raise ValueError(
            "occurrence is available with lazy=False only: every source stores one "
            "occurrence per IDS, so the lazy path reads occurrence 0 by construction"
        )


def extract(
    name: str,
    shot: Any,
    source: str | None = None,
    *,
    lazy: bool = True,
    occurrence: Any = None,
    label: Any = "shot",
    **options: Any,
) -> Any:
    """Open what plot ``name`` needs of ``shot`` and return its view model, undrawn.

    The same IDS :func:`render` opens, lazily by default; the model is what
    ``vaft.omas.extract_<name>`` builds from the loaded ODS, so
    ``.to_xarray()`` works on it.  A computed view whose builder needs an
    OMAS ODS (``CallableRecipe.backend == "omas"``) is served from the lazy
    store by reading exactly the paths its recipe declares
    (:func:`vaft.plot.backend.recipes.materialise_reads`); one that
    deep-copies a whole IDS fetches that IDS whole.  Only the extraction options are taken
    (:data:`vaft.plot.backend.options.EXTRACTION_OPTIONS`); a rendering
    keyword such as ``ax=`` is refused by name.
    """
    from vaft.omas.entries import normalize_entries
    from vaft.plot.backend.facade import refuse_render_options
    from vaft.plot.backend.recipes import build_model
    from vaft.plot.backend.render import refuse_when_unsupported

    _refuse_lazy_occurrence(lazy, occurrence)
    refuse_render_options(name, options)
    resolved = _resolve_source(source)
    shots = _shots(shot)
    ids = _declared_ids(name)
    with contextlib.ExitStack() as stack:
        objects = _open_all(stack, shots, resolved, ids, lazy=lazy, occurrence=occurrence, name=name)
        source_object = objects[0] if len(objects) == 1 else objects
        entries = normalize_entries(source_object, label=_labels(shots, label))
        # The pointer names a discovery that sees the leaves: available_plots(shot)
        # judges by IDS domains only, a loaded ODS by what it holds.
        refuse_when_unsupported(
            name, entries, namespace="vaft.database", subject="vaft.database.load(shot)",
        )
        return build_model(name, entries, **options)


def dd(name: str) -> tuple:
    """The Data Dictionary paths plot ``name`` reads; nothing is opened."""
    from vaft.plot.backend.dd import dd_paths

    return dd_paths(name)


def _load_for_interaction(
    shot: Any, source: str | None, ids: Sequence[str], name: str, *, occurrence: Any = None
) -> Any:
    """``ids`` of one shot, loaded into memory for an explorer.

    An explorer rebuilds its figure on every widget event, long after this
    call returns, so the lazy store the static path opens -- and closes on
    the way out -- would be gone by then.  The IDS the entry point reads are
    loaded once instead; every later frame is served from memory.  Plot
    ``name`` decides which of them the shot must store (:func:`_stored_subset`).
    """
    from . import load

    shots = _shots(shot)
    if len(shots) != 1:
        raise ValueError(f"an interactive entry point explores one shot at a time; got {len(shots)}")
    resolved = _resolve_source(source)
    present = _stored_subset(shots[0], resolved, ids, name)
    return load(shots[0], source=resolved, paths=present, occurrence=occurrence)


def plot_diagnostics_time_interactive(
    shot: Any, source: str | None = None, *, occurrence: Any = None, **options: Any
) -> Any:
    """The diagnostics overview with live controls, from the database (issue #482).

    See :func:`vaft.omas.plot_diagnostics_time_interactive`.  The overview's
    IDS are loaded once, not opened lazily: the controls redraw after this
    call returns.
    """
    from vaft.omas.interactive import plot_diagnostics_time_interactive as explore

    ods = _load_for_interaction(
        shot, source, _declared_ids("diagnostics_overview"), "diagnostics_overview", occurrence=occurrence
    )
    return explore(ods, **options)


def plot_equilibrium_interactive(
    shot: Any, source: str | None = None, *, occurrence: Any = None, **options: Any
) -> Any:
    """Explore one shot's equilibrium slices, from the database.

    See :func:`vaft.omas.plot_equilibrium_interactive`; loaded once for the
    same reason as :func:`plot_diagnostics_time_interactive`, and exactly
    the IDS the explorer reads (the slice summary's and its histories'), so
    the measured plasma current and the slice markers are there.
    """
    from vaft.omas.interactive import equilibrium_explorer_ids
    from vaft.omas.interactive import plot_equilibrium_interactive as explore

    ods = _load_for_interaction(
        shot, source, equilibrium_explorer_ids(), "equilibrium_overview", occurrence=occurrence
    )
    return explore(ods, **options)


def render_to_file(
    name: str,
    shot: Any,
    path: Any,
    source: str | None = None,
    *,
    lazy: bool = True,
    occurrence: Any = None,
    label: Any = "shot",
    **options: Any,
) -> Any:
    """Render plot ``name`` for ``shot`` and write it to ``path``; returns ``path``.

    Draws without a display (:func:`vaft.plot.environment.
    use_non_interactive_backend`) and saves with :func:`vaft.plot.save_figure`,
    the format following the file extension.  This is what ``vaft plot --out``
    runs.
    """
    from vaft.plot import save_figure
    from vaft.plot.environment import use_non_interactive_backend

    if options.get("backend") == "plotly":
        # A Plotly figure is a web page; nothing else is a faithful file of it.
        if not str(path).lower().endswith((".html", ".htm")):
            raise ValueError(f"backend='plotly' writes HTML; give --out a .html path, not {path!r}")
        figure = render(name, shot, source, lazy=lazy, occurrence=occurrence, show=False, label=label, **options)
        figure.write_html(str(path), include_plotlyjs="cdn")
        return path
    use_non_interactive_backend()
    figure, _ = render(
        name, shot, source, lazy=lazy, occurrence=occurrence, show=False, label=label, **options
    )
    return save_figure(figure, path, figure_options=options.get("figure_options"))


def available_plots(
    shot: Any = None,
    source: str | None = None,
    *,
    query: str | None = None,
    detail: bool = False,
    available_only: bool | None = None,
    **filters: Any,
):
    """What can be plotted from a database shot, without downloading it.

    Without ``shot``, the registry plus what the recipes declare.  With a
    shot number, the shot's IDS domains in ``source`` are listed and a plot is
    available when every IDS it needs is present; the reasons name the missing
    IDS.  An already loaded object (an ODS or lazy ODS) is passed straight to
    :func:`vaft.omas.available_plots`, which also reports leaf-level facts.
    """
    from vaft.plot.backend.discovery import describe_by_ids, describe_entries

    if shot is None:
        return describe_entries(None, query=query, detail=detail, **filters)
    if not isinstance(shot, (int, str)) or isinstance(shot, bool):
        from vaft.omas.plotting import available_plots as available_ods_plots

        return available_ods_plots(
            shot, query=query, detail=detail, available_only=available_only, **filters
        )
    resolved = _resolve_source(source)
    present = stored_ids(shot, resolved)
    return describe_by_ids(
        present, source=f"#{shot} ({resolved})", query=query, detail=detail,
        available_only=available_only, **filters,
    )


def stored_ids(shot: Any, source: str | None = None) -> tuple[str, ...]:
    """The IDS ``shot`` stores in ``source``, listed without reading any data.

    What :func:`available_plots` judges a shot by.  A plot's declared IDS
    include optional ones (an overlay's wall); a caller that fetches IDS for
    a plot asks for the declared IDS the shot actually stores.
    """
    from .lazy_common import discover_hsds_ids
    from . import utils

    return tuple(discover_hsds_ids(_h5pyd(utils), _resolve_source(source), int(shot)))


def _h5pyd(utils_module: Any) -> Any:
    """The h5pyd module the lazy store uses (patchable in tests)."""
    from . import lazy_ods

    module = getattr(lazy_ods, "h5pyd", None)
    if module is None:
        import h5pyd

        module = h5pyd
    return module


def _adapter(name: str, description: str):
    def adapter(
        shot: Any,
        source: str | None = None,
        *,
        lazy: bool = True,
        occurrence: Any = None,
        ax: Any = None,
        show: bool = False,
        label: Any = "shot",
        **options: Any,
    ) -> tuple[Any, Any]:
        return render(
            name, shot, source, lazy=lazy, occurrence=occurrence, ax=ax, show=show,
            label=label, **options,
        )

    adapter.__name__ = adapter.__qualname__ = f"plot_{name}"
    adapter.__doc__ = (
        f"{description.rstrip('.')}.\n\n"
        f"Opens the IDS ``{name}`` declares for the shot in ``source`` and renders "
        f"with :func:`vaft.plot.{name}` through :func:`vaft.omas.plot_{name}`."
    )
    return adapter


def _extract_adapter(name: str, description: str):
    def adapter(
        shot: Any,
        source: str | None = None,
        *,
        lazy: bool = True,
        occurrence: Any = None,
        label: Any = "shot",
        **options: Any,
    ) -> Any:
        return extract(name, shot, source, lazy=lazy, occurrence=occurrence, label=label, **options)

    adapter.__name__ = adapter.__qualname__ = f"extract_{name}"
    adapter.__doc__ = (
        f"The data behind :func:`plot_{name}` for a database shot, as its view model, undrawn.\n\n"
        f"{description.rstrip('.')}.  Opens the IDS ``{name}`` declares for the shot in "
        f"``source`` (lazily by default) and builds the model :func:`vaft.omas.extract_{name}` "
        f"builds; a rendering keyword is refused.  ``.to_xarray()`` on the result gives an "
        f":class:`xarray.Dataset`; :func:`dd_{name}` lists the Data Dictionary paths it reads."
    )
    return adapter


def _dd_adapter(name: str, description: str):
    def adapter() -> tuple:
        return dd(name)

    adapter.__name__ = adapter.__qualname__ = f"dd_{name}"
    adapter.__doc__ = (
        f"The Data Dictionary paths :func:`plot_{name}` reads, without touching the database.\n\n"
        f"{description.rstrip('.')}.  The same tuple of :class:`vaft.plot.backend.dd.DDPath` "
        f"as :func:`vaft.omas.dd_{name}`."
    )
    return adapter


_FACTORIES = {"plot_": _adapter, "extract_": _extract_adapter, "dd_": _dd_adapter}


def _canonical(name: str):
    """The registered spec behind ``plot_<name>``, or ``None``.

    Looked up on demand: building every adapter at import would need the
    registry, which brings the whole plotting stack (and Matplotlib) with it,
    and this namespace must import without either.
    """
    from vaft.plot.registry import get_spec

    try:
        return get_spec(name)
    except KeyError:
        return None


def __getattr__(name: str):
    for prefix, factory in _FACTORIES.items():
        if name.startswith(prefix):
            spec = _canonical(name[len(prefix):])
            if spec is not None:
                adapter = factory(spec.name, spec.description)
                globals()[name] = adapter
                return adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    from vaft.plot.registry import canonical_names

    generated = {f"{prefix}{n}" for prefix in _FACTORIES for n in canonical_names()}
    return sorted(set(globals()) | set(__all__) | generated)
