"""Plot adapters for database shots: ``plot_<canonical-stem>(shot, source=...)``.

A database adapter names *which shot in which source* to draw; it does not
interpret data.  Each adapter asks the registry which IDS the plot needs
(:func:`vaft.plot.backend.recipes.required_ids`), opens exactly those over
HSDS -- lazily by default, so only the leaves the plot touches travel -- and
hands the resulting ODS to the OMAS adapter (issue #63).  Nothing here
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

:func:`available_plots` answers for a shot without downloading it: the
shot's IDS domains are listed and a plot is available when the IDS it needs
are present.  Leaf-level facts (channels, flagged, synthetic) need a loaded
ODS -- pass one to :func:`vaft.omas.available_plots`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

__all__ = ["available_plots", "render", "render_to_file"]


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


def _open_all(
    stack: contextlib.ExitStack, shots: Sequence[int], source: str, ids: Sequence[str],
    *, lazy: bool, occurrence: Any,
) -> list[Any]:
    from . import load, open

    objects = []
    for shot in shots:
        if lazy:
            ods = open(shot, source=source, paths=list(ids))
            if hasattr(ods, "close"):
                stack.callback(ods.close)
        else:
            ods = load(shot, source=source, paths=list(ids), occurrence=occurrence)
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
    if lazy and _asks_for_another_occurrence(occurrence):
        raise ValueError(
            "occurrence is available with lazy=False only: every source stores one "
            "occurrence per IDS, so the lazy path reads occurrence 0 by construction"
        )
    resolved = _resolve_source(source)
    shots = _shots(shot)
    ids = _declared_ids(name)
    from vaft.omas.plotting import render as render_ods

    with contextlib.ExitStack() as stack:
        objects = _open_all(stack, shots, resolved, ids, lazy=lazy, occurrence=occurrence)
        source_object = objects[0] if len(objects) == 1 else objects
        # Models copy what they read into their own arrays and renderers never
        # keep the data object, so the lazy stores may close on the way out.
        return render_ods(name, source_object, ax=ax, show=show, label=_labels(shots, label), **options)


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
        if not str(path).lower().endswith(".html"):
            raise ValueError(f"backend='plotly' writes HTML; give --out a .html path, not {path!r}")
        figure = render(name, shot, source, lazy=lazy, occurrence=occurrence, show=False, label=label, **options)
        figure.write_html(str(path), include_plotlyjs="cdn")
        return path
    use_non_interactive_backend()
    figure, _ = render(
        name, shot, source, lazy=lazy, occurrence=occurrence, show=False, label=label, **options
    )
    return save_figure(figure, path)


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
    from .lazy_common import discover_hsds_ids
    from . import utils

    present = discover_hsds_ids(_h5pyd(utils), resolved, int(shot))
    return describe_by_ids(
        present, source=f"#{shot} ({resolved})", query=query, detail=detail,
        available_only=available_only, **filters,
    )


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
    if name.startswith("plot_"):
        spec = _canonical(name[len("plot_"):])
        if spec is not None:
            adapter = _adapter(spec.name, spec.description)
            globals()[name] = adapter
            return adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    from vaft.plot.registry import canonical_names

    return sorted(set(globals()) | set(__all__) | {f"plot_{n}" for n in canonical_names()})
