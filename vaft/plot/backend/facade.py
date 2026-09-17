"""The ``dd_<stem>`` and ``extract_<stem>`` twins of every ``plot_<stem>`` adapter.

A canonical plot answers three questions (umbrella #434): what Data Dictionary
paths it reads (``dd_*``, no data touched), what data lies behind the figure
(``extract_*``, the view model, undrawn) and the figure itself (``plot_*``).
All three are generated here from one registry pass, so the surfaces cover one
set of plots by construction.  ``plot_*`` is generated only where the namespace
has not written one out by hand -- an adapter that does more than call
``render``, or whose prose says something the registry description does not,
keeps its own body.

``extract_*`` takes the extraction options only
(:data:`vaft.plot.backend.options.EXTRACTION_OPTIONS`): a rendering keyword
such as ``ax=`` or ``cmap=`` is refused by name, pointing at ``plot_*``; an
unknown keyword is refused as ``plot_*`` refuses it.  It
returns exactly what :func:`vaft.plot.backend.recipes.build_model` builds, so
``plot_*`` and ``render(extract_*(...))`` cannot diverge.

``vaft.database`` builds its own twins in :mod:`vaft.database.plotting`: its
``plot_*`` signature is ``(shot, source, lazy=, occurrence=)`` and opens IDS
selectively, so ``extract_*`` there opens the same IDS and hands the loaded
ODS to :func:`vaft.plot.backend.recipes.build_model`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from vaft.plot.registry import specs

from .options import STYLE_OPTIONS, validate_options

__all__ = ["RENDER_KEYWORDS", "install_facades", "refuse_render_options"]

#: Keywords that belong to drawing, not extraction, besides the renderers'
#: own style names: the canvas, the backend, the presentation presets and
#: the interaction switches of :func:`vaft.plot.backend.render.render_entries`.
RENDER_KEYWORDS = frozenset(
    {"ax", "show", "backend", "interactive", "controls", "interaction_backend", "format", "theme"}
)


def refuse_render_options(name: str, options: dict[str, Any]) -> None:
    """Refuse a keyword ``extract_<name>`` cannot honour because it draws nothing."""
    drawn = sorted(key for key in options if key in RENDER_KEYWORDS or key in STYLE_OPTIONS)
    if drawn:
        raise TypeError(
            f"extract_{name} builds the view model and draws nothing; "
            f"{', '.join(f'{key}=' for key in drawn)} "
            f"{'is a rendering keyword' if len(drawn) == 1 else 'are rendering keywords'} "
            f"-- pass them to plot_{name}"
        )
    validate_options(name, options)


def _dd_function(name: str, *, namespace: str, description: str) -> Callable[[], tuple]:
    def dd_function() -> tuple:
        from .dd import dd_paths

        return dd_paths(name)

    dd_function.__name__ = dd_function.__qualname__ = f"dd_{name}"
    dd_function.__module__ = f"{namespace}.plotting"
    dd_function.__doc__ = (
        f"The Data Dictionary paths :func:`{namespace}.plot_{name}` reads, without any data.\n\n"
        f"{description}\n\n"
        f"Returns a tuple of :class:`vaft.plot.backend.dd.DDPath`, data paths first; "
        f"see :func:`vaft.plot.backend.dd.dd_paths`."
    )
    return dd_function


def _extract_function(
    name: str, *, normalize: Callable[..., Sequence[tuple[str, Any]]], namespace: str,
    subject: str, description: str, model: str,
) -> Callable[..., Any]:
    def extract_function(source: Any, *, label: str | Sequence[str] = "shot", **options: Any) -> Any:
        from .recipes import build_model
        from .render import refuse_when_unsupported

        refuse_render_options(name, options)
        entries = normalize(source, label=label)
        refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
        return build_model(name, entries, **options)

    extract_function.__name__ = extract_function.__qualname__ = f"extract_{name}"
    extract_function.__module__ = f"{namespace}.plotting"
    extract_function.__doc__ = (
        f"The data behind :func:`{namespace}.plot_{name}`, as its view model, undrawn.\n\n"
        f"{description}\n\n"
        f"Returns the :class:`vaft.plot.models.{model}` that ``plot_{name}`` draws, built "
        f"from the same input and the same extraction options (``label=`` and "
        f":data:`vaft.plot.backend.options.EXTRACTION_OPTIONS`); a rendering keyword such "
        f"as ``ax=`` is refused.  ``.to_xarray()`` on the result gives an "
        f":class:`xarray.Dataset`; :func:`{namespace}.dd_{name}` lists the Data Dictionary "
        f"paths it read."
    )
    return extract_function


def _plot_function(
    name: str, *, module_globals: dict[str, Any], namespace: str, description: str,
) -> Callable[..., Any]:
    def plot_function(
        source: Any,
        *,
        ax: Any = None,
        show: bool = False,
        label: str | Sequence[str] = "shot",
        **options: Any,
    ) -> Any:
        # Looked up at call time, not closed over: an adapter written out by
        # hand reads `render` as a module global, so capturing it here would
        # make the two halves of one `__all__` disagree under monkeypatch.
        return module_globals["render"](
            name, source, ax=ax, show=show, label=label, **options
        )

    plot_function.__name__ = plot_function.__qualname__ = f"plot_{name}"
    plot_function.__module__ = f"{namespace}.plotting"
    plot_function.__doc__ = (
        f"{description}\n\n"
        f"Renders with :func:`vaft.plot.{name}`.  "
        f":func:`{namespace}.extract_{name}` returns the same figure's data undrawn, and "
        f":func:`{namespace}.dd_{name}` lists the Data Dictionary paths it reads."
    )
    return plot_function


def install_facades(
    module_globals: dict[str, Any], *, normalize: Callable[..., Any], namespace: str, subject: str,
) -> tuple[str, ...]:
    """Bind the three verbs of every canonical plot into a module.

    ``dd_<stem>`` and ``extract_<stem>`` are always generated. ``plot_<stem>``
    is generated **only when the module does not already define it**, so an
    adapter written out by hand -- because it does more than call ``render``, or
    because its prose says something the registry description does not -- keeps
    its own body.

    That asymmetry is the point. Before this, ``plot_*`` was the one verb the
    registry did not produce, so registering a plot created two of the three
    and silently not the third; the drift then surfaced as a failure of
    ``test_the_three_verbs_cover_the_same_plots`` on a push rather than on the
    PR, because that test is outside the ``core`` selection (issue #815, and
    #810 which is how it was found).

    ``normalize`` is the namespace's own entry normaliser
    (``vaft.omas.entries.normalize_entries`` or ``vaft.imas.entries.normalize_entries``);
    ``namespace``/``subject`` shape the refusal messages exactly as the
    ``plot_*`` adapters do.  Returns the names bound, for ``__all__``.
    """
    if not callable(module_globals.get("render")):
        raise TypeError(
            f"{namespace}.plotting must define `render` before installing facades; "
            "the generated plot_* adapters are thin wrappers around it."
        )

    names: list[str] = []
    for spec in specs():
        dd_name, extract_name = f"dd_{spec.name}", f"extract_{spec.name}"
        module_globals[dd_name] = _dd_function(
            spec.name, namespace=namespace, description=spec.description,
        )
        module_globals[extract_name] = _extract_function(
            spec.name, normalize=normalize, namespace=namespace, subject=subject,
            description=spec.description, model=spec.model.__name__,
        )
        names += [dd_name, extract_name]

        plot_name = f"plot_{spec.name}"
        if plot_name not in module_globals:
            module_globals[plot_name] = _plot_function(
                spec.name, module_globals=module_globals, namespace=namespace,
                description=spec.description,
            )
            names.append(plot_name)
    return tuple(names)
