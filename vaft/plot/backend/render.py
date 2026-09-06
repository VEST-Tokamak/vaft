"""The shared body of every ``plot_*`` adapter: entries in, a figure out.

A namespace normalises its input into ``(label, object)`` entries and calls
:func:`render_entries`; everything after that -- refusing an input that lacks
the plot's data, building the view model, choosing the renderer, splitting
styling from extraction options -- is the same for every data model.
"""

from __future__ import annotations

from typing import Any, Sequence

from vaft.plot.backends import renderer_for, resolve_render_backend
from vaft.plot.registry import get_spec

from .recipes import (
    EXTRACTION_OPTIONS,
    build_model,
    diagnoses_itself,
    missing_required_path,
)

__all__ = ["refuse_when_unsupported", "render_entries"]


def render_entries(
    name: str,
    entries: Sequence[tuple[str, Any]],
    *,
    ax: Any = None,
    show: bool = False,
    backend: str | None = None,
    namespace: str = "vaft.omas",
    subject: str = "ods",
    **options: Any,
) -> Any:
    """Build the view model for ``name`` from ``entries`` and render it.

    ``namespace``/``subject`` only shape the refusal message, so each adapter
    points at its own ``available_plots``.  ``backend`` picks the drawing
    library (:data:`vaft.plot.backends.RENDER_BACKENDS`): Matplotlib by
    default, returning ``(Figure, Axes)``; ``"plotly"`` returns a
    :class:`plotly.graph_objects.Figure` and takes no ``ax=``.  The model is
    built the same way whichever draws it.
    """
    backend = resolve_render_backend(backend)
    spec = get_spec(name)
    refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
    model = build_model(name, entries, **options)
    # A layout other than overlay arranges the same traces into a Panels model;
    # renderer_for hands such a model to the panels renderer, so the return
    # shape follows the layout (issue #260) and no renderer knows about layouts.
    renderer = renderer_for(spec, model, backend)
    style = {
        key: value for key, value in options.items() if key not in EXTRACTION_OPTIONS
    }
    if backend == "plotly":
        if ax is not None:
            raise TypeError(
                "ax= is a Matplotlib axes; backend='plotly' draws a new plotly Figure "
                "and returns it"
            )
        return renderer(model, show=show, **style)
    return renderer(model, ax=ax, show=show, **style)


def refuse_when_unsupported(
    name: str,
    entries: Sequence[tuple[str, Any]],
    *,
    namespace: str = "vaft.omas",
    subject: str = "ods",
) -> None:
    """Raise rather than render a figure with nothing in it.

    A path-driven adapter whose leaf is absent used to return an empty figure --
    no lines, no error, nothing to say why. That is worse than failing: it is
    also why a plot could be missing from ``available_plots(obj)`` while the
    function itself still "succeeded" (issue #290).

    The guard asks the same question ``available_plots`` asks, so the two
    agree by construction. It covers only the plain path reads: composites drop
    unsupported panels on purpose and then raise about the ones that remain, and
    the recipes that run real code raise something more specific than a missing
    path. Speaking over either would replace a good diagnosis with a worse one.
    """
    if not entries or diagnoses_itself(name):
        return
    missing = [missing_required_path(obj, name) for _, obj in entries]
    if any(path is None for path in missing):
        return
    wanted = missing[0]
    # The equilibrium hint is only offered where it applies: pointing someone at
    # an equilibrium updater because a Thomson channel is missing is worse than
    # saying nothing.
    remedy = (
        "Equilibrium profiles an EFIT g-file does not store are derived by "
        "vaft.omas.update_equilibrium_derived_profiles(ods); "
        if wanted.startswith("equilibrium.")
        else ""
    )
    raise ValueError(
        f"{name!r} requires {wanted}, which is not available in this input. "
        f"{remedy}"
        f"{namespace}.available_plots({subject}) lists what this object can already plot."
    )
