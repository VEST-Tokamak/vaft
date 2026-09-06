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

from .options import split_options, validate_options
from .recipes import (
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
    interactive: bool = False,
    controls: str | Sequence[str] = "auto",
    interaction_backend: str = "auto",
    **options: Any,
) -> Any:
    """Build the view model for ``name`` from ``entries`` and render it.

    ``namespace``/``subject`` only shape the refusal message, so each adapter
    points at its own ``available_plots``.  ``backend`` picks the drawing
    library (:data:`vaft.plot.backends.RENDER_BACKENDS`): Matplotlib by
    default, returning ``(Figure, Axes)``; ``"plotly"`` returns a
    :class:`plotly.graph_objects.Figure` and takes no ``ax=``.  The model is
    built the same way whichever draws it.

    ``interactive=True`` (issue #480) draws the same plot with the controls
    its capability record supports -- ``controls`` names a subset, or
    ``"auto"`` for all of them -- and returns a
    :class:`vaft.plot.renderers.interactive.Interactive` whose ``state``
    rebuilds and redraws the plot; ``interaction_backend`` is one of
    :data:`vaft.plot.renderers.interactive.BACKENDS`.  Every option given
    here is the control's starting value.
    """
    backend = resolve_render_backend(backend)
    spec = get_spec(name)
    validate_options(name, options)
    refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
    if interactive:
        return _render_interactive(
            spec, entries, options, backend=backend, controls=controls,
            interaction_backend=interaction_backend, show=show, ax=ax,
        )
    model = build_model(name, entries, **options)
    # A layout other than overlay arranges the same traces into a Panels model;
    # renderer_for hands such a model to the panels renderer, so the return
    # shape follows the layout (issue #260) and no renderer knows about layouts.
    renderer = renderer_for(spec, model, backend)
    _, style = split_options(options)
    if backend == "plotly":
        if ax is not None:
            raise TypeError(
                "ax= is a Matplotlib axes; backend='plotly' draws a new plotly Figure "
                "and returns it"
            )
        return renderer(model, show=show, **style)
    return renderer(model, ax=ax, show=show, **style)


def _render_interactive(
    spec: Any,
    entries: Sequence[tuple[str, Any]],
    options: dict[str, Any],
    *,
    backend: str,
    controls: str | Sequence[str],
    interaction_backend: str,
    show: bool,
    ax: Any,
) -> Any:
    """The controls of ``spec`` over ``entries``, from the capability record."""
    from vaft.plot.controls import controls_for
    from vaft.plot.navigation import ControlState
    from vaft.plot.renderers.interactive import render_controls

    from .discovery import describe_one

    if ax is not None:
        raise TypeError("interactive=True draws its own figure and takes no ax=")
    record = describe_one(spec.name, entries)
    offered = controls_for(record)
    if controls != "auto":
        wanted = [controls] if isinstance(controls, str) else list(controls)
        unknown = [name for name in wanted if name not in {c.name for c in offered}]
        if unknown:
            raise ValueError(
                f"plot_{spec.stem} offers no control named {', '.join(map(repr, unknown))}; "
                f"offered: {', '.join(c.name for c in offered) or 'none'}"
            )
        offered = tuple(c for c in offered if c.name in wanted)
    extraction, style = split_options(options)
    names = {c.name for c in offered}
    initial = {k: v for k, v in {**extraction, **style}.items() if k in names}
    fixed = {k: v for k, v in extraction.items() if k not in names}
    fixed_style = {k: v for k, v in style.items() if k not in names}
    state = ControlState(offered, initial)

    def build(chosen: dict[str, Any]) -> Any:
        return build_model(spec.name, entries, **{**fixed, **chosen})

    def draw(model: Any, **kwargs: Any) -> Any:
        return renderer_for(spec, model, backend)(model, **{**fixed_style, **kwargs})

    return render_controls(
        build, state, draw=draw, backend=interaction_backend, render_backend=backend, show=show,
    )


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
