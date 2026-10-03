"""The shared body of every ``plot_*`` adapter: entries in, a figure out.

A namespace normalises its input into ``(label, object)`` entries and calls
:func:`render_entries`; everything after that -- refusing an input that lacks
the plot's data, building the view model, choosing the renderer, splitting
styling from extraction options -- is the same for every data model.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from vaft.plot.backends import renderer_for, resolve_render_backend
from vaft.plot.registry import NON_GRAPHICAL_VIEWS, get_spec

from .options import split_options, validate_options
from .recipes import (
    build_model,
    diagnoses_itself,
    missing_required_path,
)

__all__ = [
    "STATE_SELECTORS",
    "frame_renderers",
    "refuse_when_unsupported",
    "render_entries",
    "render_text_view",
]

#: Keywords that each pick one state of a time-resolved plot (issue #1380).
STATE_SELECTORS = ("time", "time_slice", "time_index", "frame_index")


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
    animation: bool = False,
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

    ``animation=True`` (issues #1049/#1050) draws the same plot over a
    sequence of its states -- the values of its slice control, the one the
    ``interactive=True`` slider drives -- and returns a lazy
    :class:`vaft.plot._animation.Animation` with ``save("x.mp4")``;
    ``fps=``/``duration=`` set the playback, never the physics.
    """
    spec = get_spec(name)
    if spec.view in NON_GRAPHICAL_VIEWS:
        return render_text_view(
            spec, entries, ax=ax, show=show, backend=backend, namespace=namespace,
            subject=subject, interactive=interactive, animation=animation,
            controls=controls, interaction_backend=interaction_backend, **options,
        )
    backend = resolve_render_backend(backend)
    if animation:
        if interactive:
            raise TypeError(
                "animation=True and interactive=True are separate presentations of the same "
                "sequence; choose one"
            )
        if controls != "auto" or interaction_backend != "auto":
            raise TypeError(
                "controls= and interaction_backend= belong to interactive=True; "
                "animation=True offers no controls"
            )
        from vaft.plot._animation import render_animation

        refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
        return render_animation(spec, entries, options, backend=backend, show=show, ax=ax)
    validate_options(name, options)
    refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
    from vaft.plot.figure_options import as_figure_options

    figure_options = as_figure_options(options.pop("figure_options", None))
    if figure_options and (interactive or animation):
        raise TypeError(
            "figure_options= applies to a drawn figure; interactive=True and animation=True "
            "redraw their own and do not take it yet"
        )
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
    from vaft.plot.figure_options import figure_options_scope

    if backend == "plotly":
        if ax is not None:
            raise TypeError(
                "ax= is a Matplotlib axes; backend='plotly' draws a new plotly Figure "
                "and returns it"
            )
        _refuse_presentation(style, "backend='plotly' does not apply them")
        figure = renderer(model, show=False, **style)
        if figure_options:
            figure_options.apply_plotly(figure)
        if show:
            figure.show()
        return figure
    if not figure_options:
        return renderer(model, ax=ax, show=show, **style)
    with figure_options_scope(figure_options):
        result = renderer(model, ax=ax, show=False, **style)
    # A caller's own canvas keeps its other axes: edit only what was drawn.
    figure_options.apply(result[0], None if ax is None else result[1])
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return result


def render_text_view(
    spec: Any,
    entries: Sequence[tuple[str, Any]],
    *,
    ax: Any = None,
    show: bool = False,
    backend: str | None = None,
    namespace: str = "vaft.omas",
    subject: str = "ods",
    interactive: bool = False,
    animation: bool = False,
    controls: str | Sequence[str] = "auto",
    interaction_backend: str = "auto",
    **options: Any,
) -> Any:
    """A ``table`` or ``text`` view (issue #1180): the model, presented as text.

    The model is built exactly as for a figure; the renderer returns a
    :class:`vaft.plot.renderers.tables.TextView` instead of ``(Figure, Axes)``
    -- printed when ``show=True``.  The keywords that only mean something to
    a drawn figure are refused by name rather than ignored: ``ax=``,
    ``format=``, ``theme=``, ``figure_options=``, ``backend="plotly"``, the
    interaction switches and any renderer style keyword.
    """
    kind = f"plot_{spec.stem} is a {spec.view} view and returns text, not a figure"
    refused = []
    if ax is not None:
        refused.append("ax=")
    if backend not in (None, "matplotlib"):
        refused.append(f"backend={backend!r}")
    for key in ("format", "theme", "figure_options"):
        if options.get(key) not in (None, "", "none"):
            refused.append(f"{key}=")
    if interactive:
        refused.append("interactive=True")
    if animation:
        refused.append("animation=True")
    if controls != "auto" or interaction_backend != "auto":
        refused.append("controls=/interaction_backend=")
    options = {k: v for k, v in options.items() if k not in ("format", "theme", "figure_options")}
    extraction, style = split_options(options)
    refused += [f"{key}=" for key in sorted(style)]
    if refused:
        raise TypeError(
            f"{kind}; {', '.join(refused)} "
            f"{'applies' if len(refused) == 1 else 'apply'} to drawn figures only "
            "(Matplotlib/Plotly presentation). Print it, or export it with "
            ".text(), .markdown() or .html()."
        )
    validate_options(spec.name, extraction)
    refuse_when_unsupported(spec.name, entries, namespace=namespace, subject=subject)
    model = build_model(spec.name, entries, **extraction)
    return spec.renderer(model, show=show)


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
    # A theme is a control here (issue #710); the canvas is the controls
    # figure's, so format= alone is refused.
    _refuse_presentation(
        options, "interactive=True draws its own controls figure", keys=("format",)
    )
    record = describe_one(spec.name, entries)
    offered = controls_for(record)
    if backend == "plotly":
        # Plotly cannot apply a Matplotlib theme: the control is not offered
        # rather than raising on first use, and a theme given to the call is
        # refused as the static Plotly path refuses it.
        _refuse_presentation(options, "backend='plotly' does not apply them", keys=("theme",))
        offered = tuple(c for c in offered if c.name != "theme")
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
    # An instant the caller chose with another selector (time= on a plot whose
    # slider is time_index, time_slice= on a camera frame) pins the state the
    # same way: the slice control is not offered, since a builder takes one
    # selector at a time (issue #1380).
    pinned = [key for key in STATE_SELECTORS if extraction.get(key) is not None]
    offered = tuple(
        c for c in offered if not (c.group == "slice" and any(key != c.name for key in pinned))
    )
    # A starting value the static call accepts but the control's list does not
    # hold (selection=[0, 3], yunit="auto", a slice outside the usable ones)
    # stays what the caller fixed: that one control is not offered, and the
    # value reaches the builder as it would without interactive=True.
    unlisted = set()
    for control in offered:
        if control.name in extraction or control.name in style:
            try:
                control.validate({**extraction, **style}[control.name])
            except (TypeError, ValueError):
                unlisted.add(control.name)
    offered = tuple(c for c in offered if c.name not in unlisted)
    names = {c.name for c in offered}
    initial = {k: v for k, v in {**extraction, **style}.items() if k in names}
    fixed = {k: v for k, v in extraction.items() if k not in names}
    fixed_style = {k: v for k, v in style.items() if k not in names}
    state = ControlState(offered, initial)
    build, draw = frame_renderers(spec, entries, fixed, fixed_style, backend)
    if style.get("validity") is not None and "validity" not in fixed_style:
        # The caller stated a validity mode and it became the control's start:
        # the channels it brings in stay in, whatever the control is set to,
        # so opening the controls draws what the static call draws.
        stated = style["validity"]
        plain_build = build

        def build(chosen: Mapping[str, Any]) -> Any:
            return plain_build({"validity": stated, **chosen})
    return render_controls(
        build, state, draw=draw, backend=interaction_backend, render_backend=backend, show=show,
    )


def frame_renderers(
    spec: Any,
    entries: Sequence[tuple[str, Any]],
    fixed: Mapping[str, Any],
    fixed_style: Mapping[str, Any],
    backend: str,
) -> tuple[Any, Any]:
    """``(build, draw)`` for one plot at a chosen state.

    ``build(chosen)`` makes the view model with the chosen control values
    over the fixed options; ``draw(model, **kwargs)`` renders it.  The
    interactive controls redraw through this pair and ``animation=True``
    draws its frames through it, so a frame and a slider position of the
    same state come from the same builder and renderer (the animation only
    fixes the scale, limits and resolution across its frames).
    """

    # validity= is drawn by the renderer, but a mode the caller states also
    # tells the signal presets to keep condemned channels for that mode to
    # handle (issue #1380), so the builder sees it as the static call does.
    hints = {"validity": fixed_style["validity"]} if fixed_style.get("validity") is not None else {}

    def build(chosen: Mapping[str, Any]) -> Any:
        return build_model(spec.name, entries, **{**fixed, **hints, **chosen})

    def draw(model: Any, **kwargs: Any) -> Any:
        return renderer_for(spec, model, backend)(model, **{**fixed_style, **kwargs})

    return build, draw


def _refuse_presentation(
    options: Mapping[str, Any], because: str, *, keys: tuple[str, ...] = ("format", "theme")
) -> None:
    """Refuse ``format=``/``theme=`` where they would otherwise be dropped.

    They are Matplotlib presentation presets (issue #689) and a path that
    cannot apply them must say so rather than draw something else.  ``keys``
    narrows the refusal to the presets a path really cannot take.
    """
    named = [key for key in keys if options.get(key) not in (None, "", "none")]
    if named:
        what = " and ".join(f"{key}=" for key in named)
        verb = "is a Matplotlib presentation preset" if len(named) == 1 else "are Matplotlib presentation presets"
        raise NotImplementedError(f"{what} {verb}; {because}")


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
