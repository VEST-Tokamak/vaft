"""The shared body of every ``plot_*`` adapter: entries in, a figure out.

A namespace normalises its input into ``(label, object)`` entries and calls
:func:`render_entries`; everything after that -- refusing an input that lacks
the plot's data, building the view model, choosing the renderer, splitting
styling from extraction options -- is the same for every data model.
"""

from __future__ import annotations

from typing import Any, Sequence

from vaft.plot.models import Panels
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
    namespace: str = "vaft.omas",
    subject: str = "ods",
    **options: Any,
) -> tuple[Any, Any]:
    """Build the view model for ``name`` from ``entries`` and render it.

    ``namespace``/``subject`` only shape the refusal message, so each adapter
    points at its own ``available_plots``.
    """
    from vaft.plot.renderers.panels import render_panels

    spec = get_spec(name)
    refuse_when_unsupported(name, entries, namespace=namespace, subject=subject)
    model = build_model(name, entries, **options)
    # A layout other than overlay arranges the same traces into a Panels model.
    # The canonical renderer is typed to the single-axes model, so the panels
    # renderer draws it instead; the return shape then follows the layout, as
    # issue #260 requires, and no renderer needs to know about layouts.
    renderer = spec.renderer
    if isinstance(model, Panels) and not issubclass(spec.model, Panels):
        renderer = render_panels
    style = {
        key: value for key, value in options.items() if key not in EXTRACTION_OPTIONS
    }
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
