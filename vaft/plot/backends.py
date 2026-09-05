"""Which library draws a view model: the rendering backends (issue #491).

A canonical plot is a *view model* built from data plus a *renderer* that
draws it.  Matplotlib is the default and canonical renderer, and every spec's
registered function is a Matplotlib one.  Plotly is a second renderer for the
model kinds it covers; a plot whose model it does not cover says so instead
of falling back.  Neither the choice of backend nor the library it names
touches what is drawn: the same model, units, labels, sign and ordering
reach both.

``backend=`` on a ``plot_*`` adapter picks the drawing library.  It is not
``plot_*_interactive``'s ``backend`` (which names the *interaction* --
``auto | matplotlib | ipywidgets | none``) and not ``plot_*_overview``
(which composes members); the three are documented together in the
plotting notebook.
"""

from __future__ import annotations

from typing import Any, Callable

from .models import Panels
from .registry import PlotSpec

__all__ = [
    "RENDER_BACKENDS",
    "renderer_for",
    "render_backends_for",
    "resolve_render_backend",
    "supports_backend",
]

#: The rendering libraries a ``plot_*`` adapter may be asked for.
RENDER_BACKENDS = ("matplotlib", "plotly")


def resolve_render_backend(backend: str | None) -> str:
    """The backend name to draw with; ``None`` is Matplotlib, the default."""
    if backend is None:
        return "matplotlib"
    if backend not in RENDER_BACKENDS:
        raise ValueError(
            f"backend must be one of {', '.join(RENDER_BACKENDS)}; got {backend!r}"
        )
    return backend


def _plotly_covers(model_type: type) -> bool:
    from .plotly import PLOTLY_MODELS

    return model_type in PLOTLY_MODELS


def supports_backend(model_type: type, backend: str) -> bool:
    """Whether ``backend`` can draw a view model of ``model_type``.

    For :class:`Panels` this answers for the composite itself; whether every
    member can be drawn is decided per instance by :func:`renderer_for`.
    """
    backend = resolve_render_backend(backend)
    if backend == "matplotlib":
        return True
    return _plotly_covers(model_type)


def render_backends_for(spec: PlotSpec) -> tuple[str, ...]:
    """The backends that can draw ``spec``'s model kind, for discovery."""
    return tuple(name for name in RENDER_BACKENDS if supports_backend(spec.model, name))


def renderer_for(spec: PlotSpec, model: Any, backend: str) -> Callable[..., Any]:
    """The function that draws ``model`` for ``spec`` with ``backend``.

    Matplotlib: the spec's registered renderer, or the panels renderer when a
    layout arranged the traces into a :class:`Panels` the spec's own renderer
    is not typed for.  Plotly: the renderer for the model's kind, refused
    with :class:`NotImplementedError` when the kind -- or, for a composite,
    any member's kind -- is not covered.
    """
    backend = resolve_render_backend(backend)
    if backend == "matplotlib":
        from .renderers.panels import render_panels

        if isinstance(model, Panels) and not issubclass(spec.model, Panels):
            return render_panels
        return spec.renderer
    from .plotly import PLOTLY_MODELS, require_plotly

    require_plotly()
    kinds = [type(model)] + ([type(member) for member in model.models] if isinstance(model, Panels) else [])
    missing = [kind.__name__ for kind in kinds if kind not in PLOTLY_MODELS]
    if missing:
        raise NotImplementedError(
            f"plot_{spec.stem} does not currently support backend='plotly' "
            f"(no Plotly rendering for {', '.join(dict.fromkeys(missing))})"
        )
    return PLOTLY_MODELS[type(model)].render
