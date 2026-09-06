"""Plotly rendering of the view models (issue #491).

The modules here draw the same :mod:`vaft.plot.models` the Matplotlib
renderers draw, with the same semantics -- labels, units, the validity and
uncertainty conventions, the legend policy, the panel layout -- as native
:class:`plotly.graph_objects.Figure` objects.  None of them imports
Matplotlib, and Plotly itself is imported only when a figure is asked for.

:data:`PLOTLY_MODELS` lists the model kinds covered; a spec whose model is
not among them is refused by :func:`vaft.plot.backends.renderer_for`.
Geometry, 3-D geometry, images, animations, power spectra and the camera
overlays are not covered in this first slice.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ..models import Field2D, LineSeries, Panels, Profile1D, Spectrogram, TextPanel

__all__ = ["PLOTLY_MODELS", "PlotlyRenderer", "renderer_for_model", "require_plotly"]


class PlotlyRenderer(NamedTuple):
    """How a model kind is drawn: on its own, and into a cell of a composite."""

    render: Callable[..., Any]
    add: Callable[..., Any]


def require_plotly() -> Any:
    """Import Plotly, naming the dependency when it is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - plotly is a dependency
        raise ImportError(
            "backend='plotly' needs the plotly package, a dependency of vaft; "
            "install it with `pip install plotly`"
        ) from error
    return go


def _table() -> dict[type, PlotlyRenderer]:
    from . import fields, lines, panels, profiles, spectrograms, text

    return {
        LineSeries: PlotlyRenderer(lines.render_line_series, lines.add_line_series),
        Profile1D: PlotlyRenderer(profiles.render_profile_1d, profiles.add_profile_1d),
        Field2D: PlotlyRenderer(fields.render_field_2d, fields.add_field_2d),
        Spectrogram: PlotlyRenderer(spectrograms.render_spectrogram, spectrograms.add_spectrogram),
        TextPanel: PlotlyRenderer(text.render_text_panel, text.add_text_panel),
        Panels: PlotlyRenderer(panels.render_panels, panels.add_panels),
    }


class _Models(dict):
    """The covered model kinds, filled on first use so importing costs nothing."""

    def _fill(self) -> None:
        if not dict.__len__(self):
            dict.update(self, _table())

    def __contains__(self, key: object) -> bool:
        self._fill()
        return dict.__contains__(self, key)

    def __getitem__(self, key: Any) -> PlotlyRenderer:
        self._fill()
        return dict.__getitem__(self, key)

    def __iter__(self):
        self._fill()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._fill()
        return dict.__len__(self)

    def keys(self):
        self._fill()
        return dict.keys(self)

    def values(self):
        self._fill()
        return dict.values(self)

    def items(self):
        self._fill()
        return dict.items(self)

    def get(self, key: Any, default: Any = None) -> Any:
        self._fill()
        return dict.get(self, key, default)


#: Model kind -> how Plotly draws it.
PLOTLY_MODELS: dict[type, PlotlyRenderer] = _Models()


def renderer_for_model(model_type: type) -> PlotlyRenderer:
    try:
        return PLOTLY_MODELS[model_type]
    except KeyError:
        raise NotImplementedError(
            f"no Plotly rendering for {model_type.__name__}"
        ) from None
