"""Figure-level settings the reader may override: size, axis limits, scale.

Every plot draws with automatic defaults; these are presentation choices laid
over the drawn figure, the same for Matplotlib and Plotly, and never reach
the plot builder.  ``None`` keeps the default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

#: The colorbar axes Matplotlib adds carry this label; limits are for data.
_COLORBAR = "<colorbar>"


@dataclass(frozen=True)
class FigureSettings:
    """Width and height in pixels, axis limits in data units, log scales."""

    width: int | None = None
    height: int | None = None
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    xlog: bool = False
    ylog: bool = False

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value < 50:
                raise ValueError(f"{name} must be at least 50 px; got {value}")
        for low, high in (("xmin", "xmax"), ("ymin", "ymax")):
            lo, hi = getattr(self, low), getattr(self, high)
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"{low} must be below {high}; got {lo} >= {hi}")
        for axis, log in (("x", self.xlog), ("y", self.ylog)):
            for bound in (f"{axis}min", f"{axis}max"):
                value = getattr(self, bound)
                if log and value is not None and value <= 0:
                    raise ValueError(f"a log {axis} axis needs a positive {bound}; got {value}")

    def update(self, **changes: Any) -> "FigureSettings":
        return replace(self, **changes)

    @property
    def sized(self) -> bool:
        return self.width is not None or self.height is not None

    @property
    def key(self) -> str:
        """Changes whenever an explicit limit or scale does (Plotly uirevision)."""
        return repr((self.xmin, self.xmax, self.ymin, self.ymax, self.xlog, self.ylog))

    def apply(self, figure: Any, renderer: str, *, dpi: float | None = None) -> Any:
        """Lay the settings over ``figure`` in place; returns it.

        ``dpi`` is the resolution a Matplotlib figure will be saved at, so
        that a width in pixels is a width in pixels of the file.
        """
        if renderer == "plotly":
            return self._apply_plotly(figure)
        return self._apply_matplotlib(figure, dpi)

    def _apply_plotly(self, figure: Any) -> Any:
        if self.sized:
            figure.update_layout(width=self.width, height=self.height, autosize=self.width is None)
        for axis, low, high, log in (
            (figure.update_xaxes, self.xmin, self.xmax, self.xlog),
            (figure.update_yaxes, self.ymin, self.ymax, self.ylog),
        ):
            if log:
                axis(type="log")
            if low is not None or high is not None:
                # A log axis takes its range in decades.
                convert = (lambda v: None if v is None else math.log10(v)) if log else (lambda v: v)
                axis(range=[convert(low), convert(high)])
        return figure

    def _apply_matplotlib(self, figure: Any, dpi: float | None = None) -> Any:
        if self.sized:
            dpi = dpi or figure.get_dpi()
            width_in, height_in = figure.get_size_inches()
            figure.set_size_inches(
                self.width / dpi if self.width else width_in,
                self.height / dpi if self.height else height_in,
            )
        for ax in figure.axes:
            if ax.get_label() == _COLORBAR:
                continue
            if self.xlog:
                ax.set_xscale("log")
            if self.ylog:
                ax.set_yscale("log")
            if self.xmin is not None or self.xmax is not None:
                ax.set_xlim(left=self.xmin, right=self.xmax)
            if self.ymin is not None or self.ymax is not None:
                ax.set_ylim(bottom=self.ymin, top=self.ymax)
        return figure


EXPORT_FORMATS = ("png", "svg", "pdf")


__all__ = ["EXPORT_FORMATS", "FigureSettings"]
