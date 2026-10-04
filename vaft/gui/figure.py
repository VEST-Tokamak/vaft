"""What the browser pane does with a figure: its display size and the export formats.

Everything about the figure itself -- limits, scales, legend, type, colour
map -- is a :class:`vaft.plot.FigureOptions`, the reproducible options of
#1421 (see :mod:`vaft.gui.options_form`).  Only the size of the pane the
preview is shown in stays here: it is how the reader looks at the figure,
not part of the figure, and is never written into reproduced code.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPORT_FORMATS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class DisplaySize:
    """The preview pane's width and height in pixels; ``None`` fits the page."""

    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value < 50:
                raise ValueError(f"{name} must be at least 50 px; got {value}")


__all__ = ["DisplaySize", "EXPORT_FORMATS"]
