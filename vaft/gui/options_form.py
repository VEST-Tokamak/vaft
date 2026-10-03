"""The Figure Options form, generated from :class:`vaft.plot.FigureOptions` (#1421).

Every field of the options dataclass becomes one widget whose empty state
means *inherited* -- the format, theme or plot keeps deciding -- so the form
reads back intent only, the way :meth:`FigureOptions.to_dict` writes it.
The widgets are derived from the field types rather than listed here, so a
field added to :class:`~vaft.plot.FigureOptions` later shows up in the form
(in its section when :data:`SECTIONS` names it, under *More* otherwise).
Validation stays the dataclass's: a value it refuses is reported and the
form goes back to the last accepted options.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from ._require import require_panel

#: Where each field is shown; a field not named here goes under "More".
SECTIONS: dict[str, tuple[str, ...]] = {
    "Axes": ("title", "xlabel", "ylabel", "xlim", "ylim", "xscale", "yscale"),
    "Ticks and grid": ("grid", "minor_ticks", "tick_direction"),
    "Legend": ("legend", "legend_loc", "legend_ncols", "legend_frame"),
    "Type": ("font_size", "label_size", "tick_size", "title_size", "legend_size", "font_family", "math_fontset"),
    "Series": ("line_scale", "marker_scale"),
    "Scalar field": ("cmap", "clim", "colorbar_label"),
}

#: Labels where the field name alone reads poorly.
LABELS = {
    "xlabel": "x label", "ylabel": "y label", "xlim": "x limits", "ylim": "y limits",
    "xscale": "x scale", "yscale": "y scale", "legend_loc": "legend location",
    "legend_ncols": "legend columns", "legend_frame": "legend frame", "minor_ticks": "minor ticks",
    "tick_direction": "tick direction", "font_size": "base size [pt]", "label_size": "axis-label size [pt]",
    "tick_size": "tick-label size [pt]", "title_size": "title size [pt]", "legend_size": "legend size [pt]",
    "font_family": "text font (comma-separated)", "math_fontset": "math font", "line_scale": "line scale",
    "marker_scale": "marker scale", "cmap": "colour map", "clim": "colour range",
    "colorbar_label": "colorbar label",
}

INHERIT = "(inherited)"


def _choices() -> dict[str, tuple[str, ...]]:
    """The vocabulary of every choice field, from the options module itself."""
    from vaft.plot import figure_options as module

    choices = {
        "xscale": module.AXIS_SCALES, "yscale": module.AXIS_SCALES,
        "legend_loc": module.LEGEND_LOCATIONS, "tick_direction": module.TICK_DIRECTIONS,
        "math_fontset": module.MATH_FONTSETS,
    }
    # A module that grows a field with a vocabulary may publish it here.
    choices.update(getattr(module, "FIELD_CHOICES", {}))
    return choices


def _kind(annotation: Any) -> str:
    text = str(annotation)
    if text.startswith("bool"):
        return "bool"
    if text.startswith("int"):
        return "int"
    if text.startswith("float"):
        return "float"
    if text.startswith("tuple[float"):
        return "pair"
    if text.startswith("tuple[str"):
        return "names"
    return "str"


class FigureOptionsForm:
    """Widgets for every :class:`~vaft.plot.FigureOptions` field, grouped in cards."""

    def __init__(self, on_change: Callable[["FigureOptionsForm"], None] | None = None) -> None:
        pn = require_panel()
        from vaft.plot import FigureOptions

        self._options_class = FigureOptions
        self._on_change = on_change
        #: The field whose widget changed last (``None`` after a reset).
        self.changed: str | None = None
        self._updating = False
        self.widgets: dict[str, list[Any]] = {}
        self.kinds: dict[str, str] = {}
        choices = _choices()
        for item in dataclasses.fields(FigureOptions):
            kind = _kind(item.type)
            label = LABELS.get(item.name, item.name.replace("_", " "))
            if item.name in choices:
                kind = "choice"
                widgets = [pn.widgets.Select(label=label, options=[INHERIT, *choices[item.name]], value=INHERIT)]
            elif kind == "bool":
                widgets = [pn.widgets.Select(label=label, options=[INHERIT, "on", "off"], value=INHERIT)]
            elif kind == "int":
                widgets = [pn.widgets.IntInput(label=label, value=None, placeholder="inherited")]
            elif kind == "float":
                widgets = [pn.widgets.FloatInput(label=label, value=None, placeholder="inherited")]
            elif kind == "pair":
                widgets = [
                    pn.widgets.FloatInput(label=f"{label}: low", value=None, placeholder="inherited"),
                    pn.widgets.FloatInput(label=f"{label}: high", value=None, placeholder="inherited"),
                ]
            else:
                widgets = [pn.widgets.TextInput(label=label, value="", placeholder="inherited")]
            for widget in widgets:
                widget.param.watch(self._changed, "value")
            self.widgets[item.name] = widgets
            self.kinds[item.name] = kind
        self.reset_button = pn.widgets.Button(label="Reset all to inherited")
        self.reset_button.on_click(lambda _event: self.reset())

    # -- reading and writing ---------------------------------------------------------
    def value(self) -> Any:
        """The :class:`FigureOptions` the widgets say; ``ValueError`` when refused."""
        data: dict[str, Any] = {}
        for name, widgets in self.widgets.items():
            kind = self.kinds[name]
            if kind == "pair":
                low, high = (widget.value for widget in widgets)
                if low is not None or high is not None:
                    data[name] = (low, high)
                continue
            value = widgets[0].value
            if kind in ("choice", "bool"):
                if value == INHERIT:
                    continue
                data[name] = {"on": True, "off": False}.get(value, value) if kind == "bool" else value
            elif kind == "names":
                faces = [face.strip() for face in (value or "").split(",") if face.strip()]
                if faces:
                    data[name] = tuple(faces)
            elif kind == "str":
                # An empty box is inherited; a title made of spaces hides it.
                if value:
                    data[name] = "" if not value.strip() else value
            elif value is not None:
                data[name] = value
        return self._options_class(**data)

    def show(self, options: Any, *, keep: str | None = None) -> None:
        """Put the widgets on ``options`` without reporting a change.

        ``keep`` names a field left as typed.
        """
        self._updating = True
        try:
            for name, widgets in self.widgets.items():
                if name == keep:
                    continue
                kind = self.kinds[name]
                value = getattr(options, name)
                if kind == "pair":
                    low, high = value if value is not None else (None, None)
                    widgets[0].value, widgets[1].value = low, high
                elif kind == "choice":
                    widgets[0].value = INHERIT if value is None else value
                elif kind == "bool":
                    widgets[0].value = INHERIT if value is None else ("on" if value else "off")
                elif kind == "names":
                    widgets[0].value = ", ".join(value) if value else ""
                elif kind == "str":
                    widgets[0].value = "" if value is None else (value or " ")
                else:
                    widgets[0].value = value
        finally:
            self._updating = False

    def reset(self) -> None:
        """Every field back to inherited (not to the value it resolved to)."""
        self.show(self._options_class())
        self.changed = None
        if self._on_change is not None:
            self._on_change(self)

    def _changed(self, event: Any) -> None:
        if self._updating:
            return
        self.changed = next((name for name, widgets in self.widgets.items() if event.obj in widgets), None)
        if self._on_change is not None:
            self._on_change(self)

    # -- layout ------------------------------------------------------------------------
    def cards(self) -> list[Any]:
        """One collapsed card per section, *More* last for fields no section names."""
        pn = require_panel()
        placed = {name for names in SECTIONS.values() for name in names}
        sections = {title: [name for name in names if name in self.widgets] for title, names in SECTIONS.items()}
        rest = [name for name in self.widgets if name not in placed]
        if rest:
            sections["More"] = rest
        return [
            pn.Card(
                *[widget for name in names for widget in self.widgets[name]],
                title=title, collapsed=True, sizing_mode="stretch_width",
            )
            for title, names in sections.items() if names
        ] + [self.reset_button]


__all__ = ["FigureOptionsForm", "INHERIT", "LABELS", "SECTIONS"]
