"""Reproduce the figure on screen: Python, CLI, or Python with Pylustrator (#1421).

The text always comes from one :class:`vaft.plot.PlotRequest` -- the same
object the export draws -- so the copied code, the copied command and the
downloaded file never interpret the settings differently.  Only intent is
written: what the reader left inherited does not appear.
"""

from __future__ import annotations

from typing import Any, Callable

from ._require import require_panel

INHERIT = "(inherited)"

#: Clipboard copy in the browser; a page served on localhost is a secure
#: context, which navigator.clipboard requires.
_COPY_JS = "navigator.clipboard.writeText(text.value);"


class ReproducePanel:
    """Format and theme of the reproduced figure, and the code that draws it.

    ``request`` returns the :class:`~vaft.plot.PlotRequest` of what is on
    screen (or raises when nothing is drawn).
    """

    def __init__(self, request: Callable[[], Any], *, on_error: Callable[[BaseException], Any] | None = None) -> None:
        pn = require_panel()
        from vaft.plot.presentation import FORMATS, THEMES

        self._request = request
        self._on_error = on_error
        self.format = pn.widgets.Select(label="Format", options=[INHERIT, *FORMATS], value=INHERIT)
        self.theme = pn.widgets.Select(label="Theme", options=[INHERIT, *THEMES], value=INHERIT)
        self.python = pn.widgets.Button(label="Copy Python")
        self.cli = pn.widgets.Button(label="Copy CLI")
        self.pylustrator = pn.widgets.Button(label="Python + Pylustrator")
        self.text = pn.widgets.TextAreaInput(label="Reproduce", value="", rows=8, sizing_mode="stretch_width")
        self.copy = pn.widgets.Button(label="Copy to clipboard", disabled=True)
        self.copy.js_on_click(args={"text": self.text}, code=_COPY_JS)
        self.python.on_click(lambda _event: self._click("python"))
        self.cli.on_click(lambda _event: self._click("cli"))
        self.pylustrator.on_click(lambda _event: self._click("pylustrator"))

    def presentation(self) -> dict[str, Any]:
        """The ``format``/``theme`` the reader chose, leaving inherited ones out."""
        chosen = {}
        if self.format.value != INHERIT:
            chosen["format"] = self.format.value
        if self.theme.value != INHERIT:
            chosen["theme"] = self.theme.value
        return chosen

    def write(self, kind: str) -> str:
        """Fill the text box with the ``kind`` of reproduction and return it."""
        request = self._request()
        if kind == "cli":
            text = request.to_cli()
        else:
            text = request.to_python(pylustrator=kind == "pylustrator")
        self.text.value = text
        self.copy.disabled = False
        return text

    def _click(self, kind: str) -> None:
        try:
            self.write(kind)
        except Exception as error:
            # Nothing drawn (or nothing reproducible): no stale code may stay
            # in the box to be copied as if it drew the screen.
            self.text.value = ""
            self.copy.disabled = True
            if self._on_error is None:
                raise
            self._on_error(error)

    def widgets(self) -> list[Any]:
        pn = require_panel()
        return [
            pn.pane.Markdown(
                "Format and theme apply to the exported file and the copied code; "
                "the preview keeps the screen format.", margin=(0, 10), styles={"font-size": "0.85em"},
            ),
            self.format, self.theme, self.python, self.cli, self.pylustrator, self.text, self.copy,
        ]


__all__ = ["ReproducePanel"]
