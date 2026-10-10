"""One page for every screen: the responsive rules of ``vaft gui`` (#1865).

The GUI is one Panel application on desktops, tablets and phones -- no second
frontend.  The template (Panel's FastListTemplate) lays the sidebar and the
main area side by side at a fixed sidebar width, which on a phone leaves the
main area -- status, errors, the figure -- with no width at all.  These rules
change only the arrangement:

* **tablet** (narrower than 1100 px): the sidebar narrows so the figure keeps
  room;
* **phone** (narrower than 768 px, or a phone held sideways): the controls
  become a drawer over the page at full width, and the main area -- status,
  errors, the figure -- has the whole width beneath it.  The ☰ button opens
  and closes the drawer; on a phone the page opens with the drawer closed, so
  the figure is the first thing seen (:data:`PHONE_START_JS`).  The drawer keeps the
  template's own height and scrolling, so nothing inside it is resized.

Wide scientific content scrolls inside its own box instead of widening the
page: tables are wrapped by :func:`scrolling_markdown`.  Every control stays
reachable at every width; nothing is hidden on small screens.
"""

from __future__ import annotations

from typing import Any

from ._require import require_panel

__all__ = ["PHONE_MAX_WIDTH", "PHONE_QUERY", "PHONE_START_JS", "RESPONSIVE_CSS", "SIDEBAR_WIDTH", "TABLET_MAX_WIDTH", "page", "scrolling_markdown"]

#: Sidebar width on a desktop, in pixels.
SIDEBAR_WIDTH = 360
#: Below this width the sidebar narrows (tablets, small laptops).
TABLET_MAX_WIDTH = 1100
#: Below this width the page stacks into one column (phones).
PHONE_MAX_WIDTH = 767

RESPONSIVE_CSS = f"""
@media (max-width: {TABLET_MAX_WIDTH}px) {{
  #sidebar:not(.hidden) {{ min-width: 300px; max-width: 300px; }}
}}
@media (max-width: {PHONE_MAX_WIDTH}px), (max-height: 500px) and (max-width: 950px) {{
  #content {{ position: relative; }}
  #main {{ width: 100%; min-width: 0; }}
  #sidebar:not(.hidden) {{
    position: absolute; top: 0; left: 0; z-index: 10;
    width: 100%; min-width: 100%; max-width: 100%;
    background: var(--background-color, #fff); border-right: 0;
  }}
}}
"""

#: A Markdown pane whose tables scroll sideways inside it on a narrow screen.
_SCROLLING_TABLE = """
:host { max-width: 100%; overflow-x: auto; display: block; }
table { min-width: 36em; }
"""


def scrolling_markdown(text: str = "", **options: Any) -> Any:
    """A Markdown pane that keeps a wide table inside a horizontal scroll box."""
    pn = require_panel()
    options.setdefault("sizing_mode", "stretch_width")
    stylesheets = list(options.pop("stylesheets", [])) + [_SCROLLING_TABLE]
    return pn.pane.Markdown(text, stylesheets=stylesheets, **options)


#: The media query that makes a screen a phone (same as in :data:`RESPONSIVE_CSS`).
PHONE_QUERY = f"(max-width: {PHONE_MAX_WIDTH}px), (max-height: 500px) and (max-width: 950px)"


#: Runs once the page has loaded: on a phone, close the drawer with the
#: template's own ``closeNav`` (the function behind the ☰ button), so the
#: figure is what the reader sees first and ☰ reopens the controls.
PHONE_START_JS = (
    "window.addEventListener('load', function () {"
    f"  if (!window.matchMedia('{PHONE_QUERY}').matches) return;"
    "  var tries = 0;"
    "  (function close() {"
    "    if (typeof window.closeNav === 'function') { window.closeNav(); }"
    "    else if (tries++ < 50) { setTimeout(close, 100); }"
    "  })();"
    "});"
)


def _data_uri(script: str) -> str:
    import base64

    return "data:text/javascript;base64," + base64.b64encode(script.encode()).decode()


def page(sidebar: list[Any], main: list[Any], *, title: str = "VAFT", **template: Any) -> Any:
    """The template every ``vaft gui`` page uses, with the responsive rules.

    ``template`` passes branding (``logo``, ``favicon``, ``header_background``)
    and other FastListTemplate options through.
    """
    pn = require_panel()
    raw_css = [RESPONSIVE_CSS, *template.pop("raw_css", [])]
    built = pn.template.FastListTemplate(
        title=title, sidebar=sidebar, main=main, sidebar_width=SIDEBAR_WIDTH, raw_css=raw_css, **template,
    )
    built.config.js_files = {**built.config.js_files, "vaft_phone_start": _data_uri(PHONE_START_JS)}
    return built
