"""Responsive layout of ``vaft gui`` (#1865): one page for phones, tablets and desktops."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from vaft.gui import layout

pn = pytest.importorskip("panel")

from vaft.gui import app as gui_app  # noqa: E402


# -- the rules, without a browser ------------------------------------------------------
def test_the_phone_rules_and_the_phone_script_name_the_same_screens():
    assert f"max-width: {layout.PHONE_MAX_WIDTH}px" in layout.RESPONSIVE_CSS
    assert f"max-width: {layout.TABLET_MAX_WIDTH}px" in layout.RESPONSIVE_CSS
    assert layout.PHONE_QUERY in layout.RESPONSIVE_CSS, "the drawer CSS and the closing script agree"
    assert layout.PHONE_QUERY in layout.PHONE_START_JS and "closeNav" in layout.PHONE_START_JS


def test_page_carries_the_rules_the_script_and_the_branding():
    built = layout.page([pn.pane.Markdown("side")], [pn.pane.Markdown("main")], logo="x.png", raw_css=["/* mine */"])
    assert built.config.raw_css[0] == layout.RESPONSIVE_CSS and "/* mine */" in built.config.raw_css
    assert built.config.js_files["vaft_phone_start"].startswith("data:text/javascript;base64,")
    assert built.sidebar_width == layout.SIDEBAR_WIDTH and built.logo == "x.png"


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


def test_every_served_page_is_the_responsive_one(ods, monkeypatch):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    shell = gui_app.build_shell(workspace="database")
    try:
        view = shell.view()
        assert layout.RESPONSIVE_CSS in view.config.raw_css and view.logo == gui_app.BRANDING["logo"]
        explorer = shell.workspaces["plots"].app
        assert layout.RESPONSIVE_CSS in explorer.view().config.raw_css, "the explorer page alone too"
        table = shell.workspaces["database"].table
        assert any("overflow-x: auto" in str(sheet) for sheet in table.stylesheets), "wide tables scroll in place"
    finally:
        shell.close()


# -- in a real browser (skipped where no headless Chromium is installed) --------------------
def _chrome() -> str | None:
    for candidate in (
        os.environ.get("VAFT_GUI_CHROME"),
        *sorted(Path.home().glob(
            "Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"
        )),
        *sorted(Path.home().glob(".cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell")),
        shutil.which("chromium"), shutil.which("google-chrome"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


_PROBE = (
    '<script>setTimeout(function(){document.body.setAttribute("data-probe",'
    'document.getElementById("sidebar").className)},5000)</script>'
)


@pytest.mark.skipif(_chrome() is None, reason="no headless Chromium (set VAFT_GUI_CHROME)")
@pytest.mark.parametrize(("size", "closed"), [("375,812", True), ("812,375", True), ("768,1024", False), ("1280,900", False)])
def test_a_phone_opens_on_the_figure_and_a_desktop_on_the_controls(ods, monkeypatch, tmp_path, size, closed):
    monkeypatch.setattr("vaft.gui.state.load_source", lambda source: ods)
    monkeypatch.setattr(pn.state, "onload", lambda callback: callback())
    shell = gui_app.build_shell(sample=39915, plot="plasma_current_time")
    try:
        path = tmp_path / "page.html"
        shell.view().save(str(path), resources="inline")
    finally:
        shell.close()
    path.write_text(path.read_text().replace("</body>", _PROBE + "</body>"))
    dom = subprocess.run(
        [_chrome(), "--headless", "--disable-gpu", f"--window-size={size}", "--virtual-time-budget=8000",
         "--dump-dom", path.as_uri()],
        capture_output=True, text=True, timeout=180,
    ).stdout
    assert 'data-probe="' in dom, "the page did not finish loading"
    state = dom.split('data-probe="', 1)[1].split('"', 1)[0]
    assert ("hidden" in state) is closed, f"{size}: sidebar class {state!r}"
