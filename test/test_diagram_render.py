"""SVG rendering, notebook display and freshness of the committed assets (#890)."""

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import vaft.diagram
from vaft.diagram import _render, build

HAS_TEX = all(shutil.which(tool) for tool in ("latex", "dvisvgm"))
needs_tex = pytest.mark.skipif(not HAS_TEX, reason="latex and dvisvgm are not installed")
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "diagrams"


def _modules_after(statement):
    code = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_importing_the_namespace_loads_no_renderer_or_plotting_stack():
    loaded = _modules_after("import vaft, vaft.diagram")
    assert "matplotlib" not in loaded
    assert not {"vaft.diagram._render", "vaft.diagram._magnetic_island", "vaft.formula.catalog"} & loaded


def test_vaft_exposes_diagram_lazily():
    import vaft

    assert "diagram" in vaft.__all__
    assert "vaft.diagram" not in _modules_after("import vaft")


def test_the_tikz_source_needs_no_tex(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    d = vaft.diagram.magnetic_island()
    assert d.tikz.startswith("% VAFT diagram template")
    assert "\\begin{tikzpicture}" in d.tikz and "%%VAFT-BODY%%" not in d.tikz


def test_rendering_without_the_toolchain_says_what_is_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(vaft.diagram.DiagramToolchainError, match="VAFT diagram build toolchain"):
        vaft.diagram.magnetic_island().svg


def test_without_the_toolchain_a_canonical_diagram_is_served_from_its_committed_asset(monkeypatch):
    """The tutorials run offline with no TeX: a diagram built with the exact
    arguments of a committed asset displays that asset, byte for byte."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    d = vaft.diagram.rational_surface()
    assert d.svg == (ASSETS / "rational_surface.svg").read_text(encoding="utf-8")


def test_without_the_toolchain_a_non_canonical_diagram_still_refuses(monkeypatch):
    """A nearby picture is not the same picture: other arguments, other source."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(vaft.diagram.DiagramToolchainError):
        vaft.diagram.rational_surface(m=3, n=2).svg


def test_without_the_toolchain_an_edited_committed_asset_is_not_served(monkeypatch, tmp_path):
    """The manifest's SVG hash guards the file itself: a stale or hand-edited
    asset beside a correct manifest entry is refused, not displayed."""
    import shutil as _shutil

    for name in ("manifest.json", "rational_surface.svg"):
        _shutil.copy(ASSETS / name, tmp_path / name)
    (tmp_path / "rational_surface.svg").write_text("<svg>edited</svg>\n", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(_render, "committed_assets_dir", lambda: tmp_path)
    with pytest.raises(vaft.diagram.DiagramToolchainError):
        vaft.diagram.rational_surface().svg


def test_without_a_source_checkout_there_is_no_committed_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(_render, "committed_assets_dir", lambda: tmp_path)
    with pytest.raises(vaft.diagram.DiagramToolchainError):
        vaft.diagram.rational_surface().svg


def test_save_refuses_an_unknown_format(tmp_path):
    with pytest.raises(ValueError, match=".svg"):
        vaft.diagram.magnetic_island().save(tmp_path / "island.png")


def test_the_template_ships_with_the_package():
    text = _render.template()
    assert text.count("\n%%VAFT-BODY%%\n") == 1


def test_no_template_style_is_defined_twice():
    # a second definition silently replaces the first: a new family's "region"
    # once shrank every stability chart's region labels
    names = re.findall(r"^\s*([\w ]+)/\.style=", _render.template(), re.M)
    assert sorted({n for n in names if names.count(n) > 1}) == []


def test_the_committed_reference_diagrams_are_fresh():
    assert build.check(ASSETS) == []


def test_check_catches_a_hand_edited_or_orphaned_asset(tmp_path):
    for item in ASSETS.iterdir():
        shutil.copy(item, tmp_path / item.name)
    target = tmp_path / "magnetic_island_top.svg"
    target.write_text(target.read_text().replace("</svg>", "<!-- edited --></svg>"))
    (tmp_path / "stray.svg").write_text("<svg/>")
    problems = build.check(tmp_path)
    assert any("magnetic_island_top.svg" in p and "hand" in p for p in problems)
    assert any("stray.svg" in p for p in problems)


@needs_tex
@pytest.mark.parametrize("projection", ["poloidal", "top", "3d"])
def test_rendered_svg_is_valid_and_self_contained(projection):
    d = vaft.diagram.magnetic_island(projection=projection)
    svg = d._repr_svg_()
    root = ET.fromstring(svg.encode())
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    hrefs = re.findall(r"href=['\"]([^'\"]*)['\"]", svg)
    assert all(h.startswith("#") for h in hrefs), "external reference in the SVG"
    assert "<image" not in svg and "<!--" not in svg
    for marker in ("/tmp", "/private/", "/Users/", "/home/", "vaft-diagram-", "dvisvgm"):
        assert marker not in svg


@needs_tex
def test_rendering_is_deterministic_and_saves(tmp_path):
    first = vaft.diagram.magnetic_island(projection="top").save(tmp_path / "a.svg").read_text()
    second = vaft.diagram.magnetic_island(projection="top").svg
    assert first == second


def _copy_assets(tmp_path):
    for item in ASSETS.iterdir():
        shutil.copy(item, tmp_path / item.name)
    return tmp_path


def test_check_catches_a_stale_or_missing_asset(tmp_path, monkeypatch):
    _copy_assets(tmp_path)
    (tmp_path / "magnetic_island_3d.svg").unlink()
    problems = build.check(tmp_path)
    assert any("magnetic_island_3d.svg" in p and "missing" in p for p in problems)
    # a change to the render recipe makes every asset stale
    monkeypatch.setattr(_render, "RENDER_RECIPE", _render.RENDER_RECIPE + " changed")
    problems = build.check(tmp_path)
    assert sum("stale" in p for p in problems) == len(build.CANONICAL) - 1  # all but the missing one


def test_the_committed_assets_are_checked_out_with_lf_everywhere():
    out = subprocess.run(
        ["git", "check-attr", "eol", "--", "docs/assets/diagrams/manifest.json",
         "docs/assets/diagrams/magnetic_island_poloidal.svg"],
        cwd=ASSETS.parents[2], capture_output=True, text=True,
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    assert out.stdout.count("eol: lf") == 2, out.stdout


def test_check_reports_a_raising_builder_and_goes_on(tmp_path, monkeypatch):
    """One builder raising aborted the whole freshness report with a traceback; it is now one problem line and
    the other assets are still checked. Cold review 0.8.0 diagram-A F4."""
    _copy_assets(tmp_path)
    monkeypatch.setitem(build.CANONICAL, "magnetic_island_top.svg", ("magnetic_island", {"projection": "nope"}))
    problems = build.check(tmp_path)
    assert problems == [p for p in problems if "magnetic_island_top.svg" in p and "builder raised ValueError" in p]
    assert len(problems) == 1


def test_the_tex_pipe_is_decoded_as_utf8_with_replacement(monkeypatch, tmp_path):
    """``text=True`` alone decodes with the locale codec and strict errors, so a stray byte in a TeX message
    raised UnicodeDecodeError instead of the render's own report. Cold review 0.8.0 diagram-A F5."""
    seen = {}

    class _Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _Done()

    monkeypatch.setattr(_render.subprocess, "run", fake_run)
    _render._run(["latex"], tmp_path, "latex")
    assert seen["encoding"] == "utf-8" and seen["errors"] == "replace" and seen["text"] is True
