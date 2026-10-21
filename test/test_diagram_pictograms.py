"""Semantic vector pictograms (#2101): registry, restricted SVG conversion, placement, renderers, provenance."""

import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

import vaft.diagram
from vaft.diagram import _animation, _lucide, _platform, _vaft_concepts, build
from vaft.diagram import pictograms as P
from vaft.diagram._render import Diagram, tikz_document
from vaft.diagram._scene import Image, Label, Polyline, Scene

ROOT = Path(__file__).resolve().parents[1]


def _bounds(points):
    xs, ys = zip(*points)
    return min(xs), max(xs), min(ys), max(ys)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_names_are_stable_identifiers():
    assert P.names() == tuple(P.SYMBOLS)
    assert len(set(P.names())) == len(P.names())
    assert all(re.fullmatch(r"[a-z][a-z_]*", name) for name in P.names())
    assert all(P.symbol(name).meaning for name in P.names())


def test_unknown_name_is_rejected_with_the_valid_names():
    with pytest.raises(KeyError, match="registered: experiment"):
        P.symbol("tokamac")
    with pytest.raises(KeyError):
        P.pictogram("tokamac", (0, 0), role="symbol:x")


def test_every_source_is_traceable():
    for sym in P.SYMBOLS.values():
        if sym.source == "vaft":
            assert sym.name in P._VAFT
        else:
            icon, version = re.fullmatch(r"lucide:([a-z0-9-]+)@(.+)", sym.source).groups()
            assert icon in _lucide.ICONS and version == _lucide.LUCIDE_VERSION


def test_distinct_symbols_never_share_geometry():
    """Two semantic names drawn identically would merge distinct concepts in a reader's eye."""
    geometry = [sym.strokes for sym in P.SYMBOLS.values()]
    assert len(set(geometry)) == len(geometry)


# --------------------------------------------------------------------------
# vendored Lucide: pinned, unmodified, licensed, nothing unused
# --------------------------------------------------------------------------


def test_vendored_markup_is_unmodified():
    for name, (markup, digest) in _lucide.ICONS.items():
        assert hashlib.sha256(markup.encode()).hexdigest() == digest, name


def test_vendored_release_and_licence_are_recorded():
    assert re.fullmatch(r"\d+\.\d+\.\d+", _lucide.LUCIDE_VERSION)
    assert _lucide.LUCIDE_VERSION in _lucide.LUCIDE_SOURCE
    assert re.fullmatch(r"[0-9a-f]{40}", _lucide.LUCIDE_TARBALL_SHA1)
    assert "ISC License" in _lucide.LUCIDE_LICENSE and "Cole Bemis" in _lucide.LUCIDE_LICENSE
    for notices in ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.ko.md"):
        text = (ROOT / notices).read_text(encoding="utf-8")
        assert "Lucide" in text and _lucide.LUCIDE_VERSION in text


def test_every_vendored_icon_is_registered():
    used = {sym.source.split(":")[1].split("@")[0] for sym in P.SYMBOLS.values() if sym.source != "vaft"}
    assert used == set(_lucide.ICONS)


# --------------------------------------------------------------------------
# restricted SVG conversion
# --------------------------------------------------------------------------


def test_geometry_is_normalised_into_the_unit_box():
    for sym in P.SYMBOLS.values():
        for points, _closed in sym.strokes:
            x0, x1, y0, y1 = _bounds(points)
            assert -0.5 <= x0 and x1 <= 0.5 and -0.5 <= y0 and y1 <= 0.5, sym.name


@pytest.mark.parametrize("markup", ['<circle cx="12" cy="12" r="9"/>', '<path d="M3 12a9 9 0 0 0 18 0a9 9 0 0 0-18 0"/>',
                                    '<ellipse cx="12" cy="12" rx="9" ry="9"/>'])
def test_curves_are_flattened_within_the_tolerance(markup):
    for points, _closed in P._strokes(markup):
        for a, b in zip(points, points[1:]):
            assert abs(math.dist(a, (12, 12)) - 9) < 1e-9  # vertices on the circle
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            assert 9 - math.dist(mid, (12, 12)) <= P.TOLERANCE + 1e-12  # chords within the bound


def test_cubic_flattening_stays_within_the_tolerance():
    (points, _), = P._strokes('<path d="M2 20C2 2 22 2 22 20"/>')
    # sample the true curve densely and check each sample is near the polyline
    def bezier(t):
        p = ((2, 20), (2, 2), (22, 2), (22, 20))
        c = ((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3)
        return sum(c[i] * p[i][0] for i in range(4)), sum(c[i] * p[i][1] for i in range(4))

    def seg_dist(q, a, b):
        ax, ay, bx, by = *a, *b
        t = max(0.0, min(1.0, ((q[0] - ax) * (bx - ax) + (q[1] - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
        return math.dist(q, (ax + t * (bx - ax), ay + t * (by - ay)))

    for k in range(201):
        q = bezier(k / 200)
        assert min(seg_dist(q, a, b) for a, b in zip(points, points[1:])) <= P.TOLERANCE + 1e-9


def test_relative_commands_compact_arc_flags_and_close_then_continue():
    # book-open writes arc flags without separators ("0 00-"); scale continues after "z"
    strokes = P._strokes('<path d="M2 5a2 2 0 012-2h4zV9"/>')
    (first, closed), (second, open_) = strokes
    assert closed and not open_
    assert first[0] == (2, 5) and first[-1] == (8, 3)
    assert second == [(2, 5), (2, 9)]


@pytest.mark.parametrize("markup", ['<text x="1">a</text>', '<path d="M2 2T4 4"/>', '<g/>', "<polygon points='1 2'/>"])
def test_unsupported_svg_is_refused(markup):
    with pytest.raises(ValueError):
        P._strokes(markup)


def test_a_zero_length_stroke_becomes_a_ring():
    (dot,) = P._normalise(P._strokes('<path d="M12 12h.01"/>'))
    points, closed = dot
    assert closed and len(points) >= 8


# --------------------------------------------------------------------------
# placement and roles
# --------------------------------------------------------------------------


def test_placement_translates_and_scales_uniformly():
    unit = P.pictogram("database", (0.0, 0.0), 1.0, role="symbol:a")
    placed = P.pictogram("database", (2.0, 3.0), 0.8, role="symbol:a")
    assert placed == unit.transformed(scale=0.8, offset=(2.0, 3.0))
    for item in placed.items:
        x0, x1, y0, y1 = _bounds(item.points)
        assert 1.6 <= x0 and x1 <= 2.4 and 2.6 <= y0 and y1 <= 3.4


def test_items_are_polylines_with_the_role_and_style():
    scene = P.pictogram("researcher", (0, 0), role="symbol:actor:human", style=P.PICTOGRAM_MUTED)
    assert scene.items and all(isinstance(i, Polyline) for i in scene.items)
    assert {i.role for i in scene.items} == {"symbol:actor:human"}
    assert {i.style for i in scene.items} == {P.PICTOGRAM_MUTED}
    assert scene.role("symbol:actor:human") == scene.items


@pytest.mark.parametrize("role", ["node:x", "", "edge:a->b"])
def test_roles_outside_the_symbol_namespace_are_refused(role):
    with pytest.raises(ValueError, match="symbol:"):
        P.pictogram("database", (0, 0), role=role)


def test_size_must_be_positive():
    with pytest.raises(ValueError):
        P.pictogram("database", (0, 0), 0.0, role="symbol:x")


def test_generation_is_deterministic():
    assert P.pictogram("history", (1, 2), role="symbol:x") == P.pictogram("history", (1, 2), role="symbol:x")
    assert P._build() == P.SYMBOLS
    # and the same across processes (no hash-seed or ordering dependence)
    code = "import vaft.diagram.pictograms as p, hashlib; print(hashlib.sha256(repr(p.SYMBOLS).encode()).hexdigest())"
    digests = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**__import__("os").environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2")}
    assert len(digests) == 1


def test_badge_is_a_disc_under_the_icon_and_corner_clips_it_on():
    from vaft.diagram._concept import box

    b = box(0.0, 0.0, 4.0, 1.0, "x")
    at = P.corner(b)
    assert at == pytest.approx((-1.5, 0.66))
    assert P.corner(b, 1.0)[1] - 0.5 == pytest.approx(0.5 - 0.15)  # the disc dips 0.15 cm into the box
    disc, *icon = P.badge("signal", at, 0.62, role="symbol:node:x").items
    assert disc.style == P.BADGE and disc.closed
    assert icon and all(i.style == P.PICTOGRAM for i in icon)
    assert max(math.dist(p, at) for i in icon for p in i.points) < 0.31  # the icon sits inside its disc
    muted_disc, *muted = P.badge("integration", at, 0.62, role="symbol:x", style=P.PICTOGRAM_MUTED).items
    assert muted_disc.style == P.BADGE_MUTED and {i.style for i in muted} == {P.PICTOGRAM_MUTED}


# --------------------------------------------------------------------------
# renderers and hashing
# --------------------------------------------------------------------------


def test_tikz_is_vector_paths_without_external_assets():
    tikz = tikz_document(P.pictogram_catalog().scene)
    assert "\\draw[draw=islanddark,line width=0.7pt]" in tikz
    assert "vaftimage" not in tikz.split("\\begin{document}")[1] and "http" not in tikz.split("\\begin{document}")[1]


def test_a_symbol_change_restales_only_the_diagrams_that_draw_it(monkeypatch):
    workflow, gallery = vaft.diagram.scientific_workflow(), P.pictogram_catalog()
    island = vaft.diagram.magnetic_island()
    moved = P.Symbol("validation", "x", "vaft", (((((0.0, 0.0), (0.1, 0.1))), False),))
    monkeypatch.setitem(P.SYMBOLS, "validation", moved)
    assert vaft.diagram.scientific_workflow().source_sha256 != workflow.source_sha256
    assert P.pictogram_catalog().source_sha256 != gallery.source_sha256
    assert vaft.diagram.magnetic_island().source_sha256 == island.source_sha256
    # a symbol the demonstration does not use leaves it untouched
    monkeypatch.setitem(P.SYMBOLS, "validation", P._build()["validation"])
    monkeypatch.setitem(P.SYMBOLS, "discovery", moved)
    assert vaft.diagram.scientific_workflow().source_sha256 == workflow.source_sha256


def test_the_animation_renderer_draws_the_pictogram_styles():
    assert _animation._LINE[P.PICTOGRAM]["color"] == _animation._COLOURS["islanddark"]
    assert _animation._LINE[P.PICTOGRAM_MUTED]["ls"] == "--"
    assert _animation._FILL[P.BADGE]["facecolor"] == "white"
    assert _animation._FILL[P.BADGE_MUTED]["ls"] == "--"
    figure, ax = _animation.draw_scene(P.badge("database", (0, 0), 1.0, role="symbol:x"))
    lines = ax.get_lines()
    assert lines and all(line.get_color() == _animation._COLOURS["islanddark"] for line in lines)
    assert len(ax.patches) == 1 and ax.patches[0].get_zorder() < min(line.get_zorder() for line in lines)


def test_importing_the_registry_needs_no_renderer_or_network():
    code = ("import socket, sys; socket.socket = None; import vaft.diagram.pictograms as p; "
            "p.pictogram('database', (0, 0), role='symbol:x'); "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith(('matplotlib', 'vaft.diagram._render')))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# --------------------------------------------------------------------------
# the gallery and the two demonstrations
# --------------------------------------------------------------------------


def test_gallery_draws_every_symbol_and_is_canonical():
    scene = P.pictogram_catalog().scene
    for name in P.names():
        assert scene.role(f"symbol:catalog:{name}")
        assert scene.role(f"catalog:{name}")
    assert not [i for i in P.pictogram_catalog(labels=False).scene.items if isinstance(i, Label)]
    assert build.CANONICAL["pictogram_catalog.svg"] == ("pictogram_catalog", {})
    assert vaft.diagram.pictogram_catalog is P.pictogram_catalog


@pytest.mark.parametrize("name, modules, symbols", [
    ("scientific_workflow", (_vaft_concepts,), {"group:sources": "tokamak", **{
        f"node:{k}": v for k, v in _vaft_concepts.WORKFLOW_SYMBOLS.items()}}),
    ("vest_data_platform_overview", (_platform,), {f"stage:{k}": v for k, v in _platform.OVERVIEW_SYMBOLS.items()}),
])
@pytest.mark.parametrize("labels", [True, False])
def test_demonstrations_only_add_symbols(monkeypatch, name, modules, symbols, labels):
    full = getattr(vaft.diagram, name)(labels=labels)
    items = full.scene.items
    for role, sym in symbols.items():
        drawn = full.scene.role(f"symbol:{role}")
        assert drawn, role
        icon = [i for i in drawn if i.style != P.BADGE]
        assert len(icon) == len(P.symbol(sym).strokes)
        assert any(i.role == role for i in items if not i.role.startswith("symbol:"))  # beside an existing item
    plain = [i for i in items if not i.role.startswith("symbol:")]
    for module in modules:
        for helper in ("badge", "pictogram"):
            if hasattr(module, helper):
                monkeypatch.setattr(module, helper, lambda *a, **k: Scene(()))
    bare = getattr(vaft.diagram, name)(labels=labels)
    assert plain == list(bare.scene.items)
    assert full.model == bare.model


def test_the_overview_still_draws_only_the_machine_picture():
    images = [i for i in vaft.diagram.vest_data_platform_overview().scene.items if isinstance(i, Image)]
    assert [i.name for i in images] == [_platform.MACHINE_IMAGE]


def test_pictogram_roles_never_shadow_topology_roles():
    for name in ("scientific_workflow", "vest_data_platform_overview"):
        diagram = getattr(vaft.diagram, name)()
        symbols = [i for i in diagram.scene.items if i.role.startswith("symbol:")]
        assert symbols and all(isinstance(i, Polyline) for i in symbols)  # never a Label, a node or an edge
        roles = {i.role for i in diagram.scene.items if not i.role.startswith("symbol:")}
        # each pictogram annotates an item that exists, under its own namespace
        assert {i.role[len("symbol:"):] for i in symbols} <= roles


def test_the_symbol_helpers_are_not_gallery_builders():
    from vaft.diagram import docs_catalog

    builders = docs_catalog.builder_names()
    assert "pictogram_catalog" in builders
    assert not {"pictogram", "badge", "corner", "symbol", "names"} & set(builders)


def test_documentation_lists_every_symbol():
    text = (ROOT / "docs" / "_guide" / "Diagrams.md").read_text(encoding="utf-8")
    section = text.split("## Semantic pictograms", 1)[1].split("\n## ", 1)[0]
    for name in P.names():
        assert f"`{name}`" in section, name
    assert "pictogram_catalog.svg" in section
