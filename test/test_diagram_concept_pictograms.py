"""Semantic pictograms across the core VAFT concept diagrams.

The shared registry and its placement helpers are tested in ``test_diagram_pictograms.py``;
this module checks how four builders use them: every mapped concept gets its symbol, the same
concept draws the same way in both infrastructure variants, and pictograms only ever add
``symbol:`` polylines -- topology, text, roles and model are what the builder draws without them.
"""

import pytest

import vaft.diagram
from vaft.diagram import _research_concepts as R
from vaft.diagram import _vaft_concepts as V
from vaft.diagram import pictograms as P
from vaft.diagram._scene import Label, Polyline, Scene

#: (builder, kwargs, module, {annotated role: semantic name})
CASES = (
    ("vaft_four_pillars", {}, V, {f"pillar:{k}": v for k, v in V.PILLAR_SYMBOLS.items()}),
    ("human_ai_interface", {}, V, {(k if k.startswith("actor:") else f"interface:{k}"): v
                                    for k, v in V.INTERFACE_SYMBOLS.items()}),
    ("fusion_science_knowledge_lifecycle", {}, V, {f"node:{k}": v for k, v in V.LIFECYCLE_SYMBOLS.items()}),
    ("experimental_research_infrastructure", {"organization": "fragmented"}, R,
     {f"source:{s}": R.INFRA_SOURCE_SYMBOLS[s] for _, s, *_ in R.INFRA_LANES}),
    ("experimental_research_infrastructure", {"organization": "integrated"}, R,
     {f"source:{k}": v for k, v in R.INFRA_SOURCE_SYMBOLS.items()}),
)
IDS = [f"{name}-{kw.get('organization', '')}".rstrip("-") for name, kw, *_ in CASES]


def _icon(items):
    """A drawn pictogram's own strokes, without its badge disc, in drawing order."""
    return [i for i in items if i.style not in (P.BADGE, P.BADGE_MUTED)]


def _shape(items):
    """Stroke geometry relative to its own first point: the same symbol anywhere compares equal."""
    x0, y0 = items[0].points[0]
    return tuple(tuple((round(x - x0, 6), round(y - y0, 6)) for x, y in i.points) for i in items)


@pytest.mark.parametrize("name, kwargs, module, mapping", CASES, ids=IDS)
def test_every_mapped_concept_draws_its_symbol(name, kwargs, module, mapping):
    scene = getattr(vaft.diagram, name)(**kwargs).scene
    for role, symbol in mapping.items():
        assert scene.role(role), f"{role} is not drawn"
        icon = _icon(scene.role(f"symbol:{role}"))
        assert len(icon) == len(P.symbol(symbol).strokes), role
        assert all(isinstance(i, Polyline) for i in scene.role(f"symbol:{role}"))
    drawn = {i.role[len("symbol:"):] for i in scene.items if i.role.startswith("symbol:")}
    assert drawn == set(mapping)  # nothing beyond the mapping: no icon on every box for symmetry


@pytest.mark.parametrize("name, kwargs, module, mapping", CASES, ids=IDS)
@pytest.mark.parametrize("labels", [True, False])
def test_pictograms_only_add_symbol_items(monkeypatch, name, kwargs, module, mapping, labels):
    full = getattr(vaft.diagram, name)(**kwargs, labels=labels)
    for helper in ("badge", "pictogram"):
        if hasattr(module, helper):
            monkeypatch.setattr(module, helper, lambda *a, **k: Scene(()))
    bare = getattr(vaft.diagram, name)(**kwargs, labels=labels)
    assert [i for i in full.scene.items if not i.role.startswith("symbol:")] == list(bare.scene.items)
    assert full.model == bare.model
    assert not [i for i in full.scene.items if i.role.startswith("symbol:") and isinstance(i, Label)]


def test_both_infrastructure_variants_draw_a_source_identically():
    frag = vaft.diagram.experimental_research_infrastructure("fragmented").scene
    integ = vaft.diagram.experimental_research_infrastructure("integrated").scene
    shared = [s for _, s, *_ in R.INFRA_LANES]
    for source in shared:
        a, b = _icon(frag.role(f"symbol:source:{source}")), _icon(integ.role(f"symbol:source:{source}"))
        # the integrated badge draws the icon smaller: compare after normalising the scale
        scale = (max(p[0] for i in a for p in i.points) - min(p[0] for i in a for p in i.points)) / (
            max(p[0] for i in b for p in i.points) - min(p[0] for i in b for p in i.points))
        scaled = [Polyline(tuple((x * scale, y * scale) for x, y in i.points), i.style, i.role, i.closed) for i in b]
        flat = lambda items: [c for stroke in _shape(items) for point in stroke for c in point]  # noqa: E731
        assert flat(a) == pytest.approx(flat(scaled), abs=1e-5), source
    # calibration is a source only in the integrated figure (a private copy in the fragmented one)
    assert "calibration" not in shared and integ.role("symbol:source:calibration")


def test_lifecycle_stages_are_visually_distinct():
    assert len(set(V.LIFECYCLE_SYMBOLS.values())) == len(V.LIFECYCLE) == 7
    assert list(V.LIFECYCLE_SYMBOLS) == [k for k, *_ in V.LIFECYCLE]


def test_a_planned_interface_keeps_a_muted_symbol():
    scene = vaft.diagram.human_ai_interface().scene
    planned = {k for k, *_, p in V.INTERFACES if p}
    assert planned == {"mcp"}
    for key, *_ in V.INTERFACES:
        disc, *icon = scene.role(f"symbol:interface:{key}")
        muted = key in planned
        assert disc.style == (P.BADGE_MUTED if muted else P.BADGE), key
        assert {i.style for i in icon} == {P.PICTOGRAM_MUTED if muted else P.PICTOGRAM}, key


def test_one_symbol_per_concept_across_the_rollout():
    """A semantic name means one concept: no two different concepts of this rollout share a symbol."""
    concepts = {}
    for mapping in (V.PILLAR_SYMBOLS, V.INTERFACE_SYMBOLS, V.LIFECYCLE_SYMBOLS, R.INFRA_SOURCE_SYMBOLS):
        for key, symbol in mapping.items():
            concepts.setdefault(symbol, set()).add(key.split(":")[-1])
    # names shared only where the registry meaning covers both: stores of data, raw measured signals
    shared = {s: keys for s, keys in concepts.items() if len(keys) > 1}
    assert shared == {"database": {"repository", "machine_db"}, "signal": {"daq", "raw"}}


def test_pillar_text_starts_below_a_common_icon_row():
    scene = vaft.diagram.vaft_four_pillars().scene
    tops = set()
    for key in V.PILLAR_SYMBOLS:
        outline, text = scene.role(f"pillar:{key}")
        icon = scene.role(f"symbol:pillar:{key}")
        icon_bottom = min(p[1] for i in icon for p in i.points)
        assert text.anchor == "north" and text.at[1] < icon_bottom
        assert max(p[1] for p in outline.points) > max(p[1] for i in icon for p in i.points)
        tops.add(round(text.at[1], 6))
    assert len(tops) == 1
