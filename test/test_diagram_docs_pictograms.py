"""Semantic pictograms in the documentation-facing concept diagrams (#2113).

The registry and placement helpers are tested in ``test_diagram_pictograms.py`` (#2101). Here: each
of the four figures draws exactly its mapped symbols, pictograms only add ``symbol:`` polylines,
the scientific content stays as it was (``full`` ecosystem untouched, no icon on quality metadata,
the lifecycle still from ``vaft._ecosystem``), and the documents that embed the figures still do.
"""

from pathlib import Path

import pytest

import vaft.diagram
from vaft import _ecosystem as ecosystem
from vaft.diagram import _ecosystem as E
from vaft.diagram import _research_concepts as R
from vaft.diagram import _vaft_concepts as V
from vaft.diagram import pictograms as P
from vaft.diagram._scene import Label, Polyline, Scene

ROOT = Path(__file__).resolve().parents[1]

#: (builder, kwargs, module, {annotated role: semantic name})
CASES = (
    ("fusion_research_ecosystem", {"detail": "presentation"}, R,
     {f"role:{k}": v for k, v in R.PRESENTATION_ROLE_SYMBOLS.items()}),
    ("scientific_provenance_chain", {}, V, {f"step:{k}": v for k, v in V.PROVENANCE_SYMBOLS.items()}),
    ("external_code_integration", {}, E, {f"lifecycle:{k}": v for k, v in E.INTEGRATION_SYMBOLS.items()}),
    ("machine_research_archive", {}, V, {f"track:{k}": v for k, v in V.ARCHIVE_SYMBOLS.items()}),
)
IDS = [name for name, *_ in CASES]


@pytest.mark.parametrize("name, kwargs, module, mapping", CASES, ids=IDS)
def test_every_mapped_concept_draws_its_symbol_and_nothing_else(name, kwargs, module, mapping):
    scene = getattr(vaft.diagram, name)(**kwargs).scene
    for role, symbol in mapping.items():
        assert scene.role(role), f"{role} is not drawn"
        drawn = scene.role(f"symbol:{role}")
        assert all(isinstance(i, Polyline) for i in drawn)
        assert len([i for i in drawn if i.style != P.BADGE]) == len(P.symbol(symbol).strokes), role
    assert {i.role[len("symbol:"):] for i in scene.items if i.role.startswith("symbol:")} == set(mapping)


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


def test_the_full_ecosystem_draws_no_pictogram():
    scene = vaft.diagram.fusion_research_ecosystem(detail="full").scene
    assert not [i for i in scene.items if i.role.startswith("symbol:")]


def test_provenance_keeps_quality_and_versioning_unmarked():
    """Provenance is not validation: quality and versioned metadata carry no product symbol."""
    scene = vaft.diagram.scientific_provenance_chain().scene
    assert not scene.role("symbol:quality") and not scene.role("symbol:versioned")
    assert list(V.PROVENANCE_SYMBOLS) == [k for k, *_ in V.PROVENANCE]


def test_integration_symbols_follow_the_lifecycle_registry():
    keys = [k for k, _ in ecosystem.INTEGRATION_LIFECYCLE]
    for key in E.INTEGRATION_SYMBOLS:
        assert key.split(":")[0] in keys, key
    model = vaft.diagram.external_code_integration().model
    assert model["lifecycle"] == tuple(keys)
    # local and Slurm execution are told apart; upstream and VAFT-owned steps keep their own symbols
    assert E.INTEGRATION_SYMBOLS["execute:local"] != E.INTEGRATION_SYMBOLS["execute:slurm"]
    assert len(set(E.INTEGRATION_SYMBOLS.values())) == len(E.INTEGRATION_SYMBOLS)


def test_integration_text_keeps_its_width():
    """The icon column is added beside the text, so no lifecycle label rewraps."""
    scene = vaft.diagram.external_code_integration().scene
    for key, _ in ecosystem.INTEGRATION_LIFECYCLE:
        roles = ("lifecycle:execute:local", "lifecycle:execute:slurm") if key == "execute" else (f"lifecycle:{key}",)
        for role in roles:
            (label,) = [i for i in scene.role(role) if isinstance(i, Label)]
            assert label.style.endswith("text width=3.50cm" if key == "execute" else "text width=7.70cm"), role


def test_one_symbol_per_track_and_none_per_entry():
    scene = vaft.diagram.machine_research_archive().scene
    assert not [i for i in scene.items if i.role.startswith("symbol:entry:")]
    assert len(set(V.ARCHIVE_SYMBOLS.values())) == len(V.ARCHIVE_TRACKS)


@pytest.mark.parametrize("document, asset", [
    ("README.md", "fusion_research_ecosystem_presentation.svg"),
    ("README.ko.md", "fusion_research_ecosystem_presentation.svg"),
    ("docs/_guide/Pipelines.md", "scientific_provenance_chain.svg"),
    ("docs/_guide/External_codes.md", "external_code_integration.svg"),
    ("install/README.md", "external_code_integration.svg"),
    ("docs/_pages/about.md", "machine_research_archive.svg"),
])
def test_the_documents_still_embed_the_one_canonical_asset(document, asset):
    assert asset in (ROOT / document).read_text(encoding="utf-8")
    assert (ROOT / "docs" / "assets" / "diagrams" / asset).is_file()
