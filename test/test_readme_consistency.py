"""Keep the two READMEs, and the identity they state, from drifting apart.

`README.ko.md` was a faithful translation that fell four sections and one dead
API behind its English counterpart. Nothing caught that, because nothing
compared them. These tests pin the parts of #330's reframing that a reader would
notice if they rotted: the positioning statement, the four infrastructure
concepts, the order they appear in, and the promise that long-term ambitions are
not presented as shipped features.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "README.md"
KOREAN = ROOT / "README.ko.md"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
NOTICES_KO = ROOT / "THIRD_PARTY_NOTICES.ko.md"

#: The identity narrative #330 prescribes, in order. Each README states it in
#: its own language, so they are matched by position rather than by text.
IDENTITY_SECTIONS = 3

#: Capabilities that must not be described as current functionality (#330 §4).
LONG_TERM_TERMS = (
    "knowledge graph",
    "digital twin",
    "autonomous research",
    "scientific agent",
)


def headings(path: Path, level: str = "## ") -> list[str]:
    """Return the document's headings at one level, in order.

    Fenced code blocks are skipped: several samples contain shell comments that
    begin with `#`.
    """
    found: list[str] = []
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith(level) and not line.startswith(level + "#"):
            found.append(line[len(level):].strip())
    return found


# ---------------------------------------------------------------------------
# The identity both files must state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_the_positioning_statement_leads(path):
    """A reader must learn what VAFT is before anything else (#330 acceptance)."""
    text = path.read_text(encoding="utf-8")
    opening = text[: text.index("\n## ")]
    for phrase in ("VAFT", "VEST"):
        assert phrase in opening
    # The claim that distinguishes the new framing from "a Python library".
    assert "infrastructure" in opening or "인프라" in opening, (
        f"{path.name}: the opening does not state that VAFT is infrastructure"
    )


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_the_four_infrastructure_concepts_are_present_and_ordered(path):
    """All four, in #330's order. They are the shared conceptual vocabulary."""
    text = path.read_text(encoding="utf-8")
    positions = []
    for concept in (
        "Integrated Standardized Interface",
        "Version-Controlled Data Pipeline",
        "IMAS-FAIR Database",
        "Machine & Research Archive",
    ):
        assert concept in text, f"{path.name}: missing concept {concept!r}"
        positions.append(text.index(concept))
    assert positions == sorted(positions), f"{path.name}: concepts are out of order"


def test_both_readmes_open_with_the_same_identity_sections():
    """The landing narrative must stay aligned even though the prose differs.

    Only the leading identity sections are compared. The English file carries a
    Reference tail the Korean has never had, and forcing that to match would
    mean translating material that is itself scheduled to move to the site.
    """
    english = headings(ENGLISH)[:IDENTITY_SECTIONS]
    korean = headings(KOREAN)[:IDENTITY_SECTIONS]
    assert len(english) == len(korean) == IDENTITY_SECTIONS
    # Same shape: "what it is", "what you can do", "the research".
    assert "VAFT" in english[0] and "VAFT" in korean[0]
    assert english[1].endswith("?") and korean[1].endswith("?")
    assert "VEST" in english[2] and "VEST" in korean[2]


# ---------------------------------------------------------------------------
# Do not sell the future as the present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_long_term_direction_is_labelled_not_claimed(path):
    """#330 §4: these must be visibly future, never current functionality."""
    text = path.read_text(encoding="utf-8").lower()
    for term in LONG_TERM_TERMS:
        if term not in text:
            continue
        window = text[max(0, text.index(term) - 700): text.index(term) + 200]
        assert any(
            marker in window
            for marker in ("long-term", "not current", "장기 방향", "현재 기능이 아니")
        ), (
            f"{path.name}: {term!r} appears without a nearby marker saying it is "
            "long-term direction rather than shipped functionality"
        )


# ---------------------------------------------------------------------------
# Licences must ship with the source
# ---------------------------------------------------------------------------


def test_third_party_notices_left_the_readme_but_not_the_repository():
    """Reproducing these licences is a distribution obligation, not a web page."""
    assert NOTICES.is_file() and NOTICES_KO.is_file()
    for path in (NOTICES, NOTICES_KO):
        body = path.read_text(encoding="utf-8")
        assert "OPEN-ADAS" in body
        assert "OMFIT" in body
        assert "MIT" in body or "Permission is hereby granted" in body
    for path in (ENGLISH, KOREAN):
        assert "THIRD_PARTY_NOTICES" in path.read_text(encoding="utf-8"), (
            f"{path.name} must link the notices it no longer contains"
        )


# ---------------------------------------------------------------------------
# The README should point onward, not carry everything
# ---------------------------------------------------------------------------


def test_the_readme_routes_to_the_deeper_surfaces():
    """#330: the README is a landing page, not the complete manual."""
    text = ENGLISH.read_text(encoding="utf-8")
    for target in ("tutorial/README.md", "notebooks/README.md",
                   "install/README.md", "vest-tokamak.github.io/vaft"):
        assert target in text, f"README.md does not link {target}"


def test_no_relative_link_is_broken():
    """Every in-repo link the READMEs make must resolve."""
    for path in (ENGLISH, KOREAN, NOTICES, NOTICES_KO):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?://|#|mailto:)([^)#]+)", text):
            assert (ROOT / target).exists(), f"{path.name}: broken link to {target}"
