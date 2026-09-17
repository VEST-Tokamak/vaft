"""Keep the two READMEs, and the identity they state, from drifting apart.

`README.ko.md` was a faithful translation that fell four sections and one dead
API behind its English counterpart. Nothing caught that, because nothing
compared them. These tests pin the parts of #330's and #529's reframing that a reader
would notice if they rotted: the core message, the positioning statement, the
four framework concepts, the order they appear in, and the promise that
long-term ambitions are not presented as shipped features.
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

#: The top-level message #529 organises the landing page around. Each README
#: states it in its own language; the English is pinned verbatim because it is
#: the sentence the rest of the page is written to support.
CORE_MESSAGE = "Integrate fusion science knowledge so it can be discovered, verified, compared, and studied."

#: What the four #529 verbs are called in each README, so a translation that
#: quietly drops one of them fails here rather than in a reader's head.
CORE_VERBS = {
    ENGLISH: ("discovered", "verified", "compared", "studied"),
    KOREAN: ("찾고", "검증하고", "비교하고", "연구"),
}

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
    # The claim that distinguishes the framing from "a Python library". The
    # project settled on "framework" rather than "infrastructure"; both READMEs,
    # CONTRIBUTING.md and the repository's About text say the same word.
    assert "scientific framework" in opening or "과학 프레임워크" in opening, (
        f"{path.name}: the opening does not state that VAFT is a scientific framework"
    )
    assert "infrastructure" not in opening and "인프라" not in opening, (
        f"{path.name}: the opening still uses the retired word 'infrastructure'"
    )


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_the_core_message_opens_the_page_with_all_four_verbs(path):
    """#529: one statement of what the four concepts are ultimately *for*."""
    text = path.read_text(encoding="utf-8")
    opening = text[: text.index("\n## ")]
    first_quote = next(line for line in opening.splitlines() if line.startswith("> **"))
    for verb in CORE_VERBS[path]:
        assert verb in first_quote, (
            f"{path.name}: the core message has lost {verb!r}: {first_quote!r}"
        )
    if path is ENGLISH:
        assert CORE_MESSAGE in first_quote


def test_the_overview_says_what_vaft_is_then_enables_then_where_it_runs():
    """#529: what VAFT is -> what it enables -> VEST as the reference implementation."""
    text = ENGLISH.read_text(encoding="utf-8")
    opening = text[: text.index("\n## ")]
    # The core message above repeats the four verbs, so each step is searched
    # for *after* the one before it; a plain index() would match the quote.
    is_ = opening.index("scientific framework")
    enables = opening.find("can be discovered, verified, compared, and studied", is_)
    reference = opening.find("reference\nimplementation", enables)
    if reference == -1:
        reference = opening.find("reference implementation", enables)
    assert is_ < enables < reference, (
        "the overview must say what VAFT is, then what it enables, then name VEST "
        "as the reference implementation"
    )


def test_traceable_and_verifiable_are_not_used_as_synonyms():
    """#529: traceable and reproducible workflows *enable* verifiable results."""
    text = ENGLISH.read_text(encoding="utf-8")
    pipeline = text[text.index("### Version-Controlled Data Pipeline"):
                    text.index("### IMAS-FAIR Database")]
    assert "traceable" in pipeline.lower() and "reproducible" in pipeline.lower()
    assert "verifiable" in pipeline.lower(), (
        "the pipeline section must say what traceability and reproducibility enable"
    )


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_the_four_framework_concepts_are_present_and_ordered(path):
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


# ---------------------------------------------------------------------------
# Installation advice must not contradict itself across surfaces
# ---------------------------------------------------------------------------

INSTALLATION_PAGE = ROOT / "docs" / "_guide" / "Installation.md"

#: Phrases that steer a reader away from the PyPI package. README.md is the
#: PyPI long description and offers `pip install vaft` as the released package,
#: so no other surface may call that route deprecated (cold review docs F6).
PYPI_DISCOURAGEMENT = (
    "not the recommended",
    "not recommended",
    "no longer recommended",
    "older pypi package",
    "권장하지 않",
)


def test_no_surface_discourages_the_pypi_release_the_readme_offers():
    assert "pip install vaft" in ENGLISH.read_text(encoding="utf-8")
    for path in (ENGLISH, KOREAN, INSTALLATION_PAGE):
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            if "pypi" not in line.lower():
                continue
            window = " ".join(lines[max(0, number - 2):number + 1]).lower()
            hit = [phrase for phrase in PYPI_DISCOURAGEMENT if phrase in window]
            assert not hit, f"{path.name}:{number} discourages the PyPI release: {hit}"
        assert "pip install vaft" in "\n".join(lines), f"{path.name} never names the PyPI install"


@pytest.mark.skipif(not INSTALLATION_PAGE.is_file(), reason="this branch has no docs/ directory")
def test_the_installation_page_names_every_optional_dependency_group():
    """It said "`dev` is the only group" after three more were added (cold review docs F5)."""
    tomllib = pytest.importorskip("tomllib")  # absent on Python 3.10

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    text = INSTALLATION_PAGE.read_text(encoding="utf-8")
    assert "only optional-dependency group" not in text
    missing = [name for name in project["optional-dependencies"] if f"`{name}`" not in text]
    assert not missing, f"Installation.md does not mention the extras {missing}"
