"""Keep both README landing pages short, aligned, and linked to the manual."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "README.md"
KOREAN = ROOT / "README.ko.md"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
NOTICES_KO = ROOT / "THIRD_PARTY_NOTICES.ko.md"

#: The approved brand message (#1763, "domains" wording approved in #1872) leads both language versions.
CORE_MESSAGES = {
    ENGLISH: "Connecting Nuclear Fusion Knowledge Across Domains for Integrated Tokamak Research",
    KOREAN: "여러 연구 영역의 핵융합 지식을 연결해 통합적인 토카막 연구를 돕습니다",
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
    assert "VAFT" in opening
    assert "VEST" in text
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
def test_the_approved_message_opens_the_page(path):
    """The first highlighted statement is the approved brand line, without a period."""
    text = path.read_text(encoding="utf-8")
    opening = text[: text.index("\n## ")]
    first_quote = next(line for line in opening.splitlines() if line.startswith("> **"))
    assert first_quote == f"> **{CORE_MESSAGES[path]}**"


def test_the_approved_message_matches_the_homepage():
    homepage = (ROOT / "docs" / "index.markdown").read_text(encoding="utf-8")
    assert f'<p class="vaft-hero-tagline">{CORE_MESSAGES[ENGLISH]}</p>' in homepage
    brand = (ROOT / "docs" / "assets" / "brand")
    assert f'TAGLINE = "{CORE_MESSAGES[ENGLISH]}"' in (
        brand / "make_brand_assets.py"
    ).read_text(encoding="utf-8")
    assert CORE_MESSAGES[ENGLISH] in (brand / "vaft-social-preview.svg").read_text(encoding="utf-8")


def test_the_overview_says_what_vaft_is_then_enables_then_where_it_runs():
    """#529: what VAFT is -> what it enables -> VEST as the reference implementation."""
    text = ENGLISH.read_text(encoding="utf-8")
    is_ = text.index("scientific framework")
    enables = text.index("It connects experimental data", is_)
    reference = text.index("reference implementation", enables)
    assert is_ < enables < reference, (
        "the overview must say what VAFT is, then what it enables, then name VEST "
        "as the reference implementation"
    )


def test_traceable_and_verifiable_are_not_used_as_synonyms():
    """#529: traceable and reproducible workflows *enable* verifiable results."""
    text = ENGLISH.read_text(encoding="utf-8")
    credibility = next(line for line in text.splitlines() if line.startswith("- **Credibility:**"))
    assert all(word in credibility for word in ("traceable", "reproducible", "verifiable"))


@pytest.mark.parametrize("path", [ENGLISH, KOREAN], ids=["en", "ko"])
def test_the_four_perspectives_are_present_and_ordered(path):
    """The concise viewpoints follow #330's conceptual order."""
    text = path.read_text(encoding="utf-8")
    labels = (
        ("Representation", "Research infrastructure", "Credibility", "Research practice and portability")
        if path == ENGLISH else ("표현", "연구 인프라", "신뢰성", "연구 방식과 이식성")
    )
    positions = [text.index(f"- **{label}:**") for label in labels]
    assert positions == sorted(positions), f"{path.name}: concepts are out of order"


def test_both_readmes_have_the_same_short_landing_structure():
    sections = (
        ("What VAFT connects", "Four enabling perspectives", "How results are produced", "Research with VAFT",
         "Architecture across devices", "VEST reference implementation", "Quick start", "Learn more"),
        ("VAFT가 연결하는 것", "이를 가능하게 하는 네 관점", "결과가 만들어지는 과정", "VAFT로 할 수 있는 연구",
         "여러 장치에 적용하는 구조", "VEST 참조 구현", "빠른 시작", "자세한 문서"),
    )
    diagrams = (
        "fusion_research_ecosystem_presentation.svg",
        "vaft_four_pillars.svg",
        "scientific_workflow.svg",
        "machine_agnostic_architecture.svg",
        "vest_data_platform_overview.svg",
    )
    for path, expected in zip((ENGLISH, KOREAN), sections):
        text = path.read_text(encoding="utf-8")
        assert headings(path) == list(expected)
        assert len(text.splitlines()) <= 100
        images = re.findall(r"!\[([^]]+)\]\(([^)]+)\)", text)
        assert len(images) == len(diagrams)
        assert all(alt.strip() for alt, _ in images)
        assert tuple(url.rsplit("/", 1)[-1] for _, url in images) == diagrams
        assert all(
            url.startswith("https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/")
            for _, url in images
        )
        assert "fusion_science_knowledge_lifecycle.svg" not in text
        assert "vaft.omas.sample_ods()" in text


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


def test_details_removed_from_readme_are_available_in_the_documentation():
    destinations = {
        "docs/_guide/Database.md": ("Source namespaces and data products", "derived_cache"),
        "docs/_guide/Profiles.md": ("result.slice_statuses", "efit_configuration.json"),
        "docs/_guide/Equilibrium_representations.md": ("Hausdorff", "AMBIGUOUS"),
    }
    for file, details in destinations.items():
        content = (ROOT / file).read_text(encoding="utf-8")
        for detail in details:
            assert detail in content, f"{file}: missing migrated technical detail {detail}"


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
