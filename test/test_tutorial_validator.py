"""Regression tests for the standalone tutorial validator."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "verify_tutorial_under_test", ROOT / "test" / "verify_tutorial.py"
)
VALIDATOR = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def test_machine_path_pattern_allows_urls_but_rejects_absolute_paths():
    for value in ("https://example.test/data", "http://127.0.0.1:5101"):
        assert VALIDATOR.MACHINE_PATH.search(value) is None

    for value in ("/Users/name/checkout", "/home/name/checkout", "C:/checkout"):
        assert VALIDATOR.MACHINE_PATH.search(value) is not None


def test_inventory_ignores_runtime_output_data(monkeypatch, tmp_path):
    tutorial = tmp_path / "tutorial"
    for name in ("common", "01", "02", "03", "04", "05", "06"):
        (tutorial / "figures" / name).mkdir(parents=True)
    scratch = tutorial / "outputs" / "scratch.csv"
    scratch.parent.mkdir()
    scratch.write_text("value\n1\n", encoding="utf-8")

    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(VALIDATOR, "TUTORIAL", tutorial)
    failures = []
    VALIDATOR._validate_inventory(failures)

    assert all("scratch.csv" not in failure for failure in failures)


FRESHNESS_SPEC = spec_from_file_location(
    "verify_tutorial_freshness_under_test",
    ROOT / "test" / "verify_tutorial_freshness.py",
)
FRESHNESS = module_from_spec(FRESHNESS_SPEC)
assert FRESHNESS_SPEC.loader is not None
FRESHNESS_SPEC.loader.exec_module(FRESHNESS)

TUTORIAL = ROOT / "tutorial"
DECK_01 = TUTORIAL / "01_getting_started_with_vaft.pdf"
DECK_02 = TUTORIAL / "02_operation_scenario_and_vacuum_fields.pdf"


def test_page_counter_reads_decks_packed_into_object_streams():
    # pdfTeX compresses the page dictionaries, so a plain byte scan finds none.
    payload = DECK_02.read_bytes()
    assert b"/Type /Page" not in payload
    assert VALIDATOR.count_pdf_pages(payload) == 4
    assert VALIDATOR.count_pdf_pages(DECK_01.read_bytes()) == 10


def test_pdf_problems_accepts_a_committed_deck():
    assert VALIDATOR.pdf_problems(DECK_02.read_bytes()) == []


def test_pdf_problems_rejects_damaged_artifacts():
    payload = DECK_02.read_bytes()

    assert VALIDATOR.pdf_problems(b"not a pdf at all" * 100) == [
        "not a plausible compiled PDF"
    ]
    assert VALIDATOR.pdf_problems(payload[:200]) == ["not a plausible compiled PDF"]

    truncated = VALIDATOR.pdf_problems(payload[: len(payload) // 2])
    assert any("truncated" in problem for problem in truncated)


def test_pairing_requires_a_rebuilt_pdf_for_a_changed_deck_source():
    failures = FRESHNESS.pairing_failures({"tutorial/01_getting_started_with_vaft.tex"})
    assert len(failures) == 1
    assert "01_getting_started_with_vaft.pdf" in failures[0]


def test_pairing_accepts_a_deck_source_committed_with_its_pdf():
    assert (
        FRESHNESS.pairing_failures(
            {
                "tutorial/01_getting_started_with_vaft.tex",
                "tutorial/01_getting_started_with_vaft.pdf",
            }
        )
        == []
    )


def test_pairing_requires_a_rebuild_when_a_session_figure_changes():
    failures = FRESHNESS.pairing_failures({"tutorial/figures/03/profile.pdf"})
    assert len(failures) == 1
    assert "03_equilibrium_and_kinetic_profiles.pdf" in failures[0]


def test_pairing_requires_every_deck_to_rebuild_for_a_shared_figure():
    failures = FRESHNESS.pairing_failures({"tutorial/figures/common/logo.pdf"})
    assert len(failures) == len(VALIDATOR.SESSIONS)


def test_pairing_ignores_changes_that_do_not_feed_a_deck():
    assert FRESHNESS.pairing_failures({"tutorial/README.md", "vaft/__init__.py"}) == []


def test_compare_accepts_a_rebuild_with_the_committed_page_structure(tmp_path):
    for source in (DECK_01, DECK_02):
        (tmp_path / source.name).write_bytes(source.read_bytes())

    assert FRESHNESS.compare_failures(tmp_path, tmp_path) == []


def test_compare_rejects_a_stale_committed_deck(tmp_path):
    committed = tmp_path / "committed"
    rebuilt = tmp_path / "rebuilt"
    committed.mkdir()
    rebuilt.mkdir()

    # The committed artifact carries a different deck's page count.
    (committed / DECK_01.name).write_bytes(DECK_02.read_bytes())
    (rebuilt / DECK_01.name).write_bytes(DECK_01.read_bytes())

    failures = FRESHNESS.compare_failures(committed, rebuilt)
    assert len(failures) == 1
    assert "stale" in failures[0]


def test_compare_rejects_a_damaged_rebuild(tmp_path):
    committed = tmp_path / "committed"
    rebuilt = tmp_path / "rebuilt"
    committed.mkdir()
    rebuilt.mkdir()

    payload = DECK_01.read_bytes()
    (committed / DECK_01.name).write_bytes(payload)
    (rebuilt / DECK_01.name).write_bytes(payload[: len(payload) // 2])

    failures = FRESHNESS.compare_failures(committed, rebuilt)
    assert failures
    assert all("rebuilt deck" in failure for failure in failures)


def test_compare_rejects_an_inventory_that_changed_during_the_rebuild(tmp_path):
    committed = tmp_path / "committed"
    rebuilt = tmp_path / "rebuilt"
    committed.mkdir()
    rebuilt.mkdir()

    (committed / DECK_01.name).write_bytes(DECK_01.read_bytes())
    (committed / DECK_02.name).write_bytes(DECK_02.read_bytes())
    (rebuilt / DECK_01.name).write_bytes(DECK_01.read_bytes())

    failures = FRESHNESS.compare_failures(committed, rebuilt)
    assert len(failures) == 1
    assert "deck inventory changed" in failures[0]


def test_deck_pdfs_ignores_appledouble_sidecars(tmp_path):
    (tmp_path / DECK_01.name).write_bytes(DECK_01.read_bytes())
    (tmp_path / f"._{DECK_01.name}").write_bytes(b"\x00\x05\x16\x07sidecar")

    assert set(FRESHNESS.deck_pdfs(tmp_path)) == {DECK_01.stem}
