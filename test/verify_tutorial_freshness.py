#!/usr/bin/env python3
"""Enforce the freshness contract for the committed tutorial slide decks.

Committed deck PDFs cannot be compared byte-for-byte against an independent
rebuild. TeX Live stamps its own version into every file it produces, so a deck
built on TeX Live 2024 never matches the same source built on TeX Live 2023 even
with ``SOURCE_DATE_EPOCH`` and ``FORCE_SOURCE_DATE`` pinned. Byte comparison was
standing in for two properties that *are* portable across TeX distributions, and
this script checks them directly:

``pairing``
    Every deck input touched by a change also ships a rebuilt PDF, so a source
    edit can never land with a stale artifact beside it.

``compare``
    An independent rebuild reproduces the page structure of the committed
    artifact, so a committed PDF cannot have come from unrelated sources.

The two checks are complementary: ``pairing`` catches an edit that was never
rebuilt, and ``compare`` catches a committed artifact that does not correspond
to the committed sources. Neither one detects a hand-forged PDF whose page count
happens to match, which byte comparison would have caught; that trade is
deliberate and is the price of building on a different TeX Live release than the
one that produced the committed files.
"""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    spec = spec_from_file_location(
        "verify_tutorial_contract", Path(__file__).with_name("verify_tutorial.py")
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()

TUTORIAL_PREFIX = "tutorial/"
FIGURE_PREFIX = "tutorial/figures/"
COMMON_FIGURES = "tutorial/figures/common/"


def deck_stems() -> dict[str, str]:
    """Map each two-digit session number to its deck stem."""
    return {
        Path(entry["tex"]).stem[:2]: Path(entry["tex"]).stem
        for entry in VALIDATOR.SESSIONS.values()
    }


def changed_paths(base: str, head: str) -> set[str]:
    """Return the repository-relative paths that differ between two refs."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def pairing_failures(changed: set[str]) -> list[str]:
    """Require a rebuilt PDF beside every changed deck input."""
    stems = deck_stems()
    failures: list[str] = []

    for path in sorted(changed):
        if not path.startswith(TUTORIAL_PREFIX) or not path.endswith(".tex"):
            continue
        pdf = f"{path[: -len('.tex')]}.pdf"
        if pdf not in changed:
            failures.append(
                f"{path} changed but {pdf} did not: rebuild the deck with "
                "`make -C tutorial slides` and commit the PDF"
            )

    touched_common = sorted(
        path for path in changed if path.startswith(COMMON_FIGURES)
    )
    for session, stem in sorted(stems.items()):
        pdf = f"{TUTORIAL_PREFIX}{stem}.pdf"
        if pdf in changed:
            continue
        session_prefix = f"{FIGURE_PREFIX}{session}/"
        touched = sorted(
            path for path in changed if path.startswith(session_prefix)
        ) or touched_common
        if touched:
            failures.append(
                f"{touched[0]} changed but {pdf} did not: rebuild the deck with "
                "`make -C tutorial slides` and commit the PDF"
            )

    return failures


def deck_pdfs(directory: Path) -> dict[str, Path]:
    """Return the committed deck PDFs in a directory, keyed by stem."""
    return {
        path.stem: path
        for path in sorted(directory.glob("[0-9][0-9]_*.pdf"))
        if not path.name.startswith("._")
    }


def compare_failures(committed: Path, rebuilt: Path) -> list[str]:
    """Require the rebuild to reproduce the committed page structure."""
    before = deck_pdfs(committed)
    after = deck_pdfs(rebuilt)
    failures: list[str] = []

    if set(before) != set(after):
        failures.append(
            f"deck inventory changed during the rebuild: committed "
            f"{sorted(before)}, rebuilt {sorted(after)}"
        )

    for stem in sorted(set(before) & set(after)):
        payload = after[stem].read_bytes()
        problems = VALIDATOR.pdf_problems(payload)
        if problems:
            failures.extend(f"{stem}.pdf: rebuilt deck {problem}" for problem in problems)
            continue
        expected = VALIDATOR.count_pdf_pages(before[stem].read_bytes())
        actual = VALIDATOR.count_pdf_pages(payload)
        if expected != actual:
            failures.append(
                f"{stem}.pdf: committed deck has {expected} page(s) but the "
                f"rebuild produced {actual}: the committed PDF is stale"
            )

    return failures


def _report(failures: list[str], success: str) -> int:
    if failures:
        print("Tutorial slide freshness check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(success)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    pairing = commands.add_parser(
        "pairing", help="require a rebuilt PDF beside every changed deck input"
    )
    pairing.add_argument("--base", required=True, help="base ref of the change")
    pairing.add_argument("--head", default="HEAD", help="head ref of the change")

    compare = commands.add_parser(
        "compare", help="require the rebuild to match the committed page structure"
    )
    compare.add_argument(
        "--committed", required=True, type=Path, help="directory of committed PDFs"
    )
    compare.add_argument(
        "--rebuilt", required=True, type=Path, help="directory of rebuilt PDFs"
    )

    args = parser.parse_args(argv)

    if args.command == "pairing":
        changed = changed_paths(args.base, args.head)
        return _report(
            pairing_failures(changed),
            "Tutorial slide freshness check passed: every changed deck input "
            "ships a rebuilt PDF.",
        )

    return _report(
        compare_failures(args.committed, args.rebuilt),
        "Tutorial slide freshness check passed: the independent rebuild "
        "reproduces the committed page structure.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
