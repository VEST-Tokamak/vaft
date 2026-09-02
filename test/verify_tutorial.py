#!/usr/bin/env python3
"""Verify the source and artifact contract for the introductory tutorial."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
import zlib

import nbformat


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorial"

# Session 01 is a beginner's walkthrough of the everyday VAFT workflow, so it
# follows the workflow itself rather than the analysis-session structure the
# later sessions share (issue #185). Its headings are pinned per-session here.
SESSION_01_HEADINGS = [
    "What is VAFT",
    "Imports",
    "Loading Data",
    "ODS Structure",
    "Geometry Overview",
    "How VAFT Names Its Plots",
    "Diagnostics",
    "Reading the Axes",
    "Equilibrium",
    "Exercise",
    "Summary",
    "Additional Resources",
]

SESSIONS = {
    1: {
        "notebook": "01_getting_started_with_vaft.ipynb",
        "tex": "01_getting_started_with_vaft.tex",
        "headings": SESSION_01_HEADINGS,
        # Session 01 runs entirely from packaged data; it has no lab branch.
        "modes": ["offline"],
    },
    2: {
        "notebook": "02_operation_scenario_and_vacuum_fields.ipynb",
        "tex": "02_operation_scenario_and_vacuum_fields.tex",
    },
    3: {
        "notebook": "03_equilibrium_and_kinetic_profiles.ipynb",
        "tex": "03_equilibrium_and_kinetic_profiles.tex",
    },
    4: {
        "notebook": "04_fluctuations_and_transient_events.ipynb",
        "tex": "04_fluctuations_and_transient_events.tex",
    },
    5: {
        # Issue #185 intentionally gives the notebook and presentation
        # different stems for this session.
        "notebook": "05_mhd_stability_and_3d_perturbations.ipynb",
        "tex": "05_mhd_linear_stability_and_3d_perturbed_equilibrium.tex",
    },
    6: {
        "notebook": "06_operational_space_and_statistics.ipynb",
        "tex": "06_operational_space_and_statistics.tex",
    },
}

# The analysis-session structure shared by sessions 02-06.
DEFAULT_HEADINGS = [
    "Session Overview",
    "Physical Context",
    "Load / Prepare Data",
    "Guided Analysis",
    "Interpretation Checkpoints",
    "Integrated Analysis",
    "Independent Exercise",
    "Takeaways and Next Steps",
]

MACHINE_PATH = re.compile(
    r"(?:/(?:Users|home|srv|Volumes)/|(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/])"
)
RUNTIME_DIRECTORIES = {".build", "outputs"}
FORBIDDEN_DATA_SUFFIXES = {
    ".csv",
    ".h5",
    ".hdf5",
    ".json",
    ".nc",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".tsv",
    ".xlsx",
}

MINIMUM_PDF_BYTES = 1_000
_PDF_STREAM = re.compile(rb"(?<!end)stream\r?\n")
_PDF_PAGE = re.compile(rb"/Type\s*/Page(?![s/A-Za-z])")


def _inflate_pdf(payload: bytes) -> bytes:
    """Return the payload joined with every FlateDecode stream that inflates.

    pdfTeX packs the page dictionaries into compressed object streams, so they
    are invisible until the streams are inflated.
    """
    blobs = [payload]
    for match in _PDF_STREAM.finditer(payload):
        end = payload.find(b"endstream", match.end())
        if end < 0:
            continue
        try:
            blobs.append(zlib.decompressobj().decompress(payload[match.end() : end]))
        except zlib.error:
            continue
    return b"".join(blobs)


def count_pdf_pages(payload: bytes) -> int:
    """Count page objects in a PDF without depending on a third-party parser."""
    return len(_PDF_PAGE.findall(_inflate_pdf(payload)))


def pdf_problems(payload: bytes) -> list[str]:
    """Report structural defects in a compiled deck PDF."""
    if not payload.startswith(b"%PDF-") or len(payload) < MINIMUM_PDF_BYTES:
        return ["not a plausible compiled PDF"]
    problems: list[str] = []
    if not payload.rstrip().endswith(b"%%EOF"):
        problems.append("is truncated: the %%EOF trailer is missing")
    if count_pdf_pages(payload) < 1:
        problems.append("declares no pages")
    return problems


def _source_text(cell: nbformat.NotebookNode) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _validate_inventory(failures: list[str]) -> None:
    expected_notebooks = {entry["notebook"] for entry in SESSIONS.values()}
    expected_tex = {entry["tex"] for entry in SESSIONS.values()}
    expected_pdfs = {Path(name).with_suffix(".pdf").name for name in expected_tex}

    def artifact_names(pattern: str) -> set[str]:
        return {
            path.name
            for path in TUTORIAL.glob(pattern)
            if not path.name.startswith("._")
        }

    inventories = (
        ("notebook", expected_notebooks, artifact_names("*.ipynb")),
        ("TeX source", expected_tex, artifact_names("*.tex")),
        ("PDF", expected_pdfs, artifact_names("*.pdf")),
    )
    for label, expected, actual in inventories:
        if actual != expected:
            failures.append(
                f"{label} inventory: expected {sorted(expected)}, found {sorted(actual)}"
            )

    figure_root = TUTORIAL / "figures"
    expected_dirs = {"common", "01", "02", "03", "04", "05", "06"}
    actual_dirs = {path.name for path in figure_root.iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        failures.append(
            f"figure directories: expected {sorted(expected_dirs)}, found {sorted(actual_dirs)}"
        )

    for path in TUTORIAL.rglob("*"):
        relative = path.relative_to(TUTORIAL)
        if any(part in RUNTIME_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES:
            failures.append(f"tutorial-only data artifact is not allowed: {path.relative_to(ROOT)}")


def _validate_notebook(
    session: int,
    filename: str,
    failures: list[str],
    expected_headings: list[str] | None = None,
    expected_modes: list[str] | None = None,
) -> None:
    required = expected_headings or DEFAULT_HEADINGS
    expected_modes = expected_modes or ["offline", "lab"]
    path = TUTORIAL / filename
    if not path.exists():
        return
    try:
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
    except Exception as error:
        failures.append(f"{filename}: invalid notebook: {type(error).__name__}: {error}")
        return

    metadata = book.metadata.get("vaft_tutorial", {})
    if metadata.get("session") != session:
        failures.append(f"{filename}: metadata.vaft_tutorial.session must be {session}")
    if metadata.get("status") not in {"scaffold", "complete"}:
        failures.append(f"{filename}: tutorial status must be scaffold or complete")
    if list(metadata.get("modes", [])) != expected_modes:
        failures.append(f"{filename}: tutorial modes must be {expected_modes}")

    headings: list[str] = []
    for index, cell in enumerate(book.cells):
        source = _source_text(cell)
        if cell.cell_type == "markdown":
            headings.extend(
                line[3:].strip()
                for line in source.splitlines()
                if line.startswith("## ") and not line.startswith("### ")
            )
            continue
        if cell.cell_type != "code":
            continue
        if cell.get("execution_count") is not None:
            failures.append(f"{filename}: cell {index} has an execution count")
        if cell.get("outputs", []) != []:
            failures.append(f"{filename}: cell {index} has stored outputs")
        try:
            tree = ast.parse(source, filename=f"{filename}:cell-{index}")
        except SyntaxError as error:
            failures.append(f"{filename}: cell {index} does not compile: {error}")
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and MACHINE_PATH.search(node.value)
            ):
                failures.append(
                    f"{filename}: cell {index} contains machine-specific path {node.value!r}"
                )

    if headings != required:
        failures.append(
            f"{filename}: expected section order {required}, found {headings}"
        )


def _validate_deck(session: int, filename: str, failures: list[str]) -> None:
    path = TUTORIAL / filename
    if not path.exists():
        return
    source = path.read_text(encoding="utf-8")
    required_fragments = (
        r"\documentclass[aspectratio=169]{beamer}",
        rf"\graphicspath{{{{figures/common/}}{{figures/{session:02d}/}}}}",
        r"\begin{document}",
        r"\end{document}",
    )
    for fragment in required_fragments:
        if fragment not in source:
            failures.append(f"{filename}: missing required fragment {fragment!r}")
    if re.search(r"\\(?:input|include)\s*\{", source):
        failures.append(f"{filename}: decks must not depend on input/include files")

    pdf = path.with_suffix(".pdf")
    if pdf.exists():
        for problem in pdf_problems(pdf.read_bytes()):
            failures.append(f"{pdf.name}: {problem}")


def validate() -> list[str]:
    failures: list[str] = []
    if not TUTORIAL.is_dir():
        return ["tutorial directory is missing"]

    _validate_inventory(failures)
    for session, entry in SESSIONS.items():
        _validate_notebook(
            session,
            entry["notebook"],
            failures,
            entry.get("headings"),
            entry.get("modes"),
        )
        _validate_deck(session, entry["tex"], failures)
    return failures


def main() -> int:
    failures = validate()
    if failures:
        print("Tutorial validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Tutorial validation passed: 6 clean notebooks and 6 standalone slide decks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
