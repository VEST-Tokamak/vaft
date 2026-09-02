#!/usr/bin/env python3
"""Normalize Jupyter notebook outputs for version control.

Keep only static, reviewable code-cell results: text, images, and ordinary
stdout/stderr streams.  This script intentionally leaves cell sources and
notebook/kernel metadata alone.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import nbformat
from nbformat import NotebookNode


ALLOWED_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "image/png",
        "image/jpeg",
        "image/svg+xml",
    }
)
#: MIME types whose payload is human-readable text rather than base64 data.
_TEXT_MIME_TYPES = frozenset({"text/plain", "text/markdown", "image/svg+xml"})
_WIDGET_METADATA_KEYS = frozenset({"widgets", "widget_state"})
_RENDERING_CELL_METADATA_KEYS = frozenset({"slideshow", "tags"})

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Absolute paths leak the machine that ran the notebook -- a home directory, a
#: checkout location, a per-run temporary directory -- and would differ for
#: every reader. `test_notebook_reliability` already bans such literals in cell
#: sources; outputs are held to the same standard. Order matters: the checkout
#: lives under a home directory, and temp directories are matched before the
#: generic home rule so their tails are dropped rather than kept.
_PATH_REDACTIONS = (
    (re.compile(re.escape(str(_REPO_ROOT))), "<repo>"),
    (re.compile(r"/var/folders/[^\s\"\'\],)]+"), "<tmp>"),
    (re.compile(r"/private/tmp/[^\s\"\'\],)]+"), "<tmp>"),
    (re.compile(r"(?:/Users|/home)/[^/\s\"\']+"), "~"),
)


def redact_paths(text: str) -> str:
    """Replace machine-specific absolute paths with stable placeholders."""
    for pattern, replacement in _PATH_REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _clean_output(output: NotebookNode) -> NotebookNode | None:
    """Return a normalized output, or ``None`` when it is not reviewable."""
    if output.output_type == "stream":
        if output.get("name") in {"stdout", "stderr"}:
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            return nbformat.v4.new_output(
                output_type="stream", name=output.name, text=redact_paths(text)
            )
        return None

    if output.output_type not in {"display_data", "execute_result"}:
        return None

    data = output.get("data", {})
    allowed_data = {}
    for mime_type in data:
        if mime_type not in ALLOWED_MIME_TYPES:
            continue
        value = data[mime_type]
        if mime_type in _TEXT_MIME_TYPES:
            if isinstance(value, list):
                value = "".join(value)
            value = redact_paths(value)
        allowed_data[mime_type] = value
    if not allowed_data:
        return None

    cleaned = nbformat.v4.new_output(output_type=output.output_type, data=allowed_data)
    if output.output_type == "execute_result":
        cleaned.execution_count = None
    return cleaned


def clean_notebook(notebook: NotebookNode) -> bool:
    """Apply the output policy in place and return whether it changed."""
    changed = False

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        cleaned_metadata = {
            key: cell.metadata[key]
            for key in cell.metadata
            if key in _RENDERING_CELL_METADATA_KEYS
        }
        if cleaned_metadata != cell.metadata:
            cell.metadata = cleaned_metadata
            changed = True

        if cell.get("execution_count") is not None:
            cell.execution_count = None
            changed = True

        outputs = cell.get("outputs", [])
        cleaned_outputs = []
        for output in outputs:
            cleaned = _clean_output(output)
            if cleaned is not None:
                cleaned_outputs.append(cleaned)

        if cleaned_outputs != outputs:
            cell.outputs = cleaned_outputs
            changed = True

    for key in _WIDGET_METADATA_KEYS:
        if key in notebook.metadata:
            del notebook.metadata[key]
            changed = True

    return changed


def clean_path(path: Path) -> bool:
    """Clean *path*, rewriting it only when its normalized content differs."""
    with path.open(encoding="utf-8") as source:
        notebook = nbformat.read(source, as_version=nbformat.NO_CONVERT)

    if not clean_notebook(notebook):
        return False

    with path.open("w", encoding="utf-8") as destination:
        nbformat.write(notebook, destination, version=nbformat.NO_CONVERT)
    return True


def main(paths: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="notebooks to normalize")
    arguments = parser.parse_args(paths)

    invalid_paths = [path for path in arguments.paths if path.suffix != ".ipynb"]
    if invalid_paths:
        parser.error("only .ipynb files are supported: " + ", ".join(map(str, invalid_paths)))

    for path in arguments.paths:
        if clean_path(path):
            print(f"normalized {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
