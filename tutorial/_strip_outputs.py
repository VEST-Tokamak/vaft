#!/usr/bin/env python3
"""Strip a tutorial notebook to its committed form: no outputs, no execution counts.

Tutorial notebooks are source artifacts (see ``tutorial/README.md``): every code
cell is committed with ``outputs: []`` and ``execution_count: null``, which
``test/verify_tutorial.py`` enforces. Learners and reviewers still run them and
save the results from Jupyter. This script lets git store the stripped form
while the working copy keeps its outputs.

Three ways to use it:

* ``python tutorial/_strip_outputs.py --install`` registers it as the
  ``vaft-tutorial-outputs`` clean filter in this clone's git config. The
  repository's ``.gitattributes`` applies that filter to ``tutorial/**/*.ipynb``
  only. Afterwards an executed tutorial shows no change in ``git diff`` and
  ``git add`` stores it without outputs. (``git status`` sees the file size
  change and may list it until the next ``git add``.) ``--uninstall`` removes
  the entry.
* ``python tutorial/_strip_outputs.py`` (no arguments) is the filter itself:
  notebook JSON on stdin, the stripped notebook on stdout.
* ``python tutorial/_strip_outputs.py NOTEBOOK...`` strips files in place.

Only what Jupyter adds by running and saving a notebook is undone: code-cell
outputs and execution counts, the per-cell ``execution`` timing, ``collapsed``
and ``scrolled`` metadata, and the kernel details a running kernel writes into
``metadata.language_info`` (only its ``name`` is kept). Sources and the rest of
the notebook metadata are untouched; in particular a different ``kernelspec``
(running under another kernel) is a real change and still shows as one. The
JSON is written as Jupyter writes it (sorted keys, one-space indent, UTF-8
unescaped, trailing newline), and the committed tutorials are stored in that
form, so a stripped clean notebook is byte-identical to its input.

The installed command names the interpreter that ran ``--install``. If that
environment is removed, git warns that the filter cannot run and stores the
notebook unfiltered (``test/verify_tutorial.py`` then rejects it): run
``--install`` again from the new environment.

Notebooks elsewhere in the repository (``notebooks/``) keep their outputs on
purpose and are not covered by the filter.

The script uses only the standard library, so the filter works in any Python
that can run git hooks, without VAFT or nbformat installed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FILTER_NAME = "vaft-tutorial-outputs"
#: Cell metadata Jupyter writes while running or viewing a cell.
TRANSIENT_CELL_METADATA = ("execution", "collapsed", "scrolled")


def strip_notebook(notebook: dict) -> dict:
    """Remove run results from ``notebook`` in place and return it."""
    metadata = notebook.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("language_info"), dict):
        language = metadata["language_info"]
        metadata["language_info"] = {"name": language["name"]} if "name" in language else {}
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        metadata = cell.get("metadata")
        if isinstance(metadata, dict):
            for key in TRANSIENT_CELL_METADATA:
                metadata.pop(key, None)
    return notebook


def strip_text(text: str) -> str:
    """Strip notebook JSON text; return it unchanged if it is not a notebook."""
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        # A half-resolved merge or a non-notebook file: never lose content in a filter.
        return text
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        return text
    return json.dumps(strip_notebook(notebook), indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def install() -> int:
    top = _git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        print("not inside a git checkout", file=sys.stderr)
        return 1
    # git runs a clean filter from the top of the working tree, so the script path
    # is relative to it; the interpreter is the one running this command.
    interpreter = Path(sys.executable).as_posix()
    command = f'"{interpreter}" tutorial/_strip_outputs.py'
    for key, value in (("clean", command), ("smudge", "cat")):
        result = _git("config", "--local", f"filter.{FILTER_NAME}.{key}", value)
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            return 1
    print(f"registered the {FILTER_NAME} clean filter: {command}")
    print("tutorial/*.ipynb outputs stay in your working copy and are no longer committed.")
    return 0


def uninstall() -> int:
    result = _git("config", "--local", "--remove-section", f"filter.{FILTER_NAME}")
    if result.returncode != 0 and "no such section" not in result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
        return 1
    print(f"removed the {FILTER_NAME} filter from this clone's git config")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--install", action="store_true", help="register the git clean filter in this clone")
    group.add_argument("--uninstall", action="store_true", help="remove the git clean filter from this clone")
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks to strip in place (default: stdin to stdout)")
    args = parser.parse_args(argv)

    if args.install:
        return install()
    if args.uninstall:
        return uninstall()
    if not args.notebooks:
        data = sys.stdin.buffer.read()
        try:
            out = strip_text(data.decode("utf-8")).encode("utf-8")
        except UnicodeDecodeError:
            out = data  # not a UTF-8 notebook: hand it back to git untouched
        sys.stdout.buffer.write(out)
        return 0
    for path in args.notebooks:
        text = path.read_text(encoding="utf-8")
        stripped = strip_text(text)
        if stripped != text:
            path.write_text(stripped, encoding="utf-8", newline="\n")
            print(f"stripped {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
