#!/usr/bin/env python3
"""Execute notebooks in a real Jupyter kernel and report rendered figures.

The static tests in ``test/test_notebook_reliability.py`` compile cells and
check the backend guard, but they cannot tell whether a figure actually reaches
a frontend. Jupyter Lab and VS Code both drive an ipykernel and paint only the
outputs that kernel emits, so an ``image/png`` output here is exactly what those
frontends would display. What this cannot check is the painting itself -- for
that, open the notebook and look.

Kept out of the pytest suite deliberately: a run takes minutes and several
notebooks need sample data or a configured database.

Usage::

    python notebooks/_verify_rendering.py                 # the offline set
    python notebooks/_verify_rendering.py NAME [NAME ...] # specific notebooks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"

#: Notebooks that run without a database connection or external codes, with the
#: environment each one requires.
OFFLINE_NOTEBOOKS: dict[str, dict[str, str]] = {
    "plotting_sample_using_vaft_plot_module.ipynb": {},
    "vest_experimental_data_list.ipynb": {},
    "confinement_time_scaling.ipynb": {},
    "verification_and_validation.ipynb": {},
    "database_initialization_and_load.ipynb": {"VAFT_DOCS_READ_ONLY": "1"},
}


def _execute(path: Path, extra_env: dict[str, str], timeout: int):
    previous = {key: os.environ.get(key) for key in extra_env}
    had_backend = "MPLBACKEND" in os.environ
    backend = os.environ.pop("MPLBACKEND", None)
    os.environ.update(extra_env)
    try:
        book = nbformat.read(str(path), as_version=4)
        NotebookClient(
            book,
            timeout=timeout,
            kernel_name="python3",
            allow_errors=True,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        ).execute()
        return book
    finally:
        if had_backend:
            os.environ["MPLBACKEND"] = backend
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summarize(book) -> tuple[int, list[str]]:
    images = 0
    errors = []
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(f"{output.ename}: {output.evalue}")
            elif output.output_type in ("display_data", "execute_result"):
                if "image/png" in output.get("data", {}):
                    images += 1
    return images, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="notebook names (default: the offline set)")
    parser.add_argument("--timeout", type=int, default=900, help="per-notebook timeout in seconds")
    arguments = parser.parse_args(argv)

    targets = (
        {name: OFFLINE_NOTEBOOKS.get(name, {}) for name in arguments.notebooks}
        if arguments.notebooks
        else OFFLINE_NOTEBOOKS
    )

    failures = 0
    for name, extra_env in targets.items():
        path = NOTEBOOKS / name
        if not path.exists():
            print(f"MISSING  {name}")
            failures += 1
            continue
        book = _execute(path, extra_env, arguments.timeout)
        images, errors = _summarize(book)
        ok = images > 0 and not errors
        failures += not ok
        detail = f"{images} rendered figure(s), {len(errors)} error(s)"
        if errors:
            detail += f" | first: {errors[0]}"
        print(f"{'OK      ' if ok else 'FAILED  '}{name}: {detail}")

    print(f"\n{len(targets) - failures}/{len(targets)} notebook(s) rendered figures without errors")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
