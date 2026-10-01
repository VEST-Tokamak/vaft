"""``vaft setup [profile]``: report or prepare the runtime environment.

Runs :func:`vaft.setup` with the same safety rules: explicit ``MPLBACKEND``
and headless settings are kept, scientific choices are never touched, and the
``database`` profile only diagnoses.  A backend chosen here lasts only for
this command's process, so from a shell the command is mostly a report; call
``vaft.setup()`` inside a notebook to change how its figures draw.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

from vaft._setup import PROFILES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaft setup",
        description=(
            "Report or prepare VAFT's runtime environment (plot backend, database "
            "configuration). Scientific settings are never changed."
        ),
    )
    parser.add_argument("profile", nargs="?", default="auto", type=str.lower, choices=PROFILES)
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    from vaft._setup import setup

    result = setup(arguments.profile)
    if arguments.format == "json":
        print(json.dumps(result.as_dict(), indent=2))
    elif arguments.format == "markdown":
        print(result._repr_markdown_())
    else:
        print(result)
    return 0


__all__ = ["main"]
