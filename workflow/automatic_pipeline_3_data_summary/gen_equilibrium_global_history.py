"""Materialize the equilibrium-global database summary.

Extraction and merge semantics live in :mod:`vaft.database`; this module is a
thin compatibility wrapper for scheduled Pipeline 3 jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vaft import database
from vaft.database._summary import get_summary_preset


OUTPUT_FILENAME = "equilibrium_global_history.xlsx"


def generate_equilibrium_global_history_excel(
    shot_range: tuple[int, int] | None = None,
    *,
    directory: str = "public",
    output_path: str | None = None,
    rebuild: bool = False,
):
    """Query an inclusive shot range and materialize it as an Excel history."""
    destination = output_path or str(Path(__file__).with_name(OUTPUT_FILENAME))
    frame = database.summary(
        shot_range,
        preset="equilibrium_global",
        source=directory,
    )
    definition = get_summary_preset("equilibrium_global")
    return database.export_summary(
        frame,
        destination,
        mode="replace" if rebuild else "upsert",
        key_columns=None if rebuild else definition.key_columns,
        replace_groups=None if rebuild else definition.replace_groups,
    )


def _shot_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("shot range must have the form START:END") from exc
    if start > end:
        raise argparse.ArgumentTypeError("shot range START must not exceed END")
    return start, end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-range", type=_shot_range)
    parser.add_argument("--directory", default="public")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    generate_equilibrium_global_history_excel(
        arguments.shot_range,
        directory=arguments.directory,
        output_path=arguments.output,
        rebuild=arguments.rebuild,
    )
