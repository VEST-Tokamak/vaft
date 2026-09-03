"""Materialize the volume-averaged core/equilibrium database summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from vaft import database
from vaft.database._summary import get_summary_preset


OUTPUT_FILENAME = "volume_averaged_parameters.xlsx"


def generate_volume_averaged_parameter_sheet(
    shot_range: tuple[int, int] | None = None,
    *,
    source: str | None = None,
    directory: str | None = None,
    output_path: str | None = None,
    rebuild: bool = False,
):
    destination = output_path or str(Path(__file__).with_name(OUTPUT_FILENAME))
    frame = database.summary(
        shot_range, preset="volume_averaged", source=source, directory=directory
    )
    definition = get_summary_preset("volume_averaged")
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
    parser.add_argument("--source", default=None)
    parser.add_argument("--directory", default=None, help="deprecated alias for --source")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    generate_volume_averaged_parameter_sheet(
        arguments.shot_range,
        source=arguments.source,
        directory=arguments.directory,
        output_path=arguments.output,
        rebuild=arguments.rebuild,
    )
