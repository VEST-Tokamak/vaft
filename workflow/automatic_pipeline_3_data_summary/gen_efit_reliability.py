"""Materialize reconstructed-vs-measured EFIT constraint summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from vaft import database
from vaft.database._summary import get_summary_preset


MAGNETIC_OUTPUT_FILENAME = "efit_magnetic_reliability_history.xlsx"
KINETIC_OUTPUT_FILENAME = "efit_kinetic_reliability_history.xlsx"


def generate_efit_reliability_history(
    shot_range: tuple[int, int] | None = None,
    *,
    source: str = "public",
    magnetic_output_path: str | None = None,
    kinetic_output_path: str | None = None,
    rebuild: bool = False,
):
    """Materialize independent magnetic and kinetic reliability tables."""
    outputs = {}
    for group, destination, default_name in (
        ("magnetic", magnetic_output_path, MAGNETIC_OUTPUT_FILENAME),
        ("kinetic", kinetic_output_path, KINETIC_OUTPUT_FILENAME),
    ):
        preset = f"efit_{group}_reliability"
        definition = get_summary_preset(preset)
        frame = database.summary(shot_range, preset=preset, source=source)
        outputs[group] = database.export_summary(
            frame,
            destination or str(Path(__file__).with_name(default_name)),
            mode="replace" if rebuild else "upsert",
            key_columns=None if rebuild else definition.key_columns,
            replace_groups=None if rebuild else definition.replace_groups,
        )
    return outputs


def _shot_range(value: str) -> tuple[int, int]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "shot range must have the form START:END"
        ) from exc
    if start > end:
        raise argparse.ArgumentTypeError("shot range START must not exceed END")
    return start, end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-range", type=_shot_range)
    parser.add_argument("--source", default="public")
    parser.add_argument("--magnetic-output", default=None)
    parser.add_argument("--kinetic-output", default=None)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    generate_efit_reliability_history(
        arguments.shot_range,
        source=arguments.source,
        magnetic_output_path=arguments.magnetic_output,
        kinetic_output_path=arguments.kinetic_output,
        rebuild=arguments.rebuild,
    )
