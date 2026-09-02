#!/usr/bin/env python3
"""Render one stage's canonical validation plots into its FileDB ``plot/`` dir.

The stage's plot set is declared once in :mod:`vaft.validation`; this script is
the workflow's thin driver for it, so a new stage or a new figure is a registry
edit rather than a new script.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vaft-mpl-"))

from vaft.omas.vest_upstream import sha256_file
from vaft.validation import render_stage_plots


LOGGER = logging.getLogger("vaft.generate_stage_plots")


def _csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _load_source(stage: str, path: Path):
    if stage == "raw":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    from omas import load_omas_json

    return load_omas_json(str(path), consistency_check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Declared validation stage.")
    parser.add_argument("--input", required=True, type=Path, help="The stage's data product.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Canonical plot/ directory.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output plot manifest JSON.")
    parser.add_argument("--shot", default=None, type=int)
    parser.add_argument(
        "--stage-manifest",
        default=None,
        type=Path,
        help="The stage's own manifest, for metrics that the ODS cannot carry.",
    )
    parser.add_argument(
        "--required-fields",
        default="",
        help="Comma-separated raw field codes highlighted in the raw QA overview.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    source = _load_source(args.stage, args.input)
    manifest = render_stage_plots(
        args.stage,
        source,
        args.output_dir,
        shot=args.shot,
        required_fields=_csv_ints(args.required_fields),
        stage_manifest=args.stage_manifest,
    )
    if args.shot is not None:
        manifest["shot"] = int(args.shot)
    # Ties the persisted figures to the exact product they validate: the plot
    # rule runs after (and can be re-run independently of) the stage that wrote
    # it, so the stage's own manifest cannot reference these in the other
    # direction without creating a cycle.
    manifest["input"] = {"name": args.input.name, "sha256": sha256_file(args.input)}

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated = [row for row in manifest["plots"] if row["status"] == "generated"]
    LOGGER.info(
        "Stage %s: wrote %d validation plot(s) to %s",
        args.stage,
        len(generated),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
