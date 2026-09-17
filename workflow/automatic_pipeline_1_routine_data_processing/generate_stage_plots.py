#!/usr/bin/env python3
"""Render one stage's canonical validation plots into its FileDB ``plot/`` dir.

The stage's plot set is declared once in :mod:`vaft.database.production_qa`; this script is
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
from vaft.database.composition import compose_stage_products
from vaft.database.production_qa import STAGE_PLOT_COMPANIONS, render_stage_plots


LOGGER = logging.getLogger("vaft.generate_stage_plots")


def _csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _load_source(stage: str, path: Path):
    if stage == "raw":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    from vaft.omas import load

    return load(path)


def _companions(stage: str, requested: list[str]) -> dict[str, Path]:
    """Pair `--compose-with` arguments against what the stage declares it needs.

    Both directions are errors.  A stage that declares a companion cannot render
    without it -- its figures would simply report the IDS missing, which reads
    as a data defect rather than a wiring one.  A stage that declares none must
    not accept one, or `--compose-with` becomes a way to feed arbitrary products
    into a figure set that never asked for them.
    """
    declared = STAGE_PLOT_COMPANIONS.get(stage, ())
    paired: dict[str, Path] = {}
    for item in requested:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--compose-with expects STAGE=PATH; got {item!r}")
        if name not in declared:
            raise SystemExit(
                f"Stage {stage!r} declares no companion {name!r} in "
                f"STAGE_PLOT_COMPANIONS (declared: {declared or 'none'})."
            )
        paired[name] = Path(value)
    missing = tuple(name for name in declared if name not in paired)
    if missing:
        raise SystemExit(
            f"Stage {stage!r} needs --compose-with for: " + ", ".join(missing)
        )
    return paired


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
        "--compose-with",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="A companion stage product this stage's figures need (see STAGE_PLOT_COMPANIONS).",
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
    companions = _companions(args.stage, args.compose_with)
    composed_with: list[dict[str, str]] = []
    if companions:
        # `compose_stage_products` composes a diagnostics product with an eddy
        # one specifically, because that pairing has a physical invariant to
        # check. It is not a generic merge, so a companion it cannot express is
        # refused here rather than reaching it as the wrong argument -- and
        # refused by name, since the registry is what a reader would edit.
        companion = companions.get("diagnostics")
        if companion is None or len(companions) > 1:
            raise SystemExit(
                f"Stage {args.stage!r} declares companions "
                f"{tuple(companions)} in STAGE_PLOT_COMPANIONS, but the only "
                "composition implemented here is diagnostics + eddy. Teach "
                "vaft.database.composition how to compose the new pairing "
                "before declaring it."
            )
        source, provenance = compose_stage_products(
            diagnostics=companion,
            eddy=args.input,
            eddy_manifest=args.stage_manifest,
        )
        composed_with.append(
            {
                "stage": "diagnostics",
                "name": companion.name,
                "sha256": provenance["diagnostics"]["sha256"],
            }
        )
    else:
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
    if composed_with:
        # Without this the manifest claims the figures came from one product
        # when they came from two, and the second one is the half carrying the
        # magnetics they are validated against.
        manifest["composed_with"] = composed_with

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
