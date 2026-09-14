#!/usr/bin/env python3
"""Bring a packaged sample's PF constraint tree onto the current coilset (#708).

    PYTHONPATH=$PWD python workflow/reference_validation/refresh_sample_coilset.py \\
        vaft/data/samples/39915/source/pipeline-until-efit.json.gz

The packaged samples were built when the constraint tree carried twenty-six PF
channels -- the solenoid in eight segments plus every other circuit split
upper/lower -- and the k-file writer selected the sixteen the Green table
describes **by position**.  That selection was removed in #701, so a stored
twenty-six-channel tree handed to today's writer takes the *first* sixteen:
PF1-1..PF1-8 then PF2, PF3, PF4, into the slots that belong to PF5, PF6, PF9
and PF10.  Different coils, at different radii, and until #708 no error.

This drops the ten channels the current coilset does not include and renumbers
what remains.

**It deliberately does not re-derive the values.**  Re-running the constraint
builder would also re-run the box average, which has changed since these
samples were frozen (#433), so the numbers would move for reasons that have
nothing to do with the coilset.  A packaged reference exists to be stable; the
convention is corrected and the data is left exactly as it was recorded.

The ten channels are only droppable because they are **empty**: PF2, PF3, PF4,
PF7 and PF8 measure exactly 0.0 A in every slice of every packaged sample --
those circuits have no acquisition channel at all, so the mapping writes a
literal zero.  The script verifies that per slice and refuses rather than
discard a channel carrying current.

Afterwards regenerate the paired artifacts, which is what actually ships::

    python workflow/reference_validation/generate_paired_sample.py \\
        --canonical-source vaft/data/samples/39915/source/pipeline-until-efit.json.gz \\
        --manifest vaft/data/samples/39915/manifest.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

CONSTRAINT = "constraints.pf_current"


def _load(path: Path):
    from omas import load_omas_json

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = handle.read()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(payload)
        name = staged.name
    try:
        return load_omas_json(name, consistency_check=False)
    finally:
        Path(name).unlink(missing_ok=True)


def _save(ods, path: Path) -> None:
    from omas import save_omas_json

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        name = staged.name
    try:
        save_omas_json(ods, name)
        payload = Path(name).read_text(encoding="utf-8")
    finally:
        Path(name).unlink(missing_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)


def coilset_group_names() -> tuple[str, ...]:
    """The groups the shipped Green table describes, in its order."""
    from vaft.machine_mapping.efund_geometry import EFIT16_GROUP_NAMES

    return tuple(EFIT16_GROUP_NAMES)


def refresh(ods, *, groups: Sequence[str]) -> dict[str, Any]:
    """Rewrite every slice's ``pf_current`` to ``groups``; report what moved."""
    import copy

    import numpy as np

    slices = len(ods["equilibrium.time_slice"]) if "equilibrium.time_slice" in ods else 0
    report: dict[str, Any] = {"slices": slices, "before": None, "after": len(groups), "dropped": []}
    dropped_names: set[str] = set()

    for index in range(slices):
        root = f"equilibrium.time_slice.{index}.{CONSTRAINT}"
        if root not in ods:
            continue
        count = len(ods[root])
        by_name = {str(ods[f"{root}.{position}.source"]): position for position in range(count)}
        if report["before"] is None:
            report["before"] = count
        if tuple(by_name) == tuple(groups):
            continue

        missing = [name for name in groups if name not in by_name]
        if missing:
            raise SystemExit(
                f"slice {index} has no channel for {', '.join(missing)}; "
                f"it carries {', '.join(by_name)}"
            )

        for name, position in by_name.items():
            if name in groups:
                continue
            measured = float(ods[f"{root}.{position}.measured"])
            if measured != 0.0:
                raise SystemExit(
                    f"slice {index}: channel {name} measures {measured} A and is not in the "
                    "coilset; refusing to discard a circuit that carried current"
                )
            dropped_names.add(name)

        kept = [copy.deepcopy(ods[f"{root}.{by_name[name]}"]) for name in groups]
        del ods[root]
        for position, channel in enumerate(kept):
            ods[f"{root}.{position}"] = channel

    report["dropped"] = sorted(dropped_names)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the sample's canonical pipeline product")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    source = args.source.expanduser()
    if not source.is_file():
        print(f"{source}: no such file", file=sys.stderr)
        return 2

    groups = coilset_group_names()
    ods = _load(source)
    report = refresh(ods, groups=groups)

    if report["before"] == report["after"]:
        print(f"{source.name}: already on the current coilset ({report['after']} channels)")
        return 0
    print(
        f"{source.name}: {report['before']} -> {report['after']} PF channels over "
        f"{report['slices']} slices; dropped {', '.join(report['dropped'])} (all empty)"
    )
    if args.dry_run:
        print("dry run; nothing written")
        return 0
    _save(ods, source)
    print(f"wrote {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
