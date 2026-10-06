#!/usr/bin/env python3
"""Compose ``<shot>.json.gz`` diagnostics+eddy products for ``weight_scan.py --products-dir``.

The study drivers read one "pipeline-until-efit" ODS per shot, as the packaged
reference set ships them.  A FileDB keeps the stages apart (the eddy stage
owns only ``pf_passive``), so this joins them with the same function the
pipeline's constraint stage uses, ``vaft.database.composition.compose_stage_products``,
and writes one gzipped OMAS JSON per shot plus a manifest naming its sources.

    python workflow/efit_uncertainty_calibration/compose_products.py \\
        --filedb ~/runs/campaign/filedb --shots 39915,39916,42962 --output products/
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compose(filedb: Path, shot: int, output: Path) -> dict:
    from omas import save_omas_json
    from vaft.database.composition import compose_stage_products

    diagnostics = filedb / "omas" / "diagnostics" / str(shot) / "output" / "diagnostics.json.gz"
    eddy = filedb / "omas" / "eddy" / str(shot) / "output" / "eddy.json.gz"
    manifest = filedb / "omas" / "eddy" / str(shot) / "metadata" / "manifest.json"
    ods, report = compose_stage_products(diagnostics=diagnostics, eddy=eddy,
                                         eddy_manifest=manifest if manifest.is_file() else None)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{shot}.json.gz"
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "product.json"
        save_omas_json(ods, str(staged))
        with staged.open("rb") as source, gzip.open(target, "wb") as sink:
            sink.write(source.read())
    return {"shot": shot, "product": str(target), "sha256": _sha256(target),
            "diagnostics": {"path": str(diagnostics), "sha256": _sha256(diagnostics)},
            "eddy": {"path": str(eddy), "sha256": _sha256(eddy)},
            "composition": json.loads(json.dumps(report, default=str))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--shots", required=True, help="comma-separated shot numbers")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    entries = []
    for shot in (int(s) for s in args.shots.split(",")):
        entry = compose(args.filedb.expanduser(), shot, args.output.expanduser())
        print(f"{shot}: {entry['product']}", flush=True)
        entries.append(entry)
    (args.output.expanduser() / "manifest.json").write_text(json.dumps({"products": entries}, indent=1) + "\n",
                                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
