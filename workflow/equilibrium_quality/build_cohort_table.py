#!/usr/bin/env python3
"""Build the #1644 equilibrium-quality cohort table from study records and their EFIT products.

    python workflow/equilibrium_quality/build_cohort_table.py \\
        --records ~/runs/campaign/tierA_analysis.json \\
        --filedb ~/runs/campaign/filedb --output ~/runs/campaign/atlas/equilibrium_quality

``--records`` is a study analysis JSON (``{"records": [...]}``: the #1331 Tier A
analysis, or a ``weight_scan.py`` table).  With ``--filedb``, each row also
gets the :mod:`vaft.omas.efit_quality` and ``validate_equilibrium`` columns of
its slice, read from that shot's magnetic EFIT product and matched **by time**
(never by position), and only for the rows of the setting that made the product
(``--product-setting``).  Writes ``cohort_table.csv``, ``summary.json`` (cohort
counts, the rule x cohort census, the generic-validation crosswalk) and a
manifest naming the inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ods(path: Path):
    from omas import load_omas_json

    if path.suffix != ".gz":
        return load_omas_json(str(path), consistency_check=False)
    with gzip.open(path, "rt", encoding="utf-8") as handle, \
            tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(handle.read())
        name = staged.name
    try:
        return load_omas_json(name, consistency_check=False)
    finally:
        Path(name).unlink(missing_ok=True)


class ProductEvidence:
    """Evidence columns for a row, from its shot's EFIT product, matched by time."""

    def __init__(self, filedb: Path, setting: str, tolerance_s: float = 5.0e-4):
        # The product holds ONE setting's reconstruction: evidence goes only to
        # that setting's rows, never to another setting of the same slice.
        self.filedb, self.setting, self.tolerance_s, self.cache = filedb, setting, tolerance_s, {}

    def product(self, shot: int):
        if shot not in self.cache:
            directory = self.filedb / "omas" / "efit" / "magnetic" / str(shot) / "output"
            paths = sorted(directory.glob("efit.json*"))
            self.cache[shot] = _load_ods(paths[0]) if paths else None
        return self.cache[shot]

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        from vaft.validation.equilibrium_quality import efit_evidence_columns

        if row.get("setting") != self.setting:
            return {"evidence_status": f"product is setting {self.setting!r}"}
        ods = self.product(row["shot"])
        if ods is None:
            return {"evidence_status": "no EFIT product"}
        times = np.asarray(ods["equilibrium.time"], float)
        index = int(np.argmin(np.abs(times - row["time_s"])))
        if not math.isclose(times[index], row["time_s"], abs_tol=self.tolerance_s):
            return {"evidence_status": f"no product slice within {self.tolerance_s} s"}
        try:
            return {"evidence_status": "ok", "evidence_time_s": float(times[index]),
                    **efit_evidence_columns(ods, index)}
        except Exception as error:  # evidence for one slice must not stop the table
            return {"evidence_status": f"error: {error!r}"[:200]}


def main(argv: Sequence[str] | None = None) -> int:
    from vaft.validation.equilibrium_quality import (
        CRITERIA_PATH, equilibrium_quality_crosswalk, equilibrium_quality_failure_census,
        equilibrium_quality_summary, equilibrium_quality_table, load_study_criteria,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--filedb", type=Path, default=None)
    parser.add_argument("--product-setting", default="statistical_891",
                        help="the study setting the FileDB's magnetic EFIT products were made with")
    parser.add_argument("--criteria", type=Path, default=CRITERIA_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    criteria = load_study_criteria(args.criteria)
    records = json.loads(args.records.read_text(encoding="utf-8"))["records"]
    evidence = ProductEvidence(args.filedb.expanduser(), args.product_setting) if args.filedb else None
    table = equilibrium_quality_table(records, criteria=criteria, evidence=evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "cohort_table.csv", index=False)
    summary = {
        "summary": equilibrium_quality_summary(table),
        "census": equilibrium_quality_failure_census(table),
        "crosswalk": equilibrium_quality_crosswalk(table),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str, allow_nan=True) + "\n",
                                              encoding="utf-8")
    manifest = {"issue": 1644, "records": str(args.records), "records_sha256": _sha256(args.records),
                "criteria": str(args.criteria), "criteria_sha256": _sha256(args.criteria),
                "criteria_version": getattr(criteria, "CRITERIA_VERSION", None),
                "filedb": None if args.filedb is None else str(args.filedb),
                "product_setting": args.product_setting if args.filedb else None, "rows": int(len(table))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    cohorts = summary["summary"]["cohorts"]
    print(f"{len(table)} rows, {summary['summary']['slices']} slices: "
          + ", ".join(f"{k} {v['slices']}" for k, v in cohorts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
