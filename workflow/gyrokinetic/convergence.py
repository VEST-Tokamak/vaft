"""Resolution convergence table from several ``run_linear.py`` trees (#1354).

Run ``run_linear.py`` once at the production resolution (``--base``) and once per varied
setting (``--variant``), each into its own ``--out`` on the same state, surface, field
model and ky. This script pairs the CGYRO records by (state, surface, field, ky) and writes
``convergence.csv``: for each variant, the varied parameters, ``gamma`` and ``omega`` and
their change relative to the base, and whether the change is inside ``--tolerance``.

The resolution of each tree is read from its ``run_manifest.json`` -- the same record the
runs were made with -- so a variant is described by what it was, not by a directory name.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


_NUMERICS = ("n_energy", "n_xi", "n_theta", "n_radial", "box_size", "delta_t",
             "delta_t_method", "max_time", "freq_tol")


def _resolution(record: dict) -> dict:
    """The numerics a record was *run* with (its own provenance), ky excluded."""
    resolution = (record.get("provenance") or {}).get("resolution") or {}
    return {k: resolution.get(k) for k in _NUMERICS}


def _tree(root: Path) -> dict[tuple, dict]:
    """Solved CGYRO records keyed by (state, surface, field, ky).

    Each record is described by its own ``provenance.resolution``, not by the tree's
    manifest: a tree resumed with other flags holds records of several resolutions,
    and the manifest only names the last invocation.
    """
    records: dict[tuple, dict] = {}
    for path in root.rglob("record.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("code") != "cgyro" or "provenance" not in record:
            continue
        key = (record["shot"], round(record["time_efit_s"], 4), record["efit_lineage"],
               round(record["r_over_a"], 3), record["field_model"], round(record["ky"], 4))
        records[key] = record
    return records


def _relative(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference in (None, 0.0):
        return None
    return (value - reference) / abs(reference)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--variant", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    base = _tree(args.base)
    rows = []
    for variant in args.variant:
        records = _tree(variant)
        for key, record in sorted(records.items()):
            reference = base.get(key)
            if reference is None:
                continue
            base_resolution = _resolution(reference)
            varied = {k: v for k, v in _resolution(record).items()
                      if base_resolution.get(k) != v}
            d_gamma = _relative(record.get("gamma"), reference.get("gamma"))
            d_omega = _relative(record.get("omega_ion_negative"),
                                reference.get("omega_ion_negative"))
            rows.append({
                "shot": key[0], "time_efit_s": key[1], "efit_lineage": key[2],
                "r_over_a": key[3], "field_model": key[4], "ky": key[5],
                "variant": variant.name,
                "varied": json.dumps(varied, sort_keys=True),
                "gamma_base": reference.get("gamma"), "gamma": record.get("gamma"),
                "omega_base": reference.get("omega_ion_negative"),
                "omega": record.get("omega_ion_negative"),
                "rel_change_gamma": d_gamma, "rel_change_omega": d_omega,
                "qualified_base": reference.get("qualified"),
                "qualified": record.get("qualified"),
                "within_tolerance": (d_gamma is not None and abs(d_gamma) <= args.tolerance
                                     and (d_omega is None or abs(d_omega) <= args.tolerance)),
                "elapsed_s": record.get("elapsed_s"),
            })
    args.out.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["variant"]
    with open(args.out / "convergence.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "convergence_base.json").write_text(
        json.dumps({"base": str(args.base), "tolerance": args.tolerance,
                    "resolutions": sorted({json.dumps(_resolution(r), sort_keys=True)
                                           for r in base.values()})},
                   indent=1), encoding="utf-8")
    print(json.dumps({"rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
