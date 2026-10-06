#!/usr/bin/env python3
"""Per-slice EFIT model-form spread from the #579 ensemble, for confinement analyses (#548).

    python workflow/efit_uncertainty_calibration/model_spread.py \\
        --tables full/table_*.json pilot/v2_table_39915.json ... --output model_spread.csv

One row per ``(shot, time_efit_s, efit_lineage)`` -- the #1454 state key, with
``time_efit_s`` rounded to 1e-4 s and lineage ``magnetics`` -- and, for each of
two member sets, the spread of ``w_mhd_J``, ``beta_p`` and ``li``:

* ``viable`` -- fit-quality ``good`` members of the (2,1) and (1,2) bases, the
  two that converge and give the high-W branch (#579 pilot);
* ``admissible`` -- every admissible member of the 120-setting ensemble.

Per quantity and set: ``n``, ``median``, the 16th and 84th percentiles, and
``sigma_log = (ln q84 - ln q16) / 2`` (a 1-sigma relative width; NaN when a
value is not positive).  ``bimodal`` flags a set whose W q84/q16 exceeds
``--bimodal-ratio``: two solution branches, which a single sigma would hide.
Thomson never selects a member.  A schema is written beside the CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"^p(?P<p>\d)f(?P<f>\d)_")
VIABLE_BASES = {(2, 1), (1, 2)}
QUANTITIES = (("w_mhd_J", "wmhd"), ("beta_p", "betap"), ("li", "li"))
SETS = ("viable", "admissible")


def _criteria():
    spec = importlib.util.spec_from_file_location("model_spread_criteria", HERE / "criteria.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stats(values: list[float]) -> dict[str, float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not finite.size:
        return {"n": 0, "median": math.nan, "q16": math.nan, "q84": math.nan, "sigma_log": math.nan}
    q16, median, q84 = np.percentile(finite, [16, 50, 84])
    sigma_log = 0.5 * (math.log(q84) - math.log(q16)) if q16 > 0 and q84 > 0 else math.nan
    return {"n": int(finite.size), "median": float(median), "q16": float(q16), "q84": float(q84),
            "sigma_log": sigma_log}


def build(tables: Sequence[Path], bimodal_ratio: float) -> list[dict[str, Any]]:
    criteria = _criteria()
    members: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {s: [] for s in SETS})
    for path in tables:
        for record in json.loads(Path(path).read_text(encoding="utf-8"))["records"]:
            if "error" in record or record.get("setting") == "routine":
                continue
            evaluation = criteria.evaluate(record)
            if evaluation["verdicts"]["admissible"]["status"] != "pass":
                continue
            key = (int(record["shot"]), int(record["time_ms"]))
            members[key]["admissible"].append(record)
            match = NAME.match(record["setting"])
            basis = (int(match["p"]), int(match["f"])) if match else None
            if evaluation["good"] and basis in VIABLE_BASES:
                members[key]["viable"].append(record)
    rows = []
    for (shot, time_ms), sets in sorted(members.items()):
        row: dict[str, Any] = {"shot": shot, "time_efit_s": round(time_ms * 1e-3, 4), "efit_lineage": "magnetics"}
        for name in SETS:
            for column, key in QUANTITIES:
                stats = _stats([float((r.get("scalars") or {}).get(key, math.nan)) for r in sets[name]])
                for stat, value in stats.items():
                    row[f"{column}_{name}_{stat}"] = value
            w = row[f"w_mhd_J_{name}_q16"], row[f"w_mhd_J_{name}_q84"]
            row[f"{name}_bimodal"] = bool(w[0] > 0 and w[1] / w[0] > bimodal_ratio) if all(map(math.isfinite, w)) else False
        rows.append(row)
    return rows


SCHEMA_DESCRIPTION = {
    "shot": "VEST shot number",
    "time_efit_s": "EFIT slice time [s], rounded to 1e-4 (the #1454 state key)",
    "efit_lineage": "magnetics (magnetics-only EFIT)",
    "<q>_<set>_n": "members contributing (q in w_mhd_J, beta_p, li; set in viable, admissible)",
    "<q>_<set>_median": "ensemble median",
    "<q>_<set>_q16 / _q84": "16th / 84th percentile over the members",
    "<q>_<set>_sigma_log": "(ln q84 - ln q16)/2: 1-sigma relative width; NaN if a bound is not positive",
    "<set>_bimodal": "W q84/q16 above the bimodal ratio: two solution branches, do not use one sigma",
    "sets": "viable = fit-quality good members of the (2,1) and (1,2) bases; admissible = all admissible members",
    "li": "EFIT a-file li (the record's scalars.li), not converted to another li definition",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bimodal-ratio", type=float, default=2.0)
    args = parser.parse_args(argv)
    rows = build(args.tables, args.bimodal_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["shot"])
        writer.writeheader()
        writer.writerows(rows)
    schema = {"issue": [579, 548], "columns": SCHEMA_DESCRIPTION, "bimodal_ratio": args.bimodal_ratio,
              "tables": [str(t) for t in args.tables], "rows": len(rows)}
    args.output.with_suffix(".schema.json").write_text(json.dumps(schema, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} slices -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
