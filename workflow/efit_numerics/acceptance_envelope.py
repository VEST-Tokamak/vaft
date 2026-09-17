"""What EFIT will accept, against what VEST actually produces (#171).

    PYTHONPATH=$PWD python workflow/efit_numerics/acceptance_envelope.py \\
        --table test/data/efit_acceptance_envelope.json --markdown /tmp/envelope.md

EFIT does not merely fit an equilibrium, it judges one. ``chkerr`` tests every
solution against the ``&incheck`` bounds it reads from ``mhdin.dat`` in the
table directory and raises the numbered failures that set ``jflag``. Those
bounds are dimensioned for a machine, and EFIT's built-in ones are DIII-D's:
``aminor_min = 30 cm``, ``rcntr_min = 90 cm``. The packaged VEST table softens
them part-way, to 25 cm and 30 cm, which is still above what VEST produces.

So this audit asks one question the #171 baseline could not: **of the failures
that reject every VEST slice, which are statements about the fit and which are
statements about the machine the bounds were written for?** It runs nothing --
the baseline's a-files already hold the answer.

The proposed VEST envelope is derived from the limiter through
:func:`vaft.machine_mapping.efund_geometry.vest_acceptance_envelope`, never
fitted to a discharge. A bound tuned until 39915 passes would prove nothing,
which is the same trap the Green-table work had to avoid.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_acceptance_envelope.json"

#: The a-file scalar each geometric criterion is tested against, and the
#: ``chkerr`` failure number it raises. Lengths in the a-file are centimetres,
#: as the envelope is.
CRITERIA: tuple[tuple[str, str, str, int], ...] = (
    ("aminor", "aminor_min", "aminor_max", 5),
    ("elong", "elong_min", "elong_max", 6),
    ("rcntr", "rcntr_min", "rcntr_max", 7),
    ("rcurrt", "rcurrt_min", "rcurrt_max", 8),
    ("zcntr", "zcntr_min", "zcntr_max", 9),
    ("zcurrt", "zcurrt_min", "zcurrt_max", 10),
    ("li", "li_min", "li_max", 2),
    ("qstar", "qstar_min", "qstar_max", 13),
)

#: EFIT's own compiled-in defaults, for the record (``set_eparm.F90``).
EFIT_BUILTIN = {"aminor_min": 30.0, "rcntr_min": 90.0}


def observed(baseline_dir: Path) -> dict[str, list[float]]:
    """Every a-file scalar the baseline produced, by name.

    Zeros are kept and reported rather than filtered: a reconstruction that
    collapsed writes ``aminor = 0``, and a floor that rejects it is doing its
    job. Confusing that with a floor that rejects a real 21 cm plasma is
    exactly the error this audit exists to prevent.
    """
    from vaft.data import read_aeqdsk

    values: dict[str, list[float]] = {}
    for path in sorted(glob.glob(str(baseline_dir / "shot_*" / "a0*"))):
        record = read_aeqdsk(path)
        for name, *_ in CRITERIA:
            value = record.scalars.get(name)
            if value is not None and np.isfinite(float(value)):
                values.setdefault(name, []).append(float(value))
    return values


def audit(packaged: Any, proposed: Any, samples: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Per criterion: the bound, what VEST produced, and who is at fault."""
    rows = []
    for scalar, low_name, high_name, failure in CRITERIA:
        values = np.asarray(samples.get(scalar, []), dtype=float)
        nonzero = values[values != 0.0]
        low, high = getattr(packaged, low_name), getattr(packaged, high_name)
        new_low, new_high = getattr(proposed, low_name), getattr(proposed, high_name)
        outside = int(((values < low) | (values > high)).sum()) if values.size else 0
        outside_nonzero = int(((nonzero < low) | (nonzero > high)).sum()) if nonzero.size else 0
        still_outside = int(((nonzero < new_low) | (nonzero > new_high)).sum()) if nonzero.size else 0
        rows.append(
            {
                "scalar": scalar,
                "failure": failure,
                "packaged": [low, high],
                "efit_builtin": [EFIT_BUILTIN.get(low_name), EFIT_BUILTIN.get(high_name)],
                "proposed": [new_low, new_high],
                "changed": (low, high) != (new_low, new_high),
                "samples": int(values.size),
                "median": float(np.median(nonzero)) if nonzero.size else None,
                "range": [float(nonzero.min()), float(nonzero.max())] if nonzero.size else None,
                "outside_packaged": outside,
                "outside_packaged_excluding_collapsed": outside_nonzero,
                "outside_proposed_excluding_collapsed": still_outside,
                # The finding this audit is for: a bound that rejects a real
                # reconstruction is about the machine, not about the fit.
                "rejects_real_equilibria": outside_nonzero > 0,
            }
        )
    return rows


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# EFIT's acceptance envelope against what VEST produces (#171)", ""]
    lines.append(
        "`chkerr` tests every reconstruction against the `&incheck` bounds EFIT reads from the "
        "table directory. These are the bounds, and what the reference set actually produced."
    )
    lines.append("")
    lines.append("| a-file scalar | failure | packaged bound | VEST median | VEST range | rejected by the packaged bound | proposed bound | rejected by the proposal |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in payload["criteria"]:
        span = "–" if row["range"] is None else f"{row['range'][0]:.3g} … {row['range'][1]:.3g}"
        median = "–" if row["median"] is None else f"{row['median']:.3g}"
        mark = " **⚠**" if row["rejects_real_equilibria"] else ""
        lines.append(
            f"| {row['scalar']} | #{row['failure']} "
            f"| {row['packaged'][0]:.4g} … {row['packaged'][1]:.4g} | {median} | {span} "
            f"| {row['outside_packaged_excluding_collapsed']} of {row['samples']}{mark} "
            f"| {row['proposed'][0]:.4g} … {row['proposed'][1]:.4g} "
            f"| {row['outside_proposed_excluding_collapsed']} |"
        )
    lines.append("")
    lines.append("Counts exclude collapsed reconstructions, which write zeros and are correctly rejected by any floor.")
    lines.append("")
    offenders = [row for row in payload["criteria"] if row["rejects_real_equilibria"]]
    if offenders:
        lines.append("## Bounds that reject real VEST equilibria")
        lines.append("")
        for row in offenders:
            lines.append(
                f"- **{row['scalar']}** (failure #{row['failure']}): bound "
                f"{row['packaged'][0]:.4g}–{row['packaged'][1]:.4g}, VEST median {row['median']:.3g}. "
                f"{row['outside_packaged_excluding_collapsed']} of {row['samples']} reconstructions are "
                "outside it, and no termination setting can move that."
            )
        lines.append("")
    lines.append("## The proposed envelope, and where each bound comes from")
    lines.append("")
    lines.append(
        "Geometric bounds are derived from the limiter outline in the canonical static ODS: "
        f"R {payload['limiter']['r'][0]:.3f}–{payload['limiter']['r'][1]:.3f} m, "
        f"Z {payload['limiter']['z'][0]:.3f}–{payload['limiter']['z'][1]:.3f} m. "
        "A plasma cannot be wider than the vessel that holds it, and cannot be narrower than a "
        f"few grid cells ({payload['grid']['cells']} cells of {payload['grid']['cell_cm']:.2f} cm) "
        "without ceasing to be resolved. Physics bounds — li, betap, qstar, elongation — and the "
        "consistency tolerances are left exactly as they were: what they should be is a separate "
        "argument, and moving them alongside the geometry would confound the two."
    )
    lines.append("")
    lines.append(f"Packaged envelope sha256 `{payload['packaged_sha256'][:16]}`, proposed `{payload['proposed_sha256'][:16]}`.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="the #171 baseline output directory")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--era", default="vest-pre-43017-pf1906")
    args = parser.parse_args(argv)

    import f90nml

    from vaft.code.efit.config import EFITAcceptanceEnvelope
    from vaft.data.resources import data_path
    from vaft.machine_mapping.efund_geometry import vest_acceptance_envelope
    from vaft.omas.vest_upstream import build_static_ods

    bundled = f90nml.read(str(data_path("efit/mhdin.dat")))["incheck"]
    packaged = EFITAcceptanceEnvelope(**{
        field.name: bundled[field.name]
        for field in fields(EFITAcceptanceEnvelope)
        if field.name in bundled
    })
    ods, _ = build_static_ods(args.era)
    proposed = vest_acceptance_envelope(ods)

    outline = ods["wall.description_2d.0.limiter.unit.0.outline"]
    r = np.asarray(outline["r"], dtype=float)
    z = np.asarray(outline["z"], dtype=float)

    samples = observed(args.baseline.expanduser())
    payload = {
        "schema_version": SCHEMA,
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "era": args.era,
        "limiter": {"r": [float(r.min()), float(r.max())], "z": [float(z.min()), float(z.max())]},
        "grid": {"cells": 5, "cell_cm": (1.2 - 0.05) / 128 * 100.0},
        "packaged": packaged.to_namelist(),
        "packaged_sha256": packaged.sha256,
        "proposed": proposed.to_namelist(),
        "proposed_sha256": proposed.sha256,
        "criteria": audit(packaged, proposed, samples),
    }
    payload["summary"] = {
        "criteria": len(payload["criteria"]),
        "rejecting_real_equilibria": [row["scalar"] for row in payload["criteria"] if row["rejects_real_equilibria"]],
        "changed_by_the_proposal": [row["scalar"] for row in payload["criteria"] if row["changed"]],
        "still_rejected_by_the_proposal": [
            row["scalar"] for row in payload["criteria"] if row["outside_proposed_excluding_collapsed"] > 0
        ],
    }
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    text = markdown(payload)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"table: {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
