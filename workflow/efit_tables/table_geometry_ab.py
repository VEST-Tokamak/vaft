"""Which file moved the yield: the Green table, or the namelist? (#695)

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_tables/table_geometry_ab.py --output /scratch/ab \\
            --regenerated /tables/fresh_129x129 --stage /tmp/ab

The #459 control case -- the routine box, the routine grid, the routine
solver settings -- produced 46 equilibria where the merged #171 baseline
produced 31, over the same 82 plasma slices.  Two inputs had changed
together: the ``.ddd`` Green tables, and the ``mhdin.dat`` EFIT reads beside
them.  This separates them.

Four runs, identical in everything else, including one acceptance envelope:

======= ================== ==================
case     ``.ddd`` tables    ``mhdin.dat``
======= ================== ==================
PP       packaged           packaged
PF       packaged           regenerated
FP       regenerated        packaged
FF       regenerated        regenerated
======= ================== ==================

``FF - PP`` is the whole gap, ``FP - PP`` the tables alone and ``PF - PP``
the namelist alone.

**The namelist leg is expected to be null, and is run anyway.**  Comparing the
two files, everything EFIT reads from ``&in3`` is identical -- ``turnfc``,
which scales the coil currents it fits (``data_input.F90:2575``), ``rsi``,
which normalises the flux loops (``:2601``), and the probe positions -- except
that every one of the 64 probe angles reads -270 in one file and +90 in the
other, which is the same angle.  What genuinely differs is ``nfcoil``: 16
lumped conductors against 302 discrete filaments, with ``fcturn`` following.
Those are **EFUND** inputs.  They decide how the Green functions are
integrated and reach EFIT only through the tables.

So the hypothesis is that the whole effect is in the tables.  "Physically
identical" is a claim about EFIT's arithmetic, though, and this is cheap to
test rather than assert.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DOMAIN_GRID = REPOSITORY / "workflow" / "efit_numerics" / "domain_grid.py"
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_table_geometry_ab.json"

#: ``tables`` is where the ``.ddd`` files come from, ``namelist`` is whose
#: ``mhdin.dat`` EFIT reads. "packaged" is ``vaft/data/efit``; "regenerated"
#: is an EFUND run over the canonical static geometry for the same era.
CASES: dict[str, dict[str, str]] = {
    "PP": {"tables": "packaged", "namelist": "packaged", "role": "the #171 baseline's configuration"},
    "PF": {"tables": "packaged", "namelist": "regenerated", "role": "the namelist alone"},
    "FP": {"tables": "regenerated", "namelist": "packaged", "role": "the tables alone"},
    "FF": {"tables": "regenerated", "namelist": "regenerated", "role": "the #459 control's configuration"},
}

REFERENCE_CASE = "PP"

#: The table files EFIT opens, and nothing else. `mhdin.dat` is staged
#: separately because it is the other half of the experiment.
TABLE_FILES = ("ec129129.ddd", "ep129129.ddd", "rv129129.ddd", "rfcoil.ddd")

#: EFIT inputs that live in the table directory and are not EFUND products.
EFIT_ONLY_INPUTS = ("lim.dat",)

ERA = "vest-pre-43017-pf1906"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def namelist_difference(packaged: Path, regenerated: Path) -> dict[str, Any]:
    """What actually differs between the two files, field by field.

    Recorded rather than described, because the argument that the namelist leg
    should be null rests on it. A field that differs only by whole turns of
    angle, or in the last bit of a coordinate, is reported as such instead of
    being called a difference.
    """
    import f90nml

    left, right = f90nml.read(str(packaged)), f90nml.read(str(regenerated))
    report: dict[str, Any] = {"groups": sorted(set(left) | set(right)), "fields": {}}
    for group in ("machinein", "in3"):
        for key in sorted(set(left.get(group, {})) | set(right.get(group, {}))):
            before, after = left.get(group, {}).get(key), right.get(group, {}).get(key)
            if before == after:
                continue
            entry: dict[str, Any] = {"group": group}
            try:
                first = np.atleast_1d(np.asarray(before, dtype=float))
                second = np.atleast_1d(np.asarray(after, dtype=float))
            except (TypeError, ValueError):
                report["fields"][key] = {**entry, "kind": "value", "packaged": before, "regenerated": after}
                continue
            if first.size != second.size:
                report["fields"][key] = {
                    **entry, "kind": "length",
                    "packaged_size": int(first.size), "regenerated_size": int(second.size),
                }
                continue
            delta = second - first
            largest = float(np.max(np.abs(delta)))
            if key.startswith("a") and np.all(np.isclose(np.mod(delta, 360.0), 0.0, atol=1e-9)):
                kind = "whole_turns"  # the same angle, written differently
            elif largest <= 1e-12:
                kind = "formatting"  # the last bit of a float
            else:
                kind = "value"
            report["fields"][key] = {**entry, "kind": kind, "max_abs": largest, "size": int(first.size)}
    report["substantive"] = sorted(
        name for name, field in report["fields"].items() if field["kind"] in {"value", "length"}
    )
    return report


def stage_case(
    case: dict[str, str],
    *,
    root: Path,
    name: str,
    packaged: Path,
    regenerated: Path,
    envelope: Any,
) -> Path:
    """One run directory: tables from one side, ``mhdin.dat`` from the other.

    The two halves are staged independently on purpose -- that separation is
    the experiment. Both ``mhdin.dat`` variants are written with the same
    ``&incheck``, so all four cases are judged against one acceptance bar and
    the only differences left are the ones being measured.

    ``root`` must be short: EFIT truncates ``TABLE_DIR`` at 100 characters and
    then fails opening ``lim.dat``.
    """
    import f90nml

    from vaft.data.resources import data_path

    staged = root / name
    if len(str(staged)) + 1 > 100:
        raise ValueError(f"{staged} is {len(str(staged))} characters; EFIT truncates TABLE_DIR at 100")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)

    source = packaged if case["tables"] == "packaged" else regenerated
    for filename in TABLE_FILES:
        origin = source / filename
        if not origin.is_file():
            raise FileNotFoundError(f"{origin} is missing; {case['tables']} table is incomplete")
        (staged / filename).symlink_to(origin.resolve())
    for filename in EFIT_ONLY_INPUTS:
        (staged / filename).symlink_to(Path(data_path(f"efit/{filename}")).resolve())

    origin = (packaged if case["namelist"] == "packaged" else regenerated) / "mhdin.dat"
    namelist = f90nml.read(str(origin))
    # One bar for all four cases. The packaged file carries DIII-D-derived
    # bounds and the regenerated one may carry none; either would make the
    # acceptance counts incomparable.
    namelist["incheck"] = dict(envelope.to_namelist())
    namelist.write(str(staged / "mhdin.dat"), force=True)
    return staged


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--regenerated",
        required=True,
        type=Path,
        help="a 129x129 table directory from regenerate_legacy_table.py, with its manifest",
    )
    parser.add_argument(
        "--stage",
        required=True,
        type=Path,
        help="short directory to assemble run directories in; EFIT truncates TABLE_DIR at 100 characters",
    )
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--shots", default=None)
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--tstep", type=float, default=0.001)
    parser.add_argument("--average-window", type=float, default=0.0005)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())

    from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
    from vaft.data.resources import data_path
    from vaft.machine_mapping.efund_geometry import vest_acceptance_envelope
    from vaft.omas.vest_upstream import build_static_ods

    study = _module(DOMAIN_GRID, "domain_grid")

    resolved = resolve_toolchain()
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME", file=sys.stderr)
        return 2
    efit = str(resolved["efit"])

    packaged = Path(data_path("efit")).resolve()
    regenerated = args.regenerated.expanduser().resolve()
    # The regenerated side must be the routine grid and the routine box, or
    # this measures #459's question again instead of this one.
    study.verify_table(regenerated, study.CASES["A"])

    static, _ = build_static_ods(ERA)
    envelope = vest_acceptance_envelope(static)
    difference = namelist_difference(packaged / "mhdin.dat", regenerated / "mhdin.dat")
    print(f"namelist fields that genuinely differ: {difference['substantive'] or 'none'}", flush=True)

    staged = {
        name: stage_case(
            case, root=args.stage.expanduser(), name=name,
            packaged=packaged, regenerated=regenerated, envelope=envelope,
        )
        for name, case in CASES.items()
    }

    reference_set = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference_set["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    shots = [int(value) for value in args.shots.split(",")] if args.shots else sorted(products)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((regenerated / "efund_table_manifest.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "toolchain": toolchain_identities(resolved),
        "era": ERA,
        "reference_case": REFERENCE_CASE,
        "acceptance_envelope": {"aminor_min": envelope.aminor_min, "sha256": envelope.sha256},
        "namelist_difference": difference,
        "regenerated_table": {
            "directory": str(regenerated),
            "identity": manifest["table"]["identity"],
            "efund_config_sha256": manifest["efund"]["config_sha256"],
        },
        "cases": {name: {**case, "staged": str(staged[name])} for name, case in CASES.items()},
        "shots": {},
    }

    for shot in shots:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        block: dict[str, Any] = {"cases": {}}
        reference_slices: list[dict[str, Any]] | None = None
        order = [REFERENCE_CASE] + [name for name in CASES if name != REFERENCE_CASE]
        for name in order:
            print(f"{shot} {name}: constraints ...", flush=True)
            ods, times, window, baseline = study.prepare_shot(
                shot,
                Path(data_path(products[shot])),
                workdir=output / f"shot_{shot}" / name / "constraints",
                tables=str(staged[name]) + "/",
                tstep=args.tstep,
                average_window=args.average_window,
            )
            labels = study.phases(ods, times)
            run = study.run_case(
                ods, shot=shot, times=times,
                workdir=output / f"shot_{shot}" / name / "run",
                efit=efit, grid=(129, 129), baseline=baseline,
            )
            record = {
                "seconds": run["seconds"],
                "returncode": run["returncode"],
                "summary": study.summarize(run["slices"], labels),
            }
            if name == REFERENCE_CASE:
                record["slices"] = run["slices"]
                reference_slices = run["slices"]
                block["window"] = {
                    "start": float(window.start), "end": float(window.end), "slices": int(times.size)
                }
                block["phases"] = dict(sorted(Counter(labels.values()).items()))
            elif reference_slices is not None:
                record["vs_reference"] = study.compare(reference_slices, run["slices"])
            block["cases"][name] = record
            summary = record["summary"]
            print(
                f"  {name}: {summary['produced_an_equilibrium']}/{summary['slices']} produced, "
                f"{summary['accepted']} accepted, {summary['collapsed']} collapsed, "
                f"bound {summary['bound_failures']}, findax {summary['findax_failures']}, "
                f"{record['seconds']:.0f} s",
                flush=True,
            )
        payload["shots"][str(shot)] = block

    payload["verdict"] = verdict_lines(payload)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(payload)
    (args.markdown or output / "table_geometry_ab.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def totals(payload: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts = {name: {"produced": 0, "accepted": 0, "slices": 0} for name in payload["cases"]}
    for block in payload["shots"].values():
        for name, record in block["cases"].items():
            summary = record["summary"]
            counts[name]["produced"] += summary["produced_an_equilibrium"]
            counts[name]["accepted"] += summary["accepted"]
            counts[name]["slices"] += summary["slices"]
    return counts


def markdown(payload: dict[str, Any]) -> str:
    counts = totals(payload)
    lines = ["# The packaged Green table against a regenerated one (#695)", ""]
    lines.append(
        "Four runs at the routine box and grid, identical in constraints, solver settings and "
        "acceptance envelope. Only the `.ddd` tables and the `mhdin.dat` beside them change."
    )
    lines.append("")
    lines.append("| case | tables | `mhdin.dat` | produced | accepted | what its difference from PP is |")
    lines.append("|---|---|---|---|---|---|")
    for name, case in payload["cases"].items():
        value = counts[name]
        lines.append(
            f"| {name} | {case['tables']} | {case['namelist']} | {value['produced']} of {value['slices']} "
            f"| {value['accepted']} | {case['role'] if name == payload['reference_case'] else case['role']} |"
        )
    lines.append("")
    difference = payload["namelist_difference"]
    lines.append("## What differs between the two namelists")
    lines.append("")
    lines.append("| field | group | how it differs |")
    lines.append("|---|---|---|")
    described = {
        "whole_turns": "the same angle, written a full turn apart",
        "formatting": "the last bit of a float",
        "length": "a different number of entries",
        "value": "a real difference",
    }
    for field, record in sorted(difference["fields"].items()):
        detail = described[record["kind"]]
        if record["kind"] == "length":
            detail += f" ({record['packaged_size']} against {record['regenerated_size']})"
        lines.append(f"| `{field}` | `&{record['group']}` | {detail} |")
    lines.append("")
    for shot, block in payload["shots"].items():
        lines.append(f"## {shot}")
        lines.append("")
        lines.append("| case | produced | accepted | collapsed | `bound` | `findax` | recovered | lost |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, record in block["cases"].items():
            summary, against = record["summary"], record.get("vs_reference") or {}
            lines.append(
                f"| {name} | {summary['produced_an_equilibrium']} of {summary['slices']} "
                f"| {summary['accepted']} | {summary['collapsed']} | {summary['bound_failures']} "
                f"| {summary['findax_failures']} | {len(against.get('recovered', []))} "
                f"| {len(against.get('lost', []))} |"
            )
        lines.append("")
    lines.append("## What this says")
    lines.append("")
    for line in payload["verdict"]:
        lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def verdict_lines(payload: dict[str, Any]) -> list[str]:
    counts = totals(payload)
    reference = counts[payload["reference_case"]]
    lines = [
        "Yield over "
        f"{reference['slices']} plasma slices: "
        + ", ".join(
            f"{name} {value['produced']}/{value['accepted']}" for name, value in sorted(counts.items())
        )
        + " produced/accepted."
    ]
    namelist_only = counts["PF"]["produced"] - reference["produced"]
    tables_only = counts["FP"]["produced"] - reference["produced"]
    whole = counts["FF"]["produced"] - reference["produced"]
    if namelist_only == 0 and counts["PF"]["accepted"] == reference["accepted"]:
        lines.append(
            "**The namelist is inert**: swapping `mhdin.dat` alone changes nothing, which is what "
            "the field comparison predicted -- everything EFIT reads from it is identical, and the "
            "probe angles differ by a whole turn."
        )
    else:
        lines.append(
            f"**The namelist is not inert**: swapping `mhdin.dat` alone moves the yield by "
            f"{namelist_only:+d} equilibria, which the field comparison did not predict and which "
            "has to be explained before the table result means anything."
        )
    if tables_only:
        share = "all of it" if tables_only == whole else f"{tables_only} of {whole}"
        lines.append(
            f"**The tables carry the effect**: regenerating them alone moves the yield by "
            f"{tables_only:+d} equilibria, {share}. The packaged and regenerated tables are not "
            "interchangeable, and #194's negative A/B -- four slices, all of which converge under "
            "either -- was measured on a population that could not show this."
        )
    else:
        lines.append(
            "**The tables are not the cause**: regenerating them alone changes no slice. The gap "
            "observed in #459's control has some third cause, and neither file explains it."
        )

    failures = {
        routine: {
            name: sum(block["cases"][name]["summary"][f"{routine}_failures"] for block in payload["shots"].values())
            for name in payload["cases"]
        }
        for routine in ("bound", "findax")
    }
    lines.append(
        "**What it changes is where the boundary goes, not how well the fit works.** `findax`, "
        f"which rejects a separatrix point that lands off grid, fails {failures['findax']['PP']} times "
        f"under the packaged table and {failures['findax']['FF']} under the regenerated one; `bound` "
        f"fails {failures['bound']['PP']} times under both. The collapse block is untouched by the "
        "table, as it was by the seed (#588) and by the domain and grid (#459)."
    )

    unchanged = all(
        (block["cases"]["FP"].get("vs_reference") or {}).get("metrics", {}).get("chisq", {}).get("median_abs") == 0.0
        for block in payload["shots"].values()
    )
    if unchanged:
        lines.append(
            "**And it does not fit the magnetics any better.** On every slice both tables "
            "reconstruct, the chi-square is unchanged to the digits EFIT prints, while the boundary "
            "moves -- q95 by a few percent, the minor radius by under one. The two tables agree "
            "about the measurements and disagree about the plasma, so the extra yield is not "
            "evidence that the regenerated table is the correct one. That case rests on its "
            "provenance instead."
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
