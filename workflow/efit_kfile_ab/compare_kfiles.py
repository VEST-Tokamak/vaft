"""Did that refactor change what EFIT is asked to solve? (#708)

    PYTHONPATH=$PWD python workflow/efit_kfile_ab/compare_kfiles.py emit /scratch/before
    # ... change the code ...
    PYTHONPATH=$PWD python workflow/efit_kfile_ab/compare_kfiles.py emit /scratch/after
    PYTHONPATH=$PWD python workflow/efit_kfile_ab/compare_kfiles.py compare /scratch/before /scratch/after

Issue #708 moves VEST's machine description out of ``vaft/code/efit`` a stage at
a time.  Most of those stages are supposed to change **nothing** -- the same
k-file, from the same data, written by machine-neutral code reading a
configuration instead of a literal.  "Supposed to" is not evidence, and the
k-file is the whole of what EFIT is asked to solve, so the evidence is the
bytes.

This runs no EFIT.  It builds the constraints and writes the k-files for the
reference discharges, which is the part a refactor can break, and compares two
such directories file by file.  A stage that claims to be behaviour-preserving
cites a run of this with every file identical; a stage that cannot be reports
which lines moved and why, using the per-line reader
``workflow/efit_tables/ab_efit_table.py`` already has.

The constraints ODS is compared too, separately: a stage may legitimately
change what is stored there -- dropping a field nothing reads, say -- while
leaving the k-file untouched, and conflating the two would hide both.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE_SET = REPOSITORY / "test" / "data" / "efit_reference_set.json"
BASELINE = REPOSITORY / "workflow" / "efit_numerics" / "baseline_termination.py"
PIPELINE = (
    REPOSITORY
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
    / "generate_constraints_ods.py"
)
AB_TABLE = REPOSITORY / "workflow" / "efit_tables" / "ab_efit_table.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_product(path: Path):
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


def emit(output: Path, *, shots: Sequence[int] | None = None, tstep: float, average_window: float) -> int:
    """Write the k-files and the constraints ODS for each reference discharge.

    The table directory is the packaged one because ``nfsum`` is read from the
    ``mhdin.dat`` inside it (``kfile._machine_count``): point this at a
    directory without one and the writer silently falls back to the channel
    count, which is a different experiment from the one being compared.
    """
    from vaft.code.efit import generate_constraints_ods, generate_kfile
    from vaft.code.efit.config import EFITScientificConfig
    from vaft.data.resources import data_path
    from vaft.omas.vacuum_magnetics import quality_gate
    from vaft.validation.efit_channels import decide_efit_channels, efit_probe_count
    from omas import load_omas_json

    baseline = _module(BASELINE, "baseline_termination")
    pipeline = _module(PIPELINE, "generate_constraints_ods_wrapper")

    reference = json.loads(REFERENCE_SET.read_text(encoding="utf-8"))
    products = {
        int(item["shot"]): item["files"]["product"]["path"]
        for item in reference["shots"]
        if item["files"]["product"] and item["files"]["product"]["exists"]
    }
    wanted = list(shots) if shots else sorted(products)

    tables = str(Path(data_path("efit")).resolve()) + "/"
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for shot in wanted:
        if shot not in products:
            print(f"{shot}: no packaged pre-EFIT product; skipped", file=sys.stderr)
            continue
        ods = _load_product(Path(data_path(products[shot])))
        times, _ = pipeline._select_times(ods, "auto", tstep, None, None)
        times = np.asarray(times, dtype=float)

        source = copy.deepcopy(ods)
        if "equilibrium" in source:
            del source["equilibrium"]
        source["equilibrium.time"] = times
        gated, _ = quality_gate(source, window=(float(times[0]), float(times[-1])))
        decisions = decide_efit_channels(gated, times, nbprobe=efit_probe_count(gated))

        workdir = output / f"shot_{shot}"
        workdir.mkdir(parents=True, exist_ok=True)
        generate_constraints_ods(
            gated,
            shot,
            str(workdir),
            tables,
            times,
            list(baseline.DEFAULT_UNCERTAINTY),
            list(baseline.DEFAULT_WEIGHTING),
            decisions=decisions,
            average_window=average_window,
        )
        constraints = workdir / f"{shot}_constraints.json"
        generate_kfile(
            load_omas_json(str(constraints), consistency_check=False),
            shot,
            save_dir=str(workdir),
            config=EFITScientificConfig(),
        )
        count = len(list((workdir / "kfile").glob("k0*")))
        written += count
        print(f"{shot}: {count} k-files", flush=True)
    print(f"{written} k-files under {output}")
    return 0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(before: Path, after: Path) -> int:
    """Byte-compare two emitted directories; return 0 only if the k-files match."""
    left = {p.relative_to(before): p for p in sorted(before.rglob("k0*"))}
    right = {p.relative_to(after): p for p in sorted(after.rglob("k0*"))}

    only_left = sorted(str(name) for name in set(left) - set(right))
    only_right = sorted(str(name) for name in set(right) - set(left))
    differing = sorted(
        str(name) for name in set(left) & set(right) if _digest(left[name]) != _digest(right[name])
    )
    identical = len(set(left) & set(right)) - len(differing)

    print(f"k-files: {len(left)} before, {len(right)} after, {identical} identical, {len(differing)} differing")
    for name in only_left:
        print(f"  only in {before.name}: {name}")
    for name in only_right:
        print(f"  only in {after.name}: {name}")

    if differing:
        # The per-line reader the table A/B already uses, so a difference is
        # reported the same way wherever it is found.
        ab = _module(AB_TABLE, "ab_efit_table")
        for name in differing[:5]:
            first, second = left[Path(name)], right[Path(name)]
            lines = [
                line
                for line in ab.difflib.unified_diff(
                    first.read_text(errors="replace").splitlines(),
                    second.read_text(errors="replace").splitlines(),
                    lineterm="",
                    n=0,
                )
                if line[:1] in "+-" and not line.startswith(("+++", "---"))
            ]
            print(f"  {name}:")
            for line in lines[:12]:
                print(f"      {line}")
        if len(differing) > 5:
            print(f"  ... and {len(differing) - 5} more")

    # Reported separately: a stage may change what is stored without changing
    # what EFIT is asked to solve.
    stored_left = {p.relative_to(before): p for p in sorted(before.rglob("*_constraints.json"))}
    stored_right = {p.relative_to(after): p for p in sorted(after.rglob("*_constraints.json"))}
    moved = [
        str(name)
        for name in sorted(set(stored_left) & set(stored_right))
        if _digest(stored_left[name]) != _digest(stored_right[name])
    ]
    if moved:
        print(f"constraints ODS changed for {len(moved)} shot(s): {', '.join(moved)}")
        for name in moved:
            before_keys = _parameter_keys(stored_left[Path(name)])
            after_keys = _parameter_keys(stored_right[Path(name)])
            gone = sorted(before_keys - after_keys)
            added = sorted(after_keys - before_keys)
            if gone:
                print(f"    keys removed: {', '.join(gone)}")
            if added:
                print(f"    keys added:   {', '.join(added)}")
    else:
        print("constraints ODS identical")

    ok = not differing and not only_left and not only_right
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def _parameter_keys(path: Path) -> set[str]:
    """Every ``IN1``/``INWANT`` key name stored in the constraints ODS."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def walk(node: Any, trail: tuple[str, ...] = ()) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if trail and trail[-1] in {"IN1", "INWANT"}:
                    found.add(f"{trail[-1]}.{key}")
                walk(value, trail + (str(key),))

    walk(payload)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    emitter = sub.add_parser("emit", help="write k-files for the reference discharges")
    emitter.add_argument("output", type=Path)
    emitter.add_argument("--shots", default=None, help="comma-separated; default: the whole reference set")
    emitter.add_argument("--tstep", type=float, default=0.001)
    emitter.add_argument("--average-window", type=float, default=0.0005)

    comparer = sub.add_parser("compare", help="byte-compare two emitted directories")
    comparer.add_argument("before", type=Path)
    comparer.add_argument("after", type=Path)

    args = parser.parse_args(argv)
    if args.command == "emit":
        shots = [int(value) for value in args.shots.split(",")] if args.shots else None
        return emit(
            args.output.expanduser(),
            shots=shots,
            tstep=args.tstep,
            average_window=args.average_window,
        )
    return compare(args.before.expanduser(), args.after.expanduser())


if __name__ == "__main__":
    raise SystemExit(main())
