"""Controlled EFIT A/B: the same constraints, the same binary, two Green tables (issue #194).

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install python workflow/efit_tables/ab_efit_table.py \\
        --eddy-ods /scratch/39915_eddy_noeq.json --shot 39915 \\
        --table-a vaft/data/efit/ --table-b /scratch/tables/legacy-39915/ \\
        --times 0.316 0.319 0.323 0.325 --output /scratch/ab --markdown /scratch/ab/ab.md

Everything is held fixed except the table directory the k-files point at:
one constraints ODS is built from the eddy product (with table A's
``mhdin.dat`` supplying the machine counts, which both tables share), the
k-files for arm B are written from a copy whose ``INPUT_DIR``/``TABLE_DIR``
are rewritten, and the tool refuses to run if the two k-file sets differ in
any other line.  One ``efit`` executable, resolved once from ``EFITHOME``,
runs both arms with ``EFITScientificConfig()``.  Per slice it reports
convergence, chi-square, iterations and the final GS error from the log,
axis and boundary from the g-file, and the a-file scalars, and sets both arms
beside the stored pipeline reference a/g-files for the same shot.

No EFUND or EFIT setting is tuned here; the table is the only variable.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import shutil
import sys
import time as _clock
from pathlib import Path
from typing import Any

import numpy as np
from omas import load_omas_json

from vaft.code.efit import generate_constraints_ods
from vaft.code.efit.config import EFITScientificConfig
from vaft.code.efit.efund import table_identity
from vaft.code.efit.magnetic import EFITConfig, prepare_efit_inputs, resolved_efit_configuration, run_efit
from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
from vaft.data import read_aeqdsk
from vaft.data.eqdsk import read_geqdsk

DEFAULT_UNCERTAINTY = [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]
DEFAULT_WEIGHTING = [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]
DEFAULT_TIMES = [0.316, 0.319, 0.323, 0.325]
DEFAULT_WINDOW = 0.0005

_ITERATION = re.compile(r"\bt=\s*(\d+)\s+it=\s*(\d+)\s+chi2=\s*([0-9.E+-]+).*?err=\s*([0-9.E+-]+)")
_DIR_LINE = re.compile(r"^\s*(INPUT_DIR|TABLE_DIR)\s*=", re.IGNORECASE)
A_SCALARS = ("chisq", "ipmhd", "betap", "betat", "li", "q95", "qstar", "rm", "zm", "rcntr", "zcntr", "rcurrt", "zcurrt", "aminor", "elong", "cdflux", "condno", "terror")
#: EFIT stores TABLE_DIR in a fixed-length character variable; a longer path
#: is silently truncated and the run fails opening lim.dat.
TABLE_DIR_MAX_LENGTH = 100
#: Read from TABLE_DIR by EFIT but not produced by EFUND: the limiter outline.
LIMITER_FILE = "lim.dat"


def _table_dir(path: Path) -> str:
    # absolute(), not resolve(): a symlink is the documented way to give EFIT
    # a path under its 100-character limit, and resolving it would undo that.
    text = str(path.expanduser().absolute())
    return text if text.endswith("/") else text + "/"


def iterations_from_log(text: str) -> list[dict[str, Any]]:
    """Per-slice iteration count, last chi2 and last GS error from EFIT's terminal log.

    One ``it=`` line per outer iteration, the counter restarting at 1 on every
    slice; a boundary-finder error is attributed to the block it follows.
    """
    blocks: list[dict[str, Any]] = []
    current: list[tuple[int, float, float]] = []

    def flush() -> None:
        if current:
            blocks.append(
                {
                    "iterations_n": max(it for it, _c, _e in current),
                    "chi2_log": current[-1][1],
                    "gs_error_log": current[-1][2],
                    "bound_error": False,
                }
            )

    for line in text.splitlines():
        found = _ITERATION.search(line)
        if found:
            if int(found.group(2)) == 1:
                flush()
                current = []
            current.append((int(found.group(2)), float(found.group(3)), float(found.group(4))))
            continue
        if "ERROR in bound" in line and current:
            flush()
            blocks[-1]["bound_error"] = True
            current = []
    flush()
    return blocks


def _key_us(name: str) -> int:
    key = name.split(".", 1)[1]
    ms, _, us = key.partition("_")
    return int(ms) * 1000 + (int(us) if us else 0)


def afile_metrics(path: Path) -> dict[str, Any]:
    a = read_aeqdsk(path)
    row: dict[str, Any] = {"afile": path.name, "time_ms": float(a.time_ms), "jflag": int(a.jflag), "lflag": int(a.lflag)}
    for name in A_SCALARS:
        value = a.scalars.get(name)
        row[name] = float(value) if value is not None else None
    return row


def gfile_metrics(path: Path) -> dict[str, Any]:
    g = read_geqdsk(path)
    rb = np.asarray(g.get("RBBBS", []), dtype=float).reshape(-1)
    zb = np.asarray(g.get("ZBBBS", []), dtype=float).reshape(-1)
    return {
        "gfile": path.name,
        "rmaxis": float(g.get("RMAXIS", np.nan)),
        "zmaxis": float(g.get("ZMAXIS", np.nan)),
        "simag": float(g.get("SIMAG", np.nan)),
        "sibry": float(g.get("SIBRY", np.nan)),
        "current": float(g.get("CURRENT", np.nan)),
        "nbbbs": int(rb.size),
        "r_lcfs_min": float(rb.min()) if rb.size else None,
        "r_lcfs_max": float(rb.max()) if rb.size else None,
        "z_lcfs_min": float(zb.min()) if zb.size else None,
        "z_lcfs_max": float(zb.max()) if zb.size else None,
    }


def _slice_key(path: Path) -> str:
    return path.name.split(".", 1)[1]


def collect_arm(workdir: Path, shot: int, kfiles: list[Path], stdout: str) -> list[dict[str, Any]]:
    log_path = workdir / "run_efit.out"
    text = log_path.read_text(errors="replace") if log_path.exists() else stdout
    progress = iterations_from_log(text)
    keys = [_slice_key(path) for path in sorted(kfiles, key=lambda p: _key_us(p.name))]
    by_key = dict(zip(keys, progress))
    rows = []
    for key in keys:
        row: dict[str, Any] = {"key": key, "kfile": f"k0{shot}.{key}", "afile": None, "gfile": None, "jflag": 0}
        afile = workdir / f"a0{shot}.{key}"
        gfile = workdir / f"g0{shot}.{key}"
        if afile.is_file():
            row.update(afile_metrics(afile))
        if gfile.is_file():
            row.update(gfile_metrics(gfile))
        row.update(by_key.get(key, {}))
        rows.append(row)
    return rows


def rewrite_table_dir(ods, table_dir: str) -> None:
    parameters = ods["equilibrium.code.parameters"]
    for index in range(len(parameters["time_slice"])):
        parameters[f"time_slice.{index}.IN1.INPUT_DIR"] = table_dir
        parameters[f"time_slice.{index}.IN1.TABLE_DIR"] = table_dir


def kfile_diff(dir_a: Path, dir_b: Path, shot: int) -> dict[str, Any]:
    """The lines in which the two arms' k-files differ; must be the two directory lines only."""
    files_a = {p.name: p for p in dir_a.glob(f"k0{shot}.*")}
    files_b = {p.name: p for p in dir_b.glob(f"k0{shot}.*")}
    report: dict[str, Any] = {"names_equal": sorted(files_a) == sorted(files_b), "files": {}}
    for name in sorted(set(files_a) | set(files_b)):
        if name not in files_a or name not in files_b:
            report["files"][name] = {"present": [name in files_a, name in files_b]}
            continue
        lines_a = files_a[name].read_text(errors="replace").splitlines()
        lines_b = files_b[name].read_text(errors="replace").splitlines()
        changed = [line for line in difflib.unified_diff(lines_a, lines_b, lineterm="", n=0) if line[:1] in "+-" and not line.startswith(("+++", "---"))]
        other = [line for line in changed if not _DIR_LINE.match(line[1:])]
        removed = [line for line in changed if line.startswith("-")]
        report["files"][name] = {"changed_lines": len(removed), "non_directory_changes": other[:10]}
    report["only_directory_lines_differ"] = report["names_equal"] and all(
        not entry.get("non_directory_changes") and entry.get("changed_lines", 0) == 2 for entry in report["files"].values()
    )
    return report


def run_arm(name: str, source, *, shot: int, times: list[float], table_dir: str, workdir: Path, efit: str, scientific: EFITScientificConfig, stack_size_kb: int | None) -> dict[str, Any]:
    ods = copy.deepcopy(source)
    rewrite_table_dir(ods, table_dir)
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    config = EFITConfig(
        executable=efit,
        workdir=workdir,
        shot=shot,
        times=list(times),
        args=("129",),
        npprime=scientific.profile.kppcur,
        nffprime=scientific.profile.kffcur,
        stack_size_kb=stack_size_kb,
        provenance={"arm": name, "table_dir": table_dir},
    )
    inputs = prepare_efit_inputs(ods, config)
    started = _clock.perf_counter()
    result = run_efit(inputs, config)
    seconds = _clock.perf_counter() - started
    return {
        "arm": name,
        "table_dir": table_dir,
        "table": table_identity(table_dir),
        "workdir": str(workdir),
        "returncode": result.returncode,
        "status": result.status,
        "reason": getattr(result, "reason", ""),
        "seconds": seconds,
        "afiles": len(result.afiles),
        "gfiles": len(result.gfiles),
        "kfiles": [path.name for path in inputs.kfiles],
        "configuration": resolved_efit_configuration(config),
        "run_manifest": json.loads(Path(inputs.manifest).read_text(encoding="utf-8")) if inputs.manifest else None,
        "slices": collect_arm(workdir, shot, list(inputs.kfiles), result.stdout),
    }


def reference_rows(table_a: Path, shot: int) -> dict[str, dict[str, Any]]:
    rows = {}
    for afile in sorted(table_a.glob(f"a0{shot}.*")):
        key = _slice_key(afile)
        row = afile_metrics(afile)
        gfile = table_a / f"g0{shot}.{key}"
        if gfile.is_file():
            row.update(gfile_metrics(gfile))
        rows[key] = row
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"
    return str(value)


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# EFIT table A/B", ""]
    tc = payload["toolchain"]
    for role, identity in tc.items():
        if identity:
            lines.append(f"- {role}: `{identity['path']}` sha256 `{identity['sha256'][:12]}` build `{identity.get('build_revision')}`")
    lines.append(f"- constraints: shot {payload['shot']}, times {payload['times']}, window ±{payload['average_window'] * 1e3:.2f} ms, fit 0")
    lines.append(f"- k-files differ only in the two directory lines: **{payload['kfile_diff']['only_directory_lines_differ']}**")
    lines.append("")
    for arm in payload["arms"]:
        table = arm["table"]
        lines.append(f"## Arm {arm['arm']}: `{arm['table_dir']}`")
        lines.append("")
        lines.append(f"- table provenance: {table['provenance']}; identity `{table.get('identity')}`; mhdin sha256 `{(table.get('mhdin_sha256') or '')[:12]}`")
        lines.append(f"- efit exit {arm['returncode']} ({arm['status']}) in {arm['seconds']:.0f} s; {arm['afiles']} a-files, {arm['gfiles']} g-files")
        lines.append("")
    columns = ("jflag", "lflag", "chisq", "terror", "iterations_n", "gs_error_log", "bound_error", "rmaxis", "zmaxis", "rm", "zm", "rcurrt", "zcurrt", "r_lcfs_min", "r_lcfs_max", "z_lcfs_min", "z_lcfs_max", "aminor", "elong", "betap", "li", "q95", "cdflux", "ipmhd", "condno")
    keys = sorted({row["key"] for arm in payload["arms"] for row in arm["slices"]}, key=lambda k: int(k.replace("_", "")))
    for key in keys:
        lines.append(f"## Slice {key}")
        lines.append("")
        header = ["metric"] + [arm["arm"] for arm in payload["arms"]] + (["stored reference"] if key in payload["reference"] else [])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for column in columns:
            cells = [column]
            for arm in payload["arms"]:
                row = next((r for r in arm["slices"] if r["key"] == key), {})
                cells.append(_fmt(row.get(column)))
            if key in payload["reference"]:
                cells.append(_fmt(payload["reference"][key].get(column)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Decision inputs")
    lines.append("")
    for row in payload["decision"]:
        lines.append(f"- {row}")
    return "\n".join(lines) + "\n"


def decision_lines(payload: dict[str, Any]) -> list[str]:
    out = []
    arms = {arm["arm"]: {row["key"]: row for row in arm["slices"]} for arm in payload["arms"]}
    a_rows, b_rows = arms.get("A", {}), arms.get("B", {})
    for key in sorted(set(a_rows) | set(b_rows), key=lambda k: int(k.replace("_", ""))):
        ra, rb = a_rows.get(key, {}), b_rows.get(key, {})
        chi_a, chi_b = ra.get("chisq"), rb.get("chisq")
        shift = None
        if all(isinstance(v, float) for v in (ra.get("rmaxis"), rb.get("rmaxis"), ra.get("zmaxis"), rb.get("zmaxis"))):
            shift = float(np.hypot(rb["rmaxis"] - ra["rmaxis"], rb["zmaxis"] - ra["zmaxis"]))
        rel = None
        if isinstance(chi_a, float) and isinstance(chi_b, float) and chi_a:
            rel = (chi_b - chi_a) / abs(chi_a)
        out.append(
            f"slice {key}: jflag A {ra.get('jflag')} → B {rb.get('jflag')}; chisq A {_fmt(chi_a)} → B {_fmt(chi_b)}"
            + (f" ({rel:+.1%})" if rel is not None else "")
            + (f"; axis shift {shift * 100:.2f} cm" if shift is not None else "")
            + f"; iterations A {ra.get('iterations_n')} → B {rb.get('iterations_n')}"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eddy-ods", required=True, type=Path, help="OMAS JSON of the shot's eddy-stage product (equilibrium stripped or not)")
    parser.add_argument("--shot", type=int, default=39915)
    parser.add_argument("--table-a", required=True, type=Path)
    parser.add_argument("--table-b", required=True, type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--average-window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--efit-home", default=None)
    parser.add_argument("--efit", default=None, help="explicit efit executable (wins over EFITHOME)")
    parser.add_argument("--stack-size-kb", type=int, default=32768)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())
    resolved = resolve_toolchain(efit_executable=args.efit)
    if resolved.get("efit") is None:
        print("no efit executable: set EFITHOME or pass --efit", file=sys.stderr)
        return 2
    efit = str(resolved["efit"])
    identities = toolchain_identities(resolved)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    table_a, table_b = _table_dir(args.table_a), _table_dir(args.table_b)
    for label, table_dir in (("A", table_a), ("B", table_b)):
        if len(table_dir) > TABLE_DIR_MAX_LENGTH:
            print(
                f"table {label} path is {len(table_dir)} characters; EFIT truncates TABLE_DIR at "
                f"{TABLE_DIR_MAX_LENGTH} and then fails to open {LIMITER_FILE}. Use a shorter path (a symlink will do).",
                file=sys.stderr,
            )
            return 2
    # The limiter outline is an EFIT input read from TABLE_DIR, not an EFUND
    # product; a generated table directory borrows A's so the arms differ in
    # Green tables only.
    if not (Path(table_b) / LIMITER_FILE).is_file():
        if not (Path(table_a) / LIMITER_FILE).is_file():
            print(f"neither table directory has {LIMITER_FILE}", file=sys.stderr)
            return 2
        shutil.copy(Path(table_a) / LIMITER_FILE, Path(table_b) / LIMITER_FILE)
        print(f"copied {LIMITER_FILE} from table A into table B (EFIT reads it from TABLE_DIR; EFUND does not write it)")
    times = [float(t) for t in args.times]

    print(f"loading {args.eddy_ods} ...")
    source = load_omas_json(str(args.eddy_ods), consistency_check=False)
    if "equilibrium" in source:
        del source["equilibrium"]
    source["equilibrium.time"] = np.asarray(times)
    constraints_dir = output / "constraints"
    shutil.rmtree(constraints_dir, ignore_errors=True)
    constraints_dir.mkdir(parents=True)
    started = _clock.perf_counter()
    generate_constraints_ods(
        source, args.shot, str(constraints_dir), table_a, times, list(DEFAULT_UNCERTAINTY), list(DEFAULT_WEIGHTING),
        broken=[], fit=0, average_window=args.average_window,
    )
    constraints_seconds = _clock.perf_counter() - started
    print(f"constraints built in {constraints_seconds:.0f} s")

    scientific = EFITScientificConfig()
    arms = []
    for name, table_dir in (("A", table_a), ("B", table_b)):
        print(f"arm {name}: {table_dir}")
        arms.append(run_arm(name, source, shot=args.shot, times=times, table_dir=table_dir, workdir=output / f"arm_{name}", efit=efit, scientific=scientific, stack_size_kb=args.stack_size_kb))
        print(f"  exit {arms[-1]['returncode']} in {arms[-1]['seconds']:.0f} s, {arms[-1]['afiles']} a-files")
    diff = kfile_diff(output / "arm_A", output / "arm_B", args.shot)
    if not diff["only_directory_lines_differ"]:
        print("k-files differ beyond the two directory lines; the A/B is not controlled", file=sys.stderr)
        print(json.dumps(diff, indent=2)[:4000], file=sys.stderr)
    payload = {
        "shot": args.shot,
        "times": times,
        "average_window": args.average_window,
        "eddy_ods": str(args.eddy_ods),
        "toolchain": identities,
        "scientific": scientific.to_dict(),
        "scientific_sha256": scientific.sha256,
        "constraints_seconds": constraints_seconds,
        "kfile_diff": diff,
        "arms": arms,
        "reference": reference_rows(Path(table_a), args.shot),
    }
    payload["decision"] = decision_lines(payload)
    (output / "ab_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(payload)
    (args.markdown or output / "ab_report.md").write_text(text, encoding="utf-8")
    print("\n".join(payload["decision"]))
    return 0 if diff["only_directory_lines_differ"] else 1


if __name__ == "__main__":
    sys.exit(main())
