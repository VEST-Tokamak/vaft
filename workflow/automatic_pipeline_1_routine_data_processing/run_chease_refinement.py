#!/usr/bin/env python3
"""Run CHEASE refinement for EFIT g-files and collect refined g-files."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from vaft.code.chease import CHEASEConfig, find_chease_executable, prepare_chease_inputs, run_chease


LOGGER = logging.getLogger("vaft.run_chease_refinement")


def _bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def _read_manifest(path: Path) -> tuple[Path, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Input gfile manifest not found: {path}")
    return tuple(Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_outputs(manifest: Path, status: Path, refined: tuple[Path, ...], status_text: str) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    status.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in refined) + ("\n" if refined else ""), encoding="utf-8")
    status.write_text(status_text.rstrip() + "\n", encoding="utf-8")


def _time_label(path: Path) -> str:
    parts = path.name.split(".")
    return parts[-1] if len(parts) > 1 else path.stem


def _write_runs_summary(output_dir: Path, shot: int, executable: str, gfiles: tuple[Path, ...], refined: tuple[Path, ...], records: list[dict]) -> None:
    """``chease_runs.json`` must exist on every exit path, including the skip
    paths where CHEASE never runs, so the ``chease`` FileDB stage always has a
    summary to read -- the same reasoning as ``_write_plot_manifest``.
    """
    summary = {
        "shot": int(shot),
        "executable": executable,
        "input_gfiles": [str(path) for path in gfiles],
        "refined_gfiles": [str(path) for path in refined],
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chease_runs.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _write_plot_manifest(plots_dir: Path) -> Path:
    """List the staged comparison figures, even when there are none.

    Snakemake declares this file as a real output of the CHEASE rule, so it has
    to exist on every exit path -- including the skip paths where CHEASE never
    runs -- rather than only when figures happen to have been produced.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = plots_dir / "plot_refined_gfiles_generated.txt"
    figures = sorted(path for path in plots_dir.glob("*.png"))
    manifest.write_text(
        "".join(f"{path}\n" for path in figures), encoding="utf-8"
    )
    return manifest


def _stage_plot(result_workdir: Path, gfile: Path, plots_dir: Path) -> Path | None:
    source = result_workdir / "chease_comparison.png"
    if not source.exists():
        return None
    plots_dir.mkdir(parents=True, exist_ok=True)
    target = plots_dir / f"{_time_label(gfile)}.png"
    shutil.copy2(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--gfile-manifest", required=True, type=Path, help="Input EFIT gfile manifest.")
    parser.add_argument("--output", required=True, type=Path, help="Output refined gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Output CHEASE status file.")
    parser.add_argument("--run", default="true", help="Whether to run CHEASE.")
    parser.add_argument("--executable", default="", help="CHEASE executable path.")
    parser.add_argument("--timeout", default="", help="Optional per-gfile timeout in seconds.")
    parser.add_argument("--target-psin", default=0.993, type=float, help="Boundary contour psin for EXPEQ.")
    parser.add_argument("--relax", default=0.5, type=float, help="CHEASE RELAX value.")
    parser.add_argument("--nideal", default=6, type=int, help="CHEASE NIDEAL value.")
    parser.add_argument("--nw", default=513, type=int, help="CHEASE NRBOX/NZBOX value.")
    parser.add_argument("--auto-cocos", default="true", help="Normalize signs to CHEASE COCOS-02 input convention.")
    parser.add_argument("--output-cocos", default="input", help="CHEASE output sign convention handling.")
    parser.add_argument("--preserve-boundary-limiter", default="true", help="Restore EFIT boundary/limiter in staged output.")
    parser.add_argument("--create-plot", default="true", help="Create comparison plots for refined gfiles.")
    parser.add_argument(
        "--plot-dir",
        default="",
        # Deliberately a string: `type=Path` turns both the empty default and an
        # explicit `--plot-dir ""` (what the shot_first layout passes) into
        # Path('.'), which is truthy and silently redirected every figure and the
        # manifest into the working directory.
        help="Canonical FileDB plot/ directory for the CHEASE comparison figures.",
    )
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    gfiles = _read_manifest(args.gfile_manifest)
    output_dir = args.output.parent
    work_root = output_dir / "work"
    # Issue #139: comparison figures belong in the canonical `chease/{shot}/plot/`
    # artifact, not in an ad hoc subdirectory of `output/`.
    plots_dir = Path(args.plot_dir) if args.plot_dir.strip() else output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    executable = Path(args.executable).expanduser() if args.executable else None
    config_probe = CHEASEConfig(executable=str(executable) if executable else None)
    resolved_executable = find_chease_executable(config_probe)
    if not _bool(args.run):
        _write_plot_manifest(plots_dir)
        _write_runs_summary(output_dir, args.shot, str(resolved_executable or args.executable), gfiles, (), [])
        _write_outputs(args.output, args.status, (), f"skipped: chease.run=false; input_gfiles={len(gfiles)}")
        return 0
    if resolved_executable is None:
        _write_plot_manifest(plots_dir)
        _write_runs_summary(output_dir, args.shot, args.executable, gfiles, (), [])
        _write_outputs(args.output, args.status, (), f"skipped: CHEASE executable unavailable: {args.executable}; input_gfiles={len(gfiles)}")
        return 0

    refined: list[Path] = []
    records: list[dict] = []
    timeout = float(args.timeout) if str(args.timeout).strip() else None
    for gfile in gfiles:
        if not gfile.exists():
            records.append({"input": str(gfile), "status": "missing_input"})
            continue

        run_workdir = work_root / gfile.name
        config = CHEASEConfig(
            executable=str(resolved_executable),
            workdir=run_workdir,
            target_psin=args.target_psin,
            relax=args.relax,
            nideal=args.nideal,
            nw=args.nw,
            auto_cocos=_bool(args.auto_cocos),
            output_cocos=args.output_cocos,
            preserve_boundary_limiter=_bool(args.preserve_boundary_limiter),
            create_plot=_bool(args.create_plot),
            timeout=timeout,
        )
        try:
            inputs = prepare_chease_inputs(gfile, config)
            result = run_chease(inputs, config)
            record = {
                "input": str(gfile),
                "workdir": str(run_workdir),
                "returncode": result.returncode,
                "refined_geqdsk": str(result.refined_geqdsk) if result.refined_geqdsk else "",
                "comparison": dict(result.comparison),
            }
            if result.returncode == 0 and result.refined_geqdsk is not None and result.refined_geqdsk.exists():
                staged = output_dir / gfile.name
                shutil.copy2(result.refined_geqdsk, staged)
                refined.append(staged)
                plot = _stage_plot(run_workdir, gfile, plots_dir)
                record["status"] = "completed"
                record["staged"] = str(staged)
                record["plot"] = str(plot) if plot else ""
            else:
                record["status"] = "failed"
                record["stderr"] = result.stderr
            records.append(record)
        except Exception as exc:
            LOGGER.exception("CHEASE failed for %s", gfile)
            records.append({"input": str(gfile), "workdir": str(run_workdir), "status": "error", "error": str(exc)})

    _write_runs_summary(output_dir, args.shot, str(resolved_executable), gfiles, tuple(refined), records)
    _write_plot_manifest(plots_dir)

    failed = [record for record in records if record.get("status") not in {"completed"}]
    if refined and failed:
        status_text = f"partial: refined_gfiles={len(refined)}; failed={len(failed)}"
    elif refined:
        status_text = f"completed: refined_gfiles={len(refined)}"
    else:
        status_text = f"failed: refined_gfiles=0; failed={len(failed)}"
    _write_outputs(args.output, args.status, tuple(refined), status_text)
    return 0 if refined else 1


if __name__ == "__main__":
    raise SystemExit(main())
