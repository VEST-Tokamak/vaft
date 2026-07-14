#!/usr/bin/env python3
"""Optionally run EFIT and collect gfile outputs."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from vaft.code.efit import EFITConfig, EFITInputs, run_efit


LOGGER = logging.getLogger("vaft.run_efit_reconstruction")


def _bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _write_outputs(gfile_manifest: Path, status_file: Path, status: str, gfiles: tuple[Path, ...] = ()) -> None:
    gfile_manifest.parent.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    gfile_manifest.write_text("\n".join(str(path) for path in gfiles) + ("\n" if gfiles else ""), encoding="utf-8")
    status_file.write_text(status.rstrip() + "\n", encoding="utf-8")


def _stage_outputs(workdir: Path, paths: tuple[Path, ...], subdir: str) -> tuple[Path, ...]:
    target_dir = workdir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in paths:
        source = Path(source)
        target = target_dir / source.name
        if source.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
        staged.append(target)
    return tuple(sorted(staged))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--kfile-manifest", required=True, type=Path, help="Input kfile manifest.")
    parser.add_argument("--gfile-manifest", required=True, type=Path, help="Output gfile manifest.")
    parser.add_argument("--status", required=True, type=Path, help="Output EFIT status file.")
    parser.add_argument("--run", default="false", help="Whether to run EFIT.")
    parser.add_argument("--executable", default="", help="EFIT executable path.")
    parser.add_argument("--args", default="129", help="Whitespace-separated EFIT command arguments.")
    parser.add_argument("--timeout", default="", help="Optional EFIT timeout in seconds.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.kfile_manifest.exists():
        raise FileNotFoundError(f"Kfile manifest not found: {args.kfile_manifest}")

    kfiles = tuple(Path(line.strip()) for line in args.kfile_manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    workdir = args.kfile_manifest.parent.parent
    executable = Path(args.executable).expanduser() if args.executable else None

    if not _bool(args.run):
        _write_outputs(args.gfile_manifest, args.status, f"skipped: efit.run=false; kfiles={len(kfiles)}")
        return 0
    if executable is None or not executable.exists():
        _write_outputs(args.gfile_manifest, args.status, f"skipped: EFIT executable unavailable: {args.executable}; kfiles={len(kfiles)}")
        return 0

    timeout = float(args.timeout) if str(args.timeout).strip() else None
    efit_args = [item for item in args.args.split() if item]
    config = EFITConfig(
        executable=str(executable),
        workdir=workdir,
        shot=args.shot,
        args=tuple(efit_args),
        timeout=timeout,
    )
    result = run_efit(EFITInputs(workdir=workdir, kfiles=kfiles), config)
    output_text = result.stdout + "\n" + result.stderr
    failure_markers = ("Invalid line in namelist", "Fortran runtime error", "Error termination")
    marker = next((text for text in failure_markers if text in output_text), None)
    if result.returncode != 0 or marker:
        _write_outputs(
            args.gfile_manifest,
            args.status,
            f"failed: returncode={result.returncode}; marker={marker or 'none'}\n{result.stderr or result.stdout}",
        )
        return int(result.returncode or 1)

    if not result.gfiles:
        _write_outputs(args.gfile_manifest, args.status, "completed_no_gfiles: returncode=0; gfiles=0")
        return 0
    gfiles = _stage_outputs(workdir, result.gfiles, "gfile")
    _stage_outputs(workdir, result.afiles, "afile")
    _stage_outputs(workdir, result.mfiles, "mfile")
    parse_status = f"; parse_errors={len(result.parse_errors)}" if result.parse_errors else ""
    _write_outputs(args.gfile_manifest, args.status, f"completed: returncode=0; gfiles={len(gfiles)}{parse_status}", gfiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
