#!/usr/bin/env python3
"""Optionally run EFIT and collect gfile outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

from vaft.code.efit import EFITConfig, EFITInputs, run_efit


LOGGER = logging.getLogger("vaft.run_efit_reconstruction")


def _bool(text: str) -> bool:
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case(path: Path) -> str:
    name = path.name[1:] if path.name[:1].lower() in {"k", "g", "a", "m"} else path.name
    if name.lower().endswith(".nc"):
        name = name[:-3]
    return name


def _artifact_payload(workdir: Path, shot: int, status: str, parse_errors=()) -> dict:
    cases: dict[str, dict] = {}
    for kind, subdir in (("kfile", "kfile"), ("gfile", "gfile"), ("mfile", "mfile"), ("afile", "afile")):
        directory = workdir / subdir
        if not directory.exists():
            continue
        for path in sorted(directory.glob(f"{kind[0]}0{shot}.*")):
            if path.is_file():
                cases.setdefault(_case(path), {})[kind] = {
                    "path": str(path), "sha256": _sha256(path), "size": path.stat().st_size
                }
    logs = []
    for path in sorted(workdir.rglob("*")):
        if path.is_file() and (path.suffix == ".log" or path.name in {"run_efit.out", "run_efit.err"}):
            logs.append({"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size})
    for artifacts in cases.values():
        if status.startswith("skipped"):
            artifacts["disposition"] = "skipped"
        elif status.startswith("failed"):
            artifacts["disposition"] = "runtime_failed"
        elif "gfile" in artifacts:
            artifacts["disposition"] = "collected"
        else:
            artifacts["disposition"] = "missing_gfile"
    return {
        "schema_version": 1, "shot": shot, "status": status, "cases": cases,
        "logs": logs, "parse_errors": list(parse_errors),
    }


def _write_outputs(gfile_manifest: Path, status_file: Path, artifact_manifest: Path, workdir: Path, shot: int, status: str, gfiles: tuple[Path, ...] = (), parse_errors=()) -> None:
    gfile_manifest.parent.mkdir(parents=True, exist_ok=True)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    gfile_manifest.write_text("\n".join(str(path) for path in gfiles) + ("\n" if gfiles else ""), encoding="utf-8")
    status_file.write_text(status.rstrip() + "\n", encoding="utf-8")
    artifact_manifest.write_text(json.dumps(_artifact_payload(workdir, shot, status, parse_errors), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--artifact-manifest", required=True, type=Path, help="Structured EFIT artifact manifest.")
    parser.add_argument("--run", default="false", help="Whether to run EFIT.")
    parser.add_argument("--executable", default="", help="EFIT executable path.")
    parser.add_argument("--args", default="129", help="Whitespace-separated EFIT command arguments.")
    parser.add_argument("--timeout", default="", help="Optional EFIT timeout in seconds.")
    args = parser.parse_args()

    # force=True: vaft.database.raw installs a root handler at import time, which makes
    # basicConfig() a no-op without this, silently dropping our INFO logs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    if not args.kfile_manifest.exists():
        raise FileNotFoundError(f"Kfile manifest not found: {args.kfile_manifest}")

    kfiles = tuple(Path(line.strip()) for line in args.kfile_manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    workdir = args.kfile_manifest.parent.parent
    executable = Path(args.executable).expanduser() if args.executable else None

    if not _bool(args.run):
        _write_outputs(args.gfile_manifest, args.status, args.artifact_manifest, workdir, args.shot, f"skipped: efit.run=false; kfiles={len(kfiles)}")
        return 0
    if executable is None or not executable.exists():
        _write_outputs(args.gfile_manifest, args.status, args.artifact_manifest, workdir, args.shot, f"skipped: EFIT executable unavailable: {args.executable}; kfiles={len(kfiles)}")
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
            args.gfile_manifest, args.status, args.artifact_manifest, workdir, args.shot,
            f"failed: returncode={result.returncode}; marker={marker or 'none'}\n{result.stderr or result.stdout}",
        )
        return int(result.returncode or 1)

    if not result.gfiles:
        _write_outputs(args.gfile_manifest, args.status, args.artifact_manifest, workdir, args.shot, "completed_no_gfiles: returncode=0; gfiles=0")
        return 0
    gfiles = _stage_outputs(workdir, result.gfiles, "gfile")
    _stage_outputs(workdir, result.afiles, "afile")
    _stage_outputs(workdir, result.mfiles, "mfile")
    parse_status = f"; parse_errors={len(result.parse_errors)}" if result.parse_errors else ""
    _write_outputs(
        args.gfile_manifest, args.status, args.artifact_manifest, workdir, args.shot,
        f"completed: returncode=0; gfiles={len(gfiles)}{parse_status}", gfiles,
        result.parse_errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
