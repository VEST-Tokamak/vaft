"""Manifest a stable local copy and extract contiguous CGYRO CI records.

Copy native files from a read-only source first. This command never runs a solver
or copies restart checkpoints. Full source copies stay outside the repository.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vaft.code.gacode.cgyro import read_field_history
from vaft.code.gacode.cgyro.outputs import parse_grids


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture(source, destination, *, origin, window=(100.0, 104.0)):
    source, dest = Path(source), Path(destination)
    if (
        source.resolve() == dest.resolve()
        or source.resolve() in dest.resolve().parents
        or dest.resolve() in source.resolve().parents
    ):
        raise ValueError(
            "source and fixture destination must be separate non-overlapping directories"
        )
    history = read_field_history(source, time_window=window)
    if not np.all(np.diff(history.records) == 1):
        raise ValueError("fixture must use contiguous raw records")
    start, stop = int(history.records[0]), int(history.records[-1]) + 1
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "origin": origin,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_version": history.metadata["version"],
        "window": list(window),
        "raw_record_range": [start, stop],
        "extraction": "contiguous native Fortran records; no resampling; restart checkpoints excluded",
        "files": {},
    }
    nr, ntheta, nky, _ = history.raw.shape
    grid = parse_grids(source / "out.cgyro.grids")
    previous = (
        json.loads((dest / "manifest.json").read_text())
        if (dest / "manifest.json").is_file()
        else {}
    )
    if any(Path(name).name != name for name in previous.get("files", {})):
        raise ValueError("fixture manifest contains a nonlocal file name")
    shapes = {
        "bin.cgyro.kxky_phi": (nr, ntheta, nky),
        "bin.cgyro.ky_flux": (grid["n_species"], 3, grid["n_field"], nky),
        "bin.cgyro.freq": (2, nky),
    }
    for p in sorted(source.iterdir()):
        if not p.is_file() or p.name.startswith("bin.cgyro.restart"):
            continue
        if not (
            p.name.startswith("input.cgyro")
            or p.name
            in {
                "out.cgyro.time",
                "out.cgyro.grids",
                "out.cgyro.equilibrium",
                "out.cgyro.version",
                "out.cgyro.info",
            }
            or p.name in shapes
            or p.name == "record.json"
        ):
            continue
        before = p.stat()
        source_sha = sha(p)
        target = dest / p.name
        if target.is_symlink():
            raise ValueError("fixture output must not be a symbolic link")
        if p.name in shapes:
            dtype = (
                history.raw.dtype
                if "kxky" in p.name
                else np.dtype(
                    "float64" if history.raw.dtype == np.complex128 else "float32"
                )
            )
            count = int(np.prod(shapes[p.name]))
            with p.open("rb") as f:
                f.seek(start * count * dtype.itemsize)
                data = f.read((stop - start) * count * dtype.itemsize)
            if len(data) != (stop - start) * count * dtype.itemsize:
                raise ValueError(f"incomplete fixture source {p}")
            target.write_bytes(data)
        elif p.name == "out.cgyro.time":
            target.write_text(
                "".join(p.read_text().splitlines(keepends=True)[start:stop])
            )
        else:
            shutil.copy2(p, target)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source changed during capture: {p}")
        manifest["files"][p.name] = {
            "source_sha256": source_sha,
            "source_bytes": before.st_size,
            "fixture_sha256": sha(target),
            "fixture_bytes": target.stat().st_size,
        }
    for name in set(previous.get("files", {})) - set(manifest["files"]):
        (dest / name).unlink(missing_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--origin", required=True)
    p.add_argument("--window", nargs=2, type=float, default=(100.0, 104.0))
    a = p.parse_args()
    capture(a.source, a.destination, origin=a.origin, window=a.window)
