#!/usr/bin/env python3

"""Audit every ``camera_visible`` product against the frame-selection rule.

``vaft.machine_mapping.camera_visible`` keeps only the frames between the
first and last non-dark frame (plus two of padding). This script checks, shot
by shot, that what the FileDB product and HSDS ``main`` hold is what the rule
selects today, and cross-checks the kept window against the plasma current.
It reads only; nothing is rebuilt or republished.

Per shot it compares four things:

stored
    The selection written into the FileDB product's
    ``ids_properties.comment`` (``parse_frame_selection``) and its frame times.
replication
    The product's ``metadata/replication.json``: state ``validated`` and the
    same ``product_sha256`` as the manifest means HSDS round-tripped exactly
    this product.
hsds
    The same comment read from HSDS through the lazy reader (no frames are
    downloaded), which must name the same retained range.
expected
    ``select_valid_frames`` recomputed from the raw BMPs under
    ``legacy/camera_visible/{shot}/``.

and the I_p window (``|I_p| > --ip-threshold``) from the shot's
``omas/diagnostics`` product. I_p is a cross-check only: the rule itself is
image-based.

Run on the server, where the raw frames live::

    ./audit_camera_visible_frames.py --root /srv/vest.filedb --jobs 12 --out audit.tsv
    ./audit_camera_visible_frames.py --root /srv/vest.filedb --shot 48909 --shot 48902
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import sys
from typing import Any

COLUMNS = (
    "shot", "header", "flags",
    "total_frames", "stored_rule", "stored_first", "stored_last", "stored_frames",
    "expected_first", "expected_last", "expected_dark_level", "expected_threshold",
    "replication", "hsds",
    "camera_t0", "camera_t1", "kept_t0", "kept_t1",
    "ip_max_kA", "ip_t_on", "ip_t_off", "error",
)


def _ip_window(root: Path, shot: int, threshold: float) -> tuple[float | None, float | None, float | None]:
    """``(t_on, t_off, max|I_p|)`` from the diagnostics product, or Nones."""
    import numpy as np

    path = root / "omas" / "diagnostics" / str(shot) / "output" / "diagnostics.json.gz"
    if not path.exists():
        return None, None, None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        magnetics = json.load(handle).get("magnetics", {})
    channels = magnetics.get("ip") or []
    if not channels:
        return None, None, None
    time = np.asarray(channels[0]["time"], dtype=float)
    current = np.abs(np.asarray(channels[0]["data"], dtype=float))
    above = time[current > threshold]
    peak = float(np.nanmax(current)) if current.size else None
    if above.size == 0:
        return None, None, peak
    return float(above[0]), float(above[-1]), peak


def _stored(product: Path) -> tuple[dict[str, Any] | None, int | None]:
    import h5py

    from vaft.machine_mapping.camera_visible import parse_frame_selection

    with h5py.File(product, "r") as handle:
        comment = handle["camera_visible/ids_properties/comment"][()]
        frames = len(handle["camera_visible/channel/0/detector/0/frame"])
    if isinstance(comment, bytes):
        comment = comment.decode("utf-8", "replace")
    return parse_frame_selection(str(comment)), frames


def _replication(stage_dir: Path) -> str:
    try:
        manifest = json.loads((stage_dir / "metadata" / "manifest.json").read_text())
        record = json.loads((stage_dir / "metadata" / "replication.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "missing"
    if record.get("state") != "validated":
        return f"state={record.get('state')}"
    if record.get("product_sha256") != manifest.get("output", {}).get("sha256"):
        return "stale"
    return "validated"


def _hsds(shot: int, stored: dict[str, Any] | None) -> str:
    from vaft.database.lazy_ods import open_ods
    from vaft.machine_mapping.camera_visible import parse_frame_selection

    try:
        with open_ods(shot, "main", ids="camera_visible") as ods:
            remote = parse_frame_selection(str(ods["camera_visible.ids_properties.comment"]))
    except Exception as error:  # noqa: BLE001 - reported per shot, never fatal
        return f"missing:{type(error).__name__}"
    if stored is None or remote is None:
        return "unparsed"
    keys = ("first_retained", "last_retained", "total_frames", "rule")
    return "ok" if all(remote[k] == stored[k] for k in keys) else "mismatch"


def audit_shot(root: Path, shot: int, ip_threshold: float, check_hsds: bool, recompute: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"shot": shot}
    flags: list[str] = []
    shot_dir = root / "legacy" / "camera_visible" / str(shot)
    stage_dir = root / "omas" / "camera_visible" / str(shot)
    product = stage_dir / "output" / "camera_visible.h5"
    try:
        from vaft.machine_mapping.camera_visible import (
            CameraFrameSelectionError,
            _load_raw_frame,
            _parse_bmp_header,
            frame_time_ms,
            select_valid_frames,
        )

        header_path = shot_dir / f"{shot}_bmp.txt"
        header = _parse_bmp_header(header_path)
        # Both layouts say `Type: GX-8`; only the original export has the
        # `Top Frame,...` rows, the BatchConv2 re-export has `TopFrame:` keys.
        text = header_path.read_text(errors="replace")
        row["header"] = "original" if "\nTop Frame," in text else "reexport"
        total = header.total_frames
        row["total_frames"] = total

        def at(index: int) -> float:
            return frame_time_ms(index, total, header.start_time_ms, header.end_time_ms) / 1000.0

        row["camera_t0"], row["camera_t1"] = round(at(0), 5), round(at(total - 1), 5)

        stored = None
        if product.exists():
            stored, row["stored_frames"] = _stored(product)
            if stored is None:
                flags.append("comment_unparsed")
            else:
                row["stored_rule"], row["stored_first"], row["stored_last"] = (
                    stored["rule"], stored["first_retained"], stored["last_retained"])
                if (stored["first_retained"], stored["last_retained"]) == (0, total - 1):
                    flags.append("full_range_retained")
            row["replication"] = _replication(stage_dir)
            if row["replication"] != "validated":
                flags.append("replication_not_validated")
            if check_hsds:
                row["hsds"] = _hsds(shot, stored)
                if row["hsds"] != "ok":
                    flags.append(f"hsds_{row['hsds'].split(':')[0]}")
        else:
            flags.append("no_product")

        kept = None
        if recompute:
            frames = [_load_raw_frame(shot_dir, shot, index) for index in range(total)]
            try:
                selection = select_valid_frames(frames)
            except CameraFrameSelectionError:
                flags.append("expected_all_dark")
            else:
                # Compare stored frames with stored frames: the comment names
                # the first/last frame present, not the padded interval.
                kept = (selection.first_retained, selection.last_retained)
                row["expected_first"], row["expected_last"] = kept
                row["expected_dark_level"] = round(selection.dark_level, 1)
                row["expected_threshold"] = round(selection.threshold, 1)
                if stored is not None and (stored["first_retained"], stored["last_retained"]) != kept:
                    flags.append("nonconformant")
                if not product.exists():
                    flags.append("product_missing_but_selectable")
        if kept is None and stored is not None:
            kept = (stored["first_retained"], stored["last_retained"])
        if kept is not None:
            row["kept_t0"], row["kept_t1"] = round(at(kept[0]), 5), round(at(kept[1]), 5)

        t_on, t_off, peak = _ip_window(root, shot, ip_threshold)
        row["ip_max_kA"] = None if peak is None else round(peak / 1e3, 2)
        row["ip_t_on"] = None if t_on is None else round(t_on, 5)
        row["ip_t_off"] = None if t_off is None else round(t_off, 5)
        if peak is None:
            flags.append("no_ip")
        elif t_on is None:
            if kept is not None:
                flags.append("no_plasma_but_frames")
        else:
            if t_off < row["camera_t0"] or t_on > row["camera_t1"]:
                flags.append("camera_misses_ip_window")
            elif kept is not None and (row["kept_t1"] < t_on or row["kept_t0"] > t_off):
                flags.append("frames_miss_ip_window")
    except Exception as error:  # noqa: BLE001 - one bad shot must not stop the audit
        row["error"] = f"{type(error).__name__}: {error}"[:200]
        flags.append("error")
    row["flags"] = ",".join(flags) or "ok"
    return row


def discover(root: Path) -> list[int]:
    shots: set[int] = set()
    for tree in (root / "legacy" / "camera_visible", root / "omas" / "camera_visible"):
        if tree.is_dir():
            shots.update(int(p.name) for p in tree.iterdir() if p.is_dir() and p.name.isdigit())
    return sorted(shots)


def _run(arguments: tuple[Path, int, float, bool, bool]) -> dict[str, Any]:
    return audit_shot(*arguments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="FileDB root, e.g. /srv/vest.filedb")
    parser.add_argument("--shot", type=int, action="append", help="Audit only these shots")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("camera_visible_audit.tsv"))
    parser.add_argument("--ip-threshold", type=float, default=5e3, help="|I_p| in A that marks plasma")
    parser.add_argument("--no-hsds", action="store_true", help="Skip the HSDS comparison")
    parser.add_argument("--no-recompute", action="store_true", help="Skip recomputing from raw BMPs")
    args = parser.parse_args(argv)

    shots = args.shot or discover(args.root)
    work = [(args.root, s, args.ip_threshold, not args.no_hsds, not args.no_recompute) for s in shots]
    counts: Counter[str] = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for done, row in enumerate(pool.map(_run, work, chunksize=4), start=1):
                writer.writerow(row)
                handle.flush()
                counts.update(row["flags"].split(","))
                if done % 200 == 0:
                    print(f"{done}/{len(work)}", file=sys.stderr, flush=True)

    summary = {"shots": len(work), "flags": dict(counts.most_common())}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
