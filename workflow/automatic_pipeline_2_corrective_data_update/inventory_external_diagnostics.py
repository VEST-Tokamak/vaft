#!/usr/bin/env python3

"""Locate scattered VEST soft X-ray and FAST-camera data and classify it.

Raw data for both diagnostics lives outside the FileDB, spread over several
directories that grew independently: acquisition dumps, per-request copies made
for collaborators, working copies inside analysis repositories, and a camera
export tree whose directory names carry an acquisition-rate suffix. Nothing
reads them the same way, and no two of them agree on a naming convention.

This module is the read-only half of the consolidation. It walks a set of
source roots, decides what every artifact is, and writes an inventory that
``consolidate_external_diagnostics.py`` turns into a move plan. It never
modifies, moves, or deletes anything.

The classification rules are pure functions over names and header text so they
can be tested without touching the 250 GB they were written for.

Run::

    ./inventory_external_diagnostics.py --output inventory.json
    ./inventory_external_diagnostics.py --output inventory.json --report -
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator, Sequence

# --------------------------------------------------------------------------
# Conventions shared with the consolidation and ingest steps
# --------------------------------------------------------------------------

#: Diagnostic keys that resolve under the canonical ``legacy/`` FileDB domain,
#: i.e. data a vaft machine mapping can read today.
MAPPED_DIAGNOSTICS = ("soft_x_rays", "camera_visible", "camera_visible_fluctuation")

#: Keys that live under ``unmapped/`` instead: real data, no reader yet.
UNMAPPED_DIAGNOSTICS = (
    "camera_visible_arranged",
    "camera_visible_mcf",
    "camera_ccd_2013",
    "hard_x_rays",
)

#: A camera acquisition at or above this rate is fluctuation-grade and is
#: reserved for the pending fluctuation routine (issue #161) rather than being
#: swept into routine IDS generation.
FLUCTUATION_FRAME_RATE_HZ = 50_000

#: Files that are export-tool residue rather than data.
JUNK_NAMES = frozenset({"Thumbs.db", ".DS_Store"})

#: Hash everything smaller than this so duplicate copies collapse to one
#: canonical source. Above it, size plus name is the identity we use; the
#: multi-gigabyte frame directories are not duplicated across roots.
MAX_HASH_BYTES = 100 * 1024 * 1024

_CAMERA_DIR_RE = re.compile(r"^(?P<shot>\d{3,6})(?P<suffix>[^/]*)$")
_CAMERA_FRAME_INDEXED_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d{8})\.bmp$", re.IGNORECASE)
_CAMERA_FRAME_ARRANGED_RE = re.compile(
    r"^(?P<stem>.+)_(?P<time_ms>\d+(?:\.\d+)?)_ms\.(?:bmp|png)$", re.IGNORECASE
)
_CAMERA_HEADER_RE = re.compile(r"^(?P<stem>.+)_bmp\.txt$", re.IGNORECASE)

_SXR_DIGITIZER_RE = re.compile(r"^digitizer_(?P<daq>\d+)_(?P<shot>\d+)\.csv$", re.IGNORECASE)
_HXR_DIGITIZER_RE = re.compile(
    r"^digitizer_hxr_(?P<variant>[A-Za-z]+)_(?P<daq>\d+)_(?P<shot>\d+)\.csv$", re.IGNORECASE
)
_MCF_RE = re.compile(r"^(?P<shot>\d{3,6})\.mcf$", re.IGNORECASE)
_CCD_DIR_RE = re.compile(
    r"^shot\s*#\s*(?P<shot>\d{3,6})(?![0-9])(?P<note>.*)$", re.IGNORECASE
)

_HEADER_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_ .]*)\s*:\s*(?P<value>.*)$")


# --------------------------------------------------------------------------
# Pure classifiers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraHeader:
    """The handful of GX-8 header fields consolidation and ingest care about."""

    frame_rate_hz: int | None = None
    frame_size: str | None = None
    frame_count: int | None = None
    top_frame: int | None = None
    bottom_frame: int | None = None
    shutter_speed: str | None = None
    recorded_at: str | None = None


def parse_camera_header(text: str) -> CameraHeader:
    """Extract the fields we key on from a ``{stem}_bmp.txt`` header.

    Parsed by field name rather than by line number. ``vaft`` reads this file
    positionally (``camera_visible.py`` pins ``Frames`` to line 16 and
    ``TopFrame`` to line 75), which is correct for the 150-line GX-8 headers it
    was written against but would silently mis-read a shorter variant. Here we
    only need a handful of values and would rather get ``None`` than a wrong
    number, so we look them up by key.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = _HEADER_FIELD_RE.match(line.strip())
        if match:
            fields.setdefault(match.group("key").strip(), match.group("value").strip())

    def _int(key: str) -> int | None:
        raw = fields.get(key)
        if raw is None:
            return None
        try:
            return int(float(raw.split()[0]))
        except (ValueError, IndexError):
            return None

    return CameraHeader(
        frame_rate_hz=_int("Frame_Rate"),
        frame_size=fields.get("Frame_Size"),
        frame_count=_int("Frames"),
        top_frame=_int("TopFrame"),
        bottom_frame=_int("BottomFrame"),
        shutter_speed=fields.get("ShutterSpeed"),
        recorded_at=fields.get("Rec_Time"),
    )


@dataclass(frozen=True)
class CameraDirectory:
    """What one directory of exported camera frames turned out to be."""

    shot: int | None
    suffix: str
    stem: str | None
    family: str
    diagnostic: str | None
    frame_count: int
    header_name: str | None
    header: CameraHeader
    junk: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.diagnostic is None


def classify_camera_directory(
    dir_name: str,
    member_names: Sequence[str],
    header_text: str | None = None,
) -> CameraDirectory:
    """Decide what a single exported-camera directory holds.

    Three families appear in the export tree:

    ``raw_indexed``
        ``{stem}_{frame:08d}.bmp`` straight off the camera. This is the only
        family ``vaft.machine_mapping.camera_visible`` can read, and the only
        one that preserves original frame indices.
    ``arranged``
        ``{stem}_{time_ms}_ms.bmp`` produced by the legacy ``bmp_arranger``
        tool: a dark-frame-rejected, reindexed subset. Derived output, so it is
        not eligible as fixture input for the fluctuation work.
    ``empty``
        A directory with no frames at all.

    A ``raw_indexed`` directory recorded at or above
    :data:`FLUCTUATION_FRAME_RATE_HZ` is classified as
    ``camera_visible_fluctuation`` and kept out of routine ingest.
    """
    members = [name for name in member_names if not name.startswith("._")]
    junk = tuple(sorted(name for name in members if name in JUNK_NAMES))
    members = [name for name in members if name not in JUNK_NAMES]

    dir_match = _CAMERA_DIR_RE.match(dir_name)
    shot = int(dir_match.group("shot")) if dir_match else None
    suffix = dir_match.group("suffix") if dir_match else ""

    header_name = next((name for name in members if _CAMERA_HEADER_RE.match(name)), None)
    header = parse_camera_header(header_text) if header_text else CameraHeader()

    indexed = [name for name in members if _CAMERA_FRAME_INDEXED_RE.match(name)]
    arranged = [name for name in members if _CAMERA_FRAME_ARRANGED_RE.match(name)]

    if indexed:
        family = "raw_indexed"
        stem = _CAMERA_FRAME_INDEXED_RE.match(indexed[0]).group("stem")
        frame_count = len(indexed)
    elif arranged:
        family = "arranged"
        stem = _CAMERA_FRAME_ARRANGED_RE.match(arranged[0]).group("stem")
        frame_count = len(arranged)
    else:
        family = "empty"
        stem = _CAMERA_HEADER_RE.match(header_name).group("stem") if header_name else None
        frame_count = 0

    diagnostic: str | None
    reason: str | None = None
    if shot is None:
        diagnostic, reason = None, f"directory name {dir_name!r} does not start with a shot number"
    elif family == "empty":
        diagnostic, reason = None, "no frames in directory"
    elif family == "arranged":
        diagnostic = "camera_visible_arranged"
    elif header_name is None:
        diagnostic, reason = None, "raw frames present but no _bmp.txt header to date them"
    elif header.frame_rate_hz is None:
        diagnostic, reason = None, (
            "header file is present but holds no parsable Frame_Rate; several of "
            "these turned out to be filesystem index records written over the "
            "header, not text"
        )
    elif header.frame_rate_hz >= FLUCTUATION_FRAME_RATE_HZ:
        diagnostic = "camera_visible_fluctuation"
    else:
        diagnostic = "camera_visible"

    return CameraDirectory(
        shot=shot,
        suffix=suffix,
        stem=stem,
        family=family,
        diagnostic=diagnostic,
        frame_count=frame_count,
        header_name=header_name,
        header=header,
        junk=junk,
        reason=reason,
    )


@dataclass(frozen=True)
class DigitizerFile:
    """One X-ray digitizer CSV, soft or hard."""

    diagnostic: str
    shot: int
    daq_label: str
    variant: str | None = None


def classify_digitizer_filename(name: str) -> DigitizerFile | None:
    """Map a digitizer CSV filename onto a diagnostic, shot, and DAQ label.

    Soft X-ray files are ``digitizer_{daq}_{shot}.csv``, which is the pattern
    ``vaft.machine_mapping.soft_x_rays`` resolves. Hard X-ray files from the
    same acquisition program carry an extra ``hxr_{variant}`` segment and feed
    a different (still unimplemented) diagnostic, so they must not be swept in
    with the soft X-ray set. The hard X-ray pattern is checked first because
    the soft pattern is the looser of the two.
    """
    hxr = _HXR_DIGITIZER_RE.match(name)
    if hxr:
        return DigitizerFile(
            diagnostic="hard_x_rays",
            shot=int(hxr.group("shot")),
            daq_label=hxr.group("daq"),
            variant=hxr.group("variant"),
        )
    sxr = _SXR_DIGITIZER_RE.match(name)
    if sxr:
        return DigitizerFile(
            diagnostic="soft_x_rays",
            shot=int(sxr.group("shot")),
            daq_label=sxr.group("daq"),
        )
    return None


def classify_mcf_filename(name: str) -> int | None:
    """Return the shot for a vendor ``{shot}.mcf`` container, else ``None``."""
    match = _MCF_RE.match(name)
    return int(match.group("shot")) if match else None


def classify_ccd_directory(name: str) -> tuple[int, str] | None:
    """Return ``(shot, note)`` for a 2013-era ``shot #NNNN`` CCD directory.

    The note matters: several shots were exported more than once under
    condition labels (``shot #5378_ECHplasma`` beside ``shot #5378_Swing-down``)
    and would collide if reduced to the shot number alone.
    """
    match = _CCD_DIR_RE.match(name.strip())
    if not match:
        return None
    return int(match.group("shot")), match.group("note").strip(" _-")


def normalised_camera_names(stem: str, shot: int, member_names: Iterable[str]) -> dict[str, str]:
    """Map each camera file onto its name under the bare-shot convention.

    Export directories carry an acquisition suffix on both the directory and
    every file inside it (``27134_50kHz/27134_50kHz_00000000.bmp``,
    ``36976-N001/36976-N001_bmp.txt``). ``camera_visible`` looks for
    ``{shot}/{shot}_{frame:08d}.bmp`` and ``{shot}/{shot}_bmp.txt``, so the
    suffix has to come off the filenames as well as the directory. The suffix
    itself is not lost: it is recorded in the per-shot provenance.

    Junk files map to nothing and are dropped.
    """
    renames: dict[str, str] = {}
    for name in member_names:
        if name.startswith("._") or name in JUNK_NAMES:
            continue
        if not name.startswith(stem):
            # Not one of this directory's own artifacts; carry it over as-is
            # rather than inventing a name for it.
            renames[name] = name
            continue
        renames[name] = f"{shot}{name[len(stem):]}"
    return renames


# --------------------------------------------------------------------------
# Filesystem walk
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """One inventoried artifact: a shot's directory, or a standalone file."""

    source: str
    kind: str
    diagnostic: str | None
    shot: int | None
    size_bytes: int
    file_count: int = 1
    detail: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    duplicate_of: str | None = None
    reason: str | None = None


def _iter_names(directory: Path) -> list[str]:
    try:
        return sorted(entry.name for entry in os.scandir(directory))
    except OSError:
        return []


def _directory_size(directory: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)``, ignoring junk and AppleDouble."""
    total = 0
    count = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.startswith("._") or name in JUNK_NAMES:
                continue
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
            count += 1
    return total, count


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _read_header(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def scan_camera_export_tree(root: Path) -> Iterator[Entry]:
    """Inventory an export tree of one directory per camera acquisition."""
    for name in _iter_names(root):
        if name.startswith("._"):
            continue
        directory = root / name
        if not directory.is_dir():
            continue
        members = _iter_names(directory)
        header_name = next((m for m in members if _CAMERA_HEADER_RE.match(m)), None)
        header_text = _read_header(directory / header_name) if header_name else None
        info = classify_camera_directory(name, members, header_text)
        size, files = _directory_size(directory)
        yield Entry(
            source=str(directory),
            kind="camera_directory",
            diagnostic=info.diagnostic,
            shot=info.shot,
            size_bytes=size,
            file_count=files,
            detail={
                "family": info.family,
                "suffix": info.suffix,
                "stem": info.stem,
                "frame_count": info.frame_count,
                "header_name": info.header_name,
                "header": asdict(info.header),
                "junk": list(info.junk),
            },
            reason=info.reason,
        )


def scan_digitizer_tree(root: Path) -> Iterator[Entry]:
    """Inventory soft and hard X-ray digitizer CSVs anywhere under ``root``."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "._"))]
        for name in sorted(filenames):
            if name.startswith("._") or not name.lower().endswith(".csv"):
                continue
            info = classify_digitizer_filename(name)
            if info is None:
                continue
            path = Path(dirpath) / name
            try:
                size = path.stat().st_size
            except OSError:
                continue
            yield Entry(
                source=str(path),
                kind="digitizer_csv",
                diagnostic=info.diagnostic,
                shot=info.shot,
                size_bytes=size,
                detail={"daq_label": info.daq_label, "variant": info.variant},
                sha256=_sha256(path) if size <= MAX_HASH_BYTES else None,
            )


def scan_mcf_tree(root: Path) -> Iterator[Entry]:
    """Inventory vendor ``.mcf`` containers, for which no reader exists yet."""
    for name in _iter_names(root):
        if name.startswith("._"):
            continue
        shot = classify_mcf_filename(name)
        if shot is None:
            continue
        path = root / name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        yield Entry(
            source=str(path),
            kind="mcf_file",
            diagnostic="camera_visible_mcf",
            shot=shot,
            size_bytes=size,
        )


def scan_ccd_tree(root: Path, *, max_depth: int = 4) -> Iterator[Entry]:
    """Inventory the 2013-era ``shot #NNNN`` CCD directories.

    Shot directories sit both at the top of this tree and one level down inside
    grouping folders (``Camera/``, ``camera back up/``), so the walk descends a
    little rather than assuming a flat layout. A directory that parses as a
    shot is not descended into: its contents are that shot's frames.
    """
    def _walk(directory: Path, depth: int) -> Iterator[Entry]:
        loose_bytes = 0
        loose_files = 0
        for name in _iter_names(directory):
            if name.startswith("._") or name in JUNK_NAMES:
                continue
            child = directory / name
            parsed = classify_ccd_directory(Path(name).stem if child.is_file() else name)

            if child.is_dir():
                if parsed is None:
                    if depth < max_depth:
                        yield from _walk(child, depth + 1)
                    else:
                        size, files = _directory_size(child)
                        loose_bytes += size
                        loose_files += files
                    continue
                shot, note = parsed
                size, files = _directory_size(child)
                yield Entry(
                    source=str(child),
                    kind="ccd_directory",
                    diagnostic="camera_ccd_2013",
                    shot=shot,
                    size_bytes=size,
                    file_count=files,
                    detail={"note": note, "original_name": name},
                )
                continue

            # Loose per-shot artifacts sit beside the frame directories:
            # `shot #5256.avi` and its `shot #5256_avi.txt` sidecar.
            try:
                size = child.stat().st_size
            except OSError:
                continue
            if parsed is None:
                loose_bytes += size
                loose_files += 1
                continue
            shot, note = parsed
            yield Entry(
                source=str(child),
                kind="ccd_file",
                diagnostic="camera_ccd_2013",
                shot=shot,
                size_bytes=size,
                detail={"note": note, "original_name": name, "suffix": child.suffix},
            )

        # Never let unclassified bulk vanish from the report: a directory that
        # held data we could not attribute to a shot is called out by name.
        if loose_files:
            yield Entry(
                source=str(directory),
                kind="unclassified_residue",
                diagnostic=None,
                shot=None,
                size_bytes=loose_bytes,
                file_count=loose_files,
                reason="files in this directory could not be attributed to a shot",
            )

    yield from _walk(root, 1)


def scan_geometry_tree(root: Path) -> Iterator[Entry]:
    """Inventory soft X-ray line-of-sight geometry as one indivisible asset."""
    if not root.is_dir():
        return
    size, files = _directory_size(root)
    yield Entry(
        source=str(root),
        kind="geometry_directory",
        diagnostic="soft_x_rays_geometry",
        shot=None,
        size_bytes=size,
        file_count=files,
    )


#: Which scanner to run over which root. Kept as data so a new source location
#: is one line here rather than a new branch in the walk.
SCANNERS = {
    "camera_export": scan_camera_export_tree,
    "digitizer": scan_digitizer_tree,
    "mcf": scan_mcf_tree,
    "ccd": scan_ccd_tree,
    "geometry": scan_geometry_tree,
}


DEFAULT_SOURCES: tuple[tuple[str, str], ...] = (
    ("camera_export", "/Volumes/hsyun_ssd/DATA/VEST_DATA/Camera/2_Exported_BMP"),
    ("mcf", "/Volumes/hsyun_ssd/DATA/VEST_DATA/Camera/1_MCF"),
    ("ccd", "/Volumes/hsyun_ssd/DATA/VEST_DATA/2. Diagnostics/3. CCD Camera"),
    ("digitizer", "/Volumes/hsyun_ssd/SXRacq"),
    ("digitizer", "/Volumes/hsyun_ssd/vest_data/softX-ray"),
    ("geometry", "/Volumes/hsyun_ssd/vest_data/softX-ray/LOS"),
    # Working copies made for analysis and for collaborators. Mostly duplicates
    # of the acquisition tree, but not entirely: the viewer directories carry
    # shot 45540, which the acquisition dumps never covered.
    ("digitizer", "/Users/yun/git/VEST_Soft X-ray/data/raw"),
    ("digitizer", "/Users/yun/git/VEST_Soft X-ray/VEST_SXR_Viewer"),
    ("digitizer", "/Users/yun/git/VEST_Soft X-ray/update"),
    ("digitizer", "/Users/yun/Downloads/VEST_SXR_Viewer"),
    ("digitizer", "/Volumes/hsyun_ssd/Downloads/VEST_SXR_Viewer"),
)


def mark_duplicates(entries: Sequence[Entry]) -> None:
    """Point every repeated file at the first copy seen, in place.

    The soft X-ray CSVs for a few shots were copied into analysis repositories
    and download folders; only one copy needs to reach the archive. Identity is
    the content hash, not the path, so a renamed copy is still caught.
    """
    first_by_hash: dict[str, str] = {}
    for entry in entries:
        if entry.sha256 is None:
            continue
        canonical = first_by_hash.setdefault(entry.sha256, entry.source)
        if canonical != entry.source:
            entry.duplicate_of = canonical


def build_inventory(sources: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Walk every configured source and return the serialisable inventory."""
    entries: list[Entry] = []
    scanned: list[dict[str, Any]] = []
    for scanner_name, raw_root in sources:
        root = Path(raw_root).expanduser()
        if not root.exists():
            scanned.append({"scanner": scanner_name, "root": str(root), "status": "missing"})
            continue
        found = list(SCANNERS[scanner_name](root))
        entries.extend(found)
        scanned.append(
            {
                "scanner": scanner_name,
                "root": str(root),
                "status": "ok",
                "entries": len(found),
                "bytes": sum(entry.size_bytes for entry in found),
            }
        )

    mark_duplicates(entries)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fluctuation_frame_rate_hz": FLUCTUATION_FRAME_RATE_HZ,
        "sources": scanned,
        "entries": [asdict(entry) for entry in entries],
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _gib(value: int) -> float:
    return value / float(1 << 30)


def format_report(inventory: dict[str, Any]) -> str:
    """Render the inventory as the summary a human actually reads."""
    entries = inventory["entries"]
    lines = [
        f"VEST external diagnostic inventory  ({inventory['generated_at']})",
        "",
        "Sources",
    ]
    for source in inventory["sources"]:
        if source["status"] == "missing":
            lines.append(f"  MISSING  {source['root']}")
        else:
            lines.append(
                f"  {source['entries']:>6} entries  {_gib(source['bytes']):>8.2f} GiB  {source['root']}"
            )

    by_diagnostic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = entry["diagnostic"] or "NEEDS REVIEW"
        by_diagnostic[key].append(entry)

    lines += ["", "Classification"]
    for key in sorted(by_diagnostic):
        group = by_diagnostic[key]
        live = [e for e in group if not e["duplicate_of"]]
        shots = {e["shot"] for e in live if e["shot"] is not None}
        total = sum(e["size_bytes"] for e in live)
        duplicates = len(group) - len(live)
        destination = (
            "legacy/" if key in MAPPED_DIAGNOSTICS
            else "legacy/" if key == "soft_x_rays_geometry"
            else "unmapped/" if key in UNMAPPED_DIAGNOSTICS
            else "-"
        )
        note = f"  ({duplicates} duplicate copies)" if duplicates else ""
        lines.append(
            f"  {key:<28} {destination:<10} {len(live):>5} entries  "
            f"{len(shots):>5} shots  {_gib(total):>8.2f} GiB{note}"
        )

    review = [e for e in entries if e["diagnostic"] is None]
    if review:
        lines += ["", f"Needs review ({len(review)})"]
        reasons = Counter(e["reason"] or "unspecified" for e in review)
        for reason, count in reasons.most_common():
            lines.append(f"  {count:>5}  {reason}")
        for entry in review[:10]:
            if entry["size_bytes"]:
                lines.append(f"        {entry['source']}")

    fluctuation = sorted(
        {e["shot"] for e in entries if e["diagnostic"] == "camera_visible_fluctuation"}
    )
    if fluctuation:
        lines += [
            "",
            f"Fluctuation-grade camera shots reserved for issue #161 ({len(fluctuation)})",
            "  " + " ".join(str(shot) for shot in fluctuation),
        ]

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        action="append",
        metavar="SCANNER:PATH",
        help=(
            "Override the default source list. SCANNER is one of: "
            + ", ".join(sorted(SCANNERS))
        ),
    )
    parser.add_argument("--output", default="inventory.json", help="Where to write the inventory")
    parser.add_argument(
        "--report",
        nargs="?",
        const="-",
        help="Also write the human-readable summary here ('-' for stdout)",
    )
    args = parser.parse_args(argv)

    if args.source:
        sources = []
        for raw in args.source:
            scanner, _, path = raw.partition(":")
            if scanner not in SCANNERS:
                parser.error(f"Unknown scanner {scanner!r}; expected one of {sorted(SCANNERS)}")
            if not path:
                parser.error(f"--source {raw!r} is missing a path")
            sources.append((scanner, path))
    else:
        sources = list(DEFAULT_SOURCES)

    inventory = build_inventory(sources)
    Path(args.output).expanduser().write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    if args.report:
        report = format_report(inventory)
        if args.report == "-":
            print(report)
        else:
            Path(args.report).expanduser().write_text(report + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
