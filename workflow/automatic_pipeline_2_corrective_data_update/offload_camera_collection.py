#!/usr/bin/env python3

"""Offload a recovered FAST-camera collection to vestserver, verified, then purge.

The Windows collector (BatchConv2 re-export of archived ``.mcf`` files) leaves
a flat directory::

    _collect/{tag}.mcf.zst          raw vendor container, zstd-compressed
    _collect/{tag}/{tag}_{i:08d}.bmp  re-exported, dark-trimmed frames
    _collect/{tag}/{tag}_bmp.txt      BatchConv2 "GX-8" header
    _collect/{tag}/_convert_done.json written last, once the export is complete

``tag`` is a bare shot number, or a shot with a suffix (``42462__v2``,
``25231__fast``) when one shot had more than one recording. Only tags with
``_convert_done.json`` and at least one frame are touched. Subcommands, in
order:

``plan``
    Classify every tag, list what the server already holds, and write
    ``offload_plan.tsv`` plus ``variants_review.tsv`` (suffixed tags, for a
    per-shot decision by a person -- their frames are never pushed here).
``push``
    rsync frames of bare, converted shots the server lacks to
    ``legacy/camera_visible/{shot}/`` (never overwriting), and the ``.mcf.zst``
    of every converted tag, variants included, to the raw archive directory.
``verify``
    Frames: per-file name and size must match. Containers: sha256 on both sides
    and ``zstd -t`` on the server. Writes ``offload_manifest.tsv``.
``purge``
    Delete local ``.mcf.zst`` files whose manifest row is verified. Dry run
    unless ``--execute``. Frames are never deleted here.

Every step is re-runnable; later collector output is picked up by running the
same sequence again. A size match alone is not accepted for containers: an
``rsync --partial`` resume has left a truncated file in place before.

Run::

    ./offload_camera_collection.py plan
    ./offload_camera_collection.py push --limit 1000
    ./offload_camera_collection.py verify
    ./offload_camera_collection.py purge            # dry run
    ./offload_camera_collection.py purge --execute
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence

LOGGER = logging.getLogger("offload_camera_collection")

DEFAULT_COLLECT = Path("/Volumes/hsyun_ssd/mcf/_collect")
DEFAULT_HOST = "user1@147.46.36.244"
DEFAULT_PORT = 2222
DEFAULT_REMOTE_LEGACY = "/srv/vest.filedb/legacy/camera_visible"
DEFAULT_REMOTE_MCF = "/mnt/backup/vaft/camera_mcf"

CONTAINER_SUFFIX = ".mcf.zst"
DONE_MARKER = "_convert_done.json"
PLAN_NAME = "offload_plan.tsv"
VARIANTS_NAME = "variants_review.tsv"
MANIFEST_NAME = "offload_manifest.tsv"
PURGED_NAME = "offloaded_mcf.tsv"

PLAN_FIELDS = (
    "tag", "shot", "variant", "converted", "bmp_n", "bmp_bytes", "zst_bytes",
    "server_has_frames", "frames_action", "mcf_action",
)
MANIFEST_FIELDS = (
    "tag", "kind", "file", "bytes", "sha256", "remote_path", "status", "detail", "verified_at",
)


@dataclass(frozen=True)
class Remote:
    host: str
    port: int
    legacy: str
    mcf: str

    def ssh(self) -> list[str]:
        return ["ssh", "-p", str(self.port), "-o", "BatchMode=yes", self.host]

    def rsync_shell(self) -> str:
        return f"ssh -p {self.port} -o BatchMode=yes"

    def run(self, script: str, *, stdin: str | None = None) -> str:
        result = subprocess.run(
            [*self.ssh(), script], input=stdin, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"remote command failed ({result.returncode}): {result.stderr.strip()}")
        return result.stdout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _shot_of(tag: str) -> str:
    return tag.split("__", 1)[0]


def _frames(shot_dir: Path) -> list[Path]:
    # `._*` are exFAT AppleDouble sidecars, not frames.
    return sorted(
        path for path in shot_dir.iterdir()
        if path.suffix == ".bmp" and not path.name.startswith("._")
    )


def _shipped_files(shot_dir: Path) -> list[Path]:
    """Every file that belongs on the server: frames, header, provenance JSON."""
    return sorted(
        path for path in shot_dir.iterdir()
        if path.is_file() and not path.name.startswith("._") and path.name != "Thumbs.db"
    )


def _write_tsv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _out_dir(args: argparse.Namespace) -> Path:
    out = Path(args.out) if args.out else args.collect.parent / "_collect_logs" / "offload"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def classify(collect: Path, server_shots: set[str]) -> list[dict]:
    tags: set[str] = set()
    for entry in collect.iterdir():
        if entry.name.startswith(("._", "_")):
            continue
        if entry.name.endswith(CONTAINER_SUFFIX):
            tags.add(entry.name[: -len(CONTAINER_SUFFIX)])
        elif entry.is_dir():
            tags.add(entry.name)

    rows = []
    for tag in sorted(tags, key=lambda t: (int(_shot_of(t)) if _shot_of(t).isdigit() else 0, t)):
        shot_dir = collect / tag
        container = collect / f"{tag}{CONTAINER_SUFFIX}"
        frames = _frames(shot_dir) if shot_dir.is_dir() else []
        converted = (shot_dir / DONE_MARKER).exists() and bool(frames)
        variant = not tag.isdigit()
        server_has = tag in server_shots if not variant else _shot_of(tag) in server_shots

        if not converted:
            frames_action = "wait"
        elif variant:
            frames_action = "review"
        elif server_has:
            frames_action = "keep_server"
        else:
            frames_action = "push"
        mcf_action = "push" if converted and container.exists() else (
            "wait" if container.exists() else "none"
        )
        rows.append({
            "tag": tag,
            "shot": _shot_of(tag),
            "variant": int(variant),
            "converted": int(converted),
            "bmp_n": len(frames),
            "bmp_bytes": sum(path.stat().st_size for path in frames),
            "zst_bytes": container.stat().st_size if container.exists() else 0,
            "server_has_frames": int(server_has),
            "frames_action": frames_action,
            "mcf_action": mcf_action,
        })
    return rows


def _variant_rows(collect: Path, plan: list[dict]) -> list[dict]:
    by_shot: dict[str, list[dict]] = {}
    for row in plan:
        by_shot.setdefault(row["shot"], []).append(row)
    out = []
    for shot, rows in sorted(by_shot.items()):
        if not any(int(row["variant"]) for row in rows):
            continue
        for row in rows:
            meta = {}
            meta_path = collect / row["tag"] / "_mcf_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            header = collect / row["tag"] / f"{row['tag']}_bmp.txt"
            rec_time = ""
            if header.exists():
                for line in header.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("Rec_Time:"):
                        rec_time = line.split(":", 1)[1].strip()
                        break
            out.append({
                "shot": shot,
                "tag": row["tag"],
                "converted": row["converted"],
                "bmp_n": row["bmp_n"],
                "frame_rate": meta.get("frame_rate", ""),
                "frame_range": "-".join(map(str, meta.get("frame_range", []))),
                "rec_time": rec_time,
                "zst_bytes": row["zst_bytes"],
                "server_has_frames": row["server_has_frames"],
                "decision": "",
            })
    return out


def cmd_plan(args: argparse.Namespace, remote: Remote) -> int:
    listing = remote.run(f"ls {shlex.quote(remote.legacy)}")
    server_shots = {name for name in listing.split() if name.isdigit()}
    plan = classify(args.collect, server_shots)
    out = _out_dir(args)
    _write_tsv(out / PLAN_NAME, PLAN_FIELDS, plan)

    variants = _variant_rows(args.collect, plan)
    variants_path = out / VARIANTS_NAME
    # Keep decisions a person already wrote.
    previous = {row["tag"]: row.get("decision", "") for row in _read_tsv(variants_path)}
    for row in variants:
        row["decision"] = previous.get(row["tag"], "")
    _write_tsv(variants_path, tuple(variants[0]) if variants else ("shot",), variants)

    def count(key: str, value: str) -> int:
        return sum(1 for row in plan if row[key] == value)

    push_frames = [row for row in plan if row["frames_action"] == "push"]
    push_mcf = [row for row in plan if row["mcf_action"] == "push"]
    LOGGER.info("server holds %d shots", len(server_shots))
    LOGGER.info(
        "tags %d | frames push %d (%.1f GB), keep_server %d, review %d, wait %d",
        len(plan), len(push_frames), sum(r["bmp_bytes"] for r in push_frames) / 1e9,
        count("frames_action", "keep_server"), count("frames_action", "review"),
        count("frames_action", "wait"),
    )
    LOGGER.info(
        "containers push %d (%.1f GB), wait %d",
        len(push_mcf), sum(r["zst_bytes"] for r in push_mcf) / 1e9, count("mcf_action", "wait"),
    )
    LOGGER.info("wrote %s and %s (%d variant rows)", out / PLAN_NAME, variants_path, len(variants))
    return 0


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------

def _rsync(remote: Remote, source: Path, files: list[str], dest: str, *, dry_run: bool) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as handle:
        handle.write("\n".join(files) + "\n")
        list_path = handle.name
    command = [
        "rsync", "-rlt", "--partial", "--ignore-existing", "--chmod=Du=rwx,Fu=rw",
        "--exclude=._*", "--exclude=Thumbs.db", f"--files-from={list_path}",
        "-e", remote.rsync_shell(), "--info=stats1",
    ]
    if dry_run:
        command.append("--dry-run")
    command += [f"{source}/", f"{remote.host}:{dest}/"]
    LOGGER.info("rsync %d entries -> %s%s", len(files), dest, " (dry run)" if dry_run else "")
    subprocess.run(command, check=True)


def cmd_push(args: argparse.Namespace, remote: Remote) -> int:
    plan = _read_tsv(_out_dir(args) / PLAN_NAME)
    if not plan:
        LOGGER.error("no plan; run `plan` first")
        return 2
    verified = {
        (row["tag"], row["kind"]) for row in _read_tsv(_out_dir(args) / MANIFEST_NAME)
        if row["status"] == "verified"
    }
    frames = [r["tag"] for r in plan if r["frames_action"] == "push" and (r["tag"], "frames") not in verified]
    containers = [r["tag"] for r in plan if r["mcf_action"] == "push" and (r["tag"], "mcf") not in verified]
    if args.limit:
        frames, containers = frames[: args.limit], containers[: args.limit]
    if args.what in ("frames", "all") and frames:
        # A trailing slash would copy the contents; a bare name copies the directory.
        _rsync(remote, args.collect, frames, remote.legacy, dry_run=args.dry_run)
    if args.what in ("mcf", "all") and containers:
        remote.run(f"mkdir -p {shlex.quote(remote.mcf)}")
        _rsync(
            remote, args.collect, [f"{tag}{CONTAINER_SUFFIX}" for tag in containers],
            remote.mcf, dry_run=args.dry_run,
        )
    LOGGER.info("pushed frames for %d shots, %d containers", len(frames), len(containers))
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _remote_sizes(remote: Remote, tags: list[str]) -> dict[str, dict[str, int]]:
    script = (
        f"cd {shlex.quote(remote.legacy)} && while read t; do "
        "[ -d \"$t\" ] && find \"$t\" -maxdepth 1 -type f ! -name '._*' -printf '%h/%f\\t%s\\n'; "
        "done"
    )
    out: dict[str, dict[str, int]] = {}
    for line in remote.run(script, stdin="\n".join(tags) + "\n").splitlines():
        rel, size = line.rsplit("\t", 1)
        tag, name = rel.split("/", 1)
        out.setdefault(tag, {})[name] = int(size)
    return out


def _remote_container_checks(remote: Remote, names: list[str], jobs: int) -> dict[str, tuple[int, str, str]]:
    """``name -> (bytes, sha256, zstd_test)`` for containers on the server."""
    script = (
        f"cd {shlex.quote(remote.mcf)} && xargs -P {jobs} -I{{}} sh -c "
        "'if [ -f \"$1\" ]; then s=$(stat -c %s \"$1\"); h=$(sha256sum \"$1\" | cut -d\" \" -f1); "
        "if zstd -tq \"$1\" 2>/dev/null; then z=ok; else z=fail; fi; "
        "printf \"%s\\t%s\\t%s\\t%s\\n\" \"$1\" \"$s\" \"$h\" \"$z\"; "
        "else printf \"%s\\t0\\t-\\tmissing\\n\" \"$1\"; fi' _ {}"
    )
    out = {}
    for line in remote.run(script, stdin="\n".join(names) + "\n").splitlines():
        name, size, digest, test = line.split("\t")
        out[name] = (int(size), digest, test)
    return out


def cmd_verify(args: argparse.Namespace, remote: Remote) -> int:
    out = _out_dir(args)
    plan = _read_tsv(out / PLAN_NAME)
    manifest = {(row["tag"], row["kind"]): row for row in _read_tsv(out / MANIFEST_NAME)}

    def done(tag: str, kind: str) -> bool:
        return manifest.get((tag, kind), {}).get("status") == "verified"

    frame_tags = [r["tag"] for r in plan if r["frames_action"] == "push" and not done(r["tag"], "frames")]
    container_tags = [r["tag"] for r in plan if r["mcf_action"] == "push" and not done(r["tag"], "mcf")]
    if args.limit:
        frame_tags, container_tags = frame_tags[: args.limit], container_tags[: args.limit]

    bad = 0
    for start in range(0, len(frame_tags), 500):
        batch = frame_tags[start:start + 500]
        remote_sizes = _remote_sizes(remote, batch)
        for tag in batch:
            local = {path.name: path.stat().st_size for path in _shipped_files(args.collect / tag)}
            there = remote_sizes.get(tag, {})
            missing = sorted(set(local) - set(there))
            differ = sorted(name for name in set(local) & set(there) if local[name] != there[name])
            ok = not missing and not differ
            bad += not ok
            manifest[(tag, "frames")] = {
                "tag": tag, "kind": "frames", "file": f"{tag}/", "bytes": sum(local.values()),
                "sha256": "", "remote_path": f"{remote.legacy}/{tag}",
                "status": "verified" if ok else "mismatch",
                "detail": f"{len(local)} files" if ok else f"missing {missing[:3]} differ {differ[:3]}",
                "verified_at": _now(),
            }
        _write_tsv(out / MANIFEST_NAME, MANIFEST_FIELDS, manifest.values())
        LOGGER.info("frames verified %d/%d", min(start + 500, len(frame_tags)), len(frame_tags))

    for start in range(0, len(container_tags), 50):
        batch = container_tags[start:start + 50]
        names = [f"{tag}{CONTAINER_SUFFIX}" for tag in batch]
        checks = _remote_container_checks(remote, names, args.jobs)
        for tag, name in zip(batch, names):
            local_path = args.collect / name
            local_size = local_path.stat().st_size
            remote_size, remote_digest, test = checks.get(name, (0, "-", "missing"))
            local_digest = _sha256(local_path) if remote_size == local_size else ""
            ok = remote_size == local_size and remote_digest == local_digest and test == "ok"
            bad += not ok
            manifest[(tag, "mcf")] = {
                "tag": tag, "kind": "mcf", "file": name, "bytes": local_size,
                "sha256": local_digest, "remote_path": f"{remote.mcf}/{name}",
                "status": "verified" if ok else "mismatch",
                "detail": f"zstd {test}" if ok else (
                    f"size {remote_size} vs {local_size}, sha {'=' if remote_digest == local_digest else '!='}, zstd {test}"
                ),
                "verified_at": _now(),
            }
        _write_tsv(out / MANIFEST_NAME, MANIFEST_FIELDS, manifest.values())
        LOGGER.info("containers verified %d/%d", min(start + 50, len(container_tags)), len(container_tags))

    LOGGER.info("verify done: %d mismatches; manifest %s", bad, out / MANIFEST_NAME)
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------

def cmd_purge(args: argparse.Namespace, remote: Remote) -> int:
    out = _out_dir(args)
    rows = [
        row for row in _read_tsv(out / MANIFEST_NAME)
        if row["kind"] == "mcf" and row["status"] == "verified"
    ]
    purged = {row["tag"] for row in _read_tsv(out / PURGED_NAME)}
    targets = []
    for row in rows:
        path = args.collect / row["file"]
        if row["tag"] in purged or not path.exists():
            continue
        # The frames must still be there: they are the only reason the
        # container is no longer needed locally.
        if not (args.collect / row["tag"] / DONE_MARKER).exists():
            LOGGER.warning("skip %s: no %s beside it", row["file"], DONE_MARKER)
            continue
        if path.stat().st_size != int(row["bytes"]):
            LOGGER.warning("skip %s: size changed since verification", row["file"])
            continue
        targets.append((row, path))

    total = sum(path.stat().st_size for _, path in targets)
    LOGGER.info("%d verified containers, %.1f GB%s", len(targets), total / 1e9,
                "" if args.execute else " (dry run; pass --execute to delete)")
    if not args.execute:
        return 0

    log_path = out / PURGED_NAME
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if new_file:
            writer.writerow(["tag", "file", "bytes", "sha256", "remote_path", "purged_at"])
        for row, path in targets:
            path.unlink()
            sidecar = path.with_name(f"._{path.name}")
            if sidecar.exists():
                sidecar.unlink()
            writer.writerow([row["tag"], row["file"], row["bytes"], row["sha256"], row["remote_path"], _now()])
            handle.flush()
    LOGGER.info("purged %d containers; log %s", len(targets), log_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", type=Path, default=DEFAULT_COLLECT)
    parser.add_argument("--out", help="state directory (default: <collect>/../_collect_logs/offload)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--remote-legacy", default=DEFAULT_REMOTE_LEGACY)
    parser.add_argument("--remote-mcf", default=DEFAULT_REMOTE_MCF)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    push = sub.add_parser("push")
    push.add_argument("--what", choices=("frames", "mcf", "all"), default="all")
    push.add_argument("--limit", type=int, default=0, help="at most N shots per kind")
    push.add_argument("--dry-run", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--limit", type=int, default=0)
    verify.add_argument("--jobs", type=int, default=4, help="parallel sha256/zstd -t on the server")
    purge = sub.add_parser("purge")
    purge.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    remote = Remote(args.host, args.port, args.remote_legacy, args.remote_mcf)
    command = {"plan": cmd_plan, "push": cmd_push, "verify": cmd_verify, "purge": cmd_purge}
    return command[args.command](args, remote)


if __name__ == "__main__":
    sys.exit(main())
