"""Large generated documentation media kept outside git (#2117).

GIFs and long trajectory sidecars are published as assets of one GitHub
Release (``docs-media``), not committed.  ``docs/media.lock.json`` pins each
one by SHA-256 and size; the docs build fetches them into the extracted
source tree before Jekyll runs and fails on any mismatch.

Asset names are content addressed (``<sha256[:16]>-<basename>``), so every
version can live in the one release side by side and nothing is ever
overwritten: an old commit's lock keeps resolving.

    python docs/scripts/media.py fetch    # materialize into this checkout (local preview)
    python docs/scripts/media.py verify   # local files match the lock
    python docs/scripts/media.py lock     # re-pin the lock from local files (after regenerating)
    python docs/scripts/media.py upload   # maintainers: publish missing assets with `gh`

Standard library only: docs/build.py imports this, and it must run where
nothing but Python and git are installed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

LOCK_NAME = "media.lock.json"
SCHEMA_VERSION = 1
DOCS = Path(__file__).resolve().parents[1]
DOWNLOAD = "https://github.com/{repository}/releases/download/{release}/{asset}"


class MediaError(RuntimeError):
    """A locked file is missing, unreachable or does not match its digest."""


def default_cache() -> Path:
    configured = os.environ.get("VAFT_DOCS_MEDIA_CACHE")
    if configured:
        return Path(configured).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "vaft-docs-media"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_name(sha256: str, path: str) -> str:
    return f"{sha256[:16]}-{Path(path).name}"


def load_lock(docs: Path) -> dict | None:
    """The lock in ``docs``, or None when this tree keeps no media outside git."""
    lock = docs / LOCK_NAME
    if not lock.is_file():
        return None
    data = json.loads(lock.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise MediaError(f"{lock}: unsupported schema_version {data.get('schema_version')!r}")
    for key in ("repository", "release", "include", "files"):
        if key not in data:
            raise MediaError(f"{lock}: missing {key!r}")
    return data


def _matches(path: str, include: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in include)


def _verified(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["bytes"]
        and _sha256(path) == entry["sha256"]
    )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "vaft-docs-media"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_cached(lock: dict, path: str, cache: Path, *, offline: bool = False) -> Path:
    """Return a verified cached copy of the locked ``path``, downloading if needed."""
    entry = lock["files"][path]
    cached = cache / entry["sha256"]
    if _verified(cached, entry):
        return cached
    if offline:
        raise MediaError(f"{path}: not in the media cache ({cache}) and fetching is disabled")
    asset = entry.get("asset") or asset_name(entry["sha256"], path)
    base = lock.get("base_url") or DOWNLOAD.format(
        repository=lock["repository"], release=lock["release"], asset="{asset}"
    )
    url = base.format(asset=asset)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, prefix=".partial-", delete=False) as handle:
        partial = Path(handle.name)
    try:
        try:
            _download(url, partial)
        except (urllib.error.URLError, OSError) as error:
            raise MediaError(f"{path}: cannot fetch {url}: {error}") from error
        got = _sha256(partial)
        if got != entry["sha256"] or partial.stat().st_size != entry["bytes"]:
            raise MediaError(
                f"{path}: sha256 mismatch for {url}\n"
                f"    locked {entry['sha256']} ({entry['bytes']} bytes)\n"
                f"    got    {got} ({partial.stat().st_size} bytes)"
            )
        partial.replace(cached)
    finally:
        partial.unlink(missing_ok=True)
    return cached


def materialize(docs: Path, cache: Path | None = None, *, offline: bool = False) -> list[str]:
    """Place every locked file under ``docs``; return the paths that were placed.

    A file already present with the locked digest is left alone.  Every failure
    is collected first, so one run names everything that is missing.
    """
    lock = load_lock(docs)
    if lock is None:
        return []
    cache = cache or default_cache()
    placed, failures = [], []
    for path, entry in sorted(lock["files"].items()):
        target = docs / path
        if _verified(target, entry):
            continue
        try:
            source = ensure_cached(lock, path, cache, offline=offline)
        except MediaError as error:
            failures.append(str(error))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        placed.append(path)
    if failures:
        raise MediaError(
            "documentation media could not be materialized:\n  "
            + "\n  ".join(failures)
            + "\nThese files live outside git (docs/media.lock.json). Fetch them once with "
            "network access:\n    python docs/scripts/media.py fetch\n"
            f"(cache: {cache}; set VAFT_DOCS_MEDIA_CACHE to share one)"
        )
    return placed


def verify(docs: Path) -> list[str]:
    """Problems with the locked files present under ``docs`` (missing ones included)."""
    lock = load_lock(docs)
    if lock is None:
        return []
    problems = []
    for path, entry in sorted(lock["files"].items()):
        target = docs / path
        if not target.is_file():
            problems.append(f"{path}: missing (run `python docs/scripts/media.py fetch`)")
        elif not _verified(target, entry):
            problems.append(f"{path}: does not match its locked sha256/size")
    return problems


def relock(docs: Path) -> dict:
    """Re-pin every local file matching the lock's ``include`` patterns."""
    lock = load_lock(docs)
    if lock is None:
        raise MediaError(f"{docs / LOCK_NAME} does not exist")
    files = {}
    for target in sorted(p for p in docs.rglob("*") if p.is_file()):
        path = target.relative_to(docs).as_posix()
        if _matches(path, lock["include"]):
            sha256 = _sha256(target)
            files[path] = {
                "sha256": sha256,
                "bytes": target.stat().st_size,
                "asset": asset_name(sha256, path),
            }
    lock["files"] = files
    (docs / LOCK_NAME).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def upload(docs: Path) -> list[str]:
    """Upload locked assets the release does not have yet (maintainers, needs `gh`)."""
    lock = load_lock(docs)
    if lock is None:
        return []
    listed = subprocess.run(
        ["gh", "release", "view", lock["release"], "--repo", lock["repository"],
         "--json", "assets", "--jq", ".assets[].name"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    missing = [
        (path, entry) for path, entry in sorted(lock["files"].items())
        if entry["asset"] not in listed
    ]
    uploaded = []
    with tempfile.TemporaryDirectory() as scratch:
        for path, entry in missing:
            source = docs / path
            if not _verified(source, entry):
                raise MediaError(f"{path}: local file does not match the lock; run `lock` first")
            staged = Path(scratch) / entry["asset"]
            shutil.copyfile(source, staged)
            subprocess.run(
                ["gh", "release", "upload", lock["release"], str(staged),
                 "--repo", lock["repository"]],
                check=True,
            )
            uploaded.append(entry["asset"])
    return uploaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "verify", "lock", "upload"))
    parser.add_argument("--docs", type=Path, default=DOCS, help="docs directory (default: this one)")
    parser.add_argument("--cache", type=Path, default=None, help="media cache (default: %(default)s)")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "fetch":
            placed = materialize(arguments.docs, arguments.cache)
            print(f"{len(placed)} file(s) placed; all locked media present")
        elif arguments.command == "verify":
            problems = verify(arguments.docs)
            for line in problems:
                print(f"error: {line}")
            if problems:
                return 1
            print("all locked media present and matching")
        elif arguments.command == "lock":
            lock = relock(arguments.docs)
            print(f"locked {len(lock['files'])} file(s)")
        else:
            uploaded = upload(arguments.docs)
            print(f"uploaded {len(uploaded)} asset(s): {' '.join(uploaded) or '-'}")
    except MediaError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
