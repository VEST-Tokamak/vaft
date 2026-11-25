#!/usr/bin/env python3
"""Inventory and measure VAFT's data assets for the storage-policy audit (#2307).

Read-only: every number comes from ``git`` plumbing (``ls-tree``, ``cat-file``,
``rev-list``, ``grep``) run against a ref, never from the working tree, so the
audit works in a sparse checkout and never modifies the repository. It does
not import ``vaft``.

Two products:

``docs/architecture/data_assets/inventory.yaml`` (``inventory``)
    Deterministic: sorted rows, repository-relative paths, no timestamps, no
    commit hashes. One row per *asset group* -- the hand-curated table
    :data:`GROUPS` below joined with what the tree at ``--ref`` holds for that
    group (files, bytes, Git LFS pointers, wheel membership, text-search
    consumers) -- plus one row per registered reference sample, read from its
    ``manifest.yaml`` and checked against the stored blobs. ``--check`` exits
    1 when the committed file differs from a fresh build. It is a review aid,
    not a CI gate: the consumer lists change with every test that mentions a
    sample.

``measure`` (JSON on stdout or ``--output``)
    Dated and ref-pinned: current-tree bytes per group for each ``--ref``, the
    history (blob versions, raw and packed bytes) per group over the refs a
    fresh clone receives, Git LFS pointer and object sizes, the largest
    historical blobs and the published PyPI artifact sizes. These numbers
    depend on time and on the local object store, so they are never compared
    by ``--check``; the JSON records the method and its limitations.

Limitations (also written into both products):

* Consumers are found by text search for path fragments and API names. Paths
  built at run time (``data_path(f"legacy/{shot}.h5")``) are missed, and a hit
  does not prove the file is read.
* ``rev-list --objects`` names a blob after the first path it meets, so a blob
  stored under two paths is counted once, under one of them.
* Packed (on-disk) sizes depend on how the local repository was packed; the
  GitHub-side size is reported separately from the API.
* Wheel membership is evaluated from ``[tool.setuptools.package-data]`` globs,
  not by building a wheel.

Usage::

    python workflow/data_asset_audit/audit_assets.py inventory            # write
    python workflow/data_asset_audit/audit_assets.py inventory --check    # exit 1 on drift
    git fetch --prune --tags origin
    python workflow/data_asset_audit/audit_assets.py measure \\
        --ref origin/develop --ref origin/main --output measurement.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("docs/architecture/data_assets/inventory.yaml")
SCHEMA_VERSION = 1

#: Where consumers are searched; ``vaft/data`` itself is excluded.
CONSUMER_AREAS = (".github", "docs", "notebooks", "test", "tutorial", "vaft", "workflow")
CONSUMER_EXCLUDES = ("vaft/data", "docs/assets", "docs/architecture/data_assets")

#: A tree file at least this large that no group claims is listed as unassigned.
UNASSIGNED_THRESHOLD = 1_000_000

LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

LIMITATIONS = (
    "consumers are a text search for path fragments and API names; run-time built paths are missed",
    "a blob stored under several paths is attributed to the first path rev-list reports",
    "packed (on-disk) bytes depend on the local repository's packing",
    "wheel membership is evaluated from pyproject package-data globs plus the setup.py 39915 substitution, not by building a wheel",
)

# Storage categories of #2307 section 4.
CATEGORIES = {
    "source": "Canonical source/configuration (Git)",
    "fixture": "Lightweight offline fixture (Git / package)",
    "reference": "Versioned scientific reference (GitHub Release)",
    "regenerable": "Regenerable calculation artifact (generated cache)",
    "persistent": "Persistent experimental/computational data (FileDB / HSDS)",
    "public": "External public dataset (upstream registry + cache)",
    "docs_media": "Documentation media (#2117 policy)",
    "undecided": "Decision backlog: not classified, must not be treated as disposable",
}


@dataclass(frozen=True)
class Group:
    """One curated asset group.

    ``paths`` are ``fnmatch`` patterns on repository-relative paths; the first
    group whose pattern matches claims a file, so specific groups come first.
    ``history_only`` groups match paths that no longer exist on the audited
    ref but still occupy history.
    """

    id: str
    paths: tuple[str, ...]
    category: str
    current_placement: str
    proposed_placement: str
    retrieval: str
    migration: tuple[str, ...]
    decision: str  # decided | proposed | open
    consumers: tuple[str, ...] = ()  # extended regexes for git grep -E
    notes: str = ""
    history_only: bool = False
    exclude: tuple[str, ...] = field(default=())


SHOT = "[0-9][0-9][0-9][0-9][0-9]"

GROUPS: tuple[Group, ...] = (
    Group(
        id="samples.shot.manifests",
        paths=(f"vaft/data/samples/{SHOT}/manifest.yaml", f"vaft/data/samples/{SHOT}/*.md"),
        category="source",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout; the manifest is the identity record I-3/I-4 resolve against",
        migration=("#2328",),
        decision="decided",
        notes="Manifests stay in Git: available_samples() lists shots from them even when the bytes are remote.",
    ),
    Group(
        id="samples.shot.representations",
        paths=(f"vaft/data/samples/{SHOT}/omas.json.gz", f"vaft/data/samples/{SHOT}/imas.nc"),
        category="reference",
        current_placement="git",
        proposed_placement="github-release",
        retrieval="vaft.data.sample() -> checkout, else VAFT_CACHE_DIR/references/samples, else Release download (I-4)",
        migration=("#2328", "#2329", "#2331", "#207"),
        decision="decided",
        consumers=(
            r"data/samples/[0-9]{5}/(omas\.json\.gz|imas\.nc)",
            r"\bsample\(\s*[0-9]{5}",
            r"sample_ods\(",
            r"sample_manifest\(",
        ),
        notes=(
            "Umbrella #2304 section 3.2. The wheel copy of 39915 (wheel_samples) is a separate, "
            "packaged projection and stays."
        ),
    ),
    Group(
        id="samples.shot.sources",
        paths=(f"vaft/data/samples/{SHOT}/source/*",),
        category="reference",
        current_placement="git",
        proposed_placement="github-release",
        retrieval="Release asset named by the manifest (source.raw_source, generation.canonical_source)",
        migration=("#2328", "#207"),
        decision="proposed",
        consumers=(r"pipeline-until-efit", r"_daq_raw\.json\.gz"),
        notes=(
            "canonical_source is the input of the generation command that produces omas.json.gz/imas.nc. "
            "Keep it as long as the representations are kept: regenerability does not replace the "
            "published bytes (#2307 section 3.3 applies to samples too)."
        ),
    ),
    Group(
        id="samples.shot.stage_products",
        paths=(f"vaft/data/samples/{SHOT}/*",),
        category="undecided",
        current_placement="git",
        proposed_placement="git (pending size review)",
        retrieval="checkout",
        migration=("#2328",),
        decision="open",
        consumers=(r"data/samples/[0-9]{5}/[a-z_]+/",),
        notes=(
            "Per-shot stage products added after the main representations (e.g. 39915/mhd_stability "
            "CHEASE/GPEC JSON from #2306). Individually small but growing; decide whether they ship "
            "with the shot's Release bundle."
        ),
    ),
    Group(
        id="samples.tables",
        paths=("vaft/data/samples/vest_tier_a/*", "vaft/data/samples/vest_operational_history/*"),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        consumers=(r"vest_tier_a", r"vest_operational_history"),
        notes=(
            "Tier A / operational-history tables (#2277). The snapshot pins source SHA-256 values "
            "byte-exact (.gitattributes -text). Largest file is constraint_points.csv (6.1 MB); "
            "revisit if it grows."
        ),
    ),
    Group(
        id="samples.wheel_projection",
        paths=("vaft/data/wheel_samples/*",),
        category="fixture",
        current_placement="git + wheel",
        proposed_placement="git + wheel",
        retrieval="importlib resources from the installed package",
        migration=(),
        decision="decided",
        consumers=(r"wheel_samples",),
        notes="Packaged offline projection; test/verify_dist.py caps the wheel at 26 MiB.",
    ),
    Group(
        id="samples.kinetic_efit",
        paths=("vaft/data/kineticEfit/*",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        consumers=(r"kineticEfit",),
        notes="48224 historical validation reference (OMFIT era); its manifest pins these g-files by SHA-256.",
    ),
    Group(
        id="efit.green_tables_lfs",
        paths=("vaft/data/efit/*.ddd", "vaft/data/efit_table/*"),
        category="regenerable",
        current_placement="git-lfs",
        proposed_placement="github-release (single baseline) + VAFT_CACHE_DIR/generated/efund",
        retrieval="exact-identity local table -> EFUND generation -> baseline download -> explicit failure (I-5)",
        migration=("#2330", "#2331", "#207"),
        decision="decided",
        consumers=(r"\.ddd\b", r"efund_table_manifest", r"rv129129|ec129129|ep129129|rfcoil"),
        notes=(
            "Umbrella #2304 section 3.3. #695: regenerated tables changed EFIT yield, so a regenerated "
            "table substitutes for the baseline only after a stated numerical comparison."
        ),
    ),
    Group(
        id="efit.generated_tables_plain",
        paths=("vaft/data/efit/brzgfc.dat", "vaft/data/efit/mhdout.dat"),
        category="regenerable",
        current_placement="git",
        proposed_placement="github-release (with the .ddd baseline) + VAFT_CACHE_DIR/generated/efund",
        retrieval="same resolution as efit.green_tables_lfs (I-5)",
        migration=("#2330", "#207"),
        decision="proposed",
        consumers=(r"brzgfc", r"mhdout\.dat"),
        notes=(
            "Written by the same EFUND run as the .ddd tables and listed under table.files of "
            "efund_table_manifest.json (identity e28142c7...), but stored as plain Git (6.9 MB). "
            "mhdout.dat is excluded from the table identity (#793)."
        ),
    ),
    Group(
        id="efit.gfile_fixtures",
        paths=("vaft/data/efit/g0*",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout / vaft.data.sample_geqdsk()",
        migration=(),
        decision="proposed",
        consumers=(r"g0[0-9]{5}\.[0-9]{5}", r"sample_geqdsk\("),
        notes="g040330.* are 4.3 MB each; the other g-files are under 1.2 MB.",
    ),
    Group(
        id="efit.inputs",
        paths=("vaft/data/efit/*",),
        category="source",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="decided",
        notes="mhdin.dat, efund_table_manifest.json and other EFUND/EFIT text inputs: the identity inputs of the Green tables.",
    ),
    Group(
        id="legacy.raw_fixtures",
        paths=("vaft/data/legacy/*",),
        category="undecided",
        current_placement="git",
        proposed_placement="decision backlog (likely github-release for the files above 5 MB)",
        retrieval="checkout",
        migration=("#2328", "#207"),
        decision="open",
        consumers=(
            r"data/legacy/",
            r"digitizer_[0-9]+_[0-9]{5}",
            r"shot_4[0-9]{4}\.json\.gz",
            r"LID_1_100",
        ),
        notes=(
            "193 MB: SXR digitizer CSVs for 45531 (pinned by the 45531 manifest), 41514.h5, "
            "shot_44740/45531 JSON, camera PNGs, IMPA .mat files. Several files have no textual "
            "consumer (e.g. 41514.h5) and may be read through run-time paths; resolve before any move."
        ),
    ),
    Group(
        id="data.small_fixtures",
        paths=("vaft/data/sample/*",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        notes="Legacy IMPA sample JSON (< 1 MB each).",
    ),
    Group(
        id="data.geometry_config",
        paths=(
            "vaft/data/geometry/*",
            "vaft/data/gpec/*",
            "vaft/data/nubeam/*",
            "vaft/data/unified/*",
            "vaft/data/confinement/*",
            "vaft/data/shotlog/*",
            "vaft/data/public/*",
            "vaft/data/*",
        ),
        category="source",
        current_placement="git (+ wheel where listed in package-data)",
        proposed_placement="git",
        retrieval="checkout / package resources",
        migration=(),
        decision="decided",
        notes="Machine geometry, code inputs, small registries and the public-data source registry (bytes are fetched to the public cache).",
    ),
    Group(
        id="notebooks.outputs",
        paths=("notebooks/*.ipynb",),
        category="undecided",
        current_placement="git",
        proposed_placement="decide with #2117",
        retrieval="checkout",
        migration=(),
        decision="open",
        notes=(
            "Most notebooks keep normalized outputs on purpose (notebooks/_clean_outputs.py). Every "
            "re-run stores a full new blob; conference_operational_space_atlas.ipynb is 10 MB per version."
        ),
    ),
    Group(
        id="notebooks.data",
        paths=("notebooks/data/*",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=("#2331",),
        decision="proposed",
    ),
    Group(
        id="docs.media",
        paths=("docs/assets/*",),
        category="docs_media",
        current_placement="git",
        proposed_placement="per #2117 (docs-media release, SHA-pinned)",
        retrieval="#2117 / PR #2133",
        migration=("#2117",),
        decision="decided",
        notes="Owned by #2117; this audit only reports it.",
    ),
    Group(
        id="workflow.results",
        paths=("workflow/*.xlsx", "workflow/*.csv", "workflow/*.csv.gz", "workflow/*.json", "workflow/*.yml"),
        category="undecided",
        current_placement="git",
        proposed_placement="decision backlog",
        retrieval="checkout",
        migration=(),
        decision="open",
        notes="Dated study outputs committed next to workflows (e.g. automatic_pipeline_3_data_summary/*.xlsx 6.9 MB).",
    ),
    Group(
        id="test.fixtures",
        paths=("test/data/*",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="decided",
    ),
    # Paths no longer in the tree that still occupy history: #207 candidates.
    Group(
        id="history.retired_sample_layouts",
        paths=(
            "vaft/data/omas/*",
            "vaft/data/imas/*",
            "vaft/data/[0-9][0-9][0-9][0-9][0-9].json",
            "vaft/data/*_imas_equilibrium.h5",
            "packaging/wheel_samples/*",
        ),
        category="reference",
        current_placement="history only",
        proposed_placement="purge after consumers migrate",
        retrieval="superseded by vaft/data/samples and wheel_samples",
        migration=("#207",),
        decision="proposed",
        history_only=True,
        notes="Earlier sample layouts. Confirm each is superseded by a registered sample before #207 removes it.",
    ),
    Group(
        id="history.gh_pages",
        paths=("develop/*", "reference/*", "guide/*", "assets/*", "api/*"),
        category="docs_media",
        current_placement="gh-pages branch",
        proposed_placement="per #2117 (single-commit gh-pages)",
        retrieval="generated by docs.yml",
        migration=("#2117",),
        decision="decided",
        history_only=True,
        notes="Generated site output on the gh-pages branch; counted by path prefix.",
    ),
)


#: History attribution checks the retired layouts first: the ``vaft/data/*``
#: catch-all would otherwise claim them.
HISTORY_ORDER: tuple[Group, ...] = tuple(g for g in GROUPS if g.history_only) + tuple(
    g for g in GROUPS if not g.history_only
)


# --------------------------------------------------------------------------- git helpers


def git(*args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        input=input,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def rev_parse(ref: str) -> str:
    return git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


@dataclass(frozen=True)
class TreeFile:
    path: str
    size: int
    oid: str


def tree_files(ref: str) -> list[TreeFile]:
    out = git("ls-tree", "-r", "-l", "-z", ref)
    files = []
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        if kind != b"blob":
            continue
        files.append(TreeFile(path.decode(), int(size), oid.decode()))
    return files


def read_blobs(oids: Iterable[str]) -> Iterator[tuple[str, bytes]]:
    """Yield ``(oid, content)`` with one ``cat-file --batch`` process."""
    oids = list(oids)
    if not oids:
        return
    proc = subprocess.Popen(
        ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    try:
        for oid in oids:
            proc.stdin.write(oid.encode() + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            yield oid, data
    finally:
        proc.stdin.close()
        proc.wait()


# --------------------------------------------------------------------------- classification


def group_of(path: str, groups: Iterable[Group]) -> Group | None:
    for group in groups:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in group.paths):
            return group
    return None


def lfs_pointer(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(LFS_PREFIX):
        return None
    fields = dict(
        line.split(" ", 1) for line in data.decode().splitlines() if " " in line
    )
    return {"oid": fields.get("oid", ""), "size": int(fields.get("size", 0))}


def wheel_globs(ref: str) -> list[str]:
    pyproject = tomllib.loads(git("show", f"{ref}:pyproject.toml").decode())
    data = pyproject.get("tool", {}).get("setuptools", {}).get("package-data", {})
    return [f"vaft/{pattern}" for pattern in data.get("vaft", [])]


#: ``setup.py``'s ``build_py`` hook copies these wheel_samples files over the
#: package-data matches of the same name under ``samples/39915`` at build time.
WHEEL_SUBSTITUTES = {
    "vaft/data/wheel_samples/39915/manifest.yaml": "vaft/data/samples/39915/manifest.yaml",
    "vaft/data/wheel_samples/39915/omas.json.gz": "vaft/data/samples/39915/omas.json.gz",
}


def in_wheel(path: str, globs: Iterable[str]) -> bool:
    if path in WHEEL_SUBSTITUTES:
        return True
    if path in WHEEL_SUBSTITUTES.values():
        return False
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in globs)


def consumers(ref: str, patterns: Iterable[str]) -> list[str]:
    patterns = list(patterns)
    if not patterns:
        return []
    args = ["grep", "-l", "-I", "-E"]
    for pattern in patterns:
        args += ["-e", pattern]
    args += [ref, "--", *CONSUMER_AREAS, *(f":!{p}" for p in CONSUMER_EXCLUDES)]
    try:
        out = git(*args).decode()
    except subprocess.CalledProcessError as exc:  # exit 1 = no match
        if exc.returncode == 1:
            return []
        raise
    prefix = f"{ref}:"
    return sorted({line[len(prefix):] if line.startswith(prefix) else line for line in out.splitlines()})


def by_area(paths: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path in paths:
        counts[path.split("/", 1)[0]] += 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- inventory


def sample_rows(ref: str, files: list[TreeFile]) -> list[dict[str, Any]]:
    """One row per registered sample manifest, checked against the stored blobs."""
    by_path = {f.path: f for f in files}
    manifests = sorted(
        f for f in by_path if re.fullmatch(r"vaft/data/samples/\d{5}/manifest\.yaml", f)
    )
    rows = []
    for manifest_path in manifests:
        directory = manifest_path.rsplit("/", 1)[0]
        manifest = yaml.safe_load(git("show", f"{ref}:{manifest_path}")) or {}
        representations = []
        to_hash = {}
        for name, record in sorted((manifest.get("representations") or {}).items()):
            path = f"{directory}/{record.get('path', '')}"
            stored = by_path.get(path)
            representations.append(
                {
                    "name": name,
                    "path": path,
                    "package": record.get("package", "not recorded"),
                    "manifest_size": record.get("size", "not recorded"),
                    "stored_size": stored.size if stored else "missing",
                    "manifest_sha256": record.get("sha256", "not recorded"),
                }
            )
            if stored:
                to_hash[stored.oid] = len(representations) - 1
        for oid, data in read_blobs(to_hash):
            pointer = lfs_pointer(data)
            row = representations[to_hash[oid]]
            if pointer:
                row["stored_sha256"] = f"lfs:{pointer['oid']}"
            else:
                row["stored_sha256"] = hashlib.sha256(data).hexdigest()
        for row in representations:
            row.setdefault("stored_sha256", "missing")
            row["sha256_matches"] = row["stored_sha256"] == row["manifest_sha256"]
        source = manifest.get("source") or {}
        generation = manifest.get("generation") or {}
        directory_files = [f for f in files if f.path.startswith(directory + "/")]
        rows.append(
            {
                "shot": manifest.get("shot", "not recorded"),
                "directory": directory,
                "reference_id": manifest.get("reference_id", "not recorded"),
                "imas_dd_version": manifest.get("imas_dd_version", "not recorded"),
                "source_kind": source.get("kind", "not recorded"),
                "canonical_source": (
                    f"{directory}/{generation['canonical_source']}"
                    if generation.get("canonical_source")
                    else "not recorded"
                ),
                "generation_command": generation.get("command", "not recorded"),
                "representations": representations,
                "directory_files": len(directory_files),
                "directory_bytes": sum(f.size for f in directory_files),
            }
        )
    return rows


def build_inventory(ref: str) -> dict[str, Any]:
    files = tree_files(ref)
    globs = wheel_globs(ref)
    claimed: dict[str, list[TreeFile]] = defaultdict(list)
    unassigned = []
    for f in files:
        group = group_of(f.path, (g for g in GROUPS if not g.history_only))
        if group is None:
            if f.size >= UNASSIGNED_THRESHOLD:
                unassigned.append({"path": f.path, "bytes": f.size})
            continue
        claimed[group.id].append(f)

    # Find LFS pointers among small blobs only: a pointer is ~130 bytes.
    small = {f.oid: f.path for fs in claimed.values() for f in fs if f.size < 1024}
    pointers = {
        small[oid]: pointer
        for oid, data in read_blobs(small)
        if (pointer := lfs_pointer(data)) is not None
    }

    rows = []
    for group in GROUPS:
        members = sorted(claimed.get(group.id, []), key=lambda f: f.path)
        hits = consumers(ref, group.consumers)
        lfs = [pointers[f.path] for f in members if f.path in pointers]
        rows.append(
            {
                "id": group.id,
                "paths": list(group.paths),
                "category": group.category,
                "current_placement": group.current_placement,
                "proposed_placement": group.proposed_placement,
                "retrieval": group.retrieval,
                "migration": list(group.migration),
                "decision": group.decision,
                "history_only": group.history_only,
                "tree": (
                    "not applicable"
                    if group.history_only
                    else {
                        "files": len(members),
                        "bytes": sum(f.size for f in members),
                        "largest": [
                            {"path": f.path, "bytes": f.size}
                            for f in sorted(members, key=lambda f: (-f.size, f.path))[:5]
                        ],
                        "lfs_pointers": len(lfs),
                        "lfs_object_bytes": sum(p["size"] for p in lfs),
                        "wheel_files": sum(in_wheel(f.path, globs) for f in members),
                        "wheel_bytes": sum(f.size for f in members if in_wheel(f.path, globs)),
                    }
                ),
                "consumers": (
                    "not searched"
                    if not group.consumers
                    else {"by_area": by_area(hits), "files": hits}
                ),
                "notes": group.notes,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "workflow/data_asset_audit/audit_assets.py inventory",
        "issue": "#2307",
        "categories": CATEGORIES,
        "limitations": list(LIMITATIONS),
        "totals": {
            "files": len(files),
            "bytes": sum(f.size for f in files),
        },
        "groups": rows,
        "samples": sample_rows(ref, files),
        "unassigned_large_files": sorted(unassigned, key=lambda r: r["path"]),
    }


def dump(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- measurement


def clone_refs() -> list[str]:
    """Refs a fresh ``git clone`` downloads: every origin branch and every tag."""
    out = git("for-each-ref", "--format=%(refname)", "refs/remotes/origin", "refs/tags").decode()
    return [r for r in out.splitlines() if r and not r.endswith("/HEAD")]


def history(refs: list[str]) -> dict[str, Any]:
    if not refs:
        return {"refs": 0, "total": {"blobs": 0, "raw": 0, "packed": 0}, "by_group": {}, "largest_blobs": []}
    listing = git("rev-list", "--objects", *refs).decode().splitlines()
    names = {}
    for line in listing:
        oid, _, path = line.partition(" ")
        if path:
            names.setdefault(oid, path)
    check = git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize) %(objectsize:disk)",
        input="\n".join(names).encode() + b"\n",
    ).decode()
    per_group: dict[str, dict[str, int]] = defaultdict(lambda: {"blobs": 0, "raw": 0, "packed": 0})
    largest = []
    total = {"blobs": 0, "raw": 0, "packed": 0}
    for line in check.splitlines():
        oid, kind, raw, packed = line.split()
        if kind != "blob":
            continue
        path = names[oid]
        group = group_of(path, HISTORY_ORDER)
        key = group.id if group else "unassigned"
        for bucket in (per_group[key], total):
            bucket["blobs"] += 1
            bucket["raw"] += int(raw)
            bucket["packed"] += int(packed)
        largest.append((int(packed), int(raw), path))
    largest.sort(reverse=True)
    return {
        "refs": len(refs),
        "total": total,
        "by_group": dict(sorted(per_group.items(), key=lambda kv: -kv[1]["packed"])),
        "largest_blobs": [
            {"path": p, "raw_bytes": r, "packed_bytes": d} for d, r, p in largest[:40]
        ],
    }


def pypi_sizes(project: str = "vaft") -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/json", timeout=20) as resp:
            info = json.load(resp)
    except Exception as exc:  # network is optional for the audit
        return {"error": f"{type(exc).__name__}: {exc}"}
    version = info["info"]["version"]
    return {
        "version": version,
        "files": [
            {"filename": u["filename"], "packagetype": u["packagetype"], "bytes": u["size"]}
            for u in info["urls"]
        ],
    }


def github_size(repo: str = "VEST-Tokamak/vaft") -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/{repo}", "--jq", ".size"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        return {"repository_size_kib": int(out.strip())}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def measure(refs: list[str], *, network: bool) -> dict[str, Any]:
    trees = {}
    for ref in refs:
        files = tree_files(ref)
        per_group: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
        for f in files:
            group = group_of(f.path, (g for g in GROUPS if not g.history_only))
            bucket = per_group[group.id if group else "unassigned"]
            bucket["files"] += 1
            bucket["bytes"] += f.size
        small = {f.oid: f.path for f in files if f.size < 1024}
        lfs = [
            {"path": small[oid], **pointer}
            for oid, data in read_blobs(small)
            if (pointer := lfs_pointer(data)) is not None
        ]
        trees[ref] = {
            "commit": rev_parse(ref),
            "files": len(files),
            "bytes": sum(f.size for f in files),
            "by_group": dict(sorted(per_group.items(), key=lambda kv: -kv[1]["bytes"])),
            "lfs_pointers": sorted(lfs, key=lambda r: r["path"]),
            "lfs_object_bytes": sum(r["size"] for r in lfs),
        }
    scope = clone_refs()
    result = {
        "schema_version": SCHEMA_VERSION,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "method": {
            "tree": "git ls-tree -r -l <ref>; blob sizes as stored in Git (LFS files count as pointers)",
            "history": (
                "git rev-list --objects over every refs/remotes/origin/* branch and tag (what a full "
                "clone downloads), then git cat-file --batch-check objectsize / objectsize:disk"
            ),
            "lfs": "pointer files (< 1 KiB blobs starting with the LFS spec line); object size from the pointer",
            "packages": "PyPI JSON API for the latest release",
            "github": "gh api repos/<repo> .size (KiB, GitHub's own accounting)",
            "local_pack": "git count-objects -v size-pack",
        },
        "limitations": list(LIMITATIONS),
        "trees": trees,
        "history": {"scope": "clone (origin branches + tags)", **history(scope)},
        "local_pack_kib": int(
            dict(
                line.split(": ", 1)
                for line in git("count-objects", "-v").decode().splitlines()
            ).get("size-pack", 0)
        ),
    }
    if network:
        result["pypi"] = pypi_sizes()
        result["github"] = github_size()
    return result


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="write or check the deterministic inventory")
    inv.add_argument("--ref", default="HEAD", help="tree to inventory (default HEAD)")
    inv.add_argument("--check", action="store_true", help="exit 1 if the committed file differs")
    inv.add_argument("--output", type=Path, default=OUTPUT)
    mea = sub.add_parser("measure", help="dated footprint measurement as JSON")
    mea.add_argument("--ref", action="append", dest="refs", help="tree(s) to measure; repeatable")
    mea.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    mea.add_argument("--no-network", action="store_true", help="skip PyPI and GitHub API calls")
    args = parser.parse_args(argv)

    if args.command == "inventory":
        text = dump(build_inventory(args.ref))
        target = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            if current != text:
                print(f"{args.output} is stale; rerun without --check", file=sys.stderr)
                return 1
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
        return 0

    result = measure(args.refs or ["HEAD"], network=not args.no_network)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
