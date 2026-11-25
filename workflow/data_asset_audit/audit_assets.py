#!/usr/bin/env python3
"""Inventory and measure VAFT's data assets for the storage-policy audit (#2307).

Read-only: every number comes from ``git`` plumbing (``ls-tree``, ``cat-file``,
``rev-list``, ``grep``) run against a ref, never from the working tree, so the
audit works in a sparse checkout and never modifies the repository. It does
not import ``vaft``. Because it reads a *ref*, a change to this script or its
groups shows up only after it is committed: commit, regenerate, commit.

Two products:

``docs/architecture/data_assets/inventory.yaml`` (``inventory``)
    Deterministic: sorted rows, repository-relative paths, no timestamps, no
    commit hashes. One row per *asset group* -- the hand-curated table
    :data:`GROUPS` below joined with what the tree at ``--ref`` holds for that
    group (files, bytes, Git LFS pointers, wheel membership, text-search
    consumers) -- plus one row per registered reference sample, read from its
    ``manifest.yaml``, with every SHA-256 the manifest pins checked against the
    stored blob. The audit's own files (:data:`AUDIT_PATHS`) are left out of
    every count, so the inventory does not depend on itself. ``--check`` exits
    1 when the committed file differs from a fresh build. It is a review aid,
    not a CI gate: the consumer lists change with every test that mentions a
    sample.

``measure`` (JSON on stdout or ``--output``)
    Dated and ref-pinned: current-tree bytes per group for each ``--ref``, the
    history (blob versions, raw and packed bytes) per group over the refs a
    fresh clone receives, every Git LFS pointer in that history, every sample
    ``reference_id`` ever registered and whether its bytes are still in
    history, the largest historical blobs, and the published PyPI and GitHub
    sizes. These numbers depend on time and on the local object store, so they
    are never compared by ``--check``; the JSON records the method and its
    limitations.

Limitations (also written into both products):

* Consumers are found by text search for path fragments and API names. Paths
  built at run time (``data_path(f"legacy/{shot}.h5")``) are missed, broad
  patterns over-count, and a hit does not prove the file is read.
* ``rev-list --objects`` names a blob after the first path it meets, so a blob
  stored under two paths is counted once, under one of them.
* Packed (on-disk) sizes depend on how the local repository was packed; the
  GitHub-side size is reported separately from the API.
* Wheel membership is evaluated from ``pyproject.toml`` (package-data globs,
  package ``.py`` files) plus the ``setup.py`` 39915 swap, not by building a
  wheel. The sdist (``MANIFEST.in``) is not modelled; its published size is
  reported.
* In a partial clone (``--filter=blob:none``) absent blobs are counted as
  missing, not measured.

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
import functools
import hashlib
import json
import posixpath
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("docs/architecture/data_assets/inventory.yaml")
SCHEMA_VERSION = 2

#: The audit's own code, test and outputs: excluded from counts and consumers.
AUDIT_PATHS = (
    "workflow/data_asset_audit/**",
    "test/test_data_asset_audit.py",
    "docs/architecture/data_assets/**",
)

#: Where consumers are searched; ``vaft/data`` itself is excluded.
CONSUMER_AREAS = (".github", "docs", "notebooks", "test", "tutorial", "vaft", "workflow")
CONSUMER_EXCLUDES = (
    "vaft/data",
    "docs/assets",
    "docs/architecture/data_assets",
    "workflow/data_asset_audit",
    "test/test_data_asset_audit.py",
)

#: A tree file at least this large that no group claims is listed as unassigned.
UNASSIGNED_THRESHOLD = 1_000_000

LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

LIMITATIONS = (
    "consumers are a text search for path fragments and API names; run-time built paths are missed and broad patterns over-count",
    "a blob stored under several paths is attributed to the first path rev-list reports",
    "packed (on-disk) bytes depend on the local repository's packing",
    "wheel membership is evaluated from pyproject.toml plus the setup.py 39915 swap, not by building a wheel; the sdist is not modelled",
    "in a partial clone, absent blobs are counted as missing, not measured",
)

# Storage categories of #2307 section 4.
CATEGORIES = {
    "source": "Canonical source/configuration (Git)",
    "fixture": "Lightweight offline fixture (Git / package)",
    "reference": "Versioned scientific reference (GitHub Release, pinned by manifest/lockfile)",
    "historical_reference": "Independent historical validation reference (archived Release asset; never purged without an archive or a recorded decision)",
    "regenerable": "Regenerable calculation artifact (generated cache; never the only copy of a published reference)",
    "persistent": "Persistent experimental/computational data (FileDB / HSDS)",
    "public": "External public dataset (upstream registry + cache)",
    "docs_media": "Documentation media (#2117 policy)",
    "undecided": "Decision backlog: not classified, must not be treated as disposable",
}


@dataclass(frozen=True)
class Group:
    """One curated asset group.

    ``paths`` are globs on repository-relative paths in setuptools/glob style:
    ``*`` stays within one directory, ``**`` crosses directories. The first
    group whose pattern matches claims a file, so specific groups come first.
    ``history_only`` groups describe content that is no longer in the audited
    tree but still occupies history.
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


SHOT = "[0-9][0-9][0-9][0-9][0-9]"

GROUPS: tuple[Group, ...] = (
    Group(
        id="samples.shot.manifests",
        paths=(f"vaft/data/samples/{SHOT}/manifest.yaml", f"vaft/data/samples/{SHOT}/*.md"),
        category="source",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout; the manifest is the identity record #2328/#2329 resolve against",
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
        retrieval="vaft.data.sample() -> checkout, else VAFT_CACHE_DIR/references/samples, else Release download (#2329)",
        migration=("#2328", "#2329", "#2331", "#207"),
        decision="decided",
        consumers=(
            r"data/samples/[0-9]{5}/(omas\.json\.gz|imas\.nc)",
            r"sample\( *[0-9]{5}",
            r"sample_ods\(",
            r"sample_manifest\(",
        ),
        notes=(
            "Umbrella #2304 section 3.2. Earlier versions of these files (superseded reference_ids) "
            "are historical references: see the measurement's historical_sample_references."
        ),
    ),
    Group(
        id="samples.shot.sources",
        paths=(f"vaft/data/samples/{SHOT}/source/**",),
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
            "published bytes."
        ),
    ),
    Group(
        id="samples.shot.pinned_parts",
        paths=(
            "vaft/data/samples/48224/efit_magnetic_server/**",
            "vaft/data/samples/48224/equilibria/**",
        ),
        category="fixture",
        current_placement="git",
        proposed_placement="git (moves only together with its sample)",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        consumers=(r"samples/48224/(efit_magnetic_server|equilibria)",),
        notes="Files the 48224 manifest pins by SHA-256 as part of that sample (k-/g-/m-/a-files, equilibria).",
    ),
    Group(
        id="samples.shot.stage_products",
        paths=(f"vaft/data/samples/{SHOT}/**",),
        category="undecided",
        current_placement="git",
        proposed_placement="git (pending size review)",
        retrieval="checkout",
        migration=("#2328",),
        decision="open",
        consumers=(r"data/samples/[0-9]{5}/mhd_stability/",),
        notes=(
            "Per-shot stage products added after the main representations (39915/mhd_stability "
            "CHEASE/GPEC JSON from #2306). Individually small but growing; decide whether they ship "
            "with the shot's Release bundle."
        ),
    ),
    Group(
        id="samples.tables",
        paths=("vaft/data/samples/vest_tier_a/**", "vaft/data/samples/vest_operational_history/**"),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        consumers=(r"vest_tier_a", r"vest_operational_history"),
        notes=(
            "Tier A / operational-history tables (#2277). The snapshot pins source SHA-256 values "
            "byte-exact (.gitattributes -text). Largest file is constraint_points.csv (6.1 MB)."
        ),
    ),
    Group(
        id="samples.wheel_projection",
        paths=("vaft/data/wheel_samples/**",),
        category="fixture",
        current_placement="git; manifest.yaml + omas.json.gz also in the wheel",
        proposed_placement="git (+ wheel for the two swapped files)",
        retrieval="importlib resources from the installed package",
        migration=(),
        decision="decided",
        consumers=(r"wheel_samples",),
        notes=(
            "setup.py copies wheel_samples/39915/{manifest.yaml,omas.json.gz} over samples/39915 at "
            "build time; wheel_samples/39915/imas.nc (10.7 MB) is repository-only. test/verify_dist.py "
            "caps the wheel at 26 MiB."
        ),
    ),
    Group(
        id="samples.kinetic_efit",
        paths=("vaft/data/kineticEfit/**",),
        category="historical_reference",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="proposed",
        consumers=(r"kineticEfit",),
        notes="48224 historical validation reference (OMFIT era, kept by #1712/#1718); its manifest pins these files by SHA-256.",
    ),
    Group(
        id="efit.green_tables_lfs",
        paths=("vaft/data/efit/*.ddd", "vaft/data/efit_table/**"),
        category="reference",
        current_placement="git-lfs",
        proposed_placement="github-release (single baseline) + VAFT_CACHE_DIR/generated/efund for other identities",
        retrieval="exact-identity local table -> EFUND generation -> baseline download -> explicit failure (#2330)",
        migration=("#2330", "#2331", "#207"),
        decision="decided",
        consumers=(r"\.ddd", r"efund_table_manifest", r"rv129129|ec129129|ep129129|rfcoil"),
        notes=(
            "Umbrella #2304 section 3.3. The published baseline is a reference, not a cache entry: #695 "
            "showed table generations change EFIT yield. Two earlier generations exist only in LFS history "
            "(see the measurement's lfs_history)."
        ),
    ),
    Group(
        id="efit.generated_tables_plain",
        paths=("vaft/data/efit/brzgfc.dat", "vaft/data/efit/mhdout.dat"),
        category="reference",
        current_placement="git",
        proposed_placement="github-release (with the .ddd baseline)",
        retrieval="same resolution as efit.green_tables_lfs (#2330)",
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
        notes="g040330.* are 4.3 MB each; the other g-files are under 1.2 MB. The consumer pattern also matches FileDB g-file names and over-counts.",
    ),
    Group(
        id="efit.inputs",
        paths=("vaft/data/efit/**",),
        category="source",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="decided",
        notes="mhdin.dat, efund_table_manifest.json and other EFUND/EFIT text inputs: the identity inputs of the Green tables.",
    ),
    Group(
        id="efit.table_records",
        paths=("workflow/efit_tables/reference/**",),
        category="source",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=("#2330",),
        decision="decided",
        consumers=(r"legacy-39915-129129", r"compare_bundled_vs_legacy"),
        notes=(
            "Identity record and comparison of the pre-#695 packaged table against its regeneration. "
            "Provenance of a historical reference: keep with the archived table."
        ),
    ),
    Group(
        id="legacy.raw_fixtures",
        paths=("vaft/data/legacy/**",),
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
        paths=("vaft/data/sample/**",),
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
        paths=("vaft/data/**",),
        category="source",
        current_placement="git (+ wheel where listed in package-data)",
        proposed_placement="git",
        retrieval="checkout / package resources",
        migration=(),
        decision="decided",
        notes="Machine geometry, code inputs, small registries, vaft/data Python modules and the public-data source registry (bytes are fetched to the public cache).",
    ),
    Group(
        id="notebooks.data",
        paths=("notebooks/data/**",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=("#2331",),
        decision="proposed",
    ),
    Group(
        id="notebooks.outputs",
        paths=("notebooks/**/*.ipynb",),
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
        id="docs.media",
        paths=("docs/assets/**",),
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
        paths=(
            "workflow/**/*.xlsx",
            "workflow/**/*.csv",
            "workflow/**/*.csv.gz",
            "workflow/**/*.json",
            "workflow/**/*.yml",
        ),
        category="undecided",
        current_placement="git",
        proposed_placement="decision backlog",
        retrieval="checkout",
        migration=(),
        decision="open",
        notes="Dated study outputs and records committed next to workflows (e.g. automatic_pipeline_3_data_summary/*.xlsx 6.9 MB). Not disposable: some are provenance.",
    ),
    Group(
        id="test.fixtures",
        paths=("test/data/**",),
        category="fixture",
        current_placement="git",
        proposed_placement="git",
        retrieval="checkout",
        migration=(),
        decision="decided",
    ),
    # History only.
    Group(
        id="history.retired_layouts",
        paths=(
            "vaft/data/omas/**",
            "vaft/data/imas/**",
            "vaft/data/[0-9][0-9][0-9][0-9][0-9].json",
            "vaft/data/*_imas_equilibrium.h5",
            "packaging/wheel_samples/**",
        ),
        category="historical_reference",
        current_placement="history only",
        proposed_placement="archive or retire item by item before #207",
        retrieval="git history",
        migration=("#207",),
        decision="open",
        history_only=True,
        notes=(
            "Earlier sample layouts: vaft/data/omas/{39915,41524,41672}.json, an IMAS HDF5 set under "
            "vaft/data/imas, vaft/data/{39915,39916,39917}.json, 44740_imas_equilibrium.h5 and "
            "packaging/wheel_samples/39915. 39916, 39917 and 44740 have no registered sample "
            "(39917 is a confinement reference shot), and the others are not proven identical to the "
            "registered samples. None is superseded until shown to be."
        ),
    ),
    Group(
        id="history.gh_pages_current",
        paths=(),
        category="docs_media",
        current_placement="origin/gh-pages (single commit)",
        proposed_placement="per #2117",
        retrieval="generated by docs.yml",
        migration=("#2117",),
        decision="decided",
        history_only=True,
        notes="Attributed by ref, not by path: blobs reachable from origin/gh-pages and from no code branch.",
    ),
    Group(
        id="history.gh_pages_backups",
        paths=(),
        category="docs_media",
        current_placement="backup/gh-pages-* branches and archive/gh-pages-* tags",
        proposed_placement="retention decision under #2117",
        retrieval="git history",
        migration=("#2117",),
        decision="open",
        history_only=True,
        notes=(
            "Attributed by ref: blobs reachable only from gh-pages backup branches or archive tags. "
            "Recovering this space means deleting a backup branch and an archive tag."
        ),
    ),
)

#: History attribution checks the retired layouts first: the ``vaft/data/**``
#: catch-all would otherwise claim them.
HISTORY_ORDER: tuple[Group, ...] = tuple(g for g in GROUPS if g.history_only) + tuple(
    g for g in GROUPS if not g.history_only
)
TREE_GROUPS: tuple[Group, ...] = tuple(g for g in GROUPS if not g.history_only)


# --------------------------------------------------------------------------- globbing


@functools.lru_cache(maxsize=None)
def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.index("]", i)
            out.append(pattern[i : end + 1])
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(path: str, pattern: str) -> bool:
    """Glob match where ``*`` stays within a directory and ``**`` crosses them."""
    return _glob_regex(pattern).match(path) is not None


def is_audit_path(path: str) -> bool:
    return any(glob_match(path, pattern) for pattern in AUDIT_PATHS)


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


def tree_files(ref: str, *, include_audit: bool = False) -> list[TreeFile]:
    out = git("ls-tree", "-r", "-l", "-z", ref)
    files = []
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        if kind != b"blob":
            continue
        name = path.decode()
        if not include_audit and is_audit_path(name):
            continue
        files.append(TreeFile(name, int(size), oid.decode()))
    return files


def read_blobs(oids: Iterable[str]) -> Iterator[tuple[str, bytes | None]]:
    """Yield ``(oid, content)`` with one ``cat-file --batch`` process; ``None`` if absent."""
    oids = list(dict.fromkeys(oids))
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
            if len(header) < 3 or header[1] == b"missing":
                yield oid, None
                continue
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            yield oid, data
    finally:
        proc.stdin.close()
        proc.wait()


def batch_check(oids: Iterable[str]) -> dict[str, tuple[str, int, int] | None]:
    """``oid -> (type, size, disk size)``, or ``None`` for an absent object."""
    oids = list(dict.fromkeys(oids))
    if not oids:
        return {}
    out = git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize) %(objectsize:disk)",
        input="\n".join(oids).encode() + b"\n",
    ).decode()
    result: dict[str, tuple[str, int, int] | None] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            result[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
        elif parts:
            result[parts[0]] = None
    return result


def lfs_object_dir() -> Path:
    common = git("rev-parse", "--git-common-dir").decode().strip()
    base = Path(common)
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / "lfs" / "objects"


# --------------------------------------------------------------------------- classification


def group_of(path: str, groups: Iterable[Group]) -> Group | None:
    for group in groups:
        if any(glob_match(path, pattern) for pattern in group.paths):
            return group
    return None


def lfs_pointer(data: bytes | None) -> dict[str, Any] | None:
    if not data or not data.startswith(LFS_PREFIX):
        return None
    fields = dict(
        line.split(" ", 1) for line in data.decode().splitlines() if " " in line
    )
    return {"oid": fields.get("oid", ""), "size": int(fields.get("size", 0))}


@dataclass(frozen=True)
class WheelRules:
    data_globs: tuple[str, ...]
    package_excludes: tuple[str, ...]


def wheel_rules(ref: str) -> WheelRules:
    pyproject = tomllib.loads(git("show", f"{ref}:pyproject.toml").decode())
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    data = setuptools.get("package-data", {})
    find = setuptools.get("packages", {}).get("find", {}) if isinstance(setuptools.get("packages"), dict) else {}
    return WheelRules(
        data_globs=tuple(f"vaft/{pattern}" for pattern in data.get("vaft", [])),
        package_excludes=tuple(find.get("exclude", [])),
    )


#: ``setup.py``'s ``build_py`` hook copies these wheel_samples files over the
#: package-data matches of the same name under ``samples/39915`` at build time.
WHEEL_SUBSTITUTES = {
    "vaft/data/wheel_samples/39915/manifest.yaml": "vaft/data/samples/39915/manifest.yaml",
    "vaft/data/wheel_samples/39915/omas.json.gz": "vaft/data/samples/39915/omas.json.gz",
}


def in_wheel(path: str, rules: WheelRules) -> bool:
    if path in WHEEL_SUBSTITUTES:
        return True
    if path in WHEEL_SUBSTITUTES.values():
        return False
    if path.startswith("vaft/") and path.endswith(".py"):
        package = posixpath.dirname(path).replace("/", ".")
        return not any(fnmatch.fnmatchcase(package, pattern) for pattern in rules.package_excludes)
    return any(glob_match(path, pattern) for pattern in rules.data_globs)


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


# --------------------------------------------------------------------------- manifests


def manifest_pins(manifest: Any) -> list[tuple[str, str, str]]:
    """Every ``(field, path, sha256)`` a sample manifest pins.

    Two shapes occur: a mapping with ``path`` and ``sha256``, and a sibling pair
    ``<name>``/``<name>_sha256`` (or ``<name>_source``/``<name>_sha256``).
    ``previous_*`` hashes are history records, not pins.
    """
    pins: list[tuple[str, str, str]] = []

    def visit(node: Any, trail: list[str]) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
                pins.append((".".join(trail) or "<root>", node["path"], node["sha256"]))
            for key, value in node.items():
                key = str(key)
                if key.endswith("_sha256") and isinstance(value, str) and not key.startswith("previous"):
                    base = key[: -len("_sha256")]
                    for candidate in (base, f"{base}_source"):
                        target = node.get(candidate)
                        if isinstance(target, str):
                            pins.append((".".join(trail + [candidate]), target, value))
                            break
                visit(value, trail + [key])
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, trail + [str(index)])

    visit(manifest, [])
    return pins


def resolve_pin(path: str, directory: str, tree: dict[str, TreeFile]) -> str | None:
    """Repository path of a pinned file, trying sample-, data- and repo-relative forms."""
    if " " in path.strip():
        return None  # prose such as "FileDB raw/45531/..." names an external location
    candidates = [
        posixpath.normpath(posixpath.join(directory, path)),
        posixpath.normpath(posixpath.join("vaft/data", path)),
        posixpath.normpath(path),
    ]
    found = next((c for c in candidates if c in tree), None)
    if found is None and "/" not in path:
        # A bare file name (e.g. an em-coupling asset): accept a unique match under vaft/data.
        matches = [p for p in tree if p.startswith("vaft/data/") and posixpath.basename(p) == path]
        found = matches[0] if len(matches) == 1 else None
    return found


def content_sha256(data: bytes) -> str:
    """SHA-256 of the content; for an LFS pointer, the object's own SHA-256."""
    pointer = lfs_pointer(data)
    if pointer and pointer["oid"].startswith("sha256:"):
        return pointer["oid"][len("sha256:") :]
    return hashlib.sha256(data).hexdigest()


def find_key(node: Any, wanted: str) -> Any:
    if isinstance(node, dict):
        if wanted in node:
            return node[wanted]
        for value in node.values():
            found = find_key(value, wanted)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, wanted)
            if found is not None:
                return found
    return None


def sample_consumer_patterns(shot: int) -> tuple[str, ...]:
    return (
        rf"samples/{shot}([^0-9]|$)",
        rf"sample(_ods|_manifest)?\( *(shot *= *)?{shot}([^0-9]|$)",
    )


# --------------------------------------------------------------------------- inventory


def sample_rows(ref: str, files: list[TreeFile]) -> list[dict[str, Any]]:
    """One row per registered sample manifest, every pin checked against the stored blob."""
    tree = {f.path: f for f in files}
    manifests = sorted(
        f for f in tree if re.fullmatch(r"vaft/data/samples/\d{5}/manifest\.yaml", f)
    )
    loaded = {}
    wanted_oids = set()
    for manifest_path in manifests:
        directory = manifest_path.rsplit("/", 1)[0]
        manifest = yaml.safe_load(git("show", f"{ref}:{manifest_path}")) or {}
        resolved = [
            (field, raw, sha, resolve_pin(raw, directory, tree))
            for field, raw, sha in manifest_pins(manifest)
        ]
        wanted_oids.update(tree[r].oid for *_, r in resolved if r)
        loaded[manifest_path] = (directory, manifest, resolved)
    digests = {oid: content_sha256(data) for oid, data in read_blobs(wanted_oids) if data is not None}

    rows = []
    for manifest_path in manifests:
        directory, manifest, resolved = loaded[manifest_path]
        pins = []
        for field, raw, sha, path in resolved:
            if path is None:
                status = "not in tree"
            else:
                stored = digests.get(tree[path].oid)
                status = "missing blob" if stored is None else ("match" if stored == sha else "mismatch")
            pins.append({"field": field, "path": path or raw, "sha256": sha, "status": status})
        counts: dict[str, int] = defaultdict(int)
        for pin in pins:
            counts[pin["status"]] += 1
        representations = [
            {
                "name": name,
                "path": f"{directory}/{record.get('path', '')}",
                "package": record.get("package", "not recorded"),
                "bytes": tree[f"{directory}/{record.get('path', '')}"].size
                if f"{directory}/{record.get('path', '')}" in tree
                else "missing",
            }
            for name, record in sorted((manifest.get("representations") or {}).items())
        ]
        source = manifest.get("source") or {}
        generation = manifest.get("generation") or {}
        required_ids = find_key(manifest, "required_ids")
        directory_files = [f for f in files if f.path.startswith(directory + "/")]
        shot = manifest.get("shot", "not recorded")
        hits = consumers(ref, sample_consumer_patterns(shot)) if isinstance(shot, int) else []
        rows.append(
            {
                "shot": shot,
                "directory": directory,
                "reference_id": manifest.get("reference_id", "not recorded"),
                "imas_dd_version": manifest.get("imas_dd_version", "not recorded"),
                "source_kind": source.get("kind", "not recorded"),
                "source_git_commit": source.get("git_commit", "not recorded"),
                "authority": source.get("filedb_root", "not recorded"),
                "canonical_source": (
                    f"{directory}/{generation['canonical_source']}"
                    if generation.get("canonical_source")
                    else "not recorded"
                ),
                "generation_command": generation.get("command", "not recorded"),
                "required_ids": required_ids if required_ids is not None else "not recorded",
                "recorded_sections": sorted(str(k) for k in manifest),
                "representations": representations,
                "pins": pins,
                "pin_status": dict(sorted(counts.items())),
                "directory_files": len(directory_files),
                "directory_bytes": sum(f.size for f in directory_files),
                "consumers": {"by_area": by_area(hits), "files": hits},
            }
        )
    return rows


def build_inventory(ref: str) -> dict[str, Any]:
    files = tree_files(ref)
    rules = wheel_rules(ref)
    claimed: dict[str, list[TreeFile]] = defaultdict(list)
    unassigned = []
    for f in files:
        group = group_of(f.path, TREE_GROUPS)
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
                        "wheel_files": sum(in_wheel(f.path, rules) for f in members),
                        "wheel_bytes": sum(f.size for f in members if in_wheel(f.path, rules)),
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
        "excluded_audit_paths": list(AUDIT_PATHS),
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


def _objects(refs: list[str]) -> dict[str, str]:
    """``oid -> first path`` for every object reachable from ``refs`` (paths only for blobs/trees)."""
    if not refs:
        return {}
    names: dict[str, str] = {}
    for line in git("rev-list", "--objects", *refs).decode().splitlines():
        oid, _, path = line.partition(" ")
        names.setdefault(oid, path)
    return names


def _first_commit(blob: str, refs: list[str]) -> str:
    out = git(
        "log", "--reverse", "--format=%h %ad", "--date=short", f"--find-object={blob}", *refs
    ).decode().splitlines()
    return out[0] if out else "unknown"


def history(refs: list[str]) -> dict[str, Any]:
    empty = {"blobs": 0, "raw": 0, "packed": 0, "missing": 0}
    if not refs:
        return {
            "refs": 0,
            "total": dict(empty),
            "by_group": {},
            "largest_blobs": [],
            "lfs_history": [],
            "historical_sample_references": [],
            "retired_paths": [],
        }
    pages_refs = [r for r in refs if "gh-pages" in r]
    code_refs = [r for r in refs if r not in pages_refs]
    current_pages = [r for r in pages_refs if r.endswith("/origin/gh-pages")]
    code = _objects(code_refs)
    pages_current = _objects(current_pages)
    pages_all = _objects(pages_refs)
    names = dict(code)
    for oid, path in pages_all.items():
        names.setdefault(oid, path)
    info = batch_check(names)

    per_group: dict[str, dict[str, int]] = defaultdict(lambda: dict(empty))
    total = dict(empty)
    largest = []
    small_blobs: dict[str, str] = {}
    sample_blobs: dict[str, str] = {}
    manifest_blobs: dict[str, str] = {}
    retired: dict[str, dict[str, int]] = defaultdict(lambda: {"blobs": 0, "raw": 0, "packed": 0})
    for oid, path in names.items():
        meta = info.get(oid)
        if oid in code:
            group = group_of(path, HISTORY_ORDER)
            key = group.id if group else "unassigned"
        elif oid in pages_current:
            key = "history.gh_pages_current"
        else:
            key = "history.gh_pages_backups"
        if meta is None:
            per_group[key]["missing"] += 1
            total["missing"] += 1
            continue
        kind, raw, packed = meta
        if kind != "blob":
            continue
        for bucket in (per_group[key], total):
            bucket["blobs"] += 1
            bucket["raw"] += raw
            bucket["packed"] += packed
        largest.append((packed, raw, path))
        if oid in code:
            if raw < 1024:
                small_blobs[oid] = path
            if key == "samples.shot.representations":
                sample_blobs[oid] = path
            if re.fullmatch(r"vaft/data/samples/\d{5}/manifest\.yaml", path):
                manifest_blobs[oid] = path
            if key == "history.retired_layouts":
                bucket = retired[path]
                bucket["blobs"] += 1
                bucket["raw"] += raw
                bucket["packed"] += packed

    # Every LFS pointer ever committed, and whether its object is in the local LFS store.
    lfs_store = lfs_object_dir()
    lfs_history = []
    for oid, data in read_blobs(small_blobs):
        pointer = lfs_pointer(data)
        if pointer is None:
            continue
        digest = pointer["oid"].removeprefix("sha256:")
        local = (lfs_store / digest[:2] / digest[2:4] / digest).is_file()
        lfs_history.append(
            {
                "path": small_blobs[oid],
                "object_sha256": digest,
                "bytes": pointer["size"],
                "first_commit": _first_commit(oid, code_refs),
                "object_in_local_store": local,
            }
        )

    # Every sample reference_id ever registered, and whether its bytes survive in history.
    present = {
        content_sha256(data) for _, data in read_blobs(sample_blobs) if data is not None
    }
    references: dict[str, dict[str, Any]] = {}
    for oid, data in read_blobs(manifest_blobs):
        if data is None:
            continue
        try:
            manifest = yaml.safe_load(data) or {}
        except yaml.YAMLError:
            continue
        reference_id = manifest.get("reference_id")
        if not reference_id:
            continue
        entry = references.setdefault(
            reference_id,
            {"reference_id": reference_id, "shot": manifest.get("shot"), "representations": {}},
        )
        for name, record in (manifest.get("representations") or {}).items():
            sha = record.get("sha256") if isinstance(record, dict) else None
            if sha:
                entry["representations"][f"{name}:{sha}"] = {
                    "name": name,
                    "sha256": sha,
                    "bytes_in_history": sha in present,
                }

    largest.sort(reverse=True)
    return {
        "refs": len(refs),
        "gh_pages_refs": pages_refs,
        "total": total,
        "by_group": dict(sorted(per_group.items(), key=lambda kv: -kv[1]["packed"])),
        "largest_blobs": [
            {"path": p, "raw_bytes": r, "packed_bytes": d} for d, r, p in largest[:40]
        ],
        "lfs_history": sorted(lfs_history, key=lambda r: (r["path"], r["first_commit"])),
        "historical_sample_references": [
            {**{k: v for k, v in entry.items() if k != "representations"},
             "representations": sorted(entry["representations"].values(), key=lambda r: (r["name"], r["sha256"]))}
            for entry in sorted(references.values(), key=lambda e: e["reference_id"])
        ],
        "retired_paths": [
            {"path": path, **sizes} for path, sizes in sorted(retired.items())
        ],
    }


def pypi_sizes(project: str = "vaft") -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/json", timeout=20) as resp:
            info = json.load(resp)
    except Exception as exc:  # network is optional for the audit
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "version": info["info"]["version"],
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
        files = tree_files(ref, include_audit=True)
        per_group: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
        for f in files:
            group = group_of(f.path, TREE_GROUPS)
            bucket = per_group[group.id if group else "unassigned"]
            bucket["files"] += 1
            bucket["bytes"] += f.size
        small = {f.oid: f.path for f in files if f.size < 1024}
        lfs = [
            {"path": small[oid], **pointer}
            for oid, data in read_blobs(small)
            if (pointer := lfs_pointer(data)) is not None
        ]
        info = batch_check(f.oid for f in files)
        trees[ref] = {
            "commit": rev_parse(ref),
            "files": len(files),
            "bytes": sum(f.size for f in files),
            "packed_bytes": sum(m[2] for m in info.values() if m),
            "missing_blobs": sum(m is None for m in info.values()),
            "by_group": dict(sorted(per_group.items(), key=lambda kv: -kv[1]["bytes"])),
            "lfs_pointers": sorted(lfs, key=lambda r: r["path"]),
            "lfs_object_bytes": sum(r["size"] for r in lfs),
        }
    scope = clone_refs()
    result = {
        "schema_version": SCHEMA_VERSION,
        "measured_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "method": {
            "tree": "git ls-tree -r -l <ref>; blob sizes as stored in Git (LFS files count as pointers); packed_bytes = local objectsize:disk of the tree's blobs",
            "history": (
                "git rev-list --objects over every refs/remotes/origin/* branch and tag (what a full "
                "clone downloads), then git cat-file --batch-check objectsize / objectsize:disk. "
                "Blobs reachable from no code branch are attributed by ref to the gh-pages groups."
            ),
            "lfs": "pointer files (< 1 KiB blobs starting with the LFS spec line); object size from the pointer; presence checked in the local .git/lfs/objects",
            "historical_sample_references": "every vaft/data/samples/<shot>/manifest.yaml blob in history; bytes_in_history = a representation blob in history hashes to the recorded sha256",
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
