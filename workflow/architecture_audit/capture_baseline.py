#!/usr/bin/env python3
"""Capture frozen, array-by-array baselines of the offline stage chain.

The preprocessing consolidation moves configuration, magnetics mapping and
preprocessing into canonical modules.  Its numerical-equivalence evidence is a
comparison against what the *current* implementation produces from the same
frozen inputs -- so that "current" has to be recorded, per array, before the
migration starts.  This script records it.

For each packaged sample that ships its raw DAQ dump (39915, 41524, 41672,
42962 -- see ``test/data/consolidation/case_manifest.yaml``) it runs the four
pipeline-1 stage scripts exactly as the Snakefile invokes them,

    static -> diagnostics -> eddy -> constraints

with the sample's own frozen ``source/pipeline-config.yaml`` supplying every
stage parameter, and writes one JSON per shot holding, for every ODS leaf of
every stage product, a sha256 of its canonical bytes plus summary statistics
(shape, dtype, finite count, min, max, mean, rms).  A later run of the same
script on the migrated code is diffed against it leaf by leaf; a summary
statistic says *how much* a hash mismatch moved.

Isolation
---------
* Every product is written under ``--output-root``, which is required, must be
  empty or absent, and is refused if it lies anywhere inside this repository,
  under ``/srv``, or under ``$VAFT_FILEDB_DIR`` as the caller's shell has it.
  Nothing here writes into a stored product.
* The real HSDS protection is that the four stage scripts never import an
  HSDS client and HSDS replication is a separate pipeline rule
  (``replicate_to_hsds.py``) that is never invoked.  On top of that the
  children run with ``cwd`` = the root and ``HOME`` = ``<root>/home`` (so
  neither ``./.hscfg`` nor ``~/.hscfg`` is found), ``HS_ENDPOINT`` set to an
  unroutable address (an environment variable overrides any config file), the
  caller's other ``HS_*`` / ``H5PYD_*`` variables removed, and
  ``VAFT_FILEDB_DIR`` pointed into the root.
* Parent and children import the ``vaft`` of the checkout this script lives in
  (``sys.path`` / ``PYTHONPATH`` are set to it), never whatever an installed
  or main checkout would resolve to, so the recorded revision and machine era
  belong to the code that built the products.

Usage::

    python workflow/architecture_audit/capture_baseline.py \\
        --output-root /path/to/scratch/baseline-pre-migration \\
        --shot 39915

Do not point it at a production root; do not commit its output.  The compact
summary is the committable form::

    python workflow/architecture_audit/capture_baseline.py summarize \\
        /path/to/scratch/baseline-pre-migration --code-sha <sha> \\
        --output test/data/consolidation/baseline_<sha>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path[:1]:
    # Path-run scripts otherwise import whichever `vaft` is installed (often the
    # main checkout), not the one whose stage scripts the children run.
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
SAMPLES_DIR = REPO_ROOT / "vaft" / "data" / "samples"
PIPELINE_DIR = REPO_ROOT / "workflow" / "automatic_pipeline_1_routine_data_processing"

#: Packaged samples that carry the raw DAQ dump the chain starts from.
SHOTS_WITH_RAW: tuple[int, ...] = (39915, 41524, 41672, 42962)
STAGES: tuple[str, ...] = ("static", "diagnostics", "eddy", "constraints")
BASELINE_SCHEMA = "vaft.consolidation.baseline/1"

#: Environment variables an HSDS client would read; removed from the children.
_HSDS_ENV_PREFIXES = ("HS_", "H5PYD_")
#: TEST-NET-1 (RFC 5737): never routed, so a stray HSDS client fails fast.
UNROUTABLE_HSDS_ENDPOINT = "http://192.0.2.1:1"


class IsolationError(RuntimeError):
    """The requested output root is not an isolated scratch location."""


# ---------------------------------------------------------------------------
# Canonical leaf hashing
# ---------------------------------------------------------------------------

def _canonical_array(value: np.ndarray) -> np.ndarray:
    """Native-endian, C-contiguous; every NaN mapped to one bit pattern.

    Two NaNs with different payloads are the same physics; their raw bytes are
    not.  Negative zero is kept distinct on purpose -- a sign flip that only
    shows at zero is still a sign flip.
    """
    # np.array(order="C"), not np.ascontiguousarray: the latter promotes a 0-d
    # scalar to shape (1,), which would hash ``1.0`` and ``[1.0]`` alike.
    array = np.array(value, order="C")
    if array.dtype.byteorder not in ("=", "|"):
        array = array.astype(array.dtype.newbyteorder("="))
    if np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.complexfloating):
        array = array.copy()
        if np.issubdtype(array.dtype, np.complexfloating):
            real, imag = array.real.copy(), array.imag.copy()
            real[np.isnan(real)] = np.nan
            imag[np.isnan(imag)] = np.nan
            array = real + 1j * imag
        else:
            array[np.isnan(array)] = np.nan
    return array


def leaf_record(value: Any) -> dict[str, Any]:
    """Hash and summarize one ODS leaf.

    The hash covers dtype, shape and bytes, so ``[1.0]`` and ``1.0`` and
    ``[1]`` are three different leaves.  Strings hash their UTF-8 encoding.
    """
    if isinstance(value, (str, bytes)):
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return {"kind": "string", "sha256": hashlib.sha256(raw).hexdigest(), "length": len(raw)}

    array = np.asarray(value)
    if array.dtype.kind in "OUS":
        text = json.dumps(array.tolist(), sort_keys=True, default=str).encode("utf-8")
        return {
            "kind": "object",
            "shape": list(array.shape),
            "sha256": hashlib.sha256(text).hexdigest(),
        }

    canonical = _canonical_array(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes())
    record: dict[str, Any] = {
        "kind": "numeric",
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": digest.hexdigest(),
    }
    if canonical.dtype.kind in "biuf" and canonical.size:
        numbers = canonical.astype(float, copy=False).reshape(-1)
        finite = numbers[np.isfinite(numbers)]
        record["n_finite"] = int(finite.size)
        record["n_nan"] = int(np.isnan(numbers).sum())
        if finite.size:
            record.update(
                {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "rms": float(np.sqrt(np.mean(finite * finite))),
                }
            )
    return record


def ods_leaf_records(ods: Any) -> dict[str, dict[str, Any]]:
    """``{flat path: leaf_record}`` for every leaf of an ODS, sorted by path.

    ``ODS.flat()`` only *reads*; it is called once on a freshly loaded product
    that is discarded afterwards, so OMAS's create-on-read behaviour cannot
    leak into anything stored.
    """
    flat = ods.flat()
    return {str(path): leaf_record(flat[path]) for path in sorted(flat, key=str)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def check_output_root(
    root: Path, *, environ: Mapping[str, str] | None = None, require_empty: bool = True
) -> Path:
    """Refuse any output root that is not an empty, isolated scratch location."""
    env = os.environ if environ is None else environ
    resolved = Path(root).expanduser().resolve()
    forbidden: list[Path] = [REPO_ROOT, Path("/srv")]
    filedb = env.get("VAFT_FILEDB_DIR")
    if filedb:
        forbidden.append(Path(filedb))
    for parent in forbidden:
        if _is_within(resolved, parent):
            raise IsolationError(
                f"output root {resolved} lies inside {parent}; baselines go to an "
                "isolated scratch directory, never into stored products or a checkout"
            )
    if resolved.exists() and not resolved.is_dir():
        raise IsolationError(f"output root {resolved} exists and is not a directory")
    if require_empty and resolved.exists() and any(resolved.iterdir()):
        raise IsolationError(
            f"output root {resolved} is not empty; pass a fresh directory so no "
            "existing product can be overwritten or mistaken for this run's"
        )
    return resolved


def child_environment(root: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The stage scripts' environment: this checkout's vaft, no reachable HSDS."""
    env = {
        key: value
        for key, value in (os.environ if environ is None else environ).items()
        if not key.startswith(_HSDS_ENV_PREFIXES)
    }
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    env["VAFT_FILEDB_DIR"] = str(Path(root) / "filedb")
    env["HOME"] = str(Path(root) / "home")  # no ~/.hscfg
    env["USERPROFILE"] = env["HOME"]
    env["HS_ENDPOINT"] = UNROUTABLE_HSDS_ENDPOINT  # overrides any .hscfg
    return env


# ---------------------------------------------------------------------------
# Stage invocation (mirrors the pipeline-1 Snakefile rules)
# ---------------------------------------------------------------------------

def _csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def stage_commands(shot: int, config: Mapping[str, Any], machine_version: str, out: Path) -> list[tuple[str, list[str], dict[str, Path]]]:
    """``(stage, argv, products)`` for the four stages, as the Snakefile builds them."""
    raw = SAMPLES_DIR / str(shot) / "source" / f"vest_{shot}_daq_raw.json.gz"
    diagnostics_cfg = config.get("diagnostics", {}) or {}
    eddy_cfg = config.get("eddy", {}) or {}
    filament = eddy_cfg.get("filament", {}) or {}
    constraints_cfg = config.get("constraints", {}) or {}
    processing = ((config.get("vest", {}) or {}).get("magnetics", {}) or {}).get("processing", {}) or {}

    paths = {
        "static_ods": out / "static" / f"{machine_version}.json",
        "static_manifest": out / "static" / f"{machine_version}.manifest.json",
        "diagnostics_ods": out / "diagnostics" / "omas.json",
        "diagnostics_manifest": out / "diagnostics" / "manifest.json",
        "eddy_ods": out / "eddy" / "omas.json",
        "eddy_manifest": out / "eddy" / "manifest.json",
        "constraints_ods": out / "constraints" / "omas.json",
    }
    py = sys.executable
    return [
        (
            "static",
            [py, str(PIPELINE_DIR / "generate_static_ods.py"),
             "--machine-version", machine_version,
             "--output", str(paths["static_ods"]),
             "--metadata", str(paths["static_manifest"])],
            {"ods": paths["static_ods"], "manifest": paths["static_manifest"]},
        ),
        (
            "diagnostics",
            [py, str(PIPELINE_DIR / "generate_diagnostics_ods.py"),
             "--shot", str(shot),
             "--raw-dump", str(raw),
             "--static-ods", str(paths["static_ods"]),
             "--output", str(paths["diagnostics_ods"]),
             "--metadata", str(paths["diagnostics_manifest"]),
             "--tstart", str(diagnostics_cfg.get("tstart", 0.26)),
             "--tend", str(diagnostics_cfg.get("tend", 0.36)),
             "--dt", str(diagnostics_cfg.get("dt", 4e-5)),
             "--run", str(diagnostics_cfg.get("run", 1)),
             "--vest-magnetics-processing-json", json.dumps(processing)],
            {"ods": paths["diagnostics_ods"], "manifest": paths["diagnostics_manifest"]},
        ),
        (
            "eddy",
            [py, str(PIPELINE_DIR / "generate_eddy_ods.py"),
             "--shot", str(shot),
             "--diagnostics-ods", str(paths["diagnostics_ods"]),
             "--diagnostics-manifest", str(paths["diagnostics_manifest"]),
             "--static-ods", str(paths["static_ods"]),
             "--output", str(paths["eddy_ods"]),
             "--metadata", str(paths["eddy_manifest"]),
             "--filament-r", _csv(filament.get("r", [0.35, 0.35, 0.35])),
             "--filament-z", _csv(filament.get("z", [0.25, 0.0, -0.25])),
             "--filament-fraction", _csv(filament.get("fraction", [0.3333333] * 3)),
             "--dt-sub", str(eddy_cfg.get("dt_sub", 5e-5))],
            {"ods": paths["eddy_ods"], "manifest": paths["eddy_manifest"]},
        ),
        (
            "constraints",
            [py, str(PIPELINE_DIR / "generate_constraints_ods.py"),
             "--shot", str(shot),
             "--eddy-ods", str(paths["eddy_ods"]),
             "--diagnostics-ods", str(paths["diagnostics_ods"]),
             "--eddy-manifest", str(paths["eddy_manifest"]),
             "--diagnostics-manifest", str(paths["diagnostics_manifest"]),
             "--output", str(paths["constraints_ods"]),
             # Written into k-files only; left empty so no machine-local table
             # path enters the product (recorded in the baseline).
             "--efit-table-dir", "",
             "--timeset", str(constraints_cfg.get("timeset", "auto")),
             "--tstep", str(constraints_cfg.get("tstep", 0.001)),
             "--average-window", str(constraints_cfg.get("average_window", 0.0005)),
             "--uncertainty", _csv(constraints_cfg.get("uncertainty", [])) or _csv(
                 [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2]),
             "--weighting", _csv(constraints_cfg.get("weighting", [])) or _csv(
                 [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01]),
             "--detect-broken", "true" if constraints_cfg.get("detect_broken", False) else "false",
             "--fl-correct-option", str(constraints_cfg.get("fl_correct_option", 0)),
             "--gaussian-fit-option", str(constraints_cfg.get("gaussian_fit_option", 1))],
            {"ods": paths["constraints_ods"]},
        ),
    ]


def _git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _versions() -> dict[str, str]:
    import scipy

    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    try:
        import omas

        out["omas"] = getattr(omas, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        out["omas"] = "missing"
    return out


def capture_shot(shot: int, root: Path, *, stages: Sequence[str] = STAGES, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run the stage chain for one packaged shot into ``root/<shot>`` and hash it.

    ``root`` is checked for isolation here too (the shot directory must be
    fresh), not only by :func:`main`.
    """
    import yaml

    from vaft.database._local import load_ods
    from vaft.omas.vest_upstream import machine_era_for_shot

    if int(shot) not in SHOTS_WITH_RAW:
        raise ValueError(f"shot {shot} has no packaged raw dump; choose from {SHOTS_WITH_RAW}")
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; choose from {STAGES}")
    sample = SAMPLES_DIR / str(shot) / "source"
    config_path = sample / "pipeline-config.yaml"
    raw_path = sample / f"vest_{shot}_daq_raw.json.gz"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    machine_version = machine_era_for_shot(int(shot)).name

    root = Path(root).expanduser().resolve()
    check_output_root(root, environ=environ, require_empty=False)  # isolated location
    out = check_output_root(root / str(shot), environ=environ)  # and fresh per shot
    (root / "home").mkdir(parents=True, exist_ok=True)
    env = child_environment(root, environ=environ)
    record: dict[str, Any] = {
        "schema": BASELINE_SCHEMA,
        "shot": int(shot),
        "machine_version": machine_version,
        "revision": _git_revision(),
        "versions": _versions(),
        "inputs": {
            "raw_dump": {"path": str(raw_path.relative_to(REPO_ROOT)), "sha256": sha256_file(raw_path)},
            "pipeline_config": {"path": str(config_path.relative_to(REPO_ROOT)), "sha256": sha256_file(config_path)},
        },
        "isolation": {
            "hsds_replication": "not invoked (replicate_to_hsds.py is never called)",
            "hsds_endpoint": UNROUTABLE_HSDS_ENDPOINT,
            "child_cwd": "output root",
            "hsds_env_removed": sorted(
                key for key in (os.environ if environ is None else environ) if key.startswith(_HSDS_ENV_PREFIXES)
            ),
            "efit_table_dir": "",
        },
        "stages": {},
    }
    for stage, argv, products in stage_commands(int(shot), config, machine_version, out):
        if stage not in stages:
            continue
        for product in products.values():
            product.parent.mkdir(parents=True, exist_ok=True)
        log_path = out / "logs" / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                argv, env=env, cwd=str(root), stdout=log, stderr=subprocess.STDOUT, check=False
            )
        entry: dict[str, Any] = {
            "argv": [Path(argv[1]).name, *argv[2:]],
            "returncode": int(result.returncode),
            "seconds": round(time.monotonic() - started, 3),
        }
        if result.returncode != 0 or not products["ods"].exists():
            entry["status"] = "failed"
            entry["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            record["stages"][stage] = entry
            # Downstream stages consume this product; stop rather than hash
            # something built from a missing input.
            break
        ods, _ = load_ods(products["ods"])
        entry["status"] = "captured"
        entry["product_sha256"] = sha256_file(products["ods"])
        if "manifest" in products and products["manifest"].exists():
            entry["manifest_sha256"] = sha256_file(products["manifest"])
        entry["leaves"] = ods_leaf_records(ods)
        record["stages"][stage] = entry
    baseline_path = out / "baseline.json"
    baseline_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Compact, committable summary of a captured root
# ---------------------------------------------------------------------------

SUMMARY_SCHEMA = "vaft.consolidation.baseline-summary/1"
#: Hex digits kept of each sha256 in the summary: 64 bits, ample to detect a
#: change between two captures (this is a diff key, not a security digest).
SUMMARY_DIGEST_HEX = 16
_STAT_KEYS = ("n_finite", "min", "max", "mean", "rms")


def _short(digest: str) -> str:
    return digest[:SUMMARY_DIGEST_HEX]


def _collapse(path: str) -> str:
    """``a.3.b.12.c`` -> ``a.:.b.:.c``: the IMAS array a scalar leaf belongs to."""
    return ".".join(":" if part.isdigit() else part for part in path.split("."))


def _stats(record: Mapping[str, Any]) -> list[Any]:
    return [record.get(key) for key in _STAT_KEYS]


def summarize_stage(leaves: Mapping[str, Mapping[str, Any]], *, reference: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Per-array records for one stage product.

    * every leaf with more than one element: ``path -> [sha, shape, n_finite,
      min, max, mean, rms]``;
    * scalar and string leaves, grouped by their collapsed path (the IMAS array
      they are elements of): ``collapsed -> [group sha, count, n_finite, min,
      max, mean, rms]``, the group sha covering every member path and digest in
      order, so any changed, added or removed element changes it;
    * leaves identical (same path, same digest) to ``reference`` -- the static
      product a later stage copied them from -- are counted, not repeated.
    """
    arrays: dict[str, list[Any]] = {}
    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    inherited = hashlib.sha256()
    inherited_count = 0
    for path in sorted(leaves):
        record = leaves[path]
        if reference is not None and reference.get(path, {}).get("sha256") == record["sha256"]:
            # Which leaves are inherited is itself evidence: digest the paths,
            # so inheriting Q instead of P changes the summary.
            inherited.update(f"{path};".encode("utf-8"))
            inherited_count += 1
            continue
        shape = record.get("shape")
        if shape and int(np.prod(shape)) > 1:
            arrays[path] = [_short(record["sha256"]), list(shape), *_stats(record)]
        else:
            groups.setdefault(_collapse(path), []).append((path, record))
    grouped: dict[str, list[Any]] = {}
    for name, members in sorted(groups.items()):
        digest = hashlib.sha256()
        values: list[float] = []
        for path, record in members:
            digest.update(f"{path}={record['sha256']};".encode("ascii"))
            if record.get("kind") == "numeric" and record.get("n_finite"):
                values.append(float(record["mean"]))  # a scalar's mean is its value
        finite = np.asarray(values, dtype=float)
        stats: list[Any] = [int(finite.size), None, None, None, None]
        if finite.size:
            stats = [int(finite.size), float(finite.min()), float(finite.max()),
                     float(finite.mean()), float(np.sqrt(np.mean(finite * finite)))]
        grouped[name] = [_short(digest.hexdigest()), len(members), *stats]
    return {"leaf_count": len(leaves), "inherited_from_static": inherited_count,
            "inherited_digest": _short(inherited.hexdigest()),
            "arrays": arrays, "scalar_groups": grouped}


#: The code whose behaviour a baseline records.  ``code_sha`` may name a
#: different commit than the one captured (a branch carrying only tests and
#: this script) only if these trees are identical between the two.
CODE_TREES = ("vaft", "workflow/automatic_pipeline_1_routine_data_processing")


def _same_code_trees(left: str, right: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", left, right, "--", *CODE_TREES],
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git diff {left} {right} failed; cannot verify the code identity")
    return result.returncode == 0


def _stage_manifest_status(shot_dir: Path, stage: str) -> dict[str, Any]:
    """The stage's own verdict (e.g. diagnostics ``partial``) and which channels
    it did not produce, so a later mismatch can be explained without the full
    capture.  Empty when the stage wrote no manifest."""
    candidates = sorted((shot_dir / stage).glob("*manifest.json"))
    if not candidates:
        return {}
    manifest = json.loads(candidates[0].read_text(encoding="utf-8"))
    out: dict[str, Any] = {"stage_status": manifest.get("status")}
    channels = manifest.get("channel_status")
    if isinstance(channels, dict):
        not_ok: dict[str, Any] = {}
        for name, state in sorted(channels.items()):
            status = state.get("status") if isinstance(state, dict) else state
            if status in ("ok", "success", "available"):
                continue
            not_ok[str(name)] = {"status": status}
            if isinstance(state, dict) and "missing_channels" in state:
                not_ok[str(name)]["missing_channels"] = state["missing_channels"]
        out["channels_not_ok"] = not_ok
    return out


def summarize_root(root: Path, *, code_sha: str) -> dict[str, Any]:
    """Compact summary of every ``<root>/<shot>/baseline.json`` under ``root``.

    The full per-leaf baselines stay in the isolated root; the summary records
    their sha256 so a later phase can localize a mismatch from them.  Refuses a
    root with no capture, captures from more than one revision or a dirty tree,
    or a ``code_sha`` whose code trees differ from the captured commit's.
    """
    root = Path(root)
    shots: dict[str, Any] = {}
    static_by_era: dict[str, Any] = {}
    revisions: set[tuple[str, bool]] = set()
    baselines = sorted(root.glob("*/baseline.json"))
    if not baselines:
        raise RuntimeError(f"no <shot>/baseline.json under {root}")
    for baseline_path in baselines:
        record = json.loads(baseline_path.read_text(encoding="utf-8"))
        stages = record["stages"]
        static_leaves = stages.get("static", {}).get("leaves")
        era = record["machine_version"]
        if static_leaves is not None and era not in static_by_era:
            static_by_era[era] = {
                "product_sha256": stages["static"]["product_sha256"],
                **summarize_stage(static_leaves),
            }
        elif static_leaves is not None:
            first = static_by_era[era]
            again = summarize_stage(static_leaves)
            if (again["arrays"], again["scalar_groups"]) != (first["arrays"], first["scalar_groups"]):
                raise RuntimeError(f"two different static products for era {era}: the capture is not deterministic")
        entry: dict[str, Any] = {
            "machine_version": era,
            "inputs": record["inputs"],
            "versions": record.get("versions"),
            # Identifies *this* capture's per-leaf file (which also holds run
            # timings); a re-capture never matches it.  Compare leaf digests.
            "full_baseline_sha256": sha256_file(baseline_path),
            "stages": {},
        }
        for stage, data in stages.items():
            stage_manifest = _stage_manifest_status(baseline_path.parent, stage)
            if stage == "static":
                entry["stages"][stage] = {"status": data["status"], "product_sha256": data.get("product_sha256"),
                                          "manifest_sha256": data.get("manifest_sha256"), **stage_manifest}
                continue
            entry["stages"][stage] = {
                "status": data["status"],
                "product_sha256": data.get("product_sha256"),
                "manifest_sha256": data.get("manifest_sha256"),
                **stage_manifest,
                **(summarize_stage(data["leaves"], reference=static_leaves) if "leaves" in data else {}),
            }
        shots[str(record["shot"])] = entry
        revisions.add((record["revision"]["commit"], bool(record["revision"]["dirty"])))
    if len(revisions) != 1:
        raise RuntimeError(f"captures from more than one revision: {sorted(revisions)}")
    ((commit, dirty),) = revisions
    if dirty:
        raise RuntimeError(f"captured from a dirty tree at {commit}; re-capture from a clean checkout")
    if not _same_code_trees(code_sha, commit):
        raise RuntimeError(f"{CODE_TREES} differ between --code-sha {code_sha} and the captured {commit}")
    return {
        "schema": SUMMARY_SCHEMA,
        "code_sha": code_sha,
        "captured_revision": {"commit": commit, "dirty": False},
        "code_trees_identical_to_code_sha": list(CODE_TREES),
        "digest_hex_digits": SUMMARY_DIGEST_HEX,
        "how_to_compare": (
            "Re-capture at the new revision, summarize, and compare arrays / scalar_groups / "
            "inherited_digest per stage. full_baseline_sha256 identifies this capture's own "
            "per-leaf file (which records run timings) and never matches a re-capture."
        ),
        "array_record": ["sha256_prefix", "shape", *_STAT_KEYS],
        "scalar_group_record": ["group_sha256_prefix", "count", *_STAT_KEYS],
        "static_by_era": static_by_era,
        "shots": shots,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "summarize":
        summary_parser = argparse.ArgumentParser(prog="capture_baseline.py summarize")
        summary_parser.add_argument("root", type=Path, help="A completed capture root.")
        summary_parser.add_argument("--code-sha", required=True, help="The code revision the root was captured at.")
        summary_parser.add_argument("--output", required=True, type=Path, help="Summary JSON to write.")
        summary_args = summary_parser.parse_args(argv[1:])
        summary = summarize_root(summary_args.root, code_sha=summary_args.code_sha)
        summary_args.output.write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {summary_args.output} ({summary_args.output.stat().st_size} bytes)")
        return 0
    parser.add_argument("--output-root", required=True, type=Path, help="Fresh, isolated scratch directory.")
    parser.add_argument(
        "--shot", type=int, action="append", choices=SHOTS_WITH_RAW,
        help="Packaged shot to capture (repeatable); default all four.",
    )
    parser.add_argument(
        "--stage", action="append", choices=STAGES,
        help="Restrict to these stages (repeatable); later stages need earlier ones.",
    )
    args = parser.parse_args(argv)
    root = check_output_root(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    failed = False
    for shot in args.shot or SHOTS_WITH_RAW:
        record = capture_shot(shot, root, stages=tuple(args.stage or STAGES))
        for stage, entry in record["stages"].items():
            leaves = len(entry.get("leaves", {}))
            print(f"{shot} {stage}: {entry['status']} ({leaves} leaves, {entry['seconds']} s)")
            failed |= entry["status"] != "captured"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
