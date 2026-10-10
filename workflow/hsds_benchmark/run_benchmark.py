#!/usr/bin/env python3
"""Reproducible, read-only HSDS I/O benchmark for VAFT's access paths (#1869).

    python workflow/hsds_benchmark/run_benchmark.py --shots 39915,41524 \\
        --clients 1,2,4 --repeat 3 --output runs/hsds-bench

Each (scenario, shot, clients, cache state, repeat) runs in fresh child
processes -- one per client, started together -- so in-process caches start
cold and peak RSS is per client. Every client is wrapped in
:func:`vaft.database.instrumentation.record_io` (request count, observed bytes,
header latency, hsget/hsload time) and reads the same probe values, whose
digests are compared across scenarios: a fast path that returns different data
is reported as incorrect, not as fast.

Scenarios (all read-only):

* ``lazy_scalar``   -- ``open()``; per-slice scalars (global quantities) and
  per-probe/coil leaves, one selection each: the scalar-heavy traversal.
* ``lazy_large``    -- ``open()``; whole 2-D ``psi`` of every slice and every
  probe field: large-array selections.
* ``prefetch``      -- ``open()`` + ``prefetch`` of the IDS, then the scalar traversal.
* ``eager_canonical`` / ``eager_h5image`` -- ``load(..., transport=...)``,
  cold (empty cache directory) and warm (same directory, second run).
* ``eager_full``    -- ``load()`` with the default transport.

Writes ``results.jsonl`` (one record per client run) and ``summary.json``
(context + p50/p95 per group). Run against production only with low client
counts; the issue asks for no destructive load tests there.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCENARIOS = ("lazy_scalar", "lazy_large", "prefetch", "eager_canonical", "eager_h5image", "eager_full")
EAGER = {"eager_canonical": "canonical", "eager_h5image": "h5image", "eager_full": "auto"}
#: IDS every scenario reads, and the probe values whose digests must agree.
IDS = ("equilibrium", "magnetics", "pf_active")
SCALARS = ("ip", "li_3", "beta_pol", "magnetic_axis.r", "magnetic_axis.z")


def digest(value: Any) -> str:
    array = np.asarray(value)
    payload = array.tobytes() if array.dtype != object else repr(value).encode()
    return hashlib.sha256(f"{array.dtype}|{array.shape}|".encode() + payload).hexdigest()[:16]


def _try(ods: Any, path: str) -> Any:
    try:
        return ods[path]
    except Exception:  # an absent path is part of the shot, not a failure
        return None


def _count(ods: Any, path: str) -> int:
    try:
        return len(ods[path])
    except Exception:
        return 0


def read_probes(ods: Any, *, large: bool) -> dict[str, str]:
    """Read the probe set from an ODS (lazy or eager) and return value digests.

    The large set is a superset of the small one, so every value a lazy
    scalar traversal reads is also read -- and compared -- by the eager runs.
    """
    digests: dict[str, str] = {}

    def keep(path: str) -> None:
        value = _try(ods, path)
        if value is not None:
            digests[path] = digest(value)

    keep("equilibrium.time")
    keep("magnetics.ip.0.data")
    for i in range(_count(ods, "equilibrium.time_slice")):
        for name in SCALARS:
            keep(f"equilibrium.time_slice.{i}.global_quantities.{name}")
        if large:
            keep(f"equilibrium.time_slice.{i}.profiles_2d.0.psi")
    for i in range(_count(ods, "pf_active.coil")):
        keep(f"pf_active.coil.{i}.current.data")
    for i in range(_count(ods, "magnetics.b_field_pol_probe")):
        keep(f"magnetics.b_field_pol_probe.{i}.position.r")
        if large:
            keep(f"magnetics.b_field_pol_probe.{i}.field.data")
    return digests


def run_scenario(scenario: str, shot: int, source: str, cache_dir: str | None) -> dict[str, Any]:
    """One client: run ``scenario`` and return wall time, I/O summary, peak RSS and probe digests."""
    import vaft.database as database
    from vaft.database.instrumentation import record_io

    started = time.perf_counter()
    error = None
    digests: dict[str, str] = {}
    run_id = os.environ.get("VAFT_BENCHMARK_RUN_ID") or uuid.uuid4().hex[:12]
    with record_io(run_id=run_id) as recorder:
        try:
            if scenario in EAGER:
                ods = database.load(shot, source, cache=cache_dir or "off", transport=EAGER[scenario])
                digests = read_probes(ods, large=True)
            else:
                ods = database.open(shot, source=source)
                if scenario == "prefetch":
                    for ids in IDS:
                        try:
                            ods.prefetch(ids)
                        except Exception:  # an IDS the shot does not have
                            pass
                digests = read_probes(ods, large=scenario == "lazy_large")
        except Exception as exc:  # a failed client is a result, recorded with the rest
            error = f"{type(exc).__name__}: {exc}"[:300]
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024
    return {"wall_s": time.perf_counter() - started, "io": recorder.summary(), "peak_rss_mb": peak_mb,
            "digests": digests, "error": error}


def _client(args: tuple[str, int, str, str | None, str]) -> dict[str, Any]:
    scenario, shot, source, cache_dir, run_id = args
    os.environ["VAFT_BENCHMARK_RUN_ID"] = run_id
    return run_scenario(scenario, shot, source, cache_dir)


def run_group(scenario: str, shot: int, source: str, clients: int, cache_state: str, cache_root: Path,
              run_id: str, runner: Callable[[tuple], dict] = _client) -> list[dict[str, Any]]:
    """``clients`` concurrent child processes; each client gets its own cache directory."""
    dirs = []
    for index in range(clients):
        d = cache_root / f"{scenario}-{shot}-c{clients}-{index}"
        if cache_state == "cold" and d.exists():
            shutil.rmtree(d)
        dirs.append(str(d) if scenario in EAGER else None)
    jobs = [(scenario, shot, source, dirs[i], run_id) for i in range(clients)]
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=clients, mp_context=context) as pool:
        results = list(pool.map(runner, jobs))
    return results


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def context_record(source: str, shots: Sequence[int], run_id: str) -> dict[str, Any]:
    """Who, what and against which server: enough to reproduce and to correlate server logs."""
    import h5pyd
    import omas

    import vaft

    root = Path(vaft.__file__).resolve().parents[1]
    server: dict[str, Any] = {}
    try:
        info = h5pyd.getServerInfo()
        server = {k: info.get(k) for k in ("hsds_version", "node_count", "state", "endpoint", "name", "about")}
    except Exception as exc:
        server = {"error": repr(exc)[:200]}
    identity: dict[str, Any] = {}
    for shot in shots:
        for ids in IDS:
            try:
                with h5pyd.File(f"/{source}/{shot}/{ids}.h5", "r") as handle:
                    identity[f"{shot}/{ids}"] = {"modified": handle.modified}
            except Exception:
                identity[f"{shot}/{ids}"] = None
    return {
        "run_id": run_id, "source": source, "shots": list(shots), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vaft_version": getattr(vaft, "__version__", None), "vaft_commit": _git(root, "rev-parse", "HEAD"),
        "vaft_dirty": bool(_git(root, "status", "--porcelain")), "h5pyd_version": h5pyd.__version__,
        "omas_version": omas.__version__, "python": platform.python_version(), "host": platform.node(),
        "server": server, "domains": identity,
    }


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """p50/p95 wall time and I/O per (scenario, shot, clients, cache_state), plus correctness across scenarios."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["scenario"], r["shot"], r["clients"], r["cache_state"]), []).append(r)
    reference: dict[int, dict[str, str]] = {}
    for r in records:  # eager canonical is the reference for every probe it read
        if r["scenario"] == "eager_canonical" and not r["error"]:
            reference.setdefault(r["shot"], {}).update(r["digests"])
    rows = []
    for (scenario, shot, clients, cache_state), rs in sorted(groups.items()):
        ok = [r for r in rs if not r["error"]]
        wall = np.asarray([r["wall_s"] for r in ok], dtype=float)
        mismatched = sorted({p for r in ok for p, d in r["digests"].items()
                             if p in reference.get(shot, {}) and reference[shot][p] != d})
        compared = sum(1 for r in ok for p in r["digests"] if p in reference.get(shot, {}))
        rows.append({
            "scenario": scenario, "shot": shot, "clients": clients, "cache_state": cache_state, "runs": len(rs),
            "errors": len(rs) - len(ok),
            "wall_p50_s": float(np.percentile(wall, 50)) if wall.size else None,
            "wall_p95_s": float(np.percentile(wall, 95)) if wall.size else None,
            "requests_median": float(np.median([r["io"]["request_count"] for r in ok])) if ok else None,
            "response_bytes_observed_median": float(np.median([r["io"]["response_bytes_observed"] for r in ok])) if ok else None,
            "subprocess_seconds_median": float(np.median([r["io"]["subprocess_seconds"] for r in ok])) if ok else None,
            "peak_rss_mb_max": max((r["peak_rss_mb"] for r in ok), default=None),
            "probe_values": len(ok[0]["digests"]) if ok else 0,
            "compared_to_eager_canonical": compared, "mismatched_paths": mismatched,
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", default="39915", help="comma-separated shots")
    parser.add_argument("--source", default="main")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--clients", default="1", help="comma-separated client counts, e.g. 1,2,4,8")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    shots = [int(s) for s in args.shots.split(",") if s.strip()]
    scenarios = [s for s in args.scenarios.split(",") if s.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenarios {unknown}; choose from {SCENARIOS}")
    clients = [int(c) for c in args.clients.split(",") if c.strip()]
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    args.output.mkdir(parents=True, exist_ok=True)
    context = context_record(args.source, shots, run_id)
    records: list[dict[str, Any]] = []
    results_path = args.output / "results.jsonl"
    with tempfile.TemporaryDirectory(prefix="vaft-hsds-bench-") as tmp, results_path.open("w", encoding="utf-8") as out:
        cache_root = Path(tmp)
        for shot in shots:
            for scenario in scenarios:
                states = ("cold", "warm") if scenario in EAGER else ("cold",)
                for n in clients:
                    for repeat in range(args.repeat):
                        for state in states:
                            for index, result in enumerate(run_group(scenario, shot, args.source, n, state, cache_root, run_id)):
                                record = {"run_id": run_id, "scenario": scenario, "shot": shot, "clients": n,
                                          "client_index": index, "repeat": repeat, "cache_state": state, **result}
                                records.append(record)
                                out.write(json.dumps(record) + "\n")
                                out.flush()
                            print(f"{scenario} shot={shot} clients={n} repeat={repeat} {state}: "
                                  f"{[round(r['wall_s'], 2) for r in records[-n:]]} s", flush=True)
    context["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = {"context": context, "groups": summarise(records),
               "notes": ["requests and header latency are observed in-process; hsget/hsload traffic is timed per subprocess",
                         "response bytes are Content-Length values where the server sent one",
                         "correctness: probe digests compared with eager canonical for the same shot"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{len(records)} client runs -> {results_path}; summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
