"""Opt-in, read-only timing comparison of VAFT's ODS loading paths.

This test never writes to HSDS.  It contrasts complete eager materialization,
selective eager materialization, direct Lazy ODS selection, and the historical
read-only h5image snapshot.  It is intentionally opt-in because each cold run
downloads real public data::

    VAFT_RUN_HSDS_BENCHMARK=1 VAFT_BENCHMARK_SHOT=39915 \
      python -m pytest -q -s test/test_hsds_load_benchmark.py --repeat=5
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import h5py
import numpy as np
import omas
import pytest

from vaft.database import load as load_database
from vaft.database import open as open_database
from vaft.database import staging
from vaft.imas.omas_imas import load_omas_imas


@dataclass(frozen=True)
class Branch:
    ids: str
    path: str
    native_path: tuple[object, ...]


BRANCHES = (
    Branch("equilibrium", "equilibrium.time_slice.0.profiles_2d.0.psi", ("time_slice", 0, "profiles_2d", 0, "psi")),
    Branch("magnetics", "magnetics.time", ("time",)),
    Branch("pf_active", "pf_active.coil.0.current.data", ("coil", 0, "current", "data")),
    Branch("barometry", "barometry.time", ("time",)),
)


def _seconds(callable_: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def _summary(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=float)
    return {
        "count": int(values.size),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
    }


def _value_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "logical_bytes": int(array.nbytes), "shape": list(array.shape)}


def _same_value(left: Any, right: Any) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in "fci" or right_array.dtype.kind in "fci":
        return bool(np.allclose(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _native_value(ids: Any, branch: Branch) -> Any:
    value = ids
    for component in branch.native_path:
        value = value[component] if isinstance(component, int) else getattr(value, component)
    return np.asarray(value)


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_context() -> dict[str, Any]:
    import h5pyd

    try:
        endpoint = str(h5pyd.getServerInfo().get("endpoint") or "unknown")
    except Exception:
        endpoint = "unavailable"
    return {
        "vaft_commit": _git_sha(Path(__file__).resolve().parents[1]),
        "vest_server_commit": os.environ.get("VEST_SERVER_COMMIT"),
        "python": sys.version.split()[0],
        "h5py": h5py.__version__,
        "h5pyd": h5pyd.__version__,
        "omas": getattr(omas, "__version__", None),
        "endpoint_fingerprint": hashlib.sha256(endpoint.encode()).hexdigest()[:16],
    }


def _load_legacy_h5image(directory: str, shot: int, output: Path) -> tuple[omas.ODS, dict[str, float | int]]:
    """Materialize the historical single-image domain and restore its ODS."""
    import h5pyd

    image_path = output / f"{shot}.legacy.omas.h5"
    with h5pyd.File(f"hdf5://{directory}/{shot}.h5", "r") as domain:
        payload, transfer_seconds = _seconds(lambda: domain["h5image"][:])
    _, write_seconds = _seconds(lambda: payload.tofile(image_path))
    legacy = omas.ODS(consistency_check=False)
    _, conversion_seconds = _seconds(lambda: legacy.load(str(image_path), consistency_check=False))
    return legacy, {
        "downloaded_bytes": int(payload.nbytes),
        "download_seconds": transfer_seconds,
        "local_image_write_seconds": write_seconds,
        "ods_restore_seconds": conversion_seconds,
        "total_seconds": transfer_seconds + write_seconds + conversion_seconds,
    }


def _eager_branch(
    directory: str, shot: int, branch: Branch, root: Path, *, requested_ids: tuple[str, ...] | None
) -> tuple[Any, Any, dict[str, Any]]:
    stage_dir = root / ("full" if requested_ids is None else branch.ids)
    stage_dir.mkdir(parents=True, exist_ok=True)
    plan, staging_seconds = _seconds(
        lambda: staging.stage_imas_shot(
            directory, shot, stage_dir, requested_ids=requested_ids, cache="off"
        )
    )
    ods, conversion_seconds = _seconds(
        lambda: load_omas_imas(
            paths=[branch.ids], consistency_check=False, verbose=False, uri="imas:hdf5?path=" + str(stage_dir)
        )
    )
    value, lookup_seconds = _seconds(lambda: ods[branch.path])
    timing = plan["timings"]
    return value, ods, {
        "staging_seconds": staging_seconds,
        "master_fetch_seconds": timing["master_fetch_seconds"],
        "domain_resolution_seconds": timing["domain_resolution_seconds"],
        "per_domain_hsget_seconds": float(sum(timing["domain_fetch_seconds"].values())),
        "partial_master_seconds": timing["partial_master_seconds"],
        "staged_files": plan["files"],
        "cache_hits": plan["cache_hits"],
        "conversion_seconds": conversion_seconds,
        "lookup_seconds": lookup_seconds,
        "total_seconds": staging_seconds + conversion_seconds + lookup_seconds,
    }


def _convert_staged_branch(stage_dir: Path, branch: Branch) -> tuple[Any, dict[str, float]]:
    """Measure only IMAS-to-ODS conversion and lookup for an already staged shot."""
    ods, conversion_seconds = _seconds(
        lambda: load_omas_imas(
            paths=[branch.ids], consistency_check=False, verbose=False, uri="imas:hdf5?path=" + str(stage_dir)
        )
    )
    value, lookup_seconds = _seconds(lambda: ods[branch.path])
    return value, {"conversion_seconds": conversion_seconds, "lookup_seconds": lookup_seconds}


def _lazy_branch(directory: str, shot: int, branch: Branch) -> tuple[Any, dict[str, Any]]:
    lazy, open_seconds = _seconds(
        lambda: open_database(shot, source=directory, paths=branch.ids)
    )
    try:
        value, selection_seconds = _seconds(lambda: lazy[branch.path])
        _, cached_seconds = _seconds(lambda: lazy[branch.path])
        metrics = lazy.store.metrics
        opened_ids = list(lazy.store.opened_ids)
    finally:
        lazy.close()
    return value, {
        "open_seconds": open_seconds,
        "metadata_and_first_selection_seconds": selection_seconds,
        "cached_second_access_seconds": cached_seconds,
        "total_seconds": open_seconds + selection_seconds,
        "opened_ids": opened_ids,
        "hsget_subprocess": False,
        **metrics,
    }


def _native_eager_branch(directory: str, shot: int, branch: Branch) -> tuple[Any, dict[str, Any]]:
    ids, load_seconds = _seconds(
        lambda: load_database(
            shot, source=directory, representation="imas", paths=branch.ids, cache="off"
        )
    )
    value, lookup_seconds = _seconds(lambda: _native_value(ids, branch))
    return value, {
        "load_seconds": load_seconds,
        "lookup_seconds": lookup_seconds,
        "total_seconds": load_seconds + lookup_seconds,
    }


def _native_lazy_branch(directory: str, shot: int, branch: Branch) -> tuple[Any, dict[str, Any]]:
    handle, open_seconds = _seconds(
        lambda: open_database(shot, source=directory, representation="imas", paths=branch.ids)
    )
    try:
        ids, ids_seconds = _seconds(handle.get)
        value, selection_seconds = _seconds(lambda: _native_value(ids, branch))
        _, cached_seconds = _seconds(lambda: _native_value(ids, branch))
        metrics = handle.metrics
    finally:
        handle.close()
    return value, {
        "open_seconds": open_seconds,
        "lazy_ids_seconds": ids_seconds,
        "first_selection_seconds": selection_seconds,
        "cached_second_access_seconds": cached_seconds,
        "total_seconds": open_seconds + ids_seconds + selection_seconds,
        "hsget_subprocess": False,
        **metrics,
    }


def benchmark_load_paths(directory: str, shot: int, *, repeat: int = 5, warmup: int = 1) -> dict[str, Any]:
    """Execute repeatable, read-only timing passes and return JSON-ready data."""
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup must be >= 0")
    measurements: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {branch.ids: [] for branch in BRANCHES}
        for method in ("eager_full", "eager_selective", "native_full", "native_eager", "lazy_hsds", "native_lazy", "legacy_h5image")
    }
    branch_state: dict[str, dict[str, Any]] = {branch.ids: {"path": branch.path} for branch in BRANCHES}

    for iteration in range(warmup + repeat):
        with tempfile.TemporaryDirectory(prefix="vaft-hsds-load-benchmark-") as temporary:
            root = Path(temporary)
            legacy, legacy_stats = _load_legacy_h5image(directory, shot, root)
            full_stage_dir = root / "eager-full" / "full"
            full_plan, full_staging_seconds = _seconds(
                lambda: staging.stage_imas_shot(
                    directory, shot, full_stage_dir, requested_ids=None, cache="off"
                )
            )
            full_timing = full_plan["timings"]
            native_full, native_full_seconds = _seconds(
                lambda: load_database(shot, source=directory, representation="imas", paths=None, cache="off")
            )
            for branch in BRANCHES:
                full_value, full_conversion = _convert_staged_branch(full_stage_dir, branch)
                full_stats = {
                    "staging_seconds": full_staging_seconds,
                    "master_fetch_seconds": full_timing["master_fetch_seconds"],
                    "domain_resolution_seconds": full_timing["domain_resolution_seconds"],
                    "per_domain_hsget_seconds": float(sum(full_timing["domain_fetch_seconds"].values())),
                    "partial_master_seconds": 0.0,
                    "staged_files": full_plan["files"],
                    "cache_hits": full_plan["cache_hits"],
                    "total_seconds": full_staging_seconds
                    + full_conversion["conversion_seconds"]
                    + full_conversion["lookup_seconds"],
                    **full_conversion,
                }
                selective_value, _selective_ods, selective_stats = _eager_branch(
                    directory, shot, branch, root / "eager-selective", requested_ids=(branch.ids,)
                )
                legacy_value, legacy_lookup = _seconds(lambda branch=branch: legacy[branch.path])
                lazy_value, lazy_stats = _lazy_branch(directory, shot, branch)
                native_eager_value, native_eager_stats = _native_eager_branch(directory, shot, branch)
                native_lazy_value, native_lazy_stats = _native_lazy_branch(directory, shot, branch)
                native_full_value, native_full_lookup = _seconds(
                    lambda branch=branch: _native_value(native_full[branch.ids], branch)
                )
                native_full_stats = {
                    "load_seconds": native_full_seconds,
                    "lookup_seconds": native_full_lookup,
                    "total_seconds": native_full_seconds + native_full_lookup,
                }
                legacy_record = {**legacy_stats, "lookup_seconds": legacy_lookup, "total_seconds": legacy_stats["total_seconds"] + legacy_lookup}
                state = branch_state[branch.ids]
                state.update(
                    {
                        "eager_value": _value_summary(full_value),
                        "selective_matches_full_eager": _same_value(selective_value, full_value),
                        "lazy_matches_full_eager": _same_value(lazy_value, full_value),
                        "native_full_matches_full_eager": _same_value(native_full_value, full_value),
                        "native_eager_matches_full_eager": _same_value(native_eager_value, full_value),
                        "native_lazy_matches_full_eager": _same_value(native_lazy_value, full_value),
                        "legacy_matches_full_eager": _same_value(legacy_value, full_value),
                    }
                )
                if not state["legacy_matches_full_eager"]:
                    state["legacy_stale"] = True
                if iteration >= warmup:
                    measurements["eager_full"][branch.ids].append(full_stats)
                    measurements["eager_selective"][branch.ids].append(selective_stats)
                    measurements["native_full"][branch.ids].append(native_full_stats)
                    measurements["native_eager"][branch.ids].append(native_eager_stats)
                    measurements["lazy_hsds"][branch.ids].append(lazy_stats)
                    measurements["native_lazy"][branch.ids].append(native_lazy_stats)
                    measurements["legacy_h5image"][branch.ids].append(legacy_record)

    methods: dict[str, Any] = {}
    for method, per_branch in measurements.items():
        methods[method] = {"per_branch": {}}
        for ids_name, records in per_branch.items():
            numeric_keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (int, float))})
            methods[method]["per_branch"][ids_name] = {
                "metrics": {key: _summary([float(record[key]) for record in records]) for key in numeric_keys},
                "samples": records,
            }

    recommendations: list[str] = []
    for branch in BRANCHES:
        name = branch.ids
        omas_lazy = methods["lazy_hsds"]["per_branch"][name]["metrics"]["total_seconds"]["median_seconds"]
        selective = methods["eager_selective"]["per_branch"][name]["metrics"]["total_seconds"]["median_seconds"]
        native_lazy = methods["native_lazy"]["per_branch"][name]["metrics"]["total_seconds"]["median_seconds"]
        native_eager = methods["native_eager"]["per_branch"][name]["metrics"]["total_seconds"]["median_seconds"]
        if omas_lazy > selective * 1.1:
            recommendations.append(
                f"{name}: direct OMAS lazy is slower than selective eager; profile metadata indexing and consider a server-side metadata index."
            )
        if native_lazy > native_eager * 1.1:
            recommendations.append(
                f"{name}: native adapter overhead dominates; optimize the IMAS path codec and context/index cache."
            )
        if native_lazy <= native_eager * 1.1:
            recommendations.append(
                f"{name}: native lazy is competitive; retain direct HSDS selections instead of introducing local staging."
            )

    return {
        "schema_version": 2,
        "directory": directory,
        "shot": int(shot),
        "repeat": repeat,
        "warmup_excluded": warmup,
        "runtime": _runtime_context(),
        "branches": branch_state,
        "methods": methods,
        "recommendations": recommendations,
        "notes": [
            "Selective eager mode uses a temporary partial master and only requested IDS plus dataset_description.",
            "Lazy values report client-side selection metrics; compressed wire bytes and proxy request IDs are not measured.",
            "legacy_stale=true records a historical h5image mismatch without failing this benchmark.",
        ],
    }


@pytest.mark.integration
def test_hsds_load_path_benchmark(request: pytest.FixtureRequest) -> None:
    if os.environ.get("VAFT_RUN_HSDS_BENCHMARK") != "1":
        pytest.skip("set VAFT_RUN_HSDS_BENCHMARK=1 to run the read-only HSDS benchmark")
    directory = os.environ.get("VAFT_BENCHMARK_DIRECTORY", "public")
    shot = int(os.environ.get("VAFT_BENCHMARK_SHOT", "39915"))
    repeat = int(request.config.getoption("--repeat"))
    report_path = Path(os.environ.get("VAFT_BENCHMARK_REPORT", f"/tmp/vaft-hsds-load-benchmark-{directory}-{shot}.json"))
    report = benchmark_load_paths(directory, shot, repeat=repeat)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote benchmark report: {report_path}")
