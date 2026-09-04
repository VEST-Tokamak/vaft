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

from vaft.database import open as open_database
from vaft.database import staging
from vaft.database.h5image import derived_filename, is_derived_filename
from vaft.imas.omas_imas import load_omas_imas


@dataclass(frozen=True)
class Branch:
    ids: str
    path: str
    native_path: tuple[object, ...]


BRANCHES = (
    Branch(
        "equilibrium",
        "equilibrium.time_slice.0.profiles_2d.0.psi",
        ("time_slice", 0, "profiles_2d", 0, "psi"),
    ),
    Branch("magnetics", "magnetics.time", ("time",)),
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
    return {
        "dtype": str(array.dtype),
        "logical_bytes": int(array.nbytes),
        "shape": list(array.shape),
    }


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
        value = (
            value[component]
            if isinstance(component, int)
            else getattr(value, component)
        )
    return np.asarray(value)


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
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


def _hsstat_size(domain: str) -> dict[str, int | None]:
    try:
        result = subprocess.run(
            ["hsstat", domain], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {"total_size": None, "allocated_bytes": None}
    values: dict[str, int | None] = {"total_size": None, "allocated_bytes": None}
    for line in result.stdout.splitlines():
        key, _, value = line.strip().partition(":")
        if key in values:
            values[key] = int(value.strip())
    return values


def _storage_summary(directory: str, shot: int) -> dict[str, Any]:
    import h5pyd

    names = sorted(
        name
        for name in h5pyd.Folder(f"/{directory}/{shot}/")
        if name.endswith(".h5") and not is_derived_filename(name)
    )
    canonical = {name: _hsstat_size(f"/{directory}/{shot}/{name}") for name in names}
    derived = {
        derived_filename(name): _hsstat_size(
            f"/{directory}/{shot}/{derived_filename(name)}"
        )
        for name in names
    }
    return {
        "canonical": canonical,
        "h5image": derived,
        "canonical_allocated_bytes": sum(
            value["allocated_bytes"] or 0 for value in canonical.values()
        ),
        "h5image_allocated_bytes": sum(
            value["allocated_bytes"] or 0 for value in derived.values()
        ),
    }


def _load_legacy_h5image(
    directory: str, shot: int, output: Path
) -> tuple[omas.ODS, dict[str, float | int]]:
    """Materialize the historical single-image domain and restore its ODS."""
    import h5pyd

    image_path = output / f"{shot}.legacy.omas.h5"
    transfer_seconds = 0.0
    write_seconds = 0.0
    downloaded_bytes = 0
    with h5pyd.File(f"hdf5://{directory}/{shot}.h5", "r") as domain:
        image = domain["h5image"]
        with image_path.open("wb") as output_file:
            for offset in range(0, image.shape[0], 4 * 1024 * 1024):
                started = time.perf_counter()
                block = np.asarray(
                    image[offset : min(offset + 4 * 1024 * 1024, image.shape[0])],
                    dtype=np.uint8,
                )
                transfer_seconds += time.perf_counter() - started
                started = time.perf_counter()
                output_file.write(block.tobytes())
                write_seconds += time.perf_counter() - started
                downloaded_bytes += int(block.nbytes)
    legacy = omas.ODS(consistency_check=False)
    _, conversion_seconds = _seconds(
        lambda: legacy.load(str(image_path), consistency_check=False)
    )
    return legacy, {
        "downloaded_bytes": downloaded_bytes,
        "download_seconds": transfer_seconds,
        "local_image_write_seconds": write_seconds,
        "ods_restore_seconds": conversion_seconds,
        "total_seconds": transfer_seconds + write_seconds + conversion_seconds,
    }


def _eager_branch(
    directory: str,
    shot: int,
    branch: Branch,
    root: Path,
    *,
    requested_ids: tuple[str, ...] | None,
    transport: str = "canonical",
) -> tuple[Any, Any, dict[str, Any]]:
    stage_dir = root / transport / ("full" if requested_ids is None else branch.ids)
    stage_dir.mkdir(parents=True, exist_ok=True)
    plan, staging_seconds = _seconds(
        lambda: staging.stage_imas_shot(
            directory,
            shot,
            stage_dir,
            requested_ids=requested_ids,
            cache="off",
            transport=transport,
        )
    )
    ods, conversion_seconds = _seconds(
        lambda: load_omas_imas(
            paths=[branch.ids],
            consistency_check=False,
            verbose=False,
            uri="imas:hdf5?path=" + str(stage_dir),
        )
    )
    value, lookup_seconds = _seconds(lambda: ods[branch.path])
    timing = plan["timings"]
    return (
        value,
        ods,
        {
            "staging_seconds": staging_seconds,
            "master_fetch_seconds": timing["master_fetch_seconds"],
            "domain_resolution_seconds": timing["domain_resolution_seconds"],
            "per_domain_hsget_seconds": float(
                sum(timing["domain_fetch_seconds"].values())
            ),
            "partial_master_seconds": timing["partial_master_seconds"],
            "staged_files": plan["files"],
            "cache_hits": plan["cache_hits"],
            "transports": plan["transports"],
            "conversion_seconds": conversion_seconds,
            "lookup_seconds": lookup_seconds,
            "total_seconds": staging_seconds + conversion_seconds + lookup_seconds,
        },
    )


def _convert_staged_branch(
    stage_dir: Path, branch: Branch
) -> tuple[Any, dict[str, float]]:
    """Measure only IMAS-to-ODS conversion and lookup for an already staged shot."""
    ods, conversion_seconds = _seconds(
        lambda: load_omas_imas(
            paths=[branch.ids],
            consistency_check=False,
            verbose=False,
            uri="imas:hdf5?path=" + str(stage_dir),
        )
    )
    value, lookup_seconds = _seconds(lambda: ods[branch.path])
    return value, {
        "conversion_seconds": conversion_seconds,
        "lookup_seconds": lookup_seconds,
    }


def _lazy_branch(
    directory: str, shot: int, branch: Branch
) -> tuple[Any, dict[str, Any]]:
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


def _native_from_staged(stage_dir: Path, names: list[str]) -> dict[str, Any]:
    """Convert already-downloaded IMAS images to eager native IDS objects."""
    import imas

    factory = imas.IDSFactory("3.41.0")
    with imas.DBEntry(
        "imas:hdf5?path=" + str(stage_dir), "r", dd_version="3.41.0"
    ) as entry:
        return {name: entry.get(name, 0) for name in names if factory.exists(name)}


def _native_eager_branch(
    directory: str,
    shot: int,
    branch: Branch,
    root: Path,
    *,
    transport: str = "canonical",
) -> tuple[Any, dict[str, Any]]:
    stage_dir = root / transport / branch.ids
    stage_dir.mkdir(parents=True, exist_ok=True)
    plan, download_seconds = _seconds(
        lambda: staging.stage_imas_shot(
            directory,
            shot,
            stage_dir,
            requested_ids=(branch.ids,),
            cache="off",
            transport=transport,
        )
    )
    native, conversion_seconds = _seconds(
        lambda: _native_from_staged(stage_dir, [branch.ids])
    )
    ids = native[branch.ids]
    value, lookup_seconds = _seconds(lambda: _native_value(ids, branch))
    timing = plan["timings"]
    return value, {
        "download_seconds": download_seconds,
        "master_fetch_seconds": timing["master_fetch_seconds"],
        "per_domain_hsget_seconds": float(sum(timing["domain_fetch_seconds"].values())),
        "object_conversion_seconds": conversion_seconds,
        "lookup_seconds": lookup_seconds,
        "transports": plan["transports"],
        "total_seconds": download_seconds + conversion_seconds + lookup_seconds,
    }


def _native_lazy_branch(
    directory: str, shot: int, branch: Branch
) -> tuple[Any, dict[str, Any]]:
    handle, open_seconds = _seconds(
        lambda: open_database(
            shot, source=directory, representation="imas", paths=branch.ids
        )
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


def _full_eager_pass(
    directory: str, shot: int, *, transport: str
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
]:
    """Stage one complete shot, measure both conversions, then release all files."""
    with tempfile.TemporaryDirectory(
        prefix=f"vaft-hsds-{transport}-full-"
    ) as temporary:
        stage_dir = Path(temporary)
        plan, staging_seconds = _seconds(
            lambda: staging.stage_imas_shot(
                directory,
                shot,
                stage_dir,
                requested_ids=None,
                cache="off",
                transport=transport,
            )
        )
        available_ids = [
            name[:-3]
            for name in plan["files"]
            if name.endswith(".h5") and name != "master.h5"
        ]
        ods, ods_conversion_seconds = _seconds(
            lambda: load_omas_imas(
                consistency_check=False,
                verbose=False,
                uri="imas:hdf5?path=" + str(stage_dir),
                available_ids=available_ids,
            )
        )
        native_names = [name for name in available_ids if name != "dataset_description"]
        native, native_conversion_seconds = _seconds(
            lambda: _native_from_staged(stage_dir, native_names)
        )
        timing = plan["timings"]
        domain_fetch_seconds = float(sum(timing["domain_fetch_seconds"].values()))
        ods_values: dict[str, np.ndarray] = {}
        native_values: dict[str, np.ndarray] = {}
        ods_stats: dict[str, dict[str, Any]] = {}
        native_stats: dict[str, dict[str, Any]] = {}
        for branch in BRANCHES:
            ods_value, ods_lookup = _seconds(lambda branch=branch: ods[branch.path])
            native_value, native_lookup = _seconds(
                lambda branch=branch: _native_value(native[branch.ids], branch)
            )
            # Detach representative values before the temporary backing store closes.
            ods_values[branch.ids] = np.asarray(ods_value).copy()
            native_values[branch.ids] = np.asarray(native_value).copy()
            common = {
                "master_fetch_seconds": timing["master_fetch_seconds"],
                "domain_resolution_seconds": timing["domain_resolution_seconds"],
                "domain_fetch_seconds": domain_fetch_seconds,
                "staged_files": plan["files"],
                "cache_hits": plan["cache_hits"],
                "transports": plan["transports"],
            }
            ods_stats[branch.ids] = {
                **common,
                "staging_seconds": staging_seconds,
                "object_conversion_seconds": ods_conversion_seconds,
                "lookup_seconds": ods_lookup,
                "total_seconds": staging_seconds + ods_conversion_seconds + ods_lookup,
            }
            native_stats[branch.ids] = {
                **common,
                "download_seconds": staging_seconds,
                "object_conversion_seconds": native_conversion_seconds,
                "lookup_seconds": native_lookup,
                "total_seconds": staging_seconds
                + native_conversion_seconds
                + native_lookup,
            }
        return ods_values, ods_stats, native_values, native_stats


def _selective_eager_pass(
    directory: str, shot: int, branch: Branch, *, transport: str, representation: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Measure one selective load and delete its staging before the next case."""
    with tempfile.TemporaryDirectory(
        prefix=f"vaft-hsds-{transport}-{branch.ids}-"
    ) as temporary:
        root = Path(temporary)
        if representation == "omas":
            value, _ods, stats = _eager_branch(
                directory,
                shot,
                branch,
                root,
                requested_ids=(branch.ids,),
                transport=transport,
            )
        else:
            value, stats = _native_eager_branch(
                directory, shot, branch, root, transport=transport
            )
        return np.asarray(value).copy(), stats


def _legacy_pass(
    directory: str, shot: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="vaft-hsds-legacy-") as temporary:
        legacy, common = _load_legacy_h5image(directory, shot, Path(temporary))
        values: dict[str, np.ndarray] = {}
        records: dict[str, dict[str, Any]] = {}
        for branch in BRANCHES:
            value, lookup = _seconds(lambda branch=branch: legacy[branch.path])
            values[branch.ids] = np.asarray(value).copy()
            records[branch.ids] = {
                **common,
                "lookup_seconds": lookup,
                "total_seconds": common["total_seconds"] + lookup,
            }
        return values, records


def benchmark_load_paths(
    directory: str,
    shot: int,
    *,
    repeat: int = 5,
    warmup: int = 1,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Execute repeatable, read-only timing passes and return JSON-ready data."""
    if repeat < 1 or warmup < 0:
        raise ValueError("repeat must be >= 1 and warmup must be >= 0")
    measurements: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {branch.ids: [] for branch in BRANCHES}
        for method in (
            "eager_full",
            "eager_full_h5image",
            "eager_selective",
            "eager_selective_h5image",
            "native_full",
            "native_full_h5image",
            "native_eager",
            "native_eager_h5image",
            "lazy_hsds",
            "native_lazy",
            "legacy_h5image",
        )
    }
    branch_state: dict[str, dict[str, Any]] = {
        branch.ids: {"path": branch.path} for branch in BRANCHES
    }
    completed_iterations = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1,
            "directory": directory,
            "shot": int(shot),
            "repeat": repeat,
            "warmup": warmup,
        }
        if all(checkpoint.get(key) == value for key, value in expected.items()):
            completed_iterations = int(checkpoint.get("completed_iterations", 0))
            measurements = checkpoint["measurements"]
            branch_state = checkpoint["branch_state"]

    for iteration in range(completed_iterations, warmup + repeat):
        legacy_values, legacy_records = _legacy_pass(directory, shot)
        full_values, full_stats, native_full_values, native_full_stats = (
            _full_eager_pass(directory, shot, transport="canonical")
        )
        h5_full_values, h5_full_stats, h5_native_full_values, h5_native_full_stats = (
            _full_eager_pass(directory, shot, transport="h5image")
        )
        for branch in BRANCHES:
            full_value = full_values[branch.ids]
            selective_value, selective_stats = _selective_eager_pass(
                directory, shot, branch, transport="canonical", representation="omas"
            )
            h5_selective_value, h5_selective_stats = _selective_eager_pass(
                directory, shot, branch, transport="h5image", representation="omas"
            )
            lazy_value, lazy_stats = _lazy_branch(directory, shot, branch)
            native_eager_value, native_eager_stats = _selective_eager_pass(
                directory, shot, branch, transport="canonical", representation="imas"
            )
            h5_native_eager_value, h5_native_eager_stats = _selective_eager_pass(
                directory, shot, branch, transport="h5image", representation="imas"
            )
            native_lazy_value, native_lazy_stats = _native_lazy_branch(
                directory, shot, branch
            )
            state = branch_state[branch.ids]
            state.update(
                {
                    "eager_value": _value_summary(full_value),
                    "selective_matches_full_eager": _same_value(
                        selective_value, full_value
                    ),
                    "h5image_full_matches_full_eager": _same_value(
                        h5_full_values[branch.ids], full_value
                    ),
                    "h5image_selective_matches_full_eager": _same_value(
                        h5_selective_value, full_value
                    ),
                    "lazy_matches_full_eager": _same_value(lazy_value, full_value),
                    "native_full_matches_full_eager": _same_value(
                        native_full_values[branch.ids], full_value
                    ),
                    "native_eager_matches_full_eager": _same_value(
                        native_eager_value, full_value
                    ),
                    "h5image_native_full_matches_full_eager": _same_value(
                        h5_native_full_values[branch.ids], full_value
                    ),
                    "h5image_native_eager_matches_full_eager": _same_value(
                        h5_native_eager_value, full_value
                    ),
                    "native_lazy_matches_full_eager": _same_value(
                        native_lazy_value, full_value
                    ),
                    "legacy_matches_full_eager": _same_value(
                        legacy_values[branch.ids], full_value
                    ),
                }
            )
            if not state["legacy_matches_full_eager"]:
                state["legacy_stale"] = True
            if iteration >= warmup:
                measurements["eager_full"][branch.ids].append(full_stats[branch.ids])
                measurements["eager_full_h5image"][branch.ids].append(
                    h5_full_stats[branch.ids]
                )
                measurements["eager_selective"][branch.ids].append(selective_stats)
                measurements["eager_selective_h5image"][branch.ids].append(
                    h5_selective_stats
                )
                measurements["native_full"][branch.ids].append(
                    native_full_stats[branch.ids]
                )
                measurements["native_full_h5image"][branch.ids].append(
                    h5_native_full_stats[branch.ids]
                )
                measurements["native_eager"][branch.ids].append(native_eager_stats)
                measurements["native_eager_h5image"][branch.ids].append(
                    h5_native_eager_stats
                )
                measurements["lazy_hsds"][branch.ids].append(lazy_stats)
                measurements["native_lazy"][branch.ids].append(native_lazy_stats)
                measurements["legacy_h5image"][branch.ids].append(
                    legacy_records[branch.ids]
                )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "schema_version": 1,
                "directory": directory,
                "shot": int(shot),
                "repeat": repeat,
                "warmup": warmup,
                "completed_iterations": iteration + 1,
                "measurements": measurements,
                "branch_state": branch_state,
            }
            temporary_checkpoint = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".tmp"
            )
            temporary_checkpoint.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
            , encoding="utf-8")
            temporary_checkpoint.replace(checkpoint_path)

    methods: dict[str, Any] = {}
    for method, per_branch in measurements.items():
        methods[method] = {"per_branch": {}}
        for ids_name, records in per_branch.items():
            numeric_keys = sorted(
                {
                    key
                    for record in records
                    for key, value in record.items()
                    if isinstance(value, (int, float))
                }
            )
            methods[method]["per_branch"][ids_name] = {
                "metrics": {
                    key: _summary([float(record[key]) for record in records])
                    for key in numeric_keys
                },
                "samples": records,
            }

    recommendations: list[str] = []
    for branch in BRANCHES:
        name = branch.ids
        omas_lazy = methods["lazy_hsds"]["per_branch"][name]["metrics"][
            "total_seconds"
        ]["median_seconds"]
        selective = methods["eager_selective"]["per_branch"][name]["metrics"][
            "total_seconds"
        ]["median_seconds"]
        native_lazy = methods["native_lazy"]["per_branch"][name]["metrics"][
            "total_seconds"
        ]["median_seconds"]
        native_eager = methods["native_eager"]["per_branch"][name]["metrics"][
            "total_seconds"
        ]["median_seconds"]
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

    full_canonical = methods["eager_full"]["per_branch"]["equilibrium"]["metrics"][
        "total_seconds"
    ]["median_seconds"]
    full_h5image = methods["eager_full_h5image"]["per_branch"]["equilibrium"][
        "metrics"
    ]["total_seconds"]["median_seconds"]
    if full_h5image <= full_canonical * 0.8:
        recommendations.append(
            "Per-IDS h5image meets the full eager latency threshold and should be preferred by auto transport."
        )
    else:
        recommendations.append(
            "Per-IDS h5image misses the full eager latency threshold; keep it opt-in and prefer canonical transport."
        )

    return {
        "schema_version": 3,
        "directory": directory,
        "shot": int(shot),
        "repeat": repeat,
        "warmup_excluded": warmup,
        "runtime": _runtime_context(),
        "storage": _storage_summary(directory, shot),
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
    report_path = Path(
        os.environ.get(
            "VAFT_BENCHMARK_REPORT",
            f"/tmp/vaft-hsds-load-benchmark-{directory}-{shot}.json",
        )
    )
    checkpoint_path = report_path.with_suffix(report_path.suffix + ".checkpoint")
    report = benchmark_load_paths(
        directory,
        shot,
        repeat=repeat,
        checkpoint_path=checkpoint_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote benchmark report: {report_path}")
