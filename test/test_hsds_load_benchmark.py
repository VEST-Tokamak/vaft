"""Opt-in, read-only timing comparison for the three HSDS ODS load paths.

Run against a known public shot, never as part of the normal unit-test suite::

    VAFT_RUN_HSDS_BENCHMARK=1 \
    VAFT_BENCHMARK_SHOT=39915 \
    VAFT_BENCHMARK_REPORT=/tmp/vaft-hsds-load-benchmark.json \
    python -m pytest -q -s test/test_hsds_load_benchmark.py

The benchmark deliberately makes no HSDS writes.  It compares the current eager
IMAS staging path, direct Lazy ODS selections, and the pre-existing legacy
``hsload --h5image`` domain at ``/{directory}/{shot}.h5``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import omas
import pytest

from vaft.database import open_ods
from vaft.database import ods as eager_module
from vaft.imas import load_omas_imas


@dataclass(frozen=True)
class Branch:
    """A representative leaf used to time one IDS branch."""

    ids: str
    path: str


BRANCHES = (
    Branch("equilibrium", "equilibrium.time_slice.0.profiles_2d.0.psi"),
    Branch("magnetics", "magnetics.time"),
    Branch("pf_active", "pf_active.coil.0.current.data"),
    Branch("barometry", "barometry.time"),
)


def _seconds(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


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


def _load_legacy_h5image(directory: str, shot: int, output: Path) -> tuple[omas.ODS, dict[str, float | int]]:
    """Materialize the historical single-image domain and restore its ODS."""
    import h5pyd

    domain_uri = f"hdf5://{directory}/{shot}.h5"
    image_path = output / f"{shot}.legacy.omas.h5"
    with h5pyd.File(domain_uri, "r") as domain:
        payload, transfer_seconds = _seconds(lambda: domain["h5image"][:])
    _, write_seconds = _seconds(lambda: payload.tofile(image_path))
    legacy = omas.ODS(consistency_check=False)
    _, conversion_seconds = _seconds(
        lambda: legacy.load(str(image_path), consistency_check=False)
    )
    return legacy, {
        "downloaded_bytes": int(payload.nbytes),
        "download_seconds": transfer_seconds,
        "local_image_write_seconds": write_seconds,
        "ods_conversion_seconds": conversion_seconds,
        "total_seconds": transfer_seconds + write_seconds + conversion_seconds,
    }


def benchmark_load_paths(directory: str, shot: int) -> dict[str, Any]:
    """Execute one read-only timing pass and return a JSON-serializable report."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "directory": directory,
        "shot": int(shot),
        "branches": {},
        "methods": {},
        "notes": [
            "Eager IMAS staging downloads the complete shot before paths= limits conversion.",
            "Legacy h5image is a complete compressed ODS image; IDS lookup after load is local.",
            "Lazy selection reports returned logical bytes, not compressed HTTP wire bytes.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="vaft-hsds-load-benchmark-") as temporary:
        root = Path(temporary)
        eager_stage = root / "eager" / str(shot)
        eager_stage.mkdir(parents=True)

        _, eager_download_seconds = _seconds(
            lambda: eager_module._download_remote_shot(directory, shot, eager_stage)
        )
        staged_files = sorted(eager_stage.glob("*.h5"))
        report["methods"]["eager_imas"] = {
            "download_seconds": eager_download_seconds,
            "downloaded_bytes": sum(path.stat().st_size for path in staged_files),
            "downloaded_files": [path.name for path in staged_files],
            "per_branch": {},
        }

        legacy, legacy_stats = _load_legacy_h5image(directory, shot, root)
        report["methods"]["legacy_h5image"] = {
            **legacy_stats,
            "per_branch": {},
        }
        report["methods"]["lazy_hsds"] = {"per_branch": {}}

        for branch in BRANCHES:
            eager, eager_conversion_seconds = _seconds(
                lambda branch=branch: load_omas_imas(
                    paths=[branch.ids],
                    consistency_check=False,
                    verbose=False,
                    uri="imas:hdf5?path=" + str(eager_stage),
                )
            )
            eager_value, eager_lookup_seconds = _seconds(lambda: eager[branch.path])

            legacy_value, legacy_lookup_seconds = _seconds(lambda: legacy[branch.path])

            lazy_ods, lazy_open_seconds = _seconds(
                lambda branch=branch: open_ods(shot, directory=directory, ids=branch.ids)
            )
            try:
                lazy_value, lazy_selection_seconds = _seconds(
                    lambda: lazy_ods[branch.path]
                )
                _, lazy_cached_seconds = _seconds(lambda: lazy_ods[branch.path])
                opened_ids = list(lazy_ods.store.opened_ids)
            finally:
                lazy_ods.close()

            report["branches"][branch.ids] = {
                "path": branch.path,
                "eager_value": _value_summary(eager_value),
                "lazy_matches_eager": _same_value(lazy_value, eager_value),
                "legacy_matches_eager": _same_value(legacy_value, eager_value),
            }
            report["methods"]["eager_imas"]["per_branch"][branch.ids] = {
                "ods_conversion_seconds": eager_conversion_seconds,
                "local_lookup_seconds": eager_lookup_seconds,
                "projected_cold_total_seconds": eager_download_seconds
                + eager_conversion_seconds
                + eager_lookup_seconds,
            }
            report["methods"]["legacy_h5image"]["per_branch"][branch.ids] = {
                "local_lookup_seconds": legacy_lookup_seconds,
                "projected_cold_total_seconds": legacy_stats["total_seconds"]
                + legacy_lookup_seconds,
            }
            report["methods"]["lazy_hsds"]["per_branch"][branch.ids] = {
                "open_seconds": lazy_open_seconds,
                "selection_seconds": lazy_selection_seconds,
                "cached_second_access_seconds": lazy_cached_seconds,
                "projected_cold_total_seconds": lazy_open_seconds + lazy_selection_seconds,
                "returned_logical_bytes": int(np.asarray(lazy_value).nbytes),
                "opened_ids": opened_ids,
                "hsget_subprocess": False,
            }
    return report


@pytest.mark.integration
def test_hsds_load_path_benchmark() -> None:
    """Write a manual benchmark report when explicitly enabled by the operator."""
    if os.environ.get("VAFT_RUN_HSDS_BENCHMARK") != "1":
        pytest.skip("set VAFT_RUN_HSDS_BENCHMARK=1 to run the read-only HSDS benchmark")
    directory = os.environ.get("VAFT_BENCHMARK_DIRECTORY", "public")
    shot = int(os.environ.get("VAFT_BENCHMARK_SHOT", "39915"))
    report_path = Path(
        os.environ.get(
            "VAFT_BENCHMARK_REPORT",
            f"/tmp/vaft-hsds-load-benchmark-{directory}-{shot}.json",
        )
    )
    report = benchmark_load_paths(directory, shot)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote benchmark report: {report_path}")
