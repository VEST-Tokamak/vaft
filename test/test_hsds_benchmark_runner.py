"""The #1869 benchmark runner's logic, without HSDS: probes, scenarios, summary and correctness."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

RUNNER = Path(__file__).resolve().parents[1] / "workflow" / "hsds_benchmark" / "run_benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("hsds_benchmark_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeODS(dict):
    """Dict keyed by full path; ``len`` of an AOS path is the count of its elements."""

    def __init__(self, data, counts):
        super().__init__(data)
        self.counts, self.prefetched = counts, []

    def __getitem__(self, path):
        if path in self.counts:
            return [None] * self.counts[path]
        return dict.__getitem__(self, path)

    def prefetch(self, ids):
        self.prefetched.append(ids)


def _ods(psi_scale=1.0):
    data = {
        "equilibrium.time": np.array([0.31, 0.32]),
        "magnetics.ip.0.data": np.arange(4.0),
        "pf_active.coil.0.current.data": np.ones(3),
        "magnetics.b_field_pol_probe.0.position.r": 0.5,
        "magnetics.b_field_pol_probe.0.field.data": np.zeros(4),
    }
    for i in range(2):
        data[f"equilibrium.time_slice.{i}.global_quantities.ip"] = 1e5 + i
        data[f"equilibrium.time_slice.{i}.profiles_2d.0.psi"] = psi_scale * np.ones((3, 3)) * i
    return FakeODS(data, {"equilibrium.time_slice": 2, "pf_active.coil": 1, "magnetics.b_field_pol_probe": 1})


def test_probes_read_scalars_always_and_large_arrays_only_when_asked():
    bench = _load()
    small, large = bench.read_probes(_ods(), large=False), bench.read_probes(_ods(), large=True)
    assert "equilibrium.time_slice.1.global_quantities.ip" in small
    assert "equilibrium.time_slice.1.global_quantities.li_3" not in small  # absent leaves are skipped
    assert not any("psi" in p for p in small) and any("psi" in p for p in large)
    assert "magnetics.b_field_pol_probe.0.position.r" in small and "magnetics.b_field_pol_probe.0.field.data" in large
    assert set(small) <= set(large)  # every lazy scalar value is compared against the eager runs
    assert bench.digest(np.ones(3)) != bench.digest(np.ones(4)) != bench.digest(np.ones((2, 2)))


def test_scenarios_use_the_public_entry_points_and_record_io(monkeypatch):
    bench = _load()
    import vaft.database as database

    calls = []
    lazy = _ods()
    monkeypatch.setattr(database, "load", lambda shot, source, cache, transport: calls.append(("load", transport, cache)) or _ods())
    monkeypatch.setattr(database, "open", lambda shot, source: calls.append(("open",)) or lazy)
    eager = bench.run_scenario("eager_h5image", 39915, "main", "/tmp/cache-x")
    prefetch = bench.run_scenario("prefetch", 39915, "main", None)
    assert calls == [("load", "h5image", "/tmp/cache-x"), ("open",)]
    assert lazy.prefetched == list(bench.IDS)
    for result in (eager, prefetch):
        assert result["error"] is None and result["wall_s"] >= 0 and result["peak_rss_mb"] > 0
        assert result["io"]["request_count"] == 0  # nothing went over HTTP here

    monkeypatch.setattr(database, "load", lambda *a, **k: (_ for _ in ()).throw(OSError("HSDS unreachable")))
    failed = bench.run_scenario("eager_canonical", 39915, "main", None)
    assert failed["error"].startswith("OSError") and failed["digests"] == {}


def test_summary_flags_a_path_whose_value_differs_from_eager_canonical():
    bench = _load()
    good = bench.read_probes(_ods(), large=True)
    wrong = bench.read_probes(_ods(psi_scale=2.0), large=True)

    def record(scenario, digests, wall, error=None):
        return {"scenario": scenario, "shot": 39915, "clients": 1, "cache_state": "cold", "wall_s": wall,
                "error": error, "digests": digests, "peak_rss_mb": 100.0,
                "io": {"request_count": 10, "response_bytes_observed": 1000, "subprocess_seconds": 0.5}}

    rows = {r["scenario"]: r for r in bench.summarise([
        record("eager_canonical", good, 2.0), record("eager_canonical", good, 4.0),
        record("lazy_large", wrong, 1.0), record("lazy_scalar", good, 0.5, error="Timeout"),
    ])}
    assert rows["eager_canonical"]["wall_p50_s"] == 3.0 and rows["eager_canonical"]["mismatched_paths"] == []
    assert rows["lazy_large"]["mismatched_paths"] == ["equilibrium.time_slice.1.profiles_2d.0.psi"]
    assert rows["lazy_large"]["compared_to_eager_canonical"] == len(wrong)
    assert rows["lazy_scalar"]["errors"] == 1 and rows["lazy_scalar"]["wall_p50_s"] is None
    assert rows["lazy_large"]["reference_available"] and rows["lazy_large"]["missing_paths"] == []


def _record(scenario, digests, shot=39915, error=None, probe_errors=()):
    return {"scenario": scenario, "shot": shot, "clients": 1, "cache_state": "cold", "wall_s": 1.0,
            "error": error, "digests": digests, "peak_rss_mb": 100.0, "probe_errors": list(probe_errors),
            "io": {"request_count": 1, "response_bytes_observed": 1, "subprocess_seconds": 0.0}}


def test_summary_flags_missing_values_failed_reads_and_an_absent_reference():
    bench = _load()
    full = bench.read_probes(_ods(), large=True)
    scalar = bench.read_probes(_ods(), large=False)
    dropped = {p: d for p, d in full.items() if p != "pf_active.coil.0.current.data"}
    rows = {(r["scenario"], r["shot"]): r for r in bench.summarise([
        _record("eager_canonical", full),
        _record("lazy_scalar", scalar),  # scalar scenarios are not expected to read psi / fields
        _record("eager_h5image", dropped, probe_errors=["pf_active.coil.0.current.data: OSError: 503"]),
        _record("lazy_scalar", scalar, shot=41524),  # no canonical run for this shot
    ])}
    assert rows[("lazy_scalar", 39915)]["missing_paths"] == []
    assert rows[("eager_h5image", 39915)]["missing_paths"] == ["pf_active.coil.0.current.data"]
    assert rows[("eager_h5image", 39915)]["probe_errors"] == 1
    assert not rows[("lazy_scalar", 41524)]["reference_available"]
    assert rows[("lazy_scalar", 41524)]["compared_to_eager_canonical"] == 0


def test_canonical_runs_that_disagree_are_reported_not_overwritten():
    bench = _load()
    a = bench.read_probes(_ods(), large=True)
    b = bench.read_probes(_ods(psi_scale=2.0), large=True)
    rows = {r["scenario"]: r for r in bench.summarise([_record("eager_canonical", a), _record("eager_canonical", b)])}
    assert rows["eager_canonical"]["canonical_disagreement"] == ["equilibrium.time_slice.1.profiles_2d.0.psi"]


def test_a_failed_read_is_recorded_while_an_absent_path_is_not():
    bench = _load()

    class Broken(FakeODS):
        def __getitem__(self, path):
            if path == "magnetics.ip.0.data":
                raise OSError("HSDS 503")
            return super().__getitem__(path)

    ods = _ods()
    errors = []
    digests = bench.read_probes(Broken(dict(ods), ods.counts), large=False, errors=errors)
    assert "magnetics.ip.0.data" not in digests
    assert len(errors) == 1 and errors[0].startswith("magnetics.ip.0.data: OSError")  # li_3 is absent: not an error


def _fake_client(args, barrier, queue, index):
    """Stands in for the runner's client: the process mechanics without HSDS."""
    barrier.wait(timeout=60)
    queue.put((index, {"wall_s": 0.0, "io": {}, "peak_rss_mb": 0.0, "digests": {}, "probe_errors": [],
                       "error": None, "pid": os.getpid()}))


def test_each_client_runs_in_its_own_process(tmp_path):
    bench = _load()
    results = bench.run_group("lazy_scalar", 1, "main", 3, "cold", tmp_path, "t", target=_fake_client)
    pids = {r["pid"] for r in results}
    assert len(pids) == 3 and os.getpid() not in pids
    assert all(r["error"] is None for r in results)


def test_a_client_that_dies_without_reporting_does_not_hang_the_group(tmp_path):
    bench = _load()  # loaded under a name spawned children cannot import: they die at unpickling
    results = bench.run_group("lazy_scalar", 1, "main", 2, "cold", tmp_path, "t")
    assert [r["error"].startswith("client exited with code") for r in results] == [True, True]
