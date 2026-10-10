"""The #1869 benchmark runner's logic, without HSDS: probes, scenarios, summary and correctness."""

from __future__ import annotations

import importlib.util
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


def test_unknown_scenarios_are_refused(tmp_path, capsys):
    bench = _load()
    try:
        bench.main(["--scenarios", "lazy_scalar,warp_drive", "--output", str(tmp_path)])
    except SystemExit as exit_:
        assert exit_.code == 2
    assert "warp_drive" in capsys.readouterr().err
