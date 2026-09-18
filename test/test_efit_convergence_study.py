"""The convergence study (#924): the bookkeeping its verdict rests on.

The study runs EFIT; what is pinned here is how it plans the runs, pairs
slices, measures the distance between two equilibria and picks the references
and the recommendation.  Every fixture has at least two slices, because a
single slice cannot tell a per-slice rule from a positional one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_numerics" / "convergence_study.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("convergence_study", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = loaded
    try:
        spec.loader.exec_module(loaded)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield loaded
    sys.modules.pop(spec.name, None)


def test_the_iteration_cap_never_lets_the_cumulative_counter_reach_efits_array_bound(module):
    for inner in (1, 3, 5, 10):
        cap = module.max_iterations_for(inner)
        assert cap * inner < module.ITERATION_ARRAY_BOUND
        assert (cap + 1) * inner >= module.ITERATION_ARRAY_BOUND
    assert module.max_iterations_for(10) == 51
    with pytest.raises(ValueError):
        module.max_iterations_for(0)


def test_the_plan_is_the_full_coarse_cross_then_the_fine_references(module):
    plan = module.configurations()

    coarse = [case for case in plan if case["table"] == module.PACKAGED]
    generated = [case for case in plan if case["table"] == module.GENERATED]
    assert {case["grid"] for case in coarse} == {module.ROUTINE_GRID}
    assert len(coarse) == len(module.INNER_ITERATIONS) * len(module.ERROR_MINIMUM)
    # Every grid-reference setting runs on both generated grids, so the grid
    # is the only difference between a pair.
    for inner, error_minimum in module.FINE_SETTINGS:
        pair = [c for c in generated if (c["inner_iterations"], c["error_minimum"]) == (inner, error_minimum)]
        assert sorted(c["grid"] for c in pair) == [module.ROUTINE_GRID, module.FINE_GRID]
    assert plan.index(generated[0]) > plan.index(coarse[-1])
    assert len({case["name"] for case in plan}) == len(plan)
    assert all(case["max_iterations"] * case["inner_iterations"] < 515 for case in plan)


def test_slices_are_matched_to_the_shot_by_time_not_position(module):
    base = [0.320, 0.321, 0.3315, 0.331, 0.342]

    chosen = module.select_times(base, [0.342, 0.321, 0.331])

    # Ascending, and the value that was asked for -- not base[0..2].
    assert chosen == [0.321, 0.331, 0.342]
    with pytest.raises(ValueError, match="0.333"):
        module.select_times(base, [0.321, 0.333])


def _circle(radius, n, r0=0.4, z0=0.0, phase=0.0):
    angle = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    return r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)


def test_boundary_distance_is_symmetric_and_independent_of_sampling(module):
    a = _circle(0.300, 400)
    b = _circle(0.301, 97, phase=0.3)

    forward = module.boundary_distance(*a, *b)
    backward = module.boundary_distance(*b, *a)

    assert forward == pytest.approx(backward)
    # Concentric circles are 1 mm apart everywhere, up to the chord sag of the
    # coarser polygon (r * (1 - cos(pi/97)) ~ 0.16 mm).
    assert forward["rms"] == pytest.approx(1.0e-3, abs=2.0e-4)
    assert forward["hausdorff"] == pytest.approx(1.0e-3, abs=2.0e-4)
    assert module.boundary_distance(*a, *a)["hausdorff"] == 0.0


def _mapping(*, radius=0.3, axis_r=0.4, pressure=1.0e3, nw=65):
    """A circular equilibrium whose flux is a paraboloid; enough for the geometry."""
    r = np.linspace(0.05, 0.9, nw)
    z = np.linspace(-0.5, 0.5, nw)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    simag, sibry = -0.02, 0.0
    psi = simag + (sibry - simag) * ((rr - axis_r) ** 2 + zz**2) / radius**2
    boundary_r, boundary_z = _circle(radius, 120, r0=axis_r)
    x = np.linspace(0.0, 1.0, 33)
    return {
        "NW": nw, "NH": nw, "RLEFT": 0.05, "RDIM": 0.85, "ZMID": 0.0, "ZDIM": 1.0,
        "PSIRZ": psi, "SIMAG": simag, "SIBRY": sibry,
        "RMAXIS": axis_r, "ZMAXIS": 0.0,
        "RBBBS": boundary_r, "ZBBBS": boundary_z,
        "PRES": pressure * (1.0 - x), "PPRIME": np.full(x.size, pressure / (sibry - simag)),
        "FFPRIM": np.full(x.size, 0.05), "QPSI": 1.0 + 2.0 * x**2,
    }


def _record(mapping, **scalars):
    base = {"betap": 0.5, "li": 1.0, "q95": 3.0, "wmhd": 1.0e3, "area": 0.28, "volume": 0.7}
    base.update(scalars)
    return {"geqdsk": mapping, "scalars": base}


def test_an_equilibrium_is_zero_distance_from_itself(module):
    record = _record(_mapping())

    distance = module.compare(record, record)

    for key, value in distance.items():
        assert value == pytest.approx(0.0, abs=1e-12), key


def test_compare_reports_a_moved_boundary_and_changed_scalars(module):
    reference = _record(_mapping())
    moved = _record(_mapping(radius=0.302, axis_r=0.401), betap=0.51)

    distance = module.compare(moved, reference)

    assert distance["lcfs_rms_mm"] == pytest.approx(2.0, abs=1.1)
    assert distance["axis_shift_mm"] == pytest.approx(1.0, rel=1e-6)
    assert distance["betap_relative"] == pytest.approx(0.02)
    assert distance["li_relative"] == 0.0
    assert np.isfinite(distance["jphi_midplane_relative"])


def _row(module, shot, time_ms, inner, error_minimum, *, grid=129, table="packaged",
         exit_path="iconvr=2", seconds=10.0):
    case = {"grid": grid, "table": table, "inner_iterations": inner, "error_minimum": error_minimum,
            "name": module.case_name(grid, inner, error_minimum, table)}
    return {"case": case, "exit_path": exit_path, "collapsed": False, "geqdsk": {"stub": time_ms},
            "shot": shot, "time_ms": time_ms, "seconds": seconds}


def test_the_reference_is_the_tightest_run_that_stopped_on_its_criterion(module):
    records = [
        _row(module, 41672, 331, 1, 1e-2),
        _row(module, 41672, 331, 3, 1e-3),
        # Tighter, but it ran out of iterations: not a converged answer.
        _row(module, 41672, 331, 10, 1e-4, exit_path="iterations_exhausted"),
        _row(module, 41672, 331, 5, 1e-4),
        _row(module, 41672, 331, 10, 1e-4, grid=257, table="generated"),
        # The generated 129 table's run is not a candidate for the routine reference.
        _row(module, 41672, 331, 10, 1e-4, table="generated"),
    ]

    coarse = module.choose_reference(records, 129, "packaged")
    fine = module.choose_reference(records, 257, "generated")

    assert coarse["case"]["name"] == module.case_name(129, 5, 1e-4)
    assert fine["case"]["grid"] == 257
    assert module.choose_reference(records[:1], 257, "generated") is None


def test_grid_error_pairs_the_two_generated_grids_at_one_setting(module):
    records = [
        # The tightest setting converged on 257 but not on 129: not a pair.
        _row(module, 41672, 331, 10, 1e-4, grid=257, table="generated"),
        _row(module, 41672, 331, 10, 1e-4, table="generated", exit_path="iterations_exhausted"),
        _row(module, 41672, 331, 5, 1e-4, grid=257, table="generated"),
        _row(module, 41672, 331, 5, 1e-4, table="generated"),
        # Same setting on the packaged table: never paired with 257.
        _row(module, 41672, 331, 10, 1e-4),
    ]

    pair = module.same_setting_pair(records, (129, "generated"), (257, "generated"))

    assert [r["case"]["name"] for r in pair] == [
        module.case_name(129, 5, 1e-4, "generated"),
        module.case_name(257, 5, 1e-4, "generated"),
    ]
    assert module.same_setting_pair(records[:2], (129, "generated"), (257, "generated")) is None


def _verdict_row(module, shot, time_ms, inner, error_minimum, *, lcfs, seconds, converged=True, ratio=1.0):
    case = {"grid": 129, "table": "packaged", "inner_iterations": inner, "error_minimum": error_minimum,
            "name": module.case_name(129, inner, error_minimum)}
    return {
        "shot": shot, "time_ms": time_ms, "case": case, "converged": converged, "seconds": seconds,
        "iteration_error": {"lcfs_rms_mm": lcfs, "betap_relative": 1e-3, "li_relative": 1e-3},
        "gs_edge_over_floor": ratio,
    }


def test_the_recommendation_is_the_cheapest_setting_within_tolerance_on_every_slice(module):
    rows = [
        # Routine: fast, but 3 mm off on one slice.
        _verdict_row(module, 41672, 331, 1, 1e-2, lcfs=3.0, seconds=5),
        _verdict_row(module, 41672, 342, 1, 1e-2, lcfs=0.2, seconds=5),
        # Within tolerance everywhere.
        _verdict_row(module, 41672, 331, 3, 1e-3, lcfs=0.5, seconds=20),
        _verdict_row(module, 41672, 342, 3, 1e-3, lcfs=0.1, seconds=20),
        # Within tolerance and cheaper, but one slice did not converge.
        _verdict_row(module, 41672, 331, 1, 1e-3, lcfs=0.4, seconds=8),
        _verdict_row(module, 41672, 342, 1, 1e-3, lcfs=0.0, seconds=8, converged=False),
        # The tightest: within tolerance, expensive.
        _verdict_row(module, 41672, 331, 10, 1e-4, lcfs=0.0, seconds=200),
        _verdict_row(module, 41672, 342, 10, 1e-4, lcfs=0.0, seconds=200),
    ]

    verdict = module.recommend(rows)

    assert verdict["recommended"] == module.case_name(129, 3, 1e-3)
    routine = verdict["cases"][module.case_name(129, 1, 1e-2)]
    assert not routine["within_tolerance"]
    assert any("331" in failure and "lcfs" in failure for failure in routine["failures"])
    assert not verdict["cases"][module.case_name(129, 1, 1e-3)]["within_tolerance"]


def test_no_recommendation_when_nothing_is_within_tolerance(module):
    rows = [
        _verdict_row(module, 41672, 331, 1, 1e-2, lcfs=3.0, seconds=5),
        _verdict_row(module, 41672, 342, 1, 1e-2, lcfs=0.1, seconds=5, ratio=1.5),
    ]

    assert module.recommend(rows)["recommended"] is None
