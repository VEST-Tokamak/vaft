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


def test_the_magnetic_fit_uses_only_fitted_channels_in_their_own_units(module):
    variables = {
        # Two fitted probes, 10% and 0% off; a third is off by a factor of 10 but unweighted.
        "expmpi": np.array([0.1, 0.2, 0.3]), "cmpr2": np.array([0.11, 0.2, 3.0]),
        "fwtmp2": np.array([1.0, 1.0, 0.0]),
        "silopt": np.array([0.01, -0.02]), "csilop": np.array([0.01, -0.02]),
        "fwtsi": np.array([1.0, 1.0]),
    }

    fit = module.magnetic_fit(variables)

    expected = np.sqrt(np.mean([0.01**2, 0.0])) / np.sqrt(np.mean([0.1**2, 0.2**2]))
    assert fit["probe"] == pytest.approx(expected)
    assert fit["probe_n"] == 2
    assert fit["loop"] == 0.0
    assert np.isnan(module.magnetic_fit({})["probe"])


def test_the_uncertainty_model_reaches_the_constraint_config(module):
    case = module.configurations()[0]

    assert module._scientific(case).constraints.uncertainty_mode == "legacy_weight"
    statistical = module._scientific(case, "standard_deviation")
    assert statistical.constraints.uncertainty_mode == "standard_deviation"
    assert statistical.numerics.error_minimum == case["error_minimum"]


KFILE = """ &IN1
 ISHOT = 41672
 ERRMIN = 0.01
 /
 &INWANT
 FITDZ = 1
 /
"""


def test_overrides_land_inside_the_named_block_only_and_are_recorded(module, tmp_path):
    path = tmp_path / "k041672.00331"
    path.write_text(KFILE)

    written = module.patch_namelist(path, "INWANT", {"FITDELZ": True, "ERRDELZ": 0.06, "STABDZ": 1e-4})

    assert written == [" FITDELZ = .TRUE.", " ERRDELZ = 0.06", " STABDZ = 0.0001"]
    lines = path.read_text().splitlines()
    # &IN1 is untouched, and the keys close &INWANT.
    assert lines[:4] == [" &IN1", " ISHOT = 41672", " ERRMIN = 0.01", " /"]
    assert lines[4:] == [" &INWANT", " FITDZ = 1", *written, " /"]


def test_fitdelz_goes_where_efit_reads_it(module):
    # data_input.F90:222-228 -- written into &IN1, EFIT refuses the k-file.
    assert module.FITDELZ_GROUP == "INWANT"


def test_overrides_refuse_a_key_the_writer_already_set(module, tmp_path):
    path = tmp_path / "k041672.00331"
    path.write_text(KFILE)

    with pytest.raises(ValueError, match="ERRMIN"):
        module.patch_namelist(path, "IN1", {"ERRMIN": 1e-4})
    assert path.read_text() == KFILE


def test_the_fitted_vertical_shift_is_read_per_slice_from_its_last_iteration(module):
    log = "\n".join([
        " r=  0 t=   321 it=  1 chi2=1.0E+02 zm= 1.0E-02 err=5.0E-01 dz= 1.0E-03 delz= 0.000E+00 dj=0.000E+00",
        " r=  0 t=   321 it=  9 chi2=1.0E+02 zm= 1.0E-02 err=5.0E-03 dz= 1.0E-04 delz=-2.500E-03 dj=1.000E-02",
        " r=  0 t=   331 it=  4 chi2=1.0E+02 zm= 1.0E-02 err=5.0E-03 dz= 1.0E-04 delz= 7.000E-03 dj=1.000E-02",
        " r=  0 t=   342 it=  4 chi2=1.0E+02 zm= 1.0E-02 err=5.0E-03 dz= 1.0E-04 chigam= 0.00E+00",
    ])

    assert module.final_delz(log, 321) == pytest.approx(-2.5e-3)
    assert module.final_delz(log, 331) == pytest.approx(7.0e-3)
    # FITDELZ off prints no delz column.
    assert module.final_delz(log, 342) is None


def test_probes_are_excluded_by_name_not_by_position(module):
    from omas import ODS

    ods = ODS(consistency_check=False)
    for index, name in enumerate(["MagneticFieldProbe_C4-03", "MagneticFieldProbe_C4-04", "MagneticFieldProbe_C4-06"]):
        ods[f"magnetics.b_field_pol_probe.{index}.name"] = name

    assert module.probe_indexes(ods, ["MagneticFieldProbe_C4-04"]) == {"MagneticFieldProbe_C4-04": 1}
    with pytest.raises(ValueError, match="C4-05"):
        module.probe_indexes(ods, ["MagneticFieldProbe_C4-05"])


# --------------------------------------------------------------------------
# The recorded run (2026-09-18).  A re-run that reverses any of these is a
# finding, and should fail here rather than be discovered in a README.
# --------------------------------------------------------------------------

RECORDED = Path(__file__).resolve().parent / "data" / "efit_convergence_study.json"


@pytest.fixture(scope="module")
def recorded():
    import json

    return json.loads(RECORDED.read_text(encoding="utf-8"))


def _rows(recorded, *, table="packaged", grid=129, inner=None, error_minimum=None):
    return [
        row for row in recorded["analysis"]["rows"]
        if row["case"]["table"] == table and row["case"]["grid"] == grid
        and (inner is None or row["case"]["inner_iterations"] == inner)
        and (error_minimum is None or row["case"]["error_minimum"] == error_minimum)
    ]


def test_the_recorded_run_is_on_an_efit_whose_ip_chi_squared_matches_its_solve(recorded):
    # #918: stock EFIT's NXITER > 1 stop test is contaminated; the research
    # build that removes the vessel term from the reported Ip chi-square is 3b5dae5.
    assert recorded["efit_install"]["source_revision"] == "3b5dae5"
    assert recorded["toolchain"]["efit"]["sha256"].startswith("4a4e645e")


def test_every_setting_that_reaches_errmin_1e_4_lands_on_the_same_equilibrium(recorded):
    rows = [r for r in _rows(recorded, error_minimum=1e-4) if r["converged"]]
    assert len({r["case"]["inner_iterations"] for r in rows}) == 4
    assert max(r["iteration_error"]["lcfs_rms_mm"] for r in rows) < 0.1


def test_the_routine_stop_is_not_converged(recorded):
    routine = _rows(recorded, inner=1, error_minimum=1e-2)
    assert all(r["converged"] for r in routine)
    distances = [r["iteration_error"]["lcfs_rms_mm"] for r in routine]
    assert np.median(distances) > 5.0
    assert max(distances) > 50.0


def test_more_inner_iterations_only_lose_slices(recorded):
    for error_minimum in (1e-2, 1e-3, 1e-4):
        kept = {
            inner: sum(r["converged"] for r in _rows(recorded, inner=inner, error_minimum=error_minimum))
            for inner in (1, 3, 5, 10)
        }
        assert all(kept[1] > kept[inner] for inner in (3, 5, 10)), (error_minimum, kept)


def test_the_grid_moves_the_converged_equilibrium_by_about_a_millimetre(recorded):
    grid = [ref["discretization"] for ref in recorded["analysis"]["references"].values()
            if ref["discretization"]]
    assert len(grid) >= 8
    assert max(d["lcfs_rms_mm"] for d in grid) < 1.5
    # ... while the edge residual falls several-fold, so most of the converged
    # 129 residual is the grid.
    assert all(d["gs_257"]["edge"] < 0.5 * d["gs_129"]["edge"] for d in grid)


def test_a_regenerated_129_table_changes_nothing(recorded):
    controls = [ref["table_regeneration"] for ref in recorded["analysis"]["references"].values()
                if ref["table_regeneration"]]
    assert controls
    assert max(c["lcfs_rms_mm"] for c in controls) < 1e-3


def test_no_setting_is_recommended_under_legacy_sigma(recorded):
    # The nearest, NXITER=1 / ERRMIN=1e-4, loses 39915 @ 327 to `bound`; the
    # README records why a recommendation waits for the sigma contract.
    verdict = recorded["analysis"]["recommendation"]
    assert verdict["recommended"] is None
    nearest = verdict["cases"]["g129_nx1_err1e-04"]["failures"]
    assert nearest == ["39915@327: did not stop on its criterion"]


STATISTICAL = Path(__file__).resolve().parent / "data" / "efit_convergence_study_standard_deviation.json"


def test_no_slice_converges_under_the_statistical_sigma_as_configured():
    # README: probe sigma 1% is ~4x below the residual the fit reaches, and
    # probe C4-04 carries 70% of the chi-square; 39915 oscillates to the cap,
    # 41524 and 41672 die in `bound`/`findax` within 27 iterations.
    import json

    recorded = json.loads(STATISTICAL.read_text(encoding="utf-8"))
    assert recorded["uncertainty_mode"] == "standard_deviation"
    rows = recorded["analysis"]["rows"]
    assert len(rows) == 81
    assert not any(row["converged"] for row in rows)
    exits = {(row["shot"], row["exit_path"]) for row in rows}
    assert exits == {
        (39915, "iterations_exhausted"),
        (41524, "solver_error"),
        (41672, "solver_error"),
    }


FITDELZ = Path(__file__).resolve().parent / "data" / "efit_fitdelz_experiment.json"


def test_fitdelz_does_not_remove_the_upward_drift_under_legacy_sigma():
    # README: the drift between ERRMIN 1e-2 and 1e-4 is a rigid upward shift,
    # and EFIT's own rigid-shift fit leaves the large ones where they were.
    import json

    runs = json.loads(FITDELZ.read_text(encoding="utf-8"))["runs"]
    for label in ("legacy", "legacy_fitdelz"):
        drifts = {(s["shot"], s["time_ms"]): s["drift"] for s in runs[label]["slices"] if s["drift"]}
        assert drifts[(41672, 321)]["rigid_dz_mm"] > 150.0
        assert drifts[(41524, 327)]["rigid_dz_mm"] > 50.0
        # A rigid shift is almost all of it.
        assert drifts[(41672, 321)]["after_shift_mm"] < 0.2 * drifts[(41672, 321)]["lcfs_rms_mm"]
    assert runs["legacy_fitdelz"]["namelist_overrides"] == {"FITDELZ": True}


def test_no_statistical_sigma_run_converges_with_or_without_fitdelz():
    import json

    runs = json.loads(FITDELZ.read_text(encoding="utf-8"))["runs"]
    for label in ("sd_without_c4_04", "sd_without_c4_04_fitdelz", "sd_without_c4_04_fitdelz_forced"):
        block = runs[label]
        assert block["uncertainty_mode"] == "standard_deviation"
        assert block["excluded_probes"] == ["MagneticFieldProbe_C4-04"]
        assert not any(s["errmin_1e-4"]["converged"] for s in block["slices"])
    # Left to ERRDELZ = 0.06 the shift never switches on; forced on, it runs away.
    assert all((s["errmin_1e-4"]["delz_m"] or 0.0) == 0.0 for s in runs["sd_without_c4_04_fitdelz"]["slices"])
    assert max(abs(s["errmin_1e-4"]["delz_m"] or 0.0) for s in runs["sd_without_c4_04_fitdelz_forced"]["slices"]) > 1.0
