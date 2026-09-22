"""The #891 sigma calibration: its ladder, its floor, its cold-start guard and its verdict rule.

The study runs EFIT; what is pinned here is the bookkeeping the verdict rests
on.  Fixtures carry two slices, because a floor or a summary that is right on
one slice can still be computed from the wrong one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_uncertainty_calibration" / "sigma_calibration.py"


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("sigma_calibration", SCRIPT)
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


def test_the_ladder_is_each_family_alone_and_both_together_per_floor(module):
    plan = module.rungs()

    assert len(plan) == 16 * len(module.FLOORS) * len(module.BASES)
    assert len({rung["name"] for rung in plan}) == len(plan)
    baseline = plan[0]
    assert baseline["multipliers"] == {"bpol_probe": 1.0, "flux_loop": 1.0} and baseline["floor"] == 0.0
    assert baseline["basis"] == list(module.BASES[0])
    assert {tuple(rung["basis"]) for rung in plan} == set(module.BASES)
    # The x1 point serves every ladder, once.
    assert sorted(baseline["ladders"]) == sorted("+".join(ladder) for ladder in module.LADDERS)
    for rung in plan:
        m = rung["multipliers"]
        assert m["bpol_probe"] == 1.0 or m["flux_loop"] == 1.0 or m["bpol_probe"] == m["flux_loop"]


def test_a_multiplier_widens_the_sigma_through_uncertainty_scales(module):
    rung = {"multipliers": {"bpol_probe": 8.0, "flux_loop": 1.0}}

    # uncertainty_scales divides the submitted sigma; the diamagnetic flux is
    # made inactive on every standard_deviation rung (#1027).
    assert module.uncertainty_scales_for(rung) == {
        "bpol_probe": 0.125, "flux_loop": 1.0, "diamagnetic_flux": module.DIAMAGNETIC_INACTIVE_SCALE,
    }


def test_the_chi_square_target_follows_the_number_of_fitted_constraints(module):
    """SAICON 80 is below a statistical sigma's expected chi-square (#1027)."""
    ods = ODS(consistency_check=False)
    root = "equilibrium.time_slice.0.constraints"
    for j, w in enumerate((1.0, 1.0, 0.0, 1.0)):
        ods[f"{root}.bpol_probe.{j}.weight"] = w
    for j in range(2):
        ods[f"{root}.flux_loop.{j}.weight"] = 1.0
    ods[f"{root}.ip.weight"] = 1.0
    ods[f"{root}.diamagnetic_flux.weight"] = 1.0  # not counted: held inactive

    n = module.fitted_constraint_count(ods)

    assert n == 3 + 2 + 1
    assert module.chi_squared_target(75) == pytest.approx(75 + 3 * np.sqrt(150))
    assert module.chi_squared_target(75) > 80.0 > 75


def _constraints(*slices):
    ods = ODS(consistency_check=False)
    for index, probes in enumerate(slices):
        for j, (measured, sigma, weight) in enumerate(probes):
            node = f"equilibrium.time_slice.{index}.constraints.bpol_probe.{j}"
            ods[f"{node}.measured"] = measured
            ods[f"{node}.measured_error_upper"] = sigma
            ods[f"{node}.weight"] = weight
    return ods


def test_the_floor_raises_only_sigma_below_it_per_slice(module):
    ods = _constraints(
        # median |m| of the fitted three = 0.10 -> floor 0.002; the 0-weight channel does not vote
        [(0.10, 0.001, 1.0), (0.001, 0.00001, 1.0), (0.20, 0.004, 1.0), (5.0, 0.05, 0.0)],
        # a second slice with a different median: 0.04 -> floor 0.0008
        [(0.04, 0.0004, 1.0), (0.05, 0.002, 1.0), (0.0, 0.0, 1.0)],
    )

    changes = module.apply_sigma_floor(ods, 0.02, families=("bpol_probe",))

    first, second = changes
    assert first["floor"] == pytest.approx(0.002) and first["raised"] == 2 and first["fitted"] == 3
    assert second["floor"] == pytest.approx(0.0008) and second["raised"] == 2
    sigma = lambda i, j: float(ods[f"equilibrium.time_slice.{i}.constraints.bpol_probe.{j}.measured_error_upper"])
    assert sigma(0, 0) == pytest.approx(0.002)
    assert sigma(0, 1) == pytest.approx(0.002)
    assert sigma(0, 2) == pytest.approx(0.004)  # already above
    assert sigma(0, 3) == pytest.approx(0.05)  # weight 0: untouched
    assert sigma(1, 1) == pytest.approx(0.002)
    assert module.apply_sigma_floor(ods, 0.0) == []


def test_a_run_that_did_not_start_where_the_baseline_did_is_refused(module):
    baseline = module.require_same_start(None, "abc", "first")
    assert baseline == "abc"
    assert module.require_same_start(baseline, "abc", "second") == "abc"
    with pytest.raises(RuntimeError, match="cold-start"):
        module.require_same_start(baseline, "xyz", "warm-started rung")
    with pytest.raises(RuntimeError, match="no initialization fingerprint"):
        module.require_same_start(baseline, None, "unrecorded")


def _row(name, *, probe_m=1.0, loop_m=1.0, converged=9, probe_chi2=1.0, loop_chi2=1.0, residual=0.05, drift=1.0):
    return {
        "rung": name, "uncertainty_mode": "standard_deviation",
        "multipliers": {"bpol_probe": probe_m, "flux_loop": loop_m}, "floor": 0.0, "ladders": [],
        "slices": 9, "converged_tight": converged, "converged_loose": 9,
        "probe_reduced_chi2": probe_chi2, "loop_reduced_chi2": loop_chi2,
        "probe_residual": residual, "loop_residual": 0.05, "probe_sigma": 0.01,
        "median_abs_drift_mm": drift, "median_lcfs_to_reference_mm": 3.0, "median_iterations_tight": 30,
    }


def test_the_operating_range_is_every_rung_that_meets_all_four_criteria(module):
    rows = [
        _row("narrow_but_diverging", probe_m=1, converged=4, probe_chi2=90.0),
        _row("narrowest_converging", probe_m=4, residual=0.040),
        _row("passes", probe_m=8, residual=0.050),
        _row("chi2_too_small", probe_m=32, probe_chi2=0.1, residual=0.055),
        _row("gave_the_probes_up", probe_m=16, residual=0.070),
        _row("still_drifting", probe_m=4, loop_m=4, drift=12.0, residual=0.045),
        {**_row("legacy_reference"), "uncertainty_mode": "legacy_weight"},
    ]

    verdict = module.operating_range(rows)

    assert verdict["narrowest_converging"] == {"unspecified": "narrowest_converging"}
    assert verdict["range"] == ["narrowest_converging", "passes"]
    failures = verdict["failures"]
    assert any("converges on 4/9" in f for f in failures["narrow_but_diverging"])
    assert any("probe reduced chi2" in f for f in failures["chi2_too_small"])
    assert any("probe residual" in f for f in failures["gave_the_probes_up"])
    assert any("drift" in f for f in failures["still_drifting"])
    assert "legacy_reference" not in failures  # a reference, never a candidate


def test_no_range_is_reported_when_nothing_converges_everywhere(module):
    verdict = module.operating_range([_row("baseline", converged=0, probe_chi2=240.0)])

    assert verdict["range"] == []
    assert verdict["narrowest_converging"] == {"unspecified": None}
    assert any("no rung converges everywhere" in f for f in verdict["failures"]["baseline"])


def test_the_summary_counts_convergence_on_the_tight_stop_only(module):
    plan = [module.rungs()[0]]
    name = plan[0]["name"]
    records = [
        {"rung": name, "shot": 41672, "time_ms": t, "error_minimum": e, "converged": ok, "iterations_n": 20,
         "fit": {"probe_reduced_chi2": 2.0, "loop_reduced_chi2": 1.0, "probe_median_relative": 0.05,
                 "loop_median_relative": 0.04, "probe_sigma_median": 0.01},
         "drift": {"dz_mm": dz} if e == 1e-4 else None}
        for t, e, ok, dz in ((321, 1e-2, True, None), (321, 1e-4, True, -3.0),
                             (331, 1e-2, True, None), (331, 1e-4, False, 7.0))
    ]

    (row,) = module.summarize(records, plan)

    assert row["slices"] == 2
    assert row["converged_tight"] == 1 and row["converged_loose"] == 2
    assert row["median_abs_drift_mm"] == pytest.approx(5.0)
    assert row["probe_reduced_chi2"] == pytest.approx(2.0)


def test_each_profile_basis_is_judged_against_its_own_narrowest_rung(module):
    """A wide sigma on (2,1) must not be compared with (1,1)'s residual."""
    rows = [
        {**_row("p1f1_x2", probe_m=2, residual=0.030), "basis": [1, 1]},
        {**_row("p1f1_x8", probe_m=8, residual=0.060), "basis": [1, 1]},
        {**_row("p2f1_x4", probe_m=4, residual=0.050), "basis": [2, 1]},
        {**_row("p2f1_x8", probe_m=8, residual=0.060), "basis": [2, 1]},
    ]

    verdict = module.operating_range(rows)

    assert verdict["narrowest_converging"] == {"p1f1": "p1f1_x2", "p2f1": "p2f1_x4"}
    # 0.060 is over 1.5x (1,1)'s 0.030 but within 1.5x (2,1)'s 0.050.
    assert "p1f1_x8" not in verdict["range"]
    assert "p2f1_x8" in verdict["range"]


def test_a_checkpoint_is_reused_only_for_the_same_plan_and_start(module, tmp_path):
    import json

    plan = module.rungs()[:2]
    path = tmp_path / "records.json"
    path.write_text(json.dumps({
        "rungs": [rung["name"] for rung in plan],
        "initialization_fingerprint": "abc",
        "records": [{"rung": plan[0]["name"]}],
    }))

    assert module.load_checkpoint(path, plan, None)["records"] == [{"rung": plan[0]["name"]}]
    assert module.load_checkpoint(path, plan, "abc") is not None
    # Another initial state, or another plan, is not this scan.
    assert module.load_checkpoint(path, plan, "xyz") is None
    assert module.load_checkpoint(path, module.rungs()[:3], None) is None
    assert module.load_checkpoint(tmp_path / "missing.json", plan, None) is None


# --------------------------------------------------------------------------
# The recorded run (2026-09-21).  A re-run that reverses these is a finding.
# --------------------------------------------------------------------------

RECORDED = Path(__file__).resolve().parent / "data" / "efit_sigma_calibration.json"


@pytest.fixture(scope="module")
def recorded():
    import json

    return json.loads(RECORDED.read_text(encoding="utf-8"))


def test_every_recorded_run_cold_started_from_the_same_state(recorded):
    fingerprints = {r["initialization"]["sha256"] for r in recorded["records"]}
    assert fingerprints == {recorded["initialization_fingerprint"]}
    assert len(recorded["records"]) == 1188
    assert recorded["toolchain"]["efit"]["sha256"].startswith("4a4e645e")


def test_no_rung_is_in_the_operating_range_and_the_current_sigma_converges_nowhere(recorded):
    rows = {r["rung"]: r for r in recorded["summary"]}
    assert recorded["operating_range"]["range"] == []
    assert rows["p1f1_probe_x1_loop_x1_floor0pct"]["converged_tight"] == 0
    # The best rungs lose only 41672 @ 342, to findax after the iconvr=2 exit.
    best = max(r["converged_tight"] for r in rows.values())
    assert best == 8
    lost = {
        (x["shot"], x["time_ms"]) for x in recorded["records"]
        if x["rung"] == "p1f1_probe_x16_loop_x16_floor2pct" and x["error_minimum"] == 1e-4 and not x["converged"]
    }
    assert lost == {(41672, 342)}


def test_the_weighted_magnetics_fit_the_data_better_and_land_elsewhere(recorded):
    rows = {r["rung"]: r for r in recorded["summary"]}
    legacy = rows["p1f1_legacy_reference"]
    weighted = rows["p1f1_probe_x16_loop_x16_floor2pct"]
    assert weighted["probe_residual"] < 0.5 * legacy["probe_residual"]
    assert weighted["median_lcfs_to_reference_mm"] > 50.0
