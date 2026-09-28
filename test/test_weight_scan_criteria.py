"""The #891 stage-2 per-slice criteria and the weight scan's settings."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "workflow" / "efit_uncertainty_calibration"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def criteria():
    return _load("weight_criteria", ROOT / "criteria.py")


@pytest.fixture(scope="module")
def scan():
    return _load("weight_scan", ROOT / "weight_scan.py")


def _good_record(**overrides):
    record = {
        "setting": "s", "converged": True, "pressure_min": 0.0, "ip_measured": 80.0e3,
        "scalars": {"betap": 0.2, "wmhd": 150.0, "q95": 6.0, "ipmhd": 80.5e3},
        "fit": {"probe_n": 60, "probe_reduced_chi2": 1.0, "loop_n": 11, "loop_reduced_chi2": 0.8,
                "ip_n": 1, "ip_chi2": 0.7, "ip_reduced_chi2": 0.7, "pf_n": 16, "pf_reduced_chi2": 50.0,
                "dia_n": 1, "dia_chi2": 1.2, "dia_reduced_chi2": 1.2},
        "virial": {"beta_p_integral": 0.2, "beta_p_pair_13": 0.25, "denominator_pair_13": 1.7},
        "gs": {"whole": 0.005},
        "thomson": {"log_ratio": math.log(1 / 2.0)},
    }
    record.update(overrides)
    return record


def test_a_good_slice_passes_every_criterion_and_pf_is_reported_not_graded(criteria):
    result = criteria.evaluate(_good_record())
    assert result["good"]
    assert {v["status"] for v in result["verdicts"].values()} == {"pass"}
    pf = result["verdicts"]["measurement"]["families"]["pf"]
    assert pf["reduced_chi2"] == 50.0 and not pf["graded"]


@pytest.mark.parametrize("change, reason", [
    ({"converged": False}, "not converged"),
    ({"pressure_min": -5.0}, "pressure_min"),
    ({"scalars": {"betap": -0.3, "wmhd": 150.0, "q95": 6.0, "ipmhd": 80.5e3}}, "betap"),
    ({"scalars": {"betap": 0.2, "wmhd": 150.0, "q95": 1.1, "ipmhd": 80.5e3}}, "q95"),
    ({"scalars": {"betap": 0.2, "wmhd": 150.0, "q95": 6.0, "ipmhd": 90.0e3}}, "ip"),
])
def test_the_admissibility_veto_makes_a_slice_bad_whatever_else_passes(criteria, change, reason):
    result = criteria.evaluate(_good_record(**change))
    assert not result["good"]
    assert result["verdicts"]["admissible"]["status"] == "fail"
    assert any(r.startswith(reason) for r in result["verdicts"]["admissible"]["reasons"])


def test_a_family_outside_the_chi2_band_fails_the_measurement_criterion(criteria):
    record = _good_record()
    record["fit"] = dict(record["fit"], loop_reduced_chi2=0.1)
    verdict = criteria.measurement(record)
    assert verdict["status"] == "fail" and verdict["reasons"] == ["loop reduced chi2 0.1"]


def test_an_inactive_diamagnetic_row_is_not_graded(criteria):
    record = _good_record(inactive_families=["dia"])
    record["fit"] = dict(record["fit"], dia_reduced_chi2=1e-12)
    assert criteria.measurement(record)["status"] == "pass"


def test_the_virial_check_is_indeterminate_near_the_pair_13_singularity(criteria):
    record = _good_record(virial={"beta_p_integral": 0.2, "beta_p_pair_13": 0.9, "denominator_pair_13": 0.05})
    assert criteria.virial(record)["status"] == "indeterminate"
    record["virial"]["denominator_pair_13"] = 1.0
    assert criteria.virial(record)["status"] == "fail"  # ln(0.2/0.9) beyond ln 2


@pytest.mark.parametrize("p_e_over_p_recon, status", [(0.5, "pass"), (1.0, "pass"), (0.3, "fail"), (1.5, "fail")])
def test_the_thomson_band_is_p_e_to_three_p_e(criteria, p_e_over_p_recon, status):
    record = _good_record(thomson={"log_ratio": math.log(p_e_over_p_recon)})
    assert criteria.thomson(record)["status"] == status


def test_thomson_is_not_available_off_the_samples_and_never_counts_against(criteria):
    result = criteria.evaluate(_good_record(thomson=None))
    assert result["verdicts"]["thomson"]["status"] == "not_available" and result["good"]


def test_summarize_counts_good_slices_per_setting(criteria):
    records = [_good_record(setting="a"), _good_record(setting="a", pressure_min=-1.0), _good_record(setting="b")]
    rows = {row["setting"]: row for row in criteria.summarize(records)}
    assert rows["a"]["good"] == 1 and rows["a"]["slices"] == 2
    assert rows["a"]["admissible"] == {"pass": 1, "fail": 1}
    assert rows["b"]["good"] == 1


def test_stage1_holds_probe_and_loop_and_scans_basis_by_diamagnetic_sigma(scan):
    settings = scan.stage1_settings()
    assert settings[0] == {"name": "routine", "routine": True}
    scanned = settings[1:]
    assert len(scanned) == len(scan.BASES) * len(scan.DIAMAGNETIC_MULTIPLIERS) == 15
    assert len({s["name"] for s in scanned}) == 15
    assert {(s["probe"], s["loop"], s["floor"]) for s in scanned} == {(16, 2, 0.02)}


def test_uncertainty_scales_divide_sigma_and_an_inactive_row_is_scaled_out(scan):
    active = {"probe": 16, "loop": 2, "dia": 4}
    assert scan.uncertainty_scales(active) == {"bpol_probe": 1 / 16, "flux_loop": 0.5, "plasma_current": 1.0,
                                               "diamagnetic_flux": 0.25}
    off = dict(active, dia=None)
    assert scan.uncertainty_scales(off)["diamagnetic_flux"] == scan.calibration.DIAMAGNETIC_INACTIVE_SCALE
    assert "diamagnetic_flux" in scan.fitted_families(active)
    assert "diamagnetic_flux" not in scan.fitted_families(off)
    assert scan.inactive_families(off) == ("dia",) and scan.inactive_families(active) == ()


def test_stage2_grids_probe_loop_dia_basis_and_both_ip_sigmas(scan):
    settings = scan.stage2_settings()[1:]
    assert len(settings) == 2 * 2 * 3 * 2 * 2 == 48
    assert len({s["name"] for s in settings}) == 48
    two_percent = [s for s in settings if s["ip"] == 0.4]
    assert len(two_percent) == 24 and all("_ip_x0.4_" in s["name"] for s in two_percent)
    # 5 % / 0.4 = 2 %: a multiplier below one narrows the sigma.
    assert scan.uncertainty_scales(two_percent[0])["plasma_current"] == pytest.approx(2.5)


# --- the study rules adopted 2026-09-28 ---------------------------------------

def test_single_channel_families_are_judged_by_z_not_by_a_chi2_band(criteria):
    """One channel's chi-square is one z**2: z = 0.3 is a perfectly good fit that
    the old (0.5, 2) band on chi2r = 0.09 would have failed."""
    record = _good_record()
    record["fit"] = dict(record["fit"], ip_chi2=0.09, ip_reduced_chi2=0.09, dia_chi2=3.9, dia_reduced_chi2=3.9)
    verdict = criteria.measurement(record)
    assert verdict["status"] == "pass"
    assert verdict["families"]["ip"]["z"] == pytest.approx(0.3)
    record["fit"] = dict(record["fit"], dia_chi2=4.41, dia_reduced_chi2=4.41)
    verdict = criteria.measurement(record)
    assert verdict["status"] == "fail" and verdict["reasons"] == ["dia |z| 2.1"]


@pytest.mark.parametrize("ratio, status", [(0.5, "pass"), (0.3, "pass"), (1.03, "pass"), (0.25, "fail"), (1.05, "fail")])
def test_the_ip_ratio_band_admits_a_ramp_up_closed_surface_current_below_the_rogowski(criteria, ratio, status):
    scalars = {"betap": 0.2, "wmhd": 150.0, "q95": 6.0, "ipmhd": ratio * 80.0e3}
    assert criteria.admissible(_good_record(scalars=scalars))["status"] == status


def test_setting_calibration_is_the_median_over_admissible_slices(criteria):
    records = [_good_record(), _good_record(), _good_record(pressure_min=-1.0)]
    records[0]["fit"] = dict(records[0]["fit"], probe_reduced_chi2=0.9, loop_reduced_chi2=1.1)
    records[1]["fit"] = dict(records[1]["fit"], probe_reduced_chi2=1.1, loop_reduced_chi2=1.2)
    records[2]["fit"] = dict(records[2]["fit"], probe_reduced_chi2=50.0, loop_reduced_chi2=50.0)
    result = criteria.setting_calibration(records)
    assert result["over"] == "admissible" and result["slices"] == 2
    assert result["families"]["probe"]["median_reduced_chi2"] == pytest.approx(1.0)
    assert result["calibrated"]
    none_admissible = [_good_record(pressure_min=-1.0)]
    assert criteria.setting_calibration(none_admissible)["over"] == "converged"


def test_next_multipliers_reaches_the_band_on_a_one_over_m_squared_model(scan, criteria):
    true_scale = {"probe": 5.3, "loop": 0.8}   # the multiplier that would give chi2r = 1
    multipliers = dict(scan.STAGE3_START)
    low, high = criteria.CRITERIA["calibration_band"]
    for round_index in range(3):
        medians = {f: (true_scale[f] / m) ** 2 for f, m in multipliers.items()}
        if all(low <= v <= high for v in medians.values()):
            break
        multipliers = scan.next_multipliers(multipliers, medians)
    medians = {f: (true_scale[f] / m) ** 2 for f, m in multipliers.items()}
    assert all(low <= v <= high for v in medians.values()) and round_index <= 2


def test_next_multipliers_clamps_and_keeps_a_family_without_a_median(scan):
    out = scan.next_multipliers({"probe": 60.0, "loop": 1.0}, {"probe": 100.0, "loop": float("nan")})
    assert out == {"probe": 64.0, "loop": 1.0}


def test_stage3_grids_basis_dia_and_ip_and_solves_probe_and_loop(scan):
    cells = scan.stage3_cells()
    assert len(cells) == 3 * 3 * 3 == 27
    ip_sigmas = sorted({0.05 * c["ip"] for c in cells})
    assert ip_sigmas == pytest.approx([0.05, 0.20, 0.50])
    setting = scan.cell_setting(cells[0], scan.STAGE3_START)
    assert setting["probe"] == 4.0 and setting["loop"] == 1.0
    assert scan.uncertainty_scales(setting)["plasma_current"] == pytest.approx(1.0 / cells[0]["ip"])


def test_backoff_steps_halfway_back_in_log(scan):
    assert scan.backoff_multipliers({"probe": 8.0, "loop": 2.0}, {"probe": 12.9, "loop": 3.18}) == {
        "probe": 10.2, "loop": 2.52}


def test_merging_refuses_records_from_different_initial_states(scan):
    same = [{"setting": "a", "fingerprint": "x"}, {"setting": "b", "fingerprint": "x"},
            {"setting": "routine", "fingerprint": "legacy"}]
    assert scan.check_fingerprints(same) == "x"
    with pytest.raises(RuntimeError, match="different initial states"):
        scan.check_fingerprints(same + [{"setting": "c", "fingerprint": "y"}])
