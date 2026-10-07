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


@pytest.mark.parametrize("p_over_p_e, status", [(1.0, "pass"), (1.5, "pass"), (2.0, "pass"),
                                               (2.5, "fail"), (3.0, "fail"), (0.8, "fail")])
def test_the_thomson_band_is_p_e_to_two_p_e(criteria, p_over_p_e, status):
    """No fast ions, T_i <= T_e and n_i <= n_e: 1 <= p/p_e <= 2 (criteria v2)."""
    record = _good_record(thomson={"log_ratio": math.log(1.0 / p_over_p_e)})
    assert criteria.thomson(record)["status"] == status


def test_a_thomson_fail_is_reported_beside_good_never_inside_it(criteria):
    """A numerically sound fit inconsistent with Thomson stays good, flagged inconsistent."""
    result = criteria.evaluate(_good_record(thomson={"log_ratio": math.log(1.0 / 2.7)}))
    assert result["verdicts"]["thomson"]["status"] == "fail"
    assert result["good"] is True and result["physically_consistent"] is False
    assert result["criteria_version"] == 2
    consistent = criteria.evaluate(_good_record(thomson={"log_ratio": math.log(1.0 / 1.8)}))
    assert consistent["good"] is True and consistent["physically_consistent"] is True
    assert "thomson" not in criteria.FIT_QUALITY and criteria.PHYSICAL_CONSISTENCY == ("thomson",)


def test_slice_labels_keep_the_fit_label_and_flag_consistency_separately(criteria):
    records = [
        {**_good_record(thomson={"log_ratio": math.log(1.0 / 2.7)}), "setting": "a", "shot": 1, "time_ms": 300},
        {**_good_record(thomson={"log_ratio": math.log(1.0 / 1.5)}), "setting": "b", "shot": 1, "time_ms": 300},
        {**_good_record(thomson={"log_ratio": math.log(1.0 / 2.7)}), "setting": "a", "shot": 2, "time_ms": 300},
        {**_good_record(thomson=None), "setting": "a", "shot": 3, "time_ms": 300},
    ]
    labels = {row["shot"]: row for row in criteria.slice_labels(records)}
    assert [labels[s]["label"] for s in (1, 2, 3)] == ["good", "good", "good"]
    assert labels[1]["consistent"] == ["b"] and labels[1]["inconsistent"] == ["a"]
    assert labels[2]["consistent"] == [] and labels[2]["inconsistent"] == ["a"]
    assert labels[3]["consistent"] == [] and labels[3]["inconsistent"] == []


def test_thomson_is_not_available_off_the_samples_and_never_counts_against(criteria):
    result = criteria.evaluate(_good_record(thomson=None))
    assert result["verdicts"]["thomson"]["status"] == "not_available" and result["good"]
    assert result["physically_consistent"] is None


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


@pytest.mark.parametrize("ratio, sigma, status", [
    (0.5, 0.05, "pass"), (0.3, 0.05, "pass"), (0.25, 0.05, "fail"),
    (1.09, 0.05, "pass"), (1.11, 0.05, "fail"),  # upper edge 1 + 2 sigma
    (1.39, 0.20, "pass"), (1.41, 0.20, "fail"),
])
def test_the_ip_ratio_band_is_0p3_to_one_plus_two_sigma(criteria, ratio, sigma, status):
    scalars = {"betap": 0.2, "wmhd": 150.0, "q95": 6.0, "ipmhd": ratio * 80.0e3}
    record = _good_record(scalars=scalars, ip_sigma_relative=sigma)
    assert criteria.admissible(record)["status"] == status


def test_ip_sigma_comes_from_the_fit_and_the_routine_uses_the_reference(criteria):
    record = _good_record()
    record["fit"] = dict(record["fit"], ip_sigma_median=16.0e3)  # 20 % of 80 kA
    assert criteria.ip_sigma_relative(record) == pytest.approx(0.20)
    # the routine's legacy weight is not a sigma, however large
    routine = dict(record, setting="routine")
    assert criteria.ip_sigma_relative(routine) == pytest.approx(0.05)
    assert criteria.ip_ratio_band(routine) == pytest.approx((0.3, 1.10))


def test_slice_labels_name_the_unreconstructible_slices(criteria):
    good = dict(_good_record(), shot=1, time_ms=10)
    negative = dict(_good_record(pressure_min=-5.0), shot=1, time_ms=11)
    routine = dict(_good_record(), shot=1, time_ms=11, setting="routine")
    thomson_fail = dict(_good_record(thomson={"log_ratio": -2.0}), shot=1, time_ms=12)
    labels = criteria.slice_labels([good, negative, routine, thomson_fail])
    # criteria v2: a Thomson fail leaves the fit label alone and is flagged beside it
    assert [(x["time_ms"], x["label"]) for x in labels] == [
        (10, "good"), (11, "unreconstructible"), (12, "good")]
    assert labels[2]["inconsistent"] == ["s"] and labels[0]["consistent"] == ["s"]
    # the routine is reported, never counted towards the label
    assert labels[1]["routine"] == "pass"


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
    steps = 0
    while True:
        medians = {f: (true_scale[f] / m) ** 2 for f, m in multipliers.items()}
        if all(low <= v <= high for v in medians.values()):
            break
        multipliers = scan.next_multipliers(multipliers, medians)
        steps += 1
        assert steps <= 3, "did not reach the band in three steps"


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


def test_stage4_exits_on_psi_alone_over_five_bases(scan):
    cells = scan.stage4_cells()
    assert [c["basis"] for c in cells] == [[1, 1], [2, 1], [1, 2], [1, 3], [2, 2]]
    assert {(c["dia"], c["ip"]) for c in cells} == {(16, 4.0)}
    assert [c["basis"] for c in scan.stage4_cells([(1, 3)])] == [[1, 3]]
    setting = scan.cell_setting(cells[0], scan.STAGE4_START)
    assert setting["name"].endswith("_psiexit") and setting["psi_exit"]
    assert scan.exit_chi_squared(setting, 60) == scan.PSI_EXIT_SAICON
    stage3 = scan.cell_setting(scan.stage3_cells()[0], scan.STAGE3_START)
    assert "psi_exit" not in stage3
    assert scan.exit_chi_squared(stage3, 60) == pytest.approx(60 + 3 * (120 ** 0.5))


def test_stage5_is_the_working_setting_at_two_tolerances(scan):
    settings = scan.working_settings()
    assert settings[0]["routine"]
    study = settings[1:]
    assert [s["error_minimum"] for s in study] == [1.0e-4, 1.0e-3]
    assert len({s["name"] for s in study}) == 2
    assert all(s["basis"] == [2, 1] and s["psi_exit"] for s in study)
    assert scan.uncertainty_scales(study[0])["plasma_current"] == pytest.approx(0.25)


def _calibration(probe, loop, calibrated=False):
    return {"calibrated": calibrated, "families": {"probe": {"median_reduced_chi2": probe},
                                                   "loop": {"median_reduced_chi2": loop}}}


def test_a_repeated_backoff_steps_towards_the_last_round_that_converged(scan):
    nan = float("nan")
    s = {"multipliers": {"probe": 8.0, "loop": 2.0}, "done": False, "history": []}
    scan.advance_cell(s, 0, "a", _calibration(2.6, 2.5))          # converged -> step out
    stepped = dict(s["multipliers"])
    scan.advance_cell(s, 1, "b", _calibration(nan, nan))          # nothing converged -> back off
    first = dict(s["multipliers"])
    assert first == scan.backoff_multipliers({"probe": 8.0, "loop": 2.0}, stepped)
    scan.advance_cell(s, 2, "c", _calibration(nan, nan))          # again: towards round 0, not round 1
    assert s["multipliers"] == scan.backoff_multipliers({"probe": 8.0, "loop": 2.0}, first)
    assert s["multipliers"]["probe"] < first["probe"] and not s["done"]


def test_cell_status_and_the_multipliers_actually_run(scan):
    nan = float("nan")
    never = {"multipliers": {"probe": 8.0, "loop": 2.0}, "done": False, "history": []}
    scan.advance_cell(never, 0, "a", _calibration(nan, nan))
    assert never["done"] and never["status"] == "no_converged_slice"
    clamped = {"multipliers": {"probe": 64.0, "loop": 2.0}, "done": False, "history": []}
    scan.advance_cell(clamped, 0, "a", _calibration(9.0, 1.0))    # probe wants > 64: clamped, no move
    assert clamped["done"] and clamped["status"] == "stalled"
    running = {"multipliers": {"probe": 4.0, "loop": 1.0}, "done": False, "history": []}
    scan.advance_cell(running, 0, "a", _calibration(4.0, 1.0))
    scan.finish_cell(running)
    assert running["status"] == "rounds_exhausted"
    assert running["multipliers"] == {"probe": 4.0, "loop": 1.0}   # what was run, not the next step
    assert running["proposed_multipliers"] == {"probe": 8.0, "loop": 1.0}


def test_a_slice_without_a_fit_is_not_good(criteria):
    record = _good_record(fit=None)
    result = criteria.evaluate(record)
    assert result["verdicts"]["measurement"]["status"] == "not_available" and not result["good"]


def test_backoff_steps_halfway_back_in_log(scan):
    assert scan.backoff_multipliers({"probe": 8.0, "loop": 2.0}, {"probe": 12.9, "loop": 3.18}) == {
        "probe": 10.2, "loop": 2.52}


def test_merging_refuses_records_from_different_initial_states(scan):
    same = [{"setting": "a", "fingerprint": "x"}, {"setting": "b", "fingerprint": "x"},
            {"setting": "routine", "fingerprint": "legacy"}]
    assert scan.check_fingerprints(same) == "x"
    with pytest.raises(RuntimeError, match="different initial states"):
        scan.check_fingerprints(same + [{"setting": "c", "fingerprint": "y"}])


def test_the_579_ensemble_is_the_full_grid_around_the_working_setting(scan):
    settings = scan.sensitivity_settings()
    assert len(settings) == 120 == len({s["name"] for s in settings})
    assert {tuple(s["basis"]) for s in settings} == set(scan.SENSITIVITY_BASES)
    assert {s["dia"] for s in settings} == {None, 4, 16, 64}
    working = [s for s in settings if s["basis"] == [2, 1] and s["dia"] == 16
               and (s["probe"], s["loop"]) == (3.62, 2.15)]
    assert len(working) == 1 and working[0]["psi_exit"] and working[0]["ip"] == 4.0
    # every setting writes uncertainty scales the writer accepts
    assert all(scan.uncertainty_scales(s)["bpol_probe"] > 0 for s in settings)


def test_a_products_dir_extends_the_reference_set(scan, tmp_path, monkeypatch):
    (tmp_path / "42962.json.gz").write_bytes(b"")
    monkeypatch.setattr(scan, "_CONTEXT", {})
    monkeypatch.setitem(scan._INPUTS, "products_dir", tmp_path)
    monkeypatch.setitem(scan._INPUTS, "thomson_root", tmp_path)
    ctx = scan._context(None)
    assert ctx["products"][42962] == tmp_path / "42962.json.gz"
    assert 39915 in ctx["products"] and ctx["thomson_root"] == tmp_path


@pytest.mark.parametrize("stage", ["3", "4"])
def test_the_settings_refusal_names_every_stage_that_takes_them(scan, tmp_path, capsys, stage):
    # --stage 6 (#1663) takes --settings like 1, 2 and 5; the refusal text must
    # agree with the --settings help and the dispatch (cold review 0.8.0 delta-absorb-17 F5).
    settings = tmp_path / "settings.json"
    settings.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        scan.main(["--output", str(tmp_path / "out"), "--table", str(tmp_path / "table"),
                   "--stage", stage, "--settings", str(settings)])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--settings applies to stages 1, 2, 5 and 6" in err
    assert "stages 3 and 4 solve their own" in err
