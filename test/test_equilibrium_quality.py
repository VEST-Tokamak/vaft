"""Equilibrium-quality cohorts (#1644): a composed review surface, not a new status model."""

from __future__ import annotations

import math

import pytest

from vaft.validation import equilibrium_quality as eq


@pytest.fixture(scope="module")
def criteria():
    return eq.load_study_criteria()


def _record(shot=1, time_ms=300, setting="s", **overrides):
    record = {
        "setting": setting, "shot": shot, "time_ms": time_ms, "converged": True, "pressure_min": 0.0,
        "ip_measured": 80.0e3,
        "scalars": {"betap": 0.2, "wmhd": 150.0, "q95": 6.0, "ipmhd": 80.5e3},
        "fit": {"probe_n": 60, "probe_reduced_chi2": 1.0, "loop_n": 11, "loop_reduced_chi2": 0.8,
                "ip_n": 1, "ip_chi2": 0.7, "ip_reduced_chi2": 0.7, "pf_n": 16, "pf_reduced_chi2": 50.0,
                "dia_n": 1, "dia_chi2": 1.2, "dia_reduced_chi2": 1.2},
        "virial": {"beta_p_integral": 0.2, "beta_p_pair_13": 0.25, "denominator_pair_13": 1.7},
        "gs": {"whole": 0.005},
        "thomson": {"log_ratio": math.log(1 / 1.5)},
    }
    record.update(overrides)
    return record


def test_a_row_keeps_raw_metric_verdict_and_classification_apart(criteria):
    table = eq.equilibrium_quality_table([_record()], criteria=criteria)
    row = table.iloc[0]
    assert row["quality_label"] == "good" and bool(row["setting_good"]) and row["physically_consistent"] is True
    assert row["admissible_status"] == "pass" and row["measurement_status"] == "pass"
    assert all(row[name] == "pass" for name, _ in eq.ADMISSIBILITY_RULES)
    assert row["beta_p"] == pytest.approx(0.2) and row["p_over_p_e_points"] == pytest.approx(1.5)
    assert row["failure_reasons"] == ""


def test_each_admissibility_sub_rule_is_read_back_from_the_criterion(criteria):
    bad = _record(pressure_min=-3.0, scalars={"betap": 0.2, "wmhd": 150.0, "q95": 1.5, "ipmhd": 80.5e3})
    row = eq.equilibrium_quality_table([bad], criteria=criteria).iloc[0]
    assert row["admissible_status"] == "fail" and row["quality_label"] == "unreconstructible"
    assert row["rule_pressure_nonnegative"] == "fail" and row["rule_q95"] == "fail"
    assert row["rule_convergence"] == "pass" and row["rule_beta_p_positive"] == "pass"
    assert "pressure_min" in row["failure_reasons"]


def test_a_measurement_family_failure_is_attributed_to_its_family(criteria):
    record = _record(fit={**_record()["fit"], "loop_reduced_chi2": 7.0})
    row = eq.equilibrium_quality_table([record], criteria=criteria).iloc[0]
    assert row["measurement_status"] == "fail" and row["rule_loop_fit"] == "fail"
    assert row["rule_probe_fit"] == "pass" and row["quality_label"] == "admissible"


def test_thomson_is_a_separate_dimension_never_part_of_good(criteria):
    record = _record(thomson={"log_ratio": math.log(1 / 2.7)})
    row = eq.equilibrium_quality_table([record], criteria=criteria).iloc[0]
    assert row["quality_label"] == "good" and bool(row["setting_good"])
    assert row["thomson_status"] == "fail" and row["physically_consistent"] is False


def test_a_slice_is_counted_once_with_its_best_setting(criteria):
    records = [
        _record(setting="a", fit={**_record()["fit"], "probe_reduced_chi2": 9.0}),  # admissible only
        _record(setting="b"),                                                         # good
        _record(shot=2, setting="a", converged=False),                                # unreconstructible
        {"setting": "c", "shot": 2, "time_ms": 300, "error": "boom"},                 # an attempt with no output
    ]
    table = eq.equilibrium_quality_table(records, criteria=criteria)
    assert len(table) == 3  # the errored attempt has no row of its own
    slices = eq.slice_cohorts(table)
    assert list(slices["quality_label"]) == ["good", "unreconstructible"]
    assert slices.iloc[0]["setting"] == "b"
    summary = eq.equilibrium_quality_summary(table)
    assert summary["slices"] == 2 and summary["cohorts"]["good"]["slices"] == 1
    assert summary["cohorts"]["unreconstructible"]["slices"] == 1


def test_the_census_reports_status_fractions_and_the_dominant_cause(criteria):
    records = [_record(shot=s, converged=False) for s in (1, 2, 3)] + [_record(shot=4, pressure_min=-1.0)]
    census = eq.equilibrium_quality_failure_census(eq.equilibrium_quality_table(records, criteria=criteria))
    cell = census["matrix"]["rule_convergence"]["unreconstructible"]
    assert cell["n"] == 4 and cell["fail"] == pytest.approx(0.75) and cell["pass"] == pytest.approx(0.25)
    assert census["pareto"]["unreconstructible"][0] == ("rule_convergence", 3)


def test_the_crosswalk_compares_only_where_both_layers_answered(criteria):
    table = eq.equilibrium_quality_table([_record(), _record(shot=2, converged=False)], criteria=criteria)
    absent = {row["generic"]: row for row in eq.equilibrium_quality_crosswalk(table)}
    assert absent["verification.convergence"]["available"] is False
    table["validation.verification.convergence"] = ["pass", "pass"]  # generic says both converged
    walk = {row["generic"]: row for row in eq.equilibrium_quality_crosswalk(table)}
    entry = walk["verification.convergence"]
    assert entry["decided"] == 2 and entry["agreement"] == pytest.approx(0.5)
    assert entry["counts"] == {"pass|pass": 1, "pass|fail": 1}
    table["validation.verification.convergence"] = ["pass", "warn"]  # warn has no study counterpart
    entry = {row["generic"]: row for row in eq.equilibrium_quality_crosswalk(table)}["verification.convergence"]
    assert entry["agreement"] == pytest.approx(0.5) and entry["agreement_warn_as_fail"] == pytest.approx(1.0)


def test_efit_evidence_columns_reuse_the_existing_layers():
    from vaft.omas.sample import sample_ods

    columns = eq.efit_evidence_columns(sample_ods(), 0)
    assert "evidence.global_reduced_chi2" in columns and "evidence.uncertainty_model" in columns
    # evidence never shares a name with a study column (e.g. the record's ip_z)
    assert all(key.startswith(("evidence.", "validation.")) for key in columns)
    assert any(key.startswith("validation.diagnostic_fit.") for key in columns)
    assert any(key.startswith("validation.verification.") for key in columns)
    assert all(value in ("pass", "warn", "fail", "indeterminate", "not_available")
               for key, value in columns.items() if key.startswith("validation."))


def test_routine_rows_are_reported_but_never_a_cohort_member(criteria):
    records = [_record(setting="routine"), _record(setting="a", converged=False),
               _record(shot=2, setting="routine")]
    table = eq.equilibrium_quality_table(records, criteria=criteria)
    assert set(table["setting"]) == {"routine", "a"}
    slices = eq.slice_cohorts(table)
    # the good routine does not lend its passing rules to the unreconstructible slice,
    # and a routine-only slice is not a cohort member at all
    assert list(slices["shot"]) == [1] and slices.iloc[0]["setting"] == "a"
    assert slices.iloc[0]["rule_convergence"] == "fail"


def test_the_best_row_is_taken_whole_not_stitched_from_settings(criteria):
    records = [_record(setting="a", thomson=None),  # good, no Thomson there
               _record(setting="b", thomson={"log_ratio": math.log(1 / 2.7)})]
    slices = eq.slice_cohorts(eq.equilibrium_quality_table(records, criteria=criteria))
    row = slices.iloc[0]
    assert row["setting"] == "a" and row["physically_consistent"] is None
    assert math.isnan(row["p_over_p_e_points"])  # not b's value


def test_the_pareto_never_counts_thomson_as_what_blocks_good(criteria):
    records = [_record(fit={**_record()["fit"], "probe_reduced_chi2": 9.0},
                       thomson={"log_ratio": math.log(1 / 2.7)})]
    census = eq.equilibrium_quality_failure_census(eq.equilibrium_quality_table(records, criteria=criteria))
    assert dict(census["pareto"]["admissible"]) == {"rule_probe_fit": 1}
    assert census["matrix"]["thomson_status"]["admissible"]["fail"] == pytest.approx(1.0)


def test_an_ungraded_family_is_not_available_not_pass(criteria):
    record = _record(fit={**_record()["fit"], "dia_n": 0})
    row = eq.equilibrium_quality_table([record], criteria=criteria).iloc[0]
    assert row["rule_dia_fit"] == "not_available" and row["rule_ip_fit"] == "pass"


def test_product_evidence_goes_only_to_the_setting_that_made_the_product(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(eq.__file__).resolve().parents[2] / "workflow" / "equilibrium_quality" / "build_cohort_table.py"
    spec = importlib.util.spec_from_file_location("build_cohort_table", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = module.ProductEvidence(tmp_path, "statistical_891")
    assert "another" not in evidence({"shot": 1, "time_s": 0.3, "setting": "statistical_891"})["evidence_status"]
    assert evidence({"shot": 1, "time_s": 0.3, "setting": "p1f1"})["evidence_status"].startswith("product is setting")


def test_constraint_points_use_the_efit_quality_tables():
    from vaft.omas.sample import sample_ods

    points = eq.equilibrium_quality_constraint_points(sample_ods(), 0)
    families = {p["family"] for p in points}
    assert {"bpol_probe", "flux_loop"} <= families
    assert all({"measured", "reconstructed", "z", "fitted", "state", "unit"} <= set(p) for p in points)
    probe = [p for p in points if p["family"] == "bpol_probe"]
    assert probe[0]["unit"] == "mT"


def test_representative_cases_are_deterministic_and_say_why(criteria):
    probe = lambda chi2: {**_record()["fit"], "probe_reduced_chi2": chi2}
    records = [
        _record(shot=1, fit=probe(1.0)), _record(shot=2, fit=probe(1.1)), _record(shot=3, fit=probe(1.9)),
        _record(shot=4, thomson={"log_ratio": math.log(1 / 2.7)}),             # good, inconsistent
        _record(shot=5, fit=probe(6.0)), _record(shot=6, fit=probe(30.0)),      # admissible-only: probe fit
        _record(shot=7, converged=False), _record(shot=8, converged=False),     # unreconstructible
    ]
    table = eq.equilibrium_quality_table(records, criteria=criteria)
    first = eq.select_representative_cases(table)
    assert [c["case"] for c in first] == list(eq.REPRESENTATIVE_CASES)
    assert first == eq.select_representative_cases(table.sample(frac=1.0, random_state=3))  # order-independent
    cases = {c["case"]: c for c in first}
    assert cases["good_consistent"]["shot"] == 2          # the median probe chi2 of 1.0 / 1.1 / 1.9
    assert cases["good_inconsistent"]["shot"] == 4
    assert cases["admissible_only"]["shot"] in (5, 6) and "rule_probe_fit" in cases["admissible_only"]["reason"]
    assert cases["unreconstructible"]["shot"] == 7 and "rule_convergence" in cases["unreconstructible"]["reason"]


def test_a_case_without_candidates_says_so(criteria):
    cases = {c["case"]: c for c in eq.select_representative_cases(
        eq.equilibrium_quality_table([_record()], criteria=criteria))}
    assert cases["good_inconsistent"]["shot"] is None and "no candidate" in cases["good_inconsistent"]["reason"]


def test_the_confinement_funnel_counts_with_the_confinement_module():
    import pandas as pd

    table = pd.DataFrame({
        "efit_quality": ["good", "good", "good", "admissible", "admissible"],
        "rule_finite": ["True", "True", "False", "True", "True"],
        "rule_ip_min": [True, True, True, True, False],
        "rule_dwdt_fraction": [True, False, True, True, True],
        "rule_ip_change_per_tau": [True, True, True, True, True],
        "accepted": [True, False, False, True, False],
    })
    funnel = eq.equilibrium_quality_confinement_funnel(table)
    good = funnel["cohorts"]["good"]
    assert (good["candidates"], good["selected"], good["rejected"]) == (3, 1, 2)
    removed = {row["rule"]: row["removed_in_sequence"] for row in good["exclusions"]}
    assert removed == {"finite": 1, "ip_min": 0, "dwdt_fraction": 1, "ip_change_per_tau": 0}
    assert funnel["cohorts"]["admissible"]["selected"] == 1
