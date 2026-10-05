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
    assert "global_reduced_chi2" in columns and "uncertainty_model" in columns
    assert any(key.startswith("validation.diagnostic_fit.") for key in columns)
    assert any(key.startswith("validation.verification.") for key in columns)
    assert all(value in ("pass", "warn", "fail", "indeterminate", "not_available")
               for key, value in columns.items() if key.startswith("validation."))
