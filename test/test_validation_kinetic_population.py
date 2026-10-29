"""Population V&V of the kinetic-state atlas: input contract and evidence tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaft.validation import kinetic_population as kp

SNAPSHOT = Path(__file__).resolve().parents[1] / "notebooks/data/verification_and_validation"


def _tables():
    """Three states: a matched magnetics slice, an unmatched magnetics slice, a matched kinetic slice."""
    state = pd.DataFrame({
        "contract_version": ["1", "1", "1"],
        "shot": [40000, 40000, 40001],
        "time_efit_s": [0.3201, 0.3221, 0.3201],
        "efit_lineage": ["magnetics", "magnetics", "electron_kinetic"],
        "efit_quality": ["good", "admissible", "good"],
        "efit_product": ["m/40000.json.gz", "m/40000.json.gz", "k/40001.json.gz"],
        "efit_product_sha256": ["aa", "aa", "bb"],
        "ts_status": ["matched", "unmatched", "matched"],
        "ts_reason": ["", "the nearest Thomson time is 2 ms away", ""],
        "dt_ts_efit_s": [0.0002, 0.002, -0.0001],
        "ts_tolerance_s": [0.001, 0.001, 0.001],
        "r_sum": [2.0, np.nan, 1.5], "r_w": [3.0, np.nan, 1.8], "r_w_full": [3.2, np.nan, 1.9],
        "r_w_reason": ["", "", ""],
        "virial_status": ["pass", "pass", np.nan],
    })
    profiles = pd.DataFrame({
        "contract_version": ["1"] * 4,
        "shot": [40000, 40000, 40001, 40001],
        "time_efit_s": [0.3201, 0.3201, 0.3201, 0.3201],
        "efit_lineage": ["magnetics", "magnetics", "electron_kinetic", "electron_kinetic"],
        "channel": [0, 1, 0, 1],
        "p_e_pa": [100.0, 80.0, 100.0, 90.0], "p_efit_pa": [200.0, 160.0, 150.0, 135.0], "r_p": [2.0, 2.0, 1.5, 1.5],
    })
    virial = pd.DataFrame({
        "shot": [40000, 40000, 40001], "time_efit_s": [0.3201, 0.3221, 0.3201],
        "efit_lineage": ["magnetics", "magnetics", "electron_kinetic"],
        "efit_product": ["m/40000.json.gz", "m/40000.json.gz", "k/40001.json.gz"],
        "efit_product_sha256": ["aa", "aa", "bb"],
        "e1_normalized": [-0.003, -0.002, -0.001], "e2_normalized": [-0.002, -0.001, -0.001],
        "e3_normalized": [-0.001, 0.0, 0.0], "rms": [0.002, 0.0013, 0.0008], "identities_available": [3, 3, 3],
        "beta_p_volume": [0.27, 0.25, 0.16], "beta_p_pair_13": [0.27, 0.26, 0.16],
        "li_volume": [0.61, 0.60, 0.67], "li_pair_13": [0.62, 0.60, 0.67],
    })
    return state, profiles, virial


def test_a_clean_population_joins_one_to_one_with_no_issues():
    state, profiles, virial = _tables()
    pop = kp.assemble_kinetic_population(state, profiles, virial, manifest={"contract_version": "1",
                                                                           "rows": {"state": 3, "profiles": 4}})
    assert pop.issues.empty and not pop.failed
    assert pop.population.virial_join.tolist() == ["joined"] * 3
    assert pop.population.validation_role.tolist() == ["held_out_thomson", "held_out_thomson",
                                                       "thomson_constrained_consistency"]
    assert len(pop.profiles) == 4


def test_optional_conditioning_is_unavailable_for_old_exports_and_masked_on_bad_products():
    state, profiles, virial = _tables()
    old = kp.virial_evidence(kp.assemble_kinetic_population(state, profiles, virial))
    assert set(old.loc[old.family == "conditioning", "status"]) == {"unavailable"}
    for column in kp.CONDITIONING_QUANTITIES:
        virial[column] = 0.75
    virial["rt_over_r0_available"] = [True, False, True]
    virial.loc[0, "efit_product_sha256"] = "bad"
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    evidence = kp.virial_evidence(pop)
    conditioning = evidence.loc[evidence.family == "conditioning"]
    bad = conditioning.loc[(conditioning.shot == 40000) & (conditioning.time_efit_s == 0.3201)]
    assert set(bad.status) == {"failed"} and bad.value.isna().all()
    rt = conditioning.loc[(conditioning.quantity == "rt_over_r0_available") & (conditioning.time_efit_s == 0.3221)]
    assert rt.iloc[0].status == "available" and rt.iloc[0].value == 0
    summary = kp.population_evidence_summary(pop)
    alpha = summary.loc[(summary.quantity == "alpha") & (summary.efit_lineage == "magnetics")].iloc[0]
    assert (alpha.states, alpha.available, alpha.failed) == (2, 1, 1)


def test_time_keys_join_by_rounded_value_not_by_position():
    state, profiles, virial = _tables()
    virial = virial.iloc[::-1].reset_index(drop=True)  # shuffled
    virial["time_efit_s"] = virial["time_efit_s"] + 2e-6  # float noise below the 1e-4 s key
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    row = pop.population.set_index(list(kp.STATE_KEY)).loc[(40000, 0.3221, "magnetics")]
    assert row.virial_join == "joined" and row.beta_p_volume == 0.25


@pytest.mark.parametrize("table", ["state", "virial"])
def test_duplicate_state_keys_break_the_contract(table):
    tables = dict(zip(("state", "profiles", "virial"), _tables()))
    tables[table] = pd.concat([tables[table], tables[table].iloc[:1]], ignore_index=True)
    with pytest.raises(kp.PopulationContractError, match="duplicate"):
        kp.assemble_kinetic_population(tables["state"], tables["profiles"], tables["virial"])


def test_a_missing_required_column_or_unknown_lineage_breaks_the_contract():
    state, profiles, virial = _tables()
    with pytest.raises(kp.PopulationContractError, match="r_w_full"):
        kp.assemble_kinetic_population(state.drop(columns="r_w_full"), profiles, virial)
    state.loc[0, "efit_lineage"] = "kinetic"
    with pytest.raises(kp.PopulationContractError, match="lineage"):
        kp.assemble_kinetic_population(state, profiles, virial)


def test_a_product_hash_mismatch_is_a_failure_and_its_virial_values_are_not_used():
    state, profiles, virial = _tables()
    virial.loc[0, "efit_product_sha256"] = "cc"
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    assert pop.failed
    assert pop.issues.loc[pop.issues.check == "efit_product_identity", "shot"].tolist() == [40000]
    first = pop.population.iloc[0]
    assert first.virial_join == "product_mismatch" and np.isnan(first.beta_p_volume)
    ev = kp.virial_evidence(pop)
    assert set(ev.loc[(ev.shot == 40000) & (ev.time_efit_s == 0.3201), "status"]) == {"failed"}
    with pytest.raises(kp.PopulationContractError, match="efit_product_identity"):
        kp.assemble_kinetic_population(state, profiles, virial, strict=True)


def test_time_mismatch_missing_channels_and_stray_profiles_are_recorded():
    state, profiles, virial = _tables()
    state.loc[2, "dt_ts_efit_s"] = 0.004  # matched beyond its tolerance
    profiles = profiles.loc[profiles.shot == 40000].copy()  # the kinetic state loses its channels
    stray = profiles.iloc[:1].assign(time_efit_s=0.3221)  # a channel for the unmatched state
    ghost = profiles.iloc[:1].assign(shot=49999)  # a channel for no state at all
    profiles = pd.concat([profiles, stray, ghost], ignore_index=True)
    profiles.loc[1, "p_e_pa"] = np.nan
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    checks = dict(zip(pop.issues.check, pop.issues.status))
    assert checks == {"profile_of_unmatched_state": "fail", "profile_key_in_state": "fail",
                      "missing_channels": "fail", "channel_values": "not_available", "ts_time_tolerance": "fail"}
    assert len(pop.profiles) == 2  # only channels of matched states remain
    pressure = kp.pressure_evidence(pop)
    kinetic = pressure.loc[pressure.efit_lineage == "electron_kinetic"]
    assert set(kinetic.status) == {"failed"} and "tolerance" in kinetic.reason.iloc[0]


def test_unavailable_virial_values_are_explicit_not_dropped():
    state, profiles, virial = _tables()
    virial.loc[1, "li_pair_13"] = np.nan
    virial = virial.drop(index=2)
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    assert pop.population.virial_join.tolist() == ["joined", "joined", "missing"]
    assert {"virial_values", "virial_join"} <= set(pop.issues.check) and not pop.failed
    ev = kp.virial_evidence(pop)
    li = ev.loc[(ev.quantity == "li_pair_13")].set_index("shot").status
    assert li.loc[40001] == "unavailable"
    none = kp.assemble_kinetic_population(state, profiles, None)
    assert set(none.population.virial_join) == {"not_supplied"}
    assert set(kp.virial_evidence(none).status) == {"unavailable"}


def test_pressure_support_and_roles_stay_separate_and_unmatched_states_are_counted():
    state, profiles, virial = _tables()
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    pressure = kp.pressure_evidence(pop)
    assert set(pressure.quantity) == set(kp.PRESSURE_RATIOS) and pressure.support.nunique() == 3
    unmatched = pressure.loc[pressure.time_efit_s == 0.3221]
    assert set(unmatched.status) == {"unavailable"} and "2 ms" in unmatched.reason.iloc[0]
    summary = kp.population_evidence_summary(pop)
    r_w = summary.loc[(summary.quantity == "r_w") & (summary.efit_lineage == "magnetics")].iloc[0]
    assert (r_w.states, r_w.available, r_w.unavailable, r_w.failed) == (2, 1, 1, 0)
    assert r_w.validation_role == "held_out_thomson" and r_w["median"] == 3.0
    families = set(summary.family)
    assert families == {"pressure", "identity", "volume_integral", "pair_13_closure", "conditioning"}


def test_the_study_verdict_rides_along_and_is_never_recomputed():
    state, profiles, virial = _tables()
    state.loc[0, "virial_status"] = "fail"
    ev = kp.virial_evidence(kp.assemble_kinetic_population(state, profiles, virial))
    first = ev.loc[(ev.shot == 40000) & (ev.time_efit_s == 0.3201)]
    assert set(first.virial_status) == {"fail"}
    assert set(first.loc[first.family != "conditioning", "status"]) == {"available"}
    assert set(ev.loc[ev.efit_lineage == "electron_kinetic", "virial_status"]) == {"not_available"}


def test_manifest_counts_contract_version_and_schema_units_are_checked():
    state, profiles, virial = _tables()
    schema = {"properties": {**{c: {"description": ""} for c in kp.STATE_COLUMNS},
                             "dt_ts_efit_s": {"description": "t_TS - t_EFIT [ms]"}}}
    schema["properties"].pop("r_w_full")
    pop = kp.assemble_kinetic_population(
        state, profiles, virial, manifest={"contract_version": "2", "rows": {"state": 4, "profiles": 4}},
        schemas={"state": schema})
    found = pop.issues.loc[pop.issues.scope == "population"]
    assert found.shot.isna().all()
    assert set(found.check) == {"manifest_rows", "contract_version", "schema_declares_column", "unit"}
    assert "[ms]" in found.loc[found.check == "unit", "detail"].iloc[0]


def test_unknown_lineage_has_no_role():
    with pytest.raises(ValueError):
        kp.validation_role("ion_kinetic")


@pytest.mark.skipif(not (SNAPSHOT / "v1/state.csv").is_file(), reason="Tier A snapshot arrives with #1836")
def test_tier_a_snapshot_assembles_cleanly_and_reproduces_the_notebook_counts():
    pop = kp.read_kinetic_population(SNAPSHOT)
    manifest = json.loads((SNAPSHOT / "v1/MANIFEST.json").read_text(encoding="utf-8"))
    assert not pop.failed and len(pop.population) == manifest["rows"]["state"]
    summary = kp.population_evidence_summary(pop)
    r_sum = summary.loc[(summary.quantity == "r_sum")].set_index("efit_lineage")
    state = pd.read_csv(SNAPSHOT / "v1/state.csv")
    matched = state.loc[state.ts_status == "matched"]
    for lineage, group in matched.groupby("efit_lineage"):
        assert r_sum.loc[lineage, "available"] == group.r_sum.notna().sum()
        assert r_sum.loc[lineage, "median"] == pytest.approx(group.r_sum.median())
    assert (summary.states == summary.available + summary.unavailable + summary.failed).all()


def test_absent_hashes_on_both_sides_are_not_an_identity():
    state, profiles, virial = _tables()
    state["efit_product_sha256"] = np.nan
    virial["efit_product_sha256"] = np.nan
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    assert set(pop.population.virial_join) == {"product_mismatch"} and pop.failed


def test_a_matched_state_without_channels_is_not_available_pressure_evidence():
    state, profiles, virial = _tables()
    profiles = profiles.loc[profiles.shot == 40000]  # the kinetic state keeps its ratios but loses its channels
    pressure = kp.pressure_evidence(kp.assemble_kinetic_population(state, profiles, virial))
    kinetic = pressure.loc[pressure.efit_lineage == "electron_kinetic"]
    assert set(kinetic.status) == {"failed"} and kinetic.value.isna().all()
    assert "without profile channels" in kinetic.reason.iloc[0]


def test_a_ratio_on_an_unmatched_state_is_flagged_and_never_reported():
    state, profiles, virial = _tables()
    state.loc[1, "r_sum"] = 7.0
    state.loc[1, "ts_reason"] = np.nan  # an empty CSV cell: the reason falls back to the status, never "nan"
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    assert pop.issues.loc[pop.issues.check == "ratio_without_match", "status"].tolist() == ["fail"]
    unmatched = kp.pressure_evidence(pop).query("time_efit_s == 0.3221")
    assert unmatched.value.isna().all() and set(unmatched.reason) == {"ts_status unmatched"}


def test_population_level_issues_survive_a_scope_split_and_the_strict_message_names_keys():
    state, profiles, virial = _tables()
    virial.loc[0, "efit_product_sha256"] = "cc"
    pop = kp.assemble_kinetic_population(state, profiles, virial, manifest={"rows": {"state": 9}})
    assert set(pop.issues.scope) == {"state", "population"}
    with pytest.raises(kp.PopulationContractError, match=r"\(40000, 0.3201, 'magnetics'\)"):
        kp.assemble_kinetic_population(state, profiles, virial, strict=True)


def test_bad_key_values_and_non_numeric_virial_cells_are_contract_findings_not_crashes():
    state, profiles, virial = _tables()
    broken = state.copy()
    broken.loc[0, "shot"] = np.nan
    with pytest.raises(kp.PopulationContractError, match="shot"):
        kp.assemble_kinetic_population(broken, profiles, virial)
    virial["identities_available"] = virial["identities_available"].astype(object)
    virial.loc[0, "identities_available"] = "three"
    virial["rms"] = virial["rms"].astype(object)
    virial.loc[0, "rms"] = "n/a"
    pop = kp.assemble_kinetic_population(state, profiles, virial)
    assert "rms" in pop.issues.loc[pop.issues.check == "virial_values", "detail"].iloc[0]
