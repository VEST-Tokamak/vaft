"""The neoclassical summary's bootstrap <-> NEO core_transport correspondence (#1655).

Synthetic products for every match status, and the real 48224 NEO product built by
``build_neoclassical_ods`` from the recorded fixture run. No solver runs.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omas import ODS

from vaft import database
from vaft.database import _summary

ROOT = Path(__file__).resolve().parents[1]
PROFILE_RUN = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_profile"
SAMPLE = ROOT / "vaft" / "data" / "kineticEfit" / "ods_48224_300ms.json"


def _product(time=0.3, flux_time=0.3, version="b493397") -> ODS:
    ods = ODS(consistency_check=False)
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.array([time])
    ods["core_profiles.code.name"] = "NEO"
    ods["core_profiles.code.version"] = version
    ods["core_profiles.profiles_1d.0.time"] = time
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = np.linspace(0.0, 1.0, 11)
    ods["core_profiles.profiles_1d.0.j_bootstrap"] = np.where(
        (np.linspace(0, 1, 11) >= 0.3) & (np.linspace(0, 1, 11) <= 0.8), 1.0e4, np.nan)
    ods["core_transport.code.name"] = "NEO"
    ods["core_transport.code.version"] = version
    model = "core_transport.model.0"
    ods[f"{model}.identifier.index"] = 5
    ods[f"{model}.identifier.name"] = "neoclassical"
    base = f"{model}.profiles_1d.0"
    ods[f"{base}.time"] = flux_time
    ods[f"{base}.grid_flux.rho_tor_norm"] = np.array([0.3, 0.5, 0.8])
    ods[f"{base}.electrons.energy.flux"] = np.array([1.0, -3.0, 2.0])
    ods[f"{base}.electrons.particles.flux"] = np.array([1e18, 2e18, 3e18])
    ods[f"{base}.ion.0.energy.flux"] = np.array([10.0, 20.0, 5.0])
    ods[f"{base}.ion.1.energy.flux"] = np.array([1.0, 1.0, 1.0])
    return ods


def _row(ods):
    (row,) = _summary.extract_neoclassical(ods, 1)
    return row


def test_a_single_neo_slice_at_the_same_time_is_matched():
    row = _row(_product())
    assert row["transport_match_status"] == "matched"
    assert (row["transport_model_index"], row["transport_profile_index"]) == (0, 0)
    assert row["transport_code_version"] == "b493397"
    assert row["q_e_peak_abs_W_m2"] == 3.0
    assert row["q_i_peak_abs_W_m2"] == 21.0          # all ions present, summed per point
    assert row["gamma_e_peak_abs_m2_s"] == 3e18
    assert (row["transport_rho_grid_min"], row["transport_rho_grid_max"]) == (0.3, 0.8)
    assert row["transport_ion_count"] == 2


def test_bootstrap_columns_are_unchanged():
    row = _row(_product())
    assert row["points_solved"] == 6 and row["rho_solved_min"] == pytest.approx(0.3)
    assert row["has_core_transport"] is True
    columns = _summary.NEOCLASSICAL_COLUMNS
    assert columns[:columns.index("has_core_transport") + 1] == (
        "shot", "cp_index", "time_s", "i_bootstrap_kA", "j_bootstrap_peak_A_m2", "rho_at_peak",
        "rho_solved_min", "rho_solved_max", "points_solved", "b0_T", "has_core_transport")
    assert database.get_summary_preset("neoclassical").key_columns == ("shot", "cp_index")


def _unmatched(row, status):
    assert row["transport_match_status"] == status
    assert row["q_e_points"] == row["q_i_points"] == row["gamma_e_points"] == 0
    assert np.isnan(row["q_e_peak_abs_W_m2"]) and np.isnan(row["transport_model_index"])
    assert row["points_solved"] == 6           # the bootstrap row stays usable


def test_no_transport_keeps_a_usable_bootstrap_row():
    ods = _product()
    del ods["core_transport"]
    _unmatched(_row(ods), "no_neoclassical_model")


def test_a_time_mismatch_is_not_bridged_to_the_nearest_slice():
    _unmatched(_row(_product(flux_time=0.3005)), "no_slice_at_time")


def test_two_neoclassical_models_are_ambiguous_not_first():
    ods = _product()
    ods["core_transport.model.1"] = copy.deepcopy(ods["core_transport.model.0"])
    _unmatched(_row(ods), "ambiguous_models")


def test_two_slices_at_one_time_are_ambiguous():
    ods = _product()
    ods["core_transport.model.0.profiles_1d.1"] = copy.deepcopy(ods["core_transport.model.0.profiles_1d.0"])
    _unmatched(_row(ods), "ambiguous_slices")


def test_another_producer_or_revision_is_not_a_match():
    ods = _product()
    ods["core_transport.model.0.code.name"] = "NCLASS"
    _unmatched(_row(ods), "producer_mismatch")
    ods = _product()
    ods["core_transport.code.version"] = "other"
    _unmatched(_row(ods), "producer_mismatch")
    ods = _product()
    ods["core_profiles.code.name"] = "VAFT kinetic mapper"
    _unmatched(_row(ods), "producer_mismatch")


def test_turbulent_models_are_not_neoclassical():
    ods = _product()
    ods["core_transport.model.0.identifier.index"] = 6
    _unmatched(_row(ods), "no_neoclassical_model")


def test_partial_coverage_and_missing_channels_stay_distinct():
    ods = _product()
    base = "core_transport.model.0.profiles_1d.0"
    ods[f"{base}.electrons.energy.flux"] = np.array([0.0, np.nan, 0.0])
    del ods[f"{base}.ion.1.energy.flux"]
    del ods[f"{base}.electrons.particles.flux"]
    row = _row(ods)
    assert row["transport_match_status"] == "matched"
    assert row["q_e_points"] == 2 and row["q_e_peak_abs_W_m2"] == 0.0     # physical zero
    assert row["rho_q_e_min"] == 0.3 and row["rho_q_e_max"] == 0.8
    assert row["q_i_points"] == 0 and np.isnan(row["q_i_peak_abs_W_m2"])   # an ion missing
    assert row["gamma_e_points"] == 0 and np.isnan(row["gamma_e_peak_abs_m2_s"])


def test_the_slice_time_wins_over_a_misaligned_time_base():
    ods = _product(time=0.3)
    ods["core_profiles.time"] = np.array([0.5])       # a stale homogeneous base
    row = _row(ods)
    assert row["time_s"] == 0.5                        # the reported column is unchanged
    assert row["transport_match_status"] == "matched"  # the match uses the slice's own time


def test_each_source_is_its_own_occurrence(monkeypatch, tmp_path):
    products = {"public": _product(), "kinetic-efit/neoclassical": _product(flux_time=0.31)}
    current = {}

    def fake_open(_shot, **kwargs):
        return nullcontext(products[current["source"]])

    monkeypatch.setattr(database, "open", fake_open)
    frames = []
    for source in products:
        current["source"] = source
        frames.append(database.summary((42, 42), preset="neoclassical", source=source))
    table = pd.concat(frames, ignore_index=True)
    assert table["source"].tolist() == list(products)
    assert table["transport_match_status"].tolist() == ["matched", "no_slice_at_time"]
    path = tmp_path / "neo.csv"
    database.export_summary(table, path)
    assert pd.read_csv(path)["transport_match_status"].tolist() == ["matched", "no_slice_at_time"]


@pytest.mark.skipif(not (SAMPLE.exists() and PROFILE_RUN.exists()),
                    reason="the packaged 48224 sample and the NEO fixture run are repository assets")
def test_the_real_neo_product_matches_itself():
    from vaft.omas.vest_upstream import build_neoclassical_ods

    ods, manifest = build_neoclassical_ods(shot=48224, state=SAMPLE, run_directory=PROFILE_RUN,
                                           z_eff=2.0)
    row = _row(ods)
    assert row["transport_match_status"] == "matched", row["transport_match_status"]
    assert row["q_e_points"] > 1 and row["q_i_points"] > 1
    assert row["transport_rho_grid_min"] >= row["rho_solved_min"] - 1e-9
    assert row["transport_rho_grid_max"] <= row["rho_solved_max"] + 1e-9
    assert manifest["time_s"] == pytest.approx(float(ods["core_transport.model.0.profiles_1d.0.time"]))
