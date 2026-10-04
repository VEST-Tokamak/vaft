"""The canonical core_transport summary retains model and slice identity."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
from omas import ODS

from vaft import database
from vaft.database import _summary


def _transport_ods() -> ODS:
    ods = ODS(consistency_check=False)
    for index, parameters in ((0, "SAT_RULE=0"), (1, "SAT_RULE=3")):
        model = f"core_transport.model.{index}"
        ods[f"{model}.identifier.index"] = 6
        ods[f"{model}.identifier.name"] = "anomalous"
        ods[f"{model}.code.name"] = "TGLF"
        ods[f"{model}.code.parameters"] = parameters
        base = f"{model}.profiles_1d.0"
        ods[f"{base}.time"] = 0.3
        ods[f"{base}.grid_flux.rho_tor_norm"] = np.array([0.3, 0.6, 0.8])
        ods[f"{base}.electrons.energy.flux"] = np.array([1.0, -4.0, 2.0])
        ods[f"{base}.electrons.particles.flux"] = np.array([2.0, 3.0, 4.0])
        ods[f"{base}.ion.0.energy.flux"] = np.array([2.0, 1.0, 3.0])
        ods[f"{base}.ion.1.energy.flux"] = np.array([0.5, 0.5, 0.5])
    ods["core_transport.model.2.identifier.index"] = 5
    ods["core_transport.model.2.profiles_1d.0.time"] = 0.3
    ods["core_transport.model.0.profiles_1d.1.time"] = 0.4
    ods["core_transport.model.0.profiles_1d.1.grid_flux.rho_tor_norm"] = np.array([0.4])
    ods["core_transport.model.0.profiles_1d.1.electrons.energy.flux"] = np.array([7.0])
    return ods


def test_extract_separates_models_and_time_slices():
    rows = _summary.extract_turbulent_transport(_transport_ods(), 42)
    assert [(row["model_index"], row["profile_index"], row["time_s"]) for row in rows] == [
        (0, 0, 0.3), (0, 1, 0.4), (1, 0, 0.3)
    ]
    assert rows[0]["configuration_sha256"] != rows[2]["configuration_sha256"]
    assert rows[0]["configuration_status"] == "recorded"
    assert rows[0]["q_e_peak_abs_W_m2"] == 4.0
    assert rows[0]["q_i_peak_abs_W_m2"] == 3.5
    assert rows[0]["gamma_e_peak_abs_m2_s"] == 4.0
    assert rows[1]["rho_flux_min"] == 0.4
    assert np.isnan(rows[1]["q_i_peak_abs_W_m2"])


def test_partial_grid_and_missing_ion_flux_do_not_create_a_false_total():
    ods = _transport_ods()
    base = "core_transport.model.0.profiles_1d.0"
    ods[f"{base}.grid_flux.rho_tor_norm"] = np.array([0.3, np.nan, 0.8])
    ods[f"{base}.electrons.energy.flux"] = np.array([1.0, 100.0, 2.0])
    del ods[f"{base}.ion.1.energy.flux"]
    row = _summary.extract_turbulent_transport(ods, 42)[0]
    assert row["points_on_grid"] == 2
    assert row["q_e_points"] == 2
    assert row["q_e_peak_abs_W_m2"] == 2.0
    assert row["q_i_points"] == 0
    assert np.isnan(row["q_i_peak_abs_W_m2"])


def test_unrecorded_configuration_is_not_guessed_from_code_name():
    ods = _transport_ods()
    del ods["core_transport.model.0.code.parameters"]
    row = _summary.extract_turbulent_transport(ods, 42)[0]
    assert row["code_name"] == "TGLF"
    assert row["configuration_status"] == "unrecorded"
    assert row["configuration_sha256"] is None


def test_summary_adds_source_and_shot_timestamp_and_exports(monkeypatch, tmp_path):
    ods = _transport_ods()
    ods["dataset_description.pulse_time_begin"] = "2024-05-01T12:30:00"
    opened = []

    def fake_open(_shot, **kwargs):
        opened.append(kwargs["paths"])
        return nullcontext(ods)

    monkeypatch.setattr(database, "open", fake_open)
    result = database.summary((42, 42), preset="turbulent_transport", source="public")
    assert result.columns[:2].tolist() == ["shot", "pulse_time_begin"]
    assert len(result) == 3
    assert result["source"].tolist() == ["public"] * 3
    assert result["pulse_time_begin"].tolist() == ["2024-05-01T12:30:00"] * 3
    assert opened == [["core_transport", "dataset_description"]]
    path = tmp_path / "transport.csv"
    database.export_summary(result, path)
    restored = pd.read_csv(path)
    assert len(restored) == 3
    assert restored["configuration_sha256"].nunique() == 2
    assert pd.Timestamp(restored.loc[0, "pulse_time_begin"]) == pd.Timestamp("2024-05-01T12:30:00")
