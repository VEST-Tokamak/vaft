"""Classical core_transport projection and the ``classical_transport`` summary (#1654).

The packaged 48224 kinetic ODS supplies one real resolved state; everything else is
synthetic records written through the mapper. No solver runs.
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
from vaft.machine_mapping import classical
from vaft.process.transport_state import (
    CLASSICAL_MODEL,
    TransportStateKey,
    assess_tglf_readiness,
    classical_heat_fluxes,
    resolve_transport_state,
    surface_toroidal_field,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "vaft" / "data" / "kineticEfit" / "ods_48224_300ms.json"


@pytest.fixture(scope="module")
def resolved():
    from omas import load_omas_json

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    state = resolve_transport_state(ods, TransportStateKey(48224, 0.3, "magnetics"),
                                    efit_quality="good")
    assert state.resolved, state.reasons
    records = []
    for surface in assess_tglf_readiness(state, (0.3, 0.5, 0.7)).surfaces:
        local = surface.local_input
        records.append(classical_heat_fluxes(local, surface_toroidal_field(state.profile, local)))
    return state, records


class _Profile:
    """The two arrays the r/a -> rho_tor_norm bridge reads: rho_tor_norm = 0.9 r/a."""

    rmin = np.linspace(0.0, 0.3, 31)
    rho = np.linspace(0.0, 0.9, 31)


def _record(r, q_e=10.0, q_i=50.0, chi_e=0.1, chi_i=1.0, *, z="z=1", valid=True, model=None):
    return {
        "r_over_a": r,
        "electron_energy_flux_W_m2": q_e,
        "electron_particle_flux_m2_s": None,
        "ion_energy_flux_W_m2": {z: q_i},
        "ion_particle_flux_m2_s": {},
        "chi_e_m2_s": chi_e,
        "chi_i_m2_s": chi_i,
        "tau_e_s": 1e-6, "tau_i_s": 1e-4,
        "coulomb_logarithm": 13.0, "coulomb_logarithm_ion": 14.0,
        "coulomb_log_valid": valid,
        "gamma1_perp": 4.0, "b_tesla": 0.2,
        "model": CLASSICAL_MODEL if model is None else model,
    }


def _write(ods, records, time, identity, lineage="magnetics"):
    return classical.core_transport_from_classical(
        ods, records, _Profile(), time=time, state_identity=identity, efit_lineage=lineage)


# --------------------------------------------------------------------------- contract


def test_the_summary_and_the_mapper_agree_on_the_identifier_and_the_envelope():
    assert _summary.CLASSICAL_MODEL_INDEX == classical.CLASSICAL_MODEL_INDEX < 0
    assert _summary.CLASSICAL_MODEL_NAME == classical.CLASSICAL_MODEL_NAME
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3), _record(0.5)], 0.3, "state-a")
    text = ods["core_transport.model.0.code.parameters"]
    ours, theirs = classical.classical_parameters(text), _summary._classical_parameters(text)
    for key in ("formulation_sha256", "efit_lineage", "slices"):
        assert ours[key] == theirs[key]
    assert ours["formulation"] == CLASSICAL_MODEL


def test_a_real_state_maps_its_fluxes_and_coefficients_to_separate_leaves(resolved):
    state, records = resolved
    ods = ODS(consistency_check=False)
    report = classical.core_transport_from_classical(
        ods, records, state.profile, time=0.3, state_identity=state.identity,
        efit_lineage="magnetics")
    assert report["model"] == 0 and report["profile_index"] == 0, report
    records = [record.as_record() for record in records]   # the mapper took the typed results
    base = "core_transport.model.0.profiles_1d.0"
    q_e = np.asarray(ods[f"{base}.electrons.energy.flux"])
    chi_e = np.asarray(ods[f"{base}.electrons.energy.d"])
    assert q_e == pytest.approx([r["electron_energy_flux_W_m2"] for r in records])
    assert chi_e == pytest.approx([r["chi_e_m2_s"] for r in records])
    assert f"{base}.electrons.particles.flux" not in ods
    grid = np.asarray(ods[f"{base}.grid_flux.rho_tor_norm"])
    assert np.all(np.diff(grid) > 0) and grid[-1] < 1.0

    (row,) = _summary.extract_classical_transport(ods, 48224)
    assert row["state_identity"] == state.identity
    assert row["configuration_status"] == "recorded"
    assert row["q_e_peak_abs_W_m2"] == pytest.approx(max(abs(r["electron_energy_flux_W_m2"]) for r in records))
    assert row["chi_i_max_m2_s"] == pytest.approx(max(r["chi_i_m2_s"] for r in records))
    assert row["main_ion_z"] == 1.0
    assert row["particle_flux_status"] == "not_evaluated"


def test_coefficients_never_reach_a_flux_column():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3, chi_e=1e6, chi_i=1e6), _record(0.5, chi_e=1e6, chi_i=1e6)], 0.3, "s")
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["q_e_peak_abs_W_m2"] == 10.0 and row["q_i_peak_abs_W_m2"] == 50.0
    assert row["chi_e_max_m2_s"] == 1e6
    flux_columns = [c for c in _summary.CLASSICAL_TRANSPORT_COLUMNS if c.startswith(("q_", "gamma_"))]
    assert all("chi" not in c and "scale" not in c for c in flux_columns)


# --------------------------------------------------------------------------- identity


def test_lineages_and_formulations_get_separate_entries_and_states_separate_slices():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3)], 0.30, "mag-030")
    _write(ods, [_record(0.3)], 0.32, "mag-032")
    _write(ods, [_record(0.3)], 0.30, "ek-030", lineage="electron_kinetic")
    other = dict(CLASSICAL_MODEL, coefficients={"electron": 3.2, "ion": 2.0})
    _write(ods, [_record(0.3, model=other)], 0.30, "mag-030")
    rows = _summary.extract_classical_transport(ods, 1)
    keys = [(r["model_index"], r["profile_index"], r["efit_lineage"], r["state_identity"]) for r in rows]
    assert keys == [(0, 0, "magnetics", "mag-030"), (0, 1, "magnetics", "mag-032"),
                    (1, 0, "electron_kinetic", "ek-030"), (2, 0, "magnetics", "mag-030")]
    assert rows[0]["formulation_sha256"] != rows[3]["formulation_sha256"]


def test_the_same_state_is_idempotent_and_a_second_state_at_one_time_is_refused():
    ods = ODS(consistency_check=False)
    first = _write(ods, [_record(0.3)], 0.3, "state-a")
    again = _write(ods, [_record(0.3, q_e=11.0)], 0.3, "state-a")
    assert (first["model"], first["profile_index"]) == (again["model"], again["profile_index"]) == (0, 0)
    refused = _write(ods, [_record(0.3, q_e=99.0)], 0.3, "state-b")
    assert refused["model"] is None and "already holds state" in refused["skipped"][-1]
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["q_e_peak_abs_W_m2"] == 11.0 and row["state_identity"] == "state-a"


def test_surfaces_that_disagree_are_not_merged():
    ods = ODS(consistency_check=False)
    mixed_ion = _write(ods, [_record(0.3), _record(0.5, z="z=2")], 0.3, "s")
    assert mixed_ion["model"] is None and "main ion" in mixed_ion["skipped"][-1]
    other = dict(CLASSICAL_MODEL, terms="something else")
    mixed_model = _write(ods, [_record(0.3), _record(0.5, model=other)], 0.3, "s")
    assert mixed_model["model"] is None and "formulation" in mixed_model["skipped"][-1]
    assert "core_transport.model" not in ods


# --------------------------------------------------------------------------- coverage


def test_an_undefined_surface_is_dropped_not_filled():
    ods = ODS(consistency_check=False)
    report = _write(ods, [_record(0.3), _record(0.5, valid=False), _record(0.7, q_e=float("nan"))],
                    0.3, "s")
    assert any("10 eV" in reason for reason in report["skipped"])
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["points_on_grid"] == row["q_e_points"] == 1
    assert row["rho_q_e_min"] == row["rho_q_e_max"] == pytest.approx(0.27)


def test_partial_and_missing_channels_stay_distinct_from_zero():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3, q_e=0.0), _record(0.5, q_e=0.0)], 0.3, "s")
    base = "core_transport.model.0.profiles_1d.0"
    ods[f"{base}.ion.0.energy.flux"] = np.array([np.nan, 5.0])
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["q_e_points"] == 2 and row["q_e_peak_abs_W_m2"] == 0.0   # physical zero
    assert row["q_i_points"] == 1 and row["rho_q_i_min"] == pytest.approx(0.45)
    del ods[f"{base}.ion.0.energy.flux"]
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["q_i_points"] == 0 and np.isnan(row["q_i_peak_abs_W_m2"])   # missing


def test_foreign_entries_and_unrecorded_states_are_not_guessed():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3)], 0.3, "s")
    foreign = copy.deepcopy(ods["core_transport.model.0"])
    ods["core_transport.model.1"] = foreign
    ods["core_transport.model.1.identifier.name"] = "someone_elses_private_model"
    ods["core_transport.model.2"] = copy.deepcopy(ods["core_transport.model.0"])
    ods["core_transport.model.2.identifier.index"] = 5
    ods["core_transport.model.2.identifier.name"] = "neoclassical"
    ods["core_transport.model.0.profiles_1d.1"] = copy.deepcopy(ods["core_transport.model.0.profiles_1d.0"])
    ods["core_transport.model.0.profiles_1d.1.time"] = 0.4
    rows = _summary.extract_classical_transport(ods, 1)
    assert [(r["model_index"], r["configuration_status"], r["state_identity"]) for r in rows] == [
        (0, "recorded", "s"), (0, "state_unrecorded", None)]
    del ods["core_transport.model.0.code.parameters"]
    assert {r["configuration_status"] for r in _summary.extract_classical_transport(ods, 1)} == {"unrecorded"}


def test_a_stored_particle_flux_is_reported_not_hidden():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3)], 0.3, "s")
    ods["core_transport.model.0.profiles_1d.0.electrons.particles.flux"] = np.array([1.0])
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["particle_flux_status"] == "stored"


# --------------------------------------------------------------------------- preset


def test_the_preset_keys_by_source_and_model_and_round_trips(monkeypatch, tmp_path):
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3), _record(0.5)], 0.30, "mag-030")
    _write(ods, [_record(0.3)], 0.30, "ek-030", lineage="electron_kinetic")
    ods["dataset_description.pulse_time_begin"] = "2024-05-01T12:30:00"
    monkeypatch.setattr(database, "open", lambda _shot, **kwargs: nullcontext(ods))
    result = database.summary((42, 42), preset="classical_transport", source="public")
    assert result.columns[:2].tolist() == ["shot", "pulse_time_begin"]
    assert result["efit_lineage"].tolist() == ["magnetics", "electron_kinetic"]
    definition = database.get_summary_preset("classical_transport")
    assert definition.key_columns == ("source", "shot", "model_index", "profile_index", "time_s")
    path = tmp_path / "classical.csv"
    database.export_summary(result, path)
    restored = pd.read_csv(path)
    assert restored["state_identity"].tolist() == ["mag-030", "ek-030"]
    assert restored["particle_flux_status"].tolist() == ["not_evaluated"] * 2
    database.export_summary(result.assign(source="other"), path, mode="upsert",
                            key_columns=definition.key_columns,
                            replace_groups=definition.replace_groups)
    assert len(pd.read_csv(path)) == 4


def test_a_write_without_a_state_identity_is_refused():
    ods = ODS(consistency_check=False)
    report = classical.core_transport_from_classical(ods, [_record(0.3)], _Profile(), time=0.3,
                                                     state_identity="")
    assert report["model"] is None and "state_identity" in report["skipped"][-1]
    assert "core_transport.model" not in ods


def test_an_envelope_slice_without_an_identity_is_not_recorded():
    ods = ODS(consistency_check=False)
    _write(ods, [_record(0.3)], 0.3, "s")
    path = "core_transport.model.0.code.parameters"
    ods[path] = ods[path].replace('state_identity="s"', 'state_identity=""')
    (row,) = _summary.extract_classical_transport(ods, 1)
    assert row["configuration_status"] == "state_unrecorded" and row["state_identity"] is None


def test_the_projection_is_unchanged_by_the_typed_result(resolved):
    """#1899: the typed result and its as_record() project to identical core_transport."""
    state, records = resolved
    typed, plain = ODS(consistency_check=False), ODS(consistency_check=False)
    for ods, given in ((typed, records), (plain, [r.as_record() for r in records])):
        classical.core_transport_from_classical(ods, given, state.profile, time=0.3,
                                                state_identity=state.identity, efit_lineage="magnetics")
    assert _summary.extract_classical_transport(typed, 1) == _summary.extract_classical_transport(plain, 1)
