"""Classical, neoclassical and turbulent transport of one state on one axis.

Covers the closed-form neoclassical writer (identifier 5, told apart from NEO by
its producer), the guards that keep NEO's mapper and the database summary from
mistaking it for NEO, and the ``transport_hierarchy_*`` recipes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from omas import ODS

from _synthetic_inputs import make_transport_hierarchy
from vaft.machine_mapping import neoclassical as mapping
from vaft.plot.backend.recipes import build_model


class _Profile:
    """The rmin/rho pair the r/a -> rho_tor_norm bridge reads."""

    rho = np.linspace(0.0, 1.0, 21)
    rmin = 0.3 * np.linspace(0.0, 1.0, 21) ** 1.1


def _record(r, flux, z=1.0):
    return {"r_over_a": r, "ion_energy_flux_W_m2": {f"z={z:g}": flux}, "model": {"m": 1}}


def test_chang_hinton_writer_lands_a_neoclassical_entry_named_by_its_producer():
    ods = ODS(consistency_check=False)
    report = mapping.core_transport_from_chang_hinton(
        ods, [_record(0.5, -2.0), _record(0.3, 5.0)], _Profile(), time=0.3, state_identity="s1")
    base = f"core_transport.model.{report['model']}"
    assert ods[f"{base}.identifier.index"] == 5
    assert ods[f"{base}.code.name"] == mapping.CHANG_HINTON_CODE_NAME
    flux = ods[f"{base}.profiles_1d.0.ion.0.energy.flux"]
    np.testing.assert_allclose(flux, [5.0, -2.0])           # sorted by radius
    assert np.all(np.diff(ods[f"{base}.profiles_1d.0.grid_flux.rho_tor_norm"]) > 0)
    assert f"{base}.profiles_1d.0.electrons.energy.flux" not in ods
    assert ods["core_transport.ids_properties.homogeneous_time"] == 0


def test_chang_hinton_writer_is_idempotent_and_refuses_a_second_state():
    ods = ODS(consistency_check=False)
    records = [_record(0.3, 1.0), _record(0.5, 2.0)]
    first = mapping.core_transport_from_chang_hinton(ods, records, _Profile(), time=0.3, state_identity="s1")
    again = mapping.core_transport_from_chang_hinton(ods, records, _Profile(), time=0.3, state_identity="s1")
    assert (first["model"], first["profile_index"]) == (again["model"], again["profile_index"])
    other = mapping.core_transport_from_chang_hinton(ods, records, _Profile(), time=0.3, state_identity="s2")
    assert other["model"] is None and "a second state is not overwritten" in other["skipped"][-1]
    nothing = mapping.core_transport_from_chang_hinton(ods, records, _Profile(), time=0.3, state_identity="")
    assert nothing["model"] is None


def test_neo_mapper_does_not_reuse_the_closed_form_entry():
    ods = ODS(consistency_check=False)
    mapping.core_transport_from_chang_hinton(ods, [_record(0.3, 1.0)], _Profile(), time=0.3, state_identity="s")
    assert mapping._model_position(ods) == 1
    ods["core_transport.model.1.identifier.index"] = 5
    ods["core_transport.model.1.code.name"] = "NEO"
    assert mapping._model_position(ods) == 1


def test_summary_counts_only_neo_as_the_neoclassical_flux_product():
    from vaft.database import _summary

    assert _summary._CLOSED_FORM_NEOCLASSICAL_PRODUCERS == (mapping.CHANG_HINTON_CODE_NAME,)
    ods = ODS(consistency_check=False)
    for name in ("core_profiles", "core_transport"):
        ods[f"{name}.code.name"] = "NEO"
        ods[f"{name}.code.version"] = "v"
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.array([0.3])
    ods["core_profiles.profiles_1d.0.time"] = 0.3
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = np.linspace(0, 1, 5)
    ods["core_profiles.profiles_1d.0.j_bootstrap"] = np.ones(5)
    neo = "core_transport.model.0"
    ods[f"{neo}.identifier.index"] = 5
    ods[f"{neo}.profiles_1d.0.time"] = 0.3
    ods[f"{neo}.profiles_1d.0.grid_flux.rho_tor_norm"] = np.array([0.3, 0.5])
    ods[f"{neo}.profiles_1d.0.electrons.energy.flux"] = np.array([1.0, 2.0])
    ods[f"{neo}.profiles_1d.0.ion.0.energy.flux"] = np.array([1.0, 2.0])
    ods["core_transport.model.1"] = copy.deepcopy(ods[neo])
    ods["core_transport.model.1.code.name"] = mapping.CHANG_HINTON_CODE_NAME
    (row,) = _summary.extract_neoclassical(ods, 1)
    assert row["transport_match_status"] == "matched"
    assert row["transport_model_index"] == 0


def _model(name, ods, **options):
    return build_model(name, [("", ods)], **options)


def test_recipe_draws_closed_forms_as_lines_and_solvers_as_markers():
    model = _model("transport_hierarchy_profile_ion_energy_flux", make_transport_hierarchy(None))
    styles = {series.label: series.style for series in model.series}
    assert list(styles) == ["classical: Braginskii", "neoclassical: Chang-Hinton",
                            "neoclassical: NEO", "turbulent: TGLF"]
    assert styles["classical: Braginskii"]["linestyle"] == "-"
    assert styles["neoclassical: NEO"]["linestyle"] == "none" and styles["neoclassical: NEO"]["marker"]
    # one colour per family, closed form and solver alike
    assert styles["neoclassical: NEO"]["color"] == styles["neoclassical: Chang-Hinton"]["color"]
    assert model.metadata["y_scale"]["scale"] == "symlog"
    assert model.metadata["models"]["neoclassical: NEO"]["surfaces"] == 6


def test_electron_panel_has_no_closed_form_neoclassical():
    model = _model("transport_hierarchy_profile_electron_energy_flux", make_transport_hierarchy(None))
    assert [series.label for series in model.series] == [
        "classical: Braginskii", "neoclassical: NEO", "turbulent: TGLF"]


def test_ion_panel_takes_the_main_ion_not_the_first():
    ods = make_transport_hierarchy(None)
    base = "core_transport.model.2.profiles_1d.0"
    ods[f"{base}.ion.1.z_ion"] = 1.0
    ods[f"{base}.ion.1.energy.flux"] = ods[f"{base}.ion.0.energy.flux"] * 7.0
    ods[f"{base}.ion.0.z_ion"] = 6.0
    model = _model("transport_hierarchy_profile_ion_energy_flux", ods)
    neo = next(series for series in model.series if series.label == "neoclassical: NEO")
    np.testing.assert_allclose(neo.y, ods[f"{base}.ion.1.energy.flux"])


def test_models_are_matched_by_time_not_by_slice_position():
    ods = make_transport_hierarchy(None)
    ods["core_transport.model.3.profiles_1d.0.time"] = 0.31   # TGLF from another instant
    model = _model("transport_hierarchy_profile_ion_energy_flux", ods)
    assert "turbulent: TGLF" not in [series.label for series in model.series]
    assert "turbulent: TGLF" in model.metadata["unmatched"]
    with pytest.raises(ValueError):
        _model("transport_hierarchy_profile_ion_energy_flux", ODS())
