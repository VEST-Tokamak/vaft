"""The plasma formalism of each GACODE-suite run (#1735, Phase C2 of #1723).

NEO, TGLF, TGLF-NN and CGYRO share VAFT's GACODE plumbing but not their physics. Each
record is derived from the ``KEY=VALUE`` input the run was written with. The
cross-model tests at the end pin the distinctions #1735 section 21 says must not
collapse.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from vaft.code.formalism import UNKNOWN, PlasmaFormalism
from vaft.code.gacode import cgyro, neo, tglf
from vaft.code.gacode.neo.inputs import neo_parameters
from vaft.code.gacode._formalism import (
    neo_plasma_formalism,
    tglf_effective_controls,
    tglf_plasma_formalism,
    tglf_surrogate_plasma_formalism,
)

DATA = Path(__file__).parent / "data" / "gacode"


def _input_tglf(name: str) -> dict:
    from vaft.code.mitim.compare import read_input_tglf

    return read_input_tglf(DATA / name / "input.tglf")


def _roundtrip(record: PlasmaFormalism) -> None:
    assert PlasmaFormalism.from_dict(json.loads(record.to_json())) == record


# --------------------------------------------------------------------------- NEO
def test_a_standard_neo_run_is_drift_kinetic_neoclassical_transport():
    record = neo.plasma_formalism()                       # VAFT's defaults: numerical, PROFILE_MODEL=2
    assert (record.scientific_operation, record.bulk_description, record.kinetic_equation) == (
        "neoclassical_transport", "kinetic", "drift_kinetic")
    assert (record.regime, record.spatial_domain, record.kinetic_population) == ("steady_state", "local", ("all",))
    assert record.extensions["collision_model_name"] == "full_linearized_fokker_planck"
    assert record.extensions["geometry_model"] == "input_gacode_shape"   # PROFILE_MODEL=2
    _roundtrip(record)


def test_neo_reads_what_was_written_not_what_was_configured():
    """extra_parameters reach input.neo after the named fields; the record follows the file."""
    config = neo.NEOConfig(extra_parameters={"sim_model": 0, "ae_flag": 1})
    record = neo.plasma_formalism(config)
    assert record.bulk_description == "reduced" and record.derived_from == "drift_kinetic"
    assert record.kinetic_equation is None and record.kinetic_population is None   # analytic: no distribution
    assert record.extensions["species_model"] == "adiabatic_electrons"
    written = neo_parameters(neo.NEOConfig(n_species=3, extra_parameters={"sim_model": 0, "ae_flag": 1}))
    assert neo.plasma_formalism(parameters=written) == record      # config path == written-input path


@pytest.mark.parametrize("sim, kinetic", [(0, False), (1, True), (2, True), (3, False), (4, False), (5, False)])
def test_only_the_numerical_neo_models_solve_the_drift_kinetic_equation(sim, kinetic):
    record = neo_plasma_formalism({"SIM_MODEL": sim})
    assert (record.bulk_description == "kinetic") is kinetic
    assert (record.kinetic_equation == "drift_kinetic") is kinetic
    assert record.scientific_operation == "neoclassical_transport"


@pytest.mark.parametrize("profile_model, equilibrium_model, geometry", [
    (1, 0, "s_alpha"), (1, 1, "large_aspect_ratio"), (1, 2, "miller"), (2, 0, "input_gacode_shape"),
])
def test_neo_geometry_is_read_the_way_neo_reads_it(profile_model, equilibrium_model, geometry):
    """EQUILIBRIUM_MODEL only applies with PROFILE_MODEL=1 (neo_check.f90), the Phase A trap."""
    record = neo_plasma_formalism({"PROFILE_MODEL": profile_model, "EQUILIBRIUM_MODEL": equilibrium_model})
    assert record.extensions["geometry_model"] == geometry


def test_an_unknown_neo_simulation_model_is_refused():
    with pytest.raises(ValueError, match="SIM_MODEL"):
        neo_plasma_formalism({"SIM_MODEL": 9})


# --------------------------------------------------------------------------- CGYRO
@pytest.mark.parametrize("field_model, nonlinear, operation, field", [
    ("es", False, "microstability", "electrostatic"),
    ("em-aperp", False, "microstability", "electromagnetic"),
    ("em-aperp-bpar", True, "turbulent_transport", "electromagnetic"),
])
def test_cgyro_is_first_principles_local_delta_f_gyrokinetics(field_model, nonlinear, operation, field):
    config = cgyro.CGYROConfig(field_model=field_model, nonlinear=nonlinear, n_toroidal=8 if nonlinear else 1)
    record = cgyro.plasma_formalism(config)
    assert (record.bulk_description, record.kinetic_equation, record.distribution_formulation) == (
        "kinetic", "gyrokinetic", "delta_f")
    assert (record.scientific_operation, record.field_model, record.spatial_domain) == (operation, field, "local")
    assert record.extensions["rotation"] == "none" and record.extensions["collision_model_name"] == "sugama"
    _roundtrip(record)


def test_cgyro_reads_electrons_geometry_and_domain_from_the_written_input():
    """Phase A's gap: an extra_parameters override used to contradict the hard-coded record."""
    config = cgyro.CGYROConfig(extra_parameters={"AE_FLAG": 1, "EQUILIBRIUM_MODEL": 1, "GAMMA_E": 0.1})
    record = cgyro.plasma_formalism(config)
    assert record.kinetic_population == ("ions",)
    assert record.extensions["species_model"] == "adiabatic_electrons"
    assert record.extensions["geometry_model"] == "s_alpha"
    assert record.extensions["rotation"] == {"gamma_e": 0.1, "gamma_p": 0.0, "mach": 0.0}
    legacy = cgyro.formalism(config)
    assert (legacy["species_model"], legacy["geometry_model"]) == ("adiabatic_electrons", "s_alpha")
    global_spectral = cgyro.plasma_formalism(parameters={"GLOBAL_FLAG": 1})
    assert global_spectral.spatial_domain == UNKNOWN          # never a bare "global"


def test_cgyro_field_model_and_regime_follow_the_written_input():
    """An extra_parameters N_FIELD / NONLINEAR_FLAG is what CGYRO runs (merged last)."""
    record = cgyro.plasma_formalism(cgyro.CGYROConfig(field_model="es", extra_parameters={"N_FIELD": 3}))
    assert record.field_model == "electromagnetic" and record.extensions["field_model"] == "em-aperp-bpar"
    nonlinear = cgyro.plasma_formalism(parameters={"NONLINEAR_FLAG": 1, "N_FIELD": 2})
    assert (nonlinear.regime, nonlinear.scientific_operation) == ("nonlinear", "turbulent_transport")
    from_gacode = cgyro.plasma_formalism(parameters={"PROFILE_MODEL": 2})
    assert from_gacode.extensions["rotation"] == UNKNOWN    # shear then comes from input.gacode


def test_the_1353_record_of_a_default_cgyro_run_is_unchanged():
    assert cgyro.formalism(cgyro.CGYROConfig(), solver_version="v") == {
        "distribution_formulation": "delta_f", "spatial_domain": "local", "numerical_representation": "continuum",
        "field_model": "es", "regime": "linear", "topology_domain": "closed_flux_surface",
        "geometry_model": "miller", "species_model": "kinetic_electrons", "solver": "cgyro", "solver_version": "v"}


# --------------------------------------------------------------------------- TGLF
def test_tglf_is_a_reduced_model_derived_from_gyrokinetics():
    record = tglf_plasma_formalism(_input_tglf("tglf_vest_39915_r0.70_sat3-em-bper"))
    assert record.bulk_description == "reduced" and record.derived_from == "gyrokinetic"
    assert record.kinetic_equation is None                     # it solves no gyrokinetic equation
    assert (record.scientific_operation, record.regime, record.field_model) == (
        "turbulent_transport", "quasilinear", "electromagnetic")
    assert record.extensions["closure"] == "gyro_landau_fluid"
    _roundtrip(record)


def test_the_saturation_rule_and_its_presets_are_recorded_as_effective_values():
    """SAT_RULE drives the linear presets too (tglf_startup.f90); the input's own value is not what ran."""
    sat0 = tglf_plasma_formalism(_input_tglf("tglf_vest_39915_r0.70_sat0-es"))
    sat2 = tglf_plasma_formalism(_input_tglf("tglf_vest_39915_r0.70_sat2-em-bper"))
    assert (sat0.extensions["sat_rule"], sat0.extensions["xnu_model"], sat0.extensions["units"]) == (0, 2, "GYRO")
    assert (sat2.extensions["sat_rule"], sat2.extensions["xnu_model"], sat2.extensions["units"]) == (2, 3, "CGYRO")
    assert sat2.extensions["wdia_trapped"] == 1.0 and sat0.extensions["wdia_trapped"] == 0.0
    # Same physical family, different model-reduction choices.
    assert (sat0.bulk_description, sat0.derived_from) == (sat2.bulk_description, sat2.derived_from)
    assert sat0 != sat2


def test_tglf_without_the_transport_model_is_linear_microstability():
    record = tglf.plasma_formalism(tglf.TGLFConfig(use_transport_model=False))
    assert (record.scientific_operation, record.regime) == ("microstability", "linear")


def test_tglf_follows_an_extra_parameters_override_of_the_saturation_rule():
    record = tglf.plasma_formalism(tglf.TGLFConfig(sat_rule=0, extra_parameters={"sat_rule": 3}))
    assert record.extensions["sat_rule"] == 3 and record.extensions["xnu_model"] == 3


@pytest.mark.parametrize("sat", range(4))
@pytest.mark.parametrize("units", [None, "GYRO", "CGYRO"])
@pytest.mark.parametrize("nmodes", [None, 2, 4])
def test_mitim_comparison_and_the_record_agree_on_the_presets(sat, units, nmodes):
    """Two implementations of tglf_startup.f90's presets (MITIM's input comparison keeps
    its own, unchanged); on every canonical input they must give the same values."""
    from vaft.code.mitim.compare import effective_tglf_controls

    parameters = {"SAT_RULE": sat, **({} if units is None else {"UNITS": units}),
                  **({} if nmodes is None else {"NMODES": nmodes})}
    mitim = effective_tglf_controls(parameters)
    record = tglf_effective_controls(parameters)
    assert mitim["XNU_MODEL"] == record["xnu_model"]
    assert mitim["WDIA_TRAPPED"] == record["wdia_trapped"]
    assert mitim.get("UNITS", "GYRO") == record["units"]
    assert mitim.get("NMODES", 2) == record["nmodes"]


# --------------------------------------------------------------------------- TGLF-NN
def test_tglf_nn_is_the_tglf_variant_it_reproduces_realized_by_a_network():
    surrogate = tglf_surrogate_plasma_formalism(sat_rule=2, electromagnetic=True, model="tglfnn-ukaea")
    parent = tglf_plasma_formalism({"SAT_RULE": 2, "USE_BPER": True})
    physical = ("scientific_operation", "bulk_description", "derived_from", "kinetic_equation",
                "spatial_domain", "topology_domain", "regime", "field_model")
    assert all(getattr(surrogate, name) == getattr(parent, name) for name in physical)
    assert surrogate.solver == "tglf-nn" and parent.solver == "tglf"
    assert surrogate.extensions["realization"] == "learned_surrogate"
    assert surrogate.extensions["surrogate_of"] == "tglf" and surrogate.extensions["sat_rule"] == 2
    # SAT2 implies XNU_MODEL=3 and WDIA_TRAPPED=1; the variant's own UNITS/NMODES are not stated.
    assert (surrogate.extensions["xnu_model"], surrogate.extensions["wdia_trapped"]) == (3, 1.0)
    assert surrogate.extensions["units"] == surrogate.extensions["nmodes"] == UNKNOWN
    assert surrogate.kinetic_equation is None                  # a network is no governing equation
    _roundtrip(surrogate)


def test_tglf_nn_takes_its_variant_from_the_model_not_the_request():
    unknown = tglf_surrogate_plasma_formalism(sat_rule=None, electromagnetic=None, model="m",
                                              parameters={"SAT_RULE": 3, "USE_BPER": True})
    assert unknown.extensions["sat_rule"] == UNKNOWN and unknown.field_model == UNKNOWN
    # A malformed request key never breaks the record of a prediction already made.
    assert tglf_surrogate_plasma_formalism(sat_rule=3, electromagnetic=True, model="m",
                                           parameters={"SAT_RULE": 9, "NMODES": "x"}).extensions["sat_rule"] == 3


# --------------------------------------------------------------------------- run provenance
@pytest.mark.parametrize("module", ["neo", "tglf", "cgyro"])
def test_each_runner_records_the_formalism_of_what_it_wrote(module):
    import importlib

    runner = importlib.import_module(f"vaft.code.gacode.{module}.runner")
    assert '"plasma_formalism": plasma_formalism(configuration, parameters=inputs.parameters).as_dict()' \
        in inspect.getsource(runner)


def test_surrogate_predictions_record_their_formalism():
    from vaft.code.gacode.tglf.surrogate import inference

    assert '"plasma_formalism": tglf_surrogate_plasma_formalism(' in inspect.getsource(inference)


# --------------------------------------------------------------------------- distinctions that must not collapse
def test_neo_and_cgyro_are_both_kinetic_but_not_the_same_formalism():
    neo_record, cgyro_record = neo.plasma_formalism(), cgyro.plasma_formalism()
    assert neo_record.bulk_description == cgyro_record.bulk_description == "kinetic"
    assert (neo_record.kinetic_equation, cgyro_record.kinetic_equation) == ("drift_kinetic", "gyrokinetic")
    assert neo_record.scientific_operation != cgyro_record.scientific_operation


def test_tglf_and_cgyro_share_ancestry_not_formalism():
    tglf_record, cgyro_record = tglf.plasma_formalism(), cgyro.plasma_formalism()
    assert tglf_record.derived_from == cgyro_record.kinetic_equation == "gyrokinetic"
    assert (tglf_record.bulk_description, tglf_record.kinetic_equation) == ("reduced", None)
    assert (cgyro_record.bulk_description, cgyro_record.kinetic_equation) == ("kinetic", "gyrokinetic")


def test_tglf_nn_is_not_first_principles_gyrokinetics():
    record = tglf_surrogate_plasma_formalism(sat_rule=3, electromagnetic=True, model="m")
    assert record.bulk_description != "kinetic" and record.kinetic_equation is None


def test_one_local_state_through_three_codes_gives_three_records():
    """Same surface, same input.gacode-derived state: three models, never "GACODE"."""
    records = {neo.plasma_formalism(), tglf.plasma_formalism(), cgyro.plasma_formalism()}
    assert len(records) == 3
    assert {r.solver for r in records} == {"neo", "tglf", "cgyro"}      # no record is "gacode"
