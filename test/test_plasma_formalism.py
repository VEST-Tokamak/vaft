"""The #1727 plasma-formalism record: the seven conceptual cases, its rules, CGYRO.

Each case is the classification the Phase A audit (#1725,
``docs/_guide/Plasma_models.md``) reached for a real backend mode, built as a
record.  No solver runs.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vaft.code.formalism import AXES, UNKNOWN, PlasmaFormalism

CASES = {
    # DCON, kin_flag=f
    "ideal_mhd": dict(scientific_operation="ideal_stability", bulk_description="fluid", fluid_model="ideal_mhd",
                      spatial_domain="whole_volume", regime="static", field_model="mhd_displacement", solver="dcon"),
    # RDCON + RMATCH: T_e, n_e set eta and the mass density -- still no kinetic equation.
    "resistive_mhd": dict(scientific_operation="resistive_stability", bulk_description="fluid",
                          fluid_model="resistive_mhd", spatial_domain="whole_volume", regime="linear",
                          field_model="mhd_displacement", solver="rdcon"),
    # DCON, kin_flag=t: PENTRC's bounce-averaged drift kinetics inside delta W.
    "hybrid_mhd_drift_kinetic": dict(scientific_operation="ideal_stability", bulk_description="hybrid",
                                     fluid_model="ideal_mhd", kinetic_equation="drift_kinetic",
                                     kinetic_coupling="energy", kinetic_population=("thermal_ions",),
                                     distribution_formulation="delta_f", orbit_representation="bounce_averaged",
                                     spatial_domain="whole_volume", regime="static",
                                     field_model="mhd_displacement", solver="dcon",
                                     extensions={"collision_model": "harmonic"}),
    # NEO
    "drift_kinetic": dict(scientific_operation="neoclassical_transport", bulk_description="kinetic",
                          kinetic_equation="drift_kinetic", kinetic_population=("all",),
                          distribution_formulation="delta_f", orbit_representation="guiding_center",
                          spatial_domain="local", topology_domain="closed_flux_surface", regime="steady_state",
                          solver="neo"),
    # CGYRO linear electrostatic
    "local_delta_f_gyrokinetic": dict(scientific_operation="microstability", bulk_description="kinetic",
                                      kinetic_equation="gyrokinetic", kinetic_population=("all",),
                                      distribution_formulation="delta_f", orbit_representation="gyrocenter",
                                      spatial_domain="local", topology_domain="closed_flux_surface",
                                      regime="linear", field_model="electrostatic", solver="cgyro"),
    # TGLF
    "gyrokinetic_derived_reduced": dict(scientific_operation="turbulent_transport", bulk_description="reduced",
                                        derived_from="gyrokinetic", spatial_domain="local",
                                        topology_domain="closed_flux_surface", regime="quasilinear",
                                        field_model="electromagnetic", solver="tglf",
                                        extensions={"sat_rule": 2}),
    # ASCOT5 SIM_MODE=2
    "guiding_center_particle": dict(scientific_operation="orbit_following", bulk_description="particle",
                                    kinetic_equation="fokker_planck", kinetic_population=("fast_ions",),
                                    distribution_formulation="full_f", orbit_representation="guiding_center",
                                    spatial_domain="whole_volume", regime="time_dependent",
                                    field_model="prescribed", solver="ascot5"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_audited_case_is_representable_and_round_trips(name):
    record = PlasmaFormalism(**CASES[name])
    data = record.as_dict()
    assert data["schema_version"] == 1
    assert set(data) == {"schema_version", *(f.name for f in record.__dataclass_fields__.values())}
    assert PlasmaFormalism.from_dict(json.loads(record.to_json())) == record


def test_kinetic_profiles_never_make_a_fluid_run_kinetic():
    with pytest.raises(ValueError, match="hybrid"):
        PlasmaFormalism(**{**CASES["resistive_mhd"], "kinetic_equation": "drift_kinetic"})
    # There is deliberately no field for "kinetic profiles were used".
    assert not any("profile" in name for name in PlasmaFormalism.__dataclass_fields__)


def test_a_hybrid_is_distinct_from_a_standalone_kinetic_solver():
    hybrid = PlasmaFormalism(**CASES["hybrid_mhd_drift_kinetic"])
    standalone = PlasmaFormalism(**CASES["drift_kinetic"])
    assert hybrid.kinetic_equation == standalone.kinetic_equation == "drift_kinetic"
    assert (hybrid.bulk_description, hybrid.fluid_model, hybrid.kinetic_coupling) == ("hybrid", "ideal_mhd", "energy")
    assert (standalone.bulk_description, standalone.fluid_model, standalone.kinetic_coupling) == ("kinetic", None, None)


@pytest.mark.parametrize("change, message", [
    ({"bulk_description": "fluid", "fluid_model": None, "kinetic_equation": "gyrokinetic"}, "fluid"),
    ({"fluid_model": "ideal_mhd"}, "hybrid"),
    ({"kinetic_equation": None, "distribution_formulation": None, "kinetic_population": None,
      "orbit_representation": None}, "kinetic_equation"),
    ({"kinetic_coupling": "pressure"}, "kinetic_coupling"),
    ({"derived_from": "gyrokinetic"}, "derived_from"),
    ({"regime": "turbulent"}, "regime"),
    ({"kinetic_population": "all"}, "collection"),
])
def test_generic_impossibilities_are_refused(change, message):
    with pytest.raises((ValueError, TypeError), match=message):
        PlasmaFormalism(**{**CASES["local_delta_f_gyrokinetic"], **change})


def test_a_hybrid_needs_its_coupling_and_a_particle_its_orbits():
    with pytest.raises(ValueError, match="kinetic_coupling"):
        PlasmaFormalism(**{**CASES["hybrid_mhd_drift_kinetic"], "kinetic_coupling": None})
    with pytest.raises(ValueError, match="orbit_representation"):
        PlasmaFormalism(**{**CASES["guiding_center_particle"], "orbit_representation": None})
    with pytest.raises(ValueError, match="Fokker-Planck"):
        PlasmaFormalism(**{**CASES["guiding_center_particle"], "kinetic_equation": "gyrokinetic"})
    # An orbit integrator alone (SIMPLE) solves no kinetic equation.
    simple = PlasmaFormalism(**{**CASES["guiding_center_particle"], "kinetic_equation": None,
                                "distribution_formulation": None, "solver": "simple"})
    assert simple.kinetic_equation is None


def test_not_applicable_and_unknown_are_different_and_both_allowed():
    ideal = PlasmaFormalism(**CASES["ideal_mhd"])
    assert ideal.distribution_formulation is None  # not applicable to ideal MHD
    legacy = PlasmaFormalism(**{**CASES["hybrid_mhd_drift_kinetic"], "orbit_representation": UNKNOWN})
    assert legacy.orbit_representation == UNKNOWN
    with pytest.raises(ValueError, match="no meaning"):
        PlasmaFormalism(**{**CASES["ideal_mhd"], "distribution_formulation": UNKNOWN})
    with pytest.raises(ValueError, match="required"):
        PlasmaFormalism(**{**CASES["ideal_mhd"], "bulk_description": UNKNOWN})


def test_the_axes_are_independent_where_physics_says_they_are():
    # delta-f is not local; global is not full-f; electromagnetic is not nonlinear.
    base = CASES["local_delta_f_gyrokinetic"]
    for change in ({"spatial_domain": "radially_global"}, {"distribution_formulation": "full_f"},
                   {"field_model": "electromagnetic", "regime": "linear"}, {"regime": "nonlinear"}):
        PlasmaFormalism(**{**base, **change})
    assert set(AXES) <= set(PlasmaFormalism.__dataclass_fields__)


def test_serialization_refuses_a_newer_schema_and_unknown_fields():
    data = PlasmaFormalism(**CASES["ideal_mhd"]).as_dict()
    with pytest.raises(ValueError, match="schema_version"):
        PlasmaFormalism.from_dict({**data, "schema_version": 2})
    with pytest.raises(ValueError, match="unknown"):
        PlasmaFormalism.from_dict({**data, "fidelity": "high"})


def test_the_record_is_immutable():
    record = PlasmaFormalism(**CASES["gyrokinetic_derived_reduced"])
    with pytest.raises(AttributeError):
        record.regime = "linear"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.extensions["sat_rule"] = 3  # type: ignore[index]


# ---------------------------------------------------------------------------
# CGYRO: the reference integration
# ---------------------------------------------------------------------------

#: The #1353 record as it was before #1727, for every configuration that varies it.
LEGACY = {
    (field_model, nonlinear): {
        "distribution_formulation": "delta_f", "spatial_domain": "local", "numerical_representation": "continuum",
        "field_model": field_model, "regime": "nonlinear" if nonlinear else "linear",
        "topology_domain": "closed_flux_surface", "geometry_model": "miller", "species_model": "kinetic_electrons",
        "solver": "cgyro", "solver_version": "abc123",
    }
    for field_model in ("es", "em-aperp", "em-aperp-bpar") for nonlinear in (False, True)
}


@pytest.mark.parametrize("field_model, nonlinear", sorted(LEGACY))
def test_cgyro_keeps_its_1353_record_and_derives_it_from_the_shared_one(field_model, nonlinear):
    from vaft.code.gacode import cgyro
    from vaft.code.gacode.cgyro import CGYROConfig

    config = CGYROConfig(field_model=field_model, nonlinear=nonlinear, n_toroidal=8 if nonlinear else 1)
    assert cgyro.formalism(config, solver_version="abc123") == LEGACY[(field_model, nonlinear)]
    shared = cgyro.plasma_formalism(config)
    assert (shared.bulk_description, shared.kinetic_equation) == ("kinetic", "gyrokinetic")
    assert shared.field_model == ("electrostatic" if field_model == "es" else "electromagnetic")
    assert shared.extensions["field_model"] == field_model  # the detail is kept, not coarsened away
    assert shared.scientific_operation == ("turbulent_transport" if nonlinear else "microstability")
    assert PlasmaFormalism.from_dict(shared.as_dict()) == shared


def test_importing_the_record_is_light():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, vaft.code.formalism\n"
         "print(','.join(sorted(m for m in sys.modules if m.startswith("
         "('pandas', 'omas', 'matplotlib', 'vaft.code.gacode', 'vaft.database')))))"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
