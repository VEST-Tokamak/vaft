"""Canonical species/population state and physics-specific projections (#1567, validation cases A-E)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.process.impurity import (
    composition_record_text,
    resolve_impurity_composition,
    resolve_radial_composition,
)
from vaft.process.species import (
    PROJECTION_POLICY,
    CanonicalSpeciesState,
    SpeciesComponent,
    composition_moments,
    project_species_state,
    species_state_from_composition,
    species_state_from_core_profiles,
)
from vaft.spectroscopy import Species

RHO = np.linspace(0.0, 1.0, 6)
NE = 1e19 * (1.0 - 0.6 * RHO**2)


def _ods(ions, *, time=0.3):
    """core_profiles with one slice; ``ions`` = list of (path suffix -> value) dicts."""
    ods = ODS()
    ods["core_profiles.time"] = np.array([time])
    base = "core_profiles.profiles_1d.0"
    ods[f"{base}.time"] = time
    ods[f"{base}.grid.rho_tor_norm"] = RHO
    ods[f"{base}.electrons.density_thermal"] = NE
    for k, fields in enumerate(ions):
        for key, value in fields.items():
            ods[f"{base}.ion.{k}.{key}"] = value
    return ods


def _ion(label, z_n, a, z_ion, fraction, **extra):
    fields = {"label": label, "element.0.z_n": z_n, "element.0.a": a, "z_ion": z_ion,
              "density_thermal": fraction * NE}
    fields.update(extra)
    return fields


# --- Case A: one main ion + one impurity -------------------------------------------------


def test_case_a_main_ion_and_one_impurity():
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 1 - 6 / 30), _ion("C6+", 6.0, 12.011, 6.0, 1 / 30)])
    state = species_state_from_core_profiles(ods)
    assert [c.component_id for c in state.components] == ["D-2/Z1/thermal", "C-12/Z6/thermal"]
    m = composition_moments(state)
    np.testing.assert_allclose(m["quasineutrality_residual"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["zeff"], 2.0)
    np.testing.assert_allclose(m["fraction_thermal_main"], 0.8)
    np.testing.assert_allclose(m["dilution_thermal_main"], 0.2)
    np.testing.assert_allclose(m["fraction_impurity"], 0.2)
    eff = project_species_state(state, "turbulence", "effective_impurity")
    pseudo = eff.components[-1]
    np.testing.assert_allclose(pseudo.charge, 6.0)        # one impurity: the reduction is the identity
    np.testing.assert_allclose(pseudo.density, NE / 30)


# --- Case B: one element, several charge states ------------------------------------------


def test_case_b_charge_states_stay_distinct_and_reduce_on_their_moments():
    states = {f"state.{q}.{k}": v for q, (z, f) in enumerate(((4, 0.004), (5, 0.006), (6, 0.01)))
              for k, v in (("z_min", float(z)), ("z_max", float(z)), ("density", f * NE))}
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 1 - (4 * 0.004 + 5 * 0.006 + 6 * 0.01)),
                {"label": "C", "element.0.z_n": 6.0, "element.0.a": 12.011, "z_ion": 5.0, **states}])
    state = species_state_from_core_profiles(ods)
    assert [c.component_id for c in state.components][1:] == ["C-12/Z4/thermal", "C-12/Z5/thermal", "C-12/Z6/thermal"]
    atomic = project_species_state(state, "atomic")
    assert atomic.method == "charge_state_resolved" and len(atomic.components) == 4
    m = composition_moments(state)
    eff = project_species_state(state, "turbulence", "effective_impurity")
    pseudo = eff.components[-1]
    s1 = 4 * 0.004 + 5 * 0.006 + 6 * 0.01
    s2 = 16 * 0.004 + 25 * 0.006 + 36 * 0.01
    np.testing.assert_allclose(pseudo.charge, s2 / s1)
    np.testing.assert_allclose(pseudo.density / NE, s1**2 / s2)
    reduced = CanonicalSpeciesState(eff.components, rho=RHO, n_e=NE)
    np.testing.assert_allclose(composition_moments(reduced)["zeff"], m["zeff"])          # Z^2 kept
    np.testing.assert_allclose(reduced.charge_density(), state.charge_density())        # charge kept
    assert "gyrokinetic response" in eff.not_guaranteed and "atomic radiation" in eff.not_guaranteed


def test_a_bundled_ion_keeps_its_mean_square_charge_and_refuses_an_atomic_projection():
    z1, z2 = np.full(6, 5.0), np.full(6, 26.0)          # half C4+, half C6+
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 0.95), _ion("C", 6.0, 12.011, 5.0, 0.01, z_ion_1d=z1, z_ion_square_1d=z2)])
    state = species_state_from_core_profiles(ods)
    carbon = state.components[1]
    assert carbon.bundled and carbon.component_id == "C-12/Zbundle/thermal"
    np.testing.assert_allclose(composition_moments(state)["zeff"], (0.95 + 0.01 * 26.0) / 1.0)
    with pytest.raises(ValueError, match="charge-state bundles"):
        project_species_state(state, "atomic")


# --- Case C: several elements (the #1565 VEST mixture) ------------------------------------


def test_case_c_vest_mixture_explicit_and_effective_agree_on_moments_not_identity():
    state = species_state_from_composition(resolve_impurity_composition(machine_preset="vest"), NE, rho=RHO,
                                           main_isotope=2)
    explicit = project_species_state(state, "turbulence")
    effective = project_species_state(state, "turbulence", "effective_impurity")
    assert explicit.method == "explicit" and effective.method == "effective_impurity"
    assert explicit.state_id == effective.state_id and explicit.projection_id != effective.projection_id
    pseudo = effective.components[-1]
    assert pseudo.component_id == "pseudo(C+O)/Zeff/thermal"
    np.testing.assert_allclose(pseudo.charge, 50 / 7)
    np.testing.assert_allclose(pseudo.density / NE, 49 / 2150)
    np.testing.assert_allclose(pseudo.mass_amu * pseudo.density, sum(c.mass_amu * c.density for c in state.components[1:]))
    for projection in (explicit, effective):
        rebuilt = CanonicalSpeciesState(projection.components, rho=RHO, n_e=NE)
        np.testing.assert_allclose(composition_moments(rebuilt)["zeff"], 2.0)
        np.testing.assert_allclose(composition_moments(rebuilt)["fraction_thermal_main"], 36 / 43)


# --- Case D: one nuclide, two kinetic populations ------------------------------------------


def test_case_d_thermal_and_fast_populations_stay_distinct():
    fast = 0.05 * NE
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 0.75, density_fast=fast), _ion("C6+", 6.0, 12.011, 6.0, 1 / 30)])
    state = species_state_from_core_profiles(ods)
    ids = [c.component_id for c in state.components]
    assert ids == ["D-2/Z1/thermal", "D-2/Z1/fast_unspecified", "C-12/Z6/thermal"]
    m = composition_moments(state)
    np.testing.assert_allclose(m["fraction_thermal_main"], 0.75)
    np.testing.assert_allclose(m["fraction_fuel"], 0.80)            # the fast D counts as fuel ...
    np.testing.assert_allclose(m["fraction_fast_ion"], 0.05)        # ... and as fast
    np.testing.assert_allclose(m["quasineutrality_residual"], 0.0, atol=1e-12)
    for target in ("turbulence", "gyrokinetic", "fusion"):
        kept = [c.component_id for c in project_species_state(state, target).components]
        assert "D-2/Z1/fast_unspecified" in kept and "D-2/Z1/thermal" in kept   # never merged


def test_density_minus_fast_is_the_thermal_density():
    ods = _ods([{"label": "D+", "element.0.z_n": 1.0, "element.0.a": 2.014, "z_ion": 1.0,
                 "density": 0.8 * NE, "density_fast": 0.1 * NE}])
    state = species_state_from_core_profiles(ods)
    np.testing.assert_allclose(state.components[0].density, 0.7 * NE)


# --- Case E: one state, many solver projections ---------------------------------------------


def test_case_e_one_state_many_projections_with_distinct_provenance():
    state = species_state_from_composition(resolve_impurity_composition(machine_preset="vest"), NE, rho=RHO)
    cases = {
        ("resistive", None), ("turbulence", "effective_impurity"), ("turbulence", None),
        ("neoclassical", None), ("atomic", None), ("fusion", None),
    }
    projections = [project_species_state(state, t, m) for t, m in cases]
    assert len({p.state_id for p in projections}) == 1
    assert len({p.projection_id for p in projections}) == len(projections)
    resistive = next(p for p in projections if p.target_physics == "resistive")
    assert resistive.components == () and "zeff" in resistive.moments
    record = projections[0].record()
    assert {"state_id", "projection_id", "method", "preserves", "not_guaranteed", "source_components"} <= set(record)


# --- policy and identity --------------------------------------------------------------------


@pytest.mark.parametrize("target, method", [("neoclassical", "effective_impurity"),
                                            ("gyrokinetic", "effective_impurity"),
                                            ("resistive", "explicit"), ("atomic", "effective_impurity")])
def test_a_target_refuses_a_method_its_physics_does_not_allow(target, method):
    state = species_state_from_composition(resolve_impurity_composition(machine_preset="vest"), NE, rho=RHO)
    with pytest.raises(ValueError, match="does not allow"):
        project_species_state(state, target, method)


def test_policy_targets_are_closed():
    assert set(PROJECTION_POLICY) == {"quasineutrality", "resistive", "classical", "neoclassical",
                                      "turbulence", "gyrokinetic", "atomic", "fusion"}
    with pytest.raises(ValueError, match="target_physics"):
        project_species_state(species_state_from_composition(
            resolve_impurity_composition(machine_preset="vest"), NE, rho=RHO), "everything")


def test_components_validate_charge_population_and_uniqueness():
    with pytest.raises(ValueError, match="Z_n = 6"):
        SpeciesComponent(Species("C", 12), 7.0, "thermal", NE, 12.0)
    with pytest.raises(ValueError, match="population"):
        SpeciesComponent(Species("C", 12), 6.0, "warm", NE, 12.0)
    with pytest.raises(ValueError, match="bundled"):
        SpeciesComponent(Species("C", 12), 5.5, "thermal", NE, 12.0)
    c6 = SpeciesComponent(Species("C", 12), 6.0, "thermal", NE, 12.0)
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalSpeciesState((c6, c6))


def test_reading_core_profiles_creates_no_path_and_reads_origin():
    record = composition_record_text("assumed", "impurity_model_preset")
    ods = _ods([_ion("H+", 1.0, 1.008, 1.0, 0.8), _ion("C6+", 6.0, 12.011, 6.0, 1 / 30,
                                                       **{"density_fit.parameters": record})])
    before = sorted(ods["core_profiles.profiles_1d.0.ion.1"].keys())
    state = species_state_from_core_profiles(ods)
    assert sorted(ods["core_profiles.profiles_1d.0.ion.1"].keys()) == before
    assert state.components[1].origin == "assumed" and state.components[0].origin == "unlabelled"
    assert state.components[0].component_id == "H-1/Z1/thermal"


def test_a_radial_composition_gives_one_component_per_charge_state(tmp_path):
    from test_impurity_charge_states import _element

    tables = {"C": _element(tmp_path, "C", 6), "O": _element(tmp_path, "O", 8)}
    radial = resolve_radial_composition(300.0 * (1 - 0.9 * RHO**2), NE, RHO, {"C": 1, "O": 1}, tables=tables)
    state = species_state_from_composition(radial, NE)
    ids = [c.component_id for c in state.components]
    assert ids[0] == "H-1/Z1/thermal" and len(ids) == 1 + 6 + 8
    np.testing.assert_allclose(composition_moments(state)["zeff"], radial.zeff, rtol=1e-9)
    assert project_species_state(state, "atomic").method == "charge_state_resolved"
