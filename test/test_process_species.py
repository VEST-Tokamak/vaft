"""Canonical species/population state and physics-specific projections (#1567, validation cases A-E)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.process.impurity import (
    composition_from_fractions,
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
                                           main_isotope=2)    # the preset names its main ion "H"
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


# --- cold-review findings ------------------------------------------------------------------


def test_a_resolved_bundled_ion_builds_its_component_on_the_radial_charge_moments():
    # cold review 0.8.0 delta-absorb-17 species F2: an ODS-resolved composition labels a bundled
    # ion with one scalar <Z> (#1769) and carries <Z>(rho), <Z^2>(rho) beside it
    z1 = np.linspace(3.0, 6.0, 6)
    z2 = z1**2 + 0.5 * z1 * (6.0 - z1)          # <Z>^2 <= <Z^2> <= Z_n <Z>, a real spread
    ods = _ods([_ion("H", 1.0, 1.008, 1.0, 0.9),
                {"label": "C", "element.0.z_n": 6.0, "element.0.a": 12.011, "density_thermal": 0.01 * NE,
                 "z_ion_1d": z1, "z_ion_square_1d": z2}])
    resolved = resolve_impurity_composition(ods=ods, use_measured_zeff=False)
    assert resolved.mean_charge is not None and abs(resolved.species[0].charge_state - round(resolved.species[0].charge_state)) > 1e-3
    state = species_state_from_composition(resolved, NE)
    carbon = state.components[1]
    assert carbon.bundled and carbon.component_id == "C-12/Zbundle/thermal"
    np.testing.assert_allclose(carbon.charge, z1)
    np.testing.assert_allclose(carbon.mean_square_charge, z2)
    np.testing.assert_allclose(composition_moments(state)["zeff"], resolved.zeff, rtol=1e-12)


def test_a_kept_and_a_merged_component_of_one_nuclide_with_distinct_charges_merge():
    # cold review 0.8.0 delta-absorb-17 species-docs F1: selecting the merged components by
    # dataclass equality compared an array charge with a scalar one and raised
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 0.9),
                {"label": "C", "element.0.z_n": 6.0, "element.0.a": 12.011, "z_ion": 5.0,
                 "state.0.z_min": 4.0, "state.0.z_max": 6.0, "state.0.density_thermal": 0.01 * NE,
                 "state.0.z_average_1d": np.full(6, 5.0), "state.0.z_average_square_1d": np.full(6, 26.0),
                 "density_fast": 0.001 * NE}])
    state = species_state_from_core_profiles(ods)
    assert [c.component_id for c in state.components] == ["D-2/Z1/thermal", "C-12/Z4-6/thermal", "C-12/Z5/fast_unspecified"]
    for target, method in (("turbulence", "effective_impurity"), ("fusion", "fusion")):
        ids = [c.component_id for c in project_species_state(state, target, method).components]
        assert ids == ["D-2/Z1/thermal", "C-12/Z5/fast_unspecified", "pseudo(C)/Zeff/thermal"], method
    thermal, fast = state.components[1], state.components[2]
    c_fast = SpeciesComponent(Species("C", 12), 5.5, "nbi_fast", 0.001 * NE, 12.0, bundled=True)
    direct = project_species_state(CanonicalSpeciesState((state.components[0], thermal, c_fast), rho=RHO),
                                   "turbulence", "effective_impurity")
    assert [c.component_id for c in direct.components][1:] == ["C-12/Zbundle/nbi_fast", "pseudo(C)/Zeff/thermal"]
    assert thermal != fast and thermal == thermal      # identity, never an array comparison


def test_a_mean_square_charge_on_a_fixed_charge_ion_is_ignored_and_noted():
    # cold review 0.8.0 delta-absorb-17 species-docs F6: z_ion_square_1d beside a scalar z_ion = Z_n
    # (a non-VAFT writer) made the whole slice read raise unless it was exactly Z^2
    ods = _ods([_ion("H", 1.0, 1.008, 1.0, 0.9),
                _ion("C", 6.0, 12.011, 6.0, 0.01, z_ion_square_1d=np.full(6, 35.0))])
    state = species_state_from_core_profiles(ods)
    carbon = state.components[1]
    assert carbon.component_id == "C-12/Z6/thermal" and not carbon.bundled and carbon.mean_square_charge is None
    np.testing.assert_allclose(carbon.z_moment(2), 36.0)
    assert "z_ion_square_1d ignored" in state.provenance["notes"]["ion.1 (C)"]
    # a bundled scalar z_ion keeps its stored <Z^2>
    bundled = species_state_from_core_profiles(
        _ods([_ion("H", 1.0, 1.008, 1.0, 0.9), _ion("C", 6.0, 12.011, 5.5, 0.01, z_ion_square_1d=np.full(6, 31.0))]))
    np.testing.assert_allclose(bundled.components[1].z_moment(2), 31.0)
    assert bundled.provenance["notes"] == {}


def test_a_fixed_charge_ion_beside_a_bundled_one_keeps_its_integer_charge():
    ods = _ods([_ion("H", 1.0, 1.008, 1.0, 0.9),
                {"label": "C", "element.0.z_n": 6.0, "element.0.a": 12.011, "density_thermal": 0.01 * NE,
                 "z_ion_1d": np.linspace(3.0, 6.0, 6)},
                _ion("O", 8.0, 15.999, 8.0, 0.002)])
    state = species_state_from_composition(resolve_impurity_composition(ods=ods, use_measured_zeff=False), NE)
    assert [c.component_id for c in state.components] == ["H-1/Z1/thermal", "C-12/Zbundle/thermal", "O-16/Z8/thermal"]
    assert not state.components[2].bundled and state.components[2].charge == 8.0


def test_a_non_integer_label_without_moments_is_a_bundle_at_that_charge():
    composition = composition_from_fractions(["C"], [1], [4.5], target_zeff=1.5)
    state = species_state_from_composition(resolve_impurity_composition(composition=composition), NE, rho=RHO)
    carbon = state.components[1]
    assert carbon.bundled and carbon.charge == 4.5 and carbon.mean_square_charge is None


def test_the_packaged_product_resolved_on_real_adf11_tables_round_trips(tmp_path):
    # the verifier's real-ADF11 probe: the OPEN-ADAS tables are read from the local cache only
    from pathlib import Path

    from omas import load_omas_json

    import vaft
    from vaft.ods_access import path_value
    from vaft.process.impurity import populate_radial_impurity_profiles

    cache = Path.home() / "Library/Caches/vaft/open_adas/adf11"
    tables = {s: (cache / f"acd96_{s.lower()}.dat", cache / f"scd96_{s.lower()}.dat") for s in ("C", "O")}
    if not all(p.is_file() for pair in tables.values() for p in pair):
        pytest.skip("no cached OPEN-ADAS C/O tables (never downloaded by a test)")
    sample = Path(vaft.__file__).resolve().parent / "data/kineticEfit/ods_48224_300ms.json"
    ods = load_omas_json(str(sample), consistency_check=False)
    base = "core_profiles.profiles_1d.0"
    rho, te = np.asarray(ods[f"{base}.grid.rho_tor_norm"]), np.asarray(ods[f"{base}.electrons.temperature"])
    ne = path_value(ods, f"{base}.electrons.density_thermal")
    ne = np.asarray(ods[f"{base}.electrons.density"] if ne is None else ne, dtype=float)
    radial = resolve_radial_composition(te, ne, rho, {"C": 1, "O": 1},
                                        tables={k: tuple(map(str, v)) for k, v in tables.items()}, time=0.3)
    out = populate_radial_impurity_profiles(ods, radial)
    resolved = resolve_impurity_composition(out, machine_preset="vest")
    assert resolved.kind == "derived" and resolved.mean_charge is not None
    state = species_state_from_composition(resolved, ne)
    assert [c.component_id for c in state.components[1:]] == ["C-12/Zbundle/thermal", "O-16/Zbundle/thermal"]
    np.testing.assert_allclose(composition_moments(state)["zeff"], resolved.zeff, rtol=1e-9)


def _component(element, a, z, density, **kw):
    return SpeciesComponent(Species(element, a), z, kw.pop("population", "thermal"), density,
                            kw.pop("mass", float(a)), **kw)


def test_the_effective_impurity_keeps_mass_density_where_the_mix_varies():
    d = _component("H", 2, 1.0, NE * 0.9, mass=2.014)
    c = _component("C", 12, 6.0, 1e17 * (1 - RHO) + 1e14)
    w = _component("W", 184, 20.0, 1e15 * RHO + 1e12, bundled=True)
    state = CanonicalSpeciesState((d, c, w), rho=RHO)
    pseudo = project_species_state(state, "turbulence", "effective_impurity").components[-1]
    np.testing.assert_allclose(pseudo.mass_amu * pseudo.density, 12 * c.density + 184 * w.density, rtol=1e-12)
    np.testing.assert_allclose(pseudo.density * pseudo.charge, 6 * c.density + 20 * w.density, rtol=1e-12)


def test_an_undefined_impurity_point_stays_undefined_not_zero():
    c = _component("C", 12, 6.0, np.where(RHO == RHO[2], np.nan, NE / 30))
    state = CanonicalSpeciesState((_component("H", 1, 1.0, 0.8 * NE, mass=1.008), c), rho=RHO)
    pseudo = project_species_state(state, "turbulence", "effective_impurity").components[-1]
    assert np.isnan(pseudo.density[2]) and np.all(np.isfinite(np.delete(pseudo.density, 2)))


def test_states_and_an_ion_level_fast_density_both_survive():
    ods = _ods([{"label": "D", "element.0.z_n": 1.0, "element.0.a": 2.014, "z_ion": 1.0,
                 "density_fast": 0.1 * NE, "state.0.z_min": 1.0, "state.0.z_max": 1.0,
                 "state.0.density_thermal": 0.9 * NE}])
    ids = [c.component_id for c in species_state_from_core_profiles(ods).components]
    assert ids == ["D-2/Z1/thermal", "D-2/Z1/fast_unspecified"]


def test_a_state_spanning_several_charges_is_a_bundle_with_its_own_averages():
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 0.95),
                {"label": "C", "element.0.z_n": 6.0, "element.0.a": 12.011, "z_ion": 5.0,
                 "state.0.z_min": 4.0, "state.0.z_max": 6.0, "state.0.density": 0.01 * NE,
                 "state.0.z_average_1d": np.full(6, 5.0), "state.0.z_average_square_1d": np.full(6, 26.0),
                 "state.1.z_min": 4.0, "state.1.z_max": 5.0, "state.1.density": 0.001 * NE}])
    state = species_state_from_core_profiles(ods)
    ids = [c.component_id for c in state.components]
    assert ids[1:] == ["C-12/Z4-6/thermal", "C-12/Z4-5/thermal"]
    np.testing.assert_allclose(state.components[1].z_moment(2), 26.0)
    with pytest.raises(ValueError, match="charge-state bundles"):
        project_species_state(state, "atomic")


def test_the_main_ion_is_the_composition_s_own():
    deuterium = composition_from_fractions(["C"], [1], [6], target_zeff=2.0, main_ion="D")
    state = species_state_from_composition(resolve_impurity_composition(composition=deuterium), NE, rho=RHO)
    assert state.components[0].component_id == "D-2/Z1/thermal"
    np.testing.assert_allclose(composition_moments(state)["zeff"], 2.0)
    with pytest.raises(ValueError, match="names its main ion"):
        species_state_from_composition(resolve_impurity_composition(composition=deuterium), NE, rho=RHO,
                                       main_isotope=3)


def test_state_id_tracks_mean_square_charge_temperature_and_not_order():
    def state(z2, te, order=1):
        h = _component("H", 1, 1.0, 0.95 * NE, mass=1.008, temperature=te)
        c = _component("C", 12, 5.0, 0.01 * NE, bundled=True, mean_square_charge=np.full(6, z2))
        return CanonicalSpeciesState((h, c)[::order], rho=RHO)

    base = state(26.0, np.full(6, 100.0))
    assert base.state_id != state(28.0, np.full(6, 100.0)).state_id
    assert base.state_id != state(26.0, np.full(6, 200.0)).state_id
    assert base.state_id == state(26.0, np.full(6, 100.0), order=-1).state_id


@pytest.mark.parametrize("z2", [1.0, 40.0])
def test_an_unphysical_mean_square_charge_is_refused(z2):
    with pytest.raises(ValueError, match="<Z>\\^2 <= <Z\\^2> <= Z_n <Z>"):
        _component("C", 12, 5.0, NE, bundled=True, mean_square_charge=np.full(6, z2))


def test_the_fusion_projection_keeps_helium_3():
    d = _component("H", 2, 1.0, 0.8 * NE, mass=2.014)
    he3 = _component("He", 3, 2.0, 0.05 * NE, mass=3.016)
    he4 = _component("He", 4, 2.0, 0.05 * NE, mass=4.003)
    ids = [c.component_id for c in project_species_state(CanonicalSpeciesState((d, he3, he4), rho=RHO), "fusion").components]
    assert "He-3/Z2/thermal" in ids and "pseudo(He)/Zeff/thermal" in ids


def test_hydrogen_isotopes_must_match_a_mass_and_other_elements_read():
    ods = _ods([_ion("X", 1.0, 4.0, 1.0, 1.0)])
    with pytest.raises(ValueError, match="no hydrogen isotope"):
        species_state_from_core_profiles(ods)
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 0.98), _ion("Si", 14.0, 28.09, 14.0, 0.001)])
    assert species_state_from_core_profiles(ods).components[1].component_id == "Si-28/Z14/thermal"


def test_n_e_is_the_total_electron_density():
    ods = _ods([_ion("D+", 1.0, 2.014, 1.0, 1.0)])
    ods["core_profiles.profiles_1d.0.electrons.density"] = 1.1 * NE
    np.testing.assert_allclose(species_state_from_core_profiles(ods).n_e, 1.1 * NE)
