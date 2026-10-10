"""Species projections of a resolved transport state (#1567 Phase C).

One C/O state -- the packaged 48224 state with its C6+ closure split into C6+ and
O8+ at the same charge density -- is projected for resistive, TGLF (explicit and
effective impurity) and NEO physics (#1567 Case E): one upstream identity, distinct
projection identities, and the policy of #1709 enforced at readiness.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vaft.process.transport_state import (
    TransportStateKey,
    assess_neo_readiness,
    assess_tglf_readiness,
    project_transport_state,
    resolve_transport_state,
    run_identity,
    transport_species_state,
)

SAMPLE = Path(__file__).resolve().parents[1] / "vaft" / "data" / "kineticEfit" / "ods_48224_300ms.json"


@pytest.fixture(scope="module")
def state():
    from omas import load_omas_json

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    del ods["core_profiles.profiles_1d.0.ion"]
    resolved = resolve_transport_state(ods, TransportStateKey(48224, 0.3, "magnetics"), efit_quality="good")
    assert resolved.resolved, resolved.reasons
    return resolved


def _with_oxygen(state, oxygen_share=0.5):
    """The state with half (or a radial share) of the C6+ charge moved to O8+."""
    profile = state.profile
    names = list(profile.name)
    c = names.index(next(n for n in names if n.startswith("C")))
    ni = np.atleast_2d(np.asarray(profile.ni, dtype=float))
    share = np.broadcast_to(np.asarray(oxygen_share, dtype=float), ni[c].shape)
    carbon, oxygen = ni[c] * (1.0 - share), ni[c] * share * 6.0 / 8.0
    ti = np.atleast_2d(np.asarray(profile.ti, dtype=float))
    rows = lambda field: None if field is None else np.vstack([np.atleast_2d(field), np.atleast_2d(field)[c]])
    new = dataclasses.replace(
        profile, z=np.append(profile.z, 8.0), mass=np.append(profile.mass, 15.999),
        name=(*names, "O8+"), type=(*profile.type, "[therm]"),
        ni=np.vstack([np.where(np.arange(ni.shape[0])[:, None] == c, carbon, ni), oxygen]),
        ti=np.vstack([ti, ti[c]]), vtor=rows(profile.vtor), vpol=rows(profile.vpol))
    return dataclasses.replace(state, profile=new)


def test_the_canonical_state_is_the_species_list_the_solvers_read(state):
    species = transport_species_state(state)
    profile = state.profile
    assert [c.component_id.split("/")[0] for c in species.components] == ["H-1", "C-12"]
    assert all(c.origin == "assumed" for c in species.components)            # policy closure
    np.testing.assert_allclose(species.charge_density(), np.asarray(profile.ne) * 1e19, rtol=1e-6)
    np.testing.assert_allclose(species.components[0].temperature, np.atleast_2d(profile.ti)[0] * 1e3)
    assert species.time == pytest.approx(0.3)


def test_one_state_many_projections(state):
    co = _with_oxygen(state)
    resistive = project_transport_state(co, "resistive")
    explicit = project_transport_state(co, "turbulence")
    reduced = project_transport_state(co, "turbulence", "effective_impurity")
    neo = project_transport_state(co, "neoclassical")
    projections = (resistive, explicit, reduced, neo)
    assert {p.canonical_state_id for p in projections} == {co.identity}
    assert len({p.species_state_id for p in projections}) == 1
    assert len({p.projection_id for p in projections}) == 4
    assert resistive.profile is None and explicit.profile is co.profile and neo.profile is co.profile
    # the effective impurity keeps the charge density and the Z^2 moment, with one impurity
    full, lumped = co.profile, reduced.profile
    assert list(lumped.name) == ["H+", "C+O_eff"] and len(lumped.z) == 2
    for power in (1, 2):
        np.testing.assert_allclose(
            np.sum(np.atleast_2d(lumped.ni) * np.asarray(lumped.z)[:, None] ** power, axis=0),
            np.sum(np.atleast_2d(full.ni) * np.asarray(full.z)[:, None] ** power, axis=0), rtol=1e-10)
    assert lumped.provenance["species_projection"]["method"] == "effective_impurity"
    assert "species_projection" not in (full.provenance or {})          # the state is not touched
    with pytest.raises(ValueError):
        project_transport_state(co, "neoclassical", "effective_impurity")


def test_readiness_runs_on_the_projection_it_is_given(state):
    co = _with_oxygen(state)
    plain = assess_tglf_readiness(co, (0.5,))
    explicit = assess_tglf_readiness(co, (0.5,), projection=project_transport_state(co, "turbulence"))
    reduced = assess_tglf_readiness(co, (0.5,), projection=project_transport_state(co, "turbulence",
                                                                                     "effective_impurity"))
    assert explicit.summary() == plain.summary()
    assert "species_projection_effective_impurity" in reduced.conditions
    assert reduced.surfaces[0].ready and plain.surfaces[0].ready
    assert len(reduced.surfaces[0].local_input.zs) == len(plain.surfaces[0].local_input.zs) - 1
    assert assess_neo_readiness(co, projection=project_transport_state(co, "neoclassical")).runnable
    with pytest.raises(ValueError, match="neoclassical"):
        assess_neo_readiness(co, projection=project_transport_state(co, "turbulence"))
    with pytest.raises(ValueError, match="another transport state"):
        assess_tglf_readiness(state, (0.5,), projection=project_transport_state(co, "turbulence"))


def test_the_run_identity_names_the_projection_only_when_one_is_used(state):
    co = _with_oxygen(state)
    parameters = {"sat_rule": 3}
    base = run_identity(co, solver="tglf", parameters=parameters, surface=0.5)
    explicit = run_identity(co, solver="tglf", parameters=parameters, surface=0.5,
                            projection=project_transport_state(co, "turbulence"))
    reduced = run_identity(co, solver="tglf", parameters=parameters, surface=0.5,
                           projection=project_transport_state(co, "turbulence", "effective_impurity"))
    assert base == run_identity(co, solver="tglf", parameters=parameters, surface=0.5)
    assert explicit == base                # the state's own species: one run, one cache key
    assert reduced != base
    with pytest.raises(ValueError, match="another transport state"):
        run_identity(state, solver="tglf", parameters=parameters,
                     projection=project_transport_state(co, "turbulence"))


def test_a_radially_varying_effective_charge_is_refused_for_a_profile(state):
    rho = np.asarray(state.profile.rho, dtype=float)
    co = _with_oxygen(state, oxygen_share=0.2 + 0.6 * rho)
    with pytest.raises(ValueError, match="varies with radius"):
        project_transport_state(co, "turbulence", "effective_impurity")
    # the explicit and moments-only projections do not need one charge per species
    assert project_transport_state(co, "turbulence").profile is co.profile
    assert project_transport_state(co, "resistive").profile is None


def test_an_unresolved_state_has_no_species(state):
    broken = dataclasses.replace(copy.copy(state), status="insufficient", reasons=("no_profile",), profile=None)
    with pytest.raises(ValueError, match="not resolved"):
        transport_species_state(broken)


def _with_fast_and_rotation(co):
    """``co`` with a fast H species after the main ion and per-species rotation rows."""
    p = co.profile
    n = len(p.z)
    ni, ti = np.atleast_2d(p.ni), np.atleast_2d(p.ti)
    order = [0, 0] + list(range(1, n))
    vtor = np.vstack([np.full(ni.shape[1], 10.0 * (k + 1)) for k in range(n + 1)])
    new = dataclasses.replace(
        p, z=np.asarray(p.z)[order], mass=np.asarray(p.mass)[order],
        name=(p.name[0], "H_fast", *p.name[1:]), type=(p.type[0], "[fast]", *p.type[1:]),
        ni=np.vstack([ni[0] * 0.9, ni[0] * 0.1, *ni[1:]]), ti=np.vstack([ti[0], ti[0] * 20, *ti[1:]]),
        vtor=vtor, vpol=None)
    return dataclasses.replace(co, profile=new)


def test_the_effective_impurity_keeps_fast_ions_and_takes_rows_by_species(state):
    co = _with_fast_and_rotation(_with_oxygen(state))
    before = {k: np.array(v, copy=True) for k, v in (("ni", co.profile.ni), ("ti", co.profile.ti),
                                                       ("vtor", co.profile.vtor))}
    lumped = project_transport_state(co, "turbulence", "effective_impurity").profile
    assert list(lumped.name) == ["H+", "H_fast", "C+O_eff"] and list(lumped.type)[1] == "[fast]"
    np.testing.assert_array_equal(np.atleast_2d(lumped.ti)[1], np.atleast_2d(co.profile.ti)[1])
    # the pseudo-ion carries the rotation row of the heaviest merged element (O, row 3)
    np.testing.assert_array_equal(lumped.vtor, np.asarray(co.profile.vtor)[[0, 1, 3]])
    for name, value in before.items():                         # the state is not touched
        np.testing.assert_array_equal(getattr(co.profile, name), value)


def test_the_inferred_ti_probe_follows_the_projection(state):
    co = _with_oxygen(state)
    probe = dataclasses.replace(co.profile, ti=np.atleast_2d(co.profile.ti) * 1.1)
    with_probe = dataclasses.replace(co, fill_probe=probe)
    reduced = project_transport_state(with_probe, "turbulence", "effective_impurity")
    assert len(reduced.fill_probe.z) == 2
    np.testing.assert_allclose(np.atleast_2d(reduced.fill_probe.ti), np.atleast_2d(reduced.profile.ti) * 1.1)
    stripped = project_transport_state(co, "turbulence", "effective_impurity")
    stripped = dataclasses.replace(stripped, canonical_state_id=with_probe.identity)
    with pytest.raises(ValueError, match="fill probe"):
        assess_tglf_readiness(with_probe, (0.5,), projection=stripped)


def test_a_label_is_trusted_only_where_its_mass_agrees(state):
    from vaft.process.transport_state import _profile_nuclide

    assert _profile_nuclide("SI", 14.0, 28.085)[0].element == "Si"      # not neutral sulphur
    assert _profile_nuclide("C6+", 6.0, 12.011)[0].element == "C"
    assert _profile_nuclide("D+", 1.0, 2.014)[0].mass_number == 2
    projected = project_transport_state(_with_oxygen(state), "turbulence", "effective_impurity")
    with pytest.raises(ValueError, match="already a species projection"):
        transport_species_state(dataclasses.replace(state, profile=projected.profile))
