"""Modelling Z_eff as a real impurity, on both sides of the comparison (#803).

NEO builds its collision operator from the species list and ignores `input.gacode`'s
`z_eff` column, so a file declaring `z_eff = 2` beside one hydrogenic ion describes a
plasma at `Z_eff = 1` to the solver that reads it. Every comparison in #747 and #776
was made that way: analytic models at 2, NEO at 1.

The fix is not to stop writing the column but to make the species list realize it --
`impurity="C"` gives the main ion and a carbon impurity the densities that satisfy
quasi-neutrality and the requested charge together. The analytic providers take the
same argument, so both sides model one plasma.

Two fixture pairs make the sensitivity measurable offline: `neo_vest_48224_profile`
and `neo_vest_48224_conductivity` are the pure-hydrogen runs, and
`neo_vest_48224_carbon` and `neo_vest_48224_carbon_conductivity` the same cases with
carbon at Z_eff = 2. Which model looks closer to NEO is decided by that choice, which
is the result this file pins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.inputs import IMPURITIES, ProfileConversionError, impurity_fractions
from vaft.code.gacode.neo.outputs import collect_neo_outputs

FIXTURES = Path(__file__).parent / "data" / "gacode"
HYDROGEN_RUN = FIXTURES / "neo_vest_48224_profile"
CARBON_RUN = FIXTURES / "neo_vest_48224_carbon"
CARBON_SIGMA_RUN = FIXTURES / "neo_vest_48224_carbon_conductivity"

SAMPLE = None
try:  # pragma: no cover - depends on the repository-only sample being present
    from vaft.data.resources import data_path

    _candidate = Path(data_path("kineticEfit/ods_48224_300ms.json"))
    SAMPLE = _candidate if _candidate.exists() else None
except Exception:
    SAMPLE = None

requires_sample = pytest.mark.skipif(
    SAMPLE is None, reason="the packaged 48224 kinetic sample is a repository-only asset"
)


@pytest.fixture
def state():
    from omas import load_omas_json

    return load_omas_json(str(SAMPLE), consistency_check=False)


# --------------------------------------------------------------------------
# the fractions
# --------------------------------------------------------------------------


def test_carbon_at_z_eff_two_is_the_textbook_pair():
    main, impurity = impurity_fractions(2.0, 6.0)
    assert main == pytest.approx(0.8)
    assert impurity == pytest.approx(1.0 / 30.0)


@pytest.mark.parametrize("z_eff", [1.0, 1.5, 2.0, 3.0, 5.9])
@pytest.mark.parametrize("charge", [6.0, 8.0, 2.0])
def test_the_pair_always_satisfies_both_conditions(z_eff, charge):
    """Quasi-neutrality and the Z_eff definition, for every reachable combination."""
    if z_eff > charge:
        pytest.skip("unreachable by construction")
    main, impurity = impurity_fractions(z_eff, charge)
    assert main + charge * impurity == pytest.approx(1.0)
    assert main + charge**2 * impurity == pytest.approx(z_eff)
    assert main >= 0.0 and impurity >= 0.0


def test_an_unreachable_charge_is_refused_with_the_bounds():
    with pytest.raises(ValueError, match="unreachable"):
        impurity_fractions(7.0, 6.0)
    with pytest.raises(ValueError, match="unreachable"):
        impurity_fractions(0.5, 6.0)


def test_z_eff_one_means_no_impurity_at_all():
    """The degenerate case must not leave a zero-density species in the list."""
    main, impurity = impurity_fractions(1.0, 6.0)
    assert main == pytest.approx(1.0) and impurity == pytest.approx(0.0)


# --------------------------------------------------------------------------
# the converter
# --------------------------------------------------------------------------


@requires_sample
def test_the_species_list_realizes_the_requested_charge(state):
    """The whole point: what NEO reads must be the plasma the caller asked for."""
    from vaft.code.gacode.inputs import prepare_gacode_profile

    profile = prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0, impurity="C")
    assert profile.name == ("H+", "C6+")
    densities = np.asarray(profile.ni, dtype=float)
    charges = np.asarray(profile.z, dtype=float)
    electrons = np.asarray(profile.ne, dtype=float)

    np.testing.assert_allclose((densities * charges[:, None]).sum(axis=0), electrons, rtol=1e-9)
    np.testing.assert_allclose(
        (densities * charges[:, None] ** 2).sum(axis=0) / electrons, 2.0, rtol=1e-9
    )


@requires_sample
def test_without_an_impurity_the_column_and_the_species_disagree(state):
    """The defect #803 is about, kept as a test so it cannot come back unnoticed."""
    from vaft.code.gacode.inputs import prepare_gacode_profile

    profile = prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0)
    densities = np.asarray(profile.ni, dtype=float)
    charges = np.asarray(profile.z, dtype=float)
    from_species = (densities * charges[:, None] ** 2).sum(axis=0) / np.asarray(profile.ne)
    np.testing.assert_allclose(from_species, 1.0, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(profile.z_eff), 2.0, rtol=1e-9)


@requires_sample
def test_the_substitution_of_the_measured_main_ion_is_recorded(state):
    """n_H drops to 0.8 n_e; a measurement was overridden and must say so."""
    from vaft.code.gacode.inputs import prepare_gacode_profile

    profile = prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0, impurity="C")
    record = profile.provenance["ni"]
    assert record["kind"] == "policy_assumption"
    assert "quasi-neutrality" in record["reason"]
    assert profile.provenance["ti"]["kind"] == "policy_assumption"


@requires_sample
def test_an_impurity_without_a_target_charge_is_refused(state):
    from vaft.code.gacode.inputs import prepare_gacode_profile

    with pytest.raises(ProfileConversionError, match="needs a z_eff"):
        prepare_gacode_profile(state, rho_max=0.95, impurity="C")


@requires_sample
def test_an_unknown_impurity_lists_the_known_ones(state):
    from vaft.code.gacode.inputs import prepare_gacode_profile

    with pytest.raises(ProfileConversionError, match="unknown impurity"):
        prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0, impurity="Xe")
    assert {"C", "O", "N", "He"} <= set(IMPURITIES)


# --------------------------------------------------------------------------
# what the solver then reports
# --------------------------------------------------------------------------


def test_the_carbon_run_really_ran_at_two_and_the_hydrogen_run_at_one():
    """The fixtures are the evidence; assert they are what they claim."""
    assert float(np.nanmean(collect_neo_outputs(CARBON_RUN).effective_charge)) == pytest.approx(2.0)
    assert float(np.nanmean(collect_neo_outputs(HYDROGEN_RUN).effective_charge)) == pytest.approx(1.0)
    assert collect_neo_outputs(CARBON_RUN).n_species == 3
    assert collect_neo_outputs(HYDROGEN_RUN).n_species == 2


def test_the_carbon_fixture_declares_the_impurity_in_its_own_input():
    text = (CARBON_RUN / "input.gacode").read_text(encoding="utf-8")
    species = text.split("# name", 1)[1].splitlines()[1].split()
    assert species == ["H+", "C6+"], species


# --------------------------------------------------------------------------
# the analytic side, told the same model
# --------------------------------------------------------------------------


@requires_sample
def test_the_provider_dilutes_the_ion_channel_by_the_impurity(state):
    """n_i = n_e is exact only at Z_eff = 1; carbon makes it 0.8333 n_e."""
    from vaft.omas.neoclassical import compute_bootstrap_current

    plain = compute_bootstrap_current(state, model="sauter", z_eff=2.0, rho_range=(0.05, 1.0))
    doped = compute_bootstrap_current(
        state, model="sauter", z_eff=2.0, impurity="C", rho_range=(0.05, 1.0)
    )
    assert plain.provenance["ion_density_over_electron"] == pytest.approx(1.0)
    assert doped.provenance["ion_density_over_electron"] == pytest.approx(1.0 - 1.0 / 6.0)
    assert doped.provenance["impurity"] == "C"
    # Less ion pressure, so less bootstrap current, but not by much: the shift is
    # small enough that it cannot be what decides a model ordering.
    ratio = float(np.nanmax(doped.j_bootstrap) / np.nanmax(plain.j_bootstrap))
    assert 0.95 < ratio < 1.0


@requires_sample
def test_omitting_the_impurity_leaves_the_old_answer_untouched(state):
    """Every result published before #803 must be reproducible bit for bit."""
    from vaft.omas.neoclassical import compute_bootstrap_current

    without = compute_bootstrap_current(state, model="redl", z_eff=2.0, rho_range=(0.05, 1.0))
    explicit = compute_bootstrap_current(
        state, model="redl", z_eff=2.0, impurity=None, rho_range=(0.05, 1.0)
    )
    np.testing.assert_array_equal(without.j_bootstrap, explicit.j_bootstrap)


# --------------------------------------------------------------------------
# the guard, and the sensitivity it exists to protect
# --------------------------------------------------------------------------


@requires_sample
def test_comparing_against_a_run_of_a_different_plasma_is_refused(state):
    from vaft.validation.neoclassical import bootstrap_models

    with pytest.raises(ValueError, match="different plasmas"):
        bootstrap_models(state, z_eff=2.0, impurity="C", solver_charge=1.0)


@requires_sample
def test_the_guard_passes_when_both_sides_agree(state):
    from vaft.validation.neoclassical import bootstrap_models

    models = bootstrap_models(state, z_eff=2.0, impurity="C", solver_charge=2.0)
    assert models["provenance"]["solver_charge"] == pytest.approx(2.0)


@requires_sample
def test_a_measured_zeff_that_contradicts_the_impurity_is_refused(state):
    """The defect the fix itself reintroduced: the column must not fight the species.

    A state carrying a measured zeff and a caller naming an impurity at a different
    charge would write a column NEO ignores beside a species list it obeys -- the
    original #803 failure, in the other direction. The packaged sample has no zeff,
    which is exactly why this needs a synthesized one.
    """
    from vaft.code.gacode.inputs import prepare_gacode_profile

    grid = np.asarray(state["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    state["core_profiles.profiles_1d.0.zeff"] = np.full(grid.size, 3.5)
    with pytest.raises(ProfileConversionError, match="contradicts the species"):
        prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0, impurity="C")


@requires_sample
def test_a_measured_zeff_that_agrees_is_kept_and_the_override_recorded(state):
    from vaft.code.gacode.inputs import prepare_gacode_profile

    grid = np.asarray(state["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    state["core_profiles.profiles_1d.0.zeff"] = np.full(grid.size, 2.0)
    profile = prepare_gacode_profile(state, rho_max=0.95, z_eff=2.0, impurity="C")
    assert profile.provenance["z_eff"]["kind"] == "derived"
    assert profile.provenance["z_eff"]["overrode_measured"] == pytest.approx(2.0)
    np.testing.assert_allclose(np.asarray(profile.z_eff), 2.0, rtol=1e-9)


@requires_sample
def test_an_impurity_that_would_have_zero_density_is_refused(state):
    """z_eff = 1 needs no impurity; writing one with zero density is what
    `_require_positive` refuses for every measured profile."""
    from vaft.code.gacode.inputs import prepare_gacode_profile

    with pytest.raises(ProfileConversionError, match="zero density"):
        prepare_gacode_profile(state, rho_max=0.95, z_eff=1.0, impurity="C")


def test_a_run_reports_the_ion_fraction_its_species_list_implies():
    """Z_eff alone does not identify a plasma; the ion fraction is the other half."""
    carbon = collect_neo_outputs(CARBON_RUN)
    hydrogen = collect_neo_outputs(HYDROGEN_RUN)
    assert float(np.nanmean(carbon.ion_density_fraction)) == pytest.approx(1 - 1 / 6)
    assert float(np.nanmean(hydrogen.ion_density_fraction)) == pytest.approx(1.0)


@requires_sample
def test_the_guard_catches_a_forgotten_impurity_not_only_a_wrong_charge(state):
    """Same Z_eff, different plasma: a charge-only check passes this one.

    The analytic side with no impurity has n_i = n_e whatever its Z_eff says, so
    comparing it against a carbon run is the other half of the #803 mismatch.
    """
    from vaft.validation.neoclassical import bootstrap_models

    carbon = collect_neo_outputs(CARBON_RUN)
    with pytest.raises(ValueError, match="different species|n_i/n_e"):
        bootstrap_models(
            state,
            z_eff=2.0,
            solver_charge=2.0,
            solver_ion_fraction=float(np.nanmean(carbon.ion_density_fraction)),
        )


@requires_sample
def test_an_unchecked_ion_fraction_says_so_rather_than_looking_checked(state):
    from vaft.validation.neoclassical import bootstrap_models

    rows = bootstrap_models(state, z_eff=2.0, solver_charge=2.0)
    assert "ion_fraction_unchecked" in rows["provenance"]


@requires_sample
def test_the_conductivity_records_that_the_impurity_did_not_enter_it(state):
    """sigma depends on the ion species only through Z_eff, so the argument is
    accepted for symmetry and must say that it changed nothing."""
    from vaft.omas.neoclassical import compute_conductivity

    result = compute_conductivity(state, model="sauter", z_eff=2.0, impurity="C")
    assert result.provenance["impurity"] == "C"
    assert result.provenance["impurity_affects_result"] is False
    plain = compute_conductivity(state, model="sauter", z_eff=2.0)
    np.testing.assert_array_equal(result.conductivity_parallel, plain.conductivity_parallel)


@requires_sample
def test_which_model_looks_closer_is_decided_by_the_impurity_assumption(state):
    """The finding of #803, pinned as a number rather than left as prose.

    At the charge each NEO run actually used, the ordering reverses: Redl is closer
    on the pure-hydrogen run and Sauter on the carbon one. That spread is larger than
    the difference between the models, which is why neither ordering can be quoted
    without the impurity model beside it.
    """
    from vaft.machine_mapping.neoclassical import core_profiles_from_neo
    from vaft.validation.neoclassical import bootstrap_models, model_comparison

    def measure(run, charge, impurity):
        from omas import load_omas_json

        ods = load_omas_json(str(SAMPLE), consistency_check=False)
        core_profiles_from_neo(ods, collect_neo_outputs(run), time=0.3, time_index=0)
        models = bootstrap_models(
            ods, z_eff=charge, impurity=impurity, rho_range=(0.05, 1.0), solver_charge=charge
        )
        compared = model_comparison(models, reference="neo")["models"]
        return {
            name: compared[name]["integrated_relative_difference"] for name in ("sauter", "redl")
        }

    hydrogen = measure(HYDROGEN_RUN, 1.0, None)
    carbon = measure(CARBON_RUN, 2.0, "C")

    assert abs(hydrogen["redl"]) < abs(hydrogen["sauter"]), hydrogen
    assert abs(carbon["sauter"]) < abs(carbon["redl"]), carbon
    # And the spread between the two assumptions exceeds the spread between models.
    between_models = abs(carbon["sauter"] - carbon["redl"])
    between_assumptions = abs(carbon["redl"] - hydrogen["redl"])
    assert between_assumptions > between_models, (between_assumptions, between_models)


@requires_sample
def test_the_conductivity_orders_the_models_the_same_way_as_the_bootstrap(state):
    """A second, independent quantity agreeing is what makes the ordering credible."""
    from vaft.machine_mapping.neoclassical import core_profiles_from_neo
    from vaft.omas.neoclassical import compute_conductivity
    from omas import load_omas_json

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    core_profiles_from_neo(
        ods,
        collect_neo_outputs(CARBON_RUN),
        time=0.3,
        time_index=0,
        conductivity=collect_neo_outputs(CARBON_SIGMA_RUN),
    )
    solver = np.asarray(ods["core_profiles.profiles_1d.0.conductivity_parallel"], dtype=float)

    distance = {}
    for name in ("sauter", "redl"):
        analytic = np.asarray(
            compute_conductivity(ods, model=name, z_eff=2.0, impurity="C").conductivity_parallel
        )
        both = np.isfinite(solver) & np.isfinite(analytic)
        distance[name] = abs(float(np.mean(solver[both] / analytic[both])) - 1.0)

    assert distance["sauter"] < distance["redl"], distance
    assert distance["sauter"] < 0.10
