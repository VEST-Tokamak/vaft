"""NEO's parallel conductivity: the second run, its mapping, and what it agrees with.

Issue #744, closing #550's mapping audit. NEO reports no conductivity from a
transport solve; ``vgen`` obtains one by running the same case again with a unit
parallel electric field and every gradient switched off, so that ``jpar`` is the
response to the field alone (``vgen/src/vgen_compute_neo.f90:240-260``).

Two fixtures, both real seven-surface runs on the packaged VEST 48224 state:
``neo_vest_48224_profile`` is the transport run phase 5 committed, and
``neo_vest_48224_conductivity`` is its gradient-free companion. Nothing here needs
GACODE installed.

The comparison against the analytic models carries a caveat the conductivity is the
first quantity sensitive enough to expose: **NEO builds its collision operator from
the species list, not from the ``z_eff`` column of ``input.gacode``**, which it
ignores. Evaluating Sauter at 2.0 against a NEO run that used 1.0 is a 53 percent
error -- an order more than the difference between Sauter and Redl -- so every
comparison below evaluates at the charge the run itself reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaft.code.gacode.neo import NEOConfig, conductivity_parameters, neo_parameters
from vaft.code.gacode.neo.outputs import collect_neo_outputs
from vaft.machine_mapping.neoclassical import core_profiles_from_neo

FIXTURES = Path(__file__).parent / "data" / "gacode"
TRANSPORT_RUN = FIXTURES / "neo_vest_48224_profile"
CONDUCTIVITY_RUN = FIXTURES / "neo_vest_48224_conductivity"
REG18_RUN = FIXTURES / "neo_reg18"

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


@pytest.fixture(scope="module")
def transport():
    return collect_neo_outputs(TRANSPORT_RUN)


@pytest.fixture(scope="module")
def conductivity():
    return collect_neo_outputs(CONDUCTIVITY_RUN)


@pytest.fixture
def ods():
    from omas import ODS

    state = ODS(consistency_check=False)
    state["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.1509])
    return state


# --------------------------------------------------------------------------
# staging the second run
# --------------------------------------------------------------------------


def test_the_conductivity_case_zeroes_every_species_gradient():
    """One unzeroed species leaves part of the bootstrap drive in the answer."""
    parameters = conductivity_parameters(NEOConfig(n_species=3))
    assert float(parameters["EPAR0"]) == 1.0
    for index in (1, 2, 3):
        assert float(parameters[f"PROFILE_DLNNDR_{index}_SCALE"]) == 0.0
        assert float(parameters[f"PROFILE_DLNTDR_{index}_SCALE"]) == 0.0


def test_the_physics_settings_are_the_transport_case_unchanged():
    """Both runs must describe the same plasma, or the ratio is meaningless."""
    config = NEOConfig(n_species=2, n_radial=7, rmin_over_a=0.2, rmin_over_a_2=0.8)
    transport_parameters = neo_parameters(config)
    sigma_parameters = conductivity_parameters(config)
    for key, value in transport_parameters.items():
        assert sigma_parameters[key] == value, key


def test_a_caller_supplied_epar0_is_overridden_not_honoured():
    """EPAR0 is the definition of this run, not a preference."""
    config = NEOConfig(n_species=2, extra_parameters={"EPAR0": 0.0})
    assert float(conductivity_parameters(config)["EPAR0"]) == 1.0


def test_the_committed_fixture_was_staged_that_way():
    """The fixture is evidence only if its own input.neo says what it is."""
    text = (CONDUCTIVITY_RUN / "input.neo").read_text(encoding="utf-8")
    assert "EPAR0=1.0" in text
    assert "PROFILE_DLNNDR_1_SCALE=0.0" in text
    assert "PROFILE_DLNTDR_2_SCALE=0.0" in text


def test_the_two_runs_are_otherwise_the_same_case():
    def settings(path):
        return dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )

    transport = settings(TRANSPORT_RUN / "input.neo")
    sigma = settings(CONDUCTIVITY_RUN / "input.neo")
    for key, value in transport.items():
        assert sigma[key] == value, key
    assert set(sigma) - set(transport) == {
        "EPAR0",
        "PROFILE_DLNNDR_1_SCALE",
        "PROFILE_DLNTDR_1_SCALE",
        "PROFILE_DLNNDR_2_SCALE",
        "PROFILE_DLNTDR_2_SCALE",
    }


# --------------------------------------------------------------------------
# the run's own effective charge
# --------------------------------------------------------------------------


def test_a_run_reports_the_charge_its_species_list_implies(conductivity):
    """Hydrogen and electrons is Z_eff = 1, whatever input.gacode's column says."""
    charge = np.asarray(conductivity.effective_charge, dtype=float)
    np.testing.assert_allclose(charge, 1.0, rtol=1e-9)


def test_the_column_in_input_gacode_says_two_and_neo_ignored_it():
    """The exact trap the property exists for, on the file that carries it.

    The committed run's `input.gacode` declares `z_eff = 2` and lists one ion.
    NEO read the species list and ran at 1. Both halves are asserted here because
    each alone is unremarkable: a file saying 2, and a run reporting 1, are only a
    trap when they are the same run.
    """
    text = (CONDUCTIVITY_RUN / "input.gacode").read_text(encoding="utf-8")
    assert "# z_eff" in text, "the fixture's input.gacode carries no z_eff block"
    # Each row of a profile block is "<index> <value>"; the block ends at the next "#".
    declared = []
    for row in text.split("# z_eff", 1)[1].splitlines()[1:]:
        if not row.strip() or row.lstrip().startswith("#"):
            break  # the next block; a skip here would read the whole file
        declared.append(float(row.split()[1]))
    assert declared and all(abs(value - 2.0) < 1e-9 for value in declared), declared[:5]

    species = text.split("# name", 1)[1].splitlines()[1].split()
    assert species == ["H+"], species

    native = collect_neo_outputs(CONDUCTIVITY_RUN)
    np.testing.assert_allclose(np.asarray(native.effective_charge, dtype=float), 1.0)


def test_a_multi_species_run_reports_a_charge_above_one():
    """reg18 is DIII-D with carbon, so its Z_eff must be neither 1 nor invented."""
    native = collect_neo_outputs(REG18_RUN)
    charge = np.asarray(native.effective_charge, dtype=float)
    assert np.all(charge > 1.5) and np.all(charge < 2.5)


# --------------------------------------------------------------------------
# the mapping
# --------------------------------------------------------------------------


def test_the_conductivity_is_written_when_the_companion_run_is_given(
    ods, transport, conductivity
):
    report = core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=conductivity
    )
    assert "conductivity_parallel" in report["written"]
    values = np.asarray(ods["core_profiles.profiles_1d.0.conductivity_parallel"])
    assert np.all(values[np.isfinite(values)] > 0.0)


def test_without_a_companion_run_it_stays_unset_and_says_how_to_get_one(ods, transport):
    report = core_profiles_from_neo(ods, transport, time=0.3, time_index=0)
    assert "core_profiles.profiles_1d.0.conductivity_parallel" not in ods
    reason = next(r for r in report["skipped"] if "conductivity_parallel" in r)
    assert "run_neo_conductivity_case" in reason


def test_a_transport_run_passed_as_the_companion_is_refused(ods, transport):
    """The two runs' outputs are indistinguishable; only their inputs differ.

    A transport result accepted here would be read as a conductivity and would be
    wrong by whatever the bootstrap drive contributes -- silently, since the number
    that came out would still look like a conductivity.
    """
    from vaft.code.gacode.neo import NEOResult

    result = NEOResult(
        returncode=0,
        workdir=TRANSPORT_RUN,
        outputs_native=transport,
        provenance={"parameters": dict(neo_parameters(NEOConfig(n_species=2, n_radial=7)))},
    )
    report = core_profiles_from_neo(ods, transport, time=0.3, time_index=0, conductivity=result)
    assert "conductivity_parallel" not in report["written"]
    assert any("was not staged with EPAR0=1" in r for r in report["skipped"])


def test_a_transport_run_read_off_disk_is_refused_too(ods, transport):
    """The common path: `collect_neo_outputs` returns settings-free NeoOutputs.

    The guard first read the settings only from a NEOResult's provenance, so a bare
    container -- what every caller who reads a finished directory has -- skipped the
    check entirely and wrote -193 .. 31 S/m from a transport run. The run directory
    is known, and its input.neo is still in it, so there is no reason to give up.
    """
    report = core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=transport
    )
    assert "conductivity_parallel" not in report["written"]
    assert "core_profiles.profiles_1d.0.conductivity_parallel" not in ods
    assert any("was not staged with EPAR0=1" in r for r in report["skipped"])


def test_a_non_positive_conductivity_is_refused_whatever_the_settings_said(ods, transport):
    """The second, independent check: sigma = jpar/E_par is positive by construction.

    It holds even when the settings cannot be found at all -- a directory that no
    longer exists, say -- which is the only case the parameter check cannot cover.
    """
    import dataclasses

    orphaned = dataclasses.replace(transport, directory=None)
    report = core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=orphaned
    )
    assert "conductivity_parallel" not in report["written"]
    assert any("positive conductivity" in r for r in report["skipped"])


def test_the_dimensionalisation_is_vgens_own_recipe(ods, transport, conductivity):
    """sigma = jpar * (e n_0 v_t0) / (T_0[eV]/a), asserted against the raw file."""
    core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=conductivity
    )
    scales = conductivity.normalisation
    elementary_charge = 1.602176634e-19
    expected = (
        np.asarray(conductivity.bootstrap_current, dtype=float)
        * elementary_charge
        * np.asarray(scales.density_norm, dtype=float) * 1e19
        * np.asarray(scales.velocity_norm_times_a, dtype=float)
        / (
            np.asarray(scales.temperature_norm, dtype=float) * 1e3
            / np.asarray(scales.a_meters, dtype=float)
        )
    )
    written = np.asarray(ods["core_profiles.profiles_1d.0.conductivity_parallel"])
    np.testing.assert_allclose(written, expected, rtol=1e-12)


def test_it_carries_no_b_unit_factor_unlike_the_bootstrap_current(
    ods, transport, conductivity
):
    """A current over the field that drove it: the COCOS mirroring cancels.

    j_bootstrap needs B_unit/B0 and a sign; sigma needs neither, and applying the
    bootstrap current's conversion here would make it negative on VEST.
    """
    core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=conductivity
    )
    values = np.asarray(ods["core_profiles.profiles_1d.0.conductivity_parallel"])
    assert np.all(np.asarray(conductivity.normalisation.b_unit) < 0.0), "VEST B_unit is negative"
    assert np.all(values[np.isfinite(values)] > 0.0)


def test_the_provenance_distinguishes_the_two_runs(ods, transport, conductivity):
    """An acceptance criterion: both runs' settings, distinguishably."""
    core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=conductivity
    )
    parameters = str(ods["core_profiles.code.parameters"])
    assert 'run="second, gradient-free"' in parameters
    assert "EPAR0=1" in parameters
    assert 'z_eff="1"' in parameters
    assert "vgen_compute_neo.f90" in parameters


def test_an_unsolved_companion_is_reported_not_written(ods, transport, tmp_path):
    from vaft.code.gacode.neo.outputs import NeoOutputs

    broken = NeoOutputs(directory=tmp_path, errors=("ERROR: (NEO) rho_star too large",))
    report = core_profiles_from_neo(
        ods, transport, time=0.3, time_index=0, conductivity=broken
    )
    assert "conductivity_parallel" not in report["written"]
    assert any("did not solve" in r for r in report["skipped"])


# --------------------------------------------------------------------------
# what it agrees with
# --------------------------------------------------------------------------


@requires_sample
def test_the_mapped_conductivity_agrees_with_sauter_at_the_charge_neo_used():
    """The acceptance criterion: checked against the analytic model on a real run.

    Evaluated at Z_eff = 1, which is what the run's species list gives. Agreement
    to ~10 percent is what a fitted model against a drift-kinetic solve should
    look like; the models are not expected to be exact.
    """
    from omas import load_omas_json
    from vaft.omas.neoclassical import compute_conductivity

    state = load_omas_json(str(SAMPLE), consistency_check=False)
    native = collect_neo_outputs(CONDUCTIVITY_RUN)
    charge = float(np.nanmean(np.asarray(native.effective_charge, dtype=float)))

    core_profiles_from_neo(
        state, collect_neo_outputs(TRANSPORT_RUN), time=0.3, time_index=0,
        conductivity=native,
    )
    mapped = np.asarray(state["core_profiles.profiles_1d.0.conductivity_parallel"])
    analytic = compute_conductivity(state, model="sauter", z_eff=charge)
    predicted = np.asarray(analytic.conductivity_parallel)

    overlap = np.isfinite(mapped) & np.isfinite(predicted)
    assert int(np.count_nonzero(overlap)) > 20
    ratio = mapped[overlap] / predicted[overlap]
    assert np.all(ratio > 0.85) and np.all(ratio < 1.20), (ratio.min(), ratio.max())
    assert abs(float(np.mean(ratio)) - 1.0) < 0.10


@requires_sample
def test_evaluating_at_the_wrong_charge_moves_it_far_more_than_the_model_choice():
    """Why the charge is read from the run and not from the input file.

    The Sauter-Redl difference is a few percent; using Z_eff = 2 against a run that
    used 1 is tens of percent. A comparison that got this wrong would be reporting
    a species mismatch as a model difference.
    """
    from omas import load_omas_json
    from vaft.omas.neoclassical import compute_conductivity

    state = load_omas_json(str(SAMPLE), consistency_check=False)
    band = (0.2, 0.75)
    sauter_1 = compute_conductivity(state, model="sauter", z_eff=1.0, rho_range=band)
    redl_1 = compute_conductivity(state, model="redl", z_eff=1.0, rho_range=band)
    sauter_2 = compute_conductivity(state, model="sauter", z_eff=2.0, rho_range=band)

    def mean_ratio(a, b):
        left = np.asarray(a.conductivity_parallel)
        right = np.asarray(b.conductivity_parallel)
        both = np.isfinite(left) & np.isfinite(right)
        return float(np.mean(left[both] / right[both]))

    model_difference = abs(mean_ratio(redl_1, sauter_1) - 1.0)
    charge_difference = abs(mean_ratio(sauter_2, sauter_1) - 1.0)
    assert model_difference < 0.10
    assert charge_difference > 0.25
    assert charge_difference > 3 * model_difference


@requires_sample
def test_the_provider_needs_no_vacuum_field_unlike_the_bootstrap_current():
    """sigma is not normalised by a field, so a state without b0 still answers."""
    from omas import load_omas_json
    from vaft.omas.neoclassical import compute_bootstrap_current, compute_conductivity

    state = load_omas_json(str(SAMPLE), consistency_check=False)
    for path in (
        "equilibrium.vacuum_toroidal_field.b0",
        "core_profiles.vacuum_toroidal_field.b0",
    ):
        if path in state:
            del state[path]

    result = compute_conductivity(state, model="sauter", z_eff=1.0)
    assert np.isfinite(np.asarray(result.conductivity_parallel)).any()
    with pytest.raises(ValueError, match="b0|B0"):
        compute_bootstrap_current(state, model="sauter", z_eff=1.0)


@requires_sample
def test_the_neoclassical_correction_reduces_the_spitzer_value():
    """Trapped particles cannot carry current: sigma_neo < sigma_Spitzer, always."""
    from omas import load_omas_json
    from vaft.omas.neoclassical import compute_conductivity

    state = load_omas_json(str(SAMPLE), consistency_check=False)
    result = compute_conductivity(state, model="sauter", z_eff=1.0, rho_range=(0.05, 1.0))
    sigma = np.asarray(result.conductivity_parallel)
    spitzer = np.asarray(result.spitzer)
    both = np.isfinite(sigma) & np.isfinite(spitzer)
    assert np.all(sigma[both] < spitzer[both])
    # And the reduction deepens outward, where the trapped fraction rises.
    trapped = np.asarray(result.trapped_fraction)[both]
    reduction = (sigma / spitzer)[both]
    assert reduction[np.argmax(trapped)] < reduction[np.argmin(trapped)]
