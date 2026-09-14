"""NEO -> IMAS projection (issue #550, phase 5).

The fixture under `test/data/gacode/neo_vest_48224_profile/` is a real seven-radius
NEO run on the packaged VEST 48224 kinetic state, so these assert against output a
solver actually produced. Nothing here needs GACODE installed.

The central claim under test is that neither mapped quantity is a copy. NEO's
bootstrap current dimensionalises to `<J.B>/B_unit`; IMAS wants `<J.B>/B0` with B0
the vacuum field. On VEST those differ by a factor of about two *and* a sign, and a
sign-dropping bug passes every magnitude check -- so the sign is asserted on its own.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from vaft.code.gacode.neo.outputs import collect_neo_outputs
from vaft.machine_mapping.neoclassical import (
    FLUX_MULTIPLIER,
    NEOCLASSICAL_MODEL_INDEX,
    core_profiles_from_neo,
    core_transport_from_neo,
)
from vaft.ods_access import path_count

FIXTURES = Path(__file__).parent / "data" / "gacode"
PROFILE_RUN = FIXTURES / "neo_vest_48224_profile"
SINGLE_RUN = FIXTURES / "neo_vest_48224"

ELEMENTARY_CHARGE = 1.602176634e-19

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
def native():
    return collect_neo_outputs(PROFILE_RUN)


@pytest.fixture
def ods():
    """A minimal ODS carrying only what the mapping reads: the vacuum field."""
    from omas import ODS

    blank = ODS(consistency_check=False)
    blank["equilibrium.vacuum_toroidal_field.b0"] = np.array([0.150869643])
    blank["equilibrium.vacuum_toroidal_field.r0"] = 0.4
    return blank


@pytest.fixture
def mapped(ods, native):
    core_profiles_from_neo(ods, native, time=0.3, time_index=0)
    core_transport_from_neo(ods, native, time=0.3, time_index=0)
    return ods


def _b0(ods) -> float:
    return float(np.ravel(ods["equilibrium.vacuum_toroidal_field.b0"])[0])


# --------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------


def test_the_fixture_is_a_real_multi_radius_run(native):
    assert native.solved
    assert native.grid.n_radial == 7
    assert native.normalisation is not None and native.coordinates is not None
    # B_unit varies across the profile and is negative under GACODE's COCOS 2,
    # which is exactly what makes the conversion non-trivial.
    b_unit = np.asarray(native.normalisation.b_unit)
    assert np.all(b_unit < 0.0)
    assert b_unit.min() != b_unit.max()


# --------------------------------------------------------------------------
# core_profiles.j_bootstrap
# --------------------------------------------------------------------------


def test_the_bootstrap_current_is_derived_not_copied(mapped, native):
    """Dimensionalise, then convert <J.B>/B_unit into IMAS's <J.B>/B0."""
    scales = native.normalisation
    expected = (
        np.asarray(native.bootstrap_current)
        * ELEMENTARY_CHARGE
        * np.asarray(scales.density_norm) * 1e19
        * np.asarray(scales.velocity_norm_times_a)
        * np.asarray(scales.b_unit)
        / _b0(mapped)
    )
    written = np.asarray(mapped["core_profiles.profiles_1d.0.j_bootstrap"])
    np.testing.assert_allclose(written, expected, rtol=1e-12)
    # And it is emphatically not the normalised value, nor the value without the
    # B_unit/B0 conversion.
    assert not np.allclose(written, np.asarray(native.bootstrap_current))


def test_the_b_unit_sign_is_carried_not_dropped(mapped, native):
    """A sign-dropping bug keeps every magnitude right, so assert the sign alone.

    B_unit is negative for VEST, so j_bootstrap must come out opposite in sign to
    NEO's normalised jparB at every radius.
    """
    written = np.asarray(mapped["core_profiles.profiles_1d.0.j_bootstrap"])
    normalised = np.asarray(native.bootstrap_current)
    assert np.all(np.sign(written) == -np.sign(normalised))


def test_the_conversion_factor_is_about_two_on_vest(mapped, native):
    """B_unit is not B0: treating them as the same is a factor-of-two error."""
    ratio = np.abs(np.asarray(native.normalisation.b_unit) / _b0(mapped))
    assert np.all(ratio > 1.8) and np.all(ratio < 2.4)


def test_the_bootstrap_current_is_co_field_where_the_pressure_falls(mapped):
    """VEST 48224 has Ip parallel to Bt, so the bootstrap current is positive.

    Not at the innermost surface, where the profile turns over; checked over the
    mid-radius band where the pressure gradient actually drives it.
    """
    rho = np.asarray(mapped["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    current = np.asarray(mapped["core_profiles.profiles_1d.0.j_bootstrap"])
    band = current[rho > 0.3]
    assert band.size >= 4
    assert np.all(band > 0.0)


def test_the_profile_is_on_neos_own_coordinate_bridge(mapped, native):
    """out.neo.exprhon is what takes r/a back to a flux coordinate IMAS knows."""
    np.testing.assert_allclose(
        np.asarray(mapped["core_profiles.profiles_1d.0.grid.rho_tor_norm"]),
        np.asarray(native.coordinates["rho_tor_norm"]),
    )


def test_the_vacuum_field_it_normalised_by_is_recorded(mapped):
    """Otherwise a reader cannot reconstruct <J.B> from j_bootstrap."""
    assert float(
        np.ravel(mapped["core_profiles.vacuum_toroidal_field.b0"])[0]
    ) == pytest.approx(_b0(mapped))


def test_current_bootstrap_is_left_unset(mapped, native, ods):
    """It is a toroidal current in amperes, not the integral of a parallel one."""
    assert "core_profiles.global_quantities.current_bootstrap" not in mapped
    report = core_profiles_from_neo(ods, native, time=0.3, time_index=0)
    assert any("current_bootstrap" in reason for reason in report["skipped"])


def test_conductivity_is_left_unset(mapped, native, ods):
    assert "core_profiles.profiles_1d.0.conductivity_parallel" not in mapped
    report = core_profiles_from_neo(ods, native, time=0.3, time_index=0)
    assert any("conductivity_parallel" in reason for reason in report["skipped"])


def test_without_a_vacuum_field_the_bootstrap_current_is_refused(native):
    """j_bootstrap is defined as <J.B>/B0; a different B0 is a different number."""
    from omas import ODS

    blank = ODS(consistency_check=False)
    report = core_profiles_from_neo(blank, native)
    assert report["written"] == []
    assert any("b0" in reason for reason in report["skipped"])
    assert "core_profiles.profiles_1d.0.j_bootstrap" not in blank


def test_an_explicit_vacuum_field_overrides_the_ods(ods, native):
    core_profiles_from_neo(ods, native, b0=0.3)
    a = np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"]).copy()
    from omas import ODS

    other = ODS(consistency_check=False)
    core_profiles_from_neo(other, native, b0=0.6)
    b = np.asarray(other["core_profiles.profiles_1d.0.j_bootstrap"])
    np.testing.assert_allclose(a, 2.0 * b, rtol=1e-12)


# --------------------------------------------------------------------------
# core_transport
# --------------------------------------------------------------------------


def test_the_model_is_identified_as_neoclassical(mapped):
    assert int(mapped["core_transport.model.0.identifier.index"]) == NEOCLASSICAL_MODEL_INDEX
    assert str(mapped["core_transport.model.0.identifier.name"]) == "neoclassical"


def test_the_flux_multiplier_says_the_energy_flux_is_already_total(mapped):
    """NEO's eflux includes convection, so IMAS must not add the particle flux."""
    assert float(mapped["core_transport.model.0.flux_multiplier"]) == FLUX_MULTIPLIER == 0.0


def test_fluxes_land_on_grid_flux_and_not_on_a_diffusivity_grid(mapped, native):
    base = "core_transport.model.0.profiles_1d.0"
    np.testing.assert_allclose(
        np.asarray(mapped[f"{base}.grid_flux.rho_tor_norm"]),
        np.asarray(native.coordinates["rho_tor_norm"]),
    )
    # NEO produces fluxes, not a D/V split; inventing one would be a claim.
    assert f"{base}.grid_d.rho_tor_norm" not in mapped
    assert f"{base}.grid_v.rho_tor_norm" not in mapped


def test_the_fluxes_are_dimensionalised_with_the_expnorm_scales(mapped, native):
    scales = native.normalisation
    density = np.asarray(scales.density_norm) * 1e19
    velocity = np.asarray(scales.velocity_norm_times_a)
    temperature = np.asarray(scales.temperature_norm) * 1e3 * ELEMENTARY_CHARGE
    electron = int(np.argmin(np.asarray(native.species_charge)))
    base = "core_transport.model.0.profiles_1d.0"
    np.testing.assert_allclose(
        np.asarray(mapped[f"{base}.electrons.particles.flux"]),
        np.atleast_2d(native.transport["particle_flux"])[electron] * density * velocity,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(mapped[f"{base}.electrons.energy.flux"]),
        np.atleast_2d(native.transport["energy_flux"])[electron] * density * velocity * temperature,
        rtol=1e-12,
    )


def test_species_are_attributed_to_the_right_channel(mapped, native):
    """The electron is the negative-charge entry; everything else is an ion."""
    charge = np.asarray(native.species_charge)
    electron = int(np.argmin(charge))
    ions = [i for i in range(charge.size) if i != electron]
    base = "core_transport.model.0.profiles_1d.0"
    assert path_count(mapped, f"{base}.ion") == len(ions)
    for position, species in enumerate(ions):
        assert float(mapped[f"{base}.ion.{position}.z_ion"]) == pytest.approx(charge[species])
        np.testing.assert_allclose(
            np.asarray(mapped[f"{base}.ion.{position}.particles.flux"]),
            np.atleast_2d(native.transport["particle_flux"])[species]
            * np.asarray(native.normalisation.density_norm) * 1e19
            * np.asarray(native.normalisation.velocity_norm_times_a),
            rtol=1e-12,
        )


def test_electrons_are_not_written_as_an_ion(mapped, native):
    """A transposed species index would put the electron flux on ion 0."""
    base = "core_transport.model.0.profiles_1d.0"
    electron_flux = np.asarray(mapped[f"{base}.electrons.particles.flux"])
    ion_flux = np.asarray(mapped[f"{base}.ion.0.particles.flux"])
    assert not np.allclose(electron_flux, ion_flux)


def test_calling_twice_does_not_add_a_second_model(ods, native):
    for _ in range(2):
        core_transport_from_neo(ods, native, time=0.3, time_index=0)
    assert path_count(ods, "core_transport.model") == 1


def test_a_second_model_of_another_kind_is_not_overwritten(ods, native):
    ods["core_transport.model.0.identifier.index"] = 6  # anomalous
    core_transport_from_neo(ods, native)
    assert path_count(ods, "core_transport.model") == 2
    assert int(ods["core_transport.model.1.identifier.index"]) == NEOCLASSICAL_MODEL_INDEX


# --------------------------------------------------------------------------
# Absent products are skipped, never faked
# --------------------------------------------------------------------------


@pytest.mark.parametrize("removed", ["out.neo.expnorm", "out.neo.exprhon"])
def test_a_run_without_the_scales_or_the_grid_is_skipped_not_zero_filled(
    tmp_path, ods, removed
):
    """NEO writes both only for PROFILE_MODEL >= 2, so this is a real run shape."""
    case = tmp_path / "case"
    shutil.copytree(PROFILE_RUN, case)
    (case / removed).unlink()
    partial = collect_neo_outputs(case)

    profiles = core_profiles_from_neo(ods, partial)
    transport = core_transport_from_neo(ods, partial)
    assert profiles["written"] == [] and transport["written"] == []
    assert any("PROFILE_MODEL" in reason for reason in profiles["skipped"])
    assert "core_profiles.profiles_1d.0.j_bootstrap" not in ods
    assert "core_transport.model.0.profiles_1d.0.grid_flux.rho_tor_norm" not in ods


def test_a_run_with_no_transport_product_writes_nothing(tmp_path, ods):
    case = tmp_path / "case"
    shutil.copytree(PROFILE_RUN, case)
    (case / "out.neo.transport").unlink()
    partial = collect_neo_outputs(case)
    assert core_profiles_from_neo(ods, partial)["written"] == []
    assert core_transport_from_neo(ods, partial)["written"] == []


def test_a_single_surface_run_maps_to_a_one_point_profile(ods):
    """N_RADIAL defaults to 1, and a one-point profile is honest, not an error."""
    single = collect_neo_outputs(SINGLE_RUN)
    report = core_profiles_from_neo(ods, single, time=0.3)
    assert report["written"] == ["j_bootstrap"]
    assert np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"]).size == 1


# --------------------------------------------------------------------------
# Interface hygiene
# --------------------------------------------------------------------------


def test_the_native_result_is_not_mutated(mapped, native):
    before = np.asarray(native.bootstrap_current).copy()
    core_profiles_from_neo(mapped, native)
    np.testing.assert_array_equal(np.asarray(native.bootstrap_current), before)


def test_either_the_wrapper_or_the_container_is_accepted(ods, native):
    """Callers hold an NEOResult; tests and readers hold a NeoOutputs."""
    from vaft.code.gacode.neo import NEOResult

    wrapped = NEOResult(returncode=0, workdir=PROFILE_RUN, outputs_native=native)
    core_profiles_from_neo(ods, wrapped)
    assert "core_profiles.profiles_1d.0.j_bootstrap" in ods


def test_the_report_says_what_was_written_and_what_was_skipped(ods, native):
    report = core_transport_from_neo(ods, native)
    assert "electrons.energy.flux" in report["written"]
    assert report["model"] == 0
    assert all(isinstance(reason, str) and reason for reason in report["skipped"])


def test_the_written_ods_survives_a_save_load_round_trip(mapped, tmp_path):
    """A path that only looks present is the failure mode flat() hides."""
    from omas import load_omas_json, save_omas_json

    path = tmp_path / "round_trip.json"
    save_omas_json(mapped, str(path))
    back = load_omas_json(str(path), consistency_check=False)
    np.testing.assert_allclose(
        np.asarray(back["core_profiles.profiles_1d.0.j_bootstrap"]),
        np.asarray(mapped["core_profiles.profiles_1d.0.j_bootstrap"]),
    )
    np.testing.assert_allclose(
        np.asarray(back["core_transport.model.0.profiles_1d.0.electrons.energy.flux"]),
        np.asarray(mapped["core_transport.model.0.profiles_1d.0.electrons.energy.flux"]),
    )
    assert int(back["core_transport.model.0.identifier.index"]) == NEOCLASSICAL_MODEL_INDEX


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_records_the_solver_and_the_derivation(mapped, native):
    assert str(mapped["core_profiles.code.name"]) == "NEO"
    assert str(mapped["core_transport.code.name"]) == "NEO"
    parameters = str(mapped["core_transport.code.parameters"])
    assert "derived_by" in parameters
    assert "B_unit" in parameters
    assert native.version["revision"].split()[0] in parameters


def test_provenance_says_which_bootstrap_current_was_written(mapped):
    """NEO writes three: its own solve, and its Sauter and Redl theory columns."""
    parameters = str(mapped["core_profiles.code.parameters"])
    assert "drift-kinetic" in parameters


def test_the_provenance_fragment_cannot_close_the_envelope_early(mapped):
    """A literal </parameters> inside a fragment truncates the document (#642)."""
    for ids in ("core_profiles", "core_transport"):
        parameters = str(mapped[f"{ids}.code.parameters"])
        assert parameters.endswith("</parameters>")
        assert "</parameters>" not in parameters[: -len("</parameters>")]


def test_mapping_a_second_slice_does_not_repeat_the_provenance(ods, native):
    for index in (0, 1):
        core_transport_from_neo(ods, native, time=0.3 + 0.1 * index, time_index=index)
    assert str(ods["core_transport.code.parameters"]).count("<neo>") == 1


# --------------------------------------------------------------------------
# Physics sanity, against an independent path
# --------------------------------------------------------------------------


@requires_sample
def test_the_mapped_current_agrees_with_the_analytic_model_in_magnitude():
    """NEO through the adapter, against Sauter computed from the same ODS.

    Two independent paths -- a drift-kinetic solve dimensionalised through
    out.neo.expnorm, and `vaft.formula.neoclassical` evaluated in SI on the
    packaged profiles. They are different physics models, so they are not
    expected to agree closely; landing within a factor of two, with the same
    sign and a peak at a similar radius, is what says the units are right.

    psi_N < 0.05 is excluded: the packaged sample's innermost surfaces are not
    trustworthy (#317).
    """
    from omas import load_omas_json

    from vaft.formula.neoclassical import (
        electron_collisionality_sauter,
        ion_collisionality_sauter,
        sauter_bootstrap_current,
    )

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    native = collect_neo_outputs(PROFILE_RUN)
    core_profiles_from_neo(ods, native, time=0.3, time_index=0)
    mapped_rho = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    mapped_current = np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"])

    eq = ods["equilibrium.time_slice.0.profiles_1d"]
    get = lambda key: np.asarray(eq[key], dtype=float)  # noqa: E731
    psi_rad = get("psi") / (2 * np.pi)
    rmin = 0.5 * (get("r_outboard") - get("r_inboard"))
    rmaj = 0.5 * (get("r_outboard") + get("r_inboard"))
    eps = np.divide(rmin, rmaj, out=np.zeros_like(rmin), where=rmaj > 0)
    cp = ods["core_profiles.profiles_1d.0"]
    ne = np.asarray(cp["electrons.density_thermal"], dtype=float)
    te = np.asarray(cp["electrons.temperature"], dtype=float)
    ti = np.asarray(cp["ion.0.temperature"], dtype=float)
    charge = 1.602176634e-19
    pe, pi_ = ne * te * charge, ne * ti * charge
    gradient = lambda y: np.gradient(y, psi_rad)  # noqa: E731

    usable = (ne > 1e15) & (te > 1) & (ti > 1) & (eps > 1e-3) & (get("rho_tor_norm") > 0.05)
    # Divide inside the usable band only: the fitted profiles reach zero at the
    # boundary, and a logarithmic gradient there is not a number.
    safe = lambda y: np.divide(gradient(y), y, out=np.zeros_like(y), where=usable)  # noqa: E731
    analytic = sauter_bootstrap_current(
        get("trapped_fraction")[usable],
        electron_collisionality_sauter(ne[usable], te[usable], get("q")[usable],
                                       rmaj[usable], eps[usable], 2.0),
        ion_collisionality_sauter(ne[usable], ti[usable], get("q")[usable],
                                  rmaj[usable], eps[usable], 1.0),
        2.0, get("f")[usable], pe[usable], pi_[usable],
        gradient(pe + pi_)[usable],
        safe(te)[usable], safe(ti)[usable],
    ) / abs(float(np.ravel(ods["equilibrium.vacuum_toroidal_field.b0"])[0]))

    analytic_rho = get("rho_tor_norm")[usable]
    band = mapped_rho > 0.3
    interpolated = np.interp(mapped_rho[band], analytic_rho, analytic)
    ratio = mapped_current[band] / interpolated
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), ratio
    # Both peak in the outer half, not at the axis.
    assert 0.35 < analytic_rho[np.argmax(analytic)] < 0.8
    assert 0.35 < mapped_rho[np.argmax(mapped_current)] < 0.8


# --------------------------------------------------------------------------
# End to end, against a real NEO build
# --------------------------------------------------------------------------

INSTALLED_GACODE_HOME = os.environ.get("GACODEHOME") or os.environ.get("GACODE_ROOT")
INSTALLED_GACODE_PLATFORM = os.environ.get("GACODE_PLATFORM")


@requires_sample
@pytest.mark.skipif(
    not INSTALLED_GACODE_HOME, reason="the end-to-end chain requires $GACODEHOME"
)
def test_the_whole_chain_runs_from_the_packaged_ods_to_an_ids(tmp_path):
    """ODS -> input.gacode -> NEO -> NeoOutputs -> core_profiles + core_transport.

    The fixtures make every other test in this file offline; this one proves the
    chain they stand in for actually runs, and that a freshly produced result
    maps to the same numbers as the stored one.
    """
    from omas import load_omas_json

    from vaft.code.gacode.inputs import prepare_gacode_profile
    from vaft.code.gacode.neo import NEOConfig, run_neo_case

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    profile = prepare_gacode_profile(ods, rho_max=0.95, z_eff=2.0)
    config = NEOConfig(
        home=INSTALLED_GACODE_HOME,
        platform=INSTALLED_GACODE_PLATFORM,
        n_radial=7,
        rmin_over_a=0.2,
        rmin_over_a_2=0.8,
    )
    result = run_neo_case(profile, tmp_path / "case", config)
    assert result.ok

    report = core_profiles_from_neo(ods, result, time=0.3, time_index=0)
    assert report["written"] == ["j_bootstrap"]
    assert core_transport_from_neo(ods, result, time=0.3, time_index=0)["written"]

    stored = collect_neo_outputs(PROFILE_RUN)
    np.testing.assert_allclose(
        np.asarray(result.outputs_native.bootstrap_current),
        np.asarray(stored.bootstrap_current),
        rtol=1e-6,
    )
    current = np.asarray(ods["core_profiles.profiles_1d.0.j_bootstrap"])
    rho = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"])
    assert np.all(current[rho > 0.3] > 0.0), "co-field: VEST has Ip parallel to Bt"
    assert np.max(np.abs(current)) < 1.0e6, "a kA/m^2-scale plasma, not MA/m^2"
