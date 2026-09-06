"""NUBEAM results as the fast-ion population in `distributions`.

No NUBEAM installation needed: the fixture is synthetic, with zone measures
and energies chosen so every conversion has an arithmetically checkable answer.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.nubeam.outputs import NUBEAMOutputs, NUBEAMRadialGrid
from vaft.machine_mapping.distributions import (
    NBI_PROCESS_INDEX,
    distributions_from_nubeam,
)

ZONES = 4
#: Joules per keV, as the mapping uses it.
KEV = 1.602176634e-16

BASE = "distributions.distribution.0.profiles_1d.0"
TOTALS = "distributions.distribution.0.global_quantities.0"


def _grid():
    # Enclosed volume rising by exactly 2 m^3 per zone and area by 0.5 m^2, so
    # a density is the per-zone value halved or doubled and can be read by eye.
    return NUBEAMRadialGrid(
        rho=np.linspace(0.0, 1.0, ZONES + 1),
        volume=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
        area=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )


@pytest.fixture
def outputs(tmp_path):
    return NUBEAMOutputs(
        workdir=tmp_path,
        runid="TESTRUN",
        profiles={
            # Species-resolved: (species, zone).
            "nbeami": np.array([[1e18, 2e18, 3e18, 4e18]]),
            "eperp_beami": np.array([[20.0, 20.0, 20.0, 20.0]]),
            "epll_beami": np.array([[10.0, 10.0, 10.0, 10.0]]),
            "sbtherm": np.array([[1e16, 2e16, 3e16, 4e16]]),
            # Already summed over species by NUBEAM: (zone,).
            "pbe": np.array([100.0, 200.0, 300.0, 400.0]),
            "pbi": np.array([10.0, 20.0, 30.0, 40.0]),
            "tqbe": np.array([0.1, 0.2, 0.3, 0.4]),
            "tqbi": np.array([0.01, 0.02, 0.03, 0.04]),
            "tqbjxb": np.array([0.001, 0.002, 0.003, 0.004]),
            "curbeam": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        grid=_grid(),
    )


@pytest.fixture
def two_species(tmp_path):
    return NUBEAMOutputs(
        workdir=tmp_path,
        runid="TWOSPEC",
        profiles={
            "nbeami": np.array([[1e18, 2e18, 3e18, 4e18], [5e17, 6e17, 7e17, 8e17]]),
            "eperp_beami": np.full((2, ZONES), 20.0),
            "epll_beami": np.full((2, ZONES), 10.0),
            "pbe": np.array([100.0, 200.0, 300.0, 400.0]),
            "curbeam": np.array([1.0, 2.0, 3.0, 4.0]),
        },
        grid=_grid(),
    )


# --------------------------------------------------------------------------
# What NUBEAM already has, IMAS gets unchanged
# --------------------------------------------------------------------------


def test_the_fast_ion_density_is_copied_not_divided(outputs):
    """`nbeami` is m^-3 in the Plasma State, which is what IMAS documents.

    Every other profile in this mapping is a per-zone integral needing a zone
    measure. Dividing this one as well would be wrong by the zone volume.
    """
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    assert np.allclose(ods[f"{BASE}.density_fast"], [1e18, 2e18, 3e18, 4e18])


def test_the_totals_are_exact_sums_of_the_per_zone_integrals(outputs):
    """`global_quantities` needs no division, so nothing here is derived."""
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    assert ods[f"{TOTALS}.collisions.electrons.power_thermal"] == pytest.approx(1000.0)
    assert ods[f"{TOTALS}.collisions.ion.0.power_thermal"] == pytest.approx(100.0)
    assert ods[f"{TOTALS}.collisions.electrons.torque_thermal_tor"] == pytest.approx(1.0)
    assert ods[f"{TOTALS}.torque_tor_j_radial"] == pytest.approx(0.01)
    assert ods[f"{TOTALS}.thermalisation.particles"] == pytest.approx(1e17)


def test_the_toroidal_current_survives_without_a_field_aligned_assumption(outputs):
    """This is the field `core_sources` does not have.

    There, NUBEAM's toroidal current has to be converted into `j_parallel`
    under an explicit assumption worth 53% on the validated VEST case. Here it
    is the native quantity: a sum for the total, a zone-area quotient for the
    density.
    """
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    assert ods[f"{TOTALS}.current_tor"] == pytest.approx(10.0)
    # Zone area is 0.5 m^2 throughout, so the density is twice the per-zone amps.
    assert np.allclose(ods[f"{BASE}.current_tor"], [2.0, 4.0, 6.0, 8.0])


def test_the_shielded_current_does_not_claim_to_be_the_unshielded_one(outputs):
    """`curbeam` is shielded, and IMAS separates the two by field name.

    `current_tor` includes the electron back-current, `current_fast_tor`
    excludes it. Writing the shielded value into the field that promises an
    unshielded one would understate the back-current by exactly the shielding.
    """
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    assert f"{BASE}.current_fast_tor" not in ods


# --------------------------------------------------------------------------
# The derived half
# --------------------------------------------------------------------------


def test_densities_divide_by_the_zone_measure_not_the_enclosed_one(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    # Zone volume is 2 m^3 throughout.
    assert np.allclose(
        ods[f"{BASE}.collisions.electrons.power_thermal"], [50.0, 100.0, 150.0, 200.0]
    )
    assert np.allclose(
        ods[f"{BASE}.thermalisation.particles"], [5e15, 1e16, 1.5e16, 2e16]
    )


def test_the_parallel_pressure_is_twice_the_parallel_energy_density(outputs):
    """p_par is the integral of m v_par^2 f, not of half of it.

    NUBEAM reports <E_par> as a mean energy per particle in keV -- the file's
    own units attribute says so -- so the pressure follows from the density
    and that mean, with the factor of two that the definition carries.
    """
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    expected = 2.0 * np.array([1e18, 2e18, 3e18, 4e18]) * 10.0 * KEV
    assert np.allclose(ods[f"{BASE}.pressure_fast_parallel"], expected)


def test_the_scalar_pressure_is_two_thirds_of_the_energy_density(outputs):
    """(p_par + 2 p_perp)/3, which is the isotropic-equivalent pressure."""
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    n = np.array([1e18, 2e18, 3e18, 4e18])
    expected = (2.0 / 3.0) * n * (10.0 + 20.0) * KEV
    assert np.allclose(ods[f"{BASE}.pressure_fast"], expected)

    # And the two pressures are consistent: p = (p_par + 2 p_perp)/3 with
    # p_perp the perpendicular energy density itself.
    p_par = np.asarray(ods[f"{BASE}.pressure_fast_parallel"])
    p_perp = n * 20.0 * KEV
    assert np.allclose(ods[f"{BASE}.pressure_fast"], (p_par + 2.0 * p_perp) / 3.0)


def test_the_stored_energy_reintegrates_from_the_pressures(outputs):
    """`energy_fast` is the volume integral of the same energy density."""
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    n = np.array([1e18, 2e18, 3e18, 4e18])
    expected = float(np.sum(n * 30.0 * KEV * 2.0))  # 2 m^3 per zone
    assert ods[f"{TOTALS}.energy_fast"] == pytest.approx(expected)
    assert ods[f"{TOTALS}.particles_fast_n"] == pytest.approx(float(np.sum(n * 2.0)))


# --------------------------------------------------------------------------
# Refusing rather than guessing
# --------------------------------------------------------------------------


def test_profiles_sit_at_zone_centres_not_boundaries(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    centres = np.asarray(ods[f"{BASE}.grid.rho_tor_norm"])
    assert centres.size == ZONES
    assert np.allclose(centres, [0.125, 0.375, 0.625, 0.875])


def test_an_absent_profile_is_skipped_not_zero_filled(outputs, tmp_path):
    thin = NUBEAMOutputs(
        workdir=tmp_path,
        runid="THIN",
        profiles={"nbeami": np.array([[1e18, 2e18, 3e18, 4e18]])},
        grid=_grid(),
    )
    ods = ODS()
    report = distributions_from_nubeam(ods, thin)

    assert f"{BASE}.pressure_fast_parallel" not in ods
    assert f"{TOTALS}.current_tor" not in ods
    assert any("pressure_fast_parallel" in entry for entry in report["skipped"])


def test_a_result_with_no_profiles_is_refused(tmp_path):
    empty = NUBEAMOutputs(workdir=tmp_path, runid="EMPTY", grid=_grid())
    with pytest.raises(ValueError, match="carries no profiles"):
        distributions_from_nubeam(ODS(), empty)


def test_a_result_with_no_grid_is_refused(tmp_path):
    ungridded = NUBEAMOutputs(
        workdir=tmp_path, runid="NOGRID", profiles={"nbeami": np.ones((1, ZONES))}
    )
    with pytest.raises(ValueError, match="carries no radial grid"):
        distributions_from_nubeam(ODS(), ungridded)


def test_an_unnamed_species_stays_unnamed(outputs):
    """NUBEAM's profiles carry no species label, so none is invented."""
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    assert "distributions.distribution.0.species.ion.label" not in ods


def test_a_named_species_is_recorded(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs, species=["D"])

    assert ods["distributions.distribution.0.species.ion.label"] == "D"


# --------------------------------------------------------------------------
# Species resolution
# --------------------------------------------------------------------------


def test_each_beam_species_gets_its_own_distribution(two_species):
    ods = ODS()
    report = distributions_from_nubeam(ods, two_species)

    assert report["distributions"] == [0, 1]
    assert np.allclose(
        ods["distributions.distribution.1.profiles_1d.0.density_fast"],
        [5e17, 6e17, 7e17, 8e17],
    )


def test_summed_profiles_are_skipped_when_species_are_resolved(two_species):
    """NUBEAM sums these over species, so no single entry may claim them.

    Repeating them into every entry would double-count for any consumer that
    adds the distributions up, which is the ordinary way to get a total.
    """
    ods = ODS()
    report = distributions_from_nubeam(ods, two_species)

    for index in (0, 1):
        stem = f"distributions.distribution.{index}"
        assert f"{stem}.global_quantities.0.current_tor" not in ods
        assert f"{stem}.profiles_1d.0.collisions.electrons.power_thermal" not in ods
    assert any("beam species" in entry for entry in report["skipped"])


# --------------------------------------------------------------------------
# Contract shared with every other mapping in this package
# --------------------------------------------------------------------------


def test_the_process_is_identified_as_nbi(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    stem = "distributions.distribution.0.process.0"
    assert ods[f"{stem}.type.index"] == NBI_PROCESS_INDEX
    # NUBEAM sums over injectors and energy components, which the data
    # dictionary spells as 0 in all three of these.
    assert ods[f"{stem}.nbi_unit"] == 0
    assert ods[f"{stem}.nbi_beamlets_group"] == 0
    assert ods[f"{stem}.nbi_energy.index"] == 0


def test_calling_twice_does_not_add_a_second_distribution(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs)
    first = len(ods["distributions.distribution"])
    distributions_from_nubeam(ods, outputs)

    assert len(ods["distributions.distribution"]) == first


def test_a_later_time_slice_does_not_overwrite_an_earlier_one(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs, time=0.1, time_index=0)
    distributions_from_nubeam(ods, outputs, time=0.2, time_index=1)

    assert ods[f"{BASE}.time"] == pytest.approx(0.1)
    assert ods["distributions.distribution.0.profiles_1d.1.time"] == pytest.approx(0.2)
    assert np.allclose(ods["distributions.time"], [0.1, 0.2])


def test_the_derived_quantities_are_recorded_as_derived(outputs):
    ods = ODS()
    distributions_from_nubeam(ods, outputs)

    parameters = ods["distributions.code.parameters"]
    assert "TESTRUN" in parameters
    assert "DERIVED" in parameters
    assert ods["distributions.code.name"] == "NUBEAM"
    assert ods["distributions.ids_properties.homogeneous_time"] == 1


def test_the_result_is_not_mutated(outputs):
    before = {name: np.array(values, copy=True) for name, values in outputs.profiles.items()}
    distributions_from_nubeam(ODS(), outputs)

    for name, values in before.items():
        assert np.array_equal(outputs.profiles[name], values)


def test_the_written_ods_survives_a_save_load_round_trip(outputs, tmp_path):
    ods = ODS()
    distributions_from_nubeam(ods, outputs, species=["D"])

    path = tmp_path / "distributions.json"
    ods.save(str(path))
    reloaded = ODS()
    reloaded.load(str(path))

    assert np.allclose(
        reloaded[f"{BASE}.density_fast"], ods[f"{BASE}.density_fast"]
    )
    assert reloaded[f"{TOTALS}.current_tor"] == pytest.approx(
        ods[f"{TOTALS}.current_tor"]
    )
