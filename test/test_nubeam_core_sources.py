"""NUBEAM results as a `core_sources` NBI source term.

No NUBEAM installation needed: the fixture is synthetic, with zone measures
chosen so every conversion has an arithmetically checkable answer.
"""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.code.nubeam.outputs import NUBEAMOutputs, NUBEAMRadialGrid
from vaft.machine_mapping.core_sources import (
    NBI_SOURCE_INDEX,
    core_sources_from_nubeam,
)

ZONES = 4
BASE = "core_sources.source.0.profiles_1d.0"


@pytest.fixture
def outputs(tmp_path):
    # Enclosed volume rising by exactly 2 m^3 per zone, area by 0.5 m^2, so a
    # density is the per-zone value halved (or doubled) and can be read by eye.
    return NUBEAMOutputs(
        workdir=tmp_path,
        runid="TESTRUN",
        profiles={
            "pbe": np.array([100.0, 200.0, 300.0, 400.0]),
            "pbi": np.array([10.0, 20.0, 30.0, 40.0]),
            "curbeam": np.array([1.0, 2.0, 3.0, 4.0]),
            "sbedep": np.array([1e16, 2e16, 3e16, 4e16]),
            "tqbe": np.array([0.1, 0.2, 0.3, 0.4]),
            "tqbi": np.array([0.01, 0.02, 0.03, 0.04]),
            "tqbjxb": np.array([0.001, 0.002, 0.003, 0.004]),
        },
        grid=NUBEAMRadialGrid(
            rho=np.linspace(0.0, 1.0, ZONES + 1),
            volume=np.array([0.0, 2.0, 4.0, 6.0, 8.0]),
            area=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        ),
    )


# --------------------------------------------------------------------------
# The cumulative fields are exact
# --------------------------------------------------------------------------


def test_cumulative_fields_are_a_running_sum_in_nubeams_own_units(outputs):
    """NUBEAM writes per-zone watts; `*_inside` is watts inside a surface, so
    the mapping is a cumsum with no conversion and nothing derived."""
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)

    np.testing.assert_allclose(
        ods[f"{BASE}.electrons.power_inside"], [100.0, 300.0, 600.0, 1000.0]
    )
    np.testing.assert_allclose(
        ods[f"{BASE}.total_ion_power_inside"], [10.0, 30.0, 60.0, 100.0]
    )
    np.testing.assert_allclose(
        ods[f"{BASE}.current_parallel_inside"], [1.0, 3.0, 6.0, 10.0]
    )


def test_the_last_cumulative_value_is_the_total(outputs):
    """This is the property that lets a run self-check against its step log."""
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    assert ods[f"{BASE}.electrons.power_inside"][-1] == pytest.approx(
        outputs.profiles["pbe"].sum()
    )


# --------------------------------------------------------------------------
# The density fields are the per-zone integrals over the zone measure
# --------------------------------------------------------------------------


def test_density_divides_by_the_zone_volume(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    # zone volume is 2 m^3 everywhere
    np.testing.assert_allclose(
        ods[f"{BASE}.electrons.energy"], [50.0, 100.0, 150.0, 200.0]
    )


def test_current_density_divides_by_the_zone_area_not_volume(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    # zone area is 0.5 m^2 everywhere
    np.testing.assert_allclose(ods[f"{BASE}.j_parallel"], [2.0, 4.0, 6.0, 8.0])


def test_density_reintegrates_to_the_cumulative_total(outputs):
    """The two forms must describe the same physics."""
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    density = np.asarray(ods[f"{BASE}.electrons.energy"])
    reintegrated = (density * outputs.grid.zone_volume).sum()
    assert reintegrated == pytest.approx(ods[f"{BASE}.electrons.power_inside"][-1])


# --------------------------------------------------------------------------
# Torque
# --------------------------------------------------------------------------


def test_collisional_torque_sums_electrons_and_ions_but_not_jxb(outputs):
    """JxB has its own IMAS field; adding it to momentum_tor would double it."""
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    expected = (outputs.profiles["tqbe"] + outputs.profiles["tqbi"]) / 2.0
    np.testing.assert_allclose(ods[f"{BASE}.momentum_tor"], expected)
    np.testing.assert_allclose(
        ods[f"{BASE}.momentum_tor_j_cross_b_field"], outputs.profiles["tqbjxb"] / 2.0
    )


def test_torque_inside_includes_every_channel(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    total = (
        outputs.profiles["tqbe"] + outputs.profiles["tqbi"] + outputs.profiles["tqbjxb"]
    ).sum()
    assert ods[f"{BASE}.torque_tor_inside"][-1] == pytest.approx(total)


# --------------------------------------------------------------------------
# Identity, grid and provenance
# --------------------------------------------------------------------------


def test_the_source_is_identified_as_nbi_by_the_dd_enumeration(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    assert ods["core_sources.source.0.identifier.index"] == NBI_SOURCE_INDEX == 2
    assert ods["core_sources.source.0.identifier.name"] == "nbi"


def test_profiles_sit_at_zone_centres_not_boundaries(outputs):
    """The grid has n+1 boundaries; a zone average belongs at the centre."""
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    rho = np.asarray(ods[f"{BASE}.grid.rho_tor_norm"])
    assert rho.size == ZONES
    np.testing.assert_allclose(rho, [0.125, 0.375, 0.625, 0.875])


def test_the_two_inexact_mappings_are_recorded(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    parameters = ods["core_sources.code.parameters"]
    assert "per_zone_to_density" in parameters
    assert "j_parallel" in parameters
    assert "average(J.B)/B0" in parameters
    assert ods["core_sources.code.name"] == "NUBEAM"


def test_calling_twice_does_not_add_a_second_nbi_source(outputs):
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    first = set(ods.flat().keys())
    core_sources_from_nubeam(ods, outputs)
    assert set(ods.flat().keys()) == first


# --------------------------------------------------------------------------
# Absence
# --------------------------------------------------------------------------


def test_an_absent_profile_is_skipped_not_zero_filled(outputs):
    """A hydrogen run has no fusion channels; nothing may invent them."""
    del outputs.profiles["curbeam"]
    ods = ODS()
    report = core_sources_from_nubeam(ods, outputs)

    assert any("curbeam" in entry for entry in report["skipped"])
    written = set(ods.flat().keys())
    assert not any("current_parallel_inside" in k for k in written)
    assert not any("j_parallel" in k for k in written)


def test_a_result_with_no_profiles_is_refused(tmp_path):
    ods = ODS()
    with pytest.raises(ValueError, match="state_changes"):
        core_sources_from_nubeam(ods, NUBEAMOutputs(workdir=tmp_path, profiles={}))


def test_no_grid_and_no_rho_is_refused_rather_than_guessed(outputs):
    outputs.grid = None
    ods = ODS()
    with pytest.raises(ValueError, match="radial grid"):
        core_sources_from_nubeam(ods, outputs)


def test_the_result_is_not_mutated(outputs):
    import copy

    before = copy.deepcopy(outputs.profiles["pbe"])
    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    np.testing.assert_array_equal(outputs.profiles["pbe"], before)


def test_the_written_ods_survives_a_save_load_round_trip(outputs, tmp_path):
    """A path that only looks present is the failure mode flat() hides."""
    from omas import load_omas_json, save_omas_json

    ods = ODS()
    core_sources_from_nubeam(ods, outputs)
    target = tmp_path / "cs.json"
    save_omas_json(ods, str(target))
    assert set(load_omas_json(str(target)).flat().keys()) == set(ods.flat().keys())
