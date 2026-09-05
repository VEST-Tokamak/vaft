"""VEST NBI geometry as the `nbi` IDS.

Each test here pins one conversion where the obvious reading is wrong, because
these are the failures that would produce a plausible-looking but incorrect
beam model rather than an error.
"""

from __future__ import annotations

import math

import pytest
from omas import ODS

from vaft.machine_mapping.nbi import nbi, nbi_run_conditions

GROUP = "nbi.unit.0.beamlets_group.0"


@pytest.fixture
def ods():
    out = ODS()
    nbi(out)
    return out


# --------------------------------------------------------------------------
# Tangency radius and injection direction
# --------------------------------------------------------------------------


def test_tangency_radius_is_a_magnitude(ods):
    """NUBEAM signs srtcen; IMAS documents tangency_radius as a major radius."""
    assert ods[f"{GROUP}.tangency_radius"] == pytest.approx(0.22129)
    assert ods[f"{GROUP}.tangency_radius"] > 0


def test_the_injection_sense_is_kept_not_discarded(ods):
    """Taking the magnitude is only correct because the sense goes here."""
    assert ods[f"{GROUP}.direction"] == -1


# --------------------------------------------------------------------------
# Widths: mdescr gives half, IMAS wants full
# --------------------------------------------------------------------------


def test_source_widths_are_doubled_to_full_width(ods):
    # mdescr: half_width 0.06, half_height 0.21
    assert ods[f"{GROUP}.width_horizontal"] == pytest.approx(0.12)
    assert ods[f"{GROUP}.width_vertical"] == pytest.approx(0.42)


def test_aperture_widths_are_doubled_too(ods):
    # mdescr: ap half 0.094646 x 0.2317
    assert ods["nbi.unit.0.aperture.0.x1_width"] == pytest.approx(0.189292)
    assert ods["nbi.unit.0.aperture.0.x2_width"] == pytest.approx(0.4634)


# --------------------------------------------------------------------------
# Divergence: one Gaussian component, both axes
# --------------------------------------------------------------------------


def test_divergence_is_one_component_carrying_both_axes(ods):
    """IMAS components are populations with a particle fraction each. One
    component per axis would assert two populations of 100% of the beam."""
    written = {k for k in ods.flat() if "divergence_component" in k}
    indices = {k.split("divergence_component.")[1].split(".")[0] for k in written}
    assert indices == {"0"}

    assert ods[f"{GROUP}.divergence_component.0.horizontal"] == pytest.approx(
        math.radians(1.0)
    )
    assert ods[f"{GROUP}.divergence_component.0.vertical"] == pytest.approx(
        math.radians(1.0)
    )
    assert ods[f"{GROUP}.divergence_component.0.particles_fraction"] == 1.0


def test_angles_are_radians_not_degrees(ods):
    assert ods[f"{GROUP}.divergence_component.0.horizontal"] < 0.1
    assert ods[f"{GROUP}.position.phi"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# What is deliberately absent
# --------------------------------------------------------------------------


def test_an_unfocused_beam_gets_no_focal_length(ods):
    """NUBEAM spells 'unfocused' as 1.2e11 m; writing that would be a
    hundred-million-kilometre focal length."""
    assert not any("focus" in key for key in ods.flat())


def test_the_static_mapping_carries_no_beam_energy_or_power(ods):
    """Those are per-case modelling inputs, not machine description (#490 s5)."""
    keys = set(ods.flat())
    assert not any("energy" in k for k in keys)
    assert not any("power_launched" in k for k in keys)


def test_the_absences_are_reported_with_reasons():
    out = ODS()
    report = nbi(out)
    joined = " ".join(report["absent"])
    assert "unfocused" in joined
    assert "modelling inputs" in joined


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_says_this_is_not_as_built(ods):
    comment = ods["nbi.ids_properties.comment"]
    assert "NUBEAM" in comment
    assert "Not as-built" in comment
    assert "#265" in comment


def test_species_is_hydrogen(ods):
    assert ods["nbi.unit.0.species.label"] == "H"
    assert ods["nbi.unit.0.species.a"] == pytest.approx(1.0)
    assert ods["nbi.unit.0.species.z_n"] == pytest.approx(1.0)


def test_calling_twice_does_not_add_a_second_unit(ods):
    first = set(ods.flat())
    nbi(ods)
    assert set(ods.flat()) == first


# --------------------------------------------------------------------------
# Run conditions, kept separate from the machine description
# --------------------------------------------------------------------------


class _Native:
    runid = "TESTRUN"
    beam_conditions = {"energy_keV": 10.0, "power_W": 200000.0, "power_fractions": [1.0, 0.0, 0.0]}


def test_run_conditions_convert_kev_to_ev(ods):
    nbi_run_conditions(ods, _Native())
    assert ods["nbi.unit.0.energy.data"][0] == pytest.approx(10000.0)
    assert ods["nbi.unit.0.power_launched.data"][0] == pytest.approx(200000.0)


def test_power_fractions_are_full_half_third(ods):
    nbi_run_conditions(ods, _Native())
    fractions = ods["nbi.unit.0.beam_power_fraction.data"]
    assert len(fractions) == 3
    assert fractions[0][0] == pytest.approx(1.0)
    assert fractions[1][0] == pytest.approx(0.0)


def test_a_result_without_conditions_is_refused(ods):
    class Empty:
        beam_conditions = {}

    with pytest.raises(ValueError, match="beam conditions"):
        nbi_run_conditions(ods, Empty())


def test_the_written_ods_survives_a_save_load_round_trip(ods, tmp_path):
    from omas import load_omas_json, save_omas_json

    target = tmp_path / "nbi.json"
    save_omas_json(ods, str(target))
    assert set(load_omas_json(str(target)).flat().keys()) == set(ods.flat().keys())
