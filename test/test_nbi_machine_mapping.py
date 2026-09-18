"""VEST NBI geometry as the `nbi` IDS.

Each test here pins one conversion where the obvious reading is wrong, because
these are the failures that would produce a plausible-looking but incorrect
beam model rather than an error.
"""

from __future__ import annotations

import math

import numpy as np
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
    assert ods[f"{GROUP}.position.phi"] < 2 * math.pi


def test_the_position_is_the_source_not_the_port(ods):
    """IMAS ``beamlets_group.position`` is the beamlet-group (source) centre.

    The port, 2MR at 300 deg, is where the beam line crosses the vessel; it was
    written as ``position.phi`` with no ``position.r`` at all (cold review
    machine-mapping F4).  The source lies upstream of it on the same line.
    """
    r = ods[f"{GROUP}.position.r"]
    phi = ods[f"{GROUP}.position.phi"]
    tangency = ods[f"{GROUP}.tangency_radius"]
    assert r == pytest.approx(math.hypot(2.625867, 0.22129))
    # Upstream of a clockwise beam is larger phi: 300 deg + 11.3 deg.
    assert math.degrees(phi) == pytest.approx(311.3, abs=0.1)

    # The line from the source along the injection sense must pass the machine
    # axis at the tangency radius and cross R = 0.8 m at the port.
    source = np.array([r * math.cos(phi), r * math.sin(phi)])
    # unit vector toward the tangency point: clockwise (direction -1) from above
    angle_to_tangency = phi + ods[f"{GROUP}.direction"] * math.atan2(2.625867, tangency)
    foot = tangency * np.array([math.cos(angle_to_tangency), math.sin(angle_to_tangency)])
    along = (foot - source) / np.linalg.norm(foot - source)
    assert np.linalg.norm(foot - source) == pytest.approx(2.625867)
    assert float(np.dot(along, foot)) == pytest.approx(0.0, abs=1e-9)
    crossing = foot - math.sqrt(0.8**2 - tangency**2) * along
    assert np.linalg.norm(crossing) == pytest.approx(0.8)
    assert math.degrees(math.atan2(crossing[1], crossing[0])) % 360.0 == pytest.approx(300.0)


def test_a_source_that_cannot_be_placed_is_reported_not_put_at_the_port(monkeypatch):
    import vaft.machine_mapping.nbi as module

    document = module.load_yaml(module.package_data_path("vest.yaml"))
    del document[0]["nbi"]["unit"][0]["source"]["port_radius"]
    monkeypatch.setattr(module, "load_yaml", lambda _path: document)
    out = ODS()
    report = nbi(out)
    assert not any(key.endswith(("position.phi", "position.r")) for key in out.flat())
    assert "position.r and position.phi" in " ".join(report["absent"])


def test_the_beam_still_runs_in_the_negative_toroidal_direction(ods):
    """The port and the NUBEAM-derived direction are independent and agree.

    2 o'clock to 7 o'clock is toward increasing clock number, which is
    clockwise from above, which is negative phi.  If placing the source at its
    port ever flips this, one of the two facts has been misread.
    """
    assert ods[f"{GROUP}.direction"] == -1


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
