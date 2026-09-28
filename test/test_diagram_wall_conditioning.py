"""Wall conditioning (#1051): methods kept apart by what they draw, no quantity without its source."""

import pytest

import vaft.diagram
from vaft.diagram import _wall_conditioning as wc
from vaft.diagram._scene import Label


def _roles(diagram):
    return {getattr(item, "role", "") for item in diagram.scene.items}


def _text(diagram):
    return " ".join(item.text for item in diagram.scene.items if isinstance(item, Label))


def test_baking_is_thermal_only():
    roles = _roles(vaft.diagram.wall_conditioning_baking())
    assert {"heat", "desorption", "pump_port"} <= roles
    # no glow, anode, ion bombardment or coating
    assert not roles & {"glow", "anode", "ion_flux", "deposition_plasma", "boron_layer"}


def test_every_gdc_shares_the_apparatus_template():
    template = {"glow", "anode", "cathode", "gas_inlet", "ion_flux", "pump_port"}
    for gas in wc.GASES:
        assert template <= _roles(vaft.diagram.wall_conditioning_gdc(gas))


def test_he_gdc_releases_retained_hydrogen_and_hydrogen_gdc_does_not():
    he = vaft.diagram.wall_conditioning_gdc("He")
    d2 = vaft.diagram.wall_conditioning_gdc("D2")
    assert "released_hydrogen" in _roles(he) and "volatile_product" not in _roles(he)
    assert "volatile_product" in _roles(d2) and "released_hydrogen" not in _roles(d2)
    assert "H/D" in _text(he) and "He$^+$" in _text(he)
    assert "D$^+$" in _text(d2)
    assert he.model["state_change"] != d2.model["state_change"]


def test_boronization_names_no_precursor_unless_given():
    generic = vaft.diagram.wall_conditioning_boronization()
    assert "B-containing precursor" in _text(generic) and "B$_2$H$_6$" not in _text(generic)
    assert "boron_layer" in _roles(generic) and "thickness" not in _roles(generic)
    named = vaft.diagram.wall_conditioning_boronization(precursor="B$_2$H$_6$")
    assert "B$_2$H$_6$" in _text(named)


def test_numbers_need_their_source():
    with pytest.raises(ValueError, match="source"):
        vaft.diagram.wall_conditioning_baking(temperature=150.0)
    with pytest.raises(ValueError, match="source"):
        vaft.diagram.wall_conditioning_boronization(thickness_nm=50.0)
    assert "temperature" not in _roles(vaft.diagram.wall_conditioning_baking())
    baked = vaft.diagram.wall_conditioning_baking(temperature=150.0, temperature_source="machine log")
    assert "machine log" in _text(baked) and baked.model["temperature"] == 150.0
    coated = vaft.diagram.wall_conditioning_boronization(thickness_nm=50.0, thickness_source="QCM")
    assert "thickness" in _roles(coated) and "QCM" in _text(coated)


def test_the_sequence_composes_the_single_methods():
    steps = ("baking", "D2", "He", "boronization")
    seq = vaft.diagram.wall_conditioning_sequence(steps)
    assert seq.model["steps"] == steps
    assert seq.model["state_changes"] == tuple(wc.WALL_STATE_CHANGE[s] for s in steps)
    transitions = [i for i in seq.scene.items if getattr(i, "role", "") == "transition" and isinstance(i, Label)]
    assert len(transitions) == len(steps) - 1
    roles = _roles(seq)
    assert {"heat", "released_hydrogen", "volatile_product", "boron_layer"} <= roles


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_gdc("Ar")
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_sequence(())
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_sequence(("baking", "bake"))
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_baking(labels="yes")


def test_labels_off_leaves_only_geometry():
    for diagram in (vaft.diagram.wall_conditioning_baking(labels=False),
                    vaft.diagram.wall_conditioning_gdc("He", labels=False),
                    vaft.diagram.wall_conditioning_boronization(labels=False),
                    vaft.diagram.wall_conditioning_sequence(labels=False)):
        assert not [i for i in diagram.scene.items if isinstance(i, Label)]
