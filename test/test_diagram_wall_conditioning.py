"""Wall conditioning (#1051): methods kept apart by what they draw, no quantity without its source."""

import numpy as np
import pytest

import vaft.diagram
from vaft.diagram import _wall_conditioning as wc
from vaft.diagram._scene import Arrow, Label


def _roles(diagram):
    return {getattr(item, "role", "") for item in diagram.scene.items}


def _labels(diagram, role=None):
    return [i.text for i in diagram.scene.items if isinstance(i, Label) and (role is None or i.role == role)]


def test_baking_is_thermal_only():
    roles = _roles(vaft.diagram.wall_conditioning_baking())
    assert {"heat", "desorption", "pump_port"} <= roles
    assert not roles & {"glow", "anode", "cathode", "ion_flux", "deposition_plasma", "boron_layer"}


def test_every_gdc_shares_the_apparatus_geometry():
    # geometry, not labels: the template survives labels=False
    template = {"glow", "anode", "cathode", "gas_inlet", "inlet_port", "ion_flux", "pump_port"}
    for gas in wc.GASES:
        diagram = vaft.diagram.wall_conditioning_gdc(gas, labels=False)
        assert template <= _roles(diagram)
        assert "boron_layer" not in _roles(diagram)
        assert diagram.name == "wall_conditioning_gdc"


def test_ions_cross_the_sheath_onto_the_wall():
    diagram = vaft.diagram.wall_conditioning_gdc("D2")
    ions = [i for i in diagram.scene.items if isinstance(i, Arrow) and i.role == "ion_flux"]
    assert len(ions) >= 6
    def to_wall(p):
        return min(p[0], wc._W - p[0], p[1], wc._H - p[1])

    for arrow in ions:
        start, end = np.asarray(arrow.start), np.asarray(arrow.end)
        # each arrow starts at the glow edge, runs towards the wall and ends at its surface
        assert to_wall(end) < to_wall(start)
        assert to_wall(end) < wc._T + 0.05


def test_he_gdc_releases_retained_hydrogen_and_hydrogen_gdc_does_not():
    he = vaft.diagram.wall_conditioning_gdc("He")
    d2 = vaft.diagram.wall_conditioning_gdc("D2")
    assert "released_hydrogen" in _roles(he) and "volatile_product" not in _roles(he)
    assert "volatile_product" in _roles(d2) and "released_hydrogen" not in _roles(d2)
    assert _labels(he, "released_hydrogen") == ["H/D"]
    # a different arrow style, not only different words
    style = {i.role: i.style for i in he.scene.items + d2.scene.items if isinstance(i, Arrow)}
    assert style["released_hydrogen"] != style["volatile_product"]
    assert he.model["state_change"] != d2.model["state_change"]


def test_reactive_products_follow_the_feed_isotope():
    assert _labels(vaft.diagram.wall_conditioning_gdc("D2"), "volatile_product") == ["D$_2$O, CD$_4$"]
    assert _labels(vaft.diagram.wall_conditioning_gdc("H2"), "volatile_product") == ["H$_2$O, CH$_4$"]


def test_boronization_deposits_and_names_no_precursor_unless_given():
    generic = vaft.diagram.wall_conditioning_boronization()
    assert {"deposition_plasma", "deposition", "boron_layer", "precursor_inlet"} <= _roles(generic)
    text = " ".join(_labels(generic))
    assert "B-containing" in text and "B$_2$H$_6$" not in text and "thickness" not in _roles(generic)
    named = vaft.diagram.wall_conditioning_boronization(precursor="B$_2$H$_6$")
    assert "B$_2$H$_6$" in _labels(named, "precursor_inlet")
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_boronization(precursor="")


def test_numbers_need_their_source_and_unit():
    with pytest.raises(ValueError, match="source"):
        vaft.diagram.wall_conditioning_baking(temperature=420.0)
    with pytest.raises(ValueError, match="source"):
        vaft.diagram.wall_conditioning_boronization(thickness_nm=50.0)
    with pytest.raises(ValueError, match="without"):
        vaft.diagram.wall_conditioning_baking(temperature_source="machine log")
    with pytest.raises(ValueError, match="positive"):
        vaft.diagram.wall_conditioning_baking(temperature="hot", temperature_source="log")
    with pytest.raises(ValueError, match="temperature_unit"):
        vaft.diagram.wall_conditioning_baking(temperature=420.0, temperature_unit="F", temperature_source="log")
    assert "temperature" not in _roles(vaft.diagram.wall_conditioning_baking())
    baked = vaft.diagram.wall_conditioning_baking(temperature=150.0, temperature_unit="degC",
                                                  temperature_source="machine log")
    (text,) = _labels(baked, "temperature")
    assert "150" in text and "circ" in text and "machine log" in text
    coated = vaft.diagram.wall_conditioning_boronization(thickness_nm=50.0, thickness_source="QCM")
    assert "QCM" in _labels(coated, "thickness")[0]


def test_the_sequence_composes_the_single_stages():
    stages = ("baking", "D2_gdc", "He_gdc", "boronization")
    seq = vaft.diagram.wall_conditioning_sequence(stages)
    assert seq.model["stages"] == stages
    assert seq.model["state_changes"] == tuple(wc.WALL_STATE_CHANGE[s] for s in stages)
    assert len(_labels(seq, "transition")) == len(stages) - 1
    assert "plasma_operation" in _roles(seq)
    # each panel is the single-stage diagram itself
    singles = {"baking": vaft.diagram.wall_conditioning_baking(),
               "D2_gdc": vaft.diagram.wall_conditioning_gdc("D2"),
               "He_gdc": vaft.diagram.wall_conditioning_gdc("He"),
               "boronization": vaft.diagram.wall_conditioning_boronization()}
    for stage in stages:
        assert wc._stage_items(stage, 0.0, True) == list(singles[stage].scene.items)
    # panels do not overlap: every panel's geometry stays within its own pitch
    for k, stage in enumerate(stages):
        x0 = k * wc.SEQUENCE_PITCH
        xs = [x for item in wc._stage_items(stage, x0, True) if hasattr(item, "points") for x, _ in item.points]
        assert min(xs) > x0 - 1.3 and max(xs) < x0 + wc._W + 1.3
    assert "plasma_operation" not in _roles(vaft.diagram.wall_conditioning_sequence(plasma_operation=False))


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_gdc("Ar")
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_sequence(())
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_sequence(("baking", "D2"))
    with pytest.raises(ValueError):
        vaft.diagram.wall_conditioning_baking(labels="yes")


def test_labels_off_leaves_only_geometry():
    for diagram in (vaft.diagram.wall_conditioning_baking(labels=False),
                    vaft.diagram.wall_conditioning_gdc("He", labels=False),
                    vaft.diagram.wall_conditioning_boronization(labels=False),
                    vaft.diagram.wall_conditioning_sequence(labels=False)):
        assert not _labels(diagram)
