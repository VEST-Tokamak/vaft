"""Plasma-wall interaction (#1047): definitions evaluated, species kept apart, nothing fabricated."""

import re

import pytest

import vaft.diagram
from vaft.diagram import _pwi as pw
from vaft.diagram._scene import Label
from vaft.formula.pwi import (
    binary_collision_energy_transfer_factor,
    mean_reflected_energy_fraction,
    recycling_coefficient,
    sputtering_threshold_bohdansky,
)


def test_the_energy_transfer_factor_is_the_elastic_head_on_value():
    assert binary_collision_energy_transfer_factor(1.0, 1.0) == pytest.approx(1.0)
    assert binary_collision_energy_transfer_factor(2.014, 183.84) == pytest.approx(0.0428, abs=1e-4)
    # symmetric in the two masses, below one otherwise
    assert binary_collision_energy_transfer_factor(2, 12) == pytest.approx(binary_collision_energy_transfer_factor(12, 2))
    with pytest.raises(ValueError):
        binary_collision_energy_transfer_factor(0.0, 1.0)


def test_reflection_and_recycling_definitions():
    assert mean_reflected_energy_fraction(0.5, 0.2) == pytest.approx(0.4)
    for bad in ((0.0, 0.0), (0.5, 0.6), (1.2, 0.1)):
        with pytest.raises(ValueError):
            mean_reflected_energy_fraction(*bad)
    assert recycling_coefficient(0.3, 0.6, 1.0) == pytest.approx(0.9)
    assert recycling_coefficient(0.3, 0.9, 1.0) > 1.0  # an outgassing wall returns more than it gets
    with pytest.raises(ValueError):
        recycling_coefficient(-0.1, 0.5, 1.0)


def test_the_bohdansky_threshold_for_d_on_w_is_the_known_order():
    # E_s(W) = 8.68 eV: D on W threshold ~ 200-230 eV in the literature; heavy self-sputtering branch 8 E_s
    assert 190.0 < sputtering_threshold_bohdansky(8.68, 2.014, 183.84) < 240.0
    assert sputtering_threshold_bohdansky(8.68, 183.84, 183.84) == pytest.approx(8 * 8.68)
    gamma = binary_collision_energy_transfer_factor(2.014, 183.84)
    assert sputtering_threshold_bohdansky(8.68, 2.014, 183.84) == pytest.approx(8.68 / (gamma * (1 - gamma)))


def test_the_sputtering_diagram_needs_the_binding_energy_for_a_threshold():
    without = vaft.diagram.plasma_wall_interaction_sputtering().model
    assert without["threshold_eV"] is None and without["gamma"] == pytest.approx(0.0428, abs=1e-4)
    with_es = vaft.diagram.plasma_wall_interaction_sputtering(surface_binding_energy=8.68).model
    assert with_es["threshold_eV"] == pytest.approx(sputtering_threshold_bohdansky(8.68, pw.MASS_U["D"], pw.MASS_U["W"]))


def test_species_are_checked_and_kept_apart():
    m = vaft.diagram.plasma_wall_interaction_processes("deuterium", "tungsten").model
    assert (m["projectile"], m["target"]) == ("D", "W")
    d = vaft.diagram.plasma_wall_interaction_processes("D", "W")
    projectile = [i for i in d.scene.items if i.role in ("implanted", "molecule")]
    target = [i for i in d.scene.items if i.role == "sputtered_atom"]
    assert all(i.style == "opoint" for i in projectile) and all(i.style == "xpoint" for i in target)
    with pytest.raises(ValueError):
        vaft.diagram.plasma_wall_interaction_processes("unobtainium", "W")


def test_no_coefficient_is_drawn():
    for name in ("plasma_wall_interaction_reflection", "plasma_wall_interaction_recycling",
                 "plasma_wall_interaction_energy_partition"):
        text = " ".join(i.text for i in getattr(vaft.diagram, name)().scene.items
                        if isinstance(i, Label) and i.role != "equations")
        # no coefficient value: no decimal number and no percentage outside the defining equations
        assert not re.search(r"\d\.\d|\d\s*\\?%", text), name


@pytest.mark.parametrize("name", ["plasma_wall_interaction_processes", "plasma_wall_interaction_reflection",
                                  "plasma_wall_interaction_sputtering", "plasma_wall_interaction_recycling",
                                  "plasma_wall_interaction_energy_partition"])
def test_every_pwi_diagram_is_deterministic_and_exported(name):
    fn = getattr(vaft.diagram, name)
    assert fn().tikz == fn().tikz
    assert name in vaft.diagram.__all__
    assert fn().scene.role("note") and not fn(labels=False).scene.role("note")
    with pytest.raises(ValueError):
        fn(labels="yes")
