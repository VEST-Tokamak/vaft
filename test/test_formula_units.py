"""Unit scale factors of vaft.formula.units (#2163): exact, explicit, never guessed."""
from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from vaft.formula.units import (
    IncompatibleUnitsError,
    UnknownUnitError,
    convert_unit,
    parse_unit,
    unit_conversion_factor,
)


@pytest.mark.parametrize("source, target, factor", [
    ("A", "MA", 1e-6),
    ("MA", "kA", 1e3),
    ("W", "MW", 1e-6),
    ("T", "G", 1e4),
    ("keV", "J", 1e3 * 1.602176634e-19),
    ("eV", "keV", 1e-3),
    ("m^-3", "1e19 m^-3", 1e-19),
    ("1e20 m^-3", "1e19 m^-3", 10.0),
    ("%", "-", 1e-2),
    ("ms", "s", 1e-3),
    ("deg", "rad", math.pi / 180),
    ("% m T/MA", "m T/A", 1e-8),
])
def test_engineering_unit_factors(source, target, factor):
    assert unit_conversion_factor(source, target) == pytest.approx(factor, rel=1e-15)
    assert unit_conversion_factor(target, source) == pytest.approx(1 / factor, rel=1e-15)


def test_a_compound_tag_keeps_its_dimension():
    spec = parse_unit("% m T/MA")
    assert spec.scale == pytest.approx(1e-8)
    assert spec.dimensions == parse_unit("T m/A").dimensions


@pytest.mark.parametrize("source, target", [("A", "T"), ("m^-3", "m^-2"), ("MW", "MJ"), ("-", "m")])
def test_different_dimensions_never_convert(source, target):
    with pytest.raises(IncompatibleUnitsError, match="dimension"):
        unit_conversion_factor(source, target)


def test_flux_per_radian_is_a_cocos_question_not_a_scale():
    with pytest.raises(IncompatibleUnitsError, match="convert_cocos"):
        unit_conversion_factor("Wb/rad", "Wb")


def test_an_angle_is_not_silently_dimensionless():
    with pytest.raises(IncompatibleUnitsError, match="angle"):
        unit_conversion_factor("rad", "-")


@pytest.mark.parametrize("tag", ["", "varies", "any", "Mm", "1e19", "m^x", "A / / s", "furlong"])
def test_unknown_or_malformed_tags_raise_instead_of_guessing(tag):
    with pytest.raises(UnknownUnitError):
        parse_unit(tag)


def test_convert_unit_returns_a_new_value():
    held = np.array([1.0e5, 2.0e5])
    converted = convert_unit(held, "A", "MA")
    np.testing.assert_allclose(converted, [0.1, 0.2])
    np.testing.assert_array_equal(held, [1.0e5, 2.0e5])
    assert convert_unit(5.0, "%", "-") == pytest.approx(0.05)


def test_the_greenwald_kernel_is_unchanged_by_the_unit_layer():
    from vaft.formula.stability import greenwald_density

    ip_amperes = 1.0e5
    in_ma = convert_unit(ip_amperes, "A", "MA")
    # greenwald_density still takes MA and returns 1e19 m^-3, exactly as before.
    assert greenwald_density(in_ma, 0.3) == pytest.approx(10 * 0.1 / (math.pi * 0.3**2))


def test_the_unit_module_stays_out_of_the_star_namespace_and_is_light():
    code = (
        "import sys\n"
        "from vaft.formula import *\n"
        "assert 'vaft.formula.units' not in sys.modules\n"
        "import vaft.formula.units\n"
        "assert 'pint' not in sys.modules\n"
    )
    # -P: the interpreter must import this checkout, not whatever the current directory holds.
    subprocess.run([sys.executable, "-P", "-c", code], check=True)
