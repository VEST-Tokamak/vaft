r"""Unit scale factors: explicit, exact conversions between declared unit tags.

Formula and process docstrings declare every unit as a ``[unit]`` tag, and the
callers of a function that takes ``I_p [MA]`` or ``n [1e19 m^-3]`` convert by
hand. This module turns such a tag into a scale factor and a dimension, so a
conversion can be checked and stated instead of retyped (#2163).

It converts only what a unit tag can decide on its own. Two values with
convertible units are not thereby the same physical quantity: whether a
line-averaged and a separatrix density may be compared is a question of
quantity identity, answered by :mod:`vaft.data.representation`, never here.

Notation
--------
s    : scale of a unit in SI base units                                    [-]
d    : base-dimension exponents (L, M, T, I), as in vaft.formula.dimensional [-]
k    : exponent of the plane angle                                          [-]

Conventions
-----------
**Only the symbols in the table are known.** An unknown symbol raises; no
prefix is inferred, so ``ms`` is a millisecond only because it is listed.

**Angles are tracked, not erased.** The radian is dimensionless in SI, but a
flux in Wb and a flux in Wb/rad are two COCOS conventions, not one quantity in
two units: the factor between them is $2\pi$ only together with the sign
changes of the COCOS transform (Sauter and Medvedev 2013). The plane angle
therefore keeps its own exponent $k$, and two units whose $k$ differ never
convert.

**Percent is a scale.** ``%`` is the dimensionless unit with scale 0.01, so a
beta fraction and the same beta in percent convert. Normalized beta,
``% m T/MA``, is a different quantity with a different dimension.

References
----------
.. [1] BIPM, The International System of Units (SI), 9th edition (2019),
       Tables 2, 4, 7 and 8 (base units, derived units, prefixes, non-SI units
       accepted for use); the gauss is defined in Table 9 (CGS units).
.. [2] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013) 293,
       Table I (the poloidal-flux exponent e_Bp).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from .dimensional import BASE_DIMENSIONS

__all__ = [
    "UnitSpec",
    "UnknownUnitError",
    "IncompatibleUnitsError",
    "parse_unit",
    "unit_conversion_factor",
    "convert_unit",
]

_EV = 1.602176634e-19  # J, exact since the 2019 SI redefinition

#: Symbol -> (scale in SI, (L, M, T, I) exponents, plane-angle exponent).
_SYMBOLS: dict[str, tuple[float, tuple[int, int, int, int], int]] = {
    "-": (1.0, (0, 0, 0, 0), 0),
    "%": (1e-2, (0, 0, 0, 0), 0),
    "m": (1.0, (1, 0, 0, 0), 0),
    "cm": (1e-2, (1, 0, 0, 0), 0),
    "mm": (1e-3, (1, 0, 0, 0), 0),
    "kg": (1.0, (0, 1, 0, 0), 0),
    "s": (1.0, (0, 0, 1, 0), 0),
    "ms": (1e-3, (0, 0, 1, 0), 0),
    "us": (1e-6, (0, 0, 1, 0), 0),
    "A": (1.0, (0, 0, 0, 1), 0),
    "kA": (1e3, (0, 0, 0, 1), 0),
    "MA": (1e6, (0, 0, 0, 1), 0),
    "C": (1.0, (0, 0, 1, 1), 0),
    "J": (1.0, (2, 1, -2, 0), 0),
    "kJ": (1e3, (2, 1, -2, 0), 0),
    "MJ": (1e6, (2, 1, -2, 0), 0),
    "eV": (_EV, (2, 1, -2, 0), 0),
    "keV": (1e3 * _EV, (2, 1, -2, 0), 0),
    "W": (1.0, (2, 1, -3, 0), 0),
    "kW": (1e3, (2, 1, -3, 0), 0),
    "MW": (1e6, (2, 1, -3, 0), 0),
    "V": (1.0, (2, 1, -3, -1), 0),
    "kV": (1e3, (2, 1, -3, -1), 0),
    "Pa": (1.0, (-1, 1, -2, 0), 0),
    "kPa": (1e3, (-1, 1, -2, 0), 0),
    "T": (1.0, (0, 1, -2, -1), 0),
    "mT": (1e-3, (0, 1, -2, -1), 0),
    "G": (1e-4, (0, 1, -2, -1), 0),
    "Wb": (1.0, (2, 1, -2, -1), 0),
    "H": (1.0, (2, 1, -2, -2), 0),
    "Ohm": (1.0, (2, 1, -3, -2), 0),
    "rad": (1.0, (0, 0, 0, 0), 1),
    "deg": (math.pi / 180.0, (0, 0, 0, 0), 1),
}

_FACTOR = re.compile(r"^(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
_POWER = re.compile(r"^(?P<symbol>[A-Za-z%-]+)(?:\^(?P<exponent>[+-]?[0-9]+))?$")


class UnknownUnitError(ValueError):
    """A unit tag uses a symbol this module does not know; nothing is guessed."""


class IncompatibleUnitsError(ValueError):
    """Two units differ in dimension or in plane-angle exponent and cannot convert."""


@dataclass(frozen=True)
class UnitSpec:
    """A parsed unit: its SI scale, base-dimension exponents and angle exponent."""

    label: str
    scale: float
    dimensions: tuple[Fraction, Fraction, Fraction, Fraction]
    angle: Fraction = Fraction(0)

    @property
    def dimensionless(self) -> bool:
        """True when every base-dimension and angle exponent is zero."""
        return not any(self.dimensions) and not self.angle

    def same_dimension(self, other: "UnitSpec") -> bool:
        """True when both units have the same base-dimension and angle exponents."""
        return self.dimensions == other.dimensions and self.angle == other.angle


def _fold(spec: UnitSpec, symbol: str, exponent: Fraction) -> UnitSpec:
    try:
        scale, dims, angle = _SYMBOLS[symbol]
    except KeyError:
        raise UnknownUnitError(
            f"unknown unit symbol {symbol!r}; known symbols are {sorted(_SYMBOLS)}"
        ) from None
    return UnitSpec(
        label=spec.label,
        scale=spec.scale * scale ** float(exponent),
        dimensions=tuple(d + exponent * e for d, e in zip(spec.dimensions, dims)),
        angle=spec.angle + exponent * angle,
    )


def parse_unit(unit):
    r"""Parse a ``[unit]`` tag into its scale, dimension and angle exponent.

    $$u = f \prod_j s_j^{p_j},\qquad d(u) = \sum_j p_j\, d(s_j)$$

    A tag is an optional positive leading numeric factor ``f`` followed by
    symbols ``s_j`` with optional integer powers ``^p_j``, separated by
    spaces. A ``/`` divides every symbol after it, so ``% m T/MA`` is
    percent times metre times tesla per megaampere.

    Parameters
    ----------
    unit : str
        The tag without its brackets, e.g. ``"MA"``, ``"1e19 m^-3"`` [-].

    Returns
    -------
    UnitSpec
        Scale in SI, (L, M, T, I) exponents and the plane-angle exponent [-].

    Raises
    ------
    UnknownUnitError
        The tag is empty, malformed, or uses a symbol not in the table.

    Convention
    ----------
    No SI prefix is inferred: ``ms`` is the millisecond because the table
    lists it, and ``Mm`` is unknown. ``-`` and ``%`` are dimensionless with
    scales 1 and 0.01. ``rad`` and ``deg`` carry a plane-angle exponent of one,
    so ``Wb/rad`` keeps an angle exponent of -1 that ``Wb`` does not have.

    References
    ----------
    .. [1] BIPM, The International System of Units (SI), 9th edition (2019),
           Tables 2, 4, 7 and 8.
    """
    if not isinstance(unit, str) or not unit.strip():
        raise UnknownUnitError(f"a unit tag must be a non-empty string, got {unit!r}")
    spec = UnitSpec(label=unit.strip(), scale=1.0, dimensions=(Fraction(0),) * len(BASE_DIMENSIONS))
    tokens = unit.replace("/", " / ").split()
    if tokens and _FACTOR.match(tokens[0]):
        factor = float(tokens.pop(0))
        if not factor > 0:
            raise UnknownUnitError(f"unit tag {unit!r} has a non-positive factor")
        spec = UnitSpec(spec.label, factor, spec.dimensions, spec.angle)
    sign = 1
    seen = False
    for token in tokens:
        if token == "/":
            if sign < 0:
                raise UnknownUnitError(f"unit tag {unit!r} divides twice")
            sign = -1
            continue
        match = _POWER.match(token)
        if match is None:
            raise UnknownUnitError(f"cannot read {token!r} in unit tag {unit!r}")
        exponent = Fraction(match.group("exponent") or 1) * sign
        spec = _fold(spec, match.group("symbol"), exponent)
        seen = True
    if not seen or tokens[-1] == "/":
        raise UnknownUnitError(f"unit tag {unit!r} names no unit" if not seen else f"unit tag {unit!r} ends in '/'")
    return spec


def unit_conversion_factor(source, target):
    r"""Factor that turns a value in ``source`` units into ``target`` units.

    $$x_{\mathrm{target}} = \frac{s_{\mathrm{source}}}{s_{\mathrm{target}}}\, x_{\mathrm{source}}$$

    Parameters
    ----------
    source : str
        Unit tag of the value as held, e.g. ``"A"`` [-].
    target : str
        Unit tag the value is wanted in, e.g. ``"MA"`` [-].

    Returns
    -------
    float
        The multiplicative factor, 1e-6 for A to MA [-].

    Raises
    ------
    UnknownUnitError
        Either tag cannot be parsed.
    IncompatibleUnitsError
        The units differ in base dimension, or in plane-angle exponent.

    Convention
    ----------
    Only scale is converted. A unit change that a convention decides, such as
    poloidal flux in Wb against Wb/rad (COCOS 11-18 against 1-8), is refused
    with a pointer to :func:`vaft.process.equilibrium.convert_cocos`, because
    the factor $2\pi$ alone would leave the signs of the other convention.

    References
    ----------
    .. [1] BIPM, The International System of Units (SI), 9th edition (2019),
           Table 8 (the electronvolt) and Table 9 (the gauss).
    .. [2] O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184 (2013)
           293, Table I.
    """
    a = parse_unit(source)
    b = parse_unit(target)
    if a.dimensions != b.dimensions:
        raise IncompatibleUnitsError(f"{source!r} and {target!r} differ in dimension")
    if a.angle != b.angle:
        hint = (
            " Poloidal flux per radian against full flux is a COCOS convention; "
            "use vaft.process.equilibrium.convert_cocos."
            if a.dimensions == parse_unit("Wb").dimensions else ""
        )
        raise IncompatibleUnitsError(f"{source!r} and {target!r} differ in plane-angle exponent.{hint}")
    return a.scale / b.scale


def convert_unit(value, source, target):
    r"""Return ``value`` expressed in ``target`` units; the input is not modified.

    $$x_{\mathrm{target}} = c\, x_{\mathrm{source}},\qquad c = s_{\mathrm{source}}/s_{\mathrm{target}}$$

    Parameters
    ----------
    value : float or array_like
        The value as held, in ``source`` units [varies].
    source : str
        Unit tag of ``value`` [-].
    target : str
        Unit tag of the result [-].

    Returns
    -------
    float or numpy.ndarray
        ``value`` times :func:`unit_conversion_factor` [varies].

    Raises
    ------
    UnknownUnitError
        Either tag cannot be parsed.
    IncompatibleUnitsError
        The units cannot convert.

    Convention
    ----------
    A new value is returned, never an in-place rescale, so the caller's array
    keeps its original units. An array is returned as ``numpy.ndarray``.

    References
    ----------
    .. [1] BIPM, The International System of Units (SI), 9th edition (2019).
    """
    factor = unit_conversion_factor(source, target)
    if isinstance(value, (int, float)):
        return value * factor
    import numpy as np

    return np.asarray(value) * factor
