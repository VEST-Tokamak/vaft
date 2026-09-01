"""VAFT scientific display policy (issue #256).

IMAS owns canonical storage units; this module owns how those values are
*displayed*: which unit conversions exist per physical quantity, which display
unit a subject prefers, how scaled axes (``n_e [10^18 m^-3]``) and scientific
notation are chosen, and the canonical title / channel-label grammar.

Three separate concepts (issue #256):

``unit``
    the axis bracket text, e.g. ``kA`` or ``10^18 m^-3``;
``scale``
    the multiplier from IMAS canonical values to displayed values;
``notation``
    how tick values are written (``auto`` / ``plain`` / ``scientific`` /
    ``scaled_axis`` / ``percent``).

The contract enforced here: changing the display unit always changes the
numeric scaling and the label together, and an unsupported unit raises instead
of silently falling back to factor 1.

The policy this module implements is documented in
``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "DisplaySpec",
    "QUANTITIES",
    "QuantityDisplay",
    "SUBJECT_UNIT_DEFAULTS",
    "SUBJECT_NOTATION_DEFAULTS",
    "channel_label",
    "figure_title",
    "quantity_for_unit",
    "resolve_display",
    "subject_display_name",
]

NOTATIONS = ("auto", "plain", "scientific", "scaled_axis", "percent")

#: Torr in pascal.
_PA_PER_TORR = 133.322368


@dataclass(frozen=True)
class QuantityDisplay:
    """Conversion capability for one physical quantity.

    ``units`` maps each allowed display-unit label to the multiplier from the
    IMAS canonical unit; the canonical unit itself maps to ``1.0``.
    """

    name: str
    canonical_unit: str
    units: Mapping[str, float]
    default: str
    notation: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        if self.units.get(self.canonical_unit) != 1.0:
            raise ValueError(
                f"{self.name}: canonical unit {self.canonical_unit!r} must map to 1.0"
            )
        if self.default not in self.units:
            raise ValueError(
                f"{self.name}: default {self.default!r} is not an allowed unit"
            )
        if self.notation not in NOTATIONS:
            raise ValueError(f"{self.name}: unknown notation {self.notation!r}")


_QUANTITIES = (
    QuantityDisplay("current", "A", {"A": 1.0, "kA": 1e-3, "MA": 1e-6}, "kA"),
    QuantityDisplay(
        "current_turns", "A-turns", {"A-turns": 1.0, "kA-turns": 1e-3}, "kA-turns"
    ),
    QuantityDisplay("voltage", "V", {"V": 1.0, "mV": 1e3}, "V"),
    QuantityDisplay("magnetic_field", "T", {"T": 1.0, "mT": 1e3}, "mT"),
    QuantityDisplay("magnetic_flux", "Wb", {"Wb": 1.0, "mWb": 1e3}, "mWb"),
    QuantityDisplay("temperature", "eV", {"eV": 1.0, "keV": 1e-3}, "eV"),
    QuantityDisplay(
        "density",
        "m^-3",
        {"m^-3": 1.0, "10^18 m^-3": 1e-18, "10^19 m^-3": 1e-19},
        "10^18 m^-3",
        notation="scaled_axis",
    ),
    QuantityDisplay(
        "line_density",
        "m^-2",
        {"m^-2": 1.0, "10^18 m^-2": 1e-18},
        "10^18 m^-2",
        notation="scaled_axis",
    ),
    QuantityDisplay(
        "pressure",
        "Pa",
        {
            "Pa": 1.0,
            "mPa": 1e3,
            "kPa": 1e-3,
            "Torr": 1.0 / _PA_PER_TORR,
            "mTorr": 1e3 / _PA_PER_TORR,
        },
        "Pa",
    ),
    QuantityDisplay("energy", "J", {"J": 1.0, "kJ": 1e-3}, "J"),
    QuantityDisplay("power", "W", {"W": 1.0, "kW": 1e-3, "MW": 1e-6}, "kW"),
    QuantityDisplay(
        "current_density", "A/m^2", {"A/m^2": 1.0, "MA/m^2": 1e-6}, "MA/m^2"
    ),
    QuantityDisplay("velocity", "m/s", {"m/s": 1.0, "km/s": 1e-3}, "km/s"),
    QuantityDisplay("time", "s", {"s": 1.0, "ms": 1e3, "us": 1e6}, "s"),
)

#: Quantity name -> :class:`QuantityDisplay`.
QUANTITIES: dict[str, QuantityDisplay] = {q.name: q for q in _QUANTITIES}

_CANONICAL_UNIT_TO_QUANTITY = {}
for _q in _QUANTITIES:
    if _q.canonical_unit in _CANONICAL_UNIT_TO_QUANTITY:
        raise ValueError(f"ambiguous canonical unit {_q.canonical_unit!r}")
    _CANONICAL_UNIT_TO_QUANTITY[_q.canonical_unit] = _q.name
del _q

#: (subject, quantity) -> preferred display unit where it differs from the
#: quantity-wide default (issue #256: subject+quantity level preference).
SUBJECT_UNIT_DEFAULTS: dict[tuple[str, str], str] = {
    ("tf_coil", "magnetic_field"): "T",
    ("barometry", "pressure"): "Torr",
}

#: (subject, quantity) -> notation override.
SUBJECT_NOTATION_DEFAULTS: dict[tuple[str, str], str] = {
    ("barometry", "pressure"): "scientific",
}


#: Dimensionless quantities that still carry a display convention, keyed by
#: ``(subject, quantity)`` because the stored unit -- nothing at all -- cannot
#: tell ``beta_t`` from ``beta_p``.  Toroidal beta is conventionally read as a
#: percentage; poloidal beta, normalized beta, ``li`` and ``q`` are not, which
#: is why a beta family plot cannot put all three on one shared axis.
DIMENSIONLESS_DISPLAY: dict[tuple[str, str], tuple[str, float, str]] = {
    ("equilibrium", "beta_t"): ("%", 100.0, "percent"),
}


def quantity_for_unit(canonical_unit: str) -> str | None:
    """Map an IMAS canonical unit string to its display quantity, if any.

    Units outside the table (``a.u.``, ``Pa/Wb``, ``T m`` ...) return ``None``:
    they are pass-through quantities with no conversion offered.
    """
    return _CANONICAL_UNIT_TO_QUANTITY.get(canonical_unit)


@dataclass(frozen=True)
class DisplaySpec:
    """The resolved display choice for one plotted quantity.

    ``scale`` multiplies IMAS canonical values into displayed values; ``unit``
    is the matching axis bracket text.  Renderers must use both together.
    """

    quantity: str
    unit: str
    scale: float
    notation: str = "auto"


def _auto_unit(quantity: QuantityDisplay, data: Any) -> str:
    """Pick the allowed unit that puts the median |value| in [1, 1000)."""
    values = np.abs(np.asarray(data, dtype=float).ravel())
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return quantity.default
    magnitude = float(np.median(values))
    for unit, factor in sorted(quantity.units.items(), key=lambda item: -item[1]):
        scaled = magnitude * factor
        if 1.0 <= scaled < 1000.0:
            return unit
    return quantity.default


def resolve_display(
    canonical_unit: str,
    *,
    unit: str | None = None,
    subject: str | None = None,
    quantity: str | None = None,
    data: Any = None,
) -> DisplaySpec:
    """Resolve the display unit/scale/notation for one plotted quantity.

    ``canonical_unit`` is the IMAS storage unit of the data.  ``unit`` may be
    an explicit display unit (always wins), ``"auto"`` (magnitude-based within
    the allowed units, requires ``data``), or ``None`` for the subject/quantity
    default.  ``subject`` and ``quantity`` name the canonical plot identity and
    select the subject-specific preferences.  Unknown units raise
    :class:`ValueError` naming the alternatives; pass-through quantities accept
    only their canonical unit.
    """
    quantity_name = quantity_for_unit(canonical_unit)
    if quantity_name is None:
        convention = DIMENSIONLESS_DISPLAY.get((subject, quantity))
        if convention is not None and not canonical_unit:
            label, factor, notation = convention
            if unit not in (None, "auto", label):
                raise ValueError(
                    f"{subject}/{quantity} is displayed as {label!r}; "
                    f"got unit={unit!r}"
                )
            return DisplaySpec(
                quantity=quantity, unit=label, scale=factor, notation=notation
            )
        # Pass-through: no conversion table for this unit.
        if unit not in (None, "auto", canonical_unit):
            raise ValueError(
                f"no display conversions exist for unit {canonical_unit!r}; "
                f"got unit={unit!r}"
            )
        return DisplaySpec(
            quantity=canonical_unit, unit=canonical_unit, scale=1.0, notation="plain"
        )
    quantity = QUANTITIES[quantity_name]

    notation = quantity.notation
    if subject is not None:
        notation = SUBJECT_NOTATION_DEFAULTS.get((subject, quantity_name), notation)

    if unit is None:
        chosen = None
        if subject is not None:
            chosen = SUBJECT_UNIT_DEFAULTS.get((subject, quantity_name))
        chosen = chosen or quantity.default
    elif unit == "auto":
        if data is None:
            raise ValueError('unit="auto" requires the plotted data')
        chosen = _auto_unit(quantity, data)
    else:
        chosen = unit
    if chosen not in quantity.units:
        raise ValueError(
            f"unsupported display unit {chosen!r} for {quantity_name} "
            f"[{canonical_unit}]; supported units: "
            f"{', '.join(sorted(quantity.units))}"
        )
    return DisplaySpec(
        quantity=quantity_name,
        unit=chosen,
        scale=quantity.units[chosen],
        notation=notation,
    )


def subject_display_name(subject: str) -> str:
    """Human-readable subject name for titles: ``b_field_probe`` -> ``B field probe``."""
    text = subject.replace("_", " ")
    return text[:1].upper() + text[1:]


def figure_title(
    heading: str,
    unit: str | None,
    *,
    shot: str | int | None = None,
    time_s: float | None = None,
    coordinates: bool = False,
) -> str:
    """Canonical figure title: ``<Heading> [<unit>] #<shot>`` (issue #256).

    ``heading`` is used verbatim — humanize a subject with
    :func:`subject_display_name` first, so quantity tokens such as ``beta_p``
    keep their underscores.  ``time_s`` appends ``@ <t> s`` for selected-time
    views; ``coordinates`` appends the figure-level channel coordinate
    convention ``— (R [m], Z [m])``.
    """
    parts = [heading]
    if unit:
        parts.append(f"[{unit}]")
    if shot not in (None, ""):
        parts.append(f"#{shot}")
    title = " ".join(parts)
    if time_s is not None:
        title += f" @ {time_s:g} s"
    if coordinates:
        title += " — (R [m], Z [m])"
    return title


def channel_label(index: int, r: float | None = None, z: float | None = None) -> str:
    """Canonical multi-index channel label: ``[3] (0.82, 0.00)``."""
    if r is None or z is None:
        return f"[{index}]"
    return f"[{index}] ({r:.2f}, {z:.2f})"
