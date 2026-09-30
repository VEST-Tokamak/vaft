"""
Operational boundaries from the literature: one data model for density, transition and stability limits.

A published limit is more than its formula. It predicts one specific
quantity (line-averaged density, not "the density"). It has a permitted side.
It was fitted on some machines under some assumptions. It may or may not carry
a quoted uncertainty. And it comes from one paper and equation. A
:class:`Boundary` keeps all of that next to the evaluation rule. Then
:func:`evaluate_boundary` can report how far a state sits from the boundary,
on which side, and whether the inputs lie inside the fitted domain.
:func:`boundary_curve` returns the boundary as arrays on a chosen 2-D
projection, and a plot draws only that; no plotting code lives here (#1067).

Existing formula functions stay where they are. An entry *references* one
through ``function`` instead of restating its coefficients, so the Greenwald
entry evaluates ``stability.greenwald_density`` itself.

Notation
--------
b      : boundary value of the target quantity, in the target's unit      [varies]
x      : operating value of the same quantity, same unit                   [varies]
ratio  : x / b                                                             [-]
margin : signed normalised distance, positive on the permitted side        [-]

Conventions
-----------
Every input and target is a :class:`BoundaryQuantity` with an explicit unit
string, and callers pass values in exactly that unit. Nothing is converted
behind the caller's back. For ``allowed_side = "below"`` (permitted
$x < b$) the margin is $(b - x)/b$. For ``"above"`` (permitted $x > b$) it is
$(x - b)/b$. A positive margin is always the permitted side, and a zero
margin lies on the boundary.

References
----------
.. [1] Issue #1067 (data model), #1068 (density-limit family), #1066
       (transition and access boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Optional

import numpy as np

from .stability import greenwald_density

__all__ = [
    "BoundaryQuantity",
    "BoundarySource",
    "Applicability",
    "Uncertainty",
    "Boundary",
    "BoundaryWindow",
    "BoundaryEvaluation",
    "WindowEvaluation",
    "BoundaryCurve",
    "boundary_value",
    "evaluate_boundary",
    "evaluate_window",
    "boundary_curve",
    "get_boundary",
    "list_boundaries",
    "hugill_coordinates",
]

_FORMS = ("power_law", "threshold", "function")
_SIDES = ("below", "above")
_HARDNESS = ("hard", "soft", "probabilistic")
_ORIGINS = ("published", "derived", "reproduced", "fitted")


def _frozen_mapping(value) -> Mapping:
    return MappingProxyType(dict(value or {}))


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryQuantity:
    """A physical quantity with a fixed identity and unit.

    ``name`` is the identity, for example ``line_average_density`` versus
    ``edge_density``. Two quantities with the same unit but different names
    are not interchangeable.
    """

    name: str
    symbol: str
    unit: str
    definition: str = ""

    def __post_init__(self):
        if not self.name or not self.unit:
            raise ValueError("a BoundaryQuantity needs a name and a unit ('-' when dimensionless)")


@dataclass(frozen=True)
class BoundarySource:
    """Where a boundary comes from: citation, equation number and DOI."""

    citation: str
    equation: str = ""
    doi: str = ""
    note: str = ""


@dataclass(frozen=True)
class Applicability:
    """The domain a boundary was established on.

    ``ranges`` maps an input name to the closed interval ``(low, high)`` that
    was covered, in that input's unit. Use ``None`` for an open end. List only
    ranges the source states; an unknown range is left out, not guessed.
    """

    machine_class: str = ""
    ranges: Mapping[str, tuple] = field(default_factory=dict, hash=False)
    assumptions: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "ranges", _frozen_mapping({k: tuple(v) for k, v in dict(self.ranges or {}).items()}))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        for name, bounds in self.ranges.items():
            if len(bounds) != 2:
                raise ValueError(f"range for {name!r} must be (low, high), not {bounds!r}")
            low, high = bounds
            if low is not None and high is not None and low > high:
                raise ValueError(f"range for {name!r} has low > high: {bounds!r}")


@dataclass(frozen=True)
class Uncertainty:
    """Uncertainty the source itself reports, never an invented one.

    ``coefficient`` is the one-sigma absolute uncertainty of the leading
    coefficient. ``exponents`` holds the one-sigma uncertainty of each
    exponent, keyed by input name. ``None`` or an empty mapping means the
    source gives no value.
    """

    coefficient: Optional[float] = None
    exponents: Mapping[str, float] = field(default_factory=dict, hash=False)
    note: str = ""

    def __post_init__(self):
        object.__setattr__(self, "exponents", _frozen_mapping(self.exponents))


@dataclass(frozen=True)
class Boundary:
    """One published or derived operational boundary.

    ``form`` sets how :func:`boundary_value` evaluates the boundary:

    * ``"power_law"``: $b = C\\prod_i x_i^{\\alpha_i}$ with ``coefficient``
      $C$ and ``exponents`` $\\alpha_i$, with every input in its declared unit.
    * ``"threshold"``: $b = C$, a constant with no inputs.
    * ``"function"``: $b = f(**inputs)$, calling an existing VAFT formula
      function so its coefficients are not restated.

    ``allowed_side`` says which side of $b$ the operating value is permitted
    on. ``hardness`` records how the literature treats crossing it: ``soft``
    for an empirical limit that operation can exceed.
    """

    key: str
    family: str
    target: BoundaryQuantity
    inputs: tuple
    form: str
    allowed_side: str
    sources: tuple
    coefficient: Optional[float] = None
    exponents: Mapping[str, float] = field(default_factory=dict, hash=False)
    function: Optional[Callable] = None
    hardness: str = "soft"
    origin: str = "published"
    basis: str = "empirical"
    event: str = ""
    regime: str = ""
    applicability: Applicability = field(default_factory=Applicability)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "exponents", _frozen_mapping(self.exponents))
        if self.form not in _FORMS:
            raise ValueError(f"form must be one of {_FORMS}, not {self.form!r}")
        if self.allowed_side not in _SIDES:
            raise ValueError(f"allowed_side must be one of {_SIDES}, not {self.allowed_side!r}")
        if self.hardness not in _HARDNESS:
            raise ValueError(f"hardness must be one of {_HARDNESS}, not {self.hardness!r}")
        if self.origin not in _ORIGINS:
            raise ValueError(f"origin must be one of {_ORIGINS}, not {self.origin!r}")
        if not self.sources:
            raise ValueError(f"boundary {self.key!r} needs at least one BoundarySource")
        names = [q.name for q in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError(f"boundary {self.key!r} has duplicate input names: {names}")
        if self.form == "power_law":
            if self.coefficient is None or set(self.exponents) != set(names):
                raise ValueError(
                    f"power-law boundary {self.key!r} needs a coefficient and one exponent per input"
                )
        elif self.form == "threshold":
            if self.coefficient is None or self.inputs:
                raise ValueError(f"threshold boundary {self.key!r} needs a coefficient and no inputs")
        elif self.function is None:
            raise ValueError(f"function boundary {self.key!r} needs a function")
        unknown = set(self.applicability.ranges) - set(names)
        if unknown:
            raise ValueError(f"applicability ranges name unknown inputs: {sorted(unknown)}")

    @property
    def input_names(self) -> tuple:
        return tuple(q.name for q in self.inputs)

    def input(self, name: str) -> BoundaryQuantity:
        for quantity in self.inputs:
            if quantity.name == name:
                return quantity
        raise KeyError(f"boundary {self.key!r} has no input {name!r}; inputs are {self.input_names}")


@dataclass(frozen=True)
class BoundaryWindow:
    """An access window: permitted between a lower and an upper boundary.

    Both boundaries must bound the same target quantity. The lower one is
    permitted ``above`` and the upper one ``below``.
    """

    key: str
    regime: str
    lower: Boundary
    upper: Boundary

    def __post_init__(self):
        lower, upper = self.lower.target, self.upper.target
        if (lower.name, lower.unit) != (upper.name, upper.unit):
            raise ValueError("a window's lower and upper boundaries must share one target quantity")
        if self.lower.allowed_side != "above" or self.upper.allowed_side != "below":
            raise ValueError("a window needs a lower boundary permitted 'above' and an upper one permitted 'below'")


@dataclass(frozen=True, eq=False)
class BoundaryEvaluation:
    """How one operating state compares with one boundary."""

    key: str
    target: BoundaryQuantity
    boundary_value: object
    operating_value: object
    ratio: object
    difference: object
    margin: object
    allowed: object
    allowed_side: str
    extrapolated: tuple
    warnings: tuple

    @property
    def in_domain(self) -> bool:
        """True when no input lies outside the source's stated ranges."""
        return not self.extrapolated


@dataclass(frozen=True, eq=False)
class WindowEvaluation:
    """How one operating state compares with an access window."""

    key: str
    lower: BoundaryEvaluation
    upper: BoundaryEvaluation
    inside: object


@dataclass(frozen=True, eq=False)
class BoundaryCurve:
    """A boundary sampled on a 2-D projection, ready to overlay.

    ``allowed_side`` says where permitted operating points lie relative to
    the curve: ``"below"`` or ``"above"`` it along y, or ``"left"`` or
    ``"right"`` of it along x when the boundary's target is on the x axis.
    The ``x`` and ``y`` arrays are read-only.
    """

    key: str
    x: np.ndarray
    y: np.ndarray
    x_quantity: BoundaryQuantity
    y_quantity: BoundaryQuantity
    allowed_side: str
    fixed: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "fixed", _frozen_mapping(self.fixed))
        for name in ("x", "y"):
            array = np.array(getattr(self, name), dtype=float)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def xy(self) -> np.ndarray:
        """``(N, 2)`` array of ``[x, y]``: the ``vaft.diagram`` ``Chart.curves`` layout."""
        return np.stack([self.x, self.y], axis=-1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def _scalar_or_array(result):
    return float(result) if np.ndim(result) == 0 else result


def _check_inputs(boundary: Boundary, inputs: Mapping) -> None:
    missing = [name for name in boundary.input_names if name not in inputs]
    extra = [name for name in inputs if name not in boundary.input_names]
    if missing or extra:
        raise TypeError(
            f"boundary {boundary.key!r} takes inputs {boundary.input_names}; "
            f"missing {missing}, unexpected {extra}"
        )


def boundary_value(boundary: Boundary, **inputs):
    r"""Value of a boundary's target quantity at the given inputs.

    $$b = C\prod_i x_i^{\alpha_i}\quad\text{(power law)},\qquad b = C\quad\text{(threshold)},\qquad b = f(x_1,\ldots)\quad\text{(function)}$$

    Parameters
    ----------
    boundary : Boundary
        The boundary to evaluate [-].

    Returns
    -------
    float or np.ndarray
        Boundary value, in the unit of ``boundary.target`` [varies].

    Raises
    ------
    TypeError
        An input is missing or not one of the boundary's inputs.

    Convention
    ----------
    Each keyword input is a value in the unit its ``BoundaryQuantity``
    declares. No unit conversion happens here; a caller holding amperes for a
    boundary declared in MA converts first.

    Notes
    -----
    Array inputs broadcast with NumPy rules.
    """
    _check_inputs(boundary, inputs)
    if boundary.form == "threshold":
        return float(boundary.coefficient)
    if boundary.form == "function":
        return _scalar_or_array(boundary.function(**inputs))
    value = boundary.coefficient
    for name, exponent in boundary.exponents.items():
        value = value * np.asarray(inputs[name], dtype=float) ** exponent
    return _scalar_or_array(value)


def _extrapolated(boundary: Boundary, inputs: Mapping) -> tuple:
    outside = []
    for name, (low, high) in boundary.applicability.ranges.items():
        values = np.asarray(inputs[name], dtype=float)
        below = bool(np.any(~np.isfinite(values))) or (low is not None and bool(np.any(values < low)))
        above = high is not None and bool(np.any(values > high))
        if below or above:
            outside.append(name)
    return tuple(outside)


def evaluate_boundary(boundary: Boundary, operating_value, **inputs) -> BoundaryEvaluation:
    r"""Distance and side of an operating value relative to a boundary.

    $$\mathrm{margin} = \frac{b - x}{b}\ \ (\text{permitted below}),\qquad \mathrm{margin} = \frac{x - b}{b}\ \ (\text{permitted above})$$

    Parameters
    ----------
    boundary : Boundary
        The boundary to compare against [-].
    operating_value : float or np.ndarray
        Operating value $x$ of the boundary's target quantity, in the
        target's unit [varies].

    Returns
    -------
    BoundaryEvaluation
        Boundary value $b$, ratio $x/b$, difference $x - b$, signed margin
        (positive on the permitted side), permitted flag, the inputs that lie
        outside the source's stated ranges, and warnings [-].

    Raises
    ------
    TypeError
        An input is missing or not one of the boundary's inputs.

    Convention
    ----------
    The margin is normalised by the boundary value. A positive margin is
    always the permitted side, whichever way the boundary points, so margins
    of different boundaries share one sign convention. ``allowed`` is strict:
    a state exactly on the boundary is not counted as permitted. A boundary
    value that is zero, negative or not finite gives NaN margin and ratio, is
    never permitted, and adds a warning. Pass magnitudes where a sign
    convention could make the boundary negative.

    Notes
    -----
    Crossing a ``soft`` boundary is not a prediction of failure. For example,
    $f_G > 1$ is routinely reached. The evaluation reports the geometry, and
    interpreting it is left to the caller.
    """
    b = boundary_value(boundary, **inputs)
    x = np.asarray(operating_value, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    difference = x - b_arr
    signed = -difference if boundary.allowed_side == "below" else difference
    valid = np.isfinite(b_arr) & (b_arr > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(valid, signed / np.where(valid, b_arr, 1.0), np.nan)
        ratio = np.where(valid, x / np.where(valid, b_arr, 1.0), np.nan)
    allowed = valid & (signed > 0)
    extrapolated = _extrapolated(boundary, inputs)
    warnings = tuple(
        f"{name} lies outside the range {boundary.applicability.ranges[name]} "
        f"[{boundary.input(name).unit}] covered by {boundary.key!r}, or is not finite"
        for name in extrapolated
    )
    if not np.all(valid):
        warnings += (
            f"{boundary.key!r} is non-positive or not finite at some inputs; margin and ratio are NaN "
            "and the state is not counted as permitted there",
        )
    return BoundaryEvaluation(
        key=boundary.key,
        target=boundary.target,
        boundary_value=b,
        operating_value=_scalar_or_array(x),
        ratio=_scalar_or_array(ratio),
        difference=_scalar_or_array(difference),
        margin=_scalar_or_array(margin),
        allowed=bool(allowed) if np.ndim(allowed) == 0 else allowed,
        allowed_side=boundary.allowed_side,
        extrapolated=extrapolated,
        warnings=warnings,
    )


def evaluate_window(window: BoundaryWindow, operating_value, **inputs) -> WindowEvaluation:
    r"""Position of an operating value relative to an access window.

    $$b_\mathrm{lower} < x < b_\mathrm{upper}$$

    Parameters
    ----------
    window : BoundaryWindow
        The access window [-].
    operating_value : float or np.ndarray
        Operating value of the window's target quantity, in its unit [varies].

    Returns
    -------
    WindowEvaluation
        One evaluation per edge and ``inside``, true where both are
        permitted [-].

    Raises
    ------
    TypeError
        An input is missing or unused by both edges.

    Convention
    ----------
    ``inputs`` holds the union of both edges' inputs. Each edge receives only
    the inputs it declares.
    """
    known = set(window.lower.input_names) | set(window.upper.input_names)
    extra = [name for name in inputs if name not in known]
    if extra:
        raise TypeError(f"window {window.key!r} does not use inputs {extra}")
    lower = evaluate_boundary(
        window.lower, operating_value, **{k: inputs[k] for k in window.lower.input_names if k in inputs}
    )
    upper = evaluate_boundary(
        window.upper, operating_value, **{k: inputs[k] for k in window.upper.input_names if k in inputs}
    )
    inside = np.logical_and(lower.allowed, upper.allowed)
    return WindowEvaluation(
        key=window.key, lower=lower, upper=upper, inside=bool(inside) if np.ndim(inside) == 0 else inside
    )


def boundary_curve(boundary: Boundary, sweep: str, values, swap_axes: bool = False, **fixed) -> BoundaryCurve:
    r"""A boundary sampled along one input, as $(x, y) = (x_\mathrm{sweep}, b)$ arrays.

    $$y_k = b\left(x_\mathrm{sweep} = v_k,\ \text{other inputs fixed}\right)$$

    Parameters
    ----------
    boundary : Boundary
        The boundary to sample [-].
    sweep : str
        Name of the input placed on the x axis [-].
    values : array_like
        Values of the swept input, in its declared unit [varies].
    swap_axes : bool
        Put the boundary's target on x and the swept input on y, as in a
        Hugill diagram, whose x axis is the density-like quantity [-].

    Returns
    -------
    BoundaryCurve
        ``x`` is the swept input and ``y`` the boundary value. The curve
        carries both quantities (identity and unit), the permitted side along
        y, and the fixed inputs [-].

    Raises
    ------
    KeyError
        ``sweep`` is not an input of the boundary.
    TypeError
        A fixed input is missing, unknown, an array, or duplicates the swept
        one; or ``values`` is not 1-D.

    Convention
    ----------
    The permitted side is the boundary's own ``allowed_side``, read along the
    y axis: ``"below"`` means operating points under the curve are permitted.
    With ``swap_axes`` the target is read along x, so ``"below"`` becomes
    ``"left"`` and ``"above"`` becomes ``"right"``.
    """
    x_quantity = boundary.input(sweep)
    if sweep in fixed:
        raise TypeError(f"{sweep!r} is swept; do not also fix it")
    non_scalar = [name for name, value in fixed.items() if np.ndim(value) != 0]
    if non_scalar:
        raise TypeError(f"fixed inputs must be scalars; {non_scalar} are arrays")
    x = np.atleast_1d(np.asarray(values, dtype=float))
    if x.ndim != 1:
        raise TypeError("values must be a 1-D sequence")
    y = np.broadcast_to(np.asarray(boundary_value(boundary, **{sweep: x}, **fixed), dtype=float), x.shape)
    fixed = {name: float(value) for name, value in fixed.items()}
    if swap_axes:
        side = {"below": "left", "above": "right"}[boundary.allowed_side]
        return BoundaryCurve(key=boundary.key, x=y, y=x, x_quantity=boundary.target, y_quantity=x_quantity,
                             allowed_side=side, fixed=fixed)
    return BoundaryCurve(key=boundary.key, x=x, y=y, x_quantity=x_quantity, y_quantity=boundary.target,
                         allowed_side=boundary.allowed_side, fixed=fixed)


def hugill_coordinates(n_e, R_geo, B_t, a, kappa_a, I_p):
    r"""Hugill-diagram coordinates of an operating state: $(\bar n_e R/B_T,\ 1/q_\mathrm{cyl})$.

    $$x = \frac{\bar n_e R_{geo}}{B_T},\qquad y = \frac{1}{q_{cyl}} = \frac{R_{geo}\,|I_p|}{5\,a^2\kappa_a B_T}\quad(I_p\ \mathrm{in\ MA})$$

    Parameters
    ----------
    n_e : float or np.ndarray
        Line-averaged electron density [1e19 m^-3].
    R_geo : float or np.ndarray
        Geometric major radius [m].
    B_t : float or np.ndarray
        Vacuum toroidal field at ``R_geo``, magnitude [T].
    a : float or np.ndarray
        Minor radius [m].
    kappa_a : float or np.ndarray
        Area elongation $S/(\pi a^2)$ [-].
    I_p : float or np.ndarray
        Plasma current; its sign is dropped [MA].

    Returns
    -------
    tuple of (float or np.ndarray)
        Murakami parameter $\bar n_e R/B_T$ [1e19 m^-2 T^-1] and $1/q_{cyl}$ [-].

    Raises
    ------
    ValueError
        A negative density, or a non-positive field, radius, minor radius or
        elongation.

    Convention
    ----------
    $q_{cyl}$ is ``equilibrium.q_cyl_from_B_R_epsilon_kappa_I`` with
    $\varepsilon = a/R_{geo}$. The ``"greenwald_hugill"`` boundary uses the
    same convention, so a state at $\bar n_e = n_G$ lies exactly on it. This
    is the cylindrical $q$, not $q_{95}$. At VEST aspect ratio the two differ
    substantially, and plotting $1/q_{95}$ against this boundary is only
    approximate.

    Numerical notes
    ---------------
    $1/q_{cyl}$ is formed directly rather than as ``1/q_cyl_from_...``, so a
    zero current maps to the origin $y = 0$ instead of raising. A whole time
    trace, including start-up and termination, can therefore be projected
    without cropping.
    """
    n = np.asarray(n_e, dtype=float)
    B_abs = np.abs(np.asarray(B_t, dtype=float))
    R = np.asarray(R_geo, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    kappa = np.asarray(kappa_a, dtype=float)
    if np.any(n < 0):
        raise ValueError("n_e must be non-negative")
    for name, value in (("B_t", B_abs), ("R_geo", R), ("a", a_arr), ("kappa_a", kappa)):
        if np.any(~(value > 0)):
            raise ValueError(f"{name} must be positive and finite")
    x = n * R / B_abs
    y = np.abs(np.asarray(I_p, dtype=float)) * R / (5.0 * a_arr**2 * kappa * B_abs)
    return _scalar_or_array(x), _scalar_or_array(y)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REGISTRY: dict = {}


def _register(entry):
    if entry.key in _REGISTRY:
        raise ValueError(f"boundary key {entry.key!r} is already registered")
    _REGISTRY[entry.key] = entry
    return entry


def get_boundary(key: str):
    """A registered boundary or window by key.

    Parameters
    ----------
    key : str
        Registry key, for example ``"greenwald"`` [-].

    Returns
    -------
    Boundary or BoundaryWindow
        The registered entry [-].

    Raises
    ------
    KeyError
        No entry has that key; the message lists the keys that exist.
    """
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no boundary {key!r}; registered: {sorted(_REGISTRY)}") from None


def list_boundaries(family: Optional[str] = None) -> tuple:
    """Keys of the registered boundaries, optionally restricted to one family.

    Parameters
    ----------
    family : str, optional
        Family name, for example ``"density_limit"`` [-].

    Returns
    -------
    tuple of str
        Sorted registry keys [-].
    """
    return tuple(sorted(
        key for key, entry in _REGISTRY.items()
        if family is None or getattr(entry, "family", getattr(getattr(entry, "lower", None), "family", None)) == family
    ))


# ------------------------------------------------------------------
# Quantities and entries
# ------------------------------------------------------------------

_PLASMA_CURRENT_MA = BoundaryQuantity(
    "plasma_current", "|I_p|", "MA",
    "Magnitude of the total toroidal plasma current; the sign convention (COCOS) is dropped.",
)
_MINOR_RADIUS = BoundaryQuantity("minor_radius", "a", "m", "Plasma minor radius (half the midplane width).")
_LINE_AVERAGE_DENSITY = BoundaryQuantity(
    "line_average_density", r"\bar n_e", "1e19 m^-3",
    "Line-averaged electron density along a (near-)central chord; not the volume average.",
)

_register(Boundary(
    key="greenwald",
    family="density_limit",
    target=_LINE_AVERAGE_DENSITY,
    inputs=(_PLASMA_CURRENT_MA, _MINOR_RADIUS),
    form="function",
    function=lambda plasma_current, minor_radius: greenwald_density(np.abs(plasma_current), minor_radius),
    allowed_side="below",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="density_limit",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "compared with the line-averaged electron density",
            "empirical operational limit: f_G > 1 is reached (e.g. with peaked profiles) and is not a disruption criterion",
        ),
    ),
    sources=(
        BoundarySource("M. Greenwald et al., Nucl. Fusion 28 (1988) 2199", equation="Eq. (1)",
                       doi="10.1088/0029-5515/28/12/009",
                       note="n = kappa * J_avg [1e20 m^-3, MA m^-2], i.e. I_p/(pi a^2) for elliptical cross-sections"),
        BoundarySource("M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27", equation="Eq. (1.3), p. R28",
                       doi="10.1088/0741-3335/44/8/201",
                       note="n_G = I_P/(pi a^2), line-averaged density in 1e20 m^-3"),
    ),
    notes="Evaluated by vaft.formula.stability.greenwald_density; coefficients are not restated here.",
))

_MURAKAMI_PARAMETER = BoundaryQuantity(
    "murakami_parameter", r"\bar n_e R/B_T", "1e19 m^-2 T^-1",
    "Line-averaged electron density times geometric major radius over the vacuum toroidal field there.",
)
_INVERSE_Q_CYL = BoundaryQuantity(
    "inverse_cylindrical_q", "1/q_cyl", "-",
    "Inverse cylindrical safety factor in the equilibrium.q_cyl_from_B_R_epsilon_kappa_I convention.",
)
_AREA_ELONGATION = BoundaryQuantity("area_elongation", "kappa_a", "-", "Area elongation S/(pi a^2).")

_register(Boundary(
    key="greenwald_hugill",
    family="density_limit",
    target=_MURAKAMI_PARAMETER,
    inputs=(_INVERSE_Q_CYL, _AREA_ELONGATION),
    form="power_law",
    coefficient=50.0 / np.pi,
    exponents={"inverse_cylindrical_q": 1.0, "area_elongation": 1.0},
    allowed_side="below",
    hardness="soft",
    origin="derived",
    basis="empirical",
    event="density_limit",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "the Greenwald limit rewritten in Hugill-diagram coordinates; carries every Greenwald assumption",
            "y axis is the cylindrical q of equilibrium.q_cyl_from_B_R_epsilon_kappa_I, not q95",
            "R and B_T in the Murakami parameter are the same R_geo and B_T used for q_cyl",
        ),
    ),
    sources=(
        BoundarySource(
            "Derived in VAFT: stability.greenwald_density, n_G[1e19] = 10 I_p/(pi a^2), with "
            "equilibrium.q_cyl_from_B_R_epsilon_kappa_I, q_cyl = 5 a^2 kappa_a B_T/(R I_p[MA]); "
            "a, R and B_T cancel, leaving nR/B = (50 kappa_a/pi)(1/q_cyl)",
        ),
        BoundarySource("M. Greenwald et al., Nucl. Fusion 28 (1988) 2199", equation="text after Eq. (1)",
                       doi="10.1088/0029-5515/28/12/009",
                       note="for high-aspect-ratio, low-beta circular plasmas the limit is (5/pi) B/(qR) in 1e20 m^-3, "
                            "i.e. the 50/pi slope here with kappa_a = 1"),
        BoundarySource("G. Verdoolaege et al., Nucl. Fusion 61 (2021) 076006", equation="Sec. 2",
                       note="q_cyl convention"),
    ),
    notes="Same line as vaft.diagram hugill(). On the same diagram the Murakami limit is 'murakami_hugill' "
          "(n R/B_T = 1) and the current limit is 'low_q' (q_psi > 2, not q_cyl).",
))

# Murakami et al. 1976 give the scaling as a line on Fig. 1 (maximum line-averaged
# density in 1e19 m^-3 against B_T/R_0 in T/m, slope one on log-log axes), not as a
# printed formula; Greenwald 2002 (Sec. 1.2.1, p. R29) states it as n_M = B_T/R.
_TOROIDAL_FIELD = BoundaryQuantity("toroidal_field", "B_T", "T", "Vacuum toroidal field at the major radius R_0, magnitude.")
_MAJOR_RADIUS = BoundaryQuantity("major_radius", "R_0", "m", "Major radius.")
_MURAKAMI_SOURCES = (
    BoundarySource("M. Murakami, J. D. Callen and L. A. Berry, Nucl. Fusion 16 (1976) 347", equation="Fig. 1 and Table I",
                   doi="10.1088/0029-5515/16/2/020",
                   note="13 Ohmic, hydrogenic, mostly circular devices; the solid line through the black dots passes about 1.1e19 m^-3 at 1 T/m; a lower dashed line fits the ORMAK constant-q(a) scan"),
    BoundarySource("M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27", equation="Sec. 1.2.1, p. R29",
                   doi="10.1088/0741-3335/44/8/201", note="n_M = B_T/R, the 'Murakami limit'"),
    BoundarySource("M. Greenwald et al., Nucl. Fusion 28 (1988) 2199", equation="Sec. 4 (Summary), pp. 2206-2207",
                   doi="10.1088/0029-5515/28/12/009",
                   note="the operating space is bounded by the minimum of the Murakami, Hugill and fuelling limits"),
)
_MURAKAMI_APPLICABILITY = dict(
    machine_class="tokamak",
    assumptions=(
        "Ohmically heated hydrogenic plasmas with stationary gas filling; cold-gas injection "
        "(Alcator, Pulsator) reached about 2.5 times the line",
        "mostly circular cross-sections with q(a) near 5",
        "compared with the maximum line-averaged electron density",
        "interpreted as global power balance between Ohmic input and radiation (Greenwald 2002); "
        "later data with auxiliary heating exceed it",
    ),
)
_MURAKAMI_UNCERTAINTY = Uncertainty(note=(
    "No fit uncertainty is published. The coefficient 1 follows Greenwald 2002; the solid line in "
    "Murakami's Fig. 1 sits about 10-15 % higher, and the stationary-fill devices of Table I lie at "
    "0.8-1.45 times B_T/R_0."
))

_register(Boundary(
    key="murakami",
    family="density_limit",
    target=_LINE_AVERAGE_DENSITY,
    inputs=(_TOROIDAL_FIELD, _MAJOR_RADIUS),
    form="power_law",
    coefficient=1.0,
    exponents={"toroidal_field": 1.0, "major_radius": -1.0},
    allowed_side="below",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="density_limit",
    applicability=Applicability(ranges={"toroidal_field": (0.6, 7.5), "major_radius": (0.40, 1.09)},
                                **_MURAKAMI_APPLICABILITY),
    uncertainty=_MURAKAMI_UNCERTAINTY,
    sources=_MURAKAMI_SOURCES,
    notes="n_M [1e19 m^-3] = B_T [T] / R_0 [m]. Ranges are those of the 13 devices in Table I.",
))

_register(Boundary(
    key="murakami_hugill",
    family="density_limit",
    target=_MURAKAMI_PARAMETER,
    inputs=(),
    form="threshold",
    coefficient=1.0,
    allowed_side="below",
    hardness="soft",
    origin="derived",
    basis="empirical",
    event="density_limit",
    applicability=Applicability(**_MURAKAMI_APPLICABILITY),
    uncertainty=_MURAKAMI_UNCERTAINTY,
    sources=_MURAKAMI_SOURCES,
    notes="The Murakami limit divided by B_T/R_0: a vertical line n R/B_T = 1 [1e19 m^-2 T^-1] on the Hugill diagram. "
          "Uses the same R and B_T as hugill_coordinates.",
))

_EDGE_Q_MHD = BoundaryQuantity(
    "edge_safety_factor", "q_psi", "-",
    "Safety factor of the MHD equilibrium at the boundary (the paper plots 1/q_psi); not the cylindrical q.",
)

_register(Boundary(
    key="low_q",
    family="current_limit",
    target=_EDGE_Q_MHD,
    inputs=(),
    form="threshold",
    coefficient=2.0,
    allowed_side="above",
    hardness="hard",
    origin="published",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "disruptive limit on plasma current stated as q_psi > 2",
            "q_psi is the equilibrium safety factor; on the Hugill diagram's 1/q_cyl axis this is a horizontal "
            "line only where q_psi and q_cyl coincide (circular, high aspect ratio)",
        ),
    ),
    sources=(
        BoundarySource("M. Greenwald et al., Nucl. Fusion 28 (1988) 2199", equation="Sec. 4 (Summary), p. 2207; also p. 2200",
                       doi="10.1088/0029-5515/28/12/009",
                       note="'the disruptive limit on plasma current (q_psi > 2)'"),
    ),
))

# Giacomin et al., PRL 128 (2022) 185003, Eq. (12): the maximum edge density set by
# turbulent transport across the separatrix. It is written as a function because
# (1 + kappa^2) is not a power of one input; the exponents are those printed in Eq. (12).
_GIACOMIN_ALPHA = 3.3


def _giacomin_2022_edge_density_limit(mass_number, minor_radius, separatrix_power, major_radius,
                                      edge_safety_factor_95, elongation, toroidal_field):
    """Eq. (12) of Giacomin et al. 2022, in 1e20 m^-3."""
    kappa = np.asarray(elongation, dtype=float)
    return (_GIACOMIN_ALPHA
            * np.asarray(mass_number, dtype=float) ** (1 / 6)
            * np.asarray(minor_radius, dtype=float) ** (3 / 14)
            * np.asarray(separatrix_power, dtype=float) ** (10 / 21)
            * np.asarray(major_radius, dtype=float) ** (-43 / 42)
            * np.asarray(edge_safety_factor_95, dtype=float) ** (-22 / 21)
            * (1.0 + kappa**2) ** (-1 / 3)
            * np.asarray(toroidal_field, dtype=float) ** (2 / 3))


_EDGE_DENSITY = BoundaryQuantity(
    "edge_density", "n_e,edge", "1e20 m^-3",
    "Edge electron density at the MARFE onset: Thomson-scattering average over rho_pol 0.85-0.95; "
    "not the line-averaged or separatrix density.",
)

_register(Boundary(
    key="giacomin_edge",
    family="density_limit",
    target=_EDGE_DENSITY,
    inputs=(
        BoundaryQuantity("mass_number", "A", "-", "Mass number of the main plasma ions."),
        _MINOR_RADIUS,
        BoundaryQuantity("separatrix_power", "P_SOL", "MW",
                         "Power crossing the separatrix: total coupled power minus core radiated power."),
        _MAJOR_RADIUS,
        BoundaryQuantity("edge_safety_factor_95", "q95", "-", "Safety factor at the 95 % flux surface."),
        BoundaryQuantity("elongation", "kappa", "-", "Plasma elongation."),
        _TOROIDAL_FIELD,
    ),
    form="function",
    function=_giacomin_2022_edge_density_limit,
    allowed_side="below",
    hardness="soft",
    origin="published",
    basis="first_principles_scaling_with_one_fitted_constant",
    event="MARFE_onset",
    regime="L_mode",
    applicability=Applicability(
        machine_class="tokamak",
        ranges={"toroidal_field": (1.4, 3.0), "major_radius": (0.9, 3.0), "separatrix_power": (0.1, 9.0)},
        assumptions=(
            "validated on AUG, JET and TCV (carbon and metal walls; NBI, ECRH and ICRH); "
            "line-averaged densities 2e19-1.1e20 m^-3, plasma current 0.1-2.5 MA",
            "L-mode density limit; in the H-mode scenario it is reached after the H-L back transition",
            "the target is the edge density at the MARFE onset (a precursor of the disruption), "
            "not the line-averaged density of the Greenwald limit",
            "alpha may depend on plasma shape and divertor geometry (not resolved by the database)",
        ),
    ),
    uncertainty=Uncertainty(
        coefficient=0.3,
        note="alpha = 3.3 +- 0.3 (one constant for all tokamaks; per-machine variance below 10 %). "
             "Indicative 20 % uncertainty on the measured and predicted edge density; P_SOL from "
             "bolometry can be uncertain by up to 50 %.",
    ),
    sources=(
        BoundarySource(
            "M. Giacomin, A. Pau, P. Ricci et al., Phys. Rev. Lett. 128 (2022) 185003",
            equation="Eq. (12); alpha = 3.3 +- 0.3, p. 4 and Fig. 3(a)", doi="10.1103/PhysRevLett.128.185003",
            note="n_lim = alpha A^(1/6) a^(3/14) P_SOL^(10/21) R0^(-43/42) q^(-22/21) (1+kappa^2)^(-1/3) "
                 "B_T^(2/3), n_lim in 1e20 m^-3, P_SOL in MW, R0 and a in m, B_T in T, q = q95",
        ),
    ),
    notes="Inputs are magnitudes (B_T > 0, q95 > 0); a negative value gives a non-finite limit, which "
          "evaluate_boundary reports and never counts as permitted. "
          "Compare with n_e,edge measured at rho_pol 0.85-0.95. Greenwald and Murakami bound the "
          "line-averaged density instead, so the two families are shown side by side, not substituted.",
))
