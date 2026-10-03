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

Examples
--------
Greenwald limit on the Hugill diagram, and one operating state on it
(``test/test_formula_boundaries_vest_sample.py`` does this for the packaged
VEST sample)::

    import numpy as np
    from vaft.formula import boundaries as B
    line = B.get_boundary("greenwald_hugill")
    curve = B.boundary_curve(line, "inverse_cylindrical_q", np.linspace(0, 0.6, 61),
                             swap_axes=True, area_elongation=1.5)   # curve.xy -> (61, 2)
    x, y = B.hugill_coordinates(n_e, R_geo, B_t, a, kappa_a, I_p)  # 1e19 m^-3, m, T, m, -, MA
    B.evaluate_boundary(line, x, inverse_cylindrical_q=y, area_elongation=kappa_a).ratio  # = f_G

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

from .constants import MU0
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
    "threshold_curve",
    "same_quantity",
    "get_boundary",
    "list_boundaries",
    "hugill_coordinates",
    "kink_coordinates",
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
    r"""Uncertainty the source itself reports, never an invented one.

    ``coefficient`` is the one-sigma absolute uncertainty of the leading
    coefficient; ``coefficient_factor`` is the multiplicative factor when the
    source quotes the coefficient as $C e^{\pm s}$ (then the factor is
    $e^{s}$; the source may not state its confidence level). ``exponents`` holds the one-sigma uncertainty of each exponent,
    keyed by input name. ``rms_relative`` is the relative RMS scatter of the
    fit about the data. ``None`` or an empty mapping means the source gives
    no value.
    """

    coefficient: Optional[float] = None
    exponents: Mapping[str, float] = field(default_factory=dict, hash=False)
    note: str = ""
    coefficient_factor: Optional[float] = None
    rms_relative: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "exponents", _frozen_mapping(self.exponents))
        if self.coefficient_factor is not None and not self.coefficient_factor >= 1.0:
            raise ValueError("coefficient_factor is a multiplicative factor e^s and must be >= 1")
        if self.rms_relative is not None and not self.rms_relative >= 0.0:
            raise ValueError("rms_relative must be >= 0")


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

    A regime-transition threshold carries ``source_regime`` and
    ``target_regime`` (for example ``"L_mode"`` and ``"H_mode"``); its permitted
    side is the side on which the target regime is accessible. An entry that
    only bounds where another relation is valid (for example the density of
    minimum L-H power) leaves both empty and says so in its notes. ``branch``
    names the part of a non-monotonic dependence the relation describes, for
    example the high-density branch of the L-H threshold. It is metadata:
    nothing checks it automatically, so a caller must not evaluate a branch
    relation outside that branch.
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
    source_regime: str = ""
    target_regime: str = ""
    branch: str = ""

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
        if bool(self.source_regime) != bool(self.target_regime):
            raise ValueError(f"boundary {self.key!r} needs both source_regime and target_regime, or neither")
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
    never permitted, and adds a warning. An operating value that is not
    finite (a NaN gap in a trace) is likewise reported as not permitted, with
    NaN margin and ratio and a warning: ``allowed`` is False there, not
    unknown, so count violations from ``margin`` when a trace has gaps. Pass
    magnitudes where a sign convention could make the boundary negative.

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
    x_finite = np.isfinite(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(valid, signed / np.where(valid, b_arr, 1.0), np.nan)
        ratio = np.where(valid, x / np.where(valid, b_arr, 1.0), np.nan)
    allowed = valid & x_finite & (signed > 0)
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
    if not np.all(x_finite):
        warnings += (
            "the operating value is not finite at some inputs; margin and ratio are NaN "
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


def same_quantity(a: BoundaryQuantity, b: BoundaryQuantity) -> bool:
    r"""Whether two quantities are the same physical quantity in the same unit.

    Parameters
    ----------
    a, b : BoundaryQuantity
        The quantities to compare [-].

    Returns
    -------
    bool
        ``True`` when both ``name`` (the identity) and ``unit`` match. The
        symbol and the free-text definition are not compared [-].

    Convention
    ----------
    Identity is the ``name``: ``edge_safety_factor`` ($q_\psi$),
    ``edge_safety_factor_95`` ($q_{95}$) and ``inverse_cylindrical_q`` are
    different quantities even where their values are close. A boundary may be
    drawn on an axis only when this returns ``True`` for that axis (#1425).
    """
    return a.name == b.name and a.unit == b.unit


def threshold_curve(boundary: Boundary, axis: BoundaryQuantity, values, target_axis: str = "y") -> BoundaryCurve:
    r"""A threshold boundary drawn as a straight line across a 2-D projection.

    $$\text{target} = C \quad\text{for every value of the other axis}$$

    Parameters
    ----------
    boundary : Boundary
        A ``"threshold"`` boundary, for example ``"murakami_hugill"`` [-].
    axis : BoundaryQuantity
        The quantity on the other axis, which the threshold does not depend on [-].
    values : array_like
        Sample points along that axis, in its declared unit [varies].
    target_axis : str
        ``"y"`` draws a horizontal line (target on y); ``"x"`` a vertical one [-].

    Returns
    -------
    BoundaryCurve
        The line, carrying both quantities and the permitted side: ``"below"``
        / ``"above"`` for a horizontal line, ``"left"`` / ``"right"`` for a
        vertical one [-].

    Raises
    ------
    TypeError
        ``boundary`` is not a threshold, or ``values`` is not 1-D.
    ValueError
        ``target_axis`` is neither ``"x"`` nor ``"y"``, or ``axis`` is the
        boundary's own target.

    Convention
    ----------
    :func:`boundary_curve` sweeps one of a boundary's inputs, and a threshold
    has none, so this is its counterpart. The value is the registered
    coefficient; nothing is restated.
    """
    if boundary.form != "threshold":
        raise TypeError(f"boundary {boundary.key!r} is a {boundary.form!r}, not a threshold; use boundary_curve")
    if target_axis not in ("x", "y"):
        raise ValueError(f"target_axis must be 'x' or 'y', not {target_axis!r}")
    if same_quantity(axis, boundary.target):
        raise ValueError(f"the other axis cannot be the threshold's own target {boundary.target.name!r}")
    v = np.atleast_1d(np.asarray(values, dtype=float))
    if v.ndim != 1:
        raise TypeError("values must be a 1-D sequence")
    level = np.full(v.shape, float(boundary_value(boundary)))
    if target_axis == "x":
        side = {"below": "left", "above": "right"}[boundary.allowed_side]
        return BoundaryCurve(key=boundary.key, x=level, y=v, x_quantity=boundary.target, y_quantity=axis,
                             allowed_side=side)
    return BoundaryCurve(key=boundary.key, x=v, y=level, x_quantity=axis, y_quantity=boundary.target,
                         allowed_side=boundary.allowed_side)


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

_GREENWALD_FRACTION = BoundaryQuantity(
    "greenwald_fraction", "f_G", "-",
    "Line-averaged electron density over the Greenwald density I_p/(pi a^2).",
)

_register(Boundary(
    key="greenwald_fraction_unity",
    family="density_limit",
    target=_GREENWALD_FRACTION,
    inputs=(),
    form="threshold",
    coefficient=1.0,
    allowed_side="below",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="density_limit",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "the 'greenwald' limit written as a fraction: f_G = n_e,line / n_G with n_G = I_p/(pi a^2)",
            "empirical operational limit: f_G > 1 is reached (e.g. with peaked profiles) and is not a disruption criterion",
        ),
    ),
    sources=(
        BoundarySource("M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27", equation="Eq. (1.3), p. R28",
                       doi="10.1088/0741-3335/44/8/201",
                       note="n_G = I_P/(pi a^2), line-averaged density in 1e20 m^-3; the limit is n/n_G = 1"),
    ),
    notes="Same limit as 'greenwald', on the fraction axis.",
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

# Troyon et al. 1984 give the n = 1 free-boundary limit as (beta A)_max ~ 2.2 I_N with
# I_N = mu0 I A^2 / T_S and A = R/a (p. 214). T_S is T = r B_phi at the surface, which equals the
# vacuum value R B (T(psi_s = 0) = T_vac, p. 210). The values printed for INTOR (286 T m, p. 210)
# and JET (105 T m at R = 2.96 m, p. 213) are ten times R B: only T_S = R B in SI puts JET's
# n = 1 points of Fig. 8 (about 7 % at 10 MA, A = 2.36) on the line of Fig. 10.
# Substituting gives beta[%] <= 2.2 mu0 I/(a B): 2.2 * mu0 * 1e6 ~ 2.76 with I in MA, the
# origin of the commonly quoted beta_N <= 2.8. The paper does not print 2.8 or a q95 factor.
_NORMALIZED_BETA = BoundaryQuantity(
    "normalized_beta", "beta_N", "% m T/MA",
    "beta[%] a[m] B_T[T] / I_p[MA]. Troyon's beta is 2 int p dV / int B^2 dV (total field); "
    "for low beta this is close to the toroidal beta 2 mu0 <p> / B_T^2 used by stability.beta_N_from_beta_a_B0_Ip.",
)

_register(Boundary(
    key="troyon",
    family="beta_limit",
    target=_NORMALIZED_BETA,
    inputs=(),
    form="threshold",
    coefficient=2.2 * MU0 * 1e6,
    allowed_side="below",
    hardness="soft",
    origin="published",
    basis="ideal_mhd_numerical",
    event="beta_limit",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "ideal MHD, n = 1 free-boundary kink, no conducting wall; the limit is where the normalised "
            "growth rate squared reaches 1e-4 (p. 210)",
            "stability to all n gives a lower limit at high current (Fig. 8); this entry is the n = 1 line",
            "pressure profile optimised for ballooning stability; q_0 near the Mercier limit, q_s near 2",
            "JET- and INTOR-like shapes: R/a from 2.36 to 4, elongation 1.6-1.68, triangularity 0.3",
            "the paper notes resistivity may make the ideal limit soft",
        ),
    ),
    uncertainty=Uncertainty(note=(
        "No fit uncertainty is published; the coefficient 2.2 is read as 'approximately' from the fit "
        "to the INTOR, R/a = 3 and JET cases in Fig. 10."
    )),
    sources=(
        BoundarySource("F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209",
                       equation="p. 214, (beta A)_max ~ 2.2 I_N with I_N = mu0 I A^2/T_S; Fig. 10",
                       doi="10.1088/0741-3335/26/1A/319",
                       note="T_S = r B_phi at the surface = R B_vac; the printed INTOR (286 T m) and JET "
                            "(105 T m) values are 10x R B and are read that way to match Fig. 10; beta in %, "
                            "beta = 2 int p dV / int B^2 dV (p. 210)"),
    ),
    notes="Coefficient 2.2 * mu0 * 1e6 ~ 2.76 %·m·T/MA (substituting T_S = R B_T into I_N). Compare with "
          "stability.beta_N_from_beta_a_B0_Ip, which returns beta_N in the same %·m·T/MA convention (#349).",
))


# ------------------------------------------------------------------
# L-H power threshold (#1066)
# ------------------------------------------------------------------

_LOW_ASPECT_RATIO_EVIDENCE = (
    "spherical tokamaks: MAST (A ~ 1.45) and NSTX (A ~ 1.32) sit 1.6x and 3.7x above the conventional-"
    "aspect-ratio basis P_thr0 of Takizuka 2004 Eq. (1); Pegasus (A ~ 1.2, B_T ~ 0.15 T, Ohmic) measures P_LH "
    "7-15x the ITPA08 (Martin 2008) scaling, the ratio grows as A -> 1, and no density minimum is seen "
    "(Thome et al. 2017)",
    "NSTX P_LH nearly doubles from 0.7 to 1.0 MA, of which the |B|_out parameterisation accounts for "
    "only ~30 % (Kaye et al., PPPL-4635, 2011); on MAST the X-point height changes P_th by up to 3x "
    "(Andrew et al. 2019)",
)


_LOSS_POWER = BoundaryQuantity(
    "loss_power", "P_L", "MW",
    "Loss power P_L = P_OHM + P_abs - dW/dt - P_Floss (Martin 2008, Eq. 1): Ohmic plus absorbed "
    "auxiliary power, minus the stored-energy change and fast-ion orbit and charge-exchange losses. "
    "Not P_aux, P_abs or P_SOL.",
)
_LINE_AVERAGE_DENSITY_1E20 = BoundaryQuantity(
    "line_average_density", r"\bar n_e", "1e20 m^-3",
    "Line-averaged electron density along a (near-)central chord; not the volume average.",
)
_PLASMA_SURFACE_AREA = BoundaryQuantity("plasma_surface_area", "S", "m^2", "Area of the last closed flux surface.")
_ASPECT_RATIO = BoundaryQuantity("aspect_ratio", "R/a", "-", "Major over minor radius.")

_register(Boundary(
    key="martin_2008_lh",
    family="lh_threshold",
    target=_LOSS_POWER,
    inputs=(_LINE_AVERAGE_DENSITY_1E20, _TOROIDAL_FIELD, _PLASMA_SURFACE_AREA),
    form="power_law",
    coefficient=0.0488,
    exponents={"line_average_density": 0.717, "toroidal_field": 0.803, "plasma_surface_area": 0.941},
    allowed_side="above",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="L_to_H",
    regime="L_mode",
    source_regime="L_mode",
    target_regime="H_mode",
    branch="high_density",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "ITPA threshold database, SELEC2007: deuterium, single null with the ion grad-B drift towards "
            "the X point, elongation >= 1.2, q95 >= 2.5, P_rad/P_L < 0.5, no Ohmic or ECRH-only transitions",
            "fitted devices: Alcator C-Mod, ASDEX Upgrade, DIII-D, JET, JFT-2M, JT-60U (1024 time slices); "
            "aspect ratio covers only a limited range; spherical tokamaks are not in the fit",
            "high-density branch: below the density of minimum threshold (ryter_2014_nmin) the measured "
            "threshold rises above the scaling",
            "compare with the loss power P_L, not the auxiliary or separatrix power; roughly 1/M with ion mass",
        ) + _LOW_ASPECT_RATIO_EVIDENCE,
    ),
    uncertainty=Uncertainty(
        coefficient_factor=float(np.exp(0.057)),
        exponents={"line_average_density": 0.035, "toroidal_field": 0.032, "plasma_surface_area": 0.019},
        rms_relative=0.308,
        note="Standard errors of the log-linear fit; RMS 30.8 %. Table 1 gives the 95 % interval for "
             "ITER: 28-96 MW at 0.5e20 m^-3 and 46-160 MW at 1e20 m^-3.",
    ),
    sources=(
        BoundarySource("Y. R. Martin et al., J. Phys.: Conf. Ser. 123 (2008) 012033",
                       equation="Eq. (2); P_L definition Eq. (1); Table 1", doi="10.1088/1742-6596/123/1/012033",
                       note="P_Thresh = 0.0488 e^(+-0.057) n_e20^(0.717+-0.035) B_T^(0.803+-0.032) "
                            "S^(0.941+-0.019), MW"),
        BoundarySource("F. Ryter et al., Nucl. Fusion 54 (2014) 083003", equation="Eq. (1), p. 2",
                       doi="10.1088/0029-5515/54/8/083003",
                       note="restates the scaling (0.049, 0.72, 0.80, 0.94) and that it is fitted on the "
                            "high-density branch"),
    ),
    notes="P_L / P_Thresh is the evaluation ratio. The permitted side is 'above': H-mode access needs "
          "P_L > P_Thresh. It is a normalisation against one published fit, not a transition prediction.",
))

_register(Boundary(
    key="ryter_2014_nmin",
    family="lh_threshold",
    target=_LINE_AVERAGE_DENSITY,
    inputs=(_PLASMA_CURRENT_MA, _TOROIDAL_FIELD, _MINOR_RADIUS, _ASPECT_RATIO),
    form="power_law",
    coefficient=0.7,
    exponents={"plasma_current": 0.34, "toroidal_field": 0.62, "minor_radius": -0.95, "aspect_ratio": 0.4},
    allowed_side="above",
    hardness="soft",
    origin="published",
    basis="semi_empirical",
    event="L_to_H_threshold_minimum",
    regime="L_mode",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=(
            "density at which the L-H power threshold is minimum; the high-density-branch scaling "
            "(martin_2008_lh) applies above it. Below it H-mode is still accessible but needs more power "
            "(the low-density branch), so 'above' marks the validity of martin_2008_lh, not H-mode access",
            "derived for deuterium from the Martin threshold scaling and an L-mode confinement scaling, "
            "with n_e,min set by tau_E / tau_ei = 9 (the ASDEX Upgrade minimum of P_L-H, Fig. 9)",
            "checked against ASDEX Upgrade (C and W walls), Alcator C-Mod, DIII-D, JET-ILW and JFT-2M "
            "(Fig. 10); the JT-60U prediction is too high",
        ),
    ),
    uncertainty=Uncertainty(note="No fit uncertainty is published; Fig. 10 compares measured and predicted n_e,min."),
    sources=(
        BoundarySource("F. Ryter et al., Nucl. Fusion 54 (2014) 083003", equation="Eq. (3), p. 7",
                       doi="10.1088/0029-5515/54/8/083003",
                       note="n_e,min ~ 0.7 I_p^0.34 B_T^0.62 a^-0.95 (R/a)^0.4 in 1e19 m^-3 using MA, T and m. "
                            "Eq. (4) for the minimum power is not registered: as printed it gives ~62 MW for ITER "
                            "at full field against the paper's ~41 MW, and ~22 against ~16 MW at half field. "
                            "Eq. (1) at n_e,min gives ~44 and ~16 MW"),
    ),
))

# Takizuka et al. (ITPA H-mode Power Threshold Database Working Group), PPCF 46 (2004) A227,
# Eq. (4): the threshold scaling that brings MAST and NSTX closer to conventional tokamaks
# through the absolute field at the outer midplane and an aspect-ratio factor F(A)^gamma.
# The paper gives gamma = 0.5 +- 0.5 ("rather uncertain"); 0.5 is its central value and the
# registered boundary's default, 0 and 1 bound the range, and Thome et al. 2017 evaluate the
# scaling at the maximum gamma = 1.
TAKIZUKA_GAMMA_DEFAULT = 0.5


def _takizuka_outer_field(toroidal_field, plasma_current, minor_radius, aspect_ratio):
    """|B|_out = (B_tout^2 + B_pout^2)^0.5, B_tout = B_t A/(A+1), B_pout = (mu0 I_p/2 pi a)(1 + 1/A)."""
    A = np.asarray(aspect_ratio, dtype=float)
    b_tout = np.asarray(toroidal_field, dtype=float) * A / (A + 1.0)
    b_pout = MU0 * np.asarray(plasma_current, dtype=float) * 1e6 / (2.0 * np.pi * np.asarray(minor_radius, dtype=float)) * (1.0 + 1.0 / A)
    return np.sqrt(b_tout**2 + b_pout**2)


def _takizuka_aspect_factor(aspect_ratio):
    """F(A) = 0.1 A / f(A) with the untrapped fraction f(A) = 1 - (2/(1+A))^0.5."""
    A = np.asarray(aspect_ratio, dtype=float)
    return 0.1 * A / (1.0 - np.sqrt(2.0 / (1.0 + A)))


def _takizuka_2004_threshold(line_average_density, toroidal_field, plasma_current, minor_radius,
                             aspect_ratio, plasma_surface_area, effective_charge, *,
                             gamma=TAKIZUKA_GAMMA_DEFAULT):
    """Eq. (4) of Takizuka et al. 2004 in MW, with the aspect-ratio exponent explicit.

    ``gamma`` is the exponent of F(A): the paper gives 0.5 +- 0.5, so the
    published range is 0 <= gamma <= 1 with 0.5 the central value used by the
    registered boundary; Thome et al. 2017 evaluate the scaling at gamma = 1.
    F(A)^gamma is 1.0-1.85 across that range at NSTX (A = 1.32) and 1.0-1.22 at
    VEST (A = 1.7). A value outside [0, 1] is refused: it is outside what the
    source states.
    """
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in the published range [0, 1]; got {gamma!r}")
    b_out = _takizuka_outer_field(toroidal_field, plasma_current, minor_radius, aspect_ratio)
    return (0.072 * b_out**0.7 * np.asarray(line_average_density, dtype=float) ** 0.7
            * np.asarray(plasma_surface_area, dtype=float) ** 0.9
            * (np.asarray(effective_charge, dtype=float) / 2.0) ** 0.7
            * _takizuka_aspect_factor(aspect_ratio) ** gamma)



_register(Boundary(
    key="takizuka_2004_lh",
    family="lh_threshold",
    target=_LOSS_POWER,
    inputs=(_LINE_AVERAGE_DENSITY_1E20, _TOROIDAL_FIELD, _PLASMA_CURRENT_MA, _MINOR_RADIUS, _ASPECT_RATIO,
            _PLASMA_SURFACE_AREA,
            BoundaryQuantity("effective_charge", "Z_eff", "-", "Effective ion charge.")),
    form="function",
    function=_takizuka_2004_threshold,
    allowed_side="above",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="L_to_H",
    regime="L_mode",
    source_regime="L_mode",
    target_regime="H_mode",
    applicability=Applicability(
        machine_class="tokamak including spherical tokamaks",
        assumptions=(
            "ITPA threshold database (2003) including MAST and NSTX; conventional data span 2.4 < A < 6.2",
            "F(A)^gamma with gamma = 0.5 +- 0.5 is 'rather uncertain' (the paper's own words); even with it "
            "NSTX sits ~2x above the scaling (MAST ~1.0x)",
            "the Z_eff dependence rests on a sparse Z_eff store; the paper imposes Z_eff = 2 where it is missing",
        ) + _LOW_ASPECT_RATIO_EVIDENCE,
    ),
    uncertainty=Uncertainty(
        note="gamma = 0.5 +- 0.5 for the F(A)^gamma factor. Scatter sigma = 0.31 of ln(P_thr/P_thr,new). "
             "The ITER prediction band is 25-70 MW (from the S-exponent error and the 2-sigma JT-60U scatter).",
    ),
    sources=(
        BoundarySource(
            "ITPA H-mode Power Threshold Database Working Group (presented by T. Takizuka), "
            "Plasma Phys. Control. Fusion 46 (2004) A227",
            equation="Eq. (4), p. A232; |B|_out definition p. A229; F(A) and f(A) p. A232",
            doi="10.1088/0741-3335/46/5A/024",
            note="P_thr,new = 0.072 |B|_out^0.7 n20^0.7 S^0.9 (Z_eff/2)^0.7 F(A)^gamma [MW]. The printed "
                 "|B|_out definition gives 4.47 T for ITER where the text quotes 4.3 T (Eq. (2) then 44.8 "
                 "against 43 MW)",
        ),
        BoundarySource("K. E. Thome et al., Nucl. Fusion 57 (2017) 022018", equation="Sec. 2 and abstract",
                       doi="10.1088/0029-5515/57/2/022018",
                       note="Pegasus A ~ 1.2: P_LH exceeds ITPA08 by 7-15x; uses this scaling as 'ITPA04' and still "
                            "finds P_LH ~6x above it with gamma = 1 and Z_eff ~ 1 (Sec. 4)"),
        BoundarySource("S. M. Kaye et al., 'L-H threshold studies in NSTX', PPPL-4635 (2011)",
                       equation="Sec. II.C",
                       note="Ip dependence of P_LH at low A larger than the |B|_out form explains"),
        BoundarySource("Y. Andrew et al., Plasma 2 (2019) 328", equation="abstract", doi="10.3390/plasma2030024",
                       note="MAST: P_th rises 3x over a 10-12 cm X-point height scan"),
    ),
    notes="The registered boundary evaluates F(A)^gamma at the paper's central value gamma = 0.5 "
          "(TAKIZUKA_GAMMA_DEFAULT); gamma = 0 and 1 bound its range, and _takizuka_2004_threshold "
          "takes gamma as a keyword to evaluate either bound (Thome 2017 use gamma = 1). At low "
          "aspect ratio even this scaling underestimates measured thresholds (Pegasus ~6x, Thome 2017). "
          "A VEST-like device (A ~ 1.7, B_T ~ 0.15 T; the repo's own values, not from these papers) lies "
          "inside the fitted aspect-ratio range but below the field and size of the fitted data: treat "
          "the ratio as indicative only.",
))


# ------------------------------------------------------------------
# Internal inductance - edge q (#1422)
# ------------------------------------------------------------------
#
# Two different objects, kept apart on purpose:
#
# * Wesson et al. 1989, Fig. 6 (p. 645): the JET *empirical* l_i-q_psi operating
#   space. The lower boundary (labelled "Empirical stability boundary") is the
#   stability boundary for rotating MHD modes during the current rise, with the
#   kink and double-tearing region below it; the upper boundary is where
#   density-limit disruptions occur. l_i = 2/(mu0^2 R I^2) int B_theta^2 dtau
#   over the plasma volume (p. 645), and the x axis is q_psi, "the actual value
#   of q at the plasma edge" (p. 642).
# * Cheng, Furth and Boozer 1987, Fig. 4 (p. 357): the *theoretical* domain of
#   MHD-stable current profiles of a pressureless straight cylinder without a
#   conducting wall, for q(0) = 1.01, computed for m, n <= 20 (p. 352). The
#   lower (jig-saw) bound is mainly ideal external kinks, the upper bound
#   low-order resistive kinks, mainly m/n = 2/1 and 3/2 (p. 354). The figure
#   plots l_i/2; the tables below are l_i.
#
# The vertices were digitized for #1422 from 600-dpi renders of the published
# pages, with the axes calibrated on the printed tick marks. For Cheng Fig. 4
# the printed closed form MAX(l_i/2) = [1 + 2 ln(q(a)/q(0))]/4 is recovered to
# within 0.01 in l_i/2, which bounds the digitization error. The earlier
# vaft.formula.stability.empirical_li_qa arrays are the same Wesson lower
# boundary (within ~0.03); its "Fig. 5" attribution was wrong (Fig. 5 is the
# Hugill diagram).

_LI3 = BoundaryQuantity(
    "internal_inductance_li3", "l_i(3)", "-",
    "2 int B_p^2 dV / (mu0^2 I_p^2 R): the IMAS DD global_quantities.li_3 form. Wesson 1989 (p. 645) uses "
    "this form with R the plasma major radius; the DD uses the reference major radius R_0. l_i(3) scales as "
    "1/R, so the two differ by R_geo/R_0 where those differ (a few percent on VEST, r0 = 0.4 m); carry the "
    "radius used alongside the value.",
)
_LI_CYLINDER = BoundaryQuantity(
    "internal_inductance_cylinder", "l_i", "-",
    "Internal inductance of a straight circular cylinder, 2 int_0^a B_theta^2 r dr / (a^2 B_theta(a)^2). "
    "Equal to the l_i(3) form in the cylinder limit; a toroidal l_i is not this quantity.",
)
_Q_A_CYLINDER = BoundaryQuantity(
    "cylinder_edge_safety_factor", "q(a)", "-",
    "Edge safety factor of the straight-cylinder model of Cheng et al. 1987, q(a) = a B_z / (R B_theta(a)) "
    "with the periodicity length 2 pi R; neither q_psi, q95 nor the shaped q_cyl of a toroidal plasma.",
)

#: Wesson 1989 Fig. 6 lower boundary: at each integer q_psi, l_i(3) at the top and
#: bottom of the vertical edge of the tooth; between integers the boundary rises
#: linearly from bottom(n) to top(n + 1).
_WESSON_1989_TEETH = (
    # q_psi, top, bottom
    (2.0, 0.956, 0.687),
    (3.0, 0.931, 0.609),
    (4.0, 0.883, 0.492),
    (5.0, 0.715, 0.430),
    (6.0, 0.700, 0.338),
    (7.0, 0.674, 0.294),
    (8.0, 0.678, 0.298),
    (9.0, 0.674, 0.294),
    (10.0, 0.678, 0.295),
)
#: Wesson 1989 Fig. 6 upper boundary (density-limit disruptions), l_i(3) against q_psi.
_WESSON_1989_UPPER = (
    (2.0, 0.954), (2.5, 1.029), (3.0, 1.111), (3.5, 1.190), (4.0, 1.261), (4.5, 1.324), (5.0, 1.393),
    (5.5, 1.460), (6.0, 1.523), (6.5, 1.584), (7.0, 1.630), (7.5, 1.677), (8.0, 1.722), (8.5, 1.760),
    (9.0, 1.798), (9.5, 1.834), (10.0, 1.866),
)
#: Cheng 1987 Fig. 4 lower (jig-saw) bound, l_i/2 as printed: (q(a), top, bottom) as above.
#: Beyond q(a) = 6 the bound is a slowly rising curve, given by _CHENG_1987_LOWER_TAIL.
_CHENG_1987_TEETH = (
    (2.0, 0.545, 0.345),
    (3.0, 0.505, 0.355),
    (4.0, 0.489, 0.355),
    (5.0, 0.465, 0.413),
    (6.0, 0.457, 0.442),
)
_CHENG_1987_LOWER_TAIL = (
    (6.0, 0.442), (6.25, 0.444), (6.5, 0.448), (6.75, 0.451), (7.0, 0.455), (7.25, 0.457), (7.5, 0.460),
    (7.75, 0.462),
)
#: Cheng 1987 Fig. 4 upper bound (low-order resistive kinks), l_i/2 against q(a).
_CHENG_1987_UPPER = (
    (2.0, 0.545), (2.25, 0.588), (2.5, 0.628), (2.75, 0.665), (3.0, 0.699), (3.25, 0.731), (3.5, 0.760),
    (3.75, 0.788), (4.0, 0.814), (4.25, 0.840), (4.5, 0.865), (4.75, 0.890), (5.0, 0.914), (5.25, 0.937),
    (5.5, 0.960), (5.75, 0.981), (6.0, 1.001), (6.25, 1.021), (6.5, 1.039), (6.75, 1.056), (7.0, 1.072),
    (7.25, 1.088), (7.5, 1.103), (7.75, 1.116),
)


def _sawtooth(q, teeth, tail=None):
    """Piecewise-linear jig-saw: on [n, n+1) from bottom(n) up to top(n+1); NaN outside the drawn range."""
    q = np.asarray(q, dtype=float)
    out = np.full(q.shape, np.nan)
    for (q0, _, bottom), (q1, top, _) in zip(teeth[:-1], teeth[1:]):
        inside = (q >= q0) & (q < q1)
        out[inside] = bottom + (top - bottom) * (q[inside] - q0) / (q1 - q0)
    q_last, top_last, bottom_last = teeth[-1]
    if tail is None:  # the last vertical edge, like every other one, takes its bottom value at q itself
        out[q == q_last] = bottom_last if bottom_last is not None else top_last
    else:
        tq, tv = np.array(tail).T
        inside = (q >= tq[0]) & (q <= tq[-1])
        out[inside] = np.interp(q[inside], tq, tv)
    return out


def _tabulated(q, table):
    q = np.asarray(q, dtype=float)
    tq, tv = np.array(table).T
    return np.where((q >= tq[0]) & (q <= tq[-1]), np.interp(q, tq, tv), np.nan)


_WESSON_SOURCE = BoundarySource(
    "J. A. Wesson et al., Nucl. Fusion 29 (1989) 641", equation="Fig. 6 and l_i definition, p. 645",
    doi="10.1088/0029-5515/29/4/009",
    note="digitized from the published figure (#1422); axes q_psi and l_i = 2 int B_theta^2 dtau/(mu0^2 R I^2)",
)
_WESSON_APPLICABILITY = dict(
    machine_class="JET (conventional aspect ratio, R = 3 m), 1985-88 operation",
    ranges={"edge_safety_factor": (2.0, 10.0)},
    assumptions=(
        "empirical JET operating boundary, not a stability calculation",
        "x is q_psi at the plasma edge, not q95 and not the cylindrical q_c of the Hugill diagram",
        "transfer to spherical tokamaks is untested: low aspect ratio changes both q_psi and l_i at fixed profile",
    ),
)

_register(Boundary(
    key="wesson_1989_jet_li_qpsi_lower",
    family="li_q",
    target=_LI3,
    inputs=(_EDGE_Q_MHD,),
    form="function",
    function=lambda edge_safety_factor: _sawtooth(edge_safety_factor, _WESSON_1989_TEETH),
    allowed_side="above",
    hardness="soft",
    origin="reproduced",
    basis="empirical",
    event="mhd_instability",
    branch="lower",
    applicability=Applicability(**_WESSON_APPLICABILITY),
    sources=(_WESSON_SOURCE,),
    notes="'Empirical stability boundary': rotating MHD modes during the current rise; below it is the kink "
          "and double tearing region. Same boundary as the legacy stability.empirical_li_qa arrays.",
))

_register(Boundary(
    key="wesson_1989_jet_li_qpsi_upper",
    family="li_q",
    target=_LI3,
    inputs=(_EDGE_Q_MHD,),
    form="function",
    function=lambda edge_safety_factor: _tabulated(edge_safety_factor, _WESSON_1989_UPPER),
    allowed_side="below",
    hardness="soft",
    origin="reproduced",
    basis="empirical",
    event="disruption",
    branch="upper",
    applicability=Applicability(**_WESSON_APPLICABILITY),
    sources=(_WESSON_SOURCE,),
    notes="Upper boundary of Fig. 6: the region where major (density-limit) disruptions occur.",
))

_CHENG_SOURCE = BoundarySource(
    "C. Z. Cheng, H. P. Furth and A. H. Boozer, Plasma Phys. Control. Fusion 29 (1987) 351",
    equation="Fig. 4, p. 357 (plots l_i/2); bound origins p. 354; m, n <= 20 p. 352",
    doi="10.1088/0741-3335/29/3/006",
    note="digitized from the published figure (#1422); stored as l_i = 2 x the plotted l_i/2",
)
_CHENG_APPLICABILITY = dict(
    machine_class="theory: pressureless straight circular cylinder, no conducting wall",
    ranges={"cylinder_edge_safety_factor": (2.0, 7.75)},
    assumptions=(
        "q(0) = 1.01; for q(0) >= 1 no stable profile exists below q(a) = 2",
        "modes up to m, n = 20 examined; a conducting wall relaxes the lower bound but barely moves the upper",
        "toroidicity, finite beta and a separatrix are not included (p. 366)",
    ),
)

_register(Boundary(
    key="cheng_1987_li_qa_lower",
    family="li_q",
    target=_LI_CYLINDER,
    inputs=(_Q_A_CYLINDER,),
    form="function",
    function=lambda cylinder_edge_safety_factor: 2.0 * _sawtooth(
        cylinder_edge_safety_factor, _CHENG_1987_TEETH, _CHENG_1987_LOWER_TAIL),
    allowed_side="above",
    hardness="hard",
    origin="reproduced",
    basis="ideal_mhd_numerical",
    event="external_kink",
    branch="lower",
    applicability=Applicability(**_CHENG_APPLICABILITY),
    sources=(_CHENG_SOURCE,),
    notes="Jig-saw lower bound of the MHD-stable domain, mainly ideal external kinks.",
))

_register(Boundary(
    key="cheng_1987_li_qa_upper",
    family="li_q",
    target=_LI_CYLINDER,
    inputs=(_Q_A_CYLINDER,),
    form="function",
    function=lambda cylinder_edge_safety_factor: 2.0 * _tabulated(cylinder_edge_safety_factor, _CHENG_1987_UPPER),
    allowed_side="below",
    hardness="hard",
    origin="reproduced",
    basis="resistive_mhd_numerical",
    event="resistive_kink",
    branch="upper",
    applicability=Applicability(**_CHENG_APPLICABILITY),
    sources=(_CHENG_SOURCE,),
    notes="Upper bound of the MHD-stable domain, low-order resistive kinks (mainly m/n = 2/1 and 3/2).",
))

_register(Boundary(
    key="cheng_1987_qa_min",
    family="li_q",
    target=_Q_A_CYLINDER,
    inputs=(),
    form="threshold",
    coefficient=2.0,
    allowed_side="above",
    hardness="hard",
    origin="published",
    basis="theoretical",
    event="external_kink",
    applicability=Applicability(**{**_CHENG_APPLICABILITY, "ranges": {}}),
    sources=(BoundarySource(
        "C. Z. Cheng, H. P. Furth and A. H. Boozer, Plasma Phys. Control. Fusion 29 (1987) 351",
        equation="p. 354 ('For q(0) = 1, there can be no stability when q(a) < 2'); left edge of Fig. 4",
        doi="10.1088/0741-3335/29/3/006",
    ),),
    notes="The left edge of the stable domain for q(0) >= 1. On the cylinder's q(a), not q_psi (that is 'low_q').",
))


# ------------------------------------------------------------------
# Low-beta external kink limit of an elongated tokamak (Freidberg 2008)
# ------------------------------------------------------------------
#
# J. P. Freidberg, Plasma Physics and Fusion Energy (CUP 2008), Sec. 13 (pp. 405-406):
# the kink safety factor of an elliptical tokamak is Eq. (13.158) with G(kappa) ~ 1 over
# 1 < kappa < 2, i.e. Eq. (13.160), q* = 2 pi a^2 kappa B0 / (mu0 R0 I). The low-beta kink
# limit Eq. (13.162), q* >= (1 + kappa)/2, is stated for THAT definition, and Eq. (13.163)
# is the same limit written as a maximum current, I_max = (2 pi a^2 B0 / mu0 R0) 2 kappa/(1 + kappa).
# The (1 + kappa^2)/2 form of Eq. (13.171) is the Princeton-group definition used with the
# beta_N fit (13.172); it is a different quantity and is not paired with (13.162) here (#1524).

_KINK_Q_STAR = BoundaryQuantity(
    "kink_safety_factor_elliptic", "q_*", "-",
    "Freidberg (2008) Eq. (13.160): 2 pi a^2 kappa B0 / (mu0 R0 I), the G(kappa) ~ 1 form of Eq. (13.158). "
    "Not the (1 + kappa^2)/2 definition of Eq. (13.171).",
)
_ELONGATION = BoundaryQuantity("elongation", "kappa", "-", "Plasma elongation.")
_FREIDBERG_SOURCE = dict(citation="J. P. Freidberg, Plasma Physics and Fusion Energy, Cambridge University Press (2008)",
                         doi="10.1017/CBO9780511755705")
_FREIDBERG_APPLICABILITY = dict(
    machine_class="tokamak, elongated elliptical cross-section",
    ranges={"elongation": (1.0, 2.0)},
    assumptions=(
        "low-beta external kink from the surface-current model; the coupled-harmonic result is approximate",
        "q* is Eq. (13.160) (kappa in the numerator), not q95 and not the (1 + kappa^2)/2 form",
        "derived for conventional aspect ratio; spherical-tokamak use is an extrapolation",
    ),
)


def kink_coordinates(a, R0, B0, kappa, I_p):
    r"""Freidberg's kink safety factor of an elongated tokamak, $q_* = 2\pi a^2\kappa B_0/(\mu_0 R_0 I)$.

    $$q_* = \frac{2\pi a^2 \kappa B_0}{\mu_0 R_0 I_p} = \frac{5\,a^2\kappa B_0}{R_0\,I_p[\mathrm{MA}]}$$

    Parameters
    ----------
    a : float or np.ndarray
        Minor radius [m].
    R0 : float or np.ndarray
        Major radius [m].
    B0 : float or np.ndarray
        Vacuum toroidal field at ``R0``; its sign is dropped [T].
    kappa : float or np.ndarray
        Elongation [-].
    I_p : float or np.ndarray
        Plasma current; its sign is dropped [MA].

    Returns
    -------
    float or np.ndarray
        Kink safety factor $q_*$ of Eq. (13.160), NaN where an input is NaN [-].

    Raises
    ------
    ValueError
        A finite non-positive (or infinite) minor radius, major radius, field
        magnitude or elongation.

    Convention
    ----------
    Eq. (13.160) is the $G(\kappa) \approx 1$ form of Eq. (13.158), the definition with
    which the kink limit ``"freidberg_2008_kink_qstar"`` (Eq. 13.162) is stated. It is
    not ``vaft.formula.equilibrium.kink_safety_factor`` (#1524). A zero current maps to
    an infinite $q_*$.

    References
    ----------
    .. [1] J. P. Freidberg, *Plasma Physics and Fusion Energy*, Cambridge University
           Press (2008), Eqs. (13.158)-(13.160), p. 405.
    """
    a_arr, R, B_abs, k = (np.asarray(v, dtype=float) for v in (a, R0, np.abs(np.asarray(B0, dtype=float)), kappa))
    for name, value in (("a", a_arr), ("R0", R), ("B0", B_abs), ("kappa", k)):
        # NaN propagates (a population table has gaps); a finite non-positive value or an infinity is an error
        if np.any(np.isinf(value)) or np.any(np.isfinite(value) & ~(value > 0)):
            raise ValueError(f"{name} must be positive and finite where it is given")
    current = np.abs(np.asarray(I_p, dtype=float)) * 1e6
    with np.errstate(divide="ignore"):
        q = 2.0 * np.pi * a_arr**2 * k * B_abs / (MU0 * R * current)
    return _scalar_or_array(q)


_register(Boundary(
    key="freidberg_2008_kink_qstar",
    family="current_limit",
    target=_KINK_Q_STAR,
    inputs=(_ELONGATION,),
    form="function",
    function=lambda elongation: (1.0 + np.asarray(elongation, dtype=float)) / 2.0,
    allowed_side="above",
    hardness="hard",
    origin="published",
    basis="ideal_mhd_analytic",
    event="external_kink",
    applicability=Applicability(**_FREIDBERG_APPLICABILITY),
    sources=(BoundarySource(**_FREIDBERG_SOURCE, equation="Eq. (13.162), p. 406",
                            note="q* >= (1 + kappa)/2 for 1 < kappa < 2, with q* of Eq. (13.160)"),),
    notes="The minimum stable kink safety factor rises with elongation; the current limit it implies is "
          "'freidberg_2008_kink_current'.",
))

_register(Boundary(
    key="freidberg_2008_kink_current",
    family="current_limit",
    target=_PLASMA_CURRENT_MA,
    inputs=(_MINOR_RADIUS, _MAJOR_RADIUS, _TOROIDAL_FIELD, _ELONGATION),
    form="function",
    function=lambda minor_radius, major_radius, toroidal_field, elongation: (
        2.0 * np.pi * np.asarray(minor_radius, dtype=float) ** 2 * np.abs(np.asarray(toroidal_field, dtype=float))
        / (MU0 * np.asarray(major_radius, dtype=float))
        * 2.0 * np.asarray(elongation, dtype=float) / (1.0 + np.asarray(elongation, dtype=float)) * 1e-6),
    allowed_side="below",
    hardness="hard",
    origin="published",
    basis="ideal_mhd_analytic",
    event="external_kink",
    applicability=Applicability(**_FREIDBERG_APPLICABILITY),
    sources=(BoundarySource(**_FREIDBERG_SOURCE, equation="Eq. (13.163), p. 406",
                            note="I_max = (2 pi a^2 B0 / mu0 R0) 2 kappa/(1 + kappa): Eq. (13.162) through Eq. (13.160)"),),
    notes="Rises by 4/3 from kappa = 1 to 2. The printed Eq. (7) of Yun et al., PPCF 67 (2025) 115021 does not "
          "reproduce its own Fig. 8(b); this relation does (#1524).",
))


# ------------------------------------------------------------------
# Edge-q proxies from global shape and engineering parameters (#1456)
# ------------------------------------------------------------------
#
# Two families, kept apart (they are different quantities and neither is q_a):
#
# * Menard's cylindrical safety factor (Phys. Plasmas 11 (2004) 639, PPPL-3908 preprint p. 9, after
#   D'Ippolito et al., Phys. Fluids 21 (1978) 1600): q* = eps (1 + kappa^2) pi a B_T0 / (mu0 I_P)
#   = pi a^2 B_T0 (1 + kappa^2) / (mu0 R0 I_P), the (1 + kappa^2)/2 form of Freidberg Eq. (13.171).
#   In Menard's low-aspect-ratio ideal no-wall scans, <beta_N> degrades below q* = 2 and no stable
#   case is found below q* = 1.
# * The ITER design-guideline q95 (Post et al., ITER Physics, ITER Documentation Series No. 21, IAEA
#   1991, Table 1-2, after Uckan, ITER Physics Design Guidelines, IAEA/ITER/DS-10, 1990, p. 10):
#   q_psi(95%) ~ q_* f(eps), with q_* = (5 a^2 B / R I)[1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)]/2 and
#   f(eps) = (1.17 - 0.65 eps)/(1 - eps^2)^2; guideline q95 >= 3.0 (baseline) | 2.1 (extended
#   performance) for kappa < 2. A conventional-aspect-ratio fit: at VEST's A ~ 1.3 it is an
#   extrapolation (Akers et al., NF 40 (2000) 1223, give a START-based ST correction, not registered
#   here until its coefficients are read from the paper).

_KINK_Q_STAR_CYL = BoundaryQuantity(
    "kink_safety_factor_cylindrical", "q^*", "-",
    "Menard et al. (2004) cylindrical safety factor pi a^2 B_T0 (1 + kappa^2)/(mu0 R0 I_P) "
    "(Freidberg Eq. 13.171). Not q95, not q_a, not Freidberg's Eq. (13.160).",
)
_Q95 = BoundaryQuantity("edge_safety_factor_95", "q95", "-", "Safety factor at the 95 % flux surface.")
_Q95_ITER_ESTIMATE = BoundaryQuantity(
    "edge_safety_factor_95_estimate_iter", "q_{95,ITER}", "-",
    "q95 from the ITER design-guideline formula (Post et al. 1991, Table 1-2) evaluated on global shape "
    "and engineering parameters; an estimate, not an equilibrium q95.",
)
_TRIANGULARITY = BoundaryQuantity("triangularity", "delta", "-", "Plasma triangularity (average of upper and lower).")
_MENARD_2004_SOURCE = BoundarySource(
    "J. E. Menard et al., Phys. Plasmas 11 (2004) 639 (preprint PPPL-3908)",
    equation="p. 9 of PPPL-3908: q* = eps (1 + kappa^2) pi a B_T0 / mu0 I_P; Fig. 3d",
    doi="10.1063/1.1640623",
    note="after D'Ippolito, Freidberg, Goedbloed and Rem, Phys. Fluids 21 (1978) 1600",
)
_POST_1991_SOURCE = BoundarySource(
    "D. E. Post et al., ITER Physics, ITER Documentation Series No. 21, IAEA, Vienna (1991)",
    equation="Table 1-2 (Summary of ITER physics guidelines): q_psi(95%) ~ q_* f(eps)",
    note="after N. A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, IAEA/ITER/DS-10 (1990), "
         "p. 10; '(x | y)' is (baseline performance | extended performance)",
)


def _positive_geometry(**values):
    out = {}
    for name, value in values.items():
        arr = np.asarray(value, dtype=float)
        if np.any(np.isinf(arr)) or np.any(np.isfinite(arr) & ~(arr > 0)):
            raise ValueError(f"{name} must be positive and finite where it is given")
        out[name] = arr
    return out


def cylindrical_kink_coordinates(a, R0, B0, kappa, I_p):
    r"""Menard's cylindrical safety factor, $q^* = \pi a^2 B_{T0}(1+\kappa^2)/(\mu_0 R_0 I_P)$.

    $$q^* = \epsilon\,(1+\kappa^2)\,\frac{\pi a B_{T0}}{\mu_0 I_P}
          = \frac{\pi a^2 B_{T0}\,(1+\kappa^2)}{\mu_0 R_0 I_P}$$

    Parameters
    ----------
    a : float or np.ndarray
        Minor radius [m].
    R0 : float or np.ndarray
        Major radius [m].
    B0 : float or np.ndarray
        Vacuum toroidal field at ``R0``; its sign is dropped [T].
    kappa : float or np.ndarray
        Elongation [-].
    I_p : float or np.ndarray
        Plasma current; its sign is dropped [MA].

    Returns
    -------
    float or np.ndarray
        Cylindrical safety factor $q^*$, NaN where an input is NaN [-].

    Raises
    ------
    ValueError
        A finite non-positive (or infinite) minor radius, major radius, field
        magnitude or elongation.

    Convention
    ----------
    A global, shape-weighted proxy of the edge safety factor, not $q_a$ or $q_{95}$:
    Menard et al. use it because $q(1)$ and $q(0.95)$ at the current limit vary by a
    factor two with aspect ratio and shape while $q^*$ does not. It is the
    $(1+\kappa^2)/2$ form of Freidberg Eq. (13.171), and differs from Freidberg's
    Eq. (13.160) (``kink_coordinates``) by the factor $(1+\kappa^2)/(2\kappa)$.

    References
    ----------
    .. [1] J. E. Menard et al., Phys. Plasmas 11 (2004) 639; preprint PPPL-3908, p. 9.
    .. [2] D. A. D'Ippolito, J. P. Freidberg, J. P. Goedbloed and J. Rem, Phys. Fluids 21 (1978) 1600.
    """
    g = _positive_geometry(a=a, R0=R0, B0=np.abs(np.asarray(B0, dtype=float)), kappa=kappa)
    current = np.abs(np.asarray(I_p, dtype=float)) * 1e6
    with np.errstate(divide="ignore"):
        q = np.pi * g["a"] ** 2 * g["B0"] * (1.0 + g["kappa"] ** 2) / (MU0 * g["R0"] * current)
    return _scalar_or_array(q)


def _finite_field(toroidal_field):
    """|B_T|; zero is allowed (a current limit is a line through the origin of the I_p-B_T plane)."""
    b = np.abs(np.asarray(toroidal_field, dtype=float))
    if np.any(np.isinf(b)):
        raise ValueError("toroidal_field must be finite where it is given")
    return b


def _menard_current(minor_radius, major_radius, toroidal_field, elongation):
    g = _positive_geometry(a=minor_radius, R0=major_radius, kappa=elongation)
    b = _finite_field(toroidal_field)
    return np.pi * g["a"] ** 2 * b * (1.0 + g["kappa"] ** 2) / (MU0 * g["R0"]) * 1e-6


def _iter_current(minor_radius, major_radius, toroidal_field, elongation, triangularity, q95=2.1):
    g = _positive_geometry(a=minor_radius, R0=major_radius, kappa=elongation)
    b = _finite_field(toroidal_field)
    if np.any(np.isfinite(g["a"] / g["R0"]) & (g["a"] / g["R0"] >= 1.0)):
        raise ValueError("a must be smaller than R0")
    return _iter_q95_per_ma(g["a"], g["R0"], b, g["kappa"], np.asarray(triangularity, dtype=float)) / q95


def _iter_q95_per_ma(a, R0, B0, kappa, delta):
    """q95 * I_p[MA] of the ITER guideline formula (Post et al. 1991, Table 1-2)."""
    eps = a / R0
    shape = (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
    geometry = (1.17 - 0.65 * eps) / (1.0 - eps**2) ** 2
    return 5.0 * a**2 * B0 / R0 * shape * geometry


def iter_q95_coordinates(a, R0, B0, kappa, delta, I_p):
    r"""The ITER design-guideline $q_{95}$ estimate from global shape, $q_{95} \approx q_* f(\epsilon)$.

    $$q_{95} \approx \frac{5a^2B}{R\,I_p[\mathrm{MA}]}\,
      \frac{1+\kappa^2(1+2\delta^2-1.2\delta^3)}{2}\,\frac{1.17-0.65\epsilon}{(1-\epsilon^2)^2},
      \qquad \epsilon = a/R$$

    Parameters
    ----------
    a : float or np.ndarray
        Minor radius [m].
    R0 : float or np.ndarray
        Major radius [m].
    B0 : float or np.ndarray
        Vacuum toroidal field at ``R0``; its sign is dropped [T].
    kappa : float or np.ndarray
        Elongation of the 95 % flux surface, kappa_95 [-].
    delta : float or np.ndarray
        Triangularity of the 95 % flux surface, delta_95; may be zero or negative [-].
    I_p : float or np.ndarray
        Plasma current; its sign is dropped [MA].

    Returns
    -------
    float or np.ndarray
        Estimated $q_{95}$, NaN where an input is NaN [-].

    Raises
    ------
    ValueError
        A finite non-positive (or infinite) minor radius, major radius, field
        magnitude or elongation, or $a \ge R_0$.

    Convention
    ----------
    The formula is calibrated on the 95 % flux-surface shape ($\kappa_{95}$,
    $\delta_{95}$); fed the last-closed-surface $\kappa$ and $\delta$, which are larger,
    it over-estimates $q_{95}$ (for the ITER design point by about 27 %).
    A fit for conventional aspect ratio used for ITER design; at a spherical
    tokamak's $A \approx 1.3$ it is an extrapolation and over-estimates $q_{95}$
    (Akers et al. 2000 give a START-based correction, not registered here). It is an
    estimate from global parameters, not an equilibrium $q_{95}$.

    References
    ----------
    .. [1] D. E. Post et al., *ITER Physics*, ITER Documentation Series No. 21, IAEA (1991),
           Table 1-2.
    .. [2] N. A. Uckan and ITER Physics Group, *ITER Physics Design Guidelines: 1989*,
           IAEA/ITER/DS-10, IAEA (1990), p. 10.
    """
    g = _positive_geometry(a=a, R0=R0, B0=np.abs(np.asarray(B0, dtype=float)), kappa=kappa)
    if np.any(np.isfinite(g["a"] / g["R0"]) & (g["a"] / g["R0"] >= 1.0)):
        raise ValueError("a must be smaller than R0")
    d = np.asarray(delta, dtype=float)
    if np.any(np.isinf(d)):
        raise ValueError("delta must be finite where it is given")
    current = np.abs(np.asarray(I_p, dtype=float))
    with np.errstate(divide="ignore"):
        q = _iter_q95_per_ma(g["a"], g["R0"], g["B0"], g["kappa"], d) / current
    return _scalar_or_array(q)


__all__ += ["cylindrical_kink_coordinates", "iter_q95_coordinates"]

_register(Boundary(
    key="menard_2004_qstar_min",
    family="current_limit",
    target=_KINK_Q_STAR_CYL,
    inputs=(),
    form="threshold",
    coefficient=1.0,
    allowed_side="above",
    hardness="hard",
    origin="published",
    basis="ideal_mhd_numerical",
    event="external_kink",
    applicability=Applicability(
        machine_class="tokamak including spherical tokamaks",
        ranges={},
        assumptions=(
            "ideal MHD, no wall, high bootstrap fraction equilibria at A = 1.6-3.3 (NSTX-like shapes)",
            "'no stable cases are found with q* below 1'; <beta_N> already degrades below q* = 2",
        ),
    ),
    sources=(_MENARD_2004_SOURCE,),
    notes="The current limit on Menard's q*; the beta_N degradation below q* = 2 is a softer, beta-dependent bound.",
))

_register(Boundary(
    key="menard_2004_qstar_current",
    family="current_limit",
    target=_PLASMA_CURRENT_MA,
    inputs=(_MINOR_RADIUS, _MAJOR_RADIUS, _TOROIDAL_FIELD, _ELONGATION),
    form="function",
    function=_menard_current,
    allowed_side="below",
    hardness="hard",
    origin="derived",
    basis="ideal_mhd_numerical",
    event="external_kink",
    applicability=Applicability(
        machine_class="tokamak including spherical tokamaks",
        assumptions=("'menard_2004_qstar_min' (q* >= 1) written as a maximum current through Menard's q*",
                     "Menard's scans cover A = 1.6-3.3; below A = 1.6 it is an extrapolation"),
    ),
    sources=(_MENARD_2004_SOURCE,),
    notes="I_max = pi a^2 B_T0 (1 + kappa^2)/(mu0 R0); the q* = 1 limit as a current.",
))

_register(Boundary(
    key="iter_1991_q95_min",
    family="current_limit",
    target=_Q95,
    inputs=(),
    form="threshold",
    coefficient=2.1,
    allowed_side="above",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="tokamak",
        ranges={},
        assumptions=(
            "ITER design guideline for kappa = b/a < 2: q95 >= 3.0 (baseline) | 2.1 (extended performance)",
            "conventional aspect ratio (ITER A = 3); a design margin, not a measured stability limit",
        ),
    ),
    sources=(_POST_1991_SOURCE,),
    notes="The extended-performance value; the baseline guideline is 3.0.",
))

_register(Boundary(
    key="iter_1991_q95_estimate_min",
    family="current_limit",
    target=_Q95_ITER_ESTIMATE,
    inputs=(),
    form="threshold",
    coefficient=2.1,
    allowed_side="above",
    hardness="soft",
    origin="published",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="tokamak",
        ranges={},
        assumptions=(
            "the same guideline applied to the guideline's own q95 formula, as the ITER design does",
            "conventional aspect ratio; at A ~ 1.3 both the formula and the limit are extrapolated",
        ),
    ),
    sources=(_POST_1991_SOURCE,),
    notes="'iter_1991_q95_min' on the estimate of 'iter_q95_coordinates'.",
))

_register(Boundary(
    key="iter_1991_q95_current",
    family="current_limit",
    target=_PLASMA_CURRENT_MA,
    inputs=(_MINOR_RADIUS, _MAJOR_RADIUS, _TOROIDAL_FIELD, _ELONGATION, _TRIANGULARITY),
    form="function",
    function=_iter_current,
    allowed_side="below",
    hardness="soft",
    origin="derived",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="tokamak",
        assumptions=("'iter_1991_q95_estimate_min' (q95 >= 2.1) written as a maximum current through the "
                     "guideline formula", "conventional aspect ratio; extrapolated at A ~ 1.3",
                     "the formula takes the 95 % surface kappa and delta; LCFS values raise the current limit"),
    ),
    sources=(_POST_1991_SOURCE,),
    notes="I_max [MA] = 5 a^2 B/R [1 + kappa^2(1 + 2 delta^2 - 1.2 delta^3)]/2 (1.17 - 0.65 eps)/(1 - eps^2)^2 / 2.1.",
))


# The START low-aspect-ratio correction (Akers et al., Nucl. Fusion 40 (2000) 1223, Sec. 2.1, p. 1227):
# I_N = (5/(A q95)) f(A) [1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)]/2 with, from START equilibrium data,
# f(A) = 1.17 C sqrt(A/(A - 1)); C = 1.0 for natural limiter plasmas, 0.77 for double null. Same shaping
# factor and normalisation as the ITER guideline (Akers' ref. [14] is Post et al. 1991).

_Q95_START_ESTIMATE = BoundaryQuantity(
    "edge_safety_factor_95_estimate_start", "q_{95,START}", "-",
    "q95 from the START low-aspect-ratio scaling of Akers et al. (2000) on global shape and engineering "
    "parameters; an estimate, not an equilibrium q95.",
)
_AKERS_2000_SOURCE = BoundarySource(
    "R. J. Akers et al., Nucl. Fusion 40 (2000) 1223",
    equation="Sec. 2.1, p. 1227: I_N = (5/(A q95)) f(A) [1 + kappa^2(1 + 2 delta^2 - 1.2 delta^3)]/2, "
             "f(A) = 1.17 C sqrt(A/(A - 1)), C = 1.0 (limiter) | 0.77 (double null)",
    doi="10.1088/0029-5515/40/6/317",
    note="the START scaling; the ITER f(A) of the same paper is Post et al. 1991 (its ref. [14])",
)
#: C of Akers et al. (2000): natural limiter plasma, double null.
AKERS_2000_C = {"limiter": 1.0, "double_null": 0.77}


def _start_q95_per_ma(a, R0, B0, kappa, delta, c):
    A = R0 / a
    shape = (1.0 + kappa**2 * (1.0 + 2.0 * delta**2 - 1.2 * delta**3)) / 2.0
    return 5.0 * a**2 * B0 / R0 * shape * 1.17 * c * np.sqrt(A / (A - 1.0))


def start_q95_coordinates(a, R0, B0, kappa, delta, I_p, configuration="limiter"):
    r"""The START low-aspect-ratio $q_{95}$ estimate of Akers et al. (2000) from global shape.

    $$q_{95} \approx \frac{5a^2B}{R\,I_p[\mathrm{MA}]}\,
      \frac{1+\kappa^2(1+2\delta^2-1.2\delta^3)}{2}\;1.17\,C\sqrt{\frac{A}{A-1}},
      \qquad A = R/a$$

    Parameters
    ----------
    a : float or np.ndarray
        Minor radius [m].
    R0 : float or np.ndarray
        Major radius [m].
    B0 : float or np.ndarray
        Vacuum toroidal field at ``R0``; its sign is dropped [T].
    kappa : float or np.ndarray
        Elongation of the 95 % flux surface, kappa_95 [-].
    delta : float or np.ndarray
        Triangularity of the 95 % flux surface, delta_95 [-].
    I_p : float or np.ndarray
        Plasma current; its sign is dropped [MA].
    configuration : {"limiter", "double_null"}
        Sets C = 1.0 (natural limiter plasma) or 0.77 (double null) [-].

    Returns
    -------
    float or np.ndarray
        Estimated $q_{95}$, NaN where an input is NaN [-].

    Raises
    ------
    ValueError
        A finite non-positive (or infinite) minor radius, major radius, field
        magnitude or elongation, an infinite triangularity, $a \ge R_0$, or an
        unknown configuration.

    Convention
    ----------
    A fit to START equilibria (spherical tokamak), more conservative in aspect ratio
    than the ITER guideline formula (``iter_q95_coordinates``), whose shaping factor and
    normalisation it shares. Like it, it is written for the 95 % surface shape.

    References
    ----------
    .. [1] R. J. Akers et al., Nucl. Fusion 40 (2000) 1223, Sec. 2.1, p. 1227.
    """
    if configuration not in AKERS_2000_C:
        raise ValueError(f"configuration must be one of {sorted(AKERS_2000_C)}, not {configuration!r}")
    g = _positive_geometry(a=a, R0=R0, B0=np.abs(np.asarray(B0, dtype=float)), kappa=kappa)
    if np.any(np.isfinite(g["a"] / g["R0"]) & (g["a"] / g["R0"] >= 1.0)):
        raise ValueError("a must be smaller than R0")
    d = np.asarray(delta, dtype=float)
    if np.any(np.isinf(d)):
        raise ValueError("delta must be finite where it is given")
    current = np.abs(np.asarray(I_p, dtype=float))
    with np.errstate(divide="ignore"):
        q = _start_q95_per_ma(g["a"], g["R0"], g["B0"], g["kappa"], d, AKERS_2000_C[configuration]) / current
    return _scalar_or_array(q)


def _start_current(minor_radius, major_radius, toroidal_field, elongation, triangularity, q95=2.1):
    g = _positive_geometry(a=minor_radius, R0=major_radius, kappa=elongation)
    b = _finite_field(toroidal_field)
    if np.any(np.isfinite(g["a"] / g["R0"]) & (g["a"] / g["R0"] >= 1.0)):
        raise ValueError("a must be smaller than R0")
    return _start_q95_per_ma(g["a"], g["R0"], b, g["kappa"], np.asarray(triangularity, dtype=float),
                             AKERS_2000_C["limiter"]) / q95


__all__ += ["start_q95_coordinates"]

_register(Boundary(
    key="akers_2000_q95_estimate_min",
    family="current_limit",
    target=_Q95_START_ESTIMATE,
    inputs=(),
    form="threshold",
    coefficient=2.1,
    allowed_side="above",
    hardness="soft",
    origin="derived",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="spherical tokamak (START equilibria), limiter plasma (C = 1.0)",
        ranges={},
        assumptions=(
            "the ITER extended-performance guideline q95 >= 2.1 (Post et al. 1991) applied to the START q95 "
            "estimate; Akers et al. state the scaling, not a q95 limit (their Fig. 4 uses q95 = 3)",
        ),
    ),
    sources=(_AKERS_2000_SOURCE, _POST_1991_SOURCE),
    notes="The q95 guideline on the START estimate of 'start_q95_coordinates' (limiter, C = 1.0).",
))

_register(Boundary(
    key="akers_2000_q95_current",
    family="current_limit",
    target=_PLASMA_CURRENT_MA,
    inputs=(_MINOR_RADIUS, _MAJOR_RADIUS, _TOROIDAL_FIELD, _ELONGATION, _TRIANGULARITY),
    form="function",
    function=_start_current,
    allowed_side="below",
    hardness="soft",
    origin="derived",
    basis="empirical",
    event="disruption",
    applicability=Applicability(
        machine_class="spherical tokamak (START equilibria), limiter plasma (C = 1.0)",
        assumptions=("'akers_2000_q95_estimate_min' (q95 >= 2.1) written as a maximum current through the START "
                     "scaling", "the scaling takes the 95 % surface kappa and delta; LCFS values raise the limit"),
    ),
    sources=(_AKERS_2000_SOURCE, _POST_1991_SOURCE),
    notes="I_max [MA] = 5 a^2 B/R [1 + kappa^2(1 + 2 delta^2 - 1.2 delta^3)]/2 * 1.17 sqrt(A/(A - 1)) / 2.1.",
))
