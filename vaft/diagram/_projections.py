"""Canonical operational-space projections: what each axis of a reference diagram *is* (#1425).

A projection names the exact physical quantity on each axis -- by the same
:class:`~vaft.formula.boundaries.BoundaryQuantity` identity the registered
boundaries use -- its literature references, and the registered boundaries
that are compatible with it by default. Compatibility is decided by quantity
identity (name and unit), never by shape or a similar name: a boundary on ``q_psi`` is not
drawn on a ``q_cyl`` axis, and ``q95`` is neither.

``vaft.diagram`` builds its reference diagrams from these projections and
:mod:`vaft.formula.boundaries`; ``vaft.plot`` will place measured or modelled
states on the same axes and overlay the same boundaries (#1425 phase 2), so
the boundary physics has one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from vaft.formula.boundaries import Boundary, BoundaryQuantity, get_boundary


class IncompatibleBoundary(ValueError):
    """A registered boundary whose quantities are not this projection's axes."""


@dataclass(frozen=True)
class OperationalProjection:
    """One literature reference space: its axes, references and default boundaries.

    ``ratio`` is the quantity ``y/x`` when the reference space plots it as a
    family of lines through the origin (Troyon's $\\beta_N = \\beta_T/I_N$);
    a threshold boundary on the ratio is then a line through the origin.
    """

    key: str
    x: BoundaryQuantity
    y: BoundaryQuantity
    references: Tuple[str, ...]
    default_boundaries: Tuple[str, ...]
    ratio: Optional[BoundaryQuantity] = None
    assumptions: str = ""


def _target_and_inputs(boundary: Boundary) -> Tuple[str, Tuple[str, ...]]:
    return boundary.target.name, boundary.input_names


def placement(projection: OperationalProjection, boundary: Boundary, fixed: Mapping[str, float] = ()) -> dict:
    """How a registered boundary sits on a projection, or :class:`IncompatibleBoundary`.

    Returns ``{"kind": ..., "sweep": ...}``: ``"curve"`` with the input swept
    along the other axis (every remaining input must be in ``fixed``),
    ``"vertical"`` / ``"horizontal"`` for a threshold on the x / y quantity,
    or ``"ratio"`` for a threshold on the projection's ratio quantity.
    """
    target, inputs = _target_and_inputs(boundary)
    axes = {projection.x.name: "x", projection.y.name: "y"}
    fixed = dict(fixed)
    # a name can carry two units in the registry (line_average_density is 1e19 and 1e20 m^-3)
    units = {projection.x.name: projection.x.unit, projection.y.name: projection.y.unit}
    if projection.ratio is not None:
        units[projection.ratio.name] = projection.ratio.unit
    for quantity in (boundary.target, *boundary.inputs):
        if quantity.name in units and quantity.unit != units[quantity.name]:
            raise IncompatibleBoundary(
                f"boundary {boundary.key!r} gives {quantity.name!r} in {quantity.unit!r}; "
                f"projection {projection.key!r} plots it in {units[quantity.name]!r}")
    if boundary.form == "threshold":
        if target in axes:
            return {"kind": "vertical" if axes[target] == "x" else "horizontal", "sweep": None}
        if projection.ratio is not None and target == projection.ratio.name:
            return {"kind": "ratio", "sweep": None}
        raise IncompatibleBoundary(
            f"boundary {boundary.key!r} bounds {target!r}, which is not an axis of projection {projection.key!r} "
            f"({projection.x.name!r}, {projection.y.name!r})")
    if target not in axes:
        raise IncompatibleBoundary(f"boundary {boundary.key!r} bounds {target!r}, not an axis of {projection.key!r}")
    other = projection.y.name if axes[target] == "x" else projection.x.name
    if other not in inputs:
        raise IncompatibleBoundary(
            f"boundary {boundary.key!r} on {target!r} does not depend on the other axis {other!r} of {projection.key!r}")
    missing = [name for name in inputs if name != other and name not in fixed]
    if missing:
        raise IncompatibleBoundary(f"boundary {boundary.key!r} needs fixed values for {missing} on {projection.key!r}")
    return {"kind": "curve", "sweep": other, "swap_axes": axes[target] == "x"}


def compatible_boundaries(projection: OperationalProjection, keys=None) -> Tuple[str, ...]:
    """The registered keys (default: the projection's defaults) that sit on this projection.

    A key whose boundary needs fixed inputs counts as compatible here; the
    caller supplies them when it draws the curve.
    """
    chosen = []
    for key in (projection.default_boundaries if keys is None else keys):
        boundary = get_boundary(key)
        try:
            placement(projection, boundary, {name: 0.0 for name in boundary.input_names})
        except IncompatibleBoundary:
            continue
        chosen.append(key)
    return tuple(chosen)


# --- quantities the reference axes use -------------------------------------------------------------

_MURAKAMI = get_boundary("murakami_hugill").target
_INVERSE_Q_CYL = get_boundary("greenwald_hugill").input("inverse_cylindrical_q")
_NORMALIZED_BETA = get_boundary("troyon").target
_NORMALIZED_CURRENT = BoundaryQuantity(
    "normalized_current", "I_p/(a B_T)", "MA m^-1 T^-1",
    "Plasma current magnitude over minor radius times the vacuum toroidal field at R_0.")
_TOROIDAL_BETA = BoundaryQuantity(
    "toroidal_beta", "beta_T", "%", "Toroidal beta 2 mu0 <p> / B_T^2, in percent.")

PROJECTIONS: Mapping[str, OperationalProjection] = {
    "hugill": OperationalProjection(
        key="hugill",
        x=_MURAKAMI,
        y=_INVERSE_Q_CYL,
        references=(
            "M. Greenwald et al., Nucl. Fusion 28 (1988) 2199",
            "M. Murakami, J. D. Callen and L. A. Berry, Nucl. Fusion 16 (1976) 347",
        ),
        default_boundaries=("greenwald_hugill",),
        assumptions="murakami_hugill (the vertical line nR/B_T = 1) also sits on these axes and is drawn on request; y is the cylindrical q of equilibrium.q_cyl_from_B_R_epsilon_kappa_I; "
                    "q95 and the equilibrium edge q_psi are different quantities and are not substituted.",
    ),
    "troyon": OperationalProjection(
        key="troyon",
        x=_NORMALIZED_CURRENT,
        y=_TOROIDAL_BETA,
        references=("F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209",),
        default_boundaries=("troyon",),
        ratio=_NORMALIZED_BETA,
        assumptions="Troyon's beta uses the total field; the toroidal beta on y is close to it at low beta.",
    ),
}


def get_projection(key: str) -> OperationalProjection:
    """The canonical projection ``key``; ``KeyError`` names the known ones."""
    try:
        return PROJECTIONS[key]
    except KeyError:
        raise KeyError(f"no operational projection {key!r}; known: {sorted(PROJECTIONS)}") from None
