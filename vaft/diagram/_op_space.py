"""Canonical operational-space projections and their compatible boundaries (#944, #1425).

A projection is a pair of axes with a fixed physical meaning -- the Hugill
plane is $(\\bar n_e R/B_T,\\ 1/q_\\mathrm{cyl})$, not "density against q" --
plus the registered boundaries of :mod:`vaft.formula.boundaries` that belong
on it. Both the reference diagram (:mod:`vaft.diagram`) and the data plot
(:mod:`vaft.plot.operational_space`) read the same projection, so a diagram
and a population plot of the same space draw the same lines.

A boundary is drawn only when its quantities are the axes' quantities
exactly (``boundaries.same_quantity``: identity and unit). ``low_q`` targets
the equilibrium edge $q_\\psi$, so it is omitted from the Hugill plane, whose
y axis is $1/q_\\mathrm{cyl}$, and from any $q_{95}$ axis. The one exception is
an explicit transform the projection declares and documents, such as the
Troyon line on the $(I_p/aB_T,\\ \\beta_T)$ plane, which is the definition of
$\\beta_N$ rather than an approximation.

Nothing here reads ODS, database or shot data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from vaft.formula import boundaries as _b

__all__ = [
    "AXIS_QUANTITIES",
    "OperationalProjection",
    "OverlayPlan",
    "get_projection",
    "list_projections",
    "overlay_plan",
    "requested_boundaries",
]


# ---------------------------------------------------------------------------
# Axis quantities that no registered boundary targets yet
# ---------------------------------------------------------------------------

def _registered(key: str, name: Optional[str] = None) -> _b.BoundaryQuantity:
    """A quantity as a registered boundary declares it (target, or the named input)."""
    entry = _b.get_boundary(key)
    return entry.target if name is None else entry.input(name)


#: Axis quantities by name. Those a boundary already declares are taken from
#: the registry, so their identity cannot drift from the boundary's.
AXIS_QUANTITIES: Dict[str, _b.BoundaryQuantity] = {
    q.name: q for q in (
        _registered("greenwald_hugill"),
        _registered("greenwald_hugill", "inverse_cylindrical_q"),
        _registered("greenwald_hugill", "area_elongation"),
        _registered("troyon"),
        _registered("low_q"),
        _registered("giacomin_edge", "edge_safety_factor_95"),
        _registered("martin_2008_lh"),
        _b.BoundaryQuantity(
            "internal_inductance_li3", "l_i(3)", "-",
            "IMAS DD global_quantities.li_3: 2 int B_p^2 dV / (mu0^2 I_p^2 R_0), R_0 the reference major radius.",
        ),
        _b.BoundaryQuantity(
            "greenwald_fraction", "f_G", "-",
            "Line-averaged electron density over the Greenwald density I_p/(pi a^2).",
        ),
        _b.BoundaryQuantity(
            "normalized_current", "I_p/(a B_T)", "MA m^-1 T^-1",
            "Plasma current over minor radius times vacuum toroidal field (Troyon's I_N up to mu0).",
        ),
        _b.BoundaryQuantity(
            "toroidal_beta", "beta_T", "%",
            "Toroidal beta 2 mu0 <p> / B_T^2, in percent.",
        ),
    )
}


def _q(name: str) -> _b.BoundaryQuantity:
    return AXIS_QUANTITIES[name]


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

#: An explicit transform: (boundary, x samples, y samples) -> BoundaryCurve.
Transform = Callable[[_b.Boundary, np.ndarray, np.ndarray], _b.BoundaryCurve]


@dataclass(frozen=True)
class OperationalProjection:
    """A literature-defined 2-D operational space.

    ``x`` and ``y`` are exact quantity identities. ``default_boundaries`` are
    the registered keys drawn by default; each must be compatible with the
    axes, directly or through a declared ``transforms`` entry. ``diagram`` is
    the :mod:`vaft.diagram` builder for the reference picture, if any.
    """

    key: str
    title: str
    x: _b.BoundaryQuantity
    y: _b.BoundaryQuantity
    default_boundaries: Tuple[str, ...] = ()
    references: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    diagram: Optional[str] = None
    transforms: Mapping[str, Transform] = field(default_factory=dict)


@dataclass(frozen=True)
class OverlayPlan:
    """The boundaries a projection can draw, and those it refused with the reason."""

    projection: str
    curves: Tuple[_b.BoundaryCurve, ...]
    omitted: Tuple[Tuple[str, str], ...]

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(c.key for c in self.curves)


def _troyon_on_current_plane(boundary: _b.Boundary, x: np.ndarray, y: np.ndarray) -> _b.BoundaryCurve:
    """beta_N <= C on the (I_p/(a B_T), beta_T) plane: beta_T = C * I_p/(a B_T).

    This is the definition of beta_N (stability.beta_N_from_beta_a_B0_Ip:
    beta_N = beta[%] a B / I[MA]); the coefficient is the registered one.
    """
    level = float(_b.boundary_value(boundary))
    return _b.BoundaryCurve(key=boundary.key, x=x, y=level * x, x_quantity=_q("normalized_current"),
                            y_quantity=_q("toroidal_beta"), allowed_side=boundary.allowed_side,
                            fixed={"normalized_beta": level})


_PROJECTIONS: Dict[str, OperationalProjection] = {}


def _register(projection: OperationalProjection) -> None:
    if projection.key in _PROJECTIONS:
        raise ValueError(f"projection {projection.key!r} is already registered")
    for key in projection.default_boundaries:
        _b.get_boundary(key)  # unknown keys fail at import, not at plot time
    _PROJECTIONS[projection.key] = projection


_register(OperationalProjection(
    key="hugill",
    title="Hugill diagram",
    x=_q("murakami_parameter"),
    y=_q("inverse_cylindrical_q"),
    default_boundaries=("greenwald_hugill", "murakami_hugill"),
    references=(
        "J. Hugill, as reproduced in M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27, Fig. 1",
        "M. Greenwald et al., Nucl. Fusion 28 (1988) 2199",
    ),
    assumptions=(
        "y is the cylindrical q of equilibrium.q_cyl_from_B_R_epsilon_kappa_I, not q95 and not q_psi",
        "R and B_T in the Murakami parameter are those used for q_cyl",
        "low_q (q_psi > 2) is not a default: q_psi is not the y quantity",
    ),
    diagram="hugill",
))

_register(OperationalProjection(
    key="troyon",
    title="Troyon beta limit",
    x=_q("normalized_current"),
    y=_q("toroidal_beta"),
    default_boundaries=("troyon",),
    references=("F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209, Fig. 10",),
    assumptions=(
        "the beta_N threshold is drawn through the definition beta_T[%] = beta_N I_p/(a B_T)",
        "Troyon's beta uses the total field; at low beta it is close to the toroidal beta on this axis",
    ),
    diagram="troyon",
    transforms={"troyon": _troyon_on_current_plane},
))

_register(OperationalProjection(
    key="beta_n_li",
    title="Normalized beta against internal inductance",
    x=_q("internal_inductance_li3"),
    y=_q("normalized_beta"),
    default_boundaries=("troyon",),
    references=("F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209",),
    assumptions=("li is the DD li_3; the Troyon line does not depend on li in this entry",),
))

_register(OperationalProjection(
    key="q95_li",
    title="Internal inductance against q95",
    x=_q("edge_safety_factor_95"),
    y=_q("internal_inductance_li3"),
    default_boundaries=(),
    assumptions=(
        "no registered boundary is defined on q95 and li_3; low_q (q_psi) and the li-q_a "
        "literature boundaries use other q and li definitions and are not drawn here",
    ),
))

_register(OperationalProjection(
    key="greenwald_fraction_power",
    title="Greenwald fraction against loss power",
    x=_q("loss_power"),
    y=_q("greenwald_fraction"),
    default_boundaries=(),
    assumptions=("f_G = 1 is a reference level, not a registered boundary on this plane",),
))


def get_projection(key: str) -> OperationalProjection:
    """A registered operational-space projection by key.

    Raises
    ------
    KeyError
        No projection has that key; the message lists the keys that exist.
    """
    try:
        return _PROJECTIONS[key]
    except KeyError:
        raise KeyError(f"no projection {key!r}; registered: {sorted(_PROJECTIONS)}") from None


def list_projections() -> Tuple[str, ...]:
    """Keys of the registered projections, sorted."""
    return tuple(sorted(_PROJECTIONS))


# ---------------------------------------------------------------------------
# Overlay selection
# ---------------------------------------------------------------------------

def _curve(projection: OperationalProjection, boundary: _b.Boundary, xs: np.ndarray, ys: np.ndarray,
           fixed: Mapping[str, float]) -> Union[_b.BoundaryCurve, str]:
    """The boundary on this projection, or the reason it cannot be drawn."""
    px, py = projection.x, projection.y
    if boundary.key in projection.transforms:
        return projection.transforms[boundary.key](boundary, xs, ys)
    if boundary.form == "threshold":
        if _b.same_quantity(boundary.target, px):
            return _b.threshold_curve(boundary, py, ys, target_axis="x")
        if _b.same_quantity(boundary.target, py):
            return _b.threshold_curve(boundary, px, xs, target_axis="y")
        return (f"targets {boundary.target.name} [{boundary.target.unit}], "
                f"which is neither axis ({px.name}, {py.name})")
    for target_on, sweep_axis, values, swap in (("y", px, xs, False), ("x", py, ys, True)):
        target_axis = py if target_on == "y" else px
        if not _b.same_quantity(boundary.target, target_axis):
            continue
        sweep = next((q.name for q in boundary.inputs if _b.same_quantity(q, sweep_axis)), None)
        if sweep is None:
            return f"no input of {boundary.key} is the {sweep_axis.name} axis"
        needed = [n for n in boundary.input_names if n != sweep]
        missing = [n for n in needed if n not in fixed]
        if missing:
            return f"needs fixed input(s) {missing}"
        return _b.boundary_curve(boundary, sweep, values, swap_axes=swap,
                                 **{n: float(fixed[n]) for n in needed})
    return (f"targets {boundary.target.name} [{boundary.target.unit}], "
            f"which is neither axis ({px.name}, {py.name})")


def requested_boundaries(projection: OperationalProjection,
                         boundaries: Union[str, bool, None, Sequence[str]] = "default") -> Tuple[str, ...]:
    """Normalise a ``boundaries=`` argument to registry keys, in order and without repeats.

    ``"default"`` or ``True`` give the projection's defaults; ``False`` or ``None`` give none;
    a sequence of keys is taken as given. Any other string raises ``ValueError``.
    """
    if boundaries is True:
        return tuple(projection.default_boundaries)
    if boundaries is False or boundaries is None:
        return ()
    if isinstance(boundaries, str):
        if boundaries != "default":
            raise ValueError(f"boundaries must be 'default', True, False or a list of keys, not {boundaries!r}")
        return tuple(projection.default_boundaries)
    return tuple(dict.fromkeys(boundaries))


def overlay_plan(projection: Union[str, OperationalProjection], boundaries: Union[str, bool, Sequence[str]] = "default",
                 *, x_range: Tuple[float, float], y_range: Tuple[float, float],
                 fixed: Optional[Mapping[str, float]] = None, samples: int = 201) -> OverlayPlan:
    """Which boundaries to draw on a projection, sampled over the given ranges.

    Parameters
    ----------
    projection : str or OperationalProjection
        The projection, or its registry key.
    boundaries : "default", False or sequence of str
        ``"default"`` uses the projection's ``default_boundaries``; ``False``
        draws none; a list names registered boundaries explicitly.
    x_range, y_range : (float, float)
        Sampling ranges, in the axis quantities' units.
    fixed : mapping, optional
        Values of the boundary inputs that are not on an axis, for example
        ``{"area_elongation": 1.6}``, in the inputs' declared units.
    samples : int
        Points per curve.

    Returns
    -------
    OverlayPlan
        The curves, and ``(key, reason)`` for every requested boundary that is
        not compatible with the axes. Nothing is drawn on a near match.
    """
    proj = get_projection(projection) if isinstance(projection, str) else projection
    keys = requested_boundaries(proj, boundaries)
    xs = np.linspace(float(x_range[0]), float(x_range[1]), samples)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), samples)
    curves, omitted = [], []
    for key in keys:
        entry = _b.get_boundary(key)
        if not isinstance(entry, _b.Boundary):
            omitted.append((key, "windows are not drawn as single curves"))
            continue
        result = _curve(proj, entry, xs, ys, dict(fixed or {}))
        if isinstance(result, str):
            omitted.append((key, result))
        else:
            curves.append(result)
    return OverlayPlan(projection=proj.key, curves=tuple(curves), omitted=tuple(omitted))
