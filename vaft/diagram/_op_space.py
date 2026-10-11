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
from vaft.diagram._projection_interpretation import ProjectionInterpretation

__all__ = [
    "AXIS_QUANTITIES",
    "OperationalProjection",
    "OverlayPlan",
    "get_projection",
    "list_projections",
    "overlay_plan",
    "placement",
    "Placement",
    "IncompatibleBoundary",
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
        _registered("wesson_1989_jet_li_qpsi_lower"),
        _registered("cheng_1987_li_qa_lower"),
        _registered("cheng_1987_li_qa_lower", "cylinder_edge_safety_factor"),
        _registered("freidberg_2008_kink_qstar"),
        _registered("freidberg_2008_kink_qstar", "elongation"),
        _registered("freidberg_2008_kink_current"),
        _registered("greenwald_fraction_unity"),
        _registered("freidberg_2008_kink_current", "toroidal_field"),
        _registered("menard_2004_qstar_min"),
        _registered("iter_1991_q95_estimate_min"),
        _registered("akers_2000_q95_estimate_min"),
        _registered("greenwald_hugill_st", "inverse_cylindrical_q_st"),
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

@dataclass(frozen=True)
class OperationalProjection:
    """A literature-defined 2-D operational space.

    ``x`` and ``y`` are exact quantity identities. ``default_boundaries`` are
    the registered keys drawn by default; each must be compatible with the
    axes (see :func:`placement`). ``ratio`` is the quantity ``y/x`` when the
    reference space reads it as a family of lines through the origin -- Troyon's
    $\\beta_N = \\beta_T/I_N$ -- so a limit on it is a line through the origin.
    ``diagram`` is the :mod:`vaft.diagram` builder for the reference picture, if any.
    ``interpretation`` states what the space is for and the conventions of its
    axes (#1624); see :mod:`vaft.diagram._projection_interpretation`.
    """

    key: str
    title: str
    x: _b.BoundaryQuantity
    y: _b.BoundaryQuantity
    default_boundaries: Tuple[str, ...] = ()
    references: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    diagram: Optional[str] = None
    ratio: Optional[_b.BoundaryQuantity] = None
    interpretation: Optional[ProjectionInterpretation] = None


@dataclass(frozen=True)
class OverlayPlan:
    """The boundaries a projection can draw, and those it refused with the reason."""

    projection: str
    curves: Tuple[_b.BoundaryCurve, ...]
    omitted: Tuple[Tuple[str, str], ...]

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(c.key for c in self.curves)


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
    title="Toroidal beta against normalised current (Troyon plane)",
    x=_q("normalized_current"),
    y=_q("toroidal_beta"),
    default_boundaries=("troyon", "strait_1988_diiid_beta_n_envelope", "taylor_1995_diiid_beta_n_record",
                        "garstka_2002_st_beta_n_reference", "sabbagh_2006_nstx_beta_n_record"),
    references=("F. Troyon et al., Plasma Phys. Control. Fusion 26 (1984) 209, Fig. 10",
                "experimental beta_N references (#1691): Strait 1988, Taylor 1995, Garstka 2002, Sabbagh 2006"),
    assumptions=(
        "beta_N = beta_T / (I_p/(a B_T)) is the ratio of the axes, so its threshold is a line through the origin",
        "Troyon's beta uses the total field; at low beta it is close to the toroidal beta on this axis",
    ),
    diagram="troyon",
    ratio=_q("normalized_beta"),
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
    default_boundaries=("iter_1991_q95_min",),
    references=("D. E. Post et al., ITER Physics, ITER Documentation Series No. 21 (1991), Table 1-2",),
    assumptions=(
        "the ITER design guideline q95 >= 2.1 (extended performance) is the only registered limit on q95; "
        "low_q (q_psi) and the li-q_a literature boundaries use other q and li definitions and are not drawn here",
    ),
))

_register(OperationalProjection(
    key="greenwald_fraction_power",
    title="Greenwald fraction against loss power",
    x=_q("loss_power"),
    y=_q("greenwald_fraction"),
    default_boundaries=("greenwald_fraction_unity",),
    references=("M. Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27",),
    assumptions=("f_G uses the line-averaged density and n_G = I_p/(pi a^2)",),
))


_register(OperationalProjection(
    key="li_qa_wesson",
    title="JET empirical l_i-q_psi operating space",
    x=_q("edge_safety_factor"),
    y=_q("internal_inductance_li3"),
    default_boundaries=("wesson_1989_jet_li_qpsi_lower", "wesson_1989_jet_li_qpsi_upper", "low_q"),
    references=("J. A. Wesson et al., Nucl. Fusion 29 (1989) 641, Fig. 6",),
    assumptions=(
        "x is q_psi at the plasma edge; a q95 column is not accepted in its place",
        "low_q (q_psi > 2, Greenwald 1988) is the same quantity as this x axis and closes the q = 2 edge",
        "y is the l_i(3) form 2 int B_p^2 dV/(mu0^2 I_p^2 R); Wesson's R is the plasma major radius, the DD's R_0",
        "empirical JET boundaries; their transfer to spherical tokamaks is untested",
    ),
    diagram="li_qa",
))

_register(OperationalProjection(
    key="li_qa_cheng",
    title="Cheng-Furth-Boozer MHD-stable l_i-q(a) domain",
    x=_q("cylinder_edge_safety_factor"),
    y=_q("internal_inductance_cylinder"),
    default_boundaries=("cheng_1987_li_qa_lower", "cheng_1987_li_qa_upper", "cheng_1987_qa_min"),
    references=("C. Z. Cheng, H. P. Furth and A. H. Boozer, Plasma Phys. Control. Fusion 29 (1987) 351, Fig. 4",),
    assumptions=(
        "pressureless straight cylinder, no wall, q(0) = 1.01: a theoretical domain, not a fit to data",
        "axes are the cylinder's q(a) and l_i; toroidal q_psi, q95 and l_i(3) are other quantities",
    ),
    diagram="li_qa",
))


_register(OperationalProjection(
    key="qstar_in",
    title="Kink safety factor against normalized current",
    x=_q("normalized_current"),
    y=_q("kink_safety_factor_elliptic"),
    default_boundaries=("freidberg_2008_kink_qstar",),
    references=("J. P. Freidberg, Plasma Physics and Fusion Energy (2008), Eqs. (13.160) and (13.162)",),
    assumptions=(
        "q* is Freidberg's Eq. (13.160), 2 pi a^2 kappa B0/(mu0 R0 I); the limit (1 + kappa)/2 is drawn at one "
        "elongation, so pass the kappa it should represent (the largest kappa is the most restrictive line)",
    ),
))

_register(OperationalProjection(
    key="hugill_st",
    title="Spherical-tokamak Hugill diagram",
    x=_q("murakami_parameter"),
    y=_q("inverse_cylindrical_q_st"),
    default_boundaries=("sykes_2000_st_hugill", "greenwald_hugill_st", "murakami_hugill"),
    references=("A. Sykes et al., First results from MAST, IAEA FEC 2000, Fig. 12; Nucl. Fusion 41 (2001) 1423",),
    assumptions=(
        "y is the spherical-tokamak 1/q_cyl = R I_p/(2.5 a^2 (1 + kappa^2) B_T), not the conventional Hugill 1/q_cyl",
        "the Hugill and Greenwald lines depend on kappa: pass the elongation they should represent",
        "Murakami is a historical conventional-tokamak reference here, not a spherical-tokamak limit",
    ),
))

_register(OperationalProjection(
    key="ip_bt",
    title="Plasma current against toroidal field",
    x=_q("toroidal_field"),
    y=_q("plasma_current"),
    default_boundaries=("freidberg_2008_kink_current", "menard_2004_qstar_current", "iter_1991_q95_current",
                        "akers_2000_q95_current"),
    references=("PPCF 67 (2025) 115021, Fig. 6(b) (engineering operational space)",
                "J. P. Freidberg, Plasma Physics and Fusion Energy (2007), Eq. (13.163)",
                "J. E. Menard et al., Phys. Plasmas 11 (2004) 639",
                "D. E. Post et al., ITER Physics, ITER Documentation Series No. 21 (1991), Table 1-2",
                "R. J. Akers et al., Nucl. Fusion 40 (2000) 1223, Sec. 2.1"),
    assumptions=(
        "every limit is a current at fixed shape: pass the minor radius, major radius, elongation and "
        "triangularity it should represent (a mean shape for a shot database)",
    ),
))

_register(OperationalProjection(
    key="qstar_cyl_in",
    title="Cylindrical safety factor against normalized current",
    x=_q("normalized_current"),
    y=_q("kink_safety_factor_cylindrical"),
    default_boundaries=("menard_2004_qstar_min",),
    references=("J. E. Menard et al., Phys. Plasmas 11 (2004) 639",),
    assumptions=("q* = pi a^2 B_T0 (1 + kappa^2)/(mu0 R0 I_P), Menard's definition; not Freidberg's Eq. (13.160)",),
))

_register(OperationalProjection(
    key="q95_estimate_in",
    title="ITER-guideline q95 estimate against normalized current",
    x=_q("normalized_current"),
    y=_q("edge_safety_factor_95_estimate_iter"),
    default_boundaries=("iter_1991_q95_estimate_min",),
    references=("D. E. Post et al., ITER Physics, ITER Documentation Series No. 21 (1991), Table 1-2",),
    assumptions=("q95 from the guideline formula on global shape; conventional-aspect-ratio fit, extrapolated "
                 "at A ~ 1.3",),
))

_register(OperationalProjection(
    key="q95_start_estimate_in",
    title="START q95 estimate against normalized current",
    x=_q("normalized_current"),
    y=_q("edge_safety_factor_95_estimate_start"),
    default_boundaries=("akers_2000_q95_estimate_min",),
    references=("R. J. Akers et al., Nucl. Fusion 40 (2000) 1223, Sec. 2.1",),
    assumptions=("q95 from the START low-aspect-ratio scaling on global shape (limiter, C = 1.0)",),
))

_register(OperationalProjection(
    key="ip_kappa",
    title="Plasma current against elongation",
    x=_q("elongation"),
    y=_q("plasma_current"),
    default_boundaries=("freidberg_2008_kink_current",),
    references=("J. P. Freidberg, Plasma Physics and Fusion Energy (2008), Eq. (13.163)",),
    assumptions=("the current limit is drawn for one a, R0 and B0 (the population median unless given)",),
))

_register(OperationalProjection(
    key="lh_threshold",
    title="Loss power against line-averaged density (L-H access)",
    x=_registered("martin_2008_lh", "line_average_density"),
    y=_q("loss_power"),
    default_boundaries=("martin_2008_lh", "takizuka_2004_lh"),
    references=("Y. R. Martin et al., J. Phys.: Conf. Ser. 123 (2008) 012033, Eq. 2",
                "T. Takizuka et al., Plasma Phys. Control. Fusion 46 (2004) A227, Eq. 4"),
    assumptions=(
        "density in 1e20 m^-3, the unit of the Martin and Takizuka fits; the Ryter low-density minimum is stated "
        "in 1e19 m^-3 and is therefore not drawn on this axis",
        "the thresholds are drawn for one B_T, S, I_p, a, A and Z_eff (population median unless given); VEST lies "
        "outside the fitted range of both",
    ),
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

class IncompatibleBoundary(ValueError):
    """A registered boundary whose quantities are not a projection's axes (raised only by ``strict`` calls)."""


@dataclass(frozen=True)
class Placement:
    """How a registered boundary sits on a projection, decided without sampling it.

    ``kind`` is one of:
    - ``"curve"``: swept along the other axis (``sweep``), the target on y or, with ``swap_axes``, on x;
    - ``"horizontal"`` / ``"vertical"``: constant on y / x, either a threshold or a boundary whose
      inputs are all off-axis and given in ``fixed``;
    - ``"ratio"``: a limit on the projection's ``ratio`` quantity, a line through the origin;
    - ``"incompatible"``: not drawable, with ``reason``.
    ``needs`` lists the inputs that must be fixed to draw it.
    """

    key: str
    kind: str
    sweep: Optional[str] = None
    swap_axes: bool = False
    needs: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def drawable(self) -> bool:
        return self.kind != "incompatible"


def placement(projection: Union[str, OperationalProjection], boundary: Union[str, _b.Boundary],
              fixed: Optional[Mapping[str, float]] = None, *, strict: bool = False) -> Placement:
    """Classify a registered boundary on a projection by exact quantity identity (name and unit).

    Parameters
    ----------
    projection : str or OperationalProjection
        The projection, or its registry key.
    boundary : str or Boundary
        The boundary, or its registry key.
    fixed : mapping, optional
        Inputs supplied with fixed values. Without them a boundary that needs
        them is ``incompatible`` with the reason saying which.
    strict : bool
        Raise :class:`IncompatibleBoundary` instead of returning an incompatible placement.

    Returns
    -------
    Placement
        Pure metadata; :func:`overlay_plan` samples it.
    """
    proj = get_projection(projection) if isinstance(projection, str) else projection
    b = _b.get_boundary(boundary) if isinstance(boundary, str) else boundary
    fixed = dict(fixed or {})

    def refuse(reason):
        if strict:
            raise IncompatibleBoundary(f"boundary {b.key!r} on projection {proj.key!r}: {reason}")
        return Placement(b.key, "incompatible", reason=reason)

    if not isinstance(b, _b.Boundary):
        return refuse("windows are not drawn as single curves")
    # one name in two units (line_average_density is registered in 1e19 and 1e20 m^-3) is never mixed
    plotted = {q.name: q.unit for q in (proj.x, proj.y, proj.ratio) if q is not None}
    for q in (b.target, *b.inputs):
        if q.name in plotted and q.unit != plotted[q.name]:
            return refuse(f"gives {q.name} in {q.unit}; the projection plots it in {plotted[q.name]}")
    off_axis = tuple(n for n in b.input_names)
    if b.form == "threshold" or all(n in fixed for n in off_axis) and not any(
            _b.same_quantity(q, a) for q in b.inputs for a in (proj.x, proj.y)):
        needs = off_axis
        if _b.same_quantity(b.target, proj.x):
            return Placement(b.key, "vertical", needs=needs)
        if _b.same_quantity(b.target, proj.y):
            return Placement(b.key, "horizontal", needs=needs)
        if proj.ratio is not None and _b.same_quantity(b.target, proj.ratio):
            return Placement(b.key, "ratio", needs=needs)
    for target_axis, sweep_axis, swap in ((proj.y, proj.x, False), (proj.x, proj.y, True)):
        if not _b.same_quantity(b.target, target_axis):
            continue
        sweep = next((q.name for q in b.inputs if _b.same_quantity(q, sweep_axis)), None)
        if sweep is None:
            missing = [n for n in off_axis if n not in fixed]
            return refuse(f"no input of {b.key} is the {sweep_axis.name} axis, and {missing} are not fixed")
        needs = tuple(n for n in b.input_names if n != sweep)
        missing = [n for n in needs if n not in fixed]
        if missing:
            return refuse(f"needs fixed input(s) {missing}")
        return Placement(b.key, "curve", sweep=sweep, swap_axes=swap, needs=needs)
    if proj.ratio is not None and _b.same_quantity(b.target, proj.ratio):
        missing = [n for n in off_axis if n not in fixed]
        if missing:
            return refuse(f"needs fixed input(s) {missing}")
        return refuse("a limit on the ratio must not depend on either axis, and this one does")
    axes = ", ".join(q.name for q in (proj.x, proj.y))
    return refuse(f"targets {b.target.name} [{b.target.unit}], which is neither axis ({axes})")


def _curve(projection: OperationalProjection, boundary: _b.Boundary, xs: np.ndarray, ys: np.ndarray,
           fixed: Mapping[str, float]) -> Union[_b.BoundaryCurve, str]:
    """The boundary on this projection, sampled, or the reason it cannot be drawn."""
    place = placement(projection, boundary, fixed)
    if not place.drawable:
        return place.reason
    used = {n: float(fixed[n]) for n in place.needs}
    if place.kind == "curve":
        return _b.boundary_curve(boundary, place.sweep, ys if place.swap_axes else xs,
                                 swap_axes=place.swap_axes, **used)
    level = float(_b.boundary_value(boundary, **used))
    px, py = projection.x, projection.y
    if place.kind == "ratio":   # y / x = level
        return _b.BoundaryCurve(key=boundary.key, x=xs, y=level * xs, x_quantity=px, y_quantity=py,
                                allowed_side=boundary.allowed_side, fixed={**used, boundary.target.name: level})
    if place.kind == "horizontal":
        return _b.BoundaryCurve(key=boundary.key, x=xs, y=np.full(xs.shape, level), x_quantity=px,
                                y_quantity=boundary.target, allowed_side=boundary.allowed_side, fixed=used)
    side = {"below": "left", "above": "right"}[boundary.allowed_side]
    return _b.BoundaryCurve(key=boundary.key, x=np.full(ys.shape, level), y=ys, x_quantity=boundary.target,
                            y_quantity=py, allowed_side=side, fixed=used)


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


# dimensionless-similarity projections (#1624) register themselves through _register
from vaft.diagram import _similarity_space  # noqa: E402,F401
