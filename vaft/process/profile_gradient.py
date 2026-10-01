"""Profile gradients with an explicit coordinate, gradient coordinate and reference length.

A normalized gradient such as ``a/L_T`` is a derived quantity, not a radial
coordinate, and it is defined only once three separate choices are stated
(issue #551):

``coordinate``
    The abscissa: where on the profile each value is placed.
``gradient_coordinate``
    The physical radial variable the derivative is taken with respect to.
``reference_length``
    The macroscopic length that multiplies the derivative.

A ``convention`` is a code-specific preset that resolves the last two, and
nothing more: it is recorded beside the explicit values it resolved to, never
instead of them.

The chain, and where each step's state changes::

    one equilibrium slice
    -> every registered radial coordinate on its psi grid      [RadialCoordinateMap]
    -> profile on its own grid, differentiated there
    -> chain rule d/dx_g = (dx_p/dx_g) d/dx_p through the map
    -> times the reference length                              [ReferenceLength]
    -> placed on the requested abscissa                         [ProfileGradient]

Nothing is extrapolated.  A point outside the radial support of the map, a
mapping that is not monotonic, a profile value that is not positive, a
coordinate the equilibrium cannot supply, and a convention whose reference
length depends on configuration the caller did not pass are all refused with
:class:`ValueError`.

Notation
--------
psi_norm      : normalized poloidal flux, (psi - psi_axis)/(psi_boundary - psi_axis)  [-]
rho_pol_norm  : sqrt(psi_norm)                                                         [-]
rho_tor_norm  : sqrt(Phi/Phi_boundary), Phi = int q dpsi                               [-]
R_in, R_out   : major radius of the inboard/outboard crossing at the axis height        [m]
r_minor       : (R_out - R_in)/2                                                         [m]
a_minor       : r_minor on the last closed flux surface                                 [m]
r_minor_norm  : r_minor/a_minor                                                          [-]
L             : reference length                                                         [m]
x_p, x_g, x_c : profile, gradient and plot coordinate                                   [-]

Conventions
-----------
``r_minor`` is the half-width of each surface **at the magnetic-axis height**,
from the two midplane crossings -- the IMAS ``r_inboard``/``r_outboard`` leaves
and the GACODE ``rmin`` VAFT writes.  It is not the contour half-width
``(max R - min R)/2`` that
:func:`vaft.process.equilibrium.derive_global_descriptors` reports as
``minor_radius``, and it is never ``R_out - R_axis``, ``rho_tor_norm``,
``rho_pol_norm`` or ``sqrt(psi_norm)``.  The existing coordinate tuples in
:mod:`vaft.plot` and :mod:`vaft.process.profile` are unchanged; only this layer
reads :data:`RADIAL_COORDINATES`.

Provenance
----------
.. [DD] IMAS data dictionary, ``equilibrium.time_slice.profiles_1d``:
   ``r_inboard``/``r_outboard`` are the major radii of the inboard and outboard
   crossings of each flux surface at the height of the magnetic axis.
.. [GACODE] GACODE ``input.gacode`` and the TGLF/CGYRO input lists
   (https://gacode.io/input_gacode.html, https://gacode.io/tglf/tglf_list.html,
   https://gacode.io/cgyro/cgyro_list.html): ``r`` is the midplane minor radius
   ``rmin``, ``a`` its value on the last surface, and ``RLTS``/``DLNTDR`` are
   ``-a d ln T/dr``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "CONVENTIONS",
    "RADIAL_COORDINATES",
    "REFERENCE_LENGTHS",
    "ConventionPreset",
    "ProfileGradient",
    "RadialCoordinate",
    "RadialCoordinateMap",
    "ReferenceLength",
    "UNSET",
    "profile_gradient",
    "radial_coordinate_map",
    "radial_coordinate_map_from_arrays",
    "resolve_convention",
    "resolve_reference_length",
]


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RadialCoordinate:
    """One radial coordinate of the registry: its name, unit, kind, range and definition."""

    name: str
    unit: str
    kind: str
    """``"flux"`` for a flux label, ``"geometric"`` for a length or a ratio of lengths."""
    range: tuple[float, float] | None
    """Axis and boundary value when the coordinate is normalized, else ``None``."""
    definition: str


RADIAL_COORDINATES: Mapping[str, RadialCoordinate] = {
    item.name: item
    for item in (
        RadialCoordinate(
            "psi_norm", "1", "flux", (0.0, 1.0),
            "(psi - psi_axis)/(psi_boundary - psi_axis)",
        ),
        RadialCoordinate("rho_pol_norm", "1", "flux", (0.0, 1.0), "sqrt(psi_norm)"),
        RadialCoordinate(
            "rho_tor_norm", "1", "flux", (0.0, 1.0),
            "sqrt(Phi/Phi_boundary), Phi = int q dpsi; never the sqrt(psi_norm) proxy",
        ),
        RadialCoordinate(
            "r_inboard", "m", "geometric", None,
            "major radius of the inboard crossing of the surface at the magnetic-axis height",
        ),
        RadialCoordinate(
            "r_outboard", "m", "geometric", None,
            "major radius of the outboard crossing of the surface at the magnetic-axis height",
        ),
        RadialCoordinate(
            "r_center", "m", "geometric", None,
            "(r_outboard + r_inboard)/2, the midplane centre of the surface",
        ),
        RadialCoordinate(
            "r_minor", "m", "geometric", None,
            "(r_outboard - r_inboard)/2, the midplane half-width of the surface; "
            "not R_out - R_axis and not the contour half-width",
        ),
        RadialCoordinate(
            "r_minor_norm", "1", "geometric", (0.0, 1.0),
            "r_minor/a_minor with a_minor = r_minor on the last closed surface; "
            "not rho_pol_norm, rho_tor_norm or sqrt(psi_norm)",
        ),
    )
}

#: Accepted ``reference_length`` names.  ``None`` asks for the dimensional gradient.
REFERENCE_LENGTHS: tuple[str | None, ...] = (
    None, "a_minor", "R_major_axis", "R_major_surface", "L_ref",
)


class _Unset:
    """Marks an argument the caller did not pass, where ``None`` has a meaning."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __reduce__(self):
        return (_Unset, ())


#: The default of :func:`profile_gradient` arguments that a convention may resolve.
UNSET = _Unset()


def _coordinate(name: Any) -> RadialCoordinate:
    if name not in RADIAL_COORDINATES:
        raise ValueError(
            f"{name!r} is not a radial coordinate; choose one of "
            f"{', '.join(RADIAL_COORDINATES)}. Normalization lengths (a_minor, "
            "R_major, L_ref) are reference_length values, not coordinates."
        )
    return RADIAL_COORDINATES[name]


# --------------------------------------------------------------------------
# the coordinate map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RadialCoordinateMap:
    """Every registered radial coordinate on one set of flux surfaces.

    ``coordinates`` maps each name of :data:`RADIAL_COORDINATES` to a
    :class:`~vaft.data.equilibrium.DerivedValue` on the same surfaces, in the
    same order.  An entry that could not be formed carries ``value=None`` and a
    ``reason``; an entry that exists only on part of the surfaces carries NaN
    on the rest, never an extrapolated number.  ``a_minor`` and
    ``R_major_axis`` are the scalar reference lengths of the slice, in the same
    form.
    """

    coordinates: Mapping[str, Any]
    a_minor: Any
    R_major_axis: Any
    time: float | None = None
    source: str = "unknown"

    def __getitem__(self, name: str):
        return self.coordinates[_coordinate(name).name]

    def values(self, name: str) -> np.ndarray:
        """The coordinate's values, refusing an unavailable one by name."""
        item = self[name]
        if not item.available:
            raise ValueError(f"{name} is unavailable on this equilibrium: {item.reason}")
        return np.asarray(item.value, dtype=float)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(name for name, item in self.coordinates.items() if item.available)


def _provenance(method: str, fields: tuple[str, ...] = (), *, time=None, convention=None,
                source_type: str = "native", notes: tuple[str, ...] = (),
                interpolation: str | None = None):
    from vaft.data.equilibrium import DerivationProvenance

    return DerivationProvenance(
        method=method, source_type=source_type, source_fields=fields, source_time=time,
        convention=convention, notes=notes, interpolation=interpolation,
    )


def _value(value, unit, definition, provenance, quality=None):
    from vaft.data.equilibrium import DerivedValue

    return DerivedValue(value, unit, definition, provenance, quality or {})


def _missing(unit, definition, reason, provenance):
    from vaft.data.equilibrium import DerivedValue

    return DerivedValue(None, unit, definition, provenance, reason=reason)


def _midplane_branches(eq) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str | None]:
    """R_in and R_out of every ``psi_1d`` surface at the axis height, or why not.

    The flux along the line ``Z = Z_axis`` is read from a bicubic spline of the
    grid, each side of the axis is cut at the first point where the normalized
    flux stops increasing outward, and the radius is inverted as a monotone
    PCHIP function of ``sqrt(psi_norm)``, in which the radius is close to
    linear near the axis.  A surface beyond the last crossing on a branch gets
    NaN: this is the logic of
    :func:`vaft.omas.update.update_equilibrium_profiles_1d_radial_coordinates`
    and :func:`vaft.process.equilibrium.psi_to_radial` without their
    extrapolation.
    """
    from scipy.interpolate import PchipInterpolator, RectBivariateSpline

    if eq.r is None or eq.z is None or eq.psi is None or eq.psi.shape != (eq.r.size, eq.z.size):
        return None, None, "a correctly shaped R/Z/psi grid is required"
    if eq.magnetic_axis is None or eq.psi_1d is None:
        return None, None, "the magnetic axis and psi_1d are required"
    if eq.psi_axis is None or eq.psi_boundary is None or eq.psi_axis == eq.psi_boundary:
        return None, None, "distinct psi_axis and psi_boundary are required"
    r_axis, z_axis = (float(v) for v in eq.magnetic_axis)
    if not (eq.r[0] < r_axis < eq.r[-1] and eq.z[0] <= z_axis <= eq.z[-1]):
        return None, None, "the magnetic axis is outside the R/Z grid"
    span = float(eq.psi_boundary - eq.psi_axis)
    spline = RectBivariateSpline(eq.r, eq.z, eq.psi, kx=3, ky=3)
    row = (spline(eq.r, np.array([z_axis]))[:, 0] - eq.psi_axis) / span
    target = np.sqrt(np.clip((eq.psi_1d - eq.psi_axis) / span, 0.0, None))

    def branch(indices: np.ndarray) -> np.ndarray:
        s_values, r_values = [0.0], [r_axis]
        for index in indices:
            s = float(row[index])
            if s <= s_values[-1]:
                if s_values[-1] == 0.0 and s <= 0.0:
                    continue  # within rounding of the axis, before the flux rises
                break
            s_values.append(s)
            r_values.append(float(eq.r[index]))
        out = np.full(target.size, np.nan)
        if len(s_values) < 3:
            return out
        x = np.sqrt(np.asarray(s_values))
        inside = target <= x[-1]
        # anti-alias: not a time series. Inverts the monotone midplane flux of one
        # equilibrium slice, R against sqrt(psi_norm); there is no sample rate.
        out[inside] = PchipInterpolator(x, np.asarray(r_values), extrapolate=False)(target[inside])
        return out

    outboard = branch(np.where(eq.r > r_axis)[0])
    inboard = branch(np.where(eq.r < r_axis)[0][::-1])
    return inboard, outboard, None


def radial_coordinate_map(source: Any, time: float | None = None) -> RadialCoordinateMap:
    """Every registered radial coordinate on one equilibrium slice's psi grid.

    The eight names of :data:`RADIAL_COORDINATES` are derived on the slice's own
    ``psi_1d`` surfaces, each as a :class:`~vaft.data.equilibrium.DerivedValue`
    with its definition and derivation record, together with the two scalar
    reference lengths ``a_minor`` and ``R_major_axis`` [-].

    Parameters
    ----------
    source : EquilibriumData, ODS, GEQDSK, IMAS equilibrium IDS or path
        Adapted through :func:`vaft.process.equilibrium.as_equilibrium`.  Needs
        the R-Z flux map, the magnetic axis and ``psi_1d``; ``rho_tor_norm``
        additionally needs ``q`` [-].
    time : float, optional
        The slice to use, matched by time within the adaptive tolerance of
        :func:`vaft.process.atomic.find_time_match_index`.  Required when the
        source holds more than one slice [s].

    Returns
    -------
    RadialCoordinateMap
        The coordinates, and ``a_minor`` and ``R_major_axis`` in metres.  An
        unavailable entry carries its reason; a surface outside the support of
        the midplane inversion carries NaN [-].

    Raises
    ------
    ValueError
        No slice at *time*, *time* missing for a multi-slice source, or a record
        whose time contradicts *time*.

    Processing steps
    ----------------
    1. Select the slice by time and adapt it to one ``EquilibriumData``.
    2. ``psi_norm`` and ``rho_pol_norm`` by normalization
       (:func:`vaft.process.equilibrium.derive_radial_coordinates`).
    3. ``rho_tor_norm`` from ``q`` through :func:`vaft.data._derived.rho_tor_profile`,
       unavailable when ``q`` cannot support a monotonic toroidal flux.
    4. ``r_inboard``/``r_outboard`` from the flux along ``Z = Z_axis``,
       inverted on each side of the axis without extrapolation.
    5. ``r_center``, ``r_minor``, then ``a_minor`` on the ``psi_norm = 1``
       surface and ``r_minor_norm = r_minor/a_minor``.

    Convention
    ----------
    The midplane is the horizontal line through the magnetic axis, so
    ``r_minor`` is the IMAS ``(r_outboard - r_inboard)/2``.  It differs from the
    contour half-width ``(max R - min R)/2`` that
    :func:`vaft.process.equilibrium.derive_global_descriptors` reports as
    ``minor_radius``, and from ``R_out - R_axis`` whenever the surfaces are
    Shafranov-shifted.  Every coordinate is COCOS-independent: each is either a
    ratio of fluxes with the same sign and scale or a length.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Surfaces that do not cross the axis height on both sides inside the grid
    (a branch that turns over before the boundary, an axis outside the grid)
    are NaN, and ``a_minor`` is unavailable when the boundary surface is.
    ``rho_tor_norm`` inherits the trapezoidal ``q`` integral and its axis-``q``
    sensitivity (#317).

    Provenance
    ----------
    .. [1] The IMAS ``profiles_1d.r_inboard``/``r_outboard`` definition, as
       :func:`vaft.omas.update.update_equilibrium_profiles_1d_radial_coordinates`
       computes it; this keeps that construction and drops its extrapolation
       (#551).
    """
    from vaft.data._derived import rho_tor_profile
    from vaft.process._equilibrium_parametric import as_equilibrium, derive_radial_coordinates

    eq = _select_slice(source, time, as_equilibrium)
    prov = dict(time=eq.time, convention=eq.convention,
                source_type=str(eq.metadata.get("source_type", "native")))
    base = derive_radial_coordinates(eq)
    coordinates: dict[str, Any] = {}
    psi_n_item, rho_pol_item = base["psi_n"], base["rho_pol_n"]
    reg = RADIAL_COORDINATES
    coordinates["psi_norm"] = _rename(psi_n_item, reg["psi_norm"])
    coordinates["rho_pol_norm"] = _rename(rho_pol_item, reg["rho_pol_norm"])

    if not psi_n_item.available:
        reason = psi_n_item.reason
        for name in ("rho_tor_norm", "r_inboard", "r_outboard", "r_center", "r_minor", "r_minor_norm"):
            coordinates[name] = _missing(reg[name].unit, reg[name].definition, reason,
                                         _provenance("unavailable", **prov))
        a_minor = _missing("m", _A_MINOR_DEFINITION, reason, _provenance("unavailable", **prov))
        return RadialCoordinateMap(coordinates, a_minor, _axis_radius(eq, prov), eq.time, _describe(eq))

    psi_n = np.asarray(psi_n_item.value, dtype=float)
    monotonic = bool(np.all(np.diff(psi_n) > 0.0))

    rho = None if eq.q is None else rho_tor_profile(eq.q, eq.psi_1d)
    if rho is None or not monotonic:
        coordinates["rho_tor_norm"] = _missing(
            "1", reg["rho_tor_norm"].definition,
            "q on the psi_1d grid does not give a monotonic toroidal flux"
            if monotonic else "psi_norm is not strictly increasing",
            _provenance("unavailable", ("q", "psi_1d"), **prov))
    else:
        coordinates["rho_tor_norm"] = _value(
            rho.rho_tor_norm, "1", reg["rho_tor_norm"].definition,
            _provenance("cumulative trapezoidal integral of q dpsi (rho_tor_profile)",
                        ("q", "psi_1d"), **prov))

    inboard, outboard, reason = _midplane_branches(eq) if monotonic else (None, None, "psi_norm is not strictly increasing")
    fields = ("profiles_2d.psi", "magnetic_axis", "psi_1d")
    method = "bicubic flux along Z = Z_axis, monotone PCHIP inversion in sqrt(psi_norm), no extrapolation"
    if reason is not None:
        for name in ("r_inboard", "r_outboard", "r_center", "r_minor", "r_minor_norm"):
            coordinates[name] = _missing(reg[name].unit, reg[name].definition, reason,
                                         _provenance("unavailable", fields, **prov))
        a_minor = _missing("m", _A_MINOR_DEFINITION, reason, _provenance("unavailable", fields, **prov))
        return RadialCoordinateMap(coordinates, a_minor, _axis_radius(eq, prov), eq.time, _describe(eq))

    boundary = int(np.argmin(np.abs(psi_n - 1.0)))
    at_boundary = abs(psi_n[boundary] - 1.0) < 1e-9
    _geometric(coordinates, inboard, outboard, boundary if at_boundary else None,
               _provenance(method, fields, interpolation="pchip", **prov),
               "no psi_1d surface is at psi_norm = 1" if not at_boundary else None)
    a_minor = coordinates.pop("_a_minor")
    return RadialCoordinateMap(coordinates, a_minor, _axis_radius(eq, prov), eq.time, _describe(eq))


_A_MINOR_DEFINITION = (
    "midplane minor radius of the last closed flux surface, (R_out - R_in)/2 at the "
    "magnetic-axis height; differs from derive_global_descriptors' minor_radius, which "
    "is the contour half-width (max R - min R)/2"
)


def _rename(item, coordinate: RadialCoordinate):
    from dataclasses import replace

    return replace(item, unit=coordinate.unit, definition=coordinate.definition)


def _describe(eq) -> str:
    return str(eq.metadata.get("source_type", eq.convention.source))


def _axis_radius(eq, prov):
    if eq.magnetic_axis is None or not np.isfinite(eq.magnetic_axis[0]):
        return _missing("m", "major radius of the magnetic axis", "no magnetic axis",
                        _provenance("unavailable", ("magnetic_axis",), **prov))
    return _value(float(eq.magnetic_axis[0]), "m", "major radius of the magnetic axis",
                  _provenance("direct read", ("magnetic_axis",), **prov))


def _geometric(coordinates, inboard, outboard, boundary, provenance, boundary_reason):
    """Fill the five midplane coordinates and ``_a_minor`` from R_in and R_out."""
    reg = RADIAL_COORDINATES
    inboard = np.asarray(inboard, dtype=float)
    outboard = np.asarray(outboard, dtype=float)
    support = np.isfinite(inboard) & np.isfinite(outboard)
    quality = {"surfaces_in_support": int(np.count_nonzero(support)), "surfaces": int(support.size)}
    centre = 0.5 * (outboard + inboard)
    minor = 0.5 * (outboard - inboard)
    for name, values in (("r_inboard", inboard), ("r_outboard", outboard),
                         ("r_center", centre), ("r_minor", minor)):
        coordinates[name] = _value(values, reg[name].unit, reg[name].definition, provenance, quality)
    if boundary is None or not np.isfinite(minor[boundary]) or minor[boundary] <= 0.0:
        reason = boundary_reason or "the boundary surface does not cross the axis height on both sides"
        coordinates["_a_minor"] = _missing("m", _A_MINOR_DEFINITION, reason, provenance)
        coordinates["r_minor_norm"] = _missing("1", reg["r_minor_norm"].definition, reason, provenance)
        return
    a = float(minor[boundary])
    coordinates["_a_minor"] = _value(a, "m", _A_MINOR_DEFINITION, provenance, {"surface_index": boundary})
    coordinates["r_minor_norm"] = _value(minor / a, "1", reg["r_minor_norm"].definition, provenance, quality)


def _select_slice(source, time, as_equilibrium):
    """One ``EquilibriumData``, chosen by time and never by position alone."""
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process.atomic import find_time_match_index

    if isinstance(source, EquilibriumData):
        _check_time(source.time, time, "the equilibrium")
        return source
    times = None
    try:
        times = np.asarray(source["equilibrium.time"], dtype=float).reshape(-1)
    except Exception:
        times = None
    if times is None or times.size == 0:
        try:
            count = len(source["equilibrium.time_slice"])
        except Exception:
            count = 1
        if count > 1:
            raise ValueError(
                f"the source holds {count} equilibrium slices and no equilibrium.time to choose "
                "among them by; a slice is never chosen by position alone")
        eq = as_equilibrium(source)
        return _select_slice(eq, time, as_equilibrium)
    if time is None:
        if times.size != 1:
            raise ValueError(
                f"the source holds {times.size} equilibrium slices; pass time= to choose one")
        return as_equilibrium(source, time_index=0)
    index = find_time_match_index(times, float(time))
    if index is None:
        raise ValueError(f"no equilibrium slice at t = {time} s (slices span {times.min()} to {times.max()} s)")
    return as_equilibrium(source, time_index=index)


def _check_time(have, want, what):
    """Refuse a requested time the record cannot confirm."""
    if want is None:
        return
    if have is None:
        raise ValueError(f"{what} records no time, so t = {want} s cannot be confirmed")
    if not np.isclose(float(have), float(want), atol=1e-6, rtol=0.0):
        raise ValueError(f"{what} is at t = {have} s, not the requested {want} s")


def radial_coordinate_map_from_arrays(
    *,
    r_inboard: Any,
    r_outboard: Any,
    psi_norm: Any = None,
    rho_tor_norm: Any = None,
    magnetic_axis_r: float | None = None,
    time: float | None = None,
    source: str = "caller-supplied arrays",
) -> RadialCoordinateMap:
    """A coordinate map from per-surface arrays a code or file already carries.

    For a source that is not an equilibrium -- an ``input.gacode`` profile, a
    stored ``profiles_1d`` -- the midplane radii are given rather than solved.
    The surfaces are taken in the order given, axis first, and the **last one**
    is the reference surface of ``a_minor``; that is the LCFS for a complete
    profile and GACODE's own ``a = rmin[-1]`` for a truncated one [-].

    Parameters
    ----------
    r_inboard : array_like
        Inboard midplane major radius of each surface [m].
    r_outboard : array_like
        Outboard midplane major radius of each surface [m].
    psi_norm : array_like, optional
        Normalized poloidal flux of each surface; ``rho_pol_norm`` is derived
        from it [-].
    rho_tor_norm : array_like, optional
        Normalized toroidal-flux radius of each surface.  Refused when it is the
        ``sqrt(psi_norm)`` proxy [-].
    magnetic_axis_r : float, optional
        Major radius of the magnetic axis, the ``R_major_axis`` reference
        length [m].
    time : float, optional
        The time the arrays describe, recorded only [s].
    source : str, optional
        What the arrays came from, recorded in every provenance entry [-].

    Returns
    -------
    RadialCoordinateMap
        Every coordinate the arrays determine; the rest unavailable with a
        reason [-].

    Raises
    ------
    ValueError
        Arrays of different lengths, fewer than three surfaces, or non-finite
        midplane radii.

    Defaults
    --------
    ``source`` is a hard-coded label; every other default is ``None``, meaning
    "not supplied", and the coordinates that need it are unavailable rather
    than guessed.

    Convention
    ----------
    ``r_minor = (r_outboard - r_inboard)/2`` and ``r_center`` their mean, as in
    :func:`radial_coordinate_map`; ``a_minor`` is ``r_minor`` on the last
    surface, recorded as such in its provenance notes.  A ``rho_tor_norm``
    equal to ``sqrt(psi_norm)`` to within
    :data:`vaft.data._derived.RHO_POL_PROXY_TOLERANCE` is the proxy older
    files store under the toroidal label, and is marked unavailable.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    Nothing checks that the arrays describe nested surfaces beyond the
    monotonicity :func:`profile_gradient` demands when it maps between them.
    The proxy test needs ``psi_norm``; without it a proxy cannot be recognised.

    Provenance
    ----------
    .. [1] GACODE ``input.gacode``: ``rmin`` and ``rmaj`` are the midplane
       half-width and centre, and ``a`` is ``rmin`` at the last grid point
       (https://gacode.io/input_gacode.html).
    """
    from vaft.data._derived import is_rho_pol_proxy

    inboard = np.asarray(r_inboard, dtype=float).reshape(-1)
    outboard = np.asarray(r_outboard, dtype=float).reshape(-1)
    if inboard.size != outboard.size or inboard.size < 3:
        raise ValueError("r_inboard and r_outboard must be the same length, at least three surfaces")
    if not (np.all(np.isfinite(inboard)) and np.all(np.isfinite(outboard))):
        raise ValueError("the midplane radii must be finite")
    reg = RADIAL_COORDINATES
    notes = ("a_minor is r_minor on the last supplied surface",)
    prov = _provenance("caller-supplied arrays", ("r_inboard", "r_outboard"), time=time,
                       source_type=source, notes=notes)
    coordinates: dict[str, Any] = {}
    psi = None
    if psi_norm is None:
        for name in ("psi_norm", "rho_pol_norm"):
            coordinates[name] = _missing("1", reg[name].definition, "psi_norm was not supplied", prov)
    else:
        psi = np.asarray(psi_norm, dtype=float).reshape(-1)
        if psi.size != inboard.size:
            raise ValueError("psi_norm must have one value per surface")
        coordinates["psi_norm"] = _value(psi, "1", reg["psi_norm"].definition, prov)
        coordinates["rho_pol_norm"] = _value(np.sqrt(np.clip(psi, 0.0, None)), "1",
                                             reg["rho_pol_norm"].definition, prov)
    if rho_tor_norm is None:
        coordinates["rho_tor_norm"] = _missing("1", reg["rho_tor_norm"].definition,
                                               "rho_tor_norm was not supplied", prov)
    else:
        rho = np.asarray(rho_tor_norm, dtype=float).reshape(-1)
        if rho.size != inboard.size:
            raise ValueError("rho_tor_norm must have one value per surface")
        if psi is not None and is_rho_pol_proxy(rho, psi):
            coordinates["rho_tor_norm"] = _missing(
                "1", reg["rho_tor_norm"].definition,
                "the supplied rho_tor_norm is sqrt(psi_norm), the rho_pol proxy, not a "
                "toroidal-flux radius", prov)
        else:
            coordinates["rho_tor_norm"] = _value(rho, "1", reg["rho_tor_norm"].definition, prov)
    _geometric(coordinates, inboard, outboard, inboard.size - 1, prov, None)
    a_minor = coordinates.pop("_a_minor")
    axis = (
        _missing("m", "major radius of the magnetic axis", "magnetic_axis_r was not supplied", prov)
        if magnetic_axis_r is None
        else _value(float(magnetic_axis_r), "m", "major radius of the magnetic axis", prov)
    )
    order = tuple(RADIAL_COORDINATES)
    return RadialCoordinateMap({name: coordinates[name] for name in order}, a_minor, axis, time, source)


# --------------------------------------------------------------------------
# reference lengths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceLength:
    """A normalizing length together with what it physically is.

    ``name`` is the :data:`REFERENCE_LENGTHS` entry and ``symbol`` its notation
    (``a``, ``R_0``, ``R``, ``L_ref``).  ``length`` is the
    :class:`~vaft.data.equilibrium.DerivedValue` carrying the value, unit,
    definition and provenance; it is ``None`` only for ``name=None``, the
    dimensional gradient.  ``R_major_surface`` is the one local length: its
    value is an array on the map's surfaces.
    """

    name: str | None
    symbol: str | None
    length: Any = None

    @property
    def value(self):
        return None if self.length is None else self.length.value

    def record(self) -> dict[str, Any] | None:
        """The plain-mapping form stored in a :class:`ProfileGradient`'s metadata."""
        if self.name is None:
            return None
        value = self.length.value
        return {
            "name": self.name,
            "definition": self.length.definition,
            "symbol": self.symbol,
            "value": value.tolist() if isinstance(value, np.ndarray) else value,
            "unit": self.length.unit,
            "provenance": _plain(asdict(self.length.provenance)),
        }


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


_SYMBOLS = {"a_minor": "a", "R_major_axis": "R_0", "R_major_surface": "R", "L_ref": "L_ref"}


def resolve_reference_length(
    name: str | None,
    coordinate_map: RadialCoordinateMap,
    metadata: Mapping[str, Any] | None = None,
) -> ReferenceLength:
    """The reference length a normalized gradient is multiplied by, with its definition.

    Turns a :data:`REFERENCE_LENGTHS` name into a value with a unit, a symbol, a
    physical definition and the provenance of the equilibrium it came from, so
    that ``a/L_T`` and ``R/L_T`` are never reported by label alone [m].

    Parameters
    ----------
    name : str or None
        ``None`` (the dimensional gradient), ``"a_minor"``, ``"R_major_axis"``,
        ``"R_major_surface"`` or ``"L_ref"`` [-].
    coordinate_map : RadialCoordinateMap
        The slice the length is read from [-].
    metadata : mapping, optional
        For ``"L_ref"`` only: ``metadata["reference_length"]`` must be a mapping
        with a positive ``value`` in metres and a non-empty ``definition``; an
        optional ``symbol`` and ``source`` are recorded [m].

    Returns
    -------
    ReferenceLength
        The resolved length; for ``name=None`` one with no length at all [m].

    Raises
    ------
    ValueError
        An unknown name, a length the slice cannot supply (no LCFS crossing, no
        axis), or ``"L_ref"`` without an explicit definition.

    Convention
    ----------
    ``a_minor`` is the midplane ``r_minor`` of the last closed surface -- not the
    contour half-width ``(max R - min R)/2`` of
    :func:`vaft.process.equilibrium.derive_global_descriptors`.  ``R_major_axis``
    is the magnetic-axis major radius, symbol ``R_0``; ``R_major_surface`` is
    ``r_center`` on each surface, a local length.  ``L_ref`` is never assumed:
    an unknown ``L_ref`` does not become ``a_minor``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``L_ref`` is accepted only in metres; a code that normalizes by a
    dimensionless or flux-based length must be described by its own preset.

    Provenance
    ----------
    .. [1] Issue #551 sections 5, 6 and 15: a reference length keeps its value,
       unit, definition and source, and an ambiguous one is refused.
    """
    if name not in REFERENCE_LENGTHS:
        raise ValueError(
            f"reference_length must be one of {', '.join(map(repr, REFERENCE_LENGTHS))}; got {name!r}")
    if name is None:
        return ReferenceLength(None, None, None)
    if name == "a_minor":
        item = coordinate_map.a_minor
    elif name == "R_major_axis":
        item = coordinate_map.R_major_axis
    elif name == "R_major_surface":
        centre = coordinate_map["r_center"]
        if not centre.available:
            item = centre
        else:
            from dataclasses import replace

            item = replace(centre, definition="local major radius of each surface, r_center = "
                                              "(r_outboard + r_inboard)/2")
    else:
        item = _explicit_length(metadata)
    if not item.available:
        raise ValueError(f"reference_length={name!r} is unavailable: {item.reason}")
    symbol = _SYMBOLS[name]
    if name == "L_ref":
        symbol = str((metadata or {}).get("reference_length", {}).get("symbol", symbol))
    return ReferenceLength(name, symbol, item)


def _explicit_length(metadata):
    record = (metadata or {}).get("reference_length")
    if not isinstance(record, Mapping):
        raise ValueError(
            "reference_length='L_ref' needs metadata={'reference_length': {'value': ..., "
            "'unit': 'm', 'definition': ...}}; an L_ref without a stated definition is not "
            "assumed to be any particular length")
    definition = str(record.get("definition") or "").strip()
    if "unit" not in record:
        raise ValueError("the L_ref record has no unit; state 'unit': 'm'")
    unit = str(record["unit"])
    try:
        value = float(record.get("value"))
    except (TypeError, ValueError):
        value = float("nan")
    if not definition:
        raise ValueError("the L_ref record has no definition; say what length it is")
    if unit != "m":
        raise ValueError(f"L_ref must be given in m, not {unit!r}")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"L_ref must be a positive finite length; got {record.get('value')!r}")
    source = str(record.get("source", "caller metadata"))
    return _value(value, "m", definition, _provenance("caller-supplied", ("metadata.reference_length",),
                                                      source_type=source))


# --------------------------------------------------------------------------
# code conventions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConventionPreset:
    """What one code's gradient normalization is, and where that statement comes from.

    ``gradient_coordinate`` and ``reference_length`` are filled for a code whose
    normalization is fixed; ``None`` there means it depends on the run, and
    :func:`resolve_convention` reads the run's ``metadata`` instead.  ``required``
    names the metadata keys that decide it; ``source`` is the documentation it is
    taken from.
    """

    name: str
    description: str
    source: str
    gradient_coordinate: str | None = None
    reference_length: str | None = None
    required: tuple[str, ...] = ()


CONVENTIONS: Mapping[str, ConventionPreset] = {
    item.name: item
    for item in (
        ConventionPreset(
            "tglf", "RLTS_*/RLNS_* = -a d ln(T, n)/dr, r = rmin the midplane minor radius, "
            "a = rmin on the last surface of input.gacode",
            "https://gacode.io/tglf/tglf_list.html; https://gacode.io/input_gacode.html",
            "r_minor", "a_minor",
        ),
        ConventionPreset(
            "cgyro", "DLNTDR/DLNNDR = -a d ln(T, n)/dr, the same GACODE r and a as TGLF",
            "https://gacode.io/cgyro/cgyro_list.html; https://gacode.io/input_gacode.html",
            "r_minor", "a_minor",
        ),
        ConventionPreset(
            "gs2", "tprim/fprim = -(1/T) dT/d rho_N with rho_N = rho/L_ref and rho selected "
            "by irho; irho = 2 is the midplane half-diameter, and L_ref is the LCFS midplane "
            "half-diameter a except in a Miller (local_eq) run, where it is the run's own",
            "https://gyrokinetics.gitlab.io/gs2/page/namelists/index.html (irho, rhoc, "
            "local_eq, tprim, fprim); "
            "https://gyrokinetics.gitlab.io/gs2/page/user_manual/normalisations.html",
            required=("irho", "local_eq"),
        ),
        ConventionPreset(
            "gkw", "rlt/rln = R_ref/L_T = -(1/T) dT/dpsi with psi = (R_max - R_min)/(2 R_ref); "
            "R_ref depends on geom_type",
            "GKW manual, doc/manual/practise.tex "
            "(https://bitbucket.org/gkw/gkw/src/develop/doc/manual/)",
            required=("geom_type",),
        ),
        ConventionPreset(
            "gene", "omt/omn = -L_ref d ln(T, n)/dx; both x and L_ref depend on the run's "
            "geometry and units settings, so only the run's own statement of them is used",
            "Goerler et al., J. Comput. Phys. 230 (2011) 7053, "
            "http://genecode.org/PAPERS_2/YJCPH3630.pdf; the GENE manual is not public",
            required=("gradient_coordinate", "reference_length"),
        ),
    )
}


def resolve_convention(name: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The explicit gradient coordinate and reference length a code convention stands for.

    A convention is a shortcut, not a definition: this returns the two explicit
    choices of :func:`profile_gradient` it resolves to, with the documentation
    it was resolved from, and refuses when the code's normalization depends on
    run settings the caller did not pass [-].

    Parameters
    ----------
    name : str
        One of :data:`CONVENTIONS`: ``tglf``, ``cgyro``, ``gs2``, ``gkw``,
        ``gene`` [-].
    metadata : mapping, optional
        The run's own settings.  ``gs2`` needs ``irho`` and ``local_eq`` (and
        ``reference_length`` when ``local_eq`` is true); ``gkw`` needs
        ``geom_type`` (and ``reference_length`` for ``s-alpha``); ``gene``
        needs ``gradient_coordinate`` and ``reference_length``.  A
        ``reference_length`` entry is a mapping with ``value`` [m],
        ``unit`` and ``definition`` [-].

    Returns
    -------
    dict
        ``convention``, ``gradient_coordinate``, ``reference_length`` (a
        :data:`REFERENCE_LENGTHS` name), ``basis`` (the rule applied),
        ``source`` (the documentation), and ``metadata_used`` [-].

    Raises
    ------
    ValueError
        An unknown convention, a missing required setting, or a setting whose
        meaning the documentation does not fix.

    Convention
    ----------
    ``tglf`` and ``cgyro``: ``r_minor`` and ``a_minor``, the GACODE midplane
    ``rmin`` and its last-surface value.  ``gs2``: ``irho = 2`` with
    ``local_eq = False`` is ``r_minor`` and ``a_minor``; with ``local_eq = True``
    it is ``r_minor`` and the run's ``L_ref``; ``irho`` 1, 3 and 4 are refused.
    ``gkw``: ``circ`` is ``r_minor`` and ``R_major_axis``; ``s-alpha`` is
    ``r_minor`` and the run's stated ``L_ref``; ``miller``, ``fourier``, ``mxh``
    and ``chease`` are refused.  ``gene``: whatever the run states, never a
    fixed major radius.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    GS2 ``irho = 1`` or ``3`` labels surfaces by a flux ratio while ``tprim`` is
    defined against ``rho/L_ref``, and the documentation does not say how the
    two combine, so they are refused rather than guessed.  GKW's ``psi`` is the
    half-width between the surface's own ``R_max`` and ``R_min``; it equals the
    registry's axis-height ``r_minor`` only for a surface symmetric about the
    axis height, so only the circular geometries, where that holds by
    construction, are resolved.  The GENE manual is not
    publicly available, so no GENE rule is encoded.

    Provenance
    ----------
    .. [1] GACODE: https://gacode.io/tglf/tglf_list.html,
       https://gacode.io/cgyro/cgyro_list.html, https://gacode.io/input_gacode.html.
    .. [2] GS2 namelist reference, ``irho``, ``rhoc``, ``local_eq``, ``tprim`` and
       ``fprim``: https://gyrokinetics.gitlab.io/gs2/page/namelists/index.html; the
       reference length, "half the diameter of the last closed flux surface,
       measured on the midplane":
       https://gyrokinetics.gitlab.io/gs2/page/user_manual/normalisations.html.
    .. [3] GKW manual, ``doc/manual/practise.tex``: ``psi = (R_max - R_min)/(2 R_ref)``,
       ``R_ref/L_T = -(1/T) dT/dpsi``, and ``R_ref`` per geometry
       (https://bitbucket.org/gkw/gkw/src/develop/doc/manual/).
    .. [4] T. Goerler et al., "The global version of the gyrokinetic turbulence
       code GENE", J. Comput. Phys. 230 (2011) 7053: a generic reference length
       and ``1/L_T = -d ln T/dx`` (http://genecode.org/PAPERS_2/YJCPH3630.pdf).
    """
    if name not in CONVENTIONS:
        raise ValueError(f"unknown convention {name!r}; choose one of {', '.join(CONVENTIONS)}")
    preset = CONVENTIONS[name]
    metadata = dict(metadata or {})
    missing = [key for key in preset.required if key not in metadata]
    if missing:
        raise ValueError(
            f"convention={name!r} depends on the run's {', '.join(missing)}: {preset.description}. "
            f"Pass them in metadata; nothing is assumed ({preset.source})")
    used = {key: metadata[key] for key in preset.required}
    if name in ("tglf", "cgyro"):
        gradient, reference, basis = preset.gradient_coordinate, preset.reference_length, preset.description
    elif name == "gs2":
        gradient, reference, basis = _gs2(metadata)
    elif name == "gkw":
        gradient, reference, basis = _gkw(metadata)
    else:
        gradient = metadata["gradient_coordinate"]
        _coordinate(gradient)
        reference = "L_ref"
        _require_length_record(metadata, "gene")
        basis = "the run's own x and L_ref, as supplied in metadata"
    if reference == "L_ref":
        used["reference_length"] = metadata["reference_length"]
    return {
        "convention": name,
        "gradient_coordinate": gradient,
        "reference_length": reference,
        "basis": basis,
        "source": preset.source,
        "metadata_used": used,
    }


def _require_length_record(metadata, code):
    if not isinstance(metadata.get("reference_length"), Mapping):
        raise ValueError(
            f"convention={code!r} here normalizes by the run's own L_ref; pass "
            "metadata['reference_length'] = {'value': ..., 'unit': 'm', 'definition': ...}")


def _gs2(metadata):
    irho = metadata["irho"]
    if irho != 2:
        raise ValueError(
            f"GS2 irho = {irho!r} is not resolved: irho 1 and 3 label surfaces by a flux "
            "ratio while tprim is defined against rho/L_ref, and the GS2 documentation does "
            "not state how they combine; irho 4 is the dipole rho_mid. Only irho = 2 is fixed.")
    local_eq = metadata["local_eq"]
    if not isinstance(local_eq, (bool, np.bool_)):
        raise ValueError(f"GS2 local_eq must be a boolean; got {local_eq!r}")
    if local_eq:
        _require_length_record(metadata, "gs2")
        return ("r_minor", "L_ref",
                "irho = 2, local_eq: rho_N = midplane half-diameter / L_ref, L_ref the run's own")
    return ("r_minor", "a_minor",
            "irho = 2, numerical equilibrium: L_ref = a, the LCFS midplane half-diameter")


def _gkw(metadata):
    geometry = str(metadata["geom_type"]).strip().lower()
    if geometry == "circ":
        return ("r_minor", "R_major_axis",
                "geom_type circ: psi = r/R_ref with R_ref the magnetic-axis major radius")
    if geometry == "s-alpha":
        _require_length_record(metadata, "gkw")
        return ("r_minor", "L_ref",
                "geom_type s-alpha: psi = r/R_ref with R_ref the run's own choice")
    if geometry in ("miller", "fourier", "mxh", "chease"):
        raise ValueError(
            f"GKW geom_type {geometry!r} is not resolved: its psi and R_ref come from the "
            "surface's own R extremes (or CHEASE's R0EXP), which equal this registry's "
            "axis-height crossings only for a surface symmetric about the axis height, and "
            "nothing here checks that.")
    raise ValueError(f"unknown GKW geom_type {metadata['geom_type']!r}")


# --------------------------------------------------------------------------
# the gradient
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileGradient:
    """A normalized logarithmic gradient, with everything needed to reproduce it.

    ``values`` are ``-L d ln f/dx_g`` at the abscissa ``coordinate``, whose
    name is ``coordinate_name``; ``unit`` is ``"1"`` for a normalized gradient
    and the inverse of the gradient coordinate's unit for ``L = None``.
    ``metadata`` is the issue #551 section 10 record: ``quantity``,
    ``source_quantity``, ``plot_coordinate``, ``profile_coordinate``,
    ``gradient_coordinate``, ``reference_length``, ``requested_convention``,
    ``resolved_convention``, ``mathematical_definition``, ``unit``, ``method``
    and ``time``.
    """

    values: np.ndarray
    coordinate: np.ndarray
    coordinate_name: str
    unit: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: Gradient coordinates a reference length may normalize: lengths that grow outward.
#: ``r_inboard`` falls outward and ``r_center`` falls under a Shafranov shift, so
#: a decay length against either would change sign.
_OUTWARD_LENGTHS = ("r_minor", "r_outboard")


def _strictly_monotonic(values: np.ndarray) -> bool:
    step = np.diff(values)
    return bool(np.all(step > 0.0) or np.all(step < 0.0))


def _spline(x: np.ndarray, y: np.ndarray):
    """The result to a new abscissa: a not-a-knot cubic, as GACODE's ``cub_spline1``."""
    from scipy.interpolate import CubicSpline

    order = np.argsort(x)
    # anti-alias: not a time series. A cubic spline over one radial coordinate of a
    # single slice, GACODE's cub_spline order; there is no sample rate to reduce.
    return CubicSpline(x[order], y[order])


def _regular(name: str, x: np.ndarray) -> np.ndarray:
    """The variable a relation is regular in at the axis: ``sqrt`` of ``psi_norm``, else itself."""
    return np.sqrt(np.clip(x, 0.0, None)) if name == "psi_norm" else x


def _map_relation(name: str, x_map: np.ndarray, y_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """A map relation ``y(x)`` evaluated at *points* by a not-a-knot cubic spline.

    The spline is differentiated afterwards, so it has to be smooth across the
    map's surfaces; PCHIP's limited node slopes are not.  Every radius goes as ``sqrt(psi_norm)`` near the axis, which no polynomial in
    ``psi_norm`` follows across the first interval of the map; a profile on a
    finer grid there would see the kink.  So a relation whose argument is
    ``psi_norm`` is interpolated in ``sqrt(psi_norm)``, in which it is regular.
    """
    from scipy.interpolate import CubicSpline

    x_map, points = _regular(name, x_map), _regular(name, points)
    # _check_support has already refused anything beyond rounding of the map's ends
    points = np.clip(points, np.min(x_map), np.max(x_map))
    order = np.argsort(x_map)
    # anti-alias: not a time series. Evaluates a radial relation of one slice on the
    # profile's own grid; there is no sample rate to reduce.
    return CubicSpline(x_map[order], y_map[order], extrapolate=False)(points)


def _check_support(points: np.ndarray, support: np.ndarray, what: str) -> None:
    low, high = float(np.min(support)), float(np.max(support))
    tolerance = 1e-12 * max(1.0, abs(low), abs(high))
    outside = (points < low - tolerance) | (points > high + tolerance)
    if np.any(outside):
        raise ValueError(
            f"{what}: {int(np.count_nonzero(outside))} point(s) lie outside the radial support "
            f"[{low:.6g}, {high:.6g}]; nothing is extrapolated")


def _on_map(coordinate_map, source_name, target_name, points, what):
    """``target`` as a function of ``source`` from the map, at ``points``."""
    source = coordinate_map.values(source_name)
    target = coordinate_map.values(target_name)
    valid = np.isfinite(source) & np.isfinite(target)
    if np.count_nonzero(valid) < 3:
        raise ValueError(f"{what}: fewer than three surfaces carry both {source_name} and {target_name}")
    source, target = source[valid], target[valid]
    if not _strictly_monotonic(source):
        raise ValueError(f"{what}: {source_name} is not strictly monotonic on the map")
    _check_support(points, source, what)
    return source, target


def profile_gradient(
    profile: Any,
    grid: Any,
    profile_coordinate: str,
    *,
    equilibrium: Any,
    coordinate: str | None = None,
    gradient_coordinate: Any = UNSET,
    reference_length: Any = UNSET,
    convention: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    at: Any = None,
    time: float | None = None,
    quantity: str | None = None,
    source_quantity: str = "f",
) -> ProfileGradient:
    """The normalized logarithmic gradient ``-L d ln f/dx_g`` of a radial profile.

    The derivative is taken on the profile's own grid, carried to the gradient
    coordinate by the chain rule through the equilibrium's coordinate map,
    multiplied by the reference length, and placed on the requested abscissa;
    the result keeps every one of those choices in its metadata [-].

    Parameters
    ----------
    profile : array_like
        Strictly positive profile values, such as a temperature or density, in
        any unit -- only ``ln f`` is differentiated [-].
    grid : array_like
        Where the profile is sampled, in *profile_coordinate*, strictly
        monotonic, at least three points [-].
    profile_coordinate : str
        The :data:`RADIAL_COORDINATES` name *grid* is in [-].
    equilibrium : RadialCoordinateMap, EquilibriumData, ODS, GEQDSK or path
        The slice whose surfaces relate the coordinates; anything but a map
        goes through :func:`radial_coordinate_map` [-].
    coordinate : str, optional
        The abscissa of the result, a :data:`RADIAL_COORDINATES` name [-].
    gradient_coordinate : str, optional
        What the derivative is taken with respect to, a
        :data:`RADIAL_COORDINATES` name [-].
    reference_length : str or None, optional
        One of :data:`REFERENCE_LENGTHS`; ``None`` gives the dimensional
        gradient.  Must be stated, here or through *convention* [-].
    convention : str, optional
        A :data:`CONVENTIONS` preset resolving *gradient_coordinate* and
        *reference_length*; an explicit value that disagrees with it is refused [-].
    metadata : mapping, optional
        Run settings for *convention* and the ``L_ref`` record, as
        :func:`resolve_convention` and :func:`resolve_reference_length` read
        them [-].
    at : array_like, optional
        Abscissa values, in *coordinate*, to interpolate the result to after
        differentiating; by default the result stays on the profile's grid [-].
    time : float, optional
        The equilibrium slice, when *equilibrium* is not already a map [s].
    quantity : str, optional
        Name of the result, recorded as ``quantity`` [-].
    source_quantity : str, optional
        Name of the differentiated profile, used in ``quantity`` and in the
        ``mathematical_definition`` string [-].

    Returns
    -------
    ProfileGradient
        Values, abscissa, unit and the resolved record.  The unit is ``1`` with a
        reference length, ``m^-1`` without one for a length gradient
        coordinate, and ``1`` for a normalized one [-].

    Raises
    ------
    ValueError
        An unknown coordinate, a profile value that is not positive, a mapping
        that is not strictly monotonic, a point outside the map's radial
        support, an unavailable coordinate or reference length, a reference
        length with a gradient coordinate that is not an outward length, an unstated
        reference length, or a convention that conflicts with the explicit
        arguments or cannot be resolved.

    Processing steps
    ----------------
    1. Resolve *convention*, if any, into the gradient coordinate and the
       reference length, and refuse an explicit argument that disagrees.
    2. ``d ln f/dx_p`` on the profile's own grid by the three-point Lagrange
       derivative (``numpy.gradient``, second-order edges -- GACODE's
       ``bound_deriv``).
    3. When ``x_g != x_p``: ``x_g`` on the profile grid from the map by a cubic
       spline, and ``d ln f/dx_g = (dx_p/dx_g) d ln f/dx_p`` with both
       derivatives three-point on the profile grid.  For a ``psi_norm`` profile
       both are taken in ``sqrt(psi_norm)``, in which radii and ``ln f`` are
       regular at the axis.
    4. Multiply by ``-L``, the reference length resolved on the same map; a
       local ``R_major_surface`` is splined to the profile grid first.
    5. Express the profile grid in *coordinate*, and spline the result to *at*
       when given -- differentiate first, then interpolate.

    Defaults
    --------
    ``coordinate`` defaults to *profile_coordinate* and ``gradient_coordinate``
    to ``r_minor``, the conventional gyrokinetic choice, when no convention is
    given.  ``reference_length`` has no default: an unstated one is refused, so
    the result never silently becomes ``a/L``.  ``source_quantity = "f"`` is a
    hard-coded placeholder name.

    Convention
    ----------
    ``r_minor`` is the midplane half-width ``(R_out - R_in)/2`` at the
    magnetic-axis height, and ``a_minor`` its LCFS value; neither is the
    contour half-width of :func:`vaft.process.equilibrium.derive_global_descriptors`.
    The sign is that of a decay length: with respect to an outward-increasing
    gradient coordinate, a profile falling outward has a positive gradient.  A
    reference length is accepted only with ``r_minor`` or ``r_outboard``, the
    lengths that increase outward.  The same physical derivative under two reference
    lengths differs exactly by their ratio, so ``R_0/L_T = (R_0/a) a/L_T``.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The gradient coordinate is splined across the map's surfaces, so the result
    is only as smooth as the map.  A gradient with respect to a flux label is
    undefined where that label is stationary in the profile coordinate -- at
    the axis, against a radius -- and such a point is refused.  The map comes
    from one slice; profile and equilibrium at different times are the caller's
    responsibility, and *time* only selects the slice.

    Provenance
    ----------
    .. [1] GACODE ``bound_deriv.f90`` and the expro order of operations --
       differentiate on the grid, then interpolate -- as transcribed in
       :func:`vaft.code.gacode.tglf.bound_deriv` and verified against
       ``locpargen`` (``test/data/gacode/tglf_locpargen_48224``).
    .. [2] Issue #551: coordinate, gradient coordinate, reference length and
       convention are four separate roles.
    """
    profile_item = _coordinate(profile_coordinate)
    coordinate = profile_coordinate if coordinate is None else coordinate
    _coordinate(coordinate)

    resolved = None
    if convention is not None:
        resolved = resolve_convention(convention, metadata)
        for argument, value in (("gradient_coordinate", gradient_coordinate),
                                ("reference_length", reference_length)):
            if value is not UNSET and value != resolved[argument]:
                raise ValueError(
                    f"convention={convention!r} resolves {argument} to {resolved[argument]!r}, "
                    f"which contradicts the explicit {value!r}; pass one or the other")
        gradient_coordinate = resolved["gradient_coordinate"]
        reference_length = resolved["reference_length"]
    if gradient_coordinate is UNSET:
        gradient_coordinate = "r_minor"
    if reference_length is UNSET:
        raise ValueError(
            "state reference_length (None for the dimensional gradient, or one of "
            f"{', '.join(n for n in REFERENCE_LENGTHS if n)}) or a convention; it is never assumed")
    gradient_item = _coordinate(gradient_coordinate)
    if reference_length is not None and gradient_coordinate not in _OUTWARD_LENGTHS:
        raise ValueError(
            f"reference_length={reference_length!r} normalizes a gradient per metre of an "
            f"outward radius, and {gradient_coordinate} is not one "
            f"({'dimensionless' if gradient_item.unit != 'm' else 'not increasing outward'}); "
            f"use one of {', '.join(_OUTWARD_LENGTHS)} or reference_length=None")

    values = np.asarray(profile, dtype=float).reshape(-1)
    x_p = np.asarray(grid, dtype=float).reshape(-1)
    if values.size != x_p.size or values.size < 3:
        raise ValueError("profile and grid must be the same length, at least three points")
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(x_p))):
        raise ValueError("profile and grid must be finite")
    if np.any(values <= 0.0):
        raise ValueError("a logarithmic gradient needs a strictly positive profile")
    if not _strictly_monotonic(x_p):
        raise ValueError(f"the profile grid in {profile_coordinate} is not strictly monotonic")

    if isinstance(equilibrium, RadialCoordinateMap):
        cmap = equilibrium
        _check_time(cmap.time, time, "the coordinate map")
    else:
        cmap = radial_coordinate_map(equilibrium, time)

    # 2. on the profile's own grid
    dlnf_dxp = np.gradient(np.log(values), x_p, edge_order=2)

    # 3. the chain rule through the map
    if gradient_coordinate == profile_coordinate:
        dlnf_dxg = dlnf_dxp
        chain = "none (the profile is on the gradient coordinate)"
    else:
        what = f"mapping {profile_coordinate} -> {gradient_coordinate}"
        xp_map, xg_map = _on_map(cmap, profile_coordinate, gradient_coordinate, x_p, what)
        if not _strictly_monotonic(xg_map):
            raise ValueError(f"{what}: {gradient_coordinate} is not strictly monotonic in "
                             f"{profile_coordinate} on the map")
        xg_here = _map_relation(profile_coordinate, xp_map, xg_map, x_p)
        if not (np.all(np.isfinite(xg_here)) and _strictly_monotonic(xg_here)):
            raise ValueError(f"{what}: {gradient_coordinate} is not strictly monotonic in "
                             f"{profile_coordinate} on the profile grid")
        # Both derivatives in the variable the map is regular in: a radius and ln f go
        # as sqrt(psi_norm) near the axis, which a three-point rule in psi_norm cannot follow.
        u = _regular(profile_coordinate, x_p)
        dxg_du = np.gradient(xg_here, u, edge_order=2)
        flat = np.abs(dxg_du) <= 1e-9 * np.max(np.abs(dxg_du))
        if gradient_coordinate == "psi_norm":
            # psi_norm goes as the square of every other coordinate at the axis, so
            # d psi_norm/du vanishes there exactly -- whatever the discrete value is
            flat |= np.abs(xg_here) <= 1e-12
        if np.any(flat):
            raise ValueError(
                f"{what}: {gradient_coordinate} is stationary in {profile_coordinate} at "
                f"{profile_coordinate} = {x_p[flat][:3].tolist()} (a flux label against a radius "
                "at the axis), so the derivative there is 0/0; drop those points or use a "
                "geometric gradient coordinate")
        dlnf_dxg = np.gradient(np.log(values), u, edge_order=2) / dxg_du
        chain = (f"d/d{gradient_coordinate} = (d{profile_coordinate}/d{gradient_coordinate}) "
                 f"d/d{profile_coordinate}, both derivatives taken in "
                 f"{'sqrt(psi_norm)' if profile_coordinate == 'psi_norm' else profile_coordinate}")

    # 4. the reference length
    reference = resolve_reference_length(reference_length, cmap, metadata)
    if reference.name is None:
        scale = 1.0
    elif reference.name == "R_major_surface":
        xp_map, centre = _on_map(cmap, profile_coordinate, "r_center", x_p, "R_major_surface")
        scale = _map_relation(profile_coordinate, xp_map, centre, x_p)
    else:
        scale = float(reference.value)
    result = -scale * dlnf_dxg
    if not np.all(np.isfinite(result)):
        raise ValueError("the gradient is not finite at every point; nothing is filled in")

    # 5. the abscissa
    if coordinate == profile_coordinate:
        x_c = x_p
    else:
        xp_map, xc_map = _on_map(cmap, profile_coordinate, coordinate, x_p,
                                 f"mapping {profile_coordinate} -> {coordinate}")
        x_c = _map_relation(profile_coordinate, xp_map, xc_map, x_p)
    if at is not None:
        target = np.atleast_1d(np.asarray(at, dtype=float))
        if not _strictly_monotonic(x_c):
            raise ValueError(f"the result is not single-valued in {coordinate}; it cannot be interpolated")
        _check_support(target, x_c, f"interpolating to {coordinate}")
        result = _spline(x_c, result)(target)
        x_c = target

    if reference.name is None:
        unit = "1" if gradient_item.unit == "1" else "m^-1"
        definition = f"-d(log({source_quantity})) / d({gradient_coordinate})"
    else:
        unit = "1"
        definition = f"-{reference.symbol} * d(log({source_quantity})) / d({gradient_coordinate})"
    record = {
        "quantity": quantity or f"{source_quantity}_gradient",
        "source_quantity": source_quantity,
        "plot_coordinate": coordinate,
        "profile_coordinate": profile_item.name,
        "gradient_coordinate": gradient_coordinate,
        "reference_length": reference.record(),
        "requested_convention": convention,
        "resolved_convention": resolved,
        "mathematical_definition": definition,
        "unit": unit,
        "method": {
            "derivative": "three-point Lagrange on the profile grid (numpy.gradient, edge_order=2)",
            "chain_rule": chain,
            "map_interpolation": "cubic spline of the map relations onto the profile grid, in sqrt(psi_norm) for a psi_norm profile",
            "interpolation": "cubic spline (not-a-knot) to `at`, after differentiating",
        },
        "time": cmap.time,
        "equilibrium_source": cmap.source,
    }
    return ProfileGradient(np.asarray(result, dtype=float), np.asarray(x_c, dtype=float),
                           coordinate, unit, record)
