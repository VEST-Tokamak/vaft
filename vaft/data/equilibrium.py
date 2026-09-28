"""Portable, format-independent representations of axisymmetric equilibria.

These objects are deliberately small scientific working models.  They do not
replace GEQDSK, OMAS, or IMAS as persistence and interchange formats.
Numerical values use SI units unless the accompanying ``unit`` says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

# The findings model moved to the validation layer (issue #337): a report of
# objections is an assessment, and `vaft.validation.model` is where the one
# status vocabulary lives.  Re-exported here because this module and
# `vaft.data` have always been where callers imported it from.
from vaft.validation.model import ValidationIssue, ValidationReport


class Topology(str, Enum):
    """How the last closed flux surface is bounded.

    The primary distinction is diverted versus limited.  A plasma is diverted
    when a saddle point of psi is *relevant to the boundary*: its flux matches
    the boundary flux to within what the grid can resolve, and the confined
    region's level-set topology is consistent with a separatrix through it.
    ``UPPER_SINGLE_NULL``/``LOWER_SINGLE_NULL``/``DOUBLE_NULL`` refine that by
    where those X-points sit relative to the magnetic axis; ``DIVERTED`` is used
    when a boundary-relevant X-point exists but cannot be attributed to a branch.
    ``LIMITED`` means no boundary-relevant X-point exists and the LCFS is in
    contact with the wall.  ``AMBIGUOUS`` means the classification could not be
    made robustly -- typically a grid-clipped contour, insufficient resolution,
    or no wall against which to confirm a limited boundary.
    """

    LIMITED = "limited"
    LOWER_SINGLE_NULL = "lower_single_null"
    UPPER_SINGLE_NULL = "upper_single_null"
    DOUBLE_NULL = "double_null"
    DIVERTED = "diverted"
    AMBIGUOUS = "ambiguous"

    @property
    def is_diverted(self) -> bool:
        """True when a boundary-relevant X-point was identified."""
        return self in _DIVERTED_TOPOLOGIES

    @property
    def is_limited(self) -> bool:
        return self is Topology.LIMITED

    @property
    def is_determinate(self) -> bool:
        return self is not Topology.AMBIGUOUS


_DIVERTED_TOPOLOGIES = frozenset({
    Topology.LOWER_SINGLE_NULL, Topology.UPPER_SINGLE_NULL,
    Topology.DOUBLE_NULL, Topology.DIVERTED,
})


@dataclass(frozen=True)
class EquilibriumConvention:
    """Coordinate and sign convention, and the evidence used to identify it.

    Carries what an equilibrium's numbers *mean*, which the arrays themselves
    cannot say.  ``cocos`` is a declared index, 1 to 18, or ``None``;
    ``candidates`` are the indices identification left open; ``psi_per_radian``
    is the storage family, and it is the field that fixes the unit of every
    flux quantity on :class:`EquilibriumData`.  The three sign fields record
    what was observed for the plasma current, the vacuum field and the safety
    factor.  ``clockwise_phi`` is a fact about the machine, not the data, and
    is what distinguishes an odd index from its even partner.
    """

    cocos: int | None = None
    candidates: tuple[int, ...] = ()
    psi_per_radian: bool | None = None
    clockwise_phi: bool | None = None
    ip_sign: int | None = None
    bt_sign: int | None = None
    q_sign: int | None = None
    source: str = "unknown"
    identified: tuple[int, ...] = ()
    """Indices the observable signs and flux scale support, independently of
    whatever was declared.  Kept even when a declaration wins, so a declaration
    the data contradicts can be reported rather than silently trusted."""

    @property
    def ambiguous(self) -> bool:
        return self.cocos is None and len(self.candidates) != 1

    @property
    def contradicted(self) -> bool:
        """True when a declared index is not among the ones the data supports."""
        return bool(self.identified) and self.cocos is not None and self.cocos not in self.identified


@dataclass(frozen=True)
class DerivationProvenance:
    method: str
    source_type: str = "native"
    source_fields: tuple[str, ...] = ()
    source_time: float | None = None
    radial_coordinate: str | None = None
    interpolation: str | None = None
    fit_range: tuple[float, float] | None = None
    tolerances: Mapping[str, float] = field(default_factory=dict)
    convention: EquilibriumConvention | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DerivedValue:
    """A value together with its definition, provenance, and availability.

    A derived quantity that could not be formed is returned as one of these
    with ``value`` at ``None`` and ``reason`` saying why, rather than being
    absent or silently substituted.  ``unit`` and ``definition`` are the
    published description of the number; ``provenance`` records the method, the
    source fields read, the source time, and the convention it was computed
    under, so a value carries the COCOS that shaped it.
    """

    value: Any | None
    unit: str
    definition: str
    provenance: DerivationProvenance
    quality: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None and self.reason is None


@dataclass(frozen=True)
class Contour:
    r: np.ndarray
    z: np.ndarray
    closed: bool = True

    def __post_init__(self) -> None:
        r = np.asarray(self.r, dtype=float).reshape(-1)
        z = np.asarray(self.z, dtype=float).reshape(-1)
        if r.size != z.size:
            raise ValueError("contour r and z arrays must have equal length")
        if r.size and (not np.all(np.isfinite(r)) or not np.all(np.isfinite(z))):
            raise ValueError("contour coordinates must be finite")
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "z", z)

    @property
    def points(self) -> np.ndarray:
        return np.column_stack((self.r, self.z))


@dataclass(frozen=True)
class EquilibriumData:
    """One axisymmetric equilibrium, shape-normalized for numerical algorithms.

    The *shape* is normalized -- one grid layout, one profile layout, whatever
    the source was -- and the *convention* deliberately is not.  Nothing is
    converted to an internal standard on construction, so the fields below mean
    what the source meant by them.

    **The unit of every flux field is a property of :attr:`convention`, not of
    the field.**  ``psi``, ``psi_axis``, ``psi_boundary`` and ``psi_1d`` are in
    weber when ``convention.psi_per_radian`` is ``False`` (COCOS 11-18), in
    weber per radian when it is ``True`` (COCOS 1-8), and of unknown scale when
    it is ``None``.  ``pprime`` and ``ffprime`` are per that same flux unit.
    Code that forms a poloidal field from ``psi`` must consult the convention;
    a factor of ``2*pi`` in the field is a factor of ``(2*pi)**2`` in the
    poloidal beta.

    A default-constructed record carries no convention at all, and the field
    calculations then fall back to the historical weber-per-radian form, which
    is *not* :data:`vaft.data.cocos.VAFT_INTERNAL_COCOS`; that mismatch is
    tracked in #603.  Nothing enforces agreement between a psi array and the
    convention beside it, so a record rebuilt field-by-field can be made to lie.

    Units of the remaining fields: ``r``, ``z``, ``magnetic_axis``, ``lcfs``,
    ``limiter`` and ``r0`` in metres; ``pressure`` in pascal; ``f`` in
    tesla-metre; ``q`` dimensionless; ``ip`` in ampere; ``bt0`` in tesla;
    ``time`` in seconds.  ``psi`` is indexed ``(R, Z)``, matching ``r.size`` by
    ``z.size``.
    """

    r: np.ndarray | None = None
    z: np.ndarray | None = None
    psi: np.ndarray | None = None
    psi_axis: float | None = None
    psi_boundary: float | None = None
    magnetic_axis: tuple[float, float] | None = None
    lcfs: Contour | None = None
    limiter: Contour | None = None
    psi_1d: np.ndarray | None = None
    pressure: np.ndarray | None = None
    f: np.ndarray | None = None
    q: np.ndarray | None = None
    pprime: np.ndarray | None = None
    """dp/dpsi as the source carried it.  Kept rather than re-derived because it
    transforms by the ``PPRIME`` COCOS factor, which is the inverse of ``PSI``,
    and because differentiating pressure against psi cannot round-trip a file."""
    ffprime: np.ndarray | None = None
    """F dF/dpsi, transforming by ``F_FPRIME``; see :attr:`pprime`."""
    ip: float | None = None
    bt0: float | None = None
    r0: float | None = None
    time: float | None = None
    convention: EquilibriumConvention = field(default_factory=EquilibriumConvention)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("r", "z", "psi_1d", "pressure", "f", "q", "pprime", "ffprime"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, np.asarray(value, dtype=float).reshape(-1))
        if self.psi is not None:
            object.__setattr__(self, "psi", np.asarray(self.psi, dtype=float))


@dataclass(frozen=True)
class GlobalEquilibriumDescriptors:
    values: Mapping[str, DerivedValue]
    radial_coordinates: Mapping[str, DerivedValue] = field(default_factory=dict)
    rational_surfaces: Mapping[float, tuple[DerivedValue, ...]] = field(default_factory=dict)
    validation: ValidationReport = field(default_factory=ValidationReport)

    def __getitem__(self, name: str) -> DerivedValue:
        return self.values[name]


@dataclass(frozen=True)
class MillerSurface:
    r: float
    r0: float
    z0: float
    kappa: float
    delta: float
    #: Squareness.  Zero is the five-parameter Miller surface, so a caller
    #: that never sets it gets exactly the surface it got before (#867).
    zeta: float = 0.0
    radial_value: float | None = None
    radial_coordinate: str = "psi_n"
    d_r0_dr: float | None = None
    d_kappa_dr: float | None = None
    d_delta_dr: float | None = None
    q: float | None = None
    magnetic_shear: float | None = None
    alpha: float | None = None
    #: Inboard indentation, the bean-shaping coefficient of #941.  Last, so
    #: positional construction is unchanged; zero is the surface without it.
    indentation: float = 0.0


@dataclass(frozen=True)
class MillerFitResult:
    surface: MillerSurface
    contour: Contour
    reconstructed: Contour
    rms_error: float
    normalized_rms_error: float
    max_error: float
    hausdorff_distance: float
    converged: bool
    accepted: bool
    reason: str | None
    provenance: DerivationProvenance


@dataclass(frozen=True)
class MillerSequenceResult:
    fits: tuple[MillerFitResult, ...]
    derivative_reason: str | None = None
    provenance: DerivationProvenance | None = None


#: Homogeneous bases a :class:`SolovevEquilibrium` can be expanded in, with
#: their sizes.  ``"classic"`` is the original five-term up-down symmetric
#: basis; the two Cerfon-Freidberg bases add the higher even-Z terms (seven,
#: enough for a double-null separatrix) and the odd-Z terms (twelve, needed for
#: an up-down asymmetric single null) of Cerfon and Freidberg (2010).
SOLOVEV_BASIS_SIZES: Mapping[str, int] = {
    "classic": 5,
    "cerfon_freidberg_even": 7,
    "cerfon_freidberg": 12,
}

#: Linear functionals of psi a :class:`SolovevConstraint` can pin.
SOLOVEV_CONSTRAINT_KINDS = (
    "psi", "dpsi_dr", "dpsi_dz", "d2psi_dr2", "d2psi_dz2", "d2psi_drdz",
)


@dataclass(frozen=True)
class GuazzottoFreidbergEquilibrium:
    """An analytic Guazzotto-Freidberg (2021, Part 1) equilibrium in normalized form (#1148).

    ``psi(x, y) = sum_j coefficients[j] * Y_j(h_n y) * X_j(x)`` solves
    ``(1 + eps_hat x) psi_xx + psi_yy/(1 + eps**2) = -alpha**2 (1 + eps_hat nu x) psi``
    with ``psi = 1`` on the magnetic axis and ``0`` on the plasma surface, in
    ``R = R0 sqrt(1 + eps**2 + 2 eps x)``, ``Z = a y``.  Pressure and ``F**2``
    are quadratic in psi, so p, p' and J_phi vanish on the surface.  The
    eigenvalue ``alpha`` fixes the flux on axis once ``R0``, ``B0`` and ``beta0``
    (or ``q0``) are chosen; see :func:`guazzotto_freidberg_to_equilibrium`.

    Part 2 (#1149) adds an edge current pedestal ``current_pedestal`` (f_J),
    for which ``coefficients`` expand ``psi_J = psi + f_J/(1 - f_J)``;
    toroidal flow at axis Mach number ``mach_number`` with
    ``adiabatic_index`` 2 or inf, which changes the source to
    ``alpha**2 (1 + nu G(x))``; and the surface-current inputs
    ``pressure_pedestal`` (f_P) and ``bootstrap_fraction`` (f_B), which leave
    the interior flux unchanged and enter only the plasma parameters.  All
    default to zero, which is Part 1.
    """

    topology: str
    inverse_aspect_ratio: float
    nu: float
    alpha: float
    coefficients: np.ndarray
    separation_h: np.ndarray
    separation_k: np.ndarray
    magnetic_axis: tuple[float, float]
    elongation: float | None = None
    triangularity: float | None = None
    x_point_elongation: float | None = None
    x_point_triangularity: float | None = None
    eigen_residual: float = 0.0
    condition_number: float = float("nan")
    series_terms: int = 250
    status: str = "converged"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    current_pedestal: float = 0.0
    pressure_pedestal: float = 0.0
    bootstrap_fraction: float = 0.0
    mach_number: float = 0.0
    adiabatic_index: float | None = None


@dataclass(frozen=True)
class CurrentMomentRepresentation:
    """Toroidal current density reduced to its total, centroid and central moments (#943).

    ``central_moments[(p, q)]`` is the *normalized central* moment
    ``mu_pq = (1/I_p) * integral (R - R_c)**p (Z - Z_c)**q J_phi dA`` in
    m**(p+q), for ``2 <= p + q <= max_order``; ``mu_10 = mu_01 = 0`` by the
    centroid's definition and are not stored.  These describe the current
    distribution, not the boundary: ``mu_20`` is not a minor radius and a third
    moment is not a triangularity.
    """

    total_current: float
    centroid_r: float
    centroid_z: float
    central_moments: Mapping[tuple[int, int], float]
    max_order: int
    current_density_source: str
    provenance: DerivationProvenance

    @property
    def covariance(self) -> np.ndarray:
        """The second-order tensor ``[[mu_20, mu_11], [mu_11, mu_02]]`` [m^2]."""
        m = self.central_moments
        return np.array([[m[(2, 0)], m[(1, 1)]], [m[(1, 1)], m[(0, 2)]]], dtype=float)


@dataclass(frozen=True)
class FourierSurface:
    """A closed contour as a truncated Fourier series in a uniform arc-length angle (#945).

    ``R(theta) = sum_m r_cos[m] cos(m theta) + r_sin[m] sin(m theta)`` and the
    same for ``Z``, for ``m = 0 .. modes``.  ``theta = 2 pi s / L`` is arc
    length ``s`` counted counter-clockwise in the (R, Z) plane over the
    perimeter ``L``, with its origin where the first harmonic of ``R`` peaks,
    i.e. ``r_sin[1] = 0`` and ``r_cos[1] > 0``.  So the ``m = 0`` terms are the
    perimeter centroid, ``r_sin[0] = z_sin[0] = 0``, and an up-down symmetric
    surface -- racetracks included -- has ``r_sin = 0`` and ``z_cos[1:] = 0``.
    """

    r_cos: np.ndarray
    r_sin: np.ndarray
    z_cos: np.ndarray
    z_sin: np.ndarray
    radial_value: float | None = None
    radial_coordinate: str = "psi_n"
    angle_convention: str = "arc_length"

    def __post_init__(self) -> None:
        arrays = [np.asarray(getattr(self, name), dtype=float).reshape(-1) for name in ("r_cos", "r_sin", "z_cos", "z_sin")]
        if len({a.size for a in arrays}) != 1 or arrays[0].size < 2:
            raise ValueError("FourierSurface needs four coefficient arrays of one common length, at least two (m = 0, 1)")
        if arrays[1][0] != 0.0 or arrays[3][0] != 0.0:
            raise ValueError("the m = 0 sine coefficients must be zero")
        if self.angle_convention != "arc_length":
            raise ValueError(f"unsupported angle convention {self.angle_convention!r}; only 'arc_length' is defined")
        for name, value in zip(("r_cos", "r_sin", "z_cos", "z_sin"), arrays):
            object.__setattr__(self, name, value)

    @property
    def modes(self) -> int:
        """Highest poloidal harmonic kept."""
        return int(self.r_cos.size - 1)

    @property
    def reference_r(self) -> float:
        """The ``m = 0`` radial term: the perimeter centroid's major radius."""
        return float(self.r_cos[0])

    @property
    def reference_z(self) -> float:
        """The ``m = 0`` vertical term: the perimeter centroid's height."""
        return float(self.z_cos[0])


@dataclass(frozen=True)
class FourierFitResult:
    surface: FourierSurface
    contour: Contour
    reconstructed: Contour
    rms_error: float
    normalized_rms_error: float
    max_error: float
    hausdorff_distance: float
    accepted: bool
    reason: str | None
    provenance: DerivationProvenance


@dataclass(frozen=True)
class FourierSequenceResult:
    fits: tuple[FourierFitResult, ...]
    skipped: tuple[float, ...] = ()
    provenance: DerivationProvenance | None = None

    def coefficient(self, name: str, m: int, *, accepted_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(radial values, coefficient)`` of one family and harmonic across the sequence."""
        if name not in ("r_cos", "r_sin", "z_cos", "z_sin"):
            raise ValueError(f"unknown coefficient family {name!r}")
        items = [f for f in self.fits if f.accepted or not accepted_only]
        radial = np.array([f.surface.radial_value for f in items], dtype=float)
        values = np.array([getattr(f.surface, name)[m] for f in items], dtype=float)
        return radial, values


@dataclass(frozen=True)
class SolovevFit:
    """A Solov'ev model fitted to an existing equilibrium, with its fidelity (#1166).

    ``status`` is ``"accepted"``, ``"poor_fidelity"`` (a valid model that
    misses the equilibrium by more than the tolerance), ``"not_representable"``
    (the fitted flux has no closed boundary) or ``"failed"`` (no model).
    ``metrics`` holds only what was evaluated.
    """

    model: "SolovevEquilibrium | None"
    metrics: Mapping[str, Any]
    status: str
    reason: str | None
    provenance: DerivationProvenance


@dataclass(frozen=True)
class MXHChebyshevRepresentation:
    """Flux surfaces as MXH shapes with shifted-Chebyshev radial profiles (Xie & Li 2026; #1166).

    ``profiles[name] = (edge_value, coefficients)`` for ``h``, ``v``,
    ``kappa``, ``a``, ``c0`` and ``c1..cM``, ``s1..sM``; each profile is
    ``edge_value + sum_l coefficients[l] (1 - rho**2) T_l(2 rho**2 - 1)`` in
    ``rho = sqrt(psi_N)``.  ``status`` is ``"accepted"``, ``"poor_fidelity"``
    or ``"failed"`` (too few closed surfaces for the radial order).
    """

    r0: float
    z0: float
    harmonics: int
    radial_order: int
    profiles: Mapping[str, tuple[float, tuple[float, ...]]]
    parameter_count: int
    metrics: Mapping[str, float]
    status: str
    reason: str | None
    provenance: DerivationProvenance


@dataclass(frozen=True)
class GradShafranovResidualModes:
    """The Grad-Shafranov residual projected onto poloidal harmonics, surface by surface (#948).

    Row ``i`` is the surface at ``radial_values[i]``; ``cos[i, m]`` and
    ``sin[i, m]`` are the residual's harmonics in T/m with the arc-length angle
    of :class:`FourierSurface`, ``rms[i]`` its RMS on that surface, and
    ``scale`` the whole-plasma RMS of the source for normalization.
    """

    radial_values: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    rms: np.ndarray
    scale: float
    radial_coordinate: str = "psi_n"
    angle_convention: str = "arc_length"
    skipped: tuple[float, ...] = ()
    provenance: DerivationProvenance | None = None

    def amplitude(self, m: int) -> np.ndarray:
        """``sqrt(cos**2 + sin**2)`` of harmonic *m* on every surface [T/m]."""
        return np.hypot(self.cos[:, m], self.sin[:, m])


@dataclass(frozen=True)
class SolovevConstraint:
    """One linear condition on psi at a point: ``kind`` of psi equals ``value``.

    ``kind`` is one of :data:`SOLOVEV_CONSTRAINT_KINDS`, or ``"combination"``,
    in which case ``combination`` lists ``(kind, weight)`` pairs and the
    condition is ``sum(weight * kind(psi)) = value`` -- the form a boundary
    curvature condition takes.
    """

    r: float
    z: float
    kind: str
    value: float
    combination: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "combination":
            pairs = tuple((str(k), float(w)) for k, w in self.combination)
            if not pairs:
                raise ValueError("a combination constraint needs at least one (kind, weight) pair")
            if any(k not in SOLOVEV_CONSTRAINT_KINDS for k, _ in pairs):
                raise ValueError(f"combination kinds must be among {SOLOVEV_CONSTRAINT_KINDS}")
            object.__setattr__(self, "combination", pairs)
        elif self.combination:
            raise ValueError("combination is only meaningful with kind='combination'")


@dataclass(frozen=True)
class SolovevEquilibrium:
    coefficients: np.ndarray
    pprime: float
    ffprime: float
    rref: float
    psi_boundary: float = 0.0
    pressure_boundary: float = 0.0
    f_boundary: float = 1.0
    f_sign: int = 1
    rank: int | None = None
    residual_norm: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    basis: str = "classic"

    def __post_init__(self) -> None:
        if self.basis not in SOLOVEV_BASIS_SIZES:
            raise ValueError(f"basis must be one of {tuple(SOLOVEV_BASIS_SIZES)}, got {self.basis!r}")
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(-1)
        size = SOLOVEV_BASIS_SIZES[self.basis]
        if coefficients.size != size:
            raise ValueError(f"the {self.basis!r} Solovev basis requires {size} homogeneous coefficients, got {coefficients.size}")
        if self.rref <= 0:
            raise ValueError("rref must be positive")
        object.__setattr__(self, "coefficients", coefficients)


@dataclass(frozen=True)
class StationaryPoint:
    """A point where grad(psi) vanishes, classified by its Hessian.

    ``kind`` is ``"o"`` for an extremum (positive Hessian determinant, a
    magnetic axis candidate) and ``"x"`` for a saddle (negative determinant,
    an X-point candidate).  Being a saddle does not make a point a physical
    X-point; see :class:`Topology` for the boundary-relevance criteria.
    """

    r: float
    z: float
    psi: float
    psi_n: float
    kind: str
    hessian_determinant: float
    curvature: float = 0.0
    """Smaller Hessian eigenvalue magnitude, d2psi/dl2 along the flattest axis.

    Near a stationary point psi varies quadratically, so a flux offset dpsi
    displaces its level set by about ``sqrt(2*dpsi/curvature)``.  That is the
    scale on which a separatrix contour retreats from an X-point.
    """


@dataclass(frozen=True)
class XPoint:
    r: float
    z: float
    psi: float
    psi_n: float
    active: bool
    hessian_determinant: float


@dataclass(frozen=True)
class Gap:
    name: str
    angle: float
    distance: DerivedValue
    plasma_point: tuple[float, float] | None = None
    wall_point: tuple[float, float] | None = None


@dataclass(frozen=True)
class StrikePoint:
    r: float
    z: float
    branch: str
    flux_expansion: DerivedValue
    incidence_angle: DerivedValue


@dataclass(frozen=True)
class BoundaryRepresentation:
    lcfs: Contour | None
    limiter: Contour | None
    x_points: tuple[XPoint, ...]
    topology: Topology
    d_r_sep: DerivedValue
    gaps: tuple[Gap, ...]
    strike_points: tuple[StrikePoint, ...]
    fourier_coefficients: Mapping[str, np.ndarray]
    fourier_reconstruction_error: DerivedValue
    provenance: DerivationProvenance
    reason: str | None = None
    stationary_points: tuple[StationaryPoint, ...] = ()
    wall_contact_distance: DerivedValue | None = None


__all__ = [
    "BoundaryRepresentation", "Contour", "DerivationProvenance", "DerivedValue",
    "EquilibriumConvention", "EquilibriumData", "Gap", "GlobalEquilibriumDescriptors",
    "MillerFitResult", "MillerSequenceResult", "MillerSurface", "SolovevConstraint",
    "SolovevEquilibrium", "StationaryPoint", "StrikePoint", "Topology", "ValidationIssue",
    "ValidationReport", "XPoint", "CurrentMomentRepresentation", "GradShafranovResidualModes", "FourierFitResult",
    "GuazzottoFreidbergEquilibrium", "FourierSequenceResult", "FourierSurface", "MXHChebyshevRepresentation",
    "SolovevFit", "SOLOVEV_BASIS_SIZES", "SOLOVEV_CONSTRAINT_KINDS",
]
