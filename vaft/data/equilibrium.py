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
    """Coordinate/sign convention and the evidence used to identify it."""

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
    """A value together with its definition, provenance, and availability."""

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
class ValidationIssue:
    severity: str
    code: str
    field: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)


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
    """One axisymmetric equilibrium normalized for numerical algorithms."""

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
    ip: float | None = None
    bt0: float | None = None
    r0: float | None = None
    time: float | None = None
    convention: EquilibriumConvention = field(default_factory=EquilibriumConvention)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("r", "z", "psi_1d", "pressure", "f", "q"):
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
    radial_value: float | None = None
    radial_coordinate: str = "psi_n"
    d_r0_dr: float | None = None
    d_kappa_dr: float | None = None
    d_delta_dr: float | None = None
    q: float | None = None
    magnetic_shear: float | None = None
    alpha: float | None = None


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


@dataclass(frozen=True)
class SolovevConstraint:
    r: float
    z: float
    kind: str
    value: float


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

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(-1)
        if coefficients.size != 5:
            raise ValueError("SolovevEquilibrium requires five homogeneous coefficients")
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
    "ValidationReport", "XPoint",
]
