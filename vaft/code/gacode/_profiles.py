"""The typed VAFT-side representation of a GACODE profile set.

``input.gacode`` is the GACODE suite's shared profile spine: NEO, TGLF and
CGYRO all read it.  It is *not* VAFT's kinetic state.  The canonical state
stays in IMAS/OMAS ``equilibrium`` and ``core_profiles``, and this object is a
deterministic, provenance-preserving projection of it, so that one external
code family's conventions never leak back into the schema-facing layer.

Two rules the fields encode:

* **Optional means absent, not zero.**  Every field GACODE treats as optional is
  ``None`` when VAFT does not have it.  ``expro`` itself omits an all-zero
  profile when it writes, so a zero-filled array is indistinguishable from a
  physical zero once written -- which is exactly the confusion issue #550 asks
  this layer not to create.
* **Units are GACODE's, not SI.**  Densities are 10^19 m^-3, temperatures keV,
  currents MA and MA/m^2, conductivity MSiemens/m.  The conversion happens once,
  at the boundary, and is recorded in ``provenance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: How a field of a GACODEProfile came to hold what it holds.  Kept alongside
#: the values because "we measured this", "we derived it", "the machine policy
#: assumed it" and "the caller supplied it" are different scientific claims,
#: and a GACODE file cannot express the difference on its own.
PROVENANCE_KINDS = (
    "measured",
    "derived",
    "policy_assumption",
    "caller_supplied",
    "unavailable",
)

#: Shape harmonics ``expro`` carries beyond the elementary Miller set.
SHAPE_COS_FIELDS = tuple(f"shape_cos{index}" for index in range(7))
SHAPE_SIN_FIELDS = tuple(f"shape_sin{index}" for index in range(3, 7))

#: Volumetric source and sink terms.  Homogeneous family, carried as a mapping
#: rather than as thirty dataclass fields, because VAFT models none of them
#: individually yet and a round trip must still preserve them.
SOURCE_FIELDS = (
    "qohme", "qbeame", "qbeami", "qrfe", "qrfi", "qfuse", "qfusi",
    "qbrem", "qsync", "qline", "qei", "qione", "qioni", "qcxi",
    "qpar_beam", "qpar_wall", "qmom",
)


@dataclass
class GACODEProfile:
    """One GACODE profile set: geometry, species, kinetics and provenance.

    Attributes
    ----------
    rho
        Normalised square-root toroidal flux, the GACODE radial coordinate.
        This is ``sqrt(Phi/Phi_boundary)``, and it is **not** ``sqrt(psi_N)``.
    z
        Ion charge numbers, one per ion species; its length defines ``n_ion``.
    ni, ti, vpol, vtor
        Per-ion profiles, shaped ``(n_ion, n_exp)``.
    sources
        Volumetric source terms keyed by their GACODE tag.
    extra
        Tags read from a file that this class does not model, kept so that a
        read/write round trip loses nothing.
    provenance
        Per-field record, keyed by field name, whose ``kind`` is one of
        :data:`PROVENANCE_KINDS`.
    """

    # Radial coordinate and geometry
    rho: np.ndarray
    z: np.ndarray

    rmin: Optional[np.ndarray] = None
    polflux: Optional[np.ndarray] = None
    q: Optional[np.ndarray] = None
    w0: Optional[np.ndarray] = None
    rmaj: Optional[np.ndarray] = None
    zmag: Optional[np.ndarray] = None
    kappa: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None
    zeta: Optional[np.ndarray] = None
    shape: Mapping[str, np.ndarray] = field(default_factory=dict)

    # Species identity
    name: Sequence[str] = ()
    type: Sequence[str] = ()
    mass: Optional[np.ndarray] = None
    masse: float = 5.4488741e-04
    ze: float = -1.0

    # Global scalars
    shot: Optional[int] = None
    time: Optional[int] = None
    torfluxa: Optional[float] = None
    rcentr: Optional[float] = None
    bcentr: Optional[float] = None
    current: Optional[float] = None

    # Kinetic profiles
    ne: Optional[np.ndarray] = None
    ni: Optional[np.ndarray] = None
    te: Optional[np.ndarray] = None
    ti: Optional[np.ndarray] = None
    ptot: Optional[np.ndarray] = None
    z_eff: Optional[np.ndarray] = None
    vpol: Optional[np.ndarray] = None
    vtor: Optional[np.ndarray] = None

    # Current and conductivity
    fpol: Optional[np.ndarray] = None
    johm: Optional[np.ndarray] = None
    jbs: Optional[np.ndarray] = None
    jrf: Optional[np.ndarray] = None
    jnb: Optional[np.ndarray] = None
    jbstor: Optional[np.ndarray] = None
    sigmapar: Optional[np.ndarray] = None

    sources: Mapping[str, np.ndarray] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    #: Free-text header lines, in expro's fixed six-line order.
    header: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=float)
        self.z = np.atleast_1d(np.asarray(self.z, dtype=float))
        if self.rho.ndim != 1:
            raise ValueError(f"rho must be one-dimensional; got shape {self.rho.shape}")
        if self.rho.size < 2:
            raise ValueError("rho must have at least two points")
        if self.mass is not None:
            self.mass = np.atleast_1d(np.asarray(self.mass, dtype=float))
            if self.mass.size != self.z.size:
                raise ValueError(
                    f"mass has {self.mass.size} entries but z has {self.z.size}"
                )

    @property
    def n_exp(self) -> int:
        """Number of radial points."""
        return int(self.rho.size)

    @property
    def n_ion(self) -> int:
        """Number of ion species."""
        return int(self.z.size)

    def missing(self) -> tuple[str, ...]:
        """Names this profile records as unavailable, in provenance order."""
        return tuple(
            sorted(
                name
                for name, record in self.provenance.items()
                if record.get("kind") == "unavailable"
            )
        )

    def check_neo_requirements(self) -> tuple[str, ...]:
        """Fields NEO needs for ``PROFILE_MODEL=2`` that this profile lacks.

        A precondition check, not a verdict: it reports what is missing and
        leaves the decision to the caller, per the boundary in issue #253.
        """
        required = ("rmin", "polflux", "q", "rmaj", "ne", "te", "ni", "ti",
                    "torfluxa", "rcentr", "bcentr", "current")
        return tuple(name for name in required if getattr(self, name, None) is None)
