"""Records for analytic confinement-regime plasma states (#1045).

An analytic plasma state is a set of prescribed kinetic profiles -- electron
density, electron and ion temperature -- on the normalized poloidal flux
``psi_norm``, with the ion densities and the pressures derived from them.  It
is **not** an equilibrium: it carries no geometry, and a pedestal or an
internal transport barrier placed on a fixed magnetic geometry does not make
that geometry Grad-Shafranov consistent with the new pressure.  The
constructors and the projection onto ``(R, Z)`` live in
:mod:`vaft.process.profile`.

Units are the ones :mod:`vaft.data.kinetic_profiles` fixes for every kinetic
profile in the package: densities in m^-3, temperatures in eV, pressures in
Pa, and the radial coordinate dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .kinetic_profiles import KineticProfiles, PsiNormalization

#: Elementary charge [C], the eV-to-J factor of ``p = e n T``; the CODATA
#: value, as :data:`vaft.formula.constants.QE`.
_QE = 1.602176634e-19

#: Relative tolerance of the derived-quantity consistency checks.
_CONSISTENCY_RTOL = 1e-12


def _sealed(values) -> np.ndarray:
    """A read-only *copy*: a caller editing its own array cannot reach into the record."""
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _consistent(name: str, stored: np.ndarray, expected: np.ndarray) -> None:
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    if not np.all(np.abs(stored - expected) <= _CONSISTENCY_RTOL * scale):
        raise ValueError(
            f"{name} disagrees with the stored densities and temperatures; a state's pressures and "
            "ion densities are derived, not specified -- build it with "
            "vaft.process.profile.compose_plasma_state"
        )

__all__ = [
    "ANALYTIC_STATE_UNITS",
    "AnalyticPlasmaState",
    "AnalyticProfile",
    "BarrierStep",
]

#: Units of every array an :class:`AnalyticPlasmaState` carries.  The same set
#: as :data:`vaft.data.kinetic_profiles.KINETIC_UNITS`, so a state converts to a
#: :class:`~vaft.data.kinetic_profiles.KineticProfiles` without a factor.
ANALYTIC_STATE_UNITS: Mapping[str, str] = MappingProxyType({
    "psi_norm": "-",
    "n_e": "m^-3",
    "n_i": "m^-3",
    "n_impurity": "m^-3",
    "T_e": "eV",
    "T_i": "eV",
    "p_e": "Pa",
    "p_i": "Pa",
    "p_total": "Pa",
    "dp_e_dpsi_norm": "Pa",
    "dp_i_dpsi_norm": "Pa",
    "dp_total_dpsi_norm": "Pa",
})

#: The arrays of a state, in the order they are reported.
_STATE_ARRAYS = tuple(name for name in ANALYTIC_STATE_UNITS if name != "psi_norm")


@dataclass(frozen=True)
class BarrierStep:
    """One localized tanh step of a profile, in normalized poloidal flux.

    The step is Groebner's tanh with the full-width convention of
    :func:`vaft.formula.equilibrium.modified_tanh_profile` (``core_slope = 0``),
    rescaled so that it is exactly one on the magnetic axis and exactly zero
    at ``psi_norm = 1``.  ``position`` is the centre of the steep-gradient
    layer, ``width`` its *full* width (knee at ``position - width/2``, foot at
    ``position + width/2``), and ``height`` the *whole* step amplitude, from the
    separatrix side (``psi_norm = 1``, where the step is zero) to the axis side
    (``psi_norm = 0``, where it is ``height``), in the profile's unit.  The
    change between the knee and the foot is only ``tanh(1)``, about 0.76, of
    it for a well-localized layer.  An edge pedestal and
    an internal transport barrier are the same object at different positions.
    """

    position: float
    width: float
    height: float

    @property
    def knee(self) -> float:
        """Inner edge of the steep-gradient layer, ``position - width/2`` [-]."""
        return self.position - 0.5 * self.width

    @property
    def foot(self) -> float:
        """Outer edge of the steep-gradient layer, ``position + width/2`` [-]."""
        return self.position + 0.5 * self.width


@dataclass(frozen=True)
class AnalyticProfile:
    """One analytic kinetic profile: a smooth core plus up to two barrier steps.

    ``f(psi_N) = separatrix_value + core_amplitude * C(psi_N) + sum_b height_b * S_b(psi_N)``,
    where ``C = (1 - psi_N**core_alpha)**core_beta`` is the generalized
    parabolic core shape of :func:`vaft.formula.equilibrium.generalized_parabolic_profile`
    and each ``S_b`` is a normalized :class:`BarrierStep`.  ``axis_value`` and
    ``separatrix_value`` hold exactly; when a pedestal is present,
    ``pedestal_top_value`` holds exactly at the pedestal knee.  The amplitudes
    are what the constructor solved for, kept so the profile can be evaluated
    anywhere without re-solving.
    """

    quantity: str
    unit: str
    axis_value: float
    separatrix_value: float
    core_alpha: float
    core_beta: float
    core_amplitude: float
    pedestal: BarrierStep | None = None
    pedestal_top_value: float | None = None
    itb: BarrierStep | None = None
    coordinate: str = "psi_norm"

    @property
    def barriers(self) -> tuple[BarrierStep, ...]:
        """The barrier steps present, pedestal first [-]."""
        return tuple(step for step in (self.pedestal, self.itb) if step is not None)


@dataclass(frozen=True, eq=False)
class AnalyticPlasmaState:
    """Prescribed kinetic profiles and the pressures derived from them, on ``psi_norm``.

    ``n_e``, ``T_e`` and ``T_i`` are the prescribed channels, whose analytic
    definitions are in :attr:`profiles`.  Everything else is derived from them
    and from the declared composition, never specified independently: a
    hydrogenic main ion of density ``n_i`` and one fully stripped impurity of
    charge :attr:`impurity_charge` and density ``n_impurity`` with
    ``n_e = n_i + Z_I n_impurity`` and ``Z_eff n_e = n_i + Z_I**2 n_impurity``;
    ``p_e = e n_e T_e`` and ``p_i = e (n_i + n_impurity) T_i``, all ions at
    ``T_i``; ``p_total = p_e + p_i``.  Construction checks those relations
    on the stored arrays to 1e-12 relative, so a record whose pressures or ion
    densities were specified independently is refused.  The pressure
    gradients are the analytic derivatives against ``psi_norm`` (the product
    rule on the channels' analytic derivatives), in Pa per unit ``psi_norm``;
    they are not re-checked, since that would need the analytic profiles.
    ``psi_norm`` must be 1-D, strictly increasing and inside ``[0, 1]``, and
    every array is copied and sealed on construction.

    The coordinate is the normalized poloidal flux.  :attr:`rho_pol_norm` is
    ``sqrt(psi_norm)`` by definition; the toroidal-flux radius needs a safety
    factor and therefore an equilibrium, which this record does not carry.
    Nothing here is Grad-Shafranov consistent with any geometry the state is
    later projected onto.
    """

    label: str
    psi_norm: np.ndarray
    n_e: np.ndarray
    n_i: np.ndarray
    n_impurity: np.ndarray
    T_e: np.ndarray
    T_i: np.ndarray
    p_e: np.ndarray
    p_i: np.ndarray
    p_total: np.ndarray
    dp_e_dpsi_norm: np.ndarray
    dp_i_dpsi_norm: np.ndarray
    dp_total_dpsi_norm: np.ndarray
    profiles: Mapping[str, AnalyticProfile]
    z_eff: float = 1.0
    impurity_charge: float = 6.0
    coordinate: str = "psi_norm"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grid = _sealed(self.psi_norm)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError(f"psi_norm must be a non-empty 1-D grid, got shape {grid.shape}")
        if not np.all(np.isfinite(grid)) or grid[0] < 0.0 or grid[-1] > 1.0:
            raise ValueError("psi_norm must be finite and inside [0, 1]")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("psi_norm must be strictly increasing")
        object.__setattr__(self, "psi_norm", grid)
        for name in _STATE_ARRAYS:
            values = _sealed(getattr(self, name))
            if values.shape != grid.shape:
                raise ValueError(
                    f"{name} has shape {values.shape} but psi_norm has {grid.shape}; "
                    "every profile of a state is on its one coordinate"
                )
            object.__setattr__(self, name, values)
        z = float(self.impurity_charge)
        _consistent("n_i + Z n_impurity", self.n_i + z * self.n_impurity, self.n_e)
        _consistent("Z_eff", self.n_i + z * z * self.n_impurity, self.z_eff * self.n_e)
        _consistent("p_e", self.p_e, _QE * self.n_e * self.T_e)
        _consistent("p_i", self.p_i, _QE * (self.n_i + self.n_impurity) * self.T_i)
        _consistent("p_total", self.p_total, self.p_e + self.p_i)
        _consistent("dp_total_dpsi_norm", self.dp_total_dpsi_norm,
                    self.dp_e_dpsi_norm + self.dp_i_dpsi_norm)
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __len__(self) -> int:
        return int(self.psi_norm.size)

    @property
    def rho_pol_norm(self) -> np.ndarray:
        """``sqrt(psi_norm)``, the poloidal-flux radius [-]."""
        return np.sqrt(np.clip(self.psi_norm, 0.0, None))

    @property
    def dp_total_drho_pol_norm(self) -> np.ndarray:
        """``dp_total/d rho_pol_norm = 2 rho_pol_norm dp_total/dpsi_norm`` [Pa]."""
        return 2.0 * self.rho_pol_norm * self.dp_total_dpsi_norm

    def unit(self, name: str) -> str:
        """The unit of one of the state's arrays."""
        return ANALYTIC_STATE_UNITS[name]

    def to_kinetic_profiles(self) -> KineticProfiles:
        """The state as the package's kinetic-profile container, same units, same grid.

        ``n_impurity`` becomes ``n_z`` and ``T_i`` is also ``T_z``; the
        per-species pressures and the gradients go to ``extras``.  The
        coordinate is recorded as generated, not read, so nothing downstream
        mistakes it for a file's.
        """
        extras = {name: getattr(self, name) for name in
                  ("p_e", "p_i", "dp_e_dpsi_norm", "dp_i_dpsi_norm", "dp_total_dpsi_norm")}
        provenance = {name: f"analytic plasma state {self.label!r} (#1045)" for name in
                      ("n_e", "T_e", "T_i")}
        provenance.update({name: "derived from n_e, T_e, T_i and the declared composition"
                           for name in ("n_i", "n_z", "p_total", *extras)})
        return KineticProfiles(
            psi_norm=self.psi_norm,
            n_e=self.n_e,
            n_i=self.n_i,
            n_z=self.n_impurity,
            T_e=self.T_e,
            T_i=self.T_i,
            T_z=self.T_i,
            p_total=self.p_total,
            normalization=PsiNormalization(method="analytic", source="vaft.process.profile"),
            extras=extras,
            provenance=provenance,
            source=f"analytic plasma state {self.label!r}",
        )
