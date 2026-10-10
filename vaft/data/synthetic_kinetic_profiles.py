"""Records for synthetic kinetic profiles generated from a magnetic equilibrium (#122).

An equilibrium fixes ``p(psi)`` but not how it splits into densities and
temperatures.  These records keep the four kinds of input the generator in
:mod:`vaft.process.profile` needs apart -- profile *shapes*, scalar
*normalizations*, *temperature* assumptions and the ion *composition* -- and
the result keeps what was assumed apart from what was derived, together with
every residual the construction was checked against.

The fidelity ladder the records implement (issue #122, Levels 0-3):

* **Level 0** -- :class:`SqrtPressureSplit`: ``n_e`` and ``T_e`` share the
  shape ``sqrt(p_eq/p_eq(0))``, the legacy
  :func:`vaft.process.profile.core_profiles_from_eq` closure.
* **Level 1** -- a :class:`ProfileSpec` whose shape is an
  :class:`~vaft.data.analytic_plasma_state.AnalyticProfile` (the #1045/#552
  kernels) or a :class:`TabulatedProfile`, used as given.
* **Level 2** -- the same shapes normalized by a :class:`ScalarTarget`
  (axis, separatrix, line- or volume-averaged value, Greenwald fraction), a
  peaking factor, a ``T_i/T_e`` ratio or an electron pressure fraction.
* **Level 3** -- a :class:`GradientProfile`: a prescribed ``a/L_f``
  integrated from a declared boundary value.

Every level is an *assumption-driven* profile.  None is measured, fitted or
transport-predicted, and the result says so in its provenance.

Units are those of :data:`vaft.data.kinetic_profiles.KINETIC_UNITS`:
densities in m^-3, temperatures in eV, pressures in Pa, volume in m^3, the
radial coordinates dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .analytic_plasma_state import AnalyticPlasmaState, AnalyticProfile
from .atomic import ATOMIC_NUMBERS as _ATOMIC_NUMBERS
from .atomic import STANDARD_ATOMIC_WEIGHTS as _ATOMIC_WEIGHTS
from .kinetic_profiles import KineticProfiles, PsiNormalization, Species

__all__ = [
    "ION_SPECIES",
    "PRESSURE_CLOSURES",
    "PRESSURE_CONSTRAINTS",
    "SYNTHETIC_PROFILE_STATUSES",
    "SYNTHETIC_PROFILE_UNITS",
    "SCALAR_TARGET_KINDS",
    "Composition",
    "GradientProfile",
    "IonSpecies",
    "PressureClosureReport",
    "ProfileSpec",
    "ScalarTarget",
    "SqrtPressureSplit",
    "SyntheticKineticProfiles",
    "SyntheticKineticSpec",
    "SyntheticProfileError",
    "TabulatedProfile",
    "TargetResidual",
    "TemperatureAssumption",
]

#: Elementary charge [C], the eV-to-J factor of ``p = e n T`` (CODATA, as
#: :data:`vaft.formula.constants.QE`).
_QE = 1.602176634e-19

#: Relative tolerance of the stored-array consistency checks.
_CONSISTENCY_RTOL = 1e-12

#: Nuclear charge and standard atomic weight [u] of the species a composition
#: may name: a view of :data:`vaft.data.atomic.ATOMIC_NUMBERS` and
#: :data:`vaft.data.atomic.STANDARD_ATOMIC_WEIGHTS` (IUPAC 2021 abridged,
#: isotope masses for D and T), which own the values.  Vocabulary, not a
#: machine setting.
ION_SPECIES: Mapping[str, tuple[int, float]] = MappingProxyType({
    symbol: (_ATOMIC_NUMBERS[symbol], _ATOMIC_WEIGHTS[symbol]) for symbol in _ATOMIC_NUMBERS
})

#: The scalar normalizations a profile may be scaled to.
SCALAR_TARGET_KINDS = ("axis", "separatrix", "line_average", "volume_average", "greenwald_fraction")

#: What the kinetic pressure is held to.
PRESSURE_CONSTRAINTS = ("equilibrium", "thermal_energy", "kinetic")

#: Which kinetic variable a pressure constraint solves for.
PRESSURE_CLOSURES = {
    "equilibrium": ("temperature", "density", "sqrt_split"),
    "thermal_energy": ("temperature_amplitude", "density_amplitude"),
    "kinetic": (None,),
}

#: Outcome of a generation.  ``invalid_*`` and ``coordinate_mapping_failed``
#: are raised as :class:`SyntheticProfileError` before anything is built; the
#: others are carried on a returned :class:`SyntheticKineticProfiles`.
SYNTHETIC_PROFILE_STATUSES = (
    "success",
    "invalid_equilibrium",
    "invalid_profile_model",
    "invalid_gradient_model",
    "invalid_normalization",
    "invalid_composition",
    "constraint_not_reached",
    "pressure_closure_failed",
    "quasineutrality_failed",
    "zeff_constraint_failed",
    "coordinate_mapping_failed",
    "numerically_suspect",
)

#: Units of every array a :class:`SyntheticKineticProfiles` carries.
SYNTHETIC_PROFILE_UNITS: Mapping[str, str] = MappingProxyType({
    "psi_norm": "-", "rho_pol_norm": "-", "rho_tor_norm": "-", "psi": "Wb", "volume": "m^3",
    "n_e": "m^-3", "n_i": "m^-3", "n_impurity": "m^-3", "T_e": "eV", "T_i": "eV",
    "z_eff": "-", "p_e": "Pa", "p_i": "Pa", "p_total": "Pa", "p_eq": "Pa",
})

_COORDINATES = ("psi_norm", "rho_pol_norm", "rho_tor_norm")


class SyntheticProfileError(ValueError):
    """An assumption set the generator refuses, with the status that names why.

    ``status`` is one of :data:`SYNTHETIC_PROFILE_STATUSES`.  A subclass of
    :class:`ValueError`, so a caller catching the package's usual refusal
    catches this one too.
    """

    def __init__(self, status: str, message: str):
        if status not in SYNTHETIC_PROFILE_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        super().__init__(f"{status}: {message}")
        self.status = status


def _sealed(values) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _finite_scalar(name: str, value, status: str) -> float:
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise SyntheticProfileError(status, f"{name} must be a finite scalar, got {value!r}")
    return float(value)


# --- profile shapes ------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TabulatedProfile:
    """A user-supplied profile on a declared radial coordinate.

    ``x`` is strictly increasing in ``coordinate`` (``psi_norm``,
    ``rho_pol_norm`` or ``rho_tor_norm``), ``values`` are in ``unit``.
    ``interpolation`` is ``"pchip"`` (shape-preserving, no overshoot) or
    ``"linear"``.  ``extrapolation`` is ``"refuse"`` -- the grid must lie
    inside ``[x[0], x[-1]]`` -- or ``"hold"``, which continues the end values
    flat and is recorded as such.  The profile is dimensional; a
    :class:`ScalarTarget` rescales it without changing its shape.
    """

    x: np.ndarray
    values: np.ndarray
    coordinate: str = "rho_tor_norm"
    interpolation: str = "pchip"
    extrapolation: str = "refuse"
    unit: str = ""

    def __post_init__(self) -> None:
        x, v = _sealed(self.x), _sealed(self.values)
        if self.coordinate not in _COORDINATES:
            raise SyntheticProfileError("invalid_profile_model",
                                        f"coordinate must be one of {_COORDINATES}, got {self.coordinate!r}")
        if x.ndim != 1 or x.shape != v.shape or x.size < 2:
            raise SyntheticProfileError("invalid_profile_model",
                                        "x and values must be 1-D, of equal length and at least 2 points")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
            raise SyntheticProfileError("invalid_profile_model", "x and values must be finite")
        if np.any(np.diff(x) <= 0.0) or x[0] < 0.0 or x[-1] > 1.0:
            raise SyntheticProfileError("invalid_profile_model",
                                        "x must be strictly increasing inside [0, 1]")
        if self.interpolation not in ("pchip", "linear"):
            raise SyntheticProfileError("invalid_profile_model",
                                        f"interpolation must be 'pchip' or 'linear', got {self.interpolation!r}")
        if self.extrapolation not in ("refuse", "hold"):
            raise SyntheticProfileError("invalid_profile_model",
                                        f"extrapolation must be 'refuse' or 'hold', got {self.extrapolation!r}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "values", v)


@dataclass(frozen=True, eq=False)
class GradientProfile:
    r"""A profile defined by its normalized logarithmic gradient (Level 3).

    $a/L_f(x) = -\,d\ln f/dx$ is piecewise linear between the knots ``x``
    (which must run from exactly 0 to exactly 1 in ``coordinate``), and
    $\ln f(x) = \ln f_b + \int_x^{x_b} a/L_f\,ds$ with $f_b$ =
    ``boundary_value`` at $x_b$ = ``boundary_position``; the integral of the
    piecewise-linear gradient is taken exactly.  ``x`` is the declared
    normalized radius and plays the role of $r/a$: ``a`` here is the
    normalization of that coordinate, not a geometric minor radius.  A
    :class:`ScalarTarget` rescales ``f`` and leaves every ``a/L_f`` unchanged.

    The gradient is *prescribed*.  A profile built from it is an assumption,
    never a transport prediction, however the numbers were chosen.
    """

    x: np.ndarray
    a_over_L: np.ndarray
    boundary_value: float
    boundary_position: float = 1.0
    coordinate: str = "rho_tor_norm"
    unit: str = ""

    def __post_init__(self) -> None:
        x, g = _sealed(self.x), _sealed(self.a_over_L)
        if self.coordinate not in _COORDINATES:
            raise SyntheticProfileError("invalid_gradient_model",
                                        f"coordinate must be one of {_COORDINATES}, got {self.coordinate!r}")
        if x.ndim != 1 or x.shape != g.shape or x.size < 2:
            raise SyntheticProfileError("invalid_gradient_model",
                                        "x and a_over_L must be 1-D, of equal length and at least 2 knots")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(g))):
            raise SyntheticProfileError("invalid_gradient_model", "x and a_over_L must be finite")
        if np.any(np.diff(x) <= 0.0) or x[0] != 0.0 or x[-1] != 1.0:
            raise SyntheticProfileError("invalid_gradient_model",
                                        "the knots must be strictly increasing from exactly 0 to exactly 1")
        value = _finite_scalar("boundary_value", self.boundary_value, "invalid_gradient_model")
        if value <= 0.0:
            raise SyntheticProfileError("invalid_gradient_model",
                                        f"boundary_value must be positive (a log-gradient needs f > 0), got {value!r}")
        position = _finite_scalar("boundary_position", self.boundary_position, "invalid_gradient_model")
        if not 0.0 <= position <= 1.0:
            raise SyntheticProfileError("invalid_gradient_model",
                                        f"boundary_position must lie in [0, 1], got {position!r}")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "a_over_L", g)
        object.__setattr__(self, "boundary_value", value)
        object.__setattr__(self, "boundary_position", position)

    def gradient(self, x) -> np.ndarray:
        """``a/L_f`` at ``x`` by linear interpolation between the knots [-]."""
        return np.interp(np.asarray(x, dtype=float), self.x, self.a_over_L)

    def integral(self, x) -> np.ndarray:
        """``int_0^x a/L_f ds``, exact for the piecewise-linear gradient [-]."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        knots, g = self.x, self.a_over_L
        cumulative = np.r_[0.0, np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(knots))]
        k = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, knots.size - 2)
        dx = x - knots[k]
        slope = (g[k + 1] - g[k]) / (knots[k + 1] - knots[k])
        return cumulative[k] + g[k] * dx + 0.5 * slope * dx * dx


@dataclass(frozen=True)
class ScalarTarget:
    """One scalar a profile is normalized to by scaling its amplitude.

    ``kind`` is one of :data:`SCALAR_TARGET_KINDS`:

    * ``"axis"`` -- the value at ``psi_norm = 0``;
    * ``"separatrix"`` -- the value at ``psi_norm = 1``;
    * ``"line_average"`` -- ``int f dR / int dR`` along the horizontal chord
      through the magnetic axis, between its two LCFS crossings;
    * ``"volume_average"`` -- ``int f dV / V`` with ``V(psi)`` the enclosed
      volume of the traced flux surfaces;
    * ``"greenwald_fraction"`` -- the line average above divided by
      ``n_G = I_p/(pi a^2)`` [1e20 m^-3, MA, m], with ``a`` half the radial
      extent of the LCFS; electron density only.

    Scaling never changes the shape; a target is met by normalization, and
    the achieved value is recomputed from the final profile.
    """

    kind: str
    value: float

    def __post_init__(self) -> None:
        if self.kind not in SCALAR_TARGET_KINDS:
            raise SyntheticProfileError("invalid_normalization",
                                        f"kind must be one of {SCALAR_TARGET_KINDS}, got {self.kind!r}")
        value = _finite_scalar(f"{self.kind} target", self.value, "invalid_normalization")
        if value <= 0.0:
            raise SyntheticProfileError("invalid_normalization",
                                        f"a {self.kind} target must be positive, got {value!r}")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ProfileSpec:
    """One kinetic channel: a shape, and optionally what normalizes it.

    ``shape`` is an :class:`~vaft.data.analytic_plasma_state.AnalyticProfile`
    (Level 1, the #1045 composition of #552 kernels, in ``psi_norm``), a
    :class:`TabulatedProfile` (Level 1) or a :class:`GradientProfile`
    (Level 3).  ``target`` scales the amplitude (Level 2).  ``peaking_factor``
    -- ``f(axis)/<f>_V`` -- is met by solving the core peaking exponent
    ``core_beta`` of an analytic shape, keeping its axis, pedestal-top and
    separatrix values in proportion; it is refused for the other shapes.
    The exponent is searched in ``[1, 40]`` (below one the core gradient is
    infinite at an endpoint), so a peaking factor below the one the shape has
    at ``core_beta = 1`` -- or above the one at 40 -- is unreachable by design
    and comes back ``not_reached`` with status ``constraint_not_reached``.
    A shape's channel is checked: an ``AnalyticProfile`` must be of the slot's
    quantity and unit, a tabulated or gradient shape in the slot's unit or
    unitless.
    """

    shape: Any
    target: ScalarTarget | None = None
    peaking_factor: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (AnalyticProfile, TabulatedProfile, GradientProfile)):
            raise SyntheticProfileError(
                "invalid_profile_model",
                f"shape must be an AnalyticProfile, TabulatedProfile or GradientProfile, got {type(self.shape).__name__}")
        if self.target is not None and not isinstance(self.target, ScalarTarget):
            raise SyntheticProfileError("invalid_normalization", "target must be a ScalarTarget")
        if self.peaking_factor is not None:
            peaking = _finite_scalar("peaking_factor", self.peaking_factor, "invalid_normalization")
            if not isinstance(self.shape, AnalyticProfile):
                raise SyntheticProfileError(
                    "invalid_normalization",
                    "a peaking factor is solved through the core exponent of an AnalyticProfile; "
                    "a tabulated or gradient shape has no such free parameter")
            if peaking <= 1.0:
                raise SyntheticProfileError("invalid_normalization",
                                            f"peaking_factor = axis / volume average must exceed 1, got {peaking!r}")
            object.__setattr__(self, "peaking_factor", peaking)


@dataclass(frozen=True)
class TemperatureAssumption:
    """How the ion temperature follows, exactly one route of three.

    * ``ti_over_te`` -- ``T_i = r T_e`` with ``r`` a positive scalar or a
      :class:`TabulatedProfile` of the ratio;
    * ``T_i`` -- an independent :class:`ProfileSpec`;
    * ``electron_pressure_fraction`` -- ``p_e/(p_e + p_i)`` in ``(0, 1)``,
      held locally, which fixes ``T_i`` from ``T_e`` and the composition.

    ``source`` is recorded in provenance, e.g. ``"assumed"`` or the machine
    policy a statistical ratio came from; a ratio stays an assumption whatever
    its source, and is never reported as a measured ``T_i``.
    """

    ti_over_te: Any = None
    T_i: ProfileSpec | None = None
    electron_pressure_fraction: float | None = None
    source: str = "assumed"

    def __post_init__(self) -> None:
        given = [v is not None for v in (self.ti_over_te, self.T_i, self.electron_pressure_fraction)]
        if sum(given) != 1:
            raise SyntheticProfileError(
                "invalid_profile_model",
                "give exactly one of ti_over_te, T_i and electron_pressure_fraction; "
                "the ion temperature is never defaulted")
        if self.ti_over_te is not None and not isinstance(self.ti_over_te, TabulatedProfile):
            ratio = _finite_scalar("ti_over_te", self.ti_over_te, "invalid_profile_model")
            if ratio <= 0.0:
                raise SyntheticProfileError("invalid_profile_model", f"ti_over_te must be positive, got {ratio!r}")
            object.__setattr__(self, "ti_over_te", ratio)
        if isinstance(self.ti_over_te, TabulatedProfile):
            if np.any(self.ti_over_te.values <= 0.0):
                raise SyntheticProfileError("invalid_profile_model", "a tabulated ti_over_te must be positive")
            if self.ti_over_te.unit not in ("", "-"):
                raise SyntheticProfileError("invalid_profile_model",
                                            f"ti_over_te is dimensionless, got unit {self.ti_over_te.unit!r}")
        if self.T_i is not None and not isinstance(self.T_i, ProfileSpec):
            raise SyntheticProfileError("invalid_profile_model", "T_i must be a ProfileSpec")
        if self.electron_pressure_fraction is not None:
            fraction = _finite_scalar("electron_pressure_fraction", self.electron_pressure_fraction,
                                      "invalid_normalization")
            if not 0.0 < fraction < 1.0:
                raise SyntheticProfileError("invalid_normalization",
                                            f"electron_pressure_fraction must lie in (0, 1), got {fraction!r}")
            object.__setattr__(self, "electron_pressure_fraction", fraction)

    @property
    def route(self) -> str:
        """``"ratio"``, ``"profile"`` or ``"partition"`` [-]."""
        if self.ti_over_te is not None:
            return "ratio"
        return "profile" if self.T_i is not None else "partition"


@dataclass(frozen=True)
class Composition:
    """A hydrogenic main ion and at most one fully stripped impurity.

    ``z_eff`` is the *local* effective charge, uniform in radius:
    ``Z_eff n_e = n_i + Z_I^2 n_I`` at every point, with quasi-neutrality
    ``n_e = n_i + Z_I n_I``.  Those two fix both ion densities
    (:func:`vaft.formula.atomic.impurity_fraction_from_effective_charge`).
    ``z_eff > 1`` needs an impurity; ``impurity_charge`` defaults to the
    impurity's nuclear charge (fully stripped).  Species are those of
    :data:`ION_SPECIES`.
    """

    main_ion: str = "H"
    impurity: str | None = None
    z_eff: float = 1.0
    impurity_charge: float | None = None

    def __post_init__(self) -> None:
        if self.main_ion not in ("H", "D", "T"):
            raise SyntheticProfileError(
                "invalid_composition",
                f"the main ion must be hydrogenic (H, D or T), got {self.main_ion!r}; the "
                "quasi-neutrality/Z_eff closure assumes a Z = 1 main ion")
        z_eff = _finite_scalar("z_eff", self.z_eff, "invalid_composition")
        object.__setattr__(self, "z_eff", z_eff)
        if self.impurity is None:
            if self.impurity_charge is not None:
                raise SyntheticProfileError("invalid_composition", "impurity_charge given without an impurity")
            if z_eff != 1.0:
                raise SyntheticProfileError(
                    "invalid_composition",
                    f"z_eff = {z_eff!r} needs an impurity species; a pure hydrogenic plasma has z_eff = 1")
            return
        if self.impurity not in ION_SPECIES or ION_SPECIES[self.impurity][0] < 2:
            raise SyntheticProfileError(
                "invalid_composition",
                f"unknown or hydrogenic impurity {self.impurity!r}; choose one of "
                f"{sorted(k for k, v in ION_SPECIES.items() if v[0] > 1)}")
        charge = ION_SPECIES[self.impurity][0] if self.impurity_charge is None else self.impurity_charge
        charge = _finite_scalar("impurity_charge", charge, "invalid_composition")
        if not 1.0 < charge <= ION_SPECIES[self.impurity][0]:
            raise SyntheticProfileError(
                "invalid_composition",
                f"impurity_charge must lie in (1, {ION_SPECIES[self.impurity][0]}] for {self.impurity}, got {charge!r}")
        if not 1.0 <= z_eff <= charge:
            raise SyntheticProfileError(
                "invalid_composition",
                f"Z_eff = {z_eff!r} is unreachable with {self.impurity}{charge:g}+ in a hydrogenic plasma: "
                f"it must lie in [1, {charge:g}]")
        object.__setattr__(self, "impurity_charge", charge)


@dataclass(frozen=True)
class SqrtPressureSplit:
    """The Level 0 amplitude: exactly one of an axis ``T_e`` or a fixed ``n_e/T_e``.

    With ``k = p/(e n_e T_e)`` fixed by the composition and ``T_i/T_e``
    (``k = 2`` for ``n_i = n_e``, ``T_i = T_e``), ``T_e = T_e0 g`` and
    ``n_e = p(0)/(k e T_e0) g`` with ``g = sqrt(p/p(0))``, or
    ``T_e = sqrt(p/(k e C))`` and ``n_e = C T_e``: the legacy
    :func:`vaft.process.profile.core_profiles_from_eq` and
    :func:`~vaft.process.profile.core_profiles_from_eq_ratio` closures.
    """

    te_axis: float | None = None
    ne_over_te: float | None = None

    def __post_init__(self) -> None:
        if (self.te_axis is None) == (self.ne_over_te is None):
            raise SyntheticProfileError("invalid_normalization", "give exactly one of te_axis and ne_over_te")
        for name in ("te_axis", "ne_over_te"):
            value = getattr(self, name)
            if value is not None:
                value = _finite_scalar(name, value, "invalid_normalization")
                if value <= 0.0:
                    raise SyntheticProfileError("invalid_normalization", f"{name} must be positive, got {value!r}")
                object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class SyntheticKineticSpec:
    """Everything the generator is told, separate from the equilibrium it is applied to.

    The same spec can be applied to successive equilibria (the #123
    iteration) and gives a deterministic result for each.

    ``pressure_constraint`` is one of :data:`PRESSURE_CONSTRAINTS` and
    ``closure`` one of :data:`PRESSURE_CLOSURES` for it:

    * ``"equilibrium"`` -- ``p_kin = p_eq`` locally.  ``closure`` names the
      solved variable: ``"temperature"`` solves ``T_e(psi)`` with ``n_e``
      and the ``T_i`` route held (``T_e`` must not be given);
      ``"density"`` solves ``n_e(psi)`` with ``T_e`` held (``n_e`` must not
      be given); ``"sqrt_split"`` is Level 0 and needs :attr:`sqrt_split`.
    * ``"thermal_energy"`` -- a partial constraint: one amplitude
      (``"temperature_amplitude"`` scales ``T_e`` and ``T_i`` together,
      ``"density_amplitude"`` scales ``n_e``) makes
      ``W = 3/2 int p dV`` equal the equilibrium's; the local profile is
      reported, not forced.
    * ``"kinetic"`` -- every channel as specified; the pressure mismatch is
      reported.  ``closure`` must be ``None``.

    ``psi_norm`` is the output grid, strictly increasing from 0 to 1;
    ``None`` is 101 points uniform in ``rho_pol_norm``.

    **Edge treatment of a local closure.**  Where ``p_eq`` vanishes at the
    separatrix a local closure divides a vanishing pressure among held
    profiles that do not vanish, and would drive the solved channel to zero
    or below.  The *edge region* is every grid point beyond the leading run
    of points with ``p_eq >= edge_floor * max(p_eq)``; its inner end is
    reported as ``closure_exact_up_to_psi_n``.  Inside that point the closure
    is exact; beyond it the solved channel (``T_e`` for ``"temperature"``,
    ``n_e`` for ``"density"``) is carried linearly in ``psi_norm`` from its
    last exact value to ``edge_value`` at the separatrix, or held flat at the
    last exact value when ``edge_value`` is ``None``, and the local pressure
    residual there is reported separately, never hidden.  ``edge_floor`` is
    also the floor of the region over which every pressure residual is
    judged.  A solved channel that is non-positive *inside* the exact region
    still fails the closure.  ``edge_value`` is in the solved channel's unit
    (eV or m^-3) and belongs only to those two closures.
    """

    temperature: TemperatureAssumption
    composition: Composition = field(default_factory=Composition)
    n_e: ProfileSpec | None = None
    T_e: ProfileSpec | None = None
    pressure_constraint: str = "kinetic"
    closure: str | None = None
    sqrt_split: SqrtPressureSplit | None = None
    psi_norm: np.ndarray | None = None
    label: str = "synthetic"
    edge_floor: float = 1e-2
    edge_value: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, TemperatureAssumption):
            raise SyntheticProfileError("invalid_profile_model", "temperature must be a TemperatureAssumption")
        if not isinstance(self.composition, Composition):
            raise SyntheticProfileError("invalid_composition", "composition must be a Composition")
        for name in ("n_e", "T_e"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ProfileSpec):
                raise SyntheticProfileError("invalid_profile_model", f"{name} must be a ProfileSpec")
            if value is not None:
                _check_channel(name, value)
        if self.temperature.T_i is not None:
            _check_channel("T_i", self.temperature.T_i)
        floor = _finite_scalar("edge_floor", self.edge_floor, "invalid_normalization")
        if not 0.0 < floor < 1.0:
            raise SyntheticProfileError("invalid_normalization", f"edge_floor must lie in (0, 1), got {floor!r}")
        object.__setattr__(self, "edge_floor", floor)
        if self.edge_value is not None:
            if (self.pressure_constraint, self.closure) not in (("equilibrium", "temperature"),
                                                                ("equilibrium", "density")):
                raise SyntheticProfileError(
                    "invalid_normalization",
                    "edge_value is the separatrix value of the channel a local temperature or density "
                    "closure solves; this closure solves none")
            edge = _finite_scalar("edge_value", self.edge_value, "invalid_normalization")
            if edge <= 0.0:
                raise SyntheticProfileError("invalid_normalization", f"edge_value must be positive, got {edge!r}")
            object.__setattr__(self, "edge_value", edge)
        if self.pressure_constraint not in PRESSURE_CONSTRAINTS:
            raise SyntheticProfileError(
                "invalid_profile_model",
                f"pressure_constraint must be one of {PRESSURE_CONSTRAINTS}, got {self.pressure_constraint!r}")
        allowed = PRESSURE_CLOSURES[self.pressure_constraint]
        if self.closure not in allowed:
            raise SyntheticProfileError(
                "invalid_profile_model",
                f"closure {self.closure!r} does not belong to pressure_constraint="
                f"{self.pressure_constraint!r}; choose one of {allowed}")
        if self.psi_norm is not None:
            grid = _sealed(self.psi_norm)
            if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)):
                raise SyntheticProfileError("coordinate_mapping_failed",
                                            "psi_norm must be a finite 1-D grid of at least 3 points")
            if grid[0] != 0.0 or grid[-1] != 1.0 or np.any(np.diff(grid) <= 0.0):
                raise SyntheticProfileError("coordinate_mapping_failed",
                                            "psi_norm must be strictly increasing from exactly 0 to exactly 1")
            object.__setattr__(self, "psi_norm", grid)


#: The unit each channel is stored in; a tabulated or gradient shape must
#: declare it or leave its unit empty.  Nothing is converted.
_CHANNEL_UNITS = {"n_e": "m^-3", "T_e": "eV", "T_i": "eV"}


def _check_channel(channel: str, spec: "ProfileSpec") -> None:
    """Refuse a shape built for another channel, in a unit the channel is not
    stored in, or normalized to a target only another channel can carry."""
    if spec.target is not None and spec.target.kind == "greenwald_fraction" and channel != "n_e":
        # Refused when the spec is built, not after the geometry is traced
        # (cold review 0.8.0 plasma-state-and-chease F7).
        raise SyntheticProfileError(
            "invalid_normalization",
            f"a Greenwald fraction normalizes n_e, not {channel}; it is a line-averaged density over "
            "n_G = I_p/(pi a^2)")
    shape = spec.shape
    if isinstance(shape, AnalyticProfile):
        if shape.quantity != channel or shape.unit != _CHANNEL_UNITS[channel]:
            raise SyntheticProfileError(
                "invalid_profile_model",
                f"the {channel} slot got a {shape.quantity!r} profile in {shape.unit!r}; expected "
                f"{channel!r} in {_CHANNEL_UNITS[channel]!r}")
    elif shape.unit not in ("", _CHANNEL_UNITS[channel]):
        raise SyntheticProfileError(
            "invalid_profile_model",
            f"the {channel} shape is in {shape.unit!r}; {channel} is stored in {_CHANNEL_UNITS[channel]!r} "
            "and no unit is converted -- give the values in that unit")


# --- results -------------------------------------------------------------------


@dataclass(frozen=True)
class IonSpecies:
    """One ion species of a generated state: label, charge, mass and role."""

    label: str
    z: float
    a: float
    role: str

    @property
    def element(self) -> str:
        """The element symbol, hydrogen for D and T [-]."""
        return "H" if self.label in ("D", "T") else self.label


@dataclass(frozen=True)
class TargetResidual:
    """One requested constraint beside the value recomputed from the final profiles.

    ``achieved`` is always recomputed, never copied from ``requested``.
    ``solver_status`` says how the constraint was imposed: ``"linear"`` for an
    amplitude scaling, ``"brentq"`` for a solved shape parameter,
    ``"local"`` for a pointwise identity, ``"not_reached"`` when it failed.
    """

    name: str
    definition: str
    requested: float
    achieved: float
    residual: float
    relative_residual: float
    solver_status: str
    tolerance: float
    met: bool


@dataclass(frozen=True, eq=False)
class PressureClosureReport:
    """How the kinetic pressure compares with the equilibrium's, and why.

    ``held`` and ``solved`` name the variables the closure kept and solved.
    ``relative_residual = (p_kin - p_eq)/max(p_eq)``; the maximum and RMS are
    over the exact region: the leading run of points with
    ``p_eq >= pressure_floor * max(p_eq)``, which ends at
    ``closure_exact_up_to_psi_n``.  ``edge_max_relative_residual`` is the
    same measure beyond that point, where a local closure carries its solved
    channel to the separatrix instead of solving it (see
    :class:`SyntheticKineticSpec`).  ``locally_consistent`` is the local test
    on the exact region only: an integrated agreement (``thermal_energy``)
    never sets it.
    """

    mode: str
    closure: str | None
    held: tuple[str, ...]
    solved: tuple[str, ...]
    absolute_residual: np.ndarray | None
    relative_residual: np.ndarray | None
    max_relative_residual: float
    rms_relative_residual: float
    thermal_energy_eq: float
    thermal_energy_kin: float
    thermal_energy_relative_difference: float
    pressure_floor: float
    tolerance: float
    locally_consistent: bool
    closure_exact_up_to_psi_n: float = float("nan")
    edge_max_relative_residual: float = float("nan")

    def __post_init__(self) -> None:
        for name in ("absolute_residual", "relative_residual"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sealed(value))


def _consistent(name: str, stored: np.ndarray, expected: np.ndarray) -> None:
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    if not np.all(np.abs(stored - expected) <= _CONSISTENCY_RTOL * scale):
        raise ValueError(
            f"{name} disagrees with the stored densities and temperatures; a synthetic state's "
            "pressures are derived -- build it with vaft.process.profile.generate_synthetic_kinetic_profiles"
        )


_RESULT_ARRAYS = ("psi_norm", "rho_pol_norm", "volume", "n_e", "n_i", "n_impurity", "T_e", "T_i",
                  "z_eff", "p_e", "p_i", "p_total", "p_eq")


@dataclass(frozen=True, eq=False)
class SyntheticKineticProfiles:
    """A synthetic kinetic state generated from an equilibrium, with its checks.

    Arrays are on one grid, :attr:`psi_norm` (normalized poloidal flux of
    the source equilibrium), with :attr:`rho_pol_norm` ``= sqrt(psi_norm)``,
    :attr:`rho_tor_norm` from that equilibrium's ``q`` (``None`` when it
    cannot be derived) and :attr:`psi` in Wb, COCOS 11 (``None`` when the
    source convention does not fix it).  ``n_i`` is the main ion,
    ``n_impurity`` the impurity; ``p_e = e n_e T_e``,
    ``p_i = e (n_i + n_impurity) T_i``, ``p_total = p_e + p_i`` -- checked on
    construction to 1e-12, so no stored pressure can disagree with the stored
    densities and temperatures.  ``p_eq`` is the source equilibrium's pressure
    on the grid, never overwritten; ``NaN`` where the source carries none.

    ``status`` is one of :data:`SYNTHETIC_PROFILE_STATUSES`; only a
    ``"success"`` result may be written to ``core_profiles``.
    ``fidelity_level`` is the highest of 0-3 any channel used.  Nothing here
    is measured or transport-predicted: ``provenance["profile_basis"]`` is
    always ``"assumed"``.
    """

    label: str
    status: str
    message: str
    fidelity_level: int
    psi_norm: np.ndarray
    rho_pol_norm: np.ndarray
    rho_tor_norm: np.ndarray | None
    psi: np.ndarray | None
    volume: np.ndarray
    n_e: np.ndarray
    n_i: np.ndarray
    n_impurity: np.ndarray
    T_e: np.ndarray
    T_i: np.ndarray
    z_eff: np.ndarray
    p_e: np.ndarray
    p_i: np.ndarray
    p_total: np.ndarray
    p_eq: np.ndarray
    species: tuple[IonSpecies, ...]
    pressure: PressureClosureReport
    targets: tuple[TargetResidual, ...]
    validation: Mapping[str, Any]
    resolved: Mapping[str, Any]
    spec: SyntheticKineticSpec
    provenance: Mapping[str, Any]
    time: float | None = None

    def __post_init__(self) -> None:
        if self.status not in SYNTHETIC_PROFILE_STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        grid = _sealed(self.psi_norm)
        for name in _RESULT_ARRAYS:
            values = _sealed(getattr(self, name))
            if values.shape != grid.shape:
                raise ValueError(f"{name} has shape {values.shape}; psi_norm has {grid.shape}")
            object.__setattr__(self, name, values)
        for name in ("rho_tor_norm", "psi"):
            values = getattr(self, name)
            if values is not None:
                values = _sealed(values)
                if values.shape != grid.shape:
                    raise ValueError(f"{name} has shape {values.shape}; psi_norm has {grid.shape}")
                object.__setattr__(self, name, values)
        _consistent("p_e", self.p_e, _QE * self.n_e * self.T_e)
        _consistent("p_i", self.p_i, _QE * (self.n_i + self.n_impurity) * self.T_i)
        _consistent("p_total", self.p_total, self.p_e + self.p_i)
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "targets", tuple(self.targets))
        for name in ("validation", "resolved", "provenance"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __len__(self) -> int:
        return int(self.psi_norm.size)

    @property
    def ok(self) -> bool:
        """Whether every requested constraint was met [-]."""
        return self.status == "success"

    def unit(self, name: str) -> str:
        """The unit of one of the result's arrays."""
        return SYNTHETIC_PROFILE_UNITS[name]

    def coordinate(self, name: str) -> np.ndarray:
        """The grid expressed in ``psi_norm``, ``rho_pol_norm`` or ``rho_tor_norm`` [-]."""
        if name not in _COORDINATES:
            raise ValueError(f"coordinate must be one of {_COORDINATES}, got {name!r}")
        values = getattr(self, name)
        if values is None:
            raise ValueError(f"{name} is unavailable for this equilibrium: {self.provenance.get('rho_tor_norm')}")
        return values

    def a_over_L(self, channel: str, coordinate: str = "rho_pol_norm") -> np.ndarray:
        """``-d ln f/dx`` of ``n_e``, ``T_e`` or ``T_i``, by second-order differences on the grid [-].

        An independent reconstruction from the stored profile, used to check a
        prescribed gradient; ``NaN`` where the profile is not positive.
        """
        values = np.asarray(getattr(self, channel), dtype=float)
        x = self.coordinate(coordinate)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f = np.where(values > 0.0, np.log(np.where(values > 0.0, values, 1.0)), np.nan)
            return -np.gradient(log_f, x, edge_order=2)

    def to_kinetic_profiles(self) -> KineticProfiles:
        """The state as the package's kinetic-profile container, same units, same grid.

        ``n_impurity`` becomes ``n_z``, ``T_i`` is also ``T_z``; ``p_eq``, the
        species pressures, ``z_eff`` and the volume go to ``extras``.  The
        coordinate is recorded as generated, not read.
        """
        extras = {"p_e": self.p_e, "p_i": self.p_i, "p_eq": self.p_eq, "z_eff": self.z_eff,
                  "volume": self.volume}
        provenance = {name: f"synthetic ({self.provenance.get('channels', {}).get(name, 'derived')})"
                      for name in ("n_e", "T_e", "T_i")}
        provenance.update({name: "derived from n_e, T_e, T_i and the declared composition"
                           for name in ("n_i", "n_z", "p_total", "p_e", "p_i", "z_eff")})
        provenance["p_eq"] = "source equilibrium pressure on the grid"
        species = tuple(Species(label=s.label, n=float("nan"), z=s.z, a=s.a) for s in self.species)
        return KineticProfiles(
            psi_norm=self.psi_norm, n_e=self.n_e, n_i=self.n_i, n_z=self.n_impurity,
            T_e=self.T_e, T_i=self.T_i, T_z=self.T_i, p_total=self.p_total,
            normalization=PsiNormalization(method="generated", source="vaft.process.profile"),
            species=species, extras=extras, provenance=provenance,
            source=f"synthetic kinetic profiles {self.label!r} (#122)",
        )


def _species_for_state(state: AnalyticPlasmaState) -> str | None:
    """The impurity symbol whose nuclear charge equals a #1045 state's impurity charge."""
    if state.z_eff == 1.0:
        return None
    for symbol, (z, _) in ION_SPECIES.items():
        if z == state.impurity_charge:
            return symbol
    raise SyntheticProfileError(
        "invalid_composition",
        f"no species of ION_SPECIES has charge {state.impurity_charge:g}; name the impurity explicitly")
