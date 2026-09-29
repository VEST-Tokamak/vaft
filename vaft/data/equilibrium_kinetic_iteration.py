"""Records for the self-consistent equilibrium / kinetic-profile iteration (#123).

The iteration in :mod:`vaft.code.chease_kinetic_iteration` composes three
existing pieces -- the #122 synthetic kinetic profiles, a CHEASE-compatible
pressure source built from them, and a fixed-boundary CHEASE solve -- and
repeats until the kinetic pressure and the equilibrium pressure agree.  These
records keep what that iteration is told apart from what it computes:

* :class:`EquilibriumKineticSpec` -- the closure mode, the #122 kinetic spec
  (re-applied unchanged to every equilibrium), the current / q policy and the
  convergence criteria;
* :class:`CurrentPolicy` -- the non-pressure Grad-Shafranov source, which
  pressure alone does not fix, and the one normalization CHEASE imposes;
* :class:`ConvergenceCriteria` -- every tolerance, the iteration cap, the
  optional under-relaxation and the divergence / oscillation / stagnation
  detectors;
* :class:`PressureSource`, :class:`CurrentSource` -- the ``p``/``p'`` and
  ``FF'`` actually handed to CHEASE, with the method that built them;
* :class:`IterationState` -- one immutable state (the initial one, or one
  CHEASE update) with its equilibrium, kinetic profiles, residuals and
  validation;
* :class:`SelfConsistentState` -- the whole history and the final ODS.

The converged state is an *assumption-driven, self-consistent* state: the
kinetic profiles are declared assumptions and the equilibrium is re-solved
for their pressure.  Nothing in it is transport-predicted or reconstructed
from measurements, and the provenance says so.

Units: pressures in Pa, flux in the source g-file's own units (Wb/rad), ``p'``
in Pa per that unit, ``FF'`` in T^2 m^2 per that unit, lengths in m, the
radial coordinates dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .analytic_plasma_state import AnalyticProfile
from .synthetic_kinetic_profiles import SyntheticKineticProfiles, SyntheticKineticSpec

__all__ = [
    "CLOSURE_MODES",
    "CURRENT_NORMALIZATIONS",
    "CURRENT_POLICIES",
    "ITERATION_STATUSES",
    "STATE_STATUSES",
    "ConvergenceCriteria",
    "CurrentPolicy",
    "CurrentSource",
    "EquilibriumKineticSpec",
    "IterationState",
    "PressureSource",
    "SelfConsistentState",
]

#: Which pressure is authoritative.
CLOSURE_MODES = ("equilibrium_pressure", "kinetic_pressure")

#: How the non-pressure source ``FF'`` of the next solve is obtained.
CURRENT_POLICIES = ("preserve_ffprime_shape", "analytic_ffprime_shape")

#: The one normalization CHEASE imposes, and its ``NCSCAL``.
CURRENT_NORMALIZATIONS: Mapping[str, int] = MappingProxyType({"plasma_current": 2, "q95": 1})

#: Outcome of the whole iteration.
ITERATION_STATUSES = (
    "converged",
    "max_iterations",
    "diverged",
    "oscillating",
    "stagnated",
    "drifting",
    "kinetic_generation_failed",
    "invalid_pressure_source",
    "chease_failed",
    "validation_failed",
)

#: Outcome of one state: ``initial`` for the starting state, ``accepted`` for
#: a CHEASE update that passed validation and produced a valid kinetic state,
#: otherwise the failure that stopped the iteration there.
STATE_STATUSES = (
    "initial",
    "accepted",
    "kinetic_generation_failed",
    "invalid_pressure_source",
    "chease_failed",
    "validation_failed",
)


def _sealed(values) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _frozen_mapping(value) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CurrentPolicy:
    """The non-pressure source of the next CHEASE solve, and its normalization.

    Pressure alone does not determine a Grad-Shafranov equilibrium; this
    policy states what does, and nothing else is inferred.

    ``kind``:

    * ``"preserve_ffprime_shape"`` -- ``FF'(psi_N)`` keeps the normalized
      shape of the *initial* equilibrium's ``FFPRIM``, re-sampled from that
      one array every iteration (never from a re-interpolated one), with the
      amplitude carried over from the previous solve (its least-squares
      projection on the shape);
    * ``"analytic_ffprime_shape"`` -- ``FF' = -A dg/dpsi_N`` for the declared
      ``current_profile`` ``g``, an
      :class:`~vaft.data.analytic_plasma_state.AnalyticProfile` in the role of
      ``F^2/2`` above its edge value (the #1166 scope-B convention of
      :class:`vaft.code.chease_synthesis.ZeroDimensionalEquilibriumSpec`),
      with ``A`` carried over as above.

    ``normalization`` is the one quantity CHEASE imposes (``NCSCAL``):
    ``"plasma_current"`` (2) holds the initial equilibrium's ``I_p``;
    ``"q95"`` (1) holds ``q`` at ``psi_N = 0.95``.  Both targets are taken
    once from the initial equilibrium and written into every CHEASE input as
    the requested value: ``I_p`` as the input ``CURRENT`` (read as
    ``CURRT``), ``q95`` by rescaling the input ``QPSI`` so the adapter's
    ``QSPEC`` sample at 0.95 equals the target (``QPSI`` reaches ``EXPEQ``
    only through ``QSPEC`` and its sign).  Neither is carried from the
    previous solve, so a solver bias cannot accumulate; the achieved value is
    checked against the target on every update.  CHEASE rescales ``p'`` and
    ``FF'`` together to meet it, so the ``FF'`` amplitude -- and with it the
    split of the current between the pressure and ``FF'`` terms -- is
    re-solved; q is always an output, never written after a solve.
    """

    kind: str = "preserve_ffprime_shape"
    normalization: str = "plasma_current"
    current_profile: AnalyticProfile | None = None

    def __post_init__(self) -> None:
        if self.kind not in CURRENT_POLICIES:
            raise ValueError(f"current policy kind must be one of {CURRENT_POLICIES}, got {self.kind!r}")
        if self.normalization not in CURRENT_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {tuple(CURRENT_NORMALIZATIONS)} (the CHEASE NCSCAL paths), "
                f"got {self.normalization!r}")
        if self.kind == "analytic_ffprime_shape" and not isinstance(self.current_profile, AnalyticProfile):
            raise ValueError("analytic_ffprime_shape needs current_profile, an AnalyticProfile g(psi_N)")
        if self.kind == "preserve_ffprime_shape" and self.current_profile is not None:
            raise ValueError("preserve_ffprime_shape keeps the initial FF' shape; a current_profile contradicts it")

    @property
    def ncscal(self) -> int:
        """CHEASE's ``NCSCAL`` for the normalization [-]."""
        return int(CURRENT_NORMALIZATIONS[self.normalization])

    def describe(self) -> dict[str, str]:
        """What the policy holds and what it leaves to CHEASE, in words [-]."""
        shape = ("normalized FF'(psi_N) shape of the initial equilibrium" if self.kind == "preserve_ffprime_shape"
                 else "FF' shape -dg/dpsi_N of the declared analytic current profile g")
        held = "plasma current I_p of the initial equilibrium" if self.normalization == "plasma_current" \
            else "q at psi_N = 0.95 of the initial equilibrium"
        other = "q95, q0 and q(psi)" if self.normalization == "plasma_current" else "I_p, q0 and q(psi)"
        return {
            "held_shape": shape,
            "held_normalization": held,
            "chease_ncscal": str(self.ncscal),
            "re_solved": f"FF' amplitude, psi(R,Z), {other}, l_i, beta, the magnetic axis",
            "never": "q is never written after a solve; no current model is inferred from the pressure",
        }


@dataclass(frozen=True)
class ConvergenceCriteria:
    """Tolerances, the iteration cap and the failure detectors.

    Every residual is on the kinetic grid (the #122 spec's ``psi_norm``, the
    same in every iteration).  A tolerance of ``None`` disables that
    criterion; a state is ``converged`` only when every enabled one holds:

    * ``pressure_rtol`` -- ``max|p_kin - p_eq| / max p_kin`` of the *new*
      equilibrium, re-extracted after the solve;
    * ``profile_rtol`` -- the change since the previous state of ``p_kin``,
      ``p_eq``, ``n_e``, ``T_e``, ``T_i`` and ``n_i``, each as
      ``max|f_n - f_{n-1}| / max|f_n|``;
    * ``q_rtol`` -- the same for ``|q|``;
    * ``coordinate_atol`` -- ``max|rho_tor_norm_n - rho_tor_norm_{n-1}|``;
    * ``scalar_rtol`` -- the relative change of ``q0``, ``q95``, ``l_i``,
      ``beta_p`` and ``W_th``;
    * ``axis_atol`` -- the change of ``R_axis`` and ``Z_axis`` [m].

    ``normalization_rtol`` bounds the miss of the held ``I_p`` or ``q95``
    and ``boundary_rtol`` the solved boundary's distance from the requested
    one (in minor radii); both are *validation*, applied to every update.

    ``relaxation`` is the under-relaxation factor ``alpha`` in ``(0, 1]`` of
    the pressure handed to CHEASE:
    ``p_src,n = p_src,n-1 + alpha (p_kin,n - p_src,n-1)``, ``p_src,0`` the
    initial equilibrium's pressure.  Only that exchanged pressure is
    relaxed; the raw and the relaxed pressure are both kept.

    Detectors, checked after convergence, in this order (``r`` is the
    pressure residual, ``s`` the signed thermal-energy residual
    ``(W_kin - W_eq)/W_eq``):

    * ``diverged`` -- ``r`` grew in each of the last ``divergence_window``
      iterations and exceeds its initial value;
    * ``oscillating`` -- ``s`` alternated in sign over the last
      ``oscillation_window`` states and ``|s_n| >= oscillation_ratio
      |s_{n-2}|`` (an alternating but decaying residual is convergence);
    * ``drifting`` -- some criterion is unmet and an equilibrium scalar
      (``q0``, ``q95``, ``l_i``, ``beta_p``, ``W_th``, ``W_kin``, the axis,
      ``I_p``) moved in one direction over the last ``stagnation_window``
      updates by relative steps above ``scalar_rtol`` that are not decaying
      (``|step_n| >= stagnation_ratio |step_first|``): a runaway, whatever
      the pressure residual says;
    * ``stagnated`` -- no drift, and every *unmet* criterion's worst metric
      satisfies ``m_n >= stagnation_ratio m_{n - stagnation_window}``.  Met
      criteria never count, so a pressure residual sitting at its floor does
      not make a stagnation.

    ``monotone_rtol`` is the largest outward rise of the pressure handed to
    CHEASE, relative to its maximum, that is removed (by a running minimum
    from the axis, and counted in the source's provenance) rather than
    refused; above it the source is ``invalid_pressure_source``.
    """

    pressure_rtol: float | None = 1e-3
    profile_rtol: float | None = 1e-3
    q_rtol: float | None = 1e-3
    coordinate_atol: float | None = 1e-3
    scalar_rtol: float | None = 1e-3
    axis_atol: float | None = 1e-4
    normalization_rtol: float = 1e-3
    boundary_rtol: float = 1e-2
    max_iterations: int = 10
    relaxation: float = 1.0
    divergence_window: int = 2
    oscillation_window: int = 4
    oscillation_ratio: float = 0.9
    stagnation_window: int = 3
    stagnation_ratio: float = 0.95
    monotone_rtol: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("pressure_rtol", "profile_rtol", "q_rtol", "coordinate_atol", "scalar_rtol", "axis_atol"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive or None, got {value!r}")
        for name in ("normalization_rtol", "boundary_rtol"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.monotone_rtol) < 1e-2:
            raise ValueError("monotone_rtol must lie in [0, 1e-2)")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 < float(self.relaxation) <= 1.0:
            raise ValueError(f"relaxation must lie in (0, 1], got {self.relaxation!r}")
        for name in ("divergence_window", "stagnation_window"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if int(self.oscillation_window) < 3:
            raise ValueError("oscillation_window must be at least 3 (two sign changes)")
        for name in ("oscillation_ratio", "stagnation_ratio"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

    def enabled(self) -> dict[str, float]:
        """The enabled tolerances by name [-, m for axis_atol]."""
        names = ("pressure_rtol", "profile_rtol", "q_rtol", "coordinate_atol", "scalar_rtol", "axis_atol")
        return {name: float(getattr(self, name)) for name in names if getattr(self, name) is not None}


@dataclass(frozen=True, eq=False)
class EquilibriumKineticSpec:
    """What the self-consistent iteration is told, separate from the equilibrium.

    ``closure`` is one of :data:`CLOSURE_MODES`:

    * ``"equilibrium_pressure"`` -- the equilibrium pressure is authoritative;
      ``kinetic`` must use ``pressure_constraint="equilibrium"`` so the #122
      generator decomposes ``p_eq`` into kinetic channels.  One pass, no
      CHEASE solve.
    * ``"kinetic_pressure"`` -- the kinetic pressure is authoritative;
      ``kinetic`` must use ``pressure_constraint="kinetic"`` (every channel
      as declared).  ``"equilibrium"`` is refused: it makes ``p_kin = p_eq``
      by construction and the update would be an identity.
      ``"thermal_energy"`` is refused too: it scales ``W_kin`` to the
      *current* equilibrium's ``W_eq``, which the update then sets from
      ``p_kin``, so the amplitude is a neutral mode and the converged ``W``
      depends on the path.  It needs a ``W`` target fixed in the spec, which
      is not implemented.

    ``kinetic`` is re-applied unchanged to every new equilibrium, so shape
    functions are regenerated on the new coordinates, never re-interpolated
    from an earlier iteration's arrays.
    """

    kinetic: SyntheticKineticSpec
    closure: str = "kinetic_pressure"
    current_policy: CurrentPolicy = field(default_factory=CurrentPolicy)
    convergence: ConvergenceCriteria = field(default_factory=ConvergenceCriteria)
    label: str = "self-consistent"

    def __post_init__(self) -> None:
        if not isinstance(self.kinetic, SyntheticKineticSpec):
            raise ValueError("kinetic must be a SyntheticKineticSpec (#122)")
        if self.closure not in CLOSURE_MODES:
            raise ValueError(f"closure must be one of {CLOSURE_MODES}, got {self.closure!r}")
        if not isinstance(self.current_policy, CurrentPolicy):
            raise ValueError("current_policy must be a CurrentPolicy")
        if not isinstance(self.convergence, ConvergenceCriteria):
            raise ValueError("convergence must be a ConvergenceCriteria")
        constraint = self.kinetic.pressure_constraint
        if self.closure == "equilibrium_pressure" and constraint != "equilibrium":
            raise ValueError(
                "closure='equilibrium_pressure' decomposes p_eq: the kinetic spec needs "
                f"pressure_constraint='equilibrium', got {constraint!r}")
        if self.closure == "kinetic_pressure" and constraint == "equilibrium":
            raise ValueError(
                "closure='kinetic_pressure' makes the kinetic pressure authoritative; a kinetic spec with "
                "pressure_constraint='equilibrium' sets p_kin = p_eq by construction, so the CHEASE update "
                "would be an identity. Use 'kinetic'")
        if self.closure == "kinetic_pressure" and constraint == "thermal_energy":
            raise ValueError(
                "closure='kinetic_pressure' with pressure_constraint='thermal_energy' is ill-posed: W_kin is "
                "normalized to the evolving equilibrium's W_eq, which the update sets from W_kin, so the "
                "amplitude is a neutral mode and the converged W depends on the iteration path. It needs a W "
                "target fixed in the spec (not implemented); declare the amplitude with a ScalarTarget and "
                "use pressure_constraint='kinetic'")


@dataclass(frozen=True, eq=False)
class PressureSource:
    """The pressure handed to CHEASE for one update, and how it was built.

    ``raw_pressure`` is ``p_kin`` on the kinetic grid ``kinetic_psi_norm``;
    ``relaxed_pressure`` is what was used after under-relaxation (identical
    for ``relaxation = 1``).  ``psi_norm`` is the g-file's uniform grid on
    which ``pressure`` (``PRES``) and ``pprime`` (``PPRIME = dp/dpsi``) were
    written; ``psi_span = psi_boundary - psi_axis`` of the equilibrium the
    derivative was taken on.  ``rise_max_relative`` is the largest outward
    rise of the used pressure relative to its maximum, and
    ``rise_points_removed`` how many points a running minimum lowered to
    remove rises below ``ConvergenceCriteria.monotone_rtol`` (zero for a
    monotone pressure).  The round trip ``p -> p' -> p`` is checked on the
    g-file actually handed to CHEASE
    (:func:`vaft.code.chease_kinetic_iteration.pressure_roundtrip`), not here.
    """

    kinetic_psi_norm: np.ndarray
    raw_pressure: np.ndarray
    relaxed_pressure: np.ndarray
    relaxation: float
    psi_norm: np.ndarray
    pressure: np.ndarray
    dpressure_dpsi_norm: np.ndarray
    pprime: np.ndarray
    psi_span: float
    method: str
    rise_max_relative: float = 0.0
    rise_points_removed: int = 0

    def __post_init__(self) -> None:
        for name in ("kinetic_psi_norm", "raw_pressure", "relaxed_pressure", "psi_norm", "pressure",
                     "dpressure_dpsi_norm", "pprime"):
            object.__setattr__(self, name, _sealed(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class CurrentSource:
    """The ``FF'`` handed to CHEASE for one update, and how it was built.

    ``shape`` is the declared normalized shape on ``psi_norm``, ``amplitude``
    the factor carried over from the previous solve, ``ffprime = amplitude *
    shape`` in the g-file's convention.
    """

    psi_norm: np.ndarray
    shape: np.ndarray
    amplitude: float
    ffprime: np.ndarray
    method: str

    def __post_init__(self) -> None:
        for name in ("psi_norm", "shape", "ffprime"):
            object.__setattr__(self, name, _sealed(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class IterationState:
    """One immutable state of the iteration.

    ``index`` 0 is the initial state (``status == "initial"``: the given
    equilibrium and the kinetic profiles generated on it); ``index >= 1`` is
    one CHEASE update: the ``pressure_source`` and ``current_source`` built
    from state ``index - 1``, the CHEASE working directory, the solved
    equilibrium, the kinetic profiles *regenerated* on it from the same spec,
    and the residuals of that new pair.

    ``equilibrium`` is the solved (or initial) g-file object, ``geqdsk_path``
    where it is on disk.  ``p_eq``, ``p_kin``, ``q`` (``|q|``) and
    ``rho_tor_norm`` are on the kinetic grid ``psi_norm``.  ``scalars`` are
    recomputed from the equilibrium (``ip``, ``q0``, ``q95``, ``qmin``,
    ``li``, ``beta_p``, ``beta_t``, ``beta_n``, ``magnetic_axis_r``,
    ``magnetic_axis_z``, ``shafranov_shift``, ``w_th``); ``metrics`` are the
    residuals and the changes since the previous state; ``validation`` the
    CHEASE-update checks with their values.
    """

    index: int
    status: str
    message: str
    equilibrium: Any = None
    geqdsk_path: Path | None = None
    kinetic: SyntheticKineticProfiles | None = None
    psi_norm: np.ndarray | None = None
    p_eq: np.ndarray | None = None
    p_kin: np.ndarray | None = None
    q: np.ndarray | None = None
    rho_tor_norm: np.ndarray | None = None
    scalars: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    pressure_source: PressureSource | None = None
    current_source: CurrentSource | None = None
    chease_workdir: Path | None = None
    chease_inputs: Mapping[str, Any] = field(default_factory=dict)
    rational_surfaces: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATE_STATUSES:
            raise ValueError(f"unknown state status {self.status!r}")
        for name in ("psi_norm", "p_eq", "p_kin", "q", "rho_tor_norm"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sealed(value))
        for name in ("scalars", "metrics", "validation", "chease_inputs", "rational_surfaces"):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))

    @property
    def accepted(self) -> bool:
        """Whether this state is a usable equilibrium / kinetic pair [-]."""
        return self.status in ("initial", "accepted")


#: What each quantity of the iteration is: held, recomputed from the kinetic
#: state, recomputed by CHEASE, a numerical control, or a convergence metric.
_ROLES = (
    ("kinetic assumptions (the #122 spec: shapes, targets, T_i route, composition)", "held"),
    ("fixed boundary (the initial equilibrium's RBBBS, target_psin = 1)", "held"),
    ("vacuum field F_edge = R0 B0", "held"),
    ("p_kin, p_e, p_i, n_e, T_e, T_i, n_i, Z_eff", "recomputed from the kinetic state"),
    ("PRES and PPRIME handed to CHEASE", "recomputed from the kinetic state"),
    ("psi(R,Z), p_eq, q(psi), FF' amplitude, magnetic axis, l_i, beta", "recomputed by CHEASE"),
    ("rho_tor_norm(psi_N), V(psi), line-average chord", "recomputed from the new equilibrium"),
    ("CHEASE mesh, iteration cap, relaxation", "numerical control"),
    ("pressure residual, profile/q/coordinate/scalar changes", "convergence metric"),
)


@dataclass(frozen=True, eq=False)
class SelfConsistentState:
    """The result of the equilibrium / kinetic iteration: every state, and the final ODS.

    ``status`` is one of :data:`ITERATION_STATUSES` and ``message`` says why.
    ``initial`` is state 0; ``iterations`` are the CHEASE updates in order,
    failed ones included; ``final`` is the last *accepted* state (the initial
    one when nothing was accepted, ``None`` when the initial state itself
    failed).  ``ods`` holds that final state's ``equilibrium`` and
    ``core_profiles`` when the final state has a valid kinetic state, else
    ``None``; an equilibrium this workflow did not solve (the initial one)
    keeps its own ``code`` and comment.  ``baseline``, when requested, is a
    CHEASE re-solve of the initial equilibrium with its *own* ``p'`` and
    ``FF'``: it separates the change of solver (EFIT to CHEASE) from the
    change of pressure.

    Nothing here is transport-predicted or reconstructed: the kinetic
    profiles are declared assumptions and the equilibrium is re-solved for
    their pressure, so ``provenance["state_basis"]`` is
    ``"assumption-driven self-consistent"``.
    """

    spec: EquilibriumKineticSpec
    status: str
    message: str
    initial: IterationState
    iterations: tuple[IterationState, ...]
    final: IterationState | None
    ods: Any
    policy: Mapping[str, Any]
    provenance: Mapping[str, Any]
    baseline: IterationState | None = None

    def __post_init__(self) -> None:
        if self.status not in ITERATION_STATUSES:
            raise ValueError(f"unknown iteration status {self.status!r}")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        for name in ("policy", "provenance"):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))

    @property
    def converged(self) -> bool:
        """Whether every enabled criterion was met [-]."""
        return self.status == "converged"

    @property
    def states(self) -> tuple[IterationState, ...]:
        """The initial state followed by every iteration [-]."""
        return (self.initial,) + self.iterations

    @property
    def initial_equilibrium(self):
        """The g-file object the iteration started from [-]."""
        return self.initial.equilibrium

    @property
    def final_equilibrium(self):
        """The final accepted state's g-file object, ``None`` when there is none [-]."""
        return None if self.final is None else self.final.equilibrium

    @property
    def final_core_profiles(self) -> SyntheticKineticProfiles | None:
        """The final accepted state's kinetic profiles [-]."""
        return None if self.final is None else self.final.kinetic

    def history(self, name: str) -> np.ndarray:
        """One metric or scalar per state, ``NaN`` where a state lacks it [-]."""
        values = []
        for state in self.states:
            source = state.metrics if name in state.metrics else state.scalars
            values.append(float(source.get(name, np.nan)))
        return np.asarray(values, dtype=float)

    @property
    def convergence_history(self) -> dict[str, np.ndarray]:
        """Every metric recorded on any state, one value per state [-]."""
        names = sorted({key for state in self.states for key in state.metrics})
        return {name: self.history(name) for name in names}

    @property
    def pressure_residuals(self) -> dict[str, np.ndarray]:
        """The ``p_kin`` - ``p_eq`` residuals per state [-]."""
        return {name: self.history(name) for name in
                ("pressure_max_relative", "pressure_rms_relative", "thermal_energy_relative")}

    @property
    def profile_residuals(self) -> dict[str, np.ndarray]:
        """The per-channel changes since the previous state [-]."""
        return {name: self.history(name) for name in
                ("p_kin_change", "p_eq_change", "n_e_change", "T_e_change", "T_i_change", "n_i_change",
                 "q_change")}

    @property
    def coordinate_mapping_history(self) -> list[np.ndarray | None]:
        """``rho_tor_norm`` on the kinetic grid, per state [-]."""
        return [state.rho_tor_norm for state in self.states]

    @property
    def validation(self) -> list[Mapping[str, Any]]:
        """Each state's validation record [-]."""
        return [state.validation for state in self.states]

    def assumption_table(self) -> list[tuple[str, str]]:
        """``(quantity, role)`` rows: held, recomputed from the kinetic state, by CHEASE, a control or a metric [-]."""
        if self.spec.closure == "equilibrium_pressure":
            return [
                ("kinetic assumptions (the #122 spec: held channels, T_i route, composition)", "held"),
                ("the given equilibrium: psi(R,Z), p_eq, q(psi), boundary", "held (no CHEASE update)"),
                ("the channel the closure solves so that p_kin = p_eq", "recomputed from the kinetic state"),
                ("n_i, Z_eff, p_e, p_i", "recomputed from the kinetic state"),
                ("pressure closure residual", "convergence metric"),
            ]
        rows = list(_ROLES)
        policy = self.spec.current_policy.describe()
        rows.insert(3, (policy["held_shape"], "held"))
        rows.insert(4, (policy["held_normalization"] + f" (CHEASE NCSCAL = {policy['chease_ncscal']})", "held"))
        return rows
