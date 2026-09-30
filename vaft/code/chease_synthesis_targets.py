"""Several 0D targets at once: an outer solve around the CHEASE synthesis (#120).

CHEASE imposes exactly one normalization per fixed-boundary solve -- the
plasma current (``NCSCAL = 2``) or q95 (``NCSCAL = 1``), see
:func:`vaft.code.chease_synthesis.synthesize_equilibrium_from_0d`.  Every
other 0D descriptor is an *output* of the boundary and the two source shapes.
To meet a second (or third) target, something in the source model has to
move, so this module wraps the single-target synthesis in an outer root-find
over the free knobs of that model.

Target -> knob mapping
----------------------
One normalization is imposed by CHEASE: ``plasma_current`` when it is among
the targets, else ``q95``, else the spec's own ``plasma_current``.  Each
remaining target claims one knob family:

========================  =====================  =====================================
target                    knob family            default knob (bounds)
========================  =====================  =====================================
``beta_p``, ``beta_n``    pressure share         ``pressure_fraction`` (0.02, 0.9)
``q95`` (Ip imposed)      current-profile shape  ``current_beta`` (0.3, 6.0)
``li``, ``q0``            current-profile shape  ``current_beta`` (0.3, 6.0)
========================  =====================  =====================================

At a fixed normalization, the pressure share moves the pressure-driven part of
the source, hence beta_p and beta_N; the ``FF'`` peaking exponent moves how
the current is distributed over the fixed boundary, hence l_i, q0 and q95.
The families are not decoupled -- a more peaked current also raises beta_p at
fixed pressure share -- which is why two free targets are solved together.

Reachability rules (refused before any solve, as ``unsupported_target``):

* two targets on one family (``beta_p`` with ``beta_n``; any two of ``q95``,
  ``li``, ``q0`` without the q95 normalization): one knob cannot set two
  descriptors;
* a current-shape target when the spec carries a ``current_profile``: the
  profile replaces the exponents, so there is no current-shape knob;
* a target that is not a descriptor of the synthesis at all.

**Ip together with q95.**  At a fixed boundary and fixed Ip, q95 moves only
through the current-profile shape, and it moves little: the edge safety factor
is set mostly by the boundary and the total current.  Every step toward the
q95 target also changes l_i and q0, which are *outputs* of that solve and are
reported, not held.  A q95 far from the geometric value (for VEST at 100 kA and
0.1 T, anything outside about 2.1-2.5) is ``not_reachable`` within any sensible
exponent range; it needs a different current, field or boundary.

Solver
------
Knobs are bounded and iterated as a coordinate in ``[0, 1]`` across their
bounds -- linear for ``pressure_fraction``, logarithmic for the shape
exponents, which act multiplicatively.  One free target: a bracketing secant
(regula falsi with a bisection safeguard once a sign change is found, a secant
clipped to the bounds before).  Two free targets: damped Newton with a
forward-difference Jacobian, the step projected onto the bounds and halved
until the tolerance-scaled residual norm decreases.  Every CHEASE solve --
initial, probe, Jacobian column, trial step -- is kept in the history with the
knob values, the requested targets and what that solve *achieved*.  An
achieved value is only ever read from the solved equilibrium; a descriptor the
solve did not produce is absent, never filled in from the request.

End states: ``converged``, ``max_iterations``, ``not_reachable`` (the target
lies outside what the knob bounds reach, or the Jacobian is singular),
``stalled`` (no descent inside the bounds), ``chease_failed`` (no solution or a
timeout), ``validation_failed`` (the single-target synthesis rejected the
solution: boundary, signs, the imposed target, a requested pressure shape),
``unsupported_target`` and ``invalid_request`` (refused before solving).

The 1-D ``not_reachable`` verdict assumes the descriptor is monotone in the
knob between the bounds, which holds for the knobs above; it is asserted only
after the relevant bound itself has been solved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from . import chease_synthesis as _synthesis
from .chease import CHEASEConfig
from .chease_synthesis import SyntheticEquilibriumResult, ZeroDimensionalEquilibriumSpec

__all__ = [
    "DEFAULT_KNOBS",
    "MULTI_TARGET_STATUSES",
    "MultiTargetSynthesisResult",
    "TARGET_DESCRIPTORS",
    "TargetIterationRecord",
    "TargetKnob",
    "assign_knobs",
    "synthesize_equilibrium_to_targets",
]

#: Targets the outer solve accepts, and the achieved descriptor each is read from.
TARGET_DESCRIPTORS = {"plasma_current": "ip", "q95": "q95", "beta_p": "beta_p", "beta_n": "beta_n",
                      "li": "li_virial", "q0": "q0"}

#: CHEASE's normalizations, in the order one is chosen for imposition.
_IMPOSABLE = ("plasma_current", "q95")

#: The knob family a free target claims.
_FAMILY = {"beta_p": "pressure_share", "beta_n": "pressure_share",
           "q95": "current_shape", "li": "current_shape", "q0": "current_shape"}

#: The spec fields each family may move.
_FAMILY_KNOBS = {"pressure_share": ("pressure_fraction",), "current_shape": ("current_beta", "current_alpha")}

#: Default relative tolerance of every target.
_DEFAULT_TOLERANCE = 0.01

#: End states of :func:`synthesize_equilibrium_to_targets`.
MULTI_TARGET_STATUSES = ("converged", "max_iterations", "not_reachable", "stalled", "chease_failed",
                         "validation_failed", "unsupported_target", "invalid_request")

#: Single-target states that mean nothing was solved.
_CHEASE_FAILURES = ("non_converged", "timeout")


@dataclass(frozen=True)
class TargetKnob:
    """A spec field the outer solve may move, and its bounds.

    ``name`` is a :class:`~vaft.code.chease_synthesis.ZeroDimensionalEquilibriumSpec`
    field (``pressure_fraction``, ``current_beta`` or ``current_alpha``);
    the solve never leaves ``[lower, upper]``.
    """

    name: str
    lower: float
    upper: float

    @property
    def logarithmic(self) -> bool:
        """Shape exponents act multiplicatively, so they are iterated in ``log``."""
        return self.name != "pressure_fraction"

    def clip(self, value: float) -> float:
        return float(min(max(value, self.lower), self.upper))

    def to_unit(self, value: float) -> float:
        """The knob value as a coordinate in ``[0, 1]`` across its bounds (clipped)."""
        f = np.log if self.logarithmic else float
        return float((f(self.clip(value)) - f(self.lower))/(f(self.upper) - f(self.lower)))

    def from_unit(self, u: float) -> float:
        """Inverse of :meth:`to_unit`; exact at the bounds."""
        u = min(max(float(u), 0.0), 1.0)
        if u in (0.0, 1.0):
            return self.lower if u == 0.0 else self.upper
        if self.logarithmic:
            return float(np.exp(np.log(self.lower) + u*(np.log(self.upper) - np.log(self.lower))))
        return float(self.lower + u*(self.upper - self.lower))


#: Default knob per family.  Bounds: numerical convenience -- ``pressure_fraction``
#: stays clear of 0 (a pressure profile would be discarded) and of 1 (FF' would
#: vanish); ``current_beta`` spans a hollow-ish 0.3 to a strongly peaked 6.
DEFAULT_KNOBS = {"pressure_share": TargetKnob("pressure_fraction", 0.02, 0.9),
                 "current_shape": TargetKnob("current_beta", 0.3, 6.0)}


@dataclass
class TargetIterationRecord:
    """One CHEASE solve of the outer iteration: what was asked, what it gave.

    ``achieved`` holds only descriptors read from that solve's equilibrium;
    ``residuals`` are ``achieved/requested - 1`` for the targets it produced.
    """

    solve: int
    iteration: int
    purpose: str
    knobs: Mapping[str, float]
    requested: Mapping[str, float]
    achieved: Mapping[str, float]
    residuals: Mapping[str, float]
    status: str
    reason: Optional[str]
    result: Optional[SyntheticEquilibriumResult] = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class MultiTargetSynthesisResult:
    """The outcome of :func:`synthesize_equilibrium_to_targets`.

    ``record`` is the solve the result reports: the converged one, else the
    valid solve with the smallest tolerance-scaled residual (``None`` when no
    solve succeeded).  ``achieved``/``residuals`` come from that record only.
    """

    status: str
    reason: Optional[str]
    spec: ZeroDimensionalEquilibriumSpec
    targets: Mapping[str, float]
    imposed: Optional[str] = None
    assignment: Mapping[str, TargetKnob] = field(default_factory=dict)
    tolerances: Mapping[str, float] = field(default_factory=dict)
    history: list = field(default_factory=list)
    record: Optional[TargetIterationRecord] = None
    reachable: Mapping[str, tuple] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "converged"

    @property
    def result(self) -> Optional[SyntheticEquilibriumResult]:
        """The single-target synthesis of the reported solve."""
        return self.record.result if self.record is not None else None

    @property
    def final_spec(self) -> Optional[ZeroDimensionalEquilibriumSpec]:
        return self.result.spec if self.result is not None else None

    @property
    def achieved(self) -> Mapping[str, float]:
        return dict(self.record.achieved) if self.record is not None else {}

    @property
    def residuals(self) -> Mapping[str, float]:
        return dict(self.record.residuals) if self.record is not None else {}

    @property
    def iterations(self) -> int:
        return max((r.iteration for r in self.history), default=0)

    def table(self) -> list[dict]:
        """Requested vs achieved per target (achieved ``None`` when the solve did not produce it)."""
        rows = []
        for name, want in self.targets.items():
            got = self.achieved.get(name)
            rows.append({"target": name, "requested": want, "achieved": got,
                         "residual": self.residuals.get(name), "tolerance": self.tolerances.get(name),
                         "control": "CHEASE normalization" if name == self.imposed
                         else (self.assignment[name].name if name in self.assignment else None)})
        return rows


def assign_knobs(spec: ZeroDimensionalEquilibriumSpec, targets: Mapping[str, float],
                 knobs: Mapping[str, Any] | None = None) -> tuple[str, dict[str, TargetKnob]]:
    """Choose the imposed normalization and one bounded knob per remaining target.

    Parameters
    ----------
    spec : ZeroDimensionalEquilibriumSpec
        The request whose source model supplies the knobs [-].
    targets : Mapping[str, float]
        Target name to value, names from :data:`TARGET_DESCRIPTORS` [-].
    knobs : Mapping[str, Any], optional
        Per-target override: a :class:`TargetKnob` or ``(name, lower, upper)``;
        the name must belong to the target's family [-].

    Returns
    -------
    tuple
        The imposed target name and the ``{target: TargetKnob}`` assignment [-].

    Raises
    ------
    ValueError
        A combination without a knob, with its reason; the message starts
        with ``unsupported_target:`` or ``invalid_request:``.
    """
    unknown = sorted(set(targets) - set(TARGET_DESCRIPTORS))
    if unknown:
        raise ValueError(f"unsupported_target: {unknown} are not descriptors the synthesis can be driven "
                         f"to; supported: {sorted(TARGET_DESCRIPTORS)}")
    imposed = next((name for name in _IMPOSABLE if name in targets), "plasma_current")
    free = [name for name in targets if name != imposed]
    families: dict[str, str] = {}
    for name in free:
        family = _FAMILY[name]
        if family in families:
            raise ValueError(
                f"unsupported_target: {families[family]!r} and {name!r} both need the {family.replace('_', ' ')} "
                "knob; one knob cannot set two descriptors"
                + (" (q95 counts as a current-shape target once the plasma current is imposed)"
                   if "q95" in (name, families[family]) else ""))
        families[family] = name
    if "current_shape" in families and spec.current_profile is not None:
        raise ValueError(f"unsupported_target: {families['current_shape']!r} needs the current-profile shape "
                         "exponents, but the spec's current_profile replaces them; there is no knob left")
    assignment = {}
    for name in free:
        family = _FAMILY[name]
        choice = (knobs or {}).get(name, DEFAULT_KNOBS[family])
        if not isinstance(choice, TargetKnob):
            choice = TargetKnob(str(choice[0]), float(choice[1]), float(choice[2]))
        if choice.name not in _FAMILY_KNOBS[family]:
            raise ValueError(f"unsupported_target: {name!r} is moved by {_FAMILY_KNOBS[family]}, not {choice.name!r}")
        if not choice.lower < choice.upper:
            raise ValueError(f"invalid_request: knob {choice.name} needs lower < upper")
        if choice.name == "pressure_fraction" and not (0 <= choice.lower and choice.upper < 1):
            raise ValueError("invalid_request: pressure_fraction bounds must lie in [0, 1)")
        if choice.name != "pressure_fraction" and not choice.lower > 0:
            raise ValueError(f"invalid_request: {choice.name} bounds must be positive")
        assignment[name] = choice
    return imposed, assignment


def synthesize_equilibrium_to_targets(
    spec: ZeroDimensionalEquilibriumSpec, targets: Mapping[str, float], *,
    knobs: Mapping[str, Any] | None = None, config: CHEASEConfig | None = None,
    tolerance: float = _DEFAULT_TOLERANCE, tolerances: Mapping[str, float] | None = None,
    max_iterations: int = 10, jacobian_step: float = 0.05,
    workdir_factory: Callable[[int], Any] | None = None, **synthesis_options: Any,
) -> MultiTargetSynthesisResult:
    """Meet two or three 0D targets by iterating the source model around CHEASE.

    One target is imposed by CHEASE as its normalization; each other target
    moves one bounded knob of the source model (see the module docstring for
    the mapping and the refusals).  The spec's own knob values, clipped into
    the bounds, are the start.  With neither ``plasma_current`` nor ``q95``
    among the targets, the spec's ``plasma_current`` is imposed (and checked
    by the single-target synthesis, not listed in :meth:`~MultiTargetSynthesisResult.table`).

    Parameters
    ----------
    spec : ZeroDimensionalEquilibriumSpec
        Boundary, field and source model; its knob fields are the initial guess [-].
    targets : Mapping[str, float]
        Target name to requested value: ``plasma_current`` [A], ``q95``,
        ``q0``, ``beta_p``, ``beta_n`` [%, as in the descriptors], ``li``
        (the virial l_i) [-].
    knobs : Mapping[str, Any], optional
        Per-target knob override, :class:`TargetKnob` or ``(name, lower, upper)`` [-].
    config : CHEASEConfig, optional
        CHEASE settings for every solve; a coarse ``ns``/``nt`` keeps the
        iteration cheap.  Each solve gets its own working directory under
        ``config.workdir`` (``solve_NN``), or a fresh temporary one [-].
    tolerance : float, optional
        Relative tolerance of every target (numerical convenience) [-].
    tolerances : Mapping[str, float], optional
        Per-target relative tolerances overriding *tolerance* [-].
    max_iterations : int, optional
        Largest number of outer iterations (secant or Newton steps) [-].
    jacobian_step : float, optional
        First secant probe and forward-difference step, as a fraction of each
        knob's range (in ``log`` for the shape exponents) [-].
    workdir_factory : callable, optional
        ``solve_index -> workdir`` overriding the per-solve directories [-].
    **synthesis_options
        Passed to every :func:`~vaft.code.chease_synthesis.synthesize_equilibrium_from_0d`
        call (boundary and pressure-shape tolerances) [-].

    Returns
    -------
    MultiTargetSynthesisResult
        The end state and its reason, the imposed target, the knob assignment,
        every solve's requested/achieved record, the reported solve, and the
        reachable range seen at the knob bounds when a target is out of reach [-].
    """
    targets = {str(k): float(v) for k, v in targets.items()}
    tolerances = {name: float((tolerances or {}).get(name, tolerance)) for name in targets}
    out = MultiTargetSynthesisResult("invalid_request", None, spec, targets, tolerances=tolerances)
    if not targets:
        out.reason = "no targets"
        return out
    try:
        imposed, assignment = assign_knobs(spec, targets, knobs)
    except ValueError as error:
        out.status, out.reason = str(error).split(": ", 1)
        return out
    out.imposed, out.assignment = imposed, assignment
    bad = [name for name, value in targets.items() if not (math.isfinite(value) and (
        value != 0 if name == "plasma_current" else value > (1 if name == "q95" else 0)))]
    if bad or any(not t > 0 for t in tolerances.values()) or max_iterations < 1:
        out.reason = (f"targets {bad} are out of range (q95 > 1, others positive, Ip non-zero)" if bad
                      else "tolerances must be positive and max_iterations at least 1")
        return out
    base = spec
    if "plasma_current" in targets:
        base = replace(base, plasma_current=targets["plasma_current"])
    names = list(assignment)
    loop = _Loop(base, targets, imposed, assignment, tolerances, config, workdir_factory, synthesis_options, out)
    loop.default_tolerance = float(tolerance)
    x0 = [assignment[n].clip(getattr(base, assignment[n].name)) for n in names]
    try:
        if not names:
            loop.check(loop.evaluate(x0, 0, "initial"))
            loop.end("validation_failed", "the imposed target was not met")
        elif len(names) == 1:
            loop.secant(x0[0], max_iterations, jacobian_step)
        else:
            loop.newton(np.asarray(x0), max_iterations, jacobian_step)
    except _Stop:
        pass
    return out


class _Stop(Exception):
    """The loop reached an end state."""


class _Loop:
    def __init__(self, base, targets, imposed, assignment, tolerances, config, workdir_factory, options, out):
        self.base, self.targets, self.imposed, self.assignment = base, targets, imposed, assignment
        self.names = list(assignment)
        self.tolerances, self.config, self.factory, self.options, self.out = (
            tolerances, config, workdir_factory, options, out)
        self.last: TargetIterationRecord | None = None

    # --- one solve -------------------------------------------------------------

    def _config(self, index: int) -> CHEASEConfig:
        from pathlib import Path
        import tempfile

        if self.factory is not None:
            workdir = self.factory(index)
        elif self.config is not None and Path(self.config.workdir) != Path("."):
            workdir = Path(self.config.workdir)/f"solve_{index:02d}"
        else:
            workdir = Path(tempfile.mkdtemp(prefix=f"vaft-0d-targets-{index:02d}-"))
        settings = dict(vars(self.config)) if self.config is not None else {}
        settings["workdir"] = workdir
        return CHEASEConfig(**settings)

    def evaluate(self, x: Sequence[float], iteration: int, purpose: str) -> TargetIterationRecord:
        knob_values = {self.assignment[n].name: float(v) for n, v in zip(self.names, x)}
        spec = replace(self.base, **knob_values)
        index = len(self.out.history)
        kwargs = dict(self.options)
        if self.imposed == "q95":
            kwargs.update(target="q95", q95=self.targets["q95"], q95_tolerance=self.tolerances["q95"])
        else:
            kwargs.update(target="plasma_current", current_tolerance=self.tolerances.get(
                "plasma_current", self.default_tolerance))
        result = _synthesis.synthesize_equilibrium_from_0d(spec, config=self._config(index), **kwargs)
        achieved, residuals = {}, {}
        for name, want in self.targets.items():
            descriptor = TARGET_DESCRIPTORS[name]
            value = (result.achieved or {}).get(descriptor)
            if value is None or not math.isfinite(float(value)):
                continue                                   # absent: never the requested value
            achieved[name] = float(value)
            residuals[name] = float(value)/abs(want) - 1
        record = TargetIterationRecord(index, iteration, purpose, knob_values, dict(self.targets), achieved,
                                       residuals, result.status, result.reason, result)
        self.out.history.append(record)
        self.last = record
        return record

    # --- bookkeeping -----------------------------------------------------------

    def merit(self, record: TargetIterationRecord) -> float:
        if not record.ok or any(n not in record.residuals for n in self.targets):
            return math.inf
        return float(np.sqrt(sum((record.residuals[n]/self.tolerances[n])**2 for n in self.targets)))

    def converged(self, record: TargetIterationRecord) -> bool:
        return record.ok and all(n in record.residuals and abs(record.residuals[n]) <= self.tolerances[n]
                                 for n in self.targets)

    def end(self, status: str, reason: str | None, record: TargetIterationRecord | None = None):
        valid = [r for r in self.out.history if r.ok and math.isfinite(self.merit(r))]
        self.out.status, self.out.reason = status, reason
        self.out.record = record if record is not None else (min(valid, key=self.merit) if valid else None)
        raise _Stop

    def check(self, record: TargetIterationRecord) -> None:
        """End on a failed solve; end converged on a solve that meets every target."""
        if record.status in _CHEASE_FAILURES:
            self.end("chease_failed", f"solve {record.solve} ({record.purpose}, {record.knobs}): "
                     f"{record.status}: {record.reason}")
        if not record.ok:
            self.end("validation_failed", f"solve {record.solve} ({record.purpose}, {record.knobs}): "
                     f"{record.status}: {record.reason}")
        missing = [n for n in self.targets if n not in record.residuals]
        if missing:
            self.end("validation_failed", f"solve {record.solve} did not produce {missing}")
        if self.converged(record):
            self.end("converged", None, record)

    def _range(self, name: str, records) -> tuple:
        values = [r.achieved[name] for r in records if name in r.achieved]
        return (min(values), max(values)) if values else ()

    # --- 1-D: bracketing secant ------------------------------------------------

    def secant(self, x0: float, max_iterations: int, step: float) -> None:
        name = self.names[0]
        knob = self.assignment[name]
        points: list[tuple[float, float, TargetIterationRecord]] = []   # (u, residual, record)

        def solve(u, iteration, purpose):
            record = self.evaluate([knob.from_unit(u)], iteration, purpose)
            self.check(record)
            points.append((u, record.residuals[name], record))

        u0 = knob.to_unit(x0)
        solve(u0, 0, "initial")
        solve(u0 + step if u0 + step <= 1 else u0 - step, 0, "probe")
        stale = 0
        for iteration in range(1, max_iterations + 1):
            bracket = _tightest_bracket(points)
            if bracket is not None:
                ua, fa, ub, fb = bracket
                width = ub - ua
                guess = ub - fb*(ub - ua)/(fb - fa)
                # Bisection safeguard: a guess hugging an end, or a bracket that stopped shrinking.
                if stale >= 2 or not (ua + 0.02*width < guess < ub - 0.02*width):
                    guess, purpose, stale = 0.5*(ua + ub), "bisection", 0
                else:
                    purpose = "regula_falsi"
                solve(guess, iteration, purpose)
                new = _tightest_bracket(points)
                stale = stale + 1 if new[2] - new[0] > 0.5*width else 0
                continue
            (ua, fa, _), (ub, fb, _) = points[-2], points[-1]
            if fa == fb:
                self._unreachable_1d(name, knob, points, "the descriptor does not respond to the knob")
            guess = ub - fb*(ub - ua)/(fb - fa)
            clipped = min(max(guess, 0.0), 1.0)
            if clipped != guess:
                if any(abs(p[0] - clipped) < 1e-12 for p in points):
                    self._unreachable_1d(name, knob, points, None)
                solve(clipped, iteration, "bound")
                continue
            solve(guess, iteration, "secant")
        self.end("max_iterations", f"{name} not within {self.tolerances[name]:g} after {max_iterations} iterations")

    def _unreachable_1d(self, name, knob, points, why):
        lo, hi = self._range(name, [p[2] for p in points])
        self.out.reachable = {name: (lo, hi)}
        nearest = min(points, key=lambda p: abs(p[1]))
        self.end("not_reachable", (why + "; " if why else "") +
                 f"{name} = {self.targets[name]:g} is outside the range {lo:.4g}..{hi:.4g} of the solves; the "
                 f"closest, {nearest[2].achieved[name]:.4g}, is at {knob.name} = {knob.from_unit(nearest[0]):.4g} "
                 f"(bounds [{knob.lower:g}, {knob.upper:g}]), and the residual keeps its sign there")

    # --- 2-D: damped Newton ----------------------------------------------------

    def newton(self, x0: Sequence[float], max_iterations: int, step: float) -> None:
        knobs = [self.assignment[n] for n in self.names]
        scale = np.array([self.tolerances[n] for n in self.names])

        def solve(u, iteration, purpose, trial=False):
            record = self.evaluate([k.from_unit(v) for k, v in zip(knobs, u)], iteration, purpose)
            if trial and not record.ok:
                return record, None                        # a failed trial step is halved, not fatal
            self.check(record)
            return record, np.array([record.residuals[n] for n in self.names])/scale

        u = np.array([k.to_unit(x) for k, x in zip(knobs, x0)])
        _, f = solve(u, 0, "initial")
        for iteration in range(1, max_iterations + 1):
            jacobian = np.empty((2, 2))
            for k in range(2):
                du = step if u[k] + step <= 1 else -step
                probe = u.copy()
                probe[k] += du
                _, fk = solve(probe, iteration, "jacobian")
                jacobian[:, k] = (fk - f)/du
            if not np.all(np.isfinite(jacobian)) or np.linalg.cond(jacobian) > 1e8:
                self.end("not_reachable", "the knobs do not move the targets independently here "
                         f"(finite-difference Jacobian {jacobian.tolist()} is singular)")
            full = -np.linalg.solve(jacobian, f)
            # Knobs whose Newton step leaves the bounds: the linearized root lies outside the box.
            outward = [k for k in range(2) if not 0.0 <= u[k] + full[k] <= 1.0]
            # Project the Newton step onto the box once, then damp along the projected direction.
            direction = np.clip(u + full, 0.0, 1.0) - u
            norm, damping, accepted = float(np.linalg.norm(f)), 1.0, False
            for _ in range(4):
                if np.max(np.abs(damping*direction)) < 1e-6:
                    break
                trial = u + damping*direction
                _, ft = solve(trial, iteration, "newton" if damping == 1.0 else "line_search", trial=True)
                if ft is not None and np.linalg.norm(ft) < norm:
                    u, f, accepted = trial, ft, True
                    break
                damping *= 0.5
            if accepted:
                continue
            if outward:
                ranges = {n: self._range(n, self.out.history) for n in self.names}
                self.out.reachable = ranges
                self.end("not_reachable", "the Newton step leaves the bounds of "
                         + ", ".join(f"{knobs[k].name} [{knobs[k].lower:g}, {knobs[k].upper:g}]" for k in outward)
                         + " and no projected step reduces the residual; the targets lie outside what "
                         "the bounds reach (ranges seen: "
                         + ", ".join(f"{n} {r[0]:.4g}..{r[1]:.4g}" for n, r in ranges.items() if r) + ")")
            self.end("stalled", "no damped Newton step inside the bounds reduces the residual "
                     f"(tolerance-scaled norm {norm:.3g})")
        self.end("max_iterations", f"{self.names} not within tolerance after {max_iterations} iterations")


def _tightest_bracket(points):
    """The narrowest adjacent pair of solved points with a residual sign change, or None."""
    ordered = sorted(points, key=lambda p: p[0])
    best = None
    for (ua, fa, _), (ub, fb, _) in zip(ordered, ordered[1:]):
        if fa*fb < 0 and (best is None or ub - ua < best[2] - best[0]):
            best = (ua, fa, ub, fb)
    return best
