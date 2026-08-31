"""Free-boundary PF-coil-current scans with TokaMaker (issue #67).

A TokaMaker-only realization of the free-boundary scan contract, mirroring the
fixed-boundary CHEASE scan conventions (issue #66: control axes, case status
taxonomy, atomic scan/case manifests with config-sha resume, continuation with
explicit break marking) without sharing code yet — the common core is deferred
until both implementations are mature.

    scan(ods,
         controls={"PF6": {"offset_A": [-400, -200, 0, 200]},
                   "PF5_L": {"scale": [1.0, 1.2, 1.4]}},
         mode="product",              # or "zip" for a current trajectory
         hold=("ip", "profile_shape"),
         continuation=True,
         config=TokaMakerConfig(shot=..., time=..., include_vessel=..., ...),
         workdir="fb_scan").run(resume=True)

Every case records BOTH the commanded coil currents (baseline measured
currents with the control applied) and the materialized ones reported by the
solver, the held targets and the achieved global quantities, the scan-grade
topology classification of the solved equilibrium (limited / near-null /
single-null / double-null, X-points, dRsep, limiter contact), and
discontinuity flags relative to the previous converged case. Failed and
non-converged cases stay visible in the manifests; nothing is bridged.

Continuation warm-starts each case from the previous converged case's flux
state inside ONE TokaMaker instance (the solver state carries; after a failure
the last converged state is restored). ``continuation=False`` cold-starts
every case from ``init_psi`` for branch comparison. ``refine_on_failure=N``
inserts up to N bisection steps between the last converged and a failed
commanded state before giving up on that case.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ._oft import get_oft_env, import_oft
from .config import TokaMakerConfig, TokaMakerInputs
from .inputs import prepare_tokamaker_inputs
from .mesh import build_tokamaker_mesh
from .runner import _apply_profiles, _apply_vsc, _configure_tokamaker, _json_safe
from .topology import TopologyReport, classify_boundary

_log = logging.getLogger(__name__)

SCAN_MANIFEST_NAME = "scan_manifest.json"
CASE_MANIFEST_NAME = "case_manifest.json"
SCAN_SCHEMA_VERSION = 1

CONTROL_MODES = ("absolute_A", "offset_A", "scale")
HOLDABLE = ("ip", "profile_shape", "pax", "axis_position")

_NOT_CONVERGED_MARKERS = ("maxits", "matrix solve", "converge")


class CaseStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_CONVERGED = "not_converged"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CoilControl:
    """One scan axis: a coil set varied in one mode."""

    coil: str
    mode: str                                # absolute_A | offset_A | scale
    values: tuple[float, ...]

    def apply(self, baseline_A: float, value: float) -> float:
        if self.mode == "absolute_A":
            return float(value)
        if self.mode == "offset_A":
            return float(baseline_A + value)
        return float(baseline_A * value)


@dataclass(frozen=True)
class ScanCase:
    case_id: str
    index: int
    requested: Mapping[str, Mapping[str, float]]     # {coil: {mode: value}}
    commanded: Mapping[str, float]                   # full {coil_set: A}
    workdir: Path
    config: TokaMakerConfig
    config_sha: str


@dataclass
class FreeBoundaryCaseResult:
    case_id: str
    index: int
    status: CaseStatus
    requested: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    commanded_currents: Mapping[str, float] = field(default_factory=dict)
    materialized_currents: Mapping[str, float] = field(default_factory=dict)
    held: Mapping[str, Any] = field(default_factory=dict)
    achieved: Mapping[str, Any] = field(default_factory=dict)    # solver stats
    topology: Optional[Mapping[str, Any]] = None                 # TopologyReport.to_dict()
    solver_x_points: tuple[tuple[float, float], ...] = ()
    solver_diverted: Optional[bool] = None
    discontinuity: Mapping[str, Any] = field(default_factory=dict)
    continuation_from: Optional[str] = None
    continuation_break: bool = False
    refinement_history: tuple[Mapping[str, Any], ...] = ()
    gfile: Optional[Path] = None
    manifest: Optional[Path] = None
    error: str = ""
    report: Optional[TopologyReport] = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status is CaseStatus.SUCCEEDED


@dataclass
class FreeBoundaryScanResult:
    workdir: Path
    cases: tuple[FreeBoundaryCaseResult, ...]
    manifest: Path

    @property
    def succeeded(self) -> tuple[FreeBoundaryCaseResult, ...]:
        return tuple(case for case in self.cases if case.ok)

    @property
    def failed(self) -> tuple[FreeBoundaryCaseResult, ...]:
        return tuple(case for case in self.cases if not case.ok)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def _config_sha(config: TokaMakerConfig) -> str:
    payload = json.dumps(
        _json_safe(asdict(config)), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _value_slug(value: float) -> str:
    return re.sub(r"[^0-9a-zA-Z+.-]", "", f"{value:g}").replace("+", "")


def _case_id(index: int, requested: Mapping[str, Mapping[str, float]]) -> str:
    parts = [
        f"{coil}_{mode.split('_')[0]}_{_value_slug(value)}"
        for coil, spec in sorted(requested.items())
        for mode, value in spec.items()
    ]
    return f"case_{index:03d}_" + "-".join(parts) if parts else f"case_{index:03d}"


def _parse_controls(controls: Mapping[str, Mapping[str, Sequence[float]]]) -> tuple[CoilControl, ...]:
    axes = []
    for coil, spec in controls.items():
        if not isinstance(spec, Mapping) or len(spec) != 1:
            raise ValueError(
                f"Control for coil {coil!r} must be a single-entry mapping "
                f"{{mode: values}} with mode in {CONTROL_MODES}; got {spec!r}"
            )
        (mode, values), = spec.items()
        if mode not in CONTROL_MODES:
            raise ValueError(
                f"Unknown control mode {mode!r} for coil {coil!r}; "
                f"valid modes: {', '.join(CONTROL_MODES)}"
            )
        values = tuple(float(v) for v in values)
        if not values:
            raise ValueError(f"Control for coil {coil!r} has no values")
        axes.append(CoilControl(coil=str(coil).upper(), mode=mode, values=values))
    if not axes:
        raise ValueError("At least one coil control is required")
    return tuple(sorted(axes, key=lambda axis: axis.coil))


def _status_for_error(exc: Exception) -> CaseStatus:
    message = str(exc).lower()
    if isinstance(exc, ValueError) and any(m in message for m in _NOT_CONVERGED_MARKERS):
        return CaseStatus.NOT_CONVERGED
    return CaseStatus.FAILED


def _axis_of(report: Optional[TopologyReport], stats: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    centroid = stats.get("Ip_centroid")
    if centroid is not None:
        try:
            return float(centroid[0]), float(centroid[1])
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------- #
#  Scan object
# --------------------------------------------------------------------------- #
class FreeBoundaryScan:
    """Materialized scan: inspect ``.cases``, then ``.run()`` or ``.dry_run()``."""

    def __init__(
        self,
        ods: Any,
        *,
        axes: tuple[CoilControl, ...],
        mode: str,
        hold: tuple[str, ...],
        continuation: bool,
        refine_on_failure: int,
        jump_thresholds: Mapping[str, float],
        config: TokaMakerConfig,
        workdir: Path,
        classify_kwargs: Mapping[str, Any],
    ):
        self._ods = ods
        self.axes = axes
        self.mode = mode
        self.hold = hold
        self.continuation = continuation
        self.refine_on_failure = int(refine_on_failure)
        self.jump_thresholds = dict(jump_thresholds)
        self.workdir = workdir
        self.classify_kwargs = dict(classify_kwargs)

        # Baseline inputs at config.time: measured coil currents + held targets.
        base_config = replace(config, workdir=workdir)
        self.base_inputs: TokaMakerInputs = prepare_tokamaker_inputs(ods, base_config)
        self.baseline_currents: dict[str, float] = dict(self.base_inputs.coil_currents)
        for axis in self.axes:
            if axis.coil not in self.baseline_currents:
                raise ValueError(
                    f"Controlled coil {axis.coil!r} is not a coil set of this "
                    "machine model. Available sets: "
                    + ", ".join(sorted(self.baseline_currents))
                )
        self.held: dict[str, Any] = self._resolve_hold(config)
        self.config = replace(config, mesh_file=self.base_inputs.mesh_file)
        self.cases: tuple[ScanCase, ...] = self._materialize_cases()

    def _resolve_hold(self, config: TokaMakerConfig) -> dict[str, Any]:
        unknown = sorted(set(self.hold) - set(HOLDABLE))
        if unknown:
            raise ValueError(
                f"Unknown hold quantities: {', '.join(unknown)}. "
                f"Holdable: {', '.join(HOLDABLE)}"
            )
        held: dict[str, Any] = {}
        if "ip" in self.hold:
            held["ip"] = self.base_inputs.targets["Ip"]
        if "profile_shape" in self.hold:
            held["profile_shape"] = {
                "alpha_f_a": config.alpha_f_a, "alpha_f_b": config.alpha_f_b,
                "alpha_p_a": config.alpha_p_a, "alpha_p_b": config.alpha_p_b,
            }
        if "pax" in self.hold:
            if config.pax is None:
                raise ValueError("hold 'pax' requires TokaMakerConfig.pax to be set")
            held["pax"] = config.pax
        if "axis_position" in self.hold:
            if config.r0_target is None and config.v0_target is None:
                raise ValueError(
                    "hold 'axis_position' requires r0_target and/or v0_target"
                )
            held["axis_position"] = {"r0": config.r0_target, "v0": config.v0_target}
        return held

    def _materialize_cases(self) -> tuple[ScanCase, ...]:
        if self.mode == "product":
            import itertools

            combos = itertools.product(*[
                [(axis, value) for value in axis.values] for axis in self.axes
            ])
            points = [dict(combo) for combo in combos]
        elif self.mode == "zip":
            lengths = {len(axis.values) for axis in self.axes}
            if len(lengths) != 1:
                raise ValueError(
                    "mode='zip' requires every control to carry the same number "
                    f"of values; got lengths {sorted(len(a.values) for a in self.axes)}"
                )
            points = [
                {axis: axis.values[k] for axis in self.axes}
                for k in range(lengths.pop())
            ]
        else:
            raise ValueError(f"Unknown scan mode {self.mode!r}; use 'product' or 'zip'")

        cases = []
        for index, point in enumerate(points):
            requested = {
                axis.coil: {axis.mode: value} for axis, value in point.items()
            }
            commanded = dict(self.baseline_currents)
            for axis, value in point.items():
                commanded[axis.coil] = axis.apply(self.baseline_currents[axis.coil], value)
            case_id = _case_id(index, requested)
            case_dir = self.workdir / case_id
            case_config = replace(
                self.config, workdir=case_dir, coil_currents=dict(commanded)
            )
            cases.append(ScanCase(
                case_id=case_id,
                index=index,
                requested=requested,
                commanded=commanded,
                workdir=case_dir,
                config=case_config,
                config_sha=_config_sha(case_config),
            ))
        return tuple(cases)

    # ----------------------------------------------------------------- #
    def _scan_manifest_payload(self, statuses: Mapping[str, str]) -> dict[str, Any]:
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "solver": "tokamaker",
            "shot": self.base_inputs.shot,
            "time": self.base_inputs.time,
            "mode": self.mode,
            "hold": list(self.hold),
            "held": self.held,
            "continuation": self.continuation,
            "refine_on_failure": self.refine_on_failure,
            "jump_thresholds": self.jump_thresholds,
            "baseline_currents_A": self.baseline_currents,
            "axes": [
                {"coil": axis.coil, "mode": axis.mode, "values": list(axis.values)}
                for axis in self.axes
            ],
            "cases": [
                {
                    "case_id": case.case_id,
                    "index": case.index,
                    "config_sha256": case.config_sha,
                    "status": statuses.get(case.case_id, CaseStatus.PENDING.value),
                }
                for case in self.cases
            ],
        }

    def dry_run(self) -> FreeBoundaryScanResult:
        """Materialize case directories and PENDING manifests without solving."""
        results = []
        for case in self.cases:
            manifest = _write_json_atomic(
                case.workdir / CASE_MANIFEST_NAME,
                self._case_payload_base(case, CaseStatus.PENDING),
            )
            results.append(FreeBoundaryCaseResult(
                case_id=case.case_id, index=case.index, status=CaseStatus.PENDING,
                requested=case.requested, commanded_currents=case.commanded,
                held=self.held, manifest=manifest,
            ))
        manifest = _write_json_atomic(
            self.workdir / SCAN_MANIFEST_NAME,
            self._scan_manifest_payload({c.case_id: CaseStatus.PENDING.value for c in self.cases}),
        )
        return FreeBoundaryScanResult(self.workdir, tuple(results), manifest)

    def _case_payload_base(self, case: ScanCase, status: CaseStatus) -> dict[str, Any]:
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "case_id": case.case_id,
            "index": case.index,
            "status": status.value,
            "config_sha256": case.config_sha,
            "requested": case.requested,
            "commanded_currents_A": case.commanded,
            "held": self.held,
        }

    def _resumable(self, case: ScanCase) -> Optional[FreeBoundaryCaseResult]:
        manifest = case.workdir / CASE_MANIFEST_NAME
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text())
        except Exception:
            return None
        if payload.get("status") != CaseStatus.SUCCEEDED.value:
            return None
        if payload.get("config_sha256") != case.config_sha:
            return None
        gfile_name = payload.get("gfile")
        gfile = case.workdir / gfile_name if gfile_name else None
        if gfile is None or not gfile.is_file():
            return None
        return FreeBoundaryCaseResult(
            case_id=case.case_id,
            index=case.index,
            status=CaseStatus.SUCCEEDED,
            requested=payload.get("requested", case.requested),
            commanded_currents=payload.get("commanded_currents_A", case.commanded),
            materialized_currents=payload.get("materialized_currents_A", {}),
            held=payload.get("held", self.held),
            achieved=payload.get("achieved", {}),
            topology=payload.get("topology"),
            solver_x_points=tuple(
                tuple(p) for p in payload.get("solver_x_points", [])
            ),
            solver_diverted=payload.get("solver_diverted"),
            discontinuity=payload.get("discontinuity", {}),
            continuation_from=payload.get("continuation_from"),
            continuation_break=payload.get("continuation_break", False),
            refinement_history=tuple(payload.get("refinement_history", [])),
            gfile=gfile,
            manifest=manifest,
        )

    # ----------------------------------------------------------------- #
    def run(
        self,
        resume: bool = True,
        on_case: Optional[Callable[[FreeBoundaryCaseResult], None]] = None,
    ) -> FreeBoundaryScanResult:
        """Execute the scan with one TokaMaker instance across all cases."""
        oft = import_oft()
        env = get_oft_env(self.config.nthreads)
        base = self.base_inputs
        if not base.mesh_file.is_file():
            build_tokamaker_mesh(base.geometry, base.mesh_file, self.config)

        results: list[FreeBoundaryCaseResult] = []
        statuses: dict[str, str] = {}
        shot = int(base.shot)
        ctime = int(round(base.time * 1000))

        mygs = oft.TokaMaker(env)
        try:
            _configure_tokamaker(oft, mygs, base, self.config)
            _apply_vsc(mygs, self.config)
            _apply_profiles(oft, mygs, self.config)

            psi_good: Any = None            # flux state of the last converged case
            last_good: Optional[FreeBoundaryCaseResult] = None
            solver_is_warm = False          # solver state == psi_good

            for case in self.cases:
                if resume:
                    reloaded = self._resumable(case)
                    if reloaded is not None:
                        # resumed cases do not warm the live solver chain
                        results.append(reloaded)
                        statuses[case.case_id] = reloaded.status.value
                        last_good = reloaded
                        psi_good = None
                        solver_is_warm = False
                        if on_case is not None:
                            on_case(reloaded)
                        continue

                result = self._run_case(
                    oft, mygs, case, shot, ctime,
                    psi_good=psi_good,
                    solver_is_warm=solver_is_warm,
                    last_good=last_good,
                )
                if result.ok:
                    psi_good = mygs.get_psi(False)
                    solver_is_warm = True
                    last_good = result
                elif psi_good is not None:
                    # drop the diverged iterate before the next case
                    try:
                        mygs.set_psi(psi_good)
                    except Exception:  # pragma: no cover - defensive
                        _log.warning("Could not restore the last converged flux")
                results.append(result)
                statuses[case.case_id] = result.status.value
                _write_json_atomic(
                    self.workdir / SCAN_MANIFEST_NAME,
                    self._scan_manifest_payload(statuses),
                )
                if on_case is not None:
                    on_case(result)
        finally:
            try:
                mygs.reset()
            except Exception:  # pragma: no cover - defensive
                _log.warning("TokaMaker reset failed after the scan", exc_info=True)

        manifest = _write_json_atomic(
            self.workdir / SCAN_MANIFEST_NAME, self._scan_manifest_payload(statuses)
        )
        return FreeBoundaryScanResult(self.workdir, tuple(results), manifest)

    # ----------------------------------------------------------------- #
    def _solve_currents(
        self, mygs, currents: Mapping[str, float], *, cold: bool
    ) -> None:
        """One solve at the given commanded currents (raises on failure)."""
        mygs.set_coil_currents(dict(currents))
        mygs.set_targets(**self.base_inputs.targets)
        if cold:
            cfg = self.config
            mygs.init_psi(cfg.init_r0, cfg.init_z0, cfg.init_a0,
                          cfg.init_kappa, cfg.init_delta)
        mygs.solve()

    def _run_case(
        self, oft, mygs, case: ScanCase, shot: int, ctime: int, *,
        psi_good, solver_is_warm: bool,
        last_good: Optional[FreeBoundaryCaseResult],
    ) -> FreeBoundaryCaseResult:
        case.workdir.mkdir(parents=True, exist_ok=True)
        cold = (not self.continuation) or not solver_is_warm
        continuation_from = (
            last_good.case_id if (self.continuation and not cold and last_good) else None
        )
        refinements: list[dict[str, Any]] = []

        status = CaseStatus.PENDING
        error = ""
        try:
            self._solve_currents(mygs, case.commanded, cold=cold)
            status = CaseStatus.SUCCEEDED
        except Exception as exc:
            status = _status_for_error(exc)
            error = str(exc)
            _log.warning("Scan case %s failed: %s", case.case_id, exc)
            if (
                self.continuation
                and self.refine_on_failure > 0
                and last_good is not None
                and psi_good is not None
            ):
                status, error = self._refine_toward(
                    mygs, case, last_good, psi_good, refinements, error
                )

        result = FreeBoundaryCaseResult(
            case_id=case.case_id,
            index=case.index,
            status=status,
            requested=case.requested,
            commanded_currents=case.commanded,
            held=self.held,
            continuation_from=continuation_from,
            continuation_break=bool(
                self.continuation and last_good is not None and not solver_is_warm
            ),
            refinement_history=tuple(refinements),
            error=error,
        )

        if status is CaseStatus.SUCCEEDED:
            stats = dict(mygs.get_stats())
            materialized = dict(mygs.get_coil_currents()[0])
            try:
                xp_array, diverted = mygs.get_xpoints()
                solver_xp = tuple(
                    (float(p[0]), float(p[1])) for p in np.atleast_2d(xp_array)
                ) if xp_array is not None and len(xp_array) else ()
            except Exception:
                solver_xp, diverted = (), None
            gfile = case.workdir / f"g{shot:06d}.{ctime:05d}"
            mygs.save_eqdsk(
                str(gfile),
                nr=self.config.eqdsk_nr,
                nz=self.config.eqdsk_nz,
                lcfs_pad=self.config.eqdsk_lcfs_pad,
                run_info=f"# {shot} {ctime}ms",
                cocos=self.config.eqdsk_cocos,
            )
            report = classify_boundary(gfile, **self.classify_kwargs)
            result.achieved = stats
            result.materialized_currents = materialized
            result.solver_x_points = solver_xp
            result.solver_diverted = None if diverted is None else bool(diverted)
            result.gfile = gfile
            result.topology = report.to_dict()
            result.report = report
            result.discontinuity = self._discontinuity(last_good, result)

        payload = self._case_payload_base(case, status)
        payload.update({
            "materialized_currents_A": result.materialized_currents,
            "achieved": result.achieved,
            "topology": result.topology,
            "solver_x_points": [list(p) for p in result.solver_x_points],
            "solver_diverted": result.solver_diverted,
            "discontinuity": result.discontinuity,
            "continuation_from": result.continuation_from,
            "continuation_break": result.continuation_break,
            "refinement_history": refinements,
            "gfile": result.gfile.name if result.gfile else None,
            "error": error,
        })
        result.manifest = _write_json_atomic(case.workdir / CASE_MANIFEST_NAME, payload)
        return result

    def _refine_toward(
        self, mygs, case: ScanCase,
        last_good: FreeBoundaryCaseResult, psi_good,
        refinements: list[dict[str, Any]], error: str,
    ) -> tuple[CaseStatus, str]:
        """Bisect between the last converged and the failed commanded currents."""
        origin = dict(last_good.commanded_currents)
        target = dict(case.commanded)
        for step in range(1, self.refine_on_failure + 1):
            try:
                mygs.set_psi(psi_good)
            except Exception:  # pragma: no cover - defensive
                break
            # march half the remaining distance, then retry the target
            midpoint = {
                name: 0.5 * (origin.get(name, value) + value)
                for name, value in target.items()
            }
            entry: dict[str, Any] = {"step": step, "commanded": midpoint}
            try:
                self._solve_currents(mygs, midpoint, cold=False)
                entry["converged"] = True
                origin = midpoint
                psi_good = mygs.get_psi(False)
            except Exception as exc:
                entry["converged"] = False
                entry["error"] = str(exc)
                refinements.append(entry)
                continue
            refinements.append(entry)
            try:
                self._solve_currents(mygs, target, cold=False)
                refinements.append({"step": step, "commanded": target, "converged": True,
                                    "target_retry": True})
                return CaseStatus.SUCCEEDED, ""
            except Exception as exc:
                refinements.append({"step": step, "commanded": target, "converged": False,
                                    "target_retry": True, "error": str(exc)})
                error = str(exc)
        return _status_for_error(ValueError(error)), error

    def _discontinuity(
        self,
        previous: Optional[FreeBoundaryCaseResult],
        current: FreeBoundaryCaseResult,
    ) -> dict[str, Any]:
        if previous is None or not previous.ok:
            return {"reference": None, "flagged": False}
        info: dict[str, Any] = {"reference": previous.case_id, "flagged": False}

        prev_axis = _axis_of(previous.report, previous.achieved)
        curr_axis = _axis_of(current.report, current.achieved)
        if prev_axis and curr_axis:
            jump = float(np.hypot(curr_axis[0] - prev_axis[0], curr_axis[1] - prev_axis[1]))
            info["axis_jump_m"] = jump
            if jump > self.jump_thresholds["axis_m"]:
                info["flagged"] = True

        prev_drsep = (previous.topology or {}).get("d_r_sep")
        curr_drsep = (current.topology or {}).get("d_r_sep")
        if prev_drsep is not None and curr_drsep is not None:
            jump = abs(float(curr_drsep) - float(prev_drsep))
            info["d_r_sep_jump_m"] = jump
            if jump > self.jump_thresholds["d_r_sep_m"]:
                info["flagged"] = True

        prev_topology = (previous.topology or {}).get("topology")
        curr_topology = (current.topology or {}).get("topology")
        info["topology_changed"] = (
            prev_topology is not None and curr_topology is not None
            and prev_topology != curr_topology
        )
        return info


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def scan(
    ods: Any,
    *,
    controls: Mapping[str, Mapping[str, Sequence[float]]],
    mode: str = "product",
    hold: Sequence[str] = ("ip", "profile_shape"),
    continuation: bool = True,
    refine_on_failure: int = 0,
    jump_thresholds: Optional[Mapping[str, float]] = None,
    config: Optional[TokaMakerConfig] = None,
    workdir: Optional[Path | str] = None,
    active_tolerance: float = 2.0e-3,
    near_null_band: float = 5.0e-2,
) -> FreeBoundaryScan:
    """Materialize a TokaMaker free-boundary PF-coil-current scan.

    ``controls`` maps coil-set names (use ``TokaMakerConfig.split_coils`` for
    independent upper/lower halves) to a single ``{mode: values}`` entry with
    mode ``absolute_A`` | ``offset_A`` | ``scale``. ``mode="product"`` crosses
    the axes; ``mode="zip"`` marches them together as one current trajectory
    (the issue-#67 target-assisted continuation without an optimizer).
    """
    if config is None:
        raise ValueError(
            "A TokaMakerConfig is required (shot/time identify the baseline state)"
        )
    root = Path(workdir if workdir is not None else Path(config.workdir) / "fb_scan").expanduser()
    thresholds = {"axis_m": 0.05, "d_r_sep_m": 0.02}
    thresholds.update(dict(jump_thresholds or {}))
    return FreeBoundaryScan(
        ods,
        axes=_parse_controls(controls),
        mode=mode,
        hold=tuple(hold),
        continuation=continuation,
        refine_on_failure=refine_on_failure,
        jump_thresholds=thresholds,
        config=config,
        workdir=root,
        classify_kwargs={
            "active_tolerance": active_tolerance,
            "near_null_band": near_null_band,
        },
    )
