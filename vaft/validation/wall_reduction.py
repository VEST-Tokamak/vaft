"""Reduced-wall order study: full versus reduced wall, and both versus the
measured magnetics (vaft #494, VEST-Tokamak/vfit#10).

The segment-wise eigenbasis (:mod:`vaft.process.wall_modes`, #473) can
represent the wall current in any number of modes; this module answers how
many, and which.  Three layers, kept apart on purpose:

1. **full ↔ reduced** (:func:`order_convergence`): the same integrator drives
   the full 950-loop wall and the reduced circuit; the error of the reduced
   wall current, of the field it makes at the magnetics, on a ring around the
   plasma and on the equilibrium grid, is reported per retained order and
   per selection rule.  This is pure numerics; no measurement enters.
2. **representation order** (:func:`representation_order`): the smallest
   order meeting a set of response tolerances, per rule and per metric, so
   an application that only needs the flux loops can read its own answer.
3. **measurement ↔ model** (:func:`experimental_comparison`): the #190
   plasma-free benchmark path (:mod:`vaft.validation.vacuum_benchmark`)
   evaluated for the full wall and for reduced walls side by side, so the
   reader can tell a vessel-model error (both disagree with the data alike)
   from a reduction error (only the reduced one does).

Metrics only, no verdicts: the tolerances that turn these into a
representation order are the study's, passed in and reported with it.

The selection rules are those of :func:`vaft.process.wall_modes.mode_scores`
(``tau``, ``drive_gain``, ``response_energy``, ``output_weight``), plus
``uniform`` (the same count in every segment) and ``moments`` (the
drive-independent patterns of :func:`vaft.process.wall_modes.moment_patterns`,
which are not eigenmodes and are reported as the enrichment the contract
does not yet include).
"""

from __future__ import annotations

import copy
import time as _clock
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_ORDERS",
    "DEFAULT_RULES",
    "WallReductionError",
    "drive_set",
    "experimental_comparison",
    "find_plasma_free_shots",
    "observation_set",
    "order_convergence",
    "representation_order",
    "wall_system",
]

#: Retained totals the convergence study evaluates by default; 19 is one mode
#: per segment of the VEST wall.
DEFAULT_ORDERS: tuple[int, ...] = (1, 2, 4, 8, 16, 19, 32, 64, 128, 256)
#: Selection rules evaluated by default.
DEFAULT_RULES: tuple[str, ...] = ("tau", "drive_gain", "output_weight", "uniform", "moments")
#: The response metrics :func:`order_convergence` reports, and the reference
#: each is relative to.
METRICS: tuple[str, ...] = (
    "current_l2", "current_dissipation", "probe", "flux_loop",
    "boundary_psi", "boundary_b", "grid_psi",
)


class WallReductionError(ValueError):
    """The ODS or the requested study cannot be evaluated as asked."""


# ---------------------------------------------------------------------------
# The system under study
# ---------------------------------------------------------------------------

def wall_system(
    ods: Any,
    *,
    remap_em_coupling: bool = False,
    gap_factor: float | None = None,
    **basis_options: Any,
) -> dict[str, Any]:
    """The wall matrices, the eigenbasis and the shot's PF drive, once.

    ``R_mat``/``M_mat``/``L_mat`` follow :mod:`vaft.process.electromagnetics`'s
    naming (diagonal resistance, passive-passive inductance, passive-to-coil
    coupling); ``drive`` is the measured coil waveform ``(n_times, n_coils)``
    on ``time``.  ``remap_em_coupling`` rebuilds a coupling that the basis
    refuses (see :func:`vaft.omas.process_wrapper.compute_wall_mode_basis_ods`).
    """
    from vaft.omas.process_wrapper import (
        compute_impedance_matrices_ods,
        compute_wall_mode_basis_ods,
    )

    basis = compute_wall_mode_basis_ods(
        ods, gap_factor=gap_factor, remap_em_coupling=remap_em_coupling, **basis_options
    )
    R_mat, L_mat, M_mat = compute_impedance_matrices_ods(ods, [])
    time = np.asarray(ods["pf_active.time"], dtype=float)
    n_coils = len(ods["pf_active.coil"])
    drive = np.vstack(
        [np.asarray(ods[f"pf_active.coil.{i}.current.data"], dtype=float) for i in range(n_coils)]
    ).T
    if drive.shape[0] != time.size:
        raise WallReductionError("pf_active coil waveforms do not share pf_active.time")
    return {
        "R_mat": R_mat, "M_mat": M_mat, "L_mat": L_mat, "basis": basis,
        "time": time, "drive": drive, "n_coils": n_coils, "n_loops": R_mat.shape[0],
    }


def drive_set(system: Mapping[str, Any], kinds: Sequence[str] = ("shot", "single_coil", "step")) -> dict[str, np.ndarray]:
    """Source waveforms the reduced wall is tested under, ``(n_times, n_coils)`` each.

    ``shot`` is the measured PF programme; ``single_coil`` keeps only the
    coil with the largest excursion (one column of the response at a time);
    ``step`` switches that coil from zero to its peak a fifth of the way into
    the window -- the fast transient a slowly driven study would never see.
    """
    time, drive = system["time"], system["drive"]
    out: dict[str, np.ndarray] = {}
    excursion = np.ptp(drive, axis=0)
    loud = int(np.argmax(excursion)) if excursion.size else 0
    for kind in kinds:
        if kind == "shot":
            out[kind] = drive
        elif kind == "single_coil":
            single = np.zeros_like(drive)
            single[:, loud] = drive[:, loud]
            out[kind] = single
        elif kind == "step":
            step = np.zeros_like(drive)
            onset = time[0] + 0.2 * (time[-1] - time[0])
            step[:, loud] = float(np.max(np.abs(drive[:, loud])) or 1.0) * (time >= onset)
            out[kind] = step
        else:
            raise WallReductionError(f"unknown drive kind {kind!r}")
    return out


def _inside(points_r: np.ndarray, points_z: np.ndarray, outline_r: np.ndarray, outline_z: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon on a closed outline."""
    inside = np.zeros(points_r.size, dtype=bool)
    n = outline_r.size
    j = n - 1
    for i in range(n):
        ri, zi, rj, zj = outline_r[i], outline_z[i], outline_r[j], outline_z[j]
        crosses = (zi > points_z) != (zj > points_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (rj - ri) * (points_z - zi) / (zj - zi) + ri
        inside ^= crosses & (points_r < x)
        j = i
    return inside


def observation_set(
    ods: Any,
    *,
    n_coils: int | None = None,
    per_family: int | None = None,
    window: tuple[float, float] | None = None,
    ring_shrink: float = 0.85,
    ring_points: int = 64,
    grid_shape: tuple[int, int] = (25, 33),
) -> dict[str, Any]:
    """Wall-column response matrices at every observation the study scores.

    ``probe`` and ``flux_loop`` are the usable magnetics selected exactly as
    the #190 benchmark selects them (:func:`vaft.omas.vacuum_magnetics.vacuum_response`),
    with the probe angle projected by the one shared rule; ``boundary_*`` is
    a ring of ``ring_points`` on the limiter outline shrunk by ``ring_shrink``
    toward its centroid (where a plasma boundary lives); ``grid_psi`` is
    ``grid_shape`` points inside the limiter (the equilibrium grid).  Each
    matrix has the wall loops as columns, so ``G @ I_w`` is the wall's own
    contribution -- the quantity a reduced wall must preserve.
    """
    from vaft.formula.magnetics import project_poloidal_field
    from vaft.omas.process_wrapper import compute_point_response_matrices_ods
    from vaft.omas.vacuum_magnetics import FLUX_LOOP, vacuum_response

    if n_coils is None:
        n_coils = len(ods["pf_active.coil"])
    rows, (psi, b_z, b_r, positions) = vacuum_response(ods, per_family=per_family, window=window)
    probe_rows, loop_rows = [], []
    for position, row in enumerate(rows):
        if row["kind"] == FLUX_LOOP:
            loop_rows.append(psi[position, n_coils:])
        else:
            probe_rows.append(project_poloidal_field(b_r[position], b_z[position], row["poloidal_angle"])[n_coils:])
    out: dict[str, Any] = {
        "channels": [(row["kind"], int(row["index"]), row["name"]) for row in rows],
        "probe": np.array(probe_rows) if probe_rows else np.empty((0, psi.shape[1] - n_coils)),
        "flux_loop": np.array(loop_rows) if loop_rows else np.empty((0, psi.shape[1] - n_coils)),
    }

    outline_r = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.r"], dtype=float)
    outline_z = np.asarray(ods["wall.description_2d.0.limiter.unit.0.outline.z"], dtype=float)
    r0, z0 = outline_r.mean(), outline_z.mean()
    ring_r = r0 + ring_shrink * (outline_r - r0)
    ring_z = z0 + ring_shrink * (outline_z - z0)
    pick = np.linspace(0, ring_r.size - 1, min(ring_points, ring_r.size)).astype(int)
    ring = np.column_stack([ring_r[pick], ring_z[pick]])
    psi_b, bz_b, br_b = compute_point_response_matrices_ods(ods, ring)
    out["boundary"] = ring
    out["boundary_psi"] = psi_b[:, n_coils:]
    out["boundary_b"] = np.vstack([br_b[:, n_coils:], bz_b[:, n_coils:]])

    nr, nz = grid_shape
    gr, gz = np.meshgrid(np.linspace(outline_r.min(), outline_r.max(), nr),
                         np.linspace(outline_z.min(), outline_z.max(), nz), indexing="ij")
    keep = _inside(gr.ravel(), gz.ravel(), outline_r, outline_z)
    grid = np.column_stack([gr.ravel()[keep], gz.ravel()[keep]])
    (psi_g,) = compute_point_response_matrices_ods(ods, grid, components=("psi",))
    out["grid"] = grid
    out["grid_psi"] = psi_g[:, n_coils:]
    return out


# ---------------------------------------------------------------------------
# Full versus reduced
# ---------------------------------------------------------------------------

def _relative(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-300))


def _selection(system: Mapping[str, Any], rule: str, M: int, scores: Mapping[str, np.ndarray]):
    """The retained basis ``V`` for a rule and total: ``(V, labels, M_repr)``."""
    from vaft.process import wall_modes as wm

    basis = system["basis"]
    if rule == "moments":
        order = max(1, int(np.ceil(M / system["n_coils"])))
        V = wm.moment_patterns(system["R_mat"], system["M_mat"], system["L_mat"], order)[:, :M]
        return V, tuple(("moments", k) for k in range(V.shape[1])), (V.shape[1],)
    if rule == "uniform":
        per = max(1, int(round(M / len(basis.segments))))
        keep = wm.select_slowest(basis, [per] * len(basis.segments))
    elif rule in scores:
        keep = wm.select_by_score(basis, scores[rule], M)
    else:
        raise WallReductionError(f"unknown selection rule {rule!r}; have {sorted(scores)} + uniform, moments")
    return basis.V(keep), basis.labels(keep), tuple(int(k.size) for k in keep)


def order_convergence(
    ods_or_system: Any,
    *,
    rules: Sequence[str] = DEFAULT_RULES,
    orders: Sequence[int] = DEFAULT_ORDERS,
    drives: Sequence[str] = ("shot", "single_coil", "step"),
    observation: Mapping[str, Any] | None = None,
    scores_drive: str = "shot",
    dt_sub: float = 5.0e-5,
    **system_options: Any,
) -> list[dict[str, Any]]:
    """Full-versus-reduced response error for every (rule, order, drive).

    The full wall is solved once per drive; each reduced wall is the same
    circuit projected on the retained basis and integrated by the same
    routine.  Rankings that need a drive are scored on ``scores_drive`` and
    then tested on every drive, so the ``step`` row measures how a selection
    made for the PF programme transfers to a transient it never saw.

    Every row carries the rule, the retained total ``M_total`` and per-segment
    ``M_repr``, the drive, the current error (Euclidean, dissipation-weighted
    and per segment), the relative error of each observation class and the
    probe peak error, and the reduced solve's wall time.  One row per drive
    with ``rule="full"`` records the reference cost.
    """
    from vaft.process import wall_modes as wm
    from vaft.process.electromagnetics import solve_eddy_currents

    system = ods_or_system if isinstance(ods_or_system, Mapping) and "basis" in ods_or_system \
        else wall_system(ods_or_system, **system_options)
    if observation is None:
        observation = {}
    basis, R_mat, M_mat, L_mat = system["basis"], system["R_mat"], system["M_mat"], system["L_mat"]
    time = system["time"]
    r = np.diag(np.asarray(R_mat, dtype=float))
    all_drives = drive_set(system, tuple(drives) + (() if scores_drive in drives else (scores_drive,)))
    G_probe = observation.get("probe")

    full: dict[str, tuple[np.ndarray, float]] = {}
    for name, waveform in all_drives.items():
        started = _clock.perf_counter()
        I_full = solve_eddy_currents(R_mat, L_mat, M_mat, waveform, time, dt_sub=dt_sub)
        full[name] = (I_full, _clock.perf_counter() - started)
    scores = wm.mode_scores(basis, R_mat, M_mat, L_mat, G=G_probe,
                            drive=all_drives[scores_drive], time=time, dt_sub=dt_sub)

    rows: list[dict[str, Any]] = []
    for name in drives:
        rows.append({"rule": "full", "M_total": basis.n_elements, "M_repr": basis.n_modes(),
                     "drive": name, "cost_s": full[name][1]})
    produced: set[tuple[str, int]] = set()
    for rule in rules:
        for M in orders:
            if M > basis.n_elements:
                continue
            V, labels, M_repr = _selection(system, rule, int(M), scores)
            if (rule, int(V.shape[1])) in produced:      # a saturated rule repeats itself
                continue
            produced.add((rule, int(V.shape[1])))
            ops = wm.combined_operators(V, R_mat, M_mat, L_mat)
            for name in drives:
                I_full, _ = full[name]
                started = _clock.perf_counter()
                _, I_red = wm.solve_reduced_eddy(ops, all_drives[name], time, V=V, dt_sub=dt_sub)
                cost = _clock.perf_counter() - started
                err = I_red - I_full
                row: dict[str, Any] = {
                    "rule": rule, "M_total": int(V.shape[1]), "M_repr": M_repr, "drive": name,
                    "current_l2": _relative(err, I_full),
                    "current_dissipation": float(np.sqrt(np.sum(r * err**2) / max(np.sum(r * I_full**2), 1e-300))),
                    "current_by_segment": {
                        seg.id: float(np.sqrt(np.sum(r[seg.index] * err[:, seg.index]**2)
                                              / max(np.sum(r * I_full**2), 1e-300)))
                        for seg in basis.segments
                    },
                    "tau_min_retained": float(min(basis.tau()[i] for i, lab in enumerate(basis.labels()) if lab in set(labels))) if rule not in ("moments",) else float("nan"),
                    "cost_s": cost,
                }
                for key in ("probe", "flux_loop", "boundary_psi", "boundary_b", "grid_psi"):
                    G = observation.get(key)
                    if G is None or np.size(G) == 0:
                        continue
                    y_full, y_err = I_full @ G.T, err @ G.T
                    row[key] = _relative(y_err, y_full)
                    if key == "probe":
                        row["probe_peak"] = float(np.max(np.abs(y_err)) / max(np.max(np.abs(y_full)), 1e-300))
                rows.append(row)
    return rows


def representation_order(
    rows: Iterable[Mapping[str, Any]],
    tolerances: Mapping[str, float],
    *,
    drive: str = "shot",
) -> dict[str, Any]:
    """The smallest retained order meeting ``tolerances`` -- per rule, per metric and jointly.

    ``tolerances`` maps a metric of :data:`METRICS` to its relative bound.
    For each rule the result lists ``by_metric`` (the smallest ``M_total``
    whose value is within that metric's bound, or ``None``) and ``joint``
    (the smallest order within every bound at once, with its ``M_repr``).
    When the joint order differs materially between metrics the
    application-specific orders are the answer, and the reader sees that
    here rather than in a single number (vfit #10 §6).
    """
    unknown = set(tolerances) - set(METRICS)
    if unknown:
        raise WallReductionError(f"unknown metrics {sorted(unknown)}; choose from {METRICS}")
    by_rule: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("drive") == drive and row.get("rule") != "full":
            by_rule.setdefault(str(row["rule"]), []).append(row)
    out: dict[str, Any] = {"drive": drive, "tolerances": dict(tolerances), "rules": {}}
    for rule, entries in by_rule.items():
        entries = sorted(entries, key=lambda e: int(e["M_total"]))
        by_metric: dict[str, int | None] = {}
        for metric, bound in tolerances.items():
            hit = next((e for e in entries if metric in e and e[metric] <= bound), None)
            by_metric[metric] = None if hit is None else int(hit["M_total"])
        joint = next(
            (e for e in entries if all(metric in e and e[metric] <= bound for metric, bound in tolerances.items())),
            None,
        )
        out["rules"][rule] = {
            "by_metric": by_metric,
            "joint": None if joint is None else {"M_total": int(joint["M_total"]), "M_repr": tuple(joint["M_repr"])},
            "evaluated": [int(e["M_total"]) for e in entries],
        }
    return out


# ---------------------------------------------------------------------------
# Measurement versus model
# ---------------------------------------------------------------------------

def _with_wall_currents(ods: Any, currents: np.ndarray) -> Any:
    """A copy of ``ods`` whose passive loops carry ``currents`` ``(n_times, n_loops)``."""
    working = copy.deepcopy(ods)
    for index in range(currents.shape[1]):
        working[f"pf_passive.loop.{index}.current"] = currents[:, index]
    return working


def experimental_comparison(
    ods: Any,
    selections: Mapping[str, Any],
    *,
    window: tuple[float, float] | None = None,
    per_family: int | None = None,
    min_wall_authority: float | None = None,
    dt_sub: float = 5.0e-5,
    **system_options: Any,
) -> dict[str, Any]:
    """The #190 benchmark for the full wall and for reduced walls, side by side.

    ``selections`` maps a name to a retained basis: a ``keep`` tuple for the
    eigenbasis, or an R-orthonormal matrix ``V`` (an enrichment).  The full
    PF-only wall comes from :func:`vaft.validation.vacuum_benchmark.benchmark_wall_currents`
    on a copy; each reduced wall is the same drive through the projected
    circuit, written into another copy, and both are evaluated by
    :func:`vaft.omas.vacuum_magnetics.vacuum_residual_metrics` over the same
    plasma-free window (the interval and its evidence come from
    :func:`vaft.validation.vacuum_benchmark.plasma_free_interval`; the window
    opens after the solver history :func:`vaft.validation.vacuum_benchmark.solver_history_check` asks for).

    Two blocks per model, and that is the point: ``measurement`` is the
    residual against the data (a vessel-model question) and ``reduction`` is
    the reduced model's distance from the full one at the same channels (a
    truncation question).  ``no_wall`` and the coil-only term are reported so
    the size of the wall term itself is visible.
    """
    from vaft.omas.vacuum_magnetics import (
        DEFAULT_MIN_WALL_AUTHORITY,
        synthetic_vacuum_magnetics,
        vacuum_residual_metrics,
        vacuum_response,
    )
    from vaft.process import wall_modes as wm
    from vaft.validation.vacuum_benchmark import (
        DEFAULT_HISTORY_TIME_CONSTANTS,
        _static_model,
        benchmark_wall_currents,
        plasma_free_interval,
        wall_time_constants,
    )

    authority = DEFAULT_MIN_WALL_AUTHORITY if min_wall_authority is None else float(min_wall_authority)
    interval = plasma_free_interval(ods)
    constants = wall_time_constants(ods)
    if window is None:
        opens = interval.start + DEFAULT_HISTORY_TIME_CONSTANTS * float(constants[0])
        window = (float(opens), float(interval.end))
    full_ods = benchmark_wall_currents(ods, dt_sub=dt_sub)
    system = wall_system(full_ods, **system_options)
    I_full = np.vstack(
        [np.asarray(full_ods[f"pf_passive.loop.{i}.current"], dtype=float) for i in range(system["n_loops"])]
    ).T
    rows, response = vacuum_response(full_ods, per_family=per_family, window=window)

    def evaluate(working: Any) -> tuple[tuple, dict[str, Any]]:
        # The same selection arguments as vacuum_response above, so the two
        # paths select identically by construction (an explicit channel list
        # would re-order the rows and the response bundle would be refused).
        built = synthetic_vacuum_magnetics(working, per_family=per_family, window=window, response=response)
        return built, vacuum_residual_metrics(built, window=window, min_wall_authority=authority)

    full_channels, full_metrics = evaluate(full_ods)
    zero_channels, zero_metrics = evaluate(_with_wall_currents(full_ods, np.zeros_like(I_full)))

    def distance(reduced_channels: tuple) -> dict[str, Any]:
        """Reduced model versus full model at the same channels, over the window."""
        per_kind: dict[str, list[float]] = {}
        worst: tuple[float, str] = (0.0, "")
        for full_c, red_c in zip(full_channels, reduced_channels):
            mask = (full_c.time >= window[0]) & (full_c.time <= window[1]) & full_c.usable
            if mask.sum() < 2:
                continue
            ref = full_c.eddy_term[mask]
            err = red_c.eddy_term[mask] - ref
            rel = _relative(err, ref)
            per_kind.setdefault(full_c.kind, []).append(rel)
            scale = max(float(np.max(np.abs(full_c.measured[mask]))), 1e-300)
            of_reading = float(np.max(np.abs(err)) / scale)
            if of_reading > worst[0]:
                worst = (of_reading, full_c.name)
        return {
            "wall_term_relative": {kind: float(np.sqrt(np.mean(np.square(v)))) for kind, v in per_kind.items()},
            "worst_channel_fraction_of_reading": worst[0],
            "worst_channel": worst[1],
        }

    models: dict[str, Any] = {}
    for name, selection in selections.items():
        if isinstance(selection, np.ndarray):
            V, M_repr = selection, (int(selection.shape[1]),)
        else:
            V, M_repr = system["basis"].V(selection), tuple(int(np.asarray(k).size) for k in selection)
        ops = wm.combined_operators(V, system["R_mat"], system["M_mat"], system["L_mat"])
        _, I_red = wm.solve_reduced_eddy(ops, system["drive"], system["time"], V=V, dt_sub=dt_sub)
        reduced_channels, reduced_metrics = evaluate(_with_wall_currents(full_ods, I_red))
        models[name] = {
            "M_total": int(V.shape[1]), "M_repr": M_repr,
            "measurement": reduced_metrics["summary"],
            "reduction": distance(reduced_channels),
            "current": wm.reconstruction_error(I_full, I_red, system["R_mat"]),
        }
    return {
        "shot": ods.get("dataset_description.data_entry.pulse", None),
        "interval": dict(interval),
        "window": tuple(float(w) for w in window),
        "slowest_time_constant": float(constants[0]),
        "channels": [row["name"] for row in rows],
        "static_model": _static_model(full_ods),
        "basis_digest": system["basis"].digest(),
        "full": {"measurement": full_metrics["summary"]},
        "no_wall": {"measurement": zero_metrics["summary"]},
        "models": models,
    }


# ---------------------------------------------------------------------------
# Finding plasma-free shots
# ---------------------------------------------------------------------------

def find_plasma_free_shots(
    shots: Iterable[int],
    *,
    max_plasma_current: float = 5.0e3,
    min_coil_current: float = 1.0e3,
    raw_source: Any = None,
    loaders: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Classify shots by whether they can serve as PF-only benchmark cases.

    A shot is ``plasma_free`` when the processed plasma current never exceeds
    ``max_plasma_current`` [A] while some PF coil exceeds ``min_coil_current``
    [A]; ``undriven`` when neither does; ``plasma`` when the plasma current
    does; ``daq_missing`` when a required raw signal is absent.  Every entry
    carries the peaks it was judged on.

    The default threshold comes from scanning 180 VEST shots around the
    packaged samples: shots whose plasma current peaks at 20 kA or more are
    unambiguous plasmas (median 50 kA), while coil-driven shots in which no
    plasma formed still show 3--4.5 kA on the compensated Rogowski (the
    eddy-compensation residual of a 10--19 kA PF programme), and a few
    hundred amperes when nothing fired.  A ``plasma_free`` verdict is
    therefore a candidate: :func:`vaft.validation.vacuum_benchmark.plasma_free_interval`
    on the mapped ODS is the judge of where the plasma-free stretch is,
    and it copes with a failed breakdown as it does with a plasma shot.

    The readers are the canonical mappers (:func:`vaft.machine_mapping.magnetics.vfit_plasma_current`,
    :func:`vaft.machine_mapping.pf_active.vfit_pf`); ``loaders`` overrides
    them with ``{"plasma_current": f(shot), "pf": f(shot)}`` for tests and
    archives.  VAFT has no shot catalogue, so the caller names the range.
    """
    from vaft.database.raw import RawSignalUnavailableError

    if loaders is None:
        from vaft.machine_mapping.magnetics import vfit_plasma_current
        from vaft.machine_mapping.pf_active import vfit_pf

        loaders = {
            "plasma_current": lambda shot: vfit_plasma_current(shot, raw_source=raw_source),
            "pf": lambda shot: vfit_pf(shot, raw_source=raw_source),
        }
    out: list[dict[str, Any]] = []
    for shot in shots:
        entry: dict[str, Any] = {"shot": int(shot)}
        try:
            _, ip = loaders["plasma_current"](int(shot))
            _, coils = loaders["pf"](int(shot))
        except RawSignalUnavailableError as reason:
            entry.update({"class": "daq_missing", "reason": str(reason)})
            out.append(entry)
            continue
        ip_peak = float(np.nanmax(np.abs(ip))) if np.size(ip) else 0.0
        coil_peak = float(max((np.nanmax(np.abs(c)) for c in coils if np.size(c)), default=0.0))
        if ip_peak > max_plasma_current:
            klass = "plasma"
        elif coil_peak >= min_coil_current:
            klass = "plasma_free"
        else:
            klass = "undriven"
        entry.update({"class": klass, "plasma_current_peak": ip_peak, "coil_current_peak": coil_peak})
        out.append(entry)
    return out
