"""Magnetic response split into PF, wall and plasma parts, with an explicit plasma drive (#1795).

The routine eddy stage drives the vessel with the measured Rogowski current
on fixed filaments (:func:`vaft.omas.vest_upstream.build_eddy_ods`).  That is
a VEST production policy, and it makes a question like "do the magnetics
support the Rogowski current?" circular: the measurement under test has
already shaped the wall current the magnetics are compared against.  On
vestserver a free filament fit prefers 0.5-0.9 x the Rogowski current on most
native-era shots (#918, H2); answering whether that is the current or the
wall needs a forward model whose plasma drive is the caller's choice.

This module is that forward model's ODS-level contract:

* :func:`wall_currents_for_drive` integrates the passive currents for coil
  currents from ``pf_active`` and a plasma source history the caller passes
  in -- never ``magnetics.ip``.  An empty source list is the PF-only wall;
  a zero history with ``magnetics.ip`` populated gives the same wall.
* :func:`decompose_magnetic_response` evaluates, at one time and on one
  channel ordering, the measured signal and the PF, PF-driven wall,
  plasma-driven wall and direct plasma contributions, each in the channel's
  own unit (T for B-pol probes, Wb of full poloidal flux for flux loops),
  together with the response matrices behind them, so a caller can refit
  any part without rebuilding geometry.
* :func:`reduced_wall_response` maps those wall columns onto a segment-wise
  wall eigenbasis (:func:`vaft.omas.process_wrapper.compute_wall_mode_basis_ods`),
  the reduced unknowns an inverse study can afford to fit.

Nothing here writes the ODS, with one exception shared with the eddy stage:
an ODS without ``em_coupling`` gets it materialized from its geometry
(:func:`vaft.omas.process_wrapper.ensure_em_coupling`) by the impedance
build.  The response matrices are the same exact
Green's functions :mod:`vaft.omas.vacuum_magnetics` uses
(:func:`vaft.omas.process_wrapper.compute_point_response_matrices_ods`),
with coil columns weighted by ``turns_with_sign`` and passive loops at their
outline centroid, so a decomposition and a vacuum benchmark subtract the
same model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vaft.formula.magnetics import project_poloidal_field
from vaft.ods_access import path_count as _count
from vaft.omas.vacuum_magnetics import FLUX_LOOP, select_vacuum_channels
from vaft.validation.imas import VALIDITY_VALID, validity_mask

__all__ = [
    "VALIDITY_HALF_WIDTH_S",
    "MagneticDecomposition",
    "decompose_magnetic_response",
    "reduced_wall_response",
    "wall_currents_for_drive",
]

#: Half-width of the interval around the evaluation time over which a channel
#: must carry a usable sample [s] -- the EFIT constraint box average's (#433).
VALIDITY_HALF_WIDTH_S = 0.0005


def _pf_currents(ods: Any) -> tuple[np.ndarray, np.ndarray]:
    """``pf_active`` time and coil currents ``(n_coil, n_t)`` [s], [A]."""
    time = np.asarray(ods["pf_active.time"], dtype=float).reshape(-1)
    n_coil = _count(ods, "pf_active.coil")
    if n_coil == 0:
        raise ValueError("ODS carries no pf_active coils")
    coils = np.array(
        [np.asarray(ods[f"pf_active.coil.{i}.current.data"], dtype=float).reshape(-1) for i in range(n_coil)]
    )
    if coils.shape[1] != time.size:
        raise ValueError(f"pf_active coil currents have {coils.shape[1]} samples against {time.size} times")
    return time, coils


def _plasma_history(
    sources: Sequence[Sequence[float]],
    currents: np.ndarray | Sequence[Sequence[float]] | None,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validated ``(n_src, 2)`` positions and ``(n_src, n_t)`` currents on ``time``."""
    points = np.asarray(sources, dtype=float).reshape(-1, 2) if len(sources) else np.zeros((0, 2))
    if points.shape[0] == 0:
        if currents is not None and np.size(currents):
            raise ValueError("plasma_currents given without plasma_sources")
        return points, np.zeros((0, time.size))
    if currents is None:
        raise ValueError(
            "plasma_sources need plasma_currents: the drive is explicit here, "
            "magnetics.ip is a measurement and is never read as one"
        )
    history = np.asarray(currents, dtype=float)
    if history.ndim == 1:
        history = history[None, :]
    if history.shape != (points.shape[0], time.size):
        raise ValueError(
            f"plasma_currents must be (n_sources, n_times) = ({points.shape[0]}, {time.size}) "
            f"on pf_active.time, got {history.shape}"
        )
    if not np.isfinite(history).all():
        raise ValueError("plasma_currents must be finite")
    return points, history


def wall_currents_for_drive(
    ods: Any,
    plasma_sources: Sequence[Sequence[float]] = (),
    plasma_currents: np.ndarray | Sequence[Sequence[float]] | None = None,
    *,
    dt_sub: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Passive-loop currents for the ODS coil programme plus an explicit plasma drive.

    Parameters
    ----------
    ods : ODS
        Carries ``pf_active`` (coil currents and geometry), ``pf_passive``
        (loops, resistances) and ``em_coupling`` (materialized when absent);
        nothing else is written.
    plasma_sources : sequence of (r, z)
        Plasma ring filaments [m]; empty for the PF-only wall.
    plasma_currents : array_like, optional
        ``(n_sources, n_times)`` current of each filament on ``pf_active.time``
        [A].  Required with sources.  ``magnetics.ip`` is never substituted.
    dt_sub : float, optional
        Integration substep [s]; the eddy stage's default when unset.

    Returns
    -------
    time : numpy.ndarray
        ``pf_active.time`` [s].
    wall : numpy.ndarray
        ``(n_loop, n_times)`` passive-loop currents [A], ordered as ``pf_passive.loop``.
    """
    from vaft.omas.process_wrapper import DT_SUB, compute_impedance_matrices_ods
    from vaft.process.electromagnetics import solve_eddy_currents

    time, coils = _pf_currents(ods)
    points, history = _plasma_history(plasma_sources, plasma_currents, time)
    R_mat, L_mat, M_mat = compute_impedance_matrices_ods(ods, [tuple(p) for p in points])
    drive = np.vstack([coils, history]).T
    wall = solve_eddy_currents(R_mat, L_mat, M_mat, drive, time, dt_sub=DT_SUB if dt_sub is None else dt_sub)
    return time, np.asarray(wall, dtype=float).T


@dataclass(frozen=True)
class MagneticDecomposition:
    """The magnetic signals at one time, split by source, on one channel ordering.

    Every vector is ``(n_channels,)`` in the channel's own unit (``units``);
    every response matrix maps source currents [A] to those units.  ``wall_pf``
    and ``wall_plasma`` are the wall currents driven by the coils alone and by
    the plasma drive alone (the eddy problem is linear and starts from rest,
    so they add to the wall); they are ``None`` when the caller supplied the
    wall currents.
    """

    time: float
    channels: tuple[Mapping[str, Any], ...]
    units: tuple[str, ...]
    measured: np.ndarray
    pf: np.ndarray
    wall: np.ndarray
    plasma: np.ndarray
    wall_pf: np.ndarray | None
    wall_plasma: np.ndarray | None
    coil_currents: np.ndarray
    wall_currents: np.ndarray
    plasma_sources: np.ndarray
    plasma_currents: np.ndarray
    response_coil: np.ndarray
    response_wall: np.ndarray
    response_plasma: np.ndarray
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def total(self) -> np.ndarray:
        """PF + wall + plasma: the combined forward response."""
        return self.pf + self.wall + self.plasma

    @property
    def residual(self) -> np.ndarray:
        """Measured minus the combined forward response."""
        return self.measured - self.total

    @property
    def is_flux_loop(self) -> np.ndarray:
        return np.array([row["kind"] == FLUX_LOOP for row in self.channels], dtype=bool)

    def plasma_current(self) -> float:
        """Total plasma source current at ``time`` [A], whatever the representation."""
        return float(np.sum(self.plasma_currents))


def _project(rows: Sequence[Mapping[str, Any]], psi: np.ndarray, b_z: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """Per-channel response rows: flux for loops, the probe's own axis for B-pol probes."""
    out = np.empty_like(psi)
    for i, row in enumerate(rows):
        if row["kind"] == FLUX_LOOP:
            out[i] = psi[i]
        else:
            out[i] = project_poloidal_field(b_r[i], b_z[i], row["poloidal_angle"])
    return out


def _at(time_axis: np.ndarray, values: np.ndarray, t: float) -> np.ndarray:
    """Rows of ``values`` (``(n, n_t)``) linearly interpolated at ``t``; refuses extrapolation."""
    if not (time_axis[0] <= t <= time_axis[-1]):
        raise ValueError(f"time {t} s is outside the source grid {time_axis[0]}..{time_axis[-1]} s")
    return np.array([np.interp(t, time_axis, row) for row in values]) if len(values) else np.zeros(0)


def _measured_at(ods: Any, row: Mapping[str, Any], t0: float, min_validity: int) -> float:
    """A selected channel's measurement at ``t0``, read from valid samples only.

    Selection asks whether *any* sample in the validity window is usable (the
    eddy-benchmark question); this point-in-time read asks the stricter one:
    the two samples bracketing ``t0`` must themselves be valid, else the
    interpolation would return the flagged value -- an integrator that railed
    one sample before ``t0`` gave its rail, not the field (cold review 0.8.0
    delta-absorb-19 F2).  An ODS carrying no validity accepts every sample.
    """
    node = f"magnetics.{row['kind']}.{row['index']}.{'flux' if row['kind'] == FLUX_LOOP else 'field'}"
    time, data = np.asarray(row["time"], dtype=float), np.asarray(row["data"], dtype=float)
    accepted = validity_mask(ods, node, min_validity=min_validity)
    if accepted.size != time.size:
        accepted = np.full(time.size, bool(accepted.all()))
    after = int(np.searchsorted(time, t0))
    bracket = {after} if after < time.size and time[after] == t0 else {after - 1, after}
    if not all(accepted[i] for i in bracket):
        raise ValueError(f"{row['name']} is invalid at {t0} s (a bracketing sample is below validity "
                         f"{min_validity}); pass channels= to exclude it")
    # anti-alias: a point read at t0 between two valid samples, no sample rate is reduced
    return float(np.interp(t0, time[accepted], data[accepted]))


def decompose_magnetic_response(
    ods: Any,
    time: float,
    *,
    plasma_sources: Sequence[Sequence[float]] = (),
    plasma_currents: np.ndarray | Sequence[Sequence[float]] | None = None,
    wall_currents: np.ndarray | None = None,
    channels: Sequence[tuple[str, int]] | None = None,
    min_validity: int | None = None,
    dt_sub: float | None = None,
) -> MagneticDecomposition:
    """Measured magnetics at ``time`` and their PF, wall and plasma parts.

    Parameters
    ----------
    ods : ODS
        Diagnostics plus machine description (``pf_active``, ``pf_passive``,
        ``em_coupling``, ``magnetics``).  Only a missing ``em_coupling`` is written.
    time : float
        Evaluation time [s]; must lie inside the coil and channel grids.
    plasma_sources, plasma_currents
        The explicit plasma drive (see :func:`wall_currents_for_drive`).  It
        sets both the direct plasma field and the plasma-driven wall.
        ``magnetics.ip`` is never read.
    wall_currents : array_like, optional
        ``(n_loop, n_times)`` passive currents on ``pf_active.time`` to use
        instead of integrating them -- e.g. the stored ``pf_passive`` currents
        of a routine product, which the eddy stage writes on ``pf_active.time``.
        Only the shape is checked.  ``wall_pf``/``wall_plasma`` are then ``None``.
    channels : sequence of (kind, index), optional
        Explicit channel selection; every usable probe and flux loop otherwise
        (:func:`vaft.omas.vacuum_magnetics.select_vacuum_channels`).
    min_validity : int, optional
        Validity floor for the selection (its default when unset).
    dt_sub : float, optional
        Eddy integration substep [s].

    Returns
    -------
    MagneticDecomposition
    """
    from vaft.omas.process_wrapper import compute_point_response_matrices_ods

    t0 = float(time)
    pf_time, coils = _pf_currents(ods)
    if not (pf_time[0] <= t0 <= pf_time[-1]):
        raise ValueError(f"time {t0} s is outside the source grid {pf_time[0]}..{pf_time[-1]} s")
    points, history = _plasma_history(plasma_sources, plasma_currents, pf_time)
    # validity is judged around the evaluation time (the EFIT box-average half-width),
    # so a channel invalid there never enters
    select_kwargs: dict[str, Any] = {
        "per_family": None, "channels": channels, "window": (t0 - VALIDITY_HALF_WIDTH_S, t0 + VALIDITY_HALF_WIDTH_S),
    }
    if min_validity is not None:
        select_kwargs["min_validity"] = min_validity
    rows = select_vacuum_channels(ods, **select_kwargs)
    if not rows:
        raise ValueError("no magnetic channel carries usable measured data")
    positions = np.array([[row["r"], row["z"]] for row in rows], dtype=float)
    psi, b_z, b_r = compute_point_response_matrices_ods(
        ods, positions, plasma_points=points if len(points) else None, components=("psi", "bz", "br")
    )
    response = _project(rows, psi, b_z, b_r)
    n_coil, n_loop = coils.shape[0], _count(ods, "pf_passive.loop")
    if response.shape[1] != n_coil + n_loop + len(points):
        raise ValueError(
            f"response has {response.shape[1]} columns, expected {n_coil} coils + {n_loop} loops + {len(points)} plasma"
        )
    g_coil = response[:, :n_coil]
    g_wall = response[:, n_coil : n_coil + n_loop]
    g_plasma = response[:, n_coil + n_loop :]

    wall_pf = wall_plasma = None
    if wall_currents is None:
        _, w_pf = wall_currents_for_drive(ods, dt_sub=dt_sub)
        if len(points):
            _, w_all = wall_currents_for_drive(ods, points, history, dt_sub=dt_sub)
        else:
            w_all = w_pf
        wall_t = _at(pf_time, w_all, t0)
        wall_pf = g_wall @ _at(pf_time, w_pf, t0)
        wall_plasma = g_wall @ (wall_t - _at(pf_time, w_pf, t0))
        source = "integrated from pf_active and the explicit plasma drive"
    else:
        w = np.asarray(wall_currents, dtype=float)
        if w.shape != (n_loop, pf_time.size):
            raise ValueError(f"wall_currents must be (n_loop, n_times) = ({n_loop}, {pf_time.size}), got {w.shape}")
        wall_t = _at(pf_time, w, t0)
        source = "supplied by the caller"

    coil_t = _at(pf_time, coils, t0)
    plasma_t = _at(pf_time, history, t0)
    for row in rows:
        if not (row["time"][0] <= t0 <= row["time"][-1]):
            raise ValueError(f"time {t0} s is outside the {row['name']} grid")
    measured = np.array([_measured_at(ods, row, t0, VALIDITY_VALID if min_validity is None else min_validity)
                         for row in rows])
    if not np.isfinite(measured).all():
        bad = [row["name"] for row, v in zip(rows, measured) if not np.isfinite(v)]
        raise ValueError(f"non-finite measured signal at {t0} s on {bad}; pass channels= to exclude them")
    return MagneticDecomposition(
        time=t0,
        channels=tuple({k: row[k] for k in ("kind", "index", "name", "r", "z", "family")} for row in rows),
        units=tuple(row["unit"] for row in rows),
        measured=measured,
        pf=g_coil @ coil_t,
        wall=g_wall @ wall_t,
        plasma=g_plasma @ plasma_t,
        wall_pf=wall_pf,
        wall_plasma=wall_plasma,
        coil_currents=coil_t,
        wall_currents=wall_t,
        plasma_sources=points,
        plasma_currents=plasma_t,
        response_coil=g_coil,
        response_wall=g_wall,
        response_plasma=g_plasma,
        provenance={
            "plasma_drive": f"{len(points)} explicit filament(s); magnetics.ip not read",
            "wall_currents": source,
            "response": "compute_point_response_matrices_ods (exact elliptic), probes projected on their poloidal_angle",
        },
    )


def reduced_wall_response(
    decomposition: MagneticDecomposition, basis: Any, keep: Sequence[np.ndarray] | None = None
) -> np.ndarray:
    """The decomposition's wall columns in a wall eigenbasis: ``G_wall @ V``.

    ``basis`` is a :class:`vaft.process.wall_modes.WallModeBasis` built on the
    same ``pf_passive`` -- only the element count is checked, so a basis from
    another wall of the same size is not caught here (e.g. :func:`vaft.omas.process_wrapper.compute_wall_mode_basis_ods`);
    ``keep`` selects retained modes per segment.  Returns ``(n_channels, n_modes)``
    in the channel units per unit modal amplitude.
    """
    from vaft.process.wall_modes import reduce_response

    if decomposition.response_wall.shape[1] != basis.n_elements:
        raise ValueError(
            f"basis spans {basis.n_elements} wall elements, the decomposition {decomposition.response_wall.shape[1]}"
        )
    return reduce_response(decomposition.response_wall, basis, keep)
