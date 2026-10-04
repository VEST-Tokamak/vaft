"""Direct, solver-independent PF-current fit for a prescribed equilibrium.

The Green functions use full weber and positive right-handed toroidal current.
The input equilibrium is converted to COCOS 11 before its plasma current is
integrated; all returned fluxes are full weber and circuit currents are ampere.
Passive conductors are open circuits in this static free-space calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.constants import mu_0 as MU0
from scipy.optimize import lsq_linear

from vaft.data.equilibrium import EquilibriumData
from vaft.process.electromagnetics import compute_point_response_matrices
from vaft.process._equilibrium_parametric import convert_cocos
from vaft.process.equilibrium import fractional_cell_weights_from_boundary


@dataclass(frozen=True)
class CoilFitResult:
    """A direct inverse fit; flux fields are Wb, fields T, currents A."""

    coil_names: tuple[str, ...]
    currents_A: Mapping[str, float]
    boundary_points_m: np.ndarray
    x_points_m: np.ndarray
    plasma_psi_Wb: np.ndarray
    coil_psi_Wb: np.ndarray
    plasma_br_T: np.ndarray
    plasma_bz_T: np.ndarray
    coil_br_T: np.ndarray
    coil_bz_T: np.ndarray
    residuals: Mapping[str, np.ndarray]
    rms_relative_flux: float
    max_relative_flux: float
    rms_normal_field_T: float | None
    max_saddle_field_T: float | None
    singular_values: np.ndarray
    rank: int
    condition_number: float
    regularization_norm: float
    active_bounds: tuple[str, ...]
    integrated_ip_A: float
    plasma_current_rel_error: float | None
    optimizer_success: bool
    accepted: bool
    status: str


def _coil_sources(machine: Any) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        coils = machine["pf_active.coil"]
    except (KeyError, TypeError) as exc:
        raise ValueError("machine needs pf_active.coil geometry") from exc
    names: list[str] = []
    rs: list[float] = []
    zs: list[float] = []
    turns: list[float] = []
    groups: list[int] = []
    for index in range(len(coils)):
        coil = coils[index]
        name = str(coil.get("name", f"PF{index + 1}")).strip().upper()
        if not name or name in names:
            raise ValueError(f"duplicate or empty PF circuit name {name!r}")
        names.append(name)
        for element_index in range(len(coil["element"])):
            element = coil[f"element.{element_index}"]
            geometry = element["geometry.rectangle"]
            r, z, turn = float(geometry["r"]), float(geometry["z"]), float(element["turns_with_sign"])
            if not np.isfinite([r, z, turn]).all() or r <= 0:
                raise ValueError(f"invalid element geometry or turns in {name}")
            rs.append(r)
            zs.append(z)
            turns.append(turn)
            groups.append(index)
    if not rs or any(not np.any(np.asarray(turns)[np.asarray(groups) == i]) for i in range(len(names))):
        raise ValueError("every PF circuit needs at least one nonzero-turn element")
    return tuple(names), np.asarray(rs), np.asarray(zs), np.asarray(turns), np.asarray(groups)


def _boundary_points(eq: EquilibriumData, count: int) -> tuple[np.ndarray, np.ndarray]:
    if eq.lcfs is None or not eq.lcfs.closed or count < 8:
        raise ValueError("a closed LCFS and at least eight boundary samples are required")
    points = np.asarray(eq.lcfs.points, dtype=float)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    length = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    distance = np.r_[0.0, np.cumsum(length)]
    unique = np.r_[True, np.diff(distance) > 0]
    if distance[-1] <= 0 or np.count_nonzero(unique) < 4:
        raise ValueError("LCFS has zero length or too few distinct points")
    samples = np.linspace(0.0, distance[-1], count, endpoint=False)
    points = np.column_stack((np.interp(samples, distance[unique], points[unique, 0]),
                              np.interp(samples, distance[unique], points[unique, 1])))
    # A central difference at each regular sample gives the surface tangent.
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm == 0):
        raise ValueError("LCFS tangent is undefined")
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / norm[:, None]
    return points, normal


def _plasma_filaments(eq: EquilibriumData, samples_per_axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if any(value is None for value in (eq.r, eq.z, eq.psi, eq.psi_1d, eq.pprime, eq.ffprime)):
        raise ValueError("R/Z/psi grid and psi_1d/pprime/ffprime profiles are required")
    r, z, psi = np.asarray(eq.r), np.asarray(eq.z), np.asarray(eq.psi)
    profile_psi = np.asarray(eq.psi_1d)
    if r.size < 3 or z.size < 3 or psi.shape != (r.size, z.size):
        raise ValueError("plasma grid is too small or has the wrong shape")
    if np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0) or np.any(r <= 0):
        raise ValueError("R and Z axes must increase strictly and R must be positive")
    if not (profile_psi.size == len(eq.pprime) == len(eq.ffprime)) or profile_psi.size < 2:
        raise ValueError("source profiles must share a grid of at least two points")
    order = np.argsort(profile_psi)
    if np.any(np.diff(profile_psi[order]) <= 0):
        raise ValueError("psi_1d profile coordinates must be distinct")
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fraction = fractional_cell_weights_from_boundary(r, z, eq.lcfs.r, eq.lcfs.z, samples_per_axis)
    pp = np.interp(psi, profile_psi[order], np.asarray(eq.pprime)[order])
    ffp = np.interp(psi, profile_psi[order], np.asarray(eq.ffprime)[order])
    # COCOS 11 stores full Wb, while the Grad-Shafranov source uses flux per
    # radian.  Both stored derivatives therefore need a factor of 2*pi.
    # dR*dZ converts current density into the current of each toroidal ring.
    jphi = -2.0 * np.pi * (rr * pp + ffp / (MU0 * rr))
    current = jphi * np.gradient(r)[:, None] * np.gradient(z)[None, :] * fraction
    valid = (fraction > 0) & np.isfinite(current) & (current != 0)
    if not np.any(valid):
        raise ValueError("plasma current distribution is empty")
    return rr[valid], zz[valid], current[valid]


def _response(points: np.ndarray, rs: np.ndarray, zs: np.ndarray, turns: np.ndarray | None = None,
              groups: np.ndarray | None = None, n_groups: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_point_response_matrices(points[:, 0], points[:, 1], rs, zs,
                                           turns=turns, groups=groups, n_groups=n_groups,
                                           components=("psi", "br", "bz"))


def fit_free_boundary_coils(
    equilibrium: EquilibriumData,
    machine: Any,
    *,
    method: str = "flux",
    boundary_samples: int = 96,
    x_points: Sequence[tuple[float, float]] | None = None,
    regularization: float = 0.0,
    current_penalties: Mapping[str, float] | None = None,
    current_bounds: Mapping[str, tuple[float, float]] | None = None,
    flux_weight: float | None = None,
    field_weight: float | None = None,
    current_scale_A: float | None = None,
    samples_per_axis: int = 5,
    flux_tolerance: float = 0.02,
    field_tolerance_T: float = 0.01,
    ip_tolerance: float = 0.05,
) -> CoilFitResult:
    """Fit physical PF circuit currents to a prescribed LCFS in free space.

    ``method='flux'`` fits relative full-weber flux. ``'flux_normal'`` also
    fits B dot n on regular LCFS samples. Explicit X-points always receive
    separate Br=Bz=0 rows. ``flux_weight`` [1/Wb] and ``field_weight`` [1/T]
    scale the objective; omitted weights normalize by target flux span and
    characteristic plasma field. ``regularization`` penalizes current divided
    by ``current_scale_A`` (default |Ip|/number of circuits). Bounds are A.
    ``accepted`` additionally checks physical residual tolerances and rank;
    inspect diagnostics even when the optimizer reports success.
    """
    if method not in ("flux", "flux_normal"):
        raise ValueError("method must be 'flux' or 'flux_normal'")
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    tolerances = (flux_tolerance, field_tolerance_T, ip_tolerance)
    if not np.isfinite(tolerances).all() or any(value < 0 for value in tolerances):
        raise ValueError("residual tolerances must be finite and nonnegative")
    if equilibrium.convention.contradicted:
        raise ValueError("equilibrium COCOS declaration contradicts observed signs")
    if equilibrium.metadata.get("source_type") == "guazzotto_freidberg":
        model = equilibrium.metadata.get("model")
        if model is not None and (model.pressure_pedestal or model.bootstrap_fraction or model.mach_number):
            raise ValueError("Guazzotto surface-current or flow terms are not represented by volume J_phi")
    eq = convert_cocos(equilibrium, 11)
    names, cr, cz, turns, groups = _coil_sources(machine)
    points, normals = _boundary_points(eq, boundary_samples)
    xp = np.empty((0, 2)) if x_points is None else np.asarray(x_points, dtype=float).reshape(-1, 2)
    if xp.size and (not np.isfinite(xp).all() or np.any(xp[:, 0] <= 0)):
        raise ValueError("X-point coordinates must be finite with positive R")
    all_points = np.vstack((points, xp))
    pr, pz, pi = _plasma_filaments(eq, samples_per_axis)
    plasma_psi = np.zeros(len(all_points))
    plasma_br = np.zeros(len(all_points))
    plasma_bz = np.zeros(len(all_points))
    # Chunk the exact plasma Green response so large analytic grids do not
    # materialize an O(boundary_points * plasma_cells) three-field array.
    for start in range(0, len(pi), 1024):
        stop = start + 1024
        psi, br, bz = _response(all_points, pr[start:stop], pz[start:stop])
        plasma_psi += psi @ pi[start:stop]
        plasma_br += br @ pi[start:stop]
        plasma_bz += bz @ pi[start:stop]
    coil_psi, coil_br, coil_bz = _response(all_points, cr, cz, turns, groups, len(names))
    if not all(np.isfinite(v).all() for v in (plasma_psi, plasma_br, plasma_bz, coil_psi, coil_br, coil_bz)):
        raise ValueError("nonfinite Green response; check source/observation separation")
    n = len(points)
    flux_rows = coil_psi[1:n] - coil_psi[0]
    flux_rhs = -(plasma_psi[1:n] - plasma_psi[0])
    flux_scale = abs(float(eq.psi_boundary - eq.psi_axis))
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        raise ValueError("nonzero psi_boundary - psi_axis is required")
    field_scale = max(float(np.hypot(plasma_br[:n], plasma_bz[:n]).max()), 1e-9)
    fw = 1.0 / flux_scale if flux_weight is None else float(flux_weight)
    bw = 1.0 / field_scale if field_weight is None else float(field_weight)
    if not np.isfinite([fw, bw]).all() or fw <= 0 or bw <= 0:
        raise ValueError("objective weights must be finite and positive")
    rows = [fw * flux_rows]
    rhs = [fw * flux_rhs]
    regular = np.ones(n, dtype=bool)
    if method == "flux_normal":
        normal_rows = coil_br[:n] * normals[:, :1] + coil_bz[:n] * normals[:, 1:]
        normal_rhs = -(plasma_br[:n] * normals[:, 0] + plasma_bz[:n] * normals[:, 1])
        if len(xp):
            # A separatrix has two tangents at its saddle.  Its sampled
            # contour can approach either branch, so use saddle Br/Bz there.
            spacing = 2.0 * max(np.max(np.diff(eq.r)), np.max(np.diff(eq.z)))
            regular &= np.min(np.linalg.norm(points[:, None, :] - xp[None, :, :], axis=2), axis=1) > spacing
        if not np.any(regular):
            raise ValueError("no regular LCFS samples remain away from the X-points")
        rows.append(bw * normal_rows[regular])
        rhs.append(bw * normal_rhs[regular])
    if len(xp):
        rows += [bw * coil_br[n:], bw * coil_bz[n:]]
        rhs += [-bw * plasma_br[n:], -bw * plasma_bz[n:]]
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    scale = float(current_scale_A if current_scale_A is not None else
                  (abs(eq.ip) / len(names) if eq.ip else 1000.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("current_scale_A must be positive")
    sv = np.linalg.svd(a * scale, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a * scale))
    condition = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    penalties = np.array([1.0 if current_penalties is None else current_penalties.get(k, 1.0) for k in names], dtype=float)
    if current_penalties is not None and set(current_penalties) - set(names):
        raise ValueError(f"unknown current penalties: {sorted(set(current_penalties) - set(names))}")
    if not np.isfinite(penalties).all() or np.any(penalties < 0):
        raise ValueError("current penalties must be finite and nonnegative")
    if regularization:
        a_solve = np.vstack((a * scale, regularization * np.diag(penalties)))
        b_solve = np.r_[b, np.zeros(len(names))]
    else:
        a_solve, b_solve = a * scale, b
    bounds = current_bounds or {}
    if set(bounds) - set(names):
        raise ValueError(f"unknown current bounds: {sorted(set(bounds) - set(names))}")
    lower = np.array([bounds.get(k, (-np.inf, np.inf))[0] for k in names], dtype=float) / scale
    upper = np.array([bounds.get(k, (-np.inf, np.inf))[1] for k in names], dtype=float) / scale
    if np.any(lower >= upper) or np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError("each current bound must have lower < upper")
    solved = lsq_linear(a_solve, b_solve, bounds=(lower, upper), lsmr_tol="auto")
    currents = solved.x * scale
    full_psi = plasma_psi + coil_psi @ currents
    full_br = plasma_br + coil_br @ currents
    full_bz = plasma_bz + coil_bz @ currents
    flux_residual = full_psi[1:n] - full_psi[0]
    normal_residual = full_br[:n] * normals[:, 0] + full_bz[:n] * normals[:, 1]
    saddle_residual = np.column_stack((full_br[n:], full_bz[n:]))
    rms_flux = float(np.sqrt(np.mean(flux_residual**2)) / flux_scale)
    max_flux = float(np.max(np.abs(flux_residual)) / flux_scale)
    rms_normal = float(np.sqrt(np.mean(normal_residual[regular]**2))) if method == "flux_normal" else None
    max_saddle = float(np.max(np.abs(saddle_residual))) if len(xp) else None
    active = tuple(k for i, k in enumerate(names) if np.isclose(currents[i], lower[i] * scale, atol=1e-6, rtol=0)
                   or np.isclose(currents[i], upper[i] * scale, atol=1e-6, rtol=0))
    ip_error = (abs(float(np.sum(pi)) / eq.ip - 1.0) if eq.ip else None)
    accepted = bool(solved.success and (rank == len(names) or regularization > 0) and rms_flux <= flux_tolerance
                    and (rms_normal is None or rms_normal <= field_tolerance_T)
                    and (max_saddle is None or max_saddle <= field_tolerance_T)
                    and (ip_error is None or ip_error <= ip_tolerance))
    return CoilFitResult(
        names, dict(zip(names, map(float, currents))), points, xp,
        plasma_psi, coil_psi @ currents, plasma_br, plasma_bz,
        coil_br @ currents, coil_bz @ currents,
        {"relative_flux_Wb": flux_residual, "normal_field_T": normal_residual,
         "saddle_field_T": saddle_residual},
        rms_flux, max_flux, rms_normal, max_saddle, sv, rank, condition,
        float(regularization * np.linalg.norm(penalties * solved.x)), active,
        float(np.sum(pi)), ip_error, bool(solved.success), accepted,
        "accepted" if accepted else ("optimizer_failed" if not solved.success else "residual_or_conditioning"),
    )
