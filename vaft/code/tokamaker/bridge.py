"""Independent TokaMaker fixed-boundary vacuum-flux PF-current fit.

``get_vfixed`` supplies the required *external* flux after a fixed-boundary
Grad-Shafranov solve. No plasma filament integration or target vacuum field
from the direct fitter enters this route. OFT's flux is per radian and its
Green orientation is opposite VAFT's full-weber ring response; multiply the
OFT samples by ``-2*pi`` before fitting VAFT coil responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import lsq_linear

from vaft.data.equilibrium import EquilibriumData
from vaft.process._equilibrium_coil_fit import _coil_sources, _response

from ._oft import get_oft_env, import_oft
from .config import TokaMakerConfig
from .runner import _apply_profiles, _json_safe


@dataclass(frozen=True)
class VFixedFitResult:
    """Fixed-boundary vacuum-field fit in physical coil A and full Wb."""

    coil_names: tuple[str, ...]
    currents_A: Mapping[str, float]
    boundary_points_m: np.ndarray
    required_vacuum_psi_Wb: np.ndarray
    fitted_vacuum_psi_Wb: np.ndarray
    relative_flux_residual_Wb: np.ndarray
    rms_relative_flux: float
    max_relative_flux: float
    singular_values: np.ndarray
    rank: int
    condition_number: float
    regularization_norm: float
    active_bounds: tuple[str, ...]
    bounds_complete: bool
    optimizer_success: bool
    accepted: bool
    status: str
    fixed_stats: Mapping[str, Any] | None = None
    fixed_profile_mode: str | None = None


def fit_vfixed_samples(
    points_m: np.ndarray,
    vfixed_psi_per_rad: np.ndarray,
    machine: Any,
    *,
    flux_scale_Wb: float,
    regularization: float = 0.0,
    current_scale_A: float = 1000.0,
    current_bounds: Mapping[str, tuple[float, float]] | None = None,
    current_penalties: Mapping[str, float] | None = None,
    flux_tolerance: float = 0.02,
    max_flux_tolerance: float = 0.04,
) -> VFixedFitResult:
    """Fit physical PF currents to TokaMaker ``get_vfixed()`` samples.

    Parameters
    ----------
    points_m : array
        Native ``get_vfixed`` boundary samples, shape ``(n, 2)`` [m].
    vfixed_psi_per_rad : array
        Required external vacuum flux returned by OFT [Wb/rad].
    machine : ODS-like
        ``pf_active`` rectangle geometry with signed turns [-].
    flux_scale_Wb : float
        Nonzero target axis-to-boundary full-weber flux span for residuals [Wb].
    regularization : float, optional
        Tikhonov coefficient on scaled circuit currents [-].
    current_scale_A : float, optional
        Scale of current variables for conditioning [A].
    current_bounds : mapping, optional
        Physical current lower/upper limits per circuit [A].
    current_penalties : mapping, optional
        Nonnegative Tikhonov multiplier per circuit [-].
    flux_tolerance : float, optional
        RMS relative-flux acceptance threshold [-].
    max_flux_tolerance : float, optional
        Maximum relative-flux acceptance threshold [-].

    Returns
    -------
    VFixedFitResult
        Required and fitted external fields, residuals, conditioning and currents [-].

    Processing steps
    ----------------
    Convert OFT flux into VAFT's Green orientation, remove the additive
    boundary gauge, assemble the exact signed-turn coil response, and solve
    bounded regularized least squares. The PF response alone is shared with
    the direct fitter; the required vacuum field comes only from OFT.

    Defaults
    --------
    As a numerical convenience, the current scale is 1000 A. Missing bounds
    are unbounded in the optimization but cannot yield an accepted fit.

    Convention
    ----------
    OFT ``get_vfixed`` and ``eval_green`` use Wb/rad; VAFT's exact ring Green
    kernel uses full Wb with opposite sign. The fitted currents are physical
    amperes in the ``pf_active.turns_with_sign`` convention.

    Applicability
    -------------
    Static axisymmetric vacuum coil response for OFT fixed-boundary samples.

    Limitations
    -----------
    This linear fit does not prove free-boundary closure or hardware feasibility
    without explicit current bounds. The fixed solve's profile accuracy is
    reported separately from its external-field residual.

    Provenance
    ----------
    .. [1] OFT ``TokaMaker.get_vfixed`` and ``TokaMaker.util.eval_green``;
       VAFT ``compute_point_response_matrices`` exact ring response.
    """
    points = np.asarray(points_m, dtype=float)
    vfixed = np.asarray(vfixed_psi_per_rad, dtype=float)
    scalar_values = (flux_scale_Wb, current_scale_A, flux_tolerance, max_flux_tolerance, regularization)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or vfixed.shape != (len(points),) or not np.isfinite(points).all()
            or np.any(points[:, 0] <= 0) or not np.isfinite(vfixed).all()):
        raise ValueError("finite get_vfixed points (n,2) and flux samples (n,) are required")
    if not np.isfinite(scalar_values).all() or flux_scale_Wb <= 0 or current_scale_A <= 0 or any(
        value < 0 for value in (flux_tolerance, max_flux_tolerance, regularization)
    ):
        raise ValueError("flux/current scales must be positive and penalties/tolerances nonnegative")
    names, cr, cz, turns, groups = _coil_sources(machine)
    response, _, _ = _response(points, cr, cz, turns, groups, len(names))
    if not np.isfinite(response).all():
        raise ValueError("nonfinite PF response; check coil separation from sample points")
    required = -2.0 * np.pi * vfixed
    a = response[1:] - response[0]
    b = required[1:] - required[0]
    sv = np.linalg.svd(a * current_scale_A / flux_scale_Wb, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a * current_scale_A / flux_scale_Wb))
    condition = float(sv[0] / sv[-1]) if rank == len(names) and sv[-1] > 0 else float("inf")
    penalties = current_penalties or {}
    if set(penalties) - set(names):
        raise ValueError(f"unknown current penalties: {sorted(set(penalties) - set(names))}")
    penalty = np.array([penalties.get(name, 1.0) for name in names], dtype=float)
    if not np.isfinite(penalty).all() or np.any(penalty < 0):
        raise ValueError("current penalties must be finite and nonnegative")
    bounds = current_bounds or {}
    if set(bounds) - set(names):
        raise ValueError(f"unknown current bounds: {sorted(set(bounds) - set(names))}")
    complete = set(bounds) == set(names) and all(np.isfinite(bounds[name]).all() for name in names)
    lower = np.array([bounds.get(name, (-np.inf, np.inf))[0] for name in names], dtype=float)
    upper = np.array([bounds.get(name, (-np.inf, np.inf))[1] for name in names], dtype=float)
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower >= upper):
        raise ValueError("each current bound must have lower < upper")
    a_scaled = a * current_scale_A / flux_scale_Wb
    b_scaled = b / flux_scale_Wb
    if regularization:
        a_scaled = np.vstack((a_scaled, regularization * np.diag(penalty)))
        b_scaled = np.r_[b_scaled, np.zeros(len(names))]
    solved = lsq_linear(a_scaled, b_scaled, bounds=(lower / current_scale_A, upper / current_scale_A))
    well_posed = np.linalg.matrix_rank(a_scaled) == len(names)
    currents = solved.x * current_scale_A
    fitted = response @ currents
    residual = (fitted[1:] - fitted[0]) - b
    rms = float(np.sqrt(np.mean(residual**2)) / flux_scale_Wb)
    maximum = float(np.max(np.abs(residual)) / flux_scale_Wb)
    active = tuple(name for i, name in enumerate(names) if np.isclose(currents[i], lower[i], atol=1e-6, rtol=0)
                   or np.isclose(currents[i], upper[i], atol=1e-6, rtol=0))
    residual_ok = rms <= flux_tolerance and maximum <= max_flux_tolerance
    accepted = bool(solved.success and complete and residual_ok and well_posed)
    status = ("accepted" if accepted else "optimizer_failed" if not solved.success else
              "residual_or_conditioning" if not residual_ok or not well_posed
              else "bounds_unverified")
    return VFixedFitResult(names, dict(zip(names, map(float, currents))), points, required, fitted,
                           residual, rms, maximum, sv, rank, condition,
                           float(regularization * np.linalg.norm(penalty * solved.x)), active,
                           complete, bool(solved.success), accepted, status)


def fit_free_boundary_coils_vfixed(
    equilibrium: EquilibriumData,
    machine: Any,
    workdir: str | Path,
    *,
    config: TokaMakerConfig | None = None,
    **fit_options: Any,
) -> VFixedFitResult:
    """Solve a TokaMaker fixed-boundary equilibrium and fit its vacuum field.

    Parameters
    ----------
    equilibrium : EquilibriumData
        Target with a closed LCFS, positive canonical Ip, Bt and declared COCOS [-].
    machine : ODS-like
        ``pf_active`` geometry supplying the inverse response [-].
    workdir : path
        New empty directory for OFT working files and the samples [path].
    config : TokaMakerConfig, optional
        Fixed solve resolution, order, threads and power-law profile shape [-].
    **fit_options : any
        Passed to :func:`fit_vfixed_samples` [-].

    Returns
    -------
    VFixedFitResult
        Linear fit and native fixed-solve summary [-].

    Processing steps
    ----------------
    Mesh the target LCFS as a plasma-only domain, solve with OFT's fixed
    boundary setting and profile shape, sample ``get_vfixed``, then fit the
    machine PF response. The solver is reset even when a step raises.

    Defaults
    --------
    As a numerical convenience, the existing adapter's power-law profiles
    are used. Equilibrium profile transfer is added in the next stage.

    Convention
    ----------
    The target's full-weber flux span normalizes the fit. The OFT samples
    are Wb/rad and converted only in :func:`fit_vfixed_samples`.

    Applicability
    -------------
    Static, closed, positive-R LCFS and positive canonical Ip. OFT must be installed.

    Limitations
    -----------
    Until arbitrary-profile transfer is implemented, the fixed solve matches
    the target boundary and Ip but uses power-law source *shape*. This is
    explicitly a fixed-solver cross-check, not yet an analytic profile match.
    The installed OFT API permits only positive Ip targets; signed-current
    orientation support is deferred to the profile-transfer stage.

    Provenance
    ----------
    .. [1] OFT fixed-boundary tutorial and ``TokaMaker.get_vfixed``.
    """
    from vaft.process._equilibrium_parametric import convert_cocos

    eq = convert_cocos(equilibrium, 11)
    if equilibrium.convention.contradicted:
        raise ValueError("equilibrium COCOS declaration contradicts observed signs")
    model = eq.metadata.get("model") if eq.metadata.get("source_type") == "guazzotto_freidberg" else None
    if model is not None and (model.pressure_pedestal or model.bootstrap_fraction or model.mach_number):
        raise ValueError("Guazzotto surface-current or flow terms are unsupported by this static fixed solve")
    if eq.lcfs is None or not eq.lcfs.closed:
        raise ValueError("closed LCFS is required")
    if eq.ip is None or not np.isfinite(eq.ip) or eq.ip <= 0:
        raise ValueError("TokaMaker fixed solve requires finite positive canonical Ip")
    if eq.r0 is None or eq.bt0 is None or eq.magnetic_axis is None:
        raise ValueError("target r0, bt0 and magnetic_axis are required")
    if eq.psi_boundary is None or eq.psi_axis is None:
        raise ValueError("target axis and boundary flux are required")
    flux_scale = abs(float(eq.psi_boundary - eq.psi_axis))
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        raise ValueError("nonzero target full-weber flux span is required")
    contour = np.asarray(eq.lcfs.points, dtype=float)
    if np.any(contour[:, 0] <= 0) or not np.isfinite(contour).all():
        raise ValueError("LCFS must be finite with positive R")
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    if len(contour) < 3:
        raise ValueError("LCFS needs at least three distinct points")
    names = _coil_sources(machine)[0]
    output = Path(workdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = config or TokaMakerConfig()
    oft = import_oft()
    env = get_oft_env(cfg.nthreads)
    solver = oft.TokaMaker(env)
    try:
        mesh = oft.meshing.gs_Domain()
        mesh.define_region("plasma", cfg.dx_plasma, "plasma")
        mesh.add_polygon(contour, "plasma")
        pts, cells, regions = mesh.build_mesh()
        solver.setup_mesh(pts, cells, regions)
        solver.settings.free_boundary = False
        solver.settings.pm = not cfg.quiet
        if cfg.maxits is not None:
            solver.settings.maxits = int(cfg.maxits)
        if cfg.urf is not None:
            solver.settings.urf = float(cfg.urf)
        if cfg.nl_tol is not None:
            solver.settings.nl_tol = float(cfg.nl_tol)
        solver.setup(order=cfg.order, F0=float(cfg.f0 if cfg.f0 is not None else eq.r0 * eq.bt0))
        solver.set_targets(Ip=float(eq.ip))
        _apply_profiles(oft, solver, cfg)
        radius = .5 * (float(np.max(contour[:, 0])) - float(np.min(contour[:, 0])))
        height = .5 * (float(np.max(contour[:, 1])) - float(np.min(contour[:, 1])))
        solver.init_psi(float(eq.magnetic_axis[0]), float(eq.magnetic_axis[1]), radius,
                        height / radius, cfg.init_delta)
        solver.solve()
        points, vfixed = solver.get_vfixed()
        points, vfixed = np.array(points, copy=True), np.array(vfixed, copy=True)
        stats = solver.get_stats()
    finally:
        solver.reset()
    np.savez(output / "vfixed_samples.npz", points_m=points, psi_per_rad=vfixed)
    manifest = {"profile_mode": "power_law", "target_Ip_A": eq.ip,
                "F0_T_m": cfg.f0 if cfg.f0 is not None else eq.r0 * eq.bt0,
                "dx_plasma_m": cfg.dx_plasma, "order": cfg.order,
                "source_shape": {"alpha_f_a": cfg.alpha_f_a, "alpha_f_b": cfg.alpha_f_b,
                                 "alpha_p_a": cfg.alpha_p_a, "alpha_p_b": cfg.alpha_p_b},
                "fixed_stats": stats}
    (output / "fixed_boundary.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    fit_options.setdefault("current_scale_A", abs(float(eq.ip)) / len(names))
    fit = fit_vfixed_samples(points, vfixed, machine, flux_scale_Wb=flux_scale,
                             **fit_options)
    return VFixedFitResult(**{**fit.__dict__, "fixed_stats": stats, "fixed_profile_mode": "power_law"})


__all__ = ["VFixedFitResult", "fit_vfixed_samples", "fit_free_boundary_coils_vfixed"]
