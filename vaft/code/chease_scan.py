"""Fixed-boundary equilibrium scans over CHEASE inputs (issue #66).

``scan_tes`` and ``scan_tokamaker`` vary a field of their solver's config.  That
shape does not fit CHEASE: :class:`~vaft.code.chease.CHEASEConfig` holds
numerical settings -- mesh, relaxation, the q95 constraint -- and none of the
physics a study actually varies.  What CHEASE reads is ``EXPEQ``, and ``EXPEQ``
carries three things: the boundary, ``p'`` (with the edge pressure) and ``FF'``.

So a variation here is expressed in those terms and in the physics they move:
scaling the pressure moves beta, scaling ``FF'`` moves the current profile and
with it ``li``, and the boundary is re-shaped through the Miller parameters
already fitted from the equilibrium itself, so "10% more elongation" means 10%
more than *this* discharge had rather than an absolute number pulled from
nowhere.

Each case is a fresh CHEASE solve in its own directory.  A case that fails to
converge is recorded and the scan continues: a shape excursion that CHEASE
cannot solve is a result about the excursion, not a reason to lose the ten
cases that did solve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .chease import CHEASEConfig, CHEASEResult, _copy_geqdsk, refine_equilibrium


__all__ = [
    "CHEASEScanCase",
    "EquilibriumVariation",
    "apply_equilibrium_variation",
    "scan_chease",
]


@dataclass(frozen=True)
class EquilibriumVariation:
    """One perturbation of a CHEASE input, named by the physics it moves.

    The scales are relative to the source equilibrium, so the defaults are the
    unperturbed case and are worth including in a scan as its control.
    ``elongation_scale`` and ``triangularity_shift`` act on the Miller
    parameters fitted from the source boundary.

    ``current_peaking`` redistributes ``FF'`` toward the axis as
    ``(1 - psi_n)**current_peaking``, renormalized to preserve ``int|FF'|``
    (the absolute integral, so a sign-changing profile keeps its magnitude
    rather than its signed area).  CHEASE rescales the total current anyway, so
    that factor does not change the solve -- it is there to keep the written
    input readable as a redistribution.  What moves ``li`` is the shape.  Scaling ``FF'`` uniformly does not: CHEASE rescales the total
    current (``ncscal``), so a constant factor is precisely the degree of
    freedom it normalizes away, and ``li_3`` comes back unchanged to four
    decimal places whether or not the q95 constraint is on.  Measured on the
    packaged 48224 case, peaking 0 -> 1 -> 2 moves ``li_3`` 0.52 -> 1.01 ->
    1.57 and ``q0`` 1.86 -> 0.81 -> 0.45.
    """

    label: str
    pressure_scale: float = 1.0
    current_peaking: float = 0.0
    elongation_scale: float = 1.0
    triangularity_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.current_peaking < 0.0:
            raise ValueError(
                "current_peaking redistributes FF' as (1 - psi_n)**peaking and "
                "must be non-negative; a negative exponent is unbounded at the "
                f"boundary. Got {self.current_peaking}"
            )
        if self.pressure_scale <= 0.0:
            raise ValueError(f"pressure_scale must be positive; got {self.pressure_scale}")
        if self.elongation_scale <= 0.0:
            raise ValueError(f"elongation_scale must be positive; got {self.elongation_scale}")

    @property
    def reshapes_boundary(self) -> bool:
        """Whether this variation touches the boundary at all."""
        return self.elongation_scale != 1.0 or self.triangularity_shift != 0.0


@dataclass(frozen=True)
class CHEASEScanCase:
    """What one point of a scan produced, converged or not."""

    variation: EquilibriumVariation
    workdir: Path
    result: Optional[CHEASEResult] = None
    error: Optional[str] = None
    #: ``(r0, a, kappa, delta)`` the boundary actually carried into the solve.
    shape: Optional[tuple[float, float, float, float]] = None

    @property
    def converged(self) -> bool:
        """Whether CHEASE actually solved this case.

        ``run_chease`` subprocesses with ``check=False``, so a non-zero exit
        comes back as a :class:`CHEASEResult` rather than an exception -- and
        the most likely failure, a mesh that will not converge, exits 1 and
        writes no refined g-file.  "Did not raise" is therefore not "converged",
        and a caller reading ``result.refined_geqdsk`` on the strength of it
        gets ``None``.
        """
        return self.result is not None and self.result.ok


def _reshape_boundary(
    geqdsk: Any, variation: EquilibriumVariation
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Re-shape the boundary by perturbing its own fitted Miller parameters."""
    from vaft.process._equilibrium_parametric import MillerSurface
    from vaft.process.equilibrium import evaluate_miller, fit_miller_surface

    r_bnd = np.asarray(geqdsk["RBBBS"], dtype=float).ravel()
    z_bnd = np.asarray(geqdsk["ZBBBS"], dtype=float).ravel()
    fit = fit_miller_surface((r_bnd, z_bnd))
    if not fit.converged:
        raise ValueError(
            "the source boundary could not be fitted with a Miller shape "
            f"(normalized rms {fit.normalized_rms_error:.3g}), so a relative "
            "shape variation has nothing to be relative to"
        )
    base = fit.surface
    shaped = MillerSurface(
        base.r,
        base.r0,
        base.z0,
        base.kappa * float(variation.elongation_scale),
        base.delta + float(variation.triangularity_shift),
    )
    # A g-file outline is closed -- its last point repeats its first -- and a
    # consumer doing polygon containment on the result is entitled to that.
    closed = np.isclose(r_bnd[0], r_bnd[-1]) and np.isclose(z_bnd[0], z_bnd[-1])
    count = r_bnd.size - 1 if closed else r_bnd.size
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    new_r, new_z = evaluate_miller(shaped, theta)
    if closed:
        new_r = np.append(new_r, new_r[0])
        new_z = np.append(new_z, new_z[0])
    return new_r, new_z, (shaped.r0, shaped.r, shaped.kappa, shaped.delta)


def apply_equilibrium_variation(
    geqdsk: Any,
    variation: EquilibriumVariation,
    *,
    refit_boundary: bool = False,
) -> tuple[Any, Optional[tuple[float, float, float, float]]]:
    """Return a copy of ``geqdsk`` carrying ``variation``, and its shape.

    Only the three quantities ``EXPEQ`` actually carries are touched -- the
    boundary, the pressure pair and ``FF'``.  ``FPOL`` is deliberately left
    alone: CHEASE recomputes ``F`` from the ``FF'`` it is given, so rewriting
    it here would put a number in the file that the solve does not read and a
    reader might believe.

    ``refit_boundary`` re-evaluates the boundary from its Miller fit even when
    the variation does not change the shape.  :func:`scan_chease` turns it on
    for every case of a scan that contains any shape variation, so that the
    control differs from a shape case by the knob alone.  Without it the
    control keeps the raw outline while a shape case carries the fit residual
    too -- about 1.3% in elongation on the packaged 48224 boundary, which is an
    uncontrolled variable sitting inside a controlled comparison.
    """
    modified = _copy_geqdsk(geqdsk)
    shape: Optional[tuple[float, float, float, float]] = None

    if variation.reshapes_boundary or refit_boundary:
        new_r, new_z, shape = _reshape_boundary(geqdsk, variation)
        modified["RBBBS"] = new_r
        modified["ZBBBS"] = new_z
        modified["NBBBS"] = int(new_r.size)

    if variation.pressure_scale != 1.0:
        scale = float(variation.pressure_scale)
        # p and p' scale together: p' is dp/dpsi and psi is untouched here.
        modified["PRES"] = np.asarray(geqdsk["PRES"], dtype=float) * scale
        modified["PPRIME"] = np.asarray(geqdsk["PPRIME"], dtype=float) * scale

    if variation.current_peaking != 0.0:
        ffprime = np.asarray(geqdsk["FFPRIM"], dtype=float)
        psi_norm = np.linspace(0.0, 1.0, ffprime.size)
        weight = (1.0 - psi_norm) ** float(variation.current_peaking)
        # Renormalize so |FF'| keeps its integral: the written input then reads
        # as a redistribution rather than an amplitude change. CHEASE rescales
        # the total current regardless, so this does not steer the solve.
        before = float(np.trapezoid(np.abs(ffprime), psi_norm))
        after = float(np.trapezoid(np.abs(ffprime * weight), psi_norm))
        if after <= 0.0:
            raise ValueError("current_peaking left no FF' to redistribute")
        modified["FFPRIM"] = ffprime * weight * (before / after)

    return modified, shape


def scan_chease(
    source: Any,
    variations: Sequence[EquilibriumVariation],
    *,
    config: Optional[CHEASEConfig] = None,
    workdir: Path | str | None = None,
    on_case: Optional[Callable[[CHEASEScanCase], None]] = None,
    keep_going: bool = True,
) -> list[CHEASEScanCase]:
    """Re-solve one equilibrium under each of ``variations``.

    ``source`` is anything :func:`~vaft.code.chease.prepare_chease_inputs`
    accepts -- a path, a ``GEQDSK`` or an ODS.  Each case gets its own
    sub-directory named after its variation, so the inputs and the CHEASE log
    of a case that failed are still there to look at.

    ``config`` supplies the numerical settings.  Note that ``CHEASEConfig``'s
    own default of ``nideal=11`` is untuned for VEST and does not converge on
    it; the production settings are ``nideal=6, nw=513, target_psin=0.993,
    relax=0.5``.

    With ``keep_going`` a case that raises is recorded with its message and the
    scan continues, because a variation CHEASE cannot solve is usually the
    interesting part of a scan rather than a reason to abandon it.
    """
    from .chease import _coerce_geqdsk

    if not variations:
        raise ValueError("a scan needs at least one variation")
    labels = [item.label for item in variations]
    if len(set(labels)) != len(labels):
        raise ValueError(f"variation labels must be unique; got {labels}")
    for label in labels:
        # The label becomes a directory name under the scan root.
        if not label or label in (".", "..") or set(label) & set("/\\"):
            raise ValueError(
                f"variation label {label!r} is not a usable directory name; "
                "labels name a case's sub-directory under the scan root"
            )

    base_config = config or CHEASEConfig()
    root = Path(workdir) if workdir is not None else Path(base_config.workdir)
    root.mkdir(parents=True, exist_ok=True)
    geqdsk = _coerce_geqdsk(source)

    # A scan that varies the shape at all compares every case against the same
    # Miller representation, so the control is not the odd one out.
    refit = any(item.reshapes_boundary for item in variations)

    cases: list[CHEASEScanCase] = []
    for variation in variations:
        case_dir = root / variation.label
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            modified, shape = apply_equilibrium_variation(
                geqdsk, variation, refit_boundary=refit
            )
            result = refine_equilibrium(modified, replace(base_config, workdir=case_dir))
            case = CHEASEScanCase(
                variation,
                case_dir,
                result=result,
                shape=shape,
                error=None if result.ok else (
                    f"CHEASE exited {result.returncode}; see {case_dir / 'chease.log'}"
                ),
            )
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            if not keep_going:
                raise
            case = CHEASEScanCase(
                variation, case_dir, error=f"{type(error).__name__}: {error}"
            )
        cases.append(case)
        if on_case is not None:
            on_case(case)
    return cases
