"""Configuration, input, and result objects for the TokaMaker forward adapter.

TokaMaker (Open FUSION Toolkit, arXiv:2311.07719) is a free-boundary
Grad-Shafranov solver with a Python API. This adapter drives its *forward*
static mode: prescribed VEST PF coil currents plus global targets (Ip and,
optionally, axis pressure / axis position) produce a self-consistent
free-boundary equilibrium, exported as a standard EFIT g-file.

These dataclasses follow the common ``vaft.code.base`` protocol so the
TokaMaker adapter reads like the existing EFIT/CHEASE/GPEC/TES adapters, even
though the solve happens in-process rather than via a subprocess: the
``returncode`` convention is kept (0 = converged, 1 = failed) with the solver
error message stored in ``TokaMakerResult.error`` (the analogue of ``stderr``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TokaMakerConfig:
    """Runtime configuration and physics targets for a TokaMaker forward solve.

    Scalars that default to ``None`` (``ip``, ``f0``, ``coil_currents`` ...)
    are read from the ODS at ``time`` when not given explicitly, which makes a
    parameter scan (e.g. an Ip scan) a matter of overriding a single field.
    """

    # --- runtime ---
    workdir: Path | str = Path(".")
    nthreads: int = 2                         # OFT_env threads; first construction per kernel wins
    quiet: bool = True                        # suppress TokaMaker per-iteration printing
    # TokaMaker's own maxits default (40) is marginal for VEST forward solves
    # whenever the Ip target strays from the measured current; 100 converges
    # comfortably across +-20% Ip scans on shot 39915. None -> solver default.
    maxits: Optional[int] = 100
    urf: Optional[float] = None               # under-relaxation factor override
    nl_tol: Optional[float] = None            # nonlinear convergence tolerance override

    # --- case identification ---
    shot: Optional[int] = None
    time: Optional[float] = None              # seconds

    # --- constraint source ---
    # "equilibrium": read the Ip target from the equilibrium IDS slice nearest
    #                ``time`` (magnetics fallback).
    # "magnetics"  : read Ip from magnetics only and DO NOT touch equilibrium.
    constraint_source: str = "equilibrium"
    # Explicit time-slice index into the chosen source. When set it overrides
    # ``time`` for slice selection (``time`` is then derived from the slice).
    time_index: Optional[int] = None

    # --- global targets (explicit values always win over the ODS) ---
    ip: Optional[float] = None                # [A]; None -> from ODS per constraint_source
    pax: Optional[float] = None               # [Pa] axis-pressure target (optional)
    ip_ratio: Optional[float] = None          # I_{P,FF'}/I_{P,P'} split, ~ 1/beta_p - 1 (optional)
    r0_target: Optional[float] = None         # magnetic-axis R target [m] (optional)
    v0_target: Optional[float] = None         # magnetic-axis Z target [m] (optional)

    # --- vacuum toroidal field ---
    f0: Optional[float] = None                # F0 = R0*B0 [T·m]; None -> tf.b_field_tor_vacuum_r at ``time``
    bt0: Optional[float] = None               # alternative: Bt [T] at ``major_r`` (f0 wins when both set)
    major_r: float = 0.40

    # --- flux-function profiles: (1 - psi_hat^a)^b power laws ---
    alpha_f_a: float = 1.5                    # FF' inner exponent
    alpha_f_b: float = 2.0                    # FF' outer exponent
    alpha_p_a: float = 4.0                    # P'  inner exponent
    alpha_p_b: float = 1.0                    # P'  outer exponent
    nprof: int = 40                           # sample count for the profile tables

    # --- initial plasma guess for init_psi (R0, Z0, a, kappa, delta) ---
    init_r0: float = 0.35
    init_z0: float = 0.00
    init_a0: float = 0.20
    init_kappa: float = 1.5
    init_delta: float = 0.0

    # --- finite elements ---
    order: int = 2                            # Lagrange polynomial degree (2-4)

    # --- mesh / geometry ---
    # Explicit mesh cache; None -> workdir/vest_gs_mesh_<geometry hash>.h5.
    # Pin this for scans so every point shares one mesh build.
    mesh_file: Optional[Path | str] = None
    dx_plasma: float = 0.015                  # target edge length [m] inside the limiter
    dx_coil: float = 0.010                    # target edge length [m] in coil regions
    dx_vacuum: float = 0.05                   # target edge length [m] in the vacuum/boundary region
    # None -> read from ods['wall'] limiter outline.
    # Otherwise an explicit (r_array, z_array) polygon.
    limiter: Optional[tuple[Sequence[float], Sequence[float]]] = None
    # None -> interpolate pf_active coil currents [A] at ``time``.
    # Otherwise an explicit {coil_name: amps} mapping (names as in the geometry).
    coil_currents: Optional[Mapping[str, float]] = None
    exclude_coils: tuple[str, ...] = ()       # coil (set) names to drop from mesh and currents

    # --- vessel seam (reserved for time-dependent/eddy work; unused in v1) ---
    # A static solve assigns no current to conductor regions, so including the
    # vessel cannot change the answer here — it only inflates the mesh.
    include_vessel: bool = False
    dx_conductor: float = 0.02
    eta_vessel: float = 7.4e-7                # [Ohm·m]

    # --- gEQDSK output ---
    eqdsk_nr: int = 129
    eqdsk_nz: int = 129
    eqdsk_lcfs_pad: float = 0.01
    # COCOS 2 matches the VEST EFIT g-files and the ascending-psi assumption in
    # vaft.data.eqdsk (TokaMaker's own default is 7; only 2 and 7 are accepted).
    eqdsk_cocos: int = 2


@dataclass
class TokaMakerInputs:
    """Prepared input bundle for a TokaMaker run."""

    workdir: Path
    geometry: dict                            # {"limiter": [[r,z],...], "coils": {name: {...}}}
    mesh_file: Path                           # resolved cache target (may not exist yet)
    mesh_exists: bool
    coil_currents: dict[str, float]           # [A] per coil set at ``time``
    targets: dict[str, float]                 # kwargs for TokaMaker.set_targets (Ip, pax, ...)
    f0: float                                 # F0 = R0*B0 [T·m]
    shot: int
    time: float                               # seconds
    ods: Any = None
    files: tuple[Path, ...] = ()


@dataclass
class TokaMakerResult:
    """Collected TokaMaker run status, output files, and parsed equilibrium."""

    returncode: Optional[int]                 # 0 converged, 1 failed, None = collect-only
    workdir: Path
    gfile: Optional[Path] = None
    stats_file: Optional[Path] = None         # tokamaker_result.json sidecar
    mesh_file: Optional[Path] = None
    error: str = ""                           # solver error message on failure
    logs: tuple[Path, ...] = ()
    geqdsk: tuple[Any, ...] = ()
    ods: Any = None
    scalars: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0
