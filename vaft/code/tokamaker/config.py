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

    # --- vessel conductors (time-dependent / eddy work) ---
    # A static solve assigns no current to conductor regions (all eddy terms in
    # TokaMaker are gated on dt > 0), so including the vessel leaves static
    # results physically unchanged — it only refines the mesh. The evolution
    # and stability entry points REQUIRE include_vessel=True.
    include_vessel: bool = False
    dx_conductor: float = 0.02                # per-region cap [m]; actual dx = clamp(thickness)
    dx_conductor_min: float = 0.004           # per-region floor [m] (thin-strip mesh cost guard)
    # SUS316LN; exactly reproduces the packaged pf_passive W2-W10 loop resistances
    # via R = 2*pi*R*eta/A. W1 carries a black-box per-loop calibration (issue #191)
    # and gets this uniform default unless overridden.
    eta_vessel: float = 7.8e-7                # [Ohm·m]
    vessel_eta: Optional[Mapping[str, float]] = None   # per segment/region override [Ohm·m]
    vessel_noncontinuous: tuple[str, ...] = ()         # regions with zero net toroidal current
    exclude_vessel_segments: tuple[str, ...] = ("W11",)  # 0.1 mm tungsten tiles: not structure
    # Minimum clearance enforced between conductor regions [m]. The filament
    # segments physically abut (and slightly overlap at corner joints); meshing
    # needs disjoint region polygons with no T-junctions, so every region is
    # shrunk by vessel_gap/2 per side and vertical runs are clamped out of
    # horizontal bands. Must stay well above gs_Domain's 1e-4 merge threshold.
    vessel_gap: float = 3.0e-4

    # --- vertical stability control (optional) ---
    # Name of one pf_active coil (e.g. "PF9") whose upper/lower rectangles become
    # their OWN coil sets (<name>_U/<name>_L) wired as a Vertical Stability Coil
    # pair with gains +1/-1; the virtual '#VSC' amplitude is regularized to 0.
    vsc_coil: Optional[str] = None
    vsc_weight: float = 1.0e-2                # coil_reg_term weight on '#VSC'

    # --- quasi-static evolution ---
    evolve_times: Optional[Sequence[float]] = None   # explicit slice times [s]; wins over start/end/dt
    evolve_start: Optional[float] = None             # else np.arange(start, end, dt)
    evolve_end: Optional[float] = None
    evolve_dt: Optional[float] = None                # [s]; times must round to distinct integer ms
    evolve_on_failure: str = "continue"              # "continue" (keep last converged psi0) | "stop"
    evolve_vacuum: bool = False                      # vac_solve: no plasma, no targets, no g-files
    # False disables the set_psi_dt wall term: every slice becomes an
    # independent static solve (the "coil-only" control of the vacuum benchmark).
    evolve_eddy: bool = True
    # (r, z) points where B/psi are evaluated after each step (vacuum benchmark).
    evolve_field_probes: Optional[Sequence[tuple[float, float]]] = None

    # --- stability eigenvalue solves ---
    wall_neigs: int = 8                       # eig_wall mode count
    td_neigs: int = 8                         # eig_td mode count
    td_omega: float = -1.0e4                  # eig_td ARPACK shift [1/s]
    td_include_bounds: bool = False
    td_damping_scale: float = -1.0            # <0 disables artificial plasma damping

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


@dataclass
class TokaMakerEvolutionInputs:
    """Prepared inputs for a quasi-static evolution across a shot."""

    base: TokaMakerInputs                     # from prepare_tokamaker_inputs at times[0]
    times: tuple[float, ...]                  # strictly increasing slice times [s]
    coil_waveforms: Mapping[str, Any]         # {coil_set: I(times) [A]} aligned arrays
    ip_targets: Any                           # Ip(times) [A]; zeros in vacuum mode
    vacuum: bool = False                      # vac_solve mode (no plasma/targets/g-files)


@dataclass
class TokaMakerStepRecord:
    """One quasi-static evolution step."""

    index: int
    time: float                               # [s]
    converged: bool
    error: str = ""
    gfile: Optional[Path] = None
    stats: Mapping[str, Any] = field(default_factory=dict)
    coil_currents_A: Mapping[str, float] = field(default_factory=dict)
    vessel_currents_A: Mapping[str, float] = field(default_factory=dict)  # net I per conductor region
    probe_fields: Mapping[str, Any] = field(default_factory=dict)         # {"br":[], "bz":[], "psi":[]}


@dataclass
class TokaMakerEvolutionResult:
    """Collected quasi-static evolution: per-step records plus the merged IDS."""

    returncode: Optional[int]                 # 0 = every step converged, 1 = any failure
    workdir: Path
    times: tuple[float, ...] = ()
    steps: tuple[TokaMakerStepRecord, ...] = ()
    gfiles: tuple[Path, ...] = ()             # converged plasma slices, time order
    sidecar_file: Optional[Path] = None       # tokamaker_evolution.json
    mesh_file: Optional[Path] = None
    error: str = ""
    ods: Any = None                           # merged multi-slice equilibrium IDS
    scalars: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class TokaMakerStabilityResult:
    """Wall eigenmodes and/or vertical-stability growth rate."""

    returncode: Optional[int]                 # 0 = solve/eig succeeded, 1 = failed
    workdir: Path
    tau_wall_s: tuple[float, ...] = ()        # wall L/R times [s], descending
    gamma_s: Optional[float] = None           # vertical growth rate [1/s]; > 0 unstable
    eig_file: Optional[Path] = None           # .npz with eig_vals/eig_vecs/mesh arrays
    stats_file: Optional[Path] = None         # tokamaker_stability.json sidecar
    gfile: Optional[Path] = None              # underlying equilibrium (eig_td path)
    error: str = ""
    scalars: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0
