"""Configuration, input, and result objects for the TES forward-equilibrium adapter.

TES (Tokamak Equilibrium Solver) is a fixed/free-boundary Grad-Shafranov solver.
Given a machine geometry (limiter), an external-coil set, and global plasma
targets (Ip, Bt, betap), it produces a self-consistent equilibrium and writes a
standard EFIT g-file/a-file plus a ``.RESULT`` summary.

These dataclasses follow the common ``vaft.code.base`` protocol so the TES
adapter reads like the existing EFIT/CHEASE/GPEC adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TESConfig:
    """Runtime configuration and physics targets for a TES forward solve.

    Scalars that default to ``None`` (``ip0_kA``, ``bt0``, ``betap``) are read
    from the ODS at ``time`` when not given explicitly, which is what makes a
    parameter scan (e.g. an Ip scan) a matter of overriding a single field.
    """

    # --- runtime ---
    executable: Optional[str] = None          # explicit rtes; else $TESHOME/bin/rtes, then legacy $RTES
    workdir: Path | str = Path(".")
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    niter: int = 0                            # force a fixed number of Picard loops (0 -> use ERRTOL)
    restart: Optional[str] = None             # restart file passed to ``rtes -r``
    emit_namelist: bool = False               # also write a human-readable namelist next to the C-input

    # --- case identification ---
    shot: Optional[int] = None
    time: Optional[float] = None              # seconds

    # --- constraint source ---
    # "equilibrium": read targets (Ip, betap, Bt) from the equilibrium IDS.
    # "magnetics"  : read Ip from magnetics only and DO NOT touch equilibrium;
    #                betap (and profile shape) must then be supplied explicitly.
    constraint_source: str = "equilibrium"
    # Explicit time-slice index into the chosen source. When set it overrides
    # ``time`` for slice selection (``time`` is then derived from the slice).
    time_index: Optional[int] = None

    # --- control flags ---
    mag_diagnostics: int = 1                  # synthesise magnetic-diagnostic signals if 1
    mse_diagnostics: int = 0

    # --- computational grid ---
    nr: int = 129
    nz: int = 129
    rmin: float = 0.07
    rmax: float = 0.80
    zmin: float = -0.80
    zmax: float = 0.80

    # --- initial plasma guess (R0, Z0, a0) ---
    init_r0: float = 0.30
    init_z0: float = 0.00
    init_a0: float = 0.20

    # --- equilibrium constraint / profile shape ---
    prof_type: int = 0
    ip0_kA: Optional[float] = None            # None -> magnetics.ip at ``time`` [kA]
    major_r: float = 0.40
    bt0: Optional[float] = None               # None -> tf.b_field_tor_vacuum_r / r0 at ``time`` [T]
    betap: Optional[float] = None             # None -> equilibrium beta_pol at ``time``
    betap_type: int = 0                       # 0: <Bp^2>|edge (EFIT-consistent), 1: volume <Bp^2>
    alpha_p_a: float = 1.5
    alpha_p_b: float = 2.0
    alpha_f_a: float = 1.5
    alpha_f_b: float = 2.0

    # --- output resolution ---
    nflux: int = 50
    ntheta: int = 180

    # --- numerics ---
    gps: float = 1.0
    relax_sor: float = 1.1
    relax_shp: float = 1.0
    tikhonov_factor: float = 1.0
    errtol_loop: float = 1.0e-9
    errtol_shape: float = 1.0e-5

    # --- shape control ---
    fix_shape: int = 1
    flux_linkage: tuple[int, float] = (0, 0.0)
    nxpt: int = 2
    xpr: Sequence[float] = (0.30, 0.30)
    xpz: Sequence[float] = (-0.55, 0.55)
    active: Sequence[int] = (1, 1)
    snowflake: Sequence[int] = (0, 0)
    drsep: tuple[int, float] = (0, 0.0)
    dsep: tuple[int, float] = (0, 0.0)
    isor: Sequence[float] = (0.640, 0.110, 0.300, 0.300)
    isoz: Sequence[float] = (0.000, 0.000, 0.550, -0.550)
    grpid: Sequence[int] = tuple(range(1, 19))

    # --- limiter ---
    # None -> read from ods['wall'] limiter outline and clip to the grid.
    # Otherwise an explicit (r_array, z_array) polygon (already inside the grid).
    limiter: Optional[tuple[Sequence[float], Sequence[float]]] = None
    limiter_grid_margin: float = 1.5          # grid cells of margin when clipping limiter to grid

    # --- coils ---
    # Include the passive structure (pf_passive eddy loops) as external coils.
    eddy: bool = False


@dataclass
class TESInputs:
    """Prepared input bundle for a TES run."""

    workdir: Path
    cinput: Path                              # the C-format file ``rtes`` consumes
    ods: Any = None
    namelist: Optional[Path] = None
    files: tuple[Path, ...] = ()


@dataclass
class TESResult:
    """Collected TES run status, output files, and parsed equilibrium."""

    returncode: Optional[int]
    workdir: Path
    gfile: Optional[Path] = None
    afile: Optional[Path] = None
    result_file: Optional[Path] = None
    bndry: Optional[Path] = None
    logs: tuple[Path, ...] = ()
    stdout: str = ""
    stderr: str = ""
    geqdsk: tuple[Any, ...] = ()
    ods: Any = None
    scalars: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0
