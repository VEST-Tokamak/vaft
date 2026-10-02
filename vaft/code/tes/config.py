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
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ..base import RunOutcome

if TYPE_CHECKING:
    from ..execution import ExecutionBackend


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
    # Default: no shape control. FIX_SHAPE 0 makes TES keep the prescribed coil
    # currents (only its vertical stabilizer acts), so the boundary is whatever
    # the measured PF state produces: on VEST that is usually a plasma limited
    # on the inboard (center-stack) face. The legacy deck instead re-fitted all
    # 18 coil groups each iteration toward a double null at (0.30, +-0.55) with
    # an iso-flux point 5 mm off the inboard wall, which forbids a limited
    # state (issue #1469). It remains available as ``legacy_double_null()``.
    fix_shape: int = 0
    flux_linkage: tuple[int, float] = (0, 0.0)
    nxpt: int = 0
    xpr: Sequence[float] = ()
    xpz: Sequence[float] = ()
    active: Sequence[int] = ()
    snowflake: Sequence[int] = ()
    drsep: tuple[int, float] = (0, 0.0)
    dsep: tuple[int, float] = (0, 0.0)
    isor: Sequence[float] = ()
    isoz: Sequence[float] = ()
    grpid: Sequence[int] = ()

    # --- limiter ---
    # None -> read from ods['wall'] limiter outline and clip to the grid.
    # Otherwise an explicit (r_array, z_array) polygon, used verbatim (already
    # inside the grid, no resampling).
    limiter: Optional[tuple[Sequence[float], Sequence[float]]] = None
    limiter_grid_margin: float = 1.5          # grid cells of margin when clipping limiter to grid
    # TES tests the boundary flux only AT the listed limiter points, never along
    # the edges between them. The canonical VEST center-stack face is a single
    # edge from Z = -0.575 to +0.575 m, so without resampling an inboard
    # midplane contact is invisible to TES. Edges of the ODS limiter are split
    # to at most this length [m]; None keeps the polygon vertices only. Edges
    # the grid clip creates (across the chamber necks) are not wall and stay
    # whole.
    limiter_spacing: Optional[float] = 0.01

    # --- coils ---
    # How pf_active becomes TES coil rows. TES treats every row outside its
    # grid as ONE filament at (R, Z), so the row layout is the field model.
    # "elements": one filament per pf_active element (ampere-turns, one coil
    #             group each). This follows the real winding; PF1 alone has 158.
    # "legacy"  : the ported VEST_tes layout, PF1/PF2 lumped into 5+5 blocks
    #             and the other coils into one row per half (18 groups). The
    #             PF1 blocks are 0.24 m apart only ~0.08 m from the inboard
    #             wall, and their ripple puts a spurious field null at
    #             R ~ 0.13 m on the midplane of 39915 @ 325 ms (issue #1469).
    #             Shape control (fix_shape=1) needs these 18 groups: TES caps a
    #             group at 10 rows.
    coil_model: str = "elements"
    # pf_active coils TES's shape control may re-fit (fix_shape=1 with
    # coil_model="elements"). Each is written as one row per up/down half in
    # its own group, ids 1..k in this order; ``grpid`` defaults to them. The
    # lumping applies whenever this is set, so leave it empty without
    # fix_shape.
    shape_coils: tuple[str, ...] = ()
    # Include the passive structure (pf_passive eddy loops) as external coils.
    eddy: bool = False

    # --- execution ---
    backend: Optional["ExecutionBackend"] = None  # None -> LocalBackend (vaft.code.execution)

    @classmethod
    def limited_shape_control(
        cls,
        isor: Sequence[float],
        isoz: Sequence[float],
        shape_coils: Sequence[str] = ("PF5", "PF6", "PF9", "PF10"),
        **overrides: Any,
    ) -> "TESConfig":
        """Hold a limited boundary by re-fitting the outer PF coils.

        With the measured coil currents fixed, a forward solve reaches only the
        plasma currents those coils hold in radial balance (40-60 kA at 39915 @
        325 ms). Here TES re-fits ``shape_coils`` every iteration so that the
        boundary flux passes through the iso-flux points ``(isor, isoz)``. There
        are no X-points (NXPT 0), so TES takes the target flux at the iso-flux
        point nearest the limiter: put one on the wall where the plasma should
        rest, e.g. from ``vaft.code.tes.inputs.limited_iso_points``. FIX_SHAPE
        replaces TES's vertical stabilizer, so include points above and below
        the axis. The measured currents are the starting point; ``.RESULT``
        reports each coil's change (``result.scalars["coils"]``).
        """
        if len(isor) != len(isoz) or len(isor) < 2:
            raise ValueError("limited_shape_control needs >= 2 iso-flux points with matching R and Z")
        settings = dict(
            fix_shape=1,
            nxpt=0,
            isor=tuple(float(v) for v in isor),
            isoz=tuple(float(v) for v in isoz),
            shape_coils=tuple(str(c).upper() for c in shape_coils),
            coil_model="elements",
        )
        settings.update(overrides)
        return cls(**settings)

    @classmethod
    def legacy_double_null(cls, **overrides: Any) -> "TESConfig":
        """Approximately the pre-#1469 deck: double-null shape control, legacy coils.

        ``fix_shape=1`` re-fits the 18 legacy coil groups every iteration so the
        field vanishes at (0.30, +-0.55) and the boundary flux passes through
        the iso-flux points, one of them (0.110, 0). The limiter is passed as
        its clipped vertices only. Values copied from the legacy
        ``_39915_tes.in``. It is not a byte-for-byte reproduction: the limiter
        is now clipped at the grid box instead of losing its outside vertices,
        and in-grid rows are folded per grid node.

        Before #1469 the deck wrote pf_passive currents 1000x too small, so
        ``eddy=True`` had no effect. They are physical now, and on 39915 @
        325 ms this shape-control fit then diverges; the preset is meant for
        comparisons with ``eddy=False``.
        """
        legacy = dict(
            fix_shape=1,
            nxpt=2,
            xpr=(0.30, 0.30),
            xpz=(-0.55, 0.55),
            active=(1, 1),
            snowflake=(0, 0),
            isor=(0.640, 0.110, 0.300, 0.300),
            isoz=(0.000, 0.000, 0.550, -0.550),
            grpid=tuple(range(1, 19)),
            coil_model="legacy",
            limiter_spacing=None,
        )
        legacy.update(overrides)
        return cls(**legacy)


@dataclass
class TESInputs:
    """Prepared input bundle for a TES run."""

    workdir: Path
    cinput: Path                              # the C-format file ``rtes`` consumes
    ods: Any = None
    namelist: Optional[Path] = None
    files: tuple[Path, ...] = ()


@dataclass
class TESResult(RunOutcome):
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

    #: ``"completed"``, ``"timeout"`` or ``"queue_timeout"`` (#1016).
    runtime_status: str = "completed"
    #: Wall time from launch to exit or stop [s].
    elapsed_s: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0
