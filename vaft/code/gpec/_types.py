"""Dataclasses shared by the GPEC-suite orchestration and solver modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import CoilSet3D

    from ..execution import ExecutionBackend
    from ._coil_input import CoilInputSpec

GPEC_HOME_ENV = "GPECHOME"
DEFAULT_MODULES = ("dcon", "rdcon", "stride", "gpec")
DEFAULT_MODES = (1, 2)
SUPPORTED_MODULES = frozenset(DEFAULT_MODULES)
STABILITY_MODULES = frozenset(("dcon", "rdcon", "stride"))


@dataclass(frozen=True)
class DCONOptions:
    """DCON-specific namelist overrides.

    ``mer_flag``/``bal_flag``/``thmax0`` gate DCON's *local* stability criteria,
    and they matter for more than reproducibility: DCON zero-fills the whole
    local-stability spline before running either scan (``dcon/dcon.F:148-160``),
    so a criterion that was never evaluated is written to the netCDF as exactly
    zero -- which is also its marginal value. Recording what was requested is
    what lets VAFT tell "marginally stable everywhere" from "never computed".

    ``bal_flag`` defaults to the packaged namelist's ``f``. That is deliberately
    unchanged here, but note it makes DCON the odd one out: ``rdcon.in`` and
    ``stride.in`` both ship ``bal_flag=t``.
    """

    sas_flag: bool = False
    qhigh: float = 20.2
    psiedge: float = 1.0
    mer_flag: bool = True
    bal_flag: bool = False
    thmax0: float = 1.0


@dataclass(frozen=True)
class RDCONOptions:
    """RDCON-specific namelist overrides.

    ``rmatch`` wants one resistivity and one mass density *per rational
    surface*, and the packaged ``rmatch.in`` supplies a single scalar, so it
    stops with ``eta requires N non-zero elements`` before writing
    ``globalsol.bin`` (#716). Delta-prime is unaffected -- RDCON computes it
    and ``rmatch`` does not -- so this is opt-in rather than required.

    Supplying ``t_e``/``n_e`` fills both arrays from the plasma: after RDCON
    reports where its rational surfaces are, the profiles are evaluated there
    and written into ``rmatch.in``. Reading RDCON's own surfaces rather than
    re-deriving them is what makes the arrays line up with the ``msing`` the
    code will clamp to.

    ``None`` throughout preserves the packaged template verbatim.
    """

    #: Electron temperature [eV] and density [m^-3] of the bulk plasma, with
    #: the normalized poloidal flux they are given on. All three or none.
    t_e: Optional[Sequence[float]] = None
    n_e: Optional[Sequence[float]] = None
    psi_norm: Optional[Sequence[float]] = None

    #: Passed through to :func:`vaft.process.equilibrium.resistive_layer_parameters`.
    ion_mass_amu: float = 1.0
    z_eff: float = 2.0
    ln_lambda: float = 17.0

    def __post_init__(self) -> None:
        # Checked here, where the caller's mistake is, rather than after RDCON
        # has already run: a malformed triple used to raise out of the suite
        # between the solver and its companion, and a descending coordinate
        # was interpolated as if it were sorted.
        if not self.has_kinetic_profiles:
            return
        sizes = {
            name: len(getattr(self, name)) for name in ("psi_norm", "t_e", "n_e")
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(
                "RDCONOptions psi_norm, t_e and n_e must have one length; got "
                + ", ".join(f"{name}: {size}" for name, size in sizes.items())
            )
        coordinate = [float(value) for value in self.psi_norm]
        if len(coordinate) < 2 or any(
            not later > earlier for earlier, later in zip(coordinate, coordinate[1:])
        ):
            raise ValueError(
                "RDCONOptions.psi_norm must be strictly increasing (core to edge); "
                "reverse an outboard-in profile together with its t_e and n_e"
            )

    @property
    def has_kinetic_profiles(self) -> bool:
        return (
            self.t_e is not None
            and self.n_e is not None
            and self.psi_norm is not None
        )


@dataclass(frozen=True)
class STRIDEOptions:
    """STRIDE-specific namelist overrides (none exposed yet -- packaged defaults only)."""


@dataclass(frozen=True)
class IdealGPECOptions:
    """Ideal-GPEC-specific namelist overrides.

    ``coil_specs`` selects and excites 3D coil sets (see
    :class:`vaft.code.gpec.CoilInputSpec`): when set, ``coil.in`` and the
    referenced ``.dat`` files are generated from ``coil_config`` instead of
    copying the packaged template verbatim.  ``None`` preserves the legacy
    template behavior; an explicit ``GPECCaseInputs.coil_in`` always wins
    over both.

    ``machine`` is the GPEC ``machine`` word and the ``<machine>_<set>.dat``
    file prefix.  For ``"vest"`` (default) ``coil_config`` defaults to the
    packaged VEST geometry and the direction words to
    :data:`vaft.machine_mapping.conventions.VEST_GPEC_COIL_DIRECTIONS`.  For
    any other machine ``coil_config`` (name -> ``CoilSet3D``), ``ip_direction``
    and ``bt_direction`` are all required: they are machine facts, never
    inherited from a template.  Derive the direction words from the machine's
    sign contract with
    :func:`vaft.machine_mapping.conventions.gpec_coil_directions` rather than
    writing the pair by hand.

    These four sit here rather than on :class:`GPECSuiteConfig`, next to
    ``coil_data_dir``, because only the ideal-GPEC stage consumes them and
    ``__post_init__`` validates them as one group: DCON, RDCON and STRIDE
    never see a coil.  Move them up only if a stability module starts needing
    the machine word.
    """

    coil_flag: bool = True
    coil_specs: Optional[Sequence["CoilInputSpec"]] = None
    machine: str = "vest"
    coil_config: Optional[Mapping[str, "CoilSet3D"]] = None
    ip_direction: Optional[str] = None
    bt_direction: Optional[str] = None

    def __post_init__(self) -> None:
        # A config error is knowable here; refusing at construction keeps it
        # from surfacing after DCON has run and gpec.in / vac.in are staged.
        if self.machine != "vest":
            missing = [
                name
                for name in ("coil_specs", "coil_config", "ip_direction", "bt_direction")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"machine {self.machine!r}: {', '.join(missing)} must be given explicitly "
                    "(only 'vest' has packaged coil geometry and direction words); or pass an "
                    "explicit GPECCaseInputs.coil_in"
                )
        # An explicitly empty selection is refused rather than read as "use the
        # packaged template" -- and refused here, for the reason above: it used
        # to surface from inside `prepare`, which is the late failure this
        # method exists to prevent.
        if self.coil_specs is not None and len(self.coil_specs) == 0:
            raise ValueError(
                "coil_specs is empty: at least one coil set name is required; pass "
                "None to use the packaged coil.in template"
            )


@dataclass(frozen=True)
class GPECSuiteConfig:
    """Runtime and VEST-default configuration for the GPEC suite.

    ``gpec_home`` defaults to ``None``, in which case the installation root is
    read from ``$GPECHOME``.  Preparing a case never needs it; only running a
    module does, and a missing installation is reported there.

    Per-solver namelist overrides live on the ``dcon``/``rdcon``/``stride``/
    ``gpec`` sub-options below rather than as flat fields on this dataclass,
    so the field count here stays fixed as solver-specific knobs accumulate.

    ``verify_outputs`` opts into content-level success checks (does the
    produced ``.nc`` actually contain the expected physics variable, not just
    exist) via each solver's ``check_success``. Off by default so tests and
    trivial stub executables -- which produce no real ``.nc`` content -- keep
    working; real production runs should set it.
    """

    gpec_home: Path | str | None = None
    executable_dir: Path | str | None = None
    modules: Sequence[str] = DEFAULT_MODULES
    modes: Sequence[int] = DEFAULT_MODES
    run_mode: str = "run_if_available"
    templates_dir: Path | str | None = None
    coil_data_dir: Path | str | None = None
    psilow: float = 1e-2
    psihigh: float = 0.994
    verify_outputs: bool = False
    timeout: Optional[float] = 1200.0
    env: Mapping[str, str] = field(default_factory=dict)
    dcon: DCONOptions = field(default_factory=DCONOptions)
    rdcon: RDCONOptions = field(default_factory=RDCONOptions)
    stride: STRIDEOptions = field(default_factory=STRIDEOptions)
    gpec: IdealGPECOptions = field(default_factory=IdealGPECOptions)
    backend: Optional["ExecutionBackend"] = None  # None -> LocalBackend (vaft.code.execution)


@dataclass
class GPECCaseInputs:
    """Materialized inputs for one shot/time GPEC-suite case."""

    shot: int
    time_ms: int | str | None
    geqdsk: Path
    workdir: Path
    coil_in: Path | None = None
    # Ideal GPEC consumes the DCON files for the same time/mode.  In the
    # canonical FileDB layout DCON may live in a separate code-specific tree.
    dcon_workdir: Path | None = None


@dataclass
class GPECModuleRun:
    """Status for one module/mode directory."""

    module: str
    mode: int
    workdir: Path
    returncode: Optional[int] = None
    #: ``prepared`` | ``completed`` | ``stable`` | ``skipped`` | ``failed``.
    #:
    #: ``stable`` is a *successful* outcome, kept apart from ``completed``
    #: because a stable equilibrium produces no unstable-mode output and so
    #: cannot be told from a broken run by what it wrote (issue #423).
    status: str = "prepared"
    reason: str = ""
    logs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    commands: tuple[str, ...] = ()
    #: Expected files a companion executable would have written, which this run
    #: directory does not have.  Non-empty is not a failure -- companions are
    #: optional by construction -- but it is the difference between a DCON cell
    #: that has an eigenfunction and one that never will, so it is reported
    #: rather than left for a caller to rediscover by listing the directory.
    missing_optional_outputs: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether this cell produced a usable result.

        ``stable`` counts: the solver ran and found no unstable mode, which is a
        physics result. Excluding it would make every stable discharge read as
        an unusable cell (issue #423).
        """
        return self.status in {"completed", "stable"} and self.returncode == 0


@dataclass
class GPECSuiteResult:
    """Suite status, records, and collected artifacts."""

    returncode: Optional[int]
    workdir: Path
    shot: int | None = None
    time_ms: int | str | None = None
    records: tuple[GPECModuleRun, ...] = ()
    logs: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    parsed: Any = None
    #: SHA-256 of the equilibrium this suite consumed.
    #:
    #: The stability end of the provenance chain. Without it, "which CHEASE
    #: equilibrium produced this stability result" is answerable only by
    #: trusting that the file at a recorded path never changed -- which is the
    #: assumption `replication.is_reusable` exists to stop relying on. Empty
    #: when the equilibrium could not be read, which is a fact about this run
    #: rather than a reason to fail it.
    input_equilibrium_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0
