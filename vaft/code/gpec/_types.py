"""Dataclasses shared by the GPEC-suite orchestration and solver modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

GPEC_HOME_ENV = "GPECHOME"
DEFAULT_MODULES = ("dcon", "rdcon", "stride", "gpec")
DEFAULT_MODES = (1, 2)
SUPPORTED_MODULES = frozenset(DEFAULT_MODULES)
STABILITY_MODULES = frozenset(("dcon", "rdcon", "stride"))


@dataclass(frozen=True)
class DCONOptions:
    """DCON-specific namelist overrides."""

    sas_flag: bool = False
    qhigh: float = 20.2
    psiedge: float = 1.0


@dataclass(frozen=True)
class RDCONOptions:
    """RDCON-specific namelist overrides (none exposed yet -- packaged defaults only)."""


@dataclass(frozen=True)
class STRIDEOptions:
    """STRIDE-specific namelist overrides (none exposed yet -- packaged defaults only)."""


@dataclass(frozen=True)
class IdealGPECOptions:
    """Ideal-GPEC-specific namelist overrides."""

    coil_flag: bool = True


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


@dataclass
class GPECCaseInputs:
    """Materialized inputs for one shot/time GPEC-suite case."""

    shot: int
    time_ms: int | str | None
    geqdsk: Path
    workdir: Path
    coil_in: Path | None = None
    dcon_workdir: Path | None = None


@dataclass
class GPECModuleRun:
    """Status for one module/mode directory."""

    module: str
    mode: int
    workdir: Path
    returncode: Optional[int] = None
    status: str = "prepared"
    reason: str = ""
    logs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    commands: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0


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

    @property
    def ok(self) -> bool:
        return self.returncode == 0
