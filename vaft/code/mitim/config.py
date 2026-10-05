"""Configuration and result objects for the MITIM-fusion adapter (#1588 stage A1).

MITIM (https://github.com/pabloprf/MITIM-fusion, MIT licence) needs Python
3.10-3.12 and pulls tensorflow, botorch and torch. VAFT's interpreter is newer
and pinned differently, so MITIM runs in its **own** interpreter: VAFT never
imports ``mitim_tools``. The adapter launches a small driver script with that
interpreter through the ordinary execution backend, exactly as it launches any
other external code.

MITIM reads its machine configuration from ``$MITIM_CONFIG``. VAFT writes one
per run (:func:`mitim_user_config`) into the run directory, records it, and
never edits MITIM's own ``templates/config_user.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from ..base import RunOutcome

if TYPE_CHECKING:
    from ..execution import ExecutionBackend
    from ..gacode._types import GACODEConfig

#: Environment variable naming the interpreter of the isolated MITIM environment.
MITIM_PYTHON_ENV = "VAFT_MITIM_PYTHON"

#: MITIM releases this adapter has been run against. Anything else is reported as
#: ``unsupported_version`` rather than assumed compatible.
SUPPORTED_MITIM_VERSIONS = ("5.3.0",)

#: The MITIM machine name the per-run configuration defines. Every MITIM code
#: preference points at it, so MITIM runs its codes inside the allocation VAFT
#: already holds and never submits jobs of its own.
VAFT_MACHINE = "vaft_local"

#: Every MITIM ``preferences`` key of the supported release (``config_user_example.json``).
MITIM_CODES = (
    "trxpl", "profiles_gen", "tgyro", "tglf", "neo", "cgyro", "qualikiz", "gx", "astra",
    "eq", "scruncher", "ntcc", "get_fbm", "transp", "idl", "eped",
)


@dataclass(frozen=True)
class MITIMConfig:
    """Runtime configuration for one MITIM-driven run.

    Parameters
    ----------
    python : str or None
        Interpreter of the isolated MITIM environment. ``None`` reads
        ``$VAFT_MITIM_PYTHON`` [-].
    gacode : GACODEConfig or None
        The GACODE build MITIM runs TGLF/NEO from, resolved exactly as VAFT's own
        GACODE adapters resolve it (``$GACODEHOME``). ``None`` uses the default [-].
    cores : int
        Cores MITIM may use inside the run (``cores_per_node`` of its machine) [-].
    timeout : float or None
        Wall-clock limit of the whole driver [s]. Reaching it is a result
        (``runtime_status="timeout"``), not an exception.
    backend : ExecutionBackend or None
        Where the driver runs (local, or a VAFT Slurm backend) [-].
    env : mapping
        Extra environment, applied last [-].
    verbose_level : int
        MITIM's ``verbose_level`` preference [-].
    """

    python: Optional[str] = None
    gacode: Optional["GACODEConfig"] = None
    cores: int = 1
    timeout: Optional[float] = None
    backend: Optional["ExecutionBackend"] = None
    env: Mapping[str, str] = field(default_factory=dict)
    verbose_level: int = 1

    def interpreter(self) -> Optional[str]:
        """The configured MITIM interpreter, or None when neither source names one."""
        value = self.python or os.environ.get(MITIM_PYTHON_ENV, "").strip()
        return value or None


def mitim_user_config(config: MITIMConfig, workdir: str | Path, *, username: str | None = None) -> dict:
    """The ``$MITIM_CONFIG`` document for one run.

    One machine, ``vaft_local``, with ``machine = "local"`` and its scratch inside
    the run directory; every code preference points at it. MITIM therefore runs
    its codes as local processes of the allocation the VAFT backend obtained, and
    the scheduler stays VAFT's (no second, nested Slurm submission).
    """
    scratch = Path(workdir).resolve() / "mitim_scratch"
    return {
        "preferences": {
            "verbose_level": str(int(config.verbose_level)),
            "dpi_notebook": "80",
            **{code: VAFT_MACHINE for code in MITIM_CODES},
        },
        VAFT_MACHINE: {
            "machine": "local",
            "username": username or os.environ.get("USER", "vaft"),
            "scratch": str(scratch) + "/",
            "modules": "",
            "cores_per_node": int(config.cores),
            "gpus_per_node": 0,
        },
    }


@dataclass
class MITIMResult(RunOutcome):
    """One MITIM driver run.

    ``result`` is what the driver wrote to ``result.json``. ``record`` is the
    provenance VAFT wrote to ``record.json``: the availability snapshot, the
    per-run MITIM configuration, the GACODE build and the launch outcome.
    """

    returncode: Optional[int]
    workdir: Path
    stdout: str = ""
    stderr: str = ""
    runtime_status: str = "completed"
    elapsed_s: Optional[float] = None
    result: Optional[Mapping[str, Any]] = None
    record: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """The driver exited 0 *and* reported success in ``result.json``."""
        return self.returncode == 0 and bool(self.result) and self.result.get("status") == "ok"
