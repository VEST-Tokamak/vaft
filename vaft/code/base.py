"""Common protocol objects for external code adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from .execution import ExecutionBackend


@dataclass(frozen=True)
class CodeConfig:
    """Base runtime configuration for a fusion-code adapter."""

    executable: Optional[str] = None
    workdir: Path | str = Path(".")
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    # None runs locally; see vaft.code.execution.
    backend: Optional["ExecutionBackend"] = None


@dataclass
class CodeInputs:
    """Base input bundle for a code run."""

    workdir: Path
    files: tuple[Path, ...] = ()
    ods: Any = None


class RunOutcome:
    """``status`` and ``timed_out`` for a result that carries ``runtime_status``.

    Every subprocess adapter's result has the same three outcome fields
    (#1016): ``returncode`` (``None`` when the program was stopped),
    ``runtime_status`` (``"completed"``, ``"timeout"`` or ``"queue_timeout"``;
    see :mod:`vaft.code.execution`) and ``elapsed_s``.
    """

    runtime_status: str
    ok: bool

    @property
    def status(self) -> str:
        """``"completed"`` when the result is usable (``ok``), else ``"failed"``."""
        return "completed" if self.ok else "failed"

    @property
    def timed_out(self) -> bool:
        """The program was stopped by a time limit, running or queued."""
        return self.runtime_status in ("timeout", "queue_timeout")


@dataclass
class CodeResult(RunOutcome):
    """Base result bundle returned by a code adapter."""

    returncode: Optional[int]
    workdir: Path
    stdout: str = ""
    stderr: str = ""
    logs: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    parsed: Any = None
    runtime_status: str = "completed"
    elapsed_s: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CodeRunner(Protocol):
    """Protocol implemented by Python-first code runners."""

    def run(self, inputs: CodeInputs, config: CodeConfig) -> CodeResult:
        """Run the configured external code."""
        ...


__all__ = [
    "CodeConfig",
    "CodeInputs",
    "RunOutcome",
    "CodeResult",
    "CodeRunner",
]
