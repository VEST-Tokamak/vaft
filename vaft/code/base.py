"""Common protocol objects for external code adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class CodeConfig:
    """Base runtime configuration for a fusion-code adapter."""

    executable: Optional[str] = None
    workdir: Path | str = Path(".")
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None


@dataclass
class CodeInputs:
    """Base input bundle for a code run."""

    workdir: Path
    files: tuple[Path, ...] = ()
    ods: Any = None


@dataclass
class CodeResult:
    """Base result bundle returned by a code adapter."""

    returncode: Optional[int]
    workdir: Path
    stdout: str = ""
    stderr: str = ""
    logs: tuple[Path, ...] = ()
    outputs: Mapping[str, tuple[Path, ...]] = field(default_factory=dict)
    parsed: Any = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CodeRunner(Protocol):
    """Protocol implemented by Python-first code runners."""

    def run(self, inputs: CodeInputs, config: CodeConfig) -> CodeResult:
        """Run the configured external code."""
        ...
