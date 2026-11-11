"""What happened to one external-code stage: run, sample, or neither -- kept apart.

A workflow that can either run an external code or load a precomputed result of
it has four separate facts to report per stage, and collapsing any two of them
misleads: a precomputed sample standing in for a run that *failed* must not read
as a successful run, and a run that completed but whose output failed its check
must not read as a usable one.

* ``execution`` -- what happened to the attempt to run the code:
  ``not attempted``, ``computed``, ``failed``, ``unavailable`` (no launchable
  executable) or ``incomplete`` (it ran, its output is partial);
* ``source`` -- where the data the analysis uses came from: ``live`` or
  ``precomputed sample`` (empty until decided);
* ``validation`` -- whether that data passed its check: ``passed``, ``failed``
  or ``not evaluated``;
* ``reason`` -- why the sample was used, when it was.

:func:`executable_ready` is the preflight half: found is not ready, so it also
requires a regular, executable file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

__all__ = [
    "DATA_SOURCES",
    "EXECUTION_STATES",
    "VALIDATION_STATES",
    "StageStatus",
    "executable_ready",
    "format_stage_table",
]

EXECUTION_STATES = ("not attempted", "computed", "failed", "unavailable", "incomplete")
DATA_SOURCES = ("", "live", "precomputed sample")
VALIDATION_STATES = ("passed", "failed", "not evaluated")


@dataclass
class StageStatus:
    """The four facts of one stage; assignments are checked against their vocabularies."""

    name: str
    execution: str = "not attempted"
    source: str = ""
    validation: str = "not evaluated"
    reason: str = ""

    def __setattr__(self, key: str, value) -> None:
        allowed = {"execution": EXECUTION_STATES, "source": DATA_SOURCES, "validation": VALIDATION_STATES}.get(key)
        if allowed is not None and value not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, not {value!r}")
        super().__setattr__(key, value)

    @property
    def fell_back(self) -> bool:
        """Whether the analysis uses a sample although a run was attempted or wanted."""
        return self.source == "precomputed sample" and self.execution != "not attempted"


def executable_ready(path: Optional[Path | str]) -> tuple[bool, str]:
    """``(ready, detail)`` for one executable: found, a regular file, and executable.

    ``detail`` is the path when ready and the reason otherwise. An environment
    variable that points somewhere is not readiness; this checks the file.
    """
    if path is None:
        return False, "not found"
    candidate = Path(path)
    if not candidate.is_file():
        return False, f"{candidate} is not a file"
    if not os.access(candidate, os.X_OK):
        return False, f"{candidate} is not executable"
    return True, str(candidate)


def format_stage_table(stages: Iterable[StageStatus]) -> str:
    """A fixed-width text table of stages, one line each, for printing."""
    lines = [f"{'stage':8s} {'execution':14s} {'data source':19s} {'validation':14s} reason"]
    for stage in stages:
        lines.append(
            f"{stage.name:8s} {stage.execution:14s} {stage.source or '-':19s} {stage.validation:14s} {stage.reason}"
        )
    return "\n".join(lines)
