"""Did a chunk's Snakemake run finish (some shots failing), or fail as a whole?

Under ``--keep-going`` a shot whose rule fails makes Snakemake exit 1 after
running everything else: the chunk finished, with per-shot failures recorded in
the products. A lock, a broken Snakefile, a failed run-scoped rule (one every
shot needs), a crash or a kill is a failure of the run itself: rerunning the
chunk is the answer, not marking it done. Which rules are run-scoped is the
workflow's to say (:class:`~.spec.WorkflowSpec`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..worker.runner import _LOCK_MARKERS

__all__ = ["ChunkVerdict", "chunk_verdict"]

_WHOLE_RUN_MARKERS = (*_LOCK_MARKERS, "SyntaxError", "MissingInputException")
_PER_SHOT_WORKFLOW_ERROR = "At least one job did not complete successfully"


@dataclass(frozen=True)
class ChunkVerdict:
    """``done``: the chunk ran to its end; ``reason`` says why it did not."""

    done: bool
    failed_rules: tuple[str, ...] = ()
    reason: str = ""


def chunk_verdict(
    returncode: int,
    log: str,
    *,
    run_scoped_rules: Iterable[str] = (),
    missing_outputs: Iterable[str] = (),
) -> ChunkVerdict:
    """Judge one chunk's Snakemake run from its exit code and log text.

    ``missing_outputs`` names required run-scoped outputs the run did not write.
    """
    failed = tuple(re.findall(r"^Error in rule (\w+):", log, flags=re.MULTILINE))
    missing = sorted(missing_outputs)
    if returncode == 0:
        if missing:
            return ChunkVerdict(False, failed, f"required output not written: {', '.join(missing)}")
        return ChunkVerdict(True, failed)
    if not failed:
        return ChunkVerdict(False, failed, f"exit {returncode} without a failed rule (killed or crashed)")
    if "Exiting because a job execution failed" not in log:
        return ChunkVerdict(False, failed, "Snakemake did not finish its keep-going run")
    for marker in _WHOLE_RUN_MARKERS:
        if marker in log:
            return ChunkVerdict(False, failed, f"whole-run failure: {marker}")
    if re.search(r"^Traceback", log, flags=re.MULTILINE):
        return ChunkVerdict(False, failed, "whole-run failure: a traceback in Snakemake itself")
    scoped = sorted(set(failed) & set(run_scoped_rules))
    if scoped:
        return ChunkVerdict(False, failed, f"run-scoped rule failed: {', '.join(scoped)}")
    # A WorkflowError is per-shot only when its message is the keep-going summary.
    # The message is on the same line or on the next non-blank one.
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("WorkflowError"):
            continue
        message = line.partition(":")[2].strip()
        if not message:
            message = next((text.strip() for text in lines[index + 1:] if text.strip()), "")
        if not message.startswith(_PER_SHOT_WORKFLOW_ERROR):
            return ChunkVerdict(False, failed, f"WorkflowError: {message}")
    if missing:
        return ChunkVerdict(False, failed, f"required output not written: {', '.join(missing)}")
    return ChunkVerdict(True, failed)
