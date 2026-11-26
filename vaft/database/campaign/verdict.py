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
# Snakemake >= 8 runs a ``run:`` job in a spawned subprocess and logs that
# job's failure as a traceback chain ending in this exception: one job failed,
# not Snakemake.
_PER_JOB_EXCEPTIONS = frozenset({"SpawnedJobError", "snakemake.exceptions.SpawnedJobError"})
_CHAIN_LINKS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)


@dataclass(frozen=True)
class ChunkVerdict:
    """``done``: the chunk ran to its end; ``reason`` says why it did not."""

    done: bool
    failed_rules: tuple[str, ...] = ()
    reason: str = ""


def _whole_run_traceback(log: str) -> bool:
    """Whether ``log`` holds a traceback chain that is not one job's failure.

    A chain is consecutive tracebacks joined by Python's "During handling" or
    "direct cause" lines; it is judged by the exception that ends it.
    """
    lines = log.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].startswith("Traceback"):
            index += 1
            continue
        final = ""
        while index < len(lines):
            # One traceback: up to its first unindented line, the exception.
            index += 1
            while index < len(lines) and (
                not lines[index].strip() or lines[index][0].isspace()
            ):
                index += 1
            if index >= len(lines):
                break
            final = lines[index].partition(":")[0].strip()
            # Does a chained traceback follow?
            ahead = index + 1
            while ahead < len(lines) and (
                not lines[ahead].strip() or lines[ahead][0].isspace()
                or lines[ahead].strip() in _CHAIN_LINKS
            ):
                ahead += 1
            linked = any(
                lines[k].strip() in _CHAIN_LINKS for k in range(index + 1, ahead)
            )
            if not (linked and ahead < len(lines) and lines[ahead].startswith("Traceback")):
                break
            index = ahead
        if final not in _PER_JOB_EXCEPTIONS:
            return True
        index += 1
    return False


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
    if _whole_run_traceback(log):
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
