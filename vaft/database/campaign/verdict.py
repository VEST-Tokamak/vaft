"""Was a chunk's non-zero Snakemake exit a per-shot outcome or a failed run?

Under ``--keep-going`` a shot whose stage fails ends the run with exit 1, an
``Error in rule`` block, "Exiting because a job execution failed" and
"WorkflowError: At least one job did not complete successfully." That is a
result, recorded in the shot's own status. Anything else -- the directory lock,
a syntax error, missing input, a traceback in Snakemake itself, a failed
run-scoped rule, a missing checkpoint verdict, or a Snakemake killed before it
finished -- means the chunk did not run and must not be marked done.

This is the rule ``workflow/cluster_regeneration/run_lane.sh`` applied in shell,
kept here so other runners judge a run the same way (the deployed lanes still
use the shell rule; ``test/test_campaign.py`` pins that the two agree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..worker.runner import _LOCK_MARKERS

__all__ = ["ChunkVerdict", "RUN_SCOPED_RULES", "chunk_verdict"]

#: Rules whose failure stops the whole run: they produce what every shot needs.
RUN_SCOPED_RULES = frozenset(
    {"generate_static_ods", "generate_efit_table", "preflight_raw_dumps", "select_impa_shots", "plot_static"}
)

_WHOLE_RUN_MARKERS = (*_LOCK_MARKERS, "SyntaxError", "MissingInputException")
_PER_SHOT_WORKFLOW_ERRORS = ("At least one job did not complete successfully",)


@dataclass(frozen=True)
class ChunkVerdict:
    """``done``: the chunk ran to its end; ``reason`` says why it did not."""

    done: bool
    failed_rules: tuple[str, ...] = ()
    reason: str = ""


def chunk_verdict(returncode: int, log: str, *, has_eligible_verdict: bool) -> ChunkVerdict:
    """Judge one chunk's Snakemake run from its exit code and log text."""
    failed = tuple(re.findall(r"^Error in rule (\w+):", log, flags=re.MULTILINE))
    if returncode == 0:
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
    scoped = sorted(set(failed) & RUN_SCOPED_RULES)
    if scoped:
        return ChunkVerdict(False, failed, f"run-scoped rule failed: {', '.join(scoped)}")
    # As `grep -A1 "^WorkflowError"`: the WorkflowError line and the one after
    # it must be the bare header and the keep-going summary, nothing else.
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("WorkflowError"):
            continue
        for candidate in lines[index:index + 2]:
            if re.match(r"^WorkflowError:?$", candidate) or candidate == "--":
                continue
            if any(candidate.startswith(text) for text in _PER_SHOT_WORKFLOW_ERRORS):
                continue
            return ChunkVerdict(False, failed, f"WorkflowError: {candidate.strip()}")
    if not has_eligible_verdict:
        return ChunkVerdict(False, failed, "no preflight verdict was written")
    return ChunkVerdict(True, failed)
