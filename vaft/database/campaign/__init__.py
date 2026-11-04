"""Run a pipeline over many shots as lanes of resumable chunks, and measure it.

The routine worker (:mod:`vaft.database.worker`) processes new shots as they
arrive; a *campaign* re-processes thousands at once on a cluster: shots dealt
over lanes (one FileDB root and one scheduler allocation each), each lane run in
sequential chunks that resume where they stopped. This package holds the parts
that are not tied to one cluster:

* :mod:`.lanes` -- lane planning, chunk splits and each lane's resume state;
* :mod:`.verdict` -- whether a chunk's Snakemake run finished (per-shot failures)
  or failed as a whole;
* :mod:`.smklog` -- jobs, slot occupancy and per-rule cost from a Snakemake log;
* :mod:`.benchmark` -- what a lane cost: CPU utilization (from the job cgroup),
  booked CPU-hours per shot, throughput, peak memory, per-rule job-seconds;
* :mod:`.compare` -- whether two runs' products are the same science.

Scheduler submission, environment pinning and host-specific mounts stay in the
site scripts (``workflow/cluster_regeneration``); job execution and resource
accounting are :mod:`vaft.code`'s.
"""

from .benchmark import compare_reports, lane_report
from .compare import compare_products, summarize
from .lanes import LaneState, chunk_name, disjoint, plan_lanes, read_shots, split_chunks
from .smklog import occupancy, parse_log, rule_costs
from .verdict import ChunkVerdict, chunk_verdict

__all__ = [
    "ChunkVerdict",
    "LaneState",
    "chunk_name",
    "chunk_verdict",
    "compare_products",
    "compare_reports",
    "disjoint",
    "lane_report",
    "occupancy",
    "parse_log",
    "plan_lanes",
    "read_shots",
    "rule_costs",
    "split_chunks",
    "summarize",
]
