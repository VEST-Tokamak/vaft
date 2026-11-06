"""Run a Snakemake workflow over many shots as lanes of resumable chunks, and measure it.

The routine worker (:mod:`vaft.database.worker`) processes new shots as they
arrive; a *campaign* re-processes thousands at once: shots dealt over lanes (one
FileDB root each), each lane run in sequential chunks that resume where they
stopped. The same code runs on a workstation, a server, or a Slurm cluster:

* :mod:`.spec` -- what the workflow states about itself (``CAMPAIGN_*`` names in
  its ``paths.py``); this package names no rule, product or solver;
* :mod:`.lanes` -- lane planning, chunk splits and each lane's resume state;
* :mod:`.run` -- one lane's chunks, with CPU and memory measured by
  :class:`vaft.code.accounting.UsageMeter` wherever it runs;
* :mod:`.verdict` -- whether a chunk finished (per-shot failures) or failed as a whole;
* :mod:`.launch` -- every lane started through a :mod:`vaft.code` backend
  (local processes, or Slurm jobs under the execution policy);
* :mod:`.benchmark` -- what a lane cost: utilization, CPU-hours per shot,
  throughput, memory, per-rule job-seconds;
* :mod:`.compare` -- whether two runs' products are the same science.

Site matters -- hosts, accounts, partitions, mount namespaces, raw-data staging --
stay outside: on the command line, in environment variables, or in hook commands
a site keeps in its own (unpublished) configuration.
"""

from .benchmark import compare_reports, lane_report
from .compare import compare_products, summarize
from .lanes import LaneState, chunk_name, disjoint, plan_lanes, read_shots, split_chunks
from .launch import lane_dirs, lane_request, launch_lanes
from .run import LaneOptions, run_lane
from .spec import WorkflowSpec
from .verdict import ChunkVerdict, chunk_verdict

__all__ = [
    "ChunkVerdict",
    "LaneOptions",
    "LaneState",
    "WorkflowSpec",
    "chunk_name",
    "chunk_verdict",
    "compare_products",
    "compare_reports",
    "disjoint",
    "lane_dirs",
    "lane_report",
    "lane_request",
    "launch_lanes",
    "plan_lanes",
    "read_shots",
    "run_lane",
    "split_chunks",
    "summarize",
]
