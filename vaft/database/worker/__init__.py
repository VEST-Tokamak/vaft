"""New-shot pipeline worker for the VEST server (issue #58).

Polls the VEST SQL ``shot`` table, classifies each settled shot, runs the
routine Snakemake pipeline for it, and keeps durable per-shot and per-stage
state in SQLite.  Run it with ``vaft pipeline-worker run``; read its state for
monitoring with :func:`read_worker_state`.
"""

from .classify import Classification, Classifier, RawFieldClassifier, ShotObservation
from .config import WorkerConfig, WorkerConfigError, load_worker_config
from .service import CycleReport, PipelineWorker
from .state import SHOT_STATES, WorkerState, read_worker_state

__all__ = [
    "SHOT_STATES",
    "Classification",
    "Classifier",
    "CycleReport",
    "PipelineWorker",
    "RawFieldClassifier",
    "ShotObservation",
    "WorkerConfig",
    "WorkerConfigError",
    "WorkerState",
    "load_worker_config",
    "read_worker_state",
]
