"""Server-only configuration for the new-shot pipeline worker (issue #58).

Every operational setting -- where state lives, how often SQL is polled, which
workflow runs and how -- comes from a deployment's own YAML file.  Nothing here
has a library default that could make a workstation behave like the server:
a key the worker cannot run without is required, and a missing one is an error
naming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


#: Environment variable naming the worker configuration file.
CONFIG_ENVIRONMENT_VARIABLE = "VAFT_WORKER_CONFIG"

_REQUIRED = (
    "state_db",
    "log_dir",
    "workflow_dir",
    "pipeline_config",
    "first_shot",
    "poll_interval",
    "quiet_seconds",
    "cores",
    "snakemake_cmd",
)


class WorkerConfigError(ValueError):
    """Raised when the worker configuration cannot be used."""


@dataclass(frozen=True)
class WorkerConfig:
    """What one deployment of the worker runs, and where it keeps its state."""

    state_db: Path
    log_dir: Path
    workflow_dir: Path
    pipeline_config: Path
    #: The first shot the worker is responsible for.  Everything below it
    #: belongs to batch regeneration, not to the worker.
    first_shot: int
    poll_interval: float
    #: A shot whose inventory does not (yet) match the previous shot's is
    #: processed once no field has been uploaded for this long.  Measured
    #: gaps inside one VEST upload reach 261 s.
    quiet_seconds: float
    cores: int
    snakemake_cmd: tuple[str, ...]
    extra_args: tuple[str, ...] = ()
    max_shots_per_run: int = 20
    max_attempts: int = 3
    #: How long after processing a shot keeps being re-checked for fields
    #: that arrived late (VEST has backfilled Plasma Current a day later).
    recheck_seconds: float = 3 * 86400.0
    #: How many times late fields may trigger a reprocess of one shot.
    max_reprocess: int = 3
    #: Seconds before a pipeline run is terminated; ``None`` waits forever.
    run_timeout: float | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    #: ``module:callable`` returning a :class:`~vaft.database.worker.classify.Classifier`
    #: given the pipeline config.  ``None`` uses the raw-field classifier.
    classifier: str | None = None
    #: Also record ``vaft.omas.shot_class`` from the diagnostics product.  For
    #: reference only -- it does not gate any stage until #57 lands.
    record_shot_class: bool = False
    #: The per-shot stages to run and judge, e.g. ``(raw, diagnostics, eddy)``.
    #: Written into each run's config as ``stages``, which narrows ``rule all``
    #: (the workflow's ``paths.stage_scope``).  ``None`` runs the whole pipeline.
    stages: tuple[str, ...] | None = None
    #: Disk guard: no new batch starts while any guarded filesystem has less
    #: than this many GB (1e9 bytes) free.  ``None`` disables the guard.
    min_free_gb: float | None = None
    #: Free space at which a paused worker resumes (hysteresis).  A
    #: configuration file that leaves it out gets ``min_free_gb * 1.05``, so a
    #: filesystem hovering at the limit does not pause and resume every cycle.
    #: ``None`` (the dataclass default) resumes at ``min_free_gb`` itself.
    resume_free_gb: float | None = None
    #: Paths whose filesystems are guarded.  Empty: the pipeline's ``base_dir``
    #: (where every product lands), ``log_dir`` and the state database's directory.
    disk_paths: tuple[Path, ...] = ()

    @property
    def snakefile(self) -> Path:
        return self.workflow_dir / "Snakefile"

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """The process environment runs see: ``base`` overlaid with ``env``."""
        merged = dict(os.environ if base is None else base)
        for key, value in self.env.items():
            merged[str(key)] = os.path.expandvars(str(value))
        return merged


def _path(value: Any, *, relative_to: Path, key: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if "$" in expanded:
        # An unset variable would otherwise leave a literal `${...}` directory
        # beside the config -- for `state_db`, a fresh database whose empty
        # watermark re-runs every shot from `first_shot`.
        raise WorkerConfigError(f"{key}: unset environment variable in {value!r}")
    path = Path(expanded)
    return path if path.is_absolute() else (relative_to / path)


def _command(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split()
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        raise WorkerConfigError(f"{key} must be a string or a list of strings")
    return tuple(parts)


def _stages(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise WorkerConfigError(f"stages must be a list of stage names (got {value!r})")
    if not value:
        raise WorkerConfigError("stages is empty; omit it to run the whole pipeline")
    return tuple(value)


def _path_list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise WorkerConfigError(f"{key} must be a path or a list of paths")


def _optional_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise WorkerConfigError(f"{key} must be a number of GB or null (got {value!r})")
    return float(value)


def worker_config_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> WorkerConfig:
    """Build a :class:`WorkerConfig` from parsed YAML.

    Relative paths resolve against ``base_dir`` -- the directory holding the
    configuration file -- never against the process's working directory.
    """
    if not isinstance(data, Mapping):
        raise WorkerConfigError("the worker configuration must be a mapping")
    missing = [key for key in _REQUIRED if data.get(key) is None]
    if missing:
        raise WorkerConfigError(
            "worker configuration is missing required key(s): " + ", ".join(missing)
        )
    known = set(WorkerConfig.__dataclass_fields__)
    unknown = sorted(set(data) - known)
    if unknown:
        raise WorkerConfigError("unknown worker configuration key(s): " + ", ".join(unknown))

    snakemake_cmd = _command(data["snakemake_cmd"], "snakemake_cmd")
    if not snakemake_cmd:
        raise WorkerConfigError("snakemake_cmd must not be empty")
    timeout = data.get("run_timeout")
    min_free_gb = _optional_float(data.get("min_free_gb"), "min_free_gb")
    resume_free_gb = _optional_float(data.get("resume_free_gb"), "resume_free_gb")
    if resume_free_gb is None and min_free_gb is not None:
        # Hysteresis by default: without a margin, free space oscillating by a
        # fraction of a GB around the limit flips pause/resume every cycle.
        resume_free_gb = min_free_gb * 1.05
    config = WorkerConfig(
        state_db=_path(data["state_db"], relative_to=base_dir, key="state_db"),
        log_dir=_path(data["log_dir"], relative_to=base_dir, key="log_dir"),
        workflow_dir=_path(data["workflow_dir"], relative_to=base_dir, key="workflow_dir"),
        pipeline_config=_path(data["pipeline_config"], relative_to=base_dir, key="pipeline_config"),
        first_shot=int(data["first_shot"]),
        poll_interval=float(data["poll_interval"]),
        quiet_seconds=float(data["quiet_seconds"]),
        cores=int(data["cores"]),
        snakemake_cmd=snakemake_cmd,
        extra_args=_command(data.get("extra_args") or [], "extra_args"),
        max_shots_per_run=int(data.get("max_shots_per_run", 20)),
        max_attempts=int(data.get("max_attempts", 3)),
        recheck_seconds=float(data.get("recheck_seconds", 3 * 86400.0)),
        max_reprocess=int(data.get("max_reprocess", 3)),
        run_timeout=None if timeout is None else float(timeout),
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        classifier=data.get("classifier"),
        record_shot_class=bool(data.get("record_shot_class", False)),
        stages=_stages(data.get("stages")),
        min_free_gb=min_free_gb,
        resume_free_gb=resume_free_gb,
        disk_paths=tuple(
            _path(item, relative_to=base_dir, key="disk_paths")
            for item in _path_list(data.get("disk_paths"), "disk_paths")
        ),
    )
    for name in ("poll_interval", "quiet_seconds", "cores", "max_shots_per_run", "max_attempts"):
        if getattr(config, name) <= 0:
            raise WorkerConfigError(f"{name} must be positive")
    if config.run_timeout is not None and config.run_timeout <= 0:
        raise WorkerConfigError("run_timeout must be positive or null")
    if config.recheck_seconds < 0 or config.max_reprocess < 0:
        raise WorkerConfigError("recheck_seconds and max_reprocess must not be negative")
    if config.min_free_gb is None and (config.resume_free_gb is not None or config.disk_paths):
        raise WorkerConfigError("resume_free_gb and disk_paths need min_free_gb")
    if config.min_free_gb is not None:
        if config.min_free_gb <= 0:
            raise WorkerConfigError("min_free_gb must be positive or null")
        if config.resume_free_gb is not None and config.resume_free_gb < config.min_free_gb:
            raise WorkerConfigError("resume_free_gb must not be below min_free_gb")
    if config.first_shot <= 0:
        raise WorkerConfigError("first_shot must be a positive shot number")
    if not config.snakefile.is_file():
        raise WorkerConfigError(f"workflow_dir has no Snakefile: {config.workflow_dir}")
    return config


def load_worker_config(path: str | os.PathLike[str] | None = None) -> WorkerConfig:
    """Read the worker configuration from ``path`` or ``$VAFT_WORKER_CONFIG``."""
    if path is None:
        path = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not path:
        raise WorkerConfigError(
            f"no worker configuration given; pass --config or set {CONFIG_ENVIRONMENT_VARIABLE}"
        )
    config_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WorkerConfigError(f"worker configuration not found: {config_path}") from None
    return worker_config_from_mapping(data or {}, base_dir=config_path.parent)


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Snakemake's own rule for ``--configfile`` over the Snakefile's ``configfile:``."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, what: str) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise WorkerConfigError(f"{what} not found: {path}") from None


def load_pipeline_config(config: WorkerConfig) -> dict[str, Any]:
    """The configuration Snakemake will actually run with.

    The Snakefile loads its own ``config.yaml`` (by the Snakefile's path, #1530)
    and Snakemake merges ``--configfile`` over it, so a key the deployment's
    file leaves out falls back to the workflow's.  The worker reproduces that merge, so the runner,
    the harvester and the checks below all see what Snakemake sees.

    Checked for the two settings a worker cannot run under: a raw stage that
    does not read SQL would never see a new shot, and a Conda directive breaks
    job bookkeeping in a non-interactive shell (DEPLOYMENT.md, section 4).
    """
    default = config.workflow_dir / "config.yaml"
    data = _read_yaml(default, "workflow configuration") if default.is_file() else {}
    if config.pipeline_config.resolve() != default.resolve():
        data = _deep_merge(data, _read_yaml(config.pipeline_config, "pipeline configuration"))
    raw = data.get("raw") or {}
    if raw.get("mode", "sql") != "sql":
        raise WorkerConfigError(
            "the worker processes shots as SQL records them, so the pipeline "
            f"configuration needs raw.mode: sql (got {raw.get('mode')!r})"
        )
    data["conda"] = None
    if config.stages is not None:
        # The worker's scope wins over the pipeline file's, so the runner's
        # per-run config, Snakemake and the harvester all narrow the same way.
        data["stages"] = list(config.stages)
    return data


__all__ = [
    "CONFIG_ENVIRONMENT_VARIABLE",
    "WorkerConfig",
    "WorkerConfigError",
    "load_pipeline_config",
    "load_worker_config",
    "worker_config_from_mapping",
]
