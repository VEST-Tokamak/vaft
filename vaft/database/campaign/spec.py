"""What a campaign knows about the workflow it runs: only what the workflow states.

A workflow directory holds its ``Snakefile`` and a ``paths.py``. The campaign
reads the ``CAMPAIGN_*`` names from that ``paths.py`` (each optional) and its
``PipelinePaths``/``resource_profile_args`` helpers. Nothing about a particular
pipeline -- rule names, products, solvers -- is written in this package, so any
Snakemake workflow that states these names can run as a campaign.

``CAMPAIGN_RUN_SCOPED_RULES``
    rules whose failure fails the whole run (they make what every shot needs);
``CAMPAIGN_RUN_SCOPED_OUTPUTS``
    ``PipelinePaths`` methods (no shot argument) of run-scoped outputs, cleared
    before each chunk and kept per chunk after it;
``CAMPAIGN_REQUIRED_OUTPUTS``
    those of them a finished run must have written;
``CAMPAIGN_CHECKPOINT_RULES``, ``CAMPAIGN_PLOT_RULE_PREFIX``
    rules whose cost a benchmark reports apart;
``CAMPAIGN_SHOT_INPUTS``
    ``PipelinePaths`` methods of the files a shot needs before it can run;
``CAMPAIGN_COMPARED_PRODUCTS``, ``CAMPAIGN_COMPARED_DIRECTORIES``
    per-shot products (and directories) two runs are compared on;
``CAMPAIGN_SOLVER_PROCESSES``
    process names a lane's runaway-memory guard watches (``name*`` for a prefix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Optional, Union

__all__ = ["WorkflowSpec"]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class WorkflowSpec:
    """A workflow directory and what its ``paths.py`` states for campaigns."""

    directory: Path
    module: Optional[ModuleType] = field(default=None, compare=False, repr=False)
    run_scoped_rules: frozenset = frozenset()
    run_scoped_outputs: tuple = ()
    required_outputs: tuple = ()
    checkpoint_rules: tuple = ()
    plot_rule_prefix: Optional[str] = None
    shot_inputs: tuple = ()
    compared_products: tuple = ()
    compared_directories: Mapping[str, tuple] = field(default_factory=dict)
    solver_processes: tuple = ()

    @classmethod
    def load(cls, directory: PathLike) -> "WorkflowSpec":
        directory = Path(directory).resolve()
        if not (directory / "Snakefile").is_file():
            raise FileNotFoundError(f"{directory} has no Snakefile")
        module = None
        if (directory / "paths.py").is_file():
            from ..worker.status import load_pipeline_paths_module

            module = load_pipeline_paths_module(directory)

        def get(name: str, default: Any) -> Any:
            return getattr(module, f"CAMPAIGN_{name}", default) if module is not None else default

        return cls(
            directory=directory,
            module=module,
            run_scoped_rules=frozenset(get("RUN_SCOPED_RULES", ())),
            run_scoped_outputs=tuple(get("RUN_SCOPED_OUTPUTS", ())),
            required_outputs=tuple(get("REQUIRED_OUTPUTS", ())),
            checkpoint_rules=tuple(get("CHECKPOINT_RULES", ())),
            plot_rule_prefix=get("PLOT_RULE_PREFIX", None),
            shot_inputs=tuple(get("SHOT_INPUTS", ())),
            compared_products=tuple(get("COMPARED_PRODUCTS", ())),
            compared_directories=dict(get("COMPARED_DIRECTORIES", {})),
            solver_processes=tuple(get("SOLVER_PROCESSES", ())),
        )

    @property
    def snakefile(self) -> Path:
        return self.directory / "Snakefile"

    def merged_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """``config`` deep-merged over the workflow's ``config.yaml``, as Snakemake will see it.

        Environment references in the defaults are expanded with the workflow's
        own ``expand_config_environment`` where it has one; a reference the
        environment cannot answer leaves the defaults as written (the run config
        overrides what the campaign itself reads, e.g. ``base_dir``).
        """
        import yaml

        defaults: dict[str, Any] = {}
        location = self.directory / "config.yaml"
        if location.is_file():
            defaults = yaml.safe_load(location.read_text(encoding="utf-8")) or {}
            expand = getattr(self.module, "expand_config_environment", None) if self.module is not None else None
            if expand is not None:
                try:
                    defaults = expand(defaults)
                except Exception:  # noqa: BLE001 - an unset variable the run config overrides anyway
                    pass
        return _deep_merge(defaults, config)

    def paths(self, config: Mapping[str, Any]) -> Any:
        """The workflow's ``PipelinePaths`` for a run config, or ``None`` without one."""
        if self.module is None or not hasattr(self.module, "PipelinePaths"):
            return None
        return self.module.PipelinePaths.from_config(config)

    def resource_profile_args(self, config: Mapping[str, Any]) -> list[str]:
        """``--set-resources``/``--set-threads`` for ``config`` (already merged), via the workflow."""
        helper = getattr(self.module, "resource_profile_args", None) if self.module is not None else None
        return list(helper(config)) if helper is not None else []


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
