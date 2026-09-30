"""Read back what a pipeline run left for each shot (issue #58).

The worker never infers a shot's state from Snakemake's exit code: one run
carries several shots, ``--keep-going`` lets the healthy ones finish past a
broken one, and a legitimately empty stage (a vacuum shot's EFIT) is not a
failure.  Instead it reads, per shot, the products the routine pipeline
declares -- each stage manifest's ``status``, each replication record's
``state`` -- and the raw preflight's verdict.

The target list mirrors ``rule all`` in
``workflow/automatic_pipeline_1_routine_data_processing/Snakefile``
(``eligible_pipeline_outputs`` and ``replication_records``) and resolves every
path through that workflow's own ``paths.PipelinePaths``, so no path is spelled
here.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from string import Template
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

from vaft.database.sources import STAGE_REPLICATION, replicable_stages

from . import state as S
from .classify import RAW_ONLY


#: Statuses that mean "this stage produced what it set out to".
OK_STATUSES = frozenset({"success", "completed", "validated", "present"})
#: Present in the tree, stage finished, and nothing for the worker to retry:
#: anything else a manifest may say (``partial``, ``skipped``, ``no_output``,
#: ``failed`` for one solver cell, a ``replicated`` record whose read-back
#: disagreed ...).  Snakemake will not rebuild an existing output, so none of
#: these is retryable by re-running the workflow.
MISSING = "missing"
UNREADABLE = "unreadable"

MANIFEST = "manifest"
REPLICATION = "replication"
#: Validation plots: required rule outputs, judged by presence only.  What a
#: plot manifest says about the figures does not change the shot's state.
PLOT = "plot"
PRESENT = "present"

_STAGE_MANIFESTS = ("diagnostics", "eddy", "efit", "chease")
_STABILITY_MODULES = ("dcon", "rdcon", "stride")
#: The Snakefile's ``EMPTY_VALIDATION_STAGES``: stages whose plot set may be empty.
_EMPTY_VALIDATION_STAGES = frozenset({"chease", "eddy", "efit", "mhd_linear"})


@dataclass(frozen=True)
class StageTarget:
    stage: str
    product: str | None
    kind: str
    path: str


@dataclass(frozen=True)
class StageStatus:
    stage: str
    product: str | None
    kind: str
    status: str
    path: str

    @property
    def label(self) -> str:
        name = self.stage if self.product is None else f"{self.stage}/{self.product}"
        return name if self.kind == MANIFEST else f"{name} ({self.kind})"


@dataclass(frozen=True)
class ShotOutcome:
    state: str
    reason: str
    stages: tuple[StageStatus, ...]


def load_pipeline_paths_module(workflow_dir: Path) -> ModuleType:
    """Import the workflow's ``paths.py`` from the configured checkout.

    Loaded by file location under a private name, so it is the module that
    Snakemake itself will import from ``workflow_dir`` -- never whatever
    ``paths`` happens to be importable.
    """
    location = Path(workflow_dir) / "paths.py"
    name = f"_vaft_worker_pipeline_paths_{abs(hash(str(location.resolve())))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline paths from {location}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expand(value: Any, environment: Mapping[str, str]) -> Any:
    """Expand ``$VAR``/``${VAR}`` as the Snakefile does, from ``environment``."""
    if isinstance(value, dict):
        return {key: _expand(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, environment) for item in value]
    if isinstance(value, str) and "$" in value:
        return Template(value).safe_substitute(environment)
    return value


class PipelineHarvester:
    """Evaluate each shot against the routine pipeline's declared products."""

    def __init__(
        self,
        pipeline_config: Mapping[str, Any],
        *,
        workflow_dir: Path,
        environment: Mapping[str, str] | None = None,
    ):
        config = _expand(dict(pipeline_config), dict(os.environ if environment is None else environment))
        self.config = config
        module = load_pipeline_paths_module(workflow_dir)
        self._module = module
        self.paths = module.PipelinePaths.from_config(config)
        gpec = config.get("gpec") or {}
        self.gpec_modules = list(gpec.get("modules", ["dcon", "rdcon", "stride", "gpec"]))
        self.stability_modules = [m for m in self.gpec_modules if m in _STABILITY_MODULES]
        hsds = config.get("hsds") or {}
        self.replicate = bool(hsds.get("replicate", False)) and self.paths.layout == module.FILEDB

    # -- targets -----------------------------------------------------------
    def raw_targets(self, shot: int) -> list[StageTarget]:
        return [StageTarget("raw", None, MANIFEST, self.paths.raw_manifest(shot))]

    def raw_output_files(self, shot: int) -> list[str]:
        """The files that make up the raw stage -- Snakemake targets for a raw-only shot."""
        return [self.paths.raw_dump(shot), self.paths.raw_manifest(shot)]

    def targets(self, shot: int) -> list[StageTarget]:
        """Every product ``rule all`` requests for an eligible shot."""
        targets = self.raw_targets(shot)
        for stage in _STAGE_MANIFESTS:
            targets.append(
                StageTarget(stage, None, MANIFEST, getattr(self.paths, f"{stage}_manifest")(shot))
            )
        for module in self.stability_modules:
            targets.append(
                StageTarget("mhd_linear", module, MANIFEST, self.paths.mhd_linear_manifest(shot, module))
            )
        targets.extend(self.plot_targets(shot))
        if "gpec" in self.gpec_modules:
            targets.append(
                StageTarget("gpec_ideal", "gpec", MANIFEST, self.paths.gpec_ideal_manifest(shot, "gpec"))
            )
        if self.replicate:
            for stage in replicable_stages():
                if STAGE_REPLICATION[stage].optional:
                    continue
                products = self.stability_modules if stage == "mhd_linear" else [None]
                for product in products:
                    targets.append(
                        StageTarget(
                            stage,
                            product,
                            REPLICATION,
                            self.paths.replication_record(shot, stage, product),
                        )
                    )
        return targets

    def plot_targets(self, shot: int) -> list[StageTarget]:
        """The Snakefile's ``validation_plot_products`` for one shot."""
        if self.paths.layout != self._module.FILEDB:
            return []
        from vaft.database.production_qa import stage_plot_filenames

        p = self.paths
        targets = [StageTarget("raw", name, PLOT, p.raw_plot(shot, name))
                   for name in stage_plot_filenames("raw", required_only=True)]
        targets.append(StageTarget("raw", "plot_manifest", PLOT, p.raw_plot_manifest(shot)))
        for stage in ("diagnostics", "eddy", "efit", "chease"):
            if stage not in _EMPTY_VALIDATION_STAGES:
                targets += [StageTarget(stage, name, PLOT, p.stage_plot(shot, stage, name))
                            for name in stage_plot_filenames(stage, required_only=True)]
            targets.append(StageTarget(stage, "plot_manifest", PLOT, p.stage_plot_manifest(shot, stage)))
        for module in self.stability_modules:
            targets.append(StageTarget("mhd_linear", f"{module}/plot_manifest", PLOT,
                                       p.stage_plot_manifest(shot, "mhd_linear", module)))
        targets.append(StageTarget("chease", "refinement_plot_manifest", PLOT, p.chease_plot_manifest(shot)))
        return targets

    # -- reading -----------------------------------------------------------
    @staticmethod
    def read(target: StageTarget) -> StageStatus:
        path = Path(target.path)
        if not path.exists():
            status = MISSING
        elif target.kind == PLOT:
            status = PRESENT
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = UNREADABLE
            else:
                key = "state" if target.kind == REPLICATION else "status"
                value = data.get(key) if isinstance(data, dict) else None
                status = str(value) if value else UNREADABLE
        return StageStatus(target.stage, target.product, target.kind, status, target.path)

    def preflight_exclusion(self, shot: int) -> str | None:
        """The raw preflight's reason for excluding ``shot``, if it did."""
        path = Path(self.paths.preflight_excluded())
        try:
            entries = json.loads(path.read_text(encoding="utf-8")).get("excluded_shots", [])
        except (OSError, ValueError, AttributeError):
            return None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and str(entry.get("shot")) == str(int(shot)):
                missing = entry.get("missing_field_codes")
                detail = f" (field codes {', '.join(map(str, missing))})" if missing else ""
                return f"raw preflight: {entry.get('reason', 'excluded')}{detail}"
        return None

    def harvest(
        self, shot: int, *, applicable_stages: Sequence[str] | None, classification_reason: str | None = None
    ) -> ShotOutcome:
        """The shot's state after a run, with the status of each declared product."""
        raw_only = applicable_stages is not None and tuple(applicable_stages) == RAW_ONLY
        if raw_only:
            stages = tuple(self.read(t) for t in self.raw_targets(shot))
            if any(s.status in (MISSING, UNREADABLE) for s in stages):
                return ShotOutcome(S.FAILED, _missing_reason(stages), stages)
            return ShotOutcome(
                S.EXCLUDED,
                f"raw archived only ({classification_reason or 'classifier limited it to raw'})",
                stages,
            )

        raw = tuple(self.read(t) for t in self.raw_targets(shot))
        if all(s.status not in (MISSING, UNREADABLE) for s in raw):
            excluded = self.preflight_exclusion(shot)
            if excluded is not None:
                return ShotOutcome(S.EXCLUDED, excluded, raw)

        stages = tuple(self.read(t) for t in self.targets(shot))
        if any(s.status in (MISSING, UNREADABLE) for s in stages):
            return ShotOutcome(S.FAILED, _missing_reason(stages), stages)
        incomplete = [s for s in stages if s.status not in OK_STATUSES]
        if incomplete:
            return ShotOutcome(
                S.PARTIAL,
                "; ".join(f"{s.label}={s.status}" for s in incomplete),
                stages,
            )
        return ShotOutcome(S.COMPLETED, "every declared product succeeded", stages)


def _missing_reason(stages: Sequence[StageStatus]) -> str:
    missing = [f"{s.label}={s.status}" for s in stages if s.status in (MISSING, UNREADABLE)]
    return "incomplete outputs: " + ", ".join(missing)


__all__ = [
    "MANIFEST",
    "MISSING",
    "OK_STATUSES",
    "REPLICATION",
    "PipelineHarvester",
    "ShotOutcome",
    "StageStatus",
    "StageTarget",
    "load_pipeline_paths_module",
]
