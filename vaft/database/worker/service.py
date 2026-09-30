"""The new-shot pipeline worker's cycle (issue #58).

One cycle: recover what a crash left behind, detect new SQL shots, settle and
classify them, run the routine pipeline for a batch, and harvest each shot's
products into durable state.  Every step is idempotent, so a cycle can be
killed at any point and the next one continues from the state file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Protocol

from . import state as S
from .classify import RAW_ONLY, Classifier, ShotObservation, load_classifier
from .config import WorkerConfig, load_pipeline_config
from .poll import ShotSource, SqlShotSource, is_settled
from .runner import RunPlan, RunResult, SnakemakeRunner
from .status import PipelineHarvester


logger = logging.getLogger(__name__)


class Runner(Protocol):
    def run(self, plan: RunPlan, *, on_start: Callable[[list[str], str, int], None] | None = None) -> RunResult: ...

    def unlock_after_crash(self, stale_pid: int | None) -> bool: ...

    def stop(self) -> None: ...


@dataclass
class CycleReport:
    recovered: list[int] = field(default_factory=list)
    detected: list[int] = field(default_factory=list)
    queued: list[int] = field(default_factory=list)
    excluded: list[int] = field(default_factory=list)
    run_id: int | None = None
    run_shots: list[int] = field(default_factory=list)
    busy: bool = False
    outcomes: dict[int, str] = field(default_factory=dict)
    gave_up: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


class PipelineWorker:
    """Poll SQL, classify, run and harvest.  Collaborators are injectable for tests."""

    def __init__(
        self,
        config: WorkerConfig,
        *,
        state: S.WorkerState | None = None,
        source: ShotSource | None = None,
        classifier: Classifier | None = None,
        runner: Runner | None = None,
        harvester: PipelineHarvester | None = None,
        pipeline_config: Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.config = config
        self.pipeline_config = dict(pipeline_config) if pipeline_config is not None else load_pipeline_config(config)
        self.state = state if state is not None else S.WorkerState(config.state_db)
        self.source = source if source is not None else SqlShotSource()
        self.classifier = classifier if classifier is not None else load_classifier(
            config.classifier, self.pipeline_config
        )
        self.runner = runner if runner is not None else SnakemakeRunner(config, self.pipeline_config)
        self.harvester = harvester if harvester is not None else PipelineHarvester(
            self.pipeline_config, workflow_dir=config.workflow_dir, environment=config.environment()
        )
        self.clock = clock
        self._stop = threading.Event()

    # -- the cycle ---------------------------------------------------------
    def recover(self) -> list[int]:
        """Undo what a previous worker left mid-run; see :meth:`WorkerState.recover`."""
        stale = self.state.unfinished_runs()
        shots = self.state.recover()
        for run in stale:
            if self.runner.unlock_after_crash(run.get("pid")):
                self.state.event("unlocked", run_id=run["run_id"], detail="stale worker run")
        if shots:
            logger.warning("recovered shots left running by a previous worker: %s", shots)
        return shots

    def detect(self) -> list[int]:
        after = self.state.watermark(self.config.first_shot - 1)
        found = self.source.new_shots(after)
        added = self.state.add_detected(found)
        for shot in added:
            logger.info("detected shot %s", shot)
        return added

    def settle(self, report: CycleReport) -> None:
        now = self.clock()
        for row in self.state.shots(state=S.DETECTED):
            shot = row["shot"]
            codes = self.source.field_codes(shot)
            previous = self.state.observe_field_count(shot, len(codes))
            if not is_settled(
                # A row SQL recorded without a time settles from when the
                # worker first saw it, rather than never.
                record_datetime=row["record_datetime"] or row["detected_at"],
                field_count=len(codes),
                previous_field_count=previous,
                settle_seconds=self.config.settle_seconds,
                now=now,
            ):
                continue
            verdict = self.classifier.classify(
                ShotObservation(shot=shot, record_datetime=row["record_datetime"], field_codes=codes)
            )
            if verdict.runnable:
                self.state.settle(
                    shot,
                    state=S.QUEUED,
                    classification=verdict.label,
                    reason=verdict.reason,
                    applicable_stages=verdict.applicable_stages,
                )
                report.queued.append(shot)
                logger.info("queued shot %s (%s)", shot, verdict.label)
            else:
                self.state.settle(
                    shot,
                    state=S.EXCLUDED,
                    classification=verdict.label,
                    reason=verdict.reason,
                    applicable_stages=(),
                )
                report.excluded.append(shot)
                logger.info("excluded shot %s: %s", shot, verdict.reason)

    def run_batch(self, report: CycleReport) -> None:
        batch = self.state.runnable(
            limit=self.config.max_shots_per_run, max_attempts=self.config.max_attempts
        )
        if not batch:
            return
        full = tuple(row["shot"] for row in batch if row["applicable_stages"] is None)
        raw_only = [row["shot"] for row in batch if row["applicable_stages"] == RAW_ONLY]
        unsupported = [
            row for row in batch if row["applicable_stages"] not in (None, RAW_ONLY)
        ]
        for row in unsupported:
            # A classifier asked for a stage subset the routine pipeline cannot
            # request on its own yet (#57).  Refuse rather than run everything.
            self.state.conclude(
                row["shot"],
                state=S.GAVE_UP,
                reason=f"unsupported stage subset {list(row['applicable_stages'])} (#57)",
                run_id=None,
                count_attempt=False,
            )
        if not [*full, *raw_only]:
            return
        run_id, shots = self.state.start_run([*full, *raw_only])
        full = tuple(shot for shot in full if shot in shots)
        raw_only = [shot for shot in raw_only if shot in shots]
        report.run_id, report.run_shots = run_id, shots
        if not shots:
            self.state.finish_run(run_id, exit_code=None, outcome="empty")
            return
        try:
            self._execute(run_id, full, raw_only, batch, report)
        except BaseException as error:
            # Never leave shots `running` behind a live worker: recovery only
            # happens at start-up.  A persistent error (a wrong snakemake_cmd)
            # spends attempts and ends in gave_up rather than looping forever.
            self.state.finish_run(run_id, exit_code=None, outcome="error")
            for shot in shots:
                row = self.state.shot(shot)
                if row is not None and row["state"] == S.RUNNING:
                    self.state.conclude(
                        shot, state=S.FAILED, reason=f"worker error: {type(error).__name__}: {error}",
                        run_id=run_id, count_attempt=isinstance(error, Exception),
                    )
            raise

    def _execute(self, run_id: int, full: tuple[int, ...], raw_only: list[int],
                 batch: list[dict[str, Any]], report: CycleReport) -> None:
        shots = [*full, *raw_only]
        file_targets = tuple(
            path for shot in raw_only for path in self.harvester.raw_output_files(shot)
        )
        plan = RunPlan(run_id=run_id, full_shots=full, file_targets=file_targets)
        logger.info("run %s: pipeline for %s, raw only for %s", run_id, list(full), raw_only)

        def started(command: list[str], log_path: str, pid: int) -> None:
            self.state.record_run_process(run_id, command=command, log_path=log_path, pid=pid)

        result = self.runner.run(plan, on_start=started)
        if result.busy:
            self.state.finish_run(run_id, exit_code=result.exit_code, outcome="busy")
            for shot in shots:
                self.state.conclude(
                    shot, state=S.QUEUED, reason="workflow directory locked by another run",
                    run_id=run_id, count_attempt=False,
                )
            report.busy = True
            logger.warning("run %s: workflow directory is locked; will retry", run_id)
            return

        applicable = {row["shot"]: row for row in batch}
        outcomes = {
            shot: self.harvester.harvest(
                shot,
                applicable_stages=applicable[shot]["applicable_stages"],
                classification_reason=applicable[shot]["classification"],
            )
            for shot in shots
        }
        # The raw preflight is one checkpoint over every shot in the run, so a
        # shot whose raw dump failed holds back the rest of the batch.  Only
        # that shot spends an attempt; the others are requeued as blocked.
        raw_failed = sorted(
            shot for shot, outcome in outcomes.items()
            if shot in full and outcome.state == S.FAILED and outcome.stages
            and outcome.stages[0].stage == "raw" and outcome.stages[0].status in ("missing", "unreadable")
        )
        for shot in shots:
            outcome = outcomes[shot]
            self.state.record_stage_status(shot, outcome.stages, run_id=run_id)
            if outcome.state == S.FAILED and result.interrupted:
                # A stop request is not the shot's fault; do not spend an attempt.
                self.state.conclude(shot, state=S.QUEUED, reason="interrupted: " + outcome.reason,
                                    run_id=run_id, count_attempt=False)
                report.outcomes[shot] = S.QUEUED
                continue
            if outcome.state == S.FAILED and shot in full and raw_failed and shot not in raw_failed:
                reason = (f"blocked: raw stage failed for shot(s) {raw_failed} in the same run; "
                          + outcome.reason)
                self.state.conclude(shot, state=S.QUEUED, reason=reason, run_id=run_id,
                                    count_attempt=False)
                report.outcomes[shot] = S.QUEUED
                continue
            if outcome.state == S.FAILED and result.timed_out:
                reason = f"run timed out after {self.config.run_timeout:g} s; {outcome.reason}"
            else:
                reason = outcome.reason
            self.state.conclude(
                shot, state=outcome.state, reason=reason, run_id=run_id,
                count_attempt=outcome.state == S.FAILED,
            )
            report.outcomes[shot] = outcome.state
            log = logger.warning if outcome.state == S.FAILED else logger.info
            log("shot %s: %s (%s)", shot, outcome.state, reason)
            if self.config.record_shot_class and outcome.state in (S.COMPLETED, S.PARTIAL):
                self._record_shot_class(shot)
        outcome_label = "interrupted" if result.interrupted else (
            "timed_out" if result.timed_out else ("ok" if result.exit_code == 0 else "failed")
        )
        self.state.finish_run(run_id, exit_code=result.exit_code, outcome=outcome_label)

    def _record_shot_class(self, shot: int) -> None:
        """Reference-only physics class from the diagnostics product (#57 will gate on it)."""
        try:
            from vaft.omas import load
            from vaft.omas.shot_class import shot_class

            product = Path(self.harvester.paths.diagnostics_ods(shot))
            verdict = shot_class(load(product))
        except Exception as error:  # reference information must never fail a shot
            self.state.event("shot_class_failed", shot=shot, detail=f"{type(error).__name__}: {error}")
            return
        self.state.set_reference_class(shot, verdict.label)

    def run_cycle(self, *, recover: bool = False) -> CycleReport:
        report = CycleReport()
        if recover:
            report.recovered = self.recover()
        report.detected = self.detect()
        self.settle(report)
        if not self._stop.is_set():
            self.run_batch(report)
        report.gave_up = self.state.give_up_exhausted(self.config.max_attempts)
        return report

    # -- the loop ----------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()
        self.runner.stop()

    def run_forever(self) -> None:
        """Cycle until :meth:`stop`; the first cycle recovers a crashed predecessor.

        An error in one cycle -- SQL unreachable, a malformed product -- is
        logged and recorded, and the next cycle tries again: a transient
        outage must not end the service.
        """
        first = True
        while not self._stop.is_set():
            try:
                report = self.run_cycle(recover=first)
                first = False
                if report.run_id is not None and not report.busy and self.state.shots(state=S.QUEUED):
                    continue  # a backlog is waiting; do not sleep between batches
            except Exception as error:
                logger.exception("worker cycle failed")
                try:
                    self.state.event("cycle_error", detail=f"{type(error).__name__}: {error}")
                except Exception:
                    logger.exception("could not record the cycle error")
            self._stop.wait(self.config.poll_interval)


__all__ = ["CycleReport", "PipelineWorker"]
