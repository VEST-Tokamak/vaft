"""The new-shot pipeline worker (issue #58).

Detection, settling, classification, scheduling, restart recovery and retry
are exercised against a fake SQL source and a fake runner that writes the
products the routine pipeline declares, resolved through the pipeline's own
``paths.py`` -- so the harvest reads real paths without running Snakemake.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pytest
import yaml

from vaft.database import raw
from vaft.database.worker import state as S
from vaft.database.worker.classify import (
    DAQ_MISSING,
    RAW_INCOMPLETE,
    RAW_ONLY,
    RawFieldClassifier,
    ShotObservation,
)
from vaft.database.worker.config import (
    WorkerConfigError,
    load_pipeline_config,
    worker_config_from_mapping,
)
from vaft.database.worker.poll import SqlShotSource, is_settled
from vaft.database.worker.runner import HSDS_RESOURCE, RunPlan, RunResult, SnakemakeRunner
from vaft.database.worker.service import PipelineWorker
from vaft.database.worker.status import PipelineHarvester


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / "workflow" / "automatic_pipeline_1_routine_data_processing"
REQUIRED = (1, 12, 25, 59, 109)
FULL_FIELDS = frozenset({*REQUIRED, 2, 3})
T0 = datetime(2026, 9, 30, 12, 0, 0)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeSource:
    """SQL as the worker sees it.  Returns every shot, ignoring ``after``,
    so each poll re-observes the known ones -- the duplicate case."""

    def __init__(self):
        self.shots: dict[int, tuple[datetime, frozenset[int]]] = {}

    def add(self, shot, recorded=T0, fields=FULL_FIELDS):
        self.shots[shot] = (recorded, frozenset(fields))

    def new_shots(self, after):
        return sorted((shot, recorded) for shot, (recorded, _) in self.shots.items())

    def field_codes(self, shot):
        return self.shots[shot][1]


class FakeRunner:
    """Writes the declared products instead of running Snakemake.

    ``script`` maps shot -> list of per-run behaviours: ``"ok"`` writes every
    target with success, ``"fail"`` writes raw and diagnostics only,
    ``"skipped"`` marks EFIT skipped, ``"preflight"`` writes raw and a
    preflight exclusion.
    """

    def __init__(self, harvester: PipelineHarvester):
        self.harvester = harvester
        self.plans: list[RunPlan] = []
        self.script: dict[int, list[str]] = {}
        self.busy = False
        self.unlocked: list[int | None] = []

    def _write(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def run(self, plan, *, on_start=None):
        self.plans.append(plan)
        if on_start is not None:
            on_start(["snakemake"], "/dev/null", 4242)
        if self.busy:
            return RunResult(exit_code=1, log_path="", busy=True)
        failed = False
        for shot in plan.full_shots:
            behaviour = (self.script.get(shot) or ["ok"]).pop(0) if self.script.get(shot) else "ok"
            targets = self.harvester.targets(shot)
            if behaviour == "preflight":
                self._write(targets[0].path, {"status": "success"})
                self._write(
                    self.harvester.paths.preflight_excluded(),
                    {"excluded_shots": [{"shot": shot, "reason": "missing_required_raw_signal",
                                          "missing_field_codes": [109]}]},
                )
                continue
            if behaviour == "fail":
                targets, failed = targets[:2], True
            elif behaviour == "rawfail":
                targets, failed = [], True
            elif behaviour == "blocked":  # raw done, preflight never ran
                targets, failed = targets[:1], True
            for target in targets:
                status = "skipped" if behaviour == "skipped" and target.stage == "efit" else "success"
                self._write(target.path, {"status": status})
        for path in plan.file_targets:
            if path.endswith(".json"):
                self._write(path, {"status": "success"})
        return RunResult(exit_code=1 if failed else 0, log_path="")

    def unlock_after_crash(self, stale_pid):
        self.unlocked.append(stale_pid)
        return True

    def stop(self):
        pass


@pytest.fixture()
def setup(tmp_path):
    pipeline = {
        "base_dir": str(tmp_path / "filedb"),
        "layout": "filedb",
        "equilibrium_family": "magnetic",
        "refinement": "chease",
        "raw": {"mode": "sql", "preflight": {"required_fields": list(REQUIRED)}},
        "gpec": {"modules": ["dcon"]},
        "hsds": {"replicate": False},
    }
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    config = worker_config_from_mapping(
        {
            "state_db": "state/worker.sqlite",
            "log_dir": "logs",
            "workflow_dir": str(WORKFLOW),
            "pipeline_config": str(pipeline_path),
            "first_shot": 100,
            "poll_interval": 60,
            "settle_seconds": 300,
            "cores": 2,
            "snakemake_cmd": "snakemake",
            "max_attempts": 2,
        },
        base_dir=tmp_path,
    )
    clock = {"now": T0 + timedelta(hours=1)}

    def make_worker(runner=None, source=None, state=None):
        pipeline_config = load_pipeline_config(config)
        harvester = PipelineHarvester(pipeline_config, workflow_dir=WORKFLOW, environment={})
        runner = runner or FakeRunner(harvester)
        worker = PipelineWorker(
            config,
            state=state or S.WorkerState(config.state_db),
            source=source,
            runner=runner,
            harvester=harvester,
            pipeline_config=pipeline_config,
            clock=lambda: clock["now"],
        )
        return worker, runner

    return config, make_worker, clock


def settle_cycles(worker, n=2):
    """Detection and settling need two polls to agree on the field count."""
    reports = [worker.run_cycle() for _ in range(n)]
    return reports


# --------------------------------------------------------------------------- #
# Detection and settling
# --------------------------------------------------------------------------- #
def test_a_new_shot_is_detected_and_queued_exactly_once(setup):
    config, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)

    first = worker.run_cycle()
    assert first.detected == [101]
    assert first.run_id is None  # one poll cannot show the inventory is stable
    second = worker.run_cycle()
    assert second.detected == []
    assert second.queued == [101]
    assert second.outcomes == {101: S.COMPLETED}
    for _ in range(3):
        report = worker.run_cycle()
        assert report.detected == [] and report.run_id is None

    assert [plan.full_shots for plan in runner.plans] == [(101,)]
    kinds = [event["kind"] for event in worker.state.events(shot=101)]
    assert kinds.count("detected") == 1 and kinds.count("enqueued") == 1
    assert worker.state.watermark(0) == 101


def test_shots_below_first_shot_are_not_the_workers(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, _ = make_worker(source=source)
    calls = []
    original = source.new_shots
    source.new_shots = lambda after: calls.append(after) or original(after)
    worker.run_cycle()
    assert calls == [99]  # first_shot=100, so SQL is asked for shot >= 100


def test_a_shot_still_being_written_is_not_queued(setup):
    _, make_worker, clock = setup
    source = FakeSource()
    source.add(101, fields={1, 12})
    worker, runner = make_worker(source=source)

    worker.run_cycle()
    source.add(101, fields={1, 12, 25})  # inventory still growing
    assert worker.run_cycle().queued == []
    assert worker.run_cycle().queued == [101]  # unchanged for two polls now

    source.add(102, recorded=clock["now"] - timedelta(seconds=10))
    settle_cycles(worker, 3)
    assert worker.state.shot(102)["state"] == S.DETECTED  # too recent
    clock["now"] += timedelta(minutes=10)
    worker.run_cycle()
    assert worker.state.shot(102)["state"] != S.DETECTED


@pytest.mark.parametrize(
    ("count", "previous", "age", "expected"),
    [(5, None, 900, False), (5, 4, 900, False), (5, 5, 10, False), (5, 5, 900, True), (0, 0, 900, True)],
)
def test_is_settled(count, previous, age, expected):
    assert is_settled(
        record_datetime=T0, field_count=count, previous_field_count=previous,
        settle_seconds=300, now=T0 + timedelta(seconds=age),
    ) is expected


# --------------------------------------------------------------------------- #
# Classifier first
# --------------------------------------------------------------------------- #
def test_raw_field_classifier():
    classifier = RawFieldClassifier(REQUIRED)
    obs = lambda codes: ShotObservation(1, T0, frozenset(codes))
    assert classifier.classify(obs(())).label == DAQ_MISSING
    assert classifier.classify(obs(())).applicable_stages == ()
    partial = classifier.classify(obs({1, 12}))
    assert partial.label == RAW_INCOMPLETE and partial.applicable_stages == RAW_ONLY
    assert "25, 59, 109" in partial.reason
    assert classifier.classify(obs(FULL_FIELDS)).applicable_stages is None


def test_classification_happens_before_any_work_is_scheduled(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, fields=())         # DAQ wrote nothing
    source.add(102, fields={1, 12})    # required fields missing
    source.add(103)
    worker, runner = make_worker(source=source)
    settle_cycles(worker)

    assert worker.state.shot(101)["state"] == S.EXCLUDED
    assert worker.state.shot(101)["classification"] == DAQ_MISSING
    (plan,) = runner.plans
    assert plan.full_shots == (103,)  # 101 never reaches Snakemake
    assert sorted(plan.file_targets) == sorted(worker.harvester.raw_output_files(102))
    assert worker.state.shot(102)["state"] == S.EXCLUDED
    assert "raw archived only" in worker.state.shot(102)["reason"]
    assert worker.state.shot(103)["state"] == S.COMPLETED

    for _ in range(3):
        worker.run_cycle()
    assert len(runner.plans) == 1  # excluded shots are never retried


def test_the_raw_preflight_verdict_excludes_a_shot(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.script[101] = ["preflight"]
    settle_cycles(worker)
    row = worker.state.shot(101)
    assert row["state"] == S.EXCLUDED
    assert "missing_required_raw_signal" in row["reason"] and "109" in row["reason"]


def test_an_intentionally_empty_stage_is_partial_not_failed(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.script[101] = ["skipped"]
    settle_cycles(worker)
    row = worker.state.shot(101)
    assert row["state"] == S.PARTIAL and row["attempts"] == 0
    assert "efit=skipped" in row["reason"]
    statuses = {(s["stage"], s["product"]): s["status"] for s in worker.state.stage_status(101)}
    assert statuses[("efit", "")] == "skipped"
    assert statuses[("diagnostics", "")] == "success"


# --------------------------------------------------------------------------- #
# Retry and recovery
# --------------------------------------------------------------------------- #
def test_a_failed_shot_is_retried_and_completes(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.script[101] = ["fail", "ok"]

    settle_cycles(worker)
    row = worker.state.shot(101)
    assert row["state"] == S.FAILED and row["attempts"] == 1
    assert "incomplete outputs" in row["reason"]

    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED
    assert [plan.full_shots for plan in runner.plans] == [(101,), (101,)]


def test_a_shot_that_keeps_failing_is_given_up(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.script[101] = ["fail", "fail", "ok"]
    for _ in range(5):
        worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.GAVE_UP
    assert len(runner.plans) == 2  # max_attempts

    assert worker.state.requeue(101, reason="operator retry")
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED


def test_one_bad_raw_dump_does_not_spend_the_batchs_attempts(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    source.add(102)
    worker, runner = make_worker(source=source)
    runner.script[101] = ["rawfail", "rawfail"]
    runner.script[102] = ["blocked", "blocked", "ok"]
    for _ in range(4):
        worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.GAVE_UP
    row = worker.state.shot(102)
    assert row["state"] == S.COMPLETED and row["attempts"] == 0
    assert any("blocked: raw stage failed for shot(s) [101]" in (e["detail"] or "")
               for e in worker.state.events(shot=102))


def test_a_busy_workflow_directory_does_not_spend_an_attempt(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.busy = True
    report = settle_cycles(worker)[-1]
    assert report.busy
    row = worker.state.shot(101)
    assert row["state"] == S.QUEUED and row["attempts"] == 0
    runner.busy = False
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED


def test_restart_resumes_a_run_the_previous_worker_left_behind(setup):
    config, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    source.add(102)
    worker, _ = make_worker(source=source)
    worker.run_cycle()
    # The worker queued, started a run and died before harvesting it.
    for shot in (101, 102):
        worker.state.settle(shot, state=S.QUEUED, classification="unclassified",
                            reason="", applicable_stages=None)
    run_id, claimed = worker.state.start_run([101, 102])
    assert claimed == [101, 102]
    worker.state.record_run_process(run_id, command=["snakemake"], log_path="x", pid=999999)
    worker.state.close()

    restarted, runner = make_worker(source=source, state=S.WorkerState(config.state_db))
    assert restarted.state.watermark(0) == 102
    report = restarted.run_cycle(recover=True)
    assert report.recovered == [101, 102]
    assert runner.unlocked == [999999]
    assert report.detected == []  # nothing detected twice
    assert report.outcomes == {101: S.COMPLETED, 102: S.COMPLETED}
    assert [run["outcome"] for run in restarted.state.runs()] == ["abandoned", "ok"]


def test_operator_exclusion_is_final(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    worker.run_cycle()
    worker.state.settle(101, state=S.QUEUED, classification="unclassified", reason="", applicable_stages=None)
    assert worker.state.exclude(101, reason="operator: calibration shot")
    worker.run_cycle()
    assert runner.plans == []


# --------------------------------------------------------------------------- #
# Runner, configuration, CLI and SQL
# --------------------------------------------------------------------------- #
def test_the_snakemake_command(setup, tmp_path):
    config, _, _ = setup
    runner = SnakemakeRunner(config, load_pipeline_config(config))
    plan = RunPlan(run_id=7, full_shots=(101, 103), file_targets=("/x/raw.json.gz",))
    configfile = runner.write_configfile(plan)
    written = yaml.safe_load(configfile.read_text())
    assert written["shots"] == [101, 103] and written["conda"] is None
    command = runner.command(plan, configfile)
    # Targets precede every option: `--resources` takes nargs="*" and would
    # swallow a trailing target (seen in a real dry-run).
    assert command[:3] == ["snakemake", "all", "/x/raw.json.gz"]
    for flag in ("--keep-going", "--rerun-incomplete"):
        assert flag in command
    assert command[command.index("--scheduler") + 1] == "greedy"
    assert command[command.index("--resources") + 1] == HSDS_RESOURCE == "hsds=1"
    assert command[command.index("--directory") + 1] == str(WORKFLOW)
    assert command[-1] == HSDS_RESOURCE
    assert not any(flag.startswith("--force") for flag in command)

    raw_only = runner.command(RunPlan(run_id=8, full_shots=(), file_targets=("/x/a",)), configfile)
    assert "all" not in raw_only


def test_worker_config_requires_its_server_settings(tmp_path):
    with pytest.raises(WorkerConfigError, match="poll_interval"):
        worker_config_from_mapping(
            {"state_db": "s", "log_dir": "l", "workflow_dir": "w", "pipeline_config": "p",
             "first_shot": 1, "settle_seconds": 1, "cores": 1, "snakemake_cmd": "snakemake"},
            base_dir=tmp_path,
        )
    with pytest.raises(WorkerConfigError, match="unknown"):
        worker_config_from_mapping(
            {"state_db": "s", "log_dir": "l", "workflow_dir": "w", "pipeline_config": "p",
             "first_shot": 1, "poll_interval": 1, "settle_seconds": 1, "cores": 1,
             "snakemake_cmd": "snakemake", "pol_interval": 5},
            base_dir=tmp_path,
        )


def test_the_pipeline_must_read_sql(setup, tmp_path):
    config, _, _ = setup
    config.pipeline_config.write_text(yaml.safe_dump({"raw": {"mode": "archive"}}), encoding="utf-8")
    with pytest.raises(WorkerConfigError, match="raw.mode: sql"):
        load_pipeline_config(config)


def test_the_example_configuration_loads(monkeypatch, tmp_path):
    monkeypatch.setenv("VAFT_CHECKOUT", str(REPO))
    monkeypatch.setenv("VAFT_FILEDB_DIR", str(tmp_path))
    config = worker_config_from_mapping(
        yaml.safe_load((WORKFLOW / "worker.example.yaml").read_text(encoding="utf-8")),
        base_dir=WORKFLOW,
    )
    assert config.workflow_dir == WORKFLOW
    monkeypatch.delenv("VAFT_CHECKOUT")
    with pytest.raises(WorkerConfigError, match="unset environment variable"):
        worker_config_from_mapping(
            yaml.safe_load((WORKFLOW / "worker.example.yaml").read_text(encoding="utf-8")),
            base_dir=WORKFLOW,
        )


def test_the_pipeline_config_is_merged_over_the_workflows_own(setup):
    config, _, _ = setup
    config.pipeline_config.write_text(yaml.safe_dump({"base_dir": "/x"}), encoding="utf-8")
    merged = load_pipeline_config(config)
    # Keys the deployment file leaves out come from workflow/config.yaml, as in Snakemake.
    workflow_default = yaml.safe_load((WORKFLOW / "config.yaml").read_text(encoding="utf-8"))
    assert merged["base_dir"] == "/x"
    assert merged["gpec"] == workflow_default["gpec"]
    assert merged["conda"] is None


def test_a_worker_error_mid_run_does_not_strand_shots(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)

    def explode(plan, *, on_start=None):
        raise FileNotFoundError("snakemake: not found")

    runner.run = explode
    worker.run_cycle()
    with pytest.raises(FileNotFoundError):
        worker.run_cycle()
    row = worker.state.shot(101)
    assert row["state"] == S.FAILED and row["attempts"] == 1
    assert worker.state.runs()[-1]["outcome"] == "error"


def test_a_shot_excluded_after_selection_is_not_claimed(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    source.add(102)
    worker, _ = make_worker(source=source)
    worker.run_cycle()
    for shot in (101, 102):
        worker.state.settle(shot, state=S.QUEUED, classification="unclassified",
                            reason="", applicable_stages=None)
    worker.state.exclude(102, reason="operator")
    _, claimed = worker.state.start_run([101, 102])
    assert claimed == [101]
    assert worker.state.shot(102)["state"] == S.EXCLUDED


def test_retrying_a_never_settled_shot_classifies_it_again(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    worker.run_cycle()  # detected, not yet settled
    worker.state.exclude(101, reason="operator")
    worker.state.requeue(101, reason="changed my mind")
    assert worker.state.shot(101)["state"] == S.DETECTED


def test_missing_sql_credentials_stop_the_worker_without_prompting(monkeypatch, tmp_path):
    monkeypatch.setattr(raw, "mysql_connector", object())
    monkeypatch.setattr(raw, "CONFIG_FILE", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("the worker must never prompt"))
    with pytest.raises(WorkerConfigError, match="credentials"):
        SqlShotSource()


def test_one_worker_per_state_file(tmp_path):
    from vaft.cli.pipeline_worker import WorkerAlreadyRunningError, worker_lock

    with worker_lock(tmp_path / "w.sqlite"):
        with pytest.raises(WorkerAlreadyRunningError):
            with worker_lock(tmp_path / "w.sqlite"):
                pass


def test_status_is_readable_read_only(setup, capsys):
    config, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, _ = make_worker(source=source)
    settle_cycles(worker)
    worker.state.close()

    reader = S.read_worker_state(config.state_db)
    assert reader.shot(101)["state"] == S.COMPLETED
    with pytest.raises(Exception):
        reader.requeue(101, reason="read-only")
    reader.close()

    from vaft.cli import pipeline_worker

    config_path = config.state_db.parent.parent / "worker.yaml"
    config_path.write_text(yaml.safe_dump({
        "state_db": str(config.state_db), "log_dir": str(config.log_dir),
        "workflow_dir": str(config.workflow_dir), "pipeline_config": str(config.pipeline_config),
        "first_shot": 100, "poll_interval": 60, "settle_seconds": 300, "cores": 2,
        "snakemake_cmd": "snakemake",
    }), encoding="utf-8")
    assert pipeline_worker.main(["status", "--config", str(config_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"] == {"completed": 1} and summary["watermark"] == 101
    assert pipeline_worker.main(["status", "--config", str(config_path), "--shot", "101"]) == 0
    assert "completed" in capsys.readouterr().out


def test_shot_field_codes_reads_the_table_holding_the_shot(monkeypatch):
    queries = []

    class Cursor:
        def __init__(self):
            self.rows = []

        def execute(self, query, params):
            queries.append(query)
            if "COUNT" in query:
                self.rows = [(0,)] if "shotDataWaveform_2" in query else [(3,)]
            else:
                self.rows = [(1,), (12,), (25,)]

        def fetchone(self):
            return self.rows[0]

        def fetchall(self):
            return self.rows

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    class Pool:
        def get_connection(self):
            return Conn()

    monkeypatch.setattr(raw, "DB_POOL", Pool())
    monkeypatch.setattr(raw, "_GENERATION_CACHE", {})
    assert raw.shot_field_codes(48900) == frozenset({1, 12, 25})
    assert "shotDataWaveform_3" in queries[-1]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_killed_run_releases_its_own_lock(setup, tmp_path):
    config, _, _ = setup
    marker = tmp_path / "unlocked"
    fake = tmp_path / "fake_snakemake.sh"
    fake.write_text(
        "#!/bin/sh\n"
        "trap '' TERM\n"  # first, so a slow start cannot let SIGTERM win
        f'case " $* " in *" --unlock "*) touch "{marker}"; exit 0;; esac\n'
        "sleep 30\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    config = dataclasses.replace(config, snakemake_cmd=(str(fake),), run_timeout=2.0)
    runner = SnakemakeRunner(config, load_pipeline_config(config))
    runner.terminate_grace = 0.5
    result = runner.run(RunPlan(run_id=1, full_shots=(101,)))
    assert result.timed_out and result.exit_code != 0
    assert marker.exists()  # SIGKILL leaves a lock only the worker could have held
