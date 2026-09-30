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
from vaft.database.worker.poll import SqlShotSource, upload_finished
from vaft.database.worker.runner import HSDS_RESOURCE, RunPlan, RunResult, SnakemakeRunner, _runs_in
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
#: "The upload finished long ago" -- past any quiet_seconds.
DONE = 10**6


class FakeSource:
    """SQL as the worker sees it.  Returns every shot, ignoring ``after``,
    so each poll re-observes the known ones -- the duplicate case.

    ``quiet`` is seconds since the shot's last field row; the default says
    the upload is long over, ``0`` that it is still arriving."""

    def __init__(self):
        self.shots: dict[int, tuple[datetime, frozenset[int], float]] = {}

    def add(self, shot, recorded=T0, fields=FULL_FIELDS, quiet=DONE):
        self.shots[shot] = (recorded, frozenset(fields), quiet)

    def new_shots(self, after):
        return sorted((shot, entry[0]) for shot, entry in self.shots.items())

    def upload_status(self, shot):
        return self.shots[shot][1], self.shots[shot][2]

    def field_codes_by_shot(self, shots):
        return {shot: self.shots[shot][1] for shot in shots if shot in self.shots}


class FakeRunner:
    """Writes the declared products instead of running Snakemake.

    ``script`` maps shot -> list of per-run behaviours: ``"ok"`` writes every
    target with success, ``"fail"`` writes raw and diagnostics only,
    ``"skipped"`` marks EFIT skipped, ``"preflight"`` writes raw and a
    preflight exclusion.
    """

    def __init__(self, harvester: PipelineHarvester, source: FakeSource | None = None):
        self.harvester = harvester
        self.source = source
        #: shot -> field codes the dump fails to load (SQL lists them anyway)
        self.dropped: dict[int, set[int]] = {}
        self.plans: list[RunPlan] = []
        self.script: dict[int, list[str]] = {}
        self.busy = False
        self.stale_lock = False
        self.unlocks = 0

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
                payload = {"status": status}
                if target.stage == "raw" and target.kind == "manifest" and self.source is not None:
                    # What the dump holds: SQL's inventory at run time.
                    payload["inventory"] = {"field_codes": sorted(
                        set(self.source.shots[shot][1]) - self.dropped.get(shot, set()))}
                self._write(target.path, payload)
        for path in plan.file_targets:
            if path.endswith(".json"):
                self._write(path, {"status": "success"})
        return RunResult(exit_code=1 if failed else 0, log_path="")

    def unlock_if_stale(self):
        if self.stale_lock:
            self.unlocks += 1
            self.busy = False  # the lock is gone
            return True
        return False

    def stop(self):
        pass


@pytest.fixture()
def setup(tmp_path, monkeypatch):
    # The fake runner never starts Snakemake, so the process scan the re-check
    # makes before moving a raw dump would only see other sessions' runs.
    monkeypatch.setattr("vaft.database.worker.service.live_snakemake_in", lambda workflow_dir: False)
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
            "quiet_seconds": 600,
            "cores": 2,
            "snakemake_cmd": "snakemake",
            "max_attempts": 2,
            "max_reprocess": 1,
        },
        base_dir=tmp_path,
    )
    clock = {"now": T0 + timedelta(hours=1)}

    def make_worker(runner=None, source=None, state=None):
        pipeline_config = load_pipeline_config(config)
        harvester = PipelineHarvester(pipeline_config, workflow_dir=WORKFLOW, environment={})
        runner = runner or FakeRunner(harvester, source)
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


def settle_cycles(worker, n=1):
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
    assert first.detected == [101] and first.queued == [101]
    assert first.outcomes == {101: S.COMPLETED}
    for _ in range(4):
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


def test_a_shot_runs_as_soon_as_its_inventory_matches_the_previous_shots(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(100)  # the reference: finished long ago
    source.add(101, fields={1, 12}, quiet=5)
    worker, runner = make_worker(source=source)

    assert worker.run_cycle().queued == [100]
    assert worker.state.shot(101)["state"] == S.DETECTED  # still uploading
    source.add(101, fields=FULL_FIELDS, quiet=5)  # every field shot 100 had ...
    assert worker.run_cycle().queued == []  # ... but the last one landed 5 s ago
    source.add(101, fields=FULL_FIELDS, quiet=40)
    report = worker.run_cycle()
    assert report.queued == [101] and report.outcomes == {101: S.COMPLETED}
    (enqueued,) = [e for e in worker.state.events(shot=101) if e["kind"] == "enqueued"]
    assert "matches the previous shot" in enqueued["detail"]


def test_a_shot_missing_a_previous_field_waits_for_the_quiet_limit(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(100, fields={*FULL_FIELDS, 102})  # had Plasma Current
    source.add(101, fields=FULL_FIELDS, quiet=120)  # Ip not (yet) uploaded
    worker, _ = make_worker(source=source)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.DETECTED
    source.add(101, fields=FULL_FIELDS, quiet=600)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED
    (enqueued,) = [e for e in worker.state.events(shot=101) if e["kind"] == "enqueued"]
    assert "no field uploaded for 600 s" in enqueued["detail"]


@pytest.mark.parametrize(
    ("codes", "reference", "quiet", "finished"),
    [
        ({1, 2}, {1, 2}, 40, True),      # matches the previous shot
        ({1, 2, 3}, {1, 2}, 40, True),   # a superset is fine
        ({1, 2}, {1, 2}, 5, False),      # a match mid-upload is not finished yet
        ({1}, {1, 2}, 0, False),         # still waiting for field 2
        ({1}, {1, 2}, 600, True),        # ... until the upload is quiet
        ({1}, None, 0, False),           # no reference: quiet decides
        ((), {1}, 600, True),            # nothing uploaded at all, and quiet
        ((), None, None, False),         # shot row gone: never decide blindly
    ],
)
def test_upload_finished(codes, reference, quiet, finished):
    result = upload_finished(
        field_codes=frozenset(codes), reference=None if reference is None else frozenset(reference),
        quiet_seconds=quiet, quiet_limit=600,
    )
    assert (result is not None) is finished


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
    source.add(101, quiet=0)
    source.add(102, quiet=0)
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
    runner.stale_lock = True
    assert restarted.state.watermark(0) == 102
    report = restarted.run_cycle(recover=True)
    assert report.recovered == [101, 102]
    assert runner.unlocks == 1
    assert report.detected == []  # nothing detected twice
    assert report.outcomes == {101: S.COMPLETED, 102: S.COMPLETED}
    assert [run["outcome"] for run in restarted.state.runs()] == ["abandoned", "ok"]


def test_operator_exclusion_is_final(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, quiet=0)
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
             "first_shot": 1, "quiet_seconds": 1, "cores": 1, "snakemake_cmd": "snakemake"},
            base_dir=tmp_path,
        )
    with pytest.raises(WorkerConfigError, match="unknown"):
        worker_config_from_mapping(
            {"state_db": "s", "log_dir": "l", "workflow_dir": "w", "pipeline_config": "p",
             "first_shot": 1, "poll_interval": 1, "quiet_seconds": 1, "cores": 1,
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
    with pytest.raises(FileNotFoundError):
        worker.run_cycle()
    row = worker.state.shot(101)
    assert row["state"] == S.FAILED and row["attempts"] == 1
    assert worker.state.runs()[-1]["outcome"] == "error"


def test_a_shot_excluded_after_selection_is_not_claimed(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, quiet=0)
    source.add(102, quiet=0)
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
    source.add(101, quiet=0)
    worker, runner = make_worker(source=source)
    worker.run_cycle()  # detected, still uploading
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
        "first_shot": 100, "poll_interval": 60, "quiet_seconds": 300, "cores": 2,
        "snakemake_cmd": "snakemake",
    }), encoding="utf-8")
    assert pipeline_worker.main(["status", "--config", str(config_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"] == {"completed": 1} and summary["watermark"] == 101
    assert pipeline_worker.main(["status", "--config", str(config_path), "--shot", "101"]) == 0
    assert "completed" in capsys.readouterr().out


def _fake_pool(monkeypatch, answer):
    queries = []

    class Cursor:
        rows: list = []

        def execute(self, query, params):
            queries.append(query)
            self.rows = answer(query)

        def fetchone(self):
            return self.rows[0] if self.rows else None

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
    return queries


def test_shot_upload_status_reads_the_table_holding_the_shot(monkeypatch):
    def answer(query):
        if "COUNT" in query:
            return [(0,)] if "shotDataWaveform_2" in query else [(4,)]
        # (field, seconds since that row was uploaded); 110 is a processed
        # triple-probe signal the raw dump leaves out.
        return [(1, 300), (12, 90), (25, 45), (110, 40)]

    queries = _fake_pool(monkeypatch, answer)
    codes, quiet = raw.shot_upload_status(48900)
    assert codes == frozenset({1, 12, 25}) and quiet == 40
    assert "shotDataWaveform_3" in queries[-1] and "recordDataTime" in queries[-1]


def test_a_shot_without_fields_is_quiet_since_its_record(monkeypatch):
    def answer(query):
        if "COUNT" in query:
            return [(0,)]
        return [(700,)]  # seconds since shot.recordDateTime

    queries = _fake_pool(monkeypatch, answer)
    assert raw.shot_upload_status(48901) == (frozenset(), 700.0)
    assert "FROM shot WHERE" in queries[-1]


def test_field_codes_by_shot_keeps_the_fuller_table(monkeypatch):
    def answer(query):
        if "shotDataWaveform_2" in query:
            return [(48900, 1), (48900, 2)]
        return [(48900, 1), (48900, 2), (48900, 3), (48901, 5), (48901, 111)]

    _fake_pool(monkeypatch, answer)
    assert raw.field_codes_by_shot([48901, 48900]) == {48900: frozenset({1, 2, 3}), 48901: frozenset({5})}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_killed_run_releases_its_own_lock(setup, tmp_path):
    config, _, _ = setup
    marker = tmp_path / "unlocked"
    fake = tmp_path / "fake_snakemake.sh"
    fake.write_text(
        "#!/bin/sh\n"
        f'case " $* " in *" --unlock "*) touch "{marker}"; exit 0;; esac\n'
        "sleep 30\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    config = dataclasses.replace(config, snakemake_cmd=(str(fake),), run_timeout=0.5)
    runner = SnakemakeRunner(config, load_pipeline_config(config))
    runner.terminate_grace = 0.5
    # A Snakemake that ignores SIGTERM.  Ignored from birth -- inherited across
    # exec -- rather than by a `trap` the child might not reach before the
    # signal on a loaded machine.
    import signal

    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        result = runner.run(RunPlan(run_id=1, full_shots=(101,)))
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert result.timed_out and result.exit_code != 0
    assert marker.exists()  # SIGKILL leaves a lock only the worker could have held


def test_an_empty_shot_is_excluded_once_quiet_and_revived_by_a_late_upload(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, fields=(), quiet=100)
    worker, runner = make_worker(source=source)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.DETECTED  # the DAQ may still write
    source.add(101, fields=(), quiet=600)
    worker.run_cycle()
    assert worker.state.shot(101)["classification"] == DAQ_MISSING
    assert runner.plans == []

    source.add(101, fields=FULL_FIELDS, quiet=10)  # the upload arrived after all
    report = worker.run_cycle()
    assert report.reprocessed == [101]
    source.add(101, fields=FULL_FIELDS, quiet=600)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED


def test_late_fields_reprocess_a_finished_shot(setup):
    config, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED
    raw_manifest = Path(worker.harvester.paths.raw_manifest(101))
    assert raw_manifest.exists()

    # The next day VEST backfills Plasma Current for this shot.
    source.add(101, fields={*FULL_FIELDS, 102})
    report = worker.run_cycle()
    assert report.reprocessed == [101]
    assert report.outcomes == {101: S.COMPLETED}  # re-classified and re-run in the same cycle
    assert len(runner.plans) == 2
    row = worker.state.shot(101)
    assert row["reprocessed"] == 1 and 102 in row["field_codes"]
    # The old dump was moved aside, never deleted.
    (moved,) = (config.log_dir / "superseded" / "101").glob("*/" + raw_manifest.name)
    assert json.loads(moved.read_text())["inventory"]["field_codes"] == sorted(FULL_FIELDS)
    assert [e["kind"] for e in worker.state.events(shot=101)].count("reprocess") == 1

    # max_reprocess=1: a second late arrival is recorded, not acted on.
    source.add(101, fields={*FULL_FIELDS, 102, 13})
    worker.run_cycle()
    worker.run_cycle()
    assert len(runner.plans) == 2
    kinds = [e["kind"] for e in worker.state.events(shot=101)]
    assert kinds.count("late_fields_ignored") == 1


@pytest.mark.parametrize("live", [True, None])
def test_late_fields_are_not_acted_on_while_another_snakemake_runs(setup, monkeypatch, live):
    """cold review 0.8.0 workflows F3: the raw dump is a manual run's input too.

    ``recheck`` moved it before ``run_batch`` took the Snakemake lock, so a
    manual run on the same shot lost its input mid-run.  A live (or
    undeterminable) Snakemake in the workflow directory defers the move.
    """
    config, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED
    raw_manifest = Path(worker.harvester.paths.raw_manifest(101))

    scans = []

    def fake_live(workflow_dir):
        scans.append(workflow_dir)
        return live

    monkeypatch.setattr("vaft.database.worker.service.live_snakemake_in", fake_live)
    source.add(101, fields={*FULL_FIELDS, 102})
    report = worker.run_cycle()

    assert scans == [config.workflow_dir]
    assert report.reprocessed == [] and len(runner.plans) == 1
    assert raw_manifest.exists() and not (config.log_dir / "superseded").exists()
    row = worker.state.shot(101)
    assert row["state"] == S.COMPLETED and row["reprocessed"] == 0
    kinds = [e["kind"] for e in worker.state.events(shot=101)]
    assert kinds.count("recheck_deferred") == 1 and "reprocess" not in kinds

    # Once the other run is gone the late fields are acted on as before.
    monkeypatch.setattr("vaft.database.worker.service.live_snakemake_in", lambda workflow_dir: False)
    report = worker.run_cycle()
    assert report.reprocessed == [101] and len(runner.plans) == 2
    assert worker.state.shot(101)["reprocessed"] == 1


def test_a_stale_lock_found_while_busy_is_removed(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    runner.busy = runner.stale_lock = True
    report = settle_cycles(worker)[-1]
    assert report.busy and runner.unlocks == 1
    assert "stale" in worker.state.shot(101)["reason"]
    worker.run_cycle()
    assert worker.state.shot(101)["state"] == S.COMPLETED


def test_retrying_a_raw_only_shot_classifies_it_again(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, fields={1, 12})
    worker, _ = make_worker(source=source)
    settle_cycles(worker)
    assert worker.state.shot(101)["state"] == S.EXCLUDED
    assert worker.state.requeue(101, reason="upload finished")
    assert worker.state.shot(101)["state"] == S.DETECTED


def test_a_relative_base_dir_is_harvested_where_snakemake_writes_it(setup):
    config, _, _ = setup
    pipeline_config = load_pipeline_config(config)
    pipeline_config["base_dir"] = "relative/filedb"
    harvester = PipelineHarvester(pipeline_config, workflow_dir=WORKFLOW, environment={})
    assert harvester.raw_targets(101)[0].path.startswith(str(WORKFLOW / "relative" / "filedb"))


@pytest.mark.parametrize(
    ("cmdline", "cwd", "expected"),
    [
        (["python", "/env/bin/snakemake", "--cores", "4"], str(WORKFLOW), True),
        (["snakemake", "all", "--directory", str(WORKFLOW)], "/", True),
        (["snakemake", "--directory", "/elsewhere"], str(WORKFLOW), False),
        (["python", "generate_eddy_ods.py"], str(WORKFLOW), False),
        ([], str(WORKFLOW), False),
    ],
)
def test_which_processes_count_as_a_snakemake_in_the_workflow(cmdline, cwd, expected):
    assert _runs_in(cmdline, cwd, WORKFLOW) is expected


def test_run_timeout_and_reprocess_are_validated(tmp_path):
    base = {"state_db": "s", "log_dir": "l", "workflow_dir": str(WORKFLOW), "pipeline_config": "p",
            "first_shot": 1, "poll_interval": 1, "quiet_seconds": 300, "cores": 1,
            "snakemake_cmd": "snakemake"}
    with pytest.raises(WorkerConfigError, match="run_timeout"):
        worker_config_from_mapping({**base, "run_timeout": 0}, base_dir=tmp_path)
    with pytest.raises(WorkerConfigError, match="max_reprocess"):
        worker_config_from_mapping({**base, "max_reprocess": -1}, base_dir=tmp_path)


def test_an_operator_excluded_shot_is_never_reprocessed(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101)
    worker, runner = make_worker(source=source)
    worker.run_cycle()
    assert worker.state.exclude(101, reason="operator: bad calibration")
    source.add(101, fields={*FULL_FIELDS, 102})  # a late field
    report = worker.run_cycle()
    assert report.reprocessed == [] and len(runner.plans) == 1
    assert worker.state.shot(101)["state"] == S.EXCLUDED
    assert not worker.state.reopen(101, reason="must refuse")


def test_a_field_the_dump_drops_is_not_a_late_arrival(setup):
    _, make_worker, _ = setup
    source = FakeSource()
    source.add(101, fields={*FULL_FIELDS, 250})
    worker, runner = make_worker(source=source)
    runner.dropped[101] = {250}  # listed by SQL, but its series fails to load
    worker.run_cycle()
    for _ in range(3):
        assert worker.run_cycle().reprocessed == []
    assert len(runner.plans) == 1


def test_a_schema_1_state_file_is_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "INSERT INTO meta VALUES ('schema_version', '1');"
        "CREATE TABLE shots (shot INTEGER PRIMARY KEY, record_datetime TEXT, detected_at TEXT NOT NULL,"
        " field_count INTEGER, field_count_at TEXT, settled_at TEXT, state TEXT NOT NULL,"
        " classification TEXT, applicable_stages TEXT, reason TEXT, reference_class TEXT,"
        " attempts INTEGER NOT NULL DEFAULT 0, last_run_id INTEGER, updated_at TEXT NOT NULL);"
        "INSERT INTO shots(shot, detected_at, state, updated_at) VALUES (5, 'x', 'completed', 'x');"
    )
    conn.commit()
    conn.close()
    with S.WorkerState(path) as state:
        row = state.shot(5)
        assert row["reprocessed"] == 0 and row["field_codes"] is None
        assert state.exclude(5, reason="operator")
