"""The corrective worker (#1842): upload detection, state, manifest harvest and the run cycle.

No Snakemake, no network, no repository data: the scanner reads a temporary
directory, the runner is a fake that writes the manifests a real run would, and
the harvester resolves its paths through the real routine pipeline's ``paths.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path

import pytest

from vaft.database.corrective_lineage import CORRECTIVE_LINEAGE, DEFAULT_STAGES
from vaft.database.worker import corrective as cw
from vaft.database.worker.config import WorkerConfigError
from vaft.database.worker.runner import RunPlan, RunResult

ROOT = Path(__file__).resolve().parents[1]
PIPELINE1 = ROOT / "workflow" / "automatic_pipeline_1_routine_data_processing"
PIPELINE2 = ROOT / "workflow" / "automatic_pipeline_2_corrective_data_update"

pytestmark = pytest.mark.skipif(not PIPELINE1.exists(), reason="workflow scripts are not part of the distribution")

NOW = datetime(2026, 10, 12, 12, 0, 0)


# --- helpers ---------------------------------------------------------------------------------------


def signature(*names: str, age_s: float = 3600.0, size: int = 100) -> cw.UploadSignature:
    mtime = int((NOW.timestamp() - age_s) * 1e9)
    return cw.UploadSignature(tuple(sorted((name, size, mtime) for name in names)))


class FakeScanner:
    def __init__(self, found=None):
        self.found = dict(found or {})
        self.error = None

    def scan(self):
        if self.error:
            raise OSError(self.error)
        return dict(self.found)


class Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


@pytest.fixture
def pipeline_config(tmp_path):
    return {"base_dir": str(tmp_path / "filedb"), "layout": "filedb", "shots": [],
            "external": {"data_root": str(tmp_path / "incoming")}, "hsds": {"replicate": False}}


@pytest.fixture
def config(tmp_path):
    return cw.CorrectiveWorkerConfig(
        state_db=tmp_path / "state" / "corrective.db", log_dir=tmp_path / "logs",
        workflow_dir=PIPELINE2, pipeline_config=tmp_path / "pipeline.yaml", paths_workflow_dir=PIPELINE1,
        first_shot=39000, poll_interval=1.0, quiet_seconds=60.0, cores=1, snakemake_cmd=("snakemake",),
        max_attempts=3,
    )


@pytest.fixture
def harvester(pipeline_config):
    return cw.CorrectiveHarvester(pipeline_config, paths_workflow_dir=PIPELINE1)


def write_manifest(harvester, shot, stage, status):
    path = Path(getattr(harvester.paths, f"{stage}_manifest")(shot))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage, "status": status}), encoding="utf-8")


def write_all(harvester, shot, **statuses):
    """Every stage of the lineage ``unavailable`` unless named."""
    for stage in CORRECTIVE_LINEAGE:
        write_manifest(harvester, shot, stage, statuses.get(stage, "unavailable"))


class FakeRunner:
    """Writes the manifests a run would, as scripted per shot; records the plans it was given."""

    def __init__(self, harvester, script, *, busy=False, interrupted=False, exit_code=0):
        self.harvester, self.script = harvester, script
        self.plans, self.busy, self.interrupted, self.exit_code = [], busy, interrupted, exit_code
        self.stopped = False
        self.unlocked = False

    def run(self, plan, *, on_start=None):
        self.plans.append(plan)
        if on_start:
            on_start(["snakemake"], "log", 1)
        if not (self.busy or self.interrupted):
            for shot in plan.full_shots:
                script = self.script[shot]
                if script is not None:
                    write_all(self.harvester, shot, **script)
        return RunResult(exit_code=self.exit_code, log_path="log", busy=self.busy, interrupted=self.interrupted)

    def unlock_if_stale(self):
        self.unlocked = True
        return True

    def stop(self):
        self.stopped = True


def make_worker(config, harvester, pipeline_config, scanner, runner, clock, state=None):
    return cw.CorrectiveWorker(
        config, state=state or cw.CorrectiveState(config.state_db), scanner=scanner, runner=runner,
        harvester=harvester, pipeline_config=pipeline_config, clock=clock,
    )


# --- the upload scanner ----------------------------------------------------------------------------


def test_the_scanner_groups_thomson_and_ces_files_by_shot(tmp_path):
    root = tmp_path / "incoming"
    (root / "thomson_scattering").mkdir(parents=True)
    (root / "legacy").mkdir()
    for name in ("Shot40330_v10.mat", "CES_40330.mat", "IDS_48226.mat", "NeTe_Shot48226_v9_rev.mat", "notes.txt"):
        (root / name).write_bytes(b"x")
    (root / "thomson_scattering" / "41000_NeTe.mat").write_bytes(b"x")
    (root / "legacy" / "NeTe_Shot42000.mat").write_bytes(b"x")
    found = cw.UploadScanner(root).scan()
    assert sorted(found) == [40330, 41000, 42000, 48226]
    assert [name for name, _, _ in found[40330].files] == ["CES_40330.mat", "Shot40330_v10.mat"]
    assert [name for name, _, _ in found[48226].files] == ["IDS_48226.mat", "NeTe_Shot48226_v9_rev.mat"]
    assert [name for name, _, _ in found[41000].files] == [os.path.join("thomson_scattering", "41000_NeTe.mat")]


def test_a_replaced_or_touched_upload_changes_the_signature(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    path = root / "Shot40330_v10.mat"
    path.write_bytes(b"x")
    before = cw.UploadScanner(root).scan()[40330]
    assert cw.UploadScanner(root).scan()[40330].digest == before.digest
    path.write_bytes(b"xy")
    after_size = cw.UploadScanner(root).scan()[40330]
    assert after_size.digest != before.digest
    os.utime(path, ns=(1, 2_000_000_000_000))
    assert cw.UploadScanner(root).scan()[40330].digest != after_size.digest


def test_an_unreadable_upload_root_is_an_error_not_an_empty_scan(tmp_path):
    with pytest.raises(OSError, match="not available"):
        cw.UploadScanner(tmp_path / "unmounted").scan()


# --- the state -------------------------------------------------------------------------------------


def test_a_new_upload_is_pending_and_an_unchanged_one_is_not_news(config):
    with cw.CorrectiveState(config.state_db) as state:
        assert state.observe(40330, signature("a.mat"), at=NOW) == "new"
        assert state.shot(40330)["state"] == cw.PENDING and state.shot(40330)["force"] == 0
        assert state.observe(40330, signature("a.mat"), at=NOW) is None


def test_a_changed_upload_reopens_a_settled_shot_and_forces_the_rerun(config):
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(40330, signature("a.mat"), at=NOW)
        state.new_run([40330], at=NOW)
        state.settle(40330, state=cw.COMPLETED, reason="ok", attempts=0, force=False, at=NOW)
        assert state.observe(40330, signature("a.mat", size=999), at=NOW) == "changed"
        row = state.shot(40330)
        assert row["state"] == cw.PENDING and row["force"] == 1 and row["attempts"] == 0


def test_a_change_during_a_run_is_acted_on_when_the_run_ends(config):
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(40330, signature("a.mat"), at=NOW)
        state.new_run([40330], at=NOW)
        state.observe(40330, signature("a.mat", size=999), at=NOW)
        assert state.shot(40330)["state"] == cw.RUNNING
        state.settle(40330, state=cw.COMPLETED, reason="ok", attempts=0, force=False, at=NOW)
        row = state.shot(40330)
        assert row["state"] == cw.PENDING and row["force"] == 1
        assert "changed while the run" in row["reason"]


def test_an_operator_exclusion_survives_a_changed_upload(config):
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(40330, signature("a.mat"), at=NOW)
        assert state.exclude(40330, reason="bad file")
        state.observe(40330, signature("a.mat", size=999), at=NOW)
        assert state.shot(40330)["state"] == cw.EXCLUDED
        assert state.ready(now=NOW.timestamp(), quiet_seconds=0, limit=10) == []
        assert state.requeue(40330, reason="fixed")
        assert state.shot(40330)["state"] == cw.PENDING


def test_only_quiet_pending_shots_are_ready_lowest_first_and_limited(config):
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(40002, signature("b.mat", age_s=10), at=NOW)      # still arriving
        state.observe(40003, signature("c.mat", age_s=500), at=NOW)
        state.observe(40001, signature("a.mat", age_s=500), at=NOW)
        ready = state.ready(now=NOW.timestamp(), quiet_seconds=60.0, limit=10)
        assert [row["shot"] for row in ready] == [40001, 40003]
        assert [row["shot"] for row in state.ready(now=NOW.timestamp(), quiet_seconds=60.0, limit=1)] == [40001]


def test_a_killed_worker_leaves_its_shots_pending_and_forced(config):
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(40330, signature("a.mat"), at=NOW)
        state.new_run([40330], at=NOW)
        assert state.unfinished_runs() == 1
        assert state.recover(at=NOW) == [40330]
        row = state.shot(40330)
        assert row["state"] == cw.PENDING and row["force"] == 1 and state.unfinished_runs() == 0


# --- reading back the manifests --------------------------------------------------------------------


def test_every_stage_of_the_lineage_has_a_manifest_path_in_the_routine_paths(harvester):
    stages = [t.stage for t in harvester.targets(48226)]
    assert tuple(stages) == tuple(s for s in CORRECTIVE_LINEAGE if s in DEFAULT_STAGES)
    assert all("/omas/" in t.path and t.path.endswith("manifest.json") for t in harvester.targets(48226))


def test_a_shot_with_a_product_is_completed_and_absent_stages_are_not_failures(harvester):
    write_all(harvester, 48226, thomson="success", ces="success", core_profiles="success", composition="success",
              ti_inferred="unavailable", resistive_zeff="no_output", kinetic_efit="success")
    outcome = harvester.harvest(48226)
    assert outcome.state == cw.COMPLETED
    assert outcome.reason == "produced: thomson, ces, core_profiles, composition, kinetic_efit"


def test_a_shot_where_every_stage_said_unavailable_is_unavailable_not_failed(harvester):
    write_all(harvester, 40001)
    outcome = harvester.harvest(40001)
    assert outcome.state == cw.UNAVAILABLE and "no stage produced a product" in outcome.reason


def test_a_failed_or_missing_or_unreadable_manifest_is_a_failure(harvester):
    write_all(harvester, 40001, thomson="success", composition="failed")
    assert harvester.harvest(40001).state == cw.FAILED and "composition=failed" in harvester.harvest(40001).reason
    Path(harvester.paths.ces_manifest(40001)).unlink()
    assert "ces=missing" in harvester.harvest(40001).reason
    Path(harvester.paths.ces_manifest(40001)).write_text("{not json")
    assert "ces=unreadable" in harvester.harvest(40001).reason


def test_a_failed_replication_is_a_failure_and_a_skipped_one_is_not(pipeline_config):
    pipeline_config["hsds"] = {"replicate": True}
    harvester = cw.CorrectiveHarvester(pipeline_config, paths_workflow_dir=PIPELINE1)
    write_all(harvester, 48226, thomson="success")
    for stage in CORRECTIVE_LINEAGE:
        record = Path(harvester.paths.replication_record(48226, stage))
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"state": "validated" if stage == "thomson" else "skipped"}))
    assert harvester.harvest(48226).state == cw.COMPLETED
    Path(harvester.paths.replication_record(48226, "thomson")).write_text(json.dumps({"state": "failed"}))
    outcome = harvester.harvest(48226)
    assert outcome.state == cw.FAILED and "thomson (replication)=failed" in outcome.reason


def test_a_stage_scope_judges_only_the_stages_asked_for(pipeline_config):
    pipeline_config["stages"] = ["thomson", "ces"]
    harvester = cw.CorrectiveHarvester(pipeline_config, paths_workflow_dir=PIPELINE1)
    for stage in ("thomson", "ces"):
        write_manifest(harvester, 40001, stage, "success")
    # no composition, kinetic ... manifests exist at all: out of scope, so not judged
    assert [t.stage for t in harvester.targets(40001)] == ["thomson", "ces"]
    assert harvester.harvest(40001).state == cw.COMPLETED


# --- the cycle -------------------------------------------------------------------------------------


def test_a_quiet_new_upload_is_run_harvested_and_completed(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {"thomson": "success", "core_profiles": "success"}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    report = worker.run_cycle()
    assert report.new == [48226] and report.run_shots == [48226]
    assert report.outcomes == {48226: cw.COMPLETED}
    assert runner.plans[0].force is False
    row = worker.state.shot(48226)
    assert row["state"] == cw.COMPLETED and row["attempts"] == 0 and row["force"] == 0
    assert {s["stage"] for s in worker.state.stage_status(48226)} == set(DEFAULT_STAGES)


def test_an_upload_that_is_still_arriving_waits(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat", age_s=5)})
    runner = FakeRunner(harvester, {48226: {}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    assert worker.run_cycle().run_shots == [] and runner.plans == []


def test_a_replaced_upload_reruns_the_shot_forced(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {"thomson": "success"}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    worker.run_cycle()
    scanner.found = {48226: signature("Shot48226_v10.mat")}
    report = worker.run_cycle()
    assert report.changed == [48226] and report.run_shots == [48226]
    assert runner.plans[1].force is True
    assert worker.state.shot(48226)["force"] == 0, "settled again"


def test_a_failed_shot_is_retried_forced_then_given_up_on(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {"thomson": "success", "composition": "failed"}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    first = worker.run_cycle()
    row = worker.state.shot(48226)
    assert first.outcomes == {48226: cw.PENDING} and row["attempts"] == 1 and row["force"] == 1
    worker.run_cycle()
    assert runner.plans[1].force is True
    third = worker.run_cycle()
    assert third.gave_up == [48226] and worker.state.shot(48226)["state"] == cw.GAVE_UP
    assert worker.run_cycle().run_shots == [], "a shot that was given up on is not scheduled again"
    assert worker.state.requeue(48226, reason="operator") and worker.run_cycle().run_shots == [48226]


def test_a_busy_workflow_is_not_a_failed_attempt(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {}}, busy=True, exit_code=1)
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    report = worker.run_cycle()
    row = worker.state.shot(48226)
    assert report.busy and row["state"] == cw.PENDING and row["attempts"] == 0


def test_a_requested_stop_releases_the_shots_without_judging_them(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {}}, interrupted=True)
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    worker.run_cycle()
    assert worker.state.shot(48226)["state"] == cw.PENDING and worker.state.shot(48226)["attempts"] == 0


def test_the_hold_file_stops_runs_but_not_the_recording_of_uploads(config, harvester, pipeline_config, tmp_path):
    hold = tmp_path / "HOLD"
    held_config = cw.CorrectiveWorkerConfig(**{**config.__dict__, "hold_file": hold})
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {}})
    worker = make_worker(held_config, harvester, pipeline_config, scanner, runner, Clock())
    hold.write_text("")
    report = worker.run_cycle()
    assert report.held and report.new == [48226] and runner.plans == []
    hold.unlink()
    assert worker.run_cycle().run_shots == [48226]


def test_shots_outside_the_configured_range_are_never_recorded(config, harvester, pipeline_config):
    ranged = cw.CorrectiveWorkerConfig(**{**config.__dict__, "first_shot": 40000, "last_shot": 41000})
    scanner = FakeScanner({39999: signature("a.mat"), 40500: signature("b.mat"), 41001: signature("c.mat")})
    runner = FakeRunner(harvester, {40500: {"thomson": "success"}})
    worker = make_worker(ranged, harvester, pipeline_config, scanner, runner, Clock())
    report = worker.run_cycle()
    assert report.new == [40500] and [row["shot"] for row in worker.state.shots()] == [40500]


def test_a_failed_scan_is_reported_and_the_queue_still_runs(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {"thomson": "success"}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    worker.state.observe(48226, signature("Shot48226_v9.mat"), at=NOW)
    scanner.error = "mount lost"
    report = worker.run_cycle()
    assert report.scan_error == "mount lost" and report.run_shots == [48226]


def test_recover_puts_interrupted_shots_back_and_unlocks_a_stale_lock(config, harvester, pipeline_config):
    scanner = FakeScanner({48226: signature("Shot48226_v9.mat")})
    runner = FakeRunner(harvester, {48226: {"thomson": "success"}})
    worker = make_worker(config, harvester, pipeline_config, scanner, runner, Clock())
    worker.state.observe(48226, signature("Shot48226_v9.mat"), at=NOW)
    worker.state.new_run([48226], at=NOW)
    report = worker.run_cycle(recover=True)
    assert report.recovered == [48226] and runner.unlocked
    assert report.run_shots == [48226] and runner.plans[0].force is True


# --- the runner command ----------------------------------------------------------------------------


def test_a_forced_plan_adds_forcerun_for_the_two_root_rules_after_everything_else(config, pipeline_config):
    runner = cw.CorrectiveRunner(config, pipeline_config)
    plain = runner.command(cw.CorrectivePlan(run_id=1, full_shots=(48226,)), Path("c.yaml"))
    forced = runner.command(cw.CorrectivePlan(run_id=1, full_shots=(48226,), force=True), Path("c.yaml"))
    assert "--forcerun" not in plain
    assert forced[: len(plain)] == plain
    assert forced[len(plain):] == ["--forcerun", "generate_thomson_ods", "generate_ces_ods"]
    assert forced[1] == "all"
    # a plain RunPlan (the routine worker's) is unaffected
    assert "--forcerun" not in runner.command(RunPlan(run_id=2, full_shots=(1,)), Path("c.yaml"))


# --- configuration ---------------------------------------------------------------------------------


def _workflow(tmp_path):
    p2 = tmp_path / "automatic_pipeline_2_corrective_data_update"
    p1 = tmp_path / "automatic_pipeline_1_routine_data_processing"
    p2.mkdir()
    p1.mkdir()
    (p2 / "Snakefile").write_text("")
    (p1 / "paths.py").write_text("")
    (p2 / "config.yaml").write_text("base_dir: /srv/f\nlayout: filedb\nexternal:\n  data_root: /srv/incoming\n  ces_options: ids\n")
    return p2, p1


def _mapping(p2, **extra):
    return {"state_db": "state.db", "log_dir": "logs", "workflow_dir": str(p2), "pipeline_config": "p.yaml",
            "first_shot": 39000, "poll_interval": 30, "quiet_seconds": 120, "cores": 4,
            "snakemake_cmd": "snakemake", **extra}


def test_relative_paths_resolve_against_the_config_file_and_paths_dir_defaults_beside_the_workflow(tmp_path):
    p2, p1 = _workflow(tmp_path)
    config = cw.corrective_config_from_mapping(_mapping(p2), base_dir=tmp_path / "deploy")
    assert config.state_db == tmp_path / "deploy" / "state.db"
    assert config.paths_workflow_dir == p1 and config.stages is None and config.hold_file is None


@pytest.mark.parametrize("key", ["state_db", "workflow_dir", "first_shot", "quiet_seconds", "snakemake_cmd"])
def test_a_missing_required_key_is_named(tmp_path, key):
    p2, _ = _workflow(tmp_path)
    data = _mapping(p2)
    del data[key]
    with pytest.raises(WorkerConfigError, match=key):
        cw.corrective_config_from_mapping(data, base_dir=tmp_path)


def test_unknown_keys_and_bad_values_are_refused(tmp_path):
    p2, _ = _workflow(tmp_path)
    with pytest.raises(WorkerConfigError, match="unknown"):
        cw.corrective_config_from_mapping(_mapping(p2, classifier="x"), base_dir=tmp_path)
    with pytest.raises(WorkerConfigError, match="last_shot"):
        cw.corrective_config_from_mapping(_mapping(p2, last_shot=10), base_dir=tmp_path)
    with pytest.raises(WorkerConfigError, match="stages"):
        cw.corrective_config_from_mapping(_mapping(p2, stages=["composition"]), base_dir=tmp_path)
    with pytest.raises(WorkerConfigError, match="positive"):
        cw.corrective_config_from_mapping(_mapping(p2, cores=0), base_dir=tmp_path)


def test_the_pipeline_config_is_the_workflows_own_overlaid_by_the_deployments_and_needs_an_upload_root(tmp_path):
    p2, _ = _workflow(tmp_path)
    (tmp_path / "p.yaml").write_text("external:\n  data_root: /mnt/uploads\nhsds:\n  replicate: true\n")
    config = cw.corrective_config_from_mapping(_mapping(p2, stages=["thomson", "ces"]), base_dir=tmp_path)
    merged = cw.load_corrective_pipeline_config(config)
    assert merged["external"] == {"data_root": "/mnt/uploads", "ces_options": "ids"}
    assert merged["stages"] == ["thomson", "ces"] and merged["conda"] is None
    (tmp_path / "p.yaml").write_text("external:\n  data_root: ''\n")
    with pytest.raises(WorkerConfigError, match="data_root"):
        cw.load_corrective_pipeline_config(config)


# --- the command line ------------------------------------------------------------------------------


def _cli_config(tmp_path):
    p2, p1 = _workflow(tmp_path)
    (tmp_path / "p.yaml").write_text("external:\n  data_root: " + str(tmp_path / "incoming") + "\n")
    deploy = tmp_path / "corrective.yaml"
    deploy.write_text(
        "state_db: state.db\nlog_dir: logs\nworkflow_dir: " + str(p2) + "\npipeline_config: p.yaml\n"
        "first_shot: 39000\npoll_interval: 30\nquiet_seconds: 0\ncores: 1\nsnakemake_cmd: snakemake\n"
    )
    return deploy


def test_the_cli_status_retry_and_exclude_act_on_the_state_file(tmp_path, capsys):
    from vaft.cli.corrective_worker import main

    deploy = _cli_config(tmp_path)
    config = cw.load_corrective_config(deploy)
    with cw.CorrectiveState(config.state_db) as state:
        state.observe(48226, signature("a.mat"), at=NOW)
        state.new_run([48226], at=NOW)
        state.settle(48226, state=cw.FAILED, reason="composition=failed", attempts=1, force=True, at=NOW)
    assert main(["status", "--config", str(deploy), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"] == {"failed": 1} and summary["attention"][0]["shot"] == 48226
    assert main(["status", "--config", str(deploy), "--shot", "48226"]) == 0
    assert "composition=failed" in capsys.readouterr().out
    assert main(["retry", "--config", str(deploy), "--shot", "48226"]) == 0
    assert main(["exclude", "--config", str(deploy), "--shot", "48226", "--reason", "bad upload"]) == 0
    assert main(["status", "--config", str(deploy), "--shot", "99999"]) == 1
    assert main(["retry", "--config", str(deploy), "--shot", "99999"]) == 1


def test_the_cli_refuses_a_second_worker_on_the_same_state_file(tmp_path, capsys):
    from vaft.cli.corrective_worker import main
    from vaft.cli.pipeline_worker import worker_lock

    deploy = _cli_config(tmp_path)
    config = cw.load_corrective_config(deploy)
    with worker_lock(config.state_db):
        assert main(["run", "--config", str(deploy), "--once"]) == 2
    assert "another pipeline worker holds" in capsys.readouterr().err


def test_the_command_is_registered_with_the_dispatcher():
    from vaft.cli import _main

    module, summary = _main._COMMANDS["corrective-worker"]
    assert module == ".corrective_worker" and "Thomson" in summary


def test_the_example_configuration_names_only_known_keys_and_all_required_ones():
    import yaml

    example = yaml.safe_load((PIPELINE2 / "corrective.example.yaml").read_text(encoding="utf-8"))
    assert set(cw._REQUIRED) <= set(example)
    assert set(example) <= set(cw.CorrectiveWorkerConfig.__dataclass_fields__)
    # and a stage list the example suggests in its comments would be accepted
    from vaft.database.corrective_lineage import stage_scope

    assert stage_scope(["thomson", "ces", "core_profiles"])


def test_the_example_loads_once_its_variables_and_paths_exist(tmp_path, monkeypatch):
    import shutil

    checkout = tmp_path / "checkout"
    shutil.copytree(ROOT / "workflow", checkout / "workflow", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    monkeypatch.setenv("VAFT_CHECKOUT", str(checkout))
    monkeypatch.setenv("VAFT_FILEDB_DIR", str(tmp_path / "filedb"))
    deploy = tmp_path / "corrective.yaml"
    deploy.write_text((PIPELINE2 / "corrective.example.yaml").read_text(encoding="utf-8"))
    config = cw.load_corrective_config(deploy)
    assert config.state_db == tmp_path / "filedb" / "pipeline" / "corrective" / "corrective.sqlite"
    assert config.hold_file == tmp_path / "filedb" / "pipeline" / "corrective" / "HOLD"
    assert config.paths_workflow_dir == checkout / "workflow" / "automatic_pipeline_1_routine_data_processing"
    assert config.snakemake_cmd == ("snakemake",) and config.stages is None
