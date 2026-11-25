"""Campaign orchestration (vaft.database.campaign): lanes, the lane runner, chunk
verdicts, Snakemake log costs (vaft.code.snakemake), the benchmark report and
product equivalence. Nothing here needs a scheduler: the runner is exercised on a
toy workflow with local Snakemake, and launching through a backend with the local
one.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vaft.code.snakemake import occupancy, parse_log, rule_costs
from vaft.database.campaign import (
    LaneOptions,
    LaneState,
    WorkflowSpec,
    chunk_verdict,
    compare_products,
    disjoint,
    lane_report,
    plan_lanes,
    run_lane,
    split_chunks,
    summarize,
)
from vaft.database.campaign.__main__ import main as campaign_main

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow" / "automatic_pipeline_1_routine_data_processing"
SPEC = WorkflowSpec.load(WORKFLOW)
RUN_SCOPED = SPEC.run_scoped_rules


# -- lanes -------------------------------------------------------------------

def test_shots_are_dealt_round_robin_over_sorted_unique_shots():
    assert plan_lanes([5, 1, 3, 2, 4, 3], 2) == [[1, 3, 5], [2, 4]]
    with pytest.raises(ValueError):
        plan_lanes([1], 0)


@pytest.mark.skipif(shutil.which("split") is None, reason="needs coreutils split")
def test_chunks_match_the_shell_split_names_and_contents(tmp_path):
    shots = list(range(100, 1105))
    (tmp_path / "shots.txt").write_text("".join(f"{s}\n" for s in shots))
    (tmp_path / "shell").mkdir()
    subprocess.run(["split", "-l", "500", "-d", "-a", "3", str(tmp_path / "shots.txt"),
                    str(tmp_path / "shell" / "shots.")], check=True)
    split_chunks(shots, 500, tmp_path / "py")
    shell = {p.name: p.read_text() for p in (tmp_path / "shell").iterdir()}
    python = {p.name: p.read_text() for p in (tmp_path / "py").iterdir()}
    assert shell == python and sorted(python) == ["shots.000", "shots.001", "shots.002"]


def test_a_lane_resumes_from_its_done_chunks_and_keeps_its_invariants(tmp_path):
    lane = LaneState(tmp_path / "lane0")
    split_chunks(list(range(10)), 4, lane.chunks_dir)
    with pytest.raises(FileExistsError):
        split_chunks(list(range(10)), 4, lane.chunks_dir)
    assert lane.pending() == ["shots.000", "shots.001", "shots.002"]
    lane.mark_done("shots.000")
    lane.mark_done("shots.000")
    assert lane.done() == ["shots.000"]
    assert lane.pending(max_chunks=1) == ["shots.001"]
    lane.ensure("PIN_SHA", "abc")
    lane.ensure("PIN_SHA", "abc")
    with pytest.raises(ValueError, match="fixed at PIN_SHA=abc"):
        lane.ensure("PIN_SHA", "def")
    assert not lane.stop_requested
    (lane.path / "STOP").touch()
    assert lane.stop_requested
    assert lane.shots() == list(range(10))
    with pytest.raises(KeyError):
        lane.mark_done("shots.099")


def test_disjoint_lists_shots_in_two_runs():
    assert disjoint([1, 2, 3], [3, 4], [4, 5, 1]) == [1, 3, 4]
    assert disjoint([1, 1], [2]) == []


def test_plan_refuses_shots_another_run_already_holds(tmp_path, capsys):
    old = tmp_path / "old"
    split_chunks([1, 2, 3], 2, old / "lane0" / "chunks")
    (tmp_path / "shots.txt").write_text("3\n4\n5\n")
    assert campaign_main(["plan", str(tmp_path / "shots.txt"), str(tmp_path / "new"),
                          "--lanes", "2", "--chunk", "1", "--exclude", str(old)]) == 2
    (tmp_path / "shots.txt").write_text("4\n5\n6\n")
    assert campaign_main(["plan", str(tmp_path / "shots.txt"), str(tmp_path / "new"),
                          "--lanes", "2", "--chunk", "1", "--exclude", str(old)]) == 0
    assert LaneState(tmp_path / "new" / "lane0").shots() == [4, 6]


# -- chunk verdict ----------------------------------------------------------------

_PER_SHOT = """Error in rule generate_eddy_ods:
    jobid: 12
Exiting because a job execution failed. Look above for error message
WorkflowError:
At least one job did not complete successfully.
"""


@pytest.mark.parametrize(
    ("returncode", "log", "eligible", "done", "reason"),
    [
        (0, "", True, True, ""),
        (0, "", False, False, "required output not written"),
        (1, _PER_SHOT, True, True, ""),
        (1, _PER_SHOT, False, False, "required output not written"),
        (1, "Killed\n", True, False, "without a failed rule"),
        (1, "Error in rule a:\n", True, False, "did not finish its keep-going"),
        (1, _PER_SHOT + "LockException\n", True, False, "LockException"),
        (1, _PER_SHOT + "Traceback (most recent call last):\n", True, False, "traceback"),
        (1, _PER_SHOT.replace("generate_eddy_ods", "preflight_raw_dumps"), True, False, "run-scoped"),
        (1, _PER_SHOT + "WorkflowError:\nSomething else broke\n", True, False, "Something else broke"),
    ],
)
def test_a_chunk_verdict_separates_per_shot_failures_from_failed_runs(returncode, log, eligible, done, reason):
    missing = () if eligible else ("preflight_eligible",)
    verdict = chunk_verdict(returncode, log, run_scoped_rules=RUN_SCOPED, missing_outputs=missing)
    assert verdict.done is done
    assert reason in verdict.reason


# -- Snakemake log -----------------------------------------------------------

_LOG = """Provided cores: 4
[Tue Nov  3 12:00:00 2026]
localcheckpoint preflight_raw_dumps:
    jobid: 1
    threads: 4
[Tue Nov  3 12:01:00 2026]
Finished jobid: 1 (Rule: preflight_raw_dumps)
[Tue Nov  3 12:01:00 2026]
localrule generate_diagnostics_ods:
    jobid: 2
    wildcards: shot=39915
[Tue Nov  3 12:01:00 2026]
localrule plot_raw:
    jobid: 3
    wildcards: shot=39915
[Tue Nov  3 12:01:30 2026]
Finished jobid: 3 (Rule: plot_raw)
[Tue Nov  3 12:02:00 2026]
Error in rule generate_diagnostics_ods:
    message: None
    jobid: 2
[Tue Nov  3 12:02:00 2026]
localrule generate_eddy_ods:
    jobid: 4
"""


def test_the_log_gives_each_jobs_rule_wildcards_threads_and_interval():
    jobs = parse_log(_LOG)
    assert [(j.rule, j.threads, j.seconds, j.failed) for j in jobs] == [
        ("preflight_raw_dumps", 4, 60.0, False),
        ("generate_diagnostics_ods", 1, 60.0, True),
        ("plot_raw", 1, 30.0, False),
        ("generate_eddy_ods", 1, None, False),
    ]
    assert jobs[1].wildcards == "shot=39915"
    costs = rule_costs(jobs)
    assert costs["preflight_raw_dumps"]["job_seconds"] == 240.0 and costs["generate_diagnostics_ods"]["failed"] == 1
    assert "generate_eddy_ods" not in costs


def test_slot_occupancy_counts_threads_held_over_the_span():
    stats = occupancy(parse_log(_LOG), cores=4)
    # 12:00-12:01 all 4 slots; 12:01-12:01:30 two; 12:01:30-12:02 one -> (240+60+30)/(120*4)
    assert stats["span_s"] == 120.0
    assert stats["slot_occupancy"] == pytest.approx(330 / 480)
    assert stats["below_quarter"] == pytest.approx(30 / 120)


# -- benchmark report ----------------------------------------------------------

def test_a_lane_report_combines_the_lane_record_and_the_snakemake_log(tmp_path):
    lane = tmp_path / "lane0"
    (lane / "logs").mkdir(parents=True)
    usage = {"wall_s": 120.0, "cpu_s": 240.0, "booked_cpus": 4, "utilization": 0.5, "peak_mem_mb": 3584.0,
             "max_process_rss_mb": 1024.0, "source": "cgroup"}
    (lane / "lane.jsonl").write_text(json.dumps(
        {"chunk": "shots.000", "shots": 2, "ran": 2, "returncode": 1, "done": True, "cores": 4,
         "usage": usage, "log": "shots.000.attempt-1.log"}) + "\n")
    (lane / "logs" / "shots.000.attempt-1.log").write_text(_LOG)
    report = lane_report(lane, SPEC)
    totals = report["totals"]
    assert totals["shots"] == 2 and totals["cpu_util"] == pytest.approx(0.5)
    assert totals["booked_cpu_h_per_shot"] == pytest.approx(120 * 4 / 3600 / 2)
    assert totals["shots_per_hour"] == pytest.approx(60.0)
    assert totals["mem_max_gb"] == 3.5 and totals["max_rss_gb"] == 1.0
    assert totals["checkpoint_share"] == pytest.approx(60 / 120)
    assert totals["plot_share"] == pytest.approx(30 / (240 + 60 + 30))
    assert report["rules"]["plot_raw"]["jobs"] == 1
    # Without the workflow's spec the pipeline-specific shares are not guessed.
    plain = lane_report(lane)["totals"]
    assert plain["plot_share"] is None and plain["checkpoint_share"] is None and plain["cpu_util"] == 0.5


# -- product equivalence -------------------------------------------------------

def _paths(root):
    from vaft.database.worker.status import load_pipeline_paths_module

    return load_pipeline_paths_module(WORKFLOW).PipelinePaths.from_config({"base_dir": str(root), "layout": "filedb"})


def _write_gz(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(data, handle)


def _shot(root, shot, *, value=1.0, created="2026-11-03", kfile=b"K"):
    p = _paths(root)
    _write_gz(p.diagnostics_ods(shot), {"magnetics": {"ip": {"data": [value, 2.0]},
                                                      "ids_properties": {"creation_date": created}}})
    Path(p.diagnostics_manifest(shot)).parent.mkdir(parents=True, exist_ok=True)
    Path(p.diagnostics_manifest(shot)).write_text(json.dumps({"status": "success", "created": created}))
    status = Path(p.efit_status(shot))
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text("success\n")
    (status.parent.parent / "kfile").mkdir(exist_ok=True)
    (status.parent.parent / "kfile" / "k039915.00300").write_bytes(kfile)
    (status.parent.parent / "log").mkdir(exist_ok=True)
    (status.parent.parent / "log" / "efit.log").write_text(created)


def test_products_are_same_provenance_or_differs(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    _shot(ref, 1)
    _shot(new, 1)
    _shot(ref, 2)
    _shot(new, 2, created="2026-11-04")
    _shot(ref, 3)
    _shot(new, 3, value=1.5, kfile=b"K2")
    report = compare_products(new, ref, [1, 2, 3], spec=SPEC)
    assert report[1]["diagnostics_ods"]["verdict"] == "same"
    assert report[2]["diagnostics_ods"]["verdict"] == "provenance"
    assert report[2]["diagnostics_manifest"]["verdict"] == "provenance"
    assert report[2]["efit_files"]["verdict"] == "same"  # logs aside
    assert report[3]["diagnostics_ods"]["verdict"] == "differs"
    assert report[3]["diagnostics_ods"]["content"] == ["magnetics.ip.data"]
    assert report[3]["efit_files"]["differing"] == ["kfile/k039915.00300"]
    assert report[1]["eddy_ods"]["verdict"] == "absent_both"
    assert summarize(report)["diagnostics_ods"] == {"same": 1, "provenance": 1, "differs": 1}


def test_the_reference_is_taken_from_whichever_lane_holds_the_shot(tmp_path):
    lane0, lane1, new = tmp_path / "lane0", tmp_path / "lane1", tmp_path / "new"
    _shot(lane1, 7)
    _shot(new, 7)
    report = compare_products(new, lane0, [7], spec=SPEC, ref_roots=[lane0, lane1])
    assert report[7]["diagnostics_ods"]["verdict"] == "same"


# -- resource profile ----------------------------------------------------------

def test_a_resource_profile_becomes_set_resources_and_set_threads():
    from vaft.database.worker.status import load_pipeline_paths_module

    helper = load_pipeline_paths_module(WORKFLOW).resource_profile_args
    assert helper({}) == []
    assert helper({"resource_profile": {"run_chease": {"mem_mb": 4000}, "run_efit_reconstruction": {"mem_mb": 2000, "threads": 1}}}) == [
        "--set-resources", "run_chease:mem_mb=4000", "run_efit_reconstruction:mem_mb=2000",
        "--set-threads", "run_efit_reconstruction=1",
    ]
    with pytest.raises(ValueError, match="takes only mem_mb and threads"):
        helper({"resource_profile": {"run_chease": {"disk_mb": 1}}})


def test_the_worker_books_rules_from_the_same_profile(tmp_path):
    from vaft.database.worker.runner import RunPlan, SnakemakeRunner

    # Build the command through the real class with a minimal config object.
    class _Config:
        workflow_dir = WORKFLOW
        snakefile = WORKFLOW / "Snakefile"
        cores = 4
        snakemake_cmd = ("snakemake",)
        extra_args = ("--nolock",)
        log_dir = tmp_path

    runner = SnakemakeRunner(_Config(), {"resource_profile": {"run_chease": {"mem_mb": 4000}}})
    command = runner.command(RunPlan(run_id=1, full_shots=(39915,)), tmp_path / "c.yaml")
    assert command[-3:] == ["--set-resources", "run_chease:mem_mb=4000", "--nolock"]


def test_a_killed_job_whose_id_an_appended_run_reuses_ends_open():
    log = ("[Tue Nov  3 12:00:00 2026]\nrule a:\n    wildcards: shot=1\n    threads: 2\n    jobid: 5\n"
           "[Tue Nov  3 13:00:00 2026]\nrule b:\n    jobid: 5\n[Tue Nov  3 13:00:10 2026]\nFinished job 5.\n")
    jobs = parse_log(log)
    assert [(j.rule, j.end is None, j.wildcards, j.threads) for j in jobs] == [
        ("a", True, "shot=1", 2), ("b", False, "", 1)]


def test_an_error_block_with_a_long_message_still_closes_its_job():
    log = ("[Tue Nov  3 12:00:00 2026]\nrule a:\n    jobid: 3\n[Tue Nov  3 12:00:05 2026]\nError in rule a:\n"
           "    message: one\n      two\n      three\n      four\n      five\n      six\n      seven\n    jobid: 3\n")
    (job,) = parse_log(log)
    assert job.failed and job.seconds == 5.0


def test_a_workflow_error_is_per_shot_only_for_the_keep_going_summary():
    tail = "Error in rule a:\n    jobid: 1\nExiting because a job execution failed.\n"
    for summary in ("WorkflowError: At least one job did not complete successfully.\n",
                    "WorkflowError:\n\nAt least one job did not complete successfully.\n"):
        assert chunk_verdict(1, tail + summary).done is True
    assert chunk_verdict(1, tail + "WorkflowError:\n\nlater text\n").reason == "WorkflowError: later text"


def test_plan_splits_every_lane_and_refuses_a_replan(tmp_path):
    (tmp_path / "shots.txt").write_text("1\n2\n3\n")
    args = ["plan", str(tmp_path / "shots.txt"), str(tmp_path / "run"), "--lanes", "2", "--chunk", "2", "--pin", "abc"]
    assert campaign_main(args) == 0
    lane0 = LaneState(tmp_path / "run" / "lane0")
    assert lane0.chunks() == ["shots.000"] and lane0.shots() == [1, 3]
    assert (lane0.path / "PIN_SHA").read_text().strip() == "abc"
    assert campaign_main(args) == 2


def test_the_library_names_no_pipeline_rule_product_or_site():
    """Pipeline knowledge lives in the workflow's paths.py; site knowledge outside the repository."""
    import re

    package = Path(__file__).resolve().parents[1] / "vaft" / "database" / "campaign"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for name in (*SPEC.run_scoped_rules, *SPEC.solver_processes, *SPEC.compared_products, "srvns", "regen2147"):
        assert not re.search(rf"\b{re.escape(name)}\b", text), name


# -- the lane runner on a toy workflow ---------------------------------------------

_TOY_PATHS = """
from pathlib import Path

CAMPAIGN_RUN_SCOPED_RULES = frozenset({"select"})
CAMPAIGN_RUN_SCOPED_OUTPUTS = ("selection",)
CAMPAIGN_REQUIRED_OUTPUTS = ("selection",)
CAMPAIGN_SHOT_INPUTS = ("raw",)
CAMPAIGN_COMPARED_PRODUCTS = ("product",)


class PipelinePaths:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    @classmethod
    def from_config(cls, config):
        return cls(config["base_dir"])

    def raw(self, shot):
        return f"{self.base_dir}/raw/{shot}.json"

    def product(self, shot):
        return f"{self.base_dir}/product/{shot}.json"

    def selection(self):
        return f"{self.base_dir}/selection.json"
"""

_TOY_SNAKEFILE = """
import json
SHOTS = [int(s) for s in config["shots"]]
BASE = config["base_dir"]

rule all:
    input: expand(BASE + "/product/{shot}.json", shot=SHOTS), BASE + "/selection.json"

rule select:
    output: BASE + "/selection.json"
    run:
        Path = __import__("pathlib").Path
        Path(output[0]).write_text(json.dumps(SHOTS))

rule product:
    input: BASE + "/raw/{shot}.json", BASE + "/selection.json"
    output: BASE + "/product/{shot}.json"
    run:
        if int(wildcards.shot) == config.get("bad_shot"):
            raise ValueError("this shot fails")
        open(output[0], "w").write(open(input[0]).read())
"""


def _toy(tmp_path, shots, *, bad_shot=None, raw=None):
    workflow = tmp_path / "workflow"
    workflow.mkdir()
    (workflow / "Snakefile").write_text(_TOY_SNAKEFILE)
    (workflow / "paths.py").write_text(_TOY_PATHS)
    config = tmp_path / "run.yaml"
    config.write_text(json.dumps({"bad_shot": bad_shot}))
    lane = tmp_path / "lane0"
    lane.mkdir()
    (lane / "shots.txt").write_text("".join(f"{s}\n" for s in shots))
    for shot in shots if raw is None else raw:
        target = lane / "filedb" / "raw" / f"{shot}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"shot": shot}))
    # Snakemake's source cache goes under HOME; keep it inside the test.
    env = {"HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "cache")}
    return lane, LaneOptions(workflow_dir=workflow, config=config, cores=2, chunk_size=2, env=env)


def _records(lane):
    return [json.loads(line) for line in (lane / "lane.jsonl").read_text().splitlines()]


@pytest.mark.skipif(subprocess.run([sys.executable, "-c", "import snakemake"], capture_output=True).returncode != 0,
                    reason="needs snakemake")
def test_a_lane_runs_its_chunks_here_resumes_and_measures(tmp_path):
    lane, options = _toy(tmp_path, [1, 2, 3])
    assert run_lane(lane, options) == 0
    state = LaneState(lane)
    assert state.done() == ["shots.000", "shots.001"]
    records = _records(lane)
    assert [r["ran"] for r in records] == [2, 1] and all(r["done"] for r in records)
    usage = records[0]["usage"]
    assert usage["source"] in {"rusage", "cgroup"} and usage["cpu_s"] > 0 and usage["booked_cpus"] == 2
    assert (lane / "chunks" / "shots.001.selection.json").read_text() == "[3]"  # cleared per chunk
    assert (lane / "filedb" / "product" / "3.json").exists()
    # Resume: nothing left, nothing rerun.
    assert run_lane(lane, options) == 0 and len(_records(lane)) == 2
    report = lane_report(lane)["totals"]
    assert report["shots"] == 3 and report["cpu_util"] is not None


@pytest.mark.skipif(subprocess.run([sys.executable, "-c", "import snakemake"], capture_output=True).returncode != 0,
                    reason="needs snakemake")
def test_a_failed_shot_finishes_the_chunk_and_a_missing_input_keeps_it_open(tmp_path):
    lane, options = _toy(tmp_path, [1, 2], bad_shot=2, raw=[2])
    assert run_lane(lane, options) == 0
    (record,) = _records(lane)
    assert record["done"] and record["failed_rules"] == ["product"] and record["missing_inputs"] == 1
    assert LaneState(lane).done() == []  # shot 1 had no raw: the chunk stays open
    assert (lane / "missing_raw.txt").read_text() == "shots.000 1\n"


def test_a_lane_refuses_a_different_pin_or_chunk_size(tmp_path):
    lane, options = _toy(tmp_path, [1])
    LaneState(lane).ensure("PIN_SHA", "aaa")
    options.pin = "bbb"
    assert run_lane(lane, options) == 2


def test_launch_runs_every_pending_lane_through_a_backend(tmp_path, monkeypatch):
    from vaft.code.execution import LocalBackend
    from vaft.database.campaign.launch import launch_lanes, lane_request

    run = tmp_path / "run"
    for index in range(2):
        LaneState(run / f"lane{index}").ensure("chunk_size", "1")
        split_chunks([index], 1, run / f"lane{index}" / "chunks")
    (run / "lane1" / "done_chunks.txt").write_text("shots.000\n")  # finished: not launched
    request = lane_request(run / "lane0", ["--workflow-dir", "w"], cores=8, memory_mb=1000)
    assert request.command[3:] == ["run-lane", str((run / "lane0").resolve()), "--cores", "8", "--workflow-dir", "w"]
    assert request.resources.threads_per_task == 8 and request.env["OMP_NUM_THREADS"] == "1"
    calls = []
    monkeypatch.setattr("vaft.database.campaign.launch.submit_request",
                        lambda backend, request: calls.append(request) or type("H", (), {"job_id": None})())
    handles = launch_lanes(run, LocalBackend(), ["--workflow-dir", "w"], cores=8)
    assert list(handles) == ["lane0"] and calls[0].workdir == (run / "lane0").resolve()


@pytest.mark.skipif(subprocess.run([sys.executable, "-c", "import snakemake"], capture_output=True).returncode != 0,
                    reason="needs snakemake")
def test_launch_with_the_local_backend_runs_every_lane_end_to_end(tmp_path, monkeypatch):
    lane, options = _toy(tmp_path, [1, 2, 3])
    for key, value in options.env.items():
        monkeypatch.setenv(key, value)
    run = tmp_path / "run"
    (tmp_path / "shots.txt").write_text("1\n2\n3\n")
    assert campaign_main(["plan", str(tmp_path / "shots.txt"), str(run), "--lanes", "2", "--chunk", "1"]) == 0
    for index, shots in ((0, [1, 3]), (1, [2])):
        for shot in shots:
            raw = run / f"lane{index}" / "filedb" / "raw" / f"{shot}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("{}")
    assert campaign_main(["launch", str(run), "--backend", "local", "--cores", "1",
                          "--workflow-dir", str(options.workflow_dir), "--config", str(options.config)]) == 0
    assert LaneState(run / "lane0").done() == ["shots.000", "shots.001"]
    assert LaneState(run / "lane1").done() == ["shots.000"]
    # Finished lanes are not launched again.
    assert campaign_main(["launch", str(run), "--backend", "local", "--cores", "1",
                          "--workflow-dir", str(options.workflow_dir), "--config", str(options.config)]) == 0
    assert len((run / "lane0" / "lane.jsonl").read_text().splitlines()) == 2


_SITE = Path(__file__).resolve().parents[1] / "workflow" / "cluster_regeneration"


@pytest.mark.skipif(subprocess.run(["cp", "--version"], capture_output=True).returncode != 0
                    or shutil.which("bash") is None, reason="needs GNU cp (-l)")
def test_the_site_raw_hook_links_complete_dumps_only(tmp_path):
    root = tmp_path / "root"
    for shot, files in ((1, ("raw.json.gz", "manifest.json")), (2, ("raw.json.gz",))):
        for kind in files:
            name = f"vest_{shot}_daq_{kind}"
            (root / "raw" / str(shot)).mkdir(parents=True, exist_ok=True)
            (root / "raw" / str(shot) / name).write_text(kind)
    (tmp_path / "chunk").write_text("1\n2\n3\n")
    filedb = tmp_path / "filedb"
    result = subprocess.run(["bash", str(_SITE / "stage_raw.sh"), "lane", "shots.000", str(tmp_path / "chunk"),
                             str(filedb)], env={"PATH": "/usr/bin:/bin", "REGEN_ROOT": str(root)})
    assert result.returncode == 0
    assert sorted(p.name for p in (filedb / "raw").iterdir()) == ["1"]  # 2 lacks its manifest, 3 its dump


def test_no_site_specific_name_is_committed_in_the_regeneration_tooling():
    import re

    text = "\n".join(path.read_text(encoding="utf-8") for path in _SITE.iterdir() if path.is_file())
    for pattern in (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"\w+@\w[\w.-]*\.\w+", r"-p \d{3,5}\b"):
        found = [match for match in re.findall(pattern, text) if not match.startswith("127.")]
        assert not found, (pattern, found)


def test_the_runner_reads_paths_under_the_workflows_own_defaults(monkeypatch):
    monkeypatch.delenv("VAFT_FILEDB_DIR", raising=False)
    merged = SPEC.merged_config({"base_dir": "/lane/filedb", "efit": {"table_dir": "/t/"}})
    assert merged["layout"] == "filedb" and merged["base_dir"] == "/lane/filedb"
    assert merged["efit"]["table_dir"] == "/t/" and len(merged["efit"]) > 1  # deep, not replaced
    assert SPEC.paths(merged).raw_dump(1).startswith("/lane/filedb/raw/")


def test_the_guard_watches_prefixed_names_like_the_efit_loader():
    from vaft.database.campaign.run import _RssGuard

    guard = _RssGuard.__new__(_RssGuard)
    guard.names = frozenset(SPEC.solver_processes)
    assert guard._watched("ld-linux-x86-64") and guard._watched("rdcon") and not guard._watched("python3")


def test_a_site_lane_command_gets_the_lanes_cores_and_memory(tmp_path):
    from vaft.database.campaign.launch import lane_request

    request = lane_request(tmp_path, [], cores=12, memory_mb=4096, lane_command=["bash", "run_lane.sh"])
    assert request.command == ["bash", "run_lane.sh", str(tmp_path.resolve())]
    assert (request.env["CORES"], request.env["MEM_MB"]) == ("12", "4096")


def test_the_usage_meter_reports_a_child_peak_only_for_its_own_interval():
    from vaft.code.accounting import UsageMeter

    # Above this test process's own size: on Linux a forked child counts its parent's pages.
    big = "b = bytearray(512 * 1024 * 1024)\nb[::4096] = b'x' * len(b[::4096])"
    subprocess.run([sys.executable, "-c", big], check=True)
    meter = UsageMeter(booked_cpus=1, cgroup=None).start()
    subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert meter.stop().max_process_rss_mb is None  # the earlier, larger child is not this interval's
