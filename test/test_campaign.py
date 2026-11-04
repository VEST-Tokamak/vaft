"""Campaign orchestration (vaft.database.campaign): lanes, chunk verdicts, Snakemake
log costs, the benchmark report and product equivalence.

The #2147 regeneration ran these steps as shell in workflow/cluster_regeneration;
the tests pin that the Python versions keep its on-disk layout (a run started by
the shell resumes here) and its verdicts.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vaft.database.campaign import (
    LaneState,
    chunk_verdict,
    compare_products,
    disjoint,
    lane_report,
    occupancy,
    parse_log,
    plan_lanes,
    rule_costs,
    split_chunks,
    summarize,
)
from vaft.database.campaign.__main__ import main as campaign_main

WORKFLOW = Path(__file__).resolve().parents[1] / "workflow" / "automatic_pipeline_1_routine_data_processing"


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


# -- chunk verdict (the run_lane.sh rule) -----------------------------------

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
        (1, _PER_SHOT, True, True, ""),
        (1, _PER_SHOT, False, False, "no preflight verdict"),
        (1, "Killed\n", True, False, "without a failed rule"),
        (1, "Error in rule a:\n", True, False, "did not finish its keep-going"),
        (1, _PER_SHOT + "LockException\n", True, False, "LockException"),
        (1, _PER_SHOT + "Traceback (most recent call last):\n", True, False, "traceback"),
        (1, _PER_SHOT.replace("generate_eddy_ods", "preflight_raw_dumps"), True, False, "run-scoped"),
        (1, _PER_SHOT + "WorkflowError:\nSomething else broke\n", True, False, "Something else broke"),
    ],
)
def test_a_chunk_verdict_separates_per_shot_failures_from_failed_runs(returncode, log, eligible, done, reason):
    verdict = chunk_verdict(returncode, log, has_eligible_verdict=eligible)
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

def test_a_lane_report_combines_the_cgroup_line_and_the_snakemake_log(tmp_path):
    lane = tmp_path / "lane0"
    (lane / "logs").mkdir(parents=True)
    (lane / "lane.log").write_text(
        "2026-11-03 11:59:00 lane start config=x sha=y cores=4 chunk=2 host=n001 job=1\n"
        "2026-11-03 12:00:00 shots.000 start (2 shots)\n"
        "2026-11-03 12:02:00 shots.000 exit=1 wall=120s cpu_s=240 cpu_util=0.500 "
        "chunk_mem_max_gb=3.5 job_mem_peak_gb=3.5 log=shots.000.attempt-1.log\n"
    )
    (lane / "logs" / "shots.000.snakemake.log").write_text(_LOG)
    report = lane_report(lane)
    totals = report["totals"]
    assert totals["shots"] == 2 and totals["cpu_util"] == pytest.approx(0.5)
    assert totals["booked_cpu_h_per_shot"] == pytest.approx(120 * 4 / 3600 / 2)
    assert totals["shots_per_hour"] == pytest.approx(60.0)
    assert totals["mem_max_gb"] == 3.5
    assert totals["checkpoint_share"] == pytest.approx(60 / 120)
    assert totals["plot_share"] == pytest.approx(30 / (240 + 60 + 30))
    assert report["rules"]["plot_raw"]["jobs"] == 1


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
    report = compare_products(new, ref, [1, 2, 3], workflow_dir=WORKFLOW)
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
    report = compare_products(new, lane0, [7], workflow_dir=WORKFLOW, ref_roots=[lane0, lane1])
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


def test_the_verdict_reads_workflow_error_like_grep_a1():
    tail = "Error in rule a:\n    jobid: 1\nExiting because a job execution failed.\n"
    assert chunk_verdict(1, tail + "WorkflowError: At least one job did not complete successfully.\n",
                         has_eligible_verdict=True).done is False
    assert chunk_verdict(1, tail + "WorkflowError:\n\nlater text\n", has_eligible_verdict=True).done is False


def test_plan_writes_what_launch_lanes_reads_and_refuses_a_replan(tmp_path):
    (tmp_path / "shots.txt").write_text("1\n2\n3\n")
    args = ["plan", str(tmp_path / "shots.txt"), str(tmp_path / "run"), "--lanes", "2", "--chunk", "2", "--pin", "abc"]
    assert campaign_main(args) == 0
    assert (tmp_path / "run" / "shots_all.txt").read_text() == "1\n2\n3\n"
    assert (tmp_path / "run" / "PIN_SHA").read_text().strip() == "abc"
    assert campaign_main(args) == 2
    assert (tmp_path / "run" / "lane0" / "shots.txt").read_text() == "1\n3\n"
