"""Resource accounting from the job cgroup, scheduler outcome classes, and the
execution policy (vaft.code.accounting, vaft.code.slurm._job_outcome,
vaft.code.policy).

The cluster-a regeneration was cancelled by the cluster administrator for using a
third of what it booked; sacct's TotalCPU of every cancelled job read 0. These
pin the record that survives a cancel, what a cancel is called, and the gate a
shared cluster's agreement is enforced by.
"""

from __future__ import annotations

import json
import shutil
import sys
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from vaft.code.accounting import (
    ResourceUsage,
    cgroup_cpu_seconds,
    cgroup_memory_peak_mb,
    job_cgroup_dir,
    parse_usage_record,
    usage_record_lines,
)
from vaft.code.policy import ExecutionPolicy, SubmissionRefused, load_policy, open_gate
from vaft.code.slurm import _job_outcome


def _cgroup(root, usage_us=1_000_000, user_us=800_000, system_us=200_000, peak=2**30):
    job = root / "system.slice" / "slurmstepd.scope" / "job_42"
    (job / "step_batch" / "user" / "task_0").mkdir(parents=True)
    (job / "cpu.stat").write_text(f"usage_usec {usage_us}\nuser_usec {user_us}\nsystem_usec {system_us}\n")
    (job / "memory.peak").write_text(f"{peak}\n")
    return job


def test_the_job_cgroup_is_found_above_the_task_cgroup(tmp_path):
    job = _cgroup(tmp_path)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/system.slice/slurmstepd.scope/job_42/step_batch/user/task_0\n")
    assert job_cgroup_dir(proc, tmp_path) == job
    proc.write_text("0::/user.slice/user-1000.slice/session-3.scope\n")
    assert job_cgroup_dir(proc, tmp_path) is None


def test_cgroup_cpu_and_memory_are_read_in_seconds_and_mib(tmp_path):
    job = _cgroup(tmp_path)
    assert cgroup_cpu_seconds(job) == (1.0, 0.8, 0.2)
    assert cgroup_memory_peak_mb(job) == 1024.0
    assert cgroup_cpu_seconds(tmp_path) is None and cgroup_memory_peak_mb(tmp_path) is None


def test_utilization_is_cpu_seconds_over_wall_times_booked_cpus():
    usage = ResourceUsage(wall_s=100.0, cpu_s=560.0, booked_cpus=16, source="cgroup")
    assert usage.utilization == pytest.approx(0.35)
    assert usage.as_dict()["utilization"] == pytest.approx(0.35)
    assert ResourceUsage(wall_s=0.0, cpu_s=1.0, booked_cpus=16, source="cgroup").utilization is None
    assert ResourceUsage(wall_s=10.0, cpu_s=None, booked_cpus=16, source="cgroup").utilization is None


def test_a_usage_record_is_the_difference_of_its_first_and_last_samples():
    text = "100 1000000 800000 200000 1073741824\n160 33000000 30000000 3000000 2147483648\n"
    usage = parse_usage_record(text, booked_cpus=16)
    assert (usage.wall_s, usage.cpu_s, usage.user_s, usage.system_s) == (60.0, 32.0, 29.2, 2.8)
    assert usage.peak_mem_mb == 2048.0 and usage.source == "cgroup"
    assert usage.utilization == pytest.approx(32.0 / (60 * 16))
    assert parse_usage_record(text.splitlines()[0], 16) is None
    assert parse_usage_record(None, 16) is None
    no_peak = parse_usage_record("100 1000000 - - -\n110 2000000 - - -\n", 1)
    assert (no_peak.cpu_s, no_peak.user_s, no_peak.peak_mem_mb) == (1.0, None, None)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_job_script_snippet_samples_the_job_cgroup(tmp_path):
    job = _cgroup(tmp_path)
    record = tmp_path / "usage"
    script = "\n".join(usage_record_lines(str(record))) + (
        f"\nvaft_job_cg={job}\nvaft_usage_sample\n"
        f"printf 'usage_usec 9000000\\nuser_usec 8000000\\nsystem_usec 1000000\\n' > {job}/cpu.stat\n"
        "vaft_usage_sample\n"
    )
    subprocess.run(["bash", "-c", script], check=True)
    usage = parse_usage_record(record.read_text(), booked_cpus=2)
    assert usage.cpu_s == 8.0 and usage.peak_mem_mb == 1024.0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_snippet_writes_nothing_outside_a_job_cgroup(tmp_path):
    record = tmp_path / "usage"
    script = "\n".join(usage_record_lines(str(record))) + "\nvaft_job_cg=\nvaft_usage_sample\n"
    subprocess.run(["bash", "-c", script], check=True)
    assert not record.exists()


# -- what a scheduler outcome is called --------------------------------------

def _outcome(**kw):
    base = dict(cancelled=False, recorded=None, state=None, exit_code=None, began=True, walltime_kill=False)
    base.update(kw)
    return _job_outcome(**base)


@pytest.mark.parametrize(
    ("state", "status"),
    [("OUT_OF_MEMORY", "memory_limit"), ("CANCELLED", "cancelled"), ("NODE_FAIL", "node_fail"),
     ("PREEMPTED", "node_fail"), ("BOOT_FAIL", "node_fail")],
)
def test_a_job_the_scheduler_ended_carries_its_reason(state, status):
    returncode, timed_out, never_started, _, runtime_status = _outcome(
        state=state, exit_code=137, recorded="terminated 100 "
    )
    assert (returncode, timed_out, never_started, runtime_status) == (None, True, False, status)


def test_a_program_that_exited_before_the_cancel_keeps_its_exit():
    returncode, timed_out, _, _, status = _outcome(state="CANCELLED", recorded="0")
    assert (returncode, timed_out, status) == (0, False, "completed")


def test_the_backends_own_max_wait_cancel_is_a_timeout_or_a_queue_timeout():
    assert _outcome(cancelled=True, state="CANCELLED", recorded="terminated 5 ")[4] == "timeout"
    assert _outcome(cancelled=True, state="CANCELLED", began=False)[4] == "queue_timeout"


def test_a_time_limit_and_a_clean_exit_are_unchanged():
    assert _outcome(state="TIMEOUT", recorded="terminated 5 ")[4] == "timeout"
    assert _outcome(state="COMPLETED", recorded="0")[:2] == (0, False)
    assert _outcome(state="FAILED", recorded="3")[4] == "completed"


# -- execution policy --------------------------------------------------------

def _write_policy(tmp_path, clusters, default=None, approvals=()):
    import yaml

    (tmp_path / "approvals").mkdir(exist_ok=True)
    for approval in approvals:
        (tmp_path / "approvals" / f"{approval['id']}.json").write_text(json.dumps(approval))
    data = {"approvals_dir": "approvals", "audit_log": "audit.jsonl", "clusters": clusters}
    if default is not None:
        data["default"] = default
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(data))
    return ExecutionPolicy.load(path)


def test_no_policy_file_means_no_policy(tmp_path):
    assert load_policy({}) is None
    gate = open_gate(cluster="cluster-a", cpus=128, mem_mb=None, partition=None, environ={})
    assert gate.policy is None
    gate.submitted("1", ["sbatch"])  # a no-op


def test_a_named_but_missing_policy_file_raises():
    with pytest.raises(FileNotFoundError):
        load_policy({"VAFT_EXECUTION_POLICY": "/nonexistent/policy.yaml"})


def test_unknown_policy_keys_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown policy keys"):
        _write_policy(tmp_path, {"cluster-a": {"requires_aproval": True}})


def test_approval_is_checked_for_cluster_and_expiry(tmp_path):
    now = datetime(2026, 11, 4, tzinfo=timezone.utc)
    policy = _write_policy(
        tmp_path,
        {"cluster-a": {"requires_approval": True}},
        approvals=[
            {"id": "ok", "cluster": "cluster-a", "approver": "admin", "expires": (now + timedelta(days=1)).isoformat()},
            {"id": "old", "cluster": "cluster-a", "approver": "admin", "expires": (now - timedelta(days=1)).isoformat()},
            {"id": "other", "cluster": "vest", "approver": "admin"},
        ],
    )
    check = lambda approval: policy.check(cluster="cluster-a", cpus=1, mem_mb=1, partition=None, approval_id=approval, now=now)
    assert check("ok").allowed
    assert "expired" in check("old").reasons[0]
    assert "is for cluster 'vest'" in check("other").reasons[0]
    assert "none given" in check("").reasons[0]
    assert "not found" in check("missing").reasons[0]
    assert not policy.check(cluster="cluster-a", cpus=1, mem_mb=1, partition=None, approval_id="../ok", now=now).allowed


def test_the_tighter_of_cluster_and_approval_limits_applies(tmp_path):
    policy = _write_policy(
        tmp_path,
        {"cluster-a": {"requires_approval": True, "max_cpus": 32, "max_mem_mb": 262144}},
        approvals=[{"id": "a", "cluster": "cluster-a", "approver": "admin", "max_cpus": 16, "max_mem_mb": 65536}],
    )
    decision = policy.check(cluster="cluster-a", cpus=32, mem_mb=65536, partition=None, approval_id="a")
    assert not decision.allowed and decision.limits["max_cpus"] == 16
    assert policy.check(cluster="cluster-a", cpus=16, mem_mb=65536, partition=None, approval_id="a").allowed


def test_max_concurrent_counts_this_logs_jobs_that_are_still_live(tmp_path):
    policy = _write_policy(tmp_path, {"cluster-a": {"max_concurrent": 2}})
    decision = policy.check(cluster="cluster-a", cpus=1, mem_mb=None, partition=None)
    for job in ("10", "11", "12"):
        policy.record("submitted", decision, argv=["sbatch"], job_id=job)
    assert policy.recorded_jobs("cluster-a") == ["10", "11", "12"]
    live = lambda ids: len({"11", "12"} & set(ids))
    assert "2 jobs already active" in policy.check(
        cluster="cluster-a", cpus=1, mem_mb=None, partition=None, active_jobs=live
    ).reasons[0]
    assert policy.check(cluster="cluster-a", cpus=1, mem_mb=None, partition=None, active_jobs=lambda ids: 1).allowed


def test_the_default_entry_covers_clusters_the_file_does_not_name(tmp_path):
    policy = _write_policy(tmp_path, {}, default={"requires_approval": True})
    assert not policy.check(cluster="somewhere", cpus=1, mem_mb=None, partition=None).allowed


def test_a_refusal_raises_and_is_audited(tmp_path):
    _write_policy(tmp_path, {"cluster-a": {"requires_approval": True}})
    env = {"VAFT_EXECUTION_POLICY": str(tmp_path / "policy.yaml")}
    with pytest.raises(SubmissionRefused, match="requires a recorded approval"):
        open_gate(cluster="cluster-a", cpus=16, mem_mb=65536, partition="g1", options=["--cpus-per-task=16"], environ=env)
    (entry,) = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert entry["event"] == "refused" and entry["argv"] == ["--cpus-per-task=16"] and entry["requester"]


def test_a_cluster_is_found_by_any_of_its_hosts(tmp_path):
    policy = _write_policy(tmp_path, {"cluster-a": {"requires_approval": True, "hosts": ["login.cluster-a.example"]}})
    decision = policy.check(cluster="login.cluster-a.example", cpus=1, mem_mb=1, partition=None)
    assert not decision.allowed and decision.cluster == "cluster-a"


def test_later_scheduler_options_are_what_is_checked(tmp_path):
    # sbatch takes the last value: an extra_args --cpus-per-task/--mem/--partition must not slip past.
    policy = _write_policy(tmp_path, {"cluster-a": {"max_cpus": 16, "max_mem_mb": 65536, "partitions": ["g1"]}})
    check = lambda options: policy.check(cluster="cluster-a", cpus=16, mem_mb=65536, partition="g1", options=options)
    assert check(["--cpus-per-task=16", "--mem=65536M", "--partition=g1"]).allowed
    assert "64 CPUs requested" in check(["--cpus-per-task=16", "-c", "64"]).reasons[0]
    assert "limit 65536 MiB" in check(["--mem=65536M", "--mem=220G"]).reasons[0]
    assert "partition 'gpu'" in check(["--partition=g1", "-p", "gpu"]).reasons[0]
    assert "limit 65536 MiB" in check(["--mem-per-cpu=8G", "--cpus-per-task=16"]).reasons[0]


def test_an_approval_binds_its_requester_and_its_own_file_name(tmp_path, monkeypatch):
    policy = _write_policy(
        tmp_path, {"cluster-a": {"requires_approval": True}},
        approvals=[{"id": "mine", "cluster": "cluster-a", "approver": "admin", "requester": "someone-else"}],
    )
    assert "was given to 'someone-else'" in policy.check(
        cluster="cluster-a", cpus=1, mem_mb=1, partition=None, approval_id="mine"
    ).reasons[0]
    (tmp_path / "approvals" / "copy.json").write_text((tmp_path / "approvals" / "mine.json").read_text())
    assert policy.approval("copy") is None


def test_an_unknown_active_count_refuses_under_max_concurrent(tmp_path):
    policy = _write_policy(tmp_path, {"cluster-a": {"max_concurrent": 2}})
    decision = policy.check(cluster="cluster-a", cpus=1, mem_mb=None, partition=None, active_jobs=lambda ids: None)
    assert "cannot count active jobs" in decision.reasons[0]


def test_a_failed_audit_write_after_sbatch_warns_instead_of_raising(tmp_path, monkeypatch):
    _write_policy(tmp_path, {"cluster-a": {}})
    env = {"VAFT_EXECUTION_POLICY": str(tmp_path / "policy.yaml")}
    gate = open_gate(cluster="cluster-a", cpus=1, mem_mb=None, partition=None, environ=env)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(gate.policy), "record", broken)
    with pytest.warns(RuntimeWarning, match="could not audit job 77"):
        gate.submitted("77", ["sbatch"])


def test_a_naive_expiry_is_local_time():
    from vaft.code.policy import _parse_time

    assert _parse_time("2026-11-04T12:00:00").utcoffset() == datetime.now().astimezone().utcoffset()


def test_an_oom_kill_the_batch_shell_survived_is_still_the_memory_limit():
    returncode, timed_out, _, _, status = _outcome(state="OUT_OF_MEMORY", recorded="137")
    assert (returncode, timed_out, status) == (None, True, "memory_limit")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["-c64"], "64 CPUs requested"),
        (["-pgpu"], "partition 'gpu'"),
        (["--exclusive"], "--exclusive books a whole node"),
        (["--mem=0"], "whole memory (--mem=0"),
        (["--nodes=2", "--ntasks-per-node=2", "-c", "8"], "32 CPUs requested"),
        (["--nodes=2", "--mem=48G"], "limit 65536 MiB"),
    ],
)
def test_whole_node_and_joined_options_cannot_slip_past(tmp_path, options, message):
    policy = _write_policy(tmp_path, {"cluster-a": {"max_cpus": 16, "max_mem_mb": 65536, "partitions": ["g1"]}})
    decision = policy.check(cluster="cluster-a", cpus=16, mem_mb=65536, partition="g1", options=options)
    assert any(message in reason for reason in decision.reasons), decision.reasons


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(pid, 0) terminates on Windows")
def test_a_step_this_host_launched_counts_while_its_client_lives():
    import os
    import socket
    import time

    from vaft.code.slurm import SlurmBackend, _step_record

    host, now = socket.gethostname(), int(time.time())
    backend = SlurmBackend(mode="step")
    records = [
        _step_record(os.getpid()),  # this process: alive and started before the record
        f"srun:{host}:999999999:{now}",  # gone
        f"srun:other-host:{os.getpid()}:{now}",  # another host's pid means nothing here
        f"srun:{host}:{os.getpid()}:1000",  # recorded before this process started: a recycled pid
    ]
    assert backend._active_jobs(records) == 1


def test_an_sbatch_without_a_parseable_id_counts_for_an_hour():
    import time

    from vaft.code.slurm import UNPARSED_ACTIVE_S, SlurmBackend

    now = int(time.time())
    backend = SlurmBackend(mode="batch")
    fresh, stale = f"unparsed:{now - 60}", f"unparsed:{now - int(UNPARSED_ACTIVE_S) - 60}"
    assert backend._active_jobs([fresh, stale]) == 1


def test_exclusive_with_memory_per_cpu_books_the_whole_node(tmp_path):
    policy = _write_policy(tmp_path, {"cluster-a": {"max_mem_mb": 65536}})
    decision = policy.check(
        cluster="cluster-a", cpus=1, mem_mb=None, partition=None, options=["--exclusive", "--mem-per-cpu=1G"]
    )
    assert any("whole memory" in reason or "whole node" in reason for reason in decision.reasons), decision.reasons
