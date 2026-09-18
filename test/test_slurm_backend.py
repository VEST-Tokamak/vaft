"""SlurmBackend against fake Slurm commands (#1017).

No cluster is needed: ``sbatch``, ``squeue``, ``sacct``, ``scancel`` and
``srun`` are small Python programs put first on ``PATH``. They behave the way
the real ones do where the backend depends on it:

* ``sbatch`` prints the job id at once and runs the job script asynchronously
  (``run_job.py``), resolving ``--output``/``--error`` against ``--chdir``, so
  ``squeue`` sees it ``RUNNING`` first. ``FAKE_SLURM_OUTCOME`` changes what
  happens: ``TERM_AFTER=<s>`` sends the script ``TERM`` after that many seconds
  (a walltime kill); a terminal state such as ``NODE_FAIL`` or ``PENDING``
  never runs it.
* ``squeue -j`` keeps listing a finished job with its final state, as a real
  controller does until ``MinJobAge``; ``FAKE_SQUEUE_FLAKY=<n>`` makes the first
  ``n`` polls fail with a socket timeout.
* ``sacct`` reports the recorded state unless ``FAKE_SLURM_NO_SACCT`` is set;
  ``FAKE_SACCT_LAG=<n>`` makes its first ``n`` answers a stale ``RUNNING``.

Every invocation is appended to ``calls.jsonl`` so tests can assert on options.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from vaft.compat import IS_WINDOWS
from vaft.code import ExecutableNotLaunchable, ExecutionRequest, ResourceRequest
from vaft.code.execution import BACKEND_ENV, LocalBackend, default_backend, resolve_backend
from vaft.code.slurm import SCRATCH_DIRECTORY, SlurmBackend, exported_environment, walltime

pytestmark = pytest.mark.skipif(IS_WINDOWS, reason="Slurm and its job scripts are POSIX")

_COMMON = textwrap.dedent(
    """
    import json, os, signal, subprocess, sys, time
    from pathlib import Path
    state = Path(os.environ["FAKE_SLURM_DIR"])
    with (state / "calls.jsonl").open("a") as log:
        log.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
    def record(job, text):
        (state / f"{job}.state").write_text(text)
    """
)

_FAKES = {
    "sbatch": """
        options = dict(a.split("=", 1) for a in sys.argv[1:-1] if a.startswith("--") and "=" in a)
        if os.environ.get("FAKE_SLURM_REJECT"):
            print("sbatch: error: invalid partition specified", file=sys.stderr)
            sys.exit(1)
        if os.environ.get("FAKE_SLURM_BANNER_ONLY"):
            print("Welcome to the cluster")
            sys.exit(0)
        (state / "sbatch_env.json").write_text(json.dumps(dict(os.environ)))
        counter = state / "next_id"
        job = int(counter.read_text()) if counter.exists() else 1000
        counter.write_text(str(job + 1))
        outcome = os.environ.get("FAKE_SLURM_OUTCOME", "")
        if outcome in ("NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "CANCELLED"):
            record(job, outcome + "|0:9")
        elif outcome == "PENDING":
            record(job, "PENDING|0:0")
        else:
            record(job, "RUNNING|0:0")
            # Run the job asynchronously, as a controller does.
            subprocess.Popen([sys.executable, str(state / "run_job.py"), str(job), json.dumps(options),
                              sys.argv[-1], outcome], start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.environ.get("FAKE_SLURM_BANNER"):
            print("Submitted to the cluster; have a nice day")
        cluster = os.environ.get("FAKE_SLURM_CLUSTER")
        print(f"{job};{cluster}" if cluster else str(job))
    """,
    "squeue": """
        flaky = state / "flaky"
        if os.environ.get("FAKE_SQUEUE_FLAKY"):
            count = int(flaky.read_text()) if flaky.exists() else 0
            if count < int(os.environ["FAKE_SQUEUE_FLAKY"]):
                flaky.write_text(str(count + 1))
                print("slurm_load_jobs error: Socket timed out on send/recv operation", file=sys.stderr)
                sys.exit(1)
        job = sys.argv[sys.argv.index("-j") + 1]
        path = state / f"{job}.state"
        if not path.exists():
            print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
            sys.exit(1)
        print(path.read_text().split("|")[0])
    """,
    "sacct": """
        if os.environ.get("FAKE_SLURM_NO_SACCT"):
            print("sacct: error: Slurm accounting storage is disabled", file=sys.stderr)
            sys.exit(1)
        lag = state / "sacct_lag"
        if os.environ.get("FAKE_SACCT_LAG"):
            count = int(lag.read_text()) if lag.exists() else 0
            if count < int(os.environ["FAKE_SACCT_LAG"]):
                lag.write_text(str(count + 1))
                print("RUNNING|0:0")  # slurmdbd has not caught up yet
                sys.exit(0)
        job = sys.argv[sys.argv.index("-j") + 1]
        print((state / f"{job}.state").read_text())
    """,
    "scancel": """
        record(sys.argv[-1], "CANCELLED by 501|0:15")
    """,
    "srun": """
        (state / "srun_env.json").write_text(json.dumps(dict(os.environ)))
        command = sys.argv[sys.argv.index("--") + 1:]
        os.execvp(command[0], command)
    """,
}


_RUN_JOB = """
import json, os, signal, subprocess, sys, time
from pathlib import Path
state = Path(os.environ["FAKE_SLURM_DIR"])
job, options, script, outcome = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], sys.argv[4]
chdir = Path(options.get("--chdir", "."))
def where(key):
    return chdir / options[key].replace("%%", "%")
env = dict(os.environ)
if os.environ.get("FAKE_JOB_END_NOW"):
    env["SLURM_JOB_END_TIME"] = str(int(time.time()))
out = open(where("--output"), "w")
err = open(where("--error"), "w") if "--error" in options else subprocess.STDOUT
child = subprocess.Popen(["bash", script], cwd=chdir, stdout=out, stderr=err, env=env, start_new_session=True)
if outcome.startswith("TERM_AFTER="):
    time.sleep(float(outcome.split("=")[1]))
    os.killpg(child.pid, signal.SIGTERM)
    child.wait()
    final = "TIMEOUT|0:15"
else:
    rc = child.wait()
    final = ("COMPLETED" if rc == 0 else "FAILED") + f"|{rc}:0"
(state / f"{job}.state").write_text(final)
"""


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """Fake Slurm on PATH; returns a reader for the recorded calls."""
    bin_dir = tmp_path / "fake-slurm"
    bin_dir.mkdir()
    for name, body in _FAKES.items():
        tool = bin_dir / name
        tool.write_text(f"#!{sys.executable}\n{_COMMON}\n{textwrap.dedent(body)}", encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "run_job.py").write_text(_RUN_JOB, encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(bin_dir))
    for name in (
        "SLURM_JOB_ID", "SLURM_STEP_ID", "FAKE_SLURM_OUTCOME", "FAKE_SLURM_REJECT",
        "FAKE_SLURM_NO_SACCT", "FAKE_SQUEUE_FLAKY", "FAKE_SLURM_CLUSTER", "FAKE_SLURM_BANNER",
        "FAKE_SLURM_BANNER_ONLY", "FAKE_JOB_END_NOW", "FAKE_SACCT_LAG",
    ):
        monkeypatch.delenv(name, raising=False)

    def calls(tool=None):
        lines = (bin_dir / "calls.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        return [r for r in records if tool is None or r[0] == tool]

    calls.directory = bin_dir
    return calls


def _workdir(tmp_path: Path) -> Path:
    work = tmp_path / "case"
    work.mkdir()
    return work


def _python(work, code: str, **request) -> ExecutionRequest:
    return ExecutionRequest(command=(sys.executable, "-c", code), workdir=work, **request)


def _batch(**options) -> SlurmBackend:
    backend = SlurmBackend(mode="batch", poll_interval=0.01, **options)
    backend.status_grace = backend.accounting_grace = 0.2
    backend.cancel_grace = 0.5
    return backend


# -- pure helpers -------------------------------------------------------------


def test_walltime_rounds_up_to_whole_minutes():
    assert [walltime(s) for s in (1, 60, 61, 600.5)] == ["1", "1", "2", "11"]


def test_only_the_changed_part_of_the_environment_is_exported():
    parent = {"PATH": "/usr/bin", "HOME": "/home/u"}
    overlay = {
        "PATH": "/usr/bin", "HOME": "/elsewhere", "EFIT_FLAG": "1",
        "SLURM_MEM_PER_CPU": "1000", "SBATCH_ACCOUNT": "x", "BASH_FUNC_ml%%": "() { :; }",
    }
    assert exported_environment(overlay, parent) == {"HOME": "/elsewhere", "EFIT_FLAG": "1"}


def test_constructor_rejects_bad_options():
    with pytest.raises(ValueError):
        SlurmBackend(mode="cluster")
    with pytest.raises(ValueError):
        SlurmBackend(max_wait=0)
    with pytest.raises(ValueError, match="array"):
        SlurmBackend(extra_args=["--array=1-4"])


# -- batch mode ---------------------------------------------------------------


def test_batch_job_captures_output_and_exit_status(tmp_path, slurm):
    work = _workdir(tmp_path)
    result = _batch(partition="short", account="vest").run(
        _python(
            work,
            "import os, sys; print(os.getcwd()); print('err', file=sys.stderr); sys.exit(3)",
            timeout=90,
            resources=ResourceRequest(ntasks=4, threads_per_task=2, memory_mb=2048),
            label="efit",
        )
    )
    assert (result.returncode, result.timed_out, result.job_id) == (3, False, "1000")
    assert Path(result.stdout.strip()).resolve() == work.resolve()
    assert result.stderr.strip() == "err"
    (sbatch,) = slurm("sbatch")
    assert {
        "--parsable", "--no-requeue", "--export=ALL", "--job-name=vaft-efit",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=2048M", "--time=2",
        "--partition=short", "--account=vest",
    } <= set(sbatch)
    assert result.launcher[1:] == tuple(sbatch[1:])
    # A failed job keeps its scratch, script private, for inspection.
    (scratch,) = (work / SCRATCH_DIRECTORY).iterdir()
    assert stat.S_IMODE((scratch / "job.sh").stat().st_mode) == 0o700


def test_clean_batch_job_removes_its_scratch(tmp_path, slurm):
    work = _workdir(tmp_path)
    result = _batch().run(_python(work, "print('ok')"))
    assert (result.returncode, result.stdout.strip()) == (0, "ok")
    assert not (work / SCRATCH_DIRECTORY).exists()


def test_a_relative_workdir_is_resolved_before_submission(tmp_path, slurm, monkeypatch):
    _workdir(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = _batch().run(_python(Path("case"), "print('ok')"))
    assert (result.returncode, result.stdout.strip()) == (0, "ok")
    (sbatch,) = slurm("sbatch")
    assert f"--chdir={(tmp_path / 'case').resolve()}" in sbatch


def test_batch_job_gets_stdin_env_delta_and_thread_defaults(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1000")  # an enclosing allocation's
    work = _workdir(tmp_path)
    code = (
        "import os, sys; print(sys.stdin.read().split());"
        "print(os.environ['VAFT_FLAG'], os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])"
    )
    overlay = {**os.environ, "VAFT_FLAG": "it's quoted"}  # a full snapshot, as EFIT passes
    result = _batch().run(
        _python(work, code, stdin="2\n1\nk039915.00319\n", env=overlay,
                resources=ResourceRequest(threads_per_task=1))
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "['2', '1', 'k039915.00319']"
    assert lines[1] == "it's quoted 8 1"
    submitted = json.loads((slurm.directory / "sbatch_env.json").read_text())
    assert "SLURM_MEM_PER_CPU" not in submitted


def test_log_mode_merges_streams_into_the_log(tmp_path, slurm):
    work = _workdir(tmp_path)
    log = work / "logs" / "neo.log"
    result = _batch().run(
        _python(work, "import sys; print('out', flush=True); print('err', file=sys.stderr)", log_path=log)
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert result.log_path == log.resolve()
    assert log.read_text().split() == ["out", "err"]
    (sbatch,) = slurm("sbatch")
    assert not any(option.startswith("--error=") for option in sbatch)


def test_a_percent_in_the_path_is_escaped_for_slurm(tmp_path, slurm):
    work = tmp_path / "run%j"
    work.mkdir()
    result = _batch().run(_python(work, "print('ok')"))
    assert result.stdout.strip() == "ok"
    (sbatch,) = slurm("sbatch")
    assert any("run%%j" in option for option in sbatch if option.startswith("--output="))


def test_a_finished_job_still_listed_by_squeue_ends_the_wait(tmp_path, slurm):
    """A real controller lists COMPLETED jobs until MinJobAge; that is not live."""
    result = _batch(max_wait=30).run(_python(_workdir(tmp_path), "print('ok')"))
    assert (result.returncode, result.timed_out) == (0, False)
    assert not slurm("scancel")


def test_a_transient_squeue_failure_is_not_read_as_completion(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SQUEUE_FLAKY", "3")
    result = _batch().run(_python(_workdir(tmp_path), "print('ok')"))
    assert (result.returncode, result.stdout.strip()) == (0, "ok")
    assert len(slurm("squeue")) >= 4


def test_a_walltime_kill_is_a_timeout(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=0.5")
    result = _batch().run(_python(_workdir(tmp_path), "import time; time.sleep(30)", timeout=30))
    assert result.timed_out
    assert result.returncode is None


def test_a_walltime_kill_is_a_timeout_without_accounting(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=1.2")
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    result = _batch().run(_python(_workdir(tmp_path), "import time; time.sleep(30)", timeout=1))
    assert result.timed_out
    assert result.returncode is None


def test_a_termination_short_of_the_walltime_is_not_a_timeout(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=0.5")
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    result = _batch().run(_python(_workdir(tmp_path), "import time; time.sleep(30)", timeout=600))
    assert not result.timed_out
    assert result.returncode == 143
    assert "state unknown" in result.stderr


def test_max_wait_cancels_a_job_stuck_in_the_queue(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "PENDING")
    result = _batch(max_wait=0.05).run(_python(_workdir(tmp_path), "pass"))
    assert result.timed_out
    assert result.returncode is None
    assert "max_wait" in result.stderr
    assert slurm("scancel") == [["scancel", result.job_id]]


def test_an_interrupt_while_waiting_cancels_the_job(tmp_path, slurm, monkeypatch):
    import vaft.code.slurm as slurm_module

    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "PENDING")

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(slurm_module.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _batch().run(_python(_workdir(tmp_path), "pass"))
    assert slurm("scancel") == [["scancel", "1000"]]


@pytest.mark.parametrize("state", ["NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "CANCELLED"])
def test_a_job_ended_around_the_program_is_non_zero_with_its_state(tmp_path, slurm, monkeypatch, state):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", state)
    result = _batch().run(_python(_workdir(tmp_path), "pass"))
    assert not result.timed_out
    assert result.returncode == 128 + 9
    assert state in result.stderr


def test_a_job_ended_around_the_program_without_accounting(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "NODE_FAIL")
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    result = _batch().run(_python(_workdir(tmp_path), "pass"))
    assert (result.timed_out, result.returncode) == (False, 1)
    assert "state unknown" in result.stderr


def test_accounting_lag_is_waited_out(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=0.5")
    monkeypatch.setenv("FAKE_SACCT_LAG", "2")
    backend = _batch()
    backend.accounting_grace = 10
    result = backend.run(_python(_workdir(tmp_path), "import time; time.sleep(30)", timeout=600))
    assert result.timed_out  # TIMEOUT, once slurmdbd caught up, not a 143


def test_the_job_end_time_decides_a_walltime_kill_without_accounting(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=0.5")
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    monkeypatch.setenv("FAKE_JOB_END_NOW", "1")
    result = _batch().run(_python(_workdir(tmp_path), "import time; time.sleep(30)", timeout=600))
    assert result.timed_out


def test_a_squeue_that_never_answers_raises_rather_than_hanging(tmp_path, slurm, monkeypatch):
    import vaft.code.slurm as slurm_module

    monkeypatch.setattr(slurm_module, "_UNKNOWN_POLLS", 3)
    monkeypatch.setenv("FAKE_SQUEUE_FLAKY", "1000")
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    with pytest.raises(RuntimeError, match="may still be running"):
        _batch().run(_python(_workdir(tmp_path), "pass"))


def test_the_job_id_is_the_last_line_sbatch_prints(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_BANNER", "1")  # a site wrapper's banner first
    result = _batch().run(_python(_workdir(tmp_path), "pass"))
    assert (result.returncode, result.job_id) == (0, "1000")


def test_an_sbatch_that_prints_no_job_id_is_a_launch_failure(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_BANNER_ONLY", "1")
    with pytest.raises(ExecutableNotLaunchable, match="no job id"):
        _batch().run(_python(_workdir(tmp_path), "pass"))


def test_scheduler_variables_do_not_reach_sbatch(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("SRUN_CPUS_PER_TASK", "16")
    monkeypatch.setenv("SBATCH_ACCOUNT", "outer")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    _batch().run(_python(_workdir(tmp_path), "pass"))
    submitted = json.loads((slurm.directory / "sbatch_env.json").read_text())
    assert "SRUN_CPUS_PER_TASK" not in submitted and "SBATCH_ACCOUNT" not in submitted
    assert submitted["SLURM_CONF"] == "/etc/slurm/slurm.conf"


def test_exit_status_does_not_need_accounting(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_NO_SACCT", "1")
    result = _batch().run(_python(_workdir(tmp_path), "import sys; sys.exit(5)"))
    assert result.returncode == 5


def test_a_federated_job_id_queries_its_cluster(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_CLUSTER", "other")
    result = _batch().run(_python(_workdir(tmp_path), "pass"))
    assert (result.returncode, result.job_id) == (0, "1000")
    assert all(call[1:3] == ["-M", "other"] for call in slurm("squeue") + slurm("sacct"))


def test_a_rejected_submission_is_a_launch_failure(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_REJECT", "1")
    with pytest.raises(ExecutableNotLaunchable, match="invalid partition"):
        _batch(partition="nope").run(_python(_workdir(tmp_path), "pass"))


def test_missing_slurm_is_a_launch_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ExecutableNotLaunchable, match="sbatch"):
        _batch().run(_python(_workdir(tmp_path), "pass"))


# -- step mode ----------------------------------------------------------------


def test_step_mode_inside_an_allocation_runs_srun(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    work = _workdir(tmp_path)
    backend = SlurmBackend()
    assert backend.resolved_mode() == "step"
    result = backend.run(
        _python(
            work, "import sys; print(sys.stdin.read().strip())", stdin="hello", timeout=120,
            resources=ResourceRequest(ntasks=4, threads_per_task=2), label="neo",
        )
    )
    assert (result.returncode, result.stdout.strip(), result.job_id) == (0, "hello", "4242")
    (srun,) = slurm("srun")
    assert {
        "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--time=2",
        f"--chdir={work.resolve()}", "--export=ALL",
    } <= set(srun)
    # Unique per step, so a step whose srun had to be killed can be found.
    assert any(option.startswith("--job-name=vaft-neo-") for option in srun)
    assert "--overlap" not in srun
    assert not slurm("sbatch")


def test_a_step_launched_from_a_step_overlaps(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    SlurmBackend().run(_python(_workdir(tmp_path), "pass"))
    (srun,) = slurm("srun")
    assert "--overlap" in srun


def test_a_step_does_not_inherit_the_enclosing_steps_binding(tmp_path, slurm, monkeypatch):
    """`srun python driver.py` exports its own CPU mask; the new step must not reuse it."""
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    for name, value in {
        "SLURM_CPU_BIND": "quiet,mask_cpu:0xFF", "SLURM_CPU_BIND_LIST": "0xFF",
        "SLURM_MEM_PER_CPU": "1000", "SLURM_CPUS_PER_TASK": "8", "SRUN_CPUS_PER_TASK": "8",
    }.items():
        monkeypatch.setenv(name, value)
    SlurmBackend().run(_python(_workdir(tmp_path), "pass", resources=ResourceRequest(memory_mb=512)))
    environment = json.loads((slurm.directory / "srun_env.json").read_text())
    assert not [key for key in environment if key.startswith(("SLURM_CPU_BIND", "SLURM_MEM_PER", "SLURM_CPUS_PER", "SRUN_"))]
    assert environment["SLURM_JOB_ID"] == "4242"


def test_a_step_timeout_interrupts_srun_and_is_returned(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    backend = SlurmBackend(step_grace=10)
    result = backend.run(
        _python(_workdir(tmp_path), "import time; print('started', flush=True); time.sleep(30)", timeout=1.5)
    )
    assert result.timed_out
    assert result.returncode is None
    assert "started" in result.stdout


def test_step_log_mode_writes_the_log(tmp_path, slurm, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    work = _workdir(tmp_path)
    log = work / "gpec.log"
    result = SlurmBackend().run(_python(work, "print('out')", log_path=log))
    assert (result.returncode, result.stdout) == (0, "")
    assert log.read_text().strip() == "out"


# -- selection and adapters ---------------------------------------------------


def test_auto_mode_outside_an_allocation_is_batch(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert SlurmBackend().resolved_mode() == "batch"


def test_environment_selects_the_default_backend(monkeypatch):
    for name in ("VAFT_SLURM_MODE", "VAFT_SLURM_QOS", "VAFT_SLURM_ACCOUNT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert isinstance(default_backend(), LocalBackend)
    monkeypatch.setenv(BACKEND_ENV, "slurm")
    monkeypatch.setenv("VAFT_SLURM_PARTITION", "short")
    monkeypatch.setenv("VAFT_SLURM_MAX_WAIT", "600")
    backend = resolve_backend(None)
    assert isinstance(backend, SlurmBackend)
    assert (backend.partition, backend.max_wait) == ("short", 600.0)
    monkeypatch.setenv(BACKEND_ENV, "pbs")
    with pytest.raises(ValueError, match=BACKEND_ENV):
        default_backend()


def test_an_adapter_runs_unchanged_through_slurm(tmp_path, slurm):
    """End to end: TES, with only `backend=` changed, as a batch job."""
    from vaft.code.tes.config import TESConfig, TESInputs
    from vaft.code.tes.runner import run_tes

    work = _workdir(tmp_path)
    cinput = work / "rtes.in"
    cinput.write_text("DUMMY")
    program = work / "rtes"
    program.write_text("#!/bin/sh\necho solved $1\n", encoding="utf-8")
    program.chmod(0o755)
    result = run_tes(
        TESInputs(workdir=work, cinput=cinput),
        TESConfig(executable=str(program), timeout=120, backend=_batch()),
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "solved rtes.in"
    assert slurm("sbatch")
