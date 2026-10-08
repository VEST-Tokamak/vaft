"""RemoteSlurmBackend against a fake ssh, a fake rsync and the fake Slurm.

No network and no cluster. ``ssh`` and ``rsync`` are small Python programs put
first on ``PATH``; the "remote" filesystem is a second directory tree under
``tmp_path``, so every path the backend sends has to be translated to reach it
and a missed translation fails the test rather than silently working.

* ``ssh`` ignores its options (after recording them), takes the destination and
  the one command string, and runs it with ``bash -c`` -- which finds the fake
  ``sbatch``/``squeue``/``sacct``/``scancel`` on the same ``PATH``.
  ``FAKE_SSH_UNREACHABLE`` makes every call fail the way a refused connection
  does, with ssh's own exit status 255; ``FAKE_SSH_FLAKY=<n>:<prefix>`` fails
  the first ``n`` calls whose command starts with ``prefix``.
* ``rsync`` implements the subset the backend emits: ``-a`` (modes kept, so the
  job script stays executable), ``-e``, ``-m`` with ``--include``/``--exclude``,
  and a ``host:path`` endpoint on either side. Its filter follows rsync's
  rules -- a pattern is anchored at the transfer root only with a leading
  ``/``, and otherwise matches any trailing part of the path -- and
  ``test_the_fetch_rules_select_the_same_files_real_rsync_would`` checks that
  against the real program where one is installed.
* the Slurm fakes are :mod:`test_slurm_backend`'s, so both suites answer to one
  description of how a controller behaves; only ``sacct`` is overridden here, to
  be able to report Start/End stamps.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import test_slurm_backend as fake_slurm
from vaft.compat import IS_WINDOWS
from vaft.code import ExecutableNotLaunchable, ExecutionRequest, ResourceRequest
from vaft.code.execution import BACKEND_ENV, default_backend
from vaft.code.remote import (
    FETCH_ALL,
    FETCH_NONE,
    UNFORWARDED_ENVIRONMENT,
    RemoteHost,
    RemoteSlurmBackend,
)
from vaft.code.slurm import SCRATCH_DIRECTORY

pytestmark = pytest.mark.skipif(IS_WINDOWS, reason="ssh, rsync and job scripts are POSIX")


_SSH = """
    # The destination is the last argument before the command; everything
    # before it is options this fake does not need to honour.
    command = sys.argv[-1]
    if os.environ.get("FAKE_SSH_UNREACHABLE"):
        print("ssh: connect to host port 22: Connection refused", file=sys.stderr)
        sys.exit(255)
    flaky = os.environ.get("FAKE_SSH_FLAKY", "")  # '<n>:<command prefix>'
    if flaky:
        limit, _, prefix = flaky.partition(":")
        if command.startswith(prefix):
            counter = state / "ssh_flaky"
            count = int(counter.read_text()) if counter.exists() else 0
            if count < int(limit):
                counter.write_text(str(count + 1))
                print("ssh: connect to host port 22: Connection timed out", file=sys.stderr)
                sys.exit(255)
    os.execvp("bash", ["bash", "-c", command])
"""

_RSYNC = """
    import fnmatch, shutil
    argv, options, paths = sys.argv[1:], [], []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "-e":
            index += 2
            continue
        if argument.startswith("-"):
            options.append(argument)
            index += 1
            continue
        paths.append(argument)
        index += 1
    includes = [o.split("=", 1)[1] for o in options if o.startswith("--include=")]
    excludes = [o.split("=", 1)[1] for o in options if o.startswith("--exclude=")]
    wanted = [p for p in includes if p != "*/"]
    source, destination = paths[-2], paths[-1]
    def local(path):
        # 'host:/path' or 'user@host:/path' -> '/path'; a local path keeps its ':'
        head, separator, tail = path.partition(":")
        return tail if separator and "/" not in head else path
    source, destination = local(source), local(destination)
    def matches(relative, pattern):
        # rsync anchors a pattern that starts with '/' at the transfer root and
        # otherwise matches it against any trailing part of the path.
        if pattern.startswith("/"):
            return fnmatch.fnmatch(relative, pattern[1:])
        parts = relative.split("/")
        return any(fnmatch.fnmatch("/".join(parts[i:]), pattern) for i in range(len(parts)))
    def keep(relative):
        if not wanted:
            return True
        if any(matches(relative, p) for p in wanted):
            return True
        return not any(p == "*" for p in excludes)
    if source.endswith("/"):
        root = Path(source.rstrip("/"))
        if not root.is_dir():
            print(f"rsync: change_dir \\"{root}\\" failed: No such file or directory (2)", file=sys.stderr)
            sys.exit(23)
        for item in sorted(root.rglob("*")):
            if not item.is_file():
                continue
            relative = str(item.relative_to(root))
            if not keep(relative):
                continue
            target = Path(destination.rstrip("/")) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    else:
        if not Path(source).exists():
            print(f"rsync: link_stat \\"{source}\\" failed: No such file or directory (2)", file=sys.stderr)
            sys.exit(23)
        target = Path(destination)
        if target.is_dir():
            target = target / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
"""

# The shared fake prints only State|ExitCode; the remote backend asks for the
# stamps too, and reports run_s from them.
_SACCT = """
    job = sys.argv[sys.argv.index("-j") + 1]
    record = (state / f"{job}.state").read_text().strip()
    window = os.environ.get("FAKE_SACCT_WINDOW", "")
    print(f"{record}|{window}" if window else record)
"""


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """Fake ssh, rsync and Slurm on PATH; returns a reader for recorded calls."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    fakes = {**fake_slurm._FAKES, "sacct": _SACCT, "ssh": _SSH, "rsync": _RSYNC}
    for name, body in fakes.items():
        tool = bin_dir / name
        tool.write_text(
            f"#!{sys.executable}\n{fake_slurm._COMMON}\n{textwrap.dedent(body)}", encoding="utf-8"
        )
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    (bin_dir / "run_job.py").write_text(fake_slurm._RUN_JOB, encoding="utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_DIR", str(bin_dir))
    for name in (
        "SLURM_JOB_ID", "SLURM_STEP_ID", "FAKE_SLURM_OUTCOME", "FAKE_SLURM_REJECT",
        "FAKE_SLURM_NO_SACCT", "FAKE_SQUEUE_FLAKY", "FAKE_SLURM_CLUSTER", "FAKE_SLURM_BANNER",
        "FAKE_SLURM_BANNER_ONLY", "FAKE_JOB_END_NOW", "FAKE_SACCT_LAG",
        "FAKE_SSH_UNREACHABLE", "FAKE_SSH_FLAKY", "FAKE_SACCT_WINDOW",
        "FAKE_SLURM_REJECT_MESSAGE",
    ):
        monkeypatch.delenv(name, raising=False)

    def calls(tool=None):
        records = [
            json.loads(line) for line in (bin_dir / "calls.jsonl").read_text().splitlines()
        ]
        return [r for r in records if tool is None or r[0] == tool]

    calls.directory = bin_dir
    return calls


@pytest.fixture
def side(tmp_path):
    """``(local, remote)`` roots: two trees that share no path."""
    local, remote = tmp_path / "local", tmp_path / "remote"
    (local / "case").mkdir(parents=True)
    remote.mkdir()
    return local, remote


def _host(local: Path, remote: Path, **options) -> RemoteHost:
    return RemoteHost(
        host="cluster.invalid",
        work_root=str(remote / "work"),
        path_maps=((str(local), str(remote / "mapped")),),
        **options,
    )


def _backend(host: RemoteHost, **options) -> RemoteSlurmBackend:
    backend = RemoteSlurmBackend(host, poll_interval=0.01, max_poll_interval=0.02, **options)
    backend.status_grace = backend.accounting_grace = 0.2
    backend.cancel_grace = 0.5
    return backend


def _shell(work: Path, body: str, **request) -> ExecutionRequest:
    """A request running ``body`` with ``sh``; ``sh`` needs no translation."""
    return ExecutionRequest(command=("/bin/sh", "-c", body), workdir=work, **request)


# -- RemoteHost, no subprocess at all ----------------------------------------


def test_host_rejects_configuration_it_cannot_use():
    with pytest.raises(ValueError, match="host"):
        RemoteHost(host="  ", work_root="/scratch")
    with pytest.raises(ValueError, match="absolute"):
        RemoteHost(host="h", work_root="scratch")
    with pytest.raises(ValueError, match="port"):
        RemoteHost(host="h", work_root="/scratch", port=0)
    with pytest.raises(ValueError, match="absolute"):
        RemoteHost(host="h", work_root="/scratch", path_maps=(("relative", "/remote"),))
    with pytest.raises(ValueError, match="fetch"):
        RemoteHost(host="h", work_root="/scratch", fetch="some")
    with pytest.raises(ValueError, match="empty fetch"):
        RemoteHost(host="h", work_root="/scratch", fetch=[])


def test_path_maps_apply_in_order_and_only_at_a_boundary():
    host = RemoteHost(
        host="h",
        work_root="/scratch",
        path_maps=(("/data/runs/special", "/fast/special"), ("/data/runs", "/slow/runs")),
    )
    # The first matching pair wins, so a narrower prefix listed first keeps its
    # own destination instead of being swallowed by the broader one.
    assert host.remote_path("/data/runs/special/39915") == "/fast/special/39915"
    assert host.remote_path("/data/runs/39915") == "/slow/runs/39915"
    assert host.remote_path("/data/runs") == "/slow/runs"
    # A sibling whose name merely starts with the prefix is not under it.
    assert host.remote_path("/data/runs2/39915") is None
    assert host.translate("/data/runs2/39915") == "/data/runs2/39915"
    # Arguments that are not paths pass through untouched.
    assert [host.translate(a) for a in ("vacuum", "-n", "4")] == ["vacuum", "-n", "4"]
    # An option carrying a path after '=' matches no prefix as a whole, and
    # would otherwise reach the node naming a directory that is not there.
    assert host.translate("--home=/data/runs/39915") == "--home=/slow/runs/39915"
    assert host.translate("mode=strict") == "mode=strict"


def test_environment_values_are_translated_per_path_segment():
    host = RemoteHost(host="h", work_root="/scratch", path_maps=(("/opt/gpec", "/apps/gpec"),))
    env = host.forwarded_environment(
        {
            "GPEC_HOME": "/opt/gpec",
            "SEARCH": "/opt/gpec/lib:/usr/lib:/opt/gpec/extra",
            "RUN_MODE": "strict",
            "PATH": "/opt/gpec/bin:/usr/bin",
            "DYLD_LIBRARY_PATH": "/opt/gpec/lib",
            "LC_ALL": "C",
        }
    )
    assert env == {
        "GPEC_HOME": "/apps/gpec",
        "SEARCH": "/apps/gpec/lib:/usr/lib:/apps/gpec/extra",
        "RUN_MODE": "strict",
    }
    # The dropped ones describe the submitting machine, not the compute node.
    assert {"PATH", "DYLD_", "LC_"} <= set(UNFORWARDED_ENVIRONMENT)


def test_ssh_is_always_batch_mode_and_carries_the_login_options(cluster):
    host = RemoteHost(
        host="h", work_root="/scratch", user="u", port=2222,
        identity_file="/keys/id", ssh_options=("-o", "StrictHostKeyChecking=accept-new"),
    )
    prefix = host.ssh_prefix()
    assert prefix[1:] == [
        "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-p", "2222", "-i", "/keys/id",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    assert host.ssh_argv("squeue")[-2:] == ["u@h", "squeue"]
    assert host.remote("/scratch/x") == "u@h:/scratch/x"
    # ssh takes the first value given for an option, so batch mode comes first
    # and a caller's own options cannot turn prompting back on -- a host that
    # would prompt has to fail instead of blocking a batch for ever.
    argued = RemoteHost(
        host="h", work_root="/scratch", ssh_options=("-o", "BatchMode=no")
    ).ssh_prefix()
    assert argued.index("BatchMode=yes") < argued.index("BatchMode=no")


def test_host_repr_does_not_spell_out_the_path_maps():
    host = RemoteHost(host="h", work_root="/scratch", path_maps=(("/a", "/b"), ("/c", "/d")))
    assert repr(host) == "RemoteHost(host='h', work_root='/scratch', path_maps=2, fetch='all')"


def test_backend_rejects_configuration_it_cannot_use():
    host = RemoteHost(host="h", work_root="/scratch")
    with pytest.raises(TypeError, match="RemoteHost"):
        RemoteSlurmBackend("h")
    with pytest.raises(ValueError, match="max_wait"):
        RemoteSlurmBackend(host, max_wait=0)
    with pytest.raises(ValueError, match="max_poll_interval"):
        RemoteSlurmBackend(host, poll_interval=30, max_poll_interval=10)
    with pytest.raises(ValueError, match="backoff"):
        RemoteSlurmBackend(host, backoff=0.5)
    with pytest.raises(ValueError, match="array"):
        RemoteSlurmBackend(host, extra_args=["--array=1-4"])


def test_from_environment_reads_the_documented_variables(monkeypatch):
    for name, value in {
        "VAFT_REMOTE_HOST": "login", "VAFT_REMOTE_WORK_ROOT": "/scratch/runs",
        "VAFT_REMOTE_USER": "u", "VAFT_REMOTE_PORT": "2222",
        "VAFT_REMOTE_SSH_OPTIONS": "-o 'ServerAliveInterval=30'",
        "VAFT_REMOTE_PATH_MAPS": "/opt/gpec=/apps/gpec,/data=/work/data",
        "VAFT_REMOTE_SETUP": "module purge; module load gpec",
        "VAFT_REMOTE_FETCH": "*.nc,*.out", "VAFT_REMOTE_KEEP": "0",
        "VAFT_SLURM_PARTITION": "short", "VAFT_SLURM_MAX_WAIT": "7200",
    }.items():
        monkeypatch.setenv(name, value)
    backend = RemoteSlurmBackend.from_environment()
    host = backend.host
    assert (host.host, host.user, host.port) == ("login", "u", 2222)
    assert host.path_maps == (("/opt/gpec", "/apps/gpec"), ("/data", "/work/data"))
    assert host.setup_lines == ("module purge", "module load gpec")
    assert (host.fetch, host.keep_remote) == (("*.nc", "*.out"), False)
    assert host.ssh_options == ("-o", "ServerAliveInterval=30")
    assert (backend.partition, backend.max_wait) == ("short", 7200.0)


def test_from_environment_says_what_is_missing(monkeypatch):
    monkeypatch.delenv("VAFT_REMOTE_HOST", raising=False)
    monkeypatch.delenv("VAFT_REMOTE_WORK_ROOT", raising=False)
    with pytest.raises(ValueError, match="VAFT_REMOTE_HOST and VAFT_REMOTE_WORK_ROOT"):
        RemoteHost.from_environment()
    monkeypatch.setenv("VAFT_REMOTE_HOST", "login")
    with pytest.raises(ValueError, match="VAFT_REMOTE_WORK_ROOT"):
        RemoteHost.from_environment()
    monkeypatch.setenv("VAFT_REMOTE_WORK_ROOT", "/scratch")
    monkeypatch.setenv("VAFT_REMOTE_PATH_MAPS", "/opt/gpec")
    with pytest.raises(ValueError, match="local=remote"):
        RemoteHost.from_environment()


def test_the_backend_environment_variable_selects_it(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "remote")
    monkeypatch.setenv("VAFT_REMOTE_HOST", "login")
    monkeypatch.setenv("VAFT_REMOTE_WORK_ROOT", "/scratch/runs")
    backend = default_backend()
    assert isinstance(backend, RemoteSlurmBackend)
    assert backend.host.host == "login"
    monkeypatch.setenv(BACKEND_ENV, "elsewhere")
    with pytest.raises(ValueError, match="remote"):
        default_backend()


# -- one job, end to end ------------------------------------------------------


def test_a_mapped_working_directory_runs_at_its_remote_path(side, cluster):
    local, remote = side
    work = local / "case"
    (remote / "mapped" / "codes" / "bin").mkdir(parents=True)
    program = remote / "mapped" / "codes" / "bin" / "prog"
    program.write_text('#!/bin/sh\necho "home=$CODE_HOME"\npwd\n', encoding="utf-8")
    program.chmod(0o755)

    result = _backend(_host(local, remote)).run(
        ExecutionRequest(
            # Both the executable and the environment name this machine's tree.
            command=(str(local / "codes" / "bin" / "prog"),),
            workdir=work,
            env={"CODE_HOME": str(local / "codes")},
            log_path=work / "logs" / "run.log",
            timeout=90,
            resources=ResourceRequest(ntasks=4, threads_per_task=2, memory_mb=2048),
            label="prog",
        )
    )

    remote_work = str(remote / "mapped" / "case")
    assert (result.returncode, result.timed_out, result.job_id) == (0, False, "1000")
    log = (work / "logs" / "run.log").read_text()
    # The program ran from the host's own build, with the host's own home, in
    # the host's copy of the working directory.
    assert f"home={remote / 'mapped' / 'codes'}" in log
    assert remote_work in log
    assert str(local) not in log
    (sbatch,) = cluster("sbatch")
    assert {
        "--parsable", "--no-requeue", "--export=ALL", "--job-name=vaft-prog",
        f"--chdir={remote_work}", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
        "--mem=2048M", "--time=2", f"--output={remote_work}/logs/run.log",
    } <= set(sbatch)
    # The launcher records the submission, not the ssh login.
    assert result.launcher[0] == "sbatch"


def test_the_job_script_carries_the_setup_lines_and_only_portable_environment(side, cluster):
    local, remote = side
    work = local / "case"
    host = _host(local, remote, setup_lines=("module purge", "module load code/1.2"))
    _backend(host).run(
        _shell(
            work,
            "true",
            env={"CODE_HOME": str(local / "codes"), "PATH": "/opt/local/bin", "RUN": "1"},
            resources=ResourceRequest(threads_per_task=3),
            label="setup",
        )
    )
    (script,) = (remote / "mapped" / "case" / SCRATCH_DIRECTORY).rglob("job.sh")
    lines = script.read_text().splitlines()
    assert lines[:3] == ["#!/bin/bash", "module purge", "module load code/1.2"]
    assert f"cd {remote / 'mapped' / 'case'} || exit 1" in lines
    assert f"export CODE_HOME={remote / 'mapped' / 'codes'}" in lines
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"' in lines
    assert "export RUN=1" in lines
    # A submitting machine's PATH would shadow what the modules just set up.
    assert not any(line.startswith("export PATH=") for line in lines)
    # rsync has to keep the mode, or the submitted script is not executable.
    assert stat.S_IMODE(script.stat().st_mode) == 0o700


def test_an_unmapped_working_directory_is_staged_under_the_work_root(side, cluster):
    local, remote = side
    work = local.parent / "elsewhere"  # outside every path pair
    work.mkdir()
    (work / "input.dat").write_text("7\n", encoding="utf-8")
    result = _backend(_host(local, remote)).run(
        _shell(work, "cat input.dat > echoed.dat", label="staged")
    )
    assert result.returncode == 0
    (staged,) = (remote / "work").iterdir()
    assert staged.name.startswith("staged-")
    assert (staged / "input.dat").read_text() == "7\n"
    # And the output came back to the local working directory.
    assert (work / "echoed.dat").read_text() == "7\n"


# -- fetch policy -------------------------------------------------------------


def _produce(work: Path, host: RemoteHost) -> tuple[Path, Path]:
    """Run a job that writes one big and one small output; return both trees.

    The job also says something on stdout, so the three policy tests can check
    that the log came back with content rather than merely existing -- ``run``
    creates it either way.
    """
    result = _backend(host).run(
        _shell(
            work,
            "echo ran-on-the-host && mkdir -p out && echo big > out/field.nc "
            "&& echo small > out/summary.txt",
            log_path=work / "run.log",
            label="outputs",
        )
    )
    assert result.returncode == 0
    (staged,) = (Path(host.work_root)).iterdir()
    return work, staged


def test_fetch_all_brings_the_whole_working_directory_back(side, cluster):
    local, remote = side
    work = local.parent / "elsewhere"
    work.mkdir()
    work, staged = _produce(work, _host(local, remote, fetch=FETCH_ALL))
    assert (work / "out" / "field.nc").read_text() == "big\n"
    assert (work / "out" / "summary.txt").read_text() == "small\n"
    assert "ran-on-the-host" in (work / "run.log").read_text()


def test_fetch_patterns_bring_back_only_what_they_name(side, cluster):
    local, remote = side
    work = local.parent / "elsewhere"
    work.mkdir()
    work, staged = _produce(work, _host(local, remote, fetch=("*.txt",)))
    assert (work / "out" / "summary.txt").read_text() == "small\n"
    assert not (work / "out" / "field.nc").exists()
    # The run's own record comes back whatever the policy says -- and the
    # pattern would have excluded the log along with everything else.
    assert "ran-on-the-host" in (work / "run.log").read_text()
    # And the full output is still on the host.
    assert (staged / "out" / "field.nc").read_text() == "big\n"


def test_fetch_none_brings_back_only_the_log_and_the_status(side, cluster):
    local, remote = side
    work = local.parent / "elsewhere"
    work.mkdir()
    work, staged = _produce(work, _host(local, remote, fetch=FETCH_NONE))
    assert not (work / "out").exists()
    assert "ran-on-the-host" in (work / "run.log").read_text()
    assert (staged / "out" / "field.nc").exists()


def test_the_remote_copy_is_kept_unless_asked_otherwise(side, cluster):
    local, remote = side
    for name, keep in (("kept", True), ("dropped", False)):
        work = local.parent / name
        work.mkdir()
        _backend(_host(local, remote, keep_remote=keep)).run(
            _shell(work, "echo done > out.txt", label=name)
        )
        staged = [path for path in (remote / "work").iterdir() if path.name.startswith(name)]
        assert bool(staged) is keep, f"keep_remote={keep} left {staged}"
        assert (work / "out.txt").read_text() == "done\n"


@pytest.mark.parametrize("label", ["tables/../../scratch2", "../..", "a b", ".hidden", "x;rm"])
def test_a_label_that_is_not_one_path_component_is_refused_before_anything_is_staged(side, cluster, label):
    """``label`` names the remote staging directory, which ``keep_remote=False``
    then ``rm -rf``s, and the local scratch directory: ``..`` or a separator in
    it would reach outside both (cold review 0.8.0 plot-gui-packaging F2)."""
    local, remote = side
    work = local.parent / "elsewhere"
    work.mkdir()
    with pytest.raises(ValueError, match="label"):
        _backend(_host(local, remote, keep_remote=False)).run(_shell(work, "true", label=label))
    assert not (cluster.directory / "calls.jsonl").exists()  # no ssh, no rsync: refused before the host was contacted
    assert not (work / SCRATCH_DIRECTORY).exists()
    assert not (remote / "work").exists()


def test_a_mapped_working_directory_is_never_removed(side, cluster):
    """``keep_remote=False`` clears what this backend staged, nothing else.

    A path pair points at the caller's own tree on the host -- a shared project
    directory, say -- and deleting that because one run finished would destroy
    work the run never owned.
    """
    local, remote = side
    work = local / "case"
    _backend(_host(local, remote, keep_remote=False)).run(
        _shell(work, "echo done > out.txt", label="mapped")
    )
    assert (remote / "mapped" / "case" / "out.txt").read_text() == "done\n"


@pytest.mark.parametrize(
    "patterns",
    [("*.txt", "out/*.nc"), ("/out/*.nc",), ("*.nc",)],
    ids=["tail", "anchored", "extension"],
)
def test_the_fetch_rules_select_the_same_files_real_rsync_would(tmp_path, cluster, patterns):
    """The shim's filter is only a test double; rsync is the authority.

    The rule set the backend builds for a pattern policy is run through both
    the shim and the real program, locally and without ssh, so a policy that
    passes here cannot quietly mean something else on a cluster. rsync matches
    a pattern against any trailing part of the path unless it starts with
    ``/``, which is not what a plain prefix test would do.
    """
    # The fixture's shim is first on PATH under the same name; look past it.
    elsewhere = [
        entry for entry in os.environ["PATH"].split(os.pathsep) if entry != str(cluster.directory)
    ]
    rsync = shutil.which("rsync", path=os.pathsep.join(elsewhere))
    if rsync is None:
        pytest.skip("the real rsync is not installed")
    source = tmp_path / "src"
    for relative in ("summary.txt", "out/field.nc", "out/notes.txt", "deep/out/trace.txt",
                     "deep/out/other.nc"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    rules = ["-m", "--include=*/", *[f"--include={p}" for p in patterns], "--exclude=*"]

    def selected(program, destination):
        destination.mkdir()
        done = subprocess.run(
            [str(program), "-a", *rules, f"{source}/", f"{destination}/"],
            capture_output=True, text=True,
        )
        assert done.returncode == 0, done.stderr
        return sorted(
            str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()
        )

    by_real = selected(rsync, tmp_path / "real")
    assert by_real, "the rule set selected nothing, so the comparison proves nothing"
    assert selected(cluster.directory / "rsync", tmp_path / "shim") == by_real


# -- the limits, returned rather than raised ---------------------------------


def test_a_job_slurm_kills_on_its_walltime_comes_back_as_a_timeout(side, cluster, monkeypatch):
    local, remote = side
    work = local / "case"
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "TERM_AFTER=0.3")
    result = _backend(_host(local, remote)).run(
        _shell(work, "sleep 30", timeout=60, log_path=work / "run.log", label="slow")
    )
    assert (result.returncode, result.timed_out) == (None, True)
    assert result.runtime_status == "timeout"
    # The Slurm state that ended it is appended to the log the job wrote.
    assert "slurm job 1000 ended TIMEOUT" in (work / "run.log").read_text()
    # The node's own stamps stand in for accounting that reported no window.
    assert result.run_s is not None and result.run_s < 60.0


def test_a_job_cancelled_in_the_queue_comes_back_as_a_queue_timeout(side, cluster, monkeypatch):
    local, remote = side
    work = local / "case"
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "PENDING")  # never runs
    result = _backend(_host(local, remote), max_wait=0.05).run(
        _shell(work, "sleep 30", timeout=60, log_path=work / "run.log", label="queued")
    )
    assert (result.returncode, result.timed_out) == (None, True)
    assert result.runtime_status == "queue_timeout"
    # The log the job never wrote is still the place the reason is recorded.
    assert "it never started" in (work / "run.log").read_text()
    assert any(call[0] == "scancel" for call in cluster())


def test_a_queued_cancel_does_not_leave_an_earlier_run_in_the_log(side, cluster, monkeypatch):
    """A job that wrote no log must not inherit the file sitting at that path.

    The same working directory is normally reused across attempts, and the
    adapter reads ``log_path`` to explain a failure; a previous run's output
    there would be read as this run's.
    """
    local, remote = side
    work = local / "case"
    log = work / "run.log"
    log.write_text("output of an earlier, successful attempt\n", encoding="utf-8")
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "PENDING")
    result = _backend(_host(local, remote), max_wait=0.05).run(
        _shell(work, "true", timeout=60, log_path=log, label="requeued")
    )
    assert result.runtime_status == "queue_timeout"
    assert "earlier" not in log.read_text()
    assert "it never started" in log.read_text()


def test_run_s_comes_from_the_scheduler_stamps(side, cluster, monkeypatch):
    local, remote = side
    work = local / "case"
    monkeypatch.setenv("FAKE_SACCT_WINDOW", "2026-10-03T10:00:00|2026-10-03T10:02:30")
    result = _backend(_host(local, remote)).run(_shell(work, "true", label="stamped"))
    assert result.run_s == 150.0
    # elapsed_s is this call's own wall time, queue and transfers included.
    assert result.elapsed_s < 150.0


def test_without_accounting_stamps_there_is_no_run_s(side, cluster):
    local, remote = side
    result = _backend(_host(local, remote)).run(_shell(local / "case", "true", label="plain"))
    assert (result.returncode, result.run_s) == (0, None)


# -- interrupts ---------------------------------------------------------------


def test_an_interrupt_while_waiting_cancels_the_job_and_is_re_raised(side, cluster, monkeypatch):
    local, remote = side
    backend = _backend(_host(local, remote))
    polls = {"n": 0}
    original = RemoteSlurmBackend._poll

    def interrupted(self, job_id, clusters):
        polls["n"] += 1
        if polls["n"] > 1:
            raise KeyboardInterrupt
        return original(self, job_id, clusters)

    monkeypatch.setattr(RemoteSlurmBackend, "_poll", interrupted)
    monkeypatch.setenv("FAKE_SLURM_OUTCOME", "QUEUE_FOR=3")  # still queued when it lands
    with pytest.raises(KeyboardInterrupt):
        backend.run(_shell(local / "case", "true", label="stopped"))
    cancels = [call for call in cluster() if call[0] == "scancel"]
    assert cancels and cancels[0][-1] == "1000"
    assert (cluster.directory / "1000.state").read_text().startswith("CANCELLED")


# -- submissions that never become a job --------------------------------------


def test_a_refused_submission_is_not_launchable(side, cluster, monkeypatch):
    local, remote = side
    monkeypatch.setenv("FAKE_SLURM_REJECT", "1")
    with pytest.raises(ExecutableNotLaunchable, match="sbatch rejected the job"):
        _backend(_host(local, remote)).run(_shell(local / "case", "true", label="refused"))
    # Submitted once: a rejection is reported, never retried, because a retry
    # could duplicate a job the controller had in fact accepted.
    assert len(cluster("sbatch")) == 1


def test_a_rejection_quoting_a_connection_error_is_still_not_retried(side, cluster, monkeypatch):
    """A site login banner shares stderr with the scheduler's own message.

    If any connection-sounding text in that stream counted, a banner
    mentioning a timeout would make every plain rejection retry -- and a
    submission the controller did accept could then be sent twice. Only ssh's
    own exit status 255 admits ssh's wording.
    """
    local, remote = side
    monkeypatch.setenv("FAKE_SLURM_REJECT", "1")
    monkeypatch.setenv("FAKE_SLURM_REJECT_MESSAGE", "banner: Connection timed out earlier today")
    with pytest.raises(ExecutableNotLaunchable, match="sbatch rejected the job"):
        _backend(_host(local, remote)).run(_shell(local / "case", "true", label="banner"))
    assert len(cluster("sbatch")) == 1


def test_a_host_that_cannot_be_reached_is_not_launchable(side, cluster, monkeypatch):
    local, remote = side
    monkeypatch.setenv("FAKE_SSH_UNREACHABLE", "1")
    with pytest.raises(ExecutableNotLaunchable, match="cannot prepare the remote working"):
        _backend(_host(local, remote)).run(_shell(local / "case", "true", label="unreachable"))


def test_a_submission_with_no_job_id_is_not_launchable(side, cluster, monkeypatch):
    local, remote = side
    monkeypatch.setenv("FAKE_SLURM_BANNER_ONLY", "1")
    with pytest.raises(ExecutableNotLaunchable, match="printed no job id"):
        _backend(_host(local, remote)).run(_shell(local / "case", "true", label="silent"))


def test_a_site_banner_before_the_job_id_is_tolerated(side, cluster, monkeypatch):
    local, remote = side
    monkeypatch.setenv("FAKE_SLURM_BANNER", "1")
    monkeypatch.setenv("FAKE_SLURM_CLUSTER", "alpha")
    result = _backend(_host(local, remote)).run(_shell(local / "case", "true", label="banner"))
    assert (result.job_id, result.returncode) == ("1000", 0)


def test_a_submission_is_retried_only_while_the_host_was_never_reached(side, cluster, monkeypatch):
    """A connect-time failure cannot have left a job behind, so retrying is safe.

    Anything else is reported at once: a retry there could submit the same job
    twice, and two jobs writing one working directory is worse than a failure.
    """
    local, remote = side
    monkeypatch.setenv("FAKE_SSH_FLAKY", "2:sbatch")
    result = _backend(_host(local, remote)).run(_shell(local / "case", "true", label="retried"))
    assert result.returncode == 0
    assert len(cluster("sbatch")) == 1  # three ssh attempts, one accepted job


def test_stdin_reaches_the_program_on_the_host(side, cluster):
    local, remote = side
    work = local / "case"
    result = _backend(_host(local, remote)).run(
        _shell(work, "cat > copied.txt", stdin="seventeen\n", label="fed")
    )
    assert result.returncode == 0
    assert (work / "copied.txt").read_text() == "seventeen\n"


def test_a_missing_working_directory_is_its_own_error(side, cluster):
    local, remote = side
    with pytest.raises(FileNotFoundError, match="working directory"):
        _backend(_host(local, remote)).run(_shell(local / "absent", "true"))


# -- every heavy launcher has to go through the backend ----------------------
#
# A remote backend is only reachable by an adapter that asks for one. An
# adapter that starts its own subprocess runs on the submitting machine
# whatever VAFT_EXECUTION_BACKEND says, and the failure is silent: the run
# simply happens in the wrong place.


class _Recorder:
    """A backend that records the request instead of running anything."""

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    def run(self, request):
        from vaft.code.execution import ExecutionResult

        self.requests.append(request)
        return ExecutionResult(returncode=0, stdout="", stderr="")


def test_the_gpec_suite_launcher_goes_through_the_configured_backend(tmp_path):
    from vaft.code.gpec import GPECSuiteConfig
    from vaft.code.gpec import _runtime as runtime

    recorder = _Recorder()
    work = tmp_path / "run"
    work.mkdir()
    returncode, log = runtime.run_subprocess(
        tmp_path / "bin" / "dcon",
        work,
        work / "dcon.log",
        config=GPECSuiteConfig(backend=recorder, timeout=60),
    )
    assert (returncode, log) == (0, work / "dcon.log")
    (request,) = recorder.requests
    assert request.command == (str(tmp_path / "bin" / "dcon"),)


def test_the_flare_launcher_goes_through_the_configured_backend(tmp_path):
    from vaft.code.flare import FlareConfig, run_flare

    recorder = _Recorder()
    work = tmp_path / "flare"
    work.mkdir()
    result = run_flare(
        "poincare_plot",
        config=FlareConfig(
            backend=recorder, workdir=work, executable=tmp_path / "bin" / "flare", processes=2
        ),
    )
    assert result.returncode == 0
    (request,) = recorder.requests
    assert request.command[:4] == (
        str(tmp_path / "bin" / "flare"), "-n", "2", "poincare_plot",
    )
    assert request.resources.ntasks == 2


@pytest.mark.parametrize(
    "module",
    ["vaft/code/flare.py", "vaft/code/gpec", "vaft/code/pentrc"],
)
def test_no_heavy_adapter_starts_its_own_process(module):
    """``run_pentrc`` shares the GPEC chokepoint; nothing here may bypass it.

    Checked on the source rather than at runtime, because a bypass added to
    one branch of one adapter would otherwise only show up on a cluster.
    """
    root = Path(__file__).resolve().parent.parent / module
    sources = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    assert sources, f"{module} has no Python sources to check"
    offenders = [
        f"{source.name}:{number}: {line.strip()}"
        for source in sources
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
        if any(
            call in line
            for call in ("subprocess.run(", "subprocess.Popen(", "subprocess.check_call(",
                         "subprocess.check_output(", "os.system(", "os.execv")
        )
    ]
    assert offenders == [], (
        "these launch a program without the execution backend, so a remote or "
        f"Slurm backend would never see them: {offenders}"
    )


# -- opt-in, against a real cluster ------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("VAFT_REMOTE_LIVE_TEST"),
    reason="set VAFT_REMOTE_LIVE_TEST=1 and the VAFT_REMOTE_* variables to submit for real",
)
def test_live_submission_against_a_real_host(tmp_path):
    """Submit a trivial job to the host ``VAFT_REMOTE_*`` describes.

    Opt-in, and never run in CI: it needs an ssh-reachable Slurm cluster, a
    key the agent already holds, and an account that may be charged. Set
    ``VAFT_SLURM_PARTITION`` as the site requires.
    """
    work = tmp_path / "live"
    work.mkdir()
    backend = RemoteSlurmBackend.from_environment(
        {**os.environ, "VAFT_SLURM_MAX_WAIT": os.environ.get("VAFT_SLURM_MAX_WAIT", "900")}
    )
    result = backend.run(
        _shell(work, "echo live > produced.txt", timeout=120, log_path=work / "live.log", label="live")
    )
    assert result.returncode == 0, result.stderr or (work / "live.log").read_text()
    assert (work / "produced.txt").read_text().strip() == "live"
    assert result.run_s is not None and result.run_s >= 0.0
