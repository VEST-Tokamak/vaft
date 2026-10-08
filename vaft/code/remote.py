"""Run external codes on a remote Slurm cluster over ssh.

:class:`RemoteSlurmBackend` is a drop-in
:class:`~vaft.code.execution.ExecutionBackend` for the case where the machine
preparing a run is not the machine that can run it: a laptop that holds the
inputs and a cluster that holds the build. ``GPECSuiteConfig(backend=
RemoteSlurmBackend(host))`` runs the suite there and returns the same result a
local run would, so no adapter changes.

What one :meth:`RemoteSlurmBackend.run` does:

1. ``rsync`` the request's working directory up to the host.
2. Write the same batch script :mod:`vaft.code.slurm` writes, with the host's
   paths and its setup lines (``module load ...``) in front.
3. ``ssh <host> sbatch --parsable --no-requeue`` it.
4. Poll ``squeue``, then ``sacct``, over ssh with a growing interval.
5. ``rsync`` back what the fetch policy names, then report.

The contract in :mod:`vaft.code.execution` is kept whole: ``run`` blocks until
the job is done; a time limit or a queue wait that ran out comes back as a
result (``timed_out=True``, ``runtime_status="timeout"`` or
``"queue_timeout"``), never as an exception; ``KeyboardInterrupt``, ``SIGTERM``
and ``SIGHUP`` ``scancel`` the job and are then re-raised unchanged; a
submission the host refuses raises
:class:`~vaft.code._executables.ExecutableNotLaunchable`; and ``run_s`` is how
long the job itself ran, from the ``sacct`` start and end stamps -- or, where
accounting is not configured, from the job script's own stamps on the node --
so the queue wait that ``elapsed_s`` includes is left out of it.

Things only this backend has to say:

* **Paths.** The two machines do not share a filesystem, so every path that
  travels has to be rewritten. :class:`RemoteHost` takes ordered
  ``local -> remote`` prefix pairs and applies the first one that matches at a
  path boundary. The working directory, ``log_path``, each argument of
  ``command`` and each path-valued entry of ``env`` go through them, which is
  how a local ``GPECHOME`` reaches the job as the host's own build. A working
  directory no pair matches is staged into a fresh directory under the host's
  work root instead.
* **Environment.** Only the part of ``ExecutionRequest.env`` that differs from
  the submitting process is written into the job script, as for a local Slurm
  submission -- but ``--export=ALL`` there carries the *host's* ssh environment,
  not the laptop's, and that is the point: the job's search paths come from the
  host's login environment and from ``setup_lines``. Variables whose value only
  means something on the submitting machine are therefore dropped before the
  comparison (:data:`UNFORWARDED_ENVIRONMENT`): a macOS ``PATH`` or
  ``DYLD_LIBRARY_PATH`` reaching a Linux compute node would shadow the modules
  the job just loaded.
* **What comes back.** The job log and the scratch directory always do -- they
  are the run's record, and the exit status is in them. The fetch policy
  (``"all"``, a list of glob patterns, or ``"none"``) governs the rest of the
  working directory, so a run whose outputs are tens of GB can leave them on
  the cluster and bring back an extract.
* **What stays.** The remote copy is kept (``keep_remote=True``), because it is
  usually the only full copy of the outputs. ``keep_remote=False`` removes it
  after the fetch, and then only a directory this backend created under the
  work root -- never one a path pair pointed at. A removal that fails is not an
  error; the copy simply stays, which is the default anyway.
* **Credentials.** None are handled here. ``ssh`` always runs with
  ``BatchMode=yes``, so a host that would prompt fails instead of hanging; keys
  and agents are the caller's business. No host name, user or path of any
  particular cluster belongs in this module or its tests.
* **One connection per step.** Every ssh and rsync call stands on its own, so a
  dropped connection loses a poll rather than the job. A submission is retried
  only where retrying cannot duplicate a job: on ssh's own exit status 255
  together with ssh's own wording for a connection it never made, or on
  ``Unable to contact slurm controller`` from a command that did run. Anything
  else -- including a rejection whose text merely mentions a connection, as a
  site login banner sharing that stream might -- is reported at once. A failure
  that will not pass, such as a refused key, is reported rather than retried.
* **An interrupt fetches nothing.** ``KeyboardInterrupt``, ``SIGTERM`` and
  ``SIGHUP`` cancel the job and propagate at once, so the outputs a cancelled
  job did write stay on the host rather than delaying the exit by a transfer.
  They are where ``keep_remote`` left them, under the name in the error's
  traceback.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from . import _process_tree
from ._executables import ExecutableNotLaunchable
from .execution import RUNTIME_QUEUE_TIMEOUT, ExecutionRequest, ExecutionResult
from .slurm import (
    LIVE_STATES,
    SCRATCH_DIRECTORY,
    _TERMINATED,
    _job_note,
    _job_outcome,
    _job_script,
    _parse_accounting,
    _read,
    _scratch_name,
    _slurm_path,
    walltime,
)

#: Fetch everything the run left in the working directory.
FETCH_ALL = "all"
#: Fetch nothing beyond the log and the scratch directory.
FETCH_NONE = "none"

#: Environment variables never forwarded to a remote host. Their value
#: describes the submitting machine -- its interpreter, its libraries, its
#: home, its locale, its loaded modules -- and on the compute node it would
#: either mean nothing or shadow what ``setup_lines`` just set up. An entry
#: ending in ``_`` is a prefix; the rest are exact names.
#: ``SLURM_*``/``SBATCH_*``/``SRUN_*`` are already dropped by
#: :func:`vaft.code.slurm.exported_environment`.
UNFORWARDED_ENVIRONMENT: tuple[str, ...] = (
    "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_RUN_PATH", "LIBRARY_PATH",
    "DYLD_", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "PKG_CONFIG_PATH",
    "PYTHONPATH", "PYTHONHOME", "PYTHONEXECUTABLE", "VIRTUAL_ENV", "CONDA_",
    "MANPATH", "INFOPATH", "HOME", "USER", "LOGNAME", "SHELL", "SHLVL",
    "PWD", "OLDPWD", "TMPDIR", "TMP", "TEMP", "TERM", "DISPLAY", "XDG_",
    "SSH_", "LANG", "LC_", "LOADEDMODULES", "MODULEPATH", "MODULESHOME",
    "BASH_FUNC_",
)
_UNFORWARDED_PREFIXES = tuple(name for name in UNFORWARDED_ENVIRONMENT if name.endswith("_"))
_UNFORWARDED_NAMES = frozenset(name for name in UNFORWARDED_ENVIRONMENT if not name.endswith("_"))

#: ssh's own words for a connection it never established. Taken together with
#: ssh's exit status 255 they mean nothing ran on the host, so a retry cannot
#: leave a duplicate job running.
_SSH_UNREACHED = (
    "Connection refused",
    "Connection timed out",
    "Connection closed by remote host",
    "Could not resolve hostname",
    "kex_exchange_identification",
    "Operation timed out",
    "No route to host",
)
#: An sbatch error that means the controller was never reached, so the job
#: cannot have been accepted. Only this one counts for a command that did run,
#: because a site login banner lands in the same stream and must not be able
#: to turn a plain rejection into a retry.
_CONTROLLER_UNREACHED = ("Unable to contact slurm controller",)
#: Consecutive unanswered polls before accounting is asked instead.
_UNKNOWN_POLLS = 10
#: ssh's own exit status for a transport failure, as opposed to the remote
#: command's status.
_SSH_TRANSPORT_FAILURE = 255
#: Slurm states in which the program itself is running.
_RUNNING_STATES = frozenset({"RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT", "SIGNALING"})
#: Errors from ``sacct`` or from the link to it worth retrying within
#: ``accounting_grace``.
_ACCOUNTING_TRANSIENT = (
    "Socket timed out", "Unable to contact", "slurmdbd", "Connection refused",
    "Connection closed", "Operation timed out", "Connection timed out",
)


def _boundary(text: str, prefix: str) -> bool:
    """Whether ``text`` is ``prefix`` or lies under it.

    A prefix test on the raw strings would make ``/data/runs2`` a child of
    ``/data/runs``; only a match at a path separator counts.
    """
    trimmed = prefix.rstrip("/") or "/"
    return text == trimmed or text.startswith(trimmed if trimmed == "/" else trimmed + "/")


@dataclass(frozen=True)
class RemoteHost:
    """An ssh-reachable host that runs Slurm, and how its paths line up with ours.

    Parameters
    ----------
    host : str
        Host name or address for ssh [n/a].
    work_root : str
        Absolute directory on the host under which a working directory that no
        ``path_maps`` pair covers is staged [n/a].
    user, port, identity_file : optional
        ssh login name, port and private key. No password is ever supplied:
        ``BatchMode=yes`` is always passed, so a host that would prompt fails
        at once instead of blocking a batch [n/a].
    ssh_options : sequence of str
        Further ssh arguments, verbatim and in order (for example
        ``("-o", "StrictHostKeyChecking=accept-new")``) [n/a].
    path_maps : sequence of (str, str)
        Ordered ``(local prefix, remote prefix)`` pairs, both absolute. The
        first pair that matches a path at a boundary is used, so the caller
        controls precedence by order [n/a].
    setup_lines : sequence of str
        Shell lines placed at the top of the job script, before anything else
        runs: ``module load``, a ``source`` of a build's environment [n/a].
    fetch : "all" | "none" | sequence of str
        What to bring back from the working directory: everything, nothing, or
        the files matching these rsync glob patterns. The job log and the
        scratch directory always come back regardless [n/a].
    keep_remote : bool
        Keep the host's copy of the working directory after the fetch. Default
        true: it is normally the only complete copy. Setting it false removes
        only a directory staged under ``work_root`` [n/a].
    rsync_options : sequence of str
        Further rsync arguments for both directions, verbatim [n/a].
    ssh_executable, rsync_executable : str
        Programs to run locally; looked up on ``PATH`` [n/a].

    Notes
    -----
    Applicability: Machine-independent. Nothing here is specific to one
    cluster; a site's host name, account and module lines are configuration,
    not code.
    """

    host: str
    work_root: str
    user: Optional[str] = None
    port: Optional[int] = None
    identity_file: Optional[str] = None
    ssh_options: tuple[str, ...] = ()
    path_maps: tuple[tuple[str, str], ...] = ()
    setup_lines: tuple[str, ...] = ()
    fetch: Union[str, tuple[str, ...]] = FETCH_ALL
    keep_remote: bool = True
    rsync_options: tuple[str, ...] = ()
    ssh_executable: str = "ssh"
    rsync_executable: str = "rsync"

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError("host must be a non-empty host name")
        if not str(self.work_root).startswith("/"):
            raise ValueError(f"work_root must be an absolute path, got {self.work_root!r}")
        if self.port is not None and not 1 <= int(self.port) <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        maps = tuple((str(local), str(remote)) for local, remote in self.path_maps)
        for local, remote in maps:
            if not Path(local).is_absolute() or not remote.startswith("/"):
                raise ValueError(
                    f"path_maps needs absolute local and remote prefixes, got {(local, remote)!r}"
                )
        object.__setattr__(self, "path_maps", maps)
        object.__setattr__(self, "ssh_options", tuple(str(arg) for arg in self.ssh_options))
        object.__setattr__(self, "setup_lines", tuple(str(line) for line in self.setup_lines))
        object.__setattr__(self, "rsync_options", tuple(str(arg) for arg in self.rsync_options))
        if isinstance(self.fetch, str):
            policy: Union[str, tuple[str, ...]] = self.fetch.strip().lower()
            if policy not in (FETCH_ALL, FETCH_NONE):
                raise ValueError(
                    f"fetch must be {FETCH_ALL!r}, {FETCH_NONE!r} or a sequence of "
                    f"glob patterns, got {self.fetch!r}"
                )
        else:
            policy = tuple(str(pattern) for pattern in self.fetch)
            if not policy:
                raise ValueError(f"an empty fetch pattern list says nothing; use {FETCH_NONE!r}")
        object.__setattr__(self, "fetch", policy)

    def __repr__(self) -> str:
        # No credential, and the host only as the caller gave it.
        return (
            f"RemoteHost(host={self.host!r}, work_root={self.work_root!r}, "
            f"path_maps={len(self.path_maps)}, fetch={self.fetch!r})"
        )

    # -- ssh / rsync ------------------------------------------------------

    @property
    def target(self) -> str:
        """``user@host`` or just ``host``: the ssh and rsync destination."""
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_prefix(self) -> list[str]:
        """The ssh command up to but not including the destination.

        ``BatchMode=yes`` comes first so a caller's ``ssh_options`` can still
        override later settings but cannot turn prompting back on by accident
        (ssh takes the first value given for an option). ``LogLevel=ERROR``
        keeps ssh's own informational chatter -- key-exchange notices, host-key
        additions -- out of the scheduler's stderr, where it would otherwise be
        quoted back as if it were the reason a submission failed. Real
        diagnostics ("Connection refused", "Could not resolve hostname") are
        errors and still come through. A site's own login banner is sent during
        authentication and is not ssh's to suppress, so it may still prefix an
        error message.
        """
        argv = [_tool(self.ssh_executable), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
        if self.port is not None:
            argv += ["-p", str(int(self.port))]
        if self.identity_file:
            argv += ["-i", str(Path(self.identity_file).expanduser())]
        argv += list(self.ssh_options)
        return argv

    def ssh_argv(self, remote_command: str) -> list[str]:
        """Full argv to run one shell command on the host."""
        return [*self.ssh_prefix(), self.target, remote_command]

    def rsync_prefix(self) -> list[str]:
        """The rsync command up to but not including the paths.

        ``-a`` so modes and times survive (the job script must stay
        executable), and ``-e`` so the transfer uses exactly the ssh settings
        the control connection uses.
        """
        return [
            _tool(self.rsync_executable),
            "-a",
            "-e",
            shlex.join(self.ssh_prefix()),
            *self.rsync_options,
        ]

    def remote(self, path: str) -> str:
        """``path`` as the host sees it; return a destination string for rsync."""
        return f"{self.target}:{path}"

    # -- path translation -------------------------------------------------

    def remote_path(self, local: Any) -> Optional[str]:
        """``local`` under the first matching ``path_maps`` pair, else ``None``."""
        text = str(local)
        for prefix, remote in self.path_maps:
            if _boundary(text, prefix):
                tail = text[len(prefix.rstrip("/")) :].lstrip("/")
                return posixpath.join(remote.rstrip("/") or "/", tail) if tail else (remote.rstrip("/") or "/")
        return None

    def translate(self, text: Any) -> str:
        """One command argument with any mapped path prefix rewritten.

        A bare path is rewritten whole. An option that carries its path after
        an ``=`` (``--home=/opt/gpec``) has the path alone rewritten, because
        the argument as a whole matches no prefix and would otherwise reach
        the compute node naming a directory that does not exist there. An
        argument that is not a mapped path is returned unchanged, so task
        names, namelist keys and numbers pass through.
        """
        value = str(text)
        mapped = self.remote_path(value)
        if mapped is not None:
            return mapped
        key, separator, tail = value.partition("=")
        if separator and not key.startswith("/"):
            mapped = self.remote_path(tail)
            if mapped is not None:
                return f"{key}={mapped}"
        return value

    def translate_value(self, text: Any) -> str:
        """One environment value, treating it as a ``:``-separated path list.

        ``GPECHOME`` is a single path and ``SOMETHING_PATH`` may be several;
        each segment is translated on its own and segments that match no pair
        are left alone, so a value that is not a path survives untouched.
        """
        value = str(text)
        if ":" not in value:
            return self.translate(value)
        return ":".join(self.translate(segment) for segment in value.split(":"))

    def forwarded_environment(self, env: Mapping[str, str]) -> dict[str, str]:
        """``env`` with submitting-machine-only names dropped and paths translated.

        See :data:`UNFORWARDED_ENVIRONMENT` for what is dropped and why.
        """
        return {
            str(key): self.translate_value(value)
            for key, value in env.items()
            if str(key) not in _UNFORWARDED_NAMES and not str(key).startswith(_UNFORWARDED_PREFIXES)
        }

    # -- construction -----------------------------------------------------

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "RemoteHost":
        """Build a host from ``VAFT_REMOTE_*`` variables.

        ``VAFT_REMOTE_HOST`` and ``VAFT_REMOTE_WORK_ROOT`` are required; the
        rest keep the constructor defaults.

        ==============================  ===================================
        variable                        value
        ==============================  ===================================
        ``VAFT_REMOTE_HOST``            host name
        ``VAFT_REMOTE_WORK_ROOT``       absolute staging directory
        ``VAFT_REMOTE_USER``            login name
        ``VAFT_REMOTE_PORT``            ssh port
        ``VAFT_REMOTE_IDENTITY``        private key file
        ``VAFT_REMOTE_SSH_OPTIONS``     further ssh arguments, shell-quoted
        ``VAFT_REMOTE_PATH_MAPS``       ``local=remote`` pairs, comma separated
        ``VAFT_REMOTE_SETUP``           job-script lines, ``;`` separated
        ``VAFT_REMOTE_FETCH``           ``all``, ``none``, or comma-separated globs
        ``VAFT_REMOTE_KEEP``            ``0`` to delete the host's copy
        ==============================  ===================================
        """
        env = os.environ if environ is None else environ
        host = (env.get("VAFT_REMOTE_HOST") or "").strip()
        work_root = (env.get("VAFT_REMOTE_WORK_ROOT") or "").strip()
        missing = [
            name
            for name, value in (("VAFT_REMOTE_HOST", host), ("VAFT_REMOTE_WORK_ROOT", work_root))
            if not value
        ]
        if missing:
            raise ValueError(f"the remote backend needs {' and '.join(missing)} to be set")
        port = (env.get("VAFT_REMOTE_PORT") or "").strip()
        maps = []
        for pair in (env.get("VAFT_REMOTE_PATH_MAPS") or "").split(","):
            if pair.strip():
                local, separator, remote = pair.partition("=")
                if not separator or not remote.strip():
                    raise ValueError(
                        f"VAFT_REMOTE_PATH_MAPS entries are 'local=remote', got {pair.strip()!r}"
                    )
                maps.append((local.strip(), remote.strip()))
        fetch_text = (env.get("VAFT_REMOTE_FETCH") or FETCH_ALL).strip()
        fetch: Union[str, tuple[str, ...]]
        if fetch_text.lower() in (FETCH_ALL, FETCH_NONE):
            fetch = fetch_text.lower()
        else:
            fetch = tuple(part.strip() for part in fetch_text.split(",") if part.strip())
        keep = (env.get("VAFT_REMOTE_KEEP") or "1").strip().lower()
        return cls(
            host=host,
            work_root=work_root,
            user=(env.get("VAFT_REMOTE_USER") or "").strip() or None,
            port=int(port) if port else None,
            identity_file=(env.get("VAFT_REMOTE_IDENTITY") or "").strip() or None,
            ssh_options=tuple(shlex.split(env.get("VAFT_REMOTE_SSH_OPTIONS") or "")),
            path_maps=tuple(maps),
            setup_lines=tuple(
                line.strip() for line in (env.get("VAFT_REMOTE_SETUP") or "").split(";") if line.strip()
            ),
            fetch=fetch,
            keep_remote=keep not in ("0", "false", "no", "off"),
        )


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise ExecutableNotLaunchable(f"cannot launch on a remote host: {name!r} is not on PATH")
    return found


class RemoteSlurmBackend:
    """Run each request as a batch job on an ssh-reachable Slurm cluster.

    Parameters
    ----------
    host : RemoteHost
        Where to run, and how that host's paths line up with this machine's.
    partition, account, qos : str, optional
        ``sbatch`` placement options [n/a].
    extra_args : sequence of str
        Further ``sbatch`` options, verbatim. Job arrays and heterogeneous jobs
        are refused: one request is one status [n/a].
    max_wait : float, optional
        Upper bound on the time the job may spend queued or running before it
        is cancelled and the result becomes a timeout [s]. The cancel itself
        lands on time -- the poll interval is shortened so as not to sleep
        past the limit -- but ``run`` then blocks a little longer for the
        cancel to be confirmed, the files to come back and accounting to
        settle, so ``elapsed_s`` exceeds ``max_wait``.
    poll_interval : float
        Interval of the first poll [s].
    max_poll_interval : float
        Ceiling the interval backs off to, so a day-long job costs a few
        hundred ssh round trips rather than thousands [s].
    backoff : float
        Factor the interval grows by after each poll [n/a].
    keep_scratch : bool
        Keep the local scratch copy (job script, streams, status) even after a
        clean exit [n/a].
    """

    #: How long a finished job's status file and accounting may lag [s].
    status_grace: float = 15.0
    accounting_grace: float = 30.0
    #: After cancelling, how long to wait for the job to leave the queue [s].
    cancel_grace: float = 60.0

    def __init__(
        self,
        host: RemoteHost,
        *,
        partition: Optional[str] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
        extra_args: Sequence[str] = (),
        max_wait: Optional[float] = None,
        poll_interval: float = 10.0,
        max_poll_interval: float = 60.0,
        backoff: float = 1.5,
        keep_scratch: bool = False,
    ) -> None:
        if not isinstance(host, RemoteHost):
            raise TypeError(f"host must be a RemoteHost, got {type(host).__name__}")
        if max_wait is not None and max_wait <= 0:
            raise ValueError(f"max_wait must be > 0 or None, got {max_wait}")
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if max_poll_interval < poll_interval:
            raise ValueError(
                f"max_poll_interval must be >= poll_interval, got {max_poll_interval} < {poll_interval}"
            )
        if backoff < 1.0:
            raise ValueError(f"backoff must be >= 1, got {backoff}")
        extra = tuple(str(arg) for arg in extra_args)
        refused = [arg for arg in extra if arg.startswith(("--array", "-a", "--het", ":"))]
        if refused:
            raise ValueError(f"job arrays and heterogeneous jobs are not supported: {refused}")
        self.host = host
        self.partition = partition
        self.account = account
        self.qos = qos
        self.extra_args = extra
        self.max_wait = max_wait
        self.poll_interval = float(poll_interval)
        self.max_poll_interval = float(max_poll_interval)
        self.backoff = float(backoff)
        self.keep_scratch = keep_scratch

    @classmethod
    def from_environment(cls, environ: Optional[Mapping[str, str]] = None) -> "RemoteSlurmBackend":
        """Build a backend from ``VAFT_REMOTE_*`` and ``VAFT_SLURM_*`` variables.

        The host comes from :meth:`RemoteHost.from_environment`; placement and
        ``max_wait`` come from the same ``VAFT_SLURM_PARTITION``,
        ``VAFT_SLURM_ACCOUNT``, ``VAFT_SLURM_QOS`` and ``VAFT_SLURM_MAX_WAIT``
        a local Slurm submission reads, so moving a pipeline from the cluster's
        own login node to ssh changes only ``VAFT_EXECUTION_BACKEND`` and the
        ``VAFT_REMOTE_*`` block.
        """
        env = os.environ if environ is None else environ
        max_wait = env.get("VAFT_SLURM_MAX_WAIT")
        return cls(
            RemoteHost.from_environment(env),
            partition=env.get("VAFT_SLURM_PARTITION") or None,
            account=env.get("VAFT_SLURM_ACCOUNT") or None,
            qos=env.get("VAFT_SLURM_QOS") or None,
            max_wait=float(max_wait) if max_wait else None,
        )

    def __repr__(self) -> str:
        return (
            f"RemoteSlurmBackend({self.host!r}, partition={self.partition!r}, "
            f"account={self.account!r}, qos={self.qos!r})"
        )

    # -- the run ----------------------------------------------------------

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        local_work = Path(request.workdir)
        if not local_work.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {request.workdir}")
        local_work = local_work.resolve()

        host = self.host
        name = _scratch_name(request.label)  # refuses a label that is not one path component
        mapped = host.remote_path(local_work)
        remote_work = mapped or posixpath.join(host.work_root.rstrip("/"), name)
        local_scratch = local_work / SCRATCH_DIRECTORY / name
        local_scratch.mkdir(parents=True)
        remote_scratch = posixpath.join(remote_work, SCRATCH_DIRECTORY, name)

        # The request as the compute node will see it.
        remote_request = replace(
            request,
            command=tuple(host.translate(part) for part in request.command),
            env=host.forwarded_environment(request.env),
        )
        remote_stdin = None
        if request.stdin is not None:
            (local_scratch / "stdin").write_text(request.stdin, encoding="utf-8")
            remote_stdin = posixpath.join(remote_scratch, "stdin")
        script = local_scratch / "job.sh"
        script.write_text(
            _job_script(
                remote_request,
                remote_work,
                remote_stdin,
                remote_scratch,
                setup_lines=host.setup_lines,
            ),
            encoding="utf-8",
        )
        script.chmod(0o700)

        local_log = Path(request.log_path).resolve() if request.log_path is not None else None
        remote_log = self._remote_log(local_log, local_work, remote_work, remote_scratch)
        if remote_log is not None:
            streams = [f"--output={_slurm_path(remote_log)}"]  # stderr joins it without --error
        else:
            streams = [
                f"--output={_slurm_path(posixpath.join(remote_scratch, 'stdout'))}",
                f"--error={_slurm_path(posixpath.join(remote_scratch, 'stderr'))}",
            ]
        argv = [
            "sbatch", "--parsable", "--no-requeue", "--export=ALL",
            f"--job-name=vaft-{request.label}" if request.label else "--job-name=vaft",
            f"--chdir={remote_work}", *streams,
            *self._sizing(request), *self._placement(), *self.extra_args,
            posixpath.join(remote_scratch, "job.sh"),
        ]

        if local_log is not None:
            # Emptied before anything is sent, as a local launch empties it
            # before the program starts. The same working directory is reused
            # across attempts, so an earlier run's output left at this path
            # would travel up with the inputs and come straight back as if
            # this run had written it -- and a job cancelled in the queue
            # writes nothing of its own to overwrite it.
            local_log.parent.mkdir(parents=True, exist_ok=True)
            local_log.write_text("", encoding="utf-8")

        started = time.monotonic()
        needed = [remote_work]
        if remote_log is not None:
            # sbatch does not create the --output directory, and a caller whose
            # log sits in a directory the working directory does not have yet
            # would otherwise lose the whole run to it.
            needed.append(posixpath.dirname(remote_log))
        self._push(local_work, remote_work, needed)
        job_id, cluster = self._submit(argv)
        # Every later scheduler call has to name the federation member that
        # accepted the job, or it asks the wrong controller. Carried as an
        # argument, not on ``self``: one backend may serve several runs.
        clusters = ["-M", cluster] if cluster else []
        began = False
        try:
            # SIGTERM and SIGHUP have to cancel the job, like Ctrl-C; without
            # the guard the default disposition ends Python and leaves the job
            # running on the cluster.
            with _process_tree.forward_termination():
                try:
                    cancelled, began = self._wait(job_id, clusters, started)
                except BaseException:
                    self._cancel(job_id, clusters)
                    raise
        except _process_tree.Terminated as stop:
            stop.redeliver()
            raise

        self._fetch(local_work, remote_work, local_scratch, remote_scratch, local_log, remote_log)
        # The script stamps ``started`` as the program begins, so a cancelled
        # job without it never left the queue and can have no exit record to
        # wait ``status_grace`` for.
        begin = (_read(local_scratch / "started") or "").strip()
        began = began or bool(begin)
        recorded = (
            None
            if (cancelled and not began)
            else self._recorded(local_scratch / "returncode", remote_scratch)
        )
        state, exit_code, scheduler_run_s = self._accounting(job_id, clusters)

        returncode, timed_out, never_started, cancelled = _job_outcome(
            cancelled=cancelled,
            recorded=recorded,
            state=state,
            exit_code=exit_code,
            began=began,
            walltime_kill=self._walltime_kill(recorded or "", begin, request),
        )
        run_s = scheduler_run_s
        if run_s is None and recorded is not None and recorded.startswith(_TERMINATED) and begin.isdigit():
            # No accounting: fall back to the node's own clock at the start and
            # at the TERM that stopped the program.
            fields = recorded.split()
            if len(fields) >= 2 and fields[1].isdigit():
                run_s = float(max(int(fields[1]) - int(begin), 0))

        note = _job_note(
            job_id,
            max_wait=self.max_wait,
            never_started=never_started,
            cancelled=cancelled,
            state=state,
            recorded=recorded,
            timed_out=timed_out,
        )
        if local_log is not None:
            stdout = stderr = ""
            local_log.parent.mkdir(parents=True, exist_ok=True)
            if note:
                with local_log.open("a", encoding="utf-8") as log:
                    log.write(f"\n{note}\n")
        else:
            stdout = _read(local_scratch / "stdout") or ""
            stderr = _read(local_scratch / "stderr") or ""
            if note:
                stderr = f"{stderr}\n{note}" if stderr else note

        if not host.keep_remote and mapped is None:
            # Only a directory this run staged under the work root; a path pair
            # points at the caller's own tree and is never removed.
            self._ssh(f"rm -rf -- {shlex.quote(remote_work)}")
        if returncode == 0 and not self.keep_scratch:
            shutil.rmtree(local_scratch, ignore_errors=True)
            try:
                local_scratch.parent.rmdir()
            except OSError:
                pass
        return ExecutionResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            elapsed_s=time.monotonic() - started,
            launcher=tuple(argv),
            log_path=local_log,
            job_id=job_id,
            runtime_status=RUNTIME_QUEUE_TIMEOUT if never_started else "",
            run_s=run_s,
        )

    # -- sbatch options ---------------------------------------------------

    def _sizing(self, request: ExecutionRequest) -> list[str]:
        resources = request.resources
        cpus = int(resources.ntasks) * int(resources.threads_per_task or 1)
        options = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}"]
        if resources.memory_mb is not None:
            options.append(f"--mem={int(resources.memory_mb)}M")
        if request.timeout is not None:
            options.append(f"--time={walltime(request.timeout)}")
        return options

    def _placement(self) -> list[str]:
        return [
            f"--{name}={value}"
            for name, value in (("partition", self.partition), ("account", self.account), ("qos", self.qos))
            if value
        ]

    def _remote_log(
        self, local_log: Optional[Path], local_work: Path, remote_work: str, remote_scratch: str
    ) -> Optional[str]:
        """Where the job writes its log, given where the caller wants it locally.

        A log inside the working directory keeps its place there; one outside
        it, and outside every path pair, has nowhere of its own on the host and
        is written into the scratch directory, from where the fetch puts it
        back where the caller asked.
        """
        if local_log is None:
            return None
        mapped = self.host.remote_path(local_log)
        if mapped is not None:
            return mapped
        if _boundary(str(local_log), str(local_work)):
            tail = str(local_log)[len(str(local_work)) :].lstrip("/")
            return posixpath.join(remote_work, tail)
        return posixpath.join(remote_scratch, local_log.name)

    # -- transfers --------------------------------------------------------

    def _rsync(self, argv: list[str], what: str, *, raises: Optional[type] = RuntimeError) -> bool:
        """One rsync; ``False`` if it failed and ``raises`` is ``None``.

        ``raises`` is the error for a transfer that must work:
        :class:`~vaft.code._executables.ExecutableNotLaunchable` before the
        submission, since the program then never starts, and a
        ``RuntimeError`` afterwards, since by then the job has run and only the
        report is lost. ``None`` is for the job log, which a job cancelled in
        the queue never created -- that has to come back as a
        ``queue_timeout`` result, not as an exception.
        """
        done = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        if done.returncode == 0:
            return True
        if raises is None:
            return False
        message = (done.stderr or done.stdout).strip()
        raise raises(f"cannot {what} over rsync: {message}")

    def _push(self, local_work: Path, remote_work: str, directories: Sequence[str]) -> None:
        host = self.host
        # rsync creates the destination but not its parents, and neither rsync
        # nor sbatch creates the directory --output points at.
        made = self._ssh(
            shlex.join(["mkdir", "-p", "--", *dict.fromkeys(directories)])
        )
        if made.returncode != 0:
            message = (made.stderr or made.stdout).strip()
            raise ExecutableNotLaunchable(
                f"cannot prepare the remote working directory: {message}"
            )
        self._rsync(
            [*host.rsync_prefix(), f"{local_work}/", host.remote(f"{remote_work}/")],
            "send the working directory",
            raises=ExecutableNotLaunchable,
        )

    def _fetch(
        self,
        local_work: Path,
        remote_work: str,
        local_scratch: Path,
        remote_scratch: str,
        local_log: Optional[Path],
        remote_log: Optional[str],
    ) -> None:
        """Bring back the record of the run, then whatever the policy names.

        The scratch directory and the log come back first and always: they hold
        the exit status, and reporting on a run without them is guesswork. The
        log is the one optional transfer: a job cancelled in the queue wrote
        none, and ``run`` has already emptied the local file, so there is
        nothing to bring back and nothing stale left behind.
        """
        host = self.host
        prefix = host.rsync_prefix()
        self._rsync(
            [*prefix, host.remote(f"{remote_scratch}/"), f"{local_scratch}/"],
            "fetch the job's status files",
        )
        if remote_log is not None and local_log is not None:
            local_log.parent.mkdir(parents=True, exist_ok=True)
            self._rsync(
                [*prefix, host.remote(remote_log), str(local_log)],
                "fetch the job log",
                raises=None,
            )
        if host.fetch == FETCH_NONE:
            return
        patterns = () if host.fetch == FETCH_ALL else tuple(host.fetch)
        rules: list[str] = []
        if patterns:
            # Descend into every directory, keep what matches, drop the rest,
            # then prune the directories that kept nothing (-m).
            rules = ["-m", "--include=*/", *[f"--include={pattern}" for pattern in patterns], "--exclude=*"]
        self._rsync(
            [*prefix, *rules, host.remote(f"{remote_work}/"), f"{local_work}/"],
            "fetch the working directory",
        )

    # -- scheduler over ssh -----------------------------------------------

    def _ssh(self, remote_command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.host.ssh_argv(remote_command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def _submit(self, argv: list[str]) -> tuple[str, str]:
        """``(job id, cluster)``; the cluster is empty off a federation."""
        command = shlex.join(argv)
        for attempt in range(3):
            submitted = self._ssh(command)
            if submitted.returncode == 0:
                # Site wrappers and login banners print first; the id is last.
                # ``--parsable`` prints ``<id>;<cluster>`` on a federation.
                lines = [line.strip() for line in submitted.stdout.splitlines() if line.strip()]
                job_id, _, cluster = (lines[-1] if lines else "").partition(";")
                if not job_id.isdigit():
                    raise ExecutableNotLaunchable(
                        f"sbatch accepted {argv[-1]} but printed no job id: {submitted.stdout!r}"
                    )
                return job_id, cluster.split()[0] if cluster.strip() else ""
            message = (submitted.stderr or submitted.stdout).strip()
            transport = submitted.returncode == _SSH_TRANSPORT_FAILURE
            markers = _SSH_UNREACHED if transport else _CONTROLLER_UNREACHED
            if attempt == 2 or not any(marker in message for marker in markers):
                what = "the remote host could not be reached" if transport else "sbatch rejected the job"
                raise ExecutableNotLaunchable(f"{what} for {argv[-1]}: {message}")
            time.sleep(self.poll_interval)
        raise AssertionError("unreachable")  # pragma: no cover

    def _poll(self, job_id: str, clusters: Sequence[str]) -> tuple[Optional[str], Optional[bool]]:
        """``(state, live)`` from ``squeue``; ``live`` is ``None`` if unanswered."""
        listed = self._ssh(shlex.join(["squeue", *clusters, "-h", "-j", job_id, "-o", "%T"]))
        self._last_poll_error = (listed.stderr or listed.stdout).strip()
        if listed.returncode != 0:
            # Only a purged job is gone; a controller or a link that did not
            # answer is not.
            return None, (False if "Invalid job id" in listed.stderr else None)
        states = [line.strip().split()[0] for line in listed.stdout.splitlines() if line.strip()]
        state = states[0] if states else None
        return state, bool(set(states) & LIVE_STATES)

    def _wait(self, job_id: str, clusters: Sequence[str], started: float) -> tuple[bool, bool]:
        """Block until the job is done.

        Returns ``(cancelled, began)``: whether ``max_wait`` cancelled it, and
        whether the program was ever seen running -- which is what separates a
        queue timeout from a run that was stopped.
        """
        cancelled_at: Optional[float] = None
        began = False
        unknown = 0
        interval = self.poll_interval
        while True:
            state, live = self._poll(job_id, clusters)
            if state in _RUNNING_STATES:
                began = True
            if live is False:
                return cancelled_at is not None, began
            unknown = unknown + 1 if live is None else 0
            if unknown >= _UNKNOWN_POLLS:
                # squeue has stopped answering over this link; accounting may
                # still know, and it answers on a fresh connection.
                final, _, _ = self._accounting(job_id, clusters)
                if final is not None:
                    # Whether the program ran is then decided by the script's
                    # own ``started`` stamp, which the fetch brings back.
                    return cancelled_at is not None, began
                raise RuntimeError(
                    f"slurm job {job_id} could not be followed: squeue failed {unknown} times "
                    f"in a row ({getattr(self, '_last_poll_error', '')}) and accounting has no "
                    "final state; the job may still be running"
                )
            now = time.monotonic()
            if cancelled_at is None and self.max_wait is not None and now - started > self.max_wait:
                self._cancel(job_id, clusters)
                cancelled_at = now
            elif cancelled_at is not None and now - cancelled_at > self.cancel_grace:
                return True, began
            sleep_for = interval
            if cancelled_at is not None:
                # The job is going away; watch for it at the original cadence
                # rather than at whatever the backoff has grown to, so the
                # cancel is seen through promptly.
                sleep_for = min(interval, self.poll_interval)
            elif self.max_wait is not None:
                # Never sleep past ``max_wait``: the interval has been backing
                # off, and a job would otherwise keep its place in the queue
                # for up to one grown interval beyond the limit it was given.
                sleep_for = max(0.0, min(interval, self.max_wait - (now - started)))
            time.sleep(sleep_for)
            interval = min(interval * self.backoff, self.max_poll_interval)

    def _cancel(self, job_id: str, clusters: Sequence[str]) -> None:
        for _ in range(3):
            if self._ssh(shlex.join(["scancel", *clusters, job_id])).returncode == 0:
                return
            time.sleep(1.0)

    def _recorded(self, status: Path, remote_scratch: str) -> Optional[str]:
        """The script's own record: an exit status, ``terminated``, or ``None``.

        The file came back with the scratch directory. A record that has not
        landed yet is re-fetched until ``status_grace`` runs out: the cluster's
        shared filesystem can lag the controller that reported the job gone.
        """
        deadline = time.monotonic() + self.status_grace
        remote = posixpath.join(remote_scratch, status.name)
        while True:
            text = _read(status)
            if text is not None and text.strip():
                value = text.strip()
                if value.startswith(_TERMINATED) or value.lstrip("-").isdigit():
                    return value
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(1.0, self.status_grace))
            self._rsync(
                [*self.host.rsync_prefix(), self.host.remote(remote), str(status)],
                "fetch the job's exit status",
                raises=None,  # the record is simply not there yet
            )

    @staticmethod
    def _walltime_kill(recorded: str, begin: str, request: ExecutionRequest) -> bool:
        """Whether a ``terminated`` record is the time limit rather than a cancel.

        Decided on the node's own clock: against ``SLURM_JOB_END_TIME`` when
        the job had one, else against the walltime actually requested (whole
        minutes), less a minute for the prolog that runs before ``started``.
        """
        fields = recorded.split()
        if len(fields) < 2 or not fields[1].isdigit():
            return False
        ended = int(fields[1])
        if len(fields) >= 3 and fields[2].isdigit():
            return ended >= int(fields[2]) - 10
        if request.timeout is None or not begin.isdigit():
            return False
        return ended - int(begin) >= int(walltime(request.timeout)) * 60 - 60

    def _accounting(
        self, job_id: str, clusters: Sequence[str]
    ) -> tuple[Optional[str], Optional[int], Optional[float]]:
        """``(state, exit code, run_s)`` from ``sacct``, or all ``None`` without it."""
        command = shlex.join(
            ["sacct", *clusters, "-j", job_id, "-X", "-n", "-P", "-o", "State,ExitCode,Start,End"]
        )
        deadline = time.monotonic() + self.accounting_grace
        while True:
            report = self._ssh(command)
            parsed = _parse_accounting(report.stdout) if report.returncode == 0 else None
            if parsed is not None:
                state, exit_code, run_s = parsed
                # slurmdbd lags the controller; wait for the final state.
                if state is not None and state not in LIVE_STATES:
                    return state, exit_code, run_s
            elif report.returncode != 0 and not any(
                marker in (report.stderr or report.stdout) for marker in _ACCOUNTING_TRANSIENT
            ):
                return None, None, None  # accounting is not configured here
            if time.monotonic() >= deadline:
                return None, None, None
            time.sleep(min(1.0, self.accounting_grace))


__all__ = [
    "FETCH_ALL",
    "FETCH_NONE",
    "UNFORWARDED_ENVIRONMENT",
    "RemoteHost",
    "RemoteSlurmBackend",
]
