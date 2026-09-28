"""A local launch stops the whole process tree, not just its direct child (#1016).

Real programs: a Python (or ``sh``) parent that starts a sleeping grandchild
and writes the grandchild's pid to a file. Every test asserts that grandchild
is gone -- or, for a normal exit, deliberately left alone. The grace period is
patched down so a forced stop takes a fraction of a second.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vaft.code import ExecutionRequest, LocalBackend
from vaft.code import _process_tree
from vaft.code._process_tree import ProcessTree

POSIX = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups and signals")

# Parent: start a grandchild, publish its pid, announce, then sleep. argv[1] is
# the pid file; argv[2] == "stubborn" makes the grandchild ignore SIGTERM.
_TREE = r"""
import pathlib, subprocess, sys, time
grandchild = (
    "import signal, time\n"
    + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if sys.argv[2:] == ["stubborn"] else "")
    + "time.sleep(60)\n"
)
child = subprocess.Popen([sys.executable, "-c", grandchild])
time.sleep(0.2)  # let the grandchild install its disposition
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print("started", flush=True)
time.sleep(60)
"""


@pytest.fixture(autouse=True)
def short_grace(monkeypatch):
    monkeypatch.setattr(_process_tree, "TERMINATE_GRACE_S", 1.0)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers signal 0 until its new parent (init) reaps it.
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat:
            return stat.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        pass
    probe = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = probe.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _gone(pid: int, within: float = 5.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _pid(pid_file: Path, within: float = 10.0) -> int:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if pid_file.is_file() and pid_file.read_text().strip():
            return int(pid_file.read_text())
        time.sleep(0.02)
    raise AssertionError(f"{pid_file} never appeared")


def _kill(pid: int) -> None:
    if _alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGKILL)


def _kill_published(pid_file: Path) -> None:
    if pid_file.is_file() and pid_file.read_text().strip():
        _kill(int(pid_file.read_text()))


def _signal_once_started(pid_file: Path, send) -> threading.Thread:
    """Call ``send()`` once the tree has published its grandchild, or after 10 s anyway."""

    def run():
        try:
            _pid(pid_file)
        except AssertionError:
            pass
        send()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _tree_request(tmp_path: Path, *extra: str, **request) -> ExecutionRequest:
    script = tmp_path / "tree.py"
    script.write_text(_TREE, encoding="utf-8")
    return ExecutionRequest(
        command=(sys.executable, str(script), str(tmp_path / "grandchild.pid"), *extra),
        workdir=tmp_path,
        **request,
    )


def test_timeout_stops_the_grandchild_and_keeps_partial_output(tmp_path):
    result = LocalBackend().run(_tree_request(tmp_path, timeout=5.0))
    grandchild = _pid(tmp_path / "grandchild.pid")
    try:
        assert result.timed_out
        assert result.returncode is None
        assert "started" in result.stdout
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


def test_timeout_with_a_log_keeps_the_partial_log(tmp_path):
    log = tmp_path / "logs" / "run.log"
    result = LocalBackend().run(_tree_request(tmp_path, timeout=5.0, log_path=log))
    grandchild = _pid(tmp_path / "grandchild.pid")
    try:
        assert result.timed_out and result.returncode is None
        assert result.log_path == log
        assert (result.stdout, result.stderr) == ("", "")
        assert "started" in log.read_text(encoding="utf-8")
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


@POSIX
def test_a_shell_launcher_that_does_not_exec_is_stopped_whole(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    command = f"sleep 60 & echo $! > {pid_file}; echo started; wait"
    result = LocalBackend().run(
        ExecutionRequest(command=("sh", "-c", command), workdir=tmp_path, timeout=4.0)
    )
    grandchild = _pid(pid_file)
    try:
        assert result.timed_out
        assert "started" in result.stdout
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="environment token is read from /proc")
def test_a_descendant_orphaned_before_the_stop_is_found_by_its_token(tmp_path):
    """A launcher that exits and leaves its background job holding the pipe.

    The job was reparented to init before the timeout, so no parent-id walk
    reaches it; the inherited ``VAFT_PROCESS_TREE`` token does.
    """
    pid_file = tmp_path / "orphan.pid"
    command = f"sleep 60 & echo $! > {pid_file}; echo started; exit 0"
    result = LocalBackend().run(
        ExecutionRequest(command=("sh", "-c", command), workdir=tmp_path, timeout=4.0)
    )
    orphan = _pid(pid_file)
    try:
        assert result.timed_out
        assert "started" in result.stdout
        assert _gone(orphan)
    finally:
        _kill(orphan)


@POSIX
def test_a_grandchild_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    started = time.monotonic()
    result = LocalBackend().run(_tree_request(tmp_path, "stubborn", timeout=5.0))
    grandchild = _pid(tmp_path / "grandchild.pid")
    try:
        assert result.timed_out
        assert _gone(grandchild)
        # timeout + grace, not the grandchild's 60 s sleep
        assert time.monotonic() - started < 20.0
    finally:
        _kill(grandchild)


@POSIX
def test_keyboard_interrupt_stops_the_tree_and_is_raised(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    # A real SIGINT, as a terminal Ctrl-C delivers: ``_thread.interrupt_main``
    # only sets a flag and would not break the blocking wait.
    sender = _signal_once_started(pid_file, lambda: os.kill(os.getpid(), signal.SIGINT))
    try:
        with pytest.raises(KeyboardInterrupt):
            LocalBackend().run(_tree_request(tmp_path))
        sender.join()
        assert _gone(_pid(pid_file))
    finally:
        _kill_published(pid_file)


@POSIX
def test_the_tree_stays_in_the_callers_process_group(tmp_path):
    """Group signals (Ctrl-Z, a supervisor's SIGSTOP or SIGKILL) keep reaching the solver."""
    pid_file = tmp_path / "grandchild.pid"
    seen = []
    sender = _signal_once_started(
        pid_file, lambda: seen.append(os.getpgid(_pid(pid_file)))
    )
    try:
        LocalBackend().run(_tree_request(tmp_path, timeout=5.0))
        sender.join()
        assert seen == [os.getpgrp()]
    finally:
        _kill_published(pid_file)


@POSIX
def test_a_callers_own_sigterm_handler_keeps_its_choice(tmp_path):
    """A handler that drains gracefully is not overridden: the run goes on to its timeout."""
    pid_file = tmp_path / "grandchild.pid"
    seen: list[int] = []

    def callers_handler(signum, frame):
        seen.append(signum)

    previous = signal.signal(signal.SIGTERM, callers_handler)
    try:
        sender = _signal_once_started(pid_file, lambda: os.kill(os.getpid(), signal.SIGTERM))
        result = LocalBackend().run(_tree_request(tmp_path, timeout=5.0))
        sender.join()
        assert seen == [signal.SIGTERM]
        assert result.timed_out
        assert signal.getsignal(signal.SIGTERM) is callers_handler
        assert _gone(_pid(pid_file))
    finally:
        signal.signal(signal.SIGTERM, previous)
        _kill_published(pid_file)


@POSIX
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_a_signal_to_a_default_python_ends_it_by_the_signal_with_no_orphan(tmp_path, signame):
    """A stopped job or a dropped ssh session: Python dies of the signal, its solver with it."""
    signum = getattr(signal, signame)
    request = _tree_request(tmp_path)
    driver = tmp_path / "driver.py"
    import vaft

    source = str(Path(vaft.__file__).resolve().parents[1])
    driver.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        # Import the vaft under test, not whichever checkout is installed editable.
        "sys.meta_path[:] = [f for f in sys.meta_path if '__editable__' not in getattr(f, '__module__', '')]\n"
        f"sys.path.insert(0, {source!r})\n"
        "import vaft\n"
        f"assert Path(vaft.__file__).resolve().is_relative_to(Path({source!r})), vaft.__file__\n"
        "from vaft.code import ExecutionRequest, LocalBackend\n"
        f"LocalBackend().run(ExecutionRequest(command={tuple(request.command)!r}, workdir=Path({str(tmp_path)!r})))\n",
        encoding="utf-8",
    )
    python = subprocess.Popen([sys.executable, str(driver)], cwd=tmp_path)
    try:
        grandchild = _pid(tmp_path / "grandchild.pid", within=30.0)
        python.send_signal(signum)
        assert python.wait(timeout=20) == -signum
        assert _gone(grandchild)
    finally:
        if python.poll() is None:
            python.kill()
        _kill_published(tmp_path / "grandchild.pid")


@POSIX
def test_off_the_main_thread_no_handler_is_installed_and_timeout_still_stops_the_tree(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    during: list[object] = []
    results = []

    def run():
        results.append(LocalBackend().run(_tree_request(tmp_path, timeout=5.0)))

    worker = threading.Thread(target=run)
    worker.start()
    _pid(tmp_path / "grandchild.pid")
    during.append(signal.getsignal(signal.SIGTERM))
    worker.join(timeout=30)
    grandchild = _pid(tmp_path / "grandchild.pid")
    try:
        assert during == [before]
        assert results and results[0].timed_out
        assert _gone(grandchild)
    finally:
        _kill(grandchild)


@POSIX
def test_a_normal_exit_leaves_a_detached_background_process_alone(tmp_path):
    """No new behaviour on success: only a timeout or an interrupt stops the tree."""
    pid_file = tmp_path / "background.pid"
    command = f"sleep 30 >/dev/null 2>&1 & echo $! > {pid_file}"
    result = LocalBackend().run(
        ExecutionRequest(command=("sh", "-c", command), workdir=tmp_path, timeout=10.0)
    )
    background = _pid(pid_file)
    try:
        assert result.returncode == 0 and not result.timed_out
        assert _alive(background)
    finally:
        _kill(background)


def test_a_replaced_subprocess_run_still_intercepts_the_launch(tmp_path, monkeypatch):
    """The seam older adapter tests rely on: no real program starts while it is patched."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 5, stdout="faked", stderr="")

    def refuse(*args, **kwargs):
        raise AssertionError("a real process tree was started")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ProcessTree, "start", classmethod(refuse))
    result = LocalBackend().run(
        ExecutionRequest(command=(sys.executable, "-c", "pass"), workdir=tmp_path)
    )
    assert calls == [[sys.executable, "-c", "pass"]]
    assert (result.returncode, result.stdout) == (5, "faked")


def test_an_unlaunchable_program_is_still_the_shared_error(tmp_path):
    from external_code_stubs import write_unlaunchable_file
    from vaft.code import ExecutableNotLaunchable

    program = write_unlaunchable_file(tmp_path / "solver")
    with pytest.raises(ExecutableNotLaunchable) as raised:
        LocalBackend().run(ExecutionRequest(command=(str(program),), workdir=tmp_path))
    assert isinstance(raised.value.__cause__, OSError)


@POSIX
def test_only_the_default_disposition_is_taken_over_and_it_is_restored():
    """``nohup``: an ignored SIGHUP is left ignored; nothing stays installed afterwards."""
    previous_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    previous_term = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with _process_tree.forward_termination():
            during = (signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGTERM))
        assert during[0] is signal.SIG_IGN
        assert callable(during[1])
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGHUP, previous_hup)
        signal.signal(signal.SIGTERM, previous_term)
