"""Run one lane: its chunks in order, each one Snakemake run, resumable.

The same code runs a lane on a workstation, on a server, or inside a scheduler
allocation; nothing here asks which. CPU and memory come from
:class:`vaft.code.accounting.UsageMeter` (the job cgroup inside a Slurm job,
``getrusage`` elsewhere). What is particular to a site -- a mount namespace that
shows the lane's FileDB under the path the config names, staging raw inputs
before a chunk, pruning solver scratch after it -- comes in as a command
``prefix`` and ``before_chunk``/``after_chunk`` hook commands, from the caller.

For each chunk not yet done (``done_chunks.txt``):

1. stop if ``STOP`` exists or ``max_chunks`` chunks already ran in this call;
2. run ``before_chunk`` (``<hook> <lane> <chunk> <chunk shots file> <filedb>``);
3. keep the shots whose inputs (``CAMPAIGN_SHOT_INPUTS``) exist under the lane's
   FileDB; the others go to ``missing_raw.txt`` and keep the chunk open;
4. write the chunk's config (the run config plus ``cores`` and ``shots``), clear
   the run-scoped outputs (otherwise the previous chunk's verdicts would be
   reused under ``--rerun-triggers mtime``), unlock, run Snakemake;
5. judge the run (:func:`~.verdict.chunk_verdict`); a whole-run failure stops
   the lane with status 1;
6. keep a copy of the run-scoped outputs as ``chunks/<chunk>.<name>.json``,
   run ``after_chunk``, record the chunk in ``lane.jsonl``/``lane.log`` and, with
   every input present, mark it done.

One runner per lane at a time (an ``flock`` on ``<lane>/.lock``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

from vaft.code.accounting import UsageMeter

from .lanes import LaneState, read_shots, split_chunks
from .spec import WorkflowSpec
from .verdict import chunk_verdict

__all__ = ["LaneOptions", "run_lane"]

PathLike = Union[str, Path]

#: Exit statuses of :func:`run_lane`.
DONE, FAILED, REFUSED, BUSY = 0, 1, 2, 75
#: How long a second runner waits for the lane lock before giving up [s].
LOCK_WAIT_S = 60.0


@dataclass
class LaneOptions:
    """How to run a lane; everything site-specific is an argument here."""

    workflow_dir: Path
    config: Path
    cores: int
    #: The lane's FileDB as this process sees it (default ``<lane>/filedb``).
    filedb: Optional[Path] = None
    #: ``base_dir`` written into each chunk config; default the ``filedb`` path.
    #: Set it when ``prefix`` shows the FileDB to Snakemake under another path.
    base_dir: Optional[str] = None
    chunk_size: Optional[int] = None
    pin: Optional[str] = None
    #: Total memory Snakemake may book at once (``--resources mem_mb``).
    mem_mb: Optional[int] = None
    #: Prepended to every Snakemake command (e.g. a namespace wrapper, ``nice``).
    prefix: Sequence[str] = ()
    before_chunk: Sequence[str] = ()
    after_chunk: Sequence[str] = ()
    max_chunks: Optional[int] = None
    allow_missing_inputs: bool = False
    #: Kill one watched solver above this many MiB of RSS, or the largest when
    #: the watched solvers together pass ``guard_total_mb``. Linux only.
    guard_kill_mb: Optional[int] = None
    guard_total_mb: Optional[int] = None
    extra_args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _Lane:
    def __init__(self, path: Path) -> None:
        self.state = LaneState(path)
        self.path = self.state.path

    def log(self, text: str) -> None:
        line = f"{_now()} {text}"
        print(line, flush=True)
        with (self.path / "lane.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def record(self, entry: Mapping[str, Any]) -> None:
        with (self.path / "lane.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": _now(), **entry}, default=str) + "\n")


def _hook(command: Sequence[str], lane: Path, chunk: str, shots_file: Path, filedb: Path, env) -> int:
    if not command:
        return 0
    return subprocess.run([*command, str(lane), chunk, str(shots_file), str(filedb)], env=env).returncode


def _missing_outputs(paths: Any, names: Sequence[str]) -> list[str]:
    return [name for name in names if not Path(getattr(paths, name)()).exists()]


def run_lane(lane: PathLike, options: LaneOptions) -> int:
    """Run ``lane``'s pending chunks; ``0`` done or stopped, ``1`` failed, ``2`` refused, ``75`` busy."""
    import fcntl

    lane_path = Path(lane).resolve()
    lane_path.mkdir(parents=True, exist_ok=True)
    handle = (lane_path / ".lock").open("w")
    deadline = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                print(f"another runner holds {lane_path}/.lock; not starting", file=sys.stderr)
                return BUSY
            time.sleep(1.0)
    try:
        return _run(_Lane(lane_path), options)
    finally:
        handle.close()


def _run(lane: _Lane, options: LaneOptions) -> int:
    spec = WorkflowSpec.load(options.workflow_dir)
    filedb = Path(options.filedb or lane.path / "filedb").resolve()
    work = lane.path / "work"
    for directory in (filedb, work, lane.path / "chunks", lane.path / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        if options.pin:
            lane.state.ensure("PIN_SHA", options.pin)
        chunk_size = options.chunk_size or int((lane.path / "chunk_size").read_text().strip())
        lane.state.ensure("chunk_size", str(chunk_size))
    except (ValueError, FileNotFoundError) as error:
        lane.log(f"refusing: {error}")
        return REFUSED
    if not lane.state.chunks():
        shots_file = lane.path / "shots.txt"
        if not shots_file.exists():
            lane.log("refusing: no chunks and no shots.txt")
            return REFUSED
        split_chunks(read_shots(shots_file), chunk_size, lane.state.chunks_dir)

    base = yaml.safe_load(Path(options.config).read_text(encoding="utf-8")) or {}
    base["base_dir"] = options.base_dir or str(filedb)
    # What exists is read through the FileDB as this process sees it, under the
    # config Snakemake will use (the workflow's config.yaml under the run's).
    effective = spec.merged_config(base)
    local_paths = spec.paths({**effective, "base_dir": str(filedb)})
    env = {**os.environ, **options.env}
    command = [*options.prefix, sys.executable, "-m", "snakemake", "--snakefile", str(spec.snakefile),
               "--directory", str(work)]
    lane.log(f"lane start cores={options.cores} chunk={chunk_size} workflow={spec.directory} "
             f"pin={options.pin or '-'} host={platform.node()}")

    started = 0
    for name in lane.state.pending():
        if lane.state.stop_requested:
            lane.log(f"STOP before {name}")
            return DONE
        if options.max_chunks is not None and started >= options.max_chunks:
            lane.log(f"max_chunks={options.max_chunks} reached before {name}")
            return DONE
        started += 1
        chunk_file = lane.state.chunks_dir / name
        if _hook(options.before_chunk, lane.path, name, chunk_file, filedb, env) != 0:
            lane.log(f"{name}: before_chunk hook failed; lane stopped")
            return FAILED
        shots = lane.state.chunk_shots(name)
        ready, missing = [], []
        for shot in shots:
            inputs = [Path(getattr(local_paths, method)(shot)) for method in spec.shot_inputs] if local_paths else []
            (ready if all(path.exists() for path in inputs) else missing).append(shot)
        _note_missing(lane, name, missing)
        if not ready:
            lane.log(f"{name}: no shot has its inputs, nothing to run")
            if options.allow_missing_inputs:
                lane.state.mark_done(name)
            continue

        config = {**base, "cores": options.cores, "shots": ready}
        config_path = lane.state.chunks_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        for output in spec.run_scoped_outputs:
            Path(getattr(local_paths, output)()).unlink(missing_ok=True)
            (lane.state.chunks_dir / f"{name}.{output}.json").unlink(missing_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
        attempt = lane.path / "logs" / f"{name}.attempt-{stamp}.log"
        run = [*command, "--configfile", str(config_path)]
        lane.log(f"{name} start ({len(ready)} shots)")
        with attempt.open("w", encoding="utf-8") as log:
            subprocess.run([*run, "--unlock"], stdout=log, stderr=subprocess.STDOUT, env=env)
            snakemake = [
                *run, "--cores", str(options.cores), "--scheduler", "greedy", "--keep-going",
                "--rerun-incomplete", "--rerun-triggers", "mtime", "--printshellcmds", "--nocolor",
                *(["--resources", f"mem_mb={options.mem_mb}"] if options.mem_mb else []),
                *spec.resource_profile_args(effective), *options.extra_args,
            ]
            meter = UsageMeter(options.cores).start()
            try:
                process = subprocess.Popen(snakemake, stdout=log, stderr=subprocess.STDOUT, env=env)
                guard = _RssGuard(lane, name, process.pid, spec.solver_processes, options)
                guard.start()
                try:
                    returncode = process.wait()
                finally:
                    guard.stop()
            finally:
                usage = meter.stop()
        shutil.copyfile(attempt, lane.path / "logs" / f"{name}.snakemake.log")
        text = attempt.read_text(encoding="utf-8", errors="replace")
        verdict = chunk_verdict(
            returncode, text, run_scoped_rules=spec.run_scoped_rules,
            missing_outputs=_missing_outputs(local_paths, spec.required_outputs) if local_paths else (),
        )
        for output in spec.run_scoped_outputs:
            source = Path(getattr(local_paths, output)())
            if source.exists():
                shutil.copy2(source, lane.state.chunks_dir / f"{name}.{output}.json")
        lane.record({
            "chunk": name, "shots": len(shots), "ran": len(ready), "missing_inputs": len(missing),
            "returncode": returncode, "done": verdict.done, "reason": verdict.reason,
            "failed_rules": sorted(set(verdict.failed_rules)), "cores": options.cores,
            "usage": usage.as_dict(), "log": attempt.name,
        })
        util = usage.utilization
        lane.log(
            f"{name} exit={returncode} wall={usage.wall_s:.0f}s cpu_s={usage.cpu_s:.0f}"
            + (f" cpu_util={util:.3f}" if util is not None else "")
            + (f" mem_max_gb={usage.peak_mem_mb / 1024:.1f}" if usage.peak_mem_mb else "")
            + (f" max_rss_gb={usage.max_process_rss_mb / 1024:.1f}" if usage.max_process_rss_mb else "")
            + f" source={usage.source} log={attempt.name}"
        )
        if not verdict.done:
            lane.log(f"{name}: {verdict.reason}; lane stopped, see {attempt}")
            return FAILED
        if _hook(options.after_chunk, lane.path, name, chunk_file, filedb, env) != 0:
            lane.log(f"{name}: after_chunk hook failed; lane stopped")
            return FAILED
        if missing and not options.allow_missing_inputs:
            lane.log(f"{name}: {len(missing)} shot(s) without inputs (missing_raw.txt); chunk left open")
            continue
        lane.state.mark_done(name)
    lane.log("lane done")
    return DONE


def _note_missing(lane: _Lane, chunk: str, shots: Sequence[int]) -> None:
    if not shots:
        return
    record = lane.path / "missing_raw.txt"
    known = set(record.read_text(encoding="utf-8").splitlines()) if record.exists() else set()
    with record.open("a", encoding="utf-8") as handle:
        for shot in shots:
            if f"{chunk} {shot}" not in known:
                handle.write(f"{chunk} {shot}\n")
    lane.log(f"{chunk}: {len(shots)} shot(s) without inputs left out")


class _RssGuard:
    """Kill a runaway solver of this chunk's process tree before it takes the allocation.

    Reads ``/proc`` (Linux); elsewhere, or without limits or watched names, it
    does nothing. A process is re-read before the kill, so a reused pid of
    another program is never hit.
    """

    def __init__(self, lane: _Lane, chunk: str, root: int, names: Sequence[str], options: LaneOptions) -> None:
        self.lane, self.chunk, self.root, self.names = lane, chunk, root, frozenset(names)
        self.kill_mb, self.total_mb = options.guard_kill_mb, options.guard_total_mb
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.names or not (self.kill_mb or self.total_mb) or not Path("/proc/self/stat").exists():
            return
        self._thread = threading.Thread(target=self._loop, name="vaft-lane-rss-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @staticmethod
    def _stat(pid: int) -> Optional[tuple[str, int]]:
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        comm = text[text.index("(") + 1:text.rindex(")")]
        return comm, int(text[text.rindex(")") + 2:].split()[1])

    @staticmethod
    def _rss_mb(pid: int) -> Optional[int]:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except OSError:
            return None
        return None

    def _watched(self, comm: str) -> bool:
        """A watched name, or one starting with a watched ``prefix*``."""
        return comm in self.names or any(
            name.endswith("*") and comm.startswith(name[:-1]) for name in self.names
        )

    def _descends(self, pid: int) -> bool:
        for _ in range(64):
            if pid == self.root:
                return True
            stat = self._stat(pid)
            if stat is None or stat[1] <= 1:
                return False
            pid = stat[1]
        return False

    def _loop(self) -> None:
        while not self._stop.wait(5.0):
            watched = []
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                stat = self._stat(pid)
                if stat is None or not self._watched(stat[0]) or not self._descends(pid):
                    continue
                rss = self._rss_mb(pid)
                if rss is not None:
                    watched.append((rss, pid, stat[0]))
            if not watched:
                continue
            total = sum(rss for rss, _, _ in watched)
            rss, pid, comm = max(watched)
            if (self.kill_mb and rss > self.kill_mb) or (self.total_mb and total > self.total_mb):
                stat, now = self._stat(pid), self._rss_mb(pid)
                if stat is not None and stat[0] == comm and now is not None and now >= rss // 2:
                    try:
                        os.kill(pid, 9)
                    except OSError:
                        continue
                    self.lane.log(f"{self.chunk}: killed {comm} pid {pid} at {now} MB RSS "
                                  f"(watched {total} MB; limits {self.kill_mb}/{self.total_mb} MB)")
