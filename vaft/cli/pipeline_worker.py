"""Poll VEST SQL for new shots and run the routine pipeline on them (issue #58).

    vaft pipeline-worker run --config /srv/vaft/worker.yaml          # the service
    vaft pipeline-worker run --config worker.yaml --once             # one cycle
    vaft pipeline-worker status --config worker.yaml [--shot N] [--json]
    vaft pipeline-worker retry --config worker.yaml --shot N
    vaft pipeline-worker exclude --config worker.yaml --shot N --reason "..."

The configuration is server-only; see ``worker.example.yaml`` beside the
routine pipeline and ``DEPLOYMENT.md``.  ``--config`` may be replaced by
``$VAFT_WORKER_CONFIG``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Iterable, Iterator

from vaft.cli.raw_redump import _lock_file_nonblocking, _unlock_file
from vaft.database.worker.config import WorkerConfigError, load_worker_config
from vaft.database.worker.state import WorkerState, read_worker_state


class WorkerAlreadyRunningError(RuntimeError):
    """Raised when another worker holds the state file's lock."""


@contextmanager
def worker_lock(state_db: Path) -> Iterator[None]:
    """One worker per state file, for the worker's whole lifetime."""
    state_db.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_db.with_name(state_db.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            _lock_file_nonblocking(handle)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            raise WorkerAlreadyRunningError(
                f"another pipeline worker holds {lock_path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            _unlock_file(handle)


def _run(args: argparse.Namespace) -> int:
    from vaft.database.worker.service import PipelineWorker

    config = load_worker_config(args.config)
    with worker_lock(config.state_db):
        worker = PipelineWorker(config)
        if args.once:
            report = worker.run_cycle(recover=True)
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0

        def request_stop(signum, _frame):
            logging.getLogger(__name__).info("signal %s: stopping after the current step", signum)
            worker.stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        worker.run_forever()
    return 0


def _status(args: argparse.Namespace) -> int:
    config = load_worker_config(args.config)
    with read_worker_state(config.state_db) as state:
        if args.shot is not None:
            row = state.shot(args.shot)
            if row is None:
                print(f"shot {args.shot} is not known to the worker", file=sys.stderr)
                return 1
            payload = {
                "shot": row,
                "stages": state.stage_status(args.shot),
                "events": state.events(shot=args.shot),
            }
            if args.json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"shot {row['shot']}: {row['state']} ({row['classification']}) "
                      f"attempts={row['attempts']} -- {row['reason']}")
                for stage in payload["stages"]:
                    product = f"/{stage['product']}" if stage["product"] else ""
                    print(f"  {stage['stage']}{product} [{stage['kind']}]: {stage['status']}")
            return 0
        summary = {
            "watermark": state.watermark(config.first_shot - 1),
            "counts": state.counts(),
            "attention": [
                {k: row[k] for k in ("shot", "state", "attempts", "reason")}
                for s in ("failed", "gave_up")
                for row in state.shots(state=s)
            ],
        }
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(f"watermark: {summary['watermark']}")
            for name, count in sorted(summary["counts"].items()):
                print(f"  {name:10s} {count}")
            for row in summary["attention"]:
                print(f"  ! {row['shot']} {row['state']} (attempts {row['attempts']}): {row['reason']}")
    return 0


def _retry(args: argparse.Namespace) -> int:
    config = load_worker_config(args.config)
    with WorkerState(config.state_db) as state:
        if not state.requeue(args.shot, reason=f"operator retry: {args.reason}"):
            print(f"shot {args.shot} is unknown or currently running", file=sys.stderr)
            return 1
    print(f"shot {args.shot} requeued")
    return 0


def _exclude(args: argparse.Namespace) -> int:
    config = load_worker_config(args.config)
    with WorkerState(config.state_db) as state:
        if not state.exclude(args.shot, reason=f"operator: {args.reason}"):
            print(f"shot {args.shot} is unknown or currently running", file=sys.stderr)
            return 1
    print(f"shot {args.shot} excluded")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli pipeline-worker", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", help="worker YAML (default: $VAFT_WORKER_CONFIG)")

    run = subparsers.add_parser("run", help="poll SQL and run the pipeline on new shots")
    add_config(run)
    run.add_argument("--once", action="store_true", help="run one cycle and exit")

    status = subparsers.add_parser("status", help="summarise the worker's durable state")
    add_config(status)
    status.add_argument("--shot", type=int)
    status.add_argument("--json", action="store_true")

    retry = subparsers.add_parser("retry", help="queue a shot again with a fresh attempt budget")
    add_config(retry)
    retry.add_argument("--shot", type=int, required=True)
    retry.add_argument("--reason", default="requested")

    exclude = subparsers.add_parser("exclude", help="never schedule a shot again")
    add_config(exclude)
    exclude.add_argument("--shot", type=int, required=True)
    exclude.add_argument("--reason", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handlers = {"run": _run, "status": _status, "retry": _retry, "exclude": _exclude}
    try:
        return handlers[args.command](args)
    except (WorkerConfigError, WorkerAlreadyRunningError, FileNotFoundError) as error:
        print(f"pipeline-worker: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
