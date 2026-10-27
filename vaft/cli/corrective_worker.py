"""Watch the Thomson/CES uploads and run the corrective pipeline on them.

    vaft corrective-worker run --config /srv/vaft/corrective.yaml        # the service
    vaft corrective-worker run --config corrective.yaml --once           # one cycle
    vaft corrective-worker status --config corrective.yaml [--shot N] [--json]
    vaft corrective-worker retry --config corrective.yaml --shot N
    vaft corrective-worker exclude --config corrective.yaml --shot N --reason "..."

Pipeline 2 (``workflow/automatic_pipeline_2_corrective_data_update``) builds the
Thomson lineage for shots that were processed long ago, whenever someone uploads
a ``.mat`` file for them.  This worker notices the upload, waits until it has
stopped changing, runs the pipeline for that shot and reads the stage manifests
back.  It is independent of ``vaft pipeline-worker`` (own state file, own lock),
and ``hold_file`` in its configuration pauses it while a regeneration owns the
same shots.  See ``corrective.example.yaml`` beside the pipeline and its README.
``--config`` may be replaced by ``$VAFT_CORRECTIVE_WORKER_CONFIG``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import signal
import sys
from typing import Iterable

from vaft.cli.pipeline_worker import WorkerAlreadyRunningError, worker_lock
from vaft.database.worker.config import WorkerConfigError
from vaft.database.worker.corrective import (
    CorrectiveState,
    CorrectiveWorker,
    load_corrective_config,
    read_corrective_state,
)


def _run(args: argparse.Namespace) -> int:
    config = load_corrective_config(args.config)
    with worker_lock(config.state_db):
        worker = CorrectiveWorker(config)

        def request_stop(signum, _frame):
            logging.getLogger(__name__).info("signal %s: stopping after the current step", signum)
            worker.stop()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        if args.once:
            report = worker.run_cycle(recover=True)
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0
        worker.run_forever()
    return 0


def _status(args: argparse.Namespace) -> int:
    config = load_corrective_config(args.config)
    with read_corrective_state(config.state_db) as state:
        if args.shot is not None:
            row = state.shot(args.shot)
            if row is None:
                print(f"shot {args.shot} has no recorded upload", file=sys.stderr)
                return 1
            payload = {"shot": dict(row), "stages": state.stage_status(args.shot),
                       "events": state.events(shot=args.shot)}
            if args.json:
                print(json.dumps(payload, indent=2, default=str))
            else:
                print(f"shot {row['shot']}: {row['state']} attempts={row['attempts']}"
                      f"{' (forced rerun pending)' if row['force'] else ''} -- {row['reason']}")
                for stage in payload["stages"]:
                    print(f"  {stage['stage']} [{stage['kind']}]: {stage['status']}")
            return 0
        summary = {
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
            for name, count in sorted(summary["counts"].items()):
                print(f"  {name:12s} {count}")
            for row in summary["attention"]:
                print(f"  ! {row['shot']} {row['state']} (attempts {row['attempts']}): {row['reason']}")
    return 0


def _retry(args: argparse.Namespace) -> int:
    config = load_corrective_config(args.config)
    with CorrectiveState(config.state_db) as state:
        if not state.requeue(args.shot, reason=f"operator retry: {args.reason}", at=datetime.now()):
            print(f"shot {args.shot} is unknown or currently running", file=sys.stderr)
            return 1
    print(f"shot {args.shot} requeued (the next run is forced)")
    return 0


def _exclude(args: argparse.Namespace) -> int:
    config = load_corrective_config(args.config)
    with CorrectiveState(config.state_db) as state:
        if not state.exclude(args.shot, reason=f"operator: {args.reason}", at=datetime.now()):
            print(f"shot {args.shot} is unknown or currently running", file=sys.stderr)
            return 1
    print(f"shot {args.shot} excluded")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli corrective-worker", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", help="corrective worker YAML (default: $VAFT_CORRECTIVE_WORKER_CONFIG)")

    run = subparsers.add_parser("run", help="watch the uploads and run the corrective pipeline")
    add_config(run)
    run.add_argument("--once", action="store_true", help="run one cycle and exit")

    status = subparsers.add_parser("status", help="summarise the worker's durable state")
    add_config(status)
    status.add_argument("--shot", type=int)
    status.add_argument("--json", action="store_true")

    retry = subparsers.add_parser("retry", help="queue a shot again, forced, with a fresh attempt budget")
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
        print(f"corrective-worker: {error}", file=sys.stderr)
        return 2


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
