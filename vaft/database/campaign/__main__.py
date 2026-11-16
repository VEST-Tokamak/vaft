"""``python -m vaft.database.campaign {plan,run-lane,launch,benchmark,compare,verdict}``."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .benchmark import compare_reports, lane_report
from .compare import compare_products, summarize
from .lanes import LaneState, disjoint, plan_lanes, read_shots, split_chunks
from .launch import launch_lanes
from .run import LaneOptions, run_lane
from .spec import WorkflowSpec
from .verdict import chunk_verdict


def _command(text: str | None) -> list[str]:
    return shlex.split(text) if text else []


def _lane_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--workflow-dir", type=Path, required=required)
    parser.add_argument("--config", type=Path, required=required, help="the run config (YAML)")
    parser.add_argument("--filedb", type=Path, help="the lane's FileDB as seen here (default <lane>/filedb)")
    parser.add_argument("--base-dir", help="base_dir Snakemake sees (when --prefix maps the FileDB elsewhere)")
    parser.add_argument("--chunk", type=int, help="chunk size (fixed once the lane is split)")
    parser.add_argument("--pin", help="code SHA the lane is pinned at (fixed for the lane's life)")
    parser.add_argument("--mem-mb", type=int, help="memory Snakemake may book at once (--resources mem_mb)")
    parser.add_argument("--prefix", help="command prepended to Snakemake (shell words)")
    parser.add_argument("--before-chunk", help="hook: <cmd> <lane> <chunk> <shots file> <filedb>")
    parser.add_argument("--after-chunk", help="hook, same arguments, after a finished chunk")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--allow-missing-inputs", action="store_true")
    parser.add_argument("--guard-kill-mb", type=int, help="kill one watched solver above this RSS")
    parser.add_argument("--guard-total-mb", type=int, help="kill the largest when watched solvers pass this")


def _lane_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--workflow-dir", str(args.workflow_dir.resolve()), "--config", str(args.config.resolve())]
    for flag in ("filedb", "base_dir", "chunk", "pin", "mem_mb", "prefix", "before_chunk", "after_chunk",
                 "max_chunks", "guard_kill_mb", "guard_total_mb"):
        value = getattr(args, flag)
        if value is not None:
            argv += [f"--{flag.replace('_', '-')}", str(value)]
    if args.allow_missing_inputs:
        argv.append("--allow-missing-inputs")
    return argv


def _backend(args: argparse.Namespace):
    if args.backend == "local":
        from vaft.code.execution import LocalBackend

        return LocalBackend()
    from vaft.code.slurm import SlurmBackend

    env = os.environ
    return SlurmBackend(
        mode="batch",
        partition=args.partition or env.get("VAFT_SLURM_PARTITION"),
        account=args.account or env.get("VAFT_SLURM_ACCOUNT"),
        qos=args.qos or env.get("VAFT_SLURM_QOS"),
        extra_args=_command(args.slurm_args or env.get("VAFT_SLURM_ARGS")),
        cluster=args.cluster,
        approval=args.approval,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.database.campaign", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="deal shots over lanes and split each into chunks")
    plan.add_argument("shots_file", type=Path)
    plan.add_argument("run_dir", type=Path)
    plan.add_argument("--lanes", type=int, required=True)
    plan.add_argument("--chunk", type=int, required=True)
    plan.add_argument("--pin", help="code SHA every lane is pinned at")
    plan.add_argument("--exclude", type=Path, nargs="*", default=(), help="run dirs whose shots must not overlap")

    lane = commands.add_parser("run-lane", help="run one lane's pending chunks here")
    lane.add_argument("lane", type=Path)
    lane.add_argument("--cores", type=int, required=True)
    _lane_arguments(lane)

    launch = commands.add_parser("launch", help="start every pending lane of a run through a backend")
    launch.add_argument("run_dir", type=Path)
    launch.add_argument("--backend", choices=("local", "slurm"), required=True)
    launch.add_argument("--cores", type=int, required=True)
    launch.add_argument("--memory-mb", type=int, help="memory booked per lane")
    launch.add_argument("--time-s", type=float, help="time limit per lane")
    launch.add_argument("--lanes", nargs="*", help="only these lane names")
    launch.add_argument("--partition")
    launch.add_argument("--account")
    launch.add_argument("--qos")
    launch.add_argument("--cluster", help="cluster name for the execution policy")
    launch.add_argument("--approval", help="approval id for the execution policy")
    launch.add_argument("--slurm-args", help="further sbatch options (shell words)")
    launch.add_argument("--wait", action="store_true", help="wait for the lanes (always for --backend local)")
    launch.add_argument("--lane-command", help="site script the job runs as '<cmd> <lane>' (shell words)")
    _lane_arguments(launch, required=False)

    bench = commands.add_parser("benchmark", help="cost report of one or more lanes")
    bench.add_argument("lanes", nargs="+", help="label=lane_dir or lane_dir")
    bench.add_argument("--workflow-dir", type=Path, help="to report checkpoint and plot shares")

    compare = commands.add_parser("compare", help="product equivalence of two FileDB roots")
    compare.add_argument("new_root", type=Path)
    compare.add_argument("ref_roots", type=Path, nargs="+")
    compare.add_argument("--shots-file", type=Path, required=True)
    compare.add_argument("--workflow-dir", type=Path, required=True)
    compare.add_argument("--layout", default="filedb")
    compare.add_argument("--out", type=Path)

    verdict = commands.add_parser("verdict", help="judge one chunk's Snakemake log")
    verdict.add_argument("log", type=Path)
    verdict.add_argument("--returncode", type=int, required=True)
    verdict.add_argument("--workflow-dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "plan":
        shots = read_shots(args.shots_file)
        lanes_dirs = [lane for run in args.exclude for lane in sorted(run.glob("lane*"))]
        overlap = disjoint(shots, *(LaneState(lane).shots() for lane in lanes_dirs))
        if overlap:
            print(f"{len(overlap)} shots are already in another run's lanes, e.g. {overlap[:5]}", file=sys.stderr)
            return 2
        if any(args.run_dir.glob("lane*/chunks")) or any(args.run_dir.glob("lane*/shots.txt")):
            print(f"{args.run_dir} is already planned; its lanes' shots and chunks are fixed", file=sys.stderr)
            return 2
        for index, lane_shots in enumerate(plan_lanes(shots, args.lanes)):
            state = LaneState(args.run_dir / f"lane{index}")
            state.ensure("chunk_size", str(args.chunk))
            if args.pin:
                state.ensure("PIN_SHA", args.pin)
            split_chunks(lane_shots, args.chunk, state.chunks_dir)
            (state.path / "shots.txt").write_text("".join(f"{s}\n" for s in lane_shots), encoding="utf-8")
        print(json.dumps({"lanes": args.lanes, "shots": len(set(shots))}))
        return 0
    if args.command == "run-lane":
        options = LaneOptions(
            workflow_dir=args.workflow_dir, config=args.config, cores=args.cores, filedb=args.filedb,
            base_dir=args.base_dir, chunk_size=args.chunk, pin=args.pin, mem_mb=args.mem_mb,
            prefix=_command(args.prefix), before_chunk=_command(args.before_chunk),
            after_chunk=_command(args.after_chunk), max_chunks=args.max_chunks,
            allow_missing_inputs=args.allow_missing_inputs, guard_kill_mb=args.guard_kill_mb,
            guard_total_mb=args.guard_total_mb,
        )
        return run_lane(args.lane, options)
    if args.command == "launch":
        lane_command = _command(args.lane_command)
        if not lane_command and (args.workflow_dir is None or args.config is None):
            parser.error("launch needs --workflow-dir and --config, or --lane-command")
        handles = launch_lanes(args.run_dir, _backend(args), [] if lane_command else _lane_argv(args),
                               cores=args.cores, memory_mb=args.memory_mb, timeout=args.time_s,
                               lanes=args.lanes, lane_command=lane_command)
        print(json.dumps({name: handle.job_id for name, handle in handles.items()}))
        if args.backend == "local" or args.wait:
            failed = 0
            for name, handle in handles.items():
                result = handle.result()
                print(json.dumps({"lane": name, "returncode": result.returncode,
                                  "status": result.runtime_status}))
                failed += result.returncode != 0
            return 1 if failed else 0
        return 0
    if args.command == "benchmark":
        spec = WorkflowSpec.load(args.workflow_dir) if args.workflow_dir else None
        reports = {}
        for item in args.lanes:
            label, _, path = item.rpartition("=")
            reports[label or path] = lane_report(path, spec)
        print(json.dumps({"table": compare_reports(reports), "reports": reports}, indent=1, default=str))
        return 0
    if args.command == "compare":
        report = compare_products(args.new_root, args.ref_roots[0], read_shots(args.shots_file),
                                  spec=WorkflowSpec.load(args.workflow_dir), layout=args.layout,
                                  ref_roots=args.ref_roots)
        if args.out:
            args.out.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1), encoding="utf-8")
        print(json.dumps(summarize(report), indent=1))
        return 0
    spec = WorkflowSpec.load(args.workflow_dir) if args.workflow_dir else None
    result = chunk_verdict(args.returncode, args.log.read_text(encoding="utf-8", errors="replace"),
                           run_scoped_rules=spec.run_scoped_rules if spec else ())
    print(json.dumps({"done": result.done, "reason": result.reason, "failed_rules": list(result.failed_rules)}))
    return 0 if result.done else 1


if __name__ == "__main__":
    raise SystemExit(main())
