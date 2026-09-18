"""Reproduce issue #666 against local clones: python -m vaft.code.nice.validate_reference.

Reports preserve failures; this command never treats process exit zero as
scientific acceptance. Output directories must be new to avoid stale tables.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from . import (
    NiceConfig,
    constraint_family_configs,
    prepare_nice_inputs,
    run_nice,
    vest_reference_parameter_file,
    write_study_report,
)

PIN = "7ad1ea8f3da4fee25a61a7c2c01b1773db5f4906"


def collect_report(native, reports, repo):
    """Recollect immutable native runs after parser changes, without rerunning."""
    from . import collect_nice_outputs, compare_equilibria
    from vaft.omas import load

    rows, references = {}, {}
    for manifest in sorted(native.rglob("nice_case_manifest.json")):
        provenance = json.loads(manifest.read_text())
        if "process_returncode" not in provenance:
            continue  # explicitly unsupported, pre-execution families
        result = collect_nice_outputs(manifest.parent)
        row = record(result)
        if result.ods is not None:
            shot, time = provenance["shot"], provenance["time_s"]
            if shot not in references:
                references[shot] = load(
                    repo
                    / f"vaft/data/samples/{shot}/source/pipeline-until-efit.json.gz"
                )
            row["equilibrium_comparison"] = compare_equilibria(
                result.ods, references[shot], time
            )
            row[
                "comparison_warning"
            ] = "Unconverged NICE output is diagnostic evidence, NOT an accepted reconstruction. Stored EFIT reference used; not rerun with shared constraints."
        rows[str(manifest.parent.relative_to(native))] = row
    write_study_report(reports / "collected.json", rows)
    efit_logs = {
        p.name: p.read_text(errors="replace")
        for p in (native / "efit-exact").glob("run_efit.*")
    }
    if efit_logs:
        write_study_report(reports / "efit_logs.json", efit_logs)


def condition(ods, shot, times, work, repo, table_dir=None):
    from vaft.code.efit import generate_constraints_ods
    from vaft.validation.magnetics import validate_magnetics_signals
    from vaft.validation.model import ValidationStatus

    bad = [
        q.index + 1
        for q in validate_magnetics_signals(ods, kinds=("b_field_pol_probe",))
        if q.status is not ValidationStatus.NOT_AVAILABLE and q.valid_fraction == 0
    ]
    work.mkdir(parents=True, exist_ok=True)
    generate_constraints_ods(
        ods,
        shot,
        str(work),
        str(table_dir or repo / "vaft/data/efit") + "/",
        np.asarray(times),
        [1e-4, 1e-4, 5e-2, 3e-2, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2],
        [1, 1, 1, 0.1, 0.1, 0.1, 0.01, 0.01],
        broken=sorted(set(bad + [65, 66, 67, 68, 72, 74])),
        fit=1,
    )
    return {
        "whole_record_broken_probes_one_based": bad,
        "excluded_flux_channels_legacy_one_based": [65, 66, 67, 68, 72, 74],
        "conditioner": "vaft.code.efit.generate_constraints_ods",
        "averaging_half_width_s": 0.0005,
    }


def record(result):
    record = {
        key: getattr(result, key)
        for key in (
            "returncode",
            "process_succeeded",
            "converged",
            "scientifically_usable",
            "termination_reason",
            "nonlinear_iterations",
            "objective",
            "iteration_history",
            "diagnostic_residuals",
            "parsing_errors",
            "provenance",
        )
    }
    record["logs"] = {p.name: p.read_text(errors="replace") for p in result.logs}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nice-home", type=Path, default=Path.home() / "git/nice")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--efit-executable", type=Path)
    parser.add_argument("--efit-table-dir", type=Path)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--focus-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.nice_home), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        parser.error(f"NICE revision {revision} is not pinned {PIN}")
    repo = Path(__file__).resolve().parents[3]
    if args.collect_only:
        collect_report(args.native_dir, args.report_dir, repo)
        return
    if args.native_dir.exists() or args.report_dir.exists():
        parser.error("use fresh native and report directories; preserve earlier runs")
    args.native_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[3]
    from vaft.omas import load

    def load_case(shot):
        source = repo / f"vaft/data/samples/{shot}/source/pipeline-until-efit.json.gz"
        return load(source), hashlib.sha256(source.read_bytes()).hexdigest()

    def config(shot, time, folder, digest):
        return NiceConfig(
            shot=shot,
            time=float(time),
            workdir=folder,
            executable=str(args.executable.resolve()),
            nice_home=args.nice_home,
            source_revision=PIN,
            input_snapshot_hash=digest,
            parameter_file=vest_reference_parameter_file(),
            timeout=args.timeout,
            diagnostic_source="equilibrium_constraints",
            correct_active_response=True,
            flux_loop_input_sign=-1,
            solver_tolerances={"epsStopRecon": 1e-10, "iterMaxRecon": 30},
        )

    def execute(ods, cfg):
        try:
            result = run_nice(prepare_nice_inputs(ods, cfg), cfg)
            print(f"{cfg.shot} {cfg.time:.6f}: {result.termination_reason}", flush=True)
            return record(result), result
        except Exception as exc:
            print(f"{cfg.shot} {cfg.time:.6f}: {type(exc).__name__}: {exc}", flush=True)
            return {
                "scientifically_usable": False,
                "error": f"{type(exc).__name__}: {exc}",
            }, None

    ods, digest = load_case(41672)
    conditioning = condition(
        ods,
        41672,
        [0.331, 0.332],
        args.native_dir / "focus-condition",
        repo,
        args.efit_table_dir,
    )
    base = config(41672, 0.331, args.native_dir / "focus", digest)
    focus, _ = execute(ods, base)
    focus.update({"requested_time_s": 0.331, "conditioning": conditioning})
    write_study_report(args.report_dir / "focus.json", focus)
    if args.efit_executable:
        from vaft.code.efit import (
            EFITConfig,
            EFITConstraintConfig,
            prepare_efit_inputs,
            run_efit,
        )

        try:
            ec = EFITConfig(
                executable=str(args.efit_executable.resolve()),
                shot=41672,
                workdir=args.native_dir / "efit-exact",
                timeout=60,
                args=("129",),
                constraints=EFITConstraintConfig(
                    uncertainty_mode="standard_deviation",
                    use_diamagnetic_flux=False,
                    wall_current_mode="measured",
                    passive_structure_mode="fixed_currents",
                ),
            )
            er = run_efit(prepare_efit_inputs(ods, ec), ec)
            ef = {
                "requested_time_s": 0.331,
                "returncode": er.returncode,
                "status": er.status,
                "reason": er.reason,
                "usable": er.usable,
                "slice_statuses": [asdict(s) for s in er.slice_statuses],
                "gfiles": [str(p) for p in er.gfiles],
                "parse_errors": er.parse_errors,
                "artifact_hashes": er.artifact_hashes,
            }
        except Exception as exc:
            ef = {"requested_time_s": 0.331, "error": f"{type(exc).__name__}: {exc}"}
        write_study_report(args.report_dir / "efit_exact.json", ef)
    families = {}
    for name, cfg in constraint_family_configs(base).items():
        slug = "".join(c.lower() if c.isalnum() else "_" for c in name)
        families[name], _ = execute(
            ods, replace(cfg, workdir=args.native_dir / "families" / slug)
        )
    write_study_report(args.report_dir / "families.json", families)
    if args.focus_only:
        collect_report(args.native_dir, args.report_dir, repo)
        return
    overall = {}
    for shot, count in ((39915, 9), (41524, 6), (41672, 19)):
        ods, digest = load_case(shot)
        times = np.asarray(ods["equilibrium.time"], float).copy()
        if len(times) != count:
            raise ValueError(f"unexpected reference window for {shot}: {times}")
        conditioning = condition(
            ods,
            shot,
            times,
            args.native_dir / f"condition-{shot}",
            repo,
            args.efit_table_dir,
        )
        rows, results = [], []
        for t in times:
            row, result = execute(
                ods, config(shot, t, args.native_dir / str(shot) / f"{t:.6f}", digest)
            )
            rows.append({"time_s": float(t), **row})
            if result is not None:
                results.append(result)
            write_study_report(
                args.report_dir / f"{shot}.json",
                {"conditioning": conditioning, "slices": rows},
            )
        overall[str(shot)] = {
            "requested": count,
            "usable": sum(r.get("scientifically_usable") is True for r in rows),
            "times_s": times.tolist(),
            "preparation_failures": count - len(results),
        }
    write_study_report(args.report_dir / "overall.json", overall)
    collect_report(args.native_dir, args.report_dir, repo)


if __name__ == "__main__":
    main()
