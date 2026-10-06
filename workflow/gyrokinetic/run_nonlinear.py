"""Nonlinear CGYRO on one Lane T state and surface, resumable across allocations (#1354).

The state is resolved exactly as ``run_linear.py`` resolves it (Lane T's
``resolve_transport_state``; same ``state_identity`` as #1482), and the local input is
the same TGLF-derived CGYRO input, so the saturated flux compares directly with the
#1482 SAT0-3 ``Q/Q_GB`` on that surface.

A production run is long: each invocation runs until ``--max-time`` (``a/c_s``) or the
allocation's limit and leaves ``bin.cgyro.restart``; ``--restart`` continues it (CGYRO
appends to its time records, so the whole trace survives). ``--smoke`` caps the run at a
few ``a/c_s`` to measure the cost per unit time before a production submission.

The field model defaults to ``em-aperp``. Electrostatic runs with kinetic electrons carry a
spurious high-frequency branch at low ``k_y`` (``|omega| ~ 200 c_s/a``, growing faster at
lower ``k_y`` and with finer ``theta``; the ``omega_H`` mode) that finite beta removes. On
39915 r/a 0.7 it grew alone at ``k_y = 0.1`` and drove the whole ES nonlinear run (#1484),
so an ES run whose ``n=1`` sits at ``k_y <= ES_MIN_KY`` is refused unless
``--allow-es-low-ky`` is given.

Writes ``<out>/<state slug>/r<r/a>/<field>/nonlinear/`` (native CGYRO files plus
``record.json``) and prints the measured wall time per ``a/c_s``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent


def _run_linear_module():
    spec = importlib.util.spec_from_file_location("run_linear", HERE / "run_linear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_local_summary(workdir: Path, local) -> None:
    """What the locality QA needs from the local input, so ``build_nonlinear.py`` can
    judge the run later without re-resolving the state."""
    norm = local.normalisation
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "local_summary.json").write_text(json.dumps({
        "r_over_a": local.r_over_a,
        "geometry": dict(local.geometry),
        "species": {k: [float(x) for x in v] for k, v in local.species.items()},
        "names": list(local.names),
        "normalisation": None if norm is None else {
            "gyroradius": norm.gyroradius, "minor_radius": norm.minor_radius,
            "b_unit": norm.b_unit, "sound_speed": norm.sound_speed,
            "electron_density": norm.electron_density,
            "electron_temperature": norm.electron_temperature},
    }, indent=1), encoding="utf-8")


# Lowest n=1 k_y rho_s an electrostatic nonlinear run may use: the ES omega_H branch was
# unstable at every k_y <= 0.2 checked on 39915 r/a 0.7 (job 768747, #1484).
ES_MIN_KY = 0.2


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", required=True, help="shot:time_s:lineage")
    parser.add_argument("--r-over-a", type=float, required=True)
    parser.add_argument("--field-model", default="em-aperp")
    parser.add_argument("--n-toroidal", type=int, default=16)
    parser.add_argument("--ky", type=float, default=0.12, help="k_y rho_s of n=1")
    parser.add_argument("--box-size", type=int, default=4)
    parser.add_argument("--n-radial", type=int, default=128)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-xi", type=int, default=24)
    parser.add_argument("--n-energy", type=int, default=8)
    parser.add_argument("--delta-t", type=float, default=0.002)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--print-step", type=int, default=100)
    parser.add_argument("--n-mpi", type=int, default=32)
    parser.add_argument("--n-omp", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="MAX_TIME=3: cost measurement")
    parser.add_argument("--local-only", action="store_true",
                        help="write local_summary.json for an existing run; run nothing")
    parser.add_argument("--gacode-home")
    parser.add_argument("--ti-te-ratio", default="policy")
    parser.add_argument("--amp", type=float,
                        help="CGYRO AMP, the initial amplitude of the n>0 modes (default 0.1): "
                             "a larger seed shortens the linear growth phase")
    parser.add_argument("--allow-es-low-ky", action="store_true",
                        help=f"run ES even though n=1 sits at k_y <= {ES_MIN_KY}")
    args = parser.parse_args(argv)
    if (args.field_model == "es" and args.ky <= ES_MIN_KY and not args.local_only
            and not args.allow_es_low_ky):
        raise SystemExit(
            f"electrostatic nonlinear run with n=1 at k_y={args.ky} <= {ES_MIN_KY}: the ES "
            "omega_H branch is unstable there (#1484); use --field-model em-aperp or pass "
            "--allow-es-low-ky")

    rl = _run_linear_module()
    rl._assert_checkout()
    from vaft.code.gacode.cgyro import CGYROConfig, prepare_cgyro_input, run_cgyro, stage_cgyro_case
    from vaft.process.transport_state import resolve_transport_state

    helpers = rl._lane_t_helpers()
    shot, time, lineage = args.state.split(":")
    shot, time = int(shot), float(time)
    labels = helpers.load_labels(args.labels)
    states, _ = helpers.enumerate_states(args.filedb, labels, [shot], [lineage])
    match = [s for s in states if abs(s[0].time_efit_s - time) < 6e-4]
    if not match:
        raise SystemExit(f"state {args.state} not found")
    key, label, source, cp_path, eq_path = match[0]
    ratio = args.ti_te_ratio if args.ti_te_ratio == "policy" else float(args.ti_te_ratio)
    ods = helpers.compose(helpers._load(eq_path), helpers._load(cp_path))
    state = resolve_transport_state(
        ods, key, efit_quality=label, quality_source=source, ti_te_ratio=ratio,
        inputs={"core_profiles": {"path": str(cp_path), "sha256": helpers._sha256(cp_path)},
                "equilibrium": {"path": str(eq_path), "sha256": helpers._sha256(eq_path)},
                "profile_mapped_on": "magnetics"})
    if not state.resolved:
        raise SystemExit(f"{key}: not resolved {state.reasons}")
    state_key = {"shot": key.shot, "time_efit_s": key.time_efit_s,
                 "efit_lineage": key.efit_lineage, "state_identity": state.identity}

    max_time = 3.0 if args.smoke else args.max_time
    config = CGYROConfig(
        nonlinear=True, field_model=args.field_model, n_toroidal=args.n_toroidal,
        ky=args.ky, box_size=args.box_size, n_radial=args.n_radial, n_theta=args.n_theta,
        n_xi=args.n_xi, n_energy=args.n_energy, delta_t=args.delta_t, max_time=max_time,
        print_step=args.print_step, n_mpi=args.n_mpi, n_omp=args.n_omp,
        restart=args.restart, home=args.gacode_home,
        extra_parameters={} if args.amp is None else {"AMP": args.amp},
    )
    local = prepare_cgyro_input(state.profile, args.r_over_a)
    workdir = (args.out / key.slug() / f"r{args.r_over_a:.2f}" / args.field_model
               / ("smoke" if args.smoke else "nonlinear"))
    if args.local_only:
        _write_local_summary(workdir, local)
        print(f"wrote {workdir / 'local_summary.json'}")
        return 0
    staged = stage_cgyro_case(local, workdir, config, state_key=state_key)
    result = run_cgyro(staged, config, check=False)
    native = result.outputs_native
    t_end = None if native is None or native.time is None else float(native.time[-1])
    record = {
        **state_key, "r_over_a": args.r_over_a, "field_model": args.field_model,
        "status": "executed" if result.executed else "failed",
        "runtime_status": result.runtime_status, "returncode": result.returncode,
        "elapsed_s": result.elapsed_s, "sim_time": t_end,
        "exit_message": None if native is None else native.exit_message,
        "errors": [] if native is None else list(native.errors),
        "n_mpi": args.n_mpi, "restart": args.restart,
        "provenance": {k: result.provenance.get(k) for k in
                       ("gacode_commit", "version", "resolution", "formalism", "input_sha256")},
    }
    history = workdir / "record_history.jsonl"
    with open(history, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=float) + "\n")
    (workdir / "record.json").write_text(json.dumps(record, indent=1, default=float),
                                         encoding="utf-8")
    if native is not None:
        native.write_json(workdir / "cgyro_outputs.json")
    _write_local_summary(workdir, local)
    if t_end and result.elapsed_s:
        start = 0.0 if not args.restart else None
        rate = result.elapsed_s / t_end if start == 0.0 else None
        print(json.dumps({"sim_time": t_end, "elapsed_s": result.elapsed_s,
                          "wall_s_per_a_cs": rate,
                          "core_hours_per_a_cs": None if rate is None
                          else rate * args.n_mpi * args.n_omp / 3600.0}))
    print(json.dumps({k: record[k] for k in ("status", "exit_message", "errors", "sim_time")}))
    return 0 if result.executed else 1


if __name__ == "__main__":
    raise SystemExit(main())
