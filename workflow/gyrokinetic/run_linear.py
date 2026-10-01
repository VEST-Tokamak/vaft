"""Linear CGYRO ky scans on Lane T's good states, beside TGLF on the same input (#1354).

For each state (shot, time, EFIT lineage) and surface, the profile is resolved through
Lane T's :func:`vaft.process.transport_state.resolve_transport_state` -- the same
resolution the TGLF atlas and the #1482 sensitivity used -- and projected once with
:func:`~vaft.code.gacode.cgyro.prepare_cgyro_input`. Then, per ky and field model:

* a linear CGYRO run (``N_TOROIDAL=1``, one run per ky), and
* a linear TGLF run at the **same** ky on the **same** local input
  (``USE_TRANSPORT_MODEL=F``, ``KY=<ky>``), which isolates the eigenvalue solver from
  the saturation rule. The SAT0-3 spectra already on disk from #1482 are overlaid later
  by ``build_linear.py``, not re-run.

This script needs Lane T's code at run time (``vaft.process.transport_state`` and
``workflow/transport_atlas/run_tglf.py``), so it runs from a checkout that has both
branches -- see ``README.md``. It refuses to start when ``vaft`` resolves outside the
checkout it was launched from (the editable-install import trap).

Output tree (``--out``)::

    run_manifest.json
    runs.jsonl                                one line per (state, surface, field, ky, code)
    <state slug>/r0.70/<field>/cgyro/ky0.300/ native CGYRO directory + record.json
    <state slug>/r0.70/<field>/tglf/ky0.300/  native TGLF directory + record.json
    <state slug>/r0.70/local.json             the local input both codes were given

Each run is resumable: a record whose input hash matches and whose status is
``solved`` is reused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: The #1482 sensitivity states (#1453): shot, EFIT time [s], lineage.
GOOD_STATES = (
    (39916, 0.321, "magnetics"),
    (39915, 0.317, "magnetics"),
    (40330, 0.321, "magnetics"),
    (42962, 0.332, "magnetics"),
    (42962, 0.332, "electron_kinetic"),
)
DEFAULT_SURFACES = (0.6, 0.7, 0.8)
#: Ion scale plus the electron-scale extension: #1482 found the fastest TGLF growth
#: at k_y rho_s > 1 on 74 % of surfaces.
DEFAULT_KY = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _assert_checkout() -> None:
    import vaft

    location = Path(vaft.__file__).resolve()
    if REPO not in location.parents:
        raise SystemExit(
            f"vaft resolves to {location}, not this checkout ({REPO}). Put the checkout "
            "first on sys.path (see README.md) before running."
        )


def _lane_t_helpers():
    """Lane T's FileDB helpers, imported from this checkout by path (not a package)."""
    path = REPO / "workflow" / "transport_atlas" / "run_tglf.py"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: this needs the Lane T branch merged in")
    spec = importlib.util.spec_from_file_location("lane_t_run_tglf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_backend(args) -> Any:
    if args.backend != "slurm":
        return None
    from vaft.code.slurm import SlurmBackend

    # One job per run, like Lane T, but sized for MPI: --mem is mandatory on tdst,
    # otherwise the default reserves a whole node.
    return SlurmBackend(partition=args.partition, account=args.account, mode=args.slurm_mode,
                        max_wait=args.max_wait,
                        extra_args=(f"--mem={int(args.mem_mb)}M",))


def cgyro_config(args, field_model: str, ky: float, backend: Any):
    from vaft.code.gacode.cgyro import CGYROConfig

    return CGYROConfig(
        field_model=field_model, ky=float(ky), n_energy=args.n_energy, n_xi=args.n_xi,
        n_theta=args.n_theta, n_radial=args.n_radial, max_time=args.max_time,
        delta_t=args.delta_t, delta_t_method=args.delta_t_method, freq_tol=args.freq_tol,
        n_mpi=args.n_mpi, n_omp=1, backend=backend, timeout=args.timeout,
        home=args.gacode_home,
    )


def tglf_linear_config(field_model: str, ky: float, args):
    from vaft.code.gacode.cgyro import TGLF_FIELD_MODEL
    from vaft.code.gacode.tglf import TGLFConfig

    field = TGLF_FIELD_MODEL[field_model]
    return TGLFConfig(
        sat_rule=args.tglf_sat_rule, use_transport_model=False,
        use_bper=field != "es", use_bpar=field == "em-bper-bpar",
        extra_parameters={"KY": float(ky), "NMODES": 4},
        home=args.gacode_home, timeout=600.0,
    )


def _record(path: Path) -> Optional[dict]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def run_cgyro_job(job: dict) -> dict:
    """One CGYRO run; any failure becomes this run's record, never the scan's end."""
    workdir = Path(job["workdir"])
    try:
        return _run_cgyro_job(job, workdir)
    except Exception as error:  # noqa: BLE001 - one failed run must not stop the scan
        record = {**job["meta"], "code": "cgyro", "status": "failed",
                  "runtime_status": "error", "errors": [f"{type(error).__name__}: {error}"]}
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "record.json").write_text(json.dumps(record, indent=1, default=str),
                                             encoding="utf-8")
        return record


def _run_cgyro_job(job: dict, workdir: Path) -> dict:
    from omas import ODS

    from vaft.code.gacode.cgyro import run_cgyro, stage_cgyro_case
    from vaft.machine_mapping.gyrokinetics import gyrokinetics_local_from_cgyro
    from vaft.omas import save as save_ods

    staged = stage_cgyro_case(job["local"], workdir, job["config"], state_key=job["state_key"])
    previous = _record(workdir / "record.json")
    if previous and previous.get("input_sha256") == staged.provenance["input_sha256"]:
        decayed = previous.get("status") == "decayed" or any(
            "Underflow in calculation of frequency error" in str(e)
            for e in previous.get("errors") or ())
        # A decayed (stable) mode is a result too: rerunning it reproduces the underflow.
        if previous.get("status") == "solved" or decayed:
            return {**previous, "status": "decayed" if decayed else "solved", "cached": True}
    result = run_cgyro(staged, job["config"], check=False)
    native = result.outputs_native
    omega = None if native is None else native.frequency_ion_negative
    record = {
        **job["meta"], "code": "cgyro",
        "status": ("solved" if result.ok
                   else "decayed" if native is not None and native.decayed else "failed"),
        "qualified": bool(result.qualified),
        "runtime_status": result.runtime_status, "returncode": result.returncode,
        "elapsed_s": result.elapsed_s,
        "exit_message": None if native is None else native.exit_message,
        "errors": [] if native is None else list(native.errors),
        "gamma": None if not result.ok else float(native.final_growth_rate[0]),
        "omega_ion_negative": None if omega is None else float(omega[0]),
        "omega_native": None if not result.ok else float(native.final_frequency[0]),
        "ion_direction": None if native is None else native.ion_direction,
        "sim_time": None if native is None or native.time is None else float(native.time[-1]),
        "input_sha256": staged.provenance["input_sha256"],
        "provenance": {k: result.provenance.get(k) for k in
                       ("gacode_commit", "version", "resolution", "formalism", "state_key")},
        "cached": False,
    }
    # The record is written before the optional products, so a mapping failure cannot
    # leave the previous run's record standing for this run.
    (workdir / "record.json").write_text(json.dumps(record, indent=1, default=float),
                                         encoding="utf-8")
    if native is not None:
        native.write_json(workdir / "cgyro_outputs.json")
        if result.ok:
            ods = ODS(consistency_check=False)
            report = gyrokinetics_local_from_cgyro(
                ods, job["local"], native, provenance=result.provenance,
                time=job["state_key"]["time_efit_s"])
            if report["written"]:
                save_ods(ods, workdir / "gyrokinetics_local.json")
            record["imas_skipped"] = report["skipped"]
            (workdir / "record.json").write_text(json.dumps(record, indent=1, default=float),
                                                 encoding="utf-8")
    return record


def run_tglf_job(job: dict) -> dict:
    from vaft.code.gacode.tglf import run_tglf
    from vaft.code.gacode.tglf.inputs import TGLFInputs, tglf_parameters, write_input_tglf

    workdir = Path(job["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "record.json").unlink(missing_ok=True)  # never leave a previous verdict
    parameters = tglf_parameters(job["local"].tglf, job["config"])
    written = write_input_tglf(parameters, workdir / "input.tglf")
    staged = TGLFInputs(workdir=workdir, files=(written,), local=job["local"].tglf,
                        input_tglf=written, parameters=parameters,
                        provenance={"state_key": job["state_key"]})
    try:
        result = run_tglf(staged, job["config"], check=False)
    except Exception as error:  # noqa: BLE001
        record = {**job["meta"], "code": "tglf", "status": "failed",
                  "errors": [f"{type(error).__name__}: {error}"]}
        (workdir / "record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return record
    eigenvalues = tglf_single_mode_eigenvalues(workdir)
    native = result.outputs_native
    errors = [] if native is None else list(native.errors)
    record = {
        **job["meta"], "code": "tglf",
        "status": "solved" if eigenvalues and not errors else "failed",
        "runtime_status": result.runtime_status, "returncode": result.returncode,
        "errors": errors,
        # (frequency, growth rate) per mode in c_s/a, TGLF's sign: ion direction < 0.
        "eigenvalues": eigenvalues,
    }
    (workdir / "record.json").write_text(json.dumps(record, indent=1, default=float),
                                         encoding="utf-8")
    return record


def tglf_single_mode_eigenvalues(workdir: Path) -> list[list[float]]:
    """``[[omega, gamma], ...]`` per mode from a ``USE_TRANSPORT_MODEL=F`` run.

    TGLF's single-ky branch writes no spectrum file: it prints ``(wr,wi):`` per mode
    (``tglf/src/tglf.f90``), and the launcher captures stdout in ``out.tglf.run``.
    Sorted by growth rate, most unstable first.
    """
    for name in ("out.tglf.run", "tglf.log"):
        path = Path(workdir) / name
        if not path.is_file():
            continue
        modes = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("(wr,wi):"):
                values = line.split(":", 1)[1].split()
                try:
                    modes.append([float(values[0]), float(values[1])])
                except (IndexError, ValueError):
                    continue
        if modes:
            return sorted(modes, key=lambda mode: -mode[1])
    return []


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--states", nargs="*", default=None,
                        help="shot:time_s:lineage; default the five #1482 states")
    parser.add_argument("--surfaces", type=float, nargs="*", default=list(DEFAULT_SURFACES))
    parser.add_argument("--ky", type=float, nargs="*", default=list(DEFAULT_KY))
    parser.add_argument("--field-models", nargs="*", default=["es", "em-aperp"])
    parser.add_argument("--ti-te-ratio", default="policy")
    parser.add_argument("--n-energy", type=int, default=8)
    parser.add_argument("--n-xi", type=int, default=24)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-radial", type=int, default=8)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--delta-t", type=float, default=0.01)
    parser.add_argument("--delta-t-method", type=int, default=1)
    parser.add_argument("--freq-tol", type=float, default=1e-3)
    parser.add_argument("--n-mpi", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2,
                        help="CGYRO runs at once; workers * n_mpi cores in flight")
    parser.add_argument("--gacode-home")
    parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    parser.add_argument("--slurm-mode", default="auto")
    parser.add_argument("--partition", default="lowpri-short")
    parser.add_argument("--account")
    parser.add_argument("--mem-mb", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=4 * 3600.0)
    parser.add_argument("--max-wait", type=float, default=24 * 3600.0)
    parser.add_argument("--tglf-sat-rule", type=int, default=0,
                        help="irrelevant to a linear eigenvalue; recorded only")
    parser.add_argument("--skip-tglf", action="store_true")
    parser.add_argument("--max-jobs", type=int, help="stop after this many CGYRO jobs (smoke)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    _assert_checkout()
    import vaft
    from vaft.code.gacode.cgyro import gacode_revision, prepare_cgyro_input
    from vaft.process.transport_state import TransportStateKey, resolve_transport_state

    helpers = _lane_t_helpers()
    labels = helpers.load_labels(args.labels)
    wanted = GOOD_STATES if not args.states else tuple(
        (int(s.split(":")[0]), float(s.split(":")[1]), s.split(":")[2]) for s in args.states)
    shots = sorted({shot for shot, _, _ in wanted})
    lineages = sorted({lineage for _, _, lineage in wanted})
    states, counts = helpers.enumerate_states(args.filedb, labels, shots, lineages)
    chosen = []
    for key, label, source, cp_path, eq_path in states:
        for shot, time, lineage in wanted:
            if key.shot == shot and key.efit_lineage == lineage and abs(key.time_efit_s - time) < 6e-4:
                chosen.append((key, label, source, cp_path, eq_path))
    missing = [w for w in wanted if not any(
        c[0].shot == w[0] and c[0].efit_lineage == w[2] and abs(c[0].time_efit_s - w[1]) < 6e-4
        for c in chosen)]
    if missing:
        print(f"WARNING: states not found in the FileDB/labels: {missing}", file=sys.stderr)

    ratio = args.ti_te_ratio if args.ti_te_ratio == "policy" else float(args.ti_te_ratio)
    backend = build_backend(args)
    args.out.mkdir(parents=True, exist_ok=True)
    loaded: dict[Path, Any] = {}

    def cached(path: Path) -> Any:
        if path not in loaded:
            loaded[path] = helpers._load(path)
        return loaded[path]

    cgyro_jobs: list[dict] = []
    tglf_jobs: list[dict] = []
    surfaces_written = []
    for key, label, source, cp_path, eq_path in chosen:
        ods = helpers.compose(cached(eq_path), cached(cp_path))
        state = resolve_transport_state(
            ods, key, efit_quality=label, quality_source=source, ti_te_ratio=ratio,
            inputs={"core_profiles": {"path": str(cp_path), "sha256": helpers._sha256(cp_path)},
                    "equilibrium": {"path": str(eq_path), "sha256": helpers._sha256(eq_path)},
                    "profile_mapped_on": "magnetics"})
        if not state.resolved:
            print(f"{key}: not resolved {state.reasons}", file=sys.stderr)
            continue
        state_key = {"shot": key.shot, "time_efit_s": key.time_efit_s,
                     "efit_lineage": key.efit_lineage, "state_identity": state.identity}
        for r_over_a in args.surfaces:
            try:
                local = prepare_cgyro_input(state.profile, r_over_a)
            except ValueError as error:
                print(f"{key} r/a={r_over_a}: {error}", file=sys.stderr)
                continue
            absent = local.tglf.check_tglf_requirements()
            if absent:
                # Refused here, as prepare_cgyro_case would: a NaN must not reach input.cgyro.
                print(f"{key} r/a={r_over_a}: local input lacks {', '.join(absent)}",
                      file=sys.stderr)
                continue
            surface_dir = args.out / key.slug() / f"r{r_over_a:.2f}"
            surface_dir.mkdir(parents=True, exist_ok=True)
            (surface_dir / "local.json").write_text(json.dumps({
                "state": state.summary(), "r_over_a": r_over_a,
                "geometry": dict(local.geometry), "names": list(local.names),
                "species": {k: np.asarray(v).tolist() for k, v in local.species.items()},
                "betae_unit": local.betae_unit, "nu_ee": local.nu_ee,
                "lambda_star": local.lambda_star, "ipccw": local.ipccw, "btccw": local.btccw,
                "provenance": local.provenance,
            }, indent=1, default=str), encoding="utf-8")
            surfaces_written.append(str(surface_dir))
            for field_model in args.field_models:
                for ky in args.ky:
                    meta = {**state_key, "efit_quality": label, "r_over_a": r_over_a,
                            "field_model": field_model, "ky": float(ky)}
                    cgyro_jobs.append({
                        "workdir": str(surface_dir / field_model / "cgyro" / f"ky{ky:.3f}"),
                        "local": local, "state_key": state_key, "meta": meta,
                        "config": cgyro_config(args, field_model, ky, backend)})
                    if not args.skip_tglf:
                        tglf_jobs.append({
                            "workdir": str(surface_dir / field_model / "tglf" / f"ky{ky:.3f}"),
                            "local": local, "state_key": state_key, "meta": meta,
                            "config": tglf_linear_config(field_model, ky, args)})

    if args.max_jobs:
        cgyro_jobs = cgyro_jobs[: args.max_jobs]
        tglf_jobs = tglf_jobs[: args.max_jobs]
    print(f"{len(chosen)} states, {len(surfaces_written)} surfaces, "
          f"{len(cgyro_jobs)} CGYRO + {len(tglf_jobs)} TGLF runs", flush=True)

    records: list[dict] = []
    if not args.dry_run:
        for job in tglf_jobs:  # ~1 s each, serial on the driver's own core
            records.append(run_tglf_job(job))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
            futures = {pool.submit(run_cgyro_job, job): job for job in cgyro_jobs}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                record = future.result()
                records.append(record)
                print(f"[{done}/{len(cgyro_jobs)}] {futures[future]['workdir']} "
                      f"{record.get('status')} {record.get('exit_message')} "
                      f"gamma={record.get('gamma')} {record.get('elapsed_s')}", flush=True)
        # Rewritten, not appended: a resume returns every record (cached or new).
        with open(args.out / "runs.jsonl", "w", encoding="utf-8") as index:
            for record in records:
                index.write(json.dumps({k: v for k, v in record.items() if k != "provenance"},
                                       default=float) + "\n")

    manifest = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv[:1] + list(argv if argv is not None else sys.argv[1:]),
        "vaft": vaft.__file__, "vaft_version": vaft.__version__,
        "vaft_git": helpers._git_sha(REPO),
        "gacode_revision": gacode_revision(cgyro_config(args, "es", 0.3, None)),
        "states_requested": [list(w) for w in wanted], "states_missing": [list(m) for m in missing],
        "enumeration": counts, "cgyro_jobs": len(cgyro_jobs), "tglf_jobs": len(tglf_jobs),
        "resolution": cgyro_config(args, "es", 0.3, None).resolution(),
        "n_mpi": args.n_mpi,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str),
                                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
