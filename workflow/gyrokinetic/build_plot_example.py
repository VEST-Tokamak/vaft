"""The registered gyrokinetic plot family on one real Lane Y state (#1591 docs example).

For one Lane T state and a few surfaces, builds the IMAS inputs the registered plots
read, from products that already exist (nothing is run):

- ``gyrokinetics_local`` from CGYRO: the per-``k_y`` linear runs of ``run_linear.py``
  (each run's own ``gyrokinetics_local.json``) merged into one scan per surface with
  :func:`~vaft.machine_mapping.gyrokinetics.merge_linear_scan`;
- ``gyrokinetics_local`` from TGLF: the #1482 sensitivity run of the same state,
  surface and field model, through
  :func:`~vaft.machine_mapping.gyrokinetics.gyrokinetics_local_from_tglf`;
- ``core_transport`` from TGLF, one ODS per SAT rule across the surfaces, through Lane
  T's :func:`~vaft.machine_mapping.turbulence.core_transport_from_tglf`.

The local inputs are rebuilt from the state exactly as ``run_linear.py`` builds them,
and a TGLF run whose ``states.jsonl`` names a different ``state_identity`` is refused.

Writes the ODS JSON files and the figures (``gyrokinetics_overview`` per surface for
CGYRO and TGLF, a CGYRO-TGLF growth-rate overlay, ``turbulent_transport_overview``
over the SAT rules) under ``--out``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Optional

TGLF_FIELD = {"es": "es", "em-aperp": "em-bper"}


def _run_linear_module():
    spec = importlib.util.spec_from_file_location(
        "run_linear", Path(__file__).with_name("run_linear.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_input_tglf(path: Path) -> dict[str, Any]:
    """``input.tglf`` as ``{KEY: value}`` (the parameters the mapping records)."""
    raw: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value in (".true.", ".false."):
            raw[key.strip()] = value == ".true."
            continue
        try:
            raw[key.strip()] = float(value)
        except ValueError:
            raw[key.strip()] = value
    return raw


def _state(rl, args):
    from vaft.process.transport_state import resolve_transport_state

    helpers = rl._lane_t_helpers()
    shot, time, lineage = args.state.split(":")
    states, _ = helpers.enumerate_states(args.filedb, helpers.load_labels(args.labels),
                                         [int(shot)], [lineage])
    match = [s for s in states if abs(s[0].time_efit_s - float(time)) < 6e-4]
    if not match:
        raise SystemExit(f"state {args.state} not found")
    key, label, source, cp_path, eq_path = match[0]
    ods = helpers.compose(helpers._load(eq_path), helpers._load(cp_path))
    state = resolve_transport_state(
        ods, key, efit_quality=label, quality_source=source, ti_te_ratio="policy",
        inputs={"core_profiles": {"path": str(cp_path), "sha256": helpers._sha256(cp_path)},
                "equilibrium": {"path": str(eq_path), "sha256": helpers._sha256(eq_path)},
                "profile_mapped_on": "magnetics"})
    if not state.resolved:
        raise SystemExit(f"{key}: not resolved {state.reasons}")
    return key, state


def _tglf_run(args, key, sat: int, field: str, r_over_a: float, identity: str) -> Optional[Path]:
    root = args.sat_runs / f"{key.shot}-{round(key.time_efit_s * 1000)}-{key.efit_lineage}" \
        / f"tglf-sat{sat}-{TGLF_FIELD[field]}"
    states = root / "states.jsonl"
    if not states.is_file():
        return None
    for line in states.read_text().splitlines():
        row = json.loads(line)
        if row.get("state_identity") not in (None, identity):
            raise SystemExit(f"{root}: built from state {row['state_identity']}, not {identity}")
    run = root / str(key.shot) / key.efit_lineage / f"{round(key.time_efit_s * 1000):05d}" \
        / f"r{r_over_a:.2f}"
    return run if (run / "out.tglf.gbflux").is_file() else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--linear", type=Path, required=True, help="run_linear.py --out")
    parser.add_argument("--sat-runs", type=Path, required=True, help="#1482 sensitivity runs/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", default="39915:0.317:magnetics")
    parser.add_argument("--surfaces", type=float, nargs="*", default=[0.6, 0.7, 0.8])
    parser.add_argument("--field-model", default="em-aperp")
    parser.add_argument("--tglf-sat", type=int, default=2,
                        help="SAT rule of the TGLF gyrokinetics_local (its linear preset)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    from omas import ODS, load_omas_json, save_omas_json

    import vaft.omas
    from vaft.code.gacode.cgyro import prepare_cgyro_input
    from vaft.code.gacode.tglf.outputs import collect_tglf_outputs
    from vaft.machine_mapping.gyrokinetics import gyrokinetics_local_from_tglf, merge_linear_scan
    from vaft.machine_mapping.turbulence import core_transport_from_tglf
    from vaft.plot.style import save_figure

    rl = _run_linear_module()
    rl._assert_checkout()
    key, state = _state(rl, args)
    ms = f"{round(key.time_efit_s * 1000):05d}"
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{key.shot}_{ms}_{key.efit_lineage}"
    summary: dict[str, Any] = {"state": args.state, "state_identity": state.identity,
                               "field_model": args.field_model, "surfaces": {}}

    pairs: dict[int, list] = {sat: [] for sat in range(4)}
    for r_over_a in args.surfaces:
        local = prepare_cgyro_input(state.profile, r_over_a)
        surface = f"r{r_over_a:.2f}"
        entry: dict[str, Any] = {}

        runs = sorted((args.linear / str(key.shot) / key.efit_lineage / ms / surface
                       / args.field_model / "cgyro").glob("ky*/gyrokinetics_local.json"))
        if runs:
            cgyro = merge_linear_scan(load_omas_json(str(path)) for path in runs)
            save_omas_json(cgyro, str(args.out / f"gk_cgyro_{tag}_{surface}.json"))
            figure, _ = vaft.omas.plot_gyrokinetics_overview(cgyro)
            save_figure(figure, args.out / f"gk_overview_cgyro_{tag}_{surface}.png", dpi=150)
            entry["cgyro_runs"] = len(runs)

        for sat in range(4):
            run = _tglf_run(args, key, sat, args.field_model, r_over_a, state.identity)
            if run is not None:
                pairs[sat].append((local.tglf, collect_tglf_outputs(run)))
        run = _tglf_run(args, key, args.tglf_sat, args.field_model, r_over_a, state.identity)
        if run is not None:
            tglf = ODS()
            report = gyrokinetics_local_from_tglf(
                tglf, local.tglf, collect_tglf_outputs(run),
                parameters=read_input_tglf(run / "input.tglf"), time=key.time_efit_s)
            if report["written"]:
                save_omas_json(tglf, str(args.out / f"gk_tglf_sat{args.tglf_sat}_{tag}_{surface}.json"))
                figure, _ = vaft.omas.plot_gyrokinetics_overview(tglf)
                save_figure(figure, args.out / f"gk_overview_tglf_{tag}_{surface}.png", dpi=150)
                if runs:
                    figure, _ = vaft.omas.plot_gyrokinetics_spectrum_growth_rate(
                        {"CGYRO": cgyro, f"TGLF SAT{args.tglf_sat}": tglf})
                    save_figure(figure, args.out / f"gk_growth_cgyro_tglf_{tag}_{surface}.png", dpi=150)
            entry["tglf_skipped"] = report["skipped"]
        summary["surfaces"][surface] = entry

    transport = {}
    for sat, surfaces in pairs.items():
        if not surfaces:
            continue
        ods = ODS()
        report = core_transport_from_tglf(ods, surfaces, state.profile, time=key.time_efit_s)
        if report["written"]:
            transport[f"TGLF SAT{sat}"] = ods
            save_omas_json(ods, str(args.out / f"core_transport_tglf_sat{sat}_{tag}.json"))
    if transport:
        figure, _ = vaft.omas.plot_turbulent_transport_overview(transport)
        save_figure(figure, args.out / f"turbulent_transport_overview_{tag}.png", dpi=150)
    summary["core_transport"] = sorted(transport)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
