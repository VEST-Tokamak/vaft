"""Direct GPEC -> PENTRC scan of the signed NTV torque over coil-group phasors (#1887).

Each candidate excitation ``c`` (one complex phasor per coil group, the #1178
convention ``I_k = A cos(n phi_k + delta)``, ``c = A exp(+i delta)``) gets its
own cell: DCON once per cell (a few seconds; ``euler.bin`` is identical in
every cell and is hard-linked to the first), GPEC with that cell's
``coil.in``, then :func:`vaft.code.gpec.run_pentrc` and the canonical reader.
Nothing is overwritten between candidates, and a cell with a ``result.json``
is not rerun, so an interrupted scan resumes.

Design (``--design``):
  probes      the K^2 probes of vaft.process.ntv_coil_response.quadratic_probe_design
  reference   c_ref plus 0.5x, 2x, -1x and exp(1.2i)x copies (invariance checks)
  held_out    --held-out random phasors (seeded), amplitudes 0.3-1.5 units
  confirm     the optima of the Q a previous --from scan qualifies (gpec_confirmation)

PENTRC tolerances matter more than anything else here: with the example's
defaults (atol_psi = rtol_psi = 1e-4) small and cancelling torques are off by
several percent and the quadratic model is wrongly rejected; ``--tolerance
converged`` (1e-6, rtol_xlmda 1e-7) is what qualified it.  Do not lower
``atol_xlmda`` below ~1e-10: PENTRC derives its energy tolerance from it and
the energy LSODE then stalls (#1929); the log is capped here for that reason.

Usage (the upstream DIII-D-like example, ~3-5 min per candidate on a laptop)::

    python workflow/gpec_ntv_coil_scan/scan_ntv_coil_torque.py --out runs/ntv1887 \\
        --design probes reference held_out --tolerance converged
    python workflow/gpec_ntv_coil_scan/scan_ntv_coil_torque.py --out runs/ntv1887 \\
        --design confirm --from runs/ntv1887/scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from vaft.code import gpec
from vaft.process.ntv_coil_response import (
    coil_phasors,
    fixed_amplitude_phase_extrema,
    hermitian_torque_matrix,
    quadratic_probe_design,
    two_group_phase_extrema,
)

GPEC_HOME = Path(os.environ.get("GPECHOME", Path.home() / "git" / "GPEC"))
EXAMPLE = GPEC_HOME / "docs" / "examples" / "DIIID_ideal_example"
TOLERANCES = {
    "default": {"atol_xlmda": 1e-9, "rtol_xlmda": 1e-6, "atol_psi": 1e-4, "rtol_psi": 1e-4},
    "converged": {"atol_xlmda": 1e-9, "rtol_xlmda": 1e-7, "atol_psi": 1e-6, "rtol_psi": 1e-6},
}
METHODS = ("fgar", "tgar", "pgar")
LOG_CAP_BYTES = 50_000_000


def sector_angles(coil_dir: Path, machine: str, group: str) -> np.ndarray:
    """Toroidal centroid of every coil of ``<machine>_<group>.dat`` [rad]."""
    lines = (coil_dir / f"{machine}_{group}.dat").read_text().split("\n")
    count = int(lines[0].split()[0])
    xyz = np.array([[float(v) for v in line.split()] for line in lines[1:] if line.strip()]).reshape(count, -1, 3)
    return np.array([np.angle(np.mean(np.exp(1j * np.arctan2(c[:, 1], c[:, 0])))) for c in xyz])


def sector_currents(c: np.ndarray, angles: dict, groups, n: int, unit: float) -> dict:
    """``I_k = Re(c_g exp(+i n phi_k)) * unit`` per group [A per turn]."""
    return {g: unit * np.real(c[i] * np.exp(1j * n * angles[g])) for i, g in enumerate(groups)}


def write_coil_in(template: Path, out: Path, coil_dir: Path, currents: dict, groups) -> None:
    """The template's coil.in with its data_dir and every ``coil_cur(k, j)`` replaced."""
    text = re.sub(r'data_dir\s*=\s*"[^"]*"', f'data_dir="{coil_dir}"', template.read_text(), count=1)
    names = {int(k): v for k, v in re.findall(r'coil_name\((\d+)\)\s*=\s*"([^"]+)"', text)}
    if sorted(names.values()) != sorted(groups):
        raise ValueError(f"{template} activates {sorted(names.values())}, the scan drives {sorted(groups)}")

    def replace(match):
        k, j = int(match.group(1)), int(match.group(2))
        return f"coil_cur({k},{j})= {currents[names[k]][j - 1]:.6f}"

    out.write_text(re.sub(r"coil_cur\((\d+),(\d+)\)\s*=\s*[-+0-9.eE]+", replace, text))


def templates(root: Path, case: Path, tolerance: str) -> Path:
    """The case directory with VAFT's packaged pentrc.in (the adapter's keys) at the chosen tolerances."""
    target = root / f"templates_{tolerance}"
    if target.is_dir():
        return target
    shutil.copytree(case, target)
    text = (Path(gpec.__file__).resolve().parents[2] / "data" / "gpec" / "pentrc.in").read_text()
    for key, value in TOLERANCES[tolerance].items():
        text, count = re.subn(rf"^(\s*{key}\s*=\s*)\S+", rf"\g<1>{value:g}", text, flags=re.M)
        if count != 1:
            raise ValueError(f"packaged pentrc.in does not declare {key} once")
    (target / "pentrc.in").write_text(text)
    return target


def candidates(design, groups, held_out: int, seed: int, previous: Path | None):
    k = len(groups)
    out = []
    if "probes" in design:
        out += [(label, "probe", c) for label, c in quadratic_probe_design(k)]
    if "reference" in design:
        ref = coil_phasors(np.ones(k), np.resize([0.0, 1.0, -0.8], k))  # the study's reference
        out += [("ref", "reference", ref), ("ref_x0.5", "invariance", 0.5 * ref), ("ref_x2", "invariance", 2 * ref),
                ("ref_neg", "invariance", -ref), ("ref_rot1.2", "invariance", np.exp(1.2j) * ref)]
    if "held_out" in design:
        rng = np.random.default_rng(seed)
        out += [(f"held{i}", "held_out", coil_phasors(rng.uniform(0.3, 1.5, k), rng.uniform(0, 2 * np.pi, k)))
                for i in range(held_out)]
    if "confirm" in design:
        rows = {r["sample_id"]: r for r in json.loads(previous.read_text())["samples"]}
        q = hermitian_torque_matrix({s: r["torque_n_m"]["fgar"] for s, r in rows.items()}, k)
        three = fixed_amplitude_phase_extrema(q, np.ones(k), grid_points=72, refine=3)
        out += [("opt_max", "gpec_confirmation", coil_phasors(np.ones(k), three["phases_max"])),
                ("opt_min", "gpec_confirmation", coil_phasors(np.ones(k), three["phases_min"]))]
        if k >= 2:
            two = two_group_phase_extrema(q[:2, :2], (1.0, 1.0))
            if not two["phase_independent"]:
                for name in ("max", "min"):
                    phases = np.zeros(k)
                    phases[1] = two[f"delta_phi_{name}"]
                    amplitudes = np.zeros(k)
                    amplitudes[:2] = 1.0
                    out.append((f"opt2_{name}", "gpec_confirmation", coil_phasors(amplitudes, phases)))
    return out


def run_cell(cell: Path, c, role, args, tpl: Path, angles, shared_euler: list) -> dict:
    done = cell / "result.json"
    if done.is_file():
        return json.loads(done.read_text())
    if cell.exists():
        shutil.rmtree(cell)
    cell.mkdir(parents=True)
    for source in tpl.iterdir():  # namelists, equilibrium and kinetic file; coil.in is written below
        if source.is_file() and source.name != "coil.in":
            shutil.copy2(source, cell / source.name)
    # GPEC reads pentrc.in itself (its kinetic table), so it sees the case's own;
    # run_pentrc then rewrites it from the packaged template in tpl.
    shutil.copy2(args.case / "pentrc.in", cell / "pentrc.in")
    currents = sector_currents(c, angles, args.groups, args.n, args.unit)
    write_coil_in(tpl / "coil.in", cell / "coil.in", args.coil_dir, currents, args.groups)
    record = {"sample_id": cell.name, "role": role, "c_re": np.real(c).tolist(), "c_im": np.imag(c).tolist()}
    for program in ("dcon", "gpec"):
        started = time.time()
        try:
            proc = subprocess.run([str(GPEC_HOME / "bin" / program)], cwd=cell, capture_output=True, text=True,
                                  timeout=args.timeout)
            ok = proc.returncode == 0 and "Normal termination" in proc.stdout
            (cell / f"{program}.log").write_text(proc.stdout[-200000:] + proc.stderr[-20000:])
        except subprocess.TimeoutExpired:
            ok = False
        record[f"{program}_ok"], record[f"{program}_seconds"] = ok, round(time.time() - started, 1)
        if not ok:
            break
        if program == "dcon":
            if shared_euler:
                (cell / "euler.bin").unlink()
                os.link(shared_euler[0], cell / "euler.bin")
            else:
                shared_euler.append(cell / "euler.bin")
    for dump in cell.glob("gpec_*_fun_n*.out"):  # field dumps PENTRC never reads, ~150 MB each
        dump.unlink()
    if record.get("gpec_ok"):
        config = gpec.GPECSuiteConfig(gpec_home=str(GPEC_HOME), templates_dir=str(tpl), timeout=args.timeout)
        options = gpec.PENTRCOptions(methods=METHODS, main_ion="deuterium", impurity="carbon",
                                     collision_operator="harmonic", bounce_harmonics=4)
        started = time.time()
        run = gpec.run_pentrc(cell, mode=args.n, options=options, kinetic_file=next(tpl.glob("*.gpeckf")),
                              config=config)
        record.update(pentrc_status=run.status, pentrc_reason=run.reason, pentrc_seconds=round(time.time() - started, 1))
        log = cell / "pentrc.log"
        if log.is_file() and log.stat().st_size > LOG_CAP_BYTES:  # a stalled LSODE repeats one warning (#1929)
            data = log.read_bytes()
            log.write_bytes(data[:20000] + b"\n... truncated %d bytes ...\n" % len(data) + data[-5000:])
        if run.status == "completed":
            with gpec.read_pentrc_output(run.outputs[0]) as output:
                record["torque_n_m"] = {m: float(gpec.torque_profile(output, m)[1][-1]) for m in METHODS}
    done.write_text(json.dumps(record, indent=1, default=str))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=EXAMPLE)
    parser.add_argument("--coil-dir", type=Path, default=GPEC_HOME / "coil")
    parser.add_argument("--machine", default="d3d")
    parser.add_argument("--groups", nargs="+", default=["c", "il", "iu"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--unit", type=float, default=1000.0, help="A per turn of one phasor unit")
    parser.add_argument("--design", nargs="+", default=["probes", "reference", "held_out"],
                        choices=["probes", "reference", "held_out", "confirm"])
    parser.add_argument("--held-out", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1887)
    parser.add_argument("--from", dest="previous", type=Path)
    parser.add_argument("--tolerance", choices=sorted(TOLERANCES), default="converged")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    tpl = templates(args.out, args.case, args.tolerance)
    angles = {g: sector_angles(args.coil_dir, args.machine, g) for g in args.groups}
    shared: list = []
    rows = []
    for sample, role, c in candidates(args.design, args.groups, args.held_out, args.seed, args.previous):
        row = run_cell(args.out / args.tolerance / sample, c, role, args, tpl, angles, shared)
        rows.append(row)
        print(sample, role, row.get("pentrc_status", "gpec failed"), row.get("torque_n_m", {}).get("fgar"), flush=True)
    summary = {
        "schema": "vaft.ntv_coil_scan/1",
        "case": {"directory": str(args.case), "groups": args.groups, "n": args.n, "phasor_unit_a_per_turn": args.unit,
                 "sector_angles": {g: a.tolist() for g, a in angles.items()}},
        "pentrc": {"tolerances": TOLERANCES[args.tolerance], "methods": list(METHODS)},
        "samples": rows,
    }
    (args.out / f"scan_{args.tolerance}.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
