"""Issue #666: run NICE on noiseless Solovev-shaped synthetic diagnostics.

The analytic solution defines the LCFS and toroidal-current-density shape.
External diagnostics are evaluated independently with the axisymmetric filament
Green function; the analytic interior polynomial is deliberately not evaluated
outside the plasma.  The synthetic input includes NICE's native active-coil
response and no passive response, matching the already-conditioned native files.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from vaft.data.equilibrium import SolovevConstraint
from vaft.formula.constants import MU0
from vaft.formula.magnetics import project_poloidal_field
from vaft.process.electromagnetics import compute_point_response_matrices
from vaft.process.equilibrium import evaluate_solovev, solve_solovev_constraints

BASE = Path("/tmp/nice331-cocos-input-v2-20260914/focus")
ROOT = Path("/tmp/nice331-solovev-20260914")
EXE = Path("/tmp/nice-cocos-build-20260914/nice_recon")


def solovev_filaments(ip: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    r0, kappa, r_in = 0.40, 1.40, 0.22
    psi_boundary = (r0**2 - r_in**2) ** 2 / 4.0
    r_out = np.sqrt(r0**2 + 2.0 * np.sqrt(psi_boundary))
    z_top = kappa * np.sqrt(psi_boundary) / r0
    pprime = -2.0 * (1.0 + 1.0 / kappa**2) / MU0
    constraints = [
        SolovevConstraint(r_out, 0.0, "psi", psi_boundary),
        SolovevConstraint(r_in, 0.0, "psi", psi_boundary),
        SolovevConstraint(r0, z_top, "psi", psi_boundary),
        SolovevConstraint(r0, 0.0, "psi", 0.0),
        SolovevConstraint(r0, 0.0, "dpsi_dr", 0.0),
    ]
    model = solve_solovev_constraints(
        constraints, pprime=pprime, ffprime=0.0, rref=r0,
        psi_boundary=psi_boundary, f_boundary=0.12,
    )
    # Cell-centred quadrature.  Normalize the sign to the positive-Ip COCOS 11
    # convention used by this issue while retaining the Solovev j_phi shape.
    r_edges = np.linspace(r_in, r_out, 121)
    z_edges = np.linspace(-z_top, z_top, 141)
    r = (r_edges[:-1] + r_edges[1:]) / 2
    z = (z_edges[:-1] + z_edges[1:]) / 2
    rr, zz = np.meshgrid(r, z, indexing="ij")
    fields = evaluate_solovev(model, rr, zz)
    inside = (fields["psi"] >= -1e-12) & (fields["psi"] <= psi_boundary)
    cell_area = np.diff(r_edges)[:, None] * np.diff(z_edges)[None, :]
    raw_current = fields["j_phi"][inside] * cell_area[inside]
    currents = raw_current * (ip / raw_current.sum())
    meta = {
        "axis_m": [r0, 0.0], "r_in_m": r_in, "r_out_m": float(r_out),
        "z_top_m": float(z_top), "kappa": kappa,
        "psi_boundary_Wb_per_rad": float(psi_boundary),
        "quadrature_filaments": int(inside.sum()),
        "raw_integrated_current_A": float(raw_current.sum()),
        "normalized_current_A": float(currents.sum()),
    }
    return rr[inside], zz[inside], currents, meta


def plasma_response(channels, src_r, src_z, currents):
    values = []
    for channel in channels:
        geometry = channel["geometry"]
        point = geometry if channel["family"] == "bpol_probe" else geometry["positions"][0]
        psi, bz, br = compute_point_response_matrices(
            [point["r"]], [point["z"]], src_r, src_z,
            turns=np.ones(src_r.size), components=("psi", "bz", "br"),
        )
        if channel["family"] == "bpol_probe":
            value = project_poloidal_field(br[0], bz[0], geometry["poloidal_angle"])
        else:
            value = psi[0]
        values.append(float(value @ currents))
    return np.asarray(values)


def run_case(name: str, changes: dict[str, object], measurements, channels):
    case = ROOT / name
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(BASE / "input", case / "input")
    shutil.copy2(BASE / "nice_case_manifest.json", case / "nice_case_manifest.json")
    (case / "output").mkdir()
    (case / "restart").mkdir()
    tree = ET.parse(case / "input/param.xml")
    root = tree.getroot()
    for key, value in changes.items():
        node = root.find(key)
        if node is None:
            node = ET.SubElement(root, key)
        node.text = str(value)
    tree.write(case / "input/param.xml")
    for family, filename, sign in (
        ("bpol_probe", "Bprobes_meas.txt", 1.0),
        ("flux_loop", "fluxloops_meas.txt", -1.0),
    ):
        rows = [(c, y) for c, y in zip(channels, measurements) if c["family"] == family]
        text = [str(len(rows))]
        text += [f"{sign*y:.17g} {c['uncertainty']:.17g} 0" for c, y in rows]
        (case / "input" / filename).write_text("\n".join(text) + "\n")
    with (case / "stdout.log").open("w") as stdout, (case / "stderr.log").open("w") as stderr:
        proc = subprocess.run([EXE], cwd=case, stdout=stdout, stderr=stderr, timeout=120)
    provenance = json.loads((case / "nice_case_manifest.json").read_text())
    synthetic_iter = iter(measurements)
    for channel in provenance["diagnostic_channels"]:
        if channel["enabled"] and channel["family"] in ("bpol_probe", "flux_loop"):
            value = float(next(synthetic_iter))
            channel["synthetic_replaced_value"] = value
            channel["value"] = value
            channel["conditioned_value"] = value
    provenance["synthetic_diagnostics"] = True
    provenance.update(process_returncode=proc.returncode, process_timed_out=False)
    (case / "nice_case_manifest.json").write_text(json.dumps(provenance, indent=2, allow_nan=True))
    log = (case / "stdout.log").read_text(errors="replace")
    from vaft.code.nice import collect_nice_outputs
    collected = collect_nice_outputs(case)
    recovery = {}
    for family, measured_file, computed_file in (
        ("bpol_probe", "Bprobes_meas.txt", "vacth_Bprobes_comp.txt"),
        ("flux_loop", "fluxloops_meas.txt", "vacth_fluxloops_comp.txt"),
    ):
        measured = np.loadtxt(case / "input" / measured_file, skiprows=1, ndmin=2)
        computed = np.loadtxt(case / "output" / computed_file, ndmin=1)
        # Native flux-loop output has the opposite sign to the input-file convention.
        sign = -1.0 if family == "flux_loop" else 1.0
        normalized = (sign * computed - measured[:, 0]) / measured[:, 1]
        recovery[family] = {
            "rms_sigma": float(np.sqrt(np.mean(normalized**2))),
            "max_abs_sigma": float(np.max(np.abs(normalized))),
        }
    return {
        "name": name, "returncode": proc.returncode,
        "converged": collected.converged,
        "scientifically_usable": collected.scientifically_usable,
        "termination_reason": collected.termination_reason,
        "objective": collected.objective,
        "vacth_diagnostic_recovery": recovery,
        "iterations": len(re.findall(r"relresidX=", log)),
        "last_relative_residual": (re.findall(r"relresidX=([^\s]+)", log) or [None])[-1],
        "vacth_ip_A": (re.findall(r"Ip from TH = ([^\s]+)", log) or [None])[-1],
        "barycenter": (re.findall(r"_Bary=\(([^)]+)\)", log) or [None])[-1],
        "invalid": bool(re.search(r"plasma valid = 0|IsValid\\(\\)=0", log)),
        "cost_lines": re.findall(r"costM?=[^\n]+", log)[-5:],
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "nice_case_manifest.json").read_text())
    channels = [
        c for c in manifest["diagnostic_channels"]
        if c["enabled"] and c["family"] in ("bpol_probe", "flux_loop")
    ]
    ip = float(np.loadtxt(BASE / "input/Ip_B0.txt", ndmin=1)[0])
    src_r, src_z, currents, truth = solovev_filaments(ip)
    plasma = plasma_response(channels, src_r, src_z, currents)
    native_active = np.asarray([
        next(a["native"] for a in manifest["active_response_audit"] if a["ods_path"] == c["ods_path"])
        for c in channels
    ])
    measurements = plasma + native_active
    results = [
        run_case("vacth_only", {"algoVacTHonly": 1}, measurements, channels),
        run_case("full_default", {}, measurements, channels),
        run_case("direct_10", {"iterMaxDirInitRecon": 10}, measurements, channels),
    ]
    payload = {
        "description": "Noiseless Solovev-shaped plasma plus NICE-native active-coil response",
        "base_case": str(BASE), "executable": str(EXE), "truth": truth,
        "plasma_signal_norms": {
            family: float(np.linalg.norm([y for c, y in zip(channels, plasma) if c["family"] == family]))
            for family in ("bpol_probe", "flux_loop")
        },
        "results": results,
    }
    (ROOT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
