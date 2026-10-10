"""Build the NUBEAM fast-ion reference the analytic closure is checked against (#1606, Lane L).

Runs the packaged VEST NUBEAM case (``vaft.code.nubeam.packaged_vest_case``),
or reads an existing run directory, and writes one small JSON record with what
:func:`vaft.process.kinetic_closure.fast_ion_slowing_down_estimate` needs to
reproduce NUBEAM's plasma -- the zone grid and volumes, n_e, T_e, the thermal
ion species, the beam energy and mass, and the **birth rate per zone** -- next
to NUBEAM's own fast-ion density and energy density and its power balance.

The birth rate comes from the deposition markers of the **last** step (gyro-centre
R, Z in cm, weight = particles per step), mapped to ``rho_tor_norm`` through the
case's G-EQDSK (rho = sqrt of the normalised toroidal flux, NUBEAM's own radial
coordinate).  ``sbedep`` is not used: it counts the electrons ionisation frees,
and a charge-exchange birth frees none, so on VEST it carries only ~20 % of
the births.

Usage::

    NUBEAMHOME=~/git/nubeam/local python3 workflow/kinetic_closure/nubeam_fast_ion_reference.py \\
        --run --out test/data/nubeam/vest_fast_ion_reference.json
    python3 workflow/kinetic_closure/nubeam_fast_ion_reference.py --workdir RUN_DIR --out FILE

``--run`` uses the notebook's "reduced" size (5 x 1 ms steps, 2000 markers,
about 20 minutes serial) in a temporary directory that is removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

QE = 1.602176634e-19


def _step_of(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _variables(path: Path, names) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as data:
        out = {}
        for name in names:
            if name not in data.variables:
                continue
            variable = data.variables[name]
            if variable.dtype.kind in "SU" or variable.dtype == "S1":
                out[name] = [str(s).strip() for s in netCDF4.chartostring(variable[:])]
            else:
                out[name] = np.asarray(variable[:], dtype=float)
        return out


def births_per_zone(birth_file: Path, gfile: Path, rho_edges: np.ndarray, step_s: float):
    """Birth rate [1/s] in each rho_tor_norm zone, and the fraction born outside the LCFS."""
    from scipy.interpolate import RectBivariateSpline

    from vaft.data import read_geqdsk

    import netCDF4

    with netCDF4.Dataset(birth_file) as data:
        key = next(k for k in data.variables if k.startswith("bs_wght_") and k.endswith("_MCBEAM"))
        species = key[len("bs_wght_"): -len("_MCBEAM")]
        r = np.asarray(data[f"bs_rgc_{species}_MCBEAM"][:], dtype=float) / 100.0
        z = np.asarray(data[f"bs_zgc_{species}_MCBEAM"][:], dtype=float) / 100.0
        weight = np.asarray(data[key][:], dtype=float) / step_s
        energy = np.asarray(data[f"bs_einj_{species}_MCBEAM"][:], dtype=float)
    g = read_geqdsk(str(gfile))
    nw, nh = int(g.get("NW")), int(g.get("NH"))
    r_grid = g.get("RLEFT") + np.linspace(0.0, g.get("RDIM"), nw)
    z_grid = g.get("ZMID") - 0.5 * g.get("ZDIM") + np.linspace(0.0, g.get("ZDIM"), nh)
    psi = np.asarray(g.get("PSIRZ"), dtype=float)
    psi = psi if psi.shape == (nh, nw) else psi.T
    psi_n = (RectBivariateSpline(z_grid, r_grid, psi).ev(z, r) - g.get("SIMAG")) / (g.get("SIBRY") - g.get("SIMAG"))
    q = np.asarray(g.get("QPSI"), dtype=float)
    grid = np.linspace(0.0, 1.0, q.size)
    flux = np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(grid))))
    inside = (psi_n >= 0.0) & (psi_n <= 1.0)
    # anti-alias: spatial interpolation over psi_N, not time -- no sample rate to reduce
    rho = np.sqrt(np.interp(psi_n[inside], grid, flux) / flux[-1])
    rate, _ = np.histogram(rho, bins=rho_edges, weights=weight[inside])
    return rate, float(weight[~inside].sum() / weight.sum()), species, np.unique(np.round(energy, 3))


def build_reference(workdir: Path, *, step_s: float, steps: int, size: str) -> dict:
    from vaft.code import nubeam
    from vaft.data.atomic import STANDARD_ATOMIC_WEIGHTS, parse_species

    result = nubeam.collect_nubeam_outputs(workdir)
    native = result.outputs_native
    runid = native.runid
    state = _variables(workdir / f"{runid}.cdf",
                       ["rho", "vol", "S_name", "ns", "Ts", "q_S", "kvolt_nbi", "power_nbi"])
    changes = _variables(workdir / "state_changes.cdf",
                         ["nbeami", "eperp_beami", "epll_beami", "pbe", "pbi", "pbth"])
    births = sorted(workdir.glob(f"{runid}_birth_cpu*"), key=_step_of)
    rate, outside, species, energies = births_per_zone(births[-1], workdir / "equilibrium.gfile",
                                                        state["rho"], step_s)
    names = state["S_name"]
    ns, ts = np.atleast_2d(state["ns"]), np.atleast_2d(state["Ts"])
    ions = []
    for k, name in enumerate(names[1:], 1):
        parsed = parse_species(name)
        # q_S is stored with NUBEAM's own elementary charge (Z = 1.0000146 here)
        charge = float(abs(state["q_S"][k]) / QE)
        charge = float(round(charge)) if abs(charge - round(charge)) < 1e-3 else charge
        ions.append({"label": name, "Z": charge,
                     "A": STANDARD_ATOMIC_WEIGHTS[parsed.element], "density_m3": ns[k].tolist()})
    beam_mass = STANDARD_ATOMIC_WEIGHTS[parse_species(species).element]
    n_fast = np.atleast_2d(changes["nbeami"])[0]
    mean_energy = (np.atleast_2d(changes["eperp_beami"])[0] + np.atleast_2d(changes["epll_beami"])[0]) * 1e3
    deposited = float(np.sum(rate) * float(np.ravel(state["kvolt_nbi"])[0]) * 1e3 * QE)
    balance = {block.species: {name: float(value) for name, value in block.entries.items()}
               for block in native.power_balance}
    return {
        "provenance": {
            "case": "vaft/data/nubeam/vest_case (packaged VEST NUBEAM case)",
            "run_size": size, "steps": steps, "step_s": step_s,
            "birth_markers": f"{births[-1].name} (last step), gyro-centre, rho_tor_norm from the case G-EQDSK",
            "births_outside_lcfs_fraction": outside,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "issue": "#1606 (analytic fast-ion closure vs NUBEAM)",
        },
        "rho_edges": np.asarray(state["rho"]).tolist(),
        "volume_edges_m3": np.asarray(state["vol"]).tolist(),
        "n_e_m3": ns[0].tolist(),
        "T_e_eV": (ts[0] * 1e3).tolist(),
        "ions": ions,
        "beam": {"species": species, "A_b": beam_mass, "Z_b": 1.0, "energy_eV": float(energies.max()),
                 "energies_eV": energies.tolist(), "power_W": float(np.sum(state["power_nbi"]))},
        "birth_rate_per_zone_s": rate.tolist(),
        "nubeam": {
            "n_fast_m3": n_fast.tolist(),
            "W_fast_J_m3": (n_fast * mean_energy * QE).tolist(),
            "deposited_W": deposited,
            "electron_heating_W": float(np.sum(changes["pbe"])),
            "ion_heating_W": float(np.sum(changes["pbi"])),
            "thermalisation_W": float(np.sum(changes["pbth"])),
            "power_balance": balance,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="store_true", help="run the packaged VEST case first")
    source.add_argument("--workdir", type=Path, help="an existing NUBEAM run directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    step_s, steps, size = 1e-3, 5, "reduced (5 x 1 ms, 2000 markers)"
    temporary = None
    try:
        if args.run:
            from vaft.code import nubeam

            case = nubeam.packaged_vest_case()
            runid = nubeam.inputf_runid((case.input_dir / "inputf").read_text(encoding="utf-8"))
            config = nubeam.NUBEAMConfig(runid=runid, repeat_count=f"{steps}x{step_s:g}", nptcls=2000)
            # NUBEAM's path budget (~95 characters) forces a short directory.
            temporary = Path(tempfile.mkdtemp(prefix="vaft-nb-", dir="/tmp"))
            result = nubeam.run_nubeam_case(case.input_dir, gfile=case.gfile, workdir=temporary, config=config)
            if not result.ok:
                raise SystemExit(f"NUBEAM failed; see {temporary}")
            workdir = temporary
        else:
            workdir = args.workdir
        record = build_reference(workdir, step_s=step_s, steps=steps, size=size)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
