"""Build the offline MHD-stability sample of Tutorial 05 from real solver runs.

The tutorial reads only what this script leaves in
``vaft/data/samples/39915/mhd_stability/``: IMAS-compatible OMAS JSON products
and one manifest. Every product is computed here, through the same VAFT
adapters a live tutorial run uses, from the EFIT reconstruction of the packaged
39915 reference sample (``vaft.omas.sample_ods(39915)``, the reconstruction
Tutorials 03 and 04 use) at 319 ms, exported to a g-file::

    EFIT slice --CHEASE--> stable refined equilibrium
               --CHEASE (Ip x 2.8, total-current constraint)--> synthetic unstable equilibrium
    each --DCON (+ MATCH, solutions.bin), mpsi 256 and 512--> mhd_linear
    stable --RDCON, mpsi 256 / 512 / 1024--> mhd_linear + ntms
    stable --DCON --> GPEC, UP alone and LOW alone, n = 1, 200 A--> mhd_linear

Native solver workdirs (``euler.bin``, ``solutions.bin``, ``psi_in.bin``,
``gpec_*_output_n1.nc``, ...) live only in a scratch directory and are deleted
at the end: they exist to produce the mapped products, not to be read by the
tutorial.  Each mapped product is saved, read back, and re-checked against the
quantity it was accepted on before the set is published.

Usage (needs CHEASE and the GPEC suite)::

    PYTHONPATH=$PWD python workflow/tutorial_05_mhd_samples/build_mhd_stability_sample.py \\
        --chease ~/git/chease/src-f90/chease --gpec-home ~/git/GPEC

The run is idempotent: when the published manifest records the same input
fingerprint (source g-file, this script, solver revisions, configuration) it
does nothing unless ``--force``.  A new set is built in a staging directory and
swapped in only after every acceptance check has passed, so a failed or partial
run never replaces a validated sample.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_SAMPLE = REPOSITORY / "vaft" / "data" / "samples" / "39915" / "omas.json.gz"
DEFAULT_OUTPUT = REPOSITORY / "vaft" / "data" / "samples" / "39915" / "mhd_stability"
PARENT_MANIFEST = "../manifest.yaml"
SHOT, TIME_MS, N_TOR = 39915, 319, 1

#: The synthetic unstable variant: CHEASE re-solves the same boundary at this
#: multiple of the EFIT plasma current (ncscal=2), lowering q_edge to just
#: below 3. Chosen in the 2026-11-11 scan: pressure x2..x8 and current peaking
#: 0.5..4 stay stable on this reconstruction; Ip x2.0 is unstable only in a
#: narrow window (q_edge 4.07, neighbours at 3.87 and 4.28 stable); Ip x2.7..x3.0
#: (q_edge 3.01..2.71) is unstable at mpsi 256 and 512 with the same dominant m.
#: x2.8 sits inside that window: W_t ~ -1.2, m = 3 -- an m/n = 3/1 external kink.
UNSTABLE_CURRENT_SCALE = 2.8
UNSTABLE_LABEL = "high_ip"
DCON_RESOLUTIONS = (256, 512)
RDCON_RESOLUTIONS = (256, 512, 1024)
GPEC_RESOLUTION = 256
COIL_SETS = ("UP", "LOW")
COIL_AMPLITUDE_A = 200.0
COIL_PHASE_DEG = 0.0
#: Short scratch root: the GPEC suite reads paths into 128-character Fortran strings.
SCRATCH_PREFIX = "/tmp/vaft-t05s-"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _revision(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "commit": None}
    root = path if path.is_dir() else path.parent
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


class AcceptanceError(RuntimeError):
    """A product failed the check it is published on; nothing is swapped in."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


# --------------------------------------------------------------------- CHEASE
def export_source_geqdsk(scratch: Path) -> Path:
    """The sample ODS's EFIT slice at TIME_MS, as the g-file the solvers read."""
    import vaft
    from vaft.data.eqdsk import from_omas, write_geqdsk

    ods = vaft.omas.load(SOURCE_SAMPLE)
    times = np.asarray(ods["equilibrium.time"], dtype=float)
    index = int(np.argmin(np.abs(times - TIME_MS * 1e-3)))
    _require(abs(times[index] * 1e3 - TIME_MS) < 0.5, f"the sample has no slice at {TIME_MS} ms")
    target = scratch / f"g0{SHOT}.{TIME_MS:05d}"
    write_geqdsk(from_omas(ods, index), target)
    return target


def run_chease(scratch: Path, executable: Path, source: Path) -> dict[str, Any]:
    from vaft.code.chease import CHEASEConfig
    from vaft.code.chease_scan import EquilibriumVariation, scan_chease
    from vaft.data import read_geqdsk
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    config = CHEASEConfig(executable=str(executable), create_plot=False)
    variations = [
        EquilibriumVariation(label="stable"),
        EquilibriumVariation(label=UNSTABLE_LABEL, plasma_current_scale=UNSTABLE_CURRENT_SCALE),
    ]
    cases = scan_chease(source, variations, config=config, workdir=scratch / "chease")
    out: dict[str, Any] = {}
    for case in cases:
        label = case.variation.label
        _require(case.converged, f"CHEASE did not converge for {label}: {case.error}")
        refined = Path(case.result.refined_geqdsk)
        geqdsk = read_geqdsk(refined)
        q = np.asarray(geqdsk["QPSI"], dtype=float)
        descriptors = derive_global_descriptors(as_equilibrium(refined)).values

        def value(name: str) -> float | None:
            item = descriptors.get(name)
            return None if item is None or item.value is None else float(item.value)

        out[label] = {
            "geqdsk": refined,
            "ods": geqdsk.to_omas(),
            "variation": {"plasma_current_scale": case.variation.plasma_current_scale},
            "comparison_to_efit": {k: float(v) for k, v in dict(case.result.comparison).items()},
            "q_axis": float(q[0]),
            "q_edge": float(q[-1]),
            "q95": value("q95"),
            "beta_n": value("beta_n"),
            "ip_a": value("ip"),
            "target_psin": case.target_psin,
            "chease_log": str(refined.parent / "chease.log"),
        }
    return out


# ----------------------------------------------------------------- GPEC suite
def _suite_config(gpec_home: Path, modules: tuple[str, ...], mpsi: int, con_flag: bool | None, coil=None):
    from vaft.code import gpec

    options = {}
    if coil is not None:
        options["gpec"] = gpec.IdealGPECOptions(coil_specs=(coil,))
    return gpec.GPECSuiteConfig(
        gpec_home=gpec_home,
        modules=modules,
        modes=(N_TOR,),
        mpsi=mpsi,
        verify_outputs=True,
        timeout=3600.0,
        dcon=gpec.DCONOptions(termbycross_flag=False, con_flag=con_flag),
        **options,
    )


def _run_case(root: Path, geqdsk: Path, config) -> Any:
    from vaft.code import gpec

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    staged = root / f"g0{SHOT}.{TIME_MS:05d}"
    shutil.copy2(geqdsk, staged)
    return gpec.run_gpec_suite_case(
        gpec.GPECCaseInputs(shot=SHOT, time_ms=TIME_MS, geqdsk=staged, workdir=root / "run"), config
    )


def run_dcon(scratch: Path, gpec_home: Path, equilibria: dict[str, Any]) -> dict[str, Any]:
    from vaft.code import gpec
    from vaft.omas.vest_upstream import build_mhd_linear_ods

    products: dict[str, Any] = {}
    for label, record in equilibria.items():
        for mpsi in DCON_RESOLUTIONS:
            root = scratch / f"dcon-{label}-{mpsi}"
            result = _run_case(root, record["geqdsk"], _suite_config(gpec_home, ("dcon",), mpsi, False))
            [run] = result.records
            _require(run.status in ("completed", "stable"), f"DCON {label} mpsi {mpsi}: {run.status} {run.reason}")
            problems = gpec.validate_dcon_result(run.workdir, N_TOR, verify_outputs=True)
            _require(not problems, f"DCON {label} mpsi {mpsi}: {'; '.join(problems)}")
            native = gpec.read_dcon_output(run.workdir, mode=N_TOR)
            _require(native.eigenfunction is not None, f"DCON {label} mpsi {mpsi}: no solutions.bin eigenfunction")
            ods, manifest = build_mhd_linear_ods(
                shot=SHOT, time_values=[TIME_MS], workdir=root / "run", modules=("dcon",), modes=(N_TOR,)
            )
            products[f"dcon_{label}_mpsi{mpsi}"] = {
                "ods": ods,
                "check": {
                    "W_t": float(np.real(native.total1)),
                    "W_p": float(np.real(native.plasma1)),
                    "W_v": float(np.real(native.vacuum1)),
                    "m_pol_dominant": int(native.m_pol_dominant),
                    "edge_treatment": native.edge_treatment,
                    "solver_status": run.status,
                },
                "stage_manifest_status": manifest.get("status"),
            }
    # Acceptance (#142: the sign is the verdict, two resolutions must agree).
    for label, expect_negative in (("stable", False), (UNSTABLE_LABEL, True)):
        checks = [products[f"dcon_{label}_mpsi{m}"]["check"] for m in DCON_RESOLUTIONS]
        signs = {c["W_t"] < 0 for c in checks}
        _require(signs == {expect_negative}, f"DCON {label}: W_t signs {[c['W_t'] for c in checks]}")
        if expect_negative:
            dominant = {c["m_pol_dominant"] for c in checks}
            _require(len(dominant) == 1, f"DCON {label}: dominant m differs between resolutions {dominant}")
    return products


def run_rdcon(scratch: Path, gpec_home: Path, geqdsk: Path) -> dict[str, Any]:
    from vaft.code import gpec
    from vaft.machine_mapping.mhd_linear import extract_rdcon_stability
    from vaft.omas.vest_upstream import build_mhd_linear_ods

    products: dict[str, Any] = {}
    for mpsi in RDCON_RESOLUTIONS:
        root = scratch / f"rdcon-{mpsi}"
        result = _run_case(root, geqdsk, _suite_config(gpec_home, ("rdcon",), mpsi, None))
        [run] = result.records
        _require(run.status in ("completed", "stable"), f"RDCON mpsi {mpsi}: {run.status} {run.reason}")
        native = gpec.read_pest3_matching_output(run.workdir, solver="rdcon", mode=N_TOR)
        ods, _ = build_mhd_linear_ods(
            shot=SHOT, time_values=[TIME_MS], workdir=root / "run", modules=("rdcon",), modes=(N_TOR,)
        )
        [row] = extract_rdcon_stability(ods)
        diagonal = native.delta_prime_diagonal()
        _require(len(row["surfaces"]) == len(diagonal) > 0, f"RDCON mpsi {mpsi}: surfaces not mapped")
        products[f"rdcon_stable_mpsi{mpsi}"] = {
            "ods": ods,
            "check": {
                "solver_status": run.status,
                "surfaces": [
                    {"m": int(s["m"]), "psi_n": float(s["psi_n"]), "delta_prime_real": float(s["delta_prime_real"])}
                    for s in diagonal
                ],
            },
        }
    return products


def run_gpec(scratch: Path, gpec_home: Path, geqdsk: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from vaft.code import gpec
    from vaft.machine_mapping.coils_non_axisymmetric_geometry import CoilExcitation, load_vest_3d_coil_config
    from vaft.omas import compute_gpec_resonant_table
    from vaft.omas.vest_upstream import build_gpec_ideal_ods

    coil_config = load_vest_3d_coil_config()
    products: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for name in COIL_SETS:
        excitation = CoilExcitation.from_mode(
            name, COIL_AMPLITUDE_A, N_TOR, COIL_PHASE_DEG,
            sector_angles_deg=coil_config.coil_sets[name].sector_angles_deg,
        )
        coil = gpec.CoilInputSpec.from_excitation(excitation)
        root = scratch / f"gpec-{name}"
        result = _run_case(root, geqdsk, _suite_config(gpec_home, ("dcon", "gpec"), GPEC_RESOLUTION, False, coil))
        runs = {r.module: r for r in result.records}
        _require(runs["gpec"].status == "completed", f"GPEC {name}: {runs['gpec'].status} {runs['gpec'].reason}")
        native = gpec.read_gpec_netcdf(runs["gpec"].workdir, mode=N_TOR)
        _require(native.profile is not None and native.profile.Phi_res is not None,
                 f"GPEC {name}: no total resonant field (Phi_res) in the profile output")
        ods, _ = build_gpec_ideal_ods(shot=SHOT, time_values=[TIME_MS], workdir=root / "run", modes=(N_TOR,))
        table = compute_gpec_resonant_table(ods)
        native_total = np.abs(np.asarray(native.profile.Phi_res)).ravel()
        mapped_total = np.array([abs(r["Phi_res"]) for r in table["rows"]])
        products[f"gpec_{name}"] = {
            "ods": ods,
            "check": {
                "currents_a": [float(c) for c in excitation.currents_a],
                "dcon_W_t": float(np.real(gpec.read_dcon_output(runs["dcon"].workdir, mode=N_TOR).total1)),
                "energy_vacuum": float(native.control.energy_vacuum),
                "energy_plasma": float(native.control.energy_plasma),
                "n_rational_surfaces": len(table["rows"]),
                "abs_phi_res_native": [float(x) for x in native_total],
                "abs_phi_res_derived": [float(x) for x in mapped_total],
                "abs_phi_res_vacuum": [float(abs(r["Phi_res_vacuum"])) for r in table["rows"]],
            },
        }
    # The packaged template's con_flag=t, kept as evidence: GPEC then writes
    # only the vacuum resonant field.  Not published as a product.
    root = scratch / "gpec-UP-con_flag_t"
    excitation = CoilExcitation.from_mode("UP", COIL_AMPLITUDE_A, N_TOR, COIL_PHASE_DEG,
                                          sector_angles_deg=coil_config.coil_sets["UP"].sector_angles_deg)
    result = _run_case(root, geqdsk, _suite_config(gpec_home, ("dcon", "gpec"), GPEC_RESOLUTION, True,
                                                   gpec.CoilInputSpec.from_excitation(excitation)))
    runs = {r.module: r for r in result.records}
    baseline: dict[str, Any] = {"gpec_status": runs["gpec"].status}
    if runs["gpec"].status == "completed":
        native = gpec.read_gpec_netcdf(runs["gpec"].workdir, mode=N_TOR)
        baseline.update(
            dcon_W_t=float(np.real(gpec.read_dcon_output(runs["dcon"].workdir, mode=N_TOR).total1)),
            energy_vacuum=float(native.control.energy_vacuum),
            total_resonant_field_written=bool(native.profile is not None and native.profile.Phi_res is not None),
        )
    evidence["con_flag_true_UP"] = baseline
    return products, evidence


# ------------------------------------------------------------ verify + publish
def _reload_and_recheck(path: Path, key: str, check: dict[str, Any]) -> str:
    import vaft
    from vaft.machine_mapping.mhd_linear import extract_dcon_stability, extract_rdcon_stability

    ods = vaft.omas.load(path)
    if key.startswith("dcon_"):
        [row] = extract_dcon_stability(ods)
        energy = float(ods["mhd_linear.time_slice.0.toroidal_mode.0.energy_perturbed"])
        _require(np.sign(energy) == np.sign(check["W_t"]), f"{key}: W_t sign lost on reload")
        m_dom = int(ods["mhd_linear.time_slice.0.toroidal_mode.0.m_pol_dominant"])
        _require(m_dom == check["m_pol_dominant"], f"{key}: dominant m lost on reload")
        xi = np.asarray(ods["mhd_linear.time_slice.0.toroidal_mode.0.plasma.displacement_perpendicular.real"])
        _require(xi.ndim == 2 and xi.size > 0, f"{key}: eigenfunction lost on reload")
        _ = row
    elif key.startswith("rdcon_"):
        [row] = extract_rdcon_stability(ods)
        got = [complex(s["delta_prime"]).real for s in row["surfaces"]]
        want = [s["delta_prime_real"] for s in check["surfaces"]]
        _require(np.allclose(got, want, rtol=1e-12, atol=0.0), f"{key}: Delta' changed on reload")
    elif key.startswith("gpec_"):
        from vaft.omas import compute_gpec_resonant_table

        table = compute_gpec_resonant_table(ods)
        got = [abs(r["Phi_res"]) for r in table["rows"]]
        _require(np.allclose(got, check["abs_phi_res_derived"], rtol=1e-9), f"{key}: Phi_res changed on reload")
    return "passed"


def fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_sample_sha256": _sha256(SOURCE_SAMPLE),
        "source_time_ms": TIME_MS,
        "script_sha256": _sha256(Path(__file__)),
        "chease": _revision(Path(args.chease).resolve()),
        "gpec": _revision(Path(args.gpec_home).resolve()),
        "configuration": {
            "unstable_current_scale": UNSTABLE_CURRENT_SCALE,
            "dcon_resolutions": list(DCON_RESOLUTIONS),
            "rdcon_resolutions": list(RDCON_RESOLUTIONS),
            "gpec_resolution": GPEC_RESOLUTION,
            "coil_sets": list(COIL_SETS),
            "coil_amplitude_a": COIL_AMPLITUDE_A,
            "coil_phase_deg": COIL_PHASE_DEG,
            "dcon": {"con_flag": False, "termbycross_flag": False},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chease", required=True, help="CHEASE executable")
    parser.add_argument("--gpec-home", required=True, help="GPEC installation root ($GPECHOME)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-scratch", action="store_true", help="keep the native solver workdirs")
    parser.add_argument("--force", action="store_true", help="rebuild even if the fingerprint is unchanged")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    inputs = fingerprint(args)
    manifest_path = output / "manifest.yaml"
    if manifest_path.is_file() and not args.force:
        current = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if current.get("inputs") == json.loads(json.dumps(inputs)) and current.get("status") == "validated":
            print(f"{output}: validated sample already built from these inputs; nothing to do (--force rebuilds)")
            return 0

    import vaft

    scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX))
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        print("CHEASE ...", flush=True)
        source = export_source_geqdsk(scratch)
        chease = run_chease(scratch, Path(args.chease).expanduser(), source)
        print("DCON ...", flush=True)
        dcon = run_dcon(scratch, Path(args.gpec_home).expanduser(), chease)
        print("RDCON ...", flush=True)
        rdcon = run_rdcon(scratch, Path(args.gpec_home).expanduser(), chease["stable"]["geqdsk"])
        print("GPEC ...", flush=True)
        gpec_products, evidence = run_gpec(scratch, Path(args.gpec_home).expanduser(), chease["stable"]["geqdsk"])

        products: dict[str, Any] = {}
        for label, record in chease.items():
            name = f"equilibrium_chease_{label}.json.gz"
            vaft.omas.save(record["ods"], staging / name)
            products[f"chease_{label}"] = {
                "path": name,
                "ids": ["equilibrium"],
                "stage": "chease",
                "check": {k: v for k, v in record.items() if k not in ("geqdsk", "ods", "chease_log")},
            }
        for group in (dcon, rdcon, gpec_products):
            for key, record in group.items():
                name = f"{key}.json.gz"
                vaft.omas.save(record["ods"], staging / name)
                ids = sorted({k.split(".")[0] for k in record["ods"].flat()} - {"dataset_description"})
                products[key] = {"path": name, "ids": ids, "stage": key.split("_")[0], "check": record["check"]}
        for key, record in products.items():
            path = staging / record["path"]
            record["validation"] = "passed" if key.startswith("chease_") else _reload_and_recheck(path, key, record["check"])
            record["sha256"] = _sha256(path)
            record["size"] = path.stat().st_size

        manifest = {
            "schema_version": 1,
            "kind": "tutorial-05-mhd-stability-sample",
            "status": "validated",
            "shot": SHOT,
            "time_ms": TIME_MS,
            "parent_manifest": PARENT_MANIFEST,
            "built": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "build_seconds": round(time.monotonic() - started),
            "regenerate": "PYTHONPATH=$PWD python workflow/tutorial_05_mhd_samples/build_mhd_stability_sample.py "
                          "--chease <CHEASE executable> --gpec-home <GPECHOME>",
            "source": {
                "sample": "vaft/data/samples/39915/omas.json.gz",
                "time_ms": TIME_MS,
                "lineage": "EFIT reconstruction of the 39915 reference sample (pipeline until-efit-v5), "
                           "slice at 319 ms, exported with vaft.data.eqdsk.from_omas; the legacy 2023 "
                           "g-file vaft/data/efit/g039915.00319 is no longer the source",
            },
            "vaft": _revision(REPOSITORY),
            "inputs": json.loads(json.dumps(inputs)),
            "conventions": {
                "psi": "psi_N normalised poloidal flux, 0 on axis, 1 at the CHEASE boundary",
                "cocos": "CHEASE g-files keep the source signs (output_cocos='input'; see <case>.cocos_export.json "
                         "in a fresh build); ODS psi in Wb per IMAS DD",
                "dcon_energy": "W_t, W_p, W_v are DCON's normalised least-stable eigenvalues (not joules); "
                               "the sign of W_t is the ideal verdict (#142)",
                "dcon_eigenfunction": "xi.grad(psi) per poloidal harmonic m in Hamada coordinates, arbitrary "
                                      "eigenvector normalisation, from MATCH's solutions.bin",
                "delta_prime": "classical (diagonal) RDCON Delta' in ntms.mode[].deltaw[name=classical]; full "
                               "complex matrix in mhd_linear.code.parameters",
                "coil_excitation": "I_k = A cos(n phi_k + phase) per sector, A per turn, 20 turns per sector "
                                   "(vaft.machine_mapping.coils_non_axisymmetric_geometry.CoilExcitation.from_mode)",
                "gpec_field": "J b.grad(psi) on (psi_N, m), GPEC Hamada coordinates; total resonant flux derived "
                              "by vaft.omas.compute_gpec_resonant_table, vacuum pair copied from GPEC",
                "n_tor": N_TOR,
            },
            "evidence": evidence,
            "products": products,
            "not_distributed": [
                "euler.bin", "psi_in.bin", "solutions.bin", "vacuum.bin", "dcon_output_n1.nc",
                "rdcon_output_n1.nc", "gpec_*_output_n1.nc", "CHEASE native outputs",
            ],
            "redistribution": "Derived from VEST shot 39915 data already redistributed with VAFT; "
                              "same terms as the parent sample.",
        }
        (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    except Exception:
        print(f"build failed; scratch kept at {scratch}, staging at {staging}", file=sys.stderr)
        raise
    # Swap in only now: everything above passed.
    backup = output.with_name(output.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    staging.rename(output)
    if backup.exists():
        shutil.rmtree(backup)
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    total = sum(p.stat().st_size for p in output.iterdir())
    print(f"published {len(products)} products, {total / 1e6:.1f} MB, to {output}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    raise SystemExit(main())
