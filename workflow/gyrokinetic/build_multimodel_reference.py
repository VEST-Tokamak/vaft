"""Remap retained #39915 SAT native outputs into one multi-model ODS.

Requires the original per-SAT state.json and native run directories, copied
read-only from the retained Lane Y SAT tree. SI scales and radial coordinates
come from that state's recorded normalization and core_transport report.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omas import ODS, load_omas_json, save_omas_json

from vaft.code.gacode.cgyro import cgyro_input_from_tglf, collect_cgyro_outputs
from vaft.code.gacode.result_identity import parameters, plain
from vaft.code.gacode.tglf import collect_tglf_outputs
from vaft.code.gacode.tglf.inputs import TGLFNormalisation, read_local_input
from vaft.machine_mapping.gyrokinetics import core_transport_from_cgyro
from vaft.machine_mapping.turbulence import core_transport_from_tglf


def build(root, out, *, nonlinear=None, window=(100, 1037)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import vaft.omas

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ods = ODS()
    reports = []
    locals_by_r = {}
    state_id = None
    profile = None
    for sat in range(4):
        directory = root / f"sat{sat}"
        state = json.loads((directory / "state.json").read_text())
        rows = state["surfaces"]
        pairs = []
        grid = state["core_transport"]["surfaces"]
        radii = [v["r_over_a"] for v in grid]
        rho = [v["rho_tor_norm"] for v in grid]
        profile = SimpleNamespace(rmin=np.r_[0.0, radii, 1.0], rho=np.r_[0.0, rho, 1.0])
        for row in rows:
            r = row["r_over_a"]
            info = row["local"]
            n = info["normalisation"]
            # Exact inverse of retained flux unit definitions, not a new physical closure.
            ne = n["gamma_gb_m2_s"] / (n["c_s_m_s"] * (n["rho_s_m"] / n["a_m"]) ** 2)
            te = n["q_gb_W_m2"] / n["gamma_gb_m2_s"]
            norm = TGLFNormalisation(
                electron_density=ne,
                electron_temperature=te,
                sound_speed=n["c_s_m_s"],
                gyroradius=n["rho_s_m"],
                minor_radius=n["a_m"],
                b_unit=n["b_unit_T"],
            )
            np.testing.assert_allclose(
                [norm.energy_flux, norm.particle_flux],
                [n["q_gb_W_m2"], n["gamma_gb_m2_s"]],
                rtol=1e-12,
            )
            local = read_local_input(
                directory / f"r{r:.2f}",
                normalisation=norm,
                names=info["species"],
                provenance={
                    "vexb_shear": {
                        "kind": "unavailable",
                        "reason": "retained state has no rotation diagnostic",
                    }
                },
            )
            pairs.append((local, collect_tglf_outputs(directory / f"r{r:.2f}")))
            locals_by_r[r] = local
        state_id = state["state_identity"]
        report = core_transport_from_tglf(
            ods, pairs, profile, time=state["time_efit_s"], state_identity=state_id
        )
        native_values = [v["electron_energy_flux_W_m2"] for v in grid]
        m = report["model"]
        np.testing.assert_allclose(
            ods[f"core_transport.model.{m}.profiles_1d.0.electrons.energy.flux"],
            native_values,
            rtol=1e-12,
        )
        reports.append(report)
    if nonlinear:
        native = collect_cgyro_outputs(nonlinear)
        local = cgyro_input_from_tglf(locals_by_r[0.7])
        # Verify native simulation physics before carrying a retained state identity.
        from vaft.code.gacode.result_identity import read_native_input

        effective, _ = read_native_input(nonlinear, "CGYRO")
        for key in ("Z", "MASS", "DENS", "TEMP", "DLNNDR", "DLNTDR"):
            np.testing.assert_allclose(
                local.species[key],
                [effective[f"{key}_{i + 1}"] for i in range(local.n_species)],
                rtol=1e-10,
            )
        for key, value in local.geometry.items():
            np.testing.assert_allclose(value, effective[key], rtol=1e-10)
        reports.append(
            core_transport_from_cgyro(
                ods,
                [(local, native, window)],
                profile,
                time=0.317,
                state_identity=state_id,
            )
        )
    save_omas_json(ods, str(out / "multimodel.json"))
    loaded = load_omas_json(str(out / "multimodel.json"))
    for m in range(len(ods["core_transport.model"])):
        assert parameters(
            loaded[f"core_transport.model.{m}.code.parameters"]
        ) == parameters(ods[f"core_transport.model.{m}.code.parameters"])
    fig, axis = vaft.omas.plot_turbulent_transport_profile_energy_flux(
        loaded, time=0.317, time_tolerance=0.0
    )
    axis.legend(loc="upper left", fontsize=7, ncol=2)
    fig.set_size_inches(9, 6)
    axis.set_title(
        "TGLF QL vs CGYRO NL selected-window transport; convergence unverified"
    )
    fig.savefig(out / "multimodel_energy_flux.png", dpi=140)
    plt.close(fig)
    (out / "validation.json").write_text(
        json.dumps(
            plain(
                {
                    "reports": reports,
                    "native_SI_flux_match": True,
                    "roundtrip_identity_match": True,
                }
            ),
            indent=2,
        )
        + "\n"
    )
    return loaded


def linear_reference(sat_root, linear_root, out):
    """Fresh native-to-IMAS linear mapping, with explicit source-run provenance."""
    import matplotlib.pyplot as plt

    import vaft.omas
    from vaft.code.gacode.result_identity import encode, read_native_input
    from vaft.machine_mapping.gyrokinetics import (
        gyrokinetics_local_from_cgyro,
        gyrokinetics_local_from_tglf,
        merge_linear_scan,
    )

    root, linear_root, out = Path(sat_root), Path(linear_root), Path(out)
    state = json.loads((root / "sat0/state.json").read_text())
    info = next(row["local"] for row in state["surfaces"] if row["r_over_a"] == 0.7)
    n = info["normalisation"]
    ne = n["gamma_gb_m2_s"] / (n["c_s_m_s"] * (n["rho_s_m"] / n["a_m"]) ** 2)
    norm = TGLFNormalisation(
        electron_density=ne,
        electron_temperature=n["q_gb_W_m2"] / n["gamma_gb_m2_s"],
        sound_speed=n["c_s_m_s"],
        gyroradius=n["rho_s_m"],
        minor_radius=n["a_m"],
        b_unit=n["b_unit_T"],
    )
    local = read_local_input(
        root / "sat0/r0.70",
        normalisation=norm,
        names=info["species"],
        provenance={"vexb_shear": {"kind": "unavailable"}},
    )
    renamed = cgyro_input_from_tglf(local)
    scans = []
    runs = []
    for path in sorted(linear_root.glob("ky*")):
        native = collect_cgyro_outputs(path)
        raw, _ = read_native_input(path, "CGYRO")
        for key, values in renamed.species.items():
            np.testing.assert_allclose(
                values,
                [raw[f"{key}_{i + 1}"] for i in range(renamed.n_species)],
                rtol=1e-10,
            )
        for key, value in renamed.geometry.items():
            np.testing.assert_allclose(value, raw[key], rtol=1e-10)
        ods = ODS()
        report = gyrokinetics_local_from_cgyro(
            ods,
            renamed,
            native,
            time=0.317,
            provenance={
                "parameters": raw,
                "state_identity": state["state_identity"],
                "version": native.version,
            },
        )
        if report["written"]:
            scans.append(ods)
            runs.extend(
                parameters(ods["gyrokinetics_local.code.parameters"])["local_runs"]
            )
    merged = merge_linear_scan(scans)
    record = parameters(merged["gyrokinetics_local.code.parameters"])
    record["local_runs"] = runs
    merged["gyrokinetics_local.code.parameters"] = encode(
        record, merged["gyrokinetics_local.code.parameters"]
    )
    entries = {"CGYRO": merged}
    for sat in range(4):
        directory = root / f"sat{sat}/r0.70"
        native = collect_tglf_outputs(directory)
        raw, _ = read_native_input(directory, "TGLF")
        ods = ODS()
        gyrokinetics_local_from_tglf(
            ods,
            local,
            native,
            parameters=raw,
            time=0.317,
            provenance={
                "state_identity": state["state_identity"],
                "version": native.version,
            },
        )
        entries[f"TGLF SAT{sat}"] = ods
        save_omas_json(ods, str(out / f"gk_tglf_sat{sat}.json"))
    save_omas_json(merged, str(out / "gk_cgyro.json"))
    fig, axis = vaft.omas.plot_gyrokinetics_spectrum_growth_rate(
        entries,
        comparison_intent="controlled_sensitivity",
        controlled_differences=("model.solver", "model.effective_configuration"),
    )
    axis.legend(loc="upper right", fontsize=7, ncol=2)
    fig.set_size_inches(9, 6)
    fig.savefig(out / "linear_growth_comparison.png", dpi=140)
    plt.close(fig)
    from vaft.process.gyrokinetic_comparison import (
        assess_comparison,
        local_result_signature,
    )

    assessment = {
        name: dict(
            assess_comparison(
                local_result_signature(merged),
                local_result_signature(ods),
                intent="controlled_sensitivity",
                controlled=("model.solver", "model.effective_configuration"),
            ).axes
        )
        for name, ods in entries.items()
        if name != "CGYRO"
    }
    (out / "linear_comparison.json").write_text(
        json.dumps(plain(assessment), indent=2) + "\n"
    )
    return entries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nonlinear", type=Path)
    p.add_argument("--linear", type=Path)
    a = p.parse_args()
    build(a.runs, a.out, nonlinear=a.nonlinear)
    if a.linear:
        linear_reference(a.runs, a.linear, a.out)
