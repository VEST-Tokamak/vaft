"""PORTALS (TGLF + NEO) synthetic closed loop with a manufactured source (#1588 C1b).

Runs in the MITIM interpreter; imports MITIM, NumPy/SciPy and the standard library.

1. Strip the heat sources; run PORTALS from the true profile with no BO iteration:
   evaluation 0 is TGLF + NEO at the truth, giving a/L_Te and Q_e^tr at each radius.
2. Manufacture qohme so that PORTALS's own targets equal Q_e^tr there (targets only,
   iterated on the nodes; no transport call).
3. Scale a/L_Te at the predicted radii by (1 + perturbation) in PORTALS's own
   powerstate, write that profile out as the start, and let PORTALS flux-match from
   there; report the best evaluation. Only the node gradients move: PORTALS varies
   nothing else, so a start that also moved Te outside the last radius (its anchor)
   would make the truth unreachable.

PORTALS's ``predicted_roa`` is converted to rho on the given file and its profile
resolution step then extrapolates to rho = 1, so the radii it actually solves are
reported as ``mitim_r_over_a`` beside the request.
"""

import copy
import csv
import json
import sys
import traceback
from pathlib import Path

__all__ = ["main"]

SOURCES = ("qohme", "qrfe", "qbeame", "qione", "qohmi", "qrfi", "qbeami", "qioni", "qei",
           "qbrem", "qsync", "qline", "qfuse", "qfusi")


def _key(profiles, name):
    for key in profiles:
        if key.split("(")[0] == name:
            return key
    return None


def _rows(folder):
    with open(Path(folder) / "Outputs" / "optimization_data.csv") as handle:
        return list(csv.DictReader(handle))


def main(argument_file):
    args = json.loads(Path(argument_file).read_text())
    out = {"status": "error", "capability": "portals_tglf_closed_loop", "model": "tglf_neo"}
    try:
        import numpy as np
        from scipy.interpolate import PchipInterpolator
        from mitim_modules.portals import PORTALSmain
        from mitim_tools.gacode_tools import PROFILEStools
        from mitim_tools.opt_tools import STRATEGYtools

        work = Path(args["folder"]).resolve()
        roa = [float(x) for x in args["r_over_a"]]
        n = len(roa)

        def setup(folder, training, iterations):
            pf = PORTALSmain.portals(Path(folder).resolve())
            parameters = pf.portals_parameters
            parameters["solution"]["predicted_channels"] = ["te"]
            parameters["solution"]["predicted_roa"] = roa
            parameters["target"]["options"]["targets_evolve"] = []
            tglf = parameters["transport"]["options"]["tglf"]
            tglf["run"]["code_settings"] = args.get("tglf_code_settings", "SAT3")
            tglf["run"]["extraOptions"] = dict(args.get("tglf_extra_options") or {})
            tglf["use_scan_trick_for_stds"] = None
            parameters["transport"]["options"]["neo"]["run"]["extraOptions"] = dict(
                args.get("neo_extra_options") or {})
            pf.optimization_options["initialization_options"]["initial_training"] = training
            pf.optimization_options["convergence_options"]["maximum_iterations"] = iterations
            return pf

        state = PROFILEStools.gacode_state(Path(args["input_gacode"]).resolve())
        stripped = []
        for name in SOURCES:
            key = _key(state.profiles, name)
            if key is not None:
                state.profiles[key] = state.profiles[key] * 0.0
                stripped.append(key)
        state.derive_quantities()

        # 1. Truth: evaluation 0 of a run that starts at the true profile.
        truth_pf = setup(work / "truth", 1, 0)
        truth_pf.prep(copy.deepcopy(state))
        STRATEGYtools.MITIM_BO(truth_pf, cold_start=True, askQuestions=False).run()
        row = _rows(work / "truth")[0]
        aLte_true = np.array([float(row[f"aLte_{i + 1}"]) for i in range(n)])
        q_true = np.array([float(row[f"Qe_tr_turb_{i + 1}"]) + float(row[f"Qe_tr_neoc_{i + 1}"])
                           for i in range(n)])
        ps = truth_pf.powerstate
        rho_k = ps.plasma["rho"][0, 1:].detach().numpy()
        mitim_roa = ps.plasma["roa"][0, 1:].detach().numpy()

        # 2. Manufactured source, iterated on the nodes with targets only.
        rho = np.asarray(state.profiles["rho(-)"], dtype=float)
        r = np.asarray(state.derived["r"], dtype=float)
        volp = np.asarray(state.derived["volp_geo"], dtype=float)
        drho_dr = np.gradient(rho, r)
        inside = volp > 0
        key = _key(state.profiles, "qohme") or "qohme(MW/m^3)"
        volp_k = ps.plasma["volp"][0, 1:].detach().numpy()
        nodes = q_true * volp_k
        history = []
        for index in range(int(args.get("manufacture_iterations", 4))):
            power = PchipInterpolator(np.concatenate(([0.0], rho_k)), np.concatenate(([0.0], nodes)),
                                      extrapolate=True)
            q = np.zeros_like(r)
            q[inside] = power.derivative()(rho[inside]) * drho_dr[inside] / volp[inside]
            state.profiles[key] = q
            state.derive_quantities()
            probe = setup(work / f"manufacture_{index}", 1, 0)
            probe.prep(copy.deepcopy(state))
            probe.powerstate.calculateProfileFunctions()
            probe.powerstate.calculateTargets()
            got = probe.powerstate.plasma["QeMWm2"][0, 1:].detach().numpy()
            history.append(float(np.max(np.abs(got - q_true) / np.maximum(np.abs(q_true), 1e-30))))
            nodes = nodes + (q_true - got) * volp_k

        # 3. Perturbed start: the node gradients only, through PORTALS's own powerstate.
        eps = float(args["perturbation"])
        starter = setup(work / "start", 1, 0)
        starter.prep(copy.deepcopy(state))
        x_true = starter.powerstate.plasma["aLte"][:, 1:].clone()
        starter.powerstate.modify(x_true * (1.0 + eps))
        start = starter.powerstate.from_powerstate(write_input_gacode=work / "input.gacode_start")
        run_pf = setup(work / "run", int(args.get("initial_training", 5)),
                       int(args.get("maximum_iterations", 10)))
        run_pf.prep(start)
        STRATEGYtools.MITIM_BO(run_pf, cold_start=True, askQuestions=False).run()
        rows = _rows(work / "run")

        def residual(entry):
            return [float(entry[f"Qe_tr_turb_{i + 1}"]) + float(entry[f"Qe_tr_neoc_{i + 1}"])
                    - float(entry[f"Qe_tar_{i + 1}"]) for i in range(n)]

        best = min(rows, key=lambda entry: float(np.linalg.norm(residual(entry))))
        first = rows[0]
        out.update(
            status="ok", stripped_sources=stripped, manufacture_history=history,
            r_over_a=roa, mitim_r_over_a=mitim_roa.tolist(), rho_tor_norm=rho_k.tolist(),
            aLte_true=aLte_true.tolist(), transport_at_truth_MWm2=q_true.tolist(),
            target_after_manufacture_MWm2=got.tolist(),
            transport_after_manufacture_MWm2=q_true.tolist(),
            aLte_start=[float(first[f"aLte_{i + 1}"]) for i in range(n)],
            model_minus_required_at_start_MWm2=residual(first),
            aLte_recovered=[float(best[f"aLte_{i + 1}"]) for i in range(n)],
            final_model_MWm2=[float(best[f"Qe_tr_turb_{i + 1}"]) + float(best[f"Qe_tr_neoc_{i + 1}"])
                              for i in range(n)],
            final_required_MWm2=[float(best[f"Qe_tar_{i + 1}"]) for i in range(n)],
            evaluations=len(rows), best_iteration=int(float(best["Iteration"])),
            residual_history=[float(np.linalg.norm(residual(entry))) for entry in rows],
        )
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
