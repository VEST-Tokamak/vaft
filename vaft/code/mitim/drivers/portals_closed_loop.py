"""PORTALS/powertorch synthetic closed loop with a manufactured source (#1588 C1).

Runs in the MITIM interpreter; imports MITIM and the standard library only.

1. Strip every heat source and sink from the input.gacode, so the targets are what
   this script puts there and nothing else.
2. Evaluate the chosen transport model at the given (true) profile: Q_e^tr(r_k).
3. Manufacture an electron source q_e(r) (as qohme) whose volume-integrated power
   equals Q_e^tr(r_k) * V'(r_k) at every predicted radius, in MITIM's own geometry,
   so the true profile is flux-matched by construction.
4. Start from the true gradients scaled by (1 + perturbation) and let the solver
   flux-match; the recovered a/L_Te must return to the true one.

MITIM's residual is S = P_target - P_transport; the result also reports VAFT's
R = Q_model - Q_required (= -S) at the perturbed start, whose sign is fixed: a
steeper start carries more flux than required, R > 0.
"""

import copy
import json
import sys
import traceback
from pathlib import Path

__all__ = ["main"]

#: input.gacode heat-source columns (MW/m^3) whose sum makes the fixed targets,
#: plus the exchange/radiation/fusion terms subtracted from them.
SOURCES = ("qohme", "qrfe", "qbeame", "qione", "qohmi", "qrfi", "qbeami", "qioni", "qei",
           "qbrem", "qsync", "qline", "qfuse", "qfusi")


def _key(profiles, name):
    for key in profiles:
        if key.split("(")[0] == name:
            return key
    return None


def main(argument_file):
    args = json.loads(Path(argument_file).read_text())
    out = {"status": "error", "capability": "portals_closed_loop", "model": args["model"]}
    try:
        import numpy as np
        import torch
        from scipy.interpolate import PchipInterpolator
        from mitim_tools.gacode_tools import PROFILEStools
        from mitim_modules.powertorch import STATEtools
        from mitim_modules.powertorch.physics_models import targets_analytic, transport_analytic

        rho_k = torch.tensor([float(x) for x in args["rho_tor_norm"]], dtype=torch.double)
        state = PROFILEStools.gacode_state(Path(args["input_gacode"]).resolve())
        stripped = []
        for name in SOURCES:
            key = _key(state.profiles, name)
            if key is not None:
                state.profiles[key] = state.profiles[key] * 0.0
                stripped.append(key)
        state.derive_quantities()

        def build(st):
            if args["model"] != "analytic":
                raise ValueError(f"model {args['model']!r} is not supported by this driver")
            n = rho_k.shape[0]
            transport = {"evaluator": transport_analytic.diffusion_model,
                         "options": {"chi_e": torch.ones(n, dtype=torch.double) * float(args["chi_e"]),
                                     "chi_i": torch.ones(n, dtype=torch.double) * float(args["chi_i"])}}
            target = {"evaluator": targets_analytic.analytical_model,
                      "options": {"targets_evolve": [], "target_evaluator_method": "powerstate",
                                  "force_zero_particle_flux": True, "percent_error": 1.0}}
            return STATEtools.powerstate(st, evolution_options={"ProfilePredicted": ["te"],
                                                                "rhoPredicted": rho_k},
                                         transport_options=transport, target_options=target)

        truth = build(state)
        truth.calculate()
        q_tr = truth.plasma["QeMWm2_tr"][0, 1:].detach().numpy()
        volp_k = truth.plasma["volp"][0, 1:].detach().numpy()
        roa_k = truth.plasma["roa"][0, 1:].detach().numpy()
        a = float(truth.plasma["a"][0])
        r_k = roa_k * a

        # Manufactured cumulative power through (0, 0) and (r_k, Q_k V'_k); q = P'/V'.
        r = np.asarray(state.derived["r"], dtype=float)
        volp = np.asarray(state.derived["volp_geo"], dtype=float)
        power = PchipInterpolator(np.concatenate(([0.0], r_k)), np.concatenate(([0.0], q_tr * volp_k)),
                                  extrapolate=True)
        q = np.zeros_like(r)
        inside = volp > 0
        q[inside] = power.derivative()(r[inside]) / volp[inside]
        key = _key(state.profiles, "qohme") or "qohme(MW/m^3)"
        state.profiles[key] = q
        state.derive_quantities()

        matched = build(state)
        matched.calculate()
        target_k = matched.plasma["QeMWm2"][0, 1:].detach().numpy()
        transport_k = matched.plasma["QeMWm2_tr"][0, 1:].detach().numpy()
        x_true = matched.plasma["aLte"][:, 1:].clone()

        perturbed = copy.deepcopy(matched)
        perturbed.calculate(X=x_true * (1.0 + float(args["perturbation"])))
        start_aLte = perturbed.plasma["aLte"][0, 1:].detach().numpy()
        start_model = perturbed.plasma["QeMWm2_tr"][0, 1:].detach().numpy()
        start_required = perturbed.plasma["QeMWm2"][0, 1:].detach().numpy()
        perturbed.flux_match(algorithm=args.get("algorithm", "root"))
        perturbed.calculate(X=perturbed.plasma["aLte"][:, 1:].clone())
        recovered = perturbed.plasma["aLte"][0, 1:].detach().numpy()

        out.update(
            status="ok",
            stripped_sources=stripped,
            rho_tor_norm=rho_k.tolist(), r_over_a=roa_k.tolist(),
            transport_at_truth_MWm2=q_tr.tolist(),
            target_after_manufacture_MWm2=target_k.tolist(),
            transport_after_manufacture_MWm2=transport_k.tolist(),
            aLte_true=x_true[0].tolist(), aLte_start=start_aLte.tolist(),
            aLte_recovered=recovered.tolist(),
            model_minus_required_at_start_MWm2=(start_model - start_required).tolist(),
            final_model_MWm2=perturbed.plasma["QeMWm2_tr"][0, 1:].detach().numpy().tolist(),
            final_required_MWm2=perturbed.plasma["QeMWm2"][0, 1:].detach().numpy().tolist(),
        )
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
