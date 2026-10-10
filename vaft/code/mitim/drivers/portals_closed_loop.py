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
    args = json.loads(Path(argument_file).read_text(encoding="utf-8"))
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
            # powerstate rewrites the profiles object it is given (resolution change,
            # gradient flattening), so each build gets its own copy. Its resolution
            # increase also extrapolates the profile to rho = 1, and VAFT's input.gacode
            # stops at rho_max (~0.94): that moves the minor radius and every r/a by a few
            # percent (48224: a 0.2773 -> 0.2888 m), so this loop keeps VAFT's grid.
            return STATEtools.powerstate(copy.deepcopy(st), increase_profile_resol=False,
                                         evolution_options={"ProfilePredicted": ["te"],
                                                            "rhoPredicted": rho_k},
                                         transport_options=transport, target_options=target)

        truth = build(state)
        truth.calculate()
        q_tr = truth.plasma["QeMWm2_tr"][0, 1:].detach().numpy()
        volp_k = truth.plasma["volp"][0, 1:].detach().numpy()
        roa_k = truth.plasma["roa"][0, 1:].detach().numpy()

        # Manufactured cumulative power through (0, 0) and (rho_k, Q_k V'_k), built in
        # rho because that is the coordinate the fixed targets are interpolated in; the
        # source is q = (dP/drho)(drho/dr) / V'.
        rho = np.asarray(state.profiles["rho(-)"], dtype=float)
        r = np.asarray(state.derived["r"], dtype=float)
        volp = np.asarray(state.derived["volp_geo"], dtype=float)
        drho_dr = np.gradient(rho, r)
        inside = volp > 0
        key = _key(state.profiles, "qohme") or "qohme(MW/m^3)"
        nodes = q_tr * volp_k
        history = []
        # The discrete chain (PCHIP, gradient, MITIM's trapezoid volume integral and its
        # interpolation to rho_k) misses the nodes by a few percent; a few fixed-point
        # corrections of the nodes remove that, and the remaining error is reported.
        for _ in range(int(args.get("manufacture_iterations", 4))):
            power = PchipInterpolator(np.concatenate(([0.0], rho_k.numpy())),
                                      np.concatenate(([0.0], nodes)), extrapolate=True)
            q = np.zeros_like(r)
            q[inside] = power.derivative()(rho[inside]) * drho_dr[inside] / volp[inside]
            state.profiles[key] = q
            state.derive_quantities()
            matched = build(state)
            matched.calculate()
            got = matched.plasma["QeMWm2"][0, 1:].detach().numpy()
            history.append(float(np.max(np.abs(got - q_tr) / np.maximum(np.abs(q_tr), 1e-30))))
            nodes = nodes + (q_tr - got) * volp_k
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
            manufacture_history=history,
            rho_tor_norm=rho_k.tolist(), r_over_a=[float(x) for x in args["r_over_a"]],
            mitim_r_over_a=roa_k.tolist(),
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
    Path("result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
