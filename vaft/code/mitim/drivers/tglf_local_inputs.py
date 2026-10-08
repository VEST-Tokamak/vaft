"""MITIM's own TGLF local inputs at exact r/a, written but not run (#1588 A2).

Runs in the MITIM interpreter; imports MITIM and the standard library only.
``prep()`` always converts from rho_tor_norm, so this calls the state converter
directly with ``r_is_rho=False``: the surfaces are exactly the r/a VAFT asked for.
"""

import json
import sys
import traceback
from pathlib import Path

__all__ = ["main"]


def main(argument_file):
    args = json.loads(Path(argument_file).read_text(encoding="utf-8"))
    out = {"status": "error", "capability": "tglf_local_inputs"}
    try:
        import numpy as np
        from mitim_tools.gacode_tools import PROFILEStools, TGLFtools
        from mitim_tools.misc_tools.PLASMAtools import md_u

        folder = Path(args["folder"]).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        state = PROFILEStools.gacode_state(Path(args["input_gacode"]).resolve())
        state.derive_quantities(mi_ref=md_u)
        roa = [float(r) for r in args["r_over_a"]]
        inputs = state.to_tglf(r=roa, code_settings=args["code_settings"], r_is_rho=False)
        files = {}
        for label, parameters in inputs.items():
            tglf_input = TGLFtools.TGLFinput.initialize_in_memory(parameters)
            tglf_input.file = folder / f"input.tglf_{float(label):.6f}"
            tglf_input.write_state()
            files[f"{float(label):.6f}"] = str(tglf_input.file)
        # MITIM's own r/a -> rho_tor_norm map, from the same state, for the bridge check.
        rho = np.interp(roa, np.asarray(state.derived["roa"]), np.asarray(state.profiles["rho(-)"]))
        out.update(status="ok", files=files, r_over_a=roa, rho_tor_norm=[float(x) for x in rho],
                   code_settings=args["code_settings"])
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
