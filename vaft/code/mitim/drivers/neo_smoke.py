"""MITIM NEOtools at a few radii on one input.gacode: the stage-A1 smoke capability.

Runs in the MITIM interpreter; imports MITIM and the standard library only.
"""

import json
import sys
import traceback
from pathlib import Path

__all__ = ["main"]


def main(argument_file):
    args = json.loads(Path(argument_file).read_text())
    out = {"status": "error", "capability": "neo_smoke"}
    try:
        import numpy as np
        from mitim_tools.gacode_tools import NEOtools

        folder = Path(args["folder"]).resolve()
        neo = NEOtools.NEO(rhos=np.asarray(args["rhos"], dtype=float))
        neo.prep(Path(args["input_gacode"]).resolve(), folder)
        options = {"cold_start": True}
        if args.get("code_settings"):
            options["code_settings"] = args["code_settings"]
        neo.run(args.get("subfolder", "smoke/"), **options)
        neo.read(label="smoke")
        runs = sorted({str(path.parent) for path in folder.rglob("out.neo.transport")})
        out.update(status="ok" if runs else "error", run_directories=runs,
                   rhos=list(args["rhos"]))
        if not runs:
            out["error"] = "MITIM finished but no out.neo.transport was written"
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
