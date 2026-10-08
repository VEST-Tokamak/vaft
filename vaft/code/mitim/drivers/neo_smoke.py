"""MITIM NEOtools at a few radii on one input.gacode: the stage-A1 smoke capability.

Runs in the MITIM interpreter; imports MITIM and the standard library only.
"""

import json
import re
import shutil
import sys
import traceback
from pathlib import Path

__all__ = ["main"]


#: MITIM 5.3.0 runs every radius in one folder and suffixes each file with it,
#: e.g. ``out.neo.transport_0.5000``.
SUFFIXED = re.compile(r"^(?P<base>(?:out|input)\.neo(?:\.[A-Za-z0-9_]+?)?)_(?P<rho>\d+\.\d+)$")


def split_by_radius(source, target):
    """Copy each radius's files, unsuffixed, into ``target/rho_<value>/`` (one NEO run each)."""
    source, target = Path(source), Path(target)
    if target.exists():
        shutil.rmtree(target)
    radii = set()
    for path in sorted(source.iterdir()) if source.is_dir() else ():
        match = SUFFIXED.match(path.name)
        if not match:
            continue
        directory = target / f"rho_{match['rho']}"
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, directory / match["base"])
        radii.add(str(directory))
    return sorted(r for r in radii if (Path(r) / "out.neo.transport").is_file())


def main(argument_file):
    args = json.loads(Path(argument_file).read_text(encoding="utf-8"))
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
        if args.get("extra_options"):
            options["extraOptions"] = dict(args["extra_options"])
        neo.run(args.get("subfolder", "smoke/"), **options)
        neo.read(label="smoke")
        runs = split_by_radius(folder / args.get("subfolder", "smoke/"), folder / "by_radius")
        out.update(status="ok" if runs else "error", run_directories=runs,
                   rhos=list(args["rhos"]))
        if not runs:
            out["error"] = "MITIM finished but no out.neo.transport was written"
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
