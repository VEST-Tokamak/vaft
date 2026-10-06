"""MITIM TGLFtools run at given rho_tor_norm, outputs regrouped per radius (#1588 A2).

Runs in the MITIM interpreter; imports MITIM and the standard library only.
"""

import json
import re
import shutil
import sys
import traceback
from pathlib import Path

__all__ = ["main", "split_by_radius"]

#: MITIM 5.3.0 runs every radius in one folder and suffixes each file with it.
SUFFIXED = re.compile(r"^(?P<base>(?:out|input)\.tglf(?:\.[A-Za-z0-9_]+?)?)_(?P<rho>\d+\.\d+)$")


def split_by_radius(source, target, marker="out.tglf.gbflux"):
    """Copy each radius's files, unsuffixed, into ``target/rho_<value>/``."""
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
    return sorted(r for r in radii if (Path(r) / marker).is_file())


def main(argument_file):
    args = json.loads(Path(argument_file).read_text())
    out = {"status": "error", "capability": "tglf_run"}
    try:
        import numpy as np
        from mitim_tools.gacode_tools import TGLFtools

        folder = Path(args["folder"]).resolve()
        subfolder = args.get("subfolder", "run/")
        tglf = TGLFtools.TGLF(rhos=np.asarray(args["rho_tor_norm"], dtype=float))
        tglf.prep(Path(args["input_gacode"]).resolve(), folder, cold_start=True, forceIfcold_start=True)
        tglf.run(subfolder, code_settings=args["code_settings"],
                 extraOptions=dict(args.get("extra_options") or {}),
                 cold_start=True, forceIfcold_start=True)
        runs = split_by_radius(folder / subfolder, folder / "by_radius")
        out.update(status="ok" if runs else "error", run_directories=runs,
                   rho_tor_norm=list(args["rho_tor_norm"]))
        if not runs:
            out["error"] = "MITIM finished but no out.tglf.gbflux was written"
    except Exception as error:  # reported, not raised: the caller reads result.json
        out["error"] = f"{type(error).__name__}: {error}"
        out["traceback"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, indent=1))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
