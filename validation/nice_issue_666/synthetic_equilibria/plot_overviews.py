"""Collect NICE equilibrium ODS artifacts and render canonical overviews."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaft.code.nice import collect_nice_outputs
from vaft.omas import load, plot_equilibrium_overview, save

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "vaft/data/samples/41672/source/pipeline-until-efit.json.gz"
NATIVE = Path("/tmp/nice331-solovev-20260914")
OUTPUT = Path(__file__).resolve().parent


def main():
    machine = load(SOURCE)
    cases = {
        "solovev": NATIVE / "full_default",
        "tokamaker": NATIVE / "tokamaker_full_default",
    }
    for name, case in cases.items():
        result = collect_nice_outputs(case)
        if result.ods is None or not result.converged:
            raise RuntimeError(f"{name}: no converged NICE equilibrium ODS")
        ods = result.ods
        # Native NICE tables contain the equilibrium only. Add immutable machine
        # context for the canonical overview without altering equilibrium data.
        ods["wall"] = machine["wall"]
        ods["dataset_description.data_entry.pulse"] = 41672
        ods_path = OUTPUT / f"nice_{name}_equilibrium_331ms.json.gz"
        png_path = OUTPUT / f"nice_{name}_equilibrium_overview_331ms.png"
        save(ods, ods_path)
        figure, _axes = plot_equilibrium_overview(ods, time_slice=0, show=False)
        figure.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"{name}: {ods_path} {png_path}")


if __name__ == "__main__":
    main()
