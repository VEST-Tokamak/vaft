"""Build the compact, read-only notebooks used by the documentation exporter."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def notebook(title: str, sections: list[tuple[str, str]]) -> nbf.NotebookNode:
    cells = [nbf.v4.new_markdown_cell(f"# {title}")]
    for heading, source in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        cells.append(nbf.v4.new_code_cell(source))
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "vaft_docs": {"read_only": True},
        },
    )


COMMON = """from pathlib import Path
import os

OUTPUT_DIR = Path(os.environ.get("VAFT_DOCS_OUTPUT_DIR", "notebooks/outputs/docs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
"""


def replace_tagged_cell(book: nbf.NotebookNode, tag: str, source: str) -> None:
    """Replace a generated cell while leaving the authored notebook intact."""
    book.cells = [cell for cell in book.cells if tag not in cell.metadata.get("tags", [])]
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = [tag]
    book.cells.append(cell)


def patch_retained_notebooks() -> None:
    plotting_path = NOTEBOOKS / "plotting_sample_using_vaft_plot_module.ipynb"
    plotting = nbf.read(plotting_path, as_version=4)
    plotting.metadata["vaft_docs"] = {"read_only": True}
    nbf.write(plotting, plotting_path)

    fluctuation_path = NOTEBOOKS / "fluctuation_diagnostics_analysis.ipynb"
    fluctuation = nbf.read(fluctuation_path, as_version=4)
    for cell in fluctuation.cells:
        if cell.cell_type == "code" and 'output_dir = Path("outputs/fluctuation_diagnostics_analysis")' in cell.source:
            cell.source = cell.source.replace(
                'output_dir = Path("outputs/fluctuation_diagnostics_analysis")',
                'output_dir = Path(os.environ.get("VAFT_DOCS_OUTPUT_DIR", "notebooks/outputs/docs"))',
            ).replace(
                "from pathlib import Path",
                "from pathlib import Path\nimport os",
            )
            break
    fluctuation.metadata["vaft_docs"] = {"read_only": True}
    nbf.write(fluctuation, fluctuation_path)

    equilibrium_path = NOTEBOOKS / "equilibrium_refinement_using_chease.ipynb"
    equilibrium = nbf.read(equilibrium_path, as_version=4)
    first_code = next(cell for cell in equilibrium.cells if cell.cell_type == "code")
    if "VAFT_DOCS_OUTPUT_DIR" not in first_code.source:
        first_code.source = COMMON + "\n" + first_code.source
    replace_tagged_cell(
        equilibrium,
        "vaft-docs-export",
        """fig.savefig(OUTPUT_DIR / "equilibrium-inputs.png", dpi=180, bbox_inches="tight")
readiness = {
    "sample": str(sample_path),
    "prepared_files": [path.name for path in inputs.files],
    "chease_executable": str(executable) if executable else None,
    "execution_mode": "external-binary" if executable else "input-preparation",
}
import json
(OUTPUT_DIR / "equilibrium-readiness.txt").write_text(json.dumps(readiness, indent=2) + "\\n")
print(readiness)""",
    )
    equilibrium.metadata["vaft_docs"] = {"read_only": True}
    nbf.write(equilibrium, equilibrium_path)

    external_path = NOTEBOOKS / "initialize_external_fusion_codes.ipynb"
    external = nbf.read(external_path, as_version=4)
    replace_tagged_cell(
        external,
        "vaft-docs-export",
        COMMON
        + """
import json
report = {}
for variable, relative_executables in external_codes.items():
    configured = os.environ.get(variable)
    if not configured:
        report[variable] = {"status": "not configured", "required": False}
        continue
    root = Path(configured).expanduser()
    missing = [str(root / relative) for relative in relative_executables
               if not (root / relative).is_file() or not os.access(root / relative, os.X_OK)]
    report[variable] = {
        "status": "ready" if not missing else "incomplete",
        "missing_or_not_executable": missing,
    }
text = json.dumps(report, indent=2)
print(text)
(OUTPUT_DIR / "external-code-readiness.txt").write_text(text + "\\n", encoding="utf-8")
""",
    )
    external.metadata["vaft_docs"] = {"read_only": True}
    nbf.write(external, external_path)


DATABASE = notebook(
    "Database Initialization and Read-Only Load",
    [
        (
            "Offline first result",
            COMMON
            + """
import matplotlib.pyplot as plt
import vaft

if os.environ.get("VAFT_DOCS_READ_ONLY") != "1":
    raise RuntimeError("Set VAFT_DOCS_READ_ONLY=1 before executing this notebook")

def _blocked_write(*args, **kwargs):
    raise RuntimeError("Remote database writes are disabled during documentation export")

vaft.database.save = _blocked_write
ods = vaft.omas.sample_ods()
roots = sorted(ods.keys())
print("Offline sample roots:", roots)
vaft.plot.magnetics_time_ip(ods)
fig = plt.gcf()
fig.suptitle("VAFT packaged sample: plasma current")
fig.savefig(OUTPUT_DIR / "first-result.png", dpi=180, bbox_inches="tight")
plt.show()
""",
        ),
        (
            "Public HSDS verification",
            """import json
import numpy as np

with vaft.database.open(39915, source="public", paths="equilibrium") as remote:
    times = np.asarray(remote["equilibrium.time"], dtype=float)
    summary = {
        "shot": 39915,
        "namespace": "public",
        "representation": "lazy OMAS",
        "ids": ["equilibrium"],
        "time_slices": int(times.size),
        "first_time_s": float(times[0]),
        "last_time_s": float(times[-1]),
    }

text = json.dumps(summary, indent=2)
print(text)
(OUTPUT_DIR / "hsds-39915.txt").write_text(text + "\\n", encoding="utf-8")
""",
        ),
    ],
)


CONVERSION = notebook(
    "Read and Convert an IMAS-Aligned Data Structure",
    [
        (
            "Deterministic local round-trip",
            COMMON
            + """
import json
import tempfile
import vaft

ods = vaft.omas.sample_ods()
with tempfile.TemporaryDirectory(prefix="vaft-docs-roundtrip-") as tmp:
    target = Path(tmp) / "sample.json.gz"
    vaft.omas.save(ods, target)
    restored = vaft.omas.load(target)
    before = sorted(ods.keys())
    after = sorted(restored.keys())
    summary = {
        "format": "OMAS JSON using IMAS paths",
        "root_count_before": len(before),
        "root_count_after": len(after),
        "roots_equal": before == after,
        "representative_roots": before[:10],
    }

text = json.dumps(summary, indent=2)
print(text)
if not summary["roots_equal"]:
    raise AssertionError("The local OMAS/IMAS-path round-trip changed root IDS names")
(OUTPUT_DIR / "imas-roundtrip.txt").write_text(text + "\\n", encoding="utf-8")
""",
        )
    ],
)


KINETIC = notebook(
    "Kinetic Profiles from Paired VEST Diagnostics",
    [
        (
            "Build shot 48224 at 300 ms",
            COMMON
            + """
import matplotlib.pyplot as plt
import numpy as np
import vaft
from vaft.code.kineticEfit import build_kinetic_core_profiles
from vaft.data import read_geqdsk

shot = 48224
time_ms = 300.0
data_root = Path("vaft/data/kineticEfit")
geq = read_geqdsk(data_root / "g048224.00300")
ods = geq.to_omas()
vaft.machine_mapping.dataset_description(
    ods, source=shot,
    options={"source_type": "shot", "description": "Paired documentation sample"},
)
vaft.machine_mapping.thomson_scattering(ods, shot, data_root / "NeTe_48224.mat")
vaft.machine_mapping.charge_exchange(
    ods, shotnumber=shot, options="ids", mat_file=data_root / "IDS_48224.mat"
)
ods = build_kinetic_core_profiles(
    ods, geq, time_ms,
    te_mode="polynomial", ne_mode="polynomial",
    ti_mode="polynomial", vtor_mode="polynomial",
)

profile = ods["core_profiles.profiles_1d.0"]
rho = np.asarray(profile["grid.rho_tor_norm"], dtype=float)
te = np.asarray(profile["electrons.temperature"], dtype=float)
ne = np.asarray(profile["electrons.density"], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].plot(rho, te, color="tab:red", lw=2)
axes[0].set(xlabel=r"$\\rho_{tor,norm}$", ylabel=r"$T_e$ [eV]", title="Electron temperature")
axes[1].plot(rho, ne / 1e19, color="tab:blue", lw=2)
axes[1].set(xlabel=r"$\\rho_{tor,norm}$", ylabel=r"$n_e$ [$10^{19}$ m$^{-3}$]", title="Electron density")
for ax in axes:
    ax.grid(alpha=0.25)
fig.suptitle("VEST shot 48224 at 300 ms")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "kinetic-profile.png", dpi=180, bbox_inches="tight")
plt.show()
print({"shot": shot, "time_ms": time_ms, "grid_points": int(rho.size)})
""",
        )
    ],
)


CONFINEMENT = notebook(
    "Confinement-Time Scaling",
    [
        (
            "Deterministic operations-analysis example",
            COMMON
            + """
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = np.linspace(0.290, 0.345, 12)
tau_exp = np.linspace(2.25, 3.05, 12) * 1e-3
tau_ipb89 = tau_exp * np.linspace(0.88, 1.08, 12)
tau_h98 = tau_exp * np.linspace(0.94, 1.12, 12)
h_factor = tau_exp / tau_ipb89
table = pd.DataFrame({
    "time_s": time,
    "tau_exp_s": tau_exp,
    "tau_IPB89_s": tau_ipb89,
    "tau_H98y2_s": tau_h98,
    "H_IPB89": h_factor,
})

fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)
axes[0].plot(time * 1e3, tau_exp * 1e3, "o-", label="experiment")
axes[0].plot(time * 1e3, tau_ipb89 * 1e3, "^-", label="IPB89")
axes[0].plot(time * 1e3, tau_h98 * 1e3, "s-", label="H98y2")
axes[0].set(ylabel="Confinement time [ms]", title="Deterministic VEST operations example")
axes[0].legend()
axes[1].plot(time * 1e3, h_factor, "o-", color="tab:purple")
axes[1].axhline(1.0, color="black", ls="--", lw=1)
axes[1].set(xlabel="Time [ms]", ylabel="H factor")
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confinement-scaling.png", dpi=180, bbox_inches="tight")
plt.show()

summary = {
    "rows": len(table),
    "mean_tau_exp_ms": round(float(table.tau_exp_s.mean() * 1e3), 4),
    "mean_H_IPB89": round(float(table.H_IPB89.mean()), 4),
}
print(json.dumps(summary, indent=2))
(OUTPUT_DIR / "confinement-scaling.txt").write_text(json.dumps(summary, indent=2) + "\\n")
""",
        )
    ],
)


PIPELINE = notebook(
    "Automated Pipeline Overview",
    [
        (
            "Routine, corrective and summary stages",
            COMMON
            + """
import json
import matplotlib.pyplot as plt

stages = [
    ("Raw DAQ", "routine", "archived raw dump"),
    ("Diagnostics IDS", "routine", "mapped OMAS/IMAS data"),
    ("EFIT + CHEASE", "routine", "equilibrium"),
    ("Profiles", "corrective", "core_profiles"),
    ("DCON/RDCON/GPEC", "routine", "mhd_linear"),
    ("History tables", "summary", "multi-shot reports"),
]

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis("off")
for index, (name, workflow, output) in enumerate(stages):
    x = index / (len(stages) - 1)
    ax.text(x, 0.55, name, ha="center", va="center", fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "fc": "#eef2ff", "ec": "#7c6ee6"})
    ax.text(x, 0.2, output, ha="center", va="center", fontsize=8, color="#475569")
    if index < len(stages) - 1:
        ax.annotate("", xy=(x + 0.145, 0.55), xytext=(x + 0.055, 0.55),
                    arrowprops={"arrowstyle": "->", "color": "#475569"})
fig.suptitle("VAFT Snakemake workflow products")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pipeline-overview.png", dpi=180, bbox_inches="tight")
plt.show()

summary = [{"stage": name, "workflow": workflow, "output": output} for name, workflow, output in stages]
print(json.dumps(summary, indent=2))
(OUTPUT_DIR / "pipeline-overview.txt").write_text(json.dumps(summary, indent=2) + "\\n")
""",
        )
    ],
)


for filename, built in {
    "database_initialization_and_load.ipynb": DATABASE,
    "read_and_convert_data_structure.ipynb": CONVERSION,
    "kinetic_efit_end_to_end.ipynb": KINETIC,
    "confinement_time_scaling.ipynb": CONFINEMENT,
    "automated_pipeline_overview.ipynb": PIPELINE,
}.items():
    nbf.write(built, NOTEBOOKS / filename)
    print(f"wrote {NOTEBOOKS / filename}")

patch_retained_notebooks()
