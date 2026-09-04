"""Visual review of the plotting revision (machine parts, active channels,
flux styles, three-column overview, environment-aware interaction).

Run it in IPython and every figure of the seven items opens, with the
objects left in the namespace to poke at::

    ipython -i notebooks/_plotting_revision_review.py

or, in a notebook cell, ``%run notebooks/_plotting_revision_review.py``.
Under a plain ``python`` the figures are saved beside this file instead.

Names left behind: ``ods`` (shot 39915), ``figures`` (item -> figure),
``interactive`` (the navigator result; ``interactive.navigator.select(t)``
moves every panel), ``catalog`` (discovery for the shot).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt

import vaft
import vaft.omas
import vaft.plot
from vaft.plot.environment import default_interaction_backend, detect_environment

ods = vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))
environment = detect_environment()
print(f"environment: {environment}  -> interactive backend {default_interaction_backend(environment)!r}")

figures: dict[str, object] = {}

# 1. Machine cross-section: coils from their rectangles, one legend entry per set.
figures["1_machine_poloidal"], _ = vaft.omas.plot_machine_geometry_poloidal(ods)
figures["1_pf_coils"], _ = vaft.omas.plot_pf_coil_geometry_poloidal(ods)

# 2. Only channels carrying a valid signal by default; selection="all" restores everything.
figures["2_probes_active"], _ = vaft.omas.plot_b_field_probe_time_field(ods)
figures["2_probes_all"], _ = vaft.omas.plot_b_field_probe_time_field(ods, selection="all", title="B-field Probes, selection='all'")
figures["2_pf_currents"], _ = vaft.omas.plot_pf_coil_time_current_turns(ods, layout="subplots")

# 3. Top view with the diagnostics that store a toroidal position.
figures["3_topview"], _ = vaft.omas.plot_machine_geometry_topview(ods)

# 4. Diamagnetic flux: the measured waveform, and nothing repeated under synthetic="both".
figures["4_diamagnetic"], _ = vaft.omas.plot_diamagnetic_flux_time(ods, synthetic="both")

# 5. Poloidal-flux styles side by side.
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, style in zip(axes, vaft.plot.PSI_STYLES):
    vaft.omas.plot_equilibrium_field_psi(ods, time_slice=4, style=style, ax=ax)
    ax.set_title(f"style={style!r}")
fig.tight_layout()
figures["5_psi_styles"] = fig

# 6. The three-column slice overview.
figures["6_overview"], _ = vaft.omas.plot_equilibrium_overview(ods)

# 7. Interaction chosen for this environment (slider on a live canvas, widget in a notebook, none in a script).
interactive = vaft.omas.plot_equilibrium_interactive(ods)
figures["7_interactive"] = interactive.figure

catalog = vaft.omas.available_plots(ods, query="equilibrium", detail=True)

if environment.live_figures or environment.in_kernel:
    plt.show()
else:
    out = pathlib.Path(__file__).resolve().parent / "_plotting_revision_review"
    out.mkdir(exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(out / f"{name}.png", dpi=110)
    print(f"no live canvas here: figures saved under {out}", file=sys.stderr)
