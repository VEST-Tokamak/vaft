"""Reproduce particle-motion reference artifacts and numerical validation evidence.

Run from the repository root:
    python -m workflow.particle_motion.run --output docs/assets/particle-motion
No external orbit solver, video encoder or display server is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vaft
import vaft.diagram as diagram
from vaft.plot import ParticleOrbitView, particle_overview
from vaft.process.particle import particle_motion_case


if (
    not Path(vaft.__file__)
    .resolve()
    .is_relative_to(Path(__file__).resolve().parents[2])
):
    raise RuntimeError(
        "Run python -m workflow.particle_motion.run from this source checkout"
    )

EXAMPLES = (
    ("cartesian", "gyration", "gyration"),
    ("mirror", "magnetic_mirror", "magnetic_mirror"),
    ("full-gc", "grad_b_drift", "grad_b_drift"),
    ("invariants", "magnetic_mirror", "guiding_center_invariants"),
    ("tokamak", "tokamak_trapped", "trapped_and_passing_orbits"),
    ("symmetry-breaking", "symmetry_breaking", "toroidal_symmetry_breaking"),
)

#: Examples without a committed static SVG. Their gyro-resolved orbits make a
#: ~440 KB SVG that no page embeds; the GIF and panels carry the same scene,
#: and the site has a 150 MiB budget (docs/build.py MAX_SITE_MIB).
NO_STATIC = {"tokamak", "symmetry-breaking"}


def generate(output, frames=12, dpi=55):
    output.mkdir(parents=True, exist_ok=True)
    experiments = {}
    panels = {}
    animated_panels = {}
    rows = []
    # A family without a static SVG must not keep one from an earlier run.
    for slug in NO_STATIC:
        (output / f"{slug}.svg").unlink(missing_ok=True)
    for slug, kind, builder in EXAMPLES:
        print(f"Computing {slug}: {kind}", flush=True)
        if kind not in experiments:
            experiments[kind] = particle_motion_case(kind)
        experiment = experiments[kind]
        full, gc = experiment.full, experiment.guiding_center
        end = min(full.time[-1], gc.time[-1])
        times = np.linspace(0, end, frames)
        function = getattr(diagram, builder)
        projection = (
            "parallel"
            if kind == "magnetic_mirror"
            else ("poloidal" if kind.startswith("tokamak") else "3d")
        )
        options = (
            {"projection": projection}
            if builder in ("gyration", "magnetic_mirror", "grad_b_drift")
            else {}
        )
        static = None if slug in NO_STATIC else f"{slug}.svg"
        if static:
            function(trajectory=experiment, time=end, **options).save(output / static)
        movie = function(
            trajectory=experiment, time=times, animation=True, fps=8, dpi=dpi, **options
        )
        movie.save(output / f"{slug}.gif")
        well = {}
        if kind == "magnetic_mirror":
            s = np.linspace(-9.99, 9.99, 200)
            well = {
                "bounce_coordinate": lambda x: x[2],
                "well_coordinate": s,
                "well_B": 1 + (s / 10) ** 2,
            }
        view = ParticleOrbitView.from_comparison(
            experiment, title=kind.replace("_", " "), **well
        )
        # Panels depend only on the experiment, so examples sharing a case share one file.
        if kind not in panels:
            panels[kind] = f"{slug}-diagnostics.png"
            fig, _ = particle_overview(view, time=end)
            fig.savefig(output / panels[kind], dpi=100)
            plt.close(fig)
        if slug == "mirror":
            particle_overview(
                view, time=times[::2], animation=True, dpi=50, fps=4
            ).save(output / "mirror-diagnostics.gif")
            animated_panels[kind] = "mirror-diagnostics.gif"
        d = view.gc_diagnostics
        rows.append(
            {
                "id": slug,
                "builder": builder,
                "case": kind,
                "static": static,
                "animation": f"{slug}.gif",
                "quantitative_panels": panels[kind],
                "animated_panels": animated_panels.get(kind),
                "driver": {"name": "time", "unit": "s", "values": times.tolist()},
                "field": full.field.metadata(),
                "full_termination": full.termination,
                "gc_termination": gc.termination,
                "gc_energy_residual_max": float(np.max(np.abs(d["energy_residual"]))),
                "gc_P_phi_residual_max": float(
                    np.max(np.abs(d.get("P_phi_residual", [0.0])))
                ),
                "ordering_warning": experiment.metadata["ordering_warning"],
                "rho_over_L_B_max": experiment.metadata["rho_over_L_B_max"],
                "orbit_class": d["orbit_class"],
                "bounce_cycles": d["bounce_cycles"],
                "bounce_period_s": d.get("bounce_period_s", []),
                "transit_period_s": d.get("transit_period_s", []),
                "fidelity": experiment.metadata["fidelity"],
            }
        )
    # Preserve every native timestamp while keeping generated review diffs small:
    # fixed-step timebases are stored as their exact recurrence (decoded bit for
    # bit by vaft.data.particle.native_timebases), and sidecars are compact JSON.
    for filename in [f"{slug}.gif.json" for slug, _, _ in EXAMPLES] + [
        "mirror-diagnostics.gif.json"
    ]:
        sidecar = output / filename
        sidecar.write_text(
            json.dumps(json.loads(sidecar.read_text()), separators=(",", ":")) + "\n"
        )
    document = {
        "schema_version": 1,
        "generator": "workflow/particle_motion/run.py",
        "units": "SI",
        "presentation": {"frames": frames, "dpi": dpi},
        "scope": "nonrelativistic collisionless static fields; weak-flow Hamiltonian GC",
        "deferred": [
            "time-dependent fields",
            "slowly varying mirror second-invariant experiment",
            "collisions and transport",
            "external-solver cross-validation",
        ],
        "examples": rows,
    }
    (output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(f"Wrote {len(rows)} reproducible examples to {output}", flush=True)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/particle-motion")
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=55)
    args = parser.parse_args()
    if args.frames < 2 or args.dpi < 20:
        parser.error("at least two frames and dpi >= 20 required")
    generate(args.output, args.frames, args.dpi)
