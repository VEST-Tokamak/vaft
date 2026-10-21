"""Provenance and freshness of the particle-motion references (#2045).

``docs/assets/particle-motion`` publishes numbers the orbit code produced
(``bounce_period_s``, ``gc_energy_residual_max``, ``rho_over_L_B_max`` ...)
next to the GIFs and panels drawn from the same runs. Like the animation
manifest written by :mod:`vaft.diagram._replay`, ``manifest.json`` records the
commit it was generated on and a SHA-256 for every scientific source the rows
depend on, so a later change to the integrators, fields, diagnostics or
builders is caught instead of silently publishing stale evidence.

``check_references`` is what the documentation catalog gate runs (digests
only, no orbit is traced); ``python -m workflow.particle_motion.run --check``
also re-traces the cases and compares the published rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

SCHEMA_VERSION = 2
ROOT = Path(__file__).resolve().parents[2]
REFERENCES = ROOT / "docs/assets/particle-motion"

#: ``(artifact slug, particle_motion_case kind, vaft.diagram builder)``.
EXAMPLES = (
    ("cartesian", "gyration", "gyration"),
    ("mirror", "magnetic_mirror", "magnetic_mirror"),
    ("full-gc", "grad_b_drift", "grad_b_drift"),
    ("invariants", "magnetic_mirror", "guiding_center_invariants"),
    ("tokamak", "tokamak_trapped", "trapped_and_passing_orbits"),
    ("symmetry-breaking", "symmetry_breaking", "toroidal_symmetry_breaking"),
)

#: The integrators, benchmark fields, diagnostics and the generator itself; the
#: builder modules and the formulas they reach are added from the import graph.
SCIENTIFIC_SOURCES = (
    "vaft/formula/particle.py",
    "vaft/data/particle.py",
    "vaft/process/particle.py",
    "vaft/process/particle_fields.py",
    "vaft/plot/particle.py",
    "vaft/diagram/_particle_references.py",
    "workflow/particle_motion/run.py",
)

#: Roundoff-level maxima: their exact value is platform noise, their order is not.
_RESIDUAL_KEYS = frozenset({"gc_energy_residual_max", "gc_P_phi_residual_max"})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generated_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def scientific_source_paths():
    """Repository-relative paths every published row depends on, sorted."""
    import inspect

    import vaft.diagram as diagram

    from .docs_catalog import formula_references

    paths = {ROOT / p for p in SCIENTIFIC_SOURCES}
    for _, _, builder in EXAMPLES:
        function = inspect.unwrap(getattr(diagram, builder))
        path = Path(inspect.getsourcefile(function)).resolve()
        paths.add(path)
        for name in formula_references(path, function.__name__):
            paths.add(ROOT / ("/".join(name.split(".")[:-1]) + ".py"))
    return sorted(p.relative_to(ROOT).as_posix() for p in paths)


def scientific_sources(root=ROOT):
    return [
        {"path": p, "sha256": digest(Path(root) / p)} for p in scientific_source_paths()
    ]


def bounce_coordinate(kind):
    """Signed arc coordinate of the on-axis parabolic mirror; None elsewhere."""
    return (lambda x: x[2]) if kind == "magnetic_mirror" else None


def reference_row(kind, experiment, diagnostics):
    """The published scientific fields of one example, from its traced case."""
    import numpy as np

    full, gc = experiment.full, experiment.guiding_center
    return {
        "case": kind,
        "field": full.field.metadata(),
        "full_termination": full.termination,
        "gc_termination": gc.termination,
        "gc_energy_residual_max": float(np.max(np.abs(diagnostics["energy_residual"]))),
        "gc_P_phi_residual_max": float(
            np.max(np.abs(diagnostics.get("P_phi_residual", [0.0])))
        ),
        "ordering_warning": experiment.metadata["ordering_warning"],
        "rho_over_L_B_max": experiment.metadata["rho_over_L_B_max"],
        "orbit_class": diagnostics["orbit_class"],
        "bounce_cycles": diagnostics["bounce_cycles"],
        "bounce_period_s": diagnostics.get("bounce_period_s", []),
        "transit_period_s": diagnostics.get("transit_period_s", []),
        "fidelity": experiment.metadata["fidelity"],
    }


def recompute_row(kind):
    """Trace ``kind`` again exactly as the generator did and return its row."""
    from vaft.process.particle import analyze_particle_orbit, particle_motion_case

    experiment = particle_motion_case(kind)
    diagnostics = analyze_particle_orbit(
        experiment.guiding_center, bounce_coordinate=bounce_coordinate(kind)
    )
    return reference_row(kind, experiment, diagnostics)


def _same(recorded, fresh, key):
    if isinstance(fresh, dict):
        return (
            isinstance(recorded, dict)
            and recorded.keys() == fresh.keys()
            and all(_same(recorded[k], fresh[k], key) for k in fresh)
        )
    if isinstance(fresh, (list, tuple)):
        return (
            isinstance(recorded, list)
            and len(recorded) == len(fresh)
            and all(_same(a, b, key) for a, b in zip(recorded, fresh))
        )
    if isinstance(fresh, bool) or not isinstance(fresh, (int, float)):
        return recorded == fresh
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        return False
    abs_tol = 1e-12 if key in _RESIDUAL_KEYS else 0.0
    return math.isclose(recorded, fresh, rel_tol=1e-9, abs_tol=abs_tol)


def check_references(root=ROOT, *, directory=None, recompute=()):
    """Validate the committed manifest; raise ``ValueError`` naming what is stale.

    Sources are read under ``root``; ``directory`` (default
    ``root/docs/assets/particle-motion``) holds the manifest. ``recompute``
    lists example ids whose rows are re-traced and compared (``True`` for all
    six); the default verifies digests only.
    """
    root = Path(root)
    if directory is None:
        directory = root / "docs/assets/particle-motion"
    manifest = json.loads(
        (Path(directory) / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported particle-motion manifest version; regenerate references"
        )
    commit = manifest.get("generated_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("particle-motion manifest records no generated_commit")
    recorded = manifest.get("sources") or []
    expected = scientific_source_paths()
    if [s.get("path") for s in recorded] != expected:
        raise ValueError(
            "particle-motion manifest sources drift: recorded "
            f"{[s.get('path') for s in recorded]}, expected {expected}; "
            "regenerate references"
        )
    for source in recorded:
        p = Path(source["path"])
        if (
            p.is_absolute()
            or ".." in p.parts
            or digest(root / p) != source["sha256"]
        ):
            raise ValueError(
                f"Stale particle-motion scientific source: {source['path']}; "
                "regenerate references"
            )
    rows = manifest.get("examples") or []
    if [(r.get("id"), r.get("case"), r.get("builder")) for r in rows] != list(EXAMPLES):
        raise ValueError("particle-motion manifest examples drift; regenerate references")
    wanted = {r["id"] for r in rows} if recompute is True else set(recompute)
    if wanted - {r["id"] for r in rows}:
        raise ValueError(f"unknown particle-motion example: {sorted(wanted)}")
    fresh = {}
    for row in rows:
        if row["id"] not in wanted:
            continue
        if row["case"] not in fresh:
            fresh[row["case"]] = recompute_row(row["case"])
        for key, value in fresh[row["case"]].items():
            if not _same(row.get(key), value, key):
                raise ValueError(
                    f"Stale particle-motion reference: {row['id']} {key} "
                    f"(recorded {row.get(key)!r}, recomputed {value!r}); "
                    "regenerate references"
                )
    return rows
