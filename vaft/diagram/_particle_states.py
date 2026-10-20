"""Time-selected scenes from precomputed Lorentz/GC trajectories (#2003–#2009).

All integrations and quantitative reductions belong to process.particle. This
module only selects, projects and annotates the resulting scientific states.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from vaft.data.particle import ParticleComparison

from ._projection import project
from ._render import Diagram
from ._scene import Arrow, Label, Marker, Polyline, Scene


@dataclass(eq=False, frozen=True)
class ParticleSceneState:
    """Computed experiment, selected physical time and interpolated positions."""

    comparison: ParticleComparison
    time: float
    particle_position: np.ndarray
    gc_position: np.ndarray


@lru_cache(maxsize=12)
def _reference(kind):
    from vaft.process.particle import particle_motion_case

    return particle_motion_case(kind)


def _project(points, projection):
    points = np.asarray(points)
    if projection == "3d":
        return project(points)
    if projection in ("poloidal",):
        return np.stack(
            [np.hypot(points[..., 0], points[..., 1]), points[..., 2]], axis=-1
        )
    if projection == "parallel":
        return np.stack([points[..., 2], points[..., 0]], axis=-1)
    return points[..., :2]


def _trail(trajectory, time):
    start = float(trajectory.time[0])
    if time <= start:
        return trajectory.sample(time)["position"]
    # Draw on a fixed physical-time grid, not the solver's adaptive nodes: their
    # placement follows platform-dependent step-size control, and a committed
    # reference SVG must hash the same on every platform. Rendering density
    # does not change scientific samples or marker position.
    return trajectory.sample(np.linspace(start, time, 1200))["position"]


def _scene(comparison, time, projection, labels, name):
    full, gc = comparison.full, comparison.guiding_center
    full_position = full.sample(time)["position"][0]
    gc_position = gc.sample(time)["position"][0]
    points = np.vstack(
        [_project(full.position, projection), _project(gc.position, projection)]
    )
    if "length_m" in full.field.provenance and "mirror" in full.field.name:
        L = full.field.provenance["length_m"]
        points = np.vstack(
            [points, _project(np.array([[-0.6, 0, -L], [0.6, 0, L]]), projection)]
        )
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    scale = min(
        8 / max(float(high[0] - low[0]), 1.0), 4 / max(float(high[1] - low[1]), 1.0)
    )

    def screen(x):
        return (_project(x, projection) - center) * scale

    items = [
        Label((-5.0, -4.0), "", role="bounds"),
        Label((5.0, 4.0), "", role="bounds"),
    ]
    kind = comparison.metadata.get("case", name)
    field = full.field
    # Geometry is fixed over physical time; trajectories move through it.
    if kind in ("gyration", "exb_drift", "grad_b_drift"):
        # Straight lines are exact only for these reference field geometries.
        # For an arbitrary supplied field, retain the local field arrow instead
        # of inventing background field lines from the requested scene name.
        if (
            field.provenance.get("geometry") == "uniform"
            or field.name == "divergence-free slab gradient"
        ):
            origin = gc.position[0]
            direction = field.magnetic(origin)
            direction = direction / np.linalg.norm(direction)
            axis = np.eye(3)[np.argmin(np.abs(direction))]
            transverse = np.cross(direction, axis)
            transverse /= np.linalg.norm(transverse)
            for offset in (-1.0, 0.0, 1.0):
                line = (
                    origin + offset * transverse
                    + np.linspace(-1.0, 5.0, 80)[:, None] * direction
                )
                if projection != "perpendicular":
                    items.append(Polyline.of(screen(line), "surface", role="field_line"))
    elif kind in ("magnetic_mirror", "mirror_passing"):
        length = field.provenance["length_m"]
        z = np.linspace(-0.99 * length, 0.99 * length, 100)
        for radius in (-0.6, 0.0, 0.6):
            r = radius / np.sqrt(1 + (z / length) ** 2)
            items.append(
                Polyline.of(
                    screen(np.column_stack([r, np.zeros(len(z)), z])),
                    "surface",
                    role="field_line",
                )
            )
    elif field.provenance.get("R0_m") is not None:
        R0 = field.provenance["R0_m"]
        angle = np.linspace(0, 2 * np.pi, 120)
        # Flux-section reference circles, explicitly geometry rather than field lines.
        for r in (0.25, 0.5, 0.75):
            if projection == "poloidal":
                ring = np.column_stack(
                    [R0 + r * np.cos(angle), np.zeros(len(angle)), r * np.sin(angle)]
                )
            else:
                ring = np.column_stack(
                    [
                        (R0 + r) * np.cos(angle),
                        (R0 + r) * np.sin(angle),
                        np.zeros(len(angle)),
                    ]
                )
            items.append(Polyline.of(screen(ring), "surface", role="reference_surface"))

    for tr, style, role in (
        (full, "orbit ion", "particle"),
        (gc, "guiding centre", "gc"),
    ):
        trail = screen(_trail(tr, time))
        if len(trail) >= 2:
            items.append(Polyline.of(trail, style, role=f"{role}_trail"))
        items.append(
            Marker(
                tuple(trail[-1]),
                "o" if role == "particle" else "x",
                "opoint" if role == "particle" else "xpoint",
                role=f"{role}_marker",
            )
        )
    a, z = screen(gc_position), screen(full_position)
    items.append(Arrow(tuple(a), tuple(z), "vector", role="gyration_displacement"))
    B = field.magnetic(gc_position)
    E = field.electric(gc_position)
    for vector, role, text in (
        (B, "magnetic_vector", "$\\mathbf{B}$"),
        (E, "electric_vector", "$\\mathbf{E}$"),
    ):
        if np.linalg.norm(vector) == 0:
            continue
        tip = screen(gc_position + vector / np.linalg.norm(vector) / scale)
        items.append(Arrow(tuple(a), tuple(tip), "field arrow", role=role))
        if labels:
            items.append(
                Label(tuple(tip + [0.15, 0.15]), text, "small label", role=role)
            )
    if labels:
        title = name.replace("_", " ").capitalize()
        if name == "stochastic_ripple_orbit":
            title = "Static 3D orbit perturbation (stochasticity not established)"
        elif name == "ripple_well_formation":
            title = "Orbit sampling of a stationary ripple field"
        elif name == "trapped_and_passing_orbits":
            title = str(kind).replace("_", " ").capitalize()
        warning = (
            "ordering warning; " if comparison.metadata.get("ordering_warning") else ""
        )
        items.extend(
            [
                Label((0.0, 3.65), title, "title", role="title"),
                Label(
                    (0.0, 3.05),
                    f"$t={time:.4g}$ s; blue: Lorentz orbit, red: integrated GC",
                    "subtitle",
                    role="time",
                ),
                Label(
                    (0.0, -3.25),
                    "$m\\dot{\\mathbf{v}}=q(\\mathbf{E}+\\mathbf{v}\\times\\mathbf{B})$; "
                    "$\\dot{\\mathbf{X}},\\dot{p}_\\parallel$: Hamiltonian GC",
                    "note",
                    role="equations",
                ),
                Label(
                    (0.0, -3.7),
                    f"static SI reference; q={full.charge:g} C, m={full.mass:g} kg; "
                    + warning
                    + "GC: weak-flow adiabatic approximation",
                    "note",
                    role="validity",
                ),
            ]
        )
    return Scene(tuple(items)), ParticleSceneState(
        comparison, time, full_position, gc_position
    )


def particle_states(
    name,
    *,
    case=None,
    trajectory=None,
    time=None,
    projection="3d",
    labels=True,
    animation=False,
    fps=None,
    duration=None,
    dpi=None,
    frame_label=True,
):
    """Shared static/dynamic state selector; no integration occurs in a frame."""
    if (
        not isinstance(labels, bool)
        or not isinstance(animation, bool)
        or not isinstance(frame_label, bool)
    ):
        raise ValueError("labels, animation and frame_label must be booleans")
    if projection not in ("3d", "perpendicular", "poloidal", "parallel", "top"):
        raise ValueError(
            "projection must be 3d, perpendicular, poloidal, parallel or top"
        )
    if not animation and (
        fps is not None or duration is not None or dpi is not None or not frame_label
    ):
        raise ValueError("presentation options require animation=True")
    comparison = _reference(case or name) if trajectory is None else trajectory
    if not isinstance(comparison, ParticleComparison):
        raise TypeError(
            "trajectory must be a computed ParticleComparison from process.particle"
        )
    lo = max(comparison.full.time[0], comparison.guiding_center.time[0])
    hi = min(comparison.full.time[-1], comparison.guiding_center.time[-1])
    if time is None:
        time = np.linspace(lo, hi, 60) if animation else hi
    values = np.asarray(time, float)
    if values.ndim > 1 or not np.all(np.isfinite(values)):
        raise ValueError(
            "time must be a finite scalar or one-dimensional physical coordinate"
        )
    if np.any(values < lo) or np.any(values > hi):
        raise ValueError("time is outside the common valid trajectory interval")
    if not animation:
        if values.ndim != 0:
            raise ValueError("time sequence requires animation=True")
        scene, state = _scene(comparison, float(values), projection, labels, name)
        return Diagram(name, scene, model=state)
    values = np.atleast_1d(values)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(
            "animation requires at least two strictly increasing physical times"
        )
    from ._animation import animate_states

    return animate_states(
        name,
        lambda t: _scene(comparison, t, projection, labels, name)[0],
        driver="time",
        values=values,
        coordinate="time",
        unit="s",
        fps=fps,
        duration=duration,
        dpi=dpi,
        frame_label=frame_label,
        bounds=((-5.5, 5.5), (-4.5, 4.5)),
        options={
            "projection": projection,
            "labels": labels,
            "interpolation": "linear; no extrapolation",
            "native_full_time_s": comparison.full.time.tolist(),
            "native_gc_time_s": comparison.guiding_center.time.tolist(),
            "full_termination": comparison.full.termination,
            "gc_termination": comparison.guiding_center.termination,
            "field": comparison.full.field.metadata(),
            "full_orbit": comparison.full.provenance_record(),
            "guiding_center": comparison.guiding_center.provenance_record(),
            "experiment": dict(comparison.metadata),
        },
    )


def gyration(
    *,
    trajectory=None,
    time=None,
    projection="3d",
    labels=True,
    animation=False,
    fps=None,
    duration=None,
    dpi=None,
    frame_label=True,
):
    """Computed uniform-field gyration and helical motion, selected by physical time.

    Pass a process.particle ParticleComparison to replay an existing experiment.
    A scalar time returns its static scientific scene; a sequence with
    animation=True returns lazy frames/save/metadata. The default illustrative
    SI particle has q=1 C, m=1 kg. GC is an independently integrated reduced model.
    """
    return particle_states(
        "gyration",
        trajectory=trajectory,
        time=time,
        projection=projection,
        labels=labels,
        animation=animation,
        fps=fps,
        duration=duration,
        dpi=dpi,
        frame_label=frame_label,
    )


def grad_b_drift(
    *,
    trajectory=None,
    time=None,
    projection="perpendicular",
    labels=True,
    animation=False,
    fps=None,
    duration=None,
    dpi=None,
    frame_label=True,
):
    """Computed grad-B drift in a solenoidal slab; time has units of seconds.

    The guiding-center limit assumes rho/L_B small. Signed charge reverses
    the drift. See gyration for the shared static/animation trajectory contract.
    """
    return particle_states(
        "grad_b_drift",
        trajectory=trajectory,
        time=time,
        projection=projection,
        labels=labels,
        animation=animation,
        fps=fps,
        duration=duration,
        dpi=dpi,
        frame_label=frame_label,
    )


def magnetic_mirror(
    *,
    trajectory=None,
    time=None,
    projection="parallel",
    labels=True,
    animation=False,
    fps=None,
    duration=None,
    dpi=None,
    frame_label=True,
):
    """Computed reflection and bounce in a static solenoidal magnetic mirror.

    The well is stationary. Particle and GC markers advance in physical time;
    quantitative B(s), energy and invariant panels belong to vaft.plot.particle.
    Supply a mirror_passing comparison to display a loss-cone trajectory.
    """
    return particle_states(
        "magnetic_mirror",
        trajectory=trajectory,
        time=time,
        projection=projection,
        labels=labels,
        animation=animation,
        fps=fps,
        duration=duration,
        dpi=dpi,
        frame_label=frame_label,
    )
