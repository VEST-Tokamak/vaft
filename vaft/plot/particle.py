"""Typed orbit views with synchronized geometry, stationary well and diagnostics.

These native-trajectory views do not claim an IMAS IDS mapping. They are public
through this module, separate from the IDS renderer registry. Preparation is
explicit and independent of frame rendering; every frame selects one physical
time on the same precomputed view model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .models import ViewModel

__all__ = ["ParticleOrbitView", "particle_overview"]


@dataclass(frozen=True)
class ParticleOrbitView(ViewModel):
    """Prepared native-time diagnostics and common-time full/GC comparison.

    Construct with from_comparison once. The renderer performs no field or
    orbit computation. well_coordinate is signed arc length along a specified
    line; well_B is stationary and is never recomputed from the moving marker.
    """

    comparison: object
    full_diagnostics: Mapping
    gc_diagnostics: Mapping
    well_coordinate: np.ndarray | None = None
    well_B: np.ndarray | None = None
    well_marker_coordinate: np.ndarray | None = None
    title: str = "Particle and guiding-center motion"

    def __post_init__(self):
        from types import MappingProxyType
        from vaft.data.particle import ParticleComparison

        if not isinstance(self.comparison, ParticleComparison):
            raise TypeError("ParticleOrbitView requires a ParticleComparison")
        for attr, trajectory in (
            ("full_diagnostics", self.comparison.full),
            ("gc_diagnostics", self.comparison.guiding_center),
        ):
            diagnostics = dict(getattr(self, attr))
            for key in (
                "time",
                "B_magnitude",
                "v_parallel",
                "v_perp",
                "energy_parallel",
                "energy_perpendicular",
                "energy_residual",
                "mu_residual",
            ):
                value = np.asarray(diagnostics.get(key), float)
                if value.shape != trajectory.time.shape or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        f"{attr}.{key} must match finite native trajectory samples"
                    )
            if not np.array_equal(diagnostics["time"], trajectory.time):
                raise ValueError(
                    "diagnostic timestamps must match their native trajectory"
                )
            for key, value in diagnostics.items():
                if isinstance(value, np.ndarray):
                    frozen = value.copy()
                    frozen.setflags(write=False)
                    diagnostics[key] = frozen
            object.__setattr__(self, attr, MappingProxyType(diagnostics))
        optional = (self.well_coordinate, self.well_B, self.well_marker_coordinate)
        if any(value is not None for value in optional):
            if not all(value is not None for value in optional):
                raise ValueError(
                    "a well needs coordinates, B and marker coordinates together"
                )
            s, B, marker = (np.asarray(value, float) for value in optional)
            if (
                s.ndim != 1
                or len(s) < 2
                or B.shape != s.shape
                or np.any(np.diff(s) <= 0)
                or marker.shape != self.comparison.guiding_center.time.shape
                or not np.all(np.isfinite(np.r_[s, B, marker]))
                or np.any(B <= 0)
                or marker.min() < s[0]
                or marker.max() > s[-1]
            ):
                raise ValueError("invalid stationary well or marker coordinates")
            for name, value in zip(
                ("well_coordinate", "well_B", "well_marker_coordinate"), (s, B, marker)
            ):
                value = value.copy()
                value.setflags(write=False)
                object.__setattr__(self, name, value)

    @classmethod
    def from_comparison(
        cls,
        comparison,
        *,
        bounce_coordinate=None,
        well_coordinate=None,
        well_B=None,
        title="Particle and guiding-center motion",
    ):
        """Prepare diagnostics once, with an optional explicitly defined static well.

        bounce_coordinate maps GC positions to signed field-line arc length.
        For the on-axis parabolic mirror use lambda x: x[2]. Other cases
        must supply their own physically qualified coordinate.
        """
        from vaft.data.particle import ParticleComparison
        from vaft.process.particle import analyze_particle_orbit

        if not isinstance(comparison, ParticleComparison):
            raise TypeError("from_comparison requires a ParticleComparison")
        full = analyze_particle_orbit(comparison.full)
        gc = analyze_particle_orbit(
            comparison.guiding_center, bounce_coordinate=bounce_coordinate
        )
        marker = None
        if well_coordinate is not None or well_B is not None:
            if well_coordinate is None or well_B is None or bounce_coordinate is None:
                raise ValueError(
                    "a stationary well requires coordinates, B values and a marker arc coordinate"
                )
            well_coordinate, well_B = (
                np.asarray(well_coordinate, float),
                np.asarray(well_B, float),
            )
            if (
                well_coordinate.ndim != 1
                or len(well_coordinate) < 2
                or well_B.shape != well_coordinate.shape
                or np.any(np.diff(well_coordinate) <= 0)
                or not np.all(np.isfinite(well_coordinate + well_B))
                or np.any(well_B <= 0)
            ):
                raise ValueError(
                    "well must have finite increasing coordinates and positive matching B"
                )
            marker = np.array(
                [bounce_coordinate(x) for x in comparison.guiding_center.position]
            )
            if (
                not np.all(np.isfinite(marker))
                or marker.min() < well_coordinate[0]
                or marker.max() > well_coordinate[-1]
            ):
                raise ValueError(
                    "well marker falls outside the supplied stationary well"
                )
        return cls(comparison, full, gc, well_coordinate, well_B, marker, title)


def _selected(model, time):
    c = model.comparison
    lo = max(c.full.time[0], c.guiding_center.time[0])
    hi = min(c.full.time[-1], c.guiding_center.time[-1])
    if not np.isfinite(time) or time < lo or time > hi:
        raise ValueError("time is outside the common valid trajectory")
    return model, float(time)


def _draw(state, *, ax=None, show=False):
    import matplotlib.pyplot as plt

    model, t = state
    c = model.comparison
    if ax is None:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
        axes = axes.ravel()
    else:
        axes = np.asarray(ax, dtype=object).ravel()
        if len(axes) != 9:
            raise ValueError("particle_overview requires nine axes")
        fig = axes[0].figure
        if any(a.figure is not fig for a in axes):
            raise ValueError("all nine axes must belong to one figure")
    full, gc = c.full, c.guiding_center
    native = (
        (full, model.full_diagnostics, "#1c67d2", "Lorentz"),
        (gc, model.gc_diagnostics, "#d63027", "GC"),
    )
    torus = full.field.provenance.get("R0_m") is not None
    mirror = "mirror" in full.field.name
    for tr, d, color, label in native:
        if torus:
            horizontal, vertical = (
                np.hypot(tr.position[:, 0], tr.position[:, 1]),
                tr.position[:, 2],
            )
            axes[0].set(xlabel="R [m]", ylabel="Z [m]")
        elif mirror:
            horizontal, vertical = tr.position[:, 2], tr.position[:, 0]
            axes[0].set(xlabel="z [m]", ylabel="x [m]")
        else:
            horizontal, vertical = tr.position[:, 0], tr.position[:, 1]
            axes[0].set(xlabel="x [m]", ylabel="y [m]")
        visible = tr.time <= t
        axes[0].plot(horizontal, vertical, color=color, alpha=0.15, lw=0.8)
        end = [np.interp(t, tr.time, horizontal), np.interp(t, tr.time, vertical)]
        axes[0].plot(
            np.r_[horizontal[visible], end[0]],
            np.r_[vertical[visible], end[1]],
            color=color,
            lw=0.8,
            label=label,
        )
        axes[0].plot(*end, "o", color=color, ms=4)
        for axis, values, ylabel in (
            (axes[2], d["v_parallel"], "$v_\\parallel$ [m/s]"),
            (axes[3], d["energy_residual"], "$(H-H_0)/H_{scale}$"),
            (axes[4], d["mu_residual"], "$(\\mu-\\mu_0)/\\mu_{scale}$"),
        ):
            axis.plot(tr.time, values, color=color, lw=0.8, label=label)
            axis.plot(t, np.interp(t, tr.time, values), "o", ms=3, color=color)
            axis.set(xlabel="Time [s]", ylabel=ylabel)
            axis.axvline(t, color="gray", lw=0.5, ls="--")
        axes[2].plot(
            tr.time,
            d["v_perp"],
            color=color,
            lw=0.6,
            ls=":",
            label=f"{label} perpendicular",
        )
        axes[6].plot(
            tr.time,
            d["energy_parallel"],
            color=color,
            lw=0.8,
            label=f"{label} parallel",
        )
        axes[6].plot(
            tr.time,
            d["energy_perpendicular"],
            color=color,
            lw=0.8,
            ls=":",
            label=f"{label} perpendicular",
        )
        if "P_phi" in d:
            for key, style, label_part in (
                ("P_phi", "-", "total"),
                ("P_phi_mechanical", "--", "mechanical"),
                ("P_phi_magnetic", ":", "field"),
            ):
                axes[7].plot(
                    tr.time,
                    d[key],
                    color=color,
                    lw=0.7,
                    ls=style,
                    label=f"{label} {label_part}",
                )
        else:
            axes[7].plot(
                tr.time, d["rho_over_L_B"], color=color, lw=0.8, label=f"{label} rho/LB"
            )
            axes[7].plot(
                tr.time,
                d["parallel_curvature_ordering"],
                color=color,
                lw=0.8,
                ls=":",
                label=f"{label} parallel curvature",
            )
    axes[6].set(xlabel="Time [s]", ylabel="Kinetic energy partitions [J]")
    axes[7].set(
        xlabel="Time [s]",
        ylabel="Canonical momentum [J s]"
        if "P_phi" in model.gc_diagnostics
        else "Ordering parameters",
    )
    for axis in (axes[6], axes[7]):
        axis.axvline(t, color="gray", lw=0.5, ls="--")
        axis.legend(fontsize=6)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8)
    axes[2].set_title("Parallel: solid; perpendicular: dotted", fontsize=10)
    axes[4].set_title("GC magnetic moment is imposed", fontsize=10)
    if model.well_coordinate is not None:
        axes[1].plot(model.well_coordinate, model.well_B, color="black")
        s = np.interp(t, gc.time, model.well_marker_coordinate)
        B = np.interp(t, gc.time, model.gc_diagnostics["B_magnitude"])
        axes[1].plot(s, B, "o", color="#d63027")
        axes[1].set(
            xlabel="Signed field-line arc coordinate [m]",
            ylabel="B [T]",
            title="Stationary spatial magnetic well",
        )
    else:
        for tr, d, color, label in native:
            axes[1].plot(tr.time, d["B_magnitude"], color=color, lw=0.8, label=label)
        axes[1].axvline(t, color="gray", lw=0.5, ls="--")
        axes[1].set(xlabel="Time [s]", ylabel="B along orbit [T]")
    if "P_phi_residual" in model.gc_diagnostics:
        for tr, d, color, label in native:
            axes[5].plot(tr.time, d["P_phi_residual"], color=color, lw=0.8, label=label)
        axes[5].set(ylabel="$(P_\\phi-P_{\\phi,0})/P_{scale}$", xlabel="Time [s]")
    else:
        axes[5].plot(c.time, c.separation, color="black", lw=0.8)
        axes[5].set(ylabel="Local full-center / GC separation [m]", xlabel="Time [s]")
    axes[5].axvline(t, color="gray", lw=0.5, ls="--")
    axes[8].plot(
        c.time, c.separation, color="black", lw=0.8, label="Local-center subtraction"
    )
    average_time = c.metadata.get("gyroaverage_time_s", [])
    average_error = c.metadata.get("gyroaverage_separation_m", [])
    if len(average_time) and len(average_time) == len(average_error):
        axes[8].plot(
            average_time,
            average_error,
            color="#886633",
            lw=0.8,
            label="One-period time average",
        )
    axes[8].axvline(t, color="gray", lw=0.5, ls="--")
    axes[8].set(xlabel="Time [s]", ylabel="Full-center / GC separation [m]")
    axes[8].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{model.title} — t={t:.4g} s\n"
        f"full: {full.termination}; GC: {gc.termination}; GC is an adiabatic weak-flow approximation",
        fontsize=12,
    )
    if c.metadata.get("ordering_warning"):
        fig.text(
            0.5,
            0.005,
            "Ordering warning: inspect rho/LB, parallel-curvature and E×B flow ratios",
            ha="center",
            color="#b22222",
            fontsize=8,
        )
    if show:
        plt.show()
    return fig, axes


def particle_overview(
    model,
    *,
    time=None,
    animation=False,
    ax=None,
    show=False,
    fps=None,
    duration=None,
    dpi=None,
    frame_label=True,
):
    """Render synchronized native particle trajectories and invariant diagnostics.

    Interpretation
    --------------
    Geometry markers, parallel velocity, static magnetic well or sampled B,
    energy, magnetic moment and canonical momentum use one physical timestamp.
    Full-orbit/GC separation includes initialization and gyroaveraging effects.

    Options
    -------
    Pass a prepared ParticleOrbitView. Scalar time selects a static view; a
    strictly increasing sequence and animation=True use the existing lazy
    frames/save/metadata contract. fps and duration affect presentation only.
    Nine supplied axes are authoritative for a static view.

    Limitations
    -----------
    Static fields and nonrelativistic collisionless trajectories only. Constant
    GC mu is imposed, not independent conservation evidence. A changing
    canonical momentum alone is not a transport or torque measurement.
    """
    if not isinstance(model, ParticleOrbitView):
        raise TypeError("particle_overview requires a prepared ParticleOrbitView")
    c = model.comparison
    lo, hi = (
        max(c.full.time[0], c.guiding_center.time[0]),
        min(c.full.time[-1], c.guiding_center.time[-1]),
    )
    if not isinstance(animation, bool) or not isinstance(frame_label, bool):
        raise ValueError("animation and frame_label must be booleans")
    if time is None:
        time = np.linspace(lo, hi, 60) if animation else hi
    values = np.asarray(time, float)
    if not animation:
        if values.ndim != 0:
            raise ValueError("a time sequence requires animation=True")
        if (
            fps is not None
            or duration is not None
            or dpi is not None
            or not frame_label
        ):
            raise ValueError("presentation options require animation=True")
        return _draw(_selected(model, float(values)), ax=ax, show=show)
    if ax is not None or show:
        raise ValueError("animation is lazy; use frames(), save() or show()")
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            "animation needs at least two finite strictly increasing physical times"
        )
    for value in values:
        _selected(model, value)
    from ._animation import Animation, Driver, _presentation

    presentation = {
        k: v
        for k, v in (("fps", fps), ("duration", duration), ("dpi", dpi))
        if v is not None
    }
    rate, length, timing, resolution = _presentation(presentation)
    if length is not None:
        rate = len(values) / length
    return Animation(
        plot="particle_overview",
        label="",
        driver=Driver(
            "time", "Time", "time", "s", tuple(range(len(values))), tuple(values)
        ),
        selection={"kind": "time", "value": values.tolist()},
        fps=rate,
        timing=timing,
        dpi=resolution or 100,
        image=False,
        vmin=None,
        vmax=None,
        frame_label=frame_label,
        options={
            "field": c.full.field.metadata(),
            "experiment": dict(c.metadata),
            "interpolation": "linear; no extrapolation",
        },
        style={},
        build=lambda chosen: _selected(model, values[int(chosen["time"])]),
        draw=_draw,
    )
