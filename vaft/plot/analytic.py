"""Plots of analytic equilibrium objects, not of an ODS (issue #952).

The teaching models of :mod:`vaft.process.equilibrium` -- Miller flux
surfaces and the Solov'ev / Cerfon-Freidberg equilibria of
:func:`vaft.process.equilibrium.solovev_example` -- are plain records, not
IDS data, so they are drawn here rather than through the recipe registry that
every ``plot_*`` adapter reads an ODS with.  Each plot is split the way the
registry's are: a ``*_model`` function builds the typed view model
(:class:`~vaft.plot.models.GeometryLayers`, :class:`~vaft.plot.models.Field2D`)
and the ``plot_*`` function hands it to the generic renderer, so the figure
can be inspected or restyled like any other.

``vaft.process`` is imported inside the functions: importing ``vaft.plot``
loads no physics package and no data model.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .intent import palette
from .models import Field2D, GeometryLayer, GeometryLayers

__all__ = [
    "miller_surfaces_model",
    "plot_miller_surfaces",
    "plot_solovev_equilibrium",
    "solovev_equilibrium_model",
]

#: psi_N levels drawn inside the plasma, and just outside it.
_INSIDE_LEVELS = tuple(np.round(np.linspace(0.1, 0.9, 9), 2))
_OUTSIDE_LEVELS = (1.05, 1.15, 1.3, 1.5)


def _surfaces(surfaces: Any) -> tuple[Any, ...]:
    from vaft.data.equilibrium import MillerSurface

    if isinstance(surfaces, MillerSurface):
        return (surfaces,)
    items = tuple(surfaces)
    if not items or not all(isinstance(item, MillerSurface) for item in items):
        raise TypeError("expected a MillerSurface or a non-empty sequence of them")
    return items


_SHAPE_SYMBOLS = {"r": "a", "kappa": "κ", "delta": "δ", "zeta": "ζ", "r0": "R0", "z0": "Z0"}


def _surface_label(surface: Any, varying: Sequence[str]) -> str:
    parts = [f"{_SHAPE_SYMBOLS[key]}={getattr(surface, key):.3g}" for key in varying]
    return ", ".join(parts) or f"a={surface.r:.3g}"


def miller_surfaces_model(
    surfaces: Any, *, theta_points: int = 256, labels: Sequence[str] | None = None,
    title: str | None = None,
) -> GeometryLayers:
    """Miller flux surfaces as closed outlines in the poloidal plane.

    ``surfaces`` is one :class:`vaft.data.equilibrium.MillerSurface` or a
    sequence of them (:func:`vaft.process.equilibrium.miller_surfaces` builds
    a scan).  Each surface is evaluated with
    :func:`vaft.process.equilibrium.evaluate_miller` on ``theta_points``
    angles and labelled by the shape parameters that vary across the set, so a
    triangularity scan reads ``δ=0.1``, ``δ=0.3`` ... in the legend.  Its
    geometric centre is marked when it moves (a Shafranov-shift scan).
    """
    from vaft.process.equilibrium import evaluate_miller

    items = _surfaces(surfaces)
    theta = np.linspace(0.0, 2.0 * np.pi, int(theta_points), endpoint=True)
    varying = [
        key for key in ("r", "kappa", "delta", "zeta", "r0", "z0")
        if len({round(float(getattr(item, key)), 12) for item in items}) > 1
    ]
    if labels is not None and len(labels) != len(items):
        raise ValueError(f"received {len(labels)} labels for {len(items)} surfaces")
    layers: list[GeometryLayer] = []
    for position, surface in enumerate(items):
        r, z = evaluate_miller(surface, theta)
        layers.append(GeometryLayer(
            r=np.asarray(r, dtype=float), z=np.asarray(z, dtype=float), kind="polyline",
            label=labels[position] if labels is not None else _surface_label(surface, varying),
            style={"color": palette(position), "lw": 1.6},
        ))
    if "r0" in varying or "z0" in varying:
        layers.append(GeometryLayer(
            r=np.array([item.r0 for item in items], dtype=float),
            z=np.array([item.z0 for item in items], dtype=float),
            kind="points", label="geometric centres",
            style={"marker": "+", "color": "feature:axis", "markersize": 8},
        ))
    heading = title or (
        "Miller surfaces"
        + (f": scan in {', '.join(_SHAPE_SYMBOLS[key] for key in varying)}" if varying else "")
    )
    return GeometryLayers(layers=tuple(layers), title=heading)


def plot_miller_surfaces(
    surfaces: Any, *, ax: Any = None, show: bool = False, theta_points: int = 256,
    labels: Sequence[str] | None = None, title: str | None = None, **style: Any,
):
    """Draw Miller flux surfaces; returns ``(Figure, Axes)``.

    See :func:`miller_surfaces_model` for what is drawn; ``style`` goes to
    :func:`vaft.plot.render_geometry_layers`.
    """
    from .renderers.geometry import render_geometry_layers

    model = miller_surfaces_model(surfaces, theta_points=theta_points, labels=labels, title=title)
    return render_geometry_layers(model, ax=ax, show=show, **style)


def _equilibrium_layers(eq: Any, representation: Any, *, x_points: bool) -> list[GeometryLayer]:
    layers: list[GeometryLayer] = []
    if eq.limiter is not None and np.asarray(eq.limiter.r).size > 1:
        layers.append(GeometryLayer(
            r=np.append(eq.limiter.r, eq.limiter.r[0]), z=np.append(eq.limiter.z, eq.limiter.z[0]),
            kind="polyline", label="wall", style={"color": "feature:wall", "lw": 1.2},
        ))
    if eq.lcfs is not None:
        layers.append(GeometryLayer(
            r=np.asarray(eq.lcfs.r, dtype=float), z=np.asarray(eq.lcfs.z, dtype=float),
            kind="polygon", label="LCFS", style={"color": "feature:boundary", "lw": 1.6},
        ))
    if eq.magnetic_axis is not None:
        layers.append(GeometryLayer(
            r=np.array([eq.magnetic_axis[0]]), z=np.array([eq.magnetic_axis[1]]), kind="points",
            label="magnetic axis",
            style={"marker": "+", "color": "feature:axis", "markersize": 10, "markeredgewidth": 1.5},
        ))
    if x_points and representation is not None:
        for active, label, style in (
            (True, "X-point", {"marker": "x", "color": "feature:boundary", "markersize": 10,
                               "markeredgewidth": 2.0}),
            (False, "saddle off the boundary", {"marker": "x", "color": "emphasis:low",
                                                "markersize": 7, "markeredgewidth": 1.0}),
        ):
            points = [point for point in representation.x_points if bool(point.active) is active]
            if points:
                layers.append(GeometryLayer(
                    r=np.array([point.r for point in points]), z=np.array([point.z for point in points]),
                    kind="points", label=label, style=style,
                ))
        strikes = representation.strike_points
        if strikes:
            layers.append(GeometryLayer(
                r=np.array([strike.r for strike in strikes]), z=np.array([strike.z for strike in strikes]),
                kind="points", label="strike points",
                style={"marker": "o", "color": "feature:boundary", "markersize": 5},
            ))
    return layers


def solovev_equilibrium_model(
    equilibrium: Any, *, x_points: bool = True, levels: Iterable[float] | None = None,
    title: str | None = None,
) -> Field2D:
    """Normalized flux of an analytic (or any gridded) equilibrium, with its boundary features.

    ``equilibrium`` is an :class:`vaft.data.equilibrium.EquilibriumData`, such
    as :func:`vaft.process.equilibrium.solovev_example` returns, or anything
    :func:`vaft.process.equilibrium.as_equilibrium` adapts.  The map is
    ``psi_N = (psi - psi_axis)/(psi_boundary - psi_axis)`` contoured at
    ``levels`` (0.1 ... 0.9 inside, 1, and a few surfaces outside), so it reads
    the same in any COCOS; the wall, the LCFS and the axis are drawn over it
    and, with ``x_points``, the saddles of psi classified by
    :func:`vaft.process.equilibrium.derive_boundary_representation` -- the
    boundary-relevant X-points strongly, the others faintly -- with the strike
    points and the topology in the title.
    """
    from vaft.process.equilibrium import as_equilibrium, derive_boundary_representation

    eq = as_equilibrium(equilibrium)
    if eq.psi is None or eq.r is None or eq.z is None:
        raise ValueError("the equilibrium carries no (R, Z) psi map to draw")
    if eq.psi_axis is None or eq.psi_boundary is None or eq.psi_axis == eq.psi_boundary:
        raise ValueError("psi_axis and psi_boundary are needed, and distinct, to normalize the map")
    psi_norm = (np.asarray(eq.psi, dtype=float) - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    representation = derive_boundary_representation(eq) if x_points else None
    chosen = tuple(levels) if levels is not None else (*_INSIDE_LEVELS, 1.0, *_OUTSIDE_LEVELS)
    source = (eq.metadata or {}).get("source_type", "")
    heading = title or " — ".join(
        part for part in (
            "Solov'ev equilibrium" if source == "solovev" else "Equilibrium",
            representation.topology.value.replace("_", " ") if representation is not None else "",
        ) if part
    )
    return Field2D(
        r=np.asarray(eq.r, dtype=float),
        z=np.asarray(eq.z, dtype=float),
        values=psi_norm.T,
        value_label=r"Normalized Poloidal Flux $\psi_N$",
        title=heading,
        contour_levels=list(chosen),
        filled=False,
        overlays=tuple(_equilibrium_layers(eq, representation, x_points=x_points)),
    )


def plot_solovev_equilibrium(
    equilibrium: Any, *, ax: Any = None, show: bool = False, x_points: bool = True,
    levels: Iterable[float] | None = None, title: str | None = None, **style: Any,
):
    """Draw an analytic equilibrium's flux surfaces with its X-points; returns ``(Figure, Axes)``.

    See :func:`solovev_equilibrium_model`; ``style`` goes to
    :func:`vaft.plot.render_field_2d`.
    """
    from .renderers.fields import render_field_2d

    model = solovev_equilibrium_model(equilibrium, x_points=x_points, levels=levels, title=title)
    return render_field_2d(model, ax=ax, show=show, **style)
