"""Top-view plotting wrappers backed by OMAS geometry renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def _render_topview(ods, method_name, *, time_index=None, time=None, ax=None, show=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 7)))
    else:
        fig = ax.figure
        kwargs.pop("figsize", None)
    result = getattr(ods, method_name)(time_index=time_index, time=time, ax=ax, **kwargs)
    if isinstance(result, dict) and result.get("ax") is not None:
        ax = result["ax"]
        fig = ax.figure
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if show:
        plt.show()
    return fig, ax


def equilibrium_CX_topview(ods, *, time_index=None, time=None, ax=None, show=True, **kwargs):
    """Plot the equilibrium cross-section in a machine top view."""
    return _render_topview(
        ods, "plot_equilibrium_CX_topview",
        time_index=time_index, time=time, ax=ax, show=show, **kwargs,
    )


def lh_antennas_CX_topview(ods, *, time_index=None, time=None, ax=None, show=True, **kwargs):
    """Plot lower-hybrid antenna geometry in a machine top view."""
    return _render_topview(
        ods, "plot_lh_antennas_CX_topview",
        time_index=time_index, time=time, ax=ax, show=show, **kwargs,
    )


def ec_launchers_CX_topview(ods, *, time_index=None, time=None, ax=None, show=True, **kwargs):
    """Plot electron-cyclotron launcher geometry in a machine top view."""
    return _render_topview(
        ods, "plot_ec_launchers_CX_topview",
        time_index=time_index, time=time, ax=ax, show=show, **kwargs,
    )


def pellets_trajectory_CX_topview(ods, *, time_index=None, time=None, ax=None, show=True, **kwargs):
    """Plot pellet trajectories in a machine top view."""
    return _render_topview(
        ods, "plot_pellets_trajectory_CX_topview",
        time_index=time_index, time=time, ax=ax, show=show, **kwargs,
    )


# Match the names of the underlying ODS methods while keeping concise VAFT aliases.
plot_equilibrium_CX_topview = equilibrium_CX_topview
plot_lh_antennas_CX_topview = lh_antennas_CX_topview
plot_ec_launchers_CX_topview = ec_launchers_CX_topview
plot_pellets_trajectory_CX_topview = pellets_trajectory_CX_topview


_LAYERS = {
    "equilibrium": ("equilibrium", equilibrium_CX_topview),
    "lh_antennas": ("lh_antennas", lh_antennas_CX_topview),
    "ec_launchers": ("ec_launchers", ec_launchers_CX_topview),
    "pellets": ("pellets", pellets_trajectory_CX_topview),
}


def plot_topview(
    ods,
    *,
    time_index=None,
    time=None,
    layers=None,
    ax=None,
    show=True,
    figsize=(7, 7),
    **kwargs,
):
    """Overlay each requested top-view layer that is present in ``ods``."""
    selected = tuple(_LAYERS) if layers is None else tuple(layers)
    unknown = sorted(set(selected) - set(_LAYERS))
    if unknown:
        raise ValueError(f"Unknown top-view layers: {unknown}")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rendered = []
    for layer in selected:
        root, renderer = _LAYERS[layer]
        if root not in ods:
            continue
        renderer(
            ods, time_index=time_index, time=time, ax=ax,
            show=False, **kwargs,
        )
        rendered.append(layer)
    if not rendered:
        raise ValueError("None of the requested top-view IDS layers are present")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    if show:
        plt.show()
    return fig, ax


__all__ = [
    "ec_launchers_CX_topview",
    "equilibrium_CX_topview",
    "lh_antennas_CX_topview",
    "pellets_trajectory_CX_topview",
    "plot_ec_launchers_CX_topview",
    "plot_equilibrium_CX_topview",
    "plot_lh_antennas_CX_topview",
    "plot_pellets_trajectory_CX_topview",
    "plot_topview",
]
