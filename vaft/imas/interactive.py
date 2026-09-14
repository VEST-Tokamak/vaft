"""Interactive entry points over native IMAS input (issues #261, #482).

The twins of :func:`vaft.omas.plot_equilibrium_interactive` and
:func:`vaft.omas.plot_diagnostics_time_interactive`.  Neither is a view:
each hands the same options to the same entry point, differing only in how
the data is reached.  The diagnostics explorer goes through the shared
adapter body, which reads native IDS objects; the equilibrium explorer's
builders take an ODS, so its input is converted once, per IDS the overview
declares, exactly as a code-backed recipe is (``IDSEntry.as_ods_for``).
"""

from __future__ import annotations

from typing import Any

from vaft.plot.renderers.interactive import BACKENDS

__all__ = ["plot_diagnostics_time_interactive", "plot_equilibrium_interactive", "BACKENDS"]


def plot_diagnostics_time_interactive(
    source: Any,
    *,
    backend: str = "auto",
    show: bool = False,
    label: Any = "shot",
    **options: Any,
) -> Any:
    """The diagnostics overview with live controls, from native IMAS input.

    See :func:`vaft.omas.plot_diagnostics_time_interactive`; ``backend``
    names the interaction, ``render_backend=`` the drawing library.
    """
    from .plotting import render

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    render_backend = options.pop("render_backend", None)
    return render(
        "diagnostics_overview", source, show=show, label=label, interactive=True,
        interaction_backend=backend, backend=render_backend, **options,
    )


def plot_equilibrium_interactive(source: Any, **options: Any) -> Any:
    """Explore one shot's equilibrium slices, from native IMAS input.

    See :func:`vaft.omas.plot_equilibrium_interactive`.  The one entry is
    converted to an ODS holding the IDS the equilibrium overview and its
    plasma-current history read, then handed to the OMAS entry point.
    """
    from vaft.omas.interactive import plot_equilibrium_interactive as explore
    from vaft.plot.backend.recipes import required_ids

    from .entries import normalize_entries

    entries = normalize_entries(source)
    if len(entries) != 1:
        raise ValueError(
            "plot_equilibrium_interactive explores one shot at a time; "
            f"got {len(entries)} entries"
        )
    _, entry = entries[0]
    ids = dict.fromkeys(
        (*required_ids("equilibrium_overview"), "magnetics", "dataset_description")
    )
    return explore(entry.as_ods_for(tuple(ids)), **options)
