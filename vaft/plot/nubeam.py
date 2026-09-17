"""Views of a NUBEAM run, built from the native result container.

These are deliberately *not* registered in the plot catalog, and they are not
the registered NBI plots' poorer cousins -- they answer a different question.

``vaft.machine_mapping.core_sources`` now maps a NUBEAM result into IMAS, and
the heating and current-drive profiles that have a home there are registered as
``nbi_profile_*``: those read an ODS, so a saved result plots long after its run
directory is gone. What is here is everything that has *no* IMAS home -- the
deposition markers, the lost fast ions, the step log's power budget -- plus the
profiles in NUBEAM's own per-zone units rather than the densities IMAS asks
for. Registering these would mean inventing schema for them, which is the wrong
way round. So they take the native ``NUBEAMOutputs`` directly.

Everything else follows the house architecture: the builders produce the same
frozen view models every other plot uses, and the renderers delegate to the
shared bodies, inheriting the ``ax``/``show``/return contract unchanged.

**Units are the hazard here.** One NUBEAM run writes four different systems --
the birth file is centimetres and degrees and carries no ``units`` attributes
at all, the lost-particle record is metres, the step log is per cubic
centimetre, and ``state_changes.cdf`` is per zone. Every conversion happens in
the builders below and each one names the units it converted from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from .models import GeometryLayer, GeometryLayers, Panels, Profile1D, Series, TextPanel
from .renderers.geometry import render_geometry_layers
from .renderers.panels import render_panels
from .renderers.profiles import render_profile_1d

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vaft.code.nubeam.outputs import NUBEAMOutputs

__all__ = [
    "build_nubeam_deposition_poloidal",
    "build_nubeam_deposition_topview",
    "build_nubeam_lost_fast_ions",
    "build_nubeam_power_accounting",
    "build_nubeam_profile",
    "nubeam_deposition_poloidal",
    "nubeam_deposition_topview",
    "nubeam_lost_fast_ions",
    "nubeam_power_accounting",
    "nubeam_profile",
]

#: Birth-file lengths are centimetres. The file says so nowhere -- it carries
#: no ``units`` attributes -- but the VEST run puts R at 17-67, which is only
#: inside the 0.10-0.76 m limiter once divided by 100.
_CM_PER_M = 100.0

#: What each plottable profile is, and the unit NUBEAM writes it in.
#:
#: These are per-zone integrals, not densities: ``pbe`` is the watts deposited
#: in a zone, not W/m^3. Converting would need the zone volumes and would make
#: the result a derived quantity, so it is left in NUBEAM's own units and
#: labelled as such.
PROFILE_UNITS: dict[str, tuple[str, str]] = {
    "pbe": ("W", "Electron heating"),
    "pbi": ("W", "Ion heating"),
    "pbth": ("W", "Thermalization"),
    "curbeam": ("A", "Beam-driven current"),
    "curfusn": ("A", "Fusion-product current"),
    "tqbe": ("N.m", "Torque to electrons"),
    "tqbi": ("N.m", "Torque to ions"),
    "tqbjxb": ("N.m", "JxB torque"),
    "nbeami": ("m^-3", "Fast ion density"),
    "eperp_beami": ("keV", "Fast ion <E_perp>"),
    "epll_beami": ("keV", "Fast ion <E_par>"),
    "sbedep": ("s^-1", "Beam electron deposition"),
    "sbtherm": ("s^-1", "Beam ion thermalization"),
    "pfuse": ("W", "Fusion electron heating"),
    "pfusi": ("W", "Fusion ion heating"),
}


class NUBEAMPlotError(ValueError):
    """Raised when a NUBEAM result cannot supply what a view needs."""


def _native(outputs: Any) -> "NUBEAMOutputs":
    """Accept either the result bundle or the native container."""
    inner = getattr(outputs, "outputs_native", None)
    return inner if inner is not None else outputs


def _title(default: str, override: Optional[str], runid: str) -> str:
    """Resolve a plot title.

    ``None`` takes the default, and an explicit string -- including ``""`` --
    is used as given. Panels of these plots sit side by side often enough that
    suppressing a repeated title has to be possible.
    """
    if override is not None:
        return override
    return f"{default} -- {runid}" if runid else default


def _profile_values(native: "NUBEAMOutputs", quantity: str) -> np.ndarray:
    if quantity not in native.profiles:
        available = ", ".join(sorted(native.profiles)) or "none"
        raise NUBEAMPlotError(
            f"{quantity!r} is not in this NUBEAM result. A hydrogen run has no "
            f"fusion products, so pfuse/pfusi/curfusn are legitimately absent. "
            f"Available: {available}"
        )
    values = np.asarray(native.profiles[quantity], dtype=float)
    if values.ndim == 0:
        raise NUBEAMPlotError(f"{quantity!r} is a scalar, not a profile")
    # Always (species, zone). Some profiles are plain 1-D and some arrive
    # species-resolved -- or gas- and beam-species-resolved, which is 3-D --
    # so normalise here and let one code path below build the series.
    return values.reshape(-1, values.shape[-1])


def build_nubeam_profile(
    outputs: Any,
    quantity: str,
    *,
    rho: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> Profile1D:
    """A radial NUBEAM profile in NUBEAM's own units.

    The abscissa is the zone index normalised to [0, 1] unless *rho* is given.
    NUBEAM's own ``rho`` grid is toroidal-flux based -- checked against
    ``sqrt(phit/phit_edge)`` on the VEST run to 3e-4 -- and lives in the plasma
    state rather than in ``state_changes.cdf``, so pass it in when plotting
    against a labelled coordinate.
    """
    native = _native(outputs)
    values = _profile_values(native, quantity)
    unit, heading = PROFILE_UNITS.get(quantity, ("", quantity))
    zones = values.shape[-1]

    if rho is None:
        # Zone centres on a uniform grid: honest about being an index, and
        # never mislabelled as rho_tor_norm.
        x = (np.arange(zones) + 0.5) / zones
        coordinate_label = "normalized zone index"
    else:
        x = np.asarray(rho, dtype=float)
        if x.size == zones + 1:
            # NUBEAM's rho grid holds zone boundaries; profiles are zone
            # averages, so plot them at the centres.
            x = 0.5 * (x[:-1] + x[1:])
        if x.size != zones:
            raise NUBEAMPlotError(
                f"rho has {x.size} points but {quantity!r} has {zones} zones "
                f"(a boundary grid of {zones + 1} is also accepted)"
            )
        coordinate_label = r"$\rho_{tor,norm}$"

    series = tuple(
        Series(
            x=x,
            y=row,
            label=f"species {index + 1}" if values.shape[0] > 1 else heading,
        )
        for index, row in enumerate(values)
    )
    return Profile1D(
        series=series,
        coordinate_label=coordinate_label,
        y_label=heading,
        y_unit=unit,
        # The y-label already names the quantity, so the title carries only
        # provenance. Pass title="" to drop it in a panel grid.
        title=_title("NUBEAM", title, native.runid),
    )


def nubeam_profile(
    outputs: Any,
    quantity: str,
    *,
    rho: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
    **style: Any,
):
    """Render a radial NUBEAM profile."""
    return render_profile_1d(
        build_nubeam_profile(outputs, quantity, rho=rho, title=title),
        ax=ax,
        show=show,
        **style,
    )


def _birth_columns(native: "NUBEAMOutputs") -> dict[str, np.ndarray]:
    if native.birth is None or not native.birth.columns:
        raise NUBEAMPlotError(
            "this NUBEAM result carries no deposition markers; set "
            "nltrk_dep0 = 1 in the init namelist to have NUBEAM write them"
        )
    return {k: np.asarray(v, dtype=float) for k, v in native.birth.columns.items()}


def build_nubeam_deposition_poloidal(outputs: Any, *, title: Optional[str] = None) -> GeometryLayers:
    """Beam deposition markers in the poloidal plane.

    Converted from the birth file's centimetres to metres.
    """
    native = _native(outputs)
    columns = _birth_columns(native)
    layer = GeometryLayer(
        r=columns["r"] / _CM_PER_M,
        z=columns["z"] / _CM_PER_M,
        kind="points",
        label=f"deposition ({native.birth.count} markers)",
        style={"marker": "o", "markersize": 2, "color": "#e41a1c", "alpha": 0.6},
    )
    return GeometryLayers(
        layers=(layer,),
        title=_title("Beam deposition (poloidal)", title, native.runid),
    )


def build_nubeam_deposition_topview(outputs: Any, *, title: Optional[str] = None) -> GeometryLayers:
    """Beam deposition markers seen from above.

    The historical VEST workflow looks at deposition this way, and the geometry
    supports it: the birth file stores a toroidal angle per marker. Projected
    the way every other VAFT top view is, ``x = R cos(phi)``, ``y = R
    sin(phi)``, with ``GeometryLayer.r``/``.z`` carrying x/y.

    Converted from centimetres and *degrees*; the file states neither.
    """
    native = _native(outputs)
    columns = _birth_columns(native)
    radius = columns["r"] / _CM_PER_M
    angle = np.deg2rad(columns["zeta"])

    layers = [
        GeometryLayer(
            r=radius * np.cos(angle),
            z=radius * np.sin(angle),
            kind="points",
            label=f"deposition ({native.birth.count} markers)",
            style={"marker": "o", "markersize": 2, "color": "#e41a1c", "alpha": 0.6},
        )
    ]
    # A ring at the outermost deposition radius, so the projection is readable
    # without requiring the machine outline to be available.
    turn = np.linspace(0.0, 2.0 * np.pi, 181)
    edge = float(radius.max())
    layers.append(
        GeometryLayer(
            r=edge * np.cos(turn),
            z=edge * np.sin(turn),
            kind="polyline",
            label=f"R = {edge:.2f} m",
            style={"color": "0.6", "linewidth": 0.8},
        )
    )
    return GeometryLayers(
        layers=tuple(layers),
        x_label="x [m]",
        y_label="y [m]",
        title=_title("Beam deposition (top view)", title, native.runid),
    )


def nubeam_deposition_poloidal(
    outputs: Any, *, title: Optional[str] = None, ax=None, show: bool = False, **style
):
    """Render beam deposition markers in the poloidal plane."""
    return render_geometry_layers(
        build_nubeam_deposition_poloidal(outputs, title=title), ax=ax, show=show, **style
    )


def nubeam_deposition_topview(
    outputs: Any, *, title: Optional[str] = None, ax=None, show: bool = False, **style
):
    """Render beam deposition markers from above."""
    return render_geometry_layers(
        build_nubeam_deposition_topview(outputs, title=title), ax=ax, show=show, **style
    )


def build_nubeam_lost_fast_ions(outputs: Any, *, title: Optional[str] = None) -> GeometryLayers:
    """Where NUBEAM stopped following fast ions, in the poloidal plane.

    Split by loss channel rather than named for one. NUBEAM's step log calls
    the whole channel "bad orbit loss", but ``lstype`` distinguishes prompt
    loss from orbit loss and a run can be entirely prompt -- the VEST case is.
    Naming the plot after one channel would assert something the data denies.

    Already in metres; no conversion, unlike the deposition markers.
    """
    native = _native(outputs)
    if native.lost is None:
        raise NUBEAMPlotError(
            "this NUBEAM result carries no lost-particle record; it is read "
            "from <runid>_xplasma_out.cdf, which the run did not produce"
        )
    if native.lost.count == 0:
        raise NUBEAMPlotError("this NUBEAM run lost no fast ions")

    columns = {k: np.asarray(v, dtype=float) for k, v in native.lost.columns.items()}
    prompt = np.asarray(native.lost.prompt)
    layers = []
    for mask, label, color in (
        (prompt, "prompt loss", "#377eb8"),
        (~prompt, "orbit loss", "#ff7f00"),
    ):
        if not mask.any():
            continue
        layers.append(
            GeometryLayer(
                r=columns["rlost"][mask],
                z=columns["zlost"][mask],
                kind="points",
                label=f"{label} ({int(mask.sum())})",
                style={"marker": "x", "markersize": 5, "color": color},
            )
        )
    return GeometryLayers(
        layers=tuple(layers),
        title=_title("Lost fast ions", title, native.runid),
    )


def nubeam_lost_fast_ions(
    outputs: Any, *, title: Optional[str] = None, ax=None, show: bool = False, **style
):
    """Render lost fast ions in the poloidal plane, split by loss channel."""
    return render_geometry_layers(
        build_nubeam_lost_fast_ions(outputs, title=title), ax=ax, show=show, **style
    )


def build_nubeam_power_accounting(
    outputs: Any, *, title: Optional[str] = None
) -> Panels:
    """NUBEAM's own end-of-step power budget, as a text panel per species.

    A text panel rather than a chart on purpose: the architecture has no
    canonical bar model, and introducing one is a design decision rather than
    something a NUBEAM plot should settle. The numbers are NUBEAM's, including
    its own residual -- nothing is recomputed here.
    """
    native = _native(outputs)
    if not native.power_balance:
        raise NUBEAMPlotError(
            "this NUBEAM result carries no power balance; it is parsed from "
            "the step log, which the run did not produce"
        )

    panels = []
    for balance in native.power_balance:
        lines = []
        injected = balance.injected
        if injected:
            lines.append(f"injected {injected / 1e3:,.1f} kW")
            lines.append("")
        fractions = balance.fractions()
        for name, watts in sorted(balance.sinks().items(), key=lambda kv: -kv[1]):
            share = f"{100 * fractions[name]:5.1f}%" if fractions else "     "
            lines.append(f"{name:<24s} {watts / 1e3:8.2f} kW  {share}")
        if balance.residual is not None:
            lines.append("")
            closure = (
                f" ({100 * balance.residual / injected:+.2f}%)" if injected else ""
            )
            lines.append(f"{'residual':<24s} {balance.residual / 1e3:8.2f} kW{closure}")
        panels.append(TextPanel(lines=tuple(lines), title=balance.species))

    return Panels(
        models=tuple(panels),
        ncols=len(panels),
        share_x=False,
        suptitle=_title("NUBEAM power balance", title, native.runid),
    )


def nubeam_power_accounting(
    outputs: Any, *, title: Optional[str] = None, ax=None, show: bool = False, **style
):
    """Render NUBEAM's power balance."""
    return render_panels(
        build_nubeam_power_accounting(outputs, title=title), ax=ax, show=show, **style
    )
