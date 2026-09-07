"""Discovery-driven controls: what a reader may change about one plot (issue #480).

A :class:`~vaft.plot.discovery.PlotCapability` already states every option
a plot takes and what this input supports -- the channel presets, the
layouts, the display units, the validity and uncertainty modes, the sign
policy, the stored slices.  :func:`controls_for` turns those facts into
:class:`ControlSpec` entries, in a fixed order, with the defaults the plot
would apply on its own.  Widget toolkits are adapters over these specs
(:func:`vaft.plot.renderers.interactive.render_controls`); this module
imports none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .display import COORDINATE_LABELS, PSI_STYLES
from .selection import REGION_PRESETS, REPRESENTATIVE_PRESETS, SIGNAL_PRESETS
from .style import UNCERTAINTY_MODES, VALIDITY_MODES

__all__ = ["CONTROL_GROUPS", "CONTROL_KINDS", "ControlSpec", "controls_for"]

#: ``toggle`` (on/off), ``choice`` (one of ``options``), ``multi`` (several
#: of ``options``), ``range`` (an integer position in ``(min, max, step)``),
#: ``text`` (free text).
CONTROL_KINDS = ("toggle", "choice", "multi", "range", "text")

#: The order controls are offered in, and the group each belongs to.
CONTROL_GROUPS = ("slice", "selection", "layout", "display", "model", "style", "backend")

def _plain(label: str) -> str:
    """A mathtext axis label as plain widget text: ``$\\rho_N$`` -> ``rho_N``."""
    import re

    text = re.sub(r"\$\\?([A-Za-z]+)(_[A-Za-z0-9]+)?\$", lambda m: m.group(1) + (m.group(2) or ""), label)
    text = text.replace("\\sqrt{", "sqrt(").replace("}", ")").replace("$", "")
    return text.replace("\\", "")


#: The value a ``choice`` control uses to say "no overlay" for an option
#: that is otherwise absent.
NONE = "none"


@dataclass(frozen=True)
class ControlSpec:
    """One thing a reader may change: the option it drives and its choices.

    ``name`` is the keyword handed to ``build_model``/``render_entries``;
    ``options`` the values a ``choice``/``multi`` may take or the
    ``(min, max, step)`` of a ``range``; ``labels`` the text shown for each
    option when it differs from the value (channel positions, slice times).
    """

    name: str
    kind: str
    label: str
    default: Any = None
    options: tuple[Any, ...] = ()
    labels: tuple[str, ...] = ()
    group: str = "model"

    def __post_init__(self) -> None:
        if self.kind not in CONTROL_KINDS:
            raise ValueError(f"kind must be one of {', '.join(CONTROL_KINDS)}; got {self.kind!r}")
        if self.group not in CONTROL_GROUPS:
            raise ValueError(f"group must be one of {', '.join(CONTROL_GROUPS)}; got {self.group!r}")
        if self.kind == "choice":
            if not self.options:
                raise ValueError(f"control {self.name!r} offers no choices")
            if self.default is not None and self.default not in self.options:
                raise ValueError(f"control {self.name!r}: default {self.default!r} is not one of its choices")
        elif self.kind == "multi":
            if not self.options:
                raise ValueError(f"control {self.name!r} offers no choices")
            chosen = tuple(self.default or ())
            if any(value not in self.options for value in chosen):
                raise ValueError(f"control {self.name!r}: default {self.default!r} is not within its choices")
        elif self.kind == "range":
            if len(self.options) != 3:
                raise ValueError(f"control {self.name!r}: a range needs (min, max, step)")
            low, high, _ = self.options
            if self.default is not None and not low <= self.default <= high:
                raise ValueError(f"control {self.name!r}: default {self.default!r} is outside [{low}, {high}]")
        if self.labels and self.kind in ("choice", "multi") and len(self.labels) != len(self.options):
            raise ValueError(f"control {self.name!r}: one label per option, got {len(self.labels)} for {len(self.options)}")

    def validate(self, value: Any) -> Any:
        """``value`` coerced to what the option takes, or ``ValueError``."""
        if self.kind == "toggle":
            return bool(value)
        if self.kind == "choice":
            if value not in self.options:
                raise ValueError(f"{self.name} must be one of {', '.join(map(str, self.options))}; got {value!r}")
            return value
        if self.kind == "multi":
            chosen = tuple(value) if not isinstance(value, str) else (value,)
            bad = [v for v in chosen if v not in self.options]
            if bad:
                raise ValueError(f"{self.name}: {bad!r} not among {', '.join(map(str, self.options))}")
            return chosen
        if self.kind == "range":
            low, high, step = self.options
            position = int(value)
            if not low <= position <= high:
                raise ValueError(f"{self.name} must lie in [{low}, {high}]; got {value!r}")
            return position
        return str(value)


def controls_for(
    record: Any, *, include_style: bool = True, include_backend: bool = False
) -> tuple[ControlSpec, ...]:
    """The controls a plot's capability record supports, in offer order.

    Only facts the record states produce a control: a plot with one layout
    offers no layout control, a record without validity facts no validity
    control.  ``include_style`` adds the renderer-side modes (validity,
    uncertainty); ``include_backend`` adds the rendering library, off by
    default because changing it replaces the figure object.
    """
    controls: list[ControlSpec] = []
    controls.extend(_slice_controls(record))
    controls.extend(_selection_controls(record))
    controls.extend(_layout_controls(record))
    controls.extend(_display_controls(record))
    controls.extend(_model_controls(record))
    if include_style:
        controls.extend(_style_controls(record))
    if include_backend and len(record.backends) > 1:
        controls.append(ControlSpec(
            "backend", "choice", "Rendering backend", record.backends[0], tuple(record.backends), group="backend",
        ))
    return tuple(controls)


def _slice_controls(record: Any) -> list[ControlSpec]:
    slices: Mapping[str, Any] = getattr(record, "slices", None) or {}
    usable = tuple(int(i) for i in slices.get("usable", ()))
    if len(usable) < 2:
        return []
    times = slices.get("times", ())
    labels = tuple(
        f"{i}: {float(times[i]) * 1e3:.1f} ms" if i < len(times) else str(i) for i in usable
    )
    selected = slices.get("selected", usable[len(usable) // 2])
    return [ControlSpec(
        "time_slice", "choice", "Equilibrium slice",
        int(selected) if selected in usable else usable[len(usable) // 2],
        usable, labels, group="slice",
    )]


def _selection_controls(record: Any) -> list[ControlSpec]:
    channels: Mapping[str, Any] = record.channels or {}
    if not channels.get("total"):
        return []
    presets: list[str] = list(SIGNAL_PRESETS)
    if channels.get("regions"):
        presets = list(REGION_PRESETS) + presets
    if channels.get("representatives"):
        presets = list(REPRESENTATIVE_PRESETS) + presets
    controls = [ControlSpec("selection", "choice", "Channels", SIGNAL_PRESETS[0], tuple(presets), group="selection")]
    identifiers = tuple(channels.get("identifiers") or ())
    positions = tuple(channels.get("positions") or ())
    if identifiers:
        controls.append(ControlSpec(
            "channels", "multi", "Individual channels", (),
            tuple(range(len(identifiers))),
            tuple(f"{p}  {i}" for i, p in zip(identifiers, positions)) if len(positions) == len(identifiers) else tuple(map(str, identifiers)),
            group="selection",
        ))
    return controls


def _layout_controls(record: Any) -> list[ControlSpec]:
    layouts = tuple(record.layouts or ())
    if len(layouts) < 2:
        return []
    return [ControlSpec("layout", "choice", "Layout", layouts[0], layouts, group="layout")]


def _display_controls(record: Any) -> list[ControlSpec]:
    display: Mapping[str, Any] = record.display or {}
    units = tuple(display.get("units") or ())
    if len(units) < 2 or display.get("unit") is None:
        return []
    is_map = "convention" in display
    name = "units" if is_map else "yunit"
    return [ControlSpec(name, "choice", "Unit", display["unit"], units, group="display")]


def _model_controls(record: Any) -> list[ControlSpec]:
    controls: list[ControlSpec] = []
    abscissa: Mapping[str, Any] = getattr(record, "abscissa", None) or {}
    options = tuple(abscissa.get("options") or ())
    if len(options) > 1:
        default = abscissa.get("default") if abscissa.get("default") in options else options[0]
        controls.append(ControlSpec("x", "choice", "Abscissa", default, options))
    analysis: Mapping[str, Any] = getattr(record, "analysis", None) or {}
    methods = tuple(analysis.get("methods") or ())
    if len(methods) > 1:
        default = analysis.get("default") if analysis.get("default") in methods else methods[0]
        controls.append(ControlSpec("method", "choice", "Analysis method", default, methods))
    coordinates: Mapping[str, Any] = getattr(record, "coordinates", None) or {}
    options = tuple(coordinates.get("options") or ())
    if len(options) > 1:
        default = coordinates.get("default") if coordinates.get("default") in options else options[0]
        controls.append(ControlSpec(
            "coordinate", "choice", "Radial coordinate", default, options,
            tuple(_plain(COORDINATE_LABELS.get(name, name)) for name in options),
        ))
    display: Mapping[str, Any] = record.display or {}
    if "convention" in display and record.model in ("Field2D", "Panels") and record.name != "equilibrium_field_psi_vacuum":
        controls.append(ControlSpec("style", "choice", "Flux map style", PSI_STYLES[0], tuple(PSI_STYLES)))
    synthetic: Mapping[str, Any] = record.synthetic or {}
    if synthetic.get("overlay") and synthetic.get("available", True):
        controls.append(ControlSpec("synthetic", "choice", "Reconstruction overlay", NONE, (NONE, "equilibrium", "both")))
    orientation: Mapping[str, Any] = record.orientation or {}
    options = tuple(orientation.get("options") or ())
    if len(options) > 1:
        controls.append(ControlSpec("orientation", "choice", "Sign", orientation.get("default", options[0]), options))
    return controls


def _style_controls(record: Any) -> list[ControlSpec]:
    controls: list[ControlSpec] = []
    validity: Mapping[str, Any] = record.validity or {}
    if validity.get("available"):
        modes = tuple(validity.get("modes") or VALIDITY_MODES)
        controls.append(ControlSpec("validity", "choice", "Flagged samples", modes[0], modes, group="style"))
    uncertainty: Mapping[str, Any] = record.uncertainty or {}
    if uncertainty.get("available"):
        modes = tuple(uncertainty.get("modes") or UNCERTAINTY_MODES)
        controls.append(ControlSpec("uncertainty", "choice", "Uncertainty", modes[0], modes, group="style"))
    return controls
