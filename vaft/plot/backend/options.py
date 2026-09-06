"""The option vocabulary of ``build_model``/``render_entries`` (issue #480).

Every keyword an adapter accepts is either an *extraction* option -- it
shapes the view model -- or a renderer *style* keyword.  The split used to be
an unvalidated frozenset, so a misspelt ``selecton=`` leaked into Matplotlib
and failed somewhere else.  The schema below names every extraction option
with its kind and, where one exists, the vocabulary it draws from; the style
set is read off the renderers' own signatures, so a new renderer keyword is
known here the day it is added.  The discovery-driven control layer
(:mod:`vaft.plot.controls`) reads the same schema to offer controls.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "EXTRACTION_OPTIONS",
    "OPTION_SCHEMA",
    "OptionSpec",
    "STYLE_OPTIONS",
    "choices_for",
    "split_options",
    "validate_options",
]

#: Option kinds: ``choice`` (one of a vocabulary), ``multi`` (several of a
#: vocabulary), ``int``, ``float``, ``bool``, ``str``, ``range`` (a pair),
#: ``any`` (checked by the recipe that consumes it).
OPTION_KINDS = ("choice", "multi", "int", "float", "bool", "str", "range", "any")


@dataclass(frozen=True)
class OptionSpec:
    """One extraction option: its kind and, for a vocabulary, where it lives.

    ``vocabulary`` names a constant of :mod:`vaft.plot.backend.recipes`,
    :mod:`vaft.plot.selection`, :mod:`vaft.plot.display` or
    :mod:`vaft.plot.style` (``"recipes.LAYOUTS"``), resolved lazily so this
    module can be imported before the recipes are.
    """

    name: str
    kind: str = "any"
    vocabulary: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in OPTION_KINDS:
            raise ValueError(f"kind must be one of {', '.join(OPTION_KINDS)}; got {self.kind!r}")


def _specs() -> tuple[OptionSpec, ...]:
    return (
        OptionSpec("selection", "any", "selection.PRESETS", "channel preset, index list or identifier list"),
        OptionSpec("channel", "any", description="one channel index or identifier"),
        OptionSpec("channels", "any", description="explicit channel indices or identifiers"),
        OptionSpec("layout", "choice", "recipes.LAYOUTS", "how channels are arranged"),
        OptionSpec("synthetic", "choice", "recipes.SYNTHETIC_MODES", "overlay the reconstruction's prediction"),
        OptionSpec("orientation", "choice", "recipes.ORIENTATIONS", "sign shown as stored or intuitive"),
        OptionSpec("style", "choice", "display.PSI_STYLES", "how a psi map is drawn"),
        OptionSpec("units", "str", description="display unit of a 2-D map"),
        OptionSpec("yunit", "str", description="display unit of the y axis"),
        OptionSpec("xunit", "str", description="display unit of the x axis"),
        OptionSpec("time_slice", "int", description="stored equilibrium slice index"),
        OptionSpec("time", "float", description="a time in seconds, snapped to a stored slice"),
        OptionSpec("time_range", "range", description="(start, stop) in seconds"),
        OptionSpec("centre", "range", description="(r0, z0) in metres the poloidal angle is measured about"),
        OptionSpec("angle", "choice", "recipes.ANGLE_SOURCES", "where a sensor's poloidal angle comes from"),
        OptionSpec("overlay", "multi", "recipes.CAMERA_OVERLAYS", "camera overlays"),
        OptionSpec("projection", "any", description="camera projection method"),
        # Spectroscopy: emission= names the species or line, line_index= the
        # position in the stored processed_line array beneath it.  The
        # vocabulary is the input's own labels, not a module constant, so it
        # cannot be a "choice" here.
        OptionSpec("emission", "any", description="spectral line, ion or element to draw"),
        OptionSpec("line_index", "int", description="position in the stored processed_line array"),
        OptionSpec("coordinate", "choice", "display.PROFILE_COORDINATES", "radial coordinate of a 1-D profile"),
        OptionSpec("x", "choice", "recipes.ABSCISSA_NAMES", "quantity on the abscissa of a line plot"),
        OptionSpec("contour_levels"), OptionSpec("detector"),
        OptionSpec("detrend"), OptionSpec("direction"), OptionSpec("dphi_deg"),
        OptionSpec("field_line_start"), OptionSpec("fit_ranges"), OptionSpec("flux_surface_levels"),
        OptionSpec("frame_index", "int"), OptionSpec("frame_indices"), OptionSpec("intrinsics_path"),
        OptionSpec("log_y", "bool"), OptionSpec("marker_frequencies"), OptionSpec("max_frequency", "float"),
        OptionSpec("max_harmonics", "int"), OptionSpec("max_length_m", "float"), OptionSpec("n_tor", "int"),
        OptionSpec("ncols", "int"), OptionSpec("noverlap", "int"), OptionSpec("nperseg", "int"),
        OptionSpec("per_family", "bool"),
        # wall eigenmode views (vaft #473)
        OptionSpec("basis"), OptionSpec("segment"), OptionSpec("mode"), OptionSpec("max_modes", "int"),
        OptionSpec("whole_wall", "bool"), OptionSpec("remap_em_coupling", "bool"), OptionSpec("rows"),
        OptionSpec("rules"), OptionSpec("orders"), OptionSpec("drive"), OptionSpec("metrics"),
        OptionSpec("which"), OptionSpec("rule"), OptionSpec("M"), OptionSpec("grid_shape"),
        OptionSpec("phi0", "float"), OptionSpec("pose_path"), OptionSpec("quantity"), OptionSpec("r0", "float"),
        OptionSpec("reference_slopes"), OptionSpec("sample_rate", "float"), OptionSpec("series_label", "str"),
        OptionSpec("shot", "int"), OptionSpec("show_lcfs", "bool"), OptionSpec("show_magnetic_axis", "bool"),
        OptionSpec("show_wall", "bool"), OptionSpec("sigma", "float"), OptionSpec("time_resolution", "float"),
        OptionSpec("title", "str"), OptionSpec("use_wall_boundary", "bool"), OptionSpec("window"),
        OptionSpec("window_size", "float"), OptionSpec("x_limits", "range"), OptionSpec("z0", "float"),
    )


#: Option name -> :class:`OptionSpec`.
OPTION_SCHEMA: Mapping[str, OptionSpec] = {spec.name: spec for spec in _specs()}

#: The extraction option names, the split every adapter applies.
EXTRACTION_OPTIONS: frozenset[str] = frozenset(OPTION_SCHEMA)

#: Options an adapter passes on internally (besides leading-underscore keys);
#: never offered, never refused.
INTERNAL_OPTIONS: frozenset[str] = frozenset()


def choices_for(spec: OptionSpec) -> tuple[Any, ...] | None:
    """The vocabulary of a ``choice``/``multi`` option, resolved lazily."""
    if spec.vocabulary is None:
        return None
    module_name, constant = spec.vocabulary.split(".")
    if module_name == "recipes":
        from . import recipes as module
    elif module_name == "selection":
        from vaft.plot import selection as module
    elif module_name == "display":
        from vaft.plot import display as module
    elif module_name == "style":
        from vaft.plot import style as module
    else:  # pragma: no cover - a typo in this module
        raise ValueError(f"unknown vocabulary module {module_name!r}")
    return tuple(getattr(module, constant))


def _style_options() -> frozenset[str]:
    """Renderer keywords, read off the base Matplotlib renderers' signatures.

    A registered ``spec.renderer`` takes ``**style`` and forwards it to the
    base renderer of its model kind, so the base renderers are the source.
    """
    from vaft.plot import renderers

    functions = [getattr(renderers, name) for name in dir(renderers) if name.startswith("render_")]
    try:
        from vaft.plot.renderers.text import render_text_panel

        functions.append(render_text_panel)
    except ImportError:  # pragma: no cover - the text panel is drawn by panels
        pass
    names: set[str] = set()
    for function in functions:
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            continue
        for parameter in parameters.values():
            if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
                continue
            if parameter.name in ("model", "ax", "show"):
                continue
            names.add(parameter.name)
    # The Plotly renderers take **style and forward what they understand;
    # the composite renderer threads per-member styles through too.
    names.update({"colorbar_ax"})
    return frozenset(names)


STYLE_OPTIONS: frozenset[str] = _style_options()


def validate_options(name: str, options: Mapping[str, Any]) -> None:
    """Refuse an option no builder and no renderer would read.

    Leading-underscore keys are internal plumbing (``_panel_member``).  A
    ``choice`` option whose value is a string outside its vocabulary is
    refused here too, naming the vocabulary; other kinds are checked by the
    recipe that consumes them.
    """
    for key, value in options.items():
        if key.startswith("_") or key in INTERNAL_OPTIONS or key in STYLE_OPTIONS:
            continue
        spec = OPTION_SCHEMA.get(key)
        if spec is None:
            raise ValueError(
                f"{name!r} does not take an option named {key!r}; extraction options: "
                f"{', '.join(sorted(EXTRACTION_OPTIONS))}; renderer style options: "
                f"{', '.join(sorted(STYLE_OPTIONS))}"
            )
        if spec.kind == "choice" and isinstance(value, str):
            choices = _plot_scoped_choices(name, key) or choices_for(spec)
            if choices is not None and value not in choices:
                raise ValueError(
                    f"{key} must be one of {', '.join(map(str, choices))}; got {value!r}"
                )


def _plot_scoped_choices(name: str, key: str) -> tuple[Any, ...] | None:
    """A vocabulary the plot itself narrows or widens.

    ``coordinate`` (issue #479) and ``x`` (issue #481) are declared per
    recipe, so the schema's static list is only the union: what a given plot
    accepts is asked of the plot.
    """
    if key not in ("coordinate", "x"):
        return None
    from . import recipes

    resolve = recipes.coordinate_options_for if key == "coordinate" else recipes.abscissa_options_for
    try:
        return resolve(name)
    except Exception:  # pragma: no cover - an unknown name is refused later by get_spec
        return None


def split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(extraction, style)``: the two halves every adapter routes apart."""
    extraction = {k: v for k, v in options.items() if k in EXTRACTION_OPTIONS or k in INTERNAL_OPTIONS or k.startswith("_")}
    style = {k: v for k, v in options.items() if k not in extraction}
    return extraction, style
