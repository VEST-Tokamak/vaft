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
        OptionSpec("smooth", "float", description="rolling-median window in seconds applied to line traces"),
        # A dense time base is indexed, not chosen from a list: the vacuum map
        # runs over the PF samples, thousands of them, where time_slice= names
        # one of a handful of stored equilibria.
        OptionSpec("time_index", "int", description="position on a dense time base"),
        # Startup views (issue #888).
        OptionSpec("p_Pa", "float", description="fill pressure in pascal for a Lloyd threshold"),
        OptionSpec("ec_frequency_Hz", "float",
                   description="EC source frequency in Hz whose resonance is marked; None draws none"),
        OptionSpec("rz", "range", description="(R, Z) observation point in metres"),
        OptionSpec("markers", "bool", description="mark each entry's breakdown onset"),
        OptionSpec("seeds", description="(R, Z) seeds of traced vacuum field lines, in metres"),
        OptionSpec("max_turns", "float", description="toroidal turns a traced vacuum field line is cut to"),
        OptionSpec("resolution", "int", description="points per axis of a computed 2-D grid"),
        OptionSpec("centre", "range", description="(r0, z0) in metres the poloidal angle is measured about"),
        OptionSpec("angle", "choice", "recipes.ANGLE_SOURCES", "where a sensor's poloidal angle comes from"),
        OptionSpec("overlay", "multi", "recipes.CAMERA_OVERLAYS", "what is drawn over a map"),
        OptionSpec("projection", "any", description="camera projection method"),
        OptionSpec("theta_deg_range", "range",
                   description="toroidal sweep of a camera overlay, in degrees"),
        # Spectroscopy: emission= names the species or line, line_index= the
        # position in the stored processed_line array beneath it.  The
        # vocabulary is the input's own labels, not a module constant, so it
        # cannot be a "choice" here.
        OptionSpec("emission", "any", description="spectral line, ion or element to draw"),
        OptionSpec("line_index", "int", description="position in the stored processed_line array"),
        OptionSpec("coordinate", "choice", "display.PROFILE_COORDINATES", "radial coordinate of a 1-D profile"),
        OptionSpec("x", "choice", "recipes.ABSCISSA_NAMES", "quantity on the abscissa of a line plot"),
        OptionSpec("method", "choice", "recipes.SPECTROGRAM_METHODS", "how a time-frequency map is computed"),
        OptionSpec("field", "choice", "recipes.EQUILIBRIUM_FIELD_NAMES", "quantity a 2-D equilibrium map draws"),
        # A composite's panels, by member name (issue #482).  The vocabulary
        # is the composite's own member list, so it is scoped per plot.
        OptionSpec("members", "multi", description="which panels of an overview to draw"),
        OptionSpec("frequency_range", "range", description="(f0, f1) in Hz: the analysed band"),
        # The wrapped-n fit (issue #485): which bands to fit, how many to find,
        # which n to test, and whether the fitted line is drawn beside the points.
        OptionSpec("frequencies", description="bands to fit, in Hz; None finds the strongest"),
        OptionSpec("num_modes", "int", description="how many bands to fit when none are named"),
        OptionSpec("candidate_n", description="toroidal mode numbers the fit may choose from"),
        OptionSpec("show_fit", "bool", description="draw the fitted line beside the measured points"),
        OptionSpec("preprocess", "bool", description="filter the raw probe voltages before analysis"),
        OptionSpec("n_frequencies", "int", description="wavelet scales across the band (method='cwt')"),
        OptionSpec("target_df", "float", description="frequency resolution a window is sized for"),
        OptionSpec("highpass_cutoff", "float", description="trend filter cut-off in Hz"),
        # camera fluctuation views (issue #161): the background and transform
        # windows in frames, the pixel box summed, and the filtered band.
        OptionSpec("background_frames", "int", description="local temporal mean width, in frames"),
        OptionSpec("window_frames", "int", description="short-time transform window, in frames"),
        OptionSpec("normalisation_frames", "int", description="local emission average width, in frames"),
        OptionSpec("region", "range", description="(row_start, row_stop, column_start, column_stop) pixel box"),
        OptionSpec("centre_frequency", "float", description="MHD band centre in Hz; the magnetics' dominant mode"),
        OptionSpec("half_width", "float", description="half the filtered bandwidth in Hz"),
        # Fluctuation diagnostics (issue #1005): the two channels a coherence
        # compares, and the spectral ridge a spectrogram overlays.
        OptionSpec("x_signal", description="reference channel of a coherence: index, 'diagnostic:index|name', IDS path or name"),
        OptionSpec("y_signal", description="channel compared against x_signal, in the same forms"),
        OptionSpec("track", description="overlay the tracked spectral ridge: True, or (f0, f1) search band in Hz"),
        OptionSpec("max_jump", "float", description="largest ridge frequency step between windows, in Hz"),
        OptionSpec("overlap", "float", description="fractional overlap between short-time windows"),
        # Read by a builder, so offered by the schema: before they were listed
        # validate_options refused them and no adapter could pass them on
        # (cold review plot G7).
        OptionSpec("min_wall_authority", "float",
                   description="wall-current authority below which a vacuum residual is not drawn"),
        OptionSpec("show_uncertainty", "bool", description="draw the stored uncertainty of a verification"),
        OptionSpec("source", "int",
                   description="position in core_sources.source of an NBI profile (vaft.omas / vaft.imas; "
                               "the vaft.database adapters take source= as the database source)"),
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
        # Neoclassical bootstrap comparison: the effective charge and the radial
        # band are physics choices the provider refuses to make for the caller.
        OptionSpec("z_eff", "float"), OptionSpec("impurity", "str"),
        OptionSpec("rho_range", "range"),
        OptionSpec("ion_index", "int"), OptionSpec("include_stored", "bool"),
        OptionSpec("models"), OptionSpec("order"),
        # Kinetic profile fits (issue #952): the equilibrium the channels are
        # mapped through (an ODS, a GEQDSK, a path, or {name: equilibrium} to
        # compare mappings) and the model fitted through them.
        OptionSpec("equilibrium", description="equilibrium (or {name: equilibrium}) the channels are mapped through"),
        OptionSpec("fitting_function", "str", description="profile model: polynomial, exponential, gp, linear, ..."),
        OptionSpec("which"), OptionSpec("rule"), OptionSpec("M"), OptionSpec("grid_shape"),
        OptionSpec("phi0", "float"), OptionSpec("pose_path"), OptionSpec("quantity"), OptionSpec("r0", "float"),
        OptionSpec("reference_slopes"), OptionSpec("sample_rate", "float"), OptionSpec("series_label", "str"),
        OptionSpec("shot", "int"), OptionSpec("show_lcfs", "bool"), OptionSpec("show_magnetic_axis", "bool"),
        OptionSpec("show_wall", "bool"), OptionSpec("sigma", "float"), OptionSpec("time_resolution", "float"),
        OptionSpec("title", "str"), OptionSpec("use_wall_boundary", "bool"), OptionSpec("window"),
        OptionSpec("window_size", "float"), OptionSpec("x_limits", "range"), OptionSpec("z0", "float"),
        # Island and coil-spectrum views (issue #886).
        OptionSpec("unit", "str", description="display unit of a coil current or perturbed field; 'auto' picks one"),
        OptionSpec("modes", description="toroidal mode numbers of a coil-current spectrum; None draws all"),
        OptionSpec("psi_n", "float", description="normalized poloidal flux of the surface a poloidal spectrum is cut at"),
        OptionSpec("pedestal", description="fitted pedestal whose top is marked on a psi_N abscissa"),
        OptionSpec("phi_deg", "float", description="toroidal angle in degrees of an island cross-section"),
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
        if key == "time" and value is not None:
            # Accepted-then-ignored is the defect (cold review plot G1): a plot
            # with no instant to choose says so before anything is built.
            from . import recipes

            if name in recipes.RECIPES and not recipes.time_axis_of(name):
                raise ValueError(recipes.no_time_option_message(name))
        if key == "members" and _plot_scoped_choices(name, key) is None:
            raise ValueError(
                f"{name!r} is not a panel composite and takes no members=; "
                "members= picks the panels of one such as diagnostics_overview"
            )
        if spec.kind == "choice" and isinstance(value, str):
            choices = _plot_scoped_choices(name, key) or choices_for(spec)
            if choices is not None and value not in choices:
                raise ValueError(
                    f"{key} must be one of {', '.join(map(str, choices))}; got {value!r}"
                )


def _plot_scoped_choices(name: str, key: str) -> tuple[Any, ...] | None:
    """A vocabulary the plot itself narrows or widens.

    ``coordinate`` (issue #479), ``x`` (issue #481), ``field`` and
    ``overlay`` (issue #483) are declared per recipe, so the schema's static
    list is only the union: what a given plot accepts is asked of the plot.
    """
    if key not in ("coordinate", "x", "field", "overlay", "members"):
        return None
    from . import recipes

    resolve = {
        "coordinate": recipes.coordinate_options_for,
        "x": recipes.abscissa_options_for,
        "field": recipes.field_options_for,
        "overlay": recipes.overlay_options_for,
        "members": recipes.member_options_for,
    }[key]
    try:
        return resolve(name)
    except Exception:  # pragma: no cover - an unknown name is refused later by get_spec
        return None


def split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(extraction, style)``: the two halves every adapter routes apart."""
    extraction = {k: v for k, v in options.items() if k in EXTRACTION_OPTIONS or k in INTERNAL_OPTIONS or k.startswith("_")}
    style = {k: v for k, v in options.items() if k not in extraction}
    return extraction, style
