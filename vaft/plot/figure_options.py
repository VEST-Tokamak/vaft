"""Reproducible figure options: what a user may change about a canonical figure (issue #1421).

The presentation contract (issue #689) decides how large a figure is
(``format``) and which visual grammar it uses (``theme``).  What it leaves
open are the adjustments a scientist makes to one canonical plot and would
make again for the next shot: an axis range, a log scale, a legend that
needs two columns, a colorbar range shared between figures, panel labels for
a paper, the resolution of the exported file.

:class:`FigureOptions` holds exactly those, and nothing else.  The rule for
what belongs here, from the issue: *a setting that should reasonably be
reused for another shot with the same canonical plot*.  A one-off nudge --
moving a legend a few millimetres, aligning one annotation -- belongs to a
finishing tool such as Pylustrator (issue #1202), not here.

Every field defaults to ``None``, which means *inherit*: the format, the
theme or the plot's own canonical choice decides.  Only what a caller sets
explicitly is applied, and only that is serialised, so a reproduced figure
follows any later improvement of the defaults it did not override.  One
object, three faces:

* ``plot_<name>(..., figure_options=FigureOptions(xlim=(0.2, 0.4)))`` or the
  same as a plain ``dict``;
* ``vaft plot <name> --shot N --figure-option xlim=(0.2, 0.4)`` -- see
  :meth:`FigureOptions.cli_args` and :meth:`FigureOptions.from_cli`;
* :meth:`FigureOptions.python` for the constructor call that rebuilds it.

Typography fields take part in the render (they are ``rcParams`` applied
inside the presentation context, after the format and theme, so they win);
everything else adjusts the finished axes; the export fields apply when the
figure is written (:meth:`FigureOptions.save`).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "AXIS_SCALES",
    "COLOR_NORMS",
    "FigureOptions",
    "LEGEND_LOCATIONS",
    "MATH_FONTSETS",
    "as_figure_options",
    "reproduce_cli",
    "reproduce_python",
]

#: Axis scales a user may choose; Matplotlib's names.
AXIS_SCALES = ("linear", "log", "symlog")

#: Colour normalisations of a scalar field.  ``centered`` keeps zero at the
#: middle of the colormap, as a signed perturbation field needs.
COLOR_NORMS = ("linear", "log", "symlog", "centered")

#: Named legend positions -- Matplotlib's; coordinates are out of scope.
LEGEND_LOCATIONS = (
    "best", "upper right", "upper left", "lower left", "lower right", "right",
    "center left", "center right", "lower center", "upper center", "center",
)

#: Matplotlib's ``mathtext.fontset`` values.
MATH_FONTSETS = ("dejavusans", "dejavuserif", "cm", "stix", "stixsans", "custom")

_TICK_DIRECTIONS = ("in", "out", "inout")

#: Fields read as booleans; a CLI value arrives as text ("false", "no", ...).
_BOOL_FIELDS = ("minor_ticks", "mirror_ticks", "grid", "legend", "legend_frame", "panel_labels", "transparent")
_TRUE_WORDS = ("true", "yes", "on", "1")
_FALSE_WORDS = ("false", "no", "off", "0")

#: Fields applied while drawing or saving, not to the finished axes.
_RENDER_FIELDS = frozenset({"font_family", "math_fontset", "font_size", "dpi", "transparent"})


@dataclass(frozen=True)
class FigureOptions:
    """Explicit overrides of one figure's presentation; ``None`` inherits.

    Axes fields apply to every data axes of the figure (each panel of a
    composite); colorbar fields to every scalar-field image or mesh and the
    colorbar attached to it.
    """

    # -- typography (applied while the figure is drawn) --------------------
    #: Text font, or fonts in order of preference; replaces the theme's.
    font_family: str | tuple[str, ...] | None = None
    #: ``mathtext.fontset`` -- ``dejavusans``, ``stixsans``, ``stix``, ``cm``...
    math_fontset: str | None = None
    #: The base size in points; label, tick, title and legend sizes keep the
    #: format's ratios to it.
    font_size: float | None = None

    # -- axes --------------------------------------------------------------
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    xscale: str | None = None
    yscale: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    #: The figure's title (a composite's suptitle); ``""`` hides it.
    title: str | None = None

    # -- ticks and grid ----------------------------------------------------
    tick_direction: str | None = None
    minor_ticks: bool | None = None
    #: Ticks on the top and right spines as well.
    mirror_ticks: bool | None = None
    grid: bool | None = None

    # -- legend ------------------------------------------------------------
    #: ``False`` removes the legend; ``True`` draws one where the policy did not.
    legend: bool | None = None
    legend_loc: str | None = None
    legend_ncols: int | None = None
    legend_frame: bool | None = None

    # -- scalar fields -----------------------------------------------------
    clim: tuple[float, float] | None = None
    cmap: str | None = None
    norm: str | None = None

    # -- annotation --------------------------------------------------------
    #: ``(a)``, ``(b)``, ... in the corner of each panel, row by row.
    panel_labels: bool | None = None

    # -- export ------------------------------------------------------------
    dpi: float | None = None
    transparent: bool | None = None

    def __post_init__(self) -> None:
        for name in ("xlim", "ylim", "clim"):
            value = getattr(self, name)
            if value is not None:
                pair = tuple(float(v) for v in value)
                if len(pair) != 2 or not all(np.isfinite(pair)) or pair[0] == pair[1]:
                    raise ValueError(f"{name} must be two different finite numbers; got {value!r}")
                object.__setattr__(self, name, pair)
        if isinstance(self.font_family, list):
            object.__setattr__(self, "font_family", tuple(self.font_family))
        _choice("xscale", self.xscale, AXIS_SCALES)
        _choice("yscale", self.yscale, AXIS_SCALES)
        _choice("norm", self.norm, COLOR_NORMS)
        _choice("legend_loc", self.legend_loc, LEGEND_LOCATIONS)
        _choice("tick_direction", self.tick_direction, _TICK_DIRECTIONS)
        _choice("math_fontset", self.math_fontset, MATH_FONTSETS)
        if self.cmap is not None:
            import matplotlib

            if self.cmap not in matplotlib.colormaps:
                raise ValueError(f"cmap {self.cmap!r} is not a registered Matplotlib colormap")
        for name in ("font_size", "dpi"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"{name} must be a positive number; got {value!r}")
        if self.legend_ncols is not None and int(self.legend_ncols) < 1:
            raise ValueError(f"legend_ncols must be at least 1; got {self.legend_ncols!r}")
        for name in _BOOL_FIELDS:
            object.__setattr__(self, name, _as_bool(name, getattr(self, name)))

    def draws_on_axes(self) -> bool:
        """Whether anything beyond typography and export is set."""
        return any(key not in _RENDER_FIELDS for key in self.explicit())

    # -- intent ------------------------------------------------------------

    def explicit(self) -> dict[str, Any]:
        """The fields a caller set, in declaration order -- what is serialised."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        }

    def __bool__(self) -> bool:
        return bool(self.explicit())

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "FigureOptions":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"unknown figure option(s) {', '.join(map(repr, unknown))}; "
                f"known: {', '.join(sorted(known))}"
            )
        return cls(**dict(values))

    # -- reproduction --------------------------------------------------------

    def cli_args(self) -> list[str]:
        """``vaft plot`` arguments that rebuild these options (``--figure-option K=V``)."""
        args: list[str] = []
        for key, value in self.explicit().items():
            args += ["--figure-option", f"{key}={value!r}"]
        return args

    @classmethod
    def from_cli(cls, items: Iterable[str]) -> "FigureOptions":
        """The inverse of :meth:`cli_args`: ``KEY=VALUE`` items, values as Python literals."""
        values: dict[str, Any] = {}
        for item in items:
            key, separator, raw = str(item).partition("=")
            if not separator or not key.strip():
                raise ValueError(f"a figure option is KEY=VALUE; got {item!r}")
            try:
                value = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                value = raw
            values[key.strip()] = value
        return cls.from_dict(values)

    def python(self) -> str:
        """The constructor call that rebuilds these options, explicit fields only."""
        body = ", ".join(f"{key}={value!r}" for key, value in self.explicit().items())
        return f"FigureOptions({body})"

    # -- application -------------------------------------------------------

    def rc(self, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The ``rcParams`` the typography fields set, given the ones in force.

        ``font_size`` rescales every size the format set by the same factor,
        so a format's label/tick/title ratios survive the override.
        """
        import matplotlib

        from .presentation import resolve_font_family

        current = dict(matplotlib.rcParams) if base is None else dict(base)
        rc: dict[str, Any] = {}
        if self.font_family is not None:
            rc["font.family"] = resolve_font_family(self.font_family)
        if self.math_fontset is not None:
            rc["mathtext.fontset"] = self.math_fontset
        if self.font_size is not None:
            old = float(current.get("font.size", 10.0))
            factor = float(self.font_size) / old
            rc["font.size"] = float(self.font_size)
            for key in ("axes.labelsize", "axes.titlesize", "xtick.labelsize",
                        "ytick.labelsize", "legend.fontsize", "figure.titlesize"):
                value = current.get(key)
                if isinstance(value, (int, float)):  # a named size follows font.size by itself
                    rc[key] = float(value) * factor
        return rc

    def context(self) -> contextlib.AbstractContextManager:
        """Apply :meth:`rc` for one render; a no-op without typography fields."""
        import matplotlib

        rc = self.rc()
        return matplotlib.rc_context(rc=rc) if rc else contextlib.nullcontext()

    def apply(self, figure: Any, axes: Any) -> None:
        """Adjust a finished figure: axes, ticks, legend, colour scale, labels.

        Only the axes the renderer drew (and their colorbars) are touched, so
        a caller's other subplots in the same figure are left alone.  A 3-D
        view takes none of the axes options; a warning says so.
        """
        data_axes = _data_axes(figure, axes)
        if self.draws_on_axes() and any(
            getattr(a, "name", "") == "3d" for a in _listed_axes(figure, axes)
        ):
            import warnings

            warnings.warn(
                "figure_options axes, legend and colour settings are not applied to a 3-D view",
                UserWarning, stacklevel=3,
            )
        for axis in data_axes:
            self._apply_axes(axis)
            self._apply_legend(axis)
        if self.title is not None:
            suptitle = getattr(figure, "_suptitle", None)
            if suptitle is not None and suptitle.get_text():
                suptitle.set_text(self.title)
                suptitle.set_visible(bool(self.title))
            else:
                for axis in data_axes:
                    axis.set_title(self.title)
        self._apply_colour_scale(data_axes)
        if self.panel_labels and len(data_axes) > 1:
            for index, axis in enumerate(data_axes):
                _panel_label(axis, index)

    def save(self, figure: Any, path: Any, **savefig_kwargs: Any) -> Any:
        """Write ``figure`` with the export fields; vector text stays text.

        PDF and PostScript embed TrueType (Type 42) fonts, so a journal's
        tools can read and edit the text.
        """
        import matplotlib

        from .style import save_figure

        if self.dpi is not None:
            savefig_kwargs.setdefault("dpi", float(self.dpi))
        if self.transparent is not None:
            savefig_kwargs.setdefault("transparent", bool(self.transparent))
        with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            return save_figure(figure, path, **savefig_kwargs)

    def _apply_axes(self, axis: Any) -> None:
        if self.xscale is not None:
            axis.set_xscale(self.xscale)
        if self.yscale is not None:
            axis.set_yscale(self.yscale)
        if self.xlim is not None:
            axis.set_xlim(*self.xlim)
        if self.ylim is not None:
            axis.set_ylim(*self.ylim)
        if self.xlabel is not None:
            axis.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            axis.set_ylabel(self.ylabel)
        if self.tick_direction is not None:
            axis.tick_params(which="both", direction=self.tick_direction)
        if self.minor_ticks is True:
            axis.minorticks_on()
        elif self.minor_ticks is False:
            axis.minorticks_off()
        if self.mirror_ticks is not None:
            axis.tick_params(which="both", top=self.mirror_ticks, right=self.mirror_ticks)
        if self.grid is not None:
            # grid(False, alpha=...) would switch the grid ON; the flag goes alone.
            axis.grid(bool(self.grid))

    def _apply_legend(self, axis: Any) -> None:
        if self.legend is False:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
            return
        wanted = (self.legend_loc, self.legend_ncols, self.legend_frame)
        legend = axis.get_legend()
        if legend is None and not (self.legend is True or any(v is not None for v in wanted)):
            return
        if legend is None:
            if self.legend is not True:
                return  # the policy drew none (a lone trace); a placement alone does not add one
            handles, labels = axis.get_legend_handles_labels()
            if not handles:
                return
        else:
            # Keep the entries the legend policy chose (it may have summarised).
            handles = list(legend.legend_handles)
            labels = [text.get_text() for text in legend.get_texts()]
            title = legend.get_title().get_text()
        from .style import _COUNT_NOTE_GID, _POLICY_LEGEND_GID

        # The rebuilt legend keeps what the policy gave the old one: its type
        # size, placement, columns, frame and identity -- only what is set changes.
        kwargs: dict[str, Any] = {"fontsize": "small"}
        gid = _POLICY_LEGEND_GID
        if legend is not None:
            kwargs["fontsize"] = legend.get_texts()[0].get_fontsize() if legend.get_texts() else "small"
            kwargs["loc"] = legend._loc  # noqa: SLF001 - keep the policy's placement
            kwargs["ncols"] = legend._ncols  # noqa: SLF001
            kwargs["frameon"] = legend.get_frame_on()
            gid = legend.get_gid() or gid
            if title:
                kwargs["title"] = title
        else:
            # A forced legend replaces the policy's "N traces" note.
            for text in list(axis.texts):
                if text.get_gid() == _COUNT_NOTE_GID:
                    text.remove()
        if self.legend_loc is not None:
            kwargs["loc"] = self.legend_loc
        if self.legend_ncols is not None:
            kwargs["ncols"] = int(self.legend_ncols)
        if self.legend_frame is not None:
            kwargs["frameon"] = bool(self.legend_frame)
        axis.legend(handles, labels, **kwargs).set_gid(gid)

    def _apply_colour_scale(self, data_axes: list[Any]) -> None:
        if self.clim is None and self.cmap is None and self.norm is None:
            return
        import matplotlib.colors as mcolors

        for mappable in _scalar_mappables(data_axes):
            if self.cmap is not None:
                mappable.set_cmap(self.cmap)
            vmin, vmax = self.clim if self.clim is not None else mappable.get_clim()
            if self.norm is not None:
                mappable.set_norm(_norm(self.norm, float(vmin), float(vmax), mcolors))
            elif self.clim is not None:
                mappable.set_clim(vmin, vmax)
            colorbar = getattr(mappable, "colorbar", None)
            if colorbar is not None:
                colorbar.update_normal(mappable)


def as_figure_options(value: Any) -> FigureOptions | None:
    """``None``, a :class:`FigureOptions` or a mapping, as options (``None`` if empty)."""
    if value is None:
        return None
    if isinstance(value, FigureOptions):
        return value or None
    if isinstance(value, Mapping):
        return FigureOptions.from_dict(value) or None
    raise TypeError(
        f"figure_options must be a FigureOptions or a mapping of its fields; got {type(value).__name__}"
    )


def _request_presets(format: str | None, theme: str | None) -> list[tuple[str, str]]:
    return [(key, value) for key, value in (("format", format), ("theme", theme)) if value]


def reproduce_cli(
    name: str,
    shot: Any,
    *,
    source: str | None = None,
    format: str | None = None,
    theme: str | None = None,
    figure_options: Any = None,
    out: str | None = None,
) -> list[str]:
    """The ``vaft plot`` argument list that draws this figure for a database shot.

    Only the intent is written: the presets named and the overrides set,
    never a resolved default, so the command follows later improvements of
    whatever it left to the defaults (issue #1421, section 8).
    """
    args = ["vaft", "plot", str(name)]
    for one in (shot if isinstance(shot, (list, tuple)) else [shot]):
        args += ["--shot", str(int(one))]
    if source:
        args += ["--source", str(source)]
    for key, value in _request_presets(format, theme):
        args += [f"--{key}", str(value)]
    options = as_figure_options(figure_options)
    if options is not None:
        args += options.cli_args()
    if out:
        args += ["--out", str(out)]
    return args


def reproduce_python(
    name: str,
    shot: Any,
    *,
    source: str | None = None,
    format: str | None = None,
    theme: str | None = None,
    figure_options: Any = None,
) -> str:
    """The Python call that draws this figure for a database shot, explicit settings only."""
    arguments = [repr(shot)]
    if source:
        arguments.append(f"source={source!r}")
    arguments += [f"{key}={value!r}" for key, value in _request_presets(format, theme)]
    options = as_figure_options(figure_options)
    lines = ["import vaft"]
    if options is not None:
        lines.append("from vaft.plot import FigureOptions")
        arguments.append(f"figure_options={options.python()}")
    lines.append(f"figure, axes = vaft.database.plot_{name}({', '.join(arguments)})")
    return "\n".join(lines)


def _as_bool(name: str, value: Any) -> bool | None:
    """``None``, a bool, or one of the yes/no words, as a bool; anything else is refused."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    word = str(value).strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be true or false; got {value!r}")


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")


def _listed_axes(figure: Any, axes: Any) -> list[Any]:
    """The renderer's axes, flattened and de-duplicated (a spanned cell may repeat)."""
    from .presentation import _axes_of

    listed = _axes_of(axes) or list(getattr(figure, "axes", []))
    unique: list[Any] = []
    for axis in listed:
        if all(axis is not seen for seen in unique):
            unique.append(axis)
    return unique


def _colorbar_axes(figure: Any) -> set[int]:
    """Identities of the figure's colorbar axes."""
    found = set()
    for axis in getattr(figure, "axes", []):
        for artist in (*axis.images, *axis.collections):
            colorbar = getattr(artist, "colorbar", None)
            if colorbar is not None:
                found.add(id(colorbar.ax))
    return found


def _data_axes(figure: Any, axes: Any) -> list[Any]:
    """The 2-D axes that draw data: the renderer's, without colorbars and hidden cells."""
    colorbars = _colorbar_axes(figure)
    return [
        axis for axis in _listed_axes(figure, axes)
        if axis.get_visible() and axis.axison and id(axis) not in colorbars
        and getattr(axis, "name", "") != "3d"
    ]


def _scalar_mappables(data_axes: list[Any]) -> list[Any]:
    """The colour-mapped artists of ``data_axes`` -- never a colorbar's own mesh."""
    from matplotlib.cm import ScalarMappable

    found = []
    for axis in data_axes:
        for artist in (*axis.images, *axis.collections):
            if isinstance(artist, ScalarMappable) and artist.get_array() is not None:
                found.append(artist)
    return found


def _norm(kind: str, vmin: float, vmax: float, mcolors: Any) -> Any:
    if kind == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if kind == "log":
        if vmax <= 0:
            raise ValueError("norm='log' needs a positive colour range; set clim=")
        return mcolors.LogNorm(vmin=vmin if vmin > 0 else vmax * 1e-3, vmax=vmax)
    if kind == "symlog":
        return mcolors.SymLogNorm(linthresh=max(abs(vmin), abs(vmax)) * 1e-2, vmin=vmin, vmax=vmax)
    half = max(abs(vmin), abs(vmax))
    return mcolors.CenteredNorm(vcenter=0.0, halfrange=half)


def _panel_label(axis: Any, index: int) -> None:
    letters = string.ascii_lowercase
    text = f"({letters[index]})" if index < len(letters) else f"({index + 1})"
    axis.text(
        0.0, 1.0, text, transform=axis.transAxes, ha="left", va="bottom",
        fontweight="bold", gid="vaft-panel-label",
    )
