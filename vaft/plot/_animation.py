"""``animation=True``: one canonical plot drawn over a sequence of its states (issues #1049/#1050).

The contract agreed on #1049:

* A plot's **sequence coordinate** is its slice-group control, the one
  :func:`vaft.plot.controls.controls_for` offers for the record and the
  slider and the GUI player already drive (``time_slice``, ``frame_index``,
  ``time_index``).  No second coordinate vocabulary exists.
* States are **selected before rendering**, against that control's options
  only: the driver keyword as a sequence, ``time_range=(t0, t1)``, or
  nothing for all of them.  N states give N frames -- no interpolation, no
  silent skipping.
* ``fps``/``duration`` are **presentation**, never physics; the physical
  coordinate of every frame is kept in :attr:`Animation.metadata`.
* The **scale is fixed over the sequence**: one ``vmin``/``vmax`` for image
  views, the union of the axis limits for every other view.
* The writer is chosen **by the output suffix** only: ``.mp4``/``.webm``
  through the private PyAV encoder (:mod:`vaft.plot._pyav`, extra
  ``vaft[video]``), ``.gif`` through Pillow, which Matplotlib already needs.

:class:`Animation` is what the call returns.  It is deliberately not part of
``vaft.plot``'s public names: it is documented by what it does, and nothing
is drawn until it is displayed, iterated or saved.
"""

from __future__ import annotations

import base64
import html
import io
import itertools
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = ["Animation", "PREVIEW_MAX_BYTES", "PREVIEW_MAX_FRAMES", "render_animation"]

#: Keywords ``animation=True`` consumes itself; none reaches a builder or renderer.
PRESENTATION_KEYS = ("fps", "duration", "timing", "dpi", "vmin", "vmax", "frame_label")
DEFAULT_FPS = 10.0
#: Frame resolution.  The renderers' own figures are 200 dpi, which makes a
#: camera frame 1300 px wide: a video of it is several times larger than the
#: screen it is watched on.
DEFAULT_DPI = 100
#: The notebook inlines a video only up to these; beyond them it shows a poster.
PREVIEW_MAX_FRAMES = 200
PREVIEW_MAX_BYTES = 8 * 2**20
SUFFIXES = (".mp4", ".webm", ".gif")
#: Colour maps whose centre colour means zero.
DIVERGING = frozenset(
    name + suffix
    for name in ("RdBu", "RdYlBu", "bwr", "coolwarm", "seismic", "PiYG", "PRGn", "BrBG", "PuOr", "RdGy")
    for suffix in ("", "_r")
)
#: What the builder snaps a scalar ``time=`` to; under ``animation=True`` it
#: would be a one-state sequence, which is a static plot.
_ONE_STATE = "one state is a static plot; drop animation=True"


@dataclass(frozen=True)
class Driver:
    """The sequence coordinate: the control's keyword and the physical value of each state."""

    name: str
    label: str
    coordinate: str
    unit: str | None
    indices: tuple[int, ...]
    values: tuple[float, ...]


def render_animation(
    spec: Any,
    entries: Sequence[tuple[str, Any]],
    options: Mapping[str, Any],
    *,
    backend: str,
    show: bool = False,
    ax: Any = None,
) -> "Animation":
    """Resolve the driver, the states and the presentation, and return the lazy result."""
    from vaft.plot.backend.options import split_options, validate_options
    from vaft.plot.backend.render import frame_renderers
    from vaft.plot.controls import controls_for

    from .backend.discovery import describe_one

    if ax is not None:
        raise TypeError("animation=True draws one figure per state and takes no ax=")
    if backend != "matplotlib":
        raise ValueError(
            f"animation=True draws its frames with Matplotlib; backend={backend!r} is not supported"
        )
    if len(entries) != 1:
        raise ValueError(
            f"animation=True draws one source over its own states; got {len(entries)} sources"
        )
    options = dict(options)
    presentation = {key: options.pop(key) for key in PRESENTATION_KEYS if key in options}
    fps, duration, timing, dpi = _presentation(presentation)

    record = describe_one(spec.name, entries)
    slices = [c for c in controls_for(record) if c.group == "slice"]
    if not slices:
        raise ValueError(
            f"plot_{spec.stem} has no sequence coordinate for this source: its capability record "
            "offers no slice control (time_slice, frame_index or time_index) to animate over"
        )
    if len(slices) > 1:
        raise ValueError(
            f"plot_{spec.stem} offers {len(slices)} slice controls "
            f"({', '.join(c.name for c in slices)}); animation=True needs exactly one"
        )
    control = slices[0]
    driver, selection = _select(spec.name, control, entries, options)
    options.pop(control.name, None)
    options.pop("time_range", None)
    validate_options(spec.name, options)
    fixed, style = split_options(options)
    if duration is not None:
        fps = len(driver.indices) / duration
    from vaft.plot.models import Image2D

    image = isinstance(spec.model, type) and issubclass(spec.model, Image2D)
    if not image and ("vmin" in presentation or "vmax" in presentation):
        raise ValueError(
            f"vmin=/vmax= fix the colour scale of an image view; plot_{spec.stem} draws "
            f"{getattr(spec.model, '__name__', spec.model)}, whose axis limits animation=True fixes itself"
        )
    build, draw = frame_renderers(spec, entries, fixed, style, backend)
    result = Animation(
        plot=spec.name, label=str(entries[0][0]), driver=driver, selection=selection,
        fps=float(fps), timing=timing, dpi=int(dpi), image=image,
        vmin=presentation.get("vmin"), vmax=presentation.get("vmax"),
        frame_label=bool(presentation.get("frame_label", True)), options=fixed, style=style, build=build, draw=draw,
    )
    if show:
        result.show()
    return result


def _presentation(given: Mapping[str, Any]) -> tuple[float, float | None, str, int]:
    fps, duration = given.get("fps"), given.get("duration")
    if fps is not None and duration is not None:
        raise ValueError(
            "give fps= or duration=, not both: the state count fixes the other, "
            "and animation=True never resamples the states to reconcile them"
        )
    for name, value in (("fps", fps), ("duration", duration)):
        if value is not None and not (_is_number(value) and math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive number of seconds' worth; got {value!r}")
    timing = given.get("timing", "uniform")
    if timing == "physical":
        raise NotImplementedError(
            "timing='physical' (frames spaced by their physical stamps) is reserved by #1049 "
            "and not implemented; the default timing='uniform' shows every state for 1/fps"
        )
    if timing != "uniform":
        raise ValueError(f"timing must be 'uniform'; got {timing!r}")
    dpi = given.get("dpi", DEFAULT_DPI)
    if not (_is_number(dpi) and dpi > 0):
        raise ValueError(f"dpi must be a positive number; got {dpi!r}")
    for name in ("vmin", "vmax"):
        if name in given and not _is_number(given[name]):
            raise ValueError(f"{name} must be a number; got {given[name]!r}")
    if "vmin" in given and "vmax" in given and not given["vmin"] < given["vmax"]:
        raise ValueError(f"vmin must be below vmax; got {given['vmin']!r}, {given['vmax']!r}")
    return float(fps if fps is not None else DEFAULT_FPS), duration, timing, int(dpi)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _states(control: Any) -> tuple[int, ...]:
    """Every state the control offers, in its own order."""
    if control.kind == "range":
        low, high, step = (int(v) for v in control.options)
        return tuple(range(low, high + 1, step))
    return tuple(int(v) for v in control.options)


def _select(
    name: str, control: Any, entries: Sequence[tuple[str, Any]], options: Mapping[str, Any],
):
    """The chosen states of ``control`` and their physical values.

    The values come from :func:`vaft.plot.backend.discovery.sequence_values`,
    which owns which time axis a plot's control pages along; the animation
    only picks among them.
    """
    from .backend.discovery import sequence_values

    coordinate, unit, axis = sequence_values(name, entries, control.name, **options)
    if control.name == "frame_index":
        # The record's control counts channel 0, detector 0; the frames that
        # exist are those of the camera the call draws.
        offered = tuple(range(len(axis)))
    else:
        offered = _states(control)
        beyond = [i for i in offered if i >= len(axis)]
        if beyond:
            raise ValueError(
                f"{name}: the {control.name} control offers {beyond[:3]}{'...' if len(beyond) > 3 else ''} "
                f"but the plot's sequence holds {len(axis)} values"
            )
    given = options.get(control.name)
    window = options.get("time_range")
    if options.get("time") is not None:
        raise ValueError(f"time= picks one instant and {_ONE_STATE}; time_range=(t0, t1) selects an interval")
    if given is not None and window is not None:
        raise ValueError(f"give {control.name}= or time_range=, not both")
    if given is not None:
        if _is_number(given):
            raise ValueError(f"{control.name}={given!r} is {_ONE_STATE}; pass a sequence of indices")
        chosen = tuple(int(v) for v in given)
        outside = [v for v in chosen if v not in offered]
        if outside:
            span = (
                f"{offered[0]}..{offered[-1]}" if control.kind == "range"
                else ", ".join(map(str, offered))
            )
            raise ValueError(f"{control.name} {outside} not offered; this plot offers {span}")
        selection = {"kind": control.name, "value": list(chosen)}
    elif window is not None:
        try:
            low, high = (float(v) for v in window)
        except (TypeError, ValueError):
            raise ValueError(f"time_range must be (start, stop) in seconds; got {window!r}") from None
        chosen = tuple(i for i in offered if low <= axis[i] <= high)
        selection = {"kind": "time_range", "value": [low, high]}
    else:
        chosen = offered
        selection = {"kind": "all", "value": None}
    if not chosen:
        raise ValueError(f"the selection holds no {control.name} state")
    repeated = sorted({i for i in chosen if chosen.count(i) > 1})
    if len(set(chosen)) >= 2 and repeated:
        raise ValueError(f"the selection repeats {control.name} {repeated}; each state is one frame")
    if len(set(chosen)) < 2:
        raise ValueError(f"the selection holds {control.name}={chosen[0]} only: {_ONE_STATE}")
    driver = Driver(
        name=control.name, label=control.label, coordinate=coordinate, unit=unit,
        indices=chosen, values=tuple(float(axis[i]) for i in chosen),
    )
    return driver, selection


class Animation:
    """One canonical plot over a sequence of its states: ``len``, ``metadata``, ``frames()``, ``save()``, ``show()``.

    Returned by ``plot_*(..., animation=True)``.  Nothing is built or drawn
    until the result is displayed, iterated or saved, and each of those
    draws the states again one at a time.
    """

    def __init__(
        self, *, plot: str, label: str, driver: Driver, selection: Mapping[str, Any], fps: float,
        timing: str, dpi: int, image: bool, vmin: Any, vmax: Any, frame_label: bool,
        options: Mapping[str, Any],
        style: Mapping[str, Any], build: Any, draw: Any,
    ) -> None:
        self.plot = plot
        self.label = label
        self.driver = driver
        self.fps = fps
        self.timing = timing
        self.dpi = dpi
        self.frame_label = frame_label
        self._selection = dict(selection)
        self._image = image
        self._given_scale = (vmin, vmax)
        self._options = dict(options)
        self._style = dict(style)
        self._build = build
        self._draw = draw
        self._scale: dict[str, Any] | None = None
        self._layout_cache: tuple[dict[int, Any], tuple[int, int]] | None = None
        self._encoder: dict[str, Any] | None = None
        self._player: Any = None

    def __len__(self) -> int:
        return len(self.driver.indices)

    @property
    def duration(self) -> float:
        """Presentation length in seconds: ``len(self) / fps``."""
        return len(self) / self.fps

    def __repr__(self) -> str:
        return f"<vaft animation of {self.plot}: {self._caption()}>"

    def _caption(self) -> str:
        values = self.driver.values
        span = (
            f"t = {min(values):.4f}-{max(values):.4f} s" if self.driver.unit == "s"
            else f"{self.driver.coordinate} {min(values)}-{max(values)}"
        )
        return f"{len(self)} frames of {self.driver.name} ({span}), {self.fps:g} fps"

    # -- scientific states -------------------------------------------------

    def _state(self, position: int) -> str:
        driver = self.driver
        return (
            f"{self.plot}: state {driver.name}={driver.indices[position]} "
            f"({driver.coordinate} {driver.values[position]:g}{' ' + driver.unit if driver.unit else ''})"
        )

    def _model(self, position: int) -> Any:
        try:
            return self._build({self.driver.name: self.driver.indices[position]})
        except Exception as error:
            raise RuntimeError(f"{self._state(position)} could not be built: {error}") from error

    def normalization(self) -> dict[str, Any]:
        """The colour scale every frame shares, computed over the states on first use.

        An image view gets one ``vmin``/``vmax``; a 2-D field whose builder
        leaves its contour levels to Matplotlib gets one set of levels over
        the sequence's range.  Either would otherwise be re-chosen per frame,
        and a colour would mean a different value in every frame.
        """
        if self._scale is not None:
            return self._scale
        if self._image:
            self._scale = self._image_scale()
        else:
            self._scale = self._field_levels() or {"policy": "the builder's own levels per state"}
        self._scale["axis_limits"] = "image extent" if self._image else "union over the states"
        return self._scale

    def _range(self, models: Sequence[Any]) -> tuple[float, float]:
        low, high = math.inf, -math.inf
        for model in models:
            values = np.asarray(model.values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                low, high = min(low, float(finite.min())), max(high, float(finite.max()))
        if not math.isfinite(low):
            raise ValueError(f"{self.plot}: no state holds a finite value to scale by")
        return low, high

    def _models(self) -> Iterator[Any]:
        return (self._model(position) for position in range(len(self)))

    def _image_scale(self) -> dict[str, Any]:
        vmin, vmax = self._given_scale
        source = "caller"
        if vmin is None or vmax is None:
            colormaps: set[str] = set()

            def models() -> Iterator[Any]:
                for model in self._models():
                    colormaps.add(str(model.cmap))
                    yield model

            low, high = self._range(models())
            source = "sequence range" if vmin is None and vmax is None else "caller and sequence range"
            if vmin is None and vmax is None and colormaps <= DIVERGING and low < 0 < high:
                # A diverging map puts zero at its centre colour; an asymmetric
                # range would move zero off it (a fluctuation image, #161).
                high = max(-low, high)
                low = -high
                source += ", symmetric about zero (diverging colour map)"
            vmin = low if vmin is None else vmin
            vmax = high if vmax is None else vmax
        return {"policy": "one colour scale over the sequence", "vmin": float(vmin),
                "vmax": float(vmax), "source": source}

    def _field_levels(self) -> dict[str, Any] | None:
        """One set of contour levels for the filled colour maps of a 2-D field sequence.

        A filled map's levels are a colour scale the builder chose from that
        state's own range; they are replaced by the same number of levels
        over the range of every filled state.  Line contours keep their own
        levels: there the levels are the message (flux surfaces at fixed
        psi_N, confined to the plasma), not a colour scale.  One sequence may
        hold both, when a slice without an axis falls back to a filled map.
        """
        from matplotlib.ticker import MaxNLocator

        from vaft.plot.models import Field2D

        models = list(self._models())
        filled = [m for m in models if isinstance(m, Field2D) and m.filled and m.value_scale == "linear"]
        if not filled:
            return None
        chosen = [_levels(model) for model in filled]
        if all(levels is not None for levels in chosen) and all(
            np.array_equal(levels, chosen[0]) for levels in chosen
        ):
            return {"policy": "the builder's contour levels, the same for every state"}
        # An int n asks Matplotlib for n+1 ticks, None for 8; explicit levels
        # keep their count.
        count = filled[0].contour_levels
        count = 7 if count is None else int(count) if isinstance(count, (int, np.integer)) else len(count) - 1
        low, high = self._range(filled)
        levels = MaxNLocator(max(count, 1) + 1, min_n_ticks=1).tick_values(low, high)
        policy = "one set of contour levels over the filled states"
        if len(filled) < len(models):
            policy += "; line-contour states keep their own levels"
        return {"policy": policy, "levels": [float(v) for v in levels], "vmin": low, "vmax": high,
                "source": "sequence range"}

    # -- frames --------------------------------------------------------------

    def _figure(self, position: int, limits: Mapping[int, Any] | None = None) -> Any:
        from vaft.plot.models import Field2D

        model = self._model(position)
        scale = self.normalization()
        if self._image and "vmin" in scale:
            model = replace(model, vmin=scale["vmin"], vmax=scale["vmax"])
        elif "levels" in scale and isinstance(model, Field2D) and model.filled:
            model = replace(model, contour_levels=tuple(scale["levels"]))
        import matplotlib.pyplot as plt

        before = set(plt.get_fignums())
        try:
            drawn = self._draw(model, ax=None, show=False)
            figure = drawn[0] if isinstance(drawn, tuple) else getattr(drawn, "figure", drawn)
            for number, axes in enumerate(_data_axes(figure)):
                if limits and number in limits:
                    axes.set_xlim(limits[number][0])
                    axes.set_ylim(limits[number][1])
            if self.frame_label:
                figure.text(0.01, 0.005, self._stamp(position), ha="left", va="bottom", fontsize=8)
        except Exception as error:
            for number in set(plt.get_fignums()) - before:
                plt.close(number)
            raise RuntimeError(f"{self._state(position)} could not be drawn: {error}") from error
        return figure

    def _stamp(self, position: int) -> str:
        """The state a frame shows, in the frame: the static title need not name it."""
        driver = self.driver
        value = driver.values[position]
        coordinate = f"t = {value:.5f} s" if driver.unit == "s" else f"{driver.coordinate} {value:g}"
        return f"{self.label}  {coordinate}  ({driver.name} {driver.indices[position]})"

    def _layout(self) -> tuple[dict[int, Any], tuple[int, int]]:
        """The union of every state's axis limits per data axes, and the largest frame.

        Drawn once over the states before any frame is kept: a view whose
        renderer fits its canvas to the content (the R-Z maps, #1026) gives
        each state its own figure size, and every frame of one video must
        share one.  Directions are preserved, so an inverted axis stays so.
        """
        import matplotlib.pyplot as plt

        if self._layout_cache is not None:
            return self._layout_cache
        union: dict[int, list[list[float]]] = {}
        height = width = 0
        for position in range(len(self)):
            figure = self._figure(position)
            try:
                panels = _data_axes(figure)
                if union and len(panels) != len(union):
                    raise RuntimeError(
                        f"{self._state(position)} draws {len(panels)} panels, the first state "
                        f"{len(union)}; one axis range per panel needs the same panels in every state"
                    )
                for number, axes in enumerate(panels):
                    spans = [axes.get_xlim(), axes.get_ylim()]
                    known = union.setdefault(number, [list(spans[0]), list(spans[1])])
                    for dim, (a, b) in enumerate(spans):
                        k0, k1 = known[dim]
                        low, high = min(a, b, k0, k1), max(a, b, k0, k1)
                        known[dim] = [high, low] if k0 > k1 else [low, high]
                # Rasterised as the frame will be: a GUI canvas reports its
                # size in logical pixels, which a HiDPI screen halves.
                h, w = _rgb(figure, self.dpi).shape[:2]
                width, height = max(width, w), max(height, h)
            finally:
                plt.close(figure)
        limits = {number: (tuple(x), tuple(y)) for number, (x, y) in union.items()}
        self._layout_cache = (limits, (height, width))
        return self._layout_cache

    def frames(self) -> Iterator[np.ndarray]:
        """``uint8`` RGB frames, one per state in order, drawn one at a time.

        Every frame has the same size.  A state drawn on a smaller canvas is
        centred on the largest one, padded with its own corner colour (the
        figure background); nothing is ever resized.
        """
        import matplotlib.pyplot as plt

        limits, size = (None, None) if self._image else self._layout()
        for position in range(len(self)):
            figure = self._figure(position, limits)
            try:
                frame = _rgb(figure, self.dpi)
            finally:
                plt.close(figure)
            if size is None:
                size = frame.shape[:2]
            yield _centred(frame, size, self.plot, position)

    # -- provenance ----------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """What the frames show (the driver and its physical values) and how they are played."""
        import vaft

        driver = self.driver
        data = {
            "plot": self.plot,
            "label": self.label,
            "vaft_version": getattr(vaft, "__version__", None),
            "driver": {
                "name": driver.name, "label": driver.label, "coordinate": driver.coordinate,
                "unit": driver.unit, "indices": list(driver.indices), "values": list(driver.values),
            },
            "selection": dict(self._selection),
            "n_states": len(self),
            "normalization": self.normalization(),
            "presentation": {"fps": self.fps, "duration": self.duration, "timing": self.timing,
                             "dpi": self.dpi, "frame_label": self.frame_label},
            "options": _jsonable(self._options),
            "style": _jsonable(self._style),
        }
        if self._encoder is not None:
            data["encoder"] = dict(self._encoder)
        return data

    # -- output ----------------------------------------------------------------

    def save(self, path: str | Path, *, sidecar: bool = True) -> Path:
        """Write the frames to ``path`` (``.mp4``, ``.webm`` or ``.gif``) and ``<path>.json``.

        The suffix picks the writer.  ``.mp4``/``.webm`` need the optional
        PyAV package (``pip install vaft[video]``); ``.gif`` needs nothing
        beyond Matplotlib.  ``sidecar=False`` skips the JSON provenance.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUFFIXES:
            raise ValueError(
                f"cannot write {path.name!r}: the suffix picks the writer, one of {', '.join(SUFFIXES)}"
            )
        if suffix != ".gif":
            from ._pyav import require_av

            require_av()  # before anything is drawn
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path = path.with_name(path.name + ".json")
        # Written beside the target and moved over it only when complete: a
        # selection that fails part-way never costs an existing movie.  The
        # old sidecar goes first, so it can never describe the new movie.
        handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=suffix, delete=False)
        handle.close()
        partial = Path(handle.name)
        try:
            self._encoder = self._write(partial, suffix, name=path.name)
            sidecar_path.unlink(missing_ok=True)
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        if sidecar:
            handle = tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix=f".{sidecar_path.name}.", delete=False,
            )
            try:
                with handle:
                    handle.write(json.dumps(self.metadata, indent=2) + "\n")
                os.replace(handle.name, sidecar_path)
            finally:
                Path(handle.name).unlink(missing_ok=True)
        return path

    def _write(self, path: Path, suffix: str, *, name: str) -> dict[str, Any]:
        if suffix == ".gif":
            return _write_gif(self.frames(), path, fps=self.fps)
        from ._pyav import encode_video

        return encode_video(self.frames(), path, fps=self.fps, name=name)

    def show(self) -> Any:
        """Play the frames in a Matplotlib window; no encoder is involved."""
        import matplotlib.pyplot as plt
        from matplotlib import animation

        frames = self.frames()
        first = next(frames)
        height, width = first.shape[:2]
        figure = plt.figure(figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.set_axis_off()
        image = axes.imshow(first)

        def update(frame: np.ndarray) -> tuple[Any, ...]:
            image.set_data(frame)
            return (image,)

        self._player = animation.FuncAnimation(
            figure, update, frames=itertools.chain([first], frames), init_func=lambda: (image,),
            interval=1000.0 / self.fps, cache_frame_data=False, repeat=False, blit=False,
        )
        plt.show()
        return self._player

    def _repr_html_(self) -> str:
        caption = html.escape(self._caption())
        video = self._inline_video()
        if video is not None:
            return (
                f'<figure><video controls loop autoplay muted playsinline '
                f'src="data:video/mp4;base64,{video}"></video>'
                f"<figcaption>{caption}</figcaption></figure>"
            )
        poster = base64.b64encode(_png(self._poster())).decode("ascii")
        return (
            f'<figure><img src="data:image/png;base64,{poster}"/>'
            f"<figcaption>First of {caption}. Inline playback needs PyAV and at most "
            f"{PREVIEW_MAX_FRAMES} frames / {PREVIEW_MAX_BYTES // 2**20} MB; "
            f"<code>.save('movie.mp4')</code> writes all of them.</figcaption></figure>"
        )

    def _inline_video(self) -> str | None:
        if len(self) > PREVIEW_MAX_FRAMES:
            return None
        try:
            from ._pyav import encode_video, require_av

            require_av()
        except ImportError:
            return None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preview.mp4"
            try:
                encode_video(self.frames(), path, fps=self.fps, codecs=("libx264", "h264"))
            except RuntimeError as error:
                if "has none of the" not in str(error):
                    raise  # a state that cannot be drawn is the caller's to see
                return None  # no H.264 encoder: a browser cannot play the fallbacks
            data = path.read_bytes()
        if len(data) > PREVIEW_MAX_BYTES:
            return None
        return base64.b64encode(data).decode("ascii")

    def _poster(self) -> np.ndarray:
        """The first state, drawn alone.

        The sequence scale and limits are used when they are already known;
        a poster is not worth building every state of a long sequence for.
        """
        import matplotlib.pyplot as plt

        limits, size = self._layout_cache if self._layout_cache is not None else (None, None)
        if self._scale is None and self._given_scale == (None, None):
            model_scale, self._scale = self._scale, {"policy": "the first state's own scale"}
            try:
                figure = self._figure(0, limits)
            finally:
                self._scale = model_scale
        else:
            figure = self._figure(0, limits)
        try:
            frame = _rgb(figure, self.dpi)
        finally:
            plt.close(figure)
        return frame if size is None else _centred(frame, size, self.plot, 0)


def _data_axes(figure: Any) -> list[Any]:
    """The figure's axes that hold data: a colorbar's axes is not one of them."""
    return [axes for axes in figure.axes if getattr(axes, "_colorbar", None) is None]


def _levels(model: Any) -> np.ndarray | None:
    levels = model.contour_levels
    if levels is None or isinstance(levels, (int, np.integer)):
        return None
    return np.asarray(levels, dtype=float)


def _centred(frame: np.ndarray, size: tuple[int, int], plot: str, position: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if (height, width) == tuple(size):
        return frame
    if height > size[0] or width > size[1]:
        raise RuntimeError(
            f"{plot}: frame {position} is {width}x{height} px, larger than the "
            f"{size[1]}x{size[0]} px of the first; the renderer changed the figure size between states"
        )
    top, left = (size[0] - height) // 2, (size[1] - width) // 2
    canvas = np.empty((size[0], size[1], 3), dtype=np.uint8)
    canvas[...] = frame[0, 0]
    canvas[top:top + height, left:left + width] = frame
    return canvas


def _rgb(figure: Any, dpi: int) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure.set_dpi(dpi)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _png(frame: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_gif(frames: Iterator[np.ndarray], path: Path, *, fps: float) -> dict[str, Any]:
    """Pillow's GIF writer.

    A GIF stores each delay in centiseconds, so ``1/fps`` is rounded to 10 ms
    (and browsers slow delays under 20 ms down).  Pillow merges identical
    consecutive frames into one frame shown for their summed delay: playback
    is unchanged, and the sidecar records how many frames the file holds.
    """
    from PIL import Image

    first = next(frames, None)
    if first is None:
        raise RuntimeError(f"no frame to write to {path.name}")
    count = 1

    def rest() -> Iterator[Any]:
        nonlocal count
        for frame in frames:
            count += 1
            yield Image.fromarray(frame)

    delay = 10 * max(1, int(round(100.0 / fps)))
    Image.fromarray(first).save(
        path, format="GIF", save_all=True, append_images=rest(), duration=delay, loop=0,
    )
    with Image.open(path) as written:
        stored = int(getattr(written, "n_frames", 1))
    return {"container": "gif", "codec": "gif", "frame_delay_ms": delay, "frames": count,
            "frames_stored": stored}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
