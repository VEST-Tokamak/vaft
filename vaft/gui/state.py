"""What the browser shows, kept apart from the widgets that show it (#1086).

A :class:`BrowserSession` holds one loaded source, its plot catalog and the
plot currently drawn.  Everything scientific is delegated to public VAFT
calls -- :func:`vaft.omas.sample_ods`, :func:`vaft.omas.load`,
:func:`vaft.database.load`, :func:`vaft.omas.available_plots` and
:func:`vaft.omas.render_plot` with ``interactive=True`` -- so the same session
can be driven from a test or a notebook without Panel, and another frontend
could replace Panel without touching it.

The drawn plot is an :class:`~vaft.plot.renderers.interactive.Interactive`
built with ``interaction_backend="none"``: its ``state`` is the toolkit-free
:class:`~vaft.plot.navigation.ControlState` whose controls come from the
plot's capability record, and setting a control redraws its ``figure`` --
in place for Matplotlib, as a new Plotly figure for ``backend="plotly"``.
Plotly is preferred where a plot declares it, because zooming, panning and
marking then happen in the browser.  The GUI only puts widgets on that state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_KINDS = ("sample", "file", "shot")
RENDERERS = ("auto", "plotly", "matplotlib")


@dataclass(frozen=True)
class Source:
    """Where the data comes from.

    ``kind`` is ``"sample"`` (a packaged shot, offline), ``"file"`` (a local
    ODS/IMAS/GEQDSK file :func:`vaft.omas.load` reads) or ``"shot"`` (a
    database shot, loaded eagerly from ``namespace``).
    """

    kind: str
    value: int | str
    namespace: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(SOURCE_KINDS)}; got {self.kind!r}")
        if self.kind in ("sample", "shot"):
            object.__setattr__(self, "value", int(self.value))
        else:
            object.__setattr__(self, "value", str(self.value))

    @property
    def label(self) -> str:
        if self.kind == "sample":
            return f"sample {self.value}"
        if self.kind == "file":
            # The parent too: samples and products share file names
            # (39915/omas.json.gz, 41524/omas.json.gz).
            path = Path(str(self.value))
            return f"{path.parent.name}/{path.name}" if path.parent.name else path.name
        return f"shot {self.value}" + (f" ({self.namespace})" if self.namespace else "")


def load_source(source: Source) -> Any:
    """The ODS behind ``source``."""
    if source.kind == "sample":
        from vaft.omas import sample_ods

        return sample_ods(source.value)
    if source.kind == "file":
        from vaft.omas import load

        return load(source.value)
    from vaft.database import load

    # Eager: a lazy store closes when the call that opened it returns, and
    # every control change rebuilds from the data (vaft/database/plotting.py).
    return load(source.value, source=source.namespace)


def sample_shots() -> tuple[int, ...]:
    """The packaged samples this install can open."""
    from vaft.data import available_samples

    return tuple(available_samples())


#: The longest movie the browser export draws (#1400).
MAX_VIDEO_FRAMES = 600


class BrowserSession:
    """The loaded sources, their catalog and the plot on screen.

    Several sources are compared on one figure: the plots receive the list
    of ODS, labelled by shot, as ``render_plot`` takes it.

    Database shots are not downloaded whole.  Opening them lists the IDS each
    shot stores (:func:`vaft.database.available_plots`, no data read), and a
    plot loads only the IDS it declares
    (:func:`vaft.plot.backend.recipes.required_ids`) the first time it is
    drawn; the IDS stay in memory for every later plot and control change.
    :meth:`load_ids` fetches IDS ahead of any plot.
    """

    def __init__(self) -> None:
        self.sources: tuple[Source, ...] = ()
        self.ods: Any = None
        self.plot: str | None = None
        self.renderer: str | None = None
        self.interactive: Any = None
        #: A composed figure (#1467) instead of one plot: its composition and
        #: what ``compose`` returned.
        self.composition: Any = None
        self._composed: Any = None
        #: The reader's explicit figure options (#1421); empty = all inherited.
        from vaft.plot import FigureOptions

        self.figure_options: Any = FigureOptions()
        self._catalog: Any = None
        #: Every plot the registry supports for these sources, the unavailable
        #: ones with the discovery record's reason (#1172).
        self._everything: Any = None
        #: Database mode: one growing ODS per shot, the IDS it holds, and the
        #: IDS the shot stores at all.
        self._shots: dict[Source, Any] = {}
        self._held: dict[Source, set[str]] = {}
        self._stored: dict[Source, set[str]] = {}

    @property
    def database(self) -> bool:
        """Whether the sources are database shots, loaded IDS by IDS."""
        return bool(self.sources) and all(source.kind == "shot" for source in self.sources)

    @property
    def label(self) -> str:
        """What is loaded, for a status line or a file name."""
        if not self.sources:
            return ""
        if len(self.sources) == 1:
            return self.sources[0].label
        kinds = {source.kind for source in self.sources}
        if kinds == {"file"}:
            return "files " + ", ".join(source.label for source in self.sources)
        values = ", ".join(str(source.value) for source in self.sources)
        return f"{'samples' if kinds == {'sample'} else 'shots'} {values}"

    def open(self, sources: Source | Sequence[Source]) -> None:
        """Load ``sources`` and their catalog; nothing changes unless both succeed.

        The previous plot stays valid until the new sources are fully usable,
        so a failed load leaves the session exactly as it was.
        """
        chosen = (sources,) if isinstance(sources, Source) else tuple(sources)
        if not chosen:
            raise ValueError("choose at least one source")
        if len(set(chosen)) != len(chosen):
            raise ValueError("each source may be chosen once")
        kinds = {source.kind for source in chosen}
        if "shot" in kinds and kinds != {"shot"}:
            raise ValueError("database shots are compared with database shots only")
        if kinds == {"shot"}:
            import omas

            everything, stored = self._discover_shots(chosen)
            self.close()
            self._stored = stored
            self._shots = {source: omas.ODS() for source in chosen}
            self._held = {source: set() for source in chosen}
            data = list(self._shots.values())
            self.sources, self.ods = chosen, data[0] if len(data) == 1 else data
            self._everything, self._catalog = everything, _available(everything)
            return
        loaded = [load_source(source) for source in chosen]
        data = loaded[0] if len(loaded) == 1 else loaded
        everything = self._discover(data)
        self.close()
        self.sources, self.ods = chosen, data
        self._everything, self._catalog = everything, _available(everything)

    def _discover(self, data: Any) -> Any:
        """Every supported plot for ``data``, unavailable ones with their reason."""
        from vaft.omas import available_plots

        return available_plots(data, available_only=False)

    def _discover_shots(self, sources: Sequence[Source]) -> tuple[list[Any], dict[Source, set[str]]]:
        """Every supported plot for the shots, and the IDS each one stores.

        A plot is available when every shot can draw it; otherwise it keeps
        the first shot's reason that it cannot, naming the shot.
        """
        from dataclasses import replace

        from vaft.database import available_plots, stored_ids

        stored = {source: set(stored_ids(source.value, source.namespace)) for source in sources}
        catalogs = [available_plots(source.value, source.namespace, available_only=False) for source in sources]
        by_shot = [{capability.name: capability for capability in catalog} for catalog in catalogs]
        combined = []
        for capability in catalogs[0]:
            records = [records.get(capability.name) for records in by_shot]
            blocked = next(
                (
                    (source, record) for source, record in zip(sources, records)
                    if record is None or getattr(record, "available", True) is False
                ),
                None,
            )
            if blocked is None:
                combined.append(capability)
                continue
            source, record = blocked
            reason = getattr(record, "reason", "") if record is not None else "not supported"
            if len(sources) > 1:
                reason = f"{source.label}: {reason or 'unavailable'}"
            try:
                combined.append(replace(capability, available=False, reason=reason))
            except TypeError:  # not a dataclass (a stub): the name still says it
                continue
        if hasattr(catalogs[0], "with_records"):
            return catalogs[0].with_records(combined), stored
        return combined, stored

    def plot_ids(self, name: str) -> list[str]:
        """The IDS plot ``name`` reads, with the one that labels a shot."""
        from vaft.plot.backend.recipes import required_ids

        return list(dict.fromkeys(["dataset_description", *required_ids(name)]))

    def plotted_ids(self) -> list[str]:
        """Every IDS some available plot reads and every open shot stores."""
        wanted = {name for capability in self.catalog() for name in self.plot_ids(capability.name)}
        if self._stored:
            wanted &= set.intersection(*self._stored.values())
        return sorted(wanted)

    def held_ids(self) -> list[str]:
        """The IDS in memory for every open database shot."""
        if not self._held:
            return []
        return sorted(set.intersection(*self._held.values()))

    def load_ids(self, names: Sequence[str]) -> list[str]:
        """Fetch ``names`` for every open database shot; the IDS newly loaded.

        All shots are read before any is changed, so a failure leaves the
        session as it was.
        """
        if not self.database:
            raise RuntimeError("IDS are loaded one by one for database shots only")
        from vaft.database import load

        fetched: dict[Source, Any] = {}
        missing: dict[Source, list[str]] = {}
        for source in self.sources:
            # A plot declares optional IDS too (an overlay's wall): only what
            # the shot stores is asked for, or the read fails on the absent one.
            wanted = [
                name for name in dict.fromkeys(names)
                if name not in self._held[source] and name in self._stored[source]
            ]
            if wanted:
                fetched[source] = load(source.value, source=source.namespace, paths=wanted)
                missing[source] = wanted
        for source, part in fetched.items():
            for name in missing[source]:
                # Held only once it is in memory: an IDS the read did not
                # return is asked for again rather than reported as loaded.
                if name in part.keys():
                    self._shots[source][name] = part[name]
                    self._held[source].add(name)
        return sorted({name for names_ in missing.values() for name in names_})

    def catalog(self) -> Any:
        """The plots available for the loaded sources."""
        if self.ods is None:
            raise RuntimeError("no source is open")
        return self._catalog

    def full_catalog(self) -> Any:
        """Every plot the registry supports here, unavailable ones with their reason."""
        if self.ods is None:
            raise RuntimeError("no source is open")
        return self._everything if self._everything is not None else self._catalog

    def capability(self, name: str) -> Any:
        """The discovery record of plot ``name`` (``None`` when it is not supported here)."""
        if self.ods is None:
            return None
        return next((record for record in self.full_catalog() if record.name == name), None)

    def grouped_plots(self) -> dict[str, list[str]]:
        """Available plot names by subject, in catalog order."""
        groups: dict[str, list[str]] = {}
        for capability in self.catalog():
            groups.setdefault(capability.subject, []).append(capability.name)
        return groups

    def supports_plotly(self, name: str) -> bool:
        """Whether ``name`` declares a Plotly renderer in its capability record."""
        return any(
            capability.name == name and "plotly" in (getattr(capability, "backends", None) or ())
            for capability in self.catalog()
        )

    def select(self, name: str, *, renderer: str = "auto", **options: Any) -> Any:
        """Draw ``name`` with its controls; the previous figure is released.

        ``renderer`` is ``"plotly"`` (zoom, pan and marking in the browser),
        ``"matplotlib"`` (a static image) or ``"auto"``: Plotly where the plot
        declares it.  A plot that cannot be drawn raises and leaves the current
        one on screen.
        """
        if self.ods is None:
            raise RuntimeError("no source is open")
        if renderer not in RENDERERS:
            raise ValueError(f"renderer must be one of {', '.join(RENDERERS)}; got {renderer!r}")
        if renderer == "auto":
            renderer = "plotly" if self.supports_plotly(name) else "matplotlib"
        if self.database:
            self.load_ids(self.plot_ids(name))
        from vaft.omas import render_plot

        if renderer == "plotly":
            options["backend"] = "plotly"
        from vaft.plot.environment import close_new_figures_on_error

        # render_controls makes its pyplot figure before the first draw; a
        # refused draw must not leave it registered in a server that runs for
        # days.
        with close_new_figures_on_error(), self.scope():
            drawn = render_plot(name, self.ods, interactive=True, interaction_backend="none", **options)
        self._release()
        self.plot, self.renderer, self.interactive = name, renderer, drawn
        return drawn

    def scope(self) -> Any:
        """The figure options' type and line settings in force for a (re)draw.

        The interactive path redraws long after :meth:`select` returns, on
        every control change; whoever changes a control enters this scope so
        the redraw is drawn with the reader's options as the first draw was.
        """
        from vaft.plot.figure_options import figure_options_scope

        return figure_options_scope(self.figure_options)

    def compose(self, composition: Any, *, renderer: str = "auto") -> Any:
        """Draw ``composition`` from the loaded sources; the previous figure is released.

        ``"auto"`` draws with Plotly when every cell's plot has a Plotly
        rendering.  Database shots load every cell's IDS first.
        """
        if self.ods is None:
            raise RuntimeError("no source is open")
        if renderer not in RENDERERS:
            raise ValueError(f"renderer must be one of {', '.join(RENDERERS)}; got {renderer!r}")
        names = [cell.plot for cell in composition.cells]
        if renderer == "auto":
            renderer = "plotly" if all(self.supports_plotly(name) for name in names) else "matplotlib"
        if self.database:
            self.load_ids(list(dict.fromkeys(ident for name in names for ident in self.plot_ids(name))))
        import vaft.omas
        from vaft.plot.environment import close_new_figures_on_error

        with close_new_figures_on_error():
            drawn = vaft.omas.compose(
                composition, self.ods, backend=renderer if renderer == "plotly" else None,
                figure_options=self.figure_options.to_dict() or None,
            )
        self._release()
        self.composition, self._composed, self.renderer = composition, drawn, renderer
        return drawn

    def request(self, *, format: str | None = None, theme: str | None = None) -> Any:
        """The :class:`vaft.plot.PlotRequest` of what is on screen.

        ``format``/``theme`` are the reproduction's; with either, the request
        is for Matplotlib, which is what they shape.  A theme the controls
        chose is carried when none is given.
        """
        from vaft.plot import DataSource, PlotRequest

        if self.plot is None and self.composition is None:
            raise RuntimeError("nothing is drawn")
        kind = self.sources[0].kind
        source = DataSource(
            kind, tuple(source.value for source in self.sources),
            self.sources[0].namespace if kind == "shot" else None,
        )
        options = {}
        if self.composition is None:
            options = self.intent_options()
            chosen_theme = options.pop("theme", None)
            theme = theme or (chosen_theme if self.renderer == "matplotlib" else None)
        backend = "plotly" if self.renderer == "plotly" and not (format or theme) else None
        return PlotRequest(
            source=source, plot=self.plot, composition=self.composition, options=options,
            format=format, theme=theme, backend=backend, figure_options=self.figure_options or None,
        )

    def call_options(self) -> dict[str, Any]:
        """The plot call's keyword arguments for the controls as they stand.

        This is what the builder receives: ``"none"`` and empty choices are
        left out, and chosen channels stand in for the preset.  Use it for a
        one-off drawing (an export); to draw the plot again *with* its
        controls, use :meth:`carry_options`.
        """
        if self.state is None:
            return {}
        return {**self.state.as_options(), **self.state.as_style()}

    def intent_options(self) -> dict[str, Any]:
        """The control values the reader changed: :meth:`call_options` less the defaults.

        What reproduced code should say -- a control left where the plot put
        it is the plot's default and stays out, so the code follows the
        plot if its default improves.
        """
        if self.state is None:
            return {}
        from vaft.plot.navigation import ControlState

        untouched = ControlState(self.state.controls)
        defaults = {**untouched.as_options(), **untouched.as_style()}
        missing = object()
        return {key: value for key, value in self.call_options().items() if defaults.get(key, missing) != value}

    def carry_options(self) -> dict[str, Any]:
        """The controls' values to redraw the plot with, controls included.

        :meth:`call_options` hands chosen channels over as ``selection=[...]``,
        which is not one of the preset's choices; passed back to an
        interactive plot it would retire the preset control and fix the
        channels for good.  Here the preset and the channels travel as the
        controls themselves.
        """
        options = self.call_options()
        state = self.state
        if state is not None and state.values.get("channels"):
            options["selection"] = state["selection"]
            options["channels"] = tuple(state["channels"])
        if state is not None and "validity" in options:
            # A validity passed to an interactive plot pins the channels it
            # brings in, whatever the control says later; at its default it
            # is no choice of the reader's and is left to the control.
            try:
                if options["validity"] == state.spec("validity").default:
                    del options["validity"]
            except KeyError:
                pass
        return options

    def export(
        self, fmt: str = "png", *, dpi: int = 150, format: str | None = None, theme: str | None = None,
    ) -> bytes:
        """What is on screen, drawn by Matplotlib as ``fmt`` with the reader's options.

        The file is the publication (Matplotlib) rendering of :meth:`request`
        whichever renderer is on screen, from the data already loaded --
        nothing is read again.  ``format``/``theme`` shape it as the copied
        code would; a ``dpi`` field of the figure options, when it has one,
        wins over ``dpi``.
        """
        from .figure import EXPORT_FORMATS

        if self.plot is None and self.composition is None:
            raise RuntimeError("no plot is drawn")
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}; got {fmt!r}")
        import io

        import vaft.omas
        from vaft.plot import save_figure
        from vaft.plot.environment import close_new_figures_on_error

        request = self.request(format=format, theme=theme)
        presentation = {key: value for key, value in (("format", request.format), ("theme", request.theme)) if value}
        options = request.figure_options.to_dict() if request.figure_options is not None else None
        # A builder that fails after pyplot made its figure must not leak it
        # into a server that runs for days.
        with close_new_figures_on_error():
            if self.composition is not None:
                drawn = vaft.omas.compose(self.composition, self.ods, figure_options=options, **presentation)
            else:
                drawn = vaft.omas.render_plot(
                    self.plot, self.ods, **dict(request.options), figure_options=options, **presentation,
                )
        figure = drawn[0] if isinstance(drawn, tuple) else drawn
        buffer = io.BytesIO()
        save_figure(figure, buffer, format=fmt, dpi=getattr(self.figure_options, "dpi", None) or dpi)
        return buffer.getvalue()

    def sequence_control(self) -> Any:
        """The plot's slice-group control -- the sequence a video walks -- or ``None``.

        It is the control the player advances (``time_slice``, ``frame_index``
        or ``time_index``); a plot without one, or a composed figure, has no
        sequence to export as a movie.
        """
        if self.composition is not None or self.state is None:
            return None
        return next((c for c in self.state.controls if c.group == "slice"), None)

    def sequence_states(self) -> tuple[Any, ...]:
        """Every state of :meth:`sequence_control` for the plot as drawn, in order; empty without one.

        Stored slices are the control's usable ones.  A dense index counts
        the states of what is on screen -- a camera's chosen channel, the
        magnetics grid -- from :func:`vaft.plot.backend.discovery.sequence_values`,
        which the animation selects against; the control's own range is
        the record's default camera.
        """
        control = self.sequence_control()
        if control is None:
            return ()
        if control.kind != "range":
            return tuple(control.options)
        from vaft.omas.entries import normalize_entries
        from vaft.plot.backend.discovery import sequence_values

        options = {k: v for k, v in self.call_options().items() if k != control.name}
        try:
            _, _, values = sequence_values(self.plot, normalize_entries(self.ods), control.name, **options)
        except (NotImplementedError, ValueError):
            low, high, step = (int(v) for v in control.options)
            return tuple(range(low, high + 1, step))
        return tuple(range(len(values)))

    def _animation(
        self, *, fps: float, step: int, dpi: int | None, format: str | None, theme: str | None,
    ) -> Any:
        """``plot_*(..., animation=True)`` over the on-screen plot's sequence (#1400).

        Every other option is the reader's, as an image export takes them;
        the slice control's value is replaced by its states, every
        ``step``-th one.  Nothing is drawn until the result is saved.
        """
        import vaft.omas

        control = self.sequence_control()
        if self.composition is not None:
            raise ValueError("a composed figure has no sequence to animate; export one plot as a video")
        if control is None:
            raise ValueError(f"{self.plot} offers no sequence for this input (no slice control), so no video")
        if self.figure_options:
            # animation=True redraws its own frames and does not apply
            # figure options yet; dropping them silently would export a
            # different figure from the one on screen.
            raise ValueError(
                "video frames do not apply figure options yet; clear the figure options to export a video"
            )
        step = int(step)
        if step < 1:
            raise ValueError(f"step must be a positive number of states; got {step}")
        states = self.sequence_states()[::step]
        if len(states) > MAX_VIDEO_FRAMES:
            # The file travels to the browser in one message and is drawn on
            # the server thread: a five-thousand-frame movie is neither.
            raise ValueError(
                f"{len(states)} frames is more than the browser export draws ({MAX_VIDEO_FRAMES}); "
                f"take every {-(-len(self.sequence_states()) // MAX_VIDEO_FRAMES)}th state or more, "
                "or write the full movie with plot_*(..., animation=True).save() in Python"
            )
        request = self.request(format=format, theme=theme)
        options = {k: v for k, v in dict(request.options).items() if k != control.name}
        presentation = {key: value for key, value in (("format", request.format), ("theme", request.theme)) if value}
        extra = {"dpi": int(dpi)} if dpi else {}
        return vaft.omas.render_plot(
            self.plot, self.ods, animation=True, fps=float(fps), **{control.name: list(states)},
            **options, **presentation, **extra,
        )

    def export_video(
        self, fmt: str = "mp4", *, fps: float = 10.0, step: int = 1, dpi: int | None = None,
        format: str | None = None, theme: str | None = None,
    ) -> bytes:
        """The on-screen plot over its sequence as an ``fmt`` movie (#1400).

        The frames are the plot's states, every ``step``-th one, shown at
        ``fps`` -- presentation only: each frame's physical time is in
        :meth:`video_metadata`.  ``mp4``/``webm`` need PyAV
        (``pip install vaft[video]``); ``gif`` does not.
        """
        import tempfile
        from pathlib import Path

        from .figure import VIDEO_FORMATS

        if fmt not in VIDEO_FORMATS:
            raise ValueError(f"video format must be one of {', '.join(VIDEO_FORMATS)}; got {fmt!r}")
        movie = self._animation(fps=fps, step=step, dpi=dpi, format=format, theme=theme)
        with tempfile.TemporaryDirectory() as folder:
            path = movie.save(Path(folder) / f"export.{fmt}", sidecar=False)
            data = path.read_bytes()
        # The provenance of exactly this file -- its states, their times and
        # the encoder's own facts (a GIF's rounded frame delay) -- kept for
        # the frame-times download, so the two never describe different runs.
        self.last_video_metadata = movie.metadata
        return data

    def video_metadata(self) -> bytes:
        """The provenance of the last :meth:`export_video` file, as JSON: each frame's state and time."""
        import json

        if getattr(self, "last_video_metadata", None) is None:
            raise RuntimeError("no video has been exported yet; download the video first")
        return (json.dumps(self.last_video_metadata, indent=2) + "\n").encode()

    @property
    def state(self) -> Any:
        return None if self.interactive is None else self.interactive.state

    @property
    def figure(self) -> Any:
        if self._composed is not None:
            return self._composed[0] if isinstance(self._composed, tuple) else self._composed
        return None if self.interactive is None else self.interactive.figure

    def _release(self) -> None:
        from vaft.plot.environment import close_figure

        # Matplotlib figures are pyplot figures, which the pyplot registry
        # keeps alive until closed.
        if self.interactive is not None and self.renderer == "matplotlib":
            close_figure(self.interactive.figure)
        if isinstance(self._composed, tuple):
            close_figure(self._composed[0])
        self.plot, self.renderer, self.interactive = None, None, None
        self.composition, self._composed = None, None

    def release(self) -> None:
        """Let go of the plot on screen (the sources stay open)."""
        self._release()

    def close(self) -> None:
        """Release the figure and forget the sources."""
        self._release()
        self.sources, self.ods, self._catalog, self._everything = (), None, None, None
        self._shots, self._held, self._stored = {}, {}, {}


def _available(records: Any) -> Any:
    """The available records, as a catalog when ``records`` is one."""
    kept = [record for record in records if getattr(record, "available", True) is not False]
    return records.with_records(kept) if hasattr(records, "with_records") else kept


__all__ = ["BrowserSession", "RENDERERS", "SOURCE_KINDS", "Source", "load_source", "sample_shots"]
