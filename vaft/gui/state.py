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
        self._catalog: Any = None
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

            catalog, stored = self._discover_shots(chosen)
            self.close()
            self._stored = stored
            self._shots = {source: omas.ODS() for source in chosen}
            self._held = {source: set() for source in chosen}
            data = list(self._shots.values())
            self.sources, self.ods, self._catalog = chosen, data[0] if len(data) == 1 else data, catalog
            return
        loaded = [load_source(source) for source in chosen]
        data = loaded[0] if len(loaded) == 1 else loaded
        catalog = self._discover(data)
        self.close()
        self.sources, self.ods, self._catalog = chosen, data, catalog

    def _discover(self, data: Any) -> Any:
        from vaft.omas import available_plots

        return available_plots(data)

    def _discover_shots(self, sources: Sequence[Source]) -> tuple[list[Any], dict[Source, set[str]]]:
        """The plots every shot can draw, and the IDS each one stores."""
        from vaft.database import available_plots, stored_ids

        stored = {source: set(stored_ids(source.value, source.namespace)) for source in sources}
        catalogs = [available_plots(source.value, source.namespace) for source in sources]
        shared = set.intersection(*({c.name for c in catalog} for catalog in catalogs))
        return [capability for capability in catalogs[0] if capability.name in shared], stored

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
        with close_new_figures_on_error():
            drawn = render_plot(name, self.ods, interactive=True, interaction_backend="none", **options)
        self._release()
        self.plot, self.renderer, self.interactive = name, renderer, drawn
        return drawn

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
        return options

    def export(self, fmt: str = "png", *, dpi: int = 150, settings: Any = None) -> bytes:
        """The current plot, with its controls, drawn by Matplotlib as ``fmt``.

        The file is always the publication (Matplotlib) rendering, whichever
        renderer is on screen; ``settings`` (a
        :class:`~vaft.gui.figure.FigureSettings`) sizes and limits it.
        """
        from .figure import EXPORT_FORMATS

        if self.interactive is None:
            raise RuntimeError("no plot is drawn")
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}; got {fmt!r}")
        import io

        from vaft.omas import render_plot
        from vaft.plot.environment import close_figure, close_new_figures_on_error

        # A builder that fails after pyplot made its figure must not leak it
        # into a server that runs for days.
        with close_new_figures_on_error():
            drawn = render_plot(self.plot, self.ods, **self.call_options())
        figure = drawn[0] if isinstance(drawn, tuple) else drawn
        try:
            sized = settings is not None and settings.sized
            if settings is not None:
                settings.apply(figure, "matplotlib", dpi=dpi)
            buffer = io.BytesIO()
            # An explicit size is the file's size; otherwise trim the margins.
            figure.savefig(buffer, format=fmt, dpi=dpi, bbox_inches=None if sized else "tight")
        finally:
            close_figure(figure)
        return buffer.getvalue()

    @property
    def state(self) -> Any:
        return None if self.interactive is None else self.interactive.state

    @property
    def figure(self) -> Any:
        return None if self.interactive is None else self.interactive.figure

    def _release(self) -> None:
        if self.interactive is not None and self.renderer == "matplotlib":
            from vaft.plot.environment import close_figure

            # interaction_backend="none" draws on a pyplot figure, which the
            # pyplot registry keeps alive until closed.
            close_figure(self.interactive.figure)
        self.plot, self.renderer, self.interactive = None, None, None

    def close(self) -> None:
        """Release the figure and forget the sources."""
        self._release()
        self.sources, self.ods, self._catalog = (), None, None
        self._shots, self._held, self._stored = {}, {}, {}


__all__ = ["BrowserSession", "RENDERERS", "SOURCE_KINDS", "Source", "load_source", "sample_shots"]
