"""The reference browser application: pick sources, a plot and its controls (#1086).

It is deliberately small -- a source picker, the plot catalog of those
sources, the controls the chosen plot declares, figure settings and export --
and exists to prove the layer: the same ``vaft gui`` works on a workstation,
over SSH port forwarding and on a cluster node, with every list and control
coming from the plot-discovery API rather than from this module.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Sequence
from typing import Any

from ._require import require_panel
from .figure import EXPORT_FORMATS, FigureSettings
from .state import BrowserSession, Source, sample_shots
from .widgets import panel_controls

#: Addresses that only this machine can reach.
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

_SOURCE_LABELS = {"Sample": "sample", "File": "file", "Database shot": "shot"}
_RENDERER_LABELS = {"Interactive": "plotly", "Static": "matplotlib"}

#: The Plotly toolbar: zoom, pan, autoscale and image export are Plotly's own;
#: the drawing tools mark lines, regions and shapes on the figure, and the
#: spike lines read coordinates off both axes.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": [
        "drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape", "toggleSpikelines",
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
_DEFAULT_HEIGHT = 520

#: What vaft.omas.load reads (vaft/database/_local.py), shown by the file source.
FILE_FORMATS = (
    "Reads OMAS `.json` `.json.gz` `.h5` `.nc` · IMAS HDF5 entry directory "
    "(`master.h5` or per-IDS `.h5`) and IMAS `.nc` · GEQDSK file, or a directory of GEQDSK files."
)


def _first_line(error: BaseException) -> str:
    text = str(error).strip().splitlines()
    return f"{type(error).__name__}: {text[0] if text else ''}"[:300]


def parse_shots(text: str) -> list[int]:
    """Shot numbers from ``"39915, 41524 41672"``; ranges are not expanded."""
    tokens = [token for token in re.split(r"[\s,;]+", text or "") if token]
    try:
        return [int(token) for token in tokens]
    except ValueError:
        raise ValueError(f"shots must be whole numbers separated by commas or spaces; got {text!r}") from None


class BrowserApp:
    """Source picker, plot selector, controls, figure settings and the figure."""

    def __init__(self, session: BrowserSession | None = None, *, plot: str | None = None) -> None:
        pn = require_panel()
        self.session = session or BrowserSession()
        self.settings = FigureSettings()
        self._preferred_plot = plot
        self._renderer_choice = "plotly"
        self._plot_count = 0
        self._updating = False

        # -- source
        samples = list(sample_shots())
        self.kind = pn.widgets.RadioButtonGroup(options=_SOURCE_LABELS, value="sample")
        self.sample = pn.widgets.MultiChoice(
            label="Sample shots", options=samples, value=samples[:1],
            placeholder="choose one or more shots",
        )
        self.path = pn.widgets.TextAreaInput(
            label="File paths (one per line; several are compared)", rows=2,
            placeholder="/path/to/data.h5",
        )
        self.formats = pn.pane.Markdown(FILE_FORMATS, margin=(0, 10), styles={"font-size": "0.85em"})
        self.browse = pn.widgets.Toggle(label="Browse server files")
        self.use_selected = pn.widgets.Button(label="Load selected", color="primary")
        self.close_browser = pn.widgets.Button(label="Close")
        self.browser: Any = None
        # Files from the reader's own computer, copied to the server first:
        # the loader reads paths, and the data may live only on the laptop.
        self.upload = pn.widgets.FileInput(multiple=True)
        self._uploads: Any = None
        self._upload_count = 0
        self.browser_box = pn.Column(visible=False, sizing_mode="stretch_width")
        self.shots = pn.widgets.TextInput(label="Shots", placeholder="39915, 41524")
        self.namespace = pn.widgets.TextInput(label="Namespace", placeholder="main")
        # Database shots are read IDS by IDS: a plot fetches what it needs when
        # first drawn, and these fetch chosen IDS ahead of any plot.
        self.ids_choice = pn.widgets.MultiChoice(
            label="Load IDS now (optional)", options=[], disabled=True,
            placeholder="after Load: IDS the available plots read",
        )
        self.ids_button = pn.widgets.Button(label="Load chosen IDS", disabled=True)
        self.load_button = pn.widgets.Button(label="Load", color="primary")
        self._source_inputs = pn.Column(self.sample)

        # -- plot
        self.plot = pn.widgets.Select(label="Plot", groups={"": []}, disabled=True)
        self.renderer = pn.widgets.RadioButtonGroup(options=_RENDERER_LABELS, value="plotly", disabled=True)
        self.controls = pn.Column()

        # -- figure settings
        self.width = pn.widgets.IntInput(label="Width [px]", value=None, start=50, step=50, placeholder="auto")
        self.height = pn.widgets.IntInput(label="Height [px]", value=None, start=50, step=50, placeholder="auto")
        self.xmin = pn.widgets.FloatInput(label="x min", value=None, placeholder="auto")
        self.xmax = pn.widgets.FloatInput(label="x max", value=None, placeholder="auto")
        self.ymin = pn.widgets.FloatInput(label="y min", value=None, placeholder="auto")
        self.ymax = pn.widgets.FloatInput(label="y max", value=None, placeholder="auto")
        self.xlog = pn.widgets.Checkbox(label="log x")
        self.ylog = pn.widgets.Checkbox(label="log y")
        self.reset = pn.widgets.Button(label="Reset to automatic")
        self._setting_widgets = {
            "width": self.width, "height": self.height, "xmin": self.xmin, "xmax": self.xmax,
            "ymin": self.ymin, "ymax": self.ymax, "xlog": self.xlog, "ylog": self.ylog,
        }

        # -- export
        self.export_format = pn.widgets.Select(label="Format", options=list(EXPORT_FORMATS), value="png", width=90)
        self.export_dpi = pn.widgets.IntInput(label="DPI", value=150, start=50, end=600, step=50, width=90)
        self.download = pn.widgets.FileDownload(
            callback=self._export, filename="figure.png", label="Download figure",
            color="success", disabled=True,
        )

        # -- main area
        self.status = pn.pane.Markdown("No source loaded.")
        self.alert = pn.pane.Alert("", alert_type="danger", visible=False)
        self.static = pn.pane.Matplotlib(None, tight=True, dpi=110, sizing_mode="stretch_width", visible=False)
        self.interactive = pn.pane.Plotly(
            None, config=PLOTLY_CONFIG, sizing_mode="stretch_width", min_height=_DEFAULT_HEIGHT, visible=False,
        )

        self.kind.param.watch(self._on_kind, "value")
        self.browse.param.watch(self._on_browse, "value")
        self.use_selected.on_click(lambda _event: self._load_selected())
        self.close_browser.on_click(lambda _event: setattr(self.browse, "value", False))
        self.upload.param.watch(self._on_upload, "value")
        self.ids_button.on_click(lambda _event: self._load_chosen_ids())
        self.load_button.on_click(lambda _event: self._load_requested())
        self.plot.param.watch(self._on_plot, "value")
        self.renderer.param.watch(self._on_renderer, "value")
        for widget in self._setting_widgets.values():
            widget.param.watch(self._on_settings, "value")
        self.reset.on_click(lambda _event: self._reset_settings())
        self.export_format.param.watch(lambda _event: self._name_download(), "value")

    # -- source ---------------------------------------------------------------
    def _on_kind(self, event: Any) -> None:
        pn = require_panel()
        self._source_inputs.objects = {
            "sample": [self.sample],
            "file": [
                self.path, self.browse,
                pn.pane.Markdown("Or upload from this computer:", margin=(10, 10, 0, 10)), self.upload,
                self.formats,
            ],
            "shot": [
                self.shots, self.namespace,
                pn.pane.Markdown(
                    "Nothing is downloaded up front: each plot loads only the IDS it reads, "
                    "the first time it is drawn.", margin=(0, 10), styles={"font-size": "0.85em"},
                ),
                self.ids_choice, self.ids_button,
            ],
        }[event.new]

    def requested_sources(self) -> list[Source]:
        """The sources the source widgets currently name."""
        kind = self.kind.value
        if kind == "sample":
            return [Source("sample", shot) for shot in self.sample.value]
        if kind == "file":
            lines = [line.strip() for line in (self.path.value or "").splitlines() if line.strip()]
            if not lines:
                raise ValueError("give a file path, or browse the server's files")
            return [Source("file", line) for line in lines]
        return [Source("shot", shot, self.namespace.value or None) for shot in parse_shots(self.shots.value)]

    def _load_requested(self) -> None:
        try:
            sources = self.requested_sources()
        except ValueError as error:
            self._show_error(error)
            return
        self.load(sources)

    def load(self, sources: Source | Sequence[Source]) -> bool:
        """Open ``sources`` and draw their first plot; ``False`` when it failed."""
        chosen = [sources] if isinstance(sources, Source) else list(sources)
        wanted_label = ", ".join(source.label for source in chosen) or "nothing"
        self._clear_error()
        self.status.object = f"Loading {wanted_label} ..."
        on_screen = self.session.plot
        try:
            self.session.open(chosen)
            groups = self.session.grouped_plots()
        except Exception as error:
            kept = self.session.label
            self.status.object = f"Could not load {wanted_label}." + (
                f" Still showing **{kept}**." if kept else ""
            )
            self._show_error(error)
            return False
        names = [name for members in groups.values() for name in members]
        self._plot_count = len(names)
        self._update_status()
        database = self.session.database
        self.ids_choice.param.update(
            options=self.session.plotted_ids() if database else [], value=[], disabled=not database,
        )
        self.ids_button.disabled = not database
        self.browse.value = False  # the chosen files are open: the browser has done its job
        if not names:
            # The previous plot was released with its sources.
            self._clear_plot()
            return True
        # The plot on screen stays when the new sources offer it: adding a
        # shot to compare should not jump to another plot.
        wanted = self._preferred_plot if self._preferred_plot in names else (
            on_screen if on_screen in names else names[0]
        )
        self._preferred_plot = None
        # Setting groups may itself pick a value and fire the watcher; the
        # explicit draw below then keys on the value that was actually chosen.
        self.plot.param.update(groups=groups, disabled=False)
        if self.plot.value == wanted:
            self.show(wanted, renderer=self._renderer_for(wanted))
        else:
            self.plot.value = wanted
        if self.session.plot is None:
            # Nothing could be drawn from the new sources (the reason is on
            # screen): no control may stay bound to the released plot.
            self._clear_plot(keep_selector=True)
        return True

    def _clear_plot(self, *, keep_selector: bool = False) -> None:
        """Nothing drawn: no figure, no controls, nothing to export."""
        self.controls.objects = []
        self.interactive.object = self.static.object = None
        self.interactive.visible = self.static.visible = False
        self.download.disabled = True
        self.renderer.disabled = True
        if not keep_selector:
            self.plot.disabled = True

    def _update_status(self) -> None:
        text = f"**{self.session.label}**: {self._plot_count} plots available."
        if self.session.database:
            held = [name for name in self.session.held_ids() if name != "dataset_description"]
            text += " In memory: " + (", ".join(held) if held else "nothing yet") + "."
        self.status.object = text

    def _load_chosen_ids(self) -> None:
        self._clear_error()
        try:
            self.session.load_ids(list(self.ids_choice.value))
        except Exception as error:
            self._show_error(error)
            return
        self.ids_choice.value = []
        self._update_status()

    # -- plot -----------------------------------------------------------------
    def _on_plot(self, event: Any) -> None:
        if event.new:
            self.show(event.new, renderer=self._renderer_for(event.new))

    def _renderer_for(self, name: str) -> str:
        """The renderer the reader last chose, where ``name`` offers it."""
        if self._renderer_choice == "matplotlib":
            return "matplotlib"
        return "auto" if self.session.ods is not None and self.session.supports_plotly(name) else "matplotlib"

    def _on_renderer(self, event: Any) -> None:
        if self._updating or not self.session.plot or event.new == self.session.renderer:
            return
        # Keep what the reader chose; the Matplotlib theme has no Plotly twin.
        carried = self.session.carry_options()
        carried.pop("theme", None)
        name = self.session.plot
        self._renderer_choice = event.new
        if not self.show(name, renderer=event.new, **carried) and not self.show(name, renderer=event.new):
            # Neither with the reader's values nor afresh: the old drawing is
            # still on screen, so the toggle goes back to naming it.
            self._renderer_choice = event.old
            self._updating = True
            try:
                self.renderer.value = self.session.renderer
            finally:
                self._updating = False

    def show(self, name: str, *, renderer: str = "auto", **options: Any) -> bool:
        """Draw ``name`` with its controls; ``False`` when it could not be drawn."""
        self._clear_error()
        try:
            drawn = self.session.select(name, renderer=renderer, **options)
        except Exception as error:
            self._show_error(error)
            return False
        # Observers run only after a redraw succeeded (a refused one raises
        # first), so a message about an earlier refused frame is cleared --
        # playback walks past a slice that cannot be drawn.
        drawn.state.subscribe(lambda _state: (self._clear_error(), self._refresh()))
        self.controls.objects = panel_controls(drawn.state, on_error=self._on_control_error)
        plotly = self.session.renderer == "plotly"
        self._updating = True
        try:
            self.renderer.param.update(
                value=self.session.renderer, disabled=not self.session.supports_plotly(name),
            )
        finally:
            self._updating = False
        self.interactive.visible, self.static.visible = plotly, not plotly
        (self.static if plotly else self.interactive).object = None
        self.download.disabled = False
        self._name_download()
        if self.session.database:
            self._update_status()  # the plot may have fetched IDS
        self._refresh()
        return True

    @property
    def figure(self) -> Any:
        """The pane showing the current plot."""
        return self.interactive if self.session.renderer == "plotly" else self.static

    def _on_control_error(self, error: Exception) -> None:
        self._show_error(error)
        # render_controls wrote the refusal on the figure it kept.
        self._refresh()

    def _refresh(self) -> None:
        figure = self.session.figure
        if figure is None:
            return
        self._size_panes()
        if self.session.renderer == "plotly":
            # Each change rebuilds the Plotly figure, and the settings go on a
            # copy so that clearing one brings the plot's own layout back.  A
            # uirevision per plot and per explicit limit keeps the reader's
            # zoom across rebuilds -- a new slice must not reset the view --
            # while a typed limit still takes effect.
            import plotly.graph_objects as go

            shown = self.settings.apply(go.Figure(figure), "plotly")
            shown.update_layout(uirevision=f"{self.session.plot}|{self.settings.key}")
            self.interactive.object = shown
        else:
            # The Matplotlib figure is redrawn in place: lay the settings over
            # it and re-render the image.
            self.settings.apply(figure, "matplotlib")
            if self.static.object is not figure:
                self.static.object = figure
            else:
                self.static.param.trigger("object")

    def _size_panes(self) -> None:
        width, height = self.settings.width, self.settings.height
        for pane in (self.interactive, self.static):
            if width is not None:
                pane.param.update(sizing_mode="fixed", width=width, height=height or _DEFAULT_HEIGHT)
            elif pane is self.interactive:
                pane.param.update(sizing_mode="stretch_width", height=height or _DEFAULT_HEIGHT)
            else:
                pane.param.update(sizing_mode="stretch_width", height=height)

    # -- figure settings ------------------------------------------------------
    def _on_settings(self, _event: Any) -> None:
        if self._updating:
            return
        self._clear_error()
        values = {name: widget.value for name, widget in self._setting_widgets.items()}
        try:
            settings = FigureSettings(**values)
        except ValueError as error:
            self._show_error(error)
            self._show_settings(self.settings)
            return
        if not self.apply_settings(settings):
            self._show_settings(self.settings)

    def _show_settings(self, settings: FigureSettings) -> None:
        """Put the Figure widgets back on ``settings`` without re-applying them.

        A refused value must not stay in its box, or the next edit of any
        other field would submit it again.
        """
        self._updating = True
        try:
            for name, widget in self._setting_widgets.items():
                widget.value = getattr(settings, name)
        finally:
            self._updating = False

    def apply_settings(self, settings: FigureSettings) -> bool:
        """Lay ``settings`` over the current plot; ``False`` when refused.

        Settings the figure refuses are not kept: every later redraw applies
        the stored settings, and one bad value must not freeze the plot.
        """
        previous, self.settings = self.settings, settings
        if self.session.plot is None:
            return True
        try:
            if self.session.renderer == "matplotlib" and previous != settings:
                # Limits and sizes are set on the drawn axes; undoing one needs
                # the plot drawn afresh, with the reader's control values.
                if not self.show(self.session.plot, renderer="matplotlib", **self.session.carry_options()):
                    raise ValueError(self.alert.object or "the plot could not be drawn")
            else:
                self._refresh()
        except Exception as error:
            self.settings = previous
            self._refresh()
            self._show_error(error)
            return False
        return True

    def _reset_settings(self) -> None:
        self._updating = True
        try:
            for name, widget in self._setting_widgets.items():
                widget.value = False if name in ("xlog", "ylog") else None
        finally:
            self._updating = False
        self.apply_settings(FigureSettings())

    # -- export ---------------------------------------------------------------
    def _name_download(self) -> None:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{self.session.plot}_{self.session.label}").strip("_")
        self.download.filename = f"{stem}.{self.export_format.value}"

    def _export(self) -> Any:
        import io

        try:
            data = self.session.export(
                self.export_format.value, dpi=int(self.export_dpi.value or 150), settings=self.settings,
            )
        except Exception as error:
            self._show_error(error)
            return None
        return io.BytesIO(data)

    # -- file browser ---------------------------------------------------------
    def _on_browse(self, event: Any) -> None:
        if event.new and self.browser is None:
            pn = require_panel()
            # The files of the machine VAFT runs on -- over SSH, the remote
            # host's -- which is where the data and the loader are.  Built on
            # first use: listing a directory is not free on a network disk.
            self.browser = pn.widgets.FileSelector(
                directory=os.getcwd(), root_directory="/", sizing_mode="stretch_width",
            )
            self.browser_box.objects = [
                pn.pane.Markdown(
                    "**Server files.** Select one or more files, or an IMAS entry / GEQDSK "
                    "directory, move them to the right-hand list and load them. " + FILE_FORMATS,
                ),
                self.browser, pn.Row(self.use_selected, self.close_browser),
            ]
        self.browser_box.visible = bool(event.new)
        self.browse.label = "Hide server files" if event.new else "Browse server files"

    def _load_selected(self) -> None:
        chosen = list(self.browser.value if self.browser is not None else ())
        if not chosen:
            self._show_error(ValueError("no file selected: move files to the right-hand list first"))
            return
        self.path.value = "\n".join(chosen)
        self.load([Source("file", path) for path in chosen])

    # -- upload -------------------------------------------------------------
    def _on_upload(self, event: Any) -> None:
        if not event.new or self._updating:
            return
        contents = event.new if isinstance(event.new, list) else [event.new]
        names = self.upload.filename
        names = names if isinstance(names, list) else [names]
        try:
            sources = self._store_upload(names, contents)
        except (OSError, ValueError) as error:
            self._show_error(error)
            return
        # The bytes are on disk now: the widget need not keep up to 512 MB in
        # memory for the session, and the same file can be uploaded again.
        self._updating = True
        try:
            self.upload.param.update(value=None, filename=None)
        finally:
            self._updating = False
        self.path.value = "\n".join(str(source.value) for source in sources)
        if self.load(sources):
            self._prune_uploads()

    def _prune_uploads(self) -> None:
        """Remove earlier upload folders that no open source reads."""
        import shutil
        from pathlib import Path

        if self._uploads is None:
            return
        root = Path(self._uploads.name)
        in_use = set()
        for source in self.session.sources:
            if source.kind != "file":
                continue
            path = Path(str(source.value))
            # The upload-N folder itself, or the one holding the file.
            in_use |= {candidate for candidate in (path, *path.parents) if candidate.parent == root}
        for folder in root.iterdir():
            if folder.is_dir() and folder not in in_use:
                shutil.rmtree(folder, ignore_errors=True)

    def _store_upload(self, names: Sequence[str], contents: Sequence[bytes]) -> list[Source]:
        """Write one upload to its own server-side folder; the sources it holds.

        The files keep their names, which the loader reads the format from.
        Uploaded together, an IMAS entry's ``master.h5`` and per-IDS files
        are one source (their folder); anything else is one source per file.
        The folders live until the browser session ends.
        """
        import tempfile
        from pathlib import Path

        if not names:
            raise ValueError("no file was uploaded")
        if self._uploads is None:
            self._uploads = tempfile.TemporaryDirectory(prefix="vaft-gui-")
        self._upload_count += 1
        folder = Path(self._uploads.name) / f"upload-{self._upload_count}"
        folder.mkdir()
        paths = []
        for name, data in zip(names, contents):
            # Only the file name: a browser-supplied path must not escape the folder.
            target = folder / Path(str(name)).name
            target.write_bytes(data)
            paths.append(target)
        if any(path.name == "master.h5" for path in paths):
            return [Source("file", str(folder))]
        return [Source("file", str(path)) for path in paths]

    # -- messages -------------------------------------------------------------
    def _show_error(self, error: BaseException) -> None:
        self.alert.object = _first_line(error)
        self.alert.visible = True

    def _clear_error(self) -> None:
        self.alert.visible = False
        self.alert.object = ""

    # -- layout ---------------------------------------------------------------
    def sidebar(self) -> list[Any]:
        pn = require_panel()
        figure = pn.Card(
            self.width, self.height, self.xmin, self.xmax, self.ymin, self.ymax,
            self.xlog, self.ylog, self.reset,
            title="Figure", collapsed=True, sizing_mode="stretch_width",
        )
        export = pn.Card(
            pn.Row(self.export_format, self.export_dpi), self.download,
            title="Export", collapsed=True, sizing_mode="stretch_width",
        )
        return [
            pn.pane.Markdown("### Source"), self.kind, self._source_inputs, self.load_button,
            pn.layout.Divider(),
            pn.pane.Markdown("### Plot"), self.plot, self.renderer, self.controls,
            figure, export,
        ]

    def main(self) -> list[Any]:
        pn = require_panel()
        # One column: the template frames each main item, and a hidden alert
        # would still leave its empty frame between the status and the figure.
        return [pn.Column(
            self.status, self.alert, self.browser_box, self.interactive, self.static,
            sizing_mode="stretch_width",
        )]

    def view(self) -> Any:
        """The page served by ``vaft gui``."""
        pn = require_panel()
        return pn.template.FastListTemplate(
            title="VAFT", sidebar=self.sidebar(), main=self.main(), sidebar_width=360,
        )

    def close(self) -> None:
        self.session.close()
        if self._uploads is not None:
            self._uploads.cleanup()
            self._uploads = None


def _as_list(value: int | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    return [int(value)] if isinstance(value, (int, str)) else [int(v) for v in value]


def build_app(
    *,
    sample: int | Sequence[int] | None = None,
    file: str | None = None,
    shot: int | Sequence[int] | None = None,
    namespace: str | None = None,
    plot: str | None = None,
) -> BrowserApp:
    """A :class:`BrowserApp`, with one kind of source loaded when the page opens.

    ``sample`` and ``shot`` take one shot or several to compare; with no
    source given, the first packaged sample is loaded.
    """
    given = [value for value in (sample, file, shot) if value is not None]
    if len(given) > 1:
        raise ValueError("give at most one of sample=, file= and shot=")
    pn = require_panel()
    # Plotly's JavaScript must be on the page before the first figure, or the
    # pane errors in the browser ("reading 'relayout'") and stays blank.
    import importlib.util

    pn.extension(*(("plotly",) if importlib.util.find_spec("plotly") else ()))
    app = BrowserApp(plot=plot)
    if sample is not None:
        initial = [Source("sample", value) for value in _as_list(sample)]
    elif file is not None:
        initial = [Source("file", file)]
    elif shot is not None:
        initial = [Source("shot", value, namespace) for value in _as_list(shot)]
    else:
        samples = sample_shots()
        initial = [Source("sample", samples[0])] if samples else []
    if initial:
        app.kind.value = initial[0].kind
        if initial[0].kind == "sample":
            app.sample.value = [source.value for source in initial]
        elif initial[0].kind == "file":
            app.path.value = str(initial[0].value)
        else:
            app.shots.value = ", ".join(str(source.value) for source in initial)
            app.namespace.value = namespace or ""
        # After the page is up, so a slow catalog shows this, not a blank tab.
        app.status.object = "Loading " + ", ".join(source.label for source in initial) + " ..."
        pn.state.onload(lambda: app.load(initial))
    return app


AUTH_MODES = ("auto", "password", "none")
#: The largest upload one browser message may carry (all files of one upload).
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def serve(
    *,
    address: str = "127.0.0.1",
    port: int = 5006,
    show: bool | None = None,
    websocket_origin: Sequence[str] | None = None,
    auth: str = "auto",
    password: str | None = None,
    **app_options: Any,
) -> Any:
    """Serve :func:`build_app` until interrupted; one app per browser session.

    ``address`` defaults to loopback: on a remote host the page is reached
    through SSH or VS Code port forwarding, never by exposing the port.
    ``show=None`` opens a browser unless the process runs under SSH.

    Loopback is private only on a machine nobody else logs in to.  On a
    shared login or compute node any local user could reach the port and read
    files and database data as you, so ``auth="auto"`` asks for a password
    whenever the server runs under SSH or binds another address.  The password
    is ``password``, else ``$VAFT_GUI_PASSWORD``, else a random one printed to
    the terminal.
    """
    pn = require_panel()
    import secrets

    import matplotlib

    if auth not in AUTH_MODES:
        raise ValueError(f"auth must be one of {', '.join(AUTH_MODES)}; got {auth!r}")
    # Figures are rendered to images for the browser; the server may have no
    # display at all (an SSH host, a compute node), and a GUI backend there
    # would fail or open windows nobody sees.
    matplotlib.use("Agg")
    remote = bool(os.environ.get("SSH_CONNECTION"))
    if address not in LOOPBACK:
        warnings.warn(
            f"serving on {address!r} exposes the GUI beyond this machine without "
            "HTTPS; prefer the default 127.0.0.1 and port forwarding",
            UserWarning,
            stacklevel=2,
        )
    if show is None:
        show = not remote
    # The browser's origin carries the port it connects to.  Bokeh admits no
    # port wildcard (an entry without a port means :80), so a forward to
    # another local port (ssh -L 8080:localhost:5006, or VS Code picking a
    # free port) is named with websocket_origin / --allow-websocket-origin.
    # Bokeh's allow-list takes host:port only -- an IPv6 literal such as
    # [::1]:5006 makes the server refuse to start -- so IPv6 addresses are not
    # listed; a browser on ::1 is named with --allow-websocket-origin.
    origins = list(websocket_origin or ()) + [f"localhost:{port}", f"127.0.0.1:{port}"]
    if address not in LOOPBACK:
        # The page is then opened by this host's name or address.
        import socket

        if address in ("0.0.0.0", "::"):
            origins += [f"{socket.gethostname()}:{port}", f"{socket.getfqdn()}:{port}"]
        elif ":" not in address:
            origins.append(f"{address}:{port}")
    protected = auth == "password" or (auth == "auto" and (remote or address not in LOOPBACK))
    options: dict[str, Any] = {}
    if protected:
        password = password or os.environ.get("VAFT_GUI_PASSWORD") or None
        if password is None:
            password = secrets.token_urlsafe(12)
            print(f"vaft gui: password for this server: {password}", flush=True)
        options.update(basic_auth=password, cookie_secret=secrets.token_urlsafe(32))
    # An upload arrives as one websocket message; Bokeh's default cap is 20 MB.
    options["websocket_max_message_size"] = MAX_UPLOAD_BYTES

    def page() -> Any:
        app = build_app(**app_options)
        pn.state.on_session_destroyed(lambda _context: app.close())
        return app.view()

    return pn.serve(
        {"/": page}, address=address, port=port, show=show,
        websocket_origin=origins, title="VAFT", **options,
    )


__all__ = ["AUTH_MODES", "BrowserApp", "LOOPBACK", "PLOTLY_CONFIG", "build_app", "parse_shots", "serve"]
