"""One reproducible description of a figure, rendered or written as code (issue #1421).

A :class:`PlotRequest` holds everything a figure was asked for: the data
(:class:`DataSource` -- packaged samples, local files or database shots), one
canonical plot with its keywords *or* a :class:`~vaft.plot.FigureComposition`
of several (#1467), the ``format`` and ``theme`` of #689, the backend, and
the reader's explicit :class:`~vaft.plot.FigureOptions`.  From that one
object come

* :meth:`PlotRequest.render` -- the figure itself;
* :meth:`PlotRequest.to_python` -- the ordinary VAFT calls that draw it,
  optionally preceded by the Pylustrator bootstrap (#1202);
* :meth:`PlotRequest.to_cli` -- the equivalent ``vaft plot`` command;
* :meth:`PlotRequest.to_dict` -- plain JSON, read back by
  :meth:`PlotRequest.from_dict` and by ``vaft plot --request``.

Only intent is written: an inherited format, theme or option never appears
in the code or the command, so a figure reproduced later picks up whatever
the canonical defaults have become.

This request describes *a figure*; the scientific content of each plot
stays in the recipe's backend-neutral view model (``vaft.plot.models``),
which is what a backend -- Matplotlib, Plotly, a future MATLAB one (#1208)
-- draws.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .composition import FigureComposition, _plain, as_composition
from .figure_options import FigureOptions, as_figure_options

__all__ = ["DATA_KINDS", "DataSource", "PlotRequest"]

DATA_KINDS = ("sample", "file", "shot")


def _cli_value(value: Any) -> str:
    """``value`` as ``vaft plot --option`` reads it back: a Python literal, or a
    bare word when the CLI would read the word as that same string."""
    import ast

    if isinstance(value, str):
        try:
            ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return repr(value)


@dataclass(frozen=True)
class DataSource:
    """Where the data comes from: ``kind`` with one value or several to compare.

    ``"sample"`` -- packaged shots (offline); ``"file"`` -- local
    ODS/IMAS/GEQDSK files :func:`vaft.omas.load` reads; ``"shot"`` --
    database shots in ``namespace`` (HSDS, ``None`` = the default source).
    """

    kind: str
    values: tuple[Any, ...]
    namespace: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in DATA_KINDS:
            raise ValueError(f"DataSource.kind must be one of {', '.join(DATA_KINDS)}; got {self.kind!r}")
        import numbers
        import os

        scalar = isinstance(self.values, (str, os.PathLike, numbers.Integral))
        values = (self.values,) if scalar else tuple(self.values)
        if not values:
            raise ValueError("DataSource needs at least one value")
        values = (
            tuple(os.fspath(value) for value in values) if self.kind == "file"
            else tuple(int(value) for value in values)
        )
        object.__setattr__(self, "values", values)
        if self.namespace is not None and self.kind != "shot":
            raise ValueError("DataSource.namespace applies to database shots only")

    def load(self) -> Any:
        """The ODS (one value) or list of ODS (several) for ``sample``/``file``."""
        import vaft.omas

        if self.kind == "shot":
            raise TypeError("database shots are opened per plot, not loaded whole; render the request")
        loader = vaft.omas.sample_ods if self.kind == "sample" else vaft.omas.load
        objects = [loader(value) for value in self.values]
        return objects[0] if len(objects) == 1 else objects

    def python(self) -> str:
        """The expression that loads the data in a script."""
        call = "vaft.omas.sample_ods({!r})" if self.kind == "sample" else "vaft.omas.load({!r})"
        calls = [call.format(value) for value in self.values]
        return calls[0] if len(calls) == 1 else "[" + ", ".join(calls) + "]"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "values": list(self.values)}
        if self.namespace is not None:
            data["namespace"] = self.namespace
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataSource":
        return cls(kind=data["kind"], values=tuple(data["values"]), namespace=data.get("namespace"))


@dataclass(frozen=True)
class PlotRequest:
    """A figure: data, one plot or a composition, presentation and explicit options.

    Exactly one of ``plot`` (a canonical plot name, with its ``options``) and
    ``composition`` is given.  ``format``/``theme``/``backend`` are ``None``
    when inherited; ``figure_options`` holds explicit overrides only.
    """

    source: DataSource
    plot: str | None = None
    composition: FigureComposition | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    format: str | None = None
    theme: str | None = None
    backend: str | None = None
    figure_options: FigureOptions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, DataSource):
            object.__setattr__(self, "source", DataSource.from_dict(self.source))
        if (self.plot is None) == (self.composition is None):
            raise ValueError("a PlotRequest draws either one plot or one composition")
        if self.composition is not None:
            object.__setattr__(self, "composition", as_composition(self.composition))
            if self.options:
                raise ValueError("a composition carries each plot's options in its cells")
        object.__setattr__(self, "options", MappingProxyType(_plain(dict(self.options))))
        options = as_figure_options(self.figure_options)
        object.__setattr__(self, "figure_options", options if options else None)
        if self.backend not in (None, "matplotlib", "plotly"):
            raise ValueError(f"backend is 'matplotlib' or 'plotly'; got {self.backend!r}")
        if self.backend == "plotly" and (self.format or self.theme):
            raise ValueError("format= and theme= apply to Matplotlib; backend='plotly' does not apply them")
        for key in ("format", "theme", "figure_options", "backend"):
            if key in self.options:
                raise ValueError(f"{key} is a field of the request, not a plot option")

    #: Equal by value, not hashable: the options are mappings.
    __hash__ = None  # type: ignore[assignment]

    # -- keyword arguments the figure is drawn with ---------------------------------
    def _presentation(self) -> dict[str, Any]:
        keywords: dict[str, Any] = {}
        if self.format is not None:
            keywords["format"] = self.format
        if self.theme is not None:
            keywords["theme"] = self.theme
        if self.backend is not None:
            keywords["backend"] = self.backend
        if self.figure_options is not None:
            keywords["figure_options"] = self.figure_options.to_dict()
        return keywords

    def render(self, *, show: bool = False) -> Any:
        """Draw the figure: ``(Figure, axes)`` with Matplotlib, a Plotly figure with Plotly."""
        keywords = self._presentation()
        if self.source.kind == "shot":
            from vaft.database import plotting

            shots = list(self.source.values)
            shot = shots[0] if len(shots) == 1 else shots
            if self.composition is not None:
                return plotting.compose(self.composition, shot, self.source.namespace, show=show, **keywords)
            # The call to_python() writes, so the code reproduces this figure.
            return plotting.render(self.plot, shot, self.source.namespace, show=show, **dict(self.options), **keywords)
        import vaft.omas

        data = self.source.load()
        if self.composition is not None:
            return vaft.omas.compose(self.composition, data, show=show, **keywords)
        return vaft.omas.render_plot(self.plot, data, show=show, **dict(self.options), **keywords)

    # -- reproducible forms ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"source": self.source.to_dict()}
        if self.plot is not None:
            data["plot"] = self.plot
        if self.composition is not None:
            data["composition"] = self.composition.to_dict()
        if self.options:
            data["options"] = dict(self.options)
        for key in ("format", "theme", "backend"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        if self.figure_options is not None:
            data["figure_options"] = self.figure_options.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlotRequest":
        known = {"source", "plot", "composition", "options", "format", "theme", "backend", "figure_options"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"PlotRequest takes no {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return cls(
            source=DataSource.from_dict(data["source"]),
            plot=data.get("plot"),
            composition=FigureComposition.from_dict(data["composition"]) if data.get("composition") else None,
            options=data.get("options", {}),
            format=data.get("format"),
            theme=data.get("theme"),
            backend=data.get("backend"),
            figure_options=FigureOptions.from_dict(data["figure_options"]) if data.get("figure_options") else None,
        )

    def to_python(self, *, pylustrator: bool = False) -> str:
        """The script that draws this figure with the ordinary VAFT API.

        ``pylustrator=True`` starts Pylustrator first (#1202), so the figure
        opens in its editor for one-off finishing; nothing else changes.
        """
        lines: list[str] = []
        if pylustrator:
            lines += ["import pylustrator", "", "pylustrator.start()", ""]
        lines.append("import vaft")
        if self.composition is not None:
            lines.append("from vaft.plot import FigureComposition")
        lines.append("")
        keywords = [f"{key}={value!r}" for key, value in {**dict(self.options), **self._presentation()}.items()]
        result = "figure" if self.backend == "plotly" else "figure, axes"
        if self.composition is not None:
            lines.append(f"composition = FigureComposition.from_dict({self.composition.to_dict()!r})")
        if self.source.kind == "shot":
            shots = list(self.source.values)
            shot = repr(shots[0] if len(shots) == 1 else shots)
            where = [f"source={self.source.namespace!r}"] if self.source.namespace is not None else []
            if self.composition is not None:
                arguments = ["composition", shot, *where, *keywords]
                lines.append(f"{result} = vaft.database.plotting.compose({', '.join(arguments)})")
            else:
                arguments = [shot, *where, *keywords]
                lines.append(f"{result} = vaft.database.plot_{self.plot}({', '.join(arguments)})")
        else:
            lines.append(f"data = {self.source.python()}")
            if self.composition is not None:
                lines.append(f"{result} = vaft.omas.compose({', '.join(['composition', 'data', *keywords])})")
            else:
                lines.append(f"{result} = vaft.omas.plot_{self.plot}({', '.join(['data', *keywords])})")
        if pylustrator:
            lines += ["", "import matplotlib.pyplot as plt", "", "plt.show()"]
        return "\n".join(lines) + "\n"

    def to_cli(self, *, out: str | None = None) -> str:
        """The ``vaft plot`` command that draws (or, with ``out``, writes) this figure."""
        words = ["vaft", "plot"]
        if self.plot is not None:
            words.append(self.plot)
        flag = {"sample": "--sample", "file": "--file", "shot": "--shot"}[self.source.kind]
        for value in self.source.values:
            words += [flag, str(value)]
        if self.source.namespace is not None:
            words += ["--source", self.source.namespace]
        if self.composition is not None:
            words += ["--compose", json.dumps(self.composition.to_dict(), separators=(",", ":"))]
        for key, value in self.options.items():
            words += ["--option", f"{key}={_cli_value(value)}"]
        for key in ("format", "theme", "backend"):
            if getattr(self, key) is not None:
                words += [f"--{key}", getattr(self, key)]
        if self.figure_options is not None:
            words += ["--figure-options", json.dumps(self.figure_options.to_dict(), separators=(",", ":"))]
        if out is not None:
            words += ["--out", out]
        return shlex.join(words)
