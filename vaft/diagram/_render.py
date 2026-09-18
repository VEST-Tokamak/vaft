"""The TikZ renderer and the :class:`Diagram` a builder returns.

Scene -> TikZ source is pure Python and needs nothing installed. TikZ -> SVG
runs ``latex`` (DVI mode) and ``dvisvgm``, and only when an SVG is actually
asked for, so ``import vaft.diagram`` and building a diagram work on a machine
with no TeX at all. ``dvisvgm --no-fonts`` turns every glyph into a path, so
the SVG carries no font or image reference and renders the same on GitHub,
in Jupyter and on the documentation site.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Optional, Union

from ._scene import Arrow, Label, Marker, Polyline, Scene

_BODY_SLOT = "%%VAFT-BODY%%"
#: Everything between the TikZ source and the committed SVG that is not in the
#: source itself. It is folded into :attr:`Diagram.source_sha256`, so changing
#: the renderer marks every committed asset stale; bump it when the dvisvgm
#: flags or :func:`normalize_svg` change.
RENDER_RECIPE = "latex -> dvisvgm --no-fonts --bbox=papersize --optimize; normalize_svg v2"
_TOOLS = ("latex", "dvisvgm")
_MARKER_RADIUS = 0.06  # cm, filled O-point dot
_CROSS_HALF = 0.09  # cm, half arm of the X-point cross


class DiagramToolchainError(RuntimeError):
    """The TeX tools a render needs are not on ``PATH``."""


def _num(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _xy(point) -> str:
    return f"({_num(point[0])},{_num(point[1])})"


def _tikz_item(item) -> str:
    if isinstance(item, Polyline):
        path = " -- ".join(_xy(p) for p in item.points)
        if item.closed:
            path += " -- cycle"
        return f"\\draw[{item.style}] {path};"
    if isinstance(item, Marker):
        if item.kind == "o":
            return f"\\fill[{item.style}] {_xy(item.at)} circle[radius={_MARKER_RADIUS}];"
        d = _num(_CROSS_HALF)
        return (f"\\draw[{item.style}] {_xy(item.at)} +(-{d},-{d}) -- +({d},{d}) "
                f"+(-{d},{d}) -- +({d},-{d});")
    if isinstance(item, Arrow):
        style = item.style if not item.both else f"{item.style},<->"
        return f"\\draw[{style}] {_xy(item.start)} -- {_xy(item.end)};"
    if isinstance(item, Label):
        return f"\\node[{item.style},anchor={item.anchor}] at {_xy(item.at)} {{{item.text}}};"
    raise TypeError(f"not a scene item: {item!r}")


def template() -> str:
    """The authored LaTeX template every scene is set into."""
    return resources.files("vaft.diagram").joinpath("templates/standalone.tex").read_text(encoding="utf-8")


def tikz_document(scene: Scene) -> str:
    """The complete, compilable LaTeX document for ``scene``."""
    body = "\n".join(_tikz_item(item) for item in scene.items)
    text = template()
    slot = f"\n{_BODY_SLOT}\n"
    if text.count(slot) != 1:
        raise RuntimeError(f"diagram template must hold exactly one {_BODY_SLOT} line")
    return text.replace(slot, f"\n{body}\n")


def _require_toolchain() -> dict:
    found = {tool: shutil.which(tool) for tool in _TOOLS}
    missing = [tool for tool, path in found.items() if path is None]
    if missing:
        raise DiagramToolchainError(
            "TikZ rendering requires the VAFT diagram build toolchain: "
            f"{', '.join(missing)} not found on PATH. Install a TeX distribution "
            "with TikZ and dvisvgm (TeX Live: texlive-latex-extra plus "
            "texlive-binaries; macOS: MacTeX), or use the committed SVGs under "
            "docs/assets/diagrams/. The TikZ source is available without it as "
            "Diagram.tikz."
        )
    return found


def _run(cmd, cwd: Path, what: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        log = cwd / "diagram.log"
        detail = log.read_text(encoding="utf-8", errors="replace")[-3000:] if log.exists() else proc.stdout[-3000:]
        raise RuntimeError(f"{what} failed (exit {proc.returncode}):\n{detail}\n{proc.stderr[-2000:]}")


_GENERATOR_COMMENT = re.compile(r"<!--.*?-->\s*", re.S)
_DEFS = re.compile(r"<defs>(.*?)</defs>", re.S)
_DEF_ITEM = re.compile(r"<(path|use) ([^>]*?)/>")
_ATTR = re.compile(r"([\w:-]+)='([^']*)'")
_GLYPH_REF = re.compile(r"href='#([^']+)'")


def _canonical_glyphs(text: str) -> str:
    """Name and order glyph definitions by their content, not by font number.

    dvisvgm numbers fonts (``g7-40``, ``g10-90``) in an order that changes
    between runs of the same DVI, so the same picture comes out with its
    ``<defs>`` permuted and renamed. A definition is either an outline
    (``<path d=...>``) or a scaled reuse of another (``<use href transform>``);
    each is keyed by what it draws, identical ones are merged, and they are
    renamed ``g0, g1, ...`` in key order with every reference re-pointed. The
    drawing is unchanged.
    """
    match = _DEFS.search(text)
    if match is None:
        return text
    inner = match.group(1)
    if _DEF_ITEM.sub("", inner).strip():
        return text  # something other than glyph definitions; leave it alone
    items = {}
    for tag, attrs in _DEF_ITEM.findall(inner):
        attr = dict(_ATTR.findall(attrs))
        items[attr.pop("id")] = (tag, attr)

    keys = {}

    def key(ident):
        if ident not in keys:
            tag, attr = items[ident]
            if tag == "path":
                keys[ident] = ("0", attr.get("d", ""))
            else:
                target = attr.get("xlink:href", attr.get("href", ""))[1:]
                keys[ident] = ("1", key(target) if target in items else target, attr.get("transform", ""))
        return keys[ident]

    for ident in items:
        key(ident)
    ordered = sorted(set(keys.values()), key=repr)
    new_id = {k: f"g{i}" for i, k in enumerate(ordered)}
    rename = {ident: new_id[k] for ident, k in keys.items()}
    first = {}
    for ident, k in keys.items():
        first.setdefault(k, ident)

    lines = []
    for k in ordered:
        tag, attr = items[first[k]]
        if tag == "path":
            lines.append(f"<path id='{new_id[k]}' d='{attr.get('d', '')}'/>")
        else:
            href_name = "xlink:href" if "xlink:href" in attr else "href"
            target = attr.get(href_name, "")[1:]
            extra = "".join(f" {name}='{value}'" for name, value in sorted(attr.items()) if name != href_name)
            lines.append(f"<use id='{new_id[k]}' {href_name}='#{rename.get(target, target)}'{extra}/>")
    defs = "<defs>\n" + "\n".join(lines) + "\n</defs>"
    head, tail = text[:match.start()], text[match.end():]
    tail = _GLYPH_REF.sub(lambda m: f"href='#{rename.get(m.group(1), m.group(1))}'", tail)
    return head + defs + tail


def normalize_svg(text: str) -> str:
    """Strip what differs between runs and machines but not in the picture.

    Removes the generator comment (it names the dvisvgm version) and makes
    glyph naming independent of dvisvgm's run-to-run font numbering. The
    render tests pin the result by rendering twice.
    """
    text = _GENERATOR_COMMENT.sub("", text)
    text = _canonical_glyphs(text)
    return text.rstrip() + "\n"


def render_svg(document: str) -> str:
    """Compile a LaTeX document to a self-contained SVG string."""
    tools = _require_toolchain()
    with tempfile.TemporaryDirectory(prefix="vaft-diagram-") as tmp:
        cwd = Path(tmp)
        (cwd / "diagram.tex").write_text(document, encoding="utf-8")
        _run([tools["latex"], "-interaction=nonstopmode", "-halt-on-error", "diagram.tex"], cwd, "latex")
        # papersize: the box TikZ and standalone computed. --exact-bbox and
        # --bbox=min over-extend the 3-D view by ~90 pt of empty margin.
        _run([tools["dvisvgm"], "--no-fonts", "--bbox=papersize", "--optimize",
              "--output=diagram.svg", "diagram.dvi"], cwd, "dvisvgm")
        return normalize_svg((cwd / "diagram.svg").read_text(encoding="utf-8"))


def render_pdf(document: str, target: Path) -> None:
    """Compile a LaTeX document to PDF at ``target`` (an optional export)."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise DiagramToolchainError("PDF export requires pdflatex on PATH.")
    with tempfile.TemporaryDirectory(prefix="vaft-diagram-") as tmp:
        cwd = Path(tmp)
        # pdflatex cannot use the dvisvgm driver line; drop it for this export.
        (cwd / "diagram.tex").write_text(document.replace("\\def\\pgfsysdriver{pgfsys-dvisvgm.def}\n", ""),
                                            encoding="utf-8")
        _run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "diagram.tex"], cwd, "pdflatex")
        shutil.copyfile(cwd / "diagram.pdf", target)


class Diagram:
    """A built scientific diagram: TikZ source now, SVG on first request.

    Displays itself inline in Jupyter through ``_repr_svg_``.
    """

    def __init__(self, name: str, scene: Scene, *, model=None):
        self.name = name
        self.scene = scene
        #: The physical model every projection of this diagram was built from.
        self.model = model
        self._svg: Optional[str] = None

    @property
    def tikz(self) -> str:
        """The complete LaTeX/TikZ document. Needs no TeX installation."""
        return tikz_document(self.scene)

    @property
    def source_sha256(self) -> str:
        """Hash of :attr:`tikz` and :data:`RENDER_RECIPE`: what the committed-asset check compares."""
        return hashlib.sha256(f"{RENDER_RECIPE}\n{self.tikz}".encode("utf-8")).hexdigest()

    @property
    def svg(self) -> str:
        """The rendered, normalised SVG text. Renders on first access."""
        if self._svg is None:
            self._svg = render_svg(self.tikz)
        return self._svg

    def _repr_svg_(self) -> str:
        return self.svg

    def save(self, path: Union[str, Path]) -> Path:
        """Write the diagram as ``.svg`` (canonical), ``.tex`` or ``.pdf``."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            path.write_text(self.svg, encoding="utf-8", newline="\n")
        elif suffix == ".tex":
            path.write_text(self.tikz, encoding="utf-8", newline="\n")
        elif suffix == ".pdf":
            render_pdf(self.tikz, path)
        else:
            raise ValueError(f"cannot save a diagram as {suffix or 'a file with no suffix'!r}; use .svg, .tex or .pdf")
        return path

    def __repr__(self) -> str:
        return f"<vaft.diagram.Diagram {self.name!r}: {len(self.scene.items)} items>"
