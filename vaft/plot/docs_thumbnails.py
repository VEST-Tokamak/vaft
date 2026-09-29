"""Render or check the committed plot thumbnails of the documentation site.

::

    python -m vaft.plot.docs_thumbnails            # re-render thumbnails whose recipe changed
    python -m vaft.plot.docs_thumbnails --force    # re-render all of them
    python -m vaft.plot.docs_thumbnails --check    # verify; renders nothing

Every plot the registry holds gets an entry in ``docs/assets/plots/manifest.json``.
A plot that one of the packaged samples (:func:`vaft.data.available_samples`) can
draw is rendered from the first sample that can, through its
``vaft.omas.plot_<name>`` adapter, and committed as ``<name>.png``; any other plot
records why it has no picture (``no_sample`` or ``failed``).  The documentation
build never runs matplotlib: ``/reference/plot/`` shows these committed files.

Freshness follows :mod:`vaft.diagram.build` in judging the *recipe* rather than
the bytes, because two matplotlib releases draw the same figure into different
PNGs.  Each rendered entry records the SHA-256 of

* ``sample_sha256`` -- the sample file it was drawn from;
* ``renderer_sha256`` -- the source of the renderer's module and the shared
  style and presentation modules;
* ``model_sha256`` -- the view model ``vaft.plot.extract(name, sample)`` returned;
* ``png_sha256`` -- the committed PNG itself.

:func:`check` treats the cases differently, by design:

problems (the check fails)
    a registered plot with no manifest entry, a rendered entry whose PNG is
    missing, a PNG or manifest entry for a name the registry does not hold, and
    a PNG that does not match its recorded hash (edited by hand);
warnings (the check passes)
    a thumbnail whose sample, renderer or -- with ``full=True``, which needs the
    samples -- view model changed since it was drawn, and a plot recorded as
    having no sample that some sample can now draw.

A stale thumbnail is labelled as such on the page instead of blocking the
build, so a renderer change does not have to re-commit every picture it touches.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import sys
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parents[1]
#: Shared modules every renderer draws through.
_SHARED_SOURCES = ("style.py", "presentation.py")
#: Thumbnail geometry: small enough to scan a gallery, large enough to read.
FIGSIZE = (5.0, 3.5)
DPI = 80
#: Palette size of the committed PNG; see :func:`_compact_png`.
PALETTE_COLORS = 128


def default_output() -> Path:
    """``docs/assets/plots`` of the source checkout this module runs from."""
    if not (_ROOT / "pyproject.toml").is_file() or not (_ROOT / "docs").is_dir():
        raise SystemExit("not running from a VAFT source checkout: pass --output DIR")
    return _ROOT / "docs" / "assets" / "plots"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_manifest(out_dir: Path) -> dict[str, dict]:
    path = out_dir / MANIFEST
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("plots", {})


def _write_manifest(out_dir: Path, entries: Mapping[str, dict]) -> None:
    document = {
        "generator": "python -m vaft.plot.docs_thumbnails",
        "note": ("Freshness is judged on the recipe (sample, renderer source, view model), "
                 "not on PNG bytes; see the module docstring."),
        "plots": {name: entries[name] for name in sorted(entries)},
    }
    (out_dir / MANIFEST).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# recipe hashes
# --------------------------------------------------------------------------


def renderer_sha256(spec) -> str:
    """The renderer's module source plus the shared style and presentation modules."""
    digest = hashlib.sha256()
    module = Path(inspect.getsourcefile(inspect.unwrap(spec.renderer))).resolve()
    for path in (module, *(_PACKAGE / name for name in _SHARED_SOURCES)):
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _feed(value: Any, digest) -> None:
    import numpy as np

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        digest.update(f"<{type(value).__name__}>".encode())
        for field in dataclasses.fields(value):
            digest.update(field.name.encode())
            _feed(getattr(value, field.name), digest)
    elif isinstance(value, np.ndarray):
        digest.update(f"<array {value.dtype} {value.shape}>".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=str):
            digest.update(repr(key).encode())
            _feed(value[key], digest)
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _feed(item, digest)
        digest.update(b"]")
    else:
        digest.update(repr(value).encode())


def model_sha256(model: Any) -> str:
    """A canonical hash of a view model: dataclasses, arrays, mappings and sequences."""
    digest = hashlib.sha256()
    _feed(model, digest)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet():
    """Rendering from sample data warns about the data; the thumbnail is still wanted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def sample_sources(names: Iterable[str]) -> dict[str, tuple[int, Any]]:
    """``name -> (shot, ods)``: the first packaged sample that can draw each plot."""
    import vaft

    wanted = set(names)
    found: dict[str, tuple[int, Any]] = {}
    for shot in vaft.data.available_samples():
        if not wanted - set(found):
            break
        with _quiet():
            ods = vaft.omas.load(vaft.data.sample(shot))
            offered = {record.name for record in vaft.omas.available_plots(ods) if record.available}
        for name in sorted(offered & wanted):
            found.setdefault(name, (shot, ods))
    return found


def _sample_sha256(shot: int) -> str:
    import vaft

    return _sha256(Path(vaft.data.sample(shot)).read_bytes())


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_one(spec, shot: int, ods, out_dir: Path) -> dict:
    """Draw one plot from ``ods`` into ``<name>.png`` and return its manifest entry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import vaft

    from .style import save_figure

    try:
        with _quiet():
            model = vaft.plot.extract(spec.name, ods)
            output = getattr(vaft.omas, f"plot_{spec.name}")(ods)
            figure = output[0] if isinstance(output, tuple) else output
            figure.set_size_inches(*FIGSIZE)
            buffer = io.BytesIO()
            save_figure(figure, buffer, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    finally:
        plt.close("all")
    path = out_dir / f"{spec.name}.png"
    path.write_bytes(_compact_png(buffer.getvalue()))
    return {
        "status": "rendered",
        "shot": shot,
        "call": f"vaft.omas.plot_{spec.name}(vaft.omas.load(vaft.data.sample({shot})))",
        "sample_sha256": _sample_sha256(shot),
        "renderer_sha256": renderer_sha256(spec),
        "model_sha256": model_sha256(model),
        "png_sha256": _sha256(path.read_bytes()),
    }


def _compact_png(data: bytes) -> bytes:
    """Quantize to a 128-colour palette without dithering: a third of the bytes.

    A line plot uses a handful of colours, and the site has a size ceiling
    (``docs/build.py``) that every committed picture counts against.  The
    quantization is deterministic, so the same figure gives the same file.
    """
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _fresh(entry: dict, spec, out_dir: Path) -> bool:
    path = out_dir / f"{spec.name}.png"
    return (
        entry.get("status") == "rendered"
        and path.is_file()
        and entry.get("png_sha256") == _sha256(path.read_bytes())
        and entry.get("renderer_sha256") == renderer_sha256(spec)
        and entry.get("sample_sha256") == _sample_sha256(entry["shot"])
    )


def build(out_dir: Path | None = None, *, force: bool = False, only: Iterable[str] | None = None) -> list[str]:
    """Render stale, missing (or, with ``force``, all) thumbnails; return what was written."""
    import vaft.plot  # noqa: F401 -- registers every renderer

    from . import registry

    out_dir = Path(out_dir or default_output())
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.name: spec for spec in registry.specs(status=None)}
    selected = set(specs) if only is None else set(only) & set(specs)
    entries = {name: entry for name, entry in _read_manifest(out_dir).items() if name in specs}
    sources = sample_sources(selected)
    written: list[str] = []
    for name in sorted(selected):
        spec = specs[name]
        if name not in sources:
            entries[name] = {"status": "no_sample",
                             "reason": "no packaged sample carries the data this plot needs"}
            (out_dir / f"{name}.png").unlink(missing_ok=True)
            continue
        shot, ods = sources[name]
        if not force and name in entries and entries[name].get("shot") == shot and _fresh(entries[name], spec, out_dir):
            continue
        try:
            entries[name] = render_one(spec, shot, ods, out_dir)
            written.append(name)
        except Exception as error:  # the sample offers the plot but cannot build it
            first_line = (str(error).strip().splitlines() or [type(error).__name__])[0]
            entries[name] = {"status": "failed", "shot": shot,
                             "reason": f"{type(error).__name__}: {first_line}"[:300]}
            (out_dir / f"{name}.png").unlink(missing_ok=True)
    for stray in out_dir.glob("*.png"):
        if stray.stem not in specs:
            stray.unlink()
    _write_manifest(out_dir, entries)
    return written


# --------------------------------------------------------------------------
# checking
# --------------------------------------------------------------------------


def stale_reason(entry: Mapping[str, Any], spec) -> str:
    """Why a rendered thumbnail may no longer show what the plot draws; ``""`` when current.

    Reads files only, so the documentation build can afford it for every plot.
    """
    if entry.get("status") != "rendered":
        return ""
    reasons = []
    try:
        if entry.get("sample_sha256") != _sample_sha256(entry["shot"]):
            reasons.append(f"sample {entry['shot']} changed")
    except Exception:  # the sample was removed
        reasons.append(f"sample {entry.get('shot')} is no longer packaged")
    if entry.get("renderer_sha256") != renderer_sha256(spec):
        reasons.append("renderer or style changed")
    return "; ".join(reasons)


def check(out_dir: Path | None = None, *, full: bool = False) -> tuple[list[str], list[str]]:
    """``(problems, warnings)``: problems fail the check, warnings only report staleness."""
    import vaft.plot  # noqa: F401

    from . import registry

    out_dir = Path(out_dir or default_output())
    specs = {spec.name: spec for spec in registry.specs(status=None)}
    recorded = _read_manifest(out_dir)
    problems: list[str] = []
    notes: list[str] = []

    for name in sorted(set(specs) - set(recorded)):
        problems.append(f"{name}: registered plot has no entry in {MANIFEST}; run python -m vaft.plot.docs_thumbnails")
    for name in sorted(set(recorded) - set(specs)):
        problems.append(f"{name}: recorded in {MANIFEST} but no longer a registered plot")
    for png in sorted(out_dir.glob("*.png")):
        if png.stem not in specs:
            problems.append(f"{png.name}: orphaned thumbnail, not a registered plot")
        elif recorded.get(png.stem, {}).get("status") != "rendered":
            problems.append(f"{png.name}: committed but {MANIFEST} records it as not rendered")

    for name in sorted(set(specs) & set(recorded)):
        entry = recorded[name]
        status = entry.get("status")
        if status not in ("rendered", "no_sample", "failed"):
            problems.append(f"{name}: unknown thumbnail status {status!r}")
            continue
        if status != "rendered":
            if not entry.get("reason"):
                problems.append(f"{name}: {status} entry records no reason")
            continue
        png = out_dir / f"{name}.png"
        if not png.is_file():
            problems.append(f"{name}: {MANIFEST} records a thumbnail but {png.name} is missing")
            continue
        if _sha256(png.read_bytes()) != entry.get("png_sha256"):
            problems.append(f"{png.name}: does not match {MANIFEST} (edited by hand?)")
        reason = stale_reason(entry, specs[name])
        if reason:
            notes.append(f"{name}: stale ({reason}); re-render with python -m vaft.plot.docs_thumbnails")

    if full:
        import vaft

        sources = sample_sources(specs)
        for name, entry in sorted(recorded.items()):
            if name not in specs:
                continue
            if entry.get("status") == "no_sample" and name in sources:
                notes.append(f"{name}: sample {sources[name][0]} can now draw it; re-render")
            if entry.get("status") == "rendered" and name in sources:
                try:
                    with _quiet():
                        current = model_sha256(vaft.plot.extract(name, sources[name][1]))
                except Exception as error:
                    notes.append(f"{name}: extraction now fails ({type(error).__name__})")
                    continue
                if sources[name][0] != entry.get("shot") or current != entry.get("model_sha256"):
                    notes.append(f"{name}: stale (the data it draws changed); re-render")
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.plot.docs_thumbnails",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the committed thumbnails; renders nothing")
    parser.add_argument("--force", action="store_true", help="re-render every thumbnail, fresh or not")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="restrict rendering to these plots")
    parser.add_argument("--output", type=Path, default=None, help="asset directory (default: docs/assets/plots)")
    arguments = parser.parse_args(argv)
    if arguments.check:
        problems, notes = check(arguments.output, full=True)
        for note in notes:
            print(f"warning: {note}")
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} problems, {len(notes)} warnings")
        return 1 if problems else 0
    written = build(arguments.output, force=arguments.force, only=arguments.only)
    print(f"rendered {len(written)} thumbnails" + (f": {', '.join(written)}" if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
