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
* ``renderer_sha256`` -- the drawing code: every module of
  ``vaft.plot.renderers`` (composites draw through each other's bodies) and
  ``style``, ``presentation``, ``models`` and ``intent``, with line endings
  normalised;
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

Not hashed: the extraction layer (``vaft.omas`` adapters, ``vaft.plot.backend``)
and packaged geometry data.  A change there shows up as a changed view model,
which only ``--check`` (``full=True``) and a re-render compare.  Rendering runs
under matplotlib's own defaults, not the machine's ``matplotlibrc``; the
manifest records the matplotlib, numpy and Pillow versions it was drawn with.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
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
#: Modules every renderer draws through, besides the ``renderers`` package:
#: styling, presentation, the view models, and the theme ``intent`` supplies.
_SHARED_SOURCES = ("style.py", "presentation.py", "models.py", "intent.py")
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


def _toolchain() -> dict[str, str]:
    """Versions that decide the PNG bytes (not the recipe): recorded, never compared."""
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ("matplotlib", "numpy", "pillow"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - all three ship with vaft
            found[package] = "unknown"
    return found


def _write_manifest(out_dir: Path, entries: Mapping[str, dict], toolchain: Mapping[str, str] | None = None) -> None:
    previous = {}
    if (out_dir / MANIFEST).is_file():
        previous = json.loads((out_dir / MANIFEST).read_text(encoding="utf-8")).get("toolchain", {})
    document = {
        "generator": "python -m vaft.plot.docs_thumbnails",
        "toolchain": dict(toolchain or previous),
        "note": ("Freshness is judged on the recipe (sample, renderer source, view model), "
                 "not on PNG bytes; see the module docstring."),
        "plots": {name: entries[name] for name in sorted(entries)},
    }
    (out_dir / MANIFEST).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# recipe hashes
# --------------------------------------------------------------------------


def _drawing_sources(spec) -> list[Path]:
    """The source files whose change can alter the picture without changing the view model.

    Every module of ``vaft.plot.renderers`` rather than only the renderer's own:
    composite renderers draw through the others' shared bodies
    (``render_line_series``, ``draw_geometry_layer``, ...).
    """
    module = Path(inspect.getsourcefile(inspect.unwrap(spec.renderer))).resolve()
    paths = {module, *(_PACKAGE / name for name in _SHARED_SOURCES)}
    paths.update(path for path in (_PACKAGE / "renderers").glob("*.py"))
    return sorted(paths, key=lambda path: path.relative_to(_ROOT).as_posix())


def _text_bytes(path: Path) -> bytes:
    """Source bytes with line endings normalised, so a CRLF checkout hashes the same."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def renderer_sha256(spec) -> str:
    """The drawing code of ``spec``: every renderer module and the shared ones."""
    digest = hashlib.sha256()
    for path in _drawing_sources(spec):
        digest.update(path.relative_to(_ROOT).as_posix().encode())
        digest.update(_text_bytes(path))
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


def sample_candidates(names: Iterable[str]) -> dict[str, list[tuple[int, Any]]]:
    """``name -> [(shot, ods), ...]``: every packaged sample that offers each plot, in sample order."""
    import vaft

    wanted = set(names)
    found: dict[str, list[tuple[int, Any]]] = {}
    for shot in vaft.data.available_samples():
        with _quiet():
            ods = vaft.omas.load(vaft.data.sample(shot))
            offered = {record.name for record in vaft.omas.available_plots(ods) if record.available}
        for name in sorted(offered & wanted):
            found.setdefault(name, []).append((shot, ods))
    return found


def sample_sources(names: Iterable[str]) -> dict[str, tuple[int, Any]]:
    """``name -> (shot, ods)``: the first packaged sample that offers each plot."""
    return {name: candidates[0] for name, candidates in sample_candidates(names).items()}


@functools.lru_cache(maxsize=None)
def _sample_file_sha256(path: str, size: int, mtime_ns: int) -> str:
    return _sha256(Path(path).read_bytes())


def _sample_sha256(shot: int) -> str:
    """Hash of a packaged sample, computed once per file state (samples are tens of MB)."""
    import vaft

    path = Path(vaft.data.sample(shot))
    stat = path.stat()
    return _sample_file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


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

    # Matplotlib's own defaults, not whatever matplotlibrc the machine has.
    defaults = {key: value for key, value in matplotlib.rcParamsDefault.items() if key != "backend"}
    try:
        with _quiet(), matplotlib.rc_context(defaults):
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


def _fresh(entry: dict, spec, ods, out_dir: Path) -> bool:
    """Nothing the picture depends on changed: sample, drawing code or view model."""
    import vaft

    path = out_dir / f"{spec.name}.png"
    if not (
        entry.get("status") == "rendered"
        and path.is_file()
        and entry.get("png_sha256") == _sha256(path.read_bytes())
        and entry.get("renderer_sha256") == renderer_sha256(spec)
        and entry.get("sample_sha256") == _sample_sha256(entry["shot"])
    ):
        return False
    try:
        with _quiet():
            return entry.get("model_sha256") == model_sha256(vaft.plot.extract(spec.name, ods))
    except Exception:
        return False


def build(out_dir: Path | None = None, *, force: bool = False, only: Iterable[str] | None = None) -> list[str]:
    """Render stale, missing (or, with ``force``, all) thumbnails; return what was written."""
    import vaft.plot  # noqa: F401 -- registers every renderer

    from . import registry

    out_dir = Path(out_dir or default_output())
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.name: spec for spec in registry.specs(status=None)}
    selected = set(specs) if only is None else set(only) & set(specs)
    entries = {name: entry for name, entry in _read_manifest(out_dir).items() if name in specs}
    candidates = sample_candidates(selected)
    written: list[str] = []
    for name in sorted(selected):
        spec = specs[name]
        if name not in candidates:
            entries[name] = {"status": "no_sample",
                             "reason": "no packaged sample carries the data this plot needs"}
            (out_dir / f"{name}.png").unlink(missing_ok=True)
            continue
        current = entries.get(name, {})
        fresh = [(shot, ods) for shot, ods in candidates[name] if shot == current.get("shot")]
        if not force and fresh and _fresh(current, spec, fresh[0][1], out_dir):
            continue
        errors = []
        for shot, ods in candidates[name]:  # the first sample that offers it may still lack a piece
            try:
                entries[name] = render_one(spec, shot, ods, out_dir)
                written.append(name)
                break
            except Exception as error:
                first_line = (str(error).strip().splitlines() or [""])[0]
                errors.append(f"shot {shot}: {type(error).__name__}: {first_line}")
        else:
            entries[name] = {"status": "failed", "reason": "; ".join(errors)[:400]}
            (out_dir / f"{name}.png").unlink(missing_ok=True)
    for stray in out_dir.glob("*.png"):
        if stray.stem not in specs:
            stray.unlink()
    _write_manifest(out_dir, entries, _toolchain() if written else None)
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
        incomplete = [key for key in ("shot", "sample_sha256", "renderer_sha256", "model_sha256", "png_sha256")
                      if not entry.get(key)]
        if incomplete:
            problems.append(f"{name}: rendered entry lacks {', '.join(incomplete)}")
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
            if entry.get("status") in ("no_sample", "failed") and name in sources:
                try:
                    with _quiet():
                        vaft.plot.extract(name, sources[name][1])
                    notes.append(f"{name}: recorded as {entry['status']} but sample {sources[name][0]} "
                                 f"can now build it; re-render")
                except Exception:
                    pass
            if entry.get("status") == "rendered" and name in sources:
                try:
                    with _quiet():
                        current = model_sha256(vaft.plot.extract(name, sources[name][1]))
                except Exception as error:
                    notes.append(f"{name}: extraction now fails ({type(error).__name__})")
                    continue
                if sources[name][0] != entry.get("shot") or current != entry.get("model_sha256"):
                    notes.append(f"{name}: stale (the data it draws changed); re-render with "
                                 f"python -m vaft.plot.docs_thumbnails --only {name}")
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
