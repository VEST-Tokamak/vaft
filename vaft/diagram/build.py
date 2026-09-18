"""Regenerate or check the committed reference diagrams.

::

    python -m vaft.diagram.build            # re-render diagrams whose source changed
    python -m vaft.diagram.build --force    # re-render all of them
    python -m vaft.diagram.build --check    # verify; needs no TeX

Freshness is judged on the generated TikZ source, not on SVG bytes: two
``dvisvgm`` releases write different (equally correct) SVG for the same
picture, so a byte comparison would fail on every machine but the last one
to render. ``manifest.json`` records, per asset, the SHA-256 of the TikZ
document it was rendered from and of the SVG itself. ``--check`` rebuilds
the TikZ (pure Python), and fails when either hash disagrees -- a stale
asset, or a hand-edited one -- or when an asset is missing or orphaned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MANIFEST = "manifest.json"

#: The reference island: a 3/2 mode on a D-shaped (elongated, triangular) plasma.
REFERENCE_ISLAND = {
    "m": 3, "n": 2, "width": 0.16, "phase": 0.0,
    "r_s": 0.55, "elongation": 1.7, "triangularity": 0.4,
}

#: asset file name -> (builder name in vaft.diagram, keyword arguments)
CANONICAL: Dict[str, Tuple[str, dict]] = {
    **{
        f"magnetic_island_{projection}.svg": ("magnetic_island", {**REFERENCE_ISLAND, "projection": projection})
        for projection in ("poloidal", "top", "3d")
    },
    # stability and operational-space charts, at their documented defaults
    **{f"{name}.svg": (name, {}) for name in ("peeling_ballooning", "s_alpha_ballooning", "hugill", "troyon")},
}


def default_output() -> Path:
    """``docs/assets/diagrams`` of the source checkout this module runs from.

    An installed package has no ``docs/`` next to it; rather than write into
    ``site-packages``, that asks for an explicit ``--output``.
    """
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "docs").is_dir():
        raise SystemExit("not running from a VAFT source checkout: pass --output DIR")
    return root / "docs" / "assets" / "diagrams"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _call_text(builder: str, kwargs: dict) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"vaft.diagram.{builder}({args})"


def _diagram(builder: str, kwargs: dict):
    import vaft.diagram

    return getattr(vaft.diagram, builder)(**kwargs)


def _read_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("diagrams", {})


def check(out_dir: Optional[Path] = None) -> List[str]:
    """Every way the committed assets disagree with the current source."""
    out_dir = Path(out_dir or default_output())
    recorded = _read_manifest(out_dir)
    problems: List[str] = []
    for name, (builder, kwargs) in CANONICAL.items():
        entry = recorded.get(name)
        svg = out_dir / name
        if entry is None:
            problems.append(f"{name}: not in {MANIFEST}")
            continue
        if not svg.exists():
            problems.append(f"{name}: missing")
            continue
        if _diagram(builder, kwargs).source_sha256 != entry.get("source_sha256"):
            problems.append(f"{name}: stale -- its TikZ source or the render recipe changed; run python -m vaft.diagram.build")
        if _sha256(svg.read_bytes()) != entry.get("svg_sha256"):
            problems.append(f"{name}: the SVG does not match {MANIFEST} (edited by hand?)")
    for name in sorted(set(recorded) - set(CANONICAL)):
        problems.append(f"{name}: recorded in {MANIFEST} but no longer canonical")
    for svg in sorted(out_dir.glob("*.svg")):
        if svg.name not in CANONICAL:
            problems.append(f"{svg.name}: orphaned asset, not produced by the build")
    return problems


def build(out_dir: Optional[Path] = None, *, force: bool = False) -> List[str]:
    """Render stale (or, with ``force``, all) canonical diagrams; return what was written."""
    out_dir = Path(out_dir or default_output())
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = _read_manifest(out_dir)
    manifest: Dict[str, dict] = {}
    written: List[str] = []
    for name, (builder, kwargs) in CANONICAL.items():
        diagram = _diagram(builder, kwargs)
        svg_path = out_dir / name
        entry = recorded.get(name, {})
        fresh = (
            not force
            and svg_path.exists()
            and entry.get("source_sha256") == diagram.source_sha256
            and entry.get("svg_sha256") == _sha256(svg_path.read_bytes())
        )
        if not fresh:
            svg_path.write_text(diagram.svg, encoding="utf-8", newline="\n")
            written.append(name)
        manifest[name] = {
            "call": _call_text(builder, kwargs),
            "source_sha256": diagram.source_sha256,
            "svg_sha256": _sha256(svg_path.read_bytes()),
        }
    document = {
        "generator": "python -m vaft.diagram.build",
        "note": "source_sha256 hashes the render recipe and the generated TikZ document; --check compares it without TeX.",
        "diagrams": manifest,
    }
    (out_dir / MANIFEST).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vaft.diagram.build", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the committed assets; needs no TeX")
    parser.add_argument("--force", action="store_true", help="re-render every diagram, fresh or not")
    parser.add_argument("--output", type=Path, default=None, help="asset directory (default: docs/assets/diagrams)")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.output)
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(CANONICAL)} diagrams, {len(problems)} problems")
        return 1 if problems else 0
    written = build(args.output, force=args.force)
    print(f"rendered {len(written)} of {len(CANONICAL)} diagrams" + (f": {', '.join(written)}" if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
