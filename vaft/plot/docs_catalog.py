"""The plot registry as a deterministic snapshot for the documentation site.

``python -m vaft.plot.docs_catalog --output docs/_data/plot_catalog.yml``
writes what ``/reference/plot/`` renders: every :class:`~vaft.plot.registry.PlotSpec`
the registry holds -- whatever its ``status`` -- with the identity
(``subject / view / quantity``) and developer block that
:func:`vaft.plot.discovery.capability_for` already derives, plus the adapter
it is reached through and the source line of its renderer.

It also lists ``entry_points``: the plotting functions users reach from
``vaft.plot`` that are not registry renderers -- every ``plot_*`` function in
``vaft.plot.__all__`` (ad-hoc and analytic figures such as
``plot_parameter_history``) and every name in ``vaft.plot._migration.LEGACY``
(the cross-shot statistics, still exported without a warning).  Deprecated,
relocated and removed names warn or raise and are not listed.

This module only *reads* :mod:`vaft.plot.registry` and :mod:`vaft.plot.discovery`;
nothing imports it, so it adds nothing to ``import vaft.plot``.  Like
``vaft.formula.catalog`` and ``vaft.process.catalog``, the snapshot is a pure
function of the source tree unless ``--provenance-commit``/``--provenance-ref``
are passed, and it records a checksum of every source file it read so
``docs/build.py`` can prove it was generated from the tree being documented.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path

SCHEMA_VERSION = 1
_GENERATOR = "python -m vaft.plot.docs_catalog --output docs/_data/plot_catalog.yml"
_PACKAGE = Path(__file__).resolve().parent
_ROOT = _PACKAGE.parents[1]


def _source_files() -> list[Path]:
    """Every file of the package: registration, backends and models all shape the snapshot."""
    files = [path for path in _PACKAGE.rglob("*.py") if "__pycache__" not in path.parts]
    return sorted([*files, _ROOT / "vaft" / "_docstring.py"])  # _docstring decides every source span


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(_ROOT).as_posix()


def _source_of(function) -> dict:
    from vaft._docstring import source_span

    return source_span(function, _ROOT)


def _row(spec) -> dict:
    from .discovery import capability_for

    capability = capability_for(spec)
    renderer = inspect.unwrap(spec.renderer)
    return {
        "id": spec.name,
        "name": spec.name,
        "status": spec.status,
        "subject": spec.subject,
        "view": spec.view,
        "quantity": spec.quantity,
        "domain": spec.domain,
        "description": spec.description,
        "model": capability.model,
        "adapter": f"vaft.omas.plot_{spec.stem}",
        "renderer": f"{renderer.__module__}.{renderer.__qualname__}",
        "ids": list(spec.ids),
        "required_paths": list(spec.required_paths),
        "optional_paths": list(spec.optional_paths),
        "backends": list(capability.backends),
        "overlays": list(capability.overlays),
        "source": _source_of(spec.renderer),
    }


def _plain_signature(function) -> str:
    signature = inspect.signature(function)
    return str(signature.replace(
        parameters=[p.replace(annotation=inspect.Parameter.empty) for p in signature.parameters.values()],
        return_annotation=inspect.Signature.empty,
    ))


def entry_point_names() -> list[tuple[str, str]]:
    """``(name, status)`` of the plotting functions that are not registry renderers."""
    import vaft.plot as plot

    from . import _migration, registry

    renderers = {id(spec.renderer) for spec in registry.specs(status=None)}
    names = [
        (name, "support")
        for name in sorted(plot.__all__)
        if name.startswith("plot_")
        and inspect.isfunction(getattr(plot, name))
        and id(getattr(plot, name)) not in renderers
    ]
    names.extend((name, "legacy") for name in sorted(_migration.LEGACY))
    return names


def _entry_point(name: str, status: str) -> dict:
    import vaft.plot as plot

    function = inspect.unwrap(getattr(plot, name))
    summary = (inspect.getdoc(function) or "").strip().split("\n\n")[0].replace("\n", " ")
    return {
        "id": name,
        "name": name,
        "status": status,
        "module": function.__module__,
        "summary": summary,
        "signature": _plain_signature(function),
        "source": _source_of(function),
    }


THUMBNAILS = Path("docs") / "assets" / "plots"


def _thumbnails(specs) -> tuple[dict[str, dict], list[Path]]:
    """``name -> thumbnail row`` from the committed manifest, and the files that decide it.

    Reads files only: whether a thumbnail is stale is judged from the sample and
    renderer hashes :mod:`vaft.plot.docs_thumbnails` recorded, so the build loads no
    sample and runs no matplotlib.
    """
    import json

    import vaft

    from . import docs_thumbnails

    manifest = _ROOT / THUMBNAILS / docs_thumbnails.MANIFEST
    if not manifest.is_file():
        return {}, []
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("plots", {})
    rows: dict[str, dict] = {}
    sources = [manifest]
    for spec in specs:
        entry = recorded.get(spec.name)
        if entry is None:
            rows[spec.name] = {"status": "missing", "png": "", "png_sha256": "", "shot": 0, "stale": "",
                               "reason": "not in the thumbnail manifest"}
            continue
        row = {
            "status": entry["status"],
            "png": "",
            "png_sha256": entry.get("png_sha256", ""),
            "shot": entry.get("shot", 0),
            "stale": docs_thumbnails.stale_reason(entry, spec),
            "reason": entry.get("reason", ""),
        }
        if entry["status"] == "rendered":
            row["png"] = (THUMBNAILS.relative_to("docs") / f"{spec.name}.png").as_posix()
            try:
                sources.append(Path(vaft.data.sample(entry["shot"])).resolve())
            except Exception:  # a removed sample is reported as stale above
                pass
        rows[spec.name] = row
    return rows, sources


def documentation_snapshot(provenance: Mapping[str, str] | None = None) -> dict:
    """Every registered plot, ordered subject (taxonomy order) -> view (``VIEWS`` order)."""
    import vaft.plot  # noqa: F401 -- importing the package registers every renderer

    from . import registry, taxonomy

    specs = registry.specs(status=None)
    subject_order = {name: index for index, name in enumerate(taxonomy.SUBJECTS)}
    view_order = {name: index for index, name in enumerate(registry.VIEWS)}
    specs = sorted(
        specs,
        key=lambda spec: (
            subject_order.get(spec.subject, len(subject_order)),
            view_order.get(spec.view, len(view_order)),
            spec.quantity,
            spec.name,
        ),
    )

    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.subject] = counts.get(spec.subject, 0) + 1

    thumbnails, thumbnail_sources = _thumbnails(specs)

    snapshot: dict = {
        "schema_version": SCHEMA_VERSION,
        "generator": _GENERATOR,
        "source": [
            {"path": _relative(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted({*_source_files(), *thumbnail_sources}, key=_relative)
        ],
        "views": list(registry.VIEWS),
        "subjects": [
            {
                "name": subject.name,
                "kind": subject.kind,
                "aliases": list(subject.aliases),
                "count": counts[subject.name],
            }
            for subject in taxonomy.SUBJECTS.values()
            if subject.name in counts
        ],
        "plots": [{**_row(spec), **({"thumbnail": thumbnails[spec.name]} if thumbnails else {})} for spec in specs],
        "entry_points": [_entry_point(name, status) for name, status in entry_point_names()],
    }
    if provenance:
        snapshot["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return snapshot


def export_documentation_snapshot(output: str | Path, provenance: Mapping[str, str] | None = None) -> Path:
    """Write the YAML snapshot and return its path."""
    import yaml

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            documentation_snapshot(provenance),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the vaft.plot registry for the documentation site.")
    parser.add_argument("--output", required=True, help="YAML destination for the snapshot")
    parser.add_argument("--provenance-commit", help="Commit the source tree was taken from, recorded in the snapshot")
    parser.add_argument("--provenance-ref", help="Ref that commit was resolved from, recorded in the snapshot")
    arguments = parser.parse_args(argv)
    provenance = {
        key: value
        for key, value in (("commit", arguments.provenance_commit), ("ref", arguments.provenance_ref))
        if value
    }
    export_documentation_snapshot(arguments.output, provenance or None)


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    main()
