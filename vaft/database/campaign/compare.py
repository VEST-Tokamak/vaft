"""Are two runs' products the same science? Stage ODS, solver files, status records.

For each shot, the stage products of two FileDB roots are compared:

* stage ODS (``diagnostics``, ``eddy``, ``efit``; ``.json.gz``) leaf by leaf;
* stage manifests leaf by leaf, ignoring provenance (when, where, which code);
* every file of the shot's EFIT directory (k-files, g-files, status) byte for byte,
  logs aside.

A difference in provenance only -- a different pin, host or timestamp -- is
``provenance``; anything else is ``differs`` and is listed. Paths come from the
workflow's own ``paths.py`` (:class:`PipelinePaths`), so the comparison follows
the same layout the products were written in.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

__all__ = ["DEFAULT_IGNORED_KEYS", "compare_products", "diff_json", "summarize"]

PathLike = Union[str, Path]

#: Keys whose values record when, where or by which code a product was made.
DEFAULT_IGNORED_KEYS = frozenset({
    "creation_date", "created", "created_at", "generated", "generated_at", "timestamp", "time_put",
    "started", "finished", "start_time", "end_time", "elapsed_s", "duration_s", "wall_s", "runtime_s",
    "host", "hostname", "user", "pid", "code_version", "version", "git_sha", "git_commit", "commit", "sha",
    "command", "argv", "cmdline", "config_path", "configfile", "provenance", "repository", "pin",
})

_ODS = ("diagnostics_ods", "eddy_ods", "efit_ods")
_MANIFESTS = ("diagnostics_manifest", "eddy_manifest", "efit_manifest")


def _load(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def diff_json(a: Any, b: Any, *, ignored: Iterable[str] = DEFAULT_IGNORED_KEYS, limit: int = 20) -> dict[str, Any]:
    """Leaf paths where ``a`` and ``b`` differ, split into provenance and content."""
    ignored = frozenset(ignored)
    content: list[str] = []
    provenance: list[str] = []

    def walk(x: Any, y: Any, path: tuple, in_provenance: bool) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y), key=str):
                walk(x.get(key, _MISSING), y.get(key, _MISSING), path + (str(key),), in_provenance or key in ignored)
            return
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and any(
            isinstance(item, (dict, list)) for item in x
        ):
            for index, (p, q) in enumerate(zip(x, y)):
                walk(p, q, path + (str(index),), in_provenance)
            return
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            if all(_equal(p, q) for p, q in zip(x, y)):
                return
        elif _equal(x, y):
            return
        (provenance if in_provenance else content).append(".".join(path))

    walk(a, b, (), False)
    return {"content": content[:limit], "content_count": len(content),
            "provenance": provenance[:limit], "provenance_count": len(provenance)}


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing)

    __hash__ = None  # type: ignore[assignment]


_MISSING = _Missing()


def _paths(root: PathLike, workflow_dir: PathLike, layout: str = "filedb"):
    from ..worker.status import load_pipeline_paths_module

    module = load_pipeline_paths_module(Path(workflow_dir))
    return module.PipelinePaths.from_config({"base_dir": str(root), "layout": layout})


def _real(path: str, root: Path) -> Path:
    # PipelinePaths built over ``root`` already resolve inside it.
    return Path(path)


def compare_products(
    new_root: PathLike,
    ref_root: PathLike,
    shots: Sequence[int],
    *,
    workflow_dir: PathLike,
    ref_roots: Optional[Sequence[PathLike]] = None,
    ignored: Iterable[str] = DEFAULT_IGNORED_KEYS,
) -> dict[int, dict[str, Any]]:
    """Per shot, per product: ``same``, ``provenance``, ``differs``, ``missing_new``/``missing_ref``.

    ``ref_roots`` lists several reference roots (the lanes of a run); each shot's
    reference is the first root that has its diagnostics product.
    """
    new_paths = _paths(new_root, workflow_dir)
    references = [Path(root) for root in (ref_roots or [ref_root])]
    ref_paths = [_paths(root, workflow_dir) for root in references]
    report: dict[int, dict[str, Any]] = {}
    for shot in shots:
        ref = next((p for p in ref_paths if Path(p.diagnostics_manifest(shot)).exists()), ref_paths[0])
        entry: dict[str, Any] = {}
        for name in _ODS + _MANIFESTS:
            entry[name] = _compare_file(Path(getattr(new_paths, name)(shot)), Path(getattr(ref, name)(shot)), ignored)
        entry["efit_files"] = _compare_tree(
            Path(new_paths.efit_status(shot)).parent.parent, Path(ref.efit_status(shot)).parent.parent
        )
        report[int(shot)] = entry
    return report


def _compare_file(new: Path, ref: Path, ignored: Iterable[str]) -> dict[str, Any]:
    if not new.exists() or not ref.exists():
        if not new.exists() and not ref.exists():
            return {"verdict": "absent_both"}
        return {"verdict": "missing_new" if not new.exists() else "missing_ref"}
    if new.read_bytes() == ref.read_bytes():
        return {"verdict": "same"}
    diff = diff_json(_load(new), _load(ref), ignored=ignored)
    verdict = "differs" if diff["content_count"] else ("provenance" if diff["provenance_count"] else "same")
    return {"verdict": verdict, **diff}


def _compare_tree(new: Path, ref: Path) -> dict[str, Any]:
    def files(root: Path) -> dict[str, Path]:
        if not root.is_dir():
            return {}
        return {
            str(p.relative_to(root)): p
            for p in root.rglob("*")
            if p.is_file() and "log" not in p.relative_to(root).parts[:1] and not p.name.endswith(".log")
        }

    a, b = files(new), files(ref)
    if not a and not b:
        return {"verdict": "absent_both"}
    only_new, only_ref = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    differing = sorted(name for name in set(a) & set(b) if a[name].read_bytes() != b[name].read_bytes())
    # The artifact manifest records the run's paths and timing; judge it as JSON.
    content = [name for name in differing if not name.endswith(".json")]
    json_diffs = [name for name in differing if name.endswith(".json")
                  and diff_json(_load(a[name]), _load(b[name]))["content_count"]]
    verdict = "same"
    if only_new or only_ref or content or json_diffs:
        verdict = "differs"
    elif differing:
        verdict = "provenance"
    return {"verdict": verdict, "files": len(set(a) | set(b)), "only_new": only_new[:20],
            "only_ref": only_ref[:20], "differing": (content + json_diffs)[:20]}


def summarize(report: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Verdict counts per product over every shot."""
    summary: dict[str, dict[str, int]] = {}
    for entry in report.values():
        for name, result in entry.items():
            counts = summary.setdefault(name, {})
            counts[result["verdict"]] = counts.get(result["verdict"], 0) + 1
    return summary
