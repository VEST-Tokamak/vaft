"""Every stage product is written through `vaft.omas.save` (#813).

The container a stage product is stored in is declared once, in
`vaft.database.filedb.OMAS_PRODUCT_SUFFIX` / `OMAS_PRODUCT_SUFFIXES`, and
`FileDB.omas_product` puts it in the file name. `vaft.omas.save` then selects
the encoder from that suffix, so a writer that goes through it cannot disagree
with the resolver.

`omas.save_omas_json` and `omas.save_omas_h5` do not look at the suffix. Either
one aimed at a resolved product path writes whatever it likes under whatever
name it was given -- and the failure is not at the write. It surfaces later, at
a read, as `BadGzipFile` from `vaft.database._local.load_ods`, or as a replica
that never appears. When `OMAS_PRODUCT_SUFFIX` moved to `.json.gz`, two such
writers were already in the tree and would have written plain JSON into a
`.json.gz` name.

So the direct writers are listed here with the reason each is not a stage
product. Anything else fails until it is listed or routed through
`vaft.omas.save`. This is the same shape as `test_code_parameters_writers.py`,
and it has the same blind spot: it reads calls through the AST, so it sees a
name at a call site, not an alias assigned three lines earlier.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("vaft", "workflow")

#: `vaft.omas.save` does not appear here: it writes through `ODS.save` and
#: `omas.omas_h5.dict2hdf5`, never through the two named below, so it is not a
#: case this scan has to excuse.

DIRECT_WRITERS = {"save_omas_json", "save_omas_h5"}

#: path -> why this call does not write a canonical stage product.
ALLOWED = {
    "vaft/code/efit/kfile.py": (
        "writes the EFIT constraints into `work/`, which is a stage's scratch "
        "artifact rather than its product: it is not in the `omas_product` "
        "grammar and has no declared container."
    ),
    "workflow/automatic_pipeline_2_corrective_data_update/update_core_profile.py": (
        "a legacy script writing the shot-first `public/` tree, which is a "
        "read-only record of the pre-canonical pipeline (#89, #138)."
    ),
    "workflow/automatic_pipeline_2_corrective_data_update/"
    "update_thomson_scattering_and_core_profile.py": (
        "same legacy shot-first `public/` writer."
    ),
}


def _calls(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name in DIRECT_WRITERS:
            found.add(name)
    return found


def test_only_listed_files_write_an_omas_product_directly():
    offenders: dict[str, set[str]] = {}
    for area in SCANNED:
        for path in sorted((ROOT / area).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ALLOWED:
                continue
            calls = _calls(path)
            if calls:
                offenders[relative] = calls

    assert not offenders, (
        "These call omas's writers directly instead of `vaft.omas.save`, which "
        "is what selects the encoder from the product's declared container:\n"
        + "\n".join(f"  {path}: {sorted(names)}" for path, names in offenders.items())
        + "\nRoute the call through `vaft.omas.save`, or add the file to "
        "ALLOWED with the reason it is not a stage product."
    )


def test_every_allowed_file_still_exists_and_still_writes_one():
    """A stale entry silently widens the allowlist for a path reused later."""
    stale = {
        path
        for path in ALLOWED
        if not (ROOT / path).exists() or not _calls(ROOT / path)
    }
    assert not stale, (
        "ALLOWED names files that no longer call a direct writer: "
        f"{sorted(stale)}. Remove them so the list keeps meaning what it says."
    )
