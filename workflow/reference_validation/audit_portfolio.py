"""Audit the reference-sample portfolio against the canonical FileDB state (#1712).

The audit has two halves, because the server's production checkout is not the
VAFT version under review:

``scan``
    Standard library only, so it can be piped to any ``python3`` without
    importing VAFT.  It walks one or more FileDB roots read-only and prints, for
    every product directory of the requested shots, the lineage path, the
    output artifact (name, size, mtime) and the stage manifest's status, sha256
    and machine version.  Nothing is written::

        ssh -p 2222 user1@147.46.36.244 \\
            python3 - scan --root /srv/vest.filedb --root ~/runs/campaign/filedb \\
            --shot 39915 --shot 48224 \\
            < workflow/reference_validation/audit_portfolio.py > server_scan.json

``report``
    Runs locally against this checkout.  It reads the packaged samples
    (``vaft/data/samples/*``), the stage registry
    (:data:`vaft.database.sources.STAGE_REPLICATION`) and the diagnostic
    registry (``vest.yaml``), merges the server scan, and assigns every
    ``shot x source x IDS x occurrence`` row one of the coverage states #1712
    defines.  It writes a YAML coverage table and a Markdown summary.

The coverage state is a statement about the *server product*; the packaged
sample is reported beside it but never upgrades a row, because a packaged
artifact passing says nothing about the canonical state (it is exactly the
trap #1712 exists to close).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import os
import sys
from pathlib import Path

#: Product directories whose output is a placeholder rather than a product.
STUB_BYTES = 1024

#: Upstream stage(s) each stage's product is computed from.  A product older
#: than any of its inputs was built from a superseded input and is stale.
UPSTREAM = {
    "eddy": ("diagnostics",),
    "efit": ("eddy",),
    "chease": ("efit",),
    "mhd_linear": ("chease",),
    "gpec_ideal": ("chease",),
    "core_profiles": ("thomson", "ces"),
    "electron_efit": ("core_profiles", "eddy"),
    "kinetic_efit": ("core_profiles", "eddy"),
    "neoclassical": ("kinetic_efit",),
}

#: Topological order of UPSTREAM, so a stale input is classified first.
UPSTREAM_ORDER = (
    "diagnostics", "eddy", "efit", "chease", "mhd_linear", "gpec_ideal",
    "thomson", "ces", "core_profiles", "electron_efit", "kinetic_efit", "neoclassical",
)

#: IDS that describe the machine rather than the discharge: built from
#: vaft.machine_mapping (the ``static`` stage) and composed into every
#: product, so they never have a per-shot stage of their own.
MACHINE_IDS = ("dataset_description", "wall", "em_coupling")

#: Coverage vocabulary, verbatim from #1712.
STATES = (
    "available",
    "not-applicable",
    "input-unavailable",
    "server-unavailable",
    "validation-failed",
    "regeneration-required",
    "solver-unavailable",
    "deferred",
    "not-reference-qualified",
)


# --------------------------------------------------------------------------
# scan (standard library only)
# --------------------------------------------------------------------------


def _mtime(path: Path) -> str:
    stamp = _dt.datetime.fromtimestamp(path.stat().st_mtime, tz=_dt.timezone.utc)
    return stamp.isoformat(timespec="seconds")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _stub_comment(path: Path) -> str | None:
    """The equilibrium comment of a placeholder product, if it is one."""
    if path.stat().st_size > STUB_BYTES:
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return "unreadable placeholder"
    for ids in data.values():
        if isinstance(ids, dict):
            comment = ids.get("ids_properties", {}).get("comment")
            if comment:
                return str(comment)
    return "placeholder without comment"


def _scan_product(root: Path, product_dir: Path) -> dict:
    rel = product_dir.relative_to(root).parts  # omas/<stage>/[lineage...]/<shot>
    record: dict = {
        "root": str(root),
        "path": "/".join(rel),
        "stage": rel[1],
        "lineage": list(rel[2:-1]),
    }
    outputs = sorted(p for p in (product_dir / "output").glob("*") if p.is_file())
    if outputs:
        out = outputs[0]
        record["output"] = {"name": out.name, "size": out.stat().st_size, "mtime": _mtime(out)}
        stub = _stub_comment(out)
        if stub:
            record["output"]["stub"] = stub
    manifest = _read_json(product_dir / "metadata" / "manifest.json")
    if manifest is not None:
        record["manifest"] = {
            key: manifest.get(key)
            for key in ("status", "machine_version", "schema_version", "stage")
        }
        output = manifest.get("output")
        if isinstance(output, dict):
            record["manifest"]["sha256"] = output.get("sha256")
        for key in ("vaft_commit", "commit", "git_commit", "vaft_version"):
            if manifest.get(key):
                record["manifest"]["commit"] = manifest[key]
        run = manifest.get("run")
        if isinstance(run, dict):
            record["manifest"]["run_keys"] = sorted(run)[:20]
            for key in ("preset", "efit_preset", "commit", "vaft_commit"):
                if run.get(key):
                    record["manifest"]["run_" + key] = run[key]
    return record


def scan(roots: list[str], shots: list[str]) -> dict:
    products: list[dict] = []
    legacy: list[dict] = []
    for raw_root in roots:
        root = Path(os.path.expanduser(raw_root))
        omas = root / "omas"
        if omas.is_dir():
            for stage_dir in sorted(p for p in omas.iterdir() if p.is_dir()):
                # A shot directory sits at depth 1 (stage), 2 (family) or
                # 4 (family/refinement/product) below the stage.
                for depth in (1, 2, 3, 4):
                    pattern = "/".join(["*"] * (depth - 1) + ["{shot}"])
                    for shot in shots:
                        for match in stage_dir.glob(pattern.format(shot=shot)):
                            if match.is_dir() and (match / "output").is_dir():
                                record = _scan_product(root, match)
                                record["shot"] = int(shot)
                                products.append(record)
        legacy_root = root / "legacy"
        if legacy_root.is_dir():
            for diag_dir in sorted(p for p in legacy_root.iterdir() if p.is_dir()):
                for shot in shots:
                    shot_dir = diag_dir / shot
                    if shot_dir.is_dir():
                        files = sorted(p for p in shot_dir.rglob("*") if p.is_file())
                        legacy.append(
                            {
                                "root": str(root),
                                "diagnostic": diag_dir.name,
                                "shot": int(shot),
                                "files": [
                                    {"name": str(p.relative_to(shot_dir)), "size": p.stat().st_size}
                                    for p in files
                                ],
                            }
                        )
    return {
        "scanned_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        "roots": roots,
        "shots": [int(s) for s in shots],
        "products": products,
        "legacy": legacy,
    }


# --------------------------------------------------------------------------
# report (imports VAFT)
# --------------------------------------------------------------------------


def _packaged_ids(path: Path) -> dict[str, dict]:
    """Top-level IDS of a packaged OMAS JSON sample, with a little lineage."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        data = json.load(handle)
    out: dict[str, dict] = {}
    for name, ids in data.items():
        info: dict = {}
        if isinstance(ids, dict):
            code = ids.get("code", {})
            if isinstance(code, dict) and code.get("name"):
                info["code"] = code["name"]
            time = ids.get("time")
            if isinstance(time, list):
                info["n_time"] = len(time)
            slices = ids.get("time_slice")
            if isinstance(slices, list):
                info["n_time"] = len(slices)
        out[name] = info
    return out


def _packaged_inventory(samples_root: Path) -> dict[int, dict]:
    import yaml

    inventory: dict[int, dict] = {}
    for manifest_path in sorted(samples_root.glob("*/manifest.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text())
        shot = int(manifest["shot"])
        omas = manifest.get("representations", {}).get("omas", {})
        entry: dict = {
            "reference_id": manifest.get("reference_id"),
            "imas_dd_version": manifest.get("imas_dd_version"),
            "packaging": omas.get("package"),
            "ids": {},
            "equilibria": sorted(manifest.get("equilibria", {}).get("entries", {})),
        }
        sample_path = manifest_path.parent / omas.get("path", "omas.json.gz")
        if sample_path.is_file():
            entry["ids"] = _packaged_ids(sample_path)
        inventory[shot] = entry
    return inventory


def _stage_of(product: dict) -> str:
    return product["stage"]


def _newer(a: dict, b: dict) -> bool:
    return a.get("output", {}).get("mtime", "") > b.get("output", {}).get("mtime", "")


def _classify(product: dict, by_stage: dict[str, dict], replication) -> tuple[str, str]:
    stage = _stage_of(product)
    entry = replication.get(stage)
    if entry is not None and entry.deferred_to:
        return "deferred", f"stage deferred to {entry.deferred_to}"
    output = product.get("output")
    if output is None:
        return "server-unavailable", "product directory has no output"
    status = (product.get("manifest") or {}).get("status")
    if status not in (None, "success", "partial"):
        detail = f": {output['stub']}" if output.get("stub") else ""
        return "validation-failed", f"stage manifest status {status!r}{detail}"
    if status is None and output.get("stub"):
        return "validation-failed", f"no manifest and placeholder output: {output['stub']}"
    reasons = ["stage manifest status 'partial'"] if status == "partial" else []
    stale = []
    for upstream in UPSTREAM.get(stage, ()):
        source = by_stage.get(upstream)
        if source is not None and source.get("_state") == "regeneration-required":
            stale.append(f"upstream {upstream} is itself stale")
        elif source is not None and _newer(source, product):
            stale.append(f"upstream {upstream} ({source['output']['mtime'][:10]}) is newer")
    if stale:
        return "regeneration-required", "; ".join(stale + reasons)
    if not (product.get("manifest") or {}).get("commit"):
        reasons.append("provenance gap: manifest records no VAFT commit")
    return "available", "; ".join(reasons)


def report(scan_path: Path, out_yaml: Path, out_md: Path, repo: Path) -> None:
    import yaml

    from vaft.database.sources import STAGE_REPLICATION
    from vaft.machine_mapping.registry import load_diagnostic_registry

    data = json.loads(scan_path.read_text())
    packaged = _packaged_inventory(repo / "vaft" / "data" / "samples")
    registry = load_diagnostic_registry()

    shots = sorted(set(data["shots"]) | set(packaged))
    rows: list[dict] = []
    for shot in shots:
        products = [p for p in data["products"] if p["shot"] == shot]
        # Every FileDB root is reported separately; staleness is judged against
        # the upstream product in the *same* root, because a root is one
        # internally consistent chain (a campaign EFIT built on its own eddy
        # is not stale merely because production re-ran eddy later).
        best: dict[tuple, dict] = {}
        for product in products:
            best[(product["root"], product["stage"], tuple(product["lineage"]))] = product
        upstream_by_root: dict[str, dict[str, dict]] = {}
        for (root, stage, _), product in best.items():
            upstream_by_root.setdefault(root, {})[stage] = product
        order = list(UPSTREAM_ORDER)
        items = sorted(
            best.items(),
            key=lambda kv: (kv[0][0], order.index(kv[0][1]) if kv[0][1] in order else len(order), kv[0]),
        )
        for (root, stage, lineage), product in items:
            by_stage = upstream_by_root[root]
            entry = STAGE_REPLICATION.get(stage)
            ids_list = entry.ids if entry is not None else ()
            source = entry.source if entry is not None else None
            occurrence = entry.occurrence if entry is not None else 0
            state, reason = _classify(product, by_stage, STAGE_REPLICATION)
            product["_state"] = state
            for ids in ids_list or ("(stage without IDS)",):
                rows.append(
                    {
                        "shot": shot,
                        "ids": ids,
                        "source": source,
                        "occurrence": occurrence,
                        "stage": stage,
                        "lineage": "/".join(lineage) or None,
                        "filedb": product["root"],
                        "output": product.get("output"),
                        "manifest_status": (product.get("manifest") or {}).get("status"),
                        "sha256": (product.get("manifest") or {}).get("sha256"),
                        "machine_version": (product.get("manifest") or {}).get("machine_version"),
                        "state": state,
                        "reason": reason,
                        "packaged": ids in packaged.get(shot, {}).get("ids", {}),
                    }
                )
        for item in data.get("legacy", []):
            if item["shot"] != shot:
                continue
            rows.append(
                {
                    "shot": shot,
                    "ids": item["diagnostic"],
                    "source": "legacy",
                    "occurrence": 0,
                    "stage": "legacy",
                    "lineage": None,
                    "filedb": item["root"],
                    "output": {"files": len(item["files"]), "size": sum(f["size"] for f in item["files"])},
                    "state": "regeneration-required",
                    "reason": "raw legacy input; the OMAS product must be composed from it",
                    "packaged": item["diagnostic"] in packaged.get(shot, {}).get("ids", {}),
                }
            )
        # Packaged IDS with no server product at all.
        for ids, info in packaged.get(shot, {}).get("ids", {}).items():
            if any(r["shot"] == shot and r["ids"] == ids for r in rows):
                continue
            rows.append(
                {
                    "shot": shot,
                    "ids": ids,
                    "source": None,
                    "occurrence": 0,
                    "stage": None,
                    "lineage": None,
                    "filedb": None,
                    "output": None,
                    "state": "server-unavailable",
                    "reason": "packaged only; no canonical server product found",
                    "packaged": True,
                    "packaged_info": info,
                }
            )

    # Portfolio view: every IDS VAFT knows how to produce (diagnostic registry
    # plus stage registry), once, with the shots that exercise it.
    stage_ids = {ids for entry in STAGE_REPLICATION.values() for ids in entry.ids}
    specs: dict[str, dict] = {}
    for key, spec in sorted(registry.items()):
        ids = spec.get("ids")
        if isinstance(ids, str) and ids != "not_developed":
            specs.setdefault(ids, {"registry_keys": [], "mapping_status": set(), "lifecycle": set()})
            specs[ids]["registry_keys"].append(key)
            specs[ids]["mapping_status"].add(spec.get("mapping_status"))
            specs[ids]["lifecycle"].add(spec.get("lifecycle"))
    for ids in stage_ids:
        specs.setdefault(ids, {"registry_keys": [], "mapping_status": {"producer"}, "lifecycle": set()})
    portfolio = []
    for ids, spec in sorted(specs.items()):
        available = sorted({r["shot"] for r in rows if r["ids"] == ids and r["state"] == "available"})
        packaged_on = sorted(s for s, p in packaged.items() if ids in p["ids"])
        stages = sorted(name for name, entry in STAGE_REPLICATION.items() if ids in entry.ids)
        deferred = [STAGE_REPLICATION[n].deferred_to for n in stages if STAGE_REPLICATION[n].deferred_to]
        mapping = sorted(m for m in spec["mapping_status"] if m)
        if available:
            state, note = "available", ""
        elif ids in MACHINE_IDS:
            state, note = "available", "machine description; composed from vaft.machine_mapping"
        elif deferred and len(deferred) == len(stages):
            state, note = "deferred", f"stage deferred to {', '.join(deferred)}"
        elif mapping == ["not_implemented"]:
            state, note = "not-reference-qualified", "mapping not implemented"
        elif not stages:
            state, note = "deferred", "no FileDB stage produces it yet"
        else:
            state, note = "regeneration-required", "producer exists; no valid server product on an audited shot"
        portfolio.append(
            {
                "ids": ids,
                "registry_keys": spec["registry_keys"],
                "mapping_status": mapping,
                "lifecycle": sorted(x for x in spec["lifecycle"] if x),
                "stages": stages,
                "server_available_on": available,
                "packaged_on": packaged_on,
                "state": state,
                "note": note,
            }
        )

    document = {
        "schema_version": 1,
        "issue": 1712,
        "scan": {k: data[k] for k in ("scanned_at", "roots", "shots")},
        "states": list(STATES),
        "packaged": {
            shot: {k: v for k, v in entry.items() if k != "ids"} | {"ids": sorted(entry["ids"])}
            for shot, entry in packaged.items()
        },
        "rows": rows,
        "portfolio": portfolio,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(document, sort_keys=False, width=120))
    out_md.write_text(_markdown(document))


_ABBREV = {
    "available": "ok",
    "regeneration-required": "STALE",
    "validation-failed": "FAIL",
    "server-unavailable": "none",
    "deferred": "deferred",
}


def _root_label(root: str | None) -> str:
    if not root:
        return "packaged-only"
    if "campaign" in root:
        return "campaign"
    return "production"


def _matrix(rows: list[dict]) -> list[str]:
    """One cell per (shot, FileDB root) x stage: state and output date."""
    stages: list[str] = []
    cells: dict[tuple, dict[str, str]] = {}
    for r in rows:
        stage = r["stage"] or "packaged"
        if r["stage"] == "legacy":
            stage = f"legacy:{r['ids']}"
        if stage not in stages:
            stages.append(stage)
        key = (r["shot"], _root_label(r.get("filedb")))
        date = ((r.get("output") or {}).get("mtime") or "")[5:10]
        label = _ABBREV.get(r["state"], r["state"])
        cell = f"{label} {date}".strip() if r["stage"] not in (None, "legacy") else label
        if r["stage"] is None:
            cell = "pkg:" + r["ids"]
            cells.setdefault(key, {}).setdefault(stage, "")
            cells[key][stage] = (cells[key][stage] + " " + r["ids"]).strip()
            continue
        if r["stage"] == "legacy":
            cell = "raw"
        cells.setdefault(key, {})[stage] = cell
    order = [s for s in ("diagnostics", "eddy", "efit", "chease", "mhd_linear", "thomson", "ces",
                         "core_profiles", "electron_efit", "kinetic_efit", "camera_visible", "shotlog")
             if s in stages]
    order += sorted(s for s in stages if s not in order)
    lines = ["| shot | FileDB | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    for (shot, root) in sorted(cells):
        row = cells[(shot, root)]
        lines.append(f"| {shot} | {root} | " + " | ".join(row.get(s, "") for s in order) + " |")
    return lines


def _markdown(document: dict) -> str:
    lines = [
        "# Reference portfolio audit (#1712)",
        "",
        f"Server scan: {document['scan']['scanned_at']} over {', '.join(document['scan']['roots'])}.",
        "",
        "## Shot x stage matrix",
        "",
        *_matrix(document["rows"]),
        "",
        "## Per shot",
        "",
        "| shot | stage | lineage | IDS | FileDB | output date | state | reason | packaged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in document["rows"]:
        out = r.get("output") or {}
        date = (out.get("mtime") or "")[:10]
        root = (r.get("filedb") or "").replace("/home/user1", "~")
        lines.append(
            f"| {r['shot']} | {r['stage'] or '-'} | {r['lineage'] or '-'} | {r['ids']} | {root or '-'} "
            f"| {date or '-'} | {r['state']} | {r['reason'] or ''} | {'yes' if r['packaged'] else ''} |"
        )
    lines += [
        "",
        "## Portfolio (registry IDS)",
        "",
        "| IDS | mapping | stages | server available on | packaged on | state | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in document["portfolio"]:
        lines.append(
            f"| {p['ids']} | {', '.join(p['mapping_status']) or '-'} | {', '.join(p['stages']) or '-'} "
            f"| {', '.join(map(str, p['server_available_on'])) or '-'} "
            f"| {', '.join(map(str, p['packaged_on'])) or '-'} | {p['state']} | {p['note']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_scan = sub.add_parser("scan", help="read-only FileDB scan (stdlib only); JSON to stdout")
    p_scan.add_argument("--root", action="append", required=True)
    p_scan.add_argument("--shot", action="append", required=True)
    p_report = sub.add_parser("report", help="merge a scan with the packaged samples")
    p_report.add_argument("scan_json", type=Path)
    p_report.add_argument("--out-yaml", type=Path, default=Path("docs/_data/reference_coverage.yml"))
    p_report.add_argument("--out-md", type=Path, default=Path("reference_coverage.md"))
    p_report.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.command == "scan":
        json.dump(scan(args.root, args.shot), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    report(args.scan_json, args.out_yaml, args.out_md, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
