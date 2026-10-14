"""Capture an Atlas V&V snapshot and recompute product-matched virial evidence.

Run on a host with the Atlas and its FileDB. Inputs are read only; --out must
be a new directory outside both inputs. The completed, validated snapshot is
published by a directory rename, so a failed calculation leaves no partial
snapshot. SSH and file transport are the caller's responsibility.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

from vaft.validation.kinetic_population import read_kinetic_population


REPOSITORY = Path(__file__).resolve().parents[2]
COPY_FILES = (
    "v1/state.csv", "v1/profiles.csv", "v1/MANIFEST.json",
    "v1/schema/state.schema.json", "v1/schema/profiles.schema.json",
    "equilibrium_quality/cohort_table.csv", "equilibrium_quality/constraint_points.csv",
    "equilibrium_quality/manifest.json", "equilibrium_quality/summary.json",
    "confinement/table.csv", "confinement/MANIFEST.json",
)

spec = importlib.util.spec_from_file_location(
    "virial_snapshot_export", Path(__file__).with_name("export_virial_identity.py"))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def _git(*args):
    return subprocess.check_output(["git", "-C", str(REPOSITORY), *args], text=True).strip()


def capture(atlas: Path, filedb: Path, out: Path) -> dict:
    """Copy byte-exact source tables, recompute virial, validate, and publish."""
    atlas, filedb, out = (Path(p).expanduser().resolve() for p in (atlas, filedb, out))
    if out.exists():
        raise FileExistsError(f"snapshot destination already exists: {out}")
    if any(out == source or source in out.parents for source in (atlas, filedb)):
        raise ValueError("snapshot destination must be outside the Atlas and FileDB")
    source_hashes = {relative: exporter._sha256(atlas / relative) for relative in COPY_FILES}
    source_manifest = json.loads((atlas / "v1/MANIFEST.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vv-capture-", dir=out.parent) as temporary:
        staged = Path(temporary) / "snapshot"
        for relative in COPY_FILES:
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(atlas / relative, destination)
        rows = exporter.export(staged / "v1/state.csv", filedb)
        virial_path = staged / "v1/virial_identity.csv"
        with virial_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=exporter.COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        population = read_kinetic_population(staged, strict=True)
        if len(rows) != len(population.population):
            raise ValueError("virial row count differs from the Atlas population")
        product_hashes = {row["efit_product"]: row["efit_product_sha256"] for row in rows}
        for relative, expected in product_hashes.items():
            if exporter._sha256(filedb / relative) != expected:
                raise ValueError(f"product changed during capture: {relative}")
        for relative, expected in source_hashes.items():
            if exporter._sha256(atlas / relative) != expected or exporter._sha256(staged / relative) != expected:
                raise ValueError(f"source changed during capture: {relative}")
        files = (*COPY_FILES, "v1/virial_identity.csv")
        counts = {}
        for relative in files:
            if relative.endswith(".csv"):
                with (staged / relative).open(newline="", encoding="utf-8") as stream:
                    counts[relative] = sum(1 for _ in csv.DictReader(stream))
        for name in ("equilibrium_quality", "confinement"):
            source_metadata = json.loads((staged / name / ("manifest.json" if name == "equilibrium_quality" else "MANIFEST.json")).read_text())
            table = f"{name}/{'cohort_table' if name == 'equilibrium_quality' else 'table'}.csv"
            if int(source_metadata["rows"]) != counts[table]:
                raise ValueError(f"manifest row count mismatch for {table}")
        export_script = "workflow/kinetic_state/export_virial_identity.py"
        capture_script = "workflow/kinetic_state/capture_verification_snapshot.py"
        revision = _git("rev-parse", "HEAD")
        manifest = {
            "snapshot_version": 2,
            "source_host": socket.gethostname(), "source_atlas": str(atlas), "source_filedb": str(filedb),
            "source_atlas_generated_at": source_manifest["generated_at"],
            "snapshot_captured_at": datetime.now(timezone.utc).isoformat(),
            "source_atlas_vaft_git": source_manifest["vaft_git"],
            "virial_compute_git": revision, "virial_compute_dirty": bool(_git("status", "--porcelain")),
            "virial_settings": {"alpha_method": "volume", "slice_tolerance_s": 0.0006},
            "virial_formula_sha256": {
                relative: exporter._sha256(REPOSITORY / relative)
                for relative in ("vaft/omas/process_wrapper.py", "vaft/formula/virial.py")},
            "virial_export_script": export_script,
            "virial_export_script_sha256": exporter._sha256(REPOSITORY / export_script),
            "snapshot_capture_script": capture_script,
            "snapshot_capture_script_sha256": exporter._sha256(REPOSITORY / capture_script),
            "products_sha256": product_hashes,
            "rows": counts, "sha256": {relative: exporter._sha256(staged / relative) for relative in files},
        }
        (staged / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(out)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--filedb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = capture(args.atlas, args.filedb, args.out)
    print(f"Captured {manifest['rows']['v1/state.csv']} states and {len(manifest['products_sha256'])} products at {args.out}")


if __name__ == "__main__":
    main()
