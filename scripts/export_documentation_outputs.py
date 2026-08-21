"""Execute the documentation notebook allowlist with read-only safeguards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ALLOWLIST = (
    "database_initialization_and_load.ipynb",
    "read_and_convert_data_structure.ipynb",
    "plotting_sample_using_vaft_plot_module.ipynb",
    "fluctuation_diagnostics_analysis.ipynb",
    "equilibrium_refinement_using_chease.ipynb",
    "kinetic_efit_end_to_end.ipynb",
    "confinement_time_scaling.ipynb",
    "initialize_external_fusion_codes.ipynb",
    "automated_pipeline_overview.ipynb",
)
EXPECTED_ARTIFACTS = (
    "first-result.png",
    "hsds-39915.txt",
    "imas-roundtrip.txt",
    "mirnov_spectrogram.png",
    "kinetic-profile.png",
    "equilibrium-inputs.png",
    "equilibrium-readiness.txt",
    "external-code-readiness.txt",
    "confinement-scaling.png",
    "confinement-scaling.txt",
    "pipeline-overview.png",
    "pipeline-overview.txt",
)

# The compact database notebook deliberately replaces vaft.database.save with a
# guard. Calls and lower-level write-enabled APIs remain forbidden.
FORBIDDEN = (
    re.compile(r"\b(?:save_ods|save_ids|save_shot_as|write_to_hdf5)\s*\("),
    re.compile(r"\bvaft\.database\.save\s*\("),
    re.compile(r"\b(?:hsload|h5py\.File)\s*\([^\n]*[\"'](?:w|a|r\+)"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def dependency_hash() -> str:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    return hashlib.sha256(freeze.encode()).hexdigest()


def validate_source(source: str, label: str) -> None:
    for pattern in FORBIDDEN:
        if pattern.search(source):
            raise RuntimeError(
                f"{label}: rejected remote or write-enabled API: {pattern.pattern}"
            )


def validate_notebook(path: Path) -> None:
    book = nbformat.read(path, as_version=4)
    if not book.metadata.get("vaft_docs", {}).get("read_only"):
        raise RuntimeError(f"{path.name}: missing vaft_docs.read_only metadata")
    source = "\n".join(cell.source for cell in book.cells if cell.cell_type == "code")
    validate_source(source, path.name)


def execute(path: Path, output: Path) -> None:
    validate_notebook(path)
    book = nbformat.read(path, as_version=4)
    client = NotebookClient(
        book,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    executed = output / "executed-notebooks"
    executed.mkdir(exist_ok=True)
    nbformat.write(book, executed / path.name)
    errors = [
        item
        for cell in book.cells
        for item in cell.get("outputs", [])
        if item.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError(f"{path.name}: contains {len(errors)} error output(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["VAFT_DOCS_READ_ONLY"] = "1"
    os.environ["VAFT_DOCS_OUTPUT_DIR"] = str(output)

    for filename in ALLOWLIST:
        path = NOTEBOOKS / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"executing {filename}", flush=True)
        execute(path, output)

    missing = [name for name in EXPECTED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing expected artifacts: {missing}")

    provenance: dict[str, Any] = {
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "vaft_version": subprocess.check_output(
            [sys.executable, "-c", "import vaft; print(vaft.__version__)"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "dependency_snapshot_sha256": dependency_hash(),
        "notebooks": [
            {"path": f"notebooks/{name}", "sha256": sha256(NOTEBOOKS / name)}
            for name in ALLOWLIST
        ],
        "artifacts": [
            {"path": name, "sha256": sha256(output / name)}
            for name in EXPECTED_ARTIFACTS
        ],
    }
    (output / "execution-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
