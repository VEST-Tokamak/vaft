#!/usr/bin/env python3
"""Regenerate a repository-only native IMAS sample from a pinned Git ODS blob."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import yaml

import vaft
from vaft.data.reference_samples import (
    semantic_sample_view,
    sha256_file,
    verify_sample_artifacts,
)
from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
from vaft.omas import compare_ods


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Sample manifest schema_version must be 1")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _git_source_bytes(repository: Path, manifest: dict) -> bytes:
    source = manifest["source"]
    specification = f"{source['git_commit']}:{source['git_path']}"
    result = subprocess.run(
        ["git", "show", specification],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"Could not read pinned legacy sample {specification}: {detail}"
        )
    digest = hashlib.sha256(result.stdout).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(
            f"Pinned legacy source checksum mismatch for {specification}: "
            f"expected {source['sha256']}, got {digest}"
        )
    return result.stdout


def _normalized_source(repository: Path, manifest: dict):
    source_bytes = _git_source_bytes(repository, manifest)
    version = str(manifest["imas_dd_version"])
    with TemporaryDirectory(prefix="vaft-legacy-sample-") as temporary:
        source_path = Path(temporary) / "legacy.json"
        source_path.write_bytes(source_bytes)
        ods = vaft.omas.load(source_path, imas_version=version)

    shot = int(manifest["shot"])
    if int(ods.get("dataset_description.data_entry.pulse", -1)) != shot:
        raise ValueError(f"Pinned legacy source does not describe shot {shot}")

    mutual_aa = np.asarray(ods["em_coupling.mutual_active_active"])
    mutual_pa = np.asarray(ods["em_coupling.mutual_passive_active"])
    mutual_pp = np.asarray(ods["em_coupling.mutual_passive_passive"])
    n_passive, n_active = mutual_pa.shape
    if mutual_aa.shape != (n_active, n_active):
        raise ValueError(f"Invalid active coupling shape: {mutual_aa.shape}")
    if mutual_pp.shape != (n_passive, n_passive):
        raise ValueError(f"Invalid passive coupling shape: {mutual_pp.shape}")
    ods["em_coupling.active_coils"] = np.asarray(
        [f"#pf_active/coil({index})" for index in range(1, n_active + 1)]
    )
    ods["em_coupling.passive_loops"] = np.asarray(
        [f"#pf_passive/loop({index})" for index in range(1, n_passive + 1)]
    )

    if "equilibrium.code.parameters" in ods:
        del ods["equilibrium.code.parameters"]
    for ids_name, signal_path in (
        ("barometry", "barometry.gauge.0.pressure.time"),
        ("coils_non_axisymmetric", "coils_non_axisymmetric.coil.0.current.time"),
    ):
        if signal_path in ods and f"{ids_name}.time" not in ods:
            ods[f"{ids_name}.time"] = ods[signal_path]
            ods[f"{ids_name}.ids_properties.homogeneous_time"] = 1
    for index in range(len(ods.get("magnetics.b_field_pol_probe", []))):
        path = f"magnetics.b_field_pol_probe.{index}.poloidal_angle"
        if path in ods:
            ods[path] = POLOIDAL_ANGLE
    return ods


def _compare(reference, candidate, manifest: dict, label: str) -> None:
    result = compare_ods(
        semantic_sample_view(reference, manifest),
        semantic_sample_view(candidate, manifest),
        scope="union",
        reference_label="normalized legacy ODS",
        candidate_label=label,
    )
    if not result.passed:
        differences = [
            entry.path
            for entry in result.entries
            if entry.classification.value == "unintended_regression"
        ]
        raise ValueError(f"Semantic parity failed for {label}: {differences[:10]}")


def generate(repository: Path, manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    normalized = _normalized_source(repository, manifest)
    record = manifest["representations"]["imas"]
    target = manifest_path.parent / record["path"]
    vaft.imas.save(
        normalized,
        target,
        imas_version=str(manifest["imas_dd_version"]),
    )
    record["size"] = target.stat().st_size
    record["sha256"] = sha256_file(target)
    manifest["generation"]["materialized_leaves"] = len(normalized.flat())
    _write_manifest(manifest_path, manifest)


def verify(repository: Path, manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    verify_sample_artifacts(manifest_path.parent, manifest)
    normalized = _normalized_source(repository, manifest)
    version = str(manifest["imas_dd_version"])
    artifact = manifest_path.parent / manifest["representations"]["imas"]["path"]
    via_omas = vaft.omas.load(artifact, imas_version=version)
    with vaft.imas.load(artifact, imas_version=version) as handle:
        if handle.info.format != "imas_netcdf" or handle.info.converted:
            raise ValueError("Repository sample is not a native IMAS NetCDF artifact")
        via_imas = handle.to_omas()
    _compare(normalized, via_omas, manifest, "IMAS artifact -> OMAS adapter")
    _compare(normalized, via_imas, manifest, "IMAS artifact -> IMAS adapter")

    expected_times = np.asarray(manifest["acceptance"]["equilibrium_times"])
    np.testing.assert_allclose(via_omas["equilibrium.time"], expected_times)
    for index in range(len(via_omas.get("magnetics.b_field_pol_probe", []))):
        path = f"magnetics.b_field_pol_probe.{index}.poloidal_angle"
        if path in via_omas:
            np.testing.assert_allclose(via_omas[path], POLOIDAL_ANGLE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repository = args.repository.resolve()
    manifest_path = args.manifest.resolve()
    if not args.verify:
        generate(repository, manifest_path)
    verify(repository, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
