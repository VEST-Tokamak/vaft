#!/usr/bin/env python3
"""Regenerate a repository-only native IMAS sample from one pipeline ODS."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

import vaft
from vaft.data.reference_samples import (
    semantic_sample_view,
    sha256_file,
    verify_sample_artifacts,
)
from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
from vaft.machine_mapping.utils import get_path, path_exists, set_path
from vaft.omas import compare_ods


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Sample manifest schema_version must be 1")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _drop_empty_probe_signal(ods, base: str) -> None:
    """Drop IMAS-default placeholders for a missing heterogeneous waveform."""
    time_path = f"{base}.time"
    data_path = f"{base}.data"
    time = np.asarray(get_path(ods, time_path)) if path_exists(ods, time_path) else np.array([])
    data = np.asarray(get_path(ods, data_path)) if path_exists(ods, data_path) else np.array([])
    if time.ndim == 0 or data.ndim == 0 or time.size == 0 or data.size == 0:
        if base in ods:
            del ods[base]


def normalized_pipeline_ods(canonical_source: Path, manifest: dict):
    """Return the pipeline ODS in the portable DD 3.41 representation."""
    version = str(manifest["imas_dd_version"])
    ods = vaft.omas.load(canonical_source, imas_version=version)
    shot = int(manifest["shot"])
    if int(ods.get("dataset_description.data_entry.pulse", -1)) != shot:
        raise ValueError(f"Pipeline source does not describe shot {shot}")

    # EFIT's parser cache is not an IMAS DD leaf. Keep the complete generated
    # source product, but omit this serializer-only cache from the native view.
    if "equilibrium.code.parameters" in ods:
        del ods["equilibrium.code.parameters"]

    # DD 3.41 accepts heterogeneous Mirnov coordinates. OMAS exposes missing
    # optional channels as scalar defaults, which cannot be written as 1-D
    # signal times. Remove only those placeholders; real waveforms retain their
    # own native-rate coordinates.
    for index in range(len(ods.get("magnetics.b_field_pol_probe", []))):
        _drop_empty_probe_signal(ods, f"magnetics.b_field_pol_probe.{index}.voltage")
        _drop_empty_probe_signal(ods, f"magnetics.b_field_pol_probe.{index}.field")
        set_path(ods, f"magnetics.b_field_pol_probe.{index}.poloidal_angle", POLOIDAL_ANGLE)

    if "barometry.gauge.0.pressure.time" in ods:
        set_path(ods, "barometry.time", ods["barometry.gauge.0.pressure.time"])
        set_path(ods, "barometry.ids_properties.homogeneous_time", 1)
    if "equilibrium.time" in ods:
        set_path(ods, "equilibrium.ids_properties.homogeneous_time", 1)
    return ods


def _compare(reference, candidate, manifest: dict, label: str) -> None:
    result = compare_ods(
        semantic_sample_view(reference, manifest),
        semantic_sample_view(candidate, manifest),
        scope="union",
        reference_label="normalized pipeline ODS",
        candidate_label=label,
    )
    if not result.passed:
        differences = [
            entry.path
            for entry in result.entries
            if entry.classification.value == "unintended_regression"
        ]
        raise ValueError(f"Semantic parity failed for {label}: {differences[:10]}")


def generate(manifest_path: Path, canonical_source: Path) -> None:
    manifest = _load_manifest(manifest_path)
    normalized = normalized_pipeline_ods(canonical_source, manifest)
    target = manifest_path.parent / manifest["representations"]["imas"]["path"]
    vaft.imas.save(normalized, target, imas_version=str(manifest["imas_dd_version"]))
    record = manifest["representations"]["imas"]
    record["size"] = target.stat().st_size
    record["sha256"] = sha256_file(target)
    manifest["generation"]["canonical_source"] = str(
        canonical_source.relative_to(manifest_path.parent)
    )
    manifest["generation"]["canonical_sha256"] = sha256_file(canonical_source)
    manifest["generation"]["materialized_leaves"] = len(normalized.flat())
    _write_manifest(manifest_path, manifest)


def verify(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    verify_sample_artifacts(root, manifest)
    source = root / manifest["generation"]["canonical_source"]
    if sha256_file(source) != manifest["generation"]["canonical_sha256"]:
        raise ValueError(f"Canonical source checksum mismatch: {source}")
    checks = (
        (manifest["source"]["raw_source"], manifest["source"]["raw_sha256"]),
        (
            manifest["pipeline"]["configuration"],
            manifest["pipeline"]["configuration_sha256"],
        ),
        (
            manifest["pipeline"]["diagnostics_manifest"],
            manifest["pipeline"]["diagnostics_manifest_sha256"],
        ),
        (
            manifest["pipeline"]["efit_artifact_manifest"],
            manifest["pipeline"]["efit_artifact_manifest_sha256"],
        ),
    )
    for relative, expected in checks:
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Pipeline source checksum mismatch: {path}")
    normalized = normalized_pipeline_ods(source, manifest)
    artifact = root / manifest["representations"]["imas"]["path"]
    version = str(manifest["imas_dd_version"])
    via_omas = vaft.omas.load(artifact, imas_version=version)
    with vaft.imas.load(artifact, imas_version=version) as handle:
        if handle.info.format != "imas_netcdf" or handle.info.converted:
            raise ValueError("Repository sample is not a native IMAS NetCDF artifact")
        via_imas = handle.to_omas()
    _compare(normalized, via_omas, manifest, "IMAS artifact -> OMAS adapter")
    _compare(normalized, via_imas, manifest, "IMAS artifact -> IMAS adapter")

    expected_times = np.asarray(manifest["acceptance"]["equilibrium_times"])
    np.testing.assert_allclose(via_omas["equilibrium.time"], expected_times)
    ip = np.asarray(
        [
            via_omas[f"equilibrium.time_slice.{index}.global_quantities.ip"]
            for index in range(len(via_omas["equilibrium.time_slice"]))
        ],
        dtype=float,
    )
    if not np.isfinite(ip).all():
        raise ValueError("Regenerated EFIT sample has a non-finite plasma current")
    for index in range(len(via_omas.get("magnetics.b_field_pol_probe", []))):
        path = f"magnetics.b_field_pol_probe.{index}.poloidal_angle"
        if path in via_omas:
            np.testing.assert_allclose(via_omas[path], POLOIDAL_ANGLE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--canonical-source", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    if args.canonical_source is not None:
        generate(manifest, args.canonical_source.resolve())
    elif not args.verify:
        parser.error("--canonical-source is required unless --verify is used")
    verify(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
