#!/usr/bin/env python3
"""Generate paired compact OMAS/IMAS artifacts from one canonical ODS.

The command accepts one canonical pipeline product only.  It never accepts
independent OMAS and IMAS inputs, which preserves the paired-generation
invariant required by issue #166.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
from omas import ODS, omas_environment
import yaml

import vaft
from vaft.data.reference_samples import (
    semantic_sample_view,
    sha256_file,
    verify_sample_artifacts,
)
from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE
from vaft.omas import compare_ods


def _compact_ods(source: ODS, selectors: list[str]) -> ODS:
    compact = ODS(consistency_check=False)
    with omas_environment(compact, dynamic_path_creation="dynamic_array_structures"):
        for path, value in sorted(source.flat().items()):
            if any(fnmatchcase(str(path), selector) for selector in selectors) and not (
                isinstance(value, np.ndarray) and value.size == 0
            ):
                compact[str(path)] = value
    if not compact.flat():
        raise ValueError(
            "No canonical ODS leaves matched manifest generation.selectors"
        )
    return compact


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


def _drop_invalid_signal(compact: ODS, base: str) -> None:
    """Remove an unavailable signal while retaining its channel metadata.

    The pipeline ODS carries a few IMAS default placeholders as scalar times or
    empty arrays.  Native IMAS requires each heterogeneous signal to have
    matching one-dimensional data and time coordinates, so those placeholders
    must not be serialized as real signals.
    """
    data = np.asarray(compact.get(f"{base}.data", []))
    time = np.asarray(compact.get(f"{base}.time", []))
    if not (
        data.ndim == 1
        and time.ndim == 1
        and data.size > 0
        and time.size == data.size
    ) and base in compact:
        del compact[base]


def _normalize_magnetics(compact: ODS) -> None:
    """Preserve every magnetic channel with portable signal coordinates."""
    if "magnetics" not in compact:
        return
    compact["magnetics.ids_properties.homogeneous_time"] = 0
    for index in range(len(compact.get("magnetics.b_field_pol_probe", []))):
        base = f"magnetics.b_field_pol_probe.{index}"
        # Issue #169: the stored sensitive axis must describe the +Bz signal.
        compact[f"{base}.poloidal_angle"] = POLOIDAL_ANGLE
        _drop_invalid_signal(compact, f"{base}.field")
        _drop_invalid_signal(compact, f"{base}.voltage")
    for index in range(len(compact.get("magnetics.b_field_tor_probe", []))):
        base = f"magnetics.b_field_tor_probe.{index}"
        _drop_invalid_signal(compact, f"{base}.field")
        _drop_invalid_signal(compact, f"{base}.voltage")
    for index in range(len(compact.get("magnetics.flux_loop", []))):
        _drop_invalid_signal(compact, f"magnetics.flux_loop.{index}.flux")
    for index in range(len(compact.get("magnetics.shunt", []))):
        _drop_invalid_signal(compact, f"magnetics.shunt.{index}.voltage")


def _materialize_compact(canonical: ODS, manifest: dict) -> ODS:
    """Select and normalize one manifest-defined representation of ``canonical``."""
    compact = _compact_ods(canonical, list(manifest["generation"]["selectors"]))
    equilibrium_slices = int(
        manifest.get("generation", {}).get("equilibrium_time_slices", 0)
    )
    if equilibrium_slices and "equilibrium.time" in compact:
        original_time = np.asarray(compact["equilibrium.time"])
        compact["equilibrium.time"] = original_time[:equilibrium_slices]
        for index in range(
            len(compact["equilibrium.time_slice"]) - 1,
            equilibrium_slices - 1,
            -1,
        ):
            del compact[f"equilibrium.time_slice.{index}"]
        for path, value in list(compact["equilibrium"].flat().items()):
            full_path = f"equilibrium.{path}"
            array = np.asarray(value)
            if (
                full_path != "equilibrium.time"
                and not full_path.startswith("equilibrium.time_slice.")
                and array.ndim
                and array.shape[0] == original_time.size
            ):
                compact[full_path] = array[:equilibrium_slices]
    if "barometry.gauge.0.pressure.time" in compact:
        compact["barometry.time"] = compact["barometry.gauge.0.pressure.time"]
        compact["barometry.ids_properties.homogeneous_time"] = 1
    if "equilibrium.time" in compact:
        compact["equilibrium.ids_properties.homogeneous_time"] = 1
    _normalize_magnetics(compact)
    return compact


def _write_artifacts(
    compact: ODS, root: Path, manifest: dict, *, canonical_source: Path
) -> None:
    """Serialize one compact ODS and update its representation records."""
    root.mkdir(parents=True, exist_ok=True)

    omas_path = root / manifest["representations"]["omas"]["path"]
    imas_path = root / manifest["representations"]["imas"]["path"]
    vaft.omas.save(compact, omas_path)
    vaft.imas.save(compact, imas_path, imas_version=str(manifest["imas_dd_version"]))

    try:
        canonical_relative = canonical_source.relative_to(root)
    except ValueError:
        # Wheel staging contains only the compact representation; its canonical
        # regeneration input remains repository-only beside the source manifest.
        canonical_relative = Path(manifest["generation"]["canonical_source"])
    manifest["generation"]["canonical_source"] = str(canonical_relative)
    manifest["generation"]["canonical_sha256"] = sha256_file(canonical_source)
    manifest["generation"]["materialized_leaves"] = len(compact.flat())
    for representation, path in (("omas", omas_path), ("imas", imas_path)):
        manifest["representations"][representation]["size"] = path.stat().st_size
        manifest["representations"][representation]["sha256"] = sha256_file(path)


def generate(canonical_source: Path, manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    canonical = vaft.omas.load(
        canonical_source, imas_version=str(manifest["imas_dd_version"])
    )
    compact = _materialize_compact(canonical, manifest)
    _write_artifacts(compact, root, manifest, canonical_source=canonical_source)

    _write_manifest(manifest_path, manifest)


def generate_wheel_artifacts(
    canonical_source: Path, manifest_path: Path, output_root: Path
) -> None:
    """Create the three-slice wheel variant from the same canonical ODS.

    Repository artifacts deliberately retain every available EFIT time slice.
    The wheel ships a small self-contained variant, selected from the identical
    canonical ODS rather than from independently supplied representation files.
    """
    manifest = deepcopy(_load_manifest(manifest_path))
    slice_count = int(manifest["packaging_policy"]["wheel_equilibrium_time_slices"])
    manifest["generation"]["equilibrium_time_slices"] = slice_count
    manifest["generation"]["distribution_variant"] = "wheel"
    manifest["generation"]["canonical_policy"] = "repository-only"
    manifest["acceptance"]["equilibrium_times"] = manifest["acceptance"][
        "equilibrium_times"
    ][:slice_count]
    canonical = vaft.omas.load(
        canonical_source, imas_version=str(manifest["imas_dd_version"])
    )
    compact = _materialize_compact(canonical, manifest)
    _write_artifacts(
        compact, output_root, manifest, canonical_source=canonical_source
    )
    _write_manifest(output_root / "manifest.yaml", manifest)


def verify(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    verify_sample_artifacts(root, manifest)
    source_checks = (
        (
            manifest["generation"]["canonical_source"],
            manifest["generation"]["canonical_sha256"],
        ),
        (manifest["pipeline"]["raw_source"], manifest["pipeline"]["raw_sha256"]),
        (manifest["pipeline"]["config"], manifest["pipeline"]["config_sha256"]),
        (
            manifest["pipeline"]["diagnostics_manifest"],
            manifest["pipeline"]["diagnostics_manifest_sha256"],
        ),
        (
            manifest["pipeline"]["efit_artifact_manifest"],
            manifest["pipeline"]["efit_artifact_manifest_sha256"],
        ),
    )
    for relative, expected in source_checks:
        source = root / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"Missing or modified sample generation source: {source}")
    version = str(manifest["imas_dd_version"])
    omas_path = root / manifest["representations"]["omas"]["path"]
    imas_path = root / manifest["representations"]["imas"]["path"]

    native_omas = vaft.omas.load(omas_path, imas_version=version)
    with vaft.imas.load(omas_path, imas_version=version) as handle:
        omas_as_imas = handle.to_omas()
    imas_as_omas = vaft.omas.load(imas_path, imas_version=version)
    with vaft.imas.load(imas_path, imas_version=version) as handle:
        native_imas = handle.to_omas()

    reference = semantic_sample_view(native_omas, manifest)
    if not reference:
        raise ValueError("Semantic comparison policy selected no reference paths")
    for label, candidate in (
        ("OMAS artifact -> IMAS adapter", omas_as_imas),
        ("IMAS artifact -> OMAS adapter", imas_as_omas),
        ("IMAS artifact -> IMAS adapter", native_imas),
    ):
        result = compare_ods(
            reference,
            semantic_sample_view(candidate, manifest),
            scope="union",
            reference_label="OMAS artifact -> OMAS adapter",
            candidate_label=label,
        )
        if not result.passed:
            differences = [
                entry.path
                for entry in result.entries
                if entry.classification.value == "unintended_regression"
            ]
            raise ValueError(f"Semantic parity failed for {label}: {differences[:10]}")

    expected_times = np.asarray(
        manifest["acceptance"]["equilibrium_times"], dtype=float
    )
    np.testing.assert_allclose(native_omas["equilibrium.time"], expected_times)
    expected_magnetics = manifest["acceptance"]["magnetics"]
    if len(native_omas["magnetics.b_field_pol_probe"]) != int(
        expected_magnetics["b_field_pol_probe_count"]
    ):
        raise ValueError("Compact sample does not contain every B-pol probe")
    if len(native_omas["magnetics.flux_loop"]) != int(
        expected_magnetics["flux_loop_count"]
    ):
        raise ValueError("Compact sample does not contain every flux loop")
    # These arrays are referenced by the machine description and by the
    # single-shot examples, so a selector that silently truncates one must fail
    # generation.
    for ids, node, key, label in (
        ("pf_active", "pf_active.coil", "coil_count", "active coil"),
        (
            "spectrometer_uv",
            "spectrometer_uv.channel",
            "channel_count",
            "filterscope channel",
        ),
        ("pf_passive", "pf_passive.loop", "loop_count", "passive loop"),
    ):
        expected_count = manifest["acceptance"].get(ids, {}).get(key)
        if expected_count is not None and len(native_omas[node]) != int(expected_count):
            raise ValueError(
                f"Compact sample does not contain every {label}: "
                f"{len(native_omas[node])} of {expected_count}"
            )

    # pf_passive carries geometry only: the loop currents are ~48 MB of the
    # canonical source and em_coupling's matrices are reproducible from the
    # packaged versioned asset (vaft.machine_mapping.em_coupling), so neither
    # belongs in a compact sample. Assert the omission rather than trusting the
    # selectors to keep expressing it.
    if manifest["acceptance"].get("pf_passive", {}).get("static_only"):
        dynamic = sorted(
            path
            for path in native_omas.flat()
            if str(path).startswith("pf_passive.")
            and (str(path).endswith(".current") or str(path) == "pf_passive.time")
        )
        if dynamic:
            raise ValueError(
                "Compact sample must carry pf_passive geometry only; found "
                + ", ".join(dynamic[:5])
            )
        if "em_coupling" in native_omas:
            raise ValueError(
                "Compact sample must not carry em_coupling: reconstruct it with "
                "vaft.machine_mapping.em_coupling instead of materializing the "
                "coupling matrices"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--canonical-source", type=Path)
    parser.add_argument("--wheel-output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.canonical_source is None:
        parser.error("--canonical-source is required unless --verify is used")
    if args.canonical_source is not None:
        generate(args.canonical_source.resolve(), args.manifest.resolve())
        if args.wheel_output is not None:
            generate_wheel_artifacts(
                args.canonical_source.resolve(),
                args.manifest.resolve(),
                args.wheel_output.resolve(),
            )
    elif args.wheel_output is not None:
        parser.error("--wheel-output requires --canonical-source")
    verify(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
