"""Paired OMAS/IMAS sample registry and semantic compatibility tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

import vaft
from vaft.data.reference_samples import semantic_sample_view, verify_sample_artifacts
from vaft.omas import compare_ods


def test_sample_registry_returns_both_packaged_artifacts():
    assert vaft.data.available_samples() == (39915, 41524, 41672)
    omas_path = vaft.data.sample(39915, representation="omas")
    imas_path = vaft.data.sample(39915, representation="imas")

    assert omas_path.name == "omas.json.gz"
    assert imas_path.name == "imas.nc"
    assert omas_path.is_file()
    assert imas_path.is_file()
    manifest = vaft.data.sample_manifest(39915)
    assert manifest["imas_dd_version"] == "3.41.0"
    assert verify_sample_artifacts(omas_path.parent, manifest) == {
        "omas": manifest["representations"]["omas"]["sha256"],
        "imas": manifest["representations"]["imas"]["sha256"],
    }


def test_sample_registry_errors_are_actionable():
    with pytest.raises(ValueError, match="available shots: 39915, 41524, 41672"):
        vaft.data.sample(1)
    with pytest.raises(ValueError, match="expected one of: imas, omas"):
        vaft.data.sample(39915, representation="pickle")


@pytest.mark.parametrize("shot", [41524, 41672])
def test_imas_only_samples_use_adapter_compatible_fallback(shot):
    omas_adapter_path = vaft.data.sample(shot, representation="omas")
    imas_adapter_path = vaft.data.sample(shot, representation="imas")

    assert omas_adapter_path == imas_adapter_path
    assert omas_adapter_path.name == "imas.nc"
    assert (
        vaft.data.sample_manifest(shot)["representations"]["imas"]["package"]
        == "repository-only"
    )


@pytest.mark.parametrize("shot", [41524, 41672])
def test_full_imas_samples_round_trip_through_both_adapters(shot):
    manifest = vaft.data.sample_manifest(shot)
    path = vaft.data.sample(shot, representation="omas")
    version = manifest["imas_dd_version"]

    via_omas = vaft.omas.load(path, imas_version=version)
    with vaft.imas.load(path, imas_version=version) as handle:
        assert handle.info.format == "imas_netcdf"
        assert handle.info.converted is False
        via_imas = handle.to_omas()

    result = compare_ods(
        semantic_sample_view(via_omas, manifest),
        semantic_sample_view(via_imas, manifest),
        scope="union",
    )
    assert result.passed
    np.testing.assert_allclose(
        via_omas["equilibrium.time"], manifest["acceptance"]["equilibrium_times"]
    )
    assert via_omas["em_coupling.mutual_passive_passive"].shape == (950, 950)
    assert via_omas["magnetics.ip.0.data"].size > 100
    for index in range(len(via_omas["magnetics.b_field_pol_probe"])):
        probe = via_omas[f"magnetics.b_field_pol_probe.{index}"]
        if "poloidal_angle" in probe:
            np.testing.assert_allclose(probe["poloidal_angle"], np.pi / 2)
    if shot in (41524, 41672):
        assert manifest["source"]["kind"] == "pipeline-until-efit"
        assert (
            manifest["pipeline"]["efit"]["successful_time_slices"]
            == len(manifest["acceptance"]["equilibrium_times"])
        )
        ip = np.asarray(
            [
                via_omas[f"equilibrium.time_slice.{index}.global_quantities.ip"]
                for index in range(len(via_omas["equilibrium.time_slice"]))
            ],
            dtype=float,
        )
        assert np.isfinite(ip).all()
        assert np.max(ip) > 1.0e5


def test_repository_only_lookup_explains_missing_wheel_artifact(monkeypatch, tmp_path):
    from vaft.data import resources

    manifest_path = resources.data_path("samples/41524/manifest.yaml")

    def installed_data_path(name=""):
        if name == "samples/41524/manifest.yaml":
            return manifest_path
        return tmp_path / name

    monkeypatch.setattr(resources, "data_path", installed_data_path)
    with pytest.raises(FileNotFoundError, match="Clone the VAFT GitHub repository"):
        resources.sample(41524, representation="omas")


def test_legacy_sample_ods_wrapper_uses_registry_and_sample_odc_is_removed():
    ods = vaft.omas.sample_ods()
    assert ods["dataset_description.data_entry.pulse"] == 39915
    assert "equilibrium" in ods
    assert not hasattr(vaft.omas, "sample_odc")


def test_reference_probe_metadata_describes_positive_bz():
    ods = vaft.omas.load(vaft.data.sample(39915, representation="omas"))
    assert len(ods["magnetics.b_field_pol_probe"]) == 76
    assert len(ods["magnetics.flux_loop"]) == 11
    for index in range(76):
        angle = ods[f"magnetics.b_field_pol_probe.{index}.poloidal_angle"]
        np.testing.assert_allclose(
            (np.cos(angle), np.sin(angle)), (0.0, 1.0), atol=1e-15
        )


@pytest.mark.parametrize("shot", [39915, 41524, 41672])
def test_native_imas_sample_records_explicit_dd_version(shot):
    with h5py.File(vaft.data.sample(shot, "imas"), "r") as handle:
        version = handle.attrs["data_dictionary_version"]
    if isinstance(version, bytes):
        version = version.decode()
    assert str(version) == "3.41.0"


def test_sample_artifact_checksum_failure_is_actionable(tmp_path):
    artifact = tmp_path / "imas.nc"
    artifact.write_bytes(b"tampered")
    manifest = {
        "representations": {
            "imas": {
                "path": artifact.name,
                "sha256": "0" * 64,
                "size": artifact.stat().st_size,
            }
        }
    }
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sample_artifacts(tmp_path, manifest)


def test_pinned_legacy_source_checksum_failure_is_actionable(monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "workflow"
        / "reference_validation"
        / "generate_legacy_imas_sample.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_sample_generator", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=0, stdout=b"tampered", stderr=b""
        ),
    )
    manifest = {
        "source": {
            "git_commit": "deadbeef",
            "git_path": "sample.json",
            "sha256": "0" * 64,
        }
    }
    with pytest.raises(ValueError, match="source checksum mismatch"):
        module._git_source_bytes(Path.cwd(), manifest)


def test_paired_sample_exercises_complete_adapter_matrix():
    manifest = vaft.data.sample_manifest(39915)
    version = manifest["imas_dd_version"]
    omas_path = vaft.data.sample(39915, "omas")
    imas_path = vaft.data.sample(39915, "imas")

    omas_native = vaft.omas.load(omas_path, imas_version=version)
    with vaft.imas.load(omas_path, imas_version=version) as handle:
        omas_converted_to_imas = handle.to_omas()
    imas_converted_to_omas = vaft.omas.load(imas_path, imas_version=version)
    with vaft.imas.load(imas_path, imas_version=version) as handle:
        assert handle.info.format == "imas_netcdf"
        assert handle.info.converted is False
        imas_native = handle.to_omas()

    reference = semantic_sample_view(omas_native, manifest)
    assert len(reference) > 100
    for candidate in (
        omas_converted_to_imas,
        imas_converted_to_omas,
        imas_native,
    ):
        result = compare_ods(
            reference,
            semantic_sample_view(candidate, manifest),
            scope="union",
        )
        assert result.passed

    np.testing.assert_allclose(
        omas_native["equilibrium.time"],
        [0.316, 0.317, 0.318, 0.319, 0.320, 0.323, 0.325, 0.326, 0.331],
    )
    assert len(omas_native["equilibrium.time"]) == manifest["pipeline"]["efit"][
        "successful_time_slices"
    ]
    assert len(omas_native["magnetics.b_field_pol_probe"]) == 76
    assert len(omas_native["magnetics.flux_loop"]) == 11
    assert omas_native["magnetics.ids_properties.homogeneous_time"] == 0
    for family, signal in (
        ("b_field_pol_probe", "field"),
        ("b_field_pol_probe", "voltage"),
        ("flux_loop", "flux"),
    ):
        for index in range(len(omas_native[f"magnetics.{family}"])):
            base = f"magnetics.{family}.{index}.{signal}"
            if f"{base}.data" not in omas_native:
                continue
            assert omas_native[f"{base}.data"].shape == omas_native[
                f"{base}.time"
            ].shape
    assert manifest["packaging_policy"]["repository_equilibrium_time_slices"] == "all"
    assert omas_native["wall.description_2d.0.limiter.unit.0.outline.r"].size > 10
    assert (
        omas_native["magnetics.ip.0.data"].shape
        == omas_native["magnetics.ip.0.time"].shape
    )
    assert (
        omas_native["pf_active.coil.0.current.data"].shape
        == omas_native["pf_active.time"].shape
    )
    assert omas_native["equilibrium.time_slice.0.profiles_1d.psi"].size > 10


def test_wheel_build_uses_the_three_slice_39915_variant(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build_lib = tmp_path / "build"
    subprocess.run(
        [sys.executable, "setup.py", "build_py", "--build-lib", str(build_lib)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sample_root = build_lib / "vaft" / "data" / "samples" / "39915"
    manifest = yaml.safe_load((sample_root / "manifest.yaml").read_text())
    assert manifest["generation"]["distribution_variant"] == "wheel"
    assert manifest["generation"]["equilibrium_time_slices"] == 3
    verify_sample_artifacts(sample_root, manifest)
    wheel_ods = vaft.omas.load(sample_root / "omas.json.gz")
    np.testing.assert_allclose(wheel_ods["equilibrium.time"], [0.316, 0.317, 0.318])
    assert len(wheel_ods["magnetics.b_field_pol_probe"]) == 76
    assert len(wheel_ods["magnetics.flux_loop"]) == 11
    with vaft.imas.load(sample_root / "imas.nc") as handle:
        wheel_native = handle.to_omas()
    np.testing.assert_allclose(
        wheel_native["equilibrium.time"], [0.316, 0.317, 0.318]
    )
    assert len(wheel_native["magnetics.b_field_pol_probe"]) == 76
    assert len(wheel_native["magnetics.flux_loop"]) == 11


def test_packaged_sample_carries_passive_geometry_without_derivable_data():
    """The 39915 sample stores measurements, not reconstructions.

    ``pf_passive`` loop currents are ~48 MB of the canonical source and
    ``em_coupling``'s matrices are a function of the PF geometry version, so the
    sample keeps the geometry and leaves both to
    :func:`vaft.machine_mapping.em_coupling.em_coupling`.
    """
    ods = vaft.omas.sample_ods()

    assert "em_coupling" not in ods
    assert len(ods["pf_passive.loop"]) == 950
    assert "time" not in ods["pf_passive"]
    currents = [
        path for path in ods.flat()
        if str(path).startswith("pf_passive.") and str(path).endswith(".current")
    ]
    assert currents == []

    # The geometry that makes the omission safe is all present.
    for index in (0, 949):
        loop = ods[f"pf_passive.loop.{index}"]
        assert len(loop["element.0.geometry.outline.r"]) > 0
        assert float(loop["resistance"]) > 0.0


def test_em_coupling_is_reconstructible_from_the_packaged_geometry():
    """The omission is only defensible while the reconstruction works."""
    from vaft.machine_mapping.em_coupling import em_coupling

    ods = vaft.omas.sample_ods()
    assert "em_coupling" not in ods

    em_coupling(ods, shot=39915)

    assert np.shape(ods["em_coupling.mutual_active_active"]) == (10, 10)
    assert np.shape(ods["em_coupling.mutual_passive_active"]) == (950, 10)
    # Richer than the sample ever stored: the passive-passive matrix was never
    # in the compact selectors at all.
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (950, 950)
    assert len(ods["em_coupling.active_coils"]) == len(ods["pf_active.coil"])
    assert len(ods["em_coupling.passive_loops"]) == len(ods["pf_passive.loop"])
