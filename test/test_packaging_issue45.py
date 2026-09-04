"""Regression coverage for packaging and configuration issue #45."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_apache_license_exists_and_matches_project_metadata():
    license_path = ROOT / "LICENSE"
    text = license_path.read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text

    project = _pyproject()["project"]
    assert project["license"]["file"] == "LICENSE"
    assert (
        "License :: OSI Approved :: Apache Software License" in project["classifiers"]
    )


def test_uv_sources_do_not_reference_missing_local_vendor_paths():
    sources = _pyproject().get("tool", {}).get("uv", {}).get("sources", {})
    for entries in sources.values():
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if "path" in entry:
                assert (ROOT / entry["path"]).exists(), entry["path"]


def test_runtime_configuration_is_declared_as_package_data():
    setuptools = _pyproject()["tool"]["setuptools"]
    package_data = set(setuptools["package-data"]["vaft"])
    assert setuptools["include-package-data"] is False
    assert "machine_mapping/vest.yaml" in package_data
    assert ".hscfg.example" in package_data
    assert "data/geometry/Coil_info.mat" in package_data
    assert (
        "data/geometry/VEST_DiscretizedCoilGeometry_Full_ver_1906.mat" in package_data
    )
    assert (
        "data/geometry/VEST_DiscretizedCoilGeometry_Full_ver_2507.mat" in package_data
    )
    assert "data/geometry/VEST_em_coupling_pf_versions.npz" in package_data
    assert "data/efit/*" not in package_data
    assert "data/imas/*.nc" not in package_data
    assert "data/legacy/*.csv" not in package_data
    assert "data/legacy/*.gz" not in package_data
    assert "data/legacy/*.h5" not in package_data
    assert "data/legacy/*.mat" not in package_data
    assert "data/omas/*.json" not in package_data
    assert "data/samples/*/manifest.yaml" in package_data
    assert "data/samples/39915/omas.json.gz" in package_data
    assert "data/samples/39915/imas.nc" in package_data
    assert "data/samples/39915/source/*" not in package_data
    assert "data/samples/41524/imas.nc" not in package_data
    assert "data/samples/41672/imas.nc" not in package_data
    assert "data/geometry/VEST_static_geometry.json.gz" in package_data
    assert not (ROOT / "vaft" / ".hscfg").exists()
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "wheel_samples" in setup_py
    assert "build_py" in setup_py

    example = (ROOT / "vaft" / ".hscfg.example").read_text(encoding="utf-8")
    assert "your_username" in example
    assert "reader\n" not in example


def test_sdist_manifest_uses_the_same_data_allowlist():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "prune vaft/data" in manifest
    assert "prune test" in manifest
    assert "global-exclude *.py[cod]" in manifest
    assert "global-exclude ._*" in manifest
    assert "include vaft/data/geometry/Coil_info.mat" in manifest
    assert (
        "include vaft/data/geometry/VEST_DiscretizedCoilGeometry_Full_ver_1906.mat"
        in manifest
    )
    assert (
        "include vaft/data/geometry/VEST_DiscretizedCoilGeometry_Full_ver_2507.mat"
        in manifest
    )
    assert "include vaft/data/geometry/VEST_em_coupling_pf_versions.npz" in manifest
    assert "include vaft/data/geometry/*.yaml" in manifest
    assert "include vaft/data/geometry/*.csv" in manifest
    assert "include vaft/data/gpec/*.in" in manifest
    assert "include vaft/data/gpec/*.dat" in manifest
    assert "include vaft/data/legacy/*.txt" in manifest
    assert "include vaft/data/legacy/*.yaml" in manifest
    assert "include vaft/data/samples/*/manifest.yaml" in manifest
    assert "include vaft/data/samples/39915/omas.json.gz" in manifest
    assert "include vaft/data/samples/39915/imas.nc" in manifest
    assert "include vaft/data/wheel_samples/39915/manifest.yaml" in manifest
    assert "include vaft/data/wheel_samples/39915/omas.json.gz" in manifest
    assert "include vaft/data/wheel_samples/39915/imas.nc" in manifest
    assert "include vaft/data/geometry/VEST_static_geometry.json.gz" in manifest


def test_vest_yaml_has_canonical_top_level_diagnostic_structure():
    content = yaml.safe_load(
        (ROOT / "vaft" / "machine_mapping" / "vest.yaml").read_text(encoding="utf-8")
    )
    defaults = content[0]

    assert {
        "magnetics",
        "pf_active",
        "tf",
        "barometry",
        "spectrometer_uv",
        "halo_current",
    } <= set(defaults)
    assert "barometry" not in defaults["pf_active"]
    assert "spectrometer_uv" not in defaults["pf_active"]
    assert "limiter_monitor" not in defaults["pf_active"]
    assert set(defaults["tf"]) == {0}
    assert defaults["halo_current"]["limiter_monitor"]["channels"][0]["field"] == 216

    assert content[20259]["pf_active"]["channel"][0]["gain"] == -50000


def test_legacy_seaborn_style_module_imports_with_supported_matplotlib():
    path = ROOT / "workflow" / "automatic_pipeline_3_data_summary" / "efit_analysis.py"
    spec = importlib.util.spec_from_file_location("issue45_efit_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def test_empty_equilibrium_workflow_placeholder_was_removed():
    path = (
        ROOT
        / "workflow"
        / "automatic_pipeline_2_corrective_data_update"
        / "update_equilibrium.py"
    )
    assert not path.exists()


def test_wheel_sample_carries_the_same_conventions_as_the_checkout_sample():
    """The swapped-in wheel sample must not drift from the mapping code.

    ``setup.py`` replaces ``vaft/data/samples/39915`` with the compact
    ``vaft/data/wheel_samples/39915`` in every build, so that copy -- not the
    checkout one -- is what users install. Regenerating the checkout sample
    without regenerating the wheel variant therefore ships conventions the
    library itself no longer writes: v0.6.0 published an IMPA
    ``b_field_tor_probe`` ``toroidal_angle`` of 0.0 (a radial sensor) while
    the mapper wrote pi/2.
    """
    import gzip
    import json

    from vaft.machine_mapping.impa import IMPA_TOROIDAL_PROBE_TOROIDAL_ANGLE
    from vaft.machine_mapping.magnetics import POLOIDAL_ANGLE

    wheel_sample = ROOT / "vaft" / "data" / "wheel_samples" / "39915" / "omas.json.gz"
    with gzip.open(wheel_sample, "rt") as handle:
        ods = json.load(handle)

    magnetics = ods["magnetics"]

    poloidal = {
        round(float(probe["poloidal_angle"]), 9)
        for probe in magnetics["b_field_pol_probe"]
        if "poloidal_angle" in probe
    }
    assert poloidal == {round(float(POLOIDAL_ANGLE), 9)}

    toroidal = {
        round(float(probe["toroidal_angle"]), 9)
        for probe in magnetics.get("b_field_tor_probe", [])
        if "toroidal_angle" in probe
    }
    assert toroidal == {round(float(IMPA_TOROIDAL_PROBE_TOROIDAL_ANGLE), 9)}
