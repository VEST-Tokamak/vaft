"""Contract tests for the canonical VEST diagnostic registry."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest
import yaml

from vaft.machine_mapping.registry import (
    AVAILABILITY_VALUES,
    LIFECYCLE_VALUES,
    MAPPING_STATUS_VALUES,
    DiagnosticRegistryError,
    documentation_snapshot,
    export_documentation_snapshot,
    load_diagnostic_registry,
    validate_diagnostic_registry,
)
from vaft.machine_mapping.utils import get_diagnostic_info, load_yaml, package_data_path


def test_registry_contains_the_legacy_inventory_and_valid_status_axes():
    registry = load_diagnostic_registry()
    required = {
        "pf_active", "tf", "gas_pumping", "gas_injection", "ech", "nbi", "helicity_injection",
        "magnetics.ip", "magnetics.flux_loop", "magnetics.b_field_pol_probe",
        "magnetics.diamagnetic_flux", "camera_visible", "spectrometer_uv", "thomson_scattering",
        "charge_exchange.ces", "charge_exchange.ion_doppler", "soft_x_rays", "wall",
    }
    assert required <= set(registry)
    for record in registry.values():
        assert record["availability"] in AVAILABILITY_VALUES
        assert record["lifecycle"] in LIFECYCLE_VALUES
        assert record["mapping_status"] in MAPPING_STATUS_VALUES


def test_implemented_registry_entries_resolve_to_real_mapping_entrypoints():
    for record in load_diagnostic_registry().values():
        # A partial mapping that names an entrypoint must resolve too.
        if record["mapping_status"] != "implemented" and "mapping" not in record:
            continue
        module = importlib.import_module(record["mapping"]["module"])
        assert callable(getattr(module, record["mapping"]["entrypoint"]))


def test_documentation_export_is_deterministic(tmp_path):
    first = export_documentation_snapshot(tmp_path / "one.yml")
    second = export_documentation_snapshot(tmp_path / "two.yml")
    assert first.read_bytes() == second.read_bytes()
    exported = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert exported == documentation_snapshot()
    assert exported["source"]["sha256"]


def test_registry_top_level_does_not_change_existing_yaml_lookup_semantics():
    assert get_diagnostic_info("39915", "tf")[0]["label"] == "TF"
    assert load_yaml(package_data_path("vest.yaml"))[20259]["pf_active"]["channel"][0]["gain"] == -50000


def test_loading_the_registry_never_imports_vaft_plot():
    # cold review 0.8.0 delta-absorb-19-docs F3: the subject check used to import
    # vaft.plot.taxonomy (and so every renderer and matplotlib) from inside the loader.
    code = (
        "import sys\n"
        "from vaft.machine_mapping.registry import load_diagnostic_registry\n"
        "records = load_diagnostic_registry()\n"
        "assert any('subject' in record for record in records.values())\n"
        "loaded = sorted(name for name in sys.modules if name == 'vaft.plot' or name.startswith('vaft.plot.'))\n"
        "assert not loaded, loaded\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_subject_membership_is_checked_against_the_vocabulary_the_caller_passes():
    record = {**load_diagnostic_registry()["pf_active"], "subject": "not_a_subject"}
    validate_diagnostic_registry({"pf_active": record})  # a bare name is schema-valid
    with pytest.raises(DiagnosticRegistryError, match="not a vaft.plot.taxonomy subject"):
        validate_diagnostic_registry({"pf_active": record}, subjects={"plasma_current"})
    with pytest.raises(DiagnosticRegistryError, match="non-empty name"):
        validate_diagnostic_registry({"pf_active": {**record, "subject": ""}})
    validate_diagnostic_registry(load_diagnostic_registry(), subjects={"pf_coil", *(
        r["subject"] for r in load_diagnostic_registry().values() if "subject" in r)})
