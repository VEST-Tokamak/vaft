"""Byte-exact, atomic Atlas snapshot capture and product/time safeguards."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "workflow/kinetic_state" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture_source(tmp_path, monkeypatch):
    module = _script("capture_verification_snapshot")
    source = tmp_path / "atlas"
    shutil.copytree(ROOT / "notebooks/data/verification_and_validation", source)
    # The historical v1 snapshot did not retain this source manifest.
    if not (source / "confinement/MANIFEST.json").exists():
        (source / "confinement/MANIFEST.json").write_text('{"rows": 133}\n')
    with (source / "v1/virial_identity.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    filedb = tmp_path / "filedb"
    with (source / "v1/state.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        state_columns, state_rows = reader.fieldnames, list(reader)
    product_hash = hashlib.sha256(b"fixture").hexdigest()
    for row in rows:
        product = filedb / row["efit_product"]
        product.parent.mkdir(parents=True, exist_ok=True)
        product.write_bytes(b"fixture")
        row["efit_product_sha256"] = product_hash
    for row in state_rows:
        row["efit_product_sha256"] = product_hash
    with (source / "v1/state.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=state_columns)
        writer.writeheader()
        writer.writerows(state_rows)
    monkeypatch.setattr(module.exporter, "export", lambda *_: rows)
    return module, source, filedb, tmp_path / "captured"


def test_capture_keeps_source_bytes_and_records_provenance(capture_source):
    module, source, filedb, out = capture_source
    before = {rel: (source / rel).read_bytes() for rel in module.COPY_FILES}
    manifest = module.capture(source, filedb, out)
    assert manifest["snapshot_version"] == 2
    assert manifest["virial_settings"]["alpha_method"] == "volume"
    assert manifest["rows"]["v1/state.csv"] == manifest["rows"]["v1/virial_identity.csv"] == 150
    assert len(manifest["products_sha256"]) == 49
    for rel, data in before.items():
        assert (out / rel).read_bytes() == (source / rel).read_bytes() == data
    for rel, expected in manifest["sha256"].items():
        assert hashlib.sha256((out / rel).read_bytes()).hexdigest() == expected
    assert json.loads((out / "snapshot_manifest.json").read_text()) == manifest


def test_capture_failure_leaves_no_partial_destination(capture_source, monkeypatch):
    module, source, filedb, out = capture_source
    def fail(*_):
        raise ValueError("product SHA-256 mismatch")
    monkeypatch.setattr(module.exporter, "export", fail)
    with pytest.raises(ValueError, match="SHA-256"):
        module.capture(source, filedb, out)
    assert not out.exists() and not list(out.parent.glob(".vv-capture-*"))


def test_capture_rejects_existing_output_and_changing_source(capture_source, monkeypatch):
    module, source, filedb, out = capture_source
    out.mkdir()
    with pytest.raises(FileExistsError):
        module.capture(source, filedb, out)
    out.rmdir()
    original = module.exporter.export
    def change(*args):
        rows = original(*args)
        with (source / "v1/state.csv").open("ab") as stream:
            stream.write(b"\n")
        return rows
    monkeypatch.setattr(module.exporter, "export", change)
    with pytest.raises(ValueError, match="source changed"):
        module.capture(source, filedb, out)
    assert not out.exists()


def test_capture_rejects_wrong_quality_counts_and_changed_products(capture_source, monkeypatch):
    module, source, filedb, out = capture_source
    metadata = source / "equilibrium_quality/manifest.json"
    original_metadata = metadata.read_text()
    data = json.loads(original_metadata)
    data["rows"] += 1
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest row count"):
        module.capture(source, filedb, out)
    metadata.write_text(original_metadata)
    original = module.exporter.export
    def change_product(*args):
        rows = original(*args)
        (filedb / rows[0]["efit_product"]).write_bytes(b"changed")
        return rows
    monkeypatch.setattr(module.exporter, "export", change_product)
    with pytest.raises(ValueError, match="product changed"):
        module.capture(source, filedb, out)
    assert not out.exists()


def test_export_checks_hash_and_slice_time_before_computing(tmp_path, monkeypatch):
    module = _script("export_virial_identity")
    product = tmp_path / "efit.json"
    product.write_bytes(b"fixture")
    state = tmp_path / "state.csv"
    def write_state(product_hash, time=0.32):
        with state.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=module.COLUMNS[:5])
            writer.writeheader()
            writer.writerow({"shot": 1, "time_efit_s": time, "efit_lineage": "magnetics",
                             "efit_product": product.name, "efit_product_sha256": product_hash})
    write_state("incorrect")
    monkeypatch.setattr(module, "_load_product", lambda _: object())
    with pytest.raises(ValueError, match="SHA-256"):
        module.export(state, tmp_path)
    def match(_ods, time, *, tolerance_s):
        if abs(time - 0.32) > tolerance_s:
            raise LookupError("outside tolerance")
        return 0, 0
    monkeypatch.setattr(module, "slice_at_time", match)
    write_state(hashlib.sha256(product.read_bytes()).hexdigest(), time=0.34)
    with pytest.raises(LookupError, match="tolerance"):
        module.export(state, tmp_path)
    write_state(hashlib.sha256(product.read_bytes()).hexdigest())
    closures = ("pair_13", "pair_23", "lao_li", "full_123")
    def compute(_ods, *, time_slice, alpha_method):
        assert time_slice == 0 and alpha_method == "volume"
        return {0: {
            "identity": {"e1_normalized": 0.01, "e2_normalized": 0.02, "e3_normalized": 0.03,
                         "rms": 0.02, "identities_available": 3},
            "volume": {"beta_p": 0.2, "li": 0.6}, "pair_13": {"beta_p": 0.21, "li": 0.61},
            "alpha": 1.3, "conditioning": {
                "denominators": dict.fromkeys(closures, 0.4),
                "distance_to_singularity": dict.fromkeys(closures, 0.1),
                "rt_denominator_ratio": 0.8, "rt_over_r0_available": True}}}
    monkeypatch.setattr(module, "compute_virial_equilibrium_quantities_ods", compute)
    row = module.export(state, tmp_path)[0]
    assert set(row) == set(module.COLUMNS)
    assert row["pair_13_denominator"] == 0.4 and row["rt_over_r0_available"] is True
