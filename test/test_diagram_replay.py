"""Scientific state parity and integrity for lazy documentation replay."""

import shutil
import json
import gzip

import pytest

from vaft.diagram import magnetic_island, exb_drift
from vaft.diagram._replay import (
    ROOT,
    RECIPES,
    catalog_rows,
    element_rows,
    encode,
    make_payload,
    validate_payload,
)


def combined(data, index):
    rows = {row["id"]: row for row in data["static"] + data["states"][index]}
    return [rows[id] for id in data["order"]]


def test_island_exact_static_states_and_deterministic_export():
    data = make_payload(RECIPES[0])
    assert encode(data) == encode(make_payload(RECIPES[0]))
    for i in (0, len(data["states"]) // 2, len(data["states"]) - 1):
        scene = magnetic_island(
            phase=data["driver"]["values"][i], **RECIPES[0]["parameters"]
        ).scene
        assert encode(combined(data, i)) == encode(element_rows(scene))
    assert data["coordinates"]["dtype"] == "float64"
    assert data["interpolation"] == "none"


def test_exb_shares_actual_stored_times_and_trace_state():
    data = make_payload(RECIPES[1])
    model = exb_drift(**RECIPES[1]["parameters"]).model
    for i in (0, len(data["states"]) // 2, len(data["states"]) - 1):
        t = data["driver"]["values"][i]
        j = round(t / model.parameters["electron_dt"])
        assert j * model.parameters["electron_dt"] == pytest.approx(t, abs=1e-12)
        rows = {r["id"]: r for r in combined(data, i)}
        assert rows["ion_particle-1"]["at"] == pytest.approx(
            0.45 * model.orbits["ion"][i, :2]
        )
        assert rows["electron_particle-1"]["at"] == pytest.approx(
            0.45 * model.orbits["electron"][j, :2]
        )
        assert data["observables"][1]["values"][i] == model.orbits["electron"][j, 0]
    assert data["driver"]["unit"] == "normalized time"


@pytest.mark.parametrize(
    "corrupt",
    [
        "nan",
        "shape",
        "duplicate",
        "unit",
        "version",
        "interpolation",
        "observable",
        "coordinates",
        "numeric_driver",
        "point",
    ],
)
def test_malformed_scientific_states_refused(corrupt):
    data = make_payload(RECIPES[0])
    if corrupt == "nan":
        data["driver"]["values"][1] = float("nan")
    elif corrupt == "shape":
        data["states"].pop()
    elif corrupt == "duplicate":
        data["order"][0] = data["order"][1]
    elif corrupt == "unit":
        data["driver"]["unit"] = ""
    elif corrupt == "version":
        data["schema_version"] = 999
    elif corrupt == "interpolation":
        data["interpolation"] = "linear"
    elif corrupt == "observable":
        data["observables"] = [{"name": "bad", "unit": "m", "values": [1]}]
    elif corrupt == "coordinates":
        data["coordinates"]["unit"] = "m"
    elif corrupt == "numeric_driver":
        data["driver"]["values"][1] = "wrong"
    elif corrupt == "point":
        row = next(r for r in data["states"][0] if r["type"] == "polyline")
        row["points"] = [[0, 1, 2], *row["points"][1:]]
    with pytest.raises(ValueError):
        validate_payload(data)


def test_committed_artifacts_and_manifest_integrity(tmp_path):
    rows = catalog_rows()
    assert len(rows) == 2 and all(r["gzip_bytes"] <= 300 * 1024 for r in rows)
    shutil.copytree(
        ROOT / "docs/assets/animations", tmp_path / "docs/assets/animations"
    )
    shutil.copytree(ROOT / "docs/assets/diagrams", tmp_path / "docs/assets/diagrams")
    for row in rows:
        for source in row["sources"]:
            destination = tmp_path / source["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source["path"], destination)
    artifact = tmp_path / "docs" / rows[0]["artifact"]
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        catalog_rows(tmp_path)


def test_manifest_uses_utf8_even_with_windows_default_encoding(monkeypatch):
    from pathlib import Path

    read_text = Path.read_text
    def windows_read_text(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)
    monkeypatch.setattr(Path, "read_text", windows_read_text)
    assert catalog_rows()[1]["title"] == "E×B particle and guiding-centre motion"


def test_valid_compressor_size_variation_is_accepted(monkeypatch):
    compress = gzip.compress

    def with_extra_header(raw, **kwargs):
        compressed = compress(raw, **kwargs)
        # A valid optional gzip FEXTRA field changes size without assuming a
        # particular compression level stays below budget on every backend.
        alternate = (
            compressed[:3] + bytes([compressed[3] | 4]) + compressed[4:10]
            + b"\x01\x00x" + compressed[10:]
        )
        assert len(alternate) != len(compressed)
        assert gzip.decompress(alternate) == raw
        return alternate

    monkeypatch.setattr(gzip, "compress", with_extra_header)
    assert len(catalog_rows()) == 2


def test_locally_measured_size_still_enforces_budget(monkeypatch):
    compress = gzip.compress
    monkeypatch.setattr(gzip, "compress", lambda raw, **kwargs: compress(raw, compresslevel=0, **kwargs))
    with pytest.raises(ValueError, match="size budget"):
        catalog_rows()


@pytest.mark.parametrize("size", [0, -1, True, "243628", 300 * 1024 + 1])
def test_recorded_size_must_be_a_valid_budgeted_metric(tmp_path, size):
    shutil.copytree(ROOT / "docs/assets/animations", tmp_path / "docs/assets/animations")
    path = tmp_path / "docs/assets/animations/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["animations"][0]["gzip_bytes"] = size
    path.write_bytes(encode(manifest))
    with pytest.raises(ValueError, match="size budget"):
        catalog_rows(tmp_path)


@pytest.mark.parametrize("field", ["driver", "parameters"])
def test_valid_checksum_cannot_hide_scientific_metadata_drift(tmp_path, field):
    from vaft.diagram._replay import digest

    shutil.copytree(ROOT / "docs/assets/animations", tmp_path / "docs/assets/animations")
    manifest_path = tmp_path / "docs/assets/animations/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["animations"][0]
    path = tmp_path / "docs" / row["artifact"]
    data = json.loads(path.read_text(encoding="utf-8"))
    if field == "driver":
        data[field]["unit"] = "degrees"
    else:
        data[field]["width"] = 0.5
    raw = encode(data)
    path.write_bytes(raw)
    row.update(sha256=digest(raw), bytes=len(raw), gzip_bytes=len(gzip.compress(raw, mtime=0)))
    manifest_path.write_bytes(encode(manifest))
    with pytest.raises(ValueError, match="scientific (driver|inputs) disagree"):
        catalog_rows(tmp_path)


def test_an_unchanged_regeneration_rewrites_nothing(tmp_path, monkeypatch):
    """#2117: identical bytes leave files and the producing commit alone."""
    import vaft.diagram._replay as replay

    recipe = {"builder": "exb_drift", "variant": "stub"}
    payload = {"states": [[], []], "fps": 2, "sources": []}
    monkeypatch.setattr(replay, "RECIPES", (recipe,))
    monkeypatch.setattr(replay, "make_payload", lambda recipe: payload)

    def at_commit(sha):
        monkeypatch.setattr(replay.subprocess, "check_output", lambda *a, **k: sha + "\n")

    at_commit("a" * 40)
    replay.generate(tmp_path)
    files = sorted(tmp_path.iterdir())
    stamps = {p.name: p.stat().st_mtime_ns for p in files}

    at_commit("b" * 40)
    (record,) = replay.generate(tmp_path)
    assert record["generated_commit"] == "a" * 40
    assert {p.name: p.stat().st_mtime_ns for p in files} == stamps

    payload["fps"] = 3
    (record,) = replay.generate(tmp_path)
    assert record["generated_commit"] == "b" * 40, "changed bytes take the new commit"
    assert (tmp_path / "manifest.json").stat().st_mtime_ns != stamps["manifest.json"]
