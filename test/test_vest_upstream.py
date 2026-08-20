import gzip
import json

import numpy as np
import pytest
from scipy import ndimage, signal

from vaft.database import raw as raw_db
from vaft.database._local import load_ods
from vaft.omas.vest_upstream import (
    archive_raw_source,
    build_diagnostics_ods,
    build_static_ods,
    machine_era_for_shot,
    write_stage_product,
)
from vaft.machine_mapping.pf_active import vfit_pf
from vaft.process.magnetics import VestMagneticsProcessingConfig, vest_md_signals


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        (43016, "vest-pre-43017-pf1906"),
        (43017, "vest-43017-45957-pf1906"),
        (45957, "vest-43017-45957-pf1906"),
        (45958, "vest-45958-45966-pf2507"),
        (45966, "vest-45958-45966-pf2507"),
        (45967, "vest-45967-plus-pf2507"),
    ],
)
def test_machine_era_boundaries_are_explicit(shot, expected):
    assert machine_era_for_shot(shot).name == expected


def test_static_product_is_not_a_reference_shot_container():
    ods, manifest = build_static_ods("vest-45958-45966-pf2507")

    assert set(ods.keys()) == {
        "wall",
        "pf_active",
        "pf_passive",
        "em_coupling",
        "magnetics",
        "tf",
    }
    assert "dataset_description" not in ods
    assert "pf_active.time" not in ods
    assert "pf_passive.time" not in ods
    assert "pf_passive.loop.0.current" not in ods
    assert np.shape(ods["em_coupling.mutual_passive_passive"]) == (950, 950)
    assert manifest["channel_status"]["pf_active"]["disabled_channels"] == [
        "PF2",
        "PF3",
        "PF4",
        "PF7",
        "PF8",
    ]


def _write_raw_dump(path, shot, fields):
    payload = {
        "shot": shot,
        "fields": {
            str(field): {"data": values, "type": "slow"}
            for field, values in fields.items()
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_unavailable_diagnostic_does_not_corrupt_valid_sibling(tmp_path):
    shot = 43017
    raw = tmp_path / "raw.json.gz"
    _write_raw_dump(raw, shot, {13: np.linspace(1.0, 2.0, 200).tolist()})
    static_path = tmp_path / "static.json.gz"
    static_manifest = tmp_path / "static-manifest.json"
    static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    write_stage_product(
        static,
        manifest,
        output=static_path,
        metadata=static_manifest,
    )

    ods, diagnostics_manifest = build_diagnostics_ods(
        shot=shot,
        raw_source=raw,
        static_ods=static_path,
        tstart=0.0,
        tend=0.005,
        dt=4e-5,
    )

    assert diagnostics_manifest["status"] == "partial"
    assert diagnostics_manifest["channel_status"]["barometry"]["status"] == "success"
    for name in ("pf_active", "spectrometer_uv", "tf", "magnetics"):
        assert diagnostics_manifest["channel_status"][name]["status"] == "unavailable"
    assert "barometry" in ods
    assert "tf" not in ods
    assert np.all(np.asarray(ods["barometry.gauge.0.pressure.data"]) > 0)


def test_explicit_raw_archive_is_copied_with_machine_readable_provenance(tmp_path):
    source = tmp_path / "source.json.gz"
    output = tmp_path / "raw" / "output" / "vest_daq_raw.json.gz"
    _write_raw_dump(source, 39915, {13: [1.0, 2.0]})

    manifest = archive_raw_source(shot=39915, source=source, output=output)

    assert output.read_bytes() == source.read_bytes()
    assert manifest["source"] == {"kind": "archive", "name": source.name}
    assert manifest["output"]["sha256"]


def test_plain_raw_archive_is_normalized_to_the_gzip_workflow_product(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"shot": 39915, "fields": {"13": {"data": [1.0]}}}),
        encoding="utf-8",
    )
    output = tmp_path / "vest_daq_raw.json.gz"

    archive_raw_source(shot=39915, source=source, output=output)

    with gzip.open(output, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["shot"] == 39915
    assert set(payload["fields"]) == {"13"}


def test_live_raw_export_honors_a_plain_json_output(tmp_path, monkeypatch):
    output = tmp_path / "vest_daq_raw.json"

    def fake_dump(shot, path):
        assert shot == 39915
        assert path.endswith(".json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"shot": shot, "fields": {"13": {"data": [1.0]}}}, handle)
        return True

    monkeypatch.setattr(raw_db, "dump_all_raw_signals_for_shot", fake_dump)

    manifest = archive_raw_source(shot=39915, output=output)

    assert json.loads(output.read_text(encoding="utf-8"))["shot"] == 39915
    assert manifest["source"] == {"kind": "vest-sql", "name": None}
    assert sorted(path.name for path in tmp_path.iterdir()) == [output.name]


def test_explicit_raw_archive_rejects_a_different_shot(tmp_path):
    source = tmp_path / "source.json.gz"
    _write_raw_dump(source, 39916, {13: [1.0, 2.0]})

    with pytest.raises(ValueError, match="shot mismatch"):
        archive_raw_source(
            shot=39915,
            source=source,
            output=tmp_path / "vest_daq_raw.json.gz",
        )


def test_stage_writer_produces_reloadable_gzip_ods(tmp_path):
    ods, manifest = build_static_ods("vest-pre-43017-pf1906")
    output = tmp_path / "static.json.gz"
    metadata = tmp_path / "manifest.json"
    write_stage_product(ods, manifest, output=output, metadata=metadata)

    reloaded, _ = load_ods(output)
    assert len(reloaded["pf_passive.loop"]) == 950
    recorded = json.loads(metadata.read_text(encoding="utf-8"))
    assert recorded["output"]["name"] == "static.json.gz"
    assert recorded["output"]["sha256"]


def test_stage_writer_is_byte_deterministic(tmp_path):
    ods, manifest = build_static_ods("vest-pre-43017-pf1906")
    outputs = []
    for index in range(2):
        output = tmp_path / str(index) / "static.json.gz"
        write_stage_product(
            ods,
            manifest,
            output=output,
            metadata=tmp_path / str(index) / "manifest.json",
        )
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_pf_current_processing_matches_the_reference_filter(tmp_path):
    shot = 39915
    source = tmp_path / "raw.json.gz"
    samples = np.sin(np.linspace(0.0, 20.0, 1200)) + np.linspace(0.0, 0.2, 1200)
    _write_raw_dump(
        source,
        shot,
        {field: samples.tolist() for field in (5, 59, 62, 65)},
    )

    _time, currents = vfit_pf(shot, raw_source=source)
    centered = samples - np.mean(samples)
    taps = signal.firwin(251, 2500.0, pass_zero="lowpass", fs=25_000.0)
    expected = ndimage.uniform_filter1d(
        signal.filtfilt(taps, 1, centered), size=10
    ) * -5e4
    np.testing.assert_allclose(currents[0], expected)


def test_missing_md_channel_keeps_later_channel_in_its_ordered_slot():
    time = np.arange(1200, dtype=float) * 4e-6
    waveform = np.sin(np.linspace(0.0, 10.0, time.size))
    channels = [
        {"field_code": 1, "kind": "b_field_pol_probe", "calibration": 1.0},
        {"field_code": 2, "kind": "b_field_pol_probe", "calibration": 1.0},
    ]

    _target_time, _flux, probes = vest_md_signals(
        39915,
        channels,
        lambda _shot, field: None if field == 1 else (time, waveform),
        config=VestMagneticsProcessingConfig(),
        allow_missing=True,
    )

    assert len(probes) == 2
    assert probes[0].size == 0
    assert probes[1].size > 0
