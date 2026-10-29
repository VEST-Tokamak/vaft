"""The frozen-baseline capture script's contract.

``workflow/architecture_audit/capture_baseline.py`` records, per ODS leaf, what
the pre-migration stage chain produces from the packaged raw dumps.  These are
the fast checks of its pieces -- the leaf hash is canonical, the output root
is isolated, the children cannot reach HSDS, and the stage argv mirrors the
pipeline-1 Snakefile.  The end-to-end run on 39915 lives in
``test_consolidation_baseline_capture_39915.py`` (``slow``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workflow" / "architecture_audit" / "capture_baseline.py"


@pytest.fixture(scope="module")
def capture():
    spec = importlib.util.spec_from_file_location("capture_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_leaf_hash_distinguishes_dtype_shape_and_value(capture):
    hashes = {
        capture.leaf_record(value)["sha256"]
        for value in (1.0, [1.0], np.array([1], dtype=np.int64), np.array([1.0, 0.0]), np.array([1.0, -0.0]))
    }
    assert len(hashes) == 5  # -0.0 kept distinct: a sign flip at zero is still one


def test_a_leaf_hash_ignores_nan_payload_and_byte_order(capture):
    quiet = np.array([1.0, np.nan])
    payload = np.array([1.0, 0.0])
    payload.view(np.uint64)[1] = 0x7FF8000000000123  # a NaN with a payload
    assert np.isnan(payload[1])
    assert capture.leaf_record(quiet)["sha256"] == capture.leaf_record(payload)["sha256"]
    big = np.array([1.5, 2.5], dtype=">f8")
    assert capture.leaf_record(big)["sha256"] == capture.leaf_record(np.array([1.5, 2.5]))["sha256"]


def test_a_numeric_leaf_carries_its_summary_statistics(capture):
    record = capture.leaf_record(np.array([[3.0, -4.0], [np.nan, 0.0]]))
    assert record["shape"] == [2, 2]
    assert record["n_finite"] == 3 and record["n_nan"] == 1
    assert record["min"] == -4.0 and record["max"] == 3.0
    assert record["mean"] == pytest.approx(-1.0 / 3.0)
    assert record["rms"] == pytest.approx(np.sqrt(25.0 / 3.0))


def test_a_string_leaf_hashes_its_utf8(capture):
    import hashlib

    assert capture.leaf_record("Flux Loop - #10")["sha256"] == hashlib.sha256(b"Flux Loop - #10").hexdigest()


def test_the_output_root_must_be_isolated(capture, tmp_path):
    with pytest.raises(capture.IsolationError):
        capture.check_output_root(ROOT / "vaft" / "data" / "samples" / "39915" / "baseline")
    with pytest.raises(capture.IsolationError):
        capture.check_output_root(ROOT / "test" / "data" / "baseline")  # anywhere in the checkout
    with pytest.raises(capture.IsolationError):
        capture.check_output_root(Path("/srv/vest.filedb/public/baseline"))
    filedb = tmp_path / "filedb"
    with pytest.raises(capture.IsolationError):
        capture.check_output_root(filedb / "x", environ={"VAFT_FILEDB_DIR": str(filedb)})
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "product.json").write_text("{}")
    with pytest.raises(capture.IsolationError, match="not empty"):
        capture.check_output_root(occupied, environ={})
    assert capture.check_output_root(tmp_path / "fresh", environ={}) == (tmp_path / "fresh").resolve()


def test_the_children_import_this_checkout_and_cannot_reach_hsds(capture, tmp_path):
    env = capture.child_environment(
        tmp_path,
        environ={"HS_ENDPOINT": "http://vestserver", "HS_USERNAME": "u", "H5PYD_CONFIG": "x",
                 "HOME": "/Users/someone", "PATH": "/bin"},
    )
    # The caller's credentials are gone and the endpoint is unroutable; an
    # environment endpoint overrides any .hscfg, and HOME no longer reaches
    # the caller's ~/.hscfg.
    assert sorted(key for key in env if key.startswith(("HS_", "H5PYD_"))) == ["HS_ENDPOINT"]
    assert env["HS_ENDPOINT"] == capture.UNROUTABLE_HSDS_ENDPOINT
    assert env["HOME"] == str(tmp_path / "home")
    assert env["PYTHONPATH"].split(":")[0] == str(ROOT)
    assert env["VAFT_FILEDB_DIR"] == str(tmp_path / "filedb")
    assert env["PATH"] == "/bin"


def test_the_stage_argv_carries_every_snakefile_flag(capture, tmp_path):
    """The flags the pipeline-1 Snakefile passes, read from the Snakefile itself."""
    import re

    import yaml

    snakefile = (ROOT / "workflow" / "automatic_pipeline_1_routine_data_processing" / "Snakefile").read_text()
    config = yaml.safe_load((ROOT / "vaft/data/samples/39915/source/pipeline-config.yaml").read_text())
    commands = {stage: argv for stage, argv, _ in capture.stage_commands(39915, config, "era", tmp_path)}
    for stage in capture.STAGES:
        rule = snakefile.split(f"rule generate_{stage}_ods:")[1].split("\nrule ")[0]
        shell = rule.split('"""')[1]
        expected = set(re.findall(r"(--[a-z][a-z-]+)", shell))
        assert expected, stage
        assert expected <= set(commands[stage]), (stage, sorted(expected - set(commands[stage])))


def test_only_shots_with_packaged_raw_are_accepted(capture, tmp_path):
    with pytest.raises(ValueError, match="no packaged raw dump"):
        capture.capture_shot(48224, tmp_path)
    for shot in capture.SHOTS_WITH_RAW:
        assert (ROOT / "vaft/data/samples" / str(shot) / "source" / f"vest_{shot}_daq_raw.json.gz").is_file()


def test_the_summary_keeps_arrays_groups_scalars_and_dedupes_static_copies(capture):
    leaves = {
        "pf_active.coil.0.current.data": capture.leaf_record(np.array([1.0, 2.0, 3.0])),
        "pf_active.coil.0.name": capture.leaf_record("PF1"),
        "equilibrium.time_slice.0.constraints.ip.measured": capture.leaf_record(np.float64(1.0e5)),
        "equilibrium.time_slice.1.constraints.ip.measured": capture.leaf_record(np.float64(3.0e5)),
    }
    static = {"pf_active.coil.0.name": capture.leaf_record("PF1")}
    summary = capture.summarize_stage(leaves, reference=static)
    assert summary["leaf_count"] == 4 and summary["inherited_from_static"] == 1
    sha, shape, n_finite, low, high, mean, rms = summary["arrays"]["pf_active.coil.0.current.data"]
    assert sha == leaves["pf_active.coil.0.current.data"]["sha256"][:16]
    assert shape == [3] and (n_finite, low, high, mean) == (3, 1.0, 3.0, 2.0)
    group = summary["scalar_groups"]["equilibrium.time_slice.:.constraints.ip.measured"]
    assert group[1:6] == [2, 2, 1.0e5, 3.0e5, 2.0e5]

    # Any change to one element changes its group digest.
    changed = dict(leaves)
    changed["equilibrium.time_slice.1.constraints.ip.measured"] = capture.leaf_record(np.float64(3.0e5 + 1.0))
    other = capture.summarize_stage(changed, reference=static)
    assert other["scalar_groups"]["equilibrium.time_slice.:.constraints.ip.measured"][0] != group[0]


def test_summarize_root_refuses_an_unverifiable_capture(capture, tmp_path):
    import json

    with pytest.raises(RuntimeError, match="no <shot>/baseline.json"):
        capture.summarize_root(tmp_path, code_sha="HEAD")

    def write(shot, commit, dirty):
        leaves = {"pf_active.coil.0.current.data": capture.leaf_record(np.arange(3.0))}
        record = {
            "shot": shot, "machine_version": "era", "inputs": {}, "revision": {"commit": commit, "dirty": dirty},
            "stages": {
                "static": {"status": "captured", "product_sha256": "s", "leaves": leaves},
                "eddy": {"status": "captured", "product_sha256": "e", "leaves": leaves},
            },
        }
        (tmp_path / str(shot)).mkdir(exist_ok=True)
        (tmp_path / str(shot) / "baseline.json").write_text(json.dumps(record))

    write(1, "HEAD", True)
    with pytest.raises(RuntimeError, match="dirty"):
        capture.summarize_root(tmp_path, code_sha="HEAD")
    write(1, "HEAD", False)
    write(2, "HEAD~1", False)
    with pytest.raises(RuntimeError, match="more than one revision"):
        capture.summarize_root(tmp_path, code_sha="HEAD")
    write(2, "HEAD", False)
    summary = capture.summarize_root(tmp_path, code_sha="HEAD")
    assert sorted(summary["shots"]) == ["1", "2"]
    assert summary["shots"]["1"]["stages"]["eddy"]["inherited_from_static"] == 1
    assert list(summary["static_by_era"]) == ["era"]
