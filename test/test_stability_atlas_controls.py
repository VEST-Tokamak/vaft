"""The #141 control-scan driver: variants, template patching, the result table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "workflow" / "stability_atlas"


@pytest.fixture(scope="module")
def scan():
    spec = importlib.util.spec_from_file_location("stability_scan_controls", ROOT / "scan_controls.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_variant_names_are_unique_and_name_their_module(scan):
    names = [v.name for v in scan.VARIANTS]
    assert len(names) == len(set(names))
    for variant in scan.VARIANTS:
        assert variant.name.startswith(variant.module + "_")
        assert variant.module in {"dcon", "rdcon", "stride"}


def test_every_dcon_variant_evaluates_ballooning(scan):
    # The packaged dcon.in ships bal_flag=f, which leaves ca1 identically zero.
    assert all(v.dcon.bal_flag for v in scan.VARIANTS if v.module == "dcon")


def test_patched_templates_change_only_the_named_keys(scan, tmp_path):
    variant = next(v for v in scan.VARIANTS if v.name == "rdcon_mpsi256")
    target = scan.materialize_templates(variant, tmp_path)
    package = scan.package_vest_dir()
    assert sorted(p.name for p in target.iterdir()) == sorted(p.name for p in package.iterdir())
    for path in package.iterdir():
        original = path.read_bytes().splitlines()
        patched = (target / path.name).read_bytes().splitlines()
        changed = [(a, b) for a, b in zip(original, patched) if a != b]
        if path.name == "equil.in":
            assert len(changed) == 1 and b"mpsi=256" in changed[0][1].replace(b" ", b"")
        else:
            assert patched == original, path.name


def test_existing_templates_are_reused_but_never_silently_replaced(scan, tmp_path):
    variant = next(v for v in scan.VARIANTS if v.name == "rdcon_nx128")
    first = scan.materialize_templates(variant, tmp_path)
    assert scan.materialize_templates(variant, tmp_path) == first
    edited = scan.Variant("rdcon_nx128", "rdcon", {"rdcon.in": {"nx": 64}})
    with pytest.raises(ValueError, match="use a new --out"):
        scan.materialize_templates(edited, tmp_path)
    assert "nx=128" in (first / "rdcon.in").read_text().replace(" ", "")


def test_a_failed_job_is_retried_and_a_finished_one_is_not(scan, tmp_path, monkeypatch):
    eq = scan.Equilibrium(39915, 319, tmp_path / "g039915.00319")
    variant = next(v for v in scan.VARIANTS if v.name == "stride_base")
    jobdir = tmp_path / eq.label / variant.name / "nn1"
    jobdir.mkdir(parents=True)
    calls = []

    def fake_suite(inputs, config):
        calls.append(config.timeout)
        raise RuntimeError("stop after the cache check")

    monkeypatch.setattr(scan, "run_gpec_suite_case", fake_suite)
    (jobdir / "result.json").write_text(json.dumps({"status": "stable", "wall_s": 1.0}))
    assert scan.run_job(eq, variant, 1, tmp_path, tmp_path, 10.0)["status"] == "stable"
    assert calls == []
    (jobdir / "result.json").write_text(json.dumps({"status": "failed", "reason": "timeout after 10 seconds"}))
    with pytest.raises(RuntimeError, match="stop after the cache check"):
        scan.run_job(eq, variant, 1, tmp_path, tmp_path, 7200.0)
    assert calls == [7200.0]


def test_rows_record_the_controls_they_ran_with(scan):
    variant = next(v for v in scan.VARIANTS if v.name == "dcon_psiedge0950")
    recorded = scan.controls(variant)
    assert recorded["dcon"]["psiedge"] == 0.95 and recorded["dcon"]["bal_flag"] is True
    assert recorded["psihigh"] == 0.994


def test_a_misspelled_key_is_refused_not_silently_ignored(scan, tmp_path):
    bad = scan.Variant("rdcon_typo", "rdcon", {"rdcon.in": {"nxx": 128}})
    with pytest.raises(KeyError):
        scan.materialize_templates(bad, tmp_path)


def test_parse_equilibrium(scan, tmp_path):
    eq = scan.parse_equilibrium(f"39915:319:{tmp_path}/g039915.00319")
    assert (eq.shot, eq.time_ms, eq.label) == (39915, 319, "39915.00319")


def test_summarize_unions_columns_across_modules(scan, tmp_path):
    for label, variant, row in (
        ("39915.00319", "dcon_base", {"module": "dcon", "W_t_min": 2.69}),
        ("39915.00319", "rdcon_base", {"module": "rdcon", "msing": 9}),
    ):
        path = tmp_path / label / variant / "nn1"
        path.mkdir(parents=True)
        (path / "result.json").write_text(json.dumps({"variant": variant, **row}))
    header = scan.summarize(tmp_path).read_text().splitlines()[0].split(",")
    assert {"variant", "module", "W_t_min", "msing"} <= set(header)
