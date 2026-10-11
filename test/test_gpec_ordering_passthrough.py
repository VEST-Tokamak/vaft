"""The advisory ordering state reaches every GPEC-suite driver (#1987).

``GPECSuiteConfig.ordering_state`` feeds the ordering check each run records
(#1823); the drivers built their configs without it, so every record checked
an empty state.  :func:`vaft.code.gpec.ordering_state_for_case` builds it from
the case's own equilibrium and never raises: a run must not be lost to an
advisory record.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from vaft.code import gpec

REPO = Path(__file__).resolve().parents[1]
VEST = REPO / "vaft" / "data" / "efit" / "g039915.00319"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_gfile_yields_its_geometric_orderings_at_its_own_time():
    state, provenance = gpec.ordering_state_for_case(VEST)
    # The g-file's single slice: geometry is there, the kinetic orderings are not.
    assert state["inverse_aspect_ratio"] == pytest.approx(0.758, abs=1e-3)
    assert math.isnan(state["lundquist_number"])
    assert provenance["source"] == "g-file" and provenance["time_s"] == pytest.approx(0.319)
    assert provenance["z_eff_source"] == "not given" and "error" not in provenance
    json.dumps(provenance, allow_nan=False)  # the manifest payload must stay plain JSON


def test_a_stage_policy_zeff_is_recorded_as_such():
    _, provenance = gpec.ordering_state_for_case(VEST, z_eff=2.0)
    assert provenance["z_eff"] == 2.0 and provenance["z_eff_source"] == "stage policy"


def test_an_unreadable_equilibrium_costs_the_state_not_the_run(tmp_path):
    state, provenance = gpec.ordering_state_for_case(tmp_path / "g000000.00000")
    assert state == {} and provenance["error"].startswith("FileNotFoundError")


def test_a_stage_ods_is_matched_by_time_never_by_position():
    state, provenance = gpec.ordering_state_for_case(VEST, ods=object())
    assert state == {} and "matched by time" in provenance["error"]


def test_the_state_reaches_the_record_ordering_check(tmp_path):
    state, _ = gpec.ordering_state_for_case(VEST)
    config = gpec.GPECSuiteConfig(modules=("dcon",), modes=(1,), run_mode="prepare_only", ordering_state=state)
    result = gpec.run_gpec_suite_case(
        gpec.GPECCaseInputs(shot=39915, time_ms=319, geqdsk=VEST, workdir=tmp_path / "run"), config
    )
    (record,) = result.records
    assert record.ordering_check is not None and "error" not in record.ordering_check


def test_pipeline_1_passes_each_gfile_its_own_state(tmp_path, monkeypatch):
    driver = _load(
        "pipeline1_run_gpec_suite",
        REPO / "workflow" / "automatic_pipeline_1_routine_data_processing" / "run_gpec_suite.py",
    )
    missing = tmp_path / "g039915.00320"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{VEST}\n{missing}\n", encoding="utf-8")
    seen = []

    def fake_suite(case, config):
        seen.append((case.geqdsk, dict(config.ordering_state)))
        return gpec.GPECSuiteResult(returncode=0, workdir=tmp_path, shot=case.shot, time_ms=case.time_ms)

    monkeypatch.setattr(driver, "run_gpec_suite_case", fake_suite)
    output = tmp_path / "out" / "manifest.json"
    monkeypatch.setattr(sys, "argv", [
        "run_gpec_suite.py", "--shot", "39915", "--refined-gfile-manifest", str(manifest),
        "--output", str(output), "--status", str(tmp_path / "out" / "status.txt"), "--z-eff", "2.0",
    ])
    try:
        driver.main()
    finally:
        sys.modules.pop("pipeline1_run_gpec_suite", None)
    (first, state), (second, empty) = seen
    assert (first, second) == (VEST, missing)
    assert state["inverse_aspect_ratio"] == pytest.approx(0.758, abs=1e-3) and empty == {}
    ordering = json.loads(output.read_text(encoding="utf-8"))["ordering"]
    assert [o["geqdsk"] for o in ordering] == [str(VEST), str(missing)]
    assert ordering[0]["z_eff"] == 2.0 and "error" in ordering[1]


def test_the_stability_atlas_jobs_pass_the_state(tmp_path, monkeypatch):
    scan = _load("stability_scan_controls_1987", REPO / "workflow" / "stability_atlas" / "scan_controls.py")
    try:
        eq = scan.Equilibrium(39915, 319, VEST)
        variant = next(v for v in scan.VARIANTS if v.name == "stride_base")
        seen = []

        def fake_suite(inputs, config):
            seen.append(dict(config.ordering_state))
            raise RuntimeError("stop after the config is built")

        monkeypatch.setattr(scan, "run_gpec_suite_case", fake_suite)
        with pytest.raises(RuntimeError, match="config is built"):
            scan.run_job(eq, variant, 1, tmp_path, tmp_path, 10.0)
    finally:
        sys.modules.pop("stability_scan_controls_1987", None)
    assert seen[0]["inverse_aspect_ratio"] == pytest.approx(0.758, abs=1e-3)
