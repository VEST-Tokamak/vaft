"""Unit tests for the free-boundary PF-coil scan framework (OFT-free)."""

import json

import numpy as np
import pytest
from omas import ODS

from tokamaker_fakes import make_fake_oft

from vaft.code.tokamaker import TokaMakerConfig
from vaft.code.tokamaker import free_boundary as fb


def _build_ods():
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.2, 0.6, 0.6, 0.2])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-0.4, -0.4, 0.4, 0.4])
    for i, (name, rc, cur) in enumerate((("PF1", 0.10, -2000.0), ("PF2", 0.70, 1000.0))):
        ods[f"pf_active.coil.{i}.name"] = name
        for j, z in enumerate((+0.5, -0.5)):
            base = f"pf_active.coil.{i}.element.{j}"
            ods[f"{base}.geometry.rectangle.r"] = rc
            ods[f"{base}.geometry.rectangle.z"] = z
            ods[f"{base}.geometry.rectangle.width"] = 0.04
            ods[f"{base}.geometry.rectangle.height"] = 0.10
            ods[f"{base}.turns_with_sign"] = 8.0
        ods[f"pf_active.coil.{i}.current.data"] = np.array([0.0, cur])
    ods["pf_active.time"] = np.array([0.0, 1.0])
    ods["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    ods["magnetics.ip.0.data"] = np.array([0.0, 100.0e3])
    ods["tf.time"] = np.array([0.0, 1.0])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.06, 0.06])
    return ods


def _config(tmp_path, **overrides):
    mesh = tmp_path / "mesh.h5"
    mesh.write_bytes(b"")
    kwargs = dict(
        shot=39915, time=0.32, workdir=tmp_path, mesh_file=mesh,
        constraint_source="magnetics",
    )
    kwargs.update(overrides)
    return TokaMakerConfig(**kwargs)


def _scan(tmp_path, monkeypatch=None, **kwargs):
    defaults = dict(
        controls={"PF2": {"offset_A": [-100.0, 0.0, 100.0]}},
        config=_config(tmp_path),
        workdir=tmp_path / "scan",
    )
    defaults.update(kwargs)
    return fb.scan(_build_ods(), **defaults)


# --------------------------------------------------------------------------- #
#  Materialization
# --------------------------------------------------------------------------- #
def test_product_mode_crosses_axes_with_commanded_math(tmp_path):
    scan = _scan(
        tmp_path,
        controls={
            "PF2": {"offset_A": [-100.0, 100.0]},
            "PF1": {"scale": [1.0, 1.5]},
        },
    )
    # baseline at t=0.32: PF1 = -640 A, PF2 = 320 A
    assert len(scan.cases) == 4
    commanded = [dict(case.commanded) for case in scan.cases]
    assert commanded[0] == {"PF1": pytest.approx(-640.0), "PF2": pytest.approx(220.0)}
    assert commanded[-1] == {"PF1": pytest.approx(-960.0), "PF2": pytest.approx(420.0)}
    # requested records the raw control, commanded the applied current
    assert scan.cases[0].requested == {"PF1": {"scale": 1.0}, "PF2": {"offset_A": -100.0}}
    # deterministic distinct ids and per-case config identity
    ids = [case.case_id for case in scan.cases]
    assert len(set(ids)) == 4
    shas = {case.config_sha for case in scan.cases}
    assert len(shas) == 4


def test_zip_mode_is_a_current_trajectory(tmp_path):
    scan = _scan(
        tmp_path,
        mode="zip",
        controls={
            "PF2": {"absolute_A": [300.0, 350.0]},
            "PF1": {"offset_A": [0.0, -50.0]},
        },
    )
    assert len(scan.cases) == 2
    assert scan.cases[1].commanded == {
        "PF1": pytest.approx(-690.0), "PF2": pytest.approx(350.0)
    }


def test_materialization_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="zip"):
        _scan(tmp_path, mode="zip", controls={
            "PF1": {"offset_A": [0.0]}, "PF2": {"offset_A": [0.0, 1.0]},
        })
    with pytest.raises(ValueError, match="control mode"):
        _scan(tmp_path, controls={"PF2": {"delta": [1.0]}})
    with pytest.raises(ValueError, match="not a coil set"):
        _scan(tmp_path, controls={"PF77": {"offset_A": [0.0]}})
    with pytest.raises(ValueError, match="Unknown hold"):
        _scan(tmp_path, hold=("ip", "beta"))
    with pytest.raises(ValueError, match="pax"):
        _scan(tmp_path, hold=("ip", "pax"))


def test_dry_run_writes_pending_manifests(tmp_path):
    scan = _scan(tmp_path)
    result = scan.dry_run()

    assert all(case.status is fb.CaseStatus.PENDING for case in result.cases)
    payload = json.loads(result.manifest.read_text())
    assert payload["schema_version"] == 1
    assert payload["solver"] == "tokamaker"
    assert [c["status"] for c in payload["cases"]] == ["pending"] * 3
    case_payload = json.loads(result.cases[0].manifest.read_text())
    assert case_payload["status"] == "pending"
    assert case_payload["config_sha256"] == scan.cases[0].config_sha


# --------------------------------------------------------------------------- #
#  Execution
# --------------------------------------------------------------------------- #
def test_run_lifecycle_and_manifests(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    scan = _scan(tmp_path)
    result = scan.run()

    names = [entry[0] for entry in calls]
    # one solver lifecycle for the whole scan
    assert names.count("init") == 1
    assert names.count("setup") == 1
    assert names.count("set_profiles") == 1
    assert names.count("reset") == 1 and names[-1] == "reset"
    # continuation: only the first case cold-starts
    assert names.count("init_psi") == 1
    assert names.count("solve") == 3

    assert [case.status.value for case in result.cases] == ["succeeded"] * 3
    case = result.cases[1]
    assert case.commanded_currents["PF2"] == pytest.approx(320.0)
    assert case.materialized_currents["PF1"] == pytest.approx(-640.0)
    assert case.achieved["Ip"] == pytest.approx(51.0e3)
    assert case.solver_x_points == ((0.45, -0.35),)
    assert case.solver_diverted is True
    assert case.gfile is not None and case.gfile.is_file()
    # fake g-files are unreadable -> classification degrades to unknown
    assert case.topology["topology"] == "unknown"
    assert result.cases[1].continuation_from == result.cases[0].case_id

    payload = json.loads(case.manifest.read_text())
    assert payload["status"] == "succeeded"
    assert payload["commanded_currents_A"]["PF2"] == pytest.approx(320.0)
    assert payload["held"]["ip"] == pytest.approx(32.0e3)  # magnetics Ip at 0.32
    scan_payload = json.loads(result.manifest.read_text())
    assert [c["status"] for c in scan_payload["cases"]] == ["succeeded"] * 3


def test_cold_start_mode_reinitializes_every_case(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    result = _scan(tmp_path, continuation=False).run()
    names = [entry[0] for entry in calls]
    assert names.count("init_psi") == 3
    assert all(case.continuation_from is None for case in result.cases)


def test_failure_is_recorded_and_chain_recovers(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(
        monkeypatch, solve_error='Error in solve: Exceeded "maxits"', solve_error_at=2
    )
    result = _scan(tmp_path).run()

    statuses = [case.status for case in result.cases]
    assert statuses == [
        fb.CaseStatus.SUCCEEDED, fb.CaseStatus.NOT_CONVERGED, fb.CaseStatus.SUCCEEDED,
    ]
    failed = result.cases[1]
    assert "maxits" in failed.error
    assert failed.gfile is None and failed.topology is None
    # the diverged iterate is dropped before the next case
    names = [entry[0] for entry in calls]
    assert "set_psi" in names
    # failed case stays visible in the scan manifest
    payload = json.loads(result.manifest.read_text())
    assert [c["status"] for c in payload["cases"]] == [
        "succeeded", "not_converged", "succeeded",
    ]


def test_unexpected_exception_maps_to_failed(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch, solve_error="disk on fire", solve_error_at=2)
    result = _scan(tmp_path).run()
    assert result.cases[1].status is fb.CaseStatus.FAILED


def test_refinement_bisects_and_recovers_the_target(tmp_path, monkeypatch):
    # target case fails once (solve #2); the midpoint (#3) and the retried
    # target (#4) converge
    make_fake_oft(
        monkeypatch, solve_error='Exceeded "maxits"', solve_error_at=2
    )
    result = _scan(tmp_path, refine_on_failure=2).run()

    case = result.cases[1]
    assert case.status is fb.CaseStatus.SUCCEEDED
    assert len(case.refinement_history) == 2
    assert case.refinement_history[0]["converged"] is True
    assert case.refinement_history[1]["target_retry"] is True
    # midpoint currents sit between the last-good and target commanded values
    mid = case.refinement_history[0]["commanded"]["PF2"]
    assert mid == pytest.approx(0.5 * (220.0 + 320.0))


def test_refinement_halves_toward_origin_while_midpoints_fail(tmp_path, monkeypatch):
    # every solve at PF2 >= 270 A fails: the target case (320 A) fails, the
    # first midpoint (270 A) fails, and the bisection must then probe the
    # quarter point (245 A) instead of retrying the same midpoint
    make_fake_oft(monkeypatch)
    original = fb.FreeBoundaryScan._solve_currents

    def gated(self, mygs, currents, *, cold):
        if currents["PF2"] >= 270.0:
            raise ValueError('Exceeded "maxits"')
        return original(self, mygs, currents, cold=cold)

    monkeypatch.setattr(fb.FreeBoundaryScan, "_solve_currents", gated)
    result = _scan(tmp_path, refine_on_failure=2).run()

    case = result.cases[1]                       # commanded PF2 = 320 A
    assert case.status is fb.CaseStatus.NOT_CONVERGED
    midpoints = [
        entry["commanded"]["PF2"] for entry in case.refinement_history
        if not entry.get("target_retry")
    ]
    assert midpoints == [pytest.approx(270.0), pytest.approx(245.0)]
    assert case.refinement_history[0]["converged"] is False
    # the quarter point converged; only the target retry after it failed
    converged_flags = {
        round(entry["commanded"]["PF2"]): entry["converged"]
        for entry in case.refinement_history
    }
    assert converged_flags[245] is True
    assert converged_flags[320] is False
    # the case below the failing region (220 A) solved normally, and the one
    # even deeper inside it (420 A) is honestly not_converged
    assert result.cases[0].status is fb.CaseStatus.SUCCEEDED
    assert result.cases[2].status is fb.CaseStatus.NOT_CONVERGED


def test_dry_run_preserves_completed_state(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    _scan(tmp_path).run()

    result = _scan(tmp_path).dry_run()
    assert [case.status.value for case in result.cases] == ["succeeded"] * 3
    scan_payload = json.loads(result.manifest.read_text())
    assert [c["status"] for c in scan_payload["cases"]] == ["succeeded"] * 3
    case_payload = json.loads(result.cases[0].manifest.read_text())
    assert case_payload["status"] == "succeeded"
    assert case_payload["achieved"]["Ip"] == pytest.approx(51.0e3)

    # resume after the dry run still reloads everything without solving
    calls, _ = make_fake_oft(monkeypatch)
    second = _scan(tmp_path).run(resume=True)
    assert "solve" not in [entry[0] for entry in calls]
    assert all(case.ok for case in second.cases)


def test_resume_reclassifies_when_thresholds_change(tmp_path, monkeypatch):
    from vaft.code.tokamaker.topology import ScanTopology, TopologyReport

    make_fake_oft(monkeypatch)
    _scan(tmp_path).run()          # topology stored with the default tolerances

    calls, _ = make_fake_oft(monkeypatch)
    monkeypatch.setattr(
        fb, "classify_boundary",
        lambda *a, **k: TopologyReport(
            topology=ScanTopology.LIMITED,
            active_tolerance=k["active_tolerance"],
            near_null_band=k["near_null_band"],
        ),
    )
    result = _scan(tmp_path, active_tolerance=1.0e-2).run(resume=True)

    assert "solve" not in [entry[0] for entry in calls]   # still no re-solve
    for case in result.cases:
        assert case.topology["topology"] == "limited"
        assert case.topology["active_tolerance"] == pytest.approx(1.0e-2)
        payload = json.loads(case.manifest.read_text())
        assert payload["topology"]["topology"] == "limited"
        assert payload["topology"]["active_tolerance"] == pytest.approx(1.0e-2)


def test_resume_reclassification_failure_keeps_the_stored_topology(tmp_path, monkeypatch):
    from vaft.code.tokamaker.topology import ScanTopology, TopologyReport

    def _report(topology, **kwargs):
        return TopologyReport(
            topology=topology,
            active_tolerance=kwargs["active_tolerance"],
            near_null_band=kwargs["near_null_band"],
            reason="transient" if topology is ScanTopology.UNKNOWN else None,
        )

    make_fake_oft(monkeypatch)
    monkeypatch.setattr(
        fb, "classify_boundary", lambda *a, **k: _report(ScanTopology.LIMITED, **k)
    )
    _scan(tmp_path).run()          # stored: limited @ default tolerances

    # a resume with new thresholds hits a transient classification failure:
    # the good stored report must survive, with its OLD tolerances
    make_fake_oft(monkeypatch)
    monkeypatch.setattr(
        fb, "classify_boundary", lambda *a, **k: _report(ScanTopology.UNKNOWN, **k)
    )
    result = _scan(tmp_path, active_tolerance=1.0e-2).run(resume=True)
    for case in result.cases:
        assert case.topology["topology"] == "limited"
        assert case.topology["active_tolerance"] == pytest.approx(2.0e-3)

    # once the failure clears, the next resume re-classifies to the new knobs
    make_fake_oft(monkeypatch)
    monkeypatch.setattr(
        fb, "classify_boundary", lambda *a, **k: _report(ScanTopology.LIMITED, **k)
    )
    result = _scan(tmp_path, active_tolerance=1.0e-2).run(resume=True)
    for case in result.cases:
        assert case.topology["active_tolerance"] == pytest.approx(1.0e-2)
        payload = json.loads(case.manifest.read_text())
        assert payload["topology"]["active_tolerance"] == pytest.approx(1.0e-2)


def test_resume_reloads_succeeded_cases_without_solving(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    scan = _scan(tmp_path)
    first = scan.run()
    assert all(case.ok for case in first.cases)

    calls, _ = make_fake_oft(monkeypatch)
    second = _scan(tmp_path).run(resume=True)
    names = [entry[0] for entry in calls]
    assert "solve" not in names                      # everything reloaded
    assert [case.status.value for case in second.cases] == ["succeeded"] * 3
    assert second.cases[2].achieved["Ip"] == pytest.approx(51.0e3)


def test_resume_reruns_when_the_config_changed(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    _scan(tmp_path).run()

    calls, _ = make_fake_oft(monkeypatch)
    changed = _scan(tmp_path, config=_config(tmp_path, maxits=222))
    changed.run(resume=True)
    names = [entry[0] for entry in calls]
    assert names.count("solve") == 3                 # sha mismatch -> full rerun


def test_discontinuity_flags_topology_and_drsep_jumps(tmp_path, monkeypatch):
    make_fake_oft(monkeypatch)
    from vaft.code.tokamaker.topology import ScanTopology, TopologyReport

    sequence = iter([
        TopologyReport(topology=ScanTopology.LIMITED, d_r_sep=None),
        TopologyReport(topology=ScanTopology.NEAR_NULL, d_r_sep=0.04),
        TopologyReport(topology=ScanTopology.LOWER_SINGLE_NULL, d_r_sep=-0.01),
    ])
    monkeypatch.setattr(fb, "classify_boundary", lambda *a, **k: next(sequence))

    result = _scan(tmp_path).run()

    assert result.cases[1].discontinuity["topology_changed"] is True
    assert result.cases[2].discontinuity["topology_changed"] is True
    assert result.cases[2].discontinuity["d_r_sep_jump_m"] == pytest.approx(0.05)
    assert result.cases[2].discontinuity["flagged"] is True   # > 0.02 threshold
    assert result.cases[0].discontinuity == {"reference": None, "flagged": False}
