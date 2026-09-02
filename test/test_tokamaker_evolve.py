"""Unit tests for the quasi-static evolution loop and its preparation (OFT-free)."""

import json

import numpy as np
import pytest
from omas import ODS

from tokamaker_fakes import make_fake_oft, make_inputs

from vaft.code.tokamaker import (
    TokaMakerConfig,
    prepare_tokamaker_evolution_inputs,
    run_tokamaker_evolution,
)
from vaft.code.tokamaker.config import TokaMakerEvolutionInputs


def _evolution_inputs(tmp_path, times=(0.300, 0.302, 0.304), vacuum=False):
    base = make_inputs(tmp_path)
    n = len(times)
    return TokaMakerEvolutionInputs(
        base=base,
        times=tuple(times),
        coil_waveforms={
            "PF1": np.linspace(-600.0, -400.0, n),
            "PF2": np.linspace(300.0, 350.0, n),
        },
        ip_targets=np.linspace(50.0e3, 52.0e3, n),
        vacuum=vacuum,
    )


def _config(tmp_path, **overrides):
    kwargs = dict(shot=39915, time=0.300, workdir=tmp_path, include_vessel=True)
    kwargs.update(overrides)
    return TokaMakerConfig(**kwargs)


def test_plasma_loop_contract(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    result = run_tokamaker_evolution(_evolution_inputs(tmp_path), _config(tmp_path))

    names = [entry[0] for entry in calls]
    assert result.ok and result.returncode == 0

    # slice 0 is a plain static solve: init_psi, never set_psi_dt before it
    first_solve = names.index("solve")
    assert "init_psi" in names[:first_solve]
    assert "set_psi_dt" not in names[:first_solve]

    # subsequent slices chain psi0 from the previous solve with the right dt
    psi_dt_calls = [entry for entry in calls if entry[0] == "set_psi_dt"]
    stepping = [entry for entry in psi_dt_calls if entry[2] > 0]
    assert len(stepping) == 2
    for prev_count, entry in zip((1, 2), stepping):
        assert np.all(entry[1] == float(prev_count))     # stamped by the fake get_psi
        assert entry[2] == pytest.approx(0.002)

    # eddy hygiene: one disable call after the loop, then a single reset last
    assert psi_dt_calls[-1][2] == -1.0
    assert names.count("reset") == 1 and names[-1] == "reset"

    # per-slice artefacts
    assert [path.name for path in result.gfiles] == ["g039915.00300", "g039915.00302", "g039915.00304"]
    assert all(rec.converged for rec in result.steps)
    # vessel current integrated over the fake conductor region W1 (reg_id 5)
    assert result.steps[0].vessel_currents_A == {"W1": pytest.approx(500.0)}
    payload = json.loads(result.sidecar_file.read_text())
    assert [step["converged"] for step in payload["steps"]] == [True, True, True]
    # fake g-files cannot be parsed, so the IDS merge fails best-effort
    assert "_merge_error" in result.scalars


def test_failure_continue_keeps_last_converged_state(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch, solve_error="diverged", solve_error_at=2)
    result = run_tokamaker_evolution(_evolution_inputs(tmp_path), _config(tmp_path))

    assert not result.ok and result.returncode == 1
    assert [rec.converged for rec in result.steps] == [True, False, True]
    assert "diverged" in result.steps[1].error
    assert result.steps[1].gfile is None

    # the slice after the failure steps from slice 0's psi across the gap
    stepping = [entry for entry in calls if entry[0] == "set_psi_dt" and entry[2] > 0]
    assert len(stepping) == 2
    assert np.all(stepping[1][1] == 1.0)                 # psi from solve #1 (slice 0)
    assert stepping[1][2] == pytest.approx(0.004)        # dt spans the failed slice


def test_failure_stop_aborts_the_loop(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch, solve_error="diverged", solve_error_at=2)
    config = _config(tmp_path, evolve_on_failure="stop")
    result = run_tokamaker_evolution(_evolution_inputs(tmp_path), config)

    assert not result.ok
    assert len(result.steps) == 2
    assert "diverged" in result.error
    names = [entry[0] for entry in calls]
    assert names[-1] == "reset"


def test_vacuum_mode_uses_vac_solve_without_targets(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    config = _config(tmp_path, evolve_vacuum=True, evolve_field_probes=((0.5, 0.1),))
    result = run_tokamaker_evolution(_evolution_inputs(tmp_path, vacuum=True), config)

    names = [entry[0] for entry in calls]
    assert result.ok
    assert "vac_solve" in names
    assert "solve" not in names
    assert "set_targets" not in names
    assert "init_psi" not in names
    assert "save_eqdsk" not in names
    assert "set_profiles" not in names
    # vac_solve does not update internal state; the runner must push it back
    assert names.index("set_psi") < names.index("get_field_eval")

    probe = result.steps[0].probe_fields
    assert probe["br"][0] == pytest.approx(0.01 * 0.5)
    assert probe["bz"][0] == pytest.approx(0.02 * 0.1)
    assert probe["psi"][0] == pytest.approx(0.05 * 0.5 * 0.1)
    assert result.gfiles == ()


def _prepare_ods():
    """Synthetic ODS rich enough for evolution preparation."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 12345
    ods["wall.description_2d.0.limiter.unit.0.outline.r"] = np.array([0.2, 0.6, 0.6, 0.2])
    ods["wall.description_2d.0.limiter.unit.0.outline.z"] = np.array([-0.4, -0.4, 0.4, 0.4])
    ods["pf_active.coil.0.name"] = "PF1"
    base = "pf_active.coil.0.element.0"
    ods[f"{base}.geometry.rectangle.r"] = 0.1
    ods[f"{base}.geometry.rectangle.z"] = 0.5
    ods[f"{base}.geometry.rectangle.width"] = 0.04
    ods[f"{base}.geometry.rectangle.height"] = 0.1
    ods[f"{base}.turns_with_sign"] = 8.0
    ods["pf_active.time"] = np.array([0.0, 1.0])
    ods["pf_active.coil.0.current.data"] = np.array([0.0, 1000.0])
    ods["magnetics.ip.0.time"] = np.array([0.0, 1.0])
    ods["magnetics.ip.0.data"] = np.array([0.0, 100.0e3])
    ods["tf.time"] = np.array([0.0, 1.0])
    ods["tf.b_field_tor_vacuum_r.data"] = np.array([0.06, 0.06])
    for i, zc in enumerate((-0.05, 0.0, 0.05)):
        loop = f"pf_passive.loop.{i}"
        ods[f"{loop}.name"] = "WA"
        ods[f"{loop}.element.0.geometry.outline.r"] = np.array([0.70, 0.71, 0.71, 0.70])
        ods[f"{loop}.element.0.geometry.outline.z"] = np.array(
            [zc - 0.025, zc - 0.025, zc + 0.025, zc + 0.025]
        )
    return ods


def test_prepare_resolves_waveforms_and_targets(tmp_path):
    config = TokaMakerConfig(
        workdir=tmp_path, include_vessel=True, constraint_source="magnetics",
        evolve_times=(0.30, 0.31, 0.32),
    )
    inputs = prepare_tokamaker_evolution_inputs(_prepare_ods(), config)

    assert inputs.times == (0.30, 0.31, 0.32)
    assert not inputs.vacuum
    assert inputs.coil_waveforms["PF1"] == pytest.approx([300.0, 310.0, 320.0])
    assert inputs.ip_targets == pytest.approx([30.0e3, 31.0e3, 32.0e3])
    assert inputs.base.time == pytest.approx(0.30)
    assert "vessel" in inputs.base.geometry


def test_prepare_vacuum_mode_skips_targets(tmp_path):
    config = TokaMakerConfig(
        workdir=tmp_path, include_vessel=True, evolve_vacuum=True,
        evolve_start=0.27, evolve_end=0.30, evolve_dt=0.01,
    )
    inputs = prepare_tokamaker_evolution_inputs(_prepare_ods(), config)
    assert inputs.vacuum
    assert inputs.base.targets == {}
    assert np.all(inputs.ip_targets == 0.0)
    assert inputs.times == pytest.approx((0.27, 0.28, 0.29))


def test_prepare_validation_errors(tmp_path):
    ods = _prepare_ods()
    with pytest.raises(ValueError, match="include_vessel"):
        prepare_tokamaker_evolution_inputs(
            ods, TokaMakerConfig(workdir=tmp_path, evolve_times=(0.30, 0.31))
        )
    with pytest.raises(ValueError, match="distinct integer milliseconds"):
        prepare_tokamaker_evolution_inputs(
            ods, TokaMakerConfig(workdir=tmp_path, include_vessel=True,
                                 constraint_source="magnetics",
                                 evolve_times=(0.3000, 0.3002)),
        )
    with pytest.raises(ValueError, match="at least 2"):
        prepare_tokamaker_evolution_inputs(
            ods, TokaMakerConfig(workdir=tmp_path, include_vessel=True,
                                 constraint_source="magnetics", evolve_times=(0.30,)),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_tokamaker_evolution_inputs(
            ods, TokaMakerConfig(workdir=tmp_path, include_vessel=True,
                                 constraint_source="magnetics",
                                 evolve_times=(0.31, 0.30)),
        )
    with pytest.raises(ValueError, match="evolve_times"):
        prepare_tokamaker_evolution_inputs(
            ods, TokaMakerConfig(workdir=tmp_path, include_vessel=True,
                                 constraint_source="magnetics"),
        )


def test_evolve_eddy_false_disables_the_wall_term(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch)
    config = _config(tmp_path, evolve_vacuum=True, evolve_eddy=False)
    result = run_tokamaker_evolution(_evolution_inputs(tmp_path, vacuum=True), config)

    assert result.ok
    stepping = [entry for entry in calls if entry[0] == "set_psi_dt" and entry[2] > 0]
    assert stepping == []                            # coil-only control: no wall term


def test_evolution_relaxes_nl_tol_unless_explicit(tmp_path, monkeypatch):
    from vaft.code.tokamaker.evolve import EVOLVE_NL_TOL

    calls, _ = make_fake_oft(monkeypatch)
    run_tokamaker_evolution(_evolution_inputs(tmp_path), _config(tmp_path))
    pushes = [entry for entry in calls if entry[0] == "update_settings"]
    assert pushes and pushes[0][1] == pytest.approx(EVOLVE_NL_TOL)

    calls2, _ = make_fake_oft(monkeypatch)
    run_tokamaker_evolution(_evolution_inputs(tmp_path), _config(tmp_path, nl_tol=1e-7))
    assert all(entry[0] != "update_settings" for entry in calls2)


def test_failed_slice_restores_the_last_converged_flux(tmp_path, monkeypatch):
    calls, _ = make_fake_oft(monkeypatch, solve_error="diverged", solve_error_at=2)
    run_tokamaker_evolution(_evolution_inputs(tmp_path), _config(tmp_path))

    names = [entry[0] for entry in calls]
    # the diverged iterate from the failed solve must be replaced by the last
    # converged flux before the next slice steps from it
    restores = [entry for entry in calls if entry[0] == "set_psi"]
    assert restores and np.all(restores[0][1] == 1.0)      # psi of slice 0
    fail_index = names.index("solve", names.index("solve") + 1)
    assert names.index("set_psi") > fail_index


def test_prepare_clears_time_index_for_the_evolution_grid(tmp_path):
    # time_index would otherwise win over `time` in _resolve_time and index
    # past the short synthetic time arrays
    config = TokaMakerConfig(
        workdir=tmp_path, include_vessel=True, constraint_source="magnetics",
        time_index=5, evolve_times=(0.30, 0.31),
    )
    inputs = prepare_tokamaker_evolution_inputs(_prepare_ods(), config)
    assert inputs.base.time == pytest.approx(0.30)
