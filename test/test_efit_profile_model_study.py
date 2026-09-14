from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_profile_models" / "profile_model_study.py"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("efit_profile_model_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_the_profile_study_pins_the_qualified_seed_and_every_numerical_default(study):
    config = study.fixed_scientific_config()

    assert config.initialization.ellipse_rzero == 0.32
    assert config.initialization.rzero == 0.4
    assert config.initialization.icinit == 2
    assert config.numerics.error_minimum == 1.0e-2
    assert config.numerics.chi_squared_target == 80.0
    assert config.numerics.convergence_mode == 2
    assert config.numerics.inner_iterations == 1


def test_the_pilot_changes_only_the_profile_block(study):
    base = study.fixed_scientific_config()
    identities = set()
    for model in study.PILOT_MODELS:
        candidate = study.scientific_for(model)
        assert candidate.initialization == base.initialization
        assert candidate.numerics == base.numerics
        assert candidate.constraints == base.constraints
        identities.add(candidate.sha256)

    assert len(identities) == len(study.PILOT_MODELS)
    assert study.BASELINE_MODEL in {model.name for model in study.PILOT_MODELS}
    assert any(model.name == "p11_zero" for model in study.PILOT_MODELS)


def test_fwtbp_models_only_use_coefficient_bases_the_executable_can_pair(study):
    for model in study.PILOT_MODELS + study.ORDER_MODELS:
        if model.fwtbp:
            assert model.kffcur >= 2
            assert model.kppcur >= model.kffcur
        study.scientific_for(model)


def test_contour_distance_is_symmetric_and_reported_in_physical_units(study):
    square = {"r": [0.0, 1.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0, 1.0]}
    shifted = {"r": [0.01, 1.01, 1.01, 0.01], "z": [0.0, 0.0, 1.0, 1.0]}

    forward = study._curve_distance(square, shifted)
    backward = study._curve_distance(shifted, square)

    assert forward == backward
    assert forward["hausdorff_m"] == pytest.approx(0.01)


def test_model_selection_always_includes_the_paired_baseline(study):
    selected = study._selected_models("p22_free", full_matrix=False)
    assert [model.name for model in selected] == ["p22_zero", "p22_free"]

    with pytest.raises(ValueError, match="unknown model"):
        study._selected_models("not-a-model", full_matrix=False)


def _produced_slice(time_ms, phase, *, magnetic_chisq):
    scalars = {
        name: 1.0
        for name in (
            "rm", "zm", "area", "volume", "li", "betap", "q95", "qmin", "wmhd",
            "cjor0", "cjor95", "cjor99", "cj1ave", "peak", "chisq", "condno",
        )
    }
    profiles = {
        name: [1.0] * 10
        for name in ("pressure", "pprime", "ffprime", "f", "q", "jphi_reference_r")
    }
    profile_diagnostics = {
        "jphi_psi_080": 1.0,
        "jphi_psi_090": 1.0,
        "jphi_psi_095": 1.0,
        "jphi_edge": 1.0,
        "jphi_edge_shell_mean": 1.0,
        "jphi_edge_shell_to_peak": 1.0,
        "pressure_negative_fraction": 0.0,
        "f_nonfinite_count": 0,
        "pprime_turns": 0,
        "ffprime_turns": 0,
        "q_turns": 0,
        "jphi_turns": 0,
    }
    return {
        "time_ms": time_ms,
        "phase": phase,
        "outcome": "accepted",
        "afile": {"scalars": scalars},
        "gfile": {
            "axis": [0.4, 0.0],
            "boundary": {"r": [0.3, 0.5, 0.5], "z": [0.0, 0.0, 0.1]},
            "profiles": profiles,
            "profile_diagnostics": profile_diagnostics,
        },
        "mfile": {
            "scalars": {
                "magnetics_chisq": magnetic_chisq,
                "magnetics_chisq_per_active_signal": magnetic_chisq / 2.0,
                "bpol_probe_chisq": magnetic_chisq * 0.75,
                "flux_loop_chisq": magnetic_chisq * 0.25,
                "plasma_current_chisq": 10.0,
                "total_chisq": 10.0 + magnetic_chisq,
            }
        },
    }


def test_headline_and_paired_statistics_exclude_vacuum_slices(study):
    baseline = {
        "seconds": 1.0,
        "slices": [
            _produced_slice(100, "ramp_up", magnetic_chisq=2.0),
            _produced_slice(101, "vacuum", magnetic_chisq=20.0),
        ],
    }
    candidate = {
        "seconds": 1.0,
        "slices": [
            _produced_slice(100, "ramp_up", magnetic_chisq=3.0),
            _produced_slice(101, "vacuum", magnetic_chisq=200.0),
        ],
    }

    summary = study.summarize_run(candidate)
    comparison = study.compare_runs(candidate, baseline)

    assert summary["produced"] == 2
    assert summary["plasma_produced"] == 1
    assert summary["magnetics_chisq"]["median"] == 3.0
    assert comparison["common_produced"] == 1
    assert comparison["fit_absolute_relative_change"]["magnetics_chisq"]["median"] == 0.5
