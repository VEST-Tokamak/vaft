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


def _produced_slice(time_ms, phase, *, magnetic_chisq, area=1.0):
    scalars = {
        name: 1.0
        for name in (
            "rm", "zm", "area", "volume", "li", "betap", "q95", "qmin", "wmhd",
            "cjor0", "cjor95", "cjor99", "cj1ave", "peak", "chisq", "condno",
        )
    }
    scalars["area"] = area
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
    mfile = None
    if magnetic_chisq is not None:
        mfile = {
            "scalars": {
                "magnetics_chisq": magnetic_chisq,
                "magnetics_chisq_per_active_signal": magnetic_chisq / 2.0,
                "bpol_probe_chisq": magnetic_chisq * 0.75,
                "flux_loop_chisq": magnetic_chisq * 0.25,
                "plasma_current_chisq": 10.0,
                "total_chisq": 10.0 + magnetic_chisq,
            }
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
        "mfile": mfile,
    }


def _run(slices, seconds=1.0):
    return {"seconds": seconds, "slices": list(slices)}


def _model(slices):
    return {"run": _run(slices)}


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


def test_a_slice_without_a_gfile_is_no_output_here_and_flagged_in_seed_basin(study):
    """The one label on which this study and the seed basin differ, on purpose.

    Everything here compares p-prime, FF-prime, the current profile or the
    boundary, and all of those live in the g-file. Sharing `outcome` keeps
    `accepted` and `collapsed` one definition across the studies; the g-file
    condition is the stated exception, not a quiet second rule.
    """
    shared_calls = []

    def shared(record):
        shared_calls.append(record)
        return "flagged" if record["afile"] else "no_output"

    with_gfile = {"afile": {"jflag": 0}, "gfile": {}, "collapsed": False}
    without_gfile = {"afile": {"jflag": 0}, "gfile": None, "collapsed": False}

    assert study.outcome_of(with_gfile, shared) == "flagged"
    assert study.outcome_of(without_gfile, shared) == "no_output"
    assert len(shared_calls) == 2, "the shared classifier decides, not a restatement"

    # Collapse is never reinterpreted: it arrives from the log parser and wins.
    assert study.outcome_of({"afile": None, "gfile": None, "collapsed": True},
                            lambda record: "collapsed") == "collapsed"


def test_signed_change_keeps_the_direction_and_separates_identical_from_unchanged(study):
    tolerance = study.UNCHANGED_RELATIVE
    record = study.signed_change([-0.5, -0.25, 0.0, tolerance / 2.0, 0.75])

    assert record["decreased"] == 2
    assert record["increased"] == 1
    assert record["unchanged"] == 2
    # An exact zero is a stronger statement than "agreed to within the file's
    # precision", so it is counted separately rather than folded in.
    assert record["identical"] == 1
    assert record["decreased"] + record["increased"] + record["unchanged"] == record["n"]
    assert record["median"] == pytest.approx(0.0)
    assert record["min"] == pytest.approx(-0.5)
    assert record["tolerance"] == tolerance

    # The absolute summary cannot tell these two apart; the signed one must.
    better = study.signed_change([-0.4, -0.4, -0.4])
    worse = study.signed_change([0.4, 0.4, 0.4])
    assert study.spread([-0.4] * 3, absolute=True)["median"] == pytest.approx(
        study.spread([0.4] * 3, absolute=True)["median"]
    )
    assert better["median"] < 0 < worse["median"]
    assert (better["decreased"], worse["decreased"]) == (3, 0)


def test_the_signed_chi_square_survives_a_cohort_that_cancels_it(study):
    """A model that helps the flat-top and hurts the ramp nets out to nothing."""
    baseline = _run([
        _produced_slice(100, "ramp_up", magnetic_chisq=10.0),
        _produced_slice(101, "quasi_stationary", magnetic_chisq=10.0),
    ])
    candidate = _run([
        _produced_slice(100, "ramp_up", magnetic_chisq=15.0),
        _produced_slice(101, "quasi_stationary", magnetic_chisq=5.0),
    ])

    comparison = study.compare_runs(candidate, baseline)
    overall = comparison["fit_signed_relative_change"]["magnetics_chisq"]
    assert overall["n"] == 2
    assert (overall["decreased"], overall["increased"]) == (1, 1)
    assert overall["median"] == pytest.approx(0.0)

    ramp = comparison["by_phase"]["ramp_up"]["magnetics_chisq_signed_relative_change"]
    flat = comparison["by_phase"]["quasi_stationary"]["magnetics_chisq_signed_relative_change"]
    assert ramp["median"] == pytest.approx(0.5)
    assert flat["median"] == pytest.approx(-0.5)

    # The absolute keys the earlier report was built on do not move.
    assert comparison["fit_absolute_relative_change"]["magnetics_chisq"]["median"] == pytest.approx(0.5)


def test_the_ensemble_reports_the_population_and_names_what_empties_it(study):
    """sigma over a population that changes between slices is not a trajectory."""
    common = [
        _produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.0),
        _produced_slice(101, "ramp_up", magnetic_chisq=10.0, area=1.0),
    ]
    models = {
        "p22_zero": _model(common),
        "p22_free": _model([
            _produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.2),
            _produced_slice(101, "ramp_up", magnetic_chisq=10.0, area=1.2),
        ]),
        # This one reconstructs only the first time.
        "p33_zero": _model([
            _produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=0.8),
        ]),
    }

    ensemble = study.summarize_ensemble(models)

    assert ensemble["models"] == ["p22_free", "p22_zero", "p33_zero"]
    assert ensemble["models_total"] == 3
    assert ensemble["population"] == "available"
    assert ensemble["times_n"] == 2
    assert ensemble["models_n_per_time"]["min"] == 2
    assert ensemble["models_n_per_time"]["max"] == 3
    assert ensemble["complete_times_n"] == 1

    # Leave-one-out names the model responsible instead of quietly dropping it.
    assert ensemble["complete_times_n_without"]["p33_zero"] == 2
    assert ensemble["complete_times_n_without"]["p22_zero"] == 1

    # The headline is over the times every model produced; the all-available
    # figure is kept beside it and is computed over a different population.
    assert ensemble["complete"]["relative_sigma"]["area"]["n"] == 1
    assert ensemble["relative_sigma"]["area"]["n"] == 2

    record = ensemble["records"][0]
    assert record["complete"] is True
    assert record["models_missing"] == []
    assert record["phase_agreement"] is True
    assert ensemble["records"][1]["models_missing"] == ["p33_zero"]

    area = record["quantities"]["area"]
    assert area["n"] == 3
    # The normalizer is stored, so relative_sigma can be checked rather than
    # recomputed from numbers the table does not carry.
    assert area["center"] == pytest.approx(1.0)
    assert area["relative_sigma"] == pytest.approx(area["sigma"] / abs(area["center"]))
    assert ensemble["statistics"]["center"] == "median"


def test_a_missing_mfile_shortens_the_chi_square_population_and_not_the_others(study):
    models = {
        "p22_zero": _model([_produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.0)]),
        "p22_free": _model([_produced_slice(100, "ramp_up", magnetic_chisq=None, area=1.2)]),
    }

    ensemble = study.summarize_ensemble(models)

    assert ensemble["complete_times_n"] == 1
    assert ensemble["complete"]["relative_sigma"]["area"]["n"] == 1
    # One model wrote no m-file, so the time is complete for the geometry and
    # short for the fit. Reporting one count for both would overstate it.
    assert ensemble["complete"]["relative_sigma"]["magnetics_chisq"]["n"] == 0


def _payload(models, comparisons, ensemble, *, shot="41672"):
    return {
        "shots": {
            shot: {
                "models": models,
                "comparisons": comparisons,
                "ensemble": ensemble,
            }
        }
    }


def test_the_report_states_the_population_the_uncertainty_is_taken_over(study):
    baseline_slices = [
        _produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.0),
        _produced_slice(101, "ramp_up", magnetic_chisq=10.0, area=1.0),
    ]
    candidate_slices = [
        _produced_slice(100, "ramp_up", magnetic_chisq=5.0, area=1.2),
    ]
    baseline, candidate = _run(baseline_slices), _run(candidate_slices)
    models = {
        "p22_zero": {
            "specification": {"kppcur": 2, "kffcur": 2},
            "summary": study.summarize_run(baseline),
            "run": baseline,
        },
        "p33_zero": {
            "specification": {"kppcur": 3, "kffcur": 3},
            "summary": study.summarize_run(candidate),
            "run": candidate,
        },
    }
    payload = _payload(
        models,
        {"p33_zero": study.compare_runs(candidate, baseline)},
        study.summarize_ensemble(models),
    )

    text = study.markdown(payload)

    assert "## Model-induced uncertainty" in text
    assert "## Which way the fit moved" in text
    # The caveat is in the body at full weight, not a footnote.
    assert "the population is not the same at every time" in text
    assert "model dropped" in text
    assert "`p33_zero` | 2 |" in text
    # The freedom each fit bought itself is beside the fit.
    assert "| coefficients |" in text
    assert "| `p33_zero` | 6 " in text


def test_the_report_says_so_when_the_two_populations_coincide(study):
    slices = [_produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.0)]
    other = [_produced_slice(100, "ramp_up", magnetic_chisq=10.0, area=1.1)]
    runs = {"p22_zero": _run(slices), "p22_free": _run(other)}
    models = {
        name: {
            "specification": {"kppcur": 2, "kffcur": 2},
            "summary": study.summarize_run(run),
            "run": run,
        }
        for name, run in runs.items()
    }
    payload = _payload(
        models,
        {"p22_free": study.compare_runs(runs["p22_free"], runs["p22_zero"])},
        study.summarize_ensemble(models),
    )

    text = study.markdown(payload)

    assert "so the two populations coincide" in text
    assert "model dropped" not in text


def test_a_single_model_run_reports_no_spread_rather_than_failing(study):
    """Re-running one model against the cache is a normal thing to ask for."""
    run = _run([_produced_slice(100, "ramp_up", magnetic_chisq=10.0)])
    models = {
        "p22_zero": {
            "specification": {"kppcur": 2, "kffcur": 2},
            "summary": study.summarize_run(run),
            "run": run,
        }
    }
    ensemble = study.summarize_ensemble(models)
    assert ensemble["times_n"] == 0
    assert ensemble["plasma_times_n"] == 1

    text = study.markdown(_payload(models, {}, ensemble))

    assert "no across-model spread to report" in text
    assert "relative sigma median" not in text


def test_dropping_the_samples_keeps_everything_derived_from_them(study):
    """The report is a table of conclusions; the samples stay with the g-files."""
    run = _run([_produced_slice(100, "ramp_up", magnetic_chisq=10.0)])
    models = {
        "p22_zero": {
            "specification": {"kppcur": 2, "kffcur": 2},
            "summary": study.summarize_run(run),
            "run": run,
        }
    }
    payload = _payload(models, {}, study.summarize_ensemble(models))
    gfile = payload["shots"]["41672"]["models"]["p22_zero"]["run"]["slices"][0]["gfile"]
    gfile["boundary"].update({"r_min": 0.3, "r_max": 0.5, "z_min": 0.0, "z_max": 0.1})

    study.drop_sampled_arrays(payload)

    assert "profiles" not in gfile
    assert "r" not in gfile["boundary"] and "z" not in gfile["boundary"]
    # What the arrays were read for survives them.
    assert gfile["profile_diagnostics"]["jphi_edge"] == 1.0
    assert gfile["boundary"]["r_min"] == 0.3 and gfile["boundary"]["z_max"] == 0.1
    assert gfile["axis"] == [0.4, 0.0]

    # Idempotent: the per-shot checkpoint calls it once per discharge.
    study.drop_sampled_arrays(payload)


def test_every_caller_of_run_model_passes_everything_it_requires(study):
    """`run_model` is shared, so a missed argument must fail here, not mid-scan.

    `constraint_information_study` drives the same solver entry point. When
    `seed_module` was added, that call site kept working until the study was
    next run -- nothing exercises `main` without EFIT, so a scan would have
    been the first thing to find out.
    """
    import ast
    import inspect

    required = {
        name
        for name, parameter in inspect.signature(study.run_model).parameters.items()
        if parameter.kind is parameter.KEYWORD_ONLY and parameter.default is parameter.empty
    }
    assert "seed_module" in required

    workflow = Path(__file__).resolve().parents[1] / "workflow"
    call_sites = 0
    for path in sorted(workflow.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", None)
            )
            if name != "run_model":
                continue
            call_sites += 1
            passed = {keyword.arg for keyword in node.keywords}
            assert required <= passed, (
                f"{path.relative_to(workflow.parent)}:{node.lineno} is missing "
                f"{sorted(required - passed)}"
            )
    assert call_sites >= 2, "expected the profile study and its sibling caller"
