from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "efit_profile_models" / "profile_model_study.py"
TABLE = Path(__file__).resolve().parent / "data" / "efit_profile_model_study.json"


@pytest.fixture(scope="module")
def table():
    import json

    return json.loads(TABLE.read_text(encoding="utf-8"))


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


def test_equilibria_without_an_mfile_are_counted_rather_than_summarized_as_none(study):
    """An EFIT built without netCDF answers none of #579's questions, quietly.

    It writes a-files and g-files, every slice classifies, geometry and
    profiles compare -- and every chi-square is `None`, because this study
    ranks models on the m-file's probe and flux-loop sums rather than on the
    a-file total. A whole scan was spent finding that out.
    """
    with_fit = _produced_slice(100, "ramp_up", magnetic_chisq=10.0)
    without_fit = _produced_slice(101, "ramp_up", magnetic_chisq=None)
    no_equilibrium = {
        "time_ms": 102, "phase": "ramp_up", "outcome": "collapsed",
        "afile": None, "gfile": None, "mfile": None,
    }

    assert study.missing_fit_measures(_run([with_fit])) == 0
    assert study.missing_fit_measures(_run([with_fit, without_fit])) == 1
    # A slice that produced nothing is not a missing measurement.
    assert study.missing_fit_measures(_run([with_fit, no_equilibrium])) == 0

    # And the summary must not present the survivors as the whole population.
    summary = study.summarize_run(_run([with_fit, without_fit]))
    assert summary["plasma_produced"] == 2
    assert summary["magnetics_chisq"]["n"] == 1


# --- the committed table, and the claims the README makes from it ----------
#
# These are written so that a change which invalidates a stated conclusion
# fails here, which is what they are for.


def test_every_model_was_measured_against_the_same_thing(study, table):
    """Nine configurations of one experiment, or there is nothing to compare."""
    assert table["schema_version"] == study.SCHEMA
    assert table["issue"] == 579
    identity = table["fixed_scientific_sha256"]
    names = {model["name"] for model in table["models"]}
    assert names == {model.name for model in study.PILOT_MODELS + study.ORDER_MODELS}
    assert study.BASELINE_MODEL in names

    for shot, block in table["shots"].items():
        requested = block["window"]["requested"]
        assert set(block["models"]) == names, shot
        digests = set()
        for name, model in block["models"].items():
            run = model["run"]
            assert run["requested_slices"] == requested, (shot, name)
            digests.add(run["scientific_sha256"])
            # Only the profile block may differ from the frozen baseline.
            scientific = run["scientific"]
            for section in ("initialization", "numerics", "constraints"):
                assert scientific[section] == json.loads(
                    json.dumps(table["fixed_scientific"][section])
                ), (shot, name, section)
        assert len(digests) == len(names), f"{shot}: two models share a configuration"
        assert identity == table["fixed_scientific_sha256"]


def test_the_chi_square_does_not_respond_to_the_profile_model(table):
    """#579's first question, answered in the negative and in the mechanism.

    EFIT's objective for VEST is the plasma-current term -- the magnetics are
    handed uncertainties four orders above their signal and contribute ~1e-9
    -- so no profile model can move the number `SAICON` tests. If a future
    change to the weighting makes the magnetics matter, this fails, and the
    README's first section has to be rewritten. That is the point.
    """
    shares, identical, compared = [], 0, 0
    for block in table["shots"].values():
        for model in block["models"].values():
            for item in model["run"]["slices"]:
                fit = (item.get("mfile") or {}).get("scalars")
                if not fit or not fit["total_chisq"]:
                    continue
                shares.append(fit["magnetics_chisq"] / fit["total_chisq"])
        for comparison in block["comparisons"].values():
            term = comparison["fit_signed_relative_change"]["plasma_current_chisq"]
            identical += term["identical"]
            compared += term["n"]

    assert len(shares) > 400
    shares.sort()
    assert shares[len(shares) // 2] < 1e-6, "the magnetics have started to matter"
    assert identical / compared > 0.95, (identical, compared)


def test_more_profile_freedom_fits_the_magnetics_worse_on_every_slice(table):
    """Unanimous, and the opposite of what added parameters should do.

    A least-squares fit cannot have a larger residual at its optimum for
    having more of them, so this is the solver trading the unweighted term
    away. `p11_zero` is the control: less freedom, better on every slice.
    """
    freer = ("p22_free", "p22_free_fwtbp", "p23_zero", "p24_zero", "p33_zero")
    for shot, block in table["shots"].items():
        for name in freer:
            term = block["comparisons"][name]["fit_signed_relative_change"]["magnetics_chisq"]
            assert term["n"] > 0 and term["decreased"] == 0, (shot, name, term)
            assert term["median"] > 0, (shot, name)
        control = block["comparisons"]["p11_zero"]["fit_signed_relative_change"]["magnetics_chisq"]
        assert control["increased"] == 0 and control["median"] < 0, (shot, control)


def test_the_order_that_moves_the_reconstruction_is_kffcur_alone(table):
    """P-prime order is inert at VEST's beta; FF-prime order is the whole axis.

    `(3,2)` adds a p' coefficient and moves the boundary by a millimetre;
    `(2,3)` adds an FF' coefficient and moves it by two centimetres; `(3,3)`
    adds both and is `(2,3)`. And `(2,4)` moves further still, so the FF'
    basis has not converged in the range VEST supports.
    """
    for shot, block in table["shots"].items():
        lcfs = {
            name: block["comparisons"][name]["lcfs_rms_mm"]["median"]
            for name in ("p32_zero", "p23_zero", "p24_zero", "p33_zero")
        }
        assert lcfs["p32_zero"] < 2.0, (shot, lcfs)
        assert lcfs["p23_zero"] > 10 * lcfs["p32_zero"], (shot, lcfs)
        # Adding the p' coefficient on top of the FF' one changes nothing.
        assert abs(lcfs["p33_zero"] - lcfs["p23_zero"]) < 0.1 * lcfs["p23_zero"], (shot, lcfs)
        # No plateau: the fourth FF' coefficient moves it further again.
        assert lcfs["p24_zero"] > 1.4 * lcfs["p23_zero"], (shot, lcfs)


def test_area_and_volume_carry_the_model_uncertainty_the_issue_asked_for(table):
    """The headline: ~17% on area and ~25% on volume, on three discharges.

    Taken over the times every model produced an equilibrium, so it is a
    spread across one ensemble rather than a different one at every slice.
    """
    for shot, block in table["shots"].items():
        ensemble = block["ensemble"]
        assert ensemble["models_total"] == 9
        complete = ensemble["complete"]["relative_sigma"]
        assert complete["area"]["n"] >= 7, shot
        assert 0.14 < complete["area"]["median"] < 0.20, (shot, complete["area"])
        assert 0.22 < complete["volume"]["median"] < 0.29, (shot, complete["volume"])
        assert 0.27 < complete["li"]["median"] < 0.34, (shot, complete["li"])
        # q95 is the one quantity the magnetics pin, and it is pinned far better.
        assert complete["q95"]["median"] < 0.5 * complete["area"]["median"], shot
        # The all-available population is reported beside it and is close here,
        # so the strict cut is a check rather than a correction.
        available = ensemble["relative_sigma"]["area"]["median"]
        assert abs(available - complete["area"]["median"]) < 0.02, shot


def test_the_free_edge_is_the_only_lever_that_moves_the_collapse_block(table):
    """Twelve null solutions become equilibria; the seed and grid moved none.

    It is not free: acceptance falls and the boundary lands 68 mm away. The
    assertion is that the block moves at all, which is what #459 needs.
    """
    recovered = collapsed_before = collapsed_after = 0
    for block in table["shots"].values():
        transitions = block["comparisons"]["p22_free"]["outcome_transitions"]
        recovered += sum(
            count for label, count in transitions.items()
            if label.startswith("collapsed->") and not label.endswith("->collapsed")
        )
        collapsed_before += block["models"]["p22_zero"]["summary"]["plasma_outcomes"].get("collapsed", 0)
        collapsed_after += block["models"]["p22_free"]["summary"]["plasma_outcomes"].get("collapsed", 0)

    assert collapsed_before == 22 and collapsed_after == 10
    assert recovered >= 12

    # And the cost, so the row is never read as a free win.
    accepted = {
        name: sum(
            block["models"][name]["summary"]["plasma_outcomes"].get("accepted", 0)
            for block in table["shots"].values()
        )
        for name in ("p22_zero", "p22_free")
    }
    assert accepted["p22_free"] < accepted["p22_zero"]
    for shot, block in table["shots"].items():
        assert block["comparisons"]["p22_free"]["lcfs_rms_mm"]["median"] > 50, shot


def test_the_report_renders_from_the_committed_table(study, table):
    text = study.markdown(table)
    assert text.startswith("# EFIT profile-model uncertainty")
    assert "## Model-induced uncertainty" in text
    assert "## Which way the fit moved" in text
    for shot in table["shots"]:
        assert f"| {shot} |" in text


def test_a_cached_model_result_is_reused_only_for_the_run_that_produced_it(study, tmp_path, monkeypatch):
    """cold review efit-workflows F2: the cache compared the schema and the
    scientific hash only, so new times, constraints, tables, executable or
    phase labels under the same workdir returned the old slices."""
    import types

    import numpy as np

    import vaft.code.efit.magnetic as magnetic

    calls = []

    def fake_run(inputs, config):
        calls.append(config)
        return types.SimpleNamespace(stdout="", returncode=0, status="completed")

    monkeypatch.setattr(magnetic, "prepare_efit_inputs", lambda constraints, config: None)
    monkeypatch.setattr(magnetic, "run_efit", fake_run)

    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "mhdin.dat").write_text("envelope A", encoding="utf-8")
    executable = tmp_path / "efit"
    executable.write_text("build A", encoding="utf-8")
    constraints = {
        "equilibrium.time": np.array([0.300, 0.301]),
        "equilibrium.code.parameters.time_slice.0.IN1.TABLE_DIR": f"{tables}/",
    }
    arguments = dict(
        shot=39915,
        times=np.array([0.300, 0.301]),
        workdir=tmp_path / "shot_39915" / "baseline",
        executable=str(executable),
        scientific=study.fixed_scientific_config(),
        baseline_module=types.SimpleNamespace(parse_slices=lambda text: []),
        seed_module=types.SimpleNamespace(outcome=lambda record: "no_output"),
        phase_by_time={300: {"current": 1.0e5, "dcurrent_dt": 0.0, "phase": "flat"}},
    )

    first = study.run_model(constraints, **arguments)
    assert len(calls) == 1
    assert study.run_model(constraints, **arguments) == first
    assert len(calls) == 1, "an identical request is served from the cache"

    def reruns(constraints=constraints, **changed):
        before = len(calls)
        payload = study.run_model(constraints, **{**arguments, **changed})
        return len(calls) == before + 1, payload

    ran, payload = reruns(times=np.array([0.300, 0.3005, 0.301]))
    assert ran and payload["requested_slices"] == 3
    assert reruns(times=arguments["times"])[0], "the 3-slice cache must not serve the 2-slice request"
    assert not reruns()[0]

    assert reruns(constraints={**constraints, "equilibrium.time": np.array([0.300, 0.302])})[0]
    assert reruns()[0]
    (tables / "mhdin.dat").write_text("envelope B, rebuilt in place", encoding="utf-8")
    assert reruns()[0]
    executable.write_text("build B at the same path", encoding="utf-8")
    assert reruns()[0]
    assert reruns(phase_by_time={300: {"current": 1.0e5, "dcurrent_dt": 0.0, "phase": "ramp_up"}})[0]
    assert reruns(shot=41672)[0]

    # A cache written before the identity existed re-runs instead of raising.
    cache = arguments["workdir"] / "model-result.json"
    legacy = json.loads(cache.read_text(encoding="utf-8"))
    del legacy["run_identity"]
    cache.write_text(json.dumps(legacy), encoding="utf-8")
    assert reruns(shot=41672)[0]
    cache.write_text("{truncated", encoding="utf-8")
    assert reruns(shot=41672)[0]


def test_the_per_shot_checkpoint_can_be_written_into_a_directory_that_does_not_exist_yet(
    study, tmp_path, monkeypatch
):
    """cold review efit-workflows F8: the checkpoint inside the shot loop wrote
    ``--table`` before the only ``mkdir``, which came after the loop, so a whole
    discharge's model matrix ran and then ``FileNotFoundError``."""
    import types

    import numpy as np

    import vaft.code.efit.toolchain as toolchain

    monkeypatch.setattr(toolchain, "resolve_toolchain", lambda: {"efit": "/nowhere/efit"})
    monkeypatch.setattr(toolchain, "toolchain_identities", lambda resolved: {})
    window = types.SimpleNamespace(start=0.300, end=0.301)
    seed = types.SimpleNamespace(
        prepare_shot=lambda shot, product, **kwargs: ({}, np.array([0.300, 0.301]), window, None)
    )
    monkeypatch.setattr(study, "_module", lambda path, name: seed)
    monkeypatch.setattr(study, "_phase_map", lambda constraints, times, cut: ({}, 0.0))
    monkeypatch.setattr(study, "run_model", lambda constraints, **kwargs: {"slices": [], "seconds": 0.0})
    monkeypatch.setattr(
        study, "summarize_run", lambda run: {"produced": 0, "requested": 2, "outcomes": {}, "seconds": 0.0}
    )
    monkeypatch.setattr(study, "summarize_ensemble", lambda models: {})
    monkeypatch.setattr(study, "markdown", lambda payload: "report")

    table = tmp_path / "new" / "tables" / "study.json"
    report = tmp_path / "new" / "reports" / "study.md"
    code = study.main(
        [
            "--output", str(tmp_path / "out"),
            "--shots", "39915",
            "--models", study.BASELINE_MODEL,
            "--packaged-envelope",
            "--table", str(table),
            "--markdown", str(report),
        ]
    )
    assert code == 0
    assert "39915" in json.loads(table.read_text(encoding="utf-8"))["shots"]
    assert report.read_text(encoding="utf-8") == "report"
