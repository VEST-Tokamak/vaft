"""The diamagnetic weight ladder (#386), and the Thomson cross-check.

Nothing here runs EFIT.  What is pinned is the arithmetic that decides where
the ladder is placed, the common-slice discipline that keeps population change
out of the physics comparison, and the one-sidedness of the Thomson test.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SCAN = REPOSITORY / "workflow" / "efit_diamagnetic_weight" / "diamagnetic_weight_scan.py"
THOMSON = REPOSITORY / "workflow" / "efit_diamagnetic_weight" / "thomson_pressure_check.py"
PROFILE_STUDY = REPOSITORY / "workflow" / "efit_profile_models" / "profile_model_study.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan():
    module = _load(SCAN, "diamagnetic_weight_scan_under_test")
    yield module
    sys.modules.pop("diamagnetic_weight_scan_under_test", None)


@pytest.fixture(scope="module")
def thomson():
    module = _load(THOMSON, "thomson_pressure_check_under_test")
    yield module
    sys.modules.pop("thomson_pressure_check_under_test", None)


@pytest.fixture(scope="module")
def profile_study():
    module = _load(PROFILE_STUDY, "profile_model_study_under_test")
    yield module
    sys.modules.pop("profile_model_study_under_test", None)


# ---------------------------------------------------------------------------
# where the ladder is placed
# ---------------------------------------------------------------------------

def test_the_baseline_rung_is_the_row_weight_vest_actually_submits(scan):
    """``FWTDLC/sigdia = 1e-4``, from the chain in the module docstring.

    ``SIGDLC = 1 * 1e4 * 1e3`` mWb (``kfile.py``), ``sigdia = 1e-3 * SIGDLC``
    (``data_input.F90:2297``), ``fwtdlc /= sigdia**1`` (``:2782``, ``:109``).
    If this number moves, the whole placement of the ladder moves with it.
    """
    baseline = scan.LADDER[0]

    assert baseline.name == scan.BASELINE
    assert baseline.uncertainty_scale == 1.0
    assert baseline.submitted_sigma_mwb == pytest.approx(1.0e7)
    assert baseline.processed_row_weight == pytest.approx(1.0e-4)


def test_the_ladder_brackets_the_stored_measurement_error(scan):
    """The point of the study: #663's ladder stopped four decades short.

    The stored 3 % error on the packaged 39915 slice implies a processed row
    weight of 2.3e4.  ``objective_scales`` x10,000 -- the far end of #663 --
    reaches 1.  This ladder has to pass through both.
    """
    stored_error_weight = 1.0 / (0.03 * 1.448e-3)
    weights = [rung.processed_row_weight for rung in scan.LADDER]

    assert min(weights) <= 1.0e-4
    assert max(weights) > stored_error_weight
    # The rung that reproduces the far end of #663, so the two are comparable.
    reached_by_663 = [
        rung.name for rung in scan.LADDER
        if rung.processed_row_weight == pytest.approx(1.0)
    ]
    assert reached_by_663 == ["sigma_x1e4"]
    decades = math.log10(stored_error_weight / scan.LADDER[0].processed_row_weight)
    assert decades == pytest.approx(8.36, abs=0.01)


def test_a_rung_scales_one_family_and_leaves_the_rest_at_unit(scan, profile_study):
    rung = next(item for item in scan.LADDER if item.uncertainty_scale == 1.0e6)
    scientific = scan.scientific_for(rung, profile_study)
    scales = dict(scientific.constraints.uncertainty_scales)

    assert scales.pop("diamagnetic_flux") == pytest.approx(1.0e6)
    assert set(scales.values()) == {1.0}
    # The objective weights are the axis this study is *not* moving.
    assert set(scientific.constraints.objective_scales.values()) == {1.0}


def test_every_rung_hashes_differently_so_a_stale_cache_cannot_be_reused(
    scan, profile_study
):
    hashes = {
        scan.scientific_for(rung, profile_study).sha256 for rung in scan.LADDER
    }

    assert len(hashes) == len(scan.LADDER)


# ---------------------------------------------------------------------------
# common-slice discipline
# ---------------------------------------------------------------------------

def _row(time_ms, outcome="accepted", **fields):
    return {
        "time_ms": time_ms,
        "outcome": outcome,
        "row_source": "afile+kfile" if outcome != "no_output" else None,
        **fields,
    }


def _geometry(lcfs_mm=0.0, area=0.0, volume=0.0, acceptance=0.0):
    return {
        "geometry": {
            "lcfs_rms_mm": {"median": lcfs_mm},
            "absolute_relative_change": {
                "area": {"median": area},
                "volume": {"median": volume},
            },
        },
        "acceptance_change_percentage_points": acceptance,
    }


def test_losing_solutions_is_not_a_branch_transition(scan):
    """#663's predicate counts a 10 pp acceptance drop as a material response.

    On this ladder acceptance falls because high weights reject slices, so
    reusing that predicate unchanged would report a branch transition every
    time solutions were lost -- which is the confusion ``w_fail`` exists to
    prevent.
    """
    assert not scan.branch_response(_geometry(acceptance=-50.0))
    assert scan.branch_response(_geometry(lcfs_mm=6.0))
    assert scan.branch_response(_geometry(area=0.03))
    assert not scan.branch_response(_geometry(lcfs_mm=4.9, area=0.019))


def test_a_row_that_could_not_be_read_is_not_a_row_that_did_not_respond(
    scan, profile_study
):
    """This build ships ``ENABLE_NETCDF=OFF``, so the m-file path yields nothing.

    Reading the row from the a-file and k-file is what makes the study work
    anyway; the guard is what stops it from reporting "unreachable" when it
    actually measured nothing at all.
    """
    rows = [dict(_row(300), row_source=None)]
    block = {
        "rungs": {
            scan.BASELINE: {
                "summary": scan.summarize_rung(rows, profile_study),
                "diamagnetic": rows,
            }
        }
    }
    for rung in scan.LADDER[1:]:
        block["rungs"][rung.name] = block["rungs"][scan.BASELINE]

    result = scan.classify_reachability(block)

    assert result["w_activation"] is None
    assert "nothing here is evidence" in result["verdict"]


def test_population_change_cannot_masquerade_as_a_smaller_residual(
    scan, profile_study
):
    """#663's own correction, pinned so it cannot be repeated here.

    The high-weight run drops the slice with the large residual instead of
    fitting it.  Aggregated over each run's own population the residual looks
    halved; on the slices both runs produced it has not moved at all.
    """
    baseline = [
        _row(300, residual_relative=0.1, p_axis=30.0),
        _row(301, residual_relative=0.9, p_axis=31.0),
    ]
    candidate = [
        _row(300, residual_relative=0.1, p_axis=30.0),
        _row(301, outcome="no_output"),
    ]

    change = scan.common_slice_change(candidate, baseline, profile_study)

    assert change["common_slices"] == 1
    assert change["relative_change"]["residual_relative"]["median"] == pytest.approx(0.0)


def test_a_real_common_slice_response_is_reported_with_its_sign(
    scan, profile_study
):
    baseline = [_row(300, residual_relative=1.0, p_axis=30.0)]
    candidate = [_row(300, residual_relative=0.2, p_axis=300.0)]

    change = scan.common_slice_change(candidate, baseline, profile_study)

    assert change["relative_change"]["residual_relative"]["median"] == pytest.approx(0.8)
    # The residual fell and the pressure rose: the signs must survive.
    assert change["signed_relative_change"]["residual_relative"]["median"] < 0
    assert change["signed_relative_change"]["p_axis"]["median"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------

def _block(scan, profile_study, responses):
    """A shot block with one entry per rung; ``responses`` keys by rung name."""
    baseline_rows = [_row(300, residual_relative=1.0, p_axis=30.0)]
    rungs = {
        scan.BASELINE: {
            "summary": scan.summarize_rung(baseline_rows, profile_study),
            "diamagnetic": baseline_rows,
        }
    }
    for rung in scan.LADDER[1:]:
        residual, produced, material = responses.get(rung.name, (1.0, True, False))
        rows = (
            [_row(300, residual_relative=residual, p_axis=30.0)]
            if produced
            else [_row(300, outcome="no_output")]
        )
        rungs[rung.name] = {
            "summary": scan.summarize_rung(rows, profile_study),
            "diamagnetic": rows,
            "versus_baseline": scan.common_slice_change(
                rows, baseline_rows, profile_study
            ),
            "branch_response": material,
        }
    return {"rungs": rungs}


def test_a_ladder_that_never_moves_the_residual_says_so_plainly(
    scan, profile_study
):
    """The answer #386 most needs to be able to receive.

    If the residual never responds anywhere on eight decades, the pressure
    deficit is not something weighting can reach, and the study has to say
    that rather than reporting a threshold it did not find.
    """
    result = scan.classify_reachability(_block(scan, profile_study, {}))

    assert result["w_activation"] is None
    assert result["w_branch"] is None
    assert "unreachable by weighting alone" in result["verdict"]


def test_losing_solutions_before_the_residual_responds_is_not_activation(
    scan, profile_study
):
    """Rejecting the slices that fit worst is not fitting them better."""
    responses = {rung.name: (1.0, False, False) for rung in scan.LADDER[5:]}
    result = scan.classify_reachability(_block(scan, profile_study, responses))

    assert result["w_activation"] is None
    assert result["w_fail"] == "sigma_x1e5"
    assert "before the residual ever responded" in result["verdict"]


def test_the_first_rung_that_moves_the_residual_is_the_activation_point(
    scan, profile_study
):
    responses = {rung.name: (0.5, True, False) for rung in scan.LADDER[6:]}
    result = scan.classify_reachability(_block(scan, profile_study, responses))

    assert result["w_activation"] == "sigma_x1e6"
    assert result["w_fail"] is None
    assert "first responded at sigma_x1e6" in result["verdict"]


def test_numerical_noise_is_not_a_response(scan, profile_study):
    """#663 measured invariance at 1e-7 through its whole ladder.

    A floor below that would report every rung as active and the study would
    conclude the opposite of the truth.
    """
    responses = {rung.name: (1.0 - 1.0e-7, True, False) for rung in scan.LADDER[1:]}
    result = scan.classify_reachability(_block(scan, profile_study, responses))

    assert result["w_activation"] is None


# ---------------------------------------------------------------------------
# the Thomson cross-check is one-sided, and must stay one-sided
# ---------------------------------------------------------------------------

def _thomson_rows(*ratios):
    return [
        {
            "path": f"g039915.0031{index}",
            "status": "pass",
            "log_ratio": math.log(ratio),
            "sum_ratio": ratio,
            "points": 5,
        }
        for index, ratio in enumerate(ratios)
    ]


def test_an_electron_pressure_far_above_the_fit_is_decisive(thomson):
    """No unmeasured ion population can make the total *smaller*."""
    summary = thomson.summarize(_thomson_rows(80.0, 90.0, 100.0))

    assert summary["decisive_slices"] == 3
    assert summary["median_sum_ratio"] == pytest.approx(90.0)
    assert "not a data problem" in thomson.verdict(summary)


def test_an_electron_pressure_below_the_fit_decides_nothing(thomson):
    """The ions are unmeasured, so a shortfall is not evidence either way."""
    summary = thomson.summarize(_thomson_rows(0.3, 0.4, 0.5))

    assert summary["decisive_slices"] == 0
    assert "says nothing either way" in thomson.verdict(summary)


def test_slices_with_no_comparable_thomson_sample_are_counted_not_averaged(
    thomson,
):
    """A missing comparison is not agreement."""
    rows = _thomson_rows(90.0) + [
        {
            "path": "g039915.00331",
            "status": "not_available",
            "reason": "the nearest Thomson time is 0.014 s away, beyond 0.001 s",
            "log_ratio": None,
            "sum_ratio": None,
        }
    ]
    summary = thomson.summarize(rows)

    assert summary["reconstructions"] == 2
    assert summary["compared"] == 1
    assert summary["statuses"]["not_available"] == 1


def test_no_comparable_slice_at_all_is_reported_as_silence(thomson):
    summary = thomson.summarize(
        [{"path": "g.1", "status": "not_available", "log_ratio": None}]
    )

    assert summary["compared"] == 0
    assert "says nothing about the pressure" in thomson.verdict(summary)


def test_the_packaged_table_directory_is_not_mistaken_for_a_run(thomson, tmp_path):
    """It ships reference g-files, and the workflow copies it beside the runs.

    Counting it would report the stored 2023 reconstruction as a result of
    this study, at whatever rung happened to sort first.
    """
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "g039915.00319").write_text("reference product", encoding="utf-8")
    run = tmp_path / "shot_39915" / "legacy_sigma"
    (run / "kfile").mkdir(parents=True)
    (run / "g039915.00319").write_text("a real output", encoding="utf-8")

    assert thomson.run_directories(tmp_path, 39915) == [run]


def test_a_ladder_whose_survivors_miss_the_thomson_window_says_so(thomson):
    """High weights need not keep the slices Thomson covers.

    This arm can fall silent exactly where it is most needed, and the report
    has to name the rungs it went silent on rather than quietly comparing
    fewer of them.
    """
    summary_with = thomson.summarize(_thomson_rows(90.0))
    summary_without = thomson.summarize([])

    assert summary_without["compared"] == 0
    assert summary_with["compared"] == 1
