"""Scientific failure cases found while reviewing the integrated gyrokinetic result API."""

import copy
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from omas import ODS

from vaft.code.gacode.cgyro import CgyroTimeNormalization, read_field_history
from vaft.code.gacode.result_identity import digest
from vaft.machine_mapping.turbulence import core_transport_from_tglf
from vaft.plot.gyrokinetics import plot_gyrokinetics_field_potential
from vaft.process.gyrokinetic_comparison import (
    assess_comparison,
    prepare_diagnostic_traces,
)
from vaft.process.gyrokinetics import reduce_time_series

DATA = Path(__file__).parent / "data/gacode/cgyro_nonlinear_39915_em"


def test_uncertainty_and_stationarity_do_not_change_with_time_units():
    # A small uncovered interval in every block must remain a gap after SI conversion.
    t = np.array(
        [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 2.0]
    )
    y = np.ones_like(t)
    y[[3, 7, 11]] = np.nan
    results = [
        reduce_time_series(t * scale, y, window=(0, 2 * scale), max_gap=0.3 * scale)
        for scale in (1.0, 1e-9, 1e6)
    ]
    assert all(r["standard_error"] is None for r in results)
    assert all(r["qualification"]["stationarity"] == "unverified" for r in results)
    assert (
        results[0]["qualification"]
        == results[1]["qualification"]
        == results[2]["qualification"]
    )
    np.testing.assert_allclose([r["coverage"] for r in results], results[0]["coverage"])


def test_outside_window_burst_cannot_change_selected_window_screen():
    t = np.arange(18.0)
    y = np.r_[np.linspace(1.0, 2.0, 17), 1e14]
    first = reduce_time_series(t[:-1], y[:-1], window=(0, 16))
    extended = reduce_time_series(t, y, window=(0, 16))
    assert extended["mean"] == first["mean"]
    assert extended["relative_drift"] == first["relative_drift"]
    assert extended["qualification"] == first["qualification"]


def test_causal_gap_break_retains_the_first_observed_sample_after_a_gap():
    trace = {"q": {"time": [0, 1, 10, 11], "values": [1, 2, 3, 4], "max_gap": 2}}
    result = prepare_diagnostic_traces(trace, [10, 11])["q"]
    for (t, y), expected in zip(result["frames"], ([1, 2, 3], [1, 2, 3, 4])):
        assert np.isnan(y).sum() == 1
        np.testing.assert_array_equal(y[np.isfinite(y)], expected)
        assert np.all(t[np.isfinite(t)] <= (10 if len(expected) == 3 else 11))
    # The valid segment at 10..11 remains drawable after the missing interval.
    np.testing.assert_array_equal(result["frames"][-1][0][-2:], [10, 11])
    np.testing.assert_array_equal(result["frames"][-1][1][-2:], [3, 4])


@pytest.mark.parametrize("gap", [np.nan, 0, -1])
def test_invalid_diagnostic_gap_refuses_implicit_bridging(gap):
    with pytest.raises(ValueError):
        prepare_diagnostic_traces(
            {"q": {"time": [0, 1, 10], "values": [1, 2, 3], "max_gap": gap}}, [1]
        )


def test_array_evidence_is_recursive_and_missing_metadata_is_unverified():
    evidence = {
        "state": {
            "species": {"z": np.array([1.0, 6.0]), "mass": np.array([1.0, 12.0])}
        },
        "qualification": {"resolution": np.array([32.0, 64.0])},
    }
    equal = assess_comparison(evidence, copy.deepcopy(evidence))
    assert equal.axes["state"]["status"] == "matched"
    assert equal.axes["qualification"]["status"] == "matched"
    missing = {"state": {"species": np.array([1.0, np.nan])}}
    assert assess_comparison(missing, missing).axes["state"]["status"] == "unverified"
    for empty in ({}, [], np.array([]), ""):
        result = assess_comparison(
            {"display": {"normalization": empty}}, {"display": {"normalization": empty}}
        )
        assert result.axes["display"]["status"] == "unverified"


def test_duplicate_realization_in_an_existing_ods_is_rejected_atomically():
    from test_gyrokinetic_result_contracts import PROFILE, pair

    ods = ODS()
    surfaces = [pair(0)]
    core_transport_from_tglf(
        ods, surfaces, PROFILE, time=0.317, state_identity="shot39915"
    )
    ods["core_transport.model.1"] = ods["core_transport.model.0"].copy()
    before = digest(ods.flat())
    with pytest.raises(ValueError, match="ambiguous|duplicate"):
        core_transport_from_tglf(
            ods, surfaces, PROFILE, time=0.317, state_identity="shot39915"
        )
    assert digest(ods.flat()) == before


def test_saved_theta_selection_cannot_silently_truncate_fractional_indices():
    with pytest.raises(ValueError, match="theta"):
        read_field_history(DATA, theta_indices=[0.25])


def test_nonfinite_native_box_is_rejected_at_reader_boundary(tmp_path):
    shutil.copytree(DATA, tmp_path / "run")
    path = tmp_path / "run/out.cgyro.grids"
    values = path.read_text().split()
    values[8] = "nan"
    path.write_text("\n".join(values), encoding="utf-8")
    with pytest.raises(ValueError, match="length|grid"):
        read_field_history(tmp_path / "run")


def test_field_si_override_cannot_contradict_known_native_scales():
    h = read_field_history(DATA)
    norm = CgyroTimeNormalization(
        0.3, 1e-17, 3.34358e-27, np.sqrt(1e-17 / 3.34358e-27), "verified run"
    )
    h = replace(h, metadata={**h.metadata, "time_normalization": norm.metadata()})
    wrong = replace(norm, reference_length_m=0.6)
    with pytest.raises(ValueError, match="contradict|disagree"):
        plot_gyrokinetics_field_potential(h, time_unit="ms", time_normalization=wrong)
