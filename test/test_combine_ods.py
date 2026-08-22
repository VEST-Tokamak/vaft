import time
import warnings

import numpy as np
import pytest
from omas import ODS

from vaft.omas import combine_ods


def test_combine_ods_merges_every_input_and_later_values_win():
    first = ODS()
    first["dataset_description.data_entry.pulse"] = 39915
    first["magnetics.time"] = np.array([0.0, 0.1])

    second = ODS()
    second["dataset_description.data_entry.pulse"] = 39916
    second["pf_active.time"] = np.array([0.0, 0.1])

    combined = combine_ods([first, second])

    assert combined["dataset_description.data_entry.pulse"] == 39916
    np.testing.assert_array_equal(combined["magnetics.time"], [0.0, 0.1])
    np.testing.assert_array_equal(combined["pf_active.time"], [0.0, 0.1])


def test_combine_ods_skips_invalid_locations_without_mutating_sources():
    contaminated = ODS(consistency_check=False)
    contaminated["dataset_description.data_entry.pulse"] = 39915
    contaminated["pf_passive.R_mat"] = np.eye(2)
    contaminated["pf_passive.L_mat"] = np.ones((2, 1))
    contaminated["pf_passive.M_mat"] = np.eye(2)

    clean = ODS()
    clean["magnetics.time"] = np.array([0.0, 0.1])

    with pytest.warns(RuntimeWarning, match="pf_passive"):
        combined = combine_ods([contaminated, clean])

    assert combined["dataset_description.data_entry.pulse"] == 39915
    np.testing.assert_array_equal(combined["magnetics.time"], [0.0, 0.1])
    assert "pf_passive.R_mat" in contaminated
    assert "pf_passive.L_mat" in contaminated
    assert "pf_passive.M_mat" in contaminated
    assert "pf_passive.R_mat" not in combined
    assert "pf_passive.L_mat" not in combined
    assert "pf_passive.M_mat" not in combined


def test_combine_ods_preserves_code_parameters_metadata_while_pruning_siblings():
    # OMAS exempts `*.code.parameters.*` from structure validation (it is
    # free-form user metadata); the fast invalid-path pre-pass must honor the
    # same exemption or it would wrongly prune legitimate metadata that a real
    # merge accepts.
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    ods["equilibrium.code.parameters.my_custom_field"] = 42
    ods["pf_passive.R_mat"] = np.eye(2)

    with pytest.warns(RuntimeWarning, match="pf_passive.R_mat"):
        combined = combine_ods([ods])

    assert combined["equilibrium.code.parameters.my_custom_field"] == 42
    assert "pf_passive.R_mat" not in combined


def test_combine_ods_prunes_every_invalid_leaf_in_one_merge_attempt():
    # Regression for #82: the transactional retry path used to copy and
    # re-apply the whole merge once per invalid leaf discovered, so a source
    # ODS with N invalid locations cost N+1 full-tree merge attempts. The
    # number of ODS.copy() calls -- one for sanitizing the input, one for the
    # trial merge -- must stay constant regardless of N.
    from unittest.mock import patch

    def contaminated(n_invalid):
        ods = ODS(consistency_check=False)
        ods["dataset_description.data_entry.pulse"] = 39915
        ods["magnetics.ip.0.time"] = np.array([0.0, 0.1])
        ods["magnetics.ip.0.data"] = np.array([0.0, 1000.0])
        for index in range(n_invalid):
            ods[f"pf_passive.bad_field_{index}"] = np.eye(2)
        return ods

    call_counts = {}
    for n_invalid in (1, 16):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with patch.object(ODS, "copy", wraps=ODS.copy, autospec=True) as spy:
                combine_ods([contaminated(n_invalid)])
        call_counts[n_invalid] = spy.call_count

    assert call_counts[1] == call_counts[16]


def _eddy_like_ods(n_loops, n_bad):
    """A synthetic stand-in for the contaminated eddy artifact from #82:
    many valid pf_passive loops plus several invalid top-level matrices
    (R_mat/L_mat/M_mat-shaped)."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    for index in range(n_loops):
        ods[f"pf_passive.loop.{index}.name"] = f"loop{index}"
        ods[f"pf_passive.loop.{index}.element.0.geometry.outline.r"] = np.random.rand(4)
        ods[f"pf_passive.loop.{index}.element.0.geometry.outline.z"] = np.random.rand(4)
    for index in range(n_bad):
        ods[f"pf_passive.bad_field_{index}"] = np.random.rand(n_loops, n_loops)
    return ods


def _time_combine(ods):
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        combine_ods([ods])
    return time.time() - start


def test_combine_ods_time_does_not_scale_with_the_invalid_leaf_count():
    # Regression for #82: on the pre-fix implementation, going from 2 to 40
    # invalid leaves on this payload multiplied the runtime several times
    # over (each extra invalid leaf cost one more full copy/update retry of
    # the growing trial merge). Asserting a bounded ratio rather than an
    # absolute threshold keeps this robust across CI hardware while still
    # catching a reintroduced per-leaf multiplier.
    n_loops = 950
    few = _time_combine(_eddy_like_ods(n_loops, n_bad=2))
    many = _time_combine(_eddy_like_ods(n_loops, n_bad=40))

    assert many < few * 1.4 + 0.15, (
        f"combine_ods scaled with invalid-leaf count: {few:.3f}s at 2 invalid "
        f"leaves vs {many:.3f}s at 40 (ratio {many / max(few, 1e-6):.1f}x)"
    )


def test_combine_ods_stays_within_a_ci_safe_time_budget_for_a_large_contaminated_input():
    # A benchmark-style regression with an explicit wall-clock target for a
    # payload sized like the #82 report (roughly 950 pf_passive loops plus
    # several invalid top-level matrices), with generous headroom for a slow
    # CI runner.
    elapsed = _time_combine(_eddy_like_ods(n_loops=950, n_bad=40))

    assert elapsed < 10.0, f"combine_ods took {elapsed:.2f}s, expected under 10s"


def test_combine_ods_keeps_array_indexed_valid_leaves():
    # Regression for a Cursor Bugbot finding on this fix: the fast pre-pass
    # validates each leaf via OMAS's imas_structure(imas_version, location)
    # with the raw, non-normalized location string (numeric AoS indices like
    # ".0." intact) -- exactly the contract OMAS's own __setitem__ uses,
    # since imas_structure normalizes internally (`ulocation = o2u(location)`)
    # before its cached lookup. Nothing before this test asserted that
    # array-indexed valid leaves actually survive alongside pruned siblings.
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 39915
    for index in range(3):
        ods[f"equilibrium.time_slice.{index}.time"] = float(index) * 0.1
        ods[f"equilibrium.time_slice.{index}.profiles_1d.pressure"] = np.linspace(0, 1, 5)
    for index in range(5):
        ods[f"pf_passive.loop.{index}.name"] = f"loop{index}"
        ods[f"pf_passive.loop.{index}.element.0.geometry.outline.r"] = np.linspace(0, 1, 4)
    ods["pf_passive.R_mat"] = np.eye(2)

    with pytest.warns(RuntimeWarning, match="pf_passive.R_mat"):
        combined = combine_ods([ods])

    for index in range(3):
        assert combined[f"equilibrium.time_slice.{index}.time"] == pytest.approx(index * 0.1)
        np.testing.assert_array_equal(
            combined[f"equilibrium.time_slice.{index}.profiles_1d.pressure"],
            np.linspace(0, 1, 5),
        )
    for index in range(5):
        assert combined[f"pf_passive.loop.{index}.name"] == f"loop{index}"
        np.testing.assert_array_equal(
            combined[f"pf_passive.loop.{index}.element.0.geometry.outline.r"],
            np.linspace(0, 1, 4),
        )
    assert "pf_passive.R_mat" not in combined
