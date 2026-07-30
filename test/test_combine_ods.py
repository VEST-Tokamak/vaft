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
