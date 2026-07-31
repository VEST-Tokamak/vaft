"""path_exists must detect missing/dead leaves on OMAS ODS inputs.

On a dynamic ODS, reading a missing path returns an EMPTY branch instead of
raising, so the old try/except implementation returned True for every path --
turning every validity guard into a no-op. That let dead b-probe channels
(48xxx campaign probes 65-68, stored with ``field: null``) through the EFIT
constraints filter, crashing input generation with ``float * ODS``.
"""

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.machine_mapping.utils import path_exists


def _probe_ods():
    ods = ODS(consistency_check=False)
    ods["magnetics.time"] = np.linspace(0.26, 0.36, 5)
    ods["magnetics.b_field_pol_probe.0.field.data"] = np.ones(5)
    ods["magnetics.b_field_pol_probe.1.name"] = "dead-probe"   # no field at all
    return ods


def test_existing_leaf_is_true():
    ods = _probe_ods()
    assert path_exists(ods, "magnetics.b_field_pol_probe.0.field.data")
    assert path_exists(ods["magnetics"], "b_field_pol_probe.0.field.data")


def test_missing_leaf_on_ods_is_false():
    """The regression: dynamic access resolves to an empty branch, not an error."""
    ods = _probe_ods()
    assert not path_exists(ods, "magnetics.b_field_pol_probe.1.field.data")
    assert not path_exists(ods["magnetics"], "b_field_pol_probe.1.field.data")
    assert not path_exists(ods, "no.such.tree")


def test_probing_does_not_fabricate_data():
    """After a (False) probe, the dead channel still has no usable field data."""
    ods = _probe_ods()
    assert not path_exists(ods, "magnetics.b_field_pol_probe.1.field.data")
    assert not path_exists(ods, "magnetics.b_field_pol_probe.1.field.data")  # stable


def test_dict_inputs_keep_working():
    d = {"a": {"b": [10, 20]}, "empty": {}}
    assert path_exists(d, "a.b")
    assert path_exists(d, "a.b.1")
    assert not path_exists(d, "a.c")
    assert not path_exists(d, "a.b.5")


def test_scalar_and_string_leaves_exist():
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 48227
    assert path_exists(ods, "dataset_description.data_entry.pulse")
